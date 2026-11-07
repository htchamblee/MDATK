"""Running the scripts MDATK generated, when the user asks for it.

Generated scripts are meant to be reviewed and normally run by hand.  On
request, MDATK runs each step's main scripts (role ``run``) with ``bash``, streams
their output to a callback and appends one entry per script to the project file's
step log.  Scripts marked "run separately" (production runs meant for a cluster
queue, long sampling runs) are started only when asked for (``include_separate``).
The main script of a step that has such a script exits with
:data:`~mdatk.workflows.base.WAITING_STATUS` until that script has finished (or, in the step's
'exit 0' waiting mode, exits 0 leaving ``<script>.waiting`` beside itself): the step
is then *waiting*, which is not a failure, but the steps that depend on it are not
started either.  Steps that use the outputs of an earlier run (``PlanNode.adopted``)
are not run again.

A step starts when every step it reads from (directly or through other steps of the
plan) that is part of the same run has finished and the CPU cores and GPUs it uses
(``Tool.run_resources``) fit in the run's budget (:mod:`mdatk.workflows.budget`), in plan
order otherwise; ``parallel`` caps the number of scripts at once.  A step that uses a GPU
gets free ones in ``CUDA_VISIBLE_DEVICES`` unless it names its own.  A failed script stops
every step that depends on it; independent steps carry on.  One step's script can be stopped
on its own (:class:`StepStops`): its dependents are not started, the other scripts go on, and
the run leaves a note that marks it partial (:func:`mdatk.workflows.runstate.stop_mark`).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Set

from .. import PROJECT_ABBREV
from ..frozen import child_env
from ..progress import Cancelled
from ..projectfile import logged
from .base import LOCAL_VAR, WAITING_STATUS, waiting_file
from .budget import Budget, Grant, Resources, Slots, budget_notes, schedule
from .runstate import stop_mark

LineFn = Callable[[str], None]
# job, queued | running | ok | failed | cancelled | skipped | waiting (for a separately run script)
# | stopped (by the user, on its own)
StateFn = Callable[["ScriptJob", str], None]
# the steps that depend on such a script do not start
STOPPED = ("failed", "cancelled", "skipped", "waiting", "stopped")


class ScriptFailed(RuntimeError):
    """A generated script exited with a non-zero status."""


class ScriptCancelled(Cancelled):
    """The run was cancelled; ``output`` holds what the script printed until then."""

    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


class Flag(Protocol):
    def is_set(self) -> bool: ...


class StepStops:
    """The steps whose scripts to stop during a run (RNAdo's stop buttons); :meth:`stop` may be called from
    any thread.  The other scripts go on; the steps that depend on a stopped one are not started."""

    def __init__(self) -> None:
        self._ids: Set[str] = set()
        self._lock = threading.Lock()

    def stop(self, node_id: str) -> None:
        with self._lock:
            self._ids.add(node_id)

    def __contains__(self, node_id: object) -> bool:
        with self._lock:
            return node_id in self._ids


class _Halt:
    """Set when the whole run is cancelled or the step is stopped (what run_script watches)."""

    def __init__(self, run: threading.Event, stops: Optional[StepStops], node_id: str):
        self.run, self.stops, self.node_id = run, stops, node_id

    def stopped(self) -> bool:
        return self.stops is not None and bool(self.node_id) and self.node_id in self.stops

    def is_set(self) -> bool:
        return self.run.is_set() or self.stopped()


def runner_problem() -> Optional[str]:
    """Why generated scripts cannot be run on this computer, or None when they can.

    The scripts are bash scripts for Linux (a workstation, a cluster or WSL); on Windows
    MDATK writes them but does not start them."""
    if os.name == "nt":
        return (f"The generated scripts are bash scripts for Linux; {PROJECT_ABBREV} on Windows writes them "
                "but cannot run them. Copy the project folder to a Linux machine or cluster, or open it in "
                "WSL (Windows Subsystem for Linux), and run each script there with: bash <script>")
    if shutil.which("bash") is None:
        return "bash was not found on PATH; the generated scripts need bash (Linux, a cluster or WSL)"
    return None


def _stop(proc) -> None:
    """Terminate the script's process group (the script and the programs it started)."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()


@dataclass
class ScriptJob:
    step: int
    title: str
    path: str  # absolute
    citations: List[str] = field(default_factory=list)
    node_id: str = ""  # the plan step ("" = unknown: waits for every job before it)
    after: List[str] = field(default_factory=list)  # steps (node ids) of this run it must wait for
    # the step's scripts marked 'run separately' (absolute): a main script waits for them to finish
    separate: List[str] = field(default_factory=list)
    role: str = "run"  # "run-separate": one of those, run on request after its step's main script
    resources: Resources = Resources()  # what one run of the step uses (Tool.run_resources)
    stop_mark: str = ""  # the note a stopped run leaves (absolute; runstate.stop_mark; "": none)

    def waiting_message(self) -> str:
        names = ", ".join(os.path.basename(p) for p in self.separate) or "a separately run script"
        return f"waiting for a separately run script: {names}"


@dataclass
class ScriptResult:
    job: ScriptJob
    status: str  # ok | failed | cancelled | skipped | waiting | stopped
    returncode: Optional[int]
    duration_s: float
    message: str = ""  # why it failed (exit status, script missing, bash missing) or what it waits for


def plan_jobs(plan, node_ids: Optional[Sequence[str]] = None,
              include_separate: bool = False) -> List[ScriptJob]:
    """The main scripts of the plan's steps, in plan order (written or not); with ``include_separate``
    each step's scripts marked 'run separately' follow its main scripts.  Steps that use the outputs of an
    earlier run (``PlanNode.adopted``) have no jobs: they are not run again."""
    from ..citations import step_citations
    from .plan import generate_step

    parents: Dict[str, Set[str]] = {}
    for e in plan.workflow.edges:
        parents.setdefault(e.target, set()).add(e.source)

    def ancestors(nid: str) -> Set[str]:
        seen: Set[str] = set()
        todo = list(parents.get(nid, ()))
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo += parents.get(n, ())
        return seen

    chosen = [n for n in plan.order if node_ids is None or n in node_ids]
    jobs: List[ScriptJob] = []
    for nid in chosen:
        step = plan.steps[nid]
        if step.tool is None or not step.tool.generates or not step.ok or step.adopted:
            continue
        if not step.files:
            step.files = generate_step(step)
        path = {f.name: os.path.join(plan.root, step.ctx.from_own(f.name)) for f in step.files}
        separate = [path[f.name] for f in step.files if f.role == "run-separate"]
        after = [a for a in chosen if a in ancestors(nid)]
        try:
            needs = step.tool.run_resources(step.settings)
        except Exception:  # noqa: BLE001 - a generator bug: one core (the planner shows the settings' issues)
            needs = Resources()
        mark = os.path.join(plan.root, stop_mark(step.ctx)) if plan.root and step.ctx is not None else ""
        for role in ("run", "run-separate") if include_separate else ("run",):
            for f in step.files:
                if f.role == role:
                    jobs.append(ScriptJob(step.node.step, step.label, path[f.name], step_citations(step), nid,
                                          after, separate, role, needs, mark))
    return jobs


def separate_scripts(plan, node_ids: Optional[Sequence[str]] = None) -> List[str]:
    """Scripts the plan marks 'run separately' (listed; started only with ``include_separate``)."""
    out = []
    for nid in plan.order:
        step = plan.steps[nid]
        if (node_ids is None or nid in node_ids) and step.ctx is not None and not step.adopted:
            out += [os.path.join(plan.root, step.ctx.from_own(f.name)) for f in step.files
                    if f.role == "run-separate"]
    return out


def adopted_steps(plan, node_ids: Optional[Sequence[str]] = None) -> List[str]:
    """Steps (node ids, plan order) that use the outputs of an earlier run and are not run again."""
    return [nid for nid in plan.order if (node_ids is None or nid in node_ids) and plan.steps[nid].adopted]


def conda_on_path(environ: Optional[Dict[str, str]], script: str,
                  on_line: Optional[LineFn] = None) -> Optional[Dict[str, str]]:
    """``environ`` with conda on its PATH when the script at ``script`` uses conda (``conda activate``, as
    the steps' *Conda environment* writes) and conda is not on it.

    ``conda init`` puts conda on PATH in ~/.bashrc, i.e. for interactive shells only: a script run by RNAdo
    started from the applications menu (or any non-interactive bash) does not have it and fails with
    "conda: command not found".  conda is found as MDATK finds it for cpptraj (``$CONDA_EXE``, the usual
    install folders, ~/.conda/environments.txt); its ``condabin`` folder (only the conda command) is put
    first on PATH, else the folder of ``bin/conda``.  Unchanged (None stays None) when the script does not
    use conda, conda is already on PATH, or no conda is found (then ``on_line`` gets a hint)."""
    try:
        with open(script, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return environ
    if not re.search(r"(?m)^[^#\n]*\bconda\s+(?:activate|shell\.bash|run)\b", text):
        return environ
    env = dict(os.environ if environ is None else environ)
    path = env.get("PATH") or os.defpath
    if shutil.which("conda", path=path):
        return environ
    from ..cleaning.conda import _registered_prefixes, candidate_roots

    found = ""
    exe = env.get("CONDA_EXE", "")
    registered = []  # the installations of the environments conda lists in ~/.conda/environments.txt
    for prefix in _registered_prefixes():
        parent = os.path.dirname(prefix)
        registered.append(os.path.dirname(parent) if os.path.basename(parent) == "envs" else prefix)
    roots = ([os.path.dirname(os.path.dirname(exe))] if exe else []) + candidate_roots() + registered
    for root in roots:
        for rel in ("condabin/conda", "bin/conda"):
            cand = os.path.join(root, rel)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                found = cand
                break
        if found:
            break
    if not found:
        if on_line is not None:
            on_line("WARNING: the script uses conda, but conda is not on PATH here (conda init sets it up "
                    "for interactive shells only) and was not found in the usual places; set CONDA_EXE to "
                    "the conda program, or start RNAdo from a terminal where conda works")
        return environ
    env["PATH"] = os.path.dirname(found) + os.pathsep + path
    if on_line is not None:
        on_line("conda is not on PATH here (conda init sets it up for interactive shells only): "
                f"using {found}")
    return env


STEP_LOG_EXT = ".log"  # <script>.sh.log beside the script: the output of its last run
DONE_FLAG = "Step completed successfully"


def step_log_path(script: str) -> str:
    """The output log of a run of ``script``: ``<script>.log`` beside it (``…_amber-md_<stamp>.sh.log``),
    so it carries the run's name and stamp and moves aside with the run (mdatk.workflows.supersede)."""
    return script + STEP_LOG_EXT


class StepLog:
    """The per-step output log of one script run: a header with the start time, every line the script
    prints (standard output and error) as it comes, and a closing line with the end time and the outcome
    (:data:`DONE_FLAG` when the step completed).  Never fails a run: a log that cannot be written is
    skipped with a note in the output."""

    def __init__(self, script: str, command: str, on_line: Optional[LineFn] = None):
        self.path = step_log_path(script)
        self._lock = threading.Lock()
        self._fh = None
        try:
            self._fh = open(self.path, "w", encoding="utf-8", errors="replace", newline="\n")
            self._fh.write(f"==== {os.path.basename(script)} - started {_now()} by {PROJECT_ABBREV} "
                           f"({command}, in {os.path.dirname(script) or '.'})\n")
            self._fh.flush()
        except OSError as exc:
            self._fh = None
            if on_line is not None:
                on_line(f"WARNING: could not write the step's output log {self.path}: {exc.strerror or exc}")

    def write(self, line: str) -> None:
        if self._fh is None:
            return
        with self._lock:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()  # readable while the script runs (tail -f)
            except (OSError, ValueError):
                pass

    def close(self, outcome: str, code: Optional[int], message: str = "", waits_next: bool = False) -> None:
        if self._fh is None:
            return
        when = _now()
        status = f" (exit status {code})" if code is not None else ""
        text = {"ok": f"{DONE_FLAG} - {when}{status}",
                "failed": f"Step failed - {when}{status}" + (f": {message}" if message else ""),
                "stopped": f"Step stopped by the user - {when}",
                "waiting": f"Step waiting - {when}{status}" + (f": {message}" if message else ""),
                "cancelled": f"Run cancelled - {when}"}.get(outcome, f"Step ended ({outcome}) - {when}")
        if outcome == "ok" and waits_next:
            text = f"Main script finished - {when}{status}; its separately run script runs next"
        self.write(f"==== {text}")
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_script(path: str, on_line: LineFn, cancel: Optional[Flag] = None,
               on_stderr: Optional[LineFn] = None, env: Optional[Dict[str, str]] = None) -> "tuple[int, str]":
    """Run ``bash path`` in the script's folder; returns (exit status, captured output).

    The script's standard error goes to ``on_line`` with its output, or to ``on_stderr`` when given
    (RNAdo highlights it); the captured output holds both streams in the order their lines arrived.
    ``env``: variables set for the script on top of MDATK's environment (e.g. CUDA_VISIBLE_DEVICES)."""
    problem = runner_problem()
    if problem:
        raise ScriptFailed(problem)
    bash = shutil.which("bash")
    if not os.path.isfile(path):
        raise ScriptFailed(f"{path} does not exist (write the plan's files first)")
    environ = child_env()
    # run here by MDATK: a separately run script activates its conda environment (base.LOCAL_VAR)
    environ = {**(os.environ if environ is None else environ), LOCAL_VAR: "1", **(env or {})}
    environ = conda_on_path(environ, path, on_line)
    # own process group: cancelling stops its children too; child_env: no bundle settings (frozen build);
    # close_fds (the default, stated): a script started from the AppImage must not inherit its keepalive
    # pipe or its fd 1023 on the mount, which would keep the AppImage mounted for as long as the script (or
    # a program it leaves running) lives; see mdatk.gui.launch.launch_rnado
    proc = subprocess.Popen([bash, path], cwd=os.path.dirname(path) or ".", stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE if on_stderr is not None else subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1, start_new_session=True, env=environ,
                            close_fds=True)
    captured: List[str] = []
    lock = threading.Lock()  # the two streams' readers both capture
    broken: List[BaseException] = []  # an error in on_stderr: re-raised like one of on_line
    stop = threading.Event()

    def watch():
        while not stop.wait(0.2):
            if cancel is not None and cancel.is_set() and proc.poll() is None:
                _stop(proc)
                return

    def read_stderr():
        try:
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip("\n")
                with lock:
                    captured.append(line)
                on_stderr(line)
        except BaseException as exc:  # noqa: BLE001 - stop the script; the caller re-raises it
            broken.append(exc)
            _stop(proc)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    err_reader = None
    if on_stderr is not None:
        err_reader = threading.Thread(target=read_stderr, name="mdatk-script-stderr", daemon=True)
        err_reader.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            with lock:
                captured.append(line)
            on_line(line)
        code = proc.wait()
        if err_reader is not None:
            err_reader.join()
        if broken:
            raise broken[0]
    except BaseException as exc:  # Ctrl-C (KeyboardInterrupt) or an error in on_line / on_stderr
        _stop(proc)  # the script has its own session, so it never sees the terminal's SIGINT
        if err_reader is not None:
            err_reader.join(1.0)
        if isinstance(exc, KeyboardInterrupt):
            raise ScriptCancelled("script run interrupted", "\n".join(captured)) from None
        raise
    finally:
        stop.set()
        watcher.join(1.0)
    if cancel is not None and cancel.is_set():
        raise ScriptCancelled("script run cancelled", "\n".join(captured))
    return code, "\n".join(captured)


def _script_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _flags_waiting(text: str, path: str) -> bool:
    """Whether the main script at ``path`` (its text) waits in the 'exit 0' mode: it leaves its
    :func:`~mdatk.workflows.base.waiting_file` instead of exiting with WAITING_STATUS."""
    return os.path.basename(waiting_file(path)) in text


def _checks_separate(path: str) -> bool:
    """Whether a step's main script checks that its separately run scripts have finished (see
    base.separate_check); older ones exit 0 having run only the part before production."""
    text = _script_text(path)
    return f"exit {WAITING_STATUS}" in text or _flags_waiting(text, path)


def _left_waiting(path: str) -> bool:
    """A main script in the 'exit 0' waiting mode that exited 0 left its waiting file: it waits."""
    return os.path.exists(waiting_file(path)) and _flags_waiting(_script_text(path), path)


def _dependencies(jobs: Sequence[ScriptJob]) -> List[Set[int]]:
    """Job index -> indices of the jobs it waits for."""
    deps: List[Set[int]] = []
    for i, job in enumerate(jobs):
        if not job.node_id:
            deps.append(set(range(i)))  # unknown step: strictly after everything before it
            continue
        deps.append({k for k in range(i) if jobs[k].node_id in job.after
                     or jobs[k].node_id == job.node_id or not jobs[k].node_id})
    return deps


def grant_text(grant: Grant) -> str:
    """What a starting script got: '8 cores, GPU 0', '1 core, GPU 1 (named by the step)'."""
    text = f"{grant.cores} core{'s' * (grant.cores != 1)}"
    if grant.gpus:
        text += f", GPU {','.join(grant.gpus)}" + ("" if grant.assigned else " (named by the step)")
    elif grant.uses_gpu:
        text += ", a GPU it finds itself (no GPU known: one such step at a time)"
    return text + (", alone (larger than the budget)" if grant.alone else "")


def _drop_stop_mark(job: ScriptJob) -> None:
    """The step runs again: the note of an earlier stopped run no longer applies."""
    if job.role == "run" and job.stop_mark:
        try:
            os.remove(job.stop_mark)
        except OSError:
            pass


def _leave_stop_mark(job: ScriptJob, line_fn: LineFn) -> None:
    """A note beside the step's files that its run was stopped: 'Check previous runs' counts it partial."""
    if not job.stop_mark:
        return
    try:
        os.makedirs(os.path.dirname(job.stop_mark) or ".", exist_ok=True)
        with open(job.stop_mark, "w", encoding="utf-8") as fh:
            fh.write(f"{PROJECT_ABBREV}: step {job.step:02d} ({job.title}) was stopped by the user on "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')} while {os.path.basename(job.path)} ran, so its "
                     "output files may be incomplete. The run counts as partial until the step runs again.\n")
    except OSError as exc:
        line_fn(f"WARNING: could not write {job.stop_mark}: {exc}")


def run_jobs(jobs: Sequence[ScriptJob], project_path: Optional[str] = None,
             on_line: LineFn = print, cancel: Optional[threading.Event] = None,
             parallel: int = 1, on_state: Optional[StateFn] = None,
             on_stderr: Optional[LineFn] = None, step_marks: Optional[bool] = None,
             budget: Optional[Budget] = None, stops: Optional[StepStops] = None) -> List[ScriptResult]:
    """Run ``jobs``: each when the jobs it depends on succeeded, at most ``parallel`` at a time (0: no
    limit) and, with a ``budget``, as many as fit in its CPU cores and GPUs (each job's ``resources``; see
    :func:`mdatk.workflows.budget.schedule`), otherwise in the given (plan) order.  A job that uses a GPU
    gets free ones in ``CUDA_VISIBLE_DEVICES`` unless its step names its own.  Without a budget only
    ``parallel`` limits and GPUs are not handed out.  Raises ScriptFailed when a script failed (after the
    independent ones finished) and ScriptCancelled when the run was cancelled.  An error of ``on_line``
    itself (e.g. a closed output pipe) stops every script and is re-raised.

    Each line a script prints starts with its step number (``[03] ``) when ``step_marks`` is true
    (default: when several scripts may run at once); ``on_stderr`` receives the scripts' standard error
    separately (see :func:`run_script`).

    A step in ``stops`` (:meth:`StepStops.stop`, during the run) has its script stopped (process group
    terminated) or, not started yet, never started: result status "stopped", its dependents not started
    ("skipped"), the other jobs go on, and its run leaves the note ``job.stop_mark``.

    A main script that exits with WAITING_STATUS, or exits 0 leaving its ``<script>.waiting`` (the
    step's 'exit 0' waiting mode), leaves its step *waiting* (result status "waiting", not an
    exception) and its dependents not started - unless the step's separate scripts are among
    ``jobs``: they then run next.  A separate script whose main script found
    it finished (exit status 0) is not run again."""
    limit = max(0, int(parallel))
    marks = (limit != 1 and len(jobs) > 1) if step_marks is None else step_marks
    stop = cancel if cancel is not None else threading.Event()
    deps = _dependencies(jobs)
    status: Dict[int, str] = {}
    results: List[ScriptResult] = []
    fatal: List[BaseException] = []  # an error of on_line / on_state: ends the whole run
    cond = threading.Condition()
    slots = Slots(budget, limit)
    with_main = {j.node_id for j in jobs if j.role == "run" and j.node_id}
    separate_next = {j.node_id for j in jobs if j.role == "run-separate" and j.node_id}
    waited: Set[str] = set()  # steps whose main script waits for their separate scripts (they run next)

    def state(i: int, value: str) -> None:
        if on_state is not None:
            try:
                on_state(jobs[i], value)
            except Exception:  # a display callback never stops the run
                pass

    def run_one(i: int, grant: Grant) -> None:
        job = jobs[i]
        rel = os.path.basename(job.path)
        prefix = f"[{job.step:02d}] " if marks else ""
        line_fn = (lambda line: on_line(prefix + line)) if prefix else on_line
        err_fn = None
        if on_stderr is not None:
            err_fn = (lambda line: on_stderr(prefix + line)) if prefix else on_stderr
        halt = _Halt(stop, stops, job.node_id)
        env = {"CUDA_VISIBLE_DEVICES": ",".join(grant.gpus)} if grant.assigned else None
        command = (f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} " if env else "") + f"bash {rel}"
        t0 = time.monotonic()
        outcome, code, message = "failed", None, ""
        steplog: Optional[StepLog] = None
        try:
            if job.role == "run-separate" and job.node_id in with_main and job.node_id not in waited:
                line_fn(f"{rel} has already finished (its step's main script says so) - not run again")
                outcome = "ok"
                return
            _drop_stop_mark(job)
            steplog = StepLog(job.path, command, line_fn)
            shown, shown_err = line_fn, err_fn

            def line_fn(line: str, _show=shown) -> None:  # noqa: F811 - the output log gets every line too
                steplog.write(line)
                _show(line)

            if shown_err is not None:
                def err_fn(line: str, _show=shown_err) -> None:  # noqa: F811
                    steplog.write(line)
                    _show(line)
            with logged(project_path, f"{job.step:02d}", job.title, job.citations) as entry:
                entry.command = command
                entry.folder = os.path.dirname(job.path)
                try:
                    code, output = run_script(job.path, line_fn, halt, on_stderr=err_fn, env=env)
                except ScriptCancelled as exc:
                    entry.output = exc.output
                    if stop.is_set() or not halt.stopped():
                        outcome = "cancelled"
                        raise
                    # stopped on its own: logged as such, not an error; the other scripts go on
                    outcome, message = "stopped", "stopped by the user"
                    entry.status, entry.summary = "stopped", message
                else:
                    entry.output = output
                    entry.summary = f"exit status {code}"
                    main = job.role == "run" and bool(job.separate)
                    # the 'exit 0' waiting mode: the script left <script>.waiting
                    flagged = main and code == 0 and _left_waiting(job.path)
                    # a main script written by MDATK 0.0.8 or older "succeeds" whether or not production ran
                    legacy = main and code == 0 and not flagged and not _checks_separate(job.path)
                    if main and (code == WAITING_STATUS or flagged or legacy):
                        if job.node_id in separate_next:  # asked for: its separate scripts run next
                            waited.add(job.node_id)
                            entry.summary += " (its separately run scripts run next)"
                            outcome = "ok"
                        else:  # not a failure, but the step is not complete
                            message = job.waiting_message() + (
                                f" ({rel} was written by an older {PROJECT_ABBREV} and cannot tell whether "
                                "that script has finished: write the step again)" if legacy else "")
                            entry.status, entry.summary = "waiting", f"exit status {code}: {message}"
                            line_fn(f"==== step {job.step:02d}: {job.title} - {message}")
                            outcome = "waiting"
                    elif code != 0:
                        message = f"exit status {code}"
                        raise ScriptFailed(f"{rel} (step {job.step:02d}) exited with status {code}")
                    else:
                        outcome = "ok"
            if outcome == "stopped":
                _leave_stop_mark(job, line_fn)
        except ScriptCancelled:
            stop.set()  # interrupted inside the script's thread (Ctrl-C in on_line): cancel the whole run
        except ScriptFailed as exc:  # non-zero exit, or the script / bash missing
            message = message or str(exc)
            try:
                line_fn(f"ERROR: {exc}")
            except BaseException as err:  # noqa: BLE001 - the output itself is broken
                fatal.append(err)
                stop.set()
        except BaseException as exc:  # noqa: BLE001 - on_line raised (run_script already stopped the script)
            fatal.append(exc)
            stop.set()
        finally:
            if steplog is not None:
                steplog.close(outcome, code, message,
                              waits_next=job.role == "run" and job.node_id in waited)
            with cond:  # always leave "running", or the scheduler would wait forever
                status[i] = outcome
                slots.give_back(grant)
                results.append(ScriptResult(job, outcome, code, time.monotonic() - t0, message))
                state(i, outcome)  # under the lock: before the next script of this step reports "running"
                cond.notify_all()

    for i in range(len(jobs)):
        state(i, "queued")
    threads: List[threading.Thread] = []
    try:
        for note in budget_notes([(f"step {j.step:02d} ({j.title})", j.resources) for j in jobs], budget):
            on_line(note)
        with cond:
            while True:
                ready = []
                for i in range(len(jobs)):
                    if i in status or stop.is_set():
                        continue
                    bad = [k for k in deps[i] if status.get(k) in STOPPED]
                    if bad:
                        status[i] = "skipped"
                        results.append(ScriptResult(jobs[i], "skipped", None, 0.0))
                        state(i, "skipped")
                        why = {"waiting": "waits for a separately run script",
                               "stopped": "was stopped"}.get(status[bad[0]], "did not succeed")
                        on_line(f"==== step {jobs[i].step:02d}: {jobs[i].title} - not started (step "
                                f"{jobs[bad[0]].step:02d} {why})")
                        continue
                    if stops is not None and jobs[i].node_id and jobs[i].node_id in stops:
                        status[i] = "stopped"  # stopped before it started
                        results.append(ScriptResult(jobs[i], "stopped", None, 0.0,
                                                    "stopped before it started"))
                        state(i, "stopped")
                        continue
                    if all(status.get(k) == "ok" for k in deps[i]):
                        ready.append((i, jobs[i].resources))
                for i, grant in schedule(ready, slots):
                    status[i] = "running"
                    state(i, "running")
                    got = f" ({grant_text(grant)})" if budget is not None else ""
                    on_line(f"==== step {jobs[i].step:02d}: {jobs[i].title} - bash "
                            f"{os.path.basename(jobs[i].path)}{got}")
                    t = threading.Thread(target=run_one, args=(i, grant), name=f"mdatk-script-{i}",
                                         daemon=True)
                    threads.append(t)
                    t.start()
                running = sum(1 for v in status.values() if v == "running")
                if running == 0 and (stop.is_set() or len(status) == len(jobs)):
                    break
                cond.wait(0.2)
    except BaseException as exc:  # Ctrl-C, or on_line failed here: stop every running script first
        stop.set()
        for t in threads:
            t.join()
        if isinstance(exc, KeyboardInterrupt):
            raise ScriptCancelled("script run interrupted") from None
        raise
    for t in threads:
        t.join()
    if fatal:
        raise fatal[0]
    if stop.is_set():
        for i in range(len(jobs)):
            if i not in status:
                state(i, "cancelled")
        raise ScriptCancelled("script run cancelled")
    failed = [r for r in results if r.status == "failed"]
    if failed:
        skipped = sum(1 for r in results if r.status == "skipped")
        names = "; ".join(f"{os.path.basename(r.job.path)} (step {r.job.step:02d}) failed: {r.message}"
                          for r in failed)
        raise ScriptFailed(names + (f"; {skipped} dependent script(s) not started" if skipped else ""))
    return results
