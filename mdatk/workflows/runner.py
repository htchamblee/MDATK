"""Running the scripts MDATK generated, when the user asks for it.

Generated scripts are meant to be reviewed and normally run by hand.  On
request, MDATK runs each step's main scripts (role ``run``) with ``bash``, streams
their output to a callback and appends one entry per script to the project file's
step log.  Scripts marked "run separately" (production runs meant for a cluster
queue, long sampling runs) are never started here.

A step starts when every step it reads from (directly or through other steps of the
plan) that is part of the same run has finished; up to ``parallel`` scripts run at
the same time, in plan order otherwise.  A failed script stops every step that
depends on it; independent steps carry on.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from .. import PROJECT_ABBREV
from ..frozen import child_env
from ..progress import Cancelled
from ..projectfile import logged

LineFn = Callable[[str], None]
StateFn = Callable[["ScriptJob", str], None]  # job, queued | running | ok | failed | cancelled | skipped


class ScriptFailed(RuntimeError):
    """A generated script exited with a non-zero status."""


class ScriptCancelled(Cancelled):
    """The run was cancelled; ``output`` holds what the script printed until then."""

    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


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


@dataclass
class ScriptResult:
    job: ScriptJob
    status: str  # ok | failed | cancelled
    returncode: Optional[int]
    duration_s: float
    message: str = ""  # why it failed (exit status, script missing, bash missing)


def plan_jobs(plan, node_ids: Optional[Sequence[str]] = None) -> List[ScriptJob]:
    """The main scripts of the plan's steps, in plan order (written or not)."""
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
        if step.tool is None or not step.tool.generates or not step.ok:
            continue
        if not step.files:
            step.files = generate_step(step)
        for f in step.files:
            if f.role == "run":
                jobs.append(ScriptJob(step.node.step, step.label,
                                      os.path.join(plan.root, step.ctx.from_own(f.name)),
                                      step_citations(step), nid,
                                      [a for a in chosen if a in ancestors(nid)]))
    return jobs


def separate_scripts(plan, node_ids: Optional[Sequence[str]] = None) -> List[str]:
    """Scripts the plan marks 'run separately' (listed, never started by MDATK)."""
    out = []
    for nid in plan.order:
        step = plan.steps[nid]
        if (node_ids is None or nid in node_ids) and step.ctx is not None:
            out += [os.path.join(plan.root, step.ctx.from_own(f.name)) for f in step.files
                    if f.role == "run-separate"]
    return out


def run_script(path: str, on_line: LineFn, cancel: Optional[threading.Event] = None) -> "tuple[int, str]":
    """Run ``bash path`` in the script's folder; returns (exit status, captured output)."""
    problem = runner_problem()
    if problem:
        raise ScriptFailed(problem)
    bash = shutil.which("bash")
    if not os.path.isfile(path):
        raise ScriptFailed(f"{path} does not exist (write the plan's files first)")
    # own process group: cancelling stops its children too; child_env: no bundle settings (frozen build)
    proc = subprocess.Popen([bash, path], cwd=os.path.dirname(path) or ".", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                            start_new_session=True, env=child_env())
    captured: List[str] = []
    stop = threading.Event()

    def watch():
        while not stop.wait(0.2):
            if cancel is not None and cancel.is_set() and proc.poll() is None:
                _stop(proc)
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            captured.append(line)
            on_line(line)
        code = proc.wait()
    except BaseException as exc:  # Ctrl-C (KeyboardInterrupt) or an error in on_line
        _stop(proc)  # the script has its own session, so it never sees the terminal's SIGINT
        if isinstance(exc, KeyboardInterrupt):
            raise ScriptCancelled("script run interrupted", "\n".join(captured)) from None
        raise
    finally:
        stop.set()
        watcher.join(1.0)
    if cancel is not None and cancel.is_set():
        raise ScriptCancelled("script run cancelled", "\n".join(captured))
    return code, "\n".join(captured)


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


def run_jobs(jobs: Sequence[ScriptJob], project_path: Optional[str] = None,
             on_line: LineFn = print, cancel: Optional[threading.Event] = None,
             parallel: int = 1, on_state: Optional[StateFn] = None) -> List[ScriptResult]:
    """Run ``jobs``: each when the jobs it depends on succeeded, at most ``parallel`` at a time,
    otherwise in the given (plan) order.  Raises ScriptFailed when a script failed (after the
    independent ones finished) and ScriptCancelled when the run was cancelled.  An error of
    ``on_line`` itself (e.g. a closed output pipe) stops every script and is re-raised."""
    parallel = max(1, int(parallel))
    stop = cancel if cancel is not None else threading.Event()
    deps = _dependencies(jobs)
    status: Dict[int, str] = {}
    results: List[ScriptResult] = []
    fatal: List[BaseException] = []  # an error of on_line / on_state: ends the whole run
    cond = threading.Condition()

    def state(i: int, value: str) -> None:
        if on_state is not None:
            try:
                on_state(jobs[i], value)
            except Exception:  # a display callback never stops the run
                pass

    def run_one(i: int) -> None:
        job = jobs[i]
        rel = os.path.basename(job.path)
        prefix = f"[{job.step:02d}] " if parallel > 1 else ""
        line_fn = (lambda line: on_line(prefix + line)) if prefix else on_line
        t0 = time.monotonic()
        outcome, code, message = "failed", None, ""
        try:
            with logged(project_path, f"{job.step:02d}", job.title, job.citations) as entry:
                entry.command = f"bash {rel}"
                entry.folder = os.path.dirname(job.path)
                try:
                    code, output = run_script(job.path, line_fn, stop)
                except ScriptCancelled as exc:
                    entry.output = exc.output
                    outcome = "cancelled"
                    raise
                entry.output = output
                entry.summary = f"exit status {code}"
                if code != 0:
                    message = f"exit status {code}"
                    raise ScriptFailed(f"{rel} (step {job.step:02d}) exited with status {code}")
                outcome = "ok"
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
            with cond:  # always leave "running", or the scheduler would wait forever
                status[i] = outcome
                results.append(ScriptResult(job, outcome, code, time.monotonic() - t0, message))
                state(i, outcome)  # under the lock: before the next script of this step reports "running"
                cond.notify_all()

    for i in range(len(jobs)):
        state(i, "queued")
    threads: List[threading.Thread] = []
    try:
        with cond:
            while True:
                running = sum(1 for v in status.values() if v == "running")
                for i in range(len(jobs)):
                    if i in status or stop.is_set():
                        continue
                    bad = [k for k in deps[i] if status.get(k) in ("failed", "cancelled", "skipped")]
                    if bad:
                        status[i] = "skipped"
                        results.append(ScriptResult(jobs[i], "skipped", None, 0.0))
                        state(i, "skipped")
                        on_line(f"==== step {jobs[i].step:02d}: {jobs[i].title} - not started (step "
                                f"{jobs[bad[0]].step:02d} did not succeed)")
                        continue
                    if running < parallel and all(status.get(k) == "ok" for k in deps[i]):
                        status[i] = "running"
                        running += 1
                        state(i, "running")
                        on_line(f"==== step {jobs[i].step:02d}: {jobs[i].title} - bash "
                                f"{os.path.basename(jobs[i].path)}")
                        t = threading.Thread(target=run_one, args=(i,), name=f"mdatk-script-{i}", daemon=True)
                        threads.append(t)
                        t.start()
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
