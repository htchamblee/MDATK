"""Running the scripts MDATK generated, when the user asks for it.

Generated scripts are meant to be reviewed and normally run by hand.  On
request, MDATK runs each step's main scripts (role ``run``) in plan order with
``bash``, streams their output to a callback and appends one entry per script
to the project file's step log.  Scripts marked "run separately" (production
runs meant for a cluster queue, long sampling runs) are never started here.
The first script that fails stops the run.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ..progress import Cancelled
from ..projectfile import logged

LineFn = Callable[[str], None]


class ScriptFailed(RuntimeError):
    """A generated script exited with a non-zero status."""


class ScriptCancelled(Cancelled):
    """The run was cancelled; ``output`` holds what the script printed until then."""

    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


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


@dataclass
class ScriptResult:
    job: ScriptJob
    status: str  # ok | failed | cancelled
    returncode: Optional[int]
    duration_s: float


def plan_jobs(plan, node_ids: Optional[Sequence[str]] = None) -> List[ScriptJob]:
    """The main scripts of the plan's steps, in plan order (written or not)."""
    from ..citations import step_citations
    from .plan import generate_step

    jobs: List[ScriptJob] = []
    for nid in plan.order:
        if node_ids is not None and nid not in node_ids:
            continue
        step = plan.steps[nid]
        if step.tool is None or not step.tool.generates or not step.ok:
            continue
        if not step.files:
            step.files = generate_step(step)
        for f in step.files:
            if f.role == "run":
                jobs.append(ScriptJob(step.node.step, step.label,
                                      os.path.join(plan.root, step.ctx.from_own(f.name)),
                                      step_citations(step)))
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
    bash = shutil.which("bash")
    if not bash:
        raise ScriptFailed("bash was not found on PATH")
    if not os.path.isfile(path):
        raise ScriptFailed(f"{path} does not exist (write the plan's files first)")
    proc = subprocess.Popen([bash, path], cwd=os.path.dirname(path) or ".", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                            start_new_session=True)  # own process group: cancelling stops its children too
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


def run_jobs(jobs: Sequence[ScriptJob], project_path: Optional[str] = None,
             on_line: LineFn = print, cancel: Optional[threading.Event] = None) -> List[ScriptResult]:
    """Run ``jobs`` one after the other; stops at (and raises for) the first failure."""
    results: List[ScriptResult] = []
    for job in jobs:
        rel = os.path.basename(job.path)
        on_line(f"==== step {job.step:02d}: {job.title} - bash {rel}")
        t0 = time.monotonic()
        with logged(project_path, f"{job.step:02d}", job.title, job.citations) as entry:
            entry.command = f"bash {rel}"
            entry.folder = os.path.dirname(job.path)
            try:
                code, output = run_script(job.path, on_line, cancel)
            except ScriptCancelled as exc:
                entry.output = exc.output
                results.append(ScriptResult(job, "cancelled", None, time.monotonic() - t0))
                raise
            entry.output = output
            entry.summary = f"exit status {code}"
            if code != 0:
                results.append(ScriptResult(job, "failed", code, time.monotonic() - t0))
                raise ScriptFailed(f"{rel} (step {job.step:02d}) exited with status {code}")
        results.append(ScriptResult(job, "ok", 0, time.monotonic() - t0))
    return results
