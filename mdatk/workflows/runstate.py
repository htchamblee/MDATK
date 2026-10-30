"""Earlier runs of planner steps: which of a step's output files exist, and under which time stamp.

Every generated file name carries the plan's time stamp (``<project>_<step>_<tag>_<stamp>.<ext>``) and
each iteration of a project may get a new one ("New stamp"), so after an update the plan no longer sees
the files its earlier runs wrote.  :func:`check_runs` looks in each step's folder for the files its output
ports name: under the plan's stamp first, otherwise under the newest other stamp that has all of them.
A complete run can be *adopted* (:func:`adopt`, stored as ``PlanNode.adopted``): the planner then points
the step's outputs at that run's files, so the steps after it read (and check) existing files, and the
step itself is neither written nor run again until the adoption is dropped (:func:`forget`).

Only the step's own files count (inputs passed through to an output belong to another step).  A step type
can also name log files that must end with a program's closing lines (``Tool.finish_marks``), so a run that
was interrupted after writing its first restart file is not mistaken for a finished one; a custom script's
run script writes such a file last, after checking the outputs.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .naming import parse_name, sanitize_project

STATES = ("done", "partial", "missing", "none")
TAIL_BYTES = 16384  # closing lines of a program's log are searched for in its last bytes


@dataclass
class RunStatus:
    """What :func:`check_step` found for one step.

    ``state`` is ``done`` (every output file of the run is there and finished), ``partial`` (some are),
    ``missing`` (none) or ``none`` (the step has no output files of its own to look for)."""

    node_id: str
    state: str
    stamp: str = ""  # time stamp of the run described ("" in a plan without time stamps)
    current: bool = True  # the run has the plan's current stamp
    adopted: bool = False  # the step already uses this run (PlanNode.adopted)
    found: List[str] = field(default_factory=list)  # plan-relative paths that are there
    missing: List[str] = field(default_factory=list)  # paths that are not, or logs without closing lines
    finished: float = 0.0  # newest modification time of the files found (seconds since the epoch)

    @property
    def date(self) -> str:
        return _dt.datetime.fromtimestamp(self.finished).strftime("%Y-%m-%d %H:%M") if self.finished else ""

    def summary(self) -> str:
        """One line, e.g. "done: run 20261015-1432, finished 2026-10-16 03:12"."""
        run = f"run {self.stamp}" if self.stamp else "run without time stamp"
        if not self.current:
            run += " (an earlier stamp)"
        if self.state == "done":
            text = f"done: {run}, finished {self.date}"
        elif self.state == "partial":
            total = len(self.found) + len(self.missing)
            text = f"partial: {run}, {len(self.found)} of {total} output files"
        elif self.state == "missing":
            text = "missing: no output files of this step yet"
        else:
            text = "nothing to look for: the step has no output files of its own"
        return text + (" - in use" if self.adopted else "")


def _step(plan, node_id: str):
    step = plan.steps.get(node_id)
    if step is None or step.tool is None or not step.tool.generates or step.ctx is None:
        return None
    return step


def expected(plan, node_id: str, stamp: str) -> List[str]:
    """Plan-relative paths of the output files the step's run with ``stamp`` writes: its own files only, and
    of those the ones its scripts make (not the input files MDATK writes with them, e.g. a FASTA file)."""
    step = _step(plan, node_id)
    if step is None:
        return []
    ctx = step.ctx.with_stamp(stamp)
    ctx.preview = True  # names only: no slow work
    try:
        values = step.tool.output_values(step.settings, ctx)
    except Exception:  # noqa: BLE001 - a generator bug: nothing to look for (the planner reports it)
        return []
    try:
        written = {ctx.from_own(f.name) for f in step.tool.generate(step.settings, ctx)}
    except Exception:  # noqa: BLE001 - errors in the settings: every output counts
        written = set()
    passed = {p for v in step.ctx.inputs.values() for p in v.paths}
    out: List[str] = []
    for value in values.values():
        for path in value.paths:
            if path not in passed and not os.path.isabs(path) and path not in out:
                out.append(path)
    made = [p for p in out if p not in written]
    return made or out


def run_stamps(plan, node_id: str) -> List[str]:
    """Time stamps found in the names of the step's files in its folder, newest first."""
    step = _step(plan, node_id)
    if step is None or not plan.root:
        return []
    folder = os.path.join(plan.root, step.ctx.folder)
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    stamps = set()
    for name in names:
        parts = parse_name(name)
        if (parts and parts["stamp"] and sanitize_project(parts["project"]) == plan.project
                and int(parts["step"]) == step.node.step):
            stamps.add(parts["stamp"])
    return sorted(stamps, reverse=True)


def _ends_with(path: str, pattern: str) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - TAIL_BYTES))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return False
    return re.search(pattern, tail) is not None


def run_status(plan, node_id: str, stamp: str) -> RunStatus:
    """The state of the step's run with ``stamp``."""
    step = _step(plan, node_id)
    status = RunStatus(node_id, "none", stamp, stamp == plan.stamp,
                       bool(step is not None and stamp and step.adopted == stamp))
    if step is None:
        return status
    paths = expected(plan, node_id, stamp)
    try:
        marks = step.tool.finish_marks(step.settings, step.ctx.with_stamp(stamp))
    except Exception:  # noqa: BLE001 - as above
        marks = []
    if not paths and not marks:
        return status
    unfinished = set()  # logs that are there but lack the closing lines (an interrupted or running program)
    for path, pattern in marks:
        if path not in paths:
            paths.append(path)
        if os.path.isfile(plan.node_path(path)) and not _ends_with(plan.node_path(path), pattern):
            unfinished.add(path)
    for path in paths:
        full = plan.node_path(path)
        if os.path.isfile(full) and os.path.getsize(full) > 0 and path not in unfinished:
            status.found.append(path)
            status.finished = max(status.finished, os.path.getmtime(full))
        else:
            status.missing.append(path)
    status.state = "done" if not status.missing else ("partial" if status.found or unfinished else "missing")
    return status


def check_step(plan, node_id: str) -> Optional[RunStatus]:
    """The step's run in use (adopted), else with the plan's stamp, else the newest other complete one;
    without a complete run, the adopted or current one with what is there.  None for steps that write
    no files (notes, input files, the analysis box) or a plan without a folder."""
    step = _step(plan, node_id)
    if step is None or not plan.root:
        return None
    candidates: List[str] = []
    for stamp in [step.adopted, plan.stamp] + run_stamps(plan, node_id):
        if (stamp or stamp == plan.stamp) and stamp not in candidates:
            candidates.append(stamp)
    seen: List[RunStatus] = []
    for stamp in candidates:
        status = run_status(plan, node_id, stamp)
        if status.state in ("done", "none"):
            return status
        seen.append(status)
    if step.adopted:  # the run in use is incomplete: say so
        return seen[0]
    # nothing complete: the current run, unless it has nothing and an earlier one has something
    return next((s for s in seen if s.found), seen[0])


def check_runs(plan, node_ids: Optional[Sequence[str]] = None) -> Dict[str, RunStatus]:
    """:func:`check_step` for every step that writes files (or those of ``node_ids``), in plan order."""
    out: Dict[str, RunStatus] = {}
    for nid in plan.order:
        if node_ids is None or nid in node_ids:
            status = check_step(plan, nid)
            if status is not None:
                out[nid] = status
    return out


def adopt(plan, runs: Dict[str, RunStatus]) -> List[str]:
    """Let the steps of complete runs in ``runs`` use them (``PlanNode.adopted``); returns the node ids
    changed.  Runs without a time stamp need no adopting (their names never change).  Resolve the plan
    again afterwards."""
    changed = []
    for nid, status in runs.items():
        step = plan.steps.get(nid)
        if step is not None and status.state == "done" and status.stamp and step.node.adopted != status.stamp:
            step.node.adopted = status.stamp
            changed.append(nid)
    return changed


def forget(workflow, node_ids: Optional[Sequence[str]] = None) -> List[str]:
    """Stop using earlier runs: those steps are written and run again.  Returns the node ids changed."""
    changed = []
    for node in workflow.nodes:
        if node.adopted and (node_ids is None or node.id in node_ids):
            node.adopted = ""
            changed.append(node.id)
    return changed
