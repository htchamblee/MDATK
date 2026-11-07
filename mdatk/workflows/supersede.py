"""Completed runs move aside before a step's files are written again.

Every file name carries the plan's time stamp, so writing a step again under the same stamp replaces the
scripts and inputs its last run was made with, and running it again replaces that run's outputs.  The
user's rule: as soon as a step whose folder holds a *complete* run of the plan's current stamp
(:func:`mdatk.workflows.runstate.run_status` says ``done``) is written again - even if it is never run
again - that run moves into ``<step folder>/superseded/<stamp>_<YYYYMMDD-HHMMSS>/``, scripts included, so the
outputs keep the scripts that made them and the run stays repeatable when the step is written with changed
settings.  :meth:`mdatk.workflows.plan.Plan.write` does this for every step it writes (unless the project
option ``workflow.supersede_runs`` is off); :func:`pending` lists beforehand what would move.

What moves (:func:`run_files`), only from inside the step's own folder and only of that stamp:

- the step's own outputs, and every file its generator writes for the stamp (scripts, inputs, helpers);
- its logs: the files its type checks for closing lines (``Tool.finish_marks``) and every other file named
  ``<project>_<NN>_*_<stamp>.*`` in the folder (in a plan without step folders only when no other step has
  the number, so another step's files never move);
- for names without a stamp (a custom script's fixed output names, plans without time stamps) the declared
  outputs and their completion markers, which tie them to the run;
- the folders a run fills (``Tool.run_folders``: FARFAR2's and SimRNA's ``runs/``, the extracted
  ``models/``, ``qscore/``, the frame check's moved decoys); without step folders only those named as the
  step's own (``<project>_<NN>_...``), and only when no other step has the step's number.

Relative paths are kept (``models/r01/S_000001.pdb`` stays ``models/r01/S_000001.pdb`` inside the run's
folder) and a manifest (``manifest.toml``) records what moved, from which step and stamp, when and why, and
under ``written_by`` what the step folder's ``MDATK_step.toml`` said then: the step that wrote the files.  That
is the step written again, unless the folder held another step's files (:attr:`Supersede.held`: a plan
renumbered or made again from scratch, a step deleted and another one added with its number); the run is
then that step's, and ``mdatk workflow manifest`` finds it as such.  A partial run stays where it is
(resuming relies on it), a step that uses an earlier run (adopted) is never written, and nothing inside
``superseded/`` is ever replaced: each run gets a folder of its own.  A step's move is all or nothing: when
one file cannot be moved, the files moved before it go back and the step is not written
(:attr:`Supersede.error`).  :mod:`mdatk.workflows.runstate` never looks inside ``superseded/``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import posixpath
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
from .naming import SUPERSEDED, parse_name, sanitize_project, step_folder, step_label
from .runstate import run_status

log = logging.getLogger(__name__)

MANIFEST = "manifest.toml"
MANIFEST_FORMAT = "mdatk-superseded-run"
TIME_FORMAT = "%Y%m%d-%H%M%S"  # the <time> of superseded/<stamp>_<time>/
NO_STAMP = "no-stamp"  # the <stamp> of a run in a plan without time stamps
LOG_STEP = "workflow supersede"  # the 'step' of the project's step log entry
REASON = ("the step's files were written again while its folder held this complete run of the plan's time "
          "stamp; the run was moved aside with the scripts and inputs that made it, so it stays repeatable "
          "(the step may have been written again with changed settings)")


@dataclass
class Supersede:
    """A completed run of one step that moves (or moved) aside before the step is written again."""

    node_id: str
    label: str  # the step's title, e.g. "03 · Amber · MD stages"
    step: int
    tool: str
    stamp: str  # the run's time stamp ("" in a plan without time stamps)
    folder: str  # plan-relative step folder the run's paths are relative to ("": the plan folder)
    store: str  # plan-relative folder of the step's superseded runs
    files: List[str] = field(default_factory=list)  # plan-relative files of the run
    folders: List[str] = field(default_factory=list)  # plan-relative folders of the run (moved whole)
    finished: float = 0.0  # modification time of the run's newest output (seconds since the epoch)
    target: str = ""  # plan-relative folder the run moved into ("": not moved yet)
    moved: List[str] = field(default_factory=list)  # paths that moved (plan-relative, where they were)
    error: str = ""  # why the run could not be moved ("" when it moved, or has not been tried)
    deleted: bool = False  # the run was deleted instead of moved (``moved``: the paths deleted)
    # the step folder holds the files of another step, by its MDATK_step.toml: what they are ("": the step's
    # own, or nothing says); the run is then that step's (mdatk.workflows.manifest.held)
    held: str = ""

    @property
    def paths(self) -> List[str]:
        return self.folders + self.files

    def inner(self, path: str) -> str:
        """``path`` relative to the step folder: its place inside the run's folder in superseded/."""
        return path[len(self.folder) + 1:] if self.folder else path

    def destination(self) -> str:
        """Where the run goes (or went), plan-relative; ``<time>`` stands for the time of writing."""
        return self.target or posixpath.join(self.store, f"{self.stamp or NO_STAMP}_<time>")

    def count(self) -> str:
        """e.g. "14 files and 1 folder"."""
        n, k = len(self.files), len(self.folders)
        text = f"{n} file{'s' if n != 1 else ''}"
        return text + (f" and {k} folder{'s' if k != 1 else ''}" if k else "")

    def summary(self) -> str:
        run = f"run {self.stamp}" if self.stamp else "run without time stamp"
        if self.deleted:
            if self.error:
                return f"{self.label}: its completed {run} could not be deleted completely: {self.error}"
            return f"{self.label}: deleted its completed {run} ({self.count()})"
        if self.error:
            return f"{self.label}: its completed {run} could not be moved aside: {self.error}"
        if self.target:
            return f"{self.label}: moved its completed {run} aside ({self.count()}) to {self.target}/"
        when = (f", finished {_dt.datetime.fromtimestamp(self.finished):%Y-%m-%d %H:%M}"
                if self.finished else "")
        return f"{self.label}: completed {run}{when} ({self.count()}) -> {self.destination()}/"

    def held_line(self) -> str:
        """Whose run it is when the step folder holds another step's files ("" when it does not)."""
        if not self.held:
            return ""
        done = self.deleted or self.target or self.error
        return f"  not this step's run: {self.held}" + ("" if done else "; it would move aside as this step's")

    def lines(self, limit: int = 20) -> List[str]:
        """:meth:`summary`, whose run it is when not the step's own (:meth:`held_line`), and the paths that
        moved (or would move), relative to the step folder."""
        paths = self.moved if (self.target or self.deleted) else self.paths  # moved / deleted, else planned
        out = [self.summary()] + ([self.held_line()] if self.held else [])
        if self.error and not self.target and not self.deleted:
            return out
        shown = [self.inner(p) + ("/" if p in self.folders else "") for p in paths]
        out += [f"  {p}" for p in shown[:limit]]
        if len(shown) > limit:
            out.append(f"  ... and {len(shown) - limit} more" + (f" (see {self.target}/{MANIFEST})"
                                                                 if self.target else ""))
        return out

    def plain(self) -> Dict[str, Any]:
        """For JSON output (``mdatk workflow write``)."""
        out = {"step": self.step, "title": self.label, "stamp": self.stamp, "from": self.folder or ".",
               "to": self.destination(), "files": [self.inner(p) for p in self.files],
               "folders": [self.inner(p) for p in self.folders], "error": self.error, "deleted": self.deleted}
        if self.held:  # not the step's own run (see held)
            out["held"] = self.held
        return out


# ---------------------------------------------------------------------------
# what belongs to a run
# ---------------------------------------------------------------------------


def _norm(path: str) -> str:
    return posixpath.normpath(path.replace(os.sep, "/"))


def _inside(path: str, folder: str) -> bool:
    """``path`` (plan-relative) lies inside the step folder ``folder`` ("" the plan folder), not in
    superseded/ and not the folder itself."""
    if not path or os.path.isabs(path):
        return False
    norm = _norm(path)
    if norm in (".", "..") or norm.startswith("../"):
        return False
    if folder:
        if not norm.startswith(folder + "/"):
            return False
        norm = norm[len(folder) + 1:]
    return bool(norm) and norm.split("/")[0] != SUPERSEDED


def _belongs(name: str, project: str, step: int, stamp: str) -> bool:
    """``name`` is one of the step's files of the run with ``stamp`` by the naming convention:
    ``<project>_<NN>_*_<stamp>.*`` (or ``<project>_<NN>_<tag>.<ext>`` without a stamp)."""
    prefix = f"{project}_{step_label(step)}_"
    if not name.startswith(prefix):
        return False
    if stamp:
        return re.match(re.escape(prefix) + r".+_" + re.escape(stamp) + r"(?:\..*)?$", name) is not None
    parts = parse_name(name)
    return bool(parts and not parts["stamp"] and sanitize_project(parts["project"]) == project
                and int(parts["step"]) == step)


def _exists(path: str) -> bool:
    return os.path.lexists(path)


def run_files(plan, node_id: str, stamp: str) -> Tuple[List[str], List[str]]:
    """(files, folders): plan-relative paths of what the step's run with ``stamp`` left in the step's own
    folder and that exist, in a stable order (see the module docstring)."""
    step = plan.steps.get(node_id)
    if step is None or step.tool is None or not step.tool.generates or step.ctx is None or not plan.root:
        return [], []
    tool, settings = step.tool, step.settings
    folder = step.ctx.folder
    ctx = step.ctx.with_stamp(stamp)
    ctx.preview = True  # names only: no slow work
    # other steps' files, and files chosen by hand that the step passes on (e.g. FARFAR2's own FASTA file)
    passed = {p for v in step.ctx.inputs.values() for p in v.paths}
    passed |= {str(settings.get(key) or "").strip() for key, opt in tool.flat_options().items()
               if opt.kind == "file"}
    root = os.path.abspath(plan.root)
    passed = {_norm(os.path.relpath(p, root) if os.path.isabs(p) and p.startswith(root + os.sep) else p)
              for p in passed if p}
    candidates: List[str] = []
    try:
        for value in tool.output_values(settings, ctx).values():
            candidates += list(value.paths)
    except Exception:  # noqa: BLE001 - a generator bug: the other rules still find the run's files
        log.debug("output names of %s failed", node_id, exc_info=True)
    try:
        candidates += [ctx.from_own(f.name) for f in tool.generate(settings, ctx)]
    except Exception:  # noqa: BLE001 - errors in the settings: the naming rule finds the scripts
        log.debug("file names of %s failed", node_id, exc_info=True)
    try:
        candidates += [path for path, _pattern in tool.finish_marks(settings, ctx)]
    except Exception:  # noqa: BLE001
        log.debug("finish marks of %s failed", node_id, exc_info=True)
    # the other files of the run in the folder, by name; without step folders only when the step number is
    # the step's own (two steps with one number would share <project>_<NN>_ in the plan folder)
    shared = not folder and any(
        s is not step and s.tool is not None and s.tool.generates and s.node.step == step.node.step
        for s in plan.steps.values())
    if not shared:
        try:
            names = sorted(os.listdir(plan.node_path(folder) if folder else plan.root))
        except OSError:
            names = []
        for name in names:
            rel = posixpath.join(folder, name) if folder else name
            full = plan.node_path(rel)
            if _belongs(name, plan.project, step.node.step, stamp) and not os.path.isdir(full):
                candidates.append(rel)
    dirs: List[str] = []
    try:
        dirs = [_norm(d) for d in tool.run_folders(settings, ctx)]
    except Exception:  # noqa: BLE001
        log.debug("run folders of %s failed", node_id, exc_info=True)
    if not folder:  # without step folders only a folder named as the step's own (<project>_<NN>_...), and
        # only when no other step has its number: the steps share runs/ and the plan folder
        own = f"{plan.project}_{step_label(step.node.step)}_"
        dirs = [d for d in dirs if not shared and "/" not in d and d.startswith(own)]
    folders: List[str] = []
    for d in dirs:
        if _inside(d, folder) and os.path.isdir(plan.node_path(d)) and not os.path.islink(plan.node_path(d)) \
                and d not in folders and not any(d.startswith(f + "/") for f in folders):
            folders = [f for f in folders if not f.startswith(d + "/")] + [d]
    files: List[str] = []
    for path in candidates:
        path = _norm(path)
        full = plan.node_path(path)
        if (path in files or path in passed or not _inside(path, folder)
                or any(path.startswith(f + "/") for f in folders)
                or not _exists(full) or (os.path.isdir(full) and not os.path.islink(full))):
            continue
        files.append(path)
    return files, folders


def _store(plan, step) -> str:
    """Plan-relative folder of the step's superseded runs: <step folder>/superseded, or in a plan without
    step folders superseded/<NN>_<tag> (the steps share the plan folder; <NN>_<tag>_<folder tag> for a step
    with a folder tag)."""
    if step.ctx.folder:
        return posixpath.join(step.ctx.folder, SUPERSEDED)
    return posixpath.join(SUPERSEDED, step_folder(step.node.step, step.tool.tag,
                                                  (step.node.folder_tag or "").strip()))


def candidate(plan, node_id: str) -> Optional[Supersede]:
    """The step's complete run of the plan's stamp that writing the step would move aside, or None (no
    such run; a partial one; a step that uses an earlier run or has errors, so it is not written)."""
    step = plan.steps.get(node_id)
    if (step is None or step.tool is None or not step.tool.generates or step.ctx is None or not plan.root
            or step.adopted or not step.ok):
        return None
    status = run_status(plan, node_id, plan.stamp)
    if status.state != "done":
        return None
    files, folders = run_files(plan, node_id, plan.stamp)
    if not files and not folders:
        return None
    return Supersede(node_id, step.label, step.node.step, step.tool.name, plan.stamp, step.ctx.folder,
                     _store(plan, step), files, folders, status.finished, held=step.held)


def pending(plan, node_ids: Optional[Sequence[str]] = None) -> Dict[str, Supersede]:
    """The completed runs that writing the steps ``node_ids`` (all by default) would move aside, in plan
    order; nothing when the plan has no folder or its project option ``supersede_runs`` is off."""
    if not plan.root or not getattr(plan.workflow, "supersede_runs", True):
        return {}
    out: Dict[str, Supersede] = {}
    for nid in plan.order:
        if node_ids is None or nid in node_ids:
            item = candidate(plan, nid)
            if item is not None:
                out[nid] = item
    return out


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------


# what a run's manifest keeps of the step folder's MDATK_step.toml (mdatk.workflows.manifest), as written_by
WRITTEN_BY = {
    "node": "Its id in the plan.",
    "tool": "Its type.",
    "title": "Its own title (\"\": none).",
    "label": "Its title as RNAdo showed it: number, title and folder tag.",
    "step": "Its step number then.",
    "folder_tag": "Its folder tag (\"\": none).",
    "project": "The plan's project name then.",
    "stamp": "The time stamp in the names of the files written (\"\": none).",
    "written": "When its files were written into the step folder.",
    "folder": "The step folder, relative to the plan folder.",
    "mdatk_version": f"The {PROJECT_ABBREV} version that wrote them.",
}


def _manifest(plan, item: Supersede, when: _dt.datetime, identity: Optional[Dict[str, Any]] = None) -> str:
    """The run's ``manifest.toml``; ``identity``: the step folder's MDATK_step.toml, kept as ``written_by``."""
    from .. import tomlio

    data: Dict[str, Any] = {
        "format": MANIFEST_FORMAT, "mdatk_version": __version__, "moved": when.replace(microsecond=0),
        "reason": REASON, "project": plan.project, "step": item.step, "title": item.label,
        "tool": item.tool, "node": item.node_id, "stamp": item.stamp, "from": item.folder or ".",
    }
    if item.finished:
        data["finished"] = _dt.datetime.fromtimestamp(item.finished).replace(microsecond=0)
    data["folders"] = [item.inner(p) for p in item.folders]
    data["files"] = [item.inner(p) for p in item.files]
    if identity:
        data["written_by"] = {k: identity[k] for k in WRITTEN_BY if identity.get(k) is not None}
    docs = {
        "format": "File type marker; do not change.",
        "mdatk_version": f"{PROJECT_ABBREV} version that moved the run.",
        "moved": "When the run was moved here.",
        "reason": "Why.",
        "project": "The plan's project name.",
        "step": "Step number of the planner step whose run this is (unless written_by below names another "
                "step: then that step's files were in its folder).",
        "title": "The step's title.",
        "tool": "The step type.",
        "node": "The step's id in the plan.",
        "stamp": "Time stamp of the run (\"\": a plan without time stamps).",
        "from": "Plan-relative folder the files came from; their paths below are relative to it and to "
                "this folder.",
        "finished": "When the run's newest output was written.",
        "folders": "Folders of the run that moved whole (their names carry no time stamp).",
        "files": "Files of the run: its outputs, the scripts, inputs and helpers written for it, its logs.",
        "written_by": "What the step folder's MDATK_step.toml said when the run moved: the step that wrote the "
                      "files there, whose run this is (another step than the one above when the plan had been "
                      "renumbered or made again from scratch, or a step deleted and another one added with its "
                      "number).",
    }
    header = [f"A completed run of step {step_label(item.step)} ({item.label}) of plan {plan.project}, moved "
              f"aside by {PROJECT_ABBREV} {__version__}",
              f"before the step's files were written again. To use it again, move its files back into "
              f"{item.folder or 'the plan folder'}/."]
    if item.held:
        header += textwrap.wrap(f"Not that step's run: {item.held} (written_by below).", 98)

    def doc(path, _stack):
        if len(path) == 1:
            return docs.get(path[0])
        if len(path) == 2 and path[0] == "written_by":
            return WRITTEN_BY.get(path[1])
        return None

    return tomlio.dumps(data, doc, header)


def _unique_folder(parent: str, name: str) -> str:
    """Create and return ``parent/name`` (``name-2``, ``name-3`` ... when it exists): never reuse one."""
    os.makedirs(parent, exist_ok=True)
    k = 1
    while True:
        path = os.path.join(parent, name if k == 1 else f"{name}-{k}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            k += 1


def _prune(top: str, stop: str) -> None:
    """Remove the empty folders inside ``top``, ``top`` itself and its empty parents below ``stop``."""
    for dirpath, _dirs, _files in sorted(os.walk(top), key=lambda w: -len(w[0])):
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
    parent = os.path.dirname(top)
    while parent != stop and parent.startswith(stop + os.sep):
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def move(plan, item: Supersede, when: Optional[_dt.datetime] = None) -> Supersede:
    """Move ``item``'s run into a new folder ``<store>/<stamp>_<time>`` with a manifest, all or nothing:
    on an error the files moved so far go back, nothing is left in superseded/ and ``item.error`` says why.
    Returns ``item``."""
    from .manifest import read_identity

    when = when or _dt.datetime.now()
    step_dir = plan.node_path(item.folder) if item.folder else plan.root
    # whose files these are: the step folder's MDATK_step.toml stays in the folder and is replaced once the
    # step's files are written, so the run's manifest keeps what it says now
    identity = read_identity(step_dir) if item.folder else None
    done: List[Tuple[str, str, str]] = []  # (from, to, plan-relative path)
    target = ""
    doing = f"create a folder in {item.store}/"
    try:
        target = _unique_folder(plan.node_path(item.store),
                                f"{item.stamp or NO_STAMP}_{when.strftime(TIME_FORMAT)}")
        for rel in item.paths:
            doing = f"move {rel}"
            src = plan.node_path(rel)
            dst = os.path.join(target, *item.inner(rel).split("/"))
            if os.path.lexists(dst):  # never replace anything in superseded/
                raise FileExistsError(f"{dst} already exists")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            done.append((src, dst, rel))
        doing = f"write {MANIFEST}"
        tmp = os.path.join(target, MANIFEST + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(_manifest(plan, item, when, identity))
        os.replace(tmp, os.path.join(target, MANIFEST))
    except OSError as exc:
        item.error = f"could not {doing}: {exc.strerror or exc}"
        stuck = []
        for src, dst, rel in reversed(done):  # put everything back
            try:
                os.makedirs(os.path.dirname(src), exist_ok=True)
                os.rename(dst, src)
            except OSError:
                stuck.append(rel)
        if target:
            for name in (MANIFEST, MANIFEST + ".tmp"):
                try:
                    os.remove(os.path.join(target, name))
                except OSError:
                    pass
        if stuck:  # nothing is lost: the files that could not go back stay in the run's folder
            item.target = os.path.relpath(target, plan.root).replace(os.sep, "/")
            item.moved = stuck[::-1]
            item.error += f"; {len(stuck)} file(s) could not be moved back and stay in {item.target}/"
        elif target:
            _prune(target, step_dir)
        log.warning("%s", item.summary())
        return item
    item.target = os.path.relpath(target, plan.root).replace(os.sep, "/")
    item.moved = [rel for _src, _dst, rel in done]
    return item


def delete(plan, item: Supersede) -> Supersede:
    """Delete ``item``'s run instead of moving it aside: the same files and folders :func:`move` would move
    (folders whole, a link as the link, never what it points to), for trial runs that are not worth the
    disk space.  Cannot be undone; on an error the rest is still tried, ``item.moved`` lists what was
    deleted and ``item.error`` what was not.  Returns ``item``."""
    item.deleted = True
    step_dir = plan.node_path(item.folder) if item.folder else plan.root
    failed: List[str] = []
    for rel in item.paths:
        path = plan.node_path(rel)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            item.moved.append(rel)
        except FileNotFoundError:
            item.moved.append(rel)
        except OSError as exc:
            failed.append(f"{item.inner(rel)} ({exc.strerror or exc})")
    for rel in item.moved:  # folders the run's files were in, emptied by the deletion
        _prune(os.path.dirname(plan.node_path(rel)), step_dir)
    if failed:
        item.error = f"{len(failed)} path(s) could not be deleted: " + ", ".join(failed[:5]) + \
            (" ..." if len(failed) > 5 else "")
        log.warning("%s", item.summary())
    return item


def move_aside(plan, node_id: str, when: Optional[_dt.datetime] = None,
               project_path: Optional[str] = None, discard: bool = False) -> Optional[Supersede]:
    """Move the step's complete run of the plan's stamp aside (:func:`candidate`, :func:`move`), or with
    ``discard`` delete it (:func:`delete`), and record it in the step log of ``project_path``; None when
    there is nothing to move."""
    item = candidate(plan, node_id)
    if item is None:
        return None
    if discard:
        delete(plan, item)
    else:
        move(plan, item, when)
    log_move(project_path, item, plan.root)
    return item


def log_move(project_path: Optional[str], item: Supersede, root: str = "") -> None:
    """Record a move (or a failed one) in the step log of the project file (no-op for an unsaved project or
    a parameter log); a failed log write never fails the write."""
    from ..projectfile import LogEntry, append_log, is_params_log

    if not project_path or not os.path.isfile(project_path) or is_params_log(project_path):
        return
    where = item.target or item.folder
    if item.deleted:
        title = (f"Could not delete all of the completed run of {item.label}" if item.error
                 else f"Deleted the completed run of {item.label}")
    else:
        title = (f"Could not move the completed run of {item.label} aside" if item.error
                 else f"Moved the completed run of {item.label} aside")
    entry = LogEntry(step=LOG_STEP, status="failed" if item.error else "ok", title=title,
                     folder=os.path.join(root, where) if root else where,
                     files=[os.path.join(root, item.target, item.inner(p)) if root else item.inner(p)
                            for p in item.moved] if item.target else [],
                     summary=item.summary() + ("" if item.error or item.deleted else f". Reason: {REASON}.")
                     + (f" Not this step's run: {item.held}." if item.held else ""))
    try:
        append_log(project_path, entry)
    except (OSError, ValueError) as exc:
        log.warning("could not write the step log of %s: %s", project_path, exc)
