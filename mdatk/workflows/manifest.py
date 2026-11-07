"""What a plan folder holds: its MANIFEST and the identity file of each step folder.

Step numbers are stored with the steps (``PlanNode.step``) and MDATK never renumbers them: a new step gets
the next free number and a template's steps are numbered after the existing ones.  A plan written by a
script of one's own, or edited by hand, can still be renumbered, and then a folder such as ``17_amber-md``
that holds one step's completed run is named for another step.  So every write
(:meth:`mdatk.workflows.plan.Plan.write`) records what the folders are:

``<plan folder>/MANIFEST``
    one line per step of the plan, tab-separated under a header line: step number, folder, node id, step
    type, title, folder tag and input files (``port=path``, ``;`` between them; plan-relative or absolute).
    Rewritten from the plan as it is at every write, also when only some steps are written (the
    stand-alone wizard's one-step plan is not the folder's plan: it leaves MANIFEST alone).  Steps that
    write no files (existing files, the analysis box) have ``-`` for number and folder; notes are left out.
``<step folder>/MDATK_step.toml``
    in every step folder that a write puts files into: the step's node id, type, title, folder tag, number
    and folder, and when.  It is not part of the step's runs (it stays when a run moves aside into
    ``superseded/``) and is replaced only when the folder is written again, so a tree written before a
    renumbering still says what each folder was.

:func:`match` lists a plan's steps against the step folders of an existing tree (``mdatk workflow
manifest``): a folder belongs to the step whose node id its identity file names (with the same type), else
to the one step with its type and folder tag, else to the one step with its type and title; a folder
without an identity file, by the folder tag in its name.  A step whose folder holds another step's files is
a conflict: writing it would put its files beside that step's.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
from .naming import SUPERSEDED, parse_folder, sanitize_tag, step_label
from .plan import _write_text, node_folder, node_folder_tag, node_title  # plan.py imports this module lazily

MANIFEST = "MANIFEST"
IDENTITY = f"{PROJECT_ABBREV}_step.toml"
IDENTITY_FORMAT = "mdatk-step-folder"
COLUMNS = ("step", "folder", "node", "tool", "title", "folder_tag", "inputs")
EMPTY = "-"  # an empty cell of MANIFEST
_CELL_BAD = re.compile(r"[\t\r\n]+")  # would break the table

_IDENTITY_DOCS = {
    "format": "File type marker; do not change.",
    "mdatk_version": f"{PROJECT_ABBREV} version that wrote the step's files into this folder.",
    "written": "When they were written.",
    "project": "The plan's project name (the <project> part of the file names).",
    "step": "The step's number when it was written (the <step> part of the folder and file names).",
    "folder": "This folder, relative to the plan folder.",
    "node": "The step's id in the plan; it stays when steps are added or renumbered.",
    "tool": "The step type.",
    "title": "The step's own title (\"\": none; its box shows the type's label).",
    "label": "The title on the step's box.",
    "folder_tag": "The step's folder tag (\"\": none).",
    "stamp": "Time stamp in the names of the files written (\"\": a plan without time stamps).",
}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _steps(plan) -> List[str]:
    """The plan's steps in plan order (steps in a loop of connections last), notes left out."""
    ids = list(plan.order) + [n for n in plan.steps if n not in plan.order]
    return [n for n in ids if plan.steps[n].node.tool != "note"]


def _writes(step) -> bool:
    return step.tool is not None and step.tool.generates


def _inputs(step) -> str:
    """The step's input files; for an existing file (no inputs, writes nothing) the file it stands for."""
    inputs = step.ctx.inputs if step.ctx is not None else {}
    pairs = [(port, p) for port, v in inputs.items() for p in v.paths]
    if not pairs and not _writes(step):
        pairs = [(port, p) for port, v in step.outputs.items() for p in v.paths]
    return "; ".join(f"{port}={path}" for port, path in pairs)


def rows(plan) -> List[Tuple[str, str, str, str, str, str, str]]:
    """The lines of the plan's MANIFEST (the columns of :data:`COLUMNS`; "" for an empty cell)."""
    out = []
    for nid in _steps(plan):
        step = plan.steps[nid]
        node = step.node
        writes = _writes(step)
        folder = (node_folder(plan.workflow, node) or ".") if writes else ""
        out.append((step_label(node.step) if writes else "", folder, node.id, node.tool, node_title(node),
                    node_folder_tag(node), _inputs(step)))
    return out


def _cell(value: Any) -> str:
    return _CELL_BAD.sub(" ", str(value)).strip() or EMPTY


def manifest_text(plan) -> str:
    """The plan folder's MANIFEST: a header line, then one tab-separated line per step."""
    lines = ["\t".join(COLUMNS)] + ["\t".join(_cell(v) for v in row) for row in rows(plan)]
    return "\n".join(lines) + "\n"


def identity_text(plan, node_id: str, when: Optional[_dt.datetime] = None) -> str:
    """The ``MDATK_step.toml`` of the step's folder."""
    from .. import tomlio

    step = plan.steps[node_id]
    node = step.node
    when = (when or _dt.datetime.now()).replace(microsecond=0)
    data = {"format": IDENTITY_FORMAT, "mdatk_version": __version__, "written": when, "project": plan.project,
            "step": node.step, "folder": step.ctx.folder, "node": node.id, "tool": node.tool,
            "title": node.title, "label": step.label, "folder_tag": node_folder_tag(node),
            "stamp": plan.stamp}
    header = [f"This folder holds the files of step {step.label} of plan {plan.project}, written by "
              f"{PROJECT_ABBREV} {__version__}.",
              "Step numbers can change when a plan changes; the node id is the step's own.",
              "MANIFEST in the plan folder lists the plan's steps; 'mdatk workflow manifest' matches",
              "a plan to the folders of a tree."]
    return tomlio.dumps(data, lambda path, _stack: _IDENTITY_DOCS.get(path[0]) if len(path) == 1 else None,
                        header)


def write_records(plan, written: Sequence[str],
                  when: Optional[_dt.datetime] = None) -> Tuple[str, Dict[str, str]]:
    """Write the identity file of the folders of the steps ``written`` (node ids whose files were just
    written) and the plan folder's MANIFEST (not for the stand-alone wizard's plan, nor before anything is
    in the plan folder).  Returns (MANIFEST path or "", {node id: identity file path})."""
    when = when or _dt.datetime.now()
    identity: Dict[str, str] = {}
    for nid in written:
        step = plan.steps.get(nid)
        if step is None or step.ctx is None or not step.ctx.folder:
            continue  # without step folders the steps share the plan folder: MANIFEST tells them apart
        path = os.path.join(plan.root, step.ctx.folder, IDENTITY)
        _write_text(path, identity_text(plan, nid, when), False)
        identity[nid] = path
    manifest = ""
    if not plan.standalone and os.path.isdir(plan.root):
        manifest = os.path.join(plan.root, MANIFEST)
        _write_text(manifest, manifest_text(plan), False)
    return manifest, identity


# ---------------------------------------------------------------------------
# reading a tree
# ---------------------------------------------------------------------------


def read_identity(folder: str) -> Optional[Dict[str, Any]]:
    """The identity file of ``folder`` as data (None: none, or not one MDATK wrote)."""
    from .. import tomlio

    try:
        data = tomlio.load(os.path.join(folder, IDENTITY))
    except (OSError, ValueError):  # missing, unreadable, or not TOML (TOMLDecodeError is a ValueError)
        return None
    return data if isinstance(data, dict) and data.get("format") == IDENTITY_FORMAT else None


@dataclass
class TreeFolder:
    """A step folder of an existing tree: what its identity file says, else what its name says."""

    name: str  # the folder's name in the tree
    step: Optional[int] = None  # the step's number when it was written (None: not known)
    tool: str = ""  # step type ("": not known, the folder has no identity file)
    tool_tag: str = ""  # the <tool> part of its name
    folder_tag: str = ""
    node: str = ""  # node id ("": no identity file)
    title: str = ""
    label: str = ""
    written: str = ""
    identity: bool = False  # the folder has an identity file

    def describe(self) -> str:
        """e.g. ``17_amber-md/ (step 17, node n12: 17 · production model 12)``."""
        if not self.identity:
            return f"{self.name}/ (no {IDENTITY})"
        step = f"step {step_label(self.step)}, " if self.step is not None else ""
        return f"{self.name}/ ({step}node {self.node}: {self.label or self.title or self.tool})"

    def plain(self) -> Dict[str, Any]:
        return {"folder": self.name, "step": self.step, "node": self.node, "tool": self.tool,
                "title": self.title, "label": self.label, "folder_tag": self.folder_tag,
                "written": self.written, "identity": self.identity}


def scan_tree(tree: str) -> List[TreeFolder]:
    """The step folders directly in ``tree``: those with an identity file, and those named like one
    (``<NN>_<tool>[_<folder tag>]``); ``superseded/`` and other folders are left out."""
    try:
        names = sorted(os.listdir(tree))
    except OSError:
        return []
    out: List[TreeFolder] = []
    for name in names:
        full = os.path.join(tree, name)
        if name == SUPERSEDED or not os.path.isdir(full):
            continue
        parsed = parse_folder(name) or {}
        data = read_identity(full)
        if data is None:
            if parsed:
                out.append(TreeFolder(name, int(parsed["step"]), "", parsed["tag"], parsed["folder_tag"]))
            continue
        step = data.get("step")
        text = {k: str(data.get(k, "")) for k in ("tool", "folder_tag", "node", "title", "label", "written")}
        out.append(TreeFolder(name, step if isinstance(step, int) and not isinstance(step, bool) else None,
                              text["tool"], parsed.get("tag", ""), text["folder_tag"], text["node"],
                              text["title"], text["label"], text["written"], True))
    return out


# ---------------------------------------------------------------------------
# matching a plan to a tree
# ---------------------------------------------------------------------------

STATES = {
    "ours": "this step's",
    "other": "holds another step's files",
    "unknown": f"there, without {IDENTITY} (written before {PROJECT_ABBREV} recorded its folders, or by "
               "hand)",
    "absent": "not there",
    "flat": "the plan writes no step folders",
}


@dataclass
class StepMatch:
    """One step of the plan and the folders of the tree that hold its files."""

    node_id: str
    step: int
    label: str  # the title on its box, e.g. "09 · production model 12"
    title: str  # the same without the number
    tool: str
    folder_tag: str
    folder: str  # the step's folder in the plan ("": the plan writes no step folders)
    state: str  # a key of STATES: what the tree's folder of that name holds
    here: Optional[TreeFolder] = None  # the tree's folder named as the plan names the step's
    how: str = ""  # how ``here`` was matched to this step (state "ours")
    found: List[Tuple[TreeFolder, str]] = field(default_factory=list)  # its other folders, and how matched

    def where(self) -> str:
        """What the tree holds of the step, in a few words."""
        text = STATES[self.state]
        if self.state == "ours" and self.how != "node id":
            text += f" (by {self.how})"
        if self.state == "other" and self.here is not None:
            text += f": {self.here.describe()}"
        if self.found:
            text += "; " + ("its files are in " if self.state in ("absent", "flat") else "and in ")
            text += ", ".join(f"{f.name}/ ({_step_of(f)}by {how})" for f, how in self.found)
        return text

    def plain(self) -> Dict[str, Any]:
        return {"step": self.step, "node": self.node_id, "title": self.title, "label": self.label,
                "tool": self.tool, "folder_tag": self.folder_tag, "folder": self.folder, "state": self.state,
                "here": self.here.plain() if self.here is not None else None, "matched_by": self.how,
                "found": [dict(f.plain(), matched_by=how) for f, how in self.found]}


def _step_of(f: TreeFolder) -> str:
    return f"step {step_label(f.step)}, " if f.step is not None else ""


@dataclass
class TreeMatch:
    """A plan's steps against the step folders of a tree (:func:`match`)."""

    tree: str
    project: str = ""  # the plan's project name
    steps: List[StepMatch] = field(default_factory=list)
    others: List[TreeFolder] = field(default_factory=list)  # the tree's step folders that are no step's

    @property
    def conflicts(self) -> List[StepMatch]:
        """The steps whose folder holds another step's files."""
        return [s for s in self.steps if s.state == "other"]

    def text(self) -> str:
        table = [("step", "folder", "node", "title", "in the tree")]
        for s in self.steps:
            table.append((step_label(s.step), f"{s.folder}/" if s.folder else ".", s.node_id, s.title,
                          s.where()))
        widths = [max(len(row[k]) for row in table) for k in range(4)]
        lines = [f"the steps of plan {self.project} against the folders of {self.tree}"]
        lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) + "  " + row[4] for row in table]
        if len(table) == 1:
            lines.append("(the plan has no steps that write files)")
        if self.others:
            lines += ["", "folders that belong to no step of the plan:"]
            lines += [f"  {f.describe()}" for f in self.others]
        if self.conflicts:
            lines += ["", f"{len(self.conflicts)} step(s) would write into a folder that holds another "
                          "step's files: " + ", ".join(s.label for s in self.conflicts)]
        return "\n".join(lines)

    def plain(self) -> Dict[str, Any]:
        return {"tree": self.tree, "project": self.project, "steps": [s.plain() for s in self.steps],
                "others": [f.plain() for f in self.others], "conflicts": [s.node_id for s in self.conflicts]}


def match(plan, tree: str = "") -> TreeMatch:
    """The plan's steps that write files against the step folders of ``tree`` (default: the plan folder),
    e.g. a tree written before the plan's steps were renumbered (see the module docstring)."""
    if not (tree or plan.root):
        raise ValueError("no tree to match: the plan has no folder")
    tree = os.path.abspath(tree or plan.root)
    folders = scan_tree(tree)
    steps = [n for n in _steps(plan) if _writes(plan.steps[n])]
    writing = set(steps)

    def node(nid):
        return plan.steps[nid].node

    def one(candidates: List[str]) -> Optional[str]:
        return candidates[0] if len(candidates) == 1 else None

    owner: Dict[str, Tuple[str, str]] = {}  # folder name -> (node id, how it was matched)
    for f in folders:  # the node id its identity file names, with the same step type
        if f.identity and f.node in writing and node(f.node).tool == f.tool:
            owner[f.name] = (f.node, "node id")
    for f in folders:  # the one step with its type and folder tag
        if f.name in owner or not f.folder_tag:
            continue
        tag = f.folder_tag.lower()
        if f.identity:
            nid = one([n for n in steps
                       if node(n).tool == f.tool and node_folder_tag(node(n)).lower() == tag])
        else:
            nid = one([n for n in steps if sanitize_tag(plan.steps[n].tool.tag) == f.tool_tag
                       and node_folder_tag(node(n)).lower() == tag])
        if nid is not None:
            owner[f.name] = (nid, "folder tag" if f.identity else "the folder tag in its name")
    for f in folders:  # the one step with its type and title
        if f.name in owner or not f.identity or not f.title.strip():
            continue
        nid = one([n for n in steps if node(n).tool == f.tool and node(n).title.strip() == f.title.strip()])
        if nid is not None:
            owner[f.name] = (nid, "title")

    by_name = {f.name: f for f in folders}
    out = TreeMatch(tree, plan.project)
    for nid in steps:
        step = plan.steps[nid]
        folder = node_folder(plan.workflow, step.node)
        here = by_name.get(folder) if folder else None
        mine, how = owner.get(here.name, ("", "")) if here is not None else ("", "")
        if not plan.workflow.step_folders:
            state = "flat"
        elif here is None:
            state = "absent"
        elif mine == nid:
            state = "ours"
        elif here.identity or mine:
            state = "other"
        else:
            state = "unknown"
        found = [(by_name[name], by) for name, (o, by) in sorted(owner.items())
                 if o == nid and name != folder]
        out.steps.append(StepMatch(nid, step.node.step, step.label, node_title(step.node), step.node.tool,
                                   node_folder_tag(step.node), folder, state, here,
                                   how if state == "ours" else "", found))
    unknown = {s.here.name for s in out.steps if s.state == "unknown"}
    out.others = [f for f in folders if f.name not in owner and f.name not in unknown]
    return out
