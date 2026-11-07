"""What a plan folder holds: its MANIFEST and the identity file of each step folder.

Step numbers are stored with the steps (``PlanNode.step``) and MDATK never renumbers them: a new step gets
the number after the highest one in the plan and a template's steps are numbered after the existing ones.  A
plan written by a script of one's own, or edited by hand, can still be renumbered, and after the last steps
are deleted the steps added next get their numbers, and folders, again; then a folder such as
``17_amber-md`` that holds one step's completed run is named for another step.  So every write
(:meth:`mdatk.workflows.plan.Plan.write`) records what the folders are:

``<plan folder>/MANIFEST``
    one line per step of the plan, tab-separated under a header line: step number, folder, node id, step
    type, title, folder tag, input files (``port=path``, ``;`` between them; plan-relative or absolute) and
    what the step's folder holds after the write: ``this step``, the step its identity file names instead
    (``n3 'production model12' (step 03, written 2026-11-07 17:18)``), ``unknown`` (a folder without one)
    or ``-`` (no folder).  Rewritten at every write, also when only some steps are written (the stand-alone
    wizard's one-step plan is not the folder's plan: it leaves MANIFEST alone).  Steps that write no files
    (existing files, the analysis box) have ``-`` for number and folder; notes are left out.
``<step folder>/MDATK_step.toml``
    in every step folder that a write puts files into: the step's node id, type, title, folder tag, number
    and folder, and when.  It is not part of the step's runs (it stays when a run moves aside into
    ``superseded/``, whose ``manifest.toml`` keeps what it said then, under ``written_by``) and is replaced
    only when the folder is written again, so a tree written before a renumbering still says what each
    folder was.

Whose files a folder holds (:func:`claim`): node ids alone do not tell, since a plan made again from scratch
(e.g. by a script) numbers its ids afresh, and its ``n3`` may be another step than the ``n3`` that wrote a
folder.  So a folder whose identity file names a type, folder tag and title belongs to the one step of the
plan (with its project) with its type and folder tag, else to the one with its type, folder tag and title
(titles count where both have one); the node id only confirms that, or tells apart steps that agree on all of
them (then the match is "by node id only": nothing else on record tells them apart).  When the records
disagree - the identity file and the folder's name, the folder tag and the title, the node id and the rest -
the folder is *ambiguous*: no step is picked.  A folder without an identity file belongs to the one step with
its type and the folder tag in its name.  A step whose folder holds another step's files, or files no step can
be told for, would put its files beside them, and writing it would move their completed run aside as its own:
the planner's check warns (:func:`held`), and :func:`match` lists a plan's steps against the step folders of
an existing tree and the runs moved into their ``superseded/`` (``mdatk workflow manifest``).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
from .naming import SUPERSEDED, parse_folder, sanitize_tag, step_label
from .plan import _write_text, node_folder, node_folder_tag, node_title  # plan.py imports this module lazily
from .registry import TOOLS

MANIFEST = "MANIFEST"
IDENTITY = f"{PROJECT_ABBREV}_step.toml"
IDENTITY_FORMAT = "mdatk-step-folder"
COLUMNS = ("step", "folder", "node", "tool", "title", "folder_tag", "inputs", "holds")
EMPTY = "-"  # an empty cell of MANIFEST
HOLDS_OWN = "this step"  # the "holds" cell of a step whose folder holds its own files
HOLDS_UNKNOWN = f"unknown (no {IDENTITY})"
# would break a table line, or not show: tabs, control characters and every line break str.splitlines() knows
_CELL_BAD = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]+")

_IDENTITY_DOCS = {
    "format": "File type marker; do not change.",
    "mdatk_version": f"{PROJECT_ABBREV} version that wrote the step's files into this folder.",
    "written": "When they were written.",
    "project": "The plan's project name (the <project> part of the file names).",
    "step": "The step's number when it was written (the <step> part of the folder and file names).",
    "folder": "This folder, relative to the plan folder.",
    "node": "The step's id in the plan (n1, n2 ...). RNAdo never gives a deleted step's id to a new step, but a "
            "plan made again from scratch (e.g. by a script) numbers its ids afresh: a step of a plan is this "
            "folder's by its type, folder tag and title below, and the id only confirms them or tells apart "
            "steps that agree on all of them.",
    "tool": "The step type.",
    "title": "The step's own title (\"\": none; its box shows the type's label).",
    "label": "The step's title as RNAdo shows it: number, title and folder tag.",
    "folder_tag": "The step's folder tag (\"\": none).",
    "stamp": "Time stamp in the names of the files written (\"\": a plan without time stamps).",
}


def one_line(text: Any) -> str:
    """``text`` on one line: tabs and line breaks become spaces."""
    return _CELL_BAD.sub(" ", str(text))


def _when(value: Any) -> str:
    """A time of an identity file or a run's manifest as ``2026-11-07 17:18`` ("" when there is none)."""
    if isinstance(value, _dt.datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    text = str(value or "").strip()
    try:
        return _dt.datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _steps(plan) -> List[str]:
    """The plan's steps in plan order (steps in a loop of connections last), notes left out."""
    ids = list(plan.order) + [n for n in plan.steps if n not in plan.order]
    return [n for n in ids if plan.steps[n].node.tool != "note"]


def _writes(step) -> bool:
    return step.tool is not None and step.tool.generates


def _writing(plan) -> List[str]:
    """The plan's steps that write files, in plan order."""
    return [n for n in _steps(plan) if _writes(plan.steps[n])]


def _inputs(step) -> str:
    """The step's input files; for an existing file (no inputs, writes nothing) the file it stands for."""
    inputs = step.ctx.inputs if step.ctx is not None else {}
    pairs = [(port, p) for port, v in inputs.items() for p in v.paths]
    if not pairs and not _writes(step):
        pairs = [(port, p) for port, v in step.outputs.items() for p in v.paths]
    return "; ".join(f"{port}={path}" for port, path in pairs)


def rows(plan, written: Collection[str] = ()) -> List[Tuple[str, str, str, str, str, str, str, str]]:
    """The lines of the plan's MANIFEST (the columns of :data:`COLUMNS`; "" for an empty cell).  The last
    column says what each step's folder holds as the folders are now (:func:`holds`); the steps of
    ``written`` count as holding their own files (the steps a write is about to write)."""
    out = []
    for nid in _steps(plan):
        step = plan.steps[nid]
        node = step.node
        writes = _writes(step)
        folder = (node_folder(plan.workflow, node) or ".") if writes else ""
        out.append((step_label(node.step) if writes else "", folder, node.id, node.tool, node_title(node),
                    node_folder_tag(node), _inputs(step), holds(plan, nid, written)))
    return out


def _cell(value: Any) -> str:
    return one_line(value).strip() or EMPTY


def manifest_text(plan, written: Collection[str] = ()) -> str:
    """The plan folder's MANIFEST: a header line, then one tab-separated line per step (``written``: see
    :func:`rows`)."""
    lines = ["\t".join(COLUMNS)] + ["\t".join(_cell(v) for v in row) for row in rows(plan, written)]
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
    header = [f"This folder holds the files of step {one_line(step.label)} of plan {plan.project}, written by "
              f"{PROJECT_ABBREV} {__version__}.",
              "Step numbers change when a plan is renumbered, and a plan made again from scratch numbers its",
              "step ids afresh: 'mdatk workflow manifest' matches a plan to the folders of a tree by type,",
              "folder tag and title (the node id only confirms them). MANIFEST in the plan folder lists the",
              "plan's steps."]
    return tomlio.dumps(data, lambda path, _stack: _IDENTITY_DOCS.get(path[0]) if len(path) == 1 else None,
                        header)


def would_write(plan) -> List[str]:
    """The steps a write of every step would put files into: those without errors that do not use an earlier
    run and have files to write (for MANIFEST's "holds" before a write: ``mdatk workflow manifest --table``)."""
    from .plan import generate_step

    out = []
    for nid in plan.order:
        step = plan.steps[nid]
        if not _writes(step) or step.adopted or not step.ok:
            continue
        try:
            files = step.files or generate_step(step)
        except Exception:  # noqa: BLE001 - writing would fail on it: nothing is written there
            continue
        if files:
            out.append(nid)
    return out


def _write_record(path: str, text: str, errors: List[str]) -> bool:
    """Write one record; on an error leave no ``.tmp`` file behind and add the error to ``errors``."""
    try:
        _write_text(path, text, False)
        return True
    except OSError as exc:
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        errors.append(f"{path}: {exc.strerror or exc}")
        return False


def write_records(plan, written: Sequence[str],
                  when: Optional[_dt.datetime] = None) -> Tuple[str, Dict[str, str], str]:
    """Write the identity file of the folders of the steps ``written`` (node ids whose files were just
    written) and the plan folder's MANIFEST (not for the stand-alone wizard's plan, nor before anything is
    in the plan folder).  A record that cannot be written never fails the write: the step files are there.
    Returns (MANIFEST path or "", {node id: identity file path}, the records not written and why or "")."""
    when = when or _dt.datetime.now()
    identity: Dict[str, str] = {}
    errors: List[str] = []
    for nid in written:
        step = plan.steps.get(nid)
        if step is None or step.ctx is None or not step.ctx.folder:
            continue  # without step folders the steps share the plan folder: MANIFEST tells them apart
        path = os.path.join(plan.root, step.ctx.folder, IDENTITY)
        if _write_record(path, identity_text(plan, nid, when), errors):
            identity[nid] = path
    manifest = ""
    if not plan.standalone and os.path.isdir(plan.root):
        path = os.path.join(plan.root, MANIFEST)
        if _write_record(path, manifest_text(plan, written), errors):  # the steps just written hold their files
            manifest = path
    return manifest, identity, "; ".join(errors)


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
    """A step folder of an existing tree, or a run moved into its ``superseded/``: what its identity file
    (for a run: its manifest) says, else what its name says."""

    name: str  # the folder relative to the tree: a step folder, or <step folder>/superseded/<run>
    step: Optional[int] = None  # the step's number when it was written (None: not known)
    tool: str = ""  # step type ("": not known, the folder has no identity file)
    tool_tag: str = ""  # the <tool> part of the step folder's name
    folder_tag: str = ""
    node: str = ""  # node id ("": no identity file)
    title: str = ""
    label: str = ""
    written: str = ""  # when the step's files were written into the folder
    identity: bool = False  # the folder has an identity file (a run: a manifest)
    project: str = ""  # the plan's project name when it was written
    moved: str = ""  # a run moved into superseded/: when it moved ("": a step folder)
    recorded: str = ""  # the step folder its identity file was written for ("": not recorded)

    def renamed(self) -> str:
        """What disagrees when the identity file was written for another folder than the one it is in (the
        folder renamed since, or the file copied from there); "" when it names this folder, or none."""
        folder = self.name.split(f"/{SUPERSEDED}/")[0]
        if not self.identity or not self.recorded or self.recorded.strip("/").lower() == folder.lower():
            return ""
        return f"its records were written for {self.recorded.strip('/')}/, not {folder}/ (renamed or copied since)"

    def _name(self) -> str:
        tool = TOOLS.get(self.tool)
        return self.title.strip() or (tool.label if tool is not None else self.tool) or "?"

    def _of(self, project: str = "") -> str:
        return f" of plan {self.project}" if self.project and project and self.project != project else ""

    def who(self, project: str = "") -> str:
        """e.g. ``03 · production model12 (node n3, written 2026-11-07 17:18)``."""
        when = _when(self.written)
        return (f"{one_line(self.label or self._name())} (node {self.node}{self._of(project)}"
                + (f", written {when})" if when else ")"))

    def short(self, project: str = "") -> str:
        """e.g. ``n3 'production model12' (step 03, written 2026-11-07 17:18)`` (MANIFEST's "holds")."""
        parts = [f"step {step_label(self.step)}"] if self.step is not None else []
        if _when(self.written):
            parts.append(f"written {_when(self.written)}")
        return (f"{self.node} '{one_line(self._name())}'{self._of(project)}"
                + (f" ({', '.join(parts)})" if parts else ""))

    def describe(self, project: str = "") -> str:
        """e.g. ``17_amber-md/ (step 17, node n12: 17 · production model 12, written 2026-11-07 17:18)``."""
        if not self.identity:
            return f"{self.name}/ (no {IDENTITY})"
        step = f"step {step_label(self.step)}, " if self.step is not None else ""
        what = f"{step}node {self.node}{self._of(project)}: {one_line(self.label or self._name())}"
        if self.moved:
            return f"{self.name}/ (a run moved aside {_when(self.moved)}; {what})"
        when = _when(self.written)
        return f"{self.name}/ ({what}" + (f", written {when})" if when else ")")

    def plain(self) -> Dict[str, Any]:
        return {"folder": self.name, "step": self.step, "node": self.node, "tool": self.tool,
                "title": self.title, "label": self.label, "folder_tag": self.folder_tag,
                "written": self.written, "identity": self.identity, "project": self.project,
                "moved": self.moved, "recorded": self.recorded}


def _number(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _record(name: str, data: Dict[str, Any], tool_tag: str = "") -> TreeFolder:
    """The :class:`TreeFolder` of ``name`` with the identity file ``data``."""
    text = {k: str(data.get(k, "") or "") for k in ("tool", "folder_tag", "node", "title", "label", "project",
                                                    "folder")}
    return TreeFolder(name, _number(data.get("step")), text["tool"], tool_tag, text["folder_tag"], text["node"],
                      text["title"], text["label"], str(data.get("written", "") or ""), True, text["project"],
                      recorded=text["folder"])


def folder_record(tree: str, name: str) -> Optional[TreeFolder]:
    """What the step folder ``name`` of ``tree`` says it is: its identity file, else its name (None: it is not
    a step folder: no identity file and not named like one)."""
    full = os.path.join(tree, name)
    parsed = parse_folder(name) or {}
    data = read_identity(full)
    if data is None:
        return TreeFolder(name, int(parsed["step"]), "", parsed["tag"], parsed["folder_tag"]) if parsed else None
    return _record(name, data, parsed.get("tag", ""))


def scan_tree(tree: str) -> List[TreeFolder]:
    """The step folders directly in ``tree``: those with an identity file, and those named like one
    (``<NN>_<tool>[_<folder tag>]``); ``superseded/`` and other folders are left out."""
    try:
        names = sorted(os.listdir(tree))
    except OSError:
        return []
    out: List[TreeFolder] = []
    for name in names:
        if name == SUPERSEDED or not os.path.isdir(os.path.join(tree, name)):
            continue
        record = folder_record(tree, name)
        if record is not None:
            out.append(record)
    return out


def moved_runs(tree: str, folders: Sequence[TreeFolder]) -> List[TreeFolder]:
    """The runs moved aside into the ``superseded/`` of the step folders ``folders`` of ``tree``, each with
    the step that wrote its files: the folder's identity file when the run moved (``written_by`` of its
    manifest), else the step written then (a run moved before MDATK recorded the identity)."""
    from .. import tomlio
    from .supersede import MANIFEST as RUN_MANIFEST, MANIFEST_FORMAT

    out: List[TreeFolder] = []
    for f in folders:
        store = os.path.join(tree, f.name, SUPERSEDED)
        try:
            runs = sorted(os.listdir(store))
        except OSError:
            continue
        for run in runs:
            try:
                data = tomlio.load(os.path.join(store, run, RUN_MANIFEST))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or data.get("format") != MANIFEST_FORMAT:
                continue
            name = f"{f.name}/{SUPERSEDED}/{run}"
            by = data.get("written_by")
            if isinstance(by, dict):
                record = _record(name, by, f.tool_tag)
            else:  # the step whose files were written again then: its title and folder tag are not recorded
                record = TreeFolder(name, _number(data.get("step")), str(data.get("tool", "") or ""), f.tool_tag,
                                    (parse_folder(f.name) or {}).get("folder_tag", ""),
                                    str(data.get("node", "") or ""), "", str(data.get("title", "") or ""),
                                    "", True, str(data.get("project", "") or ""))
            record.moved = str(data.get("moved", "") or "?")
            out.append(record)
    return out


# ---------------------------------------------------------------------------
# whose files a folder holds
# ---------------------------------------------------------------------------


def differs(f: TreeFolder, node, project: str) -> List[str]:
    """What of the step ``node`` (of a plan with ``project``) disagrees with the identity ``f`` names: of
    "type", "project", "folder tag" and "title" (titles only when both have one)."""
    out = []
    if f.tool != node.tool:
        out.append("type")
    if f.project and f.project != project:
        out.append("project")
    if f.folder_tag.strip().lower() != node_folder_tag(node).lower():
        out.append("folder tag")
    mine, theirs = f.title.strip(), (node.title or "").strip()
    if mine and theirs and mine != theirs:
        out.append("title")
    return out


@dataclass
class Claim:
    """Whose files a step folder, or a run moved aside, holds among the steps of a plan (:func:`claim`)."""

    node: str = ""  # the step's node id ("": no step of the plan can be told)
    # what told it: "folder tag" (the one step with the folder's type and folder tag), "title" (the one with
    # its type, folder tag and title), "node id" (only the node id tells it from the steps that agree on type,
    # folder tag and title) or "the folder tag in its name" (a folder without an identity file)
    how: str = ""
    confirmed: bool = False  # the identity file names that step's node id too
    doubt: str = ""  # no step is told because the records disagree: what disagrees (an ambiguous folder)

    @property
    def weak(self) -> bool:
        """Only the node id tells the step from others of its type: a plan made again from scratch gives its
        ids out afresh, so nothing on record says the files are this step's rather than another one's."""
        return self.how == "node id"

    def by(self) -> str:
        """How the step was told, for the text: e.g. ``title and node id``, ``node id only``."""
        if self.weak:
            return "node id only"
        return self.how + (" and node id" if self.confirmed else "")


def _labels(plan, ids: Sequence[str]) -> str:
    return ", ".join(one_line(plan.steps[n].label) for n in ids)


def claim(plan, f: TreeFolder, steps: Optional[Sequence[str]] = None) -> Claim:
    """Whose files ``f`` holds among the steps of ``plan`` (``steps``, by default every step that writes
    files): by its type and folder tag, then by its title; the node id only confirms the step, or tells it
    from the steps that agree on all of them.  Records that disagree (with the folder's name, the folder tag
    with the title, the node id with the rest) tell no step: :attr:`Claim.doubt` says why.  See the module
    docstring."""
    if steps is None:
        steps = _writing(plan)

    def node(nid):
        return plan.steps[nid].node

    def title(nid) -> str:
        return (node(nid).title or "").strip()

    def tag(nid) -> str:
        return node_folder_tag(node(nid)).lower()

    want_tag = f.folder_tag.strip().lower()
    if not f.identity:  # its name alone: the one step with its type and the folder tag in the name
        if not want_tag:
            return Claim()
        found = [n for n in steps if sanitize_tag(plan.steps[n].tool.tag) == f.tool_tag and tag(n) == want_tag]
        if len(found) == 1:
            return Claim(found[0], "the folder tag in its name")
        return Claim(doubt=f"its name fits {len(found)} steps ({_labels(plan, found)})") if found else Claim()
    if f.project and f.project != plan.project:
        return Claim()  # another plan's files (their names are not this plan's)
    if f.renamed():  # the identity file and the folder's name disagree
        return Claim(doubt=f.renamed())
    nid, want_title = f.node, f.title.strip()
    same = [n for n in steps if node(n).tool == f.tool and tag(n) == want_tag]  # its type and folder tag (or none)
    fits = [n for n in same if not want_title or not title(n) or title(n) == want_title]  # titles: where both have one
    exact = [n for n in fits if want_title and title(n) == want_title]

    def told(n: str, how: str) -> Claim:
        return Claim(n, how, n == nid)

    contradicted = nid in same and nid not in fits  # the step of its node id has its type and tag, another title
    if want_tag and len(fits) == 1 and (fits[0] in exact or fits[0] == nid or not contradicted):
        return told(fits[0], "folder tag")  # the one step with its type and folder tag (and no other title)
    if want_tag and same and not fits:  # the steps with its folder tag have another title
        return Claim(doubt=f"{_labels(plan, same)} {'has' if len(same) == 1 else 'have'} its folder tag"
                           + (" and node id" if nid in same else "") + f" but not its title '{one_line(want_title)}'")
    if len(exact) == 1:  # the one step with its type, folder tag and title
        return told(exact[0], "title")
    if nid in (exact or fits):  # the steps that fit are told apart by the node id alone
        return told(nid, "node id")
    if len(exact) > 1 or (want_tag and len(fits) > 1):
        many = exact or fits
        return Claim(doubt=f"its type, folder tag and title fit {len(many)} steps ({_labels(plan, many)}), its "
                           "node id none of them")
    if nid in same:  # the step of its node id, type and folder tag has another title: renamed, or another one
        return Claim(doubt=f"{one_line(plan.steps[nid].label)} has its {'folder tag and ' if want_tag else ''}node "
                           f"id but not its title '{one_line(want_title)}'")
    return Claim()


def _own_folder(plan, node_id: str) -> Optional[TreeFolder]:
    """What the step's folder in the plan folder says it is (None: the step has no folder of its own, or
    nothing is there)."""
    step = plan.steps.get(node_id)
    if step is None or not _writes(step) or step.ctx is None or not step.ctx.folder or not plan.root:
        return None
    if not os.path.isdir(os.path.join(plan.root, step.ctx.folder)):
        return None
    return folder_record(plan.root, step.ctx.folder)


def held(plan, node_id: str) -> Tuple[str, str]:
    """Whether the step's folder holds the files of another step, or files no step can be told for, by its
    identity file: (what they are, e.g. ``03_amber-md/ holds the files of 03 · production model12 (node n3,
    written 2026-11-07 17:18), now 09 · production model12``, and what may explain it: "project" when they
    are another plan's (their file names differ from this plan's), "title" when only the title tells them
    from the step's own (the step renamed since), else "").  ("", "") when they are the step's own (also by
    node id only), nothing says (no identity file) or the step has no folder of its own."""
    f = _own_folder(plan, node_id)
    if f is None or not f.identity:
        return "", ""
    owner = claim(plan, f)
    if owner.node == node_id:
        return "", ""
    text = f"{f.name}/ holds the files of {f.who(plan.project)}"
    what = differs(f, plan.steps[node_id].node, plan.project)
    if owner.node:  # another step of the plan: certainly not this one under an earlier title
        text += f", now {one_line(plan.steps[owner.node].label)}"
    elif "project" in what:
        return text, "project"
    elif f.node == node_id and what == ["title"]:  # the step renamed since, or another one: write it to tell
        return text, "title"
    elif owner.doubt:
        text += f"; {owner.doubt}"
    return text, ""


def holds(plan, node_id: str, written: Collection[str] = ()) -> str:
    """MANIFEST's "holds" cell of the step: "this step" (with "(node id only)" when nothing else on record
    tells it), the step its folder's identity file names (e.g. ``n3 'production model12' (step 03, written
    2026-11-07 17:18)``), "unknown (no MDATK_step.toml)", or "" (no folder of its own, or none yet).  A step of
    ``written`` holds its own files."""
    step = plan.steps.get(node_id)
    if step is None or not _writes(step) or step.ctx is None or not step.ctx.folder or not plan.root:
        return ""
    if node_id in written:
        return HOLDS_OWN
    f = _own_folder(plan, node_id)
    if f is None:
        return ""
    if not f.identity:
        return HOLDS_UNKNOWN
    owner = claim(plan, f)
    if owner.node == node_id:
        return HOLDS_OWN + (" (node id only)" if owner.weak else "")
    return f.short(plan.project)


# ---------------------------------------------------------------------------
# matching a plan to a tree
# ---------------------------------------------------------------------------

STATES = {
    "ours": "this step's",
    "other": "holds another step's files",
    "ambiguous": "holds files no step can be told for",
    "unknown": f"there, without {IDENTITY} (written before {PROJECT_ABBREV} recorded its folders, or by "
               "hand)",
    "absent": "not there",
    "flat": "the plan writes no step folders",
}
CONFLICTS = ("other", "ambiguous")  # the states of a step whose folder holds files that are not certainly its own
WEAK_NOTE = ("by node id only: nothing else on record tells these steps from the others of their type, and a "
             "plan made again from scratch gives node ids out afresh; titles or folder tags tell them for certain")


@dataclass
class StepMatch:
    """One step of the plan and the folders of the tree that hold its files."""

    node_id: str
    step: int
    label: str  # the step's title as RNAdo shows it, e.g. "09 · production model 12"
    title: str  # its title without the number
    tool: str
    folder_tag: str
    folder: str  # the step's folder in the plan ("": the plan writes no step folders)
    state: str  # a key of STATES: what the tree's folder of that name holds
    here: Optional[TreeFolder] = None  # the tree's folder named as the plan names the step's
    told: Claim = field(default_factory=Claim)  # whose files ``here`` holds, and how that was told
    # its other folders and moved runs, and how they were told
    found: List[Tuple[TreeFolder, Claim]] = field(default_factory=list)
    owner: str = ""  # state "other": the label of the plan's step whose files ``here`` holds ("": none)
    differs: List[str] = field(default_factory=list)  # state other/ambiguous, ``here`` names this step's id: what not
    project: str = ""  # the plan's project name

    @property
    def how(self) -> str:
        """How ``here`` was told to be this step's (state "ours"; see :class:`Claim`)."""
        return self.told.how if self.state == "ours" else ""

    @property
    def weak(self) -> bool:
        """``here`` is this step's by node id only (:attr:`Claim.weak`)."""
        return self.state == "ours" and self.told.weak

    def where(self) -> str:
        """What the tree holds of the step, in a few words."""
        text = STATES[self.state]
        if self.state == "ours" and (self.told.weak or not self.told.confirmed):
            text += f" (by {self.told.by()})"
        if self.state in CONFLICTS and self.here is not None:
            text += f": {self.here.describe(self.project)}"
            if self.owner:
                text += f", now {one_line(self.owner)}"
            if self.told.doubt:
                text += f"; {self.told.doubt}"
            elif self.differs:
                text += f"; this step has its node id with another {' and '.join(self.differs)}"
        own = f"{self.folder}/{SUPERSEDED}/"
        earlier = [f for f, _c in self.found if self.state == "ours" and f.moved and f.name.startswith(own)]
        listed = [(f, c) for f, c in self.found if f not in earlier]
        if listed:
            text += "; " + ("its files are in " if self.state in ("absent", "flat") else "and in ")
            text += ", ".join(f"{f.name}/ ({_step_of(f)}by {c.by()})" for f, c in listed)
        if earlier:
            text += f"; {len(earlier)} earlier run(s) in its {SUPERSEDED}/"
        return text

    def plain(self) -> Dict[str, Any]:
        return {"step": self.step, "node": self.node_id, "title": self.title, "label": self.label,
                "tool": self.tool, "folder_tag": self.folder_tag, "folder": self.folder, "state": self.state,
                "here": self.here.plain() if self.here is not None else None, "matched_by": self.how,
                "node_id_agrees": self.told.confirmed if self.state == "ours" else False,
                "doubt": self.told.doubt, "owner": self.owner, "differs": list(self.differs),
                "found": [dict(f.plain(), matched_by=c.how, node_id_agrees=c.confirmed) for f, c in self.found]}


def _step_of(f: TreeFolder) -> str:
    text = f"step {step_label(f.step)}, " if f.step is not None else ""
    return text + (f"moved {_when(f.moved)}, " if f.moved else "")


@dataclass
class TreeMatch:
    """A plan's steps against the step folders of a tree (:func:`match`)."""

    tree: str
    project: str = ""  # the plan's project name
    steps: List[StepMatch] = field(default_factory=list)
    # the tree's step folders and moved runs that are no step's
    others: List[TreeFolder] = field(default_factory=list)
    # of ``others`` whose identity names a step id of the plan: what of that step disagrees with it
    differ: Dict[str, Tuple[str, List[str]]] = field(default_factory=dict)  # folder -> (step label, what)
    doubt: Dict[str, str] = field(default_factory=dict)  # of ``others``: why no step can be told (Claim.doubt)

    @property
    def conflicts(self) -> List[StepMatch]:
        """The steps whose folder holds another step's files, or files no step can be told for."""
        return [s for s in self.steps if s.state in CONFLICTS]

    @property
    def weak(self) -> List[StepMatch]:
        """The steps whose folder, or another folder or run of theirs, is theirs by node id only."""
        return [s for s in self.steps if s.weak or any(c.weak for _f, c in s.found)]

    def text(self) -> str:
        table = [("step", "folder", "node", "title", "in the tree")]
        for s in self.steps:
            table.append((step_label(s.step), f"{s.folder}/" if s.folder else ".", s.node_id, one_line(s.title),
                          one_line(s.where())))
        widths = [max(len(row[k]) for row in table) for k in range(4)]
        lines = [f"the steps of plan {self.project} against the folders of {self.tree}"]
        lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) + "  " + row[4] for row in table]
        if len(table) == 1:
            lines.append("(the plan has no steps that write files)")
        if self.others:
            lines += ["", "folders that belong to no step of the plan:"]
            for f in self.others:
                line = f"  {f.describe(self.project)}"
                if f.name in self.doubt:
                    line += f"; {self.doubt[f.name]}"
                elif f.name in self.differ:
                    label, what = self.differ[f.name]
                    line += f"; the plan's {f.node} ({one_line(label)}) has another {' and '.join(what)}"
                lines.append(line)
        if self.conflicts:
            unsure = sum(s.state == "ambiguous" for s in self.conflicts)
            lines += ["", f"{len(self.conflicts)} step(s) would write into a folder that holds another "
                          "step's files" + (f" ({unsure} of them: files no step can be told for)" if unsure else "")
                      + ": " + ", ".join(one_line(s.label) for s in self.conflicts)]
        if self.weak:
            lines += ["", WEAK_NOTE]
        return "\n".join(lines)

    def plain(self) -> Dict[str, Any]:
        others = []
        for f in self.others:
            label, what = self.differ.get(f.name, ("", []))
            others.append(dict(f.plain(), differs=list(what), doubt=self.doubt.get(f.name, "")))
        return {"tree": self.tree, "project": self.project, "steps": [s.plain() for s in self.steps],
                "others": others, "conflicts": [s.node_id for s in self.conflicts],
                "by_node_id_only": [s.node_id for s in self.weak]}


def match(plan, tree: str = "") -> TreeMatch:
    """The plan's steps that write files against the step folders of ``tree`` (default: the plan folder) and
    the runs moved into their ``superseded/``, e.g. a tree written before the plan's steps were renumbered
    (see the module docstring)."""
    if not (tree or plan.root):
        raise ValueError("no tree to match: the plan has no folder")
    tree = os.path.abspath(tree or plan.root)
    folders = scan_tree(tree)
    every = folders + moved_runs(tree, folders)
    steps = _writing(plan)
    claims = {f.name: claim(plan, f, steps) for f in every}  # folder name -> whose files it holds
    by_name = {f.name: f for f in every}
    out = TreeMatch(tree, plan.project)
    for nid in steps:
        step = plan.steps[nid]
        folder = node_folder(plan.workflow, step.node)
        here = by_name.get(folder) if folder else None
        told = claims[here.name] if here is not None else Claim()
        if not plan.workflow.step_folders:
            state = "flat"
        elif here is None:
            state = "absent"
        elif told.node == nid:
            state = "ours"
        elif told.node:
            state = "other"
        elif told.doubt:
            state = "ambiguous"
        elif here.identity:
            state = "other"
        else:
            state = "unknown"
        found = [(by_name[name], c) for name, c in sorted(claims.items()) if c.node == nid and name != folder]
        mismatch = state in CONFLICTS and here is not None
        out.steps.append(StepMatch(
            nid, step.node.step, step.label, node_title(step.node), step.node.tool, node_folder_tag(step.node),
            folder, state, here, told, found, plan.steps[told.node].label if mismatch and told.node else "",
            differs(here, step.node, plan.project) if mismatch and here.identity and here.node == nid else [],
            plan.project))
    named = {s.here.name for s in out.steps if s.here is not None}  # their rows say what these hold
    out.others = [f for f in every if not claims[f.name].node and f.name not in named]
    for f in out.others:
        if claims[f.name].doubt:
            out.doubt[f.name] = claims[f.name].doubt
        if f.identity and f.node in plan.steps and plan.steps[f.node].tool is not None:
            what = differs(f, plan.steps[f.node].node, plan.project)
            if what:
                out.differ[f.name] = (plan.steps[f.node].label, what)
    return out
