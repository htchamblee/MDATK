"""Project Planner: turn a plan (steps + connections) into consistently named files.

A plan is a directed graph stored in :class:`mdatk.config.WorkflowParams`.
Resolving it walks the steps in dependency order, hands every step the file
names its upstream steps will produce, runs the step's logic checks and
collects the names of its own outputs.  Writing it generates every step's
files into ``<plan folder>/<step>_<tool>/`` plus a master script and a
Markdown summary named ``<project>_00_plan_<stamp>.{sh,md}``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import heapq
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
from ..config import PlanEdge, PlanNode, WorkflowParams
from .base import PortValue, StepContext, Tool
from .naming import file_name, new_stamp, sanitize_project, step_label
from .registry import TOOLS
from .schema import GeneratedFile, Issue, Port, errors, guess_kind, kind_label


@dataclass
class ResolvedStep:
    node: PlanNode
    tool: Optional[Tool]
    settings: Dict[str, Any] = field(default_factory=dict)
    ctx: Optional[StepContext] = None
    outputs: Dict[str, PortValue] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    files: List[GeneratedFile] = field(default_factory=list)

    @property
    def label(self) -> str:
        return step_title(self.node)

    @property
    def ok(self) -> bool:
        return not errors(self.issues)

    @property
    def adopted(self) -> str:
        """Stamp of the earlier run whose outputs this step uses ("": none); such a step is not run again."""
        return self.node.adopted if self.tool is not None and self.tool.generates else ""

    def level(self) -> str:
        levels = {i.level for i in self.issues}
        for lvl in ("error", "warning", "info"):
            if lvl in levels:
                return lvl
        return "ok"


@dataclass
class WriteReport:
    root: str
    written: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)  # node id -> reason
    kept: Dict[str, str] = field(default_factory=dict)  # node id -> stamp of the earlier run it uses
    master: str = ""
    summary: str = ""


def step_title(node: PlanNode) -> str:
    tool = TOOLS.get(node.tool)
    title = node.title or (tool.label if tool else node.tool)
    if tool is not None and not tool.generates:
        return title
    return f"{step_label(node.step)} · {title}"


def new_node_id(workflow: WorkflowParams) -> str:
    used = {n.id for n in workflow.nodes}
    k = len(workflow.nodes) + 1
    while f"n{k}" in used:
        k += 1
    return f"n{k}"


def next_step(workflow: WorkflowParams) -> int:
    steps = [n.step for n in workflow.nodes if TOOLS.get(n.tool) is not None and TOOLS[n.tool].generates]
    return max(steps, default=0) + 1


def add_node(workflow: WorkflowParams, tool: str, x: float = 40.0, y: float = 40.0,
             step: Optional[int] = None, title: str = "",
             settings: Optional[Dict[str, Any]] = None) -> PlanNode:
    if tool not in TOOLS:
        raise KeyError(f"unknown step type {tool!r}")
    t = TOOLS[tool]
    node = PlanNode(id=new_node_id(workflow), tool=tool,
                    step=step if step is not None else (next_step(workflow) if t.generates else 0),
                    title=title, x=float(x), y=float(y),
                    settings=t.normalise(settings or {}))
    workflow.nodes.append(node)
    return node


def remove_node(workflow: WorkflowParams, node_id: str) -> None:
    workflow.nodes = [n for n in workflow.nodes if n.id != node_id]
    workflow.edges = [e for e in workflow.edges if node_id not in (e.source, e.target)]


def connect(workflow: WorkflowParams, source: str, output: str, target: str, input: str) -> PlanEdge:
    """Add a connection (replacing any other edge into a single-file input)."""
    nodes = {n.id: n for n in workflow.nodes}
    if source not in nodes or target not in nodes:
        raise KeyError("unknown step")
    if source == target:
        raise ValueError("a step cannot feed itself")
    src_tool, dst_tool = TOOLS[nodes[source].tool], TOOLS[nodes[target].tool]
    src_s = src_tool.normalise(nodes[source].settings)
    dst_s = dst_tool.normalise(nodes[target].settings)
    out_port = _port(src_tool.output_ports(src_s), output)
    in_port = _port(dst_tool.input_ports(dst_s), input)
    if out_port is None or in_port is None:
        raise KeyError("unknown port")
    if not in_port.accepts(out_port.kind):
        raise ValueError(f"{in_port.label} accepts {', '.join(kind_label(k) for k in in_port.kinds)}, "
                         f"not {kind_label(out_port.kind)}")
    if _reaches(workflow, target, source):
        raise ValueError("this connection would create a loop")
    if not in_port.multiple:
        workflow.edges = [e for e in workflow.edges if not (e.target == target and e.input == input)]
    edge = PlanEdge(source, output, target, input)
    if not any(e == edge for e in workflow.edges):
        workflow.edges.append(edge)
    return edge


def _reaches(workflow: WorkflowParams, start: str, goal: str) -> bool:
    todo, seen = [start], set()
    while todo:
        cur = todo.pop()
        if cur == goal:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        todo.extend(e.target for e in workflow.edges if e.source == cur)
    return False


def _port(ports: Sequence[Port], name: str) -> Optional[Port]:
    for p in ports:
        if p.name == name:
            return p
    return None


class Plan:
    """A resolved plan.  ``root`` is the plan folder (needed for writing)."""

    def __init__(self, workflow: WorkflowParams, root: str = "", groups: Optional[Sequence] = None,
                 stamp: Optional[str] = None):
        self.workflow = workflow
        self.root = os.path.abspath(root) if root else ""
        self.groups = list(groups or [])
        if stamp is None:
            stamp = (workflow.stamp or new_stamp()) if workflow.use_stamp else ""
        self.stamp = stamp
        self.project = sanitize_project(workflow.project_name)
        self.issues: List[Issue] = []  # plan-wide problems
        self.steps: Dict[str, ResolvedStep] = {}
        self.order: List[str] = []
        self._resolve()

    # ------------------------------------------------------------ resolve
    def _resolve(self) -> None:
        nodes: Dict[str, PlanNode] = {}
        for n in self.workflow.nodes:
            if n.id in nodes or not n.id:
                self.issues.append(Issue("error", f"duplicate or empty step id {n.id!r}"))
                continue
            nodes[n.id] = n
        for nid, node in nodes.items():
            tool = TOOLS.get(node.tool)
            step = ResolvedStep(node, tool)
            if tool is None:
                step.issues.append(Issue("error", f"unknown step type {node.tool!r}"))
            else:
                step.settings = tool.normalise(node.settings, step.issues)
            self.steps[nid] = step

        edges = self._valid_edges(nodes)
        self.order, cyclic = self._toposort(nodes, edges)
        for nid in cyclic:
            self.steps[nid].issues.append(Issue("error", "this step is part of a loop of connections"))

        for nid in self.order:
            self._resolve_step(self.steps[nid], [e for e in edges if e.target == nid])
        self._check_steps_and_names()

    def _valid_edges(self, nodes: Dict[str, PlanNode]) -> List[PlanEdge]:
        valid = []
        for e in self.workflow.edges:
            src, dst = self.steps.get(e.source), self.steps.get(e.target)
            if src is None or dst is None or src.tool is None or dst.tool is None:
                self.issues.append(Issue("warning", f"ignoring a connection to a missing step "
                                                    f"({e.source} -> {e.target})"))
                continue
            if _port(src.tool.output_ports(src.settings), e.output) is None:
                dst.issues.append(Issue("warning", f"{src.label} no longer produces '{e.output}'; "
                                                   "that connection is ignored"))
                continue
            if _port(dst.tool.input_ports(dst.settings), e.input) is None:
                dst.issues.append(Issue("warning", f"this step no longer has an input '{e.input}'; "
                                                   "that connection is ignored"))
                continue
            valid.append(e)
        return valid

    def _toposort(self, nodes: Dict[str, PlanNode], edges: List[PlanEdge]) -> Tuple[List[str], List[str]]:
        indeg = {nid: 0 for nid in nodes}
        succ: Dict[str, List[str]] = {nid: [] for nid in nodes}
        for e in edges:
            indeg[e.target] += 1
            succ[e.source].append(e.target)

        def key(nid):
            n = nodes[nid]
            return (n.step, n.y, n.x, nid)

        heap = [(key(n), n) for n, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            _, nid = heapq.heappop(heap)
            order.append(nid)
            for t in succ[nid]:
                indeg[t] -= 1
                if indeg[t] == 0:
                    heapq.heappush(heap, (key(t), t))
        cyclic = [nid for nid, d in indeg.items() if d > 0]
        return order, cyclic

    def _resolve_step(self, step: ResolvedStep, incoming: List[PlanEdge]) -> None:
        tool, node = step.tool, step.node
        if tool is None:
            return
        inputs: Dict[str, PortValue] = {}
        for port in tool.input_ports(step.settings):
            values: List[PortValue] = []
            for e in (x for x in incoming if x.input == port.name):
                src = self.steps[e.source]
                pv = src.outputs.get(e.output)
                if pv is None or not pv.paths:
                    step.issues.append(Issue(
                        "error" if not port.optional else "warning",
                        f"{port.label}: {src.label} does not produce this file with its current "
                        "settings", (f"input:{port.name}",)))
                    continue
                if not port.accepts(pv.kind):
                    step.issues.append(Issue("error", f"{port.label} cannot take a "
                                                      f"{kind_label(pv.kind)}", (f"input:{port.name}",)))
                    continue
                values.append(pv)
            if len(values) > 1 and not port.multiple:
                step.issues.append(Issue("error", f"{port.label} takes one file but "
                                                  f"{len(values)} steps feed it", (f"input:{port.name}",)))
                values = values[:1]
            if values:
                merged = PortValue([p for v in values for p in v.paths], values[0].kind,
                                   dict(values[0].data), values[0].source, values[0].source_label)
                inputs[port.name] = merged
                if node.files.get(port.name):
                    step.issues.append(Issue("info", f"{port.label}: the connected file is used; the "
                                                     "hand-picked file is ignored", (f"input:{port.name}",)))
            elif node.files.get(port.name):
                paths = [p.strip() for p in node.files[port.name].split(";") if p.strip()]
                paths = [os.path.abspath(os.path.expanduser(p)) for p in paths]
                kind = guess_kind(paths[0]) if paths else port.kind
                if not port.accepts(kind):
                    kind = port.kind
                inputs[port.name] = PortValue(paths, kind, {}, "", "chosen file")
                missing = [p for p in paths if not os.path.exists(p)]
                if missing:
                    step.issues.append(Issue("warning", f"{port.label}: file not found (yet): "
                                                        f"{missing[0]}", (f"input:{port.name}",)))
            elif not port.optional:
                step.issues.append(Issue("error", f"{port.label}: connect a step that produces it or "
                                                  "choose a file", (f"input:{port.name}",)))
        step.ctx = StepContext(self.project, node.step, self.stamp, tool.tag, inputs, self.root,
                               self.workflow.step_folders, step.label, node.id, self.groups)
        if tool.generates and node.step < 1:
            step.issues.append(Issue("error", "step numbers start at 1"))
        try:
            step.issues.extend(tool.check(step.settings, step.ctx))
            step.outputs = tool.output_values(step.settings, step.ctx)
        except Exception as exc:  # a generator bug must not break the planner
            step.issues.append(Issue("error", f"internal error while checking this step: {exc}"))
            step.outputs = {}
        if step.adopted:
            self._use_adopted_run(step)
        for pv in step.outputs.values():
            pv.source = node.id
            pv.source_label = step.label

    def _use_adopted_run(self, step: ResolvedStep) -> None:
        """Point the outputs of a step that adopted an earlier run at that run's files, so later steps
        read (and check) them; the step itself is neither written nor run again."""
        stamp = step.adopted
        if stamp != self.stamp:
            try:
                step.outputs = step.tool.output_values(step.settings, step.ctx.with_stamp(stamp))
            except Exception as exc:  # a generator bug must not break the planner
                step.issues.append(Issue("error", f"internal error while naming the run of {stamp}: {exc}"))
                step.outputs = {}
        own = {p for v in step.ctx.inputs.values() for p in v.paths}  # passed through, not this step's
        missing = [p for v in step.outputs.values() for p in v.paths
                   if p not in own and self.root and not os.path.exists(self.node_path(p))]
        if missing:
            step.issues.append(Issue("warning", f"uses the outputs of its run of {stamp}, but "
                                                f"{missing[0]} is missing" +
                                     (f" (and {len(missing) - 1} more)" if len(missing) > 1 else "") +
                                     "; check previous runs again, or run the step again"))
        else:
            step.issues.append(Issue("info", f"uses the outputs of its run of {stamp}: not written or run "
                                             "again"))

    def _check_steps_and_names(self) -> None:
        seen: Dict[Tuple[int, str], str] = {}
        for nid in self.order:
            step = self.steps[nid]
            if step.tool is None or not step.tool.generates:
                continue
            key = (step.node.step, step.tool.tag)
            if key in seen:
                other = self.steps[seen[key]]
                step.issues.append(Issue("error", f"{step.label} and {other.label} would write into the "
                                                  "same folder with the same names: give one of them "
                                                  "another step number"))
            seen[key] = nid
        self._check_shared_outputs()
        numbers: Dict[int, List[str]] = {}
        for nid in self.order:
            step = self.steps[nid]
            if step.tool is not None and step.tool.generates:
                numbers.setdefault(step.node.step, []).append(step.label)
        for num, labels in numbers.items():
            if len(labels) > 1:
                self.issues.append(Issue("info", f"step number {step_label(num)} is used by "
                                                 f"{', '.join(labels)}"))

    def _check_shared_outputs(self) -> None:
        """Two steps must not write the same output file.  Built-in steps put their number and tag in every
        name, but a custom script's outputs can be given any name, and in a plan without step folders the
        steps after such a pair would read whichever file was written last."""
        claimed: Dict[str, str] = {}  # plan-relative path -> node id of the first step that writes it
        for nid in self.order:
            step = self.steps[nid]
            if step.tool is None or not step.tool.generates or step.ctx is None:
                continue
            # another step's files, and files chosen by hand that the step passes on (a FARFAR2 FASTA file)
            passed = {p for v in step.ctx.inputs.values() for p in v.paths}
            passed |= {str(step.settings.get(key) or "").strip()
                       for key, opt in step.tool.flat_options().items() if opt.kind == "file"}
            for port, value in step.outputs.items():
                for path in value.paths:
                    if os.path.isabs(path) or path in passed or os.path.dirname(path) != step.ctx.folder:
                        continue  # not written by this step
                    first = self.steps[claimed.setdefault(path, nid)]
                    if first is step or (first.node.step, first.tool.tag) == (step.node.step, step.tool.tag):
                        continue  # the same step, or a pair already reported as writing the same names
                    step.issues.append(Issue("error", f"output '{port}': {path} is also written by "
                                                      f"{first.label}; give one of them another file name"))

    # ------------------------------------------------------------ queries
    def all_issues(self) -> List[Tuple[str, Issue]]:
        out = [("", i) for i in self.issues]
        for nid in self.order + [n for n in self.steps if n not in self.order]:
            out += [(nid, i) for i in self.steps[nid].issues]
        return out

    def has_errors(self, node_ids: Optional[Sequence[str]] = None) -> bool:
        if errors(self.issues):
            return True
        ids = node_ids if node_ids is not None else list(self.steps)
        return any(not self.steps[n].ok for n in ids if n in self.steps)

    def node_path(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    # ----------------------------------------------------------- generate
    def generate(self, node_ids: Optional[Sequence[str]] = None) -> Dict[str, List[GeneratedFile]]:
        out: Dict[str, List[GeneratedFile]] = {}
        for nid in self.order:
            if node_ids is not None and nid not in node_ids:
                continue
            step = self.steps[nid]
            if step.tool is None or not step.tool.generates or not step.ok:
                continue
            step.files = generate_step(step)
            out[nid] = step.files
        return out

    def write(self, node_ids: Optional[Sequence[str]] = None, master: bool = True) -> WriteReport:
        """Write the files of every step without errors (all steps by default)."""
        if not self.root:
            raise ValueError("no plan folder set")
        report = WriteReport(self.root)
        targets = [n for n in self.order if node_ids is None or n in node_ids]
        report.kept = {n: self.steps[n].adopted for n in targets if self.steps[n].adopted}
        targets = [n for n in targets if n not in report.kept]  # their earlier files stay as they are
        for nid in targets:
            step = self.steps[nid]
            if step.tool is not None and step.tool.generates and not step.ok:
                report.skipped[nid] = "; ".join(i.message for i in errors(step.issues))
        generated = self.generate(targets)
        paths: Dict[str, str] = {}
        for nid, files in generated.items():
            for f in files:
                rel = self.steps[nid].ctx.from_own(f.name)
                if rel in paths:
                    report.skipped[nid] = f"{rel} is also written by {self.steps[paths[rel]].label}"
                paths.setdefault(rel, nid)
        for nid, files in generated.items():
            if nid in report.skipped:
                continue
            ctx = self.steps[nid].ctx
            for f in files:
                path = os.path.join(self.root, ctx.from_own(f.name))
                _write_text(path, f.content, f.executable)
                report.written.append(path)
        if self.workflow.use_stamp and not self.workflow.stamp:
            self.workflow.stamp = self.stamp
        if master and node_ids is None:
            report.master = os.path.join(self.root, file_name(self.project, 0, "plan", self.stamp, "sh"))
            _write_text(report.master, self.master_script(), True)
            report.summary = os.path.join(self.root, file_name(self.project, 0, "plan", self.stamp, "md"))
            _write_text(report.summary, self.summary_text(), False)
            report.written += [report.master, report.summary]
        return report

    def master_script(self) -> str:
        lines = ["#!/bin/bash",
                 f"# Master script of plan {self.project} - written by {PROJECT_ABBREV} {__version__} "
                 f"on {_dt.date.today().isoformat()}.",
                 "# Runs the main script of every step in plan order.  Scripts marked 'run separately'",
                 "# (e.g. production runs for a cluster) are listed but not started.",
                 "", "set -eo pipefail", 'cd "$(dirname "$0")"', ""]
        for nid in self.order:
            step = self.steps[nid]
            if step.tool is None or not step.tool.generates:
                continue
            lines.append(f"# {step.label}")
            if step.adopted:
                lines += [f"#   uses the outputs of its run of {step.adopted} (not run again)", ""]
                continue
            if not step.files and step.ok:
                step.files = generate_step(step)
            if not step.ok:
                lines.append("#   (not written: the step has errors)")
            for f in step.files:
                rel = step.ctx.from_own(f.name)
                if f.role == "run":
                    lines.append(f'bash "{rel}"')
                elif f.role == "run-separate":
                    lines.append(f"#   run separately: bash {rel}")
            lines.append("")
        return "\n".join(lines)

    def summary_text(self) -> str:
        lines = [f"# Plan {self.project}", "",
                 f"Written by {PROJECT_ABBREV} {__version__} on {_dt.date.today().isoformat()}. "
                 f"File names follow `<project>_<step>_<tool>_<time>.<ext>`"
                 + (f" with time stamp `{self.stamp}`." if self.stamp else " (no time stamp)."), ""]
        for nid in self.order:
            step = self.steps[nid]
            node = step.node
            if step.tool is None:
                continue
            if node.tool == "note":
                text = step.settings.get("text", "").strip()
                if text:
                    lines += [f"> {line}" for line in text.splitlines()] + [""]
                continue
            lines += [f"## {step.label}", ""]
            if step.tool.generates and step.ctx is not None:
                lines.append(f"Folder: `{step.ctx.folder or '.'}`  ")
            lines.append(f"Type: {step.tool.label}")
            lines.append("")
            if step.adopted:
                lines += [f"Uses the outputs of its run of `{step.adopted}` (not written or run again).", ""]
            extra = step.tool.summary_lines(step.settings, step.ctx) if step.ctx is not None else []
            if extra:
                lines += extra + [""]
            if step.ctx and step.ctx.inputs:
                lines.append("Inputs:")
                for port, value in step.ctx.inputs.items():
                    src = f" (from {value.source_label})" if value.source else ""
                    lines.append(f"- {port}: `{'`, `'.join(value.paths)}`{src}")
                lines.append("")
            if step.outputs:
                lines.append("Outputs:")
                for port, value in step.outputs.items():
                    lines.append(f"- {port}: `{'`, `'.join(value.paths)}`")
                lines.append("")
            if step.files:
                lines.append("Files written: " + ", ".join(f"`{f.name}`" for f in step.files))
                lines.append("")
            notes = [i for i in step.issues if i.level != "info"]
            if notes:
                lines.append("Checks:")
                lines += [f"- {i}" for i in notes]
                lines.append("")
        return "\n".join(lines)


def generate_step(step: ResolvedStep) -> List[GeneratedFile]:
    """The step's files, with a 'Please cite' block for the resources it uses in its shell scripts."""
    from ..citations import comment_lines, step_resources

    files = step.tool.generate(step.settings, step.ctx)
    cite = comment_lines(step_resources(step))
    if not cite:
        return files
    out = []
    for f in files:
        lines = f.content.split("\n")
        if f.role in ("run", "run-separate") and lines and lines[0].startswith("#!"):
            k = 1
            while k < len(lines) and lines[k].startswith("#"):
                k += 1  # after the header comment block
            f = dataclasses.replace(f, content="\n".join(lines[:k] + cite + lines[k:]))
        out.append(f)
    return out


def _write_text(path: str, text: str, executable: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    if executable:
        os.chmod(tmp, 0o755)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# stand-alone wizard (one step without a plan)
# ---------------------------------------------------------------------------


def scratch_node(workflow: WorkflowParams, tool: str) -> PlanNode:
    """The stand-alone wizard's settings for ``tool`` (created on first use)."""
    for node in workflow.scratch:
        if node.tool == tool:
            return node
    t = TOOLS[tool]
    node = PlanNode(id=f"wizard-{tool}", tool=tool, step=1, settings=t.normalise({}))
    workflow.scratch.append(node)
    return node


def single_step_plan(workflow: WorkflowParams, node: PlanNode, root: str = "",
                     groups: Optional[Sequence] = None) -> Plan:
    """A one-step plan used by the stand-alone wizard (inputs are hand-picked files)."""
    solo = WorkflowParams(project_name=workflow.project_name, stamp=workflow.stamp,
                          use_stamp=workflow.use_stamp, out_dir=workflow.out_dir,
                          step_folders=workflow.step_folders, nodes=[node], edges=[])
    plan = Plan(solo, root, groups)
    return plan


def analysis_inputs(plan: Plan, node_id: str) -> Tuple[str, List[str]]:
    """Absolute topology / trajectory paths feeding an 'MDATK analysis' step."""
    step = plan.steps[node_id]
    ctx = step.ctx
    if ctx is None:
        return "", []
    top = ctx.value("topology")
    traj = ctx.value("trajectory")
    top_path = plan.node_path(top.path) if top and top.paths else ""
    traj_paths = [plan.node_path(p) for p in traj.paths] if traj else []
    return top_path, traj_paths


# ---------------------------------------------------------------------------
# templates: pre-connected chains of steps
# ---------------------------------------------------------------------------

# name -> (label, [(key, tool, column, row, settings)], [(src key, output, dst key, input)])
TEMPLATES: Dict[str, Tuple[str, List[Tuple[str, str, int, int, Dict[str, Any]]],
                           List[Tuple[str, str, str, str]]]] = {
    "amber": ("Amber RNA protocol: build → helix restraints → MD → analysis", [
        ("build", "amber-build", 0, 0, {}),
        ("restraints", "amber-restraints", 1, 1, {}),
        ("md", "amber-md", 2, 0, {}),
        ("analysis", "mdatk", 3, 0, {}),
    ], [
        ("build", "topology", "restraints", "topology"),
        ("build", "topology", "md", "topology"),
        ("build", "coordinates", "md", "coordinates"),
        ("restraints", "restraints", "md", "restraints"),
        ("md", "topology", "analysis", "topology"),
        ("md", "trajectory", "analysis", "trajectory"),
    ]),
    "farfar2-amber": ("Sequence to MD ensemble: FARFAR2 → models → Amber → analysis + VMD", [
        ("fold", "farfar2", 0, 0, {}),
        ("models", "farfar2-models", 1, 0, {}),
        ("build", "amber-build", 2, 0, {}),
        ("restraints", "amber-restraints", 3, 1, {}),
        ("md", "amber-md", 4, 0, {"dry_traj": True}),
        ("analysis", "mdatk", 5, 0, {}),
        ("vmd", "vmd", 5, 1, {}),
    ], [
        ("fold", "silent", "models", "silent"),
        ("models", "best", "build", "structure"),
        ("build", "topology", "restraints", "topology"),
        ("fold", "secstruct", "restraints", "secstruct"),
        ("build", "topology", "md", "topology"),
        ("build", "coordinates", "md", "coordinates"),
        ("restraints", "restraints", "md", "restraints"),
        ("md", "topology", "analysis", "topology"),
        ("md", "trajectory", "analysis", "trajectory"),
        ("md", "dry_topology", "vmd", "topology"),
        ("md", "dry_trajectory", "vmd", "trajectory"),
    ]),
    "builder-amber": ("Ideal A-form helix → Amber build → helix restraints → MD", [
        ("helix", "structure-builder", 0, 0, {"mode": "helix"}),
        ("build", "amber-build", 1, 0, {}),
        ("restraints", "amber-restraints", 2, 1, {}),
        ("md", "amber-md", 3, 0, {}),
    ], [
        ("helix", "pdb", "build", "structure"),
        ("helix", "secstruct", "restraints", "secstruct"),
        ("build", "topology", "restraints", "topology"),
        ("build", "topology", "md", "topology"),
        ("build", "coordinates", "md", "coordinates"),
        ("restraints", "restraints", "md", "restraints"),
    ]),
    "cryoem-refine": ("Model + map → restraint dictionaries → Phenix RSR → Coot RSR", [
        ("model", "input-file", 0, 0, {"kind": "pdb"}),
        ("map", "input-file", 0, 1, {"kind": "map"}),
        ("dicts", "restraint-dicts", 1, 1, {}),
        ("phenix", "phenix-rsr", 2, 0, {}),
        ("coot", "coot-rsr", 3, 0, {}),
    ], [
        ("model", "file", "dicts", "structure"),
        ("model", "file", "phenix", "model"),
        ("map", "file", "phenix", "map"),
        ("dicts", "dictionaries", "phenix", "dictionaries"),
        ("phenix", "refined", "coot", "model"),
        ("map", "file", "coot", "map"),
        ("dicts", "dictionaries", "coot", "dictionaries"),
    ]),
    "amber-gromacs": ("Amber topology → GROMACS MD", [
        ("build", "amber-build", 0, 0, {}),
        ("gmx", "gromacs", 1, 0, {}),
    ], [
        ("build", "topology", "gmx", "topology"),
        ("build", "coordinates", "gmx", "coordinates"),
    ]),
    "amber-mdff": ("Model + map → Amber build → NAMD MDFF on the Amber topology", [
        ("model", "input-file", 0, 0, {"kind": "pdb"}),
        ("map", "input-file", 0, 1, {"kind": "map"}),
        ("build", "amber-build", 1, 0, {"box": "rect"}),
        ("mdff", "namd-mdff", 2, 0, {}),
        ("note", "note", 2, 1, {"text": (
            "NAMD reads the Amber topology as built (CP, NBfix19, 0BPh, L-RNA, Mg²⁺ pairs, HMR) and an "
            "energy comparison with sander checks it before fitting.\n"
            "Rectangular box: NAMD fits the MDFF potential into a truncated octahedron only with a small margin.\n"
            "The model is the map-frame model: tleap moves the solute, MDFF moves the map with it.\n"
            "Restraints are MDATK's L-RNA-safe ones; never add VMD's chirality / ssrestraints.\n"
            "The fitted model has Amber residue names: 'mdatk pdb clean --scheme ccd' before Phenix / Coot.")}),
    ], [
        ("model", "file", "build", "structure"),
        ("build", "topology", "mdff", "topology"),
        ("build", "coordinates", "mdff", "coordinates"),
        ("map", "file", "mdff", "map"),
        ("model", "file", "mdff", "model"),
    ]),
}

COLUMN_DX = 280
ROW_DY = 190


def apply_template(workflow: WorkflowParams, name: str, x: float = 40.0, y: float = 40.0) -> List[PlanNode]:
    """Add a template's steps (numbered after the existing ones) and connect them."""
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r} (known: {', '.join(TEMPLATES)})")
    _label, steps, edges = TEMPLATES[name]
    made: Dict[str, PlanNode] = {}
    for key, tool, col, row, settings in steps:
        if tool not in TOOLS:
            continue
        made[key] = add_node(workflow, tool, x + col * COLUMN_DX, y + row * ROW_DY, settings=settings)
    for src, out, dst, inp in edges:
        if src in made and dst in made:
            try:
                connect(workflow, made[src].id, out, made[dst].id, inp)
            except (KeyError, ValueError):
                pass
    return list(made.values())
