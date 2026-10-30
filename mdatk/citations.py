"""Citations of the programs, force fields and parameter sets a project uses.

Only what a project actually uses is cited.  Each planner step reports the
resources its settings turn on (:meth:`mdatk.workflows.base.Tool.resources`),
the analysis and video pipeline reports its own (:func:`pipeline_resources`),
and :data:`RESOURCES` maps every resource to the references that describe it.
A generated script cites its own step; a project (About dialog, BibTeX export,
parameter log) cites every step plus the pipeline.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ._citation_data import REFERENCE_DATA, RESOURCE_DATA


@dataclass(frozen=True)
class Reference:
    key: str  # BibTeX key
    text: str  # one-line citation: authors, journal year, volume, pages, doi
    bibtex: str
    doi: str = ""
    url: str = ""
    verified: str = ""  # where the details were checked: pdf:<file>, repo:<url>, pypi:<package>


@dataclass(frozen=True)
class Resource:
    id: str
    name: str
    keys: Tuple[str, ...]  # references to cite (in this order)
    note: str = ""  # e.g. "cite together with ..." or how to cite software without a paper


REFERENCES: Dict[str, Reference] = {r["key"]: Reference(**r) for r in REFERENCE_DATA}
RESOURCES: Dict[str, Resource] = {r["id"]: Resource(r["id"], r["name"], tuple(r["keys"]), r.get("note", ""))
                                  for r in RESOURCE_DATA}


def unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def keys_for(resource_ids: Iterable[str]) -> List[str]:
    """Reference keys of ``resource_ids`` in first-use order (unknown ids are ignored)."""
    keys: List[str] = []
    for rid in resource_ids:
        res = RESOURCES.get(rid)
        if res is not None:
            keys += res.keys
    return unique(k for k in keys if k in REFERENCES)


def step_resources(step) -> List[str]:
    """Resources one resolved planner step uses (its tool decides from its settings)."""
    if step.tool is None:
        return []
    by_context = getattr(step.tool, "step_resources", None)  # tools whose resources depend on their inputs
    if by_context is not None and getattr(step, "ctx", None) is not None:
        return unique(by_context(step.settings, step.ctx))
    return unique(step.tool.resources(step.settings))


def step_citations(step) -> List[str]:
    return keys_for(step_resources(step))


def plan_resources(plan) -> List[str]:
    out: List[str] = []
    for nid in plan.order:
        step = plan.steps[nid]
        if step.tool is not None and step.tool.generates:
            out += step_resources(step)
    return unique(out)


def pipeline_resources(project) -> List[str]:
    """Resources of MDATK's own analysis, rendering and video stages for ``project``."""
    out = ["mdanalysis", "numpy", "scipy", "matplotlib"]
    if project.cleaning.enabled:
        out.append("cpptraj")
    if project.output.video.enabled:
        out.append("ffmpeg")
    if getattr(project.visual, "renderer", "") == "vtk":
        out.append("vtk")
    if project.analyses.basepairs.engine == "barnaba" and any(
            s.type == "secondary" and (s.options or {}).get("source", "dynamic") != "dotbracket"
            for s in project.layout.panels):
        out.append("barnaba")
    return unique(out)


def project_resources(project, plan=None) -> List[str]:
    """Everything ``project`` uses: its planner steps (when it has a plan) and the pipeline."""
    out: List[str] = []
    if plan is None and project.workflow.nodes:
        try:
            from .workflows.plan import Plan
            plan = Plan(project.workflow, project.workflow_dir, project.groups)
        except (ValueError, KeyError, OSError):
            plan = None
    if plan is not None:
        out += plan_resources(plan)
    out += pipeline_resources(project)
    return unique(out)


def references(keys: Iterable[str]) -> List[Reference]:
    return [REFERENCES[k] for k in unique(keys) if k in REFERENCES]


def bibtex(keys: Iterable[str]) -> str:
    return "\n\n".join(r.bibtex.strip() for r in references(keys)) + "\n"


def text_lines(keys: Iterable[str]) -> List[str]:
    return [r.text for r in references(keys)]


def grouped(resource_ids: Sequence[str]) -> List[Tuple[Resource, List[Reference]]]:
    """(resource, its references) for display, in first-use order."""
    return [(RESOURCES[rid], references(RESOURCES[rid].keys)) for rid in unique(resource_ids)
            if rid in RESOURCES]


def comment_lines(resource_ids: Sequence[str], prefix: str = "# ", width: int = 100,
                  heading: Optional[str] = "Please cite") -> List[str]:
    """Citation block for a script header: one wrapped entry per reference."""
    refs = references(keys_for(resource_ids))
    if not refs:
        return []
    names = ", ".join(RESOURCES[r].name for r in unique(resource_ids) if r in RESOURCES)
    lines = [f"{prefix}{w}" for w in textwrap.wrap(f"{heading} ({names}):", width - len(prefix))] if heading else []
    for ref in refs:
        wrapped = textwrap.wrap(ref.text, width - len(prefix) - 4) or [ref.text]
        lines.append(f"{prefix}  - {wrapped[0]}")
        lines += [f"{prefix}    {w}" for w in wrapped[1:]]
    return lines
