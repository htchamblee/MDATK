"""Analysis framework: registry, cache keys and the single-pass frame runner.

To add an analysis, subclass :class:`Analysis`, decorate it with
:func:`register` and describe its outputs on the returned
:class:`~mdatk.data.AnalysisResult`.  Panels, charts, caching and the GUI pick
it up automatically.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from ..config import Project, stable_hash
from ..data import AnalysisResult
from ..progress import Reporter
from ..selections import SelectionError, heavy, select

log = logging.getLogger(__name__)

ANALYSES: Dict[str, Type["Analysis"]] = {}

_GROUP_TOKEN = re.compile(r"group:([A-Za-z0-9_.+\-']+)")

#: ``contacts@Aptamer+Mg`` is the contacts analysis run on the groups Aptamer and Mg
INSTANCE_SEP = "@"
GROUP_SEP = "+"


def register(cls: Type["Analysis"]) -> Type["Analysis"]:
    ANALYSES[cls.name] = cls
    return cls


def parse_instance(instance: str) -> Tuple[str, Tuple[str, ...]]:
    """``"contacts@A+B"`` -> ``("contacts", ("A", "B"))``; ``"rmsd"`` -> ``("rmsd", ())``."""
    base, _, groups = instance.partition(INSTANCE_SEP)
    return base, tuple(g for g in groups.split(GROUP_SEP) if g) if groups else ()


def instance_id(base: str, groups: Sequence[str] = ()) -> str:
    groups = [g for g in groups if g]
    return f"{base}{INSTANCE_SEP}{GROUP_SEP.join(groups)}" if groups else base


def split_key(key: str) -> Tuple[str, str]:
    """``"contacts@A+B/atom_pairs"`` -> ``("contacts@A+B", "atom_pairs")``."""
    instance, _, out = key.partition("/")
    return instance, out


def analysis_class(instance: str) -> Type["Analysis"]:
    base, _ = parse_instance(instance)
    if base not in ANALYSES:
        raise KeyError(f"unknown analysis {base!r} (known: {', '.join(ANALYSES)})")
    return ANALYSES[base]


def instance_project(project: Project, instance: str) -> Project:
    """The project an analysis instance runs with (its groups substituted)."""
    cls = analysis_class(instance)
    _, groups = parse_instance(instance)
    if not groups:
        return project
    if cls.group_slots == 0:
        raise ValueError(f"{cls.name} does not take residue groups ({instance!r})")
    if len(groups) != cls.group_slots:
        raise ValueError(f"{instance!r}: {cls.name} takes {cls.group_slots} group(s)")
    known = set(project.group_names())
    missing = [g for g in groups if g not in known]
    if missing:
        raise ValueError(f"{instance!r}: unknown residue group(s) {', '.join(missing)}")
    return cls.with_groups(project.copy(), groups)


def key_label(key: str, project: Project) -> str:
    """Human readable label of a data key, without computing anything."""
    instance, out = split_key(key)
    try:
        cls = analysis_class(instance)
        return cls.output_label(instance_project(project, instance), out)
    except (KeyError, ValueError):
        return key


def referenced_groups(texts: Sequence[str], project: Project) -> List[Dict[str, str]]:
    """Definitions of all residue groups referenced (directly or nested) by ``texts``."""
    by_name = {g.name: g for g in project.groups}
    seen: Dict[str, Dict[str, str]] = {}
    stack = list(texts)
    while stack:
        text = (stack.pop() or "").strip()
        if not text:
            continue
        names = set(_GROUP_TOKEN.findall(text))
        if text in by_name:
            names.add(text)
        for name in names:
            if name in seen or name not in by_name:
                continue
            g = by_name[name]
            seen[name] = {"name": g.name, "residues": g.residues, "selection": g.selection}
            stack.append(g.selection)
    return [seen[k] for k in sorted(seen)]


class AnalysisContext:
    """Shared state handed to every analysis of one run."""

    def __init__(self, universe, project: Project, topology_path: str = "",
                 reporter: Optional[Reporter] = None):
        self.u = universe
        self.project = project
        self.topology_path = topology_path
        self.reporter = reporter or Reporter()
        self.use_pbc = project.analyses.use_pbc

    def select(self, text: str, allow_empty: bool = False, field: str = ""):
        """Atoms of ``text``; a :class:`SelectionError` names ``field``, the setting the text came from
        (e.g. ``"analyses.mg.ligand_selection"``)."""
        try:
            return select(self.u, text, self.project.groups, allow_empty=allow_empty)
        except SelectionError as exc:
            raise exc.locate(field, self.project)

    def group_atoms(self, name: str, heavy_only: bool = False, field: str = ""):
        ag = self.select(f"group:{name}", field=field)
        return heavy(ag) if heavy_only else ag

    def fail(self, message: str, field: str) -> SelectionError:
        """A :class:`SelectionError` of the setting ``field`` to raise (e.g. two groups that overlap)."""
        return SelectionError(message).locate(field, self.project)

    def box(self, ts):
        if not self.use_pbc:
            return None
        return ts.dimensions

    def warn(self, message: str) -> None:
        log.warning(message)
        self.reporter.raw_log(f"WARNING: {message}")


class Analysis:
    """Base class for per-frame streaming analyses."""

    name = "base"
    label = "Base analysis"
    version = 1
    #: internal analyses (trace / proximity) feed panels but are not user-selectable
    internal = False
    #: number of residue groups a data key may choose (0, 1 or 2)
    group_slots = 0
    #: the settings (under ``analyses.<name>``) that :meth:`with_groups` rewrites for a run on other groups
    instance_settings: Tuple[str, ...] = ("group_a", "group_b")
    #: output name -> label; ``{a}`` / ``{b}`` are replaced by the group names
    OUTPUT_LABELS: Dict[str, str] = {}

    def __init__(self, ctx: AnalysisContext, instance: Optional[str] = None):
        self.ctx = ctx
        self.project = ctx.project
        self.instance = instance or self.name
        self.n_frames = 0

    # ---------------------------------------------------------- class level
    @classmethod
    def enabled(cls, project: Project) -> bool:
        return False

    @classmethod
    def params_for(cls, project: Project) -> Dict[str, Any]:
        """Parameters that influence the result (used for the cache key)."""
        return {}

    @classmethod
    def selection_texts(cls, project: Project) -> List[str]:
        """Selections / group names this analysis uses (for the cache key)."""
        return []

    @classmethod
    def expected_outputs(cls, project: Project) -> Dict[str, List[str]]:
        """Output names this analysis will produce (for GUI choices, before running)."""
        return {"series": [], "matrices": [], "pairmaps": []}

    @classmethod
    def groups(cls, project: Project) -> Tuple[str, ...]:
        """The residue groups the configured (default) analysis uses."""
        params = getattr(project.analyses, cls.name, None)
        if cls.group_slots == 2 and params is not None:
            return params.group_a, params.group_b
        return ()

    @classmethod
    def with_groups(cls, project: Project, groups: Sequence[str]) -> Project:
        """Return ``project`` (a private copy) set up to run on ``groups``."""
        params = getattr(project.analyses, cls.name)
        params.group_a, params.group_b = groups
        return project

    @classmethod
    def output_label(cls, project: Project, out: str) -> str:
        a, b = (list(cls.groups(project)) + ["A", "B"])[:2]
        template = cls.OUTPUT_LABELS.get(out)
        return template.format(a=a, b=b) if template else f"{cls.label}: {out}"

    @classmethod
    def cache_key(cls, project: Project, inputs: Dict[str, Any]) -> str:
        return stable_hash({
            "analysis": cls.name,
            "version": cls.version,
            "params": cls.params_for(project),
            "groups": referenced_groups(cls.selection_texts(project), project),
            "inputs": inputs,
            "use_pbc": project.analyses.use_pbc,
        })

    # ------------------------------------------------------- instance level
    def setup(self, frame_indices: np.ndarray) -> None:
        self.n_frames = len(frame_indices)

    def process(self, k: int, ts) -> None:
        raise NotImplementedError

    def finalize(self) -> AnalysisResult:
        raise NotImplementedError

    def new_result(self, key: str = "", **meta) -> AnalysisResult:
        project = self.project
        return AnalysisResult(name=self.instance, key=key, params=self.params_for(project),
                              arrays={}, meta=dict(meta, label=self.label, analysis=self.name))

    def field(self, name: str) -> str:
        """Project path of one of this analysis' settings: ``"group_a"`` -> ``"analyses.hbonds.group_a"``."""
        return f"analyses.{self.name}.{name}"

    def locate_error(self, exc: SelectionError) -> SelectionError:
        """A run on other groups (``contacts@A+B``, ``rmsd@A``) got its groups from a panel or a chart
        series: report a problem of a setting :meth:`with_groups` rewrote there (or, when neither asked
        for the run, at the definition of the group concerned), never at the analysis' own settings,
        which the run did not use."""
        _, groups = parse_instance(self.instance)
        own = [self.field(name) for name in self.instance_settings]
        if not groups or not any(exc.field == f or exc.field.startswith(f + ".") for f in own):
            return exc
        project = self.project
        for k, spec in enumerate(project.layout.panels):
            for j, key in enumerate(spec.series):
                if split_key(key)[0] == self.instance:
                    return exc.locate(f"layout.panels.{k}.{'groups' if j == 0 else 'series'}", project, True)
        if any(split_key(key)[0] == self.instance for key in project.output.charts.series):
            return exc.locate("output.charts.series", project, True)
        name = groups[1] if exc.field == self.field("group_b") and len(groups) > 1 else groups[0]
        for k, g in enumerate(project.groups):
            if g.name == name:
                return exc.locate(f"groups.{k}.{'selection' if g.selection.strip() else 'residues'}", project,
                                  True)
        return exc


def run_analyses(ctx: AnalysisContext, analyses: Sequence[Analysis], frame_indices: np.ndarray,
                 reporter: Optional[Reporter] = None):
    """Stream the selected frames once, feeding every analysis.

    Returns ``(results, time_ps)``; results are in the order of ``analyses``.
    """
    reporter = reporter or ctx.reporter
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    n = len(frame_indices)
    for a in analyses:
        reporter.log(f"Preparing {a.label} ...")
        try:
            a.setup(frame_indices)
        except SelectionError as exc:  # name the setting the failing selection came from
            raise a.locate_error(exc)
    times = np.empty(n, dtype=np.float64)
    traj = ctx.u.trajectory
    every = max(1, n // 200)
    reporter.log(f"Analysing {n} frame(s): {', '.join(a.label for a in analyses)}")
    for k, ts in enumerate(traj[frame_indices]):
        reporter.check_cancel()
        times[k] = ts.time
        for a in analyses:
            a.process(k, ts)
        if k % every == 0 or k == n - 1:
            reporter.progress((k + 1) / n, f"Analysing frame {k + 1}/{n}")
    return [a.finalize() for a in analyses], times
