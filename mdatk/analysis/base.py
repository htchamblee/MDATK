"""Analysis framework: registry, cache keys and the single-pass frame runner.

To add an analysis, subclass :class:`Analysis`, decorate it with
:func:`register` and describe its outputs on the returned
:class:`~mdatk.data.AnalysisResult`.  Panels, charts, caching and the GUI pick
it up automatically.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from ..config import Project, stable_hash
from ..data import AnalysisResult
from ..progress import Reporter
from ..selections import heavy, select

log = logging.getLogger(__name__)

ANALYSES: Dict[str, Type["Analysis"]] = {}

_GROUP_TOKEN = re.compile(r"group:([A-Za-z0-9_.+\-']+)")


def register(cls: Type["Analysis"]) -> Type["Analysis"]:
    ANALYSES[cls.name] = cls
    return cls


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

    def select(self, text: str, allow_empty: bool = False):
        return select(self.u, text, self.project.groups, allow_empty=allow_empty)

    def group_atoms(self, name: str, heavy_only: bool = False):
        ag = self.select(f"group:{name}")
        return heavy(ag) if heavy_only else ag

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

    def __init__(self, ctx: AnalysisContext):
        self.ctx = ctx
        self.project = ctx.project
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
        return AnalysisResult(name=self.name, key=key, params=self.params_for(project),
                              arrays={}, meta=dict(meta, label=self.label))


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
        a.setup(frame_indices)
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
