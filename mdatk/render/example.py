"""Placeholder data for laying out the figure before anything is computed.

The GUI's WYSIWYG canvas renders the real panels (same code as the video)
with this example dataset: smooth sine-like traces, residue heatmaps and maps
sized from the residue groups, and either the real reference-frame trace (when
one has been loaded) or a schematic duplex.  Nothing touches a trajectory.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..analysis.base import analysis_class, instance_project, key_label, parse_instance, split_key
from ..config import Project
from ..data import AnalysisResult, CSRBuilder, Dataset
from ..selections import parse_residue_ranges
from .panels import PANEL_TYPES

N_FRAMES = 120


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(key.encode("utf-8")))


def group_resids(project: Project, name: str, fallback: int = 12) -> List[int]:
    for g in project.groups:
        if g.name == name and g.residues.strip():
            try:
                out: List[int] = []
                for _chain, a, b in parse_residue_ranges(g.residues):
                    out.extend(range(a, b + 1))
                return out[:400] or list(range(1, fallback + 1))
            except ValueError:
                break
    return list(range(1, fallback + 1))


def _unit(base: str, out: str) -> str:
    if base == "rmsd":
        return "Å"
    if base == "energy":
        return "kcal/mol/Å" if out == "net_force" else "kcal/mol"
    if out == "q_native":
        return ""
    return "count"


def _series_values(key: str, unit: str, t: np.ndarray) -> np.ndarray:
    rng = _rng(key)
    phase = rng.uniform(0, 2 * np.pi)
    wave = np.sin(2 * np.pi * t / t[-1] * rng.uniform(0.8, 2.0) + phase)
    if unit == "count":
        base = rng.integers(8, 30)
        return np.round(base + 0.35 * base * wave + rng.normal(0, 1.0, len(t))).clip(0)
    if unit == "Å":
        return 1.2 + 0.6 * (1 + wave) / 2 + rng.normal(0, 0.05, len(t))
    if unit.startswith("kcal"):
        return -120 + 35 * wave + rng.normal(0, 4, len(t))
    return 0.8 + 0.15 * wave


def _instance_groups(project: Project, instance: str) -> Sequence[str]:
    try:
        cls = analysis_class(instance)
        return cls.groups(instance_project(project, instance)) or tuple(
            g.name for g in project.groups[:2])
    except (KeyError, ValueError):
        return tuple(g.name for g in project.groups[:2])


def _result(ds: Dataset, instance: str) -> AnalysisResult:
    res = ds.results.get(instance)
    if res is None:
        res = AnalysisResult(instance, "example", {}, {}, {"example": True})
        ds.add(res)
    return res


def _add_key(ds: Dataset, project: Project, key: str, kind: str) -> None:
    if "/" not in key:
        return
    instance, out = split_key(key)
    base, _ = parse_instance(instance)
    res = _result(ds, instance)
    t = ds.time_ps
    label = key_label(key, project)
    arr = f"ex_{len(res.arrays)}"
    groups = list(_instance_groups(project, instance)) + ["A", "B"]
    if kind == "series":
        unit = _unit(base, out)
        res.arrays[arr] = _series_values(key, unit, t)
        res.add_series(out, arr, label, unit)
    elif kind == "matrices":
        if out.endswith("_b"):
            rows = group_resids(project, groups[1])
        elif out == "occupancy":
            rows = group_resids(project, groups[0]) + group_resids(project, groups[1])
        else:
            rows = group_resids(project, groups[0])
        res.meta[arr + "_rows"] = [str(r) for r in rows]
        k = np.arange(len(rows))[:, None]
        wave = np.sin(k / 2.5 + 2 * np.pi * t[None, :] / t[-1])
        if out == "occupancy":
            res.arrays[arr] = np.digitize(wave, [0.3, 0.8]).astype(np.int8)
            res.add_matrix(out, arr, arr + "_rows", label, "class",
                           classes=["none", "outer sphere", "inner sphere"])
        else:
            res.arrays[arr] = np.round(np.clip(4 + 4 * wave, 0, None))
            res.add_matrix(out, arr, arr + "_rows", label, "count")
    elif kind == "pairmaps":
        rows, cols = group_resids(project, groups[0]), group_resids(project, groups[1])
        b = CSRBuilder()
        n, m = len(rows), len(cols)
        for f in range(len(t)):
            i = np.arange(n)
            j = (m - 1 - i * m // max(n, 1)) % max(m, 1)
            keep = np.sin(i / 2.0 + f / 8.0) > -0.3
            b.add(i[keep], j[keep], np.ones(int(keep.sum())) * 3)
        res.arrays.update(b.arrays(arr))
        res.meta[arr + "_r"] = [str(r) for r in rows]
        res.meta[arr + "_c"] = [str(c) for c in cols]
        res.add_pairmap(out, arr, arr + "_r", arr + "_c", label)


def schematic_trace(project: Project, n_frames: int) -> Dict[str, np.ndarray]:
    """Two-strand helical trace built from the first two residue groups."""
    names = project.visual.structure_groups or project.group_names()[:2] or ["A"]
    coords, chain, gidx, resid, links = [], [], [], [], []
    group_pos = {g.name: k for k, g in enumerate(project.groups)}
    for s, name in enumerate(names[:2]):
        ids = group_resids(project, name)
        for k, r in enumerate(ids):
            level = k if s == 0 else len(ids) - 1 - k
            ang = np.radians(level * 32.7 + (0 if s == 0 else 150))
            coords.append([9.0 * np.cos(ang), 9.0 * np.sin(ang), 2.8 * level])
            chain.append(s)
            gidx.append(group_pos.get(name, -1))
            resid.append(r)
            if k:
                links.append([len(coords) - 2, len(coords) - 1])
    xyz = np.asarray(coords, dtype=np.float64)
    return {"coords": np.repeat(xyz[None], n_frames, axis=0), "chain": np.asarray(chain),
            "group_index": np.asarray(gidx), "resid": np.asarray(resid),
            "links": np.asarray(links, dtype=np.int64).reshape(-1, 2),
            "ions": np.zeros((0, 3))}


def example_dataset(project: Project, preview: Optional[Dict[str, np.ndarray]] = None,
                    n_frames: int = N_FRAMES) -> Dataset:
    """Dataset with placeholder values for every data key the layout uses."""
    dt = project.inputs.dt_ps or 10.0
    ds = Dataset(np.arange(n_frames), np.arange(n_frames) * dt, time_unit=project.inputs.time_unit)
    for spec in project.layout.panels:
        cls = PANEL_TYPES.get(spec.type)
        if cls is None:
            continue
        for key in spec.series:
            _add_key(ds, project, key, cls.data_kind)
    if any(spec.type == "structure" for spec in project.layout.panels):
        src = preview if preview is not None else schematic_trace(project, n_frames)
        xyz = np.asarray(src["coords"], dtype=np.float64)
        coords = xyz if xyz.ndim == 3 else np.repeat(xyz[None], n_frames, axis=0)
        ions = np.asarray(src.get("ions", np.zeros((0, 3))), dtype=np.float64)
        ions = ions if ions.ndim == 3 else np.repeat(ions.reshape(1, -1, 3), n_frames, axis=0)
        n_res = coords.shape[1]
        trace = AnalysisResult("trace", "example", {}, {}, {"example": True, "schematic": preview is None})
        trace.arrays.update(coords=coords, ion_coords=ions, chain=np.asarray(src["chain"]),
                            group_index=np.asarray(src["group_index"]),
                            is_point=np.ones(n_res, bool), links=np.asarray(src["links"]),
                            resindex=np.arange(n_res), resid=np.asarray(src["resid"]))
        ds.add(trace)
        # proximity: every residue drifts in and out of contact so glow modes are visible
        k = np.arange(n_res)[:, None]
        dist = 5.0 + 4.0 * np.sin(k / 3.0 + np.arange(n_frames)[None, :] / 9.0)
        prox = AnalysisResult("proximity", "example", {}, {}, {"example": True})
        prox.arrays.update(min_distance=dist.astype(np.float32), member=np.zeros(n_res, bool),
                           resindex=np.arange(n_res))
        ds.add(prox)
    return ds
