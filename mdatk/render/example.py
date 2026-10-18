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
    coords, chain, gidx, resid, links, letters = [], [], [], [], [], []
    group_pos = {g.name: k for k, g in enumerate(project.groups)}
    for s, name in enumerate(names[:2]):
        ids = group_resids(project, name)
        seq = ("GCAUGGCAUCGA" * 10)[:len(ids)]
        if s:  # the complementary strand, read 5' -> 3'
            seq = "".join({"G": "C", "C": "G", "A": "U", "U": "A"}[b] for b in reversed(seq))
        for k, r in enumerate(ids):
            level = k if s == 0 else len(ids) - 1 - k
            ang = np.radians(level * 32.7 + (0 if s == 0 else 150))
            coords.append([9.0 * np.cos(ang), 9.0 * np.sin(ang), 2.8 * level])
            chain.append(s)
            gidx.append(group_pos.get(name, -1))
            resid.append(r)
            letters.append(seq[k])
            if k:
                links.append([len(coords) - 2, len(coords) - 1])
    xyz = np.asarray(coords, dtype=np.float64)
    return {"coords": np.repeat(xyz[None], n_frames, axis=0), "chain": np.asarray(chain),
            "resname": np.asarray(letters, dtype="U8"),
            "group_index": np.asarray(gidx), "resid": np.asarray(resid),
            "links": np.asarray(links, dtype=np.int64).reshape(-1, 2),
            "ions": np.zeros((0, 3))}


def schematic_structure(project: Project) -> Dict[str, object]:
    """Two-strand duplex with heavy atoms (the demo's nucleotide geometry) for the first two
    residue groups: a ``trace`` part (C4' per residue) and an ``atoms`` part, so 3D
    representations can be previewed before any real frame is loaded."""
    from ..demo import COMPLEMENT, DYAD, _level_frame, nucleotide

    names = project.visual.structure_groups or project.group_names()[:2] or ["A"]
    group_pos = {g.name: k for k, g in enumerate(project.groups)}
    ids_a = group_resids(project, names[0])[:40]
    ids_b = group_resids(project, names[1])[:len(ids_a)] if len(names) > 1 else []
    seq = ("GCAUGGCAUCGA" * 4)[:len(ids_a)]
    names_out, elements, resnames, row, xyz, bonds = [], [], [], [], [], []
    trace, chain, gidx, resid, links, letters_out = [], [], [], [], [], []
    prev_o3 = None
    strands = [(0, names[0], ids_a, seq)]
    if ids_b:
        strands.append((1, names[1], ids_b, "".join(COMPLEMENT[b] for b in reversed(seq))[:len(ids_b)]))
    for strand, gname, ids, letters in strands:
        prev_o3 = None
        for m, (r, base) in enumerate(zip(ids, letters)):
            level = m if strand == 0 else len(ids_a) - 1 - m
            rot, shift = _level_frame(level)
            nt, nb = nucleotide(base)
            keep = [k for k, a in enumerate(nt) if a["el"] != "H"]
            local = {k: len(xyz) + j for j, k in enumerate(keep)}
            rr = len(trace)
            for k in keep:
                a = nt[k]
                v = a["xyz"] if strand == 0 else DYAD @ a["xyz"]
                xyz.append(rot @ v + shift)
                names_out.append(a["name"])
                elements.append(a["el"])
                resnames.append(base)
                row.append(rr)
            bonds += [(local[i], local[j]) for i, j in nb if i in local and j in local]
            idx = {nt[k]["name"]: local[k] for k in keep}
            if prev_o3 is not None and "P" in idx:
                bonds.append((prev_o3, idx["P"]))
            prev_o3 = idx.get("O3'")
            trace.append(xyz[idx["C4'"]])
            letters_out.append(base)
            chain.append(strand)
            gidx.append(group_pos.get(gname, -1))
            resid.append(r)
            if m:
                links.append([rr - 1, rr])
    return {"trace": {"coords": np.asarray(trace), "chain": np.asarray(chain), "group_index": np.asarray(gidx),
                      "resname": np.asarray(letters_out, dtype="U8"),
                      "resid": np.asarray(resid), "links": np.asarray(links, dtype=np.int64).reshape(-1, 2),
                      "ions": np.zeros((0, 3))},
            "atoms": {"coords": np.asarray(xyz), "names": names_out, "elements": elements,
                      "resnames": resnames, "row": np.asarray(row, dtype=np.int64),
                      "bonds": np.asarray(bonds, dtype=np.int64).reshape(-1, 2)}}


def _membership(src: Dict[str, object], n_groups: int) -> np.ndarray:
    if "group_member" in src:
        return np.asarray(src["group_member"], dtype=bool)
    gi = np.asarray(src["group_index"])
    return np.array([gi == k for k in range(n_groups)], dtype=bool).reshape(n_groups, len(gi))


def _atoms_result(src: Dict[str, object], n_frames: int, n_res: int) -> AnalysisResult:
    xyz = np.asarray(src["coords"], dtype=np.float32)
    coords = np.broadcast_to(xyz[None], (n_frames,) + xyz.shape) if xyz.ndim == 2 else xyz
    res = AnalysisResult("atoms", "example", {}, {}, {"example": True, "atom_names": list(src["names"]),
                                                     "elements": list(src["elements"]),
                                                     "resnames": list(src["resnames"])})
    res.arrays.update(coords=coords, row=np.asarray(src["row"]), bonds=np.asarray(src["bonds"]),
                      resindex=np.arange(n_res))
    return res


def _example_basepairs(chain: np.ndarray, n_frames: int) -> AnalysisResult:
    """Duplex pairs between the first and the last strand (residue k with its partner), the frayed end
    opening and closing every few frames, so arc diagrams can be previewed."""
    a = np.flatnonzero(chain == chain[0]) if len(chain) else np.zeros(0, dtype=np.int64)
    b = np.flatnonzero(chain != chain[0])[::-1] if len(chain) else np.zeros(0, dtype=np.int64)
    m = min(len(a), len(b))
    builder = CSRBuilder()
    for f in range(n_frames):
        open_end = 2 if (f // 6) % 2 else 0
        i, j = a[:m - open_end], b[:m - open_end]
        builder.add(i, j, np.ones(len(i)))
    res = AnalysisResult("basepairs", "example", {}, {}, {"example": True})
    res.arrays.update(builder.arrays("map"))
    labels = [str(k) for k in range(len(chain))]
    res.meta["map_r"] = labels
    res.meta["map_c"] = labels
    res.add_pairmap("map", "map", "map_r", "map_c", "Base pairs")
    return res


def _frame_basepairs(project: Project, src, atoms_src, n_frames: int) -> Optional[AnalysisResult]:
    """Builtin base pairs of a real preview frame (the same in every example frame)."""
    from ..secstruct import PairDetector

    n_res = len(src["resname"])
    index: List[Dict[str, int]] = [{} for _ in range(n_res)]
    for k, (name, row) in enumerate(zip(atoms_src["names"], np.asarray(atoms_src["row"]))):
        if 0 <= int(row) < n_res:
            index[int(row)].setdefault(str(name).replace("*", "'"), k)
    bp = project.analyses.basepairs
    try:
        found = PairDetector([str(r) for r in src["resname"]], index, cutoff=bp.cutoff,
                             wobble=bp.wobble).detect(np.asarray(atoms_src["coords"], dtype=np.float64))
    except (ValueError, IndexError):
        return None
    builder = CSRBuilder()
    for _ in range(n_frames):
        builder.add(found[:, 0], found[:, 1], found[:, 2])
    res = AnalysisResult("basepairs", "example", {}, {}, {"example": True})
    res.arrays.update(builder.arrays("map"))
    labels = [str(k) for k in range(n_res)]
    res.meta["map_r"] = labels
    res.meta["map_c"] = labels
    res.add_pairmap("map", "map", "map_r", "map_c", "Base pairs")
    return res


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
    if any(spec.type in ("structure", "sequence", "secondary") for spec in project.layout.panels):
        atoms_src = None
        if preview is not None:
            src = preview
            atoms_src = preview.get("atoms")
        elif project.visual.representations:
            schematic = schematic_structure(project)
            src, atoms_src = schematic["trace"], schematic["atoms"]
        else:
            src = schematic_trace(project, n_frames)
        xyz = np.asarray(src["coords"], dtype=np.float64)
        coords = xyz if xyz.ndim == 3 else np.repeat(xyz[None], n_frames, axis=0)
        ions = np.asarray(src.get("ions", np.zeros((0, 3))), dtype=np.float64)
        ions = ions if ions.ndim == 3 else np.repeat(ions.reshape(1, -1, 3), n_frames, axis=0)
        n_res = coords.shape[1]
        trace = AnalysisResult("trace", "example", {}, {}, {"example": True, "schematic": preview is None})
        trace.arrays.update(coords=coords, ion_coords=ions, chain=np.asarray(src["chain"]),
                            group_index=np.asarray(src["group_index"]),
                            group_member=_membership(src, len(project.groups)),
                            is_point=np.ones(n_res, bool), links=np.asarray(src["links"]),
                            resindex=np.arange(n_res), resid=np.asarray(src["resid"]))
        chain = np.asarray(src["chain"], dtype=np.int64)
        if "resname" in src:
            trace.arrays["resname"] = np.asarray(src["resname"])
            trace.arrays["chain_name"] = (np.asarray(src["chain_name"]) if "chain_name" in src else
                                          np.array([chr(ord("A") + int(c) % 26) for c in chain]))
        trace.meta["residue_labels"] = list(src.get("labels", []))
        ds.add(trace)
        pairs = None
        if preview is not None and atoms_src is not None and "resname" in src:
            pairs = _frame_basepairs(project, src, atoms_src, n_frames)
        ds.add(pairs if pairs is not None else _example_basepairs(chain, n_frames))
        if atoms_src is not None:
            ds.add(_atoms_result(atoms_src, n_frames, n_res))
        # proximity: every residue drifts in and out of contact so glow modes are visible
        k = np.arange(n_res)[:, None]
        dist = 5.0 + 4.0 * np.sin(k / 3.0 + np.arange(n_frames)[None, :] / 9.0)
        prox = AnalysisResult("proximity", "example", {}, {}, {"example": True})
        prox.arrays.update(min_distance=dist.astype(np.float32), member=np.zeros(n_res, bool),
                           resindex=np.arange(n_res))
        ds.add(prox)
    return ds
