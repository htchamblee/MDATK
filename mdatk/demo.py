"""Synthetic demonstration system for MDATK.

Builds an idealised RNA "aptamer / target" duplex with Mg2+ ions, Na+
counter-ions and a water box, and writes it as a genuine Amber topology
(``.parm7``) plus NetCDF trajectory (``.nc``) so the full workflow - cpptraj
stripping, analyses, video, charts and PDB export - can be exercised without
real simulation data.  The trajectory contains thermal noise, a slow rigid
body drift (so fitting matters), a transient fraying of one duplex end and Mg
ions that bind, bridge and leave phosphates.

The geometry is schematic (not a force-field minimised structure); it is
meant for demonstrations and tests only.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np

from . import PROJECT_ABBREV
from .config import PanelSpec, Project
from .io.parm7 import AMBER_CHARGE_FACTOR, write_parm7

RISE = 3.0
TWIST = np.radians(32.7)
X_PYR = 4.046  # pyrimidine ring centre distance from the pair axis
X_PUR = 1.634  # purine six-ring centre distance
RING_R = 1.39

# (r*, epsilon) per atom type
LJ_TYPES: Dict[str, Tuple[float, float]] = {
    "P": (2.1, 0.2), "O2": (1.6612, 0.21), "OS": (1.6837, 0.17), "OH": (1.721, 0.2104),
    "O": (1.6612, 0.21), "CT": (1.908, 0.1094), "CA": (1.908, 0.086), "N": (1.824, 0.17),
    "H1": (1.387, 0.0157), "HA": (1.359, 0.015), "H": (0.6, 0.0157), "HO": (0.2, 0.03),
    "MG": (0.7926, 0.8947), "Na+": (1.369, 0.0874), "OW": (1.7683, 0.152), "HW": (0.0, 0.0),
}
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "MG": 12, "NA": 11}
MASSES = {"H": 1.008, "C": 12.01, "N": 14.01, "O": 16.0, "P": 30.97, "MG": 24.305, "NA": 22.99}

# sugar / phosphate offsets from C1' in the base-pair frame (strand A side)
SUGAR = {
    "O4'": (-0.78, 1.08, 0.35), "C2'": (-0.58, -1.07, 0.85), "O2'": (-0.13, -2.37, 0.55),
    "C3'": (-1.93, -0.52, 1.25), "C4'": (-2.08, 0.93, 0.85), "O3'": (-2.83, -1.37, 2.00),
    "C5'": (-3.28, 1.93, 0.40), "O5'": (-4.18, 0.93, 0.95), "P": (-4.48, 2.03, -0.35),
    "OP1": (-5.53, 2.78, -0.95), "OP2": (-5.18, 0.98, -1.15),
}
SUGAR_TYPES = {"O4'": "OS", "C2'": "CT", "O2'": "OH", "C3'": "CT", "C4'": "CT", "O3'": "OS",
               "C5'": "CT", "O5'": "OS", "P": "P", "OP1": "O2", "OP2": "O2", "C1'": "CT"}
SUGAR_CHARGES = {"P": 1.166, "OP1": -0.776, "OP2": -0.776, "O5'": -0.499, "C5'": 0.056,
                 "C4'": 0.107, "O4'": -0.355, "C1'": 0.039, "C3'": 0.202, "C2'": 0.067,
                 "O2'": -0.614, "O3'": -0.525}
# hydrogens on the sugar: name -> (parent, direction hint in the pair frame)
SUGAR_H = {"H1'": ("C1'", (0.0, -0.3, -1.0)), "H2'": ("C2'", (0.2, -0.4, 1.0)),
           "HO2'": ("O2'", (0.8, -0.5, -0.3)), "H3'": ("C3'", (-0.2, 0.0, 1.0)),
           "H4'": ("C4'", (0.0, 0.6, 1.0)), "H5'": ("C5'", (-0.3, 1.0, -0.6)),
           "H5''": ("C5'", (-0.6, 0.6, 1.0))}


def _polar(center, radius, deg):
    a = np.radians(deg)
    return np.array([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _rot2(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def base_template(base: str) -> List[Tuple[str, np.ndarray, str, str, float]]:
    """Base atoms on the strand-A side: (name, xy, element, type, charge)."""
    atoms: List[Tuple[str, np.ndarray, str, str, float]] = []
    if base in ("C", "U"):
        c = np.array([-X_PYR, 0.0])
        ring = {"N3": 0, "C4": 60, "C5": 120, "C6": 180, "N1": 240, "C2": 300}
        pos = {n: _polar(c, RING_R, d) for n, d in ring.items()}
        for n in ("N1", "C2", "N3", "C4", "C5", "C6"):
            atoms.append((n, pos[n], n[0], "N" if n[0] == "N" else "CA",
                          -0.45 if n[0] == "N" else 0.15))
        atoms.append(("O2", _polar(c, RING_R + 1.23, 300), "O", "O", -0.58))
        atoms.append(("H5", _polar(c, RING_R + 1.08, 120), "H", "HA", 0.15))
        atoms.append(("H6", _polar(c, RING_R + 1.08, 180), "H", "HA", 0.15))
        if base == "U":
            atoms.append(("O4", _polar(c, RING_R + 1.23, 60), "O", "O", -0.58))
            atoms.append(("H3", pos["N3"] + np.array([1.01, 0.0]), "H", "H", 0.40))
        else:
            n4 = _polar(c, RING_R + 1.34, 60)
            atoms.append(("N4", n4, "N", "N", -0.85))
            atoms.append(("H41", n4 + np.array([1.01, 0.0]), "H", "H", 0.42))
            atoms.append(("H42", n4 + 1.01 * _rot2([1.0, 0.0], 120), "H", "H", 0.42))
        c1 = _polar(c, RING_R + 1.47, 240)
    else:
        c = np.array([-X_PUR, 0.0])
        ring = {"N1": 0, "C6": 60, "C5": 120, "C4": 180, "N3": 240, "C2": 300}
        pos = {n: _polar(c, RING_R, d) for n, d in ring.items()}
        c5 = c + np.array([-1.871, 1.080])
        r5 = 1.1824
        pos["N7"] = _polar(c5, r5, 78)
        pos["C8"] = _polar(c5, r5, 150)
        pos["N9"] = _polar(c5, r5, 222)
        for n in ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"):
            atoms.append((n, pos[n], n[0], "N" if n[0] == "N" else "CA",
                          -0.45 if n[0] == "N" else 0.15))
        atoms.append(("H8", _polar(c5, r5 + 1.08, 150), "H", "HA", 0.15))
        if base == "A":
            n6 = _polar(c, RING_R + 1.34, 60)
            atoms.append(("N6", n6, "N", "N", -0.85))
            atoms.append(("H61", n6 + np.array([1.01, 0.0]), "H", "H", 0.42))
            atoms.append(("H62", n6 + 1.01 * _rot2([1.0, 0.0], 120), "H", "H", 0.42))
            atoms.append(("H2", _polar(c, RING_R + 1.08, 300), "H", "HA", 0.15))
        else:  # G
            atoms.append(("O6", _polar(c, RING_R + 1.23, 60), "O", "O", -0.58))
            atoms.append(("H1", pos["N1"] + np.array([1.01, 0.0]), "H", "H", 0.40))
            n2 = _polar(c, RING_R + 1.34, 300)
            atoms.append(("N2", n2, "N", "N", -0.85))
            atoms.append(("H21", n2 + np.array([1.01, 0.0]), "H", "H", 0.42))
            atoms.append(("H22", n2 + 1.01 * _rot2([1.0, 0.0], -120), "H", "H", 0.42))
        c1 = _polar(c5, r5 + 1.47, 222)
    return atoms, c1


def nucleotide(base: str):
    """Full nucleotide (strand-A side, pair frame) -> list of atom dicts and bonds."""
    base_atoms, c1 = base_template(base)
    atoms = []
    for name, xy, el, typ, q in base_atoms:
        atoms.append({"name": name, "xyz": np.array([xy[0], xy[1], 0.0]), "el": el,
                      "type": typ, "q": q})
    c1p = np.array([c1[0], c1[1], 0.0])
    atoms.append({"name": "C1'", "xyz": c1p, "el": "C", "type": "CT", "q": SUGAR_CHARGES["C1'"]})
    for name, off in SUGAR.items():
        atoms.append({"name": name, "xyz": c1p + np.array(off), "el": name[0], "type": SUGAR_TYPES[name],
                      "q": SUGAR_CHARGES[name]})
    index = {a["name"]: k for k, a in enumerate(atoms)}
    for hname, (parent, hint) in SUGAR_H.items():
        p = atoms[index[parent]]["xyz"]
        typ = "HO" if parent.startswith("O") else "H1"
        atoms.append({"name": hname, "xyz": p + (0.96 if typ == "HO" else 1.09) * _unit(hint),
                      "el": "H", "type": typ, "q": 0.42 if typ == "HO" else 0.08})
    index = {a["name"]: k for k, a in enumerate(atoms)}
    # neutralise to a net charge of -1 per nucleotide
    total = sum(a["q"] for a in atoms)
    corr = (-1.0 - total) / len(atoms)
    for a in atoms:
        a["q"] += corr
    bonds = []

    def bond(x, y):
        if x in index and y in index:
            bonds.append((index[x], index[y]))

    ring_bonds = [("N1", "C2"), ("C2", "N3"), ("N3", "C4"), ("C4", "C5"), ("C5", "C6"),
                  ("C6", "N1"), ("C5", "N7"), ("N7", "C8"), ("C8", "N9"), ("N9", "C4"),
                  ("C2", "O2"), ("C4", "O4"), ("C4", "N4"), ("N4", "H41"), ("N4", "H42"),
                  ("N3", "H3"), ("C5", "H5"), ("C6", "H6"), ("C8", "H8"), ("C6", "N6"),
                  ("N6", "H61"), ("N6", "H62"), ("C2", "H2"), ("C6", "O6"), ("N1", "H1"),
                  ("C2", "N2"), ("N2", "H21"), ("N2", "H22")]
    for x, y in ring_bonds:
        bond(x, y)
    bond("N9" if base in ("A", "G") else "N1", "C1'")
    for x, y in [("C1'", "O4'"), ("C1'", "C2'"), ("C2'", "O2'"), ("C2'", "C3'"), ("C3'", "C4'"),
                 ("C4'", "O4'"), ("C3'", "O3'"), ("C4'", "C5'"), ("C5'", "O5'"), ("O5'", "P"),
                 ("P", "OP1"), ("P", "OP2"), ("C1'", "H1'"), ("C2'", "H2'"), ("O2'", "HO2'"),
                 ("C3'", "H3'"), ("C4'", "H4'"), ("C5'", "H5'"), ("C5'", "H5''")]:
        bond(x, y)
    return atoms, bonds


def _level_frame(k: int):
    t = k * TWIST
    rot = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    return rot, np.array([0.0, 0.0, k * RISE])


DYAD = np.diag([-1.0, 1.0, -1.0])
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def _rotation(axis, angle):
    axis = _unit(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _smooth_step(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def build_system(sequence: str = "GCAUGGCAUCGA", n_frames: int = 120, dt_ps: float = 10.0,
                 solvent: bool = True, seed: int = 7):
    """Return (topology dict, coordinates [n_frames, n_atoms, 3], box)."""
    rng = np.random.default_rng(seed)
    n_bp = len(sequence)
    target = "".join(COMPLEMENT[b] for b in reversed(sequence))
    atoms: List[dict] = []
    bonds: List[Tuple[int, int]] = []
    residues: List[dict] = []
    level_of_atom: List[int] = []
    strand_of_atom: List[int] = []

    def add_residue(resname, res_atoms, res_bonds, strand, level, frame_xyz):
        start = len(atoms)
        residues.append({"name": resname, "start": start, "strand": strand, "level": level})
        for a, xyz in zip(res_atoms, frame_xyz):
            atoms.append(dict(a, xyz=xyz))
            level_of_atom.append(level)
            strand_of_atom.append(strand)
        bonds.extend((start + x, start + y) for x, y in res_bonds)
        return start

    prev_o3 = None
    for k, b in enumerate(sequence):  # aptamer: levels 0..n-1, 5'->3' along +z
        nt, nb = nucleotide(b)
        rot, shift = _level_frame(k)
        xyz = [rot @ a["xyz"] + shift for a in nt]
        start = add_residue(b, nt, nb, 0, k, xyz)
        names = [a["name"] for a in nt]
        if prev_o3 is not None:
            bonds.append((prev_o3, start + names.index("P")))
        prev_o3 = start + names.index("O3'")
    prev_o3 = None
    for m, b in enumerate(target):  # target: 5' end pairs with the aptamer 3' end
        k = n_bp - 1 - m
        nt, nb = nucleotide(b)
        rot, shift = _level_frame(k)
        xyz = [rot @ (DYAD @ a["xyz"]) + shift for a in nt]
        start = add_residue(b, nt, nb, 1, k, xyz)
        names = [a["name"] for a in nt]
        if prev_o3 is not None:
            bonds.append((prev_o3, start + names.index("P")))
        prev_o3 = start + names.index("O3'")
    n_solute_nt = len(atoms)

    def atom_index(strand, res_in_strand, name):
        ri = res_in_strand + (n_bp if strand == 1 else 0)
        end = residues[ri + 1]["start"] if ri + 1 < len(residues) else len(atoms)
        for idx in range(residues[ri]["start"], end):
            if atoms[idx]["name"] == name:
                return idx
        raise KeyError(name)

    # Mg ions: one bound to an aptamer phosphate, one bridging aptamer and
    # target oxygens near the fraying end, one visiting a target phosphate.
    duplex = np.array([a["xyz"] for a in atoms])
    bridge_site = _bridge_site(atoms, duplex, strand_of_atom, level_of_atom, n_bp)
    mg_specs = [
        ("bound", (0, 2, "OP1", "P")),
        ("bridge", bridge_site),
        ("transient", (1, 3, "OP2", "P")),
    ]
    mg_atoms = []
    for kind, spec in mg_specs:
        idx = len(atoms)
        residues.append({"name": "MG", "start": idx, "strand": 2, "level": -1})
        xyz = spec if kind == "bridge" else np.zeros(3)
        atoms.append({"name": "MG", "xyz": xyz, "el": "MG", "type": "MG", "q": 2.0})
        level_of_atom.append(-1)
        strand_of_atom.append(2)
        mg_atoms.append(idx)

    solute = np.array([a["xyz"] for a in atoms[:n_solute_nt]])
    center = solute.mean(axis=0)
    lo = solute.min(axis=0) - 12.0
    hi = solute.max(axis=0) + 12.0
    box_len = hi - lo
    shift_all = -lo  # place the system inside [0, L)
    for a in atoms:
        a["xyz"] = a["xyz"] + shift_all

    n_na = 2 * n_bp - 2 * len(mg_specs)
    na_atoms = []
    water_start = None
    if solvent:
        grid = np.arange(1.6, box_len.max(), 3.1)
        pts = np.array([(x, y, z) for x in grid if x < box_len[0] - 1.0 for y in grid
                        if y < box_len[1] - 1.0 for z in grid if z < box_len[2] - 1.0])
        from scipy.spatial import cKDTree
        tree = cKDTree(solute + shift_all)
        d, _ = tree.query(pts)
        pts = pts[d > 3.0]
        rng.shuffle(pts)
        for p in pts[:n_na]:
            idx = len(atoms)
            residues.append({"name": "Na+", "start": idx, "strand": 3, "level": -1})
            atoms.append({"name": "Na+", "xyz": p, "el": "NA", "type": "Na+", "q": 1.0})
            na_atoms.append(idx)
            level_of_atom.append(-1)
            strand_of_atom.append(3)
        water_start = len(atoms)
        for p in pts[n_na:]:
            u = _unit(rng.normal(size=3))
            v = _unit(np.cross(u, rng.normal(size=3)))
            ang = np.radians(104.52) / 2
            h1 = p + 0.9572 * (np.cos(ang) * u + np.sin(ang) * v)
            h2 = p + 0.9572 * (np.cos(ang) * u - np.sin(ang) * v)
            idx = len(atoms)
            residues.append({"name": "WAT", "start": idx, "strand": 4, "level": -1})
            atoms.append({"name": "O", "xyz": p, "el": "O", "type": "OW", "q": -0.834})
            atoms.append({"name": "H1", "xyz": h1, "el": "H", "type": "HW", "q": 0.417})
            atoms.append({"name": "H2", "xyz": h2, "el": "H", "type": "HW", "q": 0.417})
            bonds.extend([(idx, idx + 1), (idx, idx + 2)])
            level_of_atom.extend([-1] * 3)
            strand_of_atom.extend([4] * 3)

    n_atoms = len(atoms)
    base = np.array([a["xyz"] for a in atoms])
    level = np.array(level_of_atom)
    strand = np.array(strand_of_atom)
    solute_mask = np.zeros(n_atoms, bool)
    solute_mask[:n_solute_nt] = True
    rigid_mask = solute_mask.copy()
    rigid_mask[mg_atoms] = True
    centre = center + shift_all

    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    fray_w = np.clip((level - (n_bp - 5)) / 4.0, 0.0, 1.0) * (strand == 1)
    for t in range(n_frames):
        x = base.copy()
        s = t / max(1, n_frames - 1)
        fray = _smooth_step((s - 0.30) / 0.15) * (1.0 - _smooth_step((s - 0.65) / 0.15))
        # fray: target nucleotides near the top move away from the aptamer
        for k in range(n_bp):
            sel = (strand == 1) & (level == k)
            if not sel.any() or fray_w[sel][0] == 0:
                continue
            rot, _ = _level_frame(k)
            away = rot @ np.array([1.0, 0.0, 0.0])
            x[sel] += 6.5 * fray * fray_w[sel][0] * away
        x[solute_mask] += rng.normal(scale=0.12, size=(solute_mask.sum(), 3))
        for (kind, spec), idx in zip(mg_specs, mg_atoms):
            if kind != "bridge":  # the bridging ion stays with the aptamer
                st, r, oname, pname = spec
                o = x[atom_index(st, r, oname)]
                p = x[atom_index(st, r, pname)]
                dist = 2.08
                if kind == "transient":  # inner -> outer sphere -> unbound -> back
                    outer = _smooth_step((s - 0.35) / 0.1) * (1 - _smooth_step((s - 0.85) / 0.1))
                    away = _smooth_step((s - 0.55) / 0.1) * (1 - _smooth_step((s - 0.8) / 0.1))
                    dist += 2.0 * outer + 5.0 * away
                x[idx] = o + dist * _unit(o - p)
            x[idx] += rng.normal(scale=0.05, size=3)
        # rigid body drift of the solute (and its ions)
        rot = _rotation([1.0, 1.0, 0.2], np.radians(25.0) * np.sin(2 * np.pi * s))
        x[rigid_mask] = (x[rigid_mask] - centre) @ rot.T + centre + np.array([2.0, -1.0, 1.5]) * s
        if solvent:
            free = ~rigid_mask
            x[free] += rng.normal(scale=0.25, size=(int(free.sum()), 3))
        coords[t] = x
    box = np.array([box_len[0], box_len[1], box_len[2], 90.0, 90.0, 90.0])
    topo = {"atoms": atoms, "bonds": bonds, "residues": residues, "n_solute_res": 2 * n_bp + len(mg_atoms) + len(na_atoms),
            "water_start": water_start, "sequence": sequence, "target": target}
    return topo, coords, box


def _bridge_site(atoms, xyz, strand, level, n_bp):
    """Position for an Mg2+ bridging an aptamer and a target oxygen (Mg-O = 2.08 A)."""
    from scipy.spatial import cKDTree

    strand = np.asarray(strand)
    level = np.asarray(level)
    is_o = np.array([a["el"] == "O" for a in atoms])
    a_idx = np.flatnonzero(is_o & (strand == 0) & (level >= n_bp - 4))
    b_idx = np.flatnonzero(is_o & (strand == 1) & (level >= n_bp - 4))
    tree = cKDTree(xyz)
    best = None
    for i in a_idx:
        for j in b_idx:
            d = np.linalg.norm(xyz[i] - xyz[j])
            if not 2.6 <= d <= 4.0:
                continue
            axis = _unit(xyz[j] - xyz[i])
            perp = _unit(np.cross(axis, [0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [1.0, 0, 0]))
            perp2 = np.cross(axis, perp)
            mid = 0.5 * (xyz[i] + xyz[j])
            lift = np.sqrt(max(0.0, 2.08 ** 2 - (d / 2) ** 2))
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                pos = mid + lift * (np.cos(ang) * perp + np.sin(ang) * perp2)
                near = [k for k in tree.query_ball_point(pos, 2.2) if k not in (i, j)]
                clearance = min((np.linalg.norm(xyz[k] - pos) for k in near), default=2.2)
                if best is None or clearance > best[0]:
                    best = (clearance, pos)
    if best is None:  # fall back to a site next to the first aptamer oxygen
        return xyz[a_idx[0]] + np.array([2.08, 0.0, 0.0])
    return best[1]


def write_topology(path: str, topo: dict, box) -> None:
    atoms = topo["atoms"]
    residues = topo["residues"]
    bonds = topo["bonds"]
    n_atoms = len(atoms)
    type_names = sorted({a["type"] for a in atoms}, key=list(LJ_TYPES).index)
    tindex = {t: k + 1 for k, t in enumerate(type_names)}
    ntypes = len(type_names)
    acoef, bcoef = [], []
    for i in range(ntypes):
        for j in range(i + 1):
            ri, ei = LJ_TYPES[type_names[i]]
            rj, ej = LJ_TYPES[type_names[j]]
            rmin, eps = ri + rj, np.sqrt(ei * ej)
            acoef.append(eps * rmin ** 12)
            bcoef.append(2.0 * eps * rmin ** 6)
    nb_index = []
    for i in range(1, ntypes + 1):
        for j in range(1, ntypes + 1):
            hi, lo = max(i, j), min(i, j)
            nb_index.append(hi * (hi - 1) // 2 + lo)
    is_h = np.array([a["el"] == "H" for a in atoms])
    bonds_h, bonds_a = [], []
    for x, y in bonds:
        entry = (3 * x, 3 * y, 1 if (is_h[x] or is_h[y]) else 2)
        (bonds_h if (is_h[x] or is_h[y]) else bonds_a).append(entry)
    res_ptr = [r["start"] + 1 for r in residues]
    sizes = np.diff(res_ptr + [n_atoms + 1])
    # molecules: connected components
    parent = list(range(n_atoms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for x, y in bonds:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    mol_sizes = []
    last = None
    for a in range(n_atoms):
        r = find(a)
        if r != last:
            mol_sizes.append(0)
            last = r
        mol_sizes[-1] += 1
    ifbox = 1 if box is not None else 0
    pointers = [n_atoms, ntypes, len(bonds_h), len(bonds_a), 0, 0, 0, 0, 0, 0, n_atoms,
                len(residues), len(bonds_a), 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ifbox,
                int(sizes.max()), 0, 0]
    sections = [
        ("POINTERS", "i", pointers),
        ("ATOM_NAME", "a", [a["name"] for a in atoms]),
        ("CHARGE", "e", [a["q"] * AMBER_CHARGE_FACTOR for a in atoms]),
        ("ATOMIC_NUMBER", "i", [ELEMENT_Z[a["el"]] for a in atoms]),
        ("MASS", "e", [MASSES[a["el"]] for a in atoms]),
        ("ATOM_TYPE_INDEX", "i", [tindex[a["type"]] for a in atoms]),
        ("NUMBER_EXCLUDED_ATOMS", "i", [1] * n_atoms),
        ("NONBONDED_PARM_INDEX", "i", nb_index),
        ("RESIDUE_LABEL", "a", [r["name"] for r in residues]),
        ("RESIDUE_POINTER", "i", res_ptr),
        ("BOND_FORCE_CONSTANT", "e", [340.0, 300.0]),
        ("BOND_EQUIL_VALUE", "e", [1.09, 1.50]),
        ("ANGLE_FORCE_CONSTANT", "e", []),
        ("ANGLE_EQUIL_VALUE", "e", []),
        ("DIHEDRAL_FORCE_CONSTANT", "e", []),
        ("DIHEDRAL_PERIODICITY", "e", []),
        ("DIHEDRAL_PHASE", "e", []),
        ("SCEE_SCALE_FACTOR", "e", []),
        ("SCNB_SCALE_FACTOR", "e", []),
        ("SOLTY", "e", [0.0]),
        ("LENNARD_JONES_ACOEF", "e", acoef),
        ("LENNARD_JONES_BCOEF", "e", bcoef),
        ("BONDS_INC_HYDROGEN", "i", [v for b in bonds_h for v in b]),
        ("BONDS_WITHOUT_HYDROGEN", "i", [v for b in bonds_a for v in b]),
        ("ANGLES_INC_HYDROGEN", "i", []),
        ("ANGLES_WITHOUT_HYDROGEN", "i", []),
        ("DIHEDRALS_INC_HYDROGEN", "i", []),
        ("DIHEDRALS_WITHOUT_HYDROGEN", "i", []),
        ("EXCLUDED_ATOMS_LIST", "i", [0] * n_atoms),
        ("HBOND_ACOEF", "e", []),
        ("HBOND_BCOEF", "e", []),
        ("HBCUT", "e", []),
        ("AMBER_ATOM_TYPE", "a", [a["type"] for a in atoms]),
        ("TREE_CHAIN_CLASSIFICATION", "a", ["BLA"] * n_atoms),
        ("JOIN_ARRAY", "i", [0] * n_atoms),
        ("IROTAT", "i", [0] * n_atoms),
    ]
    if ifbox:
        first_solvent_mol = len(mol_sizes) + 1
        if topo.get("water_start") is not None:
            first_solvent_mol = 1 + sum(1 for start in _molecule_starts(mol_sizes)
                                        if start < topo["water_start"])
        sections += [
            ("SOLVENT_POINTERS", "i", [topo["n_solute_res"], len(mol_sizes), first_solvent_mol]),
            ("ATOMS_PER_MOLECULE", "i", mol_sizes),
            ("BOX_DIMENSIONS", "e", [90.0, float(box[0]), float(box[1]), float(box[2])]),
        ]
    sections += [
        ("RADIUS_SET", "s", ["modified Bondi radii (mbondi2)"]),
        ("RADII", "e", [1.2 if a["el"] == "H" else 1.5 for a in atoms]),
        ("SCREEN", "e", [0.8] * n_atoms),
        ("IPOL", "i", [0]),
    ]
    write_parm7(path, sections, title=f"{PROJECT_ABBREV} synthetic RNA aptamer/target demo")


def _molecule_starts(mol_sizes):
    start = 0
    for s in mol_sizes:
        yield start
        start += s


def write_trajectory(path: str, topology_path: str, coords: np.ndarray, box, dt_ps: float) -> None:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(topology_path, coords, topology_format="PARM7", format=MemoryReader,
                         dt=dt_ps, dimensions=box)
        with mda.Writer(path, n_atoms=u.atoms.n_atoms, format="NCDF") as w:
            for _ in u.trajectory:
                w.write(u.atoms)


def demo_project(topology: str, trajectory: str, out_dir: str, n_bp: int,
                 cleaning: bool, mid_time_ps: float = 300.0) -> Project:
    p = Project()
    p.inputs.topology = topology
    p.inputs.trajectories = [trajectory]
    p.inputs.time_unit = "ns"
    p.groups[0].residues = f"1-{n_bp}"
    p.groups[1].residues = f"{n_bp + 1}-{2 * n_bp}"
    p.cleaning.enabled = cleaning
    # screened electrostatics (epsilon = 4r) suit a stripped, solvent-free complex
    p.analyses.energy.dielectric = 4.0
    p.analyses.energy.distance_dependent = True
    p.analyses.energy.target_value = -60.0
    p.visual.rotation = [-90.0, 0.0, 0.0]  # helix axis vertical, viewed from the side
    p.output.out_dir = out_dir
    p.output.pdb.enabled = True
    p.output.pdb.frames = "0, last"
    p.output.pdb.times = f"{mid_time_ps:g}ps"
    p.output.pdb.selection = "Aptamer, Target"
    p.layout.preset = "large_plus_4"
    p.layout.panels.append(PanelSpec("heatmap", ["mg/occupancy"], "Mg occupancy"))
    return p


def build_demo(out_dir: str, n_frames: int = 120, dt_ps: float = 10.0, solvent: bool = True,
               sequence: str = "GCAUGGCAUCGA", seed: int = 7) -> Dict[str, str]:
    """Write ``demo.parm7``, ``demo.nc`` and ``demo_preset.json`` into ``out_dir``."""
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    topo, coords, box = build_system(sequence, n_frames, dt_ps, solvent, seed)
    top = os.path.join(out_dir, "demo.parm7")
    traj = os.path.join(out_dir, "demo.nc")
    write_topology(top, topo, box)
    write_trajectory(traj, top, coords, box, dt_ps)
    preset = os.path.join(out_dir, "demo_preset.json")
    project = demo_project("demo.parm7", "demo.nc", f"{PROJECT_ABBREV}_output", len(sequence),
                           cleaning=solvent and shutil.which("cpptraj") is not None,
                           mid_time_ps=(n_frames // 2) * dt_ps)
    data = project.to_dict()
    with open(preset, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return {"topology": top, "trajectory": traj, "preset": preset}
