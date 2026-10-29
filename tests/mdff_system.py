"""A small D-RNA + L-RNA test system for the NAMD MDFF step: helices built by MDATK, a simulated
map and a *names-and-masses-only* parm7.

The parm7 written here holds atom names, residue labels, atomic numbers, masses, atom types and a
trivial one-type Lennard-Jones table - enough for every file MDATK prepares around a topology, but
no force field.  Its title says so; it must never be simulated.  Real topologies come from the
Amber build step (tleap + ParmEd).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mdatk import builder, density, pdbtools
from mdatk.io import mrc
from mdatk.io.parm7 import write_parm7

TEST_TITLE = "MDATK TEST TOPOLOGY - NAMES AND MASSES ONLY, NO FORCE FIELD - DO NOT SIMULATE"
NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "MG": 12, "NA": 11, "CL": 17}
MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 15: 30.974, 12: 24.305, 11: 22.990, 17: 35.45, 0: 0.0}
L_TYPES = {"O3'": "LD", "O5'": "LD", "O4'": "LE", "O2'": "LH"}


def mixed_pdb(seq: str = "GCGC", gap: float = 26.0) -> str:
    """A D duplex (chains A/B) next to an L duplex (chains C/D), Amber residue names, hydrogens."""
    d = builder.build_helix(seq, chirality="D", chains=("A", "B"), scheme="amber", hydrogens=True)
    l_ = builder.build_helix(seq, chirality="L", chains=("C", "D"), scheme="amber", hydrogens=True,
                             start=(1, 1))
    out = []
    for text, dx in ((d, gap / 2), (l_, -gap / 2)):
        for line in text.splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                x = float(line[30:38]) + dx
                out.append(line[:30] + f"{x:8.3f}" + line[38:])
            elif line.startswith("TER"):
                out.append("TER")
    return "\n".join(out) + "\nEND\n"


def atoms_of(pdb_text: str):
    """[(chain, resseq, resname, atom name, element, xyz)] in file order."""
    s = pdbtools.read_pdb(pdb_text)
    out = []
    for res in s.residues():
        for a in res.atoms:
            el = (a.element or a.name.lstrip("0123456789")[:1]).upper()
            out.append((res.chain, res.resseq, res.name, a.name, el, a.xyz))
    return out


def with_solvent(atoms, waters: int = 6, mg: int = 1, spacing: float = 3.1, start=(0.0, 0.0, 14.0)):
    """Append a few waters (O, H1, H2) and Mg2+ ions as separate residues."""
    out = list(atoms)
    x0, y0, z0 = start
    for k in range(waters):
        o = (x0 + k * spacing, y0, z0)
        out += [("W", k + 1, "WAT", "O", "O", o), ("W", k + 1, "WAT", "H1", "H", (o[0] + 0.96, o[1], o[2])),
                ("W", k + 1, "WAT", "H2", "H", (o[0] - 0.24, o[1] + 0.93, o[2]))]
    for k in range(mg):
        out.append(("M", k + 1, "MG", "MG", "MG", (x0 + k * spacing, y0 + 6.0, z0)))
    return out


def write_topology(path: str, atoms, box: Optional[Sequence[float]] = None, hmr: bool = False,
                   extra_flags: Sequence[Tuple[str, str, list]] = (), l_types: bool = True) -> None:
    labels: List[str] = []
    pointers: List[int] = []
    last = None
    names, numbers, masses, types = [], [], [], []
    for k, (chain, num, resname, name, el, _xyz) in enumerate(atoms):
        key = (chain, num, resname)
        if key != last:
            labels.append(resname)
            pointers.append(k + 1)
            last = key
        z = NUMBERS.get(el, 6)
        names.append(name)
        numbers.append(z)
        masses.append(MASSES[z])
        lres = resname.startswith("L") and len(resname) <= 3 and resname not in ("LI",)
        types.append(L_TYPES.get(name, el[:2]) if (l_types and lres) else el[:2])
    if hmr:  # ParmEd HMassRepartition: solute H 3.024, the heavy atom gives the difference (water untouched)
        for k, (chain, num, resname, name, el, _xyz) in enumerate(atoms):
            if el == "H" and resname != "WAT":
                masses[k] = 3.024
    n = len(names)
    pointers_section = [0] * 31
    pointers_section[0] = n
    pointers_section[1] = 1
    pointers_section[11] = len(labels)
    pointers_section[27] = 1 if box else 0
    sections = [("POINTERS", "i", pointers_section), ("ATOM_NAME", "a", names), ("CHARGE", "e", [0.0] * n),
                ("ATOMIC_NUMBER", "i", numbers), ("MASS", "e", masses), ("ATOM_TYPE_INDEX", "i", [1] * n),
                ("NONBONDED_PARM_INDEX", "i", [1]), ("RESIDUE_LABEL", "a", labels),
                ("RESIDUE_POINTER", "i", pointers), ("LENNARD_JONES_ACOEF", "e", [1.0e6]),
                ("LENNARD_JONES_BCOEF", "e", [1.0e3]), ("AMBER_ATOM_TYPE", "a", types)]
    if box:
        sections.append(("BOX_DIMENSIONS", "e", [box[3], box[0], box[1], box[2]]))
    sections += list(extra_flags)
    write_parm7(path, sections, title=TEST_TITLE)


def write_rst7(path: str, xyz, box: Optional[Sequence[float]] = None, velocities=None) -> None:
    values = [c for p in xyz for c in p] + ([c for v in velocities for c in v] if velocities else [])
    with open(path, "w", encoding="ascii") as fh:
        fh.write("MDATK test coordinates\n")
        fh.write(f"{len(xyz):6d}\n")
        for k in range(0, len(values), 6):
            fh.write("".join(f"{v:12.7f}" for v in values[k:k + 6]) + "\n")
        if box:
            fh.write("".join(f"{v:12.7f}" for v in box) + "\n")


def write_map(path: str, atoms, resolution: float = 4.0, spacing: float = 1.5) -> mrc.DensityMap:
    xyz = np.array([a[5] for a in atoms if a[4] != "H" and a[2] not in ("WAT", "MG")])
    els = [a[4] for a in atoms if a[4] != "H" and a[2] not in ("WAT", "MG")]
    dmap = density.simulate_map(xyz, density.atom_weights(els), resolution, spacing=spacing)
    mrc.write_map(path, dmap)
    return dmap


def rotation(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    t = math.radians(degrees)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(t) * kx + (1 - math.cos(t)) * (kx @ kx)


def moved(atoms, rot: np.ndarray, shift) -> list:
    """The atoms after x -> rot x + shift (what tleap does to a solute when it solvates it)."""
    out = []
    for a in atoms:
        p = rot @ np.array(a[5]) + np.asarray(shift)
        out.append(a[:5] + (tuple(float(v) for v in p),))
    return out


def model_pdb(atoms) -> str:
    """PDB text of the solute (the map-frame model)."""
    lines = []
    for k, (chain, num, resname, name, el, (x, y, z)) in enumerate(atoms):
        if resname in ("WAT", "MG"):
            continue
        nm = f" {name:<3s}" if len(name) < 4 else name
        lines.append(f"ATOM  {k + 1:5d} {nm} {resname:>3s} {chain}{num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00          {el:>2s}")
    return "\n".join(lines) + "\nEND\n"
