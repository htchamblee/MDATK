"""Atom roles for the 3D representations (computed once per figure).

Nucleotides are recognised by their atoms, not their residue names, so
modified residues work when they keep the standard ring atoms:

* purine ring perimeter ``N9 C8 N7 C5 C6 N1 C2 N3 C4`` (glycosidic N9),
* pyrimidine ring ``N1 C2 N3 C4 C5 C6`` (glycosidic N1; C5 for pseudouridine),
* ribose ``C1' C2' C3' C4' O4'``.

Old atom names (``C1*``, ``O1P``) are normalised.  Amino acids use CA as their
backbone point.  Backbone segments break at chain changes, gaps in the residue
numbering of the topology and control points more than 8.5 Å apart.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

PURINE_RING = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
PYRIMIDINE_RING = ("N1", "C2", "N3", "C4", "C5", "C6")
SUGAR_RING = ("C1'", "C2'", "C3'", "C4'", "O4'")
BACKBONE = {"P": ("P", "O5'", "C5'", "C4'", "CA"), "C4'": ("C4'", "C3'", "C5'", "C1'", "CA")}
SEGMENT_BREAK = 8.5  # Å between consecutive control points
# Bondi van der Waals radii (Å)
VDW_RADIUS = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80, "F": 1.47,
              "MG": 1.73, "NA": 2.27, "K": 2.75, "CL": 1.75, "CA": 2.31, "ZN": 1.39}
ELEMENT_RGB = {"C": (0.56, 0.56, 0.56), "N": (0.19, 0.31, 0.97), "O": (0.90, 0.13, 0.13),
               "P": (1.00, 0.55, 0.00), "H": (0.92, 0.92, 0.92), "S": (0.90, 0.78, 0.20),
               "MG": (0.0, 0.51, 0.0)}
NUCLEOTIDE_RGB = {"A": (0.27, 0.53, 0.80), "G": (0.33, 0.66, 0.41), "C": (0.85, 0.65, 0.13),
                  "U": (0.80, 0.33, 0.33), "T": (0.80, 0.33, 0.33), "?": (0.6, 0.6, 0.6)}
_ALIASES = {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3"}


def normalise_name(name: str) -> str:
    n = str(name).strip().replace("*", "'")
    return _ALIASES.get(n, n)


_CHARMM = {"ADE": "A", "GUA": "G", "CYT": "C", "URA": "U", "URI": "U", "THY": "T"}


def base_letter(resname: str) -> str:
    """A/G/C/U/T for standard and common Amber / CHARMM residue names, else '?'."""
    r = str(resname).strip().upper()
    if r in ("PSU", "PU"):
        return "U"
    if r[:3] in _CHARMM and len(r) <= 4:
        return _CHARMM[r[:3]]
    for prefix in ("RA", "RG", "RC", "RU", "DA", "DG", "DC", "DT", "LA", "LG", "LC", "LU"):
        if r.startswith(prefix) and len(r) <= 4:
            return prefix[1]
    for k in ("A", "G", "C", "U", "T"):
        if r == k or (r.startswith(k) and r[1:] in ("3", "5", "N")):
            return k
    return "?"


def _letter_from_atoms(atoms: Dict[str, int]) -> str:
    if all(n in atoms for n in PURINE_RING):
        return "G" if "O6" in atoms else "A" if "N6" in atoms else "?"
    if all(n in atoms for n in PYRIMIDINE_RING):
        if "N4" in atoms:
            return "C"
        if "C7" in atoms or "C5M" in atoms:
            return "T"
        return "U" if "O4" in atoms else "?"
    return "?"


class Topology:
    """Per-residue atom indices (into the ``atoms`` arrays) for every style."""

    def __init__(self, names: Sequence[str], elements: Sequence[str], resnames: Sequence[str],
                 row: np.ndarray, n_res: int, bonds: np.ndarray, ref_xyz: np.ndarray,
                 chain: np.ndarray, resindex: np.ndarray, backbone_atom: str = "P"):
        self.n_res = int(n_res)
        self.row = np.asarray(row, dtype=np.int64)
        self.names = [normalise_name(n) for n in names]
        self.elements = [str(e).strip().upper() or "C" for e in elements]
        self.bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
        self.chain = np.asarray(chain, dtype=np.int64)
        self.resindex = np.asarray(resindex, dtype=np.int64)
        table: List[Dict[str, int]] = [dict() for _ in range(self.n_res)]
        for i, (r, n) in enumerate(zip(self.row, self.names)):
            if 0 <= r < self.n_res:
                table[r].setdefault(n, i)
        self.table = table
        first = {}
        for i, r in enumerate(self.row):
            first.setdefault(int(r), i)
        self.resname = [str(resnames[first[r]]) if r in first else "?" for r in range(self.n_res)]
        self.letter = [base_letter(x) for x in self.resname]
        for r in range(self.n_res):  # unknown names: tell the base by its exocyclic atoms
            if self.letter[r] == "?":
                self.letter[r] = _letter_from_atoms(table[r])
        self.radius = np.array([VDW_RADIUS.get(e, 1.7) for e in self.elements], dtype=np.float64)
        self.element_rgb = np.array([ELEMENT_RGB.get(e, (0.6, 0.4, 0.7)) for e in self.elements])
        self.is_carbon = np.array([e == "C" for e in self.elements])
        self.is_hydrogen = np.array([e == "H" for e in self.elements])

        def pick(r: int, names: Sequence[str]) -> int:
            return next((table[r][n] for n in names if n in table[r]), -1)

        self.control = np.array([pick(r, BACKBONE.get(backbone_atom, BACKBONE["P"]))
                                 for r in range(self.n_res)], dtype=np.int64)
        # the trace point: the control atom, else the residue's first atom (ligands)
        self.anchor_atom = np.array([c if c >= 0 else first.get(r, -1) for r, c in enumerate(self.control)],
                                    dtype=np.int64)
        # direction from the backbone toward the bases (orients ribbons and cartoons)
        self.side = np.array([pick(r, ("C1'", "C4'", "CB", "O")) for r in range(self.n_res)], dtype=np.int64)
        self.c1 = np.array([pick(r, ("C1'",)) for r in range(self.n_res)], dtype=np.int64)
        self.c4 = np.array([pick(r, ("C4'",)) for r in range(self.n_res)], dtype=np.int64)
        pur, pyr, sug = [], [], []
        for r in range(self.n_res):
            t = table[r]
            if all(n in t for n in PURINE_RING):
                pur.append((r, [t[n] for n in PURINE_RING], t["N9"]))
            elif all(n in t for n in PYRIMIDINE_RING):
                # pseudouridine is C-glycosidic (C1'-C5)
                gly = t["C5"] if self.resname[r].upper() in ("PSU", "PU") else t["N1"]
                pyr.append((r, [t[n] for n in PYRIMIDINE_RING], gly))
            if all(n in t for n in SUGAR_RING):
                sug.append((r, [t[n] for n in SUGAR_RING]))
        self.purine_res = np.array([p[0] for p in pur], dtype=np.int64)
        self.purine_ring = np.array([p[1] for p in pur], dtype=np.int64).reshape(-1, 9)
        self.purine_gly = np.array([p[2] for p in pur], dtype=np.int64)
        self.pyrimidine_res = np.array([p[0] for p in pyr], dtype=np.int64)
        self.pyrimidine_ring = np.array([p[1] for p in pyr], dtype=np.int64).reshape(-1, 6)
        self.pyrimidine_gly = np.array([p[2] for p in pyr], dtype=np.int64)
        self.sugar_res = np.array([s[0] for s in sug], dtype=np.int64)
        self.sugar_ring = np.array([s[1] for s in sug], dtype=np.int64).reshape(-1, 5)
        self.segments = self._segments(np.asarray(ref_xyz, dtype=np.float64))

    def _segments(self, xyz: np.ndarray) -> List[np.ndarray]:
        segs: List[List[int]] = []
        cur: List[int] = []
        for r in range(self.n_res):
            c = self.control[r]
            if c < 0:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
                continue
            if cur:
                p = cur[-1]
                broken = (self.chain[r] != self.chain[p] or self.resindex[r] - self.resindex[p] != 1
                          or np.linalg.norm(xyz[c] - xyz[self.control[p]]) > SEGMENT_BREAK)
                if broken:
                    if len(cur) > 1:
                        segs.append(cur)
                    cur = []
            cur.append(r)
        if len(cur) > 1:
            segs.append(cur)
        return [np.array(s, dtype=np.int64) for s in segs]

    @property
    def nucleic(self) -> np.ndarray:
        """Residue rows that have a base ring."""
        return np.union1d(self.purine_res, self.pyrimidine_res)

