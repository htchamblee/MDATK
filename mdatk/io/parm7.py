"""Minimal reader / writer for Amber ``.parm7`` (``prmtop``) files.

MDAnalysis parses the parts of a parm7 file that describe atoms, residues and
bonds but not the Lennard-Jones tables, which the intermolecular energy
analysis needs.  This module reads any ``%FLAG`` section using the Fortran
``%FORMAT`` statement that accompanies it, and can also write a topology
(used by :mod:`mdatk.demo` to build a self-contained synthetic system).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

AMBER_CHARGE_FACTOR = 18.2223  # parm7 charges are stored as q * 18.2223

_FMT = re.compile(r"\(\s*(\d+)\s*([aAiIeEfF])\s*(\d+)(?:\.(\d+))?\s*\)")


@dataclass
class Section:
    name: str
    count: int
    kind: str  # "a", "i", "e", "f"
    width: int
    lines: List[str]

    def values(self):
        raw: List[str] = []
        for line in self.lines:
            line = line.rstrip("\n")
            for k in range(0, len(line), self.width):
                chunk = line[k:k + self.width]
                if self.kind == "a" or chunk.strip():
                    raw.append(chunk)
        if self.kind == "a":
            return [s.strip() for s in raw]
        if self.kind == "i":
            return np.array([int(s) for s in raw], dtype=np.int64)
        return np.array([float(s.replace("D", "E").replace("d", "e")) for s in raw], dtype=np.float64)


def read_sections(path: str, wanted: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Read ``%FLAG`` sections of a parm7 file into Python lists / numpy arrays."""
    wanted_set = set(wanted) if wanted is not None else None
    out: Dict[str, object] = {}
    current: Optional[Section] = None

    def flush():
        if current is not None and (wanted_set is None or current.name in wanted_set):
            out[current.name] = current.values()

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                flush()
                name = line[5:].strip()
                current = Section(name, 0, "a", 80, [])
                continue
            if line.startswith("%FORMAT"):
                if current is None:
                    continue
                m = _FMT.search(line)
                if not m:
                    raise ValueError(f"{path}: cannot parse format line {line.strip()!r}")
                current.count = int(m.group(1))
                current.kind = m.group(2).lower()
                current.width = int(m.group(3))
                continue
            if line.startswith("%VERSION") or line.startswith("%COMMENT"):
                continue
            if current is not None and (wanted_set is None or current.name in wanted_set):
                current.lines.append(line)
    flush()
    return out


class LJTable:
    """Amber Lennard-Jones parameters looked up per atom pair."""

    def __init__(self, type_index: np.ndarray, ntypes: int, nb_index: np.ndarray,
                 acoef: np.ndarray, bcoef: np.ndarray):
        self.type_index = np.asarray(type_index, dtype=np.int64)  # 1-based per atom
        self.ntypes = int(ntypes)
        self.nb_index = np.asarray(nb_index, dtype=np.int64)
        self.acoef = np.asarray(acoef, dtype=np.float64)
        self.bcoef = np.asarray(bcoef, dtype=np.float64)

    @classmethod
    def from_parm7(cls, path: str) -> "LJTable":
        s = read_sections(path, {"POINTERS", "ATOM_TYPE_INDEX", "NONBONDED_PARM_INDEX",
                                 "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF"})
        missing = [k for k in ("POINTERS", "ATOM_TYPE_INDEX", "NONBONDED_PARM_INDEX",
                               "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF") if k not in s]
        if missing:
            raise ValueError(f"{path}: missing parm7 sections {missing}")
        ntypes = int(s["POINTERS"][1])
        return cls(s["ATOM_TYPE_INDEX"], ntypes, s["NONBONDED_PARM_INDEX"],
                   s["LENNARD_JONES_ACOEF"], s["LENNARD_JONES_BCOEF"])

    def pair_coefficients(self, i: np.ndarray, j: np.ndarray):
        """A and B coefficients (E = A/r^12 - B/r^6) for atom index arrays."""
        ti = self.type_index[i] - 1
        tj = self.type_index[j] - 1
        idx = self.nb_index[ti * self.ntypes + tj]
        a = np.zeros(len(idx))
        b = np.zeros(len(idx))
        ok = idx > 0  # negative indices refer to (unused) 10-12 H-bond terms
        a[ok] = self.acoef[idx[ok] - 1]
        b[ok] = self.bcoef[idx[ok] - 1]
        return a, b


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_FORMATS = {
    "a": ("20a4", 20, 4),
    "i": ("10I8", 10, 8),
    "e": ("5E16.8", 5, 16),
    "s": ("1a80", 1, 80),
}


def _format_values(kind: str, values: Sequence) -> List[str]:
    if kind == "s":
        return [f"{str(v)[:80]:<80s}" for v in values] or [""]
    _, per_line, width = _FORMATS[kind]
    cells = []
    for v in values:
        if kind == "a":
            cells.append(f"{str(v)[:4]:<4s}")
        elif kind == "i":
            cells.append(f"{int(v):8d}")
        else:
            cells.append(f"{float(v):16.8E}")
    lines = ["".join(cells[k:k + per_line]) for k in range(0, len(cells), per_line)]
    return lines or [""]


def write_parm7(path: str, sections: Sequence, title: str = "MDATK") -> None:
    """Write a parm7 file from ``[(flag, kind, values), ...]`` (kind: a/i/e)."""
    with open(path, "w", encoding="ascii") as fh:
        fh.write("%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/26  00:00:00\n")
        fh.write("%FLAG TITLE\n%FORMAT(20a4)\n")
        fh.write(f"{title[:80]:<80s}\n")
        for flag, kind, values in sections:
            fmt = _FORMATS[kind][0]
            fh.write(f"%FLAG {flag}\n%FORMAT({fmt})\n")
            for line in _format_values(kind, values):
                fh.write(line + "\n")
