"""Helpers shared by several analyses."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..selections import residue_labels


def residue_rows(atoms) -> Tuple[object, np.ndarray]:
    """Unique residues of ``atoms`` (sorted) and the residue row of every atom."""
    uniq = np.unique(atoms.resindices)
    residues = atoms.universe.residues[uniq]
    rows = np.searchsorted(uniq, atoms.resindices)
    return residues, rows


def without(atoms, other):
    """Atoms of ``atoms`` that are not in ``other`` (order preserved)."""
    if other is None or other.n_atoms == 0:
        return atoms
    return atoms[~np.isin(atoms.ix, other.ix)]


def labels(residues):
    return residue_labels(residues)


def per_row_counts(offsets: np.ndarray, rows: np.ndarray, values: np.ndarray, n_rows: int) -> np.ndarray:
    """Dense rows x frames matrix from CSR triplets (summing values)."""
    n_frames = len(offsets) - 1
    out = np.zeros((n_rows, n_frames), dtype=np.float32)
    if len(rows):
        frame_of = np.repeat(np.arange(n_frames), np.diff(offsets))
        np.add.at(out, (rows, frame_of), values)
    return out
