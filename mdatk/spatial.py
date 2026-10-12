"""Spatial indexing helpers.

All neighbour searches in MDATK go through :func:`pairs_within`, which uses a
:class:`scipy.spatial.cKDTree` for open boundary conditions and MDAnalysis'
cell-list (``nsgrid``) search for periodic boxes.  Both scale roughly linearly
with system size, which keeps contact / proximity analyses viable for systems
with millions of atoms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

_EMPTY_I = np.empty(0, dtype=np.int64)
_EMPTY_D = np.empty(0, dtype=np.float64)


def _usable_box(box) -> Optional[np.ndarray]:
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (6,) or np.any(box[:3] <= 0):
        return None
    return box


def pairs_within(a: np.ndarray, b: np.ndarray, cutoff: float,
                 box=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(i, j, d)`` for every pair with ``|a[i] - b[j]| <= cutoff``."""
    if len(a) == 0 or len(b) == 0:
        return _EMPTY_I, _EMPTY_I, _EMPTY_D
    box = _usable_box(box)
    if box is not None:
        from MDAnalysis.lib.distances import capped_distance

        pairs, dist = capped_distance(np.asarray(a, np.float32), np.asarray(b, np.float32),
                                      max_cutoff=cutoff, box=box, return_distances=True)
        if len(pairs) == 0:
            return _EMPTY_I, _EMPTY_I, _EMPTY_D
        return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64), dist.astype(np.float64)
    ta = cKDTree(np.asarray(a, dtype=np.float64))
    tb = cKDTree(np.asarray(b, dtype=np.float64))
    res = ta.sparse_distance_matrix(tb, cutoff, output_type="ndarray")
    if len(res) == 0:
        return _EMPTY_I, _EMPTY_I, _EMPTY_D
    return res["i"].astype(np.int64), res["j"].astype(np.int64), res["v"].astype(np.float64)


def minimum_image_vectors(vectors: np.ndarray, box=None) -> np.ndarray:
    """Apply the minimum image convention to difference vectors (orthorhombic only)."""
    box = _usable_box(box)
    if box is None or not np.allclose(box[3:], 90.0):
        return vectors
    lengths = box[:3]
    return vectors - lengths * np.round(vectors / lengths)


def kabsch(mobile: np.ndarray, reference: np.ndarray,
           weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal (optionally weighted) rotation superposing ``mobile`` onto ``reference``.

    Returns ``(R, mobile_center, reference_center)`` such that
    ``(x - mobile_center) @ R.T + reference_center`` maps mobile coordinates
    onto the reference.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if weights is None:
        w = np.full(len(mobile), 1.0 / max(1, len(mobile)))
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    mc = w @ mobile
    rc = w @ reference
    if len(mobile) < 3:
        return np.eye(3), mc, rc
    h = ((mobile - mc) * w[:, None]).T @ (reference - rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if d == 0:
        d = 1.0
    corr = np.diag([1.0, 1.0, d])
    r = vt.T @ corr @ u.T
    return r, mc, rc
