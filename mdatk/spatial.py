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
    """Apply the minimum image convention to difference vectors (n, 3).  A triclinic box (Amber's truncated
    octahedron) goes through MDAnalysis, as the pairs of :func:`pairs_within` do: a pair found across its
    boundary gets the vector of the distance it was found at."""
    box = _usable_box(box)
    if box is None:
        return vectors
    if not np.allclose(box[3:], 90.0):
        if np.any(box[3:] <= 0.0) or np.any(box[3:] >= 180.0):
            return vectors  # no cell
        from MDAnalysis.lib.distances import minimize_vectors

        vectors = np.asarray(vectors)
        if vectors.dtype not in (np.float32, np.float64):
            vectors = vectors.astype(np.float64)
        return minimize_vectors(np.ascontiguousarray(vectors.reshape(-1, 3)), box).reshape(vectors.shape)
    lengths = box[:3]
    return vectors - lengths * np.round(vectors / lengths)


def collinear(points: np.ndarray, weights: Optional[np.ndarray] = None, tolerance: float = 1e-3) -> bool:
    """True for points that define no rotation when superposed: fewer than three, or all on one line (their
    extent across the line is below ``tolerance`` times their extent along it; coincident points count too)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return True
    w = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    centred = points - (w / w.sum()) @ points
    extents = np.linalg.eigvalsh((centred * w[:, None]).T @ centred)  # squared, ascending
    return bool(extents[2] <= 0.0 or extents[1] <= tolerance * tolerance * extents[2])


def kabsch(mobile: np.ndarray, reference: np.ndarray,
           weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal (optionally weighted) rotation superposing ``mobile`` onto ``reference``.

    Returns ``(R, mobile_center, reference_center)`` such that
    ``(x - mobile_center) @ R.T + reference_center`` maps mobile coordinates
    onto the reference.  Fewer than three points give the identity (translation only); points on one line
    (:func:`collinear`) are laid onto each other's line, and the rotation about that line is not defined.
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
