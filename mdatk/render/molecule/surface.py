"""Smooth molecular surface (like VMD QuickSurf), numpy / scipy only.

Every atom is splatted trilinearly onto a grid and blurred with a Gaussian, normalised
so an isolated atom peaks at 1.  The iso-surface is extracted with naive *surface nets*:
one vertex per grid cell that straddles the level (the mean of its edge crossings) and
one quad per grid edge that crosses it.  Normals are the trilinearly sampled density
gradient, so the mesh is smooth with about a third of the triangles of marching cubes.
The mesh does not depend on the view, so it is built once per frame.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

# cube corners as (dz, dy, dx) bits and the 12 cube edges between them
_CORNERS = np.array([(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)])
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1]


def density_grid(xyz: np.ndarray, spacing: float = 1.0, sigma: float = 1.4,
                 pad: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Summed unit Gaussians on a grid [z, y, x] with ``spacing`` (Å); returns grid, origin."""
    xyz = np.asarray(xyz, np.float64)
    h = float(spacing)
    pad = 3.2 * sigma + 1.5 if pad is None else pad
    lo = xyz.min(0) - pad
    nx, ny, nz = (np.ceil((xyz.max(0) + pad - lo) / h).astype(int) + 1)
    g = (xyz - lo) / h
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    idx, w = [], []
    for dz in (0, 1):
        wz = f[:, 2] if dz else 1 - f[:, 2]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dx in (0, 1):
                wx = f[:, 0] if dx else 1 - f[:, 0]
                idx.append(((i0[:, 2] + dz) * ny + (i0[:, 1] + dy)) * nx + (i0[:, 0] + dx))
                w.append(wx * wy * wz)
    vol = np.bincount(np.concatenate(idx), np.concatenate(w), minlength=nx * ny * nz)
    vol = vol.reshape(nz, ny, nx).astype(np.float32)
    s = sigma / h
    vol = ndimage.gaussian_filter(vol, s, truncate=3.0, mode="constant")
    vol *= np.float32((2 * np.pi * s * s) ** 1.5)
    return vol, lo


def surface_nets(vol: np.ndarray, level: float, origin: np.ndarray, spacing: float):
    """Iso-surface ``vol == level``: verts (n, 3) in Å (x, y, z), triangles (m, 3) and the
    vertices' fractional grid coordinates (z, y, x).  ``vol`` must be below ``level`` on
    its border (the padding of :func:`density_grid`)."""
    inside = vol >= level
    nz, ny, nx = vol.shape
    cz, cy, cx = nz - 1, ny - 1, nx - 1
    cv = [vol[dz:dz + cz, dy:dy + cy, dx:dx + cx] for dz, dy, dx in _CORNERS]
    cm = [inside[dz:dz + cz, dy:dy + cy, dx:dx + cx] for dz, dy, dx in _CORNERS]
    n_in = sum(m.astype(np.uint8) for m in cm)
    cells = np.flatnonzero(((n_in > 0) & (n_in < 8)).ravel())
    if not len(cells):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((3, 0))
    acc = np.zeros((len(cells), 3), np.float64)
    cnt = np.zeros(len(cells), np.float64)
    for a, b in _EDGES:
        ma, mb = cm[a].ravel()[cells], cm[b].ravel()[cells]
        x = np.flatnonzero(ma != mb)
        va, vb = cv[a].ravel()[cells[x]], cv[b].ravel()[cells[x]]
        t = (level - va) / (vb - va)
        acc[x] += _CORNERS[a] + t[:, None] * (_CORNERS[b] - _CORNERS[a])
        cnt[x] += 1
    local = acc / cnt[:, None]
    kz, rem = np.divmod(cells, cy * cx)
    ky, kx = np.divmod(rem, cx)
    gz, gy, gx = kz + local[:, 0], ky + local[:, 1], kx + local[:, 2]
    verts = np.stack([origin[0] + gx * spacing, origin[1] + gy * spacing, origin[2] + gz * spacing], 1)
    vid = np.full(cz * cy * cx, -1, np.int64)
    vid[cells] = np.arange(len(cells))
    vid = vid.reshape(cz, cy, cx)
    quads = []
    # every grid edge that crosses the level is shared by four cells: one quad
    e = inside[1:-1, 1:-1, :-1] != inside[1:-1, 1:-1, 1:]  # x edges
    z, y, x = np.nonzero(e)
    z, y = z + 1, y + 1
    q = np.stack([vid[z - 1, y - 1, x], vid[z - 1, y, x], vid[z, y, x], vid[z, y - 1, x]], 1)
    quads.append(np.where(inside[z, y, x][:, None], q, q[:, ::-1]))
    e = inside[1:-1, :-1, 1:-1] != inside[1:-1, 1:, 1:-1]  # y edges
    z, y, x = np.nonzero(e)
    z, x = z + 1, x + 1
    q = np.stack([vid[z - 1, y, x - 1], vid[z, y, x - 1], vid[z, y, x], vid[z - 1, y, x]], 1)
    quads.append(np.where(inside[z, y, x][:, None], q, q[:, ::-1]))
    e = inside[:-1, 1:-1, 1:-1] != inside[1:, 1:-1, 1:-1]  # z edges
    z, y, x = np.nonzero(e)
    y, x = y + 1, x + 1
    q = np.stack([vid[z, y - 1, x - 1], vid[z, y - 1, x], vid[z, y, x], vid[z, y, x - 1]], 1)
    quads.append(np.where(inside[z, y, x][:, None], q, q[:, ::-1]))
    q = np.concatenate(quads)
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int64)
    return verts, tris, np.stack([gz, gy, gx])


def gaussian_surface(xyz: np.ndarray, spacing: float = 1.0, sigma: float = 1.4, level: float = 0.5):
    """verts, outward normals, triangles (wound outward) and the nearest atom per vertex."""
    from scipy.spatial import cKDTree

    vol, lo = density_grid(xyz, spacing, sigma)
    verts, tris, coords = surface_nets(vol, level, lo, spacing)
    if not len(tris):
        return verts, np.zeros((0, 3)), tris, np.zeros(0, np.int64)
    grads = np.gradient(vol)  # d/dz, d/dy, d/dx
    nrm = -np.stack([ndimage.map_coordinates(grads[k], coords, order=1, mode="nearest") for k in (2, 1, 0)], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    near = cKDTree(np.asarray(xyz, np.float64)).query(verts)[1]
    # wind each triangle so its face normal agrees with the gradient normals
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    bad = (fn * nrm[tris].sum(1)).sum(1) < 0
    tris[bad] = tris[bad][:, ::-1]
    return verts, nrm, tris, near
