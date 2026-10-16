"""Density maps: simulated maps, rigid-body fitting and projected "shadows".

Conventions
-----------
* Coordinates are Cartesian Å.  ``world`` coordinates are the superposed
  (aligned) coordinates used by the structural view; ``map`` coordinates are
  those of the density map file.
* A fit is a :class:`RigidTransform` taking world coordinates to map
  coordinates (``x_map = R @ x_world + t``).  The model never moves on screen:
  the map is drawn in the world frame through the inverse transform.
* Simulated maps follow ChimeraX ``molmap``: one 3D Gaussian per atom with
  amplitude proportional to the atomic number and width
  ``sigma = sigma_factor * resolution``; the grid spacing defaults to a third of
  the resolution.
* The fit score is the Pearson correlation ("correlation about the mean")
  between the simulated and the experimental map, evaluated at the simulated
  map's grid points above a fraction of its maximum (the model envelope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, optimize

from .io.mrc import DensityMap

SIGMA_FACTOR = 1.0 / (math.pi * math.sqrt(2.0))  # ChimeraX molmap default (~0.225)
ENVELOPE_FRACTION = 0.05  # simulated-map points scored: value >= fraction * max

ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12, "P": 15, "S": 16,
                 "CL": 17, "K": 19, "CA": 20, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
                 "ZN": 30, "SE": 34, "BR": 35, "RB": 37, "SR": 38, "CS": 55, "BA": 56}


class FitError(ValueError):
    pass


def atom_weights(elements: Sequence[str]) -> np.ndarray:
    """Atomic numbers for element symbols (unknown elements count as carbon)."""
    return np.array([ATOMIC_NUMBER.get(str(e).strip().upper(), 6) for e in elements],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Rigid transforms and rotations
# ---------------------------------------------------------------------------


@dataclass
class RigidTransform:
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "RigidTransform":
        v = np.asarray(values, dtype=np.float64)
        if v.size != 12:
            raise ValueError("a rigid transform needs 12 numbers (3x3 rotation + translation)")
        r = v[:9].reshape(3, 3)
        u, _s, vt = np.linalg.svd(r)  # re-orthonormalise rounded values
        r = u @ vt
        if np.linalg.det(r) < 0:
            raise ValueError("the rotation part of the transform is a reflection")
        return cls(r, v[9:].copy())

    def to_list(self, digits: int = 6) -> List[float]:
        return [round(float(x), digits) for x in np.concatenate([self.rotation.ravel(),
                                                                 self.translation])]

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=np.float64) @ self.rotation.T + self.translation

    def inverse(self) -> "RigidTransform":
        rt = self.rotation.T
        return RigidTransform(rt, -rt @ self.translation)

    def compose(self, first: "RigidTransform") -> "RigidTransform":
        """``self ∘ first`` (apply ``first``, then ``self``)."""
        return RigidTransform(self.rotation @ first.rotation,
                              self.rotation @ first.translation + self.translation)

    def about(self, centre: np.ndarray) -> "RigidTransform":
        """This transform's rotation applied about ``centre`` followed by its translation."""
        c = np.asarray(centre, dtype=np.float64)
        return RigidTransform(self.rotation, c - self.rotation @ c + self.translation)


def rotvec_matrix(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (axis * angle in radians)."""
    w = np.asarray(w, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    a, b, c, d = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def rotation_samples(step_deg: float) -> np.ndarray:
    """Near-uniform rotations of SO(3) with roughly ``step_deg`` spacing
    (super-Fibonacci spirals, Alexa 2022); returns (n, 3, 3)."""
    step = math.radians(max(float(step_deg), 3.0))
    n = max(24, int(math.ceil(8 * math.pi ** 2 / step ** 3)))
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = np.arange(n) + 0.5
    r = np.sqrt(s / n)
    big = np.sqrt(1.0 - s / n)
    alpha = 2 * math.pi * s / phi
    beta = 2 * math.pi * s / psi
    q = np.stack([r * np.sin(alpha), r * np.cos(alpha), big * np.sin(beta), big * np.cos(beta)], 1)
    return np.array([quaternion_matrix(x) for x in q])


def rotation_angle(r: np.ndarray) -> float:
    """Rotation angle (degrees) of a rotation matrix."""
    c = (np.trace(r) - 1.0) / 2.0
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))


# ---------------------------------------------------------------------------
# Simulated maps
# ---------------------------------------------------------------------------


def simulate_map(xyz: np.ndarray, weights: np.ndarray, resolution: float,
                 spacing: Optional[float] = None, sigma_factor: float = SIGMA_FACTOR,
                 pad_sigmas: float = 5.0) -> DensityMap:
    """Sum of atomic Gaussians on a cubic grid enclosing the atoms."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not len(xyz):
        raise FitError("no atoms to simulate a map from")
    if resolution <= 0:
        raise FitError("the resolution must be positive")
    sigma = sigma_factor * resolution
    h = float(spacing) if spacing else resolution / 3.0
    pad = pad_sigmas * sigma + h
    lo = np.floor((xyz.min(axis=0) - pad) / h) * h  # on whole voxels (CCP4 NSTART)
    n = np.ceil((xyz.max(axis=0) + pad - lo) / h).astype(int) + 1
    grid = np.zeros(tuple(n[::-1]), dtype=np.float64)  # [z, y, x]
    splat_gaussians(grid, (xyz - lo) / h, np.asarray(weights, dtype=np.float64), sigma / h,
                    pad_sigmas)
    return DensityMap(grid.astype(np.float32), lo, np.full(3, h))


def splat_gaussians(grid: np.ndarray, ijk: np.ndarray, weights: np.ndarray, sigma_vox: float,
                    cutoff: float = 5.0) -> None:
    """Add Gaussians (centres in fractional voxel indices x, y, z) to ``grid`` [z, y, x]."""
    rad = max(1, int(math.ceil(cutoff * sigma_vox)))
    off = np.arange(-rad, rad + 1)
    base = np.floor(ijk).astype(np.int64)
    frac = ijk - base
    shape_xyz = np.array(grid.shape[::-1])
    norm = 1.0 / (2.0 * sigma_vox * sigma_vox)
    # separable Gaussian factors per axis: (n_atoms, n_off)
    fx, fy, fz = (np.exp(-((off[None, :] - frac[:, k:k + 1]) ** 2) * norm) for k in range(3))
    for dz_i, dz in enumerate(off):
        z = base[:, 2] + dz
        okz = (z >= 0) & (z < shape_xyz[2])
        for dy_i, dy in enumerate(off):
            y = base[:, 1] + dy
            ok = okz & (y >= 0) & (y < shape_xyz[1])
            if not ok.any():
                continue
            amp = weights[ok] * fz[ok, dz_i] * fy[ok, dy_i]
            x = base[ok, 0][:, None] + off[None, :]
            okx = (x >= 0) & (x < shape_xyz[0])
            vals = amp[:, None] * fx[ok]
            zz = np.broadcast_to(z[ok][:, None], x.shape)
            yy = np.broadcast_to(y[ok][:, None], x.shape)
            np.add.at(grid, (zz[okx], yy[okx], x[okx]), vals[okx])


# ---------------------------------------------------------------------------
# Map sampling
# ---------------------------------------------------------------------------

_MAP_CACHE: dict = {}


def load_map_cached(path: str) -> DensityMap:
    """read_map with a small cache keyed by path, size and modification time."""
    import os

    from .io.mrc import read_map

    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    hit = _MAP_CACHE.get(key)
    if hit is None:
        hit = read_map(path)
        if len(_MAP_CACHE) >= 2:
            _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
        _MAP_CACHE[key] = hit
    return hit



def sample_map(dmap: DensityMap, xyz: np.ndarray, cval: float = 0.0, order: int = 1) -> np.ndarray:
    """Trilinear map values at Cartesian points (``cval`` outside the grid)."""
    ijk = dmap.xyz_to_index(xyz)
    coords = np.vstack([ijk[:, 2], ijk[:, 1], ijk[:, 0]])
    return ndimage.map_coordinates(dmap.data, coords, order=order, mode="constant", cval=cval,
                                   prefilter=False)


def map_level(dmap: DensityMap, sigmas: float) -> float:
    st = dmap.stats()
    return st["mean"] + sigmas * st["rms"]


def pearson(a: np.ndarray, b_centred: np.ndarray, b_norm: float) -> float:
    a = a - a.mean()
    na = float(np.sqrt(np.dot(a, a)))
    if na <= 0 or b_norm <= 0:
        return 0.0
    return float(np.dot(a, b_centred) / (na * b_norm))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    transform: RigidTransform  # world -> map
    cc: float
    start: str  # "manual", "global" or "identity"
    candidates: int = 1
    evaluations: int = 0
    message: str = ""


class FitProblem:
    """Correlation of a model (atoms in world coordinates) with an experimental map."""

    def __init__(self, dmap: DensityMap, xyz: np.ndarray, weights: np.ndarray, resolution: float,
                 sigma_factor: float = SIGMA_FACTOR, envelope: float = ENVELOPE_FRACTION,
                 max_points: int = 60000):
        if resolution <= 0:
            raise FitError("enter the map resolution (Å) for the simulated map")
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if len(xyz) < 3:
            raise FitError("at least three atoms are needed for a fit")
        self.dmap = dmap
        self.resolution = float(resolution)
        self.sigma_factor = sigma_factor
        self.xyz = xyz
        self.weights = np.asarray(weights, dtype=np.float64)
        self.centre = xyz.mean(axis=0)
        self.radius = float(np.linalg.norm(xyz - self.centre, axis=1).max())
        self.rg = max(float(np.sqrt(((xyz - self.centre) ** 2).sum(axis=1).mean())), 1.0)
        spacing = max(resolution / 3.0, float(np.min(dmap.voxel)) * 0.5)
        sim = simulate_map(xyz, self.weights, resolution, spacing, sigma_factor)
        self.sim = sim
        vals = sim.data.ravel().astype(np.float64)
        keep = np.flatnonzero(vals >= envelope * vals.max())
        if len(keep) > max_points:  # thin evenly; the envelope is smooth
            keep = keep[np.linspace(0, len(keep) - 1, max_points).astype(np.int64)]
        nz, ny, nx = sim.data.shape
        z, rem = np.divmod(keep, ny * nx)
        y, x = np.divmod(rem, nx)
        self.points = sim.index_to_xyz(np.stack([x, y, z], axis=1))  # world coordinates
        s = vals[keep]
        self.s_centred = s - s.mean()
        self.s_norm = float(np.sqrt(np.dot(self.s_centred, self.s_centred)))
        self.evaluations = 0

    # -- scoring
    def cc(self, transform: RigidTransform, points: Optional[np.ndarray] = None,
           s_centred: Optional[np.ndarray] = None, s_norm: Optional[float] = None) -> float:
        pts = self.points if points is None else points
        e = sample_map(self.dmap, transform.apply(pts))
        self.evaluations += 1
        if s_centred is None:
            return pearson(e, self.s_centred, self.s_norm)
        return pearson(e, s_centred, s_norm)

    def _subset(self, n: int):
        if len(self.points) <= n:
            return self.points, self.s_centred, self.s_norm
        idx = np.linspace(0, len(self.points) - 1, n).astype(np.int64)
        s = self.s_centred[idx] + 0.0
        s -= s.mean()
        return self.points[idx], s, float(np.sqrt(np.dot(s, s)))

    # -- local refinement
    def refine(self, start: RigidTransform, max_points: Optional[int] = None,
               xtol: float = 0.02, max_shift: Optional[float] = None) -> Tuple[RigidTransform, float]:
        """Maximise the correlation from ``start`` (Powell; 6 rigid-body parameters)."""
        pts, s_c, s_n = self._subset(max_points) if max_points else (None, None, None)
        c_map = start.apply(self.centre)
        rg = self.rg

        def make(p):
            rot = RigidTransform(rotvec_matrix(p[:3] / rg), p[3:].copy()).about(c_map)
            return rot.compose(start)

        def cost(p):
            if max_shift is not None and np.linalg.norm(p[3:]) > max_shift:
                return 1.0
            return -self.cc(make(p), pts, s_c, s_n)

        step = max(0.5, 0.25 * self.resolution)
        res = optimize.minimize(cost, np.zeros(6), method="Powell",
                                options={"xtol": xtol, "ftol": 1e-7, "maxfev": 4000,
                                         "direc": np.eye(6) * step})
        best = make(res.x)
        return best, self.cc(best)

    # -- global search
    def global_search(self, angular_step: float = 30.0, top: int = 12,
                      progress: Optional[Callable[[float, str], None]] = None,
                      cancelled: Optional[Callable[[], bool]] = None) -> FitResult:
        """Exhaustive rotations x FFT translations on a coarse grid, then local
        refinement of the best distinct poses."""
        rots = rotation_samples(angular_step)
        coarse = _CoarseSearch(self, angular_step)
        cands: List[Tuple[float, np.ndarray, np.ndarray]] = []
        n = len(rots)
        for k, r in enumerate(rots):
            if cancelled is not None and cancelled():
                raise FitError("cancelled")
            for score, centre in coarse.peaks(r, 2):
                cands.append((score, r, centre))
            if progress is not None and (k % 16 == 0 or k == n - 1):
                progress(0.7 * (k + 1) / n, f"global search: rotation {k + 1}/{n}")
        cands.sort(key=lambda c: -c[0])
        chosen: List[RigidTransform] = []
        min_shift = max(2.0, 0.5 * self.resolution)
        for score, r, centre in cands:
            t = RigidTransform(r, centre - r @ self.centre)
            if any(_pose_close(t, c, self.centre, min_shift, angular_step * 0.75) for c in chosen):
                continue
            chosen.append(t)
            if len(chosen) >= top:
                break
        if len(cands) > len(chosen):
            pass  # the remaining candidates are near-duplicates or lower ranked
        results = []
        for k, t in enumerate(chosen):
            if cancelled is not None and cancelled():
                raise FitError("cancelled")
            fit, cc = self.refine(t, max_points=4000, xtol=0.1)
            results.append((cc, fit))
            if progress is not None:
                progress(0.7 + 0.25 * (k + 1) / len(chosen), f"refining pose {k + 1}/{len(chosen)}")
        results.sort(key=lambda x: -x[0])
        finals = []
        for cc, fit in results[:3]:
            fit2, cc2 = self.refine(fit)
            finals.append((cc2, fit2))
        finals.sort(key=lambda x: -x[0])
        if progress is not None:
            progress(1.0, "global search done")
        best_cc, best = finals[0]
        return FitResult(best, best_cc, "global", candidates=len(chosen),
                         evaluations=self.evaluations,
                         message=f"{n} rotations ({angular_step:g}° step), "
                                 f"{len(chosen)} poses refined")


def _pose_close(a: RigidTransform, b: RigidTransform, centre: np.ndarray, shift: float,
                angle: float) -> bool:
    d = np.linalg.norm(a.apply(centre) - b.apply(centre))
    return d < shift and rotation_angle(a.rotation @ b.rotation.T) < angle


class _CoarseSearch:
    """FFT translation search for one rotation at a time on a coarse, padded grid."""

    def __init__(self, prob: FitProblem, angular_step: float):
        dmap = prob.dmap
        res = prob.resolution
        sigma = prob.sigma_factor * res
        extent = np.array(dmap.shape_xyz) * dmap.voxel
        # coarse spacing: about resolution/2, at least the map voxel, and <= ~96 points per axis
        h = max(res / 2.0, float(np.max(dmap.voxel)), float((extent + 2 * prob.radius).max()) / 96.0)
        pad = prob.radius + 3 * sigma
        lo = dmap.origin - pad
        n = np.ceil((extent + 2 * pad) / h).astype(int) + 1
        n = np.array([_fft_size(int(v)) for v in n])
        self.h, self.lo, self.n = h, lo, n
        # experimental map resampled onto the coarse grid after anti-alias smoothing
        smooth = ndimage.gaussian_filter(dmap.data.astype(np.float32),
                                         sigma=[0.5 * h / v for v in dmap.voxel[::-1]])
        sm = DensityMap(smooth, dmap.origin, dmap.voxel)
        gz, gy, gx = np.meshgrid(*(lo[k] + h * np.arange(n[k]) for k in (2, 1, 0)), indexing="ij")
        xyz = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vals = sample_map(sm, xyz, cval=np.nan).reshape(tuple(n[::-1]))
        inside = np.isfinite(vals)
        mean = float(np.nanmean(vals)) if inside.any() else 0.0
        e = np.where(inside, vals - mean, 0.0)
        self.e_ft = np.fft.rfftn(e)
        # translations whose model centre lies inside the original map box
        idx = [np.arange(n[k]) for k in range(3)]
        cent = [lo[k] + h * idx[k] for k in range(3)]
        ok = [(cent[k] >= dmap.origin[k]) & (cent[k] <= dmap.extent[k]) for k in range(3)]
        self.valid = ok[2][:, None, None] & ok[1][None, :, None] & ok[0][None, None, :]
        # Gaussian transfer function of the model blur (sigma) on the coarse grid
        k = [np.fft.fftfreq(n[2], d=h), np.fft.fftfreq(n[1], d=h), np.fft.rfftfreq(n[0], d=h)]
        k2 = k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2
        self.gauss = np.exp(-2.0 * math.pi ** 2 * sigma ** 2 * k2)
        self.rel = prob.xyz - prob.centre
        self.w = prob.weights

    def peaks(self, r: np.ndarray, count: int = 1):
        """Best (score, model-centre position in map coordinates) for rotation ``r``."""
        n = self.n
        g = (self.rel @ r.T) / self.h
        tmpl = np.zeros(tuple(n[::-1]))
        base = np.floor(g).astype(np.int64)
        f = g - base
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    np.add.at(tmpl, ((base[:, 2] + dz) % n[2], (base[:, 1] + dy) % n[1],
                                     (base[:, 0] + dx) % n[0]), self.w * wx * wy * wz)
        t_ft = np.fft.rfftn(tmpl) * self.gauss
        corr = np.fft.irfftn(np.conj(t_ft) * self.e_ft, s=tuple(n[::-1]), axes=(0, 1, 2))
        corr = np.where(self.valid, corr, -np.inf)
        out = []
        for _ in range(count):
            flat = int(np.argmax(corr))
            score = float(corr.flat[flat])
            if not np.isfinite(score):
                break
            z, y, x = np.unravel_index(flat, corr.shape)
            out.append((score, self.lo + self.h * np.array([x, y, z], dtype=np.float64)))
            # suppress the neighbourhood of this peak before the next one
            rad = 2
            corr[max(0, z - rad):z + rad + 1, max(0, y - rad):y + rad + 1,
                 max(0, x - rad):x + rad + 1] = -np.inf
        return out


def _fft_size(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n (fast FFT sizes)."""
    best = None
    p2 = 1
    while p2 < 2 * n:
        p3 = p2
        while p3 < 2 * n:
            p5 = p3
            while p5 < 2 * n:
                if p5 >= n and (best is None or p5 < best):
                    best = p5
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return best or n


def density_centroid(dmap: DensityMap, level: float) -> np.ndarray:
    """Centre (Å) of the map density above ``level`` (the whole map if none is)."""
    w = np.maximum(dmap.data.astype(np.float64) - level, 0)
    total = float(w.sum())
    if total <= 0:
        return dmap.origin + 0.5 * (np.array(dmap.shape_xyz) - 1) * dmap.voxel
    nz, ny, nx = w.shape
    cx = float((w.sum(axis=(0, 1)) * np.arange(nx)).sum()) / total
    cy = float((w.sum(axis=(0, 2)) * np.arange(ny)).sum()) / total
    cz = float((w.sum(axis=(1, 2)) * np.arange(nz)).sum()) / total
    return dmap.index_to_xyz(np.array([cx, cy, cz]))


def centre_transform(dmap: DensityMap, level: float, model_centre: np.ndarray) -> RigidTransform:
    """Placement without rotation that puts the model centre on the density centre."""
    return RigidTransform(np.eye(3), density_centroid(dmap, level) - np.asarray(model_centre, float))


def fit_model(dmap: DensityMap, xyz: np.ndarray, elements: Sequence[str], resolution: float,
              start: Optional[RigidTransform] = None, search: str = "global",
              angular_step: float = 30.0, progress=None, cancelled=None) -> FitResult:
    """Fit atoms (world coordinates) into ``dmap``.

    ``search="global"`` runs the exhaustive search; ``"manual"`` refines locally
    from ``start`` (the user's placement; identity when not given).
    """
    prob = FitProblem(dmap, xyz, atom_weights(elements), resolution)
    if search == "global":
        return prob.global_search(angular_step, progress=progress, cancelled=cancelled)
    t0 = start or RigidTransform.identity()
    cc0 = prob.cc(t0)
    fit, cc = prob.refine(t0)
    if progress is not None:
        progress(1.0, "local fit done")
    return FitResult(fit, cc, "manual" if start is not None else "identity",
                     evaluations=prob.evaluations, message=f"start CC {cc0:.3f}")


# ---------------------------------------------------------------------------
# Projected density ("shadow")
# ---------------------------------------------------------------------------


@dataclass
class Shadow:
    image: np.ndarray  # [rows (v, bottom->top), cols (u)] in [0, 1]
    extent: Tuple[float, float, float, float]  # u0, u1, v0, v1 in view coordinates (Å)


def project_density(dmap: DensityMap, world_to_map: RigidTransform, view: np.ndarray,
                    centre: np.ndarray, level: float, window: Optional[Tuple[float, float, float, float]] = None,
                    max_pixels: int = 220, max_depth: int = 220) -> Optional[Shadow]:
    """Integrate the map above ``level`` along the viewing direction.

    ``view`` (3x3) and ``centre`` define the structural view: view coordinates
    are ``(x_world - centre) @ view.T`` (x right, y up, z toward the viewer).
    Returns ``None`` when the map does not overlap the window.
    """
    to_map = world_to_map
    # map box corners in view coordinates
    lo, hi = dmap.origin, dmap.extent
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    vc = (to_map.inverse().apply(corners) - centre) @ view.T
    u0, v0, w0 = vc.min(axis=0)
    u1, v1, w1 = vc.max(axis=0)
    if window is not None:
        u0, u1 = max(u0, window[0]), min(u1, window[1])
        v0, v1 = max(v0, window[2]), min(v1, window[3])
    if u1 <= u0 or v1 <= v0:
        return None
    step = max(float(np.min(dmap.voxel)), max(u1 - u0, v1 - v0) / max_pixels,
               (w1 - w0) / max_depth)
    nu = max(2, int(math.ceil((u1 - u0) / step)) + 1)
    nv = max(2, int(math.ceil((v1 - v0) / step)) + 1)
    nw = max(2, int(math.ceil((w1 - w0) / step)) + 1)
    # view (u, v, w) -> map index (x, y, z): idx = A @ (u, v, w) + b
    m_view_to_world = view.T  # world = view.T @ uvw + centre
    r = to_map.rotation @ m_view_to_world
    off = to_map.apply(centre[None, :])[0]
    a_idx = r / dmap.voxel[:, None]
    b_idx = (off - dmap.origin) / dmap.voxel
    # output grid axes (w, v, u) with spacing `step` from (w0, v0, u0)
    # array axes of the map are (z, y, x): reorder rows and columns accordingly
    m = np.zeros((3, 3))
    for out_ax, view_ax in enumerate((2, 1, 0)):  # output axis -> view axis (w, v, u)
        for in_ax, map_ax in enumerate((2, 1, 0)):  # input axis -> map index axis (z, y, x)
            m[in_ax, out_ax] = a_idx[map_ax, view_ax] * step
    start = np.array([w0, v0, u0])
    view_start = np.array([start[2], start[1], start[0]])  # (u, v, w)
    idx0 = a_idx @ view_start + b_idx  # (x, y, z)
    offset = np.array([idx0[2], idx0[1], idx0[0]])
    data = np.maximum(dmap.data - np.float32(level), 0).astype(np.float32)
    vol = ndimage.affine_transform(data, m, offset=offset, output_shape=(nw, nv, nu), order=1,
                                   mode="constant", cval=0.0, prefilter=False)
    img = vol.sum(axis=0)
    peak = float(img.max())
    if peak <= 0:
        return None
    return Shadow((img / peak).astype(np.float32),
                  (u0 - 0.5 * step, u0 + (nu - 0.5) * step, v0 - 0.5 * step, v0 + (nv - 0.5) * step))


def shadow_rgba(shadow: Shadow, color_rgb: Sequence[float], opacity: float,
                gamma: float = 0.7) -> np.ndarray:
    """RGBA image (rows bottom->top) of a shadow in one colour with alpha ~ density."""
    a = np.clip(shadow.image, 0, 1) ** gamma * float(np.clip(opacity, 0, 1))
    out = np.zeros(shadow.image.shape + (4,), dtype=np.float32)
    out[..., :3] = np.asarray(color_rgb[:3], dtype=np.float32)
    out[..., 3] = a
    return out
