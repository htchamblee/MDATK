"""Representation geometry in numpy (renderer independent).

Every style turns one frame's coordinates into a few primitive lists:

* :class:`Mesh`: triangles with per-vertex normals,
* :class:`Spheres`: centres and radii,
* :class:`Sticks`: cylinders between two points.

Each primitive carries ``owner`` (the residue row it belongs to) and ``atom``
(an atom index or -1), so colours can be assigned per frame from the residue
colours (groups, proximity, glow) or from elements / nucleotides without
rebuilding the geometry.  Sizes are in Å.

Styles (``size`` scales the widths and radii):

``cartoon``  elliptical backbone through the control atoms, a rung from the
             backbone through C1' to the glycosidic N, and the base ring as a
             slab extruded ±0.45 Å (purines as the fused 9-atom perimeter);
``ribbon``   a flat band through the control atoms facing the bases;
``tube``     a round tube through the control atoms;
``licorice`` half-bond sticks and atom balls (radius 0.3 Å);
``vdw``      van der Waals spheres (Bondi radii).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .topology import Topology


@dataclass
class Mesh:
    verts: np.ndarray  # (n, 3)
    normals: np.ndarray  # (n, 3)
    tris: np.ndarray  # (t, 3)
    owner: np.ndarray  # (n,) residue row per vertex
    role: np.ndarray  # (n,) colour role per vertex: 0 residue, 1 nucleotide, 2 sugar tint


@dataclass
class Spheres:
    xyz: np.ndarray
    radius: np.ndarray
    owner: np.ndarray
    atom: np.ndarray  # atom index (element colour) or -1


@dataclass
class Sticks:
    a: np.ndarray
    b: np.ndarray
    radius: np.ndarray
    owner: np.ndarray
    atom: np.ndarray


@dataclass
class DrawList:
    meshes: List[Mesh] = field(default_factory=list)
    spheres: List[Spheres] = field(default_factory=list)
    sticks: List[Sticks] = field(default_factory=list)

    def extend(self, other: "DrawList") -> None:
        self.meshes += other.meshes
        self.spheres += other.spheres
        self.sticks += other.sticks


ROLE_RESIDUE, ROLE_NUCLEOTIDE, ROLE_SUGAR = 0, 1, 2

# ---------------------------------------------------------------------------
# splines and swept profiles
# ---------------------------------------------------------------------------


def catmull_rom(m: int, n_sub: int):
    """Linear maps from ``m`` control points to ``(m-1)*n_sub + 1`` samples of a uniform
    Catmull-Rom spline through them: positions ``A @ P``, derivatives ``D @ P``, and the
    control point each sample belongs to (colours switch half-way between residues)."""
    t = np.arange(n_sub) / n_sub
    basis = 0.5 * np.array([[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]], dtype=np.float64)
    w = np.stack([np.ones_like(t), t, t * t, t ** 3], 1) @ basis
    dw = np.stack([np.zeros_like(t), np.ones_like(t), 2 * t, 3 * t * t], 1) @ basis
    ns = (m - 1) * n_sub + 1
    a = np.zeros((ns, m + 2))
    d = np.zeros((ns, m + 2))
    owner = np.zeros(ns, dtype=np.int64)
    for i in range(m - 1):
        sl = slice(i * n_sub, (i + 1) * n_sub)
        a[sl, i:i + 4] = w
        d[sl, i:i + 4] = dw
        owner[sl] = np.where(t < 0.5, i, i + 1)
    a[-1, m] = 1.0
    d[-1, m - 1:m + 2] = (-0.5, 0.0, 0.5)
    owner[-1] = m - 1
    # phantom end points: p[-1] = 2 p0 - p1, p[m] = 2 p[m-1] - p[m-2]
    e = np.zeros((m + 2, m))
    e[1:m + 1] = np.eye(m)
    e[0, 0], e[0, 1] = 2.0, -1.0
    e[m + 1, m - 1], e[m + 1, m - 2] = 2.0, -1.0
    return a @ e, d @ e, owner


def profile(kind: str, half_width: float, half_height: float, k: int = 12):
    """Cross-section points (Å) and outward normals in the (binormal, normal) plane."""
    w, h = float(half_width), float(half_height)
    if kind == "ellipse":
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        pts = np.stack([w * np.cos(ang), h * np.sin(ang)], 1)
        nrm = np.stack([h * np.cos(ang), w * np.sin(ang)], 1)
        return pts, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    if kind == "band":  # flat ribbon with duplicated corners for crisp edges
        pts = np.array([[-w, -h], [w, -h], [w, -h], [w, h], [w, h], [-w, h], [-w, h], [-w, -h]], dtype=np.float64)
        nrm = np.array([[0, -1], [0, -1], [1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0]], dtype=np.float64)
        return pts, nrm
    raise ValueError(f"unknown profile {kind!r}")


def sweep_faces(ns: int, k: int, caps: bool = True) -> np.ndarray:
    """Triangles of ``ns`` rings of ``k`` vertices (outward winding), plus fan caps whose
    centres are the two extra vertices ``ns*k`` (start) and ``ns*k + 1`` (end)."""
    i = np.arange(ns - 1)[:, None]
    j = np.arange(k)[None, :]
    jn = (j + 1) % k
    a, b = i * k + j, i * k + jn
    c, d = (i + 1) * k + jn, (i + 1) * k + j
    tris = [np.stack([a, c, b], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)]
    if caps:
        jj = np.arange(k)
        tris.append(np.stack([np.full(k, ns * k), jj, (jj + 1) % k], 1))
        last = (ns - 1) * k
        tris.append(np.stack([np.full(k, ns * k + 1), last + (jj + 1) % k, last + jj], 1))
    return np.concatenate(tris).astype(np.int64)


def frames_along(pos: np.ndarray, tan: np.ndarray, side: Optional[np.ndarray]):
    """Unit tangent, normal (toward ``side`` or parallel-transported) and binormal."""
    t = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)
    if side is not None:
        n = side - (side * t).sum(1, keepdims=True) * t
        bad = np.linalg.norm(n, axis=1) < 1e-6
        if not bad.any():
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            return t, n, np.cross(t, n)
    # rotation-minimising frame
    n = np.zeros_like(t)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n[0] = ref - ref.dot(t[0]) * t[0]
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(t)):
        v = n[i - 1] - n[i - 1].dot(t[i]) * t[i]
        nv = np.linalg.norm(v)
        n[i] = v / nv if nv > 1e-9 else n[i - 1]
    return t, n, np.cross(t, n)


class Backbone:
    """Spline through the control atoms of the selected residues, swept with a profile."""

    def __init__(self, topo: Topology, res_mask: np.ndarray, kind: str, half_width: float,
                 half_height: float, n_sub: int = 8, k: int = 12):
        self.topo = topo
        self.prof, self.pn = profile(kind, half_width, half_height, k)
        self.k = len(self.prof)
        self.kind = kind
        self.parts = []
        for seg in topo.segments:
            seg = seg[res_mask[seg]]
            # keep only runs of consecutive residues of the original segment
            for run in np.split(seg, np.flatnonzero(np.diff(seg) != 1) + 1):
                if len(run) >= 2:
                    a, d, owner = catmull_rom(len(run), n_sub)
                    self.parts.append((run, a, d, owner))
        self.n_sub = n_sub

    def build(self, xyz: np.ndarray) -> "tuple[List[Mesh], Dict[int, np.ndarray]]":
        """Meshes, and the spline point at each residue's control atom (for rungs)."""
        topo = self.topo
        meshes, anchor = [], {}
        for run, a, d, owner in self.parts:
            p = xyz[topo.control[run]]
            pos, tan = a @ p, d @ p
            side = None
            if np.all(topo.side[run] >= 0):
                side = a @ (xyz[topo.side[run]] - p)
            t, n, b = frames_along(pos, tan, side)
            v = pos[:, None, :] + self.prof[None, :, 0:1] * b[:, None, :] + self.prof[None, :, 1:2] * n[:, None, :]
            nn = self.pn[None, :, 0:1] * b[:, None, :] + self.pn[None, :, 1:2] * n[:, None, :]
            ns = len(pos)
            verts = np.concatenate([v.reshape(-1, 3), pos[[0, -1]]])
            normals = np.concatenate([nn.reshape(-1, 3), -t[[0]], t[[-1]]])
            own = np.concatenate([np.repeat(run[owner], self.k), run[[0, -1]]])
            meshes.append(Mesh(verts, normals, sweep_faces(ns, self.k), own,
                               np.zeros(len(own), dtype=np.int8)))
            for i, r in enumerate(run):
                anchor[int(r)] = pos[i * self.n_sub]
        return meshes, anchor


def ring_slabs(ring: np.ndarray, half: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Closed slabs from ordered ring atoms ``ring`` (n, k, 3) extruded ±``half`` along the
    ring normal.  Returns verts, normals, triangles and the slab index of each vertex."""
    n, k, _ = ring.shape
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    centre = ring.mean(axis=1)
    if k == 9:  # purine: fan from the C4-C5 bond midpoint, which sees every perimeter edge
        centre = 0.5 * (ring[:, 3] + ring[:, 8])
    nrm = np.cross(ring, np.roll(ring, -1, axis=1)).sum(axis=1)  # Newell normal
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    h = half * nrm[:, None, :]
    top, bot = ring + h, ring - h
    edge = np.roll(ring, -1, axis=1) - ring
    side_n = np.cross(edge, nrm[:, None, :])
    side_n /= np.linalg.norm(side_n, axis=2, keepdims=True) + 1e-12
    side_v = np.stack([top, np.roll(top, -1, 1), np.roll(bot, -1, 1), bot], 2).reshape(n, 4 * k, 3)
    verts = np.concatenate([(centre + half * nrm)[:, None], top, (centre - half * nrm)[:, None], bot, side_v], 1)
    normals = np.concatenate([np.repeat(nrm[:, None], k + 1, 1), np.repeat(-nrm[:, None], k + 1, 1),
                              np.repeat(side_n, 4, axis=1)], 1)
    m = verts.shape[1]
    j = np.arange(k)
    top_t = np.stack([np.zeros(k, np.int64), 1 + j, 1 + (j + 1) % k], 1)
    bot_t = np.stack([np.full(k, k + 1), k + 2 + (j + 1) % k, k + 2 + j], 1)
    base = 2 * k + 2 + 4 * j
    side_t = np.concatenate([np.stack([base, base + 2, base + 1], 1), np.stack([base, base + 3, base + 2], 1)])
    one = np.concatenate([top_t, bot_t, side_t])
    tris = (one[None] + (np.arange(n) * m)[:, None, None]).reshape(-1, 3)
    return verts.reshape(-1, 3), normals.reshape(-1, 3), tris, np.repeat(np.arange(n), m)


# ---------------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------------


class Style:
    """Builds one style's primitives for the residues in ``res_mask``."""

    name = ""

    def __init__(self, topo: Topology, res_mask: np.ndarray, size: float = 1.0, **opts):
        self.topo = topo
        self.mask = np.asarray(res_mask, dtype=bool)
        self.size = float(size)
        self.opts = opts
        self.setup()

    def setup(self) -> None:
        pass

    def build(self, xyz: np.ndarray) -> DrawList:  # pragma: no cover - abstract
        raise NotImplementedError

    def _atoms(self) -> np.ndarray:
        """Atoms of the selected residues (hydrogens only when the analysis kept them)."""
        return np.flatnonzero(self.mask[self.topo.row])


class TubeStyle(Style):
    name = "tube"
    kind, half_width, half_height = "ellipse", 0.45, 0.45

    def setup(self):
        s = self.size
        self.backbone = Backbone(self.topo, self.mask, self.kind, self.half_width * s, self.half_height * s,
                                 n_sub=int(self.opts.get("samples", 8)))

    def build(self, xyz):
        meshes, _anchor = self.backbone.build(xyz)
        return DrawList(meshes=meshes)


class RibbonStyle(TubeStyle):
    name = "ribbon"
    kind, half_width, half_height = "band", 1.2, 0.12


class CartoonStyle(Style):
    """New Cartoon for nucleic acids: backbone + rungs + base slabs (+ optional sugar slabs)."""

    name = "cartoon"

    def setup(self):
        s = self.size
        topo = self.topo
        self.backbone = Backbone(topo, self.mask, "ellipse", 1.6 * s, 0.45 * s,
                                 n_sub=int(self.opts.get("samples", 8)))
        sel = self.mask
        self.pur = np.flatnonzero(sel[topo.purine_res]) if len(topo.purine_res) else np.zeros(0, np.int64)
        self.pyr = np.flatnonzero(sel[topo.pyrimidine_res]) if len(topo.pyrimidine_res) else np.zeros(0, np.int64)
        self.sugars = bool(self.opts.get("sugars", False))
        self.sug = np.flatnonzero(sel[topo.sugar_res]) if (self.sugars and len(topo.sugar_res)) else np.zeros(0, np.int64)

    def build(self, xyz):
        topo = self.topo
        s = self.size
        meshes, anchor = self.backbone.build(xyz)
        out = DrawList(meshes=meshes)
        # base slabs, coloured by nucleotide (role 1)
        for rows, ring_idx in ((topo.purine_res[self.pur], topo.purine_ring[self.pur]),
                               (topo.pyrimidine_res[self.pyr], topo.pyrimidine_ring[self.pyr])):
            if len(rows):
                v, n, t, slab = ring_slabs(xyz[ring_idx], 0.45 * s)
                out.meshes.append(Mesh(v, n, t, rows[slab], np.full(len(v), ROLE_NUCLEOTIDE, np.int8)))
        if len(self.sug):
            rows = topo.sugar_res[self.sug]
            v, n, t, slab = ring_slabs(xyz[topo.sugar_ring[self.sug]], 0.25 * s)
            out.meshes.append(Mesh(v, n, t, rows[slab], np.full(len(v), ROLE_SUGAR, np.int8)))
        # rungs: backbone spline point -> C1' (or C4' with sugar slabs) -> glycosidic atom
        a_pts, b_pts, owners = [], [], []
        joints, joint_owner = [], []
        for rows, gly in ((topo.purine_res[self.pur], topo.purine_gly[self.pur]),
                          (topo.pyrimidine_res[self.pyr], topo.pyrimidine_gly[self.pyr])):
            for r, g in zip(rows, gly):
                r = int(r)
                knee = topo.c4[r] if self.sugars else topo.c1[r]
                if r not in anchor or knee < 0:
                    continue
                if self.sugars:
                    a_pts += [anchor[r]]
                    b_pts += [xyz[knee]]
                    owners += [r]
                    c1 = topo.c1[r]
                    if c1 >= 0:
                        a_pts.append(xyz[c1])
                        b_pts.append(xyz[g])
                        owners.append(r)
                else:
                    a_pts += [anchor[r], xyz[knee]]
                    b_pts += [xyz[knee], xyz[g]]
                    owners += [r, r]
                    joints.append(xyz[knee])
                    joint_owner.append(r)
        if a_pts:
            n = len(a_pts)
            out.sticks.append(Sticks(np.array(a_pts), np.array(b_pts), np.full(n, 0.42 * s),
                                     np.array(owners, np.int64), np.full(n, -1, np.int64)))
        if joints:
            n = len(joints)
            out.spheres.append(Spheres(np.array(joints), np.full(n, 0.42 * s), np.array(joint_owner, np.int64),
                                       np.full(n, -1, np.int64)))
        return out


class LicoriceStyle(Style):
    name = "licorice"

    def setup(self):
        topo = self.topo
        self.atoms = self._atoms()
        sel = np.zeros(len(topo.row), bool)
        sel[self.atoms] = True
        b = topo.bonds
        # bonds with at least one selected atom are drawn from the selected side, so licorice
        # joins a neighbouring style (e.g. the cartoon at the phosphate)
        self.bonds = b[sel[b[:, 0]] | sel[b[:, 1]]] if len(b) else np.zeros((0, 2), np.int64)
        self.sel = sel

    def build(self, xyz):
        topo = self.topo
        r = 0.3 * self.size
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        mid = 0.5 * (xyz[i] + xyz[j])
        keep_i, keep_j = self.sel[i], self.sel[j]
        a = np.concatenate([xyz[i][keep_i], xyz[j][keep_j]])
        b = np.concatenate([mid[keep_i], mid[keep_j]])
        atom = np.concatenate([i[keep_i], j[keep_j]])
        out = DrawList()
        if len(a):
            out.sticks.append(Sticks(a, b, np.full(len(a), r), topo.row[atom], atom))
        at = self.atoms
        out.spheres.append(Spheres(xyz[at], np.full(len(at), r), topo.row[at], at))
        return out


class VdwStyle(Style):
    name = "vdw"

    def setup(self):
        self.atoms = self._atoms()

    def build(self, xyz):
        at = self.atoms
        return DrawList(spheres=[Spheres(xyz[at], self.topo.radius[at] * self.size, self.topo.row[at], at)])


# ---------------------------------------------------------------------------
# molecular surface: Gaussian density + marching tetrahedra
# ---------------------------------------------------------------------------

# cube corners (x, y, z offsets) and the six tetrahedra around the 0-6 diagonal
_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_TETS = np.array([[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]])


def gaussian_density(xyz: np.ndarray, spacing: float, sigma: float, pad: float):
    """Sum of unit Gaussians (an isolated atom peaks at 1) on a grid; returns grid [z, y, x], origin."""
    from scipy import ndimage

    lo = xyz.min(axis=0) - pad
    shape = np.ceil((xyz.max(axis=0) + pad - lo) / spacing).astype(int) + 2  # x, y, z
    g = (xyz - lo) / spacing
    b = np.floor(g).astype(np.int64)
    f = g - b
    nx, ny, nz = shape
    flat = np.zeros(nx * ny * nz)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                idx = ((b[:, 2] + dz) * ny + (b[:, 1] + dy)) * nx + (b[:, 0] + dx)
                flat += np.bincount(idx, wx * wy * wz, minlength=flat.size)
    s = sigma / spacing
    grid = ndimage.gaussian_filter(flat.reshape(nz, ny, nx), s, mode="constant", truncate=3.0)
    grid *= (2 * np.pi) ** 1.5 * s ** 3
    return grid, lo


def marching_tetrahedra(grid: np.ndarray, iso: float, origin: np.ndarray, spacing: float):
    """Iso-surface of ``grid`` [z, y, x] as unshared triangles: verts (3t, 3), normals pointing
    to lower values (outward for a density), tris (t, 3)."""
    v = grid
    nz, ny, nx = v.shape
    inside = v > iso
    # cubes whose corners are not all on one side
    corner_in = np.stack([inside[dz:nz - 1 + dz, dy:ny - 1 + dy, dx:nx - 1 + dx]
                          for dx, dy, dz in _CORNERS], axis=-1)
    n_in = corner_in.sum(-1)
    cz, cy, cx = np.nonzero((n_in > 0) & (n_in < 8))
    if not len(cz):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    cube = np.stack([cx, cy, cz], 1)
    pos = cube[:, None, :] + _CORNERS[None]  # (c, 8, 3) grid indices (x, y, z)
    val = v[pos[..., 2], pos[..., 1], pos[..., 0]]  # (c, 8)
    tp = pos[:, _TETS].reshape(-1, 4, 3)  # (c*6, 4, 3)
    tv = val[:, _TETS].reshape(-1, 4)
    tin = tv > iso
    k = tin.sum(1)
    tris = []

    def cut(a_pos, a_val, b_pos, b_val):
        t = (iso - a_val) / np.where(np.abs(b_val - a_val) > 1e-12, b_val - a_val, 1e-12)
        return a_pos + t[:, None] * (b_pos - a_pos)

    # one corner on its own (in or out): one triangle
    for lone_in in (True, False):
        sel = np.flatnonzero(k == (1 if lone_in else 3))
        if len(sel):
            flags = tin[sel] if lone_in else ~tin[sel]
            a = np.argmax(flags, axis=1)
            others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[a]
            P, V = tp[sel].astype(np.float64), tv[sel]
            r = np.arange(len(sel))
            pa, va = P[r, a], V[r, a]
            tri = np.stack([cut(pa, va, P[r, others[:, j]], V[r, others[:, j]]) for j in range(3)], 1)
            tris.append(tri)
    # two in, two out: a quad (two triangles)
    sel = np.flatnonzero(k == 2)
    if len(sel):
        order = np.argsort(~tin[sel], axis=1, kind="stable")  # the two inside corners first
        P, V = tp[sel].astype(np.float64), tv[sel]
        r = np.arange(len(sel))
        i0, i1, o0, o1 = (order[:, j] for j in range(4))
        q = [cut(P[r, i0], V[r, i0], P[r, o0], V[r, o0]), cut(P[r, i0], V[r, i0], P[r, o1], V[r, o1]),
             cut(P[r, i1], V[r, i1], P[r, o1], V[r, o1]), cut(P[r, i1], V[r, i1], P[r, o0], V[r, o0])]
        tris.append(np.stack([q[0], q[1], q[2]], 1))
        tris.append(np.stack([q[0], q[2], q[3]], 1))
    tri = np.concatenate(tris)  # (t, 3, 3) in grid index units (x, y, z)
    verts = tri.reshape(-1, 3)
    # outward normals: minus the density gradient (trilinear interpolation of np.gradient)
    from scipy import ndimage

    gz, gy, gx = np.gradient(v)
    coords = np.vstack([verts[:, 2], verts[:, 1], verts[:, 0]])
    grad = np.stack([ndimage.map_coordinates(g, coords, order=1, mode="nearest") for g in (gx, gy, gz)], 1)
    normals = -grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12)
    # wind every triangle so its face normal agrees with the vertex normals
    t3 = verts.reshape(-1, 3, 3)
    face = np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0])
    flip = (face * normals.reshape(-1, 3, 3).sum(1)).sum(1) < 0
    ids = np.arange(len(verts)).reshape(-1, 3)
    ids[flip] = ids[flip][:, [0, 2, 1]]
    return origin + verts * spacing, normals, ids


class SurfStyle(Style):
    """Smooth molecular surface (like VMD QuickSurf): iso-surface of summed atomic Gaussians,
    coloured by the residue of the nearest atom."""

    name = "surf"
    sigma, iso = 1.4, 0.5  # an isolated atom's surface lies ~1.65 Å from its centre

    def setup(self):
        self.atoms = self._atoms()
        self.spacing = float(self.opts.get("spacing", 1.0))

    def build(self, xyz):
        from scipy.spatial import cKDTree

        at = self.atoms
        if not len(at):
            return DrawList()
        p = xyz[at]
        sigma = self.sigma * self.size
        grid, lo = gaussian_density(p, self.spacing, sigma, pad=3 * sigma + self.spacing)
        verts, normals, tris = marching_tetrahedra(grid, self.iso, lo, self.spacing)
        if not len(verts):
            return DrawList()
        _d, near = cKDTree(p).query(verts)
        owner = self.topo.row[at[near]]
        return DrawList(meshes=[Mesh(verts, normals, tris, owner, np.zeros(len(verts), np.int8))])


STYLES = {cls.name: cls for cls in (CartoonStyle, RibbonStyle, TubeStyle, LicoriceStyle, VdwStyle, SurfStyle)}
