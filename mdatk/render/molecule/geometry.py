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
``vdw``      van der Waals spheres (Bondi radii);
``surf``     a smooth Gaussian molecular surface;
``trace``    balls at the backbone points joined by thin sticks (the 2D trace in 3D).
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
    """Spline through the control atoms of whole backbone segments, swept with a profile
    over the stretches owned by the selected residues (so a selection does not change the
    shape, and one selected residue still gets its piece)."""

    def __init__(self, topo: Topology, res_mask: np.ndarray, kind: str, half_width: float,
                 half_height: float, n_sub: int = 8, k: int = 12):
        self.topo = topo
        self.prof, self.pn = profile(kind, half_width, half_height, k)
        self.k = len(self.prof)
        self.kind = kind
        self.parts = []
        for seg in topo.segments:
            if not res_mask[seg].any():
                continue
            a, d, owner = catmull_rom(len(seg), n_sub)
            keep = np.flatnonzero(res_mask[seg[owner]])
            for run in np.split(keep, np.flatnonzero(np.diff(keep) != 1) + 1):
                own = seg[owner[run]]
                if run[-1] + 1 < len(owner):  # meet the next stretch without a gap
                    run = np.append(run, run[-1] + 1)
                    own = np.append(own, own[-1])
                if len(run) < 2:
                    continue
                at = np.flatnonzero(run % n_sub == 0)
                at = at[res_mask[seg[run[at] // n_sub]]]  # samples at the selected control atoms
                self.parts.append((seg, a[run], d[run], own, seg[run[at] // n_sub], at))
        self.n_sub = n_sub

    def build(self, xyz: np.ndarray) -> "tuple[List[Mesh], Dict[int, np.ndarray]]":
        """Meshes, and the spline point at each residue's control atom (for rungs)."""
        topo = self.topo
        meshes, anchor = [], {}
        for seg, a, d, own, at_res, at in self.parts:
            p = xyz[topo.control[seg]]
            pos, tan = a @ p, d @ p
            side = None
            if np.all(topo.side[seg] >= 0):
                side = a @ (xyz[topo.side[seg]] - p)
            t, n, b = frames_along(pos, tan, side)
            v = pos[:, None, :] + self.prof[None, :, 0:1] * b[:, None, :] + self.prof[None, :, 1:2] * n[:, None, :]
            nn = self.pn[None, :, 0:1] * b[:, None, :] + self.pn[None, :, 1:2] * n[:, None, :]
            ns = len(pos)
            verts = np.concatenate([v.reshape(-1, 3), pos[[0, -1]]])
            normals = np.concatenate([nn.reshape(-1, 3), -t[[0]], t[[-1]]])
            owner = np.concatenate([np.repeat(own, self.k), own[[0, -1]]])
            meshes.append(Mesh(verts, normals, sweep_faces(ns, self.k), owner,
                               np.zeros(len(owner), dtype=np.int8)))
            for r, i in zip(at_res, at):
                anchor[int(r)] = pos[i]
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
        self.sug = (np.flatnonzero(sel[topo.sugar_res]) if (self.sugars and len(topo.sugar_res))
                    else np.zeros(0, np.int64))
        in_seg = np.zeros(topo.n_res, bool)
        for seg in topo.segments:
            in_seg[seg] = True
        self.lone = np.flatnonzero(sel & ~in_seg & (topo.control >= 0))

    def build(self, xyz):
        topo = self.topo
        s = self.size
        meshes, anchor = self.backbone.build(xyz)
        out = DrawList(meshes=meshes)
        joints, joint_owner, joint_r = [], [], []
        for r in self.lone:  # no backbone neighbours: a ball at the control atom
            anchor[int(r)] = xyz[topo.control[r]]
            joints.append(anchor[int(r)])
            joint_owner.append(int(r))
            joint_r.append(0.9 * s)
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
                    joint_r.append(0.42 * s)
        if a_pts:
            n = len(a_pts)
            out.sticks.append(Sticks(np.array(a_pts), np.array(b_pts), np.full(n, 0.42 * s),
                                     np.array(owners, np.int64), np.full(n, -1, np.int64)))
        if joints:
            n = len(joints)
            out.spheres.append(Spheres(np.array(joints), np.array(joint_r), np.array(joint_owner, np.int64),
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


class SurfStyle(Style):
    """Smooth molecular surface (like VMD QuickSurf): iso-surface of summed atomic Gaussians
    (surface nets, see :mod:`.surface`), coloured by the residue of the nearest atom."""

    name = "surf"
    sigma, iso = 1.4, 0.5  # an isolated atom's surface lies ~1.65 Å from its centre
    LARGE = 5000  # atoms above which the grid coarsens to 1.5 Å

    def setup(self):
        self.atoms = self._atoms()
        default = 1.5 if len(self.atoms) > self.LARGE else 1.0
        self.spacing = float(self.opts.get("spacing") or default)

    def build(self, xyz):
        from .surface import gaussian_surface

        at = self.atoms
        if not len(at):
            return DrawList()
        verts, normals, tris, near = gaussian_surface(xyz[at], self.spacing, self.sigma * self.size, self.iso)
        if not len(tris):
            return DrawList()
        owner = self.topo.row[at[near]]
        return DrawList(meshes=[Mesh(verts, normals, tris, owner, np.zeros(len(verts), np.int8))])


class TraceStyle(Style):
    """The 2D trace in the scene: a ball at each residue's backbone point and half-sticks
    to its neighbours along the backbone segments (so it hides, and is hidden by, the
    other representations by depth)."""

    name = "trace"

    def setup(self):
        topo = self.topo
        self.rows = np.flatnonzero(self.mask & (topo.anchor_atom >= 0))
        pairs = [np.stack([seg[:-1], seg[1:]], 1) for seg in topo.segments]
        pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
        self.pairs = pairs[self.mask[pairs[:, 0]] | self.mask[pairs[:, 1]]]

    def build(self, xyz):
        topo, s = self.topo, self.size
        out = DrawList()
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        if len(i):
            a, b = xyz[topo.control[i]], xyz[topo.control[j]]
            mid = 0.5 * (a + b)
            mi, mj = self.mask[i], self.mask[j]
            ends = np.concatenate([a[mi], b[mj]])
            n = len(ends)
            out.sticks.append(Sticks(ends, np.concatenate([mid[mi], mid[mj]]), np.full(n, 0.38 * s),
                                     np.concatenate([i[mi], j[mj]]), np.full(n, -1, np.int64)))
        r = self.rows
        out.spheres.append(Spheres(xyz[topo.anchor_atom[r]], np.full(len(r), 0.85 * s), r,
                                   np.full(len(r), -1, np.int64)))
        return out


STYLES = {cls.name: cls for cls in (CartoonStyle, RibbonStyle, TubeStyle, LicoriceStyle, VdwStyle, SurfStyle,
                                    TraceStyle)}
