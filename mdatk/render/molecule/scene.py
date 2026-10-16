"""Representation scene: which residues each style draws, per-frame geometry and colours.

Renderer independent: :meth:`MoleculeScene.frame` returns a :class:`~.geometry.DrawList`
plus per-primitive RGBA colours, which a renderer turns into pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ...config import RepSpec
from .geometry import ROLE_NUCLEOTIDE, ROLE_SUGAR, STYLES, DrawList, Mesh, Spheres, Sticks
from .topology import NUCLEOTIDE_RGB, Topology


@dataclass
class Colored:
    """A primitive with its colours (RGB 0..1) and opacity."""

    prim: object  # Mesh, Spheres or Sticks
    rgb: np.ndarray  # per vertex (Mesh) or per primitive (Spheres / Sticks)
    opacity: float = 1.0


@dataclass
class Shading:
    """Per-render look shared by every renderer: depth cue and glow."""

    fog_rgb: np.ndarray  # colour distant parts fade toward (the panel surface)
    fog_strength: float = 0.0  # 0 = no depth cue
    glow: Optional[np.ndarray] = None  # per residue row, 0..1
    glow_rgb: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.69, 0.0]))


def depth_range(items: Sequence[Colored], axis: np.ndarray) -> "tuple[float, float]":
    """View-depth extent (Å along ``axis``) of every primitive, radii included, plus 1 Å."""
    lo, hi = np.inf, -np.inf
    for it in items:
        p = it.prim
        if isinstance(p, Mesh):
            pts, rad = [p.verts], 0.0
        elif isinstance(p, Spheres):
            pts, rad = [p.xyz], float(np.max(p.radius)) if len(p.radius) else 0.0
        else:
            pts, rad = [p.a, p.b], float(np.max(p.radius)) if len(p.radius) else 0.0
        for q in pts:
            if len(q):
                z = np.asarray(q, np.float64) @ axis
                lo, hi = min(lo, float(z.min()) - rad), max(hi, float(z.max()) + rad)
    if not np.isfinite(lo):
        return -1.0, 1.0
    return lo - 1.0, hi + 1.0


def residue_masks(specs: Sequence[RepSpec], member: np.ndarray, group_names: Sequence[str],
                  n_res: int) -> List[np.ndarray]:
    """Residue rows drawn by each representation (groups by name; [] = every residue)."""
    out = []
    for spec in specs:
        if not spec.groups:
            out.append(np.ones(n_res, dtype=bool))
            continue
        mask = np.zeros(n_res, dtype=bool)
        for name in spec.groups:
            if name in group_names and group_names.index(name) < len(member):
                mask |= member[group_names.index(name)]
        out.append(mask)
    return out


class MoleculeScene:
    def __init__(self, topo: Topology, specs: Sequence[RepSpec], masks: Sequence[np.ndarray]):
        self.topo = topo
        self.specs = list(specs)
        self.styles = []
        for spec, mask in zip(self.specs, masks):
            cls = STYLES.get(spec.style)
            if cls is None or not mask.any():
                self.styles.append(None)
                continue
            self.styles.append(cls(topo, mask, size=spec.size, sugars=spec.sugars))
        letters = topo.letter
        self.nuc_rgb = np.array([NUCLEOTIDE_RGB.get(x, NUCLEOTIDE_RGB["?"]) for x in letters])

    def frame(self, xyz: np.ndarray, residue_rgb: np.ndarray) -> List[Colored]:
        """Primitives of one frame with colours; ``residue_rgb`` is (n_res, 3)."""
        out: List[Colored] = []
        for spec, style in zip(self.specs, self.styles):
            if style is None:
                continue
            dl: DrawList = style.build(xyz)
            for m in dl.meshes:
                out.append(Colored(m, self._mesh_rgb(m, spec, residue_rgb), spec.opacity))
            for p in dl.sticks + dl.spheres:
                out.append(Colored(p, self._atom_rgb(p, spec, residue_rgb), spec.opacity))
        return out

    def _mesh_rgb(self, m: Mesh, spec: RepSpec, res_rgb: np.ndarray) -> np.ndarray:
        rgb = res_rgb[m.owner]
        if spec.color == "nucleotide":
            return self.nuc_rgb[m.owner]
        if spec.color in ("auto", "element"):
            base = m.role == ROLE_NUCLEOTIDE
            rgb = np.where(base[:, None], self.nuc_rgb[m.owner], rgb)
        sugar = m.role == ROLE_SUGAR
        if sugar.any():  # lighter tint of the residue colour
            rgb = np.where(sugar[:, None], 0.55 * rgb + 0.45, rgb)
        return rgb

    def _atom_rgb(self, p, spec: RepSpec, res_rgb: np.ndarray) -> np.ndarray:
        rgb = res_rgb[p.owner]
        if spec.color == "residue":
            return rgb
        if spec.color == "nucleotide":
            return self.nuc_rgb[p.owner]
        atom = p.atom
        has = atom >= 0
        if not has.any():
            return rgb
        el = np.zeros_like(rgb)
        el[has] = self.topo.element_rgb[atom[has]]
        if spec.color == "auto":  # carbons keep the residue colour so groups / glow stay visible
            carbon = np.zeros(len(atom), bool)
            carbon[has] = self.topo.is_carbon[atom[has]]
            use = has & ~carbon
        else:
            use = has
        return np.where(use[:, None], el, rgb)


def build_scene(atoms_result, trace_result, specs: Sequence[RepSpec], group_names: Sequence[str],
                backbone_atom: str = "P", rest: str = "trace") -> Optional[MoleculeScene]:
    """Scene from the cached ``atoms`` and ``trace`` results (None without atoms).  Residues
    no representation draws get a ``trace`` representation unless ``rest`` is "hide"."""
    if atoms_result is None or not specs:
        return None
    a, tr = atoms_result, trace_result
    xyz0 = np.asarray(a.arrays["coords"][0], dtype=np.float64)
    n_res = len(tr.arrays["resid"])
    topo = Topology(a.meta["atom_names"], a.meta["elements"], a.meta["resnames"], a.arrays["row"], n_res,
                    a.arrays["bonds"], xyz0, tr.arrays["chain"],
                    tr.arrays.get("resindex", np.arange(n_res)), backbone_atom)
    member = tr.arrays.get("group_member")
    if member is None:
        gi = np.asarray(tr.arrays["group_index"])
        member = np.array([gi == k for k in range(len(group_names))], dtype=bool).reshape(-1, n_res)
    specs = list(specs)
    masks = residue_masks(specs, np.asarray(member, dtype=bool), list(group_names), n_res)
    drawn = np.zeros(n_res, bool)
    for spec, mask in zip(specs, masks):
        if spec.style in STYLES:
            drawn |= mask
    if rest == "trace" and not drawn.all():
        specs.append(RepSpec(style="trace", color="residue"))
        masks.append(~drawn)
    return MoleculeScene(topo, specs, masks)


__all__ = ["Colored", "Shading", "MoleculeScene", "depth_range", "build_scene", "residue_masks", "Mesh",
           "Spheres", "Sticks"]
