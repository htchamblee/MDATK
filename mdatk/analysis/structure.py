"""Internal analyses feeding the structural view panel.

``trace``
    Backbone trace points per residue (first available atom of
    ``visual.trace_atoms``, e.g. C4' or P; residues without one use their heavy
    atom centroid) and the selected ions, optionally superposed onto the first
    analysed frame so the camera orientation chosen on that frame holds for the
    whole movie.
``proximity``
    For every trace residue the minimum heavy-atom distance to the proximity
    set, per frame (capped at ``analyses.proximity_search_radius``).  The
    render-time proximity cutoff can be changed freely below that radius
    without recomputation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..selections import SelectionError, heavy, resolve_selection, split_csv
from ..spatial import kabsch, pairs_within
from .base import Analysis, register
from .common import labels

LINK_MAX = 9.0  # max trace-point distance (A) for drawing a backbone link


def structure_groups(project) -> List[str]:
    names = project.visual.structure_groups or project.group_names()
    known = set(project.group_names())
    return [n for n in names if n in known]


def structure_selection_text(project) -> str:
    names = structure_groups(project)
    if not names:
        raise SelectionError("no residue groups available for the structural view")
    return " or ".join(f"group:{n}" for n in names)


def structure_residues(ctx):
    """Residues shown in the structural view (ions are drawn separately)."""
    project = ctx.project
    atoms = ctx.select(structure_selection_text(project))
    ions = ctx.select(project.analyses.mg.mg_selection, allow_empty=True)
    if ions.n_atoms:
        atoms = atoms[~np.isin(atoms.ix, ions.ix)]
    uniq = np.unique(atoms.resindices)
    return ctx.u.residues[uniq], atoms


def chain_ids(residues) -> np.ndarray:
    """Integer chain index per residue (chainID, segid, then bonded fragments)."""
    n = len(residues)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    first_atoms = residues.atoms[np.unique(residues.atoms.resindices, return_index=True)[1]]
    try:
        cid = np.asarray(first_atoms.chainIDs)
        if len(set(cid)) > 1:
            return np.unique(cid, return_inverse=True)[1].astype(np.int64)
    except (AttributeError, Exception):
        pass
    segs = np.asarray(residues.segids)
    if len(set(segs)) > 1:
        return np.unique(segs, return_inverse=True)[1].astype(np.int64)
    try:
        frag = np.asarray(first_atoms.fragindices)
        return np.unique(frag, return_inverse=True)[1].astype(np.int64)
    except Exception:
        return np.zeros(n, dtype=np.int64)


def residue_group_index(ctx, residues) -> np.ndarray:
    """Index of the first project group containing each residue (-1: none)."""
    out = np.full(len(residues), -1, dtype=np.int64)
    for gi, g in enumerate(ctx.project.groups):
        try:
            ag = ctx.select(f"group:{g.name}", allow_empty=True)
        except SelectionError:
            continue
        hit = np.isin(residues.ix, np.unique(ag.resindices)) & (out < 0)
        out[hit] = gi
    return out


class TracePoints:
    """One point per residue: the first atom of ``trace_atoms`` present, else the
    heavy-atom centroid.  Shared by the trace and atoms analyses so both superpose
    frames identically."""

    def __init__(self, ctx, residues, trace_atoms: str):
        priority = split_csv(trace_atoms) or ["C4'", "CA"]
        point_atoms, centroid_rows, centroid_atoms = [], [], []
        for row, res in enumerate(residues):
            names = {a.name: a.ix for a in res.atoms}
            chosen = next((names[n] for n in priority if n in names), None)
            if chosen is None:
                hv = heavy(res.atoms)
                centroid_rows.append(row)
                centroid_atoms.append(hv.ix if hv.n_atoms else res.atoms.ix)
                point_atoms.append(-1)
            else:
                point_atoms.append(chosen)
        self.n = len(residues)
        self.point_atoms = np.array(point_atoms, dtype=np.int64)
        self.is_point = self.point_atoms >= 0
        self.points = ctx.u.atoms[self.point_atoms[self.is_point]]
        self.centroid_rows = np.array(centroid_rows, dtype=np.int64)
        self.centroid_groups = [ctx.u.atoms[ix] for ix in centroid_atoms]

    def positions(self) -> np.ndarray:
        xyz = np.zeros((self.n, 3))
        xyz[self.is_point] = self.points.positions
        for row, ag in zip(self.centroid_rows, self.centroid_groups):
            xyz[row] = ag.positions.mean(axis=0)
        return xyz


@register
class TraceAnalysis(Analysis):
    name = "trace"
    label = "Backbone trace"
    internal = True
    version = 2  # v2: residue numbers for tags

    @classmethod
    def params_for(cls, project):
        v = project.visual
        return {"groups": structure_groups(project), "trace_atoms": v.trace_atoms,
                "align": v.align_frames, "show_ions": v.show_ions,
                "ions": project.analyses.mg.mg_selection if v.show_ions else "",
                "group_order": project.group_names()}

    @classmethod
    def selection_texts(cls, project):
        return structure_groups(project) + [project.analyses.mg.mg_selection]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        project = self.project
        residues, _ = structure_residues(self.ctx)
        if len(residues) == 0:
            raise SelectionError("structural view: no residues selected")
        self.residues = residues
        self.tracer = TracePoints(self.ctx, residues, project.visual.trace_atoms)
        self.is_point = self.tracer.is_point
        self.ions = (self.ctx.select(project.analyses.mg.mg_selection, allow_empty=True)
                     if project.visual.show_ions else self.ctx.u.atoms[[]])
        self.align = project.visual.align_frames
        n_res = len(residues)
        self.coords = np.zeros((self.n_frames, n_res, 3), dtype=np.float32)
        self.ion_coords = np.zeros((self.n_frames, self.ions.n_atoms, 3), dtype=np.float32)
        self.ref = None
        self.chain = chain_ids(residues)
        self.group_index = residue_group_index(self.ctx, residues)
        self.links = np.zeros((0, 2), dtype=np.int64)

    def process(self, k, ts):
        n_res = len(self.residues)
        xyz = self.tracer.positions()
        ions = self.ions.positions.astype(np.float64) if self.ions.n_atoms else np.zeros((0, 3))
        if self.ref is None:
            self.ref = xyz.copy()
            self._links(xyz)
        elif self.align and n_res >= 3:
            r, mc, rc = kabsch(xyz, self.ref)
            xyz = (xyz - mc) @ r.T + rc
            if len(ions):
                ions = (ions - mc) @ r.T + rc
        self.coords[k] = xyz
        if len(ions):
            self.ion_coords[k] = ions

    def _links(self, xyz):
        n = len(xyz)
        if n < 2:
            return
        i = np.arange(n - 1)
        j = i + 1
        same_chain = self.chain[i] == self.chain[j]
        consecutive = (self.residues.resindices[j] - self.residues.resindices[i]) == 1
        both_points = self.is_point[i] & self.is_point[j]
        close = np.linalg.norm(xyz[j] - xyz[i], axis=1) <= LINK_MAX
        ok = same_chain & consecutive & both_points & close
        self.links = np.stack([i[ok], j[ok]], axis=1)

    def finalize(self):
        res = self.new_result(residue_labels=labels(self.residues),
                              ion_labels=[f"{a.resname}{a.resid}" for a in self.ions],
                              group_names=self.project.group_names())
        res.arrays.update(coords=self.coords, ion_coords=self.ion_coords,
                          chain=self.chain, group_index=self.group_index,
                          is_point=self.is_point, links=self.links,
                          resindex=self.residues.ix.astype(np.int64),
                          resid=self.residues.resids.astype(np.int64))
        return res


@register
class ProximityAnalysis(Analysis):
    name = "proximity"
    label = "Proximity to residue set"
    internal = True

    @classmethod
    def expected_outputs(cls, project):
        return {"series": [], "matrices": ["min_distance"], "pairmaps": []}

    @classmethod
    def params_for(cls, project):
        return {"groups": structure_groups(project), "set": project.visual.proximity_set,
                "radius": project.analyses.proximity_search_radius,
                "ions": project.analyses.mg.mg_selection}

    @classmethod
    def selection_texts(cls, project):
        return structure_groups(project) + [project.visual.proximity_set,
                                            project.analyses.mg.mg_selection]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        project = self.project
        residues, atoms = structure_residues(self.ctx)
        self.residues = residues
        text = resolve_selection(project.visual.proximity_set, project.groups, self.ctx.u)
        self.set_atoms = heavy(self.ctx.select(f"sel:{text}"))
        members = np.isin(residues.ix, np.unique(self.set_atoms.resindices))
        self.member = members
        struct = heavy(atoms)
        struct = struct[~np.isin(struct.resindices, residues.ix[members])]
        self.struct = struct
        self.row = np.searchsorted(residues.ix, struct.resindices)
        self.radius = project.analyses.proximity_search_radius
        self.dist = np.full((len(residues), self.n_frames), np.inf, dtype=np.float32)

    def process(self, k, ts):
        if self.struct.n_atoms == 0:
            return
        i, _, d = pairs_within(self.struct.positions, self.set_atoms.positions, self.radius,
                               self.ctx.box(ts))
        if len(i):
            best = np.full(len(self.residues), np.inf)
            np.minimum.at(best, self.row[i], d)
            self.dist[:, k] = best
        self.dist[self.member, k] = 0.0

    def finalize(self):
        res = self.new_result(residue_labels=labels(self.residues), radius=self.radius)
        res.arrays.update(min_distance=self.dist, member=self.member,
                          resindex=self.residues.ix.astype(np.int64))
        res.add_matrix("min_distance", "min_distance", "residue_labels",
                       "Distance to proximity set", "value", unit="Å")
        return res


@register
class AtomsAnalysis(Analysis):
    """Per-atom coordinates of the structural-view residues for 3D representations
    and density fitting, superposed exactly like the ``trace`` analysis (same trace
    points, same reference frame), plus the topology needed to draw them."""

    name = "atoms"
    label = "Atomic coordinates (structure view)"
    internal = True
    version = 1

    @classmethod
    def params_for(cls, project):
        v = project.visual
        return {"groups": structure_groups(project), "trace_atoms": v.trace_atoms,
                "align": v.align_frames, "hydrogens": v.rep_hydrogens,
                "ions": project.analyses.mg.mg_selection, "group_order": project.group_names()}

    @classmethod
    def selection_texts(cls, project):
        return structure_groups(project) + [project.analyses.mg.mg_selection]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        residues, atoms = structure_residues(self.ctx)
        if len(residues) == 0:
            raise SelectionError("structural view: no residues selected")
        if not self.project.visual.rep_hydrogens:
            atoms = heavy(atoms)
        atoms = atoms[np.argsort(atoms.ix, kind="stable")]
        self.residues = residues
        self.atoms = atoms
        self.tracer = TracePoints(self.ctx, residues, self.project.visual.trace_atoms)
        self.align = self.project.visual.align_frames
        self.ref = None
        self.coords = np.zeros((self.n_frames, atoms.n_atoms, 3), dtype=np.float32)
        self.row = np.searchsorted(residues.ix, atoms.resindices).astype(np.int64)
        self.bonds = atom_bonds(atoms)

    def process(self, k, ts):
        xyz = self.atoms.positions.astype(np.float64)
        pts = self.tracer.positions()
        if self.ref is None:
            self.ref = pts
        elif self.align and len(pts) >= 3:
            r, mc, rc = kabsch(pts, self.ref)
            xyz = (xyz - mc) @ r.T + rc
        self.coords[k] = xyz

    def finalize(self):
        a = self.atoms
        try:
            elements = [str(e) for e in a.elements]
        except Exception:
            elements = [guess_element(n) for n in a.names]
        res = self.new_result(atom_names=[str(n) for n in a.names], elements=elements,
                              resnames=[str(r) for r in a.resnames])
        res.arrays.update(coords=self.coords, row=self.row, bonds=self.bonds,
                          resid=a.resids.astype(np.int64), atom_index=a.ix.astype(np.int64),
                          resindex=self.residues.ix.astype(np.int64))
        return res


def guess_element(name: str) -> str:
    name = name.strip().lstrip("0123456789")
    for two in ("MG", "NA", "CL", "ZN", "CA"):
        if name.upper().startswith(two) and len(name) == len(two):
            return two.capitalize()
    return name[:1].upper() if name else "X"


def atom_bonds(atoms, max_len: float = 1.95) -> np.ndarray:
    """Bonds among ``atoms`` as local index pairs: from the topology when it has
    bonds, otherwise guessed from first-frame distances within each residue and
    between consecutive residues (O3'-P)."""
    index = {int(ix): k for k, ix in enumerate(atoms.ix)}
    pairs = []
    try:
        for b in atoms.bonds:
            i, j = (int(x) for x in b.indices)
            if i in index and j in index:
                pairs.append((index[i], index[j]))
    except Exception:  # topology without bonds
        pairs = []
    if pairs:
        out = np.array(sorted({(min(p), max(p)) for p in pairs}), dtype=np.int64)
        return out.reshape(-1, 2)
    xyz = atoms.positions.astype(np.float64)
    if len(xyz) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    i, j, d = pairs_within(xyz, xyz, max_len, None)
    keep = i < j
    i, j = i[keep], j[keep]
    resi = atoms.resindices
    near = np.abs(resi[i] - resi[j]) <= 1
    h = np.array([guess_element(str(n)) == "H" for n in atoms.names])
    hh = h[i] & h[j]
    ok = near & ~hh & ((d[keep] <= 1.25) | ~(h[i] | h[j]))
    return np.stack([i[ok], j[ok]], axis=1).astype(np.int64)
