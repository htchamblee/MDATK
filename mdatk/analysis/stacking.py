"""pi-pi stacking between nucleobases (and aromatic side chains).

For every aromatic ring the centroid, the plane normal (least-squares fit) and
the planarity (RMS deviation of the ring atoms from that plane) are computed
per frame.  Two rings are stacked when

* both rings are planar (RMS deviation <= ``planarity_cutoff``),
* their centroids are within ``distance_cutoff``,
* the angle between their normals is <= ``angle_cutoff``, and
* the lateral offset of one centroid from the other ring's normal axis is
  <= ``offset_cutoff`` (the smaller of the two offsets is used).

Purines use all nine ring atoms, so a stacked partner over either ring counts.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data import CSRBuilder
from ..spatial import minimum_image_vectors, pairs_within
from .base import Analysis, register
from .common import labels

PURINE = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
PYRIMIDINE = ("N1", "C2", "N3", "C4", "C5", "C6")
PHENYL = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
AROMATIC_RESIDUES: Dict[str, tuple] = {
    "PHE": PHENYL, "TYR": PHENYL,
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"), "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"), "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HSD": ("CG", "ND1", "CD2", "CE1", "NE2"), "HSE": ("CG", "ND1", "CD2", "CE1", "NE2"),
}


def ring_atoms(residue):
    """Atom indices of the aromatic ring of a residue, or None."""
    names = {a.name: a.ix for a in residue.atoms}
    for ring in (PURINE, PYRIMIDINE):
        if all(n in names for n in ring):
            return [names[n] for n in ring]
    ring = AROMATIC_RESIDUES.get(residue.resname.upper())
    if ring and all(n in names for n in ring):
        return [names[n] for n in ring]
    return None


def ring_geometry(positions: np.ndarray):
    """Centroids, unit normals and planarity RMS for rings of equal size.

    ``positions`` has shape (n_rings, ring_size, 3).
    """
    c = positions.mean(axis=1)
    y = positions - c[:, None, :]
    _, s, vt = np.linalg.svd(y)
    normals = vt[:, -1, :]
    planarity = s[:, -1] / np.sqrt(positions.shape[1])
    return c, normals, planarity


@register
class StackingAnalysis(Analysis):
    name = "stacking"
    label = "π–π stacking"

    @classmethod
    def enabled(cls, project):
        return project.analyses.stacking.enabled

    @classmethod
    def params_for(cls, project):
        return dict(vars(project.analyses.stacking), enabled=None)

    @classmethod
    def expected_outputs(cls, project):
        series = ["count", "inter"] + (["intra"] if project.analyses.stacking.scope != "inter" else [])
        return {"series": series, "matrices": [], "pairmaps": ["map"]}

    @classmethod
    def selection_texts(cls, project):
        s = project.analyses.stacking
        return [s.selection, s.group_a, s.group_b]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        p = self.project.analyses.stacking
        self.p = p
        sel = self.ctx.select(p.selection)
        names = set(self.project.group_names())
        in_a = self.ctx.group_atoms(p.group_a).residues.ix if p.group_a in names else []
        in_b = self.ctx.group_atoms(p.group_b).residues.ix if p.group_b in names else []
        rings: List[List[int]] = []
        residues = []
        for res in sel.residues:
            r = ring_atoms(res)
            if r is not None:
                rings.append(r)
                residues.append(res)
        if not rings:
            self.ctx.warn("stacking: no aromatic rings found in the selection")
        self.ring_res = self.ctx.u.residues[[r.ix for r in residues]] if residues else None
        self.res_ix = np.array([r.ix for r in residues], dtype=np.int64)
        self.side = np.full(len(rings), -1)
        self.side[np.isin(self.res_ix, in_a)] = 0
        self.side[np.isin(self.res_ix, in_b) & (self.side < 0)] = 1
        all_idx = np.unique(np.concatenate([np.asarray(r) for r in rings])) if rings else np.empty(0, int)
        self.atoms = self.ctx.u.atoms[all_idx]
        local = {int(ix): n for n, ix in enumerate(all_idx)}
        self.by_size: Dict[int, tuple] = {}
        for n_ring, r in enumerate(rings):
            self.by_size.setdefault(len(r), ([], []))
            self.by_size[len(r)][0].append(n_ring)
            self.by_size[len(r)][1].append([local[int(x)] for x in r])
        self.by_size = {k: (np.array(v[0]), np.array(v[1])) for k, v in self.by_size.items()}
        self.n_rings = len(rings)
        if p.scope == "inter":
            self.rows = np.flatnonzero(self.side == 0)
            self.cols = np.flatnonzero(self.side == 1)
        else:
            self.rows = self.cols = np.arange(self.n_rings)
        self.row_of = np.full(self.n_rings, -1)
        self.row_of[self.rows] = np.arange(len(self.rows))
        self.col_of = np.full(self.n_rings, -1)
        self.col_of[self.cols] = np.arange(len(self.cols))
        n = self.n_frames
        self.count = np.zeros(n, dtype=np.int64)
        self.inter = np.zeros(n, dtype=np.int64)
        self.intra = np.zeros(n, dtype=np.int64)
        self.csr = CSRBuilder()
        self.cos_cut = np.cos(np.radians(p.angle_cutoff))

    def process(self, k, ts):
        if self.n_rings < 2:
            self.csr.add([], [], [])
            return
        p = self.p
        box = self.ctx.box(ts)
        pos = self.atoms.positions.astype(np.float64)
        cent = np.zeros((self.n_rings, 3))
        norm = np.zeros((self.n_rings, 3))
        planar = np.zeros(self.n_rings)
        for ring_ids, idx in self.by_size.values():
            c, nvec, pl = ring_geometry(pos[idx])
            cent[ring_ids], norm[ring_ids], planar[ring_ids] = c, nvec, pl
        if p.scope == "inter":
            i, j, _ = pairs_within(cent[self.rows], cent[self.cols], p.distance_cutoff, box)
            i, j = self.rows[i], self.cols[j]
        else:
            i, j, _ = pairs_within(cent, cent, p.distance_cutoff, box)
            keep = (i < j) & (self.res_ix[i] != self.res_ix[j])
            i, j = i[keep], j[keep]
        if len(i):
            v = minimum_image_vectors(cent[j] - cent[i], box)
            ni, nj = norm[i], norm[j]
            cosang = np.abs((ni * nj).sum(1))
            proj_i = (v * ni).sum(1)
            proj_j = (v * nj).sum(1)
            off_i = np.sqrt(np.maximum((v * v).sum(1) - proj_i ** 2, 0))
            off_j = np.sqrt(np.maximum((v * v).sum(1) - proj_j ** 2, 0))
            ok = ((cosang >= self.cos_cut) & (np.minimum(off_i, off_j) <= p.offset_cutoff)
                  & (planar[i] <= p.planarity_cutoff) & (planar[j] <= p.planarity_cutoff))
            i, j = i[ok], j[ok]
        self.count[k] = len(i)
        if len(i):
            cross = ((self.side[i] == 0) & (self.side[j] == 1)) | ((self.side[i] == 1) & (self.side[j] == 0))
            self.inter[k] = int(cross.sum())
            self.intra[k] = len(i) - self.inter[k]
            self.csr.add(self.row_of[i], self.col_of[j], np.ones(len(i)))
        else:
            self.csr.add([], [], [])

    def finalize(self):
        p = self.p
        ring_labels = labels(self.ring_res) if self.ring_res is not None else []
        res = self.new_result(row_labels=[ring_labels[r] for r in self.rows],
                              col_labels=[ring_labels[c] for c in self.cols],
                              n_rings=self.n_rings, scope=p.scope)
        res.arrays.update(self.csr.arrays("map"))
        res.arrays.update(count=self.count, inter=self.inter, intra=self.intra)
        res.add_series("count", "count", "π–π stacked pairs", "count")
        res.add_series("inter", "inter", f"{p.group_a}–{p.group_b} stacked pairs", "count")
        if p.scope != "inter":
            res.add_series("intra", "intra", "Intra-group stacked pairs", "count")
        res.add_pairmap("map", "map", "row_labels", "col_labels", "Stacking map")
        return res
