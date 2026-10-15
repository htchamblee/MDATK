"""Inter-group hydrogen bonds from basic geometric criteria.

A hydrogen bond is counted for every (H, acceptor) pair where the donor heavy
atom D and acceptor A are within ``distance_cutoff`` and, when angles are
enabled, the D-H...A angle is at least ``angle_cutoff``.  Donor hydrogens are
taken from the topology's bonds; when the topology has no bonds they are
assigned from distances on the first analysed frame.  Topologies without
hydrogens fall back to donor-acceptor distances only.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data import CSRBuilder
from ..selections import atom_elements, split_csv
from ..spatial import minimum_image_vectors, pairs_within
from .base import Analysis, register
from .common import labels, per_row_counts, residue_rows, without


@register
class HBondAnalysis(Analysis):
    name = "hbonds"
    label = "Hydrogen bonds"
    group_slots = 2
    OUTPUT_LABELS = {
        "count": "{a}\u2013{b} H-bonds",
        "per_residue_a": "{a} residue H-bonds",
        "per_residue_b": "{b} residue H-bonds",
        "map": "{a}\u2013{b} H-bond map",
    }

    @classmethod
    def enabled(cls, project):
        return project.analyses.hbonds.enabled

    @classmethod
    def params_for(cls, project):
        return dict(vars(project.analyses.hbonds), enabled=None)

    @classmethod
    def expected_outputs(cls, project):
        return {"series": ["count"], "matrices": ["per_residue_a", "per_residue_b"],
                "pairmaps": ["map"]}

    @classmethod
    def selection_texts(cls, project):
        h = project.analyses.hbonds
        return [h.group_a, h.group_b]

    # ------------------------------------------------------------------ setup
    def setup(self, frame_indices):
        super().setup(frame_indices)
        p = self.project.analyses.hbonds
        a = self.ctx.group_atoms(p.group_a)
        b = without(self.ctx.group_atoms(p.group_b), a)
        if b.n_atoms == 0:
            raise ValueError(f"H-bonds: group {p.group_b!r} lies entirely inside {p.group_a!r}")
        self.ga, self.gb = p.group_a, p.group_b
        self.ab = a | b  # sorted union; positions fetched once per frame
        self.side = np.where(np.isin(self.ab.ix, a.ix), 0, 1)
        el = atom_elements(self.ab)
        self.is_h = el == "H"
        self.has_h = bool(self.is_h.any())
        donor_el = {e.upper() for e in split_csv(p.donor_elements)}
        acc_el = {e.upper() for e in split_csv(p.acceptor_elements)}
        self.res_a, _ = residue_rows(a)
        self.res_b, _ = residue_rows(b)
        # residue row of each atom within its own side
        uniq_a = np.unique(a.resindices)
        uniq_b = np.unique(b.resindices)
        self.row = np.where(self.side == 0, np.searchsorted(uniq_a, self.ab.resindices),
                            np.searchsorted(uniq_b, self.ab.resindices))
        self.n_a, self.n_b = len(uniq_a), len(uniq_b)

        cand_donor = np.isin(el, list(donor_el))
        cand_acc = np.isin(el, list(acc_el))
        self.h_of: Dict[int, List[int]] = {}
        self.bonds_known = False
        try:
            bonds = self.ab.bonds
            self.bonds_known = len(bonds) > 0
        except Exception:
            bonds = None
        heavy_neighbors = np.zeros(self.ab.n_atoms, dtype=np.int64)
        if self.bonds_known:
            local = {int(ix): n for n, ix in enumerate(self.ab.ix)}
            for b1, b2 in bonds.indices:
                l1, l2 = local.get(int(b1)), local.get(int(b2))
                for x, y in ((l1, l2), (l2, l1)):
                    if x is None or y is None:
                        continue
                    if self.is_h[y]:
                        self.h_of.setdefault(x, []).append(y)
                    else:
                        heavy_neighbors[x] += 1
            # nitrogens carrying H or with three heavy neighbours are not acceptors
            for x in self.h_of:
                if el[x] == "N":
                    cand_acc[x] = False
            cand_acc &= ~((el == "N") & (heavy_neighbors >= 3))
        self.cand_donor = cand_donor
        self.cand_acc = cand_acc
        self.use_angle = p.use_angle and self.has_h
        if p.use_angle and not self.has_h:
            self.ctx.warn("H-bonds: topology has no hydrogens; using donor-acceptor distance only")
        self._finalize_donors_pending = self.has_h and not self.bonds_known
        if not self._finalize_donors_pending:
            self._build_donor_lists()
        self.dcut = p.distance_cutoff
        self.cos_cut = np.cos(np.radians(p.angle_cutoff))
        self.count = np.zeros(self.n_frames, dtype=np.int64)
        self.csr = CSRBuilder()

    def _build_donor_lists(self):
        donors = self.cand_donor.copy()
        if self.has_h:
            has = np.zeros_like(donors)
            for x in self.h_of:
                has[x] = True
            donors &= has
        self.donor_idx = [np.flatnonzero(donors & (self.side == s)) for s in (0, 1)]
        self.acc_idx = [np.flatnonzero(self.cand_acc & (self.side == s)) for s in (0, 1)]
        # CSR of hydrogens per donor (local indices)
        self.h_start = np.zeros(self.ab.n_atoms + 1, dtype=np.int64)
        counts = np.zeros(self.ab.n_atoms, dtype=np.int64)
        for x, hs in self.h_of.items():
            counts[x] = len(hs)
        self.h_start[1:] = np.cumsum(counts)
        self.h_flat = np.zeros(self.h_start[-1], dtype=np.int64)
        for x, hs in self.h_of.items():
            self.h_flat[self.h_start[x]:self.h_start[x] + len(hs)] = hs

    def _guess_hydrogens(self, pos, box):
        heavy_idx = np.flatnonzero(self.cand_donor)
        h_idx = np.flatnonzero(self.is_h)
        i, j, _ = pairs_within(pos[h_idx], pos[heavy_idx], 1.25, box)
        best: Dict[int, tuple] = {}
        d = np.linalg.norm(pos[h_idx[i]] - pos[heavy_idx[j]], axis=1) if len(i) else []
        for hi, hj, dist in zip(i, j, d):
            if hi not in best or dist < best[hi][1]:
                best[hi] = (hj, dist)
        for hi, (hj, _) in best.items():
            self.h_of.setdefault(int(heavy_idx[hj]), []).append(int(h_idx[hi]))
        self._build_donor_lists()

    # ---------------------------------------------------------------- frames
    def process(self, k, ts):
        pos = self.ab.positions.astype(np.float64)
        box = self.ctx.box(ts)
        if self._finalize_donors_pending:
            self._guess_hydrogens(pos, box)
            self._finalize_donors_pending = False
        rows_a: List[np.ndarray] = []
        rows_b: List[np.ndarray] = []
        total = 0
        for s in (0, 1):
            don = self.donor_idx[s]
            acc = self.acc_idx[1 - s]
            i, j, _ = pairs_within(pos[don], pos[acc], self.dcut, box)
            if len(i) == 0:
                continue
            d_atoms = don[i]
            a_atoms = acc[j]
            if self.use_angle:
                n_h = self.h_start[d_atoms + 1] - self.h_start[d_atoms]
                rep_d = np.repeat(d_atoms, n_h)
                rep_a = np.repeat(a_atoms, n_h)
                offs = np.repeat(self.h_start[d_atoms], n_h)
                within = np.arange(n_h.sum()) - np.repeat(np.cumsum(n_h) - n_h, n_h)
                hyd = self.h_flat[offs + within]
                v1 = minimum_image_vectors(pos[rep_d] - pos[hyd], box)
                v2 = minimum_image_vectors(pos[rep_a] - pos[hyd], box)
                cosang = (v1 * v2).sum(1) / np.maximum(
                    np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
                ok = cosang <= self.cos_cut  # angle >= cutoff
                d_atoms, a_atoms = rep_d[ok], rep_a[ok]
            total += len(d_atoms)
            if s == 0:
                rows_a.append(self.row[d_atoms])
                rows_b.append(self.row[a_atoms])
            else:
                rows_a.append(self.row[a_atoms])
                rows_b.append(self.row[d_atoms])
        self.count[k] = total
        if total:
            ra = np.concatenate(rows_a).astype(np.int64)
            rb = np.concatenate(rows_b).astype(np.int64)
            uniq, cnt = np.unique(ra * self.n_b + rb, return_counts=True)
            self.csr.add(uniq // self.n_b, uniq % self.n_b, cnt)
        else:
            self.csr.add([], [], [])

    def finalize(self):
        p = self.project.analyses.hbonds
        res = self.new_result(labels_a=labels(self.res_a), labels_b=labels(self.res_b),
                              group_a=self.ga, group_b=self.gb, used_angle=bool(self.use_angle),
                              distance_cutoff=p.distance_cutoff, angle_cutoff=p.angle_cutoff)
        arrays = self.csr.arrays("map")
        res.arrays.update(arrays)
        res.arrays["count"] = self.count
        res.arrays["per_res_a"] = per_row_counts(arrays["map_offsets"], arrays["map_i"],
                                                 arrays["map_v"], self.n_a)
        res.arrays["per_res_b"] = per_row_counts(arrays["map_offsets"], arrays["map_j"],
                                                 arrays["map_v"], self.n_b)
        ga, gb = self.ga, self.gb
        res.add_series("count", "count", f"{ga}–{gb} H-bonds", "count")
        res.add_matrix("per_residue_a", "per_res_a", "labels_a", f"{ga} residue H-bonds", "count")
        res.add_matrix("per_residue_b", "per_res_b", "labels_b", f"{gb} residue H-bonds", "count")
        res.add_pairmap("map", "map", "labels_a", "labels_b", f"{ga}–{gb} H-bond map")
        return res
