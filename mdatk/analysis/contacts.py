"""Heavy-atom contacts between two residue groups."""

from __future__ import annotations

import numpy as np

from ..data import CSRBuilder
from ..selections import heavy
from ..spatial import pairs_within
from .base import Analysis, register
from .common import labels, per_row_counts, residue_rows, without


@register
class ContactAnalysis(Analysis):
    name = "contacts"
    label = "Pairwise contacts"

    @classmethod
    def enabled(cls, project):
        return project.analyses.contacts.enabled

    @classmethod
    def params_for(cls, project):
        return dict(vars(project.analyses.contacts), enabled=None)

    @classmethod
    def expected_outputs(cls, project):
        return {"series": ["atom_pairs", "residue_pairs", "q_native"],
                "matrices": ["per_residue_a", "per_residue_b"], "pairmaps": ["map"]}

    @classmethod
    def selection_texts(cls, project):
        c = project.analyses.contacts
        return [c.group_a, c.group_b]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        p = self.project.analyses.contacts
        a = self.ctx.group_atoms(p.group_a)
        b = self.ctx.group_atoms(p.group_b)
        if p.heavy_only:
            a, b = heavy(a), heavy(b)
        self.same = set(a.ix.tolist()) == set(b.ix.tolist())
        if not self.same:
            b = without(b, a)
            if b.n_atoms == 0:
                raise ValueError(f"contacts: group {p.group_b!r} lies entirely inside {p.group_a!r}")
        self.a, self.b = a, b
        self.res_a, self.row_a = residue_rows(a)
        self.res_b, self.row_b = residue_rows(b)
        self.n_a, self.n_b = len(self.res_a), len(self.res_b)
        self.cutoff = p.cutoff
        self.atom_pairs = np.zeros(self.n_frames, dtype=np.int64)
        self.residue_pairs = np.zeros(self.n_frames, dtype=np.int64)
        self.csr = CSRBuilder()
        self.ga, self.gb = p.group_a, p.group_b

    def process(self, k, ts):
        i, j, _ = pairs_within(self.a.positions, self.b.positions, self.cutoff, self.ctx.box(ts))
        if self.same and len(i):
            ri = self.a.resindices[i]
            rj = self.b.resindices[j]
            keep = (self.a.ix[i] < self.b.ix[j]) & (np.abs(ri - rj) > 1)
            i, j = i[keep], j[keep]
        self.atom_pairs[k] = len(i)
        if len(i) == 0:
            self.csr.add([], [], [])
            return
        keys = self.row_a[i].astype(np.int64) * self.n_b + self.row_b[j]
        uniq, cnt = np.unique(keys, return_counts=True)
        self.residue_pairs[k] = len(uniq)
        self.csr.add(uniq // self.n_b, uniq % self.n_b, cnt)

    def finalize(self):
        res = self.new_result(labels_a=labels(self.res_a), labels_b=labels(self.res_b),
                              group_a=self.ga, group_b=self.gb, cutoff=self.cutoff)
        arrays = self.csr.arrays("map")
        offsets, mi, mj, mv = (arrays["map_offsets"], arrays["map_i"], arrays["map_j"],
                               arrays["map_v"])
        # Fraction of native (reference = first analysed frame) residue contacts.
        keys = mi.astype(np.int64) * self.n_b + mj
        native = keys[offsets[0]:offsets[1]] if self.n_frames else np.empty(0, np.int64)
        q = np.zeros(self.n_frames)
        if len(native):
            hit = np.isin(keys, native).astype(np.float64)
            frame_of = np.repeat(np.arange(self.n_frames), np.diff(offsets))
            np.add.at(q, frame_of, hit)
            q /= len(native)
        res.arrays.update(arrays)
        res.arrays["atom_pairs"] = self.atom_pairs
        res.arrays["residue_pairs"] = self.residue_pairs
        res.arrays["q_native"] = q
        res.arrays["per_res_a"] = per_row_counts(offsets, mi, mv, self.n_a)
        res.arrays["per_res_b"] = per_row_counts(offsets, mj, mv, self.n_b)
        ga, gb = self.ga, self.gb
        res.add_series("atom_pairs", "atom_pairs", f"{ga}–{gb} heavy-atom contacts", "count")
        res.add_series("residue_pairs", "residue_pairs", f"{ga}–{gb} residue contacts", "count")
        res.add_series("q_native", "q_native", "Fraction of native contacts (Q)", "")
        res.add_matrix("per_residue_a", "per_res_a", "labels_a",
                       f"{ga} residue contacts with {gb}", "count")
        res.add_matrix("per_residue_b", "per_res_b", "labels_b",
                       f"{gb} residue contacts with {ga}", "count")
        res.add_pairmap("map", "map", "labels_a", "labels_b", f"{ga}–{gb} contact map")
        return res
