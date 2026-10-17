"""Magnesium coordination: O-Mg-O bridges and per-residue Mg occupancy.

* Mg-O contacts: ligand oxygens (from the selected residue ranges) within the
  inner-sphere cutoff of a Mg2+ ion; outer-sphere contacts lie between the
  inner and outer cutoffs (water mediated once solvent is stripped).
* O-Mg-O bridges: pairs of ligand oxygens coordinated by the same ion.  With
  ``geometry`` enabled a pair only counts when its O-Mg-O angle is close to
  octahedral (90 or 180 degrees).  Bridges between group A and group B are
  reported separately, together with the number of ions forming them.
* Occupancy: every residue of the occupancy selection is classified per frame
  as 0 (no Mg), 1 (outer sphere) or 2 (inner sphere) from its closest
  coordinating atom (N/O by default).
"""

from __future__ import annotations

import numpy as np

from ..selections import element_filter, split_csv
from ..spatial import minimum_image_vectors, pairs_within
from .base import Analysis, register
from .common import labels, residue_rows, without

OCCUPANCY_CLASSES = ["none", "outer sphere", "inner sphere"]


@register
class MgAnalysis(Analysis):
    name = "mg"
    label = "Mg coordination"
    group_slots = 2
    OUTPUT_LABELS = {
        "contacts": "Mg\u2013O inner-sphere contacts",
        "outer_contacts": "Mg\u2013O outer-sphere contacts",
        "bridges": "O\u2013Mg\u2013O bridges",
        "inter_bridges": "{a}\u2013Mg\u2013{b} bridges",
        "bridging_ions": "Bridging Mg ions ({a}\u2013{b})",
        "inner_residues": "Residues with inner-sphere Mg",
        "occupancy": "Mg occupancy",
    }

    @classmethod
    def enabled(cls, project):
        return project.analyses.mg.enabled

    @classmethod
    def params_for(cls, project):
        return dict(vars(project.analyses.mg), enabled=None)

    @classmethod
    def expected_outputs(cls, project):
        return {"series": ["contacts", "outer_contacts", "bridges", "inter_bridges",
                           "bridging_ions", "inner_residues"],
                "matrices": ["occupancy"], "pairmaps": []}

    @classmethod
    def selection_texts(cls, project):
        m = project.analyses.mg
        return [m.mg_selection, m.ligand_selection, m.occupancy_selection, m.group_a, m.group_b]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        p = self.project.analyses.mg
        self.p = p
        self.mg = self.ctx.select(p.mg_selection, allow_empty=True)
        if self.mg.n_atoms == 0:
            self.ctx.warn(f"Mg: selection {p.mg_selection!r} matched no ions; Mg series will be zero")
        lig_sel = without(self.ctx.select(p.ligand_selection), self.mg)
        self.lig = element_filter(lig_sel, split_csv(p.ligand_elements))
        names = set(self.project.group_names())
        in_a = self.ctx.group_atoms(p.group_a).ix if p.group_a in names else np.empty(0, int)
        in_b = self.ctx.group_atoms(p.group_b).ix if p.group_b in names else np.empty(0, int)
        self.lig_a = np.isin(self.lig.ix, in_a)
        self.lig_b = np.isin(self.lig.ix, in_b) & ~self.lig_a

        occ_sel = without(self.ctx.select(p.occupancy_selection or p.ligand_selection), self.mg)
        self.occ_res, _ = residue_rows(occ_sel)
        self.occ_atoms = element_filter(occ_sel, split_csv(p.occupancy_elements))
        uniq = np.unique(occ_sel.resindices)
        self.occ_row = np.searchsorted(uniq, self.occ_atoms.resindices)
        self.n_occ = len(uniq)

        n = self.n_frames
        self.contacts = np.zeros(n, dtype=np.int64)
        self.outer = np.zeros(n, dtype=np.int64)
        self.bridges = np.zeros(n, dtype=np.int64)
        self.inter = np.zeros(n, dtype=np.int64)
        self.bridging_ions = np.zeros(n, dtype=np.int64)
        self.occupancy = np.zeros((self.n_occ, n), dtype=np.int8)
        self.cos90 = np.cos(np.radians(90.0 - p.angle_tolerance))  # |cos| <= this ~ 90 deg
        self.cos180 = np.cos(np.radians(180.0 - p.angle_tolerance))  # cos <= this ~ 180 deg

    def process(self, k, ts):
        if self.mg.n_atoms == 0:
            return
        p = self.p
        box = self.ctx.box(ts)
        mpos = self.mg.positions.astype(np.float64)
        if self.lig.n_atoms:
            lpos = self.lig.positions.astype(np.float64)
            i, j, d = pairs_within(mpos, lpos, p.outer_cutoff, box)
            inner = d <= p.inner_cutoff
            self.contacts[k] = int(inner.sum())
            self.outer[k] = int((~inner).sum())
            shell = inner if p.bridge_shell == "inner" else np.ones_like(inner)
            bi, bj = i[shell], j[shell]
            if len(bi) > 1:
                order = np.argsort(bi, kind="stable")
                bi, bj = bi[order], bj[order]
                bounds = np.flatnonzero(np.diff(bi)) + 1
                n_br = n_inter = n_ions = 0
                for ion_slice in np.split(np.arange(len(bi)), bounds):
                    if len(ion_slice) < 2:
                        continue
                    ion = bi[ion_slice[0]]
                    ligs = bj[ion_slice]
                    p1, p2 = np.triu_indices(len(ligs), 1)
                    l1, l2 = ligs[p1], ligs[p2]
                    if p.geometry:
                        v1 = minimum_image_vectors(lpos[l1] - mpos[ion], box)
                        v2 = minimum_image_vectors(lpos[l2] - mpos[ion], box)
                        c = (v1 * v2).sum(1) / np.maximum(
                            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
                        ok = (np.abs(c) <= self.cos90) | (c <= self.cos180)
                        l1, l2 = l1[ok], l2[ok]
                    n_br += len(l1)
                    cross = (self.lig_a[l1] & self.lig_b[l2]) | (self.lig_b[l1] & self.lig_a[l2])
                    n_cross = int(cross.sum())
                    n_inter += n_cross
                    n_ions += int(n_cross > 0)
                self.bridges[k] = n_br
                self.inter[k] = n_inter
                self.bridging_ions[k] = n_ions
        if self.occ_atoms.n_atoms:
            opos = self.occ_atoms.positions.astype(np.float64)
            _, j, d = pairs_within(mpos, opos, p.outer_cutoff, box)
            if len(j):
                best = np.full(self.n_occ, np.inf)
                np.minimum.at(best, self.occ_row[j], d)
                cls = np.zeros(self.n_occ, dtype=np.int8)
                cls[best <= p.outer_cutoff] = 1
                cls[best <= p.inner_cutoff] = 2
                self.occupancy[:, k] = cls

    def finalize(self):
        p = self.p
        res = self.new_result(occ_labels=labels(self.occ_res), n_ions=int(self.mg.n_atoms),
                              group_a=p.group_a, group_b=p.group_b)
        res.arrays.update(contacts=self.contacts, outer=self.outer, bridges=self.bridges,
                          inter=self.inter, bridging_ions=self.bridging_ions,
                          occupancy=self.occupancy,
                          inner_residues=(self.occupancy == 2).sum(axis=0).astype(np.int64))
        res.add_series("contacts", "contacts", "Mg–O inner-sphere contacts", "count")
        res.add_series("outer_contacts", "outer", "Mg–O outer-sphere contacts", "count")
        res.add_series("bridges", "bridges", "O–Mg–O bridges", "count")
        res.add_series("inter_bridges", "inter", f"{p.group_a}–Mg–{p.group_b} bridges",
                       "count")
        res.add_series("bridging_ions", "bridging_ions", "Bridging Mg ions", "count")
        res.add_series("inner_residues", "inner_residues", "Residues with inner-sphere Mg",
                       "count")
        res.add_matrix("occupancy", "occupancy", "occ_labels", "Mg occupancy", "class",
                       classes=OCCUPANCY_CLASSES)
        return res
