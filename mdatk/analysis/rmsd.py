"""Root mean square deviation against a reference frame."""

from __future__ import annotations

import re

import numpy as np
from MDAnalysis.analysis import rms

from ..spatial import kabsch
from .base import Analysis, register


def _out_key(label: str, k: int) -> str:
    key = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", label).strip("_")
    return key or f"target{k}"


@register
class RMSDAnalysis(Analysis):
    name = "rmsd"
    label = "RMSD"

    @classmethod
    def enabled(cls, project):
        return project.analyses.rmsd.enabled and bool(project.analyses.rmsd.targets)

    @classmethod
    def params_for(cls, project):
        r = project.analyses.rmsd
        return {"targets": [vars(t) for t in r.targets], "reference_frame": r.reference_frame,
                "superposition": r.superposition, "mass_weighted": r.mass_weighted}

    @classmethod
    def expected_outputs(cls, project):
        return {"series": [_out_key(t.label, k) for k, t in enumerate(project.analyses.rmsd.targets)],
                "matrices": [], "pairmaps": []}

    @classmethod
    def selection_texts(cls, project):
        out = []
        for t in project.analyses.rmsd.targets:
            out += [t.selection, t.fit_selection]
        return out

    def setup(self, frame_indices):
        super().setup(frame_indices)
        p = self.project.analyses.rmsd
        if not p.targets:
            raise ValueError("RMSD: no targets defined")
        self.ref_pos = min(max(0, p.reference_frame), self.n_frames - 1)
        self.superposition = p.superposition
        self.targets = []
        for k, t in enumerate(p.targets):
            mob = self.ctx.select(t.selection)
            fit = self.ctx.select(t.fit_selection) if t.fit_selection.strip() else None
            w = mob.masses.astype(np.float64) if p.mass_weighted else None
            wf = (fit.masses.astype(np.float64) if fit is not None else w) if p.mass_weighted else None
            self.targets.append({"label": t.label or f"target{k}", "out": _out_key(t.label, k),
                                 "mob": mob, "fit": fit, "w": w, "wf": wf})
        self.ctx.u.trajectory[int(frame_indices[self.ref_pos])]
        for t in self.targets:
            t["ref"] = t["mob"].positions.astype(np.float64).copy()
            t["ref_fit"] = t["fit"].positions.astype(np.float64).copy() if t["fit"] is not None else None
        self.values = np.zeros((len(self.targets), self.n_frames), dtype=np.float64)

    def process(self, k, ts):
        for n, t in enumerate(self.targets):
            x = t["mob"].positions.astype(np.float64)
            ref = t["ref"]
            if not self.superposition:
                diff = x - ref
            elif t["fit"] is None:
                self.values[n, k] = rms.rmsd(x, ref, weights=t["w"], center=True, superposition=True)
                continue
            else:
                xf = t["fit"].positions.astype(np.float64)
                r, mc, rc = kabsch(xf, t["ref_fit"], t["wf"])
                diff = (x - mc) @ r.T + rc - ref
            sq = (diff * diff).sum(axis=1)
            w = t["w"]
            self.values[n, k] = np.sqrt(np.average(sq, weights=w)) if w is not None else np.sqrt(sq.mean())

    def finalize(self):
        res = self.new_result(reference_frame_index=self.ref_pos)
        for n, t in enumerate(self.targets):
            arr = f"rmsd_{n}"
            res.arrays[arr] = self.values[n]
            res.add_series(t["out"], arr, f"RMSD {t['label']}", "Å")
        return res
