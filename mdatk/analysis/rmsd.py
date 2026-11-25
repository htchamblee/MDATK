"""Root mean square deviation against a reference frame."""

from __future__ import annotations

import numpy as np

from ..config import rmsd_series_key
from ..spatial import collinear, kabsch
from .base import Analysis, register

_out_key = rmsd_series_key  # one definition: Project.problems() refuses two targets with the same key


def superposed_rmsd(x: np.ndarray, ref: np.ndarray, weights=None) -> float:
    """(Weighted) RMSD of ``x`` from ``ref`` after the optimal superposition, exact for any number of atoms.

    The rotation comes from a singular value decomposition and the deviation is measured after rotating.  One
    atom gives 0; for two atoms or atoms on one line the rotation about their line is not defined but the RMSD
    is, and this finds it.  (MDAnalysis' QCP iteration returned NaN or values of several Å for such sets, and
    up to 1e-5 Å instead of 0 for identical sets of any size.)  NaN for non-finite coordinates, as QCP gave."""
    if not (np.isfinite(x).all() and np.isfinite(ref).all()):
        return float("nan")
    if weights is None:
        w = np.full(len(x), 1.0 / max(1, len(x)))
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    a = x - w @ x
    b = ref - w @ ref
    u, _, vt = np.linalg.svd((a * w[:, None]).T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    diff = a @ (vt.T @ np.diag([1.0, 1.0, d]) @ u.T).T - b
    return float(np.sqrt(w @ (diff * diff).sum(axis=1)))


@register
class RMSDAnalysis(Analysis):
    name = "rmsd"
    label = "RMSD"
    # 2: the superposed RMSD is exact for one or two atoms and atoms on one line (it was NaN or off by several Å),
    # and a fit selection that defines no rotation (fewer than three atoms, or all on one line) only translates
    version = 2
    group_slots = 1
    instance_settings = ("targets",)  # with_groups makes targets of the group

    @classmethod
    def with_groups(cls, project, groups):
        from ..config import RMSDTarget

        g = groups[0]
        project.analyses.rmsd.targets = [
            RMSDTarget("Heavy", f"group:{g} and not name H*"),
            RMSDTarget("Backbone", f"group:{g} and name C4'"),
        ]
        return project

    @classmethod
    def output_label(cls, project, out):
        for k, t in enumerate(project.analyses.rmsd.targets):
            if _out_key(t.label, k) == out:
                return f"RMSD {t.label}"
        return f"RMSD {out}"

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
            raise self.ctx.fail("RMSD: no targets defined", self.field("targets"))
        self.ref_pos = min(max(0, p.reference_frame), self.n_frames - 1)
        if p.reference_frame < 0:
            self.ctx.warn(f"RMSD: reference frame {p.reference_frame} is negative; using the first")
        elif p.reference_frame != self.ref_pos:
            self.ctx.warn(f"RMSD: reference frame {p.reference_frame} is beyond the {self.n_frames} analysed "
                          "frame(s); using the last")
        self.superposition = p.superposition
        self.targets = []
        for k, t in enumerate(p.targets):
            mob = self.ctx.select(t.selection, allow_empty=True, field=self.field(f"targets.{k}.selection"))
            if mob.n_atoms == 0:
                self.ctx.warn(f"RMSD target {t.label!r}: {t.selection!r} matched no atoms; "
                              "its series will be empty (NaN)")
            fit = (self.ctx.select(t.fit_selection, field=self.field(f"targets.{k}.fit_selection"))
                   if t.fit_selection.strip() else None)
            w = mob.masses.astype(np.float64) if p.mass_weighted else None
            wf = (fit.masses.astype(np.float64) if fit is not None else w) if p.mass_weighted else None
            self.targets.append({"label": t.label or f"target{k}", "out": _out_key(t.label, k),
                                 "mob": mob, "fit": fit, "w": w, "wf": wf})
        self.ctx.u.trajectory[int(frame_indices[self.ref_pos])]
        for t in self.targets:
            t["ref"] = t["mob"].positions.astype(np.float64).copy()
            t["ref_fit"] = t["fit"].positions.astype(np.float64).copy() if t["fit"] is not None else None
            # fewer than three fit atoms, or all on one line, define no rotation (about the line it would be
            # whatever the linear algebra library returns): such a fit only translates, and says so
            t["shift_only"] = t["fit"] is not None and collinear(t["ref_fit"], t["wf"])
            if t["shift_only"] and self.superposition:
                n_fit = t["fit"].n_atoms
                self.ctx.warn(f"RMSD target {t['label']!r}: fit selection has {n_fit} atom{'s' * (n_fit != 1)}"
                              f"{' on one line' if n_fit >= 3 else ''}; superposition is translation only")
        self.values = np.full((len(self.targets), self.n_frames), np.nan, dtype=np.float64)

    def process(self, k, ts):
        for n, t in enumerate(self.targets):
            if t["mob"].n_atoms == 0:
                continue
            x = t["mob"].positions.astype(np.float64)
            ref = t["ref"]
            if not self.superposition:
                diff = x - ref
            elif t["fit"] is None:
                self.values[n, k] = superposed_rmsd(x, ref, t["w"])
                continue
            else:
                xf = t["fit"].positions.astype(np.float64)
                if not (np.isfinite(xf).all() and np.isfinite(t["ref_fit"]).all()):
                    continue  # a blown-up frame cannot be superposed: no value (NaN)
                r, mc, rc = kabsch(xf, t["ref_fit"], t["wf"])
                if t["shift_only"]:
                    r = np.eye(3)
                diff = (x - mc) @ r.T + rc - ref
            sq = (diff * diff).sum(axis=1)
            w = t["w"]
            self.values[n, k] = np.sqrt(np.average(sq, weights=w)) if w is not None else np.sqrt(sq.mean())

    def finalize(self):
        from .base import parse_instance

        res = self.new_result(reference_frame_index=self.ref_pos)
        _, groups = parse_instance(self.instance)
        suffix = f" ({groups[0]})" if groups else ""
        for n, t in enumerate(self.targets):
            arr = f"rmsd_{n}"
            res.arrays[arr] = self.values[n]
            res.add_series(t["out"], arr, f"RMSD {t['label']}{suffix}", "Å")
        return res
