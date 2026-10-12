"""Intermolecular non-bonded interaction energy and net force between two groups.

E = sum over atom pairs (a in A, b in B, r <= cutoff) of

    332.0522 * q_a q_b / (eps r)          (or eps r^2 with distance dependence)
  + A_ab / r^12 - B_ab / r^6

Lennard-Jones coefficients come from the Amber topology when the input is a
parm7 file; otherwise approximate element-based Amber-like parameters are used
(flagged in the result metadata).  The magnitude of the net force exerted by B
on A is reported alongside the energies.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Set

import numpy as np

from ..io.loader import is_parm7
from ..io.parm7 import LJTable
from ..selections import atom_elements
from ..spatial import minimum_image_vectors, pairs_within
from .base import Analysis, register
from .common import without

COULOMB = 332.0522  # kcal mol^-1 A e^-2

# Approximate (r*, epsilon) Amber-like parameters per element: (A, kcal/mol)
GENERIC_LJ: Dict[str, tuple] = {
    "H": (1.1, 0.0157), "C": (1.908, 0.086), "N": (1.824, 0.17), "O": (1.6612, 0.21),
    "P": (2.1, 0.2), "S": (2.0, 0.25), "F": (1.75, 0.061), "MG": (0.7926, 0.8947),
    "NA": (1.369, 0.0874), "K": (1.705, 0.1937), "CL": (2.513, 0.0355), "ZN": (1.10, 0.0125),
    "CA": (1.71, 0.12), "MN": (1.2, 0.05), "BR": (2.22, 0.32),
}
_DEFAULT_LJ = (1.9, 0.1)


def _bonded_exclusions(u, a, b, depth: int) -> np.ndarray:
    """Keys (a_local * n_b + b_local) of A-B pairs separated by <= ``depth`` bonds."""
    if depth <= 0:
        return np.empty(0, dtype=np.int64)
    try:
        bonds = u.bonds.indices
    except Exception:
        return np.empty(0, dtype=np.int64)
    if len(bonds) == 0:
        return np.empty(0, dtype=np.int64)
    in_a = np.zeros(u.atoms.n_atoms, bool)
    in_b = np.zeros(u.atoms.n_atoms, bool)
    in_a[a.ix] = True
    in_b[b.ix] = True
    cross = (in_a[bonds[:, 0]] & in_b[bonds[:, 1]]) | (in_b[bonds[:, 0]] & in_a[bonds[:, 1]])
    if not cross.any():
        return np.empty(0, dtype=np.int64)
    adj: Dict[int, Set[int]] = {}
    for x, y in bonds:
        adj.setdefault(int(x), set()).add(int(y))
        adj.setdefault(int(y), set()).add(int(x))
    seeds = {int(x) for x, y in bonds[cross] for x in (x, y) if in_a[x]}
    # A atoms within depth-1 bonds of a crossing bond may reach B within depth bonds
    starts: Set[int] = set()
    for s in seeds:
        seen = {s: 0}
        q = deque([s])
        while q:
            cur = q.popleft()
            if seen[cur] >= depth - 1:
                continue
            for nb in adj.get(cur, ()):
                if nb not in seen and in_a[nb]:
                    seen[nb] = seen[cur] + 1
                    q.append(nb)
        starts.update(seen)
    a_local = {int(ix): n for n, ix in enumerate(a.ix)}
    b_local = {int(ix): n for n, ix in enumerate(b.ix)}
    keys = set()
    for s in starts:
        seen = {s: 0}
        q = deque([s])
        while q:
            cur = q.popleft()
            if seen[cur] >= depth:
                continue
            for nb in adj.get(cur, ()):
                if nb in seen:
                    continue
                seen[nb] = seen[cur] + 1
                q.append(nb)
                if in_b[nb] and nb in b_local:
                    keys.add(a_local[s] * b.n_atoms + b_local[nb])
    return np.array(sorted(keys), dtype=np.int64)


@register
class EnergyAnalysis(Analysis):
    name = "energy"
    label = "Intermolecular energy"

    @classmethod
    def enabled(cls, project):
        return project.analyses.energy.enabled

    @classmethod
    def params_for(cls, project):
        d = dict(vars(project.analyses.energy), enabled=None)
        d.pop("target_value")  # display only
        return d

    @classmethod
    def expected_outputs(cls, project):
        return {"series": ["total", "elec", "vdw", "net_force"], "matrices": [], "pairmaps": []}

    @classmethod
    def selection_texts(cls, project):
        e = project.analyses.energy
        return [e.group_a, e.group_b]

    def setup(self, frame_indices):
        super().setup(frame_indices)
        p = self.project.analyses.energy
        self.p = p
        a = self.ctx.group_atoms(p.group_a)
        b = without(self.ctx.group_atoms(p.group_b), a)
        if b.n_atoms == 0:
            raise ValueError(f"energy: group {p.group_b!r} lies entirely inside {p.group_a!r}")
        self.a, self.b = a, b
        u = self.ctx.u
        if hasattr(u.atoms, "charges"):
            self.qa = a.charges.astype(np.float64)
            self.qb = b.charges.astype(np.float64)
            self.charge_source = "topology"
        else:
            self.ctx.warn("energy: topology has no partial charges; electrostatics will be zero")
            self.qa = np.zeros(a.n_atoms)
            self.qb = np.zeros(b.n_atoms)
            self.charge_source = "none"
        self.lj = None
        if p.use_parm7 and self.ctx.topology_path and is_parm7(self.ctx.topology_path):
            try:
                self.lj = LJTable.from_parm7(self.ctx.topology_path)
                if len(self.lj.type_index) != u.atoms.n_atoms:
                    raise ValueError("atom count mismatch")
                self.lj_source = "parm7"
            except Exception as exc:
                self.ctx.warn(f"energy: could not read LJ parameters from parm7 ({exc}); "
                              "using generic element parameters")
                self.lj = None
        if self.lj is None:
            self.lj_source = "generic"
            ra, ea = self._generic(a)
            rb, eb = self._generic(b)
            self.ra, self.ea, self.rb, self.eb = ra, ea, rb, eb
        self.excl = _bonded_exclusions(u, a, b, p.exclude_bonded)
        n = self.n_frames
        self.e_el = np.zeros(n)
        self.e_vdw = np.zeros(n)
        self.force = np.zeros(n)
        self.n_pairs = np.zeros(n, dtype=np.int64)

    @staticmethod
    def _generic(atoms):
        el = atom_elements(atoms)
        rstar = np.array([GENERIC_LJ.get(e, _DEFAULT_LJ)[0] for e in el])
        eps = np.array([GENERIC_LJ.get(e, _DEFAULT_LJ)[1] for e in el])
        return rstar, eps

    def process(self, k, ts):
        p = self.p
        box = self.ctx.box(ts)
        pa = self.a.positions.astype(np.float64)
        pb = self.b.positions.astype(np.float64)
        i, j, d = pairs_within(pa, pb, p.cutoff, box)
        if len(self.excl) and len(i):
            keep = ~np.isin(i * self.b.n_atoms + j, self.excl)
            i, j, d = i[keep], j[keep], d[keep]
        self.n_pairs[k] = len(i)
        if len(i) == 0:
            return
        d = np.maximum(d, 0.5)
        qq = COULOMB * self.qa[i] * self.qb[j] / p.dielectric
        if p.distance_dependent:
            e_el = qq / (d * d)
            de_el = -2.0 * qq / d ** 3
        else:
            e_el = qq / d
            de_el = -qq / (d * d)
        if self.lj is not None:
            acoef, bcoef = self.lj.pair_coefficients(self.a.ix[i], self.b.ix[j])
        else:
            rmin = self.ra[i] + self.rb[j]
            eps = np.sqrt(self.ea[i] * self.eb[j])
            r6 = rmin ** 6
            acoef = eps * r6 * r6
            bcoef = 2.0 * eps * r6
        inv6 = 1.0 / d ** 6
        e_vdw = acoef * inv6 * inv6 - bcoef * inv6
        de_vdw = (-12.0 * acoef * inv6 * inv6 + 6.0 * bcoef * inv6) / d
        self.e_el[k] = e_el.sum()
        self.e_vdw[k] = e_vdw.sum()
        rvec = minimum_image_vectors(pa[i] - pb[j], box)
        fmag = -(de_el + de_vdw) / d
        self.force[k] = np.linalg.norm((rvec * fmag[:, None]).sum(axis=0))

    def finalize(self):
        p = self.p
        res = self.new_result(lj_source=self.lj_source, charge_source=self.charge_source,
                              group_a=p.group_a, group_b=p.group_b,
                              n_excluded_pairs=int(len(self.excl)))
        total = self.e_el + self.e_vdw
        res.arrays.update(total=total, elec=self.e_el, vdw=self.e_vdw, force=self.force,
                          n_pairs=self.n_pairs)
        ga, gb = p.group_a, p.group_b
        res.add_series("total", "total", f"{ga}–{gb} interaction energy", "kcal/mol",
                       target=p.target_value)
        res.add_series("elec", "elec", f"{ga}–{gb} electrostatic energy", "kcal/mol")
        res.add_series("vdw", "vdw", f"{ga}–{gb} van der Waals energy", "kcal/mol")
        res.add_series("net_force", "force", f"Net force of {gb} on {ga}",
                       "kcal/mol/Å")
        return res
