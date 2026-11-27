"""Intermolecular non-bonded interaction energy and net force between two groups.

E = sum over atom pairs (a in A, b in B, r <= cutoff) of

    332.0522 * q_a q_b / (eps r)          (or eps r^2 with distance dependence)
  + A_ab / r^12 - B_ab / r^6

Lennard-Jones coefficients come from the Amber topology when the input is a
parm7 file; otherwise every atom takes the parameters of one Amber atom type of
its element (:mod:`mdatk.analysis.generic_lj`, documented with its sources in
docs/GENERIC_LJ.md; flagged in the result metadata).  The magnitude of the net
force exerted by B on A is reported alongside the energies.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Set

import numpy as np

from ..io.loader import is_parm7
from ..io.parm7 import LJTable
from ..spatial import minimum_image_vectors, pairs_within
from . import generic_lj
from .base import Analysis, register
from .common import without

COULOMB = 332.0522  # kcal mol^-1 A e^-2


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
    # the A atoms within depth bonds of any B atom, along any path: also through atoms of neither group (a
    # linker residue between the groups), which a search started at the direct A-B bonds did not see
    near_b = in_b.copy()
    for _ in range(depth):
        grown = near_b.copy()
        grown[bonds[near_b[bonds[:, 1]], 0]] = True
        grown[bonds[near_b[bonds[:, 0]], 1]] = True
        near_b = grown
    starts = np.flatnonzero(near_b & in_a).tolist()
    if not starts:
        return np.empty(0, dtype=np.int64)
    adj: Dict[int, Set[int]] = {}
    for x, y in bonds:
        adj.setdefault(int(x), set()).add(int(y))
        adj.setdefault(int(y), set()).add(int(x))
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


def _has_bonds(u) -> bool:
    try:
        return len(u.bonds) > 0
    except Exception:  # noqa: BLE001 - no bonds in this topology
        return False


@register
class EnergyAnalysis(Analysis):
    name = "energy"
    label = "Intermolecular energy"
    # 2: the generic Lennard-Jones parameters (topologies that are no parm7) are those of Amber atom types, with
    # the hydrogens typed by the atom they are bonded to (mdatk.analysis.generic_lj)
    # 3: an atom's element is the topology's wherever it has one (not guessed from the name for a whole group);
    # pairs linked through atoms of neither group are excluded too; the net force of a pair closer than 0.5 Å is
    # that of 0.5 Å; virtual sites (TIP4P's extra point) have no Lennard-Jones term and are no hydrogen's parent
    version = 3
    group_slots = 2
    OUTPUT_LABELS = {
        "total": "{a}–{b} interaction energy",
        "elec": "{a}–{b} electrostatic energy",
        "vdw": "{a}–{b} van der Waals energy",
        "net_force": "Net force of {b} on {a}",
    }

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
        a = self.ctx.group_atoms(p.group_a, field=self.field("group_a"))
        b = without(self.ctx.group_atoms(p.group_b, field=self.field("group_b")), a)
        if b.n_atoms == 0:
            raise self.ctx.fail(f"energy: group {p.group_b!r} lies entirely inside {p.group_a!r}",
                                self.field("group_b"))
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
        self.lj_missing = []
        if self.lj is None:
            self.lj_source = "generic"
            self._generic_pending = True  # assigned at the first frame (a hydrogen's bonded atom may need positions)
        self.excl = _bonded_exclusions(u, a, b, p.exclude_bonded)
        if p.exclude_bonded > 0 and not _has_bonds(u):
            # (a PDB without CONECT records, a GRO file: two bonded halves of one strand give huge terms)
            self.ctx.warn("energy: the topology has no bonds; bonded pairs are not excluded "
                          f"(exclude_bonded = {p.exclude_bonded} has no effect)")
        n = self.n_frames
        self.e_el = np.zeros(n)
        self.e_vdw = np.zeros(n)
        self.force = np.zeros(n)
        self.n_pairs = np.zeros(n, dtype=np.int64)

    def _assign_generic(self, ts):
        """R* and epsilon of both groups from the generic table (docs/GENERIC_LJ.md)."""
        u = self.ctx.u
        self.ra, self.ea, miss_a = generic_lj.parameters(u, self.a, ts.dimensions)
        self.rb, self.eb, miss_b = generic_lj.parameters(u, self.b, ts.dimensions)
        self.lj_missing = sorted(set(miss_a) | set(miss_b))
        if self.lj_missing:
            self.ctx.warn("energy: no generic Lennard-Jones parameters for element(s) "
                          f"{', '.join(e or '?' for e in self.lj_missing)}: carbon's are used (R* "
                          f"{generic_lj.DEFAULT.rstar:g} Å, epsilon {generic_lj.DEFAULT.epsilon:g} kcal/mol)")
        self._generic_pending = False

    def process(self, k, ts):
        p = self.p
        box = self.ctx.box(ts)
        if self.lj is None and self._generic_pending:
            self._assign_generic(ts)
        pa = self.a.positions.astype(np.float64)
        pb = self.b.positions.astype(np.float64)
        i, j, d = pairs_within(pa, pb, p.cutoff, box)
        if len(self.excl) and len(i):
            keep = ~np.isin(i * self.b.n_atoms + j, self.excl)
            i, j, d = i[keep], j[keep], d[keep]
        self.n_pairs[k] = len(i)
        if len(i) == 0:
            return
        r = d
        d = np.maximum(r, 0.5)
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
        # the force of the (clamped) distance along the unit vector of the pair: dividing by the clamped distance
        # scaled the force of a pair closer than 0.5 Å by r / 0.5 (a pair at distance 0 has no direction)
        fmag = -(de_el + de_vdw) / np.where(r > 0.0, r, 1.0)
        self.force[k] = np.linalg.norm((rvec * fmag[:, None]).sum(axis=0))

    def finalize(self):
        p = self.p
        res = self.new_result(lj_source=self.lj_source, charge_source=self.charge_source,
                              group_a=p.group_a, group_b=p.group_b,
                              n_excluded_pairs=int(len(self.excl)))
        if self.lj_source == "generic":
            res.meta["lj_table"] = generic_lj.TABLE_NAME
            res.meta["lj_default_elements"] = list(self.lj_missing)
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
