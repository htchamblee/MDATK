"""Generic Lennard-Jones parameters of the energy analysis, for topologies without atom types (PDB, GRO, PSF ...).

A parm7 topology brings its own Lennard-Jones table and is used as it is (:mod:`mdatk.io.parm7`).  Any other
topology only tells an atom's element, so the energy analysis gives every atom the parameters of one Amber atom
type of its element.  The whole table, the rules and where every number was read are documented in
``docs/GENERIC_LJ.md``, which a test keeps identical to the tables below.  In short:

* **Atoms of molecules** take one atom type of ``parm10.dat``, the parameter file of Amber's ff10 / ff14SB / OL3 /
  OL15 force fields (Cornell et al. 1995 [parm94] and Wang, Cieplak & Kollman 2000 [parm99] for these types, which
  parm10 keeps unchanged).
* **Hydrogens** take the type of the atom they are bonded to (on N: ``H``, on O: ``HO``, on S: ``HS``, on anything
  else: ``H1``), because Amber's polar hydrogens are small or have no Lennard-Jones term at all: one value for
  every hydrogen made hydrogen-bonded contacts strongly repulsive (MDATK up to 0.1.0).
* **Monatomic ions** (an atom that is alone in its residue) take what Amber's ``leaprc.water.tip3p`` loads:
  Joung & Cheatham 2008 (monovalent, TIP3P set) and the 12-6 "CM" set of Li, Roberts, Chakravorty & Merz 2013
  (divalent, TIP3P).
* An element in neither table takes carbon's parameters, and the analysis says so.

Every value is ``(R*, epsilon)``: R* = Rmin/2 in Å and the well depth in kcal/mol, as Amber writes them.  A pair
has Rmin = R*_i + R*_j and epsilon = sqrt(eps_i eps_j) (Lorentz-Berthelot, as in Amber).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from ..selections import atom_elements
from ..spatial import pairs_within

TABLE_NAME = "Amber parm10 atom types, Joung-Cheatham and Li-Merz (CM) ions for TIP3P (docs/GENERIC_LJ.md)"
CITATION_KEYS = ("cornell1995", "wang2000", "joung2008", "li2013")

PARM10 = "parm10.dat"
JC_TIP3P = "frcmod.ionsjc_tip3p"
LM_TIP3P = "frcmod.ions234lm_126_tip3p"


class LJ(NamedTuple):
    rstar: float  # R* = Rmin/2 (Å)
    epsilon: float  # well depth (kcal/mol)
    amber_type: str  # the Amber atom type (or ion) the values are those of
    source: str  # the AmberTools parameter file they were read from


# atoms of molecules, by element: one Amber atom type each (see docs/GENERIC_LJ.md for what the choice leaves out)
COVALENT: Dict[str, LJ] = {
    "C": LJ(1.9080, 0.0860, "C", PARM10),  # every sp2 carbon type; sp3 CT has the same R* and epsilon 0.1094
    "N": LJ(1.8240, 0.1700, "N", PARM10),  # every nitrogen type of parm10
    "O": LJ(1.6612, 0.2100, "O", PARM10),  # O and O2 (carbonyl, phosphate); OH, OS and OW differ a little
    "P": LJ(2.1000, 0.2000, "P", PARM10),
    "S": LJ(2.0000, 0.2500, "S", PARM10),  # S and SH
    "F": LJ(1.75, 0.061, "F", PARM10),
    "CL": LJ(1.948, 0.265, "Cl", PARM10),
    "BR": LJ(2.22, 0.320, "Br", PARM10),
    "I": LJ(2.35, 0.40, "I", PARM10),
}

# hydrogens, by the element of the atom they are bonded to
HYDROGEN: Dict[str, LJ] = {
    "N": LJ(0.6000, 0.0157, "H", PARM10),
    "O": LJ(0.0000, 0.0000, "HO", PARM10),  # no Lennard-Jones term, as HW of the water models
    "S": LJ(0.6000, 0.0157, "HS", PARM10),
}
HYDROGEN_OTHER = LJ(1.3870, 0.0157, "H1", PARM10)  # on carbon (and any other element)
HYDROGEN_ALONE = HYDROGEN["N"]  # no bonded atom found: the small radius, which cannot make a contact repulsive

# monatomic ions (an atom alone in its residue), by element: what leaprc.water.tip3p loads
IONS: Dict[str, LJ] = {
    "LI": LJ(1.025, 0.0279896, "Li+", JC_TIP3P),
    "NA": LJ(1.369, 0.0874393, "Na+", JC_TIP3P),
    "K": LJ(1.705, 0.1936829, "K+", JC_TIP3P),
    "RB": LJ(1.813, 0.3278219, "Rb+", JC_TIP3P),
    "CS": LJ(1.976, 0.4065394, "Cs+", JC_TIP3P),
    "F": LJ(2.303, 0.0033640, "F-", JC_TIP3P),
    "CL": LJ(2.513, 0.0355910, "Cl-", JC_TIP3P),
    "BR": LJ(2.608, 0.0586554, "Br-", JC_TIP3P),
    "I": LJ(2.860, 0.0536816, "I-", JC_TIP3P),
    "MG": LJ(1.360, 0.01020237, "Mg2+", LM_TIP3P),
    "CA": LJ(1.649, 0.10592870, "Ca2+", LM_TIP3P),
    "SR": LJ(1.810, 0.22132374, "Sr2+", LM_TIP3P),
    "BA": LJ(2.019, 0.40664608, "Ba2+", LM_TIP3P),
    "MN": LJ(1.407, 0.01686710, "Mn2+", LM_TIP3P),
    "FE": LJ(1.353, 0.00941798, "Fe2+", LM_TIP3P),
    "CO": LJ(1.299, 0.00483892, "Co2+", LM_TIP3P),
    "NI": LJ(1.255, 0.00262320, "Ni2+", LM_TIP3P),
    "CU": LJ(1.218, 0.00148497, "Cu2+", LM_TIP3P),
    "ZN": LJ(1.271, 0.00330286, "Zn2+", LM_TIP3P),
    "CD": LJ(1.412, 0.01773416, "Cd2+", LM_TIP3P),
}

DEFAULT = COVALENT["C"]  # an element in neither table (reported by the analysis)
BOND_SEARCH = 1.45  # Å: a hydrogen's bonded atom is the nearest heavy atom of its residue within this (S-H: 1.34)


def lookup(element: str, alone: bool = False, bonded_to: str = "") -> Optional[LJ]:
    """Parameters of one atom: its element (upper case), whether it is alone in its residue, and for a hydrogen
    the element of the atom it is bonded to ("": none found).  None for an element in neither table."""
    if element == "H":
        if bonded_to:
            return HYDROGEN.get(bonded_to, HYDROGEN_OTHER)
        return HYDROGEN_ALONE
    if alone and element in IONS:
        return IONS[element]
    if element in COVALENT:
        return COVALENT[element]
    return IONS.get(element)  # a metal bound in a larger residue: its ion's values


def hydrogen_parents(u, atoms, elements: np.ndarray, box=None) -> np.ndarray:
    """For every atom of ``atoms``: the element of the heavy atom it is bonded to if it is a hydrogen, else "".

    The topology's bonds are used where it has them for the hydrogen; otherwise (a PDB without CONECT records, or
    with them for a ligand only) the nearest heavy atom of the hydrogen's residue within :data:`BOND_SEARCH`, at
    the current frame."""
    out = np.full(atoms.n_atoms, "", dtype=object)
    local = np.flatnonzero(elements == "H")
    if not len(local):
        return out
    h_ix = atoms.ix[local]
    try:
        bonds = np.asarray(u.bonds.indices)
    except Exception:  # noqa: BLE001 - no bonds in this topology
        bonds = np.empty((0, 2), dtype=np.int64)
    if len(bonds):
        is_h = np.zeros(u.atoms.n_atoms, dtype=bool)
        is_h[h_ix] = True
        first, second = is_h[bonds[:, 0]], is_h[bonds[:, 1]]
        hydrogens = np.concatenate([bonds[first, 0], bonds[second, 1]])
        partners = np.concatenate([bonds[first, 1], bonds[second, 0]])
        if len(partners):
            unique = np.unique(partners)
            partner_el = atom_elements(u.atoms[unique])[np.searchsorted(unique, partners)]
            where = {int(ix): k for k, ix in zip(local, h_ix)}
            for h, el in zip(hydrogens, partner_el):
                k = where[int(h)]
                if el != "H" and not out[k]:  # (a water's H-H bond of some topologies is no parent)
                    out[k] = el
    open_ = np.array([k for k in local if not out[k]], dtype=np.int64)
    if len(open_):
        seekers = atoms[open_]
        candidates = seekers.residues.atoms
        cand_el = atom_elements(candidates)
        heavy = candidates[cand_el != "H"]
        if heavy.n_atoms:
            heavy_el = cand_el[cand_el != "H"]
            i, j, d = pairs_within(seekers.positions.astype(np.float64), heavy.positions.astype(np.float64),
                                   BOND_SEARCH, box)
            best: Dict[int, Tuple[float, int]] = {}
            for hi, hj, dist in zip(i, j, d):
                if hi not in best or dist < best[hi][0]:
                    best[int(hi)] = (float(dist), int(hj))
            for hi, (_dist, hj) in best.items():
                out[open_[hi]] = heavy_el[hj]
    return out


def parameters(u, atoms, box=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(R*, epsilon, elements without parameters) for ``atoms`` of the universe ``u`` at its current frame;
    the atoms of the listed elements got :data:`DEFAULT`."""
    elements = atom_elements(atoms)
    sizes = np.bincount(u.atoms.resindices, minlength=u.residues.n_residues)
    alone = sizes[atoms.resindices] == 1
    parents = hydrogen_parents(u, atoms, elements, box)
    rstar = np.empty(atoms.n_atoms)
    eps = np.empty(atoms.n_atoms)
    missing: List[str] = []
    cache: Dict[Tuple[str, bool, str], LJ] = {}
    for k, key in enumerate(zip(elements, alone, parents)):
        lj = cache.get(key)
        if lj is None:
            lj = lookup(key[0], bool(key[1]), key[2])
            if lj is None:
                lj = DEFAULT
                if key[0] not in missing:
                    missing.append(key[0])
            cache[key] = lj
        rstar[k], eps[k] = lj.rstar, lj.epsilon
    return rstar, eps, sorted(missing)


def table_rows() -> List[Tuple[str, str, LJ]]:
    """(class, key, parameters) of every row, in the order of docs/GENERIC_LJ.md."""
    rows = [("molecule", el, lj) for el, lj in COVALENT.items()]
    rows += [("hydrogen", f"on {el}", lj) for el, lj in HYDROGEN.items()]
    rows += [("hydrogen", "on C / other", HYDROGEN_OTHER)]
    rows += [("ion", el, lj) for el, lj in IONS.items()]
    return rows
