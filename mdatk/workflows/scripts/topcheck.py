"""Checks of a finished Amber topology: net charge, the Mg2+ ions and their pair terms.

tleap and ParmEd finish without errors in several situations that silently spoil a
simulation with Mg2+, so the Amber build runs this on the final .parm7 and stops when
something is wrong:

* the system net charge (a periodic system that is not neutral gets a uniform PME
  background charge; usually a wrong ion count or an uncharged ion unit);
* the charge of every Mg2+ (residue MG) atom (+2);
* microMg / nanoMg: the ion's own Lennard-Jones terms are the published ones (they come
  from a frcmod keyed on an atom type name; if the ion carries another type they never
  apply);
* microMg / nanoMg: every solute atom type with Lennard-Jones terms has an explicit Mg2+
  pair term. With these models the ion's own LJ is a fitted effective potential that only
  reproduces Mg2+-water by the combination rule; a solute type without a pair term (e.g. a
  new oxygen type from an L-RNA library) binds Mg2+ several times too strongly.

The pair check reads the stored A/B coefficients, so it also catches a missing or misspelt
changeLJPair line, a mask that matched nothing and a later addLJType that reset a pair.
Standard library only (it runs where only AmberTools' python3 is available).
Based on the checker in the Mg bug report for MDATK 0.0.4.1.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

CHARGE_SCALE = 18.2223  # parm7 stores charges in units of e / 18.2223
# solvent = water residues (by residue name) and monatomic ions (any single-atom residue). Classified
# by residue, never by atom type name: NA is Amber's aromatic ring nitrogen (G N1, U N3), sodium is Na+
WATER = {"WAT", "HOH", "TIP3", "SOL"}
WANTED = {"POINTERS", "AMBER_ATOM_TYPE", "ATOM_TYPE_INDEX", "NONBONDED_PARM_INDEX",
          "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF", "CHARGE", "RESIDUE_LABEL", "RESIDUE_POINTER"}


def read_parm7(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    flag = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                flag = line.split()[1]
                if flag in WANTED:
                    out[flag] = []
                else:
                    flag = None
            elif line.startswith("%"):
                continue
            elif flag:
                out[flag].append(line.rstrip("\n"))
    missing = WANTED - set(out)
    if missing:
        raise ValueError(f"{path}: not an Amber topology (missing {', '.join(sorted(missing))})")
    return out


def _ints(lines: List[str]) -> List[int]:
    return [int(x) for x in " ".join(lines).split()]


def _floats(lines: List[str]) -> List[float]:
    return [float(x) for x in " ".join(lines).split()]


def _names(lines: List[str]) -> List[str]:
    """20a4 fields (type and residue names may contain '*' or '+')."""
    out: List[str] = []
    for line in lines:
        out += [line[i:i + 4].strip() for i in range(0, len(line.rstrip()), 4)]
    return [n for n in out if n]


def _lj(a: float, b: float) -> Tuple[Optional[float], Optional[float]]:
    """A/B coefficients -> (Rmin_ij, epsilon_ij); (None, None) for a zero pair."""
    if a <= 0.0 or b <= 0.0:
        return None, None
    return (2.0 * a / b) ** (1.0 / 6.0), b * b / (4.0 * a)


class Topology:
    def __init__(self, path: str):
        p = read_parm7(path)
        self.ntypes = _ints(p["POINTERS"])[1]
        self.atype = _names(p["AMBER_ATOM_TYPE"])
        self.tindex = _ints(p["ATOM_TYPE_INDEX"])
        self.ico = _ints(p["NONBONDED_PARM_INDEX"])
        self.acoef = _floats(p["LENNARD_JONES_ACOEF"])
        self.bcoef = _floats(p["LENNARD_JONES_BCOEF"])
        self.charge = [q / CHARGE_SCALE for q in _floats(p["CHARGE"])]
        labels = _names(p["RESIDUE_LABEL"])
        ptr = _ints(p["RESIDUE_POINTER"]) + [len(self.atype) + 1]
        self.resname = [""] * len(self.atype)
        self.solvent = [False] * len(self.atype)
        for r, lab in enumerate(labels):
            first, last = ptr[r] - 1, ptr[r + 1] - 1
            for i in range(first, last):
                self.resname[i] = lab
                self.solvent[i] = lab in WATER or last - first == 1

    def pair(self, i: int, j: int) -> Tuple[Optional[float], Optional[float]]:
        k = self.ico[self.ntypes * (i - 1) + (j - 1)]
        return _lj(self.acoef[k - 1], self.bcoef[k - 1]) if k > 0 else (None, None)


def check(path: str, mg_res: str = "MG", neutral: bool = True, mg_count: int = 0,
          ion_lj: Optional[Tuple[float, float]] = None, pairs: bool = False, model: str = "",
          pair_refs: Optional[Dict[str, List[float]]] = None) -> List[str]:
    """Problems found in the topology ([] = all good); prints a report."""
    top = Topology(path)
    problems: List[str] = []
    net = sum(top.charge)
    print(f"topology check of {path}")
    print(f"  net charge {net:+.4f} e")
    if neutral and abs(net) > 0.02:
        problems.append(f"the system is not neutral (net charge {net:+.4f} e): PME would add a uniform "
                        "background charge; check the ion counts and the charge of the ion units")
    # note (not a failure): Lennard-Jones types shared by solute and solvent (water, ion) atoms, e.g. HO
    # with TIP3P HW (both zero): a changeLJPair @%HO line (NBfix19) then also changes the water hydrogens
    by_type: Dict[int, Dict[str, int]] = {}
    solvent_types = {ti for ti, sv in zip(top.tindex, top.solvent) if sv}
    solute_types = {ti for ti, sv in zip(top.tindex, top.solvent) if not sv}
    for ti, name in zip(top.tindex, top.atype):
        if ti in solvent_types and ti in solute_types:
            by_type.setdefault(ti, {}).setdefault(name, 0)
            by_type[ti][name] += 1
    for ti, names in sorted(by_type.items()):
        if names:
            print(f"  note: Lennard-Jones type {ti} is shared by solute and solvent atoms: "
                  + ", ".join(f"{n} ({c})" for n, c in sorted(names.items()))
                  + "; pair terms set for one apply to all (ParmEd addLJType separates them)")
    mg = [i for i, r in enumerate(top.resname) if r == mg_res]
    print(f"  {len(mg)} {mg_res} atom(s)")
    if mg_count and len(mg) < mg_count:  # more is fine: the input structure may bring its own
        problems.append(f"{mg_count} Mg2+ ions were requested, the topology has {len(mg)} {mg_res} atoms")
    wrong_q = sorted({round(top.charge[i], 4) for i in mg if abs(top.charge[i] - 2.0) > 0.01})
    if wrong_q:
        problems.append(f"{mg_res} atoms with charge {', '.join(f'{q:+.4f}' for q in wrong_q)} e instead of "
                        "+2: the ion unit tleap added is not Mg2+ (check which library defines the unit "
                        f"{mg_res})")
    for t in sorted({top.tindex[i] for i in mg}):
        names = sorted({top.atype[i] for i in mg if top.tindex[i] == t})
        rmin, eps = top.pair(t, t)
        if rmin is None:
            problems.append(f"Mg2+ (type {'/'.join(names)}) has no Lennard-Jones terms")
            continue
        print(f"  Mg2+ type {'/'.join(names)}: Rmin/2 {rmin / 2:.4f} A, epsilon {eps:.4f} kcal/mol")
        if ion_lj and (abs(rmin / 2 - ion_lj[0]) > 5e-4 or abs(eps - ion_lj[1]) > 1e-4 * max(1.0, ion_lj[1])):
            problems.append(f"Mg2+ (type {'/'.join(names)}) has Rmin/2 {rmin / 2:.4f} A, epsilon {eps:.4f} "
                            f"kcal/mol, not the {model} values {ion_lj[0]:.4f} A / {ion_lj[1]:.4f} kcal/mol: "
                            "the ion parameter file did not apply to the ion's atom type")
        if not pairs:
            continue
        solute: Dict[int, Dict[str, int]] = {}
        for i, (ti, name) in enumerate(zip(top.tindex, top.atype)):
            if not top.solvent[i]:
                solute.setdefault(ti, {}).setdefault(name, 0)
                solute[ti][name] += 1
        bare = []
        for ti in sorted(solute):
            r_tt, e_tt = top.pair(ti, ti)
            if r_tt is None or ti == t:
                continue  # no LJ of its own (HO ...): nothing to override
            r_ij, e_ij = top.pair(t, ti)
            r_lb, e_lb = 0.5 * (rmin + r_tt), math.sqrt(eps * e_tt)
            explicit = r_ij is None or (abs(r_ij - r_lb) > 1e-4 * max(1.0, r_lb)
                                        or abs(e_ij - e_lb) > 1e-4 * max(1.0, e_lb))
            if not explicit:
                bare.append("/".join(f"{n} ({c})" for n, c in sorted(solute[ti].items())))
        # tleap merges atom types with equal Lennard-Jones terms into one LJ type, and changeLJPair acts
        # on LJ types: a mapped type sharing its LJ type with a type meant to get another Mg2+ pair
        # overwrites that type's pair (whichever line comes last wins)
        for ti in sorted(solute):
            wanted = {n: tuple(pair_refs[n]) for n in solute[ti] if pair_refs and n in pair_refs}
            if len(set(wanted.values())) > 1:
                problems.append(f"atom types {', '.join(sorted(wanted))} share one Lennard-Jones type but are "
                                "meant to get different Mg2+ pair terms; only one can apply. Map the "
                                "non-standard type onto the reference with the same Lennard-Jones terms "
                                "(its LJ twin)")
        if bare:
            problems.append(f"solute atom types without an explicit Mg2+ pair term (combination rule only, "
                            f"far too strong with {model}): {', '.join(bare)}. Map each type onto the "
                            "published type it is chemically equivalent to ('Extra Mg2+ pair types', e.g. "
                            "LE:OS) and rebuild")
        else:
            print(f"  every solute atom type with Lennard-Jones terms has an explicit Mg2+ pair term")
    for msg in problems:
        print(f"  PROBLEM: {msg}")
    print("  topology check " + ("FAILED" if problems else "passed"))
    return problems


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("usage: python3 THIS_SCRIPT TOPOLOGY.parm7", file=sys.stderr)
        return 2
    try:
        problems = check(argv[0], MG_RESIDUE, NEUTRAL, MG_COUNT, ION_LJ, PAIRS, MODEL, PAIR_REFS)
    except (OSError, ValueError, IndexError) as exc:
        print(f"topology check: {exc}", file=sys.stderr)
        return 1
    return 1 if problems else 0


# defaults when imported (the generated script sets its own)
MG_RESIDUE, NEUTRAL, MG_COUNT, ION_LJ, PAIRS, MODEL = "MG", True, 0, None, False, ""
PAIR_REFS: Dict[str, List[float]] = {}
