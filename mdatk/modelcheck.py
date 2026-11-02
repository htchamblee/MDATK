"""Plausibility checks of RNA models before they are simulated or refined further.

A decoy can be best-in-class on its score, its map fit and its base pairs and still be locally absurd: two
oxygens pressed together, a chain break that tleap will bond anyway, a loop threaded into a knot, a ribose
flipped to the other hand by real-space refinement.  Five checks, on one model or an ensemble, and a sixth
that is only reported:

clashes
    Heavy atoms closer than their van der Waals contact (Bondi radii) by at least ``clash_tol`` (0.4 Å),
    reported with the overlap so the severity shows.  An N/O pair with a donor on either side (2'-OH, the
    amino groups, G N1, U N3, terminal 5'/3' hydroxyls, A N1 / C N3 of residues given as protonated, water;
    the N/O atoms of other residues, whose protonation is not known, count as donors) is a hydrogen bond
    between ``hbond_min`` and ``hbond_max`` (2.2-3.2 Å), not a clash; closer than
    ``hbond_min`` there is no room for the hydrogen and it counts as a clash.  Two pure acceptors (a
    phosphate oxygen and O4', say) have no hydrogen to share at any distance.  A carbon one bond from an
    H-bonded N/O, against that atom's partner (or an atom bonded to it) and no closer to it than the
    H-bond itself, is the same contact seen from the next atom and is not counted either.  Atoms up to
    three bonds apart are left out: bonds come from the nucleotide templates, from distances within other
    residues and from the O3'-P link of consecutive nucleotides of a TER segment, never from residue
    numbers, so neighbouring residues (stacking, sugar-base) and every chain are checked.  Waters and ions
    are skipped unless asked for; ions are then checked against their first-shell distances (Mg2+-O 2.1 Å
    is coordination, not a clash).  Hydrogens are never part of it: their positions differ between
    programs, and the Arnott fiber helix (the builder's ideal A-form) already has H2'(i)...H6/H8(i+1) at
    1.57-1.70 Å with its H2' placed from the sugar's own geometry - the C2'(i)...C8/C6(i+1) of the fiber
    model is 3.5 Å - so smaller hydrogen radii would hide a geometry rather than fix a threshold.
links
    Every O3'(i)-P(i+1) of consecutive nucleotides within each TER segment, as tleap bonds them (by name,
    whatever the distance): up to ``link_max`` (2.0 Å) a bond; from there to ``link_fail`` (3.0 Å) a
    strained bond, a warning with the strain it puts into the first minimisation, E = k (r - r0)^2 with
    Amber's k (no 1/2; the OS-P term of parm99 / parm10.dat, 230 kcal/mol/Å² and 1.61 Å, unless read from
    the force field tleap loads: ``link_bond_from_files``), about 35 kcal/mol at 2.0 Å, 180 at 2.5 and 440
    at 3.0; longer than ``link_fail`` a break (the chain is usually in two pieces), shorter than
    ``link_min`` (1.4 Å) an overlap and a missing O3' or P an unmeasurable link, which fail.  Consecutive
    P-P outside 5.0-7.5 Å are reported too.
compaction
    End-to-end distance over contour length along C1' for loop segments, given as residue ranges or found
    as the residues unpaired in every connected dot-bracket (runs of three or more).  Below 0.20 a loop is
    flagged for inspection; there is no verdict above: a loop is meant to turn, and what distinguishes a
    tight turn from a threaded one is not in this number.
hands
    The sugar hand of every nucleotide from its geometry (:func:`mdatk.pdbtools.residue_hands`: all usable
    centres of C4', C3' and C1' must agree; the C4' hand alone is reported as well), against the hands
    expected for residue ranges, with epimerised and flat centres named.
kinks
    The virtual angle C1'(i-1)-C1'(i)-C1'(i+1) of every residue between two consecutive nucleotides of a TER
    segment (A-form: 152 degrees); below ``kink_min`` (60) a sharp kink is flagged, except at the positions
    where one is allowed (typed ranges, or the residues whose in-line probing reactivity reaches a threshold,
    read from a two-column file).  It gates a model only when asked to.
hydrogen contacts (reported only, never a gate or a filter; off unless asked for)
    MolProbity's conventions (Probe): the explicit hydrogens of the model only, none added; Probe's
    radii for hydrogens at nuclear positions (C-H 1.09 Å, as Amber, Rosetta and the builder place them),
    1.17 Å for a hydrogen on carbon and 1.00 Å for a polar or aromatic (base C-H) one, or 1.22 / 1.05 Å for
    electron-cloud positions (C-H 0.97 Å, Reduce's default for X-ray), chosen from the model's median C-H;
    Probe's heavy-atom radii (C 1.70, N 1.55, O 1.40, P 1.80 Å); an overlap of 0.4 Å or more is a clash,
    a polar hydrogen and an acceptor may overlap by 0.6 Å (a hydrogen bond) before the excess counts; the
    clashscore is clashes per 1000 atoms.  Pairs up to four bonds apart are left out.

Residues are addressed by chain, number and insertion code throughout; ranges use
:func:`mdatk.pdbtools.parse_ranges` syntax ('A:7-12, B:25-34, C:58-63', 'C', ':5-9' for a blank chain,
'*:45-73' for every chain).  Models can be gated (``max_clashes``, links, hands, P-P spacing, kinks, limits on
Rosetta score terms from a decoy table) and an ensemble filtered or ranked.  A filter fails only when fewer
than ``min_pass`` models pass (1: none); it passes on up to N of them and writes the slots it cannot fill as
empty-slot files (:func:`mdatk.pdbtools.empty_slot_text`), which every MDATK reader of models skips.

Standard library only, so that the planner can copy this module (with :mod:`mdatk.pdbtools`) into the scripts
it writes: ``mdatk pdb qc`` and the 'model-check' step run the same code.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

try:
    from . import pdbtools
except ImportError:  # copied into a generated script after the PDB tools: they are this module
    pdbtools = sys.modules[__name__]

CHECKS = ("clashes", "links", "compaction", "hands", "kinks")
ON_FAIL = ("report", "filter", "stop")
ORDERS = ("input", "clashes", "score")
H_RADII_MODES = ("auto", "nuclear", "ecloud")

CLASH_TOL = 0.4  # Å inside van der Waals contact
HBOND_WINDOW = (2.2, 3.2)  # Å, donor-acceptor distances that are a hydrogen bond, not a clash
LINK_RANGE = (1.4, 2.0)  # Å, O3'-P; ideal 1.60, past 2.0 a strained bond that minimisation has to repair
LINK_FAIL = 3.0  # Å, O3'-P longer than this: a break (the chain is usually in two pieces)
# The O3'-P bond tleap makes, for the strain a long link puts into the first minimisation: E = k (r - r0)^2
# with Amber's k (no 1/2).  OS-P of parm99 / parm10.dat (JCC 7 (1986) 230; leaprc.RNA.OL3, LJbb, ROC, YIL and
# Shaw load parm10.dat and none of their frcmods changes it; the CP type OR copies it), checked against a
# current AmberTools
LINK_K = 230.0  # kcal/mol/Å^2
LINK_R0 = 1.61  # Å
LINK_SOURCE = "MDATK's default, OS-P of Amber's parm99 / parm10.dat"
PP_RANGE = (5.0, 7.5)  # Å, consecutive phosphorus atoms
COMPACTION_FLAG = 0.20  # end-to-end / contour below this: inspect the loop
MIN_LOOP = 3  # residues of a loop segment found from dot-brackets
KINK_MIN = 60.0  # degrees, virtual C1'(i-1)-C1'(i)-C1'(i+1) below this: a sharp kink (A-form 152)

# Bondi van der Waals radii (Å): a flat distance cutoff would ignore that C...C contact is 0.36 Å longer than
# O...O, so the overlap below contact is what is counted and ranked
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80, "SE": 1.90, "F": 1.47, "CL": 1.75,
       "BR": 1.85, "I": 1.98}
VDW_OTHER = 1.70
# covalent radii (Å) for bonds found by distance (residues without a template, hydrogens)
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05, "SE": 1.20, "F": 0.57,
            "CL": 1.02, "BR": 1.20, "I": 1.39}
COVALENT_OTHER = 0.80
BOND_SLACK = 0.45  # Å beyond the covalent radii that still counts as a bond
# Typical first-shell ion-oxygen distances (Å).  With the clash tolerance below them, inner-sphere
# coordination (Mg2+-O 2.07, Na+-O 2.41, K+-O 2.80 ...) is never a clash; against carbon, phosphorus and other
# heavy atoms an ion gets this distance + 1.0 Å (the second shell), against hydrogens + 0.5 Å.
ION_CONTACT = {"MG": 2.1, "MN": 2.2, "ZN": 2.1, "CO": 2.1, "NI": 2.1, "FE": 2.1, "CU": 2.0, "CA": 2.4,
               "CD": 2.3, "SR": 2.6, "BA": 2.8, "LI": 2.0, "NA": 2.4, "K": 2.8, "RB": 2.9, "CS": 3.1}
HALIDES = frozenset({"CL", "BR", "I", "F"})  # anions: plain van der Waals contacts

# Heavy atoms that carry a donatable hydrogen, by base.  WHICH ring nitrogen donates depends on the base: G N1
# and U N3 carry a hydrogen, A N1 and C N3 do not (unless protonated), and in the pyrimidines N1 is the
# glycosidic nitrogen.  Every nucleotide adds O2' (RNA), and terminal hydroxyls O5' / O3' where there is no
# phosphate / no following link.  Everything else in RNA is an acceptor (phosphate oxygens, the ester
# O5'/O3'/O4', the carbonyls O2/O4/O6, the other ring nitrogens) or neither.
DONORS_BY_BASE = {"A": ("N6",), "G": ("N1", "N2"), "C": ("N4",), "U": ("N3",), "T": ("N3",)}
DONORS_UNKNOWN_BASE = ("N2", "N4", "N6")
PROTONATED_DONOR = {"A": "N1", "C": "N3"}  # A+ (N1-H) and C+ (N3-H)

# Hydrogen contacts on MolProbity's conventions (Probe 2.x, atomprops.h): the radius of a hydrogen goes with
# how far it sits from its parent.  At nuclear positions (C-H 1.09 Å: Amber, Rosetta, the builder) 1.17 Å on
# carbon and 1.00 Å for a polar (N-H, O-H) or aromatic (Probe's base C-H: A H2 H8, G H8, C/U H5 H6, T H6)
# hydrogen; at electron-cloud positions (C-H 0.97 Å, Reduce's default for X-ray models) 1.22 and 1.05 Å.
# Heavy atoms keep Probe's explicit-hydrogen radii (O 1.40 Å, not Bondi's 1.52).
H_RADII = {"nuclear": {"nonpolar": 1.17, "polar": 1.00, "aromatic": 1.00},
           "ecloud": {"nonpolar": 1.22, "polar": 1.05, "aromatic": 1.05}}
PROBE_HEAVY = {"C": 1.70, "N": 1.55, "O": 1.40, "P": 1.80, "S": 1.80, "SE": 1.90}
H_CLASH = 0.4  # Å of overlap: a clash in MolProbity's clashscore
H_HBOND_OVERLAP = 0.6  # Å a polar H and an acceptor may overlap (Probe's -HBRegular); the excess counts
# a median C-H below this (Å) means electron-cloud positions (0.93-0.98), above it nuclear ones (1.08-1.10)
ECLOUD_CH = 1.035
AROMATIC_H = {"A": ("H2", "H8"), "G": ("H8",), "C": ("H5", "H6"), "U": ("H5", "H6"), "T": ("H6",)}
# Ring nitrogens without a hydrogen accept one; amino and imino nitrogens do not (O always does)
ACCEPTOR_N = {"A": ("N1", "N3", "N7"), "G": ("N3", "N7"), "C": ("N3",), "U": (), "T": ()}

# Heavy-atom bonds of the nucleotide templates; other atoms of a nucleotide are bonded by distance
_SUGAR_BONDS = (("P", "OP1"), ("P", "OP2"), ("P", "OP3"), ("P", "O5'"), ("O5'", "C5'"), ("C5'", "C4'"),
                ("C4'", "O4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("C3'", "C2'"), ("C2'", "O2'"),
                ("C2'", "C1'"), ("C1'", "O4'"))
_PURINE_BONDS = (("C1'", "N9"), ("N9", "C8"), ("C8", "N7"), ("N7", "C5"), ("C5", "C6"), ("C6", "N1"),
                 ("N1", "C2"), ("C2", "N3"), ("N3", "C4"), ("C4", "C5"), ("C4", "N9"))
_PYRIMIDINE_BONDS = (("C1'", "N1"), ("N1", "C2"), ("C2", "N3"), ("N3", "C4"), ("C4", "C5"), ("C5", "C6"),
                     ("C6", "N1"))
_SUBSTITUENT_BONDS = {"A": (("C6", "N6"),), "G": (("C6", "O6"), ("C2", "N2")),
                      "C": (("C2", "O2"), ("C4", "N4")), "U": (("C2", "O2"), ("C4", "O4")),
                      "T": (("C2", "O2"), ("C4", "O4"), ("C5", "C7"))}
_ATOM_ALIASES = {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3", "C5M": "C7"}


def template_bonds(base: str) -> Tuple[Tuple[str, str], ...]:
    """Heavy-atom bonds of a nucleotide with this base (A, C, G, U, T)."""
    ring = _PURINE_BONDS if base in ("A", "G") else _PYRIMIDINE_BONDS
    return _SUGAR_BONDS + ring + _SUBSTITUENT_BONDS.get(base, ())


# --- settings -----------------------------------------------------------------------------------------------

@dataclass
class QCSettings:
    """What :func:`check_model` measures and which results fail a model (the gates)."""
    checks: Tuple[str, ...] = CHECKS
    clash_tol: float = CLASH_TOL
    hbond_min: float = HBOND_WINDOW[0]
    hbond_max: float = HBOND_WINDOW[1]
    waters: bool = False  # include waters in the clash check
    ions: str = "skip"  # "skip" or "check" (first-shell distances)
    protonated: str = ""  # ranges of protonated A+ (N1-H) / C+ (N3-H)
    hydrogen_check: bool = False  # the separate hydrogen-contact report (never a gate)
    hydrogen_radii: str = "auto"  # "auto" (from the median C-H), "nuclear" or "ecloud"
    link_min: float = LINK_RANGE[0]
    link_max: float = LINK_RANGE[1]  # longer: a strained link (a warning with its strain)
    link_fail: float = LINK_FAIL  # longer: a break (fails the links gate)
    link_bond: Optional["LinkBond"] = None  # k and r0 of the strain (None: MDATK's default)
    pp_min: float = PP_RANGE[0]
    pp_max: float = PP_RANGE[1]
    segments: str = ""  # compaction segments (ranges)
    dotbrackets: List[Tuple[str, str]] = field(default_factory=list)  # (name, text): loops unpaired in all
    compaction_flag: float = COMPACTION_FLAG
    expect_l: str = ""
    expect_d: str = ""
    kink_min: float = KINK_MIN  # degrees
    kink_allow: str = ""  # ranges where a sharp kink is allowed (e.g. flexible in in-line probing)
    max_clashes: Optional[int] = None  # gate: more clashes fail (None: no limit)
    gate_links: bool = True  # gate: a link beyond link_fail, shorter than link_min or unmeasurable fails
    gate_hands: bool = True  # gate: a hand against the expectation, an epimer or a flat centre fails
    gate_pp: bool = False  # gate: P-P spacing outside the range fails
    gate_kinks: bool = False  # gate: a sharp kink where none is allowed fails
    score_limits: str = ""  # gate: limits on decoy-table terms, e.g. 'linear_chainbreak<=1, score<=-50'
    notes: List[str] = field(default_factory=list)  # how the settings were read (force-field files ...)


# --- residues and atoms -------------------------------------------------------------------------------------

def _atom_name(name: str) -> str:
    name = name.replace("*", "'")
    return _ATOM_ALIASES.get(name, name)


def residue_kind(res) -> Tuple[str, Optional[str]]:
    """('nt', base) for a nucleotide (base from the atoms when they decide it, else from the name), 'water',
    'ion' (one heavy atom of an ion element or a halide), 'virtual' (Rosetta's VRT / XXX) or 'other', with
    None."""
    if res.name.strip().upper() in pdbtools.VIRTUAL_RESIDUES:
        return "virtual", None
    nt = pdbtools.nucleotide(res, infer=True)
    if nt is not None:
        return "nt", pdbtools.infer_base(a.name for a in res.atoms) or nt.base
    heavy = [a for a in res.atoms if not pdbtools.is_hydrogen(a, res.name)]
    if len(heavy) == 1:
        el = pdbtools.element_of(heavy[0], res.name)
        if el in ION_CONTACT or el in HALIDES:
            return "ion", None
        if el == "O":
            return "water", None
    if res.name.strip().upper() in pdbtools._WATER:
        return "water", None
    return "other", None


def _residue_label(res) -> str:
    return f"{res.chain or '_'}:{res.resseq}{res.icode}"


class _Model:
    """Flat atom table of the residues a clash check looks at, with its bond graph."""

    def __init__(self):
        self.xyz: List[Tuple[float, float, float]] = []
        self.name: List[str] = []
        self.el: List[str] = []
        self.res: List[int] = []  # residue index
        self.role: List[str] = []  # "heavy", "h" or "ion"
        self.donor: List[bool] = []
        self.bonded: List[List[int]] = []
        self.residues: List = []
        self.labels: List[str] = []  # residue labels
        self.bases: List[Optional[str]] = []  # base of each residue (None: not a nucleotide)

    def add(self, name, el, xyz, res_index, role, donor) -> int:
        self.xyz.append(xyz)
        self.name.append(name)
        self.el.append(el)
        self.res.append(res_index)
        self.role.append(role)
        self.donor.append(donor)
        self.bonded.append([])
        return len(self.xyz) - 1

    def bond(self, i: int, j: int) -> None:
        if j not in self.bonded[i]:
            self.bonded[i].append(j)
            self.bonded[j].append(i)

    def label(self, i: int) -> str:
        return f"{self.labels[self.res[i]]}:{self.name[i]}"


def _close_enough(m: _Model, i: int, j: int) -> bool:
    d = math.dist(m.xyz[i], m.xyz[j])
    bond = COVALENT.get(m.el[i], COVALENT_OTHER) + COVALENT.get(m.el[j], COVALENT_OTHER)
    return 0.4 < d < bond + BOND_SLACK


def build_model(structure, settings: QCSettings, hydrogens: bool = False) -> _Model:
    """The atom table and bond graph of ``structure`` (first altloc of every atom): heavy atoms for the clash
    check, with ``hydrogens`` the explicit hydrogens too (the hydrogen-contact check)."""
    m = _Model()
    protonated = pdbtools.parse_ranges(settings.protonated or None)
    for seg in structure.segments:
        kinds = [residue_kind(r) for r in seg]
        prev_o3: Optional[int] = None  # O3' of the previous nucleotide in this segment
        prev_c: Optional[int] = None  # carbonyl C of the previous other residue (peptide link)
        for k, res in enumerate(seg):
            kind, base = kinds[k]
            if (kind == "virtual" or (kind == "water" and not settings.waters)
                    or (kind == "ion" and settings.ions != "check")):
                prev_o3 = prev_c = None
                continue
            names = {_atom_name(a.name) for a in res.atoms}
            linked_next = (k + 1 < len(seg) and kinds[k + 1][0] == "nt"
                           and any(_atom_name(a.name) == "P" for a in seg[k + 1].atoms))
            donors: Optional[Set[str]] = None  # None: every N/O may donate (water, ligands, amino acids)
            if kind == "nt":
                donors = set(DONORS_BY_BASE.get(base or "", DONORS_UNKNOWN_BASE)) | {"O2'"}
                if "P" not in names:
                    donors.add("O5'")  # 5'-OH
                if not linked_next:
                    donors.add("O3'")  # 3'-OH
                if base in PROTONATED_DONOR and pdbtools.in_ranges(res, protonated):
                    donors.add(PROTONATED_DONOR[base])
            ri = len(m.residues)
            m.residues.append(res)
            m.labels.append(_residue_label(res))
            m.bases.append(base if kind == "nt" else None)
            index: Dict[str, int] = {}
            hs: List[int] = []
            for atom in res.atoms:
                name = _atom_name(atom.name)
                if name in index:
                    continue  # a later alternate location
                h = pdbtools.is_hydrogen(atom, res.name)
                if h and not hydrogens:
                    continue
                el = "H" if h else pdbtools.element_of(atom, res.name)
                # halides keep plain van der Waals contacts
                role = "h" if h else ("ion" if kind == "ion" and el in ION_CONTACT else "heavy")
                donor = el in ("N", "O") and (donors is None or name in donors)
                i = m.add(name, el, atom.xyz, ri, role, donor)
                index[name] = i
                if h:
                    hs.append(i)
            heavy = [i for i in index.values() if m.role[i] == "heavy"]
            if kind == "nt" and base:
                bonds = template_bonds(base)
                for a, b in bonds:
                    if a in index and b in index:
                        m.bond(index[a], index[b])
                known = {n for pair in bonds for n in pair}
                extra = [i for i in heavy if m.name[i] not in known]
                for i in extra:  # e.g. a modification: bonded to whatever it touches
                    for j in heavy:
                        if j != i and _close_enough(m, i, j):
                            m.bond(i, j)
            elif kind != "ion":
                for x, i in enumerate(heavy):
                    for j in heavy[x + 1:]:
                        if _close_enough(m, i, j):
                            m.bond(i, j)
            for hi in hs:  # each hydrogen on its nearest heavy atom of the residue
                near = min(heavy, key=lambda j: math.dist(m.xyz[hi], m.xyz[j]), default=None)
                if near is not None and math.dist(m.xyz[hi], m.xyz[near]) < 1.35:
                    m.bond(hi, near)
            if kind == "nt":
                if prev_o3 is not None and "P" in index:
                    m.bond(prev_o3, index["P"])  # the backbone link tleap / Rosetta make, whatever its length
                prev_o3 = index.get("O3'")
                prev_c = None
            elif kind == "other":
                if prev_c is not None and "N" in index and math.dist(m.xyz[prev_c], m.xyz[index["N"]]) < 2.0:
                    m.bond(prev_c, index["N"])  # peptide bond
                prev_c = index.get("C")
                prev_o3 = None
            else:
                prev_o3 = prev_c = None
    return m


def _bond_distances(m: _Model, i: int, cache: Dict[int, Dict[int, int]]) -> Dict[int, int]:
    """Atoms up to four bonds from ``i`` -> the number of bonds."""
    found = cache.get(i)
    if found is None:
        found = {i: 0}
        shell = [i]
        for depth in range(1, 5):
            shell = [b for a in shell for b in m.bonded[a] if b not in found]
            for b in shell:
                found.setdefault(b, depth)
        cache[i] = found
    return found


def pairs_within(xyz: Sequence[Tuple[float, float, float]], cutoff: float
                 ) -> Iterator[Tuple[int, int, float]]:
    """(i, j, distance) of every pair of points closer than ``cutoff`` (uniform grid, i < j): the standard
    library twin of :func:`mdatk.spatial.pairs_within` for one set of points, so this module can be copied
    into generated scripts."""
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for i, (x, y, z) in enumerate(xyz):
        cell = (math.floor(x / cutoff), math.floor(y / cutoff), math.floor(z / cutoff))
        grid.setdefault(cell, []).append(i)
    half = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
    c2 = cutoff * cutoff
    for (gx, gy, gz), members in grid.items():
        for x, i in enumerate(members):
            pi = xyz[i]
            for j in members[x + 1:]:
                d2 = (pi[0] - xyz[j][0]) ** 2 + (pi[1] - xyz[j][1]) ** 2 + (pi[2] - xyz[j][2]) ** 2
                if d2 < c2:
                    yield (i, j, math.sqrt(d2)) if i < j else (j, i, math.sqrt(d2))
        for a, b, c in half:
            other = grid.get((gx + a, gy + b, gz + c))
            if not other:
                continue
            for i in members:
                pi = xyz[i]
                for j in other:
                    d2 = (pi[0] - xyz[j][0]) ** 2 + (pi[1] - xyz[j][1]) ** 2 + (pi[2] - xyz[j][2]) ** 2
                    if d2 < c2:
                        yield (i, j, math.sqrt(d2)) if i < j else (j, i, math.sqrt(d2))


# --- clashes ------------------------------------------------------------------------------------------------

class Contact(NamedTuple):
    """Two atoms closer than their contact distance: overlap (contact - distance), distance, atom labels
    ('C:63:O2''), kind ('steric', 'donor-acceptor' closer than the H-bond minimum, 'ion', 'h-bond')."""
    overlap: float
    distance: float
    first: str
    second: str
    kind: str


@dataclass
class ClashResult:
    clashes: List[Contact] = field(default_factory=list)  # counted, largest overlap first
    hbond_contacts: List[Contact] = field(default_factory=list)  # H-bonded pairs inside contact: excused
    shadows: List[Contact] = field(default_factory=list)  # carbon contacts beside an H-bond, excused
    atoms: int = 0

    @property
    def short_hbonds(self) -> int:
        return sum(1 for c in self.clashes if c.kind == "donor-acceptor")

    @property
    def worst(self) -> float:
        return self.clashes[0].overlap if self.clashes else 0.0


def _ion_contact(el: str, other_el: str, other_role: str) -> float:
    base = ION_CONTACT.get(el, 2.4)
    if other_role == "ion":
        return base + ION_CONTACT.get(other_el, 2.4)
    if other_el in ("N", "O"):
        return base
    return base + (0.5 if other_role == "h" else 1.0)


def check_clashes(structure, settings: Optional[QCSettings] = None) -> ClashResult:
    """Clashes of the heavy atoms of one model (see the module docstring for the chemistry); hydrogens are
    left to :func:`check_hydrogen_contacts`."""
    st = settings or QCSettings()
    m = build_model(structure, st)
    out = ClashResult(atoms=len(m.xyz))
    if not m.xyz:
        return out
    tol, lo, hi = st.clash_tol, st.hbond_min, st.hbond_max
    radius = [VDW.get(e, VDW_OTHER) for e in m.el]
    biggest = max(max(radius) * 2 - tol, hi)
    if any(r == "ion" for r in m.role):
        biggest = max(biggest, max(ION_CONTACT.values()) * 2 - tol, max(ION_CONTACT.values()) + 1.0 - tol)
    is_no = [e in ("N", "O") and r == "heavy" for e, r in zip(m.el, m.role)]
    cache: Dict[int, Dict[int, int]] = {}
    candidates: List[Tuple[float, float, int, int, str]] = []
    partners: Dict[int, Dict[int, float]] = {}  # H-bonded heavy atoms (any distance in the window)
    for i, j, d in pairs_within(m.xyz, biggest + 1e-6):
        ri, rj = m.role[i], m.role[j]
        bonds = _bond_distances(m, i, cache).get(j)
        if bonds is not None and bonds <= 3:  # atoms up to three bonds apart are never checked
            continue
        if "ion" in (ri, rj):
            ion, other = (i, j) if ri == "ion" else (j, i)
            contact = _ion_contact(m.el[ion], m.el[other], m.role[other])
            if contact - d >= tol:
                candidates.append((contact - d, d, i, j, "ion"))
            continue
        contact = radius[i] + radius[j]
        if is_no[i] and is_no[j] and (m.donor[i] or m.donor[j]):
            if d < lo:  # no room for the hydrogen: worse than an ordinary clash
                candidates.append((contact - d, d, i, j, "donor-acceptor"))
                continue
            if d <= hi:
                partners.setdefault(i, {})[j] = d
                partners.setdefault(j, {})[i] = d
                if contact - d >= tol:
                    out.hbond_contacts.append(Contact(round(contact - d, 3), round(d, 3), m.label(i),
                                                      m.label(j), "h-bond"))
                continue
        if contact - d >= tol:
            candidates.append((contact - d, d, i, j, "steric"))

    def shadow(i: int, j: int, d: float) -> bool:
        # a carbon one bond from an H-bonded N/O, against that atom's partner (or an atom bonded to it), no
        # closer to it than the H-bond itself: pulled in by the bond, not a collision of its own.  A carbon
        # closer than that is not beside the H-bond but in the way of it, and counts.
        for a, b in ((i, j), (j, i)):
            if m.el[a] != "C":
                continue
            for n in m.bonded[a]:
                mates = partners.get(n, {})
                for x in [b] + m.bonded[b]:
                    if x in mates and d >= mates[x]:
                        return True
        return False

    for overlap, d, i, j, kind in candidates:
        contact = Contact(round(overlap, 3), round(d, 3), m.label(i), m.label(j), kind)
        if kind == "steric" and shadow(i, j, d):
            out.shadows.append(contact)
        else:
            out.clashes.append(contact)
    for items in (out.clashes, out.hbond_contacts, out.shadows):
        items.sort(key=lambda c: (-c.overlap, c.first, c.second))
    return out


# --- hydrogen contacts (MolProbity's conventions; reported, never a gate) ----------------------------------

@dataclass
class HydrogenResult:
    """Clashes that involve an explicit hydrogen (kind 'steric', or 'h-bond' for a polar hydrogen pressed into
    an acceptor beyond the 0.6 Å a hydrogen bond may overlap: the overlap given is the excess)."""
    clashes: List[Contact] = field(default_factory=list)  # largest overlap first
    atoms: int = 0  # all atoms checked, hydrogens included (the clashscore's denominator)
    hydrogens: int = 0
    radii: str = ""  # "nuclear" or "ecloud" ("" without hydrogens)
    median_ch: Optional[float] = None  # Å, median C-H of the model

    @property
    def clashscore(self) -> Optional[float]:
        """Clashes per 1000 atoms (MolProbity's clashscore); None without hydrogens."""
        return round(1000.0 * len(self.clashes) / self.atoms, 2) if self.hydrogens and self.atoms else None

    @property
    def worst(self) -> float:
        return self.clashes[0].overlap if self.clashes else 0.0


def _h_class(m: _Model, h: int) -> str:
    parent = m.bonded[h][0] if m.bonded[h] else None
    if parent is not None and m.el[parent] in ("N", "O"):
        return "polar"
    base = m.bases[m.res[h]]
    if parent is not None and base and m.name[h] in AROMATIC_H.get(base, ()):
        return "aromatic"
    return "nonpolar"


def _acceptor(m: _Model, i: int) -> bool:
    if m.role[i] != "heavy" or m.el[i] not in ("N", "O"):
        return False
    base = m.bases[m.res[i]]
    if m.el[i] == "O" or not base:
        return True  # an N of another residue: its protonation is not known
    return m.name[i] in ACCEPTOR_N.get(base, ()) and not m.donor[i]  # (a protonated A N1 / C N3 donates)


def check_hydrogen_contacts(structure, settings: Optional[QCSettings] = None) -> HydrogenResult:
    """Contacts of the model's explicit hydrogens on MolProbity's conventions (see the module docstring);
    waters and ions are left out, as in the clashscore.  Never a gate."""
    st = settings or QCSettings()
    m = build_model(structure, replace(st, waters=False, ions="skip"), hydrogens=True)
    hs = [i for i, r in enumerate(m.role) if r == "h"]
    out = HydrogenResult(atoms=len(m.xyz), hydrogens=len(hs))
    if not hs:
        return out
    ch = sorted(math.dist(m.xyz[h], m.xyz[m.bonded[h][0]]) for h in hs
                if m.bonded[h] and m.el[m.bonded[h][0]] == "C")
    out.median_ch = round(ch[len(ch) // 2], 3) if ch else None
    if st.hydrogen_radii in ("nuclear", "ecloud"):
        out.radii = st.hydrogen_radii
    else:
        out.radii = "ecloud" if out.median_ch is not None and out.median_ch < ECLOUD_CH else "nuclear"
    table = H_RADII[out.radii]
    radius = [table[_h_class(m, i)] if r == "h" else PROBE_HEAVY.get(e, VDW.get(e, VDW_OTHER))
              for i, (e, r) in enumerate(zip(m.el, m.role))]
    polar = {h for h in hs if _h_class(m, h) == "polar"}
    cache: Dict[int, Dict[int, int]] = {}
    for i, j, d in pairs_within(m.xyz, max(radius) + max(table.values()) + 1e-6):
        if m.role[i] != "h" and m.role[j] != "h":
            continue  # heavy-atom pairs are the clash check's
        bonds = _bond_distances(m, i, cache).get(j)
        if bonds is not None and bonds <= 4:  # a hydrogen's position follows from the torsions up to there
            continue
        overlap = radius[i] + radius[j] - d
        h, x = (i, j) if m.role[i] == "h" else (j, i)
        kind = "steric"
        if h in polar and _acceptor(m, x):  # a hydrogen bond may overlap by 0.6 Å; only the excess clashes
            overlap -= H_HBOND_OVERLAP
            kind = "h-bond"
        if overlap >= H_CLASH:
            out.clashes.append(Contact(round(overlap, 3), round(d, 3), m.label(i), m.label(j), kind))
    out.clashes.sort(key=lambda c: (-c.overlap, c.first, c.second))
    return out


# --- backbone -----------------------------------------------------------------------------------------------

class LinkBond(NamedTuple):
    """The O3'-P bond tleap makes: force constant k (kcal/mol/Å², Amber's convention: E = k (r - r0)^2, no
    1/2), length r0 (Å) and where they come from."""
    k: float
    r0: float
    source: str

    def strain(self, r: float) -> float:
        return self.k * (r - self.r0) ** 2


DEFAULT_LINK_BOND = LinkBond(LINK_K, LINK_R0, LINK_SOURCE)


class Link(NamedTuple):
    """A backbone link of consecutive nucleotides: residue labels, distance (None: unmeasurable), status
    ('ok', 'strained' between the link maximum and the failure length, 'long' beyond it, 'short', 'missing
    O3'' / 'missing P', or 'p-p' for a P-P spacing outlier) and, for a measured link that is not 'ok', the
    strain E = k (r - r0)^2 in kcal/mol that tleap's bond puts into the first minimisation."""
    first: str
    second: str
    distance: Optional[float]
    status: str
    strain: Optional[float] = None


@dataclass
class BackboneResult:
    links: List[Link] = field(default_factory=list)  # every O3'-P link of consecutive nucleotides
    pp: List[Link] = field(default_factory=list)  # P-P spacings outside the range
    pp_measured: int = 0
    bond: LinkBond = DEFAULT_LINK_BOND

    @property
    def problems(self) -> List[Link]:
        """Links that fail: longer than the failure length, too short or unmeasurable."""
        return [k for k in self.links if k.status not in ("ok", "strained")]

    @property
    def strained(self) -> List[Link]:
        """Links between the link maximum and the failure length: bonded by tleap under strain (a warning)."""
        return [k for k in self.links if k.status == "strained"]

    @property
    def strain(self) -> float:
        """Sum of the strain of the links that are not 'ok' (kcal/mol)."""
        return round(sum(k.strain or 0.0 for k in self.links), 1)

    @property
    def worst(self) -> Optional[Link]:
        measured = [k for k in self.links if k.distance is not None]
        return max(measured, key=lambda k: k.distance) if measured else None


def check_backbone(structure, link_min: float = LINK_RANGE[0], link_max: float = LINK_RANGE[1],
                   pp_min: float = PP_RANGE[0], pp_max: float = PP_RANGE[1], link_fail: float = LINK_FAIL,
                   bond: Optional[LinkBond] = None) -> BackboneResult:
    """O3'(i)-P(i+1) of consecutive nucleotides within every TER segment (what tleap bonds, gaps in the
    numbering included), with the strain of those beyond the limits, and consecutive P-P spacings."""
    bond = bond or DEFAULT_LINK_BOND
    out = BackboneResult(bond=bond)
    for seg in structure.segments:
        prev = prev_xyz = None
        for res in seg:
            if pdbtools.nucleotide(res, infer=True) is None:
                prev = None
                continue
            xyz = {_atom_name(n): v for n, v in res.coords().items()}
            if prev is not None:
                o3, p = prev_xyz.get("O3'"), xyz.get("P")
                a, b = _residue_label(prev), _residue_label(res)
                if o3 is None or p is None:
                    out.links.append(Link(a, b, None, "missing O3'" if o3 is None else "missing P"))
                else:
                    d = math.dist(o3, p)
                    status = ("long" if d > link_fail else "strained" if d > link_max else
                              "short" if d < link_min else "ok")
                    strain = None if status == "ok" else round(bond.strain(d), 1)
                    out.links.append(Link(a, b, round(d, 3), status, strain))
                if "P" in prev_xyz and p is not None:
                    out.pp_measured += 1
                    d = math.dist(prev_xyz["P"], p)
                    if not pp_min <= d <= pp_max:
                        out.pp.append(Link(a, b, round(d, 3), "p-p"))
            prev, prev_xyz = res, xyz
    return out


# The force field tleap loads, for k and r0 of the O3'-P bond: a tleap input or leaprc is followed through its
# 'source' and 'loadamberparams' lines (tleap's commands are case-insensitive), parm*.dat and frcmod files are
# read for their BOND terms, in load order, a later term replacing an earlier one as in tleap
_LEAP_PARAMS = re.compile(r"^\s*(?:\w+\s*=\s*)?loadamberparams\s+\"?([^\s\"]+)", re.IGNORECASE)
_LEAP_SOURCE = re.compile(r"^\s*source\s+\"?([^\s\"]+)", re.IGNORECASE)
LEAP_FOLDERS = ("parm", "cmd", "lib", "prep")  # tleap's default search path under $AMBERHOME/dat/leap


def amber_home(given: str = "") -> str:
    """AMBERHOME as given, else from the environment, else the installation of tleap on PATH ('' if none)."""
    home = given or os.environ.get("AMBERHOME", "")
    if not home:
        tleap = shutil.which("tleap")
        home = os.path.dirname(os.path.dirname(os.path.realpath(tleap))) if tleap else ""
    return home


def bond_terms(text: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """BOND terms of a parm*.dat or frcmod text: (type, type) in sorted order -> (k, r0).  A bond line has its
    two types in columns 1-5 ('OS-P '); angle and dihedral keys are longer ('P -OS-CT')."""
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for line in text.splitlines():
        if len(line) < 6 or line[2] != "-" or line[5] == "-":
            continue
        a, b = line[:2].strip(), line[3:5].strip()
        parts = line[5:].split()
        try:
            k, r0 = float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            continue
        if a and b:
            out[(a, b) if a <= b else (b, a)] = (k, r0)
    return out


def _leap_path(name: str, amberhome: str) -> str:
    """A file a tleap command names: here (tleap's working folder), else in tleap's search path."""
    if os.path.isfile(name):
        return name
    if amberhome and not os.path.isabs(name):
        for sub in LEAP_FOLDERS:
            path = os.path.join(amberhome, "dat", "leap", sub, name)
            if os.path.isfile(path):
                return path
    return ""


def link_bond_from_files(paths: Sequence[str], types: Sequence[str] = ("OS",), amberhome: str = ""
                         ) -> Tuple[LinkBond, List[str]]:
    """k and r0 of the P-O3' bond from the force field: ``paths`` are tleap inputs, leaprc, parm*.dat or
    frcmod files, read in order; ``types`` are the atom types O3' may have, the first one defined wins (OR,
    then OS for the CP force field).  Returns MDATK's default (with a note) when no file defines the bond;
    files that cannot be found or read are named in the notes and skipped."""
    home = amber_home(amberhome)
    terms: Dict[Tuple[str, str], Tuple[float, float]] = {}
    where: Dict[Tuple[str, str], str] = {}
    notes: List[str] = []
    seen: Set[str] = set()

    def read(path: str, depth: int) -> None:
        real = os.path.realpath(path)
        if real in seen or depth > 8:
            return
        seen.add(real)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            notes.append(f"cannot read {path}: {exc.strerror or exc}")
            return
        lines = text.splitlines()
        if any(_LEAP_PARAMS.match(ln) or _LEAP_SOURCE.match(ln) for ln in lines):  # tleap commands
            for ln in lines:
                m = _LEAP_SOURCE.match(ln) or _LEAP_PARAMS.match(ln)
                if not m:
                    continue
                found = _leap_path(m.group(1), home)
                if found:
                    read(found, depth + 1)
                else:
                    # (a file the build writes later, such as frcmod.cp, copies terms of those read)
                    notes.append(f"{m.group(1)} (named in {os.path.basename(path)}): not found"
                                 + ("" if home else " (no AMBERHOME)") + " or not written yet, skipped")
            return
        for key, value in bond_terms(text).items():
            terms[key] = value
            where[key] = path

    for p in paths:
        read(p, 0)
    for t in types:
        key = (t, "P") if t <= "P" else ("P", t)
        if key in terms:
            k, r0 = terms[key]
            return LinkBond(k, r0, f"{t}-P of {where[key]}"), notes
    wanted = " / ".join(f"{t}-P" for t in types)
    return DEFAULT_LINK_BOND, notes + [f"no {wanted} bond term in the files read: {LINK_SOURCE}"]


# --- compaction ---------------------------------------------------------------------------------------------

class Segment(NamedTuple):
    """End-to-end over contour length along C1' of a loop segment (ratio None: fewer than three C1')."""
    name: str
    residues: int
    ratio: Optional[float]


def _ratio(residues) -> Tuple[int, Optional[float]]:
    pts = [res.coords()["C1'"] for res in residues if "C1'" in res.coords()]
    if len(pts) < 3:
        return len(pts), None
    contour = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    return len(pts), (round(math.dist(pts[0], pts[-1]) / contour, 3) if contour else None)


def range_segments(structure, spec) -> List[Tuple[str, list]]:
    """(name, nucleotides) for every range of ``spec``, one segment per chain ('*' ranges per chain)."""
    out = []
    for chain, start, end in pdbtools.parse_ranges(spec or None):
        by_chain: Dict[str, list] = {}
        for res in structure.residues():
            if pdbtools.in_ranges(res, [(chain, start, end)]) and pdbtools.nucleotide(res, infer=True):
                by_chain.setdefault(res.chain, []).append(res)
        for c, residues in by_chain.items():
            numbers = "" if start is None else (f":{start}" + (f"-{end}" if end != start else ""))
            out.append((f"{c}{numbers}" if c or numbers else "_", residues))
    return out


def dotbracket_line(text: str) -> str:
    """The dot-bracket of a secondary-structure file (Rosetta secstruct, MDATK's .secstruct, Vienna .dbn):
    the first line, after '>' headers, with bracket or dot characters; strand separators (spaces, '&')
    removed, a trailing free energy '(-12.30)' dropped."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith(">") or not re.search(r"[.()\[\]{}<>]", line):
            continue
        line = re.sub(r"\s+\(\s*-?\d+(?:\.\d+)?\s*\)\s*$", "", line)
        chars = re.sub(r"[\s&]", "", line)
        bad = sorted(set(chars) - set(".-,:_()[]{}<>") - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                     - set("abcdefghijklmnopqrstuvwxyz"))
        if bad:
            raise ValueError(f"unexpected characters in the dot-bracket: {''.join(bad)}")
        return chars
    raise ValueError("no dot-bracket line found")


def loop_segments(structure, dotbrackets: Sequence[Tuple[str, str]], min_length: int = MIN_LOOP
                  ) -> Tuple[List[Tuple[str, list]], List[str]]:
    """Segments of residues unpaired in every dot-bracket (runs of ``min_length`` or more consecutive
    nucleotides of one TER segment); dot-bracket position k is the k-th nucleotide of the model in file
    order.  Returns the segments and notes (dot-brackets of another length are not used)."""
    order = [(si, res) for si, seg in enumerate(structure.segments) for res in seg
             if pdbtools.nucleotide(res, infer=True)]
    unpaired: Optional[Set[int]] = None
    notes = []
    for name, text in dotbrackets:
        try:
            db = dotbracket_line(text)
        except ValueError as exc:
            notes.append(f"{name}: {exc}; not used")
            continue
        if len(db) != len(order):
            notes.append(f"{name}: {len(db)} positions but the model has {len(order)} nucleotides; not used")
            continue
        free = {k for k, ch in enumerate(db) if ch in ".-,:_"}
        unpaired = free if unpaired is None else unpaired & free
    if not unpaired:
        return [], notes
    runs: List[List[int]] = []
    for k in sorted(unpaired):
        if (runs and runs[-1][-1] == k - 1 and order[k][0] == order[k - 1][0]
                and order[k][1].chain == order[k - 1][1].chain):
            runs[-1].append(k)
        else:
            runs.append([k])
    out = []
    for run in runs:
        if len(run) >= min_length:
            residues = [order[k][1] for k in run]
            out.append((pdbtools.ranges_text(residues), residues))
    return out, notes


def compaction(structure, segments: Sequence[Tuple[str, list]]) -> List[Segment]:
    return [Segment(name, *_ratio(residues)) for name, residues in segments]


# --- kinks --------------------------------------------------------------------------------------------------

class Kink(NamedTuple):
    """The virtual angle C1'(i-1)-C1'(i)-C1'(i+1) at a residue (degrees), and whether a kink is allowed
    there."""
    residue: str
    angle: float
    allowed: bool


@dataclass
class KinkResult:
    flagged: List[Kink] = field(default_factory=list)  # below the minimum where no kink is allowed
    allowed: List[Kink] = field(default_factory=list)  # below the minimum where one is allowed
    measured: int = 0
    sharpest: Optional[Kink] = None


def check_kinks(structure, kink_min: float = KINK_MIN, allow=None) -> KinkResult:
    """Virtual C1' angles of every nucleotide between two consecutive nucleotides of its TER segment (a
    residue that is not a nucleotide, or has no C1', breaks the run); below ``kink_min`` flagged, unless the
    residue is in the ranges ``allow`` (where a sharp kink is allowed)."""
    allowed = pdbtools.parse_ranges(allow or None)
    out = KinkResult()
    for seg in structure.segments:
        run: List[Tuple[object, Tuple[float, float, float]]] = []
        for res in list(seg) + [None]:
            c1 = None
            if res is not None and pdbtools.nucleotide(res, infer=True):
                c1 = next((xyz for n, xyz in res.coords().items() if _atom_name(n) == "C1'"), None)
            if c1 is not None:
                run.append((res, c1))
                continue
            for k in range(1, len(run) - 1):
                a, (mid, b), c = run[k - 1][1], run[k], run[k + 1][1]
                u = [a[x] - b[x] for x in range(3)]
                v = [c[x] - b[x] for x in range(3)]
                nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
                if not nu or not nv:
                    continue
                cos = max(-1.0, min(1.0, sum(p * q for p, q in zip(u, v)) / (nu * nv)))
                kink = Kink(_residue_label(mid), round(math.degrees(math.acos(cos)), 1),
                            bool(allowed) and pdbtools.in_ranges(mid, allowed))
                out.measured += 1
                if out.sharpest is None or kink.angle < out.sharpest.angle:
                    out.sharpest = kink
                if kink.angle < kink_min:
                    (out.allowed if kink.allowed else out.flagged).append(kink)
            run = []
    return out


def reactive_positions(path: str, minimum: float) -> List[int]:
    """Residue numbers whose reactivity reaches ``minimum`` in a two-column file (position, reactivity; tabs,
    commas or spaces; '#' comments, a header line and values that are not numbers - 'NaN', '-' - skipped)."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p for p in re.split(r"[\s,;]+", line.split("#", 1)[0].strip()) if p]
            if len(parts) < 2:
                continue
            try:
                pos, value = int(float(parts[0])), float(parts[1])
            except ValueError:
                continue
            if value == value and value >= minimum:  # NaN never reaches it
                out.append(pos)
    return sorted(set(out))


# --- decoy table --------------------------------------------------------------------------------------------

class DecoyTable:
    """A decoy table: columns 'run', 'tag', 'pdb' (path relative to the plan root, empty when the decoy was
    not extracted), then Rosetta score terms by name; one row per decoy.  Extra columns are kept, missing
    ones are tolerated; TSV, or CSV when the header has no tab."""

    def __init__(self, path: str, root: Optional[str] = None):
        self.path = path
        self.root = os.path.abspath(root if root is not None else os.getcwd())
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines:
            raise ValueError(f"{path}: empty decoy table")
        delimiter = "\t" if "\t" in lines[0] else ","
        reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
        rows = list(reader)
        self.columns = [c.strip() for c in rows[0]]
        self.rows: List[Dict[str, str]] = [dict(zip(self.columns, (v.strip() for v in r))) for r in rows[1:]]

    def _resolved(self, pdb: str) -> str:
        return os.path.normpath(pdb if os.path.isabs(pdb) else os.path.join(self.root, pdb))

    def row_for(self, model: str) -> Optional[Dict[str, str]]:
        """The row of a model file: by its path (relative to the plan root), the real path, else a unique
        file-name match, else a unique tag equal to the file name without extension."""
        want = os.path.normpath(os.path.abspath(model))
        with_pdb = [r for r in self.rows if r.get("pdb")]
        for r in with_pdb:
            if self._resolved(r["pdb"]) == want:
                return r
        real = os.path.realpath(want)
        for r in with_pdb:
            if os.path.realpath(self._resolved(r["pdb"])) == real:
                return r
        base = os.path.basename(model)
        same = [r for r in with_pdb if os.path.basename(r["pdb"]) == base]
        if len(same) == 1:
            return same[0]
        stem = os.path.splitext(base)[0]
        tagged = [r for r in self.rows if r.get("tag") == stem]
        return tagged[0] if len(tagged) == 1 else None

    @staticmethod
    def value(row: Optional[Dict[str, str]], term: str) -> Optional[float]:
        try:
            return float(row[term]) if row else None
        except (KeyError, TypeError, ValueError):
            return None


def read_decoy_table(path: str, root: Optional[str] = None) -> DecoyTable:
    return DecoyTable(path, root)


_LIMIT = re.compile(r"^([A-Za-z_][\w.:-]*)\s*(<=|>=|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$")
_COMPARE = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
            ">": lambda a, b: a > b}


def parse_score_limits(text: str) -> List[Tuple[str, str, float]]:
    """'linear_chainbreak<=1, score<=-50' -> [(term, operator, value)]; ValueError for anything else."""
    out = []
    for item in re.split(r"[,;\n]+", text or ""):
        item = item.strip()
        if not item:
            continue
        m = _LIMIT.match(item)
        if not m:
            raise ValueError(f"cannot read the score limit {item!r} (expected e.g. linear_chainbreak<=1)")
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


# --- one model ----------------------------------------------------------------------------------------------

@dataclass
class ModelReport:
    """Everything :func:`check_model` found for one model; ``failed`` lists the gates it fails."""
    path: str
    error: str = ""
    notes: List[str] = field(default_factory=list)
    residues: int = 0
    nucleotides: int = 0
    clashes: Optional[ClashResult] = None
    backbone: Optional[BackboneResult] = None
    compaction: List[Segment] = field(default_factory=list)
    hands: list = field(default_factory=list)  # pdbtools.ResidueHand rows
    hand_notes: List[str] = field(default_factory=list)
    hand_problems: List[str] = field(default_factory=list)
    kinks: Optional[KinkResult] = None
    hydrogens: Optional[HydrogenResult] = None  # reported only, never a gate
    score_row: Optional[Dict[str, str]] = None
    failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # reported, not failing (strained links, kinks ungated)

    @property
    def status(self) -> str:
        return "error" if self.error else ("fail" if self.failed else "pass")

    @property
    def worst_compaction(self) -> Optional[float]:
        ratios = [s.ratio for s in self.compaction if s.ratio is not None]
        return min(ratios) if ratios else None


def evaluate_gates(r: ModelReport, st: QCSettings) -> List[str]:
    """The gates a model fails, as short reasons."""
    out = []
    if r.clashes is not None and st.max_clashes is not None and len(r.clashes.clashes) > st.max_clashes:
        out.append(f"{len(r.clashes.clashes)} clashes (at most {st.max_clashes})")
    if r.backbone is not None and st.gate_links and r.backbone.problems:
        out.append(f"{len(r.backbone.problems)} O3'-P link(s) longer than {st.link_fail:g} A, shorter than "
                   f"{st.link_min:g} A or unmeasurable")
    if r.kinks is not None and st.gate_kinks and r.kinks.flagged:
        out.append(f"{len(r.kinks.flagged)} sharp kink(s): virtual C1' angle below {st.kink_min:g} deg ("
                   + ", ".join(f"{k.residue} {k.angle:.0f}" for k in r.kinks.flagged[:6]) + ")")
    if r.backbone is not None and st.gate_pp and r.backbone.pp:
        out.append(f"{len(r.backbone.pp)} P-P spacing(s) outside {st.pp_min:g}-{st.pp_max:g} A")
    if "hands" in st.checks and st.gate_hands:
        wrong = [h for h in r.hands if h.expected and h.call != h.expected and not h.defect]
        defects = [h for h in r.hands if h.defect]
        if wrong:
            out.append(f"{len(wrong)} sugar(s) not measured as the expected hand "
                       f"({pdbtools.ranges_text([h.residue for h in wrong])})")
        if defects:
            out.append(f"{len(defects)} sugar(s) with an epimerised, contradicting or flat centre "
                       f"({pdbtools.ranges_text([h.residue for h in defects])})")
    limits = parse_score_limits(st.score_limits)
    if limits:
        if r.score_row is None:
            out.append("no row in the decoy table")
        else:
            for term, op, value in limits:
                have = DecoyTable.value(r.score_row, term)
                if have is None:
                    out.append(f"no score term {term}")
                elif not _COMPARE[op](have, value):
                    out.append(f"{term} {have:g} (limit {op}{value:g})")
    return out


def evaluate_warnings(r: ModelReport, st: QCSettings) -> List[str]:
    """What is reported against a model without failing it: links bonded under strain, sharp kinks that do
    not gate."""
    out = []
    b = r.backbone
    if b is not None and b.strained:
        out.append(f"{len(b.strained)} O3'-P link(s) of {st.link_max:g}-{st.link_fail:g} A, strain "
                   f"{sum(k.strain or 0.0 for k in b.strained):.0f} kcal/mol ("
                   + ", ".join(f"{k.first}-{k.second} {k.distance:.2f} A {k.strain:.0f}"
                               for k in b.strained[:4])
                   + (" ..." if len(b.strained) > 4 else "") + ")")
    if r.kinks is not None and r.kinks.flagged and not st.gate_kinks:
        out.append(f"{len(r.kinks.flagged)} sharp kink(s) below {st.kink_min:g} deg ("
                   + ", ".join(f"{k.residue} {k.angle:.0f}" for k in r.kinks.flagged[:6]) + ")")
    return out


def check_model(source, settings: Optional[QCSettings] = None, table: Optional[DecoyTable] = None,
                path: str = "") -> ModelReport:
    """Run the checks of ``settings`` on one model (path, PDB text or Structure) and evaluate the gates."""
    st = settings or QCSettings()
    label = path or (source if isinstance(source, str) and "\n" not in source else "model")
    r = ModelReport(str(label))
    try:
        s = pdbtools.read_pdb(source)
    except (OSError, pdbtools.PDBError) as exc:
        r.error = f"cannot read the model: {exc}"
        r.failed = [r.error]
        return r
    r.notes = list(s.notes)
    residues = list(s.residues())
    r.residues = len(residues)
    r.nucleotides = sum(1 for res in residues if pdbtools.nucleotide(res, infer=True))
    if "clashes" in st.checks:
        r.clashes = check_clashes(s, st)
    if st.hydrogen_check:
        r.hydrogens = check_hydrogen_contacts(s, st)
    if "links" in st.checks:
        r.backbone = check_backbone(s, st.link_min, st.link_max, st.pp_min, st.pp_max, st.link_fail,
                                    st.link_bond)
    if "kinks" in st.checks:
        r.kinks = check_kinks(s, st.kink_min, st.kink_allow)
    if "compaction" in st.checks:
        segments = range_segments(s, st.segments)
        if st.dotbrackets:
            loops, notes = loop_segments(s, st.dotbrackets)
            segments += loops
            r.notes += notes
        r.compaction = compaction(s, segments)
    if "hands" in st.checks:
        r.hands = pdbtools.residue_hands(s, st.expect_l or None, st.expect_d or None)
        r.hand_notes, r.hand_problems = pdbtools.hand_summary(r.hands, bool(st.expect_l or st.expect_d))
    if table is not None:
        r.score_row = table.row_for(r.path)
    r.failed = evaluate_gates(r, st)
    r.warnings = evaluate_warnings(r, st)
    return r


# --- reports ------------------------------------------------------------------------------------------------

def _fmt(x: Optional[float], digits: int = 2) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def hand_counts(r: ModelReport) -> str:
    """'A:16D B:28D C:29L' - the measured hands per chain ('?' for undetermined)."""
    by_chain: Dict[str, Dict[str, int]] = {}
    for h in r.hands:
        c = by_chain.setdefault(h.residue.chain or "_", {})
        c[h.call or "?"] = c.get(h.call or "?", 0) + 1
    return " ".join(f"{chain}:" + "+".join(f"{n}{hand}" for hand, n in sorted(counts.items()))
                    for chain, counts in by_chain.items())


def report_header(score_terms: Sequence[str] = (), hydrogens: bool = False) -> List[str]:
    """Columns of the report table; the hydrogen-contact columns only when that check ran."""
    return (["model", "status", "failed", "warnings", "residues", "atoms", "clashes", "worst_overlap",
             "short_hbonds", "hbond_contacts", "shadows", "links", "link_problems", "strained_links",
             "link_strain", "worst_link", "worst_link_at", "pp_outside", "compaction", "worst_compaction",
             "hands", "hand_problems", "kinks", "sharpest_kink", "sharpest_kink_at"]
            + (["h_atoms", "h_clashes", "h_worst_overlap", "h_clashscore", "h_radii"] if hydrogens else [])
            + list(score_terms))


def report_row(r: ModelReport, score_terms: Sequence[str] = (), hydrogens: bool = False) -> List[str]:
    c, b, kk, h = r.clashes, r.backbone, r.kinks, r.hydrogens
    worst = b.worst if b else None
    sharp = kk.sharpest if kk else None
    row = [r.path, r.status, "; ".join(r.failed), "; ".join(r.warnings), str(r.residues),
           str(c.atoms) if c else "", str(len(c.clashes)) if c else "", _fmt(c.worst) if c else "",
           str(c.short_hbonds) if c else "", str(len(c.hbond_contacts)) if c else "",
           str(len(c.shadows)) if c else "",
           str(len(b.links)) if b else "", str(len(b.problems)) if b else "",
           str(len(b.strained)) if b else "", _fmt(b.strain, 1) if b else "",
           _fmt(worst.distance) if worst else "", f"{worst.first}-{worst.second}" if worst else "",
           str(len(b.pp)) if b else "",
           " ".join(f"{s.name}={_fmt(s.ratio)}" for s in r.compaction), _fmt(r.worst_compaction),
           hand_counts(r), str(len(r.hand_problems)) if r.hands else "",
           str(len(kk.flagged)) if kk else "", _fmt(sharp.angle, 1) if sharp else "",
           sharp.residue if sharp else ""]
    if hydrogens:
        on = h is not None and h.hydrogens > 0
        row += [str(h.atoms) if on else "", str(len(h.clashes)) if on else "", _fmt(h.worst) if on else "",
                _fmt(h.clashscore) if on else "", h.radii if on else ("no hydrogens" if h else "")]
    for term in score_terms:
        row.append(r.score_row.get(term, "") if r.score_row else "")
    return [v.replace("\t", " ").replace("\n", " ") for v in row]


def tsv_text(reports: Sequence[ModelReport], score_terms: Sequence[str] = ()) -> str:
    hydrogens = any(r.hydrogens is not None for r in reports)
    lines = ["\t".join(report_header(score_terms, hydrogens))]
    lines += ["\t".join(report_row(r, score_terms, hydrogens)) for r in reports]
    return "\n".join(lines) + "\n"


def report_data(r: ModelReport, score_terms: Sequence[str] = ()) -> Dict[str, object]:
    """JSON-ready details of one model."""
    out: Dict[str, object] = {"model": r.path, "status": r.status, "failed": r.failed, "warnings": r.warnings,
                              "notes": r.notes, "residues": r.residues, "nucleotides": r.nucleotides}
    if r.error:
        out["error"] = r.error
    if r.clashes is not None:
        out["clashes"] = {"atoms": r.clashes.atoms, "count": len(r.clashes.clashes),
                          "list": [c._asdict() for c in r.clashes.clashes],
                          "hbond_contacts": len(r.clashes.hbond_contacts), "shadows": len(r.clashes.shadows)}
    if r.backbone is not None:
        b = r.backbone
        out["links"] = {"count": len(b.links), "problems": [k._asdict() for k in b.problems],
                        "strained": [k._asdict() for k in b.strained], "strain": b.strain,
                        "bond": b.bond._asdict(), "worst": b.worst._asdict() if b.worst else None,
                        "pp_outside": [k._asdict() for k in b.pp]}
    if r.compaction:
        out["compaction"] = [s._asdict() for s in r.compaction]
    if r.hands:
        out["hands"] = {"by_chain": hand_counts(r), "notes": r.hand_notes, "problems": r.hand_problems}
    if r.kinks is not None:
        out["kinks"] = {"measured": r.kinks.measured, "flagged": [k._asdict() for k in r.kinks.flagged],
                        "allowed": [k._asdict() for k in r.kinks.allowed],
                        "sharpest": r.kinks.sharpest._asdict() if r.kinks.sharpest else None}
    if r.hydrogens is not None:
        h = r.hydrogens
        out["hydrogen_contacts"] = {"gate": False, "atoms": h.atoms, "hydrogens": h.hydrogens,
                                    "radii": h.radii, "median_ch": h.median_ch, "count": len(h.clashes),
                                    "clashscore": h.clashscore, "list": [c._asdict() for c in h.clashes]}
    if score_terms:
        out["scores"] = {t: (r.score_row or {}).get(t) for t in score_terms}
    return out


def _hydrogen_text(h: HydrogenResult, limit: int) -> List[str]:
    if not h.hydrogens:
        return ["  hydrogen contacts (reported only): the model has no hydrogens (explicit ones only, none "
                "are added)"]
    t = H_RADII[h.radii]
    ch = f"median C-H {h.median_ch:.2f} A" if h.median_ch is not None else "no C-H"
    out = [f"  hydrogen contacts (reported only, never a gate; MolProbity's conventions: {h.hydrogens} "
           f"explicit H, Probe's radii for {'electron-cloud' if h.radii == 'ecloud' else 'nuclear'} "
           f"positions ({ch}): H {t['nonpolar']:.2f} on carbon, {t['polar']:.2f} polar / aromatic; "
           f"overlap >= {H_CLASH:g} A, an H-bond {H_HBOND_OVERLAP:g} A more): {len(h.clashes)}, clashscore "
           f"{_fmt(h.clashscore)} per 1000 atoms ({h.atoms} atoms)"]
    for k in h.clashes[:limit]:
        why = f"; beyond the {H_HBOND_OVERLAP:g} A of an H-bond" if k.kind == "h-bond" else ""
        out.append(f"      {k.first:<14s} - {k.second:<14s} {k.distance:.2f} A ({k.overlap:.2f} "
                   f"overlap{why})")
    if len(h.clashes) > limit:
        out.append(f"      ... and {len(h.clashes) - limit} more")
    return out


def text_report(r: ModelReport, st: QCSettings, limit: int = 8) -> str:
    """Human-readable summary of one model."""
    out = [f"=== {r.path} ==="]
    if r.error:
        return "\n".join(out + [f"  ERROR: {r.error}"]) + "\n"
    out.append(f"  {r.residues} residues ({r.nucleotides} nucleotides)")
    out += [f"  note: {n}" for n in r.notes]
    c = r.clashes
    if c is not None:
        out.append(f"  clashes (heavy atoms, overlap >= {st.clash_tol:g} A inside van der Waals contact, "
                   f"{c.atoms} atoms): {len(c.clashes)}")
        for k in c.clashes[:limit]:
            why = {"donor-acceptor": f"; donor-acceptor closer than {st.hbond_min:g} A",
                   "ion": "; ion first shell"}.get(k.kind, "")
            out.append(f"      {k.first:<14s} - {k.second:<14s} {k.distance:.2f} A ({k.overlap:.2f} inside "
                       f"contact{why})")
        if len(c.clashes) > limit:
            out.append(f"      ... and {len(c.clashes) - limit} more")
        if c.hbond_contacts or c.shadows:
            out.append(f"  not counted: {len(c.hbond_contacts)} donor-acceptor contact(s) at hydrogen-bond "
                       f"distance ({st.hbond_min:g}-{st.hbond_max:g} A)"
                       + (f", {len(c.shadows)} carbon contact(s) beside one" if c.shadows else ""))
    b = r.backbone
    if b is not None:
        worst = b.worst
        out.append(f"  backbone: {len(b.links)} O3'-P links, {len(b.strained)} strained ({st.link_max:g}-"
                   f"{st.link_fail:g} A), {len(b.problems)} longer than {st.link_fail:g} A, shorter than "
                   f"{st.link_min:g} A or unmeasurable"
                   + (f" (longest {worst.first}-{worst.second} {worst.distance:.2f} A)" if worst else "")
                   + f"; {len(b.pp)} of {b.pp_measured} P-P outside {st.pp_min:g}-{st.pp_max:g} A")
        if b.strained or any(k.strain is not None for k in b.problems):
            out.append(f"      strain E = k (r - r0)^2 that tleap's bond puts into the first minimisation: k "
                       f"{b.bond.k:g} kcal/mol/A^2, r0 {b.bond.r0:g} A ({b.bond.source}); {b.strain:.0f} "
                       "kcal/mol in all")
        for k in (b.problems + b.strained + b.pp)[:limit]:
            what = f"{k.distance:.2f} A" if k.distance is not None else k.status
            strain = f"; strain {k.strain:.0f} kcal/mol" if k.strain is not None else ""
            out.append(f"      {k.first} -> {k.second}: {what} ({'P-P' if k.status == 'p-p' else k.status}"
                       f"{strain})")
    if r.compaction:
        out.append("  compaction (end-to-end / contour along C1'):")
        for s in r.compaction:
            coiled = s.ratio is not None and s.ratio < st.compaction_flag
            tag = "  <- extremely coiled, inspect it" if coiled else ""
            value = _fmt(s.ratio) or "n/a (fewer than 3 C1')"
            out.append(f"      {s.name:<14s} {value}{tag}")
    elif "compaction" in st.checks and (st.segments or st.dotbrackets):
        out.append("  compaction: no segment found")
    if r.hands:
        out.append(f"  chirality: {hand_counts(r)}")
        out += [f"      {n}" for n in r.hand_notes[1:]]
        out += [f"      {p}" for p in r.hand_problems[:limit]]
        if len(r.hand_problems) > limit:
            out.append(f"      ... and {len(r.hand_problems) - limit} more")
    kk = r.kinks
    if kk is not None:
        sharp = kk.sharpest
        out.append(f"  kinks (virtual C1' angle, {kk.measured} measured): {len(kk.flagged)} below "
                   f"{st.kink_min:g} deg" + (f", {len(kk.allowed)} more where allowed" if kk.allowed else "")
                   + (f" (sharpest {sharp.residue} {sharp.angle:.0f} deg)" if sharp else "")
                   + ("" if st.gate_kinks else "; not a gate"))
        for k in (kk.flagged + kk.allowed)[:limit]:
            out.append(f"      {k.residue:<10s} {k.angle:5.1f} deg" + ("  (allowed)" if k.allowed else ""))
    if r.hydrogens is not None:
        out += _hydrogen_text(r.hydrogens, limit)
    if r.score_row is not None:
        out.append("  decoy table: " + ", ".join(f"{k}={v}" for k, v in r.score_row.items()
                                                 if k not in ("run", "pdb") and v != "")[:300])
    out.append(f"  status: {r.status.upper()}" + (f" ({'; '.join(r.failed)})" if r.failed else "")
               + (f"; warnings: {'; '.join(r.warnings)}" if r.warnings else ""))
    return "\n".join(out) + "\n"


def ranked(reports: Sequence[ModelReport], order: str = "input", rank_term: str = "score"
           ) -> List[ModelReport]:
    """Models in input order, by fewest clashes (then smallest worst overlap), or by a decoy-table term
    (lowest first); ties keep the input order, unreadable models go last."""
    keyed = list(enumerate(reports))
    if order == "clashes":
        keyed.sort(key=lambda kr: (bool(kr[1].error), len(kr[1].clashes.clashes) if kr[1].clashes else 0,
                                   kr[1].clashes.worst if kr[1].clashes else 0.0, kr[0]))
    elif order == "score":
        def key(kr):
            v = DecoyTable.value(kr[1].score_row, rank_term)
            return (bool(kr[1].error), v is None, v if v is not None else 0.0, kr[0])
        keyed.sort(key=key)
    return [r for _, r in keyed]


# --- command line (``mdatk pdb qc`` and the copies in generated scripts) ---------------------------------

def add_qc_arguments(p) -> None:
    """The options of 'mdatk pdb qc' on an argparse parser."""
    p.add_argument("models", nargs="+", help="PDB files")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="JSON details instead of the text report")
    out.add_argument("--tsv", action="store_true", help="the report table (one row per model)")
    p.add_argument("--output", "-o", help="write the JSON / TSV here and the text report to stdout")
    p.add_argument("--checks", default=",".join(CHECKS), help="checks to run (default: all of "
                   f"{', '.join(CHECKS)})")
    g = p.add_argument_group("clashes")
    g.add_argument("--clash-tol", type=float, default=CLASH_TOL, help="overlap inside van der Waals contact "
                   f"that counts as a clash, in A (default {CLASH_TOL})")
    g.add_argument("--hbond-min", type=float, default=HBOND_WINDOW[0], help="shortest donor-acceptor "
                   f"distance that is a hydrogen bond, not a clash (default {HBOND_WINDOW[0]})")
    g.add_argument("--hbond-max", type=float, default=HBOND_WINDOW[1], help="longest such distance (default "
                   f"{HBOND_WINDOW[1]})")
    g.add_argument("--waters", action="store_true", help="include waters")
    g.add_argument("--ions", choices=("skip", "check"), default="skip", help="ions: skip (default) or check "
                   "against their first-shell distances")
    g.add_argument("--protonated", default="", help="residues with protonated A+ (N1) / C+ (N3), e.g. "
                   "'B:30'")
    g = p.add_argument_group("hydrogen contacts (reported only, never a gate)")
    g.add_argument("--hydrogen-check", action="store_true", help="also report the contacts of the model's "
                   "explicit hydrogens on MolProbity's conventions (Probe's radii, overlap >= 0.4 A, "
                   "clashscore); the clash check itself never includes hydrogens")
    g.add_argument("--hydrogen-radii", choices=H_RADII_MODES, default="auto", help="Probe's radii for "
                   "hydrogens at nuclear positions (1.17 / 1.00 A) or electron-cloud positions (1.22 / "
                   "1.05 A); auto (default): from the model's median C-H")
    g = p.add_argument_group("backbone")
    for flag, value, what in (("--link-min", LINK_RANGE[0], "shortest O3'-P link"),
                              ("--link-max", LINK_RANGE[1], "longest O3'-P link that is not strained"),
                              ("--link-fail", LINK_FAIL, "longest O3'-P link that is not a break (between "
                                                         "--link-max and this: a warning with its strain)"),
                              ("--pp-min", PP_RANGE[0], "shortest P-P spacing"),
                              ("--pp-max", PP_RANGE[1], "longest P-P spacing")):
        g.add_argument(flag, type=float, default=value, help=f"{what} (default {value})")
    g.add_argument("--link-k", type=float, help="force constant of the O3'-P bond for the strain, "
                   f"kcal/mol/A^2 (Amber's E = k (r - r0)^2; default {LINK_K:g}, or read with --link-params)")
    g.add_argument("--link-r0", type=float, help=f"its length in A (default {LINK_R0:g})")
    g.add_argument("--link-params", action="append", default=[], metavar="FILE", help="read k and r0 from "
                   "the force field: a tleap input or leaprc (its source / loadamberparams lines followed), "
                   "a parm*.dat or frcmod file; repeatable, in load order")
    g.add_argument("--link-types", default="OS", help="atom types of O3' to look up as P-TYPE, the first "
                   "defined wins (default OS; 'OR,OS' for the CP force field)")
    g.add_argument("--amberhome", default="", help="AmberTools folder for files named in a tleap input "
                   "(default: $AMBERHOME, else from tleap on PATH)")
    g = p.add_argument_group("compaction")
    g.add_argument("--segments", default="", help="loop segments, e.g. 'A:7-12, B:25-34, C:58-63'")
    g.add_argument("--secstruct", action="append", default=[], help="secondary-structure file; residues "
                   "unpaired in every one given form the loop segments (repeatable)")
    g.add_argument("--compaction-flag", type=float, default=COMPACTION_FLAG, help="flag loops whose "
                   f"end-to-end / contour ratio is below this (default {COMPACTION_FLAG})")
    g = p.add_argument_group("chirality")
    g.add_argument("--expect-l", default="", help="residues whose sugars must be L, e.g. 'C:45-73'")
    g.add_argument("--expect-d", default="", help="residues whose sugars must be D, e.g. 'A,B'")
    g = p.add_argument_group("kinks")
    g.add_argument("--kink-min", type=float, default=KINK_MIN, help="flag a virtual C1'(i-1)-C1'(i)-C1'(i+1) "
                   f"angle below this, in degrees (default {KINK_MIN:g}; A-form 152)")
    g.add_argument("--kink-allow", default="", help="residues where a sharp kink is allowed (not flagged), "
                   "e.g. the flexible positions of in-line probing: 'B:22-25, B:31'")
    g.add_argument("--kink-reactivity", metavar="FILE", help="two-column reactivity file (position, "
                   "reactivity): positions at or above --kink-reactivity-min are allowed too")
    g.add_argument("--kink-reactivity-min", type=float, help="the reactivity from which a kink is allowed")
    g.add_argument("--kink-reactivity-chain", default="*", help="chain of the file's positions (default: "
                   "every chain)")
    g = p.add_argument_group("decoy table")
    g.add_argument("--scores", help="decoy table (TSV: run, tag, pdb, score terms ...)")
    g.add_argument("--plan-root", help="folder the table's pdb paths are relative to (default: the current "
                   "one)")
    g.add_argument("--score-terms", default="score", help="score terms shown in the report (default: score)")
    g = p.add_argument_group("gates")
    g.add_argument("--max-clashes", type=int, help="fail models with more clashes")
    g.add_argument("--no-gate-links", action="store_true", help="broken (longer than --link-fail), short or "
                   "unmeasurable links do not fail a model")
    g.add_argument("--no-gate-hands", action="store_true", help="unexpected hands, epimers and flat centres "
                   "do not fail a model")
    g.add_argument("--gate-pp", action="store_true", help="P-P spacings outside the range fail a model")
    g.add_argument("--gate-kinks", action="store_true", help="a sharp kink where none is allowed fails a "
                   "model (default: reported)")
    g.add_argument("--score-limits", default="", help="limits on decoy-table terms, e.g. "
                   "'linear_chainbreak<=1, score<=-50'")
    g.add_argument("--on-fail", choices=ON_FAIL, default="report", help="report: only report (default); "
                   "filter: pass on the models that pass the gates (up to --pass N of them; fail only when "
                   "fewer than --min-pass pass); stop: exit with status 1 when any model fails")
    g.add_argument("--pass", dest="n_pass", type=int, metavar="N", help="with --on-fail filter: pass on up "
                   "to N passing models in N slots, the slots that cannot be filled written as empty-slot "
                   "files (default: every model that passes, no slots)")
    g.add_argument("--min-pass", type=int, default=1, metavar="M", help="with --on-fail filter: fail when "
                   "fewer than M models pass (default 1: only when none does; 0: never)")
    g.add_argument("--order", choices=ORDERS, default="input", help="order of the passed models: input "
                   "(default), clashes (fewest first) or score (lowest decoy-table --rank-term first)")
    g.add_argument("--rank-term", default="score", help="decoy-table term for --order score (default: "
                   "score)")
    g.add_argument("--passed-out", help="copy the passed models to these files: a pattern with {n} (1, 2 "
                   "...), e.g. 'passed-{n:02d}.pdb'")
    g.add_argument("--best-out", help="copy the first passed model to this file (an empty slot when none "
                   "passed and --min-pass is 0)")


def settings_from_args(args) -> QCSettings:
    checks = tuple(c.strip() for c in args.checks.split(",") if c.strip())
    unknown = [c for c in checks if c not in CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s) {', '.join(unknown)} (choose from {', '.join(CHECKS)})")
    dotbrackets = []
    for path in args.secstruct:
        with open(path, encoding="utf-8", errors="replace") as fh:
            dotbrackets.append((os.path.basename(path), fh.read()))
    notes: List[str] = []
    bond = DEFAULT_LINK_BOND
    if args.link_params:
        types = [t for t in re.split(r"[,\s]+", args.link_types or "") if t] or ["OS"]
        bond, notes = link_bond_from_files(args.link_params, types, args.amberhome)
    if args.link_k is not None or args.link_r0 is not None:
        given = ", ".join(f"{what} {v:g}" for what, v in (("k", args.link_k), ("r0", args.link_r0))
                          if v is not None)
        bond = LinkBond(bond.k if args.link_k is None else args.link_k,
                        bond.r0 if args.link_r0 is None else args.link_r0,
                        f"{given} as given" + ("" if args.link_k is not None and args.link_r0 is not None
                                               else f", the rest from {bond.source}"))
    allow = args.kink_allow
    if args.kink_reactivity:
        if args.kink_reactivity_min is None:
            raise ValueError("--kink-reactivity needs --kink-reactivity-min (the reactivity from which a "
                             "kink is allowed)")
        positions = reactive_positions(args.kink_reactivity, args.kink_reactivity_min)
        chain = (args.kink_reactivity_chain or "*").strip()
        runs: List[List[int]] = []
        for pos in positions:
            if runs and runs[-1][-1] == pos - 1:
                runs[-1].append(pos)
            else:
                runs.append([pos])
        items = [allow.strip()] if allow.strip() else []
        items += [f"{chain}:{r[0]}" + (f"-{r[-1]}" if len(r) > 1 else "") for r in runs]
        allow = ", ".join(items)
        notes.append(f"{os.path.basename(args.kink_reactivity)}: {len(positions)} position(s) with "
                     f"reactivity >= {args.kink_reactivity_min:g}, where a sharp kink is allowed")
    st = QCSettings(checks=checks, clash_tol=args.clash_tol, hbond_min=args.hbond_min,
                    hbond_max=args.hbond_max, waters=args.waters, ions=args.ions, protonated=args.protonated,
                    hydrogen_check=args.hydrogen_check, hydrogen_radii=args.hydrogen_radii,
                    link_min=args.link_min, link_max=args.link_max, link_fail=args.link_fail, link_bond=bond,
                    pp_min=args.pp_min, pp_max=args.pp_max, segments=args.segments, dotbrackets=dotbrackets,
                    compaction_flag=args.compaction_flag, expect_l=args.expect_l, expect_d=args.expect_d,
                    kink_min=args.kink_min, kink_allow=allow, max_clashes=args.max_clashes,
                    gate_links=not args.no_gate_links, gate_hands=not args.no_gate_hands,
                    gate_pp=args.gate_pp, gate_kinks=args.gate_kinks, score_limits=args.score_limits,
                    notes=notes)
    validate_settings(st)
    return st


def validate_settings(st: QCSettings) -> None:
    """ValueError for impossible settings (empty windows, unreadable ranges or limits)."""
    for lo, hi, what in ((st.hbond_min, st.hbond_max, "H-bond window"),
                         (st.link_min, st.link_max, "O3'-P limits"), (st.pp_min, st.pp_max, "P-P range")):
        if not lo < hi:
            raise ValueError(f"{what}: the minimum ({lo:g}) must be below the maximum ({hi:g})")
    if st.link_fail < st.link_max:
        raise ValueError(f"the O3'-P failure length ({st.link_fail:g}) must not be below the longest link "
                         f"that is not strained ({st.link_max:g})")
    if st.link_bond is not None and (st.link_bond.k < 0 or st.link_bond.r0 <= 0):
        raise ValueError("the O3'-P bond needs k >= 0 and r0 > 0")
    if st.hydrogen_radii not in H_RADII_MODES:
        raise ValueError(f"hydrogen radii: one of {', '.join(H_RADII_MODES)}")
    for text, what in ((st.segments, "segments"), (st.expect_l, "expected L"), (st.expect_d, "expected D"),
                       (st.protonated, "protonated residues"), (st.kink_allow, "kinks allowed")):
        try:
            pdbtools.parse_ranges(text or None)
        except pdbtools.PDBError as exc:
            raise ValueError(f"{what}: {exc}") from None
    parse_score_limits(st.score_limits)


def run_qc(args, stdout=None, stderr=None) -> Tuple[int, List[ModelReport]]:
    """Execute a parsed 'qc' command; returns (exit status, reports).  The report is written first, with 'M of
    K models passed'.  Inputs that are empty slots (of an earlier filter) are skipped with a note.  Status 1
    (and nothing copied): a model fails with --on-fail stop, fewer than --min-pass pass with filter, no model
    is left to check, or, whatever the policy, a model cannot be read.  With filter and --pass N the passed
    models fill N slots, and the slots left over are written as empty-slot files."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    st = settings_from_args(args)
    if args.n_pass is not None and args.n_pass < 1:
        raise ValueError("--pass: at least 1 slot")
    if args.min_pass < 0:
        raise ValueError("--min-pass: 0 or more")
    terms = [t for t in re.split(r"[,\s]+", args.score_terms or "") if t]
    table = None
    if args.scores:
        table = read_decoy_table(args.scores, args.plan_root)
        wanted = terms + ([args.rank_term] if args.order == "score" else [])
        missing = [t for t in wanted if t not in table.columns]
        if missing:
            stderr.write(f"WARNING: the decoy table has no column {', '.join(missing)}\n")
    elif args.order == "score" or st.score_limits:
        raise ValueError("--order score and --score-limits need a decoy table (--scores)")
    else:
        terms = []
    models, empty = [], []
    for path in args.models:
        slot = pdbtools.empty_slot(path)
        if slot:
            empty.append(path)
            stderr.write(f"note: skipped {path}: {slot} (an empty slot of an earlier filter, no model)\n")
        else:
            models.append(path)
    reports = [check_model(path, st, table, path=path) for path in models]
    passing = [r for r in ranked(reports, args.order, args.rank_term) if r.status == "pass"]
    filtering = args.on_fail == "filter"
    slots = (args.n_pass if args.n_pass is not None else len(passing)) if filtering else len(reports)
    filled = min(len(passing), slots) if filtering else len(reports)
    tally = (f"{len(passing)} of {len(reports)} models passed the gates"
             + (f" ({len(empty)} empty slot(s) among the inputs skipped)" if empty else "")
             + (f"; passed on {filled} in {slots} slot(s)" if filtering and args.n_pass is not None else "")
             + ("" if not filtering or slots <= filled else f", slot {slots} written empty"
                if slots == filled + 1 else f", slots {filled + 1}-{slots} written empty"))
    text_out = args.output or not (args.json or args.tsv)
    if args.json or args.tsv:
        text = (json.dumps([report_data(r, terms) for r in reports], indent=2) + "\n" if args.json
                else tsv_text(reports, terms))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            stdout.write(text)
    if text_out:
        if "links" in st.checks:
            bond = st.link_bond or DEFAULT_LINK_BOND
            stdout.write(f"O3'-P links: bonded up to {st.link_max:g} A, strained up to {st.link_fail:g} A "
                         f"(strain E = k (r - r0)^2, k {bond.k:g} kcal/mol/A^2, r0 {bond.r0:g} A: "
                         f"{bond.source}), broken beyond\n")
        for note in st.notes:
            stdout.write(f"note: {note}\n")
        for r in reports:
            stdout.write(text_report(r, st) + "\n")
        if len(reports) > 1:
            stdout.write(summary_text(reports) + "\n")
    (stdout if text_out else stderr).write(tally + "\n")
    order = ranked(reports, args.order, args.rank_term)
    unreadable = [r for r in reports if r.error]
    if unreadable:
        stderr.write(f"ERROR: {len(unreadable)} model(s) cannot be read: "
                     + ", ".join(os.path.basename(r.path) for r in unreadable[:10]) + "\n")
        return 1, reports
    if not reports and not (filtering and args.min_pass == 0):
        stderr.write("ERROR: no model to check" + (" (every input is an empty slot)" if empty else "") + "\n")
        return 1, reports
    if filtering:
        if len(passing) < args.min_pass:
            stderr.write(f"ERROR: {len(passing)} of {len(reports)} models pass the gates; at least "
                         f"{args.min_pass} needed\n")
            return 1, reports
        chosen = passing[:slots]
    elif args.on_fail == "stop":
        bad = [r for r in reports if r.status != "pass"]
        if bad:
            stderr.write(f"ERROR: {len(bad)} of {len(reports)} models fail the gates: "
                         + ", ".join(os.path.basename(r.path) for r in bad[:10]) + "\n")
            return 1, reports
        chosen = order
    else:
        chosen = order
        bad = [r for r in reports if r.status != "pass"]
        if bad:
            stderr.write(f"WARNING: {len(bad)} of {len(reports)} models fail the gates (reported only): "
                         + ", ".join(os.path.basename(r.path) for r in bad[:10]) + "\n")
    if args.passed_out:
        for n in range(1, (slots if filtering else len(chosen)) + 1):
            dst = args.passed_out.format(n=n)
            if n <= len(chosen):
                shutil.copyfile(chosen[n - 1].path, dst)
            else:  # fixed arity downstream, without pretending that N models passed
                with open(dst, "w", encoding="utf-8") as fh:
                    fh.write(pdbtools.empty_slot_text(n, slots, len(passing)))
    if args.best_out and chosen:
        shutil.copyfile(chosen[0].path, args.best_out)
    elif args.best_out and filtering:  # --min-pass 0 and none passed
        with open(args.best_out, "w", encoding="utf-8") as fh:
            fh.write(pdbtools.empty_slot_text(1, 1, 0))
    return 0, reports


def summary_text(reports: Sequence[ModelReport]) -> str:
    """One line per model: status, clashes, link problems and strain, worst compaction, hands, kinks."""
    lines = [f"{'model':40s} {'status':>6} {'clashes':>8} {'links':>6} {'strain':>7} {'compaction':>11} "
             f"{'hands':>6} {'kinks':>6}"]
    for r in reports:
        c = len(r.clashes.clashes) if r.clashes else 0
        b = len(r.backbone.problems) if r.backbone else 0
        strain = f"{r.backbone.strain:.0f}" if r.backbone and r.backbone.strain else "-"
        kinks = len(r.kinks.flagged) if r.kinks else 0
        lines.append(f"{os.path.basename(r.path)[-40:]:40s} {r.status:>6} {c:8d} {b:6d} {strain:>7} "
                     f"{_fmt(r.worst_compaction) or '-':>11} {len(r.hand_problems):6d} {kinks:6d}")
    lines.append("clashes: counted clashes; links: O3'-P links broken, short or unmeasurable; strain: "
                 "kcal/mol of the links beyond the limits; compaction: the most coiled loop; hands: "
                 "chirality problems; kinks: sharp kinks where none is allowed")
    return "\n".join(lines)


def qc_main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]) or "modelcheck",
                                     description="Plausibility checks of RNA models: clashes, O3'-P links, "
                                                 "loop compaction, sugar hands and kinks (and, reported "
                                                 "only, hydrogen contacts).")
    add_qc_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_qc(args)[0]
    except (OSError, ValueError, pdbtools.PDBError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
