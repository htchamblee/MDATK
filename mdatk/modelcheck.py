"""Plausibility checks of RNA models before they are simulated or refined further.

A decoy can be best-in-class on its score, its map fit and its base pairs and still be locally absurd: two
oxygens pressed together, a chain break that tleap will bond anyway, a loop threaded into a knot, a ribose
flipped to the other hand by real-space refinement.  Four checks, on one model or an ensemble:

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
    is coordination, not a clash).  Hydrogens are optional (Bondi H 1.2 Å; a hydrogen is left out up to
    four bonds away, as in MolProbity's probe, and the hydrogen of an H-bonded donor is excused with it).
    Hydrogen positions differ between programs: an ideal A-form helix with ideally placed hydrogens
    already has H2'(i)...H6/H8(i+1) at 1.6 Å, so heavy atoms are the default.
links
    Every O3'(i)-P(i+1) of consecutive nucleotides within each TER segment, as tleap bonds them: longer
    than ``link_max`` (2.0 Å) is a break, shorter than ``link_min`` (1.4 Å) an overlap, a missing O3' or P
    an unmeasurable link.  Consecutive P-P outside 5.0-7.5 Å are reported too.
compaction
    End-to-end distance over contour length along C1' for loop segments, given as residue ranges or found
    as the residues unpaired in every connected dot-bracket (runs of three or more).  Below 0.20 a loop is
    flagged for inspection; there is no verdict above: a loop is meant to turn, and what distinguishes a
    tight turn from a threaded one is not in this number.
hands
    The sugar hand of every nucleotide from its geometry (:func:`mdatk.pdbtools.residue_hands`: all usable
    centres of C4', C3' and C1' must agree; the C4' hand alone is reported as well), against the hands
    expected for residue ranges, with epimerised and flat centres named.

Residues are addressed by chain, number and insertion code throughout; ranges use
:func:`mdatk.pdbtools.parse_ranges` syntax ('A:7-12, B:25-34, C:58-63', 'C', ':5-9' for a blank chain,
'*:45-73' for every chain).  Models can be gated (``max_clashes``, links, hands, P-P spacing, limits on
Rosetta score terms from a decoy table) and an ensemble filtered or ranked.

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
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

try:
    from . import pdbtools
except ImportError:  # copied into a generated script after the PDB tools: they are this module
    pdbtools = sys.modules[__name__]

CHECKS = ("clashes", "links", "compaction", "hands")
ON_FAIL = ("report", "filter", "stop")
ORDERS = ("input", "clashes", "score")

CLASH_TOL = 0.4  # Å inside van der Waals contact
HBOND_WINDOW = (2.2, 3.2)  # Å, donor-acceptor distances that are a hydrogen bond, not a clash
LINK_RANGE = (1.4, 2.0)  # Å, O3'-P; ideal 1.60, past 2.0 a break that minimisation has to repair
PP_RANGE = (5.0, 7.5)  # Å, consecutive phosphorus atoms
COMPACTION_FLAG = 0.20  # end-to-end / contour below this: inspect the loop
MIN_LOOP = 3  # residues of a loop segment found from dot-brackets

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
    hydrogens: bool = False  # include hydrogens in the clash check
    waters: bool = False  # include waters in the clash check
    ions: str = "skip"  # "skip" or "check" (first-shell distances)
    protonated: str = ""  # ranges of protonated A+ (N1-H) / C+ (N3-H)
    link_min: float = LINK_RANGE[0]
    link_max: float = LINK_RANGE[1]
    pp_min: float = PP_RANGE[0]
    pp_max: float = PP_RANGE[1]
    segments: str = ""  # compaction segments (ranges)
    dotbrackets: List[Tuple[str, str]] = field(default_factory=list)  # (name, text): loops unpaired in all
    compaction_flag: float = COMPACTION_FLAG
    expect_l: str = ""
    expect_d: str = ""
    max_clashes: Optional[int] = None  # gate: more clashes fail (None: no limit)
    gate_links: bool = True  # gate: a link outside the limits or unmeasurable fails
    gate_hands: bool = True  # gate: a hand against the expectation, an epimer or a flat centre fails
    gate_pp: bool = False  # gate: P-P spacing outside the range fails
    score_limits: str = ""  # gate: limits on decoy-table terms, e.g. 'linear_chainbreak<=1, score<=-50'


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


def build_model(structure, settings: QCSettings) -> _Model:
    """The atom table and bond graph of ``structure`` for the clash check (first altloc of every atom)."""
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
            index: Dict[str, int] = {}
            hydrogens: List[int] = []
            for atom in res.atoms:
                name = _atom_name(atom.name)
                if name in index:
                    continue  # a later alternate location
                h = pdbtools.is_hydrogen(atom, res.name)
                if h and not settings.hydrogens:
                    continue
                el = "H" if h else pdbtools.element_of(atom, res.name)
                # halides keep plain van der Waals contacts
                role = "h" if h else ("ion" if kind == "ion" and el in ION_CONTACT else "heavy")
                donor = el in ("N", "O") and (donors is None or name in donors)
                i = m.add(name, el, atom.xyz, ri, role, donor)
                index[name] = i
                if h:
                    hydrogens.append(i)
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
            for hi in hydrogens:  # each hydrogen on its nearest heavy atom of the residue
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
    """Clashes of one model (see the module docstring for the chemistry)."""
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
    parent = {i: m.bonded[i][0] for i in range(len(m.xyz)) if m.role[i] == "h" and m.bonded[i]}
    is_no = [e in ("N", "O") and r == "heavy" for e, r in zip(m.el, m.role)]
    cache: Dict[int, Dict[int, int]] = {}
    candidates: List[Tuple[float, float, int, int, str]] = []
    partners: Dict[int, Dict[int, float]] = {}  # H-bonded heavy atoms (any distance in the window)
    for i, j, d in pairs_within(m.xyz, biggest + 1e-6):
        ri, rj = m.role[i], m.role[j]
        bonds = _bond_distances(m, i, cache).get(j)
        # atoms up to three bonds apart are never checked; a hydrogen up to four (as MolProbity's probe does:
        # its position follows from the torsion, e.g. H6...O4' of a base in anti)
        if bonds is not None and (bonds <= 3 or "h" in (ri, rj)):
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
        elif ri == "h" or rj == "h":
            h, x = (i, j) if ri == "h" else (j, i)
            p = parent.get(h)
            if p is not None and is_no[p] and is_no[x] and lo <= math.dist(m.xyz[p], m.xyz[x]) <= hi:
                continue  # the hydrogen of a hydrogen bond
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


# --- backbone -----------------------------------------------------------------------------------------------

class Link(NamedTuple):
    """A backbone link of consecutive nucleotides: residue labels, distance (None: unmeasurable) and status
    ('ok', 'long', 'short', 'missing O3'' / 'missing P', or 'p-p' for a P-P spacing outlier)."""
    first: str
    second: str
    distance: Optional[float]
    status: str


@dataclass
class BackboneResult:
    links: List[Link] = field(default_factory=list)  # every O3'-P link of consecutive nucleotides
    pp: List[Link] = field(default_factory=list)  # P-P spacings outside the range
    pp_measured: int = 0

    @property
    def problems(self) -> List[Link]:
        return [k for k in self.links if k.status != "ok"]

    @property
    def worst(self) -> Optional[Link]:
        measured = [k for k in self.links if k.distance is not None]
        return max(measured, key=lambda k: k.distance) if measured else None


def check_backbone(structure, link_min: float = LINK_RANGE[0], link_max: float = LINK_RANGE[1],
                   pp_min: float = PP_RANGE[0], pp_max: float = PP_RANGE[1]) -> BackboneResult:
    """O3'(i)-P(i+1) of consecutive nucleotides within every TER segment (what tleap bonds, gaps in the
    numbering included), and consecutive P-P spacings."""
    out = BackboneResult()
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
                    status = "long" if d > link_max else ("short" if d < link_min else "ok")
                    out.links.append(Link(a, b, round(d, 3), status))
                if "P" in prev_xyz and p is not None:
                    out.pp_measured += 1
                    d = math.dist(prev_xyz["P"], p)
                    if not pp_min <= d <= pp_max:
                        out.pp.append(Link(a, b, round(d, 3), "p-p"))
            prev, prev_xyz = res, xyz
    return out


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
    score_row: Optional[Dict[str, str]] = None
    failed: List[str] = field(default_factory=list)

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
        out.append(f"{len(r.backbone.problems)} O3'-P link(s) outside {st.link_min:g}-{st.link_max:g} A or "
                   "unmeasurable")
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
    if "links" in st.checks:
        r.backbone = check_backbone(s, st.link_min, st.link_max, st.pp_min, st.pp_max)
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


def report_header(score_terms: Sequence[str] = ()) -> List[str]:
    return (["model", "status", "failed", "residues", "atoms", "clashes", "worst_overlap", "short_hbonds",
             "hbond_contacts", "shadows", "links", "link_problems", "worst_link", "worst_link_at",
             "pp_outside", "compaction", "worst_compaction", "hands", "hand_problems"] + list(score_terms))


def report_row(r: ModelReport, score_terms: Sequence[str] = ()) -> List[str]:
    c, b = r.clashes, r.backbone
    worst = b.worst if b else None
    row = [r.path, r.status, "; ".join(r.failed), str(r.residues),
           str(c.atoms) if c else "", str(len(c.clashes)) if c else "", _fmt(c.worst) if c else "",
           str(c.short_hbonds) if c else "", str(len(c.hbond_contacts)) if c else "",
           str(len(c.shadows)) if c else "",
           str(len(b.links)) if b else "", str(len(b.problems)) if b else "",
           _fmt(worst.distance) if worst else "", f"{worst.first}-{worst.second}" if worst else "",
           str(len(b.pp)) if b else "",
           " ".join(f"{s.name}={_fmt(s.ratio)}" for s in r.compaction), _fmt(r.worst_compaction),
           hand_counts(r), str(len(r.hand_problems)) if r.hands else ""]
    for term in score_terms:
        row.append(r.score_row.get(term, "") if r.score_row else "")
    return [v.replace("\t", " ").replace("\n", " ") for v in row]


def tsv_text(reports: Sequence[ModelReport], score_terms: Sequence[str] = ()) -> str:
    lines = ["\t".join(report_header(score_terms))]
    lines += ["\t".join(report_row(r, score_terms)) for r in reports]
    return "\n".join(lines) + "\n"


def report_data(r: ModelReport, score_terms: Sequence[str] = ()) -> Dict[str, object]:
    """JSON-ready details of one model."""
    out: Dict[str, object] = {"model": r.path, "status": r.status, "failed": r.failed, "notes": r.notes,
                              "residues": r.residues, "nucleotides": r.nucleotides}
    if r.error:
        out["error"] = r.error
    if r.clashes is not None:
        out["clashes"] = {"atoms": r.clashes.atoms, "count": len(r.clashes.clashes),
                          "list": [c._asdict() for c in r.clashes.clashes],
                          "hbond_contacts": len(r.clashes.hbond_contacts), "shadows": len(r.clashes.shadows)}
    if r.backbone is not None:
        out["links"] = {"count": len(r.backbone.links),
                        "problems": [k._asdict() for k in r.backbone.problems],
                        "worst": r.backbone.worst._asdict() if r.backbone.worst else None,
                        "pp_outside": [k._asdict() for k in r.backbone.pp]}
    if r.compaction:
        out["compaction"] = [s._asdict() for s in r.compaction]
    if r.hands:
        out["hands"] = {"by_chain": hand_counts(r), "notes": r.hand_notes, "problems": r.hand_problems}
    if score_terms:
        out["scores"] = {t: (r.score_row or {}).get(t) for t in score_terms}
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
        out.append(f"  clashes (overlap >= {st.clash_tol:g} A inside van der Waals contact, {c.atoms} "
                   f"atoms): {len(c.clashes)}")
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
        out.append(f"  backbone: {len(b.links)} O3'-P links, {len(b.problems)} outside {st.link_min:g}-"
                   f"{st.link_max:g} A or unmeasurable"
                   + (f" (longest {worst.first}-{worst.second} {worst.distance:.2f} A)" if worst else "")
                   + f"; {len(b.pp)} of {b.pp_measured} P-P outside {st.pp_min:g}-{st.pp_max:g} A")
        for k in (b.problems + b.pp)[:limit]:
            what = f"{k.distance:.2f} A" if k.distance is not None else k.status
            out.append(f"      {k.first} -> {k.second}: {what} ({'P-P' if k.status == 'p-p' else k.status})")
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
    if r.score_row is not None:
        out.append("  decoy table: " + ", ".join(f"{k}={v}" for k, v in r.score_row.items()
                                                 if k not in ("run", "pdb") and v != "")[:300])
    out.append(f"  status: {r.status.upper()}" + (f" ({'; '.join(r.failed)})" if r.failed else ""))
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
    g.add_argument("--hydrogens", action="store_true", help="include hydrogens (default: heavy atoms)")
    g.add_argument("--waters", action="store_true", help="include waters")
    g.add_argument("--ions", choices=("skip", "check"), default="skip", help="ions: skip (default) or check "
                   "against their first-shell distances")
    g.add_argument("--protonated", default="", help="residues with protonated A+ (N1) / C+ (N3), e.g. "
                   "'B:30'")
    g = p.add_argument_group("backbone")
    for flag, value, what in (("--link-min", LINK_RANGE[0], "shortest O3'-P link"),
                              ("--link-max", LINK_RANGE[1], "longest O3'-P link"),
                              ("--pp-min", PP_RANGE[0], "shortest P-P spacing"),
                              ("--pp-max", PP_RANGE[1], "longest P-P spacing")):
        g.add_argument(flag, type=float, default=value, help=f"{what} (default {value})")
    g = p.add_argument_group("compaction")
    g.add_argument("--segments", default="", help="loop segments, e.g. 'A:7-12, B:25-34, C:58-63'")
    g.add_argument("--secstruct", action="append", default=[], help="secondary-structure file; residues "
                   "unpaired in every one given form the loop segments (repeatable)")
    g.add_argument("--compaction-flag", type=float, default=COMPACTION_FLAG, help="flag loops whose "
                   f"end-to-end / contour ratio is below this (default {COMPACTION_FLAG})")
    g = p.add_argument_group("chirality")
    g.add_argument("--expect-l", default="", help="residues whose sugars must be L, e.g. 'C:45-73'")
    g.add_argument("--expect-d", default="", help="residues whose sugars must be D, e.g. 'A,B'")
    g = p.add_argument_group("decoy table")
    g.add_argument("--scores", help="decoy table (TSV: run, tag, pdb, score terms ...)")
    g.add_argument("--plan-root", help="folder the table's pdb paths are relative to (default: the current "
                   "one)")
    g.add_argument("--score-terms", default="score", help="score terms shown in the report (default: score)")
    g = p.add_argument_group("gates")
    g.add_argument("--max-clashes", type=int, help="fail models with more clashes")
    g.add_argument("--no-gate-links", action="store_true", help="links outside the limits do not fail a "
                   "model")
    g.add_argument("--no-gate-hands", action="store_true", help="unexpected hands, epimers and flat centres "
                   "do not fail a model")
    g.add_argument("--gate-pp", action="store_true", help="P-P spacings outside the range fail a model")
    g.add_argument("--score-limits", default="", help="limits on decoy-table terms, e.g. "
                   "'linear_chainbreak<=1, score<=-50'")
    g.add_argument("--on-fail", choices=ON_FAIL, default="report", help="report: only report (default); "
                   "filter: pass on the models that pass the gates (--pass N of them, or fail); stop: exit "
                   "with status 1 when any model fails")
    g.add_argument("--pass", dest="n_pass", type=int, metavar="N", help="with --on-fail filter: the number "
                   "of passing models to pass on (default: all that pass)")
    g.add_argument("--order", choices=ORDERS, default="input", help="order of the passed models: input "
                   "(default), clashes (fewest first) or score (lowest decoy-table --rank-term first)")
    g.add_argument("--rank-term", default="score", help="decoy-table term for --order score (default: "
                   "score)")
    g.add_argument("--passed-out", help="copy the passed models to these files: a pattern with {n} (1, 2 "
                   "...), e.g. 'passed-{n:02d}.pdb'")
    g.add_argument("--best-out", help="copy the first passed model to this file")


def settings_from_args(args) -> QCSettings:
    checks = tuple(c.strip() for c in args.checks.split(",") if c.strip())
    unknown = [c for c in checks if c not in CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s) {', '.join(unknown)} (choose from {', '.join(CHECKS)})")
    dotbrackets = []
    for path in args.secstruct:
        with open(path, encoding="utf-8", errors="replace") as fh:
            dotbrackets.append((os.path.basename(path), fh.read()))
    st = QCSettings(checks=checks, clash_tol=args.clash_tol, hbond_min=args.hbond_min,
                    hbond_max=args.hbond_max, hydrogens=args.hydrogens, waters=args.waters, ions=args.ions,
                    protonated=args.protonated, link_min=args.link_min, link_max=args.link_max,
                    pp_min=args.pp_min, pp_max=args.pp_max, segments=args.segments, dotbrackets=dotbrackets,
                    compaction_flag=args.compaction_flag, expect_l=args.expect_l, expect_d=args.expect_d,
                    max_clashes=args.max_clashes, gate_links=not args.no_gate_links,
                    gate_hands=not args.no_gate_hands, gate_pp=args.gate_pp, score_limits=args.score_limits)
    validate_settings(st)
    return st


def validate_settings(st: QCSettings) -> None:
    """ValueError for impossible settings (empty windows, unreadable ranges or limits)."""
    for lo, hi, what in ((st.hbond_min, st.hbond_max, "H-bond window"),
                         (st.link_min, st.link_max, "O3'-P limits"), (st.pp_min, st.pp_max, "P-P range")):
        if not lo < hi:
            raise ValueError(f"{what}: the minimum ({lo:g}) must be below the maximum ({hi:g})")
    for text, what in ((st.segments, "segments"), (st.expect_l, "expected L"), (st.expect_d, "expected D"),
                       (st.protonated, "protonated residues")):
        try:
            pdbtools.parse_ranges(text or None)
        except pdbtools.PDBError as exc:
            raise ValueError(f"{what}: {exc}") from None
    parse_score_limits(st.score_limits)


def run_qc(args, stdout=None, stderr=None) -> Tuple[int, List[ModelReport]]:
    """Execute a parsed 'qc' command; returns (exit status, reports).  The report is written first.  Status 1
    (and nothing copied): a model fails with --on-fail stop, fewer models than needed pass with filter, or,
    whatever the policy, a model cannot be read."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    st = settings_from_args(args)
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
    reports = [check_model(path, st, table, path=path) for path in args.models]
    if args.json or args.tsv:
        text = (json.dumps([report_data(r, terms) for r in reports], indent=2) + "\n" if args.json
                else tsv_text(reports, terms))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            stdout.write(text)
    if args.output or not (args.json or args.tsv):
        for r in reports:
            stdout.write(text_report(r, st) + "\n")
        if len(reports) > 1:
            stdout.write(summary_text(reports) + "\n")
    order = ranked(reports, args.order, args.rank_term)
    unreadable = [r for r in reports if r.error]
    if unreadable:
        stderr.write(f"ERROR: {len(unreadable)} model(s) cannot be read: "
                     + ", ".join(os.path.basename(r.path) for r in unreadable[:10]) + "\n")
        return 1, reports
    if args.on_fail == "filter":
        passing = [r for r in order if r.status == "pass"]
        chosen = passing if args.n_pass is None else passing[:args.n_pass]
        want = 1 if args.n_pass is None else args.n_pass
        if len(chosen) < want:
            stderr.write(f"ERROR: {len(passing)} of {len(reports)} models pass the gates; {want} are "
                         "needed\n")
            return 1, reports
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
        for n, r in enumerate(chosen, 1):
            shutil.copyfile(r.path, args.passed_out.format(n=n))
    if args.best_out and chosen:
        shutil.copyfile(chosen[0].path, args.best_out)
    return 0, reports


def summary_text(reports: Sequence[ModelReport]) -> str:
    """One line per model: status, clashes, link problems, worst compaction, hand problems."""
    lines = [f"{'model':40s} {'status':>6} {'clashes':>8} {'links':>6} {'compaction':>11} {'hands':>6}"]
    for r in reports:
        c = len(r.clashes.clashes) if r.clashes else 0
        b = len(r.backbone.problems) if r.backbone else 0
        lines.append(f"{os.path.basename(r.path)[-40:]:40s} {r.status:>6} {c:8d} {b:6d} "
                     f"{_fmt(r.worst_compaction) or '-':>11} {len(r.hand_problems):6d}")
    lines.append("clashes: counted clashes; links: O3'-P links outside the limits; compaction: the most "
                 "coiled loop; hands: chirality problems")
    return "\n".join(lines)


def qc_main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]) or "modelcheck",
                                     description="Plausibility checks of RNA models: clashes, O3'-P links, "
                                                 "loop compaction and sugar hands.")
    add_qc_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_qc(args)[0]
    except (OSError, ValueError, pdbtools.PDBError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
