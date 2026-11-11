"""PDB hygiene for D- and L-RNA structures.

Line-based, fixed-column PDB handling that rewrites only what an operation has to change: coordinates,
occupancies, B-factors and segment ids are carried over verbatim (columns 28-76).

* Parsing repairs ChimeraX-damaged lines automatically (residue name fused with the chain id and a colon,
  ``"GC:"`` = residue ``G`` of chain ``C``) and keeps the first model only.
* Writing produces correct columns (atom-name alignment, element column), renumbered atom serials, a TER
  after every segment, END, optional SEQRES records, CONECT stripping and a record policy:
  ``keep``; ``rosetta`` (every polymer residue as ATOM - Rosetta ignores HETATM residues); ``pdb``
  (non-standard residue names such as ``0A`` as HETATM).
* L-nucleotides are detected from signed chiral volumes at C1', C3' and C4' (every usable centre must agree); residue
  ranges such as ``"C:45-73"`` override the detection and force L.
* Residue name schemes (base from the name, chirality from detection/override, terminal state from the
  segment ends that are free: a segment end that a LINK record bonds to another residue, such as the O3' / O5'
  of a chimera's 3'-3' / 5'-5' phosphate, keeps the internal name; DNA and unrecognised residues pass through
  unchanged unless DNA is asked for, see below):

  ``amber``    D: A C G U, terminals A5/A3, single-residue segment AN.  L (Dudek & Trylska,
               nucleic12_LRNA.lib): LA LC LG LU, terminals LA5/LA3.  The library has no LAN-type unit, so a
               single-residue L segment is written as LA5 (5'-OH; its O3' has no hydrogen) with a note.
  ``ccd``      for Coot/Phenix real-space refinement: D: A C G U; L: 0A 0C 0G 0U with the user's
               terminal variants 0A5/0A3 ... (these variants are not CCD entries and need their own
               restraint dictionaries); a single-residue L segment keeps the plain name.
  ``rosetta``  D: A C G U; L: 0A 0C 0G 0U; no terminal variants.
  ``pdb``      deposition naming, as ``rosetta``.
  ``standard`` plain A C G U for both hands, no terminal variants, every record ATOM: a copy for programs
               that read standard RNA only and ignore chirality (map-model scores, DSSR).  The names no
               longer record the hand, so such a copy is never meant for tleap, refinement or analyses
               that read names.

  DNA is renamed only on request (``convert_names(..., dna=True)``, as the structure builder does): D: DA DC
  DG DT (amber: DA5/DA3/DAN of DNA.OL15); L: the CCD's 0DA 0DC 0DG 0DT, amber: MDATK's LDA LDC LDG LDT (3 characters, ends too)
  ... (AmberTools ships no L-DNA library); standard: DA for both hands.  Both L-DNA spellings are read back.
  0DA and 0DT carry their CCD components' own atom names (0DA: N9A, C8A ... HN61, HN62; 0DT: C5M,
  ``CCD_ATOM_NAMES``): :func:`read_pdb` reads them as the standard names, so that every reader of a parsed
  structure sees N9, C7 ..., and :func:`write_pdb` writes them for every residue named 0DA / 0DT (and the
  standard names for every other name, LDA and DA included).

* Chains: restore ids of cpptraj output (from a reference PDB, a list per TER segment or A, B, C ...),
  split, combine and order; residue renumbering; SEQRES records and FASTA text (the polymer residues: a
  residue made only of phosphate atoms, such as a chimera's PO2 bridge or a lone 5'-phosphate, is left out;
  SEQRES names have three characters, MDATK's four-character L-DNA names are written as their CCD components).
* LINK records (bonds the polymer order does not imply, such as the 3'-3' / 5'-5' phosphate of a chimera)
  follow the residues they name: the operations that rename, renumber, re-chain or remove residues rewrite
  their residue columns (:func:`follow_links`); a record whose residue or atom is removed is dropped, one that
  names no atom of the model stays as it was, each with a note.  A record whose two symmetry operators differ
  (columns 60-65 / 67-72: one atom in a copy of the model made by the crystal symmetry) follows the names too,
  but is no bond of the model (:attr:`LinkRecord.mate`).
* Hydrogens and Rosetta's virtual residues (VRT, XXX) can be removed; :func:`infer_base` tells the base
  from the atom names when a residue name is not recognised.
* Per-residue sugar hands (:func:`residue_hands`): the three-centre call, the C4' hand alone, epimers and
  flat centres, and expected hands per residue range (``check --expect-l / --expect-d --strict --table``).
* Empty slots (:func:`empty_slot_text`, :func:`empty_slot`, :func:`is_empty_slot`): the files a model check
  writes for the slots no model fills (REMARK 951, END, no atoms); :func:`read_pdb` names one.

Every operation works in place on a :class:`Structure` and returns human-readable notes; :func:`clean`
chains them for the CLI / GUI / planner.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import math
import os
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

Vec = Tuple[float, float, float]


class PDBError(ValueError):
    """Unreadable input or an impossible request (e.g. a chain-restore mismatch)."""


CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits
SCHEMES = ("amber", "ccd", "rosetta", "pdb", "standard")
RECORD_POLICIES = ("keep", "rosetta", "pdb", "atom")
# Rosetta's virtual residues: the root of a density-scored fold tree (VRT, atoms ORIG / X / Y) and XXX.  They
# are no atoms of the molecule: tleap has no unit for them, and a map-model score would read them as atoms.
VIRTUAL_RESIDUES = frozenset({"VRT", "XXX"})
# An empty slot: a model check writes the slots it cannot fill (a filter's 'up to N', fewer models than
# planned) as a file with this remark and END, no atoms, so a step with a fixed number of files downstream
# can tell which are empty instead of a count that pretends N models passed (MDATK's readers of model sets
# skip them, read_pdb refuses one by name).
EMPTY_SLOT_REMARK = "REMARK 951"
_EMPTY_SLOT_RE = re.compile(r"^REMARK 951 +(empty slot .*)$", re.IGNORECASE)

# Signs of the chiral volumes of an L-ribose: (centre, (atom 1, atom 2, atom 3), sign), from the
# _chem_comp_chir tables of the 0A/0C/0G/0U restraint dictionaries; "N" is N9 (purines) or N1
# (pyrimidines).  V = a.(b x c) with a, b, c the vectors from the centre to atoms 1-3; D has the
# opposite signs.
MIN_CHIRAL_VOLUME = 0.5  # Å^3; real sugar centres give about 1.8-2.6, schematic models can be near 0
L_CHIRAL_SIGNS = (("C4'", ("O4'", "C3'", "C5'"), 1.0),
                  ("C3'", ("O3'", "C4'", "C2'"), -1.0),
                  ("C1'", ("O4'", "N", "C2'"), 1.0))

_AMINO = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
          "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
          "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# Amber protonation / disulfide variants: one-letter codes only, they are not standard PDB names
_AMINO_VARIANTS = {"HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E",
                   "LYN": "K"}
_STANDARD_NAMES = frozenset(_AMINO) | {"A", "C", "G", "U", "I", "N", "DA", "DC", "DG", "DT", "DI", "DN",
                                       "UNK"}
_WATER = frozenset({"HOH", "WAT", "H2O", "DOD", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "T3P", "T4P", "T4E",
                    "SPC", "OPC"})
_NT_RE = re.compile(r"^([RL0D]?)([ACGTU])([53N]?)$")
# L-DNA: the CCD's 2'-deoxy-L-ribonucleotides 0DA 0DC 0DG 0DT (no terminal variants) and the Amber-style names
# MDATK's builder writes, LDA ... (after nucleic12_LRNA.lib's LA; AmberTools has no L-DNA units); older files
# have LDA5 / LDA3 / LDAN at the ends
_LDNA_RE = re.compile(r"^(?:0D([ACGT])|LD([ACGT])([53N]?))$")
_NT_ALIASES = {"ADE": ("A", "rna"), "CYT": ("C", "rna"), "GUA": ("G", "rna"), "URA": ("U", "rna"),
               "URI": ("U", "rna"), "THY": ("T", "dna")}
_TWO_LETTER_ELEMENTS = frozenset({"MG", "NA", "CL", "ZN", "MN", "CA", "LI", "RB", "CS", "SR", "BA", "CD",
                                  "FE", "CO", "NI", "CU", "BR", "AG", "HG", "PB", "TL", "AU", "PT"})
_SEQRES_ORDER = ("MODRES", "HET", "HETNAM", "HETSYN", "FORMUL", "HELIX", "SHEET", "SSBOND", "LINK",
                 "CISPEP", "SITE", "CRYST1", "ORIGX", "SCALE", "MTRIX")
_COORD_RE = re.compile(r"-?\d+\.\d{3}")
_FLOAT_RE = re.compile(r"-?\d+\.\d+")


# --- data model ---------------------------------------------------------------------------------------

@dataclass
class Atom:
    """One ATOM/HETATM record; ``tail`` is columns 28-76 verbatim (coordinates, occupancy, B, segid)."""
    record: str
    name: str
    alt: str
    tail: str
    element: str = ""
    charge: str = ""
    serial: Optional[int] = None  # input serial, used to remap kept CONECT records
    anisou: str = ""  # columns 28-80 of a following ANISOU record

    @property
    def xyz(self) -> Vec:
        t = self.tail
        return (float(t[3:11]), float(t[11:19]), float(t[19:27]))


@dataclass
class Residue:
    name: str
    chain: str
    resseq: int
    icode: str = ""
    atoms: List[Atom] = field(default_factory=list)
    chirality: Optional[str] = None  # "D" / "L" once assign_chirality ran (override, detection or name)

    @property
    def label(self) -> str:
        return f"{self.chain or '_'}:{self.resseq}{self.icode} {self.name}"

    def coords(self) -> Dict[str, Vec]:
        """Atom name -> coordinates (first alternate location; old-style '*' read as a prime; the atoms of a CCD
        component with names of its own, such as N9A of 0DA, by their standard names: :func:`standard_atom_name`)."""
        own = _CCD_STANDARD_NAMES.get(self.name.strip().upper(), {})
        out: Dict[str, Vec] = {}
        for a in self.atoms:
            name = a.name.replace("*", "'")
            out.setdefault(own.get(name, name), a.xyz)
        return out


@dataclass
class Structure:
    """A parsed PDB file: header lines, TER-delimited segments of residues, trailing lines."""
    header: List[str] = field(default_factory=list)
    segments: List[List[Residue]] = field(default_factory=list)
    trailer: List[str] = field(default_factory=list)
    conect: List[str] = field(default_factory=list)
    model: str = ""  # MODEL line of the kept model ("" when the input had none)
    notes: List[str] = field(default_factory=list)  # parse notes

    def residues(self) -> Iterator[Residue]:
        for seg in self.segments:
            yield from seg

    def chain_ids(self) -> List[str]:
        """Chain ids in order of first appearance."""
        return list(dict.fromkeys(r.chain for r in self.residues()))


# --- residue names ------------------------------------------------------------------------------------

class ResidueName(NamedTuple):
    base: str  # A C G U (T for DNA)
    sugar: str  # "rna" / "dna"
    chirality: str  # "D" / "L" as implied by the name
    terminal: str  # "5", "3", "N" (single residue) or ""


def parse_residue_name(name: str) -> Optional[ResidueName]:
    """Recognise a nucleotide name in any supported scheme or common alias; None otherwise.

    RNA: A, A5, A3, AN, RA, RA5, RA3, RAN, ADE/CYT/GUA/URA/URI; L-RNA: LA, LA5, LA3, 0A, 0A5, 0A3 (and the
    N forms); DNA: DA, DA5, DA3, DAN, T, THY ...; L-DNA: the CCD components 0DA, 0DC, 0DG, 0DT and the
    Amber-style LDA ... of MDATK's builder (LDA5, LDA3, LDAN in files written before 0.0.17).

    A name alone can mean something else: in the CCD, LA, LU and RA are the lanthanum, lutetium and radium ions,
    LDA a detergent (lauryldimethylamine N-oxide), LDT a ligand without a sugar, LDC an L-2'-deoxycytidine
    nucleoside and LDG a D-deoxyguanosine adduct.  The readers that see the atoms (:func:`nucleotide`, the model
    check, the base-pair engines, the interfaces, the sequence panel) take a name for a nucleotide only when the
    residue has backbone atoms (a primed atom or P), which tells the ions, LDA and LDT apart; LDC and LDG have
    a sugar, so their hand is the one the geometry gives (:func:`assign_chirality` measures it).
    """
    n = name.strip().upper()
    if n in _NT_ALIASES:
        base, sugar = _NT_ALIASES[n]
        return ResidueName(base, sugar, "D", "")
    m = _NT_RE.match(n)
    if not m:
        m = _LDNA_RE.match(n)
        if not m:
            return None
        ccd, amber, term = m.groups()
        return ResidueName(ccd or amber, "dna", "L", term or "")
    pre, base, term = m.groups()
    if pre == "D" or (base == "T" and pre == ""):
        return ResidueName(base, "dna", "D", term)
    if base == "T":
        return None
    return ResidueName(base, "rna", "L" if pre in ("L", "0") else "D", term)


def scheme_name(base: str, chirality: str, terminal: str, scheme: str, sugar: str = "rna") -> str:
    """Residue name of an RNA nucleotide in ``scheme`` (terminal: "5", "3", "N" or "").

    ``sugar`` "dna": the name of a DNA nucleotide.  amber: DA DA5 DA3 DAN (the units of AmberTools'
    DNA.OL15.lib), L: LDA at every position (MDATK's own names, after nucleic12_LRNA.lib's LA: AmberTools ships
    no L-DNA library). A PDB residue name has three characters, and tleap reads no more: the Amber build maps an
    L-DNA residue at a chain end to its terminal unit (LDA5 / LDA3 of the generated library), as leaprc.DNA maps
    DA to DA5 / DA3; ccd / rosetta / pdb: DA, L: 0DA (the CCD has no terminal variants for DNA); standard: DA
    for both hands."""
    if sugar == "dna":
        if scheme == "amber":
            return "LD" + base if chirality == "L" else "D" + base + terminal
        if scheme in ("ccd", "rosetta", "pdb"):
            return ("0D" if chirality == "L" else "D") + base
        if scheme == "standard":  # as for RNA: the hand is not in the name
            return "D" + base
        raise PDBError(f"unknown residue name scheme {scheme!r} (choose from {', '.join(SCHEMES)})")
    if scheme == "amber":
        if chirality == "L":  # nucleic12_LRNA.lib defines LX, LX5 and LX3 only
            return "L" + base + {"5": "5", "3": "3", "N": "5"}.get(terminal, "")
        return base + terminal
    if scheme == "ccd":
        return "0" + base + {"5": "5", "3": "3"}.get(terminal, "") if chirality == "L" else base
    if scheme in ("rosetta", "pdb"):
        return "0" + base if chirality == "L" else base
    if scheme == "standard":  # copies for chirality-blind programs: the hand is not in the name
        return base
    raise PDBError(f"unknown residue name scheme {scheme!r} (choose from {', '.join(SCHEMES)})")


# Atom names of the CCD's L-deoxyribonucleotide components that differ from the standard (PDB v3) names of the
# same atoms: 0DA keeps old names for its adenine (N9A ..., the amino hydrogens HN61 / HN62), 0DT calls its
# methyl carbon C5M (wwPDB CCD, checked in the CCP4 monomer library and in ParmEd's CCD templates).  0DC and 0DG
# use the standard names.  A residue written under one of these names carries the component's atom names, so
# that Coot / Phenix find every atom in the dictionary; MDATK reads them back as the standard names.  Both happen
# at the file: read_pdb gives the atoms of a 0DA / 0DT their standard names and write_pdb writes the component's
# names for every residue named 0DA / 0DT, so a structure in memory always has the standard names (whatever an
# operation renames a residue to) and every reader of one finds N9, C7 ... where it looks for them.
CCD_ATOM_NAMES: Dict[str, Dict[str, str]] = {
    "0DA": {"N9": "N9A", "C8": "C8A", "N7": "N7A", "C5": "C5A", "C6": "C6A", "N6": "N6A", "N1": "N1A",
            "C2": "C2A", "N3": "N3A", "C4": "C4A", "H8": "H8A", "H2": "H2A", "H61": "HN61", "H62": "HN62",
            "HOP2": "H2P"},  # (the hydrogen of a protonated OP2: the CCD's 0DA calls it H2P)
    "0DT": {"C7": "C5M"},
}
_CCD_STANDARD_NAMES = {res: {v: k for k, v in names.items()} for res, names in CCD_ATOM_NAMES.items()}


def component_atom_name(resname: str, name: str) -> str:
    """The name of standard atom ``name`` in the residue ``resname`` (a CCD component's own spelling, e.g. N9A in
    0DA; every other residue: ``name`` unchanged)."""
    return CCD_ATOM_NAMES.get(resname.strip().upper(), {}).get(name, name)


def standard_atom_name(resname: str, name: str) -> str:
    """The standard (PDB v3) name of atom ``name`` of residue ``resname``: the inverse of
    :func:`component_atom_name` (N9A of 0DA -> N9, C5M of 0DT -> C7; every other atom unchanged)."""
    return _CCD_STANDARD_NAMES.get(resname.strip().upper(), {}).get(name, name)


def _read_component_names(structure: "Structure") -> List[str]:
    """Give the atoms of every 0DA / 0DT of a structure just read their standard names (in place); returns the
    labels of those that came with standard names where the component has its own (they are written with the
    component's)."""
    out = []
    for res in structure.residues():
        key = res.name.strip().upper()
        own = CCD_ATOM_NAMES.get(key)
        if not own:
            continue
        if any(a.name in own for a in res.atoms):
            out.append(res.label)
        back = _CCD_STANDARD_NAMES[key]
        for a in res.atoms:
            a.name = back.get(a.name, a.name)
    return out


def has_backbone_atoms(atom_names) -> bool:
    """Whether a residue with these atom names has a sugar-phosphate atom (a primed name, old-style '*' or P): the
    guard that a name is taken for a nucleotide only with (a lanthanum ion 'LA', the detergent 'LDA' are none)."""
    return any("'" in n or "*" in n or n.strip() == "P" for n in map(str, atom_names))


def _nucleotide(res: Residue) -> Optional[ResidueName]:
    """Name-based recognition, guarded by backbone atoms so that e.g. a lanthanum ion 'LA' is not RNA."""
    info = parse_residue_name(res.name)
    if info is None or not has_backbone_atoms(a.name for a in res.atoms):
        return None
    return info


def _is_polymer(res: Residue) -> bool:
    return _nucleotide(res) is not None or res.name.upper() in _AMINO or res.name.upper() in _AMINO_VARIANTS


def _is_solvent(res: Residue) -> bool:
    """Water or a single-atom residue (ion)."""
    return not _is_polymer(res) and (res.name.upper() in _WATER or len(res.atoms) == 1)


def _code(res: Residue) -> str:
    """One-letter code: nucleotide base, amino acid, else 'X'."""
    nt = _nucleotide(res)
    if nt:
        return nt.base
    return _AMINO.get(res.name.upper()) or _AMINO_VARIANTS.get(res.name.upper()) or "X"


def _chirality(res: Residue) -> Optional[str]:
    nt = _nucleotide(res)
    return res.chirality or (nt.chirality if nt else None)


def infer_base(atom_names, resname: str = "") -> Optional[str]:
    """Base letter (A, C, G, U or T) from a residue's atom names; None when they do not decide it.

    For residue names :func:`parse_residue_name` does not know (DSSR ids, Rosetta's numbered or blank names,
    a mutated residue that kept its old name): purines have N9 (or N7 and C8), A carries N6, G O6 or N2;
    pyrimidines have N1 and C2 but no purine atoms, C carries N4, U O4, T O4 and its methyl carbon (C7 or
    C5M).  Contradicting substituents (N6 with O6 / N2, N4 with O4) give None.  ``resname``: the residue's
    name, whose CCD component may spell its atoms its own way (N9A ... of 0DA: :func:`standard_atom_name`).
    """
    names = {standard_atom_name(resname, str(n).strip().replace("*", "'")).upper() for n in atom_names}
    if "N9" in names or {"N7", "C8"} <= names:
        a, g = "N6" in names, bool({"O6", "N2"} & names)
        return ("A" if a else "G") if a != g else None
    if {"N1", "C2"} <= names:
        c, u = "N4" in names, "O4" in names
        if c != u:
            return "C" if c else ("T" if {"C7", "C5M"} & names else "U")
    return None


def nucleotide(res: Residue, infer: bool = False) -> Optional[ResidueName]:
    """The residue as a nucleotide (base, sugar, chirality the name implies, terminal state), or None.

    By name, in any scheme or alias, guarded by backbone atoms (a lanthanum ion 'LA' is no RNA).  With
    ``infer``, a residue whose name is not recognised but that has a sugar (C1', C4', O4') and decisive base
    atoms (:func:`infer_base`) is recognised from its atoms: RNA when it has O2', chirality 'D' (the name says
    nothing; measure the hand with :func:`sugar_chirality`), no terminal state."""
    info = _nucleotide(res)
    if info is not None or not infer:
        return info
    names = {a.name.replace("*", "'") for a in res.atoms}
    base = infer_base(names, res.name) if {"C1'", "C4'", "O4'"} <= names else None
    if base is None:
        return None
    return ResidueName(base, "rna" if "O2'" in names else "dna", "D", "")


def element_of(atom: Atom, resname: str = "") -> str:
    """Element symbol in upper case: the element column, else from the atom name (an ion residue named after
    its element, e.g. MG or NA, gets the two-letter symbol; otherwise the first letter after any leading
    digits, so 1H5' is a hydrogen)."""
    return _element(atom, resname).upper()


def is_hydrogen(atom: Atom, resname: str = "") -> bool:
    """Hydrogen or deuterium: the element column decides when it is filled, else the name (H after any
    leading digits: H5'', 1H5', HO2')."""
    if atom.element.strip():
        return atom.element.strip().upper() in ("H", "D")
    return element_of(atom, resname) == "H"


# --- parsing ------------------------------------------------------------------------------------------

def _read_text(source) -> str:
    if isinstance(source, os.PathLike) or (isinstance(source, str) and "\n" not in source
                                           and os.path.isfile(source)):
        with open(source) as fh:
            return fh.read()
    if not isinstance(source, str):
        raise PDBError(f"expected PDB text or a path, got {type(source).__name__}")
    return source


def empty_slot_text(slot: int, slots: int, passed: int, why: str = "") -> str:
    """The file of an unfilled slot: 'empty slot k of N: only M models passed' (or ``why`` it is empty, e.g.
    'only 5 models were given') and END, no atoms."""
    why = why or f"only {passed} models passed"
    return f"{EMPTY_SLOT_REMARK} empty slot {slot} of {slots}: {why}\nEND\n"


def empty_slot(source) -> str:
    """The note of an empty-slot file ('empty slot 4 of 5: only 3 models passed'), or '' for anything else
    (a model with atoms, a missing or unreadable file).  Reads only up to the first atom record."""
    note = ""
    try:
        if isinstance(source, str) and "\n" in source:
            lines: Iterator[str] = iter(source.splitlines())
            fh = None
        else:
            fh = open(source, encoding="utf-8", errors="replace")
            lines = iter(fh)
        try:
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    return ""
                m = _EMPTY_SLOT_RE.match(line.rstrip())
                if m and not note:
                    note = m.group(1).strip()
        finally:
            if fh is not None:
                fh.close()
    except (OSError, TypeError):
        return ""
    return note


def is_empty_slot(source) -> bool:
    """Whether ``source`` (a path or PDB text) is an empty slot written by a model check
    (:func:`empty_slot`)."""
    return bool(empty_slot(source))


def _fused_field(line: str) -> Optional[Tuple[str, str, str, int]]:
    """ChimeraX 'resname+chain:' field -> (alt, resname, chain, colon index), or None for a normal line."""
    c = line.find(":", 18, 30)
    if c < 0 or line[c - 1].isspace():
        return None
    words = line[16:c - 1].split()  # [resname] or [altloc, resname]
    if len(words) == 1:
        return " ", words[0], line[c - 1], c
    if len(words) == 2 and words[0] == line[16]:
        return words[0], words[1], line[c - 1], c
    return None


def _tokens_tail(rest: str, lineno: int) -> Tuple[int, str, str, str]:
    """Fallback for shifted columns: resseq(+icode), x, y, z, occupancy, B, element from the tokens."""
    m = re.match(r"\s*(-?\d+)([A-Za-z]?)\s", rest)
    coords = list(_COORD_RE.finditer(rest, m.end())) if m else []
    if len(coords) < 3:
        raise PDBError(f"line {lineno}: cannot read the damaged ATOM record")
    x, y, z = (float(c.group()) for c in coords[:3])
    after = rest[coords[2].end():]
    nums = [float(v) for v in _FLOAT_RE.findall(after)[:2]]
    occ, bfac = (nums + [1.0, 0.0][len(nums):])[:2]
    words = after.split()
    element = words[-1] if words and words[-1].isalpha() and len(words[-1]) <= 2 else ""
    tail = f"   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}".ljust(49)
    return int(m.group(1)), m.group(2), tail, element


def _coords_ok(tail: str) -> bool:
    try:
        float(tail[3:11]), float(tail[11:19]), float(tail[19:27])
    except ValueError:
        return False
    return True


def _parse_atom(line: str, lineno: int) -> Tuple[Atom, str, str, int, str, bool]:
    line = line.ljust(80)
    fused = _fused_field(line)
    element = charge = ""
    if fused:
        alt, resname, chain, colon = fused
        rest = line[colon + 1:].ljust(58)  # aligned like columns 23-80 when the shift is uniform
        try:
            resseq, icode, tail = int(rest[0:4]), rest[4].strip(), rest[5:54]
            element, charge = rest[54:56].strip(), rest[56:58].strip()
            if not _coords_ok(tail):
                raise ValueError
        except ValueError:
            resseq, icode, tail, element = _tokens_tail(rest, lineno)
            charge = ""
    else:
        alt, resname, chain = line[16], line[17:21].strip(), line[21]
        try:
            resseq = int(line[22:26])
        except ValueError:
            raise PDBError(f"line {lineno}: unreadable residue number {line[22:26]!r}") from None
        icode, tail = line[26].strip(), line[27:76]
        element, charge = line[76:78].strip(), line[78:80].strip()
        if not _coords_ok(tail):
            raise PDBError(f"line {lineno}: unreadable coordinates")
    try:
        serial: Optional[int] = int(line[6:11])
    except ValueError:
        serial = None
    atom = Atom(record=line[:6].strip(), name=line[12:16].strip(), alt=alt.strip(), tail=tail,
                element=element, charge=charge, serial=serial)
    return atom, resname, chain.strip(), resseq, icode, bool(fused)


def read_pdb(source) -> Structure:
    """Parse PDB text or a path (ChimeraX damage repaired, first model only); notes in ``.notes``."""
    if isinstance(source, Structure):
        return copy.deepcopy(source)
    text = _read_text(source)
    s = Structure()
    seg: List[Residue] = []
    res: Optional[Residue] = None
    started = ended = skipping = False
    n_models = repaired = dropped = after_end = 0

    def close_segment():
        nonlocal seg, res
        if seg:
            s.segments.append(seg)
        seg, res = [], None

    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.rstrip()
        rec = line[:6].strip().upper()
        if not line.strip():
            continue
        if ended:
            after_end += rec in ("ATOM", "HETATM")
            continue
        if rec == "MODEL":
            n_models += 1
            if n_models == 1:
                s.model = line
            else:
                skipping = True
            continue
        if rec == "ENDMDL":
            close_segment()
            skipping = True
            continue
        if rec in ("ATOM", "HETATM", "TER", "ANISOU", "SIGATM", "SIGUIJ") and skipping:
            continue
        if rec in ("ATOM", "HETATM"):
            started = True
            atom, resname, chain, resseq, icode, fixed = _parse_atom(line, lineno)
            repaired += fixed
            if seg and seg[-1].chain != chain:
                close_segment()
            if res is None or (res.name, res.chain, res.resseq, res.icode) != (resname, chain, resseq, icode):
                res = Residue(resname, chain, resseq, icode)
                seg.append(res)
            res.atoms.append(atom)
        elif rec == "TER":
            close_segment()
        elif rec == "ANISOU" and res is not None:
            res.atoms[-1].anisou = line.ljust(80)[27:80]
        elif rec in ("SIGATM", "SIGUIJ", "ANISOU"):
            dropped += 1
        elif rec == "CONECT":
            s.conect.append(line)
        elif rec == "MASTER":
            s.notes.append("Dropped the MASTER record (its counts no longer apply)")
        elif rec == "END":
            close_segment()
            ended = True
        elif started:
            s.trailer.append(line)
        else:
            s.header.append(line)
    close_segment()
    if not s.segments:
        slot = next((m.group(1).strip() for m in map(_EMPTY_SLOT_RE.match, s.header) if m), "")
        if slot:  # the model check passed on fewer models than it has slots: say so, not 'not a PDB file'
            raise PDBError(f"an empty slot of a model check ({slot}): no model, no atoms")
        raise PDBError("no ATOM/HETATM records found (not a PDB file, or the path does not exist)")
    standard_named = _read_component_names(s)
    if standard_named:
        s.notes.append(f"{len(standard_named)} 0DA / 0DT residue(s) use standard atom names where their CCD "
                       "components have their own (0DA: N9A, C8A ... HN61, HN62; 0DT: C5M): read as the same "
                       "atoms, written with the components' names: " + _listing(standard_named))
    if repaired:
        s.notes.append(f"Repaired {repaired} ChimeraX-damaged ATOM/HETATM lines (residue name fused with "
                       f"the chain id and a colon, e.g. 'GC:' -> residue G, chain C)")
    if n_models > 1:
        s.notes.append(f"Kept model 1 of {n_models}")
    if after_end:
        s.notes.append(f"Ignored {after_end} ATOM/HETATM lines after END (only the first model is kept)")
    if dropped:
        s.notes.append(f"Dropped {dropped} SIGATM/SIGUIJ/orphan ANISOU records")
    return s


# --- writing ------------------------------------------------------------------------------------------

def _element(atom: Atom, resname: str) -> str:
    if atom.element:
        return atom.element
    name = atom.name.strip()
    bare = re.sub(r"[\d+\-]", "", name).upper()
    if bare in _TWO_LETTER_ELEMENTS and bare == re.sub(r"[\d+\-]", "", resname).upper():
        return bare  # ion residue named after its element (MG, Na+, Cl- ...)
    letters = re.sub(r"^\d+", "", name)
    return letters[:1].upper() if letters else ""


def _atom_name_field(name: str, element: str) -> str:
    """PDB alignment: 4-character names and two-letter elements start in column 13, others in 14."""
    if len(name) >= 4 or len(element) == 2:
        return name[:4].ljust(4)
    return " " + name.ljust(3)


def _residue_field(resname: str) -> str:
    """Columns 18-21: names of up to 3 characters right-justified in 18-20, 4-character names in 18-21."""
    return f"{resname:>3} " if len(resname) <= 3 else resname[:4]


def _id_columns(res: Residue) -> str:
    """Columns 18-27 (residue name, chain, residue number, insertion code); numbers above 9999 wrap."""
    resseq = res.resseq % 10000 if res.resseq > 9999 else res.resseq
    return f"{_residue_field(res.name)}{(res.chain or ' ')[:1]}{resseq:>4}{(res.icode or ' ')[:1]}"


def _atom_line(atom: Atom, res: Residue, serial: int, record: Optional[str] = None) -> str:
    """ATOM/HETATM line (or, with ``record`` 'ANISOU', the ANISOU line of the atom); the atoms of a 0DA / 0DT
    under their CCD component's names (:func:`component_atom_name`)."""
    element = _element(atom, res.name)
    name = component_atom_name(res.name, atom.name)
    body = atom.tail.ljust(49)[:49] if record is None else atom.anisou.ljust(53)[:49]
    return (f"{record or atom.record:<6}{serial % 100000:>5} {_atom_name_field(name, element)}"
            f"{(atom.alt or ' ')[:1]}{_id_columns(res)}{body}{element:>2}{atom.charge:<2}")


def _seqres_insert_index(header: List[str]) -> int:
    for k, line in enumerate(header):
        if line[:6].strip().upper().startswith(_SEQRES_ORDER):
            return k
    return len(header)


def write_pdb(structure: Structure, strip_conect: bool = True, seqres: bool = False) -> str:
    """Format a structure: serials renumbered (TER takes one), TER after every segment, END; the atoms of a
    residue named 0DA / 0DT under their CCD component's names (N9A ..., C5M: ``CCD_ATOM_NAMES``).

    ``strip_conect``: CONECT records should be stripped until final deposition processing; when kept they
    are remapped to the new serials.  ``seqres``: (re)generate the SEQRES records in the header.
    """
    header = list(structure.header)
    if seqres:
        header = [line for line in header if not line.startswith("SEQRES")]
        k = _seqres_insert_index(header)
        header[k:k] = seqres_records(structure)
    out = list(header)
    if structure.model:
        out.append(structure.model)
    serial = 0
    serial_map: Dict[int, List[int]] = {}
    for seg in structure.segments:
        if not seg:
            continue
        for res in seg:
            for atom in res.atoms:
                serial += 1
                if atom.serial is not None:
                    serial_map.setdefault(atom.serial, []).append(serial)
                out.append(_atom_line(atom, res, serial))
                if atom.anisou:
                    out.append(_atom_line(atom, res, serial, "ANISOU"))
        serial += 1
        out.append(f"TER   {serial % 100000:>5}      {_id_columns(seg[-1])}")
    if structure.model:
        out.append("ENDMDL")
    out.extend(structure.trailer)
    if not strip_conect:
        out.extend(_remap_conect(structure.conect, serial_map))
    out.append("END")
    return "\n".join(line.rstrip() for line in out) + "\n"


def _remap_conect(lines: Sequence[str], serial_map: Dict[int, List[int]]) -> List[str]:
    if any(len(v) > 1 for v in serial_map.values()):
        return []  # ambiguous input serials (wrapped numbering): CONECT cannot be remapped
    out = []
    for line in lines:
        fields = [line.ljust(31)[i:i + 5] for i in range(6, 31, 5)]
        try:
            ids = [int(f) for f in fields if f.strip()]
        except ValueError:
            continue
        if not ids or ids[0] not in serial_map:
            continue
        new = [serial_map[ids[0]][0]] + [serial_map[i][0] for i in ids[1:] if i in serial_map]
        if len(new) > 1:
            out.append("CONECT" + "".join(f"{i % 100000:>5}" for i in new))
    return out


# --- LINK records -------------------------------------------------------------------------------------

class LinkAtom(NamedTuple):
    """One end of a LINK record."""
    name: str  # atom name
    resname: str
    chain: str
    resseq: int
    icode: str = ""

    def matches(self, res: Residue) -> bool:
        return (res.chain == self.chain and res.resseq == self.resseq and (res.icode or "") == self.icode
                and res.name.strip() == self.resname)


class LinkRecord(NamedTuple):
    """A covalent bond between residues that the polymer order does not imply (e.g. a 3'-3' phosphate).

    ``sym1`` / ``sym2``: the symmetry operators of the two atoms (columns 60-65 / 67-72), '' for the identity
    (1555, or blank as most programs write it), so that a record of the model compares equal to
    ``LinkRecord(first, second, length)``."""
    first: LinkAtom
    second: LinkAtom
    length: Optional[float] = None
    sym1: str = ""
    sym2: str = ""

    @property
    def mate(self) -> bool:
        """Whether the record bonds an atom of the model to one of a copy made by the crystal symmetry (the two
        operators differ): the second atom of that name in the model is not the one bonded, so the record is no
        bond of the model."""
        return self.sym1 != self.sym2


def link_line(first: LinkAtom, second: LinkAtom, length: Optional[float] = None) -> str:
    """A LINK record in the PDB 3.3 columns: atom names aligned as in ATOM records (13-16 / 43-46), residue
    names in 18-20 / 48-50 (four-character names reach 21 / 51), symmetry 1555 and the length (74-78)."""
    def part(a: LinkAtom) -> str:
        element = _element(Atom("ATOM", a.name, "", ""), a.resname)
        resseq = a.resseq % 10000 if a.resseq > 9999 else a.resseq
        return (f"{_atom_name_field(a.name, element)} {_residue_field(a.resname)}{(a.chain or ' ')[:1]}"
                f"{resseq:>4}{(a.icode or ' ')[:1]}")
    tail = f"  {'1555':>6} {'1555':>6} {length:5.2f}" if length is not None else ""
    return f"LINK        {part(first)}{' ' * 15}{part(second)}{tail}".rstrip()


def _link_atom(line: str, start: int) -> Optional[LinkAtom]:
    """The atom of a LINK record whose name field starts at column ``start`` + 1 (12 or 42)."""
    name = line[start:start + 4].strip()
    resname = line[start + 5:start + 9].strip()
    chain = line[start + 9:start + 10].strip()
    try:
        resseq = int(line[start + 10:start + 14])
    except ValueError:
        return None
    if not name or not resname:
        return None
    return LinkAtom(name.replace("*", "'"), resname, chain, resseq, line[start + 14:start + 15].strip())


def _is_link(line: str) -> bool:
    return line[:6].strip().upper() == "LINK"


def _symmetry(text: str) -> str:
    """A symmetry operator of a LINK record: '' for the identity (1555 or blank)."""
    text = text.strip()
    return "" if text in ("", "1555") else text


def _link_record(line: str) -> Optional[LinkRecord]:
    """The record of a LINK line; None when an atom field cannot be read."""
    line = line.ljust(80)
    first, second = _link_atom(line, 12), _link_atom(line, 42)
    if first is None or second is None:
        return None
    try:
        length: Optional[float] = float(line[73:78])
    except ValueError:
        length = None
    return LinkRecord(first, second, length, _symmetry(line[59:65]), _symmetry(line[66:72]))


def link_records(structure: Structure) -> List[LinkRecord]:
    """The LINK records of the header, then those that :func:`read_pdb` kept in the trailer (malformed ones
    skipped), each with its symmetry operators (a record to a symmetry mate, :attr:`LinkRecord.mate`, is no bond
    of the model).

    The PDB format puts LINK records in the header, but a record a script appended after the coordinates
    names the same bond: read where it is, it is a bond of the model check and follows the residues
    (:func:`follow_links`) like one in the header, instead of going stale unnoticed in the trailer, which the
    tools write back as it was."""
    lines = [line for line in structure.header + structure.trailer if _is_link(line)]
    return [record for record in map(_link_record, lines) if record is not None]


# Old spellings of atom names (PDB format 2: O1P ..., the thymine methyl C5M), read as the new ones when a LINK
# record is matched with the atoms of a model, as the model check matches them
_LINK_NAME_ALIASES = {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3", "C5M": "C7"}


def _link_name(name: str, resname: str = "") -> str:
    """The name an atom of residue ``resname`` is matched by: standard (N9A of 0DA is N9), new spellings."""
    name = standard_atom_name(resname, name.strip().replace("*", "'"))
    return _LINK_NAME_ALIASES.get(name, name)


def _link_text(record: LinkRecord) -> str:
    """'A:3 G O3' - A:4 PO2 P' for notes."""
    return " - ".join(f"{a.chain or '_'}:{a.resseq}{a.icode} {a.resname} {a.name}"
                      for a in (record.first, record.second))


class _FollowedLink(NamedTuple):
    """A LINK line as read and the residues and atom names its two ends name (None: an end names no atom)."""
    line: str
    record: Optional[LinkRecord]  # None: an atom field cannot be read (no record: left alone)
    ends: Optional[Tuple[Tuple[Residue, str], Tuple[Residue, str]]]


def _residue_index(structure: Structure) -> Dict[Tuple[str, int, str], List[Residue]]:
    index: Dict[Tuple[str, int, str], List[Residue]] = {}
    for res in structure.residues():
        index.setdefault((res.chain, res.resseq, res.icode or ""), []).append(res)
    return index


def _find_link_atom(index: Dict[Tuple[str, int, str], List[Residue]], end: LinkAtom
                    ) -> Optional[Tuple[Residue, str]]:
    """The residue and atom name an end of a LINK record names: the first residue in file order with its chain,
    number, insertion code and name (:meth:`LinkAtom.matches`) that has the atom, as the model check bonds it."""
    name = _link_name(end.name, end.resname)
    for res in index.get((end.chain, end.resseq, end.icode or ""), ()):
        if end.matches(res) and any(_link_name(a.name, res.name) == name for a in res.atoms):
            return res, name
    return None


def _has_link_atom(present: Dict[int, Residue], end: Tuple[Residue, str]) -> bool:
    res, name = end
    return id(res) in present and any(_link_name(a.name, res.name) == name for a in res.atoms)


def _link_atom_name(end: LinkAtom, res: Residue) -> str:
    """The atom name of a LINK record end for its residue as it is now: the record's own spelling, unless the
    residue's new name is a CCD component that spells the atom another way than the old one did (N7 of a DA
    renamed 0DA is N7A, N7A of a 0DA renamed LDA is N7)."""
    std = standard_atom_name(end.resname, end.name)
    new = component_atom_name(res.name, std)
    return end.name if new == component_atom_name(end.resname, std) else new


def _link_now(link: _FollowedLink) -> LinkRecord:
    """The record of a followed LINK line for its residues as they are now."""
    first, second = (LinkAtom(_link_atom_name(end, res), res.name.strip(), res.chain, res.resseq, res.icode or "")
                     for end, (res, _name) in zip((link.record.first, link.record.second), link.ends))
    return link.record._replace(first=first, second=second)


def _relinked(link: _FollowedLink) -> str:
    """The LINK line with the residue columns of each end (18-27 / 48-57) written for its residue as it is now.
    An end whose residue kept its name, chain id, number and insertion code keeps its columns, and so do the
    atom names (but one a renamed residue's CCD component spells its own way: :func:`_link_atom_name`),
    alternate locations, symmetry operators and the length: the line is rewritten in place."""
    line, changed = link.line, False
    for start, end, (res, _name) in zip((17, 47), (link.record.first, link.record.second), link.ends):
        if (res.name.strip(), res.chain, res.resseq, res.icode or "") != (end.resname, end.chain, end.resseq,
                                                                          end.icode):
            line = line.ljust(start + 10)
            line, changed = line[:start] + _id_columns(res) + line[start + 10:], True
            name = _link_atom_name(end, res)
            if name != end.name:
                field = _atom_name_field(name, _element(Atom("ATOM", name, "", ""), res.name))
                line = line[:start - 5] + field + line[start - 1:]
    return line.rstrip() if changed else line


class _LinkFollower:
    """The LINK records of a structure tied to the residues and atoms they name (:func:`follow_links`).

    Every record is resolved once, before the operations run: an operation in between can leave two residues
    with one chain id, number and name (combining chains before renumbering), and a record resolved again
    after it would follow the wrong residue."""

    def __init__(self, structure: Structure):
        index = _residue_index(structure)
        self.links: List[_FollowedLink] = []
        for line in structure.header + structure.trailer:
            if not _is_link(line):
                continue
            record = _link_record(line)
            ends = None
            if record is not None:
                first, second = _find_link_atom(index, record.first), _find_link_atom(index, record.second)
                ends = (first, second) if first and second else None
            self.links.append(_FollowedLink(line, record, ends))

    def partners(self, structure: Structure, res: Residue, names: Sequence[str]) -> List[Residue]:
        """The residues that LINK records bond the atoms ``names`` of ``res`` to (both atoms still there; a
        record to a symmetry mate bonds nothing of the model)."""
        wanted = {_link_name(n) for n in names}
        present = {id(r): r for r in structure.residues()}
        out = []
        for link in self.links:
            if (link.ends is None or link.record.mate
                    or not all(_has_link_atom(present, end) for end in link.ends)):
                continue
            (first, first_atom), (second, second_atom) = link.ends
            if first is res and first_atom in wanted:
                out.append(second)
            if second is res and second_atom in wanted:
                out.append(first)
        return out

    def rewrite(self, structure: Structure) -> List[str]:
        """Rewrite the LINK lines of the header and trailer, in place, for the residues as they are now; drop
        those whose residue or atom is gone.  Returns the notes."""
        present = {id(r): r for r in structure.residues()}
        queue: Dict[str, List[_FollowedLink]] = {}
        for link in self.links:
            queue.setdefault(link.line, []).append(link)
        rewritten: List[str] = []
        dropped: List[str] = []
        unmatched: List[str] = []
        followed: List[_FollowedLink] = []

        def follow(lines: List[str]) -> List[str]:
            out = []
            for line in lines:
                todo = queue.get(line)
                link = todo.pop(0) if todo else None
                if link is None or link.record is None:  # no LINK line read here, or no record
                    out.append(line)
                elif link.ends is None:
                    unmatched.append(_link_text(link.record))
                    out.append(line)
                elif not all(_has_link_atom(present, end) for end in link.ends):
                    dropped.append(_link_text(link.record))
                else:
                    new = _relinked(link)
                    if new != line:
                        rewritten.append(_link_text(_link_now(link)))
                    followed.append(link)
                    out.append(new)
            return out

        structure.header[:] = follow(structure.header)
        structure.trailer[:] = follow(structure.trailer)
        index = _residue_index(structure)
        ambiguous = []
        for link in followed:  # what a reader of the new names finds: the first residue that fits
            now = _link_now(link)
            for (res, _name), end in zip(link.ends, (now.first, now.second)):
                found = _find_link_atom(index, end)
                if found is None or found[0] is not res:
                    ambiguous.append(f"{res.label} {end.name}")
        notes = []
        if rewritten:
            notes.append(f"Rewrote {len(rewritten)} LINK record(s) for the new residue names, numbers or chain ids: "
                         + _listing(rewritten))
        if dropped:
            notes.append(f"Dropped {len(dropped)} LINK record(s) whose residue or atom was removed: "
                         + _listing(dropped))
        if unmatched:
            notes.append(f"Kept {len(unmatched)} LINK record(s) as they were: they name no atom of the model (by "
                         "chain, residue number, insertion code, residue name and atom name), so they cannot follow "
                         "the residues: " + _listing(unmatched))
        if ambiguous:
            notes.append(f"WARNING: {len(ambiguous)} LINK record end(s) now name two residues: an earlier residue "
                         "has the same chain id, number, insertion code and name, and readers bond that one "
                         "(renumber the residues): " + _listing(ambiguous))
        return notes


_FOLLOWERS: Dict[int, _LinkFollower] = {}  # id(structure) -> the follower of the operations running on it


@contextlib.contextmanager
def follow_links(structure: Structure) -> Iterator[List[str]]:
    """Keep the LINK records of ``structure`` on the residues and atoms they name while the block renames,
    renumbers, re-chains or removes residues; yields the list that receives the notes when the block ends.

    Each record is resolved to its residues and atoms before the block runs (both ends by chain, residue number,
    insertion code and name, the atom by name; the first residue that has it, as the model check bonds them).
    Afterwards the residue columns of the records are rewritten, in place, from the residues' names, chain ids,
    numbers and insertion codes; everything else of a record stays as it was (atom names, symmetry operators,
    the length), and so does a record whose residues kept their ids.  A record whose residue or atom no longer
    exists is dropped, one that named no atom of the model stays as it was; each with a note.  Every
    operation of this module that edits residues follows the records itself; inside a block (:func:`clean`,
    :func:`to_lrna`) they leave that to the block, which resolved the records before the first of them ran.
    Nothing happens to a structure without LINK records, or when the block raises."""
    notes: List[str] = []
    key = id(structure)
    if key in _FOLLOWERS or not any(_is_link(line) for line in structure.header + structure.trailer):
        yield notes  # an outer block follows the records, or there are none
        return
    follower = _FOLLOWERS[key] = _LinkFollower(structure)
    try:
        yield notes
    finally:
        del _FOLLOWERS[key]
    notes += follower.rewrite(structure)


def _follows_links(operation):
    """Decorator of the operations ``(structure, ...) -> notes`` that edit residues: the LINK records follow
    (:func:`follow_links`), their notes after the operation's."""
    @functools.wraps(operation)
    def run(structure: Structure, *args, **kwargs) -> List[str]:
        with follow_links(structure) as link_notes:
            notes = operation(structure, *args, **kwargs)
        return notes + link_notes
    return run


# --- chirality ----------------------------------------------------------------------------------------

def chiral_volume(centre, a, b, c) -> float:
    """Signed volume a.(b x c) of the vectors from ``centre`` to atoms 1-3 (CIF _chem_comp_chir sign)."""
    va, vb, vc = ([float(p[i]) - float(centre[i]) for i in range(3)] for p in (a, b, c))
    cross = (vb[1] * vc[2] - vb[2] * vc[1], vb[2] * vc[0] - vb[0] * vc[2], vb[0] * vc[1] - vb[1] * vc[0])
    return va[0] * cross[0] + va[1] * cross[1] + va[2] * cross[2]


def sugar_chirality(coords: Dict[str, Vec], base: Optional[str] = None
                    ) -> Tuple[Optional[str], Dict[str, float]]:
    """Chirality call "D", "L" or None (no usable centre, or centres that disagree) from the C4', C3', C1'
    volumes; a nearly planar centre (below ``MIN_CHIRAL_VOLUME``) also prevents a call.

    ``coords``: atom name -> xyz of one residue; ``base`` selects the glycosidic nitrogen (N9 for A/G,
    N1 for C/U/T; unknown: N9 if present, else N1).  Returns the call and the volumes by centre.
    """
    if base in ("A", "G") or (base not in ("C", "U", "T") and "N9" in coords):
        glyco = "N9"
    else:
        glyco = "N1"
    volumes: Dict[str, float] = {}
    votes = Counter()
    for centre, partners, l_sign in L_CHIRAL_SIGNS:
        names = [glyco if p == "N" else p for p in partners]
        if centre not in coords or any(n not in coords for n in names):
            continue
        v = chiral_volume(coords[centre], *(coords[n] for n in names))
        volumes[centre] = v
        if abs(v) >= MIN_CHIRAL_VOLUME:
            votes["L" if v * l_sign > 0 else "D"] += 1
        else:
            votes["flat"] += 1  # a nearly planar centre: distorted geometry
    if not (votes["L"] or votes["D"]) or (votes["L"] and votes["D"]) or votes["flat"]:
        return None, volumes  # nothing usable, centres that disagree, or a flat centre: no call
    return ("L" if votes["L"] > votes["D"] else "D"), volumes


Range = Tuple[str, Optional[int], Optional[int]]


def parse_ranges(spec) -> List[Range]:
    """Residue ranges -> [(chain, start, end)].

    Accepts ``"C:45-73, A:1-5"``, ``"C:45"`` (one residue), ``"C"`` (whole chain), ``":45-73"`` (blank
    chain), a list of such strings, or (chain, start, end) tuples.
    """
    if spec is None:
        return []
    items = re.split(r"[,;\s]+", spec.strip()) if isinstance(spec, str) else list(spec)
    out: List[Range] = []
    for item in items:
        if isinstance(item, (tuple, list)):
            chain, a, b = (list(item) + [None, None])[:3]
            chain = str(chain).strip()
        elif not item:
            continue
        else:
            m = re.match(r"^([^:\s]?)(?::(-?\d+)(?:-(-?\d+))?)?$", item)
            if not m:
                raise PDBError(f"cannot read residue range {item!r} (expected e.g. C:45-73, C:45 or C)")
            chain, a, b = m.groups()
        start = None if a is None else int(a)
        end = start if b is None else int(b)
        out.append((chain, start, end))
    return out


def _in_ranges(res: Residue, ranges: Sequence[Range]) -> bool:
    for chain, start, end in ranges:
        if res.chain == chain and (start is None or start <= res.resseq <= end):
            return True
    return False


def _listing(items: Sequence[str], limit: int = 8) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _residue_ranges(residues: Sequence[Residue]) -> str:
    """'C:45-57, C:64-73' style summary of residues (consecutive numbers merged per chain)."""
    parts: List[List] = []
    for r in residues:
        if parts and parts[-1][0] == r.chain and r.resseq == parts[-1][2] + 1:
            parts[-1][2] = r.resseq
        else:
            parts.append([r.chain, r.resseq, r.resseq])
    return ", ".join(f"{c or '_'}:{a}" + (f"-{b}" if b != a else "") for c, a, b in parts)


def _volumes_text(vols: Dict[str, float]) -> str:
    return ", ".join(f"{k} {v:+.2f}" for k, v in vols.items())


def assign_chirality(structure: Structure, detect: bool = True, l_ranges=None) -> List[str]:
    """Set ``Residue.chirality`` of every nucleotide: override ranges (forced L) > detection > name."""
    ranges = parse_ranges(l_ranges)
    notes: List[str] = []
    l_in_d_names, d_in_l_names, undetermined, forced_d, l_dna, d_in_l_dna = [], [], [], [], [], []
    counts = Counter()
    in_range = 0
    for res in structure.residues():
        nt = _nucleotide(res)
        if nt is None:
            continue
        call, vols = sugar_chirality(res.coords(), nt.base) if detect else (None, {})
        forced = bool(ranges) and _in_ranges(res, ranges)
        in_range += forced
        if forced:
            res.chirality = "L"
            if call == "D":
                forced_d.append(f"{res.label} ({_volumes_text(vols)})")
        else:
            res.chirality = call or nt.chirality
            if detect and call is None:
                undetermined.append(res.label)
            elif call == "L" and nt.chirality == "D" and nt.sugar != "dna":  # L-DNA: the note below
                l_in_d_names.append(res)
            elif call == "D" and nt.chirality == "L":  # a scheme renames RNA, not DNA
                (d_in_l_dna if nt.sugar == "dna" else d_in_l_names).append(f"{res.label} ({_volumes_text(vols)})")
        if nt.sugar == "dna" and res.chirality == "L":
            l_dna.append(res.label)
        counts[res.chirality] += 1
    source = "sugar geometry" if detect else "residue names"
    notes.append(f"Chirality ({source}): {counts['L']} L and {counts['D']} D nucleotides"
                 + (f", {in_range} forced L by ranges" if ranges else ""))
    if l_in_d_names:
        notes.append(f"L sugars in {len(l_in_d_names)} residues with D names (e.g. cpptraj output), L names "
                     f"when a scheme is applied: {_residue_ranges(l_in_d_names)}")
    if d_in_l_names:
        notes.append("WARNING: L-named residues with D sugar geometry (D names when a scheme is applied; "
                     "force L with ranges if intended): " + _listing(d_in_l_names))
    if forced_d:
        notes.append("WARNING: forced L although the sugar geometry is D: " + _listing(forced_d))
    if undetermined:
        notes.append("Chirality undetermined (missing sugar atoms or centres that disagree), using the residue "
                     "name for "
                     + _listing(undetermined))
    if l_dna:
        notes.append("L-DNA residues keep their names (the PDB tools rename RNA only; the CCD names L-DNA 0DA, "
                     "0DC, 0DG, 0DT): " + _listing(l_dna))
    if d_in_l_dna:
        notes.append("WARNING: L-DNA names on residues with D sugar geometry (the PDB tools keep DNA names): "
                     + _listing(d_in_l_dna))
    if ranges and not in_range:
        notes.append("WARNING: the L ranges match no nucleotide")
    return notes


def in_ranges(res: Residue, ranges: Sequence[Range]) -> bool:
    """Whether a residue lies in parsed ranges; the chain '*' stands for every chain ('*:45-73')."""
    for chain, start, end in ranges:
        if (chain == "*" or res.chain == chain) and (start is None or start <= res.resseq <= end):
            return True
    return False


def ranges_text(residues: Sequence[Residue]) -> str:
    """Residues as ranges in :func:`parse_ranges` syntax ('A:1-16, B:17-44'; ':5-9' for a blank chain),
    consecutive numbers of a chain merged in the order given."""
    parts: List[List] = []
    for r in residues:
        if parts and parts[-1][0] == r.chain and r.resseq in (parts[-1][2], parts[-1][2] + 1):
            parts[-1][2] = r.resseq
        else:
            parts.append([r.chain, r.resseq, r.resseq])
    return ", ".join(f"{c}:{a}" + (f"-{b}" if b != a else "") for c, a, b in parts)


class ResidueHand(NamedTuple):
    """The sugar hand of one nucleotide (:func:`residue_hands`)."""
    residue: Residue
    base: str
    name_hand: Optional[str]  # the hand the residue name implies (0A, LA: L); None for an unrecognised name
    call: Optional[str]  # D / L from every usable centre (sugar_chirality); None: missing, flat, disagreeing
    c4: Optional[str]  # the hand of the C4' centre alone (O4'.(C3' x C5') < 0: D), None when flat or missing
    volumes: Dict[str, float]  # signed volume by centre (C4', C3', C1')
    note: str  # epimer, centres that disagree, flat centre, missing atoms ("" for a clean call)
    expected: Optional[str]  # from the expected ranges, else None

    @property
    def ok(self) -> Optional[bool]:
        """Measured as expected (None without an expectation; an undetermined hand is not as expected)."""
        return None if self.expected is None else self.call == self.expected

    @property
    def defect(self) -> bool:
        """An epimerised, contradicting or flat centre: the sugar is not a valid D or L ribose."""
        return any(k in self.note for k in ("epimer", "disagree", "flat"))


_L_SIGN = {centre: sign for centre, _partners, sign in L_CHIRAL_SIGNS}


def _hand_note(volumes: Dict[str, float]) -> str:
    """What prevents or qualifies a call: flat centres, a single centre against the others (an epimer;
    C1' is the anomeric carbon), centres that disagree without a majority, or nothing measurable."""
    if not volumes:
        return "no sugar centre measurable (missing atoms)"
    hands = {c: (None if abs(v) < MIN_CHIRAL_VOLUME else ("L" if v * _L_SIGN[c] > 0 else "D"))
             for c, v in volumes.items()}
    notes = []
    flat = [c for c, h in hands.items() if h is None]
    if flat:
        notes.append("flat centre at " + ", ".join(flat))
    measured = [c for c, h in hands.items() if h]
    votes = Counter(hands[c] for c in measured)
    if len(votes) > 1:
        major, n = votes.most_common(1)[0]
        if n > len(measured) - n:
            odd = [c for c in measured if hands[c] != major]
            what = "anomer, " if odd == ["C1'"] else ""
            notes.append(f"epimer at {', '.join(odd)} ({what}the other centres {major})")
        else:
            notes.append("centres disagree (" + ", ".join(f"{c} {hands[c]}" for c in measured) + ")")
    return "; ".join(notes)


def residue_hands(structure: Structure, expect_l=None, expect_d=None, infer: bool = True
                  ) -> List[ResidueHand]:
    """The measured sugar hand of every nucleotide, with the hand expected from ``expect_l`` / ``expect_d``
    (ranges, '*' for every chain).  The call needs every usable centre of C4', C3' and C1' to agree (as for
    the naming); a residue whose centres disagree is named as an epimer at the odd centre.  ``c4`` is the
    C4' centre alone: it misses C3' / C1' epimers (xylo sugars, alpha anomers) that real-space refinement
    can make.  ``infer`` also recognises nucleotides with unknown names from their atoms."""
    l_ranges, d_ranges = parse_ranges(expect_l), parse_ranges(expect_d)
    rows: List[ResidueHand] = []
    for res in structure.residues():
        nt = nucleotide(res, infer)
        if nt is None:
            continue
        call, vols = sugar_chirality(res.coords(), nt.base)
        v4 = vols.get("C4'")
        c4 = None if v4 is None or abs(v4) < MIN_CHIRAL_VOLUME else ("L" if v4 * _L_SIGN["C4'"] > 0 else "D")
        named = parse_residue_name(res.name)
        expected = "L" if in_ranges(res, l_ranges) else ("D" if in_ranges(res, d_ranges) else None)
        rows.append(ResidueHand(res, nt.base, named.chirality if named else None, call, c4, vols,
                                _hand_note(vols), expected))
    return rows


def hand_summary(rows: Sequence[ResidueHand], expected_given: bool = True) -> Tuple[List[str], List[str]]:
    """(notes, problems) of :func:`residue_hands` rows.  Problems are what a strict check fails on: residues
    measured against their expected hand, expected residues whose hand cannot be measured, and sugars with
    an epimerised, contradicting or flat centre.  Residues are listed as contiguous runs."""
    calls = Counter(r.call or "?" for r in rows)
    notes = [f"Sugar hands (C4'/C3'/C1' geometry): {calls['D']} D, {calls['L']} L"
             + (f", {calls['?']} undetermined" if calls["?"] else "")]
    problems: List[str] = []
    for want in ("L", "D"):
        wrong = [r.residue for r in rows if r.expected == want and r.call and r.call != want]
        if wrong:
            other = "D" if want == "L" else "L"
            problems.append(f"{len(wrong)} residue(s) expected {want} measured {other}: "
                            + ranges_text(wrong))
    unmeasured = [r.residue for r in rows if r.expected and not r.call and not r.defect]
    if unmeasured:
        problems.append(f"{len(unmeasured)} residue(s) with an expected hand cannot be measured (missing "
                        f"sugar atoms): {ranges_text(unmeasured)}")
    for r in rows:
        if r.defect:
            problems.append(f"{r.residue.label}: {r.note} ({_volumes_text(r.volumes)})")
    named = [r.residue for r in rows if r.name_hand and r.call and r.name_hand != r.call]
    if named:
        notes.append(f"{len(named)} residue name(s) imply the other hand than the geometry: "
                     + ranges_text(named))
    if expected_given and not any(r.expected for r in rows):
        notes.append("WARNING: the expected-hand ranges match no nucleotide")
    return notes, problems


def hands_table(rows: Sequence[ResidueHand]) -> str:
    """Tab-separated per-residue table of :func:`residue_hands` rows (volumes in Å^3)."""
    out = ["chain\tresidue\tname\tname_hand\tcall\tc4_hand\tvol_C4'\tvol_C3'\tvol_C1'\texpected\tok\tnote"]
    for r in rows:
        res = r.residue
        vols = [f"{r.volumes[c]:+.2f}" if c in r.volumes else "" for c in ("C4'", "C3'", "C1'")]
        ok = "" if r.ok is None else ("yes" if r.ok else "NO")
        out.append("\t".join([res.chain, f"{res.resseq}{res.icode}", res.name, r.name_hand or "",
                              r.call or "?", r.c4 or "?", *vols, r.expected or "", ok, r.note]))
    return "\n".join(out) + "\n"


# --- residue names ------------------------------------------------------------------------------------

def _phosphate_only(res: Residue) -> bool:
    """A residue of phosphate atoms only (P, OP1 ...): a lone 5'-phosphate, a chimera's PO2 bridge."""
    return bool(res.atoms) and all(a.name.replace("*", "'") in PHOSPHATE_5P for a in res.atoms)


def _linked_3prime(structure: Structure, links: Optional[_LinkFollower], res: Residue) -> bool:
    """Whether LINK records bond the O3' of ``res`` to another residue (water and ions aside): no 3' end, but a
    junction such as a chimera's 3'-3' phosphate."""
    return links is not None and any(r is not res and not _is_solvent(r)
                                     for r in links.partners(structure, res, ("O3'",)))


def _linked_5prime(structure: Structure, links: Optional[_LinkFollower], res: Residue) -> bool:
    """Whether LINK records bond the P or O5' of ``res`` to another residue than its own 5'-phosphate (a residue
    of phosphate atoms whose P is bonded to nothing else): no 5' end, but a junction such as a chimera's 5'-5'
    phosphate or the O3' of a strand before it.  Water and ions do not count, nor do records that name a
    phosphate oxygen (a metal's coordination)."""
    if links is None:
        return False
    for other in links.partners(structure, res, ("P", "O5'")):
        if other is res or _is_solvent(other):
            continue
        if _phosphate_only(other) and not any(r is not res and r is not other and not _is_solvent(r)
                                              for r in links.partners(structure, other, ("P",))):
            continue  # its own 5'-phosphate, modelled as a residue of its own
        return True
    return False


def _ends_state(five: bool, three: bool) -> str:
    return "N" if five and three else "5" if five else "3" if three else ""


def _terminals(seg: Sequence[Residue], structure: Optional[Structure] = None,
               links: Optional[_LinkFollower] = None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """(id(residue) -> terminal state, id(residue) -> the state of the segment ends alone) over the nucleotides
    of a segment.  The first nucleotide is a 5' end and the last a 3' end unless LINK records (``links``, the
    follower of ``structure``) bond that end to another residue: then the end is a junction and gets no terminal
    state; a single nucleotide is 'N' with both ends free."""
    nts = [r for r in seg if _nucleotide(r)]
    out = {id(r): "" for r in nts}
    plain = dict(out)
    if not nts:
        return out, plain
    first, last = nts[0], nts[-1]
    five = not _linked_5prime(structure, links, first)
    three = not _linked_3prime(structure, links, last)
    if len(nts) == 1:
        out[id(first)], plain[id(first)] = _ends_state(five, three), "N"
    else:
        out[id(first)], out[id(last)] = _ends_state(five, False), _ends_state(False, three)
        plain[id(first)], plain[id(last)] = "5", "3"
    return out, plain


@_follows_links
def convert_names(structure: Structure, scheme: str, only=None, terminals: bool = True,
                  dna: bool = False) -> List[str]:
    """Rename RNA residues to ``scheme``; ``only`` (ranges) limits the renaming to those residues.

    Terminal variants go to the free ends of a TER segment's nucleotides: an end that a LINK record bonds to
    another residue (the O3' / O5' of a chimera's 3'-3' / 5'-5' phosphate, the P of a strand bonded to the O3'
    of the one before it; not its own 5'-phosphate modelled as a residue, water or ions) is a junction and keeps
    the internal name, as the structure builder names it.  ``terminals=False`` writes chain ends with the
    internal names (LA rather than LA5/LA3, A rather than A5), for leap set-ups whose libraries map terminal
    residues with addPdbResMap.  ``dna``: rename the DNA residues (A, C, G, T; both hands) too, as
    :func:`scheme_name` names them (terminal variants in the amber scheme only); by default they keep their
    names (the PDB preparation renames RNA only)."""
    if scheme not in SCHEMES:
        raise PDBError(f"unknown residue name scheme {scheme!r} (choose from {', '.join(SCHEMES)})")
    ranges = parse_ranges(only)
    links = _FOLLOWERS.get(id(structure))  # the LINK records (the decorator follows them through this)
    changes = Counter()
    term_mismatch, phosphate, single_l, kept_phosphate, plain_l, l_dna = [], [], [], [], [], []
    junctions, bonded_o5 = [], []
    kept_dna = seen = 0  # DNA residues the scheme leaves alone; nucleotides it names
    for seg in structure.segments:
        terms, plain = _terminals(seg, structure, links)
        for res in seg:
            nt = _nucleotide(res)
            if nt is None or (ranges and not _in_ranges(res, ranges)):
                continue
            is_dna = nt.sugar == "dna"
            if is_dna and not (dna and nt.base in "ACGT"):
                kept_dna += 1
                continue
            seen += 1
            chir = res.chirality or nt.chirality
            has_p = any(a.name == "P" for a in res.atoms)
            # 0A5 ... stand for a 5'-OH end; a 5' end with its phosphate is the CCD component itself (a single
            # residue with a 5'-phosphate and a 3'-OH is the component, too)
            component = not is_dna and scheme == "ccd" and chir == "L" and has_p
            term = seg_term = terms[id(res)] if terminals else ""
            linked = plain[id(res)] != terms[id(res)]  # LINK records make a junction of a segment end
            if component and term in ("5", "N"):
                term = ""
                kept_phosphate.append(res.label)
            new = scheme_name(nt.base, chir, term, scheme, nt.sugar)
            free = "" if component and plain[id(res)] in ("5", "N") else plain[id(res)]  # (the end as if free)
            # LA5 stands in for LAN (L-DNA has its own LDAN)
            single_l_unit = not is_dna and seg_term == "N" and chir == "L" and nt.terminal == "5"
            if terminals and nt.terminal and nt.terminal != seg_term and not single_l_unit and not linked:
                term_mismatch.append(res.label)  # the old name shows the misplaced terminal
            if new != res.name:
                old = res.name.strip().upper()
                changes[(res.name, new)] += 1
                res.name = new
                if old in _CCD_STANDARD_NAMES:  # (atoms given the component's names in memory: standard again)
                    for a in res.atoms:
                        a.name = standard_atom_name(old, a.name)
            if terminals and linked and scheme_name(nt.base, chir, free, scheme, nt.sugar) != new:
                junctions.append(res.label)  # a free end would have another name
            if (scheme == "amber" and plain[id(res)] in ("5", "N") and terms[id(res)] not in ("5", "N")
                    and not has_p):
                bonded_o5.append(res.label)
            if scheme == "amber" and terms[id(res)] in ("5", "N") and has_p:
                phosphate.append(res.label)
            if scheme == "amber" and term == "N" and chir == "L" and not is_dna:
                single_l.append(res.label)
            if scheme == "standard" and chir == "L":
                plain_l.append(res)
            if scheme == "amber" and chir == "L" and is_dna:
                l_dna.append(res)
    notes = []
    if changes:
        summary = ", ".join(f"{a}->{b} x{n}" for (a, b), n in sorted(changes.items(), key=lambda kv: -kv[1]))
        notes.append(f"Renamed {sum(changes.values())} residues to the {scheme} scheme: {summary}")
    elif seen or not kept_dna:
        notes.append(f"Residue names already follow the {scheme} scheme")
    if kept_dna:
        notes.append(f"{kept_dna} DNA residue(s) keep their names: " + (
            "no name in the scheme for their base" if dna else
            "the PDB tools rename RNA only (the structure builder writes DNA in every scheme)"))
    if term_mismatch:
        notes.append("Terminal names that are not at a segment end (missing TER?): "
                     + _listing(term_mismatch))
    if junctions:
        notes.append("Segment ends that LINK records bond to another residue (a junction such as a chimera's 3'-3' "
                     "/ 5'-5' phosphate, or a strand bonded on) have internal names: " + _listing(junctions))
    if bonded_o5:
        notes.append("WARNING: the residues bonded through O5' by LINK records (" + _listing(bonded_o5) + ") have "
                     "internal names but no 5'-phosphate of their own: their units must not add one (tleap adds "
                     "the atoms a unit has and the file lacks)")
    if phosphate:
        notes.append("WARNING: 5'-terminal residues carry P/OP1/OP2, which the Amber 5' units do not have "
                     "(tleap cannot type them; remove them with --strip-5p or cap the end): " + _listing(phosphate))
    if kept_phosphate:
        notes.append("5'-terminal L residues with a 5'-phosphate keep the CCD component names (0A ..., which "
                     "include P/OP1/OP2/OP3; the 5' variants 0A5 ... have a 5'-OH): " + _listing(kept_phosphate))
    if single_l:
        notes.append("nucleic12_LRNA.lib has no unit for a single-residue L segment; wrote the 5' unit "
                     "(its O3' lacks a hydrogen): " + _listing(single_l))
    if l_dna:
        notes.append(f"L-DNA: {len(l_dna)} residue(s) ({ranges_text(l_dna)}) have MDATK's Amber-style names LDA, "
                     "LDC, LDG, LDT (three characters at the chain ends too); AmberTools ships no L-DNA library: the "
                     "Amber build step generates its units (LDNA.lib) from the DNA force field")
    if plain_l:
        notes.append(f"WARNING: the standard scheme gives {len(plain_l)} L residue(s) plain D names "
                     f"({ranges_text(plain_l)}): the names no longer record the hand. Use this copy only for "
                     "programs that ignore chirality (map-model scores such as Q-score, DSSR), never for "
                     "tleap, refinement or analyses that read residue names")
    return notes


def to_lrna(source, ranges, scheme: str = "amber", strip_conect: bool = True,
            terminals: bool = True) -> "CleanResult":
    """to_lrna.sh replacement: force L for ``ranges`` and rename only those residues (terminals kept unless
    ``terminals`` is False, which reproduces to_lrna.sh's plain LG/LA/LC/LU).  LINK records follow the renamed
    residues (:func:`follow_links`)."""
    s = read_pdb(source)
    notes = list(s.notes)
    with follow_links(s) as link_notes:
        notes += assign_chirality(s, detect=True, l_ranges=ranges)
        notes += convert_names(s, scheme, only=ranges, terminals=terminals)
    return CleanResult(write_pdb(s, strip_conect=strip_conect), notes + link_notes, None, s)


def set_records(structure: Structure, policy: str) -> List[str]:
    """Apply a record policy: 'keep', 'rosetta' (polymer residues as ATOM), 'pdb' (non-standard HETATM) or
    'atom' (every record ATOM, for programs that skip HETATM residues as ligands)."""
    if policy not in RECORD_POLICIES:
        raise PDBError(f"unknown record policy {policy!r} (choose from {', '.join(RECORD_POLICIES)})")
    if policy == "keep":
        return []
    changed = Counter()
    for res in structure.residues():
        if policy == "atom":
            record = "ATOM"
        elif policy == "rosetta":
            record = "ATOM" if _is_polymer(res) else None
        else:
            record = "ATOM" if res.name in _STANDARD_NAMES else "HETATM"
        if record and any(a.record != record for a in res.atoms):
            changed[(res.name, record)] += 1
            for a in res.atoms:
                a.record = record
    if not changed:
        return []
    summary = ", ".join(f"{name}->{rec} x{n}" for (name, rec), n in sorted(changed.items()))
    return [f"Record policy {policy}: {summary}"]


# --- chains -------------------------------------------------------------------------------------------

def _resegment(structure: Structure) -> None:
    """Split segments wherever the chain id changes (a TER is written between chains)."""
    segments = []
    for seg in structure.segments:
        cur: List[Residue] = []
        for res in seg:
            if cur and cur[-1].chain != res.chain:
                segments.append(cur)
                cur = []
            cur.append(res)
        if cur:
            segments.append(cur)
    structure.segments = segments


def _polymer_segments(structure: Structure) -> List[List[Residue]]:
    return [seg for seg in structure.segments if not all(_is_solvent(r) for r in seg)]


@_follows_links
def restore_chains(structure: Structure, reference=None, chain_ids: Optional[Sequence[str]] = None,
                   copy_numbers: bool = False) -> List[str]:
    """Restore chain ids (cpptraj output has none).

    ``reference``: copy ids residue by residue from a reference PDB (path, text or Structure); residues
    (water and ions excluded) are matched in order and compared by base letter, ignoring D/L and terminal
    naming; ``copy_numbers`` also copies residue numbers and insertion codes.  Otherwise ``chain_ids``
    gives one id per TER segment, else segments get A, B, C ...
    """
    if reference is not None:
        ref = read_pdb(reference)
        mine = [r for r in structure.residues() if not _is_solvent(r)]
        theirs = [r for r in ref.residues() if not _is_solvent(r)]
        if len(mine) != len(theirs):
            raise PDBError(f"chain restore: the structure has {len(mine)} residues but the reference has "
                           f"{len(theirs)} (water and ions excluded)")
        bad = [f"#{k + 1}: {a.label} vs reference {b.label}" for k, (a, b) in enumerate(zip(mine, theirs))
               if _code(a) != _code(b) or (_code(a) == "X" and a.name.upper() != b.name.upper())]
        if bad:
            raise PDBError(f"chain restore: sequence differs from the reference at {len(bad)} residue(s): "
                           + _listing(bad, 5))
        for a, b in zip(mine, theirs):
            a.chain = b.chain
            if copy_numbers:
                a.resseq, a.icode = b.resseq, b.icode
        _resegment(structure)
        return [f"Restored chain ids{' and residue numbers' if copy_numbers else ''} of {len(mine)} residues "
                f"from the reference (chains {', '.join(c or '_' for c in ref.chain_ids())})"]
    segments = _polymer_segments(structure)
    if chain_ids is not None:
        ids = [c.strip() for c in chain_ids]
        if len(ids) != len(segments):
            raise PDBError(f"chain restore: {len(ids)} chain ids given for {len(segments)} TER segments")
        if any(len(c) > 1 for c in ids):
            raise PDBError("chain ids must be single characters")
    else:
        if len(segments) > len(CHAIN_IDS):
            raise PDBError(f"{len(segments)} segments but only {len(CHAIN_IDS)} single-character chain ids")
        ids = list(CHAIN_IDS[:len(segments)])
    for seg, c in zip(segments, ids):
        for res in seg:
            res.chain = c
    _resegment(structure)
    return [f"Assigned chain ids {', '.join(c or '_' for c in ids)} to {len(segments)} TER segments"]


def _free_ids(structure: Structure) -> Iterator[str]:
    used = set(structure.chain_ids())
    for c in CHAIN_IDS:
        if c not in used:
            used.add(c)
            yield c
    raise PDBError("no unused single-character chain id left")


@_follows_links
def split_chains(structure: Structure, at=None) -> List[str]:
    """Give new unused chain ids to every further TER segment of a chain (``at`` None), or split chains
    at residues (``at``: e.g. ``"C:45"`` - residue 45 and the rest of chain C get a new id and a TER)."""
    free = _free_ids(structure)
    notes = []
    if at is None:
        seen = set()
        for seg in _polymer_segments(structure):
            chain = seg[0].chain
            if chain in seen:
                new = next(free)
                notes.append(f"Segment starting at {seg[0].label} -> chain {new}")
                for res in seg:
                    res.chain = new
            seen.add(chain)
        return notes or ["No chain spans several TER segments; nothing split"]
    order = {id(r): n for n, r in enumerate(structure.residues())}
    targets = []  # residues to split before, found by their input chain and number, in file order
    for chain, start, _ in parse_ranges(at):
        if start is None:
            raise PDBError(f"split position needs a residue number, e.g. {chain or ''}:45")
        res = next((r for r in structure.residues() if r.chain == chain and r.resseq == start), None)
        if res is None:
            raise PDBError(f"split position {chain}:{start} not found")
        targets.append(res)
    for target in sorted(targets, key=lambda r: order[id(r)]):
        chain, start = target.chain, target.resseq  # the chain may already be a new id from an earlier split
        i, k = next((i, k) for i, seg in enumerate(structure.segments) for k, r in enumerate(seg) if r is target)
        seg = structure.segments[i]
        if k > 0:
            structure.segments[i:i + 1] = [seg[:k], seg[k:]]
            i += 1
        new = next(free)
        for later in structure.segments[i:]:
            for res in later:
                if res.chain == chain:
                    res.chain = new
        notes.append(f"Split chain {chain or '_'} at residue {start}: the rest is chain {new}")
    return notes


@_follows_links
def combine_chains(structure: Structure, chains: Sequence[str], into: Optional[str] = None) -> List[str]:
    """Relabel ``chains`` as one chain (``into`` or the first listed); TER segments are kept."""
    chains = [c.strip() for c in chains]
    if not chains:
        return []
    target = chains[0] if into is None else into.strip()
    missing = [c for c in chains if c not in structure.chain_ids()]
    n = 0
    for res in structure.residues():
        if res.chain in chains:
            res.chain = target
            n += 1
    listed = ", ".join(c or "_" for c in chains)
    notes = [f"Combined chains {listed} into chain {target or '_'} ({n} residues)"]
    if missing:
        notes.append("WARNING: chains not present: " + ", ".join(missing))
    keys = Counter((r.chain, r.resseq, r.icode) for r in structure.residues() if r.chain == target)
    dup = sorted(k[1] for k, v in keys.items() if v > 1)
    if dup:
        notes.append(f"WARNING: chain {target or '_'} now has duplicate residue numbers ("
                     + _listing([str(d) for d in dup]) + "); renumber the residues")
    return notes


@_follows_links
def order_chains(structure: Structure, order: Optional[Sequence[str]] = None) -> List[str]:
    """Reorder segments by chain: listed chains first in ``order`` (alphabetical when None); water and
    ion segments stay at the end in their original order."""
    before = structure.chain_ids()
    rank = {c: k for k, c in enumerate(order)} if order is not None else None

    def key(item):
        k, seg = item
        if all(_is_solvent(r) for r in seg):
            return (2, 0, "", k)
        c = seg[0].chain
        if rank is None:
            return (0, c == "", c, k)
        return (0, 0, "", rank[c]) if c in rank else (1, 0, "", k)

    structure.segments = [seg for _, seg in sorted(enumerate(structure.segments), key=key)]
    after = structure.chain_ids()
    return [] if after == before else [f"Chain order: {' '.join(c or '_' for c in after)}"]


# --- numbering and sequence ---------------------------------------------------------------------------

PHOSPHATE_5P = ("P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P")


def _drop_empty(structure: Structure) -> None:
    """Remove residues without atoms and segments without residues."""
    structure.segments = [seg for seg in ([r for r in seg if r.atoms] for seg in structure.segments) if seg]


@_follows_links
def strip_hydrogens(structure: Structure) -> List[str]:
    """Remove hydrogens (element H / D, else by name: :func:`is_hydrogen`); tleap and most fitting programs
    rebuild or ignore them."""
    n = 0
    for res in structure.residues():
        keep = [a for a in res.atoms if not is_hydrogen(a, res.name)]
        n += len(res.atoms) - len(keep)
        res.atoms = keep
    _drop_empty(structure)
    return [f"Removed {n} hydrogen atoms"] if n else []


@_follows_links
def drop_virtual_residues(structure: Structure) -> List[str]:
    """Remove Rosetta's virtual residues (``VIRTUAL_RESIDUES``: VRT, XXX)."""
    dropped = []
    for seg in structure.segments:
        for res in seg:
            if res.name.strip().upper() in VIRTUAL_RESIDUES:
                dropped.append(res.label)
                res.atoms = []
    _drop_empty(structure)
    if not dropped:
        return []
    return [f"Removed {len(dropped)} Rosetta virtual residue(s), no atoms of the molecule: "
            + _listing(dropped)]


@_follows_links
def strip_5prime_phosphates(structure: Structure) -> List[str]:
    """Remove P/OP1/OP2/OP3 from the first nucleotide of every TER segment (Amber's 5'-terminal units
    have a 5'-OH).  A residue that has hydrogens gets HO5' where the phosphorus was (0.96 Å from O5').

    A phosphate whose P a LINK record bonds to a residue other than the nucleotide it belongs to (for a
    phosphate modelled as its own residue: the nucleotide after it) stays: it is a link, not a 5' end - the
    5'-5' phosphate of a chimera (a PO2 residue bonded to the O5' of two strands), or the phosphate of a strand
    whose P a LINK record bonds to the O3' of another.  Records that name a phosphate oxygen do not count: those
    of deposited structures give a metal ion's coordination."""
    links = _FOLLOWERS.get(id(structure))  # the LINK records (the decorator follows them through this)

    def junction(seg: List[Residue], res: Residue) -> bool:
        """Whether LINK records bond the P of ``res`` to a residue other than its own nucleotide."""
        if links is None:
            return False
        own: Optional[Residue] = res
        if _nucleotide(res) is None:  # a phosphate residue: its nucleotide follows it
            k = next(k for k, r in enumerate(seg) if r is res)
            own = next((r for r in seg[k + 1:] if _nucleotide(r)), None)
        return any(r is not res and r is not own and not _is_solvent(r)
                   for r in links.partners(structure, res, ("P",)))

    changed, skipped, emptied, linked = [], [], [], []
    for seg in structure.segments:
        res = next((r for r in seg if not _is_solvent(r)), None)
        while res is not None and _phosphate_only(res) and not junction(seg, res):
            emptied.append(res.label)  # a lone 5'-phosphate modelled as its own residue
            seg.remove(res)
            res = next((r for r in seg if not _is_solvent(r)), None)
        if res is None:
            continue
        if junction(seg, res):
            linked.append(res.label)  # the strand's 5' end is bonded on: no 5' end to strip
            continue
        if _nucleotide(res) is None:
            skipped.append(res.label)  # the segment starts with something else: its phosphates stay
            continue
        p_atoms = [a for a in res.atoms if a.name.replace("*", "'") in PHOSPHATE_5P]
        if not p_atoms:
            continue
        p = next((a for a in p_atoms if a.name == "P"), None)
        o5 = next((a for a in res.atoms if a.name.replace("*", "'") == "O5'"), None)
        has_h = any(a.name.lstrip("0123456789").startswith("H") for a in res.atoms)
        res.atoms = [a for a in res.atoms if a not in p_atoms]
        if has_h and p is not None and o5 is not None and not any(a.name == "HO5'" for a in res.atoms):
            o, d = o5.xyz, [p.xyz[k] - o5.xyz[k] for k in range(3)]
            n = math.sqrt(sum(v * v for v in d)) or 1.0
            h = [o[k] + 0.96 * d[k] / n for k in range(3)]
            tail = f"   {h[0]:8.3f}{h[1]:8.3f}{h[2]:8.3f}" + o5.tail[27:]
            res.atoms.insert(res.atoms.index(o5) + 1, Atom(o5.record, "HO5'", "", tail, "H"))
        changed.append(res.label)
    notes = []
    if changed:
        notes.append(f"Removed the 5'-terminal phosphate of {len(changed)} residue(s): " + _listing(changed))
    if emptied:
        notes.append(f"Removed {len(emptied)} 5'-terminal residue(s) that held only a phosphate: " + _listing(emptied))
    if skipped:
        notes.append("5' phosphates kept where a segment does not start with a recognised nucleotide: "
                     + _listing(skipped))
    if linked:
        notes.append("Phosphates kept where LINK records bond them to another residue (a link such as the 5'-5' "
                     "phosphate of a chimera, not a 5' end): " + _listing(linked))
    return notes


@_follows_links
def renumber_residues(structure: Structure, start: int = 1, per_chain: bool = False,
                      keep_gaps: bool = False) -> List[str]:
    """Renumber residues in file order, continuously across chains (each chain continues after the last
    number) or restarting at ``start`` per chain; ``keep_gaps`` keeps the numbering gaps (and insertion
    codes) inside a chain, otherwise numbers are sequential and insertion codes are cleared."""
    last: Dict[Optional[str], int] = {}
    prev: Optional[Residue] = None
    prev_old: Tuple[int, str] = (0, "")
    cleared = disorder = 0
    for res in structure.residues():
        key = res.chain if per_chain else None
        old = (res.resseq, res.icode)
        same_chain = prev is not None and prev.chain == res.chain
        if key not in last:
            new = start
        elif keep_gaps and same_chain and old > prev_old:
            new = last[key] + max(0, old[0] - prev_old[0])
        else:
            new = last[key] + 1
            disorder += keep_gaps and same_chain
        if not keep_gaps and res.icode:
            cleared += 1
            res.icode = ""
        prev, prev_old = res, old
        res.resseq = last[key] = new
    mode = "per chain" if per_chain else "continuously"
    notes = [f"Renumbered residues {mode} from {start}" + (" (gaps kept)" if keep_gaps else "")]
    if cleared:
        notes.append(f"Cleared {cleared} insertion codes")
    if disorder:
        notes.append(f"WARNING: {disorder} residues out of order; numbered sequentially there")
    if max((r.resseq for r in structure.residues()), default=0) > 9999:
        notes.append("WARNING: residue numbers above 9999 wrap in the PDB columns")
    return notes


def _sequence_chains(structure: Structure) -> Dict[str, List[Residue]]:
    """Chain -> residues of the polymer segments (water and ions excluded), in order.  A residue made only of
    phosphate atoms (a chimera's PO2 bridge, a lone 5'-phosphate) is no residue of the sequence either: it is
    bonded through LINK records or belongs to its nucleotide, and :func:`summary` lists it among the other
    residues (modified nucleotides, which have a sugar, stay)."""
    chains: Dict[str, List[Residue]] = {}
    for seg in _polymer_segments(structure):
        if not any(_is_polymer(r) for r in seg):
            continue  # ligand-only segment
        for res in seg:
            if not _is_solvent(res) and (_is_polymer(res) or not _phosphate_only(res)):
                chains.setdefault(res.chain, []).append(res)
    return chains


def _seqres_names(structure: Structure) -> Tuple[Dict[str, List[str]], List[str]]:
    """(chain -> the residue names of its SEQRES records, notes).  SEQRES names have three characters (columns
    20-22, 24-26 ...): a longer nucleotide name, such as the Amber-style L-DNA LDA5 / LDA3 / LDAN of older files, is written
    as its CCD component (0DA), any other one as UNK, each with a note."""
    out: Dict[str, List[str]] = {}
    mapped: Counter = Counter()
    unknown: List[str] = []
    for chain, residues in _sequence_chains(structure).items():
        names = []
        for res in residues:
            name = res.name.strip()
            nt = _nucleotide(res)
            # MDATK's Amber-style L-DNA names (LDA ...): in the CCD, LDA and LDT are other compounds
            amber_l_dna = nt is not None and nt.sugar == "dna" and nt.chirality == "L" and name.upper()[:2] == "LD"
            if len(name) > 3 or amber_l_dna:
                info = parse_residue_name(name)
                if info is not None:
                    new = scheme_name(info.base, info.chirality, "", "ccd", info.sugar)
                    mapped[(name, new)] += 1
                else:
                    new = "UNK"
                    unknown.append(res.label)
                name = new
            names.append(name)
        out[chain] = names
    notes = []
    if mapped:
        notes.append("SEQRES names are the CCD's three-character ones: wrote the CCD components of MDATK's Amber-style "
                     "L-DNA names and of other four-character names "
                     + ", ".join(f"{a}->{b} x{n}" for (a, b), n in sorted(mapped.items())))
    if unknown:
        notes.append("WARNING: SEQRES names have three characters: wrote UNK for the residue name(s) of "
                     + _listing(unknown))
    return out, notes


def seqres_records(structure: Structure) -> List[str]:
    """SEQRES records (13 residue names per line) of the modelled residues, names as written (three characters:
    MDATK's four-character L-DNA names as their CCD components, see :func:`_seqres_names`)."""
    out = []
    for chain, names in _seqres_names(structure)[0].items():
        for k in range(0, len(names), 13):
            body = " ".join(f"{n:>3}" for n in names[k:k + 13])
            out.append(f"SEQRES {k // 13 + 1:>3} {(chain or ' ')[:1]} {len(names):>4}  {body}")
    return out


def _compress(numbers: Sequence[int]) -> str:
    parts = []
    for n in numbers:
        if parts and n == parts[-1][1] + 1:
            parts[-1][1] = n
        else:
            parts.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def fasta(structure: Structure, title: str = "structure") -> str:
    """FASTA text, one entry per chain; L-nucleotides are noted in the header line."""
    out = []
    for chain, residues in _sequence_chains(structure).items():
        nts = [r for r in residues if _nucleotide(r)]
        sugars = {_nucleotide(r).sugar for r in nts}
        n_amino = sum(r.name.upper() in _AMINO or r.name.upper() in _AMINO_VARIANTS for r in residues)
        if nts and not n_amino:
            kind = "RNA" if sugars == {"rna"} else "DNA" if sugars == {"dna"} else "nucleic acid"
        else:
            kind = "protein" if n_amino and not nts else "polymer"
        l_res = [r for r in nts if _chirality(r) == "L"]
        if l_res and len(l_res) == len(nts):
            kind = "L-" + kind
        elif l_res:
            kind += f", L-nucleotides at {_compress([r.resseq for r in l_res])}"
        out.append(f">{title}_{chain or '_'} chain {chain or '_'} | {kind} | {len(residues)} residues")
        out.append("".join(_code(r) for r in residues))
    return "\n".join(out) + "\n"


# --- convenience entry point --------------------------------------------------------------------------

@dataclass
class CleanResult:
    text: str  # the PDB text
    notes: List[str]  # human-readable change notes
    fasta: Optional[str]  # FASTA text when requested
    structure: Structure


def clean(source, *, scheme: Optional[str] = None, records: Optional[str] = None, strip_conect: bool = True,
          detect_l: bool = True, l_ranges=None,
          chain_reference=None, chain_ids: Optional[Sequence[str]] = None, assign_chains: bool = False,
          copy_numbers: bool = False,
          split_segments: bool = False, split_at=None, combine: Optional[Sequence[str]] = None,
          combine_into: Optional[str] = None, sort_chains: Union[bool, Sequence[str]] = False,
          renumber: bool = False, start: int = 1, per_chain: bool = False, keep_gaps: bool = False,
          seqres: bool = False, make_fasta: bool = False, title: Optional[str] = None,
          terminal_names: bool = True, strip_5p: bool = False, strip_h: bool = False,
          drop_virtual: bool = False, rename_dna: bool = False) -> CleanResult:
    """Read, fix and write a PDB file in one call; returns the text plus change notes.

    Order: parse (ChimeraX repair, first model) -> Rosetta virtual residues (``drop_virtual``) and hydrogens
    (``strip_h``) removed -> chain restore (``chain_reference`` / ``chain_ids`` /
    ``assign_chains``) -> chirality (detection unless ``detect_l`` is False; ``l_ranges`` force L and use
    the input residue numbers with the restored chain ids) -> split / combine / ``sort_chains`` (True =
    alphabetical, or an explicit order) -> 5'-phosphate removal (``strip_5p``) -> residue names (``scheme``) ->
    renumbering -> LINK records rewritten for the residues they named in the input (:func:`follow_links`) ->
    record policy
    (``records``; None follows the scheme: 'rosetta' for rosetta, 'pdb' for pdb, 'atom' for standard, else
    'keep') -> writing.

    ``scheme='amber', strip_5p=True, strip_h=True, drop_virtual=True`` turns a Rosetta decoy into a PDB for
    tleap (L names from the geometry); ``scheme='standard'`` (with ``strip_h`` / ``drop_virtual``) writes the
    copy for chirality-blind programs.
    """
    s = read_pdb(source)
    notes = list(s.notes)
    stale = [line for line in s.header if line.startswith("REMARK 950")]
    if stale:  # base-pair annotation of the old names / numbers (mdatk.annotate)
        s.header = [line for line in s.header if not line.startswith("REMARK 950")]
        notes.append(f"Dropped {len(stale)} REMARK 950 base-pair annotation lines (they name the input "
                     "residues); annotate the cleaned file again if needed")
    with follow_links(s) as link_notes:  # every record resolved once, against the residues as read
        if drop_virtual:
            notes += drop_virtual_residues(s)
        elif scheme == "standard" and any(r.name.strip().upper() in VIRTUAL_RESIDUES for r in s.residues()):
            notes.append("WARNING: the copy keeps Rosetta virtual residues (VRT / XXX), which a map-model score "
                         "reads as atoms; --drop-virtual removes them")
        if strip_h:  # before the 5'-phosphate removal, which puts an HO5' on residues that have hydrogens
            notes += strip_hydrogens(s)
        if not s.segments:
            raise PDBError("no atoms left after removing hydrogens / virtual residues")
        if chain_reference is not None or chain_ids is not None or assign_chains:
            notes += restore_chains(s, chain_reference, chain_ids, copy_numbers)
        if detect_l or l_ranges:
            notes += assign_chirality(s, detect=detect_l, l_ranges=l_ranges)
        if split_segments:
            notes += split_chains(s)
        if split_at:
            notes += split_chains(s, split_at)
        if combine:
            notes += combine_chains(s, combine, combine_into)
        if sort_chains:
            notes += order_chains(s, None if sort_chains is True else list(sort_chains))
        if strip_5p:
            notes += strip_5prime_phosphates(s)
        if scheme:
            notes += convert_names(s, scheme, terminals=terminal_names, dna=rename_dna)
        if renumber:
            notes += renumber_residues(s, start, per_chain, keep_gaps)
    notes += link_notes
    policy = records or {"rosetta": "rosetta", "pdb": "pdb", "standard": "atom"}.get(scheme or "", "keep")
    notes += set_records(s, policy)
    if s.conect:
        notes.append(f"Stripped {len(s.conect)} CONECT records" if strip_conect else
                     "Kept CONECT records (remapped to the new atom serials)")
    n_atoms = sum(len(r.atoms) for r in s.residues())
    if n_atoms + len(s.segments) > 99999:
        notes.append("WARNING: more than 99999 atoms; serial numbers wrap in the PDB columns")
    if seqres:
        notes.append(f"Wrote SEQRES records for {len(_sequence_chains(s))} chain(s)")
        notes += _seqres_names(s)[1]
    name = title or "structure"
    is_path = isinstance(source, os.PathLike) or (isinstance(source, str) and "\n" not in source)
    if title is None and is_path:
        name = os.path.splitext(os.path.basename(str(source)))[0]
    text = write_pdb(s, strip_conect=strip_conect, seqres=seqres)
    return CleanResult(text, notes, fasta(s, name) if make_fasta else None, s)


# --- command line (``mdatk pdb`` and the copies embedded in generated scripts) -------------------------

def summary(structure: Structure) -> Dict[str, object]:
    """Chains, sequences, residue names and chirality calls of a parsed structure (for 'check')."""
    chains: Dict[str, Dict[str, object]] = {}
    for chain, residues in _sequence_chains(structure).items():
        calls = Counter(_chirality(r) or "?" for r in residues if _nucleotide(r))
        chains[chain or "_"] = {
            "residues": len(residues),
            "numbers": f"{residues[0].resseq}-{residues[-1].resseq}" if residues else "",
            "sequence": "".join(_code(r) for r in residues),
            "names": sorted({r.name for r in residues}),
            "chirality": dict(calls),
        }
    other = Counter(r.name for r in structure.residues() if not _is_polymer(r))
    return {"chains": chains, "segments": len(structure.segments), "other_residues": dict(other),
            "conect_records": len(structure.conect), "notes": list(structure.notes)}


def _clean_arguments(p) -> None:
    p.add_argument("input", help="PDB file")
    p.add_argument("--output", "-o", help="output PDB (default: print to stdout)")
    p.add_argument("--scheme", choices=SCHEMES, help="residue names: amber (LA/LA5... for the L-RNA force "
                   "field), ccd (0A... for Coot/Phenix), rosetta (0A..., all ATOM), pdb, standard (plain "
                   "A/C/G/U for both hands and all ATOM, only for programs that ignore chirality)")
    p.add_argument("--records", choices=RECORD_POLICIES, help="ATOM/HETATM policy (default: follows the "
                   "scheme)")
    p.add_argument("--strip-h", action="store_true", help="remove hydrogens (tleap rebuilds them)")
    p.add_argument("--drop-virtual", action="store_true", help="remove Rosetta's virtual residues (VRT, XXX)")
    p.add_argument("--no-terminal-names", action="store_true", help="chain ends keep the internal names "
                   "(LA, not LA5/LA3), for libraries that map terminals with addPdbResMap")
    p.add_argument("--dna", action="store_true", help="rename the DNA residues too (amber: DA5 / DA3 ... and "
                   "MDATK's L-DNA LDA ...; ccd: 0DA ...), as the structure builder names them")
    p.add_argument("--keep-conect", action="store_true", help="keep CONECT records (stripped by default "
                   "until final deposition processing)")
    p.add_argument("--no-detect", action="store_true", help="do not detect L-nucleotides from the sugar "
                   "chirality (C1'/C3'/C4'); use the residue names")
    p.add_argument("--l-ranges", help="force L for these residues, e.g. 'C:45-73' or 'C' (whole chain)")
    p.add_argument("--chain-reference", help="copy chain ids residue by residue from this PDB (e.g. the "
                   "structure a cpptraj PDB was built from)")
    p.add_argument("--chain-ids", help="chain ids per TER segment, e.g. A,B,C")
    p.add_argument("--assign-chains", action="store_true", help="assign A, B, C ... per TER segment")
    p.add_argument("--copy-numbers", action="store_true", help="with --chain-reference: also copy the "
                   "residue numbers")
    p.add_argument("--split-segments", action="store_true", help="give every TER segment its own new "
                   "chain id")
    p.add_argument("--split-at", help="split a chain before these residues, e.g. 'A:31'")
    p.add_argument("--combine", help="merge these chains, e.g. A,B")
    p.add_argument("--combine-into", help="chain id of the merged chain (default: the first)")
    p.add_argument("--sort-chains", action="store_true", help="write the chains in alphabetical order")
    p.add_argument("--renumber", action="store_true", help="renumber residues continuously across chains")
    p.add_argument("--start", type=int, default=1, help="first residue number (default 1)")
    p.add_argument("--per-chain", action="store_true", help="restart the numbering in every chain")
    p.add_argument("--keep-gaps", action="store_true", help="keep numbering gaps when renumbering")
    p.add_argument("--seqres", action="store_true", help="write SEQRES records")
    p.add_argument("--strip-5p", action="store_true", help="remove the phosphate (P/OP1/OP2/OP3) of every "
                   "chain's 5'-terminal nucleotide, as Amber's 5' units expect")
    p.add_argument("--fasta", help="also write the sequences to this FASTA file")


def clean_options(args) -> Dict[str, object]:
    """Keyword arguments of :func:`clean` from parsed command-line arguments."""
    def ids(text):
        return [c.strip() for c in text.split(",") if c.strip()] if text else None

    return dict(scheme=args.scheme, records=args.records, strip_conect=not args.keep_conect,
                detect_l=not args.no_detect, l_ranges=args.l_ranges or None,
                chain_reference=args.chain_reference, chain_ids=ids(args.chain_ids),
                assign_chains=args.assign_chains, copy_numbers=args.copy_numbers,
                split_segments=args.split_segments, split_at=args.split_at or None,
                combine=ids(args.combine), combine_into=args.combine_into, sort_chains=args.sort_chains,
                renumber=args.renumber, start=args.start, per_chain=args.per_chain, keep_gaps=args.keep_gaps,
                seqres=args.seqres, make_fasta=bool(args.fasta), terminal_names=not args.no_terminal_names,
                strip_5p=args.strip_5p, strip_h=getattr(args, "strip_h", False),
                drop_virtual=getattr(args, "drop_virtual", False), rename_dna=getattr(args, "dna", False))


def add_commands(sub) -> list:
    """Register 'clean', 'check' and 'to-lrna' on an argparse sub-parser collection (returns the parsers)."""
    c = sub.add_parser("clean", help="repair, rename, re-chain, renumber and rewrite a PDB file")
    _clean_arguments(c)
    k = sub.add_parser("check", help="report chains, sequences, residue names and sugar chirality")
    k.add_argument("input", help="PDB file")
    k.add_argument("--l-ranges", help="residues to treat as L, e.g. 'C:45-73'")
    k.add_argument("--expect-l", help="residues whose sugars must be L, e.g. 'C:45-73', 'C' or '*:45-73' "
                   "(every chain)")
    k.add_argument("--expect-d", help="residues whose sugars must be D, e.g. 'A,B'")
    k.add_argument("--strict", action="store_true", help="exit with status 1 when a residue has the other "
                   "hand than expected or cannot be measured, or a sugar has an epimerised or flat centre")
    k.add_argument("--table", action="store_true", help="print the per-residue table (chain, residue, name, "
                   "measured hands, C4'/C3'/C1' volumes, expectation, note) instead of the JSON summary")
    t = sub.add_parser("to-lrna", help="rename residues in ranges to L-RNA names (replaces to_lrna.sh)")
    t.add_argument("input", help="PDB file")
    t.add_argument("ranges", help="e.g. 'C:45-73' or 'A:1-5,A:9-12'")
    t.add_argument("--scheme", choices=SCHEMES, default="amber")
    t.add_argument("--no-terminal-names", action="store_true", help="plain LG/LA/LC/LU at chain ends, like "
                   "to_lrna.sh")
    t.add_argument("--output", "-o", help="output PDB (default: stdout)")
    return [c, k, t]


def run(args, stdout=None, stderr=None) -> Tuple[int, Dict[str, object]]:
    """Execute a parsed 'clean' / 'check' / 'to-lrna' command; returns (exit status, result)."""
    import json

    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if args.pdb_command == "check":
        s = read_pdb(args.input)
        notes = assign_chirality(s, detect=True, l_ranges=args.l_ranges or None)
        expect_l, expect_d = getattr(args, "expect_l", None) or None, getattr(args, "expect_d", None) or None
        rows = residue_hands(s, expect_l, expect_d)
        hand_notes, problems = hand_summary(rows, bool(expect_l or expect_d))
        info = summary(s)
        info["notes"] = list(info["notes"]) + notes + hand_notes
        info["problems"] = problems
        if getattr(args, "table", False):
            stdout.write(hands_table(rows))
            for note in info["notes"] + problems:
                stderr.write(f"pdbtools: {note}\n")
        else:
            stdout.write(json.dumps(info, indent=2) + "\n")
        return (1 if getattr(args, "strict", False) and problems else 0), info
    if args.pdb_command == "to-lrna":
        res = to_lrna(args.input, args.ranges, scheme=args.scheme, terminals=not args.no_terminal_names)
    else:
        res = clean(args.input, **clean_options(args))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(res.text)
    else:
        stdout.write(res.text)
    fasta_path = getattr(args, "fasta", None)
    if fasta_path and res.fasta:
        with open(fasta_path, "w", encoding="utf-8") as fh:
            fh.write(res.fasta)
    for note in res.notes:
        stderr.write(f"pdbtools: {note}\n")
    files = [f for f in (args.output, fasta_path) if f]
    return 0, {"notes": res.notes, "files": files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]) or "pdbtools",
                                     description="PDB hygiene for D- and L-RNA structures.")
    add_commands(parser.add_subparsers(dest="pdb_command", required=True, metavar="command"))
    args = parser.parse_args(argv)
    try:
        return run(args)[0]
    except (OSError, PDBError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
