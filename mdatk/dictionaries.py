"""Restraint dictionaries for the L-RNA residues of MDATK's 'ccd' naming scheme.

Coot and Phenix need an mmCIF restraint dictionary for every residue name they do not know, and its atom
names must be *identical* to the model's PDB atom names (O5', C5', ... OP1, OP2, HO5' ...).  The wwPDB
Chemical Component Dictionary (CCD) has the L-ribonucleotides 0A, 0C, 0G and 0U, but not the terminal
variants 0A5/0A3, 0C5/0C3, 0G5/0G3 and 0U5/0U3 that :mod:`mdatk.pdbtools` writes at chain ends with the
``ccd`` scheme.  acedrg (CCP4) and phenix.elbow (Phenix) generate restraints from a SMILES string but invent
their own atom names; this module supplies the SMILES and renames the generated dictionary afterwards:

* :data:`COMPONENTS` - the four L components and their eight variants: residue name, parent, isomeric
  SMILES (all four sugar centres C1', C2', C3', C4' L) and the reference atom-name graph (heavy atoms, bonds
  and which hydrogen names belong to which heavy atom), taken from the CCD entries 0A/0C/0G/0U as
  distributed in the CCP4 monomer library (acedrg-generated, ``_chem_comp.group RNA``, deprotonated
  phosphate with OP2/OP3 charged -1).
* :func:`rename_dictionary` ("finish") - reads an acedrg or eLBOW dictionary, maps every atom to the
  reference names by heavy-atom graph isomorphism (element + bonds, deterministic backtracking), names the
  hydrogens after their heavy atom with the CCD names (H5'/H5'' and H61/H62 ... from the generated
  coordinates when present), checks the sugar chirality and sets ``_chem_comp.id``, ``three_letter_code``,
  ``name`` and ``group``.  Loops and values are otherwise unchanged.
* :func:`fetch_pubchem_smiles` - optional PubChem PUG REST lookup.  It is the only network access and runs
  only when asked for explicitly ('mdatk dict pubchem', or the planner option that is off by default);
  a fetched SMILES must still match the reference graph and stereochemistry of its code.

Conventions of the variants (``<code>`` = 0A, 0C, 0G, 0U):

``<code>``   internal residue, as the CCD / monomer library: OP3 (P-O leaving atom, charged) and HO3' are
             kept in the dictionary; the polymer link (O3'(i-1)-P(i)) removes them, as for any CCD
             'RNA linking' component.  MDATK does not regenerate these (Coot / Phenix ship them), but they
             are in the table so that their SMILES can be listed and checked.
``<code>5``  5'-terminal variant: no 5'-phosphate (no P, OP1, OP2, OP3); O5' carries HO5'.  HO3' is kept as
             in the parent (leaving atom of the link to the next residue).
``<code>3``  3'-terminal variant: 5'-phosphate without the leaving OP3 (P, OP1, OP2; net charge -1, as an
             internal nucleotide) and a free 3'-OH (HO3').  A dictionary cannot be generated from such a
             fragment directly (P would be three-coordinate), so the program gets the parent molecule (the
             5'-monophosphate, OP3 included) and :func:`rename_dictionary` deletes OP3 and every restraint
             that uses it afterwards (``Component.leaving``).

``_chem_comp.group`` is ``RNA`` by default - what the monomer-library entries use and what Coot / Refmac /
Phenix need to apply their RNA backbone links between consecutive residues (the CCD itself types these
components 'L-RNA linking', a value the restraint programs do not link on).  ``non-polymer`` makes them
free ligands.

Only the Python standard library is used, so the module can be copied next to a generated script and run as
``python3 dictionaries.py finish ...`` (see :mod:`mdatk.workflows.dictstep`).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


class DictionaryError(ValueError):
    """Unreadable dictionary / SMILES, a structure that does not match its code, or a failed lookup."""


DEFAULT_GROUP = "RNA"
GROUPS = ("RNA", "non-polymer")
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# PubChem renamed the property 'IsomericSMILES' to 'SMILES' in 2025; ask for the new name first
PUBCHEM_SMILES_PROPERTIES = ("SMILES", "IsomericSMILES")
# hydrogens whose number may differ from the reference (phosphate protonation): heavy atom -> H name
PROTONATION_SITES = {"OP1": "HOP1", "OP2": "HOP2", "OP3": "HOP3"}


# --- reference data -----------------------------------------------------------------------------------
#
# Atoms, bonds, charges and hydrogen names from the CCD entries 0A, 0C, 0G, 0U (CCP4 monomer-library form,
# acedrg 243; wwPDB CCD data are public domain).  atoms: name:element[:charge]; bonds: a-b (single) / a=b
# (double); hydrogens: heavy:H,H.
#
# The SMILES are written for MDATK: L configuration at C1', C2', C3', C4' (CIP S, S, R, S - the mirror image
# of D-ribose), deprotonated phosphate as in the dictionaries, Kekule form.  The descriptors stored in the
# CCD files themselves (CACTVS / OpenEye SMILES, InChI .../m1/s1) and their ideal coordinates describe the
# D enantiomer; only their _chem_comp_chir signs are L.
#
# hpos tells the two hydrogens of a CH2 / NH2 group apart in generated coordinates, independent of the
# conformation: H:X,A,B,+ / - = sign of the chiral volume of X with neighbours A, B and H (H5'/H5'' of an
# L-nucleotide, the mirror image of the D naming; checked on L-RNA models); cis / trans = torsion H-X-A-B
# (amino hydrogens: H61, H41 and H21 point to the Watson-Crick edge).

_BACKBONE_ATOMS = "OP3:O:-1 P:P OP1:O OP2:O:-1 O5':O C5':C C4':C O4':O C3':C O3':O C2':C O2':O C1':C"
_BACKBONE_BONDS = ("OP3-P P=OP1 P-OP2 P-O5' O5'-C5' C5'-C4' C4'-O4' C4'-C3' O4'-C1' C3'-O3' C3'-C2' "
                   "C2'-O2' C2'-C1'")
_BACKBONE_H = "C5':H5',H5'' C4':H4' C3':H3' O3':HO3' C2':H2' O2':HO2' C1':H1'"
_BACKBONE_HPOS = "H5':C5',O5',C4',- H5'':C5',O5',C4',+"

_PARENT_DATA = {
    "0A": dict(
        name="L-ADENOSINE-5'-MONOPHOSPHATE", nucleoside="L-ADENOSINE", base="A",
        atoms="N9:N C8:C N7:N C5:C C6:C N6:N N1:N C2:C N3:N C4:C",
        bonds="C1'-N9 N9-C8 N9-C4 C8=N7 N7-C5 C5-C6 C5=C4 C6-N6 C6=N1 N1-C2 C2=N3 N3-C4",
        hydrogens="C8:H8 N6:H61,H62 C2:H2", hpos="H61:N6,C6,N1,cis H62:N6,C6,N1,trans",
        smiles="[O-]P(=O)([O-])OC[C@@H]1O[C@H](N2C=NC3=C(N)N=CN=C32)[C@@H](O)[C@H]1O",
        smiles5="OC[C@@H]1O[C@H](N2C=NC3=C(N)N=CN=C32)[C@@H](O)[C@H]1O"),
    "0C": dict(
        name="L-CYTIDINE-5'-MONOPHOSPHATE", nucleoside="L-CYTIDINE", base="C",
        atoms="N1:N C2:C O2:O N3:N C4:C N4:N C5:C C6:C",
        bonds="C1'-N1 N1-C2 N1-C6 C2=O2 C2-N3 N3=C4 C4-N4 C4-C5 C5=C6",
        hydrogens="N4:H41,H42 C5:H5 C6:H6", hpos="H41:N4,C4,N3,cis H42:N4,C4,N3,trans",
        smiles="[O-]P(=O)([O-])OC[C@@H]1O[C@H](N2C=CC(N)=NC2=O)[C@@H](O)[C@H]1O",
        smiles5="OC[C@@H]1O[C@H](N2C=CC(N)=NC2=O)[C@@H](O)[C@H]1O"),
    "0G": dict(
        name="L-GUANOSINE-5'-MONOPHOSPHATE", nucleoside="L-GUANOSINE", base="G",
        atoms="N9:N C8:C N7:N C5:C C6:C O6:O N1:N C2:C N2:N N3:N C4:C",
        bonds="C1'-N9 N9-C8 N9-C4 C8=N7 N7-C5 C5-C6 C5=C4 C6=O6 C6-N1 N1-C2 C2-N2 C2=N3 N3-C4",
        hydrogens="C8:H8 N1:H1 N2:H21,H22", hpos="H21:N2,C2,N1,cis H22:N2,C2,N1,trans",
        smiles="[O-]P(=O)([O-])OC[C@@H]1O[C@H](N2C=NC3=C2N=C(N)NC3=O)[C@@H](O)[C@H]1O",
        smiles5="OC[C@@H]1O[C@H](N2C=NC3=C2N=C(N)NC3=O)[C@@H](O)[C@H]1O"),
    "0U": dict(
        name="L-URIDINE-5'-MONOPHOSPHATE", nucleoside="L-URIDINE", base="U",
        atoms="N1:N C2:C O2:O N3:N C4:C O4:O C5:C C6:C",
        bonds="C1'-N1 N1-C2 N1-C6 C2=O2 C2-N3 N3-C4 C4=O4 C4-C5 C5=C6",
        hydrogens="N3:H3 C5:H5 C6:H6", hpos="",
        smiles="[O-]P(=O)([O-])OC[C@@H]1O[C@H](N2C=CC(=O)NC2=O)[C@@H](O)[C@H]1O",
        smiles5="OC[C@@H]1O[C@H](N2C=CC(=O)NC2=O)[C@@H](O)[C@H]1O"),
}
# signed chiral volumes of the L-ribose, from the _chem_comp_chir loops of the four entries: (centre,
# (atom 1, atom 2, atom 3), sign of a.(b x c)); "N" is N9 (purines) or N1 (pyrimidines).  The fourth
# neighbour of each centre is its hydrogen.  D-ribose has the opposite signs.
L_CHIRALITY = (("C4'", ("O4'", "C3'", "C5'"), 1),
               ("C3'", ("O3'", "C4'", "C2'"), -1),
               ("C2'", ("O2'", "C1'", "C3'"), 1),
               ("C1'", ("O4'", "N", "C2'"), 1))
PURINES = ("A", "G")


@dataclass(frozen=True)
class Component:
    """One residue code: the molecule given to acedrg / eLBOW and its reference atom names.

    ``atoms`` / ``bonds`` / ``hydrogens`` describe the *generation* molecule, i.e. the one ``smiles``
    encodes; ``leaving`` atoms (with their hydrogens) are removed from the dictionary after renaming, so
    the residue's own atoms are :meth:`residue_atoms`."""

    code: str
    parent: str
    base: str
    terminal: str  # "" (internal / CCD entry), "5" or "3"
    name: str  # _chem_comp.name
    smiles: str
    atoms: Tuple[Tuple[str, str, int], ...]  # heavy atoms: (name, element, formal charge)
    bonds: Tuple[Tuple[str, str, int], ...]  # (atom 1, atom 2, order)
    hydrogens: Tuple[Tuple[str, Tuple[str, ...]], ...]  # heavy atom -> its hydrogen names
    hpos: Tuple[Tuple[str, str, str, str, str], ...] = ()  # (H, X, A, B, "+" / "-" / "cis" / "trans")
    leaving: Tuple[str, ...] = ()
    description: str = ""

    @property
    def is_variant(self) -> bool:
        return bool(self.terminal)

    def h_names(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self.hydrogens)

    def residue_atoms(self) -> List[str]:
        """Atom names of the residue in the finished dictionary (heavy atoms first, CCD order)."""
        heavy = [a[0] for a in self.atoms if a[0] not in self.leaving]
        hyd = [h for x, hs in self.hydrogens if x not in self.leaving for h in hs]
        return heavy + hyd

    def glycosidic_n(self) -> str:
        return "N9" if self.base in PURINES else "N1"

    def chirality(self) -> List[Tuple[str, Tuple[str, str, str, str], int]]:
        """Reference centres: (centre, (a1, a2, a3, a4), sign of V(a1, a2, a3)); a4 is the hydrogen."""
        h = self.h_names()
        out = []
        for centre, nbrs, sign in L_CHIRALITY:
            a = tuple(self.glycosidic_n() if n == "N" else n for n in nbrs)
            out.append((centre, a + (h[centre][0],), sign))
        return out

    def graph(self, leaving: bool = True) -> "MolGraph":
        """Reference graph of the generation molecule (atom names set); ``leaving=False``: without the
        leaving atoms, i.e. the finished residue."""
        atoms = [a for a in self.atoms if leaving or a[0] not in self.leaving]
        index = {name: k for k, (name, _el, _q) in enumerate(atoms)}
        hmap = self.h_names()
        g = MolGraph()
        for name, el, q in atoms:
            g.add_atom(el, q, len(hmap.get(name, ())), name, list(hmap.get(name, ())))
        for a, b, order in self.bonds:
            if a in index and b in index:
                g.add_bond(index[a], index[b], order)
        return g

    def formula(self) -> str:
        return self.graph().formula()


def _split_atoms(text: str) -> List[Tuple[str, str, int]]:
    out = []
    for tok in text.split():
        parts = tok.split(":")
        out.append((parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 0))
    return out


def _split_bonds(text: str) -> List[Tuple[str, str, int]]:
    out = []
    for tok in text.split():
        m = re.match(r"^(.+?)([-=])(.+)$", tok)
        out.append((m.group(1), m.group(3), 2 if m.group(2) == "=" else 1))
    return out


def _split_hydrogens(text: str) -> List[Tuple[str, Tuple[str, ...]]]:
    out = []
    for tok in text.split():
        heavy, hs = tok.split(":")
        out.append((heavy, tuple(hs.split(","))))
    return out


def _split_hpos(text: str) -> List[Tuple[str, str, str, str, str]]:
    out = []
    for tok in text.split():
        h, rest = tok.split(":")
        x, a, b, kind = rest.split(",")
        out.append((h, x, a, b, kind))
    return out


PHOSPHATE = ("OP3", "P", "OP1", "OP2")


def _build_components() -> Dict[str, Component]:
    comps: Dict[str, Component] = {}
    for code, d in _PARENT_DATA.items():
        atoms = _split_atoms(_BACKBONE_ATOMS + " " + d["atoms"])
        bonds = _split_bonds(_BACKBONE_BONDS + " " + d["bonds"])
        hyd = _split_hydrogens(_BACKBONE_H + " " + d["hydrogens"])
        hpos = _split_hpos(_BACKBONE_HPOS + " " + d["hpos"])
        comps[code] = Component(code, code, d["base"], "", d["name"], d["smiles"], tuple(atoms), tuple(bonds),
                                tuple(hyd), tuple(hpos), (), "CCD entry (internal residue; OP3 and HO3' are "
                                                             "leaving atoms of the polymer link)")
        a5 = tuple(a for a in atoms if a[0] not in PHOSPHATE)
        b5 = tuple(b for b in bonds if b[0] not in PHOSPHATE and b[1] not in PHOSPHATE)
        h5 = tuple([("O5'", ("HO5'",))] + hyd)
        comps[code + "5"] = Component(code + "5", code, d["base"], "5",
                                      f"{d['nucleoside']} (5'-TERMINAL, 5'-OH)", d["smiles5"], a5, b5, h5,
                                      tuple(hpos), (), "5'-terminal variant: no 5'-phosphate, HO5' on O5'")
        comps[code + "3"] = Component(code + "3", code, d["base"], "3",
                                      f"{d['name']} (3'-TERMINAL, NO OP3)", d["smiles"], tuple(atoms),
                                      tuple(bonds), tuple(hyd), tuple(hpos), ("OP3",),
                                      "3'-terminal variant: P, OP1, OP2 without OP3, free 3'-OH (generated "
                                      "from the parent; OP3 is removed after renaming)")
    order = [c + t for c in _PARENT_DATA for t in ("5", "3")] + list(_PARENT_DATA)
    return {k: comps[k] for k in order}


# --- molecular graphs ---------------------------------------------------------------------------------

@dataclass
class MolGraph:
    """Heavy atoms with hydrogen counts; bond order 0 = aromatic / delocalised / unknown."""
    elements: List[str] = field(default_factory=list)
    charges: List[Optional[int]] = field(default_factory=list)
    hcount: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    hnames: List[List[str]] = field(default_factory=list)  # hydrogen names per heavy atom (dictionaries)
    adj: List[Dict[int, int]] = field(default_factory=list)
    # SMILES stereo: atom -> (neighbours in SMILES order, an atom index or "H", "@" / "@@")
    stereo: Dict[int, Tuple[List[Union[int, str]], str]] = field(default_factory=dict)

    def add_atom(self, element: str, charge: Optional[int] = 0, hcount: int = 0, name: str = "",
                 hnames: Optional[List[str]] = None) -> int:
        self.elements.append(element)
        self.charges.append(charge)
        self.hcount.append(hcount)
        self.names.append(name)
        self.hnames.append(list(hnames or []))
        self.adj.append({})
        return len(self.elements) - 1

    def add_bond(self, a: int, b: int, order: int = 1) -> None:
        self.adj[a][b] = order
        self.adj[b][a] = order

    def __len__(self) -> int:
        return len(self.elements)

    def formula(self) -> str:
        """Hill formula including hydrogens, with the net charge if known and non-zero."""
        counts: Dict[str, int] = {}
        for el in self.elements:
            counts[el] = counts.get(el, 0) + 1
        h = sum(self.hcount)
        if h:
            counts["H"] = counts.get("H", 0) + h
        keys = (["C"] if "C" in counts else []) + (["H"] if "C" in counts and "H" in counts else [])
        keys += sorted(k for k in counts if k not in keys)
        text = "".join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in keys)
        if all(q is not None for q in self.charges):
            q = sum(self.charges)
            if q:
                text += f"{'+' if q > 0 else '-'}{abs(q) if abs(q) > 1 else ''}"
        return text


# --- SMILES (enough for nucleotides and PubChem output: organic subset, brackets, rings, aromaticity) ---

_VALENCE = {"B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5), "S": (2, 4, 6), "F": (1,), "Cl": (1,),
            "Br": (1,), "I": (1,)}
_SMILES_TOKEN = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]|%\d\d|\d|[-=#$:/\\]|[().]|\*")
_BRACKET = re.compile(r"^\[(\d*)([A-Z][a-z]?|[a-z][a-z]?|\*)(@@|@(?:TH[12])?)?(H\d*)?([+-]+\d*)?(?::\d+)?\]$")
_BOND_ORDER = {"-": 1, "=": 2, "#": 3, "$": 4, ":": 0, "/": 1, "\\": 1}


def parse_smiles(smiles: str) -> MolGraph:
    """Heavy-atom graph of a SMILES string, with hydrogen counts, charges and tetrahedral stereo tags."""
    text = (smiles or "").strip().split()[0] if (smiles or "").strip() else ""
    if not text:
        raise DictionaryError("empty SMILES")
    pos, tokens = 0, []
    for m in _SMILES_TOKEN.finditer(text):
        if m.start() != pos:
            raise DictionaryError(f"SMILES: cannot read {text[pos:m.start()]!r} at position {pos + 1}")
        tokens.append(m.group(0))
        pos = m.end()
    if pos != len(text):
        raise DictionaryError(f"SMILES: cannot read {text[pos:]!r} at position {pos + 1}")
    el: List[str] = []
    arom: List[bool] = []
    charge: List[int] = []
    hexp: List[Optional[int]] = []  # bracket H count, None = implicit
    tag: List[str] = []
    nbrs: List[List[Union[int, Tuple[str, str]]]] = []
    has_prev: List[bool] = []
    bonds: Dict[Tuple[int, int], int] = {}
    prev: Optional[int] = None
    stack: List[Optional[int]] = []
    rings: Dict[str, Tuple[int, Optional[str]]] = {}
    pending: Optional[str] = None

    def bond(a: int, b: int, symbol: Optional[str]) -> None:
        key = (min(a, b), max(a, b))
        if key in bonds or a == b:
            raise DictionaryError("SMILES: duplicate bond / ring closure")
        if symbol is None:
            order = 0 if arom[a] and arom[b] else 1
        else:
            order = _BOND_ORDER[symbol]
        bonds[key] = order

    for tok in tokens:
        if tok == "(":
            stack.append(prev)
        elif tok == ")":
            if not stack:
                raise DictionaryError("SMILES: unbalanced ')'")
            prev = stack.pop()
        elif tok == ".":
            prev = None
        elif tok in _BOND_ORDER:
            pending = tok
        elif tok[0] == "%" or tok.isdigit():
            if prev is None:
                raise DictionaryError("SMILES: ring closure without an atom")
            digit = tok.lstrip("%")
            if digit in rings:
                other, sym = rings.pop(digit)
                bond(prev, other, pending or sym)
                nbrs[prev].append(other)
                k = nbrs[other].index(("ring", digit))
                nbrs[other][k] = prev
            else:
                rings[digit] = (prev, pending)
                nbrs[prev].append(("ring", digit))
            pending = None
        else:
            if tok.startswith("["):
                m = _BRACKET.match(tok)
                if not m:
                    raise DictionaryError(f"SMILES: cannot read the atom {tok}")
                sym = m.group(2)
                hs = m.group(4)
                h = 0 if not hs else (int(hs[1:]) if len(hs) > 1 else 1)
                q = 0
                if m.group(5):
                    signs = m.group(5)
                    digits = signs.lstrip("+-")
                    q = (int(digits) if digits else len(signs)) * (1 if signs[0] == "+" else -1)
                stereo = "@@" if (m.group(3) or "") in ("@@", "@TH2") else ("@" if m.group(3) else "")
            else:
                sym, h, q, stereo = tok, None, 0, ""
            aromatic = sym[0].islower()
            element = sym.capitalize() if sym != "*" else "*"
            idx = len(el)
            el.append(element)
            arom.append(aromatic)
            charge.append(q)
            hexp.append(h)
            tag.append(stereo)
            nbrs.append([])
            has_prev.append(prev is not None)
            if prev is not None:
                bond(prev, idx, pending)
                nbrs[prev].append(idx)
                nbrs[idx].append(prev)
            pending = None
            prev = idx
    if rings:
        raise DictionaryError(f"SMILES: unclosed ring bond {', '.join(sorted(rings))}")
    if stack:
        raise DictionaryError("SMILES: unbalanced '('")

    # implicit hydrogens of organic-subset atoms
    n = len(el)
    total_h = [0] * n
    for i in range(n):
        if hexp[i] is not None:
            total_h[i] = hexp[i]
            continue
        orders = [bonds[(min(i, j), max(i, j))] for j in nbrs[i] if isinstance(j, int)]
        vals = _VALENCE.get(el[i])
        if not vals:
            raise DictionaryError(f"SMILES: {el[i]} must be written in brackets")
        if arom[i]:
            used = sum(o if o else 1 for o in orders)
            total_h[i] = max(0, vals[0] - used - 1)
        else:
            used = sum(o if o else 1 for o in orders)
            fit = [v for v in vals if v >= used]
            total_h[i] = (fit[0] - used) if fit else 0
    # fold explicit hydrogen atoms ([H]) into their neighbour
    keep = [i for i in range(n) if not (el[i] == "H" and len(nbrs[i]) == 1)]
    new = {old: k for k, old in enumerate(keep)}
    g = MolGraph()
    for i in keep:
        g.add_atom(el[i], charge[i], total_h[i])
    for (a, b), order in bonds.items():
        if a in new and b in new:
            g.add_bond(new[a], new[b], order)
        else:  # explicit H: count it on the heavy atom
            heavy = new.get(a, new.get(b))
            if heavy is not None:
                g.hcount[heavy] += 1
    for i in keep:
        if not tag[i]:
            continue
        order: List[Union[int, str]] = []
        for j in nbrs[i]:
            order.append("H" if j not in new else new[j])
        if hexp[i]:  # the bracket H follows the preceding atom (or comes first)
            for _ in range(hexp[i]):
                order.insert(1 if has_prev[i] else 0, "H")
        if len(order) == 4:
            g.stereo[new[i]] = (order, tag[i])
    return g


# --- graph matching -----------------------------------------------------------------------------------

MAX_MATCHES = 2000


def _all_matches(ref: MolGraph, tgt: MolGraph) -> List[List[int]]:
    """Every heavy-atom isomorphism ref -> tgt (element + connectivity), up to MAX_MATCHES."""
    n = len(ref)
    if n != len(tgt) or sorted(ref.elements) != sorted(tgt.elements):
        return []
    if sum(len(a) for a in ref.adj) != sum(len(a) for a in tgt.adj):
        return []
    label_r = [(ref.elements[i], len(ref.adj[i])) for i in range(n)]
    label_t = [(tgt.elements[i], len(tgt.adj[i])) for i in range(n)]
    if sorted(label_r) != sorted(label_t):
        return []
    # visiting order: rarest label first, then always a neighbour of the mapped part (BFS)
    freq = {lab: label_t.count(lab) for lab in label_t}
    order: List[int] = []
    seen = set()
    while len(order) < n:
        start = min((i for i in range(n) if i not in seen), key=lambda i: (freq[label_r[i]], i))
        queue = [start]
        seen.add(start)
        while queue:
            i = queue.pop(0)
            order.append(i)
            for j in sorted(ref.adj[i], key=lambda j: (freq[label_r[j]], j)):
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
    results: List[List[int]] = []
    mapping = [-1] * n
    used = [False] * n

    def extend(k: int) -> None:
        if len(results) >= MAX_MATCHES:
            return
        if k == n:
            results.append(list(mapping))
            return
        r = order[k]
        mapped_nbrs = [j for j in ref.adj[r] if mapping[j] >= 0]
        if mapped_nbrs:
            cands = [t for t in tgt.adj[mapping[mapped_nbrs[0]]] if not used[t]]
        else:
            cands = [t for t in range(n) if not used[t]]
        for t in sorted(cands):
            if label_t[t] != label_r[r]:
                continue
            if any(mapping[j] not in tgt.adj[t] for j in mapped_nbrs):
                continue
            if sum(1 for u in tgt.adj[t] if used[u]) != len(mapped_nbrs):
                continue
            mapping[r] = t
            used[t] = True
            extend(k + 1)
            mapping[r] = -1
            used[t] = False

    extend(0)
    return results


def _score(ref: MolGraph, tgt: MolGraph, m: List[int]) -> Tuple[int, List[int]]:
    """Lower is better: H-count, charge and bond-order disagreements (unknown orders / charges ignored)."""
    bad = 0
    for i in range(len(ref)):
        t = m[i]
        if ref.hcount[i] != tgt.hcount[t]:
            bad += 4
        if ref.charges[i] is not None and tgt.charges[t] is not None and ref.charges[i] != tgt.charges[t]:
            bad += 2
        for j, order in ref.adj[i].items():
            if j > i:
                other = tgt.adj[t].get(m[j], 0)
                if order and other and order != other:
                    bad += 1
    return bad, m


def match_graph(ref: MolGraph, tgt: MolGraph) -> Optional[List[int]]:
    """Best heavy-atom mapping ref index -> tgt index (deterministic), None if not isomorphic."""
    matches = _all_matches(ref, tgt)
    if not matches:
        return None
    return min(_score(ref, tgt, m) for m in matches)[1]


def _parity(seq: Sequence, ref: Sequence) -> int:
    """+1 if ``seq`` is an even permutation of ``ref``, -1 if odd."""
    p = [list(ref).index(x) for x in seq]
    inv = sum(1 for a in range(len(p)) for b in range(a + 1, len(p)) if p[a] > p[b])
    return -1 if inv % 2 else 1


def _expected_sign(ref_order: Sequence[str], ref_sign: int, order: Sequence[str]) -> int:
    """Sign of V(o1, o2, o3) given sign(V(r1, r2, r3)) = ref_sign; orders list all four neighbours."""
    return ref_sign * _parity(order, ref_order)


def describe_mismatch(comp: Component, tgt: MolGraph, what: str) -> str:
    ref = comp.graph()
    return (f"{what} ({tgt.formula()}, {len(tgt)} heavy atoms) does not match {comp.code} "
            f"({ref.formula()}, {len(ref)} heavy atoms)")


def smiles_mapping(comp: Component, smiles: str) -> Tuple[Dict[int, str], List[str]]:
    """Map a SMILES string onto ``comp``'s atom names; checks graph, hydrogens and sugar stereo.

    Returns (SMILES atom index -> reference name, notes); raises DictionaryError if the SMILES is a
    different compound, a different protonation outside the phosphate, or has other / missing stereo."""
    tgt = parse_smiles(smiles)
    ref = comp.graph()
    m = match_graph(ref, tgt)
    if m is None:
        raise DictionaryError(describe_mismatch(comp, tgt, "the SMILES"))
    names = {m[i]: ref.names[i] for i in range(len(ref))}
    notes: List[str] = []
    for i in range(len(ref)):
        if ref.hcount[i] != tgt.hcount[m[i]]:
            if ref.names[i] in PROTONATION_SITES:
                notes.append(f"{ref.names[i]} has {tgt.hcount[m[i]]} H (reference {ref.hcount[i]}): different "
                             "phosphate protonation")
            else:
                raise DictionaryError(f"the SMILES has {tgt.hcount[m[i]]} H on {ref.names[i]}, {comp.code} has "
                                      f"{ref.hcount[i]} (different tautomer / protonation)")
    index = {name: i for i, name in enumerate(ref.names)}
    for centre, ref_order, sign in comp.chirality():
        t = m[index[centre]]
        if t not in tgt.stereo:
            raise DictionaryError(f"the SMILES does not specify the configuration at {centre} "
                                  "(isomeric SMILES needed)")
        order, stag = tgt.stereo[t]
        named = ["H" if x == "H" else names[x] for x in order]
        ref_named = [("H" if k == 3 else n) for k, n in enumerate(ref_order)]
        if sorted(named) != sorted(ref_named):
            raise DictionaryError(f"unexpected neighbours of {centre} in the SMILES")
        smiles_sign = 1 if stag == "@" else -1  # '@': V(n0, n1, n2) > 0
        if _expected_sign(named, smiles_sign, ref_named) != sign:
            raise DictionaryError(f"the SMILES has the opposite configuration at {centre} (D sugar or another "
                                  f"stereoisomer); {comp.code} is an L-nucleotide")
    extra = [names[t] for t in tgt.stereo if names[t] not in {c for c, _o, _s in comp.chirality()}]
    if extra:
        notes.append("stereo tags ignored at " + ", ".join(sorted(extra)))
    return names, notes


def check_smiles(code: str, smiles: str) -> List[str]:
    """Notes if ``smiles`` is a valid isomeric SMILES of ``code``; raises DictionaryError otherwise."""
    return smiles_mapping(get_component(code), smiles)[1]


def identify_smiles(smiles: str) -> List[str]:
    """Codes whose generation molecule ``smiles`` encodes (graph and sugar stereochemistry)."""
    out = []
    for code, comp in COMPONENTS.items():
        try:
            smiles_mapping(comp, smiles)
        except DictionaryError:
            continue
        out.append(code)
    return out


def get_component(code: str) -> Component:
    try:
        return COMPONENTS[code.strip().upper()]
    except KeyError:
        raise DictionaryError(f"unknown residue code {code!r} (known: {', '.join(COMPONENTS)})") from None


# --- mmCIF reading / writing --------------------------------------------------------------------------

class CifLoop:
    """A category: a ``loop_`` or a run of ``_tag value`` pairs (``pairs=True``, one row)."""

    def __init__(self, tags: Sequence[str], rows: Optional[List[List[str]]] = None, pairs: bool = False):
        self.tags = list(tags)
        self.rows = rows if rows is not None else []
        self.pairs = pairs

    @property
    def category(self) -> str:
        return self.tags[0].split(".")[0].lower() if self.tags else ""

    def col(self, item: str) -> int:
        """Index of ``_category.item`` (case-insensitive), -1 if absent."""
        for k, tag in enumerate(self.tags):
            if tag.split(".", 1)[-1].lower() == item.lower():
                return k
        return -1

    def column(self, item: str) -> List[str]:
        k = self.col(item)
        return [r[k] for r in self.rows] if k >= 0 else []


@dataclass
class CifBlock:
    name: str
    items: List[CifLoop] = field(default_factory=list)

    def get(self, category: str) -> Optional[CifLoop]:
        category = category.lower()
        for item in self.items:
            if item.category == category:
                return item
        return None


@dataclass
class CifDocument:
    header: List[str] = field(default_factory=list)  # comment lines before the first block
    blocks: List[CifBlock] = field(default_factory=list)

    def block(self, name: str) -> Optional[CifBlock]:
        for b in self.blocks:
            if b.name.lower() == name.lower():
                return b
        return None


def _cif_tokens(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):  # text field up to the next line starting with ';'
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            if i >= len(lines):
                raise DictionaryError("mmCIF: unterminated text field")
            if buf and not buf[0].strip():
                buf = buf[1:]
            yield "value", "\n".join(buf)
            i += 1
            continue
        pos, n = 0, len(line)
        while pos < n:
            c = line[pos]
            if c.isspace():
                pos += 1
                continue
            if c == "#":
                yield "comment", line[pos:]
                break
            if c in "'\"":
                end = pos + 1
                while True:
                    end = line.find(c, end)
                    if end < 0:
                        raise DictionaryError(f"mmCIF line {i + 1}: unterminated quote")
                    if end + 1 >= n or line[end + 1].isspace():
                        break
                    end += 1
                yield "value", line[pos + 1:end]
                pos = end + 1
                continue
            end = pos
            while end < n and not line[end].isspace():
                end += 1
            word = line[pos:end]
            pos = end
            low = word.lower()
            if low.startswith("data_"):
                yield "data", word[5:]
            elif low == "loop_":
                yield "loop", ""
            elif low.startswith("save_") or low in ("global_", "stop_"):
                yield "other", word
            elif word.startswith("_"):
                yield "tag", word
            else:
                yield "value", word
        i += 1


def parse_cif(text: str) -> CifDocument:
    """Blocks, loops and tag/value pairs of an mmCIF file (comments after the header are dropped)."""
    doc = CifDocument()
    block: Optional[CifBlock] = None
    loop: Optional[CifLoop] = None
    values: List[str] = []
    tag: Optional[str] = None

    def close_loop():
        nonlocal loop, values
        if loop is None:
            return
        if not loop.tags or len(values) % len(loop.tags):
            raise DictionaryError(f"mmCIF: loop {loop.category or '?'} has {len(values)} values for "
                                  f"{len(loop.tags)} columns")
        w = len(loop.tags)
        loop.rows = [values[k:k + w] for k in range(0, len(values), w)]
        loop, values = None, []

    for kind, value in _cif_tokens(text):
        if kind == "comment":
            if block is None:
                doc.header.append(value)
            continue
        if kind == "data":
            close_loop()
            block = CifBlock(value)
            doc.blocks.append(block)
            continue
        if block is None:
            # a monomer-library 'global_' section (_lib_name, _lib_version ...) before the first data_
            # block: not needed by the restraint programs, so it is dropped
            continue
        if kind == "loop":
            close_loop()
            if tag is not None:
                raise DictionaryError(f"mmCIF: {tag} has no value")
            loop = CifLoop([])
            block.items.append(loop)
        elif kind == "tag":
            if loop is not None and not values:
                loop.tags.append(value)
                continue
            close_loop()
            if tag is not None:
                raise DictionaryError(f"mmCIF: {tag} has no value")
            tag = value
        elif kind == "value":
            if loop is not None:
                values.append(value)
            elif tag is not None:
                cat = tag.split(".")[0].lower()
                last = block.items[-1] if block.items else None
                if (last is not None and last.pairs and last.category == cat
                        and last.col(tag.split(".", 1)[-1]) < 0):
                    last.tags.append(tag)
                    last.rows[0].append(value)
                else:
                    block.items.append(CifLoop([tag], [[value]], pairs=True))
                tag = None
            else:
                raise DictionaryError(f"mmCIF: value {value!r} without a tag")
    close_loop()
    if tag is not None:
        raise DictionaryError(f"mmCIF: {tag} has no value")
    return doc


_RESERVED = ("data_", "save_", "loop_", "global_", "stop_")


def cif_value(value: str) -> str:
    """A value as an mmCIF token (atom names with primes are double-quoted, as in the CCD)."""
    if "\n" in value:
        return "\n;" + value + "\n;\n"
    if value == "":
        return "''"
    needs = (any(c.isspace() for c in value) or value[0] in "_#$'\"[];" or "'" in value or '"' in value
             or value.lower().startswith(_RESERVED))
    if not needs:
        return value
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return "\n;" + value + "\n;\n"


def write_cif(doc: CifDocument) -> str:
    out: List[str] = list(doc.header)
    for block in doc.blocks:
        out += (["#"] if not out or out[-1] != "#" else []) + [f"data_{block.name}", "#"]
        for item in block.items:
            if not item.tags:
                continue
            if item.pairs and len(item.rows) == 1:
                width = max(len(t) for t in item.tags)
                for t, v in zip(item.tags, item.rows[0]):
                    out.append(f"{t.ljust(width)}   {cif_value(v)}".rstrip())
                out.append("#")
                continue
            out.append("loop_")
            out += item.tags
            cells = [[cif_value(v) for v in row] for row in item.rows]
            widths = [max([len(c) for c in col if "\n" not in c] or [1]) for col in zip(*cells)] if cells else []
            for row in cells:
                if any("\n" in c for c in row):
                    out.append(" ".join(row))
                else:
                    out.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
            out.append("#")
    return "\n".join(out) + "\n"


# --- renaming generated dictionaries ------------------------------------------------------------------

_NAME_ITEMS = re.compile(r"^(atom_id|alt_atom_id|pdbx_component_atom_id|atom_id_\d+|atom_id_centre)$", re.I)
_COMP_ITEMS = re.compile(r"^(comp_id|pdbx_component_comp_id)$", re.I)
_COORDS = (("x", "y", "z"), ("model_cartn_x", "model_cartn_y", "model_cartn_z"),
           ("pdbx_model_cartn_x_ideal", "pdbx_model_cartn_y_ideal", "pdbx_model_cartn_z_ideal"))
_OUR_HEADER = re.compile(r"renamed by MDATK|^# _chem_comp\.group ")
_ORDERS = {"single": 1, "sing": 1, "1": 1, "double": 2, "doub": 2, "2": 2, "triple": 3, "trip": 3, "3": 3}


@dataclass
class RenameResult:
    text: str
    notes: List[str]
    mapping: Dict[str, str]  # old atom name -> new name ("" = removed leaving atom)
    source_id: str = ""
    program: str = ""


def _element(symbol: str, name: str) -> str:
    sym = (symbol or "").strip()
    if not sym or sym in ".?":
        m = re.match(r"[A-Za-z]+", name)
        sym = m.group(0)[:1] if m else "X"
    return sym[:1].upper() + sym[1:].lower()


def _float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _torsion(p0, p1, p2, p3) -> float:
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    b0, b1, b2 = sub(p0, p1), sub(p2, p1), sub(p3, p2)
    n1 = math.sqrt(dot(b1, b1)) or 1.0
    b1n = (b1[0] / n1, b1[1] / n1, b1[2] / n1)
    v = sub(b0, tuple(dot(b0, b1n) * c for c in b1n))
    w = sub(b2, tuple(dot(b2, b1n) * c for c in b1n))
    return math.degrees(math.atan2(dot(cross(b1n, v), w), dot(v, w)))


def _volume(c, a, b, d) -> float:
    va = (a[0] - c[0], a[1] - c[1], a[2] - c[2])
    vb = (b[0] - c[0], b[1] - c[1], b[2] - c[2])
    vd = (d[0] - c[0], d[1] - c[1], d[2] - c[2])
    return (va[0] * (vb[1] * vd[2] - vb[2] * vd[1]) - va[1] * (vb[0] * vd[2] - vb[2] * vd[0])
            + va[2] * (vb[0] * vd[1] - vb[1] * vd[0]))


def _heavy_count(atoms: CifLoop) -> int:
    names = atoms.column("atom_id")
    symbols = atoms.column("type_symbol") or [""] * len(names)
    return sum(1 for sym, name in zip(symbols, names) if _element(sym, name) not in ("H", "D"))


def _component_block(doc: CifDocument, source_id: str, comp: Optional[Component]) -> CifBlock:
    blocks = [b for b in doc.blocks if b.get("_chem_comp_atom") and b.get("_chem_comp_atom").rows]
    if not blocks:
        raise DictionaryError("no _chem_comp_atom loop found - is this a restraint dictionary?")
    if source_id:
        hits = [b for b in blocks if b.name.lower() == f"comp_{source_id}".lower()
                or source_id in b.get("_chem_comp_atom").column("comp_id")]
        if not hits:
            raise DictionaryError(f"no component {source_id!r} in the dictionary (found: "
                                  f"{', '.join(b.name for b in blocks)})")
        return hits[0]
    if len(blocks) == 1:
        return blocks[0]
    if comp is not None:  # the block whose heavy-atom count fits the reference
        sizes = (len(comp.atoms), len(comp.atoms) - len(comp.leaving))
        hits = [b for b in blocks if _heavy_count(b.get("_chem_comp_atom")) in sizes]
        if len(hits) == 1:
            return hits[0]
    raise DictionaryError("the dictionary holds several components (" + ", ".join(b.name for b in blocks)
                          + "); choose one with --source-id")


def _dictionary_graph(block: CifBlock):
    """Heavy-atom graph of a dictionary block plus its hydrogen atoms and coordinates."""
    atoms = block.get("_chem_comp_atom")
    names = atoms.column("atom_id")
    if not names:
        raise DictionaryError("_chem_comp_atom has no atom_id column")
    symbols = atoms.column("type_symbol") or [""] * len(names)
    charges_raw = atoms.column("charge") or [None] * len(names)
    elements = [_element(s, a) for s, a in zip(symbols, names)]
    coords: Dict[str, Tuple[float, float, float]] = {}
    for cols in _COORDS:
        xyz = [atoms.column(c) for c in cols]
        if all(xyz):
            for name, x, y, z in zip(names, *xyz):
                p = (_float(x), _float(y), _float(z))
                if None not in p:
                    coords[name] = p  # type: ignore[assignment]
            if coords and any(any(abs(v) > 1e-6 for v in p) for p in coords.values()):
                break
            coords = {}
    if len(set(names)) != len(names):
        raise DictionaryError("duplicate atom names in _chem_comp_atom")
    g = MolGraph()
    index: Dict[str, int] = {}
    hydrogens: Dict[str, str] = {}
    for name, el, q in zip(names, elements, charges_raw):
        if el in ("H", "D"):
            hydrogens[name] = ""
            continue
        charge = None
        if q is not None:
            try:
                charge = int(round(float(q)))
                if abs(float(q) - charge) > 1e-6:
                    charge = None  # a partial charge, not a formal one
            except ValueError:
                charge = None
        index[name] = g.add_atom(el, charge, 0, name)
    bonds = block.get("_chem_comp_bond")
    if bonds is None or not bonds.rows:
        raise DictionaryError("no _chem_comp_bond loop - cannot match atoms without bonds")
    a1, a2 = bonds.column("atom_id_1"), bonds.column("atom_id_2")
    kinds = bonds.column("type") or bonds.column("value_order") or [""] * len(a1)
    for x, y, kind in zip(a1, a2, kinds):
        if x not in index and x not in hydrogens or y not in index and y not in hydrogens:
            raise DictionaryError(f"bond {x}-{y} names an atom missing from _chem_comp_atom")
        if x in hydrogens or y in hydrogens:
            h, heavy = (x, y) if x in hydrogens else (y, x)
            if heavy in hydrogens:
                raise DictionaryError(f"H-H bond {x}-{y}")
            if hydrogens[h]:
                raise DictionaryError(f"hydrogen {h} has two bonds")
            hydrogens[h] = heavy
            g.hcount[index[heavy]] += 1
            g.hnames[index[heavy]].append(h)
            continue
        g.add_bond(index[x], index[y], _ORDERS.get(kind.strip().lower(), 0))
    lone = [h for h, heavy in hydrogens.items() if not heavy]
    if lone:
        raise DictionaryError(f"hydrogens without a bond: {', '.join(lone)}")
    return g, coords


def _name_hydrogens(comp: Component, ref: MolGraph, tgt: MolGraph, m: List[int], coords,
                    allow_protonation: bool, notes: List[str], mirrored: bool = False) -> Dict[str, str]:
    """Old hydrogen name -> reference name for every hydrogen of the dictionary (``mirrored``: the
    coordinates are the D enantiomer, so CH2 hydrogens are told apart by the mirrored rule)."""
    hpos = {h: (x, a, b, ang) for h, x, a, b, ang in comp.hpos}
    ref_index = {n: i for i, n in enumerate(ref.names)}
    out: Dict[str, str] = {}
    taken = set(comp.residue_atoms()) | {a[0] for a in comp.atoms}
    for i, xname in enumerate(ref.names):
        gen = list(tgt.hnames[m[i]])
        want = list(ref.hnames[i])
        if len(gen) != len(want):
            if xname not in PROTONATION_SITES and not allow_protonation:
                raise DictionaryError(f"the dictionary has {len(gen)} H on {xname}, {comp.code} has {len(want)} "
                                      "(different protonation / tautomer; --allow-protonation accepts it)")
            notes.append(f"{xname} carries {len(gen)} H (reference {len(want)})")
        names = want[:len(gen)]
        k = 1
        while len(names) < len(gen):  # extra hydrogens: CCD-style name when there is one
            base = PROTONATION_SITES.get(xname) or "H" + xname.replace("'", "").replace('"', "")
            cand = base if k == 1 else f"{base}{k}"
            k += 1
            if cand not in taken:
                names.append(cand)
                taken.add(cand)
        if set(gen) == set(want):  # already the reference names (e.g. a CCD dictionary): keep them
            names = list(gen)
        elif len(gen) >= 2 and len(gen) == len(want) and all(h in hpos for h in want):
            x, a, b = hpos[want[0]][:3]
            pts = [coords.get(tgt.names[m[ref_index[n]]]) if n in ref_index else None for n in (x, a, b)]
            if all(p is not None for p in pts) and all(h in coords for h in gen):
                best = None
                for perm in _permutations(want):
                    cost = sum(_hpos_cost(hpos[n][3], coords[h], pts, mirrored) for h, n in zip(gen, perm))
                    if best is None or cost < best[0]:
                        best = (cost, perm)
                names = list(best[1])
        for old, new in zip(gen, names):
            out[old] = new
    return out


def _hpos_cost(kind: str, h, pts, mirrored: bool = False) -> int:
    """0 if hydrogen position ``h`` fits ``kind`` relative to X, A, B (``pts``), else 1."""
    x, a, b = pts
    if kind in ("+", "-"):
        return int((_volume(x, a, b, h) > 0) != ((kind == "+") != mirrored))
    cis = abs(_torsion(h, x, a, b)) < 90.0
    return int(cis != (kind == "cis"))


def _permutations(items: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    if len(items) <= 1:
        yield tuple(items)
        return
    for k, first in enumerate(items):
        for rest in _permutations(list(items[:k]) + list(items[k + 1:])):
            yield (first,) + rest


def _coordinate_hands(comp: Component, coords, mapping: Dict[str, str]) -> Dict[str, int]:
    """Sugar centre -> +1 (L) / -1 (D) from the dictionary's coordinates (centres without them left out)."""
    new_coords = {mapping.get(k, k): v for k, v in coords.items()}
    out = {}
    for centre, order, sign in comp.chirality():
        pts = [new_coords.get(n) for n in (centre,) + tuple(order[:3])]
        if all(p is not None for p in pts):
            vol = _volume(*pts)
            if abs(vol) > 0.1:
                out[centre] = 1 if (vol > 0) == (sign > 0) else -1
    return out


def _check_chirality(comp: Component, block: CifBlock, hands: Dict[str, int], mapping: Dict[str, str],
                     notes: List[str]) -> None:
    """Sugar centres of the renamed dictionary: the _chem_comp_chir signs decide, the coordinates when the
    dictionary has no chirality restraints."""
    ref = {c: (order, sign) for c, order, sign in comp.chirality()}
    chir = block.get("_chem_comp_chir")
    checked = set()
    if chir is not None and chir.rows:
        cols = [chir.col(c) for c in ("atom_id_centre", "atom_id_1", "atom_id_2", "atom_id_3", "volume_sign")]
        if min(cols) >= 0:
            for row in chir.rows:
                centre, b1, b2, b3 = (mapping.get(row[k], row[k]) for k in cols[:4])
                vs = row[cols[4]].lower()
                if centre not in ref or vs.startswith("both"):
                    continue
                order, sign = ref[centre]
                rest = [n for n in order if n not in (b1, b2, b3)]
                row_sign = 1 if vs.startswith("pos") else -1 if vs.startswith("neg") else 0
                if len(rest) != 1 or not row_sign:
                    continue
                if _expected_sign(order, sign, (b1, b2, b3, rest[0])) != row_sign:
                    raise DictionaryError(f"_chem_comp_chir: {centre} has the opposite configuration "
                                          f"({row[cols[4]]}); {comp.code} needs an L sugar - check the SMILES")
                checked.add(centre)
    wrong = [c for c, hand in hands.items() if hand < 0]
    if wrong and not checked:
        raise DictionaryError(f"the coordinates have the opposite configuration at {', '.join(wrong)}; "
                              f"{comp.code} needs an L sugar - check the SMILES")
    if wrong:
        notes.append(f"WARNING: the ideal coordinates have the D configuration at {', '.join(wrong)} although the "
                     "chirality restraints are L (programs that build the residue from these coordinates "
                     "place a D sugar)")
    missing = [c for c in ref if c not in checked and c not in hands]
    if missing:
        notes.append("configuration not checked (no chirality rows or coordinates) at " + ", ".join(missing))


def _program(doc: CifDocument) -> str:
    text = " ".join(doc.header).lower()
    for b in doc.blocks:
        gen = b.get("_pdbx_chem_comp_description_generator")
        if gen is not None:
            text += " " + " ".join(gen.column("program_name")).lower()
    if "elbow" in text:
        return "phenix.elbow"
    if "acedrg" in text:
        return "acedrg"
    return ""


def finish_dictionary(text: str, code: str, reference: Union[Component, str, None] = None,
                      group: str = DEFAULT_GROUP, name: str = "", source_id: str = "",
                      allow_protonation: bool = False, keep_names: bool = False) -> RenameResult:
    """Rename an acedrg / eLBOW restraint dictionary to the reference atom names of ``code``.

    ``reference`` defaults to :data:`COMPONENTS` [code]; ``keep_names=True`` (or a code without a
    reference) only sets the component id, name and group.  Leaving atoms of the reference (OP3 of the
    3'-terminal variants) are deleted together with every restraint row that uses them."""
    code = code.strip().upper()  # PDB residue names are upper case; Coot / Phenix match ids exactly
    if not re.fullmatch(r"[A-Za-z0-9]{1,5}", code):
        raise DictionaryError(f"residue code {code!r}: use 1-5 letters / digits")
    if isinstance(reference, str):
        reference = get_component(reference)
    comp = reference if reference is not None else COMPONENTS.get(code.upper())
    if keep_names:
        comp = None
    doc = parse_cif(text)
    block = _component_block(doc, source_id, comp)
    atoms = block.get("_chem_comp_atom")
    old_id = (atoms.column("comp_id") or [""])[0] or re.sub(r"^comp_", "", block.name, flags=re.I)
    notes: List[str] = []
    program = _program(doc)
    mapping: Dict[str, str] = {}
    removed: set = set()
    if comp is not None:
        tgt, coords = _dictionary_graph(block)
        ref = comp.graph(leaving=len(tgt) != len(comp.atoms) - len(comp.leaving) or not comp.leaving)
        m = match_graph(ref, tgt)
        if m is None:
            raise DictionaryError(describe_mismatch(comp, tgt, f"the dictionary {old_id}"))
        for i in range(len(ref)):
            mapping[tgt.names[m[i]]] = ref.names[i]
        hands = _coordinate_hands(comp, coords, mapping)
        mirrored = bool(hands) and all(h < 0 for h in hands.values())
        mapping.update(_name_hydrogens(comp, ref, tgt, m, coords, allow_protonation, notes, mirrored))
        _check_chirality(comp, block, hands, mapping, notes)
        removed.update(old for old, new in mapping.items() if new in comp.leaving)
        for i in range(len(ref)):
            if ref.names[i] in comp.leaving:
                removed.update(tgt.hnames[m[i]])
        changed = sum(1 for o, n in mapping.items() if o != n)
        notes.append(f"{changed} of {len(mapping)} atom names changed to the {comp.parent} (CCD) names"
                     + (f"; removed leaving atoms {', '.join(sorted(mapping[o] for o in removed))}"
                        if removed else ""))
        if len(set(mapping.values())) != len(mapping):
            raise DictionaryError("internal error: two atoms got the same name")
    elif not keep_names:
        notes.append(f"{code} has no reference atom names; the program's names are kept")
    new_names = {o: n for o, n in mapping.items() if o not in removed}
    # rename / delete inside the component block
    for item in block.items:
        name_cols = [k for k, t in enumerate(item.tags) if _NAME_ITEMS.match(t.split(".", 1)[-1])]
        comp_cols = [k for k, t in enumerate(item.tags) if _COMP_ITEMS.match(t.split(".", 1)[-1])]
        rows = []
        for row in item.rows:
            if any(row[k] in removed for k in name_cols):
                continue
            for k in name_cols:
                row[k] = new_names.get(row[k], row[k])
            for k in comp_cols:
                if row[k] == old_id or not row[k] or row[k] in ".?":
                    row[k] = code
            rows.append(row)
        item.rows = rows
    block.items = [it for it in block.items if it.rows or not it.pairs]
    block.name = f"comp_{code}"
    # the component list
    n_all = len(block.get("_chem_comp_atom").rows)
    n_heavy = _heavy_count(block.get("_chem_comp_atom"))
    title = name or (comp.name if comp is not None else (COMPONENTS[code.upper()].name
                                                         if code.upper() in COMPONENTS else code))
    clist = None
    for b in doc.blocks:
        loop = b.get("_chem_comp")
        if loop is not None and loop.col("id") >= 0:
            clist = loop
            break
    if clist is None:
        clist = CifLoop(["_chem_comp.id", "_chem_comp.three_letter_code", "_chem_comp.name", "_chem_comp.group",
                         "_chem_comp.number_atoms_all", "_chem_comp.number_atoms_nh", "_chem_comp.desc_level"],
                        [[old_id, old_id, "", "", "", "", "."]])
        doc.blocks.insert(0, CifBlock("comp_list", [clist]))
    row = None
    for r in clist.rows:
        if r[clist.col("id")] in (old_id, code):
            row = r
            break
    if row is None:
        row = [""] * len(clist.tags)
        clist.rows.append(row)
    for item, value in (("id", code), ("three_letter_code", code[:3]), ("name", title), ("group", group),
                        ("number_atoms_all", str(n_all)), ("number_atoms_nh", str(n_heavy))):
        k = clist.col(item)
        if k < 0:
            clist.tags.append(f"_chem_comp.{item}")
            for r in clist.rows:
                r.append(".")
            k = len(clist.tags) - 1
        row[k] = value
    for r in clist.rows:
        for k, v in enumerate(r):
            if v == "":
                r[k] = "."
    old_header = doc.header
    doc.header = [f"# {code}: restraint dictionary{' from ' + program if program else ''}, renamed by "
                  f"MDATK dictionaries.py"
                  + (f" to the wwPDB CCD atom names of {comp.parent}" if comp is not None else ""),
                  f"# _chem_comp.group {group}"]
    doc.header += [h for h in old_header if h.strip() != "#" and not _OUR_HEADER.search(h)]
    final = write_cif(doc)
    if comp is not None:
        got = set(block.get("_chem_comp_atom").column("atom_id"))
        missing = [a for a in comp.residue_atoms() if a not in got]
        if missing:
            notes.append("atoms of the model without restraints: " + ", ".join(missing))
    out_map = {o: ("" if o in removed else n) for o, n in mapping.items()}
    return RenameResult(final, notes, out_map, old_id, program)


def rename_dictionary(text: str, code: str, reference: Union[Component, str, None] = None,
                      group: str = DEFAULT_GROUP, **kwargs) -> str:
    """mmCIF text of the dictionary renamed to ``code`` and its reference atom names."""
    return finish_dictionary(text, code, reference, group=group, **kwargs).text


def is_dictionary_file(path: str) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read(2_000_000)
    except OSError:
        return False
    return "_chem_comp_atom." in text and "_chem_comp_bond." in text


def find_dictionary(sources: Sequence[str], code: str = "", exclude: Sequence[str] = ()) -> str:
    """The restraint dictionary a program wrote: files, folders (their *.cif) or glob patterns.

    acedrg and eLBOW name their outputs differently between versions; every candidate .cif holding
    _chem_comp_atom and _chem_comp_bond loops counts, files named after ``code`` first, then the newest."""
    skip = {os.path.abspath(p) for p in exclude}
    cands: List[str] = []
    for src in sources:
        if os.path.isdir(src):
            found = glob.glob(os.path.join(glob.escape(src), "*.cif"))
        elif any(c in src for c in "*?["):
            found = glob.glob(src)
        else:
            found = [src] if os.path.isfile(src) else []
        cands += [p for p in found if os.path.isfile(p) and os.path.abspath(p) not in skip]
    cands = sorted(set(cands))
    good = [p for p in cands if is_dictionary_file(p)]
    if not good:
        raise DictionaryError("no restraint dictionary (.cif with _chem_comp_atom and _chem_comp_bond loops) in "
                              + ", ".join(sources))

    def key(p):
        base = os.path.basename(p)
        return (code.lower() not in base.lower() if code else False, "tmp" in p.lower(), -os.path.getmtime(p), p)

    return sorted(good, key=key)[0]


# --- PubChem (network; only on explicit request) ------------------------------------------------------

def fetch_pubchem_smiles(cid_or_name: Union[int, str], timeout: float = 15.0) -> str:
    """Isomeric SMILES of a PubChem compound (CID or name) via PUG REST over HTTPS."""
    import socket
    import urllib.error
    import urllib.parse
    import urllib.request

    ident = str(cid_or_name).strip()
    if not ident:
        raise DictionaryError("no PubChem CID or name given")
    path = f"cid/{ident}" if ident.isdigit() else "name/" + urllib.parse.quote(ident, safe="")
    problem = ""
    for prop in PUBCHEM_SMILES_PROPERTIES:
        url = f"{PUBCHEM_REST}/compound/{path}/property/{prop}/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "MDATK dictionaries.py",
                                                   "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise DictionaryError(f"PubChem has no compound {ident!r}") from None
            if exc.code == 400:  # unknown property name on this server: try the older one
                problem = f"PubChem rejected the request for {prop} (HTTP 400)"
                continue
            raise DictionaryError(f"PubChem request failed: HTTP {exc.code} {exc.reason}") from None
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise DictionaryError(f"PubChem did not answer within {timeout:g} s") from None
            raise DictionaryError(f"could not reach PubChem ({reason}); check the network connection / "
                                  "proxy") from None
        except (socket.timeout, TimeoutError):
            raise DictionaryError(f"PubChem did not answer within {timeout:g} s") from None
        try:
            data = json.loads(raw.decode("utf-8"))
            props = data["PropertyTable"]["Properties"]
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            raise DictionaryError("PubChem returned an unreadable answer") from None
        for entry in props:
            for key in PUBCHEM_SMILES_PROPERTIES:
                if entry.get(key):
                    return str(entry[key])
        problem = f"PubChem's answer for {ident!r} holds no SMILES"
    raise DictionaryError(problem or f"no SMILES for {ident!r}")


# --- command line -------------------------------------------------------------------------------------

def add_commands(sub) -> list:
    """Register 'list', 'smiles', 'finish' and 'pubchem' on an argparse sub-parser collection."""
    p_list = sub.add_parser("list", help="residue codes with their SMILES")
    p_list.add_argument("--json", action="store_true", help="machine-readable output")
    p_smi = sub.add_parser("smiles", help="print the SMILES given to acedrg / eLBOW for a code")
    p_smi.add_argument("code")
    p_fin = sub.add_parser("finish", help="rename an acedrg / eLBOW dictionary to the model's atom names")
    p_fin.add_argument("input", nargs="+", help="dictionary .cif, or folders / glob patterns to search")
    p_fin.add_argument("--code", required=True, help="residue code, e.g. 0A5")
    p_fin.add_argument("--output", "-o", required=True, help="renamed dictionary")
    p_fin.add_argument("--group", default=DEFAULT_GROUP, help="_chem_comp.group (default RNA: polymer "
                       "linking in Coot / Phenix; 'non-polymer' for a free ligand)")
    p_fin.add_argument("--name", default="", help="_chem_comp.name (default: from the table)")
    p_fin.add_argument("--source-id", default="", help="component id in the input when it holds several")
    p_fin.add_argument("--allow-protonation", action="store_true", help="accept hydrogen counts that differ "
                       "from the reference outside the phosphate")
    p_fin.add_argument("--keep-names", action="store_true", help="only set id / name / group, keep the atom "
                       "names (codes without reference names)")
    p_pc = sub.add_parser("pubchem", help="fetch a SMILES from PubChem (network) and check it against the "
                                          "known codes")
    p_pc.add_argument("compound", help="PubChem CID or compound name")
    p_pc.add_argument("--code", help="only check against this code")
    p_pc.add_argument("--timeout", type=float, default=15.0, help="seconds (default 15)")
    return [p_list, p_smi, p_fin, p_pc]


def run(args, stdout=None, stderr=None) -> Tuple[int, Dict[str, object]]:
    """Execute a parsed dictionary command; returns (exit status, result with 'files' / 'notes')."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    cmd = args.dict_command
    if cmd == "list":
        rows = [{"code": c.code, "parent": c.parent, "terminal": c.terminal, "name": c.name, "smiles": c.smiles,
                 "leaving": list(c.leaving), "formula": c.formula()} for c in COMPONENTS.values()]
        if args.json:
            stdout.write(json.dumps(rows, indent=2) + "\n")
        else:
            for r in rows:
                extra = f"  ({', '.join(r['leaving'])} removed after generation)" if r["leaving"] else ""
                stdout.write(f"{r['code']:<4} {r['smiles']}{extra}\n")
        return 0, {"components": rows}
    if cmd == "smiles":
        comp = get_component(args.code)
        stdout.write(comp.smiles + "\n")
        if comp.leaving:
            stderr.write(f"dictionaries: {comp.code} is generated from this molecule; "
                         f"{', '.join(comp.leaving)} is removed by 'finish'\n")
        return 0, {"smiles": comp.smiles}
    if cmd == "finish":
        out_abs = os.path.abspath(args.output)
        src = find_dictionary(args.input, args.code, exclude=[out_abs])
        with open(src, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        res = finish_dictionary(text, args.code, group=args.group, name=args.name, source_id=args.source_id,
                                allow_protonation=args.allow_protonation, keep_names=args.keep_names)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(res.text)
        notes = [f"{src} ({res.program or 'unknown program'}, component {res.source_id}) -> {args.output}"]
        notes += res.notes
        for note in notes:
            stderr.write(f"dictionaries: {note}\n")
        return 0, {"files": [args.output], "notes": notes, "mapping": res.mapping}
    if cmd == "pubchem":
        smiles = fetch_pubchem_smiles(args.compound, args.timeout)
        stdout.write(smiles + "\n")
        codes = [args.code.upper()] if args.code else list(COMPONENTS)
        hits, problems = [], []
        for code in codes:
            try:
                notes = check_smiles(code, smiles)
            except DictionaryError as exc:
                problems.append(f"{code}: {exc}")
                continue
            hits.append(code)
            for n in notes:
                stderr.write(f"dictionaries: {code}: {n}\n")
        if hits:
            stderr.write(f"dictionaries: matches {', '.join(hits)}\n")
        else:
            stderr.write("dictionaries: matches no known code"
                         + (f" ({problems[0]})" if args.code and problems else "") + "\n")
        return 0, {"smiles": smiles, "matches": hits, "notes": problems}
    raise DictionaryError(f"unknown command {cmd!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]) or "dictionaries",
                                     description="Restraint dictionaries for L-RNA residues (CCD atom names).")
    add_commands(parser.add_subparsers(dest="dict_command", required=True, metavar="command"))
    args = parser.parse_args(argv)
    try:
        return run(args)[0]
    except (OSError, DictionaryError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


COMPONENTS: Dict[str, Component] = _build_components()


if __name__ == "__main__":
    sys.exit(main())
