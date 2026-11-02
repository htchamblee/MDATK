"""Interfaces between two parts of RNA models, model by model: a contact census, clustering of the
contact fingerprints, the support of rebuilt segments and interface records from a base-pair engine.

The two parts ("sides") are residue selections, so nothing assumes a register, a construct or a
numbering continuous across chains:

* chain-qualified ranges in the syntax of :func:`mdatk.pdbtools.parse_ranges` (``A:1-16,B:17-44``,
  ``C`` for a whole chain, ``:5-9`` for a blank chain id), plain ranges that apply to every chain
  (``1-40``), and project residue groups (``group:Aptamer``), mixed freely;
* side B may be left out: it is then every nucleotide not in side A.

Residues are keyed by (chain, residue number, insertion code) and read with
:func:`mdatk.pdbtools.read_pdb` (first model, first alternate location).  Only nucleotides take part:
a residue named as one in any supported scheme (A, RA5, LA, 0A, DA ...) with backbone atoms, or a
residue with an unknown name (Rosetta's '0', 'LA0', modified names) whose atoms show a sugar and a base
(the base is then inferred from the atoms).  Ions, water and Rosetta's virtual residues never count.
Hydrogens are recognised by element (atom names such as 1H5' included).  The sugar's hand (D / L) comes
from :func:`mdatk.pdbtools.sugar_chirality`.

Census (:func:`census`), counted across the A/B boundary only:

  contacts   heavy-atom pairs closer than the contact cut-off (4.5 Å), a buried-interface proxy
  stack      base-base stacking: ring centroids < 4.5 Å apart, ring normals within 30°, and at least
             2.5 Å of the centroid offset along the mean normal (a coplanar base pair has parallel
             normals too, but its offset lies in the base plane)
  wc / gu    canonical Watson-Crick and G·U wobble pairs between the sides from a base-pair engine:
             builtin (:class:`mdatk.secstruct.PairDetector`: the pair's hydrogen bonds by base
             identity, C1'-C1' 8.5-12.5 Å), barnaba or DSSR (Leontis-Westhof classes)
  other      non-canonical pairs between the sides (barnaba / DSSR only; empty for builtin)
  bph        base-phosphate hydrogen-bond contacts: a base N-H donor (A N6, G N1/N2, C N4, U/T N3;
             with ``bph_ch`` also the C-H donors of Zirbel's BPh classes, A C2/C8, G C8, C/U C5/C6,
             T C6) within the H-bond cut-off (3.5 Å) of a phosphate OP1/OP2 of the other side,
             counted per donor-acceptor atom pair, both directions
  ohph       2'-OH (O2') to OP1/OP2 of the other side, likewise
  mgbr       phosphate pairs (one per side) whose P-P distance lies in the window that lets one ion
             bind both inner-sphere (5-7 Å): a geometric opportunity, not an observed ion; every
             P-P pair across the boundary is tested, whether or not the residues touch
  residues   the residues of each side in contact, and the fingerprint: the residue pairs in contact
             (``A:8/C:60``, chain-qualified)

Clusters (:func:`cluster_census`): deterministic leader clustering of the fingerprints by Jaccard
distance, at the level of residue pairs, residues per side, or residues binned in blocks; the
representative of a cluster is its medoid (computed over every member).  Reference interfaces (named
residue sets per side, typed or taken from the census of reference models) score every member on the
residues exclusive to each reference, on both sides, normalised by the size of each exclusive set.

Segment support (:func:`segment_support`): for each residue of the rebuilt segments, its heavy-atom
contacts beyond a sequence separation (counted within a chain; residues of other chains are never
neighbours) by source: the partner side, its own side outside the rebuilt segments (scaffold), base
pairs inside the rebuilt segments (optional, a class of its own) or its own rebuilt segments (self,
which is not support).

Numpy and scipy are used (:func:`mdatk.spatial.pairs_within`); planner steps run this module through
``python3 -m mdatk decoys ...``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .pdbtools import (PDBError, Structure, infer_base, parse_ranges, parse_residue_name, read_pdb,
                       sugar_chirality)

Range = Tuple[Optional[str], Optional[int], Optional[int]]  # (chain, None: any chain; first, last; None: all)
Key = Tuple[str, int, str]  # (chain, residue number, insertion code)

PAIR_ENGINES = ("builtin", "barnaba", "dssr", "off")
INTERFACE_ENGINES = ("builtin", "barnaba", "dssr")
LEVELS = ("residues", "blocks", "pairs")
SUPPORT_PAIRS = ("off", "contacts", "builtin")
COUNTS = ("contacts", "stack", "wc", "gu", "other", "bph", "ohph", "mgbr")

PURINE_RING = ("N9", "C8", "N7", "C5", "C4", "N3", "C2", "N1", "C6")
PYRIMIDINE_RING = ("N1", "C2", "N3", "C4", "C5", "C6")
PHOSPHATE_O = ("OP1", "OP2", "O1P", "O2P")
# Base atoms that donate a hydrogen (N-H), the C-H donors of Zirbel et al.'s base-phosphate classes, and
# base acceptors (lone pairs).  G N1 and U/T N3 carry a hydrogen; A N1 and C N3 do not.
NH_DONORS = {"A": ("N6",), "G": ("N1", "N2"), "C": ("N4",), "U": ("N3",), "T": ("N3",)}
CH_DONORS = {"A": ("C2", "C8"), "G": ("C8",), "C": ("C5", "C6"), "U": ("C5", "C6"), "T": ("C6",)}
ACCEPTORS = {"A": ("N1", "N3", "N7"), "G": ("O6", "N3", "N7"), "C": ("O2", "N3"), "U": ("O2", "O4"),
             "T": ("O2", "O4")}

_PLAIN_RANGE = re.compile(r"^(-?\d+)(?:-(-?\d+))?$")
_LABEL = re.compile(r"^([^:/\s]?):(-?\d+)([A-Za-z]?)$")
_FILTER = re.compile(r"^\s*([A-Za-z_][\w.:-]*)\s*(<=|>=|==|!=|<|>)\s*"
                     r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


class InterfaceError(ValueError):
    """A selection, table or option that cannot be used."""


# --- sides -------------------------------------------------------------------------------------------

def _g(group: Any, key: str) -> str:
    return (group.get(key) if isinstance(group, dict) else getattr(group, key, "")) or ""


def group_ranges(group: Any) -> List[Range]:
    """Residue ranges of a project residue group: its residues (plain ranges apply to every chain), or the
    selection MDATK writes for a chain group ('chainID A and resid 1:40', 'chainID B')."""
    from .selections import SelectionError, parse_residue_ranges
    residues = _g(group, "residues").strip()
    if residues:
        try:
            return [(c, a, b) for c, a, b in parse_residue_ranges(residues)]
        except SelectionError as exc:
            raise InterfaceError(f"group {_g(group, 'name')!r}: {exc}") from None
    sel = _g(group, "selection").strip()
    m = re.fullmatch(r"chainID\s+(\S)(?:\s+and\s+resid\s+(.+))?", sel)
    if m:
        if not m.group(2):
            return [(m.group(1), None, None)]
        out: List[Range] = []
        for tok in m.group(2).split():
            r = re.fullmatch(r"(-?\d+)(?::(-?\d+))?", tok)
            if not r:
                break
            out.append((m.group(1), int(r.group(1)), int(r.group(2) or r.group(1))))
        else:
            return out
    raise InterfaceError(f"group {_g(group, 'name')!r} has no residue ranges MDATK can use here "
                         f"(selection {sel!r}); give its residues as ranges")


def parse_side(spec: str, groups: Sequence[Any] = ()) -> List[Range]:
    """Residue ranges of a side: 'A:1-16,B:17-44', 'C', ':5-9', '1-40' (every chain), 'group:Name'."""
    out: List[Range] = []
    by_name = {_g(g, "name"): g for g in groups}
    for tok in re.split(r"[,;\s]+", (spec or "").strip()):
        if not tok:
            continue
        if tok.lower().startswith("group:"):
            name = tok[6:]
            if name not in by_name:
                known = ", ".join(sorted(by_name)) or "none defined"
                raise InterfaceError(f"unknown residue group {name!r} (groups: {known})")
            out += group_ranges(by_name[name])
            continue
        m = _PLAIN_RANGE.match(tok)
        if m and len(tok) > 1:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) is not None else a
            out.append((None, min(a, b), max(a, b)))
            continue
        try:
            out += [(c, a, b) for c, a, b in parse_ranges(tok)]
        except PDBError as exc:
            raise InterfaceError(str(exc)) from None
    return out


def format_side(ranges: Sequence[Range]) -> str:
    """Text of ranges that :func:`parse_side` reads back (groups expanded)."""
    out = []
    for chain, a, b in ranges:
        if chain is None:
            out.append(f"{a}-{b}" if a != b or 0 <= a <= 9 else f"{a}")  # '5' alone would be chain 5
        elif a is None:
            out.append(chain if chain else ":")
        else:
            out.append(f"{chain}:{a}" + (f"-{b}" if b != a else ""))
    return ",".join(out)


def in_side(chain: str, resseq: int, ranges: Sequence[Range]) -> bool:
    for c, a, b in ranges:
        if (c is None or c == chain) and (a is None or a <= resseq <= b):
            return True
    return False


def sides_overlap(a: Sequence[Range], b: Sequence[Range]) -> List[str]:
    """Ranges of side A that overlap side B (as text), judged from the ranges alone."""
    out = []
    for ca, a1, a2 in a:
        for cb, b1, b2 in b:
            if ca is not None and cb is not None and ca != cb:
                continue
            if a1 is None or b1 is None or (a1 <= b2 and b1 <= a2):
                out.append(f"{format_side([(ca, a1, a2)])} / {format_side([(cb, b1, b2)])}")
    return out


# --- nucleotides -------------------------------------------------------------------------------------

def _infer_base(atom_names: Iterable[str]) -> Optional[str]:
    """Base letter from the atoms of a residue whose name is not a known nucleotide name; None when the
    atoms do not make a complete base ring (a fragment is never guessed)."""
    names = set(atom_names)
    if not set(PYRIMIDINE_RING) <= names:
        return None
    return infer_base(names)


def _is_hydrogen(name: str, element: str) -> bool:
    if element:
        return element.strip().upper() in ("H", "D")
    letters = name.lstrip("0123456789")
    return letters[:1].upper() in ("H", "D")


@dataclass
class Nucleotide:
    """One nucleotide of a model: identity, base, hand and atoms (first alternate location)."""
    chain: str
    resseq: int
    icode: str
    name: str
    base: str  # A C G U T
    dna: bool
    hand: Optional[str]  # "D" / "L" from the sugar geometry; None when it cannot be called
    atoms: Dict[str, np.ndarray]
    heavy: List[str]  # heavy-atom names
    index: int = 0  # position among the model's nucleotides

    @property
    def key(self) -> Key:
        return (self.chain, self.resseq, self.icode)

    @property
    def label(self) -> str:
        return residue_label(self.key)

    @property
    def canonical(self) -> str:
        return ("D" + self.base) if self.dna else self.base


def residue_label(key: Key) -> str:
    chain, resseq, icode = key
    return f"{chain}:{resseq}{icode}"


def parse_label(text: str) -> Key:
    m = _LABEL.match(text.strip())
    if not m:
        raise InterfaceError(f"cannot read residue {text!r} (expected e.g. A:12)")
    return (m.group(1), int(m.group(2)), m.group(3))


def nucleotides(structure: Structure) -> List[Nucleotide]:
    """The nucleotides of a structure (see the module docstring for what counts)."""
    out: List[Nucleotide] = []
    for res in structure.residues():
        atoms: Dict[str, np.ndarray] = {}
        heavy: List[str] = []
        for a in res.atoms:
            name = a.name.replace("*", "'")
            if name in atoms:
                continue  # an alternate location
            atoms[name] = np.array(a.xyz, dtype=np.float64)
            if not _is_hydrogen(name, a.element):
                heavy.append(name)
        if len(atoms) < 2:
            continue  # ions and other single atoms
        info = parse_residue_name(res.name)
        backbone = any("'" in n or n == "P" for n in atoms)
        if info is not None and backbone:
            base, dna = info.base, info.sugar == "dna"
        elif "C1'" in atoms:
            base = _infer_base(atoms)
            if base is None:
                continue
            dna = base == "T" or "O2'" not in atoms
        else:
            continue
        hand, _vols = sugar_chirality({n: tuple(v) for n, v in atoms.items()}, base)
        out.append(Nucleotide(res.chain, res.resseq, res.icode, res.name, base, dna, hand, atoms, heavy,
                              len(out)))
    return out


def split_sides(nts: Sequence[Nucleotide], side_a: Sequence[Range], side_b: Optional[Sequence[Range]],
                require_b: bool = True) -> Tuple[List[Nucleotide], List[Nucleotide]]:
    """The nucleotides of side A and side B (None: every nucleotide not in A); error on overlap or an empty
    side (side B may be empty without ``require_b``: a model with nothing but side A)."""
    a = [n for n in nts if in_side(n.chain, n.resseq, side_a)]
    keys_a = {n.key for n in a}
    if side_b is None:
        b = [n for n in nts if n.key not in keys_a]
    else:
        b = [n for n in nts if in_side(n.chain, n.resseq, side_b)]
        both = [n.label for n in b if n.key in keys_a]
        if both:
            raise InterfaceError(f"the sides overlap at {', '.join(both[:6])}"
                                 + (f" and {len(both) - 6} more" if len(both) > 6 else ""))
    if not a:
        raise InterfaceError("side A matches no nucleotide of the model")
    if not b and (require_b or side_b is not None):
        raise InterfaceError("side B matches no nucleotide of the model")
    return a, b


def compress_labels(keys: Iterable[Key]) -> str:
    """'A:8-9,B:25-28' from residue keys (residues with insertion codes are listed singly)."""
    parts: List[List] = []
    for chain, resseq, icode in sorted(set(keys), key=lambda k: (k[0], k[1], k[2])):
        if not icode and parts and parts[-1][0] == chain and not parts[-1][3] and resseq == parts[-1][2] + 1:
            parts[-1][2] = resseq
        else:
            parts.append([chain, resseq, resseq, icode])
    return ",".join(f"{c}:{a}{i}" if a == b else f"{c}:{a}-{b}" for c, a, b, i in parts)


# --- geometry helpers ---------------------------------------------------------------------------------

def _atom_rows(nts: Sequence[Nucleotide], names_of) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Coordinates, owner (index into ``nts``) and atom name of the atoms ``names_of(n)`` selects."""
    xyz, owner, names = [], [], []
    for k, n in enumerate(nts):
        for name in names_of(n):
            if name in n.atoms:
                xyz.append(n.atoms[name])
                owner.append(k)
                names.append(name)
    return (np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(owner, dtype=np.int64), names)


def _within(xa: np.ndarray, xb: np.ndarray, cutoff: float, strict: bool = True):
    from .spatial import pairs_within
    i, j, d = pairs_within(xa, xb, cutoff)
    if strict:
        keep = d < cutoff
        i, j, d = i[keep], j[keep], d[keep]
    return i, j, d


def ring_plane(n: Nucleotide) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(centroid, unit normal) of the base ring (least-squares plane, as mdatk.analysis.stacking), or None
    with fewer than four ring atoms."""
    ring = PURINE_RING if n.base in ("A", "G") else PYRIMIDINE_RING
    pts = np.array([n.atoms[a] for a in ring if a in n.atoms])
    if len(pts) < 4:
        return None
    c = pts.mean(axis=0)
    return c, np.linalg.svd(pts - c)[2][2]


def stacked(pa, pb, distance: float = 4.5, angle: float = 30.0, separation: float = 2.5) -> bool:
    """The stacking test of the census on two ring planes (centroid, normal)."""
    off = pa[0] - pb[0]
    if float(np.linalg.norm(off)) >= distance:
        return False
    dot = float(np.dot(pa[1], pb[1]))
    ang = math.degrees(math.acos(min(1.0, abs(dot))))
    mean = pa[1] + (pb[1] if dot > 0 else -pb[1])
    norm = float(np.linalg.norm(mean))
    along = abs(float(np.dot(off, mean / norm))) if norm > 1e-6 else 0.0
    return ang < angle and along >= separation


def _stacks(a: Sequence[Nucleotide], b: Sequence[Nucleotide], p) -> List[Tuple[int, int]]:
    planes_a = [ring_plane(n) for n in a]
    planes_b = [ring_plane(n) for n in b]
    ia = [k for k, pl in enumerate(planes_a) if pl is not None]
    ib = [k for k, pl in enumerate(planes_b) if pl is not None]
    if not ia or not ib:
        return []
    ca = np.array([planes_a[k][0] for k in ia])
    cb = np.array([planes_b[k][0] for k in ib])
    i, j, _d = _within(ca, cb, p.stack_distance)
    out = []
    for x, y in zip(i, j):
        if stacked(planes_a[ia[x]], planes_b[ib[y]], p.stack_distance, p.stack_angle, p.stack_separation):
            out.append((ia[x], ib[y]))
    return sorted(out)


def _hbond_contacts(don: Sequence[Nucleotide], acc: Sequence[Nucleotide], donors_of, cutoff: float):
    """(donor residue index, donor atom, acceptor residue index, phosphate O, distance) across two lists."""
    xd, od, nd = _atom_rows(don, donors_of)
    xa, oa, na = _atom_rows(acc, lambda n: PHOSPHATE_O)
    if not len(xd) or not len(xa):
        return []
    i, j, d = _within(xd, xa, cutoff)
    return [(int(od[x]), nd[x], int(oa[y]), na[y], float(dist)) for x, y, dist in zip(i, j, d)]


def _hydroxyl(n: Nucleotide) -> Tuple[str, ...]:
    return ("O2'",)


def _base_donors(ch: bool):
    return lambda n: NH_DONORS.get(n.base, ()) + (CH_DONORS.get(n.base, ()) if ch else ())


# --- base-pair engines -------------------------------------------------------------------------------

def engine_pairs(nts: Sequence[Nucleotide], engine: str, wobble: bool = True, cutoff: float = 3.5,
                 dssr_exe: str = "x3dna-dssr", want_stacks: bool = False, keep: str = "", stem: str = ""):
    """Base pairs among ``nts`` from an engine -> (pairs [(i, j, class)], stacks [(i, j, class)] or None,
    DSSR result or None).  Classes as REMARK 950 writes them: 'WCc', 'GUc', other Leontis-Westhof codes."""
    from .secstruct import PairDetector
    if engine == "builtin":
        names = [n.base for n in nts]
        index, rows = [], []
        for n in nts:
            index.append({a: len(rows) + k for k, a in enumerate(n.heavy)})
            rows += [n.atoms[a] for a in n.heavy]
        xyz = np.array(rows, dtype=np.float64).reshape(-1, 3)
        found = PairDetector(names, index, cutoff=cutoff, wobble=wobble).detect(xyz)
        code = {1: "WCc", 2: "GUc"}
        return [(int(i), int(j), code[int(k)]) for i, j, k in found], None, None
    if engine == "barnaba":
        from .secstruct import BarnabaRunner, import_barnaba
        import_barnaba()
        rows, atoms = [], []
        for n in nts:
            atoms.append([(a, len(rows) + k) for k, a in enumerate(n.heavy)])
            rows += [n.atoms[a] for a in n.heavy]
        frame = BarnabaRunner([n.canonical for n in nts], atoms).annotate(
            [np.array(rows).reshape(-1, 3)], stacking=want_stacks)[0]
        pairs = [(i, j, c) for i, j, c in frame.pairs if wobble or c != "GUc"]
        return pairs, (list(frame.stacks) if want_stacks else None), None
    if engine == "dssr":
        from .dssr import DssrRunner, lw_code
        runner = DssrRunner([n.canonical for n in nts], [[(a, n.atoms[a]) for a in n.heavy] for n in nts],
                            exe=dssr_exe)
        res = runner.run(keep=keep, stem=stem or "model")
        pairs = [(i, j, lw_code(lw, name)) for i, j, lw, name in res.pairs]
        pairs = [(i, j, c) for i, j, c in pairs if wobble or c != "GUc"]
        return pairs, [(i, j, "") for i, j in res.stacks], res
    raise InterfaceError(f"unknown base-pair engine {engine!r} (choose from {', '.join(PAIR_ENGINES)})")


def pair_kind(code: str) -> str:
    return "wc" if code == "WCc" else "gu" if code == "GUc" else "other"


# --- census ------------------------------------------------------------------------------------------

@dataclass
class CensusParams:
    contact: float = 4.5  # heavy-atom contact cut-off (Å)
    hbond: float = 3.5  # donor-acceptor cut-off of the base-phosphate and 2'-OH-phosphate contacts (Å)
    stack_distance: float = 4.5
    stack_angle: float = 30.0
    stack_separation: float = 2.5
    mg_window: Tuple[float, float] = (5.0, 7.0)
    pairs: str = "builtin"  # builtin | barnaba | dssr | off
    pair_cutoff: float = 3.5  # builtin engine
    wobble: bool = True
    bph_ch: bool = False
    dssr_exe: str = "x3dna-dssr"


@dataclass
class CensusRow:
    model: str
    pdb: str
    counts: Dict[str, Optional[int]]
    residues_a: List[Key]
    residues_b: List[Key]
    fingerprint: List[Tuple[Key, Key]]
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # score-table terms

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"model": self.model, "pdb": self.pdb}
        out.update(self.extra)
        for k in COUNTS:
            v = self.counts.get(k)
            out[k] = "" if v is None else v
        out["n_a"] = len(self.residues_a)
        out["n_b"] = len(self.residues_b)
        out["residues_a"] = compress_labels(self.residues_a)
        out["residues_b"] = compress_labels(self.residues_b)
        out["fingerprint"] = " ".join(f"{residue_label(a)}/{residue_label(b)}" for a, b in self.fingerprint)
        return out


def census(structure: Structure, side_a: Sequence[Range], side_b: Optional[Sequence[Range]],
           params: Optional[CensusParams] = None, model: str = "", pdb: str = "") -> CensusRow:
    """The interface census of one model (see the module docstring)."""
    p = params or CensusParams()
    nts = nucleotides(structure)
    a, b = split_sides(nts, side_a, side_b)
    counts: Dict[str, Optional[int]] = {}
    notes: List[str] = []
    # heavy-atom contacts and the fingerprint
    xa, oa, _na = _atom_rows(a, lambda n: n.heavy)
    xb, ob, _nb = _atom_rows(b, lambda n: n.heavy)
    i, j, _d = _within(xa, xb, p.contact)
    counts["contacts"] = int(len(i))
    pairs = sorted({(int(oa[x]), int(ob[y])) for x, y in zip(i, j)})
    fingerprint = [(a[x].key, b[y].key) for x, y in pairs]
    res_a = sorted({a[x].key for x, _ in pairs})
    res_b = sorted({b[y].key for _, y in pairs})
    counts["stack"] = len(_stacks(a, b, p))
    # base pairs across the boundary (each base pairs once, so intra-side pairs compete)
    if p.pairs == "off":
        counts["wc"] = counts["gu"] = counts["other"] = None
    else:
        both = list(a) + list(b)
        found, _stacks_e, dres = engine_pairs(both, p.pairs, p.wobble, p.pair_cutoff, p.dssr_exe)
        if dres is not None:
            notes += dres.notes
        cross = [pair_kind(c) for x, y, c in found if (x < len(a)) != (y < len(a))]
        counts["wc"], counts["gu"] = cross.count("wc"), cross.count("gu")
        counts["other"] = cross.count("other") if p.pairs in ("barnaba", "dssr") else None
    donors = _base_donors(p.bph_ch)
    counts["bph"] = (len(_hbond_contacts(a, b, donors, p.hbond))
                     + len(_hbond_contacts(b, a, donors, p.hbond)))
    counts["ohph"] = (len(_hbond_contacts(a, b, _hydroxyl, p.hbond))
                      + len(_hbond_contacts(b, a, _hydroxyl, p.hbond)))
    pa, _o1, _n1 = _atom_rows(a, lambda n: ("P",))
    pb, _o2, _n2 = _atom_rows(b, lambda n: ("P",))
    lo, hi = p.mg_window
    _i, _j, d = _within(pa, pb, hi, strict=False)
    counts["mgbr"] = int(np.count_nonzero(d >= lo)) if len(d) else 0
    return CensusRow(model or "model", pdb, counts, res_a, res_b, fingerprint, notes)


# --- the decoy table ---------------------------------------------------------------------------------

def _delimiter(path: str, header: str) -> Optional[str]:
    if path.lower().endswith(".csv"):
        return ","
    if "\t" in header:
        return "\t"
    if "," in header:
        return ","
    return None  # whitespace


@dataclass
class DecoyTable:
    """A table of decoys and their Rosetta score terms (the 'scores' output of the FARFAR2 models step):
    columns 'run', 'tag', 'pdb' (relative to the plan folder; empty when the decoy was not extracted) and
    every SCORE term by its name.  Extra columns are kept; 'description' stands in for a missing 'tag'."""
    path: str
    columns: List[str]
    rows: List[Dict[str, str]]

    def has(self, term: str) -> bool:
        return term in self.columns

    @staticmethod
    def number(row: Dict[str, str], term: str) -> Optional[float]:
        try:
            value = float(row.get(term, ""))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def resolve(self, row: Dict[str, str], plan_root: str = "") -> str:
        """Path of a row's PDB ('' when none), relative paths taken from ``plan_root``."""
        path = (row.get("pdb") or "").strip()
        if not path or os.path.isabs(path) or not plan_root:
            return path
        return os.path.normpath(os.path.join(plan_root, path))

    def matcher(self, plan_root: str = ""):
        """A function model path -> row (or None): by the resolved 'pdb' path, else the file name, else
        the tag equal to the file name without '.pdb' (only when unique across runs)."""
        by_path: Dict[str, Optional[Dict[str, str]]] = {}
        by_base: Dict[str, Optional[Dict[str, str]]] = {}
        by_tag: Dict[str, Optional[Dict[str, str]]] = {}

        def put(index, key, row):
            index[key] = None if key in index and index[key] is not row else row  # ambiguous -> None

        for row in self.rows:
            path = self.resolve(row, plan_root)
            if path:
                put(by_path, os.path.realpath(path), row)
                put(by_base, os.path.basename(path), row)
            tag = (row.get("tag") or "").strip()
            if tag:
                put(by_tag, tag, row)

        def match(model_path: str) -> Optional[Dict[str, str]]:
            base = os.path.basename(model_path)
            for index, key in ((by_path, os.path.realpath(model_path)), (by_base, base),
                               (by_tag, os.path.splitext(base)[0])):
                if key in index:
                    return index[key]
            return None
        return match


def read_decoy_table(path: str) -> DecoyTable:
    """Read a decoy table (TSV, CSV or whitespace-separated; '#' lines are comments)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise InterfaceError(f"{path}: empty table")
    delim = _delimiter(path, lines[0])
    if delim is None:
        rows_raw = [ln.split() for ln in lines]
    else:
        rows_raw = list(csv.reader(lines, delimiter=delim))
    columns = [c.strip() for c in rows_raw[0]]
    if columns and columns[0] == "SCORE:":
        columns = columns[1:]
        rows_raw = [r[1:] if r and r[0] == "SCORE:" else r for r in rows_raw]
    if "tag" not in columns and "description" in columns:
        columns = ["tag" if c == "description" else c for c in columns]
    if "tag" not in columns and "pdb" not in columns:
        raise InterfaceError(f"{path}: the table needs a 'tag' or 'pdb' column "
                             f"(columns: {', '.join(columns)})")
    rows = []
    for raw in rows_raw[1:]:
        if raw == rows_raw[0]:
            continue  # a repeated header
        rows.append({c: (raw[k].strip() if k < len(raw) else "") for k, c in enumerate(columns)})
    return DecoyTable(path, columns, rows)


def parse_filters(text: Any) -> List[Tuple[str, str, float]]:
    """'linear_chainbreak <= 5' (one per line, or separated by ',' / ';') -> [(term, op, value)]."""
    items = text if isinstance(text, (list, tuple)) else re.split(r"[\n,;]+", text or "")
    out = []
    for item in items:
        item = str(item).split("#", 1)[0].strip()
        if not item:
            continue
        m = _FILTER.match(item)
        if not m:
            raise InterfaceError(f"cannot read the filter {item!r} (expected e.g. 'linear_chainbreak <= 5')")
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def format_filter(f: Tuple[str, str, float]) -> str:
    return f"{f[0]}{f[1]}{f[2]:g}"


@dataclass
class FilterOutcome:
    kept: List[str]
    dropped: List[str]
    unscored: List[str]
    rows: Dict[str, Optional[Dict[str, str]]]


def apply_filters(models: Sequence[str], table: Optional[DecoyTable],
                  filters: Sequence[Tuple[str, str, float]], plan_root: str = "",
                  drop_unscored: bool = False) -> FilterOutcome:
    """Keep the models whose table row passes every filter.  A model without a row (or without a value
    for a filtered term) is kept and listed as unscored, unless ``drop_unscored``."""
    if table is None:
        if filters:
            raise InterfaceError("filters need a score table (--scores)")
        return FilterOutcome(list(models), [], [], {m: None for m in models})
    missing = [t for t, _o, _v in filters if not table.has(t)]
    if missing:
        raise InterfaceError(f"{table.path}: no column {', '.join(repr(m) for m in missing)} "
                             f"(columns: {', '.join(table.columns)})")
    match = table.matcher(plan_root)
    kept, dropped, unscored, rows = [], [], [], {}
    for m in models:
        row = match(m)
        rows[m] = row
        values = [table.number(row, t) if row else None for t, _o, _v in filters]
        if row is None or any(v is None for v in values):
            unscored.append(m)
            (dropped if drop_unscored else kept).append(m)
        elif all(_OPS[o](v, x) for (_t, o, x), v in zip(filters, values)):
            kept.append(m)
        else:
            dropped.append(m)
    return FilterOutcome(kept, dropped, unscored, rows)


def guess_plan_root(table: DecoyTable) -> str:
    """The folder a table's pdb paths are relative to, when not given: the plan folder is the table's
    folder (a plan without step folders) or its parent (the table in a step folder); the first in which a
    listed PDB exists, else the table's folder."""
    here = os.path.dirname(os.path.abspath(table.path))
    paths = [(r.get("pdb") or "").strip() for r in table.rows]
    sample = [p for p in paths if p and not os.path.isabs(p)][:20]
    for root in (here, os.path.dirname(here), os.getcwd()):
        if any(os.path.isfile(os.path.join(root, p)) for p in sample):
            return root
    return here


def models_from_table(table: DecoyTable, plan_root: str = "") -> List[str]:
    """The extracted decoys of a table (rows with a 'pdb' path), in table order."""
    return [p for p in (table.resolve(r, plan_root) for r in table.rows) if p]


# --- tables ------------------------------------------------------------------------------------------

def write_table(path: str, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    """Write rows as TSV ('-' or '': standard output)."""
    cols = list(columns) if columns is not None else (list(rows[0]) if rows else [])
    out = sys.stdout if path in ("", "-") else open(path, "w", encoding="utf-8", newline="")
    try:
        w = csv.DictWriter(out, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
    finally:
        if out is not sys.stdout:
            out.close()


def read_table(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        text = fh.read()
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        return []
    delim = _delimiter(path, lines[0]) or "\t"
    return list(csv.DictReader(lines, delimiter=delim))


def _int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# --- clustering --------------------------------------------------------------------------------------

@dataclass
class CensusEntry:
    """A census row read back for clustering."""
    model: str
    pdb: str
    pairs: List[Tuple[Key, Key]]
    counts: Dict[str, Optional[int]]

    @property
    def residues_a(self) -> Set[Key]:
        return {a for a, _ in self.pairs}

    @property
    def residues_b(self) -> Set[Key]:
        return {b for _, b in self.pairs}


def parse_fingerprint(text: str) -> List[Tuple[Key, Key]]:
    out = []
    for tok in (text or "").split():
        left, sep, right = tok.partition("/")
        if not sep:
            raise InterfaceError(f"cannot read the fingerprint item {tok!r} (expected e.g. A:8/C:60)")
        out.append((parse_label(left), parse_label(right)))
    return out


def read_census(path: str) -> List[CensusEntry]:
    rows = read_table(path)
    if rows and "fingerprint" not in rows[0]:
        hint = (" (a table of the earlier interface script: its 'fp' column has no chain ids; write the "
                "table again with 'mdatk decoys census')") if "fp" in rows[0] else ""
        raise InterfaceError(f"{path}: no 'fingerprint' column{hint}")
    out = []
    for r in rows:
        out.append(CensusEntry(r.get("model") or os.path.basename(r.get("pdb", "")), r.get("pdb", ""),
                               parse_fingerprint(r.get("fingerprint", "")),
                               {k: _int(r.get(k)) for k in COUNTS}))
    return out


def features(pairs: Sequence[Tuple[Key, Key]], level: str = "residues", block: int = 4) -> FrozenSet[str]:
    """A fingerprint at a granularity: residue pairs, the residues of each side, or residues binned into
    blocks of ``block`` consecutive numbers per chain (fixed bins: a shift across a bin edge still
    separates two fingerprints)."""
    if level == "pairs":
        return frozenset(f"{residue_label(a)}/{residue_label(b)}" for a, b in pairs)
    out = set()
    for a, b in pairs:
        for side, (chain, resseq, icode) in (("a", a), ("b", b)):
            if level == "residues":
                out.add(f"{side}|{chain}:{resseq}{icode}")
            elif level == "blocks":
                out.add(f"{side}|{chain}:{resseq // block}")
            else:
                raise InterfaceError(f"unknown level {level!r} (choose from {', '.join(LEVELS)})")
    return frozenset(out)


def _matrix(feats: Sequence[FrozenSet[str]]) -> np.ndarray:
    vocab = {f: k for k, f in enumerate(sorted(set().union(*feats)))} if feats else {}
    m = np.zeros((len(feats), max(1, len(vocab))), dtype=np.float32)
    for r, fs in enumerate(feats):
        for f in fs:
            m[r, vocab[f]] = 1.0
    return m


def _jaccard_to(m: np.ndarray, sizes: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    inter = m[rows] @ m[k]
    union = sizes[rows] + sizes[k] - inter
    return 1.0 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def leader_clusters(feats: Sequence[FrozenSet[str]], names: Sequence[str], cut: float) -> List[List[int]]:
    """Leader clustering: in order of fingerprint size (largest first; ties by name, so the input order
    does not matter) every unclaimed fingerprint seeds a cluster that takes every unclaimed one within
    ``cut`` Jaccard distance.  Members are listed seed first, then in that order."""
    order = sorted(range(len(feats)), key=lambda k: (-len(feats[k]), names[k], k))
    m = _matrix(feats)
    sizes = m.sum(axis=1)
    claimed = np.zeros(len(feats), dtype=bool)
    rank = np.array(order, dtype=np.int64)
    clusters = []
    for k in order:
        if claimed[k]:
            continue
        rest = rank[~claimed[rank]]
        d = _jaccard_to(m, sizes, rest, k)
        members = [int(x) for x in rest[d <= cut + 1e-9]]
        members = [k] + [x for x in members if x != k]
        claimed[members] = True
        clusters.append(members)
    return clusters


def medoid(feats: Sequence[FrozenSet[str]], members: Sequence[int]) -> int:
    """The member with the smallest summed Jaccard distance to all other members (exact; ties go to the
    earliest member, i.e. the seed)."""
    if len(members) < 3:
        return members[0]
    m = _matrix([feats[k] for k in members])
    sizes = m.sum(axis=1)
    total = np.zeros(len(members))
    step = 512
    for s in range(0, len(members), step):
        inter = m[s:s + step] @ m.T
        union = sizes[s:s + step, None] + sizes[None, :] - inter
        total[s:s + step] = (1.0 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)).sum(1)
    return members[int(np.argmin(total + np.arange(len(members)) * 1e-12))]


@dataclass
class Reference:
    """A reference interface: residues per side."""
    name: str
    a: Set[Key]
    b: Set[Key]


def parse_reference(text: str, groups: Sequence[Any] = (), seen: Sequence[CensusEntry] = ()) -> Reference:
    """'NAME=A_RANGES/B_RANGES' -> Reference.  Ranges without residue numbers or chains are expanded with
    the residues the census saw on that side."""
    name, sep, rest = text.partition("=")
    if not sep or not name.strip():
        raise InterfaceError(f"cannot read the reference {text!r} (expected NAME=A-ranges/B-ranges)")
    left, _, right = rest.partition("/")
    return reference_from_ranges(name.strip(), parse_side(left, groups), parse_side(right, groups), seen)


def reference_from_ranges(name: str, ra: Sequence[Range], rb: Sequence[Range],
                          seen: Sequence[CensusEntry] = ()) -> Reference:
    seen_a = set().union(*[e.residues_a for e in seen]) if seen else set()
    seen_b = set().union(*[e.residues_b for e in seen]) if seen else set()

    def expand(ranges, observed) -> Set[Key]:
        out: Set[Key] = set()
        for chain, a, b in ranges:
            if chain is not None and a is not None:
                out |= {(chain, r, "") for r in range(a, b + 1)}
            out |= {k for k in observed if in_side(k[0], k[1], [(chain, a, b)])}
        return out
    return Reference(name, expand(ra, seen_a), expand(rb, seen_b))


def references_from_census(entries: Sequence[CensusEntry]) -> List[Reference]:
    """One reference per census row (e.g. the census of reference models), named after its model."""
    out = []
    for e in entries:
        name = os.path.splitext(e.model)[0]
        out.append(Reference(name, e.residues_a, e.residues_b))
    return out


@dataclass
class ClusterParams:
    level: str = "residues"
    block: int = 4
    cut: float = 0.5
    min_size: int = 5
    broad_limit: int = 10  # a cluster whose representative touches more side-B residues is 'broad' (0: off)
    lead_gap: float = 0.15
    dispersed_below: float = 0.2
    few_clusters: int = 6


@dataclass
class Cluster:
    members: List[int]  # indices into the census entries, representative first
    shares: Dict[str, Optional[float]] = field(default_factory=dict)
    touched: Dict[str, float] = field(default_factory=dict)
    broad: bool = False

    @property
    def representative(self) -> int:
        return self.members[0]


@dataclass
class ClusterResult:
    entries: List[CensusEntry]
    clusters: List[Cluster]
    empty: List[int]  # entries without any contact (not clustered)
    references: List[Reference]
    exclusive: Dict[str, Tuple[Set[Key], Set[Key]]]
    shared: Tuple[Set[Key], Set[Key]]
    params: ClusterParams
    summary: Dict[str, Any] = field(default_factory=dict)


def _exclusive(refs: Sequence[Reference]):
    excl = {}
    for r in refs:
        others = [o for o in refs if o is not r]
        oa = set().union(*[o.a for o in others]) if others else set()
        ob = set().union(*[o.b for o in others]) if others else set()
        excl[r.name] = (r.a - oa, r.b - ob)
    shared = ((set.intersection(*[r.a for r in refs]), set.intersection(*[r.b for r in refs]))
              if len(refs) > 1 else (set(), set()))
    return excl, shared


def touched_fraction(entry: CensusEntry, exclusive: Tuple[Set[Key], Set[Key]]) -> float:
    """The share of a reference's exclusive residues (both sides) that an entry contacts."""
    ea, eb = exclusive
    size = len(ea) + len(eb)
    if not size:
        return 0.0
    return (len(entry.residues_a & ea) + len(entry.residues_b & eb)) / size


def cluster_census(entries: Sequence[CensusEntry], params: Optional[ClusterParams] = None,
                   references: Sequence[Reference] = ()) -> ClusterResult:
    """Cluster census entries by Jaccard distance of their fingerprints (see the module docstring)."""
    p = params or ClusterParams()
    if p.level not in LEVELS:
        raise InterfaceError(f"unknown level {p.level!r} (choose from {', '.join(LEVELS)})")
    entries = list(entries)
    names = [r.name for r in references]
    if len(set(names)) != len(names):
        raise InterfaceError("reference names must be unique")
    feats = [features(e.pairs, p.level, p.block) for e in entries]
    live = [k for k, f in enumerate(feats) if f]
    empty = [k for k, f in enumerate(feats) if not f]
    local = leader_clusters([feats[k] for k in live], [entries[k].model for k in live], p.cut)
    clusters = []
    for members in local:
        members = [live[x] for x in members]
        rep = medoid(feats, members)
        clusters.append(Cluster([rep] + [x for x in members if x != rep]))
    clusters.sort(key=lambda c: (-len(c.members), entries[c.representative].model))
    excl, shared = _exclusive(list(references))
    for c in clusters:
        rep = entries[c.representative]
        c.broad = bool(p.broad_limit) and len(rep.residues_b) > p.broad_limit
        if references:
            # the mean share of each reference's exclusive residues touched by the members, normalised
            # per reference so that a larger exclusive set does not win by size
            for r in references:
                c.touched[r.name] = float(np.mean([touched_fraction(entries[k], excl[r.name])
                                                   for k in c.members]))
            total = sum(c.touched.values())
            c.shares = {n: (v / total if total > 0 else None) for n, v in c.touched.items()}
    result = ClusterResult(entries, clusters, empty, list(references), excl, shared, p)
    result.summary = _cluster_summary(result)
    return result


def _cluster_summary(res: ClusterResult) -> Dict[str, Any]:
    p = res.params
    n_live = len(res.entries) - len(res.empty)
    big = [c for c in res.clusters if len(c.members) >= p.min_size]
    covered = sum(len(c.members) for c in big)
    coverage = covered / n_live if n_live else 0.0
    out: Dict[str, Any] = {
        "models": len(res.entries), "with_interface": n_live, "without_interface": len(res.empty),
        "clusters": len(res.clusters), "clusters_min_size": len(big), "covered": covered,
        "coverage": round(coverage, 4),
        "singletons": sum(1 for c in res.clusters if len(c.members) == 1),
    }
    if not n_live:
        out["pattern"] = "no interface"
    elif coverage < p.dispersed_below:
        out["pattern"] = (f"dispersed: less than {p.dispersed_below:g} of the models are in clusters of "
                          f">= {p.min_size}")
    elif len(big) <= p.few_clusters:
        out["pattern"] = f"few recurring interfaces: {len(big)} cluster(s) of >= {p.min_size}"
    else:
        out["pattern"] = (f"several recurring interfaces: {len(big)} clusters of >= {p.min_size}, "
                          "none dominant")
    if res.references:
        for label, chosen in (("all", big), ("focused", [c for c in big if not c.broad])):
            weight = sum(len(c.members) for c in chosen if any(v is not None for v in c.shares.values()))
            shares = {}
            for r in res.references:
                s = sum(len(c.members) * (c.shares.get(r.name) or 0.0) for c in chosen)
                shares[r.name] = round(s / weight, 4) if weight else None
            out[f"shares_{label}"] = shares
        shares = out["shares_all"]
        known = sorted(((v, k) for k, v in shares.items() if v is not None), reverse=True)
        if known:
            gap = known[0][0] - (known[1][0] if len(known) > 1 else 0.0)
            out["lead"] = known[0][1] if gap > p.lead_gap else ""
            out["lead_gap"] = round(gap, 4)
    return out


def cluster_rows(res: ClusterResult, rep_files: Dict[int, str]) -> List[Dict[str, Any]]:
    """One row per cluster for the clusters table."""
    out = []
    for k, c in enumerate(res.clusters, start=1):
        rep = res.entries[c.representative]
        row: Dict[str, Any] = {"cluster": k, "size": len(c.members), "representative": rep.model,
                               "representative_pdb": rep_files.get(k, ""), "source_pdb": rep.pdb}
        for key in COUNTS:
            vals = [res.entries[x].counts.get(key) for x in c.members]
            vals = [v for v in vals if v is not None]
            row[f"mean_{key}"] = round(float(np.mean(vals)), 2) if vals else ""
        row.update(n_a=len(rep.residues_a), n_b=len(rep.residues_b),
                   residues_a=compress_labels(rep.residues_a), residues_b=compress_labels(rep.residues_b),
                   broad="yes" if c.broad else "no")
        for r in res.references:
            share = c.shares.get(r.name)
            row[f"share_{r.name}"] = "" if share is None else round(share, 4)
            row[f"touched_{r.name}"] = round(c.touched.get(r.name, 0.0), 4)
        row["members"] = " ".join(res.entries[x].model for x in c.members)
        out.append(row)
    return out


def write_representatives(res: ClusterResult, template: str, n: int, base_dirs: Sequence[str] = ("",)
                          ) -> Tuple[Dict[int, str], List[str]]:
    """Copy the representatives of the ``n`` largest clusters to ``template`` ('{n}' -> 01, 02 ...)."""
    files: Dict[int, str] = {}
    notes: List[str] = []
    for k, c in enumerate(res.clusters[:n], start=1):
        src = res.entries[c.representative].pdb
        found = next((os.path.join(d, src) for d in base_dirs
                      if src and os.path.isfile(os.path.join(d, src))), "")
        if not found:
            notes.append(f"cluster {k}: the representative's PDB {src or '(no path)'} was not found")
            continue
        dst = template.replace("{n}", f"{k:02d}")
        if os.path.abspath(found) != os.path.abspath(dst):
            shutil.copyfile(found, dst)
        files[k] = dst
    if len(res.clusters) < n:
        notes.append(f"{len(res.clusters)} cluster(s) for {n} representatives: the later files are not "
                     "written")
    return files, notes


# --- segment support ---------------------------------------------------------------------------------

@dataclass
class SupportParams:
    sep: int = 3  # contacts to residues of the same chain at most this far in sequence are ignored
    cutoff: float = 4.5
    count_self: bool = False
    pairs: str = "off"  # off | contacts | builtin
    pair_cutoff: float = 3.3  # 'contacts': donor-acceptor distance of the base-edge hydrogen bonds


@dataclass
class ResidueSupport:
    key: Key
    name: str
    segment: int
    partner: int = 0
    scaffold: int = 0
    pair: int = 0
    self_: int = 0

    def supported(self, count_self: bool = False) -> bool:
        return bool(self.partner or self.scaffold or self.pair or (count_self and self.self_))

    @property
    def mark(self) -> str:
        """'#' partner, 'o' scaffold, '=' base pair inside the rebuilt segments, ':' only self, '.' none."""
        for mark, count in (("#", self.partner), ("o", self.scaffold), ("=", self.pair), (":", self.self_)):
            if count:
                return mark
        return "."


@dataclass
class SupportResult:
    model: str
    residues: List[ResidueSupport]
    segments: List[List[int]]  # indices into residues

    def longest_run(self, count_self: bool = False) -> int:
        best = 0
        for seg in self.segments:
            cur = 0
            for k in seg:
                cur = 0 if self.residues[k].supported(count_self) else cur + 1
                best = max(best, cur)
        return best


def _base_paired(n: Nucleotide, q: Nucleotide, cutoff: float) -> bool:
    """Two bases held by at least two distinct base-edge hydrogen bonds (a donor of one and an acceptor of
    the other within ``cutoff``; a bifurcated donor or acceptor counts once): a pair of any
    Leontis-Westhof family."""
    bonds = []
    for d_res, a_res in ((n, q), (q, n)):
        for d in NH_DONORS.get(d_res.base, ()):
            if d not in d_res.atoms:
                continue
            for acc in ACCEPTORS.get(a_res.base, ()):
                if acc in a_res.atoms and float(np.linalg.norm(d_res.atoms[d] - a_res.atoms[acc])) <= cutoff:
                    bonds.append(((d_res.key, d), (a_res.key, acc)))
    return any(x[0] != y[0] and x[1] != y[1] for k, x in enumerate(bonds) for y in bonds[k + 1:])


def segment_support(structure: Structure, side_a: Sequence[Range], side_b: Optional[Sequence[Range]],
                    rebuilt: Sequence[Range], params: Optional[SupportParams] = None,
                    model: str = "") -> SupportResult:
    """Contacts of every rebuilt residue by source (see the module docstring)."""
    p = params or SupportParams()
    nts = nucleotides(structure)
    a, b = split_sides(nts, side_a, side_b, require_b=False)  # one molecule: nothing is partner
    mol = {n.key: "a" for n in a}
    mol.update({n.key: "b" for n in b})
    both = [n for n in nts if n.key in mol]
    is_rebuilt = {n.key: in_side(n.chain, n.resseq, rebuilt) for n in both}
    targets = [k for k, n in enumerate(both) if is_rebuilt[n.key]]
    if not targets:
        raise InterfaceError("no rebuilt residue of the sides is in the model")
    paired: Set[Tuple[Key, Key]] = set()
    if p.pairs == "builtin":
        found, _s, _r = engine_pairs(both, "builtin", True, 3.5)
        paired = {(both[i].key, both[j].key) for i, j, _c in found} | {(both[j].key, both[i].key)
                                                                       for i, j, _c in found}
    elif p.pairs not in SUPPORT_PAIRS:
        raise InterfaceError(f"unknown pair credit {p.pairs!r} (choose from {', '.join(SUPPORT_PAIRS)})")
    xyz, owner, _names = _atom_rows(both, lambda n: n.heavy)
    tx, tow, _tn = _atom_rows([both[k] for k in targets], lambda n: n.heavy)
    i, j, _d = _within(tx, xyz, p.cutoff)
    rows: Dict[int, ResidueSupport] = {}
    for t in range(len(targets)):
        n = both[targets[t]]
        rows[t] = ResidueSupport(n.key, n.name, 0)
    counts: Dict[Tuple[int, int], int] = {}
    for x, y in zip(i, j):
        key = (int(tow[x]), int(owner[y]))
        counts[key] = counts.get(key, 0) + 1
    for (t, qk), c in counts.items():
        n, q = both[targets[t]], both[qk]
        if n.key == q.key or (n.chain == q.chain and abs(n.resseq - q.resseq) <= p.sep):
            continue  # neighbours in a chain are held by the backbone
        r = rows[t]
        if mol[q.key] != mol[n.key]:
            r.partner += c
        elif not is_rebuilt[q.key]:
            r.scaffold += c
        elif (p.pairs == "builtin" and (n.key, q.key) in paired) or \
                (p.pairs == "contacts" and _base_paired(n, q, p.pair_cutoff)):
            r.pair += c
        else:
            r.self_ += c
    residues = [rows[t] for t in range(len(targets))]
    # segments: runs of rebuilt residues that follow each other in a chain
    segments: List[List[int]] = []
    prev = None
    for t, k in enumerate(targets):
        n = both[k]
        follows = prev is not None and k == prev[0] + 1 and n.chain == prev[1].chain
        if follows and n.resseq - prev[1].resseq <= 1:
            segments[-1].append(t)
        else:
            segments.append([t])
        residues[t].segment = len(segments)
        prev = (k, n)
    return SupportResult(model or "model", residues, segments)


def pick_threshold(runs: Sequence[int]) -> Optional[int]:
    """The longest-run value above the largest gap of the sorted values (at least 5 models and a gap of 2),
    else None."""
    runs = sorted(runs)
    gap, at = 0, None
    for lo, hi in zip(runs, runs[1:]):
        if hi - lo > gap:
            gap, at = hi - lo, hi
    return at if len(runs) >= 5 and gap >= 2 else None


# --- interface records (mdatk pdb interface) ---------------------------------------------------------

RECORD_COLUMNS = ("model", "kind", "residue_a", "name_a", "hand_a", "residue_b", "name_b", "hand_b", "class",
                  "atoms", "distance")


def interface_records(structure: Structure, side_a: Sequence[Range], side_b: Optional[Sequence[Range]],
                      engine: str = "builtin", model: str = "", hbond: float = 3.5, bph_ch: bool = False,
                      params: Optional[CensusParams] = None, dssr_exe: str = "x3dna-dssr", keep: str = ""
                      ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Base pairs, stacking and base / 2'-OH to phosphate hydrogen bonds across the A/B boundary.

    builtin: canonical pairs (PairDetector), the census's stacking test and hydrogen bonds by distance;
    barnaba: its pairs and stacking, hydrogen bonds by distance; dssr: its pairs, stacking and
    hydrogen-bond list."""
    if engine not in INTERFACE_ENGINES:
        raise InterfaceError(f"unknown engine {engine!r} (choose from {', '.join(INTERFACE_ENGINES)})")
    p = params or CensusParams()
    nts = nucleotides(structure)
    a, b = split_sides(nts, side_a, side_b)
    both = list(a) + list(b)
    na = len(a)
    notes: List[str] = []
    rows: List[Dict[str, Any]] = []

    def row(kind, i, j, cls="", atoms="", dist=None):
        x, y = (both[i], both[j]) if i < na else (both[j], both[i])
        rows.append({"model": model, "kind": kind, "residue_a": x.label, "name_a": x.name,
                     "hand_a": x.hand or "", "residue_b": y.label, "name_b": y.name, "hand_b": y.hand or "",
                     "class": cls, "atoms": atoms, "distance": "" if dist is None else f"{dist:.2f}"})

    stem = os.path.splitext(model)[0] if model else "model"
    pairs, stacks, dres = engine_pairs(both, engine, True, p.pair_cutoff, dssr_exe, want_stacks=True,
                                       keep=keep, stem=stem)
    for i, j, c in pairs:
        if (i < na) != (j < na):
            row("pair", i, j, c)
    if stacks is None:
        stacks = [(x, na + y, "") for x, y in _stacks(a, b, p)]
    for i, j, c in stacks:
        if (i < na) != (j < na):
            row("stack", i, j, c)
    donors = _base_donors(bph_ch)
    if dres is not None:
        notes += dres.notes
        for i, ai, j, aj, dist in dres.hbonds:
            if (i < na) == (j < na):
                continue
            for (pi, pa_), (oi, oa_) in (((i, ai), (j, aj)), ((j, aj), (i, ai))):
                if pa_ in PHOSPHATE_O and oa_ not in PHOSPHATE_O:
                    kind = "ohph" if oa_ == "O2'" else "bph" if oa_ in donors(both[oi]) else ""
                    if kind:
                        row(kind, oi, pi, "", f"{oa_}@{both[oi].label} {pa_}@{both[pi].label}", dist)
    else:
        for don, acc, off_d, off_a in ((a, b, 0, na), (b, a, na, 0)):
            for kind, sel in (("bph", donors), ("ohph", _hydroxyl)):
                for di, dn, ai, an, dist in _hbond_contacts(don, acc, sel, hbond):
                    x, y = both[di + off_d], both[ai + off_a]
                    row(kind, di + off_d, ai + off_a, "", f"{dn}@{x.label} {an}@{y.label}", dist)
    order = {"pair": 0, "stack": 1, "bph": 2, "ohph": 3}
    rows.sort(key=lambda r: (order[r["kind"]], parse_label(r["residue_a"]), parse_label(r["residue_b"]),
                             r["atoms"]))
    return rows, notes


# --- command line ------------------------------------------------------------------------------------

def _model_files(paths: Sequence[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.pdb"), recursive=True))
        else:
            files.append(p)
    return files


def _unique_files(paths: Sequence[str]) -> List[str]:
    """``paths`` without repeats of the same file (given twice, or also listed in a table)."""
    seen: Set[str] = set()
    out = []
    for p in paths:
        key = os.path.realpath(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _model_names(paths: Sequence[str]) -> Dict[str, str]:
    """A model's name in the tables: its file name, or its path where file names repeat."""
    seen: Dict[str, int] = {}
    for p in paths:
        seen[os.path.basename(p)] = seen.get(os.path.basename(p), 0) + 1
    return {p: os.path.basename(p) if seen[os.path.basename(p)] == 1 else os.path.normpath(p) for p in paths}


def _groups(args) -> list:
    path = getattr(args, "params", None)
    if not path:
        return []
    from .config import Project
    return list(Project.load(path).groups)


def _sides(args, groups) -> Tuple[List[Range], Optional[List[Range]]]:
    side_a = parse_side(args.side_a, groups)
    if not side_a:
        raise InterfaceError("--side-a selects nothing")
    side_b = parse_side(args.side_b, groups) if (args.side_b or "").strip() else None
    return side_a, side_b


def _say(args):
    """Human-readable lines go to standard output, or to standard error when the table does."""
    stream = sys.stderr if getattr(args, "output", "-") in ("", "-") else sys.stdout
    return lambda text="": print(text, file=stream)


def _window(text: str) -> Tuple[float, float]:
    try:
        lo, hi = (float(v) for v in re.split(r"[,:\s]+|(?<=\d)-(?=\d)", text.strip()) if v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected two numbers, e.g. 5,7 (got {text!r})") from None
    if not 0 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"expected 0 <= min <= max (got {text!r})")
    return lo, hi


def _side_arguments(c, support: bool = False) -> None:
    c.add_argument("--side-a", required=True, metavar="SEL",
                   help="residues of side A: chain-qualified ranges ('A:1-16,B:17-44'), whole chains ('C'), "
                        "plain ranges for every chain ('1-40') or residue groups ('group:Aptamer', with -p)")
    c.add_argument("--side-b", default="", metavar="SEL",
                   help="residues of side B" + (" (the partner)" if support else "")
                        + "; default: every nucleotide not in side A")


def add_commands(sub) -> list:
    """The 'mdatk decoys' actions census, clusters and support (their parsers are returned)."""
    parsers = []
    c = sub.add_parser("census", help="interface census of every model between two sides: contacts, "
                                      "stacking, base pairs, base-phosphate, 2'-OH-phosphate, Mg-bridgeable "
                                      "P-P, contact fingerprint (TSV)")
    c.add_argument("models", nargs="*", help="PDB files or folders (searched for *.pdb)")
    _side_arguments(c)
    c.add_argument("--cutoff", type=float, default=4.5,
                   help="heavy-atom contact distance in Å (default 4.5)")
    c.add_argument("--hbond", type=float, default=3.5,
                   help="donor-acceptor distance of base / 2'-OH to phosphate contacts in Å (default 3.5)")
    c.add_argument("--pairs", choices=PAIR_ENGINES, default="builtin",
                   help="base-pair engine for the wc / gu / other columns (default builtin: canonical pairs "
                        "only; barnaba and dssr also count non-canonical pairs)")
    c.add_argument("--no-wobble", action="store_true", help="do not count G·U wobble pairs")
    c.add_argument("--bph-ch", action="store_true",
                   help="also count the base C-H donors of Zirbel's base-phosphate classes (A C2/C8, G C8, "
                        "C/U C5/C6)")
    c.add_argument("--stack-distance", type=float, default=4.5,
                   help="ring centroid distance in Å (default 4.5)")
    c.add_argument("--stack-angle", type=float, default=30.0,
                   help="angle between ring normals (default 30°)")
    c.add_argument("--stack-separation", type=float, default=2.5,
                   help="minimum centroid offset along the mean normal in Å (default 2.5)")
    c.add_argument("--mg-window", type=_window, default=(5.0, 7.0), metavar="MIN,MAX",
                   help="P-P distances that one ion can bridge, in Å (default 5,7)")
    c.add_argument("--dssr-exe", default="x3dna-dssr", help="x3dna-dssr executable (--pairs dssr)")
    c.add_argument("--scores", metavar="TSV",
                   help="decoy table (run, tag, pdb, score terms) for the score column and the filters")
    c.add_argument("--plan-root", default="", metavar="DIR",
                   help="folder the table's pdb paths are relative to (the plan folder; default: the "
                        "table's folder)")
    c.add_argument("--from-scores", action="store_true",
                   help="take the models from the table's pdb column (every extracted decoy)")
    c.add_argument("--filter", action="append", default=[], metavar="EXPR",
                   help="keep models whose score term passes, e.g. 'linear_chainbreak<=5' (repeatable)")
    c.add_argument("--drop-unscored", action="store_true",
                   help="drop models without a table row (default: keep them, with a note)")
    c.add_argument("--show", type=int, default=20, help="rows of the summary printed (default 20)")
    c.add_argument("--output", "-o", default="-", help="census table (TSV; default: standard output)")
    parsers.append(c)

    c = sub.add_parser("clusters", help="cluster the census fingerprints (Jaccard, leader clustering, "
                                        "medoid representatives), optionally scored against reference "
                                        "interfaces")
    c.add_argument("census", help="census table of 'mdatk decoys census'")
    c.add_argument("--level", choices=LEVELS, default="residues",
                   help="residues (default): the residues of each side; blocks: residues binned per chain; "
                        "pairs: the residue pairs")
    c.add_argument("--block", type=int, default=4, help="residues per block for --level blocks (default 4)")
    c.add_argument("--cut", type=float, default=0.5, help="Jaccard distance cut (default 0.5)")
    c.add_argument("--min-size", type=int, default=5,
                   help="clusters of at least this size count as recurring (default 5)")
    c.add_argument("--broad-limit", type=int, default=10,
                   help="mark clusters whose representative touches more side-B residues as broad "
                        "(default 10; 0: off)")
    c.add_argument("--reference", action="append", default=[], metavar="NAME=A/B",
                   help="a reference interface, e.g. 'pres=A:8-9,B:25-28/C:58-64' (repeatable)")
    c.add_argument("--reference-table", metavar="TSV",
                   help="census table of reference models: one reference per row, named after the model")
    c.add_argument("--representatives", metavar="TEMPLATE",
                   help="copy the representatives of the largest clusters to this file name, '{n}' -> 01, "
                        "02 ...")
    c.add_argument("--n-representatives", type=int, default=5, help="how many (default 5)")
    c.add_argument("--members", metavar="TSV", help="also write model -> cluster")
    c.add_argument("--write-lists", metavar="DIR", help="one file of model names per cluster (clusterK.txt)")
    c.add_argument("--lead-gap", type=float, default=0.15,
                   help="a reference leads when its size-weighted share is this much ahead (default 0.15)")
    c.add_argument("--dispersed-below", type=float, default=0.2,
                   help="coverage below which the pattern is called dispersed (default 0.2)")
    c.add_argument("--few-clusters", type=int, default=6,
                   help="at most this many recurring clusters are 'few' (default 6)")
    c.add_argument("--show", type=int, default=12, help="clusters printed (default 12)")
    c.add_argument("--output", "-o", default="-", help="clusters table (TSV; default: standard output)")
    parsers.append(c)

    c = sub.add_parser("support", help="what holds each rebuilt residue: contacts to the partner side, the "
                                       "own fixed scaffold, base pairs, or only its own rebuilt segments")
    c.add_argument("models", nargs="*", help="PDB files or folders (searched for *.pdb)")
    c.add_argument("--census", metavar="TSV",
                   help="also take the models of a census table (its pdb column: the models that passed its "
                        "score filters)")
    _side_arguments(c, support=True)
    grp = c.add_mutually_exclusive_group(required=True)
    grp.add_argument("--rebuilt", metavar="RANGES",
                     help="the rebuilt residues, e.g. 'A:7-12,B:21-38,C:58-63'")
    grp.add_argument("--fixed", metavar="RANGES",
                     help="the fixed residues (e.g. FARFAR2's -input_res); the rest of the sides is rebuilt")
    c.add_argument("--sep", type=int, default=3,
                   help="ignore contacts to residues of the same chain this close in sequence (default 3)")
    c.add_argument("--cutoff", type=float, default=4.5,
                   help="heavy-atom contact distance in Å (default 4.5)")
    c.add_argument("--run", default="4",
                   help="flag runs of this many unsupported residues (default 4), or 'auto': the largest gap "
                        "of this set's longest runs (5+ models)")
    c.add_argument("--count-self", action="store_true", help="count contacts within the rebuilt segments too")
    c.add_argument("--pairs", choices=SUPPORT_PAIRS, default="off",
                   help="credit base pairs between rebuilt residues as their own class: contacts (two "
                        "distinct base-edge hydrogen bonds within 3.3 Å, any family) or builtin "
                        "(canonical pairs)")
    c.add_argument("--residues", metavar="TSV", help="also write one row per model and rebuilt residue")
    c.add_argument("--profile", metavar="TSV",
                   help="also write, per rebuilt residue, the share of models in which it is unsupported")
    c.add_argument("--output", "-o", default="-", help="per-model table (TSV; default: standard output)")
    parsers.append(c)
    return parsers


def add_interface_command(sub):
    """'mdatk pdb interface'."""
    c = sub.add_parser("interface", help="base pairs, stacking and base / 2'-OH to phosphate hydrogen bonds "
                                         "between two sides of each model (TSV), from the builtin engine, "
                                         "barnaba or DSSR")
    c.add_argument("models", nargs="+", help="PDB files")
    _side_arguments(c)
    c.add_argument("--engine", choices=INTERFACE_ENGINES, default="builtin",
                   help="builtin (canonical pairs, geometric stacking and hydrogen bonds), barnaba, or dssr "
                        "(x3dna-dssr, licensed separately)")
    c.add_argument("--dssr-exe", default="x3dna-dssr", help="x3dna-dssr executable")
    c.add_argument("--keep", metavar="DIR", help="keep DSSR's renamed input and JSON here")
    c.add_argument("--hbond", type=float, default=3.5, help="hydrogen-bond distance in Å (builtin, barnaba)")
    c.add_argument("--bph-ch", action="store_true", help="also list base C-H donors to phosphate")
    c.add_argument("--output", "-o", default="-", help="TSV (default: standard output)")
    return c


def _census_params(args) -> CensusParams:
    return CensusParams(contact=args.cutoff, hbond=args.hbond, stack_distance=args.stack_distance,
                        stack_angle=args.stack_angle, stack_separation=args.stack_separation,
                        mg_window=tuple(args.mg_window), pairs=args.pairs, wobble=not args.no_wobble,
                        bph_ch=args.bph_ch, dssr_exe=args.dssr_exe)


def run_census(args) -> Tuple[int, Dict[str, Any]]:
    say = _say(args)
    groups = _groups(args)
    side_a, side_b = _sides(args, groups)
    filters = parse_filters(args.filter)
    table = read_decoy_table(args.scores) if args.scores else None
    root = args.plan_root or (guess_plan_root(table) if table is not None else "")
    files = _model_files(args.models)
    if args.from_scores:
        if table is None:
            raise InterfaceError("--from-scores needs --scores")
        files += models_from_table(table, root)
    files = _unique_files(files)
    if not files:
        raise InterfaceError("no models given")
    outcome = apply_filters(files, table, filters, root, args.drop_unscored)
    notes = []
    if filters:
        notes.append(f"filters {', '.join(format_filter(f) for f in filters)}: kept {len(outcome.kept)}, "
                     f"dropped {len(outcome.dropped)} of {len(files)}")
    if table is not None and outcome.unscored:
        notes.append(f"{len(outcome.unscored)} model(s) have no row (or no value) in "
                     f"{os.path.basename(table.path)}"
                     + (" and were dropped" if args.drop_unscored else " and were kept"))
    terms = ["score"] if table is not None and table.has("score") else []
    terms += [t for t, _o, _v in filters if t not in terms]
    params = _census_params(args)
    rows, bad = [], []
    names = _model_names(outcome.kept)
    for path in outcome.kept:
        try:
            s = read_pdb(path)
            r = census(s, side_a, side_b, params, model=names[path], pdb=path)
        except (OSError, PDBError, InterfaceError) as exc:
            bad.append(f"{path}: {exc}")
            continue
        row = outcome.rows.get(path)
        r.extra = {t: ("" if row is None else row.get(t, "")) for t in terms}
        rows.append(r)
        for note in r.notes:
            say(f"  {r.model}: {note}")
    for line in bad[:5]:
        print(f"skipped {line}", file=sys.stderr)
    if bad:
        print(f"{len(bad)} model(s) could not be measured", file=sys.stderr)
    if not rows:
        raise InterfaceError("no model could be measured" + (f" ({bad[0]})" if bad else ""))
    dicts = [r.as_dict() for r in rows]
    write_table(args.output, dicts)
    for note in notes:
        say(note)
    say(f"{len(rows)} models, side A {format_side(side_a)}, side B "
        f"{format_side(side_b) if side_b is not None else 'every other nucleotide'}")
    say(_census_summary(dicts, args.show))
    return 0, {"files": [] if args.output in ("", "-") else [args.output], "notes": notes,
               "models": len(rows), "skipped": len(bad)}


def _census_summary(rows: Sequence[Dict[str, Any]], show: int) -> str:
    cols = ("contacts", "stack", "wc", "gu", "other", "bph", "ohph", "mgbr", "n_a", "n_b")
    ranked = sorted(rows, key=lambda r: -int(r["contacts"]))
    lines = [f"{'model':<34} " + " ".join(f"{c:>8}" for c in cols)]
    for r in ranked[:show]:
        lines.append(f"{str(r['model'])[:34]:<34} " + " ".join(f"{str(r[c]):>8}" for c in cols))
    med = []
    for c in cols[:-2]:
        vals = [float(r[c]) for r in rows if r[c] != ""]
        if vals:
            med.append(f"{c} {np.median(vals):g}")
    lines.append(f"median over {len(rows)} models: " + ", ".join(med))
    with_contact = [r for r in rows if int(r["contacts"]) > 0]
    if rows and rows[0]["wc"] != "":
        no_wc = [r for r in with_contact if int(r["wc"]) == 0]
        lines.append(f"models with interface contacts but no Watson-Crick pair across it: {len(no_wc)} of "
                     f"{len(rows)}")
    return "\n".join(lines)


def run_clusters(args) -> Tuple[int, Dict[str, Any]]:
    say = _say(args)
    groups = _groups(args)
    entries = read_census(args.census)
    if not entries:
        raise InterfaceError(f"{args.census}: no models")
    refs: List[Reference] = [parse_reference(t, groups, entries) for t in args.reference]
    if args.reference_table:
        refs += references_from_census(read_census(args.reference_table))
    params = ClusterParams(level=args.level, block=args.block, cut=args.cut, min_size=args.min_size,
                           broad_limit=args.broad_limit, lead_gap=args.lead_gap,
                           dispersed_below=args.dispersed_below, few_clusters=args.few_clusters)
    if args.block < 1:
        raise InterfaceError("--block must be at least 1")
    res = cluster_census(entries, params, refs)
    rep_files: Dict[int, str] = {}
    notes: List[str] = []
    if args.representatives:
        rep_files, notes = write_representatives(res, args.representatives, args.n_representatives,
                                                 ("", os.path.dirname(os.path.abspath(args.census))))
    rows = cluster_rows(res, rep_files)
    columns = list(rows[0]) if rows else ["cluster", "size", "representative", "representative_pdb",
                                          "members"]
    write_table(args.output, rows, columns)
    if args.members:
        member_rows = []
        for k, c in enumerate(res.clusters, start=1):
            for x in c.members:
                member_rows.append({"model": entries[x].model, "cluster": k,
                                    "representative": "yes" if x == c.representative else "no"})
        for x in res.empty:
            member_rows.append({"model": entries[x].model, "cluster": "", "representative": ""})
        write_table(args.members, member_rows, ["model", "cluster", "representative"])
    if args.write_lists:
        os.makedirs(args.write_lists, exist_ok=True)
        for k, c in enumerate(res.clusters, start=1):
            with open(os.path.join(args.write_lists, f"cluster{k}.txt"), "w", encoding="utf-8") as fh:
                fh.write("".join(entries[x].model + "\n" for x in c.members))
    for note in notes:
        say(note)
    say(_cluster_text(res, args.show))
    files = [f for f in [args.output if args.output not in ("", "-") else "", args.members] if f]
    return 0, {"files": files + list(rep_files.values()), "notes": notes, "summary": res.summary}


def _cluster_text(res: ClusterResult, show: int) -> str:
    s, p = res.summary, res.params
    level = p.level + (f" of {p.block}" if p.level == "blocks" else "")
    lines = [f"{s['models']} models: {s['with_interface']} with an interface, "
             f"{s['without_interface']} without",
             f"clustering on {level} at Jaccard cut {p.cut:g}: {s['clusters']} clusters, "
             f"{s['clusters_min_size']} with >= {p.min_size} members covering {s['covered']} models "
             f"({100 * s['coverage']:.0f} %), {s['singletons']} singletons",
             f"pattern: {s['pattern']}"]
    if p.level == "blocks":
        lines.append("note: blocks merge surfaces that fall into the same bins; compare with --level "
                     "residues")
    if res.references:
        for r in res.references:
            ea, eb = res.exclusive[r.name]
            lines.append(f"reference {r.name}: exclusive residues A {compress_labels(ea) or '-'}, "
                         f"B {compress_labels(eb) or '-'}")
        sa, sb = res.shared
        if sa or sb:
            lines.append(f"shared by all references: A {compress_labels(sa) or '-'}, "
                         f"B {compress_labels(sb) or '-'}")
    lines.append("")
    head = (f"{'#':>3} {'size':>5} {'representative':<30} {'cont':>6} {'stack':>6} {'wc':>5} {'bph':>5} "
            f"{'nA':>3} {'nB':>3}")
    head += "".join(f" {r.name[:8]:>8}" for r in res.references)
    lines.append(head)
    for k, c in enumerate(res.clusters[:show], start=1):
        rep = res.entries[c.representative]

        def mean(key):
            vals = [res.entries[x].counts.get(key) for x in c.members]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else float("nan")
        line = (f"{k:>3} {len(c.members):>5} {rep.model[:30]:<30} {mean('contacts'):>6.0f} "
                f"{mean('stack'):>6.1f} {mean('wc'):>5.1f} {mean('bph'):>5.1f} {len(rep.residues_a):>3} "
                f"{len(rep.residues_b):>3}")
        for r in res.references:
            share = c.shares.get(r.name)
            line += f" {'-':>8}" if share is None else f" {100 * share:>7.0f}%"
        if c.broad:
            line += "  broad"
        lines.append(line)
        lines.append(f"      A {compress_labels(rep.residues_a)}   B {compress_labels(rep.residues_b)}")
    if res.references:
        lines.append("")
        lines.append("size-weighted reference shares over the clusters of >= "
                     f"{p.min_size} (exclusive residues, both sides, normalised per reference):")
        for r in res.references:
            a = s["shares_all"].get(r.name)
            f = s["shares_focused"].get(r.name)
            lines.append(f"  {r.name:<16} all {'-' if a is None else f'{100 * a:.0f} %':>6}   "
                         f"not broad {'-' if f is None else f'{100 * f:.0f} %':>6}")
        if "lead" in s:
            lines.append(f"  leading reference: {s['lead']} (ahead by {s['lead_gap']:.2f})" if s["lead"] else
                         f"  no reference leads by more than {p.lead_gap:g} "
                         f"(largest gap {s['lead_gap']:.2f})")
    return "\n".join(lines)


def run_support(args) -> Tuple[int, Dict[str, Any]]:
    say = _say(args)
    groups = _groups(args)
    side_a, side_b = _sides(args, groups)
    files = _model_files(args.models)
    if args.census:
        here = os.path.dirname(os.path.abspath(args.census))
        for r in read_table(args.census):
            path = r.get("pdb", "")
            if path and not os.path.isfile(path) and os.path.isfile(os.path.join(here, path)):
                path = os.path.join(here, path)  # written from another folder
            if path:
                files.append(path)
    files = _unique_files(files)
    if not files:
        raise InterfaceError("no models given")
    params = SupportParams(sep=args.sep, cutoff=args.cutoff, count_self=args.count_self, pairs=args.pairs)
    auto = str(args.run).strip().lower() == "auto"
    if not auto:
        try:
            threshold = int(args.run)
        except ValueError:
            raise InterfaceError("--run takes an integer or 'auto'") from None
    fixed = parse_side(args.fixed, groups) if args.fixed is not None else None  # '': nothing fixed
    rebuilt_spec = parse_side(args.rebuilt, groups) if args.rebuilt is not None else None
    results, bad = [], []
    names = _model_names(files)
    for path in files:
        try:
            s = read_pdb(path)
            if fixed is not None:
                nts = nucleotides(s)
                a, b = split_sides(nts, side_a, side_b, require_b=False)
                rebuilt = [(n.chain, n.resseq, n.resseq) for n in list(a) + list(b)
                           if not in_side(n.chain, n.resseq, fixed)]
            else:
                rebuilt = rebuilt_spec
            results.append(segment_support(s, side_a, side_b, rebuilt, params, model=names[path]))
        except (OSError, PDBError, InterfaceError) as exc:
            bad.append(f"{path}: {exc}")
    for line in bad[:5]:
        print(f"skipped {line}", file=sys.stderr)
    if not results:
        raise InterfaceError("no model could be measured" + (f" ({bad[0]})" if bad else ""))
    runs = [r.longest_run(args.count_self) for r in results]
    if auto:
        threshold = pick_threshold(runs) or 4
    rows, residue_rows = [], []
    for r in results:
        unsup = [x for x in r.residues if not x.supported(args.count_self)]
        rows.append({"model": r.model, "rebuilt": len(r.residues), "unsupported": len(unsup),
                     "longest_run": r.longest_run(args.count_self),
                     "self_only": sum(1 for x in unsup if x.self_),
                     "flagged": "yes" if r.longest_run(args.count_self) >= threshold else "no",
                     "segments": " ".join("".join(r.residues[k].mark for k in seg) for seg in r.segments),
                     "unsupported_residues": compress_labels(x.key for x in unsup)})
        for x in r.residues:
            residue_rows.append({"model": r.model, "residue": residue_label(x.key), "name": x.name,
                                 "segment": x.segment, "partner": x.partner, "scaffold": x.scaffold,
                                 "pair": x.pair, "self": x.self_,
                                 "supported": "yes" if x.supported(args.count_self) else "no"})
    write_table(args.output, rows)
    out_files = [args.output] if args.output not in ("", "-") else []
    if args.residues:
        write_table(args.residues, residue_rows)
        out_files.append(args.residues)
    tally: Dict[str, List[int]] = {}
    for x in residue_rows:
        tally.setdefault(x["residue"], [0, 0])
        tally[x["residue"]][0] += x["supported"] == "no"
        tally[x["residue"]][1] += 1
    if args.profile:
        write_table(args.profile, [{"residue": k, "unsupported": v[0], "models": v[1],
                                    "share": round(v[0] / v[1], 4)} for k, v in tally.items()])
        out_files.append(args.profile)
    say(f"{len(results)} models; contacts beyond {args.sep} in sequence (within a chain), within "
        f"{args.cutoff:g} Å" + ("; self-contact counts" if args.count_self else ""))
    say("longest-run distribution: " + ", ".join(str(x) for x in sorted(runs)))
    if auto:
        chosen = pick_threshold(runs)
        say(f"run threshold {threshold}" + (" (the largest gap of this set)" if chosen else
                                            " (no clear gap; the default 4)"))
    say(f"{sum(1 for r in rows if r['flagged'] == 'yes')} of {len(rows)} models have a run of >= {threshold} "
        "unsupported residues")
    say("segments: '#' partner, 'o' scaffold, '=' base pair inside the rebuilt segments, ':' only its own "
        "rebuilt segments, '.' nothing")
    return 0, {"files": out_files, "models": len(results), "threshold": threshold}


def run_interface(args) -> Tuple[int, Dict[str, Any]]:
    say = _say(args)
    groups = _groups(args)
    side_a, side_b = _sides(args, groups)
    rows, notes = [], []
    for path in args.models:
        s = read_pdb(path)
        recs, n = interface_records(s, side_a, side_b, args.engine, model=os.path.basename(path),
                                    hbond=args.hbond, bph_ch=args.bph_ch, dssr_exe=args.dssr_exe,
                                    keep=args.keep or "")
        rows += recs
        notes += [f"{os.path.basename(path)}: {x}" for x in n]
        kinds = [r["kind"] for r in recs]
        say(f"{os.path.basename(path)}: {kinds.count('pair')} pairs, {kinds.count('stack')} stacks, "
            f"{kinds.count('bph')} base-phosphate and {kinds.count('ohph')} 2'-OH-phosphate hydrogen bonds "
            "across the interface")
    write_table(args.output, rows, RECORD_COLUMNS)
    for note in notes:
        say(note)
    if args.engine == "dssr":
        say("DSSR does not check chirality: a D-L pair is classified like any other")
    return 0, {"files": [] if args.output in ("", "-") else [args.output], "notes": notes}


def run(args) -> Tuple[int, Dict[str, Any]]:
    action = getattr(args, "decoys_command", None) or "interface"
    return {"census": run_census, "clusters": run_clusters, "support": run_support,
            "interface": run_interface}[action](args)
