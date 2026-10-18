"""In-house starting-structure builder: ideal A-form RNA helices (D and L), base mutations that keep
the glycosidic frame, and rigid-body fitting of a model into a density map.

Ideal helices (:func:`build_helix`)
    Every nucleotide is the Arnott A-RNA fiber monomer (Arnott, Hukins & Dover 1972; the ``$ARNA``
    "ARNOTT ARNA MONOMER UNIT" of AmberTools' ``dat/leap/parm/nucgen.dat``, the fiber table of NAB /
    LEaP's nucgen), embedded below in its cylindrical coordinates (r, phi, z): C3'-endo sugars
    (pseudorotation phase about 14 degrees), chi about -166 degrees, 11 residues per turn (twist 32.7
    degrees, rise 2.81 A).  Residue k of a strand is the monomer turned by -k * 32.7 degrees about the
    helix axis (z) and moved by -k * 2.81 A, so strand 1 runs 5'->3' towards -z; the partner strand
    is the two-fold image about the x axis (the fiber dyad), which gives Watson-Crick pairs with
    C1'-C1' 10.9 A and O3'-P links of 1.59 A on both strands.  The nucgen table has no hydrogens on
    carbon and no hydroxyl hydrogens; with ``hydrogens=True`` every hydrogen is placed from ideal
    bond geometry (the amino and imino hydrogens too, with the PDB v3 / Amber names).  G-U pairs of
    an explicit partner get an approximate wobble geometry (both bases turned in their plane about
    C1', see ``WOBBLE_TURN``); other mismatches keep the Watson-Crick frame and are reported when
    their bases collide.  An L helix is the mirror image (x -> -x) of the D helix, a left-handed
    duplex of L-nucleotides with L residue names from :mod:`mdatk.pdbtools`.

Mutations (:func:`mutate`)
    The new base is superposed on the old glycosidic frame (C1', the glycosidic N and the two ring
    atoms bonded to it: purine N9/C4/C8 <-> pyrimidine N1/C2/C6); sugar and backbone atoms are kept
    verbatim, the residue name keeps its scheme, chirality and terminal variant (LG5 -> LC5, 0G -> 0C,
    G5 -> C5).  Compensatory mutations change the Watson-Crick / G-U partner of every mutated base
    (found with :class:`mdatk.secstruct.PairDetector`) to the complement unless the partner is
    mutated explicitly.

Rigid-body fit (:func:`fit_to_map`)
    The global (rotations x FFT translations, then local refinement) or local search of
    :mod:`mdatk.density` ('mdatk density fit'); only the coordinates of the written file change.

Everything runs locally; nothing here calls an external program.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pdbtools
from .pdbtools import Atom, PDBError, Residue, Structure

TWIST = 32.7  # degrees per residue: 11-fold Arnott A-RNA
RISE = 2.81  # Å per residue
SCHEMES = pdbtools.SCHEMES
PARTNER_MODES = ("auto", "none")  # or an explicit partner sequence
# names without a chosen scheme: A/C/G/U (D) and the CCD components 0A/0C/0G/0U (L), all ATOM records,
# no terminal variants
DEFAULT_SCHEME = "rosetta"
SEARCHES = ("global", "local")


class BuildError(PDBError):
    """Invalid sequence, mutation or fit request."""


# --- templates ----------------------------------------------------------------------------------------

# Arnott A-RNA monomer unit in cylindrical coordinates (name, r Å, phi degrees, z Å), copied from the
# $ARNA block of AmberTools dat/leap/parm/nucgen.dat; names in PDB v3 form (OA/OB -> OP1/OP2 by the
# D-sugar chirality at P, O1' -> O4').  Base hydrogens of the table are not used (see _hydrogens).
_BACKBONE = (("P", 8.71, 70.5, -3.75), ("OP1", 9.59, 71.2, -4.94), ("OP2", 7.50, 76.6, -3.80),
             ("O5'", 8.49, 60.1, -3.48), ("C5'", 9.75, 55.8, -3.35), ("C4'", 9.68, 46.9, -3.10),
             ("O4'", 9.15, 45.3, -1.77), ("C3'", 8.86, 41.4, -4.02), ("O3'", 9.54, 40.1, -5.25),
             ("C2'", 8.82, 33.2, -3.17), ("O2'", 10.15, 30.2, -3.14), ("C1'", 8.55, 36.9, -1.77))
_BASES = {
    "A": (("N9", 7.12, 37.2, -1.39), ("C8", 6.39, 47.0, -1.33), ("N7", 5.15, 46.2, -0.95),
          ("C5", 5.14, 30.9, -0.75), ("C6", 4.39, 17.4, -0.34), ("N6", 3.08, 15.6, -0.05),
          ("N1", 5.28, 5.5, -0.24), ("C2", 6.54, 8.7, -0.54), ("N3", 7.15, 18.0, -0.93),
          ("C4", 6.44, 27.6, -1.01)),
    "G": (("N9", 7.12, 37.2, -1.39), ("C8", 6.36, 47.0, -1.32), ("N7", 5.11, 46.0, -0.94),
          ("C5", 5.13, 30.5, -0.74), ("C6", 4.34, 17.1, -0.33), ("O6", 3.16, 13.9, -0.04),
          ("N1", 5.35, 5.5, -0.26), ("C2", 6.67, 8.5, -0.55), ("N2", 7.58, 0.7, -0.42),
          ("N3", 7.20, 18.0, -0.94), ("C4", 6.44, 27.5, -1.01)),
    "C": (("N1", 7.12, 37.2, -1.39), ("C2", 6.61, 26.8, -1.04), ("O2", 7.51, 19.9, -1.06),
          ("N3", 5.32, 24.4, -0.68), ("C4", 4.50, 36.8, -0.67), ("N4", 3.23, 36.1, -0.31),
          ("C5", 5.26, 50.6, -1.03), ("C6", 6.53, 47.6, -1.38)),
    "U": (("N1", 7.12, 37.2, -1.39), ("C2", 6.63, 26.9, -1.04), ("O2", 7.46, 19.6, -1.04),
          ("N3", 5.31, 25.2, -0.69), ("C4", 4.42, 37.8, -0.66), ("O4", 3.24, 37.1, -0.32),
          ("C5", 5.28, 50.8, -1.04), ("C6", 6.54, 47.7, -1.38)),
}
BACKBONE_NAMES = tuple(n for n, *_ in _BACKBONE)
BASE_NAMES = {b: tuple(n for n, *_ in atoms) for b, atoms in _BASES.items()}
# hydrogens: (name, parent, rule, reference atoms) - see _hydrogens
_H_RULES = {
    "sugar": (("H5'", "C5'", "ch2", ("O5'", "C4'")), ("H5''", "C5'", "ch2", ("O5'", "C4'")),
              ("H4'", "C4'", "ch", ("C5'", "O4'", "C3'")), ("H3'", "C3'", "ch", ("C4'", "O3'", "C2'")),
              ("H2'", "C2'", "ch", ("C3'", "O2'", "C1'")), ("HO2'", "O2'", "oh", ("C2'", "C3'")),
              ("H1'", "C1'", "ch", ("O4'", "C2'", "N"))),
    "A": (("H8", "C8", "sp2", ("N7", "N9")), ("H61", "N6", "nh2", ("C6", "C5")),
          ("H62", "N6", "nh2", ("C6", "N1")), ("H2", "C2", "sp2", ("N1", "N3"))),
    "G": (("H8", "C8", "sp2", ("N7", "N9")), ("H1", "N1", "nh", ("C2", "C6")),
          ("H21", "N2", "nh2", ("C2", "N3")), ("H22", "N2", "nh2", ("C2", "N1"))),
    "C": (("H41", "N4", "nh2", ("C4", "C5")), ("H42", "N4", "nh2", ("C4", "N3")),
          ("H5", "C5", "sp2", ("C4", "C6")), ("H6", "C6", "sp2", ("C5", "N1"))),
    "U": (("H3", "N3", "nh", ("C2", "C4")), ("H5", "C5", "sp2", ("C4", "C6")),
          ("H6", "C6", "sp2", ("C5", "N1"))),
}
BOND_H = {"ch": 1.09, "ch2": 1.09, "sp2": 1.08, "nh": 1.01, "nh2": 1.01, "oh": 0.96}
HO2_TORSION = 180.0  # C3'-C2'-O2'-HO2' (degrees), the hydroxyl hydrogen pointing away from O3'
P_OP3 = 1.48  # Å, P-OP3 of a 5'-phosphate (along the bond to the missing previous O3')
# G-U wobble in the fiber frame: in-plane turns (degrees, about C1' and the base normal N->C4 x N->C8 /
# N->C2 x N->C6) giving G O6-U N3 2.8 Å and G N1-U O2 3.1 Å (least squares to 2.85 Å on this template)
WOBBLE_TURN = {"G": -13.3, "U": 11.4}
GLYCOSIDIC = {"A": ("N9", "C4", "C8"), "G": ("N9", "C4", "C8"), "C": ("N1", "C2", "C6"),
              "U": ("N1", "C2", "C6")}
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
WATSON_CRICK_PAIRS = frozenset({"AU", "UA", "GC", "CG"})
WOBBLE_PAIRS = frozenset({"GU", "UG"})
CLASH = 2.5  # Å, closest allowed heavy-atom distance between the bases of a mismatched pair


def _cartesian(r: float, phi: float, z: float) -> np.ndarray:
    p = math.radians(phi)
    return np.array([r * math.cos(p), r * math.sin(p), z])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _place(a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float, angle: float, torsion: float
           ) -> np.ndarray:
    """Atom d bonded to c with angle b-c-d and torsion a-b-c-d (degrees)."""
    bc = _unit(c - b)
    n = _unit(np.cross(b - a, bc))
    m = np.cross(n, bc)
    th, ph = math.radians(angle), math.radians(torsion)
    d2 = np.array([-bond * math.cos(th), bond * math.sin(th) * math.cos(ph), bond * math.sin(th) * math.sin(ph)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def _turn_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    k = _unit(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = math.radians(degrees)
    return np.eye(3) + math.sin(t) * kx + (1 - math.cos(t)) * (kx @ kx)


def _hydrogens(heavy: Dict[str, np.ndarray], base: str) -> Dict[str, np.ndarray]:
    """Hydrogens of a D-nucleotide from ideal geometry (sp3 C-H along the negative sum of the bond
    vectors, sp2 C-H / N-H on the external bisector, amino H parallel to the opposite ring bond)."""
    out: Dict[str, np.ndarray] = {}
    glyco = GLYCOSIDIC[base][0]
    for name, parent, rule, refs in _H_RULES["sugar"] + _H_RULES[base]:
        c = heavy[parent]
        refs = [glyco if r == "N" else r for r in refs]
        d = BOND_H[rule]
        if rule in ("ch", "sp2", "nh"):
            out[name] = c - d * _unit(sum(_unit(heavy[r] - c) for r in refs))
        elif rule == "nh2":  # parallel to the ring bond refs[1] -> refs[0]: cis to the other ring neighbour
            x, other = heavy[refs[0]], heavy[refs[1]]
            out[name] = c + d * _unit(x - other)
        elif rule == "oh":
            out[name] = _place(heavy[refs[1]], heavy[refs[0]], c, d, 109.5, HO2_TORSION)
        elif rule == "ch2" and name not in out:
            u1, u2 = _unit(heavy[refs[0]] - c), _unit(heavy[refs[1]] - c)
            bis, perp = -_unit(u1 + u2), _unit(np.cross(u1, u2))
            half = math.radians(109.5 / 2)
            h1 = c + d * (math.cos(half) * bis + math.sin(half) * perp)
            h2 = c + d * (math.cos(half) * bis - math.sin(half) * perp)
            # PDB v3 names of a D-ribose: H5' gives a positive volume C5'(O5', C4', H5')
            if pdbtools.chiral_volume(c, heavy[refs[0]], heavy[refs[1]], h1) < 0:
                h1, h2 = h2, h1
            out["H5'"], out["H5''"] = h1, h2
    return out


@lru_cache(maxsize=None)
def _template(base: str) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    """The monomer of ``base`` in the helix frame: heavy atoms, then hydrogens (cached, read-only)."""
    heavy = {n: _cartesian(r, p, z) for n, r, p, z in _BACKBONE + _BASES[base]}
    hyd = _hydrogens(heavy, base)
    return tuple((n, tuple(x)) for n, x in list(heavy.items()) + list(hyd.items()))


def template(base: str, hydrogens: bool = True) -> Dict[str, np.ndarray]:
    """Atom name -> coordinates of the A-form monomer (helix frame, D sugar, strand 1, level 0)."""
    base = base.upper()
    if base not in _BASES:
        raise BuildError(f"no template for base {base!r} (A, C, G, U)")
    return {n: np.array(x) for n, x in _template(base) if hydrogens or not n.startswith("H")}


def _base_atoms(base: str, hydrogens: bool) -> List[str]:
    names = list(BASE_NAMES[base])
    if hydrogens:
        names += [h for h, *_ in _H_RULES[base]]
    return names


def _base_normal(coords: Dict[str, np.ndarray], base: str) -> np.ndarray:
    n, a, b = GLYCOSIDIC[base]
    return _unit(np.cross(coords[a] - coords[n], coords[b] - coords[n]))


def _turn_base(coords: Dict[str, np.ndarray], base: str, degrees: float, hydrogens: bool) -> None:
    """Turn the base (in place) in its plane about C1'."""
    if not degrees:
        return
    r = _turn_matrix(_base_normal(coords, base), degrees)
    c1 = coords["C1'"]
    for name in _base_atoms(base, hydrogens):
        if name in coords:
            coords[name] = (coords[name] - c1) @ r.T + c1


def _screw(xyz: np.ndarray, level: float) -> np.ndarray:
    t = math.radians(-TWIST * level)
    rot = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(xyz) @ rot.T + np.array([0.0, 0.0, -RISE * level])


def _dyad(xyz: np.ndarray) -> np.ndarray:
    return np.asarray(xyz) * np.array([1.0, -1.0, -1.0])


# --- sequences ----------------------------------------------------------------------------------------

def parse_sequence(text: str, what: str = "sequence") -> Tuple[str, List[str]]:
    """Clean an RNA sequence: whitespace, '5'-'/'-3'' and '-' are ignored, T is read as U.

    Returns the sequence and notes; raises :class:`BuildError` for anything else."""
    raw = (text or "").strip()
    seq = re.sub(r"^5'?-?|-?3'?$", "", raw.upper().replace("’", "'")) if raw else ""
    seq = re.sub(r"[\s\-]", "", seq)
    notes = []
    if "T" in seq:
        seq = seq.replace("T", "U")
        notes.append(f"{what}: T read as U (RNA)")
    bad = sorted(set(seq) - set("ACGU"))
    if bad:
        raise BuildError(f"{what}: invalid letter(s) {', '.join(bad)} (RNA: A, C, G, U)")
    if not seq:
        raise BuildError(f"{what} is empty")
    return seq, notes


def complement(seq: str) -> str:
    """Watson-Crick complement, 5'->3' (reverse complement)."""
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def resolve_partner(sequence: str, partner: str = "auto") -> Tuple[str, Optional[str], List[str]]:
    """(sequence, partner sequence 5'->3' or None, notes) for ``partner`` 'auto', 'none' or a sequence."""
    seq, notes = parse_sequence(sequence)
    mode = "auto" if partner is None else partner.strip()
    if not mode:
        raise BuildError("type the partner sequence (5'->3'), or choose the Watson-Crick complement or a "
                         "single strand")
    if mode.lower() == "none":
        return seq, None, notes
    if mode.lower() == "auto":
        return seq, complement(seq), notes + ["partner strand: Watson-Crick complement"]
    other, more = parse_sequence(mode, "partner")
    notes += more
    if len(other) != len(seq):
        raise BuildError(f"the partner has {len(other)} nucleotides but the sequence has {len(seq)}; "
                         "an explicit partner must have the same length (5'->3')")
    return seq, other, notes


def pair_types(seq: str, partner: str) -> List[str]:
    """'wc', 'wobble' or 'mismatch' for the pairs (seq[i], partner[n-1-i])."""
    out = []
    for a, b in zip(seq, reversed(partner)):
        pair = a + b
        out.append("wc" if pair in WATSON_CRICK_PAIRS else "wobble" if pair in WOBBLE_PAIRS else "mismatch")
    return out


def check_helix(sequence: str, partner: str = "auto") -> List[Tuple[str, str]]:
    """Sequence-level messages [(level, text)] (levels 'error', 'warning', 'info')."""
    try:
        seq, other, notes = resolve_partner(sequence, partner)
    except BuildError as exc:
        return [("error", str(exc))]
    out = [("info", n) for n in notes]
    if other is None:
        out.append(("info", f"single strand of {len(seq)} nucleotides in the A-form helix geometry"))
        return out
    kinds = pair_types(seq, other)
    n_wc, n_wob = kinds.count("wc"), kinds.count("wobble")
    mism = [f"{k + 1}:{seq[k]}·{other[len(seq) - 1 - k]}" for k, t in enumerate(kinds) if t == "mismatch"]
    out.append(("info", f"{len(seq)} base pairs: {n_wc} Watson-Crick, {n_wob} G·U wobble, "
                        f"{len(mism)} mismatched"))
    if (partner or "auto").strip().lower() not in ("auto", "none") and other == complement(seq):
        out.append(("info", "the explicit partner is the Watson-Crick complement"))
    if mism:
        out.append(("warning", "mismatched pairs keep the Watson-Crick frame (bases may collide; minimise "
                               "or remodel): " + ", ".join(mism[:8]) + (" ..." if len(mism) > 8 else "")))
    return out


# --- ideal helices ------------------------------------------------------------------------------------

def _tail(xyz: Sequence[float], occupancy: float = 1.0, bfactor: float = 0.0) -> str:
    x, y, z = (float(v) for v in xyz)
    if max(abs(x), abs(y), abs(z)) >= 9999.9995 or min(x, y, z) <= -999.9995:
        raise BuildError("coordinates do not fit the PDB columns")
    return f"   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}".ljust(49)


def _element(name: str) -> str:
    return name.lstrip("0123456789")[:1]


def _strand_residue(base: str, level: int, partner: bool, five: bool, three: bool, phosphate5: bool,
                    hydrogens: bool, wobble: bool) -> List[Tuple[str, np.ndarray]]:
    """Atoms of one nucleotide in the helix frame (before mirroring)."""
    t = template(base, hydrogens)
    if wobble:
        _turn_base(t, base, WOBBLE_TURN[base], hydrogens)
    extra_first: List[Tuple[str, np.ndarray]] = []
    extra_last: List[Tuple[str, np.ndarray]] = []
    if five:
        if phosphate5:  # OP3 towards the O3' the previous residue would have
            prev_o3 = _screw(t["O3'"], -1)
            extra_first.append(("OP3", t["P"] + P_OP3 * _unit(prev_o3 - t["P"])))
        else:
            if hydrogens:
                extra_last.append(("HO5'", _place(t["C4'"], t["C5'"], t["O5'"], BOND_H["oh"], 109.5, 180.0)))
            for name in ("P", "OP1", "OP2"):
                del t[name]
    if three and hydrogens:
        extra_last.append(("HO3'", _place(t["C4'"], t["C3'"], t["O3'"], BOND_H["oh"], 109.5, 180.0)))
    atoms = extra_first + [(n, t[n]) for n in BACKBONE_NAMES + BASE_NAMES[base] if n in t]
    if hydrogens:
        atoms += [(n, t[n]) for n, *_ in _H_RULES["sugar"] + _H_RULES[base]]
    atoms += extra_last
    out = []
    for name, xyz in atoms:
        if partner:
            xyz = _dyad(xyz)
        out.append((name, _screw(xyz, level)))
    return out


def _parse_pair(value, kind=str) -> Tuple:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    elif isinstance(value, (int, float)):
        parts = [value]
    else:
        parts = list(value)
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2:
        raise BuildError(f"expected two values (strand 1, strand 2), got {value!r}")
    try:
        return tuple(kind(p) for p in parts)
    except ValueError:
        raise BuildError(f"cannot read {value!r}") from None


def build_helix(sequence: str, partner: str = "auto", chirality: str = "D", chains=("A", "B"),
                start=(1, 1), scheme: Optional[str] = None, terminals: bool = True,
                hydrogens: bool = False, phosphate5: bool = False, center: bool = True,
                notes: Optional[List[str]] = None) -> str:
    """PDB text of an ideal A-form RNA: one strand (``partner='none'``) or an antiparallel duplex.

    ``partner``: 'auto' (Watson-Crick complement), 'none' or the partner sequence 5'->3' (same length;
    mismatches and G·U wobbles allowed).  ``chirality`` 'D' (right-handed) or 'L' (the mirror image).
    ``chains`` / ``start``: chain ids and first residue numbers of strand 1 and 2 (also 'A,B' / '1,1').
    ``scheme``: residue names as in :mod:`mdatk.pdbtools` (None: ``DEFAULT_SCHEME``);
    ``terminals``: 5'/3' variants for the schemes that have them (amber, ccd).  ``phosphate5`` keeps a
    5'-phosphate (P, OP1, OP2, OP3) on each strand; by default chains start with a 5'-OH.
    ``hydrogens`` adds all hydrogens; ``center`` puts the helix centre at the origin (axis along z).
    ``notes`` (a list) receives the build notes.
    """
    seq, other, msgs = resolve_partner(sequence, partner)
    chirality = (chirality or "D").strip().upper()
    if chirality not in ("D", "L"):
        raise BuildError(f"chirality must be D or L, got {chirality!r}")
    chain1, chain2 = _parse_pair(chains)
    first1, first2 = _parse_pair(start, int)
    if any(len(c) > 1 for c in (chain1, chain2)):
        raise BuildError("chain ids must be single characters")
    if other is not None and chain1 == chain2:
        raise BuildError("the two strands need different chain ids")
    chosen = scheme or None
    scheme = scheme or DEFAULT_SCHEME
    if scheme not in SCHEMES:
        raise BuildError(f"unknown residue name scheme {scheme!r} (choose from {', '.join(SCHEMES)})")
    n = len(seq)
    kinds = pair_types(seq, other) if other is not None else ["none"] * n
    # (base, helix level, partner strand, 5' end, 3' end, wobble) per residue, 5'->3'
    strands = [(chain1, first1, [(b, k, False, k == 0, k == n - 1, kinds[k] == "wobble")
                                 for k, b in enumerate(seq)])]
    if other is not None:  # strand 2 pairs residue n-1-j of strand 1 at the same level
        strands.append((chain2, first2, [(b, n - 1 - j, True, j == 0, j == n - 1, kinds[n - 1 - j] == "wobble")
                                         for j, b in enumerate(other)]))
    s = Structure()
    coords: List[np.ndarray] = []
    for chain, first, residues in strands:
        seg = []
        for k, (b, level, is_partner, five, three, wob) in enumerate(residues):
            res = Residue(b, chain, first + k, chirality=chirality)
            for name, xyz in _strand_residue(b, level, is_partner, five, three, phosphate5, hydrogens, wob):
                res.atoms.append(Atom("ATOM", name, "", "", _element(name)))
                coords.append(xyz)
            seg.append(res)
        s.segments.append(seg)
    xyz = np.array(coords, dtype=np.float64)
    if chirality == "L":
        xyz[:, 0] *= -1.0  # the mirror image: left-handed helix of L-nucleotides
    if center:
        xyz -= np.array([0.0, 0.0, 0.5 * (xyz[:, 2].min() + xyz[:, 2].max())])
    for atom, x in zip((a for r in s.residues() for a in r.atoms), xyz):
        atom.tail = _tail(x)
    out_notes = list(msgs)
    renamed = [n_ for n_ in pdbtools.convert_names(s, scheme, terminals=terminals)
               if not n_.startswith("WARNING: 5'-terminal")]
    if chosen is None:
        renamed = ["Residue names " + ("A, C, G, U" if chirality == "D" else
                                       "0A, 0C, 0G, 0U (the CCD L-ribonucleotides)") + ", ATOM records"]
    out_notes += renamed
    if scheme == "pdb":
        out_notes += pdbtools.set_records(s, "pdb")
    if phosphate5 and scheme == "amber":
        out_notes.append("WARNING: the Amber 5' units have no 5'-phosphate (tleap cannot type P/OP1/OP2/OP3 "
                         "of A5/LA5 ...); build without the 5'-phosphate for tleap")
    hand = "right-handed, D" if chirality == "D" else "left-handed, L: mirror image"
    what = f"duplex of {n} base pairs" if other is not None else f"single strand of {n} nucleotides"
    s.header = [f"REMARK 951 MDATK IDEAL A-FORM RNA ({hand}), {what}",
                f"REMARK 951 ARNOTT A-RNA FIBER MONOMER, TWIST {TWIST:.1f} DEG, RISE {RISE:.2f} A, "
                f"HELIX AXIS Z",
                f"REMARK 951 STRAND {chain1} {seq}"]
    if other is not None:
        s.header.append(f"REMARK 951 STRAND {chain2} {other}")
        out_notes += _mismatch_notes(s, seq, other, kinds)
        if kinds.count("wobble"):
            out_notes.append(f"{kinds.count('wobble')} G·U pair(s) built with an approximate wobble geometry "
                             "(bases turned in their plane about C1'); minimise before production")
    out_notes.append(f"Built an ideal A-form RNA {what} ({hand}): {sum(len(r.atoms) for r in s.residues())} "
                     f"atoms{', hydrogens included' if hydrogens else ', heavy atoms only'}")
    if notes is not None:
        notes.extend(out_notes)
    return pdbtools.write_pdb(s)


def _mismatch_notes(s: Structure, seq: str, other: str, kinds: Sequence[str]) -> List[str]:
    segs = s.segments
    out = []
    n = len(seq)
    for k, kind in enumerate(kinds):
        if kind != "mismatch":
            continue
        r1, r2 = segs[0][k], segs[1][n - 1 - k]
        b1 = [a for a in r1.atoms if a.name in BASE_NAMES[seq[k]]]
        b2 = [a for a in r2.atoms if a.name in BASE_NAMES[other[n - 1 - k]]]
        d = min(np.linalg.norm(np.subtract(a.xyz, b.xyz)) for a in b1 for b in b2)
        if d < CLASH:
            out.append(f"WARNING: mismatch {r1.chain}:{r1.resseq} {seq[k]} · {r2.chain}:{r2.resseq} "
                       f"{other[n - 1 - k]}: the bases collide (closest heavy atoms {d:.1f} Å)")
    return out


# --- mutations ----------------------------------------------------------------------------------------

_MUTATION_RE = re.compile(r"^([^:\s]?):?(-?\d+)([A-Za-z]?)(?:>|:|->)?([ACGUacgu])$")
_ALIASES = {"A": "ADE", "C": "CYT", "G": "GUA", "U": "URA"}
_SUGAR_EXTRA = frozenset({"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "HOP2", "HOP3", "H5T", "H3T"})
_ALL_BASE_NAMES = frozenset(n for names in BASE_NAMES.values() for n in names) | frozenset(
    h for b in "ACGU" for h, *_ in _H_RULES[b]) | {"H62", "H22", "H42", "C7", "H71", "H72", "H73"}


@dataclass
class Mutation:
    chain: str
    resseq: int
    icode: str
    base: str
    explicit: bool = True
    reason: str = ""

    @property
    def where(self) -> str:
        return f"{self.chain or '_'}:{self.resseq}{self.icode}"


def parse_mutations(spec) -> List[Mutation]:
    """'A:5G, B:20C' (also 'A:5>G', ':5G' for a blank chain, a list of such strings) -> mutations."""
    items = re.split(r"[,;\s]+", spec.strip()) if isinstance(spec, str) else list(spec)
    out: List[Mutation] = []
    for item in items:
        if not item:
            continue
        m = _MUTATION_RE.match(item.strip())
        if not m:
            raise BuildError(f"cannot read mutation {item!r} (expected e.g. A:5G - chain, residue number, "
                             "new base)")
        chain, num, icode, base = m.groups()
        mut = Mutation(chain, int(num), icode, base.upper())
        if any((o.chain, o.resseq, o.icode) == (mut.chain, mut.resseq, mut.icode) for o in out):
            raise BuildError(f"residue {mut.where} is mutated twice")
        out.append(mut)
    if not out:
        raise BuildError("no mutations given (e.g. A:5G, B:20C)")
    return out


def mutated_name(name: str, base: str) -> str:
    """Residue name for ``base`` in the scheme, chirality and terminal state of ``name``."""
    up = name.strip().upper()
    if up in _ALIASES.values() or up == "URI":
        return _ALIASES[base]
    m = re.match(r"^([RL0]?)([ACGU])([53N]?)$", up)
    if not m:
        raise BuildError(f"cannot rename residue {name!r} (RNA nucleotides only)")
    return m.group(1) + base + m.group(3)


def _nucleotide(res: Residue) -> Optional[pdbtools.ResidueName]:
    """Name-based nucleotide recognition guarded by backbone atoms (a lanthanum ion 'LA' is no RNA)."""
    info = pdbtools.parse_residue_name(res.name)
    if info is None or not any("'" in a.name or "*" in a.name or a.name == "P" for a in res.atoms):
        return None
    return info


def _residue_index(s: Structure) -> Tuple[List[Residue], List[Dict[str, int]], np.ndarray]:
    residues = list(s.residues())
    index, rows, n = [], [], 0
    for res in residues:
        coords = res.coords()
        index.append({name: n + k for k, name in enumerate(coords)})
        rows.append(np.array(list(coords.values()), dtype=np.float64).reshape(-1, 3))
        n += len(coords)
    xyz = np.concatenate(rows) if rows else np.zeros((0, 3))
    return residues, index, xyz


def _partners(s: Structure, cutoff: float = 3.5) -> Dict[int, Tuple[int, int]]:
    """id(residue) -> (index of the partner residue, pair kind) for canonical pairs in ``s``."""
    from .secstruct import PairDetector

    residues, index, xyz = _residue_index(s)
    names = [r.name if _nucleotide(r) else "" for r in residues]
    pairs = PairDetector(names, index, cutoff=cutoff, wobble=True).detect(xyz)
    out = {}
    for i, j, kind in pairs:
        out[id(residues[i])] = (int(j), int(kind))
        out[id(residues[j])] = (int(i), int(kind))
    return out


def _swap_base(res: Residue, base: str) -> Tuple[List[str], float]:
    """Replace the base of ``res`` in place; returns notes and the RMS of the glycosidic-frame fit."""
    from .spatial import kabsch

    info = pdbtools.parse_residue_name(res.name)
    old = info.base
    coords = res.coords()
    old_frame = ["C1'"] + list(GLYCOSIDIC[old])
    missing = [a for a in old_frame if a not in coords]
    if missing:
        raise BuildError(f"{res.label}: missing atom(s) {', '.join(missing)} of the glycosidic frame")
    before, _ = pdbtools.sugar_chirality(coords, old)
    t = template(base, hydrogens=False)
    new_frame = ["C1'"] + list(GLYCOSIDIC[base])
    mob = np.array([t[a] for a in new_frame])
    ref = np.array([coords[a] for a in old_frame])
    weights = np.array([2.0, 2.0, 1.0, 1.0])  # the glycosidic bond itself counts double
    rot, mc, rc = kabsch(mob, ref, weights)
    rms = float(np.sqrt((((mob - mc) @ rot.T + rc - ref) ** 2).sum(axis=1).mean()))
    # record, B-factor and segid of the new base atoms come from the C1' the fit used (coords() takes the
    # first alternate location); the new base is one conformer with full occupancy
    template_c1 = next((a for a in res.atoms if a.name.replace("*", "'") == "C1'"), res.atoms[0])
    altlocs = sorted({a.alt for a in res.atoms if a.alt})
    keep, dropped_h, unknown = [], [], []
    insert_at = None
    for a in res.atoms:
        name = a.name.replace("*", "'")
        is_sugar = "'" in name or name in _SUGAR_EXTRA
        if is_sugar:
            keep.append(a)
        elif name in _ALL_BASE_NAMES or name in BASE_NAMES[old]:
            if insert_at is None:
                insert_at = len(keep)
            if name.lstrip("0123456789").startswith("H"):
                dropped_h.append(name)
        else:
            unknown.append(name)
            keep.append(a)
    if insert_at is None:
        insert_at = len(keep)
    record = template_c1.record
    new_atoms = []
    for name in BASE_NAMES[base]:
        xyz = (t[name] - mc) @ rot.T + rc
        tail = template_c1.tail[:3] + _tail(xyz)[3:33] + template_c1.tail[33:]  # occupancy 1.00, C1' B-factor
        new_atoms.append(Atom(record, name, "", tail, _element(name)))
    res.atoms = keep[:insert_at] + new_atoms + keep[insert_at:]
    old_name = res.name
    res.name = mutated_name(res.name, base)
    after, vols = pdbtools.sugar_chirality(res.coords(), base)
    if before is not None and after != before:
        raise BuildError(f"{res.label}: the base swap changed the sugar chirality call ({before} -> {after}, "
                         f"{vols})")
    notes = [f"{res.chain or '_'}:{res.resseq}{res.icode} {old}->{base} ({old_name} -> {res.name}, "
             f"glycosidic-frame fit {rms:.2f} Å)"]
    if dropped_h:
        notes.append(f"{res.chain or '_'}:{res.resseq}{res.icode}: dropped {len(dropped_h)} hydrogen(s) of the "
                     f"old base ({' '.join(dropped_h)}); the new base has heavy atoms only")
    if unknown:
        notes.append(f"WARNING: {res.label}: kept unrecognised atom(s) {' '.join(unknown)}")
    if altlocs:
        notes.append(f"{res.chain or '_'}:{res.resseq}{res.icode}: alternate locations {','.join(altlocs)}; the new "
                     f"base is fitted on {template_c1.alt or altlocs[0]} and written as one conformer with "
                     "occupancy 1.00")
    return notes, rms


def _new_partner_base(new: str, current: str, wobble: bool) -> str:
    if wobble and new + current in WOBBLE_PAIRS:
        return current
    return COMPLEMENT[new]


def mutate(source, mutations, compensatory: bool = True, wobble: bool = False, cutoff: float = 3.5
           ) -> Tuple[str, List[str]]:
    """Swap bases keeping the glycosidic frame; returns (PDB text, notes).

    ``mutations``: 'A:5G, B:20C' (or :class:`Mutation` objects).  ``compensatory``: also change the
    Watson-Crick / G·U partner (detected in the input) of every mutated base to the complement
    (G<->C, A<->U; with ``wobble`` a G·U pair is kept where the new base allows it) unless the
    partner is mutated explicitly.
    """
    s = pdbtools.read_pdb(source)
    notes = list(s.notes)
    muts = list(mutations) if not isinstance(mutations, str) else parse_mutations(mutations)
    chir_notes = pdbtools.assign_chirality(s, detect=True)
    by_key = {(r.chain, r.resseq, r.icode): r for r in s.residues()}
    targets: List[Tuple[Residue, Mutation]] = []
    for m in muts:
        res = by_key.get((m.chain, m.resseq, m.icode))
        if res is None:
            raise BuildError(f"residue {m.where} not found")
        info = _nucleotide(res)
        if info is None or info.sugar != "rna":
            raise BuildError(f"{res.label} is not an RNA nucleotide")
        targets.append((res, m))
    if compensatory:
        partner_of = _partners(s, cutoff)
        residues = list(s.residues())
        explicit = {id(r) for r, _ in targets}
        for res, m in list(targets):
            hit = partner_of.get(id(res))
            if hit is None:
                notes.append(f"{m.where}: no Watson-Crick / G·U partner in the input; no compensatory change")
                continue
            other = residues[hit[0]]
            other_info = _nucleotide(other)
            if other_info is None or other_info.sugar != "rna":
                notes.append(f"{m.where}: its partner {other.chain}:{other.resseq} {other.name} is not an RNA "
                             "nucleotide; no compensatory change")
                continue
            other_base = other_info.base
            if id(other) in explicit:
                new_other = next(mm.base for rr, mm in targets if rr is other)
                pair = m.base + new_other
                if pair not in WATSON_CRICK_PAIRS and pair not in WOBBLE_PAIRS:
                    notes.append(f"WARNING: {m.where} {m.base} and its partner {other.chain}:{other.resseq} "
                                 f"{new_other} (both mutated explicitly) do not pair canonically")
                continue
            new_base = _new_partner_base(m.base, other_base, wobble)
            if new_base == other_base:
                notes.append(f"{m.where} {m.base}: partner {other.chain}:{other.resseq} {other_base} already "
                             f"pairs ({'G·U wobble' if m.base + other_base in WOBBLE_PAIRS else 'Watson-Crick'})")
                continue
            comp = Mutation(other.chain, other.resseq, other.icode, new_base, explicit=False,
                            reason=f"compensates {m.where} {m.base}")
            targets.append((other, comp))
            explicit.add(id(other))
    done = 0
    for res, m in targets:
        old = pdbtools.parse_residue_name(res.name).base
        if old == m.base:
            notes.append(f"{m.where}: already {m.base}; unchanged")
            continue
        change, _rms = _swap_base(res, m.base)
        if not m.explicit:
            change[0] += f" - compensatory, {m.reason}"
        notes += change
        done += 1
    stale = [line for line in s.header if line.startswith("REMARK 950")]
    if stale:
        s.header = [line for line in s.header if not line.startswith("REMARK 950")]
        notes.append(f"Dropped {len(stale)} REMARK 950 base-pair annotation lines (they describe the old bases)")
    if s.conect:
        notes.append(f"Stripped {len(s.conect)} CONECT records")
    notes.append(f"Mutated {done} residue(s)")
    notes += [n for n in chir_notes if n.startswith("WARNING")]
    return pdbtools.write_pdb(s), notes


# --- rigid-body fit -----------------------------------------------------------------------------------

def _atom_element(atom: Atom, resname: str) -> str:
    """Element of an atom: the element column, else the name (an ion named after its residue keeps both
    letters, e.g. MG)."""
    if atom.element:
        return atom.element
    bare = re.sub(r"[\d+\-']", "", atom.name).upper()
    if len(bare) == 2 and bare == re.sub(r"[\d+\-]", "", resname).upper():
        return bare
    return bare[:1]


def model_atoms(structure: Structure, heavy_only: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Coordinates and elements of the atoms used for fitting (first alternate location only)."""
    xyz, elements = [], []
    for res in structure.residues():
        for atom in res.atoms:
            if atom.alt not in ("", "A", "1"):
                continue
            el = _atom_element(atom, res.name).upper()
            if heavy_only and el in ("H", "D"):
                continue
            xyz.append(atom.xyz)
            elements.append(el)
    return np.array(xyz, dtype=np.float64).reshape(-1, 3), elements


def _transform_anisou(line: str, rot: np.ndarray) -> str:
    try:
        u = [int(line[28 + 7 * k:35 + 7 * k]) for k in range(6)]
    except ValueError:
        return line
    m = np.array([[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]], dtype=np.float64)
    m = rot @ m @ rot.T
    vals = [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]
    return line[:28] + "".join(f"{int(round(v)):7d}" for v in vals) + line[70:]


def transform_pdb_text(text: str, rotation: np.ndarray, translation: np.ndarray) -> Optional[str]:
    """Apply x' = R x + t to every ATOM/HETATM (and rotate ANISOU) line; all other text is kept.

    Returns None when an ATOM/HETATM line has no fixed-column coordinates (damaged file)."""
    rot = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    out = []
    for line in text.splitlines(keepends=True):
        rec = line[:6]
        if rec in ("ATOM  ", "HETATM"):
            body = line.rstrip("\r\n")
            try:
                xyz = np.array([float(body[30:38]), float(body[38:46]), float(body[46:54])])
            except ValueError:
                return None
            new = rot @ xyz + t
            if new.min() <= -999.9995 or new.max() >= 9999.9995:
                raise BuildError("the fitted coordinates do not fit the PDB columns")
            end = line[len(body):]
            line = body[:30] + f"{new[0]:8.3f}{new[1]:8.3f}{new[2]:8.3f}" + body[54:] + end
        elif rec == "ANISOU":
            end = line[len(line.rstrip("\r\n")):]
            line = _transform_anisou(line.rstrip("\r\n").ljust(70), rot).rstrip() + end
        out.append(line)
    return "".join(out)


def fit_to_map(source, map_path: str, resolution: float, search: str = "global", angular_step: float = 30.0,
               heavy_only: bool = True, progress: Optional[Callable[[float, str], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None) -> Tuple[str, float, List[str]]:
    """Rigid-body fit of a model into a density map; returns (PDB text, correlation, notes).

    ``search``: 'global' (exhaustive rotations x FFT translations, then local refinement - as 'mdatk
    density fit') or 'local' (refine the model as placed).  Only coordinates change: the written text
    is the input with transformed ATOM/HETATM (and ANISOU) records, every other line verbatim.  An ideal
    self-complementary duplex is two-fold symmetric: the global search may return its strand-swapped pose.
    """
    from .density import FitError, fit_model, load_map_cached, rotation_angle
    from .io.mrc import MapError

    if search not in SEARCHES:
        raise BuildError(f"search must be one of {', '.join(SEARCHES)}")
    if not resolution or resolution <= 0:
        raise BuildError("enter the map resolution (Å)")
    if isinstance(source, os.PathLike) or (isinstance(source, str) and "\n" not in source
                                           and os.path.isfile(source)):
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = str(source)
    s = pdbtools.read_pdb(text)
    xyz, elements = model_atoms(s, heavy_only)
    if len(xyz) < 3:
        raise BuildError("the model has fewer than three atoms to fit")
    try:
        dmap = load_map_cached(map_path)
    except (OSError, MapError) as exc:
        raise BuildError(f"density map: {exc}") from exc
    notes = list(s.notes)
    centre = xyz.mean(axis=0)
    inside = np.all(centre >= dmap.origin) and np.all(centre <= dmap.extent)
    if search == "local" and not inside:
        notes.append("WARNING: the model centre lies outside the map box; a local fit needs a rough "
                     "placement first (use the global search)")
    try:
        res = fit_model(dmap, xyz, elements, resolution, start=None,
                        search="global" if search == "global" else "manual", angular_step=angular_step,
                        progress=progress, cancelled=cancelled)
    except FitError as exc:
        raise BuildError(f"density fit: {exc}") from exc
    rot, trans = res.transform.rotation, res.transform.translation
    new_text = transform_pdb_text(text, rot, trans)
    if new_text is None:  # columns damaged (e.g. ChimeraX): rewrite through pdbtools
        moved = res.transform.apply(np.array([a.xyz for r in s.residues() for a in r.atoms]))
        k = 0
        for r in s.residues():
            for a in r.atoms:
                a.tail = a.tail[:3] + _tail(moved[k])[3:27] + a.tail[27:]
                a.anisou = ""
                k += 1
        new_text = pdbtools.write_pdb(s, strip_conect=False)
        notes.append("Rewrote the file in standard PDB columns (the input had damaged ATOM records); "
                     "ANISOU records dropped")
    shift = float(np.linalg.norm(res.transform.apply(centre[None, :])[0] - centre))
    label = "global search" if search == "global" else "local refinement"
    notes.append(f"Density fit ({label}) into {os.path.basename(map_path)} at {resolution:g} Å, "
                 f"{len(xyz)} atoms: CC {res.cc:.3f}" + (f" ({res.message})" if res.message else ""))
    notes.append(f"Model moved by a {rotation_angle(rot):.1f}° rotation and a {shift:.1f} Å shift of its centre")
    if any(line.startswith("CRYST1") for line in text.splitlines()):
        notes.append("CRYST1 / SCALE records kept as they were (they may not describe the map's cell)")
    return new_text, float(res.cc), notes


# --- command line ('mdatk build') ---------------------------------------------------------------------

@dataclass
class BuildOutcome:
    text: str
    notes: List[str] = field(default_factory=list)
    cc: Optional[float] = None


def add_commands(sub) -> list:
    """Register 'helix', 'mutate' and 'fit' on an argparse sub-parser collection (returns the parsers)."""
    h = sub.add_parser("helix", help="ideal A-form RNA helix (D or L), single strand or duplex")
    h.add_argument("--seq", "--sequence", dest="seq", required=True, help="strand 1, 5'->3' (e.g. GGCAU)")
    h.add_argument("--partner", default="auto", help="auto (Watson-Crick complement), none (single strand) or "
                   "the partner sequence 5'->3' (same length; mismatches / G·U allowed)")
    h.add_argument("--chirality", choices=("D", "L"), default="D", help="D (right-handed) or L (mirror image)")
    h.add_argument("--chains", default="A,B", help="chain ids of strand 1 and 2 (default A,B)")
    h.add_argument("--start", default="1,1", help="first residue numbers of strand 1 and 2 (default 1,1; "
                                                  "write negative numbers as --start=-5,1)")
    h.add_argument("--scheme", choices=SCHEMES, help="residue names (default: A/C/G/U and 0A/0C/0G/0U, "
                   "no terminal variants)")
    h.add_argument("--no-terminal-names", action="store_true", help="no 5'/3' residue name variants")
    h.add_argument("--phosphate5", action="store_true", help="keep a 5'-phosphate (P, OP1, OP2, OP3)")
    h.add_argument("--hydrogens", action="store_true", help="add all hydrogens (default: heavy atoms only)")
    h.add_argument("--output", "-o", required=True, help="output PDB")
    m = sub.add_parser("mutate", help="swap bases keeping the glycosidic frame (with compensatory partners)")
    m.add_argument("input", help="PDB file")
    m.add_argument("mutations", help="e.g. 'A:5G, B:20C'")
    m.add_argument("--no-compensatory", action="store_true", help="do not change the paired partners")
    m.add_argument("--wobble", action="store_true", help="keep G·U pairs where the new base allows them")
    m.add_argument("--output", "-o", required=True, help="output PDB")
    f = sub.add_parser("fit", help="rigid-body fit of a model into a density map (MRC / CCP4)")
    f.add_argument("input", help="model PDB")
    f.add_argument("map", help="density map")
    f.add_argument("--resolution", type=float, required=True, help="map resolution in Å")
    f.add_argument("--search", choices=SEARCHES, default="global", help="global search (default) or local "
                   "refinement of the model as placed")
    f.add_argument("--step", type=float, default=30.0, help="angular step of the global search (degrees)")
    f.add_argument("--output", "-o", required=True, help="output PDB")
    return [h, m, f]


def run(args, progress: Optional[Callable[[float, str], None]] = None) -> BuildOutcome:
    """Execute a parsed 'helix' / 'mutate' / 'fit' command and write its output file."""
    cmd = args.build_command
    if cmd == "helix":
        notes: List[str] = []
        text = build_helix(args.seq, args.partner, args.chirality, args.chains, args.start,
                           args.scheme or None, not args.no_terminal_names, args.hydrogens, args.phosphate5,
                           notes=notes)
        out = BuildOutcome(text, notes)
    elif cmd == "mutate":
        text, notes = mutate(args.input, args.mutations, compensatory=not args.no_compensatory,
                             wobble=args.wobble)
        out = BuildOutcome(text, notes)
    elif cmd == "fit":
        text, cc, notes = fit_to_map(args.input, args.map, args.resolution, args.search, args.step,
                                     progress=progress)
        out = BuildOutcome(text, notes, cc)
    else:
        raise BuildError(f"unknown build command {cmd!r}")
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(out.text)
    return out
