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

Other pairings and triplexes (``build_helix(pairing=..., triplex=...)``)
    'reverse-wc' (trans WC/WC), 'hoogsteen' (cis WC/Hoogsteen: A·U, G·C+ on the Hoogsteen edge of the
    strand-1 purines) and 'hh' (trans Hoogsteen/Hoogsteen A·A, as the poly(rA) duplex PDB 4JRD) give
    parallel duplexes, 'reverse-hoogsteen' (trans WC/Hoogsteen) an antiparallel one; a triplex adds a
    third strand on the Hoogsteen edge of the strand-1 purines of a Watson-Crick duplex ('hoogsteen':
    pyrimidine motif, parallel; 'reverse-hoogsteen': purine motif, antiparallel).  Every nucleotide is
    the C3'-endo / anti nucleotide of PDB 4JRD (embedded, CC0) carrying the Arnott base; partner bases are
    placed coplanar on the H-bond donors / acceptors of the two edges (``HBOND``, :func:`_pair_fit`).  The
    helix is the screw that repeats a reference level (A·U, A·A for hh; A·U-U / A·U-A for triplexes) with
    closed O3'-P links and no close contacts, hydrogens included (``CONTACT_FIT``; :func:`_helix_fit`; hh:
    the 4JRD screw itself); chi stays in the C3'-endo anti range (``CHI_RANGE``).  Mismatches start from
    the reference pair and are moved off the bases they would hit (:func:`_level_pose`).  A helix whose
    levels differ (other pairs, mismatches) is closed jointly: every level moves as a rigid body (the pairs
    keep their H-bonds and coplanar bases) and every nucleotide gets its own torsions
    (:func:`_close_levels`).  The notes report link gaps, H-bond distances, mismatches, close contacts
    (``CONTACT_REPORT``, hydrogens included even for a heavy-atom model) and the backbone torsions.  C+ is
    written as plain C.  These are starting models to be minimised.

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


def helix_kind(pairing: Optional[str] = "wc", triplex: Optional[str] = None) -> str:
    """Short description of a helix build for titles: 'A-form', 'parallel tHH', 'A-form + hoogsteen triplex'."""
    pairing = check_pairing(pairing)
    motif = check_triplex(triplex)
    if motif:
        return f"{motif} triplex"
    if pairing == "wc":
        return "A-form"
    return f"{'parallel' if PARALLEL[pairing] else 'antiparallel'} {PAIRING_LW[pairing]}"


def check_pairing(pairing: Optional[str]) -> str:
    """Normalised pairing name (None / '' -> 'wc'); raises :class:`BuildError` for unknown names."""
    value = (pairing or "wc").strip().lower()
    if value not in PAIRINGS:
        raise BuildError(f"unknown pairing {pairing!r} (choose from {', '.join(PAIRINGS)})")
    return value


def check_triplex(triplex: Optional[str]) -> Optional[str]:
    """Normalised triplex motif (None / '' / 'none' -> None)."""
    value = (triplex or "").strip().lower()
    if value in ("", "none", "no"):
        return None
    if value not in TRIPLEX_MOTIFS:
        raise BuildError(f"unknown triplex motif {triplex!r} (choose from none, {', '.join(TRIPLEX_MOTIFS)})")
    return value


def auto_partner(seq: str, pairing: str = "wc") -> str:
    """The partner strand (5'->3') of 'auto' for ``pairing``: the Watson-Crick complement for wc, the
    canonical partners (``AUTO_PARTNER``) running with strand 1 for parallel pairings."""
    mapped = "".join(AUTO_PARTNER[pairing][b] for b in seq)
    return mapped if PARALLEL[pairing] else mapped[::-1]


def resolve_partner(sequence: str, partner: str = "auto", pairing: str = "wc"
                    ) -> Tuple[str, Optional[str], List[str]]:
    """(sequence, partner sequence 5'->3' or None, notes) for ``partner`` 'auto', 'none' or a sequence."""
    pairing = check_pairing(pairing)
    seq, notes = parse_sequence(sequence)
    mode = "auto" if partner is None else partner.strip()
    if not mode:
        raise BuildError("type the partner sequence (5'->3'), or choose the Watson-Crick complement or a "
                         "single strand")
    if mode.lower() == "none":
        return seq, None, notes
    if mode.lower() == "auto":
        if pairing == "wc":
            return seq, complement(seq), notes + ["partner strand: Watson-Crick complement"]
        return seq, auto_partner(seq, pairing), notes + [
            f"partner strand: {PAIRING_LW[pairing]} partners of strand 1 ("
            + ", ".join(f"{a}·{b}" for a, b in AUTO_PARTNER[pairing].items()) + "), "
            + ("parallel" if PARALLEL[pairing] else "antiparallel")]
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


def _canonical(rule: str, a: str, b: str) -> bool:
    """Is ``a`` (strand 1 / the duplex purine) · ``b`` a canonical pair of ``rule``?  The Hoogsteen rules
    need the purine (its Hoogsteen edge) on strand 1."""
    return (rule, a, b) in _PAIR_HBONDS and (rule not in ("hoogsteen", "reverse-hoogsteen") or a in "AG")


def pairing_kinds(seq: str, other: str, pairing: str = "wc") -> List[Tuple[str, str, str]]:
    """(strand-1 base, partner base, kind) per level.  wc: kinds of :func:`pair_types`; other pairings:
    'canonical' or 'mismatch' (for the Hoogsteen pairings every strand-1 pyrimidine: the pairs use the
    Hoogsteen edge of a strand-1 purine).  Parallel pairings pair seq[i] with partner[i], antiparallel
    ones with partner[n-1-i]."""
    pairing = check_pairing(pairing)
    n = len(seq)
    partners = other if PARALLEL[pairing] else other[::-1]
    if pairing == "wc":
        return [(a, b, k) for a, b, k in zip(seq, partners, pair_types(seq, other))]
    out = [(a, b, "canonical" if _canonical(pairing, a, b) else "mismatch") for a, b in zip(seq, partners)]
    assert len(out) == n
    return out


def resolve_third(seq: str, third: str = "auto", motif: str = "hoogsteen") -> Tuple[str, List[str]]:
    """Third-strand sequence (5'->3') of a triplex and notes; 'auto' gives the canonical motif base for
    each strand-1 purine (``AUTO_THIRD``)."""
    mode = (third or "auto").strip()
    parallel = motif == "hoogsteen"
    if mode.lower() == "auto":
        mapped = "".join(AUTO_THIRD[motif][b] for b in seq)
        return (mapped if parallel else mapped[::-1]), [
            "third strand: " + ("U·A, C+·G" if parallel else "A·A, G·G") + " on the strand-1 purines, "
            + ("parallel" if parallel else "antiparallel") + " to strand 1"]
    other, notes = parse_sequence(mode, "third strand")
    if len(other) != len(seq):
        raise BuildError(f"the third strand has {len(other)} nucleotides but the sequence has {len(seq)}; "
                         "it must have the same length (5'->3')")
    return other, notes


def triplex_kinds(seq: str, third: str, motif: str) -> List[Tuple[str, str, str]]:
    """(strand-1 base, third-strand base, 'canonical' | 'mismatch') per level."""
    partners = third if motif == "hoogsteen" else third[::-1]
    return [(a, b, "canonical" if _canonical(motif, a, b) else "mismatch") for a, b in zip(seq, partners)]


def _pairing_messages(seq: str, other: str, pairing: str) -> List[Tuple[str, str]]:
    kinds = pairing_kinds(seq, other, pairing)
    lw = PAIRING_LW[pairing]
    out = [("info", f"{PAIRING_LABELS[pairing]}: {len(seq)} pairs, "
                    f"{sum(k != 'mismatch' for *_x, k in kinds)} canonical {lw}, "
                    f"{sum(k == 'mismatch' for *_x, k in kinds)} mismatched")]
    mism = [f"{i + 1}:{a}·{b}" for i, (a, b, k) in enumerate(kinds) if k == "mismatch"]
    if mism:
        out.append(("warning", f"mismatched pairs (no {lw} pair) start from the reference "
                               f"{'·'.join(REFERENCE_PAIR[pairing])} pair, are moved off the bases they would hit "
                               "and are reported: " + ", ".join(mism[:8])
                    + (" ..." if len(mism) > 8 else "")))
    pyr = [f"{i + 1}:{a}·{b}" for i, (a, b, k) in enumerate(kinds) if a in "CU"]
    if pairing in ("hoogsteen", "reverse-hoogsteen") and pyr:
        out.append(("warning", "strand 1 must be the purine strand: a Hoogsteen pair needs the Hoogsteen edge of a "
                               "strand-1 purine, so these pairs are built as mismatches: " + ", ".join(pyr[:8])
                    + (" ..." if len(pyr) > 8 else "")))
    if any(a + b in ("GC", "CG") for a, b, k in kinds if k != "mismatch" and pairing == "hoogsteen"):
        out.append(("info", "G·C+ Hoogsteen pairs need N3-protonated cytosine (C+); written as plain C "
                            "(protonate or restrain for the simulation)"))
    return out


def check_helix(sequence: str, partner: str = "auto", pairing: str = "wc", triplex: Optional[str] = None,
                third: str = "auto") -> List[Tuple[str, str]]:
    """Sequence-level messages [(level, text)] (levels 'error', 'warning', 'info')."""
    try:
        pairing = check_pairing(pairing)
        motif = check_triplex(triplex)
        seq, other, notes = resolve_partner(sequence, partner, pairing)
        if other is None and (pairing != "wc" or motif):
            raise BuildError("a single strand has no pairing: choose a partner strand (or pairing wc)"
                             if not motif else "a triplex needs the Watson-Crick duplex: choose a partner strand")
        if motif and pairing != "wc":
            raise BuildError("a triplex is built on the Watson-Crick duplex: choose pairing wc")
        third_seq, third_notes = resolve_third(seq, third, motif) if motif else ("", [])
    except BuildError as exc:
        return [("error", str(exc))]
    out = [("info", n) for n in notes]
    if other is None:
        out.append(("info", f"single strand of {len(seq)} nucleotides in the A-form helix geometry"))
        return out
    if pairing != "wc":
        out += _pairing_messages(seq, other, pairing)
        if (partner or "auto").strip().lower() not in ("auto", "none") and other == auto_partner(seq, pairing):
            out.append(("info", f"the explicit partner is the {PAIRING_LW[pairing]} partner strand of 'auto'"))
        return out
    if motif:
        out += [("info", n) for n in third_notes]
        kinds = triplex_kinds(seq, third_seq, motif)
        good = sum(k == "canonical" for *_x, k in kinds)
        out.append(("info", f"triplex ({TRIPLEX_LABELS[motif]}): {good} canonical base triple(s), "
                            f"{len(kinds) - good} mismatched"))
        mism = [f"{i + 1}:{b}·{a}" for i, (a, b, k) in enumerate(kinds) if k == "mismatch"]
        if mism:
            out.append(("warning", "third-strand mismatches (no Hoogsteen-edge pair; strand-1 pyrimidines have "
                                   "no purine Hoogsteen edge) start from the reference triple, are moved off the "
                                   "bases they would hit and are reported: "
                        + ", ".join(mism[:8]) + (" ..." if len(mism) > 8 else "")))
        if motif == "hoogsteen" and any(a == "G" and b == "C" for a, b, k in kinds if k == "canonical"):
            out.append(("info", "C+·G-C triples need N3-protonated cytosine (C+) in the third strand; written "
                                "as plain C (protonate or restrain for the simulation)"))
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


def _parse_ids(value, count: int, kind=str) -> Tuple:
    """Two values as :func:`_parse_pair`, or three for a triplex (a missing third: next free chain id /
    residue 1)."""
    if count == 2:
        return _parse_pair(value, kind)
    parts = [p.strip() for p in value.split(",")] if isinstance(value, str) else (
        [value] if isinstance(value, (int, float)) else list(value))
    if len(parts) == 3:
        try:
            return tuple(kind(p) for p in parts)
        except ValueError:
            raise BuildError(f"cannot read {value!r}") from None
    if len(parts) > 3:
        raise BuildError(f"expected three values (strand 1, 2, 3), got {value!r}")
    two = _parse_pair(parts if len(parts) == 2 else parts[0], kind)
    if kind is str:
        return two + (next(c for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if c not in two),)
    return two + (kind(1),)


def build_helix(sequence: str, partner: str = "auto", chirality: str = "D", chains=("A", "B"),
                start=(1, 1), scheme: Optional[str] = None, terminals: bool = True,
                hydrogens: bool = False, phosphate5: bool = False, center: bool = True,
                notes: Optional[List[str]] = None, pairing: str = "wc", triplex: Optional[str] = None,
                third: str = "auto") -> str:
    """PDB text of an ideal RNA helix: one strand (``partner='none'``), a duplex or a triplex.

    ``partner``: 'auto' (the partners of ``pairing``: the Watson-Crick complement for wc), 'none' or the
    partner sequence 5'->3' (same length; mismatches and G·U wobbles allowed).  ``pairing`` (``PAIRINGS``):
    'wc' (default) the antiparallel Arnott A-form duplex; 'reverse-wc' (trans WC/WC), 'hoogsteen' (cis
    WC/Hoogsteen, strand 1 the purine strand) and 'hh' (trans Hoogsteen/Hoogsteen A·A as PDB 4JRD) are
    parallel duplexes, 'reverse-hoogsteen' (trans WC/Hoogsteen) antiparallel; see :func:`_helix_strands`.
    ``triplex`` 'hoogsteen' / 'reverse-hoogsteen' adds a third strand (``third``: 'auto' or 5'->3') on the
    Hoogsteen edge of the strand-1 purines of the Watson-Crick duplex.
    ``chirality`` 'D' (right-handed) or 'L' (the mirror image).
    ``chains`` / ``start``: chain ids and first residue numbers of strand 1 and 2 (and 3: also 'A,B,C' /
    '1,1,1').  ``scheme``: residue names as in :mod:`mdatk.pdbtools` (None: ``DEFAULT_SCHEME``);
    ``terminals``: 5'/3' variants for the schemes that have them (amber, ccd).  ``phosphate5`` keeps a
    5'-phosphate (P, OP1, OP2, OP3) on each strand; by default chains start with a 5'-OH.
    ``hydrogens`` adds all hydrogens; ``center`` puts the helix centre at the origin (axis along z).
    ``notes`` (a list) receives the build notes.
    """
    pairing = check_pairing(pairing)
    motif = check_triplex(triplex)
    seq, other, msgs = resolve_partner(sequence, partner, pairing)
    chirality = (chirality or "D").strip().upper()
    if chirality not in ("D", "L"):
        raise BuildError(f"chirality must be D or L, got {chirality!r}")
    if other is None and pairing != "wc":
        raise BuildError("a single strand has no pairing: choose a partner strand (or pairing wc)")
    if motif and (other is None or pairing != "wc"):
        raise BuildError("a triplex is built on the Watson-Crick duplex: choose a partner strand and pairing wc")
    third_seq, third_msgs = resolve_third(seq, third, motif) if motif else ("", [])
    msgs += third_msgs
    ids = _parse_ids(chains, 3 if motif else 2)
    firsts = _parse_ids(start, 3 if motif else 2, int)
    chain1, chain2, first1, first2 = ids[0], ids[1], firsts[0], firsts[1]
    if any(len(c) > 1 for c in ids):
        raise BuildError("chain ids must be single characters")
    if other is not None and chain1 == chain2:
        raise BuildError("the two strands need different chain ids")
    if motif and ids[2] in (chain1, chain2):
        raise BuildError("the three strands need different chain ids")
    chosen = scheme or None
    scheme = scheme or DEFAULT_SCHEME
    if scheme not in SCHEMES:
        raise BuildError(f"unknown residue name scheme {scheme!r} (choose from {', '.join(SCHEMES)})")
    n = len(seq)
    kinds = pair_types(seq, other) if other is not None and pairing == "wc" else ["none"] * n
    # per strand: (chain, first residue number, [(base, atoms)]) 5'->3'
    built: List[Tuple[str, int, List[Tuple[str, List[Tuple[str, np.ndarray]]]]]] = []
    rebuilt = pairing != "wc" or motif is not None
    if not rebuilt:
        # (base, helix level, partner strand, 5' end, 3' end, wobble) per residue, 5'->3'
        strands = [(chain1, first1, [(b, k, False, k == 0, k == n - 1, kinds[k] == "wobble")
                                     for k, b in enumerate(seq)])]
        if other is not None:  # strand 2 pairs residue n-1-j of strand 1 at the same level
            strands.append((chain2, first2, [(b, n - 1 - j, True, j == 0, j == n - 1,
                                              kinds[n - 1 - j] == "wobble") for j, b in enumerate(other)]))
        for chain, first, residues in strands:
            built.append((chain, first, [(b, _strand_residue(b, level, is_partner, five, three, phosphate5,
                                                             hydrogens, wob))
                                         for b, level, is_partner, five, three, wob in residues]))
    else:
        kind = pairing if pairing != "wc" else "triplex-" + motif
        sequences = [seq, other] + ([third_seq] if motif else [])
        heavy, levels, fit = _helix_strands(kind, sequences)
        for chain, first, bases, strand in zip(ids, firsts, sequences, heavy):
            built.append((chain, first, [(b, _residue_atoms(h, b, j == 0, j == n - 1, phosphate5, hydrogens))
                                         for j, (b, h) in enumerate(zip(bases, strand))]))
        if hydrogens:
            _turn_terminal_hydroxyls(built)
    s = Structure()
    coords: List[np.ndarray] = []
    for chain, first, residues in built:
        seg = []
        for k, (b, atoms) in enumerate(residues):
            res = Residue(b, chain, first + k, chirality=chirality)
            for name, xyz in atoms:
                res.atoms.append(Atom("ATOM", name, "", "", _element(name)))
                coords.append(xyz)
            seg.append(res)
        s.segments.append(seg)
    xyz = np.array(coords, dtype=np.float64)
    if rebuilt:  # helix axis along z through the origin
        rot, shift = _axis_frame(fit.screw)
        xyz = xyz @ rot.T + shift
    if chirality == "L":
        xyz[:, 0] *= -1.0  # the mirror image: left-handed helix of L-nucleotides
    if center:
        xyz -= np.array([0.0, 0.0, 0.5 * (xyz[:, 2].min() + xyz[:, 2].max())])
    for atom, x in zip((a for r in s.residues() for a in r.atoms), xyz):
        atom.tail = _tail(x)
    geometry = _model_notes(s, levels, kind) if rebuilt else []
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
    if motif:
        what = f"triplex of {n} base triples"
    if not rebuilt:
        s.header = [f"REMARK 951 MDATK IDEAL A-FORM RNA ({hand}), {what}",
                    f"REMARK 951 ARNOTT A-RNA FIBER MONOMER, TWIST {TWIST:.1f} DEG, RISE {RISE:.2f} A, "
                    f"HELIX AXIS Z",
                    f"REMARK 951 STRAND {chain1} {seq}"]
    else:
        form = (f"PAIRING {PAIRING_LW[pairing]} ({'PARALLEL' if PARALLEL[pairing] else 'ANTIPARALLEL'} STRANDS)"
                if not motif else f"WATSON-CRICK DUPLEX + {motif.upper()} THIRD STRAND")
        s.header = [f"REMARK 951 MDATK IDEAL RNA ({hand}), {what}, {form}",
                    f"REMARK 951 NUCLEOTIDE OF PDB 4JRD, TWIST {fit.screw.twist:.1f} DEG, RISE {fit.screw.rise:.2f} "
                    "A, HELIX AXIS Z",
                    f"REMARK 951 STRAND {chain1} {seq}"]
    if other is not None:
        s.header.append(f"REMARK 951 STRAND {chain2} {other}")
        if not rebuilt:
            out_notes += _mismatch_notes(s, seq, other, kinds)
            if kinds.count("wobble"):
                out_notes.append(f"{kinds.count('wobble')} G·U pair(s) built with an approximate wobble geometry "
                                 "(bases turned in their plane about C1'); minimise before production")
    if motif:
        s.header.append(f"REMARK 951 STRAND {ids[2]} {third_seq} (THIRD STRAND, "
                        f"{'PARALLEL' if motif == 'hoogsteen' else 'ANTIPARALLEL'} TO STRAND {chain1})")
    if rebuilt:
        which = (f"Pairing {pairing}: {PAIRING_LABELS[pairing]}" if not motif else
                 f"Triplex, {TRIPLEX_LABELS[motif]}; the Watson-Crick duplex and the third strand are fitted "
                 "together (the Arnott A-form duplex leaves no room for a third strand)")
        out_notes.append(f"{which}; helix twist {fit.screw.twist:.1f}°, rise {fit.screw.rise:.2f} Å, nucleotides "
                         "(C3'-endo, anti) from PDB 4JRD with backbone torsions fitted to close the O3'-P links "
                         f"(reference helix: fit residual {fit.cost:.2f}, closest contacts {fit.contact:.2f} Å heavy "
                         f"atoms, {fit.contact_h:.2f} Å H-H)")
    out_notes += geometry
    atoms_note = f"{sum(len(r.atoms) for r in s.residues())} atoms" + (
        ", hydrogens included" if hydrogens else ", heavy atoms only")
    if not rebuilt:
        out_notes.append(f"Built an ideal A-form RNA {what} ({hand}): {atoms_note}")
    else:
        kind_text = f"with {PAIRING_LW[pairing]} pairs" if not motif else f"with a {motif} third strand"
        out_notes.append(f"Built an RNA {what} {kind_text} ({hand}): {atoms_note}; a starting model: "
                         "minimise before production")
    if notes is not None:
        notes.extend(out_notes)
    return pdbtools.write_pdb(s)


def _turn_terminal_hydroxyls(built) -> None:
    """HO5' / HO3' of the strand ends of a rebuilt model (``build_helix``'s per-strand residue atom lists):
    the staggered position (C-C-O-H 180, 60 or -60 degrees) farthest from the other atoms (in place)."""
    from scipy.spatial import cKDTree

    owners, rows = [], []
    for si, (_chain, _first, residues) in enumerate(built):
        for ri, (_b, atoms) in enumerate(residues):
            for name, xyz in atoms:
                owners.append((si, ri, name))
                rows.append(xyz)
    tree = cKDTree(np.array(rows))
    for si, (_chain, _first, residues) in enumerate(built):
        for ri, (_b, atoms) in enumerate(residues):
            index = {name: k for k, (name, _x) in enumerate(atoms)}
            xyz = dict(atoms)
            for h, (a, b, o) in (("HO5'", ("C4'", "C5'", "O5'")), ("HO3'", ("C4'", "C3'", "O3'"))):
                if h not in index:
                    continue
                best = None
                for torsion in (180.0, 60.0, -60.0):
                    pos = _place(xyz[a], xyz[b], xyz[o], BOND_H["oh"], 109.5, torsion)
                    near = [np.linalg.norm(pos - rows[i]) for i in tree.query_ball_point(pos, 3.0)
                            if owners[i] not in ((si, ri, h), (si, ri, o), (si, ri, b))]
                    clear = min(near, default=9.0)
                    if best is None or clear > best[0] + 1e-6:
                        best = (clear, pos)
                atoms[index[h]] = (h, best[1])


def _model_notes(s: Structure, levels: Sequence[Sequence[_Level]], kind: str) -> List[str]:
    """Geometry report (:func:`_geometry_notes`) of a rebuilt duplex or triplex."""
    residues = list(s.residues())
    labels = [f"{r.chain}:{r.resseq}" for r in residues]
    bases = [pdbtools.parse_residue_name(r.name).base for r in residues]
    res_atoms = []
    for r, base in zip(residues, bases):
        atoms = [(a.name, np.array(a.xyz)) for a in r.atoms]
        if not any(name.startswith("H") for name, _x in atoms):  # heavy-atom model: hydrogens for the check
            atoms += list(_hydrogens(dict(atoms), base).items())
        res_atoms.append(atoms)
    offsets, k = [], 0
    for seg in s.segments:
        offsets.append(k)
        k += len(seg)
    segments = [list(range(o, o + len(seg))) for o, seg in zip(offsets, s.segments)]
    n = len(s.segments[0])
    spec = _HELICES[kind]
    pairs = []
    for strand, (st, lvls) in enumerate(zip(spec[1:], levels), start=1):
        ref = spec[0].base + "·" + st.base
        if st.rule in TRIPLEX_MOTIFS and kind.startswith("triplex"):
            what = "no {} third-strand pair of {} on {}{}; placed from the " + ref + " reference triple"
        elif st.rule == "wc":
            what = "no Watson-Crick pair{3}; placed from the " + ref + " reference pair"
        else:
            what = f"no {PAIRING_LW[st.rule]} pair" + "{3}; placed from the " + ref + " reference pair"
        for lv in lvls:
            j = lv.level if st.step > 0 else n - 1 - lv.level
            hoog = st.rule in ("hoogsteen", "reverse-hoogsteen") and lv.bases[0] in "CU"
            note = what.format(st.rule, lv.bases[1], lv.bases[0],
                               " (a strand-1 pyrimidine has no purine Hoogsteen edge)" if hoog else "")
            pairs.append((offsets[0] + lv.level, offsets[strand] + j, lv.hbonds, note))
    out = _geometry_notes(labels, res_atoms, segments, pairs, bases)
    out.append(_torsion_note(s))
    plus = [lv for st, lvls in zip(spec[1:], levels) for lv in lvls
            if st.rule == "hoogsteen" and lv.canonical and lv.bases in ("GC", "CG")]
    if plus:
        out.append(f"{len(plus)} G·C+ Hoogsteen pair(s): the cytosine is written as plain C (N3 unprotonated); "
                   "protonate N3 (C+) for the force field or restrain the pair (at neutral pH it is weak)")
    return out


def _torsion_note(s: Structure) -> str:
    """Circular means of the backbone torsions per strand (degrees), with the 4JRD (A-form) values."""
    parts = []
    for seg in s.segments:
        c = [{k: np.array(v) for k, v in r.coords().items()} for r in seg]
        values: Dict[str, List[float]] = {k: [] for k in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi")}
        for j, r in enumerate(c):
            base = pdbtools.parse_residue_name(seg[j].name).base
            n, x = GLYCOSIDIC[base][0], GLYCOSIDIC[base][1]
            values["chi"].append(float(_dihedral(r["O4'"], r["C1'"], r[n], r[x])))
            values["delta"].append(float(_dihedral(r["C5'"], r["C4'"], r["C3'"], r["O3'"])))
            if "P" in r:
                values["beta"].append(float(_dihedral(r["P"], r["O5'"], r["C5'"], r["C4'"])))
            values["gamma"].append(float(_dihedral(r["O5'"], r["C5'"], r["C4'"], r["C3'"])))
            if j:
                p = c[j - 1]
                values["alpha"].append(float(_dihedral(p["O3'"], r["P"], r["O5'"], r["C5'"])))
                values["epsilon"].append(float(_dihedral(p["C4'"], p["C3'"], p["O3'"], r["P"])))
                values["zeta"].append(float(_dihedral(p["C3'"], p["O3'"], r["P"], r["O5'"])))
        means = []
        for k, v in values.items():
            if v:
                a = np.radians(v)
                means.append(f"{k} {math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())):.0f}")
        parts.append(f"{seg[0].chain or '_'}: " + " ".join(means))
    return ("Backbone torsions (means, degrees; PDB 4JRD A-form: alpha -62 beta 174 gamma 58 delta 80 epsilon -146 "
            "zeta -66 chi -175): " + "; ".join(parts))


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


# --- other pairings: parallel duplexes, (reverse) Hoogsteen pairs, triplexes -------------------------

# The 4JRD nucleotide: chain A residue 10 of PDB 4JRD (poly(rA) parallel duplex, trans Hoogsteen/
# Hoogsteen A·A pairs; PDB entries are CC0), in the helix frame of the duplex: the screw that maps the
# duplex onto itself one level on (fitted to residues A3-A11 / B2-B10) has its axis along z, strand 1
# runs 5'->3' towards -z and the A·A pair is the two-fold rotation about z.  Cylindrical coordinates
# (name, r Å, phi degrees, z Å, z relative to C1').  Its backbone is A-form (alpha -62, beta 174, gamma
# 58, delta 80, epsilon -146, zeta -66, chi -175 degrees, means over the duplex).
JRD_TWIST = 41.75  # degrees per residue, PDB 4JRD (right-handed, 8.6 residues per turn)
JRD_RISE = 3.79  # Å per residue, PDB 4JRD
_JRD_NUCLEOTIDE = (
    ("P", 5.948, -140.91, -1.014), ("OP1", 6.933, -140.76, -2.139), ("OP2", 4.963, -129.29, -0.941),
    ("O5'", 5.454, -156.26, -0.936), ("C5'", 6.651, -164.00, -0.910), ("C4'", 6.524, -177.05, -0.866),
    ("O4'", 5.857, 177.97, 0.304), ("C3'", 5.846, 175.91, -2.008), ("O3'", 6.635, 176.76, -3.173),
    ("C2'", 5.981, 162.32, -1.435), ("O2'", 7.393, 160.78, -1.394), ("C1'", 5.466, 164.59, 0.000),
    ("N9", 4.036, 160.84, 0.078), ("C8", 3.057, 176.48, 0.099), ("N7", 1.812, 166.39, 0.014),
    ("C5", 2.462, 132.68, -0.065), ("C6", 2.746, 102.37, -0.241), ("N6", 2.372, 73.42, -0.327),
    ("N1", 4.088, 101.76, -0.349), ("C2", 4.880, 115.60, -0.274), ("N3", 4.884, 131.20, -0.123),
    ("C4", 3.764, 140.93, -0.034))
PAIRINGS = ("wc", "reverse-wc", "hoogsteen", "reverse-hoogsteen", "hh")
# Leontis-Westhof class of the pair as (strand-1 edge, partner edge) / strand orientation
PAIRING_LW = {"wc": "cWW", "reverse-wc": "tWW", "hoogsteen": "cHW", "reverse-hoogsteen": "tHW", "hh": "tHH"}
PARALLEL = {"wc": False, "reverse-wc": True, "hoogsteen": True, "reverse-hoogsteen": False, "hh": True}
PAIRING_LABELS = {
    "wc": "Watson-Crick (cis WC/WC), antiparallel A-form",
    "reverse-wc": "reverse Watson-Crick (trans WC/WC), parallel",
    "hoogsteen": "Hoogsteen (cis WC/Hoogsteen: A·U, G·C+), parallel",
    "reverse-hoogsteen": "reverse Hoogsteen (trans WC/Hoogsteen: A·U, A·A, G·G), antiparallel",
    "hh": "trans Hoogsteen/Hoogsteen A·A (as PDB 4JRD), parallel",
}
TRIPLEX_MOTIFS = ("hoogsteen", "reverse-hoogsteen")
TRIPLEX_LABELS = {
    "hoogsteen": "pyrimidine motif: Hoogsteen third strand parallel to the purine strand (U·A-U, C+·G-C)",
    "reverse-hoogsteen": "purine motif: reverse-Hoogsteen third strand antiparallel to the purine strand "
                         "(A·A-U, G·G-C, also U·A-U)",
}
# canonical pairs: (atom of the strand-1 / duplex base, atom of the partner, donor 1|2, hydrogen of the donor);
# 'H3+' is the N3 proton of C+ (written as plain C)
_PAIR_HBONDS = {
    ("wc", "A", "U"): (("N1", "N3", 2, "H3"), ("N6", "O4", 1, "H61")),  # the duplex of a triplex
    ("wc", "G", "C"): (("N1", "N3", 1, "H1"), ("N2", "O2", 1, "H21"), ("O6", "N4", 2, "H41")),
    ("wc", "G", "U"): (("O6", "N3", 2, "H3"), ("N1", "O2", 1, "H1")),
    ("reverse-wc", "A", "U"): (("N1", "N3", 2, "H3"), ("N6", "O2", 1, "H61")),
    ("reverse-wc", "G", "C"): (("N1", "O2", 1, "H1"), ("N2", "N3", 1, "H21")),
    ("hoogsteen", "A", "U"): (("N7", "N3", 2, "H3"), ("N6", "O4", 1, "H62")),
    ("hoogsteen", "G", "C"): (("N7", "N3", 2, "H3+"), ("O6", "N4", 2, "H41")),
    ("reverse-hoogsteen", "A", "U"): (("N7", "N3", 2, "H3"), ("N6", "O2", 1, "H62")),
    ("reverse-hoogsteen", "A", "A"): (("N7", "N6", 2, "H61"), ("N6", "N1", 1, "H62")),
    ("reverse-hoogsteen", "G", "G"): (("N7", "N1", 2, "H1"), ("O6", "N2", 2, "H21")),
    ("hh", "A", "A"): (("N6", "N7", 1, "H62"), ("N7", "N6", 2, "H62")),
}
for (_p, _a, _b), _bonds in list(_PAIR_HBONDS.items()):  # the same pairs with the strands swapped
    if _a != _b:
        _PAIR_HBONDS.setdefault((_p, _b, _a), tuple((y, x, 3 - side, h) for x, y, side, h in _bonds))
REFERENCE_PAIR = {"reverse-wc": "AU", "hoogsteen": "AU", "reverse-hoogsteen": "AU", "hh": "AA"}
# partner of each strand-1 base for partner='auto' / third strand 'auto' (purine strand = strand 1)
AUTO_PARTNER = {"wc": COMPLEMENT, "reverse-wc": COMPLEMENT, "hoogsteen": COMPLEMENT,
                "reverse-hoogsteen": {"A": "U", "G": "G", "U": "A", "C": "G"},
                "hh": {"A": "A", "C": "C", "G": "G", "U": "U"}}
AUTO_THIRD = {"hoogsteen": {"A": "U", "G": "C", "U": "U", "C": "C"},
              "reverse-hoogsteen": {"A": "A", "G": "G", "U": "A", "C": "G"}}
HBOND = 2.9  # Å, donor-acceptor distance of the built pairs
LINK = 1.607  # Å, O3'-P
LINK_RANGE = (1.5, 1.7)  # Å, O3'-P links outside are reported as gaps
CLASH_NB = 2.4  # Å, closest allowed distance of heavy atoms three bonds apart (1-4; farther: CONTACT_REPORT)
_SUGAR = ("C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'")
_SUGAR_H = ("H1'", "H2'", "H3'", "H4'", "HO2'")  # hydrogens fixed by the sugar ring (H5'/H5'' follow O5')
_RIGID = _SUGAR + _SUGAR_H
_JRD_CHI = -170.0  # degrees, chi of the nucleosides of _core (dchi 0)
# bond geometry of the rebuilt backbone (Parkinson et al. 1996 / PDB 4JRD means)
_GEOM = {"C5'-O5'": 1.443, "C4'-C5'-O5'": 109.9, "O5'-P": 1.592, "C5'-O5'-P": 119.6, "C3'-O3'-P": 119.7,
         "O3'-P-O5'": 104.0, "P-OP": 1.485, "OP-P-OP": 119.6}
# backbone torsions of the 4JRD duplex (degrees), the targets of the backbone closure
_TORSIONS = {"alpha": -62.0, "beta": 174.0, "gamma": 58.0, "epsilon": -146.0, "zeta": -66.0}
# starting values of the helix fits (see _helix_fit): axis offset x, y (Å, in the plane of the strand-1
# base from the mean C1' of the reference level), axis tilts (degrees), twist (degrees), rise (Å), then per
# strand the backbone (chi change, gamma, beta in degrees); refined when first used.  hh: the 4JRD
# helix itself, the backbone of both strands.
# twist (degrees) and rise (Å) the helix fits are pulled towards (the first fitted helices)
_SCREW_PRIOR = {"reverse-wc": (20.66, 3.77), "hoogsteen": (37.98, 3.48), "reverse-hoogsteen": (45.7, 4.21),
                "triplex-hoogsteen": (36.19, 2.9), "triplex-reverse-hoogsteen": (27.27, 4.18)}
_HELIX_START = {
    "hh": (1.00, 53.39, 172.39),
    "reverse-wc": (-10.46, -4.59, 54.04, 27.20, 22.08, 4.18, 27.75, 73.75, 121.99, 25.10, 99.98, 163.60),
    "hoogsteen": (3.60, -4.93, -26.01, 4.09, 36.63, 3.73, -5.87, 132.12, 165.56, 18.69, 117.52, 201.05),
    "reverse-hoogsteen": (-4.56, 0.76, 15.30, -16.81, 48.08, 4.39, -15.06, 162.79, 152.69, 20.07, 105.73, 215.23),
    "triplex-hoogsteen": (0.95, -5.35, -23.66, 0.26, 25.61, 4.85, 3.15, 142.15, 173.97, -5.92, 55.06, 170.43,
                          21.23, 135.34, 170.28),
    "triplex-reverse-hoogsteen": (-0.44, -4.42, -26.00, 5.62, 21.95, 5.06, 8.96, 155.86, 157.34, -4.39, 52.09,
                                  168.87, 18.85, 98.76, 116.35),
}


def _wrap(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product over the last axis (broadcasting; faster than np.cross for many small arrays)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


def _dihedral(a, b, c, d) -> np.ndarray:
    """Dihedral angles (degrees) of point arrays (..., 3)."""
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((_cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def _angle(a, b, c) -> np.ndarray:
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _place_many(a, b, c, bond: float, angle: float, torsion) -> np.ndarray:
    """:func:`_place` for arrays of atoms (n, 3) and torsions (n,)."""
    bc = c - b
    bc = bc / np.linalg.norm(bc, axis=-1, keepdims=True)
    n = _cross(b - a, bc)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    m = _cross(n, bc)
    th, ph = math.radians(angle), np.radians(torsion)[..., None]
    return c + bond * (-math.cos(th) * bc + math.sin(th) * np.cos(ph) * m + math.sin(th) * np.sin(ph) * n)


def _rotvecs(v: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) of rotation vectors (n, 3) in radians (Rodrigues)."""
    th = np.linalg.norm(v, axis=-1)
    k = v / np.where(th > 1e-12, th, 1.0)[:, None]
    kx = np.zeros((len(v), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    kx[:, 1, 0], kx[:, 2, 0], kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * kx + (1 - c) * kx @ kx


def _rotvec(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    return np.eye(3) if th < 1e-12 else _turn_matrix(v / th, math.degrees(th))


@dataclass(frozen=True)
class _Screw:
    """Helical screw: turn by -twist degrees about ``axis`` (through ``origin``) and move -rise along it
    per level (right-handed; strand 1 runs 5'->3' towards -axis)."""
    origin: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    twist: float
    rise: float

    def apply(self, xyz, level: float = 1) -> np.ndarray:
        o, u = np.array(self.origin), _unit(np.array(self.axis))
        rot = _turn_matrix(u, -self.twist * level)
        return (np.asarray(xyz, dtype=np.float64) - o) @ rot.T + o - self.rise * level * u


@lru_cache(maxsize=None)
def _core_template(base: str) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    """Nucleoside of ``base`` on the 4JRD nucleotide (4JRD helix frame): the 4JRD sugar and backbone with
    the Arnott base superposed on the 4JRD glycosidic frame; base hydrogens and the sugar hydrogens that
    move with the sugar ring (``_SUGAR_H``) included (C: also 'H3+', the N3 proton of C+)."""
    from .spatial import kabsch

    jrd = {n: _cartesian(r, p, z) for n, r, p, z in _JRD_NUCLEOTIDE}
    arn = template(base, hydrogens=False)
    old, new = ["C1'"] + list(GLYCOSIDIC["A"]), ["C1'"] + list(GLYCOSIDIC[base])
    rot, mc, rc = kabsch(np.array([arn[a] for a in new]), np.array([jrd[a] for a in old]),
                         np.array([2.0, 2.0, 1.0, 1.0]))
    heavy = {n: x for n, x in jrd.items() if n not in BASE_NAMES["A"]}
    heavy.update({n: (arn[n] - mc) @ rot.T + rc for n in BASE_NAMES[base]})
    keep = {h for h, *_ in _H_RULES[base]} | set(_SUGAR_H)
    hyd = {n: x for n, x in _hydrogens(heavy, base).items() if n in keep}
    if base == "C":
        n3 = heavy["N3"]
        hyd["H3+"] = n3 - BOND_H["nh"] * _unit(_unit(heavy["C2"] - n3) + _unit(heavy["C4"] - n3))
    return tuple((n, tuple(x)) for n, x in list(heavy.items()) + list(hyd.items()))


def _core(base: str) -> Dict[str, np.ndarray]:
    return {n: np.array(x) for n, x in _core_template(base)}


def _moved(coords: Dict[str, np.ndarray], rot: np.ndarray, shift: np.ndarray) -> Dict[str, np.ndarray]:
    return {n: x @ rot.T + shift for n, x in coords.items()}


def _base_plane(coords: Dict[str, np.ndarray], base: str) -> Tuple[np.ndarray, np.ndarray]:
    p = np.array([coords[n] for n in BASE_NAMES[base]])
    c = p.mean(axis=0)
    return c, np.linalg.svd(p - c)[2][2]


@lru_cache(maxsize=None)
def _pair_fit(pairing: str, b1: str, b2: str) -> Tuple[np.ndarray, np.ndarray]:
    """(rotation, shift) placing the nucleoside of ``b2`` (:func:`_core`) in the ``pairing`` pair with
    the nucleoside of ``b1``: the H-bond donors and acceptors of the two edges ``HBOND`` apart, the
    donor hydrogens on the donor-acceptor lines and the partner base coplanar (least squares from two
    starts, the partner on either face).  The 4JRD A·A pair is the two-fold about the helix axis."""
    from scipy.optimize import least_squares

    from .spatial import kabsch

    if (pairing, b1, b2) == ("hh", "A", "A"):
        return np.diag([-1.0, -1.0, 1.0]), np.zeros(3)
    bonds = _PAIR_HBONDS[(pairing, b1, b2)]
    t1, t2 = _core(b1), _core(b2)
    ring1, n1 = _base_plane(t1, b1)
    ring2, n2 = _base_plane(t2, b2)
    a = np.array([t1[x] for x, *_ in bonds])
    b = np.array([t2[y] for _x, y, *_ in bonds])
    out = _unit(np.cross(n1, a[-1] - a[0]))
    if np.dot(out, a.mean(axis=0) - ring1) < 0:
        out = -out
    targets = a + HBOND * out
    centre = targets.mean(axis=0) + out * np.linalg.norm(b.mean(axis=0) - ring2)
    names1 = list(BASE_NAMES[b1]) + ["C1'"] + list(_SUGAR)
    names2 = list(BASE_NAMES[b2]) + ["C1'"] + list(_SUGAR)
    fixed = np.array([t1[n] for n in names1])
    mobile = np.array([t2[n] for n in names2])
    hbonded = {(x, y) for x, y, *_ in bonds}
    free = np.array([[(x, y) not in hbonded for y in names2] for x in names1])
    planar = np.array([t2[n] for n in BASE_NAMES[b2]])
    best = None
    for sign in (1.0, -1.0):
        r0, cs, cd = kabsch(np.vstack([b, ring2, ring2 + n2]), np.vstack([targets, centre, centre + sign * n1]),
                            np.array([10.0] * len(b) + [1.0, 1.0]))

        def move(p, x, r0=r0, cs=cs, cd=cd):
            return (x - cs) @ (_rotvec(p[:3]) @ r0).T + cd + p[3:]

        def residuals(p, move=move):
            res = []
            for x, y, side, h in bonds:
                acc, don = (move(p, t2[y]), t1[x]) if side == 1 else (t1[x], move(p, t2[y]))
                hyd = t1[h] if side == 1 else move(p, t2[h])
                d = np.linalg.norm(don - acc)
                res += [(d - HBOND) * 4.0, (d - np.linalg.norm(hyd - acc) - BOND_H["nh"]) * 3.0]
            res += list((move(p, planar) - ring1) @ n1)
            dist = np.linalg.norm(fixed[:, None] - move(p, mobile)[None], axis=-1)
            res += list((np.clip(3.1 - dist, 0.0, None) * free).ravel() * 3.0)
            return np.array(res)

        sol = least_squares(residuals, np.zeros(6))
        if best is None or sol.cost < best[0]:
            rot = _rotvec(sol.x[:3]) @ r0
            best = (sol.cost, rot, cd + sol.x[3:] - cs @ rot.T)
    return best[1], best[2]


def _nucleotides(c1: np.ndarray, glyco: np.ndarray, sugar: np.ndarray, dchi, gamma, beta) -> Dict[str, np.ndarray]:
    """Sugar, O5' and P of nucleosides given by C1' (n, 3), the glycosidic N (n, 3) and the sugar atoms
    ``_SUGAR`` (n, 7, 3) or ``_RIGID`` (n, 12, 3: with the ring hydrogens; H5'/H5'' are then placed too):
    the sugar turned by ``dchi`` about the glycosidic bond, O5' and P placed from gamma and beta (degrees,
    arrays (n,))."""
    k = _unit_rows(c1 - glyco)[:, None, :]
    t = np.radians(np.asarray(dchi, dtype=np.float64))[:, None, None]
    p = sugar - c1[:, None, :]
    sugar = (p * np.cos(t) + _cross(k, p) * np.sin(t) + k * (k * p).sum(-1, keepdims=True) * (1 - np.cos(t))
             + c1[:, None, :])
    out = {n: sugar[:, i] for i, n in enumerate(_RIGID[:sugar.shape[1]])}
    out["C1'"] = c1
    out["O5'"] = _place_many(out["C3'"], out["C4'"], out["C5'"], _GEOM["C5'-O5'"], _GEOM["C4'-C5'-O5'"], gamma)
    out["P"] = _place_many(out["C4'"], out["C5'"], out["O5'"], _GEOM["O5'-P"], _GEOM["C5'-O5'-P"], beta)
    if sugar.shape[1] > len(_SUGAR):
        out["H5'"], out["H5''"] = _methylene_h(out["C5'"], out["O5'"], out["C4'"])
    return out


def _methylene_h(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The two hydrogens of CH2 groups (n, 3) bonded to ``a`` and ``b`` as in :func:`_hydrogens` (the
    first with a positive chiral volume c(a, b, H): H5' of a D-ribose for c, a, b = C5', O5', C4')."""
    u1, u2 = _unit_rows(a - c), _unit_rows(b - c)
    bis, perp = -_unit_rows(u1 + u2), _unit_rows(_cross(u1, u2))
    half = math.radians(109.5 / 2)
    h1 = c + BOND_H["ch2"] * (math.cos(half) * bis + math.sin(half) * perp)
    h2 = c + BOND_H["ch2"] * (math.cos(half) * bis - math.sin(half) * perp)
    swap = ((a - c) * _cross(b - c, h1 - c)).sum(-1) < 0
    return np.where(swap[:, None], h2, h1), np.where(swap[:, None], h1, h2)


def _core_arrays(placed: Sequence[Dict[str, np.ndarray]], bases: str, hydrogens: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = _RIGID if hydrogens else _SUGAR
    return (np.array([c["C1'"] for c in placed]), np.array([c[GLYCOSIDIC[b][0]] for c, b in zip(placed, bases)]),
            np.array([[c[s] for s in names] for c in placed]).reshape(len(placed), len(names), 3))


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _link_residuals(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> np.ndarray:
    """Closure of the O3'(a)-P(b) links: bond length and angles (hard), epsilon / zeta / alpha (staggered,
    near the 4JRD values)."""
    def torsion(theta, ref):
        return (1 + np.cos(np.radians(3 * theta))) * 1.5 + _wrap(theta - ref) / 120.0

    eps = _dihedral(a["C4'"], a["C3'"], a["O3'"], b["P"])
    return np.concatenate([
        (np.linalg.norm(a["O3'"] - b["P"], axis=-1) - LINK) / 0.01,
        (_angle(a["C3'"], a["O3'"], b["P"]) - _GEOM["C3'-O3'-P"]) / 1.5,
        (_angle(a["O3'"], b["P"], b["O5'"]) - _GEOM["O3'-P-O5'"]) / 1.5,
        torsion(eps, _TORSIONS["epsilon"]) + np.clip(np.abs(_wrap(eps + 120.0)) - 60.0, 0.0, None) / 10.0,
        torsion(_dihedral(a["C3'"], a["O3'"], b["P"], b["O5'"]), _TORSIONS["zeta"]),
        torsion(_dihedral(a["O3'"], b["P"], b["O5'"], b["C5'"]), _TORSIONS["alpha"])])


def _torsion_priors(dchi, gamma, beta) -> np.ndarray:
    """Backbone torsions near the 4JRD values; chi (4JRD -175 + dchi) kept in the C3'-endo anti range
    (``CHI_RANGE``: outside it the base turns onto its own sugar, C6/C8 against C3')."""
    dchi = np.asarray(dchi)
    lo, hi = CHI_RANGE[0] - _JRD_CHI, CHI_RANGE[1] - _JRD_CHI
    return np.concatenate([dchi / 20.0, (np.clip(lo - dchi, 0.0, None) + np.clip(dchi - hi, 0.0, None)) / 1.0,
                           _wrap(np.asarray(gamma) - _TORSIONS["gamma"]) / 25.0,
                           _wrap(np.asarray(beta) - _TORSIONS["beta"]) / 25.0])


@dataclass(frozen=True)
class _Strand:
    """A strand of a fitted helix: reference base, pairing rule that places it on the strand-1 base of
    its level (None: strand 1 itself) and its direction (+1 with strand 1, -1 against it)."""
    base: str
    rule: Optional[str]
    step: int


# the helices built here: the strands of the reference level (strand 1 always A)
_HELICES = {
    "reverse-wc": (_Strand("A", None, 1), _Strand("U", "reverse-wc", 1)),
    "hoogsteen": (_Strand("A", None, 1), _Strand("U", "hoogsteen", 1)),
    "reverse-hoogsteen": (_Strand("A", None, 1), _Strand("U", "reverse-hoogsteen", -1)),
    "hh": (_Strand("A", None, 1), _Strand("A", "hh", 1)),
    "triplex-hoogsteen": (_Strand("A", None, 1), _Strand("U", "wc", -1), _Strand("U", "hoogsteen", 1)),
    "triplex-reverse-hoogsteen": (_Strand("A", None, 1), _Strand("U", "wc", -1),
                                  _Strand("A", "reverse-hoogsteen", -1)),
}
# --- contacts of the rebuilt helices (hydrogens included) ---
# Atom classes: 0 carbon, phosphorus, glycosidic N; 1 N-H donor; 2 acceptor (O, N without H); 3 hydroxyl O
# (donor and acceptor); 4 polar hydrogen (N-H, O-H); 5 hydrogen on carbon; 6 no contacts (the virtual C+ proton
# 'H3+', padding); 7 atom H-bonded to the partner base (strand closure).
_POLAR_H = frozenset({"H1", "H3", "H21", "H22", "H41", "H42", "H61", "H62", "HO2'", "HO5'", "HO3'"})
_DONOR_N = {"A": ("N6",), "G": ("N1", "N2"), "C": ("N4",), "U": ("N3",)}
_ACCEPTOR_N = {"A": ("N1", "N3", "N7"), "G": ("N3", "N7"), "C": ("N3",), "U": ()}
# closest distances (Å) of atoms four or more bonds apart: heavy atoms, N/O pairs that can form an H-bond
# (donor-acceptor), H-H, polar H - acceptor (an H-bond), other H-heavy
CONTACT_FIT = {"heavy": 3.1, "hbond": 2.7, "hh": 2.1, "polar": 1.8, "hx": 2.4}  # targets of the fits
CONTACT_REPORT = {"heavy": 3.0, "hbond": 2.6, "hh": 2.0, "polar": 1.6, "hx": 1.9}  # reported as contacts
CHI_RANGE = (-185.0, -145.0)  # degrees, chi of the rebuilt nucleotides (C3'-endo anti; 4JRD about -170)


def _atom_class(base: str, name: str) -> int:
    if not name or name == "H3+":
        return 6
    if name.startswith("H"):
        return 4 if name in _POLAR_H else 5
    if name == "O2'" or name in ("HO5'", "HO3'"):
        return 3
    if name.startswith("O") or name in _ACCEPTOR_N[base]:
        return 2
    return 1 if name in _DONOR_N[base] else 0


def _limit_matrix(table: Dict[str, float]) -> np.ndarray:
    """Closest allowed distance of two atom classes (``_atom_class``) from a ``CONTACT_FIT``-style table."""
    out = np.zeros((8, 8))
    hyd = (4, 5)
    for a in range(8):
        for b in range(8):
            if 6 in (a, b):
                continue
            if a in hyd and b in hyd:
                out[a, b] = table["hh"]
            elif a in hyd or b in hyd:
                h, x = (a, b) if a in hyd else (b, a)
                out[a, b] = table["polar"] if h == 4 and x in (2, 3, 7) else table["hx"]
            elif 7 in (a, b) or ({a, b} <= {1, 2, 3} and {a, b} not in ({1}, {2})):
                out[a, b] = table["hbond"]
            else:
                out[a, b] = table["heavy"]
    return out


_FIT_LIMITS = _limit_matrix(CONTACT_FIT)
_FIT_CONTACT_WEIGHT = 0.03  # Å per unit residual of the contact terms of the helix fits
_REPORT_LIMITS = _limit_matrix(CONTACT_REPORT)


@lru_cache(maxsize=None)
def _residue_graph(base: str) -> Tuple[Dict[str, int], np.ndarray]:
    """Atom index and bond-path lengths (99: none) of a nucleotide with its hydrogens (Arnott template)."""
    t = template(base, hydrogens=True)
    names = list(t)
    xyz = np.array([t[n] for n in names])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    is_h = np.array([n.startswith("H") for n in names])
    bond = (d < 1.75) & ~np.eye(len(names), dtype=bool) & ~(is_h[:, None] & is_h[None])
    bond &= ~((is_h[:, None] | is_h[None]) & (d > 1.25))
    dist = np.where(np.eye(len(names), dtype=bool), 0, 99)
    reach = np.eye(len(names), dtype=bool)
    for k in range(1, 16):
        reach = reach | ((reach.astype(int) @ bond.astype(int)) > 0)
        dist[(dist == 99) & reach] = k
    return {n: i for i, n in enumerate(names)}, dist


def _bond_path(base: str, a: str, b: str) -> int:
    index, dist = _residue_graph(base)
    return int(dist[index[a], index[b]]) if a in index and b in index else 99


def _link_path(base1: str, a: str, base2: str, b: str) -> int:
    """Bond-path length from atom ``a`` of a nucleotide to atom ``b`` of the next one (O3'-P link)."""
    return _bond_path(base1, a, "O3'") + 1 + _bond_path(base2, "P", b)


def _base_slots(base: str) -> Tuple[str, ...]:
    """Base atoms moved with the base: heavy atoms, hydrogens, C: the virtual 'H3+' (H-bonds of C+ only)."""
    return BASE_NAMES[base] + tuple(h for h, *_ in _H_RULES[base]) + (("H3+",) if base == "C" else ())


@dataclass(frozen=True)
class _HelixFit:
    screw: _Screw
    torsions: Tuple[Tuple[float, float, float], ...]  # per strand: chi change, gamma, beta
    cost: float
    contact: float  # closest heavy-atom distance (four or more bonds apart) between the atoms of the levels
    contact_h: float  # closest H-H distance (four or more bonds apart)


@lru_cache(maxsize=None)
def _reference_level(kind: str):
    """Placed nucleosides (with their hydrogens) of the reference level (4JRD frame) and the axis basis of
    strand 1 (cached: do not modify)."""
    strands = _HELICES[kind]
    t1 = _core(strands[0].base)
    placed = [t1]
    for st in strands[1:]:
        rot, shift = _pair_fit(st.rule, strands[0].base, st.base)
        placed.append(_moved(_core(st.base), rot, shift))
    n = -_base_normal(t1, strands[0].base)  # strand 1 runs 5'->3' against its glycosidic normal
    e1 = _unit(np.cross(n, [1.0, 0.0, 0.0]))
    centre = np.mean([p["C1'"] for p in placed], axis=0)
    return placed, centre, n, e1, np.cross(n, e1)


def _level_atoms(placed, strands, q, screw):
    """Atoms (names, strand index, xyz) of one level of the uniform helix, phosphate oxygens and hydrogens
    included."""
    names, owner, rows = [], [], []
    for s, (core, st) in enumerate(zip(placed, strands)):
        nt = _nucleotides(*_core_arrays([core], st.base, True), *[np.array([v]) for v in q[s]])
        prev_o3 = screw.apply(nt["O3'"][0], -st.step)
        op1, op2 = _phosphate_oxygens(nt["P"][0], prev_o3, nt["O5'"][0])
        atoms = {n: x[0] for n, x in nt.items()}
        atoms.update({"OP1": op1, "OP2": op2})
        atoms.update({n: core[n] for n in _base_slots(st.base) if n != "H3+"})
        for n, x in atoms.items():
            names.append(n)
            owner.append(s)
            rows.append(x)
    return names, np.array(owner), np.array(rows)


@lru_cache(maxsize=None)
def _contact_pairs(kind: str) -> Tuple[np.ndarray, ...]:
    """Atom pairs of the reference level scored for contacts: (atom i of level 0, atom j, level of j (0, 1
    or 2), closest allowed distance, both heavy?, both hydrogens?).  Pairs four or more bonds apart within
    a nucleotide and across the O3'-P link, all pairs between strands and with level 2; the H-bonded
    atoms of the paired bases are left out."""
    strands = _HELICES[kind]
    placed = _reference_level(kind)[0]
    q = [(0.0, 58.0, 174.0)] * len(strands)
    names, owner, _ = _level_atoms(placed, strands, q, _Screw((0, 0, 0), (0, 0, 1), 30.0, 3.0))
    bases = [strands[s].base for s in owner]
    cls = np.array([_atom_class(b, a) for a, b in zip(names, bases)])
    hbonded = set()
    for s, st in enumerate(strands[1:], start=1):
        for x, y, *_ in _PAIR_HBONDS.get((st.rule, strands[0].base, st.base), ()):
            hbonded |= {((0, x), (s, y)), ((s, y), (0, x))}
    rows = []
    for i, (a, s) in enumerate(zip(names, owner)):
        for j, (b, t) in enumerate(zip(names, owner)):
            for level in (0, 1, 2):
                if level == 0 and j <= i:
                    continue
                if s == t and level == 0:
                    path = _bond_path(bases[i], a, b)
                elif s == t and level == 1:  # O3'(level 0) -> P(level 1) or O3'(level 1) -> P(level 0)
                    path = (_link_path(bases[i], a, bases[j], b) if strands[s].step > 0
                            else _link_path(bases[j], b, bases[i], a))
                else:
                    path = 99
                if path < 4 or (level == 0 and ((s, a), (t, b)) in hbonded):
                    continue
                limit = _FIT_LIMITS[cls[i], cls[j]]
                if limit > 0:
                    rows.append((i, j, level, limit, cls[i] < 4 and cls[j] < 4, cls[i] in (4, 5) and cls[j] in (4, 5)))
    i, j, level, limit, heavy, hh = (np.array(c) for c in zip(*rows))
    return i.astype(int), j.astype(int), level.astype(int), limit.astype(float), heavy.astype(bool), hh.astype(bool)


def _level_distances(kind: str, placed, strands, q, screw) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    i, j, level, *rest = pairs = _contact_pairs(kind)
    xyz = _level_atoms(placed, strands, q, screw)[2]
    levels = np.stack([xyz, screw.apply(xyz, 1), screw.apply(xyz, 2)])
    return np.linalg.norm(xyz[i] - levels[level, j], axis=-1), pairs


def _fit_residuals(kind: str, p: np.ndarray, start: np.ndarray, contacts: bool = True) -> np.ndarray:
    strands = _HELICES[kind]
    placed, centre, n, e1, e2 = _reference_level(kind)
    screw, q = _unpack_fit(kind, p)
    out = []
    for core, st, qs in zip(placed, strands, q):
        here = _nucleotides(*_core_arrays([core], st.base), *[np.array([v]) for v in qs])
        names = list(here)
        moved = screw.apply(np.concatenate([here[k] for k in names]), st.step)
        there = {k: moved[i:i + 1] for i, k in enumerate(names)}
        out += [_link_residuals(here, there), _torsion_priors(*[np.array([v]) for v in qs])]
    if kind != "hh":
        twist, rise = _SCREW_PRIOR[kind]
        out.append(np.array([p[2] / 25.0, p[3] / 25.0, (p[4] - twist) / 12.0, (p[5] - rise) / 0.3]))
    if contacts:
        d, (_i, _j, _lv, limit, *_x) = _level_distances(kind, placed, strands, q, screw)
        out.append(np.clip(limit - d, 0.0, None) / _FIT_CONTACT_WEIGHT)
    return np.concatenate(out)


def _unpack_fit(kind: str, p: np.ndarray):
    if kind == "hh":  # the 4JRD helix: axis z through the origin of the 4JRD frame
        return _Screw((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), JRD_TWIST, JRD_RISE), [p[0:3], p[0:3]]
    placed, centre, n, e1, e2 = _reference_level(kind)
    x, y, tx, ty, tw, rise = p[:6]
    axis = _turn_matrix(e1, tx) @ _turn_matrix(e2, ty) @ n
    q = [p[6 + 3 * s:9 + 3 * s] for s in range(len(_HELICES[kind]))]
    return _Screw(tuple(centre + x * e1 + y * e2), tuple(axis), float(tw), float(rise)), q


def _closest_contact(kind: str, p: np.ndarray) -> Tuple[float, float]:
    """(closest heavy-atom distance, closest H-H distance) of the scored pairs of the reference helix."""
    screw, q = _unpack_fit(kind, p)
    d, (_i, _j, _lv, _limit, heavy, hh) = _level_distances(kind, _reference_level(kind)[0], _HELICES[kind], q, screw)
    return float(d[heavy].min()), float(d[hh].min())


@lru_cache(maxsize=None)
def _helix_fit(kind: str) -> _HelixFit:
    """Uniform helix of ``kind`` (``_HELICES``): the screw and per-strand backbone torsions that close the
    O3'-P links of the reference level repeated by the screw, without contacts (``CONTACT_FIT``, hydrogens
    included) within a nucleotide, between levels or strands; least squares from ``_HELIX_START`` (hh: the
    4JRD screw, torsions only)."""
    from scipy.optimize import least_squares

    start = np.array(_HELIX_START[kind], dtype=np.float64)
    sol = least_squares(lambda p: _fit_residuals(kind, p, start), start, x_scale=_fit_scale(kind))
    screw, q = _unpack_fit(kind, sol.x)
    return _HelixFit(screw, tuple(tuple(float(v) for v in qs) for qs in q), float(sol.cost),
                     *_closest_contact(kind, sol.x))


def _fit_scale(kind: str) -> np.ndarray:
    if kind == "hh":
        return np.array([10.0, 10.0, 10.0])
    return np.array([1.0, 1.0, 5.0, 5.0, 5.0, 0.3] + [10.0] * 3 * len(_HELICES[kind]))


_STRAND_NAMES = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'") + _SUGAR_H + (
    "H5'", "H5''")
_BASE_SLOTS = 15  # most base atoms with hydrogens (G)
_BASE_CODE = {"A": 0, "C": 1, "G": 2, "U": 3}
LEVEL_MOVE = (10.0, 0.8)  # degrees, Å: scale of the rigid level moves of the strand closure
_CLOSE_CUTOFF = 3.8  # Å, atom pairs scored for contacts by the closure (refreshed between passes)
# closure passes (contact weight, evaluations): the contact pairs are refreshed before each pass; passes
# after the second only while contacts or link gaps remain
_CLOSE_PASSES = ((0.02, 40), (0.02, 40), (0.01, 40), (0.01, 40))
_CLOSE_LSMR = {"atol": 1e-5, "btol": 1e-5}  # trust-region subproblem tolerances (speed)


def _phosphate_oxygens_many(p: np.ndarray, o3: np.ndarray, o5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`_phosphate_oxygens` for arrays (n, 3)."""
    u1, u2 = _unit_rows(o3 - p), _unit_rows(o5 - p)
    bis, perp = -_unit_rows(u1 + u2), _unit_rows(_cross(u1, u2))
    half = math.radians(_GEOM["OP-P-OP"] / 2)
    a = p + _GEOM["P-OP"] * (math.cos(half) * bis + math.sin(half) * perp)
    b = p + _GEOM["P-OP"] * (math.cos(half) * bis - math.sin(half) * perp)
    swap = (_cross(a - p, b - p) * (o5 - p)).sum(-1) > 0  # chiral volume (P; O5', a, b) > 0
    a2 = np.where(swap[:, None], b, a)
    b2 = np.where(swap[:, None], a, b)
    return a2, b2


@lru_cache(maxsize=None)
def _slot_table(base: str) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    """Atom slots of a nucleotide in :func:`_close_levels` (``_STRAND_NAMES``, then the base slots padded
    with ''), their atom classes and the pairs scored within the nucleotide (symmetric): four or more bonds
    apart and not both on the sugar ring or both on the base (rigid)."""
    names = _STRAND_NAMES + _base_slots(base)
    names += ("",) * (len(_STRAND_NAMES) + _BASE_SLOTS - len(names))
    cls = np.array([_atom_class(base, n) for n in names])
    rigid, ring = set(_RIGID) | {"C1'"}, set(_base_slots(base))
    inside = np.zeros((len(names), len(names)), dtype=bool)
    for i, a in enumerate(names):
        for k in range(i + 1, len(names)):
            b = names[k]
            if 6 in (cls[i], cls[k]) or {a, b} <= rigid or {a, b} <= ring:
                continue
            inside[i, k] = inside[k, i] = _bond_path(base, a, b) >= 4
    return names, cls, inside


@lru_cache(maxsize=None)
def _contact_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """By base (``_BASE_CODE``): slot classes (4, m), slot pairs scored within a nucleotide (4, m, m) and
    across the O3'-P link of two consecutive nucleotides, 5' one first (4, 4, m, m: four or more bonds apart)."""
    tables = [_slot_table(b) for b in "ACGU"]
    link = np.zeros((4, 4) + tables[0][2].shape, dtype=bool)
    for x, b1 in enumerate("ACGU"):
        for y, b2 in enumerate("ACGU"):
            n1, c1, _ = tables[x]
            n2, c2, _ = tables[y]
            link[x, y] = [[c1[i] != 6 and c2[k] != 6 and _link_path(b1, a, b2, b) >= 4 for k, b in enumerate(n2)]
                          for i, a in enumerate(n1)]
    return np.array([t[1] for t in tables]), np.array([t[2] for t in tables]), link


def _span_sparsity(lo: np.ndarray, hi: np.ndarray, block: int, levels: int):
    """Jacobian sparsity of residuals that depend on the variable blocks lo..hi (inclusive)."""
    from scipy.sparse import csr_matrix

    lo, hi = np.asarray(lo, dtype=int), np.asarray(hi, dtype=int)
    width = (hi - lo + 1) * block
    rows = np.repeat(np.arange(len(lo)), width)
    cols = np.repeat(lo * block, width) + np.arange(width.sum()) - np.repeat(np.cumsum(width) - width, width)
    return csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(lo), levels * block))


def _close_levels(placed: Sequence[Sequence[Dict[str, np.ndarray]]], bases: Sequence[str], steps: Sequence[int],
                  start: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[List[Dict[str, np.ndarray]]]]:
    """Backbones of the strands of a helix whose levels are not all alike (other pairs, mismatches).

    ``placed[s][k]``: nucleoside (with hydrogens) of strand s at level k, posed on the strand-1 nucleoside of
    its level (:func:`_level_pose`); ``bases[s]``: the bases of strand s by level; ``steps[s]``: +1 for a
    strand running 5'->3' with the levels, -1 against them; ``start[s]``: torsions (n, 3) by level (chi
    change, gamma, beta).  Every level moves as one rigid body, so the pairs keep their H-bonds and
    coplanar bases, and every nucleotide gets its own torsions.  Least squares over the level moves and the
    torsions: O3'-P link geometry and staggered epsilon / zeta / alpha of every strand, torsions near the
    4JRD values with chi in ``CHI_RANGE``, small level moves (``LEVEL_MOVE``) and contacts (``CONTACT_FIT``,
    hydrogens included) within nucleotides, between strands and between levels.  Sparse: each term couples
    at most four neighbouring levels.  Returns (torsions per strand by level, moved nucleosides per strand
    by level)."""
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    n_s, n = len(placed), len(placed[0])
    width = 6 + 3 * n_s  # variables per level: rotation (degrees), shift (Å), torsions of each strand
    arrays = [_core_arrays(p, b, True) for p, b in zip(placed, bases)]
    base_xyz = np.full((n_s, n, _BASE_SLOTS, 3), 1e4)
    for s in range(n_s):
        for k, (core, b) in enumerate(zip(placed[s], bases[s])):
            base_xyz[s, k, :len(_base_slots(b))] = [core[x] for x in _base_slots(b)]
    centre = np.mean([a[0] for a in arrays], axis=0)  # mean C1' of each level
    order = [np.arange(n) if st > 0 else np.arange(n)[::-1] for st in steps]
    codes = np.array([[_BASE_CODE[b] for b in bs] for bs in bases])
    cls_tab, inside_tab, link_tab = _contact_tables()
    n_atoms = len(_STRAND_NAMES) + _BASE_SLOTS
    cls = cls_tab[codes].ravel()
    atom_s, atom_k, atom_i = np.unravel_index(np.arange(n_s * n * n_atoms), (n_s, n, n_atoms))
    step_of = np.asarray(steps)

    def build(x):
        v = x.reshape(n, width)
        rot = _rotvecs(np.radians(v[:, 0:3]))

        def move(y):
            return np.einsum("nij,nmj->nmi", rot, y - centre[:, None]) + centre[:, None] + v[:, None, 3:6]

        nts, atoms = [], []
        for s, (c1, glyco, sugar) in enumerate(arrays):
            q = v[:, 6 + 3 * s:9 + 3 * s]
            nt = _nucleotides(move(c1[:, None])[:, 0], move(glyco[:, None])[:, 0], move(sugar), q[:, 0], q[:, 1],
                              q[:, 2])
            o = order[s]
            prev = np.empty((n, 3))
            prev[o[1:]] = nt["O3'"][o[:-1]]
            prev[o[0]] = _place(nt["C5'"][o[0]], nt["O5'"][o[0]], nt["P"][o[0]], LINK, _GEOM["O3'-P-O5'"],
                                _TORSIONS["alpha"])
            nt["OP1"], nt["OP2"] = _phosphate_oxygens_many(nt["P"], prev, nt["O5'"])
            nts.append(nt)
            atoms.append(np.concatenate([np.stack([nt[a] for a in _STRAND_NAMES], axis=1), move(base_xyz[s])], axis=1))
        return nts, np.array(atoms).reshape(-1, 3)

    pairs: Dict[str, np.ndarray] = {}

    def candidates(x, cutoff):
        """Atom pairs within ``cutoff``: not in the same rigid part, four or more bonds apart."""
        xyz = build(x)[1]
        real = np.nonzero(cls != 6)[0]
        found = cKDTree(xyz[real]).query_pairs(cutoff, output_type="ndarray")
        p, q = real[found[:, 0]], real[found[:, 1]]
        sp, kp, ip, sq, kq, iq = atom_s[p], atom_k[p], atom_i[p], atom_s[q], atom_k[q], atom_i[q]
        keep = (kp != kq) | (sp != sq) & ((ip < len(_STRAND_NAMES)) | (iq < len(_STRAND_NAMES)))
        same = (sp == sq) & (kp == kq)
        keep[same] = inside_tab[codes[sp[same], kp[same]], ip[same], iq[same]]
        linked = (sp == sq) & (kq - kp == step_of[sp])  # q follows p on its strand
        keep[linked] = link_tab[codes[sp[linked], kp[linked]], codes[sq[linked], kq[linked]], ip[linked], iq[linked]]
        linked = (sp == sq) & (kp - kq == step_of[sp])  # p follows q
        keep[linked] = link_tab[codes[sq[linked], kq[linked]], codes[sp[linked], kp[linked]], iq[linked], ip[linked]]
        limit = _FIT_LIMITS[cls[p], cls[q]]
        keep &= limit > 0
        # levels each pair depends on: its two atoms, and the previous O3' for the phosphate oxygens
        ext_p = np.clip(np.where((ip == 1) | (ip == 2), kp - step_of[sp], kp), 0, n - 1)
        ext_q = np.clip(np.where((iq == 1) | (iq == 2), kq - step_of[sq], kq), 0, n - 1)
        lo = np.minimum(np.minimum(kp, kq), np.minimum(ext_p, ext_q))
        hi = np.maximum(np.maximum(kp, kq), np.maximum(ext_p, ext_q))
        pairs.update(p=p[keep], q=q[keep], limit=limit[keep], lo=lo[keep], hi=hi[keep])

    def violations(xyz):
        return pairs["limit"] - np.linalg.norm(xyz[pairs["p"]] - xyz[pairs["q"]], axis=-1)

    def residuals(x, weight):
        v = x.reshape(n, width)
        nts, xyz = build(x)
        out = [_link_residuals({k: a[o[:-1]] for k, a in nt.items()}, {k: a[o[1:]] for k, a in nt.items()})
               for nt, o in zip(nts, order)]
        out += [_torsion_priors(v[:, 6 + 3 * s], v[:, 7 + 3 * s], v[:, 8 + 3 * s]) for s in range(n_s)]
        out += [(v[:, 0:3] / LEVEL_MOVE[0]).ravel(), (v[:, 3:6] / LEVEL_MOVE[1]).ravel(),
                np.clip(violations(xyz), 0.0, None) / weight]
        return np.concatenate(out)

    def solve(x, weight, evaluations):
        lo, hi = [], []  # level span of every residual
        for o in order:
            a, b = np.minimum(o[:-1], o[1:]), np.maximum(o[:-1], o[1:])
            lo += list(a) * 6
            hi += list(b) * 6
        lo += list(range(n)) * 4 * n_s
        hi += list(range(n)) * 4 * n_s
        lo += list(np.repeat(np.arange(n), 3)) * 2
        hi += list(np.repeat(np.arange(n), 3)) * 2
        lo += list(pairs["lo"])
        hi += list(pairs["hi"])
        scale = np.tile([3.0, 3.0, 3.0, 0.2, 0.2, 0.2] + [10.0] * 3 * n_s, n)
        return least_squares(residuals, x, jac_sparsity=_span_sparsity(lo, hi, width, n), max_nfev=evaluations,
                             ftol=1e-6, x_scale=scale, args=(weight,), tr_options=_CLOSE_LSMR).x

    x0 = np.concatenate([np.zeros((n, 6))] + [np.asarray(q, dtype=np.float64).reshape(n, 3) for q in start],
                        axis=1).ravel()
    # contacts within _CLOSE_CUTOFF are scored; short passes with fresh pairs keep atoms from passing
    # through each other, later ones weigh contacts more
    def done(x):
        nts, xyz = build(x)
        gaps = [np.abs(np.linalg.norm(nt["O3'"][o[:-1]] - nt["P"][o[1:]], axis=-1) - LINK) for nt, o in zip(nts, order)]
        return violations(xyz).max(initial=-1.0) <= 0.05 and np.concatenate(gaps).max(initial=0.0) <= 0.05

    x = x0
    for k, (weight, evaluations) in enumerate(_CLOSE_PASSES):
        candidates(x, _CLOSE_CUTOFF)
        if k >= 2 and done(x):
            break
        x = solve(x, weight, evaluations)
    v = x.reshape(n, width)
    rot = _rotvecs(np.radians(v[:, 0:3]))
    moved = [[{a: (y - centre[k]) @ rot[k].T + centre[k] + v[k, 3:6] for a, y in core.items()}
              for k, core in enumerate(strand)] for strand in placed]
    return [v[:, 6 + 3 * s:9 + 3 * s] for s in range(n_s)], moved


def _phosphate_oxygens(p: np.ndarray, o3: np.ndarray, o5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OP1, OP2 on the bisector plane of O3'-P-O5' (prochiral names of a D-nucleotide as in wwPDB)."""
    u1, u2 = _unit(o3 - p), _unit(o5 - p)
    bis, perp = -_unit(u1 + u2), _unit(np.cross(u1, u2))
    half = math.radians(_GEOM["OP-P-OP"] / 2)
    a = p + _GEOM["P-OP"] * (math.cos(half) * bis + math.sin(half) * perp)
    b = p + _GEOM["P-OP"] * (math.cos(half) * bis - math.sin(half) * perp)
    if pdbtools.chiral_volume(p, o5, a, b) > 0:
        a, b = b, a
    return a, b


def _strand_heavy(placed: Sequence[Dict[str, np.ndarray]], bases: str, q: np.ndarray
                  ) -> List[Dict[str, np.ndarray]]:
    """Atoms of a strand (5'->3', hydrogens included) from placed nucleosides and backbone torsions; the 5'
    residue gets P/OP1/OP2 towards a virtual O3' (used for a 5'-phosphate) stored as '_prev'."""
    nt = _nucleotides(*_core_arrays(placed, bases, True), q[:, 0], q[:, 1], q[:, 2])
    out: List[Dict[str, np.ndarray]] = []
    for j, (core, base) in enumerate(zip(placed, bases)):
        heavy = {n: core[n] for n in _base_slots(base) if n != "H3+"}
        heavy.update({n: x[j] for n, x in nt.items()})
        prev = out[-1]["O3'"] if out else _place(heavy["C5'"], heavy["O5'"], heavy["P"], LINK,
                                                   _GEOM["O3'-P-O5'"], _TORSIONS["alpha"])
        heavy["OP1"], heavy["OP2"] = _phosphate_oxygens(heavy["P"], prev, heavy["O5'"])
        heavy["_prev"] = prev
        out.append(heavy)
    return out


@dataclass
class _Level:
    """One pair of a strand with strand 1 in a rebuilt helix: strand-1 level, bases (strand 1 + partner),
    canonical?, the H-bonds of the canonical pair."""
    level: int
    bases: str
    canonical: bool
    hbonds: Tuple = ()


@lru_cache(maxsize=None)
def _level_pose(kind: str, strand: int, level: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """(rotation, shift) placing the nucleoside of ``level[strand]`` (:func:`_core`) on strand ``strand`` of
    the helix ``kind`` at a level with the bases ``level`` (strands 0 .. ``strand``, 4JRD frame of the
    strand-1 nucleoside): the pair of its rule with ``level[0]`` when canonical; a mismatch starts from the
    reference partner (``_pair_fit`` of the reference level) and is moved, coplanar with the strand-1 base
    and as little as possible, off the nucleosides of the strands before it (``CONTACT_FIT``, hydrogens
    included)."""
    from scipy.optimize import least_squares

    spec = _HELICES[kind]
    rule, a, b = spec[strand].rule, level[0], level[strand]
    if _canonical(rule, a, b):
        return _pair_fit(rule, a, b)
    rot0, shift0 = _pair_fit(rule, spec[0].base, spec[strand].base)
    others = [_core(a)] + [_moved(_core(level[t]), *_level_pose(kind, t, level[:t + 1])) for t in range(1, strand)]
    keep = set(_RIGID) | {"C1'"}

    def atoms(coords, base):
        names = [x for x in coords if x in keep or (x in _base_slots(base) and x != "H3+")]
        return np.array([coords[x] for x in names]), np.array([_atom_class(base, x) for x in names])

    fixed = [atoms(c, base) for c, base in zip(others, level)]
    fixed_xyz = np.vstack([f[0] for f in fixed])
    fixed_cls = np.concatenate([f[1] for f in fixed])
    start = _moved(_core(b), rot0, shift0)
    mobile, mobile_cls = atoms(start, b)
    limit = _FIT_LIMITS[mobile_cls[:, None], fixed_cls[None, :]]
    planar = np.array([start[x] for x in BASE_NAMES[b]])
    ring1, n1 = _base_plane(others[0], a)
    centre = start["C1'"]

    def move(p, x):
        return (x - centre) @ _rotvec(np.radians(p[:3])).T + centre + p[3:]

    def residuals(p):
        d = np.linalg.norm(move(p, mobile)[:, None] - fixed_xyz[None], axis=-1)
        return np.concatenate([p[:3] / 10.0, p[3:] / 0.5, ((move(p, planar) - ring1) @ n1) / 0.25,
                               (np.clip(limit - d, 0.0, None) / 0.05).ravel()])

    best = None
    for turn in (0.0, 25.0, -25.0, 50.0, -50.0):  # in-plane turns about C1' as starts
        sol = least_squares(residuals, np.concatenate([np.degrees(n1 * math.radians(turn)), np.zeros(3)]))
        if best is None or sol.cost < best.cost - 1e-9:
            best = sol
    r = _rotvec(np.radians(best.x[:3]))
    return r @ rot0, (shift0 - centre) @ r.T + centre + best.x[3:]


def _helix_strands(kind: str, sequences: Sequence[str]):
    """Strands (atom dicts with hydrogens, 5'->3', 4JRD frame) of the helix ``kind`` for ``sequences``
    (5'->3', strand 1 first; a strand running against strand 1 pairs its residue n-1-k with level k).
    Every level is posed by :func:`_level_pose` (canonical pairs; mismatches moved off the bases they would
    hit); a helix whose levels are not all the reference level is closed by :func:`_close_levels`.  Returns
    (strands, levels per partner strand, fit)."""
    fit = _helix_fit(kind)
    spec = _HELICES[kind]
    screw = fit.screw
    seq = sequences[0]
    n = len(seq)
    placed = [[{x: screw.apply(v, k) for x, v in _core(a).items()} for k, a in enumerate(seq)]]
    level_bases = [seq]
    at_level = [(a,) for a in seq]  # bases of the strands placed so far, per level
    levels, uniform = [], True
    for s, (st, other) in enumerate(zip(spec[1:], sequences[1:]), start=1):
        ref = _pair_fit(st.rule, spec[0].base, st.base)
        strand, lvls = [], []
        for k, a in enumerate(seq):
            b = other[k] if st.step > 0 else other[n - 1 - k]
            at_level[k] += (b,)
            canonical = _canonical(st.rule, a, b)
            rot, shift = _level_pose(kind, s, at_level[k])
            uniform = (uniform and (a, b) == (spec[0].base, st.base) and np.allclose(rot, ref[0], atol=1e-6)
                       and np.allclose(shift, ref[1], atol=1e-6))
            strand.append({x: screw.apply(v, k) for x, v in _moved(_core(b), rot, shift).items()})
            lvls.append(_Level(k, a + b, canonical, _PAIR_HBONDS[(st.rule, a, b)] if canonical else ()))
        placed.append(strand)
        level_bases.append("".join(lv.bases[1] for lv in lvls))
        levels.append(lvls)
    torsions = [np.tile(q, (n, 1)) for q in fit.torsions]
    if not uniform:  # also a single level: the torsions of the other bases
        torsions, placed = _close_levels(placed, level_bases, [st.step for st in spec], torsions)
    strands = []
    for st, nucleosides, bases, q in zip(spec, placed, level_bases, torsions):
        o = slice(None) if st.step > 0 else slice(None, None, -1)
        strands.append(_strand_heavy(list(nucleosides)[o], bases[o], np.asarray(q)[o]))
    return strands, levels, fit


def _residue_atoms(heavy: Dict[str, np.ndarray], base: str, five: bool, three: bool, phosphate5: bool,
                   hydrogens: bool) -> List[Tuple[str, np.ndarray]]:
    """Atoms of one rebuilt nucleotide in the order of :func:`_strand_residue`."""
    t = dict(heavy)
    first: List[Tuple[str, np.ndarray]] = []
    last: List[Tuple[str, np.ndarray]] = []
    if five:
        if phosphate5:
            first.append(("OP3", t["P"] + P_OP3 * _unit(t["_prev"] - t["P"])))
        else:
            if hydrogens:
                last.append(("HO5'", _place(t["C4'"], t["C5'"], t["O5'"], BOND_H["oh"], 109.5, 180.0)))
            for name in ("P", "OP1", "OP2"):
                del t[name]
    if three and hydrogens:
        last.append(("HO3'", _place(t["C4'"], t["C3'"], t["O3'"], BOND_H["oh"], 109.5, 180.0)))
    atoms = first + [(n, t[n]) for n in BACKBONE_NAMES + BASE_NAMES[base] if n in t]
    if hydrogens:
        hyd = _hydrogens(heavy, base)
        atoms += [(n, hyd[n]) for n, *_ in _H_RULES["sugar"] + _H_RULES[base]]
    return atoms + last


def _axis_frame(screw: _Screw) -> Tuple[np.ndarray, np.ndarray]:
    """(rotation, shift) taking the screw axis to z through the origin (direction kept)."""
    u = _unit(np.array(screw.axis))
    z = np.array([0.0, 0.0, 1.0])
    c = float(u @ z)
    v = np.cross(u, z)
    s = float(np.linalg.norm(v))
    rot = np.eye(3) if s < 1e-12 else _turn_matrix(v, math.degrees(math.atan2(s, c)))
    return rot, -np.array(screw.origin) @ rot.T


def _bonded_within(res_atoms: List[List[Tuple[str, np.ndarray]]], segments: List[List[int]]
                   ) -> Tuple[np.ndarray, List[Tuple[int, str]], List[Tuple[int, int]]]:
    """Atoms (xyz, hydrogens included), their (residue index, name) and the bond list: bonds within
    residues by distance (heavy atoms < 1.75 Å, hydrogens < 1.25 Å) and the O3'-P links of consecutive
    residues of a segment."""
    xyz, owner, bonds = [], [], []
    index: Dict[Tuple[int, str], int] = {}
    for r, atoms in enumerate(res_atoms):
        start = len(xyz)
        hyd = []
        for name, x in atoms:
            index[(r, name)] = len(xyz)
            xyz.append(np.asarray(x, dtype=np.float64))
            owner.append((r, name))
            hyd.append(name.lstrip("0123456789").startswith("H"))
        block = np.array(xyz[start:]).reshape(-1, 3)
        hyd = np.array(hyd, dtype=bool)
        d = np.linalg.norm(block[:, None] - block[None], axis=-1)
        cut = np.where(hyd[:, None] | hyd[None], 1.25, 1.75)
        bonded = np.triu(d < cut, 1) & ~(hyd[:, None] & hyd[None])
        bonds += [(start + i, start + j) for i, j in zip(*np.nonzero(bonded))]
    for seg in segments:
        for a, b in zip(seg, seg[1:]):
            if (a, "O3'") in index and (b, "P") in index:
                bonds.append((index[(a, "O3'")], index[(b, "P")]))
    return np.array(xyz).reshape(-1, 3), owner, bonds


def _close_contacts(res_atoms: List[List[Tuple[str, np.ndarray]]], segments: List[List[int]], bases: Sequence[str],
                    hbonds: Sequence[Tuple[int, str, int, str]] = ()) -> List[Tuple[float, str, Tuple, Tuple]]:
    """Close contacts of a model: (distance, what, (residue, atom), (residue, atom)) for heavy atoms three
    bonds apart closer than ``CLASH_NB`` and atoms four or more bonds apart (or not bonded) closer than
    ``CONTACT_REPORT`` (by atom class: heavy atoms, N/O pairs that can H-bond, H-H, polar H - acceptor,
    other H-heavy).  ``hbonds``: the H-bonded donor / acceptor pairs (residue, atom, residue, atom) of the
    canonical pairs, left out."""
    from scipy.spatial import cKDTree

    xyz, owner, bonds = _bonded_within(res_atoms, segments)
    if not len(xyz):
        return []
    cls = np.array([_atom_class(bases[r], name.lstrip("0123456789")) for r, name in owner])
    neighbours: List[set] = [set() for _ in owner]
    for a, b in bonds:
        neighbours[a].add(b)
        neighbours[b].add(a)
    shells: Dict[int, Tuple[set, set]] = {}

    def within(a):  # atoms two and three bonds from a
        if a not in shells:
            two = set().union(*(neighbours[c] for c in neighbours[a])) if neighbours[a] else set()
            three = set().union(*(neighbours[c] for c in two)) if two else set()
            shells[a] = (two, three)
        return shells[a]

    index = {k: i for i, k in enumerate(owner)}
    skip = {frozenset((index[(i, x)], index[(j, y)])) for i, x, j, y in hbonds if (i, x) in index and (j, y) in index}
    out = []
    for a, b in cKDTree(xyz).query_pairs(max(CONTACT_REPORT.values())):
        if b in neighbours[a]:
            continue
        two, three = within(a)
        if b in two:
            continue
        d = float(np.linalg.norm(xyz[a] - xyz[b]))
        hyd = cls[a] in (4, 5) or cls[b] in (4, 5)
        if b in three:  # 1-4 pairs: heavy atoms only
            if not hyd and d < CLASH_NB:
                out.append((d, f"heavy, 3 bonds, < {CLASH_NB}", owner[a], owner[b]))
            continue
        if frozenset((a, b)) in skip:
            continue
        limit = _REPORT_LIMITS[cls[a], cls[b]]
        if d < limit:
            if cls[a] in (4, 5) and cls[b] in (4, 5):
                what = "H-H"
            elif hyd:
                what = "H-bond H" if limit == CONTACT_REPORT["polar"] else "H-heavy"
            else:
                what = "N/O" if limit == CONTACT_REPORT["hbond"] else "heavy"
            out.append((d, f"{what} < {limit:.1f}", owner[a], owner[b]))
    out.sort(key=lambda c: c[0])
    return out


def _geometry_notes(labels: List[str], res_atoms: List[List[Tuple[str, np.ndarray]]], segments: List[List[int]],
                    pairs: List[Tuple[int, int, Tuple, str]], bases: Optional[Sequence[str]] = None) -> List[str]:
    """Link, H-bond, mismatch and close-contact report of a rebuilt model.

    ``labels``: residue labels; ``res_atoms``: atoms per residue (with hydrogens for an honest contact
    check); ``segments``: residue indices of each strand (5'->3'); ``pairs``: (residue i, residue j,
    H-bonds of the canonical pair or (), description); ``bases``: base letter per residue (None: 'A',
    which only matters for the N/O donor / acceptor classes)."""
    coords = [{n: np.asarray(x) for n, x in atoms} for atoms in res_atoms]
    out = []
    links = [(a, b, float(np.linalg.norm(coords[a]["O3'"] - coords[b]["P"]))) for seg in segments
             for a, b in zip(seg, seg[1:]) if "O3'" in coords[a] and "P" in coords[b]]
    if links:
        d = [x for *_, x in links]
        out.append(f"O3'-P links: {min(d):.2f}-{max(d):.2f} Å over {len(d)} links (ideal {LINK:.2f} Å)")
        gaps = [f"{labels[a]}-{labels[b]} {x:.2f} Å" for a, b, x in links if not LINK_RANGE[0] <= x <= LINK_RANGE[1]]
        if gaps:
            out.append(f"WARNING: {len(gaps)} O3'-P link(s) outside {LINK_RANGE[0]}-{LINK_RANGE[1]} Å (the backbone "
                       "could not be closed there; minimise before use): " + ", ".join(gaps[:8])
                       + (" ..." if len(gaps) > 8 else ""))
    hb = [float(np.linalg.norm(coords[i][x] - coords[j][y])) for i, j, bonds, _ in pairs for x, y, *_ in bonds]
    if hb:
        out.append(f"H-bonds of the {sum(1 for p in pairs if p[2])} canonical pair(s): donor-acceptor "
                   f"{min(hb):.2f}-{max(hb):.2f} Å")
    for i, j, bonds, what in pairs:
        if bonds:
            continue
        bi = [x for n, x in coords[i].items() if n in _ALL_BASE_NAMES and not n.startswith("H")]
        bj = [x for n, x in coords[j].items() if n in _ALL_BASE_NAMES and not n.startswith("H")]
        d = min(np.linalg.norm(x - y) for x in bi for y in bj)
        out.append(f"WARNING: mismatch {labels[i]} · {labels[j]}: {what} (closest base heavy atoms {d:.1f} Å"
                   + (", the bases collide" if d < CLASH else "") + ")")
    bases = list(bases) if bases is not None else ["A"] * len(res_atoms)
    contacts = _close_contacts(res_atoms, segments, bases,
                               [(i, x, j, y) for i, j, bonds, _ in pairs for x, y, *_ in bonds])
    limits = (f"heavy atoms < {CONTACT_REPORT['heavy']:.1f} Å (N/O pairs that can H-bond < "
              f"{CONTACT_REPORT['hbond']:.1f} Å; three bonds apart < {CLASH_NB} Å), "
              f"H-H < {CONTACT_REPORT['hh']:.1f} Å, "
              f"H-heavy < {CONTACT_REPORT['hx']:.1f} Å (H-bonded H < {CONTACT_REPORT['polar']:.1f} Å); atoms four or "
              "more bonds apart, hydrogens included")
    if contacts:
        text = [f"{labels[a[0]]} {a[1]} - {labels[b[0]]} {b[1]} {d:.2f} Å ({what})" for d, what, a, b in contacts[:6]]
        out.append(f"WARNING: {len(contacts)} close contact(s) ({limits}; minimise before use): " + ", ".join(text)
                   + (" ..." if len(contacts) > 6 else ""))
    else:
        out.append(f"No close contacts ({limits})")
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
    h.add_argument("--pairing", choices=PAIRINGS, default="wc", help="wc: antiparallel Watson-Crick A-form "
                   "(default); reverse-wc (trans WC/WC), hoogsteen (cis WC/Hoogsteen, strand 1 the purine "
                   "strand) and hh (trans Hoogsteen/Hoogsteen A·A, as PDB 4JRD): parallel strands; "
                   "reverse-hoogsteen (trans WC/Hoogsteen): antiparallel")
    h.add_argument("--triplex", choices=("none",) + TRIPLEX_MOTIFS, default="none",
                   help="add a third strand on the Hoogsteen edge of the strand-1 purines: hoogsteen (pyrimidine "
                        "motif, parallel) or reverse-hoogsteen (purine motif, antiparallel); needs --pairing wc")
    h.add_argument("--third-strand", "--third", dest="third", default="auto",
                   help="third strand 5'->3' (same length) or auto (the canonical motif bases)")
    h.add_argument("--chains", default="A,B", help="chain ids of strand 1 and 2 (and 3 of a triplex; default "
                   "A,B / A,B,C)")
    h.add_argument("--start", default="1,1", help="first residue numbers of strand 1 and 2 (and 3; default 1,1; "
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
                           notes=notes, pairing=getattr(args, "pairing", "wc"),
                           triplex=getattr(args, "triplex", None), third=getattr(args, "third", "auto"))
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
