"""Tests of the B-form DNA helices of the structure builder (mdatk.builder, ``na='dna'``): the Arnott B-DNA
fiber geometry, hydrogens, residue names per scheme and hand, the L mirror image, G·T wobbles, mutations of
DNA residues, the command line and the model check."""

import math

import numpy as np
import pytest

from mdatk import builder as B
from mdatk import pdbtools as pt
from mdatk.secstruct import WATSON_CRICK, WOBBLE, PairDetector

SEQ = "GATCGTACGT"


# --- helpers ----------------------------------------------------------------------------------------

def dist(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))


def angle(a, b, c):
    u, v = np.subtract(a, b), np.subtract(c, b)
    return float(np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = np.subtract(p0, p1), np.subtract(p2, p1), np.subtract(p3, p2)
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - b0 @ b1 * b1, b2 - b2 @ b1 * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def pucker(c):
    """Altona-Sundaralingam phase and amplitude (degrees) of a sugar."""
    ring = ["C4'", "O4'", "C1'", "C2'", "C3'"]
    nu = [dihedral(*(c[ring[(k + j) % 5]] for j in range(4))) for k in range(5)]  # nu0 .. nu4
    a = ((nu[4] + nu[1]) - (nu[3] + nu[0])) / (2 * nu[2] * (math.sin(math.radians(36)) + math.sin(math.radians(72))))
    phase = math.degrees(math.atan(a)) + (180.0 if nu[2] < 0 else 0.0)
    return phase % 360.0, nu[2] / math.cos(math.radians(phase))


def dna(seq=SEQ, **kw):
    notes = []
    text = B.build_helix(seq, na="dna", notes=notes, **kw)
    return pt.read_pdb(text), text, notes


def links(s):
    return [dist(a.coords()["O3'"], b.coords()["P"]) for seg in s.segments for a, b in zip(seg, seg[1:])]


def pairs_of(s, wobble=True):
    residues = list(s.residues())
    index, rows, n = [], [], 0
    for r in residues:
        c = r.coords()
        index.append({k: n + i for i, k in enumerate(c)})
        rows += list(c.values())
        n += len(c)
    found = PairDetector([r.name for r in residues], index, cutoff=3.5, wobble=wobble).detect(np.array(rows))
    return [(residues[i], residues[j], k) for i, j, k in found], residues


def base(res):
    return pt.parse_residue_name(res.name).base


# --- the Arnott B-DNA geometry ------------------------------------------------------------------------

@pytest.mark.parametrize("chirality", ["D", "L"])
def test_b_dna_fiber_geometry(chirality):
    s, text, notes = dna(chirality=chirality)
    one, two = s.segments
    n = len(SEQ)
    assert [len(seg) for seg in s.segments] == [n, n] and s.chain_ids() == ["A", "B"]
    sign = 1.0 if chirality == "D" else -1.0  # the mirror image: torsions and volumes of the opposite sign
    # O3'-P links on both strands as the fiber model's (1.595 Å, O3'-P-O5' 101.5°); its phosphate turned 6° about
    # O3'-O5' opens C3'-O3'-P (published 119.4°) and closes P-O5'-C5' (118.8°), and OP2 is 2.88 Å from the previous
    # C2' instead of 2.69 Å (epsilon, zeta, alpha, beta: published 155.0, -95.1, -46.9, -146.0)
    d = links(s)
    assert len(d) == 2 * (n - 1) and 1.55 <= min(d) and max(d) <= 1.65
    assert d == pytest.approx([1.595] * len(d), abs=0.002)
    for seg in s.segments:
        for a, b in zip(seg, seg[1:]):
            ca, cb = a.coords(), b.coords()
            assert angle(ca["C3'"], ca["O3'"], cb["P"]) == pytest.approx(123.2, abs=0.1)
            assert angle(ca["O3'"], cb["P"], cb["O5'"]) == pytest.approx(101.5, abs=0.1)
            assert angle(cb["P"], cb["O5'"], cb["C5'"]) == pytest.approx(115.9, abs=0.1)
            assert dist(ca["C2'"], cb["OP2"]) == pytest.approx(2.88, abs=0.01)
            epsilon = dihedral(ca["C4'"], ca["C3'"], ca["O3'"], cb["P"])
            zeta = dihedral(ca["C3'"], ca["O3'"], cb["P"], cb["O5'"])
            alpha = dihedral(ca["O3'"], cb["P"], cb["O5'"], cb["C5'"])
            beta = dihedral(cb["P"], cb["O5'"], cb["C5'"], cb["C4'"])
            assert [sign * x for x in (epsilon, zeta, alpha, beta)] == pytest.approx([155.2, -90.6, -52.8, -143.3],
                                                                                     abs=0.15)
    # C3'-exo sugars (S-type, phase 192, amplitude 36), chi -98
    for r in s.residues():
        c = r.coords()
        phase, amp = pucker({k: np.multiply(v, [sign, 1, 1]) for k, v in c.items()})
        assert phase == pytest.approx(192.0, abs=1.0) and amp == pytest.approx(36.0, abs=1.0)
        n_, x_ = ("N9", "C4") if base(r) in "AG" else ("N1", "C2")
        assert sign * dihedral(c["O4'"], c["C1'"], c[n_], c[x_]) == pytest.approx(-98.0, abs=1.0)
        # OA -> OP1: the prochiral phosphate oxygens of a D-nucleotide (as the A-RNA), mirrored for L; the turn about
        # an axis through O5' leaves the volume as published
        if "P" in c:
            assert sign * pt.chiral_volume(c["P"], c["OP1"], c["OP2"], c["O5'"]) == pytest.approx(-2.42, abs=0.02)
    # Watson-Crick pairs of the fiber dyad
    for k in range(n):
        a, b = one[k], two[n - 1 - k]
        ca, cb = a.coords(), b.coords()
        assert dist(ca["C1'"], cb["C1'"]) == pytest.approx(10.86, abs=0.01)
        pair = base(a) + base(b)
        want = {"AT": [("N1", "N3", 2.95), ("N6", "O4", 2.81)],
                "TA": [("N3", "N1", 2.95), ("O4", "N6", 2.81)],
                "GC": [("N1", "N3", 2.91), ("N2", "O2", 3.01), ("O6", "N4", 2.70)],
                "CG": [("N3", "N1", 2.91), ("O2", "N2", 3.01), ("N4", "O6", 2.70)]}[pair]
        for x, y, value in want:
            assert dist(ca[x], cb[y]) == pytest.approx(value, abs=0.01), (a.label, b.label, x, y)
    found, residues = pairs_of(s)
    assert {(residues.index(a), residues.index(b)) for a, b, k in found if k == WATSON_CRICK} == {
        (i, 2 * n - 1 - i) for i in range(n)}
    # handedness: strand 1 turns by -36° per step about +z going down for D, +36° for L
    p = [np.array(r.coords()["C1'"]) for r in one[:2]]
    turn = wrap(math.degrees(math.atan2(p[1][1], p[1][0]) - math.atan2(p[0][1], p[0][0])))
    assert turn == pytest.approx(-sign * B.DNA_TWIST, abs=0.05)
    assert p[1][2] - p[0][2] == pytest.approx(-B.DNA_RISE, abs=0.01)
    # header and notes
    hand = "right-handed, D" if chirality == "D" else "left-handed, L: mirror image"
    assert text.splitlines()[:5] == [
        f"REMARK 951 MDATK IDEAL B-FORM DNA ({hand}), duplex of {n} base pairs",
        "REMARK 951 ARNOTT B-DNA FIBER MONOMER, TWIST 36.0 DEG, RISE 3.38 A, HELIX AXIS Z",
        "REMARK 951 PHOSPHATE TURNED 6 DEG ABOUT O3'-O5' (C2'(I)-OP2(I+1) 2.88 A)",
        f"REMARK 951 STRAND A {SEQ}", f"REMARK 951 STRAND B {B.complement(SEQ, 'dna')}"]
    assert notes[-1].startswith(f"Built an ideal B-form DNA duplex of {n} base pairs ({hand}):")
    assert notes[-2] == ("Phosphates turned 6° about O3'-O5' from the Arnott B-DNA fiber model to clear "
                         "C2'(i)-OP2(i+1): 2.88 Å, C3'-O3'-P 123.2°, P-O5'-C5' 115.9°")
    assert not any("2.69" in m for m in notes)
    # the sugar hand of every residue from its geometry
    chir = pt.assign_chirality(s)
    n_l = 2 * n if chirality == "L" else 0
    assert chir[0] == f"Chirality (sugar geometry): {n_l} L and {2 * n - n_l} D nucleotides"


def test_templates_and_atom_order():
    t = B.template("T", hydrogens=False, na="dna")
    assert list(t) == ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
                       "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"]
    for b in "AGC":  # the base atoms in the order of the A-RNA template
        assert list(B.template(b, False, na="dna"))[11:] == list(B.BASE_NAMES[b])
    assert B.DNA_BACKBONE_NAMES == B.BACKBONE_NAMES[:10] + ("C1'",) and "O2'" not in B.DNA_BACKBONE_NAMES
    with pytest.raises(B.BuildError, match=r"no DNA template for base 'U' \(A, C, G, T\)"):
        B.template("U", na="dna")
    with pytest.raises(B.BuildError, match=r"no template for base 'T' \(A, C, G, U\)"):
        B.template("T")
    # the monomer of nucgen's $ABDNA: the table as published, every atom of the template at its cylindrical
    # coordinates but the phosphate, which is the published one turned by DNA_PHOSPHATE_TURN (6°) about the line
    # from O3' of the previous nucleotide (one step back on the screw) to O5' (Rodrigues' rotation, written here)
    assert B._DNA_BACKBONE[0] == ("P", 8.91, 95.2, 2.08) and B.DNA_PHOSPHATE_TURN == 6.0
    cyl = {name: np.array([r * math.cos(math.radians(phi)), r * math.sin(math.radians(phi)), z])
           for name, r, phi, z in B._DNA_BACKBONE + B._DNA_BASES["A"]}
    t = B.template("A", False, na="dna")
    assert [n for n in t if not np.allclose(t[n], cyl[n], atol=1e-9)] == ["P", "OP1", "OP2"]
    back = math.radians(B.DNA_TWIST)
    o3 = np.array([[math.cos(back), -math.sin(back), 0.0], [math.sin(back), math.cos(back), 0.0], [0, 0, 1]]
                  ) @ cyl["O3'"] + [0.0, 0.0, B.DNA_RISE]
    k = (cyl["O5'"] - o3) / np.linalg.norm(cyl["O5'"] - o3)
    th = math.radians(B.DNA_PHOSPHATE_TURN)
    for name in ("P", "OP1", "OP2"):
        v = cyl[name] - o3
        turned = o3 + v * math.cos(th) + np.cross(k, v) * math.sin(th) + k * (k @ v) * (1 - math.cos(th))
        assert t[name] == pytest.approx(turned, abs=1e-9), name
        assert dist(t[name], cyl[name]) == pytest.approx({"P": 0.106, "OP1": 0.229, "OP2": 0.229}[name], abs=1e-3)
    # the A-RNA template keeps its table's phosphate
    p = B.template("A", False)["P"]
    assert math.hypot(p[0], p[1]) == pytest.approx(8.71) and math.degrees(math.atan2(p[1], p[0])) == pytest.approx(
        70.5) and p[2] == pytest.approx(-3.75)


@pytest.mark.parametrize("chirality", ["D", "L"])
def test_dna_hydrogens(chirality):
    s, _text, _notes = dna(chirality=chirality, hydrogens=True)
    sugar_h = ["H5'", "H5''", "H4'", "H3'", "H2'", "H2''", "H1'"]
    base_h = {"A": ["H8", "H61", "H62", "H2"], "G": ["H8", "H1", "H21", "H22"], "C": ["H41", "H42", "H5", "H6"],
              "T": ["H3", "H71", "H72", "H73", "H6"]}
    sign = 1.0 if chirality == "D" else -1.0
    for seg in s.segments:
        for k, r in enumerate(seg):
            names = [a.name for a in r.atoms]
            hyd = [x for x in names if x.startswith("H")]
            ends = (["HO5'"] if k == 0 else []) + (["HO3'"] if k == len(seg) - 1 else [])
            assert hyd == sugar_h + base_h[base(r)] + ends, r.label
            assert "O2'" not in names and "HO2'" not in names
            c = r.coords()
            heavy = np.array([a.xyz for a in r.atoms if not a.name.startswith("H")])
            for x in hyd:  # every hydrogen 0.9-1.1 Å from its nearest heavy atom
                assert 0.9 < np.linalg.norm(heavy - c[x], axis=1).min() < 1.12, (r.label, x)
            # H2' is the methylene hydrogen cis to O3' (the CCD's name): negative volume C2'(C1', C3', H2') in D;
            # the L helix keeps the D names on the mirror image (as for L-RNA)
            assert dist(c["H2'"], c["O3'"]) == pytest.approx(2.50, abs=0.05) and dist(c["H2''"], c["O3'"]) > 3.2
            assert sign * pt.chiral_volume(c["C2'"], c["C1'"], c["C3'"], c["H2'"]) < -1.5
            assert sign * pt.chiral_volume(c["C2'"], c["C1'"], c["C3'"], c["H2''"]) > 1.5
            assert sign * pt.chiral_volume(c["C5'"], c["O5'"], c["C4'"], c["H5'"]) > 1.5  # as the RNA rule
            if base(r) == "T":  # the methyl, staggered on C5-C7
                for h, torsion in (("H71", 180.0), ("H72", 60.0), ("H73", -60.0)):
                    assert dist(c["C7"], c[h]) == pytest.approx(1.09, abs=0.002)
                    assert angle(c["C5"], c["C7"], c[h]) == pytest.approx(109.5, abs=0.05)
                    assert wrap(dihedral(c["C4"], c["C5"], c["C7"], c[h]) - sign * torsion) == pytest.approx(0.0,
                                                                                                         abs=0.05)
    # the hydrogens of the RNA helix are unchanged: H5'/H5'' by the same rule, no H2''
    rna = list(pt.read_pdb(B.build_helix("GCA", hydrogens=True)).residues())[1]
    assert [a.name for a in rna.atoms if a.name.startswith("H")][:7] == ["H5'", "H5''", "H4'", "H3'", "H2'", "HO2'",
                                                                         "H1'"]


def test_l_dna_is_the_exact_mirror_image():
    for kw in ({}, {"hydrogens": True, "phosphate5": True}, {"partner": "none"}, {"partner": "ATCGTACGGTC"[:10]}):
        d, _, _ = dna(**kw)
        lh, _, _ = dna(chirality="L", **kw)
        assert [(r.chain, r.resseq) for r in d.residues()] == [(r.chain, r.resseq) for r in lh.residues()]
        for rd, rl in zip(d.residues(), lh.residues()):
            assert [a.name for a in rd.atoms] == [a.name for a in rl.atoms]
            cl = rl.coords()
            for name, x in rd.coords().items():
                assert np.allclose(np.multiply(x, [-1, 1, 1]), cl[name], atol=2e-3)
    assert pt.assign_chirality(lh)[0].startswith("Chirality (sugar geometry): 20 L and 0 D")


# --- residue names ----------------------------------------------------------------------------------

@pytest.mark.parametrize("scheme, chirality, names", [
    ("amber", "D", ["DG5", "DA", "DC", "DT3", "DA5", "DG", "DT", "DC3"]),
    ("amber", "L", ["LDG5", "LDA", "LDC", "LDT3", "LDA5", "LDG", "LDT", "LDC3"]),
    ("ccd", "D", ["DG", "DA", "DC", "DT", "DA", "DG", "DT", "DC"]),
    ("ccd", "L", ["0DG", "0DA", "0DC", "0DT", "0DA", "0DG", "0DT", "0DC"]),
    ("rosetta", "L", ["0DG", "0DA", "0DC", "0DT", "0DA", "0DG", "0DT", "0DC"]),
    ("pdb", "D", ["DG", "DA", "DC", "DT", "DA", "DG", "DT", "DC"]),
    ("standard", "L", ["DG", "DA", "DC", "DT", "DA", "DG", "DT", "DC"]),
    (None, "D", ["DG", "DA", "DC", "DT", "DA", "DG", "DT", "DC"]),
    (None, "L", ["0DG", "0DA", "0DC", "0DT", "0DA", "0DG", "0DT", "0DC"]),
])
def test_dna_names_per_scheme_and_hand(scheme, chirality, names):
    s, text, notes = dna("GACT", scheme=scheme, chirality=chirality)
    assert [r.name for r in s.residues()] == names
    for r in s.residues():  # every name reads back as DNA of the built hand (the standard scheme drops the hand)
        info = pt.parse_residue_name(r.name)
        assert info.sugar == "dna" and info.chirality == ("D" if scheme == "standard" else chirality)
    if scheme is None:
        want = ("DA, DC, DG, DT" if chirality == "D" else "0DA, 0DC, 0DG, 0DT (the CCD L-deoxyribonucleotides)")
        assert f"Residue names {want}, ATOM records" in notes
    if scheme == "standard":
        assert any(n.startswith("WARNING: the standard scheme gives 8 L residue(s) plain D names") for n in notes)
    if scheme == "amber" and chirality == "L":
        assert any(n.startswith("L-DNA: 8 residue(s) (A:1-4, B:1-4) have MDATK's Amber-style names") for n in notes)
    if scheme == "pdb":
        assert all(ln.startswith("ATOM") for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM")))
    records = {ln[:6] for ln in B.build_helix("GACT", na="dna", chirality="L", scheme="pdb").splitlines()
               if ln.startswith(("ATOM", "HETATM"))}
    assert records == {"HETATM"}  # 0DA ... are no standard PDB names


def test_dna_terminal_names_single_residues_and_phosphate():
    names = [r.name for r in dna("GACT", scheme="amber", terminals=False)[0].residues()]
    assert names == ["DG", "DA", "DC", "DT", "DA", "DG", "DT", "DC"]
    assert [r.name for r in dna("G", scheme="amber")[0].residues()] == ["DGN", "DCN"]
    assert [r.name for r in dna("G", scheme="amber", chirality="L")[0].residues()] == ["LDGN", "LDCN"]
    assert [r.name for r in dna("G", scheme="ccd", chirality="L")[0].residues()] == ["0DG", "0DC"]
    s, _text, notes = dna("GACT", scheme="amber", phosphate5=True)
    first = s.segments[0][0]
    assert [a.name for a in first.atoms][:4] == ["OP3", "P", "OP1", "OP2"]
    c = first.coords()
    assert dist(c["OP3"], c["P"]) == pytest.approx(1.48, abs=0.01)
    # OP3 points at the O3' the previous nucleotide would have (the end of the axis of the phosphate turn): its angles
    # at P are those of the next link's O3'
    nxt = s.segments[0][1].coords()
    for x in ("O5'", "OP1", "OP2"):
        assert angle(c["OP3"], c["P"], c[x]) == pytest.approx(angle(c["O3'"], nxt["P"], nxt[x]), abs=0.15), x
    assert any("Amber 5' units have no 5'-phosphate" in n and "DA5/LDA5" in n for n in notes)
    assert not any(n.startswith("WARNING: 5'-terminal") for n in notes)  # the builder's own warning only
    assert "P" not in [a.name for a in dna("GACT")[0].segments[1][0].atoms]  # 5'-OH by default


# --- sequences, checks and errors ---------------------------------------------------------------------

def test_dna_sequences_checks_and_errors():
    assert B.check_na(None) == "rna" and B.check_na(" DNA ") == "dna" and B.NUCLEIC_ACIDS == ("rna", "dna")
    with pytest.raises(B.BuildError, match="unknown nucleic acid"):
        B.check_na("pna")
    assert B.parse_sequence("5'-ACG U-3'", na="dna") == ("ACGT", ["sequence: U read as T (DNA)"])
    assert B.parse_sequence("ACGT") == ("ACGU", ["sequence: T read as U (RNA)"])  # RNA as before
    with pytest.raises(B.BuildError, match=r"invalid letter\(s\) X \(DNA: A, C, G, T\)"):
        B.parse_sequence("ACGX", na="dna")
    assert B.complement("AACGT", na="dna") == "ACGTT" and B.complement("AACGU") == "ACGUU"
    assert B.auto_partner("AACG", na="dna") == "CGTT"
    assert B.resolve_partner("GGCAT", na="dna") == ("GGCAT", "ATGCC", ["partner strand: Watson-Crick complement"])
    assert B.resolve_partner("GGCAU", "AUGUC", na="dna")[1] == "ATGTC"
    assert B.pair_types("GGCAT", "ATGTC", na="dna") == ["wc", "wobble", "wc", "wc", "wc"]
    assert B.pair_types("GGCAT", "ATGTC") == ["wc", "mismatch", "wc", "mismatch", "mismatch"]  # RNA: no T pairs
    assert B.helix_kind(na="dna") == "B-form" and B.helix_kind() == "A-form"
    msgs = B.check_helix("GGCAT", "ATGTC", na="dna")
    assert ("info", "5 base pairs: 4 Watson-Crick, 1 G·T wobble, 0 mismatched") in msgs
    assert ("info", "the explicit partner is the Watson-Crick complement") in B.check_helix("GGCAT", "ATGCC",
                                                                                         na="dna")
    assert B.check_helix("GGCAT", "none", na="dna") == [
        ("info", "single strand of 5 nucleotides in the B-form helix geometry")]
    assert any(lvl == "warning" and "mismatched" in t for lvl, t in B.check_helix("GGCAT", "ATGAC", na="dna"))
    # DNA: Watson-Crick B-form duplexes and single strands only
    for kw, match in ((dict(pairing="hoogsteen"), "the hoogsteen pairing is built for RNA only"),
                      (dict(pairing="hh", partner="none"), "the hh pairing is built for RNA only"),
                      (dict(triplex="hoogsteen"), "triplexes are built for RNA only"),
                      (dict(triplex="reverse-hoogsteen", third="AGGAA"), "triplexes are built for RNA only")):
        with pytest.raises(B.BuildError, match="DNA is built as a Watson-Crick B-form duplex or a single strand: "
                                               + match):
            B.build_helix("AAGAG", na="dna", **kw)
        level, text = B.check_helix("AAGAG", kw.get("partner", "auto"), kw.get("pairing", "wc"), kw.get("triplex"),
                                    kw.get("third", "auto"), na="dna")[0]
        assert level == "error" and match in text
    with pytest.raises(B.BuildError, match="built for RNA only"):
        B.resolve_partner("AAGAG", pairing="reverse-wc", na="dna")
    with pytest.raises(B.BuildError, match="built for RNA only"):
        B.pairing_kinds("AAGAG", "CTCTT", "reverse-hoogsteen", na="dna")
    with pytest.raises(B.BuildError, match="unknown nucleic acid"):
        B.build_helix("GGC", na="xna")
    # the RNA defaults stay what they were
    assert B.build_helix("GGCAU") == B.build_helix("GGCAU", na="rna") == B.build_helix("GGCAU", na=None)
    s = pt.read_pdb(B.build_helix("GGCAU", "AAGCU", hydrogens=True))
    assert {pt.parse_residue_name(r.name).sugar for r in s.residues()} == {"rna"}


def test_dna_mismatches_keep_the_watson_crick_frame():
    s, _text, notes = dna("GGGG", partner="CCAC")
    assert any(n.startswith("WARNING: mismatch A:2 G · B:3 A: the bases collide") for n in notes)  # G2·A
    assert not any("wobble" in n for n in notes)
    assert links(s) == pytest.approx([1.595] * 6, abs=0.002)


# --- G·T wobbles ----------------------------------------------------------------------------------------

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80}  # Bondi radii, as the model check


def _wobble_fit(na, contacts=False):
    """In-plane turns of G and U / T about C1' (G strand 1, its partner the fiber dyad image, level 0) from the
    least squares of G O6-N3 and N1-O2 to 2.85 Å, as WOBBLE_TURN was derived; with ``contacts`` the heavy atoms
    of both bases also stay 0.05 Å outside the model check's clash distance (van der Waals contact - 0.4 Å) from
    their own sugar-phosphate, the neighbours of both strands (every base) and the partner backbone.  Returns
    the turns, the two H-bond distances and the worst contact (> 0: inside the clash distance).  Written
    independently of the builder's constants."""
    from scipy.optimize import least_squares

    other = "T" if na == "dna" else "U"
    twist, rise = (B.DNA_TWIST, B.DNA_RISE) if na == "dna" else (B.TWIST, B.RISE)
    names = B.DNA_BASE_NAMES if na == "dna" else B.BASE_NAMES
    backbone = B.DNA_BACKBONE_NAMES if na == "dna" else B.BACKBONE_NAMES
    own = [n for n in backbone if n not in ("C1'", "C2'", "O4'")]  # three or more bonds from the base atoms
    tg, tx = B.template("G", False, na=na), B.template(other, False, na=na)
    near1, near2 = [], []  # the neighbours (levels -1, +1) of strand 1 and of strand 2
    for b in "ACG" + other:
        for k in (-1, 1):
            for n, x in B.template(b, False, na=na).items():
                y = B._screw(x, k, twist, rise)
                near1.append((n, y))
                near2.append((n, B._dyad(y)))

    def turned(t, b, deg):
        c = {k: v.copy() for k, v in t.items()}
        B._turn_base(c, b, deg, False, na)
        return c

    def inside(here, base, env):  # (clash distance + 0.05) - distance of the base atoms (but the glycosidic N)
        atoms = names[base][1:]
        a, e = np.array([here[x] for x in atoms]), np.array([v for _n, v in env])
        limit = np.add.outer([VDW[x[0]] for x in atoms], [VDW[n[0]] for n, _v in env]) - 0.4 + 0.05
        return (limit - np.linalg.norm(a[:, None] - e[None], axis=-1)).ravel()

    def parts(p):
        g, x = turned(tg, "G", p[0]), {k: B._dyad(v) for k, v in turned(tx, other, p[1]).items()}
        hb = np.array([dist(g["O6"], x["N3"]), dist(g["N1"], x["O2"])])
        over = np.concatenate([inside(g, "G", near1 + [(n, x[n]) for n in backbone] + [(n, g[n]) for n in own]),
                               inside(x, other, near2 + [(n, g[n]) for n in backbone] + [(n, x[n]) for n in own])])
        return hb, over

    def residuals(p):
        hb, over = parts(p)
        return np.concatenate([hb - 2.85] + ([10.0 * np.clip(over, 0.0, None)] if contacts else []))

    x = least_squares(residuals, np.zeros(2)).x
    hb, over = parts(np.round(x, 1))
    return np.round(x, 1), hb, over.max() - 0.05


def test_g_t_wobble_turns_are_derived_as_the_g_u_turns():
    # the least squares of WOBBLE_TURN reproduces it, and the A-form G·U stays outside the clash distance
    turns, hb, worst = _wobble_fit("rna")
    assert turns.tolist() == [B.WOBBLE_TURN["G"], B.WOBBLE_TURN["U"]] and worst < 0
    # on the B-form the same fit turns the T onto its own sugar and its 5' neighbour
    turns, hb, worst = _wobble_fit("dna")
    assert turns.tolist() == [-13.5, 11.5] and worst > 0.3
    # kept outside the clash distance, it gives the stored turns; a good fit (at most 0.3 Å off): a wobble
    turns, hb, worst = _wobble_fit("dna", contacts=True)
    assert turns.tolist() == [B.DNA_WOBBLE_TURN["G"], B.DNA_WOBBLE_TURN["T"]] and worst < -0.04
    assert hb == pytest.approx([2.73, 3.14], abs=0.01) and np.abs(hb - 2.85).max() <= 0.3


def test_g_t_wobble_pairs_of_an_explicit_partner():
    from mdatk import modelcheck as MC

    s, text, notes = dna("GGCAT", partner="ATGTC", hydrogens=True)  # pairs G2·T, the others Watson-Crick
    g, t = s.segments[0][1], s.segments[1][3]
    assert (g.name, t.name) == ("DG", "DT")
    cg, ct = g.coords(), t.coords()
    assert dist(cg["O6"], ct["N3"]) == pytest.approx(2.73, abs=0.01)
    assert dist(cg["N1"], ct["O2"]) == pytest.approx(3.14, abs=0.01)
    found, _ = pairs_of(s)
    kinds = {(a.chain, a.resseq): k for a, b, k in found}
    assert kinds[("A", 2)] == WOBBLE and [kinds[("A", i)] for i in (1, 3, 4, 5)] == [WATSON_CRICK] * 4
    assert ("1 G·T pair(s) built with an approximate wobble geometry (bases turned in their plane about C1'); "
            "minimise before production") in notes
    assert links(s) == pytest.approx([1.595] * 8, abs=0.002)  # the turns leave the backbone alone
    # T·G with the T on strand 1 and two wobbles in a row, both hands: no heavy-atom clash of the model check (the
    # phosphates turned as built)
    for seq, partner in (("GGCAT", "ATGTC"), ("CTGAG", "CTCGG"), ("CAGGAG", "CTTTTG"), ("CATTAG", "CTGGTG")):
        for chirality in "DL":
            r = MC.check_model(B.build_helix(seq, partner, na="dna", chirality=chirality))
            assert r.failed == [] and r.clashes.clashes == [], (seq, chirality)
    assert B.pair_types("CAGGAG", "CTTTTG", na="dna").count("wobble") == 2


# --- mutations ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scheme, chirality, before, after", [
    ("amber", "D", ("DG5", "DC", "DC3"), ("DT5", "DT", "DA3")),
    ("amber", "L", ("LDG5", "LDC", "LDC3"), ("LDT5", "LDT", "LDA3")),
    (None, "L", ("0DG", "0DC", "0DC"), ("0DT", "0DT", "0DA")),
    ("ccd", "D", ("DG", "DC", "DC"), ("DT", "DT", "DA")),
])
def test_dna_mutations_keep_names_hand_and_frame(scheme, chirality, before, after):
    src = B.build_helix("GACT", na="dna", chirality=chirality, scheme=scheme, hydrogens=True)
    old = {(r.chain, r.resseq): r for r in pt.read_pdb(src).residues()}
    text, notes = B.mutate(src, "A:1T, A:3U")
    s = pt.read_pdb(text)
    new = {(r.chain, r.resseq): r for r in s.residues()}
    keys = (("A", 1), ("A", 3), ("B", 4))
    assert tuple(old[k].name for k in keys) == before and tuple(new[k].name for k in keys) == after
    assert "A:3: U read as T (DNA)" in notes
    assert any("B:4 C->A" in n and "compensatory, compensates A:1 T" in n for n in notes)
    assert any("B:2 G->A" in n and "compensatory, compensates A:3 T" in n for n in notes)
    n_l = 8 if chirality == "L" else 0
    assert pt.assign_chirality(s)[0] == f"Chirality (sugar geometry): {n_l} L and {8 - n_l} D nucleotides"
    for k in keys:  # sugar and backbone verbatim, the new pyrimidine N1 on the old glycosidic N
        o, m = old[k], new[k]
        for a in o.atoms:
            if "'" in a.name or a.name in ("P", "OP1", "OP2"):
                assert m.coords()[a.name] == o.coords()[a.name]
        glyco = "N9" if base(o) in "AG" else "N1"
        assert dist(m.coords()[B.GLYCOSIDIC[base(m)][0]], o.coords()[glyco]) < 0.1
    # the new T has the B-DNA base geometry (its methyl carbon included)
    t, ref = new[("A", 1)].coords(), B.template("T", False, na="dna")
    for a, b in (("N1", "C2"), ("C4", "O4"), ("C5", "C7"), ("C6", "N1"), ("N3", "C4")):
        assert dist(t[a], t[b]) == pytest.approx(dist(ref[a], ref[b]), abs=2e-3)
    found, _ = pairs_of(s)
    assert len(found) == 4 and all(k == WATSON_CRICK for *_x, k in found)


def test_dna_mutation_options_and_errors():
    src = B.build_helix("GGCAT", na="dna")
    _text, notes = B.mutate(src, "A:2T", wobble=True)  # G2 -> T facing C: the complement A
    assert any("compensates A:2 T" in n and "C->A" in n for n in notes)
    wob = B.build_helix("GGCAT", partner="ATGTC", na="dna")  # A:2 G · B:4 T, found as a G·T wobble
    _text, notes = B.mutate(wob, "A:2A")
    assert "A:2 A: partner B:4 T already pairs (Watson-Crick)" in notes
    _text, notes = B.mutate(B.build_helix("GGCAT", partner="ATGCC", na="dna"), "B:4T", wobble=True)
    assert "B:4 T: partner A:2 G already pairs (G·T wobble)" in notes
    _text, notes = B.mutate(src, "A:2T, B:4T")
    assert any("do not pair canonically" in n for n in notes)
    with pytest.raises(B.BuildError, match="is an RNA nucleotide: T is a DNA base"):
        B.mutate(B.build_helix("GGCAU"), "A:1T")
    assert B.mutated_name("DA5", "T") == "DT5" and B.mutated_name("0DA", "T") == "0DT"
    assert B.mutated_name("LDA3", "T") == "LDT3" and B.mutated_name("THY", "A") == "DA"
    assert B.mutated_name("T5", "G") == "DG5" and B.mutated_name("LG5", "C") == "LC5"
    assert [m.base for m in B.parse_mutations("A:1t, B:2T")] == ["T", "T"]
    # a DNA strand on an RNA strand: no compensation across the two kinds
    lines = []
    for ln in B.build_helix("GGCAU").splitlines():
        if ln.startswith("ATOM") and ln[21] == "B":
            if ln[12:16].strip() == "O2'":
                continue
            ln = ln[:17] + f"{'D' + ln[17:20].strip().rstrip('35'):>3}" + ln[20:]
        lines.append(ln)
    hybrid = "\n".join(lines) + "\n"
    _text, notes = B.mutate(hybrid, "B:2G")
    assert any("its partner A:4 A is not a DNA nucleotide; no compensatory change" in n for n in notes)
    _text, notes = B.mutate(hybrid, "A:2C")
    assert any("is not an RNA nucleotide; no compensatory change" in n for n in notes)


# --- command line, citations and the model check ------------------------------------------------------

def test_dna_cli_logs_title_and_citations(tmp_path):
    from mdatk.cli import main
    from mdatk.config import Project
    from mdatk.projectfile import read_log

    proj = tmp_path / "p.mdatk.toml"
    Project().save(str(proj))
    l_dna, d_dna, rna = tmp_path / "l_dna.pdb", tmp_path / "d_dna.pdb", tmp_path / "rna.pdb"
    assert main(["build", "helix", "--seq", "GGCAT", "--na", "dna", "--chirality", "L", "--scheme", "amber",
                 "--hydrogens", "-o", str(l_dna), "-p", str(proj)]) == 0
    s = pt.read_pdb(str(l_dna))
    assert [r.name for r in s.residues()][:2] == ["LDG5", "LDG"]
    entry = read_log(str(proj))[-1]
    assert entry["step"] == "build helix" and entry["title"] == "Ideal B-form L-DNA helix GGCAT"
    assert entry["citations"] == ["arnott1972dna"] and "Built an ideal B-form DNA" in entry["summary"]
    assert main(["build", "helix", "--seq", "GGCAT", "--nucleic-acid", "dna", "-o", str(d_dna), "-p",
                 str(proj)]) == 0
    assert read_log(str(proj))[-1]["title"] == "Ideal B-form D-DNA helix GGCAT"
    assert [r.name for r in pt.read_pdb(str(d_dna)).residues()][:2] == ["DG", "DG"]
    assert main(["build", "helix", "--seq", "GGCAT", "-o", str(rna), "-p", str(proj)]) == 0  # RNA by default
    entry = read_log(str(proj))[-1]
    assert entry["title"] == "Ideal A-form D-RNA helix GGCAT" and entry["citations"] == ["arnott1972"]
    assert [r.name for r in pt.read_pdb(str(rna)).residues()][:2] == ["G", "G"]
    assert main(["build", "helix", "--seq", "AAGAG", "--na", "dna", "--pairing", "hoogsteen",
                 "-o", str(tmp_path / "bad.pdb"), "-p", str(proj)]) == 1
    assert read_log(str(proj))[-1]["status"] == "failed"
    mut = tmp_path / "mut.pdb"
    assert main(["build", "mutate", str(l_dna), "A:1U", "--no-compensatory", "-o", str(mut)]) == 0  # U read as T
    assert [r.name for r in pt.read_pdb(str(mut)).residues()][:2] == ["LDT5", "LDG"]
    assert main(["build", "mutate", str(rna), "A:1T", "-o", str(mut)]) == 1  # RNA: no T


def test_dna_citation():
    from mdatk import citations as C
    assert list(C.RESOURCES["arnott-bdna"].keys) == ["arnott1972dna"]
    ref = C.REFERENCES["arnott1972dna"]
    assert ref.doi == "10.1016/0006-291X(72)90243-4" and "1972, 47, 1504-1509" in ref.text
    assert "$ABDNA" in C.RESOURCES["arnott-bdna"].note


@pytest.mark.parametrize("chirality", ["D", "L"])
@pytest.mark.parametrize("hydrogens", [False, True])
def test_model_check_of_dna_duplexes(chirality, hydrogens, monkeypatch):
    """No failed gate, no link problem, hands as named and no heavy-atom clash, every contact 0.05 Å or more outside
    the clash distance (the fiber model's C2'(i)-OP2(i+1) of 2.69 Å is 2.88 Å with the phosphate turned).  The
    hydrogen contacts (reported only) list H2'(i)-OP2(i+1) (2.11 Å; clearing it would take a turn that makes the
    C3'-O3'-P / P-O5'-C5' angles validation outliers), the fiber model's own H1'(i)-H5'(i+1) (1.84 Å) and, before a
    thymine, the methyl against H2' and C8 / C6 of the 5' neighbour (1.94 / 2.48 Å)."""
    from mdatk import modelcheck as MC

    s, text, _notes = dna(chirality=chirality, hydrogens=hydrogens)
    r = MC.check_model(text, MC.QCSettings(hydrogen_check=True))
    assert r.failed == [] and r.warnings == [] and r.nucleotides == 20
    assert r.backbone.problems == [] and r.backbone.strained == [] and r.backbone.pp == []
    assert {link.status for link in r.backbone.links} == {"ok"} and len(r.backbone.links) == 18
    assert r.clashes.clashes == []
    assert MC.hand_counts(r) == f"A:10{chirality} B:10{chirality}" and r.hand_problems == []
    assert all(h.call == h.name_hand == chirality for h in r.hands)  # 0DA ... read as L-DNA
    # both clash distances 0.05 Å closer: still no heavy-atom clash, hydrogen contacts of the fiber model only
    monkeypatch.setattr(MC, "H_CLASH", MC.H_CLASH - 0.05)
    near = MC.check_model(text, MC.QCSettings(hydrogen_check=True, clash_tol=MC.CLASH_TOL - 0.05))
    assert near.clashes.clashes == []
    if not hydrogens:
        assert r.hydrogens.hydrogens == 0 and near.hydrogens.clashes == []
        return
    steps = {(f"{seg[0].chain}:{a.resseq}", f"{seg[0].chain}:{b.resseq}") for seg in s.segments
             for a, b in zip(seg, seg[1:])}
    found = {}
    for c in near.hydrogens.clashes:
        (first, x), (second, y) = c.first.rsplit(":", 1), c.second.rsplit(":", 1)
        assert (first, second) in steps, c  # residue i, then i + 1 of the same strand
        found.setdefault((x, y), []).append(c.distance)
    # GATCGTACGT / ACGTACGATC: 18 steps, 5 of them from a purine to a thymine
    assert {k: len(v) for k, v in found.items()} == {("H1'", "H5'"): 18, ("H2'", "OP2"): 18, ("H2'", "H71"): 5,
                                                     ("C8", "H72"): 5}
    for k, value in ((("H1'", "H5'"), 1.84), (("H2'", "OP2"), 2.11), (("H2'", "H71"), 1.94), (("C8", "H72"), 2.48)):
        assert found[k] == pytest.approx([value] * len(found[k]), abs=0.01), k
    # as clashes (overlap 0.4 Å): H1'-H5', H2'-OP2 and the methyl's H71 at the H2' (0.40 Å); C8-H72 just outside (0.39)
    assert sorted({(c.first.rsplit(":", 1)[1], c.second.rsplit(":", 1)[1]) for c in r.hydrogens.clashes}) == [
        ("H1'", "H5'"), ("H2'", "H71"), ("H2'", "OP2")] and len(r.hydrogens.clashes) == 41


def test_dna_phosphate_turn_is_the_smallest_whole_turn_clear_of_the_heavy_atom_clash(monkeypatch):
    """DNA_PHOSPHATE_TURN: of the whole turns from 5 to 10 degrees, the smallest that keeps every heavy-atom contact
    of P, OP1 and OP2 0.05 Å outside the model check's clash distance (C2'(i)-OP2(i+1)), with C3'-O3'-P and
    P-O5'-C5' inside 4 standard deviations of Parkinson et al. 1996 (119.7 +- 1.2, 120.9 +- 1.6 degrees), where
    validation reports angle outliers.  Clearing the hydrogen contact H2'(i)-OP2(i+1) as well would take 10, past that
    limit; the contacts of the other atoms do not change with the turn."""
    from mdatk import modelcheck as MC

    limit, h_limit = MC.CLASH_TOL - 0.05, MC.H_CLASH - 0.05
    monkeypatch.setattr(MC, "H_CLASH", 0.0)  # every hydrogen overlap, and with clash_tol 0 every heavy-atom one
    settings = MC.QCSettings(hydrogen_check=True, clash_tol=0.0, checks=("clashes",))

    def phosphate(c):
        return any(x.rsplit(":", 1)[1] in ("P", "OP1", "OP2") for x in (c.first, c.second))

    worst, others, angles = {}, {}, {}
    saved = B.DNA_PHOSPHATE_TURN
    try:
        for turn in range(5, 11):
            B.DNA_PHOSPHATE_TURN = float(turn)
            B._dna_phosphate.cache_clear()
            B._template.cache_clear()
            r = MC.check_model(B.build_helix(SEQ, na="dna", hydrogens=True), settings)
            lists = (r.clashes.clashes, r.hydrogens.clashes)
            worst[turn] = [max((c.overlap for c in items if phosphate(c)), default=0.0) for items in lists]
            others[turn] = [sorted(c for c in items if not phosphate(c)) for items in lists]
            _c2_op2, c3_o3_p, p_o5_c5 = B._dna_link_geometry()
            angles[turn] = ((c3_o3_p - 119.7) / 1.2, (120.9 - p_o5_c5) / 1.6)
    finally:
        B.DNA_PHOSPHATE_TURN = saved
        B._dna_phosphate.cache_clear()
        B._template.cache_clear()
    clear = [turn for turn, (heavy, _h) in worst.items() if heavy <= limit and max(angles[turn]) < 4.0]
    assert clear[0] == B.DNA_PHOSPHATE_TURN == 6
    assert worst[5][0] > limit >= worst[6][0] and worst[6][0] == pytest.approx(0.344, abs=0.003)
    assert angles[6] == pytest.approx((2.9, 3.1), abs=0.05)
    # the hydrogen contact of the phosphate clears only at 10, where both angles are outliers
    assert [turn for turn in worst if worst[turn][1] <= h_limit] == [10] and min(angles[10]) > 4.0
    assert all(others[turn] == others[6] for turn in others)