"""Tests of the D/L chimeras of the structure builder (mdatk.chimera): the design strings and their errors, the
closure of every link for RNA and DNA, the model check of the written models, the exact mirror twins, chains,
numbering, the bridging PO2 with its LINK and CONECT records, residue names per scheme (internal names at the
junctions), hydrogens, mixed and same-hand designs, the command line, progress and cancel.

Segments of 1-3 base pairs keep the fits short; the fits are cached by design, so the tests that only read a
model again (names, chains, hydrogens) reuse them."""

import math
import re

import numpy as np
import pytest

from mdatk import builder as B
from mdatk import chimera as C
from mdatk import modelcheck as MC
from mdatk import pdbtools as pt

LINKS = ["3'-5'", "5'-3'", "3'-3'", "5'-5'"]


# --- helpers ----------------------------------------------------------------------------------------

def build(design, **kw):
    notes = []
    text = C.build_chimera(design, notes=notes, **kw)
    return pt.read_pdb(text), text, notes


def coords(res):
    return {n: np.array(x) for n, x in res.coords().items()}


def residue(s, chain, number):
    return next(r for r in s.residues() if r.chain == chain and r.resseq == number)


def dist(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))


def angle(a, b, c):
    u, v = np.subtract(a, b), np.subtract(c, b)
    return math.degrees(math.acos(float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v))))


def junction_atoms(s, link):
    """The atoms of the single junction of a two-segment design (as written): {name: xyz}."""
    if link in C.BRIDGES:
        o, c = ("O3'", "C3'") if link == "3'-3'" else ("O5'", "C5'")
        (rec_a, rec_b) = pt.link_records(s)
        ra, po2, rb = (residue(s, rec_a.first.chain, rec_a.first.resseq), residue(s, rec_a.second.chain,
                       rec_a.second.resseq), residue(s, rec_b.second.chain, rec_b.second.resseq))
        a, p, b = coords(ra), coords(po2), coords(rb)
        return {"Oa": a[o], "Ca": a[c], "P": p["P"], "Ob": b[o], "Cb": b[c]}
    chain = s.segments[0]
    hands = [r.chirality or pt.parse_residue_name(r.name).chirality for r in chain]
    k = next(i for i in range(len(chain) - 1) if hands[i] != hands[i + 1])  # the D/L step of strand 1
    a, b = coords(chain[k]), coords(chain[k + 1])
    return {"O3'": a["O3'"], "C3'": a["C3'"], "P": b["P"], "O5'": b["O5'"]}


def hand_of(res):
    nt = pt.parse_residue_name(res.name)
    return nt.chirality if nt else None


def cross_segment_contacts(s):
    """Heavy-atom distances between the D and the L residues (the two segments of a D/L design) more than three
    bonds apart, through the junction bonds the model check knows (O3'-P of a chain, LINK records)."""
    m = MC.build_model(s, MC.QCSettings())
    hands = [hand_of(m.residues[m.res[i]]) for i in range(len(m.xyz))]
    cache = {}
    out = []
    for i, j, d in MC.pairs_within(m.xyz, 3.6):
        if None in (hands[i], hands[j]) or hands[i] == hands[j]:
            continue
        if MC._bond_distances(m, i, cache).get(j, 9) <= 3:
            continue
        out.append((d, m.label(i), m.label(j)))
    return sorted(out)


def check_passes(text, links):
    """The model check of a written chimera: no failed gate or warning, no heavy-atom clash, every LINK record used
    as a bond."""
    r = MC.check_model(text, MC.QCSettings())
    assert r.failed == [] and r.warnings == [] and r.clashes.clashes == [], (r.failed, r.warnings, r.clashes.clashes)
    records = pt.link_records(pt.read_pdb(text))
    assert len(records) == 2 * sum(link in C.BRIDGES for link in links)
    assert len(r.clashes.links) == len(records) and r.clashes.links_unused == 0
    assert all(1.55 <= b.distance <= 1.65 for b in r.clashes.links)
    assert r.backbone.problems == [] and r.backbone.strained == []
    return r


# --- designs ----------------------------------------------------------------------------------------

def test_designs_are_read_with_their_hands_nucleic_acids_partners_and_directions():
    segs, links, notes = C.parse_design("D-RNA:GGCAUC 3'-5' l-dna:GAtgcc/none 3'3' D:ACGU/ACGU")
    assert [(s.chirality, s.na, s.sequence, s.partner) for s in segs] == [
        ("D", "rna", "GGCAUC", "GAUGCC"), ("L", "dna", "GATGCC", None), ("D", "rna", "ACGU", "ACGU")]
    assert links == ["3'-5'", "3'-3'"] and notes == []
    assert C._read53(links) == [True, True, False]  # after 3'-3' the third segment is read 3'->5'
    assert C.describe((segs, links)) == "5'-D-RNA(6)-p-L-DNA(6)-3'p3'-D-RNA(4)-5'"
    assert C.describe("D-RNA:AAAAAA 3'-5' L-RNA:AAAAAA") == "5'-D-RNA(6)-p-L-RNA(6)-3'"
    assert C.describe("D-DNA:AAAAAA 3'-3' L-DNA:AAAAAA") == "5'-D-DNA(6)-3'p3'-L-DNA(6)-5'"
    assert C.describe("L:GGCAUC/none 3'-5' D:GAUGCC") == "5'-L-RNA(6)-p-D-RNA(6)-3'"
    assert C.describe("D-RNA:GGCC 3'-5' D-DNA:GGCC") == "5'-D-RNA(4)-p-D-DNA(4)-3'"
    # a first link from the 5' end reads segment 1 3'->5'; 5'-5' ends at 5', so segment 2 is read 5'->3'
    assert C.describe("D:GG 5'-5' L:CC") == "3'-D-RNA(2)-5'p5'-L-RNA(2)-3'"
    assert C.describe("D:GG 5'-3' L:CC 5'-5' D:AA") == "3'-D-RNA(2)-p-L-RNA(2)-5'p5'-D-RNA(2)-3'"
    # the build's na for segments that name none; T / U read for the nucleic acid, with notes
    segs, _links, notes = C.parse_design("D:GGTA 3-5 L-RNA:UUCC/auto", na="dna")
    assert [(s.na, s.sequence, s.partner) for s in segs] == [("dna", "GGTA", "TACC"), ("rna", "UUCC", "GGAA")]
    segs, _links, notes = C.parse_design("D:GGTA/UACC 35 L:CC/-")
    assert segs[0].sequence == "GGUA" and segs[0].partner == "UACC" and segs[1].partner is None
    assert notes == ["segment 1 sequence: T read as U (RNA)"]
    for spelling in ("3'-5'", "3-5", "35", "3'5'", "3’-5’", "3′5′"):
        assert C.link_kind(spelling) == "3'-5'"
    assert [C.link_kind(t) for t in ("5-3", "33", "5'5'", "3--5", "3-4", "D:AA")] == ["5'-3'", "3'-3'", "5'-5'",
                                                                                        None, None, None]
    assert C.nucleic_acids("D-DNA:GG 3'-5' L-RNA:CC 3'-5' D-DNA:AA") == ["dna", "rna"]


@pytest.mark.parametrize("design, message", [
    ("", "the design is empty"),
    ("D:GGC", "at least two segments"),
    ("D:GGC 3'-5'", "the design ends with the link 3'-5': a segment must follow it"),
    ("3'-5' D:GGC L:GCC", "the design starts with the link 3'-5' where a segment belongs"),
    ("D:GGC 3'-5' 3'-5' L:GCC", "link 1 is followed by the link 3'-5' where a segment belongs"),
    ("D:GGC L:GCC", "segments 1 and 2 need a link between them"),
    ("D:GGC 3-4 L:GCC", "unknown link 3-4 after segment 1 (3'-5', 5'-3', 3'-3' or 5'-5'"),
    ("X:GGC 3'-5' L:GCC", "segment 1 'X:GGC': the hand must be D or L"),
    ("D-PNA:GGC 3'-5' L:GCC", "segment 1 'D-PNA:GGC': unknown nucleic acid 'PNA' (RNA or DNA)"),
    ("D:GGC 3'-5' GCC", "segment 2 'GCC': write HAND[-NA]:SEQUENCE[/PARTNER]"),
    ("D:GGC 3'-5' L:GXC", "segment 2 sequence: invalid letter(s) X (RNA: A, C, G, U)"),
    ("D:GGC 3'-5' L-DNA:GUC/GAX", "segment 2 partner: invalid letter(s) X (DNA: A, C, G, T)"),
    ("D:GGC 3'-5' L:", "segment 2 sequence is empty"),
    ("D:GGC/ 3'-5' L:GCC", "segment 1 'D:GGC/': nothing after '/'"),
    ("D:GGC/GC 3'-5' L:GCC", "segment 1: the partner has 2 nucleotides but the sequence has 3"),
    ("D:GGC 3'-5' L:GCC 5'-3' D:AA", "link 2 (5'-3') starts at a 5' end, but the free end of segment 2 (L-RNA:GCC, "
                                    "read 5'->3' after link 1 3'-5') is its 3' end: use 3'-5' or 3'-3'"),
    ("D:GGC 3'-3' L:GCC 3'-5' D:AA", "link 2 (3'-5') starts at a 3' end, but the free end of segment 2 (L-RNA:GCC, "
                                    "read 3'->5' after link 1 3'-3') is its 5' end: use 5'-3' or 5'-5'"),
])
def test_design_errors_name_the_token(design, message):
    with pytest.raises(B.BuildError) as exc:
        C.parse_design(design)
    assert message in str(exc.value)
    assert C.check_chimera(design) == [("error", str(exc.value))]
    with pytest.raises(B.BuildError):
        C.build_chimera(design)


def test_build_errors_chains_scheme_size_and_nucleic_acid(monkeypatch):
    design = "D:GC 3'-3' L:GC"  # four chains: A (with the PO2), B, then the partners C, D
    for chains, message in (("A,B,C", "the design writes 4 chains"), ("A,B,C,C", "every chain needs its own id"),
                            ("A,BB,C,D", "chain ids must be single characters")):
        with pytest.raises(B.BuildError, match=message):
            C.build_chimera(design, chains=chains)
    with pytest.raises(B.BuildError, match="unknown residue name scheme 'nope'"):
        C.build_chimera(design, scheme="nope")
    with pytest.raises(B.BuildError, match="unknown nucleic acid 'xna'"):
        C.parse_design(design, na="xna")
    monkeypatch.setattr(C, "MAX_ATOMS", 200)  # 163 heavy atoms, 257 with hydrogens, and 4 TER records
    assert C.check_chimera(design)[0][0] == "info"
    level, text = C.check_chimera(design, hydrogens=True)[0]
    assert level == "error" and text.startswith("too many atoms: 257 atoms and 4 TER records do not fit the 200")
    with pytest.raises(B.BuildError, match="too many atoms"):
        C.build_chimera(design, hydrogens=True)


def test_check_chimera_reports_the_design_pairs_and_size():
    msgs = C.check_chimera("D:GGTU 3'-5' L-DNA:GACC/GGUA 3'-3' D:ACG/none")
    assert msgs[0] == ("info", "5'-D-RNA(4)-p-L-DNA(4)-3'p3'-D-RNA(3)-5': 3 segments, 2 link(s) (3'-5', 3'-3')")
    assert ("info", "segment 1 sequence: T read as U (RNA)") in msgs
    assert ("info", "segment 2 partner: U read as T (DNA)") in msgs
    assert ("info", "segment 1 (D-RNA): 4 base pairs: 4 Watson-Crick, 0 G·U wobble, 0 mismatched") in msgs
    assert ("info", "segment 2 (L-DNA): 4 base pairs: 3 Watson-Crick, 0 G·T wobble, 1 mismatched") in msgs
    assert ("warning", "segment 2 (L-DNA): mismatched pairs keep the Watson-Crick frame (bases may collide; minimise "
                       "or remodel): 1:G·A") in msgs
    assert ("info", "segment 3 (D-RNA): single strand of 3 nucleotides in the A-form helix geometry") in msgs
    assert msgs[-1][1].startswith("392 atoms (heavy atoms only)")
    # the count is the model's: heavy atoms, hydrogens, 5'-phosphates, the bridge
    for design, kw in (("D:GC 3'-5' L:GC", {}), ("D:GC 3'-5' L:GC", {"hydrogens": True}),
                       ("D:GC 3'-5' L:GC", {"phosphate5": True}), ("D:GC 3'-3' L:GC", {})):
        s, _t, _n = build(design, **kw)
        count = sum(len(r.atoms) for r in s.residues())
        assert C._atom_count(*C.parse_design(design)[:2], kw.get("hydrogens", False),
                             kw.get("phosphate5", False)) == count


# --- every link closes, for RNA and DNA ------------------------------------------------------------

@pytest.mark.parametrize("na", ["rna", "dna"])
@pytest.mark.parametrize("link", LINKS)
def test_every_link_closes_and_passes_the_model_check(link, na):
    """Ideal bond lengths and angles at the junction (O-P 1.55-1.65 Å, angles within 10 degrees of their targets),
    no heavy-atom contact below 3.0 Å between the segments beyond the junction's 1-4 pairs (3.0 Å: the closest a
    pair four or five bonds apart through it may come, within the 0.03 Å a soft contact term leaves), and a model
    check with no clash and the LINK records used as bonds."""
    s, text, notes = build(f"D:GC {link} L:GC", na=na)
    a = junction_atoms(s, link)
    if link in C.BRIDGES:
        for o, c in (("Oa", "Ca"), ("Ob", "Cb")):
            assert 1.55 <= dist(a[o], a["P"]) <= 1.65
            assert abs(angle(a[c], a[o], a["P"]) - C.C_O_P) < 10.0
        assert abs(angle(a["Oa"], a["P"], a["Ob"]) - C.O_P_O) < 10.0
    else:
        assert 1.55 <= dist(a["O3'"], a["P"]) <= 1.65
        assert abs(angle(a["C3'"], a["O3'"], a["P"]) - C.C_O_P) < 10.0
        assert abs(angle(a["O3'"], a["P"], a["O5'"]) - C.O_P_O) < 10.0
    contacts = cross_segment_contacts(s)
    assert contacts and contacts[0][0] >= 2.97, contacts[:3]
    check_passes(text, [link])
    junction = next(n for n in notes if n.startswith("Junction 1"))
    assert f"Junction 1 ({link}, " in junction and "end pairs" in junction and "closest heavy-atom contact" in junction
    assert ("gamma of" in junction) == (na == "dna" and link not in C.BRIDGES)  # a DNA phosphodiester turns
    assert ("bridging phosphate" in junction) == (link in C.BRIDGES)
    assert notes[-1].endswith("a starting model: minimise before production")
    assert "REMARK 951 RIGID-BODY JUNCTION: MINIMISE BEFORE PRODUCTION" in text


def test_a_design_read_from_either_end_gives_the_same_junction():
    """5'-3' turns the 5' end of the structure built so far: "D:GC 5'-3' L:GC" is 5'-L-p-D-3', the mirror image of
    "D:GC 3'-5' L:GC" (GC is self-complementary), so its fit is that fit mirrored: the same lengths, angles and
    stacking, the opposite gamma / beta turns of a DNA phosphodiester."""
    def geometry(note):  # the numbers of a junction note up to its contacts, the turns left out
        text = re.sub(r"; gamma of [^;]*", "", note.split("): ", 1)[1]).split("; closest")[0]
        return re.findall(r"\d+\.\d+", text)

    def turns(note):
        return [float(v) for v in re.findall(r"turned ([-+]\d+\.\d)°.*beta ([-+]\d+\.\d)°", note)[0]]

    for na in ("rna", "dna"):
        (j53,), (j35,) = ([n for n in build(f"D:GC {link} L:GC", na=na)[2] if n.startswith("Junction 1")]
                          for link in ("5'-3'", "3'-5'"))
        assert geometry(j53) == geometry(j35) and len(geometry(j35)) == 5  # O3'-P, 2 angles, stacking, target
        if na == "dna":
            assert turns(j53) == [-t for t in turns(j35)] and all(abs(t) > 0 for t in turns(j35))
        else:
            assert "gamma" not in j53 + j35


def test_rna_six_pair_junctions_reproduce_the_first_chimera_models():
    """The RNA models of MDATK's first chimeras (6 A·U pairs per segment): 3'-5' O3'-P 1.595 Å, C3'-O3'-P 120.7,
    O3'-P-O5' 104.6 degrees, end pairs 3.82 Å apart; 3'-3' O-P 1.600 / 1.601 Å, C3'-O3'-P 119.5 / 120.4, O3'-P-O3'
    103.7 degrees and the bridge 3.12 Å from its nearest atom outside the two ester groups.  Their 3'-3' end pairs
    were 5.24 Å apart: that fit stopped at its 600 evaluations (cost 0.0329); the same objective converges to cost
    0.0311 with the end pairs 5.07 Å apart, which is what the builder writes."""
    s, text, _n = build("D-RNA:AAAAAA 3'-5' L-RNA:AAAAAA")
    a = junction_atoms(s, "3'-5'")
    assert dist(a["O3'"], a["P"]) == pytest.approx(1.595, abs=0.05)
    assert angle(a["C3'"], a["O3'"], a["P"]) == pytest.approx(120.7, abs=3.0)
    assert angle(a["O3'"], a["P"], a["O5'"]) == pytest.approx(104.6, abs=3.0)
    assert stack(s, ("A", 6), ("B", 1), ("A", 7), ("C", 6)) == pytest.approx(3.82, abs=0.05)
    check_passes(text, ["3'-5'"])
    s, text, notes = build("D-RNA:AAAAAA 3'-3' L-RNA:AAAAAA")
    a = junction_atoms(s, "3'-3'")
    assert dist(a["Oa"], a["P"]) == pytest.approx(1.600, abs=0.05)
    assert dist(a["Ob"], a["P"]) == pytest.approx(1.601, abs=0.05)
    assert angle(a["Ca"], a["Oa"], a["P"]) == pytest.approx(119.5, abs=3.0)
    assert angle(a["Cb"], a["Ob"], a["P"]) == pytest.approx(120.4, abs=3.0)
    assert angle(a["Oa"], a["P"], a["Ob"]) == pytest.approx(103.7, abs=3.0)
    assert stack(s, ("A", 6), ("C", 1), ("B", 6), ("D", 1)) == pytest.approx(5.07, abs=0.05)
    # the bridge's nearest atom, the atoms of both ester groups (O3', C3', P, O5', OP1, OP2 of A:6 and B:6) left out
    near = {"O3'", "C3'", "P", "O5'", "OP1", "OP2"}
    bridge = [np.array(x) for x in residue(s, "A", 7).coords().values()]
    others = [np.array(a.xyz) for r in s.residues() if r.name != "PO2" for a in r.atoms
              if not ((r.chain, r.resseq) in (("A", 6), ("B", 6)) and a.name in near)]
    assert min(dist(p, q) for p in bridge for q in others) == pytest.approx(3.12, abs=0.05)
    check_passes(text, ["3'-3'"])
    assert any("closest contact of the bridging phosphate" in n for n in notes)


def stack(s, a1, a2, b1, b2):
    """Distance of the centroids of two base pairs (each the mean of its two base centroids)."""
    def centroid(chain, number):
        return np.mean([a.xyz for a in residue(s, chain, number).atoms if a.name not in C._BACKBONE
                        and not a.name.startswith("H")], axis=0)
    return dist(0.5 * (centroid(*a1) + centroid(*a2)), 0.5 * (centroid(*b1) + centroid(*b2)))


# --- mirror twins ---------------------------------------------------------------------------------

@pytest.mark.parametrize("design, twin, kw", [
    ("D:GC 3'-3' L:GC", "L:GC 3'-3' D:GC", {"hydrogens": True}),
    ("D-DNA:GC 3'-5' L-DNA:GC", "L-DNA:GC 3'-5' D-DNA:GC", {"scheme": "amber"}),
    ("D:GC 5'-5' L:GC", "L:GC 5'-5' D:GC", {"scheme": "ccd"}),
])
def test_a_design_and_its_mirror_twin_are_exact_mirror_images(design, twin, kw):
    s1, text1, _n1 = build(design, **kw)
    s2, text2, _n2 = build(twin, **kw)
    r1, r2 = list(s1.residues()), list(s2.residues())
    assert [(r.chain, r.resseq) for r in r1] == [(r.chain, r.resseq) for r in r2]
    for a, b in zip(r1, r2):
        assert [x.name for x in a.atoms] == [x.name for x in b.atoms]  # the bridging oxygens keep their names
        if a.name != "PO2":
            assert hand_of(a) != hand_of(b) and pt.parse_residue_name(a.name).base == \
                pt.parse_residue_name(b.name).base
        for x, y in zip(a.atoms, b.atoms):
            (x1, y1, z1), (x2, y2, z2) = x.xyz, y.xyz
            assert (x1, y1, z1) == (-x2, y2, z2), (a.label, x.name)
    ends = [[(a.name, a.chain, a.resseq) for r in pt.link_records(s) for a in (r.first, r.second)] for s in (s1, s2)]
    assert ends[0] == ends[1] and len(ends[0]) == (4 if "'p" in C.describe(design) else 0)


# --- chains, numbering, PO2, LINK and CONECT -------------------------------------------------------

def conect_bonds(text):
    """Bonds of the CONECT records as (chain, resseq, atom name) pairs."""
    serials = {}
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serials[int(line[6:11])] = (line[21], int(line[22:26]), line[12:16].strip())
    bonds = set()
    for line in text.splitlines():
        if line.startswith("CONECT"):
            ids = [int(line[k:k + 5]) for k in range(6, len(line), 5)]
            bonds |= {frozenset((serials[ids[0]], serials[i])) for i in ids[1:]}
    return bonds


def test_a_3_3_bridge_splits_strand_1_into_chains_with_a_po2_link_and_conect_records():
    s, text, notes = build("D:GC 3'-3' L:GC", scheme="amber")
    layout = [(seg[0].chain, [(r.resseq, r.name) for r in seg]) for seg in s.segments]
    assert layout == [("A", [(1, "G5"), (2, "C"), (3, "PO2")]), ("B", [(1, "LG5"), (2, "LC")]),
                      ("C", [(1, "G5"), (2, "C3")]), ("D", [(1, "LG5"), (2, "LC3")])]
    po2 = residue(s, "A", 3)
    assert [(a.record, a.name) for a in po2.atoms] == [("HETATM", "P"), ("HETATM", "OP1"), ("HETATM", "OP2")]
    assert {a.record for r in s.residues() if r.name != "PO2" for a in r.atoms} == {"ATOM"}
    links = pt.link_records(s)
    assert [(r.first, r.second) for r in links] == [
        (pt.LinkAtom("O3'", "C", "A", 2), pt.LinkAtom("P", "PO2", "A", 3)),
        (pt.LinkAtom("P", "PO2", "A", 3), pt.LinkAtom("O3'", "LC", "B", 2))]
    assert [r.length for r in links] == [pytest.approx(1.60, abs=0.01)] * 2
    assert conect_bonds(text) == {frozenset((("A", 2, "O3'"), ("A", 3, "P"))),
                                  frozenset((("A", 3, "P"), ("B", 2, "O3'")))}
    assert [line[:6] for line in text.splitlines()].index("CONECT") > [line[:3] for line in
                                                                        text.splitlines()].index("TER")
    header = [line for line in text.splitlines() if line.startswith("REMARK 951")]
    assert header[0] == "REMARK 951 MDATK CHIMERA 5'-D-RNA(2)-3'p3'-L-RNA(2)-5'"
    assert header[1:3] == ["REMARK 951 SEGMENT 1 A:1-2 D-RNA A-FORM, PARTNER C:1-2, GC",
                           "REMARK 951 SEGMENT 2 B:1-2 L-RNA A-FORM, PARTNER D:1-2, GC"]
    assert re.fullmatch(r"REMARK 951 JUNCTION 1 3'-3' A:2:O3'-A:3:P-B:2:O3' \(PO2\): O-P 1\.60/1\.60 A, C3'-O3'-P "
                        r"\d+\.\d/\d+\.\d, O3'-P-O3' \d+\.\d DEG, END PAIRS \d\.\d\d A, CLOSEST 3\.\d\d A, BRIDGE "
                        r"2\.\d\d A", header[3]), header[3]
    assert header[4:] == ["REMARK 951 RIGID-BODY JUNCTION: MINIMISE BEFORE PRODUCTION"]
    # tleap: a PO2 unit and the bond commands, residues numbered in file order (A:1 A:2 A:3 B:1 B:2 ...)
    warning = next(n for n in notes if n.startswith("WARNING: tleap has no unit for the bridging phosphate PO2"))
    assert "bond m.3.P m.2.O3' (A:3 P - A:2 O3'); bond m.3.P m.5.O3' (A:3 P - B:2 O3')" in warning
    # chain ids in that order on request
    s2, text2, _n = build("D:GC 3'-3' L:GC", scheme="amber", chains="W, X,Y,Z")
    assert [seg[0].chain for seg in s2.segments] == ["W", "X", "Y", "Z"]
    assert [(r.first.chain, r.second.chain) for r in pt.link_records(s2)] == [("W", "W"), ("W", "X")]
    assert [line[30:54] for line in text2.splitlines() if line.startswith(("ATOM", "HETATM"))] == \
        [line[30:54] for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def test_a_5_5_bridge_and_the_order_of_runs_and_partners():
    s, text, notes = build("D:GC 5'-5' L:GC", scheme="amber")
    layout = [(seg[0].chain, [(r.resseq, r.name) for r in seg]) for seg in s.segments]
    # segment 1 is read 3'->5' (the link starts at its 5' end) and written 5'->3': A:1 is bonded through O5'
    assert layout == [("A", [(1, "G"), (2, "C3"), (3, "PO2")]), ("B", [(1, "LG"), (2, "LC3")]),
                      ("C", [(1, "G5"), (2, "C3")]), ("D", [(1, "LG5"), (2, "LC3")])]
    assert [(r.first, r.second) for r in pt.link_records(s)] == [
        (pt.LinkAtom("O5'", "G", "A", 1), pt.LinkAtom("P", "PO2", "A", 3)),
        (pt.LinkAtom("P", "PO2", "A", 3), pt.LinkAtom("O5'", "LG", "B", 1))]
    for chain in "AB":  # the bridged 5' ends: no phosphate of their own, no hydroxyl hydrogen
        assert "P" not in residue(s, chain, 1).coords()
    assert any(n.startswith("WARNING: the residues bonded to a PO2 through O5' (A:1, B:1) have internal names")
               for n in notes)
    # a 5'-3' link: the run is written 5'->3', segment 2 first; one chain of strand 1, no LINK records
    s, text, _n = build("D:GC 5'-3' L:GC", scheme="amber")
    assert [(seg[0].chain, [r.name for r in seg]) for seg in s.segments] == [
        ("A", ["LG5", "LC", "G", "C3"]), ("B", ["G5", "C3"]), ("C", ["LG5", "LC3"])]
    assert pt.link_records(s) == [] and conect_bonds(text) == set()
    assert dist(residue(s, "A", 2).coords()["O3'"], residue(s, "A", 3).coords()["P"]) == pytest.approx(1.60, abs=0.02)


def test_three_segments_two_links():
    """A phosphodiester and a bridge: strand 1 is two chains (A: segments 1 and 2 and the PO2, B: segment 3 written
    5'->3'), the partners follow in segment order."""
    s, text, notes = build("D:GC 3'-5' L:GC 3'-3' D:GC")
    assert [(seg[0].chain, [(r.resseq, r.name) for r in seg]) for seg in s.segments] == [
        ("A", [(1, "G"), (2, "C"), (3, "0G"), (4, "0C"), (5, "PO2")]), ("B", [(1, "G"), (2, "C")]),
        ("C", [(1, "G"), (2, "C")]), ("D", [(1, "0G"), (2, "0C")]), ("E", [(1, "G"), (2, "C")])]
    assert [(r.first.chain, r.first.resseq, r.second.chain, r.second.resseq) for r in pt.link_records(s)] == [
        ("A", 4, "A", 5), ("A", 5, "B", 2)]
    assert [n.split(" (")[0] for n in notes if n.startswith("Junction")][:2] == ["Junction 1", "Junction 2"]
    r = MC.check_model(text, MC.QCSettings())
    assert r.failed == [] and r.clashes.clashes == [] and len(r.clashes.links) == 2 and r.clashes.links_unused == 0
    assert MC.hand_counts(r) == "A:2D+2L B:2D C:2D D:2L E:2D"


# --- names, hydrogens, mixed designs ---------------------------------------------------------------

@pytest.mark.parametrize("scheme, terminals, names", [
    (None, True, ["G", "C", "0G", "0C", "G", "C", "0G", "0C"]),
    ("amber", True, ["G5", "C", "LG5", "LC", "G5", "C3", "LG5", "LC3"]),
    ("amber", False, ["G", "C", "LG", "LC", "G", "C", "LG", "LC"]),
    ("ccd", True, ["G", "C", "0G5", "0C", "G", "C", "0G5", "0C3"]),
    ("rosetta", True, ["G", "C", "0G", "0C", "G", "C", "0G", "0C"]),
    ("pdb", True, ["G", "C", "0G", "0C", "G", "C", "0G", "0C"]),
    ("standard", True, ["G", "C", "G", "C", "G", "C", "G", "C"]),
])
def test_names_per_scheme_with_internal_names_at_the_junction(scheme, terminals, names):
    s, _text, notes = build("D:GC 3'-3' L:GC", scheme=scheme, terminals=terminals)
    assert [r.name for r in s.residues() if r.name != "PO2"] == names
    records = {r.name: {a.record for a in r.atoms} for r in s.residues()}
    assert records["PO2"] == {"HETATM"}
    if scheme == "pdb":  # deposition records: the CCD's L components are HETATM
        assert records["0G"] == records["0C"] == {"HETATM"} and records["G"] == {"ATOM"}
    else:
        assert all(rec == {"ATOM"} for name, rec in records.items() if name != "PO2")
    if scheme is None:
        assert ("Residue names A, C, G, U (D-RNA); 0A, 0C, 0G, 0U (the CCD L-ribonucleotides) (L-RNA), ATOM records; "
                "the bridging phosphates PO2 HETATM") in notes
    else:
        assert any(n.startswith(f"Residue names of the {scheme} scheme: ") for n in notes)
    assert any(n.startswith("WARNING: tleap has no unit") for n in notes) == (scheme == "amber")
    assert any(n.startswith("WARNING: the standard scheme gives 4 L residue(s) plain D names") for n in notes) == \
        (scheme == "standard")


def test_dna_names_phosphate5_and_the_ccd_l_component():
    s, _text, notes = build("D-DNA:GC 3'-5' L-DNA:GC", scheme="amber")
    assert [(r.chain, r.name) for r in s.residues()] == [("A", "DG5"), ("A", "DC"), ("A", "LDG"), ("A", "LDC3"),
                                                         ("B", "DG5"), ("B", "DC3"), ("C", "LDG5"), ("C", "LDC3")]
    assert any(n.startswith("L-DNA: 4 residue(s) (A:3-4, C:1-2) have MDATK's Amber-style names") for n in notes)
    assert any(n.startswith("B-DNA phosphates turned 6° about O3'-O5'") for n in notes)
    # OP3 only on the free 5' ends; the CCD names an L 5' end with its phosphate as the component itself
    s, _text, notes = build("D:GC 3'-5' L:GC", scheme="ccd", phosphate5=True)
    assert [r.name for r in s.residues()] == ["G", "C", "0G", "0C3", "G", "C", "0G", "0C3"]
    op3 = [(r.chain, r.resseq) for r in s.residues() if "OP3" in r.coords()]
    assert op3 == [("A", 1), ("B", 1), ("C", 1)]
    assert "P" in residue(s, "A", 3).coords() and "OP3" not in residue(s, "A", 3).coords()  # the junction's P
    assert any(n.startswith("5'-terminal L residues with a 5'-phosphate keep the CCD component names") and
               n.endswith("C:1") for n in notes)
    _s, _text, notes = build("D:GC 3'-5' L:GC", scheme="amber", phosphate5=True)
    assert any(n.startswith("WARNING: the Amber 5' units have no 5'-phosphate") for n in notes)


def test_hydrogens_are_carried_with_their_atoms():
    """The fit moves heavy atoms; hydrogens follow them (H5' / H5'' with the gamma turn of a DNA phosphodiester),
    HO5' / HO3' only on free hydroxyl ends."""
    heavy, _t, _n = build("D-DNA:GC 3'-5' L-DNA:GC")
    s, text, _n = build("D-DNA:GC 3'-5' L-DNA:GC", hydrogens=True)
    xyz = {(r.chain, r.resseq, a.name): a.xyz for r in s.residues() for a in r.atoms}
    assert all(xyz[(r.chain, r.resseq, a.name)] == a.xyz for r in heavy.residues() for a in r.atoms)
    hydroxyls = {(r.chain, r.resseq, n) for r in s.residues() for n in r.coords() if n in ("HO5'", "HO3'")}
    assert hydroxyls == {("A", 1, "HO5'"), ("A", 4, "HO3'"), ("B", 1, "HO5'"), ("B", 2, "HO3'"), ("C", 1, "HO5'"),
                         ("C", 2, "HO3'")}
    # the turned residue (A:3, the L-DNA residue that carries the junction's P) keeps the template's C5' hydrogens
    turned, plain = coords(residue(s, "A", 3)), coords(residue(s, "C", 2))
    for c in (turned, plain):
        assert dist(c["C5'"], c["H5'"]) == pytest.approx(1.09, abs=0.002)
    for a, b, h in (("O5'", "C5'", "H5'"), ("O5'", "C5'", "H5''"), ("C4'", "C5'", "H5'")):
        assert angle(turned[a], turned[b], turned[h]) == pytest.approx(angle(plain[a], plain[b], plain[h]), abs=0.2)
    r = MC.check_model(text, MC.QCSettings(hydrogen_check=True))
    assert r.failed == [] and r.clashes.clashes == [] and r.hydrogens.hydrogens == 94


def test_mixed_and_same_hand_designs():
    """An RNA-DNA strand of one hand closes like one helix: the DNA residue that carries the P turns; a single
    strand stacks its junction base alone."""
    s, text, notes = build("D-RNA:GGC 3'-5' D-DNA:GCC")
    assert [r.name for r in s.segments[0]] == ["G", "G", "C", "DG", "DC", "DC"]
    assert [r.name for r in s.segments[2]] == ["DG", "DG", "DC"]
    j = next(n for n in notes if n.startswith("Junction 1"))
    assert "O3'-P 1.60 Å" in j and "gamma of A:4 turned" in j
    check_passes(text, ["3'-5'"])
    assert ("Residue names A, C, G, U (D-RNA); DA, DC, DG, DT (D-DNA), ATOM records") in notes
    s, text, notes = build("L:GGCAUC/none 3'-5' D:GAUGCC")
    assert [seg[0].chain for seg in s.segments] == ["A", "B"] and len(s.segments[0]) == 12
    assert "REMARK 951 SEGMENT 1 A:1-6 L-RNA A-FORM, SINGLE STRAND, GGCAUC" in text
    a = junction_atoms(s, "3'-5'")
    assert 1.55 <= dist(a["O3'"], a["P"]) <= 1.65
    check_passes(text, ["3'-5'"])


# --- command line, progress, cancel ------------------------------------------------------------------

def test_cli_builds_logs_title_and_citations(tmp_path):
    from mdatk.cli import main
    from mdatk.config import Project
    from mdatk.projectfile import read_log

    proj = tmp_path / "p.mdatk.toml"
    Project().save(str(proj))
    out = tmp_path / "c.pdb"
    assert main(["build", "chimera", "--design", "D:GC 3'-3' L:GC", "--scheme", "amber", "--chains", "P,Q,R,S",
                 "-o", str(out), "-p", str(proj)]) == 0
    s = pt.read_pdb(str(out))
    assert [seg[0].chain for seg in s.segments] == ["P", "Q", "R", "S"] and len(pt.link_records(s)) == 2
    entry = read_log(str(proj))[-1]
    assert entry["step"] == "build chimera" and entry["title"] == "Chimera 5'-D-RNA(2)-3'p3'-L-RNA(2)-5'"
    assert entry["citations"] == ["arnott1972"] and "Junction 1 (3'-3'" in entry["summary"]
    assert main(["build", "chimera", "--design", "D:GC 3'-5' L:GC", "--na", "dna", "--hydrogens", "--scheme", "amber",
                 "--no-terminal-names", "-o", str(out), "-p", str(proj)]) == 0
    entry = read_log(str(proj))[-1]
    assert entry["title"] == "Chimera 5'-D-DNA(2)-p-L-DNA(2)-3'" and entry["citations"] == ["arnott1972dna"]
    s = pt.read_pdb(str(out))
    assert [r.name for r in s.residues()][:4] == ["DG", "DC", "LDG", "LDC"]
    assert "HO5'" in residue(s, "A", 1).coords() and "H5'" in residue(s, "A", 1).coords()
    assert main(["build", "chimera", "--design", "D:GC 3'-5' L:GC", "--phosphate5", "-o", str(out)]) == 0
    assert "OP3" in pt.read_pdb(str(out)).segments[0][0].coords()
    assert main(["build", "chimera", "--design", "D-RNA:GGC 3'-5' D-DNA:GCC", "-o", str(out), "-p",
                 str(proj)]) == 0
    assert read_log(str(proj))[-1]["citations"] == ["arnott1972", "arnott1972dna"]
    assert main(["build", "chimera", "--design", "D:GC 5'-5' L:GX", "-o", str(tmp_path / "bad.pdb"), "-p",
                 str(proj)]) == 1
    entry = read_log(str(proj))[-1]
    assert entry["status"] == "failed" and entry["title"] == "Chimera D:GC 5'-5' L:GX"


def test_progress_and_cancel():
    calls = []
    s, _text, _notes = build("D:G 3'-5' L:C", progress=lambda f, m: calls.append((f, m)))
    fractions = [f for f, _m in calls]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert fractions[0] == 1 / (2 * (36 + 8))  # 36 starts and 8 refits, of the design and of its mirror image
    assert calls[0][1] == "the design: junction 1 (3'-5'): start 1 of 36"
    assert any(m.startswith("its mirror image: junction 1 (3'-5'): contacts, fit") for _f, m in calls)
    # one nucleotide per segment: both of its ends in one residue (5' free, 3' bonded; then 5' bonded, 3' free)
    assert [(r.chain, r.name) for r in s.residues()] == [("A", "G"), ("A", "0C"), ("B", "C"), ("C", "0G")]
    asked = []

    def cancel():
        asked.append(1)
        return len(asked) > 5

    with pytest.raises(B.BuildError, match="^cancelled$"):
        C.build_chimera("D:GA 3'-3' L:UC", cancel=cancel)
    assert C._fit_key(C.parse_design("D:GA 3'-3' L:UC")[0], ["3'-3'"], False) not in C._FITS
