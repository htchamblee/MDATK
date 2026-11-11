"""D/L chimeras with linked partner strands (``link_partners`` of mdatk.chimera.build_chimera, 'mdatk build chimera
--link-partners', the planner step's and RNAdo's check box): off by default and then byte-identical to the builds
of before; on, one continuous partner chain whose junction phosphodiesters close with strand 1's (bonds, angles,
contacts, sound phosphates), internal three-character names at the junctions, exact mirror twins, the errors for
bridges and single strands, chain counts and ids, the secondary structure of the planner step, the command line.

Segments of two or three base pairs keep the fits short; the fits are cached by design (a design and its twin
share theirs), so the tests that read a model again reuse them."""

import hashlib
import re

import numpy as np
import pytest

from mdatk import builder as B
from mdatk import chimera as C
from mdatk import modelcheck as MC
from mdatk import pdbtools as pt

from test_chimera import assert_sound_phosphates, build, dist, hand_of, phosphodiesters

DNA = "D-DNA:GCA 3'-5' L-DNA:GCA"  # the user's kind of junction: D-DNA 3'-5' L-DNA, partners auto
RNA53 = "D-RNA:GC 5'-3' L-RNA:GC"  # read 3'->5': strand 1 written L then D, the partner chain D then L
THREE = "D:GC 3'-5' L:GC 3'-5' D:GC"  # RNA, two junctions


def o3_p(chain):
    """O3'(i)-P(i+1) of consecutive residues of a chain (Å)."""
    return [dist(a.coords()["O3'"], b.coords()["P"]) for a, b in zip(chain, chain[1:])]


def junctions(s, text):
    """(O3' residue, P residue) of every junction phosphodiester as chain:number, from the header."""
    return [(f"{o.chain}:{o.resseq}", f"{p.chain}:{p.resseq}") for o, p in phosphodiesters(s, text)]


def ter_count(text):
    return sum(line.startswith("TER") for line in text.splitlines())


# --- off: as before -------------------------------------------------------------------------------------

# sha256 of the PDB text and the notes that MDATK 0.0.17 wrote for these builds (before the option existed)
BEFORE = [
    ("D-DNA:GCA 3'-5' L-DNA:GCA", {"scheme": "amber"},
     "a876d631fcbb5e6dec702d7f0aa645defb10752b99934b442136d02188da0109"),
    ("D-RNA:GC 5'-3' L-RNA:GC", {"hydrogens": True},
     "4661b7462163d9665b67d5d3addad4604b6b193d38148c0dfb75521fad02447f"),
    ("D:GC 3'-3' L:GC", {"scheme": "amber", "chains": "P,Q,R,S"},
     "48c45ed8a1aa53bf2db44b4b23b0393c0810532b9394036b4a3beb10b2a4a6a2"),
]


@pytest.mark.parametrize("design, kw, digest", BEFORE)
def test_off_by_default_and_then_the_builds_of_before(design, kw, digest):
    notes = []
    text = C.build_chimera(design, notes=notes, **kw)
    assert hashlib.sha256((text + "\n".join(notes)).encode()).hexdigest() == digest
    off = []
    assert C.build_chimera(design, notes=off, link_partners=False, **kw) == text and off == notes
    assert not any("linked" in n or "partner strands joined" in n for n in notes)
    assert C.chain_count(design) == C.chain_count(design, link_partners=False)


# --- on: one partner chain ----------------------------------------------------------------------------

def test_the_partner_strands_become_one_chain_with_continuous_phosphodiesters():
    s, text, notes = build(DNA, scheme="amber", link_partners=True)
    assert [(seg[0].chain, len(seg)) for seg in s.segments] == [("A", 6), ("B", 6)]
    assert ter_count(text) == 2  # no TER inside the partner chain
    a, b = s.segments
    # strand 1 5'-D(GCA)-p-L(GCA)-3', strand 2 5'-L(TGC)-p-D(TGC)-3'
    assert [(r.resseq, r.name) for r in a] == [(1, "DG5"), (2, "DC"), (3, "DA"), (4, "LDG"), (5, "LDC"), (6, "LDA")]
    assert [(r.resseq, r.name) for r in b] == [(1, "LDT"), (2, "LDG"), (3, "LDC"), (4, "DT"), (5, "DG"), (6, "DC3")]
    assert [hand_of(r) for r in b] == ["L"] * 3 + ["D"] * 3
    assert all(len(r.name) <= 3 for r in s.residues())  # three-character names only
    for chain in (a, b):
        assert all(1.5 <= d <= 1.7 for d in o3_p(chain)), o3_p(chain)
    # the partner's junction residues: B:3 keeps its O3' (no HO3' in a model with hydrogens), B:4 gains P/OP1/OP2
    junction = b[3].coords()
    assert {"P", "OP1", "OP2"} <= set(junction) and "OP3" not in junction
    assert pt.link_records(s) == []  # bonds within a chain need no LINK records
    # both phosphodiesters of the junction: in the header (strand 1's, then the partners') and sound
    assert junctions(s, text) == [("A:3", "A:4"), ("B:3", "B:4")]
    assert_sound_phosphates(s, text, 2)
    assert re.search(r"^REMARK 951 JUNCTION 1 3'-5' B:3:O3'-B:4:P: O3'-P 1\.6\d A, .*, PARTNER STRANDS$", text, re.M)
    assert any(n.startswith("Partner strands linked: B:1-6 is one continuous chain 5'-L-DNA(3)-p-D-DNA(3)-3'")
               for n in notes)
    joined = next(n for n in notes if n.startswith("Junction 1"))
    assert "partner strands joined (B:3 O3' - B:4 P): O3'-P 1.6" in joined and "gamma of B:4 turned" in joined
    assert "Built a chimera 5'-D-DNA(3)-p-L-DNA(3)-3': 2 segments, 2 chains" in notes[-1]


def test_both_links_close_without_clashes_and_pass_the_model_check():
    for design, kw in ((DNA, {"scheme": "amber"}), (RNA53, {})):
        s, text, notes = build(design, link_partners=True, **kw)
        r = MC.check_model(text, MC.QCSettings())
        assert r.failed == [] and r.warnings == [] and r.clashes.clashes == [], design
        assert r.backbone.problems == [] and r.backbone.strained == [], design
        for o_res, p_res in phosphodiesters(s, text):
            o, p = o_res.coords(), p_res.coords()
            assert dist(o["O3'"], p["P"]) == pytest.approx(C.LINK_BOND, abs=0.03)
            assert dist(p["P"], p["O5'"]) == pytest.approx(1.6, abs=0.05)
            for label, (u, v, w), target in (("C3'-O3'-P", (o["C3'"], o["O3'"], p["P"]), C.C_O_P),
                                             ("O3'-P-O5'", (o["O3'"], p["P"], p["O5'"]), C.O_P_O),
                                             ("P-O5'-C5'", (p["P"], p["O5'"], p["C5'"]), 120.0)):
                assert abs(angle(u, v, w) - target) <= 6.0, (design, label, angle(u, v, w))
        junction = next(n for n in notes if n.startswith("Junction 1"))
        contact = float(re.search(r"closest heavy-atom contact between the segments (\d+\.\d+) Å", junction)[1])
        stack = float(re.search(r"end pairs (\d+\.\d+) Å apart", junction)[1])
        assert contact >= 3.0, junction
        # RNA stacks as well as unlinked; a D/L B-DNA junction cannot with both backbones closed (named in a warning)
        assert stack <= (5.0 if design == RNA53 else 8.0), junction
        warnings = [n for n in notes if n.startswith("WARNING: junction")]
        assert all(re.fullmatch(r"WARNING: junction 1 \(3'-5'\) did not close: end pairs \d\.\d\d Å apart \(target "
                                r"3\.6\); fit cost .*", w) for w in warnings), warnings


def angle(a, b, c):
    u, v = np.subtract(a, b), np.subtract(c, b)
    return float(np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))


def test_a_design_read_3_to_5_and_its_partner_chain():
    s, text, notes = build(RNA53, scheme="amber", link_partners=True)
    a, b = s.segments
    # strand 1 is written 5'->3': the L segment first; the partner chain the reverse: D first
    assert [r.name for r in a] == ["LG5", "LC", "G", "C3"] and [r.name for r in b] == ["G5", "C", "LG", "LC3"]
    assert all(1.5 <= d <= 1.7 for chain in (a, b) for d in o3_p(chain))
    assert junctions(s, text) == [("A:2", "A:3"), ("B:2", "B:3")]
    assert C.describe_partner(RNA53) == "5'-D-RNA(2)-p-L-RNA(2)-3'"
    assert C.linked_strands(RNA53) == ("GCGC", "GCGC", ["wc"] * 4)


def test_three_segments_one_partner_chain():
    s, text, notes = build(THREE, link_partners=True, chains="X,Y")
    assert [(seg[0].chain, len(seg)) for seg in s.segments] == [("X", 6), ("Y", 6)] and ter_count(text) == 2
    assert [hand_of(r) for r in s.segments[1]] == ["D", "D", "L", "L", "D", "D"]
    assert all(1.5 <= d <= 1.7 for seg in s.segments for d in o3_p(seg))
    assert len(phosphodiesters(s, text)) == 4
    assert_sound_phosphates(s, text, 4)
    header = [ln for ln in text.splitlines() if ln.startswith("REMARK 951 SEGMENT")]
    assert header == ["REMARK 951 SEGMENT 1 X:1-2 D-RNA A-FORM, PARTNER Y:5-6, GC",
                      "REMARK 951 SEGMENT 2 X:3-4 L-RNA A-FORM, PARTNER Y:3-4, GC",
                      "REMARK 951 SEGMENT 3 X:5-6 D-RNA A-FORM, PARTNER Y:1-2, GC"]
    assert C.describe_partner(THREE) == "5'-D-RNA(2)-p-L-RNA(2)-p-D-RNA(2)-3'"


def test_a_design_and_its_mirror_twin_are_exact_mirror_images():
    twin = "L-DNA:GCA 3'-5' D-DNA:GCA"
    s1, _t1, n1 = build(DNA, scheme="amber", hydrogens=True, link_partners=True)
    s2, _t2, n2 = build(twin, scheme="amber", hydrogens=True, link_partners=True)
    r1, r2 = list(s1.residues()), list(s2.residues())
    assert [(r.chain, r.resseq) for r in r1] == [(r.chain, r.resseq) for r in r2]
    for a, b in zip(r1, r2):
        assert hand_of(a) != hand_of(b) and [x.name for x in a.atoms] == [x.name for x in b.atoms]
        for x, y in zip(a.atoms, b.atoms):
            (x1, y1, z1), (x2, y2, z2) = x.xyz, y.xyz
            assert (x1, y1, z1) == (-x2, y2, z2), (a.label, x.name)
    # the hydrogens do not change the heavy atoms
    s3, _t3, _n3 = build(DNA, scheme="amber", link_partners=True)
    heavy = {(r.label, a.name): a.xyz for r in s1.residues() for a in r.atoms if not a.name.startswith("H")}
    assert heavy == {(r.label, a.name): a.xyz for r in s3.residues() for a in r.atoms}
    # the partners' 3' end at the junction has no HO3', its 5' end no HO5'; the free ends keep theirs
    b = s1.segments[1]
    assert "HO3'" not in b[2].coords() and "HO5'" not in b[3].coords()
    assert "HO5'" in b[0].coords() and "HO3'" in b[-1].coords()


def test_the_junction_ends_of_the_partner_strands():
    """A partner 5' end at a junction carries the P (no OP3, also with phosphate5), a partner 3' end at one has
    no HO3'; the free ends are as without the option (the segments as the fit and the model get them)."""
    segments, links, _notes = C.parse_design(THREE)
    read = C._read53(links)
    for k, (p5, p3) in enumerate([("p", "free"), ("p", "bonded"), ("free", "bonded")]):
        ends = C._ends(links, read, k)
        assert C._partner_ends(ends, True) == (p5, p3) and C._partner_ends(ends) == (C._partner_end5(ends[1]), "free")
        for phosphate5 in (False, True):
            partner = C._segment_residues(k, segments[k], ends, phosphate5, linked=True)[1]
            first, last = partner[0].names, partner[-1].names
            if p5 == "p":
                assert {"P", "OP1", "OP2"} <= set(first) and "OP3" not in first and "HO5'" not in first
            else:
                assert ("OP3" in first) == phosphate5 and ("HO5'" in first) != phosphate5
            assert ("HO3'" in last) == (p3 == "free")


# --- errors, chains, the design check -----------------------------------------------------------------

@pytest.mark.parametrize("design, message", [
    ("D:GC 3'-3' L:GC", r"link partners: link 1 \(3'-3'\) is a phosphate bridging two 3' ends"),
    ("D:GC 5'-3' L:GC 5'-5' D:GC", r"link partners: link 2 \(5'-5'\)"),
    ("D:GC/none 3'-5' L:GC", r"link partners: segment 1 \(D-RNA:GC\) is a single strand \(partner none\)"),
    ("D:GC 3'-5' L:GC/-", r"link partners: segment 2 \(L-RNA:GC\) is a single strand"),
])
def test_bridges_and_single_strands_cannot_be_linked(design, message):
    with pytest.raises(B.BuildError, match=message):
        C.build_chimera(design, link_partners=True)
    level, text = C.check_chimera(design, link_partners=True)[0]
    assert level == "error" and re.match(message, text)
    assert all(level != "error" for level, _t in C.check_chimera(design))  # fine without the option
    with pytest.raises(B.BuildError, match=message):
        C.describe_partner(design)


def test_chain_count_and_ids():
    assert C.chain_count(DNA) == 3 and C.chain_count(DNA, link_partners=True) == 2
    assert C.chain_count(THREE) == 4 and C.chain_count(THREE, link_partners=True) == 2
    msgs = C.check_chimera(DNA, link_partners=True)
    assert all(level == "info" for level, _t in msgs)
    assert ("info", "partner strands linked: one continuous partner chain 5'-L-DNA(3)-p-D-DNA(3)-3', its "
                    "phosphodiesters fitted together with strand 1's at every junction") in msgs
    assert C.check_chimera(DNA, chains="P,Q", link_partners=True)[0][0] == "info"
    level, text = C.check_chimera(DNA, chains="P,Q,R", link_partners=True)[0]
    assert level == "error" and "the design writes 2 chains" in text and "3 chain id(s) given" in text
    assert C.check_chimera(DNA, chains="P,Q")[0][0] == "error"  # unlinked: three chains
    with pytest.raises(B.BuildError, match="the design writes 2 chains"):
        C.build_chimera(DNA, chains="P,Q,R", link_partners=True)
    # the atom count of the check is the model's
    for kw in ({}, {"hydrogens": True}):
        atoms = next(t for level, t in C.check_chimera(DNA, link_partners=True, **kw) if " atoms (" in t)
        s = pt.read_pdb(C.build_chimera(DNA, link_partners=True, **kw))
        assert atoms.startswith(f"{sum(len(r.atoms) for r in s.residues())} atoms"), (kw, atoms)


# --- command line, planner step, window -----------------------------------------------------------------

def test_cli_link_partners(tmp_path):
    from mdatk.cli import main

    out = tmp_path / "c.pdb"
    assert main(["build", "chimera", "--design", DNA, "--scheme", "amber", "--link-partners", "-o", str(out)]) == 0
    s = pt.read_pdb(str(out))
    assert [(seg[0].chain, len(seg)) for seg in s.segments] == [("A", 6), ("B", 6)]
    assert main(["build", "chimera", "--design", "D:GC 3'-3' L:GC", "--link-partners", "-o",
                 str(tmp_path / "bad.pdb")]) == 1


def test_planner_step_option_command_issues_and_secondary_structure():
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import StepContext

    tool = TOOLS["structure-builder"]
    assert tool.section("chimera").option("chimera_link_partners").default is False
    ctx = StepContext("p", 1, "s", tool.tag)

    def settings(**kw):
        return tool.normalise({"mode": "chimera", "design": DNA, **kw})

    off, on = settings(), settings(chimera_link_partners=True)
    assert off["chimera_link_partners"] is False
    assert "--link-partners" not in tool.command(off, ctx)
    assert tool.command(on, ctx).split(" -o ")[0].endswith(" --link-partners")
    infos = [i.message for i in tool.check(on, ctx) if i.level == "info"]
    assert any(m.startswith("partner strands linked: one continuous partner chain 5'-L-DNA(3)-p-D-DNA(3)-3'")
               for m in infos)
    bad = tool.check(settings(design="D:GC 3'-3' L:GC", chimera_link_partners=True), ctx)
    assert [i.keys for i in bad if i.level == "error"] == [("chimera_link_partners",)]
    assert not [i for i in tool.check(settings(design="D:GC 3'-3' L:GC"), ctx) if i.level == "error"]
    chains = tool.check(settings(chimera_link_partners=True, chimera_chains="P,Q,R"), ctx)
    assert [i.keys for i in chains if i.level == "error"] == [("chimera_chains",)]
    assert not [i for i in tool.check(settings(chimera_link_partners=True, chimera_chains="P,Q"), ctx)
                if i.level == "error"]
    # the secondary structure: only with linked partners of one nucleic acid
    assert list(tool.output_values(off, ctx)) == ["pdb"] and [p.name for p in tool.output_ports(off)] == ["pdb"]
    values = tool.output_values(on, ctx)
    assert list(values) == ["pdb", "secstruct"] and values["pdb"].data["chirality"] == "mixed"
    assert [p.name for p in tool.output_ports(on)] == ["pdb", "secstruct"]
    text = next(f.content for f in tool.generate(on, ctx) if f.name.endswith(".secstruct"))
    assert text.split("\n")[:2] == ["(((((( ))))))", "gcagca tgctgc"]
    mixed = settings(design="D-DNA:GC 3'-5' L-RNA:GC", chimera_link_partners=True)
    assert list(tool.output_values(mixed, ctx)) == ["pdb"]  # DNA and RNA: no secondary structure
