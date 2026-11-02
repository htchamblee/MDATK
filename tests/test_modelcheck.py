"""Model checks (mdatk.modelcheck, 'mdatk pdb qc', the 'model-check' step) and the PDB-tool additions they
need (residue hands, infer_base, the standard scheme, hydrogen / virtual-residue removal), on synthetic
models from the structure builder.  Regression tests name the defects of the scripts this replaces."""

import ast
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys

import pytest

from mdatk import builder, modelcheck as mc, pdbtools as pt
from mdatk.config import PlanEdge, PlanNode, ResidueGroup, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan, single_step_plan
from mdatk.workflows.scripts import prep_pdb

STAMP = "20261015-1432"
SEQ = "GGCAUCGAUG"


def helix(seq=SEQ, chirality="D", **kw):
    return pt.read_pdb(builder.build_helix(seq, chirality=chirality, **kw))


def residue(s, chain, resseq, icode=""):
    return next(r for r in s.residues() if (r.chain, r.resseq, r.icode) == (chain, resseq, icode))


def atom(res, name):
    return next(a for a in res.atoms if a.name == name)


def place(a, xyz):
    a.tail = f"   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + a.tail[27:]


def shift(res, vec):
    for a in res.atoms:
        place(a, [a.xyz[k] + vec[k] for k in range(3)])


def toward(a, b, distance):
    """Point at ``distance`` from atom ``b`` on the line from b to a."""
    u = [(a.xyz[k] - b.xyz[k]) / math.dist(a.xyz, b.xyz) for k in range(3)]
    return [b.xyz[k] + distance * u[k] for k in range(3)]


def pairs(result):
    return {tuple(sorted((c.first, c.second))) for c in result.clashes}


def reread(s):
    return pt.read_pdb(pt.write_pdb(s))


# --- pdbtools additions ---------------------------------------------------------------------------------

@pytest.mark.parametrize("names, base", [
    (["N9", "C8", "N7", "C6", "N6", "N1"], "A"), (["N9", "C8", "N7", "O6", "N2"], "G"),
    (["N7", "C8", "O6"], "G"), (["N1", "C2", "N3", "C4", "N4", "O2"], "C"), (["N1", "C2", "O4"], "U"),
    (["N1", "C2", "O4", "C7"], "T"), (["N1", "C2", "O4", "C5M"], "T"), (["N9", "N6", "O6"], None),
    (["N1", "C2", "N4", "O4"], None), (["CA", "N", "C", "O"], None), (["P", "OP1"], None),
])
def test_infer_base(names, base):
    assert pt.infer_base(names) == base


def test_nucleotide_recognised_from_atoms_when_the_name_is_unknown():
    s = helix("GC", partner="none")
    for r in s.residues():
        r.name = "XYZ"
    assert [pt.nucleotide(r) for r in s.residues()] == [None, None]
    assert [pt.nucleotide(r, infer=True).base for r in s.residues()] == ["G", "C"]
    rows = pt.residue_hands(s)
    assert [(r.base, r.call, r.name_hand) for r in rows] == [("G", "D", None), ("C", "D", None)]


def test_residue_hands_epimers_flat_centres_and_expectations():
    s = helix("GGCA", partner="none", chirality="L", chains="C", start=45)
    # a C3' epimer: O3' of C:46 reflected through the C2'-C3'-C4' plane (xylo sugar)
    r = residue(s, "C", 46)
    c3, c2, c4, o3 = (atom(r, n).xyz for n in ("C3'", "C2'", "C4'", "O3'"))
    a = [c2[k] - c3[k] for k in range(3)]
    b = [c4[k] - c3[k] for k in range(3)]
    n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    nn = math.sqrt(sum(x * x for x in n))
    n = [x / nn for x in n]
    v = [o3[k] - c3[k] for k in range(3)]
    dot = sum(v[k] * n[k] for k in range(3))
    place(atom(r, "O3'"), [c3[k] + v[k] - 2 * dot * n[k] for k in range(3)])
    # a flat C4': C5' of C:47 moved into the O4'-C4'-C3' plane
    r = residue(s, "C", 47)
    c4, o4, c3 = (atom(r, n).xyz for n in ("C4'", "O4'", "C3'"))
    place(atom(r, "C5'"), [c4[k] + (c4[k] - o4[k]) + (c4[k] - c3[k]) for k in range(3)])
    rows = pt.residue_hands(s, expect_l="C:45-48")
    by = {h.residue.resseq: h for h in rows}
    assert by[45].call == "L" and by[45].ok and by[45].c4 == "L" and not by[45].note
    # chirality.py called both of these 'L' / 'all L' from C4' alone
    assert by[46].call is None and by[46].c4 == "L" and "epimer at C3'" in by[46].note and by[46].defect
    assert by[47].call is None and by[47].c4 is None and "flat centre at C4'" in by[47].note
    assert by[46].ok is False and by[47].ok is False
    notes, problems = pt.hand_summary(rows)
    assert any("C:46" in p and "epimer" in p for p in problems) and any("C:47" in p for p in problems)
    table = pt.hands_table(rows).splitlines()
    assert table[0].split("\t")[:6] == ["chain", "residue", "name", "name_hand", "call", "c4_hand"]
    assert table[2].split("\t")[10] == "NO"
    # expectations: chain-qualified ranges ('A,B' is two chains, not ',' a chain), '*' for every chain
    d = helix(chains="A,B")
    assert all(h.ok for h in pt.residue_hands(d, expect_d="A,B"))
    assert all(h.ok is False for h in pt.residue_hands(d, expect_l="*"))
    _notes, problems = pt.hand_summary(pt.residue_hands(d, expect_l="A"))
    assert problems == ["10 residue(s) expected L measured D: A:1-10"]


def test_residue_hands_first_model_and_insertion_codes():
    d, mirror = helix("GCG", partner="none"), helix("GCG", partner="none", chirality="L")

    def body(s):
        return [ln for ln in pt.write_pdb(s).splitlines() if ln.startswith(("ATOM", "TER"))]

    lines = ["MODEL        1"] + body(d) + ["ENDMDL", "MODEL        2"] + body(mirror) + ["ENDMDL", "END"]
    s = pt.read_pdb("\n".join(lines) + "\n")
    assert {h.call for h in pt.residue_hands(s)} == {"D"}  # chirality.py let model 2 overwrite model 1
    r = list(d.residues())
    r[1].resseq, r[1].icode, r[2].resseq = 1, "A", 2
    rows = pt.residue_hands(reread(d))
    assert [(h.residue.resseq, h.residue.icode) for h in rows] == [(1, ""), (1, "A"), (2, "")]


def test_check_cli_expectations_strict_and_table(tmp_path, capsys):
    from mdatk.cli import main
    pdb = tmp_path / "m.pdb"
    pdb.write_text(builder.build_helix(SEQ, chains="A,C", chirality="L"))
    assert main(["pdb", "check", str(pdb), "--expect-l", "A,C", "--strict"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["problems"] == [] and info["chains"]["C"]["chirality"] == {"L": 10}
    assert main(["pdb", "check", str(pdb), "--expect-d", "C:1-3", "--strict"]) == 1
    assert "expected D measured L: C:1-3" in json.loads(capsys.readouterr().out)["problems"][0]
    assert main(["pdb", "check", str(pdb), "--table"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 21 and out[1].split("\t")[:5] == ["A", "1", "0G", "L", "L"]


def test_clean_replaces_to_amber_and_q_prep():
    """A Rosetta decoy: D strand ATOM with hydrogens, L strand HETATM named 0A ..., a VRT residue, an ion."""
    d = pt.read_pdb(builder.build_helix("GGCA", partner="none", chains="A", hydrogens=True, phosphate5=True))
    l_ = pt.read_pdb(builder.build_helix("UGCC", partner="none", chains="B", chirality="L", hydrogens=True,
                                         scheme="rosetta", phosphate5=True))
    for r in l_.residues():
        for a in r.atoms:
            a.record = "HETATM"
    def tail(x):
        return f"   {x:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00"

    vrt = pt.Residue("VRT", "C", 1, atoms=[pt.Atom("HETATM", n, "", tail(x), "X")
                                           for n, x in (("ORIG", 0.0), ("X", 1.0), ("Y", 0.0))])
    mg = pt.Residue("MG", "D", 1, atoms=[pt.Atom("HETATM", "MG", "", tail(30.0), "MG")])
    d.segments += l_.segments + [[vrt], [mg]]
    decoy = pt.write_pdb(d)
    assert "VRT" in decoy and "H5'" in decoy and " OP3 " in decoy
    amber = pt.clean(decoy, scheme="amber", strip_5p=True, strip_h=True, drop_virtual=True)
    s = amber.structure
    assert [r.name for r in s.residues()] == ["G5", "G", "C", "A3", "LU5", "LG", "LC", "LC3", "MG"]
    assert not any(pt.is_hydrogen(a, r.name) for r in s.residues() for a in r.atoms)
    first = [r for r in s.residues() if r.name in ("G5", "LU5")]
    for r in first:
        names = {a.name for a in r.atoms}
        assert not names & set(pt.PHOSPHATE_5P) and "HO5'" not in names
    assert any("Rosetta virtual residue" in n and "C:1 VRT" in n for n in amber.notes)
    assert any(n.startswith("Removed") and "hydrogen" in n for n in amber.notes)
    std = pt.clean(decoy, scheme="standard", strip_h=True, drop_virtual=True)
    names = [r.name for r in std.structure.residues()]
    assert names == ["G", "G", "C", "A", "U", "G", "C", "C", "MG"]
    assert {ln[:6] for ln in std.text.splitlines() if ln[:6] in ("ATOM  ", "HETATM")} == {"ATOM  "}
    assert any(n.startswith("WARNING: the standard scheme gives 4 L residue(s) plain D names (B:1-4)")
               for n in std.notes)
    kept = pt.clean(decoy, scheme="standard")
    assert any("keeps Rosetta virtual residues" in n for n in kept.notes)
    # the hand stays measurable from the geometry of the copy
    assert [h.call for h in pt.residue_hands(std.structure)] == ["D"] * 4 + ["L"] * 4


def test_prep_pdb_drops_virtual_residues():
    lines = [
        "ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P",
        "ATOM      2  C1'   G A   1       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  C1'   C A   2       2.000   0.000   0.000  1.00  0.00           C",
        "HETATM    4 ORIG VRT A   3       0.000   0.000   0.000  1.00  0.00           X",
        "HETATM    5  X   VRT A   3       1.000   0.000   0.000  1.00  0.00           X", "END"]
    kept, stats = prep_pdb.prepare(lines)
    assert any("VRT" in ln for ln in kept) and "virtual" not in stats
    out, stats = prep_pdb.prepare(lines, drop_virtual=True)
    assert not any("VRT" in ln for ln in out) and stats["virtual"] == 2
    assert out == kept[:2] + ["TER", "END"] and stats["atoms"] == 2


# --- clashes ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chirality", ["D", "L"])
def test_ideal_helices_are_clean(chirality):
    s = helix("GGCAUCGAUGCCAG", chirality=chirality)
    r = mc.check_model(s, path="ideal")
    assert r.status == "pass" and r.clashes.clashes == [] and r.backbone.problems == []
    assert r.backbone.pp == [] and len(r.backbone.links) == 26
    assert {h.call for h in r.hands} == {chirality} and r.nucleotides == 28


def test_chains_are_kept_apart():
    """model_sanity keyed atoms by residue number only: chains numbered from 1 collapsed into one residue."""
    s = helix(chains="A,B")
    other = helix(chains="C,D")
    for res in other.residues():
        shift(res, (40.0, 0.0, 0.0))
    s.segments += other.segments
    target = atom(residue(s, "A", 1), "C5'")
    mover = atom(residue(s, "C", 1), "O2'")
    place(mover, [target.xyz[0] + 1.2, target.xyz[1], target.xyz[2]])
    r = mc.check_model(reread(s), path="four chains")
    assert r.residues == 40 and len(r.backbone.links) == 36
    assert ("A:1:C5'", "C:1:O2'") in pairs(r.clashes)
    assert mc.hand_counts(r) == "A:10D B:10D C:10D D:10D"


def test_neighbouring_and_intra_residue_atoms_are_checked():
    """model_sanity skipped residues i and i+1 entirely (number adjacency) and never looked inside one."""
    s = helix(SEQ, partner="none")
    a3, a4 = residue(s, "A", 3), residue(s, "A", 4)
    place(atom(a3, "O2'"), toward(atom(a3, "O2'"), atom(a4, "C8"), 1.9))  # A:4 is the A
    a6 = residue(s, "A", 6)  # C
    place(atom(a6, "O2"), toward(atom(a6, "O2"), atom(a6, "O5'"), 2.3))
    got = pairs(mc.check_clashes(s))
    assert ("A:3:O2'", "A:4:C8") in got and ("A:6:O2", "A:6:O5'") in got
    # ... but nothing up to three bonds apart: O3'(i)-P(i+1) and its neighbours are bonded
    assert not any(p[0].split(":")[1] != p[1].split(":")[1] and "P" in (p[0][-1], p[1][-1]) for p in got)


def test_hbond_window_two_acceptors_and_the_lower_limit():
    s = helix()
    g, c = residue(s, "A", 1), residue(s, "B", 10)  # a G-C pair
    o6, n4 = atom(g, "O6"), atom(c, "N4")
    step = math.dist(o6.xyz, n4.xyz) - 2.32
    u = [(o6.xyz[k] - n4.xyz[k]) / math.dist(o6.xyz, n4.xyz) for k in range(3)]
    shift(c, [step * x for x in u])  # O6...N4 squeezed to 2.32 A: still a hydrogen bond by default
    r = mc.check_clashes(s)
    assert r.clashes == [] and ("A:1:O6", "B:10:N4") in {(k.first, k.second) for k in r.hbond_contacts}
    strict = mc.check_clashes(s, mc.QCSettings(hbond_min=2.5))
    assert [(k.first, k.second, k.kind) for k in strict.clashes][0] == ("A:1:O6", "B:10:N4", "donor-acceptor")
    # two pure acceptors (a phosphate oxygen and O4') have no hydrogen to share at any distance
    a5, a6 = residue(s, "A", 5), residue(s, "A", 7)
    place(atom(a5, "OP1"), toward(atom(a5, "OP1"), atom(a6, "O4'"), 2.55))
    assert ("A:5:OP1", "A:7:O4'") in pairs(mc.check_clashes(s))


def test_shadow_contacts_are_atom_based():
    """model_sanity excused all carbon contacts of a residue pair with an H-bond (here a 1.8 A C5'...C2')."""
    s = helix()
    g, c = residue(s, "A", 1), residue(s, "B", 10)
    o6, n4 = atom(g, "O6"), atom(c, "N4")
    u = [(o6.xyz[k] - n4.xyz[k]) / math.dist(o6.xyz, n4.xyz) for k in range(3)]
    shift(c, [0.4 * x for x in u])
    place(atom(c, "C5'"), toward(atom(c, "C5'"), atom(g, "C2'"), 1.8))
    r = mc.check_clashes(s)
    assert ("A:1:C2'", "B:10:C5'") in pairs(r) and r.shadows == []
    assert ("A:1:O6", "B:10:N4") in {(k.first, k.second) for k in r.hbond_contacts}
    # a carbon beside an H-bonded atom, against its partner and no closer than the H-bond, is not counted
    s = helix()
    g, c = residue(s, "A", 1), residue(s, "B", 10)
    shift(c, [0.5 * x for x in u])
    r = mc.check_clashes(s, mc.QCSettings(clash_tol=0.1))
    assert ("A:1:C6", "B:10:N4") in {(k.first, k.second) for k in r.shadows}
    assert ("A:1:C6", "B:10:N4") not in pairs(r)


def test_terminal_hydroxyls_protonation_and_old_names():
    s = helix(SEQ, partner="none")  # 5' end without phosphate: O5' of A:1 is a 5'-OH
    a1, a9 = residue(s, "A", 1), residue(s, "A", 9)
    acceptor = atom(a9, "OP1")
    place(atom(a1, "O5'"), toward(atom(a1, "O5'"), acceptor, 2.6))
    assert ("A:1:O5'", "A:9:OP1") not in pairs(mc.check_clashes(s))
    internal = atom(residue(s, "A", 4), "O5'")  # an ester O5' next to its phosphate: an acceptor
    place(atom(residue(s, "A", 8), "OP2"), toward(atom(residue(s, "A", 8), "OP2"), internal, 2.6))
    assert ("A:4:O5'", "A:8:OP2") in pairs(mc.check_clashes(s))
    # A+ (N1-H) donates when declared protonated: a phosphate oxygen on its Watson-Crick edge
    s = helix(SEQ, partner="none")
    a4 = residue(s, "A", 4)  # A
    ring = [atom(a4, n).xyz for n in ("N1", "C2", "N3", "C4", "C5", "C6")]
    centre = [sum(p[k] for p in ring) / 6 for k in range(3)]
    n1 = atom(a4, "N1").xyz
    u = [(n1[k] - centre[k]) / math.dist(n1, centre) for k in range(3)]
    place(atom(residue(s, "A", 10), "OP1"), [n1[k] + 2.55 * u[k] for k in range(3)])
    assert pairs(mc.check_clashes(s)) == {("A:10:OP1", "A:4:N1")}
    assert pairs(mc.check_clashes(s, mc.QCSettings(protonated="A:4"))) == set()
    # old-style names: O2* is the 2'-OH (a pure acceptor in the same place clashes)
    for donor, expected in (("O2'", set()), ("O4'", {("A:2:O4'", "A:8:OP1")})):
        s = helix(SEQ, partner="none")
        a8 = residue(s, "A", 8)
        o, p = atom(a8, "OP1").xyz, atom(a8, "P").xyz
        out = [o[k] + 2.6 * (o[k] - p[k]) / math.dist(o, p) for k in range(3)]
        place(atom(residue(s, "A", 2), donor), out)
        assert pairs(mc.check_clashes(s)) == expected
        for r in s.residues():
            for a in r.atoms:
                a.name = a.name.replace("'", "*")
        assert pairs(mc.check_clashes(reread(s))) == expected  # (labels use the primed names)


def test_ions_waters_and_elements():
    """model_sanity took the element from the name: Na+ 'NA' became nitrogen, Cl- carbon, Mg-O a clash."""
    s = helix(SEQ, partner="none")

    def outward(k, name, d):  # d from a phosphate oxygen of A:k, away from its phosphorus
        o, p = atom(residue(s, "A", k), name).xyz, atom(residue(s, "A", k), "P").xyz
        return [o[i] + d * (o[i] - p[i]) / math.dist(o, p) for i in range(3)]

    ions = []
    for k, (name, at) in enumerate((("MG", outward(5, "OP1", 2.1)), ("NA", outward(7, "OP2", 2.4)),
                                    ("MG", outward(9, "OP1", 1.3)))):
        ions.append(pt.Residue(name, "M", k + 1, atoms=[pt.Atom("HETATM", name, "", "   " + "0" * 24)]))
        place(ions[-1].atoms[0], at)
    s.segments.append(ions)
    s = reread(s)  # no element column: MG and NA from the residue names, not magnesium-as-M / nitrogen
    assert [mc.residue_kind(r)[0] for r in list(s.residues())[-3:]] == ["ion"] * 3
    assert mc.check_clashes(s).atoms == 213  # skipped by default
    checked = mc.check_clashes(s, mc.QCSettings(ions="check"))
    assert [(k.first, k.second, k.kind) for k in checked.clashes] == [("A:9:OP1", "M:3:MG", "ion")]
    water = pt.Residue("HOH", "W", 1, atoms=[pt.Atom("HETATM", "O", "", "   " + "0" * 24, "O")])
    place(water.atoms[0], outward(3, "OP2", 2.0))
    s.segments.append([water])
    s = reread(s)
    assert not any("W:1" in p[0] + p[1] for p in pairs(mc.check_clashes(s)))
    assert ("A:3:OP2", "W:1:O") in pairs(mc.check_clashes(s, mc.QCSettings(waters=True)))


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def test_builder_places_h2_from_the_sugar_geometry():
    """The ideal helix's H2'(i)...H6/H8(i+1) of 1.57-1.70 Å is the Arnott fiber's own geometry (C2'(i)...
    C8/C6(i+1) about 3.5 Å), not a placement bug: H2' sits 1.09 Å from C2', within 5 degrees of the exact
    tetrahedral direction (equal angles to C1', C3' and O2' as the C3'-endo sugar has them), and putting it
    exactly there moves the contact by less than 0.03 Å."""
    s = helix("GGCAUCGAUGCCAG", hydrogens=True)
    for seg in s.segments:
        for a, b in zip(seg, seg[1:]):
            x, y = a.coords(), b.coords()
            h8 = y["H8"] if "H8" in y else y["H6"]
            c2, h2 = x["C2'"], x["H2'"]
            u = [_unit([x[n][k] - c2[k] for k in range(3)]) for n in ("C1'", "C3'", "O2'")]
            d1, d2 = [u[1][k] - u[0][k] for k in range(3)], [u[2][k] - u[0][k] for k in range(3)]
            normal = _unit([d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
                            d1[0] * d2[1] - d1[1] * d2[0]])
            if sum(normal[k] * sum(v[k] for v in u) for k in range(3)) > 0:
                normal = [-n for n in normal]
            bond = [h2[k] - c2[k] for k in range(3)]
            assert math.dist(h2, c2) == pytest.approx(1.09, abs=0.002)
            assert math.degrees(math.acos(sum(p * q for p, q in zip(_unit(bond), normal)))) < 5.0
            ideal = [c2[k] + 1.09 * normal[k] for k in range(3)]
            assert 1.55 < math.dist(h2, h8) < 1.72 and abs(math.dist(ideal, h8) - math.dist(h2, h8)) < 0.03


def test_hydrogens_stay_out_of_the_clash_check_and_are_reported_apart():
    s = helix(hydrogens=True)
    heavy = mc.check_clashes(s)
    assert heavy.atoms == 420 and heavy.clashes == []  # the clash check never sees a hydrogen
    h = mc.check_hydrogen_contacts(s)
    assert h.hydrogens == 220 and h.atoms == 640 and h.radii == "nuclear"
    assert h.median_ch == pytest.approx(1.09)
    got = {(c.first.split(":")[-1], c.second.split(":")[-1]) for c in h.clashes}
    # MolProbity's conventions flag the fiber helix's H2'(i)...H6/H8(i+1) (1.17 + 1.00 Å contact)
    assert ("H2'", "H8") in got or ("H8", "H2'") in got
    assert all(c.overlap >= 0.4 for c in h.clashes) and h.clashscore == pytest.approx(
        1000 * len(h.clashes) / 640, abs=0.01)
    # four bonds from a hydrogen are left out (H6...O4' of a base in anti); a Watson-Crick imino or amino
    # hydrogen on its acceptor is a hydrogen bond (0.6 Å overlap allowed), not a clash
    assert not any("O4'" in c.first + c.second and "H6" in c.first + c.second for c in h.clashes)
    assert not any(c.first.split(":")[0] != c.second.split(":")[0] for c in h.clashes)
    # never a gate: the model passes, and the report has its own section and columns
    st = mc.QCSettings(hydrogen_check=True, max_clashes=0)
    r = mc.check_model(s, st, path="h")
    assert r.status == "pass" and r.hydrogens.clashes and not r.failed
    text = mc.text_report(r, st)
    assert "hydrogen contacts (reported only, never a gate" in text and "1.17 on carbon" in text
    head = mc.report_header((), True)
    row = dict(zip(head, mc.report_row(r, (), True)))
    assert row["h_clashes"] == str(len(h.clashes)) and row["h_radii"] == "nuclear" and row["status"] == "pass"
    assert "h_clashes" not in mc.tsv_text([mc.check_model(s, path="x")]).splitlines()[0]
    # electron-cloud positions (C-H 0.97 Å) take Probe's larger radii; an explicit choice overrides
    for res in s.residues():
        for a in res.atoms:
            if a.name.startswith("H"):
                parent = min((b for b in res.atoms if not b.name.startswith("H")),
                             key=lambda b: math.dist(a.xyz, b.xyz))
                if parent.name.startswith("C"):
                    place(a, toward(a, parent, 0.97))
    e = mc.check_hydrogen_contacts(reread(s))
    assert e.radii == "ecloud" and e.median_ch == pytest.approx(0.97, abs=0.002)
    assert mc.check_hydrogen_contacts(reread(s), mc.QCSettings(hydrogen_radii="nuclear")).radii == "nuclear"
    # explicit hydrogens only: none are added to a heavy-atom model
    bare = mc.check_hydrogen_contacts(helix())
    assert bare.hydrogens == 0 and bare.clashes == [] and bare.clashscore is None
    assert "has no hydrogens" in mc.text_report(mc.check_model(helix(), mc.QCSettings(hydrogen_check=True),
                                                               path="b"), mc.QCSettings(hydrogen_check=True))


def test_hydrogen_check_cli(tmp_path, capsys):
    from mdatk.cli import main
    pdb = tmp_path / "h.pdb"
    pdb.write_text(builder.build_helix(SEQ, hydrogens=True))
    assert main(["pdb", "qc", str(pdb), "--hydrogen-check", "--tsv", "--max-clashes", "0"]) == 0
    rows = [ln.split("\t") for ln in capsys.readouterr().out.splitlines()]
    row = dict(zip(rows[0], rows[1]))
    assert row["status"] == "pass" and int(row["h_clashes"]) > 0 and float(row["h_clashscore"]) > 0
    with pytest.raises(SystemExit):  # the hydrogens of the clash check are gone, not renamed
        main(["pdb", "qc", str(pdb), "--hydrogens"])


# --- backbone -----------------------------------------------------------------------------------------------

def test_links_follow_ter_segments_not_numbers():
    # chains numbered on from each other (A:1-10, B:11-20): no 'link' A:10 -> B:11 (model_sanity: 18 A)
    s = helix(chains="A,B", start=(1, 11))
    b = mc.check_backbone(s)
    assert len(b.links) == 18 and b.problems == [] and not any(k.first == "A:10" for k in b.links)
    # a break inside one strand that was renumbered across it (bbcheck: b == a + 1 missed it)
    s = helix(SEQ, partner="none")
    o3, p = atom(residue(s, "A", 5), "O3'"), atom(residue(s, "A", 6), "P")
    u = [(p.xyz[k] - o3.xyz[k]) / math.dist(o3.xyz, p.xyz) for k in range(3)]
    for res in list(s.residues())[5:]:
        shift(res, u)
        res.resseq += 3
    b = mc.check_backbone(reread(s))
    bad = b.strained  # 2.59 Å: bonded by tleap under strain, a warning (not a failure below 3 Å)
    assert [(k.first, k.second, k.status) for k in bad] == [("A:5", "A:9", "strained")]
    assert bad[0].distance > 2.5
    assert b.worst == bad[0] and b.problems == []
    # two TER segments of one chain with consecutive numbers are two strands, not a break
    one, two = helix("GGCAUC", partner="none"), helix("GAUGCC", partner="none", start=7)
    for res in two.residues():
        shift(res, (0.0, 30.0, 0.0))
    one.segments += two.segments
    b = mc.check_backbone(reread(one))
    assert len(b.links) == 10 and b.problems == []
    # insertion codes are residues of their own (bbcheck merged A:1 and A:1A)
    s = helix("GCAU", partner="none")
    r = list(s.residues())
    r[1].resseq, r[1].icode, r[2].resseq, r[3].resseq = 1, "A", 2, 3
    b = mc.check_backbone(reread(s))
    assert [(k.first, k.second) for k in b.links] == [("A:1", "A:1A"), ("A:1A", "A:2"), ("A:2", "A:3")]
    # the longest link is reported even when no link is broken (bbcheck printed 0.00)
    assert b.worst is not None and 1.55 < b.worst.distance < 1.65


def test_short_missing_links_and_pp_spacing():
    s = helix(SEQ, partner="none")
    a3, a4 = residue(s, "A", 3), residue(s, "A", 4)
    place(atom(a4, "P"), toward(atom(a4, "P"), atom(a3, "O3'"), 1.2))
    a7 = residue(s, "A", 7)
    a7.atoms = [a for a in a7.atoms if a.name != "O3'"]
    b = mc.check_backbone(s)
    assert [(k.first, k.status) for k in b.problems] == [("A:3", "short"), ("A:7", "missing O3'")]
    assert b.pp_measured == 8 and b.pp == []
    tight = mc.check_backbone(s, pp_min=6.0)
    assert len(tight.pp) == 8 and tight.pp[0].status == "p-p"


def _stretched(target):
    """A single strand whose O3'(A:5)-P(A:6) is ``target`` Å (residues 6 on moved along the bond)."""
    s = helix(SEQ, partner="none")
    o3, p = atom(residue(s, "A", 5), "O3'"), atom(residue(s, "A", 6), "P")
    d = math.dist(o3.xyz, p.xyz)
    u = [(p.xyz[k] - o3.xyz[k]) / d for k in range(3)]
    for res in list(s.residues())[5:]:
        shift(res, [(target - d) * x for x in u])
    return reread(s)


def test_strained_links_warn_with_their_strain_and_breaks_fail():
    """tleap bonds O3'-P by name: 2.0-3.0 Å is a strained bond (a warning with E = k (r - r0)^2, Amber's k,
    no 1/2), beyond 3.0 Å a break that fails the gate."""
    bond = mc.DEFAULT_LINK_BOND
    assert (bond.k, bond.r0) == (230.0, 1.61)
    assert [round(bond.strain(r)) for r in (2.0, 2.5, 3.0)] == [35, 182, 444]  # the user's estimate
    st = mc.QCSettings()
    r = mc.check_model(_stretched(2.5), st, path="strained")
    link = r.backbone.strained[0]
    assert (link.first, link.second, link.status) == ("A:5", "A:6", "strained")
    assert link.strain == pytest.approx(230 * (link.distance - 1.61) ** 2, abs=0.3)
    assert 182 <= link.strain < 183
    assert r.status == "pass" and r.backbone.problems == [] and r.backbone.strain == link.strain
    assert r.warnings == ["1 O3'-P link(s) of 2-3 A, strain 182 kcal/mol (A:5-A:6 2.50 A 182)"]
    text = mc.text_report(r, st)
    assert "1 strained (2-3 A)" in text and "A:5 -> A:6: 2.50 A (strained; strain 182 kcal/mol)" in text
    assert "k 230 kcal/mol/A^2, r0 1.61 A (MDATK's default, OS-P of Amber's parm99 / parm10.dat)" in text
    row = dict(zip(mc.report_header(), mc.report_row(r)))
    assert (row["strained_links"], row["link_strain"]) == ("1", f"{link.strain:.1f}")
    assert row["link_problems"] == "0"
    assert row["warnings"].startswith("1 O3'-P link(s)")
    r = mc.check_model(_stretched(3.4), st, path="broken")
    assert r.status == "fail" and r.backbone.problems[0].status == "long"
    assert r.backbone.problems[0].strain == pytest.approx(230 * (3.4 - 1.61) ** 2, abs=0.5)
    assert "1 O3'-P link(s) longer than 3 A, shorter than 1.4 A or unmeasurable" in r.failed[0]
    # the same thresholds move with the settings; another bond changes the strain
    r = mc.check_model(_stretched(2.5), mc.QCSettings(link_fail=2.4,
                                                      link_bond=mc.LinkBond(300.0, 1.6, "test")))
    assert r.status == "fail" and r.backbone.problems[0].strain == pytest.approx(243.0, abs=0.5)
    with pytest.raises(ValueError, match="failure length"):
        mc.validate_settings(mc.QCSettings(link_max=2.5, link_fail=2.0))


PARM10 = """PARM99 for DNA,RNA,AA, organic molecules, TIP3P wat. Polariz.& LP incl.02/04/99
C  12.01         0.616               sp2 C carbonyl group
P  30.97         1.538               phosphate
OS 16.00         0.465               ether and ester oxygen

C   H   HO  N   NA  NB  NC  N2  N3  NT  N*  O   OH  OS  P   O2
OH-P   230.0    1.610       JCC,7,(1986),230; NA PHOSPHATES
OS-P   230.0    1.610       JCC,7,(1986),230; NA PHOSPHATES
CT-OS  320.0    1.410       JCC,7,(1986),230; MOD SUGARS

P -OS-P    100.0      120.50
OS-P -OS    45.0      102.60

X -OS-P -X     3    2.40          0.0             3.         JCC,7,(1986),230

"""


def test_link_bond_read_from_the_force_field_tleap_loads(tmp_path, monkeypatch):
    home = tmp_path / "amber"
    (home / "dat" / "leap" / "cmd").mkdir(parents=True)
    (home / "dat" / "leap" / "parm").mkdir(parents=True)
    (home / "dat" / "leap" / "cmd" / "leaprc.RNA.OL3").write_text(
        "logFile leap.log\nparm10 = loadamberparams parm10.dat\nloadOff RNA.lib\n")
    (home / "dat" / "leap" / "parm" / "parm10.dat").write_text(PARM10)
    work = tmp_path / "step"
    work.mkdir()
    monkeypatch.chdir(work)  # tleap's working folder: the step's own files
    (work / "tleap.leap").write_text("# dry build\nsource leaprc.RNA.OL3\nloadamberparams frcmod.cp\n"
                                     "loadAmberParams frcmod.mine\nmol = loadpdb \"x.pdb\"\n")
    assert mc.bond_terms(PARM10) == {("OH", "P"): (230.0, 1.61), ("OS", "P"): (230.0, 1.61),
                                     ("CT", "OS"): (320.0, 1.41)}
    bond, notes = mc.link_bond_from_files(["tleap.leap"], ("OS",), str(home))
    assert (bond.k, bond.r0) == (230.0, 1.61) and bond.source.endswith("parm10.dat")
    assert bond.source.startswith("OS-P of ")
    assert any("frcmod.cp (named in tleap.leap): not found" in n for n in notes)
    # a frcmod loaded later replaces the term, as in tleap
    (work / "frcmod.mine").write_text("mine\nBOND\nP -OS  300.0   1.600\n\nANGLE\nP -OS-CT  100.0  120.5\n")
    bond, _ = mc.link_bond_from_files(["tleap.leap"], ("OS",), str(home))
    assert (bond.k, bond.r0, bond.source) == (300.0, 1.6, "OS-P of frcmod.mine")
    # the CP force field types O3' OR: its term wins when a file defines it, else OS-P
    (work / "frcmod.cp").write_text("cp\nMASS\nOR 16.00 0.465\n\nBOND\nOR-P   250.0    1.610\n")
    assert mc.link_bond_from_files(["tleap.leap"], ("OR", "OS"), str(home))[0].k == 250.0
    (work / "frcmod.cp").unlink()
    assert mc.link_bond_from_files(["tleap.leap"], ("OR", "OS"), str(home))[0].k == 300.0
    # nothing to read: MDATK's default, and why
    bond, notes = mc.link_bond_from_files(["missing.leap"], ("OS",), str(home))
    assert bond == mc.DEFAULT_LINK_BOND and any("cannot read missing.leap" in n for n in notes)
    assert any("no OS-P bond term" in n for n in notes)
    # the command line: the force field read, the source printed with the strain
    pdb = work / "m.pdb"
    pdb.write_text(pt.write_pdb(_stretched(2.5)))
    from mdatk.cli import main
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["pdb", "qc", str(pdb), "--checks", "links", "--link-params", "tleap.leap", "--amberhome",
                     str(home)]) == 0
    text = out.getvalue()
    assert "k 300 kcal/mol/A^2, r0 1.6 A: OS-P of frcmod.mine" in text and "strain 243 kcal/mol" in text
    assert "1 of 1 models passed the gates" in text


def test_link_bond_of_the_installed_amber_force_fields():
    """Every RNA leaprc MDATK offers loads parm10.dat, and none of their frcmods changes OS-P."""
    home = os.environ.get("MDATK_TEST_AMBERHOME", "")
    if not home or not os.path.isfile(os.path.join(home, "dat", "leap", "parm", "parm10.dat")):
        pytest.skip("AmberTools data not available (set MDATK_TEST_AMBERHOME)")
    from mdatk.workflows.amber import FORCE_FIELDS
    for _label, leaprc, _short in FORCE_FIELDS.values():
        path = os.path.join(home, "dat", "leap", "cmd", leaprc)
        bond, _notes = mc.link_bond_from_files([path], ("OS",), home)
        assert (bond.k, bond.r0) == (230.0, 1.61) and bond.source.endswith("parm10.dat"), leaprc


def test_kinks_flagged_allowed_and_gated(tmp_path, capsys):
    s = helix(SEQ, partner="none")
    plain = mc.check_kinks(s)
    assert plain.measured == 8 and plain.flagged == [] and 150 < plain.sharpest.angle < 154  # A-form 152
    # fold the chain back at A:5: C1'(6) next to C1'(4), about 20 degrees
    c4, c5 = atom(residue(s, "A", 4), "C1'").xyz, atom(residue(s, "A", 5), "C1'").xyz
    u = _unit([c4[k] - c5[k] for k in range(3)])
    place(atom(residue(s, "A", 6), "C1'"), [c5[k] + 5.5 * u[k] + (2.0 if k == 2 else 0.0) for k in range(3)])
    s = reread(s)
    k = mc.check_kinks(s)
    assert "A:5" in {x.residue for x in k.flagged} and k.sharpest.residue == "A:5" and k.sharpest.angle < 30
    allowed = mc.check_kinks(s, allow="A:4-7")
    assert allowed.flagged == [] and {x.residue for x in allowed.allowed} == {x.residue for x in k.flagged}
    # reported, a gate only when asked for
    r = mc.check_model(s, mc.QCSettings(checks=("kinks",)), path="k")
    assert r.status == "pass" and r.warnings[0].startswith(f"{len(k.flagged)} sharp kink(s) below 60 deg")
    r = mc.check_model(s, mc.QCSettings(checks=("kinks",), gate_kinks=True), path="k")
    assert r.status == "fail" and "sharp kink(s)" in r.failed[0]
    # the flexible positions of in-line probing, from a two-column reactivity file
    pdb, react = tmp_path / "k.pdb", tmp_path / "inline.txt"
    pdb.write_text(pt.write_pdb(s))
    react.write_text("# position reactivity\nposition\treactivity\n1\t0.05\n4\t0.7\n5\t0.91\n6,0.66\n"
                     "7 0.62\n8 nan\n9 -\n")
    assert mc.reactive_positions(str(react), 0.6) == [4, 5, 6, 7]
    from mdatk.cli import main
    common = ["pdb", "qc", str(pdb), "--checks", "kinks", "--gate-kinks", "--tsv"]
    assert main(common) == 0
    row = dict(zip(*[ln.split("\t") for ln in capsys.readouterr().out.splitlines()]))
    assert row["status"] == "fail" and row["sharpest_kink_at"] == "A:5" and int(row["kinks"]) >= 1
    assert main(common + ["--kink-reactivity", str(react), "--kink-reactivity-min", "0.6"]) == 0
    row = dict(zip(*[ln.split("\t") for ln in capsys.readouterr().out.splitlines()]))
    assert row["status"] == "pass" and row["kinks"] == "0"
    assert main(common + ["--kink-reactivity", str(react), "--kink-reactivity-min", "0.6",
                          "--kink-reactivity-chain", "B"]) == 0  # another chain's positions: still flagged
    assert dict(zip(*[ln.split("\t") for ln in capsys.readouterr().out.splitlines()]))["status"] == "fail"
    assert main(common + ["--kink-reactivity", str(react)]) == 1  # the threshold is needed
    assert "--kink-reactivity-min" in capsys.readouterr().err


# --- compaction ---------------------------------------------------------------------------------------------

def test_compaction_from_ranges_and_dotbrackets():
    s = helix("GGGAAACCC", partner="none", chains="B", start=20)
    segs = mc.range_segments(s, "B:22-26, B")
    assert [(n, len(r)) for n, r in segs] == [("B:22-26", 5), ("B", 9)]
    loops, notes = mc.loop_segments(s, [("p", "(((...)))"), ("i", "((.....))"), ("x", "((..))")])
    assert [(n, len(r)) for n, r in loops] == [("B:23-25", 3)] and "x: 6 positions" in notes[0]
    loops, _ = mc.loop_segments(s, [("vienna", ">seq\nGGGAAACCC\n((.....)) (-1.20)")])
    assert loops[0][0] == "B:22-26"
    result = mc.compaction(s, segs)
    assert 0.0 < result[0].ratio < 1.0 and result[1].residues == 9
    st = mc.QCSettings(segments="B:22-26", dotbrackets=[("p", "(((...)))")], compaction_flag=0.99)
    r = mc.check_model(s, st, path="hairpin")
    assert [c.name for c in r.compaction] == ["B:22-26", "B:23-25"]
    assert "extremely coiled" in mc.text_report(r, st) and r.status == "pass"  # a pointer, not a gate
    with pytest.raises(ValueError):
        mc.dotbracket_line("((#.))")


# --- reports, gates, decoy table ----------------------------------------------------------------------------

def test_decoy_table_reader(tmp_path):
    root = tmp_path / "plan"
    (root / "02_models").mkdir(parents=True)
    for k in (1, 2, 3):
        (root / "02_models" / f"m{k}.pdb").write_text("x")
    table = root / "02_models" / "decoys.tsv"
    table.write_text("# FARFAR2 decoys\nrun\ttag\tpdb\tscore\tlinear_chainbreak\textra\n"
                     "1\tS_1\t02_models/m1.pdb\t-30.5\t0.2\tq\n2\tS_2\t02_models/m2.pdb\t-20\t3.1\t\n"
                     "2\tS_3\t\t-10\t0\t\n1\tS_4\tsomewhere/else/m3.pdb\t-5\tnan-ish\t\n")
    t = mc.read_decoy_table(str(table), str(root))
    assert t.columns[:4] == ["run", "tag", "pdb", "score"] and len(t.rows) == 4
    assert t.row_for(str(root / "02_models" / "m1.pdb"))["tag"] == "S_1"  # by path from the plan root
    assert t.row_for(str(root / "02_models" / "m3.pdb"))["tag"] == "S_4"  # by a unique file name
    assert t.row_for(str(tmp_path / "S_3.pdb"))["tag"] == "S_3"  # by tag
    assert t.row_for(str(tmp_path / "other.pdb")) is None
    assert mc.DecoyTable.value(t.rows[1], "linear_chainbreak") == 3.1
    assert mc.DecoyTable.value(t.rows[3], "linear_chainbreak") is None
    csv = tmp_path / "d.csv"
    csv.write_text("tag,score\nS_9,-1\n")
    assert mc.read_decoy_table(str(csv)).rows == [{"tag": "S_9", "score": "-1"}]
    assert mc.parse_score_limits("linear_chainbreak<=1, score < -50") == [("linear_chainbreak", "<=", 1.0),
                                                                          ("score", "<", -50.0)]
    with pytest.raises(ValueError):
        mc.parse_score_limits("score = 1")


def _models(tmp_path):
    """Three models: clean D, one with an L-flipped duplex, one with a chain break."""
    paths = []
    good = helix()
    paths.append(tmp_path / "good.pdb")
    paths[-1].write_text(pt.write_pdb(good))
    paths.append(tmp_path / "flipped.pdb")
    paths[-1].write_text(builder.build_helix(SEQ, chirality="L"))
    broken = helix()
    for res in list(broken.residues())[5:10]:
        shift(res, (0.0, 0.0, 2.0))
    paths.append(tmp_path / "broken.pdb")
    paths[-1].write_text(pt.write_pdb(broken))
    return [str(p) for p in paths]


def test_qc_cli_report_filter_stop_and_outputs(tmp_path, capsys):
    from mdatk.cli import main
    models = _models(tmp_path)
    table = tmp_path / "decoys.tsv"
    table.write_text("run\ttag\tpdb\tscore\tlinear_chainbreak\n1\tgood\tgood.pdb\t-10\t0\n"
                     "1\tflipped\tflipped.pdb\t-30\t0\n1\tbroken\tbroken.pdb\t-20\t4\n")
    common = ["pdb", "qc", *models, "--expect-d", "A,B", "--scores", str(table), "--plan-root", str(tmp_path),
              "--score-terms", "score,linear_chainbreak"]
    assert main(common + ["--tsv"]) == 0
    rows = [ln.split("\t") for ln in capsys.readouterr().out.splitlines()]
    head = rows[0]
    assert head[:3] == ["model", "status", "failed"] and head[-2:] == ["score", "linear_chainbreak"]
    status = {os.path.basename(r[0]): r[1] for r in rows[1:]}
    assert status == {"good.pdb": "pass", "flipped.pdb": "fail", "broken.pdb": "fail"}
    assert rows[3][head.index("link_problems")] == "1" and rows[2][-2] == "-30"
    assert main(common + ["--json", "--score-limits", "linear_chainbreak<=1"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "linear_chainbreak 4 (limit <=1)" in data[2]["failed"]
    assert data[1]["hands"]["problems"][0].startswith("20 residue(s) expected D measured L")
    out = str(tmp_path / "passed-{n:02d}.pdb")
    assert main(common + ["--on-fail", "filter", "--pass", "1", "--passed-out", out, "--best-out",
                          str(tmp_path / "best.pdb")]) == 0
    assert (tmp_path / "best.pdb").read_text() == open(models[0]).read()
    assert (tmp_path / "passed-01.pdb").exists()
    capsys.readouterr()
    # up to N: one of three passes, so slot 1 is that model and slots 2-3 are empty slots, not a failure
    # (the filter failed whenever fewer than N passed: 38 usable decoys of 1,200 was a crash, not a finding)
    out = str(tmp_path / "slot-{n:02d}.pdb")
    assert main(common + ["--on-fail", "filter", "--pass", "3", "--passed-out", out]) == 0
    assert "1 of 3 models passed the gates; passed on 1 in 3 slot(s), slots 2-3 written empty" in \
        capsys.readouterr().out
    assert (tmp_path / "slot-01.pdb").read_text() == open(models[0]).read()
    for k in (2, 3):
        text = (tmp_path / f"slot-{k:02d}.pdb").read_text()
        assert text == f"REMARK 951 empty slot {k} of 3: only 1 models passed\nEND\n"
        assert pt.is_empty_slot(str(tmp_path / f"slot-{k:02d}.pdb"))
    assert not pt.is_empty_slot(models[0]) and not pt.is_empty_slot(str(tmp_path / "nowhere.pdb"))
    # the minimum is its own setting: fail only below it (default 1: when none passes)
    assert main(common + ["--on-fail", "filter", "--pass", "3", "--min-pass", "2"]) == 1
    assert "1 of 3 models pass the gates; at least 2 needed" in capsys.readouterr().err
    assert main(common[:3] + ["--expect-l", "*", "--on-fail", "filter", "--best-out",
                              str(tmp_path / "none.pdb")]) == 1  # none passes
    assert not (tmp_path / "none.pdb").exists()
    assert main(common[:3] + ["--expect-l", "*", "--on-fail", "filter", "--min-pass", "0", "--pass", "2",
                              "--passed-out", out, "--best-out", str(tmp_path / "none.pdb")]) == 0
    assert pt.empty_slot(str(tmp_path / "none.pdb")) == "empty slot 1 of 1: only 0 models passed"
    assert pt.empty_slot(str(tmp_path / "slot-01.pdb")) == "empty slot 1 of 2: only 0 models passed"
    capsys.readouterr()
    # an empty slot given to the check is skipped with a note, not an unreadable model
    assert main(["pdb", "qc", models[0], str(tmp_path / "slot-03.pdb"), "--tsv"]) == 0
    io_ = capsys.readouterr()
    assert len(io_.out.splitlines()) == 2 and "skipped" in io_.err and "empty slot 3 of 3" in io_.err
    assert "1 of 1 models passed the gates (1 empty slot(s) among the inputs skipped)" in io_.err
    assert main(["pdb", "qc", str(tmp_path / "slot-02.pdb")]) == 1
    assert "no model to check (every input is an empty slot)" in capsys.readouterr().err
    assert main(common + ["--on-fail", "stop"]) == 1
    # report mode passes everything on, in the order asked for (lowest score first)
    assert main(common + ["--order", "score", "--passed-out", str(tmp_path / "r{n}.pdb")]) == 0
    assert open(tmp_path / "r1.pdb").read() == open(models[1]).read()
    assert open(tmp_path / "r3.pdb").read() == open(models[0]).read()
    assert main(["pdb", "qc", *models, "--order", "clashes", "--checks", "links", "--best-out",
                 str(tmp_path / "b.pdb")]) == 0
    capsys.readouterr()
    (tmp_path / "trunc.pdb").write_text("ATOM      1  P     G A\nEND\n")
    assert main(["pdb", "qc", str(tmp_path / "trunc.pdb")]) == 1
    assert "cannot be read" in capsys.readouterr().err
    assert main(["pdb", "qc", models[0], "--order", "score"]) == 1  # needs a decoy table
    assert main(["pdb", "qc", models[0], "--checks", "clashes,knots"]) == 1


# --- the embedded copy and the planner step -----------------------------------------------------------------

def _top_level_names(module):
    tree = ast.parse(open(module.__file__).read())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_helper_bundle_runs_standalone(tmp_path):
    """The step's helper is pdbtools + modelcheck in one file: no name of the one may replace the other's."""
    from mdatk.workflows.modelcheck import helper_source
    assert _top_level_names(pt) & _top_level_names(mc) == set()
    script = tmp_path / "helper.py"
    script.write_text(helper_source("test copy"))
    models = _models(tmp_path)
    proc = subprocess.run([sys.executable, "-I", str(script), *models, "--expect-d", "*", "--tsv", "-o",
                           str(tmp_path / "r.tsv")], capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "status: FAIL" in proc.stdout and len((tmp_path / "r.tsv").read_text().splitlines()) == 4
    assert "import numpy" not in script.read_text() and "from . import" in script.read_text()


def _plan(tmp_path, settings=None, files=None, groups=()):
    node = PlanNode(id="x", tool="model-check", step=3,
                    settings=TOOLS["model-check"].normalise(settings or {}), files=dict(files or {}))
    plan = single_step_plan(WorkflowParams(project_name="apt1", stamp=STAMP), node, str(tmp_path / "plan"),
                            list(groups))
    return plan, plan.steps["x"]


def _issues(st, level=None):
    return [i.message for i in st.issues if level is None or i.level == level]


def test_model_check_step_runs_filters_and_stops(tmp_path):
    models = _models(tmp_path)
    settings = {"on_fail": "filter", "n_pass": 1, "hands": "ranges", "expect_d": "group:Duplex",
                "segments": "A:3-6"}
    plan, st = _plan(tmp_path, settings, {"models": ";".join(models)},
                     [ResidueGroup("Duplex", "1-10", "", "#000000")])
    assert not _issues(st, "error"), st.issues
    assert len(st.outputs["passed"].paths) == 1 and st.outputs["report"].kind == "table"
    report = plan.write(["x"])
    sh = next(p for p in report.written if p.endswith(".sh"))
    assert subprocess.run(["bash", "-n", sh]).returncode == 0
    text = open(sh).read()
    assert "--expect-d '*:1-10'" in text and "--on-fail filter" in text and "--pass 1" in text
    proc = subprocess.run(["bash", sh], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    folder = os.path.dirname(sh)
    best = os.path.join(folder, f"apt1_03_best_{STAMP}.pdb")
    assert open(best).read() == open(models[0]).read()
    assert os.path.exists(os.path.join(folder, f"apt1_03_passed-01_{STAMP}.pdb"))
    rows = open(os.path.join(folder, f"apt1_03_report_{STAMP}.tsv")).read().splitlines()
    assert [r.split("\t")[1] for r in rows[1:]] == ["pass", "fail", "fail"]
    # stop: the step fails and leaves no best model behind
    plan, st = _plan(tmp_path, {"on_fail": "stop", "hands": "none"}, {"models": ";".join(models)})
    sh = next(p for p in plan.write(["x"]).written if p.endswith(".sh"))
    proc = subprocess.run(["bash", sh], capture_output=True, text=True)
    assert proc.returncode != 0 and "fail the gates" in proc.stdout + proc.stderr
    assert not os.path.exists(best) and os.path.exists(os.path.join(folder, f"apt1_03_report_{STAMP}.tsv"))


def test_model_check_step_options_and_checks(tmp_path):
    models = _models(tmp_path)
    plan, st = _plan(tmp_path, {}, {"models": ";".join(models)})
    exp = TOOLS["model-check"].expected_hands(st.settings, st.ctx)
    assert exp == ("", "A:1-10, B:1-10", "good.pdb")  # automatic: as measured on the first model
    assert any("expected hands as measured on good.pdb" in m for m in _issues(st, "info"))
    assert len(st.outputs["passed"].paths) == 3
    _plan_, st = _plan(tmp_path, {"order": "score", "score_limits": "score<", "segments": "group:Nope",
                                  "hbond_min": 3.3}, {"models": models[0]})
    errors = _issues(st, "error")
    assert any("decoy table" in m for m in errors) and any("score limit" in m for m in errors)
    assert any("no residue group named Nope" in m for m in errors)
    assert any("hydrogen-bond window" in m for m in errors)
    # an L helix from the builder: every residue expected L
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    builder_settings = TOOLS["structure-builder"].normalise({"sequence": "GGCAUC", "chirality": "L"})
    wf.nodes = [PlanNode(id="h", tool="structure-builder", step=1, settings=builder_settings),
                PlanNode(id="x", tool="model-check", step=2, settings=TOOLS["model-check"].normalise({}))]
    wf.edges = [PlanEdge("h", "pdb", "x", "models")]
    plan = Plan(wf, str(tmp_path / "p2"))
    files = {f.name: f for f in plan.generate(["x"])["x"]}
    sh = files[f"apt1_02_model-check_{STAMP}.sh"].content
    assert "--expect-l '*'" in sh and "../01_build/apt1_01_helix_" in sh


def test_model_check_step_links_kinks_hydrogens_and_slots(tmp_path):
    models = _models(tmp_path)
    react = tmp_path / "inline.txt"
    react.write_text("5 0.9\n")
    settings = {"hydrogen_check": True, "kink_allow": "group:Flexible", "kink_reactivity": str(react),
                "kink_reactivity_min": 0.4, "kink_reactivity_chain": "A", "gate_kinks": True,
                "on_fail": "filter", "n_pass": 2, "min_pass": 2, "link_fail": 3.5}
    plan, st = _plan(tmp_path, settings, {"models": ";".join(models)},
                     [ResidueGroup("Flexible", "A:3-4", "", "#000000")])
    assert not _issues(st, "error"), st.issues
    args = " ".join(TOOLS["model-check"].arguments(st.settings, st.ctx))
    for part in ("--hydrogen-check", "--kink-allow A:3-4", f"--kink-reactivity {react}",
                 "--kink-reactivity-min 0.4", "--kink-reactivity-chain A", "--gate-kinks", "--pass 2",
                 "--min-pass 2", "--link-max 2 --link-fail 3.5", "--kink-min 60"):
        assert part in args
    assert "--hydrogens " not in args + " "
    assert st.outputs["passed"].data["empty_slots"] and "up to 2 passed (at least 2)" in TOOLS[
        "model-check"].describe(st.settings)
    sh = next(p for p in plan.write(["x"]).written if p.endswith(".sh"))
    proc = subprocess.run(["bash", sh], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "1 of 3 models pass the gates; at least 2 needed" in proc.stdout + proc.stderr
    _plan_, st = _plan(tmp_path, {"on_fail": "filter", "min_pass": 4, "link_fail": 1.8},
                       {"models": ";".join(models)})
    errors = _issues(st, "error")
    assert any("4 can never pass" in m for m in errors) and any("break length" in m for m in errors)


def test_model_check_reads_the_decoy_table_contract(tmp_path):
    """farfar2-models' 'scores' table: pdb paths relative to the plan root, the step runs one folder below."""
    root = tmp_path / "plan"
    (root / "02_farfar2-models").mkdir(parents=True)
    models = []
    for k, chir in ((1, "L"), (2, "D")):
        p = root / "02_farfar2-models" / f"apt1_02_model-0{k}_{STAMP}.pdb"
        p.write_text(builder.build_helix(SEQ, chirality=chir))
        models.append(str(p))
    table = root / "02_farfar2-models" / f"apt1_02_scores_{STAMP}.tsv"
    table.write_text("run\ttag\tpdb\tscore\tlinear_chainbreak\trna_vdw\n"
                     f"r01\tS_1\t02_farfar2-models/apt1_02_model-01_{STAMP}.pdb\t-40\t0.1\t3\n"
                     f"r02\tS_2\t02_farfar2-models/apt1_02_model-02_{STAMP}.pdb\t-35\t0.2\t4\n")
    plan, st = _plan(tmp_path, {"hands": "ranges", "expect_d": "A,B", "order": "score", "on_fail": "filter",
                                "n_pass": 1, "score_limits": "linear_chainbreak<=0.5"},
                     {"models": ";".join(models), "scores": str(table)})
    assert not _issues(st, "error"), st.issues
    sh = next(p for p in plan.write(["x"]).written if p.endswith(".sh"))
    assert "--plan-root .." in open(sh).read()
    proc = subprocess.run(["bash", sh], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    folder = os.path.dirname(sh)
    assert open(os.path.join(folder, f"apt1_03_best_{STAMP}.pdb")).read() == open(models[1]).read()
    text = open(os.path.join(folder, f"apt1_03_report_{STAMP}.tsv")).read()
    rows = [r.split("\t") for r in text.splitlines()]
    assert rows[0][-2:] == ["score", "linear_chainbreak"] and rows[1][-2:] == ["-40", "0.1"]


# --- Amber build --------------------------------------------------------------------------------------------

def _amber(tmp_path, settings=None, structure=""):
    node = PlanNode(id="x", tool="amber-build", step=3,
                    settings=TOOLS["amber-build"].normalise(settings or {}),
                    files={"structure": structure or "/golden/inputs/structure.pdb"})
    plan = single_step_plan(WorkflowParams(project_name="apt1", stamp=STAMP), node, str(tmp_path / "plan"))
    return plan, plan.steps["x"], {f.name: f for f in plan.generate(["x"])["x"]}


def _fake_amber(root):
    """An AmberTools data folder with leaprc.RNA.OL3 loading a parm10.dat that has the real OS-P term."""
    for sub in ("cmd", "parm", "lib"):
        (root / "dat" / "leap" / sub).mkdir(parents=True, exist_ok=True)
    (root / "dat" / "leap" / "cmd" / "leaprc.RNA.OL3").write_text("parm10 = loadamberparams parm10.dat\n")
    (root / "dat" / "leap" / "parm" / "parm10.dat").write_text(PARM10)
    return root


def test_amber_build_link_check_and_virtual_residues(tmp_path):
    _p, _st, off = _amber(tmp_path)
    sh = off[f"apt1_03_amber-build_{STAMP}.sh"].content
    assert f"apt1_03_links_{STAMP}.py" not in off and "links" not in sh  # off by default: as before
    assert "REMARK 951" not in sh  # a hand-picked structure is no model check's slot
    assert "drop_virtual=True)" in off[f"apt1_03_prep_{STAMP}.py"].content
    for mode in ("warn", "stop"):
        _p, _st, files = _amber(tmp_path, {"link_check": mode, "link_max": 2.5})
        sh = files[f"apt1_03_amber-build_{STAMP}.sh"].content
        line = next(ln for ln in sh.splitlines() if f"apt1_03_links_{STAMP}.py" in ln)
        assert (f"apt1_03_lrna_{STAMP}.pdb --checks links --link-min 1.4 --link-max 2.5 --link-fail 3 "
                f"--link-params apt1_03_tleap_{STAMP}.leap --link-types OR,OS --amberhome \"$AMBERHOME\" "
                "--on-fail") in line
        assert ("tleap was not run" in line) == (mode == "stop")
        assert sh.index('AMBERHOME="${AMBERHOME:-') < sh.index(line) < sh.index("tleap -f")
    _p, _st, ol3 = _amber(tmp_path, {"link_check": "stop", "rna_ff": "ol3", "nbfix19": False, "bph0": False,
                                     "two_pass": True})
    line = next(ln for ln in ol3[f"apt1_03_amber-build_{STAMP}.sh"].content.splitlines()
                if f"apt1_03_links_{STAMP}.py" in ln)
    assert f"--link-params apt1_03_tleap-dry_{STAMP}.leap --link-types OS " in line  # the first tleap's
    # the helper reads k and r0 from the force field the tleap input loads, warns with the strain between
    # 2 and 3 Å and ends the build (before tleap) when a link is a break
    home = _fake_amber(tmp_path / "amber")
    work = tmp_path / "run"
    work.mkdir()
    (work / "tleap.leap").write_text("source leaprc.RNA.OL3\nloadamberparams frcmod.cp\n")
    (work / "links.py").write_text(files[f"apt1_03_links_{STAMP}.py"].content)
    for target, code in ((2.5, 0), (3.4, 1)):
        (work / "m.pdb").write_text(pt.write_pdb(_stretched(target)))
        proc = subprocess.run([sys.executable, "-I", "links.py", "m.pdb", "--checks", "links",
                               "--link-params", "tleap.leap", "--link-types", "OR,OS", "--amberhome",
                               str(home), "--on-fail", "stop"], capture_output=True, text=True, cwd=work)
        assert proc.returncode == code, proc.stdout + proc.stderr
        assert "OS-P of " in proc.stdout and "parm10.dat" in proc.stdout
        status = "strained; strain 182" if code == 0 else "long; strain 737"
        assert f"A:5 -> A:6: {target:.2f} A ({status} kcal/mol)" in proc.stdout
    # an existing input: the links as tleap will see them, and the virtual residues, at planning time
    vrt = "HETATM 9999 ORIG VRT C   1       0.000   0.000   0.000  1.00  0.00           X"
    pdb = tmp_path / "broken.pdb"
    pdb.write_text(pt.write_pdb(_stretched(3.4)).replace("END\n", vrt + "\nEND\n"))
    _p, st, _f = _amber(tmp_path, {"link_check": "stop"}, str(pdb))
    warnings = _issues(st, "warning")
    assert any("1 O3′–P link(s) longer than 3 Å (a break)" in m and "stops before tleap" in m
               for m in warnings)
    assert any("1 Rosetta virtual residue(s)" in m for m in _issues(st, "info"))
    _p, st, _f = _amber(tmp_path, {"drop_virtual": False}, str(pdb))
    assert any("tleap cannot type their atoms" in m for m in _issues(st, "warning"))
    pdb.write_text(pt.write_pdb(_stretched(2.4)))
    _p, st, _f = _amber(tmp_path, {"link_check": "stop"}, str(pdb))
    warnings = _issues(st, "warning")
    assert any("1 O3′–P link(s) of 2–3 Å (A:5-A:6 2.40 Å (144 kcal/mol))" in m and "first minimisation" in m
               for m in warnings) and not any("stops before tleap" in m for m in warnings)
    node = PlanNode(id="x", tool="amber-build", step=3, files={"structure": "/golden/inputs/structure.pdb"},
                    settings=TOOLS["amber-build"].normalise({"link_check": "warn", "link_max": 2.5,
                                                             "link_fail": 2.2}))
    plan = single_step_plan(WorkflowParams(project_name="apt1", stamp=STAMP), node, str(tmp_path / "e"))
    assert any("break length" in m for m in _issues(plan.steps["x"], "error"))


@pytest.mark.parametrize("settings", [
    {}, {"two_pass": True}, {"lrna_library": "generated", "lrna_ranges": "B"}, {"water": "tip4p2005"},
    {"prep": False}, {"rna_ff": "ol3", "nbfix19": False, "bph0": False}, {"cp_source": "frcmod"},
    {"two_pass": True, "lrna_library": "generated", "lrna_ranges": "B", "water": "tip4p2005"}])
def test_amber_link_check_runs_before_every_tleap_and_helper(tmp_path, settings):
    """What stops a build stops it before tleap: the check of the PDB tleap reads comes before the dry
    tleap of a two-pass build, the solvated one, and the CP, L-RNA library and TIP4P/2005 helpers."""
    _p, _st, files = _amber(tmp_path, dict(settings, link_check="stop"))
    sh = files[f"apt1_03_amber-build_{STAMP}.sh"].content.splitlines()
    check = next(k for k, ln in enumerate(sh) if ln.startswith(f"python3 apt1_03_links_{STAMP}.py"))
    tleaps = [k for k, ln in enumerate(sh) if ln.startswith("tleap ")]
    helper = re.compile(r"python3 apt1_03_(cp-lib|lrna-lib|tip4p2005)_")
    helpers = [k for k, ln in enumerate(sh) if helper.match(ln)]
    assert tleaps and all(check < k for k in tleaps + helpers)
    assert len(tleaps) == (2 if settings.get("two_pass") else 1)
    assert any("lrna-lib" in ln for ln in sh) == (settings.get("lrna_library") == "generated")
    reading = ("python3 apt1_03_prep_", "python3 apt1_03_pdbtools_")
    # it reads the PDB as tleap will: prepared and L-named first (a two-pass build names ambpdb's PDB later)
    named = [k for k, ln in enumerate(sh) if ln.startswith(reading) and " -o apt1_03_lrna-dry_" not in ln]
    assert named and all(k < check for k in named)
    # without the check a build keeps the order it had: the helpers first, then the PDB
    _p, _st, off = _amber(tmp_path, settings)
    plain = off[f"apt1_03_amber-build_{STAMP}.sh"].content.splitlines()
    first = next((k for k, ln in enumerate(plain) if ln.startswith(reading)), len(plain))
    assert all(k < first for k, ln in enumerate(plain) if helper.match(ln))


def test_empty_slots_stop_single_structure_builds_before_tleap(tmp_path):
    """A model check filtering to up to N models passes empty slots on; an Amber build fed one stops before
    any helper or tleap runs, with the slot's message."""
    models = _models(tmp_path)
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    wf.nodes = [PlanNode(id="q", tool="model-check", step=1, settings=TOOLS["model-check"].normalise(
                    {"on_fail": "filter", "n_pass": 3, "hands": "ranges", "expect_d": "A,B"}),
                    files={"models": ";".join(models)}),
                PlanNode(id="a", tool="amber-build", step=2, settings=TOOLS["amber-build"].normalise({}))]
    wf.edges = [PlanEdge("q", "passed", "a", "structure")]
    plan = Plan(wf, str(tmp_path / "p"))
    q = plan.steps["q"]
    assert q.outputs["passed"].data.get("empty_slots") and not q.outputs["best"].data.get("empty_slots")
    assert len(q.outputs["passed"].paths) == 3
    written = plan.write(["q", "a"]).written
    qsh = next(p for p in written if p.endswith(f"apt1_01_model-check_{STAMP}.sh"))
    assert "--min-pass 1" in open(qsh).read()
    proc = subprocess.run(["bash", qsh], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 of 3 models passed the gates; passed on 1 in 3 slot(s), slots 2-3 written empty" in proc.stdout
    folder = os.path.dirname(qsh)
    slot = os.path.join(folder, f"apt1_01_passed-01_{STAMP}.pdb")
    assert open(slot).read() == open(models[0]).read()
    empty = os.path.join(folder, f"apt1_01_passed-02_{STAMP}.pdb")
    assert pt.empty_slot(empty) == "empty slot 2 of 3: only 1 models passed"
    with pytest.raises(pt.PDBError, match=r"an empty slot of a model check \(empty slot 2 of 3"):
        pt.read_pdb(empty)
    # the build reads slot 1 (a model); given an empty one it stops at once, before AMBERHOME, the
    # helpers and tleap
    ash = next(p for p in written if p.endswith(f"apt1_02_amber-build_{STAMP}.sh"))
    text = open(ash).read()
    assert text.index("REMARK 951") < text.index("AMBERHOME") < text.index("tleap -f")
    os.replace(empty, slot)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "ran"
    for tool in ("tleap", "parmed", "python3"):
        (bin_dir / tool).write_text(f"#!/bin/sh\ntouch {marker}\n")
        (bin_dir / tool).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    proc = subprocess.run(["bash", ash], capture_output=True, text=True, env=env)
    assert proc.returncode == 1 and not marker.exists()
    assert ("is an empty slot of the model check (empty slot 2 of 3: only 1 models passed): no model to "
            "build; tleap was not run") in proc.stderr
    _p, st, _f = _amber(tmp_path, {}, slot)  # chosen by hand: the plan says so, and the guard is written
    assert any("is an empty slot of a model check" in m for m in _issues(st, "warning"))
    assert "REMARK 951" in _f[f"apt1_03_amber-build_{STAMP}.sh"].content
    # the PDB preparation says so too, should it ever be run on one
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert prep_pdb.main([slot, str(tmp_path / "prep.pdb")]) == 1
    assert "is an empty slot of a model check (empty slot 2 of 3: only 1 models passed)" in err.getvalue()


def test_model_readers_skip_empty_slots(tmp_path, capsys):
    """MDATK's readers of model sets skip an empty slot with a note instead of failing on it or counting it:
    the decoy interfaces, the register census and ensemble comparison, the frame check (the map-model score:
    tests/test_ensemble.py)."""
    from mdatk import ensemble, frames, interfaces
    good = tmp_path / "good.pdb"
    good.write_text(builder.build_helix(SEQ))
    slot = tmp_path / "slot-2.pdb"
    slot.write_text(pt.empty_slot_text(2, 5, 1))
    assert interfaces._unique_files([str(good), str(slot)]) == [str(good)]
    assert "skipped" in capsys.readouterr().err
    assert ensemble.expand_models([str(tmp_path)]) == [str(good)]
    assert "empty slot 2 of 5: only 1 models passed" in capsys.readouterr().err
    models, notes, _t = frames.gather_models([str(tmp_path)])
    assert [m.path for m in models] == [str(good)] and any("empty slot 2 of 5" in n for n in notes)
