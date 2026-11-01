"""Tests of the starting-structure builder (mdatk.builder): ideal A-form helices, mutations, density fits,
the 'mdatk build' command, the planner step and the GUI window."""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from mdatk import builder as B
from mdatk import pdbtools as pt
from mdatk.secstruct import WATSON_CRICK, WOBBLE, PairDetector

USERFILES = os.environ.get("MDATK_TEST_USERFILES",
                           "/tmp/claude-0/-home-user-MDATK/85f8b95c-a2c8-519b-b4e2-60a59ac5f943/scratchpad/"
                           "userfiles03")
HELICES = os.path.join(USERFILES, "3bcdf791-helices_73.pdb")
HELICES_FIXED = os.path.join(USERFILES, "helices_fixed.pdb")  # the rebuilt starting helices


# --- helpers ----------------------------------------------------------------------------------------

def pairs_of(text, wobble=True):
    """[(residue i, residue j, kind)] found by the builtin detector, plus the residues."""
    s = pt.read_pdb(text)
    residues = list(s.residues())
    index, rows, n = [], [], 0
    for r in residues:
        c = r.coords()
        index.append({k: n + i for i, k in enumerate(c)})
        rows += list(c.values())
        n += len(c)
    found = PairDetector([r.name for r in residues], index, cutoff=3.5, wobble=wobble).detect(np.array(rows))
    return [(residues[i], residues[j], k) for i, j, k in found], residues


def heavy_atoms(s):
    out = []
    for k, r in enumerate(s.residues()):
        for a in r.atoms:
            if not a.name.startswith("H"):
                out.append((k, r, a.name, np.array(a.xyz)))
    return out


def topology_distances(atoms, links):
    """Bond-path length (capped at 3) between heavy atoms: bonds within residues by distance, plus O3'-P."""
    n = len(atoms)
    xyz = np.array([a[3] for a in atoms])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    same = np.array([[a[0] == b[0] for b in atoms] for a in atoms])
    bond = same & (d < 1.75) & ~np.eye(n, dtype=bool)
    for i, j in links:
        bond[i, j] = bond[j, i] = True
    topo = np.full((n, n), 9)
    np.fill_diagonal(topo, 0)
    reach = np.eye(n, dtype=bool)
    for step in (1, 2, 3):
        reach = reach | (reach.astype(int) @ bond.astype(int) > 0)
        topo[(topo == 9) & reach] = step
    return d, topo


def backbone_links(s, atoms):
    pos = {(a[0], a[2]): i for i, a in enumerate(atoms)}
    links = []
    k = 0
    for seg in s.segments:
        for a, b in zip(range(k, k + len(seg)), range(k + 1, k + len(seg))):
            if (a, "O3'") in pos and (b, "P") in pos:
                links.append((pos[(a, "O3'")], pos[(b, "P")]))
        k += len(seg)
    return links


def o3p_distances(s):
    return [np.linalg.norm(np.subtract(a.coords()["O3'"], b.coords()["P"]))
            for seg in s.segments for a, b in zip(seg, seg[1:])]


# --- ideal helices ----------------------------------------------------------------------------------

@pytest.mark.parametrize("chirality", ["D", "L"])
def test_ideal_duplex_geometry(chirality):
    seq = "GGCAUCGAUGCCAG"
    notes = []
    text = B.build_helix(seq, chirality=chirality, notes=notes)
    s = pt.read_pdb(text)
    assert [len(seg) for seg in s.segments] == [14, 14] and s.chain_ids() == ["A", "B"]
    # O3'(i)-P(i+1) on both strands
    d = o3p_distances(s)
    assert len(d) == 26 and 1.55 <= min(d) and max(d) <= 1.65
    # every intended Watson-Crick pair is found, with canonical C1'-C1' distances
    found, residues = pairs_of(text)
    want = {(i, 2 * len(seq) - 1 - i) for i in range(len(seq))}
    got = {(residues.index(a), residues.index(b)) for a, b, k in found if k == WATSON_CRICK}
    assert got == want and len(found) == len(seq)
    for a, b, _k in found:
        c1 = np.linalg.norm(np.subtract(a.coords()["C1'"], b.coords()["C1'"]))
        assert 10.0 <= c1 <= 11.0
    # no non-bonded heavy-atom clash (pairs more than two bonds apart)
    atoms = heavy_atoms(s)
    dist, topo = topology_distances(atoms, backbone_links(s, atoms))
    assert dist[topo >= 3].min() >= 2.5
    # chirality from the sugar geometry, names from pdbtools
    chir_notes = pt.assign_chirality(s)
    assert {r.chirality for r in s.residues()} == {chirality}
    n_l = 2 * len(seq) if chirality == "L" else 0
    assert chir_notes[0] == f"Chirality (sugar geometry): {n_l} L and {2 * len(seq) - n_l} D nucleotides"
    names = {r.name for r in s.residues()}
    assert names == ({"A", "C", "G", "U"} if chirality == "D" else {"0A", "0C", "0G", "0U"})
    assert "REMARK 951" in text and any("Built an ideal A-form RNA duplex" in n for n in notes)


def test_l_helix_is_the_mirror_image_and_left_handed():
    d = pt.read_pdb(B.build_helix("GGCAUC"))
    lh = pt.read_pdb(B.build_helix("GGCAUC", chirality="L"))
    # the coordinates of every atom name are mirrored: prochiral names (OP1/OP2, H5'/H5'') are the
    # mirror images of the D names, as in pdbtools' L conversion, the dictionaries and real L models
    for rd, rl in zip(d.residues(), lh.residues()):
        cd, cl = rd.coords(), rl.coords()
        for name, x in cd.items():
            assert np.allclose(np.multiply(x, [-1, 1, 1]), cl[name], atol=2e-3)

    # handedness: strand 1 turns by -32.7° per step about +z going down for D, the opposite for L
    def turn(s):
        p = [np.array(r.coords()["C1'"]) for r in s.segments[0]]
        a0, a1 = np.arctan2(p[0][1], p[0][0]), np.arctan2(p[1][1], p[1][0])
        return np.degrees((a1 - a0 + np.pi) % (2 * np.pi) - np.pi), p[1][2] - p[0][2]
    td, zd = turn(d)
    tl, zl = turn(lh)
    assert td == pytest.approx(-B.TWIST, abs=0.05) and zd == pytest.approx(-B.RISE, abs=0.01)
    assert tl == pytest.approx(B.TWIST, abs=0.05) and zl == pytest.approx(-B.RISE, abs=0.01)


def test_single_strand_partner_and_validation_messages():
    text = B.build_helix("GGACU", partner="none", chains="C", start=10)
    s = pt.read_pdb(text)
    assert len(s.segments) == 1 and [r.resseq for r in s.residues()] == [10, 11, 12, 13, 14]
    assert s.chain_ids() == ["C"]
    with pytest.raises(B.BuildError, match="same length"):
        B.build_helix("GGACU", partner="AGU")
    with pytest.raises(B.BuildError, match="invalid letter"):
        B.build_helix("GGXCU")
    with pytest.raises(B.BuildError, match="empty"):
        B.build_helix("  ")
    with pytest.raises(B.BuildError, match="different chain"):
        B.build_helix("GGC", chains="A,A")
    assert B.parse_sequence("5'-GGT ACU-3'")[0] == "GGUACU"
    msgs = B.check_helix("GGCAU", "AUGCC")
    assert ("info", "the explicit partner is the Watson-Crick complement") in msgs
    msgs = dict((t, lvl) for lvl, t in B.check_helix("GGCAU", "AAGUC"))
    assert any(lvl == "warning" and "mismatched" in t for t, lvl in msgs.items())
    assert B.check_helix("GGC", "GG")[0][0] == "error"


def test_explicit_partner_with_wobble_and_mismatch():
    notes = []
    # pairs: G1·U, G2·C, C3·G, A4·A (mismatch), U5·A
    text = B.build_helix("GGCAU", partner="AAGCU", notes=notes)
    found, residues = pairs_of(text)
    kinds = {(a.resseq, a.chain): k for a, b, k in found}
    assert kinds[(1, "A")] == WOBBLE and kinds[(2, "A")] == WATSON_CRICK and kinds[(5, "A")] == WATSON_CRICK
    assert (4, "A") not in kinds
    assert any("wobble geometry" in n for n in notes)
    s = pt.read_pdb(text)
    d = o3p_distances(s)
    assert 1.55 <= min(d) and max(d) <= 1.65  # the wobble turns leave the backbone alone
    # a purine-purine mismatch collides in the Watson-Crick frame and is reported
    notes = []
    B.build_helix("GGGG", partner="CCAC", notes=notes)
    assert any(n.startswith("WARNING: mismatch") for n in notes)


def test_names_terminals_hydrogens_and_phosphate():
    s = pt.read_pdb(B.build_helix("GCA", scheme="amber"))
    assert [r.name for r in s.residues()] == ["G5", "C", "A3", "U5", "G", "C3"]
    s = pt.read_pdb(B.build_helix("GCA", chirality="L", scheme="amber"))
    assert [r.name for r in s.residues()] == ["LG5", "LC", "LA3", "LU5", "LG", "LC3"]
    s = pt.read_pdb(B.build_helix("GCA", chirality="L", scheme="ccd", terminals=False))
    assert [r.name for r in s.residues()][:3] == ["0G", "0C", "0A"]
    text = B.build_helix("GCA", chirality="L", scheme="pdb")
    assert all(line.startswith("HETATM") for line in text.splitlines() if " 0G " in line[16:21])
    first = list(pt.read_pdb(B.build_helix("GCA")).residues())[0]
    assert "P" not in [a.name for a in first.atoms]  # 5'-OH by default
    first = list(pt.read_pdb(B.build_helix("GCA", phosphate5=True)).residues())[0]
    assert [a.name for a in first.atoms][:4] == ["OP3", "P", "OP1", "OP2"]
    c = first.coords()
    assert np.linalg.norm(np.subtract(c["OP3"], c["P"])) == pytest.approx(1.48, abs=0.01)
    s = pt.read_pdb(B.build_helix("GCAU", hydrogens=True))
    res = list(s.residues())
    g5, a3 = res[0], res[2]
    assert {"H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", "H1", "H21", "H22", "H8", "HO5'"} <= {
        a.name for a in g5.atoms}
    assert "HO3'" in {a.name for a in res[3].atoms} and {"H61", "H62", "H2"} <= {a.name for a in a3.atoms}
    for r in res:  # every hydrogen 0.9-1.1 Å from its nearest heavy atom
        heavy = np.array([a.xyz for a in r.atoms if not a.name.startswith("H")])
        for a in r.atoms:
            if a.name.startswith("H"):
                assert 0.9 < np.linalg.norm(heavy - a.xyz, axis=1).min() < 1.12
    # PDB v3 prochiral names on the D sugar (as in Amber / wwPDB): positive volume C5'(O5', C4', H5')
    c = g5.coords()
    assert pt.chiral_volume(c["C5'"], c["O5'"], c["C4'"], c["H5'"]) > 0
    assert pt.chiral_volume(c["P"] if "P" in c else res[1].coords()["P"], *(
        res[1].coords()[n] for n in ("O5'", "OP1", "OP2"))) < 0
    assert pt.assign_chirality(s)[0].startswith("Chirality (sugar geometry): 0 L and 8 D")


def test_helix_sugars_are_c3_endo():
    t = B.template("G", hydrogens=False)

    def dih(a, b, c, d):
        b0, b1, b2 = a - b, c - b, d - c
        b1 = b1 / np.linalg.norm(b1)
        v, w = b0 - b0 @ b1 * b1, b2 - b2 @ b1 * b1
        return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))
    delta = dih(t["C5'"], t["C4'"], t["C3'"], t["O3'"])
    nu2 = dih(t["C1'"], t["C2'"], t["C3'"], t["C4'"])
    chi = dih(t["O4'"], t["C1'"], t["N9"], t["C4"])
    assert 75 < delta < 90 and nu2 > 30 and -175 < chi < -150


@pytest.mark.parametrize("model", [HELICES, HELICES_FIXED])
def test_backbone_matches_an_experimental_a_form_helix(model):
    from mdatk.spatial import kabsch
    if not os.path.isfile(model):
        pytest.skip("user model not available")
    ref = {(r.chain, r.resseq): r.coords() for r in pt.read_pdb(model).residues()}

    def p_rmsd(seq, chirality, keys):
        mine = [r.coords() for r in pt.read_pdb(B.build_helix(seq, chirality=chirality,
                                                              phosphate5=True)).residues()]
        x = np.array([m["P"] for m, k in zip(mine, keys) if "P" in ref[k]])
        y = np.array([ref[k]["P"] for m, k in zip(mine, keys) if "P" in ref[k]])
        r, mc, rc = kabsch(x, y)
        rmsd = np.sqrt((((x - mc) @ r.T + rc - y) ** 2).sum(axis=1).mean())
        print(f"{os.path.basename(model)} {seq} {chirality}: P RMSD {rmsd:.2f} A over {len(x)}")
        return rmsd, len(x)

    # chain A 1-6 pairs with B 44-39 and A 13-16 with B 20-17 (two stems, not stacked on each other)
    rmsd, n = p_rmsd("GGGCAA", "D", [("A", i) for i in range(1, 7)] + [("B", i) for i in range(39, 45)])
    assert n == 12 and rmsd < 2.0
    rmsd, n = p_rmsd("AGCU", "D", [("A", i) for i in range(13, 17)] + [("B", i) for i in range(17, 21)])
    assert n == 8 and rmsd < 2.0
    # the L-RNA stem C45-50 / C68-73 against the L helix (and not against the D helix)
    keys = [("C", i) for i in range(45, 51)] + [("C", i) for i in range(68, 74)]
    assert p_rmsd("GCCAGA", "L", keys)[0] < 2.0
    assert p_rmsd("GCCAGA", "D", keys)[0] > 5.0


def test_sugar_stereocentres_with_rdkit():
    Chem = pytest.importorskip("rdkit.Chem")

    def cip(text, resseq):
        s = pt.read_pdb(text)
        res = [r for r in s.residues() if r.resseq == resseq][0]
        one = pt.Structure(segments=[[res]])
        mol = Chem.MolFromPDBBlock(pt.write_pdb(one), removeHs=False, sanitize=False, proximityBonding=True)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        Chem.AssignStereochemistryFrom3D(mol)
        labels = {}
        for atom in mol.GetAtoms():
            name = atom.GetPDBResidueInfo().GetName().strip()
            if name in ("C1'", "C2'", "C3'", "C4'") and atom.HasProp("_CIPCode"):
                labels[name] = atom.GetProp("_CIPCode")
        return [labels.get(n) for n in ("C1'", "C2'", "C3'", "C4'")]

    # adenosine is (2R,3R,4S,5R): C1' R, C2' R, C3' S, C4' R; the L enantiomer has every centre inverted
    for chirality, want in (("D", ["R", "R", "S", "R"]), ("L", ["S", "S", "R", "S"])):
        text = B.build_helix("AAA", partner="none", chirality=chirality, hydrogens=True)
        assert cip(text, 2) == want, chirality


# --- mutations --------------------------------------------------------------------------------------

def test_mutations_keep_the_frame_and_compensate():
    src = B.build_helix("GGCAUCG", hydrogens=True)
    before = {(r.chain, r.resseq): r for r in pt.read_pdb(src).residues()}
    text, notes = B.mutate(src, "A:1C, A:4G")
    s = pt.read_pdb(text)
    after = {(r.chain, r.resseq): r for r in s.residues()}
    assert [after[("A", i)].name for i in (1, 4)] == ["C", "G"]
    assert [after[("B", i)].name for i in (7, 4)] == ["G", "C"]  # the compensatory partners
    assert any("compensatory, compensates A:1 C" in n for n in notes)
    assert any("dropped 4 hydrogen(s) of the old base" in n for n in notes)
    # sugar / backbone atoms verbatim, the new glycosidic N on the old one
    old, new = before[("A", 4)], after[("A", 4)]
    for a in old.atoms:
        if "'" in a.name or a.name in ("P", "OP1", "OP2"):
            assert new.coords()[a.name] == old.coords()[a.name]
    assert np.linalg.norm(np.subtract(new.coords()["N9"], old.coords()["N9"])) < 0.05
    pyr = after[("A", 1)].coords()
    assert np.linalg.norm(np.subtract(pyr["N1"], before[("A", 1)].coords()["N9"])) < 0.1
    assert not any(a.name in ("H8", "H61", "H62", "H2", "N6") for a in new.atoms)  # old base and its H gone
    found, _ = pairs_of(text)
    assert len(found) == 7 and all(k == WATSON_CRICK for *_x, k in found)
    assert pt.assign_chirality(s)[0].endswith("0 L and 14 D nucleotides")


@pytest.mark.parametrize("scheme,names", [("amber", ("LG5", "LC5", "LC3", "LG3")),
                                          ("ccd", ("0G5", "0C5", "0C3", "0G3")),
                                          (None, ("0G", "0C", "0C", "0G"))])
def test_l_rna_mutation_keeps_l_names_and_chirality(scheme, names):
    src = B.build_helix("GAUC", chirality="L", scheme=scheme)
    text, notes = B.mutate(src, "A:1C")
    s = pt.read_pdb(text)
    res = {(r.chain, r.resseq): r for r in s.residues()}
    old = {(r.chain, r.resseq): r for r in pt.read_pdb(src).residues()}
    assert (old[("A", 1)].name, res[("A", 1)].name, old[("B", 4)].name, res[("B", 4)].name) == names
    assert pt.assign_chirality(s)[0].startswith("Chirality (sugar geometry): 8 L and 0 D")
    for key in (("A", 1), ("B", 4)):
        call, vols = pt.sugar_chirality(res[key].coords(), pt.parse_residue_name(res[key].name).base)
        assert call == "L" and min(abs(v) for v in vols.values()) > 1.5
    found, _ = pairs_of(text)
    assert len(found) == 4 and all(k == WATSON_CRICK for *_x, k in found)


def test_mutation_options_and_errors():
    src = B.build_helix("GGCAUCG")
    # explicit partner mutation: no compensation, mismatch warned
    _text, notes = B.mutate(src, "A:2U, B:6A")
    assert not any("compensatory" in n for n in notes)
    _text, notes = B.mutate(src, "A:2A, B:6A")
    assert any("do not pair canonically" in n for n in notes)
    # no compensation
    text, notes = B.mutate(src, "A:2U", compensatory=False)
    assert [r.name for r in pt.read_pdb(text).residues()][7 + 5] == "C"
    # wobble: a G facing U stays, a U facing G stays
    wob = B.build_helix("GGCAUCG", partner="CGAUGUC")  # pairs A:2 G · B:6 U
    text, notes = B.mutate(wob, "A:6U", wobble=True)  # C6 -> U facing G2 of strand B: keeps the G
    assert any("already pairs (G·U wobble)" in n for n in notes)
    text, notes = B.mutate(wob, "A:6U")
    assert any("compensates A:6 U" in n for n in notes)
    found, _ = pairs_of(text)
    assert [k for a, b, k in found if ("A", 6) in ((a.chain, a.resseq), (b.chain, b.resseq))] == [WATSON_CRICK]
    with pytest.raises(B.BuildError, match="not found"):
        B.mutate(src, "A:99G")
    with pytest.raises(B.BuildError, match="cannot read"):
        B.mutate(src, "A5")
    with pytest.raises(B.BuildError, match="twice"):
        B.parse_mutations("A:5G, A:5C")
    assert [(m.chain, m.resseq, m.base) for m in B.parse_mutations("A:5g; B:-2>C :7U")] == [
        ("A", 5, "G"), ("B", -2, "C"), ("", 7, "U")]
    assert B.mutated_name("LG5", "C") == "LC5" and B.mutated_name("GUA", "U") == "URA"
    assert B.mutated_name("RA3", "G") == "RG3"


# --- rigid-body fit ---------------------------------------------------------------------------------

def _moved_map(tmp_path, text, rotvec, shift, resolution=4.0):
    from mdatk.density import atom_weights, rotvec_matrix, simulate_map
    from mdatk.io.mrc import write_map
    xyz, el = B.model_atoms(pt.read_pdb(text))
    rot = rotvec_matrix(np.array(rotvec))
    moved = xyz @ rot.T + np.array(shift)
    path = tmp_path / "sim.mrc"
    write_map(str(path), simulate_map(moved, atom_weights(el), resolution))
    return str(path), moved, rot


def test_global_fit_recovers_the_pose(tmp_path):
    model = B.build_helix("GAAGAGAAGGC")
    path, moved, _rot = _moved_map(tmp_path, model, [0.4, -0.9, 0.3], [30.0, -12.0, 55.0])
    text, cc, notes = B.fit_to_map(model, path, 4.0, "global")
    xyz, _ = B.model_atoms(pt.read_pdb(text))
    assert np.sqrt(((xyz - moved) ** 2).sum(axis=1).mean()) < 1.0
    assert cc > 0.9 and any(n.startswith("Density fit (global search)") for n in notes)
    # only the coordinates changed
    a, b = model.splitlines(), text.splitlines()
    assert len(a) == len(b)
    for la, lb in zip(a, b):
        if la.startswith(("ATOM", "HETATM")):
            assert la[:30] == lb[:30] and la[54:] == lb[54:]
        else:
            assert la == lb


def test_local_fit_and_anisou_rotation(tmp_path):
    from mdatk.density import rotvec_matrix
    model = B.build_helix("GGCAU", partner="none")
    path, moved, rot = _moved_map(tmp_path, model, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    small = rotvec_matrix(np.array([0.03, -0.04, 0.02]))
    start = B.transform_pdb_text(model, small, np.array([0.8, -0.6, 0.4]))
    text, cc, notes = B.fit_to_map(start, path, 4.0, "local")
    xyz, _ = B.model_atoms(pt.read_pdb(text))
    assert np.sqrt(((xyz - moved) ** 2).sum(axis=1).mean()) < 0.3 and cc > 0.9
    line = "ANISOU    1  P     G A   1  " + "".join(f"{u:7d}" for u in (1000, 2000, 3000, 0, 0, 0)) + "       P"
    assert len(line[:28]) == 28 and line[28:35] == "   1000"
    r90 = rotvec_matrix(np.array([0.0, 0.0, np.pi / 2]))
    out = B.transform_pdb_text(line + "\n", r90, np.zeros(3))
    assert out.split()[6:9] == ["2000", "1000", "3000"] and out.rstrip().endswith("P")
    with pytest.raises(B.BuildError, match="resolution"):
        B.fit_to_map(model, path, 0, "local")


# --- command line -----------------------------------------------------------------------------------

def test_build_cli_logs_to_the_project(tmp_path, capsys):
    import json

    from mdatk.cli import main
    from mdatk.config import Project
    from mdatk.projectfile import read_log

    proj = tmp_path / "p.mdatk.toml"
    Project().save(str(proj))
    helix = tmp_path / "helix.pdb"
    assert main(["build", "helix", "--seq", "GGCAU", "--chirality", "L", "--scheme", "amber", "--hydrogens",
                 "-o", str(helix), "-p", str(proj)]) == 0
    assert [r.name for r in pt.read_pdb(str(helix)).residues()][:2] == ["LG5", "LG"]
    entry = read_log(str(proj))[-1]
    assert entry["step"] == "build helix" and str(helix) in entry["files"] and "Built" in entry["summary"]
    mut = tmp_path / "mut.pdb"
    assert main(["build", "mutate", str(helix), "A:2C", "--no-compensatory", "-o", str(mut), "-p",
                 str(proj)]) == 0
    assert [r.name for r in pt.read_pdb(str(mut)).residues()][:2] == ["LG5", "LC"]
    assert read_log(str(proj))[-1]["step"] == "build mutate"
    path, _moved, _rot = _moved_map(tmp_path, helix.read_text(), [0, 0, 0], [0.5, 0.0, 0.0])
    fitted = tmp_path / "fit.pdb"
    capsys.readouterr()
    assert main(["build", "fit", str(helix), path, "--resolution", "4", "--search", "local", "-o", str(fitted),
                 "-p", str(proj)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cc"] > 0.9
    entry = read_log(str(proj))[-1]
    assert entry["step"] == "build fit" and "CC 0." in entry["summary"]
    assert main(["build", "mutate", str(helix), "Z:1C", "-o", str(mut)]) == 1  # error reported, no traceback


# --- planner step -----------------------------------------------------------------------------------

def test_planner_step_checks_and_script(tmp_path):
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import StepContext
    from mdatk.workflows.plan import Plan, add_node, connect
    from mdatk.config import WorkflowParams

    tool = TOOLS["structure-builder"]
    assert tool.category == "Structure building"

    def issues(**settings):
        s = tool.normalise(settings)
        return [(i.level, i.message) for i in tool.check(s, StepContext("p", 1, "s", tool.tag))]

    assert ("error", "enter the sequence") in issues()
    assert any(lvl == "error" and "invalid letter" in m for lvl, m in issues(sequence="GGXU"))
    assert any(lvl == "error" and "same length" in m
               for lvl, m in issues(sequence="GGCU", partner_mode="explicit", partner="AGC"))
    assert any(lvl == "error" and "partner sequence" in m for lvl, m in issues(sequence="GGCU",
                                                                              partner_mode="explicit"))
    assert ("error", "enter the map resolution") in issues(mode="fit")
    assert any(lvl == "error" and "cannot read mutation" in m for lvl, m in issues(mode="mutate",
                                                                                  mutations="A5"))
    assert not [m for lvl, m in issues(sequence="GGCAU", chirality="L") if lvl == "error"]
    assert [p.name for p in tool.input_ports(tool.normalise({"mode": "fit"}))] == ["pdb", "map"]
    assert tool.input_ports(tool.normalise({})) == []

    # a plan: helix -> mutate -> fit (map from an existing file), scripts pass bash -n
    model = tmp_path / "start.pdb"
    model.write_text(B.build_helix("GGCAUCG"))
    wf = WorkflowParams(project_name="bld", stamp="20261018-1200")
    helix = add_node(wf, "structure-builder", settings={"sequence": "GGCAUCG", "chirality": "L",
                                                        "scheme": "amber"})
    mut = add_node(wf, "structure-builder", settings={"mode": "mutate", "mutations": "A:2C, B:3G"})
    connect(wf, helix.id, "pdb", mut.id, "pdb")
    mapf = add_node(wf, "input-file", settings={"path": str(tmp_path / "x.mrc"), "kind": "map"})
    fit = add_node(wf, "structure-builder", settings={"mode": "fit", "resolution": 4.0, "search": "local"})
    connect(wf, mut.id, "pdb", fit.id, "pdb")
    connect(wf, mapf.id, "file", fit.id, "map")
    plan = Plan(wf, str(tmp_path / "plan"))
    errors = [(plan.steps[n].label, i.message) for n, i in plan.all_issues() if i.level == "error"]
    assert errors == []
    report = plan.write()
    scripts = sorted(p for p in report.written if p.endswith(".sh") and "_build_" in os.path.basename(p))
    assert len(scripts) == 3
    texts = [open(p).read() for p in scripts]
    assert "-m mdatk build helix --seq GGCAUCG --partner auto --chirality L" in texts[0]
    assert "--scheme amber" in texts[0] and "MDATK must be importable" in texts[0]
    assert "build mutate ../" in texts[1] and "'A:2C, B:3G'" in texts[1]
    assert "build fit ../" in texts[2] and "--resolution 4 --search local" in texts[2]
    for p in scripts:
        proc = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


# --- GUI --------------------------------------------------------------------------------------------

@pytest.fixture
def main_window(demo_dry, tmp_path):
    gi = pytest.importorskip("gi")
    try:
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
    except (ValueError, ImportError):  # pragma: no cover
        pytest.skip("GTK 3 introspection data not installed")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no display (run under xvfb-run)")
    if not Gtk.init_check([])[0]:  # pragma: no cover
        pytest.skip("cannot initialise GTK")
    from mdatk.config import Project
    from mdatk.gui.rnado import RNAdoWindow  # the structure builder lives in RNAdo

    app = Gtk.Application(application_id=None)
    app.register(None)
    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "gui_out")
    win = RNAdoWindow(app, p, demo_dry["preset"])
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    win.show_all()
    yield win
    win.destroy()


def test_builder_window_builds_and_writes_a_helix(main_window, tmp_path):
    from mdatk.projectfile import read_log

    main_window.open_builder()
    dlg = main_window.builder
    assert dlg is not None and dlg.page() == "helix"
    dlg.seq.set_text("GGCAUC")
    dlg.chirality.set_active_id("L")
    dlg.scheme.set_active_id("amber")
    out = tmp_path / "gui_helix.pdb"
    dlg.helix_out.entry.set_text(str(out))
    dlg.check_current()
    buf = dlg.report.get_buffer()
    assert "6 base pairs" in buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    dlg.run_current()
    s = pt.read_pdb(str(out))
    assert [r.name for r in s.residues()][:2] == ["LG5", "LG"] and len(s.segments) == 2
    assert pt.assign_chirality(s)[0].startswith("Chirality (sugar geometry): 12 L")
    entry = read_log(main_window.project_path)[-1]
    assert entry["step"] == "build helix" and str(out) in entry["files"]
    # the mutate page on the written helix, with the suggested output name
    dlg.notebook.set_current_page(1)
    assert dlg.page() == "mutate"
    dlg.mut_input.entry.set_text(str(out))
    assert dlg.mut_out.entry.get_text().endswith("gui_helix_mut.pdb")
    dlg.mutations.set_text("A:3G")
    dlg.run_current()
    names = [r.name for r in pt.read_pdb(dlg.mut_out.entry.get_text()).residues()]
    assert names[2] == "LG" and names[6 + 3] == "LC"  # A:3 C->G and its partner B:4 G->C
    dlg.partner_mode.set_active_id("none")
    assert not dlg.chain2.get_sensitive()
    dlg.destroy()
    assert main_window.builder is None


def test_negative_start_empty_partner_dna_partner_and_altlocs(tmp_path):
    import sys
    # a negative first residue number reaches argparse as one word
    out = tmp_path / "neg.pdb"
    proc = subprocess.run([sys.executable, "-m", "mdatk", "build", "helix", "--seq", "GGCA", "--partner", "none",
                           "--start=-5,1", "-o", str(out)], capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.returncode == 0, proc.stderr
    assert [r.resseq for r in pt.read_pdb(str(out)).residues()] == [-5, -4, -3, -2]
    from mdatk.workflows import get_tool
    tool = get_tool("structure-builder")
    from mdatk.workflows.base import StepContext
    ctx = StepContext("apt1", 1, "20261018-1200", "build")
    assert "--start=-5,1" in tool.command(tool.normalise({"sequence": "GGCA", "start1": -5}), ctx)
    # an explicit but empty partner is an error, not the Watson-Crick complement
    assert B.check_helix("GGCAU", "")[0][0] == "error"
    with pytest.raises(B.BuildError, match="type the partner sequence"):
        B.build_helix("GGCAU", "")
    # compensatory mutation next to a DNA partner: noted, not an abort
    text = B.build_helix("GGCAU")
    lines = []
    for ln in text.splitlines():
        if ln.startswith("ATOM") and ln[21] == "B":
            if ln[12:16].strip() == "O2'":
                continue
            name = ln[17:20].strip().rstrip("35")
            ln = ln[:17] + f"{'D' + name:>3}" + ln[20:]
        lines.append(ln)
    hybrid, notes = B.mutate("\n".join(lines) + "\n", "A:2C")
    assert any("is not an RNA nucleotide; no compensatory change" in n for n in notes)
    # a residue with alternate locations: one new base conformer at full occupancy
    alt = []
    for ln in text.splitlines():
        if ln.startswith("ATOM") and ln[21] == "A" and ln[22:26].strip() == "3":
            alt.append(ln[:16] + "A" + ln[17:54] + "  0.60" + ln[60:])
            alt.append(ln[:16] + "B" + ln[17:54] + "  0.40" + ln[60:])
        else:
            alt.append(ln)
    mutant, notes = B.mutate("\n".join(alt) + "\n", "A:3G", compensatory=False)
    base = [ln for ln in mutant.splitlines() if ln.startswith("ATOM") and ln[21] == "A"
            and ln[22:26].strip() == "3" and ln[12:16].strip() in ("N9", "C8", "O6")]
    assert len(base) == 3 and all(ln[16] == " " and ln[54:60].strip() == "1.00" for ln in base)
    assert any("alternate locations A,B" in n for n in notes)


def test_l_ccd_names_match_pdb_clean_and_5prime_phosphates():
    a = B.build_helix("GGCAUC", chirality="L", scheme="ccd", hydrogens=True)
    b = pt.clean(B.build_helix("GGCAUC", chirality="L", scheme="amber", hydrogens=True), scheme="ccd").text
    for x, y in zip(pt.read_pdb(a).residues(), pt.read_pdb(b).residues()):
        assert x.name == y.name
        cx, cy = x.coords(), y.coords()
        assert all(np.allclose(cx[n], cy[n], atol=2e-3) for n in cx if n in cy)
    names = [r.name for r in pt.read_pdb(B.build_helix("GGAC", chirality="L", scheme="ccd",
                                                         phosphate5=True)).residues()]
    assert names[0] == "0G" and "0G5" not in names  # a 5' end with its phosphate is the CCD component


# --- other pairings: parallel duplexes, (reverse) Hoogsteen pairs, triplexes -------------------------

JRD = os.path.join(USERFILES, "4JRD.pdb")  # poly(rA) parallel duplex (PDB 4JRD), A2-11 pair with B1-10 (HHt)

# donor-acceptor pairs of the intended pairs, written out independently of mdatk.builder:
# (strand-1 / duplex base, partner base) -> [(atom of the first, atom of the partner)]
HBONDS = {
    "reverse-wc": {"AU": [("N1", "N3"), ("N6", "O2")], "UA": [("N3", "N1"), ("O2", "N6")],
                   "GC": [("N1", "O2"), ("N2", "N3")], "CG": [("O2", "N1"), ("N3", "N2")]},
    "hoogsteen": {"AU": [("N7", "N3"), ("N6", "O4")], "GC": [("N7", "N3"), ("O6", "N4")]},
    "reverse-hoogsteen": {"AU": [("N7", "N3"), ("N6", "O2")], "AA": [("N7", "N6"), ("N6", "N1")],
                          "GG": [("N7", "N1"), ("O6", "N2")]},
    "hh": {"AA": [("N6", "N7"), ("N7", "N6")]},
    "wc": {"AU": [("N1", "N3"), ("N6", "O4")], "GC": [("N1", "N3"), ("N2", "O2"), ("O6", "N4")]},
}


def barnaba_pairs(text):
    """{(chain, resseq, chain, resseq): Leontis-Westhof code} from barnaba (via mdatk.annotate)."""
    pytest.importorskip("barnaba")
    from mdatk.annotate import annotate
    out = {}
    for rec in annotate(text, engine="barnaba", stacking=False).records:
        f = rec.split()
        if f[2] == "PAIR":
            out[(f[3], int(f[4]), f[6], int(f[7]))] = f[9]
    return out


def strand_pairs(s, parallel, strand=1):
    """[(residue of strand 1, partner residue)] at each level."""
    one, other = s.segments[0], s.segments[strand]
    n = len(one)
    return [(one[k], other[k] if parallel else other[n - 1 - k]) for k in range(n)]


def check_model(text, notes, pairs, table):
    """H-bonds 2.6-3.3 Å, O3'-P links 1.5-1.7 Å, no heavy-atom contacts < 2.4 Å (3+ bonds apart)."""
    s = pt.read_pdb(text)
    for a, b in pairs:
        ca, cb = a.coords(), b.coords()
        key = pt.parse_residue_name(a.name).base + pt.parse_residue_name(b.name).base
        for x, y in table[key]:
            d = np.linalg.norm(np.subtract(ca[x], cb[y]))
            assert 2.6 <= d <= 3.3, (a.label, b.label, x, y, d)
    d = o3p_distances(s)
    assert 1.5 <= min(d) and max(d) <= 1.7
    assert any(n.startswith("O3'-P links:") for n in notes)
    atoms = heavy_atoms(s)
    dist, topo = topology_distances(atoms, backbone_links(s, atoms))
    assert dist[topo >= 3].min() >= 2.4
    # the report checks hydrogens too (placed for the check in a heavy-atom model)
    assert any(n.startswith("No close contacts (") and "hydrogens included" in n for n in notes), notes
    return s


def test_hh_duplex_matches_4jrd():
    from mdatk.spatial import kabsch
    if not os.path.isfile(JRD):
        pytest.skip("PDB 4JRD not available")
    notes = []
    text = B.build_helix("A" * 10, pairing="hh", phosphate5=True, notes=notes)
    s = check_model(text, notes, strand_pairs(pt.read_pdb(text), True), HBONDS["hh"])
    ref = {(r.chain, r.resseq): r.coords() for r in pt.read_pdb(JRD).residues() if r.name.strip() == "A"}
    mine = {(r.chain, r.resseq): r.coords() for r in s.residues()}
    keys = [(("A", k), ("A", k + 1)) for k in range(1, 11)] + [(("B", k), ("B", k)) for k in range(1, 11)]
    for atoms, limit in ((["P"], 1.5), (["P", "C4'", "C1'", "N9", "N1"], 1.5)):
        x = np.array([mine[m][a] for m, r in keys for a in atoms if a in ref[r] and a in mine[m]])
        y = np.array([ref[r][a] for m, r in keys for a in atoms if a in ref[r] and a in mine[m]])
        rot, mc, rc = kabsch(x, y)
        rmsd = np.sqrt((((x - mc) @ rot.T + rc - y) ** 2).sum(axis=1).mean())
        print(f"4JRD {atoms}: RMSD {rmsd:.2f} A over {len(x)} atoms")
        assert rmsd < limit
    codes = barnaba_pairs(text)
    assert sorted(codes.values()) == ["HHt"] * 10
    assert set(codes) == {("A", k, "B", k) for k in range(1, 11)}
    assert "REMARK 951 NUCLEOTIDE OF PDB 4JRD" in text


@pytest.mark.parametrize("pairing, seq, code", [
    ("reverse-wc", "GAUCAGUAGC", "WWt"),
    ("hoogsteen", "AAGAGAAGGA", "HWc"),
    ("reverse-hoogsteen", "AAGAAGGAAG", "HWt"),
    ("hh", "AAAAAAAA", "HHt"),
])
def test_pairings_build_the_intended_pairs(pairing, seq, code):
    notes = []
    text = B.build_helix(seq, pairing=pairing, notes=notes)
    s = pt.read_pdb(text)
    pairs = strand_pairs(s, B.PARALLEL[pairing])
    check_model(text, notes, pairs, HBONDS[pairing])
    assert [r.name for r in s.segments[1]] == list(B.auto_partner(seq, pairing))
    assert any(n.startswith(f"Pairing {pairing}:") for n in notes)
    assert f"PAIRING {B.PAIRING_LW[pairing]}" in text and not any("mismatch" in n for n in notes)
    codes = barnaba_pairs(text)
    want = {(a.chain, a.resseq, b.chain, b.resseq) for a, b in pairs}
    assert set(codes) == want and set(codes.values()) == {code}


@pytest.mark.parametrize("motif, code", [("hoogsteen", "HWc"), ("reverse-hoogsteen", "HWt")])
def test_triplex_third_strand(motif, code):
    notes = []
    seq = "AAGAGAAGGA"
    text = B.build_helix(seq, triplex=motif, notes=notes)
    s = pt.read_pdb(text)
    assert s.chain_ids() == ["A", "B", "C"] and [len(seg) for seg in s.segments] == [10, 10, 10]
    third = "".join(r.name for r in s.segments[2])
    assert third == (B.resolve_third(seq, "auto", motif)[0])
    assert third == ("UUCUCUUCCU" if motif == "hoogsteen" else "AGGAAGAGAA")
    duplex = strand_pairs(s, False, 1)
    triples = strand_pairs(s, motif == "hoogsteen", 2)
    check_model(text, notes, duplex, HBONDS["wc"])
    check_model(text, notes, triples, HBONDS[motif])
    if motif == "hoogsteen":
        assert any("G·C+ Hoogsteen pair(s): the cytosine is written as plain C" in n for n in notes)
    assert any(n.startswith("Triplex, ") for n in notes) and "THIRD STRAND" in text
    codes = barnaba_pairs(text)  # skips without barnaba
    assert {codes[(a.chain, a.resseq, b.chain, b.resseq)] for a, b in duplex} == {"WCc"}
    assert {codes[(a.chain, a.resseq, b.chain, b.resseq)] for a, b in triples} == {code}


@pytest.mark.parametrize("kw", [dict(pairing="hoogsteen"), dict(pairing="reverse-wc"),
                                dict(triplex="reverse-hoogsteen")])
def test_l_pairings_are_mirror_images(kw):
    seq = "AAGAGA"
    d = pt.read_pdb(B.build_helix(seq, **kw))
    lh = pt.read_pdb(B.build_helix(seq, chirality="L", **kw))
    for rd, rl in zip(d.residues(), lh.residues()):
        cd, cl = rd.coords(), rl.coords()
        assert set(cd) == set(cl)
        for name, x in cd.items():
            assert np.allclose(np.multiply(x, [-1, 1, 1]), cl[name], atol=2e-3)
    n = len(list(lh.residues()))
    assert pt.assign_chirality(lh)[0] == f"Chirality (sugar geometry): {n} L and 0 D nucleotides"
    assert pt.assign_chirality(d)[0] == f"Chirality (sugar geometry): 0 L and {n} D nucleotides"
    assert {r.name for r in lh.residues()} <= {"0A", "0C", "0G", "0U"}


def test_pairing_mismatches_messages_and_errors():
    # mismatches are built in the reference frame and reported; strand-1 pyrimidines in Hoogsteen pairs
    msgs = B.check_helix("AAGA", "UUUU", pairing="hoogsteen")
    assert any(lvl == "warning" and "3:G·U" in t and "mismatched" in t for lvl, t in msgs)
    assert ("info", "Hoogsteen (cis WC/Hoogsteen: A·U, G·C+), parallel: 4 pairs, 3 canonical cHW, "
                    "1 mismatched") in msgs
    notes = []
    text = B.build_helix("AAGA", "UUUU", pairing="hoogsteen", notes=notes)
    assert any(n.startswith("WARNING: mismatch A:3 · B:3: no cHW pair") for n in notes)
    assert len(pt.read_pdb(text).segments) == 2
    msgs = B.check_helix("AUGA", pairing="hoogsteen")
    assert any(lvl == "warning" and "2:U·A" in t and "purine strand" in t for lvl, t in msgs)
    assert B.auto_partner("AAGC", "reverse-hoogsteen") == "GGUU"  # antiparallel: reversed
    assert B.auto_partner("AAGC", "hoogsteen") == "UUCG" and B.auto_partner("AGA", "hh") == "AGA"
    msgs = B.check_helix("AAGAG", triplex="hoogsteen", third="UUCUA")
    assert any(lvl == "warning" and "5:A·G" in t for lvl, t in msgs)  # third-strand A on strand-1 G
    msgs = B.check_helix("AAUAG", triplex="hoogsteen")
    assert any(lvl == "warning" and "3:U·U" in t for lvl, t in msgs)  # strand-1 pyrimidine
    notes = []
    B.build_helix("AAUAG", triplex="hoogsteen", notes=notes)
    assert any("a strand-1 pyrimidine has no purine Hoogsteen edge" in n for n in notes)
    for kw, match in ((dict(pairing="hh", triplex="hoogsteen"), "Watson-Crick duplex"),
                      (dict(partner="none", pairing="hoogsteen"), "single strand"),
                      (dict(triplex="hoogsteen", third="UU"), "same length"),
                      (dict(pairing="zipper"), "unknown pairing"),
                      (dict(triplex="tetraplex"), "unknown triplex"),
                      (dict(triplex="hoogsteen", chains="A,B,A"), "three strands")):
        with pytest.raises(B.BuildError, match=match):
            B.build_helix("AAGAG", **kw)
        assert B.check_helix("AAGAG", kw.get("partner", "auto"), kw.get("pairing", "wc"), kw.get("triplex"),
                             kw.get("third", "auto"))[0][0] == "error" or "chains" in kw
    s = pt.read_pdb(B.build_helix("AAGAG", triplex="hoogsteen", chains="P,Q", start="5,10"))
    assert s.chain_ids() == ["P", "Q", "C"] and [seg[0].resseq for seg in s.segments] == [5, 10, 1]
    assert B.helix_kind() == "A-form" and B.helix_kind("hh") == "parallel tHH"
    assert B.helix_kind("wc", "reverse-hoogsteen") == "reverse-hoogsteen triplex"


def test_pairing_cli_and_planner_step(tmp_path):
    from mdatk.cli import main
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import StepContext

    out = tmp_path / "hoog.pdb"
    assert main(["build", "helix", "--seq", "AAGAG", "--pairing", "hoogsteen", "--chirality", "L",
                 "-o", str(out)]) == 0
    s = pt.read_pdb(str(out))
    assert [r.name for r in s.segments[1]] == ["0U", "0U", "0C", "0U", "0C"]
    out3 = tmp_path / "tri.pdb"
    assert main(["build", "helix", "--seq", "AAGAG", "--triplex", "reverse-hoogsteen", "--third-strand", "GAGAA",
                 "--chains", "A,B,T", "--start=1,1,7", "-o", str(out3)]) == 0
    s = pt.read_pdb(str(out3))
    assert s.chain_ids() == ["A", "B", "T"] and [r.resseq for r in s.segments[2]][0] == 7
    assert main(["build", "helix", "--seq", "AAGAG", "--pairing", "hh", "--triplex", "hoogsteen",
                 "-o", str(tmp_path / "bad.pdb")]) == 1

    tool = TOOLS["structure-builder"]
    ctx = StepContext("p", 1, "s", tool.tag)

    def issues(**settings):
        return [(i.level, i.message) for i in tool.check(tool.normalise(settings), ctx)]

    s = tool.normalise({"sequence": "AAGAG", "pairing": "hoogsteen"})
    assert ("info", "no secondary-structure output: the helix restraints (dot-bracket) assume antiparallel "
                    "Watson-Crick pairs") in issues(sequence="AAGAG", pairing="hoogsteen")
    assert [p.name for p in tool.output_ports(s)] == ["pdb"] and tool.ss_model(s) is None
    assert "--pairing hoogsteen" in tool.command(s, ctx) and "(hoogsteen)" in tool.describe(s)
    assert not [m for lvl, m in issues(sequence="AAGAG", pairing="hh") if lvl == "error"]
    assert any(lvl == "warning" and "mismatched" in m for lvl, m in issues(sequence="AAGAG", pairing="hh"))
    s = tool.normalise({"sequence": "AAGAG", "triplex": "hoogsteen", "start1": -3})
    cmd = tool.command(s, ctx)
    assert "--triplex hoogsteen --third-strand auto" in cmd and "--start=-3,1,1" in cmd
    model = tool.ss_model(s)
    assert model is not None and model.split(model.dotbracket) == "((((( ))))) ....."
    assert [p.name for p in tool.output_ports(s)] == ["pdb", "secstruct"]
    assert ("error", "type the third-strand sequence (or choose the motif bases)") in issues(
        sequence="AAGAG", triplex="hoogsteen", third_mode="explicit")
    assert any(lvl == "warning" and "third-strand mismatches" in m
               for lvl, m in issues(sequence="AAGAG", triplex="hoogsteen", third_mode="explicit", third="UUCUA"))
    assert any(lvl == "error" and "three different chain ids" in m
               for lvl, m in issues(sequence="AAGAG", triplex="hoogsteen", chains="A,B,B"))
    # pairing / triplex options are ignored for a single strand
    s = tool.normalise({"sequence": "AAGAG", "partner_mode": "none", "pairing": "hh", "triplex": "hoogsteen"})
    assert "--pairing" not in tool.command(s, ctx) and "--triplex" not in tool.command(s, ctx)


# --- rebuilt helices with hydrogens: no close contacts, pairs kept (review findings 0.0.4) -------------

POLAR_H = {"H1", "H3", "H21", "H22", "H41", "H42", "H61", "H62", "HO2'", "HO5'", "HO3'"}
_BUILT = {}


def built(pairing, motif, seq, **kw):
    """build_helix text with hydrogens (cached: the rebuilt helices take seconds)."""
    key = (pairing, motif, seq, tuple(sorted(kw.items())))
    if key not in _BUILT:
        notes = []
        text = B.build_helix(seq, pairing=pairing, triplex=motif, hydrogens=True, notes=notes, **kw)
        _BUILT[key] = (text, notes)
    return _BUILT[key]


def dist(c1, a, c2, b):
    return float(np.linalg.norm(np.subtract(c1[a], c2[b])))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = (np.subtract(p0, p1), np.subtract(p2, p1), np.subtract(p3, p2))
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - b0 @ b1 * b1, b2 - b2 @ b1 * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def close_contacts(s):
    """Close contacts of a model with hydrogens, written independently of mdatk.builder.  Bonds by distance
    within residues (H: < 1.25 Å) and the O3'-P links; atoms four or more bonds apart: H-H < 2.0 Å, polar H -
    N/O < 1.6 Å, other H-heavy < 1.9 Å, heavy atoms < 2.8 Å (N/O pairs of two residues < 2.6 Å); heavy atoms
    three bonds apart < 2.4 Å."""
    from scipy.spatial import cKDTree

    atoms = [(k, r, a.name, np.array(a.xyz)) for k, r in enumerate(s.residues()) for a in r.atoms]
    xyz = np.array([a[3] for a in atoms])
    is_h = np.array([a[2][0] == "H" for a in atoms])
    nb = [set() for _ in atoms]
    tree = cKDTree(xyz)
    for i, j in tree.query_pairs(1.75):
        d = np.linalg.norm(xyz[i] - xyz[j])
        if atoms[i][0] == atoms[j][0] and not (is_h[i] and is_h[j]) and (d < 1.25 or not (is_h[i] or is_h[j])):
            nb[i].add(j)
            nb[j].add(i)
    for i, j in backbone_links(s, atoms):
        nb[i].add(j)
        nb[j].add(i)
    out = []
    for i, j in tree.query_pairs(2.8):
        two = set().union(*(nb[c] for c in nb[i])) if nb[i] else set()
        if j in nb[i] or j in two:
            continue
        three = set().union(*(nb[c] for c in two)) if two else set()
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        names = (atoms[i][2], atoms[j][2])
        if j in three:
            limit = 0.0 if is_h[i] or is_h[j] else 2.4
        elif is_h[i] and is_h[j]:
            limit = 2.0
        elif is_h[i] or is_h[j]:
            h, x = names if is_h[i] else names[::-1]
            limit = 1.6 if h in POLAR_H and x[0] in "NO" else 1.9
        else:
            limit = 2.6 if names[0][0] in "NO" and names[1][0] in "NO" and atoms[i][0] != atoms[j][0] else 2.8
        if d < limit:
            out.append(f"{atoms[i][1].label} {names[0]} - {atoms[j][1].label} {names[1]} {d:.2f}")
    return sorted(out)


# mixed sequences: other pairs than the reference pair, strand-1 pyrimidines, mismatches
MIXED = [("reverse-wc", None, "GAGGAUCA"), ("reverse-wc", None, "AUGCAU"), ("hoogsteen", None, "GAGGAUCA"),
         ("reverse-hoogsteen", None, "GAAGAG"), ("reverse-hoogsteen", None, "AUGCAU"), ("hh", None, "GAGGAUCA"),
         ("wc", "hoogsteen", "GAGGAUCA"), ("wc", "reverse-hoogsteen", "AUGCAU")]


@pytest.mark.parametrize("pairing, motif, seq", MIXED)
def test_rebuilt_helices_have_no_close_contacts(pairing, motif, seq):
    text, notes = built(pairing, motif, seq)
    s = pt.read_pdb(text)
    assert close_contacts(s) == []
    d = o3p_distances(s)
    assert 1.5 <= min(d) and max(d) <= 1.7
    report = [n for n in notes if "close contact" in n.lower()]
    assert len(report) == 1 and "hydrogens included" in report[0]
    assert not any(k in report[0] for k in ("(H-H <", "(H-heavy <", "(H-bond H <"))


@pytest.mark.parametrize("seq", ["UCUCAG", "GAGGAUCA", "AUGCAU"])
def test_reverse_wc_bases_clear_their_own_sugars(seq):
    """Finding 1: chi about -121 put every base on its own sugar (C6-C3' 2.99 Å, H6-H3' 1.48 Å)."""
    s = pt.read_pdb(built("reverse-wc", None, seq)[0])
    for r in s.residues():
        c = r.coords()
        purine = pt.parse_residue_name(r.name).base in "AG"
        ring, h, n, x = ("C8", "H8", "N9", "C4") if purine else ("C6", "H6", "N1", "C2")
        chi = dihedral(c["O4'"], c["C1'"], c[n], c[x])
        assert chi <= -135.0 or chi >= 170.0, (r.label, chi)  # anti, -190 .. -135 (was -121)
        assert dist(c, ring, c, "C3'") >= 3.1, r.label  # was 2.99 (C6) / 3.11 (C8)
        assert dist(c, h, c, "H3'") >= 2.0 and dist(c, h, c, "H2'") >= 2.0, r.label  # H6-H3' was 1.48
    notes = built("reverse-wc", None, seq)[1]
    chis = [float(m) for m in re.findall(r"chi (-?\d+)", next(n for n in notes if n.startswith("Backbone torsions")))]
    assert all(c <= -135 or c >= 170 for c in chis[1:])  # the first one is the 4JRD reference


@pytest.mark.parametrize("pairing, motif, seq", [("reverse-hoogsteen", None, "AAGAGG"),
                                                 ("reverse-hoogsteen", None, "GAAGAG"), ("wc", "hoogsteen", "GAGGAG")])
def test_sugar_steps_keep_c2_away_from_the_next_c5(pairing, motif, seq):
    """Finding 2: C2'(i)-C5'(i+1) 2.69 Å and H2'(i)-H5'(i+1) 1.03 Å at every reverse-Hoogsteen step."""
    s = pt.read_pdb(built(pairing, motif, seq)[0])
    for seg in s.segments:
        for a, b in zip(seg, seg[1:]):
            ca, cb = a.coords(), b.coords()
            assert dist(ca, "C2'", cb, "C5'") >= 3.1, (a.label, b.label)
            assert dist(ca, "O2'", cb, "C5'") >= 3.0, (a.label, b.label)
            for hb in ("H5'", "H5''"):
                assert dist(ca, "H2'", cb, hb) >= 2.0 and dist(ca, "C2'", cb, hb) >= 2.4, (a.label, b.label, hb)


@pytest.mark.parametrize("pairing, motif, seq", [("reverse-wc", None, "AUGCAU"), ("reverse-wc", None, "GAGGAUCA"),
                                                 ("hoogsteen", None, "GAGGAUCA"),
                                                 ("reverse-hoogsteen", None, "GAGGAUCA"),
                                                 ("wc", "hoogsteen", "GAGGAUCA"),
                                                 ("wc", "reverse-hoogsteen", "GAAGAG")])
def test_closed_strands_keep_the_pair_geometry(pairing, motif, seq):
    """Finding 3: closing a partner strand turned reverse-WC C·G into amine-amine (H41-H21 1.18 Å) and
    acceptor-acceptor (O2-O6 2.80 Å) contacts.  Every canonical pair keeps donor-acceptor 2.8-3.0 Å, the
    donor hydrogen on the line (H...acceptor <= 2.05 Å, angle >= 150 degrees) and coplanar bases."""
    s = pt.read_pdb(built(pairing, motif, seq)[0])
    groups = ([(strand_pairs(s, B.PARALLEL[pairing]), HBONDS[pairing])] if not motif else
              [(strand_pairs(s, False, 1), HBONDS["wc"]), (strand_pairs(s, motif == "hoogsteen", 2), HBONDS[motif])])
    checked = 0
    for pairs, table in groups:
        for a, b in pairs:
            key = pt.parse_residue_name(a.name).base + pt.parse_residue_name(b.name).base
            if key not in table:
                continue
            ca, cb = a.coords(), b.coords()
            for x, y in table[key]:
                assert 2.8 <= dist(ca, x, cb, y) <= 3.0, (a.label, b.label, x, y)
                for don, acc, cd, cc in ((x, y, ca, cb), (y, x, cb, ca)):
                    hs = [h for h in cd if h.startswith("H") and dist(cd, h, cd, don) < 1.1]
                    if hs and acc[0] in "NO" and don[0] == "N":
                        h = min(hs, key=lambda k: dist(cd, k, cc, acc))
                        if dist(cd, h, cc, acc) < 2.6:
                            u, v = np.subtract(cd[don], cd[h]), np.subtract(cc[acc], cd[h])
                            angle = np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
                            assert dist(cd, h, cc, acc) <= 2.05 and angle >= 150, (a.label, b.label, h, acc)
            normals = []
            for c, base in ((ca, key[0]), (cb, key[1])):
                ring = np.array([c[n] for n in B.BASE_NAMES[base]])
                normals.append(np.linalg.svd(ring - ring.mean(axis=0))[2][2])
            assert np.degrees(np.arccos(min(1.0, abs(normals[0] @ normals[1])))) <= 10.0, (a.label, b.label)
            checked += 1
    assert checked >= 5
    # no amine-amine / acceptor-acceptor or hydrogen contacts between the paired bases
    assert close_contacts(s) == []


def test_contact_report_includes_hydrogens_and_close_heavy_atoms():
    """Finding 4: the report checked heavy atoms below 2.4 Å only.  The Arnott A-form has H2'(i)-H6/H8(i+1)
    of about 1.6 Å, which the report now lists; a C2'(i)-C5'(i+1) contact of 2.7 Å is listed too."""
    s = pt.read_pdb(B.build_helix("GGCAUC", partner="none", hydrogens=True))
    residues = list(s.residues())
    labels = [r.label for r in residues]
    bases = [pt.parse_residue_name(r.name).base for r in residues]
    res_atoms = [[(a.name, np.array(a.xyz)) for a in r.atoms] for r in residues]
    segments = [list(range(len(residues)))]
    found = B._close_contacts(res_atoms, segments, bases)
    assert any({a[1], b[1]} in ({"H2'", "H6"}, {"H2'", "H8"}) and what.startswith("H-H") and d < 2.0
               for d, what, a, b in found)
    note = B._geometry_notes(labels, res_atoms, segments, [], bases)[-1]
    assert note.startswith("WARNING:") and "close contact(s)" in note and "(H-H < 2.0)" in note
    # move residue 2 (rigidly) so that its C5' is 2.7 Å from C2' of residue 1: reported (heavy < 3.0)
    c1, c2 = dict(res_atoms[0]), dict(res_atoms[1])
    u = c1["C2'"] - c2["C5'"]
    shift = u * (1.0 - 2.7 / np.linalg.norm(u))
    res_atoms[1] = [(n, x + shift) for n, x in res_atoms[1]]
    found = B._close_contacts(res_atoms, segments, bases)
    assert any((a[1], b[1]) == ("C2'", "C5'") and what.startswith("heavy") and abs(d - 2.7) < 1e-6
               for d, what, a, b in found)
    # a heavy-atom model gets hydrogens for the check: the same report as with hydrogens
    for pairing, motif, seq in MIXED[:2]:
        report = [n for n in built(pairing, motif, seq)[1] if "close contact" in n.lower()]
        notes = []
        B.build_helix(seq, pairing=pairing, triplex=motif, notes=notes)
        assert [n for n in notes if "close contact" in n.lower()] == report


def tleap_executable():
    home = os.environ.get("AMBERHOME")
    for exe in ([os.path.join(home, "bin", "tleap")] if home else []) + [shutil.which("tleap")]:
        if exe and os.path.isfile(exe):
            return exe
    return None


@pytest.mark.parametrize("pairing, motif, seq", MIXED + [("hh", None, "AAAAAAAA"), ("reverse-wc", None, "UCUCAG"),
                                                         ("reverse-hoogsteen", None, "AAGAGG")])
def test_tleap_finds_no_close_contacts(tmp_path, pairing, motif, seq):
    """tleap (leaprc.RNA.OL3, check) on the Amber-named, hydrogenated model: no 'Close contact' (< 1.5 Å)."""
    exe = tleap_executable()
    if exe is None:
        pytest.skip("tleap (AmberTools) not available")
    (tmp_path / "m.pdb").write_text(built(pairing, motif, seq, scheme="amber")[0])
    (tmp_path / "leap.in").write_text("source leaprc.RNA.OL3\nm = loadpdb m.pdb\ncheck m\nquit\n")
    env = dict(os.environ)
    env.setdefault("AMBERHOME", os.path.dirname(os.path.dirname(exe)))
    out = subprocess.run([exe, "-f", "leap.in"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=600).stdout
    assert "Checking 'm'" in out, out[-2000:]
    assert "Close contact" not in out and "Created a new atom" not in out, out[-3000:]


def wait_for_builder(dlg, timeout=300):
    """Other pairings build in a background worker: run the GTK main loop until it is done."""
    import time

    from gi.repository import Gtk

    t0 = time.time()
    while getattr(dlg, "worker", None) is not None and time.time() - t0 < timeout:
        Gtk.main_iteration_do(False)
        time.sleep(0.005)
    assert getattr(dlg, "worker", None) is None


def test_builder_window_pairings(main_window, tmp_path):
    main_window.open_builder()
    dlg = main_window.builder
    dlg.seq.set_text("AAGAGA")
    dlg.pairing.set_active_id("hh")
    assert not dlg.triplex.get_sensitive() and not dlg.third.get_sensitive()
    out = tmp_path / "gui_hh.pdb"
    dlg.helix_out.entry.set_text(str(out))
    dlg.check_current()
    buf = dlg.report.get_buffer()
    assert "2 mismatched" in buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    dlg.run_current()
    wait_for_builder(dlg)
    s = pt.read_pdb(str(out))
    assert len(s.segments) == 2 and [r.name for r in s.segments[1]] == list("AAGAGA")
    assert "tHH" in buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    dlg.pairing.set_active_id("wc")
    dlg.triplex.set_active_id("reverse-hoogsteen")
    assert dlg.triplex.get_sensitive() and dlg.third.get_sensitive() and dlg.chain3.get_sensitive()
    dlg.chain3.set_text("T")
    out3 = tmp_path / "gui_triplex.pdb"
    dlg.helix_out.entry.set_text(str(out3))
    dlg.run_current()
    wait_for_builder(dlg)
    s = pt.read_pdb(str(out3))
    assert s.chain_ids() == ["A", "B", "T"] and "".join(r.name for r in s.segments[2]) == "AGAGAA"
    dlg.partner_mode.set_active_id("none")
    assert not dlg.pairing.get_sensitive() and not dlg.triplex.get_sensitive()
    dlg.destroy()
