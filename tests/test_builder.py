"""Tests of the starting-structure builder (mdatk.builder): ideal A-form helices, mutations, density fits,
the 'mdatk build' command, the planner step and the GUI window."""

import os
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


@pytest.mark.skipif(not os.path.isfile(HELICES), reason="user model not available")
def test_backbone_matches_an_experimental_a_form_helix():
    from mdatk.spatial import kabsch
    ref = {(r.chain, r.resseq): r.coords() for r in pt.read_pdb(HELICES).residues()}

    def p_rmsd(seq, chirality, keys):
        mine = [r.coords() for r in pt.read_pdb(B.build_helix(seq, chirality=chirality,
                                                              phosphate5=True)).residues()]
        x = np.array([m["P"] for m, k in zip(mine, keys) if "P" in ref[k]])
        y = np.array([ref[k]["P"] for m, k in zip(mine, keys) if "P" in ref[k]])
        r, mc, rc = kabsch(x, y)
        return np.sqrt((((x - mc) @ r.T + rc - y) ** 2).sum(axis=1).mean()), len(x)

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
    from mdatk.gui.app import MainWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "gui_out")
    win = MainWindow(app, p, demo_dry["preset"])
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
