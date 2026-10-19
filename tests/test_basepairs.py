"""Tests for the builtin base-pair detection and the 'basepairs' analysis."""

import sys
import types

import numpy as np
import pytest

from conftest import make_universe, micro_project, run_one
from test_pdbtools import BACKBONE, BASES
from mdatk.data import Dataset
from mdatk.secstruct import OTHER, WATSON_CRICK, WOBBLE, PairDetector

# H-bond atoms of idealised cis Watson-Crick-edge pairs (first base, partner)
PAIR_ATOMS = {("G", "C"): [("N1", "N3"), ("N2", "O2"), ("O6", "N4")],
              ("A", "U"): [("N6", "O4"), ("N1", "N3")],
              ("G", "U"): [("O6", "N3"), ("N1", "O2")]}


# --- geometry helpers ----------------------------------------------------------------------------------

def nucleotide(base, shift=(0.0, 0.0, 0.0)):
    """Atom name -> coordinates of an AmberTools RNA.lib nucleotide (heavy atoms, see test_pdbtools)."""
    return {n: np.array(xyz) + shift for n, *xyz in BACKBONE[:9] + BASES[base] + BACKBONE[9:]}


def _kabsch(src, dst, w):
    w = w[:, None] / w.sum()
    cs, cd = (src * w).sum(0), (dst * w).sum(0)
    u, _, vt = np.linalg.svd(((src - cs) * w).T @ (dst - cd))
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, cs, cd


def _plane(res, base):
    p = np.array([res[n] for n, *_ in BASES[base]])
    return p.mean(0), np.linalg.svd(p - p.mean(0))[2][2]


def partner(first, fbase, sbase, d=2.9, atoms=None):
    """Nucleotide ``sbase`` placed coplanar with ``first`` so that the H-bond atoms of the pair are
    about ``d`` apart (weighted rigid fit of the partner's H-bond atoms onto targets d away from the
    first base's Watson-Crick edge, plus its ring centre and plane normal)."""
    atoms = atoms or PAIR_ATOMS[(fbase, sbase)]
    second = nucleotide(sbase)
    a = np.array([first[x] for x, _ in atoms])
    b = np.array([second[y] for _, y in atoms])
    ring1, n1 = _plane(first, fbase)
    ring2, n2 = _plane(second, sbase)
    edge = (a[-1] - a[0]) / np.linalg.norm(a[-1] - a[0])
    out = np.cross(n1, edge)
    out /= np.linalg.norm(out)
    if np.dot(out, a.mean(0) - ring1) < 0:
        out = -out
    targets = a + d * out
    centre = targets.mean(0) + out * np.linalg.norm(b.mean(0) - ring2)
    best = None
    for sign in (1.0, -1.0):
        src = np.vstack([b, ring2, ring2 + n2])
        dst = np.vstack([targets, centre, centre + sign * n1])
        r, cs, cd = _kabsch(src, dst, np.array([10.0] * len(b) + [1.0, 1.0]))
        moved = {k: (v - cs) @ r.T + cd for k, v in second.items()}
        err = sum(abs(np.linalg.norm(first[x] - moved[y]) - d) for x, y in atoms)
        if best is None or err < best[0]:
            best = (err, moved)
    return best[1]


def hbond_distances(first, second, fbase, sbase):
    return [float(np.linalg.norm(first[x] - second[y])) for x, y in PAIR_ATOMS[(fbase, sbase)]]


def detector_input(residues):
    """[(resname, {atom: xyz}), ...] -> (resnames, atom_index, xyz) for PairDetector."""
    names, index, rows = [], [], []
    for resname, atoms in residues:
        names.append(resname)
        index.append({n: len(rows) + k for k, n in enumerate(atoms)})
        rows.extend(atoms.values())
    return names, index, np.array(rows)


def universe(frames):
    """frames: list of [(resname, {atom: xyz}), ...] with identical residues/atoms -> Universe."""
    atoms = [dict(name=n, resid=r + 1, resname=resname, element=n.lstrip("0123456789")[0])
             for r, (resname, res) in enumerate(frames[0]) for n in res]
    coords = [[xyz for _, res in f for xyz in res.values()] for f in frames]
    return make_universe(atoms, coords)


def moved(res, shift):
    return {k: v + np.asarray(shift) for k, v in res.items()}


# --- detector ------------------------------------------------------------------------------------------

def test_helper_builds_realistic_pairs():
    for (p, q) in PAIR_ATOMS:
        first = nucleotide(p)
        second = partner(first, p, q)
        assert all(2.7 < d < 3.1 for d in hbond_distances(first, second, p, q))
        assert 9.0 < np.linalg.norm(first["C1'"] - second["C1'"]) < 11.5  # C1'-C1' of a real pair ~10.5
        clash = min(np.linalg.norm(u - v) for u in first.values() for v in second.values())
        assert clash > 2.6


@pytest.mark.parametrize("p, q, kind, names", [
    ("G", "C", WATSON_CRICK, ("G", "C")), ("A", "U", WATSON_CRICK, ("A5", "U3")),
    ("G", "U", WOBBLE, ("RG", "RU")), ("G", "C", WATSON_CRICK, ("0G", "LC5")),
    ("A", "U", WATSON_CRICK, ("LA", "0U")), ("A", "U", WATSON_CRICK, ("DA", "DT")),
    ("G", "C", WATSON_CRICK, ("DG5", "DC3")),
])
def test_canonical_pairs_detected(p, q, kind, names):
    first = nucleotide(p)
    names_, index, xyz = detector_input([(names[0], first), (names[1], partner(first, p, q))])
    assert PairDetector(names_, index).detect(xyz).tolist() == [[0, 1, kind]]


def test_wobble_optional_and_cutoff():
    g = nucleotide("G")
    u = partner(g, "G", "U")
    names, index, xyz = detector_input([("G", g), ("U", u)])
    assert PairDetector(names, index, wobble=False).detect(xyz).shape == (0, 3)
    far = partner(g, "G", "U", d=3.8)
    names, index, xyz = detector_input([("G", g), ("U", far)])
    assert PairDetector(names, index, cutoff=3.5).detect(xyz).shape == (0, 3)
    assert PairDetector(names, index, cutoff=4.0).detect(xyz).tolist() == [[0, 1, WOBBLE]]


def test_stacked_neighbours_are_not_pairs():
    """Close H-bond atoms with the C1' atoms ~5.5 A apart (a stacked GpC step) are no pair."""
    g = nucleotide("G")
    c = partner(g, "G", "C")
    c["C1'"] = g["C1'"] + np.array([5.5, 0.0, 0.0])
    names, index, xyz = detector_input([("G", g), ("C", c)])
    assert PairDetector(names, index).detect(xyz).shape == (0, 3)
    del c["C1'"]  # without C1' the window is not checked
    names, index, xyz = detector_input([("G", g), ("C", c)])
    assert PairDetector(names, index).detect(xyz).tolist() == [[0, 1, WATSON_CRICK]]


def test_score_ties_go_to_the_ideal_c1_distance():
    """Two candidates with equal H-bond scores: the one with C1'-C1' nearer 10.5 A wins."""
    det = PairDetector(["G", "C", "C"], [{}, {}, {}])
    cand = np.array([[0, 1, WATSON_CRICK, 5.63, 1.3], [0, 2, WATSON_CRICK, 5.63, 0.0]])
    det.candidates = lambda xyz: cand
    assert det.detect(np.zeros((0, 3))).tolist() == [[0, 2, WATSON_CRICK]]


def test_gc_needs_two_of_three_bonds():
    g = nucleotide("G")
    c = partner(g, "G", "C")
    names, index, xyz = detector_input([("G", g), ("C", c)])
    det = PairDetector(names, index)
    one_off = xyz.copy()
    one_off[index[1]["O2"]] += 5.0 * np.array([0.0, 0.0, 1.0])  # break the N2-O2 bond only
    assert det.detect(one_off).tolist() == [[0, 1, WATSON_CRICK]]
    two_off = one_off.copy()
    two_off[index[1]["N4"]] += 5.0 * np.array([0.0, 0.0, 1.0])
    assert det.detect(two_off).shape == (0, 3)


def test_conflicts_resolved_by_distance():
    g = nucleotide("G")
    close = partner(g, "G", "C", d=2.9)
    loose = partner(g, "G", "C", d=3.3)
    names, index, xyz = detector_input([("C", loose), ("G", g), ("C", close)])
    assert PairDetector(names, index).detect(xyz).tolist() == [[1, 2, WATSON_CRICK]]
    # the losing C is free to pair with another G
    g2 = nucleotide("G", shift=(0.0, 0.0, 30.0))
    names, index, xyz = detector_input([("C", loose), ("G", g), ("C", close), ("G", g2),
                                        ("C", partner(g2, "G", "C"))])
    assert PairDetector(names, index).detect(xyz).tolist() == [[1, 2, 1], [3, 4, 1]]


def test_non_nucleotides_never_pair():
    g = nucleotide("G")
    c = partner(g, "G", "C")
    for name in ("LIG", "ALA", "MG", "X"):
        names, index, xyz = detector_input([("G", g), (name, c)])
        assert PairDetector(names, index).detect(xyz).shape == (0, 3)
    # a lanthanum ion named LA has no base atoms
    names, index, xyz = detector_input([("A", nucleotide("A")), ("LA", {"LA": np.zeros(3)})])
    assert PairDetector(names, index).detect(xyz).shape == (0, 3)


# --- analysis ------------------------------------------------------------------------------------------

def duplex_frames():
    """G1 A2 U3 C4 with G1-C4 and A2-U3 pairs; C4 leaves in frames 1-2, returns in 3-4."""
    g = nucleotide("G")
    c = partner(g, "G", "C")
    a = nucleotide("A", shift=(0.0, 0.0, 25.0))
    u = partner(a, "A", "U")
    frames = []
    for k in range(5):
        c_k = moved(c, (0.0, 0.0, -20.0)) if k in (1, 2) else c
        frames.append([("G5", g), ("A", a), ("U", u), ("C3", c_k)])
    return frames


def test_analysis_rows_align_with_trace_and_stride():
    u = universe(duplex_frames())
    p = micro_project({"RNA": "1-4"})
    p.analyses.basepairs.stride = 1
    res = run_one("basepairs", u, p)
    trace = run_one("trace", u, p)
    assert res.meta["residue_labels"] == trace.meta["residue_labels"]
    np.testing.assert_array_equal(res.arrays["resindex"], trace.arrays["resindex"])
    np.testing.assert_array_equal(res.arrays["count"], [2, 1, 1, 2, 2])
    ds = Dataset(np.arange(5), np.arange(5.0), {"basepairs": res})
    pm = ds.pairmap("basepairs/map")
    assert pm.shape == (4, 4)
    dense = pm.dense(0)
    assert dense[0, 3] == WATSON_CRICK and dense[1, 2] == WATSON_CRICK and dense.sum() == 2  # i < j only
    assert pm.dense(1)[0, 3] == 0
    assert ds.series("basepairs/count").values.tolist() == [2, 1, 1, 2, 2]
    assert res.meta["engine"] == "builtin"

    p.analyses.basepairs.stride = 2  # frames 1 and 3 reuse frames 0 and 2
    res = run_one("basepairs", u, p)
    np.testing.assert_array_equal(res.arrays["count"], [2, 2, 1, 1, 2])


def test_analysis_uses_structure_residues_only():
    frames = [f + [("ALA", {"N1": np.array([0.0, 0.0, 50.0]), "N3": np.array([1.0, 0.0, 50.0])})]
              for f in duplex_frames()]
    u = universe(frames)
    p = micro_project({"RNA": "1-3", "Other": "5"})
    p.visual.structure_groups = ["RNA", "Other"]
    res = run_one("basepairs", u, p)
    assert res.meta["residue_labels"] == ["G51", "A2", "U3", "ALA5"]
    assert res.arrays["count"].tolist() == [1, 1, 1, 1, 1]  # C4 is not shown, so G1 is unpaired
    from mdatk.render.panels import required_analyses
    from mdatk.config import PanelSpec
    assert "basepairs" in required_analyses(p, [PanelSpec(type="contact_map", series=["basepairs/map"])])


def test_l_rna_residues_pair():
    g = nucleotide("G")
    c = partner(g, "G", "C")
    u = universe([[("LG5", g), ("0C", c)]] * 2)
    res = run_one("basepairs", u, micro_project({"RNA": "1-2"}))
    assert res.arrays["count"].tolist() == [1, 1]


# --- barnaba engine (fake module with barnaba.annotate's return structure) ---------------------------

def fake_barnaba(monkeypatch, skip=(), extra_code="WHt"):
    """Install a stand-in for barnaba: residues are read from the PDB like mdtraj/barnaba do
    (labels NAME_RESSEQ_CHAININDEX, residues in ``skip`` left out as barnaba skips unknown residues);
    every model pairs residue 0 with the last as WCc and residues 1-2 with ``extra_code``, plus an
    'XXX' entry that barnaba emits for close bases that do not pair."""
    calls = []

    def annotate(path):
        seq, chains, models = [], [], 0
        with open(path) as fh:
            for line in fh:
                if line.startswith("MODEL"):
                    models += 1
                if line.startswith("ATOM") and models == 1:
                    name, chain, resseq = line[17:20].strip(), line[21], int(line[22:26])
                    if chain not in chains:
                        chains.append(chain)
                    label = f"{name}_{resseq}_{chains.index(chain)}"
                    if label not in seq and name not in skip:
                        seq.append(label)
        calls.append((models, list(seq)))
        n = len(seq)
        pairs = [[[np.int64(0), np.int64(n - 1)], [np.int64(1), np.int64(2)], [np.int64(0), np.int64(1)]],
                 ["WCc", extra_code, "XXX"]]
        stacks = [[[np.int64(1), np.int64(2)]], [">>"]]
        return [stacks] * models, [pairs] * models, seq

    monkeypatch.setitem(sys.modules, "barnaba", types.SimpleNamespace(annotate=annotate))
    return calls


@pytest.mark.parametrize("noncanonical", [False, True])
def test_barnaba_engine_mapping(monkeypatch, noncanonical):
    calls = fake_barnaba(monkeypatch)
    frames = duplex_frames()
    frames = [[("ALA", {"CA": np.array([0.0, 0.0, 60.0])})] + f for f in frames]  # row 0: not a nucleotide
    u = universe(frames)
    p = micro_project({"All": "1-5"})
    p.analyses.basepairs.engine = "barnaba"
    p.analyses.basepairs.noncanonical = noncanonical
    p.analyses.basepairs.stride = 2
    res = run_one("basepairs", u, p)
    assert calls[0][0] == 3  # frames 0, 2, 4 in one multi-model file
    assert calls[0][1] == ["G_1_0", "A_2_0", "U_3_0", "C_4_0"]  # canonical names, ALA left out
    first = res.meta["lw_first_frame"]
    expected = [[1, 4, "WCc"], [2, 3, "WHt"]] if noncanonical else [[1, 4, "WCc"]]
    assert first == expected
    ds = Dataset(np.arange(5), np.arange(5.0), {"basepairs": res})
    dense = ds.pairmap("basepairs/map").dense(1)
    assert dense[1, 4] == WATSON_CRICK
    assert dense[2, 3] == (OTHER if noncanonical else 0)
    assert res.meta["engine"] == "barnaba"


def test_barnaba_mapping_survives_skipped_residues(monkeypatch):
    fake_barnaba(monkeypatch, skip=("A",), extra_code="GUc")  # barnaba drops A2 (e.g. missing atoms)
    u = universe(duplex_frames())
    p = micro_project({"RNA": "1-4"})
    p.analyses.basepairs.engine = "barnaba"
    res = run_one("basepairs", u, p)
    # barnaba indices 0..2 are G1, U3, C4: WCc G1-C4 and GUc U3-C4 map to rows (0, 3) and (2, 3)
    assert res.meta["lw_first_frame"] == [[0, 3, "WCc"], [2, 3, "GUc"]]


def test_barnaba_missing_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "barnaba", None)  # import fails
    u = universe(duplex_frames())
    p = micro_project({"RNA": "1-4"})
    p.analyses.basepairs.engine = "barnaba"
    with pytest.raises(RuntimeError, match="pip install barnaba"):
        run_one("basepairs", u, p)


def test_real_barnaba_on_ideal_pairs():
    pytest.importorskip("barnaba")
    from mdatk.secstruct import BarnabaRunner
    residues = duplex_frames()[0]
    names, index, xyz = detector_input(residues)
    runner = BarnabaRunner(names, [list(i.items()) for i in index])
    (frame,) = runner.annotate([xyz], stacking=True)
    assert [(i, j) for i, j, c in frame.pairs if c == "WCc"] == [(0, 3), (1, 2)]
