import numpy as np
import pytest

from conftest import make_universe, micro_project, run_one
from mdatk.analysis.energy import COULOMB, GENERIC_LJ
from mdatk.io.parm7 import LJTable
from mdatk.spatial import kabsch, minimum_image_vectors, pairs_within


def atom(name, resid, resname, element, **kw):
    return dict(name=name, resid=resid, resname=resname, element=element, **kw)


def test_pairs_within_matches_brute_force_and_pbc():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 20, (60, 3))
    b = rng.uniform(0, 20, (50, 3))
    i, j, d = pairs_within(a, b, 4.0)
    dist = np.linalg.norm(a[:, None] - b[None], axis=2)
    assert set(zip(i.tolist(), j.tolist())) == set(zip(*np.nonzero(dist <= 4.0)))
    np.testing.assert_allclose(d, dist[i, j])
    # periodic: two atoms across the boundary are 1 A apart
    box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    i, j, d = pairs_within(np.array([[0.2, 5, 5]]), np.array([[9.2, 5, 5]]), 2.0, box)
    assert len(i) == 1 and d[0] == pytest.approx(1.0, abs=1e-4)


def test_minimum_image_vectors_in_a_triclinic_box():
    """A truncated octahedron (Amber's usual box): the pair that pairs_within finds across the boundary has the
    vector of its distance, as in an orthorhombic box, and an H-bond to an acceptor in the next cell is counted."""
    from MDAnalysis.lib.mdamath import triclinic_vectors

    box = np.array([40.0, 40.0, 40.0, 109.4712190, 109.4712190, 109.4712190])
    cell = triclinic_vectors(box).astype(np.float64)
    a = np.array([[1.0, 1.0, 1.0]])
    for shift in (cell[1], cell[2], -cell[0] + cell[2], cell[0] - cell[1] - cell[2]):
        b = a + np.array([2.0, 0.0, 0.0]) + shift
        i, j, d = pairs_within(a, b, 3.0, box)
        assert len(i) == 1 and d[0] == pytest.approx(2.0, abs=1e-4)
        v = minimum_image_vectors(b[j] - a[i], box)
        np.testing.assert_allclose(v, [[2.0, 0.0, 0.0]], atol=1e-4)
    rng = np.random.default_rng(3)
    v = rng.uniform(-100.0, 100.0, (200, 3))
    w = minimum_image_vectors(v, box)
    steps = range(-7, 8)
    images = np.array([[i, j, k] for i in steps for j in steps for k in steps]) @ cell
    shortest = np.linalg.norm(v[:, None, :] + images[None], axis=2).min(axis=1)
    np.testing.assert_allclose(np.linalg.norm(w, axis=1), shortest, atol=1e-4)
    frac = (w - v) @ np.linalg.inv(cell)
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-4)  # moved by whole box vectors only
    assert minimum_image_vectors(np.zeros((0, 3)), box).shape == (0, 3)
    ortho = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])  # as before
    np.testing.assert_allclose(minimum_image_vectors(np.array([[9.0, 0.5, -6.0]]), ortho), [[-1.0, 0.5, 4.0]])
    assert minimum_image_vectors(v, None) is v
    # the hydrogen-bond angle of a pair across the boundary
    atoms = [atom("N3", 1, "U", "N"), atom("H3", 1, "U", "H"), atom("O6", 2, "G", "O")]
    frame = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.9, 1.0, 1.0]])
    frame[2] += cell[1]
    u = make_universe(atoms, [frame], bonds=[(0, 1)])
    for ts in u.trajectory:
        ts.dimensions = box
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.use_pbc = True
    np.testing.assert_array_equal(run_one("hbonds", u, p).arrays["count"], [1])


OCTAHEDRON = np.array([30.0, 30.0, 30.0, 109.4712190, 109.4712190, 109.4712190], dtype=np.float32)


def test_minimum_image_vectors_in_a_triclinic_box_are_the_shortest_images():
    """Amber's truncated octahedron is a triclinic cell: the vectors of pairs found across a box face
    were returned as they are (only orthorhombic boxes were imaged)."""
    from MDAnalysis.lib.mdamath import triclinic_vectors

    cell = triclinic_vectors(OCTAHEDRON).astype(np.float64)
    rng = np.random.default_rng(4)
    v = rng.uniform(-60, 60, (300, 3))
    out = minimum_image_vectors(v, OCTAHEDRON)
    shifts = np.array([(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)]) @ cell
    brute = np.linalg.norm(v[:, None, :] + shifts[None], axis=2).min(axis=1)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), brute, atol=1e-9)
    frac = (out - v) @ np.linalg.inv(cell)  # each result is the vector moved by whole cell vectors
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)
    # the length is the distance pairs_within reports for the pair
    a, b = np.array([[1.0, 1.0, 1.0]]), np.array([[1.0, 1.0, 1.0]]) + cell[2] + [2.0, 0.0, 0.0]
    i, j, d = pairs_within(a, b, 3.0, OCTAHEDRON)
    assert len(i) == 1 and d[0] == pytest.approx(2.0, abs=1e-4)
    np.testing.assert_allclose(minimum_image_vectors(b - a, OCTAHEDRON), [[2.0, 0.0, 0.0]], atol=1e-4)
    # orthorhombic boxes and no box: as before
    box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    np.testing.assert_allclose(minimum_image_vectors(np.array([[9.0, -6.0, 4.0]]), box), [[-1.0, 4.0, 4.0]])
    np.testing.assert_allclose(minimum_image_vectors(np.array([[9.0, -6.0, 4.0]]), None), [[9.0, -6.0, 4.0]])


def _in_box(u, box):
    u.trajectory.dimensions_array[:] = box
    u.trajectory[0]
    return u


def test_hbond_angle_and_mg_geometry_across_a_triclinic_box_face():
    from MDAnalysis.lib.mdamath import triclinic_vectors

    cell = triclinic_vectors(OCTAHEDRON).astype(np.float64)
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.use_pbc = True
    # a linear N-H...O bond (2.9 A), the acceptor stored as its image one cell vector away
    atoms = [atom("N3", 1, "U", "N"), atom("H3", 1, "U", "H"), atom("O6", 2, "G", "O")]
    xyz = np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [7.9, 5.0, 5.0]])
    for shift in (np.zeros(3), cell[1], cell[2], -cell[1] - cell[2]):
        moved = xyz.copy()
        moved[2] += shift
        u = _in_box(make_universe(atoms, [moved], bonds=[(0, 1)]), OCTAHEDRON)
        np.testing.assert_array_equal(run_one("hbonds", u, p).arrays["count"], [1])
    # an octahedral (90 degree) O-Mg-O bridge, one oxygen stored as its image
    atoms = [atom("OP1", 1, "A", "O"), atom("OP2", 2, "U", "O"), atom("MG", 3, "MG", "MG")]
    mg = np.array([5.0, 5.0, 5.0])
    p.analyses.mg.mg_selection = "resname MG"
    p.analyses.mg.geometry = True
    for shift in (np.zeros(3), cell[0], cell[2]):
        u = _in_box(make_universe(atoms, [[mg + [2.1, 0, 0], mg + [0, 2.1, 0] + shift, mg]]), OCTAHEDRON)
        res = run_one("mg", u, p)
        np.testing.assert_array_equal(res.arrays["contacts"], [2])
        np.testing.assert_array_equal(res.arrays["inter"], [1])


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 3))
    from mdatk.render.orientation import euler_to_matrix
    r = euler_to_matrix([30, -20, 75])
    y = x @ r.T + np.array([1.0, 2.0, 3.0])
    rr, mc, rc = kabsch(x, y)
    np.testing.assert_allclose((x - mc) @ rr.T + rc, y, atol=1e-8)


def test_contacts_counts_and_native_fraction():
    atoms = [atom("C1", 1, "A", "C"), atom("O1", 1, "A", "O"),
             atom("N1", 2, "U", "N"), atom("H1", 2, "U", "H")]
    f0 = [[0, 0, 0], [1, 0, 0], [4, 0, 0], [4.5, 0, 0]]
    f1 = [[0, 0, 0], [1, 0, 0], [10, 0, 0], [10.5, 0, 0]]
    u = make_universe(atoms, [f0, f1])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.contacts.cutoff = 3.5
    res = run_one("contacts", u, p)
    np.testing.assert_array_equal(res.arrays["atom_pairs"], [1, 0])  # O1-N1 only (H ignored)
    np.testing.assert_array_equal(res.arrays["q_native"], [1.0, 0.0])
    assert res.arrays["per_res_a"].shape == (1, 2)
    assert res.meta["labels_a"] == ["A1"] and res.meta["labels_b"] == ["U2"]


def test_contacts_of_a_group_with_itself():
    """A group against itself: every residue's contacts are counted (not only those with later residues),
    and the last residue of one strand and the first of the next are no sequence neighbours."""
    # two bonded strands 1-2 and 3-4 on a 3 A square: 2 faces 3 and 1 faces 4, as in an antiparallel duplex
    atoms = [atom("C1", k, "A", "C") for k in (1, 2, 3, 4)]
    square = [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0]]
    line = [[0, 0, 0], [3, 0, 0], [30, 0, 0], [33, 0, 0]]
    p = micro_project({"Aptamer": "1-4", "Target": "1-4"})
    p.analyses.contacts.cutoff = 4.5
    u = make_universe(atoms, [square, line], bonds=[(0, 1), (2, 3)])
    res = run_one("contacts", u, p)
    # 1-3, 1-4, 2-3 and 2-4; the bonded neighbours 1-2 and 3-4 are left out
    np.testing.assert_array_equal(res.arrays["atom_pairs"], [4, 0])
    np.testing.assert_array_equal(res.arrays["residue_pairs"], [4, 0])
    pairs = set(zip(res.arrays["map_i"].tolist(), res.arrays["map_j"].tolist()))
    assert pairs == {(0, 2), (0, 3), (1, 2), (1, 3)}  # each pair once
    np.testing.assert_array_equal(res.arrays["per_res_a"][:, 0], [2, 2, 2, 2])
    np.testing.assert_array_equal(res.arrays["per_res_b"][:, 0], [2, 2, 2, 2])
    # one chain (no bonds, one segment): 2-3 are sequence neighbours as before
    res = run_one("contacts", make_universe(atoms, [square]), p)
    np.testing.assert_array_equal(res.arrays["atom_pairs"], [3])
    np.testing.assert_array_equal(res.arrays["per_res_a"][:, 0], [2, 1, 1, 2])


@pytest.mark.parametrize("with_bonds", [True, False])
def test_hbond_distance_and_angle(with_bonds):
    atoms = [atom("N3", 1, "U", "N"), atom("H3", 1, "U", "H"), atom("O6", 2, "G", "O")]
    linear = [[0, 0, 0], [1.0, 0, 0], [2.9, 0, 0]]
    bent = [[0, 0, 0], [1.0, 0, 0], [0, 2.9, 0]]
    far = [[0, 0, 0], [1.0, 0, 0], [4.0, 0, 0]]
    u = make_universe(atoms, [linear, bent, far], bonds=[(0, 1)] if with_bonds else ())
    p = micro_project({"Aptamer": "1", "Target": "2"})
    res = run_one("hbonds", u, p)
    np.testing.assert_array_equal(res.arrays["count"], [1, 0, 0])
    p.analyses.hbonds.use_angle = False
    res = run_one("hbonds", u, p)
    np.testing.assert_array_equal(res.arrays["count"], [1, 1, 0])


@pytest.mark.parametrize("with_bonds", [True, False])
def test_hbond_nitrogen_carrying_hydrogen_is_no_acceptor(with_bonds):
    """N3-H3 points at N1 (2.9 A, linear), which carries H1 itself: no acceptor, no bond - also when the
    topology has no bonds and the hydrogens are assigned by distance."""
    atoms = [atom("N3", 1, "U", "N"), atom("H3", 1, "U", "H"), atom("N1", 2, "G", "N"), atom("H1", 2, "G", "H"),
             atom("N7", 2, "G", "N")]
    to_nh = [[0, 0, 0], [1.0, 0, 0], [2.9, 0, 0], [3.9, 0, 0], [0, 50, 0]]
    to_bare_n = [[0, 0, 0], [1.0, 0, 0], [50, 0, 0], [51, 0, 0], [2.9, 0, 0]]  # N7 carries no hydrogen
    u = make_universe(atoms, [to_nh, to_bare_n], bonds=[(0, 1), (2, 3)] if with_bonds else ())
    res = run_one("hbonds", u, micro_project({"Aptamer": "1", "Target": "2"}))
    np.testing.assert_array_equal(res.arrays["count"], [0, 1])


def test_hbond_distance_only_counts_an_atom_pair_once():
    """Without the angle criterion (a topology without hydrogens, or use_angle off) two atoms that can
    both donate and accept are one hydrogen bond: it was counted once per direction."""
    p = micro_project({"Aptamer": "1", "Target": "2"})
    # no hydrogens: G N1 / N2 / O6 against C N3 / O2 / N4, three contacts of 2.9 A, then all 4 A apart
    atoms = [atom("O6", 1, "G", "O"), atom("N1", 1, "G", "N"), atom("N2", 1, "G", "N"),
             atom("N4", 2, "C", "N"), atom("N3", 2, "C", "N"), atom("O2", 2, "C", "O")]
    near = [[0, 0, 0], [0, 10, 0], [0, 20, 0], [2.9, 0, 0], [2.9, 10, 0], [2.9, 20, 0]]
    far = [[0, 0, 0], [0, 10, 0], [0, 20, 0], [4.0, 0, 0], [4.0, 10, 0], [4.0, 20, 0]]
    res = run_one("hbonds", make_universe(atoms, [near, far]), p)
    assert not res.meta["used_angle"]
    np.testing.assert_array_equal(res.arrays["count"], [3, 0])
    np.testing.assert_array_equal(res.arrays["per_res_a"], [[3, 0]])
    np.testing.assert_array_equal(res.arrays["map_v"], [3])
    # with hydrogens and the angle switched off: two hydroxyls 2.8 A apart are one bond
    atoms = [atom("O2'", 1, "A", "O"), atom("HO2'", 1, "A", "H"), atom("O2'", 2, "U", "O"), atom("HO2'", 2, "U", "H")]
    xyz = [[0, 0, 0], [0.96, 0, 0], [2.8, 0, 0], [3.5, 0.6, 0]]
    u = make_universe(atoms, [xyz], bonds=[(0, 1), (2, 3)])
    np.testing.assert_array_equal(run_one("hbonds", u, p).arrays["count"], [1])  # the angle picks the donor
    p.analyses.hbonds.use_angle = False
    np.testing.assert_array_equal(run_one("hbonds", u, p).arrays["count"], [1])


def test_mg_bridges_geometry_and_occupancy():
    atoms = [atom("OP1", 1, "A", "O"), atom("OP2", 2, "U", "O"), atom("MG", 3, "MG", "MG")]
    mg = np.array([5.0, 5.0, 5.0])

    def frame(o2):
        return [mg + [2.1, 0, 0], mg + o2, mg]

    ang60 = 2.1 * np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0])
    coords = [frame([0, 2.1, 0]), frame(ang60), frame([0, 3.5, 0]), frame([0, 9.0, 0])]
    u = make_universe(atoms, coords)
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.mg.mg_selection = "resname MG"
    res = run_one("mg", u, p)
    np.testing.assert_array_equal(res.arrays["contacts"], [2, 2, 1, 1])
    np.testing.assert_array_equal(res.arrays["outer"], [0, 0, 1, 0])
    np.testing.assert_array_equal(res.arrays["bridges"], [1, 1, 0, 0])
    np.testing.assert_array_equal(res.arrays["inter"], [1, 1, 0, 0])
    np.testing.assert_array_equal(res.arrays["bridging_ions"], [1, 1, 0, 0])
    occ = res.arrays["occupancy"]
    np.testing.assert_array_equal(occ[0], [2, 2, 2, 2])
    np.testing.assert_array_equal(occ[1], [2, 2, 1, 0])
    p.analyses.mg.geometry = True  # 60 degrees is not octahedral
    res = run_one("mg", u, p)
    np.testing.assert_array_equal(res.arrays["bridges"], [1, 0, 0, 0])
    p.analyses.mg.geometry = False
    p.analyses.mg.bridge_shell = "outer"
    res = run_one("mg", u, p)
    np.testing.assert_array_equal(res.arrays["bridges"], [1, 1, 1, 0])


def _ring(center, normal_rot=None):
    names = ["N1", "C2", "N3", "C4", "C5", "C6"]
    ang = np.radians(np.arange(6) * 60.0)
    pts = np.stack([1.39 * np.cos(ang), 1.39 * np.sin(ang), np.zeros(6)], axis=1)
    if normal_rot is not None:
        pts = pts @ normal_rot.T
    return names, pts + np.asarray(center)


def test_stacking_criteria():
    from mdatk.render.orientation import rx

    atoms = []
    for resid in (1, 2):
        for n in ["N1", "C2", "N3", "C4", "C5", "C6"]:
            atoms.append(atom(n, resid, "U", n[0]))
    frames = []
    for center, rot in (([0, 0, 3.5], None), ([2.8, 0, 3.5], None), ([0, 0, 3.5], rx(60))):
        _, r1 = _ring([0, 0, 0])
        _, r2 = _ring(center, rot)
        frames.append(np.vstack([r1, r2]))
    u = make_universe(atoms, frames)
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.stacking.selection = "all"
    res = run_one("stacking", u, p)
    np.testing.assert_array_equal(res.arrays["count"], [1, 0, 0])  # stacked, offset, tilted
    np.testing.assert_array_equal(res.arrays["inter"], [1, 0, 0])
    p.analyses.stacking.offset_cutoff = 3.0
    res = run_one("stacking", u, p)
    np.testing.assert_array_equal(res.arrays["count"], [1, 1, 0])


def test_energy_matches_analytic_coulomb_and_lj():
    atoms = [atom("NA", 1, "NA", "NA", charge=1.0), atom("CL", 2, "CL", "CL", charge=-1.0)]
    r = 5.0
    u = make_universe(atoms, [[[0, 0, 0], [r, 0, 0]]])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    res = run_one("energy", u, p)
    (rs_na, e_na), (rs_cl, e_cl) = GENERIC_LJ["NA"], GENERIC_LJ["CL"]
    rmin, eps = rs_na + rs_cl, np.sqrt(e_na * e_cl)

    def lj(x):
        return eps * ((rmin / x) ** 12 - 2 * (rmin / x) ** 6)

    assert res.arrays["elec"][0] == pytest.approx(-COULOMB / r, rel=1e-9)
    assert res.arrays["vdw"][0] == pytest.approx(lj(r), rel=1e-9)
    h = 1e-5
    de = (-COULOMB / (r + h) + lj(r + h) - (-COULOMB / (r - h) + lj(r - h))) / (2 * h)
    assert res.arrays["force"][0] == pytest.approx(abs(de), rel=1e-5)
    assert res.meta["lj_source"] == "generic"
    p.analyses.energy.dielectric = 4.0
    p.analyses.energy.distance_dependent = True
    res = run_one("energy", u, p)
    assert res.arrays["elec"][0] == pytest.approx(-COULOMB / (4.0 * r * r), rel=1e-9)


def test_parm7_lj_table_roundtrip(demo_dry):
    import MDAnalysis as mda
    from mdatk.demo import LJ_TYPES

    u = mda.Universe(demo_dry["topology"])
    table = LJTable.from_parm7(demo_dry["topology"])
    types = list(u.atoms.types)
    i = np.array([0, 5, 30, 100])
    j = np.array([1, 7, 64, 200])
    a, b = table.pair_coefficients(i, j)
    for k in range(len(i)):
        (ri, ei), (rj, ej) = LJ_TYPES[types[i[k]]], LJ_TYPES[types[j[k]]]
        rmin, eps = ri + rj, np.sqrt(ei * ej)
        assert a[k] == pytest.approx(eps * rmin ** 12, rel=1e-6)
        assert b[k] == pytest.approx(2 * eps * rmin ** 6, rel=1e-6)


def test_rmsd_superposition():
    rng = np.random.default_rng(3)
    x = rng.normal(scale=3.0, size=(12, 3))
    from mdatk.render.orientation import euler_to_matrix
    y = x @ euler_to_matrix([40, 10, -30]).T + [5, 0, 0]
    atoms = [atom(f"C{k}", 1 + k // 6, "A", "C") for k in range(12)]
    u = make_universe(atoms, [x, y])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.rmsd.targets[0].selection = "group:Aptamer or group:Target"
    res = run_one("rmsd", u, p)
    assert res.arrays["rmsd_0"][1] == pytest.approx(0.0, abs=1e-4)
    p.analyses.rmsd.targets[0].fit_selection = "group:Aptamer"
    res = run_one("rmsd", u, p)
    assert res.arrays["rmsd_0"][1] == pytest.approx(0.0, abs=1e-4)
    p.analyses.rmsd.superposition = False
    res = run_one("rmsd", u, p)
    assert res.arrays["rmsd_0"][1] > 1.0


def test_demo_system_signals(demo_dry):
    """The scripted fraying shows up in contacts, H-bonds, Mg bridges and proximity."""
    from mdatk.config import Project
    from mdatk.pipeline import Pipeline

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = demo_dry["preset"].rsplit("/", 1)[0] + "/signals_out"
    ds = Pipeline(p).analyze(["contacts", "hbonds", "mg", "stacking", "trace", "proximity"])
    hb = ds.series("hbonds/count").values
    # 6 A-U (2) + 6 G-C (3) Watson-Crick H-bonds, +/- thermal-noise extras
    assert 28 <= hb[0] <= 34
    mid = len(hb) // 2
    assert hb[mid] < hb[0]
    assert ds.series("contacts/q_native").values[mid] < 1.0
    inter = ds.series("mg/inter_bridges").values
    assert inter[0] == 1 and inter[mid] == 0
    assert ds.series("stacking/count").values.min() > 0
    trace = ds.results["trace"].arrays["coords"]
    assert trace.shape == (len(hb), 24, 3)


def _pdb_line(record, serial, name, resname, chain, resseq, x, el):
    return (f"{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
            f"  1.00  0.00          {el:>2}")


def test_chains_of_a_pdb_with_conect_records_for_ligands_only(tmp_path):
    """A one-chain PDB file whose only CONECT records are those of a ligand (as PyMOL, ChimeraX and the PDB
    write them) has bonds, but they say nothing about chains: every nucleotide became a chain of its own,
    so the structure view drew no backbone."""
    import MDAnalysis as mda
    from mdatk.analysis.structure import chain_ids

    lines, serial = [], 0
    for resseq in range(1, 5):
        for name, el, dx in (("P", "P", 0.0), ("C4'", "C", 1.5), ("N1", "N", 3.0)):
            serial += 1
            lines.append(_pdb_line("ATOM", serial, name, "U", "A", resseq, 6.0 * resseq + dx, el))
    for name, el, dx in (("C1", "C", 0.0), ("O1", "O", 1.4)):
        serial += 1
        lines.append(_pdb_line("HETATM", serial, name, "LIG", "A", 9, 40.0 + dx, el))
    conect = [f"CONECT{serial - 1:>5}{serial:>5}", f"CONECT{serial:>5}{serial - 1:>5}"]
    for name, extra in (("plain.pdb", []), ("ligand_bonds.pdb", conect)):
        path = tmp_path / name
        path.write_text("\n".join(lines + extra + ["END"]) + "\n")
        u = mda.Universe(str(path))
        np.testing.assert_array_equal(chain_ids(u.residues), [0, 0, 0, 0, 0])
        p = micro_project({"Aptamer": "1-4", "Target": "9"})
        res = run_one("trace", u, p)
        np.testing.assert_array_equal(res.arrays["links"], [[0, 1], [1, 2], [2, 3]])
    # complete bonds still tell the chains apart: two bonded strands without chain ids
    atoms = [atom("P", k, "U", "P") for k in (1, 2, 3, 4)]
    u = make_universe(atoms, [[[6.0 * k, 0, 0] for k in range(4)]], bonds=[(0, 1), (2, 3)])
    np.testing.assert_array_equal(chain_ids(u.residues), [0, 0, 1, 1])


def test_energy_excludes_bonded_pairs_across_groups():
    from mdatk.analysis.energy import _bonded_exclusions

    atoms = [atom(f"C{k}", 1 if k < 3 else 2, "A", "C") for k in range(6)]
    coords = [[[1.5 * k, 0, 0] for k in range(6)]]
    bonds = [(k, k + 1) for k in range(5)]
    u = make_universe(atoms, coords, bonds=bonds)
    a, b = u.atoms[:3], u.atoms[3:]
    keys = _bonded_exclusions(u, a, b, 3)
    pairs = {(int(k) // 3, int(k) % 3) for k in keys}  # (a_local, b_local)
    assert pairs == {(2, 0), (1, 0), (2, 1), (0, 0), (1, 1), (2, 2)}
    assert len(_bonded_exclusions(u, a, b, 1)) == 1
    assert len(_bonded_exclusions(u, a, b, 0)) == 0
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.energy.exclude_bonded = 3
    res = run_one("energy", u, p)
    assert res.meta["n_excluded_pairs"] == 6
