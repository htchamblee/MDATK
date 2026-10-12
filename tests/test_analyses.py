import numpy as np
import pytest

from conftest import make_universe, micro_project, run_one
from mdatk.analysis.energy import COULOMB, GENERIC_LJ
from mdatk.io.parm7 import LJTable
from mdatk.spatial import kabsch, pairs_within


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
