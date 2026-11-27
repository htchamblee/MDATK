"""Regression tests of the analysis items of the external bug report on MDATK 0.1.1 (MDATK-BUG-011 to 027, 029,
067, 070, 071, 073, 074): each test is the report's case with numbers that can be checked by hand, fails on 0.1.1
and passes with the fix named in its docstring."""
import logging
import math

import numpy as np
import pytest

from conftest import make_universe, micro_project, run_one


def atom(name, resid, resname, element, charge=0.0, mass=12.0):
    return dict(name=name, resid=resid, resname=resname, element=element, charge=charge, mass=mass)


def warnings_of(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("mdatk")]


# ----------------------------------------------------------------------------------------------- MDATK-BUG-011
def test_bug011_an_atoms_element_does_not_depend_on_the_group_it_is_evaluated_in():
    """One atom without an element in the topology (an Amber extra point) made every atom of the group take the
    first letter of its name: chlorine CL1 became carbon, bromine boron, a bound Mg 'M'."""
    import MDAnalysis as mda

    from mdatk.selections import atom_elements, heavy

    u = mda.Universe.empty(6, n_residues=2, atom_resindex=[0, 0, 0, 0, 0, 1], trajectory=True)
    u.add_TopologyAttr("names", ["C1", "CL1", "BR1", "MG", "HG", "EPW"])
    u.add_TopologyAttr("resnames", ["LIG", "WAT"])
    u.add_TopologyAttr("elements", ["C", "Cl", "Br", "Mg", "H", ""])
    ligand = u.select_atoms("resname LIG")
    assert list(atom_elements(ligand)) == ["C", "CL", "BR", "MG", "H"]
    assert list(atom_elements(u.atoms)) == ["C", "CL", "BR", "MG", "H", "E"]  # the extra point alone is guessed
    assert list(atom_elements(u.atoms[[5]])) == ["E"] and list(atom_elements(u.atoms[[]])) == []
    assert heavy(u.atoms).names.tolist() == ["C1", "CL1", "BR1", "MG", "EPW"]
    # a topology without elements is guessed from the names, as before (a lone MG is the ion)
    v = mda.Universe.empty(3, n_residues=2, atom_resindex=[0, 0, 1], trajectory=True)
    v.add_TopologyAttr("names", ["CA", "1HB", "MG"])
    v.add_TopologyAttr("resnames", ["ALA", "MG"])
    assert list(atom_elements(v.atoms)) == ["C", "H", "MG"]


# ----------------------------------------------------------------------------------- MDATK-BUG-012, 013, 014
def errors_of(project, field):
    """Texts of the validation errors of one setting (the demo-less project has others: no input files)."""
    return [p.message for p in project.validate() if p.field == field]


POSITIVE = ["analyses.contacts.cutoff", "analyses.hbonds.distance_cutoff", "analyses.mg.inner_cutoff",
            "analyses.mg.outer_cutoff", "analyses.stacking.distance_cutoff", "analyses.stacking.offset_cutoff",
            "analyses.stacking.planarity_cutoff", "analyses.energy.cutoff", "analyses.energy.dielectric",
            "analyses.proximity_search_radius"]
ANGLES = ["analyses.hbonds.angle_cutoff", "analyses.stacking.angle_cutoff", "analyses.mg.angle_tolerance"]


def set_field(project, path, value):
    obj = project
    *parents, leaf = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    assert hasattr(obj, leaf)
    setattr(obj, leaf, value)


@pytest.mark.parametrize("field", POSITIVE)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0, -3.5])
def test_bug012_cutoffs_radii_and_the_dielectric_must_be_finite_and_positive(field, value):
    """validate() only refused contacts.cutoff <= 0 (and 'nan <= 0' is false): a NaN H-bond cutoff counted no
    bond, -3.5 counted those of +3.5, dielectric 0 gave infinite energies, all without a message."""
    from mdatk.config import Project

    p = Project()
    assert errors_of(p, field) == []  # the defaults are valid
    set_field(p, field, value)
    (message,) = [m for m in errors_of(p, field) if "must not exceed" not in m]  # (Mg: inner against outer)
    assert message.endswith("must be positive." if math.isfinite(value) else "must be a finite number.")
    set_field(p, field, 0.01)
    assert [m for m in errors_of(p, field) if "must not exceed" not in m] == []


@pytest.mark.parametrize("field", ANGLES)
def test_bug012_angles_must_lie_between_0_and_180_degrees(field):
    from mdatk.config import Project

    p = Project()
    for value in (0.0, 25.0, 180.0):
        set_field(p, field, value)
        assert errors_of(p, field) == []
    for value in (float("nan"), float("inf"), -1.0, 180.5):
        set_field(p, field, value)
        (message,) = errors_of(p, field)
        assert message.endswith("must be between 0 and 180 degrees.")


def test_bug012_a_wrong_cutoff_stops_the_run_instead_of_giving_numbers(demo_dry, tmp_path):
    from mdatk.config import Project
    from mdatk.pipeline import Pipeline, PipelineError

    p = Project.load(demo_dry["project"])
    p.output.out_dir = str(tmp_path / "out")
    p.analyses.hbonds.distance_cutoff = float("nan")
    with pytest.raises(PipelineError, match="H-bond distance cutoff must be a finite number"):
        Pipeline(p).check(video=False, charts=False, pdb=False)


def test_bug013_two_rmsd_targets_with_the_same_data_key_are_refused():
    """'A B' and 'A_B' both give the series rmsd/A_B: the second replaced the first without a word."""
    from mdatk.analysis.rmsd import RMSDAnalysis
    from mdatk.config import Project, RMSDTarget, rmsd_series_key

    p = Project()
    assert errors_of(p, "analyses.rmsd.targets.1.label") == []
    p.analyses.rmsd.targets = [RMSDTarget("A B", "group:Aptamer and name P"),
                               RMSDTarget("A_B", "group:Target and name P"), RMSDTarget("C", "name P")]
    assert RMSDAnalysis.expected_outputs(p)["series"] == ["A_B", "A_B", "C"]
    assert rmsd_series_key("A B", 0) == rmsd_series_key("A_B", 1) == "A_B" and rmsd_series_key(" ", 2) == "target2"
    assert errors_of(p, "analyses.rmsd.targets.1.label") == [
        "RMSD targets 1 and 2 give the same data key 'A_B'; rename one."]
    assert errors_of(p, "analyses.rmsd.targets.0.label") == errors_of(p, "analyses.rmsd.targets.2.label") == []
    p.analyses.rmsd.targets[1].label = "B"
    assert errors_of(p, "analyses.rmsd.targets.1.label") == []


@pytest.mark.parametrize("value", [-2.0, float("nan"), float("inf"), float("-inf")])
def test_bug014_a_negative_or_non_finite_time_step_override_is_refused(value):
    """inputs.dt_ps = -2 gave the time axis 0, -2, -4 ...; NaN a time axis of NaN."""
    from mdatk.config import Project

    p = Project()
    assert errors_of(p, "inputs.dt_ps") == []  # not set: the trajectory's times
    for fine in (0, 0.0, 2, 0.5):  # 0 keeps meaning "from the file"
        p.inputs.dt_ps = fine
        assert errors_of(p, "inputs.dt_ps") == []
    p.inputs.dt_ps = value
    (message,) = errors_of(p, "inputs.dt_ps")
    assert message.startswith("The time step override must be a positive number of ps")


# ----------------------------------------------------------------------------------- MDATK-BUG-015, 016, 067
def rmsd_project(selection, fit="", **settings):
    from mdatk.config import RMSDTarget

    p = micro_project({"All": "1-9"})
    p.analyses.rmsd.targets = [RMSDTarget("T", selection, fit)]
    for key, value in settings.items():
        setattr(p.analyses.rmsd, key, value)
    return p


def test_bug015_rmsd_of_one_atom_is_zero_and_of_two_atoms_exact():
    """MDAnalysis' QCP superposition gave NaN for one atom and NaN or several Å for two (even for the reference
    frame itself).  Two atoms 2 Å apart, then the same pair turned by 90 degrees and moved (RMSD 0), then
    stretched to 3 Å (each atom 0.5 Å from its place: RMSD 0.5)."""
    atoms = [atom("C4'", 1, "G", "C"), atom("C4'", 2, "C", "C")]
    frames = [[[0, 0, 0], [2, 0, 0]], [[5, 5, 5], [5, 7, 5]], [[0, 0, 0], [3, 0, 0]]]
    u = make_universe(atoms, frames)
    res = run_one("rmsd", u, rmsd_project("name C4'"))
    np.testing.assert_allclose(res.arrays["rmsd_0"], [0.0, 0.0, 0.5], atol=1e-6)
    one = run_one("rmsd", u, rmsd_project("resid 1"))
    np.testing.assert_array_equal(one.arrays["rmsd_0"], [0.0, 0.0, 0.0])
    heavy = run_one("rmsd", u, rmsd_project("name C4'", mass_weighted=True))  # (equal masses: the same numbers)
    np.testing.assert_allclose(heavy.arrays["rmsd_0"], [0.0, 0.0, 0.5], atol=1e-6)


@pytest.mark.parametrize("n_atoms, on_a_line", [(2, False), (3, True), (5, True), (4, False)])
def test_bug015_rigid_copies_have_rmsd_zero(n_atoms, on_a_line):
    """100 rotated and shifted copies of float32 coordinates (the report's families: the exact answer is 0).  With
    QCP 24 of 100 two-atom copies were NaN and 11 off by up to 6 Å; atoms on one line failed the same way."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(3)
    if on_a_line:
        first = np.outer(np.arange(n_atoms) * 1.5, [0.6, 0.0, 0.8]) + 3.0
    else:
        first = rng.normal(size=(n_atoms, 3)) * 5
    frames = [first, first] + [Rotation.random(random_state=t).apply(first) + 2 for t in range(100)]
    u = make_universe([atom("P", k + 1, "G", "P") for k in range(n_atoms)], frames)
    values = run_one("rmsd", u, rmsd_project("name P")).arrays["rmsd_0"]
    assert np.isfinite(values).all()
    assert values[:2].max() < 1e-6 and values.max() < 2e-5  # (float32 coordinates: rounded after rotating)


def test_bug016_a_fit_selection_that_defines_no_rotation_only_translates_and_says_so(caplog):
    """Two fit atoms (or three on one line) turned by 90 degrees about z: no rotation is defined (about their line
    it is arbitrary), so the fit only moves the centre.  The measured atom (1, 1, 0) is then at (0, 0, 0): the RMSD
    is sqrt(2), with a warning.  0.1.1 gave no warning, and for the three atoms a rotation chosen by LAPACK."""
    atoms = [atom("P", 1, "G", "P"), atom("P", 2, "G", "P"), atom("P", 3, "G", "P"), atom("C1'", 4, "G", "C")]
    turned = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [-1, 1, 0]]  # (x, y) -> (-y, x)
    u = make_universe(atoms, [[[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], turned])
    with caplog.at_level("WARNING"):
        three = run_one("rmsd", u, rmsd_project("name C1'", "name P"))
    assert warnings_of(caplog) == [
        "RMSD target 'T': fit selection has 3 atoms on one line; superposition is translation only"]
    np.testing.assert_allclose(three.arrays["rmsd_0"], [0.0, math.sqrt(2)], atol=1e-6)
    caplog.clear()
    with caplog.at_level("WARNING"):
        two = run_one("rmsd", u, rmsd_project("name C1'", "name P and resid 1 3"))
    assert warnings_of(caplog) == ["RMSD target 'T': fit selection has 2 atoms; superposition is translation only"]
    np.testing.assert_allclose(two.arrays["rmsd_0"], [0.0, math.sqrt(2)], atol=1e-6)
    caplog.clear()
    # three fit atoms that are not on one line define the rotation: the measured atom comes back, no warning
    bent = [atom("P", 1, "G", "P"), atom("P", 2, "G", "P"), atom("C4'", 3, "G", "C"), atom("C1'", 4, "G", "C")]
    v = make_universe(bent, [[[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [-2, 0, 0], [-1, 1, 0]]])
    with caplog.at_level("WARNING"):
        full = run_one("rmsd", v, rmsd_project("name C1'", "name P C4'"))
    assert warnings_of(caplog) == []
    np.testing.assert_allclose(full.arrays["rmsd_0"], [0.0, 0.0], atol=1e-6)


def test_bug016_points_on_one_line_are_recognised():
    from mdatk.spatial import collinear, kabsch

    line = np.outer([0.0, 1.0, 2.5, 4.0], [1.0, 2.0, -1.0])
    assert collinear(line) and collinear(line[:2]) and collinear(np.zeros((4, 3)))
    assert collinear(line.astype(np.float32) + 100.0)  # (rounding does not make a line a plane)
    assert not collinear(line + [[0, 0, 0], [0, 0, 0.1], [0, 0, 0], [0, 0, 0]])
    assert not collinear(np.eye(3)) and collinear(np.eye(3), weights=[1.0, 1.0, 0.0])
    r = kabsch(np.array([[0, 0, 0], [1, 0, 0.0]]), np.array([[0, 0, 0], [0, 1, 0.0]]))[0]
    np.testing.assert_array_equal(r, np.eye(3))  # as before: fewer than three points only translate


def test_bug067_an_rmsd_reference_frame_outside_the_run_is_reported(caplog):
    """reference_frame = 999 silently used the last analysed frame, a negative one the first."""
    atoms = [atom("P", k + 1, "G", "P") for k in range(3)]
    first = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0.0]])
    u = make_universe(atoms, [first, first * 1.1, first * 1.2])
    with caplog.at_level("WARNING"):
        res = run_one("rmsd", u, rmsd_project("name P", reference_frame=999))
    assert warnings_of(caplog) == ["RMSD: reference frame 999 is beyond the 3 analysed frame(s); using the last"]
    assert res.meta["reference_frame_index"] == 2 and res.arrays["rmsd_0"][2] < 1e-6 < res.arrays["rmsd_0"][0]
    caplog.clear()
    with caplog.at_level("WARNING"):
        res = run_one("rmsd", u, rmsd_project("name P", reference_frame=-3))
    assert warnings_of(caplog) == ["RMSD: reference frame -3 is negative; using the first"]
    assert res.meta["reference_frame_index"] == 0
    caplog.clear()
    with caplog.at_level("WARNING"):
        run_one("rmsd", u, rmsd_project("name P", reference_frame=2))
    assert warnings_of(caplog) == []


# ---------------------------------------------------------------------------------------- MDATK-BUG-017, 029
def with_box(universe, box):
    for ts in universe.trajectory:
        ts.dimensions = box
    return universe


OCTAHEDRON = [30.0, 30.0, 30.0, 109.4712190, 109.4712190, 109.4712190]


def test_bug017_a_pair_at_the_cutoff_is_kept_whatever_the_size_of_the_box():
    """Two points exactly 3.5 Å apart, cutoff 3.5 Å: kept without a box, but with one MDAnalysis' single-precision
    search kept the pair for L = 25 and 31.7 Å and dropped it for 30, 40 and 60 Å."""
    from mdatk.spatial import pairs_within

    a, b = np.array([[1.0, 1.0, 1.0]]), np.array([[4.5, 1.0, 1.0]])
    assert pairs_within(a, b, 3.5)[2].tolist() == [3.5]
    for length in (25, 30, 31.7, 40, 60, 137.3):
        i, j, d = pairs_within(a, b, 3.5, box=[length, length, length, 90, 90, 90])
        assert (i.tolist(), j.tolist(), d.tolist()) == ([0], [0], [3.5]), length
        assert len(pairs_within(a, b + [1e-9, 0, 0], 3.5, box=[length, length, length, 90, 90, 90])[0]) == 0
    assert pairs_within(a, b, 3.5, box=OCTAHEDRON)[2].tolist() == [3.5]
    # the same pair across a face of the box: 1.0 and 27.5 are 3.5 Å apart in a 30 Å box
    i, j, d = pairs_within(a, np.array([[27.5, 1.0, 1.0]]), 3.5, box=[30, 30, 30, 90, 90, 90])
    assert d.tolist() == [3.5]


def test_bug017_contacts_do_not_depend_on_the_box_size():
    atoms = [atom("P", 1, "G", "P"), atom("P", 2, "C", "P")]
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.contacts.cutoff = 3.5
    p.analyses.use_pbc = True
    for length in (25.0, 30.0, 40.0):
        u = with_box(make_universe(atoms, [[[1, 1, 1], [4.5, 1, 1]]]), [length, length, length, 90, 90, 90])
        assert run_one("contacts", u, p).arrays["atom_pairs"].tolist() == [1], length


@pytest.mark.parametrize("box", [[20.0, 25.0, 30.0, 90, 90, 90], OCTAHEDRON, [25.0, 30.0, 28.0, 80, 95, 110]])
def test_bug017_points_far_outside_the_cell_get_their_exact_distance(box):
    """A point 500 box lengths away from its partner's image is at distance 0, not 1.5e-4 Å; random points up to 40
    cells out have the pairs and distances of a double-precision minimum-image reference."""
    from MDAnalysis.lib.mdamath import triclinic_vectors

    from mdatk.spatial import minimum_image_vectors, pairs_within

    box = np.array(box, dtype=np.float64)
    cell = triclinic_vectors(box.astype(np.float32)).astype(np.float64)  # the cell MDAnalysis uses
    origin = np.zeros((1, 3))
    np.testing.assert_array_equal(pairs_within(origin + 500 * cell[0], origin, 5.0, box)[2], [0.0])
    rng = np.random.default_rng(1)
    a = ((rng.random((120, 3)) - 0.3) * 1200).astype(np.float32).astype(np.float64)
    b = ((rng.random((120, 3)) - 0.3) * 1200).astype(np.float32).astype(np.float64)
    v = a[:, None, :] - b[None, :, :]
    v = v - np.round(v @ np.linalg.inv(cell)) @ cell
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ cell
    reference = np.linalg.norm(v[:, :, None, :] + shifts, axis=3).min(axis=2)
    for cutoff in (3.5, 9.0):
        i, j, d = pairs_within(a, b, cutoff, box)
        assert sorted(zip(i.tolist(), j.tolist())) == sorted(zip(*np.nonzero(reference <= cutoff)))
        np.testing.assert_allclose(d, reference[i, j], rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(minimum_image_vectors(a[i] - b[j], box), axis=1), d, atol=1e-9)


def test_bug029_points_without_coordinates_are_in_no_pair_with_or_without_a_box():
    """Without a box scipy's tree raised 'data must be finite' (the analysis stopped); with one the point was
    dropped.  Now both: a NaN or infinite point is in no pair and the others keep their indices."""
    from mdatk.spatial import pairs_within

    a = np.array([[0, 0, 0], [np.nan, 0, 0], [0.5, 0, 0]], dtype=float)
    b = np.array([[np.inf, 0, 0], [1.0, 0, 0]])
    for box in (None, [20, 20, 20, 90, 90, 90]):
        i, j, d = pairs_within(a, b, 3.0, box)
        assert sorted(zip(i.tolist(), j.tolist(), d.tolist())) == [(0, 1, 1.0), (2, 1, 0.5)]
    assert len(pairs_within(a[[1]], b, 3.0)[0]) == 0 and len(pairs_within(a, b[[0]], 3.0)[0]) == 0


def test_bug029_a_blown_up_frame_does_not_stop_the_analyses(caplog):
    """Frame 1 has one atom without coordinates: the contact of that atom is not counted in that frame, the RMSD
    of the frame is NaN, the run goes on and says which frame."""
    atoms = [atom("N1", 1, "G", "N"), atom("P", 1, "G", "P"), atom("N3", 2, "C", "N"), atom("P", 2, "C", "P")]
    frame = np.array([[0, 0, 0], [0, 5, 0], [3, 0, 0], [3, 5, 0.0]])
    blown = frame.copy()
    blown[0, 0] = np.nan
    u = make_universe(atoms, [frame, blown, frame])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.contacts.cutoff = 3.5
    with caplog.at_level("WARNING"):
        contacts = run_one("contacts", u, p)
    assert contacts.arrays["atom_pairs"].tolist() == [2, 1, 2]  # N1-N3 and P-P; frame 1: P-P only
    assert warnings_of(caplog) == ["frame 1: 1 atom has non-finite coordinates; such atoms are in no contact, bond "
                                   "or energy term, and values that need them (RMSD) are NaN"]
    rmsd = run_one("rmsd", u, rmsd_project("resid 1 2")).arrays["rmsd_0"]
    assert rmsd[0] < 1e-6 and np.isnan(rmsd[1]) and rmsd[2] < 1e-6
    fitted = run_one("rmsd", u, rmsd_project("name P", "resid 1 2")).arrays["rmsd_0"]
    assert fitted[0] < 1e-6 and np.isnan(fitted[1]) and fitted[2] < 1e-6
    for name in ("hbonds", "energy", "mg", "stacking", "trace", "proximity", "atoms"):
        p.visual.proximity_set = "group:Target"
        run_one(name, u, p)  # none stops
    caplog.clear()
    u2 = make_universe(atoms, [blown, blown * np.nan, frame])
    with caplog.at_level("WARNING"):
        run_one("contacts", u2, p)
    assert warnings_of(caplog)[1] == "2 of the 3 analysed frames have atoms with non-finite coordinates (frames 0 to 1)"


def test_bug029_a_ring_with_a_missing_atom_does_not_stop_the_stacking_analysis():
    """Two stacked six-rings 3.4 Å apart; in frame 1 one ring atom has no coordinates: no pair in that frame
    (numpy's SVD raised 'SVD did not converge')."""
    from mdatk.analysis.stacking import PYRIMIDINE

    ring = np.array([[np.cos(t), np.sin(t), 0.0] for t in np.arange(6) * np.pi / 3]) * 1.4
    atoms = [atom(n, r, "C", n[0]) for r in (1, 2) for n in PYRIMIDINE]
    frame = np.vstack([ring, ring + [0, 0, 3.4]])
    blown = frame.copy()
    blown[7] = np.nan
    u = make_universe(atoms, [frame, blown, frame])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    assert run_one("stacking", u, p).arrays["count"].tolist() == [1, 0, 1]


# ---------------------------------------------------------------------------------------- MDATK-BUG-018, 027
def run_many(names, universe, project, frames):
    from mdatk.analysis import ANALYSES, AnalysisContext, run_analyses

    ctxs = [AnalysisContext(universe, project) for _ in names]  # as the pipeline: one context per analysis
    results, _ = run_analyses(ctxs[0], [ANALYSES[n](c) for n, c in zip(names, ctxs)], np.asarray(frames))
    return dict(zip(names, results))


def mg_universe():
    """Mg next to G1 in frame 0 (2 Å), next to C2 in frame 1 (2 Å; U4 is 2 Å away across a face of the 20 Å
    box, 18 Å without it), far from everything in frame 2."""
    atoms = [atom("P", 1, "G", "P"), atom("P", 2, "C", "P"), atom("MG", 3, "MG", "MG"), atom("P", 4, "U", "P")]
    frames = [[[0, 0, 0], [10, 0, 0], [2, 0, 0], [12, 0, 0]],
              [[0, 0, 0], [10, 0, 0], [8, 0, 0], [8, 18, 0]],
              [[0, 0, 0], [10, 0, 0], [15, 10, 10], [8, 18, 0]]]
    u = with_box(make_universe(atoms, frames), [20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    u.trajectory[0]
    return u


@pytest.mark.parametrize("before", [[], ["rmsd"]])
@pytest.mark.parametrize("use_pbc, expected", [(False, ["C2"]), (True, ["C2", "U4"])])
def test_bug018_distance_selections_use_the_first_analysed_frame_and_follow_use_pbc(before, use_pbc, expected):
    """A group 'around 5 resname MG', frames 1 and 2 analysed.  0.1.1 selected at the frame the trajectory
    happened to be at: frame 0 (G1), or after the RMSD's setup its reference frame 2 (nothing: an error), and
    always with the periodic cell (U4) whatever analyses.use_pbc."""
    from mdatk.config import ResidueGroup, RMSDTarget

    p = micro_project({"Aptamer": "1-2", "Target": "4"})
    p.groups.append(ResidueGroup("Near", "", "name P and around 5 resname MG", "#000000"))
    p.analyses.rmsd.targets = [RMSDTarget("T", "name P")]
    p.analyses.rmsd.reference_frame = 1  # the second analysed frame: frame 2
    p.analyses.use_pbc = use_pbc
    p.visual.structure_groups = ["Near"]
    p.visual.show_ions = False
    trace = run_many(before + ["trace"], mg_universe(), p, [1, 2])["trace"]
    assert trace.meta["residue_labels"] == expected


def test_bug018_guessed_bonds_of_the_atoms_analysis_come_from_the_first_analysed_frame():
    """C-O 1.4 Å apart in frame 1 (bonded), 5 Å in frame 0, where the trajectory stood at setup: no bond."""
    atoms = [atom("C1'", 1, "G", "C"), atom("O4'", 1, "G", "O")]
    u = make_universe(atoms, [[[0, 0, 0], [5, 0, 0]], [[0, 0, 0], [1.4, 0, 0]]])
    u.trajectory[0]
    res = run_many(["atoms"], u, micro_project({"Aptamer": "1"}), [1])["atoms"]
    assert res.arrays["bonds"].tolist() == [[0, 1]]


def test_bug027_use_pbc_without_a_periodic_cell_is_reported_once(caplog):
    """use_pbc with a trajectory without a box (cleaning removes it after fitting) silently gave the distances
    without periodic images."""
    atoms = [atom("N1", 1, "G", "N"), atom("N3", 2, "C", "N")]
    frames = [[[1, 1, 1], [19, 1, 1]]] * 3  # 2 Å apart across a 20 Å box, 18 Å without one
    p = micro_project({"Aptamer": "1", "Target": "2"})
    p.analyses.use_pbc = True
    names = ["contacts", "hbonds", "energy"]
    with caplog.at_level("WARNING"):
        res = run_many(names, make_universe(atoms, frames), p, [0, 1, 2])
    assert res["contacts"].arrays["atom_pairs"].tolist() == [0, 0, 0]
    assert [w for w in warnings_of(caplog) if "use_pbc" in w] == [
        "use_pbc is on but frame 0 has no usable periodic cell (box: none); distances of such frames are computed "
        "without periodic images"]
    caplog.clear()
    u = with_box(make_universe(atoms, frames), [20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    u.trajectory[1].dimensions = [20.0, 20.0, 0.0, 90.0, 90.0, 90.0]  # (a frame without a whole cell)
    with caplog.at_level("WARNING"):
        res = run_many(names, u, p, [0, 1, 2])
    assert res["contacts"].arrays["atom_pairs"].tolist() == [1, 0, 1]
    assert [w for w in warnings_of(caplog) if "use_pbc" in w] == [
        "use_pbc is on but frame 1 has no usable periodic cell (box: 20 20 0 90 90 90); distances of such frames "
        "are computed without periodic images"]
    caplog.clear()
    p.analyses.use_pbc = False  # not asked for: nothing to report
    with caplog.at_level("WARNING"):
        run_many(names, make_universe(atoms, frames), p, [0, 1, 2])
    assert [w for w in warnings_of(caplog) if "use_pbc" in w] == []


def test_bug027_periodic_cell():
    from mdatk.spatial import periodic_cell

    assert periodic_cell(None) is None and periodic_cell([0, 0, 0, 90, 90, 90]) is None
    assert periodic_cell([20, 20, 20, 0, 0, 0]) is None and periodic_cell([20, 20, 20, 60, 60, 150]) is None
    assert periodic_cell([20, 20, float("nan"), 90, 90, 90]) is None
    assert periodic_cell([20, 25, 30, 90, 90, 90]).tolist() == [20, 25, 30, 90, 90, 90]
    assert periodic_cell(OCTAHEDRON) is not None


# ---------------------------------------------------------------------------- MDATK-BUG-019, 020, 021, 070
def test_bug019_no_bonds_in_the_topology_is_said_when_bonded_pairs_were_to_be_excluded(caplog):
    """Two bonded atoms (C-O, 1.4 Å) in two groups, a topology without bonds: the pair is not excluded (it
    cannot be known) and its van der Waals term is +1.5e3 kcal/mol; 0.1.1 said nothing."""
    atoms = [atom("C4'", 1, "G", "C"), atom("O3'", 2, "C", "O")]
    frame = [[[0, 0, 0], [1.4, 0, 0]]]
    p = micro_project({"Aptamer": "1", "Target": "2"})
    message = "energy: the topology has no bonds; bonded pairs are not excluded (exclude_bonded = 3 has no effect)"
    with caplog.at_level("WARNING"):
        res = run_one("energy", make_universe(atoms, frame), p)
    assert warnings_of(caplog) == [message]
    assert res.meta["n_excluded_pairs"] == 0 and res.arrays["vdw"][0] > 1000
    caplog.clear()
    with caplog.at_level("WARNING"):
        res = run_one("energy", make_universe(atoms, frame, bonds=[(0, 1)]), p)  # with bonds: excluded, no word
    assert warnings_of(caplog) == [] and res.meta["n_excluded_pairs"] == 1 and res.arrays["vdw"][0] == 0
    p.analyses.energy.exclude_bonded = 0  # nothing to exclude: nothing to say
    with caplog.at_level("WARNING"):
        run_one("energy", make_universe(atoms, frame), p)
    assert warnings_of(caplog) == []


def test_bug020_pairs_linked_through_atoms_of_neither_group_are_excluded():
    """A chain 0-1-2-3-4: atom 0 against atoms 2, 3 and 4, linked only through atom 1 (in neither group).  Within
    three bonds are 2 (two bonds) and 3 (three); 0.1.1 excluded nothing without a direct A-B bond."""
    import MDAnalysis as mda

    from mdatk.analysis.energy import _bonded_exclusions

    u = mda.Universe.empty(5, n_residues=5, atom_resindex=[0, 1, 2, 3, 4], trajectory=True)
    u.add_bonds([(0, 1), (1, 2), (2, 3), (3, 4)])
    a, b = u.atoms[[0]], u.atoms[[2, 3, 4]]
    assert _bonded_exclusions(u, a, b, 3).tolist() == [0, 1]  # keys a * n_b + b: atoms 2 and 3
    assert _bonded_exclusions(u, a, b, 2).tolist() == [0]
    assert _bonded_exclusions(u, a, b, 1).tolist() == [] == _bonded_exclusions(u, a, b, 0).tolist()
    assert _bonded_exclusions(u, u.atoms[[0]], u.atoms[[2]], 3).tolist() == [0]  # the report's case
    # a direct bond, as before; and an atom bonded to two atoms of the other group (both are found)
    assert _bonded_exclusions(u, u.atoms[[0, 1]], u.atoms[[2, 3, 4]], 1).tolist() == [3]  # 1-2
    assert _bonded_exclusions(u, u.atoms[[2]], u.atoms[[0, 1, 3, 4]], 1).tolist() == [1, 2]  # 2-1 and 2-3


def test_bug020_a_linker_residue_between_the_groups():
    """Strand halves joined by a linker atom that belongs to neither group: C3'(1) - X(2) - C5'(3), 1.5 Å each.
    The pair C3'-C5' (two bonds, 3 Å) is excluded, so no energy is left; 0.1.1 counted it."""
    atoms = [atom("C3'", 1, "G", "C"), atom("P", 2, "LNK", "P"), atom("C5'", 3, "C", "C")]
    u = make_universe(atoms, [[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]]], bonds=[(0, 1), (1, 2)])
    res = run_one("energy", u, micro_project({"Aptamer": "1", "Target": "3"}))
    assert res.meta["n_excluded_pairs"] == 1 and res.arrays["n_pairs"].tolist() == [0]
    assert res.arrays["vdw"].tolist() == [0.0]


TIP4P = [[0.0, 0.0, 0.0], [0.5859, 0.7570, 0.0], [0.5859, -0.7570, 0.0], [0.15, 0.0, 0.0]]  # O, H, H, extra point


@pytest.mark.parametrize("names, elements", [(["O", "H1", "H2", "EPW"], ["O", "H", "H", ""]),
                                             (["OW", "HW1", "HW2", "MW"], None),
                                             (["O", "H1", "H2", "M"], ["O", "H", "H", ""])])
def test_bug021_virtual_sites_have_no_lennard_jones_term(names, elements):
    """A TIP4P water from a topology without atom types: the extra point (0.15 Å from the oxygen, no element) got
    carbon's parameters (R* 1.908 Å, epsilon 0.086) with a warning, and as the 'heavy atom' nearest to the
    hydrogens (0.87 Å, the oxygen 0.96 Å) it made them hydrogens on carbon (R* 1.387 Å) instead of on oxygen."""
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    from mdatk.analysis import generic_lj as G

    u = mda.Universe.empty(4, n_residues=1, atom_resindex=[0, 0, 0, 0], trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("resnames", ["WAT"])
    if elements is not None:
        u.add_TopologyAttr("elements", elements)
    u.load_new(np.array([TIP4P], dtype=np.float32), format=MemoryReader)
    rstar, eps, missing = G.parameters(u, u.atoms)
    assert rstar.tolist() == [G.COVALENT["O"].rstar, 0.0, 0.0, 0.0]  # O, HO, HO, EP
    assert eps.tolist() == [G.COVALENT["O"].epsilon, 0.0, 0.0, 0.0] and missing == []
    assert G.virtual_sites(u.atoms).tolist() == [False, False, False, True]
    assert G.hydrogen_parents(u, u.atoms, np.array(["O", "H", "H", "E"], dtype=object)).tolist() == ["", "O", "O", ""]


def test_bug021_an_atom_with_an_element_is_no_virtual_site_and_the_rule_is_documented():
    import os

    from mdatk.analysis import generic_lj as G

    atoms = [atom("EPS", 1, "LIG", "C"), atom("LP1", 1, "LIG", "S"), atom("M", 1, "LIG", "N"),
             atom("MG", 2, "MG", ""), atom("EP1", 3, "WAT", ""), atom("LP2", 3, "WAT", "")]
    u = make_universe(atoms, [np.arange(18).reshape(6, 3) * 3.0])
    assert G.virtual_sites(u.atoms).tolist() == [False, False, False, False, True, True]
    rstar, _eps, missing = G.parameters(u, u.atoms)
    assert rstar.tolist() == [1.908, 2.0, 1.824, 1.36, 0.0, 0.0] and missing == []
    assert tuple(G.EXTRA_POINT) == (0.0, 0.0, "EP", "parm10.dat")  # "  EP   0.00   0.0000   lone pair" (MOD4 RE)
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "GENERIC_LJ.md")
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()
    assert "**A virtual site**" in text and "`EP 0.00 0.0000 lone pair`" in text
    assert all(name in text for name in ("`EPW`", "`LP…`", "`M`", "`MW`"))


def test_bug070_the_net_force_of_a_pair_closer_than_half_an_angstrom_is_that_of_half_an_angstrom():
    """'Distances below 0.5 Å are taken as 0.5 Å': a Na+ / Cl- pair at 0.5 Å and at 0.25 Å has the same energy and
    the same net force, in any direction.  0.1.1 multiplied the force vector of the clamped distance by r / 0.5:
    half the force at 0.25 Å.  At 0.5 Å: E_el = -332.0522 / 0.5 = -664.1044 kcal/mol."""
    from mdatk.analysis import generic_lj as G

    atoms = [atom("NA", 1, "NA", "NA", charge=1.0), atom("CL", 2, "CL", "CL", charge=-1.0)]
    frames = [[[0, 0, 0], [0.5, 0, 0]], [[0, 0, 0], [0.25, 0, 0]], [[0, 0, 0], [0, 0.15, 0.2]], [[1, 1, 1], [1, 1, 1]]]
    res = run_one("energy", make_universe(atoms, frames), micro_project({"Aptamer": "1", "Target": "2"}))
    elec, vdw, force = res.arrays["elec"], res.arrays["vdw"], res.arrays["force"]
    np.testing.assert_allclose(elec, [-664.1044] * 4, rtol=1e-12)
    na, cl = G.IONS["NA"], G.IONS["CL"]
    rmin, eps = na.rstar + cl.rstar, math.sqrt(na.epsilon * cl.epsilon)
    de_dr = 332.0522 / 0.25 + eps * (-12 * rmin ** 12 / 0.5 ** 13 + 12 * rmin ** 6 / 0.5 ** 7)
    np.testing.assert_allclose(vdw, [eps * ((rmin / 0.5) ** 12 - 2 * (rmin / 0.5) ** 6)] * 4, rtol=1e-12)
    np.testing.assert_allclose(force[:3], [abs(de_dr)] * 3, rtol=1e-6)  # (float32 positions: 0.15, 0.2)
    assert force[0] == force[1]
    assert force[3] == 0.0  # at distance 0 the pair has no direction
