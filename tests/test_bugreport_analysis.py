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
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


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
