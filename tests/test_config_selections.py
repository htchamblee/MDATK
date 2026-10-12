import json

import numpy as np
import pytest

from mdatk.config import PanelSpec, Project, ResidueGroup, stable_hash
from mdatk.selections import (SelectionError, expand_groups, guess_element, parse_frame_spec,
                              parse_residue_ranges, parse_time_spec, ranges_to_selection,
                              resolve_selection)


def test_project_roundtrip(tmp_path):
    p = Project()
    p.inputs.topology = str(tmp_path / "sys.parm7")
    p.inputs.trajectories = [str(tmp_path / "a.nc"), str(tmp_path / "b.nc")]
    p.output.out_dir = str(tmp_path / "out")  # relative paths resolve against the preset folder
    p.analyses.contacts.cutoff = 5.25
    p.layout.panels.append(PanelSpec("contact_map", ["contacts/map"], "Map", {"mode": "frequency"}))
    path = tmp_path / "preset.json"
    p.save(str(path))
    q = Project.load(str(path))
    assert q.to_dict() == p.to_dict()
    assert stable_hash(q.to_dict()) == stable_hash(p.to_dict())


def test_relative_paths_and_aliases(tmp_path, caplog):
    data = {"input": {"topology": "top.parm7", "trajectory": "traj.nc"},
            "output": {"out_dir": "out"}, "not_a_key": 1}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data))
    p = Project.load(str(path))
    assert p.inputs.topology == str(tmp_path / "top.parm7")
    assert p.inputs.trajectories == [str(tmp_path / "traj.nc")]
    assert p.output.out_dir == str(tmp_path / "out")
    assert "not_a_key" in caplog.text


def test_type_errors_are_reported():
    with pytest.raises(ValueError, match="analyses.contacts.cutoff"):
        Project.from_dict({"analyses": {"contacts": {"cutoff": "far"}}})
    with pytest.raises(ValueError, match="enabled"):
        Project.from_dict({"analyses": {"rmsd": {"enabled": 1}}})


def test_params_log_can_be_replayed(tmp_path):
    p = Project()
    p.analyses.hbonds.angle_cutoff = 150.0
    log = {"format": "mdatk-params-log", "parameters": p.to_dict()}
    path = tmp_path / "log.json"
    path.write_text(json.dumps(log))
    assert Project.load(str(path)).analyses.hbonds.angle_cutoff == 150.0


def test_validate_flags_problems(tmp_path):
    p = Project()
    problems = p.validate()
    assert any("topology" in m for m in problems)
    p.groups.append(ResidueGroup("Bad name", "1-3"))
    p.visual.glow_mode = "sparkle"
    problems = p.validate()
    assert any("spaces" in m for m in problems)
    assert any("glow" in m for m in problems)


def test_residue_ranges():
    assert parse_residue_ranges("1-40, 45 ,50 - 60") == [(None, 1, 40), (None, 45, 45), (None, 50, 60)]
    assert parse_residue_ranges("A:1-4;B:7") == [("A", 1, 4), ("B", 7, 7)]
    assert parse_residue_ranges("3:1") == [(None, 1, 3)]
    assert ranges_to_selection("1-3, 7") == "(resid 1:3) or (resid 7)"
    with pytest.raises(SelectionError):
        parse_residue_ranges("1-x")


def test_group_expansion_nested_and_errors():
    groups = [ResidueGroup("A", "1-5"), ResidueGroup("B", "", "group:A and name P"),
              ResidueGroup("Loop1", "", "group:Loop2"), ResidueGroup("Loop2", "", "group:Loop1")]
    assert expand_groups("group:B or resid 9", groups) == "(((resid 1:5)) and name P) or resid 9"
    assert resolve_selection("A", groups) == "((resid 1:5))"
    assert resolve_selection("sel:name P", groups) == "name P"
    with pytest.raises(SelectionError, match="unknown residue group"):
        expand_groups("group:Nope", groups)
    with pytest.raises(SelectionError, match="nested"):
        expand_groups("group:Loop1", groups)


def test_frame_and_time_specs():
    assert parse_frame_spec("0, 10, 20-50:10, last, -1", 100) == [0, 10, 20, 30, 40, 50, 99]
    with pytest.raises(SelectionError):
        parse_frame_spec("100", 100)
    np.testing.assert_allclose(parse_time_spec("1.5ns, 2000ps, 3", "ns"), [1500, 2000, 3000])
    with pytest.raises(SelectionError):
        parse_time_spec("5 parsecs", "ns")


def test_guess_element():
    assert guess_element("C4'", "A") == "C"
    assert guess_element("HO2'", "G") == "H"
    assert guess_element("MG", "MG", single_atom_residue=True) == "MG"
    assert guess_element("CA", "GLY") == "C"
    assert guess_element("Na+", "Na+", single_atom_residue=True) == "NA"
