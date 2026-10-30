"""Selection and group problems say where they are: structured problems with a field path, a GUI
page and a readable location, and messages that name the offending group, clause or token."""

import copy
import json
import pickle

import numpy as np
import pytest
from conftest import make_universe, micro_project

from mdatk.config import PanelSpec, Project, RepSpec, ResidueGroup, RMSDTarget
from mdatk.problems import TABS, Problem, describe, problem
from mdatk.selections import (SelectionError, check_selection, expand_groups, referenced_groups, select,
                              suggest)


@pytest.fixture
def universe():
    atoms = [{"name": n, "resid": r, "resname": "G", "element": n[0]}
             for r in range(1, 5) for n in ("P", "C4'", "N1")]
    return make_universe(atoms, [[[i, 0.0, 0.0] for i in range(len(atoms))]])


GROUPS = [ResidueGroup("Aptamer", "1-2"), ResidueGroup("Target", "100-112"),
          ResidueGroup("Wrapper", "", "group:Aptamr and name P")]


# ------------------------------------------------------------------ structured problems
def test_problem_is_its_full_text_and_carries_its_place():
    p = Project()
    x = problem("analyses.hbonds.group_a", "unknown residue group 'X'", p, token="X")
    assert isinstance(x, str) and str(x) == "Analyses › Hydrogen bonds › Group A: unknown residue group 'X'"
    assert (x.field, x.tab, x.key, x.level, x.token) == (
        "analyses.hbonds.group_a", "analyses", "analyses:analyses.hbonds.group_a", "error", "X")
    assert "unknown residue group" in x and x.message == "unknown residue group 'X'"
    for y in (copy.deepcopy(x), pickle.loads(pickle.dumps(x))):
        assert y == x and y.field == x.field and y.location == x.location and y.token == "X"
    assert json.loads(json.dumps([x])) == [str(x)]
    assert x.as_dict()["location"] == "Analyses › Hydrogen bonds › Group A"
    assert str(Problem("plain")) == "plain" and Problem("plain").key == ""


def test_locations_are_readable_paths():
    p = Project()
    p.visual.representations = [RepSpec(), RepSpec(groups=["Nope"])]
    assert describe("layout.panels.3.groups", p) == "Layout & panels › Panel 4 (Contact map) › Groups"
    assert describe("visual.representations.1.groups", p) == "Structure view › Representation 2 › Groups"
    assert describe("groups.1.residues", p) == "Residue groups › Group 2 (Target) › Residues"
    assert describe("analyses.rmsd.targets.0.selection", p) == \
        "Analyses › RMSD › Target 1 (Complex) › Selection"
    assert describe("output.pdb.selection") == "Outputs › Frame PDBs › Atom subset"
    assert describe("analyses.mg.inner_cutoff") == "Analyses › Mg²⁺ coordination and occupancy › Inner cutoff"
    # the GUI pages and their titles
    from mdatk.problems import tab_of
    assert {tab_of(f) for f in ("inputs.topology", "groups.0.name", "analyses.mg.mg_selection",
                                "visual.proximity_set", "density.map_path", "layout.preset",
                                "output.video.fps")} == set(TABS)


# ------------------------------------------------------------------ messages
def test_unknown_group_names_it_lists_the_groups_and_suggests():
    with pytest.raises(SelectionError) as info:
        expand_groups("group:Aptamr and name P", GROUPS)
    exc = info.value
    assert (exc.kind, exc.token, exc.suggestion) == ("unknown_group", "Aptamr", "Aptamer")
    assert str(exc) == ("unknown residue group 'Aptamr' (defined: Aptamer, Target, Wrapper); "
                        "did you mean 'Aptamer'?")
    with pytest.raises(SelectionError, match="'Aptamr' in the selection of group 'Wrapper'") as info:
        expand_groups("group:Wrapper", GROUPS)
    assert info.value.group == "Wrapper"  # the definition at fault
    assert suggest("target", ["Aptamer", "Target"]) == "Target" and suggest("xyz", ["Aptamer"]) is None
    loops = [ResidueGroup("L1", "", "group:L2"), ResidueGroup("L2", "", "group:L1")]
    with pytest.raises(SelectionError, match="nested in a circle: group:L1 → group:L2 → group:L1"):
        expand_groups("group:L1", loops)
    assert referenced_groups("group:Wrapper or resid 9", GROUPS + [ResidueGroup("Aptamr", "3")]) == \
        ["Wrapper", "Aptamr"]


def test_parse_error_shows_original_expanded_and_mdanalysis_message(universe):
    with pytest.raises(SelectionError) as info:
        select(universe, "group:Aptamer and nme P", GROUPS)
    exc = info.value
    assert exc.kind == "syntax" and exc.token == "nme" and exc.expanded == "((resid 1:2)) and nme P"
    msg = str(exc)
    assert "'group:Aptamer and nme P'" in msg and "(expanded: '((resid 1:2)) and nme P')" in msg
    assert "MDAnalysis says: Unknown selection token: 'nme'" in msg and "did you mean 'name'?" in msg
    with pytest.raises(SelectionError, match="to use the residue group write group:Aptamer"):
        select(universe, "Aptamer and name P", GROUPS)
    with pytest.raises(SelectionError, match="a parenthesis is not closed"):
        select(universe, "(resid 1 or resid 2", GROUPS)
    # the same check without a structure
    with pytest.raises(SelectionError, match="Unknown selection token: 'nme'"):
        check_selection("group:Aptamer and nme P", GROUPS)
    assert check_selection("Aptamer", GROUPS) == "((resid 1:2))"


def test_empty_match_names_the_group_or_clause_that_matched_nothing(universe):
    assert select(universe, "group:Aptamer or group:Target", GROUPS).n_atoms == 6  # Target adds nothing
    with pytest.raises(SelectionError) as info:
        select(universe, "group:Target and name P", GROUPS)
    assert info.value.kind == "empty" and info.value.token == "Target"
    assert ("group 'Target' (residues '100-112') matches no atoms - the structure's residues are "
            "numbered 1-4") in str(info.value)
    with pytest.raises(SelectionError) as info:
        select(universe, "group:Aptamer and name PX", GROUPS)
    assert info.value.token == "name PX" and "the part 'name PX' matches no atoms" in str(info.value)
    with pytest.raises(SelectionError, match="each part matches atoms, but not in this combination"):
        select(universe, "group:Aptamer and resid 3", GROUPS)
    assert select(universe, "name PX", GROUPS, allow_empty=True).n_atoms == 0


# ------------------------------------------------------------------ validation
def test_validation_problems_carry_field_and_location(tmp_path):
    p = Project()
    p.groups.append(ResidueGroup("Bad", "1-x"))
    p.groups.append(ResidueGroup("UsesBad", "", "group:Bad and name P"))
    p.analyses.hbonds.group_b = "Tagret"
    p.analyses.rmsd.targets.append(RMSDTarget("X", "group:Aptamer and nme P"))
    p.analyses.stacking.selection = "group:Bad"  # broken group: reported at its definition only
    p.visual.representations = [RepSpec(groups=["aptamer"])]
    p.visual.proximity_set = "Targt"
    p.visual.structure_groups = ["Aptamer", "Nope"]
    p.layout.panels.append(PanelSpec("timeseries", ["contacts@Aptamer+Tgt/atom_pairs"]))
    by_field = {x.field: x for x in p.problems()}
    hb = by_field["analyses.hbonds.group_b"]
    assert str(hb) == ("Analyses › Hydrogen bonds › Group B: unknown residue group 'Tagret' (defined: "
                       "Aptamer, Target, Bad, UsesBad); did you mean 'Target'?")
    assert hb.key == "analyses:analyses.hbonds.group_b" and hb.token == "Tagret"
    assert "Unknown selection token: 'nme'" in by_field["analyses.rmsd.targets.2.selection"]
    assert by_field["groups.2.residues"].location == "Residue groups › Group 3 (Bad) › Residues"
    assert "invalid residue range item '1-x'" in by_field["groups.2.residues"]
    assert "groups.3.selection" not in by_field and "analyses.stacking.selection" not in by_field
    assert "did you mean 'Aptamer'?" in by_field["visual.representations.0.groups"]
    assert "did you mean 'Target'?" in by_field["visual.proximity_set"]
    warned = by_field["visual.structure_groups"]
    assert warned.level == "warning" and "left out of the structure view" in warned
    panel = by_field["layout.panels.4.groups"]
    assert panel.level == "warning" and panel.location == "Layout & panels › Panel 5 (Time series) › Groups"
    errors = p.validate()
    assert all(x.level == "error" for x in errors)
    assert "visual.structure_groups" not in {x.field for x in errors}
    assert any("topology" in x for x in errors)  # the old string API
    # a disabled analysis is not checked; a proximity set that is not used is only a warning
    p.analyses.hbonds.enabled = False
    p.visual.glow_mode, p.visual.color_mode = "none", "groups"
    fields = {x.field: x.level for x in p.problems()}
    assert "analyses.hbonds.group_b" not in fields and fields["visual.proximity_set"] == "warning"


def test_pipeline_refuses_with_located_problems(tmp_path):
    from mdatk.pipeline import Pipeline, PipelineError

    p = Project()
    p.output.out_dir = str(tmp_path)
    p.analyses.contacts.group_a = "Aptamr"
    with pytest.raises(PipelineError, match="Pairwise contacts › Group A: unknown residue group") as info:
        Pipeline(p).run()
    assert {"inputs.topology", "analyses.contacts.group_a"} <= {x.field for x in info.value.problems}


# ------------------------------------------------------------------ run time
def test_run_time_errors_name_their_setting():
    from conftest import run_one

    atoms = [{"name": "P", "resid": r, "resname": "G", "element": "P"} for r in range(1, 5)]
    u = make_universe(atoms, np.zeros((2, 4, 3)) + np.arange(4)[None, :, None])
    p = micro_project({"A": "1-2", "Empty": "50-60"})
    p.analyses.contacts.group_a, p.analyses.contacts.group_b = "A", "Empty"
    with pytest.raises(SelectionError) as info:
        run_one("contacts", u, p)
    exc = info.value
    assert exc.field == "analyses.contacts.group_b" and exc.kind == "empty" and exc.token == "Empty"
    assert str(exc).startswith("Analyses › Pairwise contacts › Group B: selection 'group:Empty' matched no "
                               "atoms: group 'Empty' (residues '50-60') matches no atoms")
    assert exc.problem().key == "analyses:analyses.contacts.group_b"
    p.analyses.mg.ligand_selection = "group:A and nme O"
    with pytest.raises(SelectionError) as info:
        run_one("mg", u, p)
    assert info.value.field == "analyses.mg.ligand_selection" and "MDAnalysis says" in str(info.value)


def test_run_time_errors_of_panel_group_pairs_point_at_the_panel():
    from mdatk.analysis import ANALYSES, AnalysisContext, run_analyses
    from mdatk.analysis.base import instance_project

    atoms = [{"name": "P", "resid": r, "resname": "G", "element": "P"} for r in range(1, 5)]
    u = make_universe(atoms, np.zeros((2, 4, 3)) + np.arange(4)[None, :, None])
    p = micro_project({"A": "1-2", "B": "3-4", "Empty": "50-60"})
    p.layout.panels = [PanelSpec("structure"), PanelSpec("timeseries", ["contacts@A+Empty/atom_pairs"])]
    inst = "contacts@A+Empty"
    ctx = AnalysisContext(u, instance_project(p, inst))
    with pytest.raises(SelectionError) as info:
        run_analyses(ctx, [ANALYSES["contacts"](ctx, inst)], np.arange(2))
    assert info.value.field == "layout.panels.1.groups"
    assert str(info.value).startswith("Layout & panels › Panel 2 (Time series) › Groups: "
                                      "selection 'group:Empty'")


def test_pdb_export_selection_errors_are_located(demo_dry, tmp_path):
    from mdatk.pipeline import Pipeline

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path)
    p.output.pdb.frames = "0"
    p.output.pdb.selection = "Aptamer, Tget"
    assert {x.field: x.level for x in p.problems()}["output.pdb.selection"] == "error"
    p.output.pdb.enabled = False  # not exported with the outputs: only a warning
    assert {x.field: x.level for x in p.problems()}["output.pdb.selection"] == "warning"
    with pytest.raises(SelectionError) as info:
        Pipeline(p).export_pdbs()
    assert info.value.field == "output.pdb.selection"
    assert str(info.value) == ("Outputs › Frame PDBs › Atom subset: unknown residue group 'Tget' (defined: "
                               "Aptamer, Target); did you mean 'Target'?")
    p.output.pdb.selection, p.output.pdb.frames = "", "0, 9999"
    with pytest.raises(SelectionError, match=r"^Outputs › Frame PDBs › Frames: frame\(s\) \[9999\] outside"):
        Pipeline(p).export_pdbs()
