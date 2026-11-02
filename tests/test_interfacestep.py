"""Planner step 'decoy-interface' (mdatk.workflows.interfacestep): ports, checks, the generated script
and a run of it on synthetic models."""

import os
import subprocess
import sys

import pytest

from mdatk.config import PlanNode, ResidueGroup, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.base import PortValue, StepContext
from mdatk.workflows.plan import Plan, add_node, connect, single_step_plan
from test_interfaces import _two_families

STAMP = "20261015-1432"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(settings=None, files=None, groups=(), root="/tmp/mdatk-plan"):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    node = PlanNode(id="x", tool="decoy-interface", step=5,
                    settings=TOOLS["decoy-interface"].normalise(settings or {}), files=dict(files or {}))
    plan = single_step_plan(wf, node, root, list(groups))
    return plan, plan.steps["x"]


def errors(st):
    return [i.message for i in st.issues if i.level == "error"]


def script_of(plan):
    files = plan.generate(["x"])["x"]
    return next(f.content for f in files if f.role == "run")


def test_registered_with_ports_that_follow_the_settings():
    tool = TOOLS["decoy-interface"]
    assert tool.category == "Analysis" and tool.tag == "interface"
    s = tool.defaults()
    assert [p.name for p in tool.output_ports(s)] == ["census", "clusters", "representatives"]
    assert [p.name for p in tool.output_ports(dict(s, cluster=False, support=True))] == ["census", "support"]
    ins = {p.name: p for p in tool.input_ports(s)}
    assert not ins["models"].optional and ins["scores"].optional and ins["scores"].kinds == ("table",)
    assert tool.input_ports(dict(s, models_from="scores"))[0].optional
    assert s["filters"] == "linear_chainbreak <= 5" and s["contact"] == 4.5 and s["hbond"] == 3.5
    assert tool.resources(dict(s, pairs="dssr")) == ["dssr"] and tool.resources(s) == []


@pytest.mark.parametrize("settings, message", [
    ({}, "say which residues make side A"),
    ({"side_a": "A:x"}, "side a: cannot read residue range"),
    ({"side_a": "A", "side_b": "A:3-5"}, "the sides overlap (A / A:3-5)"),
    ({"side_a": "group:Nope"}, "unknown residue group 'Nope'"),
    ({"side_a": "A", "filters": "chainbreak is low"}, "cannot read the filter"),
    ({"side_a": "A", "models_from": "scores"}, "needs a connected scores table"),
    ({"side_a": "A", "mg_min": 8.0}, "the Mg-bridgeable window ends before it starts"),
    ({"side_a": "A", "references": [{"name": "a b", "residues_a": "A:1"}]}, "a name without spaces"),
    ({"side_a": "A", "references": [{"name": "r"}]}, "reference r: give its residues"),
    ({"side_a": "A", "references": [{"name": "r", "residues_a": "A:1"}, {"name": "r", "residues_b": "B"}]},
     "reference names must be unique"),
    ({"side_a": "A", "support": True}, "do not say which residues FARFAR2 kept fixed"),
    ({"side_a": "A", "support": True, "rebuilt_from": "fixed"}, "type the fixed residues"),
    ({"side_a": "A", "support": True, "rebuilt_from": "rebuilt", "rebuilt": "A:1", "run": "four"},
     "a whole number of residues, or 'auto'"),
    ({"side_a": "A", "python": ""}, "give the Python interpreter"),
])
def test_checks(settings, message):
    _plan, st = step(settings, {"models": "/x/m1.pdb;/x/m2.pdb"})
    assert any(message in m for m in errors(st)), errors(st)


def test_filter_terms_are_checked_against_an_existing_table(tmp_path):
    table = tmp_path / "scores.tsv"
    table.write_text("run\ttag\tpdb\tscore\n1\tS_1\t\t-3\n")
    _plan, st = step({"side_a": "A"}, {"models": "/x/m.pdb", "scores": str(table)})
    assert "the scores table has no column linear_chainbreak" in errors(st)
    _plan, st = step({"side_a": "A", "filters": "score < 0"}, {"models": "/x/m.pdb", "scores": str(table)})
    assert errors(st) == []
    _plan, st = step({"side_a": "A"}, {"models": "/x/m.pdb"})
    assert "the score filters are ignored without a scores table" in [i.message for i in st.issues]


def test_groups_become_ranges_in_the_script_and_the_first_model_is_checked(tmp_path):
    paths = _two_families(str(tmp_path / "models"))
    groups = [ResidueGroup("Duplex", "A:1-8, B:9-16"), ResidueGroup("Partner", "", "chainID C")]
    plan, st = step({"side_a": "group:Duplex", "side_b": "group:Partner,D",
                     "references": [{"name": "groove", "residues_a": "group:Duplex",
                                     "residues_b": "C:17-22"}]},
                    {"models": ";".join(paths)}, groups, str(tmp_path / "plan"))
    assert errors(st) == []
    sh = script_of(plan)
    assert "--side-a A:1-8,B:9-16 --side-b C,D" in sh and "--reference groove=A:1-8,B:9-16/C:17-22" in sh
    assert "group:" not in sh  # the script runs without the project file
    _plan, st = step({"side_a": "A", "side_b": "Q"}, {"models": ";".join(paths)}, (), str(tmp_path / "plan"))
    assert "first model: side B matches no nucleotide of the model" in [i.message for i in st.issues]


def test_in_a_farfar2_plan(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    ff = add_node(wf, "farfar2", settings={"sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))"})
    models = add_node(wf, "farfar2-models")
    iface = add_node(wf, "decoy-interface", settings={"side_a": "A", "side_b": "B", "support": True,
                                                      "rebuilt_from": "fixed", "fixed": "A:1-2 B:11-12",
                                                      "pairs": "dssr", "dssr_exe": "/opt/x3dna/x3dna-dssr"})
    connect(wf, ff.id, "silent", models.id, "silent")
    connect(wf, models.id, "models", iface.id, "models")
    plan = Plan(wf, str(tmp_path))
    st = plan.steps[iface.id]
    assert [i.message for i in st.issues if i.level == "error"] == []
    assert "x3dna-dssr needs its own (free academic) licence and binary" in [i.message for i in st.issues]
    files = plan.generate([iface.id])[iface.id]
    sh = next(f.content for f in files if f.role == "run")
    assert "  ../02_farfar2-models/apt1_02_model-01_20261015-1432.pdb" in sh
    assert "--pairs dssr --dssr-exe /opt/x3dna/x3dna-dssr" in sh
    assert ("decoys support --census apt1_03_census_20261015-1432.tsv --side-a A --side-b B "
            "--fixed A:1-2,B:11-12") in sh
    assert "Please cite" in sh and "Lu" in sh  # DSSR's citation in the header
    outs = st.outputs
    assert outs["representatives"].paths[0] == "03_interface/apt1_03_rep-01_20261015-1432.pdb"
    assert outs["census"].kind == "table"
    assert outs["support"].paths == ["03_interface/apt1_03_support_20261015-1432.tsv"]
    report = plan.write([iface.id])
    for path in report.written:
        if path.endswith(".sh"):
            proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr


def test_rebuilt_residues_from_the_fixed_residues_of_the_models_upstream():
    """When the models' data say what FARFAR2 kept fixed (its -input_res), the rest is rebuilt."""
    tool = TOOLS["decoy-interface"]
    s = tool.normalise({"side_a": "A,B", "side_b": "C", "support": True})
    for data, expected in (({"input_res": "A:1-6 B:39-44"}, "--fixed A:1-6,B:39-44"),
                           ({"input_res": ""}, "--fixed ''")):
        ctx = StepContext("apt1", 4, STAMP, tool.tag,
                          {"models": PortValue(["02_m/m1.pdb"], "pdb", data, "n1", "models")}, "/tmp/p")
        assert [i for i in tool.check(s, ctx) if i.level == "error"] == []
        sh = next(f.content for f in tool.generate(s, ctx) if f.role == "run")
        assert expected in sh


def test_scores_table_from_a_plan_step_and_every_decoy(tmp_path):
    tool = TOOLS["decoy-interface"]
    s = tool.normalise({"side_a": "A", "models_from": "scores", "drop_unscored": True,
                        "filters": "linear_chainbreak <= 5\nscore < 0"})
    ctx = StepContext("apt1", 4, STAMP, tool.tag,
                      {"scores": PortValue(["02_farfar2-models/apt1_02_scores.tsv"], "table", {}, "n2",
                                           "models")},
                      str(tmp_path))
    assert [i.message for i in tool.check(s, ctx) if i.level == "error"] == []
    sh = next(f.content for f in tool.generate(s, ctx) if f.role == "run")
    assert ("decoys census --side-a A --cutoff 4.5 --hbond 3.5 --pairs builtin "
            "--scores ../02_farfar2-models/apt1_02_scores.tsv --plan-root .. --from-scores "
            "--filter 'linear_chainbreak<=5' --filter 'score<0' --drop-unscored") in sh
    assert "MODELS=(" not in sh


def test_the_written_script_runs(tmp_path):
    """Census (score filter), census of a reference model, clusters with representatives and support."""
    paths = _two_families(str(tmp_path / "models"))
    scores = tmp_path / "scores.tsv"
    scores.write_text("run\ttag\tpdb\tscore\tlinear_chainbreak\n" + "".join(
        f"r1\tS_{k}\tmodels/decoy_{k}.pdb\t{-k}\t{9.0 if k == 1 else 0.2}\n" for k in range(6)))
    settings = {"side_a": "A,B", "side_b": "C,D", "python": sys.executable, "min_size": 2, "n_reps": 2,
                "support": True, "rebuilt_from": "rebuilt", "rebuilt": "A:3-6,C:18-20", "run": "auto",
                "pair_credit": "contacts", "references": [{"name": "end", "residues_a": "B:16",
                                                           "residues_b": "C:22"}]}
    plan, st = step(settings, {"models": ";".join(paths), "scores": str(scores), "references": paths[0]},
                    root=str(tmp_path / "plan"))
    assert errors(st) == []
    report = plan.write(["x"])
    sh = next(p for p in report.written if p.endswith("_interface_20261015-1432.sh"))
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(["bash", sh], capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    folder = os.path.dirname(sh)
    have = set(os.listdir(folder))
    for tag in ("census", "ref-census", "clusters", "members", "support", "support-residues",
                "support-profile"):
        assert f"apt1_05_{tag}_20261015-1432.tsv" in have, tag
    assert {"apt1_05_rep-01_20261015-1432.pdb", "apt1_05_rep-02_20261015-1432.pdb"} <= have
    census = open(os.path.join(folder, "apt1_05_census_20261015-1432.tsv")).read().splitlines()
    assert len(census) == 6 and "decoy_1.pdb" not in "".join(census)  # the chainbreak filter
    clusters = open(os.path.join(folder, "apt1_05_clusters_20261015-1432.tsv")).read()
    assert "share_end" in clusters and "share_decoy_0" in clusters  # typed and model references
    assert "kept 5, dropped 1 of 6" in proc.stdout and "run threshold" in proc.stdout
