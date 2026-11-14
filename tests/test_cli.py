import json
import os
import re

import pytest

from mdatk.cli import apply_override, main
from mdatk.config import Project


def test_apply_override():
    p = Project()
    apply_override(p, "analyses.contacts.cutoff=5")
    assert p.analyses.contacts.cutoff == 5.0 and isinstance(p.analyses.contacts.cutoff, float)
    apply_override(p, "visual.rotation=[10, 20, 30]")
    assert p.visual.rotation == [10, 20, 30]
    apply_override(p, "groups.0.residues=1-5")
    assert p.groups[0].residues == "1-5"
    apply_override(p, 'layout.panels=[{"type": "timeseries", "series": ["rmsd/Complex"]}]')
    assert p.layout.panels[0].type == "timeseries"
    apply_override(p, "inputs.stop=null")
    assert p.inputs.stop is None
    apply_override(p, 'layout.panels.0={"type": "heatmap", "series": ["mg/occupancy"]}')
    assert p.layout.panels[0].type == "heatmap" and p.layout.panels[0].series == ["mg/occupancy"]
    apply_override(p, "visual.title=true")  # strings stay strings
    assert p.visual.title == "true"
    with pytest.raises(ValueError, match="no list element"):
        apply_override(p, "groups.9.residues=1")
    with pytest.raises(ValueError):
        apply_override(p, "analyses.contacts.cutoff=far")
    with pytest.raises(ValueError):
        apply_override(p, "analyses.nope=1")
    with pytest.raises(ValueError):
        apply_override(p, "analyses.rmsd.enabled=yes")


def test_cli_demo_render_and_tools(tmp_path, capsys):
    d = tmp_path / "demo"
    assert main(["demo", str(d), "--n-frames", "10", "--no-solvent"]) == 0
    preset = str(d / "demo.mdatk.toml")
    capsys.readouterr()
    assert main(["list"]) == 0
    assert "contact_map" in capsys.readouterr().out
    assert main(["info", "-p", preset]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["groups"]["Aptamer"]["residues"] == 12 and info["problems"] == []
    args = ["-p", preset, "-q", "--set", "output.video.width=320", "--set", "output.video.height=180",
            "--set", "output.video.dpi=40", "--set", "output.charts.dpi=40",
            "--set", 'output.charts.series=["hbonds/count"]', "-o", str(tmp_path / "out")]
    assert main(["render"] + args) == 0
    out = json.loads(capsys.readouterr().out)
    assert os.path.exists(out["video"]) and len(out["charts"]) == 1 and out["pdbs"]
    from mdatk.projectfile import read_log
    entry = read_log(preset)[-1]  # the run is logged in the project file
    assert entry["step"] == "render" and entry["status"] == "ok" and out["video"] in entry["files"]
    assert main(["charts", "--from-cache", "--formats", "svg"] + args) == 0
    assert json.loads(capsys.readouterr().out)["charts"][0].endswith(".svg")
    assert main(["preview", "--from-cache", "--frame", "3", "--output",
                 str(tmp_path / "p.png")] + args) == 0
    assert os.path.exists(tmp_path / "p.png")
    capsys.readouterr()
    assert main(["export-pdb", "--frame-list", "1", "--times", "", "--selection", "Target"] + args) == 0
    pdbs = json.loads(capsys.readouterr().out)["pdbs"]
    assert len(pdbs) == 1
    assert re.fullmatch(r"MDATK_frame_00001_\d{4}-\d\d-\d\d_\d{6}\.pdb", os.path.basename(pdbs[0]))
    text = open(pdbs[0]).read()
    assert text.count("ATOM") > 100 and " U  " not in text.split("ATOM", 1)[0]
    init = tmp_path / "init.mdatk.toml"
    assert main(["init", str(init), "--top", "x.parm7"]) == 0
    assert Project.load(str(init)).inputs.topology.endswith("x.parm7")


def test_cli_errors(tmp_path, capsys):
    assert main([]) == 2
    assert main(["render", "-q", "-o", str(tmp_path)]) == 1
    assert "topology" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="needs bash")
def test_workflow_run_does_not_start_the_steps_after_a_step_with_errors(tmp_path, capsys):
    """'mdatk workflow run' left a step with errors out without a word, ran the steps reading from it on the
    files its earlier run had left and returned 0."""
    from mdatk.workflows.plan import add_node, connect
    project = Project()
    wf = project.workflow
    wf.project_name, wf.stamp, wf.out_dir = "apt1", "20261113-1200", str(tmp_path / "plan")
    out = [{"name": "result", "kind": "text", "file": "result.txt"}]
    first = add_node(wf, "user-script", settings={"script": 'echo first > "$MDATK_OUT_RESULT"\n', "outputs": out})
    reader = add_node(wf, "user-script", settings={
        "script": 'cat "$MDATK_IN_INPUT" > "$MDATK_OUT_RESULT"\necho ran >> runs.log\n',
        "inputs": [{"name": "input", "kind": "any"}], "outputs": out})
    other = add_node(wf, "user-script", settings={"script": 'echo ran >> runs.log\n'})
    connect(wf, first.id, "result", reader.id, "input")
    path = str(tmp_path / "p.mdatk.toml")
    project.save(path)
    assert main(["workflow", "write", "-p", path, "--run", "-q"]) == 0
    runs = {n: tmp_path / "plan" / f"0{n}_custom" / "runs.log" for n in (2, 3)}
    assert runs[2].read_text() == runs[3].read_text() == "ran\n"
    # the first step gets an error (its script is emptied); its files of the earlier run are still there
    project = Project.load(path)
    project.workflow.nodes[0].settings["script"] = ""
    project.save(path)
    capsys.readouterr()
    assert main(["workflow", "run", "-p", path]) == 1
    err = capsys.readouterr().err
    assert "error: 01 · Custom script has errors and was not run: the script is empty" in err
    assert "02 · Custom script: not started (it reads from a step with errors)" in err
    assert runs[2].read_text() == "ran\n" and runs[3].read_text() == "ran\nran\n"  # the independent step ran
    # asked for by itself, the reader runs (on the files that are there), as before
    assert main(["workflow", "run", "-p", path, "--step", "2"]) == 0
    assert runs[2].read_text() == "ran\nran\n"
