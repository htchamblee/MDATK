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
    preset = str(d / "demo_preset.json")
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
    init = tmp_path / "init.json"
    assert main(["init-params", str(init), "--top", "x.parm7"]) == 0
    assert Project.load(str(init)).inputs.topology.endswith("x.parm7")


def test_cli_errors(tmp_path, capsys):
    assert main([]) == 2
    assert main(["render", "-q", "-o", str(tmp_path)]) == 1
    assert "topology" in capsys.readouterr().err
