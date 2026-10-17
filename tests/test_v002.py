"""Features added in 0.0.2a: per-panel group pairs, labels, backbone RMSD, integer axes,
residue tags, timestamps and conda environments for cpptraj."""

import os
import re
import stat
import sys
import textwrap

import numpy as np
import pytest

from mdatk.analysis.base import instance_project, key_label, parse_instance, split_key
from mdatk.config import CleaningParams, PanelSpec, Project
from mdatk.pipeline import Pipeline, stamped
from mdatk.progress import Reporter


def test_instance_keys_and_labels():
    p = Project()
    assert parse_instance("contacts@Target+Aptamer") == ("contacts", ("Target", "Aptamer"))
    assert split_key("hbonds@A+B/count") == ("hbonds@A+B", "count")
    q = instance_project(p, "contacts@Target+Aptamer")
    assert (q.analyses.contacts.group_a, q.analyses.contacts.group_b) == ("Target", "Aptamer")
    assert p.analyses.contacts.group_a == "Aptamer"  # the original project is untouched
    assert key_label("contacts@Target+Aptamer/atom_pairs", p) == "Target–Aptamer heavy-atom contacts"
    assert key_label("rmsd@Target/Backbone", p) == "RMSD Backbone"
    with pytest.raises(ValueError, match="unknown residue group"):
        instance_project(p, "contacts@Nope+Target")
    with pytest.raises(ValueError, match="takes 2 group"):
        instance_project(p, "contacts@Target")
    with pytest.raises(ValueError):
        instance_project(p, "trace@Target")
    p.groups[0].name = "Bad+Name"
    assert any("may only contain" in m for m in p.validate())


def test_panel_group_pairs_are_computed_and_cached_separately(demo_dry, tmp_path):
    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "out")
    p.layout.preset = "equal_3"
    p.layout.panels = [PanelSpec("timeseries", ["contacts/atom_pairs"]),
                       PanelSpec("timeseries", ["contacts@Target+Aptamer/atom_pairs"]),
                       PanelSpec("timeseries", ["rmsd@Target/Backbone", "contacts@Nope+X/count"])]
    logs = []
    pipe = Pipeline(p, Reporter(log_cb=logs.append))
    names = pipe.required_analyses(video=True, charts=False)
    assert "contacts" in names and "contacts@Target+Aptamer" in names and "rmsd@Target" in names
    assert not any("Nope" in n for n in names)
    assert any("skipping data source 'contacts@Nope+X'" in line for line in logs)
    ds = pipe.analyze(names)
    a = ds.series("contacts/atom_pairs")
    b = ds.series("contacts@Target+Aptamer/atom_pairs")
    np.testing.assert_array_equal(a.values, b.values)  # symmetric criterion, swapped groups
    assert b.label.startswith("Target–Aptamer") and a.label.startswith("Aptamer–Target")
    assert ds.series("rmsd@Target/Backbone").label == "RMSD Backbone (Target)"
    assert pipe.analysis_keys["contacts"] != pipe.analysis_keys["contacts@Target+Aptamer"]
    cached = os.listdir(p.cache_dir)
    assert any(f.startswith("contacts@Target+Aptamer-") and f.endswith(".npz") for f in cached)
    logs.clear()
    Pipeline(p, Reporter(log_cb=logs.append)).analyze(names)
    assert not any(line.startswith("Analysing") for line in logs)


def test_default_backbone_rmsd_and_integer_count_axes(demo_dry, tmp_path):
    from matplotlib.ticker import MaxNLocator
    from mdatk.render.video import build_figure

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "out")
    assert [t.label for t in p.analyses.rmsd.targets] == ["Complex", "Backbone"]
    ds = Pipeline(p).analyze(["rmsd", "stacking", "hbonds"])
    bb = ds.series("rmsd/Backbone").values
    assert bb[0] == pytest.approx(0.0, abs=1e-6) and np.all(np.isfinite(bb)) and bb.max() > 0.2
    p.layout.preset = "equal_3"
    p.layout.panels = [PanelSpec("timeseries", ["stacking/count"]),
                       PanelSpec("timeseries", ["rmsd/Backbone"]),
                       PanelSpec("dual_axis", ["hbonds/count", "rmsd/Complex"])]
    fig, canvas, panels = build_figure(ds, p, 600, 300, 50)
    loc = panels[0].ax.yaxis.get_major_locator()
    assert isinstance(loc, MaxNLocator) and loc._integer
    assert all(float(t).is_integer() for t in panels[0].ax.get_yticks())
    assert not getattr(panels[1].ax.yaxis.get_major_locator(), "_integer", False)
    assert panels[2].ax.yaxis.get_major_locator()._integer  # count axis of a dual-axis panel
    assert panels[0].ax.get_title(loc="left") == "π–π stacked pairs"  # automatic title
    assert panels[1].ax.get_title(loc="left") == "RMSD Backbone"


def test_residue_tags_and_rotation(demo_dry, tmp_path):
    from mdatk.render.video import build_figure

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "out")
    p.visual.label_every = 5
    p.visual.label_residues = "1, 12-13"
    p.layout.preset = "single"
    p.layout.panels = [PanelSpec("structure")]
    ds = Pipeline(p).analyze(["trace", "proximity"])
    fig, canvas, (panel,) = build_figure(ds, p, 400, 300, 50)
    labels = sorted(int(t.get_text()) for t in panel.tags)
    assert labels == [1, 5, 10, 12, 13, 15, 20]
    before = panel.view.copy()
    xlim = panel.ax.get_xlim()
    panel.set_rotation([0, 90, 0])
    assert not np.allclose(before, panel.view)
    np.testing.assert_allclose(np.linalg.norm(before, axis=-1), np.linalg.norm(panel.view, axis=-1))
    assert panel.ax.get_xlim() == xlim  # framing does not depend on the orientation


def test_timestamps_can_be_disabled(demo_dry, tmp_path):
    assert stamped("MDATK_video.mp4", "2026-10-15_143205") == "MDATK_video_2026-10-15_143205.mp4"
    assert stamped("MDATK_video.mp4", "") == "MDATK_video.mp4"
    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "out")
    p.output.timestamp_files = False
    p.output.pdb.times = ""
    os.makedirs(p.out_dir)
    pdbs = Pipeline(p).export_pdbs()
    assert [os.path.basename(x) for x in pdbs] == ["MDATK_frame_00000.pdb", "MDATK_frame_00023.pdb"]


FAKE_CONDA = textwrap.dedent("""\
    #!/bin/bash
    # stand-in for `conda run --no-capture-output -p PREFIX cmd ...`
    [ "$1" = run ] || exit 2; shift
    [ "$1" = --no-capture-output ] && shift
    flag=$1; env=$2; shift 2
    echo "fake-conda activated $env"
    exec "$env/bin/$1" "${@:2}"
    """)
FAKE_CPPTRAJ = textwrap.dedent("""\
    #!{python}
    import re, sys
    script = open(sys.argv[sys.argv.index("-i") + 1]).read()
    for path in re.findall(r'(?:trajout|parmout|parmwrite out) "([^"]+)"', script):
        open(path, "w").write("fake")
    print("CPPTRAJ: fake inside env")
    """)


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def conda_root(tmp_path, monkeypatch):
    root = tmp_path / "miniforge3"
    for env in ("AmberTools26", "other"):
        (root / "envs" / env / "bin").mkdir(parents=True)
    (root / "bin").mkdir()
    _exe(root / "envs" / "AmberTools26" / "bin" / "cpptraj", FAKE_CPPTRAJ.format(python=sys.executable))
    _exe(root / "bin" / "conda", FAKE_CONDA)
    from mdatk.cleaning import conda
    monkeypatch.setattr(conda, "COMMON_ROOTS", (str(root),))
    for var in ("CONDA_EXE", "CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(conda, "_registered_prefixes", lambda: [])
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    return root


def test_conda_detection_by_cpptraj_binary(conda_root):
    from mdatk.cleaning import conda

    envs = conda.find_cpptraj_envs()
    assert [e.name for e in envs] == ["AmberTools26"]
    assert conda.detect_ambertools_env().prefix == str((conda_root / "envs" / "AmberTools26").resolve())
    assert conda.resolve_env("AmberTools26").root == str(conda_root.resolve())
    assert conda.find_conda_exe(env=envs[0]) == str(conda_root / "bin" / "conda")
    assert conda.run_command("/x/micromamba", "env", ["cpptraj"]) == ["/x/micromamba", "run", "-n", "env", "cpptraj"]


def test_cpptraj_runs_inside_conda_env(conda_root, tmp_path):
    from mdatk.cleaning import clean, cpptraj_command

    top, traj = tmp_path / "s.parm7", tmp_path / "s.nc"
    top.write_text("x")
    traj.write_text("x")
    p = CleaningParams(enabled=True, conda_env="AmberTools26")
    cmd = cpptraj_command(p)
    assert cmd[:3] == [str(conda_root / "bin" / "conda"), "run", "--no-capture-output"]
    assert cmd[-1] == "cpptraj"
    lines = []
    res = clean(p, str(top), [str(traj)], str(tmp_path / "cache"), log_cb=lines.append)
    assert any("fake-conda activated" in l for l in lines)
    assert any("fake inside env" in l for l in lines)
    assert os.path.exists(res.trajectory)
    # no env configured and cpptraj not on PATH: the environment is found automatically
    auto = []
    assert cpptraj_command(CleaningParams(enabled=True), auto.append)[-1] == "cpptraj"
    assert any("AmberTools26" in l for l in auto)
    with pytest.raises(Exception, match="not found"):
        cpptraj_command(CleaningParams(enabled=True, conda_env="missing", conda_exe="/nonexistent"))


def test_example_dataset_matches_layout():
    from mdatk.render.example import example_dataset
    from mdatk.render.video import build_figure

    p = Project()
    p.layout.preset = "large_plus_4"
    p.layout.panels = [PanelSpec("structure"), PanelSpec("timeseries", ["stacking@Aptamer+Target/count"]),
                       PanelSpec("heatmap", ["mg/occupancy"]), PanelSpec("contact_map", ["hbonds/map"]),
                       PanelSpec("dual_axis", ["rmsd/Complex", "energy/total"])]
    ds = example_dataset(p)
    assert set(ds.series_keys()) == {"stacking@Aptamer+Target/count", "rmsd/Complex", "energy/total"}
    counts = ds.series("stacking@Aptamer+Target/count").values
    assert np.all(counts == np.round(counts))
    assert ds.matrix("mg/occupancy").values.max() <= 2
    assert ds.pairmap("hbonds/map").shape == (12, 12)
    fig, canvas, panels = build_figure(ds, p, 640, 360, 40, watermark="Example")
    assert all(pn.ok for pn in panels)


def test_latest_video(tmp_path):
    gi = pytest.importorskip("gi")
    del gi
    from mdatk.gui.desktop import latest_video

    a = tmp_path / "MDATK_video_2026-01-01_000000.mp4"
    b = tmp_path / "MDATK_video_2026-01-02_000000.mp4"
    a.write_text("a")
    b.write_text("b")
    os.utime(a, (1, 1))
    assert latest_video(str(tmp_path), "MDATK_video.mp4") == str(b)
    assert latest_video(str(tmp_path / "none"), "MDATK_video.mp4") is None
    assert re.match(r".*\.mp4$", latest_video(str(tmp_path), "MDATK_video.mp4"))
