import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from mdatk import tomlio
from mdatk.config import PanelSpec, Project
from mdatk.pipeline import Pipeline
from mdatk.progress import Cancelled, Reporter


def small_project(demo, out_dir):
    p = Project.load(demo["preset"])
    p.output.out_dir = str(out_dir)
    p.output.video.width, p.output.video.height, p.output.video.dpi = 320, 180, 40
    p.output.charts.dpi = 50
    p.output.charts.formats = ["png", "svg"]
    p.output.charts.series = ["rmsd/Complex", "energy/total"]
    p.layout.preset = "equal_6"
    p.layout.panels = [
        PanelSpec("structure", [], "Structure"),
        PanelSpec("timeseries", ["contacts/atom_pairs", "contacts/residue_pairs"], "Contacts",
                  {"progressive": True}),
        PanelSpec("dual_axis", ["contacts/atom_pairs", "hbonds/count"], "Dual"),
        PanelSpec("heatmap", ["mg/occupancy"], "Mg"),
        PanelSpec("contact_map", ["contacts/map"], "Map"),
        PanelSpec("timeseries", ["no_such/series"], "Missing data"),
    ]
    return p


def frame_count(path):
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not available")
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return int(s["nb_read_frames"]), s["width"], s["height"]


def test_full_run_and_cache_reuse(demo_dry, tmp_path):
    p = small_project(demo_dry, tmp_path / "out")
    p.layout.panels[-1] = PanelSpec("timeseries", ["energy/total"], "Energy")
    logs = []
    out = Pipeline(p, Reporter(log_cb=logs.append)).run()
    assert os.path.exists(out.video)
    n, w, h = frame_count(out.video)
    assert (n, w, h) == (24, 320, 180)
    assert len(out.charts) == 4 and all(os.path.exists(c) for c in out.charts)
    log = tomlio.load(out.params_log)
    stamp = log["run"]["timestamp"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d{6}", stamp)
    assert os.path.basename(out.video) == f"MDATK_video_{stamp}.mp4"
    assert all(c.endswith((f"_{stamp}.png", f"_{stamp}.svg")) for c in out.charts)
    assert [os.path.basename(x) for x in out.pdbs] == [
        f"MDATK_frame_{f:05d}_{stamp}.pdb" for f in (0, 12, 23)]
    assert os.path.exists(os.path.join(p.out_dir, "MDATK_params_log.toml"))  # fixed name
    assert out.pdbs and "ATOM" in open(out.pdbs[0]).read()
    assert log["format"] == "mdatk-params-log" and "citation_keys" in log["run"]
    assert log["parameters"]["analyses"]["contacts"]["cutoff"] == p.analyses.contacts.cutoff
    assert Project.load(out.params_log).to_dict() == p.to_dict()  # reproducible

    # second run: every analysis comes from the cache
    logs2 = []
    Pipeline(p, Reporter(log_cb=logs2.append)).run(video=False, pdb=False)
    assert not any(line.startswith("Analysing") for line in logs2)
    assert sum("loaded from cache" in line for line in logs2) == 6  # no video: no trace/proximity

    # only the changed analysis is recomputed; visual changes recompute nothing
    p.analyses.contacts.cutoff = 5.0
    p.visual.rotation = [10.0, 20.0, 30.0]
    p.visual.proximity_cutoff = 6.0
    logs3 = []
    Pipeline(p, Reporter(log_cb=logs3.append)).analyze()
    analysing = [line for line in logs3 if line.startswith("Analysing")]
    assert len(analysing) == 1 and "Pairwise contacts" in analysing[0] and "RMSD" not in analysing[0]


def test_rerender_from_cache_without_trajectory(demo_dry, tmp_path):
    p = small_project(demo_dry, tmp_path / "out2")
    pipe = Pipeline(p)
    pipe.analyze()
    p2 = p.copy()
    p2.inputs.trajectories = ["/nonexistent.nc"]  # the cache path never touches it
    p2.visual.theme = "dark"
    p2.output.charts.series = ["hbonds/count"]
    ds = Pipeline(p2).load_cached_dataset()
    paths = Pipeline(p2).render_charts(ds)
    assert len(paths) == 2
    png = Pipeline(p2).render_preview(ds, 5, str(tmp_path / "still.png"))
    assert os.path.getsize(png) > 1000


def test_cancellation(demo_dry, tmp_path):
    p = small_project(demo_dry, tmp_path / "out3")
    rep = Reporter()
    rep.cancel()
    with pytest.raises(Cancelled):
        Pipeline(p, rep).analyze(force=True)


def test_frame_selection_and_external(demo_dry, tmp_path):
    p = small_project(demo_dry, tmp_path / "out4")
    p.inputs.start, p.inputs.stop, p.inputs.step = 2, 20, 3
    csv = tmp_path / "ext.csv"
    csv.write_text("frame,signal\n" + "".join(f"{k},{k * 2}\n" for k in range(24)))
    from mdatk.config import ExternalSeries
    p.inputs.external = [ExternalSeries(path=str(csv), label="lab", x_unit="frame")]
    ds = Pipeline(p).analyze(["rmsd"])
    np.testing.assert_array_equal(ds.frames, [2, 5, 8, 11, 14, 17])
    np.testing.assert_allclose(ds.series("external/lab.signal").values, ds.frames * 2)


def test_time_step_override_sets_the_time_axis(demo_dry, tmp_path):
    """inputs.dt_ps replaces the time stamps of the trajectory (the demo's NetCDF file stores 10 ps per
    frame): the time axis is frame number x dt, as the PDB export by time reads it."""
    from mdatk.io.pdb_export import frames_from_specs

    p = small_project(demo_dry, tmp_path / "out5")
    p.inputs.start, p.inputs.step = 2, 3
    pipe = Pipeline(p)
    ds = pipe.analyze(["rmsd"])
    np.testing.assert_allclose(ds.time_ps, ds.frames * 10.0)  # no override: the file's time stamps
    p.inputs.dt_ps = 2.0
    pipe = Pipeline(p)
    ds = pipe.analyze(["rmsd"])
    np.testing.assert_array_equal(ds.frames, [2, 5, 8, 11, 14, 17, 20, 23])
    np.testing.assert_allclose(ds.time_ps, ds.frames * 2.0)
    np.testing.assert_allclose(ds.time("ns"), ds.frames * 0.002)
    np.testing.assert_allclose(pipe.analyze([]).time_ps, ds.frames * 2.0)  # no analysis: the same axis
    # the frame exported for a time is the frame the charts show at that time
    assert frames_from_specs(pipe.universe(), "", "10ps", "ps", p.inputs.dt_ps) == [5]
    assert ds.time_ps[list(ds.frames).index(5)] == pytest.approx(10.0)


def test_trace_cache_follows_groups_that_are_not_shown(demo_dry, tmp_path):
    """The trace records the residues of every group (they colour the structure view and choose the residues
    of the 3D representations): editing a group the view does not show must not reuse the cached trace."""
    from mdatk.config import ResidueGroup

    p = small_project(demo_dry, tmp_path / "out6")
    p.groups = [ResidueGroup("Loop", "5-8", "", "#ff0000"), ResidueGroup("RNA", "1-24", "", "#2a78d6")]
    p.visual.structure_groups = ["RNA"]
    tr = Pipeline(p).analyze(["trace"]).results["trace"]
    assert np.flatnonzero(tr.arrays["group_member"][0]).tolist() == [4, 5, 6, 7]
    p.groups[0].residues = "15-18"
    logs = []
    tr = Pipeline(p, Reporter(log_cb=logs.append)).analyze(["trace"]).results["trace"]
    assert np.flatnonzero(tr.arrays["group_member"][0]).tolist() == [14, 15, 16, 17]
    assert np.flatnonzero(tr.arrays["group_index"] == 0).tolist() == [14, 15, 16, 17]
    logs.clear()
    p.groups[0].color = "#00ff00"  # a colour is applied when drawing: the cache is reused
    Pipeline(p, Reporter(log_cb=logs.append)).analyze(["trace"])
    assert any("loaded from cache" in line for line in logs)


def test_results_cached_before_the_fixes_are_not_reused(demo_dry, tmp_path):
    """Cache keys changed with the numbers: analyses whose results were corrected have a new version, and
    results cached with periodic distances or a time step override (whose pair vectors / time axis were
    wrong) get other keys; everything else keeps its key."""
    from mdatk.analysis import ANALYSES

    assert (ANALYSES["hbonds"].version, ANALYSES["contacts"].version, ANALYSES["trace"].version) == (2, 2, 5)
    p = small_project(demo_dry, tmp_path / "out7")
    pipe = Pipeline(p)
    fp = pipe.input_fingerprint(pipe.resolve_inputs())
    assert set(fp) == {"topology", "trajectories", "frames", "dt_ps"}  # as before: these caches stay valid
    plain = ANALYSES["mg"].cache_key(p, fp)
    p.analyses.use_pbc = True
    periodic = ANALYSES["mg"].cache_key(p, fp)
    p.analyses.use_pbc = False
    assert periodic != plain == ANALYSES["mg"].cache_key(p, fp)
    p.inputs.dt_ps = 2.0
    assert pipe.input_fingerprint(pipe.resolve_inputs())["time_axis"] == "frame x dt_ps"


def test_invalid_parameters_are_reported(tmp_path):
    p = Project()
    p.output.out_dir = str(tmp_path)
    from mdatk.pipeline import PipelineError
    with pytest.raises(PipelineError, match="topology"):
        Pipeline(p).run()
