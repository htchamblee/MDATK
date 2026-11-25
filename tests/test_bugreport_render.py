"""Regression tests for the rendering, chart, preview and video items of the external bug report on 0.1.1
(MDATK-BUG-031 to 038, 057 to 064 and 075 to 082; the item is named in each test)."""

import logging
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from mdatk.config import PanelSpec, Project, RepSpec, ResidueGroup
from mdatk.data import AnalysisResult, CSRBuilder, Dataset
from mdatk.render import video
from mdatk.render.example import example_dataset
from mdatk.render.video import build_figure

warnings.filterwarnings("ignore")


def project_with(panels, preset="single"):
    p = Project()
    p.layout.preset = preset
    p.layout.panels = list(panels)
    return p


def panel_text(panel) -> str:
    """Everything a panel's axes say (the "No data: ..." message of a panel that could not be drawn)."""
    return " | ".join(t.get_text() for t in panel.ax.texts)


def matrix_dataset(values, times, classes=None, name="contacts", out="per"):
    """Dataset with one per-residue matrix (rows x frames) at the given frame times (ps)."""
    values = np.asarray(values)
    ds = Dataset(np.arange(len(times)), np.asarray(times, dtype=float), time_unit="ps")
    res = AnalysisResult(name, "k", {}, {}, {})
    res.arrays["m"] = values
    res.meta["rows"] = [str(k + 1) for k in range(values.shape[0])]
    if classes:
        res.add_matrix(out, "m", "rows", "Per residue", "class", classes=list(classes))
    else:
        res.add_matrix(out, "m", "rows", "Per residue", "count")
    ds.add(res)
    return ds


# ---------------------------------------------------------------------------
# MDATK-BUG-031: no analysed frames
# ---------------------------------------------------------------------------


def empty_dataset():
    """A dataset of zero frames with every kind of output and a trace (an empty cached dataset)."""
    ds = Dataset(np.zeros(0, int), np.zeros(0))
    res = AnalysisResult("rmsd", "k", {}, {}, {})
    res.arrays.update(a=np.zeros(0), b=np.zeros(0), m=np.zeros((3, 0)))
    res.add_series("Complex", "a", "RMSD", "Å")
    res.add_series("Other", "b", "RMSD 2", "Å")
    res.meta.update(rows=["1", "2", "3"], r=["1", "2"], c=["1", "2"])
    res.add_matrix("per", "m", "rows", "Per residue", "count")
    res.arrays.update(CSRBuilder().arrays("map"))
    res.add_pairmap("map", "map", "r", "c", "Map")
    ds.add(res)
    tr = AnalysisResult("trace", "k", {}, {}, {})
    tr.arrays.update(coords=np.zeros((0, 4, 3)), ion_coords=np.zeros((0, 0, 3)), chain=np.zeros(4, int),
                     group_index=np.zeros(4, int), group_member=np.ones((1, 4), bool), is_point=np.ones(4, bool),
                     links=np.array([[0, 1], [1, 2], [2, 3]]), resindex=np.arange(4), resid=np.arange(1, 5),
                     resname=np.array(["G", "C", "A", "U"]), chain_name=np.array(["A"] * 4))
    ds.add(tr)
    return ds


@pytest.mark.parametrize("ptype, series", [
    ("timeseries", ["rmsd/Complex"]), ("dual_axis", ["rmsd/Complex", "rmsd/Other"]), ("heatmap", ["rmsd/per"]),
    ("contact_map", ["rmsd/map"]), ("structure", []), ("sequence", []), ("secondary", [])])
def test_031_panels_of_a_dataset_without_frames_show_no_data(ptype, series):
    p = project_with([PanelSpec(ptype, series, "")])
    p.visual.glow_mode = "none"
    _fig, _canvas, panels = build_figure(empty_dataset(), p, 400, 300, 100)  # was: IndexError
    assert not panels[0].ok and "No data: no frames" in panel_text(panels[0])


def test_031_rendering_without_frames_stops_with_a_message(tmp_path, monkeypatch):
    from mdatk.render.charts import render_charts

    def no_ffmpeg(*_a, **_k):
        raise AssertionError("FFmpeg must not be started for zero frames")

    monkeypatch.setattr(video, "FFmpegWriter", no_ffmpeg)
    p = project_with([PanelSpec("sequence", [], "")])  # a panel that never read the time axis
    p.visual.glow_mode = "none"
    with pytest.raises(video.VideoError, match="no analysed frames to render"):
        video.render_video(empty_dataset(), p, str(tmp_path / "v.mp4"))
    with pytest.raises(video.VideoError, match="no analysed frames to render"):
        video.render_frame(empty_dataset(), p, -1, str(tmp_path / "f.png"))  # the CLI's frame -1 of no frames
    with pytest.raises(ValueError, match="no analysed frames to render"):
        render_charts(empty_dataset(), p, str(tmp_path / "charts"))
    assert not (tmp_path / "v.mp4").exists() and not (tmp_path / "f.png").exists()


# ---------------------------------------------------------------------------
# MDATK-BUG-032: NaN coordinates in the structure view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame, value", [(2, np.nan), (0, np.nan), (2, np.inf)])
def test_032_a_non_finite_coordinate_does_not_break_the_structure_view(frame, value, caplog):
    from mdatk.render import residues

    residues._WARNED.clear()  # warnings are logged once per process
    p = project_with([PanelSpec("structure", [], "")])
    p.visual.label_every = 5
    ds = example_dataset(p, n_frames=6)
    coords = np.array(ds.results["trace"].arrays["coords"], dtype=float)
    clean = build_figure(ds, p, 500, 400, 100)[2][0]
    coords[frame, 5] = value  # one residue of one frame
    ds.results["trace"].arrays["coords"] = coords
    with caplog.at_level(logging.WARNING):
        fig, canvas, panels = build_figure(ds, p, 500, 400, 100)  # was: Axis limits cannot be NaN or Inf
    panel = panels[0]
    assert panel.ok and np.isfinite(panel.ax.get_xlim()).all() and np.isfinite(panel.ax.get_ylim()).all()
    # the view is framed by the finite points: as without the bad residue
    assert panel.ax.get_xlim() == pytest.approx(clean.ax.get_xlim(), rel=0.02)
    assert any(f"frame(s) {frame} " in r.getMessage() and "non-finite" in r.getMessage() for r in caplog.records)
    n_res = coords.shape[1]
    for pos in range(ds.n_frames):
        panel.update(pos)
        canvas.draw()
        drawn = np.asarray(panel.points.get_offsets())
        assert np.isfinite(drawn).all()
        assert len(drawn) == (n_res - 1 if pos == frame else n_res)  # the residue is left out of that frame only
        assert all(np.isfinite(seg).all() for seg in panel.lines.get_segments())


# ---------------------------------------------------------------------------
# MDATK-BUG-035, 036, 076: panel options of a bad value or type
# ---------------------------------------------------------------------------


def test_035_fewer_than_one_tick_counts_as_one():
    from mdatk.render.panels import thin_ticks

    assert thin_ticks(10, 0).tolist() == [0]  # was: ZeroDivisionError
    assert thin_ticks(10, -3).tolist() == [0]  # was: every row labelled
    assert thin_ticks(10, 20).tolist() == list(range(10)) and thin_ticks(10, 5).tolist() == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("ptype, series", [("heatmap", ["contacts/per_residue_a"]), ("contact_map", ["contacts/map"])])
def test_035_max_ticks_below_one_is_drawn_and_warned_about(ptype, series):
    p = project_with([PanelSpec(ptype, series, "", {"max_ticks": 0})])
    panel = build_figure(example_dataset(p, n_frames=6), p, 500, 400, 100)[2][0]  # was: ZeroDivisionError
    assert panel.ok and [t.get_text() for t in panel.ax.get_yticklabels()] == ["1"]
    found = [x for x in p.problems(video=True) if x.field == "layout.panels.0.options"]
    assert len(found) == 1 and found[0].level == "warning" and "'max_ticks' is 0" in found[0].message


BAD_OPTIONS = [
    ("timeseries", ["rmsd/Complex"], {"ylim": 5}, "option 'ylim' must be two numbers [low, high], not 5"),
    ("timeseries", ["rmsd/Complex"], {"ylim": [0.0, float("nan")]}, "option 'ylim' must be two numbers"),
    ("timeseries", ["rmsd/Complex"], {"ylim": "ab"}, "option 'ylim' must be two numbers"),
    ("timeseries", ["rmsd/Complex"], {"smooth": "x"}, "option 'smooth' must be a whole number of frames, not 'x'"),
    ("timeseries", ["rmsd/Complex"], {"target": "high"}, "option 'target' must be a number, not 'high'"),
    ("heatmap", ["contacts/per_residue_a"], {"vmax": "x"}, "option 'vmax' must be a number, not 'x'"),
    ("heatmap", ["contacts/per_residue_a"], {"vmin": [1]}, "option 'vmin' must be a number, not [1]"),
    ("heatmap", ["contacts/per_residue_a"], {"max_ticks": "x"}, "option 'max_ticks' must be a whole number"),
    ("contact_map", ["contacts/map"], {"mode": "frame", "frame": "x"}, "option 'frame' must be a frame number"),
    ("contact_map", ["contacts/map"], {"max_ticks": None}, "option 'max_ticks' must be a whole number, not None"),
    ("sequence", [], {"per_row": "x"}, "option 'per_row' must be a whole number (0: as many as fit), not 'x'"),
    ("sequence", [], {"per_row": -3}, "option 'per_row' must be a whole number (0: as many as fit), not -3"),
]


@pytest.mark.parametrize("ptype, series, options, message", BAD_OPTIONS)
def test_036_an_unusable_panel_option_is_reported_and_shown_not_raised(ptype, series, options, message):
    p = project_with([PanelSpec("timeseries", ["rmsd/Complex"], ""), PanelSpec(ptype, series, "", options)],
                     "equal_2")
    # validation names the panel's options before anything runs: an error when the panels are drawn
    found = [x for x in p.problems(video=True) if x.field == "layout.panels.1.options"]
    assert len(found) == 1 and found[0].level == "error" and message in found[0].message
    assert found[0] in p.validate(video=True) and found[0].location.endswith("Options")
    assert [x.level for x in p.problems(video=False) if x.field == "layout.panels.1.options"] == ["warning"]
    # and the renderer shows the reason in the panel instead of raising (TypeError / ValueError before)
    panels = build_figure(example_dataset(p, n_frames=6), p, 800, 400, 100)[2]
    assert panels[0].ok and not panels[1].ok
    assert f"No data: {message}" in panel_text(panels[1])


@pytest.mark.parametrize("ptype, series, options", [
    ("timeseries", ["rmsd/Complex"], {"ylim": [0, 5], "smooth": 5, "progressive": True, "target": 2.5}),
    ("timeseries", ["rmsd/Complex"], {"ylim": [None, 5.0], "smooth": 0, "target": None}),
    ("heatmap", ["contacts/per_residue_a"], {"vmin": 0, "vmax": 3.5, "max_ticks": 8}),
    ("contact_map", ["contacts/map"], {"mode": "frame", "frame": 3, "transpose": True, "max_ticks": 10}),
    ("contact_map", ["contacts/map"], {"mode": "frequency", "frame": "ignored in this mode"}),
    ("sequence", [], {"per_row": 12, "residues": "A:1-5"}),
    ("sequence", [], {"per_row": 0}),
    ("secondary", [], {"source": "first"}),
    ("secondary", [], {}),
])
def test_036_usable_panel_options_stay_accepted(ptype, series, options):
    from mdatk.config import panel_option_problems

    p = project_with([PanelSpec(ptype, series, "", options)])
    assert panel_option_problems(ptype, options) == []
    assert not [x for x in p.problems(video=True) if x.field == "layout.panels.0.options"]
    assert build_figure(example_dataset(p, n_frames=6), p, 500, 400, 100)[2][0].ok


def test_076_an_unknown_source_of_the_secondary_panel_is_not_taken_as_dynamic():
    message = "unknown source 'dotbraket' (option 'source': dynamic, first or dotbracket)"
    p = project_with([PanelSpec("secondary", [], "", {"source": "dotbraket", "dotbracket": "((..))"})])
    panel = build_figure(example_dataset(p, n_frames=6), p, 500, 400, 100)[2][0]
    assert not panel.ok and f"No data: {message}" in panel_text(panel)  # was: drawn with the dynamic pairs
    found = [x for x in p.problems(video=True) if x.field == "layout.panels.0.options"]
    assert [x.level for x in found] == ["error"] and message in found[0].message
    for source in ("dynamic", "first"):
        p.layout.panels[0].options = {"source": source}
        assert build_figure(example_dataset(p, n_frames=6), p, 500, 400, 100)[2][0].ok
        assert not [x for x in p.problems(video=True) if x.field == "layout.panels.0.options"]
