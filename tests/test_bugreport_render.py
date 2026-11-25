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


# ---------------------------------------------------------------------------
# Charts and heatmaps: MDATK-BUG-034, 057, 058, 059, 063, 064, 082
# ---------------------------------------------------------------------------


@pytest.fixture
def chart_figures(monkeypatch):
    """The figures the chart functions create (they only return file names)."""
    from mdatk.render import charts

    made = []

    class Capture(charts.Figure):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(charts, "Figure", Capture)
    return made


def chart_project(theme="light", dpi=60):
    p = Project()
    p.inputs.time_unit = "ps"
    p.visual.theme = theme
    c = p.output.charts
    c.dpi, c.width, c.height = dpi, 5.0, 3.0
    return p


def heat_column(ax, x: float) -> int:
    """The column of the heatmap drawn in ``ax`` that covers the time ``x``."""
    if ax.images:  # one image over the whole time range: equal columns
        x0, x1 = ax.images[0].get_extent()[:2]
        return int((x - x0) / (x1 - x0) * ax.images[0].get_array().shape[1])
    edges = np.asarray(ax.collections[0].get_coordinates())[0, :, 0]
    return int(np.searchsorted(edges, x)) - 1


UNEVEN_TIMES = np.concatenate([np.arange(12) * 10.0, 110.0 + (np.arange(12) + 1) * 2.0])  # 10 ps, then 2 ps apart


def test_057_heatmap_columns_sit_at_their_frame_times(chart_figures, tmp_path):
    from mdatk.render.charts import render_matrix_chart
    from mdatk.render.panels import time_edges

    edges = time_edges(UNEVEN_TIMES)
    assert edges[:3].tolist() == [-5.0, 5.0, 15.0] and edges[11:14].tolist() == [105.0, 111.0, 113.0]
    assert edges[-1] == 135.0 and len(edges) == 25
    assert time_edges(np.arange(4) * 2.0).tolist() == [-1.0, 1.0, 3.0, 5.0, 7.0]  # evenly spaced: as before
    assert time_edges(np.array([7.0])).tolist() == [6.5, 7.5]
    ds = matrix_dataset(np.arange(6 * 24).reshape(6, 24) % 5, UNEVEN_TIMES)
    p = chart_project()
    p.layout.preset, p.layout.panels = "single", [PanelSpec("heatmap", ["contacts/per"], "")]
    panel = build_figure(ds, p, 600, 300, 100)[2][0]
    render_matrix_chart(ds, p, "contacts/per", str(tmp_path))
    for ax in (panel.ax, chart_figures[0].axes[0]):
        # every frame's time (where the moving cursor stands) lies in that frame's column; was: column 18 at frame 12
        assert [heat_column(ax, t) for t in UNEVEN_TIMES] == list(range(24))
        assert ax.get_xlim() == (-5.0, 135.0) and ax.get_ylim() == (5.5, -0.5)


def test_057_evenly_spaced_frames_keep_the_image(chart_figures, tmp_path):
    from mdatk.render.charts import render_matrix_chart

    ds = matrix_dataset(np.arange(6 * 8).reshape(6, 8) % 5, np.arange(8) * 10.0)
    p = chart_project()
    p.layout.preset, p.layout.panels = "single", [PanelSpec("heatmap", ["contacts/per"], "")]
    panel = build_figure(ds, p, 600, 300, 100)[2][0]
    render_matrix_chart(ds, p, "contacts/per", str(tmp_path))
    for ax in (panel.ax, chart_figures[0].axes[0]):
        assert len(ax.images) == 1 and list(ax.images[0].get_extent()) == [-5.0, 75.0, 5.5, -0.5]


def test_058_running_mean_skips_frames_without_a_value():
    from mdatk.render.panels import _running_mean

    y = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10.0])
    out = _running_mean(y, 5)  # was: [1.6 nan nan nan nan nan 7. 8. 8.8 9.4]
    assert out.tolist() == pytest.approx([1.6, 1.75, 2.75, 4.0, 5.25, 6.5, 7.0, 8.0, 8.8, 9.4])
    gap = _running_mean(np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0]), 3)
    assert np.isnan(gap[2:5]).all() and np.isfinite(gap[[0, 1, 5, 6]]).all()  # NaN only where a whole window is
    full = np.arange(10.0) ** 2
    padded = np.pad(full, (2, 2), mode="edge")
    assert np.array_equal(_running_mean(full, 5), np.convolve(padded, np.ones(5) / 5, mode="valid"))  # unchanged


def test_059_every_class_of_an_ordinal_scale_has_its_own_colour():
    from mdatk.render.theme import DARK, LIGHT

    for theme in (LIGHT, DARK):
        for n in range(2, 13):
            colours = theme.ordinal(n).colors
            assert len(colours) == n and len(set(colours)) == n, (theme.name, n)  # was: 5 colours for n > 5
    # up to five classes keep their colours
    assert LIGHT.ordinal(3).colors == ["#e1e0d9", "#86b6ef", "#0d366b"]
    assert LIGHT.ordinal(5).colors == ["#e1e0d9", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    assert DARK.ordinal(5).colors == ["#2c2c2a", "#184f95", "#3987e5", "#86b6ef", "#cde2fb"]
    eight = LIGHT.ordinal(8).colors  # the same ramp, in seven even steps
    assert eight[0] == "#e1e0d9" and eight[1] == "#86b6ef" and eight[-1] == "#0d366b"
    # a class matrix of eight classes: the colour bar's eight labels show eight colours
    classes = [f"class {k}" for k in range(8)]
    ds = matrix_dataset(np.repeat(np.arange(8)[None, :], 3, axis=0), np.arange(8) * 10.0, classes)
    p = chart_project()
    p.layout.preset, p.layout.panels = "single", [PanelSpec("heatmap", ["contacts/per"], "")]
    im = build_figure(ds, p, 600, 300, 100)[2][0].ax.images[0]
    assert len({im.cmap(im.norm(k)) for k in range(8)}) == 8  # was 5: e.g. classes 0 and 1 looked alike


def test_034_a_matrix_chart_of_thousands_of_rows_is_limited_in_height(chart_figures, tmp_path, caplog):
    from mdatk.render import charts

    rows = 8000  # 0.18 in per row: 72 075 px at 50 dpi, more than Agg draws before Matplotlib 3.10 (2^16)
    ds = matrix_dataset(np.arange(rows * 4).reshape(rows, 4) % 5, np.arange(4) * 10.0)
    p = chart_project(dpi=50)
    p.output.charts.width = 3.0
    with caplog.at_level(logging.WARNING):
        (path,) = charts.render_matrix_chart(ds, p, "contacts/per", str(tmp_path))
    fig = chart_figures[0]
    assert fig.get_figheight() * fig.dpi <= 65000 < 2 ** 16  # was 72 075 px
    assert charts.MAX_CHART_PIXELS == 65000 and os.path.getsize(path) > 0
    assert any("8000 rows" in r.getMessage() and "limited to 65000 px" in r.getMessage() for r in caplog.records)
    # a chart below the limit keeps its height of 0.18 in per row
    caplog.clear()
    small = matrix_dataset(np.arange(40 * 4).reshape(40, 4) % 5, np.arange(4) * 10.0)
    charts.render_matrix_chart(small, p, "contacts/per", str(tmp_path))
    assert chart_figures[1].get_figheight() == pytest.approx(0.18 * 40 + 1.5) and not caplog.records


def chart_dataset():
    """A series, a per-residue matrix and a residue map."""
    ds = matrix_dataset(np.arange(6 * 8).reshape(6, 8) % 5, np.arange(8) * 10.0)
    res = ds.results["contacts"]
    res.arrays["s"] = np.linspace(1.0, 3.0, 8)
    res.add_series("total", "s", "Total contacts", "count")
    b = CSRBuilder()
    for _ in range(8):
        b.add(np.array([0, 1]), np.array([1, 0]), np.ones(2))
    res.arrays.update(b.arrays("map"))
    res.meta.update(r=["1", "2"], c=["1", "2"])
    res.add_pairmap("map", "map", "r", "c", "Map")
    return ds


def test_063_a_matrix_asked_for_by_name_is_charted_and_a_residue_map_is_named(tmp_path, caplog):
    from mdatk.render.charts import render_charts

    p = chart_project()
    p.output.charts.series = ["contacts/total", "contacts/per", "contacts/map", "contacts/nope"]
    with caplog.at_level(logging.WARNING):
        paths = render_charts(chart_dataset(), p, str(tmp_path))
    names = sorted(os.path.basename(f) for f in paths)
    assert names == ["MDATK_chart_contacts_per.png", "MDATK_chart_contacts_total.png"]  # was: the series only
    messages = [r.getMessage() for r in caplog.records]
    assert any("contacts/map is a residue map" in m for m in messages)  # was: dropped without a word
    assert any("contacts/nope is not available" in m for m in messages)
    assert not any("contacts/per" in m for m in messages)
    p.output.charts.include_heatmaps = True  # every matrix: the one asked for is not drawn twice
    assert len(render_charts(chart_dataset(), p, str(tmp_path / "all"))) == 2


def test_064_chart_formats_are_checked_before_the_run():
    p = Project()
    assert not [x for x in p.problems(charts=True) if x.field == "output.charts.formats"]
    p.output.charts.formats = ["png", ".PDF", "svg", "pnng"]
    found = [x for x in p.problems(charts=True) if x.field == "output.charts.formats"]
    assert len(found) == 1 and found[0].level == "error" and found[0] in p.validate(charts=True)
    assert "Unknown chart file format 'pnng'" in found[0].message and "png" in found[0].message
    assert [x.level for x in p.problems(charts=False) if x.field == "output.charts.formats"] == ["warning"]


@pytest.mark.parametrize("field, uses", [("visual.glow_color", "video"), ("visual.base_color", "video"),
                                         ("visual.ion_color", "video"), ("groups.0.color", "video"),
                                         ("output.charts.color", "charts")])
def test_064_colours_are_checked_before_the_run(field, uses):
    p = Project()
    assert not [x for x in p.problems(video=True, charts=True) if "not a colour" in x]  # the defaults are colours
    obj = p
    *path, leaf = field.split(".")
    for part in path:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    setattr(obj, leaf, "notacolour")
    found = [x for x in p.problems(video=True, charts=True) if x.field == field]
    assert len(found) == 1 and found[0].level == "error" and "'notacolour' is not a colour" in found[0].message
    off = {"video": uses != "video", "charts": uses != "charts"}  # the output that reads the colour is not made
    assert [x.level for x in p.problems(**off) if x.field == field] == ["warning"]
    setattr(obj, leaf, "tab:orange")
    assert not [x for x in p.problems(video=True, charts=True) if x.field == field]


def test_064_a_bad_colour_or_format_stops_the_run_before_the_analyses(demo_dry, tmp_path):
    from mdatk.pipeline import Pipeline, PipelineError

    for field, value, text in (("visual.glow_color", "notacolour", "'notacolour' is not a colour"),
                               ("output.charts.formats", ["png", "pnng"], "Unknown chart file format 'pnng'")):
        p = Project.load(demo_dry["preset"])
        p.output.out_dir = str(tmp_path / field)
        section, leaf = field.rsplit(".", 1)
        obj = p
        for part in section.split("."):
            obj = getattr(obj, part)
        setattr(obj, leaf, value)
        pipe = Pipeline(p)
        with pytest.raises(PipelineError) as err:  # was: the analyses ran, then Matplotlib's own error
            pipe.run(video=True, charts=True, pdb=False)
        assert text in str(err.value) and [x.field for x in err.value.problems] == [field]
        assert not os.path.exists(p.cache_dir)  # nothing was analysed


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_082_chart_title_and_colour_bar_follow_the_theme(theme, chart_figures, tmp_path):
    from matplotlib.colors import to_hex

    from mdatk.render.charts import render_matrix_chart, render_series_chart
    from mdatk.render.theme import get_theme

    t = get_theme(theme)
    p = chart_project(theme)
    render_matrix_chart(chart_dataset(), p, "contacts/per", str(tmp_path))
    render_series_chart(chart_dataset(), p, "contacts/total", str(tmp_path))
    for fig, label in zip(chart_figures, ("Per residue", "Total contacts")):
        (title,) = [x for x in fig.axes[0].get_children() if getattr(x, "get_text", lambda: "")() == label]
        assert to_hex(title.get_color()) == t.ink  # was black: unreadable on the dark surface
    bar = chart_figures[0].axes[1]
    assert to_hex(bar.yaxis.label.get_color()) == t.ink2
    assert {to_hex(x.get_color()) for x in bar.yaxis.get_ticklabels()} == {t.ink2}
