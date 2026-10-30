"""GTK tests of the WYSIWYG window; run under a display or ``xvfb-run -a pytest``."""

import glob
import os
import time

import pytest

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, Gtk
except (ValueError, ImportError):  # pragma: no cover
    pytest.skip("GTK 3 introspection data not installed", allow_module_level=True)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    pytest.skip("no display (run under xvfb-run)", allow_module_level=True)
if not Gtk.init_check([])[0]:  # pragma: no cover
    pytest.skip("cannot initialise GTK", allow_module_level=True)

from mdatk.config import PanelSpec, Project  # noqa: E402


def pump(cond, timeout=180):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        Gtk.main_iteration_do(False)
        time.sleep(0.005)
    return cond()


class Ev:
    def __init__(self, x, y, state=0, button=1, typ=None):
        self.x, self.y, self.state, self.button = x, y, state, button
        self.type = typ or Gdk.EventType.BUTTON_PRESS


@pytest.fixture
def window(demo_dry, tmp_path):
    from mdatk.gui.app import MainWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "gui_out")
    p.output.video.width, p.output.video.height, p.output.video.dpi = 320, 180, 40
    p.output.charts.dpi = 40
    p.output.charts.series = ["rmsd/Complex"]
    win = MainWindow(app, p, demo_dry["preset"])
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    win.resize(1300, 850)
    win.show_all()
    assert pump(lambda: win.canvas.render_count > 0, 30)
    yield win
    win.destroy()


def test_widgets_roundtrip_project(window):
    assert window.collect(validate=False).to_dict() == window.project.to_dict()


def test_canvas_shows_layout_with_example_data(window):
    canvas = window.canvas
    assert canvas.surface is not None and len(canvas.slots) == len(window.project.layout.panels)
    assert canvas.structure_slot == 0
    assert "Example data" in canvas.watermark
    # the side bar is always shown at its fixed width; the icon bar switches pages
    from mdatk.gui.app import PAGES, SIDE_WIDTH
    assert window.side_stack.get_visible_child_name() == "input"  # default page
    window.open_section("layout", window.slot_rows[1]["frame"])
    assert window.side_stack.get_visible_child_name() == "layout"
    assert window.page_buttons["layout"].get_active()
    for page, _icon, _title in PAGES:
        window.page_buttons[page].clicked()
        assert pump(lambda: window.side_stack.get_visible_child_name() == page, 5)
        pump(lambda: False, 0.2)  # let the page lay out
        assert window.side_panel.get_allocated_width() == SIDE_WIDTH  # never widened by its content
        hadj = window.pages[page].get_hadjustment()
        assert hadj.get_upper() <= hadj.get_page_size() + 1  # no horizontal scrolling


def test_drag_rotates_structure_and_swaps_panels(window):
    canvas = window.canvas
    sx, sy, sw, sh = canvas.slots[canvas.structure_slot]
    cx, cy = sx + sw / 2, sy + sh / 2
    canvas._on_press(None, Ev(cx, cy))
    canvas._on_motion(None, Ev(cx + 40, cy))
    canvas._on_release(None, Ev(cx + 40, cy))
    rotation = window.collect(validate=False).visual.rotation
    assert rotation[1] == pytest.approx(window.project.visual.rotation[1] + 20.0)
    before = [s.series for s in window.collect(validate=False).layout.panels]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = canvas.slots[1], canvas.slots[2]
    canvas._on_press(None, Ev(x1 + w1 / 2, y1 + h1 / 2))
    canvas._on_motion(None, Ev(x2 + w2 / 2, y2 + h2 / 2))
    canvas._on_release(None, Ev(x2 + w2 / 2, y2 + h2 / 2))
    after = [s.series for s in window.collect(validate=False).layout.panels]
    assert after[1] == before[2] and after[2] == before[1]


def test_panel_group_dropdowns_and_auto_title(window):
    row = window.slot_rows[1]
    row["title"].set_text("")
    row["type"].set_active_id("timeseries")
    row["data"].set_active_id("contacts/atom_pairs")
    row["ga"].set_active_id("Target")
    row["gb"].set_active_id("Aptamer")
    spec = window.collect(validate=False).layout.panels[1]
    assert spec.series == ["contacts@Target+Aptamer/atom_pairs"]
    assert row["title"].get_text() == "Target–Aptamer heavy-atom contacts"
    row["title"].set_text("My title")  # typed titles are kept
    row["gb"].set_active_id("")
    assert window.collect(validate=False).layout.panels[1].title == "My title"
    window._set_layout(window.project.layout.__class__(
        preset="single", panels=[PanelSpec("timeseries", ["rmsd@Aptamer/Backbone"])]))
    row = window.slot_rows[0]
    assert row["data"].get_active_id() == "rmsd/Backbone" and row["ga"].get_active_id() == "Aptamer"
    assert not row["gb"].get_sensitive()  # RMSD takes a single group


def test_swatch_survives_selection(window):
    pix = window.swatch_pixbuf("#eb6834")
    assert pix is not None and pix.get_width() == 26
    assert window.swatch_pixbuf("not a colour") is None


def test_real_frame_preview_and_render(window, tmp_path):
    window.load_preview()
    assert pump(lambda: window.worker is None)
    assert window.preview_data is not None and len(window.preview_data["resid"]) == 24
    from mdatk.render.example import example_dataset
    from mdatk.render.residues import chain_names, resnames
    trace = example_dataset(window.project, window.preview_data).results["trace"]
    assert "".join(resnames(trace)) == "GCAUGGCAUCGAUCGAUGCCAUGC" and len(set(chain_names(trace))) == 2
    assert pump(lambda: "reference frame" in window.canvas.watermark, 10)
    window.render()
    assert pump(lambda: window.worker is None)
    assert [m[0] for m in window.messages] == ["info"], window.messages
    videos = glob.glob(str(tmp_path / "gui_out" / "MDATK_video_*.mp4"))
    assert len(videos) == 1 and window.last_video == videos[0]


def test_density_map_fit_shadow_and_drag(window, tmp_path):
    import numpy as np

    from mdatk import density as D
    from mdatk.io.mrc import DensityMap, write_map
    from mdatk.pipeline import Pipeline

    xyz, el, _f, _ref = Pipeline(window.collect(validate=False)).density_frame_atoms("ref")
    true = D.RigidTransform(D.quaternion_matrix([0.7, 0.1, -0.6, 0.3]), np.array([12.0, -6.0, 30.0]))
    sim = D.simulate_map(true.apply(xyz), D.atom_weights(el), 6.0, spacing=1.5)
    path = str(tmp_path / "gui_map.mrc")
    write_map(path, DensityMap(np.pad(sim.data, 10), sim.origin - 15.0, sim.voxel))
    window.density_map_entry.set_text(path)
    window.project.density.resolution = 6.0
    window.binder.load(window.project)
    assert "Not placed" in window.density_status.get_text()
    window.fit_density("global")
    assert pump(lambda: window.worker is None)
    assert not [m for m in window.messages if m[0] == "error"], window.messages
    dp = window.project.density
    assert len(dp.transform) == 12 and dp.fit_cc > 0.9 and "CC" in window.density_status.get_text()
    assert window.preview_data is not None
    assert pump(lambda: window.canvas.structure is not None and window.canvas.structure.has_density(), 30)
    canvas = window.canvas
    # map mode: dragging moves the map, not the view
    window.map_toggle.set_active(True)
    angles = list(canvas.angles)
    sx, sy, sw, sh = canvas.slots[canvas.structure_slot]
    cx, cy = sx + sw / 2, sy + sh / 2
    before = list(dp.transform)
    canvas._on_press(None, Ev(cx, cy))
    canvas._on_motion(None, Ev(cx + 30, cy + 10))
    canvas._on_release(None, Ev(cx + 30, cy + 10))
    assert canvas.angles == angles and dp.transform != before
    assert "by hand" in window.density_status.get_text()
    assert len(dp.anchor) == len(window.preview_data["coords"])  # placement anchored to the shown frame
    rotated = list(dp.transform)
    canvas._on_press(None, Ev(cx, cy, state=Gdk.ModifierType.SHIFT_MASK))
    canvas._on_motion(None, Ev(cx + 20, cy))
    canvas._on_release(None, Ev(cx + 20, cy))
    assert D.rotation_angle(np.array(dp.transform[:9]).reshape(3, 3) @
                            np.array(rotated[:9]).reshape(3, 3).T) == pytest.approx(10.0, abs=0.5)
    window.map_toggle.set_active(False)
    window.fit_density("manual")
    assert pump(lambda: window.worker is None)
    assert dp.fit_cc > 0.9 and "local refinement" in dp.fit_info
    window.clear_density_fit()
    assert dp.transform == [] and "Not placed" in window.density_status.get_text()
    window.centre_density()
    assert len(dp.transform) == 12 and "Centred" in window.density_status.get_text()
    # choosing another map forgets the placement
    window.density_map_entry.set_text(path + ".other")
    assert dp.transform == []


def test_representation_editor_and_3d_preview(window):
    from mdatk.config import RepSpec

    window._add_rep_row(RepSpec("cartoon", ["Aptamer"], sugars=True))
    window._add_rep_row(RepSpec("licorice", ["Target"], color="element", size=0.8))
    reps = window.collect(validate=False).visual.representations
    assert [(r.style, r.groups, r.color) for r in reps] == [("cartoon", ["Aptamer"], "auto"),
                                                           ("licorice", ["Target"], "element")]
    assert reps[0].sugars and reps[1].size == pytest.approx(0.8)
    window.refresh_canvas()
    canvas = window.canvas
    assert pump(lambda: canvas.structure is not None and canvas.structure.stage is not None, 30)
    st = canvas.structure
    assert canvas._moving_artists().count(st.stage.image) == 1  # blitted once, as in the video
    assert st.stage.image.rgba is not None and st.stage.image.rgba[..., 3].any()
    assert not len(st.points.get_offsets())  # every residue is in the 3D scene (others as a trace)
    sx, sy, sw, sh = canvas.slots[canvas.structure_slot]
    cx, cy = sx + sw / 2, sy + sh / 2
    canvas._on_press(None, Ev(cx, cy))
    canvas._on_motion(None, Ev(cx + 30, cy))
    assert st.stage.quality == "fast"  # half resolution while dragging
    canvas._on_release(None, Ev(cx + 30, cy))
    assert pump(lambda: st.stage.quality == "high", 5)
    window._remove_rep_row(window.rep_rows[1])
    p = window.collect(validate=False)
    assert [r.style for r in p.visual.representations] == ["cartoon"]
    window.binder.load(p)
    assert window.collect(validate=False).to_dict() == p.to_dict()


def test_pdb_tools_window_checks_and_cleans(window, tmp_path):
    from test_pdbtools import make_pdb

    from mdatk import pdbtools
    src = tmp_path / "in.pdb"
    src.write_text(make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)]),
                             ("B", [("0G", 3, "G", True), ("0U", 4, "U", True)])]))
    window.open_pdb_tools()
    dlg = window.pdb_tools
    dlg.input.entry.set_text(str(src))
    assert dlg.output.entry.get_text().endswith("in_clean.pdb")  # suggested from the input
    dlg.run_check()
    buf = dlg.report.get_buffer()
    report = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    assert "Chain B" in report and "chirality 2 L" in report
    dlg.scheme.set_active_id("amber")
    dlg.renumber.set_active(True)
    dlg.run_clean()
    names = [r.name for r in pdbtools.read_pdb(dlg.output.entry.get_text()).residues()]
    assert names == ["G5", "C3", "LG5", "LU3"]
    assert "REMARK 950" not in open(dlg.output.entry.get_text()).read()
    assert not dlg.ann_engine.get_sensitive()
    dlg.annotate.set_active(True)
    dlg.ann_engine.set_active_id("builtin")
    dlg.ann_stacking.set_active(False)
    assert dlg.ann_engine.get_sensitive()
    dlg.run_clean()
    text = open(dlg.output.entry.get_text()).read()
    assert "REMARK 950 ENGINE builtin" in text and "REMARK 950 DBN" in text
    assert [r.name for r in pdbtools.read_pdb(text).residues()] == names
    dlg.destroy()
    assert window.pdb_tools is None


def test_citation_text_lists_project_resources(window):
    text = window.citation_text()
    assert isinstance(text, str) and text


def test_slot_defaults_for_residue_panels_and_wheel_zoom(window):
    row = window.slot_rows[1]
    row["opts"].set_text("")
    row["type"].set_active_id("secondary")
    pump(lambda: False, 0.2)
    assert '"source": "dynamic"' in row["opts"].get_text()
    assert window._read_slots()[1].series == []  # residue panels take no data keys
    z0 = window.zoom_spin.get_value()
    window._on_canvas_zoom(1.15)
    assert window.zoom_spin.get_value() == pytest.approx(z0 * 1.15)
    assert window.collect(validate=False).visual.zoom == pytest.approx(z0 * 1.15)


def test_value_widgets_ignore_the_mouse_wheel(window):
    from mdatk.gui import widgets as W
    window.open_wizard("amber-build")
    window.open_planner()
    window.open_pdb_tools()
    window.open_builder()
    pump(lambda: False, 0.5)
    missing = []

    def walk(w, where):
        if isinstance(w, W.WHEEL_TYPES) and not isinstance(w, Gtk.Scrollbar) \
                and not getattr(w, "_mdatk_no_wheel", False):
            missing.append(f"{where}: {type(w).__name__} {w.get_tooltip_text() or ''}"[:120])
        if isinstance(w, Gtk.Container):
            for c in w.get_children():
                walk(c, where)
    for top in Gtk.Window.list_toplevels():
        if top.get_visible():
            walk(top, top.get_title() or type(top).__name__)
    assert not missing, missing
    # a wheel event over a spin button leaves its value alone
    spin = W.spin(0, 100, 1)
    spin.set_value(5)
    ev = Gdk.Event.new(Gdk.EventType.SCROLL)
    ev.scroll.direction = Gdk.ScrollDirection.UP
    spin.emit("scroll-event", ev)
    assert spin.get_value() == 5


def test_builder_non_wc_helix_builds_in_the_background(window, tmp_path):
    window.open_builder()
    dlg = window.builder
    dlg.seq.set_text("AAAA")
    dlg.pairing.set_active_id("hh")
    out = tmp_path / "hh.pdb"
    dlg.helix_out.entry.set_text(str(out))
    dlg.notebook.set_current_page(0)
    dlg.run_current()
    assert dlg.worker is not None  # the window stays responsive while it builds
    assert pump(lambda: dlg.worker is None, 120) and out.exists()
    buf = dlg.report.get_buffer()
    assert "Wrote" in buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    dlg.destroy()


def _bound(window, path):
    return next(b.widget for b in window.binder.bindings if b.path == path)


def test_bad_selection_flags_its_field_and_badges_its_page(window):
    from mdatk.gui import widgets as W
    from mdatk.gui.app import PAGES
    from mdatk.problems import TABS

    assert {page: title for page, _icon, title in PAGES} == TABS  # problem locations use the page titles
    pv = window.problem_view
    badge = pv.badges["analyses"]
    assert not pv.on_page("analyses") and not badge.get_visible()
    entry = _bound(window, "analyses.mg.ligand_selection")
    hint = entry.get_tooltip_text()
    entry.set_text("group:Aptamr or group:Target")
    assert pump(lambda: badge.get_visible(), 5)  # revalidated shortly after the edit
    assert badge.get_text() == "1" and "mdatk-badge" in badge.get_style_context().list_classes()
    assert "Ligand selection" in window.page_buttons["analyses"].get_tooltip_text()
    message, level = W.flag_of(entry)
    assert level == "error" and "unknown residue group 'Aptamr'" in message
    assert "did you mean 'Aptamer'?" in message
    assert entry.get_style_context().has_class("error")
    assert entry.get_icon_name(Gtk.EntryIconPosition.SECONDARY) == "dialog-error-symbolic"
    assert entry.get_tooltip_text().startswith(message)
    (prob,) = pv.problems
    assert prob.key == "analyses:analyses.mg.ligand_selection"
    assert prob.location == "Analyses › Mg²⁺ coordination and occupancy › Ligand selection"
    assert pv.button.get_visible() and "1 error" in pv._count.get_text()
    window.page_buttons["input"].clicked()
    window.jump_to_problem(prob)  # what a click in the problems list does
    assert pump(lambda: window.side_stack.get_visible_child_name() == "analyses" and entry.has_focus(), 5)
    # the fix clears the flag, the badge and the list
    entry.set_text("group:Aptamer or group:Target")
    assert pump(lambda: not badge.get_visible(), 5)
    assert W.flag_of(entry) == ("", "") and not entry.get_style_context().has_class("error")
    assert entry.get_icon_name(Gtk.EntryIconPosition.SECONDARY) is None and entry.get_tooltip_text() == hint
    assert not pv.button.get_visible()
    # a group combo and a list editor cell; a broken group is reported at its definition only
    combo = _bound(window, "analyses.hbonds.group_b")
    combo.get_child().set_text("Tagret")
    window.group_editor.store[0][1] = "1-x"
    assert pump(lambda: pv.badges["groups"].get_visible() and badge.get_visible(), 5)
    assert W.flag_of(combo)[0].startswith("unknown residue group 'Tagret'")
    assert window.group_editor.cell_problems[(0, 1)][1] == "error"
    assert "invalid residue range item '1-x'" in W.flag_of(window.group_editor)[0]
    assert [p.field for p in pv.on_page("analyses")] == ["analyses.hbonds.group_b"]
    combo.get_child().set_text("Target")
    window.group_editor.store[0][1] = "1-12"
    assert pump(lambda: not pv.problems, 5)
    assert not pv.badges["groups"].get_visible() and window.group_editor.cell_problems == {}
    assert W.flag_of(combo) == ("", "")


def test_run_time_selection_error_flags_the_setting_it_came_from(window):
    from mdatk.gui import widgets as W

    window.group_editor.store[1][1] = "100-112"  # valid, but nothing in the structure
    window.render()
    assert pump(lambda: window.worker is None)
    kind, title, detail = window.messages[-1]
    assert kind == "error" and detail.startswith("Analyses › Pairwise contacts › Group B: selection "
                                                 "'group:Target' matched no atoms: group 'Target'")
    combo = _bound(window, "analyses.contacts.group_b")
    assert pump(lambda: W.flag_of(combo)[1] == "error", 5)
    assert window.problem_view.badges["analyses"].get_visible()
    window.group_editor.store[1][1] = "13-24"  # the group is fixed: the run-time problem goes
    assert pump(lambda: not window.problem_view.badges["analyses"].get_visible(), 5)
    assert W.flag_of(combo) == ("", "")
