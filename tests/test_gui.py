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
    # side panel slides out from the menus and keeps its fixed width
    window.open_section("layout", window.slot_rows[1]["frame"])
    assert pump(lambda: window.side_revealer.get_child_revealed(), 5)
    assert window.side_stack.get_visible_child_name() == "layout"
    window.close_side_panel()


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
