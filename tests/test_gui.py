"""GTK smoke tests; run under a display or ``xvfb-run -a pytest``."""

import os
import time

import pytest

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
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
    win.show_all()
    pump(lambda: False, 0.3)
    yield win
    win.destroy()


def test_widgets_roundtrip_project(window):
    assert window.collect(validate=False).to_dict() == window.project.to_dict()


def test_layout_editor_and_rotation(window):
    window.preset_combo.set_active_id("equal_2")
    project = window.collect(validate=False)
    assert project.layout.preset == "equal_2" and len(project.layout.panels) == 2
    window._set_layout(type(project.layout)(preset="single", panels=[PanelSpec("timeseries", ["hbonds/count"])]))
    assert window.collect(validate=False).layout.panels[0].series == ["hbonds/count"]
    window._set_rotation([12.5, -30.0, 45.0])
    assert window.collect(validate=False).visual.rotation == [12.5, -30.0, 45.0]
    assert "+12.5" in window.angle_label.get_text()


def test_preview_and_render(window, tmp_path):
    window.load_preview()
    assert pump(lambda: window.worker is None)
    assert window.preview_data is not None and len(window.preview_data["coords"]) == 24
    window.preview._last = (50, 50)

    class Ev:
        x, y = 90, 50

    window.preview._motion(window.preview, Ev())
    assert window.collect(validate=False).visual.rotation[1] != window.project.visual.rotation[1]
    window.render()
    assert pump(lambda: window.worker is None)
    kinds = [m[0] for m in window.messages]
    assert kinds == ["info"], window.messages
    assert os.path.exists(tmp_path / "gui_out" / "MDATK_video.mp4")
