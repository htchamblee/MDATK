"""GTK test of the planner's collection ports (run under xvfb-run -a): the full set of decoys of a FARFAR2
step is a large dot, hit and drawn larger than an ordinary port, and a drag from it connects a model check."""

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
cairo = pytest.importorskip("cairo")

from mdatk.config import Project  # noqa: E402
from mdatk.gui.planner import KIND_COLORS, PORT_R, SET_R  # noqa: E402


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.005)


class Ev:
    def __init__(self, x, y, button=1):
        self.x, self.y, self.button, self.state = x, y, button, 0
        self.x_root, self.y_root = x, y
        self.type = Gdk.EventType.BUTTON_PRESS


@pytest.fixture
def main(tmp_path):
    from mdatk.gui.rnado import RNAdoWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    p.workflow.project_name = "apt1"
    win = RNAdoWindow(app, p, None)
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    win.show_all()
    pump(0.5)
    yield win
    win.destroy()


def _pixel(surface, x, y):
    """(r, g, b) in 0..1 of an ARGB32 image surface (opaque)."""
    data, stride = surface.get_data(), surface.get_stride()
    b, g, r = data[int(y) * stride + int(x) * 4: int(y) * stride + int(x) * 4 + 3]
    return r / 255, g / 255, b / 255


def test_collection_port_is_larger_and_connects_a_model_check(main):
    pl = main.planner
    fold = pl.add_step("farfar2", 40, 40)
    check = pl.add_step("model-check", 520, 60)
    pump(0.2)
    canvas = pl.canvas
    canvas.set_zoom(1.0)
    ins, outs = canvas.ports(fold)
    ports = {p.name: p for p in outs}
    assert ports["decoys"].collection and not ports["silent"].collection
    assert canvas.port_radius(ports["decoys"]) == SET_R >= 1.8 * PORT_R
    assert canvas.port_radius(ports["silent"]) == PORT_R
    model_in = {p.name: p for p in canvas.ports(check)[0]}["models"]
    assert model_in.collection and canvas.port_radius(model_in) == SET_R

    # hit-testing: 12 px outside the centre still hits the large dot, not an ordinary one
    dx, dy = canvas.port_pos(fold, "decoys", True)
    sx, sy = canvas.port_pos(fold, "silent", True)
    assert PORT_R + 5 < 12 < SET_R + 5
    hit = canvas.port_at(dx + 12, dy)
    assert hit is not None and hit[1].name == "decoys" and hit[2]
    assert canvas.port_at(sx + 12, sy) is None
    assert canvas.port_at(sx + 6, sy)[1].name == "silent"
    # between the large dot and its neighbour above, the large dot wins up to its own edge
    assert canvas.port_at(dx, dy - SET_R + 1)[1].name == "decoys"
    mx, my = canvas.port_pos(check, "models", False)
    assert canvas.port_at(mx - 12, my)[1].name == "models"

    # drawn larger: 7 px from the centre the large dot is the PDB colour; an ordinary dot ends at 5 px
    alloc = canvas.get_allocation()
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, max(alloc.width, 900), max(alloc.height, 400))
    canvas._on_draw(canvas, cairo.Context(surface))
    pdb = KIND_COLORS["pdb"]
    assert _pixel(surface, dx, dy + 7) == pytest.approx(pdb, abs=0.02)
    assert _pixel(surface, sx, sy + 7) != pytest.approx(KIND_COLORS["silent"], abs=0.02)
    assert _pixel(surface, sx, sy + 3) == pytest.approx(KIND_COLORS["silent"], abs=0.02)

    # the tooltip says what the large dot is
    canvas._on_motion(canvas, Ev(dx + 10, dy))
    assert "full set" in (canvas.get_tooltip_text() or "")

    # dragging from the large dot to the model check's models connects them and switches the extraction on
    canvas._on_press(canvas, Ev(dx + 10, dy))
    canvas._on_motion(canvas, Ev(mx - 10, my))
    canvas._on_release(canvas, Ev(mx - 10, my))
    pump(0.2)
    edges = main.project.workflow.edges
    assert [(e.source, e.output, e.target, e.input) for e in edges] == [(fold.id, "decoys", check.id,
                                                                         "models")]
    assert fold.settings["extract_decoys"] is True
    step = pl.plan.steps[check.id]
    assert step.ctx.lists("models") and not [i for i in step.issues if i.level == "error"]
    # ... and onto its decoy-table input it connects the step's decoy table
    sx2, sy2 = canvas.port_pos(check, "scores", False)
    canvas._on_press(canvas, Ev(dx, dy))
    canvas._on_release(canvas, Ev(sx2, sy2))
    pump(0.2)
    assert (fold.id, "decoy_table", check.id, "scores") in [(e.source, e.output, e.target, e.input)
                                                            for e in main.project.workflow.edges]
    # an ordinary single-file input refuses the set
    build = pl.add_step("amber-build", 520, 400)
    pump(0.2)
    bx, by = canvas.port_pos(build, "structure", False)
    n = len(main.project.workflow.edges)
    canvas._on_press(canvas, Ev(dx, dy))
    canvas._on_release(canvas, Ev(bx, by))
    assert len(main.project.workflow.edges) == n and "takes one file" in pl.status.get_text()
