"""Steps on the planner canvas keep apart: a dragged step stops at another and slides along it (Shift: put it
anywhere), Duplicate and Arrange place steps clear of the others; connections have a contrasting
outline (run under xvfb-run -a)."""

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

from mdatk.config import Project  # noqa: E402
from mdatk.gui.planner import GRID, TILE_GAP, edge_casing, free_position  # noqa: E402


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
    win.resize(900, 650)
    win.show_all()
    pump(0.5)
    yield win
    win.destroy()



def _plan(main, tmp_path):
    """input-file (PDB) -> amber-build B, and a second amber-build C."""
    pdb = tmp_path / "m.pdb"
    pdb.write_text("ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P\nEND\n")
    pl = main.planner
    src = pl.add_step("input-file", 40, 40)
    src.settings.update(kind="pdb", path=str(pdb))
    b = pl.add_step("amber-build", 400, 40)
    c = pl.add_step("amber-build", 400, 380)
    pl.changed()
    pump(0.2)
    pl.canvas.set_zoom(1.0)
    canvas = pl.canvas
    ox, oy = canvas.port_pos(src, "file", True)
    bx, by = canvas.port_pos(b, "structure", False)
    canvas._on_press(canvas, Ev(ox, oy))
    canvas._on_release(canvas, Ev(bx, by))
    pump(0.1)
    return pl, src, b, c



def overlaps(a, b, gap=TILE_GAP):
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    return ax < bx + bw + gap and bx < ax + aw + gap and ay < by + bh + gap and by < ay + ah + gap


def test_free_position_stops_at_a_tile_and_slides_along_it():
    box = (300.0, 100.0, 236.0, 120.0)
    assert free_position(40, 40, 236, 120, [box]) == (40, 40)  # free: where it was asked for
    # dragged into the tile from the left: it stops TILE_GAP short of it, at the height asked for
    x, y = free_position(200, 120, 236, 120, [box], start=(0.0, 120.0))
    assert y == 120 and x + 236 + TILE_GAP <= 300 and x % GRID == 0 and x > 0
    # and slides down along its edge, then below it once it is clear
    assert free_position(200, 150, 236, 120, [box], start=(x, y)) == (x, 150)
    # dragged in from above: it stops above it, at the x asked for
    low = (300.0, 300.0, 236.0, 120.0)
    assert free_position(350, 200, 236, 120, [low], start=(350.0, 0.0)) == (350, 300 - TILE_GAP - 120)
    # the pointer's place free beyond the tile: it jumps there
    assert free_position(600, 120, 236, 120, [box], start=(0.0, 120.0)) == (600, 120)
    # no start (Duplicate): the nearest free place
    x, y = free_position(310, 110, 236, 120, [box])
    assert not overlaps((x, y, 236, 120), box)
    # never above or left of the canvas
    x, y = free_position(0, 0, 236, 120, [(0.0, 0.0, 236.0, 120.0)])
    assert x >= 0 and y >= 0 and not overlaps((x, y, 236, 120), (0, 0, 236, 120))


def test_free_position_between_two_tiles():
    boxes = [(0.0, 0.0, 236.0, 100.0), (0.0, 250.0, 236.0, 100.0)]
    x, y = free_position(0, 120, 236, 100, boxes)
    assert all(not overlaps((x, y, 236, 100), b) for b in boxes)


def test_edge_casing_contrasts_with_the_wire():
    from mdatk.gui.planner import CATEGORY_COLORS, KIND_COLORS

    def lum(c):
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    for rgb in list(KIND_COLORS.values()) + [(0.10, 0.37, 0.71)]:  # every wire (and the selected one)
        assert lum(rgb) - lum(edge_casing(rgb)) > 0.15
    assert CATEGORY_COLORS  # the canvas and the step bodies are light: a dark outline shows on both


def _box(canvas, n):
    return (n.x, n.y) + canvas.node_size(n)


def test_a_dragged_step_does_not_land_on_another(main):
    pl = main.planner
    pl.canvas.set_zoom(1.0)
    a = pl.add_step("amber-build", 40, 40)
    b = pl.add_step("amber-build", 400, 40)
    pl.changed()
    pump(0.2)
    canvas = pl.canvas
    canvas._on_press(canvas, Ev(b.x + 100, b.y + 10))
    assert canvas.drag["mode"] == "move"
    canvas._on_motion(canvas, Ev(a.x + 150, a.y + 30))  # onto a's right half
    assert not overlaps(_box(canvas, b), _box(canvas, a))
    assert a.x + 236 + TILE_GAP <= b.x <= a.x + 236 + TILE_GAP + GRID
    assert b.y == 60  # stopped beside it, at the height it was asked for
    canvas._on_release(canvas, Ev(a.x + 150, a.y + 30))
    # with Shift held it goes where it is put
    canvas._on_press(canvas, Ev(b.x + 100, b.y + 10))
    ev = Ev(a.x + 110, a.y + 20)
    ev.state = Gdk.ModifierType.SHIFT_MASK
    canvas._on_motion(canvas, ev)
    canvas._on_release(canvas, ev)
    assert (b.x, b.y) == (a.x + 10, a.y + 10)


def test_duplicate_and_arrange_keep_steps_apart(main):
    pl = main.planner
    a = pl.add_step("amber-md", 40, 40)
    pl.changed()
    pump(0.1)
    pl.duplicate(a)
    pump(0.1)
    nodes = main.project.workflow.nodes
    assert len(nodes) == 2 and not overlaps(_box(pl.canvas, nodes[0]), _box(pl.canvas, nodes[1]))
    for _ in range(3):
        pl.add_step("amber-md", 40, 40)  # all on one spot
    pl.changed()
    pump(0.1)
    pl.arrange()
    boxes = [_box(pl.canvas, n) for n in main.project.workflow.nodes]
    assert all(not overlaps(p, q, gap=0) for i, p in enumerate(boxes) for q in boxes[i + 1:])


def test_connections_are_drawn_with_an_outline(main, tmp_path):
    import cairo
    pl, src, b, _c = _plan(main, tmp_path)
    canvas = pl.canvas
    (x0, y0), _p1, _p2, (x3, y3) = canvas.edge_points(main.project.workflow.edges[0])
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, int(x3) + 40, int(max(y0, y3)) + 40)
    cr = cairo.Context(surface)
    canvas._draw_edge(cr, main.project.workflow.edges[0], False)
    surface.flush()
    data, stride = surface.get_data(), surface.get_stride()
    # a column across the wire, half way: wire colour in the middle, the outline's colour on both sides
    mx = int((x0 + x3) / 2)
    col = [tuple(data[y * stride + mx * 4: y * stride + mx * 4 + 3]) for y in range(surface.get_height())]
    lit = [y for y, px in enumerate(col) if px != (0, 0, 0)]
    assert len(lit) >= 4  # wider than the 2.2 px wire alone: the outline is there
