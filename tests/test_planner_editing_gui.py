"""GTK tests of editing a plan in RNAdo (run under xvfb-run -a): moving the end of a connection, dropping
connections on a step, deleting a connection alone, scrolling at the edge while dragging, new steps in
view, closing the project, and dialog text that can be selected."""

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
import ast  # noqa: E402
import glob  # noqa: E402

from mdatk.gui.dialogtext import selectable_text  # noqa: E402
from mdatk.gui.planner import NODE_W  # noqa: E402


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


def edges(main):
    return [(e.source, e.output, e.target, e.input) for e in main.project.workflow.edges]


def test_the_end_of_a_connection_moves_to_another_input_or_step(main, tmp_path):
    pl, src, b, c = _plan(main, tmp_path)
    canvas = pl.canvas
    assert edges(main) == [(src.id, "file", b.id, "structure")]
    bx, by = canvas.port_pos(b, "structure", False)
    cx, cy = canvas.port_pos(c, "structure", False)
    # picked up at its input and dropped on another step's input: the connection moves there
    canvas._on_press(canvas, Ev(bx, by))
    assert canvas.drag["mode"] == "rewire"
    canvas._on_motion(canvas, Ev(cx, cy))
    canvas._on_release(canvas, Ev(cx, cy))
    assert edges(main) == [(src.id, "file", c.id, "structure")]
    assert "Moved the connection" in pl.status.get_text()
    # dropped on the body of a step: its first free input that takes it
    canvas._on_press(canvas, Ev(cx, cy))
    canvas._on_release(canvas, Ev(b.x + NODE_W / 2, b.y + 40))
    assert edges(main) == [(src.id, "file", b.id, "structure")]
    # dropped on empty canvas or back where it was: nothing changes
    bx, by = canvas.port_pos(b, "structure", False)
    for x, y in ((bx, by), (900, 900)):
        canvas._on_press(canvas, Ev(bx, by))
        canvas._on_release(canvas, Ev(x, y))
        assert edges(main) == [(src.id, "file", b.id, "structure")]
    assert {n.id for n in main.project.workflow.nodes} == {src.id, b.id, c.id}


def test_a_new_connection_dropped_on_a_step_takes_its_free_input(main, tmp_path):
    pl, src, b, c = _plan(main, tmp_path)
    canvas = pl.canvas
    ox, oy = canvas.port_pos(src, "file", True)
    canvas._on_press(canvas, Ev(ox, oy))
    canvas._on_release(canvas, Ev(c.x + NODE_W / 2, c.y + 40))
    assert (src.id, "file", c.id, "structure") in edges(main)
    assert (src.id, "file", b.id, "structure") in edges(main)  # the other connection stays


def test_deleting_a_connection_from_its_menu_keeps_the_steps(main, tmp_path, monkeypatch):
    # the step selected before (here the connection's target, as after a failed attempt to move the
    # connection's end) was deleted with the connection
    pl, src, b, c = _plan(main, tmp_path)
    canvas = pl.canvas
    canvas.select(b.id)
    shown = []
    monkeypatch.setattr(Gtk.Menu, "popup_at_pointer", lambda menu, _ev: shown.append(menu))
    (a, _p1, _p2, d) = canvas.edge_points(main.project.workflow.edges[0])
    mx, my = (a[0] + d[0]) / 2, (a[1] + d[1]) / 2
    assert canvas.edge_at(mx, my) == 0
    canvas._context_menu(Ev(mx, my, button=3), mx, my)
    item = [i for i in shown[0].get_children() if isinstance(i, Gtk.MenuItem)
            and i.get_label() == "Delete connection"][0]
    assert canvas.selected is None
    item.activate()
    pump(0.1)
    assert edges(main) == []
    assert {n.id for n in main.project.workflow.nodes} == {src.id, b.id, c.id}


def test_dragging_at_the_edge_scrolls_the_canvas(main, tmp_path):
    pl, src, b, c = _plan(main, tmp_path)
    canvas = pl.canvas
    h = pl.scroller.get_hadjustment()
    pump(0.3)
    page = h.get_page_size()
    assert page > 200 and h.get_upper() > page  # the canvas is wider than the view
    start = h.get_value()
    canvas._on_press(canvas, Ev(b.x + 60, b.y + 10))
    canvas._on_motion(canvas, Ev(start + page - 5, b.y + 10))  # within the edge zone on the right
    assert canvas._edge_id is not None
    x0 = b.x
    for _ in range(10):
        canvas._edge_tick()
        pump(0.02)
    assert h.get_value() > start and b.x > x0
    canvas._on_release(canvas, Ev(h.get_value() + page - 5, b.y + 10))
    assert canvas._edge_id is None


def test_a_new_step_appears_in_the_visible_area(main, tmp_path):
    pl = main.planner
    far = pl.add_step("amber-build", 1800, 1400)  # makes the canvas larger than the view
    pump(0.3)
    h, v = pl.scroller.get_hadjustment(), pl.scroller.get_vadjustment()
    h.set_value(1000)
    v.set_value(700)
    pump(0.1)
    left, top = h.get_value(), v.get_value()
    n = pl.add_step("amber-md")
    assert left <= n.x <= left + h.get_page_size() - NODE_W
    assert top <= n.y <= top + v.get_page_size()
    assert (n.x, n.y) != (far.x, far.y)


def test_close_project_returns_to_an_empty_unsaved_plan(main, tmp_path, monkeypatch):
    pl, *_ = _plan(main, tmp_path)
    path = tmp_path / "p.mdatk.toml"
    main.project_path = str(path)
    monkeypatch.setattr(main, "_confirm_discard", lambda: True)
    main.close_project()
    pump(0.1)
    assert main.project_path is None and main.project.workflow.nodes == []
    assert main.header.get_subtitle() == "unsaved project"
    monkeypatch.setattr(main, "_confirm_discard", lambda: False)  # 'Cancel' keeps the project
    pl.add_step("amber-build", 40, 40)
    main.close_project()
    assert len(main.project.workflow.nodes) == 1


def test_dialog_text_is_selectable_and_the_button_keeps_the_focus():
    dlg = Gtk.MessageDialog(modal=True, message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.CLOSE,
                            text="Could not write the plan")
    dlg.format_secondary_text("apt1_02_custom.sh: permission denied")
    labels = selectable_text(dlg)
    assert {lb.get_text() for lb in labels} >= {"Could not write the plan",
                                                 "apt1_02_custom.sh: permission denied"}
    assert all(lb.get_selectable() for lb in labels)
    dlg.show_all()
    pump(0.3)
    assert isinstance(dlg.get_focus(), Gtk.Button)
    assert not any(lb.get_selection_bounds()[0] for lb in labels)
    dlg.destroy()


def test_every_dialog_of_the_gui_has_selectable_text():
    here = os.path.join(os.path.dirname(__file__), "..", "mdatk", "gui")
    missing = []
    for path in glob.glob(os.path.join(here, "*.py")):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            makes = any(isinstance(c.func, ast.Attribute) and c.func.attr in ("MessageDialog", "Dialog")
                        and getattr(c.func.value, "id", "") == "Gtk" for c in calls)
            if makes and not any(getattr(c.func, "id", "") == "selectable_text" for c in calls):
                missing.append(f"{os.path.basename(path)}:{fn.name}")
    assert not missing, missing


def _monitor_height():
    display = Gdk.Display.get_default()
    monitor = display.get_primary_monitor() or display.get_monitor(0)
    return monitor.get_workarea().height


def test_long_dialog_text_scrolls_and_the_dialog_fits_the_screen():
    from mdatk.gui.dialogtext import scroll_long_text

    text = "\n".join(f"• step {k:02d} (Amber MD stages): apt1_{k:02d}_amber-md.sh - 1 core"
                     for k in range(90))
    dlg = Gtk.MessageDialog(modal=True, message_type=Gtk.MessageType.QUESTION,
                            buttons=Gtk.ButtonsType.OK_CANCEL, text="Run 90 generated script(s)?")
    dlg.format_secondary_text(text)
    labels = selectable_text(dlg)
    dlg.show_all()
    pump(0.4)
    boxes = [w for w in dlg.get_message_area().get_children() if isinstance(w, Gtk.ScrolledWindow)]
    assert len(boxes) == 1 and boxes[0].get_child() is not None
    assert dlg.props.secondary_text == text  # what the dialog says is unchanged ...
    assert any(lb.get_text() == text and lb.get_selectable() for lb in labels)  # ... and still selectable
    assert dlg.get_allocated_height() <= _monitor_height()
    dlg.destroy()
    short = Gtk.MessageDialog(modal=True, buttons=Gtk.ButtonsType.OK, text="Done")
    short.format_secondary_text("one line")
    assert not scroll_long_text(short)
    short.destroy()


def test_the_run_question_for_many_steps_fits_the_screen(tmp_path, monkeypatch):
    from mdatk.gui import jobs as J
    from mdatk.workflows.runner import ScriptJob

    many = [ScriptJob(k, f"{k:02d} · Amber MD stages", str(tmp_path / f"s{k}.sh"), node_id=f"n{k}")
            for k in range(1, 61)]
    heights = []

    def run(dlg):
        dlg.show_all()
        pump(0.3)
        heights.append(dlg.get_allocated_height())
        assert any(isinstance(w, Gtk.ScrolledWindow) for w in dlg.get_message_area().get_children())
        return Gtk.ResponseType.CANCEL

    monkeypatch.setattr(Gtk.MessageDialog, "run", run)
    assert J.confirm_run(None, many, parallel=4) is None
    assert heights and heights[0] <= _monitor_height()
