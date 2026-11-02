"""GTK tests of RNAdo's CPU / GPU budget controls, the Stop scripts button and the stop button on a running
box (run under xvfb-run -a)."""

import os
import shutil
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
from mdatk.workflows import budget  # noqa: E402
from mdatk.workflows.budget import Budget, Resources, auto_cores  # noqa: E402
from mdatk.workflows.plan import connect  # noqa: E402

NEEDS_BASH = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.005)


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        pump(0.05)
    return cond()


class Ev:
    def __init__(self, x, y, button=1):
        self.x, self.y, self.button, self.state = x, y, button, 0
        self.x_root, self.y_root = x, y
        self.type = Gdk.EventType.BUTTON_PRESS


@pytest.fixture
def no_gpu(monkeypatch):
    """A computer without GPUs, whatever the test machine has."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(budget, "_smi_listing", lambda: "")


@pytest.fixture
def windows():
    made = []
    yield made
    for w in made:
        w.destroy()
    pump(0.1)


def rnado_window(windows, path=None):
    from mdatk.gui.rnado import RNAdoWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    project = Project.load(path) if path else Project()
    if not path:
        project.workflow.project_name = "apt1"
    win = RNAdoWindow(app, project, path)
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    win.show_all()
    windows.append(win)
    pump(0.3)
    return win


def saved(win, tmp_path, monkeypatch):
    from mdatk.gui import rnado

    path = str(tmp_path / "run.mdatk.toml")
    win.project.output.out_dir = str(tmp_path / "out")
    monkeypatch.setattr(rnado, "choose_save_path", lambda *_a, **_k: path)
    assert win.save_project()
    monkeypatch.setattr(Gtk.MessageDialog, "run", lambda self: Gtk.ResponseType.OK)
    return path


def test_cores_gpus_and_at_once_controls(windows, tmp_path, no_gpu):
    win = rnado_window(windows)
    pl, wf = win.planner, win.project.workflow
    assert (wf.parallel, wf.cpu_cores, wf.gpus) == (0, 0, -1)
    assert pl.cores_spin.get_text() == f"auto ({auto_cores()})"
    assert pl.parallel_spin.get_text() == "no limit"
    assert pl.gpus_combo.get_active_text() == "auto (0)"
    pl.cores_spin.set_value(4)
    pl.gpus_combo.set_active_id("2")
    pl.parallel_spin.set_value(3)
    assert (wf.parallel, wf.cpu_cores, wf.gpus) == (3, 4, 2) and pl.cores_spin.get_text() == "4"
    assert pl.budget() == Budget(4, ("0", "1"))  # two GPUs as told, none found
    # each box shows what it uses; the details say it in words
    md = pl.add_step("amber-md", 40, 40)
    namd = pl.add_step("namd-mdff", 340, 40)
    pump(0.2)
    assert pl.canvas.node_resources(md).badge() == "1c+1g"
    assert pl.canvas.node_resources(namd).badge() == "8c+1g"
    pl.canvas.select(md.id)
    assert pl.d_res.get_text().startswith("Uses 1 core, 1 GPU when it runs; RNAdo hands it a free GPU")
    pl.canvas.queue_draw()
    pump(0.2)
    # Run step… hands the jobs (with what they use), the cap and the budget to the run
    calls = []
    win.run_scripts = lambda jobs, separate=(), parent=None, **kw: calls.append((jobs, kw))
    us = pl.add_step("user-script", 40, 400)
    us.settings.update(script="true\n", cores=2, gpus=1)
    pl.changed()
    assert pl.canvas.node_resources(us).badge() == "2c+1g"
    pl.write([us.id], run=True)
    jobs, kw = calls[-1]
    assert kw["parallel"] == 3 and kw["budget"] == Budget(4, ("0", "1"))
    assert jobs[0].resources == Resources(2, 1, ())
    # saved, and read back by a new window
    path = str(tmp_path / "budget.mdatk.toml")
    win.project.save(path)
    again = rnado_window(windows, path)
    assert again.planner.cores_spin.get_value() == 4 and again.planner.gpus_combo.get_active_id() == "2"
    assert again.planner.parallel_spin.get_value() == 3


def test_an_old_project_keeps_its_parallel_value(windows, tmp_path, no_gpu):
    old = tmp_path / "old.mdatk.toml"
    old.write_text('[workflow]\nproject_name = "apt1"\nparallel = 2\n')
    win = rnado_window(windows, str(old))
    pl = win.planner
    assert pl.parallel_spin.get_value() == 2 and pl.parallel_spin.get_text() == "2"
    assert pl.cores_spin.get_value() == 0 and pl.gpus_combo.get_active_id() == "-1"


@NEEDS_BASH
def test_stop_scripts_stops_every_running_script(windows, tmp_path, monkeypatch, no_gpu):
    win = rnado_window(windows)
    pl = win.planner
    a = pl.add_step("user-script", 40, 40)
    a.settings["script"] = "sleep 30\n"
    b = pl.add_step("user-script", 40, 300)
    b.settings["script"] = "sleep 30\n"
    pl.changed()
    saved(win, tmp_path, monkeypatch)
    assert not pl.stop_button.get_sensitive()  # nothing runs
    assert pl.stop_button.get_style_context().has_class("destructive-action")
    pl.write(None, run=True)  # Run all…
    assert wait_for(lambda: pl.run_state == {a.id: "running", b.id: "running"}, 10), pl.run_state
    assert pl.stop_button.get_sensitive()
    t0 = time.monotonic()
    pl.stop_button.clicked()
    assert wait_for(lambda: not win.jobs.running, 15)
    assert time.monotonic() - t0 < 12
    pump(0.2)
    assert pl.run_state == {a.id: "cancelled", b.id: "cancelled"}
    assert not pl.stop_button.get_sensitive()
    assert "Stopping every running script" in win.log.text()


@NEEDS_BASH
def test_the_stop_button_on_a_running_box_stops_that_step_only(windows, tmp_path, monkeypatch, no_gpu):
    win = rnado_window(windows)
    pl = win.planner
    a = pl.add_step("user-script", 40, 40)
    a.settings.update(script='echo started > "$MDATK_OUT_RESULT"\nsleep 30\n',
                      outputs=[{"name": "result", "kind": "text"}])
    b = pl.add_step("user-script", 40, 320)
    b.settings["script"] = "sleep 1.5\necho b done\n"
    c = pl.add_step("user-script", 340, 40)
    c.settings.update(script="echo c ran\n", inputs=[{"name": "model", "kind": "any"}])
    connect(win.project.workflow, a.id, "result", c.id, "model")
    pl.changed()
    pump(0.2)
    assert not [i.message for _n, i in pl.plan.all_issues() if i.level == "error"]
    saved(win, tmp_path, monkeypatch)
    pl.write(None, run=True)  # Run all…
    assert wait_for(lambda: pl.run_state.get(a.id) == "running", 10), pl.run_state
    canvas = pl.canvas
    pump(0.3)  # the running box draws its stop button
    # the box's context menu offers it too
    shown = []
    monkeypatch.setattr(Gtk.Menu, "popup_at_pointer", lambda menu, _ev: shown.append(menu))
    canvas._context_menu(Ev(a.x + 100, a.y + 60, 3), a.x + 100, a.y + 60)
    labels = [item.get_label() for item in shown[-1].get_children() if isinstance(item, Gtk.MenuItem)
              and not isinstance(item, Gtk.SeparatorMenuItem)]
    assert labels[0] == "Stop this step"
    shown[-1].destroy()
    # a click on the stop square in the box's header
    bx, by, bw, bh = canvas.stop_rect(a)
    assert canvas.stop_at(bx + bw / 2, by + bh / 2) is a and canvas.stop_at(a.x + 20, a.y + 60) is None
    canvas._on_press(canvas, Ev((bx + bw / 2) * canvas.zoom, (by + bh / 2) * canvas.zoom))
    assert wait_for(lambda: any(m[1] == "Scripts finished" for m in win.messages), 20), win.messages
    pump(0.3)
    assert pl.run_state == {a.id: "stopped", b.id: "ok", c.id: "skipped"}
    assert ("info", "Scripts finished", "3 script(s) finished; 1 stopped, 1 not started") in win.messages
    text = win.log.text()
    for line in ("Stopping step 01 (01 · Custom script)", "==== step 01: 01 · Custom script - stopped",
                 "[02] b done", "==== step 03: 03 · Custom script - not started (step 01 was stopped)",
                 "Run finished: 1 script(s) ok, 0 waiting, 1 not started, 1 stopped"):
        assert line in text, (line, text)
    assert "c ran" not in text
    # the step log records it, and 'Check previous runs' calls the run partial
    from mdatk.projectfile import read_log
    entry = [e for e in read_log(win.project_path) if e.get("step") == "01"][-1]
    assert entry["status"] == "stopped" and entry["command"].startswith("bash ")
    runs = pl.check_runs([a.id])
    assert runs[a.id].state == "partial" and runs[a.id].stopped
    assert canvas.run_info(a)[0].startswith("partial (stopped) · ")
    canvas.queue_draw()
    pump(0.2)
