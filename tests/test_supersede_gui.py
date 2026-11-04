"""GTK tests (xvfb-run -a): RNAdo asks before a write moves completed runs aside, lists them in the output log
and records them in the project's step log (mdatk.workflows.supersede)."""

import os
import shutil
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

from mdatk.config import Project  # noqa: E402
from mdatk.projectfile import read_log  # noqa: E402
from mdatk.workflows import runstate, supersede  # noqa: E402
from mdatk.workflows.plan import connect  # noqa: E402
from mdatk.workflows.userscript import DONE_TEXT  # noqa: E402

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


def fake_run(plan, nid):
    """The output files and the completion marker of a finished run of step ``nid``."""
    step = plan.steps[nid]
    for path in runstate.expected(plan, nid, plan.stamp):
        with open(plan.node_path(path), "w") as fh:
            fh.write("x\n")
    for path, _pattern in step.tool.finish_marks(step.settings, step.ctx):
        with open(plan.node_path(path), "w") as fh:
            fh.write(DONE_TEXT + "\n")
    assert runstate.run_status(plan, nid, plan.stamp).state == "done"


@pytest.fixture
def win(tmp_path, monkeypatch):
    from mdatk.gui import rnado
    from mdatk.gui.rnado import RNAdoWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    p.workflow.project_name = "apt1"
    w = RNAdoWindow(app, p, None)
    w.messages = []
    w.info = lambda t, d="": w.messages.append(("info", t, d))
    w.error = lambda t, d="": w.messages.append(("error", t, d))
    w.show_all()
    pump(0.3)
    monkeypatch.setattr(rnado, "choose_save_path", lambda *_a, **_k: str(tmp_path / "s.mdatk.toml"))
    yield w
    w.destroy()
    pump(0.1)


def answering(monkeypatch, response):
    """Answer every message dialog with ``response``; returns the (question, detail) pairs asked."""
    asked = []

    def run(dlg):
        asked.append((dlg.props.text, dlg.props.secondary_text or ""))
        return response

    monkeypatch.setattr(Gtk.MessageDialog, "run", run)
    return asked


def runs_aside(folder):
    path = os.path.join(folder, "superseded")
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def test_write_all_asks_lists_and_logs_the_runs_it_moves(win, monkeypatch):
    pl = win.planner
    a = pl.add_step("user-script", 40, 40)
    a.settings.update(script='echo a > "$MDATK_OUT_T"\n', outputs=[{"name": "t", "kind": "text"}])
    b = pl.add_step("user-script", 40, 300)
    b.settings.update(script="true\n")
    pl.changed()
    assert win.save_project()
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write(None)  # nothing has run: nothing to move, nothing to ask
    assert asked == []
    stamp = win.project.workflow.stamp
    folder = os.path.join(win.project.workflow_dir, "01_custom")
    fake_run(pl.resolve(), a.id)
    before = sorted(os.listdir(folder))
    # Write all files: the question lists the step, where its run goes and its files; Cancel writes nothing
    asked = answering(monkeypatch, Gtk.ResponseType.CANCEL)
    pl.write(None)
    [(text, detail)] = asked
    assert text == "Move the completed run(s) of 1 step(s) aside and write them?"
    assert f"01 · Custom script: completed run {stamp}, finished " in detail
    assert f"(4 files) -> 01_custom/superseded/{stamp}_<time>/" in detail
    assert f"  apt1_01_t_{stamp}.txt" in detail and f"  apt1_01_custom_{stamp}.done" in detail
    assert "02 · Custom script" not in detail  # it has not run
    assert sorted(os.listdir(folder)) == before and runs_aside(folder) == []
    # Move aside and write
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write(None)
    assert len(asked) == 1
    [run] = runs_aside(folder)
    assert run.startswith(f"{stamp}_")
    assert sorted(os.listdir(os.path.join(folder, "superseded", run))) == sorted(before + ["manifest.toml"])
    log = win.log.text()
    assert f"Planner: 01 · Custom script: moved its completed run {stamp} aside (4 files) to " \
           f"01_custom/superseded/{run}/" in log
    assert f"  apt1_01_t_{stamp}.txt" in log
    assert "1 completed runs moved aside" in pl.status.get_text()
    [entry] = [e for e in read_log(win.project_path) if e["step"] == supersede.LOG_STEP]
    assert entry["title"] == "Moved the completed run of 01 · Custom script aside"
    assert entry["status"] == "ok"
    # the project option off: no question, and the run stays where it is
    fake_run(pl.resolve(), a.id)
    pl.supersede_check.set_active(False)
    pump(0.3)
    assert win.project.workflow.supersede_runs is False
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write(None)
    assert asked == [] and runs_aside(folder) == [run]
    assert os.path.exists(os.path.join(folder, f"apt1_01_t_{stamp}.txt"))
    pl.supersede_check.set_active(True)
    pump(0.3)
    assert win.save_project()
    assert Project.load(win.project_path).workflow.supersede_runs is True
    # the script wizard asks too
    pl.open_wizard(a)
    pump(0.5)
    wz = pl.wizards[a.id]
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    wz.write_files()
    pump(0.2)
    assert [t for t, _d in asked] == ["Move the completed run(s) of 1 step(s) aside and write them?"]
    assert len(runs_aside(folder)) == 2
    assert f"Script wizard: 01 · Custom script: moved its completed run {stamp} aside" in win.log.text()
    # nothing moves while a step that reads those outputs runs (or waits to); other runs do not matter
    fake_run(pl.resolve(), a.id)
    pl.run_state[b.id] = "running"
    asked = answering(monkeypatch, Gtk.ResponseType.CANCEL)
    pl.write(None)  # b does not read a: the usual question (cancelled here)
    assert [t for t, _d in asked] == ["Move the completed run(s) of 1 step(s) aside and write them?"]
    b.settings.update(inputs=[{"name": "t", "kind": "text"}])
    connect(win.project.workflow, a.id, "t", b.id, "t")
    pl.changed()
    pl.run_state[b.id] = "queued"
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write(None)
    assert [t for t, _d in asked] == ["Scripts are running"]
    assert "01 · Custom script" in asked[0][1] and "while 02 · Custom script may read them" in asked[0][1]
    assert len(runs_aside(folder)) == 2 and os.path.exists(os.path.join(folder, f"apt1_01_t_{stamp}.txt"))
    pl.run_state.clear()


@NEEDS_BASH
def test_run_step_asks_before_it_writes(win, monkeypatch):
    pl = win.planner
    a = pl.add_step("user-script", 40, 40)
    a.settings.update(script='echo a > "$MDATK_OUT_T"\n', outputs=[{"name": "t", "kind": "text"}])
    pl.changed()
    assert win.save_project()
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write([a.id], run=True)  # Run step…: only the run's own question
    assert wait_for(lambda: any(m[1] == "Scripts finished" for m in win.messages)), win.messages
    assert [t for t, _d in asked] == ["Run 1 generated script(s)?"]
    stamp = win.project.workflow.stamp
    folder = os.path.join(win.project.workflow_dir, "01_custom")
    assert runstate.run_status(pl.resolve(), a.id, stamp).state == "done"
    # running it again writes it first: the finished run is moved aside, then it runs again
    win.messages.clear()
    asked = answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write([a.id], run=True)
    assert wait_for(lambda: any(m[1] == "Scripts finished" for m in win.messages)), win.messages
    assert [t for t, _d in asked] == ["Move the completed run(s) of 1 step(s) aside and write and run them?",
                                      "Run 1 generated script(s)?"]
    [run] = runs_aside(folder)
    assert open(os.path.join(folder, "superseded", run, f"apt1_01_t_{stamp}.txt")).read() == "a\n"
    assert open(os.path.join(folder, f"apt1_01_t_{stamp}.txt")).read() == "a\n"  # the new run's


def test_a_run_that_completes_during_the_question_is_neither_moved_nor_written(win, monkeypatch):
    # step 01 done, step 02 running, step 03 (reads 02) queued: the question lists only 01; while it is
    # open (the runner works on) 02 completes and 03 starts reading it.  02's run must stay in place.
    pl = win.planner
    x = pl.add_step("user-script", 40, 40)
    x.settings.update(script='echo x > "$MDATK_OUT_T"\n', outputs=[{"name": "t", "kind": "text"}])
    a = pl.add_step("user-script", 300, 40)
    a.settings.update(script='echo a > "$MDATK_OUT_T"\n', outputs=[{"name": "t", "kind": "text"}])
    b = pl.add_step("user-script", 300, 300)
    b.settings.update(script="true\n", inputs=[{"name": "t", "kind": "text"}])
    connect(win.project.workflow, a.id, "t", b.id, "t")
    pl.changed()
    assert win.save_project()
    answering(monkeypatch, Gtk.ResponseType.OK)
    pl.write(None)
    stamp = win.project.workflow.stamp
    fake_run(pl.resolve(), x.id)
    pl.run_state.update({a.id: "running", b.id: "queued"})
    folder_x = os.path.join(win.project.workflow_dir, "01_custom")
    folder_a = os.path.join(win.project.workflow_dir, "02_custom")
    asked = []

    def run(dlg):
        asked.append((dlg.props.text, dlg.props.secondary_text or ""))
        if len(asked) == 1:  # the question is open: step 02 finishes, step 03 starts reading its output
            fake_run(pl.resolve(), a.id)
            pl.run_state.update({a.id: "done", b.id: "running"})
        return Gtk.ResponseType.OK

    monkeypatch.setattr(Gtk.MessageDialog, "run", run)
    pl.write(None)
    assert asked[0][0] == "Move the completed run(s) of 1 step(s) aside and write them?"
    assert "01 · Custom script" in asked[0][1] and "02 · Custom script" not in asked[0][1]
    assert len(runs_aside(folder_x)) == 1  # the run that was listed moved
    assert runs_aside(folder_a) == []  # the one that completed during the question did not
    assert os.path.exists(os.path.join(folder_a, f"apt1_02_t_{stamp}.txt"))
    assert runstate.run_status(pl.resolve(), a.id, stamp).state == "done"
    assert asked[1][0] == "1 step(s) not written" and "02 · Custom script" in asked[1][1]
    assert "Planner: 02 · Custom script not written: its run completed after the question" in win.log.text()
    pl.run_state.clear()


def test_the_question_can_delete_the_run_instead(win, monkeypatch):
    from mdatk.gui.dialogtext import _buttons
    from mdatk.gui.jobs import DELETE_RESPONSE

    pl = win.planner
    a = pl.add_step("user-script", 40, 40)
    a.settings.update(script='echo a > "$MDATK_OUT_T"\n', outputs=[{"name": "t", "kind": "text"}])
    pl.changed()
    assert win.save_project()
    pl.write(None)
    stamp = win.project.workflow.stamp
    folder = os.path.join(win.project.workflow_dir, "01_custom")
    fake_run(pl.resolve(), a.id)
    assert os.path.exists(os.path.join(folder, f"apt1_01_t_{stamp}.txt"))
    buttons = []

    def run(dlg):
        found = []
        _buttons(dlg, found)
        buttons.extend(b.get_label() for b in found)
        return DELETE_RESPONSE

    monkeypatch.setattr(Gtk.MessageDialog, "run", run)
    pl.write(None)
    assert "Delete them and write" in buttons and "Move aside and write" in buttons
    assert runs_aside(folder) == [] and not os.path.exists(os.path.join(folder, f"apt1_01_t_{stamp}.txt"))
    assert os.path.exists(os.path.join(folder, f"apt1_01_custom_{stamp}.sh"))  # written again
    assert f"Planner: 01 · Custom script: deleted its completed run {stamp} (4 files)" in win.log.text()
    assert "1 completed runs deleted" in pl.status.get_text()
    [entry] = [e for e in read_log(win.project_path) if e["step"] == supersede.LOG_STEP]
    assert entry["title"] == "Deleted the completed run of 01 · Custom script"
