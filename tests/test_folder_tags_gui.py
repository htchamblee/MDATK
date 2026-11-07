"""GTK test (xvfb-run -a): RNAdo's details pane has a 'Folder tag' entry next to Title and Step, the step's
box shows the tag, and the planner's own steps (Write with its MANIFEST, Check previous runs, the wizard)
use the tagged folder."""

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

import cairo  # noqa: E402

from mdatk.config import Project  # noqa: E402
from mdatk.workflows import runstate  # noqa: E402
from mdatk.workflows.manifest import IDENTITY, MANIFEST, read_identity  # noqa: E402
from mdatk.workflows.plan import connect  # noqa: E402
from mdatk.workflows.userscript import DONE_TEXT  # noqa: E402


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.005)


def wait_for(cond, timeout=20.0):
    """Pump GTK until ``cond()`` (the planner resolves an edit 250 ms after it, later on a busy machine)."""
    end = time.time() + timeout
    while time.time() < end and not cond():
        pump(0.05)
    return cond()


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
    pump(0.1)


def form_labels(entry):
    """The labels of the details form, top to bottom, and the label of ``entry``'s row."""
    grid = entry.get_parent()
    rows = {}
    for child in grid.get_children():
        top = grid.child_get_property(child, "top-attach")
        left = grid.child_get_property(child, "left-attach")
        rows.setdefault(top, {})[left] = child
    order = [rows[t][0].get_text() for t in sorted(rows)]
    mine = next(rows[t][0].get_text() for t in rows if rows[t].get(1) is entry)
    return order, mine


def drawn_badges(canvas, node):
    """The badge texts drawn on ``node``'s box."""
    texts = []
    real = canvas._draw_badge

    def spy(cr, text, *args, **kwargs):
        texts.append(text)
        return real(cr, text, *args, **kwargs)

    canvas._draw_badge = spy
    try:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 300)
        cr = cairo.Context(surface)
        cr.translate(20 - node.x, 20 - node.y)
        canvas._draw_node(cr, node)
    finally:
        del canvas._draw_badge
    return texts


def test_the_folder_tag_entry_names_the_folder_and_the_planner_uses_it(main, tmp_path):
    pl = main.planner
    src = pl.add_step("user-script", 40, 40)
    src.settings.update(script='echo m > "$MDATK_OUT_M"\n', outputs=[{"name": "m", "kind": "text"}])
    b = pl.add_step("user-script", 400, 40)
    b.settings.update(script='cat "$MDATK_IN_I" > "$MDATK_OUT_O"\n', inputs=[{"name": "i", "kind": "text"}],
                      outputs=[{"name": "o", "kind": "text"}])
    f = pl.add_step("input-file", 40, 300)
    f.settings.update(kind="pdb", path=str(tmp_path / "m.pdb"))
    connect(pl.workflow, src.id, "m", b.id, "i")
    pl.changed()
    pl.canvas.select(b.id)
    pump(0.3)
    # the entry: right after Title and Step, explained by its tooltip
    order, mine = form_labels(pl.d_tag)
    assert order[:3] == ["Title", "Step", "Folder tag"] and mine == "Folder tag"
    tip = pl.d_tag.get_tooltip_text()
    assert ("Added to the step's folder name: 'model12' makes 02_amber-build into "
            "02_amber-build_model12") in tip
    assert "File names do not change" in tip and "files already written stay where they are" in tip
    assert pl.d_tag.get_sensitive() and pl.d_tag.get_text() == ""
    assert "Folder: 02_custom/" in pl.d_files.get_text()
    assert "model12" not in drawn_badges(pl.canvas, b)
    # typing a tag: the step's folder (and the files the next steps read) carry it
    pl.d_tag.set_text("model12")
    assert b.folder_tag == "model12"
    assert wait_for(lambda: "Folder: 02_custom_model12/" in pl.d_files.get_text())
    assert pl.plan.steps[b.id].ctx.folder == "02_custom_model12"
    assert "model12" in drawn_badges(pl.canvas, b)  # the box shows it
    # a tag that cannot name a folder: the entry and the checks say so
    pl.d_tag.set_text("model 12")
    icon = Gtk.EntryIconPosition.SECONDARY
    assert wait_for(lambda: pl.d_tag.get_icon_name(icon) == "dialog-error-symbolic")
    assert any("folder tag 'model 12': use ASCII letters and digits, with '-' only between them" in i.message
               for i in pl.plan.steps[b.id].issues if i.level == "error")
    pl.d_tag.set_text("model12")
    assert wait_for(lambda: pl.d_tag.get_icon_name(Gtk.EntryIconPosition.SECONDARY) is None)
    # an existing file writes nothing: no folder, so no tag
    pl.canvas.select(f.id)
    pump(0.3)
    assert not pl.d_tag.get_sensitive()
    assert "No folder: this step writes no files" in pl.d_files.get_text()
    # Write all files: the tagged folder with its identity file, and MANIFEST, named in the log
    pl.canvas.select(b.id)
    pump(0.2)
    pl.write(None)
    root = main.project.workflow_dir
    ident = read_identity(os.path.join(root, "02_custom_model12"))
    assert (ident["node"], ident["folder_tag"], ident["step"]) == (b.id, "model12", 2)
    assert os.path.isfile(os.path.join(root, "01_custom", IDENTITY))
    assert not [n for n in os.listdir(root) if n.startswith("00_")]  # the existing file has no folder
    rows = open(os.path.join(root, MANIFEST)).read().splitlines()
    assert any(r.startswith(f"02\t02_custom_model12\t{b.id}\tuser-script\t") for r in rows)
    assert f"Manifest: {os.path.join(root, MANIFEST)}" in main.log.text()
    # Check previous runs looks in the tagged folder
    plan = pl.resolve()
    step = plan.steps[b.id]
    for path in runstate.expected(plan, b.id, plan.stamp):
        with open(plan.node_path(path), "w") as fh:
            fh.write("x\n")
    for path, _pattern in step.tool.finish_marks(step.settings, step.ctx):
        with open(plan.node_path(path), "w") as fh:
            fh.write(DONE_TEXT + "\n")
    runs = pl.scan_runs()
    assert runs[b.id].state == "done" and all(p.startswith("02_custom_model12/") for p in runs[b.id].found)
    # the step's wizard writes into (and opens) the tagged folder
    pl.open_wizard(b)
    pump(0.3)
    win = pl.wizards[b.id]
    assert win.step_folder() == os.path.join(root, "02_custom_model12")
    win.destroy()
    pump(0.1)


def fake_run(plan, nid):
    """The output files and finished logs of a completed run of step ``nid`` (the plan's stamp)."""
    step = plan.steps[nid]
    for path in runstate.expected(plan, nid, plan.stamp):
        with open(plan.node_path(path), "w") as fh:
            fh.write("x\n")
    for path, _pattern in step.tool.finish_marks(step.settings, step.ctx):
        with open(plan.node_path(path), "w") as fh:
            fh.write(DONE_TEXT + "\n")
    assert runstate.run_status(plan, nid, plan.stamp).state == "done"


def test_a_folder_tag_on_a_step_that_writes_nothing_can_be_cleared(main, tmp_path):
    pl = main.planner
    f = pl.add_step("input-file", 40, 40)
    f.settings.update(kind="pdb", path=str(tmp_path / "m.pdb"))
    f.folder_tag = "model12"  # e.g. from a project file a plan generator wrote, tagging a whole branch
    pl.changed()
    pl.canvas.select(f.id)
    pump(0.3)
    assert any(i.level == "warning" and "folder tag 'model12'" in i.message for i in pl.plan.steps[f.id].issues)
    assert pl.d_tag.get_sensitive() and pl.d_tag.get_text() == "model12"
    pl.d_name.set_text("renamed")  # another field: the tag stays
    assert wait_for(lambda: f.title == "renamed") and f.folder_tag == "model12"
    pl.d_tag.set_text("")  # cleared: the warning goes, and with no tag the entry has nothing to edit
    assert f.folder_tag == ""
    assert wait_for(lambda: not any("folder tag" in i.message for i in pl.plan.steps[f.id].issues))
    assert wait_for(lambda: not pl.d_tag.get_sensitive())


def test_a_folder_holding_another_steps_files_is_never_offered_and_the_question_says_so(main, monkeypatch,
                                                                                          tmp_path):
    from mdatk.gui import jobs, planner as planner_mod, rnado
    from mdatk.workflows.plan import remove_node

    monkeypatch.setattr(rnado, "choose_save_path", lambda *_a, **_k: str(tmp_path / "s.mdatk.toml"))
    pl = main.planner
    old = pl.add_step("user-script", 40, 40)
    old.title = "model 13"
    old.settings.update(script='echo m > "$MDATK_OUT_M"\n', outputs=[{"name": "m", "kind": "text"}])
    pl.changed()
    assert main.save_project()
    pl.write(None)
    fake_run(pl.resolve(), old.id)  # model 13's run, in 01_custom/
    pl.canvas.select(None)
    remove_node(pl.workflow, old.id)  # its box is deleted, its folder stays
    new = pl.add_step("user-script", 40, 40)
    new.title = "model 14"
    new.settings.update(script='echo m > "$MDATK_OUT_M"\n', outputs=[{"name": "m", "kind": "text"}])
    pl.changed()
    assert new.id != old.id and new.step == old.step  # a new id, the number (and folder) of before
    held = pl.plan.steps[new.id].held
    assert held.startswith(f"01_custom/ holds the files of 01 · model 13 (node {old.id}, written 20")
    assert any(i.level == "warning" and i.message.startswith(held) for i in pl.plan.steps[new.id].issues)
    # Check previous runs: the run is found, said to be model 13's, and not offered (a warning instead)
    asked, told = [], []
    monkeypatch.setattr(pl, "_ask", lambda text, detail: asked.append(text) or True)
    monkeypatch.setattr(pl, "_message", lambda text, detail="", kind=None: told.append((text, detail)))
    runs = pl.check_runs(None)
    assert runs[new.id].state == "done" and runs[new.id].held == held
    assert asked == [] and new.adopted == ""
    assert told and told[0][0] == "1 run(s) found are not their steps' own" and held in told[0][1]
    assert f"  01 · model 14: done: run {pl.plan.stamp}" in main.log.text()
    assert f"- not this step's run: {held}" in main.log.text()
    assert pl.mark_of(new.id) is None  # no check mark for another step's run
    pl.canvas.select(new.id)
    pump(0.3)
    assert not pl.d_adopt.get_sensitive()
    # Write: the confirmation says whose run would move aside
    shown = []

    class Dialog:
        def __init__(self, **kwargs):
            shown.append(kwargs.get("text", ""))

        def format_secondary_text(self, text):
            shown.append(text)

        def add_buttons(self, *args):
            pass

        def get_widget_for_response(self, response):
            return None

        def set_default_response(self, response):
            pass

        def run(self):
            return Gtk.ResponseType.CANCEL

        def destroy(self):
            pass

    monkeypatch.setattr(jobs.Gtk, "MessageDialog", Dialog)
    monkeypatch.setattr(jobs, "selectable_text", lambda dlg: None)
    assert planner_mod.confirm_supersede is jobs.confirm_supersede
    pl.write(None)  # cancelled: nothing moves
    detail = shown[1]
    assert f"  not this step's run: {held}; it would move aside as this step's" in detail
    assert "1 of these folders hold the files of another step" in detail
    assert read_identity(os.path.join(main.project.workflow_dir, "01_custom"))["node"] == old.id


def test_steps_sharing_a_number_keep_their_own_step_log(main, monkeypatch, tmp_path):
    from mdatk.gui import rnado
    from mdatk.projectfile import LogEntry, append_log

    monkeypatch.setattr(rnado, "choose_save_path", lambda *_a, **_k: str(tmp_path / "s.mdatk.toml"))
    pl = main.planner
    a = pl.add_step("user-script", 40, 40)
    b = pl.add_step("user-script", 40, 300)
    a.settings.update(script="true\n")
    b.settings.update(script="true\n")
    a.folder_tag, b.folder_tag = "model12", "model13"
    b.step = a.step
    pl.changed()
    pump(0.3)
    assert main.save_project()
    labels = [pl.plan.steps[n.id].label for n in (a, b)]
    assert labels == ["01 · Custom script · model12", "01 · Custom script · model13"]
    for label in labels:  # what the runner logs: the step number and the step's title
        append_log(main.project_path, LogEntry(step="01", title=label, summary=f"ran {label}"))
    assert "ran 01 · Custom script · model12" in main.stored_log(1, labels[0])
    assert "ran 01 · Custom script · model13" in main.stored_log(1, labels[1])
    # the details pane names the tag; the box shows its title without it, and the tag beside it
    pl.canvas.select(a.id)
    pump(0.3)
    assert "01 · Custom script · model12" in pl.d_title.get_text()
    drawn = []
    real = pl.canvas._text

    def spy(cr, text, *args, **kwargs):
        drawn.append(text)
        return real(cr, text, *args, **kwargs)

    pl.canvas._text = spy
    try:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 300)
        cr = cairo.Context(surface)
        cr.translate(20 - a.x, 20 - a.y)
        pl.canvas._draw_node(cr, a)
    finally:
        del pl.canvas._text
    assert "01 · Custom script" in drawn and "01 · Custom script · model12" not in drawn
    assert "model12" in drawn_badges(pl.canvas, a)
