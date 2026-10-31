"""GTK tests of RNAdo, MDATK's workflow planner, as a program of its own: its window, the output log panel,
'Open in RNAdo…' in the analysis window, and both programs editing one project file (xvfb-run -a)."""

import os
import shutil
import sys
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
from mdatk.workflows.plan import add_node, apply_template, remove_node  # noqa: E402

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


def capture(win):
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    return win


def project_file(tmp_path, name="co.mdatk.toml", template=None):
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    p.workflow.project_name = "apt1"
    if template:
        nodes = {n.tool: n for n in apply_template(p.workflow, template, 40.0, 40.0)}
        nodes["amber-build"].files["structure"] = str(tmp_path / "model.pdb")
        nodes["amber-restraints"].settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    path = str(tmp_path / name)
    p.save(path)
    return path


@pytest.fixture
def app():
    a = Gtk.Application(application_id=None)
    a.register(None)
    return a


@pytest.fixture
def windows():
    made = []
    yield made
    for w in made:
        w.destroy()
    pump(0.1)


def rnado_window(app, windows, path=None):
    from mdatk.gui.rnado import RNAdoWindow

    win = RNAdoWindow(app, Project.load(path) if path else None, path)
    capture(win)
    win.show_all()
    windows.append(win)
    pump(0.3)
    return win


def analysis_window(app, windows, path):
    from mdatk.gui.app import MainWindow

    win = MainWindow(app, Project.load(path), path)
    capture(win)
    win.show_all()
    windows.append(win)
    pump(0.5)
    return win


def bound(win, path):
    return next(b.widget for b in win.binder.bindings if b.path == path)


def tag_names(win, text):
    """The highlight tags on the first line of RNAdo's log that contains ``text``."""
    buf = win.log.buffer
    found = buf.get_start_iter().forward_search(text, 0, None)
    assert found, text
    return {t.props.name for t in found[0].get_tags()}


# ---------------------------------------------------------------------------
# the program
# ---------------------------------------------------------------------------


def test_rnado_opens_without_and_with_a_project_and_no_analysis_window(tmp_path, monkeypatch):
    from mdatk.gui import rnado

    def analysis_windows():
        return [w for w in Gtk.Window.list_toplevels()
                if type(w).__name__ == "MainWindow" and w.get_visible()]

    a = rnado.RNAdoApplication(None, None)
    a.register(None)
    a.activate()
    pump(0.3)
    win = a.window
    try:
        assert win.get_title() == "RNAdo" and win.header.get_subtitle() == "unsaved project"
        assert win.project_path is None and win.project.workflow.nodes == [] and not win.plan_changed()
        assert win.get_icon_list() or win.get_icon_name() == rnado.FALLBACK_ICON
        assert a.get_application_id() == rnado.APP_ID != "org.mdatk.MDATK"
        assert "ready" in win.log.text() and not analysis_windows()
    finally:
        win.destroy()
    # with a project and a tool: the plan, its file and the wizard; no RNAdo assets: a themed icon
    monkeypatch.setattr(rnado, "asset_path", lambda name: str(tmp_path / "no-assets" / name))
    path = project_file(tmp_path, template="amber")
    a = rnado.RNAdoApplication(path, "wizard")
    a.register(None)
    a.activate()
    pump(0.3)
    win = a.window
    try:
        assert win.project_path == path and win.header.get_subtitle() == path
        assert {n.tool for n in win.project.workflow.nodes} == {"amber-build", "amber-restraints", "amber-md",
                                                                "mdatk"}
        assert win.wizard is not None and win.wizard.get_transient_for() is win
        assert win.get_icon_name() == rnado.FALLBACK_ICON and not analysis_windows()
        assert {"01", "02", "03"} <= {win.log.step_combo.get_model()[i][1] for i in range(4)}
    finally:
        win.destroy()


def test_header_tools_and_about(app, windows, monkeypatch):
    win = rnado_window(app, windows)
    win.lookup_action("wizard").activate(None)  # Ctrl+Shift+W
    win.open_tool("pdb")
    win.open_tool("builder")
    pump(0.3)
    assert win.wizard and win.pdb_tools and win.builder
    for tool in (win.wizard, win.pdb_tools, win.builder):
        assert tool.get_transient_for() is win and "RNAdo" in tool.get_title()
    shown = []
    monkeypatch.setattr(Gtk.Dialog, "run",
                        lambda dlg: shown.append(dlg.get_title()) or Gtk.ResponseType.CLOSE)
    win.show_about()
    assert shown == ["About RNAdo"]
    win.destroy()
    pump(0.2)
    assert win.wizard is None and win.pdb_tools is None and win.builder is None


# ---------------------------------------------------------------------------
# the output log panel
# ---------------------------------------------------------------------------


@NEEDS_BASH
def test_a_custom_script_runs_from_rnado_into_the_log_panel(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import rnado

    win = rnado_window(app, windows)
    path = str(tmp_path / "run.mdatk.toml")
    monkeypatch.setattr(rnado, "choose_save_path", lambda *_a, **_k: path)
    pl = win.planner
    a = pl.add_step("user-script", 40, 40)
    a.settings["script"] = 'echo "hello from a"\necho "oops on stderr" >&2\n'
    b = pl.add_step("user-script", 40, 300)
    b.settings["script"] = 'echo "line of b"\n'
    pl.changed()
    pump(0.2)
    assert not [i.message for _n, i in pl.plan.all_issues() if i.level == "error"]
    assert win.save_project() and win.project_path == path  # the runs are logged in the project file
    monkeypatch.setattr(Gtk.MessageDialog, "run", lambda self: Gtk.ResponseType.OK)
    win.log.set_expanded(False)
    pl.write(None, run=True)  # Run all…
    assert win.log.expanded  # shown when the run starts
    assert wait_for(lambda: any(m[1] == "Scripts finished" for m in win.messages)), win.messages
    text = win.log.text()
    for line in ("[01] hello from a", "[01] oops on stderr", "[02] line of b", "Planner: Wrote ",
                 "==== step 01: 01 · Custom script - bash ", "==== step 02: 02 · Custom script - finished",
                 "Run finished: 2 script(s) ok"):
        assert line in text, (line, text)
    rec = next(r for r in win.log.records if r.text == "oops on stderr")
    assert (rec.step, rec.kind) == (1, "stderr") and "stderr" in tag_names(win, "oops on stderr")
    assert "mark" in tag_names(win, "[02] line of b")
    assert pl.run_state == {a.id: "ok", b.id: "ok"}
    # the step filter
    win.log.set_filter(2)
    shown = win.log.text()
    assert "[02] line of b" in shown and "[01]" not in shown and "Planner:" not in shown
    assert "step 02" in win.log.step_combo.get_active_text()
    win.log.set_filter(None)
    assert "[01] hello from a" in win.log.text()
    # the stored log of a selected step's last run
    pl.canvas.select(None)
    assert not win.log.stored_btn.get_sensitive()
    pl.canvas.select(a.id)
    assert win.log.stored_btn.get_sensitive()
    win.log.stored_btn.set_active(True)
    stored = win.log.text()
    assert stored.startswith("Step 01 · 01 · Custom script · ok · started") and "hello from a" in stored
    assert "line of b" not in stored and not win.log.step_combo.get_sensitive()
    pl.canvas.select(None)  # nothing selected: the live lines again
    assert not win.log.stored_btn.get_active() and "[02] line of b" in win.log.text()
    # copy, save as text, clear
    win.log.copy()
    saved = win.log.save_as(str(tmp_path / "out.txt"))
    assert "[02] line of b" in open(saved).read()
    win.log.clear()
    assert win.log.records == [] and win.log.text() == ""


def test_the_log_panel_stays_bounded_and_batches(app, windows):
    from mdatk.gui.logpanel import KEEP_LINES, MAX_LINES

    win = rnado_window(app, windows)
    panel = win.log
    pump(0.2)
    flushes = []
    real = panel._flush
    panel._flush = lambda: flushes.append(1) or real()
    before = len(panel.records)
    for i in range(MAX_LINES + 500):  # e.g. from a script's thread: queued, inserted together
        panel.feed_script(f"[05] line {i}")
    assert len(panel.records) == before and not flushes
    pump(0.3)
    assert len(flushes) == 1
    assert len(panel.records) == KEEP_LINES and panel.buffer.get_line_count() == KEEP_LINES + 1
    assert panel.records[-1].text == f"line {MAX_LINES + 499}" and panel.records[0].step == 5
    panel.autoscroll.set_active(False)
    panel.feed("one more")
    pump(0.2)
    assert panel.text().endswith("one more\n")


@NEEDS_BASH
def test_a_separately_run_step_shows_its_waiting_line(app, windows, tmp_path, monkeypatch):
    from mdatk.workflows.base import WAITING_STATUS
    from mdatk.workflows.runner import ScriptJob

    win = rnado_window(app, windows)
    pl = win.planner
    n = pl.add_step("vmd", 60, 60)
    pump(0.2)
    sep = tmp_path / "prod.sh"
    sep.write_text('cd "$(dirname "$0")"\ntouch prod.done\n')
    script = tmp_path / "main.sh"
    script.write_text(f'cd "$(dirname "$0")"\nif [ -f prod.done ]; then exit 0; fi\n'
                      f'echo "WAITING: production has not finished" >&2\nexit {WAITING_STATUS}\n')
    job = ScriptJob(4, "Production", str(script), node_id=n.id, separate=[str(sep)])
    both = [job, ScriptJob(4, "Production", str(sep), node_id=n.id, separate=[str(sep)], role="run-separate")]
    # OK, with 'Also run the scripts marked run separately' not ticked
    monkeypatch.setattr(Gtk.MessageDialog, "run", lambda self: Gtk.ResponseType.OK)
    win.run_scripts([job], [str(sep)], parent=win, on_state=pl._on_run_state, on_finished=pl._on_run_finished,
                    with_separate=both)
    assert wait_for(lambda: win.messages), "no result"
    assert win.messages[0][1] == "Waiting for separately run scripts" and str(sep) in win.messages[0][2]
    assert pl.run_state == {n.id: "waiting"} and not (tmp_path / "prod.done").exists()
    text = win.log.text()
    assert "[04] WAITING: production has not finished" in text
    status = "[04] ==== step 04: Production - waiting for a separately run script: prod.sh"
    assert status in text and "Run finished: 0 script(s) ok, 1 waiting" in text
    assert tag_names(win, "WAITING: production") == {"waiting"}
    assert "waiting" in tag_names(win, "==== step 04: Production - waiting")
    win.log.set_filter(4)
    assert win.log.text().splitlines() == [l for l in text.splitlines() if l.startswith("[04]")
                                           or l.startswith("==== step 04")]


# ---------------------------------------------------------------------------
# the analysis window: Open in RNAdo, the plan's files
# ---------------------------------------------------------------------------


def test_open_in_rnado_from_the_analysis_window(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import app as analysis
    from mdatk.gui import launch

    path = project_file(tmp_path)
    win = analysis_window(app, windows, path)
    for gone in ("planner", "wizard", "pdb_tools", "builder", "open_planner", "open_wizard", "open_pdb_tools",
                 "open_builder", "run_scripts"):
        assert not hasattr(win, gone), gone
    labels = [i.get_label() for i in win.workflow_button.get_popup().get_children()
              if not isinstance(i, Gtk.SeparatorMenuItem)]
    assert labels == ["Open in RNAdo…", "Script wizard: open in RNAdo…", "PDB clean-up: open in RNAdo…",
                      "Structure builder: open in RNAdo…", "Use the plan's files as inputs"]
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append(cmd)

    monkeypatch.setattr(launch.subprocess, "Popen", FakePopen)
    monkeypatch.delenv("APPIMAGE", raising=False)
    win.lookup_action("rnado").activate(None)  # Ctrl+P
    assert calls == [[sys.executable, "-m", "mdatk.gui.rnado", path]]
    win.lookup_action("wizard").activate(None)  # Ctrl+Shift+W
    for item in win.workflow_button.get_popup().get_children()[2:4]:
        item.activate()
    assert calls[1:] == [[sys.executable, "-m", "mdatk.gui.rnado", path, "--open", tool]
                         for tool in ("wizard", "pdb", "builder")]
    # unsaved changes: asked first; saved, then started
    asked = []
    monkeypatch.setattr(analysis, "ask",
                        lambda _w, text, *_a, **_k: asked.append(text) or Gtk.ResponseType.ACCEPT)
    bound(win, "cleaning.solvent_mask").set_text(":WAT,HOH")
    assert win.open_in_rnado("pdb")
    assert asked == ["Save the changes first?"] and Project.load(path).cleaning.solvent_mask == ":WAT,HOH"
    assert calls[-1][-3:] == [path, "--open", "pdb"]
    # cancelled: nothing starts
    monkeypatch.setattr(analysis, "ask", lambda *_a, **_k: Gtk.ResponseType.CANCEL)
    bound(win, "cleaning.solvent_mask").set_text(":WAT")
    assert not win.open_in_rnado() and len(calls) == 5
    # a project without a file: RNAdo can start without it
    monkeypatch.setattr(analysis, "ask", lambda *_a, **_k: Gtk.ResponseType.REJECT)
    win.new_project()
    assert win.open_in_rnado() and calls[-1] == [sys.executable, "-m", "mdatk.gui.rnado"]


def test_use_the_plans_files_as_inputs(app, windows, tmp_path):
    path = project_file(tmp_path, template="amber")
    win = analysis_window(app, windows, path)
    assert win.use_plan_inputs()
    p = win.collect(validate=False)
    assert p.inputs.topology.endswith(".parm7") and p.inputs.trajectories and p.cleaning.enabled
    assert "Analysis inputs from the plan" in win.log_buffer.get_text(win.log_buffer.get_start_iter(),
                                                                       win.log_buffer.get_end_iter(), False)
    # RNAdo removes the analysis step and saves: the analysis window resolves the plan it finds
    rn = rnado_window(app, windows, path)
    box = next(n for n in rn.project.workflow.nodes if n.tool == "mdatk")
    remove_node(rn.project.workflow, box.id)
    rn.planner.changed()
    assert rn.save_project()
    assert not win.use_plan_inputs()
    assert win.messages[-1][:2] == ("error", "The plan has no MDATK analysis step")


# ---------------------------------------------------------------------------
# both programs on one project file
# ---------------------------------------------------------------------------


def test_both_programs_keep_each_others_edits_in_either_order(app, windows, tmp_path):
    path = project_file(tmp_path)
    win = analysis_window(app, windows, path)
    rn = rnado_window(app, windows, path)
    # 1. the analysis window has unsaved edits while RNAdo saves its plan; then the analysis window saves
    bound(win, "cleaning.solvent_mask").set_text(":WAT,HOH")
    note = rn.planner.add_step("note", 40, 40)
    note.settings["text"] = "first"
    assert rn.save_project()
    assert win.save_project()
    p = Project.load(path)
    assert p.cleaning.solvent_mask == ":WAT,HOH"
    assert [n.settings.get("text") for n in p.workflow.nodes] == ["first"]
    assert [n.id for n in win.project.workflow.nodes] == [note.id]  # the analysis window took RNAdo's plan
    # 2. RNAdo has unsaved plan edits while the analysis window saves; then RNAdo saves
    vmd = rn.planner.add_step("vmd", 300, 40)
    bound(win, "cleaning.ion_mask").set_text(":Na+")
    assert win.save_project()
    p = Project.load(path)
    assert p.cleaning.ion_mask == ":Na+" and [n.tool for n in p.workflow.nodes] == ["note"]
    rn.check_disk()  # RNAdo notices the file changed: takes the analysis settings, keeps its edits
    assert rn.project.cleaning.ion_mask == ":Na+" and not rn.change_bar.get_visible() and rn.plan_changed()
    assert rn.save_project()
    p = Project.load(path)
    assert (p.cleaning.solvent_mask, p.cleaning.ion_mask) == (":WAT,HOH", ":Na+")
    assert [n.id for n in p.workflow.nodes] == [note.id, vmd.id]
    # the analysis window notices RNAdo's save without saving itself
    win.check_disk()
    assert [n.id for n in win.project.workflow.nodes] == [note.id, vmd.id]
    assert not win.analysis_changed()
    # RNAdo without edits reloads the plan another program saved; the step log grows without a question
    from mdatk.projectfile import LogEntry, append_log, edit_workflow
    edit_workflow(path, lambda wf: add_node(wf, "note", 40, 400, title="from the command line"))
    rn.check_disk()
    assert [n.title for n in rn.project.workflow.nodes][-1] == "from the command line"
    assert not rn.plan_changed()
    append_log(path, LogEntry(step="01", title="x"))
    rn.check_disk()
    win.check_disk()
    assert not rn.change_bar.get_visible() and not win.change_bar.get_visible()


def test_rnado_offers_to_reload_a_plan_both_changed(app, windows, tmp_path):
    from mdatk.gui import rnado
    from mdatk.projectfile import edit_workflow

    path = project_file(tmp_path)
    rn = rnado_window(app, windows, path)
    mine = rn.planner.add_step("note", 40, 40)
    edit_workflow(path, lambda wf: add_node(wf, "vmd", 40, 400))  # e.g. 'rnado template' in a terminal
    rn.check_disk()
    assert rn.change_bar.get_visible() and rn.plan_changed()
    assert [n.id for n in rn.project.workflow.nodes] == [mine.id]  # nothing lost before the user chooses
    rn.check_disk()  # the next focus: no new question for the same file
    rn._on_change_bar(rnado.KEEP)  # keep mine: saving replaces theirs without asking
    assert not rn.change_bar.get_visible() and rn.save_project()
    assert [n.tool for n in Project.load(path).workflow.nodes] == ["note"]
    # reload theirs
    rn.planner.add_step("note", 40, 300)
    edit_workflow(path, lambda wf: add_node(wf, "vmd", 40, 400))
    rn.check_disk()
    assert rn.change_bar.get_visible()
    rn._on_change_bar(rnado.RELOAD)
    assert [n.tool for n in rn.project.workflow.nodes] == ["note", "vmd"] and not rn.plan_changed()


def test_group_renames_reach_the_plan_rnado_saved(app, windows, tmp_path):
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    add_node(p.workflow, "note", 40, 40, settings={"text": "select group:Aptamer"})
    path = str(tmp_path / "groups.mdatk.toml")
    p.save(path)
    win = analysis_window(app, windows, path)
    rn = rnado_window(app, windows, path)
    added = rn.planner.add_step("vmd", 300, 40)
    assert rn.save_project()
    row = [r[0] for r in win.group_editor.store].index("Aptamer")
    assert win.commit_group_name(row, "Apt")  # renames the reference in the analysis window's plan copy
    assert win.save_project()
    saved = Project.load(path)
    assert saved.group_names()[:1] == ["Apt"]
    assert [n.id for n in saved.workflow.nodes][-1] == added.id  # RNAdo's step kept ...
    assert saved.workflow.nodes[0].settings["text"] == "select group:Apt"  # ... and the rename followed
    rn.check_disk()
    assert rn.project.group_names()[:1] == ["Apt"] and rn.project.workflow.nodes[0].settings["text"] == \
        "select group:Apt"


def test_the_analysis_window_offers_to_reload_its_settings(app, windows, tmp_path):
    path = project_file(tmp_path)
    win = analysis_window(app, windows, path)

    def other_program(mask):  # e.g. a second analysis window saved the file
        p = Project.load(path)
        p.cleaning.extra_strip_mask = mask
        p.save(path)

    other_program(":K+")
    win.check_disk()  # no unsaved edits here: reloaded at once
    assert bound(win, "cleaning.extra_strip_mask").get_text() == ":K+" and not win.change_bar.get_visible()
    bound(win, "cleaning.solvent_mask").set_text(":SOL")
    other_program(":Cl-")
    win.check_disk()  # unsaved edits: offered, nothing changed yet
    assert win.change_bar.get_visible() and bound(win, "cleaning.extra_strip_mask").get_text() == ":K+"
    win._on_change_bar(1)  # reload them
    assert bound(win, "cleaning.extra_strip_mask").get_text() == ":Cl-"
    assert bound(win, "cleaning.solvent_mask").get_text() != ":SOL" and not win.analysis_changed()


def test_use_for_analysis_in_rnado_saves_the_plan_for_the_analysis_window(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import rnado

    path = project_file(tmp_path, template="amber")
    rn = rnado_window(app, windows, path)
    box = next(n for n in rn.project.workflow.nodes if n.tool == "mdatk")
    box.title = "Analysis of the duplex"
    monkeypatch.setattr(rnado, "ask", lambda *_a, **_k: Gtk.ResponseType.ACCEPT)
    rn.planner.use_for_analysis(box)
    assert not rn.plan_changed()
    assert [n.title for n in Project.load(path).workflow.nodes if n.tool == "mdatk"] == ["Analysis of the duplex"]
    kind, title, detail = rn.messages[-1]
    assert title == "Files for the MDATK analysis" and ".parm7" in detail and "Use the plan's files" in detail


def _groups_project(tmp_path):
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    add_node(p.workflow, "note", 40, 40, settings={"text": "select group:Aptamer"})
    path = str(tmp_path / "groups.mdatk.toml")
    p.save(path)
    return path


def _other_program(path, **cleaning):
    """Another program (a second analysis window, `mdatk density fit --save`) saves analysis settings."""
    p = Project.load(path)
    for key, value in cleaning.items():
        setattr(p.cleaning, key, value)
    p.save(path)


def test_reloading_the_analysis_settings_drops_the_windows_group_renames(app, windows, tmp_path):
    path = _groups_project(tmp_path)
    win = analysis_window(app, windows, path)
    row = [r[0] for r in win.group_editor.store].index("Aptamer")
    assert win.commit_group_name(row, "Apt")  # not saved
    _other_program(path, extra_strip_mask=":K+")
    win.check_disk()
    assert win.change_bar.get_visible()
    win._on_change_bar(1)  # "Reload them (your unsaved changes are lost)": the rename is one of them
    assert win.project.group_names()[:1] == ["Aptamer"] and not win._group_renames
    assert win.project.workflow.nodes[0].settings["text"] == "select group:Aptamer"
    # RNAdo's next plan is taken as it is (no rename replayed on it), and saving keeps it
    rn = rnado_window(app, windows, path)
    rn.project.workflow.nodes[0].settings["text"] = "select group:Aptamer and name P"
    assert rn.plan_changed() and rn.save_project()
    win.check_disk()
    assert win.save_project()
    saved = Project.load(path)
    assert saved.group_names()[:1] == ["Aptamer"] and saved.cleaning.extra_strip_mask == ":K+"
    assert saved.workflow.nodes[0].settings["text"] == "select group:Aptamer and name P"


def test_a_plan_rnado_saves_while_the_analysis_window_asks_is_kept(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import app as analysis

    path = project_file(tmp_path)
    win = analysis_window(app, windows, path)
    rn = rnado_window(app, windows, path)
    steps = []

    def ask(_parent, text, *_a, **_k):  # "Replace them?" - meanwhile RNAdo (another process) saves a step
        assert text.startswith("Another program changed the analysis settings")
        steps.append(rn.planner.add_step("note", 40, 40 + 200 * len(steps)))
        assert rn.save_project()
        pump(1.0 if len(steps) == 1 else 0)  # the first time the file monitors fire during the question
        return Gtk.ResponseType.ACCEPT

    monkeypatch.setattr(analysis, "ask", ask)
    for mask in (":K+", ":Cl-"):
        bound(win, "cleaning.solvent_mask").set_text(f":SOL,{mask}")  # unsaved edit
        _other_program(path, extra_strip_mask=mask)
        assert win.save_project()
        saved = Project.load(path)
        assert (saved.cleaning.solvent_mask, saved.cleaning.extra_strip_mask) == (f":SOL,{mask}", "")
        assert [n.id for n in saved.workflow.nodes] == [s.id for s in steps]  # RNAdo's steps kept
        assert [n.id for n in win.project.workflow.nodes] == [s.id for s in steps]
        rn.check_disk()
        assert [n.id for n in rn.project.workflow.nodes] == [s.id for s in steps] and not rn.plan_changed()
    assert len(steps) == 2


def test_rnado_follows_group_renames_in_its_unsaved_plan(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import rnado
    from mdatk.projectfile import edit_workflow

    path = _groups_project(tmp_path)
    win = analysis_window(app, windows, path)
    rn = rnado_window(app, windows, path)
    added = rn.planner.add_step("vmd", 300, 40)  # unsaved plan edit in RNAdo
    rn.planner.open_wizard(added)
    pump(0.3)
    monkeypatch.setattr(rnado, "ask", lambda *_a, **_k: pytest.fail("asked about a plan changed by renames"))
    row = [r[0] for r in win.group_editor.store].index("Aptamer")
    assert win.commit_group_name(row, "Apt") and win.save_project()
    rn.check_disk()  # the file's plan differs from RNAdo's only by the rename: no conflict
    assert not rn.change_bar.get_visible() and rn.plan_changed() and added.id in rn.planner.wizards
    assert rn.project.group_names()[:1] == ["Apt"]
    assert rn.project.workflow.nodes[0].settings["text"] == "select group:Apt"
    assert rn.save_project()
    saved = Project.load(path)
    assert saved.workflow.nodes[0].settings["text"] == "select group:Apt"
    assert [n.id for n in saved.workflow.nodes][-1] == added.id
    # RNAdo saves before its file monitor noticed the rename: the same
    added = rn.planner.add_step("note", 300, 300)
    row = [r[0] for r in win.group_editor.store].index("Apt")
    assert win.commit_group_name(row, "Aptamer2") and win.save_project()
    assert rn.save_project()
    saved = Project.load(path)
    assert saved.group_names()[:1] == ["Aptamer2"]
    assert saved.workflow.nodes[0].settings["text"] == "select group:Aptamer2"
    assert [n.id for n in saved.workflow.nodes][-1] == added.id and not rn.plan_changed()
    # a plan another program really changed is still offered
    rn.planner.add_step("note", 600, 40)
    edit_workflow(path, lambda wf: add_node(wf, "vmd", 40, 600))
    rn.check_disk()
    assert rn.change_bar.get_visible()


def test_saving_without_edits_takes_the_newer_analysis_settings(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import app as analysis

    path = project_file(tmp_path)
    win = analysis_window(app, windows, path)
    _other_program(path, extra_strip_mask=":K+")
    assert not win.analysis_changed()
    assert win.save_project()  # Ctrl+S before the window noticed: nothing to save, theirs are taken
    assert Project.load(path).cleaning.extra_strip_mask == ":K+"
    assert bound(win, "cleaning.extra_strip_mask").get_text() == ":K+" and not win.analysis_changed()
    _other_program(path, extra_strip_mask=":Cl-")
    copy = str(tmp_path / "copy.mdatk.toml")
    monkeypatch.setattr(analysis, "choose_save_path", lambda *_a, **_k: copy)
    assert win.save_project(True)  # Save as: the copy gets the newest settings too
    assert Project.load(copy).cleaning.extra_strip_mask == ":Cl-" and win.project_path == copy
    assert Project.load(path).cleaning.extra_strip_mask == ":Cl-"


def test_rnado_takes_the_settings_its_save_merged_with(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import rnado
    from mdatk.projectfile import edit_workflow

    path = project_file(tmp_path)
    win = analysis_window(app, windows, path)
    rn = rnado_window(app, windows, path)
    for mask, residues, wait in ((":SOL", "1-5", 0), (":WAT", "1-6", 1.5)):
        rn.planner.add_step("note", 40, 40)  # unsaved RNAdo edit
        edit_workflow(path, lambda wf: add_node(wf, "vmd", 40, 400))  # e.g. 'rnado template -p' in a terminal

        def ask(*_a, **_k):  # "Another program changed the plan": meanwhile the analysis window saves
            bound(win, "cleaning.solvent_mask").set_text(mask)
            win.group_editor.store[0][1] = residues
            assert win.save_project()
            pump(wait)  # the second time RNAdo's file monitor fires during the question
            return rnado.KEEP

        monkeypatch.setattr(rnado, "ask", ask)
        assert rn.save_project()
        saved = Project.load(path)
        assert (saved.cleaning.solvent_mask, saved.groups[0].residues) == (mask, residues)
        assert [n.tool for n in saved.workflow.nodes] == [n.tool for n in rn.project.workflow.nodes]
        assert (rn.project.cleaning.solvent_mask, rn.project.groups[0].residues) == (mask, residues)
        rn.check_disk()
        assert not rn._disk.changed() and not rn.plan_changed() and not rn.change_bar.get_visible()


def test_using_the_plans_files_keeps_a_pending_question(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import app as analysis

    path = project_file(tmp_path, template="amber")
    win = analysis_window(app, windows, path)
    bound(win, "cleaning.solvent_mask").set_text(":SOL")  # unsaved edit
    _other_program(path, extra_strip_mask=":K+")
    assert win.use_plan_inputs()
    assert win.change_bar.get_visible() and bound(win, "cleaning.solvent_mask").get_text() == ":SOL"
    _other_program(path, extra_strip_mask=":Cl-")  # newer settings than the ones the bar offers: asked
    asked = []
    monkeypatch.setattr(analysis, "ask", lambda *a, **_k: asked.append(a[1]) or Gtk.ResponseType.CANCEL)
    assert not win.save_project() and asked and Project.load(path).cleaning.extra_strip_mask == ":Cl-"
    win.check_disk()
    win._on_change_bar(1)  # reload them
    assert bound(win, "cleaning.extra_strip_mask").get_text() == ":Cl-" and not win.change_bar.get_visible()


def test_opening_a_script_wizard_is_no_edit_of_the_plan(app, windows, tmp_path, monkeypatch):
    from mdatk.gui import rnado
    from mdatk.projectfile import edit_workflow

    path = project_file(tmp_path, template="amber")

    def older(wf):  # a project file from an older version: an option of the build step is missing
        for n in wf.nodes:
            if n.tool == "amber-build":
                del n.settings["mg_types"]

    edit_workflow(path, older)
    rn = rnado_window(app, windows, path)
    rn.open_tool("wizard")  # 'rnado --open wizard', the analysis window's "Script wizard: open in RNAdo…"
    pump(0.3)
    assert rn.project.workflow.scratch and not rn.plan_changed()
    rn.wizard.select_tool("gromacs")  # choosing a tool is no edit either
    pump(0.3)
    build = next(n for n in rn.project.workflow.nodes if n.tool == "amber-build")
    rn.planner.open_wizard(build)  # fills in the missing option
    pump(0.3)
    assert "mg_types" in build.settings and not rn.plan_changed()
    asked = []
    monkeypatch.setattr(rnado, "ask_unsaved", lambda *a, **_k: asked.append(a) or "cancel")
    assert not rn._on_delete() and not asked  # closing does not ask to save
    edit_workflow(path, lambda wf: add_node(wf, "vmd", 40, 600))  # another program saves the plan
    rn.check_disk()
    assert not rn.change_bar.get_visible() and rn.project.workflow.nodes[-1].tool == "vmd"
    rn.wizard.title_entry.set_text("My GROMACS run")  # an edit in the wizard still is one
    assert rn.plan_changed() and rn._on_delete() and asked
