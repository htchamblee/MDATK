"""GTK tests of RNAdo's script wizard and planner (run under xvfb-run -a)."""

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
from mdatk.workflows import TOOLS  # noqa: E402


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.005)


class Ev:
    def __init__(self, x, y, button=1, typ=None, state=0):
        self.x, self.y, self.button, self.state = x, y, button, state
        self.x_root, self.y_root = x, y
        self.type = typ or Gdk.EventType.BUTTON_PRESS


@pytest.fixture
def main(tmp_path):
    """RNAdo's window: the host of the planner and the wizard."""
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


def test_standalone_wizard_every_tool(main, tmp_path):
    main.open_wizard("amber-md")
    wz = main.wizard
    pump()
    for name, tool in TOOLS.items():
        if not tool.generates:
            continue
        wz.select_tool(name)
        pump(0.1)
        assert wz.tool.name == name and wz.notebook.get_n_pages() == 1 + len(tool.sections)
    wz.select_tool("amber-md")
    pump()
    # an option change updates the node, the checks and the preview
    engine = wz.forms["engine"]
    _opt, widget, _lab, _get, setter = engine.rows["engine"]
    setter("pmemd.MPI")
    pump(0.6)
    node = wz.node
    assert node.settings["engine"] == "pmemd.MPI"
    nproc_widget = engine.rows["nproc"][1]
    assert nproc_widget.get_sensitive()
    # inputs: choosing files clears the "missing input" errors
    for port, path in (("topology", "/x/sys.parm7"), ("coordinates", "/x/sys.rst7"), ("restraints", "/x/h.dist")):
        wz.input_widgets[port].entry.set_text(path)
    pump(0.6)
    errors = [r.issue for r in wz.issue_list.get_children() if getattr(r, "issue", None) and
              r.issue.level == "error"]
    assert errors == []
    assert wz.write_btn.get_sensitive()
    wz.preview_combo.set_active_id(wz.files[0].name)
    buf = wz.preview.get_buffer()
    assert "&cntrl" in buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    wz.write_files()
    pump()
    folder = os.path.join(main.project.workflow_dir, "01_amber-md")
    assert any(f.endswith(".sh") for f in os.listdir(folder))
    assert main.project.workflow.stamp  # fixed by the first write
    # scratch settings are kept per tool in the project's workflow
    assert any(n.tool == "amber-md" and n.settings["engine"] == "pmemd.MPI"
               for n in main.project.workflow.scratch)


def test_wizard_stage_editor(main):
    main.open_wizard("amber-md")
    wz = main.wizard
    pump()
    editor = wz.forms["stages"]
    n = len(editor.records)
    editor.select(0)
    pump(0.1)
    # minimisation stage: MD-only fields hidden
    assert not editor.form.rows["nstlim"][1].get_visible()
    editor.select(2)
    pump(0.1)
    assert editor.form.rows["nstlim"][1].get_visible()
    # radio rows hidden for the first (minimisation) record must come back with their buttons
    radios = editor.form.rows["ensemble"][1]
    assert radios.get_visible() and all(b.get_visible() for b in radios.get_children())
    editor.add("prod-free")
    editor.duplicate()
    pump(0.6)
    assert len(wz.node.settings["stages"]) == n + 2
    editor.remove()
    editor.move(-1)
    pump(0.6)
    assert len(wz.node.settings["stages"]) == n + 1
    # an error highlights the field and blocks writing
    rec = wz.node.settings["stages"][editor.current]
    editor.form.rows["dt"][4](0.004)
    editor.form.rows["shake"][4](False)
    pump(0.6)
    assert rec["dt"] == 0.004 and not wz.write_btn.get_sensitive()
    issues = [r.issue for r in wz.issue_list.get_children() if getattr(r, "issue", None)]
    assert any("needs SHAKE" in i.message for i in issues)
    wz.reset_defaults()
    pump(0.3)
    assert len(wz.node.settings["stages"]) == n


def test_planner_template_connect_and_write(main, tmp_path):
    pl = main.planner
    pump()
    pl.add_template("amber")
    pump(0.3)
    nodes = {n.tool: n for n in main.project.workflow.nodes}
    assert set(nodes) == {"amber-build", "amber-restraints", "amber-md", "mdatk"}
    assert len(main.project.workflow.edges) == 6
    # drag a connection from the build's PDB output into a new VMD step's topology input
    vmd = pl.add_step("vmd", 900, 420)
    pump(0.2)
    canvas = pl.canvas
    assert canvas.zoom < 1.0  # the template was zoomed to fit

    def at(x, y):  # canvas (model) coordinates -> event coordinates
        return Ev(x * canvas.zoom, y * canvas.zoom)

    start = canvas.port_pos(nodes["amber-build"], "pdb", True)
    end = canvas.port_pos(vmd, "topology", False)
    canvas._on_press(canvas, at(*start))
    canvas._on_motion(canvas, at(*end))
    canvas._on_release(canvas, at(*end))
    pump(0.2)
    assert any(e.source == nodes["amber-build"].id and e.target == vmd.id and e.input == "topology"
               for e in main.project.workflow.edges)
    # an incompatible connection is refused
    edges = len(main.project.workflow.edges)
    start = canvas.port_pos(nodes["amber-build"], "coordinates", True)
    canvas._on_press(canvas, at(*start))
    canvas._on_release(canvas, at(*end))
    assert len(main.project.workflow.edges) == edges
    # moving a box (positions snap to a 10 px grid)
    node = nodes["amber-md"]
    x0 = node.x
    p = (node.x + 60, node.y + 10)  # the title bar, right of the "run" check box
    canvas._on_press(canvas, at(*p))
    canvas._on_motion(canvas, at(p[0] + 100, p[1] + 50))
    canvas._on_release(canvas, at(p[0] + 100, p[1] + 50))
    assert node.x == pytest.approx(x0 + 100, abs=10)
    # fill in the missing bits through the node data and write everything
    nodes["amber-build"].files["structure"] = str(tmp_path / "model.pdb")
    nodes["amber-restraints"].settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    pl.changed()
    pump(0.2)
    assert not [i for _n, i in pl.plan.all_issues() if i.level == "error"]
    pl.write(None)
    pump(0.2)
    root = main.project.workflow_dir
    assert os.path.isdir(os.path.join(root, "03_amber-md"))
    assert any(f.endswith("_plan_" + main.project.workflow.stamp + ".sh") for f in os.listdir(root))
    # open a step in the wizard; edits there redraw the planner
    pl.open_wizard(nodes["amber-md"])
    pump(0.5)
    wz = pl.wizards[nodes["amber-md"].id]
    assert "connected" in wz.input_widgets["topology"].get_text()
    wz.title_entry.set_text("Production MD")
    pump(0.6)
    assert nodes["amber-md"].title == "Production MD"
    # the analysis step hands its files to the analysis window (RNAdo saves the plan for it)
    handed = []
    main.use_for_analysis = lambda top, trajs, clean=True: handed.append((top, trajs, clean))
    pl.use_for_analysis(nodes["mdatk"])
    pump(0.2)
    (top, trajs, clean), = handed
    assert top.endswith(".parm7") and trajs and clean
    assert main.planner is pl and main.project.workflow.nodes


def test_planner_delete_and_project_roundtrip(main, tmp_path, monkeypatch):
    from mdatk.gui import rnado

    pl = main.planner
    pl.add_template("amber-gromacs")
    pump(0.2)
    g = [n for n in main.project.workflow.nodes if n.tool == "gromacs"][0]
    pl.canvas.select(g.id)
    pl.delete_selected()
    assert all(n.tool != "gromacs" for n in main.project.workflow.nodes)
    assert all(g.id not in (e.source, e.target) for e in main.project.workflow.edges)
    note = pl.add_step("note", 40, 600)
    pl.show_details(note.id)
    pl.d_note.get_buffer().set_text("check the Mg count")
    pump(0.6)
    assert note.settings["text"] == "check the Mg count"
    path = str(tmp_path / "plan.mdatk.toml")
    assert main.plan_changed()
    monkeypatch.setattr(rnado, "choose_save_path", lambda *_a, **_k: path)
    assert main.save_project() and main.project_path == path and not main.plan_changed()
    loaded = Project.load(path)
    assert [n.tool for n in loaded.workflow.nodes] == [n.tool for n in main.project.workflow.nodes]
    main.open_path(path)
    pump(0.3)
    assert main.planner is pl and [n.id for n in pl.workflow.nodes] == [n.id for n in loaded.workflow.nodes]


def test_selecting_a_step_keeps_the_scroll_position(main):
    pl = main.planner
    main.resize(1100, 900)
    nodes = [pl.add_step("note", 60 + 500 * (k % 3), 60 + 450 * k) for k in range(6)]
    pump(0.5)
    h, v = pl.scroller.get_hadjustment(), pl.scroller.get_vadjustment()
    assert v.get_upper() > v.get_page_size() + 1500
    for node in (nodes[4], nodes[5], nodes[2]):  # first click focuses the canvas, later ones keep it
        h.set_value(min(node.x * pl.canvas.zoom - 40, h.get_upper() - h.get_page_size()))
        v.set_value(node.y * pl.canvas.zoom - 40)
        pump(0.2)
        before = (h.get_value(), v.get_value())
        x, y = (node.x + 30) * pl.canvas.zoom, (node.y + 15) * pl.canvas.zoom
        pl.canvas._on_press(pl.canvas, Ev(x, y))
        pl.canvas._on_release(pl.canvas, Ev(x, y))
        pump(0.3)
        assert pl.canvas.selected == node.id
        assert (h.get_value(), v.get_value()) == before
    assert pl.canvas.has_focus()  # the Delete key still reaches the canvas


def test_planner_run_selected_steps(main, tmp_path):
    pl = main.planner
    pl.add_template("amber")
    note = pl.add_step("note", 40, 700)
    pump(0.3)
    nodes = {n.tool: n for n in main.project.workflow.nodes}
    nodes["amber-build"].files["structure"] = str(tmp_path / "model.pdb")
    nodes["amber-restraints"].settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    pl.changed()
    pump(0.2)
    canvas, btn = pl.canvas, pl.run_selected_btn
    assert not btn.get_sensitive() and btn.get_label() == "Run selected steps"
    runs = []
    main.run_scripts = lambda jobs, separate=(), parent=None, **_kw: runs.append((jobs, list(separate)))

    def click_box(node):
        bx, by, bw, bh = canvas.check_rect(node)
        x, y = (bx + bw / 2) * canvas.zoom, (by + bh / 2) * canvas.zoom
        canvas._on_press(canvas, Ev(x, y))
        canvas._on_release(canvas, Ev(x, y))
        pump(0.1)

    # tick the MD step first, then the build: the run follows the plan order, not the click order
    click_box(nodes["amber-md"])
    click_box(nodes["amber-build"])
    assert pl.checked == {nodes["amber-md"].id, nodes["amber-build"].id}
    assert btn.get_sensitive() and btn.get_label() == "Run selected steps (2)"
    assert canvas.check_at(note.x + 14, note.y + 13) is None  # notes have no box
    assert canvas.selected != nodes["amber-build"].id  # ticking does not select or move the box
    click_box(nodes["amber-md"])  # untick
    click_box(nodes["amber-md"])  # tick again
    pl.run_selected()
    pump(0.2)
    assert len(runs) == 1
    labels = [j.title for j in runs[0][0]]
    assert len(labels) == 2 and labels == [pl.plan.steps[nodes["amber-build"].id].label,
                                           pl.plan.steps[nodes["amber-md"].id].label]
    root = main.project.workflow_dir
    assert os.path.isdir(os.path.join(root, "01_amber-build"))
    assert not os.path.isdir(os.path.join(root, "02_amber-restraints"))  # unticked: not written
    # Space toggles the selected step; deleting a ticked step drops it
    canvas.select(nodes["amber-restraints"].id)
    canvas._on_key(canvas, type("K", (), {"keyval": Gdk.KEY_space})())
    assert nodes["amber-restraints"].id in pl.checked
    canvas.select(nodes["amber-md"].id)
    pl.delete_selected()
    pump(0.1)
    assert nodes["amber-md"].id not in pl.checked and btn.get_label() == "Run selected steps (2)"
    pl.set_checked([])
    assert not btn.get_sensitive()


def test_planner_running_marks_and_parallel_setting(main, tmp_path):
    pl = main.planner
    pl.add_template("amber")
    pump(0.3)
    nodes = {n.tool: n for n in main.project.workflow.nodes}
    nodes["amber-build"].files["structure"] = str(tmp_path / "model.pdb")
    nodes["amber-restraints"].settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    pl.changed()
    pl.parallel_spin.set_value(3)
    assert main.project.workflow.parallel == 3
    calls = []
    main.run_scripts = lambda jobs, separate=(), parent=None, **kw: calls.append((jobs, kw))
    pl.set_checked([nodes["amber-build"].id, nodes["amber-restraints"].id])
    pl.run_selected()
    pump(0.2)
    jobs, kw = calls[0]
    assert kw["parallel"] == 3 and callable(kw["on_state"]) and callable(kw["on_finished"])
    assert [j.node_id for j in jobs] == [nodes["amber-build"].id, nodes["amber-restraints"].id]
    assert jobs[1].after == [nodes["amber-build"].id]
    # the run reports states on the main loop: queued -> running (spinner) -> ok / failed
    on_state = kw["on_state"]
    for j in jobs:
        on_state(j, "queued")
    assert set(pl.run_state.values()) == {"queued"}
    on_state(jobs[0], "running")
    assert pl._spin_id
    phase = pl.canvas.spin_phase
    pump(0.3)
    assert pl.canvas.spin_phase != phase  # the arc turns
    on_state(jobs[0], "ok")
    on_state(jobs[1], "running")
    on_state(jobs[1], "failed")
    pump(0.2)
    assert pl.run_state == {jobs[0].node_id: "ok", jobs[1].node_id: "failed"} and not pl._spin_id
    # the next run clears the old marks; steps never started end as cancelled
    on_state(jobs[0], "queued")
    on_state(jobs[1], "queued")
    on_state(jobs[0], "running")
    kw["on_finished"]()
    assert pl.run_state == {jobs[0].node_id: "cancelled", jobs[1].node_id: "cancelled"}
    pl.canvas.queue_draw()
    pump(0.2)  # every marker draws
    # saved with the project
    path = str(tmp_path / "p.mdatk.toml")
    main.project.save(path)
    assert Project.load(path).workflow.parallel == 3


def test_run_scripts_end_to_end_with_running_marks(main, tmp_path, monkeypatch):
    import shutil
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    from mdatk.workflows.runner import ScriptJob
    pl = main.planner
    a, b, c = (pl.add_step("vmd", 60 + 300 * k, 60) for k in range(3))
    pump(0.2)
    jobs = []
    for k, (n, after) in enumerate(((a, []), (b, []), (c, [a.id, b.id]))):
        path = tmp_path / f"s{k}.sh"
        path.write_text(f"sleep 0.5\necho done {k}\n")
        jobs.append(ScriptJob(k + 1, n.tool, str(path), node_id=n.id, after=after))
    monkeypatch.setattr(Gtk.MessageDialog, "run", lambda self: Gtk.ResponseType.OK)
    seen = []
    orig = pl._on_run_state
    pl._on_run_state = lambda j, st: (seen.append((j.node_id, st)), orig(j, st))
    main.run_scripts(jobs, parent=main, parallel=2, on_state=pl._on_run_state,
                     on_finished=pl._on_run_finished)
    end = time.time() + 15
    while time.time() < end and not any(m[1] == "Scripts finished" for m in main.messages):
        pump(0.1)
    assert any(m[1] == "Scripts finished" for m in main.messages), main.messages
    assert pl.run_state == {a.id: "ok", b.id: "ok", c.id: "ok"}
    running = [n for n, st in seen if st == "running"]
    assert running.index(c.id) == 2 and set(running[:2]) == {a.id, b.id}
    log = main.log.text()  # RNAdo's output log panel
    assert "[03] done 2" in log and "==== step 03: vmd - finished (s2.sh)" in log


def test_planner_ignores_callbacks_of_an_older_run(main, tmp_path):
    from mdatk.workflows.runner import ScriptJob
    pl = main.planner
    n = pl.add_step("vmd", 60, 60)
    pump(0.2)
    calls = []
    main.run_scripts = lambda jobs, separate=(), parent=None, **kw: calls.append(kw)
    pl._ask = lambda *_a: True  # the empty VMD step has errors: "write the others anyway?"
    pl.write([n.id], run=True)
    kw = calls[-1]
    j = ScriptJob(1, "vmd", "/x.sh", node_id=n.id)
    kw["on_state"](j, "queued")
    kw["on_state"](j, "running")
    assert pl.run_state == {n.id: "running"} and pl._spin_id
    pl.reload()  # e.g. another project was opened: the old run's marks must not land on the new boxes
    kw["on_state"](j, "ok")
    kw["on_finished"]()
    assert pl.run_state == {}
    pl.parallel_spin.set_value(100)
    assert pl.parallel_spin.get_value() == 64 and main.project.workflow.parallel == 64
    pl.shutdown()  # the window closes
    pump(0.3)
    assert not pl._spin_id
    kw["on_state"](j, "running")  # a run still going after the window closed
    assert not pl._spin_id


def test_planner_check_previous_runs_adopt_and_run_selected(main, tmp_path):
    from mdatk.gui.planner import FOOT_H
    from mdatk.workflows import runstate
    old = "20261015-1432"
    pl = main.planner
    pl.add_template("amber")
    pump(0.3)
    nodes = {n.tool: n for n in main.project.workflow.nodes}
    b, md = nodes["amber-build"], nodes["amber-md"]
    b.files["structure"] = str(tmp_path / "model.pdb")
    nodes["amber-restraints"].settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    main.project.workflow.stamp = old
    pl.changed()
    for path in runstate.expected(pl.plan, b.id, old):  # the build has run
        path = pl.plan.node_path(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("x\n")
    height = pl.canvas.node_size(b)[1]
    asked = []
    pl._ask = lambda text, _detail: asked.append(text) or False  # offered, not taken
    runs = pl.check_runs(None)
    assert runs[b.id].state == "done" and runs[md.id].state == "missing" and nodes["mdatk"].id not in runs
    assert asked == ["Use the outputs of 1 finished run(s)?"] and b.adopted == ""
    assert pl.canvas.node_size(b)[1] == height + FOOT_H  # the footer shows what was found
    assert pl.canvas.run_info(b)[0].startswith(f"done · {old} · 20")
    assert pl.canvas.run_info(md)[0] == "missing · no outputs yet"
    # the next iteration: a new stamp; the earlier run is found and used from the details pane
    pl.new_stamp()
    assert main.project.workflow.stamp != old
    pl.check_runs([b.id])
    pl.canvas.select(b.id)
    assert f"Previous run: done: run {old} (an earlier stamp)" in pl.d_prev.get_text()
    assert pl.d_adopt.get_sensitive() and not pl.d_forget.get_sensitive()
    pl.d_adopt.clicked()
    assert b.adopted == old and pl.d_forget.get_sensitive() and not pl.d_adopt.get_sensitive()
    assert "Uses the outputs of its run of" in pl.d_prev.get_text()
    assert pl.plan.steps[md.id].ctx.inp("topology").endswith(f"_parmed_{old}.parm7")
    assert pl.canvas.run_info(b)[0].startswith("uses done")
    pl.canvas.queue_draw()
    pump(0.2)  # the footers draw
    # Run selected with a step that uses an earlier run: skip it, cancel, or run it again
    calls = []
    main.run_scripts = lambda jobs, separate=(), parent=None, **kw: calls.append((jobs, list(separate), kw))
    pl.set_checked([b.id, md.id])
    pl._adopted_choice = lambda _text, _detail: "skip"
    pl.run_selected()
    jobs, separate, kw = calls[-1]
    assert [j.node_id for j in jobs] == [md.id] and len(separate) == 1
    assert [(j.node_id, j.role) for j in kw["with_separate"]] == [(md.id, "run"), (md.id, "run-separate")]
    assert not os.path.exists(os.path.join(main.project.workflow_dir, "01_amber-build",
                                           f"apt1_01_amber-build_{main.project.workflow.stamp}.sh"))
    pl._adopted_choice = lambda _text, _detail: "cancel"
    pl.run_selected()
    assert len(calls) == 1
    pl._adopted_choice = lambda _text, _detail: "rerun"
    pl.run_selected()
    assert b.adopted == "" and [j.node_id for j in calls[-1][0]] == [b.id, md.id]
    # the run marks a step that waits for its separately run script
    job = calls[-1][0][1]
    calls[-1][2]["on_state"](job, "waiting")
    assert pl.run_state[md.id] == "waiting"
    pl.canvas.queue_draw()
    pump(0.2)


def test_run_scripts_waiting_and_the_separate_scripts_option(main, tmp_path, monkeypatch):
    import shutil
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    from mdatk.workflows.base import WAITING_STATUS
    from mdatk.workflows.runner import ScriptJob
    pl = main.planner
    n = pl.add_step("vmd", 60, 60)
    pump(0.2)
    sep = tmp_path / "prod.sh"
    sep.write_text('cd "$(dirname "$0")"\ntouch prod.done\n')
    script = tmp_path / "main.sh"
    script.write_text(f'cd "$(dirname "$0")"\nif [ -f prod.done ]; then exit 0; fi\nexit {WAITING_STATUS}\n')
    job = ScriptJob(1, "md", str(script), node_id=n.id, separate=[str(sep)])
    both = [job, ScriptJob(1, "md", str(sep), node_id=n.id, separate=[str(sep)], role="run-separate")]

    def run(ticked):
        def dialog_run(dlg):
            boxes = [w for w in dlg.get_message_area().get_children() if isinstance(w, Gtk.CheckButton)]
            assert len(boxes) == 1
            boxes[0].set_active(ticked)
            return Gtk.ResponseType.OK

        monkeypatch.setattr(Gtk.MessageDialog, "run", dialog_run)
        main.messages.clear()
        main.run_scripts([job], [str(sep)], parent=main, on_state=pl._on_run_state,
                         on_finished=pl._on_run_finished, with_separate=both)
        end = time.time() + 15
        while time.time() < end and not main.messages:
            pump(0.1)
        return main.messages

    msgs = run(False)
    assert msgs and msgs[0][1] == "Waiting for separately run scripts" and str(sep) in msgs[0][2]
    assert pl.run_state == {n.id: "waiting"} and not (tmp_path / "prod.done").exists()
    pl.canvas.queue_draw()
    pump(0.2)
    msgs = run(True)  # "Also run the scripts marked 'run separately'"
    assert msgs and msgs[0][1] == "Scripts finished", msgs
    assert (tmp_path / "prod.done").exists() and pl.run_state == {n.id: "ok"}
