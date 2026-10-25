"""GTK tests of the script wizard and the Project Planner (run under xvfb-run -a)."""

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
    from mdatk.gui.app import MainWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    p.workflow.project_name = "apt1"
    win = MainWindow(app, p, None)
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    win.show_all()
    pump(0.5)
    yield win
    if win.planner is not None:
        win.planner.destroy()
    if win.wizard is not None:
        win.wizard.destroy()
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
    main.open_planner()
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
    # use the plan for analysis
    pl.use_for_analysis(nodes["mdatk"])
    pump(0.2)
    assert main.project.inputs.topology.endswith(".parm7") and main.project.inputs.trajectories
    assert main.project.cleaning.enabled
    assert main.planner is pl and main.project.workflow.nodes  # same workflow, windows kept


def test_planner_delete_and_project_roundtrip(main, tmp_path):
    main.open_planner()
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
    main.project_path = path
    project = main.collect(validate=False)
    project.save(path)
    loaded = Project.load(path)
    assert [n.tool for n in loaded.workflow.nodes] == [n.tool for n in main.project.workflow.nodes]
    main.open_path(path)
    pump(0.3)
    assert main.planner is pl and [n.id for n in pl.workflow.nodes] == [n.id for n in loaded.workflow.nodes]


def test_selecting_a_step_keeps_the_scroll_position(main):
    main.open_planner()
    pl = main.planner
    pl.resize(1100, 600)
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
    main.open_planner()
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
    main.run_scripts = lambda jobs, separate=(), parent=None: runs.append((jobs, list(separate)))

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
