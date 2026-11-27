"""Regression tests for the GUI items of the external bug report on MDATK 0.1.1 (MDATK-BUG-001 to 010,
068 and 069): a project that is opened and saved in the analysis window without edits comes back as it
was, a value a widget cannot show is kept (and flagged where it is invalid), unsaved edits and running
jobs are asked about, and the user settings file is written in one step.

The window tests need GTK 3 and a display (``xvfb-run -a pytest``); the settings-file tests do not."""

import builtins
import os
import random
import threading
import time

import pytest

from mdatk import projectfile
from mdatk.config import ExternalSeries, LayoutParams, PanelSpec, PlanNode, Project, RepSpec

try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, Gtk

    HAVE_GTK = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and Gtk.init_check([])[0]
except (ImportError, ValueError):  # pragma: no cover
    HAVE_GTK = False


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        Gtk.main_iteration_do(False)
        time.sleep(0.005)


class Asks:
    """Stands in for the Save / Don't save / Cancel question: records it and gives the queued answer."""

    def __init__(self):
        self.asked, self.answers = [], []

    def __call__(self, _parent, what, _detail=""):
        self.asked.append(what)
        assert self.answers, f"unexpected question about unsaved changes to {what}"
        return self.answers.pop(0)


@pytest.fixture
def asks(monkeypatch):
    stub = Asks()
    monkeypatch.setattr("mdatk.gui.app.ask_unsaved", stub, raising=False)
    return stub


@pytest.fixture
def make(monkeypatch, asks):
    """make(project=None, path=None) -> an analysis window (not shown); errors and messages are recorded."""
    if not HAVE_GTK:
        pytest.skip("needs GTK 3 and a display (run under xvfb-run)")
    from mdatk.cleaning import conda
    from mdatk.gui.app import MainWindow

    monkeypatch.setattr(conda, "detect_ambertools_env", lambda *a, **k: None)  # nothing is pre-filled
    windows = []

    def make_window(project=None, path=None):
        app = Gtk.Application(application_id=None)
        app.register(None)
        win = MainWindow(app, project, path)
        win.messages = []
        win.info = lambda t, d="": win.messages.append(("info", t, d))
        win.error = lambda t, d="": win.messages.append(("error", t, d))
        windows.append(win)
        return win

    yield make_window
    for win in windows:
        win.destroy()


def field(win, path):
    """The widget bound to the project setting ``path`` (the entry of a file chooser box)."""
    widget = next(b.widget for b in win.binder.bindings if b.path == path)
    return widget.entry if isinstance(widget, Gtk.Box) and not isinstance(widget, Gtk.ComboBox) else widget


def get_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj, path, value):
    parts = path.split(".")
    setattr(get_path(obj, ".".join(parts[:-1])) if parts[:-1] else obj, parts[-1], value)


def collected(win, path):
    return get_path(win.collect(validate=False), path)


def log_text(win):
    return win.log_buffer.get_text(*win.log_buffer.get_bounds(), False)


# ---------------------------------------------------------------------------
# MDATK-BUG-069: the user settings file
# ---------------------------------------------------------------------------


def settings_file():
    target = projectfile._settings_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def test_bug069_an_unreadable_settings_file_is_not_written_over(tmp_path):
    """A settings file that is cut off (no TOML) was replaced by one holding the recent list alone, and
    every other setting in it was gone for good.  It is kept under another name now."""
    target = settings_file()
    project = tmp_path / "p.mdatk.toml"
    project.write_text("")
    cut_off = 'cat_proof = true\nrecent = ["/a/b.mdatk.toml", "/c/d'
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(cut_off)
    assert projectfile.recent_projects() == [] and projectfile.user_setting("cat_proof") is None
    projectfile.add_recent(str(project))
    assert projectfile.recent_projects() == [str(project)]
    with open(target + ".unreadable", encoding="utf-8") as fh:
        assert fh.read() == cut_off  # nothing of it is lost
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(cut_off)
    projectfile.set_user_setting("cat_proof", False)
    assert projectfile.user_setting("cat_proof") is False
    with open(target + ".unreadable", encoding="utf-8") as fh:
        assert fh.read() == cut_off


def test_bug069_a_settings_file_cut_inside_a_character_is_no_error(tmp_path):
    """Cut off inside a multi-byte character the file is not even text: reading the recent projects
    raised UnicodeDecodeError (the open drop-down is filled from them when a window starts)."""
    target = settings_file()
    project = tmp_path / "p.mdatk.toml"
    project.write_text("")
    with open(target, "wb") as fh:
        fh.write('recent = ["/a/é'.encode()[:-1])
    assert projectfile.recent_projects() == []
    assert projectfile.user_setting("cat_proof", "default") == "default"
    projectfile.add_recent(str(project))
    assert projectfile.recent_projects() == [str(project)]
    assert os.path.isfile(target + ".unreadable")


def test_bug069_the_settings_file_is_written_in_one_step(tmp_path, monkeypatch):
    """The file was opened for writing in place: a crash, or the other program reading at that moment,
    saw an empty or half-written file.  It is written beside the file and renamed; a write that fails
    leaves the settings as they were."""
    target = settings_file()
    projectfile.set_user_setting("cat_proof", True)
    with open(target, encoding="utf-8") as fh:
        before = fh.read()
    written = []
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wa+"):
            written.append(os.path.basename(str(file)))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    projectfile.set_user_setting("other", 3)
    monkeypatch.setattr(builtins, "open", real_open)
    assert written and "settings.toml" not in written  # never the file itself
    assert projectfile.user_setting("cat_proof") is True and projectfile.user_setting("other") == 3

    def failing_replace(*_args):
        raise OSError("disk full")

    with open(target, encoding="utf-8") as fh:
        before = fh.read()
    monkeypatch.setattr(os, "replace", failing_replace)
    projectfile.set_user_setting("other", 4)  # logs a warning
    monkeypatch.undo()
    with open(target, encoding="utf-8") as fh:
        assert fh.read() == before
    assert os.listdir(os.path.dirname(target)) == ["settings.toml"]  # no temporary file is left


# ---------------------------------------------------------------------------
# MDATK-BUG-001: spin buttons (and sliders) never clamp or round a loaded value
# ---------------------------------------------------------------------------


def spin_paths(win):
    return [b.path for b in win.binder.bindings if isinstance(b.widget, Gtk.SpinButton)]


def test_bug001_values_outside_a_spin_buttons_range_are_kept(make):
    """Every setting shown in a spin button: a value above (below) the button's range came back as the
    range's end, and Save wrote that."""
    win = make()
    above, below = Project(), Project()
    for path in spin_paths(win):
        adj = field(win, path).get_adjustment()
        whole = isinstance(get_path(above, path), int)
        set_path(above, path, int(adj.get_upper()) + 3 if whole else adj.get_upper() + 1.25)
        set_path(below, path, int(adj.get_lower()) - 3 if whole else adj.get_lower() - 1.25)
    assert len(spin_paths(win)) > 30
    for project in (above, below):
        project.visual.representations = [RepSpec(style="tube", size=7.5, opacity=1.0),
                                          RepSpec(style="vdw", size=0.05, opacity=0.01)]
        win.load_project(project)
        got = win.collect(validate=False)
        wrong = {path: (get_path(project, path), get_path(got, path)) for path in spin_paths(win)
                 if get_path(got, path) != get_path(project, path)}
        assert not wrong
        assert got.visual.representations == project.visual.representations
    # the values of the report
    p = Project()
    p.analyses.contacts.cutoff, p.analyses.hbonds.angle_cutoff, p.analyses.energy.cutoff = 25.0, 60.0, 60.0
    p.output.video.width = 7682
    win.load_project(p)
    got = win.collect(validate=False)
    assert (got.analyses.contacts.cutoff, got.analyses.hbonds.angle_cutoff, got.analyses.energy.cutoff,
            got.output.video.width) == (25.0, 60.0, 60.0, 7682)
    assert field(win, "analyses.contacts.cutoff").get_text() == "25.00"  # shown, not only kept
    # the range is the button's own again for the next project, and editing works as before
    win.load_project(Project())
    spin = field(win, "analyses.contacts.cutoff")
    assert spin.get_adjustment().get_upper() == 20
    spin.set_value(7.5)
    assert collected(win, "analyses.contacts.cutoff") == 7.5
    spin.set_value(99)
    assert collected(win, "analyses.contacts.cutoff") == 20


def test_bug001_a_problem_of_a_spin_button_field_is_flagged(make):
    """A value outside the suggested range is kept, so the check has to be able to flag it: flagging a
    spin button raised RuntimeError (PyGObject: 'unable to get the value' of the button's 'entry'), from
    the check after an edit and when a project with such a problem was opened."""
    from mdatk.gui import widgets as W

    win = make()
    p = Project()
    p.visual.proximity_cutoff = p.analyses.proximity_search_radius + 5.0  # an error validate() reports
    p.visual.zoom = 80.0  # outside the range of the button and of validate()
    win.load_project(p)
    found = win.check_problems()
    for path in ("visual.proximity_cutoff", "visual.zoom"):
        assert [pr for pr in found if pr.field == path]
        assert W.flag_of(field(win, path))[1] == "error"
    assert collected(win, "visual.zoom") == 80.0  # flagged, not rewritten
    field(win, "visual.zoom").set_value(2.0)
    field(win, "visual.proximity_cutoff").set_value(4.0)
    assert not [pr for pr in win.check_problems() if pr.field in ("visual.proximity_cutoff", "visual.zoom")]
    assert W.flag_of(field(win, "visual.zoom")) == ("", "")


def test_bug001_digits_a_spin_button_does_not_show_are_kept(make):
    """A value with more digits than its button shows stays as loaded, also when the button loses the
    focus (GTK then reads its own text back: 4.123456 became 4.12)."""
    win = make()
    p = Project()
    p.analyses.contacts.cutoff, p.visual.zoom, p.output.charts.width = 4.123456, 1.23456, 7.77777
    p.visual.representations = [RepSpec(size=1.23456, opacity=0.54321)]
    win.load_project(p)
    for path in spin_paths(win):
        field(win, path).update()  # what GTK does when the focus leaves the button
    for row in win.rep_rows:
        row["size"].update()
        row["opacity"].update()
    got = win.collect(validate=False)
    assert (got.analyses.contacts.cutoff, got.visual.zoom, got.output.charts.width) == (4.123456, 1.23456,
                                                                                        7.77777)
    assert got.visual.representations == p.visual.representations
    assert got.to_dict() == p.to_dict()


def test_bug001_sliders_keep_values_beyond_their_range(make):
    """The rotation sliders end at ±180° and the opacity slider at 1: a loaded value beyond that, or with
    more digits than the slider has, stays until the slider is moved."""
    win = make()
    p = Project()
    p.visual.rotation = [270.0, 0.123456, -200.0]
    p.density.opacity = 0.123456
    win.load_project(p)
    got = win.collect(validate=False)
    assert got.visual.rotation == [270.0, 0.123456, -200.0] and got.density.opacity == 0.123456
    win.rot_scales[0].set_value(90.0)  # moved: its new position counts
    assert collected(win, "visual.rotation") == [90.0, 0.123456, -200.0]
    win._on_canvas_rotate([10.0, 20.0, 30.0])  # dragged in the figure
    assert collected(win, "visual.rotation") == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# MDATK-BUG-002: optional numbers keep every digit
# ---------------------------------------------------------------------------


def test_bug002_optional_numbers_keep_every_digit(make):
    """The time step override and the energy target value were shown with six significant digits, and
    the shown text is what is saved."""
    win = make()
    rng = random.Random(2)
    values = [0.0123456789, -1234567.5, 1e-07, 123456789.123, 5.0, 0.1 + 0.2]
    values += [rng.uniform(-1e6, 1e6) for _ in range(20)] + [rng.random() * 1e-5 for _ in range(10)]
    for value in values:
        p = Project()
        p.inputs.dt_ps, p.analyses.energy.target_value = value, -value
        win.load_project(p)
        got = win.collect(validate=False)
        assert (got.inputs.dt_ps, got.analyses.energy.target_value) == (value, -value)
    p = Project()
    p.inputs.dt_ps = 2.0
    win.load_project(p)
    assert field(win, "inputs.dt_ps").get_text() == "2"  # short numbers look as before
    field(win, "inputs.dt_ps").set_text("0.5")
    assert collected(win, "inputs.dt_ps") == 0.5
    field(win, "inputs.dt_ps").set_text("")
    assert collected(win, "inputs.dt_ps") is None


def test_bug002_script_wizard_numbers_keep_every_digit(make):
    """The same six digits in the script wizard's optional number fields: a random seed of 1234567 was
    shown as 1.23457e+06 and written back as 1234570."""
    from mdatk.gui.wizard import OptionForm
    from mdatk.workflows.schema import Option

    form = OptionForm([Option("ig", "Random seed (ig)", "int", None, "", optional=True),
                       Option("tol", "Tolerance", "float", None, "", optional=True, digits=8)],
                      lambda _key: None)
    for values in ({"ig": 1234567, "tol": 0.0123456789}, {"ig": 123456789012, "tol": -1234567.5},
                   {"ig": 7, "tol": 1e-05}, {"ig": None, "tol": None}):
        form.load(values)
        assert form.read() == values
    form.load({"ig": 7, "tol": 1e-05})
    assert [form.rows[key][1].get_text() for key in ("ig", "tol")] == ["7", "1e-05"]


# ---------------------------------------------------------------------------
# MDATK-BUG-003: colours stay as written
# ---------------------------------------------------------------------------

COLOR_FIELDS = ["visual.base_color", "visual.ion_color", "visual.proximity_color", "visual.glow_color",
                "density.color"]


def test_bug003_colours_are_kept_as_written(make):
    """Colours went through GDK: Matplotlib colours MDATK draws (tab:orange, C3, #rrggbbaa, grey levels)
    were saved as black, and names GDK knows were rewritten (gold as #ffd700)."""
    from mdatk.gui import widgets as W

    win = make()
    for text in ("tab:orange", "C3", "#ff000080", "0.5", "gold", "green", "#1b9e77"):
        p = Project()
        for path in COLOR_FIELDS:
            set_path(p, path, text)
        win.load_project(p)
        got = win.collect(validate=False)
        assert [get_path(got, path) for path in COLOR_FIELDS] == [text] * len(COLOR_FIELDS)
        assert not [pr for pr in win.check_problems() if pr.field in COLOR_FIELDS]
    button = field(win, "visual.glow_color")
    p = Project()
    p.visual.glow_color, p.visual.base_color = "tab:orange", "green"
    win.load_project(p)
    # the buttons show the colours as they are drawn (Matplotlib's green is not GDK's)
    assert W.rgba_to_hex(button.get_rgba()) == "#ff7f0e"
    assert W.rgba_to_hex(field(win, "visual.base_color").get_rgba()) == "#008000"
    picked = Gdk.RGBA()
    picked.parse("#123456")
    button.set_rgba(picked)  # a colour picked in the chooser is written as #rrggbb
    button.emit("color-set")
    got = win.collect(validate=False)
    assert got.visual.glow_color == "#123456" and got.visual.base_color == "green"


def test_bug003_a_text_that_is_no_colour_is_kept_and_flagged(make):
    win = make()
    p = Project()
    p.visual.glow_color = "notacolour"
    win.load_project(p)
    assert collected(win, "visual.glow_color") == "notacolour"  # not silently black
    assert [pr for pr in win.check_problems() if pr.field == "visual.glow_color"]
    assert win.problem_view.registry.find("visual.glow_color").widgets[0] is field(win, "visual.glow_color")


# ---------------------------------------------------------------------------
# MDATK-BUG-004: chart formats without a check box
# ---------------------------------------------------------------------------


def test_bug004_chart_formats_without_a_check_box_are_kept(make):
    """The three check boxes (PNG, SVG, PDF) were all that was saved: jpg and tiff were dropped."""
    win = make()
    p = Project()
    p.output.charts.formats = ["jpg", "svg", "tiff"]
    win.load_project(p)
    assert collected(win, "output.charts.formats") == ["jpg", "svg", "tiff"]
    assert win.fmt_more.get_text() == "+ jpg, tiff"  # and they are shown beside the boxes
    win.fmt_checks["png"].set_active(True)
    assert collected(win, "output.charts.formats") == ["jpg", "svg", "tiff", "png"]
    win.fmt_checks["svg"].set_active(False)
    assert collected(win, "output.charts.formats") == ["jpg", "tiff", "png"]
    for formats in (["svg", "png"], ["pdf"], ["jpg"], []):  # the order and an empty list stay too
        p.output.charts.formats = list(formats)
        win.load_project(p)
        assert collected(win, "output.charts.formats") == formats
    win.load_project(Project())
    assert collected(win, "output.charts.formats") == ["png"] and win.fmt_more.get_text() == ""
    win.fmt_checks["png"].set_active(False)  # nothing ticked: png, as before
    assert collected(win, "output.charts.formats") == ["png"]


# ---------------------------------------------------------------------------
# MDATK-BUG-005: a choice that is not in the list
# ---------------------------------------------------------------------------


def choice_paths(win):
    return [b.path for b in win.binder.bindings
            if isinstance(b.widget, Gtk.ComboBoxText) and not b.widget.get_has_entry() and b.path != "layout"]


def test_bug005_an_unknown_choice_is_not_replaced_by_the_previous_projects(make):
    """A renderer that is not in the list was saved as whatever the project opened before had chosen."""
    win = make()
    software, vulkan = Project(), Project()
    software.visual.renderer, vulkan.visual.renderer = "software", "vulkan"
    win.load_project(software)
    win.load_project(vulkan)
    assert collected(win, "visual.renderer") == "vulkan"
    combo = field(win, "visual.renderer")
    assert combo.get_active_text() == "vulkan (unknown)"  # shown as what it is
    assert [p for p in win.check_problems() if p.field == "visual.renderer"]  # and flagged
    combo.set_active_id("vtk")  # choosing replaces it
    assert collected(win, "visual.renderer") == "vtk"
    win.load_project(software)  # the added item goes with its project
    assert [row[combo.get_id_column()] for row in combo.get_model()] == ["auto", "software", "vtk"]
    assert collected(win, "visual.renderer") == "software"


def test_bug005_every_choice_keeps_a_value_that_is_not_in_its_list(make):
    win = make()
    paths = choice_paths(win)
    assert len(paths) >= 12
    p = Project()
    for path in paths:
        set_path(p, path, "zz_new")
    p.layout.preset = "grid_9"
    p.layout.panels[1].type = "radar"
    p.visual.representations = [RepSpec(style="blobs", color="rainbow")]
    win.load_project(p)
    got = win.collect(validate=False)
    assert {path: get_path(got, path) for path in paths} == dict.fromkeys(paths, "zz_new")
    assert got.layout.preset == "grid_9" and got.layout.panels == p.layout.panels
    assert got.visual.representations == p.visual.representations
    assert got.to_dict() == p.to_dict()
    flagged = {pr.field for pr in win.check_problems()}
    assert set(paths) <= flagged and {"layout", "layout.panels.1.type"} <= flagged
    win.load_project(Project())  # nothing of it stays for the next project
    assert win.collect(validate=False).to_dict() == Project().to_dict()
    assert not {pr.field for pr in win.check_problems()} & (set(paths) | {"layout", "layout.panels.1.type"})


# ---------------------------------------------------------------------------
# MDATK-BUG-006: paths stay as the file has them
# ---------------------------------------------------------------------------


def relative_project(folder):
    """A project file with relative and ~ paths in ``folder`` (the files exist); returns its path."""
    folder.mkdir(exist_ok=True)
    for name in ("sys.parm7", "run.nc", "run2.nc", "ref.pdb", "start.pdb"):
        (folder / name).write_text("")
    p = Project()
    p.inputs.topology = "sys.parm7"
    p.inputs.trajectories = ["run.nc", "sub/../run2.nc"]
    p.inputs.external = [ExternalSeries(path="data/dist.dat", label="d", delimiter=";")]
    p.cleaning.reference_file = "ref.pdb"
    p.density.map_path = "~/maps/a.mrc"
    p.output.cache_dir = "/var/tmp/mdatk_cache"  # an absolute path stays absolute
    p.workflow.nodes = [PlanNode(id="n1", tool="note", files={"in": "start.pdb;ref.pdb"})]
    path = folder / "p.mdatk.toml"
    p.save(str(path))
    return str(path)


def test_bug006_saving_keeps_relative_paths(make, tmp_path):
    """The window holds absolute paths and wrote them: after one save, a project folder could no longer
    be moved or shared."""
    path = relative_project(tmp_path / "proj")
    before = projectfile.read(path)
    assert before["inputs"]["topology"] == "sys.parm7" and before["output"]["out_dir"] == "MDATK_output"
    win = make()
    win.open_path(path)
    assert field(win, "inputs.topology").get_text() == str(tmp_path / "proj" / "sys.parm7")
    assert win.save_project() and not win.messages
    assert projectfile.read(path) == before  # every setting as the file had it
    # a path that was changed is written as it was typed; the others stay
    field(win, "inputs.topology").set_text(str(tmp_path / "elsewhere.parm7"))
    win.traj_editor.add_row([str(tmp_path / "proj" / "third.nc")])
    assert win.save_project()
    after = projectfile.read(path)
    assert after["inputs"]["topology"] == str(tmp_path / "elsewhere.parm7")
    assert after["inputs"]["trajectories"] == ["run.nc", "sub/../run2.nc", str(tmp_path / "proj" / "third.nc")]
    after["inputs"]["topology"], after["inputs"]["trajectories"] = "sys.parm7", ["run.nc", "sub/../run2.nc"]
    assert after == before
    assert not win.analysis_changed()


def test_bug006_save_as_into_another_folder_keeps_naming_the_same_files(make, tmp_path, monkeypatch):
    path = relative_project(tmp_path / "proj")
    copy = str(tmp_path / "other" / "deep" / "q.mdatk.toml")
    win = make()
    win.open_path(path)
    monkeypatch.setattr("mdatk.gui.app.choose_save_path", lambda *_a, **_k: copy)
    assert win.save_project(save_as=True) and win.project_path == copy
    data = projectfile.read(copy)
    assert data["inputs"]["topology"] == os.path.join("..", "..", "proj", "sys.parm7")
    assert data["inputs"]["trajectories"] == [os.path.join("..", "..", "proj", "run.nc"),
                                              os.path.join("..", "..", "proj", "run2.nc")]
    assert data["density"]["map_path"] == "~/maps/a.mrc" and data["output"]["cache_dir"] == "/var/tmp/mdatk_cache"
    assert data["workflow"]["nodes"][0]["files"] == {"in": os.path.join("..", "..", "proj", "start.pdb") + ";"
                                                     + os.path.join("..", "..", "proj", "ref.pdb")}
    assert Project.load(copy).to_dict() == Project.load(path).to_dict()  # the same files, the same settings
    assert win.save_project()  # and saving the copy again keeps its spelling
    assert projectfile.read(copy) == data


# ---------------------------------------------------------------------------
# MDATK-BUG-007 / 008: empty slots and panels without a slot
# ---------------------------------------------------------------------------


def two_panels():
    p = Project()
    p.layout = LayoutParams(preset="equal_4", panels=[PanelSpec("structure", [], "Structure"),
                                                      PanelSpec("timeseries", ["rmsd/Target"], "RMSD")])
    return p


def panels(win):
    return [(s.type, s.series, s.title) for s in win.collect(validate=False).layout.panels]


def test_bug007_empty_slots_are_not_saved_as_panels(make):
    """A four-slot grid with two panels was saved with four: each empty slot had become an untitled time
    series of the first data key."""
    win = make(two_panels())
    assert panels(win) == [("structure", [], "Structure"), ("timeseries", ["rmsd/Target"], "RMSD")]
    assert len(win.slot_rows) == 2 and len(win.empty_slots) == 2
    win._rebuild_slots()  # as a change of the residue groups does
    win._on_canvas_swap(0, 1)
    assert panels(win) == [("timeseries", ["rmsd/Target"], "RMSD"), ("structure", [], "Structure")]
    win._add_panel()  # 'Add a panel' in the first empty slot: what every empty slot used to become
    assert len(win.slot_rows) == 3 and len(win.empty_slots) == 1
    added = win.collect(validate=False).layout.panels[2]
    assert added.type == "timeseries" and len(added.series) == 1
    win._remove_panel(win.slot_rows[0])  # 'Remove this panel': the later ones move up
    assert [p[0] for p in panels(win)] == ["structure", "timeseries"] and len(win.empty_slots) == 2


def test_bug007_a_panel_without_data_stays_without(make):
    p = two_panels()
    p.layout.panels[1].series = []
    win = make(p)
    assert panels(win)[1] == ("timeseries", [], "RMSD")


def test_bug008_a_smaller_grid_preset_deletes_no_panel(make):
    """equal_4 -> single -> equal_4 lost every panel but the first for good; so did saving a project
    whose grid has fewer slots than it has panels."""
    win = make(two_panels())
    win.preset_combo.set_active_id("single")
    assert [p[0] for p in panels(win)] == ["structure", "timeseries"]  # saved from the smaller grid too
    assert len(win.slot_rows) == 1 and len(win.more_panels) == 1
    assert len(win.canvas.project.layout.panels) == 1  # the preview draws the panel that has a slot
    win.preset_combo.set_active_id("equal_4")
    assert panels(win) == [("structure", [], "Structure"), ("timeseries", ["rmsd/Target"], "RMSD")]
    assert win.more_panels == []
    # a project with more panels than its grid has slots
    p = Project()
    p.layout.preset = "equal_2"
    assert len(p.layout.panels) == 4
    win.load_project(p)
    assert len(win.slot_rows) == 2 and win.collect(validate=False).layout.panels == p.layout.panels
    win.slot_rows[1]["title"].set_text("Mine")  # editing a shown panel keeps the others
    got = win.collect(validate=False).layout.panels
    assert got[1].title == "Mine" and got[2:] == p.layout.panels[2:]
    win._drop_more_panels()  # 'Remove' beside the note: now they are gone
    assert len(win.collect(validate=False).layout.panels) == 2 and win.more_panels == []


# ---------------------------------------------------------------------------
# MDATK-BUG-009: unsaved edits are asked about, an untouched project is not
# ---------------------------------------------------------------------------


def awkward_project(folder):
    """A project file full of values the widgets cannot show as they are (items 001 to 008)."""
    path = relative_project(folder)
    p = Project.from_dict(projectfile.read(path))
    p.analyses.contacts.cutoff, p.analyses.hbonds.angle_cutoff, p.analyses.energy.cutoff = 25.0, 60.0, 60.0
    p.analyses.stacking.distance_cutoff = 4.123456
    p.inputs.dt_ps, p.analyses.energy.target_value = 0.0123456789, -1234567.5
    p.visual.glow_color, p.visual.proximity_color, p.visual.base_color = "tab:orange", "#ff000080", "gold"
    p.visual.rotation = [270.0, 0.123456, 0.0]
    p.visual.renderer = "vulkan"
    p.visual.representations = [RepSpec(style="tube", size=7.5, opacity=0.54321)]
    p.output.charts.formats = ["jpg", "svg"]
    p.output.video.width = 7682
    p.layout = LayoutParams(preset="equal_4", panels=two_panels().layout.panels)
    p.save(path)
    return path


def test_bug009_an_untouched_project_has_no_unsaved_changes(make, asks, tmp_path):
    """Opened and not edited, a project is the one in the file: nothing is asked when it is replaced or
    the window closes, and saving it changes no setting."""
    path = awkward_project(tmp_path / "proj")
    before = projectfile.read(path)
    win = make()
    win.open_path(path)
    assert win.collect(validate=False).to_dict() == Project.load(path).to_dict()
    assert not win.analysis_changed() and win.header.get_subtitle() == path
    for p in spin_paths(win):
        field(win, p).update()  # the focus passing through the spin buttons is no edit
    win.check_problems()
    assert not win.analysis_changed() and win.header.get_subtitle() == path
    assert win._on_close() is False  # the window may close
    win.new_project()
    win.open_path(path)
    win.close_project()
    win.open_path(path)
    assert asks.asked == []
    assert win.save_project()
    assert projectfile.read(path) == before


def test_bug009_new_open_close_and_quit_ask_about_unsaved_edits(make, asks, tmp_path):
    """New, Open, Close and closing the window dropped unsaved edits without a word."""
    path = relative_project(tmp_path / "proj")
    other = relative_project(tmp_path / "other")
    default = Project().analyses.contacts.cutoff
    win = make()
    win.open_path(path)
    field(win, "analyses.contacts.cutoff").set_value(7.5)
    win.check_problems()
    assert win.analysis_changed() and win.header.get_subtitle() == f"{path} (modified)"
    for replace in (win.new_project, win.close_project, lambda: win.open_path(other)):
        asks.answers.append("cancel")
        replace()
        assert win.project_path == path and collected(win, "analyses.contacts.cutoff") == 7.5
    asks.answers.append("cancel")
    assert win._on_close() is True  # the window stays open
    assert asks.asked == ["the project"] * 4
    # Don't save: the edits go
    asks.answers.append("discard")
    win.open_path(other)
    assert win.project_path == other and collected(win, "analyses.contacts.cutoff") == default
    assert Project.load(path).analyses.contacts.cutoff == default
    # Save: saved first, then replaced
    field(win, "analyses.contacts.cutoff").set_value(8.5)
    asks.answers.append("save")
    win.close_project()
    assert win.project_path is None and Project.load(other).analyses.contacts.cutoff == 8.5
    assert win.header.get_subtitle() == "unsaved project"
    # closing the window: Don't save lets it close, Save saves first
    win.open_path(path)
    field(win, "analyses.contacts.cutoff").set_value(9.5)
    asks.answers += ["discard", "save"]
    assert win._on_close() is False and Project.load(path).analyses.contacts.cutoff == default
    assert win._on_close() is False and Project.load(path).analyses.contacts.cutoff == 9.5
    assert asks.answers == [] and len(asks.asked) == 8
    assert not win.analysis_changed() and win.header.get_subtitle() == path


def test_bug009_a_save_that_fails_keeps_the_project(make, asks, tmp_path):
    """'Save' in the question, but the settings cannot be saved: the project stays (with its edits)."""
    path = relative_project(tmp_path / "proj")
    win = make()
    win.open_path(path)
    field(win, "inputs.stop").set_text("x")  # an invalid field is an unsaved edit that cannot be saved
    for attempt in (win.new_project, win._on_close):
        asks.answers.append("save")
        result = attempt()
        assert win.project_path == path and field(win, "inputs.stop").get_text() == "x"
        assert result is not False  # (the window does not close)
    assert [m[1] for m in win.messages] == ["Some settings are invalid"] * 2


# ---------------------------------------------------------------------------
# MDATK-BUG-010: form errors in words
# ---------------------------------------------------------------------------


def test_bug010_form_errors_name_the_field_in_words(make):
    """A non-number in a number field was reported as "inputs.stop: invalid literal for int() with base
    10: 'x'", and panel options that are no JSON with the JSON reader's own message."""
    from mdatk.problems import describe

    win = make()
    cases = {"inputs.stop": "Input & cleaning › Frames › Stop: 'x' is not a whole number",
             "cleaning.stop": f"{describe('cleaning.stop')}: 'x' is not a whole number",
             "inputs.dt_ps": "Input & cleaning › Frames › Time step override: 'x' is not a number",
             "analyses.energy.target_value": f"{describe('analyses.energy.target_value')}: 'x' is not a number"}
    for path, expected in cases.items():
        field(win, path).set_text("x")
        with pytest.raises(ValueError) as error:
            win.collect(validate=False)
        assert str(error.value) == expected
        assert [str(p) for p in win.check_problems() if p.field == path] == [expected]
        field(win, path).set_text("")
    for text, reason in (("{bad", "not valid JSON (character 2)"), ("[1, 2]", "not a JSON object")):
        win.slot_rows[0]["opts"].set_text(text)
        with pytest.raises(ValueError) as error:
            win.collect(validate=False)
        message = str(error.value)
        assert message.startswith(f"Layout & panels › Panel 1 (Structure) › Options: panel options {text!r} "
                                  f"are {reason}")
        assert "Expecting" not in message and "layout.panels" not in message
    win.slot_rows[0]["opts"].set_text("")
    win.collect(validate=False)


# ---------------------------------------------------------------------------
# MDATK-BUG-068: closing the window during a job
# ---------------------------------------------------------------------------


def test_bug068_closing_during_a_job_asks_and_waits_for_the_job(make, monkeypatch):
    """Closing the window cancelled the job without waiting: its thread ended with the program, in the
    middle of whatever it was writing or tidying up."""
    win = make()
    started, tidied = threading.Event(), []

    def job(rep):
        started.set()
        try:
            while True:
                rep.check_cancel()
                time.sleep(0.01)
        finally:
            time.sleep(0.3)  # removing its temporary files
            tidied.append(True)

    win.start_job("Waiting", job, lambda _result: None)
    assert started.wait(10)
    worker = win.worker
    questions, answers = [], [Gtk.ResponseType.CANCEL, Gtk.ResponseType.ACCEPT]

    def ask(_parent, text, detail, buttons, *_rest):
        questions.append((text, detail, [label for label, _response in buttons]))
        return answers.pop(0)

    monkeypatch.setattr("mdatk.gui.app.ask", ask)
    assert win._on_close() is True  # Cancel: the window stays and the job goes on
    assert worker.running and not tidied
    assert win._on_close() is False  # Stop it and close: the job has ended when the window closes
    assert tidied and not worker.thread.is_alive()
    assert questions == [("A job is still running", "Closing MDATK stops it.",
                          ["Cancel", "Stop it and close"])] * 2
    pump(0.2)  # the job's own end reaches the window
    assert "Cancelled." in log_text(win)


# ---------------------------------------------------------------------------
# MDATK-BUG-086: keys of a project file that the model does not know
# ---------------------------------------------------------------------------

UNKNOWN = {"int": 3, "float": 0.25, "text": "kept", "flag": True, "list": [1, 2, 3],
           "table": {"a": 1, "deep": {"b": [1.5, "x"]}}}


def file_with_unknown_keys(folder, version=None):
    """A project file with keys this version does not know at every level; returns its path."""
    from mdatk import tomlio

    folder.mkdir(exist_ok=True)
    path = str(folder / "p.mdatk.toml")
    Project().save(path)
    data = tomlio.load(path)
    if version:
        data["mdatk_version"] = version
    data["plugin_x"] = dict(UNKNOWN)  # a table of another tool
    data["future_top"] = "kept"
    data["inputs"]["future_option"] = 3
    data["analyses"]["contacts"]["future_option"] = dict(UNKNOWN)
    data["groups"][1]["future_option"] = 0.5
    data["layout"]["future_option"] = [1, 2]
    data["layout"]["panels"][2]["future_option"] = "kept"
    data["analyses"]["rmsd"]["targets"][0]["future_option"] = True
    data["workflow"]["future_budget"] = 7
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tomlio.dumps(data))
    return path


def unknown_values(path):
    from mdatk import tomlio

    data = tomlio.load(path)
    return {"plugin_x": data.get("plugin_x"), "future_top": data.get("future_top"),
            "inputs": data["inputs"].get("future_option"),
            "contacts": data["analyses"]["contacts"].get("future_option"),
            "group": [g.get("future_option") for g in data["groups"]],
            "layout": data["layout"].get("future_option"),
            "panel": [p.get("future_option") for p in data["layout"]["panels"]],
            "target": data["analyses"]["rmsd"]["targets"][0].get("future_option"),
            "workflow": data["workflow"].get("future_budget"), "version": data["mdatk_version"]}


def test_bug086_a_save_keeps_the_keys_it_does_not_know(tmp_path, caplog):
    """Keys the model does not know (a newer MDATK's settings, another tool's table, a typo) were dropped
    when the file was read, so the next save deleted them."""
    import mdatk

    path = file_with_unknown_keys(tmp_path / "proj")
    before = unknown_values(path)
    assert before["contacts"] == UNKNOWN and before["group"] == [None, 0.5] and before["panel"][2] == "kept"
    with caplog.at_level("WARNING", logger="mdatk"):
        project = Project.load(path)
        Project.load(path)  # a window reads its file again and again: told once
    told = [r.getMessage() for r in caplog.records if "unknown key" in r.getMessage()]
    assert len(told) == 1 and "9 unknown keys kept and saved back unchanged" in told[0]
    assert "project.inputs.future_option" in told[0] and "project.groups[1].future_option" in told[0]
    assert project.to_dict() == project.copy().to_dict()
    assert not hasattr(project.inputs, "future_option")  # kept, not a setting
    project.inputs.step = 2
    project.save(path)
    assert unknown_values(path) == before and Project.load(path).inputs.step == 2
    Project.load(path).save(path)  # and again
    assert unknown_values(path) == before
    # RNAdo's plan save re-reads and rewrites the file through the same model
    projectfile.edit_workflow(path, lambda wf: setattr(wf, "project_name", "apt"))
    assert unknown_values(path) == before and Project.load(path).workflow.project_name == "apt"
    assert before["version"] == mdatk.__version__


def test_bug086_the_version_of_a_newer_file_does_not_go_back(tmp_path):
    """Saving wrote this version over the mdatk_version of a file a newer MDATK had written."""
    import mdatk

    newer = file_with_unknown_keys(tmp_path / "newer", version="99.2.0")
    Project.load(newer).save(newer)
    assert unknown_values(newer)["version"] == "99.2.0"
    projectfile.edit_workflow(newer, lambda wf: None)
    assert unknown_values(newer)["version"] == "99.2.0"
    older = file_with_unknown_keys(tmp_path / "older", version="0.0.4")
    Project.load(older).save(older)
    assert unknown_values(older)["version"] == mdatk.__version__  # an older file gets this version, as before


def test_bug086_set_still_refuses_keys_the_model_does_not_know(tmp_path):
    from mdatk.cli import apply_override

    path = file_with_unknown_keys(tmp_path / "proj")
    project = Project.load(path)
    for assignment in ("inputs.future_option=4", "inputs.nope=1", "plugin_x.text=no", "future_top=x",
                       "inputs.unknown.keys=1"):
        with pytest.raises(ValueError):
            apply_override(project, assignment)
    apply_override(project, "inputs.step=5")  # a setting: the unknown keys go along
    project.save(path)
    assert unknown_values(path) == unknown_values(file_with_unknown_keys(tmp_path / "again"))
    assert Project.load(path).inputs.step == 5


def test_bug086_the_analysis_window_saves_unknown_keys_back(make, tmp_path):
    """The window builds the layout and the rows of its tables anew from the widgets: their unknown keys
    go with them (rows that were edited or moved included)."""
    path = file_with_unknown_keys(tmp_path / "proj", version="99.2.0")
    before = unknown_values(path)
    win = make()
    win.open_path(path)
    assert win.save_project() and unknown_values(path) == before
    win.group_editor.store[1][1] = "13-20"  # an edited row
    field(win, "analyses.contacts.cutoff").set_value(7.5)
    win.slot_rows[2]["title"].set_text("Mine")
    win.preset_combo.set_active_id("equal_6")
    assert win.save_project()
    saved = Project.load(path)
    assert (saved.groups[1].residues, saved.analyses.contacts.cutoff, saved.layout.preset) == ("13-20", 7.5,
                                                                                               "equal_6")
    assert unknown_values(path) == before
    win.group_editor.view.get_selection().select_path(Gtk.TreePath(1))
    win.group_editor.move(-1)  # a moved row keeps its keys
    assert win.save_project()
    assert unknown_values(path)["group"] == [0.5, None]
    assert [g.name for g in Project.load(path).groups] == ["Target", "Aptamer"]


# ---------------------------------------------------------------------------
# MDATK-BUG-097: the data labels of the panel slots follow an analysis' groups
# ---------------------------------------------------------------------------


def test_bug097_slot_data_labels_follow_the_groups_of_their_analysis(make):
    """Another group chosen for the contacts: the slot's Data choice went on saying "Aptamer–Target
    heavy-atom contacts" until the editors were rebuilt for another reason."""
    from mdatk.config import ResidueGroup

    p = Project()
    p.groups.append(ResidueGroup(name="Loop", residues="5-8"))
    label = "Aptamer–Target heavy-atom contacts"
    p.layout = LayoutParams(preset="equal_2", panels=[PanelSpec("timeseries", ["contacts/atom_pairs"], label),
                                                      PanelSpec("timeseries", ["contacts/atom_pairs"], "Mine")])
    win = make(p)
    rows = list(win.slot_rows)
    data = rows[0]["data"]

    def shown():
        return data.get_model()[data.get_active_iter()][0]

    assert shown() == f"{label}   [contacts/atom_pairs]"
    field(win, "analyses.contacts.group_b").get_child().set_text("Loop")
    assert collected(win, "analyses.contacts.group_b") == "Loop"
    assert shown() == "Aptamer–Loop heavy-atom contacts   [contacts/atom_pairs]"
    assert win.slot_rows == rows and data.get_active_id() == "contacts/atom_pairs"  # relabelled, not rebuilt
    # a title that was the data's label follows; a typed title stays
    assert [row["title"].get_text() for row in rows] == ["Aptamer–Loop heavy-atom contacts", "Mine"]
    assert [s.series for s in win.collect(validate=False).layout.panels] == [["contacts/atom_pairs"]] * 2


# ---------------------------------------------------------------------------
# MDATK-BUG-098: Open video
# ---------------------------------------------------------------------------


def test_bug098_open_video_is_about_the_current_project_and_folder(make, asks, tmp_path, monkeypatch):
    """After a Render, 'Open video' kept opening that file: after the output folder was changed, after
    another project was opened, and after 'Re-render from cache' had written a newer video."""
    from types import SimpleNamespace

    from mdatk.gui import desktop

    opened = []
    monkeypatch.setattr(desktop, "open_file", opened.append)
    first, second = tmp_path / "a", tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
    old = first / "MDATK_video_1.mp4"
    old.write_text("")
    new = second / "MDATK_video_2.mp4"
    new.write_text("")
    (second / "MDATK_video_3.mp4").mkdir()  # a folder named like a video is none
    p = Project()
    p.output.out_dir = str(first)
    win = make(p)
    win.last_video = str(old)  # as Render leaves it
    win.open_video()
    field(win, "output.out_dir").set_text(str(second))  # another output folder: its newest video
    win.open_video()
    assert opened == [str(old), str(new)]
    # another project: the last render is not its video
    other = Project()
    other.output.out_dir = str(tmp_path / "c")
    path = str(tmp_path / "c" / "q.mdatk.toml")
    other.save(path)
    win.last_video = str(old)
    asks.answers.append("discard")  # (the output folder typed above)
    win.open_path(path)
    assert win.last_video is None
    win.open_video()
    assert opened == [str(old), str(new)] and win.messages[-1][:2] == ("error", "No video has been rendered yet.")
    # 'Re-render from cache' writes a video: that is the one 'Open video' plays
    rendered = tmp_path / "c" / "MDATK_video_9.mp4"
    fake = SimpleNamespace(load_cached_dataset=lambda: None, render_charts=lambda *_a: [],
                           render_video=lambda *_a: (rendered.parent.mkdir(exist_ok=True), rendered.write_text(""),
                                                     str(rendered))[2])
    monkeypatch.setattr(win, "_pipeline", lambda *_a: fake)
    win.rerender_from_cache()
    end = time.time() + 20
    while win.last_video is None and time.time() < end:
        pump(0.05)
    assert win.last_video == str(rendered)
    win.open_video()
    assert opened[-1] == str(rendered)
    # two videos written at the same time: the later name
    twin = rendered.parent / "MDATK_video_9b.mp4"
    twin.write_text("")
    stamp = os.path.getmtime(rendered)
    os.utime(twin, (stamp, stamp))
    assert desktop.latest_video(str(rendered.parent), "MDATK_video.mp4") == str(twin)


# ---------------------------------------------------------------------------
# MDATK-BUG-102: log lines reach the window in batches
# ---------------------------------------------------------------------------


def test_bug102_a_chatty_job_costs_one_callback_per_batch_not_per_line(make):
    """Every log line of a job was one callback on the main loop: a job that prints a lot kept the
    window from redrawing and reacting."""
    win = make()
    shown = []  # the lines of each insertion into the log
    win.log_buffer.connect("insert-text", lambda _buf, _at, text, _n: shown.append(text.count("\n")))
    lines = 60000
    done = []

    def job(rep):
        for k in range(lines):
            rep.log(f"line {k}")
        return "ok"

    win.start_job("Chatty", job, lambda result: (win.append_log("the job has ended"), done.append(result)))
    end = time.time() + 120
    while not done and time.time() < end:
        pump(0.05)
    assert done == ["ok"]
    assert len(shown) < lines / 20  # batches (it was one call per line)
    assert max(shown) <= 2001  # a batch is small enough to insert at once (2 000 lines and the note)
    text = log_text(win)
    # the job's last lines are shown, before what follows the job; skipped lines are counted
    assert text.index(f"line {lines - 1}\n") < text.index("the job has ended")
    # faster than the log can show: the newest lines, and a note that counts the others
    from mdatk.gui.worker import LogBatch

    got = []
    batch = LogBatch(got.append)
    for k in range(10000):
        batch.feed(f"line {k}")
    batch.flush()
    (text,) = got
    assert text.split("\n")[0] == "… 8000 line(s) not shown: they came faster than the log can show them"
    assert text.split("\n")[1:] == [f"line {k}" for k in range(8000, 10000)]


def test_bug102_messages_of_the_logging_handler_are_batched_and_in_order(make):
    import logging

    from mdatk.gui.app import GuiLogHandler

    win = make()
    handler = GuiLogHandler(win)
    logger = logging.getLogger("mdatk.test_bugreport")
    logger.addHandler(handler)
    try:
        thread = threading.Thread(target=lambda: [logger.warning("from a thread %d", k) for k in range(50)])
        thread.start()
        thread.join()
        win.append_log("typed after them")  # what is logged later comes later
    finally:
        logger.removeHandler(handler)
    text = log_text(win)
    assert text.count("WARNING: from a thread") == 50
    assert text.index("WARNING: from a thread 49") < text.index("typed after them")


# ---------------------------------------------------------------------------
# MDATK-BUG-103: the whole detail text in dialogs; the still fits the screen
# ---------------------------------------------------------------------------


def test_bug103_dialogs_show_the_whole_detail_text(make, monkeypatch):
    """The detail of error and message dialogs was cut after 3 000 characters without a mark (the box
    scrolls long text since 0.1.0)."""
    from mdatk.gui.rnado import RNAdoWindow

    win = make()
    shown = []
    monkeypatch.setattr(Gtk.MessageDialog, "run",
                        lambda dlg: shown.append(dlg.props.secondary_text) or Gtk.ResponseType.OK, raising=False)
    type(win).error(win, "Failed", "y" * 5000)  # (the fixture replaced the window's own error and info)
    type(win).info(win, "Done", "z" * 5000)
    rnado = RNAdoWindow(win.get_application(), None, None)
    try:
        rnado._message("Failed", "r" * 5000, Gtk.MessageType.ERROR)
    finally:
        rnado.destroy()
    assert shown == ["y" * 5000, "z" * 5000, "r" * 5000]
    # a text beyond any dialog: its start, and a line that says so and where the rest is
    huge = "a" * 50000
    type(win).error(win, "Failed", huge)
    assert shown[-1].startswith("a" * 20000 + "\n…") and len(shown[-1]) < 20200
    assert "30000 more characters: the full text is in the log" in shown[-1]
    assert huge in log_text(win)


def test_bug103_a_still_fits_the_screen_in_both_directions(make, tmp_path):
    """The still preview was scaled to 1 200 pixels of width only: a portrait still was shown at its full
    height, beyond the screen."""
    from gi.repository import GdkPixbuf

    win = make()
    area_h = Gdk.Display.get_default().get_monitor(0).get_workarea().height
    path = str(tmp_path / "still.png")
    GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 400, 4 * area_h).savev(path, "png", [], [])
    before = set(Gtk.Window.list_toplevels())
    win.show_image(path)
    (dialog,) = [w for w in Gtk.Window.list_toplevels() if w not in before and isinstance(w, Gtk.Dialog)]
    try:
        images = []

        def find(widget):
            if isinstance(widget, Gtk.Image):
                images.append(widget)
            elif isinstance(widget, Gtk.Container):
                widget.foreach(find)

        find(dialog.get_content_area())
        pix = images[0].get_pixbuf()
        assert pix.get_height() <= 0.9 * area_h + 1 and pix.get_width() == round(400 * pix.get_height() / (4 * area_h))
    finally:
        dialog.destroy()
    from mdatk.gui.dialogtext import fit_size

    assert fit_size(1080, 1920, 1366, 768) == (389, 691)  # a portrait still on a laptop screen
    assert fit_size(3840, 2160, 1366, 768) == (1229, 691)
    assert fit_size(320, 180, 1366, 768) == (320, 180)  # never enlarged


# ---------------------------------------------------------------------------
# MDATK-BUG-104: the preview after 'Run cleaning now'
# ---------------------------------------------------------------------------


def test_bug104_the_real_frame_is_loaded_again_after_cleaning(make, monkeypatch):
    """The reference frame loaded before cleaning (raw: with water and ions) stayed in the preview."""
    from types import SimpleNamespace

    win = make()
    inputs = SimpleNamespace(cleaned=SimpleNamespace(reused=False, script="clean.in"), topology="c.parm7",
                             trajectories=["c.nc"])
    loads = []
    fake = SimpleNamespace(resolve_inputs=lambda: inputs,
                           structure_preview=lambda: (loads.append(1), {"coords": [], "cleaned": True})[1])
    monkeypatch.setattr(win, "_pipeline", lambda *_a: fake)

    def wait(done):
        end = time.time() + 20
        while not done() and time.time() < end:
            pump(0.05)
        return done()

    win.run_cleaning()  # no real frame in the preview: none is loaded
    assert wait(lambda: win.messages) and win.messages[0][:2] == ("info", "Cleaning finished")
    pump(0.3)
    assert not loads and win.preview_data is None
    win.preview_data = {"coords": [], "cleaned": False}  # as 'Load real frame' left it, from the raw files
    win.run_cleaning()
    assert wait(lambda: win.preview_data.get("cleaned")) and loads == [1]
    assert not [m for m in win.messages if m[0] == "error"]
