"""GTK tests of the residue-group page: renames that follow every reference, deleted groups that are
still used, and groups from the chains of a topology.  Run under a display or ``xvfb-run -a pytest``."""

import os
import re
import sys
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

sys.path.insert(0, os.path.dirname(__file__))
from test_groups import system_atoms, write_parm7_bonded  # noqa: E402

from mdatk import projectfile  # noqa: E402
from mdatk.config import Project, RepSpec  # noqa: E402


def pump(cond, timeout=60):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        Gtk.main_iteration_do(False)
        time.sleep(0.005)
    return cond()


def make_window(project, path=None):
    from mdatk.gui.app import MainWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    win = MainWindow(app, project, path)
    win.messages = []
    win.info = lambda t, d="": win.messages.append(("info", t, d))
    win.error = lambda t, d="": win.messages.append(("error", t, d))
    win.resize(1300, 850)
    win.show_all()
    assert pump(lambda: win.canvas.render_count > 0, 30)
    win.page_buttons["groups"].clicked()
    pump(lambda: False, 0.2)
    return win


@pytest.fixture
def window(demo_dry, tmp_path):
    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path / "gui_out")
    win = make_window(p, demo_dry["preset"])
    yield win
    win.destroy()


def edit_name(win, row: int):
    """Open the name cell of ``row`` like a click would; returns its entry."""
    view = win.group_editor.view
    col = view.get_column(0)
    opened = []
    handler = col.get_cells()[0].connect("editing-started", lambda _c, e, _p: opened.append(e))
    view.grab_focus()
    view.set_cursor(Gtk.TreePath(row), col, True)
    assert pump(lambda: opened, 5)
    col.get_cells()[0].disconnect(handler)
    return opened[0]


def project_text(win, tmp_path) -> str:
    path = str(tmp_path / "check.mdatk.toml")
    projectfile.save(path, win.collect(validate=False))
    return open(path, encoding="utf-8").read()


def test_renaming_a_group_renames_it_everywhere(window, tmp_path):
    win = window
    slot = win.slot_rows[1]
    slot["type"].set_active_id("contact_map")
    slot["data"].set_active_id("contacts/map")
    slot["ga"].set_active_id("Aptamer")
    slot["gb"].set_active_id("Target")
    for b in win.binder.bindings:  # type the group names like a user would
        if b.path == "visual.structure_groups":
            b.widget.set_text("Aptamer, Target")
    win._add_rep_row(RepSpec("cartoon", ["Aptamer"]))
    assert "group:Aptamer" in project_text(win, tmp_path)

    entry = edit_name(win, 0)
    entry.set_text("Binder")
    entry.activate()  # Enter
    pump(lambda: False, 0.2)
    text = project_text(win, tmp_path)
    assert not re.search(r"(?<![\w.\-])Aptamer(?![\w.\-])", text)
    assert [r[0] for r in win.group_editor.store] == ["Binder", "Target"]
    status = win.group_status.get_text()
    assert re.fullmatch(r"Renamed Aptamer → Binder in \d+ places", status), status
    assert "analyses.contacts.group_a" in win.group_status.get_tooltip_text()
    p = win.collect(validate=False)
    assert p.layout.panels[1].series[0] == "contacts@Binder+Target/map"
    assert win.slot_rows[1]["ga"].get_active_id() == "Binder"
    assert p.visual.structure_groups == ["Binder", "Target"]
    assert p.visual.representations[0].groups == ["Binder"]
    assert p.layout.panels[1].title == "Binder–Target contact map"  # an automatic title follows
    assert p.analyses.rmsd.targets[0].selection.startswith("(group:Binder or group:Target)")
    assert p.output.pdb.selection == "Binder, Target"
    assert not p.validate() or not [m for m in p.validate() if "group" in m.lower()]
    assert not win.messages  # no dialog


def test_group_names_are_checked_and_leaving_the_cell_commits(window):
    win = window
    entry = edit_name(win, 1)
    entry.set_text("two words")
    assert entry.get_style_context().has_class("error")
    assert entry.get_icon_name(Gtk.EntryIconPosition.SECONDARY) == "dialog-error-symbolic"
    assert "no spaces" in entry.get_icon_tooltip_text(Gtk.EntryIconPosition.SECONDARY)
    entry.set_text("Aptamer")
    assert "already" in entry.get_icon_tooltip_text(Gtk.EntryIconPosition.SECONDARY)
    entry.activate()
    pump(lambda: False, 0.2)
    assert [r[0] for r in win.group_editor.store] == ["Aptamer", "Target"]  # refused
    assert "not renamed" in win.group_status.get_text()
    assert win.group_status.get_style_context().has_class("error")
    assert win.collect(validate=False).analyses.contacts.group_b == "Target"

    entry = edit_name(win, 1)  # Escape cancels
    entry.set_text("Dropped")
    ev = Gdk.Event.new(Gdk.EventType.KEY_PRESS)
    ev.key.keyval = Gdk.KEY_Escape
    ev.key.window = win.group_editor.view.get_window()
    entry.event(ev)
    win.topology_entry.grab_focus()
    pump(lambda: False, 0.2)
    assert [r[0] for r in win.group_editor.store] == ["Aptamer", "Target"]

    entry = edit_name(win, 1)
    entry.set_text("Ligand")
    assert not entry.get_style_context().has_class("error")
    win.topology_entry.grab_focus()  # leaving the cell commits
    assert pump(lambda: win.group_editor.store[1][0] == "Ligand", 5)
    p = win.collect(validate=False)
    assert p.analyses.contacts.group_b == "Ligand" and p.visual.proximity_set == "Ligand"
    assert win.group_status.get_text().startswith("Renamed Target → Ligand")


def test_deleting_a_used_group_says_where_it_is_used(window):
    win = window
    slot = win.slot_rows[1]
    slot["type"].set_active_id("contact_map")
    slot["data"].set_active_id("contacts/map")
    slot["gb"].set_active_id("Target")
    win.group_editor.view.get_selection().select_path(Gtk.TreePath(1))
    win.group_editor.remove_selected()
    pump(lambda: False, 0.2)
    status = win.group_status.get_text()
    assert status.startswith("Removed group still in use")
    for where in ("analyses.contacts.group_b", "visual.proximity_set", "layout.panels[1].series[0]",
                  "output.pdb.selection"):
        assert where in status, where
    gb = win.slot_rows[1]["gb"]  # the panel keeps naming the missing group until it is changed
    assert gb.get_active_id() == "Target" and gb.get_active_text() == "Target (missing)"
    assert win.collect(validate=False).layout.panels[1].series[0] == "contacts@Aptamer+Target/map"


def test_groups_from_chains(window, tmp_path):
    """The demo's groups are user-set here (edited), so choosing a topology only offers its chains;
    the button adds them once."""
    win = window
    win.group_editor.store[1][1] = "13-23"  # no longer the defaults
    path = write_parm7_bonded(str(tmp_path / "four.parm7"), system_atoms())
    win.topology_entry.set_text(path)
    assert pump(lambda: "without a group" in win.group_status.get_text(), 30)
    assert [r[0] for r in win.group_editor.store] == ["Aptamer", "Target"]
    win.groups_from_chains()
    assert pump(lambda: win.group_status.get_text().startswith("Added"), 30)
    rows = [tuple(r) for r in win.group_editor.store]
    assert [r[:2] for r in rows] == [("Aptamer", "1-12"), ("Target", "13-23"), ("Chain_A", "1-4"),
                                     ("Chain_B", "5-8"), ("Chain_C", "9-12"), ("Chain_D", "13-16")]
    assert len({r[3] for r in rows}) == 6
    assert win.collect(validate=False).analyses.contacts.group_a == "Aptamer"
    win.groups_from_chains()
    assert pump(lambda: "already has a group" in win.group_status.get_text(), 30)
    assert len(win.group_editor.store) == 6


def test_choosing_a_topology_replaces_untouched_defaults(tmp_path):
    win = make_window(Project())
    try:
        path = write_parm7_bonded(str(tmp_path / "four.parm7"), system_atoms())
        win.topology_entry.set_text(path)
        assert pump(lambda: win.group_status.get_text().startswith("Default groups replaced"), 30)
        assert [r[0] for r in win.group_editor.store] == ["Chain_A", "Chain_B", "Chain_C", "Chain_D"]
        p = win.collect(validate=False)
        assert (p.analyses.hbonds.group_a, p.analyses.hbonds.group_b) == ("Chain_A", "Chain_B")
        assert p.analyses.mg.ligand_selection == "group:Chain_A or group:Chain_B"
        combos = [c.get_child().get_text() for c in win.group_combos]
        assert set(combos) == {"Chain_A", "Chain_B"}
    finally:
        win.destroy()
