"""GTK 3 desktop interface for MDATK: the analysis window.

The workflow planner, the script wizard, the PDB clean-up and the structure builder live in RNAdo
(:mod:`mdatk.gui.rnado`), a program of its own that this window starts ("Open in RNAdo…").  Both may
have the same project file open: this window owns the analysis settings and keeps the plan RNAdo saved
(see :mod:`mdatk.gui.projectui`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import warnings
from typing import Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Pango", "1.0")
from gi.repository import Gdk, GdkPixbuf, Gio, GLib, Gtk, Pango  # noqa: E402

from .. import PROJECT_ABBREV, PROJECT_TITLE, __version__  # noqa: E402
from .. import chaingroups  # noqa: E402
from ..analysis import expected_keys  # noqa: E402
from ..analysis.base import (analysis_class, instance_id, key_label,  # noqa: E402
                             parse_instance, split_key)
from ..cleaning import conda  # noqa: E402
from ..config import (DEFAULT_PANEL_TITLES, REP_COLORS, REP_STYLES, ExternalSeries,  # noqa: E402
                      PanelSpec, Project, RepSpec, ResidueGroup, RMSDTarget, group_name_problem,
                      layout_warnings, stable_hash)
from ..problems import Problem, field_parents  # noqa: E402
from ..problems import problem as make_problem  # noqa: E402
from ..render.coloring import COLOR_MODES  # noqa: E402
from ..render.example import example_dataset  # noqa: E402
from ..render.layout import PRESET_LABELS, PRESETS, n_slots  # noqa: E402
from ..render.orientation import format_angles  # noqa: E402
from ..render.panels import PANEL_TYPES  # noqa: E402
from ..selections import SelectionError  # noqa: E402
from . import desktop  # noqa: E402
from . import widgets as W  # noqa: E402
from .about import citation_text, export_bibtex  # noqa: E402
from .canvas import FigureCanvas  # noqa: E402
from .jobs import JobControl  # noqa: E402
from .problemview import ProblemView  # noqa: E402
from .projectui import (SAVE_ATTEMPTS, DiskState, FileWatch, ask, change_bar, choose_project,  # noqa: E402
                        choose_save_path, fill_open_menu, other_data, renamed_workflow)
from .worker import call_soon  # noqa: E402

log = logging.getLogger(__name__)

APP_ID = "org.mdatk.MDATK"
TIME_UNITS = [("ps", "ps"), ("ns", "ns"), ("us", "µs"), ("frame", "frames")]
X_UNITS = ["ps", "ns", "us", "frame", "frame1"]
DEFAULT_GROUP = ""  # "(analysis default)" entry of the per-panel group choosers
PAGES = [  # (id, icon, side panel title)
    ("input", "document-open-symbolic", "Input & cleaning"),
    ("groups", "view-list-symbolic", "Residue groups"),
    ("analyses", "x-office-spreadsheet-symbolic", "Analyses"),
    ("structure", "applications-science-symbolic", "Structure view"),
    ("layout", "view-grid-symbolic", "Layout & panels"),
    ("outputs", "video-x-generic-symbolic", "Outputs"),
]
SIDE_WIDTH = 520  # fixed width of the settings side bar (pixels)
CHECK_DELAY_MS = 400  # the settings are revalidated this long after the last edit
# file chooser filters: (name, patterns); the dialogs list the extensions in the name
TOPOLOGY_FILES = ("Topology files", ["*.parm7", "*.prmtop", "*.top", "*.pdb", "*.psf", "*.gro"])
TRAJECTORY_FILES = ("Trajectory files", ["*.nc", "*.mdcrd", "*.crd", "*.trj", "*.dcd", "*.trr", "*.gro",
                                         "*.xtc", "*.pdb", "*.xyz", "*.binpos", "*.ncdf"])
REFERENCE_FILES = ("Structure files", ["*.pdb", "*.rst7", "*.ncrst", "*.inpcrd", "*.crd", "*.gro", "*.cif"])
DATA_FILES = ("Data files", ["*.dat", "*.csv", "*.tsv", "*.txt"])
MAP_FILES = ("MRC / CCP4 maps", ["*.mrc", "*.map", "*.ccp4", "*.map.gz", "*.mrc.gz"])


# option templates filled in when a slot switches to these panel types (edit them in Options)
SLOT_DEFAULT_OPTIONS = {
    "sequence": {"residues": ""},
    "secondary": {"source": "dynamic", "residues": "", "dotbracket": ""},
}


class GuiLogHandler(logging.Handler):
    def __init__(self, window: "MainWindow"):
        super().__init__(level=logging.WARNING)
        self.window = window

    def emit(self, record):
        msg = f"{record.levelname}: {record.getMessage()}"
        from .worker import call_soon
        call_soon(self.window.append_log, msg)


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, project: Optional[Project] = None,
                 project_path: Optional[str] = None):
        super().__init__(application=app, title=PROJECT_TITLE)
        self.set_default_size(1320, 920)
        self.project = project or Project()
        self.project_path = project_path
        self.binder = W.Binder()
        self.preview_data = None  # reference-frame trace once "Load real frame" ran
        self.last_video: Optional[str] = None
        self._syncing = False
        self._loading = False
        self.group_combos: List[Gtk.ComboBoxText] = []
        self._group_names: List[str] = []  # group names as last shown (a missing one was deleted)
        self._name_edit: Optional[dict] = None  # state of the group-name cell being edited
        self._chain_request = 0  # newest chain-group request (older answers are dropped)
        self._chains_offered = ""  # topology whose chain groups were last offered automatically
        self._chains_reading = 0  # the manual chain request whose "Reading …" line is shown
        self._topology_timer = 0
        self.slot_rows: List[dict] = []
        self.busy_widgets: List[Gtk.Widget] = []
        self.pages: Dict[str, Gtk.Widget] = {}
        self.sections: Dict[str, List[Gtk.Frame]] = {}
        # co-editing with RNAdo: the file as last read or written, and this window's group renames since
        self._disk: Optional[DiskState] = None
        self._mem_other: Dict = {}  # the analysis settings as last loaded or saved (edits differ from it)
        self._group_renames: List[Tuple[str, str, List[str]]] = []  # (old, new, group names before)
        self._offered: Optional[Dict] = None  # another program's analysis settings the change bar offers
        # live problems: badges on the page buttons, flags on the fields (revalidated after edits)
        self.problem_view = ProblemView(self.jump_to_problem)
        self.runtime_problems: List[Tuple[Problem, str]] = []  # a failed job's problem + what it depends on
        self._check_source = 0
        self._destroyed = False  # set on destroy: callbacks still pending must leave the window alone
        self._next_problem: Dict[str, int] = {}

        self._build_header()
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(outer)
        self.canvas = FigureCanvas(on_rotate=self._on_canvas_rotate, on_swap=self._on_canvas_swap,
                                   on_activate=self._on_canvas_activate, on_map_move=self._on_map_moved,
                                   on_zoom=self._on_canvas_zoom)
        builders = {"input": self._tab_input, "groups": self._tab_groups,
                    "analyses": self._tab_analyses, "structure": self._tab_visual,
                    "layout": self._tab_layout, "outputs": self._tab_outputs}
        side = self._build_side_panel(builders)
        outer.pack_start(self._build_toolbar(), False, False, 0)
        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        frame = Gtk.Frame()
        frame.set_shadow_type(Gtk.ShadowType.NONE)
        frame.add(self.canvas)
        body.pack_start(side, False, False, 0)
        body.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 0)
        body.pack_start(frame, True, True, 0)
        outer.pack_start(body, True, True, 0)
        self.log_expander = Gtk.Expander(label="Log (pipeline and cpptraj output)")
        self.log_expander.add(self._build_log())
        outer.pack_start(self.log_expander, False, False, 0)
        outer.pack_start(self._build_action_bar(), False, False, 0)
        self.change_bar = change_bar("", [("Reload them", 1), ("Keep mine", 2)], self._on_change_bar)
        outer.pack_start(self.change_bar, False, False, 0)
        outer.reorder_child(self.change_bar, 0)
        self.watch = FileWatch(self, self.check_disk)
        self._install_actions()
        self._register_fields()
        W.watch_tree(self.side_stack, self.schedule_check)
        self.load_project(self.project, project_path)
        self.connect("delete-event", self._on_close)
        self.connect("destroy", lambda *_: self._stop_timers())
        self.append_log(f"{PROJECT_TITLE} {__version__} ready. Everything runs locally and offline.")

    # ================================================================ chrome
    def _build_header(self):
        hb = Gtk.HeaderBar()
        hb.set_show_close_button(True)
        hb.set_title(PROJECT_TITLE)
        self.header = hb
        self.set_titlebar(hb)
        # Open: button + drop-down with recent projects, the demo project and Browse
        open_box = Gtk.Box()
        open_box.get_style_context().add_class("linked")
        open_box.pack_start(W.button("Open project", self.open_project, "Open a project file (Ctrl+O)",
                                     "document-open-symbolic"), False, False, 0)
        self.open_menu = Gtk.Menu()
        self.open_menu_button = Gtk.MenuButton()
        self.open_menu_button.set_image(Gtk.Image.new_from_icon_name("pan-down-symbolic", Gtk.IconSize.BUTTON))
        self.open_menu_button.set_tooltip_text("Recent projects, the demo project, or browse")
        self.open_menu_button.set_popup(self.open_menu)
        self.open_menu_button.connect("toggled", lambda b: b.get_active() and self._fill_open_menu())
        open_box.pack_start(self.open_menu_button, False, False, 0)
        hb.pack_start(open_box)
        hb.pack_start(W.button("Save", self.save_project, "Save the project (Ctrl+S)",
                               "document-save-symbolic"))
        hb.pack_start(W.button("Save as…", lambda: self.save_project(True),
                               "Save the project to a new file (Ctrl+Shift+S)"))
        wf_menu = Gtk.Menu()
        for text, cb, tip in (("Open in RNAdo…", self.open_in_rnado,
                               "RNAdo, MDATK's workflow planner: lay out the steps from sequence to MD "
                               "ensemble, write their scripts and run them (Ctrl+P; saves the project "
                               "first)"),
                              ("Script wizard: open in RNAdo…", lambda: self.open_in_rnado("wizard"),
                               "Write control files and scripts for one program, in RNAdo (Ctrl+Shift+W)"),
                              ("PDB clean-up: open in RNAdo…", lambda: self.open_in_rnado("pdb"),
                               "Repair, rename (D/L-RNA naming schemes), re-chain and renumber PDB files, "
                               "in RNAdo"),
                              ("Structure builder: open in RNAdo…", lambda: self.open_in_rnado("builder"),
                               "Ideal A-form helices (D / L), base mutations and rigid-body fits into a "
                               "density map, in RNAdo"),
                              (None, None, None),
                              ("Use the plan's files as inputs", self.use_plan_inputs,
                               "Take the topology and trajectories that the plan's MDATK analysis step reads "
                               "(Input & cleaning)")):
            if text is None:
                wf_menu.append(Gtk.SeparatorMenuItem())
                continue
            item = Gtk.MenuItem(label=text)
            item.set_tooltip_text(tip)
            item.connect("activate", lambda _w, c=cb: c())
            wf_menu.append(item)
        wf_menu.show_all()
        wf = Gtk.MenuButton()
        wf_box = Gtk.Box(spacing=4)
        wf_box.pack_start(Gtk.Image.new_from_icon_name("network-workgroup-symbolic", Gtk.IconSize.BUTTON),
                          False, False, 0)
        wf_box.pack_start(Gtk.Label(label="RNAdo"), False, False, 0)
        wf_box.pack_start(Gtk.Image.new_from_icon_name("pan-down-symbolic", Gtk.IconSize.BUTTON),
                          False, False, 0)
        wf.add(wf_box)
        wf.set_tooltip_text("RNAdo, MDATK's workflow planner (a program of its own): the plan, the script "
                            "wizard (Rosetta FARFAR2, SimRNA, RNAMake, Amber, GROMACS, VMD), PDB clean-up "
                            "and the structure builder")
        wf.set_popup(wf_menu)
        self.workflow_button = wf
        hb.pack_start(wf)
        menu = Gtk.Menu()
        for text, cb in (("New project (defaults)", self.new_project),
                         ("Create demo system…", self.create_demo),
                         ("Project log…", self.show_project_log),
                         ("Export citations (BibTeX)…", self.export_citations),
                         ("About and citations", self.show_about)):
            item = Gtk.MenuItem(label=text)
            item.connect("activate", lambda _w, c=cb: c())
            menu.append(item)
        menu.show_all()
        mb = Gtk.MenuButton()
        mb.set_image(Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON))
        mb.set_popup(menu)
        hb.pack_end(mb)

    def _fill_open_menu(self) -> None:
        fill_open_menu(self.open_menu, self.open_path,
                       [("Demo project…", self.create_demo,
                         "Write the synthetic aptamer/target demo system into a folder and open it"),
                        ("Browse…", self.open_project, "Choose a project file")])

    def _build_toolbar(self):
        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4, margin=4)
        pages = Gtk.Box()
        pages.get_style_context().add_class("linked")
        self.page_buttons: Dict[str, Gtk.RadioButton] = {}
        group = None
        for page, icon, title in PAGES:
            btn = Gtk.RadioButton.new_from_widget(group)
            group = group or btn
            btn.set_mode(False)  # a toggle button, not a radio indicator
            image, badge = W.badge_overlay(Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.LARGE_TOOLBAR))
            btn.set_image(image)  # the badge counts the page's problems
            btn.set_always_show_image(True)
            btn.set_tooltip_text(title)
            btn.get_accessible().set_name(title)
            btn.connect("toggled", lambda b, p=page: b.get_active() and self._show_page(p))
            btn.connect("button-release-event", self._on_page_button_release, page)
            self.problem_view.add_page_badge(page, btn, badge)
            self.page_buttons[page] = btn
            pages.pack_start(btn, False, False, 0)
        bar.pack_start(pages, False, False, 0)
        bar.pack_end(W.button("Refresh", self.refresh_canvas,
                              "Redraw the layout preview with the current settings",
                              "view-refresh-symbolic"), False, False, 0)
        lp = W.button("Load real frame", self.load_preview,
                      "Show the real reference-frame structure in the preview "
                      "(cleaned files are used when cached)", "document-open-symbolic")
        self.busy_widgets.append(lp)
        bar.pack_end(lp, False, False, 0)
        bar.pack_end(W.hint("Drag panels to swap them \u00b7 drag the structure to rotate it "
                            "\u00b7 double-click a panel to edit it"), False, False, 8)
        return bar

    def _build_side_panel(self, builders):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.set_size_request(SIDE_WIDTH, -1)
        box.set_hexpand(False)
        head = Gtk.Box(spacing=6, margin=6)
        self.side_title = W.label("", markup=True)
        head.pack_start(self.side_title, True, True, 0)
        head.pack_end(self.problem_view.button, False, False, 0)  # every problem; click one to go there
        box.pack_start(head, False, False, 0)
        box.pack_start(Gtk.Separator(), False, False, 0)
        self.side_stack = Gtk.Stack()
        self.side_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.side_stack.set_hhomogeneous(True)
        for page, _icon, title in PAGES:
            widget = builders[page]()
            if isinstance(widget, Gtk.ScrolledWindow):  # fixed side bar width: rows fit it, never scroll sideways
                widget.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            self.pages[page] = widget
            self.sections[page] = self._find_frames(widget)
            self.side_stack.add_titled(widget, page, title)
        box.pack_start(self.side_stack, True, True, 0)
        self.side_panel = box
        self._show_page("input")
        return box

    @staticmethod
    def _find_frames(widget) -> List[Gtk.Frame]:
        found: List[Gtk.Frame] = []

        def walk(w, depth=0):
            if isinstance(w, Gtk.Frame) and w.get_label_widget() is not None:
                found.append(w)
                return
            if depth < 6 and isinstance(w, Gtk.Container):
                for child in w.get_children():
                    walk(child, depth + 1)

        walk(widget)
        return found

    @staticmethod
    def _frame_title(frame: Gtk.Frame) -> str:
        lab = frame.get_label_widget()
        return lab.get_text() if isinstance(lab, Gtk.Label) else "Section"

    def _show_page(self, page: str) -> None:
        title = dict((p, t) for p, _i, t in PAGES)[page]
        self.side_title.set_markup(f"<b>{GLib.markup_escape_text(title)}</b>")
        self.side_stack.set_visible_child_name(page)
        btn = getattr(self, "page_buttons", {}).get(page)
        if btn is not None and not btn.get_active():
            btn.set_active(True)

    def open_section(self, page: str, frame: Optional[Gtk.Frame] = None) -> None:
        """Show ``page`` in the side bar and scroll to ``frame``."""
        self._show_page(page)
        if frame is not None:
            GLib.timeout_add(60, self._scroll_to, page, frame)

    def _scroll_to(self, page, frame):
        sw = self.pages[page]
        if isinstance(sw, Gtk.ScrolledWindow):
            coords = frame.translate_coordinates(sw.get_child(), 0, 0)
            if coords:
                sw.get_vadjustment().set_value(max(0, coords[1] - 6))
        return False

    # ---------------------------------------------------------- problems
    def _register_fields(self) -> None:
        """Field paths -> widgets, so that each problem flags its field (see :mod:`.problemview`)."""
        reg = self.problem_view.registry
        for b in self.binder.bindings:
            reg.add(b.path, b.widget)
        reg.add_list("groups", self.group_editor, {"name": 0, "residues": 1, "selection": 2, "color": 3},
                     keep=lambda row: bool(row[0].strip()))
        reg.add_list("analyses.rmsd.targets", self.rmsd_editor,
                     {"label": 0, "selection": 1, "fit_selection": 2},
                     keep=lambda row: bool(row[1].strip()))
        reg.add_list("inputs.external", self.ext_editor,
                     {"path": 0, "label": 1, "x_column": 2, "x_unit": 3, "columns": 4},
                     keep=lambda row: bool(row[0].strip()))
        reg.add_provider(self._row_fields)

    def _row_fields(self) -> Dict[str, List[Gtk.Widget]]:
        """Fields of the rebuilt rows: representations and panel slots."""
        out: Dict[str, List[Gtk.Widget]] = {}
        for k, row in enumerate(self.rep_rows):
            base = f"visual.representations.{k}"
            out.update({base: [row["groups"]], f"{base}.groups": [row["groups"]],
                        f"{base}.style": [row["style"]], f"{base}.color": [row["color"]],
                        f"{base}.size": [row["size"], row["opacity"]]})
        for k, row in enumerate(self.slot_rows):
            base = f"layout.panels.{k}"
            out.update({base: [row["type"]], f"{base}.type": [row["type"]], f"{base}.series": [row["more"]],
                        f"{base}.groups": [row["ga"], row["gb"]], f"{base}.title": [row["title"]],
                        f"{base}.options": [row["opts"]]})
        return out

    def schedule_check(self) -> None:
        """Revalidate shortly after an edit (one check for a burst of changes)."""
        if self._loading:
            return
        if self._check_source:
            GLib.source_remove(self._check_source)
        self._check_source = GLib.timeout_add(CHECK_DELAY_MS, self._check_due)

    def _stop_timers(self) -> None:
        """A destroyed window must not be checked or read into by timers still pending."""
        self._destroyed = True
        self._stop_topology_timer()
        if self._check_source:
            GLib.source_remove(self._check_source)
            self._check_source = 0

    def _check_due(self) -> bool:
        self._check_source = 0
        self.check_problems()
        return False

    def _side_bar_project(self):
        """The project as the side bar shows it, and the values it could not store as problems."""
        project = self.project.copy()
        project.workflow = self.project.workflow
        errors = self.binder.store_fields(project)
        return project, [make_problem(path, message, project) for path, message in errors]

    def check_problems(self) -> List[Problem]:
        """Validate the settings as they stand and show every problem on its page and field; a failed
        job's problem stays until its setting or the residue groups change."""
        if self._check_source:
            GLib.source_remove(self._check_source)
            self._check_source = 0
        try:
            project, found = self._side_bar_project()
            found += project.problems()
        except Exception as exc:  # noqa: BLE001 - checking must never break the window
            log.warning("could not check the settings: %s", exc)
            return self.problem_view.problems
        self.runtime_problems = [(p, snap) for p, snap in self.runtime_problems
                                 if snap == self._snapshot(project, p.field)]
        known = {(p.field, p.message) for p in found}
        found += [p for p, _snap in self.runtime_problems if (p.field, p.message) not in known]
        self.problem_view.show(found)
        return found

    @staticmethod
    def _snapshot(project: Project, field: str) -> str:
        """What a run-time problem depends on: its setting and the residue groups."""
        value = None
        for path in field_parents(field):
            try:
                value = W.get_path(project, path)
                break
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        return stable_hash([value, project.groups])

    def _note_failure(self, exc: BaseException) -> None:
        """A job failed on a setting: flag it like the other problems."""
        if isinstance(exc, SelectionError) and exc.field:
            try:
                project, _errors = self._side_bar_project()
            except Exception:  # noqa: BLE001
                return
            self.runtime_problems.append((exc.problem(), self._snapshot(project, exc.field)))
        self.check_problems()

    def jump_to_problem(self, problem: Problem, focus: bool = True) -> None:
        """Show a problem's page and bring its field into view (and focus it)."""
        if problem.tab in self.pages:
            self._show_page(problem.tab)
        GLib.timeout_add(60, self._reveal_problem, problem, focus)

    def _reveal_problem(self, problem: Problem, focus: bool) -> bool:
        widget = self.problem_view.reveal(problem, focus)
        if widget is not None and problem.tab in self.pages:
            self._scroll_to(problem.tab, widget)
        return False

    def _on_page_button_release(self, _button, _event, page: str) -> bool:
        """A click on a page button with a badge also shows the page's next problem field."""
        mine = self.problem_view.on_page(page)
        if mine:
            k = self._next_problem.get(page, 0) % len(mine)
            self._next_problem[page] = k + 1
            GLib.idle_add(lambda: self.jump_to_problem(mine[k], focus=False) and False)
        return False  # the button still toggles

    # ----------------------------------------------------------- canvas
    def refresh_canvas(self) -> None:
        """Redraw the WYSIWYG figure with example data (never computes anything)."""
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.append_log(f"Preview not updated: {exc}")
            return
        try:
            ds = example_dataset(project, self.preview_data)
        except Exception as exc:  # noqa: BLE001 - a bad layout must not break the window
            self.append_log(f"Preview not updated: {exc}")
            return
        mark = ("Example data \u2013 layout preview (structure: reference frame)"
                if self.preview_data is not None else "Example data \u2013 layout preview")
        self.canvas.set_content(project, ds, mark)
        warnings_now = layout_warnings(project)
        if warnings_now and warnings_now != getattr(self, "_layout_warned", None):
            for w in warnings_now:
                self.append_log(f"Layout: {w}")
        self._layout_warned = warnings_now

    def _on_canvas_rotate(self, angles):
        self._syncing = True
        try:
            for sc, a in zip(self.rot_scales, angles):
                sc.set_value(float(a))
        finally:
            self._syncing = False
        self.angle_label.set_text(format_angles(angles))

    def _on_canvas_zoom(self, factor: float) -> None:
        self.zoom_spin.set_value(min(50.0, max(0.05, self.zoom_spin.get_value() * factor)))

    def _on_canvas_swap(self, i: int, j: int) -> None:
        specs = self._read_slots()
        if max(i, j) >= len(specs):
            return
        specs[i], specs[j] = specs[j], specs[i]
        self._rebuild_slots(specs)
        self.append_log(f"Swapped panels {i + 1} and {j + 1}")
        self.refresh_canvas()

    def _on_canvas_activate(self, slot: int) -> None:
        frame = self.slot_rows[slot]["frame"] if slot < len(self.slot_rows) else None
        self.open_section("layout", frame)

    def _build_log(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6, margin=4)
        head.pack_start(W.label("<b>Log</b> (pipeline and cpptraj output)", markup=True), False, False, 4)
        head.pack_end(W.button("Clear", lambda: self.log_buffer.set_text("")), False, False, 0)
        box.pack_start(head, False, False, 0)
        self.log_view = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log_buffer = self.log_view.get_buffer()
        self.log_end = self.log_buffer.create_mark("end", self.log_buffer.get_end_iter(), False)
        sw = Gtk.ScrolledWindow()
        sw.set_min_content_height(120)
        sw.add(self.log_view)
        box.pack_start(sw, True, True, 0)
        return box

    def _build_action_bar(self):
        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, margin=8)
        self.render_button = W.button("Render", self.render, "Run cleaning, analyses and all "
                                      "enabled outputs (Ctrl+R)", "media-playback-start-symbolic")
        self.render_button.get_style_context().add_class("suggested-action")
        self.cancel_button = W.button("Cancel", self.cancel, "Stop the running job",
                                      "process-stop-symbolic")
        self.cancel_button.set_sensitive(False)
        self.progress = Gtk.ProgressBar(show_text=True)
        self.progress.set_text("Idle")
        self.progress.set_hexpand(True)
        self.progress.set_valign(Gtk.Align.CENTER)
        bar.pack_start(self.render_button, False, False, 0)
        bar.pack_start(self.cancel_button, False, False, 0)
        bar.pack_start(self.progress, True, True, 0)
        bar.pack_end(W.button("Open video", self.open_video, "Play the most recent video",
                              "video-x-generic-symbolic"), False, False, 0)
        bar.pack_end(W.button("Open output folder", self.open_output_dir,
                              "Show the output folder in the file manager", "folder-open-symbolic"),
                     False, False, 0)
        self.busy_widgets.append(self.render_button)
        self.jobs = JobControl(self.progress, self.cancel_button, self.append_log,
                               lambda text, detail: self.error(text, detail), self.busy_widgets)
        return bar

    @property
    def worker(self):
        """The running background job (None when idle)."""
        return self.jobs.worker

    def _install_actions(self):
        app = self.get_application()
        for name, cb, accels in (("open", self.open_project, ["<Control>o"]),
                                 ("save", self.save_project, ["<Control>s"]),
                                 ("saveas", lambda: self.save_project(True), ["<Control><Shift>s"]),
                                 ("render", self.render, ["<Control>r"]),
                                 ("rnado", self.open_in_rnado, ["<Control>p"]),
                                 ("wizard", lambda: self.open_in_rnado("wizard"), ["<Control><Shift>w"])):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, c=cb: c())
            self.add_action(action)
            if app is not None:
                app.set_accels_for_action(f"win.{name}", accels)

    # ================================================================== tabs
    def _tab_input(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        notes = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        notes.set_size_request(-1, 54)
        swn = Gtk.ScrolledWindow()
        swn.set_shadow_type(Gtk.ShadowType.IN)
        swn.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        swn.add(notes)
        b.text_view("notes", notes)
        nf = W.Form()
        nf.add_wide(swn)
        nf.add_wide(W.hint("Free text saved in the project file (aims, decisions, to-dos). The steps you "
                           "run are logged there too (menu \u2192 Project log)."))
        box.pack_start(W.frame("Project notes", nf), False, False, 0)

        files = W.Form()
        top = W.file_chooser_button("Select topology", patterns=[TOPOLOGY_FILES])
        b.path_entry("inputs.topology", files.add("Topology", top,
                     "Amber .parm7 (preferred), cleaned PDB or another MDAnalysis topology"))
        self.topology_entry = top.entry
        top.entry.connect("changed", lambda *_: self._on_topology_changed())
        self.traj_editor = W.ListEditor([("Trajectory file(s), in order", 300)], height=70)
        self.traj_editor.buttons.pack_start(
            W.button("Browse…", self._browse_trajectories, icon="document-open-symbolic"),
            False, False, 0)
        files.add("Trajectories", self.traj_editor,
                  "Amber NetCDF .nc (preferred), DCD, XTC, ... (several files are concatenated)")
        b._add("inputs.trajectories", self.traj_editor,
               lambda: [r[0].strip() for r in self.traj_editor.rows() if r[0].strip()],
               lambda v: self.traj_editor.set_rows([[x] for x in v]))
        plan_files = W.button("Use the plan's files", self.use_plan_inputs,
                              "Take the topology and trajectories that the plan's MDATK analysis step reads "
                              "(the plan RNAdo saved in this project file)", "network-workgroup-symbolic")
        plan_files.set_halign(Gtk.Align.START)
        files.add_wide(plan_files)
        box.pack_start(W.frame("Input files", files), False, False, 0)

        fr = W.Form()
        start = b.spin_int("inputs.start", W.spin(0, 10 ** 9, 1))
        stop = b.optional_int("inputs.stop", W.entry("last", width=5))
        step = b.spin_int("inputs.step", W.spin(1, 10 ** 6, 1))
        fr.add_row("Start / stop / step", [start, stop, step],
                   "0-based frames of the (cleaned) trajectory; stop is exclusive (blank = last)")
        fr.add("Time step override (ps)", b.optional_float("inputs.dt_ps", W.entry("from file", width=10)),
               "Leave blank to use the trajectory's time stamps")
        fr.add("Display time unit", b.combo("inputs.time_unit", W.combo(TIME_UNITS)))
        self.ext_editor = W.ListEditor([("Path", 150), ("Label", 60), ("x col", 50),
                                        ("x unit", 50), ("Columns", 70)], height=60,
                                       defaults=["", "", "", "ps", ""], combos={3: X_UNITS})
        self.ext_editor.buttons.pack_start(W.button("Browse…", self._browse_external),
                                           False, False, 0)
        fr.add("External data (CSV/text)", self.ext_editor,
               "Extra traces (e.g. cpptraj .dat files) interpolated onto the analysed frames")
        b._add("inputs.external", self.ext_editor, self._get_external, self._set_external)
        box.pack_start(W.frame("Frames, time and external data", fr), False, False, 0)

        cl = W.Form()
        self.clean_enable = b.check("cleaning.enabled", W.check(
            "Strip and clean the trajectory with cpptraj before analysis"))
        cl.add_wide(self.clean_enable)
        cl.add("cpptraj executable", b.path_entry("cleaning.cpptraj_path",
               W.file_chooser_button("Select cpptraj")), "Name on PATH or absolute path")
        self.conda_entry = b.entry("cleaning.conda_env", W.entry("e.g. AmberTools26 (blank: cpptraj on PATH)",
                                                                 width=18))
        cl.add_row("Conda environment", [self.conda_entry,
                                         W.button("Detect", self.detect_conda,
                                                  "Find environments that contain bin/cpptraj")],
                   "Activated for every cpptraj call (conda run -n <env> cpptraj); a name or a path")
        cl.add("conda executable", b.path_entry("cleaning.conda_exe",
               W.file_chooser_button("Select conda / mamba / micromamba")),
               "Blank: found automatically (conda, mamba or micromamba)")
        cl.add("Solvent mask", b.entry("cleaning.solvent_mask", W.entry()))
        cl.add("Ion mask", b.entry("cleaning.ion_mask", W.entry()), "Mg2+ is kept by default")
        cl.add("Extra strip mask", b.entry("cleaning.extra_strip_mask", W.entry("optional")))
        cl.add_row("Imaging", [b.check("cleaning.autoimage", W.check("autoimage, anchor")),
                               b.entry("cleaning.autoimage_anchor",
                                       W.entry("first molecule", width=14))])
        cl.add_row("Fitting", [b.check("cleaning.fit", W.check("rms fit, mask")),
                               b.entry("cleaning.fit_mask", W.entry(width=14))])
        cl.add_row("Fit reference", [b.combo("cleaning.fit_reference", W.combo(
                                         [("first", "first frame"), ("frame", "frame #"),
                                          ("file", "file")])),
                                     b.spin_int("cleaning.reference_frame", W.spin(1, 10 ** 9, 1))])
        cl.add("Reference file", b.path_entry("cleaning.reference_file",
               W.file_chooser_button("Select reference structure", patterns=[REFERENCE_FILES])))
        cl.add_wide(b.check("cleaning.remove_box_after_fit", W.check(
            "Remove box information after fitting (a rotated box cannot be stored in NetCDF)")))
        cl.add_row("trajin start / stop / offset",
                   [b.spin_int("cleaning.start", W.spin(1, 10 ** 9, 1)),
                    b.optional_int("cleaning.stop", W.entry("last", width=5)),
                    b.spin_int("cleaning.offset", W.spin(1, 10 ** 6, 1))],
                   "1-based cpptraj trajin arguments")
        extra = Gtk.TextView(monospace=True)
        extra.set_size_request(-1, 50)
        swx = Gtk.ScrolledWindow()
        swx.set_shadow_type(Gtk.ShadowType.IN)
        swx.add(extra)
        b.text_view("cleaning.extra_commands", extra)
        cl.add("Extra cpptraj commands", swx, "Advanced: inserted before trajout")
        btns = Gtk.Box(spacing=6)
        btns.pack_start(W.button("Run cleaning now", self.run_cleaning, icon="system-run-symbolic"),
                        False, False, 0)
        btns.pack_start(W.button("Inspect system", self.inspect_system,
                                 "Print atom/residue/frame counts and check residue groups"),
                        False, False, 0)
        cl.add_wide(btns)
        box.pack_start(W.frame("Trajectory cleaning (cpptraj)", cl), False, False, 0)
        return W.scrolled(box)

    def _get_external(self):
        out = []
        for path, lab, xcol, xunit, cols in self.ext_editor.rows():
            if path.strip():
                out.append(ExternalSeries(path=path.strip(), label=lab.strip(), x_column=xcol.strip(),
                                          x_unit=xunit.strip() or "ps",
                                          columns=[c.strip() for c in cols.split(",") if c.strip()]))
        return out

    def _set_external(self, specs):
        self.ext_editor.set_rows([[s.path, s.label, s.x_column, s.x_unit, ", ".join(s.columns)]
                                  for s in specs])

    def _browse_trajectories(self):
        dlg = Gtk.FileChooserDialog(title="Add trajectory files", transient_for=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        dlg.set_select_multiple(True)
        W.add_filters(dlg, [TRAJECTORY_FILES])
        if dlg.run() == Gtk.ResponseType.OK:
            for f in dlg.get_filenames():
                self.traj_editor.add_row([f])
        dlg.destroy()

    def _browse_external(self):
        dlg = Gtk.FileChooserDialog(title="Add external data file", transient_for=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        W.add_filters(dlg, [DATA_FILES])
        if dlg.run() == Gtk.ResponseType.OK:
            path = dlg.get_filename()
            self.ext_editor.add_row([path, os.path.splitext(os.path.basename(path))[0], "", "ps", ""])
        dlg.destroy()

    # ----------------------------------------------------------------- groups
    def _tab_groups(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        form = W.Form()
        self.group_editor = W.ListEditor([("Name", 80), ("Residues", 110), ("Selection (overrides)", 140),
                                          ("Colour", 70)], height=180,
                                         defaults=["NewGroup", "", "", "#6250d6"])
        swatch = Gtk.CellRendererPixbuf()  # a pixbuf keeps its colour when the row is selected
        col = Gtk.TreeViewColumn("", swatch)
        col.set_cell_data_func(swatch, self._swatch)
        col.set_min_width(34)
        self.group_editor.view.append_column(col)
        self._swatches: Dict[str, GdkPixbuf.Pixbuf] = {}
        self.group_editor.buttons.pack_start(W.button("Pick colour…", self._pick_group_color,
                                                      icon="applications-graphics-symbolic"),
                                             False, False, 0)
        self.group_editor.on_change = self._groups_changed
        self.group_editor.on_edit = self._on_group_cell_edit
        name_cell = self.group_editor.view.get_column(0).get_cells()[0]
        name_cell.connect("editing-started", self._on_group_name_editing)
        form.add_wide(self.group_editor)
        row = Gtk.Box(spacing=6)
        row.pack_start(W.button("Groups from chains", self.groups_from_chains,
                                "Add one group per chain of the topology (water and ions left out); "
                                "existing groups are kept", "list-add-symbolic"), False, False, 0)
        form.add_wide(row)
        self.group_status = W.label("", wrap=True)
        self.group_status.set_selectable(True)
        form.add_wide(self.group_status)
        form.add_wide(W.hint(
            "Residues accept ranges such as “1-40, 45, 50-60” or chain/segment-qualified "
            "“A:1-40”. A selection is a raw MDAnalysis selection string. Anywhere a selection is "
            "expected you can refer to a group as group:<Name> (e.g. “group:Aptamer and name P”). "
            "Group names must not contain spaces. Renaming a group (Enter or leaving the cell) "
            "renames it everywhere the project uses it. Choosing a topology fills empty or untouched "
            "default groups with one group per chain."))
        self.binder._add("groups", self.group_editor, self._get_groups, self._set_groups)
        box.pack_start(W.frame("Residue groups", form), False, False, 0)
        return W.scrolled(box)

    def _swatch(self, _col, cell, model, it, _data):
        cell.set_property("pixbuf", self.swatch_pixbuf(model[it][3] or ""))

    def swatch_pixbuf(self, color: str) -> Optional[GdkPixbuf.Pixbuf]:
        rgba = Gdk.RGBA()
        if not rgba.parse(color):
            return None
        hexa = W.rgba_to_hex(rgba)
        if hexa not in self._swatches:
            pix = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, True, 8, 26, 16)
            pix.fill(0xFFFFFFFF)  # light frame keeps the swatch distinct on the selection colour
            pix.new_subpixbuf(2, 2, 22, 12).fill(int(hexa[1:], 16) << 8 | 0xFF)
            self._swatches[hexa] = pix
        return self._swatches[hexa]

    def _pick_group_color(self):
        k = self.group_editor.selected_index()
        if k is None:
            self.error("Select a group first.")
            return
        dlg = Gtk.ColorChooserDialog(title="Group colour", transient_for=self)
        dlg.set_rgba(W.hex_to_rgba(self.group_editor.store[k][3]))
        if dlg.run() == Gtk.ResponseType.OK:
            self.group_editor.store[k][3] = W.rgba_to_hex(dlg.get_rgba())
            self._groups_changed()
        dlg.destroy()

    def _get_groups(self):
        return [ResidueGroup(name=n.strip(), residues=r.strip(), selection=s.strip(),
                             color=c.strip() or "#2a78d6")
                for n, r, s, c in self.group_editor.rows() if n.strip()]

    def _set_groups(self, groups):
        self.group_editor.set_rows([[g.name, g.residues, g.selection, g.color] for g in groups])
        self._groups_changed()

    def _groups_changed(self):
        names = [r[0].strip() for r in self.group_editor.rows() if r[0].strip()]
        gone = [n for n in self._group_names if n not in names]
        self._group_names = names
        if gone and not self._loading:
            self._report_removed_groups(gone)
        for c in self.group_combos:
            W.set_combo_entry_items(c, names)
        if self.slot_rows and not self._loading:
            try:  # per-panel group choosers list the current group names
                self._rebuild_slots()
            except ValueError:
                pass

    def _group_combo(self, path):
        c = W.combo_entry()
        self.group_combos.append(c)
        return self.binder.combo_text(path, c)

    def _group_message(self, text: str, error: bool = False, detail: str = "") -> None:
        """Status line under the group table (logged too; ``detail`` goes to the log and the tooltip)."""
        self.group_status.set_text(text)
        self.group_status.set_tooltip_text(detail or None)
        ctx = self.group_status.get_style_context()
        (ctx.add_class if error else ctx.remove_class)("error")
        self.append_log(("WARNING: " if error else "") + text + (f"\n  {detail}" if detail else ""))

    def _group_name_problem(self, name: str, k: int) -> str:
        others = [r[0].strip() for i, r in enumerate(self.group_editor.store) if i != k]
        return group_name_problem(name.strip(), others)

    def _flag_group_name(self, entry: Gtk.Entry, k: int) -> None:
        """Mark the name being typed when it cannot name a group (red, with the reason)."""
        problem = self._group_name_problem(entry.get_text(), k)
        ctx = entry.get_style_context()
        (ctx.add_class if problem else ctx.remove_class)("error")
        icon = Gtk.EntryIconPosition.SECONDARY
        entry.set_icon_from_icon_name(icon, "dialog-error-symbolic" if problem else None)
        entry.set_icon_tooltip_text(icon, problem or None)

    def _on_group_name_editing(self, _cell, editable, path) -> None:
        """A name cell opened: flag invalid names while typing.  Leaving the cell commits the name
        (GTK 3 itself would drop the edit), Escape cancels it."""
        k = int(path)
        state = {"done": False}
        self._name_edit = state
        before = self.group_editor.store[k][0].strip()

        def finish(commit: bool) -> bool:
            if not state["done"]:
                state["done"] = True
                if commit:
                    self.commit_group_name(k, editable.get_text(), before)
            return False

        editable.connect("changed", lambda e: self._flag_group_name(e, k))
        editable.connect("key-press-event", lambda _e, ev: ev.keyval == Gdk.KEY_Escape and finish(False))
        editable.connect("focus-out-event", lambda *_: finish(True))
        self._flag_group_name(editable, k)

    def _on_group_cell_edit(self, k: int, col: int, _old: str, text: str) -> bool:
        if col != 0:
            return False
        if self._name_edit is not None:
            self._name_edit["done"] = True
        self.commit_group_name(k, text)
        return True

    def commit_group_name(self, k: int, text: str, before: Optional[str] = None) -> bool:
        """Rename the group of row ``k`` (the name it had before the edit) to ``text``, and every
        reference to it in the project; empty, invalid and taken names are refused.  ``before``: the
        name when the edit began (nothing happens if the table changed since).  True if renamed."""
        store = self.group_editor.store
        if not 0 <= k < len(store) or (before is not None and store[k][0].strip() != before):
            return False
        old, new = store[k][0].strip(), text.strip()
        if new == old:
            return False
        problem = self._group_name_problem(new, k)
        if problem:
            self._group_message(f"Group {old or '(unnamed)'} not renamed: {problem}", error=True)
            return False
        if not old or [r[0].strip() for r in store].count(old) > 1:
            store[k][0] = new  # not a group yet, or one of several of that name: no references to follow
            self._groups_changed()
            self._group_message(f"Renamed {old} → {new}" if old else f"Named the new group {new}")
            return True
        try:
            project = self.collect(validate=False)
            before = project.copy()
            names = project.group_names()
            refs = project.rename_group(old, new)
            self._group_renames.append((old, new, names))  # replayed on a plan RNAdo saves meanwhile
        except ValueError as exc:
            self._group_message(f"Group {old} not renamed; fix these settings first: {exc}", error=True)
            return False
        refs = self._adopt_project(project, refs, before, rows_in_place=True)
        self._group_message(f"Renamed {old} → {new} in {len(refs)} place{'' if len(refs) == 1 else 's'}",
                            detail=", ".join(refs))
        return True

    def _adopt_project(self, project: Project, changed: List[str], before: Project,
                       rows_in_place: bool = False) -> List[str]:
        """Make ``project`` (the settings collected from the widgets as ``before``, then changed at the
        locations ``changed``) current and reload the widgets of those settings only.  ``rows_in_place``
        updates the group rows without rebuilding the table, where a name cell may still be open.
        Automatic panel titles follow their data: a title that was its data's label, or the automatic
        or default title of a panel whose data keys changed; returns ``changed`` with those titles."""
        changed = list(changed)
        for k, (row, spec) in enumerate(zip(self.slot_rows, project.layout.panels)):
            if not (spec.series and row["auto"] and spec.title == row["auto"]):
                continue
            # a default title ("RMSD") is not its data's label: it stays while the panel's keys do
            moved = any(c.startswith(f"layout.panels[{k}].series") for c in changed)
            if moved or spec.title == key_label(spec.series[0], before):
                title = key_label(spec.series[0], project)
                if title != spec.title:
                    spec.title = title
                    changed.append(f"layout.panels[{k}].title")
        self.project = project
        loading, self._loading = self._loading, True
        try:
            if rows_in_place:
                named = [row for row in self.group_editor.store if row[0].strip()]
                for row, g in zip(named, project.groups):
                    if row[0].strip() != g.name:
                        row[0] = g.name
                    if row[2].strip() != g.selection:
                        row[2] = g.selection
            for b in self.binder.bindings:
                if rows_in_place and b.path == "groups":
                    continue
                if any(c == b.path or c.startswith((b.path + ".", b.path + "[")) for c in changed):
                    b.set(W.get_path(project, b.path))
        finally:
            self._loading = loading
        self._group_names = [r[0].strip() for r in self.group_editor.rows() if r[0].strip()]
        self._groups_changed()
        self.refresh_canvas()
        return changed

    def _report_removed_groups(self, names: List[str]) -> None:
        """Say where deleted groups are still used: those settings now name a group that is gone."""
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project
        uses = [(n, project.group_references(n)) for n in names]
        uses = [(n, refs) for n, refs in uses if refs]
        if uses:
            self._group_message(
                "Removed group still in use, change these settings: " +
                "; ".join(f"{n} in {', '.join(refs)}" for n, refs in uses), error=True)

    def groups_from_chains(self, auto: bool = False) -> None:
        """'Groups from chains': add a group for each chain of the topology that no group selects yet
        (read in the background); existing groups are kept.  ``auto`` (a topology was just chosen)
        instead fills empty or untouched default groups, and only names the chains left without one."""
        path = self.topology_entry.get_text().strip()
        if not os.path.isfile(path):
            if not auto:
                self._group_message("Choose a topology file on the Input page first.", error=True)
            return
        self._chain_request += 1
        request, existing = self._chain_request, self._get_groups()
        if not auto:
            self._chains_reading = request
            self.group_status.set_text(self._reading_text(path))

        def work():
            try:
                chains, error = chaingroups.topology_chain_groups(path, existing), None
            except Exception as exc:  # noqa: BLE001 - every reader problem is reported
                chains, error = [], exc
            call_soon(self._chains_ready, request, path, auto, chains, error)

        threading.Thread(target=work, name="mdatk-chains", daemon=True).start()

    @staticmethod
    def _reading_text(path: str) -> str:
        return f"Reading the chains of {os.path.basename(path)} …"

    def _chains_ready(self, request: int, path: str, auto: bool, chains, error) -> None:
        if self._topology_timer < 0:
            return  # the window is gone
        name = os.path.basename(path)
        # superseded by a newer request or by loading a project, or the topology changed meanwhile
        if request != self._chain_request or path != self.topology_entry.get_text().strip():
            if request == self._chains_reading and self.group_status.get_text() == self._reading_text(path):
                self._group_message(f"Groups not changed: the topology changed while the chains of {name} "
                                    "were read")
            return
        if error is not None:
            self._group_message(f"Could not read the chains of {name}: {error}", error=True)
            return
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            if not auto:
                self._group_message(f"Groups not changed; fix these settings first: {exc}", error=True)
            return
        before = project.copy()
        merge = chaingroups.merge_chain_groups(project, chains, auto=auto)
        names = before.group_names()
        for old, new in merge.replaced.items():  # defaults renamed after chains: replayed on RNAdo's plan
            self._group_renames.append((old, new, list(names)))
            names = [new if n == old else n for n in names]
        if merge.changed:
            merge.references = [c for c in self._adopt_project(project, merge.references + ["groups"], before)
                                if c != "groups"]
        if merge.changed or merge.pending or not auto:
            self._group_message(merge.summary(name), detail=", ".join(merge.references))

    def _on_topology_changed(self) -> None:
        """A topology was typed or chosen: offer its chain groups once the name stops changing."""
        if self._loading or self._topology_timer < 0:
            return
        if self._topology_timer:
            GLib.source_remove(self._topology_timer)
        self._topology_timer = GLib.timeout_add(500, self._topology_settled)

    def _topology_settled(self) -> bool:
        self._topology_timer = 0
        path = self.topology_entry.get_text().strip()
        if path != self._chains_offered and os.path.isfile(path):
            self._chains_offered = path
            self.groups_from_chains(auto=True)
        return False

    def _stop_topology_timer(self) -> None:
        if self._topology_timer > 0:
            GLib.source_remove(self._topology_timer)
        self._topology_timer = -1  # the window is gone: no more chain look-ups

    def _forget_chain_requests(self) -> None:
        """Another project was loaded: chain groups being read or about to be offered belong to the
        previous one, and so does the status line.  Loading never fills groups, so the loaded topology
        counts as offered (choosing another one offers its chains)."""
        self._chain_request += 1
        if self._topology_timer > 0:
            GLib.source_remove(self._topology_timer)
            self._topology_timer = 0
        self._chains_offered = self.topology_entry.get_text().strip()
        self.group_status.set_text("")
        self.group_status.set_tooltip_text(None)
        self.group_status.get_style_context().remove_class("error")

    # --------------------------------------------------------------- analyses
    def _tab_analyses(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        gen = W.Form()
        gen.add_wide(b.check("analyses.use_pbc", W.check(
            "Use periodic (minimum-image) distances (not needed after cpptraj imaging)")))
        gen.add("Proximity search radius (Å)", b.spin_float(
            "analyses.proximity_search_radius", W.spin(1, 30, 0.5, 1)),
            "Largest proximity cutoff usable at render time without recomputing")
        box.pack_start(W.frame("General", gen), False, False, 0)

        r = W.Form()
        r.add_wide(b.check("analyses.rmsd.enabled", W.check("Compute RMSD")))
        self.rmsd_editor = W.ListEditor([("Label", 100), ("Selection", 300), ("Fit selection (optional)", 200)],
                                        height=70, defaults=["Target", "group:Aptamer and not name H*", ""])
        r.add("Targets", self.rmsd_editor)
        b._add("analyses.rmsd.targets", self.rmsd_editor,
               lambda: [RMSDTarget(l.strip(), s.strip(), f.strip())
                        for l, s, f in self.rmsd_editor.rows() if s.strip()],
               lambda v: self.rmsd_editor.set_rows([[t.label, t.selection, t.fit_selection] for t in v]))
        r.add_row("Reference frame", [b.spin_int("analyses.rmsd.reference_frame", W.spin(0, 10 ** 9)),
                                      b.check("analyses.rmsd.superposition", W.check("superpose")),
                                      b.check("analyses.rmsd.mass_weighted", W.check("mass weighted"))],
                  "Index into the analysed frames")
        box.pack_start(W.frame("RMSD", r), False, False, 0)

        c = W.Form()
        c.add_wide(b.check("analyses.contacts.enabled", W.check("Compute heavy-atom contacts")))
        c.add_row("Groups", [self._group_combo("analyses.contacts.group_a"), W.label("vs"),
                             self._group_combo("analyses.contacts.group_b")])
        c.add_row("Cutoff (Å)", [b.spin_float("analyses.contacts.cutoff", W.spin(0.5, 20, 0.1, 2)),
                                     b.check("analyses.contacts.heavy_only", W.check("heavy atoms only"))])
        box.pack_start(W.frame("Pairwise contacts", c), False, False, 0)

        h = W.Form()
        h.add_wide(b.check("analyses.hbonds.enabled", W.check("Count inter-group hydrogen bonds")))
        h.add_row("Groups", [self._group_combo("analyses.hbonds.group_a"), W.label("vs"),
                             self._group_combo("analyses.hbonds.group_b")])
        h.add_row("D–A distance (Å) / angle", [
            b.spin_float("analyses.hbonds.distance_cutoff", W.spin(2.0, 5.0, 0.05, 2)),
            b.check("analyses.hbonds.use_angle", W.check("use D–H···A angle ≥")),
            b.spin_float("analyses.hbonds.angle_cutoff", W.spin(90, 180, 1, 0))])
        h.add_row("Donor / acceptor elements", [b.entry("analyses.hbonds.donor_elements", W.entry(width=8)),
                                                b.entry("analyses.hbonds.acceptor_elements", W.entry(width=8))])
        box.pack_start(W.frame("Hydrogen bonds", h), False, False, 0)

        m = W.Form()
        m.add_wide(b.check("analyses.mg.enabled", W.check("Analyse Mg²⁺ coordination")))
        m.add("Mg selection", b.entry("analyses.mg.mg_selection", W.entry()))
        m.add_row("Ligand selection / elements", [b.entry("analyses.mg.ligand_selection", W.entry(width=20)),
                                                  b.entry("analyses.mg.ligand_elements", W.entry(width=6))],
                  "Oxygens of these residue ranges count for Mg–O and O–Mg–O")
        m.add_row("Bridging groups", [self._group_combo("analyses.mg.group_a"), W.label("vs"),
                                      self._group_combo("analyses.mg.group_b")])
        m.add_row("Inner / outer sphere (Å)", [
            b.spin_float("analyses.mg.inner_cutoff", W.spin(1.5, 4.0, 0.05, 2)),
            b.spin_float("analyses.mg.outer_cutoff", W.spin(2.0, 8.0, 0.1, 2))])
        m.add("Bridges counted from", b.combo("analyses.mg.bridge_shell", W.combo(
            [("inner", "inner sphere"), ("outer", "inner + outer sphere")])))
        m.add_row("Coordination geometry", [
            b.check("analyses.mg.geometry", W.check("require octahedral O–Mg–O angles ±")),
            b.spin_float("analyses.mg.angle_tolerance", W.spin(5, 45, 1, 0))])
        m.add_row("Occupancy residues / atoms", [
            b.entry("analyses.mg.occupancy_selection", W.entry("default: ligand selection", width=20)),
            b.entry("analyses.mg.occupancy_elements", W.entry(width=6))])
        box.pack_start(W.frame("O–Mg–O contacts and Mg occupancy", m), False, False, 0)

        s = W.Form()
        s.add_wide(b.check("analyses.stacking.enabled", W.check("Detect π–π stacking")))
        s.add("Selection", b.entry("analyses.stacking.selection", W.entry()))
        s.add("Scope", b.combo("analyses.stacking.scope", W.combo([("all", "all ring pairs"),
                                                                   ("inter", "group A vs group B")])))
        s.add_row("Groups (A vs B)", [self._group_combo("analyses.stacking.group_a"), W.label("vs"),
                                      self._group_combo("analyses.stacking.group_b")],
                  "Used when the scope is 'group A vs group B'")
        s.add_row("Centroid distance (Å) / angle (°)", [
            b.spin_float("analyses.stacking.distance_cutoff", W.spin(2.5, 8, 0.1, 2)),
            b.spin_float("analyses.stacking.angle_cutoff", W.spin(5, 90, 1, 0))],
            "Ring-centroid distance and angle between ring normals")
        s.add_row("Offset / planarity (Å)", [
            b.spin_float("analyses.stacking.offset_cutoff", W.spin(0.5, 5, 0.1, 2)),
            b.spin_float("analyses.stacking.planarity_cutoff", W.spin(0.01, 2, 0.01, 2))],
            "Lateral centroid offset and RMS deviation of the ring atoms from their plane")
        box.pack_start(W.frame("π–π stacking", s), False, False, 0)

        bp = W.Form()
        bp.add_wide(W.hint("Used by the secondary-structure panel (and PDB annotation): pairs of the "
                           "structure-view residues, re-detected every n frames."))
        bp.add("Engine", b.combo("analyses.basepairs.engine", W.combo([
            ("builtin", "built-in (canonical pairs from H-bond geometry)"),
            ("barnaba", "Barnaba (optional; Leontis-Westhof classes)")])))
        bp.add_row("Every n frames / cutoff (Å)", [
            b.spin_int("analyses.basepairs.stride", W.spin(1, 100000)),
            b.spin_float("analyses.basepairs.cutoff", W.spin(2.0, 5.0, 0.05, 2))],
            "Pairs are re-detected every n-th analysed frame; the cutoff is the built-in engine's "
            "largest donor-acceptor distance")
        bp.add_wide(b.check("analyses.basepairs.wobble", W.check("Built-in: count G-U wobble pairs")))
        bp.add_wide(b.check("analyses.basepairs.noncanonical", W.check("Barnaba: keep non-canonical pairs")))
        box.pack_start(W.frame("Base pairs (secondary structure)", bp), False, False, 0)

        e = W.Form()
        e.add_wide(b.check("analyses.energy.enabled", W.check(
            "Compute intermolecular interaction energy and net force")))
        e.add_row("Groups", [self._group_combo("analyses.energy.group_a"), W.label("vs"),
                             self._group_combo("analyses.energy.group_b")])
        e.add_row("Cutoff (Å) / dielectric", [
            b.spin_float("analyses.energy.cutoff", W.spin(4, 50, 0.5, 1)),
            b.spin_float("analyses.energy.dielectric", W.spin(1, 100, 0.5, 1))])
        e.add_wide(b.check("analyses.energy.distance_dependent", W.check("Distance-dependent dielectric (εr)")))
        e.add_wide(b.check("analyses.energy.use_parm7", W.check("Amber LJ parameters from the parm7 when available")))
        e.add_row("Exclude pairs within", [b.spin_int("analyses.energy.exclude_bonded", W.spin(0, 3)),
                                           W.label("bonds")])
        e.add("Target value (kcal/mol)", b.optional_float("analyses.energy.target_value",
              W.entry("none", width=10)), "Drawn as a reference line on charts and panels")
        box.pack_start(W.frame("Intermolecular energy", e), False, False, 0)
        return W.scrolled(box)

    # ----------------------------------------------------------------- visual
    def _tab_visual(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        rot = W.Form()
        self.rot_scales = []
        for axis in "XYZ":
            sc = W.no_wheel(Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -180, 180, 0.5))
            sc.set_digits(1)
            sc.set_value_pos(Gtk.PositionType.RIGHT)
            sc.set_size_request(220, -1)
            sc.connect("value-changed", self._on_scale)
            self.rot_scales.append(sc)
            rot.add(f"Rotation {axis} (\u00b0)", sc)
        b._add("visual.rotation", self.rot_scales[0],
               lambda: [round(s.get_value(), 3) for s in self.rot_scales], self._set_rotation)
        self.angle_label = W.label("")
        self.angle_label.set_selectable(True)
        rot.add_wide(self.angle_label)
        self.zoom_spin = b.spin_float("visual.zoom", W.spin(0.05, 50, 0.1, 2))
        self.zoom_spin.connect("value-changed", lambda *_: None if self._loading else self.refresh_canvas())
        rot.add("Zoom", self.zoom_spin, "1 fits the whole model; 2 shows it twice as large (mouse wheel over "
                                       "the structure panel)")
        self.center_entry = center = b.entry("visual.center_on", W.entry("model centre; 12, B:12 or origin"))
        center.connect("activate", lambda *_: self.refresh_canvas())
        center.connect("focus-out-event", lambda *_: self.refresh_canvas() or False)
        rot.add("Centre on", center, "Residue (number, or chain:number) or 'origin' (0, 0, 0 of the "
                                     "coordinates) in the middle of the view; empty: the model centre")
        follow = b.check("visual.center_follow", W.check("Follow the centre residue in every frame"))
        follow.connect("toggled", lambda *_: None if self._loading else self.refresh_canvas())
        rot.add_wide(follow)
        row = Gtk.Box(spacing=6)
        row.pack_start(W.button("Reset orientation", lambda: self._set_rotation([0, 0, 0])),
                       False, False, 0)
        row.pack_start(W.button("Reset zoom and centre", self._reset_view), False, False, 0)
        rot.add_wide(row)
        rot.add_wide(W.hint("Drag the structure panel in the figure to rotate the model, use the mouse "
                            "wheel over it to zoom. Orientation, zoom and centre are set on the reference "
                            "frame and applied to every frame of the video."))
        box.pack_start(W.frame("Orientation", rot), False, False, 0)
        box.pack_start(self._representation_frame(), False, False, 0)
        box.pack_start(self._density_frame(), False, False, 0)

        tags = W.Form()
        tags.add("Label every N-th residue", b.spin_int("visual.label_every", W.spin(0, 10000)),
                 "0 turns interval labels off")
        tags.add("Label residues", b.entry("visual.label_residues", W.entry("e.g. 1, 12, 24, 40-45")),
                 "Residue numbers or ranges to tag (in addition to the interval)")
        box.pack_start(W.frame("Residue number tags", tags), False, False, 0)

        f = W.Form()
        f.add("Groups shown", b.csv("visual.structure_groups", W.entry("all groups", width=24)),
              "Comma separated group names")
        f.add("Trace atoms (priority)", b.entry("visual.trace_atoms", W.entry(width=16)),
              "First atom found per residue, e.g. P or C4' (CA for proteins)")
        f.add_row("Options", [b.check("visual.align_frames", W.check("superpose frames")),
                              b.check("visual.show_ions", W.check("show Mg\u00b2\u207a")),
                              b.check("visual.depth_cue", W.check("depth cue"))])
        f.add("Colouring", b.combo("visual.color_mode", W.combo([(m, m) for m in COLOR_MODES])))
        f.add_row("Base / ion colour", [b.color("visual.base_color", W.color_button()),
                                        b.color("visual.ion_color", W.color_button())])
        f.add("Chain palette", b.combo_text("visual.chain_palette", W.combo_entry(
            ["tab10", "Set2", "Dark2", "viridis"])))
        f.add("Proximity set", b.entry("visual.proximity_set", W.entry(width=20)),
              "Group name, or sel:<selection>")
        f.add_row("Proximity cutoff (\u00c5) / colour", [
            b.spin_float("visual.proximity_cutoff", W.spin(1, 30, 0.1, 1)),
            b.color("visual.proximity_color", W.color_button())])
        f.add_row("Glow", [b.combo("visual.glow_mode", W.combo([
            ("none", "none"), ("ever", "whole movie if ever in proximity"),
            ("fade", "while in proximity, then fade"), ("current", "only while in proximity")])),
            b.color("visual.glow_color", W.color_button())])
        f.add_row("Fade frames / strength", [b.spin_int("visual.fade_frames", W.spin(0, 10000)),
                                             b.spin_float("visual.glow_strength", W.spin(0, 3, 0.1, 1))])
        f.add_row("Line width / point size", [b.spin_float("visual.line_width", W.spin(0.2, 10, 0.1, 1)),
                                              b.spin_float("visual.point_size", W.spin(1, 200, 1, 0))])
        f.add_row("Theme / font size", [b.combo("visual.theme", W.combo([("light", "light"),
                                                                          ("dark", "dark")])),
                                        b.spin_float("visual.font_size", W.spin(6, 30, 0.5, 1))])
        f.add_row("Labels", [b.check("visual.show_time_label", W.check("time label")),
                             b.entry("visual.title", W.entry("figure title (optional)", width=20))])
        box.pack_start(W.frame("Appearance", f), False, False, 0)
        return W.scrolled(box)

    REP_STYLE_LABELS = {"cartoon": "New Cartoon (bases as slabs)", "ribbon": "Ribbon", "tube": "Tube",
                        "licorice": "Licorice", "vdw": "VDW spheres", "surf": "Surface",
                        "trace": "Trace (balls and sticks)"}
    REP_COLOR_LABELS = {"auto": "auto", "residue": "residue colours", "element": "elements",
                        "nucleotide": "nucleotides (A/G/C/U)"}

    def _representation_frame(self):
        b = self.binder
        f = W.Form()
        self.rep_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.rep_rows: List[dict] = []
        b._add("visual.representations", self.rep_box, self._read_reps, self._set_reps)
        f.add_wide(self.rep_box)
        row = Gtk.Box(spacing=6)
        row.pack_start(W.button("Add representation", lambda: self._add_rep_row(RepSpec()),
                                "Draw residue groups as cartoon, ribbon, tube, licorice, VDW or surface",
                                "list-add-symbolic"), False, False, 0)
        row.pack_start(W.button("Apply", self.refresh_canvas, "Redraw the preview"), False, False, 0)
        f.add_wide(row)
        f.add_row("Other residues / backbone atom", [
            b.combo("visual.rep_rest", W.combo([("trace", "as a trace"), ("hide", "hidden")])),
            b.combo("visual.backbone_atom", W.combo([("P", "P"), ("C4'", "C4'")])),
            b.check("visual.rep_hydrogens", W.check("hydrogens"))],
            "Residues no representation draws are shown as a 3D trace or hidden. Ribbons, cartoons and "
            "the trace run through P or C4'.")
        f.add("Renderer", b.combo("visual.renderer", W.combo([("auto", "auto (built-in)"), ("software", "built-in"),
                                                           ("vtk", "VTK (optional)")])),
              "The built-in renderer needs no extra packages and works without a display (clusters). "
              "VTK (pip install vtk) needs OpenGL; where it cannot render, the built-in one is used.")
        f.add_wide(W.hint("No representation: the 2D backbone trace. Each representation draws its groups "
                          "(empty: all shown groups) and they intersect by depth. Colouring 'auto' uses the "
                          "residue colours (groups / proximity / glow) for backbones, nucleotide colours for "
                          "cartoon bases and element colours for atoms."))
        return W.frame("Representations (3D)", f)

    def _add_rep_row(self, spec: RepSpec) -> None:
        grid = Gtk.Box(spacing=4)
        style = W.combo([(k, self.REP_STYLE_LABELS.get(k, k)) for k in REP_STYLES])
        style.set_active_id(spec.style)
        groups = W.entry("all groups", width=12)
        groups.set_text(", ".join(spec.groups))
        groups.set_tooltip_text("Residue groups (comma separated); empty: every shown group")
        color = W.combo([(k, self.REP_COLOR_LABELS.get(k, k)) for k in REP_COLORS])
        color.set_active_id(spec.color)
        size = W.spin(0.2, 5, 0.1, 1, "Size (scales widths and radii)")
        size.set_value(spec.size)
        opacity = W.spin(0.05, 1, 0.05, 2, "Opacity")
        opacity.set_value(spec.opacity)
        sugars = W.check("sugars", "Cartoon: draw the ribose rings")
        sugars.set_active(spec.sugars)
        remove = Gtk.Button.new_from_icon_name("list-remove-symbolic", Gtk.IconSize.BUTTON)
        remove.set_tooltip_text("Remove this representation")
        for w in (style, groups, color, size, opacity, sugars):
            grid.pack_start(w, False, False, 0)
        grid.pack_end(remove, False, False, 0)
        row = {"box": grid, "style": style, "groups": groups, "color": color, "size": size,
               "opacity": opacity, "sugars": sugars}
        style.connect("changed", lambda *_: sugars.set_sensitive(style.get_active_id() == "cartoon"))
        sugars.set_sensitive(spec.style == "cartoon")
        remove.connect("clicked", lambda *_: self._remove_rep_row(row))
        self.rep_rows.append(row)
        self.rep_box.pack_start(grid, False, False, 0)
        grid.show_all()
        W.watch_tree(grid, self.schedule_check)
        self.schedule_check()

    def _remove_rep_row(self, row: dict) -> None:
        self.rep_rows.remove(row)
        self.rep_box.remove(row["box"])
        self.refresh_canvas()
        self.schedule_check()

    def _read_reps(self) -> List[RepSpec]:
        return [RepSpec(style=r["style"].get_active_id() or "cartoon",
                        groups=[g.strip() for g in r["groups"].get_text().split(",") if g.strip()],
                        color=r["color"].get_active_id() or "auto", size=round(r["size"].get_value(), 3),
                        opacity=round(r["opacity"].get_value(), 3), sugars=r["sugars"].get_active())
                for r in self.rep_rows]

    def _set_reps(self, specs) -> None:
        for r in list(self.rep_rows):
            self.rep_box.remove(r["box"])
        self.rep_rows = []
        for spec in specs or []:
            self._add_rep_row(spec)

    def _density_frame(self):
        b = self.binder
        d = W.Form()
        mp = W.file_chooser_button("Select a density map", patterns=[MAP_FILES])
        d.add("Map file", b.path_entry("density.map_path", mp), "MRC2014 / CCP4 map (.mrc, .map)")
        mp.entry.connect("changed", self._on_density_map_changed)
        self.density_map_entry = mp.entry
        opacity = W.no_wheel(Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.05))
        opacity.set_digits(2)
        opacity.set_value_pos(Gtk.PositionType.RIGHT)
        opacity.set_size_request(140, -1)
        b._add("density.opacity", opacity, lambda: round(opacity.get_value(), 3),
               lambda v: opacity.set_value(float(v)))
        opacity.connect("value-changed", lambda *_: self._density_style_changed())
        d.add_row("Show / colour / opacity", [b.check("density.show", W.check("show")),
                                              b.color("density.color", W.color_button()), opacity,
                                              b.check("density.front", W.check(
                                                  "in front", "Draw the shadow over the 3D representations "
                                                  "instead of behind them"))],
                  "The map is drawn as a projected shadow behind the model")
        level = b.spin_float("density.level", W.spin(-5, 20, 0.1, 1))
        d.add_row("Contour level (σ) / resolution (Å)", [
            level, b.spin_float("density.resolution", W.spin(0.5, 50, 0.1, 1))],
            "Density below mean + level × σ casts no shadow; the resolution sets the width of "
            "the simulated map used for fitting")
        d.add_row("Fit frame / start / step (°)", [
            b.entry("density.fit_frame", W.entry("ref", width=6)),
            b.combo("density.search", W.combo([("global", "global search"), ("manual", "my placement")])),
            b.spin_float("density.angular_step", W.spin(5, 45, 1, 0))],
            "Frame 'ref' is the first analysed frame (the video's reference); any trajectory frame "
            "number works. The global search samples orientations every <step> degrees.")
        d.add_wide(b.check("density.auto_fit", W.check(
            "Fit automatically before rendering when the map has no placement yet")))
        row = Gtk.Box(spacing=6)
        for text, cb, tip in (
                ("Global search", lambda: self.fit_density("global"),
                 "Exhaustive orientation search + local refinement (maximises the correlation with a "
                 "map simulated from the model)"),
                ("Refine", lambda: self.fit_density("manual"),
                 "Local refinement from the current placement"),
                ("Centre on model", self.centre_density, "Place the map's density centre on the model "
                 "without rotating it (a starting point for moving it by hand)"),
                ("Clear", self.clear_density_fit, "Forget the placement")):
            btn = W.button(text, cb, tip)
            if text in ("Global search", "Refine"):
                self.busy_widgets.append(btn)
            row.pack_start(btn, False, False, 0)
        d.add_wide(row)
        self.map_toggle = Gtk.ToggleButton(label="Move the map with the mouse")
        self.map_toggle.set_tooltip_text("Drag in the structure panel to move the map; Shift + drag rotates "
                                         "it about its centre. Then 'Refine' to fit from there.")
        self.map_toggle.connect("toggled", lambda t: setattr(self.canvas, "map_mode", t.get_active()))
        d.add_wide(self.map_toggle)
        self.density_status = W.label("", wrap=True)
        self.density_status.set_selectable(True)
        d.add_wide(self.density_status)
        d.add_wide(W.hint("The map is shown on the real model: use 'Load real frame' (toolbar) or a fit, "
                          "which loads it. The placement is stored in the project."))
        return W.frame("Density map", d)

    def _update_density_status(self) -> None:
        dp = self.project.density
        if not dp.map_path:
            text = "No map selected."
        elif len(dp.transform) == 12:
            text = dp.fit_info or "Placed (not fitted)."
        else:
            text = "Not placed yet: run a global search, or centre it and move it by hand."
        self.density_status.set_text(text)

    def _on_density_map_changed(self, entry) -> None:
        if self._loading:
            return
        path = entry.get_text().strip()
        dp = self.project.density
        if path != dp.map_path:  # a placement belongs to one map
            dp.map_path = path
            dp.transform, dp.anchor, dp.fit_cc, dp.fit_info = [], [], 0.0, ""
            self._update_density_status()

    def _density_style_changed(self) -> None:
        if not self._loading:
            self.refresh_canvas()

    def _on_map_moved(self, values) -> None:
        from ..density import anchor_points

        dp = self.project.density
        dp.transform = list(values)
        st = self.canvas.structure
        if st is not None:  # the placement belongs to the frame shown on the canvas
            dp.anchor = anchor_points(st.centred[0] + st.center)
        dp.fit_cc = 0.0
        dp.fit_info = "Placed by hand (not fitted) - 'Refine' fits from here."
        self._update_density_status()

    def _adopt_auto_fit(self, project) -> None:
        """Keep a placement found by an automatic fit during a render job."""
        dp, new = self.project.density, project.density
        if len(new.transform) == 12 and new.transform != dp.transform and new.map_path == dp.map_path:
            self._store_density_result(new)
            self.append_log(f"Density map placement stored: {new.fit_info}")
            self.refresh_canvas()

    def _store_density_result(self, density) -> None:
        dp = self.project.density
        dp.transform, dp.fit_cc, dp.fit_info = list(density.transform), density.fit_cc, density.fit_info
        dp.anchor = [list(p) for p in density.anchor]
        self._update_density_status()

    def fit_density(self, search: str) -> None:
        try:
            project = self.collect(video=False, charts=False, pdb=False, density=True)
        except ValueError as exc:
            self.error("Please fix these parameters first", str(exc))
            return
        if not project.density.map_path:
            self.error("No density map", "Choose an MRC / CCP4 map first.")
            return
        def job(rep):
            pipe = self._pipeline(project, rep)
            pipe.fit_density(search=search)
            # the fit may have run cpptraj: show the model from the same (cleaned) files
            return project.density, pipe.structure_preview()

        def done(result):
            density, preview = result
            self._store_density_result(density)
            if preview is not None:
                self.preview_data = preview
            self.refresh_canvas()

        self.start_job("Fitting the density map", job, done)

    def centre_density(self) -> None:
        from ..density import centre_transform, load_map_cached, map_level

        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return
        st = self.canvas.structure
        if self.preview_data is None or st is None:
            self.info("Load the real reference frame first", "Use 'Load real frame' in the Layout section, "
                      "then centre the map on the model.")
            return
        try:
            dmap = load_map_cached(project.density.map_path)
        except (OSError, ValueError) as exc:
            self.error("Cannot read the density map", str(exc))
            return
        t = centre_transform(dmap, map_level(dmap, project.density.level), st.center)
        self._on_map_moved(t.to_list())
        self.project.density.fit_info = "Centred on the model (not fitted)."
        self._update_density_status()
        self.refresh_canvas()

    def clear_density_fit(self) -> None:
        dp = self.project.density
        dp.transform, dp.anchor, dp.fit_cc, dp.fit_info = [], [], 0.0, ""
        self._update_density_status()
        self.refresh_canvas()

    def _set_rotation(self, angles):
        self._syncing = True
        try:
            for sc, a in zip(self.rot_scales, angles):
                sc.set_value(float(a))
        finally:
            self._syncing = False
        self._on_scale()

    def _reset_view(self):
        self.zoom_spin.set_value(1.0)
        self.center_entry.set_text("")
        self.refresh_canvas()

    def _on_scale(self, *_):
        angles = [s.get_value() for s in self.rot_scales]
        self.angle_label.set_text(format_angles(angles))
        if not self._syncing:  # (the canvas already shows rotations it reported itself)
            self.canvas.set_rotation(angles, fast=True)

    # ----------------------------------------------------------------- layout
    def _tab_layout(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        form = W.Form()
        self.preset_combo = W.combo([(k, PRESET_LABELS.get(k, k)) for k in PRESETS])
        self.preset_combo.connect("changed", lambda *_: self._on_preset_changed())
        form.add("Grid preset", self.preset_combo)
        form.add_wide(W.hint(
            "Panels fill the slots in order (slot 1 is the large one). Drag panels in the figure "
            "to swap them. Choose the data and, for analyses that compare residue groups, which "
            "groups to use: each group choice is computed and cached on its own. \u201cMore keys\u201d "
            "adds series to a time-series panel (comma separated data keys); options are JSON, e.g. "
            "{\"mode\": \"frequency\"} for a contact map or {\"progressive\": true} for a time series."))
        self.slot_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        form.add_wide(self.slot_box)
        prev = Gtk.Box(spacing=6)
        self.preview_pos = W.spin(0, 10 ** 9)
        prev.pack_start(W.label("Real-data still of frame"), False, False, 0)
        prev.pack_start(self.preview_pos, False, False, 0)
        pb = W.button("Render still", self.preview_frame,
                      "Analyse (or load cached results) and render one frame of the video",
                      "image-x-generic-symbolic")
        self.busy_widgets.append(pb)
        prev.pack_start(pb, False, False, 0)
        form.add_wide(prev)
        box.pack_start(W.frame("Figure layout", form), False, False, 0)
        self.binder._add("layout", self.preset_combo, self._get_layout, self._set_layout)
        return W.scrolled(box)

    def _on_preset_changed(self):
        if self._loading:
            return
        self._rebuild_slots()
        self.refresh_canvas()

    def _project_for_choices(self) -> Project:
        try:
            return self.collect(validate=False, include_layout=False)
        except Exception:  # noqa: BLE001 - choices are best effort
            return self.project

    def _data_choices(self, panel_type: str, project: Project) -> List[str]:
        if panel_type not in PANEL_TYPES or PANEL_TYPES[panel_type].max_series == 0:
            return []  # structure, sequence and secondary structure use the structure-view residues
        return expected_keys(project).get(PANEL_TYPES[panel_type].data_kind, [])

    def _slot_editor(self, k: int, spec: PanelSpec, preset: str, project: Project) -> dict:
        row: dict = {"auto": ""}
        form = W.Form()
        title = f"Slot {k + 1}" + (" (large)" if k == 0 and preset.startswith("large") else "")
        frame = W.frame(title, form)
        row["frame"] = frame
        tcombo = W.combo([(t, cls.label) for t, cls in PANEL_TYPES.items()])
        form.add("Panel type", tcombo)
        data = W.no_wheel(Gtk.ComboBoxText())
        data.set_hexpand(True)
        data.set_size_request(200, -1)
        for cell in data.get_cells():  # long labels shrink with an ellipsis
            cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        form.add("Data", data)
        ga, gb = W.no_wheel(Gtk.ComboBoxText()), W.no_wheel(Gtk.ComboBoxText())
        for c in (ga, gb):
            c.append(DEFAULT_GROUP, "(analysis default)")
            for name in project.group_names():
                c.append(name, name)
            W.ellipsize_combo(c, 8)
        form.add_row("Groups", [ga, W.label("vs"), gb],
                     "Residue groups this panel's analysis compares; each choice is cached separately")
        more = W.entry("additional data keys (comma separated)")
        form.add("More keys", more)
        title_entry = W.entry("automatic")
        form.add("Title", title_entry, "Follows the selected data until you type your own title")
        opts = W.entry("{}")
        form.add("Options (JSON)", opts)
        row.update(type=tcombo, data=data, ga=ga, gb=gb, more=more, title=title_entry, opts=opts)

        primary = spec.series[0] if spec.series else ""
        base_key, groups = primary, ()
        if primary and "/" in primary:
            inst, out = split_key(primary)
            base, groups = parse_instance(inst)
            base_key = f"{base}/{out}"
        tcombo.set_active_id(spec.type)
        self._fill_data(row, spec.type, project, base_key)
        for c, name in zip((ga, gb), groups):
            if name not in project.group_names():  # a deleted group stays visible until it is replaced
                c.append(name, f"{name} (missing)")
        ga.set_active_id(groups[0] if len(groups) > 0 else DEFAULT_GROUP)
        gb.set_active_id(groups[1] if len(groups) > 1 else DEFAULT_GROUP)
        if ga.get_active_id() is None:
            ga.set_active_id(DEFAULT_GROUP)
        if gb.get_active_id() is None:
            gb.set_active_id(DEFAULT_GROUP)
        more.set_text(", ".join(spec.series[1:]))
        opts.set_text(json.dumps(spec.options) if spec.options else "")
        title_entry.set_text(spec.title)
        # a default or automatic title follows the panel; a title you typed is kept
        row["auto"] = spec.title if spec.title in (self._auto_title(row, project), *DEFAULT_PANEL_TITLES) else ""
        self._update_group_widgets(row)

        tcombo.connect("changed", lambda *_: self._on_slot_type(row))
        data.connect("changed", lambda *_: self._on_slot_data(row))
        ga.connect("changed", lambda *_: self._on_slot_data(row))
        gb.connect("changed", lambda *_: self._on_slot_data(row))
        for w in (more, title_entry, opts):
            w.connect("activate", lambda *_: self.refresh_canvas())
        return row

    def _fill_data(self, row, panel_type, project, select=""):
        data = row["data"]
        data.remove_all()
        keys = self._data_choices(panel_type, project)
        if select and select not in keys:
            keys = [select] + keys
        for key in keys:
            data.append(key, f"{key_label(key, project)}   [{key}]")
        if keys:
            data.set_active_id(select if select else keys[0])
        data.set_sensitive(bool(keys))

    def _update_group_widgets(self, row):
        key = row["data"].get_active_id() or ""
        slots = 0
        if "/" in key:
            try:
                slots = analysis_class(split_key(key)[0]).group_slots
            except KeyError:
                slots = 0
        row["ga"].set_sensitive(slots >= 1)
        row["gb"].set_sensitive(slots >= 2)

    def _primary_key(self, row, project: Optional[Project] = None) -> str:
        key = row["data"].get_active_id() or ""
        if "/" not in key:
            return key
        inst, out = split_key(key)
        base, _ = parse_instance(inst)
        try:
            cls = analysis_class(base)
        except KeyError:
            return key
        chosen = [row["ga"].get_active_id() or DEFAULT_GROUP,
                  row["gb"].get_active_id() or DEFAULT_GROUP][:cls.group_slots]
        if not any(chosen):
            return key
        defaults = list(cls.groups(project or self._project_for_choices())) or [""] * cls.group_slots
        groups = [c or d for c, d in zip(chosen, defaults)]
        if cls.group_slots == 1 and not groups[0]:
            return key
        return f"{instance_id(base, groups)}/{out}"

    def _auto_title(self, row, project) -> str:
        if (row["type"].get_active_id() or "") == "structure":
            return "Structure"
        key = self._primary_key(row, project)
        return key_label(key, project) if key else ""

    def _on_slot_type(self, row):
        project = self._project_for_choices()
        ptype = row["type"].get_active_id() or ""
        if ptype in SLOT_DEFAULT_OPTIONS and not row["opts"].get_text().strip():
            row["opts"].set_text(json.dumps(SLOT_DEFAULT_OPTIONS[ptype]))
        self._fill_data(row, row["type"].get_active_id() or "timeseries", project)
        self._on_slot_data(row, new_type=True)

    def _on_slot_data(self, row, new_type: bool = False):
        self._update_group_widgets(row)
        project = self._project_for_choices()
        auto = self._auto_title(row, project)
        current = row["title"].get_text().strip()
        # a new panel type always gets its automatic title; within a type, a typed title is kept
        if new_type or not current or current == row["auto"]:
            row["title"].set_text(auto)
        row["auto"] = auto
        if not self._loading:
            self.refresh_canvas()

    def _rebuild_slots(self, specs: Optional[List[PanelSpec]] = None):
        if specs is None:
            specs = self._read_slots()
        for child in self.slot_box.get_children():
            self.slot_box.remove(child)
        self.slot_rows = []
        preset = self.preset_combo.get_active_id() or "single"
        project = self._project_for_choices()
        loading, self._loading = self._loading, True
        try:
            for k in range(n_slots(preset)):
                spec = specs[k] if k < len(specs) else PanelSpec("timeseries", [], "")
                row = self._slot_editor(k, spec, preset, project)
                self.slot_rows.append(row)
                self.slot_box.pack_start(row["frame"], False, False, 0)
        finally:
            self._loading = loading
        self.slot_box.show_all()
        W.watch_tree(self.slot_box, self.schedule_check)
        self.schedule_check()  # the new slot widgets get their flags back

    def _read_slots(self) -> List[PanelSpec]:
        specs = []
        project = None
        for k, row in enumerate(self.slot_rows):
            text = row["opts"].get_text().strip()
            try:
                options = json.loads(text) if text else {}
                if not isinstance(options, dict):
                    raise ValueError("options must be a JSON object")
            except ValueError as exc:
                err = ValueError(f"panel options {text!r}: {exc}")
                err.field = f"layout.panels.{k}.options"  # type: ignore[attr-defined]  # flagged there
                raise err from exc
            ptype = row["type"].get_active_id() or "timeseries"
            keys = []
            if PANEL_TYPES.get(ptype) is None or PANEL_TYPES[ptype].max_series > 0:
                project = project or self._project_for_choices()
                primary = self._primary_key(row, project)
                keys = ([primary] if primary else []) + [
                    k.strip() for k in row["more"].get_text().split(",") if k.strip()]
            specs.append(PanelSpec(ptype, keys, row["title"].get_text().strip(), options))
        return specs

    def _get_layout(self):
        from ..config import LayoutParams
        return LayoutParams(preset=self.preset_combo.get_active_id() or "single",
                            panels=self._read_slots())

    def _set_layout(self, layout):
        loading, self._loading = self._loading, True
        try:
            self.preset_combo.set_active_id(layout.preset)
        finally:
            self._loading = loading
        self._rebuild_slots(list(layout.panels))

    # ---------------------------------------------------------------- outputs
    def _tab_outputs(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        g = W.Form()
        g.add("Output directory", b.path_entry("output.out_dir", W.file_chooser_button(
            "Select output directory", Gtk.FileChooserAction.SELECT_FOLDER)))
        g.add("Cache directory", b.path_entry("output.cache_dir", W.file_chooser_button(
            "Select cache directory", Gtk.FileChooserAction.SELECT_FOLDER)),
            f"Blank: <output>/{PROJECT_ABBREV}_cache (analysis .npz + JSON, cleaned trajectories)")
        g.add_wide(b.check("output.write_params_log", W.check(
            f"Write {PROJECT_ABBREV}_params_log.toml (parameters, versions, cache keys, citations) for "
            "reproducibility; open it as a project to repeat the run")))
        box.pack_start(W.frame("Locations", g), False, False, 0)

        v = W.Form()
        v.add_wide(b.check("output.video.enabled", W.check("Write MP4 video")))
        v.add("File name", b.entry("output.video.filename", W.entry()))
        v.add_row("Resolution (px)", [b.spin_int("output.video.width", W.spin(64, 7680, 2)),
                                      W.label("×"),
                                      b.spin_int("output.video.height", W.spin(64, 4320, 2))])
        v.add("DPI", b.spin_int("output.video.dpi", W.spin(10, 600, 1)))
        v.add_row("FPS / frame stride", [b.spin_int("output.video.fps", W.spin(1, 240)),
                                         b.spin_int("output.video.frame_stride", W.spin(1, 10000))])
        v.add_row("Codec / quality (CRF)", [b.combo_text("output.video.codec", W.combo_entry(
            ["libx264", "libx265", "mpeg4"])), b.spin_int("output.video.crf", W.spin(0, 51))])
        v.add("FFmpeg executable", b.path_entry("output.video.ffmpeg_path",
                                                W.file_chooser_button("Select ffmpeg")))
        box.pack_start(W.frame("Video", v), False, False, 0)

        c = W.Form()
        c.add_wide(b.check("output.charts.enabled", W.check("Write standalone time charts")))
        c.add("Series", b.csv("output.charts.series", W.entry("all series", width=24)),
              "Comma separated data keys, blank = every time series")
        c.add_wide(b.check("output.charts.include_heatmaps", W.check("Also chart per-residue heatmaps")))
        fmt_box = Gtk.Box(spacing=6)
        self.fmt_checks = {f: W.check(f.upper()) for f in ("png", "svg", "pdf")}
        for w in self.fmt_checks.values():
            fmt_box.pack_start(w, False, False, 0)
        c.add("Formats", fmt_box)
        b._add("output.charts.formats", fmt_box,
               lambda: [f for f, w in self.fmt_checks.items() if w.get_active()] or ["png"],
               lambda v: [w.set_active(f in v) for f, w in self.fmt_checks.items()])
        c.add_row("Size (in)", [b.spin_float("output.charts.width", W.spin(2, 30, 0.5, 1)),
                                W.label("×"),
                                b.spin_float("output.charts.height", W.spin(2, 30, 0.5, 1))])
        c.add("DPI", b.spin_int("output.charts.dpi", W.spin(10, 1200, 10)))
        styles = ["mdatk"]
        try:
            import matplotlib.style as mstyle
            styles += sorted(mstyle.available)
        except Exception:  # noqa: BLE001
            pass
        c.add_row("Style / line colour / width", [
            b.combo_text("output.charts.style", W.combo_entry(styles)),
            b.entry("output.charts.color", W.entry("palette", width=9)),
            b.spin_float("output.charts.line_width", W.spin(0.2, 8, 0.1, 1))])
        c.add("Title template", b.entry("output.charts.title", W.entry("{label}")))
        c.add_row("Axis labels", [b.entry("output.charts.xlabel", W.entry("auto", width=18)),
                                  b.entry("output.charts.ylabel", W.entry("auto", width=18))])
        c.add_row("Smoothing (frames)", [b.spin_int("output.charts.smoothing", W.spin(0, 100000)),
                                         b.check("output.charts.show_target", W.check("show target line"))])
        c.add("File prefix", b.entry("output.charts.prefix", W.entry()))
        box.pack_start(W.frame("Time charts", c), False, False, 0)

        p = W.Form()
        p.add_wide(b.check("output.pdb.enabled", W.check("Export frames as PDB")))
        p.add("Frames", b.entry("output.pdb.frames", W.entry("e.g. 0, 10, 20-50:10, last")))
        p.add("Times", b.entry("output.pdb.times", W.entry("e.g. 1.5ns, 2000ps")),
              "Nearest frame is used; unitless values use the display time unit")
        p.add("Atom subset", b.entry("output.pdb.selection", W.entry("all atoms; or e.g. Aptamer, Target")),
              "Group names (comma separated) or an MDAnalysis selection")
        p.add("File prefix", b.entry("output.pdb.prefix", W.entry()))
        row = Gtk.Box(spacing=6)
        eb = W.button("Export PDBs now", self.export_pdbs, icon="document-save-symbolic")
        rb = W.button("Re-render video + charts from cache", self.rerender_from_cache,
                      "Uses the last cached analyses (no trajectory access); handy after changing "
                      "colours, layout or chart style")
        self.busy_widgets += [eb, rb]
        row.pack_start(eb, False, False, 0)
        row.pack_start(rb, False, False, 0)
        p.add_wide(row)
        box.pack_start(W.frame("Frame PDBs", p), False, False, 0)
        return W.scrolled(box)

    # ============================================================ model sync
    def load_project(self, project: Project, path: Optional[str] = None,
                     state: Optional[DiskState] = None) -> None:
        """Show ``project`` (read from ``path``; ``state``: the file as read)."""
        self.project = project
        self.project_path = path
        self._loading = True
        try:
            self.binder.load(project)
        finally:
            self._loading = False
        self._forget_chain_requests()
        self._on_scale()
        self._update_density_status()
        self.header.set_subtitle(path or "unsaved project")
        self._prefill_conda()
        self.refresh_canvas()
        self.runtime_problems = []
        self.check_problems()
        self._remember_file(path, state)

    def _remember_file(self, path: Optional[str], state: Optional[DiskState] = None) -> None:
        """What the project file holds now (``state``, else read here), to notice RNAdo's saves."""
        self._group_renames = []
        self._offered = None
        self._disk = None
        if path and os.path.isfile(path):
            try:
                self._disk = state or DiskState.read(path)[0]
            except (OSError, ValueError):
                self._disk = None
        try:
            self._mem_other = other_data(self.collect(validate=False))
        except ValueError:
            self._mem_other = other_data(self.project)
        self.change_bar.hide()
        self.watch.watch(path if self._disk is not None else None)

    def _prefill_conda(self) -> None:
        """Suggest an AmberTools environment when cpptraj is not on PATH."""
        c = self.project.cleaning
        if self.conda_entry.get_text().strip() or shutil.which(c.cpptraj_path or "cpptraj"):
            return
        env = conda.detect_ambertools_env()
        if env is not None:
            self.conda_entry.set_text(env.name)
            self.append_log(f"cpptraj is not on PATH; pre-filled conda environment {env.name!r} "
                            f"({env.prefix}) - it is activated for every cpptraj call.")

    def detect_conda(self) -> None:
        envs = conda.find_cpptraj_envs()
        if not envs:
            roots = ", ".join(conda.candidate_roots()) or "none found"
            self.info("No conda environment with cpptraj found",
                      f"Searched installations: {roots}. Enter the environment name or path manually.")
            return
        self.conda_entry.set_text(envs[0].name)
        self.append_log("Conda environments with cpptraj: " +
                        ", ".join(f"{e.name} ({e.prefix})" for e in envs))

    def collect(self, validate: bool = True, include_layout: bool = True, **uses: bool) -> Project:
        """The project as the side bar shows it; ``validate`` refuses the errors of what is about to run,
        which ``uses`` names as :meth:`Project.problems` does (default: the output switches)."""
        project = self.project.copy()
        project.workflow = self.project.workflow  # the plan (RNAdo's) as read; the same object
        errors = self.binder.store(project, skip=() if include_layout else ("layout",))
        if errors:
            raise ValueError("\n".join(errors))
        if validate:
            problems = project.validate(**uses)
            if problems:
                self.check_problems()  # the fields are flagged while the message is shown
                raise ValueError("\n".join(problems))
        return project

    # ================================================================ actions
    def new_project(self):
        self.preview_data = None
        self.load_project(Project(), None)
        self.append_log("New project with default settings (unsaved).")

    def open_project(self):
        path = choose_project(self)
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        from ..projectfile import add_recent, is_params_log

        try:
            state, project = DiskState.read(path)
        except (OSError, ValueError) as exc:
            self.error(f"Could not open {path}", str(exc))
            return
        replay = is_params_log(path)
        self.load_project(project, None if replay else path, None if replay else state)
        if replay:
            self.append_log(f"Loaded the settings of the run logged in {path} (save them as a project to "
                            "keep them)")
        else:
            add_recent(path)
            self.append_log(f"Opened project {path}")

    def save_project(self, save_as: bool = False) -> bool:
        """Save the project.  When RNAdo (or the command line) saved the file since this window read it,
        the plan found there is saved, with this window's group renames applied to it: after any question
        the file is re-read and written under its lock, so a plan saved meanwhile is not lost."""
        from ..projectfile import add_recent, locked

        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Some settings are invalid", str(exc))
            return False
        path = self.project_path
        if save_as or not path:
            name = project.workflow.project_name if project.workflow.project_name != "project" else PROJECT_ABBREV
            path = choose_save_path(self, name)
            if not path:
                return False
        into_file = self._disk is not None and os.path.realpath(path) == os.path.realpath(self._disk.path)
        agreed = self._offered  # the other program's analysis settings that saving replaces (the bar)
        for _attempt in range(SAVE_ATTEMPTS):
            newer = self._newer_settings(agreed)
            if newer is not None:  # another program saved other analysis settings since this window read them
                state, disk = newer
                if not self.analysis_changed():  # saving would only put back the older ones: take theirs
                    self._take_disk(state, disk)
                    if not into_file:
                        continue  # the copy gets the newest settings
                    self.append_log(f"Nothing to save: reloaded the analysis settings another program saved "
                                    f"in {state.path}")
                    return True
                if into_file:
                    response = ask(self, "Another program changed the analysis settings in this project file",
                                   f"{state.path} was saved with other analysis settings after this window "
                                   "read it. Replace them with yours?",
                                   [("Cancel", Gtk.ResponseType.CANCEL),
                                    ("Replace them", Gtk.ResponseType.ACCEPT)],
                                   Gtk.ResponseType.CANCEL, Gtk.MessageType.WARNING)
                    if response != Gtk.ResponseType.ACCEPT:
                        return False
                agreed = state.other
            try:  # after the question: the settings as they are now (it may have taken RNAdo's plan)
                project = self.collect(validate=False)
            except ValueError as exc:
                self.error("Some settings are invalid", str(exc))
                return False
            try:
                with locked(path):
                    if not self._take_newer_plan(project, agreed):
                        continue  # their analysis settings changed again: ask about the new ones
                    project.save(path)
                    state = DiskState.read(path)[0]
            except (OSError, ValueError) as exc:
                self.error("Could not save the project", str(exc))
                return False
            break
        else:
            self.error("Could not save the project", f"Other programs kept saving {path}; try again.")
            return False
        self.project = project
        self.project_path = path
        self.header.set_subtitle(path)
        self._remember_file(path, state)
        add_recent(path)
        self.append_log(f"Saved project {path}")
        return True

    # ------------------------------------------------------------ co-editing with RNAdo
    def analysis_changed(self) -> bool:
        """The analysis settings have edits that are not saved."""
        try:
            return other_data(self.collect(validate=False)) != self._mem_other
        except ValueError:  # an invalid field: an edit
            return True

    def _newer_settings(self, agreed: Optional[Dict]) -> Optional[Tuple[DiskState, Project]]:
        """The file (its state and project) when another program saved analysis settings in it that are
        neither the ones this window read nor ``agreed`` (those saving replaces); else None."""
        disk_state = self._disk
        if disk_state is None or not disk_state.changed():
            return None
        try:
            state, disk = DiskState.read(disk_state.path)
        except (OSError, ValueError):  # reported when the save reads it again
            return None
        return (state, disk) if state.other not in (disk_state.other, agreed) else None

    def _take_newer_plan(self, project: Project, agreed: Optional[Dict]) -> bool:
        """Under the file's lock, before a save: put the plan RNAdo saved meanwhile into ``project`` (with
        this window's group renames).  False when the file has analysis settings that are neither the
        ones this window read nor ``agreed``: another program saved them after the question."""
        disk_state = self._disk
        if disk_state is None or not disk_state.changed():
            return True
        try:
            state, disk = DiskState.read(disk_state.path)
        except (OSError, ValueError) as exc:
            self.append_log(f"WARNING: could not re-read {disk_state.path} ({exc}); its plan is not merged")
            return True
        if state.other not in (disk_state.other, agreed):
            return False
        if state.workflow != disk_state.workflow:
            project.workflow = renamed_workflow(disk.workflow, self._group_renames)
            self.append_log(f"Kept the plan RNAdo saved in {os.path.basename(state.path)}"
                            + (" (with this window's group renames)" if self._group_renames else ""))
        return True

    def check_disk(self) -> None:
        """The project file changed on disk (focus, file monitor): take RNAdo's plan; reload the analysis
        settings another program changed when there are no unsaved edits here, otherwise offer to."""
        disk_state = self._disk
        if disk_state is None or not disk_state.changed():
            return
        try:
            state, disk = DiskState.read(disk_state.path)
        except (OSError, ValueError) as exc:  # e.g. half-written by a text editor: look again later
            log.warning("could not re-read %s: %s", disk_state.path, exc)
            return
        self._take_disk(state, disk)

    def _take_disk(self, state: DiskState, disk: Project) -> None:
        """Take what another program saved (the file's ``state``, read as ``disk``): RNAdo's plan, and the
        analysis settings when there are no unsaved edits here, otherwise offer to reload them."""
        disk_state = self._disk
        if state.workflow != disk_state.workflow:
            self.project.workflow = renamed_workflow(disk.workflow, self._group_renames)
            self.append_log(f"RNAdo saved the plan in {os.path.basename(state.path)}: taken over")
        if state.other == disk_state.other:
            self._disk = state
            return
        if not self.analysis_changed():
            renames = self._group_renames
            disk.workflow = self.project.workflow
            self.load_project(disk, state.path, state)
            self._group_renames = renames
            self.append_log(f"Reloaded the analysis settings another program saved in {state.path}")
            return
        # both changed: the stamp is taken (no new question at every focus), the old settings stay the base
        self._disk = DiskState(state.path, state.stamp, state.workflow, disk_state.other)
        self._offered = state.other  # saving replaces them (the bar says so)
        self.change_bar.label.set_text(f"Another program saved other analysis settings in "
                                       f"{os.path.basename(state.path)}. Reload them (your unsaved changes "
                                       "are lost), or keep yours (saving replaces them)?")
        self.change_bar.show()

    def _on_change_bar(self, response: int) -> None:
        self.change_bar.hide()
        if self._disk is None:
            return
        try:
            state, disk = DiskState.read(self._disk.path)
        except (OSError, ValueError) as exc:
            self.error(f"Could not read {self._disk.path}", str(exc))
            return
        if response == 1:  # reload: the file as it is; this window's group renames go with its other edits
            self.load_project(disk, state.path, state)
            self.append_log(f"Reloaded the analysis settings from {state.path}")
        elif response == 2:  # keep: theirs becomes the base, so saving replaces it without asking again
            if state.workflow != self._disk.workflow:
                self.project.workflow = renamed_workflow(disk.workflow, self._group_renames)
            self._disk = state
            self._offered = None
            self.append_log("Kept your analysis settings; saving replaces the other program's")

    def show_project_log(self):
        from ..projectfile import read_log

        if not self.project_path:
            self.info("The project has no file yet", "Save the project; the steps you run are then logged in it.")
            return
        entries = read_log(self.project_path)
        lines = []
        for e in entries:
            when = e.get("time")
            lines.append(f"{when}  {e.get('step', '?')}  [{e.get('status', '?')}]  {e.get('title', '')}")
            if e.get("summary"):
                lines.append(f"    {e['summary']}")
        self._show_text("Project log", "\n".join(lines) or "Nothing has been run in this project yet.")

    def _show_text(self, title: str, text: str) -> None:
        dlg = Gtk.Dialog(title=title, transient_for=self, modal=True)
        dlg.add_button(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)
        view = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.get_buffer().set_text(text)
        sw = Gtk.ScrolledWindow()
        sw.set_min_content_width(720)
        sw.set_min_content_height(420)
        sw.add(view)
        dlg.get_content_area().pack_start(sw, True, True, 0)
        dlg.show_all()
        dlg.run()
        dlg.destroy()

    def create_demo(self):
        dlg = Gtk.FileChooserDialog(title="Folder for the synthetic demo system", transient_for=self,
                                    action=Gtk.FileChooserAction.SELECT_FOLDER)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Create", Gtk.ResponseType.OK)
        resp = dlg.run()
        folder = dlg.get_filename()
        dlg.destroy()
        if resp != Gtk.ResponseType.OK or not folder:
            return

        def job(rep):
            from ..demo import build_demo
            rep.log(f"Writing synthetic RNA aptamer/target system to {folder} ...")
            return build_demo(folder)

        def done(result):
            self.append_log(f"Demo system ready: {result['project']}")
            self.open_path(result["project"])

        self.start_job("Creating demo", job, done)

    def open_in_rnado(self, tool: Optional[str] = None) -> bool:
        """'Open in RNAdo…': save the project if needed (asks first), then start RNAdo on its file as a
        program of its own, with ``tool`` ("wizard", "pdb" or "builder") open as well."""
        from .launch import launch_rnado, watch_start

        path = self.project_path
        what = {"wizard": "the script wizard", "pdb": "the PDB clean-up",
                "builder": "the structure builder"}.get(tool or "", "the plan")
        if not path:
            response = ask(self, "Save the project for RNAdo?",
                           f"RNAdo opens {what} on the project file; this project has not been saved yet.",
                           [("Cancel", Gtk.ResponseType.CANCEL),
                            ("Open RNAdo without it", Gtk.ResponseType.REJECT),
                            ("Save…", Gtk.ResponseType.ACCEPT)], Gtk.ResponseType.ACCEPT)
            if response == Gtk.ResponseType.ACCEPT:
                if not self.save_project():
                    return False
                path = self.project_path
            elif response != Gtk.ResponseType.REJECT:
                return False
        elif self.analysis_changed():
            response = ask(self, "Save the changes first?",
                           f"RNAdo reads {os.path.basename(path)}; it does not see the unsaved changes of "
                           "this window (they stay here, and saving later keeps the plan RNAdo saves).",
                           [("Cancel", Gtk.ResponseType.CANCEL), ("Don't save", Gtk.ResponseType.REJECT),
                            ("Save", Gtk.ResponseType.ACCEPT)], Gtk.ResponseType.ACCEPT)
            if response == Gtk.ResponseType.ACCEPT:
                if not self.save_project():
                    return False
            elif response != Gtk.ResponseType.REJECT:
                return False
        try:
            proc = launch_rnado(path, tool)
        except OSError as exc:
            self.error("Could not start RNAdo", str(exc))
            return False
        started = f"Started RNAdo{' (' + what + ')' if tool else ''}" + (f" on {path}" if path else "")
        # logged once RNAdo still runs a moment later; an error status before that is shown instead
        # the watch outlives this window when it is closed within the moment it takes
        watch_start(proc, lambda: None if self._destroyed else self.append_log(started),
                    lambda why: None if self._destroyed else self.error("RNAdo did not start", why))
        return True

    def use_plan_inputs(self, node_id: Optional[str] = None) -> bool:
        """'Use the plan's files as inputs': the topology and trajectories that the plan's MDATK analysis
        step reads (the plan RNAdo saved last; asks which step when there are several)."""
        from ..workflows.plan import Plan, analysis_inputs, step_title

        self.check_disk()  # the newest plan RNAdo saved
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project
        boxes = [n for n in project.workflow.nodes if n.tool == "mdatk"]
        if node_id is not None:
            boxes = [n for n in boxes if n.id == node_id]
        if not boxes:
            self.error("The plan has no MDATK analysis step",
                       "Add an 'MDATK analysis' step in RNAdo (Open in RNAdo…), connect a topology and "
                       "trajectories to it and save the plan.")
            return False
        node = boxes[0]
        if len(boxes) > 1:
            node = self._choose_box(boxes)
            if node is None:
                return False
        plan = Plan(project.workflow, project.workflow_dir, project.groups)
        top, trajs = analysis_inputs(plan, node.id)
        if not top:
            self.error("No topology", f"{step_title(node)}: connect a topology (and trajectories) to the "
                                      "analysis step in RNAdo first.")
            return False
        self.apply_inputs(top, trajs, bool(plan.steps[node.id].settings.get("clean", True)))
        return True

    def _choose_box(self, boxes):
        from ..workflows.plan import step_title

        dlg = Gtk.Dialog(title="Which analysis step?", transient_for=self, modal=True)
        dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Use its files", Gtk.ResponseType.OK)
        combo = W.combo([(n.id, step_title(n)) for n in boxes])
        combo.set_active(0)
        box = dlg.get_content_area()
        box.set_border_width(12)
        box.set_spacing(6)
        box.pack_start(W.label("The plan has several MDATK analysis steps:"), False, False, 0)
        box.pack_start(combo, False, False, 0)
        dlg.show_all()
        ok = dlg.run() == Gtk.ResponseType.OK
        chosen = combo.get_active_id()
        dlg.destroy()
        return next((n for n in boxes if n.id == chosen), None) if ok else None

    def apply_inputs(self, topology: str, trajectories: List[str], clean: bool = True) -> None:
        """Use files of the plan as this project's analysis inputs."""
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project.copy()
            project.workflow = self.project.workflow
        project.inputs.topology = topology
        project.inputs.trajectories = list(trajectories)
        if clean:
            project.cleaning.enabled = True
        offered = self._chains_offered
        # unsaved edits, not a reload: what the file held and a pending question about it stay as they were
        kept = self._disk, self._mem_other, self._group_renames, self._offered, self.change_bar.get_visible()
        self.load_project(project, self.project_path, self._disk)
        self._disk, self._mem_other, self._group_renames, self._offered, bar = kept
        self.change_bar.set_visible(bar)
        self._chains_offered = offered  # the same project with a new topology: offer its chains
        self._on_topology_changed()  # chain groups for empty or untouched default groups
        missing = [p for p in [topology] + list(trajectories) if not os.path.exists(p)]
        self.append_log(f"Analysis inputs from the plan: {topology} + {len(trajectories)} trajectory "
                        f"file(s)" + (" (not created yet - run the plan's scripts first)" if missing else ""))

    def _citation_resources(self):
        from ..citations import project_resources
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project
        return project, project_resources(project)

    def export_citations(self):
        project, resources = self._citation_resources()
        base = os.path.dirname(self.project_path) if self.project_path else project.out_dir
        export_bibtex(self, resources, base, project.workflow.project_name or PROJECT_ABBREV, self.append_log,
                      self.info, self.error)

    def citation_text(self) -> str:
        return citation_text(self._citation_resources()[1])

    def show_about(self):
        dlg = Gtk.Dialog(title=f"About {PROJECT_ABBREV}", transient_for=self, modal=True)
        dlg.add_buttons("Export BibTeX…", 1, "Close", Gtk.ResponseType.CLOSE)
        dlg.set_default_size(760, 560)
        box = dlg.get_content_area()
        box.set_spacing(6)
        box.set_border_width(12)
        box.pack_start(W.label(f"<big><b>{GLib.markup_escape_text(PROJECT_TITLE)}</b></big>  {__version__}",
                               markup=True), False, False, 0)
        box.pack_start(W.label("Offline analysis and multi-panel video rendering of molecular dynamics "
                               "trajectories of RNA systems. RNAdo, MDATK's workflow planner, writes and "
                               "runs the scripts for structure prediction and simulation. Licensed under the "
                               "GNU GPL v3.", wrap=True), False, False, 0)
        view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD)
        view.get_buffer().set_text(self.citation_text())
        sw = Gtk.ScrolledWindow()
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.add(view)
        box.pack_start(sw, True, True, 0)
        dlg.show_all()
        response = dlg.run()
        dlg.destroy()
        if response == 1:
            self.export_citations()

    def open_output_dir(self):
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project
        path = project.out_dir
        if not os.path.isdir(path):
            self.error("The output directory does not exist yet.", path)
            return
        try:
            how = desktop.open_folder(path)
            self.append_log(f"Opened {path} ({how})")
        except (OSError, RuntimeError, GLib.Error) as exc:
            self.error("Could not open the folder", str(exc))

    def open_video(self):
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project
        path = self.last_video if self.last_video and os.path.isfile(self.last_video) else \
            desktop.latest_video(project.out_dir, project.output.video.filename)
        if not path:
            self.error("No video has been rendered yet.", project.out_dir)
            return
        try:
            desktop.open_file(path)
            self.append_log(f"Opened {path}")
        except (OSError, GLib.Error) as exc:
            self.error("Could not open the video", str(exc))

    # --------------------------------------------------------------- jobs
    def _pipeline(self, project, rep):
        from ..pipeline import Pipeline
        return Pipeline(project, rep, project_path=self.project_path)

    def render(self):
        try:
            project = self.collect()
        except ValueError as exc:
            self.error("Please fix these parameters before rendering", str(exc))
            return

        def job(rep):
            return self._pipeline(project, rep).run()

        def done(outputs):
            self._adopt_auto_fit(project)
            s = outputs.summary()
            if s["video"]:
                self.last_video = s["video"]
            lines = []
            if s["video"]:
                lines.append(f"Video: {s['video']}")
            if s["charts"]:
                lines.append(f"Charts: {len(s['charts'])} file(s)")
            if s["pdbs"]:
                lines.append(f"PDB files: {len(s['pdbs'])}")
            if s["params_log"]:
                lines.append(f"Parameter log: {s['params_log']}")
            self.info("Render finished", "\n".join(lines) or "Nothing was enabled.")

        self.start_job("Rendering", job, done)

    def run_cleaning(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return
        project.cleaning.enabled = True

        def job(rep):
            return self._pipeline(project, rep).resolve_inputs()

        def done(inputs):
            c = inputs.cleaned
            self.info("Cleaning finished" if not c.reused else "Cleaned files reused",
                      f"Topology: {inputs.topology}\nTrajectory: {inputs.trajectories[0]}\n"
                      f"cpptraj script: {c.script}")

        self.start_job("Cleaning", job, done)

    def inspect_system(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return

        def job(rep):
            from ..io.loader import load_universe, system_summary
            from ..selections import select
            pipe = self._pipeline(project, rep)
            inputs = pipe.cleaned_if_available()
            u = load_universe(inputs.topology, inputs.trajectories, project.inputs.dt_ps)
            info = system_summary(u)
            rep.log(f"System ({'cleaned' if inputs.topology != project.inputs.topology else 'raw'} "
                    f"files): {json.dumps(info)}")
            for g in project.groups:
                try:
                    ag = select(u, f"group:{g.name}", project.groups)
                    rep.log(f"  group {g.name}: {ag.n_atoms} atoms, {ag.residues.n_residues} residues "
                            f"({ag.residues[0].resname}{ag.residues[0].resid}–"
                            f"{ag.residues[-1].resname}{ag.residues[-1].resid})")
                except Exception as exc:  # noqa: BLE001
                    rep.log(f"  group {g.name}: ERROR {exc}")
            return info

        self.start_job("Inspecting", job, lambda _r: None)

    def load_preview(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return

        def job(rep):
            return self._pipeline(project, rep).structure_preview()

        def done(data):
            self.preview_data = data
            self.refresh_canvas()
            self.append_log(f"Reference frame loaded: {len(data['coords'])} residues "
                            f"({'cleaned' if data['cleaned'] else 'raw'} files).")

        self.start_job("Loading preview", job, done)

    def preview_frame(self):
        try:
            project = self.collect(video=True, charts=False, pdb=False)  # renders the structure view
        except ValueError as exc:
            self.error("Please fix these parameters first", str(exc))
            return
        pos = int(self.preview_pos.get_value())

        def job(rep):
            pipe = self._pipeline(project, rep)
            ds = pipe.analyze(pipe.required_analyses(video=True, charts=False),
                              reporter=rep.sub(0, 0.9))
            path = os.path.join(tempfile.gettempdir(), f"{PROJECT_ABBREV}_still_{os.getpid()}.png")
            return pipe.render_preview(ds, min(pos, ds.n_frames - 1), path)

        def done(path):
            self._adopt_auto_fit(project)
            self.show_image(path)

        self.start_job("Rendering still", job, done)

    def export_pdbs(self):
        try:
            project = self.collect(video=False, charts=False, pdb=True)
        except ValueError as exc:
            self.error("Please fix these parameters first", str(exc))
            return

        def job(rep):
            os.makedirs(project.out_dir, exist_ok=True)
            return self._pipeline(project, rep).export_pdbs()

        self.start_job("Exporting PDBs", job,
                       lambda paths: self.info("PDB export finished", "\n".join(paths)))

    def rerender_from_cache(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return

        def job(rep):
            pipe = self._pipeline(project, rep)
            os.makedirs(project.out_dir, exist_ok=True)
            ds = pipe.load_cached_dataset()
            out = []
            if project.output.video.enabled:
                out.append(pipe.render_video(ds, rep.sub(0, 0.85)))
            if project.output.charts.enabled:
                out += pipe.render_charts(ds, rep.sub(0.85, 1.0))
            return out

        def done(paths):
            self._adopt_auto_fit(project)
            self.info("Re-render finished", f"{len(paths)} file(s) written")

        self.start_job("Re-rendering from cache", job, done)

    # ------------------------------------------------------------ job runner
    def start_job(self, title: str, job, on_success):
        if self.jobs.running:
            self.error("Another job is still running.")
            return
        if self.runtime_problems:  # the last job's problems: this job tells anew
            self.runtime_problems = []
            self.check_problems()
        # a selection error flags its field (and page)
        self.jobs.start(title, job, on_success, on_failure=self._note_failure)

    def cancel(self):
        self.jobs.cancel()

    def _on_close(self, *_):
        self.jobs.stop()
        return False

    # -------------------------------------------------------------- feedback
    def append_log(self, text: str):
        buf = self.log_buffer
        buf.insert(buf.get_end_iter(), text + "\n")
        lines = buf.get_line_count()
        if lines > 6000:
            buf.delete(buf.get_start_iter(), buf.get_iter_at_line(lines - 5000))
        buf.move_mark(self.log_end, buf.get_end_iter())
        self.log_view.scroll_mark_onscreen(self.log_end)

    def error(self, text: str, detail: str = ""):
        self.append_log(f"ERROR: {text}{(': ' + detail) if detail else ''}")
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.ERROR,
                                buttons=Gtk.ButtonsType.OK, text=text)
        if detail:
            dlg.format_secondary_text(detail[:3000])
        dlg.run()
        dlg.destroy()

    def info(self, text: str, detail: str = ""):
        self.append_log(f"{text}{(': ' + detail) if detail else ''}")
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.INFO,
                                buttons=Gtk.ButtonsType.OK, text=text)
        if detail:
            dlg.format_secondary_text(detail[:3000])
        dlg.run()
        dlg.destroy()

    def show_image(self, path: str):
        dlg = Gtk.Dialog(title=f"{PROJECT_ABBREV} preview – {os.path.basename(path)}",
                         transient_for=self, modal=False)
        dlg.add_button("Close", Gtk.ResponseType.CLOSE)
        from gi.repository import GdkPixbuf
        pix = GdkPixbuf.Pixbuf.new_from_file(path)
        screen_w = 1200
        if pix.get_width() > screen_w:
            h = int(pix.get_height() * screen_w / pix.get_width())
            pix = pix.scale_simple(screen_w, h, GdkPixbuf.InterpType.BILINEAR)
        dlg.get_content_area().pack_start(Gtk.Image.new_from_pixbuf(pix), True, True, 0)
        dlg.connect("response", lambda d, _r: d.destroy())
        dlg.show_all()
        self.append_log(f"Still frame written to {path}")


class MDATKApplication(Gtk.Application):
    def __init__(self, preset: Optional[str] = None):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.preset = preset  # project file named on the command line (nothing is opened otherwise)
        self.window: Optional[MainWindow] = None

    def do_activate(self):
        if self.window is None:
            project, path = None, None
            if self.preset:
                try:
                    project, path = Project.load(self.preset), self.preset
                    from ..projectfile import is_params_log
                    if is_params_log(path):  # replay a run: Save must not overwrite the log
                        path = None
                except (OSError, ValueError) as exc:
                    print(f"warning: could not load {self.preset}: {exc}", file=sys.stderr)
            self.window = MainWindow(self, project, path)
            logging.getLogger("mdatk").addHandler(GuiLogHandler(self.window))
        self.window.show_all()
        self.window.present()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    warnings.filterwarnings("ignore", module="MDAnalysis")
    preset = argv[0] if argv else None
    app = MDATKApplication(preset)
    return app.run([sys.argv[0]])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
