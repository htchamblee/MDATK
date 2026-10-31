"""RNAdo, MDATK's workflow planner: a program of its own.

RNAdo lays out the steps from sequence to MD ensemble on a canvas (the planner), writes their control
files and scripts, and runs them on request; its output log panel shows every line the scripts print,
marked with their step.  The script wizard, the PDB clean-up and the structure builder open from its
header bar.  It works on the same project files (``*.mdatk.toml``) as the MDATK analysis window and
owns their ``workflow`` section: saving writes only that part, so both programs can have a project
open at the same time (see :mod:`mdatk.gui.projectui`).

``rnado [PROJECT] [--open wizard|pdb|builder]`` (or ``python -m mdatk.gui.rnado``) starts it; the
analysis window starts it with "Open in RNAdo…".
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import warnings
from typing import Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, Gio, GLib, Gtk, Pango  # noqa: E402

import mdatk  # noqa: E402

from .. import __version__  # noqa: E402
from ..config import PlanNode, Project  # noqa: E402
from ..workflows.plan import step_title  # noqa: E402
from . import widgets as W  # noqa: E402
from .about import citation_text, export_bibtex  # noqa: E402
from .jobs import JobControl, confirm_run, waiting_text  # noqa: E402
from .logpanel import LogPanel  # noqa: E402
from .planner import PlannerPanel, checkable  # noqa: E402
from .projectui import (SAVE_ATTEMPTS, DiskState, FileWatch, ask, ask_unsaved, change_bar,  # noqa: E402
                        choose_project, choose_save_path, fill_open_menu, group_renames, rename_groups_in,
                        renamed_data, workflow_data)
from .worker import call_soon  # noqa: E402

log = logging.getLogger(__name__)

APP_ID = "org.mdatk.RNAdo"  # differs from the analysis window's org.mdatk.MDATK
TITLE = "RNAdo"
TAGLINE = "MDATK's workflow planner"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
FALLBACK_ICON = "network-workgroup"  # themed icon when the RNAdo assets are missing
LOG_HEIGHT = 230  # initial height of the output log panel
RELOAD, KEEP = 1, 2  # responses of the "file changed" bar


def asset_path(name: str) -> str:
    """A file of RNAdo's branding (``mdatk/assets``); it may be missing."""
    return os.path.join(os.path.dirname(mdatk.__file__), "assets", name)


def icon_pixbufs() -> List[GdkPixbuf.Pixbuf]:
    """RNAdo's icon in the sizes found (the SVG when there are no PNGs; empty when neither exists)."""
    out = []
    for size in ICON_SIZES:
        path = asset_path(f"rnado-icon-{size}.png")
        if os.path.isfile(path):
            try:
                out.append(GdkPixbuf.Pixbuf.new_from_file(path))
            except GLib.Error:
                pass
    if not out and os.path.isfile(asset_path("rnado-icon.svg")):
        try:
            out.append(GdkPixbuf.Pixbuf.new_from_file_at_size(asset_path("rnado-icon.svg"), 256, 256))
        except GLib.Error:  # no SVG loader
            pass
    return out


class LogHandler(logging.Handler):
    """Warnings of the mdatk loggers in the output log (from any thread)."""

    def __init__(self, panel: LogPanel):
        super().__init__(level=logging.WARNING)
        self.panel = panel

    def emit(self, record):
        self.panel.feed(f"{record.levelname}: {record.getMessage()}")


class RNAdoWindow(Gtk.ApplicationWindow):
    """RNAdo's window: header bar, planner, output log panel and status bar.  It is the host
    (:class:`mdatk.gui.host.Host`) of the planner and of the tool windows."""

    def __init__(self, app: Optional[Gtk.Application], project: Optional[Project] = None,
                 project_path: Optional[str] = None):
        super().__init__(application=app, title=TITLE)
        self.set_default_size(1400, 920)
        self.project = project or Project()
        self.project_path = project_path
        self.wizard = None  # the stand-alone script wizard
        self.pdb_tools = None  # PDB clean-up window
        self.builder = None  # structure builder window
        self._disk: Optional[DiskState] = None  # the project file as last read or written
        self._mem_wf: Dict = {}  # the plan as last loaded or saved: unsaved edits differ from it
        self._log_pos = 0  # paned position while the log panel is shown
        self._progress_lock = threading.Lock()
        icons = icon_pixbufs()
        if icons:
            self.set_icon_list(icons)
        else:
            self.set_icon_name(FALLBACK_ICON)
        self._build_header()
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(outer)
        self.change_bar = change_bar("", [("Reload it", RELOAD), ("Keep mine", KEEP)], self._on_change_bar)
        outer.pack_start(self.change_bar, False, False, 0)
        self.log = LogPanel(stored_log=self.stored_log, on_expanded=self._on_log_expanded)
        self.planner = PlannerPanel(self)
        self.planner.on_select = self._on_select
        self.planner.on_plan_changed = self._on_plan_changed
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self.paned.pack1(self.planner, True, False)
        self.paned.pack2(self.log, False, False)
        self.paned.set_position(920 - LOG_HEIGHT - 140)
        outer.pack_start(self.paned, True, True, 0)
        outer.pack_start(self._status_bar(), False, False, 0)
        self.jobs = JobControl(self.progress, self.cancel_button, self.append_log,
                               lambda text, detail: self.error(text, detail))
        self.watch = FileWatch(self, self.check_disk)
        self._install_actions()
        self.connect("delete-event", self._on_delete)
        self.connect("destroy", self._on_destroy)
        self._adopt(self.project, project_path)
        self.append_log(f"{TITLE} {__version__} ({TAGLINE}) ready. Everything runs on this computer.")

    # ================================================================ chrome
    def _build_header(self) -> None:
        hb = Gtk.HeaderBar()
        hb.set_show_close_button(True)
        hb.set_title(TITLE)
        self.header = hb
        self.set_titlebar(hb)
        open_box = Gtk.Box()
        open_box.get_style_context().add_class("linked")
        open_box.pack_start(W.button("Open", self.open_project, "Open a project file (Ctrl+O)",
                                     "document-open-symbolic"), False, False, 0)
        self.open_menu = Gtk.Menu()
        self.open_menu_button = Gtk.MenuButton()
        self.open_menu_button.set_image(Gtk.Image.new_from_icon_name("pan-down-symbolic",
                                                                     Gtk.IconSize.BUTTON))
        self.open_menu_button.set_tooltip_text("Recent projects, or browse")
        self.open_menu_button.set_popup(self.open_menu)
        self.open_menu_button.connect("toggled", lambda b: b.get_active() and self._fill_open_menu())
        open_box.pack_start(self.open_menu_button, False, False, 0)
        hb.pack_start(open_box)
        hb.pack_start(W.button("Save", self.save_project, "Save the plan into the project file (Ctrl+S); the "
                               "analysis settings in the file stay as they are", "document-save-symbolic"))
        menu = Gtk.Menu()
        for text, cb in (("New project", self.new_project),
                         ("Open project…", self.open_project),
                         ("Save", self.save_project),
                         ("Save as…", lambda: self.save_project(True)),
                         (None, None),
                         ("Script wizard…", self.open_wizard),
                         ("PDB clean-up…", self.open_pdb_tools),
                         ("Structure builder…", self.open_builder),
                         (None, None),
                         ("Show the output log", self.show_log),
                         ("Project log…", self.show_project_log),
                         (None, None),
                         ("About RNAdo", self.show_about)):
            if text is None:
                menu.append(Gtk.SeparatorMenuItem())
                continue
            item = Gtk.MenuItem(label=text)
            item.connect("activate", lambda _w, c=cb: c())
            menu.append(item)
        menu.show_all()
        mb = Gtk.MenuButton()
        mb.set_image(Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON))
        mb.set_tooltip_text("New, open, save, tools, About RNAdo")
        mb.set_popup(menu)
        hb.pack_end(mb)
        tools = Gtk.Box()
        tools.get_style_context().add_class("linked")
        for text, cb, tip, icon in (
                ("Script wizard", self.open_wizard, "Write control files and scripts for one program "
                 "(Ctrl+Shift+W)", "document-properties-symbolic"),
                ("PDB clean-up", self.open_pdb_tools, "Repair, rename (D/L-RNA naming schemes), re-chain "
                 "and renumber PDB files", "edit-clear-all-symbolic"),
                ("Structure builder", self.open_builder, "Ideal A-form helices (D / L), base mutations and "
                 "rigid-body fits into a density map", "applications-science-symbolic")):
            tools.pack_start(W.button(text, cb, tip, icon), False, False, 0)
        hb.pack_end(tools)

    def _fill_open_menu(self) -> None:
        fill_open_menu(self.open_menu, self.open_path,
                       [("Browse…", self.open_project, "Choose a project file")])

    def _status_bar(self) -> Gtk.Widget:
        bar = Gtk.Box(spacing=8, margin=4)
        bar.set_margin_start(8)
        bar.set_margin_end(8)
        # the planner's status line: steps, connections, errors, folder
        self.planner.remove(self.planner.status)
        self.planner.status.set_margin_bottom(0)
        self.planner.status.set_margin_top(0)
        self.planner.status.set_margin_start(0)
        self.planner.status.set_ellipsize(Pango.EllipsizeMode.END)
        bar.pack_start(self.planner.status, True, True, 0)
        self.progress = Gtk.ProgressBar(show_text=True)
        self.progress.set_text("Idle")
        self.progress.set_size_request(300, -1)
        self.progress.set_valign(Gtk.Align.CENTER)
        bar.pack_start(self.progress, False, False, 0)
        self.cancel_button = W.button("Cancel", self.cancel, "Stop the running scripts",
                                      "process-stop-symbolic")
        bar.pack_start(self.cancel_button, False, False, 0)
        return bar

    def _install_actions(self) -> None:
        app = self.get_application()
        for name, cb, accels in (("new", self.new_project, ["<Control>n"]),
                                 ("open", self.open_project, ["<Control>o"]),
                                 ("save", self.save_project, ["<Control>s"]),
                                 ("saveas", lambda: self.save_project(True), ["<Control><Shift>s"]),
                                 ("wizard", self.open_wizard, ["<Control><Shift>w"]),
                                 ("log", self.toggle_log, ["<Control>l"])):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, c=cb: c())
            self.add_action(action)
            if app is not None:
                app.set_accels_for_action(f"win.{name}", accels)

    # ================================================================ output log
    def append_log(self, text: str) -> None:
        self.log.feed(text)

    def show_log(self) -> None:
        """Show the output log panel with its live lines (a run starts)."""
        self.log.show_live()
        self.log.set_expanded(True)

    def toggle_log(self) -> None:
        self.log.set_expanded(not self.log.expanded)

    def _on_log_expanded(self, expanded: bool) -> None:
        if expanded:
            height = self.paned.get_allocated_height()
            pos = self._log_pos or (height - LOG_HEIGHT if height > 2 * LOG_HEIGHT else 0)
            if pos:
                self.paned.set_position(pos)
        else:
            self._log_pos = self.paned.get_position()
            self.paned.set_position(10 ** 6)  # as far down as the panel's header allows

    def _on_select(self, node: Optional[PlanNode]) -> None:
        if node is not None and checkable(node):
            self.log.select_step(node.step, step_title(node))
        else:
            self.log.select_step(None)

    def _on_plan_changed(self) -> None:
        self.log.set_steps([(n.step, step_title(n)) for n in self.project.workflow.nodes if checkable(n)])

    def stored_log(self, step: int, title: str) -> str:
        """The step log entry of the last run of ``step`` (the one with this title when there are several)."""
        from ..projectfile import read_log

        if not self.project_path:
            return "The project has not been saved yet, so nothing has been logged."
        entries = [e for e in read_log(self.project_path) if str(e.get("step", "")) == f"{step:02d}"]
        if not entries:
            return ""
        e = ([x for x in entries if x.get("title") == title] or entries)[-1]
        lines = [f"Step {e.get('step')} · {e.get('title', '')} · {e.get('status', '?')} · started "
                 f"{e.get('time', '?')} · {float(e.get('duration_s', 0) or 0):.1f} s"]
        if e.get("command"):
            lines.append(f"$ {e['command']}" + (f"   (in {e['folder']})" if e.get("folder") else ""))
        if e.get("summary"):
            lines.append(str(e["summary"]))
        lines.append("")
        lines += str(e.get("output") or "(no output stored)").split("\n")
        return "\n".join(lines)

    # ================================================================ feedback
    def error(self, text: str, detail: str = "") -> None:
        self.append_log(f"ERROR: {text}{(': ' + detail) if detail else ''}")
        self._message(text, detail, Gtk.MessageType.ERROR)

    def info(self, text: str, detail: str = "") -> None:
        self.append_log(f"{text}{(': ' + detail) if detail else ''}")
        self._message(text, detail, Gtk.MessageType.INFO)

    def _message(self, text: str, detail: str, kind: Gtk.MessageType) -> None:
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=kind, buttons=Gtk.ButtonsType.OK,
                                text=text)
        if detail:
            dlg.format_secondary_text(detail[:3000])
        dlg.run()
        dlg.destroy()

    # ================================================================ project
    def plan_changed(self) -> bool:
        """The plan has edits that are not saved."""
        return workflow_data(self.project) != self._mem_wf

    def _adopt(self, project: Project, path: Optional[str], state: Optional[DiskState] = None) -> None:
        """Make ``project`` (read from ``path``; None: unsaved) RNAdo's project."""
        self.project = project
        self.project_path = path
        self._disk = state
        if state is None and path and os.path.isfile(path):
            try:  # what the file holds (``project`` may already differ from it)
                self._disk = DiskState.read(path)[0]
            except (OSError, ValueError) as exc:
                log.warning("could not read %s: %s", path, exc)
        self.header.set_subtitle(path or "unsaved project")
        self.watch.watch(path)
        self.change_bar.hide()
        self._take_plan(project.workflow)

    def _take_plan(self, workflow) -> None:
        """Show ``workflow`` (loaded or reloaded); it counts as saved."""
        self.project.workflow = workflow
        self.planner.reload()
        if self.wizard is not None:  # the stand-alone wizard keeps its tool, with the new plan's settings
            self.wizard.select_tool(self.wizard.tool.name)
        self._mem_wf = workflow_data(self.project)  # after the planner's own adjustments

    def without_edit(self, change: Callable[[], None]) -> None:
        """Run ``change``, which adjusts the plan without the user's edit (a script wizard opens and fills
        in its step's defaults): a plan without unsaved edits still has none afterwards."""
        unedited = not self.plan_changed()
        change()
        if unedited:
            self._mem_wf = workflow_data(self.project)

    def _take_other(self, disk: Project) -> None:
        """The settings besides the plan as the file has them now (groups and folders the plan uses).
        Residue groups another program renamed are renamed in RNAdo's plan as well, in the plan last
        read (``_disk``) and in the saved one (``_mem_wf``): a plan that differs from the file's only by
        those renames is not a conflict, and saving does not bring the old names back."""
        renames = group_renames(self.project.groups, disk.groups)
        for f in dataclasses.fields(Project):
            if f.name != "workflow":
                setattr(self.project, f.name, getattr(disk, f.name))
        if renames:
            rename_groups_in(self.project.workflow, renames)  # in place: the planner keeps its steps
            self._mem_wf = renamed_data(self._mem_wf, renames)
            if self._disk is not None:
                self._disk.workflow = renamed_data(self._disk.workflow, renames)
            for win in [self.wizard] + list(self.planner.wizards.values()):
                if win is not None:  # their forms show the old names, and a form writes all it shows
                    win.reload()
            self.append_log("Residue groups renamed by another program, followed in the plan: "
                            + ", ".join(f"{old} → {new}" for old, new, _names in renames))
        self.planner.changed()
        if self.wizard is not None:
            self.wizard.schedule_refresh()

    def _confirm_discard(self) -> bool:
        """Before another project replaces this one: save or drop unsaved plan edits (False: cancelled)."""
        if not self.plan_changed():
            return True
        choice = ask_unsaved(self, "the plan")
        if choice == "save":
            return self.save_project()
        return choice == "discard"

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._adopt(Project(), None)
        self.append_log("New project (unsaved).")

    def open_project(self) -> None:
        path = choose_project(self)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> bool:
        from ..projectfile import add_recent, is_params_log

        if not self._confirm_discard():
            return False
        try:
            state, project = DiskState.read(path)
        except (OSError, ValueError) as exc:
            self.error(f"Could not open {path}", str(exc))
            return False
        replay = is_params_log(path)
        self._adopt(project, None if replay else path, None if replay else state)
        if replay:
            self.append_log(f"Loaded the settings of the run logged in {path} (save them as a project to "
                            "keep them)")
        else:
            add_recent(path)
            self.append_log(f"Opened project {path}")
        return True

    def save_project(self, save_as: bool = False) -> bool:
        """Save the plan.  Into the open project file only the plan is written (the analysis window may
        have the other settings open); Save as writes a complete project file."""
        from ..projectfile import add_recent, edit_workflow, locked

        if save_as or not self.project_path or not os.path.isfile(self.project_path):
            name = self.project.workflow.project_name
            path = choose_save_path(self, name if name and name != "project" else "rnado")
            if not path:
                return False
            if self._disk is not None and self._disk.changed():
                self._refresh_other()  # the newest analysis settings of the file this project came from
            try:
                self.project.save(path)
                state, _project = DiskState.read(path)
            except (OSError, ValueError) as exc:
                self.error("Could not save the project", str(exc))
                return False
            self.project_path = path
            self._disk = state
            self._mem_wf = workflow_data(self.project)
            self.header.set_subtitle(path)
            self.watch.watch(path)
            self.change_bar.hide()
            add_recent(path)
            self.append_log(f"Saved project {path}")
            return True
        path = self.project_path
        agreed: Optional[tuple] = None  # (plan, groups) of the file the user chose to replace

        def replace(wf) -> None:
            for f in dataclasses.fields(wf):
                setattr(wf, f.name, getattr(self.project.workflow, f.name))

        for _attempt in range(SAVE_ATTEMPTS):
            try:
                state, disk = DiskState.read(path)
            except (OSError, ValueError) as exc:
                self.error(f"Could not read {path}", str(exc))
                return False
            self._take_newer_other(state, disk)  # renamed groups first: a plan renamed only is no conflict
            base = self._disk.workflow if self._disk is not None else state.workflow
            if state.workflow != base and not self._agreed(agreed, state, disk):
                # another program saved a plan since RNAdo read the file
                if not self.plan_changed():
                    self._take_plan(disk.workflow)
                    self._disk = state
                    self.change_bar.hide()
                    self.append_log(f"Nothing to save: reloaded the plan another program saved in {path}")
                    return True
                choice = ask(self, "Another program changed the plan in this project file",
                             f"{path} holds a plan saved after RNAdo read it. Replace it with yours, or "
                             "reload it (your unsaved changes are lost)?",
                             [("Cancel", Gtk.ResponseType.CANCEL), ("Reload theirs", RELOAD),
                              ("Replace with mine", KEEP)], Gtk.ResponseType.CANCEL, Gtk.MessageType.WARNING)
                if choice == RELOAD:
                    self._reload_from_disk()
                    return False
                if choice != KEEP:
                    return False
                agreed = (state.workflow, disk.groups)
            # after any question: re-read, compare and write under the file's lock, so a plan or settings
            # another program saved while the question was open are not overwritten unseen
            try:
                with locked(path):
                    now, disk = DiskState.read(path)
                    self._take_newer_other(now, disk)
                    base = self._disk.workflow if self._disk is not None else now.workflow
                    if now.workflow != base and not self._agreed(agreed, now, disk):
                        continue  # a newer plan: ask about that one
                    edit_workflow(path, replace)
                    new_state, _project = DiskState.read(path)  # what was merged and written
            except (OSError, ValueError) as exc:
                self.error("Could not save the plan", str(exc))
                return False
            break
        else:
            self.error("Could not save the plan", f"Other programs kept saving {path}; try again.")
            return False
        self._disk = new_state
        self._mem_wf = workflow_data(self.project)
        self.change_bar.hide()
        add_recent(path)
        self.append_log(f"Saved the plan to {path}")
        return True

    # ------------------------------------------------------------ co-editing
    def check_disk(self) -> None:
        """The project file changed on disk (focus, file monitor): take the settings besides the plan,
        reload the plan when it has no unsaved edits here, otherwise offer to (the bar above the plan)."""
        if self._disk is None or not self._disk.changed():
            return
        try:
            state, disk = DiskState.read(self._disk.path)
        except (OSError, ValueError) as exc:  # e.g. a text editor left it half-written: look again later
            log.warning("could not re-read %s: %s", self._disk.path, exc)
            return
        self._take_newer_other(state, disk)
        if state.workflow == self._disk.workflow:  # also when the plan changed only by group renames
            self._disk = state
            return
        if not self.plan_changed():
            self._take_plan(disk.workflow)
            self._disk = state
            self.change_bar.hide()
            self.append_log(f"Reloaded the plan: another program saved {state.path}")
            return
        # both changed: the stamp is taken (no new question at every focus), the old plan stays the base
        self._disk = DiskState(state.path, state.stamp, self._disk.workflow, state.other)
        self.change_bar.label.set_text(f"Another program saved a different plan in "
                                       f"{os.path.basename(state.path)}. Reload it (your unsaved changes to "
                                       "the plan are lost), or keep yours (saving replaces it)?")
        self.change_bar.show()

    def _reload_from_disk(self) -> None:
        try:
            state, disk = DiskState.read(self._disk.path)
        except (OSError, ValueError) as exc:
            self.error(f"Could not read {self._disk.path}", str(exc))
            return
        self._take_other(disk)
        self._take_plan(disk.workflow)
        self._disk = state
        self.change_bar.hide()
        self.append_log(f"Reloaded the plan from {state.path}")

    def _take_newer_other(self, state: DiskState, disk: Project) -> None:
        """Take the settings besides the plan when the file (``state``, read as ``disk``) has new ones."""
        if self._disk is None or state.other == self._disk.other:
            return
        self._take_other(disk)
        self._disk.other = state.other
        self.append_log(f"{os.path.basename(state.path)} changed on disk: took its new analysis settings "
                        "(residue groups, folders)")

    @staticmethod
    def _agreed(agreed: Optional[tuple], state: DiskState, disk: Project) -> bool:
        """The file (``state``, read as ``disk``) holds the plan the user chose to replace (``agreed``:
        that plan and the residue groups then), with the groups renamed since followed in it."""
        if agreed is None:
            return False
        plan, groups = agreed
        return state.workflow == renamed_data(plan, group_renames(groups, disk.groups))

    def _refresh_other(self) -> None:
        try:
            state, disk = DiskState.read(self._disk.path)
        except (OSError, ValueError):
            return
        self._take_newer_other(state, disk)

    def _on_change_bar(self, response: int) -> None:
        if response == RELOAD:
            self._reload_from_disk()
            return
        self.change_bar.hide()
        if response == KEEP and self._disk is not None:
            try:  # their plan becomes the base: saving replaces it without asking again
                state, disk = DiskState.read(self._disk.path)
            except (OSError, ValueError):
                return
            self._take_newer_other(state, disk)
            self._disk = state
            self.append_log("Kept your plan; saving replaces the one the other program saved")

    def show_project_log(self) -> None:
        from ..projectfile import read_log

        if not self.project_path:
            self.info("The project has no file yet",
                      "Save the project; the steps you run are then logged in it.")
            return
        lines = []
        for e in read_log(self.project_path):
            lines.append(f"{e.get('time')}  {e.get('step', '?')}  [{e.get('status', '?')}]  "
                         f"{e.get('title', '')}")
            if e.get("summary"):
                lines.append(f"    {e['summary']}")
        self.log.feed("Project log:\n" + ("\n".join(lines) or "Nothing has been run in this project yet."))
        self.show_log()

    # ================================================================ tools
    def open_wizard(self, tool: Optional[str] = None) -> None:
        from .wizard import WizardWindow

        if self.wizard is None:
            self.wizard = WizardWindow(self, tool=tool)
            self.wizard.connect("destroy", lambda w: self.wizard is w and setattr(self, "wizard", None))
        elif tool:
            self.wizard.select_tool(tool)
        self.wizard.present()

    def open_pdb_tools(self) -> None:
        from .pdbdialog import PDBToolsWindow

        if self.pdb_tools is None:
            self.pdb_tools = PDBToolsWindow(self)
            self.pdb_tools.connect("destroy",
                                   lambda w: self.pdb_tools is w and setattr(self, "pdb_tools", None))
        self.pdb_tools.show_all()
        self.pdb_tools.present()

    def open_builder(self) -> None:
        from .builderdialog import BuilderWindow

        if self.builder is None:
            self.builder = BuilderWindow(self)
            self.builder.connect("destroy", lambda w: self.builder is w and setattr(self, "builder", None))
        self.builder.show_all()
        self.builder.present()

    def open_tool(self, name: str) -> None:
        """``rnado --open NAME``: one of :data:`mdatk.gui.launch.TOOLS`."""
        {"wizard": self.open_wizard, "pdb": self.open_pdb_tools, "builder": self.open_builder}[name]()

    def use_for_analysis(self, topology: str, trajectories: List[str], clean: bool = True) -> None:
        """The analysis window takes these files with 'Use the plan's files'; it reads the saved plan."""
        if self.plan_changed() or not self.project_path:
            choice = ask(self, "Save the plan for the analysis window?",
                         "The MDATK analysis window reads the plan from the project file.",
                         [("Cancel", Gtk.ResponseType.CANCEL), ("Save", Gtk.ResponseType.ACCEPT)],
                         Gtk.ResponseType.ACCEPT)
            if choice != Gtk.ResponseType.ACCEPT or not self.save_project():
                return
        self.info("Files for the MDATK analysis",
                  f"Topology: {topology}\nTrajectories: {len(trajectories)} file(s)"
                  + ("\nCleaning with cpptraj: on" if clean else "") +
                  f"\n\nIn the MDATK analysis window (mdatk-gui {os.path.basename(self.project_path)}) "
                  "choose Input & cleaning → 'Use the plan's files'.")

    # ================================================================ running scripts
    def run_scripts(self, jobs, separate=(), parent: Optional[Gtk.Window] = None, parallel: int = 1,
                    on_state=None, on_finished=None, with_separate=None) -> None:
        """Run generated scripts in the background after a confirmation; their output goes to the log
        panel, each line marked with its step.  ``on_state(job, state)`` and ``on_finished()`` are called on
        the GTK main loop; ``with_separate`` (the jobs including the scripts marked 'run separately') adds
        the choice to run those too."""
        from ..workflows.runner import ScriptFailed, run_jobs, runner_problem

        if not jobs:
            self.error("No scripts to run", "None of the selected steps has a script to run here.")
            return
        problem = runner_problem()  # Windows / no bash: the scripts are written but run elsewhere
        if problem:
            self.info("Scripts written, not run",
                      problem + "\n\nScripts:\n" + "\n".join(j.path for j in jobs[:20]))
            return
        if self.jobs.running:
            self.error("Scripts are still running", "Wait for them to finish, or cancel them first.")
            return
        chosen = confirm_run(parent or self, jobs, separate, self.project_path, parallel, with_separate)
        if chosen is None:
            return
        jobs = chosen
        self.show_log()
        panel, project_path, total = self.log, self.project_path, len(jobs)
        done_count = [0]
        ends = {"ok": "finished", "failed": "failed", "cancelled": "cancelled"}

        def state_cb(job, state: str) -> None:  # the runner's threads: queued in order with the output
            if state in ends:
                panel.feed_script(f"==== step {job.step:02d}: {job.title} - {ends[state]} "
                                  f"({os.path.basename(job.path)})")
            if state in ("ok", "failed", "cancelled", "skipped", "waiting"):
                with self._progress_lock:
                    done_count[0] += 1
                    n = done_count[0]
                call_soon(self.jobs.on_progress, n / total, f"{n} of {total} script(s) done")
            if on_state is not None:
                call_soon(on_state, job, state)

        def job(rep):
            try:
                return run_jobs(jobs, project_path, on_line=panel.feed_script,
                                on_stderr=lambda line: panel.feed_script(line, stderr=True),
                                cancel=rep.cancel_event, parallel=parallel, on_state=state_cb,
                                step_marks=True)
            finally:
                if on_finished is not None:
                    call_soon(on_finished)

        def done(results) -> None:
            counts = {s: sum(1 for r in results if r.status == s) for s in ("ok", "waiting", "skipped")}
            self.append_log(f"Run finished: {counts['ok']} script(s) ok, {counts['waiting']} waiting, "
                            f"{counts['skipped']} not started")
            waiting = waiting_text(results)
            if waiting:  # not a failure: the user runs those scripts, then the step again
                self.info("Waiting for separately run scripts", waiting)
                return
            self.info("Scripts finished", f"{len(results)} script(s) finished")

        def failed(exc: BaseException) -> None:
            self.append_log(f"Run stopped: {exc}")

        at_once = f", up to {parallel} at a time" if parallel > 1 else ""
        logged_in = f"; each run is logged in {os.path.basename(project_path)}" if project_path else ""
        self.append_log(f"Running {total} script(s){at_once}{logged_in}")
        self.jobs.start("Running generated scripts", job, done, on_failure=failed, expected=(ScriptFailed,))

    def cancel(self) -> None:
        self.jobs.cancel()

    # ================================================================ about
    def show_about(self) -> None:
        from ..citations import plan_resources

        dlg = Gtk.Dialog(title=f"About {TITLE}", transient_for=self, modal=True)
        dlg.add_buttons("Export BibTeX…", 1, "Close", Gtk.ResponseType.CLOSE)
        dlg.set_default_size(760, 620)
        box = dlg.get_content_area()
        box.set_spacing(6)
        box.set_border_width(12)
        wordmark = asset_path("rnado-wordmark.png")
        image = None
        if os.path.isfile(wordmark):
            try:
                pix = GdkPixbuf.Pixbuf.new_from_file_at_scale(wordmark, 520, -1, True)
                image = Gtk.Image.new_from_pixbuf(pix)
            except GLib.Error:
                image = None
        if image is not None:
            box.pack_start(image, False, False, 0)
        heading = f"<big><b>{TITLE}</b></big>  {__version__}  ·  {GLib.markup_escape_text(TAGLINE)}"
        box.pack_start(W.label(heading, markup=True, xalign=0.5), False, False, 0)
        box.pack_start(W.label(
            "Plans the steps from sequence to MD ensemble (Rosetta FARFAR2, SimRNA, RNAMake, Amber, GROMACS, "
            "NAMD, VMD, refinement, your own scripts), writes their control files and scripts with "
            "consistent names, and runs them on request. Part of the Molecular Dynamic Analysis Tool Kit "
            "(MDATK); everything runs on this computer. By the MDATK contributors. Licensed under the GNU "
            "GPL v3.", wrap=True), False, False, 0)
        plan = self.planner.plan or self.planner.resolve()
        resources = plan_resources(plan)
        view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD)
        view.get_buffer().set_text(citation_text(resources, "This plan"))
        sw = Gtk.ScrolledWindow()
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.add(view)
        box.pack_start(sw, True, True, 0)
        dlg.show_all()
        response = dlg.run()
        dlg.destroy()
        if response == 1:
            folder = os.path.dirname(self.project_path) if self.project_path else self.project.workflow_dir
            export_bibtex(self, resources, folder, self.project.workflow.project_name or "rnado",
                          self.append_log, self.info, self.error)

    # ================================================================ closing
    def _on_delete(self, *_) -> bool:
        if self.jobs.running:
            response = ask(self, "Scripts are still running", "Closing RNAdo stops them.",
                           [("Cancel", Gtk.ResponseType.CANCEL),
                            ("Stop them and close", Gtk.ResponseType.ACCEPT)],
                           Gtk.ResponseType.CANCEL, Gtk.MessageType.WARNING)
            if response != Gtk.ResponseType.ACCEPT:
                return True
            self.jobs.stop(wait=15.0)
        if self.plan_changed():
            choice = ask_unsaved(self, "the plan")
            if choice == "cancel" or (choice == "save" and not self.save_project()):
                return True
        return False

    def _on_destroy(self, *_) -> None:
        self.jobs.stop()
        for win in (self.wizard, self.pdb_tools, self.builder):
            if win is not None:
                win.destroy()


class RNAdoApplication(Gtk.Application):
    def __init__(self, preset: Optional[str] = None, tool: Optional[str] = None):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.preset = preset  # project file named on the command line
        self.tool = tool  # --open: a tool window to open as well
        self.window: Optional[RNAdoWindow] = None

    def do_activate(self):
        if self.window is None:
            project, path = None, None
            if self.preset:
                from ..projectfile import add_recent, is_params_log
                try:
                    project, path = Project.load(self.preset), os.path.abspath(self.preset)
                    if is_params_log(path):  # replay a run: Save must not overwrite the log
                        path = None
                    else:
                        add_recent(path)
                except (OSError, ValueError) as exc:
                    print(f"warning: could not load {self.preset}: {exc}", file=sys.stderr)
            self.window = RNAdoWindow(self, project, path)
            handler = LogHandler(self.window.log)
            logging.getLogger("mdatk").addHandler(handler)
            self.window.connect("destroy", lambda *_: logging.getLogger("mdatk").removeHandler(handler))
        self.window.show_all()
        self.window.present()
        if self.tool:
            tool, self.tool = self.tool, None
            self.window.open_tool(tool)


def main(argv: Optional[List[str]] = None) -> int:
    """``rnado-gui [PROJECT] [--open wizard|pdb|builder]``."""
    from ..cli import rnado_gui_parser

    args = rnado_gui_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    warnings.filterwarnings("ignore", module="MDAnalysis")
    GLib.set_application_name(TITLE)
    icons = icon_pixbufs()
    if icons:  # every window of this program (the tool windows too)
        Gtk.Window.set_default_icon_list(icons)
    else:
        Gtk.Window.set_default_icon_name(FALLBACK_ICON)
    app = RNAdoApplication(args.project, args.open)
    return app.run([sys.argv[0]])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
