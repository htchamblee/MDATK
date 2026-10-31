"""Project files in the analysis window and in RNAdo: the open drop-down with recent projects, the file
choosers, and co-editing - both programs may have the same project file open.

RNAdo owns the file's ``workflow`` section (the plan) and writes only that part
(:func:`mdatk.projectfile.edit_workflow`); the analysis window owns the rest and, when the file changed
on disk since it last read it, saves the plan it finds there.  Each window remembers what it last read
(:class:`DiskState`), notices a newer file when it gets the focus or the file monitor fires
(:class:`FileWatch`), takes the other program's part and offers to reload its own when both changed.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gio, GLib, Gtk  # noqa: E402

from ..config import Project, ResidueGroup, _to_plain  # noqa: E402
from . import widgets as W  # noqa: E402

PROJECT_FILES = ("MDATK projects", ["*.mdatk.toml"])
RUN_LOG_FILES = ("Parameter logs of runs", ["*_params_log.toml"])
WATCH_DELAY_MS = 300  # file monitor events within this time are checked once


# ---------------------------------------------------------------------------
# what a window last read
# ---------------------------------------------------------------------------


def workflow_data(project: Project) -> Dict[str, Any]:
    """The plan as plain data (to compare two versions)."""
    return _to_plain(project.workflow)


def other_data(project: Project) -> Dict[str, Any]:
    """Everything but the plan as plain data."""
    data = project.to_dict()
    data.pop("workflow", None)
    return data


def file_stamp(path: Optional[str]) -> Optional[Tuple[int, int]]:
    """(modification time, size) of ``path``, None when it does not exist."""
    try:
        st = os.stat(path) if path else None
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size) if st is not None else None


@dataclass
class DiskState:
    """The project file as a window last read or wrote it: its stamp, the plan and the other settings."""

    path: str
    stamp: Optional[Tuple[int, int]]
    workflow: Dict[str, Any]
    other: Dict[str, Any]

    @classmethod
    def of(cls, path: str, project: Project, stamp: Optional[Tuple[int, int]] = None) -> "DiskState":
        """The state of ``path`` holding ``project`` (as loaded with Project.load)."""
        return cls(path, stamp if stamp is not None else file_stamp(path), workflow_data(project),
                   other_data(project))

    @classmethod
    def read(cls, path: str) -> Tuple["DiskState", Project]:
        """Read ``path`` (the stamp first: a write during the read shows as a change next time)."""
        stamp = file_stamp(path)
        project = Project.load(path)
        return cls.of(path, project, stamp), project

    def changed(self) -> bool:
        """The file is not the one last read or written (another program saved it, or the step log grew)."""
        return file_stamp(self.path) != self.stamp


def renamed_workflow(workflow, renames: Sequence[Tuple[str, str, Sequence[str]]]):
    """A copy of ``workflow`` with residue-group renames applied: (old, new, the group names before the
    rename), as :meth:`Project.rename_group` changes every reference.  The analysis window replays its
    renames on a plan RNAdo saved meanwhile, so neither side's edit is lost."""
    wf = copy.deepcopy(workflow)
    for old, new, names in renames:
        tmp = Project(groups=[ResidueGroup(name=n) for n in names], workflow=wf)
        try:
            tmp.rename_group(old, new)
        except (KeyError, ValueError):  # the group or its new name does not fit any more: nothing to follow
            pass
    return wf


class FileWatch:
    """Calls ``on_change()`` on the main loop when ``window`` gets the focus and when the watched file's
    monitor reports a change (debounced).  The callback compares the file with what it last read."""

    def __init__(self, window: Gtk.Window, on_change: Callable[[], None]):
        self.on_change = on_change
        self.monitor: Optional[Gio.FileMonitor] = None
        self._timer = 0
        self._closed = False
        window.connect("focus-in-event", lambda *_: self._poke(0) and False)
        window.connect("destroy", lambda *_: self.close())

    def watch(self, path: Optional[str]) -> None:
        if self.monitor is not None:
            self.monitor.cancel()
            self.monitor = None
        if path and not self._closed:
            try:
                gfile = Gio.File.new_for_path(path)
                self.monitor = gfile.monitor_file(Gio.FileMonitorFlags.WATCH_MOVES, None)
                self.monitor.connect("changed", lambda *_: self._poke(WATCH_DELAY_MS))
            except GLib.Error:  # no monitor here (some network file systems): the focus check remains
                self.monitor = None

    def _poke(self, delay: int) -> bool:
        if self._closed:
            return False
        if self._timer:
            GLib.source_remove(self._timer)
        self._timer = GLib.timeout_add(delay, self._fire)
        return False

    def _fire(self) -> bool:
        self._timer = 0
        if not self._closed:
            self.on_change()
        return False

    def close(self) -> None:
        self._closed = True
        if self._timer:
            GLib.source_remove(self._timer)
            self._timer = 0
        if self.monitor is not None:
            self.monitor.cancel()
            self.monitor = None


def change_bar(text: str, buttons: Sequence[Tuple[str, int]],
               on_response: Callable[[int], None]) -> Gtk.InfoBar:
    """An info bar (hidden) that offers to reload what another program changed."""
    bar = Gtk.InfoBar()
    bar.set_message_type(Gtk.MessageType.WARNING)
    bar.label = W.label(text, wrap=True)  # type: ignore[attr-defined]
    bar.get_content_area().add(bar.label)  # type: ignore[attr-defined]
    for label, response in buttons:
        bar.add_button(label, response)
    bar.connect("response", lambda _b, r: on_response(r))
    bar.set_no_show_all(True)
    bar.label.show()  # type: ignore[attr-defined]
    return bar


# ---------------------------------------------------------------------------
# dialogs
# ---------------------------------------------------------------------------


def choose_project(parent: Optional[Gtk.Window], title: str = "Open project",
                   run_logs: bool = True) -> Optional[str]:
    """Ask for a project file to open (or a run's parameter log)."""
    dlg = Gtk.FileChooserDialog(title=title, transient_for=parent, action=Gtk.FileChooserAction.OPEN)
    dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    W.add_filters(dlg, [PROJECT_FILES, RUN_LOG_FILES] if run_logs else [PROJECT_FILES])
    resp = dlg.run()
    path = dlg.get_filename()
    dlg.destroy()
    return path if resp == Gtk.ResponseType.OK and path else None


def choose_save_path(parent: Optional[Gtk.Window], name: str, title: str = "Save project") -> Optional[str]:
    """Ask where to save a project (``name``: the suggested file name); adds the suffix when missing."""
    from ..projectfile import PROJECT_SUFFIX

    dlg = Gtk.FileChooserDialog(title=title, transient_for=parent, action=Gtk.FileChooserAction.SAVE)
    dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dlg.set_do_overwrite_confirmation(True)
    W.add_filters(dlg, [PROJECT_FILES])
    dlg.set_current_name(f"{name}{PROJECT_SUFFIX}")
    resp = dlg.run()
    path = dlg.get_filename()
    dlg.destroy()
    if resp != Gtk.ResponseType.OK or not path:
        return None
    return path if path.endswith(".toml") else path + PROJECT_SUFFIX


def ask(parent: Optional[Gtk.Window], text: str, detail: str, buttons: Sequence[Tuple[str, int]],
        default: int, kind: Gtk.MessageType = Gtk.MessageType.QUESTION) -> int:
    """A modal question with the given (label, response) buttons; returns the response."""
    dlg = Gtk.MessageDialog(transient_for=parent, modal=True, message_type=kind,
                            buttons=Gtk.ButtonsType.NONE, text=text)
    if detail:
        dlg.format_secondary_text(detail)
    for label, response in buttons:
        dlg.add_button(label, response)
    dlg.set_default_response(default)
    response = dlg.run()
    dlg.destroy()
    return response


def ask_unsaved(parent: Optional[Gtk.Window], what: str, detail: str = "") -> str:
    """'save', 'discard' or 'cancel' for unsaved changes (``what``: e.g. "the plan")."""
    response = ask(parent, f"Save the changes to {what}?",
                   detail or "Your changes are lost if you do not save them.",
                   [("Cancel", Gtk.ResponseType.CANCEL), ("Don't save", Gtk.ResponseType.REJECT),
                    ("Save", Gtk.ResponseType.ACCEPT)], Gtk.ResponseType.ACCEPT)
    return {Gtk.ResponseType.ACCEPT: "save", Gtk.ResponseType.REJECT: "discard"}.get(response, "cancel")


def fill_open_menu(menu: Gtk.Menu, open_path: Callable[[str], None],
                   items: Sequence[Tuple[str, Callable[[], None], str]]) -> None:
    """The open drop-down: recent projects, then ``items`` (label, callback, tooltip)."""
    from ..projectfile import recent_projects

    for child in menu.get_children():
        menu.remove(child)
    recent: List[str] = recent_projects()
    if recent:
        head = Gtk.MenuItem(label="Recent projects")
        head.set_sensitive(False)
        menu.append(head)
        for path in recent:
            item = Gtk.MenuItem(label=os.path.basename(path))
            item.set_tooltip_text(path)
            item.connect("activate", lambda _w, p=path: open_path(p))
            menu.append(item)
        menu.append(Gtk.SeparatorMenuItem())
    for text, cb, tip in items:
        item = Gtk.MenuItem(label=text)
        item.set_tooltip_text(tip)
        item.connect("activate", lambda _w, c=cb: c())
        menu.append(item)
    menu.show_all()
