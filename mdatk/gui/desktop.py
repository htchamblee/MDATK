"""Open folders and files with the user's desktop applications, and put MDATK and RNAdo into the
applications menu (:func:`add_launchers`, the same menu item in both windows)."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Callable, Optional

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gio, GLib, Gtk  # noqa: E402

from ..frozen import child_env  # noqa: E402
from .projectui import ask  # noqa: E402

LAUNCHERS_ITEM = "Add MDATK and RNAdo to the applications menu…"
LAUNCHERS_TIP = ("Desktop entries for MDATK and RNAdo, with their icons, in your applications menu "
                 "(~/.local/share), starting this installation or AppImage; asks first and lists the files")


def _uri(path: str) -> str:
    return GLib.filename_to_uri(os.path.abspath(path), None)


def _launch_context() -> Optional[Gio.AppLaunchContext]:
    """Launch context for desktop applications: in a frozen build without the bundle's GTK settings."""
    env = child_env()
    if env is None:
        return None
    ctx = Gio.AppLaunchContext()
    for name, value in os.environ.items():
        if name not in env:
            ctx.unsetenv(name)
        elif env[name] != value:
            ctx.setenv(name, env[name])
    return ctx


def open_folder(path: str) -> str:
    """Show ``path`` in the file manager; returns how it was opened.

    On Windows: Explorer.  Otherwise:

    1. ``org.freedesktop.FileManager1.ShowFolders`` over D-Bus (Dolphin,
       Nautilus, Nemo, Thunar, ... all implement it) - always a file manager.
    2. The application registered for ``inode/directory`` (``xdg-mime``).
    3. ``xdg-open``.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"folder does not exist: {path}")
    if os.name == "nt":  # Windows build: Explorer
        os.startfile(path)  # type: ignore[attr-defined]
        return "explorer"
    uri = _uri(path)
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        bus.call_sync("org.freedesktop.FileManager1", "/org/freedesktop/FileManager1",
                      "org.freedesktop.FileManager1", "ShowFolders",
                      GLib.Variant("(ass)", ([uri], "")), None, Gio.DBusCallFlags.NONE, 5000, None)
        return "FileManager1"
    except GLib.Error:
        pass
    app = Gio.AppInfo.get_default_for_type("inode/directory", False)
    if app is not None:
        try:
            if app.launch_uris([uri], _launch_context()):
                return app.get_id() or app.get_name()
        except GLib.Error:
            pass
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env())
        return "xdg-open"
    raise RuntimeError("no file manager could be started")


def open_file(path: str) -> None:
    """Open a file (e.g. the rendered video) with its default application."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file does not exist: {path}")
    Gio.AppInfo.launch_default_for_uri(_uri(path), _launch_context())


def latest_video(out_dir: str, filename: str) -> Optional[str]:
    """Most recent video in ``out_dir`` matching ``filename`` with or without a timestamp."""
    root, ext = os.path.splitext(os.path.basename(filename))
    candidates = glob.glob(os.path.join(out_dir, f"{glob.escape(root)}*{ext}"))
    candidates = [c for c in candidates if not os.path.basename(c).startswith(".")]
    return max(candidates, key=os.path.getmtime) if candidates else None


def add_launchers(parent: Optional[Gtk.Window], info: Callable[[str, str], None],
                  error: Callable[[str, str], None]) -> str:
    """The menu item :data:`LAUNCHERS_ITEM` (:mod:`mdatk.launchers`): asks with the files it writes and what
    the entries start (and, in the AppImage, that moving the file breaks them), then writes them, or
    removes MDATK's when asked to; ``info`` / ``error`` (the window's, which also log) report the result.
    Returns ``"installed"``, ``"removed"``, ``"cancelled"``, ``"unsupported"`` (Windows: the installer made
    Start-menu entries) or ``"failed"``."""
    from .. import launchers

    why = launchers.unsupported()
    if why:
        info("MDATK and RNAdo have Start-menu entries already", why)
        return "unsupported"
    try:
        found = launchers.status()
        detail = launchers.plan_text(found)
    except (OSError, ValueError) as exc:
        error("Could not prepare the menu entries", str(exc))
        return "failed"
    buttons = [("Cancel", Gtk.ResponseType.CANCEL)]
    if any(t.state in (launchers.INSTALLED, launchers.OUTDATED) for t in found):
        buttons.append(("Remove them", Gtk.ResponseType.REJECT))
    buttons.append(("Add to the menu", Gtk.ResponseType.ACCEPT))
    response = ask(parent, "Add MDATK and RNAdo to the applications menu?", detail, buttons,
                   Gtk.ResponseType.ACCEPT)
    if response not in (Gtk.ResponseType.ACCEPT, Gtk.ResponseType.REJECT):
        return "cancelled"
    try:
        if response == Gtk.ResponseType.REJECT:
            result = launchers.uninstall()
        else:
            result = launchers.install()
    except (OSError, RuntimeError, ValueError) as exc:
        error("Could not write the menu entries" if response == Gtk.ResponseType.ACCEPT
              else "Could not remove the menu entries", str(exc))
        return "failed"
    where = launchers.display_path(result.data_dir)
    kept = "Left alone (not written by MDATK):\n" + launchers.listing(result.kept, result.data_dir)
    if response == Gtk.ResponseType.REJECT:
        parts = [f"Removed from {where}:\n" + launchers.listing(result.removed, result.data_dir)
                 if result.removed else f"Nothing of MDATK's in {where}."]
        info("MDATK and RNAdo were removed from the applications menu",
             "\n\n".join(parts + ([kept] if result.kept else [])))
        return "removed"
    parts = []
    if result.written:
        parts.append(f"Written into {where}:\n" + launchers.listing(result.written, result.data_dir))
    elif result.unchanged:
        parts.append("They were there already: nothing changed.")
    if result.kept:
        parts.append(kept)
    parts.append("Some desktops show new entries only after the next log-in.")
    note = launchers.warning()
    if note:
        parts.append(note)
    info("MDATK and RNAdo are in the applications menu", "\n\n".join(parts))
    return "installed"
