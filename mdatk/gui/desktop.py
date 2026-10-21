"""Open folders and files with the user's desktop applications."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Optional

from gi.repository import Gio, GLib

from ..frozen import child_env


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
