"""The programs end with their windows, and the programs they start inherit nothing of an AppImage.

Started from an AppImage, MDATK holds two descriptors of the AppImage runtime: the read end of its keepalive
pipe (inherited without close-on-exec; the runtime keeps the FUSE mount, and its own process, alive while
any process holds it) and an fd 1023 on the mount.  RNAdo, the scripts RNAdo runs and the desktop
applications MDATK opens must not inherit them (mdatk.gui.launch, mdatk.workflows.runner,
mdatk.gui.desktop).  And a window attached to the application but hidden must not keep either program
running once its main window has closed (the GTK tests run under xvfb-run -a).  RNAdo, a child of the
analysis window that started it, is reaped when it ends, so it does not stay a defunct process."""

import importlib
import os
import resource
import shlex
import shutil
import stat
import subprocess
import sys
import time

import pytest

PROC_FDS = os.path.isdir("/proc/self/fd")
KEEPALIVE, MOUNT = 1000, 1023  # where the stand-ins of the runtime's descriptors are put


def _gtk():
    """Gtk when a display can be used, else None."""
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
    except (ImportError, ValueError):
        return None
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or not Gtk.init_check([])[0]:
        return None
    return Gtk


def _gio():
    try:
        import gi
        gi.require_version("Gio", "2.0")
        from gi.repository import Gio, GLib
    except (ImportError, ValueError):
        return None
    return Gio, GLib


NEEDS_PROC = pytest.mark.skipif(not PROC_FDS, reason="needs /proc/self/fd (Linux)")
NEEDS_GTK = pytest.mark.skipif(_gtk() is None, reason="needs GTK 3 and a display (xvfb-run -a)")
NEEDS_GIO = pytest.mark.skipif(_gio() is None, reason="needs PyGObject")


@pytest.fixture
def appimage_fds(tmp_path):
    """This process holds descriptors like those the AppImage runtime leaves to MDATK: an inheritable read
    end of a pipe and an inheritable fd 1023 on a directory (the mount)."""
    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= MOUNT:
        pytest.skip("the descriptor limit is too low for an fd 1023")
    for fd in (KEEPALIVE, MOUNT):
        try:
            os.fstat(fd)
        except OSError:
            continue
        pytest.skip(f"fd {fd} is in use")
    read, write = os.pipe()
    os.dup2(read, KEEPALIVE, inheritable=True)
    folder = os.open(str(tmp_path), os.O_RDONLY)
    os.dup2(folder, MOUNT, inheritable=True)
    for fd in (read, folder):
        os.close(fd)
    yield
    for fd in (KEEPALIVE, MOUNT, write):
        os.close(fd)


def _lister(tmp_path, name):
    """(a program that writes the descriptors of its process into a file, that file)."""
    out = tmp_path / f"{name}.fds"
    prog = tmp_path / name
    prog.write_text(f"#!/bin/sh\nls /proc/$$/fd > {shlex.quote(str(out))}\n")
    prog.chmod(prog.stat().st_mode | stat.S_IXUSR)
    return str(prog), out


def _fds(out, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not (out.exists() and out.read_text().strip()):
        time.sleep(0.05)
    fds = set(out.read_text().split())
    assert {"0", "1", "2"} <= fds, fds  # the listing is complete
    return fds


@NEEDS_PROC
def test_rnado_inherits_no_appimage_descriptors(appimage_fds, tmp_path, monkeypatch):
    from mdatk.gui import launch

    prog, out = _lister(tmp_path, "rnado")
    monkeypatch.setattr(launch, "rnado_command", lambda *_a, **_k: ["/bin/sh", prog])
    proc = launch.launch_rnado(str(tmp_path / "plan.mdatk.toml"))
    assert proc.wait(timeout=60) == 0
    assert not {str(KEEPALIVE), str(MOUNT)} & _fds(out)


def _state(pid):
    """The state letter of process ``pid`` (Z: defunct), None when there is no such process."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


@NEEDS_PROC
@NEEDS_GIO
def test_rnado_is_reaped_when_it_ends():
    """RNAdo is a child of the analysis window that started it.  Closed while that window stays open, it
    stayed a defunct process ("[mdatk] <defunct>" from the AppImage) until the analysis window ended: the
    watch of its start (mdatk.gui.launch.watch_start) stopped looking at it after the start time."""
    _Gio, GLib = _gio()
    from mdatk.gui import launch

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    got = []
    launch.watch_start(proc, lambda: got.append("started"), got.append, seconds=0.2, every_ms=20)
    ctx, end = GLib.MainContext.default(), time.monotonic() + 10
    try:
        while proc.returncode is None and time.monotonic() < end:  # the analysis window runs on
            ctx.iteration(False)
            time.sleep(0.01)
        state = _state(proc.pid)
        assert got == ["started"]
        assert proc.returncode == 0, f"RNAdo ended but was not reaped (process state {state})"
        assert state is None
    finally:
        if proc.returncode is None:
            proc.kill()
            proc.wait()


@NEEDS_PROC
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_scripts_inherit_no_appimage_descriptors(appimage_fds, tmp_path):
    from mdatk.workflows.runner import run_script

    script = tmp_path / "step.sh"
    script.write_text("ls /proc/$$/fd\n")
    lines = []
    status, _output = run_script(str(script), lines.append)
    assert status == 0 and {"0", "1", "2"} <= set(lines)
    assert not {str(KEEPALIVE), str(MOUNT)} & set(lines)


@NEEDS_PROC
@NEEDS_GIO
def test_desktop_programs_inherit_no_appimage_descriptors(appimage_fds, tmp_path, monkeypatch):
    """Both ways mdatk.gui.desktop starts a program itself: an application through Gio (GLib spawns it) and
    xdg-open (subprocess); the file manager over D-Bus is not started by this process."""
    Gio, GLib = _gio()
    from mdatk.gui import desktop

    def no_bus(*_a):
        raise GLib.Error("no session bus here")

    monkeypatch.setattr(desktop.Gio, "bus_get_sync", no_bus)
    # the application registered for folders: GLib spawns it
    prog, out = _lister(tmp_path, "filemanager")
    app = Gio.AppInfo.create_from_commandline(f"/bin/sh {shlex.quote(prog)}", "fd lister",
                                              Gio.AppInfoCreateFlags.NONE)
    monkeypatch.setattr(desktop.Gio.AppInfo, "get_default_for_type", lambda *_a: app)
    assert desktop.open_folder(str(tmp_path)) == "fd lister"
    assert not {str(KEEPALIVE), str(MOUNT)} & _fds(out)
    # none registered: xdg-open
    prog, out = _lister(tmp_path, "xdg-open")
    monkeypatch.setattr(desktop.Gio.AppInfo, "get_default_for_type", lambda *_a: None)
    monkeypatch.setattr(desktop.shutil, "which", lambda name: prog if name == "xdg-open" else None)
    assert desktop.open_folder(str(tmp_path)) == "xdg-open"
    assert not {str(KEEPALIVE), str(MOUNT)} & _fds(out)


@NEEDS_GTK
@pytest.mark.parametrize("module, name", [("mdatk.gui.app", "MDATKApplication"),
                                          ("mdatk.gui.rnado", "RNAdoApplication")])
def test_a_hidden_window_does_not_keep_the_program_running(module, name):
    """A window attached to the application but never shown (a dialog or tool window left hidden) keeps
    a GTK application running after its main window has closed, unless the program quits with it."""
    from gi.repository import GLib, Gtk

    app = getattr(importlib.import_module(module), name)()
    seen = {}

    def close_main_window():
        seen["hidden"] = Gtk.Window(application=app)
        seen["windows"] = len(app.get_windows())
        app.window.destroy()
        return False

    def give_up():
        seen["timed out"] = True
        app.quit()
        return False

    GLib.timeout_add(300, close_main_window)
    guard = GLib.timeout_add_seconds(30, give_up)
    try:
        status = app.run([sys.argv[0]])
    finally:
        if "timed out" not in seen:
            GLib.source_remove(guard)
        if "hidden" in seen:
            seen["hidden"].destroy()
    assert seen.get("windows") == 2  # the main window and the hidden one
    assert "timed out" not in seen and status == 0
