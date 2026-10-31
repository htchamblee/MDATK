"""Starting RNAdo as a program of its own (from the analysis window's "Open in RNAdo…").

The command depends on how MDATK runs (:func:`launch_mode`): from source or an installation (``python -m
mdatk.gui.rnado``), as a PyInstaller build (the ``rnado-gui`` program next to this one) or from the
AppImage (``$APPIMAGE rnado``: RNAdo gets a mount of its own, which stays when this program ends).  This
module does not import GTK; :func:`watch_start` uses GLib when it is called.
"""

from __future__ import annotations

import os
import shlex
import site
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

TOOLS = ("wizard", "pdb", "builder")  # what ``rnado --open`` opens next to the planner
START_S = 3.0  # RNAdo ending with an error status within this time did not start
MODES = ("source", "bundle", "appimage", "appimage-extracted")


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _same(a: str, b: str) -> bool:
    norm = os.path.normcase
    return norm(os.path.abspath(a)) == norm(os.path.abspath(b)) or \
        norm(os.path.realpath(a)) == norm(os.path.realpath(b))


def _inside(path: str, folder: str) -> bool:
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def launch_mode() -> str:
    """How RNAdo is started (one of :data:`MODES`): ``source`` (``python -m``), ``bundle`` (the ``rnado-gui``
    next to this program), ``appimage`` (``$APPIMAGE rnado``) or ``appimage-extracted`` (the same for an
    AppImage that runs from the folder it was extracted to: no FUSE, ``--appimage-extract-and-run``)."""
    if not _frozen():
        return "source"
    appimage, appdir = os.environ.get("APPIMAGE"), os.environ.get("APPDIR")
    # APPIMAGE and APPDIR alone are not enough: a terminal started from another AppImage passes them on
    if not (appimage and appdir and _inside(sys.executable, appdir)):
        return "bundle"
    if os.environ.get("APPIMAGE_EXTRACT_AND_RUN") or not os.path.ismount(appdir):
        return "appimage-extracted"
    return "appimage"


def rnado_command(project_path: Optional[str], tool: Optional[str] = None,
                  mode: Optional[str] = None) -> List[str]:
    """The command that starts RNAdo on ``project_path`` (none: an empty plan), opening ``tool`` (one of
    :data:`TOOLS`) as well; ``mode``: :func:`launch_mode` unless given."""
    if tool is not None and tool not in TOOLS:
        raise ValueError(f"unknown RNAdo tool {tool!r} (known: {', '.join(TOOLS)})")
    mode = mode or launch_mode()
    if mode in ("appimage", "appimage-extracted"):
        cmd = [os.environ["APPIMAGE"], "rnado"]
    elif mode == "bundle":  # PyInstaller one-dir build: the programs share one folder
        name = "rnado-gui.exe" if os.name == "nt" else "rnado-gui"
        cmd = [os.path.join(os.path.dirname(os.path.abspath(sys.executable)), name)]
    else:
        cmd = [sys.executable, "-m", "mdatk.gui.rnado"]
    if project_path:
        cmd.append(os.path.abspath(project_path))
    if tool:
        cmd += ["--open", tool]
    return cmd


def _site_dir(path: str) -> bool:
    """Whether ``path`` is one of this Python's site-packages folders (an installation, a venv)."""
    dirs = list(getattr(site, "getsitepackages", lambda: [])())
    user = getattr(site, "getusersitepackages", lambda: "")()
    dirs += [user] if isinstance(user, str) else list(user)
    return any(_same(path, d) for d in dirs if d)


def _source_env(env: Dict[str, str]) -> None:
    """``env`` for ``python -m mdatk.gui.rnado``: the same mdatk as this process, also from a source
    checkout that is not installed, and the user's own PYTHONPATH recorded for the scripts RNAdo runs."""
    from .. import __file__ as package_init
    from ..frozen import ORIG_PYTHONPATH

    orig = env.get("PYTHONPATH", "")
    # frozen.child_env gives it back to them: a script's other Python (ParmEd from AmberTools, a conda
    # environment) must not import this Python's packages
    env[ORIG_PYTHONPATH] = orig
    root = os.path.dirname(os.path.dirname(os.path.abspath(package_init)))
    paths = [p for p in orig.split(os.pathsep) if p]
    if not _site_dir(root) and not any(_same(root, p) for p in paths):
        env["PYTHONPATH"] = os.pathsep.join([root] + paths)


def launch_rnado(project_path: Optional[str], tool: Optional[str] = None) -> subprocess.Popen:
    """Start RNAdo (see :func:`rnado_command`) without waiting for it; raises OSError when it cannot start
    (see :func:`watch_start` for a start that fails a moment later)."""
    from ..frozen import child_env

    mode = launch_mode()
    env = child_env()  # a frozen build: none of this bundle's settings reach the other program
    env = dict(os.environ) if env is None else env
    if mode == "source":
        _source_env(env)
    else:
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"  # another PyInstaller program: it sets up its own
    if mode == "appimage-extracted":
        # Without FUSE the AppImage runs only extracted.  RNAdo's runtime reuses this program's folder (same
        # name) and must leave it in place when RNAdo ends; this program's runtime removes it at its end.
        env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
        env["NO_CLEANUP"] = "1"
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    return subprocess.Popen(rnado_command(project_path, tool, mode), env=env, stdin=subprocess.DEVNULL,
                            **kwargs)


def _status(code: int) -> str:
    return f"signal {-code}" if code < 0 else f"exit status {code}"


def watch_start(proc: subprocess.Popen, on_started: Callable[[], None], on_failed: Callable[[str], None],
                seconds: Optional[float] = None, every_ms: int = 200) -> int:
    """Watch ``proc`` (from :func:`launch_rnado`) on the GLib main loop: ``on_failed(why)`` when it ends
    with an error status within ``seconds`` (default :data:`START_S`; an AppImage that cannot mount, a
    broken installation), ``on_started()`` otherwise.  Returns the GLib source id."""
    from gi.repository import GLib

    end = time.monotonic() + (START_S if seconds is None else seconds)

    def check() -> bool:
        code = proc.poll()
        if code is None and time.monotonic() < end:
            return True
        if code:  # an error status, or a signal
            on_failed(f"RNAdo ended right after it started ({_status(code)}). Its messages, if any, are on "
                      "the standard error of this program (the terminal it was started from).\n\n"
                      f"Command: {shlex.join(str(a) for a in proc.args)}")
        else:
            on_started()
        return False

    return GLib.timeout_add(every_ms, check)
