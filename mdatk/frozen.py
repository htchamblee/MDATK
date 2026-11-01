"""Support for the stand-alone builds (PyInstaller: Linux AppImage, Windows).

From a source checkout or a pip installation every function here is a no-op or
returns ``None`` (except :func:`child_env` in RNAdo started by the analysis
window, below).  In a frozen build:

* :func:`setup` (called by the bundle's entry scripts before anything else is
  imported) puts the bundled tools folder (FFmpeg) at the front of ``PATH``,
  so the bundled FFmpeg is found before one on ``PATH``, and gives programs
  started later the library path the user had before the bundle changed it.
* :func:`child_env` is the environment for external programs (generated
  scripts, the file manager): the bundle's GTK / GObject-introspection
  settings point into the bundle and must not leak into the user's programs.

:func:`child_env` also gives external programs back the ``PYTHONPATH`` the user
had when the analysis window started RNAdo from source: ``mdatk.gui.launch``
may put MDATK in front of it and records the user's value in
:data:`ORIG_PYTHONPATH`.  Another Python a script starts (AmberTools' ParmEd, a
conda environment) must not import this Python's packages.

PyInstaller's own run-time hooks already set the GObject-introspection
typelib path, the GDK pixbuf loader cache, ``XDG_DATA_DIRS`` (icons, themes,
GSettings schemas) and the GTK prefixes for this process.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

TOOLS_DIR = "tools"  # bundle folder holding ffmpeg(.exe)

# Set by PyInstaller's run-time hooks for the frozen process (the GDK pixbuf loader cache and the
# Matplotlib configuration are temporary files outside the bundle): never passed on.
_HOOK_VARS = ("GI_TYPELIB_PATH", "GIO_MODULE_DIR", "GDK_PIXBUF_MODULE_FILE", "GTK_DATA_PREFIX",
              "GTK_EXE_PREFIX", "GTK_PATH", "PANGO_LIBDIR", "PANGO_SYSCONFDIR", "MPLCONFIGDIR")
# Passed on unless they point into the bundle.
_BUNDLE_VARS = ("GDK_PIXBUF_MODULEDIR", "GSETTINGS_SCHEMA_DIR", "FONTCONFIG_FILE", "FONTCONFIG_PATH")
_PATH_LISTS = ("LD_LIBRARY_PATH", "XDG_DATA_DIRS", "PATH")
# The user's PYTHONPATH ("" when unset) in a program that mdatk.gui.launch started with its own.
ORIG_PYTHONPATH = "MDATK_ORIG_PYTHONPATH"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def bundle_dir() -> Optional[str]:
    """The bundle's data folder (PyInstaller's ``_MEIPASS``), or None from source."""
    return os.path.abspath(sys._MEIPASS) if is_frozen() else None  # type: ignore[attr-defined]


def tools_dir() -> Optional[str]:
    """Folder with the bundled external tools (FFmpeg), if this build has one."""
    if not is_frozen():
        return None
    for base in (os.path.dirname(os.path.abspath(sys.executable)), bundle_dir()):
        path = os.path.join(base, TOOLS_DIR)
        if os.path.isdir(path):
            return path
    return None


def bundled_tool(name: str) -> Optional[str]:
    """Absolute path of a bundled tool such as ``ffmpeg`` (``ffmpeg.exe`` on Windows), or None."""
    folder = tools_dir()
    if folder is None:
        return None
    for exe in (name, name + ".exe") if os.name == "nt" else (name,):
        path = os.path.join(folder, exe)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), root]) == root
    except ValueError:  # different drives (Windows) or empty
        return False


def _no_netcdf4_warning(record: logging.LogRecord) -> bool:
    return "netCDF4 is not available" not in record.getMessage()


def setup() -> None:
    """Prepare the environment of a frozen build; does nothing from source."""
    if not is_frozen():
        return
    # The bootloader (and the AppImage's AppRun, for _internal/fallback) extended LD_LIBRARY_PATH
    # for this process (the loader has read it already); programs started from MDATK (cpptraj,
    # generated scripts) get the user's own value back.
    orig = os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
    if orig is not None:
        os.environ["LD_LIBRARY_PATH"] = orig
    if "LD_LIBRARY_PATH" in os.environ:
        root = bundle_dir() or ""
        kept = [p for p in os.environ["LD_LIBRARY_PATH"].split(os.pathsep) if p and not _inside(p, root)]
        if kept:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            del os.environ["LD_LIBRARY_PATH"]
    # The builds leave the optional netCDF4 out (scipy reads and writes Amber NetCDF);
    # MDAnalysis would warn about it on every start.
    amber_log = logging.getLogger("MDAnalysis.coordinates.AMBER")
    if _no_netcdf4_warning not in amber_log.filters:
        amber_log.addFilter(_no_netcdf4_warning)
    tools = tools_dir()
    path = os.environ.get("PATH", "")
    if tools and tools not in path.split(os.pathsep):  # bundled FFmpeg before one on PATH
        os.environ["PATH"] = tools + os.pathsep + path if path else tools


def _user_pythonpath(env: Dict[str, str]) -> None:
    """Give ``env`` the PYTHONPATH recorded in :data:`ORIG_PYTHONPATH` back."""
    orig = env.pop(ORIG_PYTHONPATH, None)
    if orig is None:
        return
    if orig:
        env["PYTHONPATH"] = orig
    else:
        env.pop("PYTHONPATH", None)


def child_env() -> Optional[Dict[str, str]]:
    """Environment for external programs: without the settings of a frozen build, and with the user's own
    PYTHONPATH (see :data:`ORIG_PYTHONPATH`); None: inherit this process's environment unchanged."""
    root = bundle_dir()
    if root is None and ORIG_PYTHONPATH not in os.environ:
        return None
    env = dict(os.environ)
    _user_pythonpath(env)
    if root is None:
        return env
    for var in _HOOK_VARS:
        env.pop(var, None)
    for var in [v for v in env if v.startswith("_PYI_")]:  # PyInstaller's own bookkeeping
        del env[var]
    for var in _BUNDLE_VARS:
        if var in env and _inside(env[var], root):
            del env[var]
    for var in _PATH_LISTS:
        if var not in env:
            continue
        kept = [p for p in env[var].split(os.pathsep) if p and not _inside(p, root)]
        tools = tools_dir()
        if var == "PATH" and tools and tools not in kept:
            kept.insert(0, tools)  # keep the bundled FFmpeg (a static program) reachable
        if kept:
            env[var] = os.pathsep.join(kept)
        else:
            del env[var]
    return env
