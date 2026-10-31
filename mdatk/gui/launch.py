"""Starting RNAdo as a program of its own (from the analysis window's "Open in RNAdo…").

The command depends on how MDATK runs: from source or an installation (``python -m mdatk.gui.rnado``),
as a PyInstaller build (the ``rnado-gui`` program next to this one) or as the AppImage (``$APPIMAGE
rnado``).  This module does not import GTK.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

TOOLS = ("wizard", "pdb", "builder")  # what ``rnado --open`` opens next to the planner


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def rnado_command(project_path: Optional[str], tool: Optional[str] = None) -> List[str]:
    """The command that starts RNAdo on ``project_path`` (none: an empty plan), opening ``tool`` (one of
    :data:`TOOLS`) as well."""
    if tool is not None and tool not in TOOLS:
        raise ValueError(f"unknown RNAdo tool {tool!r} (known: {', '.join(TOOLS)})")
    appimage = os.environ.get("APPIMAGE")
    # APPIMAGE alone is not enough: a terminal started from some other AppImage passes it on to a source run
    if appimage and _frozen():
        cmd = [appimage, "rnado"]
    elif _frozen():  # PyInstaller one-dir build: the programs share one folder
        name = "rnado-gui.exe" if os.name == "nt" else "rnado-gui"
        cmd = [os.path.join(os.path.dirname(os.path.abspath(sys.executable)), name)]
    else:
        cmd = [sys.executable, "-m", "mdatk.gui.rnado"]
    if project_path:
        cmd.append(os.path.abspath(project_path))
    if tool:
        cmd += ["--open", tool]
    return cmd


def launch_rnado(project_path: Optional[str], tool: Optional[str] = None) -> subprocess.Popen:
    """Start RNAdo (see :func:`rnado_command`) without waiting for it; raises OSError when it cannot start."""
    from .. import __file__ as package_init
    from ..frozen import child_env

    env = child_env()  # a frozen build: none of this bundle's settings reach the other program
    if env is None:
        env = dict(os.environ)
        # the same mdatk as this process, also from a source checkout that is not installed
        root = os.path.dirname(os.path.dirname(os.path.abspath(package_init)))
        paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if root not in paths:
            env["PYTHONPATH"] = os.pathsep.join([root] + paths)
    else:
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"  # another PyInstaller program: it sets up its own
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    return subprocess.Popen(rnado_command(project_path, tool), env=env, stdin=subprocess.DEVNULL, **kwargs)
