"""Choice of the 3D renderer: ``auto`` (VTK when it can render here, otherwise the built-in
software renderer), ``vtk`` or ``software``."""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def _software():
    if importlib.util.find_spec("mdatk.render.molecule.software") is None:
        return None
    from .software import SoftwareRenderer

    return SoftwareRenderer()


def get_renderer(name: str = "auto") -> Tuple[Optional[object], str]:
    """(renderer or None, note for the log / panel)."""
    from . import vtk_backend

    if name in ("auto", "vtk"):
        ok, why = vtk_backend.available()
        if ok:
            return vtk_backend.VtkRenderer(), why
        if name == "vtk":
            soft = _software()
            if soft is not None:
                return soft, f"{why}; using the built-in renderer"
            return None, why
    soft = _software()
    if soft is not None:
        return soft, "built-in renderer"
    return None, "no 3D renderer available (install vtk); showing the 2D trace"
