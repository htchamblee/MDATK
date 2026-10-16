"""Choice of the 3D renderer.

``auto`` and ``software`` use the built-in numpy renderer: no OpenGL, so it works the
same on a laptop, in the GUI and on a headless cluster node.  ``vtk`` uses VTK when it
can render here (checked once in a subprocess) and otherwise falls back to the built-in
renderer with a note; it never falls back to the 2D trace.
"""

from __future__ import annotations

import logging
from typing import Tuple

log = logging.getLogger(__name__)


def get_renderer(name: str = "auto") -> Tuple[object, str]:
    """(renderer, note for the log / panel; empty when the requested one is used)."""
    from .software import SoftwareRenderer

    if name == "vtk":
        from . import vtk_backend

        ok, why = vtk_backend.available()
        if ok:
            return vtk_backend.VtkRenderer(), ""
        return SoftwareRenderer(), f"{why}; using the built-in renderer"
    return SoftwareRenderer(), ""
