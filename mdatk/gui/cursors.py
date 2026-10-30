"""Mouse cursors that show on every platform.

The GUI names its cursors the CSS way ("grab", "all-scroll", "move", "default").  GDK on Windows maps
only some of these names to Win32 system cursors and loads the others from the cursor theme's files,
which the stand-alone build leaves out (``packaging/trim.py`` drops ``icons/Adwaita/cursors``): such a
name gives no cursor, and the pointer became invisible over the structure panel ("grab" / "all-scroll")
while the other panels ("move") kept it.  So on Windows the GUI asks for cursor types that GDK maps to
Win32 system cursors (FLEUR -> IDC_SIZEALL, HAND2 -> IDC_HAND, LEFT_PTR -> IDC_ARROW); elsewhere it keeps
the theme's named cursors and falls back to the same types when the theme lacks a name.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

import gi

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk  # noqa: E402

# CSS cursor name -> Gdk.CursorType that stands in for it (Windows has no open-hand system cursor)
TYPES = {"default": "LEFT_PTR", "move": "FLEUR", "all-scroll": "FLEUR", "grab": "HAND2"}

_cache: Dict[object, Dict[str, Optional[Gdk.Cursor]]] = {}  # display -> cursor name -> cursor


def cursor_type(name: str) -> Gdk.CursorType:
    """The cursor type for a CSS cursor name (the arrow for names without one)."""
    return getattr(Gdk.CursorType, TYPES.get(name, "LEFT_PTR"))


def named(display: Gdk.Display, name: str) -> Optional[Gdk.Cursor]:
    """The cursor ``name`` (a CSS name) on ``display``, made once per display."""
    cursors = _cache.setdefault(display, {})
    if name not in cursors:
        cursor = None if sys.platform == "win32" else Gdk.Cursor.new_from_name(display, name)
        if cursor is None:
            cursor = Gdk.Cursor.new_for_display(display, cursor_type(name))
        cursors[name] = cursor
    return cursors[name]


def set_named(window: Gdk.Window, name: str) -> None:
    """Show the cursor ``name`` over a GDK window."""
    window.set_cursor(named(window.get_display(), name))
