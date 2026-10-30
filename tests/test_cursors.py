"""Mouse cursors of the GUI (mdatk/gui/cursors.py): the theme's named cursors on Linux, cursor types that GDK
maps to Win32 system cursors on Windows.  Windows itself cannot be tested here: the platform switch is
monkeypatched, against a stand-in for GDK and against GDK's X11 back-end (which has the same cursor types)."""

import os
import time
import types

import pytest

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, Gtk
except (ValueError, ImportError):  # pragma: no cover
    pytest.skip("GTK 3 introspection data not installed", allow_module_level=True)

from mdatk.gui import cursors  # noqa: E402

T = Gdk.CursorType


class FakeCursor:
    """Records what the helper asks GDK for; ``missing`` names have no themed cursor."""
    made: list = []
    missing: set = set()

    def __init__(self, how, what):
        self.how, self.what = how, what

    @classmethod
    def new_from_name(cls, display, name):
        cls.made.append((display, "name", name))
        return None if name in cls.missing else cls("name", name)

    @classmethod
    def new_for_display(cls, display, kind):
        cls.made.append((display, "type", kind))
        return cls("type", kind)


@pytest.fixture
def fake(monkeypatch):
    FakeCursor.made, FakeCursor.missing = [], set()
    monkeypatch.setattr(cursors, "Gdk", types.SimpleNamespace(Cursor=FakeCursor, CursorType=T))
    monkeypatch.setattr(cursors, "_cache", {})
    return FakeCursor


def platform(monkeypatch, name):
    monkeypatch.setattr(cursors, "sys", types.SimpleNamespace(platform=name))
    monkeypatch.setattr(cursors, "_cache", {})


def test_names_map_to_cursor_types_with_win32_system_cursors():
    assert cursors.cursor_type("grab") == T.HAND2  # IDC_HAND
    assert cursors.cursor_type("all-scroll") == cursors.cursor_type("move") == T.FLEUR  # IDC_SIZEALL
    assert cursors.cursor_type("default") == cursors.cursor_type("no-such-name") == T.LEFT_PTR  # IDC_ARROW


def test_windows_uses_the_cursor_types_once_per_display(fake, monkeypatch):
    platform(monkeypatch, "win32")
    display, other = object(), object()
    for name, kind in (("grab", T.HAND2), ("all-scroll", T.FLEUR), ("move", T.FLEUR),
                       ("default", T.LEFT_PTR)):
        cur = cursors.named(display, name)
        assert (cur.how, cur.what) == ("type", kind), name
    assert all(how == "type" for _d, how, _w in fake.made), "Windows must not look names up in the theme"
    n = len(fake.made)
    assert cursors.named(display, "grab") is cursors.named(display, "grab") and len(fake.made) == n
    assert cursors.named(other, "grab") is not cursors.named(display, "grab") and len(fake.made) == n + 1


def test_elsewhere_the_theme_names_with_the_type_as_fallback(fake, monkeypatch):
    platform(monkeypatch, "linux")
    fake.missing = {"grab"}
    display = object()
    assert (cursors.named(display, "move").how, cursors.named(display, "move").what) == ("name", "move")
    grab = cursors.named(display, "grab")
    assert (grab.how, grab.what) == ("type", T.HAND2)
    assert fake.made == [(display, "name", "move"), (display, "name", "grab"), (display, "type", T.HAND2)]


class Ev:
    def __init__(self, x, y):
        self.x, self.y, self.state, self.button = x, y, 0, 1


@pytest.mark.skipif(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
                    reason="no display (run under xvfb-run)")
def test_canvas_sets_its_cursors_through_the_helper(monkeypatch):
    if not Gtk.init_check([])[0]:  # pragma: no cover
        pytest.skip("cannot initialise GTK")
    from mdatk.gui.canvas import FigureCanvas

    win = Gtk.Window()
    canvas = FigureCanvas()
    win.add(canvas)
    win.resize(600, 300)
    win.show_all()
    t0 = time.time()
    while canvas.get_window() is None and time.time() - t0 < 10:
        Gtk.main_iteration_do(False)
    try:
        canvas.slots, canvas.structure_slot = [(0, 0, 200, 200), (200, 0, 200, 200)], 0
        gdk_win = canvas.get_window()
        platform(monkeypatch, "win32")
        for x, map_mode, kind in ((100, False, T.HAND2), (100, True, T.FLEUR), (300, False, T.FLEUR),
                                  (500, False, T.LEFT_PTR)):
            canvas.map_mode = map_mode
            canvas._on_motion(None, Ev(x, 100))
            cur = gdk_win.get_cursor()
            assert cur is not None and cur.get_cursor_type() == kind, (x, map_mode)
        platform(monkeypatch, "linux")  # X11: the named cursors (or their types), never none
        for x in (100, 300, 500):
            canvas._on_motion(None, Ev(x, 100))
            assert gdk_win.get_cursor() is not None
    finally:
        win.destroy()
