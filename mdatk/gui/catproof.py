"""Cat-proof mode: while it is on, the program ignores the keyboard (every key press and release, in every window
of the process: the main window, tool windows, dialogs and file choosers), so a cat walking across the keys
types nothing, triggers no shortcut and deletes no step.  The mouse keeps working, and the check button that
switches the mode is ticked and unticked with it.

The keys are dropped where GDK hands events to GTK (Gdk.event_handler_set): every other event goes on to
Gtk.main_do_event, GTK's own handler, unchanged."""

from __future__ import annotations

import math
from typing import Callable, Optional

import gi

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, Gtk  # noqa: E402

KEY_EVENTS = (Gdk.EventType.KEY_PRESS, Gdk.EventType.KEY_RELEASE)
LABEL = "Cat-proof mode"
TOOLTIP = ("Ignore the keyboard in RNAdo and its tool windows and dialogs, so a cat walking across the keys "
           "types nothing and triggers no shortcut. The mouse keeps working: untick this box with it.")


class CatGuard:
    """The process-wide key filter; one per process (:func:`guard`)."""

    def __init__(self) -> None:
        self.active = False
        self.ignored = 0  # key presses dropped since the mode was last switched on
        self._installed = False

    def _filter(self, event, _data=None) -> None:
        if self.active and event.type in KEY_EVENTS:
            if event.type == Gdk.EventType.KEY_PRESS:
                self.ignored += 1
            return
        Gtk.main_do_event(event)

    def set_active(self, on: bool) -> int:
        """Switch the mode; returns the key presses ignored while it was on (when switching it off)."""
        if on and not self._installed:
            Gdk.event_handler_set(self._filter, None)
            self._installed = True
        ignored = self.ignored if self.active and not on else 0
        if on and not self.active:
            self.ignored = 0
        self.active = bool(on)
        return ignored


ICON_SIZE = (18, 16)  # pixels (logical); drawn with cairo, so it is sharp at any scale


def draw_cat_keyboard(cr, r: float, g: float, b: float) -> None:
    """A cat sitting on a keyboard, in an 18 x 16 box of cairo user units, in the colour (r, g, b): the
    keyboard outlined with its keys, the cat a filled silhouette (loaf body, head with two ears, raised tail,
    two front paws on the keys)."""
    def rrect(x, y, w, h, rad):
        cr.new_sub_path()
        cr.arc(x + w - rad, y + rad, rad, -math.pi / 2, 0)
        cr.arc(x + w - rad, y + h - rad, rad, 0, math.pi / 2)
        cr.arc(x + rad, y + h - rad, rad, math.pi / 2, math.pi)
        cr.arc(x + rad, y + rad, rad, math.pi, 1.5 * math.pi)
        cr.close_path()

    cr.set_source_rgba(r, g, b, 0.85)  # the keyboard
    cr.set_line_width(1.0)
    rrect(0.8, 10.3, 16.4, 5.0, 1.2)
    cr.stroke()
    cr.set_source_rgba(r, g, b, 0.6)
    for x in (2.4, 4.8, 7.2, 9.6, 12.0, 14.2):  # the top row of keys
        rrect(x, 11.6, 1.6, 1.0, 0.3)
        cr.fill()
    for x, w in ((2.4, 1.6), (4.8, 1.6), (7.2, 6.0), (14.2, 1.6)):  # the bottom row, with the space bar
        rrect(x, 13.3, w, 1.0, 0.3)
        cr.fill()
    cr.set_source_rgba(r, g, b, 1.0)  # the cat
    cr.save()
    cr.translate(10.0, 7.9)
    cr.scale(4.2, 2.5)
    cr.arc(0, 0, 1, 0, 2 * math.pi)  # body
    cr.restore()
    cr.fill()
    cr.arc(5.2, 5.9, 2.4, 0, 2 * math.pi)  # head
    cr.fill()
    for ear in (((3.0, 5.0), (3.2, 2.2), (5.0, 3.7)), ((5.6, 3.6), (7.3, 2.1), (7.4, 5.0))):
        cr.move_to(*ear[0])
        cr.line_to(*ear[1])
        cr.line_to(*ear[2])
        cr.close_path()
        cr.fill()
    cr.set_line_width(1.3)  # tail
    cr.set_line_cap(1)  # cairo.LINE_CAP_ROUND
    cr.move_to(13.8, 8.4)
    cr.curve_to(16.4, 8.0, 16.6, 5.4, 15.2, 4.4)
    cr.stroke()
    for x in (6.4, 9.0):  # front paws on the keys
        cr.arc(x, 10.2, 0.9, 0, 2 * math.pi)
        cr.fill()


class CatIcon(Gtk.DrawingArea):
    """The cat-on-a-keyboard icon, drawn in the widget's text colour (so it suits light and dark themes and
    greys out with the button)."""

    def __init__(self) -> None:
        super().__init__()
        self.set_size_request(*ICON_SIZE)
        self.set_valign(Gtk.Align.CENTER)
        self.connect("draw", self._draw)

    def _draw(self, widget, cr) -> bool:
        color = widget.get_style_context().get_color(widget.get_state_flags())
        w, h = widget.get_allocated_width(), widget.get_allocated_height()
        cr.translate((w - ICON_SIZE[0]) / 2, (h - ICON_SIZE[1]) / 2)
        draw_cat_keyboard(cr, color.red, color.green, color.blue)
        return False


_GUARD: Optional[CatGuard] = None


def guard() -> CatGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = CatGuard()
    return _GUARD


def check_button(on_change: Optional[Callable[[bool, int], None]] = None, active: bool = False) -> Gtk.CheckButton:
    """The 'Cat-proof mode' check button with its cat-on-a-keyboard icon, switching :func:`guard`;
    ``on_change(active, ignored)`` is called after each toggle (``ignored``: key presses dropped while the
    mode was on, when it is switched off)."""
    button = Gtk.CheckButton()
    box = Gtk.Box(spacing=4)
    button.icon = CatIcon()
    box.pack_start(button.icon, False, False, 0)
    box.pack_start(Gtk.Label(label=LABEL), False, False, 0)
    button.add(box)
    button.set_tooltip_text(TOOLTIP)
    button.set_focus_on_click(False)  # a click does not move the keyboard focus (the keys go nowhere anyway)

    def toggled(b):
        ignored = guard().set_active(b.get_active())
        if on_change is not None:
            on_change(b.get_active(), ignored)

    button.connect("toggled", toggled)
    if active:
        button.set_active(True)
    return button
