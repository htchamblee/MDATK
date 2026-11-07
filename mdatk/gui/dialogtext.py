"""Selectable text in dialogs: messages can be copied instead of copied out by hand.

:func:`selectable_text` makes every label of a dialog selectable (not those of its buttons) and keeps the
keyboard focus off the text when the dialog opens: GTK would otherwise focus the first selectable label and
select all of its text, so Enter would no longer press the default button.  A message dialog's long
detail text (e.g. the list of scripts before a run) goes into a scrolled box at most about half the screen
tall, so the dialog and its buttons never reach past the bottom of the screen."""

from __future__ import annotations

from typing import List

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

LONG_LINES = 14  # detail text with more lines (or characters) than this scrolls
LONG_CHARS = 1400
SCREEN_SHARE = 0.5  # of the monitor's work area: the most the scrolled detail text takes


def _labels(widget: Gtk.Widget, out: List[Gtk.Label]) -> None:
    if isinstance(widget, Gtk.Button):  # a button's own label is not message text
        return
    if isinstance(widget, Gtk.Label):
        out.append(widget)
        return
    if isinstance(widget, Gtk.Container):
        for child in widget.get_children():
            _labels(child, out)


def _buttons(widget: Gtk.Widget, out: List[Gtk.Button]) -> None:
    if isinstance(widget, Gtk.Button):
        out.append(widget)
        return
    if isinstance(widget, Gtk.Container):
        for child in widget.get_children():
            _buttons(child, out)


def _text_height(dialog: Gtk.Dialog) -> int:
    """The most a scrolled detail text may take: SCREEN_SHARE of the work area of the monitor the dialog
    (or its parent) is on, else of the first monitor."""
    display = Gdk.Display.get_default()
    if display is None:
        return 400
    parent = dialog.get_transient_for()
    window = parent.get_window() if parent is not None else None
    monitor = display.get_monitor_at_window(window) if window is not None else None
    monitor = monitor or display.get_primary_monitor() or (display.get_monitor(0)
                                                          if display.get_n_monitors() else None)
    if monitor is None:
        return 400
    return max(200, int(monitor.get_workarea().height * SCREEN_SHARE))


def scroll_long_text(dialog: Gtk.Dialog) -> bool:
    """Put a message dialog's long detail (secondary) text into a scrolled box (see LONG_LINES); True when
    it did.  The text, ``props.secondary_text``, stays as it was."""
    if not isinstance(dialog, Gtk.MessageDialog):
        return False
    text = dialog.props.secondary_text or ""
    if text.count("\n") < LONG_LINES and len(text) <= LONG_CHARS:
        return False
    area = dialog.get_message_area()
    children = area.get_children()
    label = next((w for w in children if isinstance(w, Gtk.Label) and w.get_text() == text
                  and w is not children[0]), None)
    if label is None:
        return False
    position = children.index(label)
    area.remove(label)
    box = Gtk.ScrolledWindow()
    box.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    box.set_propagate_natural_height(True)
    box.set_max_content_height(_text_height(dialog))
    box.set_min_content_height(160)
    label.set_line_wrap(True)
    label.set_max_width_chars(90)
    label.set_xalign(0.0)
    label.set_yalign(0.0)
    box.add(label)
    area.pack_start(box, True, True, 0)
    area.reorder_child(box, position)
    box.show_all()
    return True


def selectable_text(dialog: Gtk.Dialog) -> List[Gtk.Label]:
    """Make the text of ``dialog`` selectable (and a message dialog's long detail text scroll, see
    :func:`scroll_long_text`); call it once its content is packed, before it is shown (``run()`` /
    ``show_all()``).  Returns the labels made selectable."""
    scroll_long_text(dialog)
    labels: List[Gtk.Label] = []
    _labels(dialog.get_content_area(), labels)
    for label in labels:
        label.set_selectable(True)

    def settle() -> bool:
        focus = dialog.get_focus()
        if focus is None or isinstance(focus, Gtk.Label):  # GTK focused the text: give it to a button
            target = dialog.get_default_widget()
            if target is None:
                buttons: List[Gtk.Button] = []
                _buttons(dialog, buttons)
                target = buttons[-1] if buttons else None
            if target is not None:
                target.grab_focus()
        for label in labels:
            if label.get_selection_bounds()[0]:  # (selected, start, end)
                label.select_region(0, 0)
        return False

    dialog.connect("map", lambda *_: GLib.idle_add(settle))
    return labels
