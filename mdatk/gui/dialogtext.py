"""Selectable text in dialogs: messages can be copied instead of copied out by hand.

:func:`selectable_text` makes every label of a dialog selectable (not those of its buttons) and keeps the
keyboard focus off the text when the dialog opens: GTK would otherwise focus the first selectable label and
select all of its text, so Enter would no longer press the default button."""

from __future__ import annotations

from typing import List

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402


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


def selectable_text(dialog: Gtk.Dialog) -> List[Gtk.Label]:
    """Make the text of ``dialog`` selectable; call it once its content is packed, before it is shown
    (``run()`` / ``show_all()``).  Returns the labels made selectable."""
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
