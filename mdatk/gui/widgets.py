"""GTK 3 form helpers and two-way parameter bindings."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Pango", "1.0")
from gi.repository import Gdk, Gtk, Pango  # noqa: E402

# ---------------------------------------------------------------------------
# dotted-path access
# ---------------------------------------------------------------------------


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
    setattr(obj, parts[-1], value)


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------


def is_color(text: Any) -> bool:
    """``text`` is a colour the renderer (Matplotlib) knows: a name, tab:orange, C0, a grey level such
    as 0.5, #rrggbb or #rrggbbaa."""
    from matplotlib.colors import is_color_like

    return isinstance(text, str) and bool(text) and is_color_like(text)


def hex_to_rgba(text: str) -> Gdk.RGBA:
    """The colour ``text`` for a GTK widget: read as the renderer (Matplotlib) reads it, else as GDK
    does; black when neither knows it."""
    rgba = Gdk.RGBA()
    if is_color(text):
        from matplotlib.colors import to_rgba

        rgba.red, rgba.green, rgba.blue, rgba.alpha = (float(c) for c in to_rgba(text))
    elif not rgba.parse(text or "#000000"):
        rgba.parse("#000000")
    return rgba


def rgba_to_hex(rgba: Gdk.RGBA) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(round(rgba.red * 255)), int(round(rgba.green * 255)),
                                        int(round(rgba.blue * 255)))


# ---------------------------------------------------------------------------
# values as loaded: a widget never replaces a value it cannot show
# ---------------------------------------------------------------------------


def number_text(value: Any) -> str:
    """The text of a number that reads back as the same number: "5", "0.25", and every digit of
    0.0123456789 ("%g" alone keeps six)."""
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    short = f"{value:g}"
    return short if float(short) == value else repr(value)


def set_spin(w: Gtk.SpinButton, value: Any) -> None:
    """Show ``value`` in a spin button.  The button's range is a suggestion: it is widened to hold a
    value outside it (and is the button's own again for the next value inside it), so loading never
    clamps.  :func:`spin_value` reads the value back as given while the button still shows its text."""
    adj = w.get_adjustment()
    if not hasattr(w, "_mdatk_range"):
        w._mdatk_range = (adj.get_lower(), adj.get_upper())
    lo, hi = w._mdatk_range
    number = float(value)
    if math.isfinite(number):
        lo, hi = min(lo, number), max(hi, number)
    adj.set_lower(lo)
    adj.set_upper(hi)
    w.set_value(number)
    w._mdatk_held = (value, w.get_text())


def spin_held(w: Gtk.SpinButton) -> bool:
    """The spin button still shows the text of the value :func:`set_spin` gave it."""
    held = getattr(w, "_mdatk_held", None)
    return held is not None and w.get_text() == held[1]


def spin_value(w: Gtk.SpinButton, cast: Callable[[float], Any] = float) -> Any:
    """The value of a spin button: the one :func:`set_spin` gave it while the button still shows that
    value's text (a value with more digits than the button shows, or one it cannot show, is read back as
    it was loaded; GTK itself rounds to the shown digits when the button loses the focus), else
    ``cast`` of what the button holds."""
    if spin_held(w):
        return _as_loaded(w._mdatk_held[0], cast)
    return cast(w.get_value())


def _as_loaded(value: Any, cast: Callable[[float], Any]) -> Any:
    """A loaded number as its widget reads numbers (``cast``) when that is the same number, else as it is."""
    try:
        same = cast(float(value))
    except (TypeError, ValueError, OverflowError):
        return value
    return same if same == value else value


def _scale_moved(w: Gtk.Range) -> None:
    if not getattr(w, "_mdatk_setting", False):
        w._mdatk_held = None


def set_scale(w: Gtk.Range, value: Any) -> None:
    """Show ``value`` on a slider, which clamps it to its range; :func:`scale_value` reads the value back
    as given until the slider moves."""
    if not hasattr(w, "_mdatk_held"):
        w.connect("value-changed", _scale_moved)
    w._mdatk_setting = True
    try:
        w.set_value(float(value))
    finally:
        w._mdatk_setting = False
    w._mdatk_held = (value,)


def scale_value(w: Gtk.Range, cast: Callable[[float], Any] = float) -> Any:
    """The value of a slider: the one :func:`set_scale` gave it until the slider moves (a value outside
    its range, or with more digits than it has, is read back as it was loaded), else ``cast`` of its
    position."""
    held = getattr(w, "_mdatk_held", None)
    return _as_loaded(held[0], cast) if held is not None else cast(w.get_value())


def spin_problem(w: Gtk.SpinButton) -> str:
    """What to say about a loaded value the spin button cannot show ("" when it shows its value)."""
    if spin_held(w) and not math.isfinite(float(w._mdatk_held[0])):
        return (f"The value {w._mdatk_held[0]!r} is not a number this field can show; it is kept until you "
                "change it.")
    return ""


def set_combo(w: Gtk.ComboBox, ident: Any) -> None:
    """Select the item with id ``ident``.  A value that is not among the choices (a newer option, a typo
    in the project file) is added as "<value> (unknown)" and selected: it is shown and read back, instead
    of the item that happened to be selected before.  The added item goes when another value is set."""
    ident = "" if ident is None else str(ident)
    old = getattr(w, "_mdatk_unknown", None)
    found = w.set_active_id(ident)
    if old is not None and old != ident:
        for k, row in enumerate(w.get_model()):
            if row[w.get_id_column()] == old:
                w.remove(k)
                break
        w._mdatk_unknown = None
    if not found:
        w.append(ident, f"{ident} (unknown)" if ident else "(not set)")
        w.set_active_id(ident)
        w._mdatk_unknown = ident


def combo_unknown(w: Gtk.ComboBox) -> Optional[str]:
    """The value :func:`set_combo` added because it is not among the choices, while it is selected."""
    unknown = getattr(w, "_mdatk_unknown", None)
    return unknown if unknown is not None and (w.get_active_id() or "") == unknown else None


def combo_problem(w: Gtk.ComboBox) -> str:
    """What to say about a selected value that is not among the choices ("" for a choice)."""
    unknown = combo_unknown(w)
    return "" if unknown is None else f"Unknown value {unknown!r}: it is not one of this field's choices."


def set_color(w: Gtk.ColorButton, text: Any) -> None:
    """Show the colour ``text`` in a colour button; :func:`color_value` reads the text back as given
    until another colour is picked."""
    w.set_rgba(hex_to_rgba(text if isinstance(text, str) else ""))
    w._mdatk_held = (text, rgba_to_hex(w.get_rgba()))


def color_value(w: Gtk.ColorButton) -> Any:
    """The colour of a colour button: the text :func:`set_color` gave it (tab:orange, #rrggbbaa, gold)
    while the button still shows that colour, else the picked colour as #rrggbb."""
    shown = rgba_to_hex(w.get_rgba())
    held = getattr(w, "_mdatk_held", None)
    return held[0] if held is not None and held[1] == shown else shown


def color_problem(w: Gtk.ColorButton) -> str:
    """What to say about a loaded colour the renderer does not know ("" for a colour)."""
    value = color_value(w)
    if is_color(value):
        return ""
    return f"{value!r} is not a colour (a name such as gold or tab:orange, or #rrggbb): pick one here."


# ---------------------------------------------------------------------------
# widget factories
# ---------------------------------------------------------------------------


def _ignore_wheel(widget, _event) -> bool:
    widget.stop_emission_by_name("scroll-event")  # the widget keeps its value ...
    return False  # ... and the event reaches the scrolled sidebar / dialog instead


def no_wheel(widget):
    """Spin buttons, combo boxes and sliders ignore the mouse wheel, so scrolling a page never changes a
    value by accident (the wheel scrolls the page; the canvas keeps its wheel zoom)."""
    if not getattr(widget, "_mdatk_no_wheel", False):
        widget.connect("scroll-event", _ignore_wheel)
        widget._mdatk_no_wheel = True
    return widget


WHEEL_TYPES = (Gtk.SpinButton, Gtk.ComboBox, Gtk.Range)


def no_wheel_tree(root) -> None:
    """Apply :func:`no_wheel` to every value widget below ``root`` (including its popovers' children)."""
    if isinstance(root, WHEEL_TYPES) and not isinstance(root, Gtk.Scrollbar):
        no_wheel(root)
    if isinstance(root, Gtk.Container):
        for child in root.get_children():
            no_wheel_tree(child)


def spin(lo: float, hi: float, step: float = 1.0, digits: int = 0, tooltip: str = "") -> Gtk.SpinButton:
    adj = Gtk.Adjustment(value=lo, lower=lo, upper=hi, step_increment=step, page_increment=step * 10)
    w = no_wheel(Gtk.SpinButton(adjustment=adj, digits=digits, numeric=True))
    w.set_width_chars(5)
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


def entry(placeholder: str = "", tooltip: str = "", width: int = 20) -> Gtk.Entry:
    w = Gtk.Entry()
    w.set_width_chars(width)
    w.set_hexpand(True)
    if placeholder:
        w.set_placeholder_text(placeholder)
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


def check(label: str, tooltip: str = "") -> Gtk.CheckButton:
    w = Gtk.CheckButton(label=label)
    wrap_child(w)
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


def combo(items: Sequence[Tuple[str, str]], tooltip: str = "") -> Gtk.ComboBoxText:
    """Combo with (id, label) items."""
    w = no_wheel(Gtk.ComboBoxText())
    for ident, label in items:
        w.append(ident, label)
    ellipsize_combo(w)
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


def combo_entry(items: Sequence[str] = (), tooltip: str = "") -> Gtk.ComboBoxText:
    w = no_wheel(Gtk.ComboBoxText.new_with_entry())
    for it in items:
        w.append_text(it)
    w.get_child().set_width_chars(8)
    w.set_hexpand(True)
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


def set_combo_entry_items(w: Gtk.ComboBoxText, items: Sequence[str]) -> None:
    text = w.get_child().get_text()
    w.remove_all()
    for it in items:
        w.append_text(it)
    w.get_child().set_text(text)


def centre_vertically(widget: Gtk.Widget) -> None:
    """Centre ``widget`` (and, for a box, every widget in it) vertically: a tool-bar row is as tall as its tallest
    item (a wrapped check-button label), and buttons, entries and spin buttons would otherwise stretch to it."""
    widget.set_valign(Gtk.Align.CENTER)
    if isinstance(widget, Gtk.Box):
        for child in widget.get_children():
            centre_vertically(child)


def flow_row(groups: Sequence[Tuple[Gtk.Widget, bool, Gtk.Align]], margin: int = 6) -> Gtk.FlowBox:
    """A tool bar made of ``groups`` (widget, expand, horizontal alignment) that puts a group on the next line when
    the window is too narrow for all of them, instead of making the window wider than the screen (the row's
    narrowest width is that of its widest group).  Every widget is centred vertically (centre_vertically); a
    group that expands takes the room left on its line, one aligned to the end sits at the right edge."""
    row = Gtk.FlowBox()
    row.set_selection_mode(Gtk.SelectionMode.NONE)
    row.set_homogeneous(False)
    row.set_min_children_per_line(1)
    row.set_max_children_per_line(len(groups))
    row.set_column_spacing(12)
    row.set_row_spacing(4)
    for side in ("top", "bottom", "start", "end"):
        getattr(row, f"set_margin_{side}")(margin)
    for widget, expand, align in groups:
        row.add(widget)
        child = widget.get_parent()  # the Gtk.FlowBoxChild that holds the group
        child.set_hexpand(expand)
        child.set_halign(align)
        child.set_can_focus(False)  # Tab moves between the buttons, not to the row's cells
        centre_vertically(widget)
    return row


def wrap_child(button: Gtk.Button, chars: int = 48) -> None:
    """Let a check/radio button's label wrap so long texts fit the fixed-width side bar."""
    child = button.get_child()
    if isinstance(child, Gtk.Label):
        child.set_line_wrap(True)
        child.set_max_width_chars(chars)
        child.set_xalign(0.0)


def ellipsize_combo(w: Gtk.ComboBox, chars: int = 12) -> None:
    """Long item labels are shortened with '…' instead of widening the side bar."""
    for cell in w.get_cells():
        if isinstance(cell, Gtk.CellRendererText):
            cell.set_property("ellipsize", Pango.EllipsizeMode.END)
            cell.set_property("width-chars", chars)


def color_button(tooltip: str = "") -> Gtk.ColorButton:
    w = Gtk.ColorButton()
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


def label(text: str, xalign: float = 0.0, markup: bool = False, wrap: bool = False) -> Gtk.Label:
    w = Gtk.Label()
    if markup:
        w.set_markup(text)
    else:
        w.set_text(text)
    w.set_xalign(xalign)
    w.set_line_wrap(wrap)
    if wrap:
        w.set_max_width_chars(90)
    return w


def hint(text: str) -> Gtk.Label:
    w = label(text, wrap=True)
    w.get_style_context().add_class("dim-label")
    return w


def button(text: str, callback: Optional[Callable] = None, tooltip: str = "",
           icon: Optional[str] = None) -> Gtk.Button:
    w = Gtk.Button(label=text)
    if icon:
        w.set_image(Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.BUTTON))
        w.set_always_show_image(True)
    if callback:
        w.connect("clicked", lambda *_: callback())
    if tooltip:
        w.set_tooltip_text(tooltip)
    return w


class Form(Gtk.Grid):
    """Two-column label/widget grid."""

    def __init__(self):
        super().__init__(column_spacing=12, row_spacing=6, margin=8)
        self._row = 0

    def add(self, text: str, widget: Gtk.Widget, tooltip: str = "") -> Gtk.Widget:
        lab = label(text, wrap=True)
        lab.set_max_width_chars(18)  # narrow label column: the side bar has a fixed width
        if tooltip:
            lab.set_tooltip_text(tooltip)
            widget.set_tooltip_text(widget.get_tooltip_text() or tooltip)
        self.attach(lab, 0, self._row, 1, 1)
        self.attach(widget, 1, self._row, 1, 1)
        self._row += 1
        return widget

    def add_wide(self, widget: Gtk.Widget) -> Gtk.Widget:
        self.attach(widget, 0, self._row, 2, 1)
        self._row += 1
        return widget

    def add_row(self, text: str, widgets: Sequence[Gtk.Widget], tooltip: str = "") -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        for w in widgets:
            w.set_hexpand(False)  # GTK 3 boxes honour child hexpand over pack flags
            box.pack_start(w, False, False, 0)
        self.add(text, box, tooltip)


def frame(title: str, child: Gtk.Widget) -> Gtk.Frame:
    f = Gtk.Frame()
    lab = Gtk.Label()
    lab.set_markup(f"<b>{title}</b>")
    f.set_label_widget(lab)
    f.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
    f.add(child)
    f.set_margin_bottom(8)
    return f


def scrolled(child: Gtk.Widget) -> Gtk.ScrolledWindow:
    sw = Gtk.ScrolledWindow()
    sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, margin=10, spacing=4)
    box.pack_start(child, False, False, 0)
    sw.add(box)
    return sw


def filter_name(name: str, patterns: Sequence[str]) -> str:
    """'Trajectory files' + ['*.nc', '*.dcd'] -> 'Trajectory files (.nc, .dcd)'."""
    exts: List[str] = []
    for p in patterns:
        ext = p[1:] if p.startswith("*") else p
        if ext and ext.lower() not in [e.lower() for e in exts]:
            exts.append(ext)
    return f"{name} ({', '.join(exts)})" if exts else name


def add_filters(dlg: Gtk.FileChooser, patterns: Sequence[Tuple[str, Sequence[str]]],
                all_files: bool = True) -> None:
    """File type filters whose names list their extensions; matching ignores case."""
    for name, pats in patterns:
        flt = Gtk.FileFilter()
        flt.set_name(filter_name(name, pats))
        for p in pats:
            flt.add_pattern(p)
            if p.lower() != p:
                flt.add_pattern(p.lower())
            if p.upper() != p:
                flt.add_pattern(p.upper())
        dlg.add_filter(flt)
    if patterns and all_files:
        flt = Gtk.FileFilter()
        flt.set_name("All files")
        flt.add_pattern("*")
        dlg.add_filter(flt)


def file_chooser_button(title: str, action=Gtk.FileChooserAction.OPEN,
                        patterns: Sequence[Tuple[str, Sequence[str]]] = ()) -> Gtk.Box:
    """Entry + browse button (more flexible than Gtk.FileChooserButton)."""
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
    ent = entry(width=18)
    btn = Gtk.Button.new_from_icon_name("document-open-symbolic", Gtk.IconSize.BUTTON)
    btn.set_tooltip_text(f"Browse: {title}")

    def browse(*_):
        dlg = Gtk.FileChooserDialog(title=title, transient_for=box.get_toplevel(), action=action)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OPEN if action != Gtk.FileChooserAction.SAVE else Gtk.STOCK_SAVE,
                        Gtk.ResponseType.OK)
        add_filters(dlg, patterns)
        current = ent.get_text().strip()
        if action == Gtk.FileChooserAction.SAVE:
            dlg.set_do_overwrite_confirmation(True)
            if current:
                dlg.set_current_folder(os.path.dirname(os.path.abspath(current)))
                dlg.set_current_name(os.path.basename(current))
        elif current:
            dlg.set_filename(current)
        if dlg.run() == Gtk.ResponseType.OK:
            ent.set_text(dlg.get_filename() or "")
        dlg.destroy()

    btn.connect("clicked", browse)
    box.pack_start(ent, True, True, 0)
    box.pack_start(btn, False, False, 0)
    box.entry = ent  # type: ignore[attr-defined]
    return box


# ---------------------------------------------------------------------------
# bindings
# ---------------------------------------------------------------------------


class Binding:
    def __init__(self, path: str, widget: Gtk.Widget, getter: Callable[[], Any],
                 setter: Callable[[Any], None], problem: Optional[Callable[[], str]] = None):
        self.path = path
        self.widget = widget
        self.get = getter
        self.set = setter
        self.problem = problem  # what to say about a value the widget holds but cannot show ("": nothing)


class Binder:
    """Keeps a list of widget <-> project-path bindings."""

    def __init__(self):
        self.bindings: List[Binding] = []

    def _add(self, path, widget, getter, setter, problem=None):
        self.bindings.append(Binding(path, widget, getter, setter, problem))
        return widget

    def entry(self, path: str, w: Gtk.Entry):
        return self._add(path, w, lambda: w.get_text().strip(), lambda v: w.set_text(v or ""))

    def path_entry(self, path: str, box):
        ent = box.entry
        self._add(path, box, lambda: ent.get_text().strip(), lambda v: ent.set_text(v or ""))
        return box

    def spin_float(self, path: str, w: Gtk.SpinButton):
        return self._add(path, w, lambda: spin_value(w, float), lambda v: set_spin(w, v),
                         lambda: spin_problem(w))

    def spin_int(self, path: str, w: Gtk.SpinButton):
        return self._add(path, w, lambda: spin_value(w, lambda v: int(round(v))), lambda v: set_spin(w, v),
                         lambda: spin_problem(w))

    def check(self, path: str, w: Gtk.CheckButton):
        return self._add(path, w, lambda: bool(w.get_active()), lambda v: w.set_active(bool(v)))

    def switch(self, path: str, w: Gtk.Switch):
        return self._add(path, w, lambda: bool(w.get_active()), lambda v: w.set_active(bool(v)))

    def combo(self, path: str, w: Gtk.ComboBoxText):
        return self._add(path, w, lambda: w.get_active_id() or "", lambda v: set_combo(w, v),
                         lambda: combo_problem(w))

    def combo_text(self, path: str, w: Gtk.ComboBoxText):
        """Combo with entry bound to a free-text value."""
        child = w.get_child()
        return self._add(path, w, lambda: child.get_text().strip(), lambda v: child.set_text(v or ""))

    def color(self, path: str, w: Gtk.ColorButton):
        return self._add(path, w, lambda: color_value(w), lambda v: set_color(w, v),
                         lambda: color_problem(w))

    def optional_int(self, path: str, w: Gtk.Entry):
        def get():
            t = w.get_text().strip()
            try:
                return int(t) if t else None
            except ValueError:
                raise ValueError(f"{t!r} is not a whole number") from None
        return self._add(path, w, get, lambda v: w.set_text("" if v is None else str(v)))

    def optional_float(self, path: str, w: Gtk.Entry):
        held: List[Any] = []  # the value last shown and its text

        def get():
            t = w.get_text().strip()
            if held and t == held[1]:
                return held[0]  # as loaded
            try:
                return float(t) if t else None
            except ValueError:
                raise ValueError(f"{t!r} is not a number") from None

        def put(v):
            w.set_text("" if v is None else number_text(v))
            held[:] = [v, w.get_text().strip()]
        return self._add(path, w, get, put)

    def csv(self, path: str, w: Gtk.Entry):
        return self._add(path, w, lambda: [t.strip() for t in w.get_text().split(",") if t.strip()],
                         lambda v: w.set_text(", ".join(v or [])))

    def text_view(self, path: str, w: Gtk.TextView):
        buf = w.get_buffer()
        return self._add(path, w,
                         lambda: buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False),
                         lambda v: buf.set_text(v or ""))

    def load(self, project) -> None:
        for b in self.bindings:
            b.set(get_path(project, b.path))

    def unshown(self) -> List[Tuple[str, str]]:
        """(field path, message) of the values the widgets hold as loaded but cannot show: a choice
        that is not in its list, a colour the renderer does not know, a number that is none."""
        found = []
        for b in self.bindings:
            message = b.problem() if b.problem is not None else ""
            if message:
                found.append((b.path, message))
        return found

    def store(self, project, skip: Sequence[str] = ()) -> List[str]:
        """Write widget values into ``project``; returns a list of error messages."""
        return [f"{path}: {message}" for path, message in self.store_fields(project, skip)]

    def store_fields(self, project, skip: Sequence[str] = ()) -> List[Tuple[str, str]]:
        """Like :meth:`store`, with each error as (field path, message); a getter may name a more
        precise field than its binding in the exception's ``field`` attribute."""
        errors = []
        for b in self.bindings:
            if b.path in skip:
                continue
            try:
                set_path(project, b.path, b.get())
            except (ValueError, TypeError) as exc:
                errors.append((getattr(exc, "field", "") or b.path, str(exc)))
        return errors


# ---------------------------------------------------------------------------
# editable list (TreeView backed by a ListStore of strings)
# ---------------------------------------------------------------------------


def keep_edit_on_leave(renderer: Gtk.CellRendererText, store: Gtk.ListStore) -> Callable[[], None]:
    """Leaving a text cell that is being edited (a click on a button, on another row or field) stores what
    was typed, as Enter does: GTK 3 itself drops the edit when the cell loses the focus.  Escape still
    cancels.  The text goes through the renderer's 'edited' signal, i.e. the handlers Enter runs.  Returns
    a function that stores the text of the cell being edited now, if any (before the table is read for
    good: a save, a window that closes); the cell stays open."""
    session: dict = {}

    def commit(editable=None) -> None:
        state = session.get("open")
        if state is None or (editable is not None and state["editable"] is not editable):
            return
        text = state["editable"].get_text()
        try:
            same_row = list(store[state["path"]]) == state["row"]
        except (IndexError, TypeError, ValueError):
            same_row = False
        if not same_row or state["editable"].get_property("editing-canceled") or text == state["text"]:
            return
        state["text"] = text  # first: the handlers may ask for another commit
        renderer.emit("edited", state["path"], text)
        try:
            state["row"] = list(store[state["path"]])
        except (IndexError, TypeError, ValueError):
            session.pop("open", None)

    def close(editable=None) -> None:
        if editable is None or session.get("open", {}).get("editable") is editable:
            session.pop("open", None)

    def leave(editable, _event) -> bool:
        commit(editable)
        close(editable)
        return False  # GTK's own handler follows and closes the cell

    def started(_renderer, editable, path) -> None:
        if not isinstance(editable, Gtk.Entry):
            return
        try:
            row = list(store[path])
        except (IndexError, TypeError, ValueError):
            return
        session["open"] = {"editable": editable, "path": path, "row": row, "text": editable.get_text()}
        editable.connect("editing-done", close)  # Enter (stored by GTK) and Escape
        editable.connect("focus-out-event", leave)

    renderer.connect("editing-started", started)
    renderer.connect("editing-canceled", lambda *_: close())
    return lambda: commit()


class ListEditor(Gtk.Box):
    """Editable table of string columns with add / remove / move buttons."""

    def __init__(self, columns: Sequence[Tuple[str, int]], height: int = 140,
                 defaults: Optional[Sequence[str]] = None,
                 combos: Optional[dict] = None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.n = len(columns)
        self.store = Gtk.ListStore(*([str] * self.n))
        self.view = Gtk.TreeView(model=self.store)
        self.defaults = list(defaults or [""] * self.n)
        combos = combos or {}
        for k, (title, width) in enumerate(columns):
            if k in combos:
                model = Gtk.ListStore(str)
                for item in combos[k]:
                    model.append([item])
                rend = Gtk.CellRendererCombo(model=model, text_column=0, has_entry=True,
                                             editable=True)
            else:
                rend = Gtk.CellRendererText(editable=True)
                keep_edit_on_leave(rend, self.store)  # leaving a cell stores what was typed
            rend.connect("edited", self._edited, k)
            col = Gtk.TreeViewColumn(title, rend, text=k)
            col.set_cell_data_func(rend, self._cell_flag, k)
            col.set_resizable(True)
            col.set_min_width(width)
            col.set_expand(True)
            self.view.append_column(col)
        # cells with a problem: {(row, column): (message, level)}; tinted, the message is the row's tooltip
        self.cell_problems: Dict[Tuple[int, int], Tuple[str, str]] = {}
        self.view.set_has_tooltip(True)
        self.view.connect("query-tooltip", self._cell_tooltip)
        sw = Gtk.ScrolledWindow()
        sw.set_min_content_height(height)
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.add(self.view)
        self.scroller = sw  # carries the table's problem flag
        self.pack_start(sw, True, True, 0)
        self.buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.buttons.pack_start(button("Add", self.add_row, icon="list-add-symbolic"), False, False, 0)
        self.buttons.pack_start(button("Remove", self.remove_selected, icon="list-remove-symbolic"),
                                False, False, 0)
        for icon, tip, delta in (("go-up-symbolic", "Move the selected row up", -1),
                                 ("go-down-symbolic", "Move the selected row down", 1)):
            btn = Gtk.Button.new_from_icon_name(icon, Gtk.IconSize.BUTTON)
            btn.set_tooltip_text(tip)
            btn.connect("clicked", lambda _b, d=delta: self.move(d))
            self.buttons.pack_start(btn, False, False, 0)
        self.pack_start(self.buttons, False, False, 0)
        self.on_change: Optional[Callable[[], None]] = None
        # on_edit(row, column, old, new) runs before a typed value is stored; returning True means it
        # dealt with the edit itself (applied or refused it)
        self.on_edit: Optional[Callable[[int, int, str, str], bool]] = None

    def _edited(self, _rend, path, text, col):
        if self.on_edit is not None and self.on_edit(int(path), col, self.store[path][col], text):
            return
        self.store[path][col] = text
        if self.on_change:
            self.on_change()

    def add_row(self, values: Optional[Sequence[str]] = None):
        self.store.append([str(v) for v in (values or self.defaults)])
        if self.on_change:
            self.on_change()

    def remove_selected(self):
        model, it = self.view.get_selection().get_selected()
        if it is not None:
            model.remove(it)
            if self.on_change:
                self.on_change()

    def move(self, delta: int):
        model, it = self.view.get_selection().get_selected()
        if it is None:
            return
        path = model.get_path(it).get_indices()[0]
        other = path + delta
        if 0 <= other < len(model):
            model.swap(it, model.get_iter(other))
            if self.on_change:
                self.on_change()

    def selected_index(self) -> Optional[int]:
        model, it = self.view.get_selection().get_selected()
        return None if it is None else model.get_path(it).get_indices()[0]

    def rows(self) -> List[List[str]]:
        return [[row[k] for k in range(self.n)] for row in self.store]

    def set_rows(self, rows: Sequence[Sequence[str]]):
        self.store.clear()
        for r in rows:
            self.store.append([str(v) for v in r])

    # ------------------------------------------------------------ problems
    def set_cell_problems(self, problems: Dict[Tuple[int, int], Tuple[str, str]]) -> None:
        """Tint the cells ``{(row, column): (message, level)}`` (an empty dict clears them)."""
        if problems != self.cell_problems:
            self.cell_problems = dict(problems)
            self.view.queue_draw()

    def _cell_flag(self, _col, cell, model, it, k):
        flag = self.cell_problems.get((model.get_path(it).get_indices()[0], k))
        cell.set_property("text", model[it][k])
        if flag:
            cell.set_property("background-rgba", hex_to_rgba(FLAG_COLORS[flag[1]]["cell"]))
        cell.set_property("background-set", bool(flag))

    def _cell_tooltip(self, view, x, y, keyboard, tooltip):
        ok, _x, _y, model, path, _it = view.get_tooltip_context(x, y, keyboard)
        if not ok or path is None:
            return False
        row = path.get_indices()[0]
        texts = dict.fromkeys(msg for (r, _k), (msg, _lv) in sorted(self.cell_problems.items()) if r == row)
        if not texts:
            return False  # the table's own tooltip (its flag) applies
        tooltip.set_text("\n".join(texts))
        view.set_tooltip_row(tooltip, path)
        return True

    def focus_cell(self, row: int, column: Optional[int] = None, focus: bool = True) -> None:
        """Select ``row`` and put the cursor on ``column`` (not edited)."""
        if not 0 <= row < len(self.store):
            return
        path = Gtk.TreePath.new_from_indices([row])
        col = self.view.get_column(column) if column is not None else None
        self.view.set_cursor(path, col, False)
        self.view.scroll_to_cell(path, col, False, 0.0, 0.0)
        if focus:
            self.view.grab_focus()


# ---------------------------------------------------------------------------
# problem flags and badges
# ---------------------------------------------------------------------------

FLAG_COLORS = {"error": {"line": "#e01b24", "cell": "#fbd5d7", "badge": "#e01b24"},
               "warning": {"line": "#e5a50a", "cell": "#fbecc8", "badge": "#c64600"}}
FLAG_ICONS = {"error": "dialog-error-symbolic", "warning": "dialog-warning-symbolic"}
_PROBLEM_CSS = """
.mdatk-badge {{ background-color: {e[badge]}; color: #ffffff; border-radius: 8px; padding: 0 4px;
               font-size: 7.5pt; font-weight: bold; min-width: 8px; }}
.mdatk-badge.warning {{ background-color: {w[badge]}; }}
.mdatk-flag-error, .mdatk-flag-error > box > button, .mdatk-flag-error > button
    {{ border-color: {e[line]}; box-shadow: inset 0 0 0 1px {e[line]}; }}
.mdatk-flag-warning, .mdatk-flag-warning > box > button, .mdatk-flag-warning > button
    {{ border-color: {w[line]}; box-shadow: inset 0 0 0 1px {w[line]}; }}
scrolledwindow.mdatk-flag-error {{ border: 2px solid {e[line]}; }}
scrolledwindow.mdatk-flag-warning {{ border: 2px solid {w[line]}; }}
""".format(e=FLAG_COLORS["error"], w=FLAG_COLORS["warning"])


def install_problem_css() -> None:
    """Styles of problem badges and flags (once per screen)."""
    screen = Gdk.Screen.get_default()
    if screen is None or getattr(screen, "_mdatk_problem_css", False):
        return
    provider = Gtk.CssProvider()
    provider.load_from_data(_PROBLEM_CSS.encode())
    Gtk.StyleContext.add_provider_for_screen(screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    screen._mdatk_problem_css = True


def flag_target(widget: Gtk.Widget) -> Gtk.Widget:
    """The widget that shows a field's flag: the entry of a file chooser box or of a combo with an
    entry, the table of a list editor, else the widget itself."""
    if isinstance(widget, ListEditor):
        return widget.scroller
    # (not asked of an entry: a spin button's "entry" is a field of its C structure, and reading it raises)
    entry_box = None if isinstance(widget, Gtk.Entry) else getattr(widget, "entry", None)
    if isinstance(entry_box, Gtk.Entry):
        return entry_box
    if isinstance(widget, Gtk.ComboBox) and widget.get_has_entry():
        return widget.get_child()
    return widget


def set_flag(widget: Gtk.Widget, message: str = "", level: str = "error") -> None:
    """Show ``message`` as a problem of a value widget, or clear the flag (``message`` ""): GTK's
    error (warning) style and a red (amber) frame, an error icon at the end of entries and the message
    as tooltip, above the field's own hint, which comes back when the flag is cleared."""
    w = flag_target(widget)
    level = level if level in FLAG_ICONS else "error"
    if getattr(w, "_mdatk_flag", ("", "")) == (message, level if message else ""):
        return
    if not hasattr(w, "_mdatk_hint"):
        w._mdatk_hint = w.get_tooltip_text() or ""
    ctx = w.get_style_context()
    for cls in ("error", "warning", "mdatk-flag-error", "mdatk-flag-warning"):
        ctx.remove_class(cls)
    icon = isinstance(w, Gtk.Entry) and not isinstance(w, Gtk.SpinButton)
    if icon and getattr(w, "_mdatk_icon", False):
        w.set_icon_from_icon_name(Gtk.EntryIconPosition.SECONDARY, None)
        w._mdatk_icon = False
    w._mdatk_flag = (message, level if message else "")
    if not message:
        w.set_tooltip_text(w._mdatk_hint or None)
        return
    ctx.add_class(level)
    ctx.add_class(f"mdatk-flag-{level}")
    if icon and not w.get_icon_name(Gtk.EntryIconPosition.SECONDARY):
        w.set_icon_from_icon_name(Gtk.EntryIconPosition.SECONDARY, FLAG_ICONS[level])
        w.set_icon_tooltip_text(Gtk.EntryIconPosition.SECONDARY, message)
        w._mdatk_icon = True
    w.set_tooltip_text(message + (f"\n\n{w._mdatk_hint}" if w._mdatk_hint else ""))


def flag_of(widget: Gtk.Widget) -> Tuple[str, str]:
    """(message, level) of the flag a widget shows (("", "") when none)."""
    return getattr(flag_target(widget), "_mdatk_flag", ("", ""))


def badge_overlay(child: Gtk.Widget) -> Tuple[Gtk.Overlay, Gtk.Label]:
    """``child`` with a count badge in its top-right corner (see :func:`set_badge`)."""
    overlay = Gtk.Overlay()
    overlay.add(child)
    badge = Gtk.Label()
    badge.get_style_context().add_class("mdatk-badge")
    badge.set_halign(Gtk.Align.END)
    badge.set_valign(Gtk.Align.START)
    badge.set_no_show_all(True)  # hidden until there is something to count
    overlay.add_overlay(badge)
    overlay.set_overlay_pass_through(badge, True)
    overlay.show_all()  # a button's image is not reached by the window's show_all()
    return overlay, badge


def set_badge(badge: Gtk.Label, count: int, level: str = "error") -> None:
    """Show ``count`` on a badge (hidden at 0); a "warning" badge is amber instead of red."""
    ctx = badge.get_style_context()
    if level == "warning":
        ctx.add_class("warning")
    else:
        ctx.remove_class("warning")
    badge.set_text(str(count) if count < 100 else "99+")
    badge.set_visible(count > 0)


CHANGE_SIGNALS = ((Gtk.SpinButton, "value-changed"), (Gtk.Entry, "changed"), (Gtk.ComboBox, "changed"),
                  (Gtk.ToggleButton, "toggled"), (Gtk.ColorButton, "color-set"), (Gtk.Range, "value-changed"))


def watch_tree(root: Gtk.Widget, callback: Callable[[], None]) -> None:
    """Call ``callback()`` whenever a value widget below ``root`` changes: entries, spin buttons,
    combos, check buttons, colour buttons, sliders, text views and list editors (each widget once)."""
    if not getattr(root, "_mdatk_watched", False):
        for cls, signal in CHANGE_SIGNALS:
            if isinstance(root, cls):
                root.connect(signal, lambda *_: callback())
                break
        if isinstance(root, Gtk.TextView):
            root.get_buffer().connect("changed", lambda *_: callback())
        if isinstance(root, ListEditor):
            for signal in ("row-changed", "row-deleted", "row-inserted", "rows-reordered"):
                root.store.connect(signal, lambda *_: callback())
        root._mdatk_watched = True
    if isinstance(root, Gtk.Container):
        for child in root.get_children():
            watch_tree(child, callback)
