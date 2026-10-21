"""GTK 3 form helpers and two-way parameter bindings."""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

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


def hex_to_rgba(text: str) -> Gdk.RGBA:
    rgba = Gdk.RGBA()
    if not rgba.parse(text or "#000000"):
        rgba.parse("#000000")
    return rgba


def rgba_to_hex(rgba: Gdk.RGBA) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(round(rgba.red * 255)), int(round(rgba.green * 255)),
                                        int(round(rgba.blue * 255)))


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
                 setter: Callable[[Any], None]):
        self.path = path
        self.widget = widget
        self.get = getter
        self.set = setter


class Binder:
    """Keeps a list of widget <-> project-path bindings."""

    def __init__(self):
        self.bindings: List[Binding] = []

    def _add(self, path, widget, getter, setter):
        self.bindings.append(Binding(path, widget, getter, setter))
        return widget

    def entry(self, path: str, w: Gtk.Entry):
        return self._add(path, w, lambda: w.get_text().strip(), lambda v: w.set_text(v or ""))

    def path_entry(self, path: str, box):
        ent = box.entry
        self._add(path, box, lambda: ent.get_text().strip(), lambda v: ent.set_text(v or ""))
        return box

    def spin_float(self, path: str, w: Gtk.SpinButton):
        return self._add(path, w, lambda: float(w.get_value()), lambda v: w.set_value(float(v)))

    def spin_int(self, path: str, w: Gtk.SpinButton):
        return self._add(path, w, lambda: int(w.get_value_as_int()), lambda v: w.set_value(int(v)))

    def check(self, path: str, w: Gtk.CheckButton):
        return self._add(path, w, lambda: bool(w.get_active()), lambda v: w.set_active(bool(v)))

    def switch(self, path: str, w: Gtk.Switch):
        return self._add(path, w, lambda: bool(w.get_active()), lambda v: w.set_active(bool(v)))

    def combo(self, path: str, w: Gtk.ComboBoxText):
        return self._add(path, w, lambda: w.get_active_id() or "", lambda v: w.set_active_id(str(v)))

    def combo_text(self, path: str, w: Gtk.ComboBoxText):
        """Combo with entry bound to a free-text value."""
        child = w.get_child()
        return self._add(path, w, lambda: child.get_text().strip(), lambda v: child.set_text(v or ""))

    def color(self, path: str, w: Gtk.ColorButton):
        return self._add(path, w, lambda: rgba_to_hex(w.get_rgba()),
                         lambda v: w.set_rgba(hex_to_rgba(v)))

    def optional_int(self, path: str, w: Gtk.Entry):
        def get():
            t = w.get_text().strip()
            return int(t) if t else None
        return self._add(path, w, get, lambda v: w.set_text("" if v is None else str(v)))

    def optional_float(self, path: str, w: Gtk.Entry):
        def get():
            t = w.get_text().strip()
            return float(t) if t else None
        return self._add(path, w, get, lambda v: w.set_text("" if v is None else f"{v:g}"))

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

    def store(self, project, skip: Sequence[str] = ()) -> List[str]:
        """Write widget values into ``project``; returns a list of error messages."""
        errors = []
        for b in self.bindings:
            if b.path in skip:
                continue
            try:
                set_path(project, b.path, b.get())
            except (ValueError, TypeError) as exc:
                errors.append(f"{b.path}: {exc}")
        return errors


# ---------------------------------------------------------------------------
# editable list (TreeView backed by a ListStore of strings)
# ---------------------------------------------------------------------------


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
            rend.connect("edited", self._edited, k)
            col = Gtk.TreeViewColumn(title, rend, text=k)
            col.set_resizable(True)
            col.set_min_width(width)
            col.set_expand(True)
            self.view.append_column(col)
        sw = Gtk.ScrolledWindow()
        sw.set_min_content_height(height)
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.add(self.view)
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

    def _edited(self, _rend, path, text, col):
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
