"""Script wizard: a pop-out window that edits one generator step.

The forms are built from each tool's option schema (check boxes, radio
buttons, combo boxes, spin buttons, text fields - each with its tooltip).
Logic checks run on every change; problems are listed below the form and the
offending fields are highlighted.  A live preview shows the files that
"Write files" will create.  Programs only run when "Write and run" is used
(after a confirmation; each run is logged in the project file).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Pango  # noqa: E402

from ..config import PlanNode  # noqa: E402
from ..projectfile import logged  # noqa: E402
from ..workflows import TOOLS, tools_by_category  # noqa: E402
from ..workflows.naming import new_stamp  # noqa: E402
from ..workflows.plan import Plan, generate_step, scratch_node, single_step_plan  # noqa: E402
from ..workflows.runner import plan_jobs, separate_scripts  # noqa: E402
from ..workflows.schema import (FILE_KINDS, Issue, Option, Section, kind_label,  # noqa: E402
                                requirements_met)
from . import desktop  # noqa: E402
from . import widgets as W  # noqa: E402

LEVEL_COLORS = {"error": "#c01c28", "warning": "#b56a00", "info": "#1a5fb4"}
TYPE_KEYS = ("kind", "integrator")  # record options that switch whole groups of fields
LEVEL_ICONS = {"error": "dialog-error-symbolic", "warning": "dialog-warning-symbolic",
               "info": "dialog-information-symbolic"}


# ---------------------------------------------------------------------------
# schema -> widgets
# ---------------------------------------------------------------------------


class OptionForm(Gtk.Grid):
    """Widgets for a list of options, reading / writing a values dict."""

    def __init__(self, options: Sequence[Option], on_change: Callable[[str], None]):
        super().__init__(column_spacing=10, row_spacing=6, margin=10)
        self.options = list(options)
        self.on_change = on_change
        self.rows: Dict[str, Tuple[Option, Gtk.Widget, Optional[Gtk.Label], Callable, Callable]] = {}
        self._loading = False
        for row, opt in enumerate(self.options):
            self._add(row, opt)

    # building ------------------------------------------------------------
    def _add(self, row: int, opt: Option) -> None:
        tip = opt.tooltip or opt.label
        label: Optional[Gtk.Label] = None
        if opt.kind == "bool":
            w = Gtk.CheckButton(label=opt.label)
            w.set_tooltip_text(tip)
            w.connect("toggled", lambda *_: self._changed(opt.key))
            self.attach(w, 0, row, 2, 1)
            self.rows[opt.key] = (opt, w, None, lambda: w.get_active(), lambda v: w.set_active(bool(v)))
            return
        label = W.label(opt.label + (f" ({opt.unit})" if opt.unit else ""))
        label.set_tooltip_text(tip)
        if opt.kind == "radio":
            box = Gtk.Box(spacing=8)
            buttons: List[Tuple[str, Gtk.RadioButton]] = []
            group = None
            for value, text in opt.choices:
                rb = Gtk.RadioButton.new_with_label_from_widget(group, text)
                group = group or rb
                rb.set_tooltip_text(tip)
                rb.connect("toggled", lambda b, k=opt.key: b.get_active() and self._changed(k))
                box.pack_start(rb, False, False, 0)
                buttons.append((value, rb))

            def get(buttons=buttons):
                for value, rb in buttons:
                    if rb.get_active():
                        return value
                return buttons[0][0]

            def put(v, buttons=buttons):
                for value, rb in buttons:
                    if value == v:
                        rb.set_active(True)

            w = box
            getter, setter = get, put
        elif opt.kind == "choice":
            w = W.combo(list(opt.choices), tip)
            w.connect("changed", lambda *_: self._changed(opt.key))
            getter = (lambda w=w, o=opt: w.get_active_id() or o.choices[0][0])
            setter = (lambda v, w=w: w.set_active_id(str(v)))
        elif opt.kind in ("int", "float") and not opt.optional:
            lo = opt.minimum if opt.minimum is not None else -1e12
            hi = opt.maximum if opt.maximum is not None else 1e12
            digits = 0 if opt.kind == "int" else max(opt.digits, 1)
            step = opt.step or (1 if opt.kind == "int" else 10 ** -min(digits, 3))
            w = W.spin(lo, hi, step, digits, tip)
            w.set_width_chars(12 if hi > 1e6 else 9)
            w.connect("value-changed", lambda *_: self._changed(opt.key))
            if opt.kind == "int":
                getter = (lambda w=w: int(w.get_value()))
            else:
                getter = (lambda w=w: float(w.get_value()))
            setter = (lambda v, w=w: w.set_value(float(v)))
        elif opt.kind in ("int", "float"):
            w = W.entry(opt.placeholder or "default", tip, width=12)
            w.connect("changed", lambda *_: self._changed(opt.key))

            def get_opt(w=w, o=opt):
                text = w.get_text().strip()
                if not text:
                    return None
                try:
                    return o.coerce(text)
                except ValueError:
                    return None

            getter = get_opt
            setter = (lambda v, w=w: w.set_text("" if v is None else f"{v:g}"))
        elif opt.kind == "multiline":
            tv = Gtk.TextView(monospace=True)
            tv.set_wrap_mode(Gtk.WrapMode.NONE)
            tv.set_tooltip_text(tip + (f"\nExample: {opt.placeholder}" if opt.placeholder else ""))
            buf = tv.get_buffer()
            buf.connect("changed", lambda *_: self._changed(opt.key))
            sw = Gtk.ScrolledWindow()
            sw.set_shadow_type(Gtk.ShadowType.IN)
            sw.set_min_content_height(max(64, 18 * opt.rows))
            sw.set_hexpand(True)
            sw.add(tv)
            w = sw
            getter = (lambda b=buf: b.get_text(b.get_start_iter(), b.get_end_iter(), False))
            setter = (lambda v, b=buf: b.set_text(v or ""))
        elif opt.kind == "file":
            w = W.file_chooser_button(opt.label)
            w.set_tooltip_text(tip)
            ent = w.entry
            if opt.placeholder:
                ent.set_placeholder_text(opt.placeholder)
            ent.connect("changed", lambda *_: self._changed(opt.key))
            getter = (lambda e=ent: e.get_text().strip())
            setter = (lambda v, e=ent: e.set_text(v or ""))
        else:  # text
            w = W.entry(opt.placeholder, tip, width=24)
            w.connect("changed", lambda *_: self._changed(opt.key))
            getter = (lambda w=w: w.get_text())
            setter = (lambda v, w=w: w.set_text("" if v is None else str(v)))
        w.set_hexpand(opt.kind in ("text", "multiline", "file"))
        self.attach(label, 0, row, 1, 1)
        self.attach(w, 1, row, 1, 1)
        self.rows[opt.key] = (opt, w, label, getter, setter)

    def _changed(self, key: str) -> None:
        if not self._loading:
            self.on_change(key)

    # values --------------------------------------------------------------
    def load(self, values: Dict[str, Any]) -> None:
        self._loading = True
        try:
            for key, (opt, _w, _l, _g, setter) in self.rows.items():
                if key in values:
                    setter(values[key])
        finally:
            self._loading = False

    def read(self) -> Dict[str, Any]:
        return {key: getter() for key, (_o, _w, _l, getter, _s) in self.rows.items()}

    def update_sensitivity(self, values: Dict[str, Any], hide_keys: Sequence[str] = ()) -> None:
        """Grey out options whose requirements are not met; hide them instead when a
        requirement on one of ``hide_keys`` (e.g. a stage's type) fails."""
        for key, (opt, w, label, _g, _s) in self.rows.items():
            on = opt.is_enabled(values)
            hidden = any(k in hide_keys and not requirements_met({k: want}, values)
                         for k, want in opt.requires.items())
            w.set_sensitive(on)
            w.set_no_show_all(hidden)
            if hidden:
                w.hide()
            else:
                w.show_all()  # children of a widget hidden before its first show need show_all
            if label is not None:
                label.set_sensitive(on)
                label.set_no_show_all(hidden)
                label.set_visible(not hidden)

    def mark(self, levels: Dict[str, str]) -> None:
        """Colour the labels of options that have issues ({key: level})."""
        for key, (opt, w, label, _g, _s) in self.rows.items():
            level = levels.get(key)
            text = opt.label + (f" ({opt.unit})" if opt.unit and label is not None else "")
            markup = GLib.markup_escape_text(text)
            if level in ("error", "warning"):
                markup = f'<span foreground="{LEVEL_COLORS[level]}" weight="bold">{markup}</span>'
            target = label if label is not None else w.get_child()
            if isinstance(target, Gtk.Label):
                target.set_markup(markup)


class RecordEditor(Gtk.Paned):
    """List of records (e.g. MD stages) with a form for the selected one."""

    def __init__(self, section: Section, on_change: Callable[[str], None]):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.section = section
        self.on_change = on_change
        self.records: List[Dict[str, Any]] = []
        self.current = -1
        self._selecting = False
        self.store = Gtk.ListStore(str, str)
        self.view = Gtk.TreeView(model=self.store)
        self.view.set_headers_visible(False)
        rend = Gtk.CellRendererText()
        self.view.append_column(Gtk.TreeViewColumn("#", rend, text=0))
        rend2 = Gtk.CellRendererText()
        rend2.set_property("ellipsize", Pango.EllipsizeMode.END)
        col = Gtk.TreeViewColumn(section.title, rend2, text=1)
        col.set_expand(True)
        self.view.append_column(col)
        self.view.get_selection().connect("changed", self._on_select)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=6)
        sw = Gtk.ScrolledWindow()
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.set_min_content_width(170)
        sw.add(self.view)
        left.pack_start(sw, True, True, 0)
        bar = Gtk.Box(spacing=2)
        add = Gtk.MenuButton()
        add.set_image(Gtk.Image.new_from_icon_name("list-add-symbolic", Gtk.IconSize.BUTTON))
        add.set_tooltip_text("Add a stage from a template")
        menu = Gtk.Menu()
        for tid, label, _vals in section.templates:
            item = Gtk.MenuItem(label=label)
            item.connect("activate", lambda _w, t=tid: self.add(t))
            menu.append(item)
        menu.show_all()
        add.set_popup(menu)
        bar.pack_start(add, False, False, 0)
        for icon, tip, cb in (("edit-copy-symbolic", "Duplicate", self.duplicate),
                              ("list-remove-symbolic", "Remove", self.remove),
                              ("go-up-symbolic", "Move up", lambda: self.move(-1)),
                              ("go-down-symbolic", "Move down", lambda: self.move(1))):
            b = Gtk.Button.new_from_icon_name(icon, Gtk.IconSize.BUTTON)
            b.set_tooltip_text(tip)
            b.connect("clicked", lambda *_, c=cb: c())
            bar.pack_start(b, False, False, 0)
        left.pack_start(bar, False, False, 0)
        self.pack1(left, False, False)
        self.form = OptionForm(section.options, self._form_changed)
        self.form_sw = Gtk.ScrolledWindow()
        self.form_sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.form_sw.add(self.form)
        self.pack2(self.form_sw, True, False)

    def set_records(self, records: List[Dict[str, Any]]) -> None:
        self.records = records
        self._refresh_list(select=0 if records else -1)

    def _refresh_list(self, select: int) -> None:
        self._selecting = True
        self.store.clear()
        for k, rec in enumerate(self.records):
            name = str(rec.get(self.section.item_label) or f"#{k + 1}")
            kind = rec.get("kind") or rec.get("integrator") or ""
            self.store.append([str(k + 1), f"{name}  ({kind})" if kind else name])
        self._selecting = False
        if 0 <= select < len(self.records):
            self.view.get_selection().select_path(Gtk.TreePath(select))
        else:
            self.current = -1
            self.form.set_sensitive(False)

    def _on_select(self, sel) -> None:
        if self._selecting:
            return
        model, it = sel.get_selected()
        if it is None:
            return
        self.current = model.get_path(it).get_indices()[0]
        self.form.set_sensitive(True)
        self.form.load(self.records[self.current])
        self.form.update_sensitivity(self.records[self.current], TYPE_KEYS)

    def select(self, index: int) -> None:
        if 0 <= index < len(self.records):
            self.view.get_selection().select_path(Gtk.TreePath(index))

    def _form_changed(self, key: str) -> None:
        if not (0 <= self.current < len(self.records)):
            return
        rec = self.records[self.current]
        rec.update(self.form.read())
        self.form.update_sensitivity(rec, TYPE_KEYS)
        if key in (self.section.item_label, "kind", "integrator"):
            it = self.store.get_iter(Gtk.TreePath(self.current))
            name = str(rec.get(self.section.item_label) or f"#{self.current + 1}")
            kind = rec.get("kind") or rec.get("integrator") or ""
            self.store.set_value(it, 1, f"{name}  ({kind})" if kind else name)
        self.on_change(f"{self.section.key}.{self.current}.{key}")

    def add(self, template: str) -> None:
        rec = self.section.template(template)
        pos = self.current + 1 if self.current >= 0 else len(self.records)
        self.records.insert(pos, rec)
        self._refresh_list(pos)
        self.on_change(self.section.key)

    def duplicate(self) -> None:
        if 0 <= self.current < len(self.records):
            rec = copy.deepcopy(self.records[self.current])
            rec[self.section.item_label] = f"{rec.get(self.section.item_label, '')}-copy"
            self.records.insert(self.current + 1, rec)
            self._refresh_list(self.current + 1)
            self.on_change(self.section.key)

    def remove(self) -> None:
        if 0 <= self.current < len(self.records):
            del self.records[self.current]
            self._refresh_list(min(self.current, len(self.records) - 1))
            self.on_change(self.section.key)

    def move(self, delta: int) -> None:
        k = self.current
        j = k + delta
        if 0 <= k < len(self.records) and 0 <= j < len(self.records):
            self.records[k], self.records[j] = self.records[j], self.records[k]
            self._refresh_list(j)
            self.on_change(self.section.key)

    def mark(self, issues_by_key: Dict[str, str]) -> None:
        if 0 <= self.current < len(self.records):
            prefix = f"{self.section.key}.{self.current}."
            self.form.mark({k[len(prefix):]: v for k, v in issues_by_key.items() if k.startswith(prefix)})


# ---------------------------------------------------------------------------
# the wizard window
# ---------------------------------------------------------------------------


class WizardWindow(Gtk.Window):
    """Edit and write one step.

    Stand-alone (``node`` is None): a tool list on the left; settings are kept
    per tool in the project's workflow scratch area.  From the planner: the
    node's tool is fixed and its inputs come from the plan's connections.
    """

    def __init__(self, main, node: Optional[PlanNode] = None,
                 plan_provider: Optional[Callable[[], Plan]] = None,
                 on_changed: Optional[Callable[[], None]] = None, tool: Optional[str] = None):
        super().__init__(title="Script wizard")
        self.main = main
        self.planner_mode = node is not None
        self.plan_provider = plan_provider
        self.on_changed = on_changed
        self.node: Optional[PlanNode] = node
        self.forms: Dict[str, Any] = {}  # section key -> OptionForm / RecordEditor
        self.section_pages: Dict[str, int] = {}
        self.input_widgets: Dict[str, Gtk.Widget] = {}
        self._ports_shown: List[Tuple] = []  # input ports on the Files page (see _port_signature)
        self.plan: Optional[Plan] = None
        self.files = []
        self._refresh_id = 0
        self._loading = False
        self.set_default_size(1180, 820)
        if main is not None:
            self.set_transient_for(main)

        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.add(outer)
        if not self.planner_mode:
            outer.pack_start(self._tool_list(), False, False, 0)
            outer.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 0)
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        outer.pack_start(right, True, True, 0)
        right.pack_start(self._header(), False, False, 0)
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self.notebook = Gtk.Notebook()
        self.notebook.set_scrollable(True)
        paned.pack1(self.notebook, True, False)
        paned.pack2(self._bottom(), False, False)
        paned.set_position(470)
        right.pack_start(paned, True, True, 0)
        right.pack_start(self._actions(), False, False, 0)
        self.connect("delete-event", self._on_close)

        if self.planner_mode:
            self._load_node(self.node)
        else:
            first = tool or self._last_tool()
            self.select_tool(first)
        self.show_all()

    # ----------------------------------------------------------- chrome
    @property
    def workflow(self):
        return self.main.project.workflow

    def _last_tool(self) -> str:
        if self.workflow.scratch:
            return self.workflow.scratch[-1].tool
        return "amber-md"

    def _tool_list(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.tool_list = Gtk.ListBox()
        self.tool_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.tool_rows: Dict[str, Gtk.ListBoxRow] = {}
        for category, tools in tools_by_category():
            gen = [t for t in tools if t.generates]
            if not gen:
                continue
            head = Gtk.ListBoxRow()
            head.set_selectable(False)
            head.set_activatable(False)
            lab = W.label(f"<b>{GLib.markup_escape_text(category)}</b>", markup=True)
            lab.set_margin_top(8)
            lab.set_margin_start(8)
            head.add(lab)
            self.tool_list.add(head)
            for t in gen:
                row = Gtk.ListBoxRow()
                lab = W.label(t.label)
                lab.set_margin_start(16)
                lab.set_margin_top(3)
                lab.set_margin_bottom(3)
                row.add(lab)
                row.set_tooltip_text(t.description)
                row.tool_name = t.name  # type: ignore[attr-defined]
                self.tool_list.add(row)
                self.tool_rows[t.name] = row
        self.tool_list.connect("row-selected", self._on_tool_row)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sw.set_min_content_width(230)
        sw.add(self.tool_list)
        box.pack_start(sw, True, True, 0)
        return box

    def _header(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=10)
        self.title_label = W.label("", markup=True)
        box.pack_start(self.title_label, False, False, 0)
        self.desc_label = W.hint("")
        box.pack_start(self.desc_label, False, False, 0)
        row = Gtk.Box(spacing=8)
        row.pack_start(W.label("Project"), False, False, 0)
        self.project_entry = W.entry("project name", "The <project> part of every file name.", width=12)
        self.project_entry.set_hexpand(False)
        self.project_entry.connect("changed", self._on_naming)
        row.pack_start(self.project_entry, False, False, 0)
        row.pack_start(W.label("Step"), False, False, 0)
        self.step_spin = W.spin(1, 99, 1, 0, "The <step> number in file names and the step folder.")
        self.step_spin.connect("value-changed", self._on_naming)
        row.pack_start(self.step_spin, False, False, 0)
        row.pack_start(W.label("Title"), False, False, 0)
        self.title_entry = W.entry("(tool name)", "Name shown on the planner box.", width=16)
        self.title_entry.set_hexpand(False)
        self.title_entry.connect("changed", self._on_naming)
        row.pack_start(self.title_entry, False, False, 0)
        self.stamp_check = W.check("Time stamp", "Append the plan's time stamp to file names "
                                                "(<project>_<step>_<tool>_<time>.<ext>).")
        self.stamp_check.connect("toggled", self._on_naming)
        row.pack_start(self.stamp_check, False, False, 0)
        self.stamp_label = W.label("")
        row.pack_start(self.stamp_label, False, False, 0)
        row.pack_start(W.button("New", self._new_stamp, "Start a new time stamp (now)"), False, False, 0)
        box.pack_start(row, False, False, 0)
        row2 = Gtk.Box(spacing=8)
        row2.pack_start(W.label("Folder"), False, False, 0)
        self.folder_label = W.label("")
        self.folder_label.set_ellipsize(Pango.EllipsizeMode.START)
        self.folder_label.set_hexpand(True)
        row2.pack_start(self.folder_label, True, True, 0)
        if not self.planner_mode:
            row2.pack_start(W.button("Change…", self._choose_folder, "Folder that receives the step folders"),
                            False, False, 0)
        box.pack_start(row2, False, False, 0)
        return box

    def _bottom(self) -> Gtk.Widget:
        nb = Gtk.Notebook()
        self.issue_list = Gtk.ListBox()
        self.issue_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self.issue_list.connect("row-activated", self._on_issue)
        sw = Gtk.ScrolledWindow()
        sw.add(self.issue_list)
        self.issue_tab = W.label("Checks")
        nb.append_page(sw, self.issue_tab)
        prev = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=4)
        top = Gtk.Box(spacing=6)
        top.pack_start(W.label("File"), False, False, 0)
        self.preview_combo = W.no_wheel(Gtk.ComboBoxText())
        self.preview_combo.connect("changed", self._show_preview)
        top.pack_start(self.preview_combo, True, True, 0)
        prev.pack_start(top, False, False, 0)
        self.preview = Gtk.TextView(monospace=True, editable=False)
        self.preview.set_wrap_mode(Gtk.WrapMode.NONE)
        sw2 = Gtk.ScrolledWindow()
        sw2.set_shadow_type(Gtk.ShadowType.IN)
        sw2.add(self.preview)
        prev.pack_start(sw2, True, True, 0)
        nb.append_page(prev, W.label("Preview"))
        nb.set_size_request(-1, 230)
        self.bottom_nb = nb
        return nb

    def _actions(self) -> Gtk.Widget:
        bar = Gtk.Box(spacing=6, margin=8)
        self.status = W.hint("")
        bar.pack_start(self.status, True, True, 0)
        bar.pack_end(W.button("Close", self.close_window), False, False, 0)
        self.open_btn = W.button("Open folder", self.open_folder, "Open the step folder",
                                 "folder-open-symbolic")
        bar.pack_end(self.open_btn, False, False, 0)
        self.write_btn = W.button("Write files", self.write_files, "Write this step's files "
                                  "(blocked while there are errors)", "document-save-symbolic")
        self.write_btn.get_style_context().add_class("suggested-action")
        bar.pack_end(self.write_btn, False, False, 0)
        self.run_btn = W.button("Write and run…", lambda: self.write_files(run=True),
                                "Write this step's files, then run its main script on this computer "
                                "(asks first; logged in the project file)", "media-playback-start-symbolic")
        bar.pack_end(self.run_btn, False, False, 0)
        bar.pack_end(W.button("Defaults", self.reset_defaults, "Reset this tool's options"), False, False, 0)
        return bar

    # ----------------------------------------------------------- loading
    def _on_tool_row(self, _lb, row) -> None:
        if row is not None and hasattr(row, "tool_name") and not self._loading:
            self.select_tool(row.tool_name)

    def select_tool(self, name: str) -> None:
        if name not in TOOLS or not TOOLS[name].generates:
            name = "amber-md"
        node = scratch_node(self.workflow, name)
        # keep the most recently used tool last so the wizard reopens on it
        self.workflow.scratch.remove(node)
        self.workflow.scratch.append(node)
        row = self.tool_rows.get(name)
        if row is not None and self.tool_list.get_selected_row() is not row:
            self._loading = True
            self.tool_list.select_row(row)
            self._loading = False
        self._load_node(node)

    def _load_node(self, node: PlanNode) -> None:
        self.node = node
        tool = TOOLS[node.tool]
        self.tool = tool
        node.settings = tool.normalise(node.settings)
        self._loading = True
        try:
            self.title_label.set_markup(f"<big><b>{GLib.markup_escape_text(tool.label)}</b></big>")
            self.desc_label.set_text(tool.description)
            self.project_entry.set_text(self.workflow.project_name)
            self.step_spin.set_value(node.step)
            self.title_entry.set_text(node.title)
            self.stamp_check.set_active(self.workflow.use_stamp)
            self.project_entry.set_sensitive(not self.planner_mode)
            self.stamp_check.set_sensitive(not self.planner_mode)
            self.set_title(f"Script wizard - {tool.label}")
        finally:
            self._loading = False
        self._build_pages()
        self.refresh_now()

    def _build_pages(self) -> None:
        for child in list(self.notebook.get_children()):
            self.notebook.remove(child)
        self.forms.clear()
        self.section_pages.clear()
        self.input_widgets.clear()
        tool, node = self.tool, self.node
        files_page = self._files_page()
        self.notebook.append_page(files_page, W.label("Files"))
        for section in tool.sections:
            if section.repeat:
                editor = RecordEditor(section, self._on_option_change)
                editor.set_records(node.settings.setdefault(section.key, section.default_records()))
                page = editor
                self.forms[section.key] = editor
            else:
                form = OptionForm(section.options, self._on_option_change)
                form.load(node.settings)
                form.update_sensitivity(node.settings)
                self.forms[section.key] = form
                box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                if section.tooltip:
                    hint = W.hint(section.tooltip)
                    hint.set_margin_start(10)
                    hint.set_margin_top(8)
                    box.pack_start(hint, False, False, 0)
                box.pack_start(form, False, False, 0)
                page = Gtk.ScrolledWindow()
                page.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
                page.add(box)
            lab = W.label(section.title)
            lab.set_tooltip_text(section.tooltip or section.title)
            self.section_pages[section.key] = self.notebook.append_page(page, lab)
        self.notebook.show_all()

    def _files_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=10)
        box.pack_start(W.label("<b>Input files</b>", markup=True), False, False, 0)
        self.inputs_grid = Gtk.Grid(column_spacing=10, row_spacing=6)
        box.pack_start(self.inputs_grid, False, False, 0)
        box.pack_start(W.label("<b>Files this step produces</b>", markup=True), False, False, 0)
        self.outputs_label = W.label("", wrap=True)
        self.outputs_label.set_selectable(True)
        box.pack_start(self.outputs_label, False, False, 0)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sw.add(box)
        self._fill_inputs()
        return sw

    def _connected(self) -> Dict[str, str]:
        """Input port -> source description for connected ports (planner mode)."""
        if not self.planner_mode:
            return {}
        out = {}
        for e in self.workflow.edges:
            if e.target == self.node.id:
                src = next((n for n in self.workflow.nodes if n.id == e.source), None)
                out[e.input] = f"connected: {src.title or TOOLS[src.tool].label if src else e.source} " \
                               f"→ {e.output}"
        return out

    def _port_signature(self) -> List[Tuple]:
        """What the Files page shows of the input ports (records edit them in place, e.g. custom scripts)."""
        return [(p.name, p.label, p.kinds, p.optional, p.multiple)
                for p in self.tool.input_ports(self.node.settings)]

    def _fill_inputs(self) -> None:
        for child in list(self.inputs_grid.get_children()):
            self.inputs_grid.remove(child)
        self.input_widgets.clear()
        self._ports_shown = self._port_signature()
        ports = self.tool.input_ports(self.node.settings)
        connected = self._connected()
        if not ports:
            self.inputs_grid.attach(W.hint("This step needs no input files."), 0, 0, 2, 1)
        for row, port in enumerate(ports):
            text = port.label + ("" if not port.optional else " (optional)")
            lab = W.label(text)
            tip = port.tooltip or ""
            kinds = ", ".join(kind_label(k) for k in port.kinds)
            lab.set_tooltip_text(f"{tip}\nAccepts: {kinds}".strip())
            self.inputs_grid.attach(lab, 0, row, 1, 1)
            if port.name in connected:
                info = W.label(connected[port.name])
                info.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
                info.set_selectable(True)
                self.inputs_grid.attach(info, 1, row, 1, 1)
                self.input_widgets[port.name] = info
                continue
            patterns = [(kinds, [f"*{ext}" for k in port.kinds for ext in FILE_KINDS.get(k, ("", ()))[1]])]
            chooser = W.file_chooser_button(port.label, patterns=patterns)
            chooser.set_hexpand(True)
            ent = chooser.entry
            ent.set_width_chars(48)
            ent.set_text(self.node.files.get(port.name, ""))
            ent.set_tooltip_text(("Several files: separate with ';'. " if port.multiple else "") +
                                 f"Accepts: {kinds}")
            ent.connect("changed", lambda e, p=port.name: self._on_input_change(p, e.get_text()))
            self.inputs_grid.attach(chooser, 1, row, 1, 1)
            self.input_widgets[port.name] = chooser
        self.inputs_grid.show_all()

    # ----------------------------------------------------------- changes
    def _on_input_change(self, port: str, text: str) -> None:
        if text.strip():
            self.node.files[port] = text.strip()
        else:
            self.node.files.pop(port, None)
        self.schedule_refresh()

    def _on_option_change(self, key: str) -> None:
        if self._loading:
            return
        values = dict(self.node.settings)
        for section in self.tool.sections:
            form = self.forms.get(section.key)
            if isinstance(form, OptionForm):
                values.update(form.read())
        self.node.settings = values
        for section in self.tool.sections:
            form = self.forms.get(section.key)
            if isinstance(form, OptionForm):
                form.update_sensitivity(values)
        if self._port_signature() != self._ports_shown:
            self._fill_inputs()
        self.schedule_refresh()

    def _on_naming(self, *_):
        if self._loading or self.node is None:
            return
        self.node.step = int(self.step_spin.get_value())
        self.node.title = self.title_entry.get_text().strip()
        if not self.planner_mode:
            self.workflow.project_name = self.project_entry.get_text().strip() or "project"
            self.workflow.use_stamp = self.stamp_check.get_active()
        self.schedule_refresh()

    def _new_stamp(self):
        self.workflow.stamp = new_stamp()
        self.schedule_refresh()

    def _choose_folder(self):
        dlg = Gtk.FileChooserDialog(title="Folder for generated steps", transient_for=self,
                                    action=Gtk.FileChooserAction.SELECT_FOLDER)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Select", Gtk.ResponseType.OK)
        if dlg.run() == Gtk.ResponseType.OK and dlg.get_filename():
            self.workflow.out_dir = dlg.get_filename()
        dlg.destroy()
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if self._refresh_id:
            GLib.source_remove(self._refresh_id)
        self._refresh_id = GLib.timeout_add(250, self._refresh_timeout)

    def _refresh_timeout(self):
        self._refresh_id = 0
        self.refresh_now()
        return False

    # ----------------------------------------------------------- refresh
    def _resolve(self) -> Plan:
        if self.planner_mode and self.plan_provider is not None:
            return self.plan_provider()
        return single_step_plan(self.workflow, self.node, self.main.project.workflow_dir,
                                self.main.project.groups)

    def refresh_now(self) -> None:
        if self.node is None:
            return
        self.plan = plan = self._resolve()
        step = plan.steps.get(self.node.id)
        issues: List[Issue] = list(step.issues) if step else []
        stamp = plan.stamp if self.workflow.use_stamp else ""
        self.stamp_label.set_text(stamp + ("" if self.workflow.stamp or not stamp else " (provisional)"))
        root = self.main.project.workflow_dir
        folder = step.ctx.folder if step and step.ctx else ""
        self.folder_label.set_text(os.path.join(root, folder))
        self.files = []
        if step is not None and step.ok and step.ctx is not None:
            try:
                step.ctx.preview = True  # e.g. no PubChem requests on every change
                self.files = generate_step(step)
            except Exception as exc:  # pragma: no cover - shown to the user
                issues.append(Issue("error", f"cannot generate the files: {exc}"))
        self._show_issues(issues)
        outs = []
        if step is not None:
            for port, value in step.outputs.items():
                shown = value.paths[:3] + (["…"] if len(value.paths) > 3 else [])
                outs.append(f"• {port}: " + ", ".join(shown))
        self.outputs_label.set_text("\n".join(outs) or "(none)")
        if step is not None and step.ctx is not None:
            for port, widget in self.input_widgets.items():
                value = step.ctx.inputs.get(port)
                if isinstance(widget, Gtk.Label) and value is not None and value.source:
                    shown = ", ".join(value.paths[:2]) + (" …" if len(value.paths) > 2 else "")
                    widget.set_text(f"connected: {shown}  (from {value.source_label})")
        current = self.preview_combo.get_active_text()
        self.preview_combo.remove_all()
        for f in self.files:
            self.preview_combo.append(f.name, f.name)
        names = [f.name for f in self.files]
        if current in names:
            self.preview_combo.set_active_id(current)
        elif names:
            self.preview_combo.set_active(len(names) - 1)
        else:
            self.preview.get_buffer().set_text("(fix the errors listed under Checks to see the files)")
        errors = [i for i in issues if i.level == "error"]
        self.write_btn.set_sensitive(not errors and bool(self.files))
        self.run_btn.set_sensitive(not errors and any(f.role == "run" for f in self.files))
        n_err, n_warn = len(errors), len([i for i in issues if i.level == "warning"])
        self.status.set_text(f"{len(self.files)} files · {n_err} errors · {n_warn} warnings")
        if self.on_changed is not None:
            self.on_changed()

    def _show_issues(self, issues: Sequence[Issue]) -> None:
        for child in list(self.issue_list.get_children()):
            self.issue_list.remove(child)
        levels: Dict[str, str] = {}
        order = {"error": 0, "warning": 1, "info": 2}
        for issue in sorted(issues, key=lambda i: order[i.level]):
            row = Gtk.ListBoxRow()
            box = Gtk.Box(spacing=8, margin=4)
            box.pack_start(Gtk.Image.new_from_icon_name(LEVEL_ICONS[issue.level], Gtk.IconSize.MENU),
                           False, False, 0)
            lab = W.label(issue.message, wrap=True)
            box.pack_start(lab, True, True, 0)
            row.add(box)
            row.issue = issue  # type: ignore[attr-defined]
            self.issue_list.add(row)
            for key in issue.keys:
                if order[issue.level] < order.get(levels.get(key, "info"), 3) or key not in levels:
                    levels[key] = issue.level
        if not issues:
            row = Gtk.ListBoxRow()
            row.add(W.label("  ✓ No problems found.", markup=False))
            self.issue_list.add(row)
        self.issue_list.show_all()
        n = len([i for i in issues if i.level != "info"])
        self.issue_tab.set_text(f"Checks ({n})" if n else "Checks")
        for key, form in self.forms.items():
            form.mark(levels)

    def _show_preview(self, *_):
        name = self.preview_combo.get_active_id()
        for f in self.files:
            if f.name == name:
                self.preview.get_buffer().set_text(f.content)
                return

    def _on_issue(self, _lb, row) -> None:
        issue = getattr(row, "issue", None)
        if issue is None or not issue.keys:
            return
        key = issue.keys[0]
        if key.startswith("input:"):
            self.notebook.set_current_page(0)
            return
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] in self.section_pages and parts[1].isdigit():
            self.notebook.set_current_page(self.section_pages[parts[0]])
            self.forms[parts[0]].select(int(parts[1]))
            return
        for section in self.tool.sections:
            if section.key == key or any(o.key == key for o in section.options):
                self.notebook.set_current_page(self.section_pages[section.key])
                return

    # ----------------------------------------------------------- actions
    def reset_defaults(self) -> None:
        self.node.settings = self.tool.defaults()
        self._build_pages()
        self.refresh_now()

    def write_files(self, run: bool = False) -> None:
        plan = self._resolve()
        step = plan.steps.get(self.node.id)
        if step is None or not step.ok:
            self.refresh_now()
            return
        try:
            with logged(getattr(self.main, "project_path", None), "workflow write",
                        f"Script wizard: {step.tool.label}") as entry:
                report = plan.write([self.node.id], master=False)
                entry.files = list(report.written)
                entry.summary = f"{len(report.written)} file(s) written"
        except OSError as exc:
            self._error("Could not write the files", str(exc))
            return
        if self.workflow.use_stamp and not self.workflow.stamp:
            self.workflow.stamp = plan.stamp
        if report.skipped:
            self._error("Some files were not written", "\n".join(report.skipped.values()))
        written = [p for p in report.written]
        if self.main is not None and hasattr(self.main, "append_log"):
            self.main.append_log(f"Wizard wrote {len(written)} files to {os.path.dirname(written[0])}"
                                 if written else "Wizard wrote no files")
        self.status.set_text(f"Wrote {len(written)} files to "
                             f"{os.path.dirname(written[0]) if written else '-'}")
        self.refresh_now()
        if run and not report.skipped and hasattr(self.main, "run_scripts"):
            self.main.run_scripts(plan_jobs(plan, [self.node.id]), separate_scripts(plan, [self.node.id]),
                                  parent=self)

    def step_folder(self) -> str:
        plan = self.plan or self._resolve()
        step = plan.steps.get(self.node.id)
        folder = step.ctx.folder if step and step.ctx else ""
        return os.path.join(self.main.project.workflow_dir, folder)

    def open_folder(self) -> None:
        path = self.step_folder()
        if not os.path.isdir(path):
            path = self.main.project.workflow_dir
        if not os.path.isdir(path):
            self._error("Nothing written yet", f"{path} does not exist yet; write the files first.")
            return
        desktop.open_folder(path)

    def _error(self, text: str, detail: str = "") -> None:
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.ERROR,
                                buttons=Gtk.ButtonsType.CLOSE, text=text)
        if detail:
            dlg.format_secondary_text(detail)
        dlg.run()
        dlg.destroy()

    def close_window(self) -> None:
        self._on_close()
        self.destroy()

    def _on_close(self, *_):
        if self._refresh_id:
            GLib.source_remove(self._refresh_id)
            self._refresh_id = 0
        if self.main is not None and getattr(self.main, "wizard", None) is self:
            self.main.wizard = None
        return False
