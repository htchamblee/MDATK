"""Where the main window shows problems: badges on the page buttons, flags on the fields.

:class:`FieldRegistry` maps project field paths (``analyses.hbonds.group_a``, ``groups.1.residues``,
``layout.panels.2.groups``) to the widgets that edit them, so every
:class:`~mdatk.problems.Problem` finds its widget: exactly (the binder's paths), as a cell of a list
editor (a table row and column), through a provider of rebuilt rows (representations, panel slots), or
through its nearest registered parent path.  :class:`ProblemView` shows a list of problems: a count
badge on each page button, the error style, icon and tooltip on each field, and a problems button that
lists them all and jumps to a field when one is clicked.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Pango  # noqa: E402

from ..problems import TABS, Problem, field_parents, list_index  # noqa: E402
from . import widgets as W  # noqa: E402

Widgets = Sequence[Gtk.Widget]


class Target(NamedTuple):
    """The widgets of a field and, for a list editor cell, its (row, column)."""

    widgets: Tuple[Gtk.Widget, ...]
    cell: Optional[Tuple[int, int]] = None


class FieldRegistry:
    def __init__(self):
        self.widgets: Dict[str, Tuple[Gtk.Widget, ...]] = {}
        self.lists: Dict[str, Tuple[W.ListEditor, Dict[str, int], Callable[[List[str]], bool]]] = {}
        self.providers: List[Callable[[], Dict[str, Widgets]]] = []

    def add(self, field: str, *widgets: Gtk.Widget) -> None:
        self.widgets[field] = tuple(widgets)

    def add_list(self, prefix: str, editor: W.ListEditor, columns: Dict[str, int],
                 keep: Callable[[List[str]], bool] = lambda row: True) -> None:
        """``prefix.<i>.<name>`` is cell (row, columns[name]) of ``editor``; the project's item i is the
        i-th row that ``keep`` accepts (the editor skips incomplete rows when it stores)."""
        self.lists[prefix] = (editor, columns, keep)

    def add_provider(self, provider: Callable[[], Dict[str, Widgets]]) -> None:
        """``provider()`` returns {field: widgets} of rows that are rebuilt (asked at each lookup)."""
        self.providers.append(provider)

    def find(self, field: str) -> Optional[Target]:
        dynamic: Dict[str, Widgets] = {}
        for provider in self.providers:
            dynamic.update(provider())
        for path in field_parents(field):
            if path in self.lists:
                editor, columns, keep = self.lists[path]
                index = list_index(field, path)
                if index is not None:
                    rows = [k for k, row in enumerate(editor.rows()) if keep(row)]
                    parts = field[len(path) + 1:].split(".")  # "<index>[.<name>...]"
                    if index < len(rows):
                        column = columns.get(parts[1], -1) if len(parts) > 1 else -1
                        return Target((editor,), (rows[index], column))
                return Target((editor,))
            if path in dynamic:
                return Target(tuple(dynamic[path]))
            if path in self.widgets:
                return Target(self.widgets[path])
        return None


def _summary(problems: Sequence[Problem], limit: int = 6) -> str:
    lines = [f"• {p.location.split(' › ', 1)[-1]}: {p.message}" for p in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"… and {len(problems) - limit} more")
    return "\n".join(lines)


def _level(problems: Sequence[Problem]) -> str:
    return "error" if any(p.level == "error" for p in problems) else "warning"


class ProblemView:
    """Shows problems in the main window (see the module documentation)."""

    def __init__(self, jump: Callable[[Problem], None]):
        W.install_problem_css()
        self.registry = FieldRegistry()
        self.jump = jump  # shows a problem's page and field
        self.problems: List[Problem] = []
        self.badges: Dict[str, Gtk.Label] = {}
        self.page_buttons: Dict[str, Gtk.Widget] = {}
        self._flagged: List[Gtk.Widget] = []
        self._cells: List[W.ListEditor] = []
        self.button = Gtk.MenuButton()
        box = Gtk.Box(spacing=4)
        self._icon = Gtk.Image.new_from_icon_name("dialog-error-symbolic", Gtk.IconSize.BUTTON)
        self._count = Gtk.Label()
        box.pack_start(self._icon, False, False, 0)
        box.pack_start(self._count, False, False, 0)
        self.button.add(box)
        self.button.set_tooltip_text("Problems of the settings: click one to go to its field")
        self.button.set_no_show_all(True)
        box.show_all()
        self.popover = Gtk.Popover()
        self.button.set_popover(self.popover)
        self._list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, margin=6)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._scroller = sw  # as tall as the list, up to 420 px
        sw.add(self._list)
        sw.show_all()  # a popover shows only children that are visible
        self.popover.add(sw)

    def add_page_badge(self, page: str, button: Gtk.Widget, badge: Gtk.Label) -> None:
        self.badges[page] = badge
        self.page_buttons[page] = button
        button._mdatk_title = button.get_tooltip_text() or TABS.get(page, page)

    # ------------------------------------------------------------------ show
    def show(self, problems: Sequence[Problem]) -> None:
        """Flag the fields and badge the pages of ``problems``; clears what is no longer a problem."""
        self.problems = list(problems)
        per_widget: Dict[Gtk.Widget, List[Problem]] = {}
        cells: Dict[W.ListEditor, Dict[Tuple[int, int], Tuple[str, str]]] = {}
        for p in self.problems:
            target = self.registry.find(p.field) if p.field else None
            if target is None:
                continue
            for w in target.widgets:
                per_widget.setdefault(w, []).append(p)
            if target.cell is not None and isinstance(target.widgets[0], W.ListEditor):
                row, col = target.cell
                cols = [col] if col >= 0 else range(target.widgets[0].n)
                for c in cols:
                    old = cells.setdefault(target.widgets[0], {}).get((row, c))
                    text = p.message if old is None else f"{old[0]}\n{p.message}"
                    level = "error" if p.level == "error" or (old and old[1] == "error") else "warning"
                    cells[target.widgets[0]][(row, c)] = (text, level)
        for w in self._flagged:
            if w not in per_widget:
                W.set_flag(w, "")
        for w, items in per_widget.items():
            W.set_flag(w, "\n".join(dict.fromkeys(p.message for p in items)), _level(items))
        self._flagged = list(per_widget)
        for editor in self._cells:
            if editor not in cells:
                editor.set_cell_problems({})
        for editor, flags in cells.items():
            editor.set_cell_problems(flags)
        self._cells = list(cells)
        for page, badge in self.badges.items():
            mine = [p for p in self.problems if p.tab == page]
            W.set_badge(badge, len(mine), _level(mine))
            btn = self.page_buttons[page]
            btn.set_tooltip_text(btn._mdatk_title + (f"\n{_summary(mine)}" if mine else ""))
        self._fill_list()

    def _fill_list(self) -> None:
        for child in self._list.get_children():
            self._list.remove(child)
        n_err = sum(1 for p in self.problems if p.level == "error")
        n_warn = len(self.problems) - n_err
        parts = [f"{n_err} error{'s' if n_err != 1 else ''}"] if n_err else []
        parts += [f"{n_warn} warning{'s' if n_warn != 1 else ''}"] if n_warn else []
        self._count.set_text(", ".join(parts))
        self._icon.set_from_icon_name(W.FLAG_ICONS[_level(self.problems)], Gtk.IconSize.BUTTON)
        self.button.set_visible(bool(self.problems))
        for p in self.problems:
            btn = Gtk.Button()
            btn.set_relief(Gtk.ReliefStyle.NONE)
            row = Gtk.Box(spacing=6)
            row.pack_start(Gtk.Image.new_from_icon_name(W.FLAG_ICONS[p.level if p.level in W.FLAG_ICONS
                                                                     else "error"], Gtk.IconSize.MENU),
                           False, False, 0)
            text = Gtk.Label(xalign=0.0)
            text.set_markup(f"<b>{GLib.markup_escape_text(p.location or 'Project')}</b>\n"
                            f"{GLib.markup_escape_text(p.message)}")
            text.set_line_wrap(True)
            text.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
            text.set_width_chars(56)  # fixed: a wrapped label in a popover would be as narrow as allowed
            text.set_max_width_chars(56)
            row.pack_start(text, True, True, 0)
            btn.add(row)
            btn.connect("clicked", lambda _b, prob=p: self._activate(prob))
            self._list.pack_start(btn, False, False, 0)
        self._list.show_all()
        width = self._list.get_preferred_width()[1]  # wrapped labels: the height depends on the width
        self._scroller.set_min_content_height(min(420, self._list.get_preferred_height_for_width(width)[1]))

    def _activate(self, p: Problem) -> None:
        self.popover.popdown()
        self.jump(p)

    # ------------------------------------------------------------- queries
    def on_page(self, page: str) -> List[Problem]:
        return [p for p in self.problems if p.tab == page]

    def reveal(self, p: Problem, focus: bool = True) -> Optional[Gtk.Widget]:
        """Select a problem's table cell and (``focus``) focus its field; returns the widget to scroll
        to (None when the field has no widget)."""
        target = self.registry.find(p.field) if p.field else None
        if target is None:
            return None
        w = target.widgets[0]
        if target.cell is not None and isinstance(w, W.ListEditor):
            w.focus_cell(target.cell[0], target.cell[1] if target.cell[1] >= 0 else None, focus)
        elif focus:
            field = W.flag_target(w)
            if isinstance(field, Gtk.Entry):
                field.grab_focus_without_selecting()  # a key press must not replace the text
            elif field.get_can_focus():
                field.grab_focus()
        return w
