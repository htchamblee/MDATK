"""RNAdo's output log panel: planner messages and the output of the scripts it runs.

Lines arrive from any thread (:meth:`LogPanel.feed`, :meth:`LogPanel.feed_script`) and are inserted on
the GTK main loop in batches, so a chatty script never floods the loop with one callback per line.  A batch
holds at most :data:`MAX_BATCH` lines, the newest: when a script prints faster than the panel can show
(``cat`` of a large file, ``seq 1 1000000``) the panel skips lines, and a note says how many, but the window
stays responsive.
Every script line carries its step number (``[03] ``); stderr and lines starting with ERROR or WAITING
are highlighted.  The panel filters by step, shows the stored log of a step's last run (from the
project's step log), and keeps at most :data:`MAX_LINES` lines like the analysis window's log.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk, Pango  # noqa: E402

from . import widgets as W  # noqa: E402

MAX_LINES = 6000  # beyond this the oldest lines go ...
KEEP_LINES = 5000  # ... down to this many
FLUSH_MS = 40  # lines that arrive within this time are inserted together
# Lines one flush inserts at most, the newest (a note counts the others): a flush then takes well under
# 0.1 s, so Cancel stays reachable.  Below KEEP_LINES, so the note survives the trimming.
MAX_BATCH = 2000
# A thread that floods the panel pauses this long after every MAX_BATCH lines it queues: the main loop's
# Python code (drawing the planner, the progress bar) waits for the GIL after every GTK call, and without
# the pause it would wait for that thread most of the time.
FLOOD_PAUSE_S = 0.001
_MARK = re.compile(r"\[(\d{2,})\] ")
_STATUS = re.compile(r"==== step (\d+):")
TAGS = {  # tag -> text properties
    "mark": {"foreground": "#77767b", "weight": Pango.Weight.BOLD},
    "info": {"foreground": "#1a5fb4"},
    "status": {"weight": Pango.Weight.BOLD},
    "stderr": {"foreground": "#a51d2d"},
    "error": {"foreground": "#c01c28", "weight": Pango.Weight.BOLD},
    "waiting": {"foreground": "#b56a00", "weight": Pango.Weight.BOLD},
}


class Line(NamedTuple):
    step: Optional[int]  # the plan step the line belongs to (None: a planner message)
    kind: str  # info | out | stderr | status
    mark: str  # "[03] " before a script line
    text: str

    def tag(self) -> Optional[str]:
        """How the line is highlighted."""
        if self.text.startswith("ERROR") or (self.kind == "status" and " - failed" in self.text):
            return "error"
        if self.text.startswith("WAITING") or (self.kind == "status" and " - waiting" in self.text):
            return "waiting"
        if self.kind == "stderr":
            return "stderr"
        if self.kind in ("status", "info"):
            return self.kind
        return None


def script_line(line: str, stderr: bool = False) -> Line:
    """A line of the runner's output: ``[NN] text`` from a script, ``==== step NN: …`` status lines."""
    mark, step, text = "", None, line
    m = _MARK.match(line)
    if m:
        mark, step, text = m.group(0), int(m.group(1)), line[m.end():]
    s = _STATUS.match(text)
    if s:
        return Line(int(s.group(1)), "status", mark, text)
    return Line(step, "stderr" if stderr else "out", mark, text)


class LogPanel(Gtk.Box):
    """The output log: a header with its controls above a read-only text view (collapsible)."""

    def __init__(self, stored_log: Optional[Callable[[int, str], str]] = None,
                 on_expanded: Optional[Callable[[bool], None]] = None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.stored_log = stored_log  # (step, title) -> text of the step's last run in the step log
        self.on_expanded = on_expanded
        self.records: List[Line] = []
        self._pending: List[Line] = []
        self._skipped: Dict[Optional[int], int] = {}  # step -> queued lines dropped since the last flush
        self._lock = threading.RLock()  # RLock: a GTK handler may log while a flush holds it
        self._flush_id = 0
        self._dead = False
        self._filter: Optional[int] = None  # step shown (None: all lines)
        self._steps: Dict[int, str] = {}  # step number -> title, for the filter
        self._selected: Optional[Tuple[int, str]] = None  # the step selected in the planner
        self._updating = False
        self.pack_start(self._header(), False, False, 0)
        self.view = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.view.set_cursor_visible(False)
        self.buffer = self.view.get_buffer()
        for name, props in TAGS.items():
            self.buffer.create_tag(name, **props)
        self.end_mark = self.buffer.create_mark("end", self.buffer.get_end_iter(), False)
        self.body = Gtk.ScrolledWindow()
        self.body.set_shadow_type(Gtk.ShadowType.IN)
        self.body.set_min_content_height(60)
        self.body.add(self.view)
        self.pack_start(self.body, True, True, 0)
        self.connect("destroy", self._on_destroy)

    # ---------------------------------------------------------------- chrome
    def _header(self) -> Gtk.Widget:
        bar = Gtk.Box(spacing=4, margin=3)
        self.expand_btn = Gtk.ToggleButton()
        self.expand_btn.set_relief(Gtk.ReliefStyle.NONE)
        self.expand_btn.set_active(True)
        self.expand_btn.set_image(Gtk.Image.new_from_icon_name("pan-down-symbolic", Gtk.IconSize.BUTTON))
        self.expand_btn.set_tooltip_text("Show or hide the output log (Ctrl+L)")
        self.expand_btn.connect("toggled", self._on_expand_toggled)
        bar.pack_start(self.expand_btn, False, False, 0)
        self.title = W.label("<b>Output</b>", markup=True)
        bar.pack_start(self.title, False, False, 0)
        self.subtitle = W.hint("")
        self.subtitle.set_ellipsize(Pango.EllipsizeMode.END)
        bar.pack_start(self.subtitle, True, True, 6)
        bar.pack_start(W.label("Step"), False, False, 0)
        self.step_combo = W.combo([("", "All steps")], "Show the lines of one step only")
        self.step_combo.set_active_id("")
        W.ellipsize_combo(self.step_combo, 18)
        self.step_combo.connect("changed", self._on_filter)
        bar.pack_start(self.step_combo, False, False, 0)
        self.stored_btn = Gtk.ToggleButton(label="Last run")
        self.stored_btn.set_tooltip_text("Show the stored log of the selected step's last run (from the "
                                         "project's step log) instead of the live output")
        self.stored_btn.set_sensitive(False)
        self.stored_btn.connect("toggled", lambda *_: self._show())
        bar.pack_start(self.stored_btn, False, False, 0)
        self.autoscroll = Gtk.ToggleButton()
        self.autoscroll.set_image(Gtk.Image.new_from_icon_name("go-bottom-symbolic", Gtk.IconSize.BUTTON))
        self.autoscroll.set_active(True)
        self.autoscroll.set_tooltip_text("Follow new lines (auto-scroll)")
        self.autoscroll.connect("toggled", lambda b: b.get_active() and self._scroll_end())
        bar.pack_start(self.autoscroll, False, False, 0)
        for icon, tip, cb in (("edit-copy-symbolic", "Copy the lines shown", self.copy),
                              ("document-save-as-symbolic", "Save the lines shown as a text file",
                               self.save_as),
                              ("edit-clear-all-symbolic", "Clear the output", self.clear)):
            b = Gtk.Button.new_from_icon_name(icon, Gtk.IconSize.BUTTON)
            b.set_tooltip_text(tip)
            b.connect("clicked", lambda *_, c=cb: c())
            bar.pack_start(b, False, False, 0)
        return bar

    @property
    def expanded(self) -> bool:
        return self.expand_btn.get_active()

    def set_expanded(self, expanded: bool) -> None:
        self.expand_btn.set_active(expanded)

    def _on_expand_toggled(self, btn) -> None:
        on = btn.get_active()
        btn.get_image().set_from_icon_name("pan-down-symbolic" if on else "pan-end-symbolic",
                                           Gtk.IconSize.BUTTON)
        self.body.set_visible(on)
        if self.on_expanded is not None:
            self.on_expanded(on)

    # ---------------------------------------------------------------- input (any thread)
    def feed(self, text: str, kind: str = "info", step: Optional[int] = None) -> None:
        """A message (several lines allowed); safe from any thread."""
        self._queue([Line(step, kind, "", t) for t in str(text).split("\n")])

    def feed_script(self, line: str, stderr: bool = False) -> None:
        """A line of the script runner (see :func:`script_line`); safe from any thread."""
        self._queue([script_line(t, stderr) for t in line.split("\n")])

    def _queue(self, lines: Sequence[Line]) -> None:
        with self._lock:
            if self._dead:
                return
            self._pending.extend(lines)
            flood = len(self._pending) > 2 * MAX_BATCH
            if flood:  # dropped in chunks: constant work per line
                self._skip(len(self._pending) - MAX_BATCH)
            if not self._flush_id:
                self._flush_id = GLib.timeout_add(FLUSH_MS, self._flush)
        if flood and threading.current_thread() is not threading.main_thread():
            time.sleep(FLOOD_PAUSE_S)  # the GIL to the main loop for a moment (see FLOOD_PAUSE_S)

    def _skip(self, n: int) -> None:
        """Drop the ``n`` oldest queued lines, counted per step for the note (the lock is held)."""
        for r in self._pending[:n]:
            self._skipped[r.step] = self._skipped.get(r.step, 0) + 1
        del self._pending[:n]

    def flush(self) -> None:
        """Insert the queued lines now (main loop)."""
        with self._lock:
            if self._flush_id:
                GLib.source_remove(self._flush_id)
        self._flush()

    def _flush(self) -> bool:
        # The lock is held throughout: a script's reader thread waits here instead of competing for the
        # GIL, which every GTK call releases; with a chatty script each call would wait for it.
        with self._lock:
            if len(self._pending) > MAX_BATCH:
                self._skip(len(self._pending) - MAX_BATCH)
            pending, self._pending = self._pending, []
            skipped, self._skipped = self._skipped, {}
            self._flush_id = 0
            if (pending or skipped) and not self._dead:
                self._show_batch(pending, skipped)
        return False

    def _show_batch(self, pending: List[Line], skipped: Dict[Optional[int], int]) -> None:
        # no "see the step log": it keeps only the end of a script's output (projectfile.LOG_OUTPUT_CHARS)
        notes = [Line(step, "info", "" if step is None else f"[{step:02d}] ",
                      f"… {n} line(s) not shown: they came faster than the log can show them")
                 for step, n in sorted(skipped.items(), key=lambda kv: -1 if kv[0] is None else kv[0])]
        batch = notes + pending
        live = not self.stored_btn.get_active()
        total = len(self.records) + len(batch)
        if total > MAX_LINES:  # the oldest lines go first: the buffer never holds more than MAX_LINES
            cut = total - KEEP_LINES
            old = min(cut, len(self.records))
            n = sum(1 for r in self.records[:old] if self._visible(r))
            if n and live:
                self.buffer.delete(self.buffer.get_start_iter(), self.buffer.get_iter_at_line(n))
            self.records = self.records[old:]
            batch = batch[cut - old:]
        self.records.extend(batch)
        if live:
            self._insert([r for r in batch if self._visible(r)])
        new_steps = {r.step for r in batch if r.step is not None and r.step not in self._steps}
        if new_steps:  # offered by the filter from now on
            for s in new_steps:
                self._steps[s] = ""
            self._fill_steps()
        self._update_subtitle()

    def _on_destroy(self, *_):
        with self._lock:
            self._dead = True
            if self._flush_id:
                GLib.source_remove(self._flush_id)
                self._flush_id = 0

    # ---------------------------------------------------------------- display
    def _visible(self, line: Line) -> bool:
        return self._filter is None or line.step == self._filter

    def _insert(self, lines: Sequence[Line]) -> None:
        buf = self.buffer
        for r in lines:
            if r.mark:
                buf.insert_with_tags_by_name(buf.get_end_iter(), r.mark, "mark")
            tag = r.tag()
            if tag:
                buf.insert_with_tags_by_name(buf.get_end_iter(), r.text + "\n", tag)
            else:
                buf.insert(buf.get_end_iter(), r.text + "\n")
        if lines and self.autoscroll.get_active():
            self._scroll_end()

    def _scroll_end(self) -> None:
        self.buffer.move_mark(self.end_mark, self.buffer.get_end_iter())
        self.view.scroll_mark_onscreen(self.end_mark)

    def _show(self) -> None:
        """Fill the view anew: the stored log of the selected step, or the live lines of the filter."""
        self.buffer.set_text("")
        stored = self.stored_btn.get_active() and self._selected is not None
        self.step_combo.set_sensitive(not stored)
        if stored:
            step, title = self._selected
            text = (self.stored_log(step, title) if self.stored_log is not None else "") or \
                f"Step {step:02d} has no stored run in the project's step log yet."
            self._insert([Line(step, "stored", "", t) for t in text.rstrip("\n").split("\n")])
        else:
            self._insert([r for r in self.records if self._visible(r)])
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        if self.stored_btn.get_active() and self._selected is not None:
            step, title = self._selected
            self.subtitle.set_text(f"last run of step {step:02d}" + (f" · {title}" if title else ""))
        elif self._filter is not None:
            n = sum(1 for r in self.records if r.step == self._filter)
            self.subtitle.set_text(f"step {self._filter:02d}: {n} line(s) of {len(self.records)}")
        else:
            self.subtitle.set_text(f"{len(self.records)} line(s)" if self.records else "")

    # ---------------------------------------------------------------- steps
    def set_steps(self, steps: Sequence[Tuple[int, str]]) -> None:
        """The plan's steps (number, title) offered by the filter; steps seen in the output stay."""
        seen = {r.step for r in self.records if r.step is not None}
        new = {n: t for n, t in steps}
        new.update({s: self._steps.get(s, "") for s in seen if s not in new})
        if new != self._steps:
            self._steps = new
            self._fill_steps()

    def _fill_steps(self) -> None:
        active = self.step_combo.get_active_id() or ""
        self._updating = True
        try:
            self.step_combo.remove_all()
            self.step_combo.append("", "All steps")
            for n in sorted(self._steps):
                title = self._steps[n]
                self.step_combo.append(f"{n:02d}", f"step {n:02d}" + (f" · {title}" if title else ""))
            if not self.step_combo.set_active_id(active):
                self.step_combo.set_active_id("")
        finally:
            self._updating = False
        if (self.step_combo.get_active_id() or "") != active:
            self._on_filter()

    def set_filter(self, step: Optional[int]) -> None:
        """Show only the lines of ``step`` (None: all)."""
        if step is not None and step not in self._steps:
            self._steps[step] = ""
            self._fill_steps()
        self.step_combo.set_active_id("" if step is None else f"{step:02d}")

    def _on_filter(self, *_):
        if self._updating:
            return
        ident = self.step_combo.get_active_id() or ""
        self._filter = int(ident) if ident else None
        self._show()

    def select_step(self, step: Optional[int], title: str = "") -> None:
        """The step selected in the planner (None: none, or not a step): 'Last run' shows its stored log."""
        self._selected = (step, title) if step is not None else None
        self.stored_btn.set_sensitive(self._selected is not None)
        if self._selected is None and self.stored_btn.get_active():
            self.stored_btn.set_active(False)  # shows the live lines again
        elif self.stored_btn.get_active():
            self._show()
        if self._selected is not None:
            self.stored_btn.set_tooltip_text(f"Show the stored log of step {step:02d}'s last run (from the "
                                             "project's step log) instead of the live output")

    def show_live(self) -> None:
        """Back to the live lines (a run starts)."""
        if self.stored_btn.get_active():
            self.stored_btn.set_active(False)

    # ---------------------------------------------------------------- actions
    def text(self) -> str:
        """The lines shown."""
        self.flush()
        buf = self.buffer
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

    def clear(self) -> None:
        with self._lock:
            self._pending, self._skipped = [], {}
        self.records = []
        if self.stored_btn.get_active():
            self.stored_btn.set_active(False)
        else:
            self._show()

    def copy(self) -> None:
        Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(self.text(), -1)

    def save_as(self, path: Optional[str] = None) -> Optional[str]:
        """Write the lines shown to a text file (asks for the name unless ``path`` is given)."""
        text = self.text()
        if path is None:
            top = self.get_toplevel()
            dlg = Gtk.FileChooserDialog(title="Save the output log", action=Gtk.FileChooserAction.SAVE,
                                        transient_for=top if isinstance(top, Gtk.Window) else None)
            dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
            dlg.set_do_overwrite_confirmation(True)
            W.add_filters(dlg, [("Text files", ["*.txt", "*.log"])])
            dlg.set_current_name("rnado_output.txt")
            path = dlg.get_filename() if dlg.run() == Gtk.ResponseType.OK else None
            dlg.destroy()
            if not path:
                return None
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            self.feed(f"ERROR: could not write {path}: {exc}")
            return None
        self.feed(f"Output log saved to {path}")
        return path
