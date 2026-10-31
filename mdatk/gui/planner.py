"""The planner of RNAdo: a free-form canvas of steps (boxes) and file connections.

Boxes are generator steps (FARFAR2, Amber build, MD stages, GROMACS, VMD ...),
notes, existing files and the MDATK analysis step.  Dragging from an output
port (right edge) to an input port (left edge) connects them; the planner then
hands the upstream file names to the downstream step, so every generated
script refers to the files the previous step will write
(``<project>_<step>_<tool>_<time>.<ext>``).  Double-click a box to edit it in
the wizard.  The plan is stored in the project file (``workflow`` section).

:class:`PlannerPanel` is the canvas with its toolbar and details pane; RNAdo (:mod:`mdatk.gui.rnado`)
embeds it and is its host (:class:`mdatk.gui.host.Host`): the project, the output log, saving and
running scripts.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk, Pango  # noqa: E402

from ..config import PlanEdge, PlanNode  # noqa: E402
from ..projectfile import logged  # noqa: E402
from ..workflows import TOOLS, runstate, tools_by_category  # noqa: E402
from ..workflows.naming import new_stamp  # noqa: E402
from ..workflows.plan import (TEMPLATES, Plan, add_node, analysis_inputs, apply_template,  # noqa: E402
                              connect, next_step, remove_node, step_title)
from ..workflows.runner import plan_jobs, separate_scripts  # noqa: E402
from ..workflows.schema import Port, kind_label  # noqa: E402
from . import desktop  # noqa: E402
from . import widgets as W  # noqa: E402
from .host import Host  # noqa: E402
from .wizard import LEVEL_COLORS, WizardWindow  # noqa: E402

NODE_W = 236
HEAD_H = 26
DESC_H = 20
CHECK = 13  # side of the "run" check box in a step's header
FOOT_H = 18  # row under the ports: the step's previous run (Check previous runs) or the run it uses
MAX_PARALLEL = 64
PORT_H = 18
PORT_R = 5.0
GRID = 10

CATEGORY_COLORS = {
    "Planning": (0.45, 0.45, 0.47),
    "Structure prediction": (0.50, 0.30, 0.62),
    "Amber": (0.85, 0.42, 0.10),
    "GROMACS": (0.15, 0.55, 0.36),
    "Visualisation": (0.13, 0.50, 0.62),
    "Analysis": (0.10, 0.37, 0.71),
    "Custom": (0.62, 0.20, 0.42),
}
KIND_COLORS = {
    "pdb": (0.10, 0.37, 0.71), "parm7": (0.85, 0.42, 0.10), "rst7": (0.90, 0.65, 0.04),
    "nc": (0.15, 0.55, 0.36), "xtc": (0.15, 0.55, 0.36), "dist": (0.57, 0.25, 0.67),
    "fasta": (0.08, 0.60, 0.60), "secstruct": (0.08, 0.60, 0.60), "silent": (0.53, 0.37, 0.24),
    "map": (0.40, 0.40, 0.40), "top": (0.75, 0.11, 0.16), "gro": (0.75, 0.11, 0.16),
    "image": (0.55, 0.55, 0.55), "trafl": (0.53, 0.37, 0.24), "text": (0.55, 0.55, 0.55),
}
STATUS_COLORS = {"error": (0.75, 0.11, 0.16), "warning": (0.90, 0.60, 0.05), "info": (0.20, 0.65, 0.35),
                 "ok": (0.20, 0.65, 0.35)}
RUN_COLORS = {"done": (0.20, 0.65, 0.35), "partial": (0.90, 0.60, 0.05), "missing": (0.60, 0.60, 0.60),
              "none": (0.60, 0.60, 0.60)}


def _rgb_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in rgb))


def checkable(node: PlanNode) -> bool:
    """Steps that write and run scripts (not notes or the MDATK analysis box)."""
    tool = TOOLS.get(node.tool)
    return bool(tool is not None and tool.generates)


class PlannerCanvas(Gtk.DrawingArea):
    """Draws the plan and handles moving, connecting and selecting."""

    def __init__(self, window: "PlannerPanel"):
        super().__init__()
        self.win = window
        self.zoom = 1.0
        self.selected: Optional[str] = None  # node id
        self.selected_edge: Optional[int] = None
        self.drag: Optional[dict] = None
        self.hover_port: Optional[Tuple[str, str, bool]] = None
        self.spin_phase = 0.0  # angle of the running-script spinner
        self.set_can_focus(True)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.SCROLL_MASK |
                        Gdk.EventMask.KEY_PRESS_MASK)
        self.connect("draw", self._on_draw)
        self.connect("button-press-event", self._on_press)
        self.connect("button-release-event", self._on_release)
        self.connect("motion-notify-event", self._on_motion)
        self.connect("scroll-event", self._on_scroll)
        self.connect("key-press-event", self._on_key)

    # ---------------------------------------------------------- geometry
    @property
    def workflow(self):
        return self.win.workflow

    def ports(self, node: PlanNode) -> Tuple[List[Port], List[Port]]:
        step = self.win.plan.steps.get(node.id) if self.win.plan else None
        tool = TOOLS.get(node.tool)
        if tool is None:
            return [], []
        settings = step.settings if step is not None and step.settings else tool.normalise(node.settings)
        return tool.input_ports(settings), tool.output_ports(settings)

    def node_size(self, node: PlanNode) -> Tuple[float, float]:
        if node.tool == "note":
            lines = max(1, len((node.settings.get("text") or "").splitlines()))
            return NODE_W, HEAD_H + 14 * min(lines, 12) + 12
        ins, outs = self.ports(node)
        foot = FOOT_H if self.run_info(node) else 0
        return NODE_W, HEAD_H + DESC_H + max(len(ins), len(outs), 1) * PORT_H + 8 + foot

    def run_info(self, node: PlanNode) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        """Text and colour of the step's footer: what 'Check previous runs' found, or the run it uses."""
        run = self.win.prev_runs.get(node.id)
        if run is not None:
            text = {"done": f"done · {run.stamp or 'no stamp'} · {run.date}",
                    "partial": f"partial · {run.stamp or 'no stamp'} · {len(run.found)} of "
                               f"{len(run.found) + len(run.missing)} files",
                    "missing": "missing · no outputs yet", "none": "no output files to look for"}[run.state]
            if node.adopted:
                text = ("uses " if run.adopted else f"uses {node.adopted} · found ") + text
            return text, RUN_COLORS[run.state]
        if node.adopted and checkable(node):
            return f"uses the run of {node.adopted}", (0.10, 0.37, 0.71)
        return None

    def port_pos(self, node: PlanNode, name: str, output: bool) -> Optional[Tuple[float, float]]:
        ins, outs = self.ports(node)
        ports = outs if output else ins
        for k, p in enumerate(ports):
            if p.name == name:
                x = node.x + (NODE_W if output else 0)
                y = node.y + HEAD_H + DESC_H + (k + 0.5) * PORT_H
                return x, y
        return None

    def content_size(self) -> Tuple[int, int]:
        w, h = 1200, 800
        for n in self.workflow.nodes:
            nw, nh = self.node_size(n)
            w = max(w, int(n.x + nw + 400))
            h = max(h, int(n.y + nh + 300))
        return int(w * self.zoom), int(h * self.zoom)

    def update_size(self) -> None:
        w, h = self.content_size()
        self.set_size_request(w, h)

    def to_model(self, ex: float, ey: float) -> Tuple[float, float]:
        return ex / self.zoom, ey / self.zoom

    # ------------------------------------------------------------ hit tests
    def node_at(self, x: float, y: float) -> Optional[PlanNode]:
        for n in reversed(self.workflow.nodes):
            w, h = self.node_size(n)
            if n.x <= x <= n.x + w and n.y <= y <= n.y + h:
                return n
        return None

    @staticmethod
    def check_rect(node: PlanNode) -> Tuple[float, float, float, float]:
        return node.x + 8, node.y + (HEAD_H - CHECK) / 2, CHECK, CHECK

    def check_at(self, x: float, y: float) -> Optional[PlanNode]:
        """The step whose "run" check box is under (x, y) (2 px tolerance)."""
        for n in reversed(self.workflow.nodes):
            if not checkable(n):
                continue
            bx, by, bw, bh = self.check_rect(n)
            if bx - 2 <= x <= bx + bw + 2 and by - 2 <= y <= by + bh + 2:
                return n
        return None

    def port_at(self, x: float, y: float) -> Optional[Tuple[PlanNode, Port, bool]]:
        for n in reversed(self.workflow.nodes):
            ins, outs = self.ports(n)
            for output, ports in ((True, outs), (False, ins)):
                for p in ports:
                    pos = self.port_pos(n, p.name, output)
                    if pos and math.hypot(pos[0] - x, pos[1] - y) <= PORT_R + 5:
                        return n, p, output
        return None

    def edge_points(self, e: PlanEdge):
        nodes = {n.id: n for n in self.workflow.nodes}
        if e.source not in nodes or e.target not in nodes:
            return None
        a = self.port_pos(nodes[e.source], e.output, True)
        b = self.port_pos(nodes[e.target], e.input, False)
        if a is None or b is None:
            return None
        dx = max(40.0, abs(b[0] - a[0]) / 2)
        return a, (a[0] + dx, a[1]), (b[0] - dx, b[1]), b

    def edge_at(self, x: float, y: float) -> Optional[int]:
        best, best_d = None, 7.0
        for k, e in enumerate(self.workflow.edges):
            pts = self.edge_points(e)
            if pts is None:
                continue
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
            for i in range(41):
                t = i / 40
                mt = 1 - t
                bx = mt ** 3 * x0 + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t ** 3 * x3
                by = mt ** 3 * y0 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t ** 3 * y3
                d = math.hypot(bx - x, by - y)
                if d < best_d:
                    best, best_d = k, d
        return best

    # ---------------------------------------------------------------- draw
    def _on_draw(self, _w, cr) -> bool:
        alloc = self.get_allocation()
        cr.set_source_rgb(0.965, 0.961, 0.957)
        cr.paint()
        cr.scale(self.zoom, self.zoom)
        # dot grid
        cr.set_source_rgb(0.82, 0.81, 0.80)
        step = 20
        for gx in range(0, int(alloc.width / self.zoom) + step, step):
            for gy in range(0, int(alloc.height / self.zoom) + step, step):
                cr.rectangle(gx, gy, 1, 1)
        cr.fill()
        if not self.workflow.nodes:
            cr.set_source_rgb(0.45, 0.45, 0.45)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(14)
            cr.move_to(40, 60)
            cr.show_text("Empty plan: add steps with 'Add step' or start from a template.")
            cr.move_to(40, 84)
            cr.show_text("Drag from an output (right edge) to an input (left edge) to connect steps.")
            return False
        for k, e in enumerate(self.workflow.edges):
            self._draw_edge(cr, e, k == self.selected_edge)
        for n in self.workflow.nodes:
            self._draw_node(cr, n)
        if self.drag and self.drag.get("mode") == "connect":
            x0, y0 = self.drag["start"]
            x1, y1 = self.drag["pos"]
            cr.set_source_rgb(0.2, 0.2, 0.2)
            cr.set_dash([5, 4])
            cr.set_line_width(1.5)
            cr.move_to(x0, y0)
            dx = max(40.0, abs(x1 - x0) / 2) * (1 if self.drag["output"] else -1)
            cr.curve_to(x0 + dx, y0, x1 - dx, y1, x1, y1)
            cr.stroke()
            cr.set_dash([])
        return False

    def _draw_edge(self, cr, e: PlanEdge, selected: bool) -> None:
        pts = self.edge_points(e)
        if pts is None:
            return
        nodes = {n.id: n for n in self.workflow.nodes}
        _ins, outs = self.ports(nodes[e.source])
        kind = next((p.kind for p in outs if p.name == e.output), "text")
        rgb = KIND_COLORS.get(kind, (0.4, 0.4, 0.4))
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
        cr.set_line_width(4.0 if selected else 2.2)
        if selected:
            cr.set_source_rgb(0.10, 0.37, 0.71)
        else:
            cr.set_source_rgba(*rgb, 0.85)
        cr.move_to(x0, y0)
        cr.curve_to(x1, y1, x2, y2, x3, y3)
        cr.stroke()

    def _rounded(self, cr, x, y, w, h, r=7.0) -> None:
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()

    @staticmethod
    def _text(cr, text: str, x: float, y: float, size: float = 11.0, bold: bool = False,
              rgb=(0.1, 0.1, 0.1), max_w: Optional[float] = None, align_right: bool = False) -> None:
        cr.select_font_face("Sans", 0, 1 if bold else 0)
        cr.set_font_size(size)
        if max_w is not None:
            while text and cr.text_extents(text).x_advance > max_w:
                text = text[:-2] + "…" if len(text) > 2 else ""
        ext = cr.text_extents(text)
        cr.set_source_rgb(*rgb)
        cr.move_to(x - ext.x_advance if align_right else x, y)
        cr.show_text(text)
        cr.new_path()  # show_text leaves a current point; later arcs must not connect to it

    def _draw_node(self, cr, n: PlanNode) -> None:
        tool = TOOLS.get(n.tool)
        w, h = self.node_size(n)
        sel = n.id == self.selected
        if n.tool == "note":
            self._rounded(cr, n.x, n.y, w, h)
            cr.set_source_rgb(1.0, 0.965, 0.75)
            cr.fill_preserve()
            cr.set_source_rgb(0.10, 0.37, 0.71) if sel else cr.set_source_rgb(0.80, 0.72, 0.35)
            cr.set_line_width(2.5 if sel else 1.0)
            cr.stroke()
            self._text(cr, n.title or "Note", n.x + 10, n.y + 18, 11, True, (0.35, 0.30, 0.10), w - 20)
            lines = (n.settings.get("text") or "").splitlines()[:12]
            for k, line in enumerate(lines):
                self._text(cr, line, n.x + 10, n.y + HEAD_H + 12 + 14 * k, 10, False, (0.25, 0.22, 0.10),
                           w - 20)
            return
        cat = tool.category if tool else "Planning"
        rgb = CATEGORY_COLORS.get(cat, (0.4, 0.4, 0.4))
        # shadow
        self._rounded(cr, n.x + 2, n.y + 3, w, h)
        cr.set_source_rgba(0, 0, 0, 0.12)
        cr.fill()
        self._rounded(cr, n.x, n.y, w, h)
        cr.set_source_rgb(1, 1, 1)
        cr.fill_preserve()
        if sel:
            cr.set_source_rgb(0.10, 0.37, 0.71)
            cr.set_line_width(2.5)
        else:
            cr.set_source_rgb(0.60, 0.60, 0.59)
            cr.set_line_width(1.0)
        cr.stroke()
        # header
        cr.save()
        self._rounded(cr, n.x, n.y, w, h)
        cr.clip()
        cr.rectangle(n.x, n.y, w, HEAD_H)
        cr.set_source_rgb(*rgb)
        cr.fill()
        cr.restore()
        title_x = n.x + 9
        if checkable(n):
            bx, by, bw, bh = self.check_rect(n)
            self._rounded(cr, bx, by, bw, bh, 2.5)
            cr.set_source_rgb(1, 1, 1)
            cr.fill_preserve()
            cr.set_source_rgba(0, 0, 0, 0.35)
            cr.set_line_width(1.0)
            cr.stroke()
            if n.id in self.win.checked:
                cr.move_to(bx + 3, by + bh * 0.55)
                cr.line_to(bx + bw * 0.42, by + bh - 3)
                cr.line_to(bx + bw - 2.5, by + 3)
                cr.set_source_rgb(*rgb)
                cr.set_line_width(2.2)
                cr.stroke()
            title_x = bx + bw + 7
        run = self.win.run_state.get(n.id)
        if run:
            self._draw_run_state(cr, run, n.x + w - 31, n.y + HEAD_H / 2, rgb)
        self._text(cr, step_title(n), title_x, n.y + 17, 11, True, (1, 1, 1),
                   n.x + w - (43 if run else 25) - title_x)
        step = self.win.plan.steps.get(n.id) if self.win.plan else None
        level = step.level() if step else "ok"
        cr.new_path()
        cr.arc(n.x + w - 13, n.y + HEAD_H / 2, 5.5, 0, 2 * math.pi)
        cr.set_source_rgb(*STATUS_COLORS[level])
        cr.fill_preserve()
        cr.set_source_rgb(1, 1, 1)
        cr.set_line_width(1.2)
        cr.stroke()
        desc = ""
        if tool is not None:
            try:
                desc = tool.describe(step.settings if step and step.settings else tool.normalise(n.settings))
            except Exception:  # describe must never break drawing
                desc = tool.label
        self._text(cr, desc, n.x + 9, n.y + HEAD_H + 14, 9.5, False, (0.35, 0.35, 0.35), w - 18)
        ins, outs = self.ports(n)
        connected_in = {e.input for e in self.workflow.edges if e.target == n.id}
        for output, ports in ((False, ins), (True, outs)):
            for k, p in enumerate(ports):
                px = n.x + (w if output else 0)
                py = n.y + HEAD_H + DESC_H + (k + 0.5) * PORT_H
                prgb = KIND_COLORS.get(p.kind, (0.4, 0.4, 0.4))
                cr.new_path()
                cr.arc(px, py, PORT_R, 0, 2 * math.pi)
                cr.set_source_rgb(*prgb)
                cr.fill_preserve()
                missing = (not output and not p.optional and p.name not in connected_in
                           and not n.files.get(p.name))
                cr.set_source_rgb(0.75, 0.11, 0.16) if missing else cr.set_source_rgb(1, 1, 1)
                cr.set_line_width(2.0 if missing else 1.2)
                cr.stroke()
                hover = self.hover_port and self.hover_port[:2] == (n.id, p.name)
                label = p.label + (" *" if not output and not p.optional else "")
                # inputs and outputs share rows: each label gets its own half of the box
                if output:
                    self._text(cr, label, px - 10, py + 4, 9.5, bool(hover), (0.15, 0.15, 0.15),
                               w / 2 - 14, align_right=True)
                else:
                    self._text(cr, label, px + 10, py + 4, 9.5, bool(hover), (0.15, 0.15, 0.15),
                               w / 2 - 14 if k < len(outs) else w - 24)
        info = self.run_info(n)
        if info is not None:  # footer: the previous run found, or the run the step uses
            text, frgb = info
            fy = n.y + h - FOOT_H / 2 - 3
            cr.new_path()
            cr.arc(n.x + 14, fy, 4.0, 0, 2 * math.pi)
            cr.set_source_rgb(*frgb)
            cr.fill()
            self._text(cr, text, n.x + 23, fy + 3.5, 9.0, bool(n.adopted), (0.25, 0.25, 0.25), w - 32)

    def _draw_run_state(self, cr, state: str, cx: float, cy: float, head_rgb) -> None:
        """Run marker left of the status dot: a turning arc while the script runs, a ring while it
        is queued, then a tick (ok), a cross (failed), two bars (waiting for a separately run script)
        or a dash (cancelled / not started)."""
        r = 6.0
        cr.new_path()
        if state == "running":
            cr.set_line_width(2.4)
            cr.set_source_rgba(1, 1, 1, 0.35)
            cr.arc(cx, cy, r, 0, 2 * math.pi)
            cr.stroke()
            cr.set_source_rgb(1, 1, 1)
            cr.arc(cx, cy, r, self.spin_phase, self.spin_phase + 1.3 * math.pi)
            cr.stroke()
            return
        if state == "queued":
            cr.set_line_width(1.6)
            cr.set_dash([2.0, 2.0])
            cr.set_source_rgb(1, 1, 1)
            cr.arc(cx, cy, r, 0, 2 * math.pi)
            cr.stroke()
            cr.set_dash([])
            return
        cr.arc(cx, cy, r + 0.5, 0, 2 * math.pi)
        cr.set_source_rgb(1, 1, 1)
        cr.fill()
        cr.set_line_width(2.0)
        cr.set_line_cap(1)  # round
        if state == "ok":
            cr.set_source_rgb(*STATUS_COLORS["ok"])
            cr.move_to(cx - 3.2, cy + 0.2)
            cr.line_to(cx - 0.8, cy + 2.8)
            cr.line_to(cx + 3.4, cy - 2.6)
        elif state == "failed":
            cr.set_source_rgb(*STATUS_COLORS["error"])
            cr.move_to(cx - 2.8, cy - 2.8)
            cr.line_to(cx + 2.8, cy + 2.8)
            cr.move_to(cx + 2.8, cy - 2.8)
            cr.line_to(cx - 2.8, cy + 2.8)
        elif state == "waiting":  # "pause": its separately run script has not finished
            cr.set_source_rgb(*STATUS_COLORS["warning"])
            cr.move_to(cx - 1.8, cy - 3.0)
            cr.line_to(cx - 1.8, cy + 3.0)
            cr.move_to(cx + 1.8, cy - 3.0)
            cr.line_to(cx + 1.8, cy + 3.0)
        else:  # cancelled / skipped
            cr.set_source_rgb(0.45, 0.45, 0.45)
            cr.move_to(cx - 3.0, cy)
            cr.line_to(cx + 3.0, cy)
        cr.stroke()
        cr.set_line_cap(0)

    # -------------------------------------------------------------- events
    def _focus_in_place(self) -> None:
        """Take the keyboard focus (Delete, arrow keys) without scrolling: the viewport around the canvas
        would otherwise scroll to show the focused child's top-left corner."""
        if self.has_focus():
            return
        h, v = self.win.scroller.get_hadjustment(), self.win.scroller.get_vadjustment()
        x, y = h.get_value(), v.get_value()
        self.grab_focus()
        h.set_value(x)
        v.set_value(y)

    def _on_press(self, _w, ev) -> bool:
        self._focus_in_place()
        x, y = self.to_model(ev.x, ev.y)
        if ev.button == 3:
            self._context_menu(ev, x, y)
            return True
        if ev.button == 2:
            self.drag = {"mode": "pan", "ex": ev.x_root, "ey": ev.y_root,
                         "h": self.win.scroller.get_hadjustment().get_value(),
                         "v": self.win.scroller.get_vadjustment().get_value()}
            return True
        if ev.button != 1:
            return False
        box = self.check_at(x, y)
        if box is not None:
            if ev.type == Gdk.EventType.BUTTON_PRESS:  # a double click must not toggle twice
                self.win.toggle_checked(box.id)
            return True
        if ev.type == Gdk.EventType._2BUTTON_PRESS:
            n = self.node_at(x, y)
            if n is not None:
                self.win.open_wizard(n)
            return True
        hit = self.port_at(x, y)
        if hit is not None:
            n, p, output = hit
            start = self.port_pos(n, p.name, output)
            self.drag = {"mode": "connect", "node": n.id, "port": p.name, "output": output,
                         "start": start, "pos": (x, y), "kind": p.kind}
            return True
        n = self.node_at(x, y)
        if n is not None:
            self.select(n.id)
            # bring to front
            self.workflow.nodes.remove(n)
            self.workflow.nodes.append(n)
            self.drag = {"mode": "move", "node": n.id, "dx": x - n.x, "dy": y - n.y, "moved": False}
            self.queue_draw()
            return True
        k = self.edge_at(x, y)
        self.selected = None
        self.selected_edge = k
        self.win.show_details(None, edge=k)
        self.queue_draw()
        return True

    def _on_motion(self, _w, ev) -> bool:
        x, y = self.to_model(ev.x, ev.y)
        d = self.drag
        if d is None:
            hit = self.port_at(x, y)
            new = (hit[0].id, hit[1].name, hit[2]) if hit else None
            if new != self.hover_port:
                self.hover_port = new
                if hit:
                    n, p, output = hit
                    kinds = ", ".join(kind_label(k) for k in p.kinds)
                    self.set_tooltip_text(f"{'Output' if output else 'Input'}: {p.label}\n{kinds}"
                                          + (f"\n{p.tooltip}" if p.tooltip else ""))
                else:
                    self.set_tooltip_text(None)
                self.queue_draw()
            return False
        if d["mode"] == "pan":
            self.win.scroller.get_hadjustment().set_value(d["h"] - (ev.x_root - d["ex"]))
            self.win.scroller.get_vadjustment().set_value(d["v"] - (ev.y_root - d["ey"]))
        elif d["mode"] == "move":
            n = self.win.node(d["node"])
            if n is not None:
                n.x = max(0.0, round((x - d["dx"]) / GRID) * GRID)
                n.y = max(0.0, round((y - d["dy"]) / GRID) * GRID)
                d["moved"] = True
                self.queue_draw()
        elif d["mode"] == "connect":
            d["pos"] = (x, y)
            self.queue_draw()
        return True

    def _on_release(self, _w, ev) -> bool:
        d, self.drag = self.drag, None
        if d is None:
            return False
        x, y = self.to_model(ev.x, ev.y)
        if d["mode"] == "move" and d["moved"]:
            self.update_size()
            self.win.changed(resolve=False)
        elif d["mode"] == "connect":
            hit = self.port_at(x, y)
            if hit is not None and hit[2] != d["output"]:
                n, p, _out = hit
                if d["output"]:
                    args = (d["node"], d["port"], n.id, p.name)
                else:
                    args = (n.id, p.name, d["node"], d["port"])
                try:
                    connect(self.workflow, *args)
                    self.win.set_status(f"Connected {args[1]} → {args[3]}")
                    self.win.changed()
                except (KeyError, ValueError) as exc:
                    self.win.set_status(f"Cannot connect: {exc}", error=True)
            self.queue_draw()
        return True

    def _on_scroll(self, _w, ev) -> bool:
        if not ev.state & Gdk.ModifierType.CONTROL_MASK:
            return False
        direction = ev.direction
        if direction == Gdk.ScrollDirection.SMOOTH:
            ok, _dx, dy = ev.get_scroll_deltas()
            factor = 0.9 if dy > 0 else 1.1
        else:
            factor = 1.1 if direction == Gdk.ScrollDirection.UP else 0.9
        self.set_zoom(self.zoom * factor)
        return True

    def set_zoom(self, zoom: float) -> None:
        self.zoom = min(2.0, max(0.4, zoom))
        self.update_size()
        self.queue_draw()

    def _on_key(self, _w, ev) -> bool:
        if ev.keyval in (Gdk.KEY_Delete, Gdk.KEY_BackSpace):
            self.win.delete_selected()
            return True
        if ev.keyval == Gdk.KEY_space and self.selected:
            n = self.win.node(self.selected)
            if n is not None and checkable(n):
                self.win.toggle_checked(n.id)
                return True
        return False

    def select(self, node_id: Optional[str]) -> None:
        self.selected = node_id
        self.selected_edge = None
        self.win.show_details(node_id)
        self.queue_draw()

    def _context_menu(self, ev, x, y) -> None:
        menu = Gtk.Menu()
        n = self.node_at(x, y)
        if n is not None:
            self.select(n.id)
            if TOOLS.get(n.tool) and TOOLS[n.tool].generates:
                self._menu_item(menu, "Edit in wizard…", lambda: self.win.open_wizard(n))
                self._menu_item(menu, "Untick for 'Run selected'" if n.id in self.win.checked
                                else "Tick for 'Run selected'", lambda: self.win.toggle_checked(n.id))
                self._menu_item(menu, "Write this step's files", lambda: self.win.write([n.id]))
                ids = self.win.selected_in_order() if n.id in self.win.checked else [n.id]
                self._menu_item(menu, f"Check previous runs ({len(ids)} ticked steps)" if len(ids) > 1
                                else "Check previous runs", lambda: self.win.check_runs(ids))
                run = self.win.prev_runs.get(n.id)
                if run is not None and run.state == "done" and run.stamp and not run.adopted:
                    self._menu_item(menu, f"Use the outputs of its run of {run.stamp}",
                                    lambda: self.win.adopt_runs([n.id]))
                if n.adopted:
                    self._menu_item(menu, f"Stop using the run of {n.adopted} (run the step again)",
                                    lambda: self.win.forget_runs([n.id]))
            elif n.tool == "mdatk":
                self._menu_item(menu, "Use for MDATK analysis", lambda: self.win.use_for_analysis(n))
            self._menu_item(menu, "Duplicate", lambda: self.win.duplicate(n))
            self._menu_item(menu, "Delete", self.win.delete_selected)
        else:
            k = self.edge_at(x, y)
            if k is not None:
                self.selected_edge = k
                self._menu_item(menu, "Delete connection", self.win.delete_selected)
            else:
                self._menu_item(menu, "Tick all steps", lambda: self.win.set_checked(
                    [m.id for m in self.workflow.nodes if checkable(m)]))
                if self.win.checked:
                    self._menu_item(menu, "Untick all steps", lambda: self.win.set_checked([]))
                    self._menu_item(menu, "Check previous runs of the ticked steps",
                                    lambda: self.win.check_runs(self.win.selected_in_order()))
                menu.append(Gtk.SeparatorMenuItem())
                for category, tools in tools_by_category():
                    sub = Gtk.Menu()
                    for t in tools:
                        self._menu_item(sub, t.label, lambda t=t: self.win.add_step(t.name, x, y))
                    item = Gtk.MenuItem(label=category)
                    item.set_submenu(sub)
                    menu.append(item)
        menu.show_all()
        menu.popup_at_pointer(ev)
        self.queue_draw()

    @staticmethod
    def _menu_item(menu, text, cb) -> None:
        item = Gtk.MenuItem(label=text)
        item.connect("activate", lambda *_: cb())
        menu.append(item)


class PlannerPanel(Gtk.Box):
    """The planner: toolbar, naming bar, canvas, details pane and status line (one per RNAdo window)."""

    def __init__(self, host: Host):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.host = host
        self.on_select: Optional[Callable[[Optional[PlanNode]], None]] = None  # a box was (de)selected
        self.on_plan_changed: Optional[Callable[[], None]] = None  # the plan was resolved anew
        self.plan: Optional[Plan] = None
        self.wizards: Dict[str, WizardWindow] = {}
        self._resolve_id = 0
        self._details_node: Optional[str] = None
        self._loading = False
        self.checked: Set[str] = set()  # steps ticked for "Run selected" (not saved with the project)
        # node id -> queued | running | ok | failed | cancelled | skipped | waiting
        self.run_state: Dict[str, str] = {}
        self.prev_runs: Dict[str, runstate.RunStatus] = {}  # what 'Check previous runs' found (not saved)
        self._spin_id = 0
        self._closed = False
        self._run_token = 0  # bumped by each run and by reload: callbacks of an older run are ignored
        outer = self
        outer.pack_start(self._toolbar(), False, False, 0)
        outer.pack_start(self._naming_bar(), False, False, 0)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.canvas = PlannerCanvas(self)
        self.scroller = Gtk.ScrolledWindow()
        self.scroller.add(self.canvas)
        paned.pack1(self.scroller, True, False)
        paned.pack2(self._details(), False, False)
        paned.set_position(1020)
        outer.pack_start(paned, True, True, 0)
        self.status = W.label("")
        self.status.set_margin_start(10)
        self.status.set_margin_bottom(6)
        self.status.set_margin_top(4)
        outer.pack_start(self.status, False, False, 0)
        self.connect("destroy", lambda *_: self.shutdown())
        self.reload()
        self.show_all()
        self.show_details(None)

    def toplevel(self) -> Optional[Gtk.Window]:
        """The window around the planner (parent of its dialogs)."""
        top = self.get_toplevel()
        return top if isinstance(top, Gtk.Window) else None

    # ------------------------------------------------------------ model
    @property
    def workflow(self):
        return self.host.project.workflow

    def node(self, node_id: str) -> Optional[PlanNode]:
        return next((n for n in self.workflow.nodes if n.id == node_id), None)

    def resolve(self) -> Plan:
        self.plan = Plan(self.workflow, self.host.project.workflow_dir, self.host.project.groups)
        return self.plan

    def reload(self) -> None:
        """Re-read everything from the host's project (after a project was opened or reloaded)."""
        for win in list(self.wizards.values()):
            win.destroy()
        self.wizards.clear()
        self._loading = True
        try:
            wf = self.workflow
            self.project_entry.set_text(wf.project_name)
            self.stamp_check.set_active(wf.use_stamp)
            self.folders_check.set_active(wf.step_folders)
            self.parallel_spin.set_value(max(1, wf.parallel))
            wf.parallel = int(self.parallel_spin.get_value())  # what the run will use (1-64)
        finally:
            self._loading = False
        self.canvas.selected = None
        self.checked.clear()
        self.run_state.clear()
        self.prev_runs.clear()
        self._run_token += 1
        self.changed()

    def changed(self, resolve: bool = True) -> None:
        self.checked &= {n.id for n in self.workflow.nodes if checkable(n)}  # deleted steps
        self._update_run_selected()
        if resolve:
            self.resolve()
            self._update_status()
            self._refresh_naming()
            for win in list(self.wizards.values()):
                win.schedule_refresh()
            if self.on_plan_changed is not None:
                self.on_plan_changed()
        self.canvas.update_size()
        self.canvas.queue_draw()
        if self._details_node is not None:
            self.show_details(self._details_node)

    def _update_status(self) -> None:
        issues = self.plan.all_issues() if self.plan else []
        n_err = sum(1 for _n, i in issues if i.level == "error")
        n_warn = sum(1 for _n, i in issues if i.level == "warning")
        steps = sum(1 for n in self.workflow.nodes if TOOLS.get(n.tool) and TOOLS[n.tool].generates)
        self.set_status(f"{steps} steps · {len(self.workflow.edges)} connections · {n_err} errors · "
                        f"{n_warn} warnings · folder {self.host.project.workflow_dir}")

    def set_status(self, text: str, error: bool = False) -> None:
        if error:
            self.status.set_markup(f'<span foreground="{LEVEL_COLORS["error"]}">'
                                   f'{GLib.markup_escape_text(text)}</span>')
        else:
            self.status.set_text(text)

    # ------------------------------------------------------------ chrome
    def _toolbar(self) -> Gtk.Widget:
        bar = Gtk.Box(spacing=6, margin=6)
        add = Gtk.MenuButton(label="Add step")
        add.set_tooltip_text("Add a box for one program (placed at the top left; drag it anywhere)")
        menu = Gtk.Menu()
        for category, tools in tools_by_category():
            sub = Gtk.Menu()
            for t in tools:
                item = Gtk.MenuItem(label=t.label)
                item.set_tooltip_text(t.description)
                item.connect("activate", lambda _w, t=t: self.add_step(t.name))
                sub.append(item)
            item = Gtk.MenuItem(label=category)
            item.set_submenu(sub)
            menu.append(item)
        menu.show_all()
        add.set_popup(menu)
        bar.pack_start(add, False, False, 0)
        tmpl = Gtk.MenuButton(label="Templates")
        tmpl.set_tooltip_text("Add a pre-connected chain of steps")
        tmenu = Gtk.Menu()
        for name, (label, _steps, _edges) in TEMPLATES.items():
            item = Gtk.MenuItem(label=label)
            item.connect("activate", lambda _w, n=name: self.add_template(n))
            tmenu.append(item)
        tmenu.show_all()
        tmpl.set_popup(tmenu)
        bar.pack_start(tmpl, False, False, 0)
        bar.pack_start(W.button("Delete", self.delete_selected, "Delete the selected box or connection (Del)",
                                "edit-delete-symbolic"), False, False, 0)
        bar.pack_start(W.button("Arrange", self.arrange, "Lay the boxes out left to right in plan order"),
                       False, False, 0)
        for icon, tip, f in (("zoom-out-symbolic", "Zoom out (Ctrl+wheel)", 0.85),
                             ("zoom-in-symbolic", "Zoom in (Ctrl+wheel)", 1 / 0.85)):
            b = Gtk.Button.new_from_icon_name(icon, Gtk.IconSize.BUTTON)
            b.set_tooltip_text(tip)
            b.connect("clicked", lambda *_, f=f: self.canvas.set_zoom(self.canvas.zoom * f))
            bar.pack_start(b, False, False, 0)
        bar.pack_start(W.button("Fit", self.fit, "Zoom so that the whole plan is visible"), False, False, 0)
        bar.pack_start(W.button("Check previous runs", lambda: self.check_runs(None),
                                "Look in each step's folder for the output files of an earlier run (under "
                                "the plan's time stamp, else an earlier one) and mark the step done, partial "
                                "or missing. A step can then use the outputs it found: the steps after it "
                                "read them, and it is not written or run again", "edit-find-symbolic"),
                       False, False, 0)
        bar.pack_end(W.button("Open folder", self.open_folder, "Open the plan folder", "folder-open-symbolic"),
                     False, False, 0)
        par = Gtk.Box(spacing=4)
        par.pack_start(W.label("Parallel"), False, False, 0)
        self.parallel_spin = W.spin(1, MAX_PARALLEL, 1, 0)
        self.parallel_spin.set_tooltip_text(
            "How many scripts Run all / Run selected start at the same time. Each step still waits for the "
            "steps it reads from. GPU steps (pmemd.cuda, gmx mdrun) running together share the GPU unless "
            "their Environment section sets CUDA_VISIBLE_DEVICES; CPU steps share the cores. Saved with "
            "the project.")
        self.parallel_spin.connect("value-changed", self._on_parallel)
        par.pack_start(self.parallel_spin, False, False, 0)
        bar.pack_end(par, False, False, 0)
        self.run_selected_btn = W.button("Run selected steps", self.run_selected,
                                         "Write the ticked steps' files and run their main scripts in plan "
                                         "order (tick a step with the box in its title bar, Space, or the "
                                         "right-click menu; steps they read from are not run again)",
                                         "media-playback-start-symbolic")
        bar.pack_end(self.run_selected_btn, False, False, 0)
        bar.pack_end(W.button("Run all…", lambda: self.write(None, run=True),
                              "Write every step's files, then run the main scripts in plan order on this "
                              "computer (scripts marked 'run separately', e.g. production runs, are not "
                              "started); each run is logged in the project file", "media-playback-start-symbolic"),
                     False, False, 0)
        write = W.button("Write all files", lambda: self.write(None),
                         "Write every step's files, a master script and a plan summary",
                         "document-save-symbolic")
        write.get_style_context().add_class("suggested-action")
        bar.pack_end(write, False, False, 0)
        return bar

    def _naming_bar(self) -> Gtk.Widget:
        bar = Gtk.Box(spacing=6, margin=6)
        bar.set_margin_top(0)
        bar.pack_start(W.label("Project"), False, False, 0)
        self.project_entry = W.entry("project", "The <project> part of every file name", width=12)
        self.project_entry.set_hexpand(False)
        self.project_entry.connect("changed", self._on_naming)
        bar.pack_start(self.project_entry, False, False, 0)
        self.stamp_check = W.check("Time stamp", "Append one time stamp per plan to every file name "
                                                "(<project>_<step>_<tool>_<time>.<ext>)")
        self.stamp_check.connect("toggled", self._on_naming)
        bar.pack_start(self.stamp_check, False, False, 0)
        self.stamp_label = W.label("")
        bar.pack_start(self.stamp_label, False, False, 0)
        bar.pack_start(W.button("New stamp", self.new_stamp, "Start a new time stamp (files written "
                                                           "before keep their names)"), False, False, 0)
        self.folders_check = W.check("Step folders", "Write each step into its own folder, e.g. "
                                                    "03_amber-build/")
        self.folders_check.connect("toggled", self._on_naming)
        bar.pack_start(self.folders_check, False, False, 0)
        bar.pack_start(W.label("Folder"), False, False, 0)
        self.folder_label = W.label("")
        self.folder_label.set_ellipsize(Pango.EllipsizeMode.START)
        self.folder_label.set_hexpand(True)
        bar.pack_start(self.folder_label, True, True, 0)
        bar.pack_start(W.button("Change…", self.choose_folder, "Folder that receives the plan"), False,
                       False, 0)
        return bar

    def _refresh_naming(self) -> None:
        wf = self.workflow
        stamp = self.plan.stamp if self.plan else ""
        self.stamp_label.set_text((stamp + ("" if wf.stamp else " (set on first write)")) if wf.use_stamp
                                  else "")
        self.folder_label.set_text(self.host.project.workflow_dir)

    def _details(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=8)
        box.set_size_request(360, -1)
        self.d_title = W.label("<b>Nothing selected</b>", markup=True, wrap=True)
        box.pack_start(self.d_title, False, False, 0)
        self.d_desc = W.hint("Select a box to see its files and checks. Double-click it to edit it in "
                             "the wizard.")
        box.pack_start(self.d_desc, False, False, 0)
        form = W.Form()
        form.set_margin_start(0)
        self.d_name = W.entry("(tool name)", "Title shown on the box", width=18)
        self.d_name.connect("changed", self._on_detail_edit)
        form.add("Title", self.d_name)
        self.d_step = W.spin(1, 99, 1, 0, "Step number: the <step> part of file names and the folder")
        self.d_step.connect("value-changed", self._on_detail_edit)
        self.d_step_row = form.add("Step", self.d_step)
        box.pack_start(form, False, False, 0)
        self.d_note = Gtk.TextView()
        self.d_note.set_wrap_mode(Gtk.WrapMode.WORD)
        self.d_note.get_buffer().connect("changed", self._on_detail_edit)
        self.d_note_sw = Gtk.ScrolledWindow()
        self.d_note_sw.set_shadow_type(Gtk.ShadowType.IN)
        self.d_note_sw.set_min_content_height(120)
        self.d_note_sw.add(self.d_note)
        box.pack_start(self.d_note_sw, False, False, 0)
        btns = Gtk.Box(spacing=4)
        self.d_wizard = W.button("Edit in wizard…", lambda: self.open_wizard(self.node(self._details_node)),
                                 "Open the step's options", "document-properties-symbolic")
        btns.pack_start(self.d_wizard, False, False, 0)
        self.d_write = W.button("Write step", lambda: self.write([self._details_node]),
                                "Write only this step's files")
        btns.pack_start(self.d_write, False, False, 0)
        self.d_run = W.button("Run step…", lambda: self.write([self._details_node], run=True),
                              "Write this step's files, then run its main script on this computer")
        btns.pack_start(self.d_run, False, False, 0)
        self.d_use = W.button("Use for analysis…",
                              lambda: self.use_for_analysis(self.node(self._details_node)),
                              "Hand the topology and trajectories this step reads to the MDATK analysis "
                              "window (Input & cleaning)")
        btns.pack_start(self.d_use, False, False, 0)
        box.pack_start(btns, False, False, 0)
        self.d_files = W.label("", wrap=True)
        self.d_files.set_selectable(True)
        self.d_files.set_line_wrap_mode(Pango.WrapMode.CHAR)
        box.pack_start(self.d_files, False, False, 0)
        self.d_prev = W.label("", wrap=True)  # the previous run found, or the one the step uses
        self.d_prev.set_selectable(True)
        self.d_prev.set_line_wrap_mode(Pango.WrapMode.CHAR)
        box.pack_start(self.d_prev, False, False, 0)
        prev = Gtk.Box(spacing=4)
        self.d_check = W.button("Check previous run", lambda: self.check_runs([self._details_node]),
                                "Look in the step's folder for the output files of an earlier run")
        prev.pack_start(self.d_check, False, False, 0)
        self.d_adopt = W.button("Use this run", lambda: self.adopt_runs([self._details_node]),
                                "Use the outputs found: the steps after this one read them, and this step is "
                                "not written or run again (saved with the project)")
        prev.pack_start(self.d_adopt, False, False, 0)
        self.d_forget = W.button("Stop using it", lambda: self.forget_runs([self._details_node]),
                                 "Write and run this step again; the steps after it read its new outputs")
        prev.pack_start(self.d_forget, False, False, 0)
        box.pack_start(prev, False, False, 0)
        self.d_issues = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.pack_start(self.d_issues, False, False, 0)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sw.add(box)
        return sw

    # ----------------------------------------------------------- details
    def show_details(self, node_id: Optional[str], edge: Optional[int] = None) -> None:
        self._details_node = node_id
        n = self.node(node_id) if node_id else None
        if self.on_select is not None:
            self.on_select(n)
        for child in list(self.d_issues.get_children()):
            self.d_issues.remove(child)
        self._loading = True
        try:
            if n is None:
                if edge is not None and 0 <= edge < len(self.workflow.edges):
                    e = self.workflow.edges[edge]
                    src, dst = self.node(e.source), self.node(e.target)
                    self.d_title.set_markup("<b>Connection</b>")
                    self.d_desc.set_text(f"{step_title(src) if src else e.source}: {e.output}  →  "
                                         f"{step_title(dst) if dst else e.target}: {e.input}\n"
                                         "Press Delete to remove it.")
                else:
                    self.d_title.set_markup("<b>Nothing selected</b>")
                    self.d_desc.set_text("Select a box to see its files and checks. Double-click it to edit "
                                         "it in the wizard. Right-click the canvas to add steps.")
                for w in (self.d_name, self.d_step, self.d_wizard, self.d_write, self.d_run, self.d_use,
                          self.d_check, self.d_adopt, self.d_forget):
                    w.set_sensitive(False)
                self.d_note_sw.hide()
                self.d_files.set_text("")
                self.d_prev.set_text("")
                return
            tool = TOOLS.get(n.tool)
            self.d_title.set_markup(f"<b>{GLib.markup_escape_text(step_title(n))}</b>")
            self.d_desc.set_text(tool.description if tool else f"unknown step type {n.tool}")
            self.d_name.set_sensitive(True)
            self.d_name.set_text(n.title)
            gen = bool(tool and tool.generates)
            self.d_step.set_sensitive(gen)
            self.d_step.set_value(max(1, n.step))
            self.d_wizard.set_sensitive(bool(tool) and n.tool != "note")
            self.d_write.set_sensitive(gen)
            self.d_run.set_sensitive(gen)
            self.d_use.set_sensitive(n.tool == "mdatk")
            if n.tool == "note":
                self.d_note.get_buffer().set_text(n.settings.get("text", ""))
                self.d_note_sw.show()
            else:
                self.d_note_sw.hide()
            step = self.plan.steps.get(n.id) if self.plan else None
            lines = []
            if step is not None and step.ctx is not None:
                if step.ctx.inputs:
                    lines.append("Inputs:")
                    for port, v in step.ctx.inputs.items():
                        src = f"  (from {v.source_label})" if v.source else ""
                        lines.append(f"  {port}: {', '.join(v.paths[:2])}{' …' if len(v.paths) > 2 else ''}"
                                     f"{src}")
                if step.outputs:
                    lines.append("Outputs:")
                    for port, v in step.outputs.items():
                        lines.append(f"  {port}: {', '.join(v.paths[:2])}{' …' if len(v.paths) > 2 else ''}")
            self.d_files.set_text("\n".join(lines))
            self.d_prev.set_text(self._previous_run_text(n) if gen else "")
            run = self.prev_runs.get(n.id)
            self.d_check.set_sensitive(gen)
            self.d_adopt.set_sensitive(bool(gen and run is not None and run.state == "done" and run.stamp
                                            and not run.adopted))
            self.d_forget.set_sensitive(bool(gen and n.adopted))
            if step is not None:
                for issue in step.issues:
                    lab = W.label(f'<span foreground="{LEVEL_COLORS[issue.level]}">'
                                  f'{issue.level.upper()}</span>: {GLib.markup_escape_text(issue.message)}',
                                  markup=True, wrap=True)
                    self.d_issues.pack_start(lab, False, False, 0)
            self.d_issues.show_all()
        finally:
            self._loading = False

    def _on_detail_edit(self, *_):
        if self._loading or self._details_node is None:
            return
        n = self.node(self._details_node)
        if n is None:
            return
        n.title = self.d_name.get_text().strip()
        if TOOLS.get(n.tool) and TOOLS[n.tool].generates:
            n.step = int(self.d_step.get_value())
        if n.tool == "note":
            buf = self.d_note.get_buffer()
            n.settings["text"] = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        self.schedule_resolve()

    def schedule_resolve(self) -> None:
        if self._resolve_id:
            GLib.source_remove(self._resolve_id)
        self._resolve_id = GLib.timeout_add(250, self._resolve_timeout)

    def _resolve_timeout(self):
        self._resolve_id = 0
        self.changed()
        return False

    # ----------------------------------------------------------- editing
    def _free_spot(self) -> Tuple[float, float]:
        y = 40.0
        for n in self.workflow.nodes:
            _w, h = self.canvas.node_size(n)
            if n.x < 300:
                y = max(y, n.y + h + 30)
        return 40.0, round(y / GRID) * GRID

    def add_step(self, tool: str, x: Optional[float] = None, y: Optional[float] = None) -> PlanNode:
        if x is None or y is None:
            x, y = self._free_spot()
        settings = {}
        if tool in ("amber-build", "amber-md", "amber-restraints") and self.host.project.cleaning.conda_env:
            settings["conda_env"] = self.host.project.cleaning.conda_env
        node = add_node(self.workflow, tool, x, y, settings=settings)
        self.canvas.select(node.id)
        self.changed()
        return node

    def add_template(self, name: str) -> None:
        x, y = self._free_spot()
        nodes = apply_template(self.workflow, name, x, y)
        if self.host.project.cleaning.conda_env:
            for n in nodes:
                if n.tool.startswith("amber"):
                    n.settings["conda_env"] = self.host.project.cleaning.conda_env
        self.set_status(f"Added {len(nodes)} steps")
        self.changed()
        self.fit(only_shrink=True)

    def fit(self, only_shrink: bool = False) -> None:
        """Zoom so the whole plan fits the visible canvas area."""
        alloc = self.scroller.get_allocation()
        if not self.workflow.nodes or alloc.width < 50:
            return
        right = max(n.x + self.canvas.node_size(n)[0] for n in self.workflow.nodes) + 40
        bottom = max(n.y + self.canvas.node_size(n)[1] for n in self.workflow.nodes) + 40
        zoom = min(alloc.width / right, alloc.height / bottom, 1.0)
        if only_shrink and zoom >= self.canvas.zoom:
            return
        self.canvas.set_zoom(max(0.4, zoom))

    def duplicate(self, node: PlanNode) -> None:
        tool = TOOLS.get(node.tool)
        new = add_node(self.workflow, node.tool, node.x + 30, node.y + 30,
                       step=next_step(self.workflow) if tool and tool.generates else 0,
                       title=node.title, settings=copy.deepcopy(node.settings))
        new.files = dict(node.files)
        self.canvas.select(new.id)
        self.changed()

    def delete_selected(self) -> None:
        c = self.canvas
        if c.selected is not None:
            win = self.wizards.pop(c.selected, None)
            if win is not None:
                win.destroy()
            remove_node(self.workflow, c.selected)
            c.selected = None
            self.show_details(None)
        elif c.selected_edge is not None and 0 <= c.selected_edge < len(self.workflow.edges):
            del self.workflow.edges[c.selected_edge]
            c.selected_edge = None
            self.show_details(None)
        self.changed()

    def arrange(self) -> None:
        """Place steps in columns by dependency depth (plan order inside a column)."""
        plan = self.resolve()
        depth: Dict[str, int] = {}
        for nid in plan.order:
            preds = [e.source for e in self.workflow.edges if e.target == nid and e.source in depth]
            depth[nid] = 1 + max((depth[p] for p in preds), default=-1)
        rows: Dict[int, int] = {}
        for nid in plan.order:
            n = self.node(nid)
            if n is None or n.tool == "note":
                continue
            col = depth[nid]
            row = rows.get(col, 0)
            n.x = 40.0 + col * 290
            n.y = 40.0 + row * 200
            rows[col] = row + 1
        self.changed()

    def _on_naming(self, *_):
        if self._loading:
            return
        wf = self.workflow
        wf.project_name = self.project_entry.get_text().strip() or "project"
        wf.use_stamp = self.stamp_check.get_active()
        wf.step_folders = self.folders_check.get_active()
        self.schedule_resolve()

    def new_stamp(self) -> None:
        self.workflow.stamp = new_stamp()
        self.changed()

    def choose_folder(self) -> None:
        dlg = Gtk.FileChooserDialog(title="Plan folder", transient_for=self.toplevel(),
                                    action=Gtk.FileChooserAction.SELECT_FOLDER)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Select", Gtk.ResponseType.OK)
        if dlg.run() == Gtk.ResponseType.OK and dlg.get_filename():
            self.workflow.out_dir = dlg.get_filename()
        dlg.destroy()
        self.changed()

    # ----------------------------------------------------------- actions
    def open_wizard(self, node: Optional[PlanNode]) -> None:
        if node is None or node.tool == "note" or node.tool not in TOOLS:
            return
        win = self.wizards.get(node.id)
        if win is not None:
            win.present()
            return
        win = WizardWindow(self.host, node=node, plan_provider=self.resolve,
                           on_changed=lambda: self._wizard_changed())
        win.set_transient_for(self.toplevel())
        win.connect("destroy", lambda *_: self.wizards.pop(node.id, None))
        self.wizards[node.id] = win

    def _wizard_changed(self) -> None:
        self._update_status()
        self.canvas.update_size()
        self.canvas.queue_draw()
        if self._details_node is not None and not self._loading:
            self.show_details(self._details_node)

    def write(self, node_ids: Optional[List[str]], run: bool = False) -> None:
        plan = self.resolve()
        ids = node_ids if node_ids is not None else list(plan.steps)
        if run:
            plan = self._ask_adopted(plan, ids)
            if plan is None:
                return
        bad = [(plan.steps[n].label, i.message) for n in ids if n in plan.steps and not plan.steps[n].adopted
               for i in plan.steps[n].issues if i.level == "error"]
        if bad:
            text = "\n".join(f"• {label}: {msg}" for label, msg in bad[:15])
            if len(bad) > 15:
                text += f"\n… and {len(bad) - 15} more"
            if not self._ask("Some steps have errors", text + "\n\nWrite the other steps anyway? Steps "
                             "with errors are skipped."):
                return
        try:
            with logged(self.host.project_path, "workflow write", "Generated the plan's files") as entry:
                report = plan.write(node_ids)
                entry.folder, entry.files = report.root, list(report.written)
                entry.summary = f"{len(report.written)} file(s) written, {len(report.skipped)} step(s) skipped"
        except OSError as exc:
            self._message("Could not write the plan", str(exc), Gtk.MessageType.ERROR)
            return
        self.changed()
        msg = f"Wrote {len(report.written)} files to {report.root}"
        if report.skipped:
            msg += f" ({len(report.skipped)} steps skipped)"
        if report.kept:
            msg += f" ({len(report.kept)} steps use the outputs of an earlier run)"
        self.set_status(msg)
        self.host.append_log("Planner: " + msg)
        if report.master:
            self.host.append_log(f"Master script: {report.master}")
        if run:
            ok_ids = [n for n in (node_ids if node_ids is not None else plan.order) if n not in report.skipped]
            jobs = plan_jobs(plan, ok_ids)
            if not jobs and report.kept:
                self.set_status(msg + "; nothing else to run")
                return
            self._run_token += 1
            token = self._run_token
            separate = separate_scripts(plan, ok_ids)
            self.host.run_scripts(jobs, separate, parent=self.toplevel(),
                                  parallel=int(self.parallel_spin.get_value()),
                                  on_state=lambda j, st: self._on_run_state(j, st, token),
                                  on_finished=lambda: self._on_run_finished(token),
                                  with_separate=plan_jobs(plan, ok_ids, include_separate=True) if separate
                                  else None)

    # ------------------------------------------------------- previous runs
    def _previous_run_text(self, n: PlanNode) -> str:
        lines = []
        if n.adopted:
            lines.append(f"Uses the outputs of its run of {n.adopted}: the steps after it read them; it is "
                         "not written or run again.")
        run = self.prev_runs.get(n.id)
        if run is not None:
            lines.append("Previous run: " + run.summary())
            if run.state == "partial":
                lines += [f"  missing or unfinished: {p}" for p in run.missing[:6]]
                if len(run.missing) > 6:
                    lines.append(f"  … and {len(run.missing) - 6} more")
        elif not n.adopted:
            lines.append("Previous run: not checked ('Check previous runs')")
        return "\n".join(lines)

    def check_runs(self, node_ids: Optional[List[str]] = None) -> Dict[str, runstate.RunStatus]:
        """'Check previous runs': look for the output files of earlier runs of every step (or ``node_ids``),
        mark them on the boxes and offer to use complete runs the steps do not use yet."""
        plan = self.resolve()
        runs = runstate.check_runs(plan, node_ids)
        if node_ids is None:
            self.prev_runs = dict(runs)
        else:
            self.prev_runs.update(runs)
        counts = {s: sum(1 for r in runs.values() if r.state == s) for s in ("done", "partial", "missing")}
        self.set_status(f"Previous runs: {counts['done']} done, {counts['partial']} partial, "
                        f"{counts['missing']} missing (folder {plan.root})")
        self.host.append_log(f"Check previous runs: {counts['done']} done, {counts['partial']} partial, "
                             f"{counts['missing']} missing (folder {plan.root})")
        for nid, run in runs.items():
            self.host.append_log(f"  {plan.steps[nid].label}: {run.summary()}")
        self.changed(resolve=False)
        offer = [n for n, r in runs.items() if r.state == "done" and r.stamp and not r.adopted]
        if offer:
            text = "\n".join(f"• {plan.steps[n].label}: run {runs[n].stamp}, finished {runs[n].date}"
                             for n in offer[:15])
            if len(offer) > 15:
                text += f"\n… and {len(offer) - 15} more"
            if self._ask(f"Use the outputs of {len(offer)} finished run(s)?",
                         text + "\n\nThe steps after them then read these files, and these steps are not "
                                "written or run again (until 'Stop using it'). Saved with the project."):
                self.adopt_runs(offer)
        return runs

    def adopt_runs(self, node_ids: List[str]) -> None:
        """Let steps use the complete runs 'Check previous runs' found for them."""
        plan = self.plan or self.resolve()
        changed = runstate.adopt(plan, {n: self.prev_runs[n] for n in node_ids if n in self.prev_runs})
        for nid in changed:
            self.prev_runs[nid].adopted = True
            self.host.append_log(f"Planner: {plan.steps[nid].label} uses the outputs of its run of "
                                 f"{plan.steps[nid].node.adopted}")
        if changed:
            self.set_status(f"{len(changed)} step(s) use the outputs of an earlier run")
        self.changed()

    def forget_runs(self, node_ids: List[str]) -> None:
        """Stop using earlier runs: those steps are written and run again."""
        changed = runstate.forget(self.workflow, node_ids)
        for nid in changed:
            self.prev_runs.pop(nid, None)
            n = self.node(nid)
            self.host.append_log(f"Planner: {step_title(n) if n else nid} no longer uses an earlier run "
                                 "(it is written and run again)")
        if changed:
            self.set_status(f"{len(changed)} step(s) will be written and run again")
        self.changed()

    def _adopted_choice(self, text: str, detail: str) -> str:
        """'skip', 'rerun' or 'cancel' for steps of a run that use the outputs of an earlier run."""
        dlg = Gtk.MessageDialog(transient_for=self.toplevel(), modal=True,
                                message_type=Gtk.MessageType.QUESTION, buttons=Gtk.ButtonsType.NONE,
                                text=text)
        dlg.format_secondary_text(detail)
        dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Run them again", Gtk.ResponseType.REJECT,
                        "Skip them", Gtk.ResponseType.OK)
        dlg.set_default_response(Gtk.ResponseType.OK)
        response = dlg.run()
        dlg.destroy()
        return {Gtk.ResponseType.OK: "skip", Gtk.ResponseType.REJECT: "rerun"}.get(response, "cancel")

    def _ask_adopted(self, plan: Plan, ids: List[str]) -> Optional[Plan]:
        """Before a run: steps among ``ids`` that use an earlier run are skipped, or (on request) run again
        (they stop using it).  Returns the plan to run, or None when the user cancels."""
        kept = [n for n in ids if n in plan.steps and plan.steps[n].adopted]
        if not kept:
            return plan
        lines = []
        for nid in kept[:15]:
            step = plan.steps[nid]
            run = runstate.run_status(plan, nid, step.adopted)
            state = "complete" if run.state == "done" else f"{run.state}: {len(run.missing)} file(s) missing"
            lines.append(f"• {step.label}: run {step.adopted} ({state})")
        choice = self._adopted_choice(
            f"{len(kept)} step(s) use the outputs of an earlier run",
            "\n".join(lines) + "\n\nSkip them (the steps after them read those outputs), or run them again "
                               "(they stop using the earlier run)?")
        if choice == "cancel":
            return None
        if choice == "rerun":
            self.forget_runs(kept)
            return self.resolve()
        return plan

    # ------------------------------------------------------- running steps
    def _on_parallel(self, spin) -> None:
        if not self._loading:
            self.workflow.parallel = int(spin.get_value())

    def _on_run_state(self, job, state: str, token: Optional[int] = None) -> None:
        """A script changed state (main loop); the first 'queued' of a run clears the last run's marks."""
        if (token is not None and token != self._run_token) or self._closed:
            return  # a run started before the project was reloaded, or the window is gone
        if not job.node_id or self.node(job.node_id) is None:
            return
        if state == "queued" and not any(v in ("queued", "running") for v in self.run_state.values()):
            self.run_state.clear()
        prev = self.run_state.get(job.node_id)
        # a step with several scripts: keep 'failed' and do not fall back from running/ok to queued
        if prev == "failed" or (state == "queued" and prev is not None):
            return
        self.run_state[job.node_id] = state
        if state == "running" and not self._spin_id:
            self._spin_id = GLib.timeout_add(50, self._spin_tick)
        self.canvas.queue_draw()

    def _on_run_finished(self, token: Optional[int] = None) -> None:
        if (token is not None and token != self._run_token) or self._closed:
            return
        for nid, st in list(self.run_state.items()):
            if st in ("queued", "running"):  # never started, or the run stopped
                self.run_state[nid] = "cancelled"
        self.canvas.queue_draw()

    def _spin_tick(self) -> bool:
        if self._closed:
            self._spin_id = 0
            return False
        self.canvas.spin_phase = (self.canvas.spin_phase + 0.35) % (2 * math.pi)
        self.canvas.queue_draw()
        if any(v == "running" for v in self.run_state.values()):
            return True
        self._spin_id = 0
        return False

    # ------------------------------------------------------- run selected
    def toggle_checked(self, node_id: str) -> None:
        self.set_checked(self.checked ^ {node_id})

    def set_checked(self, node_ids) -> None:
        self.checked = {n for n in node_ids if self.node(n) is not None and checkable(self.node(n))}
        self._update_run_selected()
        self.canvas.queue_draw()

    def selected_in_order(self) -> List[str]:
        plan = self.plan or self.resolve()
        return [n for n in plan.order if n in self.checked]

    def _update_run_selected(self) -> None:
        btn = getattr(self, "run_selected_btn", None)
        if btn is None:
            return
        n = len(self.checked)
        btn.set_label(f"Run selected steps ({n})" if n else "Run selected steps")
        btn.set_sensitive(bool(n))

    def run_selected(self) -> None:
        self.resolve()
        ids = self.selected_in_order()
        if ids:
            self.write(ids, run=True)

    def open_folder(self) -> None:
        path = self.host.project.workflow_dir
        if not os.path.isdir(path):
            self._message("Nothing written yet", f"{path} does not exist yet; use 'Write all files' first.")
            return
        desktop.open_folder(path)

    def use_for_analysis(self, node: Optional[PlanNode]) -> None:
        """Hand the files an MDATK analysis step reads to the analysis window (through the host)."""
        if node is None:
            return
        plan = self.resolve()
        top, trajs = analysis_inputs(plan, node.id)
        if not top:
            self._message("No topology", "Connect a topology (and trajectory) to the analysis step first.")
            return
        clean = bool(plan.steps[node.id].settings.get("clean", True))
        self.host.use_for_analysis(top, trajs, clean)

    def _ask(self, text: str, detail: str) -> bool:
        dlg = Gtk.MessageDialog(transient_for=self.toplevel(), modal=True,
                                message_type=Gtk.MessageType.WARNING, buttons=Gtk.ButtonsType.YES_NO,
                                text=text)
        dlg.format_secondary_text(detail)
        ok = dlg.run() == Gtk.ResponseType.YES
        dlg.destroy()
        return ok

    def _message(self, text: str, detail: str = "", kind=Gtk.MessageType.INFO) -> None:
        dlg = Gtk.MessageDialog(transient_for=self.toplevel(), modal=True, message_type=kind,
                                buttons=Gtk.ButtonsType.CLOSE, text=text)
        if detail:
            dlg.format_secondary_text(detail)
        dlg.run()
        dlg.destroy()

    def shutdown(self) -> None:
        """The window closes: close the step wizards and stop the timers (callbacks of a run that goes on
        are ignored from now on)."""
        for win in list(self.wizards.values()):
            win.destroy()
        self.wizards.clear()
        if self._resolve_id:
            GLib.source_remove(self._resolve_id)
            self._resolve_id = 0
        self._closed = True
        if self._spin_id:
            GLib.source_remove(self._spin_id)
            self._spin_id = 0
