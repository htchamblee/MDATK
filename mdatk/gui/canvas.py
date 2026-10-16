"""WYSIWYG figure canvas.

Renders the actual multi-panel figure (the same panel code as the video) at
the window size, preserving the video's aspect ratio and type scale.

* Drag inside the structure panel to rotate the model (only the moving
  artists are redrawn, so rotation is smooth).  In map mode the drag moves
  the density map instead (Shift + drag rotates it about its centre).
* Drag any other panel - or the structure panel by its title strip, or any
  panel with Shift held - onto another slot to swap the two panels.  Panels
  snap to the slots of the selected grid preset.
* Double-click a panel to edit its settings.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
import cairo  # noqa: E402
from gi.repository import Gdk, Gtk  # noqa: E402

from ..config import Project  # noqa: E402
from ..data import Dataset  # noqa: E402
from ..render import layout as layouts  # noqa: E402
from ..render.orientation import drag as drag_rotation  # noqa: E402
from ..render.orientation import rx, ry  # noqa: E402
from ..render.panels import StructurePanel  # noqa: E402
from ..render.theme import get_theme  # noqa: E402
from ..render.video import build_figure  # noqa: E402

TITLE_STRIP = 0.12  # top fraction of a slot that drags the structure panel
DRAG_THRESHOLD = 4  # pixels before a press becomes a drag
PREVIEW_FRAME = 0.35  # which part of the example time axis the cursor shows

Rect = Tuple[float, float, float, float]  # x, y, w, h in widget pixels


def _rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


class FigureCanvas(Gtk.DrawingArea):
    def __init__(self, on_rotate: Optional[Callable[[List[float]], None]] = None,
                 on_swap: Optional[Callable[[int, int], None]] = None,
                 on_activate: Optional[Callable[[int], None]] = None,
                 on_map_move: Optional[Callable[[List[float]], None]] = None):
        super().__init__()
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_size_request(480, 300)
        self.set_can_focus(True)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                        | Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.LEAVE_NOTIFY_MASK)
        self.connect("draw", self._on_draw)
        self.connect("button-press-event", self._on_press)
        self.connect("button-release-event", self._on_release)
        self.connect("motion-notify-event", self._on_motion)
        self.connect("size-allocate", lambda *_: self.schedule_render())
        self.on_rotate = on_rotate
        self.on_swap = on_swap
        self.on_activate = on_activate
        self.on_map_move = on_map_move
        self.map_mode = False  # drags in the structure panel move the density map
        self.project: Optional[Project] = None
        self.dataset: Optional[Dataset] = None
        self.watermark = ""
        self.surface: Optional[cairo.ImageSurface] = None
        self.offset = (0.0, 0.0)
        self.slots: List[Rect] = []
        self.panel_types: List[str] = []
        self.structure: Optional[StructurePanel] = None
        self.structure_slot = -1
        self.angles = [0.0, 0.0, 0.0]
        self.message = "Building preview …"
        self._fig = self._canvas = self._background = None
        self._pos = 0
        self._press = None  # (x, y, slot, mode)
        self._drag_target: Optional[int] = None
        self._dragging = False
        self._pending = None
        self.render_count = 0

    # ------------------------------------------------------------ rendering
    def set_content(self, project: Project, dataset: Dataset, watermark: str = "") -> None:
        self.project = project
        self.dataset = dataset
        self.watermark = watermark
        self.angles = [float(a) for a in project.visual.rotation]
        self.schedule_render()

    def schedule_render(self, delay_ms: int = 120) -> None:
        from gi.repository import GLib

        if self._pending is not None:
            GLib.source_remove(self._pending)
        self._pending = GLib.timeout_add(delay_ms, self._render_now)

    def _render_now(self):
        self._pending = None
        self.render()
        return False

    def render(self) -> None:
        """Rebuild the figure at the current widget size."""
        alloc = self.get_allocation()
        if self.project is None or self.dataset is None or alloc.width < 50 or alloc.height < 50:
            self.queue_draw()
            return
        vp = self.project.output.video
        scale = min(alloc.width / vp.width, alloc.height / vp.height)
        w, h = max(2, int(vp.width * scale)), max(2, int(vp.height * scale))
        try:
            fig, canvas, panels = build_figure(self.dataset, self.project, w, h,
                                               dpi=vp.dpi * scale, watermark=self.watermark)
        except Exception as exc:  # noqa: BLE001 - show any layout problem in the canvas
            self.surface = None
            self.message = f"Cannot draw this layout: {exc}"
            self.queue_draw()
            return
        self._fig, self._canvas = fig, canvas
        self._pos = int(PREVIEW_FRAME * max(0, self.dataset.n_frames - 1))
        self.structure, self.structure_slot = None, -1
        for k, p in enumerate(panels):
            p.update(self._pos)
            if isinstance(p, StructurePanel) and p.ok and self.structure is None:
                self.structure, self.structure_slot = p, k
        animated = self._moving_artists()
        for a in animated:
            a.set_animated(True)
        canvas.draw()
        self._background = canvas.copy_from_bbox(fig.bbox)
        self._blit()
        cw, ch = canvas.get_width_height()
        self.offset = ((alloc.width - cw) / 2.0, (alloc.height - ch) / 2.0)
        self._compute_slots(fig, cw, ch, panels)
        self.panel_types = [s.type for s in self.project.layout.panels]
        self.render_count += 1
        self.queue_draw()

    def _compute_slots(self, fig, cw: int, ch: int, panels) -> None:
        """Pixel rectangles of the slots (hit areas for dragging).

        Slots with a panel use the panel's drawn extent (axes, labels, colour bars);
        empty slots fall back to an even split of the grid.
        """
        rows, cols, slots = layouts.preset(self.project.layout.preset)
        ox, oy = self.offset
        renderer = self._canvas.get_renderer()
        out: List[Rect] = []
        for k, (r0, r1, c0, c1) in enumerate(slots):
            rect = (ox + cw * c0 / cols, oy + ch * r0 / rows,
                    cw * (c1 - c0) / cols, ch * (r1 - r0) / rows)
            if k < len(panels):
                axes = [a for a in fig.axes if a is panels[k].ax or
                        a.get_subplotspec() is not None and a.get_subplotspec() == panels[k].ax.get_subplotspec()]
                boxes = [a.get_tightbbox(renderer) for a in axes]
                boxes = [b for b in boxes if b is not None and b.width > 0]
                if boxes:
                    x0 = min(b.x0 for b in boxes)
                    x1 = max(b.x1 for b in boxes)
                    y0 = min(b.y0 for b in boxes)
                    y1 = max(b.y1 for b in boxes)
                    rect = (ox + x0, oy + ch - y1, x1 - x0, y1 - y0)
            out.append(rect)
        self.slots = out

    def _moving_artists(self) -> list:
        """Artists redrawn on every blit: the density shadow (changes with the view)
        below the structure panel's animated artists."""
        if self.structure is None:
            return []
        return self.structure.rotation_artists() + self.structure.animated_artists()

    def _blit(self) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        if self.structure is not None and self._background is not None:
            canvas.restore_region(self._background)
            for a in self._moving_artists():
                self._fig.draw_artist(a)
        buf = np.asarray(canvas.buffer_rgba())
        h, w = buf.shape[:2]
        bgra = np.ascontiguousarray(buf[..., [2, 1, 0, 3]])
        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
        if stride != w * 4:
            padded = np.zeros((h, stride // 4, 4), dtype=np.uint8)
            padded[:, :w] = bgra
            bgra = padded
        self._buffer = bgra  # keep alive while the surface is used
        self.surface = cairo.ImageSurface.create_for_data(memoryview(bgra), cairo.FORMAT_ARGB32,
                                                          w, h, stride)

    def set_rotation(self, angles: Sequence[float]) -> None:
        """Re-orient the structure panel without rebuilding the figure."""
        self.angles = [float(a) for a in angles]
        if self.project is not None:
            self.project.visual.rotation = list(self.angles)
        if self.structure is None:
            return
        self.structure.set_rotation(self.angles)
        self.structure.update(self._pos)
        self._blit()
        self.queue_draw()

    # ---------------------------------------------------------------- input
    def slot_at(self, x: float, y: float) -> Optional[int]:
        for k, (sx, sy, sw, sh) in enumerate(self.slots):
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                return k
        return None

    def _on_press(self, _w, event):
        self.grab_focus()
        slot = self.slot_at(event.x, event.y)
        if slot is None or event.button != 1:
            return False
        if event.type == Gdk.EventType._2BUTTON_PRESS:
            self._press = None
            if self.on_activate:
                self.on_activate(slot)
            return True
        sx, sy, sw, sh = self.slots[slot]
        in_title = event.y < sy + TITLE_STRIP * sh
        shift = bool(event.state & Gdk.ModifierType.SHIFT_MASK)
        in_structure = slot == self.structure_slot and not in_title
        if in_structure and self.map_mode and self.structure is not None and self.structure.has_density():
            mode = "map-rotate" if shift else "map"
        else:
            mode = "rotate" if (in_structure and not shift) else "move"
        self._press = (event.x, event.y, slot, mode)
        self._last = (event.x, event.y)
        self._dragging = False
        return True

    def _on_motion(self, _w, event):
        if self._press is None:
            slot = self.slot_at(event.x, event.y)
            win = self.get_window()
            if win is not None:
                name = "default"
                if slot is not None:
                    sx, sy, sw, sh = self.slots[slot]
                    rotating = slot == self.structure_slot and event.y >= sy + TITLE_STRIP * sh
                    name = ("all-scroll" if self.map_mode else "grab") if rotating else "move"
                win.set_cursor(Gdk.Cursor.new_from_name(win.get_display(), name))
            return False
        x0, y0, slot, mode = self._press
        if not self._dragging and abs(event.x - x0) + abs(event.y - y0) < DRAG_THRESHOLD:
            return True
        self._dragging = True
        if mode == "rotate":
            dx, dy = event.x - self._last[0], event.y - self._last[1]
            self._last = (event.x, event.y)
            if self.structure is not None:
                self.structure.density_fast = True
            self.set_rotation(drag_rotation(self.angles, dx, dy))
            if self.on_rotate:
                self.on_rotate(list(self.angles))
        elif mode in ("map", "map-rotate"):
            dx, dy = event.x - self._last[0], event.y - self._last[1]
            self._last = (event.x, event.y)
            self._drag_map(dx, dy, mode == "map-rotate")
        else:
            self._drag_target = self.slot_at(event.x, event.y)
            self._drag_pos = (event.x, event.y)
            self.queue_draw()
        return True

    def _on_release(self, _w, event):
        press, self._press = self._press, None
        target, self._drag_target = self._drag_target, None
        if press is None:
            return False
        _x, _y, slot, mode = press
        if mode == "move" and self._dragging and target is not None and target != slot:
            if self.on_swap:
                self.on_swap(slot, target)
        if self.structure is not None and self.structure.density_fast:
            self.structure.density_fast = False  # full-quality shadow after the drag
            self.structure.update_density()
            self._blit()
        self._dragging = False
        self.queue_draw()
        return True

    def _drag_map(self, dx: float, dy: float, rotate: bool) -> None:
        st = self.structure
        st.density_fast = True
        if rotate:
            new = st.nudge_density(view_rotation=ry(dx * 0.5) @ rx(dy * 0.5))
        else:  # widget pixels -> Å in the panel's data coordinates (screen y points down)
            bbox = st.ax.get_window_extent()
            (x0, x1), (y0, y1) = st.ax.get_xlim(), st.ax.get_ylim()
            new = st.nudge_density(shift=(dx * (x1 - x0) / max(bbox.width, 1.0),
                                          -dy * (y1 - y0) / max(bbox.height, 1.0)))
        self._blit()
        self.queue_draw()
        if new is not None and self.on_map_move:
            self.on_map_move(new)

    # ----------------------------------------------------------------- draw
    def _on_draw(self, _w, cr):
        theme = get_theme(self.project.visual.theme if self.project else "light")
        cr.set_source_rgb(*_rgb(theme.page))
        cr.paint()
        if self.surface is None:
            alloc = self.get_allocation()
            cr.set_source_rgb(*_rgb(theme.muted))
            cr.select_font_face("Sans")
            cr.set_font_size(13)
            ext = cr.text_extents(self.message)
            cr.move_to(max(8, (alloc.width - ext.width) / 2), alloc.height / 2)
            cr.show_text(self.message)
            return False
        cr.set_source_surface(self.surface, *self.offset)
        cr.paint()
        if self._press is not None and self._dragging and self._press[3] == "move":
            self._draw_drag(cr)
        return False

    def _draw_drag(self, cr):
        src = self._press[2]
        sx, sy, sw, sh = self.slots[src]
        cr.set_source_rgba(0.16, 0.47, 0.84, 0.10)
        cr.rectangle(sx + 2, sy + 2, sw - 4, sh - 4)
        cr.fill()
        target = self._drag_target
        if target is not None and target != src:
            tx, ty, tw, th = self.slots[target]
            cr.set_source_rgba(0.16, 0.47, 0.84, 0.18)
            cr.rectangle(tx + 3, ty + 3, tw - 6, th - 6)
            cr.fill_preserve()
            cr.set_source_rgba(0.16, 0.47, 0.84, 0.9)
            cr.set_line_width(2.5)
            cr.set_dash([8, 5])
            cr.stroke()
            cr.set_dash([])
        label = self.panel_types[src] if src < len(self.panel_types) else "panel"
        x, y = getattr(self, "_drag_pos", (sx, sy))
        cr.select_font_face("Sans")
        cr.set_font_size(12)
        ext = cr.text_extents(label)
        cr.set_source_rgba(0.16, 0.47, 0.84, 0.92)
        cr.rectangle(x + 10, y + 8, ext.width + 14, ext.height + 10)
        cr.fill()
        cr.set_source_rgb(1, 1, 1)
        cr.move_to(x + 17, y + 13 + ext.height)
        cr.show_text(label)
