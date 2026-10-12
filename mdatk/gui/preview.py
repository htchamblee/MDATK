"""Interactive orientation preview (Cairo drawing of the reference frame trace)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk  # noqa: E402

from ..render.orientation import drag, euler_to_matrix  # noqa: E402
from ..render.theme import get_theme  # noqa: E402


def _rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


class OrientationView(Gtk.DrawingArea):
    """Drag with the left mouse button to rotate; the angles are reported back."""

    def __init__(self, on_rotate: Optional[Callable[[Sequence[float]], None]] = None):
        super().__init__()
        self.set_size_request(420, 360)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                        | Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect("draw", self._draw)
        self.connect("button-press-event", self._press)
        self.connect("button-release-event", self._release)
        self.connect("motion-notify-event", self._motion)
        self.on_rotate = on_rotate
        self.angles = [0.0, 0.0, 0.0]
        self.coords: Optional[np.ndarray] = None
        self.ions = np.zeros((0, 3))
        self.links = np.zeros((0, 2), dtype=int)
        self.colors = None
        self.ion_color = "#008300"
        self.theme = get_theme("light")
        self.message = "Load a topology and trajectory, then click “Load preview”."
        self._last = None

    # ----------------------------------------------------------------- data
    def set_structure(self, coords, ions, links, colors, ion_color: str = "#008300") -> None:
        self.coords = np.asarray(coords, dtype=float)
        self.ions = np.asarray(ions, dtype=float).reshape(-1, 3)
        self.links = np.asarray(links, dtype=int).reshape(-1, 2)
        self.colors = np.asarray(colors, dtype=float)
        self.ion_color = ion_color
        self.message = ""
        self.queue_draw()

    def set_message(self, text: str) -> None:
        self.message = text
        self.queue_draw()

    def set_angles(self, angles: Sequence[float]) -> None:
        self.angles = [float(a) for a in angles]
        self.queue_draw()

    def set_theme(self, name: str) -> None:
        self.theme = get_theme(name)
        self.queue_draw()

    # -------------------------------------------------------------- events
    def _press(self, _w, event):
        if event.button == 1:
            self._last = (event.x, event.y)
        return True

    def _release(self, _w, _event):
        self._last = None
        return True

    def _motion(self, _w, event):
        if self._last is None or self.coords is None:
            return False
        dx, dy = event.x - self._last[0], event.y - self._last[1]
        self._last = (event.x, event.y)
        self.angles = list(drag(self.angles, dx, dy))
        if self.on_rotate:
            self.on_rotate(self.angles)
        self.queue_draw()
        return True

    # ----------------------------------------------------------------- draw
    def _draw(self, _w, cr):
        alloc = self.get_allocation()
        w, h = alloc.width, alloc.height
        cr.set_source_rgb(*_rgb(self.theme.surface))
        cr.paint()
        if self.coords is None or len(self.coords) == 0:
            cr.set_source_rgb(*_rgb(self.theme.muted))
            cr.select_font_face("Sans")
            cr.set_font_size(13)
            ext = cr.text_extents(self.message)
            cr.move_to(max(8, (w - ext.width) / 2), h / 2)
            cr.show_text(self.message)
            return False
        center = self.coords.mean(axis=0)
        r = euler_to_matrix(self.angles)
        pts = (self.coords - center) @ r.T
        ions = (self.ions - center) @ r.T if len(self.ions) else np.zeros((0, 3))
        radius = max(np.linalg.norm(self.coords - center, axis=1).max(), 1.0)
        scale = 0.45 * min(w, h) / radius

        def screen(p):
            return w / 2 + p[0] * scale, h / 2 - p[1] * scale

        zspan = 2 * radius
        depth = (pts[:, 2] + radius) / zspan
        surf = np.array(_rgb(self.theme.surface))
        # links (far first)
        if len(self.links):
            seg_depth = 0.5 * (depth[self.links[:, 0]] + depth[self.links[:, 1]])
            for k in np.argsort(seg_depth):
                i, j = self.links[k]
                col = 0.5 * (self.colors[i, :3] + self.colors[j, :3])
                fog = 0.45 * (1 - seg_depth[k])
                cr.set_source_rgb(*(col * (1 - fog) + surf * fog))
                cr.set_line_width(1.2 + 2.5 * seg_depth[k])
                cr.move_to(*screen(pts[i]))
                cr.line_to(*screen(pts[j]))
                cr.stroke()
        for k in np.argsort(depth):
            col = self.colors[k, :3]
            fog = 0.45 * (1 - depth[k])
            x, y = screen(pts[k])
            cr.arc(x, y, 2.0 + 3.0 * depth[k], 0, 2 * np.pi)
            cr.set_source_rgb(*(col * (1 - fog) + surf * fog))
            cr.fill()
        for p in ions:
            x, y = screen(p)
            cr.arc(x, y, 4.5, 0, 2 * np.pi)
            cr.set_source_rgb(*_rgb(self.ion_color))
            cr.fill()
        self._draw_triad(cr, r, h)
        return False

    def _draw_triad(self, cr, r, h):
        """Small X/Y/Z axis indicator in the lower-left corner."""
        ox, oy, length = 34, h - 34, 22
        names = ("X", "Y", "Z")
        cols = ((0.89, 0.29, 0.28), (0.11, 0.69, 0.48), (0.16, 0.47, 0.84))
        axes = r @ np.eye(3)
        order = np.argsort(axes[2])
        for k in order:
            v = axes[:, k]
            cr.set_source_rgb(*cols[k])
            cr.set_line_width(2)
            cr.move_to(ox, oy)
            ex, ey = ox + v[0] * length, oy - v[1] * length
            cr.line_to(ex, ey)
            cr.stroke()
            cr.select_font_face("Sans")
            cr.set_font_size(10)
            cr.move_to(ex + 2, ey + 4)
            cr.show_text(names[k])
