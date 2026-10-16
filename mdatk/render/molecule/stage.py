"""The structural view's 3D layer: renders the scene into an RGBA image that is
drawn 1:1 into the panel's axes (above the density shadow and glow halos, below
labels and the legend)."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from matplotlib.artist import Artist

from .geometry import Spheres
from .scene import Colored, MoleculeScene

log = logging.getLogger(__name__)


class PixelImage(Artist):
    """An RGBA image (row 0 at the top) drawn at the axes' pixel box without resampling.

    When the axes' pixel size differs from the image (another dpi, a resize), the
    ``rerender`` callback is asked for a matching image first."""

    def __init__(self, ax, rerender: Callable[[int, int], Optional[np.ndarray]]):
        super().__init__()
        self.ax = ax
        self.rgba: Optional[np.ndarray] = None
        self.rerender = rerender
        self.set_zorder(3)
        self.set_figure(ax.figure)

    def get_window_extent(self, renderer=None):
        return self.ax.get_window_extent(renderer)

    def draw(self, renderer):
        if not self.get_visible():
            return
        bb = self.ax.get_window_extent(renderer)
        w, h = int(round(bb.width)), int(round(bb.height))
        if w < 2 or h < 2:
            return
        if self.rgba is None or self.rgba.shape[:2] != (h, w):
            self.rgba = self.rerender(w, h)
            if self.rgba is None:
                return
        gc = renderer.new_gc()
        gc.set_clip_rectangle(bb)
        gc.set_alpha(1.0)
        renderer.draw_image(gc, int(round(bb.x0)), int(round(bb.y0)),
                            np.ascontiguousarray(self.rgba[::-1]))
        gc.restore()
        self.stale = False


class Viewport:
    """Orthographic view matching the axes: view = (xyz - centre) @ R.T, x right, y up."""

    def __init__(self, rotation: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float],
                 width: int, height: int, zrange: Tuple[float, float]):
        self.R = np.asarray(rotation, dtype=np.float64)
        self.xlim, self.ylim = tuple(map(float, xlim)), tuple(map(float, ylim))
        self.width, self.height = int(width), int(height)
        self.zrange = tuple(map(float, zrange))
        self.sx = self.width / (self.xlim[1] - self.xlim[0])  # pixels per Å
        self.sy = self.height / (self.ylim[1] - self.ylim[0])

    def to_screen(self, xyz: np.ndarray) -> np.ndarray:
        """(n, 3) centred model coordinates -> pixel x, pixel y (row, downward), view z (Å)."""
        v = np.asarray(xyz, dtype=np.float64) @ self.R.T
        out = np.empty_like(v)
        out[:, 0] = (v[:, 0] - self.xlim[0]) * self.sx
        out[:, 1] = (self.ylim[1] - v[:, 1]) * self.sy
        out[:, 2] = v[:, 2]
        return out


class MoleculeStage:
    """Scene + renderer for one structure panel."""

    def __init__(self, panel, scene: MoleculeScene, atoms_xyz: np.ndarray, renderer):
        self.panel = panel
        self.scene = scene
        self.atoms = atoms_xyz  # [frames, atoms, 3] centred on the panel centre
        self.renderer = renderer
        self.image = PixelImage(panel.ax, self._render_size)
        panel.ax.add_artist(self.image)
        self._pos = None
        self._items: List[Colored] = []
        self.quality = "high"  # "fast" while the GUI drags

    def _items_for(self, pos: int) -> List[Colored]:
        p = self.panel
        res_rgb = p.residue_rgb(pos)
        items = self.scene.frame(self.atoms[pos], res_rgb)
        ions = p.ions_centred[pos] if p.ions_centred.size else np.zeros((0, 3))
        if len(ions):
            from matplotlib.colors import to_rgb

            n = len(ions)
            sph = Spheres(ions, np.full(n, 1.2), np.full(n, -1, np.int64), np.full(n, -1, np.int64))
            items.append(Colored(sph, np.tile(np.array(to_rgb(p.ctx.project.visual.ion_color)), (n, 1))))
        return items

    def _render_size(self, width: int, height: int) -> Optional[np.ndarray]:
        if self._pos is None:
            return None
        ax = self.panel.ax
        vp = Viewport(self.panel.view_matrix, ax.get_xlim(), ax.get_ylim(), width, height,
                      (self.panel.zmin, self.panel.zmax))
        try:
            return self.renderer.render(self._items, vp, self.panel.fog_rgb(), self.panel.fog_strength(),
                                        quality=self.quality)
        except Exception as exc:  # noqa: BLE001 - never break a video for the 3D layer
            log.warning("3D representation not drawn: %s", exc)
            return None

    def update(self, pos: int) -> None:
        """New frame (or new colours / view): re-render at the current axes size."""
        if pos != self._pos or not self._items:  # rotations reuse the frame's geometry
            self._items = self._items_for(pos)
        self._pos = pos
        bb = self.panel.ax.get_window_extent()
        w, h = int(round(bb.width)), int(round(bb.height))
        self.image.rgba = self._render_size(w, h) if w > 1 and h > 1 else None
        self.image.stale = True
