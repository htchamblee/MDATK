"""Panel types for the multi-panel figure.

Every panel is created once per render and then updated in place for each
video frame, so per-frame cost is limited to moving artists.  New panel types
register themselves with :func:`register_panel` and declare the data keys they
need, which lets the pipeline compute exactly the analyses a layout requires.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Type

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..config import PanelSpec, Project
from ..data import Dataset
from .coloring import frame_colors, glow_intensity, static_colors
from .orientation import euler_to_matrix
from .theme import Theme, get_theme, style_axes, style_legend

log = logging.getLogger(__name__)

PANEL_TYPES: Dict[str, Type["Panel"]] = {}


def register_panel(cls: Type["Panel"]) -> Type["Panel"]:
    PANEL_TYPES[cls.type_name] = cls
    return cls


class RenderContext:
    def __init__(self, dataset: Dataset, project: Project):
        self.dataset = dataset
        self.project = project
        self.theme: Theme = get_theme(project.visual.theme)
        unit = project.inputs.time_unit
        self.time = dataset.time(unit)
        self.time_unit = unit
        self.time_label = dataset.time_label(unit)
        self.font_size = project.visual.font_size

    def format_time(self, pos: int) -> str:
        frame = int(self.dataset.frames[pos])
        if self.time_unit == "frame":
            return f"frame {frame}"
        return f"t = {self.time[pos]:.3f} {self.time_unit.replace('us', chr(0xb5) + 's')}  (frame {frame})"


def integer_axis(axis) -> None:
    """Whole-number ticks only (for counts)."""
    axis.set_major_locator(MaxNLocator(integer=True))


def thin_ticks(n: int, max_ticks: int = 20) -> np.ndarray:
    step = max(1, int(np.ceil(n / max_ticks)))
    return np.arange(0, n, step)


class Panel:
    type_name = ""
    label = ""
    description = ""
    #: which dataset kind ``spec.series`` refers to: "series", "matrices" or "pairmaps"
    data_kind = "series"
    min_series = 1
    max_series = 8

    @classmethod
    def required_keys(cls, spec: PanelSpec, project: Project) -> List[str]:
        return list(spec.series)

    @classmethod
    def required_analyses(cls, spec: PanelSpec, project: Project) -> Set[str]:
        return {k.split("/", 1)[0] for k in cls.required_keys(spec, project) if "/" in k}

    def __init__(self, fig, subplotspec, spec: PanelSpec, ctx: RenderContext):
        self.fig = fig
        self.spec = spec
        self.ctx = ctx
        self.theme = ctx.theme
        self.ax = fig.add_subplot(subplotspec)
        style_axes(self.ax, self.theme, font_size=ctx.font_size)
        self.options = dict(spec.options or {})
        self.ok = True
        self.auto_title = ""
        try:
            self.setup()
        except KeyError as exc:
            self.show_message(f"No data: {exc.args[0] if exc.args else exc}")
        title = spec.title or self.auto_title  # default: the label of the data shown
        if title:
            self.ax.set_title(title, color=self.theme.ink, fontsize=ctx.font_size, loc="left")

    def setup(self) -> None:
        pass

    def show_message(self, text: str) -> None:
        self.ok = False
        self.ax.cla()
        self.ax.set_facecolor(self.theme.surface)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(self.theme.grid)
        self.ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                     color=self.theme.muted, fontsize=self.ctx.font_size * 0.85,
                     transform=self.ax.transAxes)

    def after_layout(self) -> None:
        """Called once the figure layout is final (axes sizes are known)."""

    def animated_artists(self) -> list:
        """Artists that change between frames (redrawn by the blitting video loop)."""
        return []

    def update(self, pos: int) -> None:
        if self.ok:
            self.draw_frame(pos)

    def draw_frame(self, pos: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Structural view
# ---------------------------------------------------------------------------


@register_panel
class StructurePanel(Panel):
    type_name = "structure"
    label = "Structural view (2D backbone projection)"
    description = "Projected C4'/P trace with colouring, proximity glow and ions."
    min_series = 0
    max_series = 0

    @classmethod
    def required_keys(cls, spec, project):
        return []

    @classmethod
    def required_analyses(cls, spec, project):
        need = {"trace"}
        v = project.visual
        if v.color_mode == "proximity" or v.glow_mode != "none":
            need.add("proximity")
        return need

    def setup(self):
        ds = self.ctx.dataset
        v = self.ctx.project.visual
        tr = ds.results.get("trace")
        if tr is None:
            raise KeyError("trace")
        coords = np.asarray(tr.arrays["coords"], dtype=np.float64)
        ions = np.asarray(tr.arrays["ion_coords"], dtype=np.float64)
        center = coords[0].mean(axis=0)
        self.center = center
        # centred coordinates; the view is a rotation of these (see set_rotation)
        self.centred = coords - center
        self.ions_centred = ions - center if ions.size else np.zeros((len(coords), 0, 3))
        self.links = np.asarray(tr.arrays["links"], dtype=np.int64).reshape(-1, 2)
        n_res = coords.shape[1]
        # Frame the model by its bounding sphere so any orientation fits without zooming.
        radius = float(np.linalg.norm(self.centred, axis=-1).max())
        if self.ions_centred.size:
            radius = max(radius, float(np.linalg.norm(self.ions_centred, axis=-1).max()))
        radius = max(radius, 1.0)
        self.radius = radius
        pad = 0.06 * radius + 2.0
        self.data_lo = np.array([-radius - pad, -radius - pad])
        self.data_hi = np.array([radius + pad, radius + pad])
        self.ax.set_xlim(self.data_lo[0], self.data_hi[0])
        self.ax.set_ylim(self.data_lo[1], self.data_hi[1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.zmin, self.zmax = -radius, radius
        self.density_img = None
        self.density_fast = False  # coarser shadow while the GUI drags the model or the map
        self.density_note = ""
        self._setup_density(tr)
        self.set_rotation(v.rotation)
        self.auto_title = "Structure"

        project = self.ctx.project
        group_colors = [g.color for g in project.groups]
        self.static = static_colors(v.color_mode, np.asarray(tr.arrays["chain"]),
                                    np.asarray(tr.arrays["group_index"]), group_colors,
                                    v.base_color, v.chain_palette)
        n_frames = len(coords)
        self.member = np.zeros(n_res, bool)
        self.glow = np.zeros((n_res, n_frames), np.float32)
        self.color_i = np.zeros((n_res, n_frames), np.float32)
        prox = ds.results.get("proximity")
        need_prox = v.color_mode == "proximity" or v.glow_mode != "none"
        if need_prox:
            if prox is None:
                raise KeyError("proximity")
            dist = np.asarray(prox.arrays["min_distance"], dtype=np.float64)
            member = np.asarray(prox.arrays["member"], bool)
            if not np.array_equal(prox.arrays["resindex"], tr.arrays["resindex"]):
                order = {int(ix): k for k, ix in enumerate(prox.arrays["resindex"])}
                rows = [order.get(int(ix), -1) for ix in tr.arrays["resindex"]]
                dist = np.array([dist[k] if k >= 0 else np.full(n_frames, np.inf) for k in rows])
                member = np.array([member[k] if k >= 0 else False for k in rows])
            self.member = member
            cutoff = v.proximity_cutoff
            self.glow = glow_intensity(dist, member, cutoff, v.glow_mode, v.fade_frames)
            mode = v.glow_mode if v.glow_mode != "none" else "current"
            self.color_i = glow_intensity(dist, member, cutoff, mode, v.fade_frames)

        ps = v.point_size
        glow_rgb = np.array(to_rgba(v.glow_color))[:3]
        self.glow_rgb = glow_rgb
        self.halo_outer = self.ax.scatter([], [], s=[], linewidths=0, zorder=1)
        self.halo_inner = self.ax.scatter([], [], s=[], linewidths=0, zorder=2)
        self.lines = LineCollection([], capstyle="round", zorder=3)
        self.ax.add_collection(self.lines)
        self.points = self.ax.scatter([], [], s=[], zorder=4, edgecolors=self.theme.surface,
                                      linewidths=0.6)
        self.ion_points = self.ax.scatter([], [], s=ps * 2.2, zorder=5, color=v.ion_color,
                                          edgecolors=self.theme.surface, linewidths=1.0)
        self.time_text = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes, va="top",
                                      ha="left", color=self.theme.ink2,
                                      fontsize=self.ctx.font_size * 0.85) if v.show_time_label else None
        self._residue_tags(v, tr)
        self._legend(v, tr)

    def set_rotation(self, angles) -> None:
        """Re-orient the model (degrees about X, Y, Z); used by interactive previews."""
        r = euler_to_matrix(angles)
        self.view_matrix = r
        self.view = self.centred @ r.T
        self.ion_view = self.ions_centred @ r.T
        self.update_density()

    # -- density map shadow
    def _setup_density(self, tr) -> None:
        """Load the density map when a placement is known (drawn by update_density)."""
        self._dmap = self._to_map = None
        project = self.ctx.project
        d = project.density
        if not project.density_active():
            return
        if tr.meta.get("schematic"):
            self.density_note = "density map: load the real reference frame to see it"
            return
        if len(d.transform) != 12:
            self.density_note = "density map: not fitted yet"
            return
        from ..density import RigidTransform, load_map_cached, map_level

        try:
            self._dmap = load_map_cached(d.map_path)
            self._to_map = RigidTransform.from_list(d.transform)
        except (OSError, ValueError) as exc:
            self._dmap = None
            self.density_note = f"density map: {exc}"
            log.warning("density map not drawn: %s", exc)
            return
        self._level = map_level(self._dmap, d.level)
        self._density_rgb = np.array(to_rgba(d.color))[:3]
        self.density_img = self.ax.imshow(np.zeros((2, 2, 4), np.float32), extent=(0, 1, 0, 1),
                                          origin="lower", interpolation="bilinear", zorder=0.5,
                                          aspect="auto")
        self.ax.set_xlim(self.data_lo[0], self.data_hi[0])
        self.ax.set_ylim(self.data_lo[1], self.data_hi[1])

    def set_density_transform(self, values) -> None:
        """New map placement (12 numbers, model -> map); used while dragging the map."""
        from ..density import RigidTransform

        if self._dmap is None:
            return
        self._to_map = RigidTransform.from_list(values)
        self.update_density()

    def update_density(self) -> None:
        if self.density_img is None or self._dmap is None:
            return
        from ..density import project_density, shadow_rgba

        span = 2.5 * float(max(self.data_hi - self.data_lo))
        window = (-span / 2, span / 2, -span / 2, span / 2)
        n = 110 if self.density_fast else 240
        sh = project_density(self._dmap, self._to_map, self.view_matrix, self.center, self._level,
                             window=window, max_pixels=n, max_depth=n)
        if sh is None:
            self.density_img.set_visible(False)
            return
        d = self.ctx.project.density
        self.density_img.set_data(shadow_rgba(sh, self._density_rgb, d.opacity))
        self.density_img.set_extent(sh.extent)
        self.density_img.set_visible(True)

    def has_density(self) -> bool:
        return self._dmap is not None and self.density_img is not None

    def nudge_density(self, shift=(0.0, 0.0), view_rotation=None):
        """Move the drawn map by ``shift`` (Å along the screen x/y axes) or rotate it
        about its density centre by ``view_rotation`` (3x3, screen axes).  Returns the
        new placement (12 numbers, model -> map) or None without a map."""
        from ..density import RigidTransform, density_centroid

        if not self.has_density():
            return None
        v = self.view_matrix
        t = self._to_map
        if view_rotation is not None:
            if getattr(self, "_density_centre", None) is None:
                self._density_centre = density_centroid(self._dmap, self._level)
            c = t.inverse().apply(self._density_centre[None, :])[0]  # map centre, model frame
            q = v.T @ np.asarray(view_rotation, dtype=np.float64) @ v
            move = RigidTransform(q, c - q @ c)
        else:
            move = RigidTransform(np.eye(3), v.T @ np.array([shift[0], shift[1], 0.0]))
        self._to_map = t.compose(move.inverse())
        self.update_density()
        return self._to_map.to_list()

    def rotation_artists(self) -> list:
        """Static artists that still change when the view rotates (interactive previews)."""
        return [self.density_img] if self.density_img is not None else []

    def _residue_tags(self, v, tr):
        """Residue-number labels: every N-th residue and/or an explicit list."""
        resid = np.asarray(tr.arrays["resid"]) if "resid" in tr.arrays else None
        self.tag_rows = np.zeros(0, dtype=np.int64)
        self.tags = []
        if resid is None or not (v.label_every > 0 or v.label_residues.strip()):
            return
        wanted = np.zeros(len(resid), bool)
        if v.label_every > 0:
            wanted |= (resid % int(v.label_every)) == 0
        if v.label_residues.strip():
            from ..selections import parse_residue_ranges

            for _chain, a, b in parse_residue_ranges(v.label_residues):
                wanted |= (resid >= a) & (resid <= b)
        self.tag_rows = np.flatnonzero(wanted)
        self.tags = [self.ax.text(0, 0, str(int(resid[k])), ha="center", va="center",
                                  color=self.theme.ink2, fontsize=self.ctx.font_size * 0.7,
                                  zorder=6, clip_on=True) for k in self.tag_rows]

    def after_layout(self):
        """Equal x/y scale: grow the data window to the panel's pixel aspect ratio."""
        if not self.ok:
            return
        bbox = self.ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        lo, hi = self.data_lo, self.data_hi
        span = hi - lo
        mid = 0.5 * (lo + hi)
        box_aspect = bbox.height / bbox.width
        if span[1] / span[0] < box_aspect:
            span[1] = span[0] * box_aspect
        else:
            span[0] = span[1] / box_aspect
        self.ax.set_xlim(mid[0] - span[0] / 2, mid[0] + span[0] / 2)
        self.ax.set_ylim(mid[1] - span[1] / 2, mid[1] + span[1] / 2)

    def _legend(self, v, tr):
        project = self.ctx.project
        handles = []
        ms = 7
        if v.color_mode == "chain":
            chains = np.unique(tr.arrays["chain"])
            for c in chains:
                k = int(np.flatnonzero(np.asarray(tr.arrays["chain"]) == c)[0])
                handles.append(Line2D([], [], ls="", marker="o", ms=ms, color=self.static[k],
                                      label=f"Chain {int(c) + 1}"))
        else:
            gi = np.asarray(tr.arrays["group_index"])
            for k, g in enumerate(project.groups):
                if np.any(gi == k):
                    handles.append(Line2D([], [], ls="", marker="o", ms=ms, color=g.color, label=g.name))
            if v.color_mode == "proximity":
                handles.append(Line2D([], [], ls="", marker="o", ms=ms, color=v.proximity_color,
                                      label=f"≤ {v.proximity_cutoff:g} Å of {v.proximity_set}"))
        if v.glow_mode != "none":
            handles.append(Line2D([], [], ls="", marker="o", ms=ms * 1.6, color=v.glow_color, alpha=0.5,
                                  label=f"Glow ({v.glow_mode})"))
        if self.ion_view.shape[1]:
            handles.append(Line2D([], [], ls="", marker="o", ms=ms, color=v.ion_color, label="Mg²⁺"))
        self.legend = None
        if handles:
            self.legend = self.ax.legend(handles=handles, loc="lower right", frameon=True)
            style_legend(self.legend, self.theme, self.ctx.font_size)

    def animated_artists(self):
        arts = [self.halo_outer, self.halo_inner, self.lines, self.points, self.ion_points]
        arts += self.tags
        if self.time_text is not None:
            arts.append(self.time_text)
        if self.legend is not None:  # keep the legend above moving residues
            arts.append(self.legend)
        return arts

    def draw_frame(self, pos):
        v = self.ctx.project.visual
        xyz = self.view[pos]
        span = max(self.zmax - self.zmin, 1e-6)
        depth = (xyz[:, 2] - self.zmin) / span if v.depth_cue else np.ones(len(xyz))
        colors = frame_colors(self.static, v.color_mode, v.base_color, v.proximity_color,
                              self.color_i[:, pos], self.member).copy()
        if v.depth_cue:  # fog toward the surface for distant residues
            surf = np.array(to_rgba(self.theme.surface))
            fog = (0.45 * (1.0 - depth))[:, None]
            colors[:, :3] = colors[:, :3] * (1 - fog) + surf[:3] * fog
        ps = v.point_size
        sizes = ps * (0.55 + 0.9 * depth)
        order = np.argsort(xyz[:, 2], kind="stable")
        self.points.set_offsets(xyz[order, :2])
        self.points.set_sizes(sizes[order])
        self.points.set_facecolors(colors[order])
        if len(self.links):
            i, j = self.links[:, 0], self.links[:, 1]
            segs = np.stack([xyz[i, :2], xyz[j, :2]], axis=1)
            seg_depth = 0.5 * (depth[i] + depth[j])
            seg_col = 0.5 * (colors[i] + colors[j])
            so = np.argsort(seg_depth, kind="stable")
            self.lines.set_segments(segs[so])
            self.lines.set_color(seg_col[so])
            self.lines.set_linewidths(v.line_width * (0.5 + 0.9 * seg_depth[so]))
        intensity = self.glow[:, pos] * v.glow_strength
        sel = np.flatnonzero(intensity > 0.01)
        if len(sel):
            rgba_o = np.zeros((len(sel), 4))
            rgba_o[:, :3] = self.glow_rgb
            rgba_o[:, 3] = np.clip(0.20 * intensity[sel], 0, 1)
            rgba_i = rgba_o.copy()
            rgba_i[:, 3] = np.clip(0.45 * intensity[sel], 0, 1)
            self.halo_outer.set_offsets(xyz[sel, :2])
            self.halo_outer.set_sizes(np.full(len(sel), ps * 9.0))
            self.halo_outer.set_facecolors(rgba_o)
            self.halo_inner.set_offsets(xyz[sel, :2])
            self.halo_inner.set_sizes(np.full(len(sel), ps * 3.8))
            self.halo_inner.set_facecolors(rgba_i)
        else:
            self.halo_outer.set_offsets(np.zeros((0, 2)))
            self.halo_inner.set_offsets(np.zeros((0, 2)))
        if self.ion_view.shape[1]:
            self.ion_points.set_offsets(self.ion_view[pos][:, :2])
        if self.tags:  # push each label outward, away from the projected centre
            xy = xyz[self.tag_rows, :2]
            centre = xyz[:, :2].mean(axis=0)
            out = xy - centre
            norm = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.where(norm > 1e-6, out / np.maximum(norm, 1e-6), np.array([[0.0, 1.0]]))
            pos_xy = xy + out * (0.07 * self.radius + 1.5)
            for text, (x, y) in zip(self.tags, pos_xy):
                text.set_position((x, y))
        if self.time_text is not None:
            self.time_text.set_text(self.ctx.format_time(pos))


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def _running_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 2:
        return y
    window = min(window, len(y))
    kernel = np.ones(window) / window
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    padded = np.pad(y, (pad_l, pad_r), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


@register_panel
class TimeSeriesPanel(Panel):
    type_name = "timeseries"
    label = "Time series"
    description = "One or more per-frame series against time with a moving time cursor."
    max_series = 8

    def setup(self):
        ds = self.ctx.dataset
        t = self.ctx.time
        keys = list(self.spec.series)
        if not keys:
            raise KeyError("no series selected")
        self.series = [ds.series(k) for k in keys]
        self.progressive = bool(self.options.get("progressive", False))
        smooth = int(self.options.get("smooth", 0) or 0)
        self.lines = []
        self.dots = []
        self.values = []
        for k, s in enumerate(self.series):
            color = self.theme.series_color(k)
            y = _running_mean(s.values, smooth) if smooth > 1 else s.values
            self.values.append(y)
            (line,) = self.ax.plot(t, y, color=color, lw=1.6, label=s.label, zorder=3)
            (dot,) = self.ax.plot([t[0]], [y[0]], "o", ms=6, color=color,
                                  mec=self.theme.surface, mew=1.5, zorder=5)
            self.lines.append(line)
            self.dots.append(dot)
        target = self.options.get("target", None)
        if target is None:
            target = next((s.target for s in self.series if s.target is not None), None)
        if target is not None:
            self.ax.axhline(float(target), color=self.theme.ink2, lw=1.0, ls=(0, (4, 3)),
                            label=f"Target ({float(target):g})", zorder=2)
        self.cursor = self.ax.axvline(t[0], color=self.theme.ink, lw=1.0, alpha=0.7, zorder=4)
        self.ax.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 1)
        if "ylim" in self.options:
            self.ax.set_ylim(*self.options["ylim"])
        self.ax.set_xlabel(self.ctx.time_label)
        units = {s.unit for s in self.series}
        if units == {"count"}:
            integer_axis(self.ax.yaxis)
        self.auto_title = self.series[0].label if len(self.series) == 1 else ""
        if len(self.series) == 1:
            self.ax.set_ylabel(self.series[0].axis_label)
        elif len(units) == 1 and next(iter(units)) not in ("", "count"):
            self.ax.set_ylabel(next(iter(units)))
        if len(self.series) > 1 or target is not None:
            leg = self.ax.legend(loc="best")
            style_legend(leg, self.theme, self.ctx.font_size)

    def animated_artists(self):
        return ([self.cursor] + (list(self.lines) if self.progressive else []) + list(self.dots))

    def draw_frame(self, pos):
        t = self.ctx.time
        self.cursor.set_xdata([t[pos], t[pos]])
        for line, dot, y in zip(self.lines, self.dots, self.values):
            dot.set_data([t[pos]], [y[pos]])
            if self.progressive:
                line.set_data(t[: pos + 1], y[: pos + 1])


@register_panel
class DualAxisPanel(Panel):
    type_name = "dual_axis"
    label = "Dual-axis time series"
    description = "Two series with independent y axes (left / right)."
    min_series = 2
    max_series = 2

    def setup(self):
        ds = self.ctx.dataset
        t = self.ctx.time
        if len(self.spec.series) < 2:
            raise KeyError("dual-axis panel needs two series")
        s1, s2 = ds.series(self.spec.series[0]), ds.series(self.spec.series[1])
        self.series = [s1, s2]
        c1, c2 = self.theme.series_color(0), self.theme.series_color(1)
        ax2 = self.ax.twinx()
        style_axes(ax2, self.theme, grid=False, font_size=self.ctx.font_size)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(self.theme.axis)
        self.ax2 = ax2
        (l1,) = self.ax.plot(t, s1.values, color=c1, lw=1.6, label=f"{s1.label} (left axis)")
        (l2,) = ax2.plot(t, s2.values, color=c2, lw=1.6, label=f"{s2.label} (right axis)")
        (self.d1,) = self.ax.plot([t[0]], [s1.values[0]], "o", ms=6, color=c1,
                                  mec=self.theme.surface, mew=1.5, zorder=5)
        (self.d2,) = ax2.plot([t[0]], [s2.values[0]], "o", ms=6, color=c2,
                              mec=self.theme.surface, mew=1.5, zorder=5)
        self.ax.set_ylabel(s1.axis_label)
        ax2.set_ylabel(s2.axis_label)
        if s1.unit == "count":
            integer_axis(self.ax.yaxis)
        if s2.unit == "count":
            integer_axis(ax2.yaxis)
        self.auto_title = f"{s1.label} / {s2.label}"
        self.ax.set_xlabel(self.ctx.time_label)
        self.ax.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 1)
        self.cursor = self.ax.axvline(t[0], color=self.theme.ink, lw=1.0, alpha=0.7)
        leg = self.ax.legend(handles=[l1, l2], loc="best")
        style_legend(leg, self.theme, self.ctx.font_size)

    def animated_artists(self):
        return [self.cursor, self.d1, self.d2]

    def draw_frame(self, pos):
        t = self.ctx.time
        self.cursor.set_xdata([t[pos], t[pos]])
        self.d1.set_data([t[pos]], [self.series[0].values[pos]])
        self.d2.set_data([t[pos]], [self.series[1].values[pos]])


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def _set_residue_ticks(axis_set_ticks, labels: Sequence[str], max_ticks: int = 20):
    idx = thin_ticks(len(labels), max_ticks)
    axis_set_ticks(idx, [labels[i] for i in idx])


@register_panel
class HeatmapPanel(Panel):
    type_name = "heatmap"
    label = "Residue heatmap over time"
    description = "Per-residue values (contacts, H-bonds, Mg occupancy...) versus time."
    data_kind = "matrices"
    max_series = 1

    def setup(self):
        ds = self.ctx.dataset
        if not self.spec.series:
            raise KeyError("no matrix selected")
        m = ds.matrix(self.spec.series[0])
        if m.values.size == 0:
            raise KeyError(f"{m.key} is empty")
        self.auto_title = m.label
        t = self.ctx.time
        dt = (t[1] - t[0]) if len(t) > 1 else 1.0
        values = np.ma.masked_invalid(np.asarray(m.values, dtype=np.float64))
        extent = [t[0] - dt / 2, t[-1] + dt / 2, len(m.row_labels) - 0.5, -0.5]
        if m.kind == "class" and m.classes:
            n = len(m.classes)
            cmap = self.theme.ordinal(n)
            im = self.ax.imshow(values, aspect="auto", interpolation="nearest", extent=extent,
                                cmap=cmap, vmin=-0.5, vmax=n - 0.5)
            cb = self.fig.colorbar(im, ax=self.ax, fraction=0.04, pad=0.02, ticks=range(n))
            cb.ax.set_yticklabels(m.classes)
        else:
            cmap = self.theme.sequential().with_extremes(bad=self.theme.grid)
            vmax = float(self.options.get("vmax", 0) or (values.max() if values.count() else 1.0))
            im = self.ax.imshow(values, aspect="auto", interpolation="nearest", extent=extent,
                                cmap=cmap, vmin=float(self.options.get("vmin", 0.0)),
                                vmax=max(vmax, 1e-9))
            cb = self.fig.colorbar(im, ax=self.ax, fraction=0.04, pad=0.02)
            cb.set_label(m.unit or "count", color=self.theme.ink2, fontsize=self.ctx.font_size * 0.8)
            if m.kind == "count":
                integer_axis(cb.ax.yaxis)
        cb.outline.set_edgecolor(self.theme.grid)
        cb.ax.tick_params(colors=self.theme.muted, labelcolor=self.theme.ink2,
                          labelsize=self.ctx.font_size * 0.75)
        self.ax.grid(False)
        _set_residue_ticks(self.ax.set_yticks, m.row_labels, int(self.options.get("max_ticks", 16)))
        self.ax.tick_params(axis="y", labelsize=self.ctx.font_size * 0.7)
        self.ax.set_xlabel(self.ctx.time_label)
        self.cursor = self.ax.axvline(t[0], color=self.theme.ink, lw=1.2, alpha=0.8)

    def animated_artists(self):
        return [self.cursor]

    def draw_frame(self, pos):
        t = self.ctx.time
        self.cursor.set_xdata([t[pos], t[pos]])


@register_panel
class ContactMapPanel(Panel):
    type_name = "contact_map"
    label = "Residue contact map"
    description = ("Residue x residue map: 'instant' (evolving with the movie), 'frame' "
                   "(fixed frame) or 'frequency' (fraction of frames).")
    data_kind = "pairmaps"
    max_series = 1

    def setup(self):
        ds = self.ctx.dataset
        if not self.spec.series:
            raise KeyError("no map selected")
        self.pm = ds.pairmap(self.spec.series[0])
        if 0 in self.pm.shape:
            raise KeyError(f"{self.pm.key} is empty")
        self.mode = self.options.get("mode", "instant")
        cmap = self.theme.sequential()
        if self.mode == "frequency":
            data, vmax, label = self.pm.frequency(), 1.0, "fraction of frames"
        elif self.mode == "frame":
            f = int(self.options.get("frame", 0))
            f = min(max(f, 0), self.pm.n_frames - 1)
            data, label = self.pm.dense(f), "count"
            vmax = max(1.0, float(data.max()))
        else:
            data, label = self.pm.dense(0), "count"
            vmax = max(1.0, float(self.pm.values.max()) if len(self.pm.values) else 1.0)
        self.im = self.ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap,
                                 vmin=0.0, vmax=vmax, origin="upper")
        cb = self.fig.colorbar(self.im, ax=self.ax, fraction=0.04, pad=0.02)
        cb.set_label(label, color=self.theme.ink2, fontsize=self.ctx.font_size * 0.8)
        if label == "count":
            integer_axis(cb.ax.yaxis)
        self.auto_title = self.pm.label
        cb.outline.set_edgecolor(self.theme.grid)
        cb.ax.tick_params(colors=self.theme.muted, labelcolor=self.theme.ink2,
                          labelsize=self.ctx.font_size * 0.75)
        self.ax.grid(False)
        mt = int(self.options.get("max_ticks", 14))
        _set_residue_ticks(self.ax.set_yticks, self.pm.row_labels, mt)
        _set_residue_ticks(self.ax.set_xticks, self.pm.col_labels, mt)
        self.ax.tick_params(axis="both", labelsize=self.ctx.font_size * 0.65)
        for lab in self.ax.get_xticklabels():
            lab.set_rotation(90)

    def animated_artists(self):
        return [self.im] if self.mode == "instant" else []

    def draw_frame(self, pos):
        if self.mode == "instant":
            self.im.set_data(self.pm.dense(pos))


def panel_class(type_name: str) -> Type[Panel]:
    if type_name not in PANEL_TYPES:
        raise ValueError(f"unknown panel type {type_name!r} (choose from {', '.join(PANEL_TYPES)})")
    return PANEL_TYPES[type_name]


def required_analyses(project: Project, specs: Optional[Sequence[PanelSpec]] = None) -> Set[str]:
    need: Set[str] = set()
    for spec in (project.layout.panels if specs is None else specs):
        need |= panel_class(spec.type).required_analyses(spec, project)
    return need
