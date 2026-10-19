"""Sequence and secondary-structure panels.

Both show the residues of the structure view with the same colours, proximity glow and number
tags as the structure panel, so a residue lights up in all three at once.

* ``sequence``: one-letter sequence (all chains, or the ``residues`` option such as ``"A:1-20"``)
  wrapped into as many rows as the panel needs; more than two rows log a warning.  Meant for the
  thin full-width strip of the ``*_strip`` layouts.
* ``secondary``: arc diagram on a fixed layout (residues along a line, chains separated by a gap).
  Pairs come from the ``basepairs`` analysis, re-detected every n frames (``source = "dynamic"``),
  from its first frame (``"first"``) or from a dot-bracket string (``"dotbracket"``).  Pairs within a
  chain are arcs above the line, pairs between chains transparent yellow arcs below it.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Set, Tuple

import numpy as np
from matplotlib.collections import LineCollection

from ..selections import SelectionError
from .coloring import frame_colors
from .panels import Panel, register_panel
from .residues import chain_names, one_letter, residue_rows, residue_shading, resnames, tag_rows, warn_once

log = logging.getLogger(__name__)
NOTE_ROW = 0.8  # height of the 'choose a smaller range' line, in cells

INTER_COLOR = (1.0, 0.82, 0.0, 0.55)  # transparent yellow: pairs between chains
MAX_LETTER_ROWS = 2


def _needs_proximity(project) -> bool:
    v = project.visual
    return v.color_mode == "proximity" or v.glow_mode != "none"


def _text_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "#111111" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#ffffff"


class _ResiduePanel(Panel):
    """Shared set-up: the selected trace rows, their letters, chains and shading."""

    min_series = 0
    max_series = 0

    @classmethod
    def required_keys(cls, spec, project):
        return []

    @classmethod
    def required_analyses(cls, spec, project) -> Set[str]:
        need = {"trace"}
        if _needs_proximity(project):
            need.add("proximity")
        return need

    def _residues(self):
        ds = self.ctx.dataset
        tr = ds.results.get("trace")
        if tr is None:
            raise KeyError("trace")
        spec = str(self.options.get("residues", "") or "")
        try:
            rows = residue_rows(tr, spec)
        except SelectionError as exc:
            raise KeyError(str(exc)) from None
        self.tr = tr
        self.rows = rows
        names = resnames(tr)
        self.letters = [one_letter(names[r]) for r in rows]
        self.chains = chain_names(tr)[rows]
        self.resid = np.asarray(tr.arrays["resid"])[rows]
        self.shading = residue_shading(self.ctx, tr)
        tags = set(tag_rows(tr, self.ctx.project.visual).tolist())
        self.tagged = np.array([r in tags for r in rows], dtype=bool)
        self.range_text = spec.strip()

    def _colors(self, pos: int) -> np.ndarray:
        v = self.ctx.project.visual
        sh = self.shading
        col = frame_colors(sh.static, v.color_mode, v.base_color, v.proximity_color,
                           sh.color_i[:, pos], sh.member)
        return np.asarray(col)[self.rows]

    def _glow(self, pos: int) -> np.ndarray:
        return self.shading.glow[self.rows, pos] * self.ctx.project.visual.glow_strength

    def _set_halos(self, xy: np.ndarray, pos: int, size: float) -> None:
        intensity = self._glow(pos)
        sel = np.flatnonzero(intensity > 0.01)
        if not len(sel):
            for h in (self.halo_outer, self.halo_inner):
                h.set_offsets(np.zeros((0, 2)))
            return
        rgba = np.zeros((len(sel), 4))
        rgba[:, :3] = self.shading.glow_rgb
        rgba[:, 3] = np.clip(0.22 * intensity[sel], 0, 1)
        self.halo_outer.set_offsets(xy[sel])
        self.halo_outer.set_sizes(np.full(len(sel), size * 4.0))
        self.halo_outer.set_facecolors(rgba)
        inner = rgba.copy()
        inner[:, 3] = np.clip(0.45 * intensity[sel], 0, 1)
        self.halo_inner.set_offsets(xy[sel])
        self.halo_inner.set_sizes(np.full(len(sel), size * 2.0))
        self.halo_inner.set_facecolors(inner)

    def _bare_axes(self) -> None:
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)


@register_panel
class SequencePanel(_ResiduePanel):
    type_name = "sequence"
    label = "Sequence (primary structure)"
    description = ("One-letter sequence of the structure-view residues coloured and glowing like the "
                   "structure panel, wrapped into rows. Option 'residues' limits it, e.g. 'A:1-20, B:5-30'; "
                   "'per_row' fixes the residues per row. Made for the full-width strip layouts.")

    def setup(self):
        self._residues()
        self._bare_axes()
        self.auto_title = "Sequence" + (f" ({self.range_text})" if self.range_text else "")
        # cells: residues, with a gap (and the next chain's letter) where the chain changes
        self.cells: List[Tuple[int, str]] = []  # (index into rows or -1, chain label for gaps)
        multi = len(set(self.chains.tolist())) > 1
        for k in range(len(self.rows)):
            if multi and (k == 0 or self.chains[k] != self.chains[k - 1]):
                if k:
                    self.cells.append((-1, ""))
                self.cells.append((-2, str(self.chains[k]) or "?"))
            self.cells.append((k, ""))
        n = len(self.rows)
        self.halo_outer = self.ax.scatter([], [], s=[], linewidths=0, zorder=1)
        self.halo_inner = self.ax.scatter([], [], s=[], linewidths=0, zorder=2)
        self.boxes = self.ax.scatter(np.zeros(n), np.zeros(n), marker="s", s=10, zorder=3, linewidths=0)
        fs = self.ctx.font_size
        self.texts = [self.ax.text(0, 0, ch, ha="center", va="center", fontsize=fs, zorder=4,
                                   fontweight="bold", family="monospace") for ch in self.letters]
        self.tag_texts = {k: self.ax.text(0, 0, str(int(self.resid[k])), ha="center", va="top",
                                          color=self.theme.ink2, fontsize=fs * 0.6, zorder=4)
                          for k in np.flatnonzero(self.tagged)}
        self.chain_texts = [self.ax.text(0, 0, lab, ha="center", va="center", color=self.theme.muted,
                                         fontsize=fs * 0.8, fontstyle="italic", zorder=4)
                            for idx, lab in self.cells if idx == -2]
        self.note = self.ax.text(0.995, 0.02, "", transform=self.ax.transAxes, ha="right", va="bottom",
                                 color=self.theme.muted, fontsize=fs * 0.65, zorder=5)
        self.xy = np.zeros((n, 2))
        self.cell_size = 10.0
        self._layout(800.0, 60.0, 100.0)

    def after_layout(self):
        if not self.ok:
            return
        bbox = self.ax.get_window_extent()
        if bbox.width > 0 and bbox.height > 0:
            self._layout(bbox.width, bbox.height, self.fig.dpi, warn=True)  # the real size: warn now

    def _slots(self, per_row: int) -> Tuple[List[int], int]:
        """Grid slot of every cell (a chain label never ends a row, a gap never starts one) and the
        number of rows."""
        slots, k = [], 0
        for idx, _lab in self.cells:
            if idx == -1 and k % per_row == 0:
                slots.append(-per_row)  # dropped gap: placed off the axes
                continue
            if idx == -2 and per_row > 1 and k % per_row == per_row - 1:
                k += 1
            slots.append(k)
            k += 1
        return slots, max(1, int(math.ceil(k / per_row)))

    def _layout(self, width_px: float, height_px: float, dpi: float, warn: bool = False) -> None:
        """Choose the largest cell that fits all rows into the panel; place every artist."""
        per_row_opt = int(self.options.get("per_row", 0) or 0)
        tag_space = 0.75 if len(self.tag_texts) else 0.0
        font_px = self.ctx.font_size * dpi / 72.0
        note_px = 1.3 * self.note.get_fontsize() * dpi / 72.0  # the warning line, in pixels
        cell = font_px * 1.7
        while True:
            per_row = per_row_opt or max(1, int(width_px // cell))
            slots, n_rows = self._slots(per_row)
            note_h = max(NOTE_ROW, note_px / cell) if n_rows > MAX_LETTER_ROWS else 0.0
            if (n_rows * (1.0 + tag_space) + note_h) * cell <= height_px or cell <= 6.0:
                break
            cell *= 0.92
        if per_row_opt:
            cell = min(cell, width_px / per_row)
        self.per_row, self.n_rows = per_row, n_rows
        row_h = 1.0 + tag_space
        self.ax.set_xlim(-0.5, per_row - 0.5)
        self.ax.set_ylim(n_rows * row_h - 0.5 + note_h, -0.5)  # the note gets its own line
        pts = cell * 72.0 / dpi  # cell edge in points
        self.cell_size = (0.9 * pts) ** 2
        letter = min(self.ctx.font_size, 0.62 * pts)
        chain_k = 0
        for c, (idx, _lab) in zip(slots, self.cells):
            x, y = c % per_row, (c // per_row) * row_h
            if idx >= 0:
                self.xy[idx] = (x, y)
                self.texts[idx].set_position((x, y))
                self.texts[idx].set_fontsize(letter)
                if idx in self.tag_texts:
                    self.tag_texts[idx].set_position((x, y + 0.55))
                    self.tag_texts[idx].set_fontsize(max(letter * 0.6, 4.0))
            elif idx == -2:
                self.chain_texts[chain_k].set_position((x, y))
                self.chain_texts[chain_k].set_fontsize(letter * 0.85)
                chain_k += 1
        self.boxes.set_offsets(self.xy)
        self.boxes.set_sizes(np.full(len(self.xy), self.cell_size))
        if n_rows > MAX_LETTER_ROWS:
            msg = (f"{len(self.rows)} residues need {n_rows} rows: choose a smaller range "
                   "(option 'residues', e.g. A:1-30)")
            self.note.set_text(msg)
            if warn:
                warn_once(f"sequence panel: {msg}")
        else:
            self.note.set_text("")

    def animated_artists(self):
        return ([self.halo_outer, self.halo_inner, self.boxes] + self.texts + list(self.tag_texts.values())
                + self.chain_texts + [self.note])

    def draw_frame(self, pos):
        colors = self._colors(pos)
        self.boxes.set_facecolors(colors)
        for t, c in zip(self.texts, colors):
            t.set_color(_text_color(c))
        self._set_halos(self.xy, pos, self.cell_size)


@register_panel
class SecondaryStructurePanel(_ResiduePanel):
    type_name = "secondary"
    label = "Secondary structure (arc diagram)"
    description = ("Residues along a line with base pairs as arcs (pairs within a chain above, between "
                   "chains as transparent yellow arcs below); colours and glow follow the structure panel. "
                   "Option 'source': 'dynamic' (pairs re-detected every n frames, Analyses → base pairs), "
                   "'first' (the first frame's pairs) or 'dotbracket' with option 'dotbracket'; option "
                   "'residues' limits the residues.")

    @classmethod
    def required_analyses(cls, spec, project) -> Set[str]:
        need = super().required_analyses(spec, project)
        if (spec.options or {}).get("source", "dynamic") != "dotbracket":
            need.add("basepairs")
        return need

    def setup(self):
        self._residues()
        self._bare_axes()
        self.source = str(self.options.get("source", "dynamic"))
        self.auto_title = "Secondary structure" + (f" ({self.range_text})" if self.range_text else "")
        n = len(self.rows)
        gap = 1.5
        x = np.zeros(n)
        for k in range(1, n):
            x[k] = x[k - 1] + (gap if self.chains[k] != self.chains[k - 1] else 1.0)
        self.x = x
        self.xy = np.stack([x, np.zeros(n)], axis=1)
        self.span = max(float(x[-1] - x[0]) if n else 1.0, 1.0)
        self.row_of = {int(r): k for k, r in enumerate(self.rows)}
        self.fixed: Optional[List[Tuple[int, int, float]]] = None
        self.pm = None
        if self.source == "dotbracket":
            self.fixed = self._dotbracket_pairs(str(self.options.get("dotbracket", "")))
        else:
            ds = self.ctx.dataset
            if "basepairs/map" not in ds.pairmap_keys():
                raise KeyError("basepairs/map")
            self.pm = ds.pairmap("basepairs/map")
            if self.source == "first":
                self.fixed = self._frame_pairs(0)
        fs = self.ctx.font_size
        ps = self.ctx.project.visual.point_size
        self.marker_size = ps * 1.3
        self.ax.plot([x[0] - 0.6, x[-1] + 0.6] if n else [0, 1], [0, 0], color=self.theme.grid, lw=1.0,
                     zorder=1)
        self.halo_outer = self.ax.scatter([], [], s=[], linewidths=0, zorder=2)
        self.halo_inner = self.ax.scatter([], [], s=[], linewidths=0, zorder=3)
        self.intra = LineCollection([], colors=[self.theme.ink2], linewidths=1.4, zorder=4, capstyle="round")
        self.inter = LineCollection([], colors=[INTER_COLOR], linewidths=2.0, zorder=4, capstyle="round")
        self.ax.add_collection(self.intra)
        self.ax.add_collection(self.inter)
        self.points = self.ax.scatter(x, np.zeros(n), s=self.marker_size, zorder=5,
                                      edgecolors=self.theme.surface, linewidths=0.6)
        show_letters = n <= 90
        self.labels = []  # drawn over the arcs and halos, so animated as well
        for k in range(n):
            if show_letters:
                self.labels.append(self.ax.text(x[k], -0.06, self.letters[k], ha="center", va="top", fontsize=fs * 0.7,
                             color=self.theme.ink2, family="monospace", zorder=5))
            if self.tagged[k]:
                self.labels.append(self.ax.text(x[k], -0.16, str(int(self.resid[k])), ha="center", va="top",
                                                fontsize=fs * 0.55, color=self.theme.muted, zorder=5))
        for k in range(n):
            if k == 0 or self.chains[k] != self.chains[k - 1]:
                self.labels.append(self.ax.text(x[k] - 0.6, 0.04, str(self.chains[k]), ha="right",
                                                va="bottom", fontsize=fs * 0.75, fontstyle="italic",
                                                color=self.theme.muted, zorder=5))
        self.ax.set_xlim((x[0] if n else 0) - 1.2, (x[-1] if n else 1) + 0.8)
        self.ax.set_ylim(-1.05, 1.05)
        self._last_key = None

    # -- pairs
    def _frame_pairs(self, pos: int) -> List[Tuple[int, int, float]]:
        pm = self.pm
        pos = min(pos, pm.n_frames - 1)
        s, e = pm.offsets[pos], pm.offsets[pos + 1]
        out = []
        for i, j, v in zip(pm.i[s:e], pm.j[s:e], pm.values[s:e]):
            a, b = self.row_of.get(int(i)), self.row_of.get(int(j))
            if a is not None and b is not None and a != b:
                out.append((min(a, b), max(a, b), float(v)))
        return sorted(set(out))

    def _dotbracket_pairs(self, text: str) -> List[Tuple[int, int, float]]:
        from ..secstruct import dotbracket_to_pairs

        if not text.strip():
            raise KeyError("no dot-bracket given (option 'dotbracket')")
        try:
            pairs, n, _breaks = dotbracket_to_pairs(text)
        except ValueError as exc:
            raise KeyError(f"dot-bracket: {exc}") from None
        if n != len(self.rows):
            raise KeyError(f"the dot-bracket has {n} residues, the panel shows {len(self.rows)}")
        return [(i, j, 1.0) for i, j in pairs]

    def _arcs(self, pairs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        theta = np.linspace(0.0, np.pi, 25)
        intra, inter = [], []
        for a, b, _v in pairs:
            x0, x1 = self.x[a], self.x[b]
            r = 0.5 * (x1 - x0)
            height = max(r / (0.5 * self.span), 0.06)
            xs = 0.5 * (x0 + x1) - r * np.cos(theta)
            ys = height * np.sin(theta) * 0.95
            if self.chains[a] != self.chains[b]:
                inter.append(np.stack([xs, -ys - 0.02], axis=1))
            else:
                intra.append(np.stack([xs, ys + 0.02], axis=1))
        return intra, inter

    def animated_artists(self):
        return [self.halo_outer, self.halo_inner, self.intra, self.inter, self.points] + self.labels

    def draw_frame(self, pos):
        pairs = self.fixed if self.fixed is not None else self._frame_pairs(pos)
        key = tuple(pairs)
        if key != self._last_key:  # pairs change only every n frames
            intra, inter = self._arcs(pairs)
            self.intra.set_segments(intra)
            self.inter.set_segments(inter)
            self._last_key = key
        self.points.set_facecolors(self._colors(pos))
        self._set_halos(self.xy, pos, self.marker_size * 2.2)
