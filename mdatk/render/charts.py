"""Standalone high-resolution time charts (PNG / SVG / PDF)."""

from __future__ import annotations

import contextlib
import logging
import os
import re
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..config import Project
from ..data import Dataset
from ..progress import Reporter
from .panels import _running_mean, thin_ticks
from .theme import LIGHT, get_theme, style_axes, style_legend

log = logging.getLogger(__name__)

# Internal analyses feed panels, not charts.
_INTERNAL = ("trace/", "proximity/")


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+\-]+", "_", key).strip("_")


def chart_keys(dataset: Dataset, project: Project) -> List[str]:
    requested = project.output.charts.series
    if requested:
        missing = [k for k in requested if not dataset.has(k)]
        for k in missing:
            log.warning("chart series %s is not available", k)
        return [k for k in requested if dataset.has(k) and dataset.kind_of(k) == "series"]
    return [k for k in dataset.series_keys() if not k.startswith(_INTERNAL)]


@contextlib.contextmanager
def _style(name: str):
    """Apply a matplotlib style; yields True when MDATK's own chrome should be used."""
    if not name or name == "mdatk":
        yield True
        return
    import matplotlib.style as mstyle

    ctx = mstyle.context(name)
    try:
        ctx.__enter__()
    except (OSError, ValueError) as exc:
        log.warning("unknown matplotlib style %r (%s); using MDATK chrome", name, exc)
        yield True
        return
    try:
        yield False
    finally:
        ctx.__exit__(None, None, None)


def _format_title(template: str, label: str, key: str) -> str:
    if not template:
        return label
    return template.replace("{label}", label).replace("{key}", key)


def render_series_chart(dataset: Dataset, project: Project, key: str, out_dir: str) -> List[str]:
    cp = project.output.charts
    s = dataset.series(key)
    t = dataset.time(project.inputs.time_unit)
    paths = []
    with _style(cp.style) as mdatk_chrome:
        theme = get_theme(project.visual.theme) if mdatk_chrome else LIGHT
        fig = Figure(figsize=(cp.width, cp.height), dpi=cp.dpi,
                     facecolor=theme.surface if mdatk_chrome else None)
        FigureCanvasAgg(fig)
        fig.set_layout_engine("constrained")
        ax = fig.add_subplot(1, 1, 1)
        if mdatk_chrome:
            style_axes(ax, theme, font_size=11)
        color = cp.color or (theme.series_color(0) if mdatk_chrome else None)
        smooth = int(cp.smoothing or 0)
        if smooth > 1:
            ax.plot(t, s.values, color=color, lw=cp.line_width * 0.6, alpha=0.35, label="Per frame")
            ax.plot(t, _running_mean(s.values, smooth), color=color, lw=cp.line_width,
                    label=f"Running mean ({smooth} frames)")
        else:
            ax.plot(t, s.values, color=color, lw=cp.line_width, label=s.label)
        handles = smooth > 1
        if cp.show_target and s.target is not None:
            ax.axhline(s.target, color=theme.ink2 if mdatk_chrome else "0.3", lw=1.0,
                       ls=(0, (4, 3)), label=f"Target ({s.target:g})")
            handles = True
        if handles:
            leg = ax.legend(loc="best")
            if mdatk_chrome:
                style_legend(leg, theme, 11)
        ax.set_xlabel(cp.xlabel or dataset.time_label(project.inputs.time_unit))
        ax.set_ylabel(cp.ylabel or s.axis_label)
        ax.set_title(_format_title(cp.title, s.label, key), loc="left")
        if len(t) > 1:
            ax.set_xlim(t[0], t[-1])
        base = os.path.join(out_dir, f"{cp.prefix}_{_safe(key)}")
        for fmt in cp.formats or ["png"]:
            path = f"{base}.{fmt.lower().lstrip('.')}"
            fig.savefig(path, dpi=cp.dpi, facecolor=fig.get_facecolor())
            paths.append(path)
    return paths


def render_matrix_chart(dataset: Dataset, project: Project, key: str, out_dir: str) -> List[str]:
    cp = project.output.charts
    m = dataset.matrix(key)
    t = dataset.time(project.inputs.time_unit)
    paths = []
    with _style(cp.style) as mdatk_chrome:
        theme = get_theme(project.visual.theme) if mdatk_chrome else LIGHT
        height = max(cp.height, 0.18 * len(m.row_labels) + 1.5)
        fig = Figure(figsize=(cp.width, height), dpi=cp.dpi,
                     facecolor=theme.surface if mdatk_chrome else None)
        FigureCanvasAgg(fig)
        fig.set_layout_engine("constrained")
        ax = fig.add_subplot(1, 1, 1)
        if mdatk_chrome:
            style_axes(ax, theme, grid=False, font_size=11)
        dt = (t[1] - t[0]) if len(t) > 1 else 1.0
        extent = [t[0] - dt / 2, t[-1] + dt / 2, len(m.row_labels) - 0.5, -0.5]
        values = np.ma.masked_invalid(np.asarray(m.values, dtype=np.float64))
        if m.kind == "class" and m.classes:
            n = len(m.classes)
            im = ax.imshow(values, aspect="auto", interpolation="nearest", extent=extent,
                           cmap=theme.ordinal(n), vmin=-0.5, vmax=n - 0.5)
            cb = fig.colorbar(im, ax=ax, ticks=range(n))
            cb.ax.set_yticklabels(m.classes)
        else:
            cmap = theme.sequential().with_extremes(bad=theme.grid)
            im = ax.imshow(values, aspect="auto", interpolation="nearest", extent=extent, cmap=cmap,
                           vmin=0, vmax=max(float(values.max()) if values.count() else 1.0, 1e-9))
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(m.unit or "count")
        idx = thin_ticks(len(m.row_labels), 40)
        ax.set_yticks(idx, [m.row_labels[i] for i in idx], fontsize=8)
        ax.set_xlabel(cp.xlabel or dataset.time_label(project.inputs.time_unit))
        ax.set_ylabel("Residue")
        ax.set_title(_format_title(cp.title, m.label, key), loc="left")
        base = os.path.join(out_dir, f"{cp.prefix}_{_safe(key)}")
        for fmt in cp.formats or ["png"]:
            path = f"{base}.{fmt.lower().lstrip('.')}"
            fig.savefig(path, dpi=cp.dpi, facecolor=fig.get_facecolor())
            paths.append(path)
    return paths


def render_charts(dataset: Dataset, project: Project, out_dir: str,
                  reporter: Optional[Reporter] = None) -> List[str]:
    reporter = reporter or Reporter()
    os.makedirs(out_dir, exist_ok=True)
    keys = chart_keys(dataset, project)
    mats = ([k for k in dataset.matrix_keys() if not k.startswith(_INTERNAL)]
            if project.output.charts.include_heatmaps else [])
    jobs = [("series", k) for k in keys] + [("matrix", k) for k in mats]
    paths: List[str] = []
    for n, (kind, key) in enumerate(jobs):
        reporter.check_cancel()
        if kind == "series":
            paths += render_series_chart(dataset, project, key, out_dir)
        else:
            paths += render_matrix_chart(dataset, project, key, out_dir)
        reporter.progress((n + 1) / max(1, len(jobs)), f"Chart {key}")
    reporter.log(f"Wrote {len(paths)} chart file(s) to {out_dir}")
    return paths
