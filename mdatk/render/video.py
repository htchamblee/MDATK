"""Multi-panel figure assembly and MP4 encoding through FFmpeg.

Rendering is fully headless: figures are plain :class:`matplotlib.figure.Figure`
objects drawn by the Agg canvas (no pyplot, so it is safe in worker threads),
and raw RGBA frames are piped to an FFmpeg subprocess.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .. import PROJECT_ABBREV
from ..config import Project
from ..data import Dataset
from ..progress import Reporter
from . import layout as layouts
from .panels import Panel, RenderContext, panel_class

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    pass


def find_ffmpeg(path: str = "ffmpeg") -> str:
    path = os.path.expanduser(path or "ffmpeg")
    if os.sep in path:
        if os.access(path, os.X_OK):
            return path
        raise VideoError(f"ffmpeg not executable at {path}")
    found = shutil.which(path)
    if not found:
        raise VideoError("ffmpeg was not found on PATH; install FFmpeg or set its path")
    return found


def available_encoders(ffmpeg: str) -> str:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                             timeout=30)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def choose_codec(ffmpeg: str, requested: str) -> str:
    encoders = available_encoders(ffmpeg)
    if not encoders or f" {requested} " in encoders:
        return requested
    for fallback in ("libx264", "libopenh264", "mpeg4"):
        if f" {fallback} " in encoders:
            log.warning("FFmpeg encoder %s unavailable; using %s", requested, fallback)
            return fallback
    return requested


class FFmpegWriter:
    """Pipe raw RGBA frames into FFmpeg."""

    def __init__(self, path: str, width: int, height: int, fps: float, codec: str = "libx264",
                 crf: int = 20, ffmpeg: str = "ffmpeg"):
        self.path = path
        self.width, self.height = width, height
        exe = find_ffmpeg(ffmpeg)
        codec = choose_codec(exe, codec)
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-an",
               "-vcodec", codec]
        if codec in ("libx264", "libx265"):
            cmd += ["-crf", str(int(crf)), "-preset", "medium"]
        elif codec == "mpeg4":
            cmd += ["-q:v", "3"]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.cmd = cmd
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._stderr)
        self.frames = 0

    def write(self, rgba) -> None:
        try:
            self.proc.stdin.write(rgba)
        except BrokenPipeError as exc:
            raise VideoError(f"FFmpeg stopped unexpectedly: {self._err()}") from exc
        self.frames += 1

    def _err(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace")[-2000:]

    def close(self, abort: bool = False) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        if abort:
            self.proc.kill()
        rc = self.proc.wait()
        err = self._err()
        self._stderr.close()
        if not abort and rc != 0:
            raise VideoError(f"FFmpeg exited with code {rc}: {err}")


def figure_size(width: int, height: int) -> Tuple[int, int]:
    """Even pixel dimensions (required by yuv420p)."""
    return max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)


def build_figure(dataset: Dataset, project: Project, width: Optional[int] = None,
                 height: Optional[int] = None, dpi: Optional[float] = None, watermark: str = ""):
    """Create the figure, its canvas and the panels (frozen layout)."""
    vp = project.output.video
    w, h = figure_size(width or vp.width, height or vp.height)
    dpi = dpi or vp.dpi
    ctx = RenderContext(dataset, project)
    theme = ctx.theme
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=theme.page)
    canvas = FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained", h_pad=0.08, w_pad=0.08, hspace=0.06, wspace=0.04)
    nrows, ncols, slots = layouts.preset(project.layout.preset)
    gs = GridSpec(nrows, ncols, figure=fig, height_ratios=layouts.height_ratios(project.layout.preset))
    panels: List[Panel] = []
    specs = list(project.layout.panels)
    if len(specs) > len(slots):
        log.warning("layout %s has %d slots; ignoring %d extra panel(s)", project.layout.preset,
                    len(slots), len(specs) - len(slots))
    for slot, spec in zip(slots, specs):
        r0, r1, c0, c1 = slot
        cls = panel_class(spec.type)
        panels.append(cls(fig, gs[r0:r1, c0:c1], spec, ctx))
    if project.visual.title:
        fig.suptitle(project.visual.title, color=theme.ink, fontsize=project.visual.font_size * 1.3)
    if watermark:
        fig.text(0.997, 0.003, watermark, ha="right", va="bottom", color=theme.muted,
                 fontsize=project.visual.font_size * 0.75)
    if dataset.n_frames:
        for p in panels:
            p.update(0)
    canvas.draw()
    fig.set_layout_engine("none")  # freeze positions: no re-layout (or jitter) per frame
    for p in panels:
        p.after_layout()
    return fig, canvas, panels


def _check_size(canvas, w: int, h: int) -> None:
    cw, ch = canvas.get_width_height()
    if (cw, ch) != (w, h):
        raise VideoError(f"canvas is {cw}x{ch} px but {w}x{h} was requested; "
                         "choose a DPI that divides the resolution")


def render_frame(dataset: Dataset, project: Project, pos: int, path: str,
                 width: Optional[int] = None, height: Optional[int] = None,
                 dpi: Optional[int] = None) -> str:
    """Render a single composite frame to an image file (PNG/SVG/PDF)."""
    fig, canvas, panels = build_figure(dataset, project, width, height, dpi)
    for p in panels:
        p.update(pos)
    fig.savefig(path, facecolor=fig.get_facecolor())
    return path


def video_positions(n_frames: int, stride: int) -> np.ndarray:
    return np.arange(0, n_frames, max(1, int(stride)))


def render_video(dataset: Dataset, project: Project, path: str,
                 reporter: Optional[Reporter] = None) -> str:
    reporter = reporter or Reporter()
    vp = project.output.video
    w, h = figure_size(vp.width, vp.height)
    fig, canvas, panels = build_figure(dataset, project)
    canvas.draw()
    w, h = canvas.get_width_height()
    if w % 2 or h % 2:
        raise VideoError(f"canvas size {w}x{h} is not even; adjust resolution or DPI")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    positions = video_positions(dataset.n_frames, vp.frame_stride)
    reporter.log(f"Rendering {len(positions)} video frame(s) at {w}x{h}, {vp.fps} fps -> {path}")
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                            f".{PROJECT_ABBREV}_tmp_{os.getpid()}_{os.path.basename(path)}")
    # Blitting: static content (axes, ticks, full traces, heatmaps) is drawn once;
    # each frame restores it and redraws only the artists that move.
    animated = [a for p in panels if p.ok for a in p.animated_artists()]
    for a in animated:
        a.set_animated(True)
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    writer = FFmpegWriter(tmp_path, w, h, vp.fps, vp.codec, vp.crf, vp.ffmpeg_path)
    ok = False
    try:
        every = max(1, len(positions) // 200)
        for k, pos in enumerate(positions):
            reporter.check_cancel()
            for p in panels:
                p.update(int(pos))
            canvas.restore_region(background)
            for a in animated:
                fig.draw_artist(a)
            writer.write(canvas.buffer_rgba())
            if k % every == 0 or k == len(positions) - 1:
                reporter.progress((k + 1) / len(positions), f"Video frame {k + 1}/{len(positions)}")
        ok = True
    finally:
        writer.close(abort=not ok)
        if not ok and os.path.exists(tmp_path):
            os.remove(tmp_path)
    os.replace(tmp_path, path)
    reporter.log(f"Wrote {path} ({writer.frames} frames)")
    return path
