"""Residue colouring and proximity glow for the structural view."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgba

GLOW_MODES = ("none", "ever", "fade", "current")
COLOR_MODES = ("chain", "groups", "proximity")


def glow_intensity(min_distance: np.ndarray, member: np.ndarray, cutoff: float, mode: str,
                   fade_frames: int = 0) -> np.ndarray:
    """Per-residue glow intensity (rows x frames, 0..1).

    ``ever``    1 for the whole movie if the residue ever comes within the cutoff
    ``fade``    1 while within the cutoff, then a linear fade over ``fade_frames``
    ``current`` 1 only while within the cutoff
    ``none``    always 0
    Members of the proximity set never glow.
    """
    d = np.asarray(min_distance, dtype=np.float64)
    near = (d <= cutoff) & ~np.asarray(member, bool)[:, None]
    if mode == "none":
        return np.zeros_like(d, dtype=np.float32)
    if mode == "current":
        return near.astype(np.float32)
    if mode == "ever":
        return np.repeat(near.any(axis=1, keepdims=True), d.shape[1], axis=1).astype(np.float32)
    if mode == "fade":
        out = np.zeros_like(d, dtype=np.float32)
        since = np.full(d.shape[0], np.inf)
        for t in range(d.shape[1]):
            since = np.where(near[:, t], 0.0, since + 1.0)
            if fade_frames > 0:
                out[:, t] = np.clip(1.0 - since / (fade_frames + 1.0), 0.0, 1.0)
            else:
                out[:, t] = near[:, t]
        return out
    raise ValueError(f"unknown glow mode {mode!r} (choose from {', '.join(GLOW_MODES)})")


def palette(name: str, n: int) -> List[Tuple[float, float, float, float]]:
    cmap = colormaps.get_cmap(name)
    if getattr(cmap, "N", 256) <= 20:
        return [cmap(k % cmap.N) for k in range(n)]
    return [cmap(v) for v in np.linspace(0.05, 0.95, max(n, 1))]


def static_colors(mode: str, chain: np.ndarray, group_index: np.ndarray, group_colors: List[str],
                  base_color: str, chain_palette: str) -> np.ndarray:
    """RGBA per residue that do not change over time."""
    n = len(chain)
    base = np.array(to_rgba(base_color))
    out = np.tile(base, (n, 1))
    if mode == "chain":
        chains = np.unique(chain)
        cols = palette(chain_palette, len(chains))
        for k, c in enumerate(chains):
            out[chain == c] = cols[k]
    elif mode in ("groups", "proximity"):
        for gi, color in enumerate(group_colors):
            out[group_index == gi] = to_rgba(color)
    return out


def frame_colors(static: np.ndarray, mode: str, base_color: str, proximity_color: str,
                 intensity: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Colours for one frame; in proximity mode non-members blend base -> highlight."""
    if mode != "proximity":
        return static
    base = np.array(to_rgba(base_color))
    hi = np.array(to_rgba(proximity_color))
    i = np.asarray(intensity, dtype=np.float64)[:, None]
    out = base * (1.0 - i) + hi * i
    out[member] = static[member]
    return out
