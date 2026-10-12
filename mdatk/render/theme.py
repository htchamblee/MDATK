"""Colours and chart chrome shared by the video panels and standalone charts.

The categorical order, sequential ramp and chrome inks follow a palette whose
adjacent pairs were validated for colour-vision deficiencies in both light and
dark mode; series colours are assigned in this fixed order, never cycled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                   "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str

    @property
    def series(self) -> List[str]:
        return CATEGORICAL["dark" if self.name == "dark" else "light"]

    def series_color(self, k: int) -> str:
        # Fixed order; beyond eight series the palette is not extended.
        return self.series[k % len(self.series)]

    def sequential(self) -> LinearSegmentedColormap:
        """One-hue magnitude ramp; zero recedes toward the chart surface."""
        if self.name == "dark":
            steps = [self.surface] + list(reversed(SEQUENTIAL_BLUE))
        else:
            steps = [self.surface] + SEQUENTIAL_BLUE
        return LinearSegmentedColormap.from_list(f"mdatk_seq_{self.name}", steps)

    def ordinal(self, n: int) -> ListedColormap:
        """Discrete ramp for ordered classes (class 0 recedes to the surface)."""
        if n <= 1:
            return ListedColormap([self.surface])
        ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"] if self.name != "dark" else \
               ["#184f95", "#3987e5", "#86b6ef", "#cde2fb"]
        zero = "#e1e0d9" if self.name != "dark" else "#2c2c2a"
        if n - 1 <= len(ramp):
            idx = [round(k * (len(ramp) - 1) / max(1, n - 2)) for k in range(n - 1)] if n > 2 else [1]
            cols = [ramp[i] for i in idx]
        else:
            cols = ramp
        return ListedColormap([zero] + cols[: n - 1])


LIGHT = Theme("light", surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e",
              muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
DARK = Theme("dark", surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7",
             muted="#898781", grid="#2c2c2a", axis="#383835")


def get_theme(name: str) -> Theme:
    return DARK if name == "dark" else LIGHT


def style_axes(ax, theme: Theme, grid: bool = True, font_size: float = 10.0) -> None:
    """Recessive chrome: hairline solid grid and axes, muted tick labels."""
    ax.set_facecolor(theme.surface)
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(theme.axis)
            spine.set_linewidth(0.8)
    ax.tick_params(colors=theme.muted, labelcolor=theme.ink2, labelsize=font_size * 0.85,
                   width=0.6, length=3)
    ax.xaxis.label.set_color(theme.ink2)
    ax.yaxis.label.set_color(theme.ink2)
    ax.xaxis.label.set_size(font_size * 0.9)
    ax.yaxis.label.set_size(font_size * 0.9)
    ax.title.set_color(theme.ink)
    ax.title.set_size(font_size)
    if grid:
        ax.grid(True, color=theme.grid, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)


def style_legend(legend, theme: Theme, font_size: float = 10.0) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor(theme.surface)
    legend.get_frame().set_edgecolor(theme.grid)
    legend.get_frame().set_linewidth(0.6)
    for text in legend.get_texts():
        text.set_color(theme.ink2)
        text.set_fontsize(font_size * 0.8)
