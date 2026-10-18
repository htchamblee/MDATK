"""Figure grid presets.

Each preset maps to a grid of ``rows x cols`` cells and a list of slots
``(row0, row1, col0, col1)`` (end exclusive) filled in order by the panels of
:class:`mdatk.config.LayoutParams`.  The ``*_strip`` presets add a thin
full-width row at the bottom (their last slot) for a sequence or
secondary-structure panel; :data:`HEIGHT_RATIOS` gives their row heights.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Slot = Tuple[int, int, int, int]

PRESETS: Dict[str, Tuple[int, int, List[Slot]]] = {
    "single": (1, 1, [(0, 1, 0, 1)]),
    "large_plus_2": (2, 3, [(0, 2, 0, 2), (0, 1, 2, 3), (1, 2, 2, 3)]),
    "large_plus_3": (3, 3, [(0, 3, 0, 2), (0, 1, 2, 3), (1, 2, 2, 3), (2, 3, 2, 3)]),
    "large_plus_4": (2, 4, [(0, 2, 0, 2), (0, 1, 2, 3), (0, 1, 3, 4), (1, 2, 2, 3), (1, 2, 3, 4)]),
    "large_top_2": (3, 2, [(0, 2, 0, 2), (2, 3, 0, 1), (2, 3, 1, 2)]),
    "large_top_3": (3, 3, [(0, 2, 0, 3), (2, 3, 0, 1), (2, 3, 1, 2), (2, 3, 2, 3)]),
    "equal_2": (1, 2, [(0, 1, 0, 1), (0, 1, 1, 2)]),
    "equal_3": (1, 3, [(0, 1, 0, 1), (0, 1, 1, 2), (0, 1, 2, 3)]),
    "equal_4": (2, 2, [(0, 1, 0, 1), (0, 1, 1, 2), (1, 2, 0, 1), (1, 2, 1, 2)]),
    "equal_5": (2, 6, [(0, 1, 0, 2), (0, 1, 2, 4), (0, 1, 4, 6), (1, 2, 1, 3), (1, 2, 3, 5)]),
    "equal_6": (2, 3, [(0, 1, 0, 1), (0, 1, 1, 2), (0, 1, 2, 3),
                       (1, 2, 0, 1), (1, 2, 1, 2), (1, 2, 2, 3)]),
    "single_strip": (2, 1, [(0, 1, 0, 1), (1, 2, 0, 1)]),
    "large_plus_2_strip": (3, 3, [(0, 2, 0, 2), (0, 1, 2, 3), (1, 2, 2, 3), (2, 3, 0, 3)]),
    "large_plus_3_strip": (4, 3, [(0, 3, 0, 2), (0, 1, 2, 3), (1, 2, 2, 3), (2, 3, 2, 3), (3, 4, 0, 3)]),
    "large_top_3_strip": (4, 3, [(0, 2, 0, 3), (2, 3, 0, 1), (2, 3, 1, 2), (2, 3, 2, 3), (3, 4, 0, 3)]),
    "equal_4_strip": (3, 2, [(0, 1, 0, 1), (0, 1, 1, 2), (1, 2, 0, 1), (1, 2, 1, 2), (2, 3, 0, 2)]),
}

#: relative row heights of presets whose rows are not all equal
HEIGHT_RATIOS: Dict[str, List[float]] = {
    "single_strip": [1.0, 0.22],
    "large_plus_2_strip": [1.0, 1.0, 0.4],
    "large_plus_3_strip": [1.0, 1.0, 1.0, 0.55],
    "large_top_3_strip": [1.0, 1.0, 1.0, 0.55],
    "equal_4_strip": [1.0, 1.0, 0.4],
}

PRESET_LABELS = {
    "single": "1 large panel",
    "large_plus_2": "1 large + 2 small (right)",
    "large_plus_3": "1 large + 3 small (right)",
    "large_plus_4": "1 large + 4 small (right)",
    "large_top_2": "1 large + 2 small (bottom)",
    "large_top_3": "1 large + 3 small (bottom)",
    "equal_2": "2 equal panels",
    "equal_3": "3 equal panels",
    "equal_4": "4 equal panels (2x2)",
    "equal_5": "5 equal panels",
    "equal_6": "6 equal panels (2x3)",
    "single_strip": "1 large + full-width strip",
    "large_plus_2_strip": "1 large + 2 small + strip",
    "large_plus_3_strip": "1 large + 3 small + strip",
    "large_top_3_strip": "1 large (top) + 3 small + strip",
    "equal_4_strip": "4 equal panels + strip",
}


def preset(name: str) -> Tuple[int, int, List[Slot]]:
    if name not in PRESETS:
        raise ValueError(f"unknown layout preset {name!r} (choose from {', '.join(PRESETS)})")
    return PRESETS[name]


def n_slots(name: str) -> int:
    return len(preset(name)[2])


def height_ratios(name: str) -> List[float]:
    """Row heights of a preset (all 1 unless the preset defines them)."""
    rows = preset(name)[0]
    return list(HEIGHT_RATIOS.get(name, [1.0] * rows))
