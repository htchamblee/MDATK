"""Figure grid presets.

Each preset maps to a grid of ``rows x cols`` cells and a list of slots
``(row0, row1, col0, col1)`` (end exclusive) filled in order by the panels of
:class:`mdatk.config.LayoutParams`.
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
}


def preset(name: str) -> Tuple[int, int, List[Slot]]:
    if name not in PRESETS:
        raise ValueError(f"unknown layout preset {name!r} (choose from {', '.join(PRESETS)})")
    return PRESETS[name]


def n_slots(name: str) -> int:
    return len(preset(name)[2])
