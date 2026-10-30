"""Export selected trajectory frames as PDB files."""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..selections import (SelectionError, parse_frame_spec, parse_time_spec, select, split_csv,
                          unknown_group_error)

log = logging.getLogger(__name__)


def resolve_export_selection(universe, text: str, groups: Sequence):
    """``""`` -> all atoms; ``"Aptamer, Target"`` -> union of groups; else a selection."""
    text = (text or "").strip()
    if not text:
        return universe.atoms
    names = split_csv(text)
    group_names = [g.name for g in groups]
    if names and all(n in group_names for n in names):
        text = " or ".join(f"group:{n}" for n in names)
    elif "," in text and all(re.fullmatch(r"[A-Za-z0-9_.\-]+", n) for n in names):
        # a list of group names (a selection has no commas): name the one that does not exist
        raise unknown_group_error(next(n for n in names if n not in group_names), group_names)
    return select(universe, text, groups)


def frames_from_specs(universe, frames_text: str, times_text: str, time_unit: str,
                      dt_ps: Optional[float] = None) -> List[int]:
    """Resolve frame numbers and times (nearest frame) into 0-based frame indices."""
    n = universe.trajectory.n_frames
    frames = set(parse_frame_spec(frames_text, n))
    times = parse_time_spec(times_text, time_unit if time_unit != "frame" else "ps")
    if times:
        if dt_ps:
            all_t = np.arange(n) * float(dt_ps)
        else:
            all_t = np.array([ts.time for ts in universe.trajectory], dtype=float)
        for t in times:
            k = int(np.argmin(np.abs(all_t - t)))
            if abs(all_t[k] - t) > max(1e-6, 0.5 * (np.median(np.diff(all_t)) if n > 1 else 1.0)):
                log.warning("time %.3f ps is outside the trajectory; using frame %d (%.3f ps)",
                            t, k, all_t[k])
            frames.add(k)
    return sorted(frames)


def export_frames(universe, frames: Sequence[int], atoms, out_dir: str, prefix: str,
                  progress: Optional[Callable[[float, str], None]] = None,
                  stamp: str = "") -> List[str]:
    """Write one PDB per frame; returns the written paths."""
    if not frames:
        raise SelectionError("no frames selected for PDB export")
    os.makedirs(out_dir, exist_ok=True)
    width = max(5, len(str(universe.trajectory.n_frames)))
    paths = []
    for k, f in enumerate(frames):
        universe.trajectory[int(f)]
        suffix = f"_{stamp}" if stamp else ""
        path = os.path.join(out_dir, f"{prefix}_{int(f):0{width}d}{suffix}.pdb")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms.write(path)
        paths.append(path)
        if progress:
            progress((k + 1) / len(frames), f"PDB frame {f}")
    log.info("wrote %d PDB file(s) to %s", len(paths), out_dir)
    return paths
