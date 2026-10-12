"""Loading topologies and trajectories through MDAnalysis.

Frames are always streamed: MDAnalysis keeps a single frame in memory and the
analysis runner visits each selected frame exactly once.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

_TOPOLOGY_FORMATS = {".parm7": "PARM7", ".prmtop": "PARM7", ".top": "PARM7"}
_TRAJECTORY_FORMATS = {".nc": "NCDF", ".ncdf": "NCDF", ".netcdf": "NCDF",
                       ".mdcrd": "TRJ", ".crd": "TRJ", ".trj": "TRJ",
                       ".rst7": "INPCRD", ".inpcrd": "INPCRD", ".ncrst": "NCRST"}


def is_parm7(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _TOPOLOGY_FORMATS


def load_universe(topology: str, trajectories: Sequence[str] = (), dt_ps: Optional[float] = None):
    """Create an MDAnalysis Universe, choosing formats from file extensions."""
    import MDAnalysis as mda

    if not topology:
        raise ValueError("no topology file given")
    if not os.path.exists(topology):
        raise FileNotFoundError(f"topology not found: {topology}")
    for t in trajectories:
        if not os.path.exists(t):
            raise FileNotFoundError(f"trajectory not found: {t}")
    kwargs: Dict[str, object] = {}
    ext = os.path.splitext(topology)[1].lower()
    if ext in _TOPOLOGY_FORMATS:
        kwargs["topology_format"] = _TOPOLOGY_FORMATS[ext]
    traj_exts = {os.path.splitext(t)[1].lower() for t in trajectories}
    if len(traj_exts) == 1:
        fmt = _TRAJECTORY_FORMATS.get(next(iter(traj_exts)))
        if fmt:
            kwargs["format"] = fmt
    if dt_ps:
        kwargs["dt"] = float(dt_ps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if trajectories:
            u = mda.Universe(topology, *list(trajectories), **kwargs)
        else:
            kwargs.pop("format", None)
            u = mda.Universe(topology, **kwargs)
    log.info("loaded %s: %d atoms, %d residues, %d frames", os.path.basename(topology),
             u.atoms.n_atoms, u.residues.n_residues, u.trajectory.n_frames)
    return u


def frame_indices(n_frames: int, start: int = 0, stop: Optional[int] = None,
                  step: int = 1) -> np.ndarray:
    """0-based frame indices selected by a python-style slice."""
    if step < 1:
        raise ValueError("frame step must be >= 1")
    idx = np.arange(n_frames)[slice(start, stop, step)]
    if len(idx) == 0:
        raise ValueError(f"frame selection start={start} stop={stop} step={step} "
                         f"selects no frames (trajectory has {n_frames})")
    return idx


def system_summary(u) -> Dict[str, object]:
    """Small JSON-friendly description of a loaded system."""
    resnames, counts = np.unique(u.residues.resnames, return_counts=True)
    order = np.argsort(-counts)
    traj = u.trajectory
    return {
        "n_atoms": int(u.atoms.n_atoms),
        "n_residues": int(u.residues.n_residues),
        "n_segments": int(u.segments.n_segments),
        "n_frames": int(traj.n_frames),
        "dt_ps": float(traj.dt) if traj.n_frames > 1 else None,
        "has_box": bool(traj.ts.dimensions is not None),
        "top_resnames": {str(resnames[k]): int(counts[k]) for k in order[:12]},
    }


def residue_table(u, max_rows: Optional[int] = None) -> List[Dict[str, object]]:
    rows = []
    for res in u.residues[: max_rows or None]:
        rows.append({"resid": int(res.resid), "resname": str(res.resname),
                     "segid": str(res.segid), "n_atoms": int(res.atoms.n_atoms)})
    return rows
