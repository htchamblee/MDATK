"""Residue lookup, one-letter codes and the residue colouring / glow shared by the structure,
sequence and secondary-structure panels (so all three show the same colours and glow)."""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import numpy as np
from matplotlib.colors import to_rgba

from ..selections import SelectionError, parse_residue_ranges
from .coloring import glow_intensity, static_colors

log = logging.getLogger(__name__)
_WARNED: set = set()


def warn_once(msg: str) -> None:
    """Log a warning once per process: the GUI rebuilds its panels on every preview refresh."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        log.warning(msg)


_AMINO = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
          "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
          "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C"}


def one_letter(resname: str) -> str:
    """Base letter of a nucleotide in any naming scheme (G5, RA, LG, 0A, DT ...), amino-acid code, else X."""
    from ..pdbtools import parse_residue_name

    info = parse_residue_name(resname)
    if info is not None:
        return info.base
    return _AMINO.get(resname.strip().upper(), "X")


def chain_names(tr) -> np.ndarray:
    """Chain letters of the trace rows (older caches without them: letters from the chain index)."""
    if "chain_name" in tr.arrays:
        names = np.asarray(tr.arrays["chain_name"]).astype(str)
        if any(names):
            return names
    chain = np.asarray(tr.arrays["chain"], dtype=np.int64)
    return np.array([chr(ord("A") + int(c) % 26) for c in chain])


def resnames(tr) -> np.ndarray:
    if "resname" in tr.arrays:
        return np.asarray(tr.arrays["resname"]).astype(str)
    labels = tr.meta.get("residue_labels") or []
    if len(labels) == len(tr.arrays["chain"]):  # "G12" / "A:G12"
        return np.array([lab.split(":")[-1].rstrip("0123456789-") or "X" for lab in labels])
    return np.array(["X"] * len(tr.arrays["chain"]))


def residue_rows(tr, spec: str) -> np.ndarray:
    """Trace rows of residue ranges such as ``"12"``, ``"B:12"`` or ``"A:1-20, B:5-30"`` (file order);
    all rows for an empty spec.  Raises :class:`SelectionError` for a spec matching nothing."""
    n = len(tr.arrays["chain"])
    if not spec or not spec.strip():
        return np.arange(n)
    resid = np.asarray(tr.arrays["resid"], dtype=np.int64)
    chains = chain_names(tr)
    keep = np.zeros(n, bool)
    for chain, a, b in parse_residue_ranges(spec):
        sel = (resid >= a) & (resid <= b)
        if chain:
            sel &= chains == chain
        keep |= sel
    if not keep.any():
        raise SelectionError(f"no residue of the structure view matches {spec!r}")
    return np.flatnonzero(keep)


class Shading(NamedTuple):
    static: np.ndarray  # RGBA per residue (colour mode)
    member: np.ndarray  # residues of the proximity set (never glow)
    glow: np.ndarray  # rows x frames glow intensity
    color_i: np.ndarray  # rows x frames proximity-colour intensity
    glow_rgb: np.ndarray


def residue_shading(ctx, tr, need_proximity: Optional[bool] = None) -> Shading:
    """Colours and glow of the trace rows, exactly as the structure view computes them."""
    ds = ctx.dataset
    v = ctx.project.visual
    n_res = len(tr.arrays["chain"])
    n_frames = ds.n_frames if ds.n_frames else 1
    group_colors = [g.color for g in ctx.project.groups]
    static = static_colors(v.color_mode, np.asarray(tr.arrays["chain"]), np.asarray(tr.arrays["group_index"]),
                           group_colors, v.base_color, v.chain_palette)
    member = np.zeros(n_res, bool)
    glow = np.zeros((n_res, n_frames), np.float32)
    color_i = np.zeros((n_res, n_frames), np.float32)
    if need_proximity is None:
        need_proximity = v.color_mode == "proximity" or v.glow_mode != "none"
    if need_proximity:
        prox = ds.results.get("proximity")
        if prox is None:
            raise KeyError("proximity")
        dist = np.asarray(prox.arrays["min_distance"], dtype=np.float64)
        member = np.asarray(prox.arrays["member"], bool)
        if not np.array_equal(prox.arrays["resindex"], tr.arrays["resindex"]):
            order = {int(ix): k for k, ix in enumerate(prox.arrays["resindex"])}
            rows = [order.get(int(ix), -1) for ix in tr.arrays["resindex"]]
            dist = np.array([dist[k] if k >= 0 else np.full(dist.shape[1], np.inf) for k in rows])
            member = np.array([member[k] if k >= 0 else False for k in rows])
        glow = glow_intensity(dist, member, v.proximity_cutoff, v.glow_mode, v.fade_frames)
        mode = v.glow_mode if v.glow_mode != "none" else "current"
        color_i = glow_intensity(dist, member, v.proximity_cutoff, mode, v.fade_frames)
    return Shading(static, member, glow, color_i, np.array(to_rgba(v.glow_color))[:3])


def tag_rows(tr, v) -> np.ndarray:
    """Rows that get a residue-number tag (visual.label_every / label_residues)."""
    if "resid" not in tr.arrays or not (v.label_every > 0 or v.label_residues.strip()):
        return np.zeros(0, dtype=np.int64)
    resid = np.asarray(tr.arrays["resid"])
    wanted = np.zeros(len(resid), bool)
    if v.label_every > 0:
        wanted |= (resid % int(v.label_every)) == 0
    if v.label_residues.strip():
        try:
            for _chain, a, b in parse_residue_ranges(v.label_residues):
                wanted |= (resid >= a) & (resid <= b)
        except SelectionError:
            pass
    return np.flatnonzero(wanted)
