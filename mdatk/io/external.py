"""External time-series data (CSV / whitespace text, e.g. cpptraj ``.dat`` files)."""

from __future__ import annotations

import os
import re
from typing import List, Sequence, Tuple

import numpy as np

from ..data import AnalysisResult, Dataset
from ..selections import time_unit_factor


def _split(line: str, delimiter: str) -> List[str]:
    if delimiter:
        return [t.strip() for t in line.split(delimiter)]
    if "," in line:
        return [t.strip() for t in line.split(",")]
    if "\t" in line:
        return [t.strip() for t in line.split("\t")]
    return line.split()


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_table(path: str, delimiter: str = "") -> Tuple[List[str], np.ndarray]:
    """Read a numeric table with an optional header line (``#`` headers allowed)."""
    header: List[str] = []
    rows: List[List[float]] = []
    pending_comment: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line[0] in "#@;%":
                if not rows:
                    pending_comment = _split(line.lstrip("#@;% "), delimiter)
                continue
            tokens = _split(line, delimiter)
            if not rows and not header and not all(_is_number(t) for t in tokens):
                header = tokens
                continue
            try:
                rows.append([float(t) for t in tokens])
            except ValueError as exc:
                raise ValueError(f"{path}: non-numeric data line {line!r}") from exc
    if not rows:
        raise ValueError(f"{path}: no numeric data found")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows], dtype=np.float64)
    if not header and len(pending_comment) == width:
        header = pending_comment
    if len(header) != width:
        header = [f"col{k}" for k in range(width)]
    return header, data


def _column_index(header: Sequence[str], spec: str, default: int) -> int:
    if spec == "" or spec is None:
        return default
    if spec in header:
        return list(header).index(spec)
    if re.fullmatch(r"\d+", str(spec)):
        k = int(spec)
        if k < len(header):
            return k
    raise ValueError(f"column {spec!r} not found (available: {', '.join(header)})")


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+\-]+", "_", text).strip("_") or "value"


def load_external(specs, dataset: Dataset) -> AnalysisResult:
    """Interpolate external columns onto the dataset's frames.

    ``x_unit`` may be ``ps``/``ns``/``us`` (time), ``frame`` (0-based index into
    the trajectory) or ``frame1`` (1-based, as written by cpptraj).
    """
    result = AnalysisResult(name="external", key="", params={}, arrays={}, meta={})
    for n, spec in enumerate(specs):
        if not spec.path:
            continue
        header, data = read_table(spec.path, spec.delimiter)
        xi = _column_index(header, spec.x_column, 0)
        x = data[:, xi]
        unit = spec.x_unit or "ps"
        if unit == "frame":
            target = dataset.frames.astype(float)
        elif unit == "frame1":
            target = dataset.frames.astype(float) + 1.0
        else:
            x = x * time_unit_factor(unit)
            target = dataset.time_ps
        order = np.argsort(x, kind="stable")
        x = x[order]
        cols = spec.columns or [h for k, h in enumerate(header) if k != xi]
        label = spec.label or os.path.splitext(os.path.basename(spec.path))[0]
        for col in cols:
            ci = _column_index(header, col, -1)
            y = data[order, ci]
            values = np.interp(target, x, y, left=np.nan, right=np.nan)
            out = _sanitize(f"{label}.{header[ci]}")
            arr = f"s{n}_{ci}"
            result.arrays[arr] = values
            result.add_series(out, arr, f"{label}: {header[ci]}", "")
    return result
