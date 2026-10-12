"""Standardised analysis results and the dataset handed to renderers.

Every analysis produces an :class:`AnalysisResult`: a bag of numpy arrays plus
JSON metadata that *describes* the arrays.  The description lists three kinds
of outputs, each addressed by a key ``"<analysis>/<output>"``:

``series``
    one value per analysed frame (RMSD, H-bond count, energy, ...)
``matrices``
    rows x frames (per-residue contact counts, Mg occupancy classes, ...)
``pairmaps``
    sparse per-frame residue x residue maps (instantaneous contact maps)

Renderers, charts and the GUI only ever talk to :class:`Dataset`, so adding a
new analysis requires no changes to the presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .selections import time_unit_factor

TIME_LABELS = {"ps": "Time (ps)", "ns": "Time (ns)", "us": "Time (µs)", "frame": "Frame"}


@dataclass
class AnalysisResult:
    name: str
    key: str
    params: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def outputs(self) -> Dict[str, Dict[str, Any]]:
        return self.meta.setdefault("outputs", {"series": {}, "matrices": {}, "pairmaps": {}})

    def add_series(self, out: str, array: str, label: str, unit: str = "", **extra) -> None:
        self.outputs.setdefault("series", {})[out] = dict(array=array, label=label, unit=unit, **extra)

    def add_matrix(self, out: str, array: str, rows: str, label: str, kind: str = "count",
                   **extra) -> None:
        self.outputs.setdefault("matrices", {})[out] = dict(array=array, rows=rows, label=label,
                                                           kind=kind, **extra)

    def add_pairmap(self, out: str, prefix: str, rows: str, cols: str, label: str, **extra) -> None:
        self.outputs.setdefault("pairmaps", {})[out] = dict(prefix=prefix, rows=rows, cols=cols,
                                                           label=label, **extra)


@dataclass
class Series:
    key: str
    values: np.ndarray
    label: str
    unit: str = ""
    target: Optional[float] = None

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit and self.unit != "count" else self.label


@dataclass
class Matrix:
    key: str
    values: np.ndarray  # rows x frames
    row_labels: List[str]
    label: str
    kind: str = "count"  # "count", "value" or "class"
    classes: Optional[List[str]] = None
    unit: str = ""


class PairMap:
    """Sparse per-frame residue x residue map stored in CSR-like arrays."""

    def __init__(self, key: str, offsets: np.ndarray, i: np.ndarray, j: np.ndarray,
                 values: np.ndarray, row_labels: List[str], col_labels: List[str], label: str):
        self.key = key
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.i = np.asarray(i, dtype=np.int64)
        self.j = np.asarray(j, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)
        self.row_labels = list(row_labels)
        self.col_labels = list(col_labels)
        self.label = label

    @property
    def shape(self):
        return len(self.row_labels), len(self.col_labels)

    @property
    def n_frames(self) -> int:
        return len(self.offsets) - 1

    def dense(self, frame_pos: int) -> np.ndarray:
        out = np.zeros(self.shape)
        s, e = self.offsets[frame_pos], self.offsets[frame_pos + 1]
        np.add.at(out, (self.i[s:e], self.j[s:e]), self.values[s:e])
        return out

    def frequency(self) -> np.ndarray:
        """Fraction of frames in which each residue pair is present."""
        out = np.zeros(self.shape)
        if self.n_frames == 0:
            return out
        np.add.at(out, (self.i, self.j), 1.0)
        return out / self.n_frames


class Dataset:
    """All analysis results for one frame selection, plus the time axis."""

    def __init__(self, frames: np.ndarray, time_ps: np.ndarray,
                 results: Optional[Dict[str, AnalysisResult]] = None, time_unit: str = "ns"):
        self.frames = np.asarray(frames, dtype=np.int64)
        self.time_ps = np.asarray(time_ps, dtype=np.float64)
        self.results: Dict[str, AnalysisResult] = dict(results or {})
        self.time_unit = time_unit

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def add(self, result: AnalysisResult) -> None:
        self.results[result.name] = result

    def time(self, unit: Optional[str] = None) -> np.ndarray:
        unit = unit or self.time_unit
        if unit == "frame":
            return self.frames.astype(np.float64)
        return self.time_ps / time_unit_factor(unit)

    def time_label(self, unit: Optional[str] = None) -> str:
        return TIME_LABELS.get(unit or self.time_unit, "Time")

    # ------------------------------------------------------------ discovery
    def _lookup(self, key: str, kind: str):
        if "/" not in key:
            raise KeyError(f"data key {key!r} must look like '<analysis>/<output>'")
        name, out = key.split("/", 1)
        res = self.results.get(name)
        if res is None:
            raise KeyError(f"analysis {name!r} has not been computed")
        desc = res.outputs.get(kind, {}).get(out)
        if desc is None:
            raise KeyError(f"{name} has no {kind[:-1]} output {out!r}")
        return res, desc

    def keys(self, kind: str) -> List[str]:
        out = []
        for name, res in self.results.items():
            out.extend(f"{name}/{k}" for k in res.outputs.get(kind, {}))
        return out

    def series_keys(self) -> List[str]:
        return self.keys("series")

    def matrix_keys(self) -> List[str]:
        return self.keys("matrices")

    def pairmap_keys(self) -> List[str]:
        return self.keys("pairmaps")

    def has(self, key: str) -> bool:
        try:
            self.kind_of(key)
            return True
        except KeyError:
            return False

    def kind_of(self, key: str) -> str:
        for kind in ("series", "matrices", "pairmaps"):
            try:
                self._lookup(key, kind)
                return kind
            except KeyError:
                continue
        raise KeyError(key)

    # ------------------------------------------------------------- accessors
    def series(self, key: str) -> Series:
        res, d = self._lookup(key, "series")
        return Series(key, np.asarray(res.arrays[d["array"]], dtype=np.float64), d.get("label", key),
                      d.get("unit", ""), d.get("target"))

    def matrix(self, key: str) -> Matrix:
        res, d = self._lookup(key, "matrices")
        return Matrix(key, np.asarray(res.arrays[d["array"]]), list(res.meta.get(d["rows"], [])),
                      d.get("label", key), d.get("kind", "count"), d.get("classes"), d.get("unit", ""))

    def pairmap(self, key: str) -> PairMap:
        res, d = self._lookup(key, "pairmaps")
        p = d["prefix"]
        a = res.arrays
        return PairMap(key, a[f"{p}_offsets"], a[f"{p}_i"], a[f"{p}_j"], a[f"{p}_v"],
                       res.meta.get(d["rows"], []), res.meta.get(d["cols"], []), d.get("label", key))

    def describe(self) -> Dict[str, List[str]]:
        return {"series": self.series_keys(), "matrices": self.matrix_keys(),
                "pairmaps": self.pairmap_keys()}


class CSRBuilder:
    """Accumulates per-frame sparse (i, j, value) triplets."""

    def __init__(self):
        self._i: List[np.ndarray] = []
        self._j: List[np.ndarray] = []
        self._v: List[np.ndarray] = []
        self._counts: List[int] = []

    def add(self, i, j, v) -> None:
        self._i.append(np.asarray(i, dtype=np.int32))
        self._j.append(np.asarray(j, dtype=np.int32))
        self._v.append(np.asarray(v, dtype=np.float32))
        self._counts.append(len(i))

    def arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        offsets = np.zeros(len(self._counts) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(self._counts)

        def cat(parts, dtype):
            return np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype)

        return {f"{prefix}_offsets": offsets, f"{prefix}_i": cat(self._i, np.int32),
                f"{prefix}_j": cat(self._j, np.int32), f"{prefix}_v": cat(self._v, np.float32)}
