"""Standardised cache I/O.

Layout of a cache directory::

    MDATK_cache/
        MDATK_cache.json          manifest: system, frames, time axis summary,
                                  residue groups, available analyses
        contacts-<key>.npz        compressed arrays of one analysis result
        contacts-<key>.json       parameters + output description of that result
        cleaned/<key>/...         cpptraj outputs (see mdatk.cleaning)

Each analysis result is stored under a key derived from everything that
affects it (input files, frame selection, analysis parameters and the residue
groups it references), so changing a visual parameter never invalidates an
analysis and several parameter variants can live side by side.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .. import PROJECT_ABBREV, __version__
from ..data import AnalysisResult, Dataset

log = logging.getLogger(__name__)

MANIFEST_NAME = f"{PROJECT_ABBREV}_cache.json"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


class CacheStore:
    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(cache_dir)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.cache_dir, MANIFEST_NAME)

    def _base(self, name: str, key: str) -> str:
        return os.path.join(self.cache_dir, f"{name}-{key}")

    def ensure(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

    # --------------------------------------------------------------- results
    def has(self, name: str, key: str) -> bool:
        base = self._base(name, key)
        return os.path.exists(base + ".npz") and os.path.exists(base + ".json")

    def save_result(self, result: AnalysisResult, frames: np.ndarray, time_ps: np.ndarray) -> str:
        self.ensure()
        base = self._base(result.name, result.key)
        arrays = dict(result.arrays)
        arrays["_frames"] = np.asarray(frames, dtype=np.int64)
        arrays["_time_ps"] = np.asarray(time_ps, dtype=np.float64)
        tmp = base + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, base + ".npz")
        write_json(base + ".json", {
            "project": PROJECT_ABBREV, "mdatk_version": __version__, "created": _now(),
            "name": result.name, "key": result.key, "params": result.params, "meta": result.meta,
            "arrays": sorted(result.arrays), "n_frames": int(len(frames)),
        })
        return base + ".npz"

    def load_result(self, name: str, key: str):
        """Return ``(result, frames, time_ps)`` or ``None`` if not cached / unreadable."""
        if not self.has(name, key):
            return None
        base = self._base(name, key)
        try:
            with open(base + ".json", "r", encoding="utf-8") as fh:
                info = json.load(fh)
            with np.load(base + ".npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        except Exception as exc:  # corrupt / partial cache entry
            log.warning("ignoring unreadable cache entry %s: %s", base, exc)
            return None
        frames = arrays.pop("_frames")
        time_ps = arrays.pop("_time_ps")
        result = AnalysisResult(name=info["name"], key=info["key"], params=info.get("params", {}),
                                arrays=arrays, meta=info.get("meta", {}))
        return result, frames, time_ps

    # -------------------------------------------------------------- manifest
    def read_manifest(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_manifest(self, *, system: Dict[str, Any], frame_selection: Dict[str, Any],
                       dataset: Dataset, groups: Iterable[Dict[str, Any]],
                       entries: Dict[str, str], extra: Optional[Dict[str, Any]] = None) -> str:
        """Record the latest set of results (``entries`` maps analysis name -> key)."""
        self.ensure()
        old = self.read_manifest() or {}
        known = {f"{e['name']}-{e['key']}": e for e in old.get("all_entries", [])}
        current: List[Dict[str, Any]] = []
        for name, key in entries.items():
            res = dataset.results.get(name)
            entry = {"name": name, "key": key, "file": f"{name}-{key}.npz",
                     "params": res.params if res else {},
                     "outputs": {kind: sorted(v) for kind, v in (res.outputs.items() if res else [])}}
            current.append(entry)
            known[f"{name}-{key}"] = {k: entry[k] for k in ("name", "key", "file")}
        t = dataset.time_ps
        manifest = {
            "project": PROJECT_ABBREV,
            "mdatk_version": __version__,
            "updated": _now(),
            "system": system,
            "frames": dict(frame_selection, n_frames=int(dataset.n_frames),
                           first_frame=int(dataset.frames[0]) if dataset.n_frames else None,
                           last_frame=int(dataset.frames[-1]) if dataset.n_frames else None),
            "time_axis": {"unit": "ps", "n": int(len(t)),
                          "start": float(t[0]) if len(t) else None,
                          "end": float(t[-1]) if len(t) else None,
                          "dt": float(np.median(np.diff(t))) if len(t) > 1 else None},
            "groups": list(groups),
            "available": dataset.describe(),
            "entries": current,
            "all_entries": sorted(known.values(), key=lambda e: (e["name"], e["key"])),
        }
        if extra:
            manifest.update(extra)
        write_json(self.manifest_path, manifest)
        return self.manifest_path

    def load_dataset_from_manifest(self, time_unit: str = "ns") -> Dataset:
        """Rebuild the last dataset recorded in the manifest (no trajectory needed)."""
        manifest = self.read_manifest()
        if manifest is None:
            raise FileNotFoundError(f"no {MANIFEST_NAME} in {self.cache_dir}")
        dataset: Optional[Dataset] = None
        for entry in manifest.get("entries", []):
            loaded = self.load_result(entry["name"], entry["key"])
            if loaded is None:
                raise FileNotFoundError(f"cache entry {entry['file']} is missing")
            res, frames, time_ps = loaded
            if dataset is None:
                dataset = Dataset(frames, time_ps, time_unit=time_unit)
            dataset.add(res)
        if dataset is None:
            raise FileNotFoundError("the cache manifest lists no analyses")
        return dataset
