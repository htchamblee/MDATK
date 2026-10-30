"""Residue-group and selection helpers.

Selections in MDATK are ordinary MDAnalysis selection strings with one
extension: ``group:<Name>`` expands to the selection of a residue group defined
in the project (e.g. ``"group:Aptamer and name P"``).  Groups are defined by
residue ranges (``"1-40, 45, A:50-60"``) or by a raw selection string.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

_GROUP_TOKEN = re.compile(r"group:([A-Za-z0-9_.+\-']+)")
_RANGE_ITEM = re.compile(r"^(?:(?P<chain>[A-Za-z][A-Za-z0-9]*):)?(?P<a>\d+)(?:[-:](?P<b>\d+))?$")

# Single-atom ion residues whose two-letter names would otherwise be mistaken
# for carbon / nitrogen / ... when guessing elements from atom names.
_ION_ELEMENTS = {
    "MG": "MG", "MG2": "MG", "NA": "NA", "NA+": "NA", "SOD": "NA", "K": "K", "K+": "K",
    "POT": "K", "CL": "CL", "CL-": "CL", "CLA": "CL", "CA": "CA", "CAL": "CA",
    "ZN": "ZN", "MN": "MN", "FE": "FE", "LI": "LI", "RB": "RB", "CS": "CS", "BR": "BR",
}


class SelectionError(ValueError):
    """Raised for invalid residue ranges or selections."""


def parse_residue_ranges(text: str) -> List[Tuple[Optional[str], int, int]]:
    """Parse ``"1-40, 45, A:50-60"`` into ``[(chain, first, last), ...]``."""
    items: List[Tuple[Optional[str], int, int]] = []
    if not text or not text.strip():
        return items
    normalised = re.sub(r"\s*([-:])\s*", r"\1", text.strip())
    for raw in re.split(r"[,;\s]+", normalised):
        raw = raw.strip()
        if not raw:
            continue
        m = _RANGE_ITEM.match(raw)
        if not m:
            raise SelectionError(f"invalid residue range item {raw!r}")
        a = int(m.group("a"))
        b = int(m.group("b")) if m.group("b") is not None else a
        if b < a:
            a, b = b, a
        items.append((m.group("chain"), a, b))
    return items


def ranges_to_selection(text: str, universe=None) -> str:
    """Convert residue ranges to an MDAnalysis selection string."""
    items = parse_residue_ranges(text)
    if not items:
        raise SelectionError("empty residue range")
    chain_attr = "segid"
    if universe is not None and hasattr(universe.atoms, "chainIDs"):
        chain_attr = "chainID"
    parts = []
    for chain, a, b in items:
        resid = f"resid {a}" if a == b else f"resid {a}:{b}"
        parts.append(f"({chain_attr} {chain} and {resid})" if chain else f"({resid})")
    return " or ".join(parts)


def group_selection(group, universe=None) -> str:
    """Selection string for a :class:`mdatk.config.ResidueGroup`."""
    if group.selection and group.selection.strip():
        return f"({group.selection.strip()})"
    if group.residues and group.residues.strip():
        return f"({ranges_to_selection(group.residues, universe)})"
    raise SelectionError(f"group {group.name!r} has neither residues nor a selection")


def expand_groups(text: str, groups: Sequence, universe=None, _depth: int = 0) -> str:
    """Replace ``group:<Name>`` tokens by the group's selection."""
    if _depth > 8:
        raise SelectionError("group definitions are nested too deeply (circular reference?)")
    by_name = {g.name: g for g in groups}

    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in by_name:
            raise SelectionError(
                f"unknown residue group {name!r} (defined: {', '.join(by_name) or 'none'})")
        return expand_groups(group_selection(by_name[name], universe), groups, universe, _depth + 1)

    return _GROUP_TOKEN.sub(repl, text)


def rename_group_tokens(text: str, old: str, new: str) -> str:
    """``text`` with every ``group:<old>`` token renamed to ``group:<new>``; a token naming another
    group (``group:<old>er``, ``group:<old>_2``) is read whole, as :func:`expand_groups` reads it."""
    return _GROUP_TOKEN.sub(lambda m: f"group:{new}" if m.group(1) == old else m.group(0), text)


def resolve_selection(text: str, groups: Sequence, universe=None) -> str:
    """Resolve a selection that may be a bare group name, ``sel:...`` or a selection."""
    text = (text or "").strip()
    if not text:
        raise SelectionError("empty selection")
    if text.startswith("sel:"):
        text = text[4:].strip()
    elif any(g.name == text for g in groups):
        text = f"group:{text}"
    return expand_groups(text, groups, universe)


def select(universe, text: str, groups: Sequence = (), allow_empty: bool = False):
    """Select atoms, expanding group tokens and giving readable errors."""
    resolved = resolve_selection(text, groups, universe)
    try:
        ag = universe.select_atoms(resolved)
    except Exception as exc:  # MDAnalysis raises several exception types
        raise SelectionError(f"invalid selection {text!r} -> {resolved!r}: {exc}") from exc
    if not allow_empty and ag.n_atoms == 0:
        raise SelectionError(f"selection {text!r} matched no atoms")
    return ag


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


def guess_element(name: str, resname: str = "", single_atom_residue: bool = False) -> str:
    """Guess an (upper-case) element symbol from an atom name."""
    nm = name.strip().upper()
    rn = resname.strip().upper()
    if single_atom_residue:
        for key in (nm, rn, nm.rstrip("+-0123456789"), rn.rstrip("+-0123456789")):
            if key in _ION_ELEMENTS:
                return _ION_ELEMENTS[key]
    stripped = nm.lstrip("0123456789")
    for ch in stripped:
        if ch.isalpha():
            return ch
    return ""


def atom_elements(atoms) -> np.ndarray:
    """Upper-case element symbols for an AtomGroup (topology value or guessed)."""
    n = atoms.n_atoms
    elements = None
    if hasattr(atoms, "elements"):
        try:
            el = np.asarray(atoms.elements, dtype=object)
            if n and all(isinstance(e, str) and e.strip() for e in el):
                elements = np.array([e.strip().upper() for e in el], dtype=object)
        except Exception:
            elements = None
    if elements is None:
        res_sizes = {}
        for res in atoms.residues:
            res_sizes[res.ix] = res.atoms.n_atoms
        elements = np.array([
            guess_element(a.name, a.resname, res_sizes.get(a.resindex, 2) == 1) for a in atoms
        ], dtype=object)
    return elements


def heavy_mask(atoms) -> np.ndarray:
    """Boolean mask of non-hydrogen atoms."""
    return atom_elements(atoms) != "H"


def heavy(atoms):
    return atoms[heavy_mask(atoms)]


def element_filter(atoms, elements: Iterable[str]):
    wanted = {e.strip().upper() for e in elements if e.strip()}
    return atoms[np.isin(atom_elements(atoms), list(wanted))]


def split_csv(text: str) -> List[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def residue_labels(residues) -> List[str]:
    """Readable labels such as ``"G12"`` (``"A:G12"`` when several segments exist)."""
    segids = getattr(residues, "segids", None)
    multi = segids is not None and len(set(segids)) > 1
    labels = []
    for res in residues:
        lab = f"{res.resname}{res.resid}"
        if multi:
            lab = f"{res.segid}:{lab}"
        labels.append(lab)
    return labels


# ---------------------------------------------------------------------------
# Frame / time specifications (PDB export)
# ---------------------------------------------------------------------------

_TIME_UNITS_PS = {"ps": 1.0, "ns": 1000.0, "us": 1.0e6, "fs": 1.0e-3}


def parse_frame_spec(text: str, n_frames: int) -> List[int]:
    """Parse ``"0, 10, 20-50:10, last, -1"`` into sorted unique frame indices."""
    frames: List[int] = []
    if not text or not text.strip():
        return frames
    for raw in re.split(r"[,;\s]+", text.strip()):
        if not raw:
            continue
        token = raw.lower()
        if token in ("last", "end"):
            frames.append(n_frames - 1)
            continue
        if token in ("first", "start"):
            frames.append(0)
            continue
        m = re.fullmatch(r"(\d+)-(\d+|last)(?::(\d+))?", token)
        if m:
            a = int(m.group(1))
            b = n_frames - 1 if m.group(2) == "last" else int(m.group(2))
            step = int(m.group(3) or 1)
            if step < 1:
                raise SelectionError(f"invalid frame step in {raw!r}")
            frames.extend(range(a, b + 1, step))
            continue
        try:
            v = int(token)
        except ValueError as exc:
            raise SelectionError(f"invalid frame specification {raw!r}") from exc
        frames.append(v + n_frames if v < 0 else v)
    bad = [f for f in frames if f < 0 or f >= n_frames]
    if bad:
        raise SelectionError(f"frame(s) {bad} outside trajectory (0-{n_frames - 1})")
    return sorted(set(frames))


def parse_time_spec(text: str, default_unit: str = "ps") -> List[float]:
    """Parse ``"1.5ns, 2000ps, 3"`` into times in picoseconds."""
    times: List[float] = []
    if not text or not text.strip():
        return times
    if default_unit == "frame":
        raise SelectionError("time specification needs a time unit when the display unit is 'frame'")
    for raw in re.split(r"[,;\s]+", text.strip()):
        if not raw:
            continue
        m = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)", raw)
        if not m:
            raise SelectionError(f"invalid time {raw!r}")
        unit = (m.group(2) or default_unit).lower()
        if unit not in _TIME_UNITS_PS:
            raise SelectionError(f"unknown time unit {unit!r} in {raw!r}")
        times.append(float(m.group(1)) * _TIME_UNITS_PS[unit])
    return times


def time_unit_factor(unit: str) -> float:
    """Divide picoseconds by this factor to obtain ``unit``."""
    if unit not in _TIME_UNITS_PS:
        raise ValueError(f"unknown time unit {unit!r}")
    return _TIME_UNITS_PS[unit]
