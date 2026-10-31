"""Residue-group and selection helpers.

Selections in MDATK are ordinary MDAnalysis selection strings with one
extension: ``group:<Name>`` expands to the selection of a residue group defined
in the project (e.g. ``"group:Aptamer and name P"``).  Groups are defined by
residue ranges (``"1-40, 45, A:50-60"``) or by a raw selection string.
"""

from __future__ import annotations

import difflib
import re
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .problems import Problem, describe

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
    """Raised for invalid residue ranges or selections.

    ``str()`` is the full message.  The error also says what went wrong (``kind``: "unknown_group",
    "syntax", "empty", "nesting", "range", "missing", "no_text" or "" for other problems) and carries
    the offending ``token`` (a group name, a clause, an MDAnalysis keyword), the selection as written
    (``text``) and expanded (``expanded``), MDAnalysis' own message (``detail``), a ``suggestion``
    and, when the definition of a residue group is at fault, that ``group``.  The code that selected
    names the setting the text came from with :meth:`locate`, which puts its readable location (see
    :mod:`mdatk.problems`) in front of the message and keeps its ``field`` path.
    """

    def __init__(self, message: str, *, kind: str = "", token: str = "", text: str = "",
                 expanded: str = "", detail: str = "", suggestion: str = "", group: str = ""):
        super().__init__(message)
        self.message = message
        self.kind, self.token, self.text, self.expanded = kind, token, text, expanded
        self.detail, self.suggestion, self.group = detail, suggestion, group
        self.field = ""
        self.location = ""

    def locate(self, field: str, project=None, force: bool = False) -> "SelectionError":
        """Name the setting the selection came from (``"analyses.hbonds.group_a"``); the first one
        named wins unless ``force``.  Returns the error, so ``raise exc.locate(...)`` works."""
        if field and (force or not self.field):
            self.field = field
            self.location = describe(field, project)
            self.args = (f"{self.location}: {self.message}",)
        return self

    def problem(self, level: str = "error") -> Problem:
        """The error as a :class:`~mdatk.problems.Problem` of its setting."""
        return Problem(self.message, self.field, self.location, level, self.token)


def suggest(name: str, candidates: Iterable[str]) -> Optional[str]:
    """The candidate ``name`` was probably meant to be (other case, typo), or None."""
    candidates = [c for c in candidates if c]
    for c in candidates:
        if c.lower() == name.lower():
            return c
    lower = {c.lower(): c for c in candidates}
    close = difflib.get_close_matches(name.lower(), list(lower), n=1, cutoff=0.6)
    return lower[close[0]] if close else None


def unknown_group_error(name: str, defined: Sequence[str], within: str = "") -> SelectionError:
    """"unknown residue group 'Aptamr' (defined: Aptamer, Target); did you mean 'Aptamer'?"."""
    guess = suggest(name, defined)
    where = f" in the selection of group {within!r}" if within else ""
    msg = f"unknown residue group {name!r}{where} (defined: {', '.join(defined) or 'none'})"
    if guess:
        msg += f"; did you mean {guess!r}?"
    return SelectionError(msg, kind="unknown_group", token=name, suggestion=guess or "", group=within)


def referenced_groups(text: str, groups: Sequence) -> List[str]:
    """Names of the defined groups ``text`` uses, directly (``group:<Name>`` or a bare name) or
    through their selections."""
    by_name = {g.name: g for g in groups}
    seen: List[str] = []
    stack = [text or ""]
    while stack:
        t = stack.pop().strip()
        t = t[4:] if t.startswith("sel:") else t
        for name in ([t] if t in by_name else _GROUP_TOKEN.findall(t)):
            if name in by_name and name not in seen:
                seen.append(name)
                stack.append(by_name[name].selection)
    return seen


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
            raise SelectionError(f"invalid residue range item {raw!r}", kind="range", token=raw, text=text)
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
        raise SelectionError("empty residue range", kind="range", text=text)
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
        try:
            return f"({ranges_to_selection(group.residues, universe)})"
        except SelectionError as exc:  # name the group: the error may surface in any selection using it
            raise SelectionError(f"group {group.name!r}: {exc.message} in its residues {group.residues!r}",
                                 kind=exc.kind, token=exc.token, text=group.residues,
                                 group=group.name) from None
    raise SelectionError(f"group {group.name!r} has neither residues nor a selection", kind="missing",
                         token=group.name, group=group.name)


def expand_groups(text: str, groups: Sequence, universe=None, _depth: int = 0,
                  _chain: Tuple[str, ...] = ()) -> str:
    """Replace ``group:<Name>`` tokens by the group's selection."""
    if _depth > 8:
        raise SelectionError("group definitions are nested too deeply (circular reference?)", kind="nesting")
    by_name = {g.name: g for g in groups}

    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in by_name:
            raise unknown_group_error(name, list(by_name), _chain[-1] if _chain else "")
        if name in _chain:
            loop = " → ".join(f"group:{n}" for n in _chain[_chain.index(name):] + (name,))
            raise SelectionError(f"group definitions are nested in a circle: {loop}", kind="nesting",
                                 token=name, group=name)
        return expand_groups(group_selection(by_name[name], universe), groups, universe, _depth + 1,
                             _chain + (name,))

    return _GROUP_TOKEN.sub(repl, text)


def rename_group_tokens(text: str, old: str, new: str) -> str:
    """``text`` with every ``group:<old>`` token renamed to ``group:<new>``; a token naming another
    group (``group:<old>er``, ``group:<old>_2``) is read whole, as :func:`expand_groups` reads it."""
    return _GROUP_TOKEN.sub(lambda m: f"group:{new}" if m.group(1) == old else m.group(0), text)


def resolve_selection(text: str, groups: Sequence, universe=None) -> str:
    """Resolve a selection that may be a bare group name, ``sel:...`` or a selection."""
    text = (text or "").strip()
    if not text:
        raise SelectionError("empty selection (enter a group name or an MDAnalysis selection)",
                             kind="no_text")
    if text.startswith("sel:"):
        text = text[4:].strip()
    elif any(g.name == text for g in groups):
        text = f"group:{text}"
    return expand_groups(text, groups, universe)


def _keywords() -> List[str]:
    try:  # MDAnalysis' table of selection keywords (private; only used for suggestions)
        from MDAnalysis.core.selection import _SELECTIONDICT
        return list(_SELECTIONDICT) + ["and", "or"]
    except ImportError:  # pragma: no cover
        return []


def _syntax_hint(detail: str, groups: Sequence) -> Tuple[str, str]:
    """(offending token, hint) for an MDAnalysis error message."""
    names = [g.name for g in groups]
    m = re.search(r"Unknown selection token: '([^']*)'", detail)
    if m:
        token = m.group(1)
        if token == "None":
            return "", "the selection ends where MDAnalysis expects more: a value, or a clause after and / or"
        group = suggest(token, names)
        if group:
            return token, f"to use the residue group write group:{group}"
        keyword = suggest(token, _keywords())
        return token, f"did you mean {keyword!r}?" if keyword else f"{token!r} is not a selection keyword"
    m = re.search(r"Failed to find group: ([^'\s]+)", detail)
    if m:
        group = suggest(m.group(1), names)
        return m.group(1), (f"write group:{group}, with a colon, to use the residue group" if group
                            else "residue groups are written group:<Name>")
    if "Expected: ')'" in detail:
        return "(", "a parenthesis is not closed"
    m = re.search(r"Failed to parse value: ([^'\s]+)", detail)
    return (m.group(1), "") if m else ("", "")


def syntax_error(text: str, expanded: str, detail: str, groups: Sequence = ()) -> SelectionError:
    """MDAnalysis rejected ``expanded``, the expansion of ``text``: the error shows both and MDAnalysis'
    own message (and a hint when the offending token looks like a group name or a misspelt keyword)."""
    shown = (text or "").strip()
    token, hint = _syntax_hint(detail, groups)
    msg = f"invalid selection {shown!r}"
    if expanded.strip() not in (shown, shown[4:].strip()):
        msg += f" (expanded: {expanded!r})"
    msg += f": MDAnalysis says: {detail}"
    if hint:
        msg += f" ({hint})"
    return SelectionError(msg, kind="syntax", token=token, text=shown, expanded=expanded, detail=detail,
                          suggestion=hint)


_parser = None
_parser_lock = threading.Lock()


def check_selection(text: str, groups: Sequence = ()) -> str:
    """Expand ``text`` and let MDAnalysis parse it, without a structure; returns the expansion or raises
    :class:`SelectionError`.  What needs a structure (attributes the topology lacks, matching nothing)
    is found only when the selection is applied.  A private parser keeps this safe while a background
    job selects with MDAnalysis' shared one."""
    global _parser
    expanded = resolve_selection(text, groups)
    try:
        from MDAnalysis.core.selection import SelectionParser
        from MDAnalysis.exceptions import SelectionError as MDASelectionError
    except ImportError:  # pragma: no cover - a dependency; without it only the syntax check is skipped
        return expanded
    with _parser_lock:
        if _parser is None:
            _parser = SelectionParser()
        try:
            _parser.parse(expanded, {})
        except MDASelectionError as exc:
            raise syntax_error(text, expanded, str(exc), groups) from None
        except Exception:  # noqa: BLE001 - anything else is decided when the selection is applied
            pass
    return expanded


def _clauses(text: str) -> List[str]:
    """Top-level parts of a selection joined by and / or (outside parentheses)."""
    parts, depth, start = [], 0, 0
    for m in re.finditer(r"[()]|\b(?:and|or)\b", text):
        if m.group(0) == "(":
            depth += 1
        elif m.group(0) == ")":
            depth -= 1
        elif depth == 0:
            parts.append(text[start:m.start()].strip())
            start = m.end()
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _definition(group) -> str:
    if group.selection.strip():
        return f"selection {group.selection.strip()!r}"
    return f"residues {group.residues.strip()!r}"


def explain_empty(universe, text: str, groups: Sequence) -> Tuple[str, str]:
    """Why ``text`` selects no atoms: (the empty group or clause, explanation).  Each group the text
    uses is tried on its own, then each top-level and / or part."""
    shown = (text or "").strip()
    body = shown[4:].strip() if shown.startswith("sel:") else shown
    by_name = {g.name: g for g in groups}
    names = [body] if body in by_name else list(dict.fromkeys(_GROUP_TOKEN.findall(body)))
    empty = []
    for name in names:
        try:
            if universe.select_atoms(expand_groups(f"group:{name}", groups, universe)).n_atoms == 0:
                empty.append(name)
        except Exception:  # noqa: BLE001 - diagnostics only
            continue
    if empty:
        why = "; ".join(f"group {n!r} ({_definition(by_name[n])}) matches no atoms" for n in empty)
        resids = universe.residues.resids
        if len(resids) and any(not by_name[n].selection.strip() for n in empty):
            why += f" - the structure's residues are numbered {resids.min()}-{resids.max()}"
        return empty[0], why
    clauses = [] if body in by_name else _clauses(body)
    if len(clauses) < 2:
        return body, ""
    none = []
    for clause in clauses:
        try:
            if universe.select_atoms(resolve_selection(clause, groups, universe)).n_atoms == 0:
                none.append(clause)
        except Exception:  # noqa: BLE001 - a part may not stand alone ('around 5 ...')
            continue
    if none:
        return none[0], "; ".join(f"the part {c!r} matches no atoms" for c in none)
    return "", "each part matches atoms, but not in this combination (check 'and' / 'or')"


def select(universe, text: str, groups: Sequence = (), allow_empty: bool = False):
    """Select atoms, expanding group tokens and giving readable errors."""
    resolved = resolve_selection(text, groups, universe)
    try:
        ag = universe.select_atoms(resolved)
    except Exception as exc:  # MDAnalysis raises several exception types
        raise syntax_error(text, resolved, str(exc) or type(exc).__name__, groups) from exc
    if not allow_empty and ag.n_atoms == 0:
        token, why = explain_empty(universe, text, groups)
        raise SelectionError(f"selection {text.strip()!r} matched no atoms" + (f": {why}" if why else ""),
                             kind="empty", token=token, text=text.strip(), expanded=resolved)
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
                raise SelectionError(f"invalid frame step in {raw!r}", kind="frames", token=raw)
            frames.extend(range(a, b + 1, step))
            continue
        try:
            v = int(token)
        except ValueError as exc:
            raise SelectionError(f"invalid frame specification {raw!r}", kind="frames", token=raw) from exc
        frames.append(v + n_frames if v < 0 else v)
    bad = [f for f in frames if f < 0 or f >= n_frames]
    if bad:
        raise SelectionError(f"frame(s) {bad} outside trajectory (0-{n_frames - 1})", kind="frames")
    return sorted(set(frames))


def parse_time_spec(text: str, default_unit: str = "ps") -> List[float]:
    """Parse ``"1.5ns, 2000ps, 3"`` into times in picoseconds."""
    times: List[float] = []
    if not text or not text.strip():
        return times
    if default_unit == "frame":
        raise SelectionError("time specification needs a time unit when the display unit is 'frame'",
                             kind="times")
    for raw in re.split(r"[,;\s]+", text.strip()):
        if not raw:
            continue
        m = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)", raw)
        if not m:
            raise SelectionError(f"invalid time {raw!r}", kind="times", token=raw)
        unit = (m.group(2) or default_unit).lower()
        if unit not in _TIME_UNITS_PS:
            raise SelectionError(f"unknown time unit {unit!r} in {raw!r}", kind="times", token=raw)
        times.append(float(m.group(1)) * _TIME_UNITS_PS[unit])
    return times


def time_unit_factor(unit: str) -> float:
    """Divide picoseconds by this factor to obtain ``unit``."""
    if unit not in _TIME_UNITS_PS:
        raise ValueError(f"unknown time unit {unit!r}")
    return _TIME_UNITS_PS[unit]
