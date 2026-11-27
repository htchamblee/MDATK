"""Residue-group and selection helpers.

Selections in MDATK are ordinary MDAnalysis selection strings with one
extension: ``group:<Name>`` expands to the selection of a residue group defined
in the project (e.g. ``"group:Aptamer and name P"``).  Groups are defined by
residue ranges (``"1-40, 45, A:50-60"``) or by a raw selection string.
"""

from __future__ import annotations

import difflib
import re
import sys
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .problems import Problem, describe

_GROUP_TOKEN = re.compile(r"group:([A-Za-z0-9_.+\-']+)")
# what is written after "group:", up to a space or a parenthesis: a token when it has the characters above only
_GROUP_WORD = re.compile(r"group:([^\s()]+)")
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
    "group_name", "syntax", "empty", "nesting", "range", "missing", "no_text" or "" for other problems) and carries
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


def groups_by_name(groups: Sequence) -> dict:
    """The groups by their names.  Of several groups with one name the first is meant, as by
    :meth:`mdatk.config.Project.group` (validation refuses such a project, but ``info`` and previews select
    in it all the same)."""
    by_name: dict = {}
    for g in groups:
        by_name.setdefault(g.name, g)
    return by_name


def group_name_error(name: str, within: str = "") -> SelectionError:
    """``group:<name>`` with characters a group name cannot have (a space, an umlaut, a comma)."""
    where = f" in the selection of group {within!r}" if within else ""
    return SelectionError(f"group name {name!r}{where} may only contain letters, digits, '_', '.' and '-' "
                          "(no spaces)", kind="group_name", token=name, group=within)


def referenced_groups(text: str, groups: Sequence) -> List[str]:
    """Names of the defined groups ``text`` uses, directly (``group:<Name>`` or a bare name) or
    through their selections."""
    by_name = groups_by_name(groups)
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
    by_name = groups_by_name(groups)

    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        within = _chain[-1] if _chain else ""
        if not _GROUP_TOKEN.fullmatch(m.group(0)):
            # the token stopped at the first other character: 'group:Ünï' went to MDAnalysis as it was
            # ("Unknown selection token ... did you mean 'group'?"), 'group:Aptamer,Target' as '(...),Target'
            raise group_name_error(name, within)
        if name not in by_name:
            # a defined group whose name has a space ('Bad name') is read up to it: say what is wrong with
            # the name instead of "unknown residue group 'Bad'"
            longer = [n for n in by_name if n != name and text.startswith(n, m.start(1))
                      and not _GROUP_TOKEN.fullmatch(f"group:{n}")]
            if longer:
                raise group_name_error(max(longer, key=len), within)
            raise unknown_group_error(name, list(by_name), within)
        if name in _chain:
            loop = " → ".join(f"group:{n}" for n in _chain[_chain.index(name):] + (name,))
            raise SelectionError(f"group definitions are nested in a circle: {loop}", kind="nesting",
                                 token=name, group=name)
        return expand_groups(group_selection(by_name[name], universe), groups, universe, _depth + 1,
                             _chain + (name,))

    return _GROUP_WORD.sub(repl, text)


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
    if not hint:  # 'resid 1 AND name P': AND is read as a value, and the error names what follows it
        upper = next((t for t in expanded.replace("(", " ").replace(")", " ").split()
                      if t != t.lower() and t.lower() in ("and", "or", "not")), "")
        if upper:
            token, hint = upper, f"keywords are lower case: did you mean {upper.lower()!r}?"
    msg = f"invalid selection {shown!r}"
    if expanded.strip() not in (shown, shown[4:].strip()):
        msg += f" (expanded: {expanded!r})"
    msg += f": MDAnalysis says: {detail}"
    if hint:
        msg += f" ({hint})"
    return SelectionError(msg, kind="syntax", token=token, text=shown, expanded=expanded, detail=detail,
                          suggestion=hint)


# ---------------------------------------------------------------------------
# Checks before MDAnalysis parses a selection
# ---------------------------------------------------------------------------
# MDAnalysis accepts a few malformed clauses and then selects other atoms than written (a reversed 'prop'
# comparison gets the negated operator, text glued to a 'prop' operator or behind a float range is
# dropped, 'atom' takes keywords as names), and reports others with Python's own messages.  _checked()
# reads the expanded selection token by token as MDAnalysis' parser does, rejects those clauses and
# rewrites the ones MDAnalysis would misread.

_PROP_OPS = ("<", ">", "<=", ">=", "==", "!=")
_PROP_MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
_PROP_CLAUSE = re.compile(r"([^<>=!]+)([<>=!]+)([^<>=!]+)")
_MDA_FLOAT = re.compile(r"-?\d*\.?\d*(?:e[-+]?\d+)?")  # a bound of a float range as MDAnalysis reads it
_FLOAT_RANGE = re.compile(r"(\S*?)\s*(?::| to )\s*(\S*)")


class _Rejected(Exception):
    """A clause :func:`_checked` does not hand to MDAnalysis."""

    def __init__(self, message: str, token: str = ""):
        super().__init__(message)
        self.token = token


def _rejected_error(text: str, expanded: str, exc: _Rejected) -> SelectionError:
    shown = (text or "").strip()
    msg = f"invalid selection {shown!r}"
    if expanded.strip() not in (shown, shown[4:].strip()):
        msg += f" (expanded: {expanded!r})"
    return SelectionError(f"{msg}: {exc}", kind="syntax", token=exc.token, text=shown, expanded=expanded,
                          detail=str(exc))


def _number(token: Optional[str]) -> Optional[float]:
    try:
        return float(token)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float_values(keyword: str, values: List[str], join) -> List[str]:
    """The value tokens of a float keyword (``mass 12:+13 20 to inf``) with every range written so that
    MDAnalysis reads both bounds: its pattern for a bound knows no ``+``, ``E`` or ``inf`` and ignores
    what it cannot read (``1.2e1:1.3E1`` is taken as 12 to 1.3)."""
    if not any(":" in v or v == "to" for v in values):
        return values
    try:
        joined = join(list(values))
    except Exception:  # noqa: BLE001 - MDAnalysis reports it
        return values
    out: List[str] = []
    for value in joined:
        m = _FLOAT_RANGE.fullmatch(value)
        if not m:
            out += value.split()
            continue
        bounds = [_number(b) for b in m.groups()]
        if any(b is None or b != b for b in bounds):
            raise _Rejected(f"{keyword}: cannot read the range {value!r} (write two numbers, "
                            f"e.g. '{keyword} 1.5:2')", value)
        if all(_MDA_FLOAT.fullmatch(b) for b in m.groups()):
            out += value.split()  # as written
        else:  # ±inf: the largest number
            out.append(":".join(repr(max(-sys.float_info.max, min(sys.float_info.max, b))) for b in bounds))
    return out


def _checked(expanded: str) -> str:
    """``expanded`` as it is handed to MDAnalysis; raises :class:`_Rejected` for a malformed ``prop``,
    ``atom``, ``around`` or ``point`` clause or float range."""
    try:
        from MDAnalysis.core import selection as mda
        is_keyword, properties, join = mda.is_keyword, mda.PropertySelection.props, mda.join_separated_values
        float_keywords = {t for t, c in mda._SELECTIONDICT.items() if issubclass(c, mda.FloatRangeSelection)}
    except (ImportError, AttributeError):  # pragma: no cover - without these MDAnalysis parses as before
        return expanded
    tokens = expanded.replace("(", " ( ").replace(")", " ) ").split()  # as MDAnalysis' parser splits
    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        i += 1
        if tok == "prop":
            if tokens[i:i + 1] == ["abs"]:
                out.append("abs")
                i += 1
            joined, m = "", None  # 'mass > 14', 'mass> 14', 'mass >14' or 'mass>14'
            for k in range(i, min(i + 3, len(tokens))):
                if tokens[k] in ("(", ")", "and", "or"):
                    break
                joined += tokens[k]
                m = _PROP_CLAUSE.fullmatch(joined)
                if m:
                    i = k + 1
                    break
            if not m:
                raise _Rejected("prop: expected a property, an operator and a value, e.g. 'prop mass > 14'",
                                "prop")
            name, op, value = m.groups()
            if op not in _PROP_OPS:
                raise _Rejected(f"invalid operator {op!r} in prop (use {', '.join(_PROP_OPS)})", op)
            if value in properties and name not in properties:
                # 'prop 6 < x': MDAnalysis swaps the two as well, but negates the operator (x >= 6)
                name, op, value = value, _PROP_MIRROR[op], name
            out += [name, op, value]
        elif tok == "atom":
            args = tokens[i:i + 3]
            keyword = next((a for a in args if is_keyword(a)), "")
            if len(args) < 3 or keyword or not re.fullmatch(r"[-+]?\d+", args[1]):
                hint = (f" (found the keyword {keyword!r}; write \\{keyword} for a segid or name that is "
                        "a keyword)" if keyword not in ("", "(", ")") else "")
                raise _Rejected(f"atom: expected segid resid name, e.g. 'atom A 12 CA'{hint}", keyword or "atom")
            out += [args[0].removeprefix("\\"), args[1], args[2].removeprefix("\\")]
            i += 3
        elif tok in ("around", "point"):
            n = 1 if tok == "around" else 4
            numbers = [_number(t) for t in tokens[i:i + n]]
            follows = tokens[i + n] if i + n < len(tokens) else None
            if len(numbers) < n or None in numbers or (tok == "around" and follows in (None, ")", "and", "or")):
                raise _Rejected("around: expected a distance and a selection, e.g. 'around 3.5 resname MG'"
                                if tok == "around" else
                                "point: expected x y z distance, e.g. 'point 0 0 0 5'", tok)
            if numbers[-1] < 0:
                raise _Rejected(f"{tok}: distance must not be negative", tokens[i + n - 1])
        elif tok in float_keywords:
            j = i
            while j < len(tokens) and not is_keyword(tokens[j]):
                j += 1
            out += _float_values(tok, tokens[i:j], join)
            i = j
    return " ".join(out)


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
        checked = _checked(expanded)
    except _Rejected as exc:
        raise _rejected_error(text, expanded, exc) from None
    try:
        from MDAnalysis.core.selection import SelectionParser
        from MDAnalysis.exceptions import SelectionError as MDASelectionError
    except ImportError:  # pragma: no cover - a dependency; without it only the syntax check is skipped
        return expanded
    with _parser_lock:
        if _parser is None:
            _parser = SelectionParser()
        try:
            _parser.parse(checked, {})
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
    by_name = groups_by_name(groups)
    names = [body] if body in by_name else list(dict.fromkeys(_GROUP_TOKEN.findall(body)))
    empty = []
    for name in names:
        try:
            if universe.select_atoms(_checked(expand_groups(f"group:{name}", groups, universe))).n_atoms == 0:
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
            if universe.select_atoms(_checked(resolve_selection(clause, groups, universe))).n_atoms == 0:
                none.append(clause)
        except Exception:  # noqa: BLE001 - a part may not stand alone ('around 5 ...')
            continue
    if none:
        return none[0], "; ".join(f"the part {c!r} matches no atoms" for c in none)
    return "", "each part matches atoms, but not in this combination (check 'and' / 'or')"


def select(universe, text: str, groups: Sequence = (), allow_empty: bool = False, periodic: bool = True):
    """Select atoms, expanding group tokens and giving readable errors.  Distance keywords (``around``,
    ``sphzone`` ...) are evaluated at the universe's current frame, with its periodic cell unless ``periodic``
    is false."""
    resolved = resolve_selection(text, groups, universe)
    try:
        ag = universe.select_atoms(_checked(resolved), periodic=periodic)
    except _Rejected as exc:
        raise _rejected_error(text, resolved, exc) from None
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
    """Upper-case element symbols for an AtomGroup: the topology's value, and for each atom the topology
    has none for (an Amber extra point among atoms with elements) the guess from its name.  An atom's
    element does not depend on the other atoms of the group."""
    n = atoms.n_atoms
    elements = np.full(n, "", dtype=object)
    if hasattr(atoms, "elements"):
        try:
            elements = np.array([e.strip().upper() if isinstance(e, str) else ""
                                 for e in np.asarray(atoms.elements, dtype=object)], dtype=object)
        except Exception:
            elements = np.full(n, "", dtype=object)
    unknown = np.flatnonzero(elements == "") if n else []
    if len(unknown):
        guessed = atoms[unknown]
        res_sizes = {}
        for res in guessed.residues:
            res_sizes[res.ix] = res.atoms.n_atoms
        elements[unknown] = [
            guess_element(a.name, a.resname, res_sizes.get(a.resindex, 2) == 1) for a in guessed
        ]
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
