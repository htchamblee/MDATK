"""TOML reading and commented TOML writing for MDATK's project files.

Reading uses the standard library (:mod:`tomllib`, Python 3.11+) or its
``tomli`` backport.  Writing is done here because every entry of an MDATK file
carries a descriptive comment, which general-purpose TOML writers do not emit.

The writer takes plain data (dicts, lists, str, int, float, bool, datetime and
None) and a ``doc(path, stack)`` callback that returns the comment for an entry;
``path`` is the tuple of keys from the root (list indices omitted) and ``stack``
the chain of enclosing tables.  ``None`` has no TOML representation: such
entries are written as a commented-out line, so reading the file back leaves
them at their default (all of MDATK's optional settings default to ``None``).
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 / 3.10
    import tomli as _toml  # type: ignore[no-redef]

TOMLDecodeError = _toml.TOMLDecodeError
DocFn = Callable[[Tuple[str, ...], Sequence[Dict[str, Any]]], Optional[str]]

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")
_WIDTH = 100  # comment wrap width
_INLINE_MAX = 96  # longer arrays are written one element per line


def loads(text: str) -> Dict[str, Any]:
    return _toml.loads(text)


def load(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return _toml.load(fh)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


def key(name: str) -> str:
    return name if _BARE.match(name) else quote(name)


def _scalar(text: str) -> str:
    """Lone surrogates (undecodable file names) have no TOML form: U+FFFD stands in for them."""
    return "".join("\ufffd" if 0xD800 <= ord(c) <= 0xDFFF else c for c in text)


def quote(text: str) -> str:
    out = []
    for ch in _scalar(text):
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _multiline(text: str) -> str:
    """A multi-line basic string; the newline after the opening quotes is not part of it."""
    body = []
    for ch in _scalar(text):
        code = ord(ch)
        if ch == "\\":
            body.append("\\\\")
        elif ch == '"':
            body.append('\\"')  # never three quotes in a row
        elif ch in "\n\t":
            body.append(ch)
        elif ch == "\r":
            body.append("\\r")
        elif code < 0x20 or code == 0x7F:
            body.append(f"\\u{code:04X}")
        else:
            body.append(ch)
    return '"""\n' + "".join(body) + '"""'


def value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        text = repr(v)
        return text if any(c in text for c in ".eE") else text + ".0"
    if isinstance(v, str):
        return _multiline(v) if "\n" in v else quote(v)
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return v.isoformat()
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [f"{key(str(k))} = {value(x)}" for k, x in v.items() if x is not None]
        return "{ " + ", ".join(items) + " }"
    if isinstance(v, (list, tuple)):
        items = [value(x) for x in v if x is not None]
        flat = "[" + ", ".join(items) + "]"
        if len(flat) <= _INLINE_MAX and "\n" not in flat:
            return flat
        return "[\n" + "".join(f"    {x},\n" for x in items) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _is_table(v: Any) -> bool:
    return isinstance(v, dict)


def _is_table_array(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and bool(v) and all(isinstance(x, dict) for x in v)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


class Writer:
    """Writes nested tables with a comment above every entry (see the module docstring)."""

    def __init__(self, doc: Optional[DocFn] = None):
        self.doc = doc or (lambda path, stack: None)
        self.lines: List[str] = []

    def comment(self, text: Optional[str], indent: str = "") -> None:
        if not text:
            return
        for para in str(text).splitlines() or [""]:
            wrapped = textwrap.wrap(para, _WIDTH - len(indent) - 2) or [""]
            self.lines += [f"{indent}# {w}".rstrip() for w in wrapped]

    def blank(self) -> None:
        if self.lines and self.lines[-1] != "":
            self.lines.append("")

    def document(self, data: Dict[str, Any], header: Sequence[str] = ()) -> str:
        for line in header:
            self.lines.append(f"# {line}".rstrip() if line else "#")
        if header:
            self.lines.append("")
        self._body((), data, [data])
        return "\n".join(self.lines).rstrip("\n") + "\n"

    def table(self, path: Tuple[str, ...], data: Dict[str, Any], stack: List[Dict[str, Any]],
              array: bool = False, first: bool = True) -> None:
        self.blank()
        if first:
            self.comment(self.doc(path, stack))
        name = ".".join(key(p) for p in path)
        self.lines.append(f"[[{name}]]" if array else f"[{name}]")
        self._body(path, data, stack + [data])

    def _body(self, path: Tuple[str, ...], data: Dict[str, Any], stack: List[Dict[str, Any]]) -> None:
        scalars = [(k, v) for k, v in data.items() if not _is_table(v) and not _is_table_array(v)]
        tables = [(k, v) for k, v in data.items() if _is_table(v) and v]
        empty = [(k, v) for k, v in data.items() if _is_table(v) and not v]
        arrays = [(k, v) for k, v in data.items() if _is_table_array(v)]
        for k, v in scalars + empty:
            here = path + (str(k),)
            text = self.doc(here, stack)
            if v is None:
                self.comment(text)
                self.lines.append(f"# {key(str(k))} =   (not set)")
                continue
            self.comment(text)
            self.lines.append(f"{key(str(k))} = {value(v)}")
        for k, v in tables:
            self.table(path + (str(k),), v, stack)
        for k, v in arrays:
            for i, item in enumerate(v):
                self.table(path + (str(k),), item, stack, array=True, first=i == 0)


def dumps(data: Dict[str, Any], doc: Optional[DocFn] = None, header: Sequence[str] = ()) -> str:
    return Writer(doc).document(data, header)
