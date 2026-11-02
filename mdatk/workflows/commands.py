"""The commands a generated script runs, as the script wizard's Commands tab shows them.

:func:`command_lines` reads a bash script as MDATK's generators write it and keeps what does the step's
work: the program invocations and the helper calls.  It leaves out the set-up and the bookkeeping around
them: comments, ``set``, ``cd``, the environment (the *Environment* block of :func:`.base.script_header`,
conda, modules, ``source``, ``export``), variable assignments, function definitions and calls of the
script's own functions (their bodies are read like the rest of the script), ``if`` / ``for`` / ``while``
/ ``case`` structure (a case's patterns too), checks (``[ … ]``, ``test``, ``command -v``, ``grep -q``, so the :func:`.base.require_file` lines
too), file bookkeeping (``rm``, ``mkdir``, ``mv``, ``cp``, ``ln``), messages (``echo``, ``printf``),
``exit``, and the error handling after a command (``|| { echo "ERROR: …" >&2; exit 1; }``, ``|| true``).

Continued lines (``\\``) are joined, a here-document stays with its command, and a variable the script
sets once to fixed text is shown with that text (``$PMEMD -O …`` as ``pmemd.cuda -O …``), so each line
reads as the command that runs.  The script is only read: nothing in it is run.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# First words of the commands that set up or keep the books, not the step's work
SETUP_WORDS = frozenset((
    "set", "shopt", "cd", "pushd", "popd", "umask", "ulimit", "trap",  # the shell itself
    "export", "unset", "local", "declare", "readonly", "typeset", "read", "shift",  # variables
    "source", ".", "module", "ml", "conda", "eval", "spack",  # the environment
    "[", "[[", "test", "command", "type", "hash",  # checks
    "rm", "rmdir", "mkdir", "mv", "cp", "ln", "touch", "chmod",  # file bookkeeping
    "echo", "printf",  # messages
    "exit", "return", "break", "continue", "true", "false", ":", "wait", "sleep",  # flow
    "for", "select", "case",  # loop and case heads: a variable and a list follow
))
# Reserved words that only shape the script: the words after them are read on
KEYWORDS = frozenset(("if", "then", "elif", "else", "fi", "while", "until", "do", "done", "esac", "{", "}",
                      "!", "time"))
ENVIRONMENT = "# Environment"  # script_header's heading of the environment set-up (up to a blank line)
_OPS = ("&&", "||", ";;", ";", "|", "&", "(", ")")
_DECLARE = ("export", "local", "declare", "readonly", "typeset")
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\])?(\+?)=")
_REDIRECT = re.compile(r"(?:\d*|&)(?:>>|>&|>\||>|<<<|<<-|<<|<&|<)(.*)", re.S)
_FUNCTION = re.compile(r"(?:function\s+([A-Za-z_][\w:.-]*)\s*(?:\(\s*\))?|([A-Za-z_][\w:.-]*)\s*\(\s*\))"
                       r"\s*\{?")  # function name {, name() {
_HEREDOC = re.compile(r"<<(-?)\s*(?:'([^']*)'|\"([^\"]*)\"|(\\?)([^\s;&|<>()'\"]+))")
_LEAD = re.compile(r"(?:(?:if|elif|while|until|then|else|do|time|!)\s+)+")
_VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_QUIET = re.compile(r"-[A-Za-z]*q|--(?:quiet|silent)$")

Heredoc = Tuple[int, List[str], str, bool]  # where its << is, body lines, delimiter, quoted (no expansion)


def command_lines(script: str) -> List[str]:
    """The commands ``script`` runs, in script order (see the module docstring); a command with a
    here-document holds its lines and the delimiter, separated by newlines."""
    entries = _logical_lines(script)
    functions, values = _definitions(entries)
    out: List[str] = []
    in_environment = False
    cases = _Cases()
    for line in entries:
        stripped = line.text.strip()
        if in_environment:  # the environment set-up runs up to the next blank line
            in_environment = bool(stripped)
            continue
        if stripped == ENVIRONMENT:
            in_environment = True
            continue
        for start, end, shown in _statements(line, functions, cases):
            shown = _expand(shown, values)
            for at, body, delim, quoted in line.docs:
                if start <= at < end:
                    lines = body if quoted else [_expand(b, values, quotes=False) for b in body]
                    shown += "\n" + "\n".join(lines + [delim])
            out.append(shown)
    return out


# ---------------------------------------------------------------------------
# reading the shell syntax
# ---------------------------------------------------------------------------


class _Line(NamedTuple):
    """One command of the script, maybe over several lines, as read by :func:`_logical_lines`."""

    text: str
    top: Set[int]  # the indices of text that are shell syntax at the top level (see _scan)
    comment: int  # where a comment starts, -1 for none
    docs: List[Heredoc]  # the here-documents it starts


def _scan(text: str, start: int, stack: List[str], top: Set[int]) -> int:
    """Read ``text`` from ``start`` on, in the contexts ``stack`` holds open there: add to ``top`` the indices
    that are shell syntax at the top level (outside quotes, ``$( )``, ``${ }``, back-quotes, arrays and
    escapes) and leave in ``stack`` the contexts still open at the end ("'", '"', '(' for ``$( )``,
    ``$(( ))`` and arrays, '{' for ``${ }``, '`').  Returns where a comment starts, -1 for none."""
    i, n = start, len(text)
    while i < n:
        c = text[i]
        ctx = stack[-1] if stack else ""
        if ctx == "'":
            end = text.find("'", i)
            if end < 0:
                return -1
            stack.pop()
            i = end + 1
            continue
        if c == "\\":
            i += 2
            continue
        if ctx == "`":
            if c == "`":
                stack.pop()
            i += 1
            continue
        if c == "'" and ctx != '"':
            stack.append("'")
            i += 1
            continue
        if text.startswith(("$(", "${"), i):
            stack.append(text[i + 1])
            i += 2
            continue
        if c == "`":
            stack.append("`")
        elif ctx == '"':
            if c == '"':
                stack.pop()
        elif c == '"':
            stack.append('"')
        elif ctx == "{":
            if c == "}":
                stack.pop()
        elif c == "#" and (i == 0 or text[i - 1] in " \t\n;&|()"):
            if not stack:
                return i
            end = text.find("\n", i)  # a comment inside $( ): up to the end of its line
            if end < 0:
                return -1
            i = end
            continue
        elif c == "(" and not stack and text.startswith("()", i):
            i += 2  # name(): a function definition
            continue
        elif c == "(" and (stack or (i > 0 and text[i - 1] == "=")):
            stack.append("(")  # inside $( ), or an array NAME=( … )
        elif c == ")" and ctx == "(":
            stack.pop()
        elif not stack:
            top.add(i)
        i += 1
    return -1


def _continued(text: str) -> bool:
    """Whether ``text`` ends with a backslash that continues it on the next line."""
    return (len(text) - len(text.rstrip("\\"))) % 2 == 1


def _logical_lines(script: str) -> List[_Line]:
    """The script's commands: continued lines joined with a space, a quote or bracket that spans lines kept
    with its newlines, the bodies of the here-documents taken out.  Each line is read once (a list of
    hundreds of models over as many lines stays quick)."""
    lines = script.split("\n")
    out: List[_Line] = []
    k = 0
    while k < len(lines):
        text, stack, top = lines[k], [], set()
        k += 1
        comment = _scan(text, 0, stack, top)
        while k < len(lines):
            tail = len(text.rstrip())
            if stack:  # a quote or bracket goes on
                start = len(text)
                text += "\n" + lines[k]
            elif comment < 0 and _continued(text):
                start = len(text[:-1].rstrip())
            elif comment < 0 and text.endswith(("&&", "||", "|"), 0, tail) and tail - 1 in top:
                start = tail  # a list or pipeline that goes on in the next line
            else:
                break
            if not stack:
                top.difference_update(range(start, len(text)))
                text = text[:start] + " " + lines[k].strip()
            k += 1
            comment = _scan(text, start, stack, top)
        docs: List[Heredoc] = []
        for at, delim, strip_tabs, quoted in _heredocs(text, top, comment):
            body: List[str] = []
            while k < len(lines):
                line = lines[k]
                k += 1
                if (line.lstrip("\t") if strip_tabs else line) == delim:
                    break
                body.append(line)
            docs.append((at, body, delim, quoted))
        out.append(_Line(text, top, comment, docs))
    return out


def _heredocs(text: str, top: Set[int], comment: int) -> List[Tuple[int, str, bool, bool]]:
    """(where its << is, delimiter, tabs stripped (<<-), quoted) of each here-document a line starts."""
    out = []
    end = comment if comment >= 0 else len(text)
    for i in sorted(top):
        if i >= end:
            break
        if (text.startswith("<<", i) and i + 1 in top and not text.startswith("<<<", i)
                and not (i > 0 and text[i - 1] == "<")):
            m = _HEREDOC.match(text, i)
            if m:
                if m.group(5) is not None:
                    delim, quoted = m.group(5), bool(m.group(4))
                else:
                    delim, quoted = (m.group(2) if m.group(2) is not None else m.group(3)), True
                out.append((i, delim, m.group(1) == "-", quoted))
    return out


def _segments(text: str, top: Set[int], start: int, end: int) -> List[Tuple[int, int, str]]:
    """The simple commands of ``text[start:end]``: (start, end, the operator after it: && || ;; ; | & ( )
    or "")."""
    out = []
    i = start
    while i < end:
        if i in top:
            op = next((o for o in _OPS
                       if text.startswith(o, i) and all(j in top for j in range(i, i + len(o)))), "")
            if op == "&" and ((i > 0 and text[i - 1] in "<>") or text.startswith("&>", i)):
                op = ""  # 2>&1, >&2, &> file
            elif op == "|" and i > 0 and text[i - 1] == ">":
                op = ""  # >| file
            if op:
                out.append((start, i, op))
                i += len(op)
                start = i
                continue
        i += 1
    out.append((start, end, ""))
    return out


def _words(text: str, start: int, end: int, top: Set[int]) -> List[str]:
    words, cur = [], start
    for i in range(start, end):
        if i in top and text[i] in " \t\n":
            if i > cur:
                words.append(text[cur:i])
            cur = i + 1
    if end > cur:
        words.append(text[cur:end])
    return words


def _kind(words: List[str], functions: Set[str]) -> str:
    """'work', 'setup' or '' (reserved words and redirections only) for the words of one simple command."""
    ws = list(words)
    while ws and ws[0] in KEYWORDS:
        ws.pop(0)
    assigned = False
    while ws and _ASSIGN.match(ws[0]):
        ws.pop(0)
        assigned = True
    while ws:  # redirections before the command (e.g. after the ')' of a subshell)
        m = _REDIRECT.fullmatch(ws[0])
        if not m:
            break
        ws.pop(0)
        if not m.group(1) and ws:
            ws.pop(0)  # '> file': the target is the next word
    if not ws:
        return "setup" if assigned else ""
    first = ws[0]
    if first in SETUP_WORDS or first in functions:
        return "setup"
    if first == "grep" and any(_QUIET.match(w) for w in ws[1:]):
        return "setup"  # a test, like [ ]
    return "work"


# ---------------------------------------------------------------------------
# what a line shows
# ---------------------------------------------------------------------------


def _parse(line: _Line) -> Tuple[Optional[str], List[Tuple[int, int, str]]]:
    """(the function whose head starts the line or None, the simple commands) of one logical line; a
    comment is left out."""
    text = line.text
    pos = len(text) - len(text.lstrip())
    m = _FUNCTION.match(text, pos)
    name = None
    if m:
        name, pos = m.group(1) or m.group(2), m.end()  # the body, if on this line, is read like any line
    return name, _segments(text, line.top, pos, line.comment if line.comment >= 0 else len(text))


class _Cases:
    """The ``case`` statements open where the script has been read (one may span lines)."""

    def __init__(self) -> None:
        self.open = 0  # case statements not yet closed by esac
        self.pattern = False  # a pattern comes next: after 'case WORD in' and after ';;'

    def skip(self, words: List[str], op: str) -> bool:
        """Whether a simple command (its words, the operator after it) is a case head or a pattern
        ('case "$f" in', '/*' before ')', 'a' before '|', the '(' some write before a pattern): they run
        nothing.  Counts the statements opened and closed."""
        ws = list(words)
        while ws and ws[0] in KEYWORDS and ws[0] != "esac":
            ws.pop(0)
        if ws[:1] == ["esac"]:
            self.open = max(0, self.open - 1)
            self.pattern = False
            return False
        if len(ws) >= 3 and ws[0] == "case" and ws[2] == "in":
            self.open += 1
            self.pattern = True
            ws = ws[3:]  # the first pattern may follow on the same line
            if not ws and op not in (")", "|", "("):
                return True
        if self.pattern and (op in (")", "|") or (op == "(" and not ws)):
            self.pattern = op != ")"
            return True
        if op == ";;" and self.open:
            self.pattern = True
        return False


def _statements(line: _Line, functions: Set[str],
                cases: Optional[_Cases] = None) -> List[Tuple[int, int, str]]:
    """The commands of one logical line that do the step's work, as (start, end, what is shown): split
    where ';' or '&' ends a command outside { } and ( ), stripped of control words, case patterns and the
    error handling after the command (``cases``: the case statements open before the line)."""
    text, top = line.text, line.top
    _name, segments = _parse(line)
    cases = _Cases() if cases is None else cases
    out = []
    group: List[Tuple[int, int, str, str]] = []  # (start, end, operator after, kind)
    depth = 0
    for start, end, op in segments:
        words = _words(text, start, end, top)
        if depth <= 0 and cases.skip(words, op):  # the ')' after a pattern ends no subshell
            group, depth = [], 0
            continue
        depth += words.count("{") - words.count("}") + (op == "(") - (op == ")")
        group.append((start, end, op, _kind(words, functions)))
        if depth <= 0 and op in (";", ";;", "&", ""):
            shown = _shown(text, group)
            if shown is not None:
                out.append(shown)
            group, depth = [], 0
    return out


def _shown(text: str, group: List[Tuple[int, int, str, str]]) -> Optional[Tuple[int, int, str]]:
    """(start, end, text) of one command (a pipeline or list) when it does work, else None."""
    if not any(kind == "work" for *_x, kind in group):
        return None
    end = group[-1][1]
    for j in range(len(group) - 1, 0, -1):  # what follows the last program: error handling
        if group[j][3] == "work":
            break
        if group[j - 1][2] in ("||", "&&"):
            end = group[j - 1][1]
    shown = text[group[0][0]:end].strip()
    lead = _LEAD.match(shown)  # if, then, do, ! ... before the command
    shown = shown[lead.end():] if lead else shown
    if end == group[-1][1] and group[-1][2] == "&":
        shown += " &"  # in the background
    return group[0][0], end, shown


def _constant(value: str) -> Optional[str]:
    """The value of an assignment when it is fixed text (no expansion, no command), else None."""
    if re.fullmatch(r'"[^"$`\\]*"', value) or re.fullmatch(r"'[^'$]*'", value):
        return value[1:-1]
    if re.fullmatch(r"[^\s\"'$`\\;&|<>(){}]*", value):
        return value
    return None


def _definitions(entries: List[_Line]) -> Tuple[Set[str], Dict[str, str]]:
    """(the functions the script defines, {variable: value} of the variables it sets once to fixed
    text)."""
    functions: Set[str] = set()
    seen: Dict[str, List[Optional[str]]] = {}
    for line in entries:
        name, segments = _parse(line)
        if name:
            functions.add(name)
        for start, end, _op in segments:
            words = _words(line.text, start, end, line.top)
            while words and words[0] in KEYWORDS:
                words.pop(0)
            if not words:
                continue
            if words[0] in _DECLARE:
                words = [w for w in words[1:] if not w.startswith("-")]
            elif words[0] in ("for", "select") and len(words) > 1:
                seen.setdefault(words[1], []).append(None)
                continue
            elif words[0] == "read":
                for w in words[1:]:
                    if not w.startswith("-"):
                        seen.setdefault(w, []).append(None)
                continue
            for w in words:
                a = _ASSIGN.match(w)
                if not a:
                    break
                fixed = None if a.group(2) or a.group(3) else _constant(w[a.end():])
                seen.setdefault(a.group(1), []).append(fixed)
    values = {name: v[0] for name, v in seen.items() if len(v) == 1 and v[0] is not None}
    return functions, values


def _expand(text: str, values: Dict[str, str], quotes: bool = True) -> str:
    """``text`` with ``$NAME`` / ``${NAME}`` of ``values`` replaced, except inside single quotes (``quotes``
    False: a here-document's body, where quotes are plain characters)."""
    if not values or "$" not in text:
        return text
    out: List[str] = []
    quote = ""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote == "'":
            if c == "'":
                quote = ""
        elif c == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        elif quotes and c == "'" and not quote:
            quote = "'"
        elif quotes and c == '"':
            quote = "" if quote else '"'
        elif c == "$":
            m = _VAR.match(text, i)
            name = (m.group(1) or m.group(2)) if m else None
            if name in values:
                out.append(values[name])
                i = m.end()
                continue
        out.append(c)
        i += 1
    return "".join(out)
