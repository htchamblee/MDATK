"""MDATK project files (TOML).

One human-readable file holds a project's settings, its Project Planner
workflow, free-text notes and the log of the steps run in it::

    format = "mdatk-project"
    notes = "..."     # free-text project notes
    [inputs]          # analysis settings ...
    [workflow]        # planner steps ...
    # ---- step log ----
    [[log]]           # one entry per step run (appended, newest last)

Every entry carries a comment explaining it (taken from the parameter model in
:mod:`mdatk.config` and from the script generators' option descriptions).
Log entries are appended without rewriting the file; once a project file would
grow past :data:`LOG_SPLIT_BYTES` its log moves to ``<name>.log.toml`` next to
it, so the project stays quick to open in a text editor.

The same writer produces the per-run parameter log (``MDATK_params_log.toml``),
which can be opened as a project to replay that run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import functools
import inspect
import io
import logging
import os
import time
import threading
import tokenize
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import PROJECT_ABBREV, PROJECT_TITLE, __version__
from . import tomlio

log = logging.getLogger(__name__)

PROJECT_FORMAT = "mdatk-project"
PROJECT_FORMAT_VERSION = 1
PARAMS_LOG_FORMAT = "mdatk-params-log"
PROJECT_SUFFIX = ".mdatk.toml"
LOG_MARKER = "# ---- step log (entries are appended; newest last) ----"
LOG_SPLIT_BYTES = 50 * 1024 * 1024  # beyond this the log moves to a separate file
LOG_OUTPUT_CHARS = 20000  # program output kept per log entry (its tail)
RECENT_MAX = 10

_META_DOCS = {
    "format": f"File type marker of an {PROJECT_ABBREV} project; do not change.",
    "format_version": "Version of the project file layout.",
    "mdatk_version": f"{PROJECT_TITLE} version that wrote this file.",
    "saved": "When the settings were last saved.",
    "log_file": "File next to the project that holds its step log (\"\": the log is at the end of this "
                "file; it moves out once the project file would grow too large).",
}
_LOG_DOCS = {
    (): "One entry per step run in this project: analyses, renders, generated scripts that "
        f"{PROJECT_ABBREV} ran, PDB clean-ups, density fits.",
    ("time",): "When the step started.",
    ("step",): "What ran: 'render', 'workflow write', 'pdb clean', or the planner step number of a "
               f"generated script that {PROJECT_ABBREV} ran (e.g. '03').",
    ("title",): "Short human-readable description.",
    ("status",): "ok, failed, cancelled or waiting (for a script the step leaves to be run separately).",
    ("duration_s",): "Wall time in seconds.",
    ("command",): "Command line that was run (for generated scripts).",
    ("folder",): "Working folder of the command.",
    ("files",): "Files written by the step.",
    ("summary",): "Result in a few lines.",
    ("citations",): "Citation keys of the resources the step used (see the BibTeX export).",
    ("output",): f"Program output (the last {LOG_OUTPUT_CHARS} characters).",
}


# ---------------------------------------------------------------------------
# comments for every entry
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def field_comments(cls: type) -> Dict[str, str]:
    """``{field: comment}`` from the source of a dataclass: the comment on a field's line,
    or comment lines directly above it."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    names = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    out: Dict[str, str] = {}
    pending: List[str] = []
    current: Optional[str] = None  # field whose definition is being read
    at_start = True  # at the start of a logical line
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            text = tok.string.lstrip("#").strip()
            if at_start:  # a comment on its own line describes the next field
                pending.append(text)
            elif current:
                out[current] = (out.get(current, "") + " " + text).strip()
            continue
        if tok.type == tokenize.NEWLINE:
            current, at_start = None, True
            continue
        if tok.type in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        if at_start:
            current = tok.string if tok.type == tokenize.NAME and tok.string in names else None
            if current and pending:
                out[current] = " ".join(pending)
            pending = []
            at_start = False
    return out


def class_doc(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    if not doc or doc.startswith(cls.__name__ + "("):  # dataclass auto signature
        return ""
    return doc.split("\n\n")[0].replace("\n", " ")


def _humanise(name: str) -> str:
    return name.replace("_", " ").capitalize() + "."


def _unwrap(tp: Any) -> Any:
    """List[X] -> X, Optional[X] -> X."""
    while True:
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        if origin is typing.Union:
            tp = [a for a in args if a is not type(None)][0]
        elif origin in (list, List) and args:
            tp = args[0]
        else:
            return tp


def config_doc(path: Tuple[str, ...], stack: Sequence[Dict[str, Any]] = ()) -> Optional[str]:
    """Comment for an entry of the project tree (``path`` without list indices)."""
    from .config import Project

    if len(path) == 1 and path[0] in _META_DOCS:
        return _META_DOCS[path[0]]
    tp: Any = Project
    owner: Any = None
    for i, name in enumerate(path):
        tp = _unwrap(tp)
        if not dataclasses.is_dataclass(tp):
            return _free_dict_doc(path, i, stack)
        hints = typing.get_type_hints(tp)
        if name not in hints:
            return None
        owner, tp = tp, hints[name]
    if owner is None:
        return None
    sub = _unwrap(tp)
    if dataclasses.is_dataclass(sub):
        text = field_comments(owner).get(path[-1]) or class_doc(sub)
        return text or _humanise(path[-1])
    return field_comments(owner).get(path[-1]) or _humanise(path[-1])


_PANEL_OPTION_DOCS = {
    "mode": "Contact map: 'instant' (evolves with the movie), 'frame' (one fixed frame) or 'frequency' "
            "(fraction of frames in contact).",
    "frame": "Contact map in 'frame' mode: the frame shown (0-based).",
    "transpose": "Contact map: true puts the first group (e.g. the aptamer) along x and the second along y.",
    "progressive": "Time series: draw the curves up to the current movie frame only.",
    "smooth": "Time series: running mean over this many frames (0 or 1 = raw data).",
    "target": "Time series: value drawn as a horizontal reference line.",
    "vmin": "Heatmap: lower end of the colour scale.",
    "vmax": "Heatmap: upper end of the colour scale (0 = the data maximum).",
    "max_ticks": "Heatmap: at most this many residue labels on the axis.",
    "residues": "Sequence / secondary structure: residues shown, e.g. 'A:1-20, B:5-30' (empty: all residues "
                "of the structure view).",
    "per_row": "Sequence: residues per row (0: as many as fit).",
    "source": "Secondary structure: 'dynamic' (pairs re-detected every n frames), 'first' (the first frame's "
              "pairs) or 'dotbracket' (the 'dotbracket' option).",
    "dotbracket": "Secondary structure: dot-bracket of the shown residues ('&' or a space between chains).",
}


def _free_dict_doc(path: Tuple[str, ...], i: int, stack: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Comments inside the free-form tables: panel options, hand-picked step inputs, step settings."""
    if path[:3] == ("layout", "panels", "options") and len(path) == 4:
        return _PANEL_OPTION_DOCS.get(path[3], f"Panel option '{path[3]}' (see the panel type's description).")
    if path[1:3] == ("nodes", "files") or path[1:3] == ("scratch", "files"):
        if len(path) == 4:
            return (f"File chosen by hand for the step's '{path[3]}' input (paths relative to this file; "
                    "';' separates several).")
    return _settings_doc(path, i, stack)


def _settings_doc(path: Tuple[str, ...], i: int, stack: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Comments inside a planner step's ``settings``: the generator's option descriptions."""
    if path[:3] not in (("workflow", "nodes", "settings"), ("workflow", "scratch", "settings")) or i != 3:
        if len(path) > 4 and path[:3] in (("workflow", "nodes", "settings"), ("workflow", "scratch", "settings")):
            return _record_doc(path, stack)
        return None
    node = stack[-2] if len(stack) >= 2 else {}
    opt = _tool_option(node.get("tool", ""), path[3])
    if opt is None:
        return None
    return _option_text(opt)


def _record_doc(path: Tuple[str, ...], stack: Sequence[Dict[str, Any]]) -> Optional[str]:
    node = stack[-3] if len(stack) >= 3 else {}
    tool = _tool(node.get("tool", ""))
    if tool is None:
        return None
    for s in tool.sections:
        if s.repeat and s.key == path[3]:
            for o in s.options:
                if o.key == path[-1]:
                    return _option_text(o)
    return None


def _tool(name: str):
    try:
        from .workflows import get_tool

        return get_tool(name)
    except Exception:  # noqa: BLE001 - an unknown tool simply has no comments
        return None


def _tool_option(tool_name: str, key: str):
    tool = _tool(tool_name)
    if tool is None:
        return None
    for s in tool.sections:
        if s.key == key and s.repeat:
            return s
        for o in s.options:
            if o.key == key:
                return o
    return None


def _option_text(opt) -> str:
    label = getattr(opt, "label", "") or ""
    text = getattr(opt, "help", "") or getattr(opt, "tooltip", "") or ""
    choices = getattr(opt, "choices", None)
    if choices:
        text += " Choices: " + ", ".join(str(c[0]) for c in choices) + "."
    return f"{label}: {text}".strip(": ").strip()


# ---------------------------------------------------------------------------
# project files
# ---------------------------------------------------------------------------


def _now() -> _dt.datetime:
    return _dt.datetime.now().replace(microsecond=0)


def project_text(project, log_file: str = "") -> str:
    """The settings part of a project file (no log)."""
    from .config import _to_plain

    data: Dict[str, Any] = {"format": PROJECT_FORMAT, "format_version": PROJECT_FORMAT_VERSION,
                            "mdatk_version": __version__, "saved": _now(), "log_file": log_file}
    data.update(_to_plain(project))
    header = [f"{PROJECT_TITLE} project file.",
              "Plain TOML: every entry is explained by the comment above it. Edit with care; "
              f"{PROJECT_ABBREV} rewrites this",
              "file (keeping the step log) when the project is saved."]
    return tomlio.dumps(data, config_doc, header)


def is_params_log(path: str) -> bool:
    """True for a run's parameter log (opened like a project to repeat the run, but never logged into)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    return f'format = "{PARAMS_LOG_FORMAT}"' in head


def _reject_json(path: str) -> None:
    if path.lower().endswith(".json"):
        raise ValueError(f"{path}: JSON presets are no longer supported; {PROJECT_ABBREV} "
                         f"{__version__} uses TOML project files ({PROJECT_SUFFIX})")


_THREAD_LOCK = threading.RLock()
_HELD: Dict[str, int] = {}  # lock files this process holds (changed under _THREAD_LOCK only)


@contextlib.contextmanager
def _locked(path: str):
    """Serialise saves and log appends of one project (threads of this process and other processes).
    Re-entrant: a save inside :func:`locked` takes the lock again (a second flock would wait for itself)."""
    with _THREAD_LOCK:
        lock = os.path.join(os.path.dirname(os.path.abspath(path)), f".{os.path.basename(path)}.lock")
        if _HELD.get(lock):  # the thread lock is held too, so this is the same thread
            _HELD[lock] += 1
            try:
                yield
            finally:
                _HELD[lock] -= 1
            return
        try:
            import fcntl
            fh = open(lock, "a")
        except (ImportError, OSError):  # no fcntl, or a read-only folder: thread lock only
            yield
            return
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            _HELD[lock] = 1
            yield
        finally:
            _HELD.pop(lock, None)
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()


def locked(path: str):
    """Hold the lock of project ``path`` while re-reading the file and writing it (a context manager):
    other programs' saves wait, so whatever they saved before is read and nothing slips in between."""
    return _locked(path)


def save(path: str, project) -> None:
    """Write the settings of ``project`` to ``path``, keeping the file's step log."""
    _reject_json(path)
    if is_params_log(path):
        raise ValueError(f"{path} is the parameter log of a run; save the project under another name")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with _locked(path):
        _write(path, project)


def _write(path: str, project) -> None:
    """Write the settings, keeping the step log (the caller holds the project's lock)."""
    log_part, log_file = _existing_log(path)
    text = project_text(project, log_file)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        if log_part:
            fh.write("\n" + log_part)
    os.replace(tmp, path)


def edit_workflow(path: str, edit) -> None:
    """Change only the workflow part of project ``path``: re-read the file, call ``edit(workflow)`` and
    write it back, so the other settings stay as the file has them (relative paths stay relative; the
    step log is kept).  RNAdo saves its plan this way while the analysis window may have the file open;
    the read and the write happen under the project's lock, so no other save slips in between."""
    from .config import Project

    _reject_json(path)
    if is_params_log(path):
        raise ValueError(f"{path} is the parameter log of a run; save the project under another name")
    with _locked(path):
        project = Project.from_dict(read(path))  # no path resolution: relative paths stay relative
        edit(project.workflow)
        _write(path, project)


def _existing_log(path: str) -> Tuple[str, str]:
    """(log section text incl. marker, name of a separate log file) of an existing project."""
    if not os.path.isfile(path):
        return "", ""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    log_file = ""
    for line in text.splitlines()[:60]:
        if line.startswith("log_file = "):
            try:
                log_file = tomlio.loads(line)["log_file"]
            except tomlio.TOMLDecodeError:
                log_file = ""
            break
    at = text.find("\n" + LOG_MARKER)
    return (text[at + 1:] if at >= 0 else ""), log_file


def read(path: str) -> Dict[str, Any]:
    """Plain data of a project / params-log file (raises ``ValueError`` on bad content)."""
    _reject_json(path)
    try:
        data = tomlio.load(path)
    except tomlio.TOMLDecodeError as exc:
        raise ValueError(f"{path}: not valid TOML: {exc}") from None
    fmt = data.get("format")
    if fmt == PARAMS_LOG_FORMAT:  # replay a previous run
        data = dict(data.get("parameters") or {})
        fmt = PROJECT_FORMAT
    if fmt not in (None, PROJECT_FORMAT):
        raise ValueError(f"{path}: not an {PROJECT_ABBREV} project (format={fmt!r})")
    for meta in ("format", "format_version", "mdatk_version", "saved", "log", "log_file"):
        data.pop(meta, None)
    return data


# ---------------------------------------------------------------------------
# step log
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    step: str
    title: str = ""
    status: str = "ok"
    time: _dt.datetime = field(default_factory=_now)
    duration_s: float = 0.0
    command: str = ""
    folder: str = ""
    files: List[str] = field(default_factory=list)
    summary: str = ""
    citations: List[str] = field(default_factory=list)
    output: str = ""

    def plain(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["duration_s"] = round(float(self.duration_s), 2)
        if len(self.output) > LOG_OUTPUT_CHARS:
            data["output"] = "[...]\n" + self.output[-LOG_OUTPUT_CHARS:]
        return {k: v for k, v in data.items() if v not in ("", [], None)}


def _entry_text(entry: LogEntry, with_doc: bool) -> str:
    def doc(path, _stack):
        return _LOG_DOCS.get(path[1:]) if with_doc else None

    w = tomlio.Writer(doc)
    w.table(("log",), entry.plain(), [], array=True, first=with_doc)
    return "\n".join(w.lines).lstrip("\n") + "\n"


def log_path(path: str) -> str:
    """Where the step log of project ``path`` is written."""
    _part, log_file = _existing_log(path)
    return os.path.join(os.path.dirname(os.path.abspath(path)), log_file) if log_file else path


def append_log(path: str, entry: LogEntry) -> str:
    """Append ``entry`` to the step log of project ``path``; returns the file written."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with _locked(path):
        target = log_path(path)
        if target == path and os.path.getsize(path) > LOG_SPLIT_BYTES:
            target = _split_log(path)
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            has_marker = LOG_MARKER in fh.read()
        with open(target, "a", encoding="utf-8") as fh:
            if not has_marker:
                fh.write("\n" + LOG_MARKER + "\n")
            fh.write("\n" + _entry_text(entry, with_doc=not has_marker))
    return target


def _split_log(path: str) -> str:
    """Move the log of ``path`` into ``<name>.log.toml`` and point the project at it."""
    base = os.path.basename(path)
    stem = base[: -len(PROJECT_SUFFIX)] if base.endswith(PROJECT_SUFFIX) else os.path.splitext(base)[0]
    name = f"{stem}.log.toml"
    sidecar = os.path.join(os.path.dirname(os.path.abspath(path)), name)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    at = text.find("\n" + LOG_MARKER)
    head, log_part = (text[:at + 1], text[at + 1:]) if at >= 0 else (text, LOG_MARKER + "\n")
    with open(sidecar, "a", encoding="utf-8") as fh:
        if os.path.getsize(sidecar) == 0:
            fh.write(f"# Step log of the {PROJECT_ABBREV} project {base} (moved here when the project "
                     "file grew too large).\n\n")
        fh.write(log_part)
    head = head.replace('\nlog_file = ""\n', f"\nlog_file = {tomlio.quote(name)}\n", 1)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(head)
    os.replace(tmp, path)
    log.info("project log moved to %s", sidecar)
    return sidecar


@contextlib.contextmanager
def logged(path: Optional[str], step: str, title: str = "", citations: Sequence[str] = ()):
    """Time a step and append its log entry to project ``path`` (no-op without a path).

    The block may fill in the yielded :class:`LogEntry` (files, summary, output);
    an exception marks it failed (or cancelled) and propagates."""
    entry = LogEntry(step=step, title=title, citations=list(citations))
    t0 = time.monotonic()
    try:
        yield entry
    except BaseException as exc:
        from .progress import Cancelled
        cancelled = isinstance(exc, (Cancelled, KeyboardInterrupt))
        entry.status = "cancelled" if cancelled else "failed"
        if not entry.summary and not cancelled:
            entry.summary = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        entry.duration_s = time.monotonic() - t0
        if path and os.path.isfile(path) and not is_params_log(path):
            try:
                append_log(path, entry)
            except (OSError, ValueError) as exc:  # a failed log write never fails the step
                log.warning("could not write the step log of %s: %s", path, exc)


def read_log(path: str) -> List[Dict[str, Any]]:
    """The step log entries of project ``path`` (oldest first)."""
    target = log_path(path)
    try:
        return list(tomlio.load(target).get("log", []))
    except (OSError, tomlio.TOMLDecodeError) as exc:
        log.warning("could not read the step log %s: %s", target, exc)
        return []


# ---------------------------------------------------------------------------
# per-run parameter log
# ---------------------------------------------------------------------------


def write_params_log(path: str, project, run: Dict[str, Any]) -> None:
    """``run`` (environment, inputs, outputs, cache keys, citations) plus the full settings."""
    from .config import _to_plain

    data = {"format": PARAMS_LOG_FORMAT, "mdatk_version": __version__, "created": _now(),
            "run": run, "parameters": _to_plain(project)}

    def doc(path_: Tuple[str, ...], stack):
        if path_ == ("format",):
            return f"Per-run parameter log of {PROJECT_ABBREV}; open it as a project to replay the run."
        if path_ == ("created",):
            return "When the run finished."
        if path_ == ("run",):
            return "What was run: environment, input files, cleaning, cache keys, outputs, citations."
        if path_ == ("parameters",):
            return "The complete settings of the run."
        if path_ and path_[0] == "parameters" and len(path_) > 1:
            return config_doc(path_[1:], stack[1:])
        return _META_DOCS.get(path_[-1]) if len(path_) == 1 else None

    text = tomlio.dumps(data, doc, [f"{PROJECT_TITLE} parameter log of one run."])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# recent projects (per user)
# ---------------------------------------------------------------------------


def _settings_path() -> str:
    root = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(root, PROJECT_ABBREV.lower(), "settings.toml")


def recent_projects() -> List[str]:
    """Recently opened / saved project files that still exist, newest first."""
    try:
        data = tomlio.load(_settings_path())
    except (OSError, tomlio.TOMLDecodeError):
        return []
    return [p for p in data.get("recent", []) if isinstance(p, str) and os.path.isfile(p)][:RECENT_MAX]


def add_recent(path: str) -> None:
    path = os.path.abspath(path)
    recent = [path] + [p for p in recent_projects() if p != path]
    target = _settings_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        text = tomlio.dumps({"recent": recent[:RECENT_MAX]},
                            lambda p, s: "Recently used project files, newest first.",
                            [f"{PROJECT_TITLE} user settings."])
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        log.warning("could not update the recent projects list: %s", exc)
