"""Menu entries for MDATK and RNAdo: freedesktop.org desktop entries and icons in the user's data folder.

The AppImage integrates one desktop entry (MDATK's, with an RNAdo action) and a pip installation none, so
RNAdo has no menu entry of its own there.  :func:`install` writes both, ``mdatk.desktop`` and
``rnado.desktop``, into ``$XDG_DATA_HOME/applications`` (default ``~/.local/share``) and their icons into
``$XDG_DATA_HOME/icons/hicolor/<size>/apps``; :func:`uninstall` removes them again.  Every file written
carries the mark :data:`MARK` (a key of the desktop entry, a text chunk of a PNG, a comment of the SVG),
so only files MDATK wrote are ever replaced or removed; anything else at those paths is left alone
(``replace=True`` moves it aside first).

The ``Exec`` lines start what runs now (:func:`commands`): the AppImage (``$APPIMAGE gui %f``), the
programs of a bundle folder, or the scripts of a pip / venv installation (else ``python -m``).  The
templates are ``mdatk/assets/mdatk.desktop`` and ``rnado.desktop``; the icons are the committed PNGs and
SVG in ``mdatk/assets``.  On Windows the installer makes the Start-menu entries (:func:`unsupported`).
This module does not import GTK (``mdatk desktop install|uninstall|status``, ``rnado --install-launchers``).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import __version__

MARK = "X-MDATK-Installed"  # in every file install writes; its value is the MDATK version that wrote it
PROGRAMS = ("mdatk", "rnado")  # desktop file ids and icon names
ACTION_GROUP = "Desktop Action rnado"  # the RNAdo action of MDATK's entry starts RNAdo too
MISSING, INSTALLED, OUTDATED, FOREIGN = "missing", "installed", "outdated", "not MDATK's"
TOOL_TIMEOUT_S = 30  # update-desktop-database, gtk-update-icon-cache
_PNG = b"\x89PNG\r\n\x1a\n"
_ICON = re.compile(r"^(mdatk|rnado)-icon-(\d+)\.png$")
# Desktop Entry Specification, "The Exec key": an argument holding one of these is quoted
_RESERVED = frozenset(" \t\n\"'\\><~|&;$*?#()`")


def assets_dir() -> str:
    """``mdatk/assets``: the desktop-entry templates and the icons (package data, bundled by the builds)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def unsupported() -> Optional[str]:
    """Why this system gets no desktop entries (None: it does)."""
    if os.name == "nt":
        return ("On Windows the installer (MDATK-<version>-windows-x64-setup.exe) already makes the "
                "Start-menu entries \"MDATK\" and \"RNAdo\" (and, on request, desktop shortcuts). From the "
                "zip, start mdatk-gui.exe or rnado-gui.exe, or pin them to Start yourself. No .desktop "
                "files are written on Windows.")
    if sys.platform == "darwin":
        return "Desktop entries are a Linux (freedesktop.org) menu; there is nothing to add on macOS."
    return None


def data_home() -> str:
    """``$XDG_DATA_HOME`` when it is an absolute path (the specification ignores relative ones), else
    ``~/.local/share``."""
    value = os.environ.get("XDG_DATA_HOME", "")
    return value if os.path.isabs(value) else os.path.join(os.path.expanduser("~"), ".local", "share")


# ---------------------------------------------------------------------------
# what the entries start
# ---------------------------------------------------------------------------


def _package_root() -> str:
    """The folder holding the ``mdatk`` package (site-packages, or a source checkout)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _installed() -> bool:
    """Whether this mdatk is the one this Python imports by itself (installed into it, or an editable
    installation of this checkout), not a checkout reached through PYTHONPATH or the working folder."""
    from .gui.launch import _same, _site_dir

    root = _package_root()
    if _site_dir(root):
        return True
    try:  # pip install -e: the distribution records the checkout it points at
        from importlib import metadata
        from urllib.parse import unquote, urlparse

        info = json.loads(metadata.distribution("mdatk").read_text("direct_url.json") or "{}")
        url = urlparse(info.get("url", ""))
        return bool(info.get("dir_info", {}).get("editable")) and url.scheme == "file" and \
            _same(unquote(url.path), root)
    except Exception:  # no distribution, no direct_url.json, unreadable metadata
        return False


def _interpreter(script: str) -> Optional[str]:
    """The Python a pip-written script runs: its ``#!`` line, or the ``'''exec' "PATH"`` line pip writes
    for interpreter paths with spaces.  None for ``#!/usr/bin/env python3`` and anything else."""
    try:
        with open(script, "rb") as fh:
            lines = fh.read(2048).split(b"\n")
    except OSError:
        return None
    if not lines[0].startswith(b"#!"):
        return None
    try:
        first = lines[0][2:].decode().strip()
        if first == "/bin/sh" and len(lines) > 1 and lines[1].startswith(b"'''exec' "):
            import shlex

            words = shlex.split(lines[1][len(b"'''exec' "):].decode())
        else:
            words = first.split()
    except (UnicodeDecodeError, ValueError):
        return None
    return words[0] if words and os.path.isabs(words[0]) and os.path.basename(words[0]) != "env" else None


def _runs_this_python(script: str) -> bool:
    """Whether ``script`` runs this Python (same folder, so the same venv, and the same program)."""
    interp = _interpreter(script)
    if interp is None:
        return False
    a, b = os.path.abspath(interp), os.path.abspath(sys.executable)
    # the folder too: a venv's bin/python is a link to the system Python, whose scripts are not the venv's
    return os.path.dirname(a) == os.path.dirname(b) and os.path.realpath(a) == os.path.realpath(b)


def _scripts_dirs() -> List[str]:
    """Where pip puts this Python's scripts: its own (a venv's bin), the user's (``pip install --user``)."""
    dirs = [sysconfig.get_path("scripts")]
    user = f"{os.name}_user"
    if user in sysconfig.get_scheme_names():
        try:
            dirs.append(sysconfig.get_path("scripts", user))
        except KeyError:  # pragma: no cover - a scheme without scripts
            pass
    return [d for d in dirs if d]


def _script(name: str) -> Optional[str]:
    """The installed script ``name`` (``mdatk-gui``, ``rnado-gui``) that runs this Python, or None."""
    candidates = [os.path.join(d, name) for d in _scripts_dirs()] + [shutil.which(name) or ""]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK) and _runs_this_python(path):
            return os.path.abspath(path)
    return None


def _apprun(folder: str) -> Optional[str]:
    """The ``AppRun`` of an extracted AppImage (``squashfs-root``) whose bundle is ``folder``
    (``<root>/usr/lib/mdatk``): it adds the fallback libraries the host lacks, so the entries use it."""
    parts = os.path.normpath(folder).split(os.sep)
    if parts[-3:] != ["usr", "lib", "mdatk"]:
        return None
    path = os.path.join(os.sep.join(parts[:-3]) or os.sep, "AppRun")
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


def commands(mode: Optional[str] = None) -> Dict[str, List[str]]:
    """The commands the entries start, before the file argument (``%f``): ``{"mdatk": [...], "rnado":
    [...]}``.  ``mode``: :func:`mdatk.gui.launch.launch_mode` unless given."""
    from .gui.launch import launch_mode

    mode = mode or launch_mode()
    if mode in ("appimage", "appimage-extracted"):
        program = [os.environ["APPIMAGE"]]
        if mode == "appimage-extracted":  # no FUSE here: the entries must extract it too
            program = ["env", "APPIMAGE_EXTRACT_AND_RUN=1"] + program
        return {"mdatk": program + ["gui"], "rnado": program + ["rnado"]}
    if mode == "bundle":
        folder = os.path.dirname(os.path.abspath(sys.executable))
        apprun = _apprun(folder)
        if apprun:
            return {"mdatk": [apprun, "gui"], "rnado": [apprun, "rnado"]}
        gui = os.path.join(folder, "mdatk-gui")  # Windows only (no desktop entries there): mdatk gui on Linux
        return {"mdatk": [gui] if os.path.isfile(gui) else [os.path.join(folder, "mdatk"), "gui"],
                "rnado": [os.path.join(folder, "rnado-gui")]}
    python = os.path.abspath(sys.executable)
    if _installed():
        gui, rnado = _script("mdatk-gui"), _script("rnado-gui")
        return {"mdatk": [gui] if gui else [python, "-m", "mdatk", "gui"],
                "rnado": [rnado] if rnado else [python, "-m", "mdatk.gui.rnado"]}
    # a source checkout run as it is: the menu starts it with the checkout on the path
    program = ["env", f"PYTHONPATH={_package_root()}", python]
    return {"mdatk": program + ["-m", "mdatk", "gui"], "rnado": program + ["-m", "mdatk.gui.rnado"]}


def warning(mode: Optional[str] = None) -> Optional[str]:
    """What breaks the entries (they hold a path), for the confirmation; None when nothing to say."""
    from .gui.launch import launch_mode

    mode = mode or launch_mode()
    if mode in ("appimage", "appimage-extracted"):
        return (f"The entries start the AppImage at its current path, {os.environ.get('APPIMAGE', '?')}. "
                "Moving or renaming the file breaks them: add them again from the new place (or remove them "
                "first with 'desktop uninstall').")
    if mode == "bundle":
        return (f"The entries start the programs in {os.path.dirname(os.path.abspath(sys.executable))}; "
                "moving that folder breaks them: add them again from the new place.")
    if not _installed():
        return (f"MDATK runs from the source checkout {_package_root()} without being installed: the entries "
                "start it from there with this Python.")
    return None


def exec_arg(arg: str) -> str:
    """One argument of an Exec line (Desktop Entry Specification, "The Exec key"): a literal % doubled
    (field codes), quoted in double quotes when it holds a reserved character, with ``" ` $ \\`` escaped."""
    arg = arg.replace("%", "%%")
    if arg and not _RESERVED.intersection(arg):
        return arg
    return '"' + "".join("\\" + c if c in '"`$\\' else c for c in arg) + '"'


def exec_value(argv: Sequence[str], code: str = "%f") -> str:
    """The value of an Exec key for ``argv`` and the field code ``code``, as written in the file: the
    string escapes of the specification (``\\\\``, ``\\n``, ``\\t``, ``\\r``) come on top of the quoting."""
    line = " ".join([exec_arg(a) for a in argv] + ([code] if code else []))
    return line.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------


@dataclass
class Target:
    """A file :func:`install` writes: where, what, and what is there now (:func:`status`)."""

    path: str
    data: bytes
    state: str = MISSING  # MISSING, INSTALLED (this content), OUTDATED (MDATK's, other content), FOREIGN
    exec_now: str = ""  # a desktop entry's Exec line as the file has it (OUTDATED, FOREIGN)


def _icons(assets: str) -> Dict[str, List[Tuple[str, str]]]:
    """``{program: [(source, hicolor subfolder), ...]}`` from the PNGs (every size) and RNAdo's SVG."""
    found: Dict[str, List[Tuple[str, str]]] = {p: [] for p in PROGRAMS}
    names = os.listdir(assets) if os.path.isdir(assets) else []
    pngs = [(m.group(1), int(m.group(2)), name) for name in names for m in [_ICON.match(name)] if m]
    for program, size, name in sorted(pngs):
        found[program].append((os.path.join(assets, name), f"{size}x{size}"))
    for program in PROGRAMS:
        svg = os.path.join(assets, f"{program}-icon.svg")
        if os.path.isfile(svg):
            found[program].append((svg, "scalable"))
    return found


def entry_text(program: str, cmds: Dict[str, List[str]], icon: Optional[str] = None,
               template: Optional[str] = None, version: str = __version__) -> str:
    """The desktop entry ``program`` from its template (``mdatk/assets/<program>.desktop``): the Exec lines
    from ``cmds`` (:func:`commands`; the RNAdo action of MDATK's entry starts RNAdo), ``Icon=icon`` in the
    main group when given, the template's comments replaced by a note, and the mark."""
    if template is None:
        with open(os.path.join(assets_dir(), f"{program}.desktop"), encoding="utf-8") as fh:
            template = fh.read()
    out = [f"# Written by MDATK {version} ('mdatk desktop install'); 'mdatk desktop uninstall' removes it."]
    group = None
    for line in template.splitlines():
        text = line.strip()
        if text.startswith("#"):
            continue
        if text.startswith("[") and text.endswith("]"):
            group = text[1:-1]
            out.append(text)
            if group == "Desktop Entry":
                out.append(f"{MARK}={version}")
            continue
        key = text.split("=", 1)[0].strip() if "=" in text else ""
        if key == "Exec":
            target = "rnado" if group == ACTION_GROUP else program
            text = f"Exec={exec_value(cmds[target])}"
        elif key == "Icon" and group == "Desktop Entry" and icon:
            text = f"Icon={icon}"
        elif key == MARK:
            continue
        out.append(text)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def mark_png(data: bytes, version: str = __version__) -> bytes:
    """``data`` (a PNG) with a ``tEXt`` chunk ``X-MDATK-Installed`` after its header."""
    if data[:8] != _PNG or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file")
    end = 8 + 12 + struct.unpack(">I", data[8:12])[0]
    payload = MARK.encode("latin-1") + b"\0" + version.encode("latin-1")
    chunk = struct.pack(">I", len(payload)) + b"tEXt" + payload + \
        struct.pack(">I", zlib.crc32(b"tEXt" + payload) & 0xFFFFFFFF)
    return data[:end] + chunk + data[end:]


def mark_svg(data: bytes, version: str = __version__) -> bytes:
    """``data`` (an SVG) with the comment ``<!-- X-MDATK-Installed=VERSION -->`` after its XML declaration."""
    comment = f"<!-- {MARK}={version}: written by 'mdatk desktop install' -->\n".encode()
    if data.startswith(b"<?xml"):
        end = data.find(b"?>")
        if end >= 0:
            end = data.find(b"\n", end)
            end = len(data) if end < 0 else end + 1
            return data[:end] + comment + data[end:]
    return comment + data


def is_marked(path: str) -> bool:
    """Whether the file at ``path`` was written by :func:`install` (it carries :data:`MARK`)."""
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return False
        with open(path, "rb") as fh:
            data = fh.read(16 * 1024 * 1024)
    except OSError:
        return False
    if path.endswith(".desktop"):
        return any(line.strip().startswith(f"{MARK}=".encode()) for line in data.splitlines())
    if data[:8] == _PNG:
        pos = 8
        while pos + 8 <= len(data):
            n, kind = struct.unpack(">I", data[pos:pos + 4])[0], data[pos + 4:pos + 8]
            if kind == b"tEXt" and data[pos + 8:pos + 8 + n].split(b"\0", 1)[0] == MARK.encode():
                return True
            if kind == b"IEND":
                break
            pos += 12 + n
        return False
    if path.endswith(".svg"):
        return f"<!-- {MARK}=".encode() in data[:4096]
    return False


def _exec_of(path: str) -> str:
    """The Exec line of the main group of a desktop entry ("" when unreadable or missing)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            group = None
            for line in fh:
                text = line.strip()
                if text.startswith("["):
                    group = text
                elif group == "[Desktop Entry]" and text.startswith("Exec="):
                    return text[5:]
    except OSError:
        pass
    return ""


def targets(data_dir: Optional[str] = None, mode: Optional[str] = None,
            assets: Optional[str] = None) -> List[Target]:
    """The files :func:`install` writes into ``data_dir`` (:func:`data_home`), with their states."""
    data_dir = data_dir or data_home()
    assets = assets or assets_dir()
    cmds = commands(mode)
    icons = _icons(assets)
    out: List[Target] = []
    for program in PROGRAMS:
        template = os.path.join(assets, f"{program}.desktop")
        if not os.path.isfile(template):
            raise FileNotFoundError(f"the desktop-entry template {template} is missing (a broken "
                                    "installation)")
        with open(template, encoding="utf-8") as fh:
            text = entry_text(program, cmds, program if icons[program] else None, fh.read())
        out.append(Target(os.path.join(data_dir, "applications", f"{program}.desktop"), text.encode("utf-8")))
    for program in PROGRAMS:
        for source, sub in icons[program]:
            with open(source, "rb") as fh:
                raw = fh.read()
            ext = os.path.splitext(source)[1]
            data = mark_svg(raw) if ext == ".svg" else mark_png(raw)
            out.append(Target(os.path.join(data_dir, "icons", "hicolor", sub, "apps", program + ext), data))
    for t in out:
        t.state = _state(t)
        if t.path.endswith(".desktop") and t.state in (OUTDATED, FOREIGN):
            t.exec_now = _exec_of(t.path)
    return out


def _state(t: Target) -> str:
    if not os.path.lexists(t.path):
        return MISSING
    if not is_marked(t.path):
        return FOREIGN
    try:
        with open(t.path, "rb") as fh:
            return INSTALLED if fh.read() == t.data else OUTDATED
    except OSError:
        return OUTDATED


def status(data_dir: Optional[str] = None) -> List[Target]:
    """``mdatk desktop status``: the files :func:`install` writes and what is at their paths now."""
    return targets(data_dir)


# ---------------------------------------------------------------------------
# install, uninstall
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """What :func:`install` / :func:`uninstall` did."""

    data_dir: str
    written: List[str] = field(default_factory=list)  # new or rewritten files
    unchanged: List[str] = field(default_factory=list)  # already as install writes them
    removed: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)  # files MDATK did not write, left alone
    backups: List[Tuple[str, str]] = field(default_factory=list)  # (file, its backup) with replace=True
    tools: List[str] = field(default_factory=list)  # cache updates that ran, with their outcome


def _write(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in the same folder (never half a file)."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".mdatk-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _backup(path: str) -> str:
    """Move ``path`` aside to ``PATH.bak`` (``PATH.bak2`` … when taken) and return the new name."""
    n = 1
    while True:
        backup = f"{path}.bak" + (str(n) if n > 1 else "")
        if not os.path.lexists(backup):
            os.replace(path, backup)
            return backup
        n += 1


def refresh(data_dir: str) -> List[str]:
    """Tell the desktop about the change, never failing: ``update-desktop-database`` on the applications
    folder, and the hicolor folder touched (its ``gtk-update-icon-cache`` refreshed when it has a cache;
    without one, GTK reads the folders and none is made)."""
    from .frozen import child_env

    done = []

    def run(argv: List[str]) -> None:
        try:
            proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE, timeout=TOOL_TIMEOUT_S, env=child_env())
            why = proc.stderr.decode(errors="replace").strip().splitlines()
            done.append(f"{os.path.basename(argv[0])}: " + ("done" if proc.returncode == 0 else
                        f"exit status {proc.returncode}" + (f" ({why[-1]})" if why else "")))
        except (OSError, subprocess.SubprocessError) as exc:
            done.append(f"{os.path.basename(argv[0])}: not run ({exc})")

    apps = os.path.join(data_dir, "applications")
    tool = shutil.which("update-desktop-database")
    if tool and os.path.isdir(apps):
        run([tool, "-q", apps])
    hicolor = os.path.join(data_dir, "icons", "hicolor")
    if os.path.isdir(hicolor):
        try:  # a newer folder makes GTK look again
            os.utime(hicolor)
        except OSError:
            pass
        tool = shutil.which("gtk-update-icon-cache")
        if tool and os.path.isfile(os.path.join(hicolor, "icon-theme.cache")):
            run([tool, "-q", "-f", "-t", hicolor])
    return done


def install(data_dir: Optional[str] = None, replace: bool = False, mode: Optional[str] = None,
            update_caches: bool = True) -> Result:
    """Write both desktop entries and the icons (:func:`targets`).  Files MDATK did not write stay as they
    are (listed in ``kept``), or with ``replace`` are moved to ``*.bak`` first; raises RuntimeError on
    Windows and macOS (:func:`unsupported`) and OSError when a file cannot be written."""
    why = unsupported()
    if why:
        raise RuntimeError(why)
    data_dir = data_dir or data_home()
    result = Result(data_dir)
    # the icons first: an entry never shows up in the menu before its icon
    for t in sorted(targets(data_dir, mode), key=lambda t: t.path.endswith(".desktop")):
        if t.state == INSTALLED:
            result.unchanged.append(t.path)
            continue
        if t.state == FOREIGN:
            if not replace:
                result.kept.append(t.path)
                continue
            result.backups.append((t.path, _backup(t.path)))
        _write(t.path, t.data)
        result.written.append(t.path)
    if update_caches and result.written:
        result.tools = refresh(data_dir)
    return result


def _candidates(data_dir: str) -> List[str]:
    """Where :func:`install` writes, or wrote in another version: the two entries and every ``mdatk`` /
    ``rnado`` icon of the hicolor theme, any size."""
    paths = [os.path.join(data_dir, "applications", f"{p}.desktop") for p in PROGRAMS]
    for program in PROGRAMS:
        for ext in (".png", ".svg"):
            paths += sorted(glob.glob(os.path.join(glob.escape(data_dir), "icons", "hicolor", "*", "apps",
                                                   program + ext)))
    return paths


def installed_files(data_dir: Optional[str] = None) -> List[str]:
    """The files in ``data_dir`` that :func:`install` wrote (they carry the mark), of any MDATK version."""
    return [p for p in _candidates(data_dir or data_home()) if is_marked(p)]


def uninstall(data_dir: Optional[str] = None, update_caches: bool = True) -> Result:
    """Remove the files MDATK wrote (:func:`installed_files`); files without the mark stay (``kept``)."""
    data_dir = data_dir or data_home()
    result = Result(data_dir)
    for path in installed_files(data_dir):
        os.unlink(path)
        result.removed.append(path)
    result.kept = [p for p in _candidates(data_dir) if os.path.lexists(p)]
    if update_caches and result.removed:
        result.tools = refresh(data_dir)
    return result


# ---------------------------------------------------------------------------
# reports (command line and GUI)
# ---------------------------------------------------------------------------


def display_path(path: str) -> str:
    """``path`` with ``~`` for the home folder."""
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if home not in ("", os.sep) and path.startswith(home + os.sep) else path


def listing(paths: Sequence[str], base: Optional[str] = None, indent: str = "  ") -> str:
    """``paths`` one folder per line with the file names after it, the folders relative to ``base`` (the
    data folder) when they are inside it (the GUI's confirmation and result)."""
    by_folder: Dict[str, List[str]] = {}
    for p in paths:
        by_folder.setdefault(os.path.dirname(p), []).append(os.path.basename(p))

    def name(folder: str) -> str:
        rel = os.path.relpath(folder, base) if base else ""
        return rel if base and not rel.startswith(os.pardir) else display_path(folder)

    return "\n".join(f"{indent}{name(d)}/: {', '.join(names)}" for d, names in by_folder.items())


def plan_text(found: Sequence[Target], mode: Optional[str] = None, data_dir: Optional[str] = None) -> str:
    """What :func:`install` will do with ``found`` (:func:`targets` of ``data_dir``), for the confirmation."""
    data_dir = data_dir or data_home()
    cmds = commands(mode)
    write = [t.path for t in found if t.state in (MISSING, OUTDATED)]
    same = [t.path for t in found if t.state == INSTALLED]
    foreign = [t.path for t in found if t.state == FOREIGN]
    parts = [f"In {display_path(data_dir)}:"]
    if write:
        parts.append("Files written:\n" + listing(write, data_dir))
    if same:
        parts.append("Already there:\n" + listing(same, data_dir))
    if foreign:
        parts.append("Left alone (not written by MDATK; remove them yourself, or use 'mdatk desktop install "
                     "--replace', which keeps a .bak copy):\n" + listing(foreign, data_dir))
    parts.append(f"MDATK starts as: {exec_value(cmds['mdatk'])}\n"
                 f"RNAdo starts as: {exec_value(cmds['rnado'])}")
    note = warning(mode)
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def result_text(result: Result) -> str:
    """What :func:`install` or :func:`uninstall` did, one file per line."""
    lines = [f"written    {p}" for p in result.written] + [f"unchanged  {p}" for p in result.unchanged]
    lines += [f"removed    {p}" for p in result.removed]
    lines += [f"backup     {p} -> {b}" for p, b in result.backups]
    lines += [f"kept       {p} (not written by MDATK)" for p in result.kept]
    lines += [f"ran        {t}" for t in result.tools]
    return "\n".join(lines)


def run(action: str, replace: bool = False, out: Callable[[str], None] = print) -> int:
    """``mdatk desktop install|uninstall|status`` (``rnado --install-launchers`` is ``install``)."""
    why = unsupported()
    if why:
        out(why)
        return 0
    data_dir = data_home()
    if action == "status":
        found = targets(data_dir)
        for t in found:
            out(f"{t.state:<12} {t.path}" + (f"  (Exec={t.exec_now})" if t.exec_now else ""))
        states = {t.state for t in found if t.path.endswith(".desktop")}
        if states == {INSTALLED}:
            out("MDATK and RNAdo are in the applications menu.")
        elif OUTDATED in states:
            out("The entries are MDATK's but start something else (another path or version): run "
                "'mdatk desktop install' again.")
        else:
            out("Not (fully) in the applications menu: 'mdatk desktop install' adds them.")
        cmds = commands()
        out(f"MDATK starts as: {exec_value(cmds['mdatk'])}\nRNAdo starts as: {exec_value(cmds['rnado'])}")
        return 0
    if action == "install":
        result = install(data_dir, replace=replace)
        out(result_text(result))
        if result.kept:
            out("Files that MDATK did not write were left alone: remove them, or run 'mdatk desktop install "
                "--replace' (keeps a .bak copy of each).")
        out(f"MDATK and RNAdo are in the applications menu ({data_dir}); some desktops show new entries only "
            "after the next log-in.")
        note = warning()
        if note:
            out(f"Note: {note}")
        return 0
    if action == "uninstall":
        result = uninstall(data_dir)
        out(result_text(result) or f"Nothing of MDATK's in {data_dir}.")
        return 0
    raise ValueError(f"unknown action {action!r} (install, uninstall, status)")
