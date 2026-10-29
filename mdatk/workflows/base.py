"""Base class of the script generators and the per-step context they use."""

from __future__ import annotations

import copy
import datetime as _dt
import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
from .naming import file_name, sanitize_project, step_folder
from .schema import GeneratedFile, Issue, Option, Port, Section

log = logging.getLogger(__name__)


@dataclass
class PortValue:
    """Files flowing along a planner connection.

    ``paths`` are relative to the plan folder (files produced by a step) or
    absolute (files chosen by hand).  ``data`` carries facts known when the
    files are planned, e.g. the secondary structure behind a ``.secstruct``
    file or whether a topology carries NBfix / 12-6-4 / HMR modifications.
    """

    paths: List[str]
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # producing node id ("" = chosen by hand)
    source_label: str = ""

    @property
    def path(self) -> str:
        return self.paths[0] if self.paths else ""


class StepContext:
    """Names and inputs of one step, as seen from the step's own folder.

    Generated scripts ``cd`` into their folder first, so every path handed
    to a generator is relative to that folder (absolute for hand-picked
    files).
    """

    def __init__(self, project: str, step: int, stamp: str, tag: str,
                 inputs: Optional[Dict[str, PortValue]] = None, root: str = "",
                 step_folders: bool = True, title: str = "", node_id: str = "",
                 groups: Optional[Sequence[Any]] = None):
        self.project = sanitize_project(project)
        self.step = int(step)
        self.stamp = stamp
        self.tag = tag
        self.folder = step_folder(self.step, tag) if step_folders else ""
        self.inputs: Dict[str, PortValue] = dict(inputs or {})
        self.root = root
        self.title = title
        self.node_id = node_id
        self.groups = list(groups or [])
        self.preview = False  # True while the wizard previews the files: no network or slow work

    # ------------------------------------------------------------- naming
    def name(self, tag: str, ext: str = "") -> str:
        """File name of one of this step's own files."""
        return file_name(self.project, self.step, tag, self.stamp, ext)

    def own(self, tag: str, ext: str = "") -> str:
        """Plan-relative path of one of this step's own files."""
        return posixpath.join(self.folder, self.name(tag, ext)) if self.folder else self.name(tag, ext)

    def from_own(self, name: str) -> str:
        """Plan-relative path of a file name inside this step's folder."""
        return posixpath.join(self.folder, name) if self.folder else name

    # ------------------------------------------------------------- inputs
    def has(self, port: str) -> bool:
        value = self.inputs.get(port)
        return bool(value and value.paths)

    def value(self, port: str) -> Optional[PortValue]:
        return self.inputs.get(port)

    def rel(self, path: str) -> str:
        """``path`` (plan-relative or absolute) as seen from this step's folder."""
        if not path or os.path.isabs(path):
            return path
        return posixpath.relpath(path, self.folder) if self.folder else path

    def inp(self, port: str) -> str:
        value = self.inputs.get(port)
        return self.rel(value.path) if value and value.paths else ""

    def inps(self, port: str) -> List[str]:
        value = self.inputs.get(port)
        return [self.rel(p) for p in value.paths] if value else []

    def data(self, port: str) -> Dict[str, Any]:
        value = self.inputs.get(port)
        return value.data if value else {}

    def abspath(self, path: str) -> str:
        if not path or os.path.isabs(path):
            return path
        return os.path.join(self.root, path) if self.root else path

    def existing(self, port: str) -> str:
        """Absolute path of an input that already exists on disk ("" otherwise)."""
        value = self.inputs.get(port)
        if not value or not value.paths:
            return ""
        path = self.abspath(value.path)
        return path if path and os.path.isfile(path) else ""

    def source_label(self, port: str) -> str:
        value = self.inputs.get(port)
        return value.source_label if value else ""


class Tool:
    """A script generator, i.e. a step type of the Project Planner.

    Subclasses define ``sections`` (the options), ``inputs`` / ``outputs``
    (planner ports) and implement :meth:`check`, :meth:`output_values` and
    :meth:`generate`.  Generators only *write* files; they never run the
    programs they prepare.
    """

    name = ""
    label = ""
    category = ""
    tag = ""  # step folder name and default file tag
    description = ""
    generates = True
    sections: List[Section] = []
    inputs: List[Port] = []
    outputs: List[Port] = []
    path_settings: Tuple[str, ...] = ()  # settings that hold paths relative to the project file

    # ----------------------------------------------------------- schema
    def all_options(self) -> Iterator[Tuple[Section, Option]]:
        for section in self.sections:
            for opt in section.options:
                yield section, opt

    def flat_options(self) -> Dict[str, Option]:
        return {o.key: o for s in self.sections if not s.repeat for o in s.options}

    def section(self, key: str) -> Section:
        for s in self.sections:
            if s.key == key:
                return s
        raise KeyError(key)

    def defaults(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for s in self.sections:
            if s.repeat:
                values[s.key] = s.default_records()
            else:
                for o in s.options:
                    values[o.key] = copy.deepcopy(o.default)
        return values

    def normalise(self, settings: Optional[Dict[str, Any]],
                  issues: Optional[List[Issue]] = None) -> Dict[str, Any]:
        """Complete ``settings`` with defaults and coerce types.

        Invalid values fall back to their default with a warning issue;
        unknown keys are dropped (logged), so project files from other versions
        still load.
        """
        settings = dict(settings or {})
        out = self.defaults()
        known = set(out)
        for s in self.sections:
            if s.repeat:
                raw = settings.get(s.key)
                if raw is None:
                    continue
                if not isinstance(raw, list):
                    self._bad(issues, s.key, "expected a list")
                    continue
                records = []
                for k, rec in enumerate(raw):
                    if not isinstance(rec, dict):
                        self._bad(issues, f"{s.key}.{k}", "expected an object")
                        continue
                    clean = s.record_defaults()
                    for o in s.options:
                        if o.key in rec:
                            try:
                                clean[o.key] = o.coerce(rec[o.key])
                            except ValueError as exc:
                                self._bad(issues, f"{s.key}.{k}.{o.key}", str(exc))
                    records.append(clean)
                out[s.key] = records
            else:
                for o in s.options:
                    if o.key in settings:
                        try:
                            out[o.key] = o.coerce(settings[o.key])
                        except ValueError as exc:
                            self._bad(issues, o.key, str(exc))
        unknown = sorted(set(settings) - known)
        if unknown:
            log.warning("%s: ignoring unknown settings %s", self.name, ", ".join(unknown))
        return out

    @staticmethod
    def _bad(issues, key, message):
        text = f"{message} - using the default"
        if issues is not None:
            issues.append(Issue("warning", text, (key,)))
        else:
            log.warning("%s: %s", key, text)

    def input_ports(self, settings: Dict[str, Any]) -> List[Port]:
        return list(self.inputs)

    def output_ports(self, settings: Dict[str, Any]) -> List[Port]:
        return list(self.outputs)

    # --------------------------------------------------------- behaviour
    def check(self, s: Dict[str, Any], ctx: StepContext) -> List[Issue]:
        return []

    def output_values(self, s: Dict[str, Any], ctx: StepContext) -> Dict[str, PortValue]:
        return {}

    def generate(self, s: Dict[str, Any], ctx: StepContext) -> List[GeneratedFile]:
        return []

    def describe(self, s: Dict[str, Any]) -> str:
        """Short summary shown on planner boxes."""
        return self.description.split(".")[0]

    def resources(self, s: Dict[str, Any]) -> List[str]:
        """Ids of the programs / force fields / parameter sets these settings use (see mdatk.citations)."""
        return []

    def step_resources(self, s: Dict[str, Any], ctx: Optional["StepContext"]) -> List[str]:
        """Resources of a resolved step, which may also depend on its inputs; by default :meth:`resources`."""
        return self.resources(s)


# ---------------------------------------------------------------------------
# helpers shared by the generators
# ---------------------------------------------------------------------------


def env_section(placeholder: str, extra: Sequence[Option] = ()) -> Section:
    """'Environment' options: shell lines run before any program is called."""
    return Section("environment", "Environment", [
        Option("conda_env", "Conda environment", "text", "",
               "Activate this conda environment at the top of the script "
               "(eval \"$(conda shell.bash hook)\"; conda activate <name>). Leave empty when "
               "the programs are already on PATH.", placeholder="e.g. AmberTools26"),
        Option("env_setup", "Setup commands", "multiline", "",
               "Shell lines run before any program, e.g. 'module load ...' on a cluster or "
               "'source /opt/amber26/amber.sh' on a workstation.", placeholder=placeholder),
        *extra,
    ], tooltip="How the generated scripts find the programs.")


def env_lines(s: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if s.get("conda_env", "").strip():
        lines += ['eval "$(conda shell.bash hook)"', f"conda activate {s['conda_env'].strip()}"]
    for line in (s.get("env_setup") or "").splitlines():
        if line.strip():
            lines.append(line.rstrip())
    return lines


def script_header(title: str, ctx: StepContext, s: Dict[str, Any], script: str,
                  notes: Sequence[str] = ()) -> List[str]:
    today = _dt.date.today().isoformat()
    lines = ["#!/bin/bash",
             f"# {title}",
             f"# Project {ctx.project}, step {ctx.step:02d} - written by {PROJECT_ABBREV} {__version__} "
             f"on {today}.",
             f"# {PROJECT_ABBREV} only writes this script; review it, then run it with: bash {script}"]
    lines += [f"# {n}" for n in notes]
    lines += ["", "set -eo pipefail", 'cd "$(dirname "$0")"', ""]
    env = env_lines(s)
    if env:
        lines += ["# Environment"] + env + [""]
    return lines


def require_file(path: str, what: str) -> str:
    """Shell line that stops the script when a program did not produce ``path``."""
    return f'[ -s "{path}" ] || {{ echo "ERROR: {what} did not write {path}" >&2; exit 1; }}'


def fortran_float(x: float) -> str:
    """Format like hand-written Amber input: 10.0, 0.002, 1.0e-6."""
    x = float(x)
    if x == 0:
        return "0.0"
    if 1e-4 <= abs(x) < 1e7:
        text = repr(x)
        if "e" in text or "E" in text:
            text = f"{x:.6f}".rstrip("0")
        if "." not in text:
            text += ".0"
        if text.endswith("."):
            text += "0"
        return text
    mant, exp = f"{x:.6e}".split("e")
    mant = mant.rstrip("0")
    if mant.endswith("."):
        mant += "0"
    return f"{mant}e{int(exp)}"


def shell_quote(text: str) -> str:
    if text and all(c.isalnum() or c in "@%+=:,./-_" for c in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def extra_lines(text: str) -> List[str]:
    return [line.rstrip() for line in (text or "").splitlines() if line.strip()]


def extra_flags(text: str) -> List[str]:
    """Extra command-line flags typed in a free-text box (one or more per line; '#' starts a comment,
    also after a flag - a comment left in a continued command line would swallow its backslash)."""
    out = []
    for line in (text or "").splitlines():
        line = re.split(r"(?:^|\s)#", line, maxsplit=1)[0].strip()
        if line:
            out.append(line)
    return out
