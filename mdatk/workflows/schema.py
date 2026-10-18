"""Option schema shared by the script generators, the wizard and the planner.

Every generator (:class:`mdatk.workflows.base.Tool`) describes its settings as
a list of :class:`Section` objects holding :class:`Option` objects.  The GTK
wizard builds its forms from this description (check boxes, radio buttons,
combo boxes, spin buttons and text fields, each with its tooltip), the CLI
lists it, and :meth:`Tool.normalise` uses it to fill defaults and coerce values
read from a preset.

Settings of one tool are a flat ``{key: value}`` dict; a *repeated* section
(for example the list of MD stages) stores a list of such dicts under the
section key.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

KINDS = ("bool", "choice", "radio", "int", "float", "text", "multiline", "file")

LEVELS = ("error", "warning", "info")


@dataclass
class Option:
    """One user-facing setting.

    ``kind`` selects the widget: ``bool`` (check box), ``radio`` (a few
    exclusive choices), ``choice`` (combo box), ``int`` / ``float`` (spin
    button; ``optional=True`` turns it into a text field where empty means
    "not set"), ``text``, ``multiline`` (free text, e.g. extra commands) and
    ``file`` (text field with a browse button).

    ``requires`` makes the option available only when other settings have
    given values: ``{"engine": ("pmemd.MPI",)}``, ``{"shake": True}``.  The
    wizard greys the widget out otherwise and the generators ignore it.
    """

    key: str
    label: str
    kind: str
    default: Any = None
    tooltip: str = ""
    choices: Sequence[Tuple[str, str]] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    digits: int = 0
    optional: bool = False
    placeholder: str = ""
    unit: str = ""
    requires: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"option {self.key}: unknown kind {self.kind!r}")
        if self.kind in ("choice", "radio") and not self.choices:
            raise ValueError(f"option {self.key}: {self.kind} needs choices")

    @property
    def choice_values(self) -> List[str]:
        return [c[0] for c in self.choices]

    def coerce(self, value: Any) -> Any:
        """Convert ``value`` to this option's type; raise ValueError if impossible."""
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "yes", "on", "1"):
                return True
            if isinstance(value, str) and value.lower() in ("false", "no", "off", "0", ""):
                return False
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            raise ValueError(f"{self.key}: expected true/false, got {value!r}")
        if self.kind in ("int", "float"):
            if value is None or (isinstance(value, str) and not value.strip()):
                if self.optional:
                    return None
                raise ValueError(f"{self.key}: a value is required")
            if isinstance(value, bool):
                raise ValueError(f"{self.key}: expected a number, got {value!r}")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.key}: expected a number, got {value!r}") from None
            if self.kind == "int":
                if number != int(number):
                    raise ValueError(f"{self.key}: expected an integer, got {value!r}")
                number = int(number)
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"{self.key}: must be >= {self.minimum:g}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"{self.key}: must be <= {self.maximum:g}")
            return number
        if self.kind in ("choice", "radio"):
            value = "" if value is None else str(value)
            if value not in self.choice_values:
                raise ValueError(f"{self.key}: {value!r} is not one of {', '.join(self.choice_values)}")
            return value
        return "" if value is None else str(value)

    def is_enabled(self, values: Dict[str, Any]) -> bool:
        return requirements_met(self.requires, values)


def requirements_met(requires: Dict[str, Any], values: Dict[str, Any]) -> bool:
    for key, wanted in requires.items():
        have = values.get(key)
        if wanted is True:
            if not have:
                return False
        elif wanted is False:
            if have:
                return False
        elif isinstance(wanted, (tuple, list, set, frozenset)):
            if have not in wanted:
                return False
        elif have != wanted:
            return False
    return True


@dataclass
class Section:
    """A group of options (one page / frame in the wizard).

    A ``repeat`` section holds a list of records (e.g. MD stages); every
    record has the section's options.  ``templates`` are named starting
    values offered by the wizard's "Add" menu: ``(id, label, values)``.
    ``item_label`` is the option whose value names a record in lists.
    """

    key: str
    title: str
    options: List[Option]
    tooltip: str = ""
    repeat: bool = False
    item_label: str = "name"
    templates: Sequence[Tuple[str, str, Dict[str, Any]]] = ()
    default_items: Sequence[str] = ()  # template ids that form the default list
    requires: Dict[str, Any] = field(default_factory=dict)

    def option(self, key: str) -> Option:
        for opt in self.options:
            if opt.key == key:
                return opt
        raise KeyError(key)

    def record_defaults(self) -> Dict[str, Any]:
        return {o.key: copy.deepcopy(o.default) for o in self.options}

    def template(self, ident: str) -> Dict[str, Any]:
        for tid, _label, values in self.templates:
            if tid == ident:
                record = self.record_defaults()
                record.update(copy.deepcopy(values))
                return record
        raise KeyError(f"{self.key}: unknown template {ident!r}")

    def default_records(self) -> List[Dict[str, Any]]:
        return [self.template(t) for t in self.default_items]


@dataclass
class Issue:
    """A problem found by a logic check.

    ``error`` blocks writing files, ``warning`` flags a likely mistake and
    ``info`` notes a redundant or noteworthy choice.  ``keys`` name the
    offending options (``"stages.2.dt"`` for a record of a repeated section)
    so the wizard can highlight them.
    """

    level: str
    message: str
    keys: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"unknown issue level {self.level!r}")
        self.keys = tuple(self.keys)

    def __str__(self) -> str:
        return f"{self.level.upper()}: {self.message}"


def errors(issues: Sequence[Issue]) -> List[Issue]:
    return [i for i in issues if i.level == "error"]


@dataclass
class GeneratedFile:
    """A file a generator writes.

    ``name`` is relative to the step folder.  ``role`` is ``"run"`` for the
    step's main script (called by the plan's master script),
    ``"run-separate"`` for scripts meant to be started by hand (e.g. a
    production run submitted on a cluster), ``"input"`` for control files and
    ``"helper"`` for small helper programs.
    """

    name: str
    content: str
    role: str = "input"
    description: str = ""

    @property
    def executable(self) -> bool:
        return self.role in ("run", "run-separate", "helper")


# File kinds carried by planner ports: id -> (label, typical extensions)
FILE_KINDS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "pdb": ("PDB structure", (".pdb",)),
    "fasta": ("Sequence (FASTA)", (".fasta", ".fa")),
    "secstruct": ("Secondary structure (dot-bracket)", (".secstruct", ".ss", ".dbn")),
    "map": ("Density map (MRC/CCP4)", (".mrc", ".map", ".ccp4")),
    "cif": ("Restraint dictionary (mmCIF)", (".cif",)),
    "smiles": ("SMILES", (".smi",)),
    "silent": ("Rosetta silent file", (".out",)),
    "parm7": ("Amber topology", (".parm7", ".prmtop")),
    "rst7": ("Amber coordinates / restart", (".rst7", ".ncrst", ".inpcrd", ".crd")),
    "nc": ("Amber trajectory", (".nc", ".mdcrd")),
    "dist": ("Amber distance restraints (DISANG)", (".dist", ".RST")),
    "top": ("GROMACS topology", (".top",)),
    "gro": ("GROMACS coordinates", (".gro",)),
    "xtc": ("GROMACS trajectory", (".xtc", ".trr")),
    "trafl": ("SimRNA trajectory", (".trafl",)),
    "image": ("Image / movie", (".tga", ".png", ".mp4")),
    "text": ("Text file", (".txt", ".dat")),
}


def kind_label(kind: str) -> str:
    return FILE_KINDS.get(kind, (kind, ()))[0]


def guess_kind(path: str) -> str:
    low = path.lower()
    for kind, (_label, exts) in FILE_KINDS.items():
        if low.endswith(tuple(e.lower() for e in exts)):
            return kind
    return "text"


@dataclass
class Port:
    """A named input or output file slot of a planner step.

    Inputs accept every kind listed in ``kinds``; outputs have one kind.
    ``multiple`` ports carry a list of files (e.g. trajectory segments).
    """

    name: str
    label: str
    kinds: Tuple[str, ...]
    optional: bool = False
    multiple: bool = False
    tooltip: str = ""

    @property
    def kind(self) -> str:
        return self.kinds[0]

    def accepts(self, kind: str) -> bool:
        return kind in self.kinds
