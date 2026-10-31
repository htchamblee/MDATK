"""Problems that say where they are.

A :class:`Problem` is one problem found in a project.  It *is* its full text (a ``str``), so lists
of problems still join, print, compare and serialise as before, and it also carries where the
problem lies:

* ``field``: the project path of the setting, as in the project file and the GUI bindings, e.g.
  ``analyses.hbonds.group_a``, ``groups.1.residues`` or ``layout.panels.2.groups``;
* ``tab``: the GUI side-bar page that shows the setting (``analyses``);
* ``location``: the same place for people: "Analyses › Hydrogen bonds › Group A".

:class:`mdatk.selections.SelectionError` carries the same three once the code that selected knows
which setting the selection came from (:meth:`~mdatk.selections.SelectionError.locate`), so errors
raised while analysing point at their setting too.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

#: GUI side-bar pages (ids and titles of :data:`mdatk.gui.app.PAGES`)
TABS: Dict[str, str] = {
    "input": "Input & cleaning",
    "groups": "Residue groups",
    "analyses": "Analyses",
    "structure": "Structure view",
    "layout": "Layout & panels",
    "outputs": "Outputs",
}

# first component of a field path -> the page that shows it
SECTION_TABS = {
    "notes": "input", "inputs": "input", "cleaning": "input", "groups": "groups",
    "analyses": "analyses", "visual": "structure", "density": "structure", "layout": "layout",
    "output": "outputs",
}

# readable names of field paths ('#' stands for a list index); a list item's label may name the item
_LABELS = {
    "notes": "Project notes",
    "inputs.topology": "Input files › Topology",
    "inputs.trajectories": "Input files › Trajectories",
    "inputs.start": "Frames › Start", "inputs.stop": "Frames › Stop", "inputs.step": "Frames › Step",
    "inputs.dt_ps": "Frames › Time step override",
    "inputs.external": "External data",
    "inputs.external.#": "External file {n}",
    "cleaning": "Trajectory cleaning",
    "groups.#": "Group {n}{name}",
    "groups.#.selection": "Selection (overrides)",
    "groups.#.color": "Colour",
    "analyses.rmsd": "RMSD",
    "analyses.rmsd.targets.#": "Target {n}{name}",
    "analyses.contacts": "Pairwise contacts",
    "analyses.hbonds": "Hydrogen bonds",
    "analyses.mg": "Mg²⁺ coordination and occupancy",
    "analyses.mg.occupancy_selection": "Occupancy residues",
    "analyses.stacking": "π–π stacking",
    "analyses.basepairs": "Base pairs",
    "analyses.energy": "Intermolecular energy",
    "visual.representations.#": "Representation {n}",
    "visual.structure_groups": "Appearance › Groups shown",
    "visual.proximity_set": "Appearance › Proximity set",
    "visual.color_mode": "Appearance › Colouring",
    "visual.glow_mode": "Appearance › Glow",
    "visual.center_on": "Orientation › Centre on",
    "visual.zoom": "Orientation › Zoom",
    "visual.label_residues": "Residue number tags › Label residues",
    "visual.rep_rest": "Representations › Other residues",
    "visual.backbone_atom": "Representations › Backbone atom",
    "visual.renderer": "Representations › Renderer",
    "density": "Density map",
    "density.map_path": "Map file",
    "layout.preset": "Grid preset",
    "layout.panels.#": "Panel {n}{name}",
    "layout.panels.#.series": "Data",
    "output.video": "Video",
    "output.charts": "Time charts",
    "output.pdb": "Frame PDBs",
    "output.pdb.selection": "Atom subset",
    "output.video.fps": "FPS",
}
# names of the last path component wherever it occurs
_LEAF_LABELS = {
    "group_a": "Group A", "group_b": "Group B", "groups": "Groups", "selection": "Selection",
    "fit_selection": "Fit selection", "mg_selection": "Mg selection", "ligand_selection": "Ligand selection",
    "residues": "Residues", "name": "Name", "options": "Options", "style": "Style", "color": "Colouring",
}
# short names of the panel types for "Panel 3 (Contact map)"
PANEL_NAMES = {
    "structure": "Structure", "timeseries": "Time series", "dual_axis": "Dual-axis time series",
    "heatmap": "Residue heatmap", "contact_map": "Contact map", "sequence": "Sequence",
    "secondary": "Secondary structure",
}
SEP = " › "


def tab_of(field: str) -> str:
    """GUI page id of a field path (``""``: not on a side-bar page)."""
    return SECTION_TABS.get(field.split(".", 1)[0], "") if field else ""


def _item_name(parts, k: int, project: Any) -> str:
    """" (Aptamer)": what the list item ``parts[:k + 1]`` is called, from the project."""
    if project is None:
        return ""
    try:
        obj = project
        for part in parts[:k + 1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    except (AttributeError, IndexError, TypeError, ValueError):
        return ""
    if parts[0] == "layout":  # a panel is known by its type
        kind = getattr(obj, "type", "")
        name = PANEL_NAMES.get(kind, kind)
    else:
        name = getattr(obj, "name", "") or getattr(obj, "label", "")
    return f" ({name})" if name else ""


def describe(field: str, project: Any = None) -> str:
    """Readable location of a field path, e.g. ``"layout.panels.2.groups"`` -> "Layout & panels ›
    Panel 3 (Contact map) › Groups" (item names come from ``project`` when it is given)."""
    if not field:
        return ""
    parts = field.split(".")
    out = [TABS.get(tab_of(field), parts[0].capitalize())]
    for k in range(len(parts)):
        pattern = ".".join("#" if p.isdigit() else p for p in parts[:k + 1])
        label = _LABELS.get(pattern)
        if label is None and k == len(parts) - 1 and k > 0 and not parts[k].isdigit():
            label = _LEAF_LABELS.get(parts[k], parts[k].replace("_", " ").capitalize())
        if label:
            n = int(parts[k]) + 1 if parts[k].isdigit() else 0
            out.append(label.format(n=n, name=_item_name(parts, k, project) if "{name}" in label else ""))
    return SEP.join(out)


class Problem(str):
    """A problem of a project setting: the full text (``str``) plus its ``field``, ``tab`` and
    ``location``.  ``level`` is "error" (rendering would fail) or "warning" (allowed, but the setting
    is ignored or probably unintended); ``token`` is the offending part (a group name, a clause)."""

    message: str
    field: str
    location: str
    level: str
    token: str

    def __new__(cls, message: str, field: str = "", location: str = "", level: str = "error",
                token: str = "") -> "Problem":
        self = super().__new__(cls, f"{location}: {message}" if location else message)
        self.message, self.field, self.location = message, field, location
        self.level, self.token = level, token
        return self

    def __reduce__(self):  # copies and pickles keep the attributes
        return (Problem, (self.message, self.field, self.location, self.level, self.token))

    @property
    def tab(self) -> str:
        return tab_of(self.field)

    @property
    def key(self) -> str:
        """Machine-usable place: ``"<tab>:<field>"``, e.g. ``"analyses:analyses.hbonds.group_a"``."""
        return f"{self.tab}:{self.field}" if self.field else ""

    def as_dict(self) -> Dict[str, str]:
        return {"message": self.message, "field": self.field, "tab": self.tab, "location": self.location,
                "level": self.level, "token": self.token}


def problem(field: str, message: str, project: Any = None, level: str = "error", token: str = "") -> Problem:
    """A :class:`Problem` of the setting ``field`` (its location is described from ``project``)."""
    return Problem(message, field, describe(field, project), level, token)


_INDEX = re.compile(r"\.(\d+)(?=\.|$)")


def field_parents(field: str):
    """``"groups.1.residues"`` -> ``"groups.1.residues"``, ``"groups.1"``, ``"groups"``."""
    parts = field.split(".")
    for k in range(len(parts), 0, -1):
        yield ".".join(parts[:k])


def list_index(field: str, prefix: str) -> Optional[int]:
    """The list index right after ``prefix`` in ``field`` (``"groups.1.name"``, ``"groups"`` -> 1)."""
    if not field.startswith(prefix + "."):
        return None
    m = _INDEX.match(field[len(prefix):])
    return int(m.group(1)) if m else None
