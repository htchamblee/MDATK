"""Residue groups from the chains of a loaded topology.

One group per polymer chain, water and ions left out.  The chains are, as for the ``chain`` colour
mode: the chain IDs when they name several chains (PDB, or a parm7 written with RESIDUE_CHAINID),
else the segment ids when they name several, else the bonded fragments in topology order (from the
topology's bonds, or from consecutive residues within covalent distance when it has none).  Every
group's residue spec is checked to select exactly its chain's atoms in the universe it came from,
through the selection code the analyses use: plain residue ranges when the residue numbers are
unique, else chain-qualified ranges (``A:1-40``), else a selection.

:func:`merge_chain_groups` decides what may change in a project.  Chain groups only fill groups that
are empty, or replace the untouched default groups (``Aptamer`` / ``Target``) when there are at least
as many chains and the defaults do not already select a chain each; then every reference to a
default follows the chain group that takes its place (``Aptamer`` -> the first chain, ``Target`` ->
the second; see :meth:`mdatk.config.Project.rename_group`).  Otherwise existing groups are never
changed: on request the chains that no group selects yet are added.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Project, ResidueGroup, _default_groups, group_name_problem
from .selections import _ION_ELEMENTS, SelectionError, atom_elements, select

# residue names of water models (also recognised by composition: one oxygen, hydrogens, virtual sites)
WATER_RESNAMES = frozenset({
    "WAT", "HOH", "H2O", "DOD", "SOL", "TIP", "TIP3", "TIP4", "TIP5", "TP3", "TP4", "TP5", "T3P", "T4P",
    "T5P", "T4E", "SPC", "SPCE", "OPC", "OPC3", "PL3", "WTR", "TIP3P", "TIP4P", "TIP5P"})
ION_RESNAMES = frozenset(_ION_ELEMENTS) | {"MN2", "ZN2", "CA2", "CD", "CO", "NI", "CU", "SR", "BA", "IOD"}
LINK_MAX = 2.0  # largest heavy-atom distance (Å) that bonds two consecutive residues without bond records
# distinct colours, first the default groups' blue and orange; then golden-angle hues
CHAIN_COLORS = ("#2a78d6", "#eb6834", "#2f9e6e", "#c2408f", "#7a55d3", "#18a0b4", "#8c5a2b", "#5e7391",
                "#a3b02a", "#d4789c")
_RANGE_CHAIN = re.compile(r"[A-Za-z][A-Za-z0-9]*")  # chain qualifiers residue ranges accept
_SELECTION_WORD = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class ChainGroup:
    """A residue group for one chain, checked against the universe it was made from."""

    group: ResidueGroup
    label: str  # chain ID, segment id or fragment number
    source: str  # "chainID", "segid" or "fragment"
    n_residues: int
    covered_by: str = ""  # existing group that already selects exactly this chain's atoms


@dataclass
class ChainMerge:
    """What :func:`merge_chain_groups` changed."""

    added: List[str] = field(default_factory=list)  # chain groups added
    replaced: Dict[str, str] = field(default_factory=dict)  # default group -> chain group in its place
    references: List[str] = field(default_factory=list)  # locations rewritten for the replaced defaults
    covered: List[str] = field(default_factory=list)  # chains an existing group already selects
    pending: List[str] = field(default_factory=list)  # chains left to the button (user-made groups)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.replaced)

    def summary(self, source: str) -> str:
        """One line for the GUI, ``source``: where the chains come from (a file name)."""
        parts = []
        if self.replaced:
            pairs = ", ".join(f"{a} → {b}" for a, b in self.replaced.items())
            parts.append(f"Default groups replaced by the chains of {source} ({pairs}; "
                         f"{len(self.references)} reference(s) updated)")
        if self.added:
            parts.append(f"Added {len(self.added)} group(s) from the chains of {source}: "
                         f"{', '.join(self.added)}")
        if self.pending:
            parts.append(f"{source}: {len(self.pending)} chain(s) without a group "
                         f"({', '.join(self.pending)}); 'Groups from chains' adds them")
        if not parts:
            parts.append(f"Every chain of {source} already has a group" if self.covered
                         else f"No polymer chains found in {source}")
        return ". ".join(parts) + "."


def untouched_defaults(groups: Sequence[ResidueGroup]) -> bool:
    """The groups are still exactly the defaults a new project starts with."""
    return list(groups) == _default_groups()


# ---------------------------------------------------------------------------
# chains of a universe
# ---------------------------------------------------------------------------


def polymer_mask(universe) -> np.ndarray:
    """Per residue: neither water nor an ion."""
    residues = universe.residues
    n_atoms = np.bincount(universe.atoms.resindices, minlength=len(residues))
    names = np.array([str(r).strip().upper() for r in residues.resnames], dtype=object)
    water = np.isin(names, list(WATER_RESNAMES))
    ions = (n_atoms == 1) | np.isin(names, list(ION_RESNAMES))
    for k in np.flatnonzero((n_atoms <= 5) & ~water & ~ions):  # water models by composition
        atoms = residues[k].atoms
        elements = atom_elements(atoms)
        heavy = elements != "H"
        try:  # virtual sites (TIP4P extra points) weigh nothing
            heavy &= np.asarray(atoms.masses, dtype=float) >= 0.5
        except Exception:  # noqa: BLE001 - no masses: count every non-hydrogen
            pass
        water[k] = list(elements[heavy]) == ["O"]
    return ~(water | ions)


def _labels(values) -> List[str]:
    return [str(v).strip() for v in values]


def _usable(labels: Sequence[str]) -> bool:
    """Labels that tell several chains apart (every residue labelled, MDAnalysis' default excluded)."""
    return all(labels) and len(set(labels)) > 1 and "SYSTEM" not in labels


def _fragments(universe, residues, first_atoms) -> List[int]:
    """Fragment number per residue, counted in residue order: bonded fragments, else consecutive
    residues within covalent distance, else one fragment."""
    try:
        if len(universe.bonds):
            frag = [int(f) for f in first_atoms.fragindices]
            order = {f: k for k, f in enumerate(dict.fromkeys(frag))}
            return [order[f] for f in frag]
    except Exception:  # noqa: BLE001 - no bond records
        pass
    try:
        positions = universe.atoms.positions
    except Exception:  # noqa: BLE001 - a topology without coordinates
        return [0] * len(residues)
    heavy_atoms = []
    for res in residues:
        idx = res.atoms.ix[atom_elements(res.atoms) != "H"]
        heavy_atoms.append(positions[idx if len(idx) else res.atoms.ix])
    out, k = [0], 0
    for a, b in zip(heavy_atoms[:-1], heavy_atoms[1:]):
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min()
        k += 0 if d <= LINK_MAX else 1
        out.append(k)
    return out


def polymer_chains(universe) -> List[Tuple[str, str, object]]:
    """``[(label, source, residues)]`` of the polymer chains in topology order."""
    residues = universe.residues[polymer_mask(universe)]
    if not len(residues):
        return []
    first_atoms = residues.atoms[np.unique(residues.atoms.resindices, return_index=True)[1]]
    try:
        chain_ids = _labels(first_atoms.chainIDs)
    except Exception:  # noqa: BLE001 - the topology has no chain IDs
        chain_ids = []
    segids = _labels(residues.segids)
    if _usable(chain_ids):
        labels, source = chain_ids, "chainID"
    elif _usable(segids):
        labels, source = segids, "segid"
    else:
        frag = _fragments(universe, residues, first_atoms)
        labels, source = [str(f + 1) for f in frag], "fragment"
        single = [lab for lab in (chain_ids, segids) if lab and all(lab) and len(set(lab)) == 1
                  and lab[0] != "SYSTEM"]
        if max(frag) == 0 and single:  # one chain: name it after its chain ID / segment
            labels, source = single[0], "chainID" if single[0] is chain_ids else "segid"
    arr = np.array(labels, dtype=object)
    return [(lab, source, residues[arr == lab]) for lab in dict.fromkeys(labels)]


def _runs(values: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for v in sorted(set(int(x) for x in values)):
        if out and v == out[-1][1] + 1:
            out[-1] = (out[-1][0], v)
        else:
            out.append((v, v))
    return out


def _spec_candidates(universe, label: str, source: str, residues) -> List[Tuple[str, str]]:
    """``(residues, selection)`` specs for a chain, simplest first."""
    runs = _runs(residues.resids)
    items = [f"{a}-{b}" if a != b else f"{a}" for a, b in runs]
    out = [(", ".join(items), "")]
    has_chain_ids = hasattr(universe.atoms, "chainIDs")
    qualifier_reads = "chainID" if has_chain_ids else "segid"  # see selections.ranges_to_selection
    if source == qualifier_reads and _RANGE_CHAIN.fullmatch(label):
        out.append((", ".join(f"{label}:{it}" for it in items), ""))
    if source in ("chainID", "segid") and _SELECTION_WORD.fullmatch(label):
        resid = " ".join(f"{a}:{b}" if a != b else f"{a}" for a, b in runs)
        out.append(("", f"{source} {label} and resid {resid}"))
    atoms = " ".join(f"{a}:{b}" if a != b else f"{a}" for a, b in _runs(residues.atoms.ix))
    out.append(("", f"index {atoms}"))
    return out


def _atoms_of(universe, group: ResidueGroup, groups: Sequence[ResidueGroup]):
    try:
        return np.sort(select(universe, f"group:{group.name}", groups, allow_empty=True).ix)
    except (SelectionError, ValueError, KeyError):
        return None


def _free_name(base: str, taken) -> str:
    name, k = base, 2
    while name in taken:
        name, k = f"{base}_{k}", k + 1
    return name


def _colors(used: Sequence[str]):
    """Colours not in ``used`` (case-insensitive): the palette first, then golden-angle hues."""
    taken = {c.strip().lower() for c in used}
    for c in CHAIN_COLORS:
        if c not in taken:
            yield c
    k = 0
    while True:
        r, g, b = colorsys.hsv_to_rgb((0.11 + 0.618034 * k) % 1.0, 0.62, 0.78)
        k += 1
        c = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        if c not in taken:
            yield c


def chain_groups(universe, existing: Sequence[ResidueGroup] = ()) -> List[ChainGroup]:
    """One verified group per polymer chain of ``universe``.  Names (``Chain_A``, or ``Chain_1`` ...
    for fragments) and colours differ from those of ``existing``; ``covered_by`` names the existing
    group that already selects exactly a chain's atoms."""
    taken = {g.name for g in existing}
    colours = _colors([g.color for g in existing])
    known: Dict[bytes, str] = {}
    for g in existing:
        ix = _atoms_of(universe, g, existing)
        if ix is not None and len(ix):
            known.setdefault(ix.tobytes(), g.name)
    out = []
    for label, source, residues in polymer_chains(universe):
        name = _free_name("Chain_" + re.sub(r"[^A-Za-z0-9_.\-]", "_", label), taken)
        if group_name_problem(name):
            continue
        want = np.sort(residues.atoms.ix)
        for spec, selection in _spec_candidates(universe, label, source, residues):
            group = ResidueGroup(name, spec, selection, "")
            got = _atoms_of(universe, group, [group])
            if got is not None and np.array_equal(got, want):
                break
        else:  # pragma: no cover - an index selection always matches
            continue
        taken.add(name)
        group.color = next(colours)
        out.append(ChainGroup(group, label, source, len(residues), known.get(want.tobytes(), "")))
    return out


def topology_chain_groups(path: str, existing: Sequence[ResidueGroup] = ()) -> List[ChainGroup]:
    """:func:`chain_groups` of a topology (or structure) file, read on its own (a parm7 has no
    coordinates, which :func:`mdatk.io.loader.load_universe` expects)."""
    import warnings

    import MDAnalysis as mda

    from .io.loader import is_parm7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe = mda.Universe(path, **({"topology_format": "PARM7"} if is_parm7(path) else {}))
    return chain_groups(universe, existing)


# ---------------------------------------------------------------------------
# applying them to a project
# ---------------------------------------------------------------------------


def merge_chain_groups(project: Project, chains: Sequence[ChainGroup], auto: bool = False) -> ChainMerge:
    """Bring the chain groups into ``project`` (see the module docstring for the rule).  ``auto``: the
    topology was just chosen; only empty or untouched default groups change, and the chains that no
    group selects yet are listed in ``pending`` instead of being added."""
    out = ChainMerge()
    if not chains:
        return out
    defaults = untouched_defaults(project.groups)
    fits = {g.name for g in project.groups} <= {c.covered_by for c in chains}
    if defaults and not fits and len(chains) >= len(project.groups):
        for g, c in zip(list(project.groups), chains):
            old = g.name
            new = _free_name(c.group.name, {x.name for x in project.groups if x is not g})
            out.references += project.rename_group(old, new)
            g.residues, g.selection = c.group.residues, c.group.selection  # keeps the default's colour
            out.replaced[old] = new
        out.references = list(dict.fromkeys(out.references))
        rest = chains[len(out.replaced):]
    elif not project.groups:
        rest = list(chains)
    else:
        out.covered = [c.group.name for c in chains if c.covered_by]
        rest = [c for c in chains if not c.covered_by]
        if auto:
            out.pending = [c.group.name for c in rest]
            return out
    for c in rest:
        name = _free_name(c.group.name, set(project.group_names()))
        project.groups.append(ResidueGroup(name, c.group.residues, c.group.selection, c.group.color))
        out.added.append(name)
    return out
