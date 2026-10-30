"""Residue groups: renaming that follows every reference, and groups derived from the chains of a
topology (with the rule that decides when they may replace the project's groups)."""

import dataclasses
import os
import re
import sys
import typing

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mdff_system as ms  # noqa: E402
from conftest import make_universe  # noqa: E402

from mdatk import chaingroups, projectfile  # noqa: E402
from mdatk.config import (GroupUse, PanelSpec, PlanEdge, PlanNode, Project, RepSpec,  # noqa: E402
                          ResidueGroup, RMSDTarget)
from mdatk.io.parm7 import write_parm7  # noqa: E402
from mdatk.selections import rename_group_tokens, select  # noqa: E402

OLD, NEW = "Apt", "Binder"
STANDALONE = re.compile(r"(?<![A-Za-z0-9_.\-'])Apt(?![A-Za-z0-9_.\-'])")


def referencing_project() -> Project:
    """A project that names group Apt in every kind of place, next to the look-alikes Apter / Apt_2."""
    p = Project()
    p.notes = "Binder strand first, then the target."
    p.groups = [ResidueGroup(OLD, "1-4", "", "#2a78d6"), ResidueGroup("Tgt", "5-8", "", "#eb6834"),
                ResidueGroup("Apter", "9-10", "", "#2f9e6e"), ResidueGroup("Apt_2", "11-12", "", "#c2408f"),
                ResidueGroup("Pocket", "", "group:Apt and name P", "#7a55d3")]
    a = p.analyses
    a.rmsd.targets = [RMSDTarget("Heavy", "group:Apt and not name H*", OLD),
                      RMSDTarget("Look", "(group:Apter or group:Apt_2) and name P", " Apt ")]
    for params in (a.contacts, a.hbonds, a.mg, a.stacking, a.energy):
        params.group_a, params.group_b = OLD, "Tgt"
    a.mg.ligand_selection = "group:Apt or group:Tgt"
    a.mg.occupancy_selection = OLD
    a.stacking.selection = "(group:Apt)or group:Apter"
    v = p.visual
    v.structure_groups = [OLD, "Tgt", "Apter"]
    v.representations = [RepSpec("cartoon", [OLD, "Apt_2"]), RepSpec("licorice", ["Tgt"])]
    v.proximity_set = OLD
    p.layout.panels = [PanelSpec("structure", [], "Structure"),
                       PanelSpec("contact_map", ["contacts@Apt+Tgt/map", "contacts@Apter+Tgt/map"], "",
                                 {"mode": "instant", "note": "group:Apt"}),
                       PanelSpec("timeseries", ["rmsd@Apt/Heavy", "hbonds@Tgt+Apt/count"], "")]
    p.output.charts.series = ["energy@Apt+Tgt/total", "rmsd/Heavy"]
    p.output.pdb.selection = "Apt, Tgt"
    p.workflow.nodes = [PlanNode("n1", "vmd", 1, "", settings={"selection": "group:Apt and name P",
                                                              "records": [{"sel": "group:Apt_2"}]})]
    p.workflow.edges = [PlanEdge("n1", "out", "n1", "in")]
    return p


def toml_text(project: Project, tmp_path) -> str:
    path = str(tmp_path / "p.mdatk.toml")
    projectfile.save(path, project)
    return open(path, encoding="utf-8").read()


def test_rename_group_follows_every_reference(tmp_path):
    p = referencing_project()
    before = toml_text(p, tmp_path)
    assert STANDALONE.search(before)
    changed = p.rename_group(OLD, NEW)

    text = toml_text(p, tmp_path)
    assert not STANDALONE.search(text), [line for line in text.splitlines() if STANDALONE.search(line)]
    # look-alike tokens and names are untouched
    for kept in ("group:Apter", "group:Apt_2", '"Apter"', '"Apt_2"', "contacts@Apter+Tgt/map"):
        assert text.count(kept) == before.count(kept), kept
    assert p.group_names() == [NEW, "Tgt", "Apter", "Apt_2", "Pocket"]
    a, v = p.analyses, p.visual
    assert [t.selection for t in a.rmsd.targets] == ["group:Binder and not name H*",
                                                     "(group:Apter or group:Apt_2) and name P"]
    assert [t.fit_selection for t in a.rmsd.targets] == [NEW, " Binder "]
    pairs = {(x.group_a, x.group_b) for x in (a.contacts, a.hbonds, a.mg, a.stacking, a.energy)}
    assert pairs == {(NEW, "Tgt")}
    assert (a.mg.ligand_selection, a.mg.occupancy_selection) == ("group:Binder or group:Tgt", NEW)
    assert a.stacking.selection == "(group:Binder)or group:Apter"
    assert v.structure_groups == [NEW, "Tgt", "Apter"] and v.proximity_set == NEW
    assert [r.groups for r in v.representations] == [[NEW, "Apt_2"], ["Tgt"]]
    assert p.layout.panels[1].series == ["contacts@Binder+Tgt/map", "contacts@Apter+Tgt/map"]
    assert p.layout.panels[1].options == {"mode": "instant", "note": "group:Binder"}
    assert p.layout.panels[2].series == ["rmsd@Binder/Heavy", "hbonds@Tgt+Binder/count"]
    assert p.output.charts.series == ["energy@Binder+Tgt/total", "rmsd/Heavy"]
    assert p.output.pdb.selection == "Binder, Tgt"
    assert p.groups[4].selection == "group:Binder and name P"
    assert p.workflow.nodes[0].settings == {"selection": "group:Binder and name P",
                                            "records": [{"sel": "group:Apt_2"}]}
    assert p.notes == "Binder strand first, then the target."  # free text is never rewritten
    assert len(changed) == len(set(changed)) == 22
    assert {"analyses.contacts.group_a", "visual.representations[0].groups[0]", "layout.panels[1].series[0]",
            "layout.panels[1].options.note", "output.pdb.selection", "groups[4].selection",
            "workflow.nodes[0].settings.selection"} <= set(changed)
    assert not p.validate() or all("group" not in m for m in p.validate())
    assert Project.from_dict(p.to_dict()).to_dict() == p.to_dict()


def test_renamed_selections_still_select():
    atoms = [{"name": "P", "resid": r, "resname": "G", "element": "P"} for r in range(1, 9)]
    u = make_universe(atoms, np.zeros((1, 8, 3)))
    p = referencing_project()
    p.groups = p.groups[:2] + [ResidueGroup("Pocket", "", "group:Apt and resid 2:3", "#7a55d3")]
    p.analyses.rmsd.targets[1].selection = "Apt"
    p.rename_group(OLD, NEW)
    assert select(u, p.analyses.rmsd.targets[0].selection, p.groups).n_atoms == 4
    assert select(u, p.analyses.rmsd.targets[1].selection, p.groups).n_atoms == 4
    assert select(u, "group:Pocket", p.groups).n_atoms == 2


def test_rename_group_refuses_bad_names():
    p = referencing_project()
    snapshot = p.to_dict()
    for bad in ("", "  ", "two words", "Tgt", "Apt;x"):
        with pytest.raises(ValueError):
            p.rename_group(OLD, bad)
    with pytest.raises(KeyError):
        p.rename_group("Nope", "Other")
    assert p.rename_group(OLD, OLD) == []
    p.groups.append(ResidueGroup(OLD, "20"))
    with pytest.raises(ValueError, match="several"):
        p.rename_group(OLD, "Other")
    p.groups.pop()
    assert p.to_dict() == snapshot


def test_group_references_of_a_deleted_group():
    p = referencing_project()
    p.groups = [g for g in p.groups if g.name != OLD]
    refs = p.group_references(OLD)
    assert "analyses.energy.group_b" not in refs and "analyses.energy.group_a" in refs
    assert "output.pdb.selection" in refs and "layout.panels[1].series[0]" in refs
    assert "layout.panels[1].series[1]" not in refs  # contacts@Apter+Tgt
    assert Project().group_references("Nobody") == []


def test_rename_group_tokens():
    text = "group:Apt or group:Apter or group:Apt_2 or group:Apt.x or (group:Apt)and subgroup:Apt"
    assert rename_group_tokens(text, "Apt", "B") == (
        "group:B or group:Apter or group:Apt_2 or group:Apt.x or (group:B)and subgroup:B")


def _all_string_fields(cls=Project, seen=None):
    """(class, field, hint) of every str / List[str] field reachable from ``cls``."""
    seen = set() if seen is None else seen
    if cls in seen:
        return []
    seen.add(cls)
    out = []
    hints = typing.get_type_hints(cls, include_extras=True)
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        plain = typing.get_type_hints(cls)[f.name]
        inner = [a for a in typing.get_args(plain) if dataclasses.is_dataclass(a)]
        if dataclasses.is_dataclass(plain) or inner:
            out += _all_string_fields(plain if dataclasses.is_dataclass(plain) else inner[0], seen)
        elif plain in (str, typing.List[str], typing.Optional[str]):
            out.append((cls.__name__, f.name, tp))
    return out


def test_fields_that_name_groups_are_marked():
    """A new field that holds group names or selections must say so (GroupName / Selection /
    DataKey), or a renamed group would not follow it."""
    fields = _all_string_fields()
    assert len(fields) > 60
    unmarked = []
    for owner, name, tp in fields:
        marked = "GroupUse" in repr(tp)
        if re.search(r"(^|_)groups?(_|$)|selection|_set$|series", name) and not marked:
            if (owner, name) not in {("ResidueGroup", "selection")}:  # tokens only; a bare name is no group
                unmarked.append(f"{owner}.{name}")
    assert not unmarked
    assert GroupUse("name") == GroupUse("name")


# ---------------------------------------------------------------------------------------------------
# chain groups
# ---------------------------------------------------------------------------------------------------


def system_atoms(seq="GCGC"):
    """A D duplex (chains A/B) and an L duplex (C/D), numbered 1-n per chain, with waters (chain W)
    and a Mg2+ (chain M) numbered from 1 as well."""
    return ms.with_solvent(ms.atoms_of(ms.mixed_pdb(seq)))


def write_parm7_bonded(path, atoms, chain_ids=True):
    """parm7 with bonds (covalent distances) and, optionally, RESIDUE_CHAINID; no coordinates."""
    from scipy.spatial import cKDTree

    labels, pointers, chains, last = [], [], [], None
    for k, a in enumerate(atoms):
        if a[:3] != last:
            labels.append(a[2])
            pointers.append(k + 1)
            chains.append(a[0])
            last = a[:3]
    xyz = np.array([a[5] for a in atoms])
    with_h, without_h = [], []
    for i, j in sorted(cKDTree(xyz).query_pairs(1.75)):
        hydrogen = "H" in (atoms[i][4], atoms[j][4])
        if hydrogen and np.linalg.norm(xyz[i] - xyz[j]) > 1.2:
            continue  # a hydrogen bond, not a covalent one
        (with_h if hydrogen else without_h).extend([3 * i, 3 * j, 1])
    n = len(atoms)
    ptr = [0] * 31
    ptr[0], ptr[1], ptr[2], ptr[3], ptr[11] = n, 1, len(with_h) // 3, len(without_h) // 3, len(labels)
    sections = [("POINTERS", "i", ptr), ("ATOM_NAME", "a", [a[3] for a in atoms]), ("CHARGE", "e", [0.0] * n),
                ("ATOMIC_NUMBER", "i", [ms.NUMBERS.get(a[4], 6) for a in atoms]),
                ("MASS", "e", [ms.MASSES[ms.NUMBERS.get(a[4], 6)] for a in atoms]),
                ("ATOM_TYPE_INDEX", "i", [1] * n), ("NONBONDED_PARM_INDEX", "i", [1]),
                ("RESIDUE_LABEL", "a", labels), ("RESIDUE_POINTER", "i", pointers),
                ("BONDS_INC_HYDROGEN", "i", with_h), ("BONDS_WITHOUT_HYDROGEN", "i", without_h),
                ("ANGLES_INC_HYDROGEN", "i", []), ("ANGLES_WITHOUT_HYDROGEN", "i", []),
                ("DIHEDRALS_INC_HYDROGEN", "i", []), ("DIHEDRALS_WITHOUT_HYDROGEN", "i", []),
                ("LENNARD_JONES_ACOEF", "e", [1.0e6]), ("LENNARD_JONES_BCOEF", "e", [1.0e3]),
                ("AMBER_ATOM_TYPE", "a", [a[4] for a in atoms])]
    if chain_ids:
        sections.append(("RESIDUE_CHAINID", "a", chains))
    write_parm7(path, sections, title=ms.TEST_TITLE)
    return path


def write_pdb(path, atoms, chains=True):
    lines = []
    for k, (chain, num, resname, name, el, (x, y, z)) in enumerate(atoms):
        record = "HETATM" if resname in ("WAT", "MG") else "ATOM  "
        nm = f" {name:<3s}" if len(name) < 4 else name
        lines.append(f"{record}{k + 1:5d} {nm} {resname:>3s} {chain if chains else ' '}{num:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}")
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\nEND\n")
    return path


def load(path):
    import warnings

    import MDAnalysis as mda

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mda.Universe(path, **({"topology_format": "PARM7"} if path.endswith(".parm7") else {}))


def assert_exact(u, chains, expected):
    """Every chain group selects exactly the polymer atoms of its chain (no water, no ion)."""
    assert [(c.group.name, c.group.residues, c.group.selection) for c in chains] == expected
    polymer = u.select_atoms("not resname WAT MG")
    seen = []
    for c in chains:
        ag = select(u, f"group:{c.group.name}", [c.group])
        assert not set(ag.resnames) & {"WAT", "MG"}
        assert len(set(ag.resindices)) == c.n_residues
        seen.extend(ag.ix)
    assert sorted(seen) == sorted(polymer.ix)
    colours = [c.group.color for c in chains]
    assert len(set(colours)) == len(colours) and all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colours)
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+", c.group.name) for c in chains)


def test_chain_groups_from_pdb_chain_ids(tmp_path):
    u = load(write_pdb(str(tmp_path / "m.pdb"), system_atoms()))
    chains = chaingroups.chain_groups(u)
    # every chain (and the water) is numbered from 1: the ranges need their chain
    assert_exact(u, chains, [("Chain_A", "A:1-4", ""), ("Chain_B", "B:1-4", ""), ("Chain_C", "C:1-4", ""),
                             ("Chain_D", "D:1-4", "")])
    assert {c.source for c in chains} == {"chainID"}


def test_chain_groups_from_parm7_chain_ids_and_from_bonded_fragments(tmp_path):
    atoms = system_atoms()
    u = load(write_parm7_bonded(str(tmp_path / "c.parm7"), atoms))
    assert hasattr(u.atoms, "chainIDs")
    chains = chaingroups.chain_groups(u)
    assert_exact(u, chains, [("Chain_A", "1-4", ""), ("Chain_B", "5-8", ""), ("Chain_C", "9-12", ""),
                             ("Chain_D", "13-16", "")])
    path = write_parm7_bonded(str(tmp_path / "n.parm7"), atoms, chain_ids=False)
    u = load(path)
    assert not hasattr(u.atoms, "chainIDs") and set(u.segments.segids) == {"SYSTEM"}
    chains = chaingroups.chain_groups(u)
    assert_exact(u, chains, [("Chain_1", "1-4", ""), ("Chain_2", "5-8", ""), ("Chain_3", "9-12", ""),
                             ("Chain_4", "13-16", "")])
    assert {c.source for c in chains} == {"fragment"}
    assert [c.group.name for c in chaingroups.topology_chain_groups(path)] == [c.group.name for c in chains]


def test_chain_groups_without_chain_ids_or_bonds():
    """In-memory universe: one segment, no bonds; consecutive residues within covalent distance link."""
    atoms, xyz = [], []
    for strand, x0 in ((0, 0.0), (1, 20.0)):
        for k in range(3):
            resid = 3 * strand + k + 1
            for name, el, dz in (("P", "P", 0.0), ("C4'", "C", 1.5), ("O3'", "O", 3.0)):
                atoms.append({"name": name, "resid": resid, "resname": "G", "element": el})
                xyz.append((x0, 0.0, 4.6 * k + dz))  # O3' to the next P: 1.6 A
    atoms += [{"name": n, "resid": 7, "resname": "HOH", "element": e} for n, e in (("O", "O"), ("H1", "H"),
                                                                                  ("H2", "H"))]
    xyz += [(40.0, 0, 0), (40.96, 0, 0), (39.76, 0.93, 0)]
    atoms.append({"name": "MG", "resid": 8, "resname": "MG", "element": "MG"})
    xyz.append((50.0, 0, 0))
    u = make_universe(atoms, np.array([xyz]))
    chains = chaingroups.chain_groups(u)
    assert [(c.group.name, c.group.residues, c.n_residues, c.source) for c in chains] == [
        ("Chain_1", "1-3", 3, "fragment"), ("Chain_2", "4-6", 3, "fragment")]
    assert chaingroups.polymer_mask(u).tolist() == [True] * 6 + [False, False]


def test_chain_groups_fall_back_to_atom_indices(tmp_path):
    """No chain IDs, numbering restarting in every chain: only an index selection is exact."""
    u = load(write_pdb(str(tmp_path / "n.pdb"), system_atoms(), chains=False))
    chains = chaingroups.chain_groups(u)
    assert [c.group.selection for c in chains] == ["index 0:128", "index 129:257", "index 258:386",
                                                   "index 387:515"]
    assert_exact(u, chains, [(c.group.name, "", c.group.selection) for c in chains])


def test_chain_group_names_and_colours_avoid_existing_groups(tmp_path):
    u = load(write_pdb(str(tmp_path / "m.pdb"), system_atoms()))
    existing = [ResidueGroup("Chain_A", "B:1-2", "", "#2a78d6"),
                ResidueGroup("Strand", "C:1-4", "", "#EB6834")]
    chains = chaingroups.chain_groups(u, existing)
    assert [c.group.name for c in chains] == ["Chain_A_2", "Chain_B", "Chain_C", "Chain_D"]
    assert [c.covered_by for c in chains] == ["", "", "Strand", ""]
    assert not {c.group.color for c in chains} & {"#2a78d6", "#eb6834"}


# ---------------------------------------------------------------------------------------------------
# when chain groups may replace the project's groups
# ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def four_chains(tmp_path_factory):
    d = tmp_path_factory.mktemp("chains")
    return write_parm7_bonded(str(d / "four.parm7"), system_atoms())


def test_untouched_defaults_are_replaced_and_their_references_follow(four_chains, tmp_path):
    p = Project()
    assert chaingroups.untouched_defaults(p.groups)
    chains = chaingroups.topology_chain_groups(four_chains, p.groups)
    merge = chaingroups.merge_chain_groups(p, chains, auto=True)
    assert merge.replaced == {"Aptamer": "Chain_A", "Target": "Chain_B"}
    assert merge.added == ["Chain_C", "Chain_D"]
    assert p.group_names() == ["Chain_A", "Chain_B", "Chain_C", "Chain_D"]
    assert [g.color for g in p.groups[:2]] == ["#2a78d6", "#eb6834"]  # the defaults' colours
    assert len({g.color for g in p.groups}) == 4
    a = p.analyses
    assert {(x.group_a, x.group_b) for x in (a.contacts, a.hbonds, a.mg, a.stacking, a.energy)} == {
        ("Chain_A", "Chain_B")}
    assert a.rmsd.targets[0].selection == "(group:Chain_A or group:Chain_B) and not name H*"
    assert p.visual.proximity_set == "Chain_B"
    assert not [m for m in p.validate() if "group" in m.lower()]
    text = toml_text(p, tmp_path)
    assert "group:Aptamer" not in text and "group:Target" not in text
    assert "Default groups replaced" in merge.summary("four.parm7")


def test_defaults_that_fit_the_chains_are_kept(tmp_path):
    """The demo-like case: Aptamer = 1-12 and Target = 13-24 already are the first two chains."""
    path = write_parm7_bonded(str(tmp_path / "long.parm7"), system_atoms("GCGCAUGCAUGC"))
    p = Project()
    chains = chaingroups.topology_chain_groups(path, p.groups)
    assert [c.covered_by for c in chains] == ["Aptamer", "Target", "", ""]
    merge = chaingroups.merge_chain_groups(p, chains, auto=True)
    assert not merge.changed and merge.pending == ["Chain_C", "Chain_D"]
    assert chaingroups.untouched_defaults(p.groups)
    merge = chaingroups.merge_chain_groups(p, chains)  # the button
    assert merge.added == ["Chain_C", "Chain_D"] and merge.covered == ["Chain_A", "Chain_B"]
    assert p.group_names() == ["Aptamer", "Target", "Chain_C", "Chain_D"]


def test_one_chain_keeps_the_defaults(tmp_path):
    atoms = [a for a in system_atoms("GCGCAUGCAUGCAU") if a[0] == "A"]  # 14 nt: longer than Aptamer
    path = write_parm7_bonded(str(tmp_path / "one.parm7"), atoms, chain_ids=False)
    p = Project()
    chains = chaingroups.topology_chain_groups(path, p.groups)
    assert [(c.group.name, c.group.residues, c.covered_by) for c in chains] == [
        ("Chain_1", "1-14", "")]
    merge = chaingroups.merge_chain_groups(p, chains, auto=True)
    assert not merge.changed and merge.pending == ["Chain_1"]
    assert chaingroups.untouched_defaults(p.groups)
    assert chaingroups.merge_chain_groups(p, chains).added == ["Chain_1"]
    assert p.group_names() == ["Aptamer", "Target", "Chain_1"]


def test_empty_groups_are_filled(four_chains):
    p = Project()
    p.groups = []
    merge = chaingroups.merge_chain_groups(p, chaingroups.topology_chain_groups(four_chains), auto=True)
    assert merge.added == ["Chain_A", "Chain_B", "Chain_C", "Chain_D"] and not merge.replaced
    assert p.analyses.contacts.group_a == "Aptamer"  # dangling before, untouched now


def test_user_groups_are_never_overwritten(four_chains):
    p = Project()
    p.groups[0].residues = "1-5"  # edited: no longer the defaults
    p.groups.append(ResidueGroup("Third", "9-12", "", "#123456"))
    before = [dataclasses.replace(g) for g in p.groups]
    chains = chaingroups.topology_chain_groups(four_chains, p.groups)
    merge = chaingroups.merge_chain_groups(p, chains, auto=True)
    assert not merge.changed and merge.pending == ["Chain_A", "Chain_B", "Chain_D"]
    assert merge.covered == ["Chain_C"] and p.groups == before
    assert "'Groups from chains' adds them" in merge.summary("four.parm7")
    merge = chaingroups.merge_chain_groups(p, chains)
    assert merge.added == ["Chain_A", "Chain_B", "Chain_D"] and p.groups[:3] == before
    assert p.analyses.contacts.group_a == "Aptamer"
    again = chaingroups.merge_chain_groups(p, chaingroups.topology_chain_groups(four_chains, p.groups))
    assert not again.changed and "already has a group" in again.summary("four.parm7")
