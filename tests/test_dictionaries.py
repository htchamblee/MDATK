"""Tests for mdatk.dictionaries (L-RNA restraint dictionaries) and the 'restraint-dicts' planner step."""

import io
import json
import os
import random
import stat
import subprocess
import sys
import urllib.error

import pytest

from mdatk import dictionaries as D
from mdatk.workflows import get_tool
from mdatk.workflows.base import PortValue, StepContext

STAMP = "20261018-1200"
VARIANTS = ["0A5", "0A3", "0C5", "0C3", "0G5", "0G3", "0U5", "0U3"]
# heavy atoms / all atoms of the CCD entries (number_atoms_nh / number_atoms_all of the monomer library)
CCD_COUNTS = {"0A": (23, 35), "0C": (21, 33), "0G": (24, 36), "0U": (21, 32)}
# PubChem-style (Kekule, neutral) SMILES of natural D-adenosine and its mirror image
D_ADENOSINE = "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N"
L_ADENOSINE = "C1=NC(=C2C(=N1)N(C=N2)[C@@H]3[C@H]([C@H]([C@@H](O3)CO)O)O)N"
# the descriptor stored in the CCD file of 0A (CACTVS): D configuration
CCD_0A_DESCRIPTOR = "Nc1ncnc2n(cnc12)[C@@H]3O[C@H](CO[P](O)(O)=O)[C@@H](O)[C@H]3O"


@pytest.fixture(autouse=True)
def _fresh_pubchem_cache():
    from mdatk.workflows import dictstep
    dictstep._PUBCHEM.clear()
    yield
    dictstep._PUBCHEM.clear()


# --- reference table ----------------------------------------------------------------------------------

def test_component_table_and_variant_conventions():
    assert list(D.COMPONENTS) == VARIANTS + ["0A", "0C", "0G", "0U"]
    for code, (nh, n_all) in CCD_COUNTS.items():
        comp = D.COMPONENTS[code]
        atoms = comp.residue_atoms()
        assert len(atoms) == n_all and len(comp.atoms) == nh and not comp.leaving
        assert {"OP3", "P", "OP1", "OP2", "HO3'"} <= set(atoms) and "HO5'" not in atoms
        five, three = D.COMPONENTS[code + "5"], D.COMPONENTS[code + "3"]
        a5, a3 = set(five.residue_atoms()), set(three.residue_atoms())
        assert not a5 & {"P", "OP1", "OP2", "OP3"} and {"O5'", "HO5'", "HO3'"} <= a5
        assert a5 == set(atoms) - {"P", "OP1", "OP2", "OP3"} | {"HO5'"}
        assert a3 == set(atoms) - {"OP3"} and three.leaving == ("OP3",)
        assert three.smiles == comp.smiles  # generated from the parent, OP3 removed afterwards
        assert five.parent == three.parent == code and five.terminal == "5" and three.terminal == "3"
    assert D.COMPONENTS["0A5"].formula() == "C10H13N5O4"
    assert D.COMPONENTS["0G3"].formula() == "C10H12N5O8P-2"
    with pytest.raises(D.DictionaryError, match="unknown residue code"):
        D.get_component("XYZ")


@pytest.mark.parametrize("code", list(D.COMPONENTS))
def test_builtin_smiles_match_reference_graph_and_stereo(code):
    comp = D.COMPONENTS[code]
    assert D.check_smiles(code, comp.smiles) == []
    assert D.parse_smiles(comp.smiles).formula() == comp.formula()
    assert code in D.identify_smiles(comp.smiles)


def test_smiles_checks_reject_wrong_stereo_and_compounds():
    assert D.check_smiles("0A5", L_ADENOSINE) == []
    with pytest.raises(D.DictionaryError, match="opposite configuration"):
        D.check_smiles("0A5", D_ADENOSINE)
    with pytest.raises(D.DictionaryError, match="opposite configuration"):
        D.check_smiles("0A", CCD_0A_DESCRIPTOR)
    with pytest.raises(D.DictionaryError, match="does not specify"):
        D.check_smiles("0A5", "Nc1ncnc2n(cnc12)C3OC(CO)C(O)C3O")
    with pytest.raises(D.DictionaryError, match="does not match 0G5"):
        D.check_smiles("0G5", L_ADENOSINE)
    with pytest.raises(D.DictionaryError, match="tautomer"):  # an extra H on N7 instead of N1
        D.check_smiles("0G5", "OC[C@@H]1O[C@H](N2C=[NH+]C3=C2N=C(N)NC3=O)[C@@H](O)[C@H]1O")
    # neutral phosphate: accepted with a note (protonation of OP2 / OP3 may differ)
    neutral = D.COMPONENTS["0U"].smiles.replace("[O-]P(=O)([O-])", "OP(=O)(O)")
    notes = D.check_smiles("0U3", neutral)
    assert any("protonation" in n for n in notes)
    assert D.identify_smiles(L_ADENOSINE) == ["0A5"]
    assert D.identify_smiles(D_ADENOSINE) == []


def test_smiles_parser_details():
    g = D.parse_smiles("c1ccncc1")  # aromatic: implicit H from the lowest valence
    assert g.formula() == "C5H5N" and g.hcount == [1, 1, 1, 0, 1, 1]
    g = D.parse_smiles("[NH4+].[Cl-]")
    assert g.formula() == "ClH4N" and g.charges == [1, -1]
    g = D.parse_smiles("[H][C@](F)(Cl)Br")  # explicit H atom folded into its neighbour
    assert g.formula() == "CHBrClF" and g.stereo[0][0][0] == "H"
    g = D.parse_smiles("C1CC%10CC1.C%10")
    assert len(g) == 6 and 5 in g.adj[2]
    for bad in ("C1CC", "C(C", "C)", "[Xx", "", "Q"):
        with pytest.raises(D.DictionaryError):
            D.parse_smiles(bad)


def test_graph_matching_recovers_names_deterministically():
    comp = D.COMPONENTS["0G3"]
    ref = comp.graph()
    rng = random.Random(3)
    perm = list(range(len(ref)))
    rng.shuffle(perm)
    tgt = D.MolGraph()
    inverse = {old: new for new, old in enumerate(perm)}
    for old in perm:
        tgt.add_atom(ref.elements[old], ref.charges[old], ref.hcount[old])
    for i in range(len(ref)):
        for j, order in ref.adj[i].items():
            if i < j:
                tgt.add_bond(inverse[i], inverse[j], order)
    m = D.match_graph(ref, tgt)
    assert m == D.match_graph(ref, tgt)
    symmetric = {"OP2", "OP3"}  # same element, charge, hydrogens and bond order: either way round
    assert all(perm[m[i]] == i for i in range(len(ref)) if ref.names[i] not in symmetric)
    assert {ref.names[perm[m[i]]] for i in range(len(ref)) if ref.names[i] in symmetric} == symmetric
    # a bond removed: not isomorphic any more
    a, b = next((i, j) for i in range(len(tgt)) for j in tgt.adj[i])
    del tgt.adj[a][b], tgt.adj[b][a]
    assert D.match_graph(ref, tgt) is None


# --- RDKit cross-checks (testing only; RDKit is no runtime dependency) ---------------------------------

def _rdkit_reference(comp, Chem):
    """RDKit molecule built from the reference atom-name graph (no stereo)."""
    rw = Chem.RWMol()
    index = {}
    hmap = comp.h_names()
    for name, el, q in comp.atoms:
        atom = Chem.Atom(el)
        atom.SetFormalCharge(q)
        atom.SetNumExplicitHs(len(hmap.get(name, ())))
        atom.SetNoImplicit(True)
        index[name] = rw.AddAtom(atom)
    for a, b, order in comp.bonds:
        rw.AddBond(index[a], index[b], Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("code", list(D.COMPONENTS))
def test_rdkit_smiles_formula_graph_and_l_configuration(code):
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem, rdMolDescriptors

    comp = D.COMPONENTS[code]
    mol = Chem.MolFromSmiles(comp.smiles)
    assert mol is not None
    assert rdMolDescriptors.CalcMolFormula(mol) == comp.formula()
    # the SMILES graph is the reference atom-name graph
    ref = _rdkit_reference(comp, Chem)
    assert Chem.MolToSmiles(ref, isomericSmiles=False) == Chem.MolToSmiles(mol, isomericSmiles=False)
    # RDKit keeps the SMILES atom order; map it to the names with MDATK's matcher
    names, _ = D.smiles_mapping(comp, comp.smiles)
    cip = {names[a]: lab for a, lab in Chem.FindMolChiralCenters(mol, includeUnassigned=True)}
    assert cip == {"C1'": "S", "C2'": "S", "C3'": "R", "C4'": "S"}  # L-ribose (D: R, R, S, R)
    parent = Chem.MolFromSmiles(D.COMPONENTS[comp.parent].smiles)
    pnames, _ = D.smiles_mapping(D.COMPONENTS[comp.parent], D.COMPONENTS[comp.parent].smiles)
    assert cip == {pnames[a]: lab for a, lab in Chem.FindMolChiralCenters(parent)}
    # independent check of the chirality convention: chiral volumes of an embedded conformer
    molh = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(molh, randomSeed=11) == 0
    pos = molh.GetConformer().GetPositions()
    byname = {name: tuple(pos[i]) for i, name in names.items()}
    for centre, order, sign in comp.chirality():
        vol = D._volume(byname[centre], *(byname[n] for n in order[:3]))
        assert (vol > 0) == (sign > 0), (code, centre, vol)
    # the L InChI is the mirror image of the CCD descriptor's (.../m1/s1)
    if code == "0A":
        assert Chem.MolToInchi(Chem.MolFromSmiles(CCD_0A_DESCRIPTOR)).endswith("/m1/s1")
        neutral = Chem.MolFromSmiles(comp.smiles.replace("[O-]P(=O)([O-])", "OP(=O)(O)"))
        assert Chem.MolToInchi(neutral).endswith("/m0/s1")


# --- mmCIF reading / writing --------------------------------------------------------------------------

def test_cif_parser_quotes_text_fields_pairs_and_round_trip():
    text = """# header comment
data_comp_list
_chem_comp.id   XYZ
_chem_comp.name 'name with "quotes" inside'
_chem_comp.group non-polymer
data_comp_XYZ
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
XYZ "O5'" O
XYZ 'C1' C # trailing comment
XYZ N1'x N
loop_
_x.text
;
line one
line two
;
"""
    doc = D.parse_cif(text)
    assert doc.header == ["# header comment"]
    clist = doc.block("comp_list").get("_chem_comp")
    assert clist.pairs and clist.column("name") == ['name with "quotes" inside']
    atoms = doc.block("comp_XYZ").get("_chem_comp_atom")
    assert atoms.column("atom_id") == ["O5'", "C1", "N1'x"]
    assert doc.block("comp_XYZ").get("_x").rows == [["line one\nline two"]]
    again = D.parse_cif(D.write_cif(doc))
    assert again.block("comp_XYZ").get("_chem_comp_atom").rows == atoms.rows
    assert again.block("comp_list").get("_chem_comp").rows == clist.rows
    assert again.block("comp_XYZ").get("_x").rows == [["line one\nline two"]]
    assert '"O5\'"' in D.write_cif(doc)
    with pytest.raises(D.DictionaryError, match="values for"):
        D.parse_cif("data_a\nloop_\n_a.b\n_a.c\n1 2 3\n")
    with pytest.raises(D.DictionaryError, match="unterminated"):
        D.parse_cif("data_a\n_a.b 'open\n")


# --- synthetic acedrg / eLBOW dictionaries ------------------------------------------------------------

def synthetic_dictionary(comp, style="acedrg", seed=1, coords=None, flip_chir=False, extra_h=(),
                         chir=True):
    """An acedrg- or eLBOW-like dictionary of ``comp``'s generation molecule with scrambled atom names.

    Returns (cif text, {original name: scrambled name}, {frozenset(original pair): bond value})."""
    rng = random.Random(seed)
    element = {a[0]: a[1] for a in comp.atoms}
    charge = {a[0]: a[2] for a in comp.atoms}
    hyd = [(h, x) for x, hs in comp.hydrogens for h in hs] + list(extra_h)
    element.update({h: "H" for h, _ in hyd})
    order = [a[0] for a in comp.atoms] + [h for h, _ in hyd]
    rng.shuffle(order)
    counters, new = {}, {}
    for n in order:
        e = element[n]
        counters[e] = counters.get(e, 0) + 1
        new[n] = f"{e}{counters[e]}" if style == "acedrg" else f"{e}{counters[e]:02d}"
    q = (lambda s: f"'{s}'") if style == "elbow" else (lambda s: s)
    cid = comp.code if style == "acedrg" else "LIG"
    bonds = [(a, b, o) for a, b, o in comp.bonds] + [(x, h, 1) for h, x in hyd]
    rng.shuffle(bonds)
    nbrs = {}
    for a, b, _o in bonds:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    values = {}
    L = []
    if style == "elbow":
        L += ["# electronic Ligand Builder and Optimisation Workbench (eLBOW)", "#   - a module of PHENIX", "#",
              "data_comp_list", "_chem_comp.id LIG", "_chem_comp.three_letter_code LIG",
              "_chem_comp.name 'Unknown                  '", "_chem_comp.group ligand",
              f"_chem_comp.number_atoms_all {len(order)}", "_chem_comp.desc_level .", "#", "data_comp_LIG", "#",
              "loop_", "_chem_comp_atom.comp_id", "_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol",
              "_chem_comp_atom.type_energy", "_chem_comp_atom.partial_charge", "_chem_comp_atom.x",
              "_chem_comp_atom.y", "_chem_comp_atom.z"]
    else:
        L += ["#", "data_comp_list", "loop_", "_chem_comp.id", "_chem_comp.three_letter_code", "_chem_comp.name",
              "_chem_comp.group", "_chem_comp.number_atoms_all", "_chem_comp.number_atoms_nh",
              "_chem_comp.desc_level", f"{cid} {cid} 'acedrg molecule' NON-POLYMER {len(order)} "
                                       f"{len(comp.atoms)} .", "#", f"data_comp_{cid}", "#",
              "loop_", "_chem_comp_atom.comp_id", "_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol",
              "_chem_comp_atom.type_energy", "_chem_comp_atom.charge", "_chem_comp_atom.x", "_chem_comp_atom.y",
              "_chem_comp_atom.z"]
    for n in order:
        x, y, z = coords[n] if coords else (0.0, 0.0, 0.0)
        ch = f"{charge.get(n, 0) * 0.71:.3f}" if style == "elbow" else str(charge.get(n, 0))
        L.append(f"{cid} {q(new[n])} {element[n]} {element[n]}T {ch} {x:.4f} {y:.4f} {z:.4f}")
    L += ["loop_", "_chem_comp_bond.comp_id", "_chem_comp_bond.atom_id_1", "_chem_comp_bond.atom_id_2",
          "_chem_comp_bond.type", "_chem_comp_bond.value_dist", "_chem_comp_bond.value_dist_esd"]
    for k, (a, b, o) in enumerate(bonds):
        kind = {1: "SINGLE", 2: "DOUBLE"}[o]
        kind = kind.lower() if style == "elbow" else kind
        value = f"{1.2 + 0.001 * k:.3f}"
        values[frozenset((a, b))] = value
        L.append(f"{cid} {q(new[a])} {q(new[b])} {kind} {value} 0.020")
    L += ["loop_", "_chem_comp_angle.comp_id", "_chem_comp_angle.atom_id_1", "_chem_comp_angle.atom_id_2",
          "_chem_comp_angle.atom_id_3", "_chem_comp_angle.value_angle", "_chem_comp_angle.value_angle_esd"]
    for j in sorted(nbrs):
        ns = sorted(nbrs[j])
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                L.append(f"{cid} {q(new[ns[a]])} {q(new[j])} {q(new[ns[b]])} 109.5 3.0")
    L += ["loop_", "_chem_comp_tor.comp_id", "_chem_comp_tor.id", "_chem_comp_tor.atom_id_1",
          "_chem_comp_tor.atom_id_2", "_chem_comp_tor.atom_id_3", "_chem_comp_tor.atom_id_4",
          "_chem_comp_tor.value_angle", "_chem_comp_tor.value_angle_esd", "_chem_comp_tor.period"]
    if "P" in element:
        L.append(f"{cid} alpha {q(new[chr(67) + '5' + chr(39)])} {q(new['O5' + chr(39)])} {q(new['P'])} "
                 f"{q(new['OP3'])} -60.0 10.0 3")
    L.append(f"{cid} nu0 {q(new[chr(67) + '4' + chr(39)])} {q(new['O4' + chr(39)])} "
             f"{q(new['C1' + chr(39)])} {q(new['C2' + chr(39)])} 0.0 10.0 1")
    if chir:
        L += ["loop_", "_chem_comp_chir.comp_id", "_chem_comp_chir.id", "_chem_comp_chir.atom_id_centre",
              "_chem_comp_chir.atom_id_1", "_chem_comp_chir.atom_id_2", "_chem_comp_chir.atom_id_3",
              "_chem_comp_chir.volume_sign"]
        for k, (centre, ref_order, sign) in enumerate(comp.chirality()):
            row = (ref_order[1], ref_order[0], ref_order[3]) if k % 2 else ref_order[:3]
            rest = [n for n in ref_order if n not in row]
            s = D._expected_sign(ref_order, sign, tuple(row) + tuple(rest)) * (-1 if flip_chir else 1)
            word = ("positive" if s > 0 else "negative") if style == "acedrg" else ("positiv" if s > 0 else "negativ")
            L.append(f"{cid} chir_{k} {q(new[centre])} {' '.join(q(new[n]) for n in row)} {word}")
        if "P" in element:
            L.append(f"{cid} chir_p {q(new['P'])} {q(new['O5' + chr(39)])} {q(new['OP3'])} {q(new['OP2'])} both")
    L += ["loop_", "_chem_comp_plane_atom.comp_id", "_chem_comp_plane_atom.plane_id",
          "_chem_comp_plane_atom.atom_id", "_chem_comp_plane_atom.dist_esd"]
    base = [a[0] for a in comp.atoms if "'" not in a[0] and not a[0].startswith(("P", "OP"))]
    for n in base:
        L.append(f"{cid} plan-1 {q(new[n])} 0.020")
    if style == "acedrg":
        L += ["loop_", "_pdbx_chem_comp_descriptor.comp_id", "_pdbx_chem_comp_descriptor.type",
              "_pdbx_chem_comp_descriptor.program", "_pdbx_chem_comp_descriptor.descriptor",
              f"{cid} SMILES acedrg \"{comp.smiles}\"",
              "loop_", "_pdbx_chem_comp_description_generator.comp_id",
              "_pdbx_chem_comp_description_generator.program_name",
              "_pdbx_chem_comp_description_generator.program_version",
              f"{cid} acedrg 290", "#", "data_extra_block", "_extra.value 42"]
    return "\n".join(L) + "\n", new, values


def _loop_rows(text, block, category):
    return D.parse_cif(text).block(block).get(category)


@pytest.mark.parametrize("style", ["acedrg", "elbow"])
@pytest.mark.parametrize("code", VARIANTS + ["0G"])
def test_rename_round_trip(code, style):
    comp = D.COMPONENTS[code]
    text, new, values = synthetic_dictionary(comp, style, seed=len(code) + len(style))
    res = D.finish_dictionary(text, code)
    out = D.parse_cif(res.text)
    assert [b.name for b in out.blocks][:2] == ["comp_list", f"comp_{code}"]
    clist = out.block("comp_list").get("_chem_comp")
    row = dict(zip([t.split(".")[1] for t in clist.tags], clist.rows[0]))
    assert row["id"] == row["three_letter_code"] == code and row["group"] == "RNA" and row["name"] == comp.name
    assert int(row["number_atoms_all"]) == len(comp.residue_atoms())
    block = out.block(f"comp_{code}")
    atoms = block.get("_chem_comp_atom")
    assert sorted(atoms.column("atom_id")) == sorted(comp.residue_atoms())
    assert set(atoms.column("comp_id")) == {code}
    # bonds: the reference graph with the leaving atoms removed, values carried over
    bonds = block.get("_chem_comp_bond")
    got = {frozenset((a, b)): v for a, b, v in zip(bonds.column("atom_id_1"), bonds.column("atom_id_2"),
                                                    bonds.column("value_dist"))}
    keep = set(comp.residue_atoms())
    want = {pair: v for pair, v in values.items() if pair <= keep}
    # symmetric phosphate oxygens and, without coordinates, the two H of CH2 / NH2 groups can swap
    ambiguous = {"OP1", "OP2", "OP3"} | {h for _x, hs in comp.hydrogens if len(hs) > 1 for h in hs}
    unique = {pair: v for pair, v in want.items() if not pair & ambiguous}
    assert set(got) == set(want) and all(got[p] == v for p, v in unique.items())
    # restraints with a leaving atom are gone, the others kept
    angles = block.get("_chem_comp_angle")
    n_angles = len(_loop_rows(text, block_name(text), "_chem_comp_angle").rows)
    removed = sum(1 for r in _loop_rows(text, block_name(text), "_chem_comp_angle").rows
                  if new.get("OP3") in r and comp.leaving)
    assert len(angles.rows) == n_angles - removed
    assert all("OP3" not in r for r in angles.rows) or not comp.leaving
    chir = block.get("_chem_comp_chir")
    assert {r[2] for r in chir.rows} >= {"C1'", "C2'", "C3'", "C4'"}
    assert ("P" in {r[2] for r in chir.rows}) == ("OP3" in keep)
    plane = block.get("_chem_comp_plane_atom")
    assert "C1'" not in plane.column("atom_id") and set(plane.column("atom_id")) <= keep
    if style == "acedrg":
        assert out.block("extra_block").get("_extra").rows == [["42"]]
        assert block.get("_pdbx_chem_comp_descriptor").column("descriptor") == [comp.smiles]
        assert res.program == "acedrg"
    else:
        assert res.program == "phenix.elbow" and res.source_id == "LIG"
    assert res.text.startswith(f"# {code}: restraint dictionary from {res.program}")
    # finishing again changes nothing
    assert D.finish_dictionary(res.text, code).text == res.text
    assert D.rename_dictionary(text, code) == res.text


def block_name(text):
    return [b.name for b in D.parse_cif(text).blocks if b.get("_chem_comp_atom")][0]


def test_rename_group_name_source_id_and_keep_names():
    comp = D.COMPONENTS["0U5"]
    text, _new, _v = synthetic_dictionary(comp, "acedrg")
    res = D.finish_dictionary(text, "0U5", group="non-polymer", name="my uridine")
    row = D.parse_cif(res.text).block("comp_list").get("_chem_comp").rows[0]
    assert "non-polymer" in row and "my uridine" in row
    # two components in one file: choose by --source-id or by size
    other, _n, _v = synthetic_dictionary(D.COMPONENTS["0G5"], "acedrg")
    both = text + other.replace("data_comp_list", "data_comp_list_2")
    assert "0U5" in D.finish_dictionary(both, "0U5").text
    assert D.finish_dictionary(both, "0U5", source_id="0U5").mapping
    with pytest.raises(D.DictionaryError, match="no component"):
        D.finish_dictionary(both, "0U5", source_id="ZZZ")
    # a code without reference names: ids only
    kept = D.finish_dictionary(text, "XY1")
    assert "XY1 has no reference atom names" in kept.notes[0]
    atoms = D.parse_cif(kept.text).block("comp_XY1").get("_chem_comp_atom")
    assert "C1" in atoms.column("atom_id") and set(atoms.column("comp_id")) == {"XY1"}
    with pytest.raises(D.DictionaryError, match="residue code"):
        D.finish_dictionary(text, "bad code")


def test_rename_refuses_wrong_compound_and_d_stereo():
    text, _n, _v = synthetic_dictionary(D.COMPONENTS["0A5"], "acedrg")
    with pytest.raises(D.DictionaryError, match="does not match 0G5"):
        D.finish_dictionary(text, "0G5")
    flipped, _n, _v = synthetic_dictionary(D.COMPONENTS["0A5"], "elbow", flip_chir=True)
    with pytest.raises(D.DictionaryError, match="opposite configuration"):
        D.finish_dictionary(flipped, "0A5")
    with pytest.raises(D.DictionaryError, match="restraint dictionary"):
        D.finish_dictionary("data_x\n_a.b 1\n", "0A5")
    nochir, _n, _v = synthetic_dictionary(D.COMPONENTS["0A5"], "acedrg", chir=False)
    res = D.finish_dictionary(nochir, "0A5")
    assert any("not checked" in n for n in res.notes)


def test_rename_phosphate_protonation_and_other_hydrogens():
    comp = D.COMPONENTS["0C3"]
    text, _n, _v = synthetic_dictionary(comp, "elbow", extra_h=[("HX1", "OP2"), ("HX2", "OP3")])
    res = D.finish_dictionary(text, "0C3")
    names = D.parse_cif(res.text).block("comp_0C3").get("_chem_comp_atom").column("atom_id")
    assert "OP3" not in names and "HOP3" not in names
    assert ("HOP2" in names) != ("HOP1" in names)  # the protonated non-bridging oxygen keeps its H
    assert any("carries 1 H" in n for n in res.notes)
    text, _n, _v = synthetic_dictionary(D.COMPONENTS["0G5"], "acedrg", extra_h=[("HX1", "N7")])
    with pytest.raises(D.DictionaryError, match="protonation"):
        D.finish_dictionary(text, "0G5")
    res = D.finish_dictionary(text, "0G5", allow_protonation=True)
    assert "HN7" in D.parse_cif(res.text).block("comp_0G5").get("_chem_comp_atom").column("atom_id")


def _embedded_coords(comp, seed=5):
    """Reference name -> coordinates of an RDKit conformer of ``comp``'s SMILES (hydrogens named by the
    order RDKit adds them; the test only uses them as anonymous positions)."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(comp.smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    pos = mol.GetConformer().GetPositions()
    names, _ = D.smiles_mapping(comp, comp.smiles)
    hmap = comp.h_names()
    out = {}
    for idx, name in names.items():
        out[name] = tuple(pos[idx])
        hs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetSymbol() == "H"]
        for hname, h in zip(hmap.get(name, ()), hs):
            out[hname] = tuple(pos[h])
    return out


@pytest.mark.parametrize("code", ["0A5", "0C3", "0G3"])
def test_hydrogen_names_follow_the_l_conventions(code):
    comp = D.COMPONENTS[code]
    coords = _embedded_coords(comp)
    amino = {"A": ("H61", "N6", "C6", "N1"), "C": ("H41", "N4", "C4", "N3"), "G": ("H21", "N2", "C2", "N1")}
    for mirror in (False, True):
        xyz = {n: ((-p[0] if mirror else p[0]), p[1], p[2]) for n, p in coords.items()}
        for seed in (1, 2, 3):
            text, _n, _v = synthetic_dictionary(comp, "acedrg", seed=seed, coords=xyz)
            res = D.finish_dictionary(text, code)
            atoms = D.parse_cif(res.text).block(f"comp_{code}").get("_chem_comp_atom")
            got = {n: tuple(float(v) for v in (x, y, z)) for n, x, y, z in
                   zip(atoms.column("atom_id"), atoms.column("x"), atoms.column("y"), atoms.column("z"))}
            vol = D._volume(got["C5'"], got["O5'"], got["C4'"], got["H5'"])
            vol2 = D._volume(got["C5'"], got["O5'"], got["C4'"], got["H5''"])
            # L: H5' negative, H5'' positive; mirrored (D) coordinates: named by the mirrored rule
            assert (vol < 0 < vol2) != mirror, (code, mirror, seed)
            h, n, c, ref = amino[comp.base]
            assert abs(D._torsion(got[h], got[n], got[c], got[ref])) < 90.0
            assert any("D configuration" in x for x in res.notes) == mirror
    # without chirality rows, D coordinates are refused
    xyz = {n: (-p[0], p[1], p[2]) for n, p in coords.items()}
    text, _n, _v = synthetic_dictionary(comp, "acedrg", coords=xyz, chir=False)
    with pytest.raises(D.DictionaryError, match="opposite configuration"):
        D.finish_dictionary(text, code)


def test_ccd_style_dictionary_is_left_unchanged():
    """A dictionary that already has the CCD names (e.g. the monomer-library entry) keeps them."""
    comp = D.COMPONENTS["0G"]
    text, _new, _v = synthetic_dictionary(comp, "acedrg")
    res = D.finish_dictionary(text, "0G")
    again = D.finish_dictionary(res.text, "0G3")
    assert sum(1 for o, n in again.mapping.items() if n and o != n) == 0
    assert again.mapping["OP3"] == ""


# --- finding the program's output ---------------------------------------------------------------------

def test_find_dictionary(tmp_path):
    comp = D.COMPONENTS["0A5"]
    text, _n, _v = synthetic_dictionary(comp, "acedrg")
    work = tmp_path / "work_0A5"
    (work / "0A5_TMP").mkdir(parents=True)
    (work / "0A5_TMP" / "inter.cif").write_text(text)
    (work / "notes.cif").write_text("data_x\n_a.b 1\n")
    (work / "0A5.pdb").write_text("END\n")
    (work / "0A5.cif").write_text(text)
    assert D.find_dictionary([str(work)], "0A5") == str(work / "0A5.cif")
    assert D.find_dictionary([str(work / "*.cif")]) == str(work / "0A5.cif")
    with pytest.raises(D.DictionaryError, match="no restraint dictionary"):
        D.find_dictionary([str(work)], exclude=[str(work / "0A5.cif")])
    with pytest.raises(D.DictionaryError):
        D.find_dictionary([str(tmp_path / "missing")])


# --- PubChem (mocked; no network in tests) ------------------------------------------------------------

class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_pubchem_smiles_new_and_old_property_names(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        calls.append((req.full_url, timeout))
        if "/property/SMILES/" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)
        return _Response(json.dumps({"PropertyTable": {"Properties": [
            {"CID": 1, "IsomericSMILES": L_ADENOSINE}]}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert D.fetch_pubchem_smiles(6083, timeout=3) == L_ADENOSINE
    assert calls[0][0] == f"{D.PUBCHEM_REST}/compound/cid/6083/property/SMILES/JSON"
    assert calls[1][0].endswith("/compound/cid/6083/property/IsomericSMILES/JSON") and calls[1][1] == 3
    assert calls[0][0].startswith("https://")
    calls.clear()

    def new_api(req, timeout=None):
        calls.append(req.full_url)
        return _Response(json.dumps({"PropertyTable": {"Properties": [{"CID": 2, "SMILES": "CCO"}]}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", new_api)
    assert D.fetch_pubchem_smiles("L-adenosine") == "CCO"
    assert calls == [f"{D.PUBCHEM_REST}/compound/name/L-adenosine/property/SMILES/JSON"]


@pytest.mark.parametrize("exc, message", [
    (urllib.error.HTTPError("u", 404, "Not Found", {}, None), "has no compound"),
    (urllib.error.HTTPError("u", 503, "Busy", {}, None), "HTTP 503"),
    (urllib.error.URLError("Tunnel connection failed"), "could not reach PubChem"),
    (TimeoutError(), "did not answer"),
])
def test_fetch_pubchem_errors(monkeypatch, exc, message):
    def fake(req, timeout=None):
        raise exc

    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(D.DictionaryError, match=message):
        D.fetch_pubchem_smiles("123")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Response(b"<html>"))
    with pytest.raises(D.DictionaryError, match="unreadable"):
        D.fetch_pubchem_smiles("123")
    with pytest.raises(D.DictionaryError, match="no PubChem"):
        D.fetch_pubchem_smiles(" ")


# --- command line -------------------------------------------------------------------------------------

def test_cli_list_smiles_finish_and_pubchem(tmp_path, capsys, monkeypatch):
    from mdatk.cli import main

    assert main(["dict", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["code"] for r in rows] == list(D.COMPONENTS) and rows[1]["leaving"] == ["OP3"]
    assert main(["dict", "smiles", "0c5"]) == 0
    assert capsys.readouterr().out.strip() == D.COMPONENTS["0C5"].smiles
    text, _n, _v = synthetic_dictionary(D.COMPONENTS["0C5"], "elbow")
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "0C5.cif").write_text(text)
    out = tmp_path / "0C5_final.cif"
    assert main(["dict", "finish", str(tmp_path / "w"), "--code", "0C5", "-o", str(out)]) == 0
    assert "comp_0C5" in out.read_text() and "phenix.elbow" in capsys.readouterr().err
    assert main(["dict", "finish", str(tmp_path / "w"), "--code", "0G5", "-o", str(out)]) == 1
    assert "does not match 0G5" in capsys.readouterr().err
    monkeypatch.setattr(D, "fetch_pubchem_smiles", lambda ident, timeout=15.0: L_ADENOSINE)
    assert main(["dict", "pubchem", "123"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == L_ADENOSINE and "matches 0A5" in captured.err
    monkeypatch.setattr(D, "fetch_pubchem_smiles", lambda ident, timeout=15.0: D_ADENOSINE)
    assert main(["dict", "pubchem", "60961", "--code", "0A5"]) == 0
    assert "matches no known code (0A5: the SMILES has the opposite configuration" in capsys.readouterr().err


def test_cli_finish_is_logged_in_the_project(tmp_path, capsys):
    from mdatk.cli import main
    from mdatk.config import Project
    from mdatk.projectfile import read_log, save

    project = tmp_path / "p.mdatk.toml"
    save(str(project), Project())
    text, _n, _v = synthetic_dictionary(D.COMPONENTS["0U3"], "acedrg")
    src = tmp_path / "in.cif"
    src.write_text(text)
    out = tmp_path / "0U3.cif"
    assert main(["dict", "finish", str(src), "--code", "0U3", "-o", str(out), "-p", str(project)]) == 0
    log = read_log(str(project))
    assert log[-1]["step"] == "dict finish" and str(out) in log[-1]["files"]


# --- planner step -------------------------------------------------------------------------------------

def _ctx(tmp_path, pdb=None):
    inputs = {"structure": PortValue([str(pdb)], "pdb")} if pdb else {}
    return StepContext("apt1", 4, STAMP, "dicts", inputs, str(tmp_path))


def _l_model(tmp_path):
    from test_pdbtools import make_pdb

    text = make_pdb([("A", [("G", 1, "G", False), ("A", 2, "A", False), ("C", 3, "C", False)]),
                     ("B", [("LG", 1, "G", True), ("A", 2, "A", True), ("LU", 3, "U", True)])])
    path = tmp_path / "model.pdb"
    path.write_text(text)
    return path


def test_step_codes_from_model_and_defaults(tmp_path):
    tool = get_tool("restraint-dicts")
    assert tool.category == "Refinement" and tool.outputs[0].kind == "cif" and tool.outputs[0].multiple
    s = tool.normalise({})
    assert s["program"] == "acedrg" and s["pubchem_fetch"] is False and s["group"] == "RNA"
    ctx = _ctx(tmp_path)
    assert tool.output_values(s, ctx)["dictionaries"].data["codes"] == VARIANTS
    ctx = _ctx(tmp_path, _l_model(tmp_path))
    # chain B is L (mirrored, even residue 2 named 'A'); its 5' end keeps the phosphate, so it is the
    # CCD component 0G itself and only the 3' end needs a variant
    out = tool.output_values(s, ctx)["dictionaries"]
    assert out.data["codes"] == ["0U3"]
    assert out.paths == [f"04_dicts/apt1_04_0U3_{STAMP}.cif"]
    assert not [i for i in tool.check(s, ctx) if i.level == "error"]
    # a model without L chain ends
    from test_pdbtools import make_pdb
    d = tmp_path / "d.pdb"
    d.write_text(make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)])]))
    errors = [i.message for i in tool.check(s, _ctx(tmp_path, d)) if i.level == "error"]
    assert errors and "no L-RNA chain ends" in errors[0]
    # a model produced by an earlier step (not on disk yet): all eight
    later = StepContext("apt1", 4, STAMP, "dicts", {"structure": PortValue(["03_x/m.pdb"], "pdb", source="n3")},
                        str(tmp_path))
    assert tool.output_values(s, later)["dictionaries"].data["codes"] == VARIANTS
    s2 = tool.normalise({"codes_mode": "list", "codes": "0a5, 0G3"})
    assert tool.output_values(s2, ctx)["dictionaries"].data["codes"] == ["0A5", "0G3"]
    s3 = tool.normalise({"codes_mode": "all"})
    assert len(tool.output_values(s3, ctx)["dictionaries"].paths) == 12
    assert any("already ship" in i.message for i in tool.check(s3, ctx))


def test_step_checks_custom_smiles_and_pubchem_options(tmp_path):
    tool = get_tool("restraint-dicts")
    ctx = _ctx(tmp_path)

    def issues(**kw):
        return [(i.level, i.message) for i in tool.check(tool.normalise(kw), ctx)]

    bad = issues(custom=f"0A5 = {D_ADENOSINE}")
    assert any(lvl == "error" and "opposite configuration" in m for lvl, m in bad)
    ok = issues(custom=f"0A5 = {L_ADENOSINE}")
    assert not [m for lvl, m in ok if lvl == "error"]
    new = issues(codes_mode="list", codes="0A5 XYZ", custom="XYZ = CCO")
    assert any(lvl == "warning" and "XYZ is not in MDATK's table" in m for lvl, m in new)
    missing = issues(codes_mode="list", codes="XYZ")
    assert any(lvl == "error" and "no SMILES for XYZ" in m for lvl, m in missing)
    syntax = issues(custom="0A5 OCC")
    assert any(lvl == "error" and "expected 'code = SMILES'" in m for lvl, m in syntax)
    long = issues(codes_mode="list", codes="ABCD", custom="ABCD = CCO")
    assert any("at most 3 characters" in m for _l, m in long)
    pc = issues(pubchem_fetch=True, pubchem_ids="0A5 = 12345")
    assert any(lvl == "info" and "downloaded from PubChem" in m for lvl, m in pc)
    assert any("PubChem is switched on" in m for _l, m in issues(pubchem_fetch=True))
    assert any("non-polymer" in m for _l, m in issues(group="non-polymer"))
    s = tool.normalise({"pubchem_fetch": True, "pubchem_ids": "0A5 = 1", "program": "phenix.elbow"})
    assert tool.resources(s) == ["phenix_elbow", "ccd", "pubchem"]
    assert tool.resources(tool.normalise({})) == ["acedrg", "ccd"]


def test_step_never_uses_the_network_unless_asked(tmp_path, monkeypatch):
    tool = get_tool("restraint-dicts")

    def boom(*a, **k):
        raise AssertionError("network access")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    s = tool.normalise({"pubchem_ids": "0A5 = 6083"})  # entries typed but the switch is off
    files = tool.generate(s, _ctx(tmp_path))
    assert files and tool.check(s, _ctx(tmp_path)) is not None


def test_step_pubchem_used_or_refused_when_writing(tmp_path, monkeypatch):
    tool = get_tool("restraint-dicts")
    s = tool.normalise({"pubchem_fetch": True, "pubchem_ids": "0A5 = 1\n0G5 = 2", "codes_mode": "list",
                        "codes": "0A5 0G5"})
    answers = {"1": L_ADENOSINE, "2": D_ADENOSINE}
    monkeypatch.setattr(D, "fetch_pubchem_smiles", lambda ident, timeout=15.0: answers[ident])
    files = {f.name: f for f in tool.generate(s, _ctx(tmp_path))}
    assert files[f"apt1_04_0A5_{STAMP}.smi"].content.strip() == L_ADENOSINE
    assert files[f"apt1_04_0G5_{STAMP}.smi"].content.strip() == D.COMPONENTS["0G5"].smiles
    sh = files[f"apt1_04_dicts_{STAMP}.sh"].content
    assert "SMILES from the PubChem 1" in sh and "WARNING: PubChem 2 for 0G5 not used" in sh


@pytest.mark.parametrize("program", ["acedrg", "phenix.elbow"])
def test_step_script_contents(tmp_path, program):
    tool = get_tool("restraint-dicts")
    s = tool.normalise({"program": program, "phenix_dir": "/opt/phenix-2.0", "ccp4_setup": "/opt/ccp4/setup.sh",
                        "codes_mode": "list", "codes": "0A3 XY1", "custom": "XY1 = CCO",
                        "program_args": "--extra"})
    files = {f.name: f for f in tool.generate(s, _ctx(tmp_path))}
    sh = files[f"apt1_04_dicts_{STAMP}.sh"]
    assert sh.role == "run"
    helper = files[f"apt1_04_dictionaries_{STAMP}.py"]
    assert helper.role == "helper" and "def finish_dictionary" in helper.content
    text = sh.content
    if program == "acedrg":
        assert f"acedrg -i ../apt1_04_0A3_{STAMP}.smi -r 0A3 -o 0A3 --extra" in text
        assert "source /opt/ccp4/setup.sh" in text and "phenix" not in text
    else:
        assert f"phenix.elbow ../apt1_04_0A3_{STAMP}.smi --id=0A3 --output=0A3 --extra" in text
        assert "source /opt/phenix-2.0/phenix_env.sh" in text
    assert (f"python3 apt1_04_dictionaries_{STAMP}.py finish work_0A3 --code 0A3 --group RNA "
            f"-o apt1_04_0A3_{STAMP}.cif") in text
    assert f"finish work_XY1 --code XY1 --group RNA -o apt1_04_XY1_{STAMP}.cif --keep-names" in text
    assert "leaving atom OP3" in text
    path = tmp_path / "s.sh"
    path.write_text(text)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_generated_step_runs_end_to_end_with_a_fake_acedrg(tmp_path):
    """Plan -> written files -> bash script with a stand-in 'acedrg' -> renamed dictionaries."""
    from mdatk.config import WorkflowParams
    from mdatk.workflows.plan import Plan, add_node

    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    add_node(wf, "restraint-dicts", settings={"codes_mode": "list", "codes": "0A5 0C3 0G3 0U5"})
    plan = Plan(wf, str(tmp_path / "plan"))
    assert not [i for _n, i in plan.all_issues() if i.level == "error"]
    report = plan.write()
    script = [p for p in report.written if p.endswith("_dicts_" + STAMP + ".sh")][0]
    fake = tmp_path / "bin"
    store = tmp_path / "store"
    fake.mkdir()
    store.mkdir()
    for k, code in enumerate(["0A5", "0C3", "0G3", "0U5"]):
        text, _n, _v = synthetic_dictionary(D.COMPONENTS[code], "acedrg", seed=k)
        (store / f"{code}.cif").write_text(text)
    acedrg = fake / "acedrg"
    acedrg.write_text("#!/bin/bash\n"
                      "while [ $# -gt 0 ]; do case $1 in -r) code=$2; shift;; -o) out=$2; shift;; "
                      "-i) smi=$2; shift;; esac; shift; done\n"
                      "[ -s \"$smi\" ] || exit 3\n"
                      f"mkdir -p \"${{out}}_TMP\" && cp {store}/$code.cif \"${{out}}_TMP/x.cif\"\n"
                      f"cp {store}/$code.cif \"$out.cif\" && echo END > \"$out.pdb\"\n")
    acedrg.chmod(acedrg.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{fake}{os.pathsep}{os.environ['PATH']}")
    proc = subprocess.run(["bash", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    folder = os.path.dirname(script)
    for code in ("0A5", "0C3", "0G3", "0U5"):
        out = D.parse_cif(open(os.path.join(folder, f"apt1_01_{code}_{STAMP}.cif")).read())
        names = out.block(f"comp_{code}").get("_chem_comp_atom").column("atom_id")
        assert sorted(names) == sorted(D.COMPONENTS[code].residue_atoms())
    # the copied helper runs on its own
    helper = os.path.join(folder, f"apt1_01_dictionaries_{STAMP}.py")
    proc = subprocess.run([sys.executable, "-I", helper, "smiles", "0U5"], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == D.COMPONENTS["0U5"].smiles


def test_global_section_lower_case_code_and_wizard_preview(tmp_path, monkeypatch):
    comp = D.COMPONENTS["0A3"]
    text, _new, _values = synthetic_dictionary(comp, "acedrg", seed=3)
    prefixed = "global_\n_lib_name ?\n_lib_version ?\n_lib_update ?\n" + text
    res = D.finish_dictionary(prefixed, "0a3")  # monomer-library header, lower-case code
    assert block_name(res.text) == "comp_0A3" and "0a3" not in res.text
    # the wizard preview never asks PubChem; writing asks once and warns about a neutral phosphate
    calls = []

    def fetch(ident, timeout=15.0):
        calls.append(ident)
        return D.COMPONENTS["0A3"].smiles.replace("[O-]", "O")
    monkeypatch.setattr(D, "fetch_pubchem_smiles", fetch)
    tool = get_tool("restraint-dicts")
    s = tool.normalise({"pubchem_fetch": True, "pubchem_ids": "0A3 = 6083", "codes_mode": "list", "codes": "0A3"})
    ctx = _ctx(tmp_path)
    ctx.preview = True
    sh = {f.name: f for f in tool.generate(s, ctx)}[f"apt1_04_dicts_{STAMP}.sh"].content
    assert calls == [] and "fetched when the files are written" in sh
    files = {f.name: f for f in tool.generate(s, _ctx(tmp_path))}
    tool.generate(s, _ctx(tmp_path))
    assert calls == ["6083"]  # cached for the session
    assert "WARNING: PubChem 6083 for 0A3" in files[f"apt1_04_dicts_{STAMP}.sh"].content
