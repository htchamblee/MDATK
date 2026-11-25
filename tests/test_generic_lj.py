"""The generic Lennard-Jones table of the energy analysis (mdatk.analysis.generic_lj): its values are those of
the AmberTools parameter files it names (the lines are quoted below as they stand in those files), docs/GENERIC_LJ.md
shows the same table, hydrogens take the type of the atom they are bonded to, monatomic ions their own rows, and a
hydrogen bond is no longer repulsive."""
import os
import re

import numpy as np
import pytest

from conftest import make_universe, micro_project, run_one
from mdatk.analysis import generic_lj as G

# the lines of the files' Lennard-Jones sections (parm10.dat: MOD4 RE; the frcmod files: NONBON), copied whole:
# atom type, R* = Rmin/2 (Å), epsilon (kcal/mol), comment. docs/GENERIC_LJ.md says where the files were read
SOURCE_LINES = {
    "parm10.dat": """\
  C           1.9080  0.0860             OPLS
  N           1.8240  0.1700             OPLS
  O           1.6612  0.2100             OPLS
  P           2.1000  0.2000             JCC,7,(1986),230;
  S           2.0000  0.2500             W. Cornell CH3SH and CH3SCH3 FEP's
  F           1.75    0.061              Gough et al. JCC 13,(1992),963.
  Cl          1.948   0.265              Fox, JPCB,102,8070,(98),flex.mdl CHCl3
  Br          2.22    0.320              Junmei(?)
  I           2.35    0.40               JCC,7,(1986),230;
  H           0.6000  0.0157            !Ferguson base pair geom.
  HO          0.0000  0.0000             OPLS Jorgensen, JACS,110,(1988),1657
  HS          0.6000  0.0157             W. Cornell CH3SH --> CH3OH FEP
  H1          1.3870  0.0157             Veenstra et al JCC,8,(1992),963
""",
    "frcmod.ionsjc_tip3p": """\
  Li+      1.025     0.0279896
  Na+      1.369     0.0874393
  K+       1.705     0.1936829
  Rb+      1.813     0.3278219
  Cs+      1.976     0.4065394
  F-       2.303     0.0033640
  Cl-      2.513     0.0355910
  Br-      2.608     0.0586554
  I-       2.860     0.0536816
""",
    "frcmod.ions234lm_126_tip3p": """\
Mg2+  1.360  0.01020237    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Ca2+  1.649  0.10592870    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Sr2+  1.810  0.22132374    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Ba2+  2.019  0.40664608    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Mn2+  1.407  0.01686710    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Fe2+  1.353  0.00941798    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Co2+  1.299  0.00483892    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Ni2+  1.255  0.00262320    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Cu2+  1.218  0.00148497    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Zn2+  1.271  0.00330286    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
Cd2+  1.412  0.01773416    CM set for TIP3P water from Li et al., JCTC, 2013, 9, 2733
""",
}
DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "GENERIC_LJ.md")


def source_values():
    out = {}
    for name, text in SOURCE_LINES.items():
        for line in text.splitlines():
            parts = line.split()
            out[name, parts[0]] = (float(parts[1]), float(parts[2]))
    return out


def test_every_row_is_the_line_of_its_source_file():
    values = source_values()
    rows = G.table_rows()
    assert len(rows) == len(G.COVALENT) + len(G.HYDROGEN) + 1 + len(G.IONS) == 33
    for _cls, key, lj in rows:
        assert (lj.rstar, lj.epsilon) == values[lj.source, lj.amber_type], (key, lj)
    assert len({(lj.source, lj.amber_type) for _c, _k, lj in rows}) == len(values)  # every quoted line is used
    # the ions are those of their element, the molecule rows of theirs
    assert all(lj.amber_type.rstrip("+-0123456789").upper() == el for el, lj in G.IONS.items())
    assert all(lj.amber_type.upper() == el for el, lj in G.COVALENT.items())
    assert G.DEFAULT is G.COVALENT["C"] and G.HYDROGEN_ALONE is G.HYDROGEN["N"]


def test_the_document_shows_the_table_of_the_code():
    with open(DOC, encoding="utf-8") as fh:
        text = fh.read()
    found = {}
    for line in text.splitlines():
        m = re.match(r"^\| ([^|]+) \| ([-0-9.e]+) \| ([-0-9.e]+) \| `([^`]+)` \| `([^`]+)` \|", line)
        if m:
            found[m.group(1)] = (float(m.group(2)), float(m.group(3)), m.group(4), m.group(5))
    expected = {el: tuple(lj) for el, lj in G.COVALENT.items()}
    expected.update({f"H on {el}": tuple(lj) for el, lj in G.HYDROGEN.items()})
    expected["H on C / other"] = tuple(G.HYDROGEN_OTHER)
    ions = {el: tuple(lj) for el, lj in G.IONS.items()}
    # the halogens are in two tables (a molecule's atom, an ion): the document's rows in order
    rows = [(m.group(1), (float(m.group(2)), float(m.group(3)), m.group(4), m.group(5))) for m in (
        re.match(r"^\| ([^|]+) \| ([-0-9.e]+) \| ([-0-9.e]+) \| `([^`]+)` \| `([^`]+)` \|", ln)
        for ln in text.splitlines()) if m]
    assert rows == list(expected.items()) + list(ions.items())
    assert found  # (the last row of a key; the ordered comparison above is the check)
    for key in G.CITATION_KEYS:
        from mdatk import citations
        ref = citations.references([key])[0]
        assert ref.doi and ref.doi in text, key
    assert f"within {G.BOND_SEARCH:g} Å" in text


def test_lookup_rules():
    assert G.lookup("C").amber_type == "C" and G.lookup("O", alone=True).amber_type == "O"  # a lone water oxygen
    assert G.lookup("CL", alone=True).amber_type == "Cl-" and G.lookup("CL").amber_type == "Cl"
    assert G.lookup("NA", alone=True).amber_type == "Na+" and G.lookup("NA").amber_type == "Na+"
    assert G.lookup("ZN").amber_type == "Zn2+"  # a metal bound in a larger residue
    assert [G.lookup("H", bonded_to=e).amber_type for e in ("N", "O", "S", "C", "P", "")] == \
        ["H", "HO", "HS", "H1", "H1", "H"]
    assert G.lookup("SE") is None and G.lookup("") is None


def atom(name, resid, resname, element, charge=0.0):
    return dict(name=name, resid=resid, resname=resname, element=element, charge=charge)


ATOMS = [atom("N3", 1, "U", "N"), atom("H3", 1, "U", "H"), atom("C5", 1, "U", "C"), atom("H5", 1, "U", "H"),
         atom("O2'", 1, "U", "O"), atom("HO2'", 1, "U", "H"), atom("SG", 2, "CYS", "S"), atom("HG", 2, "CYS", "H"),
         atom("HX", 2, "CYS", "H"), atom("NA", 3, "NA", "NA"), atom("CL", 4, "CL", "CL"), atom("CL1", 5, "LIG", "CL"),
         atom("C1", 5, "LIG", "C"), atom("SE", 5, "LIG", "SE")]
XYZ = np.array([[0, 0, 0], [1.01, 0, 0], [0, 5, 0], [1.09, 5, 0], [0, 10, 0], [0.96, 10, 0], [0, 15, 0],
                [1.34, 15, 0], [9, 15, 0], [20, 0, 0], [25, 0, 0], [30, 0, 0], [31.7, 0, 0], [33, 0, 0]], dtype=float)


@pytest.mark.parametrize("bonds", [(), [(0, 1), (2, 3), (4, 5), (6, 7)], [(2, 3)], [(0, 1), (2, 3), (4, 5), (6, 7), (1, 3)]])
def test_parameters_of_a_small_system(bonds):
    """With the topology's bonds, without any (the nearest heavy atom of the residue), with bonds for one hydrogen
    only (a PDB with CONECT records for a ligand), and with a hydrogen-hydrogen bond (some water topologies)."""
    u = make_universe(ATOMS, [XYZ], bonds=bonds)
    rstar, eps, missing = G.parameters(u, u.atoms)
    types = ["N", "H", "C", "H1", "O", "HO", "S", "HS", "H", "Na+", "Cl-", "Cl", "C", None]
    table = {lj.amber_type: lj for _c, _k, lj in G.table_rows()}
    expected = [table[t] if t else G.DEFAULT for t in types]
    np.testing.assert_array_equal(rstar, [lj.rstar for lj in expected])
    np.testing.assert_array_equal(eps, [lj.epsilon for lj in expected])
    assert missing == ["SE"]
    part = u.atoms[[1, 5, 9]]  # a selection without the hydrogens' parents: they are found all the same
    np.testing.assert_array_equal(G.parameters(u, part)[0], [0.6, 0.0, 1.369])


def test_a_hydrogen_bond_is_not_made_repulsive_by_its_hydrogen():
    """N3-H3 ... O6, H...O 1.9 Å: with one large radius for every hydrogen (R* 1.1 Å, up to 0.1.0) the pair H3-O6
    alone was +4.0 kcal/mol; with Amber's type H for a hydrogen on nitrogen it is +0.13."""
    atoms = [atom("N3", 1, "U", "N"), atom("H3", 1, "U", "H"), atom("O6", 2, "G", "O")]
    u = make_universe(atoms, [[[0, 0, 0], [1.01, 0, 0], [2.91, 0, 0]]])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    res = run_one("energy", u, p)

    def lj(a, b, r):
        rmin, e = a.rstar + b.rstar, np.sqrt(a.epsilon * b.epsilon)
        return e * ((rmin / r) ** 12 - 2 * (rmin / r) ** 6)

    n, h, o = G.COVALENT["N"], G.HYDROGEN["N"], G.COVALENT["O"]
    assert res.arrays["vdw"][0] == pytest.approx(lj(n, o, 2.91) + lj(h, o, 1.9), rel=1e-5)  # (float32 positions)
    assert lj(h, o, 1.9) == pytest.approx(0.135, abs=0.005)
    assert lj(G.LJ(1.1, 0.0157, "HP", ""), o, 1.9) == pytest.approx(4.0, abs=0.05)  # the table up to 0.1.0
    assert res.meta["lj_source"] == "generic" and res.meta["lj_table"] == G.TABLE_NAME
    assert res.meta["lj_default_elements"] == []


def test_an_element_without_parameters_is_reported(caplog):
    atoms = [atom("SE", 1, "MSE", "SE"), atom("CE", 1, "MSE", "C"), atom("O", 2, "HOH", "O")]
    u = make_universe(atoms, [[[0, 0, 0], [1.9, 0, 0], [5, 0, 0]]])
    p = micro_project({"Aptamer": "1", "Target": "2"})
    with caplog.at_level("WARNING"):
        res = run_one("energy", u, p)
    assert res.meta["lj_default_elements"] == ["SE"]
    assert any("no generic Lennard-Jones parameters for element(s) SE: carbon's are used" in r.getMessage()
               for r in caplog.records)


def test_the_table_is_cited_when_it_is_used(tmp_path):
    from mdatk import citations as C
    from mdatk.config import Project

    p = Project()
    p.analyses.energy.enabled = True
    p.inputs.topology = "system.pdb"
    assert "generic_lj" in C.pipeline_resources(p)
    assert set(C.keys_for(["generic_lj"])) == set(G.CITATION_KEYS)
    p.inputs.topology = "system.parm7"
    assert "generic_lj" not in C.pipeline_resources(p)
    p.analyses.energy.use_parm7 = False
    assert "generic_lj" in C.pipeline_resources(p)
    p.analyses.energy.enabled = False
    assert "generic_lj" not in C.pipeline_resources(p)
