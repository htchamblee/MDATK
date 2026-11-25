"""Regression tests of the analysis items of the external bug report on MDATK 0.1.1 (MDATK-BUG-011 to 027, 029,
067, 070, 071, 073, 074): each test is the report's case with numbers that can be checked by hand, fails on 0.1.1
and passes with the fix named in its docstring."""
import logging
import math

import numpy as np
import pytest

from conftest import make_universe, micro_project, run_one


def atom(name, resid, resname, element, charge=0.0, mass=12.0):
    return dict(name=name, resid=resid, resname=resname, element=element, charge=charge, mass=mass)


def warnings_of(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ----------------------------------------------------------------------------------------------- MDATK-BUG-011
def test_bug011_an_atoms_element_does_not_depend_on_the_group_it_is_evaluated_in():
    """One atom without an element in the topology (an Amber extra point) made every atom of the group take the
    first letter of its name: chlorine CL1 became carbon, bromine boron, a bound Mg 'M'."""
    import MDAnalysis as mda

    from mdatk.selections import atom_elements, heavy

    u = mda.Universe.empty(6, n_residues=2, atom_resindex=[0, 0, 0, 0, 0, 1], trajectory=True)
    u.add_TopologyAttr("names", ["C1", "CL1", "BR1", "MG", "HG", "EPW"])
    u.add_TopologyAttr("resnames", ["LIG", "WAT"])
    u.add_TopologyAttr("elements", ["C", "Cl", "Br", "Mg", "H", ""])
    ligand = u.select_atoms("resname LIG")
    assert list(atom_elements(ligand)) == ["C", "CL", "BR", "MG", "H"]
    assert list(atom_elements(u.atoms)) == ["C", "CL", "BR", "MG", "H", "E"]  # the extra point alone is guessed
    assert list(atom_elements(u.atoms[[5]])) == ["E"] and list(atom_elements(u.atoms[[]])) == []
    assert heavy(u.atoms).names.tolist() == ["C1", "CL1", "BR1", "MG", "EPW"]
    # a topology without elements is guessed from the names, as before (a lone MG is the ion)
    v = mda.Universe.empty(3, n_residues=2, atom_resindex=[0, 0, 1], trajectory=True)
    v.add_TopologyAttr("names", ["CA", "1HB", "MG"])
    v.add_TopologyAttr("resnames", ["ALA", "MG"])
    assert list(atom_elements(v.atoms)) == ["C", "H", "MG"]
