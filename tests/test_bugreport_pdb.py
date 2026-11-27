"""Regression tests for the PDB-tools, builder and AppImage items of the 2026-11-26 bug report addendum
(MDATK-BUG-088, 089, 090, 093, 096, 106)."""

import pytest

from mdatk import builder, pdbtools as pt


def _numbers(text):
    return [(r.chain, r.resseq) for r in pt.read_pdb(text).residues()]


# --- MDATK-BUG-088: residue numbers below -999 ----------------------------------------------------------

def test_bug088_residue_numbers_below_minus_999_are_refused():
    with pytest.raises(pt.PDBError, match=r"residue number -1002 \(A\) does not fit the PDB columns "
                                          r"\(-999 to 9999\)"):
        builder.build_helix("GCAU", start=(-1002, 1))
    text = builder.build_helix("GCAU", start=(-999, 1))
    assert _numbers(text) == [("A", -999), ("A", -998), ("A", -997), ("A", -996),
                              ("B", 1), ("B", 2), ("B", 3), ("B", 4)]
    with pytest.raises(pt.PDBError, match="residue number -2000 "):
        pt.clean(builder.build_helix("GCAU"), renumber=True, start=-2000)
