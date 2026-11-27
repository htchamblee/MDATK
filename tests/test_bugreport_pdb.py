"""Regression tests for the PDB-tools, builder and AppImage items of the 2026-11-26 bug report addendum
(MDATK-BUG-088, 089, 090, 093, 096, 106)."""

import pytest

from mdatk import builder, frames, pdbtools as pt

WRAP = "WARNING: residue numbers above 9999 wrap in the PDB columns"


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


# --- MDATK-BUG-089: --combine-into with a multi-character chain id -------------------------------------

def test_bug089_combine_into_needs_a_single_character():
    text = builder.build_helix("GCAU")
    with pytest.raises(pt.PDBError, match="chain ids must be single characters"):
        pt.clean(text, combine=["A", "B"], combine_into="XY")
    out = pt.clean(text, combine=["A", "B"], combine_into="X")
    assert {c for c, _n in _numbers(out.text)} == {"X"}
    assert any(n.startswith("Combined chains A, B into chain X (8 residues)") for n in out.notes)


# --- MDATK-BUG-090: --keep-conect with repeated input serials ------------------------------------------

def test_bug090_conect_note_says_when_the_records_are_dropped():
    atoms = [line for line in builder.build_helix("GCAU").splitlines() if line.startswith("ATOM")]
    copy = [line[:21] + "C" + line[22:] for line in atoms]  # the same serials again, as chain C
    dup = "\n".join(atoms + ["TER"] + copy + ["CONECT    1    2", "END"]) + "\n"
    out = pt.clean(dup, strip_conect=False)
    assert "CONECT" not in out.text
    assert ("WARNING: dropped 1 CONECT records: the input's atom serials repeat, so they cannot be remapped"
            in out.notes)
    assert not any(n.startswith("Kept CONECT") for n in out.notes)
    # serials that do not repeat: kept and remapped as before
    single = "\n".join(["REMARK   1 X"] + atoms + ["CONECT    1    2", "END"]) + "\n"
    out = pt.clean(single, strip_conect=False)
    assert "Kept CONECT records (remapped to the new atom serials)" in out.notes
    assert [line for line in out.text.splitlines() if line.startswith("CONECT")] == ["CONECT    1    2"]
    assert any(n.startswith("Stripped 1 CONECT") for n in pt.clean(dup).notes)


# --- MDATK-BUG-093: build helix wraps numbers above 9999 -----------------------------------------------

def test_bug093_build_helix_warns_when_numbers_wrap():
    notes = []
    text = builder.build_helix("GCAU", start=(1, 9998), notes=notes)
    assert WRAP in notes
    assert [n for c, n in _numbers(text) if c == "B"] == [9998, 9999, 0, 1]  # the wrap itself is kept
    notes = []
    builder.build_helix("GCAU", start=(1, 9996), notes=notes)  # 9996 ... 9999: no wrap, no warning
    assert WRAP not in notes


# --- MDATK-BUG-096: reversed residue ranges ------------------------------------------------------------

def test_bug096_reversed_ranges_are_an_error():
    with pytest.raises(pt.PDBError, match="residue range A:4-1 ends before it starts"):
        pt.parse_ranges("A:4-1")
    with pytest.raises(pt.PDBError, match="residue range A:9-2 ends before it starts"):
        pt.parse_ranges([("A", 9, 2)])
    assert pt.parse_ranges("A:1-4") == [("A", 1, 4)]
    assert pt.parse_ranges("A:-4--1") == [("A", -4, -1)]
    with pytest.raises(pt.PDBError, match="ends before it starts"):
        pt.clean(builder.build_helix("GCAU"), l_ranges="A:4-1")
    # a model map pairs a reversed side in reverse order, as before
    assert frames.parse_model_map("C:3-1=X:45-47") == {("X", 45): ("C", 3), ("X", 46): ("C", 2),
                                                        ("X", 47): ("C", 1)}
