"""Regression tests for the reader, selection and command-line items of the external bug report on
MDATK 0.1.1 (MDATK-BUG-028, 030, 039-056, 065, 066, 072, 083-085)."""

import numpy as np
import pytest

from conftest import make_universe
from mdatk.selections import SelectionError, check_selection, select


# ---------------------------------------------------------------------------
# selections (052-056)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def masses_universe():
    """Six atoms with the masses 10, 12, 14, 14, 16, 18 and the charges -1 ... 1.5."""
    masses = [10.0, 12.0, 14.0, 14.0, 16.0, 18.0]
    atoms = [{"name": f"A{k}", "resid": k + 1, "resname": "RES", "element": "C", "mass": m,
              "charge": -1.0 + 0.5 * k} for k, m in enumerate(masses)]
    coords = np.zeros((1, len(atoms), 3))
    coords[0, :, 0] = np.arange(len(atoms)) * 3.0
    return make_universe(atoms, coords)


def _indices(u, text):
    return list(select(u, text, allow_empty=True).indices)


def test_bug052_reversed_prop_comparison_is_mirrored_not_negated(masses_universe):
    u = masses_universe
    pairs = [("prop mass > 14", "prop 14 < mass"), ("prop mass >= 14", "prop 14 <= mass"),
             ("prop mass < 14", "prop 14 > mass"), ("prop mass <= 14", "prop 14 >= mass"),
             ("prop mass == 14", "prop 14 == mass"), ("prop mass != 14", "prop 14 != mass")]
    for plain, reversed_ in pairs:
        assert _indices(u, plain) == _indices(u, reversed_), (plain, reversed_)
    assert _indices(u, "prop 14 < mass") == [4, 5] and _indices(u, "prop 14 <= mass") == [2, 3, 4, 5]
    # every spacing MDAnalysis accepts, and 'abs'
    for text in ("prop 14<mass", "prop 14 <mass", "prop 14< mass", "not (prop 14 >= mass)"):
        assert _indices(u, text) == [4, 5], text
    assert _indices(u, "prop abs 1 <= charge") == _indices(u, "prop abs charge >= 1") == [0, 4, 5]
    assert _indices(u, "prop mass>14") == _indices(u, "prop mass > 14") == [4, 5]  # unchanged


@pytest.mark.parametrize("text, operator", [("prop mass ><14", "><"), ("prop mass <> 14", "<>"),
                                            ("prop mass => 14", "=>"), ("prop mass >> 14", ">>")])
def test_bug053_malformed_prop_operator_is_an_error(masses_universe, text, operator):
    for run in (lambda: select(masses_universe, text, allow_empty=True), lambda: check_selection(text)):
        with pytest.raises(SelectionError) as info:
            run()
        message = str(info.value)
        assert f"invalid operator {operator!r} in prop (use <, >, <=, >=, ==, !=)" in message
        assert info.value.kind == "syntax" and "float" not in message


@pytest.mark.parametrize("text", ["atom SYS 1 and", "atom and 1 A0", "atom SYS 1 )", "atom SYS 1",
                                  "atom SYS x A0", "name A1 or atom SYS 1 or"])
def test_bug054_atom_does_not_take_keywords_as_arguments(masses_universe, text):
    for run in (lambda: select(masses_universe, text, allow_empty=True), lambda: check_selection(text)):
        with pytest.raises(SelectionError, match="atom: expected segid resid name"):
            run()


def test_bug054_atom_still_selects(masses_universe):
    u = masses_universe
    assert _indices(u, "atom SYS 2 A1") == [1]
    assert _indices(u, "atom SYS 2 A1 or atom SYS 4 A3") == [1, 3]
    assert _indices(u, "(atom SYS 2 A1) and name A1") == [1]
    assert _indices(u, r"atom \SYS 2 \A1") == [1]  # the escape for names that are keywords


def test_bug055_negative_distance_is_an_error_whatever_the_atom_counts(masses_universe):
    u = masses_universe
    for text in ("around -2 resid 1", "around -2 (resid 1 and name A0)", "resid 2 and around -0.5 resid 1",
                 "point 0 0 0 -3"):
        with pytest.raises(SelectionError, match="distance must not be negative"):
            select(u, text, allow_empty=True)
        with pytest.raises(SelectionError, match="distance must not be negative"):
            check_selection(text)
    assert _indices(u, "around 3.5 resid 1") == [1] and _indices(u, "around 0 resid 1") == []
    assert _indices(u, "point 0 0 0 3.5") == [0, 1]


def test_bug056_missing_arguments_name_the_keyword(masses_universe):
    u = masses_universe
    for text, expected in (("around", "around: expected a distance and a selection, e.g. 'around 3.5 resname MG'"),
                           ("around 3", "around: expected a distance and a selection"),
                           ("around x resid 1", "around: expected a distance and a selection"),
                           ("point 1 2 3", "point: expected x y z distance"),
                           ("prop mass >", "prop: expected a property, an operator and a value")):
        with pytest.raises(SelectionError) as info:
            select(u, text, allow_empty=True)
        assert expected in str(info.value) and "NoneType" not in str(info.value), text
        with pytest.raises(SelectionError):
            check_selection(text)


def test_bug056_float_range_bounds_take_any_float(masses_universe):
    u = masses_universe
    assert _indices(u, "mass 12:14") == [1, 2, 3]  # as before
    for text in ("mass 12:+14", "mass 1.2E1:1.4E1", "mass 1.2e1:1.4E1", "mass 12 to +14", "mass 12 : +14"):
        assert _indices(u, text) == [1, 2, 3], text
    assert _indices(u, "mass 16:inf") == [4, 5] and _indices(u, "mass -inf:12") == [0, 1]
    assert _indices(u, "mass -inf:inf") == [0, 1, 2, 3, 4, 5]
    assert _indices(u, "charge -1.0:-0.5") == [0, 1] and _indices(u, "charge -1--0.5") == [0, 1]
    assert _indices(u, "mass 10 12:14 18") == [0, 1, 2, 3, 5]
    for text in ("charge :1", "mass 12:14abc", "mass 12:13:14"):
        with pytest.raises(SelectionError) as info:
            select(u, text, allow_empty=True)
        assert "cannot read the range" in str(info.value) and "float" not in str(info.value), text


def test_bug056_upper_case_keywords_get_a_hint(masses_universe):
    for text, keyword in (("resid 1 AND name A0", "and"), ("name A0 AND resid 1", "and"),
                          ("resid 1 OR resid 2", "or")):
        with pytest.raises(SelectionError) as info:
            select(masses_universe, text, allow_empty=True)
        assert f"keywords are lower case: did you mean {keyword!r}?" in str(info.value), text
        with pytest.raises(SelectionError, match="keywords are lower case"):
            check_selection(text)
