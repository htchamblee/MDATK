"""Tests for the dot-bracket utilities of mdatk.secstruct."""

import pytest

from mdatk.secstruct import dotbracket_to_pairs, pairs_to_dotbracket, unique_pairs


def test_parse_simple_hairpin():
    pairs, n, breaks = dotbracket_to_pairs("((..))")
    assert pairs == [(0, 5), (1, 4)] and n == 6 and breaks == []


def test_parse_pseudoknot_levels_and_letters():
    pairs, n, _ = dotbracket_to_pairs("((..[[..))..]]{<Aa>}")
    assert n == 20
    assert set(pairs) == {(0, 9), (1, 8), (4, 13), (5, 12), (14, 19), (15, 18), (16, 17)}


@pytest.mark.parametrize("text", ["(((&)))", "((( )))", "(((  &  )))", "&(((&)))&"])
def test_chain_separators(text):
    pairs, n, breaks = dotbracket_to_pairs(text)
    assert n == 6 and breaks == [3] and pairs == [(0, 5), (1, 4), (2, 3)]


def test_unbalanced_messages():
    with pytest.raises(ValueError, match=r"'\)' at position 3 .* no matching '\('"):
        dotbracket_to_pairs("..)")
    with pytest.raises(ValueError, match=r"'\[' at position 2 is never closed"):
        dotbracket_to_pairs("([.)")  # ')' closes '(', '[' stays open
    with pytest.raises(ValueError, match="never closed .and 1 more"):
        dotbracket_to_pairs("((.")
    with pytest.raises(ValueError, match="invalid dot-bracket character '#'"):
        dotbracket_to_pairs("(#)")


@pytest.mark.parametrize("text", ["((((....))))", "((..[[..))..]]", "(((&)))..((..))", "(([[{{<<..))]]}}>>",
                                  ".....", "(&)"])
def test_round_trip(text):
    pairs, n, breaks = dotbracket_to_pairs(text)
    assert pairs_to_dotbracket(pairs, n, breaks) == text


@pytest.mark.parametrize("text, written", [("((..[[..))..{{..]]..}}", "((..[[..))..((..]]..))"),
                                           ("AB..((..ab..))", "[{..((..]}..))")])
def test_round_trip_normalises_levels(text, written):
    # the lowest free level is used: same pairs, possibly other brackets
    pairs, n, breaks = dotbracket_to_pairs(text)
    assert pairs_to_dotbracket(pairs, n, breaks) == written
    assert dotbracket_to_pairs(written) == (pairs, n, breaks)


def test_longest_helix_gets_round_brackets():
    # a 2-pair helix opens first, the crossing 4-pair helix is the main one
    pairs = [(0, 7), (1, 6), (3, 13), (4, 12), (5, 11), (2, 14)]
    assert pairs_to_dotbracket(pairs, 15) == "[[((((]]...))))"


def test_pairs_to_dotbracket_validation():
    assert pairs_to_dotbracket([(5, 0)], 6) == "(....)"  # orientation normalised
    assert pairs_to_dotbracket([(0, 5), (0, 5)], 6) == "(....)"  # duplicates ignored
    with pytest.raises(ValueError, match="more than one base pair"):
        pairs_to_dotbracket([(0, 5), (0, 4)], 6)
    with pytest.raises(ValueError, match="invalid base pair"):
        pairs_to_dotbracket([(0, 6)], 6)
    assert pairs_to_dotbracket([], 4, chain_breaks=[0, 2, 4]) == "..&.."


def test_unique_pairs():
    keep, dropped = unique_pairs([(0, 5), (1, 5), (2, 4)])
    assert keep == [(0, 5), (2, 4)] and dropped == [(1, 5)]
