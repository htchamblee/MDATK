"""The helix-restraint helper (copied into every Amber helix-restraints step) restrains DNA and L-DNA pairs:
DNA residue names (DT, DG5 ...) and L-DNA names (LDA, LDG3 ...) give their base letter, and T matches the U the
secondary structure reads.  RNA names give the letters they always did."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "helix_dist", os.path.join(os.path.dirname(HERE), "mdatk", "workflows", "scripts", "helix_dist.py"))
helix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helix)

BONDS = {"GC": [("O6", "N4"), ("N1", "N3"), ("N2", "O2")], "CG": [("N4", "O6"), ("N3", "N1"), ("O2", "N2")],
         "AU": [("N1", "N3"), ("N6", "O4")], "UA": [("N3", "N1"), ("O4", "N6")]}
BASE_ATOMS = {"G": ["N1", "N2", "O6"], "C": ["N3", "N4", "O2"], "A": ["N1", "N6"], "T": ["N3", "O4"],
              "U": ["N3", "O4"]}


def topology(labels):
    names, pointers = [], []
    for lab in labels:
        pointers.append(len(names) + 1)
        names += ["P", "C1'"] + BASE_ATOMS[helix.base_letter(lab)]
    return names, list(labels), pointers


def test_base_letters_of_rna_dna_and_l_dna_names():
    rna = ["G", "G5", "G3", "RG", "LG", "GN", "LA5", "U", "LU3"]
    assert [helix.base_letter(x) for x in rna] == ["G", "G", "G", "G", "G", "G", "A", "U", "U"]
    dna = ["DG", "DG5", "DG3", "DGN", "DT", "LDA", "LDG3", "LDC5", "LDTN"]
    assert [helix.base_letter(x) for x in dna] == ["G", "G", "G", "G", "T", "A", "G", "C", "T"]
    assert helix.same_base("T", "U") and helix.same_base("U", "T") and not helix.same_base("T", "A")


def test_dna_and_l_dna_pairs_are_restrained():
    # 5'-D(CT)-L(AG)-3' paired with 5'-L(CT)-D(AG)-3' (the D|L chimera duplex): pair i : 9-i
    labels = ["DC5", "DT", "LDA", "LDG3", "LDC5", "LDT", "DA", "DG3"]
    names, labels, pointers = topology(labels)
    pairs = [(1, 8, "CG"), (2, 7, "UA"), (3, 6, "AU"), (4, 5, "GC")]
    text = helix.restraint_text(names, labels, pointers, pairs, BONDS, "r1=2.3,")
    assert text.count("&rst") == 3 + 2 + 2 + 3
    assert "# 2 N3 -- 7 N1" in text and "# 3 N6 -- 6 O4" in text


def test_a_wrong_base_is_still_refused():
    names, labels, pointers = topology(["DC5", "DG3"])
    with pytest.raises(ValueError, match="residue 2 is DG3 in the topology but the secondary structure expects A"):
        helix.restraint_text(names, labels, pointers, [(1, 2, "CA")], BONDS, "r1=2.3,")
