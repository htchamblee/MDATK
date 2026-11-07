"""Tests for mdatk.pdbtools (PDB hygiene: L-RNA detection, name schemes, ChimeraX repair, chains ...)."""

import hashlib
import math

import pytest

from mdatk import builder, modelcheck as mc, pdbtools as pt

# Nucleotide geometry: the A/C/G/U units of AmberTools' RNA.lib (heavy atoms plus H5'' and HO2', rounded
# to PDB precision; the sugar-phosphate backbone is identical in the four units).  L residues are made by
# mirroring x -> -x.
BACKBONE = [
    ("P", 2.314, 1.777, -0.478), ("OP1", 1.933, 3.073, -1.082), ("OP2", 3.157, 0.872, -1.292),
    ("O5'", 2.994, 2.142, 0.923), ("C5'", 2.156, 2.552, 2.020), ("H5''", 1.605, 3.450, 1.741),
    ("C4'", 3.008, 2.848, 3.243), ("O4'", 3.228, 1.605, 3.977), ("C1'", 4.559, 1.151, 3.778),
    ("C3'", 4.416, 3.378, 2.961), ("C2'", 5.230, 2.096, 2.779), ("O2'", 6.273, 2.050, 3.757),
    ("HO2'", 6.759, 1.235, 3.611), ("O3'", 5.030, 4.129, 3.998),
]
BASES = {
    "A": [("N9", 4.539, -0.297, 3.316), ("C8", 4.337, -0.846, 2.077), ("N7", 4.453, -2.141, 2.049),
          ("C5", 4.754, -2.480, 3.363), ("C6", 4.998, -3.708, 3.989), ("N6", 4.977, -4.883, 3.346),
          ("N1", 5.265, -3.689, 5.302), ("C2", 5.285, -2.520, 5.935), ("N3", 5.073, -1.309, 5.456),
          ("C4", 4.806, -1.356, 4.133)],
    "C": [("N1", 4.501, -0.268, 3.327), ("C6", 4.258, -0.582, 2.026), ("C5", 4.206, -1.878, 1.616),
          ("C4", 4.416, -2.881, 2.614), ("N4", 4.379, -4.159, 2.286), ("N3", 4.653, -2.564, 3.884),
          ("C2", 4.701, -1.262, 4.273), ("O2", 4.919, -0.928, 5.447)],
    "G": [("N9", 4.500, -0.268, 3.327), ("C8", 4.268, -0.773, 2.063), ("N7", 4.279, -2.081, 2.002),
          ("C5", 4.535, -2.475, 3.310), ("C6", 4.665, -3.772, 3.873), ("O6", 4.578, -4.865, 3.316),
          ("N1", 4.926, -3.709, 5.247), ("C2", 5.048, -2.547, 5.982), ("N2", 5.299, -2.702, 7.289),
          ("N3", 4.928, -1.333, 5.453), ("C4", 4.674, -1.377, 4.118)],
    "U": [("N1", 4.539, -0.307, 3.313), ("C6", 4.317, -0.673, 2.011), ("C5", 4.360, -1.958, 1.634),
          ("C4", 4.643, -2.997, 2.590), ("O4", 4.703, -4.199, 2.337), ("N3", 4.853, -2.538, 3.874),
          ("C2", 4.815, -1.224, 4.292), ("O2", 5.015, -0.915, 5.455)],
}


def residue_atoms(base, mirror=False, dz=0.0):
    sign = -1.0 if mirror else 1.0
    return [(n, sign * x, y, z + dz) for n, x, y, z in BACKBONE[:9] + BASES[base] + BACKBONE[9:]]


def atom_line(serial, name, resname, chain, resseq, xyz, record="ATOM", element=None):
    element = name.lstrip("0123456789")[0] if element is None else element
    field = name.ljust(4) if len(name) == 4 else " " + name.ljust(3)
    x, y, z = xyz
    return (f"{record:<6}{serial:>5} {field} {resname:>3} {chain:1}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}")


def make_pdb(segments, records=None, extra=(), header=(), first_serial=1):
    """segments: [(chain, [(resname, resseq, base, mirrored), ...]), ...] -> PDB text (TER per segment)."""
    lines, serial, k = list(header), first_serial - 1, 0
    for chain, residues in segments:
        for resname, resseq, base, mirror in residues:
            for name, x, y, z in residue_atoms(base, mirror, dz=7.0 * k):
                serial += 1
                lines.append(atom_line(serial, name, resname, chain, resseq, (x, y, z),
                                       (records or {}).get(resname, "ATOM")))
            k += 1
        serial += 1
        lines.append(f"TER   {serial:>5}      {resname:>3} {chain:1}{resseq:>4}")
    lines += list(extra)
    lines.append("END")
    return "\n".join(lines) + "\n"


def names(structure):
    return [(r.chain, r.resseq, r.name) for r in structure.residues()]


def coords(base, mirror=False):
    return {n: (x, y, z) for n, x, y, z in residue_atoms(base, mirror)}


# --- chirality ------------------------------------------------------------------------------------------

def test_chiral_volume_convention():
    assert pt.chiral_volume((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) == pytest.approx(1.0)
    assert pt.chiral_volume((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)) == pytest.approx(-1.0)


@pytest.mark.parametrize("base", "ACGU")
def test_sugar_chirality_d_and_mirrored(base):
    call, vols = pt.sugar_chirality(coords(base), base)
    assert call == "D" and set(vols) == {"C4'", "C3'", "C1'"}
    # D signs are the opposite of the L signs of the 0A/0C/0G/0U dictionaries
    assert vols["C4'"] < 0 and vols["C3'"] > 0 and vols["C1'"] < 0
    call_l, vols_l = pt.sugar_chirality(coords(base, mirror=True), base)
    assert call_l == "L"
    assert all(vols_l[k] == pytest.approx(-vols[k]) for k in vols)


def test_sugar_chirality_missing_atoms_and_tie():
    partial = {k: v for k, v in coords("G").items() if k not in ("C5'", "O3'")}
    call, vols = pt.sugar_chirality(partial, "G")
    assert call == "D" and list(vols) == ["C1'"]  # one usable centre still decides
    no_n9 = {k: v for k, v in coords("G", True).items() if k not in ("N9",)}
    assert pt.sugar_chirality(no_n9, "G")[0] == "L"  # purine: N1 is not used in place of N9
    assert set(pt.sugar_chirality(no_n9, "G")[1]) == {"C4'", "C3'"}
    assert pt.sugar_chirality({"P": (0, 0, 0)}, "A") == (None, {})
    tie = {"C4'": (0, 0, 0), "O4'": (1, 0, 0), "C3'": (0, 1, 0), "C5'": (0, 0, 1),  # C4' votes L
           "C1'": (2, 0, 0), "N9": (2, 1, 0), "C2'": (2, 0, 1)}  # C1' votes D
    call, vols = pt.sugar_chirality(tie, "A")
    assert call is None and vols["C4'"] > 0 and vols["C1'"] < 0


def test_assign_chirality_detection_override_and_fallback():
    text = make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)]),
                     ("B", [("G", 1, "G", True), ("U", 2, "U", True)]),
                     ("C", [("0A", 1, "A", False)])])
    s = pt.read_pdb(text)
    notes = pt.assign_chirality(s)
    assert [r.chirality for r in s.residues()] == ["D", "D", "L", "L", "D"]
    assert any("L sugars in 2 residues with D names" in n and "B:1-2" in n for n in notes)
    assert any(n.startswith("WARNING: L-named residues with D sugar") and "C:1 0A" in n for n in notes)

    s = pt.read_pdb(text)
    notes = pt.assign_chirality(s, l_ranges="A:2, C")
    assert [r.chirality for r in s.residues()] == ["D", "L", "L", "L", "L"]
    assert any("forced L although the sugar geometry is D" in n and "A:2 C" in n for n in notes)

    s = pt.read_pdb(text)
    pt.assign_chirality(s, detect=False)
    assert [r.chirality for r in s.residues()] == ["D", "D", "D", "D", "L"]  # names only

    keep = ("P", "OP1", "OP2", "O5'", "C5'")  # no sugar ring: undetermined -> residue name
    lines = [ln for ln in make_pdb([("A", [("LG", 1, "G", False)])]).splitlines()
             if not ln.startswith("ATOM") or ln[12:16].strip() in keep]
    s = pt.read_pdb("\n".join(lines))
    notes = pt.assign_chirality(s)
    assert next(s.residues()).chirality == "L"
    assert any(n.startswith("Chirality undetermined") and "A:1 LG" in n for n in notes)


def test_parse_ranges():
    assert pt.parse_ranges("C:45-73, A:1-5") == [("C", 45, 73), ("A", 1, 5)]
    assert pt.parse_ranges(["C:45", "B", ":3-4"]) == [("C", 45, 45), ("B", None, None), ("", 3, 4)]
    assert pt.parse_ranges([("A", 2, 9), ("B", 4)]) == [("A", 2, 9), ("B", 4, 4)]
    assert pt.parse_ranges("A:-3--1") == [("A", -3, -1)]
    assert pt.parse_ranges(None) == []
    with pytest.raises(pt.PDBError):
        pt.parse_ranges("45-73")


# --- residue names --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("A", ("A", "rna", "D", "")), ("G5", ("G", "rna", "D", "5")), ("U3", ("U", "rna", "D", "3")),
    ("CN", ("C", "rna", "D", "N")), ("RA5", ("A", "rna", "D", "5")), ("rG", ("G", "rna", "D", "")),
    ("RAN", ("A", "rna", "D", "N")), ("ADE", ("A", "rna", "D", "")), ("GUA", ("G", "rna", "D", "")),
    ("CYT", ("C", "rna", "D", "")), ("URA", ("U", "rna", "D", "")), ("URI", ("U", "rna", "D", "")),
    ("LA", ("A", "rna", "L", "")), ("LG3", ("G", "rna", "L", "3")), ("0G5", ("G", "rna", "L", "5")),
    ("0C", ("C", "rna", "L", "")), ("DA", ("A", "dna", "D", "")), ("DT3", ("T", "dna", "D", "3")),
    ("THY", ("T", "dna", "D", "")), ("HOH", None), ("MG", None), ("PSU", None), ("LT", None), ("RT", None),
])
def test_parse_residue_name(name, expected):
    got = pt.parse_residue_name(name)
    assert (tuple(got) if got else None) == expected


@pytest.mark.parametrize("name, expected", [
    ("0DA", ("A", "dna", "L", "")), ("0DC", ("C", "dna", "L", "")), ("0DG", ("G", "dna", "L", "")),
    ("0DT", ("T", "dna", "L", "")), ("LDA", ("A", "dna", "L", "")), ("LDA5", ("A", "dna", "L", "5")),
    ("LDC3", ("C", "dna", "L", "3")), ("LDGN", ("G", "dna", "L", "N")), ("ldt", ("T", "dna", "L", "")),
    ("0DA5", None), ("0DU", None), ("LDU", None), ("LDX", None), ("0D", None), ("LD5", None),
])
def test_parse_residue_name_l_dna(name, expected):
    """L-DNA: the CCD's 2'-deoxy-L-ribonucleotides 0DA ... (no terminal variants) and the Amber-style LDA ..."""
    got = pt.parse_residue_name(name)
    assert (tuple(got) if got else None) == expected


def test_lda_detergent_is_no_nucleotide():
    """LDA is also the CCD code of a detergent (lauryldimethylamine N-oxide): without backbone atoms it is no
    nucleotide, whatever its name parses as."""
    from mdatk import builder
    from mdatk.modelcheck import residue_kind

    names = ["N1", "O1", "CM1", "CM2"] + [f"C{k}" for k in range(1, 13)]
    det = [atom_line(k + 1, n, "LDA", "D", 1, (k * 1.5, 0.0, 0.0), "HETATM") for k, n in enumerate(names)]
    text = make_pdb([("A", [("DG", 1, "G", True), ("LDA", 2, "A", True)])], extra=det + ["TER"])
    s = pt.read_pdb(text)
    res = {(r.chain, r.resseq): r for r in s.residues()}
    detergent, ldna = res[("D", 1)], res[("A", 2)]
    assert pt.parse_residue_name("LDA") == ("A", "dna", "L", "")
    for r, is_nt in ((detergent, False), (ldna, True)):
        assert (pt.nucleotide(r, infer=True) is not None) == is_nt
        assert (builder._nucleotide(r) is not None) == is_nt
        assert residue_kind(r)[0] == ("nt" if is_nt else "other")
    notes = pt.assign_chirality(s)
    assert notes[0] == "Chirality (sugar geometry): 2 L and 0 D nucleotides"  # the detergent is not counted
    pt.convert_names(s, "rosetta", dna=True)
    assert [r.name for r in s.residues()] == ["0DG", "0DA", "LDA"]


def test_scheme_name_dna():
    rows = {scheme: [pt.scheme_name("T", c, t, scheme, "dna") for c in "DL" for t in ("", "5", "3", "N")]
            for scheme in pt.SCHEMES}
    assert rows == {
        "amber": ["DT", "DT5", "DT3", "DTN", "LDT", "LDT5", "LDT3", "LDTN"],
        "ccd": ["DT"] * 4 + ["0DT"] * 4, "rosetta": ["DT"] * 4 + ["0DT"] * 4, "pdb": ["DT"] * 4 + ["0DT"] * 4,
        "standard": ["DT"] * 8}
    for scheme in pt.SCHEMES:  # every name reads back as the DNA nucleotide it names
        for c in "DL":
            for t in ("", "5", "3", "N"):
                info = pt.parse_residue_name(pt.scheme_name("A", c, t, scheme, "dna"))
                assert info.base == "A" and info.sugar == "dna"
                assert info.chirality == ("D" if scheme == "standard" else c)
                assert info.terminal == (t if scheme == "amber" else "")
    assert pt.scheme_name("G", "L", "5", "amber") == "LG5"  # RNA by default, as before
    with pytest.raises(pt.PDBError):
        pt.scheme_name("G", "D", "", "charmm", "dna")


DNA_HYBRID = [("A", [("G", 1, "G", False), ("C", 2, "C", False)]),
              ("E", [("DG", 3, "G", False), ("DA", 4, "A", False), ("DC", 5, "C", False)]),  # D-DNA
              ("F", [("DG", 6, "G", True), ("DT", 7, "U", True), ("DC", 8, "C", True)]),  # L-DNA with D names
              ("G", [("DA", 9, "A", True)])]  # a single L-DNA residue


def test_convert_names_dna():
    text = make_pdb(DNA_HYBRID)
    # the PDB preparation renames RNA only (dna=False): DNA keeps its names, the L-DNA note says so
    r = pt.clean(text, scheme="amber")
    assert [n for _, _, n in names(r.structure)] == ["G5", "C3", "DG", "DA", "DC", "DG", "DT", "DC", "DA"]
    assert any(n.startswith("L-DNA residues keep their names (the PDB tools rename RNA only; the CCD names "
                            "L-DNA 0DA, 0DC, 0DG, 0DT): F:6 DG") for n in r.notes)
    assert not any(n.startswith("L sugars in") for n in r.notes)  # 'L names when a scheme is applied': RNA only
    by_scheme = {
        "amber": ["G5", "C3", "DG5", "DA", "DC3", "LDG5", "LDT", "LDC3", "LDAN"],
        "ccd": ["G", "C", "DG", "DA", "DC", "0DG", "0DT", "0DC", "0DA"],
        "rosetta": ["G", "C", "DG", "DA", "DC", "0DG", "0DT", "0DC", "0DA"],
        "pdb": ["G", "C", "DG", "DA", "DC", "0DG", "0DT", "0DC", "0DA"],
        "standard": ["G", "C", "DG", "DA", "DC", "DG", "DT", "DC", "DA"],
    }
    for scheme, want in by_scheme.items():
        s = pt.read_pdb(text)
        pt.assign_chirality(s)
        notes = pt.convert_names(s, scheme, dna=True)
        assert [n for _, _, n in names(s)] == want, scheme
        if scheme == "amber":
            # the 5'-phosphate warning covers DNA; L-DNA gets MDATK's names, with a note; LDAN is a unit name
            warn = next(n for n in notes if n.startswith("WARNING: 5'-terminal residues carry P/OP1/OP2"))
            assert "E:3 DG5" in warn and "F:6 LDG5" in warn and "G:9 LDAN" in warn
            assert any(n.startswith("L-DNA: 4 residue(s) (F:6-8, G:9) have MDATK's Amber-style names") for n in notes)
            assert not any("no unit for a single-residue L segment" in n for n in notes)
        if scheme == "ccd":
            assert not any("keep the CCD component names" in n for n in notes)  # no terminal variants for DNA
        if scheme == "standard":
            assert any(n.startswith("WARNING: the standard scheme gives 4 L residue(s) plain D names (F:6-8, G:9)")
                       for n in notes)
        again = pt.read_pdb(pt.write_pdb(s))  # the names read back with their hand (standard: D names)
        hands = [pt.parse_residue_name(r.name).chirality for r in again.residues()]
        assert hands[2:5] == ["D"] * 3 and hands[5:] == (["D"] * 4 if scheme == "standard" else ["L"] * 4)
    s = pt.read_pdb(text)
    pt.assign_chirality(s)
    pt.convert_names(s, "amber", terminals=False, dna=True)
    assert [n for _, _, n in names(s)][2:] == ["DG", "DA", "DC", "LDG", "LDT", "LDC", "LDA"]
    s = pt.read_pdb(text)
    pt.assign_chirality(s)
    pt.convert_names(s, "amber", only="F", dna=True)
    assert [n for _, _, n in names(s)] == ["G", "C", "DG", "DA", "DC", "LDG5", "LDT", "LDC3", "DA"]


def test_l_dna_names_are_l_nucleotides():
    """0DA ... read as L-DNA: counted, measured and kept by the PDB tools (no 'L-named residues with D sugar'
    warning for an L sugar), and their FASTA chain is L-DNA."""
    text = make_pdb([("F", [("0DG", 1, "G", True), ("0DT", 2, "U", True), ("LDC", 3, "C", True)])])
    r = pt.clean(text, scheme="amber", make_fasta=True)
    assert [n for _, _, n in names(r.structure)] == ["0DG", "0DT", "LDC"]
    assert r.notes[0] == "Chirality (sugar geometry): 3 L and 0 D nucleotides"
    assert not any("L-named residues with D sugar" in n for n in r.notes)
    assert "| L-DNA | 3 residues" in r.fasta
    d_geometry = make_pdb([("F", [("0DG", 1, "G", False)])])
    notes = pt.clean(d_geometry, scheme="amber").notes
    assert any(n.startswith("WARNING: L-DNA names on residues with D sugar geometry (the PDB tools keep DNA "
                            "names): F:1 0DG") for n in notes)
    assert not any("L-named residues with D sugar" in n for n in notes)  # that one promises D names: RNA only


@pytest.mark.parametrize("chir, term, amber, ccd, rosetta", [
    ("D", "", "G", "G", "G"), ("D", "5", "G5", "G", "G"), ("D", "3", "G3", "G", "G"),
    ("D", "N", "GN", "G", "G"), ("L", "", "LG", "0G", "0G"), ("L", "5", "LG5", "0G5", "0G"),
    ("L", "3", "LG3", "0G3", "0G"), ("L", "N", "LG5", "0G", "0G"),
])
def test_scheme_name(chir, term, amber, ccd, rosetta):
    got = [pt.scheme_name("G", chir, term, s) for s in ("amber", "ccd", "rosetta", "pdb")]
    assert got == [amber, ccd, rosetta, rosetta]


def test_scheme_name_unknown():
    with pytest.raises(pt.PDBError):
        pt.scheme_name("G", "D", "", "charmm")


HYBRID = [("A", [("G", 1, "G", False), ("C", 2, "C", False), ("A", 3, "A", False), ("U", 4, "U", False)]),
          ("B", [("0G", 5, "G", True), ("0C", 6, "C", True), ("0A", 7, "A", True), ("0U", 8, "U", True)]),
          ("C", [("0A", 9, "A", True)]),
          ("D", [("U", 10, "U", False)]),
          ("E", [("DG", 11, "G", False), ("PSU", 12, "U", False)])]


def test_convert_names_all_schemes_and_round_trip():
    text = make_pdb(HYBRID)
    amber = pt.clean(text, scheme="amber")
    assert [n for _, _, n in names(amber.structure)] == [
        "G5", "C", "A", "U3", "LG5", "LC", "LA", "LU3", "LA5", "UN", "DG", "PSU"]
    assert any("no unit for a single-residue L segment" in n and "C:9 LA5" in n for n in amber.notes)
    assert any("5'-terminal residues carry P/OP1/OP2" in n for n in amber.notes)
    ccd = pt.clean(amber.text, scheme="ccd")
    assert [n for _, _, n in names(ccd.structure)] == [
        "G", "C", "A", "U", "0G", "0C", "0A", "0U3", "0A", "U", "DG", "PSU"]  # 5' phosphates: 0G, not 0G5
    assert any("keep the CCD component names" in n and "B:5" in n for n in ccd.notes)
    assert not any("missing TER?" in n for n in ccd.notes)  # LG5 with its phosphate is still a 5' end
    # a 5'-OH end gets the variant
    no_p = "\n".join(ln for ln in text.splitlines()
                     if not (ln[17:20].strip() == "0G" and ln[12:16].strip() in ("P", "OP1", "OP2", "OP3"))) + "\n"
    assert [n for _, _, n in names(pt.clean(no_p, scheme="ccd").structure)][4] == "0G5"
    rosetta = pt.clean(ccd.text, scheme="rosetta")
    assert [n for _, _, n in names(rosetta.structure)] == [
        "G", "C", "A", "U", "0G", "0C", "0A", "0U", "0A", "U", "DG", "PSU"]
    assert pt.clean(rosetta.text, scheme="amber").text == amber.text
    assert pt.clean(amber.text, scheme="pdb").text == pt.clean(rosetta.text, scheme="pdb").text
    # the residue lines differ only in the residue name columns
    a = [ln for ln in amber.text.splitlines() if ln.startswith("ATOM")]
    r = [ln for ln in rosetta.text.splitlines() if ln.startswith("ATOM")]
    assert [ln[:17] + ln[20:] for ln in a] == [ln[:17] + ln[20:] for ln in r]


def test_convert_names_mirrored_plain_names_and_terminal_mismatch():
    text = make_pdb([("", [("G5", 1, "G", True), ("C", 2, "C", True), ("C3", 3, "C", True)]),
                     ("", [("A5", 4, "A", False), ("U3", 5, "U", False), ("G", 6, "G", False)])])
    r = pt.clean(text, scheme="amber")
    assert [n for _, _, n in names(r.structure)] == ["LG5", "LC", "LC3", "A5", "U", "G3"]
    assert any(n.startswith("Terminal names that are not at a segment end") and ":5 U3" in n for n in r.notes)
    assert pt.clean(text, scheme="amber", detect_l=False).structure.segments[0][0].name == "G5"


def test_to_lrna_renames_only_ranges_and_keeps_terminals():
    text = make_pdb([("A", [("G5", 1, "G", False), ("C", 2, "C", False), ("A", 3, "A", False),
                            ("U3", 4, "U", False)]),
                     ("B", [("G5", 1, "G", False), ("C3", 2, "C", False)])])
    r = pt.to_lrna(text, "A:3-4, B")
    assert names(r.structure) == [("A", 1, "G5"), ("A", 2, "C"), ("A", 3, "LA"), ("A", 4, "LU3"),
                                  ("B", 1, "LG5"), ("B", 2, "LC3")]
    assert any("forced L although the sugar geometry is D" in n for n in r.notes)
    r = pt.to_lrna(text, [("B", 1, 2)], scheme="ccd")
    assert [n for _, _, n in names(r.structure)][-2:] == ["0G", "0C3"]  # the 5' end keeps its phosphate
    assert [n for _, _, n in names(r.structure)][:4] == ["G5", "C", "A", "U3"]


# --- parsing / writing ----------------------------------------------------------------------------------

def chimerax_damage(line):
    """Fuse residue name and chain id with a colon ending in column 22, as ChimeraX does."""
    fused = f"{line[17:20].strip()}{line[21]}:"
    return line[:22 - len(fused)] + fused + line[22:]


def test_chimerax_repair():
    segs = [("C", [("G", 1, "G", True), ("0A", 2, "A", True), ("LG5", 3, "G", True)]),
            ("B", [("C", 4, "C", False), ("U", 5, "U", False)])]
    good = make_pdb(segs)
    damaged = "\n".join(chimerax_damage(ln) if ln.startswith(("ATOM", "TER")) else ln
                        for ln in good.splitlines())
    assert "  GC:   1" in damaged and "LG5C:   3" in damaged
    assert all(ln[21] == ":" for ln in damaged.splitlines() if ln.startswith(("ATOM", "TER")))
    fixed = pt.clean(damaged)
    assert fixed.text == pt.clean(good).text
    assert names(fixed.structure) == [("C", 1, "G"), ("C", 2, "0A"), ("C", 3, "LG5"), ("B", 4, "C"),
                                      ("B", 5, "U")]
    n_atoms = good.count("\nATOM") + good.startswith("ATOM")
    assert any(n.startswith(f"Repaired {n_atoms} ChimeraX-damaged") for n in fixed.notes)

    line = good.splitlines()[0]
    shifted = line[:17] + "  LG5C:" + line[22:]  # wider fused field: everything right of it shifted by 2
    res = next(pt.read_pdb(shifted).residues())
    assert (res.name, res.chain, res.resseq) == ("LG5", "C", 1)
    assert list(res.atoms[0].xyz) == pytest.approx([-2.314, 1.777, -0.478])
    ragged = "ATOM      1  P     0GC:  7   -2.314   1.777   -0.478 1.00 0.00 P"
    res = next(pt.read_pdb(ragged).residues())
    assert (res.name, res.chain, res.resseq, res.atoms[0].element) == ("0G", "C", 7, "P")
    assert list(res.atoms[0].xyz) == pytest.approx([-2.314, 1.777, -0.478])
    left = line[:17] + "G   C" + line[22:]  # ChimeraX also left-justifies residue names
    assert next(pt.read_pdb(left).residues()).name == "G"


def test_writer_columns_alignment_and_elements():
    raw = "\n".join([
        "ATOM      7 C1'    G A   1       1.000   2.000   3.000  1.00  0.00",  # left-aligned, no element
        "ATOM      8 H5''   G A   1       1.000   2.000   3.000  1.00  0.00",
        "HETATM    9 MG    MG A   2       4.000   5.000   6.000  1.00  0.00",
        "HETATM   10 Na+  Na+ A   3       4.000   5.000   6.000  1.00  0.00",
        "ATOM     11  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C",
    ])
    out = pt.clean(raw).text.splitlines()
    assert out[0] == "ATOM      1  C1'   G A   1       1.000   2.000   3.000  1.00  0.00           C"
    assert out[1][12:16] == "H5''" and out[1][76:78] == " H"
    assert out[2][12:16] == "MG  " and out[2][76:78] == "MG" and out[2][17:20] == " MG"
    assert out[3][12:16] == "Na+ " and out[3][76:78] == "NA"
    assert [ln[:6] for ln in out] == ["ATOM  ", "ATOM  ", "HETATM", "HETATM", "TER   ", "ATOM  ", "TER   ",
                                      "END"]  # TER after each chain, END
    assert out[4] == "TER       5      Na+ A   3"
    assert out[5][12:16] == " CA " and out[5][6:11] == "    6"


def test_models_header_trailer_and_anisou():
    body = make_pdb([("A", [("G", 1, "G", False)])]).splitlines()[:-1]  # without END
    second = [ln.replace("  1.00  0.00", "  1.00  9.99") for ln in body]
    anisou = "ANISOU    1  P     G A   1     1000   2000   3000    100    200    300       P"
    text = "\n".join(["HEADER    RNA", "REMARK   1 kept", "SEQRES   1 A    1    G", "CRYST1   10.000",
                      "MODEL        1", body[0], anisou] + body[1:] + ["ENDMDL", "MODEL        2"] + second
                     + ["ENDMDL", "MASTER        0", "END"])
    r = pt.clean(text)
    out = r.text.splitlines()
    assert out[:5] == ["HEADER    RNA", "REMARK   1 kept", "SEQRES   1 A    1    G", "CRYST1   10.000",
                       "MODEL        1"]
    assert out[6].startswith("ANISOU    1  P     G A   1     1000") and out[6].endswith("P")
    assert out[-3:] == [f"TER   {len(body):>5}        G A   1", "ENDMDL", "END"]
    assert "9.99" not in r.text and "MASTER" not in r.text
    assert "Kept model 1 of 2" in r.notes and any(n.startswith("Dropped the MASTER") for n in r.notes)
    with pytest.raises(pt.PDBError):
        pt.read_pdb("REMARK only\nEND\n")
    with pytest.raises(pt.PDBError):
        pt.read_pdb("/no/such/file.pdb")


def test_record_policy():
    water = [atom_line(90, "O", "HOH", "W", 1, (9, 9, 9), "HETATM"),
             atom_line(91, "H1", "HOH", "W", 1, (9, 9, 9), "HETATM"),
             atom_line(92, "MG", "MG", "W", 2, (8, 8, 8), "HETATM", "MG"), "TER"]
    text = make_pdb([("A", [("G", 1, "G", False), ("0A", 2, "A", True), ("DG", 3, "G", False)])],
                    records={"0A": "HETATM", "G": "HETATM"}, extra=water)

    def records(result):
        return {ln[17:20].strip(): ln[:6].strip() for ln in result.text.splitlines()
                if ln.startswith(("ATOM", "HETATM"))}

    het = {"HOH": "HETATM", "MG": "HETATM"}
    assert records(pt.clean(text)) == {"G": "HETATM", "0A": "HETATM", "DG": "ATOM", **het}
    ros = pt.clean(text, records="rosetta")
    assert records(ros) == {"G": "ATOM", "0A": "ATOM", "DG": "ATOM", **het}
    assert any(n.startswith("Record policy rosetta") for n in ros.notes)
    assert records(pt.clean(text, records="pdb")) == {"G": "ATOM", "0A": "HETATM", "DG": "ATOM", **het}
    assert records(pt.clean(text, scheme="rosetta"))["0A"] == "ATOM"  # the scheme picks its policy
    assert records(pt.clean(text, scheme="pdb"))["0A"] == "HETATM"
    with pytest.raises(pt.PDBError):
        pt.clean(text, records="charmm")


def test_conect_strip_and_remap():
    text = make_pdb([("A", [("G", 1, "G", False)])], first_serial=101,
                    extra=["CONECT  101  102  103", "CONECT  102  101"])
    r = pt.clean(text)
    assert "CONECT" not in r.text and "Stripped 2 CONECT records" in r.notes
    kept = pt.clean(text, strip_conect=False).text.splitlines()
    assert kept[-3:] == ["CONECT    1    2    3", "CONECT    2    1", "END"]


# --- chains ---------------------------------------------------------------------------------------------

CPPTRAJ = [("", [("G5", 1, "G", False), ("C", 2, "C", False), ("U3", 3, "U", False)]),
           ("", [("G5", 4, "G", True), ("A", 5, "A", True), ("C3", 6, "C", True)])]
REFERENCE = [("A", [("G", 1, "G", False), ("C", 2, "C", False), ("U", 3, "U", False)]),
             ("B", [("LG", 1, "G", True), ("LA", 2, "A", True), ("LC", 3, "C", True)])]
IONS = [atom_line(500, "Na+", "Na+", "", 7, (1, 1, 1), "ATOM", "NA"), "TER",
        atom_line(501, "O", "WAT", "", 8, (2, 2, 2)), atom_line(502, "H1", "WAT", "", 8, (2, 2, 2)), "TER"]


def test_restore_chains_from_reference():
    text, ref = make_pdb(CPPTRAJ, extra=IONS), make_pdb(REFERENCE)
    r = pt.clean(text, chain_reference=ref, scheme="amber")
    assert names(r.structure) == [("A", 1, "G5"), ("A", 2, "C"), ("A", 3, "U3"), ("B", 4, "LG5"),
                                  ("B", 5, "LA"), ("B", 6, "LC3"), ("", 7, "Na+"), ("", 8, "WAT")]
    assert any(n.startswith("Restored chain ids of 6 residues") for n in r.notes)
    r = pt.clean(text, chain_reference=ref, copy_numbers=True)
    assert [(c, n) for c, n, _ in names(r.structure)][:6] == [("A", 1), ("A", 2), ("A", 3),
                                                              ("B", 1), ("B", 2), ("B", 3)]
    one_segment = make_pdb([("", CPPTRAJ[0][1] + CPPTRAJ[1][1])])
    assert len(pt.clean(one_segment, chain_reference=ref).structure.segments) == 2  # TER between chains


def test_restore_chains_mismatch_errors():
    text = make_pdb(CPPTRAJ)
    short = make_pdb([REFERENCE[0], ("B", REFERENCE[1][1][:2])])
    with pytest.raises(pt.PDBError, match="has 6 residues but the reference has 5"):
        pt.clean(text, chain_reference=short)
    other = make_pdb([REFERENCE[0],
                      ("B", [("LG", 1, "G", True), ("LU", 2, "U", True), ("LC", 3, "C", True)])])
    with pytest.raises(pt.PDBError, match=r"sequence differs from the reference at 1 residue\(s\): #5"):
        pt.clean(text, chain_reference=other)


def test_restore_chains_from_list_and_alphabetical():
    text = make_pdb(CPPTRAJ, extra=IONS)
    r = pt.clean(text, chain_ids=["X", "Y"])
    assert [c for c, _, _ in names(r.structure)] == ["X"] * 3 + ["Y"] * 3 + ["", ""]
    r = pt.clean(text, assign_chains=True)
    assert [c for c, _, _ in names(r.structure)] == ["A"] * 3 + ["B"] * 3 + ["", ""]
    assert "Assigned chain ids A, B to 2 TER segments" in r.notes
    with pytest.raises(pt.PDBError, match="1 chain ids given for 2 TER segments"):
        pt.clean(text, chain_ids=["A"])


def test_split_and_combine_chains():
    text = make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)]), ("A", [("A", 3, "A", False)]),
                     ("B", [("U", 1, "U", False)])])
    r = pt.clean(text, split_segments=True)
    assert [c for c, _, _ in names(r.structure)] == ["A", "A", "C", "B"]
    five = make_pdb([("A", [(b, k + 1, b, False) for k, b in enumerate("GCAUG")])])
    r = pt.clean(five, split_at="A:3", scheme="amber")
    assert names(r.structure) == [("A", 1, "G5"), ("A", 2, "C3"), ("B", 3, "A5"), ("B", 4, "U"),
                                  ("B", 5, "G3")]
    assert r.text.count("TER") == 2
    with pytest.raises(pt.PDBError):
        pt.clean(five, split_at="A:9")
    r = pt.clean(text, combine=["A", "B"])
    assert {c for c, _, _ in names(r.structure)} == {"A"} and r.text.count("TER") == 3
    assert any("duplicate residue numbers (1)" in n for n in r.notes)
    r = pt.clean(text, combine=["A", "B"], combine_into="Z", renumber=True)
    assert [(c, n) for c, n, _ in names(r.structure)] == [("Z", 1), ("Z", 2), ("Z", 3), ("Z", 4)]


def test_order_chains():
    text = make_pdb([("C", [("G", 1, "G", False)]), ("A", [("C", 2, "C", False)]),
                     ("B", [("A", 3, "A", False)])], extra=IONS)
    assert pt.clean(text, sort_chains=True).structure.chain_ids() == ["A", "B", "C", ""]
    assert pt.clean(text, sort_chains=["B"]).structure.chain_ids() == ["B", "C", "A", ""]


# --- numbering and sequence -----------------------------------------------------------------------------

def numbered(spec):
    """spec: [(chain, [(resseq, icode), ...])] -> PDB text of G residues."""
    lines, serial = [], 0
    for chain, residues in spec:
        for resseq, icode in residues:
            for name, x, y, z in residue_atoms("G"):
                serial += 1
                ln = atom_line(serial, name, "G", chain, resseq, (x, y, z))
                lines.append(ln[:26] + (icode or " ") + ln[27:])
        lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


GAPPED = [("A", [(5, ""), (6, ""), (10, ""), (10, "A"), (11, "")]), ("B", [(20, ""), (21, "")])]


@pytest.mark.parametrize("options, expected", [
    ({}, [1, 2, 3, 4, 5, 6, 7]),
    ({"per_chain": True}, [1, 2, 3, 4, 5, 1, 2]),
    ({"keep_gaps": True}, [1, 2, 6, 6, 7, 8, 9]),
    ({"keep_gaps": True, "per_chain": True}, [1, 2, 6, 6, 7, 1, 2]),
    ({"start": 100}, [100, 101, 102, 103, 104, 105, 106]),
])
def test_renumber_residues(options, expected):
    r = pt.clean(numbered(GAPPED), renumber=True, **options)
    assert [res.resseq for res in r.structure.residues()] == expected
    icodes = [res.icode for res in r.structure.residues()]
    assert icodes[3] == ("A" if options.get("keep_gaps") else "")
    if not options.get("keep_gaps"):
        assert "Cleared 1 insertion codes" in r.notes


def test_seqres_records_and_placement():
    seq = "GCAUGCAUGCAUGCA"
    text = make_pdb([("A", [(b, k + 1, b, False) for k, b in enumerate(seq)]),
                     ("B", [("0G", 1, "G", True), ("0C", 2, "C", True)])],
                    header=["REMARK   1 note", "SEQRES   1 Z    1    A", "CRYST1   10.000"], extra=IONS)
    r = pt.clean(text, seqres=True)
    out = r.text.splitlines()
    assert out[:5] == [
        "REMARK   1 note",
        "SEQRES   1 A   15    G   C   A   U   G   C   A   U   G   C   A   U   G",
        "SEQRES   2 A   15    C   A",
        "SEQRES   1 B    2   0G  0C",
        "CRYST1   10.000"]
    line = out[1]
    assert (line[0:6], line[7:10], line[11], line[13:17], line[19:22], line[67:70]) == (
        "SEQRES", "  1", "A", "  15", "  G", "  G")
    assert "Wrote SEQRES records for 2 chain(s)" in r.notes


def test_fasta():
    text = make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False), ("LA", 3, "A", True),
                            ("LU", 4, "U", True)]),
                     ("B", [("G", 1, "G", True), ("C", 2, "C", True)]),
                     ("C", [("DG", 1, "G", False), ("PSU", 2, "U", False)])], extra=IONS)
    assert pt.clean(text, make_fasta=True, title="demo").fasta == (
        ">demo_A chain A | RNA, L-nucleotides at 3-4 | 4 residues\nGCAU\n"
        ">demo_B chain B | L-RNA | 2 residues\nGC\n"
        ">demo_C chain C | DNA | 2 residues\nGX\n")
    assert pt.clean(text).fasta is None


def test_clean_end_to_end_from_path(tmp_path):
    path = tmp_path / "solvated.pdb"
    path.write_text(make_pdb(CPPTRAJ, extra=IONS + ["CONECT  500  501"],
                             header=["CRYST1  105.626  105.626  105.626 109.47 109.47 109.47"]))
    ref = tmp_path / "ref.pdb"
    ref.write_text(make_pdb(REFERENCE))
    r = pt.clean(str(path), chain_reference=str(ref), scheme="ccd", renumber=True, per_chain=True,
                 seqres=True, make_fasta=True)
    assert names(r.structure)[:6] == [("A", 1, "G"), ("A", 2, "C"), ("A", 3, "U"), ("B", 1, "0G"),
                                      ("B", 2, "0A"), ("B", 3, "0C3")]
    assert r.fasta.startswith(">solvated_A chain A | RNA | 3 residues\nGCU\n>solvated_B chain B | L-RNA")
    out = r.text.splitlines()
    assert out[0] == "SEQRES   1 A    3    G   C   U" and out[2].startswith("CRYST1")
    assert out[-1] == "END" and "CONECT" not in r.text
    assert sum(ln.startswith("TER") for ln in out) == 4
    serials = [int(ln[6:11]) for ln in out if ln.startswith(("ATOM", "HETATM", "TER"))]
    assert serials == list(range(1, len(serials) + 1))


# --- command line --------------------------------------------------------------------------------------

def test_mdatk_pdb_cli_check_clean_and_log(tmp_path, capsys):
    import json

    from mdatk.cli import main
    from mdatk.config import Project
    from mdatk.projectfile import read_log

    pdb = tmp_path / "in.pdb"
    pdb.write_text(make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)]),
                             ("B", [("G", 3, "G", True), ("U", 4, "U", True)])]))
    assert main(["pdb", "check", str(pdb)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["chains"]["A"]["chirality"] == {"D": 2} and info["chains"]["B"]["chirality"] == {"L": 2}
    proj = tmp_path / "p.mdatk.toml"
    Project().save(str(proj))
    out, fa = tmp_path / "out.pdb", tmp_path / "out.fasta"
    assert main(["pdb", "clean", str(pdb), "-o", str(out), "--scheme", "amber", "--renumber", "--start", "10",
                 "--seqres", "--fasta", str(fa), "-p", str(proj)]) == 0
    s = pt.read_pdb(str(out))
    assert names(s) == [("A", 10, "G5"), ("A", 11, "C3"), ("B", 12, "LG5"), ("B", 13, "LU3")]
    assert "SEQRES" in out.read_text() and ">" in fa.read_text()
    entry = read_log(str(proj))[-1]
    assert entry["step"] == "pdb clean" and str(out) in entry["files"] and "Chirality" in entry["summary"]
    assert main(["pdb", "to-lrna", str(pdb), "A:2", "-o", str(tmp_path / "l.pdb")]) == 0
    assert [r.name for r in pt.read_pdb(str(tmp_path / "l.pdb")).residues()][:2] == ["G", "LC3"]


def test_embedded_copy_runs_standalone(tmp_path):
    import subprocess
    import sys

    from mdatk.workflows.scripts import embed
    script = tmp_path / "pdbtools_copy.py"
    script.write_text(embed(pt, {}, "main()", "PDB hygiene (embedded copy)"))
    pdb = tmp_path / "in.pdb"
    pdb.write_text(make_pdb([("A", [("G", 1, "G", True), ("C", 2, "C", True)])]))
    out = tmp_path / "out.pdb"
    proc = subprocess.run([sys.executable, "-I", str(script), "clean", str(pdb), "-o", str(out),
                           "--scheme", "amber"], capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert [r.name for r in pt.read_pdb(str(out)).residues()] == ["LG5", "LC3"]


def test_terminal_names_can_be_left_to_leap():
    pdb = make_pdb([("C", [("0G", 45, "G", True), ("0C", 46, "C", True), ("0A", 47, "A", True)])])
    res = pt.clean(pdb, scheme="amber", terminal_names=False)
    assert [r.name for r in res.structure.residues()] == ["LG", "LC", "LA"]  # like to_lrna.sh
    assert [r.name for r in pt.clean(pdb, scheme="amber").structure.residues()] == ["LG5", "LC", "LA3"]
    out = pt.to_lrna(make_pdb([("A", [("G", 1, "G", True), ("C", 2, "C", True)])]), "A:1-2", terminals=False)
    assert [r.name for r in out.structure.residues()] == ["LG", "LC"]


def test_split_at_several_positions_in_file_order():
    pdb = make_pdb([("A", [("G", i, "G", False) for i in range(1, 7)])])
    s = pt.read_pdb(pdb)
    pt.split_chains(s, "A:5, A:3")
    assert [(r.chain, r.resseq) for r in s.residues()] == [
        ("A", 1), ("A", 2), ("B", 3), ("B", 4), ("C", 5), ("C", 6)]


def test_single_residue_l_segment_has_no_false_terminal_warning():
    once = pt.clean(make_pdb([("C", [("0G", 45, "G", True)])]), scheme="amber").text
    again = pt.clean(once, scheme="amber")
    assert [r.name for r in again.structure.residues()] == ["LG5"]
    assert not any("missing TER" in n for n in again.notes)


def test_strip_5prime_phosphates_for_amber():
    text = make_pdb(HYBRID)
    r = pt.clean(text, scheme="amber", strip_5p=True)
    assert any("Removed the 5'-terminal phosphate of" in n for n in r.notes)
    assert not any("carry P/OP1/OP2" in n for n in r.notes)  # the Amber warning is gone
    for seg in r.structure.segments:
        first = [res for res in seg if pt._nucleotide(res)]
        if first:
            assert not {a.name for a in first[0].atoms} & {"P", "OP1", "OP2", "OP3"}
    # a later residue keeps its phosphate; without the option nothing is removed
    assert any(a.name == "P" for a in r.structure.segments[0][1].atoms)
    assert "Removed" not in " ".join(pt.clean(text, scheme="amber").notes)
    # hydrogens present: HO5' 0.96 A from O5', along the old O5'-P bond
    lines = text.splitlines()
    p_line = next(ln for ln in lines if ln[12:16].strip() == "P")
    hline = p_line[:12] + " H5'" + p_line[16:]
    lines.insert(lines.index(p_line), hline)
    r = pt.clean("\n".join(lines) + "\n", strip_5p=True)
    res = r.structure.segments[0][0]
    c = res.coords()
    assert "HO5'" in c and abs(math.dist(c["HO5'"], c["O5'"]) - 0.96) < 0.01


def test_strip_5p_keeps_internal_phosphates_and_drops_lone_phosphates():
    text = make_pdb([("A", [("XYZ", 1, "G", False), ("G", 2, "G", False), ("C", 3, "C", False)])])
    r = pt.clean(text, strip_5p=True)
    assert any(a.name == "P" for a in r.structure.segments[0][1].atoms)  # G2 is not a 5' end here
    assert any("does not start with a recognised nucleotide" in n for n in r.notes)
    lines = [ln for ln in make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)])]).splitlines()
             if not (ln.startswith("ATOM") and ln[22:26].strip() == "1" and ln[12:16].strip() not in
                     ("P", "OP1", "OP2"))]  # residue 1 keeps only its phosphate
    r = pt.clean("\n".join(lines) + "\n", strip_5p=True)
    res = list(r.structure.residues())
    assert [x.resseq for x in res] == [2] and not any(a.name == "P" for a in res[0].atoms)
    assert any("held only a phosphate" in n for n in r.notes)


def test_link_line_columns_and_round_trip():
    a, b = pt.LinkAtom("O3'", "DA", "A", 6), pt.LinkAtom("P", "PO2", "A", 7)
    line = pt.link_line(a, b, 1.6)
    assert line[:6] == "LINK  " and line[12:16] == " O3'" and line[17:20] == " DA" and line[21] == "A"
    assert line[22:26] == "   6" and line[42:46] == " P  " and line[47:50] == "PO2" and line[51] == "A"
    assert line[52:56] == "   7" and line[59:65] == "  1555" and line[66:72] == "  1555" and line[73:78] == " 1.60"
    four = pt.link_line(pt.LinkAtom("OP1", "LDA5", "B", 12, "A"), pt.LinkAtom("C4'", "A", "C", -3))
    s = pt.Structure(header=["REMARK 951 x", line, four, "LINK  garbage"])
    assert pt.link_records(s) == [
        pt.LinkRecord(a, b, 1.6),
        pt.LinkRecord(pt.LinkAtom("OP1", "LDA5", "B", 12, "A"), pt.LinkAtom("C4'", "A", "C", -3), None)]
    # LINK records stay in the header through a read / write
    text = "\n".join([line, "ATOM      1  O3'  DA A   6       0.000   0.000   0.000  1.00  0.00           O",
                      "END"]) + "\n"
    again = pt.read_pdb(pt.write_pdb(pt.read_pdb(text)))
    assert pt.link_records(again) == [pt.LinkRecord(a, b, 1.6)]
    assert a.matches(next(again.residues()))


# --- LINK records follow the residues -------------------------------------------------------------------

def _digest(text, notes=(), fasta=None):
    return hashlib.sha256("\0".join([text, "\n".join(notes), fasta or ""]).encode()).hexdigest()[:16]


def _operations_in_turn(text):
    """Every operation that renames, renumbers, re-chains or drops residues, called on its own in turn."""
    s = pt.read_pdb(text)
    notes = pt.assign_chirality(s) + pt.convert_names(s, "amber", dna=True) + pt.strip_hydrogens(s)
    notes += pt.strip_5prime_phosphates(s) + pt.drop_virtual_residues(s) + pt.split_chains(s, "A:2")
    notes += pt.combine_chains(s, ["B", "C"], "Q") + pt.order_chains(s) + pt.renumber_residues(s, 3, per_chain=True)
    notes += pt.restore_chains(s) + pt.split_chains(s)
    return _digest(pt.write_pdb(s, seqres=True), notes)


def _link_free_runs():
    """name -> digest of the text, notes and FASTA of PDB-tool runs on models without LINK records."""
    hybrid, dna = make_pdb(HYBRID), make_pdb(DNA_HYBRID)
    vrt = [atom_line(900 + k, n, "VRT", "Z", 1, (k, 0.0, 0.0)) for k, n in enumerate(("ORIG", "X", "Y"))]
    decoy = make_pdb(HYBRID[:2], extra=vrt + ["TER"], header=["REMARK 950 G1-C2 pair", "CRYST1   10.000"])
    solvated = make_pdb(CPPTRAJ, extra=IONS + ["CONECT  500  501", "CONECT  501  500"],
                        header=["CRYST1  105.626  105.626  105.626 109.47 109.47 109.47"])
    lone = "\n".join(ln for ln in make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)])]).splitlines()
                     if not (ln.startswith("ATOM") and ln[22:26].strip() == "1"
                             and ln[12:16].strip() not in ("P", "OP1", "OP2"))) + "\n"
    runs = {f"{name} {scheme}": (lambda t=text, sc=scheme: pt.clean(t, scheme=sc))
            for name, text in (("hybrid", hybrid), ("dna", dna)) for scheme in pt.SCHEMES}
    runs.update({
        "decoy for tleap": lambda: pt.clean(decoy, scheme="amber", strip_5p=True, strip_h=True, drop_virtual=True),
        "standard copy": lambda: pt.clean(decoy, scheme="standard", strip_h=True),
        "chains from a reference": lambda: pt.clean(solvated, chain_reference=make_pdb(REFERENCE), copy_numbers=True,
                                                    scheme="ccd", seqres=True, make_fasta=True, strip_conect=False),
        "chains assigned, split, sorted": lambda: pt.clean(solvated, assign_chains=True, split_at="A:2",
                                                           sort_chains=True, renumber=True, per_chain=True),
        "chain list, combined": lambda: pt.clean(solvated, chain_ids=["X", "Y"], combine=["X", "Y"],
                                                 combine_into="Z", l_ranges="Y", scheme="rosetta"),
        "segments split, ordered": lambda: pt.clean(make_pdb(HYBRID[:1] * 2 + HYBRID[2:4]), split_segments=True,
                                                    sort_chains=["C", "A"], renumber=True, start=7),
        "gaps kept": lambda: pt.clean(numbered(GAPPED), renumber=True, keep_gaps=True, start=10),
        "lone phosphate": lambda: pt.clean(lone, strip_5p=True, scheme="amber"),
        "to-lrna": lambda: pt.to_lrna(hybrid, "A:3-4, D"),
        "to-lrna ccd": lambda: pt.to_lrna(hybrid, [("B", 5, 8)], scheme="ccd", terminals=False),
    })
    out = {name: _digest(r.text, r.notes, r.fasta) for name, run in runs.items() for r in [run()]}
    out["operations in turn"] = _operations_in_turn(hybrid)
    out["operations in turn, decoy"] = _operations_in_turn(decoy)
    return out


# recorded at d07a5ef, before LINK records followed the residues
LINK_FREE_D07A5EF = {
    "hybrid amber": "03f7589cabdaac9f", "hybrid ccd": "59073daf87d78b1a", "hybrid rosetta": "2ebb9c0a56d3cd1a",
    "hybrid pdb": "c7b9038737181096", "hybrid standard": "25f9c8192b6998d0", "dna amber": "ed4bbe0a34da2096",
    "dna ccd": "50c0d91d28b574d2", "dna rosetta": "945cbb9dafb03b8e", "dna pdb": "2fe4081b30b15cf6",
    "dna standard": "c6432fadfcf7f038", "decoy for tleap": "87bfeb0cd803d26a", "standard copy": "3048c24ee1b4231f",
    "chains from a reference": "8f3dbea8a58cfe5c", "chains assigned, split, sorted": "b2b2e4858e35d9da",
    "chain list, combined": "79a0bedefebae9dd", "segments split, ordered": "e5e7485a629abea8",
    "gaps kept": "57773d327c46120a", "lone phosphate": "dcaebfe3cfc99b46", "to-lrna": "09483c3997e10797",
    "to-lrna ccd": "9eac0008a3bc413d", "operations in turn": "9e937bd003ad1cb4",
    "operations in turn, decoy": "c97bebb24865781d"}


def test_models_without_link_records_are_written_as_before():
    """Following LINK records changes nothing for a model without them: text, notes and FASTA of every name
    scheme, chain and numbering operation, hydrogen / phosphate / virtual-residue removal and to-lrna are those
    of d07a5ef, byte for byte (an intended change of that output records the digests again from
    _link_free_runs())."""
    assert _link_free_runs() == LINK_FREE_D07A5EF


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _place(atom, xyz):
    atom.tail = f"   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + atom.tail[27:]


def bridged(end="3", hydrogens=False, phosphate5=False):
    """PDB text of a chimera junction of the structure builder: single strands A:1-3 and B:1-3 (build_helix)
    joined 3'-3' (``end`` '3') or 5'-5' ('5') by a PO2 residue, A:4 / A:0, bonded to the O3' / O5' of both,
    with LINK records for the two bonds.  The geometry of tests/test_modelcheck.py's LINK tests: P
    where strand A's next phosphate is in the A-form, strand B the image of strand A under the two-fold axis
    through P that bisects O-P-O (the O-P-O plane turned 100 / 20 degrees from the helix's own), OP1 / OP2
    1.48 Å from P.  ``phosphate5``: the 3'-3' strands start with a 5'-phosphate."""
    three = end == "3"
    k, oname, at = (3, "O3'", 4) if three else (1, "O5'", 1)
    strands = [pt.read_pdb(builder.build_helix("AAAA" if three else "AAA", partner="none", chains=chain,
                                               hydrogens=hydrogens, phosphate5=phosphate5 or not three))
               for chain in "AB"]
    xyz = {r.resseq: {a.name: a.xyz for a in r.atoms} for r in strands[0].residues()}
    p = xyz[at]["P"]
    d1 = _unit([xyz[k][oname][x] - p[x] for x in range(3)])
    d2 = _unit([xyz[at]["O5'" if three else "OP3"][x] - p[x] for x in range(3)])  # the helix's own other O
    t = math.radians(100.0 if three else 20.0)  # d2 turned about d1 (Rodrigues)
    c1, dot = _cross(d1, d2), sum(d1[x] * d2[x] for x in range(3))
    d2 = [d2[x] * math.cos(t) + c1[x] * math.sin(t) + d1[x] * dot * (1 - math.cos(t)) for x in range(3)]
    n, m = _unit([d1[x] + d2[x] for x in range(3)]), _unit(_cross(d1, d2))
    cos, sin = math.cos(math.radians(57.7)), math.sin(math.radians(57.7))
    po2 = pt.Residue("PO2", "A", 4 if three else 0)
    for name, sign in (("P", 0), ("OP1", 1), ("OP2", -1)):
        po2.atoms.append(pt.Atom("HETATM", name, "", "   " + "0" * 24 + "  1.00  0.00", name[0]))
        _place(po2.atoms[-1], [p[x] + (1.48 * (sign * sin * m[x] - cos * n[x]) if sign else 0.0) for x in range(3)])
    drop = {"HO3'"} if three else {"P", "OP1", "OP2", "OP3", "HO5'"}
    for strand in strands:
        strand.segments[0] = strand.segments[0][:3]
        for res in strand.segments[0]:
            if res.resseq == k:
                res.atoms = [a for a in res.atoms if a.name not in drop]
    for res in strands[1].residues():  # the two-fold rotation about n through P
        for a in res.atoms:
            v = [a.xyz[x] - p[x] for x in range(3)]
            along = sum(v[x] * n[x] for x in range(3))
            _place(a, [p[x] + 2 * along * n[x] - v[x] for x in range(3)])
    s = strands[0]
    s.segments[0].insert(3 if three else 0, po2)
    s.segments += strands[1].segments
    s.header += [pt.link_line(pt.LinkAtom(oname, "A", chain, k), pt.LinkAtom("P", "PO2", "A", po2.resseq), 1.6)
                 for chain in "AB"]
    return pt.write_pdb(s)


def junction(end):
    """The atom pairs the LINK records of :func:`bridged` bond, as the clash check labels them."""
    k, oname, po2 = (3, "O3'", "A:4") if end == "3" else (1, "O5'", "A:0")
    return [(f"A:{k}:{oname}", f"{po2}:P"), (f"B:{k}:{oname}", f"{po2}:P")]


def link_bonds(text):
    """The LINK bonds and the clashes the model check finds in PDB text."""
    r = mc.check_clashes(pt.read_pdb(text))
    return [(b.first, b.second) for b in r.links], [(c.first, c.second) for c in r.clashes], r.links_unused


def link_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith("LINK")]


@pytest.mark.parametrize("end", ["3", "5"])
@pytest.mark.parametrize("scheme", pt.SCHEMES)
def test_link_records_follow_the_names_of_every_scheme(end, scheme):
    """The structure builder's chimeras bond a PO2 phosphate to two strands and say so in LINK records; the
    model check bonds them by chain, number, insertion code and residue name.  The residues bonded through the
    junction are no strand ends: they keep the internal names of the scheme (amber A / LA, not A3 / LA3), as the
    builder names them, and the names a scheme gives them are written into the records, which keep everything
    else."""
    text = bridged(end)
    k, po2 = (3, 4) if end == "3" else (1, 0)
    r = pt.clean(text, scheme=scheme, l_ranges="B")  # strand B as L: every scheme but standard renames it
    names = {"amber": ("A", "LA"), "ccd": ("A", "0A"), "rosetta": ("A", "0A"), "pdb": ("A", "0A"),
             "standard": ("A", "A")}[scheme]
    assert [(res.chain, res.resseq, res.name) for res in r.structure.residues() if res.resseq == k] == [
        ("A", k, names[0]), ("B", k, names[1])]
    oname = "O3'" if end == "3" else "O5'"
    assert pt.link_records(r.structure) == [
        pt.LinkRecord(pt.LinkAtom(oname, name, chain, k), pt.LinkAtom("P", "PO2", "A", po2), 1.6)
        for chain, name in zip("AB", names)]
    assert pt.link_records(pt.read_pdb(r.text)) == pt.link_records(r.structure)
    changed = [n for n in names if n != "A"]
    assert [n for n in r.notes if "LINK record(s)" in n] == ([
        f"Rewrote {len(changed)} LINK record(s) for the new residue names, numbers or chain ids: "
        + ", ".join(f"{c}:{k} {n} {oname} - A:{po2} PO2 P" for c, n in zip("AB", names) if n != "A")]
        if changed else [])
    # the residues whose name a free end would change (amber: both; ccd: the L one) are listed
    junction_note = [n for n in r.notes if n.startswith("Segment ends that LINK records bond")]
    listed = {"amber": f"A:{k} A, B:{k} LA", "ccd": f"B:{k} 0A"}.get(scheme)
    assert junction_note == ([f"Segment ends that LINK records bond to another residue (a junction such as a "
                              f"chimera's 3'-3' / 5'-5' phosphate, or a strand bonded on) have internal names: "
                              f"{listed}"] if listed else [])
    # amber: the residues bonded through O5' have no 5'-phosphate their (internal) units may add
    assert any(n.startswith(f"WARNING: the residues bonded through O5' by LINK records (A:{k} A, B:{k} LA)")
               for n in r.notes) == (scheme == "amber" and end == "5")
    # the model check bonds the junction as before the clean-up: no clash, both LINK bonds
    assert link_bonds(r.text) == link_bonds(text) == (junction(end), [], 0)
    stale = "\n".join(link_lines(text) + [ln for ln in r.text.splitlines() if not ln.startswith("LINK")])
    if changed:  # the records as they were name residues that are no longer there: the junction clashes
        assert link_bonds(stale)[1] and link_bonds(stale)[2] == len(changed)


@pytest.mark.parametrize("end", ["3", "5"])
def test_link_records_follow_numbers_and_chain_ids(end):
    """Renumbered, split, combined, reordered, chain ids restored: the records name the residues they named in
    the input, and the model check still bonds the junction."""
    k, po2, oname = (3, 4, "O3'") if end == "3" else (1, 0, "O5'")
    text = bridged(end)

    def ends(result):
        """(chain, number, name) of both ends of each record, as written."""
        recs = pt.link_records(pt.read_pdb(result.text))
        assert recs == pt.link_records(result.structure)
        return [((a.chain, a.resseq, a.name), (b.chain, b.resseq, b.name)) for a, b, *_ in recs]

    def bonded(result):
        """The model check finds both LINK bonds and no clash."""
        bonds, clashes, unused = link_bonds(result.text)
        return len(bonds) == 2 and clashes == [] and unused == 0

    # renumbered on through both chains (A:1-4, B:5-7 / A:1-4 PO2 first, B:5-7), or per chain
    r = pt.clean(text, renumber=True, start=10)
    a_end, b_end, bridge = (12, 16, 13) if end == "3" else (11, 14, 10)
    assert ends(r) == [(("A", a_end, oname), ("A", bridge, "P")), (("B", b_end, oname), ("A", bridge, "P"))]
    assert bonded(r)
    r = pt.clean(text, renumber=True, per_chain=True)
    assert ends(r)[1][0] == ("B", k, oname) and bonded(r)
    # split before the bridge (3'-3': the PO2 is chain C) or after it (5'-5': strand A is chain C)
    r = pt.clean(text, split_at="A:4" if end == "3" else "A:1")
    if end == "3":
        assert ends(r) == [(("A", 3, oname), ("C", 4, "P")), (("B", 3, oname), ("C", 4, "P"))]
    else:
        assert ends(r) == [(("C", 1, oname), ("A", 0, "P")), (("B", 1, oname), ("A", 0, "P"))]
    assert bonded(r) and any(n.startswith(f"Rewrote {2 if end == '3' else 1} LINK record(s)") for n in r.notes)
    # combined into one chain: renumbered, the records follow the residues they named, not the first A:3
    r = pt.clean(text, combine=["A", "B"], combine_into="Z", renumber=True)
    assert [x[0] for x in ends(r)] == [("Z", a_end - 9, oname), ("Z", b_end - 9, oname)] and bonded(r)
    r = pt.clean(text, combine=["A", "B"])  # not renumbered: two residues A:3 (A:1)
    assert any(n.startswith("WARNING: chain A now has duplicate residue numbers") for n in r.notes)
    assert any(n.startswith("WARNING: 1 LINK record end(s) now name two residues") and f"A:{k} A {oname}" in n
               for n in r.notes)
    # chains in another order: the records stay as they were, no note
    r = pt.clean(text, sort_chains=["B", "A"])
    assert r.structure.chain_ids() == ["B", "A"] and link_lines(r.text) == link_lines(text) and bonded(r)
    assert not any("LINK" in n for n in r.notes)
    # cpptraj-style input (no chain ids, numbered on), its chain ids restored from a list, as A B C ... or from a
    # reference; residues edited by hand follow inside follow_links
    s = pt.clean(text, renumber=True, start=10).structure
    with pt.follow_links(s) as notes:
        for res in s.residues():
            res.chain = ""
    assert notes == [f"Rewrote 2 LINK record(s) for the new residue names, numbers or chain ids: _:{a_end} A "
                     f"{oname} - _:{bridge} PO2 P, _:{b_end} A {oname} - _:{bridge} PO2 P"]
    cpptraj = pt.write_pdb(s)
    assert link_bonds(cpptraj)[0] == [(f"_:{a_end}:{oname}", f"_:{bridge}:P"), (f"_:{b_end}:{oname}", f"_:{bridge}:P")]
    for options in ({"chain_ids": ["A", "B"]}, {"assign_chains": True}, {"chain_reference": text}):
        r = pt.clean(cpptraj, **options)
        assert [res.chain for res in r.structure.residues()] == ["A"] * 4 + ["B"] * 3
        assert {x[1][0] for x in ends(r)} == {"A"} and [x[0][0] for x in ends(r)] == ["A", "B"]
        assert bonded(r)


@pytest.mark.parametrize("end", ["3", "5"])
def test_link_records_through_hydrogen_and_phosphate_removal(end):
    """A clean-up for tleap (hydrogens and 5'-phosphates removed, amber names, renumbered): the records follow,
    the strands' own 5'-phosphates go (3'-3'), and the 5'-5' bridge, no 5' end, stays.  The residues bonded
    through the bridge keep the internal unit A (no 3'-OH / 5'-OH unit that tleap would give a hydroxyl hydrogen
    on the bonded oxygen).  The model check passes the result with both LINK bonds.  A record that names a
    removed atom (a hydrogen) is dropped."""
    k, po2, oname = (3, 4, "O3'") if end == "3" else (1, 0, "O5'")
    text = bridged(end, hydrogens=True, phosphate5=True)
    lines = text.splitlines()
    lines.insert(2, pt.link_line(pt.LinkAtom("H1'", "A", "A", 2), pt.LinkAtom("OP1", "PO2", "A", po2)))
    r = pt.clean("\n".join(lines) + "\n", scheme="amber", strip_h=True, strip_5p=True, renumber=True, start=5)
    a_end, b_end, bridge = (7, 11, 8) if end == "3" else (6, 9, 5)
    assert link_lines(r.text) == [pt.link_line(pt.LinkAtom(oname, "A", chain, n), pt.LinkAtom("P", "PO2", "A", bridge),
                                               1.6) for chain, n in (("A", a_end), ("B", b_end))]
    free = "A5" if end == "3" else "A3"  # the strands' other, free ends
    assert [r_.name for seg in r.structure.segments for r_ in seg if r_.name != "PO2"] == (
        [free, "A", "A"] * 2 if end == "3" else ["A", "A", free] * 2)
    assert f"Dropped 1 LINK record(s) whose residue or atom was removed: A:2 A H1' - A:{po2} PO2 OP1" in r.notes
    if end == "3":
        assert "Removed the 5'-terminal phosphate of 2 residue(s): A:1 A, B:1 A" in r.notes
    else:
        assert ("Phosphates kept where LINK records bond them to another residue (a link such as the 5'-5' "
                "phosphate of a chimera, not a 5' end): A:0 PO2") in r.notes
        assert not any(n.startswith("Removed") and "phosphate" in n for n in r.notes)
    assert [a.name for a in next(res for res in r.structure.residues() if res.name == "PO2").atoms] == [
        "P", "OP1", "OP2"]
    assert link_bonds(r.text) == ([(f"A:{a_end}:{oname}", f"A:{bridge}:P"), (f"B:{b_end}:{oname}", f"A:{bridge}:P")],
                                  [], 0)
    report = mc.check_model(pt.read_pdb(r.text), mc.QCSettings(max_clashes=0), path="amber.pdb")
    assert (report.status, report.failed, report.warnings) == ("pass", [], [])


def test_a_phosphate_residue_is_a_5prime_phosphate_unless_bonded_to_another_strand():
    """strip_5p removes a 5'-phosphate modelled as a residue of its own whose LINK record bonds it to its
    strand's first O5' (the record goes with it); a phosphate that a record bonds to another strand is a link -
    here a strand's own 5'-phosphate bonded to the O3' of the strand before it - and stays, and the two residues
    it joins are no strand ends: internal names (C, A, not C3, A5), no 'remove the 5'-phosphate' warning."""
    s = pt.read_pdb(bridged("3", phosphate5=True))
    first = s.segments[0][0]
    cap = pt.Residue("PHO", "A", 0, atoms=[a for a in first.atoms if a.name in ("P", "OP1", "OP2", "OP3")])
    first.atoms = [a for a in first.atoms if a not in cap.atoms]
    s.segments[0].insert(0, cap)
    s.header.append(pt.link_line(pt.LinkAtom("O5'", "A", "A", 1), pt.LinkAtom("P", "PHO", "A", 0)))
    r = pt.clean(pt.write_pdb(s), strip_5p=True)
    assert "Removed 1 5'-terminal residue(s) that held only a phosphate: A:0 PHO" in r.notes
    assert "Dropped 1 LINK record(s) whose residue or atom was removed: A:1 A O5' - A:0 PHO P" in r.notes
    assert "Removed the 5'-terminal phosphate of 1 residue(s): B:1 A" in r.notes
    assert link_lines(r.text) == link_lines(bridged("3"))
    # one strand of six split in two, its link A:3 O3' - B:4 P given as a LINK record: B:4 keeps its phosphate
    s = pt.read_pdb(builder.build_helix("GGCAUC", partner="none", phosphate5=True))
    pt.split_chains(s, "A:4")
    s.header.append(pt.link_line(pt.LinkAtom("O3'", "C", "A", 3), pt.LinkAtom("P", "A", "B", 4), 1.6))
    r = pt.clean(pt.write_pdb(s), strip_5p=True, scheme="amber")
    assert "Removed the 5'-terminal phosphate of 1 residue(s): A:1 G" in r.notes
    assert any(n.startswith("Phosphates kept where LINK records bond them") and n.endswith(": B:4 A") for n in r.notes)
    assert [a.name for a in r.structure.segments[1][0].atoms][:3] == ["P", "OP1", "OP2"]
    assert link_bonds(r.text) == ([("A:3:O3'", "B:4:P")], [], 0)
    assert [res.name for res in r.structure.residues()] == ["G5", "G", "C", "A", "U", "C3"]
    assert link_lines(r.text) == [pt.link_line(pt.LinkAtom("O3'", "C", "A", 3), pt.LinkAtom("P", "A", "B", 4), 1.6)]
    assert ("Segment ends that LINK records bond to another residue (a junction such as a chimera's 3'-3' / 5'-5' "
            "phosphate, or a strand bonded on) have internal names: A:3 C, B:4 A") in r.notes
    assert not any(n.startswith("WARNING") for n in r.notes)  # B:4's phosphate is the link's, as strip_5p said


def test_a_metal_on_a_5prime_phosphate_oxygen_does_not_keep_the_phosphate():
    """LINK records of deposited structures give metal coordination: an Mg2+ on the OP1 of a 5'-phosphate is no
    link of the strand, so strip_5p removes the phosphate as without the record, which goes with its oxygen."""
    s = pt.read_pdb(builder.build_helix("GGC", partner="none", phosphate5=True))
    op1 = next(a for a in s.segments[0][0].atoms if a.name == "OP1")
    mg = pt.Residue("MG", "M", 1, atoms=[pt.Atom("HETATM", "MG", "", op1.tail, "MG")])
    _place(mg.atoms[0], [op1.xyz[0] + 2.1, op1.xyz[1], op1.xyz[2]])
    s.segments.append([mg])
    without = pt.write_pdb(s)
    s.header.append(pt.link_line(pt.LinkAtom("OP1", "G", "A", 1), pt.LinkAtom("MG", "MG", "M", 1), 2.1))
    r, plain = pt.clean(pt.write_pdb(s), strip_5p=True), pt.clean(without, strip_5p=True)
    assert "Removed the 5'-terminal phosphate of 1 residue(s): A:1 G" in plain.notes
    assert r.text == plain.text
    assert r.notes == plain.notes + [
        "Dropped 1 LINK record(s) whose residue or atom was removed: A:1 G OP1 - M:1 MG MG"]


def test_link_records_that_name_no_atom_stay_as_they_were():
    """A record that names no atom of the model (another chain, residue name or atom) cannot follow: it stays as
    it was, with a note, and so does a LINK line that is no record; the others follow."""
    text = bridged("3")
    odd = [pt.link_line(pt.LinkAtom("O3'", "A", "Z", 9), pt.LinkAtom("P", "PO2", "A", 4)),
           pt.link_line(pt.LinkAtom("O2'", "G", "A", 3), pt.LinkAtom("P", "PO2", "A", 4)),
           pt.link_line(pt.LinkAtom("N9", "A", "A", 3), pt.LinkAtom("XX", "PO2", "A", 4)),
           "LINK        unreadable"]
    links, rest = link_lines(text), [ln for ln in text.splitlines() if not ln.startswith("LINK")]
    r = pt.clean("\n".join(links[:1] + odd + links[1:] + rest) + "\n", scheme="amber", renumber=True, start=2)
    assert link_lines(r.text)[1:5] == odd
    assert [ln[17:26] for ln in link_lines(r.text)[::5]] == ["  A A   4", "  A B   8"]  # (junction: internal A)
    assert ("Kept 3 LINK record(s) as they were: they name no atom of the model (by chain, residue number, insertion "
            "code, residue name and atom name), so they cannot follow the residues: Z:9 A O3' - A:4 PO2 P, A:3 G O2' "
            "- A:4 PO2 P, A:3 A N9 - A:4 PO2 XX") in r.notes
    assert link_bonds(r.text)[1:] == ([], 3)


def test_link_records_after_the_coordinates_are_read_and_follow():
    """read_pdb keeps a LINK line that comes after the first atom record in the trailer: link_records reads it
    after the header's, the model check bonds it, and the PDB tools rewrite it where it is."""
    text = bridged("3")
    links = link_lines(text)
    body = [ln for ln in text.splitlines() if not ln.startswith(("LINK", "REMARK"))]
    moved = "\n".join(links[:1] + body[:-1] + links[1:] + ["END"]) + "\n"  # the second record after the atoms
    s = pt.read_pdb(moved)
    assert (s.header, s.trailer) == (links[:1], links[1:])
    assert pt.link_records(s) == pt.link_records(pt.read_pdb(text))
    assert link_bonds(moved) == (junction("3"), [], 0)
    r = pt.clean(moved, scheme="amber", renumber=True, start=3)
    out = r.text.splitlines()
    assert out[0][17:26] == "  A A   5" and out[-2][17:26] == "  A B   9" and out[-1] == "END"
    assert r.structure.trailer == [out[-2]] and link_bonds(r.text)[1:] == ([], 0)


def test_link_lines_are_rewritten_in_place():
    """Only the residue columns of an end whose residue changed are written: the atom names, alternate
    locations, symmetry operators and the length stay, and so does the line's place among the header records.
    A record to a symmetry mate (operator 2555) follows the names too, but bonds nothing of the model: B:3 is a
    free 3' end (LA3), A:3, bonded through the other record, the internal LA."""
    text = bridged("3")
    plain = link_lines(text)
    odd = pt.link_line(pt.LinkAtom("O3'", "A", "B", 3), pt.LinkAtom("P", "PO2", "A", 4), 2.1)
    odd = odd[:16] + "A" + odd[17:46] + "B" + odd[47:66] + "  2555" + odd[72:]  # altLocs, symmetry 2555
    body = [ln for ln in text.splitlines() if not ln.startswith(("LINK", "REMARK"))]
    head = ["HEADER    CHIMERA", plain[0], "REMARK   1 two LINK records", odd, "CRYST1   60.000   60.000   60.000"]
    r = pt.clean("\n".join(head + body) + "\n", scheme="amber", l_ranges="A,B")
    assert r.structure.header == [head[0], plain[0][:17] + " LA A   3 " + plain[0][27:], head[2],
                                  odd[:17] + "LA3 B   3 " + odd[27:], head[4]]
    mate = pt.link_records(r.structure)[1]
    assert mate == pt.LinkRecord(pt.LinkAtom("O3'", "LA3", "B", 3), pt.LinkAtom("P", "PO2", "A", 4), 2.1, "", "2555")
    assert mate.mate and not pt.link_records(r.structure)[0].mate
    assert r.text.splitlines()[:5] == r.structure.header
    unchanged = pt.clean("\n".join(head + body) + "\n")  # nothing renamed: every line as it was
    assert unchanged.structure.header == head and not any("LINK" in n for n in unchanged.notes)
    # an insertion code (column 27) is part of the number: written, cleared by renumbering, kept with the gaps
    s = pt.read_pdb(text)
    with pt.follow_links(s):
        next(res for res in s.residues() if (res.chain, res.resseq) == ("B", 3)).icode = "A"
    coded = pt.write_pdb(s)
    assert link_lines(coded)[1][21:27] == "B   3A" and link_bonds(coded)[0][1] == ("B:3A:O3'", "A:4:P")
    assert link_lines(pt.clean(coded, renumber=True).text)[1][21:27] == "B   7 "
    assert link_lines(pt.clean(coded, renumber=True, keep_gaps=True).text)[1][21:27] == "B   7A"


def test_operations_on_their_own_and_in_one_block():
    """Each operation that edits residues follows the records by itself (the structure builder names residues
    with convert_names).  A follow_links block resolves them once for several operations, which matters when one
    leaves two residues with the same id for a later one to tell apart; a block that raises leaves nothing."""
    s = pt.read_pdb(bridged("3"))
    pt.assign_chirality(s, l_ranges="A,B")
    assert pt.convert_names(s, "amber")[-1] == ("Rewrote 2 LINK record(s) for the new residue names, numbers or chain "
                                               "ids: A:3 LA O3' - A:4 PO2 P, B:3 LA O3' - A:4 PO2 P")
    assert pt.renumber_residues(s, 10)[-1].startswith("Rewrote 2 LINK record(s)")
    assert [(x.first.resseq, x.second.resseq) for x in pt.link_records(s)] == [(12, 13), (16, 13)]
    assert link_bonds(pt.write_pdb(s))[1:] == ([], 0)
    # in turn: combining leaves two residues A:3, and renumbering then takes the record of B:3 for the first
    s = pt.read_pdb(bridged("3"))
    assert pt.combine_chains(s, ["A", "B"])[-1].startswith("WARNING: 1 LINK record end(s) now name two residues")
    pt.renumber_residues(s)
    assert [x.first.resseq for x in pt.link_records(s)] == [3, 3]
    # in one block: resolved before combining, so B:3 is followed to A:7
    s = pt.read_pdb(bridged("3"))
    with pt.follow_links(s) as notes:
        assert not any("LINK" in n for n in pt.combine_chains(s, ["A", "B"]) + pt.renumber_residues(s))
        assert notes == []
    assert [x.first.resseq for x in pt.link_records(s)] == [3, 7]
    assert notes == ["Rewrote 1 LINK record(s) for the new residue names, numbers or chain ids: A:7 A O3' - A:4 PO2 P"]
    with pytest.raises(pt.PDBError):
        pt.clean(bridged("3"), split_at="A:9")
    assert pt._FOLLOWERS == {}


def test_to_lrna_and_the_command_line_follow_link_records(tmp_path):
    """to-lrna, 'mdatk pdb clean' and the copy of the PDB tools in generated scripts (standard library only, no
    __future__ import) write the records for the new names; 'mdatk pdb qc' passes the cleaned chimera."""
    import subprocess
    import sys

    from mdatk.cli import main
    from mdatk.workflows.scripts import embed

    r = pt.to_lrna(bridged("3"), "B")
    assert [x.first.resname for x in pt.link_records(r.structure)] == ["A", "LA"]  # the junction: internal
    assert link_bonds(r.text) == (junction("3"), [], 0)
    pdb, out, copy = tmp_path / "chimera.pdb", tmp_path / "amber.pdb", tmp_path / "copy.pdb"
    pdb.write_text(bridged("5", phosphate5=True))
    args = ["--scheme", "amber", "--strip-5p", "--renumber"]
    assert main(["pdb", "clean", str(pdb), "-o", str(out)] + args) == 0
    assert [x.first.resname for x in pt.link_records(pt.read_pdb(str(out)))] == ["A", "A"]
    assert main(["pdb", "qc", str(out), "--max-clashes", "0"]) == 0
    script = tmp_path / "pdbtools_copy.py"
    script.write_text(embed(pt, {}, "main()", "PDB hygiene (embedded copy)"))
    proc = subprocess.run([sys.executable, "-I", str(script), "clean", str(pdb), "-o", str(copy)] + args,
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert copy.read_text() == out.read_text() and "pdbtools: Rewrote 2 LINK record(s)" in proc.stderr


# --- LINK-bonded junctions, symmetry mates, sequences, SEQRES names, CCD atom names -----------------------

def _names(structure):
    return [(res.chain, res.resseq, res.name) for res in structure.residues()]


@pytest.mark.parametrize("design", ["D:GC 3'-3' L:GC", "D:GC 5'-5' L:GC", "D:G 3'-3' L:GC"])
@pytest.mark.parametrize("scheme", ["amber", "ccd"])
def test_clean_names_a_chimera_as_the_structure_builder_does(design, scheme):
    """The residues a chimera's PO2 bridge bonds (O3' at 3'-3', O5' at 5'-5') are no strand ends: cleaning the
    builder's default names into a scheme gives the names the builder writes in that scheme (internal names at
    the junction, terminal variants at the free ends, a single residue bonded at its 3' end its 5' unit), and the
    LINK records name them; a clean of the scheme's own build changes nothing.  Before, clean gave the bonded
    residues 3' / 5' units (amber C3 / LC3: tleap adds HO3' to an O3' bonded to the PO2)."""
    from mdatk import chimera

    built_text = chimera.build_chimera(design, scheme=scheme)
    built = pt.read_pdb(built_text)
    r = pt.clean(chimera.build_chimera(design), scheme=scheme)
    assert _names(r.structure) == _names(built)
    assert pt.link_records(r.structure) == pt.link_records(built) and len(pt.link_records(built)) == 2
    again = pt.clean(built_text, scheme=scheme)
    assert f"Residue names already follow the {scheme} scheme" in again.notes
    assert link_lines(again.text) == link_lines(built_text) and not any("Rewrote" in n for n in again.notes)
    bonded_o5 = [n for n in r.notes if n.startswith("WARNING: the residues bonded through O5' by LINK records")]
    assert len(bonded_o5) == (1 if scheme == "amber" and "5'-5'" in design else 0)
    assert link_bonds(r.text)[1:] == ([], 0)


def test_a_strand_bonded_on_and_a_lone_phosphate_cap_at_the_segment_ends():
    """The 5' end of a strand whose P a LINK record bonds to the O3' of another is no 5' end (internal name, no
    5'-phosphate warning: it is the link's phosphate); a 5' end whose own phosphate is modelled as a residue of
    its own (a cap bonded to nothing else) stays a 5' end; water and ions bonded to an end do not count."""
    s = pt.read_pdb(builder.build_helix("GGCAUC", partner="none", phosphate5=True))
    pt.split_chains(s, "A:4")
    s.header.append(pt.link_line(pt.LinkAtom("O3'", "C", "A", 3), pt.LinkAtom("P", "A", "B", 4), 1.6))
    r = pt.clean(pt.write_pdb(s), scheme="amber")
    assert [res.name for res in r.structure.residues()] == ["G5", "G", "C", "A", "U", "C3"]
    assert [n for n in r.notes if n.startswith("WARNING: 5'-terminal")] == [
        "WARNING: 5'-terminal residues carry P/OP1/OP2, which the Amber 5' units do not have (tleap cannot type them; "
        "remove them with --strip-5p or cap the end): A:1 G5"]
    # the cap: the P of a phosphate residue bonded to the first O5'; a metal on the last O3'
    s = pt.read_pdb(builder.build_helix("GGC", partner="none", phosphate5=True, chirality="L"))
    first = s.segments[0][0]
    cap = pt.Residue("PHO", "A", 0, atoms=[a for a in first.atoms if a.name in ("P", "OP1", "OP2", "OP3")])
    first.atoms = [a for a in first.atoms if a not in cap.atoms]
    s.segments[0].insert(0, cap)
    o3 = next(a for a in s.segments[0][-1].atoms if a.name == "O3'")
    s.segments.append([pt.Residue("MG", "M", 1, atoms=[pt.Atom("HETATM", "MG", "", o3.tail, "MG")])])
    s.header += [pt.link_line(pt.LinkAtom("O5'", "0G", "A", 1), pt.LinkAtom("P", "PHO", "A", 0)),
                 pt.link_line(pt.LinkAtom("O3'", "0C", "A", 3), pt.LinkAtom("MG", "MG", "M", 1))]
    for scheme, names in (("amber", ["LG5", "LG", "LC3"]), ("ccd", ["0G5", "0G", "0C3"])):
        r = pt.clean(pt.write_pdb(s), scheme=scheme)
        assert [res.name for res in r.structure.residues() if res.name not in ("PHO", "MG")] == names
        assert not any(n.startswith("Segment ends that LINK records bond") for n in r.notes)


def test_link_records_with_symmetry_operators():
    """The symmetry operators of a LINK record (columns 60-65 / 67-72) are read, '' for the identity (1555 or
    blank), so a record of the model equals LinkRecord(first, second, length); one whose operators differ bonds an
    atom to a symmetry mate (``mate``)."""
    a, b = pt.LinkAtom("N", "NH4", "A", 102), pt.LinkAtom("O3'", "A", "A", 11)
    line = pt.link_line(a, b, 2.69)
    mate = line[:59] + "  1555   7645" + line[72:]
    same = line[:59] + "  2555   2555" + line[72:]
    records = pt.link_records(pt.Structure(header=[line, mate, same, pt.link_line(a, b)]))
    assert records == [pt.LinkRecord(a, b, 2.69), pt.LinkRecord(a, b, 2.69, "", "7645"),
                       pt.LinkRecord(a, b, 2.69, "2555", "2555"), pt.LinkRecord(a, b)]
    assert [r.mate for r in records] == [False, True, False, False]


def test_sequences_leave_out_residues_made_only_of_phosphate_atoms():
    """FASTA, SEQRES and 'pdb check' list the polymer residues: a chimera's PO2 bridge (bonded through LINK
    records) and a lone 5'-phosphate modelled as a residue are none (the check lists them among the other
    residues); a modified nucleotide, which has a sugar, stays (X)."""
    text = bridged("3")
    r = pt.clean(text, seqres=True, make_fasta=True, title="c")
    assert [ln for ln in r.text.splitlines() if ln.startswith("SEQRES")] == [
        "SEQRES   1 A    3    A   A   A", "SEQRES   1 B    3    A   A   A"]
    assert r.fasta == ">c_A chain A | RNA | 3 residues\nAAA\n>c_B chain B | RNA | 3 residues\nAAA\n"
    info = pt.summary(pt.read_pdb(text))
    assert (info["chains"]["A"]["residues"], info["chains"]["A"]["numbers"], info["chains"]["A"]["sequence"],
            info["chains"]["A"]["names"]) == (3, "1-3", "AAA", ["A"])
    assert info["other_residues"] == {"PO2": 1}
    s = pt.read_pdb(builder.build_helix("GGCAUC", partner="none", phosphate5=True))
    first = s.segments[0][0]
    cap = pt.Residue("PHO", "A", 0, atoms=[a for a in first.atoms if a.name in ("P", "OP1", "OP2", "OP3")])
    first.atoms = [a for a in first.atoms if a not in cap.atoms]
    s.segments[0].insert(0, cap)
    s.segments[0][3].name = "PSU"  # A:3 (a pseudouridine, as far as names go: it has its sugar)
    r = pt.clean(pt.write_pdb(s), seqres=True, make_fasta=True, title="m")
    assert [ln for ln in r.text.splitlines() if ln.startswith("SEQRES")] == [
        "SEQRES   1 A    6    G   G PSU   A   U   C"]
    assert r.fasta == ">m_A chain A | RNA | 6 residues\nGGXAUC\n"
    assert pt.summary(pt.read_pdb(r.text))["other_residues"] == {"PHO": 1, "PSU": 1}


def test_seqres_names_have_three_characters():
    """SEQRES has three columns per residue name: MDATK's four-character Amber-style L-DNA names (LDG5, LDA3,
    LDTN ...) are written as their CCD components (0DG ...), any other long name as UNK, each with a note; the
    names of the residues themselves do not change."""
    text = builder.build_helix("GACTGACTGACTGA", na="dna", chirality="L", scheme="amber")
    r = pt.clean(text, seqres=True)
    lines = [ln for ln in r.text.splitlines() if ln.startswith("SEQRES")]
    assert lines[0] == "SEQRES   1 A   14  0DG LDA LDC LDT LDG LDA LDC LDT LDG LDA LDC LDT LDG"
    assert lines[1] == "SEQRES   2 A   14  0DA"
    assert lines[2].endswith("  0DT LDC LDA LDG LDT LDC LDA LDG LDT LDC LDA LDG LDT") and lines[3].endswith("  0DC")
    for ln in lines:  # every name in its columns 20-22, 24-26 ...
        names = ln[19:].split()
        assert [ln[19 + 4 * k:22 + 4 * k].strip() for k in range(len(names))] == names
    assert ("SEQRES names have three characters: wrote the CCD components of the four-character names LDA3->0DA x1, "
            "LDC3->0DC x1, LDG5->0DG x1, LDT5->0DT x1") in r.notes
    assert {res.name for res in r.structure.residues()} >= {"LDG5", "LDA3", "LDT5", "LDC3"}
    s = pt.read_pdb(text)
    s.segments[0][1].name = "LIGX"
    r = pt.clean(pt.write_pdb(s), seqres=True)
    assert [ln for ln in r.text.splitlines() if ln.startswith("SEQRES")][0][19:30] == "0DG UNK LDC"
    assert "WARNING: SEQRES names have three characters: wrote UNK for the residue name(s) of A:2 LIGX" in r.notes


# The atoms of the CCD's L-deoxyribonucleotides (wwPDB CCD, as ParmEd's CCD templates list them; the CCP4
# monomer library has the same names): 0DA spells its adenine and the hydrogen of a protonated OP2 its own way,
# 0DT its methyl carbon
CCD_COMPONENT_ATOMS = {
    "0DA": "C8A N9A C4A C5A N7A N3A C2A N1A C6A N6A C4' O4' C3' C2' C1' O3' C5' O5' P OP2 OP3 OP1 H8A H2A HN61 HN62 "
           "H4' H3' H2' H2'' H1' HO3' H5' H5'' H2P HOP3",
    "0DC": "OP3 P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' C1' N1 C2 O2 N3 C4 N4 C5 C6 HOP3 HOP2 H5' H5'' H4' H3' HO3' "
           "H2' H2'' H1' H41 H42 H5 H6",
    "0DG": "OP3 P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' C1' N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4 HOP3 HOP2 H5' H5'' H4' "
           "H3' HO3' H2' H2'' H1' H8 H1 H21 H22",
    "0DT": "OP3 P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' C1' N1 C2 O2 N3 C4 O4 C5 C5M C6 HOP3 HOP2 H5' H5'' H4' H3' "
           "HO3' H2' H2'' H1' H3 H71 H72 H73 H6",
}


def file_atoms(text):
    """(residue name, atom name) of every ATOM / HETATM line, as written (columns 18-21, 13-16)."""
    return [(ln[17:21].strip(), ln[12:16].strip()) for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]


def test_the_ccd_component_atom_names_are_what_the_tables_say():
    """CCD_ATOM_NAMES maps the standard names onto the components' own: with it, the standard atoms of each L-DNA
    base (and the 5'-phosphate hydrogens of 0DA) are atoms of its component, and they map back."""
    standard = {"A": "N9 C8 N7 C5 C6 N6 N1 C2 N3 C4 H8 H61 H62 H2 HOP2 HOP3",
                "C": "N1 C2 O2 N3 C4 N4 C5 C6 H41 H42 H5 H6", "G": "N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4 H8 H1 H21 H22",
                "T": "N1 C2 O2 N3 C4 O4 C5 C7 C6 H3 H71 H72 H73 H6"}
    for component, atoms in CCD_COMPONENT_ATOMS.items():
        for name in standard[component[-1]].split():
            own = pt.component_atom_name(component, name)
            assert own in atoms.split(), (component, name)
            assert pt.standard_atom_name(component, own) == name
    assert set(pt.CCD_ATOM_NAMES) == {"0DA", "0DT"}
    assert pt.component_atom_name("LDA", "N9") == "N9" == pt.standard_atom_name("LDA", "N9")


def test_0da_and_0dt_are_read_and_written_with_their_components_atom_names():
    """read_pdb gives the atoms of a 0DA / 0DT their standard names (every reader of a structure finds N9, C7 ...),
    write_pdb writes the components' names for every residue named 0DA / 0DT and the standard names for every
    other: a file round-trips byte for byte, a rename to LDA (amber) or DA (standard) writes the standard names,
    and a 0DA / 0DT with standard names in the file is written with the component's, with a note."""
    notes = []
    text = builder.build_helix("GACT", na="dna", chirality="L", hydrogens=True, notes=notes)
    assert {("0DA", "N9A"), ("0DA", "HN61"), ("0DT", "C5M")} <= set(file_atoms(text))
    said = "0DA / 0DT are written with their CCD components' own atom names (0DA: N9A, C8A ... HN61, HN62; 0DT: C5M)"
    assert any(n.startswith(said) for n in notes)
    amber_notes = []
    builder.build_helix("GACT", na="dna", chirality="L", scheme="amber", notes=amber_notes)
    assert not any(n.startswith(said) for n in amber_notes)
    s = pt.read_pdb(text)
    names = {res.name: [a.name for a in res.atoms] for res in s.residues()}
    assert "N9" in names["0DA"] and "H61" in names["0DA"] and "C7" in names["0DT"] and "N9A" not in names["0DA"]
    assert next(r for r in s.residues() if r.name == "0DA").coords().keys() >= {"N9", "C8", "N6", "C1'"}
    assert pt.write_pdb(s) == text and s.notes == []
    for scheme, residue in (("amber", "LDA"), ("standard", "DA")):
        s = pt.read_pdb(text)
        pt.assign_chirality(s)
        pt.convert_names(s, scheme, dna=True)
        atoms = [a for r, a in file_atoms(pt.write_pdb(s)) if r == residue]
        assert "N9" in atoms and "N9A" not in atoms and "HN61" not in atoms, scheme
    old = "\n".join(ln[:12] + {" N9A": " N9 ", " C5M": " C7 "}.get(ln[12:16], ln[12:16]) + ln[16:]
                    if ln.startswith("ATOM") and ln[21] == "A" else ln for ln in text.splitlines()) + "\n"
    s = pt.read_pdb(old)  # strand A's 0DA and 0DT with two standard names
    assert s.notes == ["2 0DA / 0DT residue(s) use standard atom names where their CCD components have their own "
                       "(0DA: N9A, C8A ... HN61, HN62; 0DT: C5M): read as the same atoms, written with the components' "
                       "names: A:2 0DA, A:4 0DT"]
    assert pt.write_pdb(s) == text


@pytest.mark.parametrize("scheme", [None, "ccd", "pdb", "rosetta"])
def test_every_atom_of_an_l_dna_build_is_an_atom_of_its_ccd_component(scheme):
    """The L-DNA residues of the structure builder named after the CCD components (0DA 0DC 0DG 0DT: the ccd, pdb,
    rosetta and default names) carry the components' atom names, hydrogens and 5'-phosphates included (terminal
    HO5' excepted: the components have a 5'-phosphate), so Coot / Phenix find every atom in their dictionaries;
    so do the residues a mutation gives these names and the L-DNA segment of a chimera."""
    from mdatk import chimera

    texts = [builder.build_helix("GACTAG", na="dna", chirality="L", scheme=scheme, hydrogens=h, phosphate5=p)
             for h in (False, True) for p in (False, True)]
    texts.append(builder.mutate(texts[0], "A:1A, A:3T")[0])  # 0DG -> 0DA, 0DC -> 0DT and their partners
    texts.append(chimera.build_chimera("D-DNA:GA 3'-5' L-DNA:GACT", scheme=scheme, hydrogens=True))
    for text in texts:
        atoms = file_atoms(text)
        assert {r for r, _a in atoms if r in CCD_COMPONENT_ATOMS} == set(CCD_COMPONENT_ATOMS)
        foreign = sorted({(r, a) for r, a in atoms
                          if r in CCD_COMPONENT_ATOMS and a not in CCD_COMPONENT_ATOMS[r].split() and a != "HO5'"})
        assert foreign == []
    # amber (LDA ...) and standard (DA ...) names: the standard atom names
    for other in ("amber", "standard"):
        atoms = {a for _r, a in file_atoms(builder.build_helix("GACT", na="dna", chirality="L", scheme=other))}
        assert {"N9", "C7"} <= atoms and not atoms & {"N9A", "C5M"}


def test_analyses_of_0da_files_and_of_the_same_model_named_lda_agree():
    """The model check (clashes, links, hands, kinks, the hydrogen contacts and their atom names), base-pair
    detection (annotate's builtin engine, the builder's partners, the interfaces' engine), the sugar hands and a
    mutation give the same results on a 0DA / 0DT model (CCD atom names) as on the same model named LDA ..."""
    import re

    from mdatk import annotate, interfaces

    ccd, amber = (builder.build_helix("GACTAGTC", na="dna", chirality="L", scheme=scheme, hydrogens=True)
                  for scheme in ("ccd", "amber"))
    assert ("0DA", "N1A") in file_atoms(ccd) and ("LDA", "N1") in file_atoms(amber)
    st = mc.QCSettings(hydrogen_check=True)
    reports = [mc.check_model(t, st, path="m") for t in (ccd, amber)]
    assert mc.report_data(reports[0]) == mc.report_data(reports[1]) and reports[0].status == "pass"
    assert mc.text_report(reports[0], st) == mc.text_report(reports[1], st)

    def pairs(text):  # (chain, number) of both residues and the class of every PAIR record, the names left out
        out = []
        for ln in annotate.annotate(text, engine="builtin").records:
            m = re.match(r"REMARK 950 PAIR  (.)(.{4}).\S+\s+(.)(.{4}).\S+\s+(\S+)$", ln)
            if m:
                out.append((m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), m.group(5)))
        return out

    assert pairs(ccd) == pairs(amber) and len(pairs(ccd)) == 8
    partners = [sorted(builder._partners(pt.read_pdb(t)).values()) for t in (ccd, amber)]
    assert partners[0] == partners[1] and len(partners[0]) == 16
    found = [interfaces.engine_pairs(interfaces.nucleotides(pt.read_pdb(t)), "builtin")[0] for t in (ccd, amber)]
    assert found[0] == found[1] and len(found[0]) == 8
    hands = [[(h.call, h.c4, h.volumes) for h in pt.residue_hands(pt.read_pdb(t))] for t in (ccd, amber)]
    assert hands[0] == hands[1] and {h[0] for h in hands[0]} == {"L"}
    mutated = [builder.mutate(t, "A:2G")[0] for t in (ccd, amber)]
    assert [ln[30:54] for ln in mutated[0].splitlines() if ln.startswith("ATOM")] == [
        ln[30:54] for ln in mutated[1].splitlines() if ln.startswith("ATOM")]
