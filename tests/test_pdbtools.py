"""Tests for mdatk.pdbtools (PDB hygiene: L-RNA detection, name schemes, ChimeraX repair, chains ...)."""

import math

import pytest

from mdatk import pdbtools as pt

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
