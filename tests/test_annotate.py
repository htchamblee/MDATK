"""Tests for mdatk.annotate (REMARK 950 base-pair annotation) and 'mdatk pdb annotate'."""

import sys

import numpy as np
import pytest

from test_basepairs import fake_barnaba, moved, nucleotide, partner
from test_pdbtools import atom_line
from mdatk.annotate import annotate
from mdatk.pdbtools import read_pdb


def pdb_text(chains, header=()):
    """chains: [(chain id, [(resname, resseq, {atom: xyz}), ...]), ...] -> PDB text."""
    lines, serial = list(header), 0
    for chain, residues in chains:
        for resname, resseq, atoms in residues:
            for name, xyz in atoms.items():
                serial += 1
                lines.append(atom_line(serial, name, resname, chain, resseq, tuple(xyz)))
        serial += 1
        lines.append(f"TER   {serial:>5}      {resname:>3} {chain:1}{resseq:>4}")
    return "\n".join(lines + ["END"]) + "\n"


def duplex(l_rna=False):
    g = nucleotide("G")
    c = partner(g, "G", "C")
    a = nucleotide("A", shift=(0.0, 0.0, 25.0))
    u = partner(a, "A", "U")
    lone = nucleotide("G", shift=(0.0, 60.0, 0.0))
    names = ("LG5", "LA3", "LU5", "LC3") if l_rna else ("G5", "A3", "U5", "C3")
    return [("A", [(names[0], 1, g), (names[1], 2, a)]),
            ("B", [(names[2], 3, u), (names[3], 4, c)]),
            ("C", [("G", 5, lone), ("MG", 6, {"MG": np.array([0.0, 80.0, 0.0])})])]


HEADER = ["HEADER    RNA                                     01-JAN-26   XXXX",
          "REMARK   2 RESOLUTION. 3.00 ANGSTROMS.",
          "REMARK 950 OLD ANNOTATION LINE"]


def coordinate_columns(text):
    return [ln[30:54] for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]


def test_builtin_records_and_placement(monkeypatch):
    monkeypatch.setitem(sys.modules, "barnaba", None)  # 'auto' falls back to builtin
    text = pdb_text(duplex(), HEADER)
    res = annotate(text, engine="auto")
    lines = res.text.splitlines()
    assert lines[:2] == HEADER[:2]
    assert "REMARK 950 OLD ANNOTATION LINE" not in res.text  # an earlier annotation is replaced
    first_atom = next(k for k, ln in enumerate(lines) if ln.startswith("ATOM"))
    assert lines[2:first_atom] == res.records
    assert res.records[0] == "REMARK 950 MDATK BASE-PAIR ANNOTATION"
    assert res.records[1].startswith("REMARK 950 ENGINE builtin")
    assert "REMARK 950 PAIRS 2 (WATSON-CRICK 2, WOBBLE 0, OTHER 0)" in res.records
    assert "REMARK 950 PAIR  A   1 G5  B   4 C3  WCc" in res.records
    assert "REMARK 950 PAIR  A   2 A3  B   3 U5  WCc" in res.records
    assert not any("STACK" in r for r in res.records)
    assert any("no stacking" in n for n in res.notes)
    tail = res.records[-6:]
    assert tail == ["REMARK 950 CHAINS A,B", "REMARK 950 SEQ   GA&UC", "REMARK 950 DBN   ((&))",
                    "REMARK 950 CHAINS C", "REMARK 950 SEQ   G", "REMARK 950 DBN   ."]
    assert coordinate_columns(res.text) == coordinate_columns(text)  # coordinates verbatim
    assert all(len(ln) <= 80 for ln in lines)
    again = annotate(res.text, engine="builtin")  # idempotent: records replaced, not duplicated
    assert again.text == res.text


def test_options_and_l_rna():
    res = annotate(pdb_text(duplex(l_rna=True)), engine="builtin", pairs=False, dotbracket=False)
    assert not any(" PAIR " in r or "DBN" in r for r in res.records)
    assert "REMARK 950 PAIRS 2 (WATSON-CRICK 2, WOBBLE 0, OTHER 0)" in res.records
    res = annotate(pdb_text(duplex(l_rna=True)), engine="builtin")
    assert "REMARK 950 PAIR  A   1 LG5 B   4 LC3 WCc" in res.records
    with pytest.raises(ValueError, match="unknown annotation engine"):
        annotate(pdb_text(duplex()), engine="rnaview")


def test_dotbracket_wrapped_to_60_characters():
    g = nucleotide("G")
    residues = [("U", k + 1, moved(nucleotide("U"), (0.0, 0.0, 8.0 * k))) for k in range(70)]
    residues[0] = ("G", 1, g)
    residues[-1] = ("C", 70, partner(g, "G", "C"))
    res = annotate(pdb_text([("A", residues)]), engine="builtin")
    seq = [r for r in res.records if r.startswith("REMARK 950 SEQ")]
    dbn = [r for r in res.records if r.startswith("REMARK 950 DBN")]
    assert [len(r) - 17 for r in seq] == [60, 10] and [len(r) - 17 for r in dbn] == [60, 10]
    assert dbn[0][17] == "(" and dbn[1].endswith(")")


def test_barnaba_records(monkeypatch):
    fake_barnaba(monkeypatch)
    res = annotate(pdb_text(duplex()), engine="auto")
    assert res.records[1].startswith("REMARK 950 ENGINE barnaba")
    # nucleotides G1 A2 U3 C4 G5 (MG left out): WCc G1-G5, WHt A2-U3, stacking A2-U3
    assert "REMARK 950 PAIR  A   1 G5  C   5 G   WCc" in res.records
    assert "REMARK 950 PAIR  A   2 A3  B   3 U5  WHt" in res.records
    assert "REMARK 950 STACK A   2 A3  B   3 U5  >>" in res.records
    assert "REMARK 950 PAIRS 2 (WATSON-CRICK 1, WOBBLE 0, OTHER 1)" in res.records
    assert "REMARK 950 CHAINS A,C" in res.records and "REMARK 950 DBN   (.&)" in res.records
    res = annotate(pdb_text(duplex()), engine="barnaba", stacking=False)
    assert not any("STACK" in r for r in res.records)


def test_cli_annotate_logs_step(tmp_path, monkeypatch):
    from mdatk.cli import main
    from mdatk.config import Project
    from mdatk.projectfile import read_log

    monkeypatch.setitem(sys.modules, "barnaba", None)
    pdb, out, proj = tmp_path / "in.pdb", tmp_path / "out.pdb", tmp_path / "p.mdatk.toml"
    pdb.write_text(pdb_text(duplex()))
    Project().save(str(proj))
    assert main(["pdb", "annotate", str(pdb), "-o", str(out), "--no-stacking", "-p", str(proj)]) == 0
    assert "REMARK 950 DBN   ((&))" in out.read_text()
    assert len(list(read_pdb(str(out)).residues())) == 6
    entry = read_log(str(proj))[-1]
    assert entry["step"] == "pdb annotate" and str(out) in entry["files"] and "builtin" in entry["summary"]
    assert main(["pdb", "annotate", str(pdb), "-o", str(out), "--engine", "barnaba"]) == 1  # not installed


def test_real_barnaba_annotation():
    pytest.importorskip("barnaba")
    res = annotate(pdb_text(duplex(l_rna=False)), engine="barnaba")
    assert "REMARK 950 PAIR  A   1 G5  B   4 C3  WCc" in res.records
    assert "REMARK 950 PAIR  A   2 A3  B   3 U5  WCc" in res.records
    assert "REMARK 950 DBN   ((&))" in res.records


def test_strands_from_ter_segments_ions_and_verbatim_models():
    blank = [("", res) for _chain, res in duplex()]  # Amber-style: no chain ids, TER between strands
    blank[2] = ("", [("G", 5, blank[2][1][0][2]), ("LA", 9, {"LA": np.array([0.0, 80.0, 0.0])})])
    text = pdb_text(blank)
    res = annotate(text, engine="builtin")
    assert res.records[-6:] == ["REMARK 950 CHAINS _", "REMARK 950 SEQ   GA&UC", "REMARK 950 DBN   ((&))",
                                "REMARK 950 CHAINS _", "REMARK 950 SEQ   G", "REMARK 950 DBN   ."]
    assert not any("numbering gaps" in n for n in res.notes)  # G5 -> LA9 is no gap: LA is no nucleotide
    body = [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM", "TER"))]
    models = "\n".join(["MODEL        1"] + body + ["ENDMDL", "MODEL        2"] + body + ["ENDMDL", "END"]) + "\n"
    out = annotate(models, engine="builtin")
    assert [ln for ln in out.text.splitlines() if not ln.startswith("REMARK 950")] == models.splitlines()
    assert out.text.splitlines()[0] == "REMARK 950 MDATK BASE-PAIR ANNOTATION"
    assert any("model 1 of 2" in n for n in out.notes)


def test_clean_drops_stale_annotation_master_and_missing_barnaba(monkeypatch):
    from mdatk import pdbtools
    monkeypatch.setitem(sys.modules, "barnaba", None)
    text = pdb_text(duplex(), HEADER[:2])
    ann = annotate(text.replace("END\n", "MASTER        1    0    0    0    0    0    0    0   10    3   0    0\nEND\n"),
                   engine="builtin")
    assert "MASTER" not in ann.text and any("MASTER" in n for n in ann.notes)
    cleaned = pdbtools.clean(ann.text, renumber=True, start=100)
    assert "REMARK 950" not in cleaned.text and any("REMARK 950" in n for n in cleaned.notes)
    one = pdb_text([("A", [("G", 1, nucleotide("G"))])])
    with pytest.raises(RuntimeError, match="pip install barnaba"):
        annotate(one, engine="barnaba")  # noticed even with nothing to pair
