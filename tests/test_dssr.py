"""DSSR as a base-pair engine (mdatk.dssr): the renamed copy with synthetic residue ids, reading DSSR's
JSON in several id grammars, the annotate engine and 'mdatk pdb interface --engine dssr'.

x3dna-dssr itself is licensed separately and is not installed here: a stand-in program (FAKE_DSSR) reads
the copy MDATK writes and answers with JSON in the layout of DSSR 1.x ('nts', 'pairs', 'stacks',
'hbonds'), with nucleotide ids in the grammar a test chooses."""

import json
import os
import stat
import sys

import numpy as np
import pytest

from mdatk import builder, dssr, interfaces, pdbtools
from mdatk.annotate import annotate
from mdatk.cli import main

FAKE_DSSR = r'''#!{python}
"""Stand-in for x3dna-dssr: N1/N3 Watson-Crick-edge pairs, chain runs as stacks, N/O hydrogen bonds."""
import json, math, os, sys
args = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith("-") and "=" in a)
if "--json" not in sys.argv:
    sys.exit("expected --json")
style = os.environ.get("FAKE_DSSR_STYLE", "v1")
res, order = {{}}, []
for ln in open(args["-i"]):
    if ln.startswith(("ATOM", "HETATM")):
        key = (ln[21], int(ln[22:26]))
        if key not in res:
            res[key] = {{"name": ln[17:20].strip(), "atoms": {{}}}}
            order.append(key)
        res[key]["atoms"][ln[12:16].strip()] = tuple(float(ln[c:c + 8]) for c in (30, 38, 46))

def ntid(k):
    name = res[k]["name"]
    if style == "v1":
        return f"{{k[0]}}.{{name}}{{k[1]}}"
    if style in ("v2", "nonts"):
        return f"1:{{k[0]}}.{{name}}/{{k[1]}}"
    return f"nt{{order.index(k) + 1}}"  # opaque ids: only the nts table maps them

def atom(name, k):
    return f"{{ntid(k)}}@{{name}}" if style == "v2" else f"{{name}}@{{ntid(k)}}"

def dist(p, q):
    return math.dist(p, q)

WC = {{"G": "N1", "A": "N1", "C": "N3", "U": "N3"}}
pairs, hbonds = [], []
for i, a in enumerate(order):
    for b in order[i + 1:]:
        x, y = res[a], res[b]
        na, nb = WC.get(x["name"]), WC.get(y["name"])
        if na in x["atoms"] and nb in y["atoms"] and dist(x["atoms"][na], y["atoms"][nb]) < 3.3 \
                and 8.5 < dist(x["atoms"]["C1'"], y["atoms"]["C1'"]) < 12.5:
            kind = x["name"] + y["name"]
            name = "WC" if kind in ("GC", "CG", "AU", "UA") else "Wobble" if kind in ("GU", "UG") else "~rWC"
            pairs.append({{"index": len(pairs) + 1, "nt1": ntid(a), "nt2": ntid(b),
                          "bp": f"{{x['name']}}-{{y['name']}}", "name": name, "LW": "cWW", "DSSR": "cW-W"}})
        for an, p in x["atoms"].items():
            for bn, q in y["atoms"].items():
                if an[0] in "NO" and bn[0] in "NO" and dist(p, q) < 3.3:
                    hbonds.append({{"index": len(hbonds) + 1, "atom1_id": atom(an, a),
                                   "atom2_id": atom(bn, b), "distance": round(dist(p, q), 3),
                                   "donAcc_type": "standard"}})
runs = []  # strands: residues joined O3'-P, which a stand-in may call stacked
for k in order:
    prev = runs[-1][-1] if runs else None
    if prev and "O3'" in res[prev]["atoms"] and "P" in res[k]["atoms"] \
            and dist(res[prev]["atoms"]["O3'"], res[k]["atoms"]["P"]) < 2.0:
        runs[-1].append(k)
    else:
        runs.append([k])
stacks = [{{"index": n + 1, "nts_long": ",".join(ntid(k) for k in ks)}}
          for n, ks in enumerate(runs) if len(ks) > 1]
doc = {{"num_pairs": len(pairs), "pairs": pairs, "stacks": stacks,
       "metadata": {{"version": "fake-dssr 9.9"}}}}
if style != "nohb":
    doc["hbonds"] = hbonds
if style not in ("nonts",):
    doc["nts"] = [{{"index": n + 1, "chain_name": k[0], "nt_resnum": k[1], "nt_name": res[k]["name"],
                   "nt_id": ntid(k)}} for n, k in enumerate(order)]
open("dssr-pairs.pdb", "w").write("auxiliary file DSSR leaves in its working folder\n")
json.dump(doc, open(args["-o"], "w"))
'''


def fake_dssr(tmp_path) -> str:
    path = tmp_path / "fake-x3dna-dssr"
    path.write_text(FAKE_DSSR.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def duplex(seq="GGCAUCGA", chirality="D", scheme=None, start=(1, 1)):
    return pdbtools.read_pdb(builder.build_helix(seq, chirality=chirality, chains=("A", "B"), scheme=scheme,
                                                 start=start))


# --- ids and JSON -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("A.G1", ("A", 1, "")), ("1:A.G1", ("A", 1, "")), ("A.G1^B", ("A", 1, "B")), ("AB.G12", ("AB", 12, "")),
    (".G1", ("", 1, "")), ("A.0A/45", ("A", 45, "")), ("A.5MC40", ("A", 40, "")), ("C.A-3", ("C", -3, "")),
    ("B.PSU39", ("B", 39, "")), ("A.DG1", ("A", 1, "")),
])
def test_dssr_ids_of_every_grammar_are_read(text, expected):
    """The id forms the user's script could not read (several-letter and blank chains, names with digits,
    insertion codes) all parse; MDATK's own copies only need the one-letter chain and the number."""
    assert dssr.parse_nt_id(text) == expected


def test_lw_codes_follow_the_remark_950_convention():
    assert dssr.lw_code("cWW", "WC") == "WCc" and dssr.lw_code("cWW", "Wobble") == "GUc"
    assert dssr.lw_code("cWW", "~rWC") == "WWc" and dssr.lw_code("tHS", "Sheared") == "HSt"
    assert dssr.lw_code("", "") == ""


def test_json_mapped_back_through_the_nts_table_and_either_atom_order():
    original = {0: 10, 1: 11, 9999: 12}  # slot -> caller's index (slot 9999 is chain B residue 1)
    doc = {"nts": [{"chain_name": "A", "nt_resnum": 1, "nt_id": "x1"},
                   {"chain_name": "A", "nt_resnum": 2, "nt_id": "x2"},
                   {"chain_name": "B", "nt_resnum": 1, "nt_id": "x3"}],
           "pairs": [{"nt1": "x1", "nt2": "x3", "LW": "cWW", "name": "WC"},
                     {"nt1": "x1", "nt2": "Q.G7", "LW": "cWW", "name": "WC"}],  # unknown residue
           "stacks": [{"nts_long": "x1,x2"}],
           "hbonds": [{"atom1_id": "N1@x1", "atom2_id": "x3@N3", "distance": 2.9},
                      {"atom1_id": "MG@Z.MG1", "atom2_id": "OP1@x2"}]}  # an ion: not in the copy
    res = dssr.parse_json(doc, original)
    assert res.pairs == [(10, 12, "cWW", "WC")] and res.stacks == [(10, 11)]
    assert res.hbonds == [(10, "N1", 12, "N3", 2.9)] and (res.hbonds_seen, res.hbonds_read) == (2, 1)
    assert any("could not be mapped" in n for n in res.notes)
    none = dssr.parse_json({"pairs": [], "hbonds": [{"atom1_id": "N1", "atom2_id": "N3"}]}, original)
    assert none.hbonds_seen == 1 and any("none of the 1 hydrogen bonds" in n for n in none.notes)
    missing = dssr.parse_json({"pairs": []}, original)
    assert not missing.has_hbonds and any("no hydrogen-bond list" in n for n in missing.notes)


def test_renamed_copy_keeps_nucleotides_only_and_one_model(tmp_path):
    """The user's prepare() renamed every residue (Mg and Rosetta's VRT became U), dropped ENDMDL so that
    multi-model files came out malformed, and called a flat C4' 'L'."""
    s = duplex("GCAU", chirality="L", scheme="rosetta")
    text = pdbtools.write_pdb(s).replace("END\n", "")
    text += ("HETATM 9001 MG    MG C   1       0.000   0.000  30.000  1.00  0.00          MG\n"
             "HETATM 9002 ORIG VRT D   1       0.000   0.000   0.000  1.00  0.00           X\n"
             "HETATM 9003 X    VRT D   1       1.000   0.000   0.000  1.00  0.00           X\n"
             "END\n")
    multi = ("MODEL        1\n" + text.replace("END\n", "ENDMDL\nMODEL        2\n")
             + text.replace("END\n", "ENDMDL\nEND\n"))
    nts = interfaces.nucleotides(pdbtools.read_pdb(multi))
    assert len(nts) == 8 and {n.base for n in nts} == set("GCAU") and {n.hand for n in nts} == {"L"}
    runner = dssr.DssrRunner([n.canonical for n in nts], [[(a, n.atoms[a]) for a in n.heavy] for n in nts])
    copy = runner.pdb_text()
    names = {ln[17:20].strip() for ln in copy.splitlines() if ln.startswith("ATOM")}
    assert names == set("GCAU")  # canonical names, no MG / VRT
    assert "MODEL" not in copy and "ENDMDL" not in copy and copy.rstrip().endswith("END")
    assert sorted({(ln[21], int(ln[22:26])) for ln in copy.splitlines() if ln.startswith("ATOM")}) == \
        [("A", k) for k in range(1, 9)]  # synthetic ids, one per nucleotide
    flat = {"C4'": (0.0, 0.0, 0.0), "O4'": (1.0, 0.0, 0.0), "C3'": (0.0, 1.0, 0.0), "C5'": (1.0, 1.0, 0.0)}
    assert pdbtools.sugar_chirality(flat)[0] is None  # what interfaces.Nucleotide.hand uses


@pytest.mark.parametrize("style", ["v1", "v2", "nonts", "opaque"])
def test_interface_pairs_found_with_per_chain_numbering_in_every_id_grammar(tmp_path, monkeypatch, style):
    """The user's report: DSSR found the pairs but the script counted zero across the interface (its
    crosses() compared residue numbers, which restart per chain in Amber / tleap numbering, and its id
    regex failed on several id forms).  Ids now map back by index, whatever their grammar."""
    exe = fake_dssr(tmp_path)
    monkeypatch.setenv("FAKE_DSSR_STYLE", style)
    monkeypatch.chdir(tmp_path)
    s = duplex("GGCAUCGA", start=(1, 1))  # chains A 1-8 and B 1-8
    rows, notes = interfaces.interface_records(s, interfaces.parse_side("A"), interfaces.parse_side("B"),
                                               engine="dssr", dssr_exe=exe)
    pairs = [r for r in rows if r["kind"] == "pair"]
    assert len(pairs) == 8 and {r["class"] for r in pairs} == {"WCc"}
    assert {(r["residue_a"], r["residue_b"]) for r in pairs} == {(f"A:{k}", f"B:{9 - k}")
                                                                 for k in range(1, 9)}
    assert not [r for r in rows if r["kind"] == "stack"]  # the stand-in's stacks run along each strand
    assert not os.path.exists(tmp_path / "dssr-pairs.pdb")  # DSSR's auxiliary files stay in its own folder
    row = interfaces.census(s, interfaces.parse_side("A"), None,
                            interfaces.CensusParams(pairs="dssr", dssr_exe=exe))
    assert row.counts["wc"] == 8 and row.counts["other"] == 0


def test_dssr_hydrogen_bonds_give_the_same_phosphate_contacts_as_the_geometry(tmp_path, monkeypatch):
    exe = fake_dssr(tmp_path)
    d = duplex("GGCAUCGA")
    other = pdbtools.read_pdb(builder.build_helix("CCGAUG", chirality="L", chains=("C", "D"),
                                                  scheme="rosetta"))
    for res in other.residues():  # an L duplex against the D duplex's backbone
        for a in res.atoms:
            x, y, z = a.xyz
            a.tail = f"   {x + 19.0:8.3f}{y:8.3f}{z:8.3f}" + a.tail[27:]
    s = pdbtools.Structure(segments=d.segments + other.segments)
    side_a, side_b = interfaces.parse_side("A,B"), interfaces.parse_side("C,D")

    def hb(rr):
        return sorted((r["kind"], r["atoms"]) for r in rr if r["kind"] in ("bph", "ohph"))

    geo, _ = interfaces.interface_records(s, side_a, side_b, engine="builtin", hbond=3.3)  # stand-in's cut
    assert hb(geo) == [("ohph", "O2'@C:2 OP1@A:3"), ("ohph", "O2'@C:2 OP2@A:3"), ("ohph", "O2'@C:3 OP1@A:4")]
    for style in ("v1", "v2"):  # atom ids 'O2'@A.G2' and 'A.G2@O2''
        monkeypatch.setenv("FAKE_DSSR_STYLE", style)
        rows, _ = interfaces.interface_records(s, side_a, side_b, engine="dssr", dssr_exe=exe)
        assert hb(rows) == hb(geo)
        assert {r["hand_b"] for r in rows} == {"L"} and {r["hand_a"] for r in rows} == {"D"}
    monkeypatch.setenv("FAKE_DSSR_STYLE", "nohb")
    rows, notes = interfaces.interface_records(s, side_a, side_b, engine="dssr", dssr_exe=exe)
    assert not [r for r in rows if r["kind"] in ("bph", "ohph")]
    assert any("no hydrogen-bond list" in n for n in notes)


def test_annotate_engine_dssr(tmp_path, monkeypatch):
    exe = fake_dssr(tmp_path)
    monkeypatch.setenv("FAKE_DSSR_STYLE", "v1")
    text = builder.build_helix("GGCAUC", chirality="L", scheme="amber")  # LG5 ... LC3, unknown to DSSR
    res = annotate(text, engine="dssr", dssr_exe=exe)
    assert res.engine == "dssr" and res.records[1] == "REMARK 950 ENGINE dssr fake-dssr 9.9"
    pair_lines = [r for r in res.records if r.startswith("REMARK 950 PAIR ")]
    assert [r[37:].strip() for r in pair_lines] == ["WCc"] * 6
    assert pair_lines[0][17:36] == "A   1 LG5 B   6 LC3"  # the input's own residues
    assert any(r.startswith("REMARK 950 STACK") and r.endswith("--") for r in res.records)
    assert any("does not check sugar chirality" in n for n in res.notes)
    assert "REMARK 950 PAIRS 6 (WATSON-CRICK 6, WOBBLE 0, OTHER 0)" in res.records
    assert "REMARK 950 DBN   ((((((&))))))" in res.records
    with pytest.raises(dssr.DssrError, match="not found"):
        annotate(text, engine="dssr", dssr_exe=str(tmp_path / "missing-dssr"))


def test_cli_pdb_interface_with_dssr_and_annotate(tmp_path, monkeypatch, capsys):
    exe = fake_dssr(tmp_path)
    monkeypatch.setenv("FAKE_DSSR_STYLE", "v2")
    pdb = tmp_path / "m.pdb"
    pdb.write_text(builder.build_helix("GCAUGC", chirality="D"))
    out = tmp_path / "iface.tsv"
    keep = tmp_path / "keep"
    assert main(["pdb", "interface", str(pdb), "--side-a", "A", "--side-b", "B", "--engine", "dssr",
                 "--dssr-exe", exe, "--keep", str(keep), "-o", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == list(interfaces.RECORD_COLUMNS)
    assert sum(1 for ln in lines if "\tpair\t" in ln) == 6
    assert (keep / "m_dssr.pdb").exists() and json.loads((keep / "m.json").read_text())["pairs"]
    assert "DSSR does not check chirality" in capsys.readouterr().out
    annotated = tmp_path / "a.pdb"
    assert main(["pdb", "annotate", str(pdb), "-o", str(annotated), "--engine", "dssr",
                 "--dssr-exe", exe]) == 0
    assert "REMARK 950 ENGINE dssr" in annotated.read_text()


def test_hand_of_mixed_d_and_l_partners(tmp_path, monkeypatch):
    """Interface rows carry the hand of each partner from the sugar geometry, whatever the names say."""
    exe = fake_dssr(tmp_path)
    monkeypatch.setenv("FAKE_DSSR_STYLE", "v1")
    s = duplex("GCGC")
    for res in s.segments[1]:  # chain B gets L names (0G, 0C ...) although its sugars are D
        res.name = "0" + res.name
    rows, _ = interfaces.interface_records(s, interfaces.parse_side("A"), interfaces.parse_side("B"),
                                           engine="dssr", dssr_exe=exe)
    assert {(r["name_b"][:1], r["hand_b"]) for r in rows if r["kind"] == "pair"} == {("0", "D")}
    assert np.all([r["hand_a"] == "D" for r in rows])
