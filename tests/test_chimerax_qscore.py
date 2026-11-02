"""Q-scores with ChimeraX's qscore command (mdatk.mapscore), its region ranking on dQ and the planner step,
run against a stand-in 'chimerax' that prints the per-residue table the way the user's ChimeraX log does
(tab-separated rows with blank lines between them, the dashed rule, 'Ignored bad PDB record' warnings and
the 'Overall mean Q-Score' line).  The stand-in takes each residue's Q from the B-factors of its atoms, so a
test sets the scores in the model it writes."""

import json
import math
import os
import stat
import subprocess
import sys

import pytest

from mdatk import builder, citations, ensemble, mapscore, pdbtools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = "20261015-1432"

# the user's log, as quoted (only the rows shortened)
USER_LOG = """UCSF ChimeraX version: 1.9
_warning_ | Ignored bad PDB record found on line 1
REMARK  decoy from FARFAR2 run r03
Chain	Number	Name	Qavg	Qworst	Qbb	Qsc
-------------------------------------------------------
A	1	G5	0.454	0.169	0.393	0.308

A	2	C	0.512	0.201	0.488	0.533

C	1	LG5	0.196	-0.701	0.324	0.041

Overall mean Q-Score: 0.43
"""

FAKE_CHIMERAX = r'''#!PYTHON
"""Stand-in for 'chimerax --nogui --cmd ...' with the QScore bundle's qscore command."""
import json, os, re, sys

args = sys.argv[1:]
assert args[:2] == ["--nogui", "--cmd"] and len(args) == 3, args
cmd = args[2]
style = os.environ.get("FAKE_CX_STYLE", "user")
if os.environ.get("FAKE_CX_CALLS"):
    with open(os.environ["FAKE_CX_CALLS"], "a") as fh:
        fh.write(json.dumps({"cmd": cmd, "cwd": os.getcwd()}) + "\n")
fail = set(filter(None, os.environ.get("FAKE_CX_FAIL", "").split(",")))
models = {}


def out(text=""):
    sys.stdout.write(text + "\n")


def info(text):
    if style == "nogui":
        out("INFO:")
    out(text)


def warn(text):
    if style == "nogui":
        out("WARNING:")
        out(text)
    else:
        out("_warning_ | " + text)


def pre(text):  # one <pre> message of the qscore table
    if style == "nogui":  # ChimeraX --nogui with html2text: a level line, an indented block, blank lines
        out("INFO:")
        out()
        out("    " + text)
        out()
    elif style == "raw":  # without html2text: the tags stay, the next level prefix follows on the same line
        sys.stdout.write("INFO:\n<pre>" + text + "</pre>")
    else:  # the user's log: the rows with blank lines between them
        out(text)
        out()


def error(text, code=70):
    sys.stderr.write("Traceback (most recent call last):\n  ...\n" + text + "\n")
    sys.exit(code)


def residues(path):
    order, atoms = [], {}
    for k, ln in enumerate(open(path), 1):
        rec = ln[:6].strip()
        if rec in ("ATOM", "HETATM"):
            key = (ln[21], int(ln[22:26]), ln[26].strip(), ln[17:21].strip())
            if key not in atoms:
                order.append(key)
                atoms[key] = []
            atoms[key].append((ln[12:16].strip(), float(ln[60:66])))
        elif rec == "REMARK" and not ln[6:10].strip().isdigit():
            warn(f"Ignored bad PDB record found on line {k}")
            out(ln.rstrip("\n"))
        elif rec not in ("TER", "END", "MODEL", "ENDMDL", "CRYST1", "HEADER", "REMARK", "CONECT", ""):
            warn(f"Ignored bad PDB record found on line {k}")
    return order, atoms


for part in [p.strip() for p in cmd.split(";")]:
    if not part:
        continue
    m = re.fullmatch(r'open "([^"]+)" id (\d+)( maxModels 1)?', part)
    if m:
        if not os.path.isfile(m.group(1)):
            error(f"chimerax.core.errors.UserError: No such file/path: {m.group(1)}")
        models[int(m.group(2))] = m.group(1)
        continue
    m = re.fullmatch(r"log text (.+)", part)
    if m:
        info(m.group(1))
        continue
    if part == "exit":
        sys.exit(0)
    m = re.fullmatch(r"close #(\d+)", part)
    if m:
        models.pop(int(m.group(1)))
        continue
    m = re.fullmatch(r"qscore #(\d+)((?: & ~:\S+)*) toVolume #(\d+) useGui false logDetails true(.*)", part)
    if not m or os.environ.get("FAKE_CX_NO_QSCORE"):
        error(f"chimerax.core.errors.UserError: Unknown command: {part}")
    path, volume = models[int(m.group(1))], models[int(m.group(3))]
    assert volume.endswith(".mrc") and path.endswith(".pdb"), (volume, path)
    left_out = re.findall(r"~:(\S+)", m.group(2))
    extra = m.group(4).split()
    for kw in extra[::2]:
        if kw not in ("pointsPerShell", "shellRadiusStep", "referenceGaussianSigma", "maxShellRadius"):
            error(f"chimerax.core.commands.cli.AnnotationError: Expected keyword, got {kw!r}")
    order, atoms = residues(path)
    order = [k for k in order if k[3] not in left_out]
    broken = os.path.basename(path) in fail
    if broken:
        warn(f"WARNING: the following residues in #{m.group(1)} have atoms outside the bounds of volume "
             f"#{m.group(3)} and will be excluded from the Q-score calculation:")
        out(f"#{m.group(1)}/{order[1][0]}:{order[1][1]}")
    pre("Chain\tNumber\tName\tQavg\tQworst\tQbb\tQsc")
    pre("-" * 55)
    values = []
    for k, key in enumerate(order):
        chain, number, icode, name = key
        if broken and k == 1:  # ChimeraX-QScore 1.2 prints the N/A row, then fails on it
            pre(f'{chain}\t{number}{icode}\t{name}\t"N/A"\t"N/A"\t"N/A"\t"N/A"')
            error("TypeError: cannot unpack non-iterable NoneType object")
        qs = [b for a, b in atoms[key] if not a.startswith("H")]
        q = sum(qs) / len(qs)
        values += qs
        sc = "N/A" if name in ("MG", "HOH") else f"{q + 0.01:.3f}"
        pre(f"{chain}\t{number}{icode}\t{name}\t{q:.3f}\t{min(qs) - 0.1:.3f}\t{q - 0.01:.3f}\t{sc}")
    info("")
    info(f"Overall mean Q-Score: {sum(values) / len(values):.2f}")
'''


def write(path, text):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return str(path)


@pytest.fixture
def chimerax(tmp_path, monkeypatch):
    """The stand-in's path; FAKE_CX_CALLS records every call (one JSON line each)."""
    path = tmp_path / "bin" / "chimerax"
    write(path, FAKE_CHIMERAX.replace("PYTHON", sys.executable, 1))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_CX_CALLS", str(tmp_path / "calls.jsonl"))
    for var in ("FAKE_CX_STYLE", "FAKE_CX_FAIL", "FAKE_CX_NO_QSCORE"):
        monkeypatch.delenv(var, raising=False)
    return str(path)


def calls(tmp_path):
    p = tmp_path / "calls.jsonl"
    return [json.loads(ln) for ln in open(p)] if p.exists() else []


def model_with_q(path, q_of, chirality="D", remark=True, hetatm=False, sequence="GGCA"):
    """A duplex (A 1-4, B 5-8 for GGCA) whose atoms carry q_of(chain, resseq) as B-factor; L residues get L
    names."""
    n = len(sequence)
    text = builder.build_helix(sequence, chirality=chirality, chains=("A", "B"), start=(1, n + 1))
    s = pdbtools.read_pdb(text)
    for res in s.residues():
        for a in res.atoms:
            a.tail = a.tail[:33] + f"{q_of(res.chain, res.resseq):6.2f}" + a.tail[39:]
            if hetatm:
                a.record = "HETATM"
    body = pdbtools.write_pdb(s)
    return write(path, ("REMARK  decoy from FARFAR2 run r03 (free text)\n" if remark else "") + body)


def a_map(tmp_path):
    return write(tmp_path / "in" / "map.mrc", "the stand-in does not read the map")


# --- reading the log -----------------------------------------------------------------------------------

def test_reads_the_users_log_format():
    tables = mapscore.parse_chimerax_qscore(USER_LOG)
    assert len(tables) == 1 and tables[0].complete and tables[0].overall == pytest.approx(0.43)
    rows = tables[0].rows
    assert list(rows) == [("A", 1, ""), ("A", 2, ""), ("C", 1, "")]
    assert rows[("A", 1, "")] == mapscore.QRow("G5", 0.454, 0.169, 0.393, 0.308)
    assert rows[("C", 1, "")].name == "LG5" and rows[("C", 1, "")].qworst == pytest.approx(-0.701)
    scores, q_rows, virtual = mapscore.qscore_residues(tables[0], None, "qbb")
    assert scores[("C", 1, "")] == ("LG5", 0.324, 0) and virtual == 0


def test_log_variants_are_read_tolerantly():
    rows = ["A\t12A\tG\t0.500\t0.100\t0.450\tN/A", 'B\t3\tLC\t"N/A"\t"N/A"\t"N/A"\t"N/A"',
            " \t7\tU\t0.300\t0.000\t0.250\t0.350"]
    header, rule = "Chain\tNumber\tName\tQavg\tQworst\tQbb\tQsc", "-" * 55
    overall = "Overall mean Q-Score: 0.40\n"
    nogui = "".join(f"INFO:\n\n    {r}\n\n" for r in [header, rule] + rows) + "INFO:\n\nINFO:\n" + overall
    raw = "".join(f"INFO:\n<pre>{r}</pre>" for r in [header, rule] + rows) + "INFO:\n" + overall
    spaces = "\n".join([header.replace("\t", "   "), rule] + [r.replace("\t", "  ") for r in rows]
                       + ["Overall mean Q-Score: 0.40"])
    csv_file = "Chain,Number,Name,Qavg,Qworst,Qbackbone,Qsidechain\n" + "\n".join(
        r.replace("\t", ",") for r in rows) + "\nOverall mean Q-Score: 0.40\n"
    colours = "\x1b[32m" + "\n".join([header, rule] + rows) + "\x1b[0m\nOverall mean Q-Score: 0.40\n"
    for text in (nogui, raw, spaces, csv_file, colours):
        (t,) = mapscore.parse_chimerax_qscore(text)
        assert t.complete and t.overall == pytest.approx(0.40), text
        a, b = t.rows[("A", 12, "A")], t.rows[("B", 3, "")]
        assert a.qavg == 0.5 and math.isnan(a.qsc) and b.name == "LC" and math.isnan(b.qavg), text
        assert t.rows[("", 7, "")].qbb == pytest.approx(0.25), text  # a blank chain
    # a table ChimeraX stopped inside is not complete; markers tell several apart
    two = (f"MDATK-QSCORE 1\n{header}\n{rule}\n{rows[0]}\nOverall mean Q-Score: 0.5\n"
           f"log text MDATK-QSCORE 2\nMDATK-QSCORE 2\n{header}\n{rule}\n{rows[2]}\n")
    first, second = mapscore.parse_chimerax_qscore(two)
    assert (first.marker, first.complete, second.marker, second.complete) == (1, True, 2, False)
    assert mapscore.parse_chimerax_qscore("no table\nOverall mean Q-Score: 0.4\n") == []


def test_qscore_options_are_checked():
    assert mapscore.check_qscore_options("  pointsPerShell 16\nshellRadiusStep 0.05 ") == (
        "pointsPerShell 16 shellRadiusStep 0.05", [])
    opts, warnings = mapscore.check_qscore_options("resolution 4.3")
    assert opts == "resolution 4.3" and "not a keyword of the qscore command" in warnings[0]
    for bad, why in (("useGui true", "set by MDATK"), ("logDetails false", "set by MDATK"),
                     ("outputFile x.csv", "set by MDATK"), ("pointsPerShell", "has no value"),
                     ("pointsPerShell 8; close #1", "no ';'")):
        with pytest.raises(ensemble.EnsembleError, match=why):
            mapscore.check_qscore_options(bad)


def test_commands_keep_the_model_numbering():
    one = mapscore.qscore_command([("/d/m1.pdb", [])], "/d/map.mrc", "pointsPerShell 16")
    assert one == ('open "/d/m1.pdb" id 1 maxModels 1; open "/d/map.mrc" id 2; qscore #1 toVolume #2 useGui '
                   'false logDetails true pointsPerShell 16; exit')
    two = mapscore.qscore_command([("/d/m1.pdb", []), ("/d/m2.pdb", ["VRT"])], "/d/map.mrc")
    assert two.split("; ") == [
        'open "/d/map.mrc" id 1',
        "log text MDATK-QSCORE 1", 'open "/d/m1.pdb" id 2 maxModels 1',
        "qscore #2 toVolume #1 useGui false logDetails true", "close #2",
        "log text MDATK-QSCORE 2", 'open "/d/m2.pdb" id 2 maxModels 1',
        "qscore #2 & ~:VRT toVolume #1 useGui false logDetails true", "close #2", "exit"]
    with pytest.raises(ensemble.EnsembleError, match="cannot name a file"):
        mapscore.qscore_command([('/d/a"b.pdb', [])], "/d/map.mrc")


# --- running ChimeraX ------------------------------------------------------------------------------------

def test_one_run_per_model_on_the_models_as_they_are(tmp_path, chimerax):
    """L names, HETATM records and free-text REMARK lines go to ChimeraX unchanged (no renamed copy)."""
    m1 = model_with_q(tmp_path / "in" / "d1.pdb", lambda c, r: 0.5, chirality="L", hetatm=True)
    m2 = model_with_q(tmp_path / "in" / "d2.pdb", lambda c, r: 0.2 + 0.05 * r)
    work = str(tmp_path / "work")
    scored, failed, notes = mapscore.score_models([m1, m2], a_map(tmp_path), "qscore", chimerax=chimerax,
                                                  work_dir=work)
    assert failed == {} and [s.model for s in scored] == [m1, m2]
    assert [len(json.loads(json.dumps(c))["cmd"]) > 0 for c in calls(tmp_path)] == [True, True]
    assert all(c["cwd"] == work for c in calls(tmp_path))
    assert sorted(os.listdir(work)) == ["d1_chimerax.log", "d2_chimerax.log"]  # logs only, no copies
    l_names = {r.name for r in pdbtools.read_pdb(m1).residues()}
    first = scored[0]
    assert {name for name, _q, _n in first.scores.values()} == l_names and not l_names & set("ACGU")
    assert first.scores[("A", 1, "")][1] == pytest.approx(0.5) and first.scores[("A", 1, "")][2] > 10
    second = scored[1].q_rows[("B", 8, "")]
    assert (second.qavg, second.qbb, second.qsc) == pytest.approx((0.6, 0.59, 0.61))
    assert scored[1].overall == pytest.approx(0.425, abs=0.006)
    assert any("ignored 2 PDB record" in n for n in notes)
    assert "overall mean Q-Score" in scored[0].notes[0]


@pytest.mark.parametrize("style", ["nogui", "raw"])
def test_chimerax_log_styles(tmp_path, chimerax, monkeypatch, style):
    monkeypatch.setenv("FAKE_CX_STYLE", style)
    m = model_with_q(tmp_path / "in" / "d1.pdb", lambda c, r: 0.1 * r)
    (got,), failed, _ = mapscore.score_models([m], a_map(tmp_path), "qscore", chimerax=chimerax,
                                              work_dir=str(tmp_path / "w"), column="qsc")
    assert not failed and len(got.scores) == 8 and got.scores[("B", 6, "")][1] == pytest.approx(0.61)


def test_batch_runs_open_the_map_once_and_resume_after_a_failing_model(tmp_path, chimerax, monkeypatch):
    models = [model_with_q(tmp_path / "in" / f"d{k}.pdb", lambda c, r, k=k: 0.1 * k) for k in range(1, 5)]
    monkeypatch.setenv("FAKE_CX_FAIL", "d2.pdb")
    scored, failed, _ = mapscore.score_models(models, a_map(tmp_path), "qscore", chimerax=chimerax,
                                              batch=3, work_dir=str(tmp_path / "w"))
    assert [s.model for s in scored] == [models[0], models[2], models[3]]
    assert list(failed) == [models[1]] and "outside the map box" in failed[models[1]]
    assert "#2/A:2" in failed[models[1]]
    runs = [c["cmd"] for c in calls(tmp_path)]
    assert len(runs) == 2 and runs[0].count("open ") == 4 and runs[0].count("map.mrc") == 1
    assert "d3.pdb" in runs[1] and "d4.pdb" in runs[1] and "d2.pdb" not in runs[1]  # resumed after d2
    assert [s.scores[("A", 1, "")][1] for s in scored] == pytest.approx([0.1, 0.3, 0.4])


def test_per_model_runs_report_each_failure(tmp_path, chimerax, monkeypatch):
    models = [model_with_q(tmp_path / "in" / f"d{k}.pdb", lambda c, r: 0.4) for k in range(1, 4)]
    monkeypatch.setenv("FAKE_CX_FAIL", "d1.pdb,d3.pdb")
    scored, failed, _ = mapscore.score_models(models, a_map(tmp_path), "qscore", chimerax=chimerax,
                                              work_dir=str(tmp_path / "w"))
    assert [s.model for s in scored] == [models[1]] and list(failed) == [models[0], models[2]]
    assert len(calls(tmp_path)) == 3


def test_same_named_decoys_and_blank_chains(tmp_path, chimerax):
    """FARFAR2 reuses tags in every run: r01/S_1.pdb and r02/S_1.pdb keep their own logs and scores; a blank
    chain (an empty cell in ChimeraX's table) is matched to the model's blank chain."""
    a = model_with_q(tmp_path / "in" / "r01" / "S_1.pdb", lambda c, r: 0.2, remark=False)
    b = model_with_q(tmp_path / "in" / "r02" / "S_1.pdb", lambda c, r: 0.7, remark=False)
    lines = [ln[:21] + " " + ln[22:] if ln.startswith("ATOM") and ln[21] == "A" else ln for ln in open(b)]
    write(b, "".join(lines))
    scored, failed, _ = mapscore.score_models([a, b], a_map(tmp_path), "qscore", chimerax=chimerax,
                                              work_dir=str(tmp_path / "w"))
    assert not failed and sorted(os.listdir(tmp_path / "w")) == ["S_1-2_chimerax.log", "S_1_chimerax.log"]
    assert scored[0].scores[("A", 1, "")][1] == pytest.approx(0.2)
    assert "" in {r.chain for r in scored[1].structure.residues()}  # MDATK's key of a blank chain
    assert scored[1].scores[("", 1, "")][1] == pytest.approx(0.7) and len(scored[1].scores) == 8


def test_virtual_residues_are_left_out(tmp_path, chimerax):
    m = model_with_q(tmp_path / "in" / "d1.pdb", lambda c, r: 0.5, remark=False)
    text = open(m).read().replace("END\n", "HETATM 9999 ORIG VRT X 999       0.000   0.000   0.000  1.00  "
                                         "0.90           X\nEND\n")
    write(m, text)
    (got,), failed, _ = mapscore.score_models([m], a_map(tmp_path), "qscore", chimerax=chimerax,
                                              work_dir=str(tmp_path / "w"))
    assert "qscore #1 & ~:VRT toVolume #2" in calls(tmp_path)[0]["cmd"]
    assert ("X", 999, "") not in got.scores and len(got.scores) == 8


def test_fatal_problems_stop_at_once(tmp_path, chimerax, monkeypatch):
    m = model_with_q(tmp_path / "in" / "d1.pdb", lambda c, r: 0.4)
    mp = a_map(tmp_path)
    with pytest.raises(ensemble.EnsembleError, match="ChimeraX not found"):
        mapscore.score_models([m], mp, "qscore", chimerax=str(tmp_path / "nowhere"), work_dir=str(tmp_path))
    monkeypatch.setenv("FAKE_CX_NO_QSCORE", "1")
    other = model_with_q(tmp_path / "in" / "d2.pdb", lambda c, r: 0.4)
    with pytest.raises(ensemble.EnsembleError, match="toolshed install QScore"):
        mapscore.score_models([m, other], mp, "qscore", chimerax=chimerax, work_dir=str(tmp_path / "w"))
    assert len(calls(tmp_path)) == 1  # not once per model
    with pytest.raises(ensemble.EnsembleError, match="map not found"):
        mapscore.score_models([m], str(tmp_path / "none.mrc"), "qscore", chimerax=chimerax)
    monkeypatch.delenv("FAKE_CX_NO_QSCORE")
    (_, failed, _w) = mapscore.score_models([m], mp, "qscore", chimerax=chimerax, options="resolution 4",
                                            work_dir=str(tmp_path / "w2"))
    assert "Expected keyword" in failed[m] and "status 70" in failed[m]


def test_the_executable_may_be_a_path_with_spaces_or_a_command_line(tmp_path, chimerax):
    m = model_with_q(tmp_path / "in" / "d1.pdb", lambda c, r: 0.4)
    spaced = write(tmp_path / "Program Files" / "chimerax", open(chimerax).read())
    os.chmod(spaced, 0o755)
    for exe in (spaced, f"{sys.executable} {chimerax}"):
        (got,), failed, _ = mapscore.score_models([m], a_map(tmp_path), "qscore", chimerax=exe,
                                                  work_dir=str(tmp_path / "w"))
        assert not failed and len(got.scores) == 8


def test_a_time_limit_fails_the_model(tmp_path, chimerax):
    m = model_with_q(tmp_path / "in" / "d1.pdb", lambda c, r: 0.4)
    slow = write(tmp_path / "bin" / "slow.py", "import time, sys\nprint('INFO:'); sys.stdout.flush()\n"
                                               "time.sleep(30)\n")
    (_, failed, _w) = mapscore.score_models([m], a_map(tmp_path), "qscore",
                                            chimerax=f"{sys.executable} {slow}", work_dir=str(tmp_path / "w"),
                                            timeout=1)
    assert "time limit" in failed[m]


# --- the command and the ranking on dQ ---------------------------------------------------------------------

INTERFACE = "interface=A:3-4,B:13-14"  # of the duplexes of _decoys: A 1-8, B 9-16


def _decoys(tmp_path):
    """Three decoys: the interface A:3-4 + B:13-14, the scaffold the rest (the termini A:1, A:8, B:9, B:16
    left out of the means, so their Q of 0.0 never counts).  'bright' has the best interface Q but a still
    brighter scaffold; 'sharp' the best dQ."""
    q = {"bright": (0.60, 0.80), "sharp": (0.50, 0.40), "flat": (0.30, 0.30)}
    paths = {}
    for name, (inter, scaffold) in q.items():
        def q_of(chain, r, inter=inter, scaffold=scaffold):
            return 0.0 if r in (1, 8, 9, 16) else inter if r in (3, 4, 13, 14) else scaffold
        paths[name] = model_with_q(tmp_path / "in" / f"{name}.pdb", q_of, sequence="GGCAUGCC")
    return paths


def test_map_score_ranks_on_dq_by_default(tmp_path, chimerax, capsys):
    from mdatk.cli import main
    d = _decoys(tmp_path)
    res, regs, rep = (str(tmp_path / n) for n in ("res.tsv", "reg.tsv", "report.txt"))
    argv = ["map", "score", d["bright"], d["sharp"], d["flat"], "--map", a_map(tmp_path), "--method",
            "qscore", "--chimerax", chimerax, "--region", INTERFACE, "-o", res, "--regions-out", regs,
            "--work", str(tmp_path / "w"), "--report", rep]
    assert main(argv) == 0
    text = open(rep).read()
    assert "ranked by dQ:interface (Q of interface minus Q of the reference), best first" in text
    assert text.splitlines()[4].split()[:6] == ["rank", "model", "interface", "dQ:interface", "reference",
                                                "whole"]
    order = [ln.split()[1] for ln in text.splitlines() if ln[:4].strip().isdigit()]
    assert order == ["sharp.pdb", "flat.pdb", "bright.pdb"]  # by the interface mean: bright, sharp, flat
    sharp = next(ln for ln in text.splitlines() if "sharp.pdb" in ln).split()
    assert sharp[2:5] == ["0.500", "+0.100", "0.400"]  # Q:interface, dQ:interface, reference
    assert "dQ:interface" in text and "spread -0.200 to +0.100" in text
    head = open(res).readline().rstrip("\n").split("\t")
    assert head == mapscore.QSCORE_COLUMNS
    row = next(ln.split("\t") for ln in open(res) if ln.split("\t")[1:3] == ["A", "2"])
    assert row[0] == d["bright"] and row[5:10] == ["0.800", "0.700", "0.790", "0.810", "0.800"]
    assert row[11].strip() == "qscore:qavg"
    deltas = {(ln.split("\t")[0], ln.split("\t")[1]): ln.split("\t")[7] for ln in open(regs)}
    assert deltas[(d["sharp"], "interface")] == "0.1000" and deltas[(d["bright"], "interface")] == "-0.2000"
    # the interface mean, when asked for, and Qbb as the score column
    assert main(argv[:-2] + ["--sort", "interface", "--q-column", "qbb", "--report", rep]) == 0
    text = open(rep).read()
    order = [ln.split()[1] for ln in text.splitlines() if ln[:4].strip().isdigit()]
    assert order == ["bright.pdb", "sharp.pdb", "flat.pdb"] and "ChimeraX Q-score, Qbb" in text


def test_map_score_lists_the_models_it_could_not_score(tmp_path, chimerax, monkeypatch, capsys):
    from mdatk.cli import main
    d = _decoys(tmp_path)
    monkeypatch.setenv("FAKE_CX_FAIL", "flat.pdb")
    rep, res = str(tmp_path / "report.txt"), str(tmp_path / "res.tsv")
    assert main(["map", "score", d["bright"], d["flat"], "--map", a_map(tmp_path), "--method", "qscore",
                 "--chimerax", chimerax, "--region", INTERFACE, "-o", res, "--report", rep,
                 "--batch", "2", "--work", str(tmp_path / "w")]) == 1
    text = open(rep).read()
    assert "1 model(s) not scored (not ranked):" in text and "flat.pdb: atoms outside the map box" in text
    assert {ln.split("\t")[0] for ln in list(open(res))[1:]} == {d["bright"]}
    assert "1 model(s) not scored" in capsys.readouterr().err
    monkeypatch.setenv("FAKE_CX_FAIL", "bright.pdb,flat.pdb")
    assert main(["map", "score", d["bright"], d["flat"], "--map", a_map(tmp_path), "--method", "qscore",
                 "--chimerax", chimerax, "--report", rep]) != 0
    assert "no model was scored: bright.pdb: atoms outside the map box" in capsys.readouterr().err
    assert os.path.isfile(tmp_path / "qscore" / "flat_chimerax.log")  # beside the report


def test_ensemble_regions_reads_chimerax_logs(tmp_path, capsys):
    from mdatk.cli import main
    header, rule = "Chain\tNumber\tName\tQavg\tQworst\tQbb\tQsc", "-" * 55

    def table(k, inter, scaffold):
        rows = [f"A\t{r}\tG\t{inter if r in (3, 4) else scaffold:.3f}\t0\t0\t0\n" for r in range(1, 7)]
        return f"MDATK-QSCORE {k}\n{header}\n{rule}\n" + "\n".join(rows) + "\nOverall mean Q-Score: 0.5\n"

    log = write(tmp_path / "run-001_chimerax.log", table(1, 0.6, 0.8) + table(2, 0.5, 0.4))
    assert main(["ensemble", "regions", log, "--region", "loop=A:3-4"]) == 0
    text = capsys.readouterr().out
    assert text.index("run-001_chimerax.log#2") < text.index("run-001_chimerax.log#1")  # dQ +0.1 before -0.2
    got = mapscore.read_residue_scores(write(tmp_path / "q.csv", "Chain,Number,Name,Qavg,Qworst,Qbackbone,"
                                                                 "Qsidechain\nA,1,G,0.5,0.1,0.4,N/A\n"))
    assert list(got.values())[0][("A", 1, "")] == ("G", 0.5, 0)


def test_hybrids_scored_with_chimerax(tmp_path, chimerax):
    from mdatk.cli import main
    old = model_with_q(tmp_path / "in" / "old.pdb", lambda c, r: 0.3, remark=False)
    new = model_with_q(tmp_path / "in" / "new.pdb", lambda c, r: 0.3 if c == "A" else 0.6, remark=False)
    eff = str(tmp_path / "effects.tsv")
    assert main(["pdb", "hybrids", old, new, "--body", "a=A", "--body", "b=B", "--map", a_map(tmp_path),
                 "--method", "qscore", "--chimerax", chimerax, "--batch", "4", "-o", str(tmp_path / "h"),
                 "--effects-out", eff, "--report", str(tmp_path / "h.txt")]) == 0
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in open(eff)}
    assert float(rows["b"][5]) == pytest.approx(0.3) and float(rows["a"][5]) == pytest.approx(0.0)
    assert len(calls(tmp_path)) == 1  # four hybrids in one ChimeraX run


# --- the planner step ---------------------------------------------------------------------------------------

def _issues(settings, inputs=None):
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import StepContext
    t = TOOLS["map-model-score"]
    ctx = StepContext("p", 1, STAMP, t.tag, inputs or {})
    return [(i.level, i.message) for i in t.check(t.normalise(settings), ctx)]


def test_step_options_checks_and_citations():
    from mdatk.workflows import TOOLS
    t = TOOLS["map-model-score"]
    s = t.normalise({})
    assert (s["method"], s["chimerax"], s["batch"], s["q_column"], s["qscore_options"]) == (
        "qscore", "chimerax", 1, "qavg", "")
    assert "command" not in s  # the Phenix command template is gone
    assert t.resources(s) == ["chimerax", "qscore"] and TOOLS["body-hybrids"].resources(
        TOOLS["body-hybrids"].normalise({"method": "qscore"})) == ["chimerax", "qscore"]
    keys = citations.keys_for(t.resources(s))
    assert keys == ["meng2023chimerax", "pettersen2021chimerax", "pintilie2020qscore"]
    assert citations.REFERENCES["pintilie2020qscore"].doi == "10.1038/s41592-020-0731-1"
    assert "phenix" not in t.resources(s)
    assert not [m for lvl, m in _issues({}) if lvl in ("error", "warning")]
    assert ("error", "enter the ChimeraX executable") in _issues({"chimerax": " "})
    errs = [m for lvl, m in _issues({"qscore_options": "useGui true"}) if lvl == "error"]
    assert errs and "set by MDATK" in errs[0]
    warns = [m for lvl, m in _issues({"qscore_options": "resolution 4.3"}) if lvl == "warning"]
    assert warns and "resolution is not a keyword" in warns[0]


def test_step_script_runs_chimerax_and_ranks_on_dq(tmp_path, chimerax):
    from mdatk.config import WorkflowParams
    from mdatk.workflows.plan import Plan, add_node, connect
    d = _decoys(tmp_path)
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    files = [add_node(wf, "input-file", settings={"path": p, "kind": "pdb"}) for p in d.values()]
    mp = add_node(wf, "input-file", settings={"path": a_map(tmp_path), "kind": "map"})
    ms = add_node(wf, "map-model-score", settings={
        "chimerax": chimerax, "batch": 2, "qscore_options": "pointsPerShell 8", "python": sys.executable,
        "regions": [{"name": "interface", "ranges": INTERFACE.split("=")[1]}]})
    for f in files:
        connect(wf, f.id, "file", ms.id, "model")
    connect(wf, mp.id, "file", ms.id, "map")
    plan = Plan(wf, str(tmp_path / "plan"))
    assert [i.message for _n, i in plan.all_issues() if i.level == "error"] == []
    report = plan.write()
    script = next(p for p in report.written if p.endswith(f"_mapscore_{STAMP}.sh"))
    text = open(script).read()
    assert f"--chimerax {chimerax}" in text and "--q-column qavg" in text and "--batch 2" in text
    assert "--qscore-options 'pointsPerShell 8'" in text and "Pintilie" in text
    assert "phenix" not in text.lower()
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.run(["bash", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    folder = os.path.dirname(script)
    rank = open(plan.node_path(plan.steps[ms.id].outputs["report"].paths[0])).read()
    order = [ln.split()[1] for ln in rank.splitlines() if ln[:4].strip().isdigit()]
    assert order == ["sharp.pdb", "flat.pdb", "bright.pdb"]
    # two models in the first ChimeraX run, the third alone
    assert sorted(os.listdir(os.path.join(folder, "qscore"))) == ["flat_chimerax.log", "run-001_chimerax.log"]
