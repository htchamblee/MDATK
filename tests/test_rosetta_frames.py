"""0.0.12: silent-file scores by column name and the decoy table (farfar2-models), the rna_denovo run audit
(mdatk.rosetta, 'rosetta-audit') and rigid frames of multi-body models (mdatk.frames, 'frame-check'), with
regression tests for the bugs of the scripts they replace."""

import os
import stat
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mdatk import builder, frames, pdbtools, rosetta
from mdatk.config import PlanNode, ResidueGroup, WorkflowParams
from mdatk.density import RigidTransform
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan, add_node, connect, generate_step, single_step_plan
from mdatk.workflows.scripts import silent_scores

STAMP = "20261015-1432"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def silent_text(rows, terms=("score", "rna_vdw", "linear_chainbreak"), binary=False):
    """A silent file: header, then per decoy its SCORE line (and, with ``binary``, two coordinate lines that
    end with the tag, as Rosetta writes every line of a structure)."""
    lines = ["SEQUENCE: gggcac", "SCORE: " + " ".join(terms) + " description", "REMARK BINARY SILENTFILE"]
    for tag, values in rows:
        lines.append("SCORE: " + " ".join(str(v) for v in values) + f" {tag}")
        if binary:
            lines += [f"ANNOTATED_SEQUENCE: g[RGU]gcac {tag}", f"AAAAzzzzBBBB {tag}"]
    return "\n".join(lines) + "\n"


DECOYS = [("S_000001", (-10.5, 1.0, 0.5)), ("S_000002", (-30.0, 1.0, 7.0)), ("S_000003", (-20.0, 1.0, 1.0))]


def write(path, text):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return str(path)


# ---------------------------------------------------------------------------
# silent files and the decoy table
# ---------------------------------------------------------------------------


def test_runs_of_mdatk_files_do_not_collide(tmp_path):
    # silent.py keyed decoys by the part after the last '_' of the file name: on MDATK's names that is the
    # stamp, so r01 and r02 collided and 6 decoys were read as 3
    paths = [write(tmp_path / f"apt1_01_farfar2-r0{k}_{STAMP}.out", silent_text(DECOYS)) for k in (1, 2)]
    silents = silent_scores.read_runs(paths)
    assert [s.label for s in silents] == ["r01", "r02"]
    assert sum(len(s.rows) for s in silents) == 6 and not any(s.problems for s in silents)
    out = tmp_path / "t.tsv"
    assert silent_scores.write_table(str(out), silents) == 6
    table = rosetta.read_decoy_table(str(out))
    assert [r.key for r in table.rows][:4] == ["r01/S_000001", "r01/S_000002", "r01/S_000003", "r02/S_000001"]
    assert table.columns[:3] == ["run", "tag", "pdb"]
    assert table.terms == ["score", "rna_vdw", "linear_chainbreak"]


def test_terms_by_name_when_columns_move(tmp_path):
    a = write(tmp_path / "a.out", silent_text(DECOYS))
    # another score function: one term more, linear_chainbreak in another column
    b = write(tmp_path / "b.out", silent_text([("S_000001", (-5.0, 2.0, 0.25, 9.0))],
                                              ("score", "elec_dens_fast", "linear_chainbreak", "rna_vdw")))
    silents = silent_scores.read_runs([a, b])
    vals = {(s.label, t): v for s in silents for t, v, _h in s.rows}
    assert vals[("b", "S_000001")]["linear_chainbreak"] == "0.25"
    assert vals[("a", "S_000002")]["linear_chainbreak"] == "7.0"
    assert silent_scores.terms_of(silents) == ["score", "rna_vdw", "linear_chainbreak", "elec_dens_fast"]
    lines, problems, failures, _s = rosetta.score_report([a, b], ["linear_chainbreak"],
                                                         {"linear_chainbreak": 5.0}, legacy_index=2)
    text = "\n".join(lines)
    assert "linear_chainbreak: column 3" in text and "2 SCORE header layout(s)" in text
    assert "linear_chainbreak <= 5: 3 of 4 decoys pass" in text
    assert "index 2 of the split SCORE lines holds 'rna_vdw' in the layout with 3 terms" in text
    assert "index 2 of the split SCORE lines holds 'elec_dens_fast' in the layout with 4 terms" in text
    assert "a filter on linear_chainbreak by that index reads another column" in text and not failures


def test_missing_term_is_reported_not_a_crash(tmp_path):
    # silent.py stored a header without the requested term and then died with KeyError
    a = write(tmp_path / "a.out", silent_text(DECOYS, ("score", "rna_vdw", "other")))
    lines, problems, failures, _s = rosetta.score_report([a], ["linear_chainbreak"],
                                                         {"linear_chainbreak": 5.0})
    assert any("has no column 'linear_chainbreak'" in f for f in failures)
    assert "linear_chainbreak: MISSING" in "\n".join(lines)
    with pytest.raises(ValueError, match="no column 'linear_chainbreak'"):
        silent_scores.filter_silent(a, str(tmp_path / "f.out"), "linear_chainbreak", 5.0)


def test_duplicate_tags_and_short_lines_are_reported(tmp_path):
    text = silent_text(DECOYS) + "SCORE: -99.0 1.0 0.1 S_000001\nSCORE: -1.0 S_000009\n"
    s = silent_scores.read_silent(write(tmp_path / "a.out", text))
    assert s.tags == ["S_000001", "S_000002", "S_000003", "S_000009"]
    assert s.values()["S_000001"]["score"] == "-10.5"  # the first is kept, not silently replaced
    assert s.duplicates == ["S_000001"] and len(s.problems) == 2 and s.values()["S_000009"] == {}


def test_unique_run_labels():
    a = [f"../01_farfar2/apt1_01_farfar2-r0{k}_{STAMP}.out" for k in (1, 2)]
    b = [f"../05_farfar2/apt1_05_farfar2-r0{k}_{STAMP}.out" for k in (1, 2)]
    assert silent_scores.unique_labels(a + b) == ["01-r01", "01-r02", "05-r01", "05-r02"]
    labels = silent_scores.unique_labels(["x/run_1.out", "y/run_1.out", "z/other.out"])
    assert labels == ["x-run_1", "y-run_1", "other"]


def test_filter_keeps_every_line_of_the_passing_decoys(tmp_path):
    src = write(tmp_path / "a.out", silent_text(DECOYS, binary=True))
    dst = str(tmp_path / "f.out")
    assert silent_scores.filter_silent(src, dst, "linear_chainbreak", 5.0) == (2, 1, 0)
    lines = open(dst).read().splitlines()
    assert lines[:3] == open(src).read().splitlines()[:3]  # the file's header lines
    assert not [ln for ln in lines if ln.endswith("S_000002")]
    assert len([ln for ln in lines if ln.endswith("S_000003")]) == 3
    ranking = write(tmp_path / "rank.txt",
                    f"-30.0 S_000002 {src}\n-20.0 S_000003 {src}\n-10.5 S_000001 {src}\n")
    assert silent_scores.filter_ranking(ranking, "linear_chainbreak", 5.0) == (2, 1, 0)
    assert [ln.split()[1] for ln in open(ranking)] == ["S_000003", "S_000001"]


def test_decoy_table_reader(tmp_path):
    tsv = write(tmp_path / "step" / "t.tsv", "run\ttag\tpdb\tscore\textra\tlinear_chainbreak\n"
                "r01\tS_1\tstep/models/r01/S_1.pdb\t-10\tx\t0.5\n"
                "r01\tS_2\t\t-20\ty\t7\nr02\tS_1\t\t-5\tz\tnan\n")
    write(tmp_path / "step" / "models" / "r01" / "S_1.pdb", "ATOM\n")
    t = rosetta.read_decoy_table(tsv)
    assert t.terms == ["score", "extra", "linear_chainbreak"] and t.rows[1].value("score") == -20.0
    assert t.root() == str(tmp_path) and t.pdb_path(t.rows[0]) == str(tmp_path / "step/models/r01/S_1.pdb")
    keep, notes = t.select({"linear_chainbreak": 5})
    assert [r.key for r in keep] == ["r01/S_1"] and "1 of 3 decoys pass" in notes[0]  # NaN fails
    with pytest.raises(ValueError, match="no column 'elec'"):
        t.select({"elec": 1})
    csv_path = write(tmp_path / "t.csv", "tag,run,score\nS_1,r01,-1\n")  # no pdb column, other order
    c = rosetta.read_decoy_table(csv_path)
    assert c.rows[0].run == "r01" and c.rows[0].pdb == "" and c.terms == ["score"]
    with pytest.raises(ValueError, match="not a decoy table"):
        rosetta.read_decoy_table(write(tmp_path / "bad.tsv", "a\tb\n1\t2\n"))
    assert rosetta.parse_maxima(["linear_chainbreak=5", "score=-1e2"]) == {"linear_chainbreak": 5.0,
                                                                          "score": -100.0}


def test_rosetta_scores_cli(tmp_path, capsys):
    from mdatk.cli import main
    d = tmp_path / "runs"
    for k in (1, 2):
        write(d / f"apt1_01_farfar2-r0{k}_{STAMP}.out", silent_text(DECOYS))
    for tag in ("S_000001", "S_000002", "S_000003"):
        write(tmp_path / "models" / "r01" / f"{tag}.pdb", "ATOM\n")
    write(tmp_path / "models" / "r02" / "S_000001.pdb", "ATOM\n")
    tsv = tmp_path / "t.tsv"
    code = main(["rosetta", "scores", str(d), "--max", "linear_chainbreak=5", "--tsv", str(tsv),
                 "--extracted", str(tmp_path / "models"), "--root", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 1 and "run r02: 2 of 3 decoys were not extracted" in out.out  # reported in the report too
    rows = rosetta.read_decoy_table(str(tsv)).rows
    assert len(rows) == 6 and rows[0].pdb == "models/r01/S_000001.pdb" and rows[4].pdb == ""


# ---------------------------------------------------------------------------
# farfar2-models: filter, extract every model, decoy table
# ---------------------------------------------------------------------------


def _models_plan(tmp_path, runs=2, **settings):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    fold = add_node(wf, "farfar2", settings={"sequence": "GGGCAC", "runs": runs})
    models = add_node(wf, "farfar2-models", settings=settings)
    connect(wf, fold.id, "silent", models.id, "silent")
    return Plan(wf, str(tmp_path)), models.id


def test_farfar2_models_defaults_unchanged_and_table_port(tmp_path):
    plan, mid = _models_plan(tmp_path, n_models=2)
    st = plan.steps[mid]
    files = generate_step(st)
    assert [f.name for f in files] == [f"apt1_02_farfar2-models_{STAMP}.sh"]
    assert "scores" not in files[0].content and "scores" not in st.outputs
    ports = TOOLS["farfar2-models"].output_ports(st.settings)
    assert any(p.name == "scores" and p.kind == "table" for p in ports)
    plan, mid = _models_plan(tmp_path, n_models=2, write_scores=True)
    st = plan.steps[mid]
    assert st.outputs["scores"].paths == [f"02_farfar2-models/apt1_02_scores_{STAMP}.tsv"]
    files = {f.name: f for f in generate_step(st)}
    helper = files[f"apt1_02_scores_{STAMP}.py"]
    assert helper.role == "helper" and "RUNS = [('r01'," in helper.content
    compile(helper.content, "helper", "exec")
    sh = files[f"apt1_02_farfar2-models_{STAMP}.sh"].content
    assert f"python3 apt1_02_scores_{STAMP}.py table apt1_02_scores_{STAMP}.tsv --models" in sh
    plan, mid = _models_plan(tmp_path, filter=True, filter_limits="")
    assert any("score limits" in i.message for i in plan.steps[mid].issues if i.level == "error")
    plan, mid = _models_plan(tmp_path, filter=True, filter_limits="linear_chainbreak 5")
    assert any("cannot read the score limit" in i.message
               for i in plan.steps[mid].issues if i.level == "error")


def _fake_tools(folder, scripts):
    bindir = folder / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for name, body in scripts.items():
        p = bindir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")


# extract_pdbs: with -in:file:tags the one tag, otherwise every decoy of the silent file, as <tag>.pdb in the
# current folder (DROP names a tag it "fails" to write)
FAKE_EXTRACT = textwrap.dedent('''\
    silent=""; tags=""
    for ((i=1;i<=$#;i++)); do j=$((i+1))
      [ "${!i}" = -in:file:silent ] && silent="${!j}"
      [ "${!i}" = -in:file:tags ] && tags="${!j}"
    done
    if [ -n "$tags" ]; then echo "ATOM tagged" > "$tags.pdb"; exit 0; fi
    awk '$1=="SCORE:" && $NF!="description" {print $NF}' "$silent" | while read -r t; do
      [ "$t" = "${DROP:-}" ] || echo "ATOM $t" > "$t.pdb"
    done
    ''')
FAKE_CLUSTER = ('out=""; for ((i=1;i<=$#;i++)); do [ "${!i}" = -out:file:silent ] && j=$((i+1)) && '
                'out="${!j}"; done; printf "SCORE: score linear_chainbreak description\\n'
                'SCORE: -30 9.0 S_0\\nSCORE: -20 1.0 S_1\\n" > "$out"; echo "$@" > cluster_args.txt')


def test_farfar2_models_extract_every_model(tmp_path):
    # a score filter before clustering is still possible, chosen explicitly and with a warning (0.0.12: the
    # default is after clustering, and validity filters are what goes before it)
    plan, mid = _models_plan(tmp_path, n_models=2, cluster=False, extract_all=True, extract_parallel=2,
                             dssr=False, filter=True, filter_limits="linear_chainbreak <= 5",
                             filter_when="before")
    assert not [i for i in plan.steps[mid].issues if i.level == "error"]
    assert any("changes the cluster populations" in i.message for i in plan.steps[mid].issues
               if i.level == "warning")
    plan.write([mid])
    for k in (1, 2):
        write(tmp_path / "01_farfar2" / f"apt1_01_farfar2-r0{k}_{STAMP}.out",
              silent_text(DECOYS, binary=True))
    folder = tmp_path / "02_farfar2-models"
    stale = write(folder / "models" / "r01" / "S_999999.pdb", "ATOM old run\n")  # extract_part.sh kept these
    env = _fake_tools(tmp_path, {"extract_pdbs": FAKE_EXTRACT})
    script = folder / f"apt1_02_farfar2-models_{STAMP}.sh"
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert sorted(os.listdir(folder / "models" / "r02")) == ["S_000001.pdb", "S_000002.pdb", "S_000003.pdb"]
    assert not os.path.exists(stale)
    assert not [d for d in os.listdir(folder / "models") if d.startswith(".extract")]
    # the filter ran before ranking: S_000002 (linear_chainbreak 7) is in neither filtered file
    for k in (1, 2):
        assert "S_000002" not in (folder / f"apt1_02_filtered-r0{k}_{STAMP}.out").read_text()
    table = rosetta.read_decoy_table(str(folder / f"apt1_02_scores_{STAMP}.tsv"))
    assert len(table.rows) == 6  # every decoy of every run, also the ones the filter dropped
    by = {r.key: r for r in table.rows}
    assert by["r02/S_000003"].pdb == "02_farfar2-models/models/r02/S_000003.pdb"
    assert os.path.isfile(table.pdb_path(by["r02/S_000003"])) and table.root() == str(tmp_path)
    ranked = (folder / f"apt1_02_models_{STAMP}.txt").read_text().split()
    assert ranked[:3] == ["01", "-20.0", "S_000003"]
    # a run that comes out one model short fails the step and names the missing decoy
    env["DROP"] = "S_000002"
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert proc.returncode == 1 and "were not extracted" in proc.stderr and "S_000002" in proc.stderr


def test_farfar2_models_table_names_the_ranked_models(tmp_path):
    plan, mid = _models_plan(tmp_path, n_models=2, cluster=False, dssr=False, write_scores=True)
    plan.write([mid])
    for k in (1, 2):
        write(tmp_path / "01_farfar2" / f"apt1_01_farfar2-r0{k}_{STAMP}.out", silent_text(DECOYS))
    folder = tmp_path / "02_farfar2-models"
    write(folder / f"apt1_02_models_{STAMP}.txt", "01 -99 S_000003 ../01_farfar2/stale.out\n")  # an old run
    env = _fake_tools(tmp_path, {"extract_pdbs": FAKE_EXTRACT})
    proc = subprocess.run(["bash", str(folder / f"apt1_02_farfar2-models_{STAMP}.sh")], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "linear_chainbreak" not in proc.stdout and "score: column 1" in proc.stdout  # the column audit
    by = {r.key: r.pdb for r in rosetta.read_decoy_table(str(folder / f"apt1_02_scores_{STAMP}.tsv")).rows}
    # the two best of both runs (-30.0 in r01 and r02): model-01 and model-02, the rest not extracted
    assert sorted(v for v in by.values() if v) == [f"02_farfar2-models/apt1_02_model-0{k}_{STAMP}.pdb"
                                                    for k in (1, 2)]
    assert by["r01/S_000002"] and by["r02/S_000002"] and not by["r01/S_000003"]


def test_farfar2_models_filter_after_clustering(tmp_path):
    plan, mid = _models_plan(tmp_path, n_models=2, dssr=False, filter=True,
                             filter_limits="linear_chainbreak<=5")
    assert plan.steps[mid].settings["filter_when"] == "after"  # the default
    plan.write([mid])
    for k in (1, 2):
        write(tmp_path / "01_farfar2" / f"apt1_01_farfar2-r0{k}_{STAMP}.out", silent_text(DECOYS))
    folder = tmp_path / "02_farfar2-models"
    env = _fake_tools(tmp_path, {"extract_pdbs": FAKE_EXTRACT, "rna_cluster": FAKE_CLUSTER})
    proc = subprocess.run(["bash", str(folder / f"apt1_02_farfar2-models_{STAMP}.sh")], capture_output=True,
                          text=True, env=env, cwd=str(folder))
    assert proc.returncode == 0, proc.stderr
    assert "farfar2-r01" in (folder / "cluster_args.txt").read_text()  # the unfiltered runs were clustered
    ranked = (folder / f"apt1_02_models_{STAMP}.txt").read_text().split()
    assert ranked[:3] == ["01", "-20", "S_1"]  # the best centre has linear_chainbreak 9: skipped
    assert len(rosetta.read_decoy_table(str(folder / f"apt1_02_scores_{STAMP}.tsv")).rows) == 6


# ---------------------------------------------------------------------------
# rna_denovo command lines and the run audit
# ---------------------------------------------------------------------------


def test_parse_command_expands_flags_files(tmp_path):
    write(tmp_path / "base.flags", "-cycles 50000   # sampling\n-in::file::fasta seq.fasta\n@inner.flags\n")
    write(tmp_path / "inner.flags",
          "-rna:denovo:obligate_pair_explicit 22 37 W W A\n-fixed_stems\n-heat true\n")
    cmd = rosetta.parse_command("core.init: command: /x/rna_denovo.linuxgccrelease @base.flags -s a.pdb "
                                "-in:file:s b.pdb -minimize_rna True -edensity::mapreso 5.0 -cycles 10000",
                                [str(tmp_path)])
    f = cmd.flags
    assert cmd.exe.endswith("rna_denovo.linuxgccrelease") and len(cmd.flag_files) == 2
    assert f["-cycles"] == "10000" and sorted(cmd.repeated) == ["-cycles", "-s"]  # the last one counts
    assert f["-fasta"] == "seq.fasta" and f["-s"] == "b.pdb"  # -in:file:s and -s are one flag
    assert f["-obligate_pair_explicit"] == "22 37 W W A" and f["-fixed_stems"] == "true" == f["-heat"]
    assert f["-minimize_rna"] == "true" and f["-edensity:mapreso"] == "5"
    assert rosetta.flag_key("-out::file::silent") == "-out:file:silent" != rosetta.flag_key("-in:file:silent")
    missing = rosetta.parse_command("rna_denovo @nowhere.flags -cycles 5", [str(tmp_path)])
    assert missing.problems and missing.flags["[unread flags file]"] == "nowhere.flags"


def _condition(folder, flags_text, extra="", files=None, logs=2):
    """A campaign-style condition: flags file, input files, run_N folders and logs naming the flags file."""
    for name, text in (files or {}).items():
        write(os.path.join(folder, name), text)
    write(os.path.join(folder, "run.flags"), flags_text)
    for k in range(1, logs + 1):
        os.makedirs(os.path.join(folder, f"run_{k}"), exist_ok=True)
        write(os.path.join(folder, f"run_{k}", "out.out"), "SCORE: score description\n")
        write(os.path.join(folder, f"log_{k}.txt"),
              "core.init: command: rna_denovo @../run.flags -out::file::silent out.out "
              f"-run:jran {k}{extra}\n")


def test_audit_finds_different_maps_behind_one_path(tmp_path):
    # audit_runs.py missed '-edensity::mapfile' (spelled with '::') and read no flags from an @flags file: two
    # arms whose maps differed behind the same path came out CLEAN
    flags = "-fasta ../seq.fasta -secstruct_file ../ss.txt -edensity::mapfile ../map.mrc -cycles 100\n"
    _condition(str(tmp_path / "camp" / "free"), flags, files={"seq.fasta": ">x A:1-6\nggcgcc\n",
                                                               "ss.txt": "((()))\n", "map.mrc": "MAP 4.3 A"})
    _condition(str(tmp_path / "camp" / "free_nointernalpairs"), flags,
               files={"seq.fasta": ">x A:1-6\nggcgcc\n", "ss.txt": "......\n", "map.mrc": "MAP 8 A"})
    res = rosetta.audit([str(tmp_path / "camp")], pair_suffix="_nointernalpairs", controlled="A:1-3")
    assert [c.label for c in res.conditions] == ["free", "free_nointernalpairs"]
    (pair,) = res.pairs
    assert not pair.clean and pair.other == ["[content] -edensity:mapfile"]
    assert "[secstruct]" in pair.expected and "[controlled paired in secstruct]" in pair.expected
    free = res.conditions[0].flags
    assert free["[effective condition]"] == "imposed: secstruct pairs A:1-3"
    assert res.conditions[1].flags["[effective condition]"] == "free"
    text = "\n".join(res.lines)
    assert "CONFOUNDED by 1 other difference" in text and "-out:file:silent" not in text


def test_audit_controlled_residues_are_chain_aware(tmp_path):
    # audit_runs.py read -s residues and -input_res by number alone (any chain)
    helix = builder.build_helix("GGGCCC", chains=("A", "B"), start=(1, 7))
    fasta = ">x A:1-6\ngggccc\n>x B:7-12\ngggccc\n"
    _condition(str(tmp_path / "c1"),
               "-fasta ../seq.fasta -s ../helix.pdb -input_res A:1-6 -obligate_pair 2 11\n",
               files={"seq.fasta": fasta, "helix.pdb": helix}, logs=1)
    c = rosetta.collect(str(tmp_path / "c1"), controlled=rosetta.parse_residues("B:7-8"))
    assert c.flags["[controlled in -s] helix.pdb"] == "B:7-8"  # chain B of the -s PDB
    assert c.flags["[controlled in -input_res]"] == "no"  # A:1-6 holds no B residue
    assert c.flags["[controlled in obligate pairs]"] == "no"  # 2 and 11 lie outside B:7-8
    assert c.flags["[effective condition]"].startswith(
        "imposed: fixed in the input structure (helix.pdb B:7-8)")
    c = rosetta.collect(str(tmp_path / "c1"), controlled=rosetta.parse_residues("A:2"))
    assert c.flags["[controlled in obligate pairs]"] == "A:2"  # full-model number 2 is A:2 in the FASTA
    assert c.flags["[controlled in -input_res]"] == "A:2"


def test_audit_runs_disagree_and_restarts(tmp_path):
    folder = str(tmp_path / "c")
    _condition(folder, "-cycles 100\n", logs=2)
    with open(os.path.join(folder, "log_2.txt"), "a") as fh:  # a restart with another flag appended
        fh.write("core.init: command: rna_denovo @../run.flags -cycles 7 -out:file:silent out.out\n")
    write(os.path.join(folder, "log_3.txt"), "core.init: command: extract_pdbs -in:file:silent x.out\n")
    c = rosetta.collect(folder)
    assert len(c.threads) == 2 and c.threads[1].starts == 2  # extract_pdbs' log is not a run
    assert any("2 command lines (restarts); the last differs in -cycles" in n for n in c.notes)


def test_audit_of_mdatk_plan_and_step(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = add_node(wf, "farfar2", settings={"sequence": "A: GGGCAC\nB: GUGCCC", "dotbracket": "(((((( ))))))",
                                         "runs": 2})
    b = add_node(wf, "farfar2", step=5, settings={"sequence": "A: GGGCAC\nB: GUGCCC", "runs": 2,
                                                 "dotbracket": "((((.. ..))))", "cycles": 10000})
    audit = add_node(wf, "rosetta-audit", step=6, settings={
        "pairs": "01_farfar2 = 05_farfar2", "controlled": "A:5-6 B:7-8", "allowed": "-cycles",
        "python": sys.executable, "strict": True})
    connect(wf, a.id, "silent", audit.id, "silent")
    connect(wf, b.id, "silent", audit.id, "silent")
    plan = Plan(wf, str(tmp_path))
    assert not [i for i in plan.steps[audit.id].issues if i.level != "info"], plan.steps[audit.id].issues
    plan.write()
    for folder in ("01_farfar2", "05_farfar2"):
        step = folder[:2]
        for k in (1, 2):
            os.makedirs(tmp_path / folder / "runs" / f"r0{k}", exist_ok=True)
            write(tmp_path / folder / f"apt1_{step}_farfar2-r0{k}_{STAMP}.log",
                  f"core.init: command: rna_denovo @../../apt1_{step}_farfar2_{STAMP}.flags -out:file:silent "
                  f"../../apt1_{step}_farfar2-r0{k}_{STAMP}.out -run:jran {3000 + k}\n")
    sh = (tmp_path / "06_rosetta-audit" / f"apt1_06_rosetta-audit_{STAMP}.sh").read_text()
    assert ("rosetta audit ../01_farfar2 ../05_farfar2 --pair 01_farfar2=05_farfar2 "
            "--controlled A:5-6,B:7-8") in sh
    assert "--allow=-cycles" in sh and "--strict" in sh
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run(["bash", str(tmp_path / "06_rosetta-audit" / f"apt1_06_rosetta-audit_{STAMP}.sh")],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report = (tmp_path / "06_rosetta-audit" / f"apt1_06_rosetta-audit_{STAMP}.txt").read_text()
    assert "CLEAN" in report and "same file content under other names: -fasta" in report
    matrix = (tmp_path / "06_rosetta-audit" / f"apt1_06_rosetta-audit_{STAMP}.tsv").read_text().splitlines()
    assert matrix[0] == "flag\t01_farfar2\t05_farfar2" and "-cycles\t50000\t10000" in matrix
    # without -cycles allowed the pair is confounded, and --strict fails the step
    from mdatk.cli import main
    assert main(["rosetta", "audit", str(tmp_path / "01_farfar2"), str(tmp_path / "05_farfar2"),
                 "--pair", "01_farfar2=05_farfar2", "--strict"]) == 1
    _p, st = _single("rosetta-audit", {"pairs": "x"})
    assert any("CONTROL=TREATMENT" in i.message for i in st.issues) and any(
        "connect the silent files" in i.message for i in st.issues)


def _single(tool, settings, files=None, groups=None):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    node = PlanNode(id="x", tool=tool, step=1, settings=TOOLS[tool].normalise(settings),
                    files=dict(files or {}))
    plan = single_step_plan(wf, node, "/tmp/mdatk-plan", groups)
    return plan, plan.steps["x"]


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def rot(axis, degrees):
    a = np.radians(degrees)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def two_bodies():
    """A D duplex (chains A, B) and an L duplex (chains C, D), both numbered from 1 in each chain."""
    h1 = pdbtools.read_pdb(builder.build_helix("GGCAUCGG", chains=("A", "B")))
    h2 = pdbtools.read_pdb(builder.build_helix("CCGAUGCC", chirality="L", chains=("C", "D")))
    frames.transform_structure(h2, RigidTransform(np.eye(3), np.array([25.0, 0.0, 0.0])))
    return pdbtools.Structure(segments=h1.segments + h2.segments)


def model(shift=0.0, whole=None, chains="CD", drop=""):
    """The two bodies with the second moved by ``shift`` Å along y, then all moved by ``whole``."""
    s = two_bodies()
    s.segments = [seg for seg in s.segments if seg[0].chain not in drop]
    for seg in s.segments:
        if seg[0].chain in chains and shift:
            for r in seg:
                for a in r.atoms:
                    x, y, z = np.array(a.xyz) + [0.0, shift, 0.0]
                    a.tail = f"{a.tail[:3]}{x:8.3f}{y:8.3f}{z:8.3f}{a.tail[27:]}"
    if whole is not None:
        frames.transform_structure(s, whole)
    return pdbtools.write_pdb(s)


MOVE = RigidTransform(rot([1, 2, 3], 25.0), np.array([5.0, -3.0, 2.0]))


def test_selections_and_groups():
    groups = [ResidueGroup("Aptamer", "A:1-8, B:1-8"), ResidueGroup("Target", "", "chainID C and resid 1:8"),
              ResidueGroup("Free", "1-4")]
    assert frames.parse_selection("group:Aptamer", groups) == [("A", 1, 8), ("B", 1, 8)]
    assert frames.parse_selection("Target", groups) == [("C", 1, 8)]
    assert frames.parse_selection("C,D:2-3 5-6", groups) == [("C", None, None), ("D", 2, 3), (None, 5, 6)]
    assert frames.selection_text("group:Free,C", groups) == "1-4,C"
    assert frames.bodies_text("group:Target\nHelix=D:1-8", groups) == ["Target=C:1-8", "Helix=D:1-8"]
    with pytest.raises(frames.FrameError, match="unknown residue group 'Nope'"):
        frames.parse_selection("group:Nope", groups)
    with pytest.raises(frames.FrameError, match="cannot be read as residue ranges"):
        frames.parse_selection("X", [ResidueGroup("X", "", "name P")])
    assert frames.parse_model_map("C:1-3=X:45-47") == {("X", 45): ("C", 1), ("X", 46): ("C", 2),
                                                        ("X", 47): ("C", 3)}


def test_frame_check_is_chain_aware(tmp_path):
    ref = write(tmp_path / "ref.pdb", pdbtools.write_pdb(two_bodies()))
    # two models named alike in two runs (extract_pdbs reuses tags): helix_frame.py gave both one label
    a = write(tmp_path / "r01" / "S_000001.pdb", model(0.0, MOVE))
    b = write(tmp_path / "r02" / "S_000001.pdb", model(3.0, MOVE))
    c = write(tmp_path / "r03" / "S_000001.pdb", model(0.0, MOVE, drop="CD"))  # no target: an error row
    fit = frames.parse_selection("A,B")
    bodies = frames.parse_bodies("Target=C,D")
    rows, _notes = frames.check_models(ref, [frames.ModelRef(p, p) for p in (a, b, c)], fit, bodies)
    assert rows[0].fit_rmsd < 0.01 and rows[0].bodies["Target"][1] < 0.01 and abs(rows[0].angle - 25.0) < 0.01
    assert abs(rows[1].bodies["Target"][1] - 3.0) < 0.01 and rows[1].bodies["Target"][0] == 46
    assert rows[2].error.startswith("0 atoms of Target")
    # keyed without the chain (helix_frame.py / reframe_decoys.py), chain C 1-8 overwrote chain A 1-8
    atoms, _n = frames.atoms_of(ref)
    assert len(frames.keys_in(atoms, frames.parse_selection("A"))) == 23 and len(atoms) == 92
    out = tmp_path / "check.tsv"
    frames.write_check_table(str(out), rows, bodies)
    lines = out.read_text().splitlines()
    assert lines[0].split("\t")[-3:] == ["n_Target", "rmsd_Target", "error"]
    assert len(lines) == 4  # the error row is kept
    summary = frames.summarise(rows, bodies)
    assert summary.verdict == "moved" and "3 models: 2 measured, 1 not measurable" in summary.lines[0]


def test_frame_check_verdicts_skip_models_without_the_fit_body(tmp_path):
    ref = write(tmp_path / "ref.pdb", pdbtools.write_pdb(two_bodies()))
    good = [write(tmp_path / f"m{k}.pdb", model(0.2 * k, MOVE)) for k in range(4)]
    # a model whose fit body is distorted: its target number means nothing and must not enter the statistics
    bad = two_bodies()
    for seg in bad.segments:
        if seg[0].chain == "A":
            for r in seg:
                for at in r.atoms:
                    x, y, z = np.array(at.xyz) * 1.5
                    at.tail = f"{at.tail[:3]}{x:8.3f}{y:8.3f}{z:8.3f}{at.tail[27:]}"
    worse = write(tmp_path / "bad.pdb", pdbtools.write_pdb(bad))
    fit, bodies = frames.parse_selection("A,B"), frames.parse_bodies("C,D")
    rows, _n = frames.check_models(ref, [frames.ModelRef(p, p) for p in good + [worse]], fit, bodies)
    assert [r.valid for r in rows] == [True] * 4 + [False]
    assert frames.summarise(rows, bodies).verdict == "one-frame"  # p95 of 0, 0.2, 0.4, 0.6 < 1
    assert frames.summarise(rows, bodies, same_frame=0.5).verdict == "moved"
    rows, _n = frames.check_models(ref, [frames.ModelRef(worse, worse)], fit, bodies)
    assert frames.summarise(rows, bodies).verdict == "fit-not-preserved"


def test_model_map_pairs_differently_numbered_models(tmp_path):
    ref = write(tmp_path / "ref.pdb", pdbtools.write_pdb(two_bodies()))
    s = pdbtools.read_pdb(model(1.5, MOVE))
    for r in s.residues():
        if r.chain == "C":
            r.chain, r.resseq = "X", r.resseq + 44
    m = write(tmp_path / "m.pdb", pdbtools.write_pdb(s))
    fit, bodies = frames.parse_selection("A,B"), frames.parse_bodies("C:1-8")
    rows, _n = frames.check_models(ref, [frames.ModelRef(m, m)], fit, bodies)
    assert rows[0].error
    rows, _n = frames.check_models(ref, [frames.ModelRef(m, m)], fit, bodies,
                                   model_map=frames.parse_model_map("C:1-8=X:45-52"))
    assert abs(rows[0].bodies["C:1-8"][1] - 1.5) < 0.01


def test_reframe(tmp_path, capsys):
    from mdatk.cli import main
    old = write(tmp_path / "old.pdb", pdbtools.write_pdb(two_bodies()))
    new = write(tmp_path / "new.pdb", model(0.0, MOVE))
    moved_new = write(tmp_path / "moved.pdb", model(2.0, MOVE))
    # decoys of two runs with the same tag (reframe_decoys.py wrote both to out/S_000001.pdb)
    root = tmp_path / "plan"
    for run in ("r01", "r02"):
        s = pdbtools.read_pdb(model(0.5 if run == "r02" else 0.0))
        # ANISOU columns 28-80 as pdbtools keeps them: U11 ... U23 in 29-70
        s.segments[0][0].atoms[0].anisou = " " + "".join(f"{v:7d}" for v in (100, 200, 300, 10, 20, 30)) + \
            "       P"
        write(root / "02_m" / "models" / run / "S_000001.pdb", pdbtools.write_pdb(s))
    table = write(root / "02_m" / "t.tsv", "run\ttag\tpdb\tscore\n"
                  "r01\tS_000001\t02_m/models/r01/S_000001.pdb\t-1\n"
                  "r02\tS_000001\t02_m/models/r02/S_000001.pdb\t-2\nr03\tS_000001\t\t-3\n")
    out = tmp_path / "out"
    code = main(["frame", "reframe", old, new, "--scores", table, "--fit", "A,B", "--measure", "C,D",
                 "-O", str(out), "--manifest", str(tmp_path / "m.tsv"),
                 "--scores-out", str(tmp_path / "t2.tsv")])
    text = capsys.readouterr().out
    assert code == 0 and "ONE RIGID TRANSFORM" in text and "wrote 2 moved models" in text
    a, _n = frames.atoms_of(str(out / "r01" / "S_000001.pdb"))
    b, _n = frames.atoms_of(new)
    assert max(np.abs(a[k] - b[k]).max() for k in b) < 0.002  # moved into the new frame
    c, _n = frames.atoms_of(str(out / "r02" / "S_000001.pdb"))
    assert c.keys() == a.keys() and max(np.abs(c[k] - a[k]).max() for k in a) > 0.4  # not overwritten
    first = pdbtools.read_pdb(str(out / "r01" / "S_000001.pdb")).segments[0][0].atoms[0]
    u = frames._rotate_anisou(first.anisou, MOVE.rotation.T)  # rotating back gives the input tensor
    assert [int(u[k:k + 7]) for k in range(1, 43, 7)] == pytest.approx([100, 200, 300, 10, 20, 30], abs=1)
    rows = rosetta.read_decoy_table(str(tmp_path / "t2.tsv")).rows
    # outside the plan folder the moved decoys are named by their absolute paths
    assert [r.pdb for r in rows] == [str(out / "r01" / "S_000001.pdb"), str(out / "r02" / "S_000001.pdb"), ""]
    assert main(["frame", "reframe", old, moved_new, table, "--fit", "A,B", "--measure", "C,D",
                 "-O", str(out)]) == 1
    assert "MOVED RELATIVE TO EACH OTHER" in capsys.readouterr().out
    assert main(["frame", "reframe", old, moved_new, "--fit", "C,D", "--measure", "A,B", "--tol", "1"]) == 1
    swapped = write(tmp_path / "swapped.pdb", model(0.0, MOVE).replace(" A ", " Q ").replace(" C ", " A ")
                    .replace(" Q ", " C "))
    assert main(["frame", "reframe", old, swapped, "--fit", "A", "--measure", "B"]) == 2


def test_frame_check_step(tmp_path):
    ref = write(tmp_path / "ref.pdb", pdbtools.write_pdb(two_bodies()))
    new = write(tmp_path / "new.pdb", model(0.0, MOVE))
    models = [write(tmp_path / "in" / f"m{k}.pdb", model(0.1 * k)) for k in range(3)]
    groups = [ResidueGroup("Aptamer", "A:1-8,B:1-8"), ResidueGroup("Target", "C:1-8,D:1-8")]
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    fc = add_node(wf, "frame-check", step=4, settings={"reframe": True, "python": sys.executable})
    fc.files = {"reference": ref, "new_reference": new, "models": ";".join(models)}
    plan = Plan(wf, str(tmp_path / "plan"), groups)
    st = plan.steps[fc.id]
    assert not [i for i in st.issues if i.level == "error"], st.issues
    assert st.outputs["reframed"].paths[2] == f"04_frame-check/apt1_04_reframed-03_{STAMP}.pdb"
    plan.write()
    folder = tmp_path / "plan" / "04_frame-check"
    sh = (folder / f"apt1_04_frame-check_{STAMP}.sh").read_text()
    assert "--fit A:1-8,B:1-8 --measure Target=C:1-8,D:1-8" in sh  # groups written out: no project needed
    proc = subprocess.run(["bash", str(folder / f"apt1_04_frame-check_{STAMP}.sh")], capture_output=True,
                          text=True, env=dict(os.environ, PYTHONPATH=REPO))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ONE FRAME" in (folder / f"apt1_04_frame-check_{STAMP}.txt").read_text()
    for p in st.outputs["reframed"].paths:
        assert os.path.isfile(tmp_path / "plan" / p)
    _p, st = _single("frame-check", {}, {"reference": ref})
    msgs = [i.message for i in st.issues if i.level == "error"]
    assert any("unknown residue group 'Aptamer'" in m for m in msgs)
    assert any("connect models" in m for m in msgs)


def test_frame_check_of_the_farfar2_set_with_its_table(tmp_path):
    # the set of decoys and its decoy table connected together: every decoy was measured twice and the table's
    # cut applied only to the table's copies (moved twice with reframe, too); and with fewer decoys than runs x
    # models per run (-autofilter, a failed run) the reframed-NN files the step promised were never all
    # written: the step stayed 'partial' and the steps after it were given files that did not exist
    from mdatk.workflows import runstate
    root = tmp_path / "plan"
    ref = write(tmp_path / "ref.pdb", pdbtools.write_pdb(two_bodies()))
    new = write(tmp_path / "new.pdb", model(0.0, MOVE))
    groups = [ResidueGroup("Aptamer", "A:1-8,B:1-8"), ResidueGroup("Target", "C:1-8,D:1-8")]
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    fold = add_node(wf, "farfar2", settings={"sequence": "GGGCAC", "runs": 2, "nstruct": 3, "autofilter": True})
    fc = add_node(wf, "frame-check", settings={"reframe": True, "python": sys.executable,
                                               "score_cut": "linear_chainbreak=1"})
    fc.files = {"reference": ref, "new_reference": new}
    connect(wf, fold.id, "decoys", fc.id, "models")
    connect(wf, fold.id, "decoys", fc.id, "scores")  # into a table input: the decoy table
    plan = Plan(wf, str(root), groups)
    st = plan.steps[fc.id]
    assert st.ctx.has("scores") and not [i for i in st.issues if i.level == "error"], st.issues
    assert len(st.outputs["reframed"].paths) == 6 and st.outputs["reframed"].data["empty_slots"]
    plan.write()
    # what the FARFAR2 step's decoys script leaves: four decoys (autofilter), one of them fails the cut
    decoys = [("r01", "S_000001", 0.0, 0.5), ("r01", "S_000003", 0.1, 0.2), ("r02", "S_000001", 0.2, 7.0),
              ("r02", "S_000002", 0.3, 0.1)]
    rows = ["run\ttag\tpdb\tscore\tlinear_chainbreak"]
    for run, tag, shift, chainbreak in decoys:
        write(root / "01_farfar2" / "models" / run / f"{tag}.pdb", model(shift))
        rows.append(f"{run}\t{tag}\t01_farfar2/models/{run}/{tag}.pdb\t-10\t{chainbreak}")
    write(root / "01_farfar2" / f"apt1_01_scores_{STAMP}.tsv", "\n".join(rows) + "\n")
    write(root / "01_farfar2" / f"apt1_01_decoys_{STAMP}.txt",
          "".join(f"01_farfar2/models/{run}/{tag}.pdb\n" for run, tag, _s, _c in decoys))
    folder = root / "02_frame-check"
    proc = subprocess.run(["bash", str(folder / f"apt1_02_frame-check_{STAMP}.sh")], capture_output=True,
                          text=True, env=dict(os.environ, PYTHONPATH=REPO))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report = (folder / f"apt1_02_frame-check_{STAMP}.tsv").read_text().splitlines()
    assert len(report) == 4  # each decoy once, the one the cut drops not at all
    assert sum("01_farfar2/models/r02/S_000001.pdb" in r for r in report) == 0
    assert "3 models: 3 measured" in (folder / f"apt1_02_frame-check_{STAMP}.txt").read_text()
    # exactly the six promised numbered models: the three moved decoys, then empty slots; nothing twice
    for k in range(1, 7):
        path = str(folder / f"apt1_02_reframed-{k:02d}_{STAMP}.pdb")
        assert bool(pdbtools.empty_slot(path)) == (k > 3), k
    assert pdbtools.empty_slot(str(folder / f"apt1_02_reframed-04_{STAMP}.pdb")) == \
        "empty slot 4 of 6: only 3 models passed"
    assert not (folder / f"apt1_02_reframed_{STAMP}").exists()
    moved = rosetta.read_decoy_table(str(folder / f"apt1_02_reframed-scores_{STAMP}.tsv")).rows
    assert [r.pdb for r in moved] == [f"02_frame-check/apt1_02_reframed-0{k}_{STAMP}.pdb" for k in (1, 2)] + \
        ["", f"02_frame-check/apt1_02_reframed-03_{STAMP}.pdb"]
    assert runstate.run_status(Plan(wf, str(root), groups), fc.id, STAMP).state == "done"


def test_gather_models_reads_each_file_once_and_never_superseded_runs(tmp_path):
    folder = tmp_path / "02_m"
    a = write(folder / "models" / "r01" / "S_000001.pdb", model(0.0))
    write(folder / "models" / "r02" / "S_000001.pdb", model(0.1))
    write(folder / "superseded" / f"{STAMP}_20261101-120000" / "models" / "r01" / "S_000001.pdb", model(0.2))
    models, notes, _t = frames.gather_models([str(folder), a])
    assert [os.path.relpath(m.path, folder) for m in models] == ["models/r01/S_000001.pdb",
                                                                 "models/r02/S_000001.pdb"]
    assert "1 model file(s) given more than once are read once" in notes
    # a file given that is a decoy of the table: one model with its run and tag, named like the files given
    table = write(folder / "t.tsv", "run\ttag\tpdb\tlinear_chainbreak\n"
                  "r01\tS_000001\t02_m/models/r01/S_000001.pdb\t9\n"
                  "r02\tS_000001\t02_m/models/r02/S_000001.pdb\t0\n")
    models, notes, _t = frames.gather_models([str(folder / "models")], table, str(tmp_path),
                                             {"linear_chainbreak": 5.0})
    assert [(m.label, m.run, m.tag, m.given) for m in models] == [("02_m/models/r02/S_000001.pdb", "r02",
                                                                   "S_000001", True)]
    assert "2 model(s) given are decoys of the table too: each is read once; 1 of them fail the cut and are " \
        "left out" in notes
    assert frames.output_names(models, "m-{n}.pdb") == ["m-01.pdb"]
    with pytest.raises(frames.FrameError, match="numbered by"):
        frames.fill_slots(str(tmp_path), "{stem}.pdb", 2, 1)


def test_frame_check_warns_about_density_sampled_models(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    fold = add_node(wf, "farfar2", settings={"sequence": "GGGCAC"})
    fold.files = {"map": "/data/map.mrc"}
    models = add_node(wf, "farfar2-models", settings={"write_scores": True})
    fc = add_node(wf, "frame-check", settings={"reframe": True, "fit": "A:1-3", "measure": "A:4-6"})
    fc.files = {"reference": "/data/ref.pdb", "new_reference": "/data/new.pdb"}
    connect(wf, fold.id, "silent", models.id, "silent")
    connect(wf, models.id, "scores", fc.id, "scores")
    plan = Plan(wf, str(tmp_path))
    st = plan.steps[fc.id]
    assert any("density weight 20" in i.message for i in st.issues if i.level == "warning")
    assert st.outputs["reframed_scores"].paths == [f"03_frame-check/apt1_03_reframed-scores_{STAMP}.tsv"]
    sh = {f.name: f for f in generate_step(st)}[f"apt1_03_frame-check_{STAMP}.sh"].content
    assert f"--scores ../02_farfar2-models/apt1_02_scores_{STAMP}.tsv --root .." in sh
