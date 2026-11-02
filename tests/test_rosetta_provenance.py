"""0.0.12: the run audit keys provenance off the lines Rosetta writes before 'core.init: command:' (version,
commit, repository, date, extras), not off the command line, and keeps the seed of every part: -constant_seed
with -jran is a part's identity, and two parts with the same seed and the same inputs are duplicates (an
error).  Fully expanded command lines are the normal case.  The log headers below are synthetic, written after
the lines Rosetta's core/init/init.cc and basic/random/init_random_generator.cc print."""

import io
import os
import sys

from mdatk import rosetta
from mdatk.config import WorkflowParams
from mdatk.workflows.plan import Plan, add_node, connect, generate_step

STAMP = "20261015-1432"

RELEASE = ("core.init: Rosetta version: rosetta.binary.linux.release-371 r371 2024.39+release.59628fb "
           "59628fb0c41fe3a8bdc6c24f6dcc4d73e4cad1fc https://www.rosettacommons.org "
           "2024-09-23T20:24:43.735221\n"
           "core.init: Rosetta extras: [cxx11thread, serialization]\n")
CUSTOM = ("core.init: Rosetta version: 2024.19.post.dev+42.lrna_heterochiral.3f9a1c2 "
          "3f9a1c27d5e4b6a8c9d0e1f2a3b4c5d6e7f8a9b0 git@github.com:example-lab/rosetta-lrna.git "
          "2024-05-10T14:22:03\n"
          "core.init: Rosetta extras: []\n"
          "core.init: L-RNA heterochiral_parallel step library, checkout 3f9a1c2 "
          "(branch lrna_heterochiral)\n")
VERSION_BLOCK = ("core.init: Rosetta version information\n"
                 "core.init: Package:                 devel\n"
                 "core.init: Testing Server Revision: None\n"
                 "core.init: PEP-440 Version:         2024.19.post.dev+42.lrna_heterochiral.3f9a1c2\n"
                 "core.init: Git Commit:              3f9a1c27d5e4b6a8c9d0e1f2a3b4c5d6e7f8a9b0\n"
                 "core.init: URL:                     git@github.com:example-lab/rosetta-lrna.git\n"
                 "core.init: Date:                    2024-05-10T14:22:03\n")
OLD = ("core.init: Mini-Rosetta version: rosetta.source.release-2017.08 r139 59291c2b4f "
       "2017-02-21 10:44:59 -0800 from https://www.rosettacommons.org\n")
FLAGS = "-fasta ../seq.fasta -secstruct_file ../ss.txt -nstruct 20 -cycles 20000 -minimize_rna true"


def write(path, text):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return str(path)


def log_text(header="", seed=None, flags=FLAGS, seed_flags="",
             exe="/opt/rosetta/bin/rna_denovo.linuxgccrelease", mode="constant", offset=0):
    """A log: the Rosetta header, the fully expanded command line and (``seed``) the seed line after it."""
    text = "core.init: Checking for fconfig files in pwd and ./rosetta/flags\n" + header
    text += f"core.init: command: {exe} {flags} {seed_flags} -out:file:silent out.out\n"
    if seed is not None:
        if mode == "constant":  # Rosetta colours the words; the audit reads through the escape codes
            text += (f"basic.random.init_random_generator: \x1b[0m\x1b[31m\x1b[4mConstant seed mode\x1b[0m, "
                     f"seed={seed - offset} seed_offset={offset} real_seed={seed} thread_index=0\n")
        else:
            text += (f"basic.random.init_random_generator: 'RNG device' seed mode, using '/dev/urandom', "
                     f"seed={seed} seed_offset=0 real_seed={seed}\n")
        text += f"basic.random.init_random_generator: RandomGenerator:init: Normal mode, seed={seed} " \
                "RG_type=mt19937\n"
    return text + "protocols.rna.denovo.RNA_DeNovoProtocol: Using fragment file ...\n"


def condition(folder, logs, files=None):
    """A campaign folder: seq.fasta, ss.txt and one run folder and log per part."""
    for name, text in (files or {"seq.fasta": ">x A:1-6\nggcgcc\n", "ss.txt": "((()))\n"}).items():
        write(os.path.join(folder, name), text)
    for k, text in enumerate(logs, 1):
        os.makedirs(os.path.join(folder, f"part{k:02d}"), exist_ok=True)
        write(os.path.join(folder, f"part{k:02d}", "out.out"), "SCORE: score description\n")
        write(os.path.join(folder, f"log_{k:02d}.txt"), text)
    return str(folder)


# ---------------------------------------------------------------------------
# provenance from the log header
# ---------------------------------------------------------------------------


def test_provenance_lines_before_the_command(tmp_path):
    def build(header):
        (start,) = rosetta.read_log(write(tmp_path / "x.log", log_text(header, 4101)))
        return start.build

    rel = build(RELEASE)
    assert rel.fields == {"package": "rosetta.binary.linux.release-371", "revision": "371",
                          "version": "2024.39+release.59628fb",
                          "commit": "59628fb0c41fe3a8bdc6c24f6dcc4d73e4cad1fc",
                          "repo": "https://www.rosettacommons.org", "date": "2024-09-23T20:24:43.735221"}
    assert rel.extras == "[cxx11thread, serialization]"
    assert rel.lines[0].startswith("core.init: Rosetta version:")
    assert "fconfig" not in " ".join(rel.lines)  # only provenance lines are kept, raw
    custom = build(CUSTOM)
    assert custom.fields["commit"] == "3f9a1c27d5e4b6a8c9d0e1f2a3b4c5d6e7f8a9b0"
    assert custom.fields["repo"] == "git@github.com:example-lab/rosetta-lrna.git"
    assert custom.details == [
        "L-RNA heterochiral_parallel step library, checkout 3f9a1c2 (branch lrna_heterochiral)"]
    assert "commit 3f9a1c27" in custom.describe() and "from git@github.com:example-lab" in custom.describe()
    # the -version block of the same build gives the same version (Rosetta's one-liner leaves out 'devel' and
    # a 'None' revision)
    assert build(VERSION_BLOCK).version == custom.version == (
        "2024.19.post.dev+42.lrna_heterochiral.3f9a1c2 3f9a1c27d5e4b6a8c9d0e1f2a3b4c5d6e7f8a9b0 "
        "git@github.com:example-lab/rosetta-lrna.git 2024-05-10T14:22:03")
    old = build(OLD)
    assert old.fields["repo"] == "https://www.rosettacommons.org"
    assert old.fields["date"] == "2017-02-21 10:44:59 -0800"
    assert old.fields["commit"] == "59291c2b4f" and old.fields["revision"] == "139"
    assert not build("").known
    # a restart writes a new header: each start keeps its own build and seed
    text = log_text(RELEASE, 4101) + log_text(CUSTOM, 4101)
    first, second = rosetta.read_log(write(tmp_path / "r.log", text))
    assert first.build.fields["commit"].startswith("59628fb")
    assert second.build.fields["commit"].startswith("3f9a")
    assert "real_seed=4101" in first.seed_line and "\x1b" not in first.seed_line


def test_runs_of_one_comparison_built_with_different_rosetta_versions(tmp_path):
    camp = tmp_path / "camp"
    condition(camp / "free", [log_text(CUSTOM, 4100 + k, seed_flags=f"-constant_seed -jran {4100 + k}")
                              for k in (1, 2)])
    condition(camp / "free_nointernalpairs",
              [log_text(RELEASE, 4100 + k, seed_flags=f"-constant_seed -jran {4100 + k}") for k in (1, 2)],
              files={"seq.fasta": ">x A:1-6\nggcgcc\n", "ss.txt": "......\n"})
    res = rosetta.audit([str(camp)], pair_suffix="_nointernalpairs")
    text = "\n".join(res.lines)
    assert "B1: 2024.19.post.dev+42.lrna_heterochiral.3f9a1c2 commit 3f9a1c27" in text
    assert "B2: 2024.39+release.59628fb r371 commit 59628fb0" in text
    assert "| core.init: Rosetta version: 2024.19.post.dev+42" in text  # the raw line is kept
    assert "WARNING: these runs were built with 2 different Rosetta builds (B1, B2)" in text
    (pair,) = res.pairs
    assert not pair.clean and "[rosetta version]" in pair.other and "[rosetta extras]" in pair.other
    assert "[rosetta provenance]" in pair.other  # the custom build's checkout line
    # runs within one condition built differently: the condition notes it
    condition(tmp_path / "mixed", [log_text(CUSTOM, 1, seed_flags="-constant_seed -jran 1"),
                                   log_text(RELEASE, 2, seed_flags="-constant_seed -jran 2")])
    c = rosetta.collect(str(tmp_path / "mixed"))
    assert any("disagree on [rosetta version]" in n for n in c.notes)
    assert c.flags["[rosetta version]"].startswith("(varies: ")


# ---------------------------------------------------------------------------
# seeds: per-part identity, never normalised away; duplicates are an error
# ---------------------------------------------------------------------------


def test_every_part_keeps_its_seed_from_expanded_command_lines(tmp_path):
    folder = condition(tmp_path / "camp", [
        log_text(CUSTOM, None, seed_flags=f"-constant_seed -jran {4100 + k}") for k in range(1, 4)])
    c = rosetta.collect(folder)
    assert [t.seed.value for t in c.threads] == [4101, 4102, 4103]  # -jran preserved per part
    assert [t.command.values["-run:jran"] for t in c.threads] == [["4101"], ["4102"], ["4103"]]
    assert all(t.seed.constant and t.seed.source == "flags" for t in c.threads)
    assert not c.threads[0].command.flag_files  # no @flags file: the expanded line is read as it is
    assert "-run:jran" not in c.flags and not any("disagree" in n for n in c.notes)  # seeds differ by design
    res = rosetta.audit([folder])
    text = "\n".join(res.lines)
    assert "camp: 3 run(s), seeds 4101-4103 (-constant_seed)" in text
    for k in (1, 2, 3):
        assert f"camp/log_0{k}.txt  seed 410{k} (-constant_seed, from -jran)  build B1" in text
    assert not res.duplicates and "DUPLICATE" not in text
    rosetta.write_matrix(str(tmp_path / "m.tsv"), res.conditions)
    assert "[seeds]\t4101-4103 (-constant_seed)" in (tmp_path / "m.tsv").read_text().splitlines()
    rosetta.write_runs(str(tmp_path / "runs.tsv"), res.conditions)
    rows = [ln.split("\t") for ln in (tmp_path / "runs.tsv").read_text().splitlines()]
    head = rows[0]
    assert head == list(rosetta.RUN_COLUMNS)
    got = [dict(zip(head, r)) for r in rows[1:]]
    assert [(r["log"], r["seed"], r["jran"], r["constant_seed"]) for r in got] == [
        (f"log_0{k}.txt", f"410{k}", f"410{k}", "yes") for k in (1, 2, 3)]
    assert got[0]["commit"] == "3f9a1c27d5e4b6a8c9d0e1f2a3b4c5d6e7f8a9b0" and got[0]["build"] == "B1"
    assert "checkout 3f9a1c2" in got[0]["provenance_lines"]


def test_two_parts_with_the_same_seed_are_duplicates(tmp_path, capsys):
    folder = condition(tmp_path / "camp", [log_text(RELEASE, s, seed_flags=f"-constant_seed -jran {s}")
                                           for s in (4101, 4102, 4101)])
    res = rosetta.audit([folder])
    (dup,) = res.duplicates
    assert dup.startswith("seed 4101 under -constant_seed with the same inputs in 2 runs: camp/log_01.txt, "
                          "camp/log_03.txt")
    text = "\n".join(res.lines)
    assert "DUPLICATE SEEDS (error, 1):" in text and any("duplicate seed 4101" in p for p in res.problems)
    assert "seeds 4101-4102 (-constant_seed, 1 repeated)" in text
    from mdatk.cli import main
    assert main(["rosetta", "audit", folder]) == 0  # an error in the report ...
    assert "ERROR: duplicate seed 4101" in capsys.readouterr().err
    assert main(["rosetta", "audit", folder, "--strict"]) == 1  # ... that fails a strict audit


def test_duplicates_need_constant_seed_and_the_same_inputs(tmp_path):
    # without -constant_seed Rosetta ignores -jran: the logged seeds are random, never duplicates
    rnd = condition(tmp_path / "rnd", [log_text(RELEASE, -1029384756 + k, seed_flags="-jran 7", mode="device")
                                       for k in (1, 2)])
    res = rosetta.audit([rnd])
    assert not res.duplicates and "seed random (rng device, real_seed -1029384755; -jran 7 is ignored" in \
        "\n".join(res.lines)
    # -constant_seed without -jran: every part gets Rosetta's default seed 1111111
    default = condition(tmp_path / "default", [log_text(RELEASE, None, seed_flags="-constant_seed")] * 2)
    (dup,) = rosetta.audit([default]).duplicates
    assert dup.startswith(f"seed {rosetta.DEFAULT_JRAN} under -constant_seed")
    # the logged real_seed includes -seed_offset: one -jran with two offsets is two seeds
    offsets = condition(tmp_path / "offsets", [
        log_text(RELEASE, 1000 + k, seed_flags=f"-constant_seed -jran 1000 -seed_offset {k}", offset=k)
        for k in (1, 2)])
    res = rosetta.audit([offsets])
    assert not res.duplicates and [t.seed.value for t in res.conditions[0].threads] == [1001, 1002]
    assert "-run:seed_offset" not in res.conditions[0].flags  # per-part identity, like -jran
    # a restart under -constant_seed with the same seed repeats the first start's random stream
    restart = condition(tmp_path / "restart", [log_text(RELEASE, 5, seed_flags="-constant_seed -jran 5") * 2])
    (dup,) = rosetta.audit([restart]).duplicates
    assert "restart/log_01.txt (start 1), restart/log_01.txt (start 2)" in dup and "restart" in dup


def test_same_seed_across_conditions(tmp_path):
    # two folders of one campaign with the same flags and seeds: duplicates; with another secondary structure
    # the shared seeds are a matched-seed design (a note)
    camp = tmp_path / "camp"
    logs = [log_text(CUSTOM, 4100 + k, seed_flags=f"-constant_seed -jran {4100 + k}") for k in (1, 2)]
    condition(camp / "a", logs)
    condition(camp / "b", logs)
    condition(camp / "c", logs, files={"seq.fasta": ">x A:1-6\nggcgcc\n", "ss.txt": "......\n"})
    res = rosetta.audit([str(camp)])
    assert len(res.duplicates) == 2 and res.duplicates[0].startswith(
        "seed 4101 under -constant_seed with the same inputs in 2 runs: a/log_01.txt, b/log_01.txt")
    text = "\n".join(res.lines)
    assert ("seed 4101 is shared by runs whose inputs or build differ (a/log_01.txt, b/log_01.txt, "
            "c/log_01.txt)") in text


def test_audit_step_writes_the_runs_table(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    fold = add_node(wf, "farfar2", settings={"sequence": "GGGCAC", "runs": 2})
    audit = add_node(wf, "rosetta-audit", settings={"python": sys.executable})
    connect(wf, fold.id, "silent", audit.id, "silent")
    plan = Plan(wf, str(tmp_path))
    st = plan.steps[audit.id]
    assert st.outputs["runs"].paths == [f"02_rosetta-audit/apt1_02_runs_{STAMP}.tsv"]
    sh = generate_step(st)[0].content
    assert f"--runs-tsv apt1_02_runs_{STAMP}.tsv" in sh
    out = io.StringIO()
    folder = condition(tmp_path / "ext", [log_text(RELEASE, 4101, seed_flags="-constant_seed -jran 4101")])
    from mdatk.cli import build_parser
    args = build_parser().parse_args(["rosetta", "audit", folder, "--runs-tsv", str(tmp_path / "r.tsv")])
    assert rosetta.run(args, out, io.StringIO()) == 0
    assert "\t4101\tyes\t4101\tlog\tB1\t" in (tmp_path / "r.tsv").read_text()


def test_runs_moved_aside_into_superseded_are_not_runs_of_the_condition(tmp_path, capsys):
    # 0.0.12 moves a completed run into <step folder>/superseded/<stamp>_<time>/ (logs, flags, runs/ and
    # all) before the step is written again; the audit walked into it and reported every seed of the new run
    # twice (DUPLICATE SEEDS, a failed --strict audit) and mixed the old run's flags into the condition's
    step = tmp_path / "plan" / "01_farfar2"
    flags = f"apt1_01_farfar2_{STAMP}.flags"

    def mdatk_run(folder, cycles):
        write(folder / flags, f"-fasta seq.fasta -nstruct 2 -cycles {cycles}\n")
        write(folder / "seq.fasta", ">x A:1-6\nggcgcc\n")
        for k in (1, 2):
            write(folder / "runs" / f"r0{k}" / "S_000001.pdb", "END\n")
            seeds = f"-run:constant_seed -run:jran {3000 + k}"
            write(folder / f"apt1_01_farfar2-r0{k}_{STAMP}.log",
                  log_text(RELEASE, 3000 + k, flags=f"@../../{flags}", seed_flags=seeds))

    mdatk_run(step, 20000)
    mdatk_run(step / "superseded" / f"{STAMP}_20261101-120000", 10000)
    res = rosetta.audit([str(step)])
    (cond,) = res.conditions
    assert len(cond.threads) == 2 and not res.duplicates
    assert all("superseded" not in t.log for t in cond.threads)
    assert cond.flags["-cycles"] == "20000"
    assert "01_farfar2: 2 run(s), seeds 3001-3002" in "\n".join(res.lines)
    from mdatk.cli import main
    assert main(["rosetta", "audit", str(step), "--strict"]) == 0
    assert "ERROR" not in capsys.readouterr().err
    # a folder of conditions: superseded/ is not one of them; a superseded run named as the root still is
    assert [lab for lab, _p in rosetta.find_conditions([str(step.parent)])] == ["01_farfar2"]
    write(step.parent / "superseded" / "01_farfar2" / "x.log", "")
    assert [lab for lab, _p in rosetta.find_conditions([str(step.parent)])] == ["01_farfar2"]
    (old,) = rosetta.audit([str(step / "superseded" / f"{STAMP}_20261101-120000")]).conditions
    assert len(old.threads) == 2 and old.flags["-cycles"] == "10000"
