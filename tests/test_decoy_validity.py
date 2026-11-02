"""0.0.12: the filtering order of the FARFAR2 models step.  The rule: filter on validity before clustering
(hands, base identity, closure, required pairs - judged on every decoy and counted in that order, so the
cluster populations are those of the valid set) and on quality after it (score limits on the ranked cluster
centres).  A score cut before clustering stays possible only as an explicit choice, with a warning."""

import os
import re
import stat
import subprocess
import textwrap

import pytest

from mdatk import builder, ensemble, pdbtools, rosetta
from mdatk.config import WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan, add_node, connect, generate_step
from mdatk.workflows.scripts import decoy_validity, silent_scores

STAMP = "20261015-1432"
SEQ = "A: GGGAAAA\nB: UUUUCCC"  # the FARFAR2 step numbers these A:1-7, B:8-14
STEM = "stem=A:1-B:14, A:2-B:13, A:3-B:12"
DUPLEX = builder.build_helix("GGGAAAA", chains=("A", "B"), start=(1, 8))


def moved(text, dx=0.0, chain=None, residues=None):
    """PDB text with the atoms of ``chain`` (or of the (chain, resseq) in ``residues``) shifted along x."""
    s = pdbtools.read_pdb(text)
    for res in s.residues():
        if (res.chain, res.resseq) in (residues or ()) or res.chain == chain:
            for a in res.atoms:
                x, y, z = a.xyz
                a.tail = f"   {x + dx:8.3f}{y:8.3f}{z:8.3f}" + a.tail[27:]
    return pdbtools.write_pdb(s)


def joined(*texts):
    """One PDB of several (END lines dropped: the reader stops at the first)."""
    return "\n".join(ln for t in texts for ln in t.splitlines() if not ln.startswith("END")) + "\nEND\n"


def write(path, text):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return str(path)


def silent_text(rows, terms=("score", "rna_vdw", "linear_chainbreak")):
    """A silent file whose structure lines end with the tag, as Rosetta writes them."""
    lines = ["SEQUENCE: gggaaaauuuuccc", "SCORE: " + " ".join(terms) + " description",
             "REMARK BINARY SILENTFILE"]
    for tag, values in rows:
        lines += ["SCORE: " + " ".join(str(v) for v in values) + f" {tag}",
                  f"ANNOTATED_SEQUENCE: gggaaaauuuuccc {tag}", f"AAAAzzzzBBBB {tag}"]
    return "\n".join(lines) + "\n"


# one run's decoys: tag -> (model, (score, rna_vdw, linear_chainbreak)); the best scores are invalid ones
DECOYS = {
    "S_000001": (DUPLEX, (-30.0, 1.0, 0.5)),  # valid
    "S_000002": (builder.build_helix("GGGAAAA", chains=("A", "B"), start=(1, 8), chirality="L"),
                 (-50.0, 1.0, 0.1)),  # the mirror image: every sugar L
    "S_000003": (builder.build_helix("GGGAAAA", partner="UUUUCCA", chains=("A", "B"), start=(1, 8)),
                 (-45.0, 1.0, 0.2)),  # B:14 is A where the sequence has C
    "S_000004": (moved(DUPLEX, 4.0, residues=[("A", k) for k in range(4, 8)]),
                 (-40.0, 1.0, 7.0)),  # a chain break after A:3 (and linear_chainbreak 7)
    "S_000005": (moved(DUPLEX, 10.0, chain="B"), (-35.0, 1.0, 0.3)),  # the strands apart: no stem pair
    "S_000006": (DUPLEX, (-10.0, 1.0, 1.0)),  # valid, but outside the quality limit score <= -20
}
FIRST_FAILURE = {"S_000002": "hands", "S_000003": "bases", "S_000004": "closure", "S_000005": "pairs"}
VALIDITY = dict(valid_hands=True, hands_counts="14 D", hands_d="A:*, B:*", valid_bases=True,
                valid_closure=True, closure_mode="both", valid_pairs=True, register=STEM)


def _plan(tmp_path, runs=2, fold=None, **settings):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    fold_node = add_node(wf, "farfar2", settings=fold or {"sequence": SEQ, "runs": runs})
    models = add_node(wf, "farfar2-models", settings=settings)
    connect(wf, fold_node.id, "silent", models.id, "silent")
    return Plan(wf, str(tmp_path)), models.id


def _fake_tools(folder, scripts):
    bindir = folder / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for name, body in scripts.items():
        p = bindir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")


# extract_pdbs: the decoys as $DECOYS/<tag>.pdb (with -in:file:tags only that one), into the current folder
FAKE_EXTRACT = textwrap.dedent('''\
    silent=""; tags=""
    for ((i=1;i<=$#;i++)); do j=$((i+1))
      [ "${!i}" = -in:file:silent ] && silent="${!j}"
      [ "${!i}" = -in:file:tags ] && tags="${!j}"
    done
    if [ -n "$tags" ]; then cp "$DECOYS/$tags.pdb" "$tags.pdb"; exit 0; fi
    awk '$1=="SCORE:" && $NF!="description" {print $NF}' "$silent" | while read -r t; do
      cp "$DECOYS/$t.pdb" "$t.pdb"
    done
    ''')
# rna_cluster: records the decoys it was given and writes each of them as a cluster centre
FAKE_CLUSTER = textwrap.dedent('''\
    out=""; ins=(); collect=0
    for a in "$@"; do
      case "$a" in
        -in:file:silent) collect=1; continue;;
        -out:file:silent) collect=2; continue;;
        -*) collect=0;;
      esac
      [ $collect = 1 ] && ins+=("$a")
      [ $collect = 2 ] && out="$a" && collect=0
    done
    awk '$1=="SCORE:" && $NF!="description" {print FILENAME, $NF}' "${ins[@]}" > cluster_input.txt
    { echo "SCORE: score rna_vdw linear_chainbreak description"
      awk '$1=="SCORE:" && $NF!="description"' "${ins[@]}"; } > "$out"
    ''')


def _run(tmp_path, mid_folder="02_farfar2-models"):
    for k in (1, 2):
        write(tmp_path / "01_farfar2" / f"apt1_01_farfar2-r0{k}_{STAMP}.out",
              silent_text([(tag, values) for tag, (_m, values) in DECOYS.items()]))
    for tag, (model, _v) in DECOYS.items():
        write(tmp_path / "decoys" / f"{tag}.pdb", model)
    env = _fake_tools(tmp_path, {"extract_pdbs": FAKE_EXTRACT, "rna_cluster": FAKE_CLUSTER})
    env["DECOYS"] = str(tmp_path / "decoys")
    folder = tmp_path / mid_folder
    proc = subprocess.run(["bash", str(folder / f"apt1_02_farfar2-models_{STAMP}.sh")], capture_output=True,
                          text=True, env=env, cwd=str(folder))
    return proc, folder


# ---------------------------------------------------------------------------
# the step: validity before clustering, quality after it
# ---------------------------------------------------------------------------


def test_validity_before_clustering_and_quality_after(tmp_path):
    plan, mid = _plan(tmp_path, n_models=2, dssr=False, filter=True, filter_limits="score <= -20", **VALIDITY)
    st = plan.steps[mid]
    assert not [i for i in st.issues if i.level in ("error", "warning")], st.issues
    assert any("every model of every run is extracted" in i.message for i in st.issues)  # implied
    assert st.outputs["validity"].paths == [f"02_farfar2-models/apt1_02_validity_{STAMP}.txt"]
    files = {f.name: f for f in generate_step(st)}
    helper = files[f"apt1_02_validity_{STAMP}.py"].content
    assert "import numpy" not in helper and "from mdatk" not in helper  # standard library only
    sh = files[f"apt1_02_farfar2-models_{STAMP}.sh"].content
    order = [sh.index(k) for k in ("extract_run r01", f"validity_{STAMP}.py judge", "py keep --valid",
                                   "rna_cluster -in:file:silent apt1_02_filtered-r01", "filter-ranking",
                                   "--validity")]
    assert order == sorted(order), "extract -> judge -> keep -> cluster -> quality filter -> table"
    keep = sh.split("py keep")[1].split("\n")[0]
    assert "--valid" in keep and "--limit" not in keep  # no score cut before clustering
    plan.write([mid])
    proc, folder = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    # the report counts in order: n -> after hands -> after bases -> after closure -> after pairs
    report = (folder / f"apt1_02_validity_{STAMP}.txt").read_text()
    counts = [(m.group(1), int(m.group(2)))
              for m in re.finditer(r"^  (decoys|after \w+)\s+(\d+)", report, re.M)]
    assert counts == [("decoys", 12), ("after hands", 10), ("after bases", 8), ("after closure", 6),
                      ("after pairs", 4)]
    assert "4 of 12 decoys are valid" in report and "valid decoys per run: r01 2/6, r02 2/6" in report
    # rna_cluster was given only the valid decoys - not the five best-scoring ones of the arm
    given = (folder / "cluster_input.txt").read_text().split("\n")
    assert sorted(ln.split()[1] for ln in given if ln) == ["S_000001", "S_000001", "S_000006", "S_000006"]
    assert all("filtered-r0" in ln for ln in given if ln)
    # the quality filter worked on the ranked centres: S_000006 (score -10) is not a representative
    ranked = (folder / f"apt1_02_models_{STAMP}.txt").read_text().split("\n")
    assert [ln.split()[2] for ln in ranked if ln] == ["S_000001", "S_000001"]
    # the decoy table: every decoy, with the validity columns and the flag
    table = rosetta.read_decoy_table(str(folder / f"apt1_02_scores_{STAMP}.tsv"))
    assert len(table.rows) == 12
    for col in ("valid", "invalid_by", "invalid_reason", "hands_D", "hands_L", "hands_ok", "base_errors",
                "bases_ok", "o3p_max", "closure_ok", "pairs_formed", "pairs_approached", "pairs_ok"):
        assert col in table.columns, col
    by = {r.key: r.values for r in table.rows}
    for run in ("r01", "r02"):
        for tag in DECOYS:
            v = by[f"{run}/{tag}"]
            assert v["valid"] == ("0" if tag in FIRST_FAILURE else "1"), (tag, v)
            assert v["invalid_by"] == FIRST_FAILURE.get(tag, ""), (tag, v)
    assert by["r01/S_000002"]["hands_L"] == "14" and "0 D where 14 are expected" in by["r01/S_000002"][
        "invalid_reason"]
    assert by["r01/S_000003"]["base_errors"] == "1" and "B:14 A for C" in by["r01/S_000003"]["invalid_reason"]
    assert float(by["r01/S_000004"]["o3p_max"]) > 2.0 and by["r01/S_000001"]["o3p_max"] == "1.594"
    assert by["r01/S_000005"]["pairs_formed"] == "0" and by["r01/S_000001"]["pairs_formed"] == "3"
    assert table.select({"valid": 0})[0] and len(table.select({"linear_chainbreak": 5})[0]) == 10


def test_validity_alone_clusters_the_valid_set_and_score_cut_before_only_when_chosen(tmp_path):
    # validity filters without a quality filter: every valid decoy is a candidate representative
    plan, mid = _plan(tmp_path, n_models=3, dssr=False, cluster=False, valid_closure=True)
    sh = generate_step(plan.steps[mid])[0].content
    assert "keep --valid" in sh and "filter-ranking" not in sh
    plan.write([mid])
    proc, folder = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    ranked = [ln.split()[2] for ln in (folder / f"apt1_02_models_{STAMP}.txt").read_text().splitlines()]
    assert ranked == ["S_000002", "S_000002", "S_000003"]  # closure alone: S_000004 left out
    report = (folder / f"apt1_02_validity_{STAMP}.txt").read_text()
    assert re.search(r"after closure\s+10", report)
    # the score cut before clustering: explicit, warned, applied to the valid decoys
    plan, mid = _plan(tmp_path, n_models=2, dssr=False, cluster=False, valid_closure=True, filter=True,
                      filter_limits="score >= -40", filter_when="before")
    assert any("changes the cluster populations" in i.message for i in plan.steps[mid].issues)
    sh = generate_step(plan.steps[mid])[0].content
    assert "keep --valid apt1_02_validity_20261015-1432.tsv --limit score '>=' -40.0" in sh
    plan.write([mid])
    proc, folder = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    ranked = [ln.split()[2] for ln in (folder / f"apt1_02_models_{STAMP}.txt").read_text().splitlines()]
    assert ranked == ["S_000005", "S_000005"]  # -35 is the best valid score >= -40
    assert "outside score >= -40" in proc.stdout


def test_defaults_have_no_filter_and_old_settings_migrate(tmp_path):
    plan, mid = _plan(tmp_path, n_models=2)
    st = plan.steps[mid]
    sh = generate_step(st)[0].content
    assert not st.settings["filter"] and not TOOLS["farfar2-models"].validity_on(st.settings)
    assert "validity" not in sh and " keep" not in sh and "filter" not in sh and "validity" not in st.outputs
    # the single score filter of earlier 0.0.12 builds becomes a quality limit (its 'before' choice stays and
    # is warned about)
    s = TOOLS["farfar2-models"].normalise({"filter": True, "filter_term": "elec_dens_fast", "filter_max": -40,
                                           "filter_when": "before"})
    assert s["filter_limits"] == "elec_dens_fast <= -40" and "filter_term" not in s and "filter_max" not in s
    assert s["filter_when"] == "before"
    assert TOOLS["farfar2-models"].normalise({})["filter_when"] == "after"


def test_planner_checks_of_the_validity_options(tmp_path):
    def errors(**settings):
        plan, mid = _plan(tmp_path, **settings)
        return [i.message for i in plan.steps[mid].issues if i.level == "error"]

    assert errors(valid_hands=True) and "expected counts" in errors(valid_hands=True)[0]
    assert "cannot read the hand counts" in errors(valid_hands=True, hands_counts="44 Q")[0]
    assert "cannot read residue range" in errors(valid_hands=True, hands_l="C:x")[0]
    assert "no residue group 'Target'" in errors(valid_hands=True, hands_l="group:Target")[0]
    assert not errors(valid_hands=True, hands_counts="44 D / 29 L", hands_l="C:*")
    assert "defining pairs" in errors(valid_pairs=True)[0]
    assert "cannot read the pair" in errors(valid_pairs=True, register="x=B:22+B:37")[0]
    assert "one term" in errors(valid_closure=True, closure_term="")[0]
    assert not errors(valid_closure=True, closure_term="", closure_mode="geometry")
    # base identity needs the sequence: typed, or from the FARFAR2 step
    assert not errors(valid_bases=True)
    no_sequence = {"make_inputs": False, "fasta_file": "/nowhere/x.fasta"}
    assert "needs the sequence" in errors(valid_bases=True, fold=no_sequence)[0]
    assert not errors(valid_bases=True, fold=no_sequence, valid_sequence=">x A:1-7\nGGGAAAA\n")
    plan, mid = _plan(tmp_path, valid_hands=True, hands_counts="44 D / 29 L", hands_l="C:*", hands_d="A, B",
                      valid_pairs=True, register=STEM, pairs_mode="approached", valid_bases=True)
    settings = TOOLS["farfar2-models"].validity_settings(plan.steps[mid].settings, plan.steps[mid].ctx)[0]
    assert settings["hands"] == {"counts": {"D": 44, "L": 29}, "expect_l": "C", "expect_d": "A, B"}
    assert settings["pairs"]["pairs"][0] == (("A", 1, ""), ("B", 14, "")) and settings["pairs"]["mode"] == \
        "approached"
    assert settings["pairs"]["cut"] == ensemble.PAIR_CUT and settings["pairs"]["near_c1"] == ensemble.NEAR_C1
    assert settings["bases"]["reference"][:2] == [("A", 1, "G"), ("A", 2, "G")]


# ---------------------------------------------------------------------------
# the criteria
# ---------------------------------------------------------------------------


def test_hand_counts_and_ranges_on_a_mixed_model():
    d = builder.build_helix("GGGAAAA", chains=("A", "B"), start=(1, 8))
    lrna = builder.build_helix("GCGC", chains=("C", "D"), start=(15, 19), chirality="L")
    s = pdbtools.read_pdb(joined(d, lrna))
    ok, cols, why = decoy_validity.judge_hands(s, {"counts": {"D": 14, "L": 8}, "expect_l": "C, D"})
    assert ok and cols == {"hands_D": 14, "hands_L": 8, "hands_undetermined": 0}, why
    ok, _c, why = decoy_validity.judge_hands(s, {"counts": {"D": 44, "L": 29}})
    assert not ok and why == "14 D where 44 are expected; 8 L where 29 are expected"
    ok, _c, why = decoy_validity.judge_hands(s, {"expect_l": decoy_validity.whole_chains("A:*")})
    assert not ok and "7 expected L measured D (A:1-7)" in why
    # a sugar that cannot be measured fails the counts (its hand is not known)
    cut = "\n".join(ln for ln in joined(d, lrna).splitlines()
                    if not (ln.startswith("ATOM") and ln[21] == "C" and ln[22:26].strip() == "15"
                            and ln[12:16].strip() in ("C4'", "C3'", "C1'")))
    ok, cols, why = decoy_validity.judge_hands(pdbtools.read_pdb(cut), {"counts": {"D": 14, "L": 8}})
    assert not ok and cols["hands_undetermined"] == 1 and "cannot be measured (C:15)" in why
    assert decoy_validity.parse_hand_counts("D=44, L=29") == {"D": 44, "L": 29} == \
        decoy_validity.parse_hand_counts("44 D / 29 L") == decoy_validity.parse_hand_counts("44d 29l")
    with pytest.raises(ValueError):
        decoy_validity.parse_hand_counts("44 D / 30 D")


def test_bases_against_the_sequence():
    s = pdbtools.read_pdb(DUPLEX)
    ref = ensemble.parse_reference(">c A:1-7\nGGGAAAA\n>c B:8-14\nUUUUCCC\n")
    spec = {"reference": list(zip(ref.chains, ref.numbers, ref.bases))}
    assert decoy_validity.judge_bases(s, spec) == (True, {"base_errors": 0}, "")
    longer = ensemble.parse_reference(">c A:1-8\nGGGAAAAG\n>c B:9-15\nUUUUCCC\n")  # numbered differently
    ok, cols, why = decoy_validity.judge_bases(s, {"reference": list(zip(longer.chains, longer.numbers,
                                                                         longer.bases))})
    assert not ok and "missing" in why and "does not have (B:8)" in why
    plain = ensemble.parse_reference("GGGAAAA UUUUCCC")  # no chains: numbers only
    assert decoy_validity.judge_bases(
        s, {"reference": list(zip(plain.chains, plain.numbers, plain.bases))})[0]


def test_closure_by_term_and_geometry():
    broken = pdbtools.read_pdb(DECOYS["S_000004"][0])
    spec = {"term": "linear_chainbreak", "max": 5.0}
    assert decoy_validity.judge_closure(broken, {"linear_chainbreak": "5.0"}, spec)[0]  # at most the maximum
    ok, _c, why = decoy_validity.judge_closure(broken, {"linear_chainbreak": "nan"}, spec)
    assert not ok and why == "no value of linear_chainbreak"
    ok, cols, why = decoy_validity.judge_closure(broken, {}, {"o3p": 2.0})
    assert not ok and "1 O3'-P link(s) longer than 2 A (A:3-A:4" in why and float(cols["o3p_max"]) > 2
    assert decoy_validity.judge_closure(pdbtools.read_pdb(DUPLEX), {}, {"o3p": 2.0})[0]


@pytest.mark.parametrize("dx", [0.0, 1.5, 2.5, 3.5, 5.0, 8.0])
def test_pairs_counted_as_the_register_census_counts_them(tmp_path, dx):
    reg = ensemble.parse_register(STEM)
    spec = {"name": reg.name, "pairs": [tuple(p) for p in reg.pairs], "mode": "formed",
            "cut": ensemble.PAIR_CUT, "near_n": ensemble.NEAR_N, "near_c1": tuple(ensemble.NEAR_C1),
            "wc_atom": dict(ensemble.WC_ATOM)}
    text = moved(DUPLEX, dx, chain="B")
    ok, cols, why = decoy_validity.judge_pairs(pdbtools.read_pdb(text), spec)
    census = ensemble.scan_decoy(write(tmp_path / "d.pdb", text), [reg])
    assert (cols["pairs_formed"], cols["pairs_approached"]) == (census.formed["stem"],
                                                                census.approached["stem"])
    assert ok == (census.formed["stem"] == 3)
    approached = decoy_validity.judge_pairs(pdbtools.read_pdb(text), dict(spec, mode="approached"))[0]
    assert approached == (census.approached["stem"] == 3)


def test_pairs_without_a_residue_fail_with_the_reason():
    reg = ensemble.parse_register("x=A:1-C:14")
    spec = {"name": "x", "pairs": [tuple(p) for p in reg.pairs], "cut": 3.5, "near_n": 6.0,
            "near_c1": (8, 13), "wc_atom": dict(ensemble.WC_ATOM)}
    assert decoy_validity.judge_pairs(pdbtools.read_pdb(DUPLEX), spec) == (
        False, {"pairs_formed": 0, "pairs_approached": 0}, "no residue C:14")


# ---------------------------------------------------------------------------
# the silent-file side: the decoys kept for clustering, limits on the ranking, the table's validity columns
# ---------------------------------------------------------------------------


def test_keep_runs_limits_and_validity_columns(tmp_path):
    src = write(tmp_path / "r01.out", silent_text([(tag, v) for tag, (_m, v) in DECOYS.items()]))
    valid = {("r01", "S_000001"): {"valid": "1"}, ("r01", "S_000006"): {"valid": "1"},
             ("r01", "S_000002"): {"valid": "0"}}
    dst = str(tmp_path / "r01_kept.out")
    ((label, kept, invalid, outside, unread),) = silent_scores.keep_runs([("r01", src, dst)], valid,
                                                                         [("score", "<=", -20.0)])
    assert (label, kept, invalid, outside, unread) == ("r01", 1, 4, 1, 0)
    text = open(dst).read()
    assert "S_000001" in text and "S_000006" not in text and text.startswith("SEQUENCE:")
    assert len([ln for ln in text.splitlines() if ln.endswith("S_000001")]) == 3  # every line of the decoy
    ranking = write(tmp_path / "ranking.txt",
                    "".join(f"{v[0]} {tag} {src}\n" for tag, (_m, v) in DECOYS.items()))
    assert silent_scores.filter_ranking(ranking, limits=[("score", "<=", -20.0),
                                                         ("linear_chainbreak", "<", 1.0)]) == (4, 2, 0)
    assert [ln.split()[1] for ln in open(ranking)] == ["S_000001", "S_000002", "S_000003", "S_000005"]
    with pytest.raises(ValueError, match="no column 'elec_dens_fast'"):
        silent_scores.filter_ranking(ranking, limits=[("elec_dens_fast", "<=", 0.0)])
    v = write(tmp_path / "validity.tsv",
              "run\ttag\tvalid\tinvalid_by\nr01\tS_000001\t1\t\nr01\tS_000002\t0\thands\n")
    out = str(tmp_path / "table.tsv")
    silent_scores.write_table(out, silent_scores.read_runs([("r01", src)]),
                              extra=silent_scores.read_validity(v))
    rows = {r.tag: r.values for r in rosetta.read_decoy_table(out).rows}
    assert rows["S_000002"]["invalid_by"] == "hands" and rows["S_000003"]["valid"] == ""
    assert open(out).readline().split("\t")[-2:] == ["valid", "invalid_by\n"]


def test_the_modules_copied_into_the_validity_helper_do_not_shadow_each_other():
    # the helper is one script: pdbtools, modelcheck, silent_scores and decoy_validity share one namespace,
    # so a function defined in two of them would silently replace the other's (pdbtools' and silent_scores'
    # main() are command-line entry points that the helper never calls)
    import ast
    import inspect

    from mdatk import modelcheck
    from mdatk.workflows.scripts import decoy_validity as dv

    def names(module):
        out = set()
        for node in ast.parse(inspect.getsource(module)).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out.add(node.target.id)
        return out

    seen = {}
    clashes = set()
    for module in (pdbtools, modelcheck, silent_scores, dv):
        for name in names(module):
            if name in seen and name not in ("main",):
                clashes.add(f"{name} ({seen[name]} and {module.__name__})")
            seen.setdefault(name, module.__name__)
    assert not clashes, sorted(clashes)
