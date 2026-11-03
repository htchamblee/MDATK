"""Decoy ensembles, map-model scores and their planner steps (mdatk.ensemble_stats, mdatk.ensemble,
mdatk.mapscore, mdatk.workflows.ensemblestep), on synthetic D / L duplexes built by mdatk.builder.

Regression tests for the bugs of the analysis scripts these modules replace are marked "regression"."""

import math
import os
import random
import stat
import subprocess
import sys

import numpy as np
import pytest

from mdatk import builder, ensemble, mapscore, pdbtools
from mdatk import ensemble_stats as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = "20261015-1432"


# --- helpers -------------------------------------------------------------------------------------------

def moved(text, chain=None, dz=0.0, residues=None, dx=0.0):
    """PDB text with the atoms of ``chain`` (or of the (chain, resseq) in ``residues``) shifted by dx, dz."""
    s = pdbtools.read_pdb(text)
    for res in s.residues():
        hit = (res.chain, res.resseq) in residues if residues is not None else res.chain == chain
        if hit:
            for a in res.atoms:
                x, y, z = a.xyz
                a.tail = f"   {x + dx:8.3f}{y:8.3f}{z + dz:8.3f}" + a.tail[27:]
    return pdbtools.write_pdb(s)


def write(path, text):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return str(path)


DUPLEX = builder.build_helix("GGGAAAA", chains=("A", "B"), start=(1, 8))  # A 1-7, B 8-14 (UUUUCCC)
PER_CHAIN = builder.build_helix("GGGAAAA", chains=("A", "B"), start=(1, 1))  # A 1-7, B 1-7
REFERENCE = ">c A:1-7\nGGGAAAA\n>c B:8-14\nUUUUCCC\n"
STEM = "stem=A:1-B:14, A:2-B:13, A:3-B:12"
TIP = "tip=A:4-B:11, A:5-B:10"


# --- ensemble_stats ------------------------------------------------------------------------------------

def test_fisher_exact_known_values_and_scipy():
    assert S.fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857142857142857)
    assert S.fisher_exact(0, 0, 0, 0) == 1.0
    assert S.fisher_exact(10, 0, 0, 10) == pytest.approx(1.0825088224469026e-05)
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(7)
    for _ in range(200):
        t = [rng.randint(0, 40) for _ in range(4)]
        assert S.fisher_exact(*t) == pytest.approx(stats.fisher_exact([t[:2], t[2:]])[1], rel=1e-9, abs=1e-12)


def test_clopper_pearson_large_n_regression():
    """regression: sweep.py's fallback multiplied math.comb(n, i) by a float and overflowed for n ~ 3000."""
    for k, n in ((0, 10), (10, 10), (3, 10), (30, 300), (300, 3000), (5000, 100000), (1, 1)):
        lo, hi = S.clopper_pearson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((3, 10), (30, 300), (300, 3000), (5000, 100000), (99, 100)):
        lo, hi = S.clopper_pearson(k, n)
        assert lo == pytest.approx(stats.beta.ppf(0.025, k, n - k + 1), abs=1e-10)
        assert hi == pytest.approx(stats.beta.ppf(0.975, k + 1, n - k), abs=1e-10)


def test_wilson_and_binomial_tails():
    lo, hi = S.wilson(30, 300)
    assert lo < 0.1 < hi
    assert S.binom_cdf(5, 10, 0.5) + S.binom_sf(6, 10, 0.5) == pytest.approx(1.0)
    stats = pytest.importorskip("scipy.stats")
    ci = stats.binomtest(30, 300).proportion_ci(method="wilson")
    assert (lo, hi) == pytest.approx((ci.low, ci.high), abs=1e-12)
    for k, n, p in ((3, 40, 0.2), (300, 3000, 0.09), (4500, 50000, 0.1)):
        assert S.binom_cdf(k, n, p) == pytest.approx(stats.binom.cdf(k, n, p), rel=1e-9, abs=1e-15)
        assert S.binom_sf(k, n, p) == pytest.approx(stats.binom.sf(k - 1, n, p), rel=1e-9, abs=1e-15)


def test_mann_whitney_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(20)]
    y = [rng.gauss(0.5, 1) for _ in range(25)]
    u, p, how = S.mann_whitney(x, y)
    ref = stats.mannwhitneyu(x, y, method="asymptotic")
    assert how == "asymptotic" and u == ref.statistic and p == pytest.approx(ref.pvalue, rel=1e-9)
    x, y = [1.2, 3.4, 5.6, 0.1, 2.2], [4.4, 6.6, 7.7, 8.8, 0.5, 9.9]
    u, p, how = S.mann_whitney(x, y)
    ref = stats.mannwhitneyu(x, y, method="exact")
    assert how == "exact" and u == ref.statistic and p == pytest.approx(ref.pvalue, rel=1e-12)
    x, y = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7], [2, 3, 3, 4, 6, 7, 8, 8, 9, 9, 10]  # ties
    u, p, how = S.mann_whitney(x, y)
    ref = stats.mannwhitneyu(x, y)
    assert how == "asymptotic" and u == ref.statistic and p == pytest.approx(ref.pvalue, rel=1e-9)
    with pytest.raises(ValueError):
        S.mann_whitney(x, y, method="exact")


def test_fisher_power_matches_brute_force():
    stats = pytest.importorskip("scipy.stats")
    n1, n2, p1, p2 = 12, 10, 0.2, 0.7
    brute = 0.0
    for x1 in range(n1 + 1):
        for x2 in range(n2 + 1):
            if stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1] <= 0.05:
                brute += stats.binom.pmf(x1, n1, p1) * stats.binom.pmf(x2, n2, p2)
    assert S.fisher_power_exact(p1, n1, p2, n2) == pytest.approx(brute, abs=1e-9)


def test_detectable_rates_replace_the_wilson_band_regression():
    """regression: pilot_compare's 'POWER' band (the Wilson interval around the baseline at the pilot's n) is
    not a power: a pilot at the band's edge is detected far less often than 80%."""
    lo_band, hi_band = S.wilson(10, 100)
    assert S.fisher_power(0.10, 300, hi_band, 100)[0] < 0.6
    det = S.detectable_rates(0.10, 300, 100, alpha=0.05, power=0.8)
    assert det["method"] == "exact" and det["upper"] > hi_band + 0.03 and det["lower"] < 0.1
    assert S.fisher_power(0.10, 300, det["upper"], 100)[0] == pytest.approx(0.8, abs=0.03)
    assert S.detectable_rates(0.0, 300, 300)["lower"] is None
    big = S.detectable_rates(0.5, 30000, 30000)
    assert big["method"] == "normal" and big["lower"] < 0.5 < big["upper"]


def test_percentile_and_median():
    assert S.median([3, 1, 2, float("nan")]) == 2
    assert S.percentile([1, 2, 3, 4, 5], 5) == pytest.approx(np.percentile([1, 2, 3, 4, 5], 5))
    assert S.median([]) is None


# --- parsing --------------------------------------------------------------------------------------------

def test_registers_selections_and_offsets():
    reg = ensemble.parse_register("pres=B:22-B:37, B:23-B:36;B:24-B:35")
    assert reg.name == "pres" and reg.pairs[0] == (("B", 22, ""), ("B", 37, "")) and len(reg.pairs) == 3
    assert ensemble.parse_register("A:1-A:9").name == "register"
    for bad in ("x=22-37", "x=A:1-A:1", "x=", "bad name=A:1-A:2"):
        with pytest.raises(ensemble.EnsembleError):
            ensemble.parse_register(bad)
    with pytest.raises(ensemble.EnsembleError):
        ensemble.check_registers([reg, reg])
    with pytest.raises(ensemble.EnsembleError):
        ensemble.check_registers([ensemble.parse_register("x=A:1-A:9, A:1-A:8")])
    sel = ensemble.parse_selection("A:1-16,B:17-44 C")
    assert sel == [("A", 1, 16), ("B", 17, 44), ("C", None, None)]
    assert ensemble.in_selection(("B", 20, ""), sel) and not ensemble.in_selection(("A", 17, ""), sel)
    groups = [{"name": "Aptamer", "residues": "1-44, 50"}, {"name": "Target", "residues": "C:45-73"}]
    g = ensemble.parse_selection("group:Aptamer group:Target", groups)
    assert g == [(None, 1, 44), (None, 50, 50), ("C", 45, 73)]
    assert ensemble.selection_text(g) == "*:1-44,*:50,C:45-73"
    assert ensemble.parse_selection(ensemble.selection_text(g)) == g
    with pytest.raises(ensemble.EnsembleError):
        ensemble.parse_selection("group:Nope", groups)
    assert ensemble.parse_offsets("B:-367, C:44") == {"B": -367, "C": 44}


def test_reference_forms():
    r = ensemble.parse_reference(REFERENCE)
    assert r.bases == "GGGAAAAUUUUCCC" and r.chains[7] == "B" and r.numbers[7] == 8
    assert r.base_at(("B", 14, "")) == "C" and r.position(("A", 14)) is None
    plain = ensemble.parse_reference("GGGCA AAG")
    assert plain.chains == [None] * 8 and plain.numbers == list(range(1, 9))
    assert plain.base_at(("Q", 4, "")) == "C"  # chain open: any chain
    pre = ensemble.parse_reference("A: GGG B:CCC")
    assert pre.chains == ["A"] * 3 + ["B"] * 3 and pre.numbers == [1, 2, 3, 4, 5, 6]
    again = ensemble.parse_reference(ensemble.reference_text(r))
    assert (again.chains, again.numbers, again.bases) == (r.chains, r.numbers, r.bases)
    with pytest.raises(ensemble.EnsembleError):
        ensemble.parse_reference("GGXQ")
    from mdatk.workflows import rna
    m = ensemble.reference_from_model(rna.build_model("A: GGGAAAA\nB: UUUUCCC"))
    assert (m.chains, m.numbers, m.bases) == (r.chains, r.numbers, r.bases)


def test_base_names_and_atom_inference():
    """regression: the scripts mapped names through private tables ('LG0' read as base '0', an unknown
    name read as ''); MDATK's parser names the base, and one diagnostic atom settles the rest."""
    s = pdbtools.read_pdb(builder.build_helix("GACU", chirality="L"))
    names = {"G": "LG0", "A": "0", "C": "C0X", "U": "LU0"}
    for res in s.residues():
        res.name = names.get(ensemble.residue_base(res), res.name)
    assert [ensemble.residue_base(r) for r in s.residues()][:4] == list("GACU")
    s2 = pdbtools.read_pdb(builder.build_helix("GA", chirality="L", scheme="amber"))
    assert [ensemble.residue_base(r) for r in s2.residues()] == list("GAUC")


# --- numbering ---------------------------------------------------------------------------------------------

def test_chains_located_in_the_reference():
    ref = ensemble.parse_reference(REFERENCE)
    keymap, notes = ensemble.map_keys(pdbtools.read_pdb(PER_CHAIN), ref)
    assert keymap[("B", 1, "")] == ("B", 8, "") and keymap[("A", 7, "")] == ("A", 7, "")
    assert any("offset +7" in n for n in notes)
    plain = ensemble.parse_reference("GGGAAAAUUUUCCC")
    keymap, _ = ensemble.map_keys(pdbtools.read_pdb(PER_CHAIN), plain)
    assert keymap[("B", 7, "")] == ("B", 14, "")  # chain kept from the decoy
    keymap, notes = ensemble.map_keys(pdbtools.read_pdb(PER_CHAIN), ensemble.parse_reference("CCCCCCCCCC"))
    assert keymap[("B", 1, "")] == ("B", 1, "") and all("not in the reference" in n for n in notes)
    keymap, _ = ensemble.map_keys(pdbtools.read_pdb(PER_CHAIN), None, {"B": 7})
    assert keymap[("B", 2, "")] == ("B", 9, "")
    # a missing residue (numbering gap) still locates the chain
    s = pdbtools.read_pdb(PER_CHAIN)
    s.segments[1] = [r for r in s.segments[1] if r.resseq != 3]
    keymap, _ = ensemble.map_keys(s, ref)
    assert keymap[("B", 4, "")] == ("B", 11, "")


# --- scores ----------------------------------------------------------------------------------------------

def test_decoy_table_reader_and_index(tmp_path):
    root = tmp_path / "plan"
    pdb = write(root / "05_models" / "m1.pdb", DUPLEX)
    write(root / "05_models" / "S_000002_7.pdb", DUPLEX)
    t = write(root / "05_models" / "scores.tsv",
              "RUN\tTag\tPDB\tscore\tlinear_chainbreak\textra\n"
              "r01\tS_000001_3\t05_models/m1.pdb\t-120.5\t0.4\tx\n"
              "r01\tS_000002_7\t\t-99\t7.5\n"
              "# a comment\n")
    table = ensemble.read_decoy_table(t)
    assert table.columns[:3] == ["run", "tag", "pdb"] and "linear_chainbreak" in table.terms
    idx = ensemble.ScoreIndex([table])  # paths from the parent of the table's folder
    assert idx.lookup(pdb)["score"] == "-120.5"
    assert idx.lookup(str(root / "05_models" / "S_000002_7.pdb"))["linear_chainbreak"] == "7.5"  # by tag
    assert idx.models() == [os.path.normpath(pdb)]
    c = write(tmp_path / "x.csv", "run,tag,pdb,score\nr02,t1,m1.pdb,-3\nr02,t1,m1.pdb,-4\n")
    idx = ensemble.ScoreIndex([ensemble.read_decoy_table(c)], root=str(root / "05_models"))
    assert idx.lookup(pdb)["score"] == "-3" and idx.notes  # the repeated row is reported, the first kept
    assert ensemble.run_id("apt1_04_farfar2-r03_20261015-1432.out") == "r03"


def test_embedded_scores():
    text = ("ATOM ...\n#BEGIN_POSE_ENERGIES_TABLE x\nlabel fa_atr fa_rep total\nweights 0.23 0.12 NA\n"
            "pose -10 2 -432.1\n")
    assert ensemble.embedded_score(text) == (-432.1, "0.23 0.12 NA")
    assert ensemble.embedded_score("", "denovo_95680_S_1.pdb") == (-432.0, "")


# --- pairs, closure, density ------------------------------------------------------------------------------

def test_closure_by_o3p_within_chain_segments_only():
    """regression: adjacency by residue number across chains; MDATK measures consecutive nucleotides of one
    chain segment only."""
    worst, measured, missing = ensemble.backbone_closure(pdbtools.read_pdb(PER_CHAIN))
    assert measured == 12 and missing == 0 and worst < 1.7  # 6 junctions per strand, none across chains
    broken = moved(DUPLEX, residues={("A", 4), ("A", 5), ("A", 6), ("A", 7)}, dz=8.0)
    assert ensemble.backbone_closure(pdbtools.read_pdb(broken))[0] > 2.5


def test_heavy_atoms_by_element_column():
    """regression: hydrogens recognised by the name alone."""
    s = pdbtools.read_pdb(builder.build_helix("GC", hydrogens=True))
    keyed = {ensemble.residue_key(r): r for r in s.residues()}
    heavy = ensemble.heavy_atom_coords(keyed, ensemble.parse_selection("A"))
    n_heavy = sum(1 for r in s.residues() if r.chain == "A" for a in r.atoms if a.element != "H")
    assert len(heavy) == n_heavy
    atom = pdbtools.Atom("ATOM", "HG", "", " " * 49, element="HG")  # mercury, named like a hydrogen
    assert not ensemble.is_hydrogen(atom)


def _map_of(text, tmp_path, name="map.mrc", resolution=4.0):
    from mdatk import density
    from mdatk.io import mrc
    s = pdbtools.read_pdb(text)
    atoms = [(a.xyz, a.element or a.name[0]) for r in s.residues() for a in r.atoms]
    dmap = density.simulate_map(np.array([a[0] for a in atoms]), density.atom_weights([a[1] for a in atoms]),
                                resolution, spacing=1.0)
    path = str(tmp_path / name)
    mrc.write_map(path, dmap)
    return path


def test_occupancy_and_contour(tmp_path):
    path = _map_of(DUPLEX, tmp_path)
    dmap = ensemble.load_map(path)
    keyed = {ensemble.residue_key(r): r for r in pdbtools.read_pdb(DUPLEX).residues()}
    calib_sel = ensemble.parse_selection("A:1-3,B:12-14")
    calib = ensemble.map_values(dmap, ensemble.heavy_atom_coords(keyed, calib_sel))
    level = ensemble.contour_level(calib)
    inside = ensemble.map_values(dmap, ensemble.heavy_atom_coords(keyed, ensemble.parse_selection("A:4-7")))
    assert ensemble.occupancy(inside, level)[0] > 0.9
    away = pdbtools.read_pdb(moved(DUPLEX, "B", 40.0))
    keyed2 = {ensemble.residue_key(r): r for r in away.residues()}
    out_vals = ensemble.map_values(dmap, ensemble.heavy_atom_coords(keyed2, ensemble.parse_selection("B")))
    occ, outside = ensemble.occupancy(out_vals, level)
    assert occ == 0.0 and outside > 0  # atoms outside the map count as not in density
    near = ensemble.map_values(dmap, ensemble.heavy_atom_coords(keyed, ensemble.parse_selection("A:4-7")),
                               "nearest")
    assert np.allclose(near, inside, atol=0.25 * np.nanmax(inside))
    assert ensemble.contour_level(np.array([np.nan])) is None


# --- build conditions -------------------------------------------------------------------------------------

REGS = [ensemble.parse_register(STEM), ensemble.parse_register(TIP)]


def test_secstruct_verdict_needs_the_defining_pairs_regression():
    """regression: register_census called a secstruct FORCED when '(((' and ')))' stood at the register's
    positions even if they paired with other residues."""
    ref = ensemble.parse_reference(REFERENCE)
    v, _ = ensemble.judge("(((.... ....)))", [], REGS, ref)
    assert v == {"stem": "FORCED", "tip": "UNFORCED"}
    v, _ = ensemble.judge("((((... ...))))", [], REGS, ref)
    assert v == {"stem": "FORCED", "tip": "PARTIAL"}  # 4-11 paired, 5-10 free
    v, _ = ensemble.judge("((((... ...))))", [], [ensemble.parse_register("x=A:1-B:13, A:2-B:12")], ref)
    assert v == {"x": "PARTIAL"}  # brackets at 1-2 / 12-13, but pairing 1-14 and 2-13
    v, _ = ensemble.judge("(((((.. ..)))))", [], REGS, ref)
    assert v == {"stem": "FORCED", "tip": "FORCED"}
    v, notes = ensemble.judge("(((....)))", [], REGS, ref)
    assert v == {} and "positions" in notes[0]  # wrong length: no verdict
    v, _ = ensemble.judge(None, [(4, 11), (5, 10)], REGS, ref)
    assert v == {"stem": "UNFORCED", "tip": "FORCED"}


def test_condition_from_a_rosetta_log_with_flags_files(tmp_path):
    """The MDATK FARFAR2 layout: the log in the step folder, the run in runs/r01, @flags and the secstruct
    relative to the run folder; obligate pairs as residue numbers of the FASTA's tags."""
    step = tmp_path / "plan" / "04_farfar2"
    os.makedirs(step / "runs" / "r01")
    write(step / "x.fasta", REFERENCE)
    write(step / "x.secstruct", "....... .......\nGGGAAAA UUUUCCC\n")
    write(step / "x.flags", "# flags\n-fasta ../../x.fasta\n-rna:denovo:secstruct_file=../../x.secstruct\n"
          "-obligate_pair_explicit 4 11 W W A 5 10 W W A  # tip\n")
    write(step / "x-r01.log", "core.init: Rosetta version ...\ncore.init: command: rna_denovo @../../x.flags "
          "-out:file:silent ../../x-r01.out -run:jran 3001\n")
    models = tmp_path / "plan" / "05_models"
    write(models / "d1.pdb", DUPLEX)
    c = ensemble.detect_condition(str(models), [str(models)], REGS, None, [str(step)])
    assert c.verdicts == {"stem": "UNFORCED", "tip": "FORCED"} and c.source == "log"
    assert "obligate_pair_explicit (2 pairs)" in c.detail and "x-r01.log" in c.detail
    # two logs that disagree leave the group UNKNOWN
    write(step / "y.secstruct", "(((.... ....)))\n")
    write(step / "x-r02.log", "core.init: command: rna_denovo -fasta x.fasta -secstruct_file y.secstruct\n")
    c = ensemble.detect_condition(str(models), [str(models)], REGS, None, [str(step)])
    assert not c.known and "disagree" in c.detail
    seed = ensemble.run_seeds([str(step)])["r01"]  # -jran without -constant_seed: Rosetta ignores it
    assert (seed.jran, seed.constant, seed.value) == ("3001", False, None)


def test_condition_details_from_the_command_line(tmp_path):
    ref = ensemble.parse_reference(REFERENCE)
    v, detail = ensemble.condition_from_command(
        "rna_denovo.default.linuxgccrelease -secstruct '(((.... ....)))' -s h.pdb -input_res A:1-3 B:12-14",
        [str(tmp_path)], REGS, ref)
    assert v == {"stem": "FORCED", "tip": "UNFORCED"} and "held by the -s structure" in detail
    v, detail = ensemble.condition_from_command("rna_denovo -obligate_pair A:4 B:11 A:5 B:10", [str(tmp_path)],
                                                REGS, ref)
    assert v == {"stem": "UNFORCED", "tip": "FORCED"} and "all unpaired" in detail
    log = write(tmp_path / "other.log", "make: command: gcc -o x x.c\n")  # not a Rosetta run
    assert ensemble.command_line(log) is None


def test_condition_without_a_source_is_unknown(tmp_path):
    d = tmp_path / "g"
    write(d / "a.pdb", DUPLEX)
    c = ensemble.detect_condition(str(d), [str(d)], REGS)
    assert not c.known and c.label() == "UNKNOWN"
    ss = write(tmp_path / "elsewhere" / "given.secstruct", "(((.... ....)))\n")
    ref = ensemble.parse_reference(REFERENCE)
    c = ensemble.detect_condition(str(d), [str(d)], REGS, ref, secstructs=[ss])
    assert c.verdicts["stem"] == "FORCED" and c.source == "given"


def _farfar2_layout(tmp_path, extract_logs=1):
    """MDATK's FARFAR2 step folder: the rna_denovo log beside @flags, FASTA and secstruct (which imposes the
    stem), the run in runs/r01, the decoys extracted into models/r01 with the extract_pdbs log(s) in
    models/."""
    ff = tmp_path / "plan" / "04_farfar2"
    write(ff / "x.fasta", REFERENCE)
    write(ff / "x.secstruct", "(((.... ....)))\n")
    write(ff / "x.flags", "-fasta ../../x.fasta\n-secstruct_file ../../x.secstruct\n")
    write(ff / f"x_04_farfar2-r01_{STAMP}.log", "core.init: Rosetta version x\ncore.init: command: "
          "rna_denovo.default.linuxgccrelease @../../x.flags -out:file:silent ../../x-r01.out\n")
    os.makedirs(ff / "runs" / "r01")
    for k in range(1, extract_logs + 1):
        write(ff / "models" / f"extract-r{k:02d}.log", "core.init: Rosetta version x\ncore.init: command: "
              f"extract_pdbs.default.linuxgccrelease -in:file:silent ../../x-r{k:02d}.out\n")
    for k in range(3):
        write(ff / "models" / "r01" / f"S_{k}.pdb", DUPLEX)
    return ff


def test_condition_only_from_rna_denovo_logs_regression(tmp_path):
    """regression: an extract_pdbs log beside the extracted decoys (models/extract-rNN.log) was read as the
    build's: no secondary structure on its command line made the imposed stem UNFORCED, or the group
    UNKNOWN when the rna_denovo log was found too; 60 extract logs filled the cap on the logs read."""
    ff = _farfar2_layout(tmp_path)
    assert ensemble.command_line(str(ff / "models" / "extract-r01.log")) is None
    group = str(ff / "models" / "r01")
    c = ensemble.detect_condition(group, [group], REGS[:1])
    assert not c.known and "UNFORCED" not in c.detail  # no build log near the decoys: unknown, not UNFORCED
    c = ensemble.detect_condition(group, [group], REGS[:1], secstructs=[str(ff / "x.secstruct")])
    assert c.label() == "stem FORCED" and c.source == "given"
    c = ensemble.detect_condition(group, [group], REGS[:1], log_dirs=[str(ff)])
    assert c.label() == "stem FORCED" and c.source == "log" and f"x_04_farfar2-r01_{STAMP}.log" in c.detail
    ff = _farfar2_layout(tmp_path / "many", extract_logs=61)
    group = str(ff / "models" / "r01")
    c = ensemble.detect_condition(group, [group], REGS[:1], log_dirs=[str(ff)])
    assert c.label() == "stem FORCED" and c.source == "log"


def test_seeds_compared_as_rosetta_used_them_regression(tmp_path):
    """regression: the pilot's seed check compared the -jran text, so runs without -constant_seed (Rosetta
    then ignores -jran) or with another -seed_offset 'shared their seed'."""
    logs = {
        "r01": ("-run:jran 3001", "'RNG device' seed mode, using '/dev/urandom', seed={} seed_offset=0 "
                                  "real_seed={}", (-1093216543, 771122334)),
        "r02": ("-constant_seed -jran 5000 -seed_offset {}", "Constant seed mode, seed=5000 seed_offset={} "
                                                             "real_seed={}", ((0, 5000), (7, 5007))),
        "r03": ("-run:constant_seed -run:jran 7000", "", (None, None)),  # no seed line: from the flags
    }
    for k, arm in enumerate("ab"):
        for run, (flags, line, values) in logs.items():
            v = values[k]
            if run == "r02":
                flags, line = flags.format(v[0]), line.format(*v)
            elif run == "r01":
                line = line.format(v, v)
            write(tmp_path / arm / f"x-{run}.log", f"core.init: command: rna_denovo {flags}\n"
                  + (f"core.init: {line}\n" if line else ""))
        write(tmp_path / arm / "extract-r04.log", "core.init: command: extract_pdbs -constant_seed\n")
    seeds = ensemble.run_seeds([str(tmp_path / "a")])
    assert sorted(seeds) == ["r01", "r02", "r03"]  # the extract_pdbs log is not a run's
    assert (seeds["r01"].constant, seeds["r02"].value, seeds["r03"].value) == (False, 5000, 7000)
    note = ensemble.seed_note([str(tmp_path / "a")], [str(tmp_path / "b")], ["r01", "r02", "r03", "r09"])
    assert note.startswith("seeds: 1 runs share their seed, 1 do NOT (r02)"), note
    assert "1 have a random seed (no -constant_seed" in note and "(r01)" in note
    assert "1 not found in the logs" in note


# --- census ----------------------------------------------------------------------------------------------

def _groups(tmp_path):
    """forced/: 4 of 6 decoys form both registers; forced_free/: per-chain numbering, 2 of 5 form both."""
    g1, g2 = tmp_path / "runs" / "forced", tmp_path / "runs" / "forced_free"
    for k in range(6):
        write(g1 / f"d{k}.pdb", DUPLEX if k < 4 else moved(DUPLEX, "B", 12.0))
    for k in range(5):
        write(g2 / f"d{k}.pdb", PER_CHAIN if k < 2 else moved(PER_CHAIN, "B", 15.0))
    write(g1 / "x.secstruct", "(((.... ....)))\n")
    write(g2 / "x.secstruct", "....... .......\n")
    return g1, g2


def test_census_groups_conditions_and_matched_pairs(tmp_path):
    g1, g2 = _groups(tmp_path)
    ref = ensemble.parse_reference(REFERENCE)
    res = ensemble.census(ensemble.expand_models([str(g1), str(g2)]), REGS, reference=ref, chirality=True,
                          expect_l=0, pair_suffix="_free")
    by = {os.path.basename(r.decoy.path) + os.path.basename(r.group): r for r in res.rows}
    assert all(not r.decoy.status for r in res.rows)
    assert by["d0.pdbforced"].decoy.formed == {"stem": 3, "tip": 2}
    assert by["d0.pdbforced_free"].decoy.formed == {"stem": 3, "tip": 2}  # per-chain numbering mapped
    assert by["d4.pdbforced_free"].decoy.formed == {"stem": 0, "tip": 0}
    conds = {os.path.basename(g): c.label() for g, c in res.conditions.items()}
    assert conds == {"forced": "stem FORCED, tip UNFORCED", "forced_free": "stem UNFORCED, tip UNFORCED"}
    text = res.report
    assert "matched groups on '_free'" in text and "Fisher p" in text
    assert "pooled WITHIN each condition" not in text  # two conditions: nothing pooled
    assert "stem                   4  66.67%" in text and "L-handed riboses per decoy: 0-0" in text
    cols, rows = ensemble.census_table_rows(res)
    assert cols[:8] == ["group", "condition", "model", "run", "tag", "score", "weights", "status"]
    assert "stem_all_formed" in cols and "l_riboses" in cols and len(rows) == 11


def test_census_counts_registers_of_any_size_regression(tmp_path):
    """regression: register_census counted a register as formed only at '== 3' pairs."""
    g = tmp_path / "g"
    for k in range(3):
        write(g / f"d{k}.pdb", DUPLEX)
    two = ensemble.parse_register("two=A:1-B:14, A:2-B:13")
    res = ensemble.census(ensemble.expand_models([str(g)]), [two])
    assert "two                    3 100.00%" in res.report


def test_census_keeps_chains_apart_regression(tmp_path):
    """regression: residues keyed by number alone - two chains numbered from 1 overwrote each other."""
    write(tmp_path / "g" / "d.pdb", PER_CHAIN)
    reg = ensemble.parse_register("s=A:1-B:7, A:2-B:6, A:3-B:5")  # the decoy's own per-chain numbering
    res = ensemble.census([str(tmp_path / "g" / "d.pdb")], [reg])
    assert res.rows[0].decoy.formed == {"s": 3}


def test_census_base_check_and_uncountable_decoys(tmp_path):
    g = tmp_path / "g"
    write(g / "ok.pdb", DUPLEX)
    write(g / "short.pdb", "\n".join(ln for ln in DUPLEX.splitlines() if " B  14" not in ln[17:26]) + "\n")
    wrong = ensemble.parse_reference(">c A:1-7\nGGGAAAA\n>c B:8-14\nUUUUCCG\n")
    res = ensemble.census(ensemble.expand_models([str(g)]), REGS[:1], reference=wrong)
    status = {os.path.basename(r.decoy.path): r.decoy.status for r in res.rows}
    assert status["ok.pdb"].startswith("base mismatch: B:14")
    res = ensemble.census(ensemble.expand_models([str(g)]), REGS[:1])
    status = {os.path.basename(r.decoy.path): r.decoy.status for r in res.rows}
    assert status["short.pdb"] == "no residue B:14" and "1 decoys could not be counted" in res.report


def test_census_score_deciles_within_one_weight_set(tmp_path):
    g = tmp_path / "g"
    for k in range(4):
        w = "0.23 0.12" if k < 2 else "0.5 0.12"
        write(g / f"d{k}.pdb", DUPLEX + f"label fa_atr total\nweights {w}\npose -1 {-100 - k}\n")
    res = ensemble.census(ensemble.expand_models([str(g)]), REGS)
    assert res.weight_sets == ["0.23 0.12", "0.5 0.12"]
    assert res.report.count("by score decile") == 2 and "weight set W1" in res.report


def test_census_pools_only_within_one_condition(tmp_path):
    for name, ss in (("a", "(((.... ....)))"), ("b", "(((.... ....)))"), ("c", "....... .......")):
        write(tmp_path / name / "d.pdb", DUPLEX)
        write(tmp_path / name / "x.secstruct", ss + "\n")
    res = ensemble.census(ensemble.expand_models([str(tmp_path / n) for n in "abc"]), REGS)
    assert "all stem FORCED, tip UNFORCED groups (2)" in res.report
    assert "all stem UNFORCED" not in res.report  # a single group: nothing to pool


def test_census_cli_with_scores_and_parallel_workers(tmp_path, capsys):
    from mdatk.cli import main
    g = tmp_path / "plan" / "05_models"
    paths = [write(g / f"d{k}.pdb", DUPLEX if k % 2 else moved(DUPLEX, "B", 9.0)) for k in range(10)]
    write(g / "scores.tsv", "run\ttag\tpdb\tscore\n" + "".join(
        f"r01\tt{k}\t05_models/d{k}.pdb\t{-k}\n" for k in range(10)))
    out, rep = str(tmp_path / "c.tsv"), str(tmp_path / "c.txt")
    fasta = write(tmp_path / "r.fa", REFERENCE)
    code = main(["ensemble", "census", str(g), "--register", STEM, "--fasta", fasta,
                 "--scores", str(g / "scores.tsv"), "--jobs", "2", "-o", out, "--report", rep])
    assert code == 0
    rows = open(out).read().splitlines()
    assert len(rows) == 11 and "\tt3\t-3\t" in rows[4]
    assert "5  50.00%" in open(rep).read()
    assert len(paths) == 10


def test_census_takes_the_decoys_from_the_table_and_logs_beside_the_runs(tmp_path):
    """The decoy table lists every decoy (pdb relative to the plan folder); a 'run' naming the silent file
    points at the folder of its Rosetta log."""
    from mdatk.cli import main
    plan = tmp_path / "plan"
    ff = plan / "04_farfar2"
    write(ff / "x.secstruct", "(((.... ....)))\n")
    write(ff / "x-r01.out", "SEQUENCE: x\n")
    write(ff / "x-r01.log", "core.init: command: rna_denovo -secstruct_file x.secstruct\n")
    for k in range(3):
        write(plan / "05_models" / f"m{k}.pdb", DUPLEX)
    write(plan / "05_models" / "scores.tsv", "run\ttag\tpdb\tscore\n" + "".join(
        f"04_farfar2/x-r01\tt{k}\t05_models/m{k}.pdb\t{-k}\n" for k in range(3)) + "r02\tt9\t\t-1\n")
    rep = str(tmp_path / "rep.txt")
    assert main(["ensemble", "census", "--register", STEM, "--scores", str(plan / "05_models" / "scores.tsv"),
                 "--root", str(plan), "--models-from-scores", "--report", rep]) == 0
    text = open(rep).read()
    assert "Register census of 3 decoys" in text and "stem FORCED" in text and "x-r01.log" in text


def test_census_counts_a_decoy_listed_twice_once_regression(tmp_path, monkeypatch):
    """regression: with the decoys connected (relative paths) and 'Also count the decoys the decoy table
    lists' (absolute paths), every decoy was counted twice."""
    from mdatk.cli import main
    plan = tmp_path / "plan"
    for k in range(3):
        write(plan / "01_farfar2" / "models" / "r01" / f"S_{k}.pdb", DUPLEX)
    table = write(plan / "01_farfar2" / "scores.tsv", "run\ttag\tpdb\tscore\n" + "".join(
        f"r01\tS_{k}\t01_farfar2/models/r01/S_{k}.pdb\t{-k}\n" for k in range(3)))
    os.makedirs(plan / "02_census")
    monkeypatch.chdir(plan / "02_census")
    rep = str(tmp_path / "rep.txt")
    assert main(["ensemble", "census", "../01_farfar2/models/r01", "--register", STEM, "--scores", table,
                 "--root", "..", "--models-from-scores", "--report", rep]) == 0
    text = open(rep).read()
    assert "Register census of 3 decoys" in text and "3 decoys counted in 1 group(s)" in text


# --- comparison ------------------------------------------------------------------------------------------

def _arms(tmp_path, n_a=6, n_b=6):
    root = tmp_path / "plan"
    away = moved(DUPLEX, "B", dx=10.0)
    a = [write(root / "a" / f"d{k}.pdb", DUPLEX if k < 5 else away) for k in range(n_a)]
    b = [write(root / "b" / f"d{k}.pdb", DUPLEX if k < 2 else away) for k in range(n_b)]
    lines = ["run\ttag\tpdb\tscore\tlinear_chainbreak"]
    for arm, paths in (("a", a), ("b", b)):
        for k, p in enumerate(paths):
            cb = 1.0 if k % 2 == 0 else 9.0
            lines.append(f"r{k % 3 + 1:02d}\t{arm}{k}\t{arm}/d{k}.pdb\t{-50 - k}\t{cb}")
    table = write(root / "scores.tsv", "\n".join(lines) + "\n")
    return a, b, table


def test_compare_arms_report_and_table(tmp_path):
    a, b, table = _arms(tmp_path)
    scores = ensemble.read_score_index([table], root=str(tmp_path / "plan"))
    mapf = _map_of(DUPLEX, tmp_path)
    dmap = ensemble.load_map(mapf)
    reg = ensemble.parse_register(STEM)
    calib, regions = ensemble.parse_selection("A:1-3"), ensemble.parse_selection("A:4-7,B:8-11")
    arms = [ensemble.collect_arm(lbl, paths, reg, scores=scores, closure=ensemble.Closure(), dmap=dmap,
                                 calibration=calib, regions=regions)
            for lbl, paths in (("A", a), ("B", b))]
    ensemble.apply_contour(arms)
    da = arms[0].decoys
    assert [d.formed for d in da] == [3, 3, 3, 3, 3, 0] and [d.closure for d in da][:2] == [1.0, 9.0]
    assert da[0].occupancy > 0.9 and da[5].occupancy < 0.6
    text = ensemble.compare_arms(*arms, mode="sweep", closure=ensemble.Closure())
    assert "closed (linear_chainbreak < 5)" in text and "Fisher p" in text
    assert "power: against A's" in text and "closure threshold sweep" in text and "not corrected" in text
    assert "THE STATISTICS MOVED" not in text and "regenerate" not in text  # numbers, no verdict
    cols, rows = ensemble.compare_table_rows(arms, ensemble.Closure())
    assert len(rows) == 12 and cols[0] == "arm"


def test_compare_counts_decoys_without_atoms_and_empty_arms_regression(tmp_path):
    """regression: decoys without calibration or loop atoms were dropped silently, and an arm without decoys
    divided by zero."""
    a, b, table = _arms(tmp_path, 3, 3)
    scores = ensemble.read_score_index([table], root=str(tmp_path / "plan"))
    dmap = ensemble.load_map(_map_of(DUPLEX, tmp_path))
    reg = ensemble.parse_register(STEM)
    arm = ensemble.collect_arm("A", a, reg, scores=scores, dmap=dmap,
                               calibration=ensemble.parse_selection("C"),
                               regions=ensemble.parse_selection("A:4-7"))
    assert all(d.status == "no calibration atoms" for d in arm.decoys) and len(arm.decoys) == 3
    empty = ensemble.Arm("B", reg, [], [])
    text = ensemble.compare_arms(arm, empty)
    assert "3 decoys: no calibration atoms" in text
    assert "power:" not in text  # no statement without decoys in arm B


def test_compare_cli_pilot_matched_runs_geometry(tmp_path):
    from mdatk.cli import main
    a, b, table = _arms(tmp_path)
    mapf = _map_of(DUPLEX, tmp_path)
    rep, out = str(tmp_path / "r.txt"), str(tmp_path / "t.tsv")
    code = main(["ensemble", "compare", "--mode", "pilot", "--register", STEM, "--labels", "baseline,pilot",
                 "--arm-a", *a, "--arm-b", str(tmp_path / "plan" / "b"), "--scores", table,
                 "--root", str(tmp_path / "plan"), "--closure", "geometry", "--map", mapf,
                 "--calibration", "A:1-3,B:12-14", "--regions", "A:4-7", "--matched-runs", "-o", out,
                 "--report", rep])
    assert code == 0
    text = open(rep).read()
    assert "common contour" in text and "Wilson" in text and "restricted to the 3 runs" in text
    assert "seeds not checked" in text and "O3'-P < 2" in text
    with pytest.raises(ensemble.EnsembleError):
        ensemble.run(_parsed(["ensemble", "compare", "--register", STEM, "--arm-a", *a, "--arm-b", *b]))
    with pytest.raises(ensemble.EnsembleError, match="no column 'chainbreak'"):
        ensemble.run(_parsed(["ensemble", "compare", "--register", STEM, "--arm-a", *a, "--arm-b", *b,
                              "--scores", table, "--term", "chainbreak"]))


def test_matched_runs_without_runs_fail_regression(tmp_path):
    """regression: --matched-runs without the decoy table (the only source of a decoy's run) emptied both
    arms and the comparison of 0 decoys 'succeeded'; the planner accepted the step without the table."""
    a, b, table = _arms(tmp_path, 3, 3)
    with pytest.raises(ensemble.EnsembleError, match="needs the run of every decoy"):
        ensemble.run(_parsed(["ensemble", "compare", "--mode", "pilot", "--register", STEM, "--arm-a", *a,
                              "--arm-b", *b, "--closure", "geometry", "--matched-runs"]))
    rows = open(table).read().splitlines()  # arm B's decoys in runs arm A does not have
    write(table, "\n".join(ln.replace("r0", "r1", 1) if "\tb" in ln else ln for ln in rows) + "\n")
    with pytest.raises(ensemble.EnsembleError, match="no run is in both arms"):
        ensemble.run(_parsed(["ensemble", "compare", "--mode", "pilot", "--register", STEM, "--arm-a", *a,
                              "--arm-b", str(tmp_path / "plan" / "b"), "--scores", table, "--root",
                              str(tmp_path / "plan"), "--closure", "geometry", "--matched-runs"]))
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import PortValue, StepContext
    t = TOOLS["ensemble-compare"]
    arms = {"arm_a": PortValue(a, "pdb"), "arm_b": PortValue(b, "pdb")}
    settings = t.normalise({"register_a": STEM, "closure": "geometry", "mode": "pilot", "matched_runs": True})
    errs = [i.message for i in t.check(settings, StepContext("p", 1, STAMP, t.tag, arms))
            if i.level == "error"]
    assert any("matched runs need the decoy table" in m for m in errs)
    with_table = dict(arms, scores=PortValue([table], "table"))
    errs = [i.message for i in t.check(settings, StepContext("p", 1, STAMP, t.tag, with_table))
            if i.level == "error"]
    assert errs == []


def _parsed(argv):
    from mdatk.cli import build_parser
    return build_parser().parse_args(argv)


# --- map scores --------------------------------------------------------------------------------------------

def test_parse_score_tables():
    per_atom = ("chain,resseq,resname,name,Q\nA,1,G,P,0.5\nA,1,G,C1',0.7\nA,2,C,P,0.4\nB,1,G,P,nan\n")
    got = mapscore.parse_score_table(per_atom)[""]
    assert got[("A", 1, "")] == ("G", pytest.approx(0.6), 2) and math.isnan(got[("B", 1, "")][1])
    stdout = "Starting...\nsome text\n  chain resnum  qscore\n  A  12A  0.81\n  A 13 0.79\nDone\n"
    got = mapscore.parse_score_table(stdout)[""]
    assert got[("A", 12, "A")][1] == 0.81 and len(got) == 2
    assert mapscore.parse_score_table("no table here\n1 2 3\n") is None
    # ChimeraX's qscore CSV (outputFile): 'Number' with insertion codes, "N/A" with its quotes
    chimerax = 'Chain,Number,Name,Qavg,Qworst,Qbackbone,Qsidechain\nA,7B,LG5,0.41,0.1,0.4,N/A\nA,8,G,"N/A",' \
               '"N/A","N/A","N/A"\n'
    got = mapscore.parse_score_table(chimerax)[""]
    assert got[("A", 7, "B")] == ("LG5", 0.41, 1) and math.isnan(got[("A", 8, "")][1])


def test_residue_cc(tmp_path):
    path = _map_of(DUPLEX, tmp_path, resolution=4.0)
    from mdatk import density
    s = pdbtools.read_pdb(DUPLEX)
    cc = mapscore.residue_cc(s, density.load_map_cached(path), 4.0)
    assert len(cc) == 14 and min(v for _, v, _ in cc.values()) > 0.9
    shifted = pdbtools.read_pdb(moved(DUPLEX, residues={("A", 4)}, dz=4.0))
    cc2 = mapscore.residue_cc(shifted, density.load_map_cached(path), 4.0)
    assert cc2[("A", 4, "")][1] < cc[("A", 4, "")][1] - 0.1
    with pytest.raises(ensemble.EnsembleError):
        mapscore.residue_cc(s, density.load_map_cached(path), 0)


def _scores(values):
    return {k: ("G", v, 1) for k, v in values.items()}


def test_regions_reference_excludes_named_regions_regression():
    """regression (q_regions.py): the scaffold and an interface overlapped (both means took the shared
    residues), residues were keyed by number alone, NaN entered the means and 'whole' kept the termini."""
    scores = {}
    for r in range(1, 11):
        scores[("A", r, "")] = 0.80
    for r in range(5, 9):
        scores[("A", r, "")] = 0.20
    scores[("B", 1, "")] = float("nan")
    scores[("B", 2, "")] = 0.5
    regions = [mapscore.parse_region("loop=A:5-8")]
    rows = mapscore.region_rows("m", _scores(scores), regions, ensemble.parse_selection("A:1-10"),
                                termini={("A", 1, ""), ("A", 10, "")})
    by = {r["region"]: r for r in rows}
    assert by["loop"]["mean"] == pytest.approx(0.20) and by["reference"]["mean"] == pytest.approx(0.80)
    assert by["loop"]["delta"] == pytest.approx(-0.60)  # not -0.519 as with the overlap
    assert by["reference"]["residues"] == 4 and by["whole"]["nan"] == 1 and by["whole"]["residues"] == 10
    assert by["loop"]["missing"] == 0
    rows = mapscore.region_rows("m", _scores({("A", 5, ""): 0.2}), regions)
    assert {r["region"]: r for r in rows}["loop"]["missing"] == 3
    # two chains numbered from 1 stay apart
    two = mapscore.region_rows("m", _scores({("A", 1, ""): 0.9, ("B", 1, ""): 0.1}),
                               [mapscore.parse_region("b=B:1")])
    assert two[0]["mean"] == pytest.approx(0.1) and two[1]["mean"] == pytest.approx(0.9)
    order = mapscore.rank_models(rows + two, "loop")
    assert order[0] == "m"
    with pytest.raises(ensemble.EnsembleError):
        mapscore.parse_region("whole=A:1")


def test_regions_cli_on_headless_csvs(tmp_path, capsys):
    from mdatk.cli import main
    for name, loop in (("d1", 0.3), ("d2", 0.6)):
        rows = [f"A,{r},G,{loop if 5 <= r <= 8 else 0.8}\n" for r in range(1, 13)]
        write(tmp_path / f"{name}_q.csv", "".join(rows))
    out = str(tmp_path / "r.tsv")
    assert main(["ensemble", "regions", str(tmp_path / "d1_q.csv"), str(tmp_path / "d2_q.csv"),
                 "--region", "loop=A:5-8", "--sort", "delta:loop", "-o", out]) == 0
    text = capsys.readouterr().out
    assert text.index("d2_q.csv") < text.index("d1_q.csv") and "chain termini left out" in text
    assert "rank" in open(out).readline()


def test_map_score_cli_cc(tmp_path, capsys):
    from mdatk.cli import main
    mapf = _map_of(DUPLEX, tmp_path)
    m1 = write(tmp_path / "m1.pdb", DUPLEX)
    m2 = write(tmp_path / "m2.pdb", moved(DUPLEX, residues={("A", 4), ("A", 5)}, dz=3.0))
    res, regs = str(tmp_path / "res.tsv"), str(tmp_path / "reg.tsv")
    assert main(["map", "score", m1, m2, "--map", mapf, "--method", "cc", "--resolution", "4",
                 "--region", "loop=A:4-5", "-o", res, "--regions-out", regs]) == 0
    text = capsys.readouterr().out
    assert text.index("m1.pdb") < text.index("m2.pdb")
    assert len(open(res).read().splitlines()) == 29 and len(open(regs).read().splitlines()) == 7


def test_map_score_skips_the_empty_slots_of_a_model_check(tmp_path, capsys):
    from mdatk.cli import main
    mapf = _map_of(DUPLEX, tmp_path)
    m1 = write(tmp_path / "m1.pdb", DUPLEX)
    slot = write(tmp_path / "passed-02.pdb", pdbtools.empty_slot_text(2, 2, 1))
    res = str(tmp_path / "res.tsv")
    assert main(["map", "score", m1, slot, "--map", mapf, "--method", "cc", "--resolution", "4",
                 "--region", "loop=A:4-5", "-o", res]) == 0
    text = capsys.readouterr().out
    assert "skipped passed-02.pdb: empty slot 2 of 2: only 1 models passed" in text
    assert len(open(res).read().splitlines()) == 15  # the one model (14 residues), nothing for the slot


def test_map_score_of_empty_slots_only_says_why_regression(tmp_path, capsys):
    """regression: when every input was an empty slot of a model check the command failed with a blank
    reason ('no model was scored: ')."""
    from mdatk.cli import main
    mapf = _map_of(DUPLEX, tmp_path)
    slot = write(tmp_path / "slot" / "best-1.pdb", pdbtools.empty_slot_text(1, 1, 0))
    assert main(["map", "score", slot, "--map", mapf, "--method", "cc", "--resolution", "4"]) == 1
    err = capsys.readouterr().err
    assert "every input is an empty slot of a model check" in err
    assert "best-1.pdb: empty slot 1 of 1: only 0 models passed" in err


# --- hybrids -----------------------------------------------------------------------------------------------

def test_hybrids_per_chain_numbering_and_common_atoms_regression():
    """regression (body_hybrids.py): residues keyed by number alone merged chain A 1 with chain B 1;
    overlapping bodies were resolved silently by order."""
    old = pdbtools.read_pdb(PER_CHAIN)
    new_text = moved(PER_CHAIN, "B", 2.0)
    kept = [ln for ln in new_text.splitlines()
            if not (ln[12:16].strip() == "OP1" and ln[22:26].strip() == "2")]
    new = pdbtools.read_pdb("\n".join(kept) + "\n")
    bodies = [mapscore.parse_body("a=A"), mapscore.parse_body("b=B:1-7")]
    hybrids, members, notes = mapscore.make_hybrids(old, new, bodies)
    assert sorted(hybrids) == ["nn", "no", "on", "oo"] and len(members["a"]) == len(members["b"]) == 7
    n_atoms = {c: sum(len(r.atoms) for r in s.residues()) for c, s in hybrids.items()}
    assert len(set(n_atoms.values())) == 1 and any("atoms are not in both" in n for n in notes)
    on = hybrids["on"]
    old_b1 = next(r for r in old.residues() if (r.chain, r.resseq) == ("B", 1)).atoms[0].xyz
    hyb_b1 = next(r for r in on.residues() if (r.chain, r.resseq) == ("B", 1)).atoms[0].xyz
    assert hyb_b1[2] == pytest.approx(old_b1[2] + 2.0)  # the new coordinates, verbatim
    text = pdbtools.write_pdb(on)
    assert "REMARK 951 MDATK BODY HYBRID a=o b=n" in text and text.count("TER") == 2
    with pytest.raises(ensemble.EnsembleError, match="overlap"):
        mapscore.make_hybrids(old, new, [mapscore.parse_body("a=A"), mapscore.parse_body("b=A:3")])
    with pytest.raises(ensemble.EnsembleError, match="unique"):
        mapscore.make_hybrids(old, new, bodies, match="number")
    with pytest.raises(ensemble.EnsembleError, match="at most"):
        mapscore.make_hybrids(old, new, [mapscore.parse_body(f"x{k}=A:{k}") for k in range(1, 8)])


def test_hybrids_match_by_number_when_chain_ids_differ():
    old = pdbtools.read_pdb(DUPLEX)
    renamed = pdbtools.read_pdb(DUPLEX)
    for r in renamed.residues():
        r.chain = "Z"
    body = mapscore.parse_body("t=*:1-7")
    hybrids, members, _ = mapscore.make_hybrids(old, renamed, [body], match="number")
    assert len(members["t"]) == 7 and {r.chain for r in hybrids["n"].residues()} == {"A", "B"}


def test_body_effects():
    bodies = [mapscore.Body("t", []), mapscore.Body("j", [])]
    members = {"t": [("A", 1, "")], "j": [("A", 2, "")]}
    scores = {}
    for code in mapscore.hybrid_codes(2):
        t = 0.6 if code[0] == "n" else 0.5
        j = 0.4 if code[1] == "n" else 0.5
        scores[code] = {("A", 1, ""): ("G", t, 1), ("A", 2, ""): ("G", j, 1)}
    hyb_rows, body_rows = mapscore.body_effects(bodies, scores, members)
    assert body_rows[0][0] == "t" and body_rows[0][2] == "+0.0500" and body_rows[0][5] == "+0.1000"
    assert body_rows[1][5] == "-0.1000" and len(hyb_rows) == 4


def test_hybrids_cli_with_map(tmp_path, capsys):
    from mdatk.cli import main
    mapf = _map_of(DUPLEX, tmp_path)
    old = write(tmp_path / "old.pdb", moved(DUPLEX, "B", 2.5))
    new = write(tmp_path / "new.pdb", DUPLEX)
    eff = str(tmp_path / "eff.tsv")
    assert main(["pdb", "hybrids", old, new, "--body", "a=A", "--body", "b=B", "-o", str(tmp_path / "h"),
                 "--map", mapf, "--resolution", "4", "--effects-out", eff,
                 "--scores-out", str(tmp_path / "s.tsv")]) == 0
    assert sorted(os.listdir(tmp_path / "h")) == ["hyb_nn.pdb", "hyb_no.pdb", "hyb_on.pdb", "hyb_oo.pdb"]
    rows = [ln.split("\t") for ln in open(eff).read().splitlines()]
    effect = {r[0]: float(r[2]) for r in rows[1:]}
    assert effect["b"] > 0.02 and abs(effect["a"]) < effect["b"]  # moving B back into the map helps
    assert "main effects" in capsys.readouterr().out


# --- planner steps -----------------------------------------------------------------------------------------

def _plan(tmp_path, build):
    from mdatk.config import WorkflowParams
    from mdatk.workflows.plan import Plan
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    ids = build(wf)
    return Plan(wf, str(tmp_path / "plan")), ids


def _file(wf, path, kind):
    from mdatk.workflows.plan import add_node
    return add_node(wf, "input-file", settings={"path": str(path), "kind": kind})


def test_steps_registered_with_their_categories_and_ports():
    from mdatk.workflows import TOOLS
    cats = {name: TOOLS[name].category for name in ("register-census", "ensemble-compare", "map-model-score",
                                                    "body-hybrids")}
    assert cats == {"register-census": "Analysis", "ensemble-compare": "Analysis",
                    "map-model-score": "Refinement", "body-hybrids": "Refinement"}
    census = TOOLS["register-census"]
    assert [p.name for p in census.inputs] == ["models", "scores", "secstruct", "fasta"]
    assert census.inputs[0].multiple and census.inputs[1].kinds == ("table",)
    assert [p.kind for p in census.outputs] == ["table", "text"]
    hyb = TOOLS["body-hybrids"]
    assert [p.name for p in hyb.output_ports(hyb.normalise({}))] == ["hybrids", "scores", "effects"]
    assert [p.name for p in hyb.output_ports(hyb.normalise({"method": "none"}))] == ["hybrids"]
    score = TOOLS["map-model-score"]
    assert score.resources(score.normalise({})) == ["chimerax", "qscore"]


def test_step_checks():
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import StepContext

    def issues(tool, settings, inputs=None, groups=()):
        t = TOOLS[tool]
        ctx = StepContext("p", 1, STAMP, t.tag, inputs or {}, groups=groups)
        return [(i.level, i.message) for i in t.check(t.normalise(settings), ctx)]

    from mdatk.workflows.base import PortValue
    pdbs = {"models": PortValue(["/x/a.pdb"], "pdb")}
    errs = [m for lvl, m in issues("register-census", {}, pdbs) if lvl == "error"]
    assert any("at least one register" in m for m in errs)
    errs = [m for lvl, m in issues("register-census", {"registers": [{"name": "s", "pairs": "22-37"}]}, pdbs)
            if lvl == "error"]
    assert any("cannot read the pair" in m for m in errs)
    errs = [m for lvl, m in issues("register-census", {"registers": [{"name": "s", "pairs": "A:1-A:9"}]})
            if lvl == "error"]
    assert errs == ["connect the decoys, or a decoy table that lists them"]
    ok = issues("register-census", {"registers": [{"name": "s", "pairs": "A:1-A:9"}],
                                    "reference": "GGGCCCAAAA"}, pdbs)
    assert not [m for lvl, m in ok if lvl == "error"]
    arms = {"arm_a": PortValue(["/x/a.pdb"], "pdb"), "arm_b": PortValue(["/x/b.pdb"], "pdb")}
    errs = [m for lvl, m in issues("ensemble-compare", {"register_a": "s=A:1-A:9"}, arms) if lvl == "error"]
    assert any("needs the decoy table" in m for m in errs)
    errs = [m for lvl, m in issues("ensemble-compare", {"register_a": "s=A:1-A:9", "closure": "geometry"},
                                   dict(arms, map=PortValue(["/x/m.mrc"], "map"))) if lvl == "error"]
    assert any("calibration" in m for m in errs) and any("interface regions" in m for m in errs)
    errs = [m for lvl, m in issues("map-model-score", {"method": "cc"}) if lvl == "error"]
    assert errs == ["enter the map resolution (for the simulated map)"]
    errs = [m for lvl, m in issues("map-model-score", {"regions": [{"name": "i", "ranges": "group:Loops"}]},
                                   groups=[{"name": "Loops", "residues": "25-34"}]) if lvl == "error"]
    assert errs == []
    errs = [m for lvl, m in issues("body-hybrids", {}) if lvl == "error"]
    assert any("at least one body" in m for m in errs) and any("connect the density map" in m for m in errs)
    seven = [{"name": f"b{k}", "ranges": f"A:{k}"} for k in range(7)]
    errs = [m for lvl, m in issues("body-hybrids", {"bodies": seven, "method": "none"}) if lvl == "error"]
    assert errs == ["at most 6 bodies (64 hybrids)"]


def test_plan_writes_scripts_that_pass_bash_and_run(tmp_path):
    from mdatk.workflows.plan import add_node, connect
    inputs = tmp_path / "in"
    g = inputs / "decoys"
    decoys = [write(g / f"d{k}.pdb", DUPLEX if k < 3 else moved(DUPLEX, "B", 11.0)) for k in range(5)]
    write(g / "x.secstruct", "(((.... ....)))\n")
    mapf = _map_of(DUPLEX, tmp_path)
    fasta = write(inputs / "ref.fasta", REFERENCE)
    old = write(inputs / "old.pdb", moved(DUPLEX, "B", 2.0))

    def build(wf):
        files = [_file(wf, p, "pdb") for p in decoys]
        f = _file(wf, fasta, "fasta")
        m = _file(wf, mapf, "map")
        o = _file(wf, old, "pdb")
        cen = add_node(wf, "register-census", settings={"registers": [{"name": "stem", "pairs": STEM[5:]}],
                                                        "python": sys.executable})
        cmp_ = add_node(wf, "ensemble-compare", settings={
            "register_a": STEM, "closure": "geometry", "mode": "sweep", "calibration": "A:1-3,B:12-14",
            "regions_a": "A:4-7", "python": sys.executable})
        ms = add_node(wf, "map-model-score", settings={
            "method": "cc", "resolution": 4.0, "python": sys.executable,
            "regions": [{"name": "loop", "ranges": "group:Loop"}]})
        bh = add_node(wf, "body-hybrids", settings={"bodies": [{"name": "a", "ranges": "A"},
                                                               {"name": "b", "ranges": "B"}],
                                                    "resolution": 4.0, "python": sys.executable})
        for k, node in enumerate(files):
            connect(wf, node.id, "file", cen.id, "models")
            connect(wf, node.id, "file", cmp_.id, "arm_a" if k < 3 else "arm_b")
            connect(wf, node.id, "file", ms.id, "model")
        connect(wf, f.id, "file", cen.id, "fasta")
        connect(wf, m.id, "file", cmp_.id, "map")
        connect(wf, m.id, "file", ms.id, "map")
        connect(wf, m.id, "file", bh.id, "map")
        connect(wf, o.id, "file", bh.id, "old")
        connect(wf, files[0].id, "file", bh.id, "new")
        return [cen.id, cmp_.id, ms.id, bh.id]

    from mdatk.config import ResidueGroup, WorkflowParams
    from mdatk.workflows.plan import Plan
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    ids = build(wf)
    plan = Plan(wf, str(tmp_path / "plan"), groups=[ResidueGroup("Loop", "4-7", "", "#ffffff")])
    errors = [(plan.steps[n].label, i.message) for n, i in plan.all_issues() if i.level == "error"]
    assert errors == []
    report = plan.write()
    scripts = sorted(p for p in report.written if p.endswith(".sh") and "_00_" not in os.path.basename(p))
    assert len(scripts) == 4
    env = dict(os.environ, PYTHONPATH=ROOT)
    for path in scripts:
        assert subprocess.run(["bash", "-n", path], capture_output=True).returncode == 0
        text = open(path).read()
        assert "-m mdatk" in text
        proc = subprocess.run(["bash", path], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, (path, proc.stdout[-2000:], proc.stderr[-2000:])
    cen = open(next(p for p in scripts if p.endswith(f"_census_{STAMP}.sh"))).read()
    assert "--register 'stem=A:1-B:14, A:2-B:13, A:3-B:12'" in cen and "--fasta" in cen
    ms = open(next(p for p in scripts if p.endswith(f"_mapscore_{STAMP}.sh"))).read()
    assert "--region 'loop=*:4-7'" in ms
    plan_dir = tmp_path / "plan"
    for nid in ids:
        for pv in plan.steps[nid].outputs.values():
            for p in pv.paths:
                assert os.path.getsize(plan_dir / p) > 0, p
    report_text = open(plan_dir / plan.steps[ids[0]].outputs["report"].paths[0]).read()
    assert "stem FORCED" in report_text and "3  60.00%" in report_text
    mode = os.stat(scripts[0]).st_mode
    assert mode & stat.S_IXUSR


def _census_step(tmp_path, settings, folders=("g",)):
    """A plan with a register census of decoys in ``folders`` (connected one by one), written; its script
    run with bash.  Returns (the script's text, the completed process, the plan folder)."""
    from mdatk.config import WorkflowParams
    from mdatk.workflows.plan import Plan, add_node, connect
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    cen = add_node(wf, "register-census", settings=dict({"registers": [{"name": "stem", "pairs": STEM[5:]}],
                                                         "python": sys.executable}, **settings))
    for name in folders:
        for k in range(2):
            node = _file(wf, write(tmp_path / "in" / name / f"d{k}.pdb", DUPLEX), "pdb")
            connect(wf, node.id, "file", cen.id, "models")
    plan = Plan(wf, str(tmp_path / "plan"))
    assert [i.message for _n, i in plan.all_issues() if i.level == "error"] == []
    script = next(p for p in plan.write().written if p.endswith(f"_census_{STAMP}.sh"))
    proc = subprocess.run(["bash", script], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=ROOT))
    return open(script).read(), proc, tmp_path / "plan"


def test_census_step_with_a_typed_fasta_reference_regression(tmp_path):
    """regression: a FASTA typed as the reference sequence (as its help allows) was run together into one
    shell word, which reads as a header without a sequence: 'the reference sequence is empty'."""
    text, proc, root = _census_step(tmp_path, {"reference": REFERENCE.replace(">c B", "\n>c B")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    fasta = root / "01_census" / f"apt1_01_reference_{STAMP}.fasta"
    assert f"--fasta apt1_01_reference_{STAMP}.fasta" in text and "--reference" not in text
    assert ensemble.read_reference(str(fasta)).describe() == "14 nt, A:1-7 B:8-14"
    report = open(root / "01_census" / f"apt1_01_census-report_{STAMP}.txt").read()
    assert "reference: 14 nt, A:1-7 B:8-14" in report and "stem                   2 100.00%" in report
    text, proc, _root = _census_step(tmp_path / "plain", {"reference": "A:GGGAAAA\nB:UUUUCCC"})
    assert "--reference 'A:GGGAAAA B:UUUUCCC'" in text and proc.returncode == 0  # plain text as before


def test_census_step_with_a_dashed_pair_suffix_regression(tmp_path):
    """regression: a matched-groups suffix starting with '-' (groups pres / pres-free) was read by argparse
    as an option: 'argument --pair-suffix: expected one argument'."""
    text, proc, root = _census_step(tmp_path, {"pair_suffix": "-free"}, folders=("pres", "pres-free"))
    assert "--pair-suffix=-free" in text
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = open(root / "01_census" / f"apt1_01_census-report_{STAMP}.txt").read()
    assert "matched groups on '-free'" in report


def test_qscore_work_folder_moves_with_a_superseded_run_regression(tmp_path):
    """regression: map-model score and body hybrids keep ChimeraX's logs in the step's qscore/ folder, which
    did not move with a superseded run: its report named logs that the next run overwrote."""
    from mdatk.config import WorkflowParams
    from mdatk.workflows import runstate, supersede
    from mdatk.workflows.plan import Plan, add_node, connect
    mapf = _map_of(DUPLEX, tmp_path)
    model = write(tmp_path / "in" / "m1.pdb", DUPLEX)
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    m, f = _file(wf, mapf, "map"), _file(wf, model, "pdb")
    ms = add_node(wf, "map-model-score", settings={"method": "qscore"})
    bh = add_node(wf, "body-hybrids", settings={"method": "qscore", "bodies": [{"name": "a", "ranges": "A"}]})
    cc = add_node(wf, "map-model-score", settings={"method": "cc", "resolution": 4.0})
    for node in (ms, cc):
        connect(wf, f.id, "file", node.id, "model")
        connect(wf, m.id, "file", node.id, "map")
    connect(wf, f.id, "file", bh.id, "old")
    connect(wf, f.id, "file", bh.id, "new")
    connect(wf, m.id, "file", bh.id, "map")
    root = tmp_path / "plan"
    plan = Plan(wf, str(root))
    folders = {n: plan.steps[n].tool.run_folders(plan.steps[n].settings, plan.steps[n].ctx)
               for n in (ms.id, bh.id, cc.id)}
    # listed whatever the method: a Q-score run superseded after the step was switched to CC keeps its logs
    assert folders == {ms.id: ["01_mapscore/qscore"], bh.id: ["02_hybrids/qscore"],
                       cc.id: ["03_mapscore/qscore"]}
    plan.write()
    for path in runstate.expected(plan, ms.id, STAMP):  # a completed run, with ChimeraX's log of m1
        write(root / path, "x\n")
    write(root / "01_mapscore" / "qscore" / "m1_chimerax.log", "the log of this run\n")
    item = supersede.pending(plan)[ms.id]
    assert item.folders == ["01_mapscore/qscore"]
    moved = Plan(wf, str(root)).write().superseded[ms.id]
    log = root / moved.target / "qscore" / "m1_chimerax.log"
    assert open(log).read() == "the log of this run\n"
    assert not os.path.exists(root / "01_mapscore" / "qscore")


def test_qscore_work_folder_is_the_steps_own_without_step_folders_regression(tmp_path):
    """regression: in a plan without step folders every map-model score and body hybrids step wrote ChimeraX's
    logs into one <plan>/qscore/, so one step's logs replaced another's."""
    from mdatk.config import WorkflowParams
    from mdatk.workflows.plan import Plan, add_node, connect
    mapf = _map_of(DUPLEX, tmp_path)
    model = write(tmp_path / "in" / "m1.pdb", DUPLEX)
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, step_folders=False)
    m, f = _file(wf, mapf, "map"), _file(wf, model, "pdb")
    a = add_node(wf, "map-model-score", settings={"method": "qscore"})
    b = add_node(wf, "map-model-score", settings={"method": "qscore"})
    for node in (a, b):
        connect(wf, f.id, "file", node.id, "model")
        connect(wf, m.id, "file", node.id, "map")
    plan = Plan(wf, str(tmp_path / "plan"))
    texts = {}
    for node in (a, b):
        step = plan.steps[node.id]
        assert step.ok, step.issues
        sh = [g for g in step.tool.generate(step.settings, step.ctx) if g.name.endswith(".sh")]
        texts[node.id] = sh[0].content
    na, nb = plan.steps[a.id].node.step, plan.steps[b.id].node.step
    assert f"--work apt1_{na:02d}_qscore" in texts[a.id] and f"--work apt1_{nb:02d}_qscore" in texts[b.id]
    assert plan.steps[a.id].tool.run_folders(plan.steps[a.id].settings, plan.steps[a.id].ctx) == \
        [f"apt1_{na:02d}_qscore"]


def test_census_never_reads_superseded_runs_regression(tmp_path):
    """regression: a run moved aside into a step's superseded/ (its models/ included) was counted again by
    the recursive model search of the census."""
    live = write(tmp_path / "01_farfar2" / "models" / "r01" / "S_1.pdb", DUPLEX)
    write(tmp_path / "01_farfar2" / "superseded" / "x_y" / "models" / "r01" / "S_1.pdb", DUPLEX)
    assert ensemble.expand_models([str(tmp_path / "01_farfar2")], recursive=True) == [str(live)]
