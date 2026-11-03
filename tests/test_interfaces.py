"""Interface census, fingerprint clustering, segment support and the decoy table (mdatk.interfaces), with
regression tests for the problems found in the scripts they replace (interface_census.py,
interface_clusters.py, support_check.py)."""

import copy
import os
import random

import numpy as np
import pytest

from mdatk import builder, interfaces as I, pdbtools
from mdatk.cli import main
from mdatk.config import Project, ResidueGroup

WCAT = {"A": "N1", "G": "N1", "C": "N3", "U": "N3"}  # the old census's 'WC nitrogens'


def helix(seq, partner="auto", chirality="D", chains=("A", "B"), start=(1, 1), scheme=None, **kw):
    return pdbtools.read_pdb(builder.build_helix(seq, partner=partner, chirality=chirality, chains=chains,
                                                 start=start, scheme=scheme, **kw))


def moved(structure, shift=(0.0, 0.0, 0.0), chains=None):
    s = copy.deepcopy(structure)
    for res in s.residues():
        if chains:
            res.chain = chains.get(res.chain, res.chain)
        for a in res.atoms:
            x, y, z = (v + d for v, d in zip(a.xyz, shift))
            a.tail = f"   {x:8.3f}{y:8.3f}{z:8.3f}" + a.tail[27:]
    return s


def joined(*structures):
    return pdbtools.Structure(segments=[seg for s in structures for seg in copy.deepcopy(s).segments])


def residue(name, chain, resseq, atoms, record="ATOM"):
    res = pdbtools.Residue(name, chain, resseq)
    for n, xyz in atoms.items():
        tail = f"   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
        res.atoms.append(pdbtools.Atom(record, n, "", tail))
    return res


def side(text):
    return I.parse_side(text)


def old_wc_count(s, ca, cb):
    nts = I.nucleotides(s)
    count = 0
    for x in (n for n in nts if n.chain == ca):
        for y in (n for n in nts if n.chain == cb):
            p, q = x.atoms.get(WCAT[x.base]), y.atoms.get(WCAT[y.base])
            count += p is not None and q is not None and float(np.linalg.norm(p - q)) < 3.4
    return count


# --- sides --------------------------------------------------------------------------------------------

def test_side_syntax_groups_and_overlap():
    assert side("A:1-16,B:17-44") == [("A", 1, 16), ("B", 17, 44)]
    assert side("C") == [("C", None, None)] and side(":5-9") == [("", 5, 9)]
    assert side("1-40 45") == [(None, 1, 40), (None, 45, 45)]  # plain ranges: every chain
    groups = [ResidueGroup("Aptamer", "A:1-16, B:17-44"), ResidueGroup("Target", "", "chainID C"),
              ResidueGroup("Loop", "", "chainID C and resid 58:64"), ResidueGroup("Odd", "", "segid X")]
    assert I.parse_side("group:Aptamer", groups) == [("A", 1, 16), ("B", 17, 44)]
    assert I.parse_side("group:Target, A:3", groups) == [("C", None, None), ("A", 3, 3)]
    assert I.parse_side("group:Loop", groups) == [("C", 58, 64)]
    with pytest.raises(I.InterfaceError, match="unknown residue group 'Nope'"):
        I.parse_side("group:Nope", groups)
    with pytest.raises(I.InterfaceError, match="no residue ranges"):
        I.parse_side("group:Odd", groups)
    with pytest.raises(I.InterfaceError, match="cannot read residue range"):
        I.parse_side("A:x-3")
    for text in ("A:1-16,B:17-44", "C", "1-40,45", "5-5,A:-3--1"):
        assert side(I.format_side(side(text))) == side(text)
    assert I.sides_overlap(side("A:1-10"), side("A:10-20,B")) == ["A:1-10 / A:10-20"]
    assert I.sides_overlap(side("1-5"), side("C:3")) and not I.sides_overlap(side("A"), side("B"))


# --- census -------------------------------------------------------------------------------------------

def test_census_of_a_duplex_and_fingerprint_round_trip():
    s = helix("GGCAUCGAUGCC")
    row = I.census(s, side("A"), side("B"))
    assert row.counts == {"contacts": 350, "stack": 0, "wc": 12, "gu": 0, "other": None, "bph": 0, "ohph": 0,
                          "mgbr": 0}
    d = row.as_dict()
    assert d["residues_a"] == "A:1-12" and d["residues_b"] == "B:1-12" and d["other"] == ""
    assert I.parse_fingerprint(d["fingerprint"]) == row.fingerprint
    assert I.census(s, side("A"), None).counts == row.counts  # side B: every other nucleotide
    assert I.census(helix("GGUC", partner="GGCU"), side("A"), side("B")).counts["gu"] == 2


def test_mismatches_and_trans_pairs_are_not_watson_crick():
    """The old wc column was one N1/N3 distance: it counted an A·C mismatch (two acceptors) and trans
    Watson-Crick pairs, although its docstring said trans pairs read as zero."""
    mismatch = helix("GACG", partner="CGCC")
    assert old_wc_count(mismatch, "A", "B") == 4
    assert I.census(mismatch, side("A"), side("B")).counts["wc"] == 3
    trans = helix("GAGAGA", pairing="reverse-wc")
    assert old_wc_count(trans, "A", "B") == 3
    assert I.census(trans, side("A"), side("B")).counts["wc"] == 0


def test_chains_with_the_same_numbers_stay_apart():
    """Residues were keyed by number (+ an offset), so chains numbered alike merged."""
    s = helix("GGCAUCGA", start=(1, 1))
    row = I.census(s, side("A"), side("B"))
    assert row.counts["wc"] == 8 and len(row.residues_a) == len(row.residues_b) == 8
    assert {k[0] for k in row.residues_a} == {"A"} and {k[0] for k in row.residues_b} == {"B"}
    with pytest.raises(I.InterfaceError, match="the sides overlap at B:1, B:2"):
        I.census(s, side("1-8"), side("B"))
    with pytest.raises(I.InterfaceError, match="side B matches no nucleotide"):
        I.census(s, side("A"), side("C"))


def test_ions_water_and_hydrogens_never_count():
    """HETATM ions and water inside the ranges counted as residues, and hydrogens named 1H5' (digit first)
    passed as heavy atoms."""
    heavy = helix("GCAUGC")
    withh = helix("GCAUGC", hydrogens=True)
    for res in withh.residues():
        for a in res.atoms:
            if a.name == "H5'":
                a.name, a.element = "1H5'", ""
            elif a.name.startswith("H"):
                a.element = ""
    extra = [residue("MG", "B", 3, {"MG": (0.0, 0.0, 0.0)}, "HETATM"),
             residue("HOH", "B", 4, {"O": (1.0, 2.0, 0.5), "H1": (1.5, 2.0, 0.5), "H2": (1.0, 2.5, 0.5)},
                     "HETATM")]
    withh.segments.append(extra)
    a, b = I.census(heavy, side("A"), side("B")), I.census(withh, side("A"), side("B"))
    assert a.counts == b.counts and a.residues_b == b.residues_b and len(I.nucleotides(withh)) == 12


def test_mg_bridgeable_phosphates_without_a_contact():
    """mgbr was only evaluated for residue pairs that already had a 4.5 Å contact."""
    def phosphate(name, chain, x):
        return residue(name, chain, 1, {"P": (x, 0.0, 0.0), "OP1": (x + (1.5 if x else -1.5), 0.0, 0.0),
                                       "OP2": (x, 0.0, 1.5 if x else -1.5)})
    for gap, mg, contacts in ((6.0, 1, 0), (4.0, 0, 1), (7.5, 0, 0)):
        s = pdbtools.Structure(segments=[[phosphate("G", "A", 0.0)], [phosphate("C", "B", gap)]])
        row = I.census(s, side("A"), side("B"))
        assert (row.counts["mgbr"], row.counts["contacts"] > 0) == (mg, bool(contacts)), gap
    s = pdbtools.Structure(segments=[[phosphate("G", "A", 0.0)], [phosphate("C", "B", 6.0)]])
    assert I.census(s, side("A"), side("B"), I.CensusParams(mg_window=(6.5, 8.0))).counts["mgbr"] == 0


def test_base_phosphate_donors_and_the_c_h_option():
    """bph counts N-H donors by base identity (O6 / N7 acceptors never), C-H donors only on request."""
    g = I.nucleotides(helix("G", partner="none"))[0]
    ring = np.mean([g.atoms[n] for n in I.PURINE_RING], axis=0)

    def out(name, d=3.0):
        v = g.atoms[name] - ring
        return tuple(g.atoms[name] + d * v / np.linalg.norm(v))
    seg_a = [residue("G", "A", 1, {n: tuple(x) for n, x in g.atoms.items()})]
    for donor, ch in (("N2", False), ("C8", True), ("O6", None)):  # O6 is an acceptor
        p = residue("C", "B", 1, {"P": out(donor, 4.4), "OP1": out(donor)})
        s = pdbtools.Structure(segments=[seg_a, [p]])
        plain = I.census(s, side("A"), side("B")).counts["bph"]
        with_ch = I.census(s, side("A"), side("B"), I.CensusParams(bph_ch=True)).counts["bph"]
        expected = {False: (1, 1), True: (0, 1), None: (0, 0)}[ch]
        assert (plain, with_ch) == expected, donor


def test_stacking_needs_an_offset_along_the_normal():
    g = I.nucleotides(helix("G", partner="none"))[0]
    centre, normal = I.ring_plane(g)
    atoms = {n: tuple(x) for n, x in g.atoms.items()}
    for shift, stacks in ((3.4 * normal, 1), (np.cross(normal, [1.0, 0.0, 0.0]) * 4.0, 0)):
        other = residue("G", "B", 1, {n: tuple(np.array(x) + shift) for n, x in atoms.items()})
        s = pdbtools.Structure(segments=[[residue("G", "A", 1, atoms)], [other]])
        assert I.census(s, side("A"), side("B")).counts["stack"] == stacks


def test_fingerprint_keeps_chains_negative_numbers_and_insertion_codes():
    """The old 'a-t' fingerprint had no chain and broke on negative residue numbers."""
    s = helix("GGCAUC", start=(-2, -6))
    s.segments[1][0].icode = "A"
    row = I.census(s, side("A"), side("B"))
    text = row.as_dict()["fingerprint"]
    assert "A:-2/B:-1" in text.split() and any("A/" in t or t.endswith("A") for t in text.split())
    assert I.parse_fingerprint(text) == row.fingerprint


def test_unknown_and_l_names_from_the_atoms():
    """Bases came from a name table (BASE1) or from the first A/G/C/U letter of the name, so 'CCC' read as
    C and '0' or 'LA0' needed the table; now unknown names get the base of their atoms."""
    s = helix("GGCAUC", chirality="L", scheme="rosetta")
    names = ["0", "LG0", "CCC", "PSU", "XYZ", "0C"]
    for res, name in zip(s.segments[0], names):
        res.name = name
    nts = I.nucleotides(s)
    assert [(n.name, n.base, n.hand) for n in nts[:6]] == [
        ("0", "G", "L"), ("LG0", "G", "L"), ("CCC", "C", "L"), ("PSU", "A", "L"), ("XYZ", "U", "L"),
        ("0C", "C", "L")]
    assert I._infer_base({"N1", "C2", "N3", "C4", "C5", "C6", "O4", "C7"}) == "T"
    assert I._infer_base({"MG"}) is None
    assert I.census(s, side("A"), side("B")).counts["wc"] == 6


# --- decoy table and filters ---------------------------------------------------------------------------

def _decoy_files(folder, n=3):
    os.makedirs(folder, exist_ok=True)
    text = builder.build_helix("GCAU")
    paths = []
    for k in range(n):
        path = os.path.join(folder, f"r{k:02d}_S_0001.pdb")
        with open(path, "w") as fh:
            fh.write(text)
        paths.append(path)
    return paths


def test_decoy_table_rows_match_by_path_not_by_tag(tmp_path):
    """The old filter keyed silent-file scores by part + tag, where RNAdo's file names give every run the
    same part, so one run's chainbreak was applied to another run's decoys."""
    step = tmp_path / "02_farfar2-models"
    step.mkdir()
    paths = _decoy_files(str(step / "decoys"))
    table = step / "scores.tsv"
    table.write_text("run\ttag\tpdb\tscore\tlinear_chainbreak\tatom_pair_constraint\n"
                     "r00\tS_0001\t02_farfar2-models/decoys/r00_S_0001.pdb\t-10\t0.5\t1\n"
                     "r01\tS_0001\t02_farfar2-models/decoys/r01_S_0001.pdb\t-12\t9.0\t1\n"
                     "r02\tS_0001\t\t-15\t0.1\t1\n")  # not extracted
    t = I.read_decoy_table(str(table))
    assert t.columns[:3] == ["run", "tag", "pdb"] and t.has("atom_pair_constraint")
    root = I.guess_plan_root(t)
    assert root == str(tmp_path)
    out = I.apply_filters(paths, t, I.parse_filters("linear_chainbreak <= 5"), root)
    assert out.kept == [paths[0], paths[2]] and out.dropped == [paths[1]] and out.unscored == [paths[2]]
    assert I.apply_filters(paths, t, I.parse_filters("linear_chainbreak<=5"), root, True).kept == [paths[0]]
    assert I.models_from_table(t, root) == [str(tmp_path / "02_farfar2-models/decoys/r00_S_0001.pdb"),
                                            str(tmp_path / "02_farfar2-models/decoys/r01_S_0001.pdb")]
    with pytest.raises(I.InterfaceError, match="no column 'rms'"):
        I.apply_filters(paths, t, I.parse_filters("rms < 2"), root)
    assert I.parse_filters("a<1\nb >= -2.5e1; c != 0") == [("a", "<", 1.0), ("b", ">=", -25.0),
                                                           ("c", "!=", 0.0)]
    with pytest.raises(I.InterfaceError, match="cannot read the filter"):
        I.parse_filters("linear_chainbreak is small")
    csv_table = tmp_path / "s.csv"
    csv_table.write_text("description,score\nS_0001,-3\n")
    t2 = I.read_decoy_table(str(csv_table))
    assert t2.columns == ["tag", "score"] and t2.matcher()("x/S_0001.pdb")["score"] == "-3"


# --- clusters -----------------------------------------------------------------------------------------

def entry(name, a, b, contacts=10):
    """A census entry contacting residues a (chain A) with residues b (chain C), pairwise in order."""
    pairs = [(("A", x, ""), ("C", y, "")) for x, y in zip(a, b)]
    pairs += [(("A", x, ""), ("C", b[-1], "")) for x in a[len(b):]]
    pairs += [(("A", a[-1], ""), ("C", y, "")) for y in b[len(a):]]
    return I.CensusEntry(name, f"{name}.pdb", pairs, {"contacts": contacts, "stack": 0, "wc": 0, "gu": 0,
                                                       "other": None, "bph": 0, "ohph": 0, "mgbr": 0})


def ensemble(seed=0):
    rng = random.Random(seed)
    out = []
    for k in range(30):  # one family: A 8-9, 25-28 with C 58-64
        a = sorted(rng.sample([8, 9, 25, 26, 27, 28], 5))
        out.append(entry(f"pres_{k:02d}", a, sorted(rng.sample(range(58, 65), 6))))
    for k in range(20):  # another: A 29-37 with C 51, 56-57, 60-65
        a = sorted(rng.sample(range(29, 38), 7))
        out.append(entry(f"inv_{k:02d}", a, sorted(rng.sample([51, 56, 57, 60, 61, 62, 63, 64, 65], 7))))
    out += [I.CensusEntry(f"none_{k}", f"none_{k}.pdb", [], {}) for k in range(10)]
    return out


def test_clusters_find_the_two_families_and_score_references():
    refs = [I.reference_from_ranges("pres", side("A:8-9,A:25-28,A:33-35"), side("C:58-64")),
            I.reference_from_ranges("inv", side("A:8-9,A:27,A:29-37"), side("C:51,C:56-57,C:60-65"))]
    res = I.cluster_census(ensemble(), I.ClusterParams(), refs)
    big = [c for c in res.clusters if len(c.members) >= 5]
    assert [len(c.members) for c in big] == [30, 20] and len(res.empty) == 10
    names = [{res.entries[k].model.split("_")[0] for k in c.members} for c in big]
    assert names == [{"pres"}, {"inv"}]
    assert big[0].shares["pres"] > 0.6 and big[1].shares["inv"] > 0.9
    s = res.summary
    assert s["clusters_min_size"] == 2 and s["coverage"] == 1.0 and s["pattern"].startswith("few recurring")
    assert s["lead"] == "" or s["lead"] in ("pres", "inv")
    rows = I.cluster_rows(res, {})
    assert rows[0]["size"] == 30 and "share_pres" in rows[0] and rows[0]["members"].count("pres_") == 30


def test_reference_shares_are_normalised_and_use_both_sides():
    """The old share was hits / all exclusive hits on the target side only: a cluster touching both
    references' exclusive residues fully scored 67 % for the reference with more of them."""
    member = I.CensusEntry("m", "m.pdb", [(("A", 1, ""), ("C", r, "")) for r in (10, 11, 12, 13, 20, 21)], {})
    r1 = I.Reference("big", set(), {("C", r, "") for r in (10, 11, 12, 13)})
    r2 = I.Reference("small", set(), {("C", 20, ""), ("C", 21, "")})
    res = I.cluster_census([member], I.ClusterParams(min_size=1), [r1, r2])
    assert res.clusters[0].shares == {"big": 0.5, "small": 0.5}
    # two references that differ on side A only
    r3 = I.Reference("left", {("A", 1, ""), ("A", 2, "")}, {("C", 30, "")})
    r4 = I.Reference("right", {("A", 5, ""), ("A", 6, "")}, {("C", 30, "")})
    m2 = I.CensusEntry("m2", "m2.pdb", [(("A", 1, ""), ("C", 30, "")), (("A", 2, ""), ("C", 30, ""))], {})
    res = I.cluster_census([m2], I.ClusterParams(min_size=1), [r3, r4])
    assert res.clusters[0].shares == {"left": 1.0, "right": 0.0}
    assert res.exclusive["left"] == ({("A", 1, ""), ("A", 2, "")}, set())
    assert res.shared == (set(), {("C", 30, "")})


def test_shares_tally_every_recurring_cluster_from_all_members():
    """The old summary tallied only the clusters printed (--top) and scored each from its representative
    alone."""
    refs = [I.reference_from_ranges("pres", side("A:25-28"), side("C:58-59")),
            I.reference_from_ranges("inv", side("A:29-37"), side("C:51,C:56-57"))]
    res = I.cluster_census(ensemble(), I.ClusterParams(), refs)
    for c in res.clusters:
        for r in refs:
            expected = np.mean([I.touched_fraction(res.entries[k], res.exclusive[r.name]) for k in c.members])
            assert c.touched[r.name] == pytest.approx(expected)
    weight = sum(len(c.members) for c in res.clusters if len(c.members) >= 5)
    pres = sum(len(c.members) * c.shares["pres"] for c in res.clusters if len(c.members) >= 5) / weight
    assert res.summary["shares_all"]["pres"] == pytest.approx(pres, abs=1e-4)
    text = I._cluster_text(res, show=1)  # one cluster printed, the tally still covers both
    assert "\n  2 " not in text and "size-weighted reference shares" in text
    assert f"  pres             all {f'{100 * pres:.0f} %':>6}" in text


def test_clusters_do_not_depend_on_input_order_and_the_medoid_is_exact():
    """The old medoid came from a 60-member sample for large clusters, and seeding followed the input
    order among equal fingerprint sizes."""
    rng = random.Random(3)
    core = list(range(1, 11))
    entries = [entry(f"m{k:03d}", sorted(rng.sample(core, 7)), [50, 51, 52]) for k in range(90)]
    res = I.cluster_census(entries, I.ClusterParams(cut=0.6))
    shuffled = entries[:]
    random.Random(1).shuffle(shuffled)
    res2 = I.cluster_census(shuffled, I.ClusterParams(cut=0.6))

    def named(r):
        return [(r.entries[c.representative].model, sorted(r.entries[k].model for k in c.members))
                for c in r.clusters]
    assert named(res) == named(res2)
    big = res.clusters[0]
    assert len(big.members) > 60
    feats = [I.features(e.pairs) for e in res.entries]
    total = {k: sum(I.jaccard(feats[k], feats[j]) for j in big.members) for k in big.members}
    best = min(total.values())
    assert total[big.representative] == pytest.approx(best)


def test_no_reference_lines_without_references_and_block_bins():
    res = I.cluster_census(ensemble(), I.ClusterParams())
    text = I._cluster_text(res, 12)
    assert "reference" not in text and "exclusive" not in text and "discriminating" not in text
    feats = I.features([(("A", 7, ""), ("C", -1, "")), (("B", 8, ""), ("C", 3, ""))], "blocks", 4)
    assert feats == {"a|A:1", "b|C:-1", "a|B:2", "b|C:0"}


def test_old_census_tables_are_refused(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("decoy,score,contacts,fp\nd1,-3,12,8-60|9-61\n")
    with pytest.raises(I.InterfaceError, match="no chain ids"):
        I.read_census(str(old))


# --- segment support ----------------------------------------------------------------------------------

def test_support_sequence_separation_counts_within_a_chain_only():
    """support_check.py skipped every residue within --sep in *number*, across chains: in a duplex
    A:7-12 / B:13-18 the base-paired residue 12 / 13 was reported unsupported."""
    duplex = helix("GCAUGC", start=(7, 13))
    partner = moved(helix("GGGG", partner="none", chains=("C", "D")), (60.0, 0.0, 0.0))
    s = joined(duplex, partner)
    res = I.segment_support(s, side("A,B"), side("C"), side("A:10-12"))
    by_key = {r.key: r for r in res.residues}
    assert by_key[("A", 12, "")].scaffold > 0 and by_key[("A", 12, "")].supported()
    # the old rule (|q - r| <= 3 by number, whatever the chain) leaves A:12 with nothing
    nts = {n.key: n for n in I.nucleotides(s)}
    me = nts[("A", 12, "")]
    old = 0
    for key, q in nts.items():
        if abs(q.resseq - 12) <= 3 or key[0] == "C":
            continue
        old += sum(float(np.linalg.norm(me.atoms[x] - q.atoms[y])) < 4.5 for x in me.heavy for y in q.heavy)
    assert old == 0


def test_ions_never_hold_a_rebuilt_residue():
    """An Mg2+ numbered above the old APTAMER_MAX counted as partner support."""
    s = helix("GCAUGC")
    c1 = I.nucleotides(s)[2].atoms["OP1"]
    mg = residue("MG", "C", 70, {"MG": tuple(c1 + np.array([0.0, 0.0, 2.1]))}, "HETATM")
    far = moved(helix("GG", partner="none", chains=("C", "D")), (80.0, 0.0, 0.0))
    s2 = joined(s, far)
    s2.segments.append([mg])
    res = I.segment_support(s2, side("A,B"), side("C"), side("A:3"))
    assert res.residues[0].partner == 0


def test_pair_credit_is_a_class_of_its_own_and_a_bifurcated_bond_counts_once():
    s = helix("GCAUGC")
    off = I.segment_support(s, side("A,B"), None, side("A,B"), I.SupportParams(sep=0))
    on = I.segment_support(s, side("A,B"), None, side("A,B"), I.SupportParams(sep=0, pairs="contacts"))
    can = I.segment_support(s, side("A,B"), None, side("A,B"), I.SupportParams(sep=0, pairs="builtin"))
    assert all(r.scaffold == 0 for r in on.residues)  # the old code added pair credit to 'scaffold'
    assert sum(r.pair for r in on.residues) > 0 and any(r.mark == "=" for r in on.residues)
    assert sum(r.pair for r in off.residues) == 0 and all(r.mark in ":" for r in off.residues)
    assert [r.pair > 0 for r in can.residues] == [r.pair > 0 for r in on.residues]

    def nt(base, chain, atoms):
        return I.Nucleotide(chain, 1, "", base, base, False, "D", {k: np.array(v) for k, v in atoms.items()},
                            list(atoms))
    g = nt("G", "A", {"N1": (0.0, 0.0, 0.0), "N2": (10.0, 0.0, 0.0)})
    u = nt("U", "B", {"O2": (2.9, 0.0, 0.0), "O4": (0.0, 2.9, 0.0)})
    assert not I._base_paired(g, u, 3.3)  # one donor, two acceptors: a bifurcated bond, not a pair
    u2 = nt("U", "B", {"O2": (2.9, 0.0, 0.0), "N3": (10.0, 2.9, 0.0), "O4": (12.5, 0.0, 0.0)})
    g2 = nt("G", "A", {"N1": (0.0, 0.0, 0.0), "O6": (10.0, 0.0, 0.0)})
    assert I._base_paired(g2, u2, 3.3)  # G N1-H...U O2 and U N3-H...G O6: two bonds


def test_auto_threshold_and_longest_runs():
    assert I.pick_threshold([0, 1, 0, 6, 7, 1]) == 6
    assert I.pick_threshold([0, 1, 2, 3, 3]) is None  # no gap of 2
    assert I.pick_threshold([0, 6, 7]) is None  # fewer than 5 models
    r = I.SupportResult("m", [I.ResidueSupport(("A", k, ""), "G", 1) for k in range(5)], [[0, 1, 2, 3, 4]])
    r.residues[2].self_ = 3
    assert r.longest_run() == 5 and r.longest_run(count_self=True) == 2


# --- command line -------------------------------------------------------------------------------------

def _two_families(folder):
    """Six models: a D duplex (chains A, B) with an L duplex (C, D) alongside it or at its end."""
    os.makedirs(folder, exist_ok=True)
    d = helix("GGCAUCGA", start=(1, 9))
    lx = helix("CCGAUG", chirality="L", chains=("C", "D"), start=(17, 23), scheme="rosetta")
    paths = []
    for k in range(6):
        shift = (19.5 + 0.2 * k, 0.0, 0.3 * k) if k < 3 else (0.0, 0.5 * (k % 2), 27.0 + 0.2 * k)
        path = os.path.join(folder, f"decoy_{k}.pdb")
        with open(path, "w") as fh:
            fh.write(pdbtools.write_pdb(joined(d, moved(lx, shift))))
        paths.append(path)
    return paths


def test_cli_census_clusters_and_support(tmp_path, capsys):
    paths = _two_families(str(tmp_path / "models"))
    project = Project()
    project.groups = [ResidueGroup("Duplex", "A:1-8, B:9-16"), ResidueGroup("Partner", "", "chainID C")]
    pfile = str(tmp_path / "p.mdatk.toml")
    from mdatk import projectfile
    projectfile.save(pfile, project)
    scores = tmp_path / "scores.tsv"
    scores.write_text("run\ttag\tpdb\tscore\tlinear_chainbreak\n" + "".join(
        f"r1\tS_{k}\tmodels/decoy_{k}.pdb\t{-k}\t{9.0 if k == 1 else 0.2}\n" for k in range(6)))
    census = tmp_path / "census.tsv"
    assert main(["decoys", "census", *paths, "--side-a", "group:Duplex", "--side-b", "group:Partner,D",
                 "--scores", str(scores), "--filter", "linear_chainbreak<=5", "-p", pfile,
                 "-o", str(census)]) == 0
    out = capsys.readouterr().out
    assert "kept 5, dropped 1 of 6" in out and "5 models, side A A:1-8,B:9-16, side B C,D" in out
    rows = I.read_table(str(census))
    assert [r["model"] for r in rows] == [f"decoy_{k}.pdb" for k in (0, 2, 3, 4, 5)]
    assert rows[0]["score"] == "0" and rows[0]["linear_chainbreak"] == "0.2"
    clusters = tmp_path / "clusters.tsv"
    rep = str(tmp_path / "rep-{n}.pdb")
    assert main(["decoys", "clusters", str(census), "--min-size", "2", "--representatives", rep,
                 "--n-representatives", "3", "--reference", "groove=group:Duplex/C:17-22", "-p", pfile,
                 "--members", str(tmp_path / "members.tsv"), "-o", str(clusters)]) == 0
    out = capsys.readouterr().out
    assert "2 cluster(s) for 3 representatives" in out and "reference groove" in out
    crow = I.read_table(str(clusters))
    assert [int(r["size"]) for r in crow] == [3, 2] and crow[0]["representative_pdb"].endswith("rep-01.pdb")
    assert os.path.exists(tmp_path / "rep-01.pdb") and os.path.exists(tmp_path / "rep-02.pdb")
    assert not os.path.exists(tmp_path / "rep-03.pdb")
    support = tmp_path / "support.tsv"
    assert main(["decoys", "support", "--census", str(census), "--side-a", "A,B", "--side-b", "C,D",
                 "--fixed", "A:1-2,B:15-16,C:17-18", "--run", "auto", "--pairs", "contacts",
                 "--residues", str(tmp_path / "res.tsv"), "--profile", str(tmp_path / "prof.tsv"),
                 "-o", str(support)]) == 0
    srows = I.read_table(str(support))
    assert len(srows) == 5 and {r["rebuilt"] for r in srows} == {str(16 + 12 - 6)}
    assert "run threshold 4 (no clear gap; the default 4)" in capsys.readouterr().out
    log = projectfile.read_log(pfile)
    assert [e["step"] for e in log] == ["decoys census", "decoys clusters"]
    assert main(["decoys", "census", *paths, "--side-a", "A", "--side-b", "A:1-3"]) == 1  # overlap
    assert "the sides overlap" in capsys.readouterr().err


# --- 0.0.12 review fixes ----------------------------------------------------------------------------------

def test_score_filters_are_written_with_the_threshold_as_typed():
    """format_filter used '{:g}' (6 significant digits): 'score <= -1500.125' ran as '<=-1500.12'."""
    for text, written in (("score <= -1500.125", "score<=-1500.125"),
                          ("score < -215.4375", "score<-215.4375"),
                          ("atom_pair_constraint <= 1234567", "atom_pair_constraint<=1234567"),
                          ("linear_chainbreak <= 5", "linear_chainbreak<=5"),
                          ("rms >= 1.5e-07", "rms>=1.5e-07"),
                          ("score <= -1234.567", "score<=-1234.567"), ("x != 0.1", "x!=0.1")):
        f = I.parse_filters(text)[0]
        assert I.format_filter(f) == written
        assert I.parse_filters(I.format_filter(f)) == [f]  # what the census reads back runs


@pytest.mark.parametrize("cut, shared, total", [(0.1, 9, 10), (0.35, 13, 20), (0.05, 19, 20)])
def test_fingerprints_exactly_at_the_cut_join_the_cluster(cut, shared, total):
    """The float32 matrix gave 1 - 9/10 = 0.100000024, past the 1e-9 tolerance: two clusters."""
    big = frozenset(f"a|A:{k}" for k in range(total))
    small = frozenset(f"a|A:{k}" for k in range(shared))
    assert I.jaccard(big, small) <= cut + 1e-9
    assert I.leader_clusters([big, small], ["m1", "m2"], cut) == [[0, 1]]


def test_medoid_ties_go_to_the_seed():
    """Members 0, 1 and 2 all sum to 5/3; float32 rounding picked member 2."""
    feats = [frozenset({"2"}), frozenset({"2"}), frozenset({"0", "2", "3"}), frozenset({"0", "3"})]
    assert I.medoid(feats, [0, 1, 2, 3]) == 0
    assert I.medoid(feats, [2, 0, 1, 3]) == 2  # the earliest member, whichever it is


def test_support_keeps_the_strands_of_a_file_without_chain_ids_apart():
    """With blank chain ids and numbering continuous across strands (an Amber / cpptraj frame), residues
    8 and 9 of different strands counted as backbone neighbours: their contacts were dropped and one
    segment ran across the strand break."""
    duplex = helix("GCAUGCAU", start=(1, 9))  # A:1-8 paired with B:9-16; B:9 starts without its P
    blank = copy.deepcopy(duplex)
    for res in blank.residues():
        res.chain = ""
    blank.segments = [[r for seg in blank.segments for r in seg]]  # one segment, no TER
    with_ids = I.segment_support(duplex, side("A"), side("B"), side("A:6-8,B:9-11"))
    without = I.segment_support(blank, side(":1-8"), side(":9-16"), side(":6-11"))

    def counts(res):
        return [(r.partner, r.scaffold, r.pair, r.self_) for r in res.residues]
    assert counts(without) == counts(with_ids) and without.residues[2].partner > 0  # :8 pairs with :9
    assert without.segments == with_ids.segments == [[0, 1, 2], [3, 4, 5]]
    assert [n.strand for n in I.nucleotides(blank)] == [0] * 8 + [1] * 8
    origin = np.zeros(3)
    assert not I._strand_break({"O3'": origin}, {"P": np.array([1.6, 0.0, 0.0])})
    assert I._strand_break({"O3'": origin}, {"P": np.array([4.5, 0.0, 0.0])})  # a gap
    assert I._strand_break({"O3'": origin}, {"O5'": np.array([2.6, 0.0, 0.0])})  # a 5' end
    assert not I._strand_break({}, {"C1'": origin})  # nothing to tell: joined, as before


def test_model_folders_are_searched_outside_superseded_regression(tmp_path):
    """regression: a folder given to the decoy census was searched inside superseded/, so the decoys of runs
    RNAdo moved aside were counted with the current ones."""
    def put(rel):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("END\n")
        return str(path)
    keep = [put("models/r01/S_1.pdb"), put("models/r02/S_1.pdb"), put("top.pdb")]
    put("models/superseded/a_b/models/r01/S_1.pdb")
    put("superseded/a_b/S_9.pdb")
    put(".hidden/S_3.pdb")
    assert I._model_files([str(tmp_path)]) == sorted(keep)
