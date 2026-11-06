"""The structure builder's planner step (mdatk.workflows.buildstep) with B-form DNA helices and D/L chimeras: the
options (one set of residue-name options for both builds, projects written before load as they were), checks,
ports, port data, descriptions, citations and generated files; RNA command lines as before, the chimera design
passed through bash unchanged; the steps that read the builder's hand ('mixed' for a chimera of both hands); and
the generated scripts run with the local Python."""

import os
import subprocess
import sys

from mdatk import pdbtools as pt
from mdatk.config import WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.base import StepContext, require_file
from mdatk.workflows.plan import Plan, add_node, connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = TOOLS["structure-builder"]
CTX = StepContext("p", 1, "s", TOOL.tag)
DESIGN = "D-RNA:GGCAUC 3'-5' L-RNA:GAUGCC"
NAMES = ("scheme", "terminals", "phosphate5", "hydrogens")


def settings(**kw):
    return TOOL.normalise(kw)


def issues(**kw):
    return [(i.level, i.message, i.keys) for i in TOOL.check(settings(**kw), CTX)]


def errors(**kw):
    return [m for level, m, _keys in issues(**kw) if level == "error"]


# --- options ------------------------------------------------------------------------------------------

def test_options_rna_only_pairings_shared_names_and_projects_written_before():
    flat = TOOL.flat_options()
    na = flat["nucleic_acid"]
    assert na.kind == "radio" and na.default == "rna"
    assert list(na.choices) == [("rna", "RNA (A-form)"), ("dna", "DNA (B-form)")]
    helix = [o.key for o in TOOL.section("helix").options]
    assert helix.index("nucleic_acid") + 1 == helix.index("sequence")
    assert "RNA reads T as U, DNA reads U as T" in flat["sequence"].tooltip
    assert list(flat["chirality"].choices) == [("D", "D (right-handed)"), ("L", "L (mirror image)")]
    assert ("chimera", "D/L chimera") in flat["mode"].choices
    # pairings, triplexes and their third strand: RNA only
    values = {"mode": "helix", "partner_mode": "auto", "pairing": "wc", "triplex": "hoogsteen",
              "third_mode": "explicit"}
    for key in ("pairing", "triplex", "third_mode", "third", "start3"):
        assert flat[key].is_enabled(dict(values, nucleic_acid="rna")), key
        assert not flat[key].is_enabled(dict(values, nucleic_acid="dna")), key
    assert flat["start2"].is_enabled(dict(values, nucleic_acid="dna"))
    # the chimera's own options; residue names and atoms: one set for both builds, the helix's keys
    chim = TOOL.section("chimera")
    assert [o.key for o in chim.options] == ["design", "chimera_na", "chimera_chains"]
    assert chim.requires == {"mode": "chimera"} and chim.option("design").placeholder == DESIGN
    tip = chim.option("design").tooltip
    assert "HAND[-NA]:SEQUENCE[/PARTNER]" in tip and "3'-3'" in tip and f"{DESIGN} is 5'-D-RNA(6)-p-L-RNA(6)-3'" in tip
    names = TOOL.section("names")
    assert [o.key for o in names.options] == list(NAMES) and names.requires == {"mode": ("helix", "chimera")}
    keys = [o.key for _sec, o in TOOL.all_options()]
    assert all(keys.count(k) == 1 for k in NAMES)
    # a project written before DNA and chimeras: the same values, an RNA helix
    old = {"mode": "helix", "sequence": "GGCAUC", "chirality": "L", "pairing": "hoogsteen", "scheme": "amber",
           "terminals": False, "phosphate5": True, "hydrogens": True, "chains": "P,Q"}
    problems = []
    s = TOOL.normalise(old, problems)
    assert problems == [] and {k: s[k] for k in old} == old
    assert (s["nucleic_acid"], s["design"], s["chimera_na"], s["chimera_chains"]) == ("rna", "", "rna", "")


def test_rna_command_lines_are_as_before():
    out = CTX.name("helix", "pdb")
    assert TOOL.command(settings(sequence="GGCAUC"), CTX) == (
        f"python3 -m mdatk build helix --seq GGCAUC --partner auto --chirality D --chains A,B --start=1,1 -o {out}")
    assert TOOL.command(settings(sequence="ggcatc", chirality="L", chains="A, B", start1=-5, scheme="amber",
                                 terminals=False, phosphate5=True, hydrogens=True), CTX) == (
        "python3 -m mdatk build helix --seq ggcatc --partner auto --chirality L --chains A,B --start=-5,1 "
        f"--scheme amber --no-terminal-names --phosphate5 --hydrogens -o {out}")
    assert TOOL.command(settings(sequence="AAGAG", triplex="reverse-hoogsteen", third_mode="explicit",
                                 third="GAGAA", start3=7), CTX) == (
        "python3 -m mdatk build helix --seq AAGAG --partner auto --chirality D --chains A,B --triplex "
        f"reverse-hoogsteen --third-strand GAGAA --start=1,1,7 -o {out}")
    assert TOOL.command(settings(sequence="AAGAG", partner_mode="explicit", partner="CUCUU", pairing="hh"), CTX) == (
        f"python3 -m mdatk build helix --seq AAGAG --partner CUCUU --chirality D --chains A,B --pairing hh "
        f"--start=1,1 -o {out}")


# --- DNA helices --------------------------------------------------------------------------------------

def test_dna_helix_command_checks_ports_data_describe_and_files():
    s = settings(sequence="GGCAT", nucleic_acid="dna", chirality="L")
    out = CTX.name("helix", "pdb")
    assert TOOL.command(s, CTX) == ("python3 -m mdatk build helix --seq GGCAT --partner auto --na dna "
                                    f"--chirality L --chains A,B --start=1,1 -o {out}")
    # an RNA pairing or triplex left in the settings: DNA builds its Watson-Crick B-form duplex
    stale = settings(sequence="AAGAG", nucleic_acid="dna", pairing="hoogsteen", triplex="hoogsteen",
                     third_mode="explicit", start3=7, chains="A,B")
    assert TOOL._pairing(stale) == "wc" and TOOL._triplex(stale) is None
    cmd = TOOL.command(stale, CTX)
    assert "--pairing" not in cmd and "--triplex" not in cmd and " --start=1,1 -o " in cmd
    found = issues(sequence="AAGAG", nucleic_acid="dna", pairing="reverse-hoogsteen", triplex="hoogsteen")
    assert not [m for level, m, _k in found if level != "info"]
    assert ("info", "5 base pairs: 5 Watson-Crick, 0 G·T wobble, 0 mismatched", ("sequence",)) in found
    assert not any("no secondary-structure output" in m for _l, m, _k in found)
    # the builder's DNA checks
    assert ("info", "sequence: U read as T (DNA)", ("sequence",)) in issues(sequence="GGCAU", nucleic_acid="dna")
    assert ("info", "5 base pairs: 4 Watson-Crick, 1 G·T wobble, 0 mismatched", ("sequence",)) in issues(
        sequence="GGCAT", nucleic_acid="dna", partner_mode="explicit", partner="ATGTC")
    assert any("(DNA: A, C, G, T)" in m for m in errors(sequence="GGCAX", nucleic_acid="dna"))
    assert any("(RNA: A, C, G, U)" in m for m in errors(sequence="GGCAX"))
    assert ("info", "single strand of 5 nucleotides in the B-form helix geometry", ("sequence",)) in issues(
        sequence="GGCAT", nucleic_acid="dna", partner_mode="none")
    # ports and their data: a DNA duplex feeds the secondary structure (pairs as the RNA model reads them)
    assert [p.name for p in TOOL.output_ports(s)] == ["pdb", "secstruct"]
    assert [p.name for p in TOOL.output_ports(stale)] == ["pdb", "secstruct"]
    single = settings(sequence="GGCAT", nucleic_acid="dna", partner_mode="none")
    assert [p.name for p in TOOL.output_ports(single)] == ["pdb"] and list(TOOL.output_values(single, CTX)) == ["pdb"]
    values = TOOL.output_values(s, CTX)
    assert values["pdb"].paths == [CTX.own("helix", "pdb")] and values["pdb"].data == {"chirality": "L"}
    model = values["secstruct"].data["rna_model"]
    assert model.split(model.dotbracket) == "((((( )))))" and model.sequence == "GGCAUAUGCC"
    wobble = TOOL.ss_model(settings(sequence="GGCAT", nucleic_acid="dna", partner_mode="explicit", partner="AAGTC"))
    assert wobble.split(wobble.dotbracket) == "(((.( ).)))"  # G·T bracketed, A·A not
    assert TOOL.ss_model(single) is None
    # describe, citations, files
    assert TOOL.describe(s) == "L-DNA helix GGCAT" and TOOL.describe(stale) == "D-DNA helix AAGAG"
    assert TOOL.describe(settings(nucleic_acid="dna")) == "D-DNA helix (no sequence)"
    assert TOOL.describe(settings(sequence="AAGAG", pairing="hoogsteen")) == "D-RNA helix (hoogsteen) AAGAG"
    assert TOOL.resources(s) == ["arnott-bdna"] and TOOL.resources(stale) == ["arnott-bdna"]
    assert TOOL.resources(settings(sequence="AAGAG", pairing="hoogsteen")) == ["arnott-arna", "pdb-4jrd"]
    files = TOOL.generate(s, CTX)
    assert files[0].content.splitlines()[1] == "# MDATK structure builder: Ideal B-form DNA helix"
    assert TOOL.command(s, CTX) in files[0].content.splitlines()
    assert files[1].name == CTX.name("helix", "secstruct") and files[1].content == "((((( )))))\nggcat atgcc\n"
    assert TOOL.generate(settings(sequence="GGCAU"), CTX)[1].content == "((((( )))))\nggcau augcc\n"
    assert TOOL.generate(stale, CTX)[0].content.splitlines()[1] == "# MDATK structure builder: Ideal B-form DNA helix"


# --- D/L chimeras --------------------------------------------------------------------------------------

def test_chimera_ports_data_describe_citations_and_files():
    s = settings(mode="chimera", design=DESIGN)
    assert TOOL.input_ports(s) == [] and [p.name for p in TOOL.output_ports(s)] == ["pdb"]
    values = TOOL.output_values(s, CTX)
    assert list(values) == ["pdb"] and values["pdb"].paths == [CTX.own("chimera", "pdb")]
    assert values["pdb"].data == {"chirality": "mixed"} and TOOL.chimera_hand(s) == "mixed"
    for design, hand in (("D-RNA:GGC 3'-5' D-DNA:GCC", "D"), ("L:GC 3'-3' L-DNA:GC/none", "L"),
                         ("D:GC 5'-5' L:GC", "mixed"), ("L:GC 3'-5' L:GC 3'-3' D:A", "mixed")):
        assert TOOL.output_values(settings(mode="chimera", design=design), CTX)["pdb"].data == {"chirality": hand}
    for design in ("", "D:GC 3'-5'", "D:GX 3'-5' L:GC"):  # no hand while the design does not parse
        assert TOOL.output_values(settings(mode="chimera", design=design), CTX)["pdb"].data == {}
    # describe: the strand as the design reads it
    assert TOOL.describe(s) == "chimera 5'-D-RNA(6)-p-L-RNA(6)-3'"
    assert TOOL.describe(settings(mode="chimera", design="D:GC 3'-3' L:GC", chimera_na="dna")) == \
        "chimera 5'-D-DNA(2)-3'p3'-L-DNA(2)-5'"
    assert TOOL.describe(settings(mode="chimera")) == "chimera (no design)"
    assert TOOL.describe(settings(mode="chimera", design="D:GX 3'-5' L:GC")) == "chimera D:GX 3'-5' L:GC"
    assert TOOL.describe(settings(mode="chimera", design="D-RNA:GGXAUC 3'-5' L-RNA:GAUGCC")) == \
        "chimera D-RNA:GGXAUC 3'-5' L-R…"
    # citations: the fiber models of the nucleic acids used
    assert TOOL.resources(s) == ["arnott-arna"]
    assert TOOL.resources(settings(mode="chimera", design="D-DNA:GGC 3'-5' D-RNA:GCC")) == ["arnott-bdna",
                                                                                            "arnott-arna"]
    assert TOOL.resources(settings(mode="chimera", design="D:GC 3'-5' L:GC", chimera_na="dna")) == ["arnott-bdna"]
    assert TOOL.resources(settings(mode="chimera", design="D:GC 3'-5'")) == []
    # the script: its title as 'mdatk build chimera' logs it, no secondary structure
    files = TOOL.generate(s, CTX)
    assert [f.name for f in files] == [CTX.name("build", "sh")]
    lines = files[0].content.splitlines()
    assert lines[1] == "# MDATK structure builder: Chimera 5'-D-RNA(6)-p-L-RNA(6)-3'"
    assert lines[-2:] == [TOOL.command(s, CTX), require_file(CTX.name("chimera", "pdb"), "mdatk build")]
    proc = subprocess.run(["bash", "-n"], input=files[0].content, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_chimera_checks_on_the_design_and_the_chain_ids():
    assert issues(mode="chimera") == [("error", f"enter the design, e.g. {DESIGN}", ("design",))]
    found = issues(mode="chimera", design=DESIGN)
    assert ("info", "5'-D-RNA(6)-p-L-RNA(6)-3': 2 segments, 1 link(s) (3'-5')", ("design",)) in found
    assert all(keys == ("design",) and level == "info" for level, _m, keys in found)
    assert any("503 atoms (heavy atoms only)" in m for _l, m, _k in found)
    assert any("hydrogens included" in m for _l, m, _k in issues(mode="chimera", design=DESIGN, hydrogens=True))
    # the builder's errors and warnings, on the design option
    for design, part in (("D:GGXU 3'-5' L:GC", "segment 1 sequence: invalid letter(s) X (RNA: A, C, G, U)"),
                         ("D:GC 3'-5'", "the design ends with the link 3'-5'"),
                         ("D:GC", "a chimera needs at least two segments"),
                         ("D:GC 3'-3' L:GC 3'-5' D:GC", "link 2 (3'-5') starts at a 3' end, but the free end of "
                                                        "segment 2"),
                         ("X:GC 3'-5' L:GC", "the hand must be D or L")):
        found = issues(mode="chimera", design=design)
        assert [(lvl, keys) for lvl, _m, keys in found] == [("error", ("design",))], design
        assert part in found[0][1], found
    found = issues(mode="chimera", design="D:GGC/GAC 3'-5' L:GC")
    assert any(lvl == "warning" and keys == ("design",) and "mismatched pairs" in m for lvl, m, keys in found)
    assert not errors(mode="chimera", design="D:GC 3'-5' L:GC", chimera_na="dna")
    assert any("(DNA: A, C, G, T)" in m for m in errors(mode="chimera", design="D:GX 3'-5' L:GC", chimera_na="dna"))
    # chain ids: one per chain the design writes (strand 1 split at bridges, then the partners)
    assert not errors(mode="chimera", design=DESIGN, chimera_chains="A,B,C")
    assert not errors(mode="chimera", design=DESIGN, chimera_chains=" P, Q ,R ")
    count = ("error", "chain ids: the design writes 3 chains (strand 1 split at 3'-3' / 5'-5' links, then the partner "
                      "strands): 3 single characters, e.g. A,B,C", ("chimera_chains",))
    assert count in issues(mode="chimera", design=DESIGN, chimera_chains="A,B")
    assert count in issues(mode="chimera", design=DESIGN, chimera_chains="AB,C,D")
    assert count in issues(mode="chimera", design=DESIGN, chimera_chains="A,,B,C")
    assert ("error", "chain ids: the 3 chains need different ids, e.g. A,B,C", ("chimera_chains",)) in issues(
        mode="chimera", design=DESIGN, chimera_chains="A,A,B")
    bridge = "D:GC 3'-3' L:GC"
    assert any("the design writes 4 chains" in m and m.endswith("e.g. A,B,C,D")
               for m in errors(mode="chimera", design=bridge, chimera_chains="A,B,C"))
    assert not errors(mode="chimera", design=bridge, chimera_chains="W,X,Y,Z")
    assert not errors(mode="chimera", design="D:GC/none 3'-3' L:GC/none", chimera_chains="W,X")
    assert errors(mode="chimera", design="D:GX 3'-5' L:GC", chimera_chains="A") == [
        "segment 1 sequence: invalid letter(s) X (RNA: A, C, G, U)"]  # no chain check of a design with errors
    # the Amber 5' units, for a chimera too; nothing of the chimera in the other modes
    amber = ("warning", "the Amber 5' units have no 5'-phosphate; tleap cannot type it", ("phosphate5", "scheme"))
    assert amber in issues(mode="chimera", design=DESIGN, scheme="amber", phosphate5=True)
    assert amber in issues(sequence="GGCAU", scheme="amber", phosphate5=True)
    assert not [k for _l, _m, k in issues(sequence="GGCAU", design="nonsense", chimera_chains="!") if k == ("design",)]
    assert not errors(mode="chimera", design=DESIGN, sequence="", chains="AB,C,D,E")


def test_chimera_command_line_and_the_design_through_bash(tmp_path):
    out = CTX.name("chimera", "pdb")
    assert TOOL.command(settings(mode="chimera", design=DESIGN), CTX) == (
        "python3 -m mdatk build chimera --design 'D-RNA:GGCAUC 3'\"'\"'-5'\"'\"' L-RNA:GAUGCC' -o " + out)
    s = settings(mode="chimera", design="  D:GC\t3'-3'   L:GC ", chimera_na="dna", chimera_chains=" P, Q,R ,S",
                 scheme="amber", terminals=False, phosphate5=True, hydrogens=True)
    assert TOOL.command(s, CTX) == (
        "python3 -m mdatk build chimera --design 'D:GC 3'\"'\"'-3'\"'\"' L:GC' --na dna --chains P,Q,R,S "
        f"--scheme amber --no-terminal-names --phosphate5 --hydrogens -o {out}")
    # every word reaches the program as one argument, the design with its apostrophes and primes unchanged
    for design in (DESIGN, "D-DNA:GGCAT/none 5′-5′ L-RNA:GAUGC", "L:GC 3'5' D:GC/auto 3-3 L-DNA:A/T",
                   "D:G\"C 3'-5' L:$(touch x)`id`\\n"):
        s = settings(mode="chimera", design=design, chimera_chains="A,B,C", python="printf '%s\\n'")
        proc = subprocess.run(["bash", "-c", TOOL.command(s, CTX)], capture_output=True, encoding="utf-8",
                              check=True, cwd=str(tmp_path))
        assert proc.stdout.split("\n") == ["-m", "mdatk", "build", "chimera", "--design", design, "--chains",
                                           "A,B,C", "-o", out, ""], proc.stdout
    assert os.listdir(tmp_path) == []  # nothing in the design ran


# --- the steps that read the builder's hand ------------------------------------------------------------

def _fed(tmp_path, builder_settings, tool, port, settings=None, with_map=False):
    """A plan of the builder feeding ``tool`` (its input ``port``); returns that step and its tool."""
    wf = WorkflowParams(project_name="apt1", stamp="20261106-1200")
    built = add_node(wf, "structure-builder", settings=builder_settings)
    step = add_node(wf, tool, settings=settings or {})
    connect(wf, built.id, "pdb", step.id, port)
    if with_map:
        emap = add_node(wf, "input-file", settings={"path": str(tmp_path / "x.mrc"), "kind": "map"})
        connect(wf, emap.id, "file", step.id, "map")
    plan = Plan(wf, str(tmp_path / "plan"))
    return plan.steps[step.id], TOOLS[tool]


def test_mixed_chimeras_need_l_dictionaries_an_l_library_and_no_single_hand(tmp_path):
    mixed, d_chimera = {"mode": "chimera", "design": DESIGN}, {"mode": "chimera", "design": "D:GGC 3'-5' D-DNA:GCC"}
    l_helix = {"sequence": "GGCAUC", "chirality": "L"}
    # refinement: restraint dictionaries for the L residues of a mixed chimera, as for an L helix
    for builder_settings, start in ((mixed, "the model has L residues (a D/L chimera): Phenix needs restraint "
                                            "dictionaries"),
                                    (l_helix, "the model is L-RNA: Phenix needs restraint dictionaries"),
                                    (d_chimera, None)):
        st, _tool = _fed(tmp_path, builder_settings, "phenix-rsr", "model", {"resolution": 3.5}, with_map=True)
        dicts = [i.message for i in st.issues if "restraint dictionaries" in i.message]
        assert dicts == ([] if start is None else [f"{start} for its 5'/3' variants (0A5/0A3 ...) - connect a "
                                                   "'restraint-dicts' step"]), (builder_settings, st.issues)
    # the Amber build: a mixed chimera has L residues (the generated L-RNA library), a D chimera none
    for builder_settings, has_l in ((mixed, True), (l_helix, True), (d_chimera, False)):
        st, amber = _fed(tmp_path, builder_settings, "amber-build", "structure")
        assert amber.lrna_generated(st.settings, st.ctx) is has_l, builder_settings
    # the model check: no single hand expected of a mixed chimera (reported until the model exists)
    st, check = _fed(tmp_path, mixed, "model-check", "models")
    assert check.expected_hands(st.settings, st.ctx) == ("", "", "")
    assert any(i.message.startswith("no model exists yet to measure the hands on") for i in st.issues)
    args = check.arguments(st.settings, st.ctx)
    assert not [a for a in args if a.startswith(("--expect-l", "--expect-d"))]
    for builder_settings, expected in (({"mode": "chimera", "design": "L:GC 3'-3' L-DNA:GC"},
                                        ("*", "", "the builder's L helix")),
                                       (d_chimera, ("", "*", "the builder's D helix")),
                                       (l_helix, ("*", "", "the builder's L helix"))):
        st, check = _fed(tmp_path, builder_settings, "model-check", "models")
        assert check.expected_hands(st.settings, st.ctx) == expected


# --- the generated scripts, run ------------------------------------------------------------------------

def _run_step(tmp_path, node_settings):
    """Write a plan of one builder step (the local Python), run its script; returns the step and its PDB."""
    wf = WorkflowParams(project_name="e2e", stamp="20261106-1200")
    node = add_node(wf, "structure-builder", settings=dict(node_settings, python=sys.executable))
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    st = plan.steps[node.id]
    assert not [i for i in st.issues if i.level == "error"], st.issues
    report = plan.write()
    script = os.path.join(root, st.ctx.from_own(st.ctx.name(TOOL.tag, "sh")))
    assert script in report.written
    proc = subprocess.run(["bash", script], capture_output=True, encoding="utf-8", timeout=600,
                          env=dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return st, os.path.join(root, st.outputs["pdb"].path)


def test_the_generated_chimera_script_builds_the_design(tmp_path):
    st, pdb = _run_step(tmp_path, {"mode": "chimera", "design": "D-RNA:GC 3'-5' L-RNA:GC",
                                   "chimera_chains": "P,Q,R", "scheme": "amber"})
    assert st.outputs["pdb"].data == {"chirality": "mixed"}
    with open(pdb, encoding="utf-8") as fh:
        text = fh.read()
    assert text.startswith("REMARK 951 MDATK CHIMERA 5'-D-RNA(2)-p-L-RNA(2)-3'\n")
    s = pt.read_pdb(pdb)
    assert [(seg[0].chain, [r.name for r in seg]) for seg in s.segments] == [
        ("P", ["G5", "C", "LG", "LC3"]), ("Q", ["G5", "C3"]), ("R", ["LG5", "LC3"])]
    hands = [(r.residue.chain, r.call) for r in pt.residue_hands(s)]
    assert hands == [("P", "D"), ("P", "D"), ("P", "L"), ("P", "L"), ("Q", "D"), ("Q", "D"), ("R", "L"), ("R", "L")]


def test_the_generated_dna_helix_script_builds_b_dna(tmp_path):
    st, pdb = _run_step(tmp_path, {"sequence": "GGCAT", "nucleic_acid": "dna", "chirality": "L"})
    s = pt.read_pdb(pdb)
    assert [[r.name for r in seg] for seg in s.segments] == [["0DG", "0DG", "0DC", "0DA", "0DT"],
                                                            ["0DA", "0DT", "0DG", "0DC", "0DC"]]
    assert {r.call for r in pt.residue_hands(s)} == {"L"}
    assert all("O2'" not in r.coords() for r in s.residues())
    with open(pdb, encoding="utf-8") as fh:
        assert fh.readline().startswith("REMARK 951 MDATK IDEAL B-FORM DNA")
    with open(os.path.join(os.path.dirname(pdb), st.ctx.name("helix", "secstruct")), encoding="utf-8") as fh:
        assert fh.read() == "((((( )))))\nggcat atgcc\n"
