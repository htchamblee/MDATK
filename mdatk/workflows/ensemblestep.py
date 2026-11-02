"""Planner steps for decoy ensembles and map-model scores (mdatk.ensemble, mdatk.mapscore).

* ``register-census`` (Analysis): which register (a named set of defining base pairs) every decoy formed,
  grouped by folder and by the build condition read from the Rosetta logs, never pooled across conditions.
* ``ensemble-compare`` (Analysis): two ensembles side by side - closure, defining pairs, map occupancy -
  with exact intervals, Fisher tests, a closure-threshold sweep and the detectable difference at a power.
* ``map-model-score`` (Refinement): per-residue Q-score (Phenix, on a copy with standard residue names)
  or MDATK's per-residue correlation, and models ranked by named regions.
* ``body-hybrids`` (Refinement): every old/new combination of re-fitted bodies, optionally scored, with
  each body's main effect.

Each step writes a short bash script that calls ``python3 -m mdatk ensemble|map|pdb ...`` (from the Linux
AppImage, the AppImage itself), as the structure builder does: the work needs NumPy, so MDATK must be
installed where the script runs.  Residue ranges may name project residue groups as ``group:NAME``; they
are resolved when the script is written.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

from .. import ensemble, mapscore
from .base import PortValue, StepContext, Tool, env_section, require_file, script_header, shell_quote
from .buildstep import StructureBuilderTool
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

_PYTHON = Option("python", "Python", "text", "python3", "Interpreter that can import MDATK (with NumPy and "
                 "SciPy). From the standalone AppImage the script calls that program itself instead.")
_IMPORTABLE = "MDATK must be importable by the Python below (installed, or its folder on PYTHONPATH)."


def _env(placeholder: str = "conda activate mdatk") -> Section:
    return env_section(placeholder, [_PYTHON])


def _mdatk(s: Dict[str, Any]) -> List[str]:
    return StructureBuilderTool.mdatk(s)


def _array(name: str, paths: Sequence[str]) -> List[str]:
    """A bash array of file names, one per line (decoy lists can be long)."""
    return [f"{name}=("] + [f"  {shell_quote(p)}" for p in paths] + [")"]


def _continued(words: Sequence[str], indent: str = "    ", width: int = 100) -> str:
    """One command over several lines, broken between words; an option stays on the line of its value."""
    units: List[str] = []
    for w in words:
        if (units and units[-1].startswith("-") and " " not in units[-1] and "=" not in units[-1]
                and not w.startswith("-")):
            units[-1] += " " + w
        else:
            units.append(w)
    lines, cur = [], ""
    for w in units:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur + " \\")
            cur = indent + w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    return "\n".join(lines)


def _root(ctx: StepContext) -> str:
    """The plan folder as seen from the step folder (where the decoy table's paths start)."""
    return ".." if ctx.folder else "."


def _ranges(text: str, ctx: StepContext) -> str:
    """Residue ranges with group:NAME resolved to the project group's ranges."""
    return ensemble.selection_text(ensemble.parse_selection(text, ctx.groups)) if text.strip() else ""


def _check_ranges(text: str, key: str, ctx: StepContext, out: List[Issue]) -> bool:
    if not text.strip():
        return True
    try:
        ensemble.parse_selection(text, ctx.groups)
    except ensemble.EnsembleError as exc:
        out.append(Issue("error", str(exc), (key,)))
        return False
    return True


def _paths(text: str) -> List[str]:
    return [p for p in re.split(r"[,;\s]+", text or "") if p]


def _rel_dir(path: str, ctx: StepContext) -> str:
    """A folder typed relative to the plan folder (or absolute), as seen from the step folder."""
    return path if os.path.isabs(path) else ctx.rel(path)


def _reference_issues(s: Dict[str, Any], out: List[Issue]) -> None:
    if s["reference"].strip():
        try:
            ensemble.parse_reference(s["reference"])
        except ensemble.EnsembleError as exc:
            out.append(Issue("error", str(exc), ("reference",)))
    if s["offsets"].strip():
        try:
            ensemble.parse_offsets(s["offsets"])
        except ensemble.EnsembleError as exc:
            out.append(Issue("error", str(exc), ("offsets",)))


def _plan_reference(ctx: StepContext, *ports: str) -> Optional[ensemble.Reference]:
    """The plan's sequence: the RNA model a FARFAR2 step passes along with its files."""
    for port in ports:
        model = ctx.data(port).get("rna_model")
        ref = ensemble.reference_from_model(model) if model is not None else None
        if ref is not None:
            return ref
    return None


def _reference_file(s: Dict[str, Any], ctx: StepContext, files: List[GeneratedFile], *ports: str) -> str:
    """Without a typed or connected reference, the plan's sequence written as FASTA beside the script."""
    if s["reference"].strip() or ctx.has("fasta"):
        return ""
    ref = _plan_reference(ctx, *ports)
    if ref is None:
        return ""
    name = ctx.name("reference", "fasta")
    files.append(GeneratedFile(name, ensemble.reference_text(ref), "input",
                               "reference sequence of the plan (the FARFAR2 step's)"))
    return name


def _numbering_words(s: Dict[str, Any], ctx: StepContext, ref_file: str) -> List[str]:
    q = shell_quote
    words: List[str] = []
    if s["reference"].strip():
        words += ["--reference", q(" ".join(s["reference"].split()))]
    elif ctx.has("fasta"):
        words += ["--fasta", q(ctx.inp("fasta"))]
    elif ref_file:
        words += ["--fasta", q(ref_file)]
    if s["offsets"].strip():
        words += [f"--offset={q(s['offsets'].replace(' ', ''))}"]
    return words


def _numbering_section() -> Section:
    return Section("numbering", "Numbering", [
        Option("reference", "Reference sequence", "multiline", "",
               "The construct the registers and ranges are numbered in: plain ('GGGCAAAG...', numbered "
               "from 1, chains as in the decoys), chain-prefixed ('A:GGGC B:AGCU', numbered on across "
               "chains) or FASTA with Rosetta's chain:range tags. Each chain of a decoy is located in it by "
               "its sequence, so per-chain or shifted numbering still matches, and the bases of the defining "
               "pairs are checked. Empty: the connected FASTA, else the plan's FARFAR2 sequence, else the "
               "decoys' own numbering.", placeholder="A:GGGCAAAGAAACAGCU B:AGCUAG..."),
        Option("offsets", "Chain offsets", "text", "",
               "Instead of a reference: numbers to add per chain, e.g. 'B:-367,C:44'.",
               placeholder="B:-367,C:44"),
    ], tooltip="How the decoys' residue numbers are brought to the numbering of the registers.")


REGISTER_OPTIONS = [
    Option("name", "Name", "text", "register",
           "Name of the register (letters, digits, '.', '_', '-'); it heads the columns of the table."),
    Option("pairs", "Defining pairs", "text", "",
           "Chain-qualified residue pairs, e.g. 'A:22-A:37, A:23-A:36, A:24-A:35'. A pair is formed when "
           "the Watson-Crick N1/N3 distance is below the cut.",
           placeholder="A:22-A:37, A:23-A:36, A:24-A:35"),
]
REGISTER_TEMPLATES = [("register", "Register", {"name": "register", "pairs": ""})]


def _registers(s: Dict[str, Any], key: str = "registers") -> List[ensemble.Register]:
    return [ensemble.parse_register(f"{r['name'].strip()}={r['pairs']}") for r in s[key]]


def _register_issues(s: Dict[str, Any], out: List[Issue], key: str = "registers") -> List[ensemble.Register]:
    regs = []
    for k, rec in enumerate(s[key]):
        try:
            regs.append(ensemble.parse_register(f"{rec['name'].strip()}={rec['pairs']}"))
        except ensemble.EnsembleError as exc:
            out.append(Issue("error", str(exc), (f"{key}.{k}.pairs",)))
    if not s[key]:
        out.append(Issue("error", "define at least one register (its name and defining pairs)", (key,)))
    if len(regs) == len(s[key]):
        try:
            ensemble.check_registers(regs)
        except ensemble.EnsembleError as exc:
            out.append(Issue("error", str(exc), (key,)))
    return regs


# ---------------------------------------------------------------------------
# register census
# ---------------------------------------------------------------------------


@register
class RegisterCensusTool(Tool):
    name = "register-census"
    label = "Register census (MDATK)"
    category = "Analysis"
    tag = "census"
    description = ("Which register (named set of defining base pairs) each decoy formed or approached, "
                   "grouped by folder and by the build condition read from the Rosetta logs; groups are "
                   "never pooled across conditions.")
    inputs = [
        Port("models", "Decoys (PDB)", ("pdb",), optional=True, multiple=True,
             tooltip="The decoys. Several connections are allowed (one per condition or arm): decoys are "
                     "grouped by the folder they are in."),
        Port("scores", "Decoy table", ("table",), optional=True, multiple=True,
             tooltip="Score terms by name (the FARFAR2 models step's table); it can also list the decoys."),
        Port("secstruct", "Secondary structure", ("secstruct",), optional=True, multiple=True,
             tooltip="What the decoys were built with, for groups without a Rosetta log or *.secstruct of "
                     "their own."),
        Port("fasta", "Reference sequence (FASTA)", ("fasta",), optional=True,
             tooltip="The construct the registers are numbered in (e.g. the FARFAR2 step's FASTA)."),
    ]
    outputs = [Port("census", "Census (per decoy)", ("table",)), Port("report", "Census report", ("text",))]
    sections = [
        Section("registers", "Registers", REGISTER_OPTIONS, repeat=True, item_label="name",
                templates=REGISTER_TEMPLATES,
                tooltip="Alternative pairings to count, each a set of defining pairs."),
        _numbering_section(),
        Section("conditions", "Groups and conditions", [
            Option("log_dirs", "Rosetta log folders", "text", "",
                   "Folders (relative to the plan folder, or absolute) searched for Rosetta logs besides the "
                   "decoys' own folders, e.g. the FARFAR2 step's folder. The logs' command lines (with their "
                   "@flags files) give each group's secondary structure and obligate pairs.",
                   placeholder="04_farfar2"),
            Option("group_depth", "Group by folder level", "int", 1,
                   "1: decoys in one folder form a group; 2: the folder above it.", minimum=1, maximum=10),
            Option("pair_suffix", "Matched groups", "text", "",
                   "Report each group X next to the group X<suffix> (the same set-up without, e.g., the "
                   "internal pairs), with Fisher tests. Empty: off.", placeholder="_nointernalpairs"),
            Option("use_table_models", "Also count the decoys the decoy table lists", "bool", False,
                   "Every decoy with a PDB file in the connected decoy table (e.g. all extracted decoys), "
                   "besides the connected decoys."),
        ]),
        Section("criteria", "Pairs and chirality", [
            Option("cut", "Formed: N1-N3 below", "float", ensemble.PAIR_CUT,
                   "Watson-Crick N1/N3 distance of a formed pair.", minimum=2.0, maximum=6.0, step=0.1,
                   digits=2, unit="Å"),
            Option("near", "In register: N1-N3 below", "float", ensemble.NEAR_N,
                   "A pair not bonded but in register ...", minimum=2.0, maximum=12.0, step=0.5, digits=1,
                   unit="Å"),
            Option("near_c1_min", "... with C1'-C1' from", "float", ensemble.NEAR_C1[0], "", minimum=0.0,
                   maximum=20.0, step=0.5, digits=1, unit="Å"),
            Option("near_c1_max", "... to", "float", ensemble.NEAR_C1[1], "", minimum=0.0, maximum=30.0,
                   step=0.5, digits=1, unit="Å"),
            Option("chirality", "Count L riboses", "bool", False,
                   "Sugar chirality of every residue (C1', C3', C4' volumes)."),
            Option("expect_l", "Expected L riboses", "int", None,
                   "Decoys with another number are counted (empty: not checked).", minimum=0,
                   maximum=100000, optional=True, requires={"chirality": True}),
        ]),
        _env(),
        Section("run", "Run", [
            Option("jobs", "Worker processes", "int", 1, "Decoys read in parallel.", minimum=1, maximum=256),
        ]),
    ]

    def describe(self, s):
        names = ", ".join(r["name"] for r in s["registers"]) or "no registers"
        return f"census of {names}"

    def resources(self, s):
        return []

    def check(self, s, ctx):
        out: List[Issue] = []
        _register_issues(s, out)
        _reference_issues(s, out)
        if not ctx.has("models") and not ctx.has("scores"):
            out.append(Issue("error", "connect the decoys, or a decoy table that lists them",
                             ("input:models",)))
        elif not ctx.has("models") and not s["use_table_models"]:
            out.append(Issue("error", "no decoys connected: tick 'Also count the decoys the decoy table "
                                      "lists' or connect them", ("use_table_models",)))
        if not (s["reference"].strip() or s["offsets"].strip() or ctx.has("fasta")
                or _plan_reference(ctx, "models", "scores")):
            out.append(Issue("info", "no reference sequence: the defining pairs must be in the decoys' own "
                                     "numbering, and their bases are not checked", ("reference",)))
        if s["near_c1_min"] > s["near_c1_max"]:
            out.append(Issue("error", "the C1'-C1' window is empty", ("near_c1_min", "near_c1_max")))
        if s["near"] < s["cut"]:
            out.append(Issue("warning", "the in-register limit is below the formed cut", ("near",)))
        if not s["log_dirs"].strip() and not ctx.has("secstruct"):
            out.append(Issue("info", "build conditions come only from logs or *.secstruct files beside the "
                                     "decoys; give the FARFAR2 step's folder under 'Rosetta log folders' or "
                                     "connect its secondary structure", ("log_dirs",)))
        return out

    def output_values(self, s, ctx):
        return {"census": PortValue([ctx.own("census", "tsv")], "table"),
                "report": PortValue([ctx.own("census-report", "txt")], "text")}

    def generate(self, s, ctx):
        q = shell_quote
        script = ctx.name(self.tag, "sh")
        table, report = ctx.name("census", "tsv"), ctx.name("census-report", "txt")
        files: List[GeneratedFile] = []
        ref_file = _reference_file(s, ctx, files, "models", "scores")
        words = _mdatk(s) + ["ensemble", "census"]
        for reg in _registers(s):
            words += ["--register", q(f"{reg.name}={reg.text()}")]
        words += _numbering_words(s, ctx, ref_file)
        for t in ctx.inps("scores"):
            words += ["--scores", q(t)]
        if ctx.has("scores"):
            words += ["--root", _root(ctx)]
            if s["use_table_models"]:
                words.append("--models-from-scores")
        for ss in ctx.inps("secstruct"):
            words += ["--secstruct", q(ss)]
        for d in _paths(s["log_dirs"]):
            words += ["--log-dir", q(_rel_dir(d, ctx))]
        words += ["--group-depth", str(s["group_depth"])]
        if s["pair_suffix"].strip():
            words += ["--pair-suffix", q(s["pair_suffix"].strip())]
        words += ["--cut", f"{s['cut']:g}", "--near", f"{s['near']:g}",
                  "--near-c1", f"{s['near_c1_min']:g}-{s['near_c1_max']:g}"]
        if s["chirality"]:
            words.append("--chirality")
            if s["expect_l"] is not None:
                words += ["--expect-l", str(s["expect_l"])]
        words += ["--jobs", str(s["jobs"]), "-o", table, "--report", report]
        models = ctx.inps("models")
        sh = script_header("MDATK register census", ctx, s, script, [_IMPORTABLE])
        if models:
            sh += ["# The decoys (one group per folder)"] + _array("MODELS", models) + [""]
            words.append('"${MODELS[@]}"')
        sh += [_continued(words), require_file(table, "mdatk ensemble census"),
               require_file(report, "mdatk ensemble census"), f'echo "Census: {table}, report: {report}"', ""]
        files.insert(0, GeneratedFile(script, "\n".join(sh), "run", f"counts the registers into {table}"))
        return files


# ---------------------------------------------------------------------------
# ensemble comparison
# ---------------------------------------------------------------------------

MODES = [("compare", "compare: headline rates"), ("sweep", "sweep: also the closure-threshold sweep"),
         ("pilot", "pilot: two runs of one register")]


@register
class EnsembleCompareTool(Tool):
    name = "ensemble-compare"
    label = "Ensemble comparison (MDATK)"
    category = "Analysis"
    tag = "compare"
    description = ("Two decoy ensembles side by side: backbone closure, defining pairs formed and interface "
                   "atoms in the density, with exact intervals, Fisher tests, a closure-threshold sweep and "
                   "the difference detectable at a stated power. It reports numbers, no verdict.")
    inputs = [
        Port("arm_a", "Arm A decoys (PDB)", ("pdb",), multiple=True,
             tooltip="First ensemble (e.g. a register, or the baseline run)."),
        Port("arm_b", "Arm B decoys (PDB)", ("pdb",), multiple=True, tooltip="Second ensemble."),
        Port("scores", "Decoy table", ("table",), optional=True, multiple=True,
             tooltip="Score terms by name (score, linear_chainbreak ...) for both arms."),
        Port("map", "Density map", ("map",), optional=True, tooltip="For the interface occupancy."),
        Port("fasta", "Reference sequence (FASTA)", ("fasta",), optional=True),
    ]
    outputs = [Port("table", "Comparison (per decoy)", ("table",)),
               Port("report", "Comparison report", ("text",))]
    sections = [
        Section("comparison", "Comparison", [
            Option("mode", "Mode", "radio", "compare",
                   "compare: closure, pairs, both, occupancy, best scores. sweep: also over a range of "
                   "closure cuts. pilot: two runs of the same register (e.g. after a changed fixed-helix "
                   "input), a common contour and optionally matched runs.", choices=MODES),
            Option("label_a", "Name of arm A", "text", "A", ""),
            Option("label_b", "Name of arm B", "text", "B", ""),
        ]),
        Section("registers", "Registers", [
            Option("register_a", "Register of arm A", "text", "",
                   "Name=pairs, e.g. 'pres=A:22-A:37, A:23-A:36'. In pilot mode it is the register of both "
                   "arms.", placeholder="pres=A:22-A:37, A:23-A:36"),
            Option("register_b", "Register of arm B", "text", "",
                   "Name=pairs of arm B (empty: the same as A).", placeholder="inv=A:35-B:60, A:36-B:61",
                   requires={"mode": ("compare", "sweep")}),
            Option("pair_cut", "Formed: N1-N3 below", "float", ensemble.PAIR_CUT, "", minimum=2.0,
                   maximum=6.0, step=0.1, digits=2, unit="Å"),
        ]),
        Section("closure", "Closure", [
            Option("closure", "Closed when", "radio", "term",
                   "A score term of the decoy table below a cut, or every O3'(i)-P(i+1) of consecutive "
                   "nucleotides below a distance, or not tested.",
                   choices=[("term", "score term below a cut"), ("geometry", "O3'-P below a distance"),
                            ("none", "not tested")]),
            Option("term", "Score term", "text", ensemble.CLOSURE_TERM, "Column of the decoy table.",
                   requires={"closure": "term"}),
            Option("cut", "Cut", "float", ensemble.CLOSURE_CUT, "Closed when the term is below this.",
                   minimum=-1e9, maximum=1e9, step=0.5, digits=2, requires={"closure": "term"}),
            Option("o3p_cut", "Longest O3'-P", "float", ensemble.MAX_O3P,
                   "Closed when every O3'-P is below this.", minimum=1.0, maximum=20.0, step=0.1, digits=2,
                   unit="Å", requires={"closure": "geometry"}),
            Option("sweep", "Sweep cuts", "text", "",
                   "Cuts of the sweep (empty: 1 2 3 5 8 10 15 25 inf for a term, 1.7 1.8 2 2.5 3 5 inf Å "
                   "for O3'-P).", placeholder="1 2 3 5 8 10 15 25 inf", requires={"mode": "sweep"}),
        ]),
        Section("density", "Map occupancy", [
            Option("calibration", "Calibration regions", "text", "",
                   "Residues held fixed (in pilot mode: those that did not move) whose atoms calibrate the "
                   "contour; ranges such as 'A:1-6,B:39-44' or group:NAME.",
                   placeholder="A:1-6,A:13-20,B:39-44"),
            Option("regions_a", "Interface regions (arm A)", "text", "",
                   "Residues whose heavy atoms are counted in density.", placeholder="A:25-34,B:58-63"),
            Option("regions_b", "Interface regions (arm B)", "text", "",
                   "Empty: the same as arm A (the same atoms in both arms).", placeholder="(as arm A)"),
            Option("percentile", "Contour percentile", "float", ensemble.CONTOUR_PERCENTILE,
                   "The contour is this percentile of the density at the calibration atoms.", minimum=0.0,
                   maximum=100.0, step=1.0, digits=1),
            Option("contour", "Contour", "radio", "auto",
                   "Per decoy (its own calibration atoms) or one common level for both arms; auto: common "
                   "in pilot mode, per decoy otherwise.",
                   choices=[("auto", "auto"), ("per-decoy", "per decoy"), ("common", "common")]),
            Option("interpolation", "Map values", "radio", "linear",
                   "Trilinear interpolation, or the nearest voxel (as the original scripts).",
                   choices=[("linear", "trilinear"), ("nearest", "nearest voxel")]),
        ], tooltip="Used only when a density map is connected."),
        Section("statistics", "Statistics", [
            Option("alpha", "Alpha", "float", 0.05, "Level of the tests and intervals.", minimum=0.0001,
                   maximum=0.5, step=0.01, digits=3),
            Option("power", "Power", "float", 0.8, "Power of the detectable-difference statement.",
                   minimum=0.5, maximum=0.999, step=0.05, digits=2),
            Option("interval", "Intervals", "radio", "auto",
                   "auto: Wilson in pilot mode, Clopper-Pearson otherwise.",
                   choices=[("auto", "auto"), ("clopper-pearson", "Clopper-Pearson"), ("wilson", "Wilson")]),
            Option("matched_runs", "Matched runs only", "bool", False,
                   "Keep only the runs (decoy table 'run') both arms have, e.g. runs with the same seed.",
                   requires={"mode": "pilot"}),
            Option("log_dirs_a", "Arm A Rosetta logs", "text", "",
                   "Folders of arm A's logs (plan-relative): their seeds are compared run by run.",
                   requires={"mode": "pilot", "matched_runs": True}),
            Option("log_dirs_b", "Arm B Rosetta logs", "text", "", "",
                   requires={"mode": "pilot", "matched_runs": True}),
        ]),
        _numbering_section(),
        _env(),
    ]

    def describe(self, s):
        return f"{s['mode']}: {s['label_a']} vs {s['label_b']}"

    def resources(self, s):
        return []

    def _regs(self, s):
        a = s["register_a"].strip()
        b = s["register_b"].strip() if s["mode"] != "pilot" else ""
        return a, (b or a)

    def check(self, s, ctx):
        out: List[Issue] = []
        a, b = self._regs(s)
        for key, text, label in (("register_a", a, s["label_a"]), ("register_b", b, s["label_b"])):
            if not text:
                out.append(Issue("error", "give the register of arm A (name=pairs)", ("register_a",)))
                break
            try:
                ensemble.parse_register(text, label or "register")
            except ensemble.EnsembleError as exc:
                out.append(Issue("error", str(exc), (key,)))
        _reference_issues(s, out)
        labels = [s["label_a"].strip(), s["label_b"].strip()]
        if not all(labels) or labels[0] == labels[1] or any("," in x for x in labels):
            out.append(Issue("error", "give the arms two different names (no commas)",
                             ("label_a", "label_b")))
        if s["closure"] == "term" and not ctx.has("scores"):
            out.append(Issue("error", f"closure by {s['term']} needs the decoy table: connect it, or test "
                                      "closure by O3'-P", ("closure", "input:scores")))
        if s["closure"] == "term" and not s["term"].strip():
            out.append(Issue("error", "name the score term", ("term",)))
        for key in ("calibration", "regions_a", "regions_b"):
            _check_ranges(s[key], key, ctx, out)
        if ctx.has("map"):
            if not s["calibration"].strip():
                out.append(Issue("error", "with a map, give the calibration regions (residues held fixed)",
                                 ("calibration",)))
            if not s["regions_a"].strip():
                out.append(Issue("error", "with a map, give the interface regions", ("regions_a",)))
            if s["regions_b"].strip() and s["regions_b"].split() != s["regions_a"].split():
                out.append(Issue("info", "the arms' occupancy is measured over different residues",
                                 ("regions_b",)))
        elif s["calibration"].strip() or s["regions_a"].strip():
            out.append(Issue("info", "map occupancy needs a density map; the regions are ignored",
                             ("calibration", "input:map")))
        if s["mode"] == "sweep" and s["sweep"].strip():
            try:
                ensemble.sweep_values(s["sweep"])
            except ValueError:
                out.append(Issue("error", "sweep cuts: numbers (and inf)", ("sweep",)))
        if s["mode"] == "sweep" and s["closure"] == "none":
            out.append(Issue("warning", "the sweep needs a closure test", ("closure",)))
        return out

    def output_values(self, s, ctx):
        return {"table": PortValue([ctx.own("compare", "tsv")], "table"),
                "report": PortValue([ctx.own("compare-report", "txt")], "text")}

    def generate(self, s, ctx):
        q = shell_quote
        script = ctx.name(self.tag, "sh")
        table, report = ctx.name("compare", "tsv"), ctx.name("compare-report", "txt")
        files: List[GeneratedFile] = []
        ref_file = _reference_file(s, ctx, files, "arm_a", "arm_b", "scores")
        a, b = self._regs(s)
        words = _mdatk(s) + ["ensemble", "compare", "--mode", s["mode"],
                             "--labels", q(f"{s['label_a'].strip()},{s['label_b'].strip()}"),
                             "--register-a", q(a), "--register-b", q(b), "--pair-cut", f"{s['pair_cut']:g}"]
        words += _numbering_words(s, ctx, ref_file)
        for t in ctx.inps("scores"):
            words += ["--scores", q(t)]
        if ctx.has("scores"):
            words += ["--root", _root(ctx)]
        words += ["--closure", s["closure"]]
        if s["closure"] == "term":
            words += ["--term", q(s["term"].strip()), f"--cut={s['cut']:g}"]
        elif s["closure"] == "geometry":
            words += [f"--cut={s['o3p_cut']:g}"]
        if s["mode"] == "sweep" and s["sweep"].strip():
            words += ["--sweep", q(" ".join(s["sweep"].split()))]
        if ctx.has("map"):
            regions_b = s["regions_b"] if s["regions_b"].strip() else s["regions_a"]
            words += ["--map", q(ctx.inp("map")), "--calibration", q(_ranges(s["calibration"], ctx)),
                      "--regions-a", q(_ranges(s["regions_a"], ctx)),
                      "--regions-b", q(_ranges(regions_b, ctx)),
                      "--percentile", f"{s['percentile']:g}", "--interpolation", s["interpolation"]]
            if s["contour"] != "auto":
                words += ["--contour", s["contour"]]
        words += ["--alpha", f"{s['alpha']:g}", "--power", f"{s['power']:g}"]
        if s["interval"] != "auto":
            words += ["--interval", s["interval"]]
        if s["mode"] == "pilot" and s["matched_runs"]:
            words.append("--matched-runs")
            for d in _paths(s["log_dirs_a"]):
                words += ["--log-dir-a", q(_rel_dir(d, ctx))]
            for d in _paths(s["log_dirs_b"]):
                words += ["--log-dir-b", q(_rel_dir(d, ctx))]
        words += ["-o", table, "--report", report, "--arm-a", '"${ARM_A[@]}"', "--arm-b", '"${ARM_B[@]}"']
        sh = script_header(f"MDATK ensemble comparison ({s['mode']})", ctx, s, script, [_IMPORTABLE])
        sh += ["# The two ensembles"] + _array("ARM_A", ctx.inps("arm_a"))
        sh += _array("ARM_B", ctx.inps("arm_b"))
        sh += ["", _continued(words), require_file(table, "mdatk ensemble compare"),
               require_file(report, "mdatk ensemble compare"), f'echo "Comparison: {report}"', ""]
        files.insert(0, GeneratedFile(script, "\n".join(sh), "run", f"compares the arms into {report}"))
        return files


# ---------------------------------------------------------------------------
# map-model score
# ---------------------------------------------------------------------------

REGION_OPTIONS = [
    Option("name", "Name", "text", "region", "Name of the region (letters, digits, '.', '_', '-')."),
    Option("ranges", "Residues", "text", "",
           "Chain-qualified ranges such as 'A:25-34,B:58-63', or group:NAME for a project residue group.",
           placeholder="A:25-34,B:58-63"),
]
REGION_TEMPLATES = [("region", "Region", {"name": "interface", "ranges": ""})]


def _score_options(default: str, optional: bool = False) -> List[Option]:
    choices = [("qscore", "Q-score (Phenix)"), ("cc", "per-residue CC (MDATK)")]
    if optional:
        choices = [("none", "do not score")] + choices
    return [
        Option("method", "Score", "radio", default,
               "Q-score: Phenix's Q-score program on a copy with standard residue names (L residues named "
               "A/C/G/U, all ATOM, no hydrogens: Q-score does not depend on handedness). Per-residue CC: "
               "MDATK correlates the map with the model's simulated map around each residue (no external "
               "program).", choices=choices),
        Option("command", "Q-score command", "text", mapscore.QSCORE_COMMAND,
               "Command run in the step's qscore/ folder; {model} is the standard-name copy, {map} the map "
               "(also {stem}, {resolution}). It must write per-atom or per-residue Q-scores (a table with "
               "chain / residue number / Q columns, or Q as B-factors of a PDB). The default follows "
               "Phenix's command-line convention and could not be checked against an installation: adjust "
               "it to yours.", requires={"method": "qscore"}),
        Option("resolution", "Map resolution", "float", None, "For the simulated map of the CC.",
               minimum=0.5, maximum=50.0, step=0.1, digits=2, optional=True, unit="Å",
               requires={"method": "cc"}),
        Option("radius", "Mask radius", "float", mapscore.MASK_RADIUS,
               "Grid points this close to a residue's atoms enter its CC.", minimum=0.5, maximum=10.0,
               step=0.5, digits=1, unit="Å", requires={"method": "cc"}),
        Option("keep_h", "Score hydrogens", "bool", False, "Default: heavy atoms only."),
        Option("timeout", "Time limit per model", "float", None, "Seconds (empty: none).", minimum=1.0,
               maximum=1e7, optional=True, unit="s", requires={"method": "qscore"}),
    ]


def _score_words(s: Dict[str, Any]) -> List[str]:
    q = shell_quote
    words = ["--method", s["method"]]
    if s["method"] == "qscore":
        words += ["--command", q(s["command"].strip()), "--work", "qscore"]
        if s["timeout"] is not None:
            words += ["--timeout", f"{s['timeout']:g}"]
    else:
        words += ["--resolution", f"{s['resolution']:g}", "--radius", f"{s['radius']:g}"]
    if s["keep_h"]:
        words.append("--keep-h")
    return words


def _score_issues(s: Dict[str, Any], out: List[Issue]) -> None:
    if s["method"] == "cc" and s["resolution"] is None:
        out.append(Issue("error", "enter the map resolution (for the simulated map)", ("resolution",)))
    if s["method"] == "qscore":
        cmd = s["command"].strip()
        if not cmd:
            out.append(Issue("error", "enter the Q-score command", ("command",)))
        elif "{model}" not in cmd:
            out.append(Issue("error", "the Q-score command must contain {model}", ("command",)))
        elif "{map}" not in cmd:
            out.append(Issue("warning", "the Q-score command has no {map}", ("command",)))
        out.append(Issue("info", "Q-score runs on a copy with standard residue names (D and L alike); the "
                                 "Phenix command and its output format are read tolerantly, but were not "
                                 "checked against a Phenix installation", ("command",)))


def _score_notes(s: Dict[str, Any]) -> List[str]:
    notes = [_IMPORTABLE]
    if s["method"] == "qscore":
        notes.append("Q-score: Pintilie et al., Nat. Methods 17, 328-334 (2020), "
                     "doi:10.1038/s41592-020-0731-1")
    return notes


@register
class MapModelScoreTool(Tool):
    name = "map-model-score"
    label = "Map-model score (Q-score / CC)"
    category = "Refinement"
    tag = "mapscore"
    description = ("Per-residue fit of models to a map - Phenix's Q-score on a copy with standard residue "
                   "names, or MDATK's per-residue correlation - and the models ranked by named regions "
                   "against a reference region, chain termini left out.")
    inputs = [Port("model", "Models (PDB)", ("pdb",), multiple=True, tooltip="One or more models to score."),
              Port("map", "Density map", ("map",))]
    outputs = [Port("residues", "Per-residue scores", ("table",)), Port("regions", "Regions", ("table",)),
               Port("report", "Ranking report", ("text",))]
    sections = [
        Section("score", "Score", _score_options("qscore")),
        Section("regions", "Regions", REGION_OPTIONS, repeat=True, item_label="name",
                templates=REGION_TEMPLATES, tooltip="Named regions averaged per model (e.g. the interface "
                                                    "loops)."),
        Section("ranking", "Ranking", [
            Option("reference", "Reference region", "text", "",
                   "Ranges of the reference (e.g. the fixed helices); empty: the whole model. The named "
                   "regions are always taken out of it.", placeholder="A:1-6,A:13-20,B:39-44"),
            Option("keep_termini", "Keep chain termini", "bool", False,
                   "The first and last residue of every chain segment have fewer neighbours and lower "
                   "scores; by default they are left out of every mean."),
            Option("sort", "Rank by", "text", "",
                   "A region's name, delta:NAME (its difference to the reference), reference or whole. "
                   "Empty: the first region.", placeholder="delta:interface"),
        ]),
        _env("source /usr/local/phenix-2.0/phenix_env.sh"),
    ]

    def describe(self, s):
        return ("Q-score" if s["method"] == "qscore" else "per-residue CC") + (
            f", {len(s['regions'])} region(s)" if s["regions"] else "")

    def resources(self, s):
        from ..citations import RESOURCES
        return [r for r in (["phenix"] if s["method"] == "qscore" else []) if r in RESOURCES]

    def check(self, s, ctx):
        out: List[Issue] = []
        _score_issues(s, out)
        names = []
        for k, rec in enumerate(s["regions"]):
            try:
                reg = mapscore.parse_region(f"{rec['name'].strip()}={rec['ranges']}", ctx.groups)
                names.append(reg.name)
            except ensemble.EnsembleError as exc:
                out.append(Issue("error", str(exc), (f"regions.{k}.ranges",)))
        if len(set(names)) != len(names):
            out.append(Issue("error", "region names must differ", ("regions",)))
        _check_ranges(s["reference"], "reference", ctx, out)
        sort = s["sort"].strip()
        if sort:
            target = sort.split(":", 1)[1] if sort.startswith("delta:") else sort
            if target not in names + ["reference", "whole"]:
                out.append(Issue("error", f"rank by: no region {target!r}", ("sort",)))
        return out

    def output_values(self, s, ctx):
        return {"residues": PortValue([ctx.own("residues", "tsv")], "table"),
                "regions": PortValue([ctx.own("regions", "tsv")], "table"),
                "report": PortValue([ctx.own("regions-report", "txt")], "text")}

    def generate(self, s, ctx):
        q = shell_quote
        script = ctx.name(self.tag, "sh")
        res, regs = ctx.name("residues", "tsv"), ctx.name("regions", "tsv")
        report = ctx.name("regions-report", "txt")
        words = _mdatk(s) + ["map", "score", "--map", q(ctx.inp("map"))] + _score_words(s)
        for rec in s["regions"]:
            words += ["--region", q(f"{rec['name'].strip()}={_ranges(rec['ranges'], ctx)}")]
        if s["reference"].strip():
            words += ["--reference-region", q(_ranges(s["reference"], ctx))]
        if s["keep_termini"]:
            words.append("--keep-termini")
        if s["sort"].strip():
            words += ["--sort", q(s["sort"].strip())]
        words += ["-o", res, "--regions-out", regs, "--report", report, '"${MODELS[@]}"']
        title = "Q-score" if s["method"] == "qscore" else "per-residue CC"
        sh = script_header(f"Map-model score: {title}", ctx, s, script, _score_notes(s))
        sh += ["# The models"] + _array("MODELS", ctx.inps("model")) + ["", _continued(words)]
        sh += [require_file(p, "mdatk map score") for p in (res, regs, report)]
        sh += [f'echo "Per-residue scores: {res}, regions: {regs}"', ""]
        return [GeneratedFile(script, "\n".join(sh), "run", f"scores the models into {res}")]


# ---------------------------------------------------------------------------
# body hybrids
# ---------------------------------------------------------------------------

BODY_OPTIONS = [
    Option("name", "Name", "text", "body", "Name of the body (letters, digits, '.', '_', '-')."),
    Option("ranges", "Residues", "text", "",
           "Chain-qualified ranges, e.g. 'A:1-6,B:39-44' (a body may span chains), or group:NAME.",
           placeholder="A:1-6,B:39-44"),
]


@register
class BodyHybridsTool(Tool):
    name = "body-hybrids"
    label = "Body hybrids (which re-fit body helped)"
    category = "Refinement"
    tag = "hybrids"
    description = ("Every old/new combination of up to six re-fitted bodies, over the atoms both models have "
                   "(coordinates copied verbatim); with a map, each hybrid is scored and each body's main "
                   "effect reported.")
    inputs = [Port("old", "Old model (PDB)", ("pdb",), tooltip="The model before the re-fit."),
              Port("new", "New model (PDB)", ("pdb",), tooltip="The re-fitted model."),
              Port("map", "Density map", ("map",), optional=True, tooltip="To score the hybrids.")]
    sections = [
        Section("bodies", "Bodies", BODY_OPTIONS, repeat=True, item_label="name",
                templates=[("body", "Body", {"name": "body", "ranges": ""})],
                tooltip=f"Up to {mapscore.MAX_BODIES} bodies; residues in no body keep their old "
                        "coordinates."),
        Section("matching", "Matching", [
            Option("match", "Match residues by", "radio", "chain",
                   "Chain and residue number, or the residue number alone (for two files whose chain ids "
                   "differ; the numbers must then be unique).",
                   choices=[("chain", "chain and number"), ("number", "number alone")]),
        ]),
        Section("score", "Score", _score_options("cc", optional=True)),
        _env(),
    ]

    def _bodies(self, s, ctx: Optional[StepContext] = None) -> List[mapscore.Body]:
        return [mapscore.parse_body(f"{b['name'].strip()}={b['ranges']}", ctx.groups if ctx else ())
                for b in s["bodies"]]

    def output_ports(self, s):
        ports = [Port("hybrids", "Hybrids (PDB)", ("pdb",), multiple=True)]
        if s.get("method", "cc") != "none":
            ports += [Port("scores", "Hybrid scores", ("table",)),
                      Port("effects", "Body main effects", ("table",))]
        return ports

    def describe(self, s):
        n = len(s["bodies"])
        scored = "" if s["method"] == "none" else f", scored by {s['method']}"
        return f"{n} bodies, {2 ** n} hybrids{scored}"

    def resources(self, s):
        from ..citations import RESOURCES
        return [r for r in (["phenix"] if s["method"] == "qscore" else []) if r in RESOURCES]

    def check(self, s, ctx):
        out: List[Issue] = []
        if not s["bodies"]:
            out.append(Issue("error", "define at least one body (name and residues)", ("bodies",)))
        if len(s["bodies"]) > mapscore.MAX_BODIES:
            out.append(Issue("error", f"at most {mapscore.MAX_BODIES} bodies "
                                      f"({2 ** mapscore.MAX_BODIES} hybrids)", ("bodies",)))
        names = []
        for k, rec in enumerate(s["bodies"]):
            try:
                names.append(mapscore.parse_body(f"{rec['name'].strip()}={rec['ranges']}", ctx.groups).name)
            except ensemble.EnsembleError as exc:
                out.append(Issue("error", str(exc), (f"bodies.{k}.ranges",)))
        if len(set(names)) != len(names):
            out.append(Issue("error", "body names must differ", ("bodies",)))
        if s["method"] != "none":
            if not ctx.has("map"):
                out.append(Issue("error", "connect the density map, or choose 'do not score'",
                                 ("method", "input:map")))
            _score_issues(s, out)
        if ctx.has("old") and ctx.inp("old") == ctx.inp("new"):
            out.append(Issue("warning", "the old and the new model are the same file", ("input:new",)))
        old, new = ctx.existing("old"), ctx.existing("new")
        if old and new and not [i for i in out if i.level == "error"]:
            try:
                from .. import pdbtools
                _h, _members, notes = mapscore.make_hybrids(pdbtools.read_pdb(old), pdbtools.read_pdb(new),
                                                            self._bodies(s, ctx), s["match"])
                out += [Issue("info", n) for n in notes]
            except (OSError, ValueError) as exc:
                out.append(Issue("error", str(exc), ("bodies",)))
        return out

    def _names(self, s, ctx):
        codes = mapscore.hybrid_codes(len(s["bodies"]))
        template = ctx.name("hybrid-CODE", "pdb")
        return codes, template

    def output_values(self, s, ctx):
        codes, template = self._names(s, ctx)
        out = {"hybrids": PortValue([ctx.from_own(template.replace("CODE", c)) for c in codes], "pdb")}
        if s["method"] != "none":
            out["scores"] = PortValue([ctx.own("hybrid-scores", "tsv")], "table")
            out["effects"] = PortValue([ctx.own("body-effects", "tsv")], "table")
        return out

    def generate(self, s, ctx):
        q = shell_quote
        script = ctx.name(self.tag, "sh")
        codes, template = self._names(s, ctx)
        words = _mdatk(s) + ["pdb", "hybrids", q(ctx.inp("old")), q(ctx.inp("new"))]
        for rec in s["bodies"]:
            words += ["--body", q(f"{rec['name'].strip()}={_ranges(rec['ranges'], ctx)}")]
        words += ["--match", s["match"], "--out-dir", ".", "--names", q(template.replace("CODE", "{code}"))]
        report = ctx.name("hybrids-report", "txt")
        outs = [template.replace("CODE", c) for c in codes]
        if s["method"] != "none":
            scores, effects = ctx.name("hybrid-scores", "tsv"), ctx.name("body-effects", "tsv")
            words += ["--map", q(ctx.inp("map"))] + _score_words(s)
            words += ["--scores-out", scores, "--effects-out", effects]
            outs += [scores, effects]
        words += ["--report", report]
        title = f"Body hybrids: {len(codes)} combinations of {len(s['bodies'])} bodies"
        sh = script_header(title, ctx, s, script, _score_notes(s) if s["method"] != "none" else [_IMPORTABLE])
        sh += [_continued(words)] + [require_file(p, "mdatk pdb hybrids") for p in outs]
        sh += [f'echo "Hybrids: {template.replace("CODE", "*")}; report: {report}"', ""]
        return [GeneratedFile(script, "\n".join(sh), "run", f"writes the {len(codes)} hybrids")]
