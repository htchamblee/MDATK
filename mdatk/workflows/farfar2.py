"""Rosetta FARFAR2 (rna_denovo) generator.

Writes the Rosetta FASTA / secstruct files from a typed sequence and
dot-bracket, a Rosetta flags file and a script that runs independent seeds in
parallel (``xargs -P``), each in its own folder, writing one silent file per
run.  Defaults follow a working cryo-EM-guided protocol (50 000 cycles,
-fixed_stems, -heat, density weight 20, constant seeds 3001...).
"""

from __future__ import annotations

import inspect
import posixpath
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import PROJECT_ABBREV, __version__, ensemble, modelcheck, pdbtools
from . import rna
from .base import (PortValue, StepContext, Tool, env_section, extra_flags, fortran_float,
                   require_file, script_header, shell_quote)
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section
from .scripts import decoy_validity as _validity
from .scripts import embed
from .scripts import silent_scores as _silent_scores

RUN_DIR = "runs"
MODELS_DIR = "models"  # farfar2-models: every model of every run, models/<run>/<tag>.pdb


def _flag_bool(value: bool) -> str:
    return "true" if value else "false"


def _module_body(module) -> str:
    """Source of a module without its docstring and __future__ import, as :func:`embed` copies it."""
    lines = inspect.getsource(module).splitlines()
    k = 0
    if lines and lines[0].startswith('"""'):
        k = 1
        if not (lines[0].count('"""') >= 2 and len(lines[0]) > 3):
            while k < len(lines) and '"""' not in lines[k]:
                k += 1
            k += 1
    return "\n".join(ln for ln in lines[k:] if not ln.startswith("from __future__")).strip("\n")


def extract_every_model(exe: str, parallel: int, runs: List[List[str]], db: str, helper: str) -> List[str]:
    """Shell lines that extract every model of every run into models/<run>/<tag>.pdb and check that each run
    has one PDB per SCORE line (``runs``: [label, silent file relative to the step folder, ...]; ``helper``:
    the step's copy of :mod:`silent_scores`)."""
    exe = exe.strip() or "extract_pdbs"
    m = MODELS_DIR
    lines = [f"# Extract every model of every run into {m}/<run>/<tag>.pdb. extract_pdbs writes",
             "# <tag>.pdb into the folder it runs in, and every run reuses the same tags, so each run is",
             "# extracted in a scratch folder of its own; the run's earlier models are removed first.",
             "extract_run() {",
             "  local run=$1 silent=$2 tmp f",
             f'  rm -rf "{m}/$run" "{m}/.extract-$run".*',
             f"  mkdir -p {m}",
             f'  tmp=$(mktemp -d "{m}/.extract-$run.XXXXXX")',
             f'  ( cd "$tmp" && {exe}{db} -in:file:silent "$silent" ) > "{m}/extract-$run.log" 2>&1 \\',
             f'    || echo "extract_pdbs reported an error for run $run (see {m}/extract-$run.log)" >&2',
             f'  mkdir -p "{m}/$run"',
             f'  for f in "$tmp"/*.pdb; do if [ -e "$f" ]; then mv -- "$f" "{m}/$run/"; fi; done',
             '  rm -rf "$tmp"',
             "}"]
    for label, path, *_rest in runs:
        src = path if posixpath.isabs(path) else posixpath.join("../..", path)  # from models/.extract-*
        if parallel > 1:
            lines += [f"extract_run {shell_quote(label)} {shell_quote(src)} &",
                      f'[ "$(jobs -rp | wc -l)" -lt {parallel} ] || wait -n || true']
        else:
            lines.append(f"extract_run {shell_quote(label)} {shell_quote(src)}")
    if parallel > 1:
        lines.append("wait")
    lines += ["# every run has one PDB per decoy (SCORE line) and no other PDB",
              f"python3 {helper} check-extracted {m}", ""]
    return lines


def scores_helper(ctx: StepContext, runs: List[List[str]], model_pattern: str) -> GeneratedFile:
    """The step's copy of :mod:`silent_scores` (standard library only) with its runs written in."""
    doc = (f"Rosetta silent-file scores for project {ctx.project}, step {ctx.step:02d} - written by "
           f"{PROJECT_ABBREV} {__version__}.\n\nScore terms are looked up by column name. Usage:\n"
           "    python3 THIS_SCRIPT table OUT.tsv [--extracted models] [--models MODELS.txt] "
           "[--validity VALIDITY.tsv] [--list LIST.txt]\n"
           "    python3 THIS_SCRIPT filter SILENT FILTERED [SILENT FILTERED ...] --term T --max X\n"
           "    python3 THIS_SCRIPT keep [--valid VALIDITY.tsv] [--limit TERM OP VALUE ...]\n"
           "    python3 THIS_SCRIPT filter-ranking RANKING.txt [--term T --max X] "
           "[--limit TERM OP VALUE ...]\n"
           "    python3 THIS_SCRIPT check-extracted models\n"
           "Needs only Python 3.")
    constants = {"RUNS": [tuple(r) for r in runs], "PREFIX": ctx.folder, "MODEL_PATTERN": model_pattern}
    return GeneratedFile(ctx.name("scores", "py"), embed(
        _silent_scores, constants, "main(sys.argv[1:], RUNS, PREFIX, MODEL_PATTERN)", doc), "helper",
        "silent-file scores by column name, the decoy table and the extraction check")


@register
class FarfarTool(Tool):
    name = "farfar2"
    label = "Rosetta · FARFAR2 (rna_denovo)"
    category = "Structure prediction"
    tag = "farfar2"
    description = ("De novo RNA 3D models with Rosetta FARFAR2: FASTA / secstruct from a sequence "
                   "and dot-bracket, optional fixed helices and cryo-EM density, parallel seeds.")
    inputs = [Port("structure", "Fixed helices (-s)", ("pdb",), optional=True,
                   tooltip="PDB of pre-built helices kept rigid during assembly (-s)."),
              Port("map", "Density map", ("map",), optional=True,
                   tooltip="Cryo-EM map (MRC/CCP4) to fit the models into.")]
    sections = [
        Section("sequence", "Sequence & structure", [
            Option("make_inputs", "Write FASTA and secstruct from the fields below", "bool", True,
                   "Generate the Rosetta FASTA (one entry per chain, lower case, A:1-16 numbering) and "
                   "the secstruct file. Off: use existing files."),
            Option("sequence", "Sequence", "multiline", "",
                   "One chain per line or chains separated by spaces / '&' (optionally 'A: ggcac...'). "
                   "Residues are numbered continuously across chains.",
                   placeholder="A: GGGCACGG\nB: CCGUGCCC", requires={"make_inputs": True}),
            Option("dotbracket", "Secondary structure", "multiline", "",
                   "Dot-bracket for the whole sequence (strand separators optional). Paired bases must "
                   "be complementary (G–C, A–U, G–U) or rna_denovo stops.",
                   placeholder="((((((((  ))))))))", requires={"make_inputs": True}),
            Option("fasta_name", "FASTA name", "text", "", "Name on the FASTA header lines; empty uses "
                   "the project name.", requires={"make_inputs": True}),
            Option("fasta_file", "Existing FASTA", "file", "", "FASTA file to use instead.",
                   requires={"make_inputs": False}),
            Option("secstruct_file", "Existing secstruct", "file", "", "Secondary structure file to "
                   "use instead (dot-bracket on line 1).", requires={"make_inputs": False}),
        ], tooltip="What to fold."),
        Section("sampling", "Sampling", [
            Option("nstruct", "Models per run (-nstruct)", "int", 100, "Models written by each run.",
                   minimum=1, maximum=1_000_000),
            Option("cycles", "Monte Carlo cycles (-cycles)", "int", 50000, "Fragment-assembly cycles "
                   "per model (Rosetta default 10 000).", minimum=1, maximum=100_000_000),
            Option("minimize_rna", "Full-atom refinement (-minimize_rna)", "bool", True,
                   "High-resolution minimisation after assembly - strongly recommended."),
            Option("fixed_stems", "Fixed stems (-fixed_stems)", "bool", True,
                   "Seed every helix of the secondary structure with Watson–Crick pairs."),
            Option("bps_moves", "Base-pair-step moves", "bool", False,
                   "-rna:denovo:bps_moves (helix flexibility by base pair step fragments)."),
            Option("heat", "Heat (-heat)", "bool", True, "Randomise torsions before assembly."),
            Option("close_loops", "Close loops (-close_loops)", "bool", True,
                   "Close chain breaks by cyclic coordinate descent after fragment moves."),
            Option("fuzz_fragments", "Fuzz fragments", "int", 5, "-fuzz_fragments value (0: not set).",
                   minimum=0, maximum=100),
            Option("autofilter", "Autofilter (-autofilter)", "bool", False, "Keep only models that "
                   "score among the best during the run."),
            Option("filter_lores_base_pairs", "Filter low-res base pairs", "bool", False,
                   "-filter_lores_base_pairs: discard models missing the secondary-structure pairs."),
            Option("filter_chain_closure", "Filter chain closure", "bool", True,
                   "-filter_chain_closure: discard models with open chain breaks."),
            Option("filter_chain_closure_halfway", "… also halfway", "bool", False,
                   "-filter_chain_closure_halfway", requires={"filter_chain_closure": True}),
            Option("filter_chain_closure_distance", "… distance", "float", 15.0,
                   "-filter_chain_closure_distance (Å).", minimum=0.0, maximum=100.0, step=0.5, digits=1,
                   requires={"filter_chain_closure": True}),
            Option("coord_csts", "Coordinate constraints in minimiser", "bool", True,
                   "-rna:denovo:minimize:minimizer_use_coordinate_constraints"),
            Option("dump", "Dump PDBs (-rna:denovo:out:dump)", "bool", True,
                   "Also write PDB files into each run folder."),
            Option("input_res_mode", "Residues of the -s PDB", "radio", "typed",
                   "-input_res: residues provided by the fixed-helix PDB.",
                   choices=[("typed", "As typed"), ("stems", "All helix residues")]),
            Option("input_res", "-input_res", "text", "", "Chain-qualified residues, e.g. "
                   "'A:1-6 A:13-16 B:17-20'.", placeholder="A:1-6 B:39-44",
                   requires={"input_res_mode": "typed"}),
        ], tooltip="rna_denovo options."),
        Section("density", "Density map", [
            Option("map_reso", "Map resolution (-edensity:mapreso)", "float", 4.3,
                   "Resolution of the connected map (Å).", minimum=0.5, maximum=50.0, step=0.1,
                   digits=2, unit="Å"),
            Option("cryoem_scatterers", "Cryo-EM scatterers", "bool", True,
                   "-edensity:cryoem_scatterers (electron rather than X-ray form factors)."),
            Option("w_density", "Density weight (elec_dens_fast)", "float", 20.0,
                   "Score weight of the fit to the map.", minimum=0.0, maximum=1000.0, step=1.0, digits=2),
        ], tooltip="Used only when a density map is connected or chosen."),
        Section("weights", "Score weights", [
            Option("w_coordinate_constraint", "coordinate_constraint", "float", 1.0,
                   "-set_weights coordinate_constraint (0: not set).", minimum=0.0, maximum=1000.0,
                   step=0.5, digits=2),
            Option("w_linear_chainbreak", "linear_chainbreak", "float", 10.0,
                   "-set_weights linear_chainbreak (0: not set).", minimum=0.0, maximum=1000.0,
                   step=0.5, digits=2),
        ]),
        Section("runs", "Runs", [
            Option("runs", "Independent runs", "int", 12, "Number of rna_denovo runs (seeds); each "
                   "writes its own silent file.", minimum=1, maximum=10000),
            Option("parallel", "Parallel runs", "int", 6, "Runs started at the same time (xargs -P).",
                   minimum=1, maximum=1024),
            Option("constant_seed", "Reproducible seeds", "bool", True,
                   "-run:constant_seed with -run:jran <seed base + run number>."),
            Option("seed_base", "Seed base", "int", 3000, "Run i uses seed base + i.", minimum=0,
                   maximum=2**31 - 1000000, requires={"constant_seed": True}),
            Option("nice", "Run with nice", "bool", True, "Lower the CPU priority of the runs."),
            Option("exe", "rna_denovo executable", "text", "rna_denovo",
                   "Name on PATH or full path, e.g. rna_denovo.default.linuxgccrelease."),
            Option("database", "Rosetta database", "text", "", "-database path (empty: found by the "
                   "executable)."),
            Option("extra", "Extra flags", "multiline", "", "Additional rna_denovo flags, one or more "
                   "per line (written to the flags file).", placeholder="-vary_geometry"),
        ]),
        Section("decoys", "Decoys (every model of every run)", [
            Option("extract_decoys", "Extract every decoy after the runs", "bool", False,
                   "Every model of every run as models/<run>/<tag>.pdb (runs r01, r02 ...), each run checked "
                   "against the SCORE lines of its silent file, the decoy table "
                   "<project>_<step>_scores_<time>.tsv and the list of the decoys' PDB files "
                   "<project>_<step>_decoys_<time>.txt: the outputs 'Every decoy' (the full set, a large "
                   "dot) and 'Decoy table'. Connecting either switches this on. A script of its own, "
                   "<project>_<step>_decoys_<time>.sh, extracts them; the step's script runs it after the "
                   "runs, and it can be run alone to extract again without running rna_denovo."),
            Option("extract_exe", "extract_pdbs executable", "text", "extract_pdbs",
                   "Name on PATH or full path (the Rosetta database is the one above).",
                   requires={"extract_decoys": True}),
            Option("extract_parallel", "Parallel extractions", "int", 1, "Runs extracted at the same time.",
                   minimum=1, maximum=256, requires={"extract_decoys": True}),
        ], tooltip="The full set of decoys for the steps that classify many models (model check, decoy "
                   "interfaces, register census, ensemble comparison, frame check, map-model score)."),
        env_section("export PATH=$HOME/rosetta/main/source/bin:$PATH"),
    ]

    def output_ports(self, s):
        return [Port("silent", "Silent files", ("silent",), multiple=True),
                Port("fasta", "FASTA", ("fasta",)),
                Port("secstruct", "Secondary structure", ("secstruct",)),
                Port("decoys", "Every decoy (PDB)", ("pdb",), collection=True, turns_on="extract_decoys",
                     table="decoy_table",
                     tooltip="The full set: every model of every run, extracted from the silent files after "
                             "the runs (models/<run>/<tag>.pdb), for the steps that take many models. "
                             "Connecting it switches on 'Extract every decoy after the runs'; into an input "
                             "that takes a decoy table it connects the decoy table."),
                Port("decoy_table", "Decoy table", ("table",), turns_on="extract_decoys",
                     tooltip="One row per decoy of every run: run, tag, its extracted PDB (relative to the "
                             "plan folder) and every SCORE term by name.")]

    # ------------------------------------------------------------ model
    def model(self, s) -> Tuple[Optional[rna.RNAModel], List[Issue]]:
        if not s["make_inputs"]:
            path = s["fasta_file"].strip()
            ss = s["secstruct_file"].strip()
            try:
                seq = open(path).read() if path else ""
                db = open(ss).read() if ss else ""
                return rna.build_model(seq, db), []
            except (OSError, ValueError):
                return None, []
        try:
            model = rna.build_model(s["sequence"], s["dotbracket"])
        except ValueError as exc:
            return None, [Issue("error", str(exc), ("sequence", "dotbracket"))]
        return model, []

    def check(self, s, ctx):
        out: List[Issue] = []
        if s["make_inputs"]:
            model, issues = self.model(s)
            out += issues
            if model is not None:
                if not model.chains:
                    out.append(Issue("error", "enter the RNA sequence", ("sequence",)))
                if not model.dotbracket and s["fixed_stems"]:
                    out.append(Issue("warning", "-fixed_stems needs a secondary structure",
                                     ("dotbracket", "fixed_stems")))
                for i, j in model.pairs:
                    kind = model.pair_type(i, j)
                    if kind not in rna.CANONICAL + rna.WOBBLE:
                        out.append(Issue("error", f"residues {i} and {j} ({kind[0]}–{kind[1]}) are paired "
                                                  "in the dot-bracket but not complementary; rna_denovo "
                                                  "stops on this", ("dotbracket",)))
                        break
                if not model.pairs and model.chains and s["dotbracket"].strip():
                    out.append(Issue("info", "the dot-bracket contains no pairs", ("dotbracket",)))
        else:
            if not s["fasta_file"].strip():
                out.append(Issue("error", "choose the FASTA file", ("fasta_file",)))
        if ctx.has("map"):
            if s["w_density"] <= 0:
                out.append(Issue("warning", "a density map is connected but its score weight is 0",
                                 ("w_density",)))
        else:
            if s["w_density"] != 20.0 or s["map_reso"] != 4.3:
                out.append(Issue("info", "density options are ignored without a map", ("map_reso",)))
        if ctx.has("structure"):
            if s["input_res_mode"] == "typed" and not s["input_res"].strip():
                out.append(Issue("error", "say which residues the fixed-helix PDB provides (-input_res) "
                                          "or choose 'All helix residues'", ("input_res",)))
        elif s["input_res"].strip() or s["input_res_mode"] == "stems":
            out.append(Issue("warning", "-input_res is set but no fixed-helix PDB (-s) is connected",
                             ("input_res", "input_res_mode")))
        if s["input_res_mode"] == "stems":
            model, _ = self.model(s)
            if model is None or not model.pairs:
                out.append(Issue("error", "'All helix residues' needs a secondary structure",
                                 ("input_res_mode",)))
        if s["coord_csts"] and s["w_coordinate_constraint"] == 0:
            out.append(Issue("warning", "minimiser coordinate constraints are on but their weight is 0",
                             ("w_coordinate_constraint",)))
        if s["parallel"] > s["runs"]:
            out.append(Issue("info", "more parallel slots than runs", ("parallel",)))
        if s["autofilter"] and s["nstruct"] < 20:
            out.append(Issue("info", "autofilter keeps few models when -nstruct is small", ("autofilter",)))
        if not s["minimize_rna"]:
            out.append(Issue("warning", "without -minimize_rna the models are low-resolution only",
                             ("minimize_rna",)))
        for flag in extra_flags(s["extra"]):
            name = flag.split()[0]
            if name in self._generated_flags(s, ctx):
                out.append(Issue("warning", f"extra flag {name} repeats a generated flag", ("extra",)))
        if s["extract_decoys"]:
            out.append(Issue("info", f"every decoy is extracted after the runs: up to "
                                     f"{s['runs'] * s['nstruct']} PDB files in {MODELS_DIR}/",
                             ("extract_decoys",)))
        return out

    # ------------------------------------------------------------ names
    def _inputs(self, s, ctx) -> Tuple[str, str]:
        if s["make_inputs"]:
            return ctx.name(self.tag, "fasta"), ctx.name(self.tag, "secstruct")
        return s["fasta_file"].strip(), s["secstruct_file"].strip()

    def output_values(self, s, ctx):
        fasta, ss = self._inputs(s, ctx)
        model, _ = self.model(s)
        data = {"rna_model": model} if model is not None else {}
        # the weight of the fit to the map while sampling (models drawn towards one placement in the map)
        dens = {"density_weight": float(s["w_density"]) if ctx.has("map") else 0.0}
        out = {"silent": PortValue([ctx.own(f"{self.tag}-r{k:02d}", "out") for k in range(1, s["runs"] + 1)],
                                   "silent", dict(data, **dens))}
        if fasta:
            out["fasta"] = PortValue([ctx.from_own(fasta) if s["make_inputs"] else fasta], "fasta", dict(data))
        if ss:
            out["secstruct"] = PortValue([ctx.from_own(ss) if s["make_inputs"] else ss], "secstruct",
                                         dict(data))
        if s["extract_decoys"]:
            # the decoys are known only after the runs: the extraction lists them (at most runs x nstruct)
            listing = ctx.own("decoys", "txt")
            fixed = {"input_res": self.fixed_residues(s, ctx)}  # what the decoy interfaces' support reads
            out["decoys"] = PortValue([listing], "pdb", dict(data, **dens, **fixed),
                                      lists={listing: s["runs"] * s["nstruct"]})
            out["decoy_table"] = PortValue([ctx.own("scores", "tsv")], "table", dict(data, **dens))
        return out

    def fixed_residues(self, s, ctx) -> str:
        """The residues the fixed-helix PDB provides (-input_res); '' without one: every residue is built."""
        if not ctx.has("structure"):
            return ""
        if s["input_res_mode"] == "stems":
            model, _ = self.model(s)
            return model.ranges([r for p in model.pairs for r in p]) if model is not None else ""
        return s["input_res"].strip()

    def resources(self, s):
        return ["rosetta", "farfar2"]

    def describe(self, s):
        model, _ = self.model(s)
        size = f"{model.length} nt, " if model and model.chains else ""
        return f"{size}{s['runs']} runs × {s['nstruct']} models"

    # ------------------------------------------------------------ flags
    def _generated_flags(self, s, ctx) -> List[str]:
        return [f.split()[0] for f in self._flags(s, ctx, "")]

    def _flags(self, s, ctx: StepContext, up: str) -> List[str]:
        """Flags shared by all runs; ``up`` is the path from a run folder to the step folder."""
        def rel(path: str) -> str:
            if not path or path.startswith("/"):
                return path
            return posixpath.normpath(posixpath.join(up, path)) if up else path

        fasta, ss = self._inputs(s, ctx)
        flags = []
        if s["database"].strip():
            flags.append(f"-database {s['database'].strip()}")
        flags.append(f"-fasta {rel(fasta)}")
        if ss:
            flags.append(f"-secstruct_file {rel(ss)}")
        flags.append(f"-nstruct {s['nstruct']}")
        flags.append(f"-cycles {s['cycles']}")
        flags.append(f"-minimize_rna {_flag_bool(s['minimize_rna'])}")
        if s["fixed_stems"]:
            flags.append("-fixed_stems")
        flags.append(f"-rna:denovo:bps_moves {_flag_bool(s['bps_moves'])}")
        if s["heat"]:
            flags.append("-heat")
        if s["close_loops"]:
            flags.append("-close_loops")
        if s["fuzz_fragments"]:
            flags.append(f"-fuzz_fragments {s['fuzz_fragments']}")
        flags.append(f"-autofilter {_flag_bool(s['autofilter'])}")
        flags.append(f"-filter_lores_base_pairs {_flag_bool(s['filter_lores_base_pairs'])}")
        flags.append(f"-filter_chain_closure {_flag_bool(s['filter_chain_closure'])}")
        if s["filter_chain_closure"]:
            flags.append(f"-filter_chain_closure_halfway {_flag_bool(s['filter_chain_closure_halfway'])}")
            flags.append(f"-filter_chain_closure_distance {fortran_float(s['filter_chain_closure_distance'])}")
        if s["coord_csts"]:
            flags.append("-rna:denovo:minimize:minimizer_use_coordinate_constraints")
        if s["dump"]:
            flags.append("-rna:denovo:out:dump")
        if ctx.has("structure"):
            flags.append(f"-s {rel(ctx.inp('structure'))}")
            if s["input_res_mode"] == "stems":
                model, _ = self.model(s)
                if model is not None:
                    flags.append(f"-input_res {model.ranges([r for p in model.pairs for r in p])}")
            elif s["input_res"].strip():
                flags.append(f"-input_res {s['input_res'].strip()}")
        weights = []
        if ctx.has("map"):
            if s["cryoem_scatterers"]:
                flags.append("-edensity:cryoem_scatterers")
            flags.append(f"-edensity:mapfile {rel(ctx.inp('map'))}")
            flags.append(f"-edensity:mapreso {fortran_float(s['map_reso'])}")
            if s["w_density"]:
                weights.append(f"elec_dens_fast {fortran_float(s['w_density'])}")
        if s["w_coordinate_constraint"]:
            weights.append(f"coordinate_constraint {s['w_coordinate_constraint']:g}")
        if s["w_linear_chainbreak"]:
            weights.append(f"linear_chainbreak {fortran_float(s['w_linear_chainbreak'])}")
        if weights:
            flags.append("-set_weights " + " ".join(weights))
        if s["constant_seed"]:
            flags.append("-run:constant_seed")
        return flags

    # ------------------------------------------------------------ files
    def generate(self, s, ctx):
        files: List[GeneratedFile] = []
        model, _ = self.model(s)
        name = (s["fasta_name"] or ctx.project).strip()
        if s["make_inputs"] and model is not None:
            files.append(GeneratedFile(ctx.name(self.tag, "fasta"), rna.fasta_text(model, name), "input",
                                       "Rosetta FASTA (one entry per chain)"))
            if model.dotbracket:
                files.append(GeneratedFile(ctx.name(self.tag, "secstruct"), rna.secstruct_text(model),
                                           "input", "secondary structure (line 1 is read)"))
        up = "../.."  # from runs/rNN back to the step folder
        flags_name = ctx.name(self.tag, "flags")
        flag_lines = [f"# rna_denovo flags written by {PROJECT_ABBREV} {__version__} for project "
                      f"{ctx.project}, step {ctx.step:02d}",
                      "# paths are relative to the run folders (runs/rNN)"]
        flag_lines += self._flags(s, ctx, up) + extra_flags(s["extra"])
        files.append(GeneratedFile(flags_name, "\n".join(flag_lines) + "\n", "input", "rna_denovo flags"))
        script = ctx.name(self.tag, "sh")
        silent = ctx.name(f"{self.tag}-rRUNNUM", "out").replace("RUNNUM", "${N}")
        log = ctx.name(f"{self.tag}-rRUNNUM", "log").replace("RUNNUM", "${N}")
        nice = "nice " if s["nice"] else ""
        seed = f" -run:jran $(({s['seed_base']} + I))" if s["constant_seed"] else ""
        sh = script_header(f"Rosetta FARFAR2: {s['runs']} rna_denovo runs, {s['parallel']} at a time",
                           ctx, s, script,
                           ["Each run works in runs/rNN and writes ../../" + silent.replace("${N}", "NN") + "."])
        sh = [ln for ln in sh if ln != "set -eo pipefail"]  # a failed run must not stop the others
        sh += [f"FLAGS={flags_name}", "",
               "run_one() {",
               "  I=$1",
               '  N=$(printf "%02d" "$I")',
               f'  mkdir -p "{RUN_DIR}/r$N" && cd "{RUN_DIR}/r$N" || return 0',
               f"  {nice}{s['exe'].strip()} @{up}/$FLAGS -out:file:silent {up}/{silent}{seed} \\",
               f"    > {up}/{log} 2>&1 || echo \"run $N failed - see {log}\" >&2",
               "}",
               "export -f run_one",
               "export FLAGS",
               "",
               f"seq 1 {s['runs']} | xargs -P {s['parallel']} -I{{}} bash -c 'run_one {{}}'",
               "",
               f'echo "Silent files: {ctx.name(self.tag + "-rRUNNUM", "out").replace("RUNNUM", "*")}"', ""]
        if s["extract_decoys"]:
            decoys = ctx.name("decoys", "sh")
            sh += ["# Every decoy of every run as a PDB file, the decoy table and the list of the decoys:",
                   "# a script of its own, which can also be run alone to extract again (without rna_denovo)",
                   f'bash {decoys} || {{ echo "ERROR: the decoys were not extracted (see above)" >&2; '
                   "exit 1; }", ""]
        files.append(GeneratedFile(script, "\n".join(sh), "run", "runs rna_denovo"))
        if s["extract_decoys"]:
            files += self._decoy_files(s, ctx)
        return files

    def _decoy_files(self, s, ctx) -> List[GeneratedFile]:
        """The extraction script of the full set of decoys and its copy of the silent-file scores."""
        n = ctx.name
        script, listing = n("decoys", "sh"), n("decoys", "txt")
        table, helper = n("scores", "tsv"), n("scores", "py")
        silents = [n(f"{self.tag}-r{k:02d}", "out") for k in range(1, s["runs"] + 1)]
        runs = [[label, path] for label, path in zip(_silent_scores.unique_labels(silents), silents)]
        db = f" -database {s['database'].strip()}" if s["database"].strip() else ""
        sh = script_header(f"FARFAR2: every decoy of the {s['runs']} runs as a PDB file, the decoy table and "
                           "the list of the decoys", ctx, s, script,
                           ["The step's script runs this after the rna_denovo runs; run it alone to extract "
                            "again (the runs are not repeated)."])
        sh += ["# Outputs of an earlier extraction go first: a failed one leaves nothing that looks finished",
               f"rm -f {listing} {table}", ""]
        sh += extract_every_model(s["extract_exe"], s["extract_parallel"], runs, db, helper)
        sh += ["# The decoy table (run, tag, pdb and every SCORE term by name) and the list of the",
               "# decoys' PDB files (from the plan folder) in its order, which the steps taking the set read",
               f"python3 {helper} table {table} --extracted {MODELS_DIR} --term score --list {listing}",
               require_file(table, "the decoy table"), require_file(listing, "the list of the decoys"),
               f'echo "Decoys: $(wc -l < {listing}) PDB files in {MODELS_DIR}/, listed in {listing}; '
               f'decoy table: {table}"', ""]
        return [GeneratedFile(script, "\n".join(sh), "helper", "extracts every decoy, writes the decoy table "
                              "and the list of the decoys"),
                scores_helper(ctx, runs, "")]


# ---------------------------------------------------------------------------
# follow-up: cluster, extract and annotate the models
# ---------------------------------------------------------------------------

RANK_AWK = r'''awk -v f="$1" '$1=="SCORE:" { if ($0 ~ /description/) { s=2; d=NF; for (i=2; i<=NF; i++) { if ($i=="score") s=i; if ($i=="description") d=i }; next } print $s, $d, f }' "$1"'''


@register
class FarfarModelsTool(Tool):
    name = "farfar2-models"
    label = "Rosetta · FARFAR2 models (cluster / extract / DSSR)"
    category = "Structure prediction"
    tag = "farfar2-models"
    description = ("Clusters the FARFAR2 silent files with rna_cluster, extracts the cluster centres "
                   "(or the lowest-energy models) with extract_pdbs and annotates them with x3dna-dssr.")
    inputs = [Port("silent", "Silent files", ("silent",), multiple=True,
                   tooltip="The silent files of an rna_denovo step (one per run).")]
    outputs = [Port("best", "Best model (PDB)", ("pdb",)),
               Port("models", "Ranked models (PDB)", ("pdb",), multiple=True),
               Port("scores", "Decoy table (scores)", ("table",),
                    tooltip="One row per model of every run: run, tag, the extracted PDB (plan-relative) and "
                            "every SCORE term by name; with validity filters also their columns and 'valid'. "
                            "Written with 'Write the decoy table', a filter or 'Extract every model'."),
               Port("validity", "Validity report", ("text",),
                    tooltip="With validity filters: the decoys left after each criterion in order, each "
                            "criterion on every decoy, and the valid decoys of each run.")]
    sections = [
        Section("cluster", "Clustering", [
            Option("cluster", "Cluster with rna_cluster", "bool", True,
                   "Group models by all-heavy-atom RMSD, best (lowest energy) cluster first. Off: rank all "
                   "models of all runs by score."),
            Option("radius", "Cluster radius (-cluster:radius)", "float", 2.0, "RMSD threshold (Å).",
                   minimum=0.1, maximum=50.0, step=0.5, digits=2, unit="Å", requires={"cluster": True}),
            Option("max_clusters", "Maximum clusters (-nstruct)", "int", 400, "", minimum=1,
                   maximum=100000, requires={"cluster": True}),
            Option("score_diff_cut", "Score window (-cluster:score_diff_cut)", "float", None,
                   "Only cluster models within this score of the best (empty: all).", optional=True,
                   requires={"cluster": True}),
            Option("cluster_exe", "rna_cluster executable", "text", "rna_cluster", "",
                   requires={"cluster": True}),
            Option("cluster_extra", "Extra rna_cluster flags", "multiline", "", "", requires={"cluster": True}),
        ]),
        Section("validity", "Validity filters (before clustering)", [
            Option("valid_hands", "Hands (sugar chirality)", "bool", False,
                   "Every nucleotide's sugar hand from its geometry (C4', C3', C1') against the counts and / "
                   "or ranges below; an epimerised or flat sugar fails too."),
            Option("hands_counts", "Expected counts", "text", "",
                   "How many D and L sugars every decoy has (empty: not counted).", placeholder="44 D / 29 L",
                   requires={"valid_hands": True}),
            Option("hands_l", "Residues expected L", "text", "",
                   "Chain-qualified ranges ('C' or 'C:*' for the whole chain, '*:45-73' for every chain) or "
                   "group:NAME.", placeholder="C:*", requires={"valid_hands": True}),
            Option("hands_d", "Residues expected D", "text", "", "As above.", placeholder="A:*, B:*",
                   requires={"valid_hands": True}),
            Option("valid_bases", "Base identity against the sequence", "bool", False,
                   "Every nucleotide's base (from its atoms) at its chain and number must be the sequence's; "
                   "missing and extra residues fail."),
            Option("valid_sequence", "Sequence", "multiline", "",
                   "Empty: the sequence of the connected FARFAR2 step. Otherwise FASTA with chain:range tags "
                   "('>apt A:1-16') or 'A: GGGC... B: ...'.", placeholder="A: GGGCACGG\nB: CCGUGCCC",
                   requires={"valid_bases": True}),
            Option("valid_closure", "Chain closure", "bool", False,
                   "A score term by name and / or the O3'-P distance of every link (consecutive nucleotides "
                   "of a chain)."),
            Option("closure_mode", "Closure from", "radio", "term", "",
                   choices=[("term", "a score term"), ("geometry", "O3'-P distances"), ("both", "both")],
                   requires={"valid_closure": True}),
            Option("closure_term", "Score term", "text", "linear_chainbreak",
                   "Column name in the silent files' SCORE header (looked up by name, never by position).",
                   requires={"valid_closure": True, "closure_mode": ("term", "both")}),
            Option("closure_max", "Maximum", "float", 5.0,
                   "A decoy with a larger value (or none) is invalid.", minimum=-1e9, maximum=1e9, step=0.5,
                   digits=2,
                   requires={"valid_closure": True, "closure_mode": ("term", "both")}),
            Option("closure_o3p", "Longest O3'-P", "float", 2.0, "Every O3'(i)-P(i+1) at most this (the bond "
                   "is 1.6 Å); an unmeasurable link is invalid.", minimum=1.0, maximum=20.0, step=0.1,
                   digits=2, unit="Å",
                   requires={"valid_closure": True, "closure_mode": ("geometry", "both")}),
            Option("valid_pairs", "Required pairs", "bool", False,
                   "The defining pairs of a register must be formed (Watson-Crick N1/N3 < 3.5 Å) or "
                   "approached (N1-N3 < 6 Å, C1'-C1' 8-13 Å), as the register census counts them."),
            Option("register", "Register", "text", "",
                   "NAME=pairs (the name is optional), numbered as the decoys.",
                   placeholder="pres=B:22-B:37, B:23-B:36, B:24-B:35", requires={"valid_pairs": True}),
            Option("pairs_mode", "Each pair", "radio", "formed", "",
                   choices=[("formed", "formed"), ("approached", "formed or approached")],
                   requires={"valid_pairs": True}),
        ], tooltip="What makes a decoy not a model of the construct: judged on every decoy (so every model "
                   "is extracted), in this order, before clustering. Only the valid decoys are clustered, "
                   "from <project>_<step>_filtered-rNN_<time>.out, so the cluster populations are those of "
                   "the valid set."),
        Section("filter", "Quality filters (after clustering, choosing representatives)", [
            Option("filter", "Filter the ranked models by score", "bool", False,
                   "Skip the ranked cluster centres (or ranked models) outside the score limits when the "
                   "representatives are extracted. Terms are looked up by column name; a model without a "
                   "value is skipped."),
            Option("filter_limits", "Score limits", "text", "",
                   "TERM<=VALUE (also <, >=, >), separated by commas, e.g. elec_dens_fast (map fit), "
                   "atom_pair_constraint or score.", placeholder="score <= -300, elec_dens_fast <= -40",
                   requires={"filter": True}),
            Option("filter_when", "Apply", "radio", "after",
                   "After clustering: the ranked cluster centres outside the limits are skipped. Before "
                   "clustering removes decoys by score before the populations are counted, which changes "
                   "them: use validity filters for that.",
                   choices=[("after", "after clustering (to the ranked cluster centres)"),
                            ("before", "before clustering (changes cluster populations)")],
                   requires={"filter": True}),
        ], tooltip="Score limits for choosing representatives. None by default."),
        Section("extract", "Extraction", [
            Option("n_models", "Models to extract", "int", 10, "Cluster centres (or best models) written as "
                   "PDB files <project>_<step>_model-01_<time>.pdb ...", minimum=1, maximum=10000),
            Option("extract_exe", "extract_pdbs executable", "text", "extract_pdbs", ""),
            Option("database", "Rosetta database", "text", "", "-database path (empty: found by the "
                   "executables)."),
            Option("extract_all", "Extract every model of every run", "bool", False,
                   "Also write every model as models/<run>/<tag>.pdb (run r01, r02 ... from the silent file "
                   "names). extract_pdbs works in a scratch folder per run, since every run reuses the same "
                   "tags; the earlier models of a run are removed first. The step checks that each run has "
                   "one PDB per SCORE line and fills the pdb column of the decoy table. Validity filters "
                   "extract every model in any case."),
            Option("extract_parallel", "Parallel extractions", "int", 1, "Runs extracted at the same time "
                   "when every model is extracted (also for the validity filters).", minimum=1, maximum=256),
        ]),
        Section("scores", "Decoy table", [
            Option("write_scores", "Write the decoy table", "bool", False,
                   "<project>_<step>_scores_<time>.tsv: one row per model of every run with run, tag, the "
                   "extracted PDB (relative to the plan folder; empty when not extracted) and every SCORE "
                   "term by name, and a column audit in the log. Always written with a filter or 'Extract "
                   "every model'; with validity filters it also holds their columns and 'valid' (1 / 0). "
                   "Steps that take a decoy table read it from the output 'scores'."),
        ]),
        Section("dssr", "DSSR", [
            Option("dssr", "Annotate with x3dna-dssr", "bool", True,
                   "Base pairs, stems and secondary structure of every extracted model."),
            Option("dssr_json", "Also write JSON", "bool", False, "x3dna-dssr --json output.",
                   requires={"dssr": True}),
            Option("dssr_exe", "x3dna-dssr executable", "text", "x3dna-dssr", "", requires={"dssr": True}),
        ]),
        env_section("export PATH=$HOME/rosetta/main/source/bin:$PATH"),
    ]

    def resources(self, s):
        return ["rosetta", "farfar2"] + (["dssr"] if s["dssr"] else [])

    def describe(self, s):
        return (f"rna_cluster {s['radius']:g} Å, " if s["cluster"] else "by score, ") + f"top {s['n_models']}"

    def normalise(self, settings, issues=None):
        # the single score filter of 0.0.12 development builds (filter_term <= filter_max) is a limit now
        settings = dict(settings or {})
        term, maximum = settings.pop("filter_term", None), settings.pop("filter_max", None)
        if "filter_limits" not in settings and (term is not None or maximum is not None):
            try:
                settings["filter_limits"] = (f"{str(term or 'linear_chainbreak').strip()} <= "
                                             f"{float(5.0 if maximum is None else maximum):g}")
            except (TypeError, ValueError):
                pass
        return super().normalise(settings, issues)

    @staticmethod
    def validity_on(s) -> List[str]:
        """The validity criteria switched on, in the order they are applied."""
        return [c for c, key in (("hands", "valid_hands"), ("bases", "valid_bases"),
                                 ("closure", "valid_closure"), ("pairs", "valid_pairs")) if s[key]]

    @classmethod
    def writes_table(cls, s) -> bool:
        """The decoy table is written on request, and always with a filter or the extraction of every model
        (it is their record)."""
        return bool(s["write_scores"] or s["filter"] or s["extract_all"] or cls.validity_on(s))

    def output_values(self, s, ctx):
        models = [ctx.own(f"model-{k:02d}", "pdb") for k in range(1, s["n_models"] + 1)]
        data = dict(ctx.data("silent"))
        out = {"best": PortValue(models[:1], "pdb", data), "models": PortValue(models, "pdb", data)}
        if self.writes_table(s):
            out["scores"] = PortValue([ctx.own("scores", "tsv")], "table", dict(data))
        if self.validity_on(s):
            out["validity"] = PortValue([ctx.own("validity", "txt")], "text", dict(data))
        return out

    def validity_settings(self, s, ctx) -> Tuple[Dict[str, Any], List[Issue]]:
        """The settings of the validity helper (one entry per criterion switched on) and the issues of the
        options.  Ranges are read with MDATK's PDB tools, the register with mdatk.ensemble.parse_register,
        the sequence with mdatk.ensemble.parse_reference (or taken from the FARFAR2 step)."""
        from .modelcheck import expand_groups  # the step module registers a step type: not at import time
        out: Dict[str, Any] = {}
        issues: List[Issue] = []
        if s["valid_hands"]:
            spec: Dict[str, Any] = {}
            try:
                counts = _validity.parse_hand_counts(s["hands_counts"])
            except ValueError as exc:
                issues.append(Issue("error", str(exc), ("hands_counts",)))
                counts = {}
            if counts:
                spec["counts"] = counts
            for key, name in (("hands_l", "expect_l"), ("hands_d", "expect_d")):
                if not s[key].strip():
                    continue
                try:
                    text = _validity.whole_chains(expand_groups(s[key], ctx.groups))
                    pdbtools.parse_ranges(text)
                    spec[name] = text
                except KeyError as exc:
                    issues.append(Issue("error", f"no residue group {exc.args[0]!r}", (key,)))
                except (ValueError, pdbtools.PDBError) as exc:
                    issues.append(Issue("error", str(exc), (key,)))
            if not s["hands_counts"].strip() and not s["hands_l"].strip() and not s["hands_d"].strip():
                issues.append(Issue("error", "the hands filter needs the expected counts (e.g. 44 D / 29 L) "
                                             "or the residues expected L or D",
                                    ("hands_counts", "hands_l", "hands_d")))
            out["hands"] = spec
        if s["valid_bases"]:
            ref = None
            try:
                if s["valid_sequence"].strip():
                    ref = ensemble.parse_reference(s["valid_sequence"])
                else:
                    ref = ensemble.reference_from_model(ctx.data("silent").get("rna_model"))
            except ensemble.EnsembleError as exc:
                issues.append(Issue("error", str(exc), ("valid_sequence",)))
            if ref is not None:
                out["bases"] = {"reference": list(zip(ref.chains, ref.numbers, ref.bases))}
            elif not s["valid_sequence"].strip():
                issues.append(Issue("error", "base identity needs the sequence: give it, or connect a "
                                             "FARFAR2 step whose sequence is known",
                                    ("valid_sequence", "input:silent")))
        if s["valid_closure"]:
            spec = {}
            if s["closure_mode"] in ("term", "both"):
                term = s["closure_term"].strip()
                if not re.fullmatch(r"\S+", term):
                    issues.append(Issue("error", "closure by a score term needs one term (a SCORE column "
                                                 "name)", ("closure_term",)))
                spec.update(term=term, max=float(s["closure_max"]))
            if s["closure_mode"] in ("geometry", "both"):
                spec["o3p"] = float(s["closure_o3p"])
            out["closure"] = spec
        if s["valid_pairs"]:
            if not s["register"].strip():
                issues.append(Issue("error", "required pairs need the register's defining pairs, e.g. "
                                             "pres=B:22-B:37, B:23-B:36, B:24-B:35", ("register",)))
            else:
                try:
                    reg = ensemble.parse_register(s["register"])
                    ensemble.check_registers([reg])
                    out["pairs"] = {"name": reg.name, "pairs": [tuple(p) for p in reg.pairs],
                                    "mode": s["pairs_mode"], "cut": ensemble.PAIR_CUT,
                                    "near_n": ensemble.NEAR_N, "near_c1": tuple(ensemble.NEAR_C1),
                                    "wc_atom": dict(ensemble.WC_ATOM)}
                except ensemble.EnsembleError as exc:
                    issues.append(Issue("error", str(exc), ("register",)))
        return out, issues

    @staticmethod
    def quality_limits(s) -> Tuple[List[Tuple[str, str, float]], List[Issue]]:
        """The score limits of the quality filter (mdatk.modelcheck.parse_score_limits) and their issues."""
        if not s["filter"]:
            return [], []
        try:
            limits = modelcheck.parse_score_limits(s["filter_limits"])
        except ValueError as exc:
            return [], [Issue("error", str(exc), ("filter_limits",))]
        if not limits:
            return [], [Issue("error", "the quality filter needs score limits, e.g. score <= -300",
                              ("filter_limits",))]
        return limits, []

    def check(self, s, ctx):
        out: List[Issue] = []
        out += self.quality_limits(s)[1]
        if s["filter"] and s["filter_when"] == "before":
            out.append(Issue("warning", "a score filter before clustering changes the cluster populations by "
                                        "a score that need not track quality: filter on validity before "
                                        "clustering and on quality after it", ("filter_when",)))
        if self.validity_on(s):
            out += self.validity_settings(s, ctx)[1]
            if not s["extract_all"]:
                out.append(Issue("info", "the validity filters judge every decoy as a PDB: every model of "
                                         "every run is extracted", ("extract_all",)))
        n_files = len(ctx.inps("silent"))
        if s["cluster"] and s["n_models"] > s["max_clusters"]:
            out.append(Issue("warning", "more models requested than clusters written", ("n_models",)))
        if not s["cluster"] and n_files > 1:
            out.append(Issue("info", f"models of {n_files} runs are ranked together by total score",
                             ("cluster",)))
        if s["dssr"]:
            out.append(Issue("info", "x3dna-dssr needs its own (free academic) licence and binary",
                             ("dssr",)))
        return out

    @classmethod
    def filters_before(cls, s) -> bool:
        """Whether decoys are left out before clustering (validity, or a score filter chosen 'before')."""
        return bool(cls.validity_on(s) or (s["filter"] and s["filter_when"] == "before"))

    def _runs(self, s, ctx) -> List[List[str]]:
        """[label, silent file, filtered file or ''] per input run (paths relative to the step folder)."""
        silents = ctx.inps("silent")
        labels = _silent_scores.unique_labels(silents)
        before = self.filters_before(s)
        return [[lab, path, ctx.name(f"filtered-{lab}", "out") if before else ""]
                for lab, path in zip(labels, silents)]

    def _scores_helper(self, ctx, runs) -> GeneratedFile:
        return scores_helper(ctx, runs, ctx.name("model-MODELNUM", "pdb").replace("MODELNUM", "{N}"))

    def _validity_helper(self, ctx, runs, settings) -> GeneratedFile:
        """MDATK's PDB tools, model checks and silent-file scores, then the validity criteria: one standalone
        script (standard library only) whose modules find each other in the script itself."""
        doc = (f"Validity of the FARFAR2 decoys of project {ctx.project}, step {ctx.step:02d}, before they "
               f"are clustered - written by {PROJECT_ABBREV} {__version__}.\n\n"
               "    python3 THIS_SCRIPT judge VALIDITY.tsv [--report REPORT.txt] [--extracted models]\n\n"
               "judges every decoy of every run (models/<run>/<tag>.pdb) on hands, base identity, closure "
               "and required pairs, in this order. Needs only Python 3.")
        bodies: List[str] = []
        for title, module in (("MDATK's PDB tools (mdatk.pdbtools)", pdbtools),
                              ("MDATK's model checks (mdatk.modelcheck)", modelcheck),
                              ("MDATK's silent-file scores (silent_scores)", _silent_scores)):
            bodies += [f"# --- {title} ---", _module_body(module), "", ""]
        bodies.append("# --- validity of the decoys (decoy_validity) ---")
        constants = {"RUNS": [tuple(r[:2]) for r in runs], "SETTINGS": settings, "EXTRACTED": MODELS_DIR}
        return GeneratedFile(ctx.name("validity", "py"), embed(
            _validity, constants, "validity_main(sys.argv[1:], RUNS, SETTINGS, EXTRACTED)", doc,
            imports=bodies), "helper", "validity of every decoy before clustering: hands, bases, closure, "
            "required pairs")

    def _extract_all(self, s, runs, db: str, helper: str) -> List[str]:
        return extract_every_model(s["extract_exe"], s["extract_parallel"], runs, db, helper)

    def generate(self, s, ctx):
        n = ctx.name
        script = n(self.tag, "sh")
        silents = ctx.inps("silent")
        db = f" -database {s['database'].strip()}" if s["database"].strip() else ""
        table = self.writes_table(s)
        runs = self._runs(s, ctx) if table else []
        helper = n("scores", "py")
        sh = script_header("FARFAR2 models: cluster, extract, annotate", ctx, s, script)
        sh += ["# rank_models FILE: prints 'score tag file' for every model in a silent file",
               "rank_models() {", f"  {RANK_AWK}", "}", ""]
        limits = self.quality_limits(s)[0]
        cut = "".join(f" --limit {shell_quote(t)} {shell_quote(op)} {x!r}" for t, op, x in limits)
        cut_text = _silent_scores.limits_text(limits)
        valid = self.validity_on(s)
        settings = self.validity_settings(s, ctx)[0] if valid else {}
        validity_tsv = n("validity", "tsv")
        if s["extract_all"] or valid:  # the validity filters judge every decoy as a PDB
            sh += self._extract_all(s, runs, db, helper)
        if valid:
            sh += [f"# Validity filters (before clustering), in this order: {', '.join(valid)}. Every decoy",
                   "# is judged; only the valid ones are clustered (the populations of the valid set)",
                   f"python3 {n('validity', 'py')} judge {validity_tsv} --report {n('validity', 'txt')}", ""]
        if self.filters_before(s):
            score_before = s["filter"] and s["filter_when"] == "before"
            what = (["the valid decoys"] if valid else []) + (
                [f"the decoys within {cut_text} (a score cut before clustering changes the cluster "
                 "populations)"] if score_before else [])
            sh += [f"# Cluster (or rank) only {' and '.join(what)}: one filtered silent file per run",
                   f"python3 {helper} keep" + (f" --valid {validity_tsv}" if valid else "")
                   + (cut if score_before else ""), ""]
            silents = [f for _l, _p, f in runs]
        ranking = n("ranking", "txt")
        if s["cluster"]:
            clustered = n("cluster", "out")
            flags = [f"-cluster:radius {fortran_float(s['radius'])}", f"-nstruct {s['max_clusters']}"]
            if s["score_diff_cut"] is not None:
                flags.append(f"-cluster:score_diff_cut {s['score_diff_cut']:g}")
            flags += extra_flags(s["cluster_extra"])
            sh += ["# Cluster all models (best cluster first)",
                   f"rm -f {clustered}",
                   f"{s['cluster_exe'].strip()}{db} -in:file:silent {' '.join(silents)} "
                   f"-out:file:silent {clustered} {' '.join(flags)} > {n('cluster', 'log')} 2>&1",
                   require_file(clustered, "rna_cluster"),
                   f"rank_models {clustered} > {ranking}", ""]
        else:
            sh += ["# Rank every model of every run by total score",
                   f"for f in {' '.join(silents)}; do rank_models \"$f\"; done | sort -g -k1,1 > {ranking}", ""]
        if s["filter"] and s["filter_when"] == "after":
            sh += [f"# Quality filter (after clustering): skip the ranked models outside {cut_text} (terms "
                   "by column name)", f"python3 {helper} filter-ranking {ranking}{cut}", ""]
        model = n("model-MODELNUM", "pdb").replace("MODELNUM", "${N}")
        sh += [f"# Extract the best {s['n_models']} models as numbered PDB files"]
        if table:  # the decoy table reads this list: no lines of an earlier run
            sh += [f"rm -f {n('models', 'txt')}"]
        sh += ["K=0",
               "while read -r SCORE TAG FILE; do",
               "  K=$((K + 1))",
               f"  [ $K -gt {s['n_models']} ] && break",
               '  N=$(printf "%02d" "$K")',
               '  rm -f "$TAG.pdb"',
               f"  {s['extract_exe'].strip()}{db} -in:file:silent \"$FILE\" -in:file:tags \"$TAG\" > /dev/null "
               "|| true",
               f'  if [ -s "$TAG.pdb" ]; then mv "$TAG.pdb" "{model}"; '
               f'echo "$N $SCORE $TAG $FILE" >> {n("models", "txt")}; '
               f'else echo "could not extract $TAG from $FILE" >&2; fi',
               f"done < {ranking}", ""]
        if s["dssr"]:
            exe = s["dssr_exe"].strip() or "x3dna-dssr"
            pattern = n("model-MODELNUM", "pdb").replace("MODELNUM", "[0-9][0-9]")
            sh += ["# Annotate every model with DSSR",
                   f"for f in {pattern}; do",
                   '  [ -e "$f" ] || continue',
                   f'  {exe} -i="$f" -o="${{f%.pdb}}.dssr"' +
                   (f'\n  {exe} -i="$f" --json -o="${{f%.pdb}}.json"' if s["dssr_json"] else ""),
                   "done",
                   f"{exe} --cleanup > /dev/null 2>&1 || true", ""]
        files = []
        if table:
            terms = ["score"] + [t for t, _op, _x in limits] + [settings.get("closure", {}).get("term") or ""]
            sh += ["# The decoy table: run, tag, pdb and every SCORE term of every model, by column name"
                   + (", and the validity columns" if valid else ""),
                   f"python3 {helper} table {n('scores', 'tsv')}"
                   + (f" --extracted {MODELS_DIR}" if s["extract_all"] or valid else "")
                   + f" --models {n('models', 'txt')}"
                   + (f" --validity {validity_tsv}" if valid else "")
                   + "".join(f" --term {shell_quote(t)}" for t in dict.fromkeys(terms) if t),
                   require_file(n("scores", "tsv"), "the decoy table"), ""]
            files.append(self._scores_helper(ctx, runs))
            if valid:
                files.append(self._validity_helper(ctx, runs, settings))
        sh += [require_file(n("model-01", "pdb"), "the extraction"),
               f'echo "Best model: {n("model-01", "pdb")} (ranking in {n("models", "txt")})"', ""]
        return [GeneratedFile(script, "\n".join(sh), "run", "clusters and extracts the models")] + files
