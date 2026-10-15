"""Rosetta FARFAR2 (rna_denovo) generator.

Writes the Rosetta FASTA / secstruct files from a typed sequence and
dot-bracket, a Rosetta flags file and a script that runs independent seeds in
parallel (``xargs -P``), each in its own folder, writing one silent file per
run.  Defaults follow a working cryo-EM-guided protocol (50 000 cycles,
-fixed_stems, -heat, density weight 20, constant seeds 3001...).
"""

from __future__ import annotations

import posixpath
from typing import List, Optional, Tuple

from .. import PROJECT_ABBREV, __version__
from . import rna
from .base import (PortValue, StepContext, Tool, env_section, extra_flags, fortran_float,
                   require_file, script_header)
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

RUN_DIR = "runs"


def _flag_bool(value: bool) -> str:
    return "true" if value else "false"


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
        env_section("export PATH=$HOME/rosetta/main/source/bin:$PATH"),
    ]

    def output_ports(self, s):
        return [Port("silent", "Silent files", ("silent",), multiple=True),
                Port("fasta", "FASTA", ("fasta",)),
                Port("secstruct", "Secondary structure", ("secstruct",))]

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
        out = {"silent": PortValue([ctx.own(f"{self.tag}-r{k:02d}", "out") for k in range(1, s["runs"] + 1)],
                                   "silent", dict(data))}
        if fasta:
            out["fasta"] = PortValue([ctx.from_own(fasta) if s["make_inputs"] else fasta], "fasta", dict(data))
        if ss:
            out["secstruct"] = PortValue([ctx.from_own(ss) if s["make_inputs"] else ss], "secstruct",
                                         dict(data))
        return out

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
        files.append(GeneratedFile(script, "\n".join(sh), "run", "runs rna_denovo"))
        return files


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
               Port("models", "Ranked models (PDB)", ("pdb",), multiple=True)]
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
        Section("extract", "Extraction", [
            Option("n_models", "Models to extract", "int", 10, "Cluster centres (or best models) written as "
                   "PDB files <project>_<step>_model-01_<time>.pdb ...", minimum=1, maximum=10000),
            Option("extract_exe", "extract_pdbs executable", "text", "extract_pdbs", ""),
            Option("database", "Rosetta database", "text", "", "-database path (empty: found by the "
                   "executables)."),
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

    def describe(self, s):
        return (f"rna_cluster {s['radius']:g} Å, " if s["cluster"] else "by score, ") + f"top {s['n_models']}"

    def output_values(self, s, ctx):
        models = [ctx.own(f"model-{k:02d}", "pdb") for k in range(1, s["n_models"] + 1)]
        data = dict(ctx.data("silent"))
        return {"best": PortValue(models[:1], "pdb", data), "models": PortValue(models, "pdb", data)}

    def check(self, s, ctx):
        out: List[Issue] = []
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

    def generate(self, s, ctx):
        n = ctx.name
        script = n(self.tag, "sh")
        silents = ctx.inps("silent")
        db = f" -database {s['database'].strip()}" if s["database"].strip() else ""
        sh = script_header("FARFAR2 models: cluster, extract, annotate", ctx, s, script)
        sh += ["# rank_models FILE: prints 'score tag file' for every model in a silent file",
               "rank_models() {", f"  {RANK_AWK}", "}", ""]
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
        model = n("model-MODELNUM", "pdb").replace("MODELNUM", "${N}")
        sh += [f"# Extract the best {s['n_models']} models as numbered PDB files",
               "K=0",
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
        sh += [require_file(n("model-01", "pdb"), "the extraction"),
               f'echo "Best model: {n("model-01", "pdb")} (ranking in {n("models", "txt")})"', ""]
        return [GeneratedFile(script, "\n".join(sh), "run", "clusters and extracts the models")]
