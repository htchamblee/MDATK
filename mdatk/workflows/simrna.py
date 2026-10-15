"""SimRNA generator: config file, sequence / secondary-structure / restraint
files, parallel replica-exchange runs and the clustering / all-atom
reconstruction that follows (clustering, SimRNA_trafl2pdbs)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import rna
from .base import (PortValue, Tool, env_section, extra_lines, fortran_float, require_file,
                   script_header)
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

PAIR_TYPES = ("()", "[]", "{}", "<>")


def ss_lines(model: rna.RNAModel) -> List[str]:
    """SimRNA secondary structure: one '()' line per bracket type (pseudoknots on
    extra lines), chains separated by spaces."""
    if not model.pairs:
        return []
    kinds = {}
    for i, j in model.pairs:
        kinds.setdefault(model.dotbracket[i - 1], []).append((i, j))
    lines = []
    for opener in "([{<":
        pairs = kinds.get(opener)
        if not pairs:
            continue
        chars = ["."] * model.length
        for i, j in pairs:
            chars[i - 1], chars[j - 1] = "(", ")"
        lines.append(model.split("".join(chars)))
    return lines


@register
class SimRNATool(Tool):
    name = "simrna"
    label = "SimRNA · folding"
    category = "Structure prediction"
    tag = "simrna"
    description = ("SimRNA coarse-grained folding with secondary-structure and distance restraints, "
                   "replica exchange in parallel runs, then clustering and all-atom models.")
    inputs = [Port("structure", "Start PDB (-p)", ("pdb",), optional=True,
                   tooltip="Optional: start from this PDB instead of an extended chain."),
              Port("secstruct", "Secondary structure", ("secstruct",), optional=True,
                   tooltip="Optional: a dot-bracket file / FARFAR2 step; otherwise type it below.")]
    outputs = [Port("best", "Best model (PDB)", ("pdb",)),
               Port("models", "Models (PDB)", ("pdb",), multiple=True),
               Port("trajectory", "SimRNA trajectories", ("trafl",), multiple=True)]
    sections = [
        Section("sequence", "Sequence & restraints", [
            Option("sequence", "Sequence", "multiline", "", "Chains separated by spaces / new lines; "
                   "ignored when a secondary-structure step is connected.", placeholder="GGGCAC GUGCCC"),
            Option("dotbracket", "Secondary structure", "multiline", "", "Dot-bracket; pseudoknot brackets "
                   "[] {} <> go to extra restraint lines.", placeholder="((((((  ))))))"),
            Option("use_ss", "Use the secondary structure as restraints (-S)", "bool", True, ""),
            Option("restraints", "Distance restraints", "multiline", "",
                   "SimRNA restraint lines, e.g. 'SLOPE A/20/P A/41/C4' 0 8 1' or 'WELL A/16/MB A/39/MB "
                   "5.0 10.0 5.0' (atoms P, C4', N1/N9, C2, C4/C6, MB).",
                   placeholder="SLOPE A/5/P A/30/P 0.0 8.0 1.0"),
        ], tooltip="What to fold and which restraints to apply."),
        Section("config", "Simulation (config.dat)", [
            Option("iterations", "NUMBER_OF_ITERATIONS", "int", 16_000_000, "Monte Carlo iterations.",
                   minimum=1000, maximum=10_000_000_000),
            Option("write_every", "TRA_WRITE_IN_EVERY_N_ITERATIONS", "int", 16000, "Trajectory frame "
                   "interval.", minimum=1, maximum=1_000_000_000),
            Option("init_temp", "INIT_TEMP", "float", 1.35, "Start of the simulated-annealing temperature.",
                   minimum=0.01, maximum=10.0, step=0.05, digits=2),
            Option("final_temp", "FINAL_TEMP", "float", 0.90, "End temperature (equal to INIT_TEMP: constant "
                   "temperature).", minimum=0.01, maximum=10.0, step=0.05, digits=2),
            Option("bonds_weight", "BONDS_WEIGHT", "float", 1.0, "", minimum=0.0, maximum=100.0, step=0.1,
                   digits=2),
            Option("angles_weight", "ANGLES_WEIGHT", "float", 1.0, "", minimum=0.0, maximum=100.0, step=0.1,
                   digits=2),
            Option("tors_weight", "TORS_ANGLES_WEIGHT", "float", 0.0, "", minimum=0.0, maximum=100.0,
                   step=0.1, digits=2),
            Option("eta_theta_weight", "ETA_THETA_WEIGHT", "float", 0.40, "", minimum=0.0, maximum=100.0,
                   step=0.05, digits=2),
            Option("ss_weight", "SECOND_STRC_RESTRAINTS_WEIGHT", "float", None,
                   "Weight of the secondary-structure restraints (empty: SimRNA default 1.0).",
                   optional=True),
            Option("sphere_radius", "LIMITING_SPHERE_RADIUS", "float", None,
                   "Keeps several chains together (Å; empty: not set).", optional=True),
            Option("sphere_weight", "LIMITING_SPHERE_WEIGHT", "float", None, "", optional=True),
            Option("extra_config", "Extra config lines", "multiline", "", "KEY value lines added to the "
                   "config file.", placeholder="FRACTION_OF_FRAGMENT_MOVES 0.01"),
        ], tooltip="Keys of SimRNA's config.dat (the defaults are the distributed example values)."),
        Section("runs", "Runs", [
            Option("runs", "Independent runs", "int", 8, "Runs with different random seeds.", minimum=1,
                   maximum=10000),
            Option("parallel", "Parallel runs", "int", 4, "", minimum=1, maximum=1024),
            Option("replicas", "Replicas (-E)", "int", 10, "Replica-exchange Monte Carlo replicas per run "
                   "(1: plain simulated annealing).", minimum=1, maximum=1000),
            Option("seed_base", "Seed base", "int", 1000, "Run i uses seed base + i (-R).", minimum=0,
                   maximum=2**31 - 100000),
            Option("nice", "Run with nice", "bool", True, ""),
            Option("exe", "SimRNA executable", "text", "SimRNA", ""),
            Option("simrna_dir", "SimRNA folder", "text", "$SIMRNA",
                   "Installation folder containing data/ (linked into every run folder); a path or a shell "
                   "variable."),
        ]),
        Section("cluster", "Clustering", [
            Option("cluster", "Cluster and rebuild all-atom models", "bool", True,
                   "clustering on the lowest-energy frames of all runs, then SimRNA_trafl2pdbs ... AA."),
            Option("fraction", "Fraction of frames", "float", 0.01, "Lowest-energy fraction clustered.",
                   minimum=0.0001, maximum=1.0, step=0.005, digits=4, requires={"cluster": True}),
            Option("threshold", "RMSD threshold", "float", None,
                   "Å; empty: 0.1 × number of nucleotides.", optional=True, requires={"cluster": True}),
            Option("n_models", "Models to rebuild", "int", 5, "Cluster representatives rebuilt as all-atom "
                   "PDB files.", minimum=1, maximum=100, requires={"cluster": True}),
            Option("cluster_exe", "clustering executable", "text", "clustering", "",
                   requires={"cluster": True}),
            Option("trafl2pdbs_exe", "SimRNA_trafl2pdbs executable", "text", "SimRNA_trafl2pdbs", "",
                   requires={"cluster": True}),
        ]),
        env_section("export SIMRNA=$HOME/SimRNA_64bitIntel_Linux"),
    ]

    # ------------------------------------------------------------ model
    def model(self, s, ctx) -> Tuple[Optional[rna.RNAModel], List[Issue]]:
        from .amber import secondary_structure
        model, _src, issues = secondary_structure(s, ctx)
        if model is None and not issues and s["sequence"].strip():
            try:
                model = rna.build_model(s["sequence"], "")
            except ValueError as exc:
                issues.append(Issue("error", str(exc), ("sequence",)))
        return model, issues

    def threshold(self, s, model) -> float:
        if s["threshold"] is not None:
            return float(s["threshold"])
        return round(0.1 * (model.length if model else 50), 2)

    def describe(self, s):
        return f"{s['runs']} runs × {s['replicas']} replicas"

    def output_values(self, s, ctx):
        out = {"trajectory": PortValue([ctx.own("simrna-all", "trafl")], "trafl")}
        if s["cluster"]:
            models = [ctx.own(f"model-{k:02d}", "pdb") for k in range(1, s["n_models"] + 1)]
            out["best"] = PortValue(models[:1], "pdb")
            out["models"] = PortValue(models, "pdb")
        return out

    def output_ports(self, s):
        ports = [Port("trajectory", "SimRNA trajectories", ("trafl",), multiple=True)]
        if s.get("cluster"):
            ports = [Port("best", "Best model (PDB)", ("pdb",)),
                     Port("models", "Models (PDB)", ("pdb",), multiple=True)] + ports
        return ports

    def check(self, s, ctx):
        model, out = self.model(s, ctx)
        if model is None and not ctx.has("structure"):
            if not out:
                out.append(Issue("error", "enter the sequence (or connect a starting PDB)", ("sequence",)))
            return out
        if model is not None and model.pairs and not s["use_ss"]:
            out.append(Issue("info", "the secondary structure is entered but not used as restraints",
                             ("use_ss",)))
        if model is not None and not model.pairs and s["use_ss"] and not ctx.has("structure"):
            out.append(Issue("warning", "no secondary structure: folding without restraints rarely "
                                        "converges for more than ~30 nucleotides", ("dotbracket",)))
        if model is not None:
            for i, j in model.pairs:
                if model.pair_type(i, j) not in rna.CANONICAL + rna.WOBBLE:
                    out.append(Issue("warning", f"residues {i}-{j} ({model.pair_type(i, j)}) are paired but "
                                                "not complementary", ("dotbracket",)))
                    break
            if len(model.chains) > 1 and s["sphere_radius"] is None:
                out.append(Issue("info", "several chains: LIMITING_SPHERE_RADIUS keeps them from drifting "
                                         "apart", ("sphere_radius",)))
        if (s["sphere_radius"] is None) != (s["sphere_weight"] is None):
            out.append(Issue("warning", "set both LIMITING_SPHERE_RADIUS and LIMITING_SPHERE_WEIGHT",
                             ("sphere_radius", "sphere_weight")))
        if s["write_every"] > s["iterations"]:
            out.append(Issue("error", "the trajectory interval exceeds the number of iterations",
                             ("write_every",)))
        frames = s["iterations"] // max(1, s["write_every"])
        if s["cluster"] and frames * s["runs"] * s["replicas"] * s["fraction"] < 20:
            out.append(Issue("warning", "fewer than 20 frames would be clustered; raise the fraction or the "
                                        "number of frames", ("fraction",)))
        if s["init_temp"] < s["final_temp"]:
            out.append(Issue("info", "the temperature rises during the run (INIT_TEMP < FINAL_TEMP)",
                             ("init_temp", "final_temp")))
        for line in extra_lines(s["restraints"]):
            if line.split()[0].upper() not in ("SLOPE", "WELL", "DISTANCE"):
                out.append(Issue("warning", f"restraint line not understood: {line.strip()}",
                                 ("restraints",)))
                break
        if not s["simrna_dir"].strip():
            out.append(Issue("error", "give the SimRNA folder (its data/ directory is needed)",
                             ("simrna_dir",)))
        return out

    # ------------------------------------------------------------ files
    def generate(self, s, ctx):
        n = ctx.name
        files: List[GeneratedFile] = []
        model, _ = self.model(s, ctx)
        cfg = [f"NUMBER_OF_ITERATIONS {s['iterations']}",
               f"TRA_WRITE_IN_EVERY_N_ITERATIONS {s['write_every']}",
               f"INIT_TEMP {fortran_float(s['init_temp'])}",
               f"FINAL_TEMP {fortran_float(s['final_temp'])}",
               "",
               f"BONDS_WEIGHT {fortran_float(s['bonds_weight'])}",
               f"ANGLES_WEIGHT {fortran_float(s['angles_weight'])}",
               f"TORS_ANGLES_WEIGHT {fortran_float(s['tors_weight'])}",
               f"ETA_THETA_WEIGHT {fortran_float(s['eta_theta_weight'])}"]
        if s["ss_weight"] is not None:
            cfg.append(f"SECOND_STRC_RESTRAINTS_WEIGHT {fortran_float(s['ss_weight'])}")
        if s["sphere_radius"] is not None:
            cfg.append(f"LIMITING_SPHERE_RADIUS {fortran_float(s['sphere_radius'])}")
        if s["sphere_weight"] is not None:
            cfg.append(f"LIMITING_SPHERE_WEIGHT {fortran_float(s['sphere_weight'])}")
        cfg += [ln.strip() for ln in extra_lines(s["extra_config"])]
        config = n(self.tag, "config")
        files.append(GeneratedFile(config, "\n".join(cfg) + "\n", "input", "SimRNA config.dat"))
        args = [f"-c ../../{config}"]
        if ctx.has("structure"):
            args.append(f"-p {_up(ctx.inp('structure'))}")
        elif model is not None:
            seq = n(self.tag, "seq")
            files.append(GeneratedFile(seq, model.split(model.sequence.lower()) + "\n", "input",
                                       "sequence (chains separated by spaces)"))
            args.append(f"-s ../../{seq}")
        if model is not None and s["use_ss"]:
            lines = ss_lines(model)
            if lines:
                ss = n(self.tag, "ss")
                files.append(GeneratedFile(ss, "\n".join(lines) + "\n", "input",
                                           "secondary-structure restraints"))
                args.append(f"-S ../../{ss}")
        if extra_lines(s["restraints"]):
            rfile = n(self.tag, "restraints")
            files.append(GeneratedFile(rfile, "\n".join(ln.strip() for ln in extra_lines(s["restraints"])) +
                                       "\n", "input", "distance restraints"))
            args.append(f"-r ../../{rfile}")
        if s["replicas"] > 1:
            args.append(f"-E {s['replicas']}")
        out_base = n(f"{self.tag}-rRUNNUM").replace("RUNNUM", "${N}")
        script = n(self.tag, "sh")
        sh = script_header(f"SimRNA: {s['runs']} runs × {s['replicas']} replicas, {s['parallel']} at a time",
                           ctx, s, script, ["Each run works in runs/rNN with a link to SimRNA's data folder."])
        sh = [ln for ln in sh if ln != "set -eo pipefail"]
        sdir = s["simrna_dir"].strip()
        if sdir == "$SIMRNA":
            sh.append('SIMRNA_DIR="${SIMRNA:?set SIMRNA to the SimRNA installation folder (it contains data/)}"')
        else:
            sh.append(f'SIMRNA_DIR="{sdir}"')
        nice = "nice " if s["nice"] else ""
        sh += ['[ -d "$SIMRNA_DIR/data" ] || { echo "ERROR: $SIMRNA_DIR/data not found" >&2; exit 1; }', "",
               "run_one() {", "  I=$1", '  N=$(printf "%02d" "$I")',
               '  mkdir -p "runs/r$N" && cd "runs/r$N" || return 0',
               '  ln -sfn "$SIMRNA_DIR/data" data',
               f"  {nice}{s['exe'].strip() or 'SimRNA'} {' '.join(args)} -R $(({s['seed_base']} + I)) "
               f"-o {out_base} \\",
               f"    > ../../{out_base}.log 2>&1 || echo \"run $N failed - see {out_base}.log\" >&2",
               "}", "export -f run_one", "export SIMRNA_DIR", "",
               f"seq 1 {s['runs']} | xargs -P {s['parallel']} -I{{}} bash -c 'run_one {{}}'", ""]
        files.append(GeneratedFile(script, "\n".join(sh), "run", "runs SimRNA"))
        if s["cluster"]:
            files.append(self._cluster_script(s, ctx, model))
        return files

    def _cluster_script(self, s, ctx, model) -> GeneratedFile:
        n = ctx.name
        script = n("simrna-cluster", "sh")
        all_trafl = n("simrna-all", "trafl")
        thr = self.threshold(s, model)
        sh = script_header("SimRNA: cluster the lowest-energy frames and rebuild all-atom models", ctx, s,
                           script)
        sdir = s["simrna_dir"].strip()
        sh.append('SIMRNA_DIR="${SIMRNA:?set SIMRNA to the SimRNA installation folder}"' if sdir == "$SIMRNA"
                  else f'SIMRNA_DIR="{sdir}"')
        sh += ['ln -sfn "$SIMRNA_DIR/data" data', "",
               "# all replica trajectories of all runs",
               f"cat runs/r*/*.trafl > {all_trafl}", require_file(all_trafl, "SimRNA"),
               "REF=$(ls runs/r*/*-000001.pdb 2>/dev/null | head -n 1)",
               '[ -n "$REF" ] || { echo "ERROR: no first-frame PDB found in runs/" >&2; exit 1; }', "",
               f"# cluster the lowest {s['fraction'] * 100:g} % of frames at {thr:g} Å",
               f"{s['cluster_exe'].strip() or 'clustering'} {all_trafl} {s['fraction']:g} {thr:g} "
               f"> {n('simrna-cluster', 'log')} 2>&1", "",
               f"# rebuild the first {s['n_models']} cluster representatives as all-atom models",
               "K=0",
               f"for C in {all_trafl[:-6]}_thrs*_clust[0-9][0-9].trafl; do",
               '  [ -e "$C" ] || continue',
               "  K=$((K + 1))",
               f"  [ $K -gt {s['n_models']} ] && break",
               '  N=$(printf "%02d" "$K")',
               f'  {s["trafl2pdbs_exe"].strip() or "SimRNA_trafl2pdbs"} "$REF" "$C" 1 AA > /dev/null',
               '  AA="${C%.trafl}-000001_AA.pdb"',
               f'  [ -s "$AA" ] && mv "$AA" "{n("model-MODELNUM", "pdb").replace("MODELNUM", "${N}")}"',
               "done",
               require_file(n("model-01", "pdb"), "SimRNA_trafl2pdbs"), ""]
        return GeneratedFile(script, "\n".join(sh), "run", "clusters and rebuilds models")


def _up(path: str) -> str:
    """A step-relative path as seen from runs/rNN."""
    if not path or path.startswith("/"):
        return path
    return "../../" + path
