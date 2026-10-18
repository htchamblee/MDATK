"""RNAMake generator: design RNA segments that connect two base pairs of a
scaffold - motif-graph search (structure building), sequence optimisation and
optional thermal-fluctuation scoring - with design_rna_scaffold (current
RNAMake, --options) or the legacy design_rna (-options)."""

from __future__ import annotations

import re
from typing import List

from .base import PortValue, Tool, env_section, extra_flags, fortran_float, require_file, script_header
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

_NEW = {"app": "design_rna_scaffold"}
BP_RE = re.compile(r"^[A-Za-z0-9]-?\d+[A-Za-z]?-[A-Za-z0-9]-?\d+[A-Za-z]?$")


@register
class RNAMakeTool(Tool):
    name = "rnamake"
    label = "RNAMake · design"
    category = "Structure prediction"
    tag = "rnamake"
    description = ("RNAMake design: builds motif / helix paths between two base pairs of a scaffold, "
                   "optimises their sequences and scores designs (design_rna_scaffold or design_rna).")
    inputs = [Port("structure", "Scaffold (PDB)", ("pdb",),
                   tooltip="Structure containing the two Watson-Crick base pairs to connect.")]
    outputs = [Port("designs", "Design models (PDB)", ("pdb",), multiple=True),
               Port("best", "First design (PDB)", ("pdb",))]
    sections = [
        Section("scaffold", "Scaffold", [
            Option("app", "Application", "radio", "design_rna_scaffold",
                   "design_rna_scaffold is the current RNAMake app (double-dash options); design_rna is "
                   "the 2016-2020 version (single-dash options).",
                   choices=[("design_rna_scaffold", "design_rna_scaffold"), ("design_rna", "design_rna (legacy)")]),
            Option("start_bp", "Start base pair", "text", "", "Base pair where the design starts, "
                   "<chain><residue>-<chain><residue>, e.g. A222-A251.", placeholder="A222-A251"),
            Option("end_bp", "End base pair", "text", "", "Base pair the design must reach.",
                   placeholder="A149-A154"),
            Option("designs", "Designs", "int", 10, "Number of designs.", minimum=1, maximum=100000),
            Option("extra_pdbs", "Extra motif PDBs", "text", "", "Comma-separated PDB files of extra motifs "
                   "(--extra_pdbs).", requires=_NEW),
        ], tooltip="What to connect."),
        Section("search", "Structure building (search)", [
            Option("search_type", "Search", "radio", "path_finding",
                   "path_finding (default), Monte Carlo or exhaustive search through motif / helix libraries.",
                   choices=[("path_finding", "Path finding"), ("mc", "Monte Carlo"), ("exhaustive", "Exhaustive")]),
            Option("search_cutoff", "Score cutoff (--search_cutoff)", "float", 7.5, "Accept solutions scoring "
                   "below this.", minimum=0.0, maximum=1000.0, step=0.5, digits=2, requires=_NEW),
            Option("search_max_size", "Maximum size (--search_max_size)", "int", 999999, "", minimum=1,
                   maximum=100_000_000),
            Option("search_max_motifs", "Maximum motifs", "int", 999, "--search_max_motifs", minimum=1,
                   maximum=100000, requires=_NEW),
            Option("motif_path", "Motif path", "text", "", "Comma-separated library path, e.g. "
                   "flex_helices,twoway,flex_helices (empty: RNAMake default).",
                   placeholder="flex_helices,twoway,flex_helices", requires=_NEW),
            Option("min_helix", "Minimum helix length", "int", 4, "", minimum=1, maximum=1000, requires=_NEW),
            Option("max_helix", "Maximum helix length", "int", 99, "", minimum=1, maximum=1000, requires=_NEW),
            Option("no_sterics", "Ignore clashes (--no_sterics)", "bool", False, ""),
            Option("no_bp_checks", "Skip base-pair checks", "bool", False, "--no_basepair_checks",
                   requires=_NEW),
        ], tooltip="How the motif graph (3D path) between the base pairs is built."),
        Section("sequence", "Sequence optimisation", [
            Option("skip_seq_opt", "Skip sequence optimisation", "bool", False,
                   "Keep the library sequences (--skip_sequence_optimization / -only_ideal)."),
            Option("seqs_per_design", "Sequences per design", "int", 1, "", minimum=1, maximum=1000),
            Option("seq_opt_steps", "Optimisation steps", "int", 10000, "--seq_opt_steps", minimum=1,
                   maximum=100_000_000, requires={"app": "design_rna_scaffold", "skip_seq_opt": False}),
            Option("seq_opt_cutoff", "Optimisation cutoff", "float", 7.0, "--seq_opt_cutoff", minimum=0.0,
                   maximum=1000.0, step=0.5, digits=2,
                   requires={"app": "design_rna_scaffold", "skip_seq_opt": False}),
            Option("thermo_fluc", "Thermal-fluctuation scoring", "bool", False,
                   "--thermo_fluc: sample helix flexibility to score how often the design closes.",
                   requires=_NEW),
            Option("thermo_fluc_steps", "Fluctuation steps", "int", 1_000_000, "", minimum=1,
                   maximum=1_000_000_000, requires={"app": "design_rna_scaffold", "thermo_fluc": True}),
        ], tooltip="Sequence optimisation of the designed segment."),
        Section("output", "Output", [
            Option("dump_pdbs", "Write design PDBs", "bool", True, "--dump_pdbs / -pdbs"),
            Option("dump_scaffold", "Write scaffold PDBs", "bool", False, "--dump_scaffold_pdbs",
                   requires=_NEW),
            Option("rnamake_dir", "RNAMAKE folder", "text", "$RNAMAKE",
                   "RNAMake installation (exported as RNAMAKE, which the programs need)."),
            Option("exe", "Executable", "text", "", "Empty: the application name; or a full path."),
            Option("extra", "Extra flags", "multiline", "", "Additional command-line flags."),
        ]),
        env_section("export RNAMAKE=$HOME/RNAMake"),
    ]

    def resources(self, s):
        return ["rnamake"]

    def describe(self, s):
        return f"{s['app']}: {s['start_bp'] or '?'} → {s['end_bp'] or '?'}, {s['designs']} designs"

    def output_values(self, s, ctx):
        if not s["dump_pdbs"]:
            return {}
        designs = [ctx.own(f"design-{k:02d}", "pdb") for k in range(1, s["designs"] + 1)]
        return {"designs": PortValue(designs, "pdb"), "best": PortValue(designs[:1], "pdb")}

    def output_ports(self, s):
        return list(self.outputs) if s.get("dump_pdbs") else []

    def check(self, s, ctx):
        out: List[Issue] = []
        for key in ("start_bp", "end_bp"):
            bp = s[key].strip()
            if not bp:
                out.append(Issue("error", f"give the {key.replace('_', ' ')} (e.g. A222-A251)", (key,)))
            elif not BP_RE.match(bp):
                out.append(Issue("error", f"{bp!r} is not a base pair like A222-A251", (key,)))
        if s["start_bp"].strip() and s["start_bp"].strip() == s["end_bp"].strip():
            out.append(Issue("error", "start and end base pairs are the same", ("start_bp", "end_bp")))
        if s["app"] == "design_rna":
            changed = [k for k in ("extra_pdbs", "motif_path") if s[k].strip()]
            changed += [k for k in ("thermo_fluc", "no_bp_checks", "dump_scaffold") if s[k]]
            if s["search_type"] == "exhaustive":
                changed.append("search_type")
            if changed:
                out.append(Issue("warning", "design_rna (legacy) ignores: " + ", ".join(changed),
                                 tuple(changed)))
        if s["min_helix"] > s["max_helix"]:
            out.append(Issue("error", "minimum helix length exceeds the maximum", ("min_helix", "max_helix")))
        if s["search_type"] == "exhaustive" and s["search_max_size"] > 100000:
            out.append(Issue("warning", "an exhaustive search with a large maximum size can run for a very "
                                        "long time", ("search_max_size",)))
        if not s["dump_pdbs"]:
            out.append(Issue("info", "without design PDBs only the score / out files are written; later steps "
                                     "cannot use the designs", ("dump_pdbs",)))
        if not s["rnamake_dir"].strip():
            out.append(Issue("error", "RNAMake needs the RNAMAKE folder", ("rnamake_dir",)))
        return out

    def _command(self, s, ctx) -> List[str]:
        new = s["app"] == "design_rna_scaffold"
        d = "--" if new else "-"
        exe = s["exe"].strip() or s["app"]
        args = [exe, f"{d}pdb {ctx.inp('structure')}", f"{d}start_bp {s['start_bp'].strip()}",
                f"{d}end_bp {s['end_bp'].strip()}", f"{d}designs {s['designs']}",
                f"{d}out_file {ctx.name('design', 'out')}", f"{d}score_file {ctx.name('design', 'scores')}"]
        if new:
            if s["search_type"] != "path_finding":
                args.append(f"--search_type {s['search_type']}")
            args.append(f"--search_cutoff {fortran_float(s['search_cutoff'])}")
            args.append(f"--search_max_size {s['search_max_size']}")
            args.append(f"--search_max_motifs {s['search_max_motifs']}")
            if s["motif_path"].strip():
                args.append(f"--motif_path {s['motif_path'].strip()}")
            args.append(f"--min_helix_length {s['min_helix']} --max_helix_length {s['max_helix']}")
            if s["extra_pdbs"].strip():
                args.append(f"--extra_pdbs {s['extra_pdbs'].strip()}")
            if s["no_sterics"]:
                args.append("--no_sterics")
            if s["no_bp_checks"]:
                args.append("--no_basepair_checks")
            if s["skip_seq_opt"]:
                args.append("--skip_sequence_optimization")
            else:
                args.append(f"--sequences_per_design {s['seqs_per_design']}")
                args.append(f"--seq_opt_steps {s['seq_opt_steps']} --seq_opt_cutoff "
                            f"{fortran_float(s['seq_opt_cutoff'])}")
            if s["thermo_fluc"]:
                args.append(f"--thermo_fluc --thermo_fluc_steps {s['thermo_fluc_steps']}")
            if s["dump_pdbs"]:
                args.append("--dump_pdbs")
            if s["dump_scaffold"]:
                args.append("--dump_scaffold_pdbs")
        else:
            if s["search_type"] == "mc":
                args.append("-mc")
            args.append(f"-search.max_size {s['search_max_size']}")
            if s["no_sterics"]:
                args.append("-no_sterics")
            if s["skip_seq_opt"]:
                args.append("-only_ideal")
            else:
                args.append(f"-seqs_per_design {s['seqs_per_design']}")
            if s["dump_pdbs"]:
                args.append("-pdbs")
        args += extra_flags(s["extra"])
        return args

    def generate(self, s, ctx):
        n = ctx.name
        script = n(self.tag, "sh")
        sh = script_header(f"RNAMake {s['app']}", ctx, s, script)
        rdir = s["rnamake_dir"].strip()
        sh.append('export RNAMAKE="${RNAMAKE:?set RNAMAKE to the RNAMake folder}"' if rdir == "$RNAMAKE"
                  else f'export RNAMAKE="{rdir}"')
        cmd = self._command(s, ctx)
        sh += ["", "# Build motif paths between the base pairs, optimise sequences, score designs",
               " \\\n  ".join(cmd) + f" > {n('design', 'log')} 2>&1",
               require_file(n("design", "scores"), s["app"]), ""]
        if s["dump_pdbs"]:
            model = n("design-MODELNUM", "pdb").replace("MODELNUM", "${N}")
            sh += ["# Number the design PDBs (RNAMake writes design.0.pdb, design.1.pdb ...)",
                   "for f in design.*.pdb; do",
                   '  [ -e "$f" ] || continue',
                   '  K=${f#design.}; K=${K%.pdb}',
                   '  N=$(printf "%02d" $((K + 1)))',
                   f'  mv "$f" "{model}"',
                   "done", ""]
        return [GeneratedFile(script, "\n".join(sh), "run", "runs RNAMake")]
