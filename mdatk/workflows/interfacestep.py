"""Planner step 'decoy-interface': the interface census of an ensemble of models (e.g. FARFAR2 decoys)
between two sides, clusters of the contact fingerprints with medoid representatives, and what holds the
rebuilt segments (:mod:`mdatk.interfaces`).

The step writes a bash script that calls ``python3 -m mdatk decoys census|clusters|support`` (the
step's *Python*; from the Linux AppImage, the AppImage itself), as the structure builder does: the
census needs numpy, which the embedded standard-library helpers may not use.  Sides given as project
residue groups (``group:Aptamer``) are written into the script as residue ranges, so it runs without
the project file.
"""

from __future__ import annotations

import posixpath
from typing import List, Optional

from .. import interfaces as ifc
from .base import PortValue, Tool, env_section, require_file, script_header, shell_quote
from .buildstep import StructureBuilderTool
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

PAIR_CHOICES = [("builtin", "builtin (canonical pairs)"), ("barnaba", "barnaba (Leontis-Westhof)"),
                ("dssr", "DSSR (Leontis-Westhof)"), ("off", "off")]
LEVEL_CHOICES = [("residues", "residues of each side"), ("blocks", "blocks of residues"),
                 ("pairs", "residue pairs")]
REBUILT_CHOICES = [("upstream", "not fixed in the FARFAR2 step"),
                   ("fixed", "not in the fixed residues below"), ("rebuilt", "the residues below")]
CREDIT_CHOICES = [("off", "no credit"), ("contacts", "two base-edge H-bonds (any family)"),
                  ("builtin", "canonical pairs")]
# keys of the models' PortValue.data that may carry the residues FARFAR2 kept fixed (its -input_res)
UPSTREAM_KEYS = ("input_res", "fixed_residues")


@register
class DecoyInterfaceTool(Tool):
    name = "decoy-interface"
    label = "Decoy interfaces (census / clusters / support)"
    category = "Analysis"
    tag = "interface"
    description = ("Interface census of an ensemble of models between two sides (contacts, stacking, base "
                   "pairs, base-phosphate, Mg-bridgeable phosphates), clusters of the contact fingerprints "
                   "with medoid representatives, and what holds the rebuilt segments.")
    inputs = [Port("models", "Models (PDB)", ("pdb",), multiple=True,
                   tooltip="The models to compare, e.g. the decoys of a FARFAR2 models step."),
              Port("scores", "Decoy scores", ("table",), optional=True,
                   tooltip="Table of the decoys' score terms (run, tag, pdb, score, linear_chainbreak ...) "
                           "for the score filters and the score column."),
              Port("references", "Reference models", ("pdb",), optional=True, multiple=True,
                   tooltip="Models of known interfaces: each one's census becomes a reference interface, "
                           "measured with the same cut-offs as the ensemble.")]
    sections = [
        Section("sides", "Interface", [
            Option("side_a", "Side A", "text", "",
                   "Residues of the first molecule: chain-qualified ranges (A:1-16,B:17-44), whole chains "
                   "(A,B), ranges for every chain (1-44) or residue groups of the project (group:Aptamer).",
                   placeholder="A:1-16,B:17-44"),
            Option("side_b", "Side B", "text", "",
                   "Residues of the second molecule, as for side A. Empty: every nucleotide not in side A.",
                   placeholder="C:45-73"),
            Option("models_from", "Models", "radio", "port",
                   "The connected models, or every extracted decoy listed in the pdb column of the "
                   "connected scores table.",
                   choices=[("port", "connected models"), ("scores", "every decoy in the scores table")]),
        ], tooltip="The two sides whose interface is measured; nothing assumes a register or a numbering."),
        Section("census", "Census", [
            Option("contact", "Contact distance", "float", 4.5,
                   "Heavy-atom pairs closer than this count as contacts and make the fingerprint.",
                   minimum=1.0, maximum=15.0, step=0.1, digits=2, unit="Å"),
            Option("hbond", "H-bond distance", "float", 3.5,
                   "Donor-acceptor distance of the base-phosphate and 2'-OH-phosphate contacts.",
                   minimum=2.0, maximum=5.0, step=0.1, digits=2, unit="Å"),
            Option("pairs", "Base pairs", "radio", "builtin",
                   "Engine of the Watson-Crick / G·U / non-canonical pair counts across the interface: "
                   "builtin finds canonical pairs only; barnaba (optional package) and DSSR (x3dna-dssr, "
                   "licensed separately) also classify non-canonical pairs.", choices=PAIR_CHOICES),
            Option("wobble", "Count G·U wobble pairs", "bool", True, "",
                   requires={"pairs": ("builtin", "barnaba", "dssr")}),
            Option("dssr_exe", "x3dna-dssr executable", "text", "x3dna-dssr", "Name on PATH or full path.",
                   requires={"pairs": "dssr"}),
            Option("bph_ch", "C-H donors in base-phosphate contacts", "bool", False,
                   "Also count the base C-H donors of Zirbel's base-phosphate classes (A C2/C8, G C8, "
                   "C/U C5/C6) besides the N-H donors."),
            Option("stack_distance", "Stacking: centroid distance", "float", 4.5,
                   "Ring centroids of stacked bases are closer than this.", minimum=2.0, maximum=8.0,
                   step=0.1, digits=2, unit="Å"),
            Option("stack_angle", "Stacking: normal angle", "float", 30.0,
                   "The ring normals of stacked bases differ by less than this.", minimum=1.0, maximum=90.0,
                   step=1.0, digits=0, unit="°"),
            Option("stack_separation", "Stacking: offset along the normal", "float", 2.5,
                   "At least this much of the centroid offset lies along the mean normal (a coplanar base "
                   "pair has its offset in the base plane).", minimum=0.0, maximum=5.0, step=0.1, digits=2,
                   unit="Å"),
            Option("mg_min", "Mg-bridgeable P-P from", "float", 5.0,
                   "Phosphate pairs across the interface with a P-P distance in this window could hold one "
                   "ion between them (an opportunity, not an observed ion).", minimum=0.0, maximum=20.0,
                   step=0.5, digits=2, unit="Å"),
            Option("mg_max", "… to", "float", 7.0, "", minimum=0.0, maximum=20.0, step=0.5, digits=2,
                   unit="Å"),
        ], tooltip="What is counted across the interface of every model."),
        Section("filters", "Score filters", [
            Option("filters", "Keep models with", "multiline", "linear_chainbreak <= 5",
                   "One condition per line on a term of the connected scores table (name, then <, <=, >, >=, "
                   "== or !=, then a number); every condition must hold. A large linear_chainbreak means a "
                   "severed backbone, whose loops collapse together and inflate the contacts. Ignored "
                   "without a scores table.", placeholder="linear_chainbreak <= 5"),
            Option("drop_unscored", "Drop models without a score row", "bool", False,
                   "Otherwise models the table does not list are kept (and counted in the log)."),
        ]),
        Section("clustering", "Clustering", [
            Option("cluster", "Cluster the fingerprints", "bool", True,
                   "Leader clustering of the contact fingerprints by Jaccard distance (largest first, "
                   "deterministic); each cluster's representative is its medoid."),
            Option("level", "Granularity", "radio", "residues",
                   "Residues: which residues of each side touch (pairing ignored). Blocks: residues binned "
                   "into blocks per chain, so a small register shift within a block does not split a "
                   "cluster (but nearby surfaces merge). Pairs: which residue touches which.",
                   choices=LEVEL_CHOICES, requires={"cluster": True}),
            Option("block", "Block size", "int", 4, "Residues per block.", minimum=2, maximum=50,
                   requires={"cluster": True, "level": "blocks"}),
            Option("cut", "Jaccard distance cut", "float", 0.5, "Fingerprints this close join a cluster.",
                   minimum=0.0, maximum=1.0, step=0.05, digits=2, requires={"cluster": True}),
            Option("min_size", "Recurring from", "int", 5,
                   "Clusters with at least this many members count as recurring (coverage, reference "
                   "shares).", minimum=1, maximum=100000, requires={"cluster": True}),
            Option("n_reps", "Representatives", "int", 5,
                   "Medoids of the largest clusters written as PDB outputs (rep-01 ...).", minimum=1,
                   maximum=100, requires={"cluster": True}),
            Option("broad_limit", "Broad above", "int", 10,
                   "A cluster whose representative touches more side-B residues is marked broad, and the "
                   "reference shares are also given without such clusters (0: off).", minimum=0,
                   maximum=10000, requires={"cluster": True}),
            Option("lead_gap", "Leading reference by", "float", 0.15,
                   "A reference leads when its size-weighted share is this much ahead of the next.",
                   minimum=0.0, maximum=1.0, step=0.05, digits=2, requires={"cluster": True}),
        ], tooltip="Which interfaces recur in the ensemble."),
        Section("references", "Reference interfaces", [
            Option("name", "Name", "text", "reference", "Name of the reference in the clusters table."),
            Option("residues_a", "Side A residues", "text", "",
                   "Residues of side A in the reference interface (ranges or groups, as for the sides).",
                   placeholder="A:8-9,B:25-28"),
            Option("residues_b", "Side B residues", "text", "", "Residues of side B in the reference "
                   "interface.", placeholder="C:58-64"),
        ], repeat=True, item_label="name",
            templates=[("reference", "Reference interface", {"name": "reference"})],
            tooltip="Known interfaces each cluster is compared with, on the residues exclusive to each "
                    "reference (both sides, normalised by the size of each exclusive set). Measure them at "
                    "the contact distance of the census, or connect reference models instead. Used by the "
                    "clustering."),
        Section("support", "Segment support", [
            Option("support", "Check what holds the rebuilt segments", "bool", False,
                   "For every rebuilt residue, its contacts to the partner side, to its own fixed scaffold, "
                   "base pairs (optional) or only its own rebuilt segments, which do not count as support."),
            Option("rebuilt_from", "Rebuilt residues", "radio", "upstream",
                   "Not fixed in the FARFAR2 step: the residues of the sides outside its -input_res (when "
                   "the connected models say which they are). Otherwise the complement of the fixed "
                   "residues, or the rebuilt residues, typed below.", choices=REBUILT_CHOICES,
                   requires={"support": True}),
            Option("fixed", "Fixed residues", "text", "", "E.g. FARFAR2's -input_res.",
                   placeholder="A:1-6 A:13-16 B:17-20 B:39-44 C:45-57 C:64-73",
                   requires={"support": True, "rebuilt_from": "fixed"}),
            Option("rebuilt", "Rebuilt residues", "text", "", "", placeholder="A:7-12,B:21-38,C:58-63",
                   requires={"support": True, "rebuilt_from": "rebuilt"}),
            Option("sep", "Sequence separation", "int", 3,
                   "Contacts to residues of the same chain at most this far in sequence are ignored (the "
                   "backbone holds those).", minimum=0, maximum=50, requires={"support": True}),
            Option("support_cutoff", "Contact distance", "float", 4.5, "", minimum=1.0, maximum=15.0,
                   step=0.1, digits=2, unit="Å", requires={"support": True}),
            Option("run", "Flag runs of", "text", "4",
                   "Flag models with this many consecutive unsupported residues, or 'auto' to put the "
                   "threshold at the largest gap of the ensemble's longest runs (5 or more models).",
                   placeholder="4 or auto", requires={"support": True}),
            Option("count_self", "Count self-contact as support", "bool", False,
                   "The looser definition: any long-range contact counts.", requires={"support": True}),
            Option("pair_credit", "Base pairs inside rebuilt segments", "radio", "off",
                   "Count base pairs between rebuilt residues as support, in a class of their own (e.g. "
                   "stems imposed in the secondary structure).", choices=CREDIT_CHOICES,
                   requires={"support": True}),
        ], tooltip="Whether the rebuilt parts of each model are held by anything."),
        env_section("conda activate mdatk", [
            Option("python", "Python", "text", "python3", "Interpreter that can import MDATK. From the "
                   "standalone AppImage / Windows build the script calls that program itself instead."),
        ]),
    ]

    # ------------------------------------------------------------ ports
    def input_ports(self, s):
        if s.get("models_from") == "scores":
            m = self.inputs[0]
            return [Port(m.name, m.label, m.kinds, optional=True, multiple=True, tooltip=m.tooltip),
                    *self.inputs[1:]]
        return list(self.inputs)

    def output_ports(self, s):
        out = [Port("census", "Interface census", ("table",))]
        if s.get("cluster", True):
            out += [Port("clusters", "Interface clusters", ("table",)),
                    Port("representatives", "Cluster representatives (PDB)", ("pdb",), multiple=True,
                         tooltip="Medoids of the largest clusters, largest first.")]
        if s.get("support"):
            out.append(Port("support", "Segment support", ("table",)))
        return out

    def output_values(self, s, ctx):
        data = dict(ctx.data("models"))
        out = {"census": PortValue([ctx.own("census", "tsv")], "table")}
        if s["cluster"]:
            out["clusters"] = PortValue([ctx.own("clusters", "tsv")], "table")
            reps = [ctx.own(f"rep-{k:02d}", "pdb") for k in range(1, s["n_reps"] + 1)]
            out["representatives"] = PortValue(reps, "pdb", data)
        if s["support"]:
            out["support"] = PortValue([ctx.own("support", "tsv")], "table")
        return out

    def resources(self, s):
        return [s["pairs"]] if s["pairs"] in ("barnaba", "dssr") else []

    def describe(self, s):
        a = s["side_a"].strip() or "?"
        b = s["side_b"].strip() or "the rest"
        parts = [f"{a} | {b}"]
        if s["cluster"]:
            parts.append(f"clusters ({s['level']}, {s['cut']:g})")
        if s["support"]:
            parts.append("support")
        return ", ".join(parts)

    # ------------------------------------------------------------ checks
    def _side(self, text, ctx, key, out: List[Issue]) -> Optional[List[ifc.Range]]:
        try:
            return ifc.parse_side(text, ctx.groups)
        except ifc.InterfaceError as exc:
            out.append(Issue("error", f"{key.replace('_', ' ')}: {exc}", (key,)))
            return None

    def _upstream_fixed(self, ctx) -> Optional[str]:
        data = ctx.data("models")
        for key in UPSTREAM_KEYS:
            if key in data and data[key] is not None:
                value = data[key]
                return value if isinstance(value, str) else " ".join(str(v) for v in value)
        return None

    def _filters(self, s):
        try:
            return ifc.parse_filters(s["filters"]), ""
        except ifc.InterfaceError as exc:
            return [], str(exc)

    def check(self, s, ctx):
        out: List[Issue] = []
        if not s["python"].strip():
            out.append(Issue("error", "give the Python interpreter", ("python",)))
        side_a = side_b = None
        if not s["side_a"].strip():
            out.append(Issue("error", "say which residues make side A (ranges such as A:1-16,B:17-44, or "
                                      "group:NAME)", ("side_a",)))
        else:
            side_a = self._side(s["side_a"], ctx, "side_a", out)
            if side_a == []:
                out.append(Issue("error", "side A selects nothing", ("side_a",)))
        if s["side_b"].strip():
            side_b = self._side(s["side_b"], ctx, "side_b", out)
            if side_a and side_b:
                overlap = ifc.sides_overlap(side_a, side_b)
                if overlap:
                    out.append(Issue("error", f"the sides overlap ({overlap[0]})", ("side_a", "side_b")))
        if s["models_from"] == "scores" and not ctx.has("scores"):
            out.append(Issue("error", "'every decoy in the scores table' needs a connected scores table",
                             ("models_from", "input:scores")))
        if s["models_from"] == "scores" and ctx.has("models"):
            out.append(Issue("info", "the connected models are measured too, besides the table's decoys",
                             ("models_from",)))
        filters, problem = self._filters(s)
        if problem:
            out.append(Issue("error", problem, ("filters",)))
        elif filters and not ctx.has("scores"):
            out.append(Issue("info", "the score filters are ignored without a scores table", ("filters",)))
        path = ctx.existing("scores")
        if path and filters:
            try:
                table = ifc.read_decoy_table(path)
                missing = [t for t, _o, _v in filters if not table.has(t)]
                if missing:
                    out.append(Issue("error", f"the scores table has no column {', '.join(missing)}",
                                     ("filters", "input:scores")))
            except (OSError, ifc.InterfaceError) as exc:
                out.append(Issue("warning", f"cannot read the scores table: {exc}", ("input:scores",)))
        if s["mg_min"] > s["mg_max"]:
            out.append(Issue("error", "the Mg-bridgeable window ends before it starts", ("mg_min", "mg_max")))
        if s["pairs"] == "dssr":
            out.append(Issue("info", "x3dna-dssr needs its own (free academic) licence and binary",
                             ("pairs",)))
        elif s["pairs"] == "barnaba":
            out.append(Issue("info", "the barnaba engine needs the optional package barnaba where the script "
                                     "runs (pip install barnaba)", ("pairs",)))
        out += self._check_references(s, ctx)
        out += self._check_support(s, ctx)
        model = ctx.existing("models")
        if model and side_a and not any(i.level == "error" for i in out):
            out += self._check_model(model, side_a, side_b)
        return out

    def _check_references(self, s, ctx) -> List[Issue]:
        out: List[Issue] = []
        refs = s["references"]
        if not s["cluster"]:
            if refs or ctx.has("references"):
                out.append(Issue("info", "reference interfaces are used by the clustering only",
                                 ("cluster",)))
            return out
        names = []
        for k, r in enumerate(refs):
            name = r["name"].strip()
            if not name or any(c in name for c in "=/ \t"):
                out.append(Issue("error", "a reference needs a name without spaces, '=' or '/'",
                                 (f"references.{k}.name",)))
            names.append(name)
            if not r["residues_a"].strip() and not r["residues_b"].strip():
                out.append(Issue("error", f"reference {name or k + 1}: give its residues",
                                 (f"references.{k}.residues_a",)))
            for side in ("residues_a", "residues_b"):
                try:
                    ifc.parse_side(r[side], ctx.groups)
                except ifc.InterfaceError as exc:
                    out.append(Issue("error", f"reference {name or k + 1}: {exc}",
                                     (f"references.{k}.{side}",)))
        if len(set(names)) != len(names):
            out.append(Issue("error", "reference names must be unique", ("references",)))
        return out

    def _check_support(self, s, ctx) -> List[Issue]:
        out: List[Issue] = []
        if not s["support"]:
            return out
        run = s["run"].strip().lower()
        if run != "auto" and not (run.isdigit() and int(run) >= 1):
            out.append(Issue("error", "flag runs of: a whole number of residues, or 'auto'", ("run",)))
        mode = s["rebuilt_from"]
        if mode == "upstream":
            if self._upstream_fixed(ctx) is None:
                out.append(Issue("error", "the connected models do not say which residues FARFAR2 kept "
                                          "fixed; type the fixed or the rebuilt residues", ("rebuilt_from",)))
        else:
            key = "fixed" if mode == "fixed" else "rebuilt"
            if not s[key].strip():
                out.append(Issue("error", f"type the {key} residues", (key,)))
            else:
                try:
                    ifc.parse_side(s[key], ctx.groups)
                except ifc.InterfaceError as exc:
                    out.append(Issue("error", f"{key} residues: {exc}", (key,)))
        if s["pair_credit"] != "off" and s["count_self"]:
            out.append(Issue("info", "with self-contact counted, base-pair credit changes nothing",
                             ("pair_credit",)))
        return out

    def _check_model(self, path, side_a, side_b) -> List[Issue]:
        from .. import pdbtools
        try:
            nts = ifc.nucleotides(pdbtools.read_pdb(path))
            ifc.split_sides(nts, side_a, side_b)
        except (OSError, pdbtools.PDBError) as exc:
            return [Issue("warning", f"cannot read the first model: {exc}", ("input:models",))]
        except ifc.InterfaceError as exc:
            return [Issue("warning", f"first model: {exc}", ("side_a", "side_b"))]
        return []

    # ------------------------------------------------------------ files
    @staticmethod
    def _side_options(s, ctx) -> List[str]:
        q = shell_quote
        words = ["--side-a", q(ifc.format_side(ifc.parse_side(s["side_a"], ctx.groups)))]
        if s["side_b"].strip():
            words += ["--side-b", q(ifc.format_side(ifc.parse_side(s["side_b"], ctx.groups)))]
        return words

    def _census_options(self, s, ctx) -> List[str]:
        q = shell_quote
        words = self._side_options(s, ctx)
        words += ["--cutoff", f"{s['contact']:g}", "--hbond", f"{s['hbond']:g}", "--pairs", s["pairs"]]
        if s["pairs"] != "off" and not s["wobble"]:
            words.append("--no-wobble")
        if s["pairs"] == "dssr":
            words += ["--dssr-exe", q(s["dssr_exe"].strip() or "x3dna-dssr")]
        if s["bph_ch"]:
            words.append("--bph-ch")
        for key, flag, default in (("stack_distance", "--stack-distance", 4.5),
                                   ("stack_angle", "--stack-angle", 30.0),
                                   ("stack_separation", "--stack-separation", 2.5)):
            if s[key] != default:
                words += [flag, f"{s[key]:g}"]
        if (s["mg_min"], s["mg_max"]) != (5.0, 7.0):
            words += ["--mg-window", f"{s['mg_min']:g},{s['mg_max']:g}"]
        return words

    def _rebuilt_options(self, s, ctx) -> List[str]:
        q = shell_quote
        mode = s["rebuilt_from"]
        if mode == "upstream":
            fixed = self._upstream_fixed(ctx) or ""
            return ["--fixed", q(ifc.format_side(ifc.parse_side(fixed))) if fixed.strip() else "''"]
        key = "fixed" if mode == "fixed" else "rebuilt"
        return [f"--{key}", q(ifc.format_side(ifc.parse_side(s[key], ctx.groups)))]

    @staticmethod
    def _array(name: str, paths: List[str]) -> List[str]:
        return [f"{name}=("] + [f"  {shell_quote(p)}" for p in paths] + [")"]

    def generate(self, s, ctx):
        q = shell_quote
        n = ctx.name
        script = n(self.tag, "sh")
        mdatk = StructureBuilderTool.mdatk(s)
        census, clusters, support = n("census", "tsv"), n("clusters", "tsv"), n("support", "tsv")
        notes = ["MDATK must be importable by the Python below (installed, or its folder on PYTHONPATH)."]
        sh = script_header("MDATK decoy interfaces: census" + (", clusters" if s["cluster"] else "")
                           + (", segment support" if s["support"] else ""), ctx, s, script, notes)
        models = ctx.inps("models")
        if models:
            sh += ["# The models", *self._array("MODELS", models), ""]
        census_words = mdatk + ["decoys", "census"] + (['"${MODELS[@]}"'] if models else [])
        census_words += self._census_options(s, ctx)
        if ctx.has("scores"):
            census_words += ["--scores", q(ctx.inp("scores"))]
            value = ctx.value("scores")
            if value is not None and value.source:  # a plan step's table: pdb relative to the plan folder
                root = posixpath.relpath(".", ctx.folder) if ctx.folder else "."
                census_words += ["--plan-root", q(root)]
            if s["models_from"] == "scores":
                census_words.append("--from-scores")
            filters, _problem = self._filters(s)
            census_words += [w for f in filters for w in ("--filter", q(ifc.format_filter(f)))]
            if s["drop_unscored"]:
                census_words.append("--drop-unscored")
        sh += ["# Interface census of every model (rows of the models that pass the score filters)",
               " ".join(census_words + ["-o", census]), require_file(census, "mdatk decoys census"), ""]
        if s["cluster"]:
            refs = ctx.inps("references")
            ref_census = n("ref-census", "tsv")
            if refs:
                sh += ["# Census of the reference models: each one is a reference interface",
                       *self._array("REFERENCES", refs),
                       " ".join(mdatk + ["decoys", "census", '"${REFERENCES[@]}"']
                                + self._census_options(s, ctx) + ["-o", ref_census]),
                       require_file(ref_census, "mdatk decoys census"), ""]
            reps = n("rep-REPNUM", "pdb")
            words = mdatk + ["decoys", "clusters", census, "--level", s["level"]]
            if s["level"] == "blocks":
                words += ["--block", str(s["block"])]
            words += ["--cut", f"{s['cut']:g}", "--min-size", str(s["min_size"]),
                      "--broad-limit", str(s["broad_limit"]), "--lead-gap", f"{s['lead_gap']:g}"]
            for r in s["references"]:
                a = ifc.format_side(ifc.parse_side(r["residues_a"], ctx.groups))
                b = ifc.format_side(ifc.parse_side(r["residues_b"], ctx.groups))
                words += ["--reference", q(f"{r['name'].strip()}={a}/{b}")]
            if refs:
                words += ["--reference-table", ref_census]
            words += ["--representatives", q(reps.replace("REPNUM", "{n}")), "--n-representatives",
                      str(s["n_reps"]), "--members", n("members", "tsv"), "-o", clusters]
            old = " ".join(reps.replace("REPNUM", f"{k:02d}") for k in range(1, s["n_reps"] + 1))
            sh += ["# Cluster the fingerprints; the medoids of the largest clusters become rep-01 ...",
                   f"rm -f -- {old}", " ".join(words), require_file(clusters, "mdatk decoys clusters"),
                   require_file(reps.replace("REPNUM", "01"), "mdatk decoys clusters"), ""]
        if s["support"]:
            words = mdatk + ["decoys", "support", "--census", census] + self._side_options(s, ctx)
            words += self._rebuilt_options(s, ctx)
            words += ["--sep", str(s["sep"]), "--cutoff", f"{s['support_cutoff']:g}",
                      "--run", q(s["run"].strip().lower())]
            if s["count_self"]:
                words.append("--count-self")
            if s["pair_credit"] != "off":
                words += ["--pairs", s["pair_credit"]]
            words += ["--residues", n("support-residues", "tsv"), "--profile", n("support-profile", "tsv"),
                      "-o", support]
            sh += ["# What holds the rebuilt residues of the models in the census", " ".join(words),
                   require_file(support, "mdatk decoys support"), ""]
        sh += [f'echo "Census: {census}"'] + ([f'echo "Clusters: {clusters}"'] if s["cluster"] else []) + \
              ([f'echo "Support: {support}"'] if s["support"] else []) + [""]
        return [GeneratedFile(script, "\n".join(sh), "run", "census, clusters and support of the models")]

    def summary_lines(self, s, ctx) -> List[str]:
        try:
            a = ifc.format_side(ifc.parse_side(s["side_a"], ctx.groups))
            b = (ifc.format_side(ifc.parse_side(s["side_b"], ctx.groups)) if s["side_b"].strip()
                 else "the rest")
        except ifc.InterfaceError:
            return []
        return [f"Interface between side A {a} and side B {b}."]
