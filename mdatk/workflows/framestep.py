"""Planner steps around FARFAR2 runs and their models: 'rosetta-audit' (what each rna_denovo run was given,
:mod:`mdatk.rosetta`) and 'frame-check' (did the models keep the reference's placement, and can they be moved
into a re-fitted reference's frame, :mod:`mdatk.frames`).  Both write a short bash script that calls
``python3 -m mdatk rosetta audit`` / ``python3 -m mdatk frame ...``."""

from __future__ import annotations

import os
import posixpath
import re
from typing import List

from .. import frames, rosetta
from .base import PortValue, Tool, env_section, require_file, script_header, shell_quote
from .buildstep import StructureBuilderTool
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section


def _python_option() -> Option:
    return Option("python", "Python", "text", "python3", "Interpreter that can import MDATK. From the "
                  "standalone AppImage / Windows build the script calls that program itself instead.")


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# rosetta-audit
# ---------------------------------------------------------------------------


@register
class RosettaAuditTool(Tool):
    name = "rosetta-audit"
    label = "Rosetta · run audit (what each run was given)"
    category = "Structure prediction"
    tag = "rosetta-audit"
    description = ("Reads the command line each rna_denovo run logged, with its flags files expanded, "
                   "fingerprints the input files it names and compares the conditions, also in pairs that "
                   "should differ only in the pairing; reports the Rosetta build and the seed of every run.")
    inputs = [Port("silent", "Silent files", ("silent",), optional=True, multiple=True,
                   tooltip="Silent files of FARFAR2 steps: their run logs lie beside them, and each step is "
                           "one condition.")]
    outputs = [Port("report", "Audit report", ("text",)),
               Port("table", "Flags per condition", ("table",),
                    tooltip="One row per flag (and derived entry), one column per condition."),
               Port("runs", "Seed and build per run", ("table",),
                    tooltip="One row per start of every run: the seed (-constant_seed / -jran, or the seed "
                            "Rosetta logged) and the Rosetta build named before the command line.")]
    sections = [
        Section("audit", "Audit", [
            Option("folders", "Other run folders", "multiline", "",
                   "Folders of runs made outside this plan, one per line (absolute, or relative to the plan "
                   "folder). A folder whose own logs hold rna_denovo command lines is one condition; "
                   "otherwise each of its subfolders is.", placeholder="/data/campaign"),
            Option("pairs", "Pairs", "multiline", "",
                   "CONTROL=TREATMENT, one per line: conditions that should differ only in the pairing. A "
                   "condition is named by its folder (a FARFAR2 step: 01_farfar2; a subfolder: its name).",
                   placeholder="01_farfar2=05_farfar2"),
            Option("pair_suffix", "Pair by name suffix", "text", "",
                   "Also pair every condition X with X<suffix>.", placeholder="_nointernalpairs"),
            Option("controlled", "Controlled residues", "text", "",
                   "Residues whose pairing is the variable under test, numbered as in the FASTA. The report "
                   "says how each condition gives their pairing: in the secondary structure, as obligate "
                   "pairs, or fixed in the -s structure / -input_res.", placeholder="A:22-24,A:35-37"),
            Option("allowed", "Also allowed to differ", "text", "",
                   "Flags that may differ in a pair besides -secstruct_file, -obligate_pair(_explicit) and "
                   "-filter_lores_base_pairs.", placeholder="-cycles"),
            Option("ignored", "Also ignored", "text", "",
                   "Flags left out everywhere besides those that differ per run by design (seeds, output "
                   "names, -nstruct, -database).", placeholder="-out:level"),
            Option("full_hash", "Fingerprint whole files", "bool", True,
                   "Hash every byte of each input file the flags name (off: three 1 MiB samples of large "
                   "files)."),
            Option("strict", "Fail on a confounded pair or duplicate seeds", "bool", False,
                   "The step fails when a pair differs in other flags or names an unknown condition, or when "
                   "two runs share a seed under -constant_seed with the same inputs (duplicate decoys), so "
                   "the steps after it do not run. Duplicates are an error in the report either way."),
        ], tooltip="Which runs to read and how to compare them."),
        env_section("conda activate mdatk", [_python_option()]),
    ]

    def describe(self, s):
        n = len(_lines(s["pairs"]))
        return f"{n} pair(s)" if n else "flags of every condition"

    @staticmethod
    def _step_roots(ctx) -> List[str]:
        return list(dict.fromkeys(posixpath.dirname(p) or "." for p in ctx.inps("silent")))

    def _roots(self, s, ctx) -> List[str]:
        """Condition folders as seen from the step folder: the folders of the connected silent files, then the
        typed folders."""
        roots = self._step_roots(ctx)
        for line in _lines(s["folders"]):
            path = os.path.expanduser(line)
            roots.append(path if os.path.isabs(path) else ctx.rel(path))
        return list(dict.fromkeys(roots))

    def check(self, s, ctx):
        out: List[Issue] = []
        if not ctx.has("silent") and not _lines(s["folders"]):
            out.append(Issue("error", "connect the silent files of a FARFAR2 step or give run folders",
                             ("input:silent", "folders")))
        for line in _lines(s["folders"]):
            path = os.path.expanduser(line)
            if os.path.isabs(path) and not os.path.isdir(path):
                out.append(Issue("warning", f"run folder not found (yet): {path}", ("folders",)))
        labels = {posixpath.basename(posixpath.normpath(r)) for r in self._step_roots(ctx)}
        for line in _lines(s["pairs"]):
            a, sep, b = line.partition("=")
            if not sep or not a.strip() or not b.strip():
                out.append(Issue("error", f"pair {line!r}: write CONTROL=TREATMENT", ("pairs",)))
            elif not _lines(s["folders"]):
                unknown = [x.strip() for x in (a, b) if x.strip() not in labels]
                if unknown:
                    known = ", ".join(sorted(labels)) or "none"
                    out.append(Issue("warning", f"pair {line!r}: {', '.join(unknown)} is not one of the "
                                                f"connected steps ({known})", ("pairs",)))
        if s["controlled"].strip():
            try:
                rosetta.parse_residues(s["controlled"])
            except ValueError as exc:
                out.append(Issue("error", str(exc), ("controlled",)))
        if not s["python"].strip():
            out.append(Issue("error", "give the Python interpreter", ("python",)))
        return out

    def output_values(self, s, ctx):
        return {"report": PortValue([ctx.own(self.tag, "txt")], "text"),
                "table": PortValue([ctx.own(self.tag, "tsv")], "table"),
                "runs": PortValue([ctx.own("runs", "tsv")], "table")}

    def generate(self, s, ctx):
        n = ctx.name
        script = n(self.tag, "sh")
        q = shell_quote
        words = StructureBuilderTool.mdatk(s) + ["rosetta", "audit"] + [q(r) for r in self._roots(s, ctx)]
        for line in _lines(s["pairs"]):
            words += ["--pair", q(re.sub(r"\s*=\s*", "=", line))]
        if s["pair_suffix"].strip():
            words += ["--pair-suffix", q(s["pair_suffix"].strip())]
        if s["controlled"].strip():
            words += ["--controlled", q(re.sub(r"\s+", ",", s["controlled"].strip()))]
        for key, flag in (("allowed", "--allow"), ("ignored", "--ignore")):
            for f in s[key].split():
                words.append(f"{flag}={q(f)}")  # one word: argparse would read '-cycles' as an option
        if not s["full_hash"]:
            words.append("--fast-hash")
        if s["strict"]:
            words.append("--strict")
        words += ["-o", n(self.tag, "txt"), "--tsv", n(self.tag, "tsv"), "--runs-tsv", n("runs", "tsv")]
        sh = script_header("Rosetta run audit: what each rna_denovo run was given", ctx, s, script,
                           ["MDATK must be importable by the Python below (installed, or its folder on "
                            "PYTHONPATH)."])
        sh += [" ".join(words), require_file(n(self.tag, "txt"), "mdatk rosetta audit"), ""]
        return [GeneratedFile(script, "\n".join(sh), "run", "audits the rna_denovo runs")]


# ---------------------------------------------------------------------------
# frame-check
# ---------------------------------------------------------------------------


@register
class FrameCheckTool(Tool):
    name = "frame-check"
    label = "MDATK · frame check (fit one body, measure the others)"
    category = "Analysis"
    tag = "frame-check"
    description = ("Superposes every model on one body of a reference and measures where the other bodies "
                   "went: one shared frame, or placements that were sampled. With a re-fitted reference it "
                   "decides whether one rigid transform moves the models into the new frame, and applies it.")
    inputs = [Port("reference", "Reference (PDB)", ("pdb",),
                   tooltip="The placed bodies the models were built on (e.g. the helices given to rna_denovo "
                           "with -s)."),
              Port("models", "Models (PDB)", ("pdb",), optional=True, multiple=True,
                   tooltip="Model PDB files, e.g. the ranked models of a FARFAR2 models step."),
              Port("scores", "Decoy table", ("table",), optional=True,
                   tooltip="A decoy table (run, tag, pdb, score terms): its extracted models are checked.")]
    new_reference = Port("new_reference", "Re-fitted reference (PDB)", ("pdb",),
                         tooltip="The same bodies placed anew (another map, a sharpened map).")
    sections = [
        Section("frame", "Frame", [
            Option("fit", "Fit on", "text", "group:Aptamer",
                   "Residues every model is superposed on: chain-qualified ranges ('A:1-16,B:17-44'), a "
                   "whole chain ('C'), numbers in any chain ('1-44') or a residue group (group:Aptamer).",
                   placeholder="A:1-16,B:17-44"),
            Option("measure", "Measure (one body per line)", "multiline", "group:Target",
                   "Bodies whose RMSD is measured under that superposition, one per line, optionally named: "
                   "'Target=C:1-29' or group:Target.", placeholder="Target=C:1-29"),
            Option("atoms", "Atoms", "text", "P C1' C4'", "Atom names compared, in residues present in "
                   "both the reference and the model."),
            Option("min_atoms", "Fewest atoms", "int", frames.MIN_ATOMS, "Shared atoms each selection needs.",
                   minimum=3, maximum=100000),
            Option("model_map", "Model numbering", "text", "",
                   "When the models are numbered differently from the reference: REFERENCE=MODEL ranges "
                   "whose residues pair up in order (empty: the same chains and numbers).",
                   placeholder="C:1-29=B:45-73"),
            Option("valid_fit", "Fit body found below", "float", frames.VALID_FIT,
                   "A model whose fit-body RMSD is larger does not contain the reference's fit body; it is "
                   "listed but left out of the statistics.", minimum=0.01, maximum=100.0, step=0.1,
                   digits=2, unit="Å"),
            Option("same_frame", "One frame below (95th percentile)", "float", frames.SAME_FRAME,
                   "The models share one frame when the 95th percentile of every measured body is smaller.",
                   minimum=0.01, maximum=100.0, step=0.1, digits=2, unit="Å"),
            Option("score_cut", "Decoys of the table", "text", "",
                   "Only the decoys with TERM <= VALUE (several separated by spaces), e.g. "
                   "linear_chainbreak=5.", placeholder="linear_chainbreak=5"),
            Option("jobs", "Parallel processes", "int", 1, "Models read at the same time.", minimum=1,
                   maximum=256),
            Option("strict", "Fail unless one frame", "bool", False,
                   "The step fails when the models do not share one frame, so the steps after it do not "
                   "run."),
        ], tooltip="Which body to superpose on and which to measure."),
        Section("reframe", "Re-fitted reference", [
            Option("reframe", "Move the models into a re-fitted reference's frame", "bool", False,
                   "Adds the input 'Re-fitted reference'. The old reference is superposed onto it on the "
                   "fit body; when every measured body is then carried to its new place (within the "
                   "tolerance), one rigid transform describes the re-fit and the models are moved with it. "
                   "Otherwise the bodies moved relative to each other: the step fails, and the models must "
                   "be built again on the new reference."),
            Option("tol", "Tolerance", "float", frames.TOLERANCE,
                   "Largest carried RMSD of a measured body for one rigid transform.", minimum=0.01,
                   maximum=100.0, step=0.1, digits=2, unit="Å", requires={"reframe": True}),
        ]),
        env_section("conda activate mdatk", [_python_option()]),
    ]

    def input_ports(self, s):
        return list(self.inputs) + ([self.new_reference] if s.get("reframe") else [])

    def output_ports(self, s):
        out = [Port("report", "Frame report (per model)", ("table",)),
               Port("summary", "Summary", ("text",))]
        if s.get("reframe"):
            out += [Port("reframed", "Moved models (PDB)", ("pdb",), multiple=True,
                         tooltip="The models of the input 'Models', moved into the re-fitted frame."),
                    Port("reframed_scores", "Decoy table of the moved decoys", ("table",),
                         tooltip="The input decoy table with pdb pointing at the moved decoys.")]
        return out

    def describe(self, s):
        return f"fit {s['fit'].strip() or '?'}" + (", reframe" if s["reframe"] else "")

    def _pattern(self, ctx) -> str:
        return ctx.name("reframed-MODELNUM", "pdb").replace("MODELNUM", "{n}")

    def output_values(self, s, ctx):
        out = {"report": PortValue([ctx.own(self.tag, "tsv")], "table"),
               "summary": PortValue([ctx.own(self.tag, "txt")], "text")}
        if s["reframe"]:
            n = len(ctx.inps("models"))
            if n:
                paths = [ctx.own(f"reframed-{k:02d}", "pdb") for k in range(1, n + 1)]
                out["reframed"] = PortValue(paths, "pdb", dict(ctx.data("models")))
            if ctx.has("scores"):
                out["reframed_scores"] = PortValue([ctx.own("reframed-scores", "tsv")], "table",
                                                   dict(ctx.data("scores")))
        return out

    def _selections(self, s, ctx):
        return (frames.selection_text(s["fit"], ctx.groups), frames.bodies_text(s["measure"], ctx.groups))

    def check(self, s, ctx):
        out: List[Issue] = []
        try:
            fit, bodies = self._selections(s, ctx)
            if not fit:
                out.append(Issue("error", "say which residues to superpose on", ("fit",)))
            if not bodies:
                out.append(Issue("error", "give at least one body to measure", ("measure",)))
        except ValueError as exc:
            out.append(Issue("error", str(exc), ("fit", "measure")))
        if s["model_map"].strip():
            try:
                frames.parse_model_map(s["model_map"])
            except ValueError as exc:
                out.append(Issue("error", str(exc), ("model_map",)))
        try:
            rosetta.parse_maxima(s["score_cut"].split())
        except ValueError as exc:
            out.append(Issue("error", str(exc), ("score_cut",)))
        if s["score_cut"].strip() and not ctx.has("scores"):
            out.append(Issue("warning", "a score cut is set but no decoy table is connected", ("score_cut",)))
        if not ctx.has("models") and not ctx.has("scores"):
            out.append(Issue("error", "connect models (PDB) or a decoy table",
                             ("input:models", "input:scores")))
        if not re.split(r"[,\s]+", s["atoms"].strip())[0]:
            out.append(Issue("error", "name the atoms to compare", ("atoms",)))
        if s["reframe"]:
            dens = max(float(ctx.data(p).get("density_weight") or 0.0) for p in ("models", "scores"))
            if dens > 0:
                out.append(Issue("warning", f"the models were sampled with the map in the score (density "
                                            f"weight {dens:g}): they were drawn towards the old "
                                            "placement, so moving them keeps that; it does not reproduce "
                                            "sampling in the new frame", ("reframe",)))
        if not s["python"].strip():
            out.append(Issue("error", "give the Python interpreter", ("python",)))
        return out

    def _common(self, s, ctx, fit: str, bodies: List[str]) -> List[str]:
        q = shell_quote
        words = [q(p) for p in ctx.inps("models")]
        if ctx.has("scores"):
            root = posixpath.relpath(".", ctx.folder) if ctx.folder else "."  # the plan folder
            words += ["--scores", q(ctx.inp("scores")), "--root", q(root)]
            words += [w for c in s["score_cut"].split() for w in ("--max", q(c))]
        words += ["--fit", q(fit)] + [w for b in bodies for w in ("--measure", q(b))]
        words += ["--atoms", q(",".join(a for a in re.split(r"[,\s]+", s["atoms"].strip()) if a)),
                  "--min-atoms", str(s["min_atoms"])]
        if s["model_map"].strip():
            words += ["--model-map", q(re.sub(r"\s+", ",", s["model_map"].strip()))]
        return words

    def generate(self, s, ctx):
        n = ctx.name
        script = n(self.tag, "sh")
        q = shell_quote
        fit, bodies = self._selections(s, ctx)
        common = self._common(s, ctx, fit, bodies)
        mdatk = StructureBuilderTool.mdatk(s)
        check = mdatk + ["frame", "check", q(ctx.inp("reference"))] + common + [
            "--valid-fit", f"{s['valid_fit']:g}", "--same-frame", f"{s['same_frame']:g}",
            "--jobs", str(s["jobs"]), "--tsv", n(self.tag, "tsv"), "-o", n(self.tag, "txt")]
        if s["strict"]:
            check.append("--strict")
        notes = ["MDATK must be importable by the Python below (installed, or its folder on PYTHONPATH).",
                 f"Fit on {fit}; measured: {', '.join(bodies)}."]
        sh = script_header("MDATK frame check: fit one body, measure the others", ctx, s, script, notes)
        sh += ["# Superpose every model on the fit body of the reference and measure the other bodies",
               " ".join(check), require_file(n(self.tag, "tsv"), "mdatk frame check"), ""]
        if s["reframe"]:
            words = mdatk + ["frame", "reframe", q(ctx.inp("reference")), q(ctx.inp("new_reference"))]
            words += common + ["--tol", f"{s['tol']:g}", "--out-dir", ".", "--pattern", q(self._pattern(ctx)),
                               "--decoy-pattern", q(n("reframed") + "/{run}/{tag}.pdb"),
                               "--manifest", n("reframed", "tsv"), "-o", n("reframed", "txt")]
            if ctx.has("scores"):
                words += ["--scores-out", n("reframed-scores", "tsv")]
            sh += ["# One rigid transform from the old to the re-fitted reference, if it suffices, moves the "
                   "models;", "# otherwise this stops with an error: the models must be built again",
                   " ".join(words), require_file(n("reframed", "tsv"), "mdatk frame reframe"), ""]
        return [GeneratedFile(script, "\n".join(sh), "run", "checks the frame of the models")]
