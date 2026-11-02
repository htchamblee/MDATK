"""Planner step 'model-check': plausibility of candidate models before they are simulated or refined.

Clashes (Bondi overlap of heavy atoms, hydrogen bonds excused by chemistry, exclusions along bonds), O3'-P
links of every TER segment (strained links reported with their strain) and P-P spacing, compaction of loop
segments, sugar hands against expected ones and sharp kinks, for every connected model, and on request the
hydrogen contacts on MolProbity's conventions, reported only (:mod:`mdatk.modelcheck`).  The step writes a
report table and passes the models on: all of them (report), up to N of those that pass the gates in N
slots, the unfilled ones written as empty-slot files, failing only when fewer than the minimum pass (filter),
or none when one fails (stop).  The script runs a copy of MDATK's PDB tools and model checks that the step
writes into its folder (standard library only), so the machine that runs it needs ``python3`` and nothing
else.
"""

from __future__ import annotations

import inspect
import os
import posixpath
import re
from typing import Any, List, Optional, Tuple

from .. import PROJECT_ABBREV, __version__, modelcheck, pdbtools
from .base import PortValue, Tool, env_section, require_file, script_header, shell_quote
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section
from .scripts import embed


def _body(module) -> str:
    """Source of a module without its docstring and __future__ import (as :func:`embed` copies it)."""
    lines = inspect.getsource(module).splitlines()
    k = 0
    if lines and lines[0].startswith('"""'):
        k = 1
        if not (lines[0].count('"""') >= 2 and len(lines[0]) > 3):
            while k < len(lines) and '"""' not in lines[k]:
                k += 1
            k += 1
    return "\n".join(ln for ln in lines[k:] if not ln.startswith("from __future__")).strip("\n")


def helper_source(doc: str) -> str:
    """A standalone script: MDATK's PDB tools followed by its model checks (whose ``from . import pdbtools``
    then finds the tools above in the script itself), calling ``qc_main``; standard library only."""
    tools = ["# --- MDATK's PDB tools (mdatk.pdbtools) ---", _body(pdbtools), "", "",
             "# --- MDATK's model checks (mdatk.modelcheck) ---"]
    return embed(modelcheck, {}, "qc_main(sys.argv[1:])", doc, imports=tools)


def _group(group: Any, key: str) -> str:
    return str(group.get(key, "") if isinstance(group, dict) else getattr(group, key, ""))


def expand_groups(text: str, groups) -> str:
    """Replace ``group:NAME`` items of a range text by the residues of that project residue group (its
    chain-qualified items as they are, plain numbers for every chain: '1-40' -> '*:1-40').  KeyError names
    a missing group, ValueError a group without residue ranges."""
    out = []
    for item in re.split(r"[,;\s]+", (text or "").strip()):
        if not item:
            continue
        if not item.lower().startswith("group:"):
            out.append(item)
            continue
        name = item.split(":", 1)[1]
        match = next((g for g in groups if _group(g, "name") == name), None)
        if match is None:
            raise KeyError(name)
        residues = _group(match, "residues").strip()
        if not residues:
            raise ValueError(f"group {name} has no residue ranges (only a selection)")
        for sub in re.split(r"[,;\s]+", residues):
            if sub:
                out.append(sub if ":" in sub or not re.fullmatch(r"-?\d+(?:-(-?\d+))?", sub) else f"*:{sub}")
    return ", ".join(out)


ON_FAIL_CHOICES = [("report", "report only"), ("filter", "pass on the models that pass"),
                   ("stop", "stop when one fails")]


@register
class ModelCheckTool(Tool):
    name = "model-check"
    label = "Model check (clashes · backbone · chirality)"
    category = "Structure prediction"
    tag = "model-check"
    description = ("Plausibility of candidate models before MD or refinement: steric clashes (hydrogen bonds "
                   "excused by chemistry), O3′–P links and P–P spacing, loop compaction, sugar hands and "
                   "kinks; a report table, and the models passed on, filtered or stopped.")
    inputs = [Port("models", "Models (PDB)", ("pdb",), multiple=True,
                   tooltip="The models to check, e.g. the ranked FARFAR2 models or a refined model."),
              Port("secstruct", "Secondary structures", ("secstruct",), optional=True, multiple=True,
                   tooltip="Dot-brackets of the models' sequence (e.g. two registers): residues unpaired in "
                           "all of them are the loop segments of the compaction check."),
              Port("scores", "Decoy table", ("table",), optional=True,
                   tooltip="Rosetta score terms per decoy (run, tag, pdb, terms ...), e.g. from the FARFAR2 "
                           "models step: shown in the report, limits and ranking by score.")]
    outputs = [Port("report", "Report (TSV)", ("table",),
                    tooltip="One row per model: status, clashes, links and strain, compaction, hands, kinks, "
                            "score terms."),
               Port("passed", "Passed models (PDB)", ("pdb",), multiple=True,
                    tooltip="Filter: up to N slots; the slots no passing model fills are empty-slot files "
                            "(REMARK 951, no atoms), which MDATK's steps skip."),
               Port("best", "Best model (PDB)", ("pdb",))]
    sections = [
        Section("clashes", "Clashes", [
            Option("clash_tol", "Clash tolerance", "float", modelcheck.CLASH_TOL,
                   "Two atoms clash when they are closer than their van der Waals contact (Bondi radii, so "
                   "C…C gets 0.36 Å more room than O…O) by at least this much; the overlap is reported.",
                   minimum=0.0, maximum=2.0, step=0.05, digits=2, unit="Å"),
            Option("hbond_min", "Shortest hydrogen bond", "float", modelcheck.HBOND_WINDOW[0],
                   "An N/O pair with a donor on either side (2′-OH, amino groups, G N1, U N3, terminal "
                   "hydroxyls) is a hydrogen bond, not a clash, from this distance on; closer there is no "
                   "room for the hydrogen and it counts as a clash. Two pure acceptors never H-bond.",
                   minimum=1.5, maximum=3.5, step=0.05, digits=2, unit="Å"),
            Option("hbond_max", "Longest hydrogen bond", "float", modelcheck.HBOND_WINDOW[1],
                   "Upper end of the hydrogen-bond window.", minimum=2.0, maximum=4.0, step=0.05, digits=2,
                   unit="Å"),
            Option("waters", "Include waters", "bool", False, "Check water oxygens (donors and acceptors)."),
            Option("ions", "Ions", "radio", "skip",
                   "Skip ions, or check them against their first-shell distances (Mg²⁺–O 2.1 Å, Na⁺–O 2.4 Å, "
                   "K⁺–O 2.8 Å are coordination, not clashes).",
                   choices=[("skip", "skip"), ("check", "check")]),
            Option("protonated", "Protonated A⁺ / C⁺", "text", "",
                   "Residues whose A N1 or C N3 carries a proton (a donor then), e.g. 'B:30, C:52'. Ranges "
                   "as everywhere: 'A:7-12', 'C' (whole chain), 'group:NAME' (a residue group).",
                   placeholder="B:30"),
            Option("hydrogen_check", "Report hydrogen contacts", "bool", False,
                   "A separate report of the models' explicit hydrogens on MolProbity's conventions (Probe's "
                   "radii: 1.17 Å on carbon, 1.00 Å polar or aromatic at nuclear positions, 1.22 / 1.05 Å at "
                   "electron-cloud positions; overlap ≥ 0.4 Å; clashscore). Never a gate or a filter; the "
                   "clash check above never includes hydrogens (their positions differ between programs)."),
        ], tooltip="Steric clashes of heavy atoms; atoms up to three bonds apart are left out."),
        Section("backbone", "Backbone", [
            Option("link_min", "Shortest O3′–P link", "float", modelcheck.LINK_RANGE[0],
                   "O3′(i)–P(i+1) of consecutive nucleotides in a TER segment (what tleap bonds) shorter "
                   "than this is an overlap.", minimum=0.5, maximum=3.0, step=0.05, digits=2, unit="Å"),
            Option("link_max", "Longest unstrained O3′–P link", "float", modelcheck.LINK_RANGE[1],
                   "Longer links are reported as strained, with the strain tleap's bond puts into the first "
                   "minimisation, E = k(r − r0)² with Amber's OS–P term (230 kcal/mol/Å², 1.61 Å): about 35 "
                   "kcal/mol at 2.0 Å, 180 at 2.5 Å, 440 at 3.0 Å (a warning, not a failure).", minimum=1.0,
                   maximum=10.0, step=0.1, digits=2, unit="Å"),
            Option("link_fail", "Longest O3′–P link before a break", "float", modelcheck.LINK_FAIL,
                   "Longer is a chain break (usually two pieces), which fails the backbone gate.",
                   minimum=1.0, maximum=20.0, step=0.1, digits=2, unit="Å"),
            Option("pp_min", "Shortest P–P spacing", "float", modelcheck.PP_RANGE[0],
                   "Consecutive phosphorus atoms closer than this are reported.", minimum=1.0, maximum=10.0,
                   step=0.1, digits=1, unit="Å"),
            Option("pp_max", "Longest P–P spacing", "float", modelcheck.PP_RANGE[1],
                   "Consecutive phosphorus atoms farther apart are reported.", minimum=2.0, maximum=20.0,
                   step=0.1, digits=1, unit="Å"),
        ], tooltip="Backbone continuity within each TER segment."),
        Section("compaction", "Loop compaction", [
            Option("segments", "Loop segments", "text", "",
                   "Residue ranges whose end-to-end / contour ratio along C1′ is reported, e.g. 'A:7-12, "
                   "B:25-34, C:58-63' or 'group:Loop'. A pointer for the eye, not a score: below the flag a "
                   "loop is extremely coiled and worth a look.", placeholder="A:7-12, B:25-34, C:58-63"),
            Option("ss_segments", "Loops from the secondary structures", "bool", True,
                   "Also the runs of three or more residues unpaired in every connected secondary structure "
                   "(e.g. loops in both registers; a segment that is a hairpin in one and a strand in the "
                   "other is left out)."),
            Option("compaction_flag", "Flag below", "float", modelcheck.COMPACTION_FLAG,
                   "End-to-end / contour ratio below which a loop is flagged for inspection.", minimum=0.0,
                   maximum=1.0, step=0.05, digits=2),
        ], tooltip="End-to-end over contour length of loop segments."),
        Section("hands", "Chirality", [
            Option("hands", "Expected hands", "radio", "auto",
                   "Automatic: the hands measured on the first model when the plan is written (it must "
                   "exist then), or the hand of a helix from the structure builder; otherwise the hands are "
                   "only reported. Ranges: the residues typed below. None: report the hands only. Hands are "
                   "measured from the C4′, C3′ and C1′ geometry, never from names (a silent file keeps no L "
                   "names).", choices=[("auto", "automatic"), ("ranges", "ranges"), ("none", "none")]),
            Option("expect_l", "Expected L", "text", "", "Residues whose sugars must be L, e.g. 'C:45-73', "
                   "'C' or 'group:Target'.", placeholder="C:45-73", requires={"hands": "ranges"}),
            Option("expect_d", "Expected D", "text", "", "Residues whose sugars must be D, e.g. 'A, B'.",
                   placeholder="A, B", requires={"hands": "ranges"}),
        ], tooltip="Sugar chirality from the geometry, against the hands expected."),
        Section("kinks", "Kinks", [
            Option("kink_min", "Flag kinks below", "float", modelcheck.KINK_MIN,
                   "The virtual angle C1′(i−1)–C1′(i)–C1′(i+1) of every residue between two consecutive "
                   "nucleotides of a TER segment (A-form: 152°); below this a sharp kink is flagged.",
                   minimum=0.0, maximum=180.0, step=5.0, digits=0, unit="°"),
            Option("kink_allow", "Kinks allowed at", "text", "",
                   "Residues where a sharp kink is allowed and not flagged, e.g. the flexible positions of "
                   "in-line probing: 'B:22-25, B:31' or 'group:Flexible'.", placeholder="B:22-25"),
            Option("kink_reactivity", "Reactivity file", "file", "",
                   "A two-column file (position, reactivity), e.g. in-line probing: the positions whose "
                   "reactivity reaches the threshold below are allowed a kink too."),
            Option("kink_reactivity_min", "Allowed from reactivity", "float", 0.5,
                   "The reactivity from which a position is flexible (in the file's own units).",
                   minimum=-1e6, maximum=1e6, step=0.05, digits=3),
            Option("kink_reactivity_chain", "Chain of the positions", "text", "",
                   "The chain the file's positions number (empty: every chain).", placeholder="B"),
        ], tooltip="Sharp kinks of the chain from the virtual C1′ angle."),
        Section("gates", "Pass / fail", [
            Option("on_fail", "Models that fail", "radio", "report",
                   "Report: every model is passed on and the report says which fail. Filter: only models "
                   "that pass are passed on, up to the number below; slots no passing model fills are "
                   "written as empty-slot files, and the step fails only when fewer than the minimum pass. "
                   "Stop: the step fails when any model fails.", choices=ON_FAIL_CHOICES),
            Option("max_clashes", "Most clashes", "int", None,
                   "Models with more clashes fail (empty: any number).", minimum=0, maximum=100000,
                   optional=True),
            Option("gate_links", "Backbone links must hold", "bool", True,
                   "A model with an O3′–P link longer than the break length, shorter than the shortest link "
                   "or unmeasurable fails (strained links are reported, not failed)."),
            Option("gate_hands", "Hands must be as expected", "bool", True,
                   "A model fails with a sugar of the other hand than expected (or not measurable where a "
                   "hand is expected), or a sugar with an epimerised, contradicting or flat centre."),
            Option("gate_pp", "P–P spacing must hold", "bool", False,
                   "A model with a P–P spacing outside the range fails."),
            Option("gate_kinks", "No sharp kinks", "bool", False,
                   "A model with a sharp kink where none is allowed fails (off: reported only)."),
            Option("score_limits", "Score limits", "text", "",
                   "Limits on terms of the decoy table, e.g. 'linear_chainbreak<=1, score<=-50' (needs the "
                   "decoy table).", placeholder="linear_chainbreak<=1"),
            Option("n_pass", "Models passed on (up to)", "int", 5,
                   "Slots for the passing models (at most the number connected): filled in order, those left "
                   "over written as empty-slot files that the next steps skip. The report says how many "
                   "passed of how many.", minimum=1, maximum=100000, requires={"on_fail": "filter"}),
            Option("min_pass", "Fewest passing models", "int", 1,
                   "The step fails when fewer models pass (1: only when none does; 0: never, the best model "
                   "then being an empty slot when none passes).", minimum=0, maximum=100000,
                   requires={"on_fail": "filter"}),
            Option("order", "Order", "radio", "input",
                   "Order of the passed models (the first is the best model): as connected (e.g. the "
                   "FARFAR2 ranking), fewest clashes first, or lowest decoy-table term first.",
                   choices=[("input", "as connected"), ("clashes", "fewest clashes"),
                            ("score", "score term")]),
            Option("rank_term", "Rank by", "text", "score", "Decoy-table term to rank by (lowest first).",
                   requires={"order": "score"}),
            Option("score_terms", "Score terms in the report", "text", "score linear_chainbreak",
                   "Decoy-table columns copied into the report (when the table is connected)."),
        ], tooltip="Which models fail, and what happens to them."),
        env_section("module load python   (only python3 is needed)", [
            Option("python", "Python", "text", "python3", "Interpreter that runs the check (standard library "
                   "only; any Python 3.8 or newer)."),
        ]),
    ]

    # ------------------------------------------------------------ ports
    @staticmethod
    def _n_models(ctx) -> int:
        return len(ctx.inps("models"))

    def _n_passed(self, s, ctx) -> int:
        n = self._n_models(ctx)
        return min(s["n_pass"], n) if s["on_fail"] == "filter" else n

    def output_values(self, s, ctx):
        data = dict(ctx.data("models"))
        passed = [ctx.own(f"passed-{k:02d}", "pdb") for k in range(1, self._n_passed(s, ctx) + 1)]
        filtering = s["on_fail"] == "filter"
        # a filter fills its slots in order and writes the rest as empty slots: steps that read one model
        # (an Amber build, say) then stop before doing anything with an empty one
        slots = dict(data, empty_slots=True) if filtering else dict(data)
        best = dict(data, empty_slots=True) if filtering and s["min_pass"] == 0 else dict(data)
        return {"report": PortValue([ctx.own("report", "tsv")], "table"),
                "passed": PortValue(passed, "pdb", slots),
                "best": PortValue([ctx.own("best", "pdb")], "pdb", best)}

    def describe(self, s):
        policy = {"report": "report",
                  "filter": f"filter, up to {s['n_pass']} passed (at least {s['min_pass']})",
                  "stop": "stop on failure"}
        links = f"{s['link_min']:g}–{s['link_max']:g} Å (break > {s['link_fail']:g})"
        return f"clashes ≥{s['clash_tol']:g} Å, links {links}, {policy[s['on_fail']]}"

    # ------------------------------------------------------------ settings
    @staticmethod
    def _ranges(s, key, ctx) -> str:
        """A range option with its residue groups expanded (as typed when a group is missing: check() says
        so, and the files of a plan with errors are not written)."""
        try:
            return expand_groups(s[key], ctx.groups)
        except (KeyError, ValueError):
            return s[key].strip()

    def expected_hands(self, s, ctx) -> Tuple[str, str, str]:
        """(expected L, expected D, where they come from) - typed ranges, or (automatic) the hands measured on
        the first connected model that exists, or the builder's helix hand."""
        if s["hands"] == "none":
            return "", "", ""
        if s["hands"] == "ranges":
            return self._ranges(s, "expect_l", ctx), self._ranges(s, "expect_d", ctx), "ranges"
        value = ctx.value("models")
        for path in (value.paths if value else []):
            full = ctx.abspath(path)
            if not os.path.isfile(full):
                continue
            try:
                rows = pdbtools.residue_hands(pdbtools.read_pdb(full))
            except (OSError, pdbtools.PDBError):
                return "", "", ""
            l_res = [r.residue for r in rows if r.call == "L"]
            d_res = [r.residue for r in rows if r.call == "D"]
            return pdbtools.ranges_text(l_res), pdbtools.ranges_text(d_res), os.path.basename(full)
        hand = ctx.data("models").get("chirality")
        if hand == "L":
            return "*", "", "the builder's L helix"
        if hand == "D":
            return "", "*", "the builder's D helix"
        return "", "", ""

    def check(self, s, ctx):
        out: List[Issue] = []
        for key in ("protonated", "segments", "expect_l", "expect_d", "kink_allow"):
            try:
                pdbtools.parse_ranges(expand_groups(s[key], ctx.groups) or None)
            except KeyError as exc:
                out.append(Issue("error", f"no residue group named {exc.args[0]}", (key,)))
            except (ValueError, pdbtools.PDBError) as exc:
                out.append(Issue("error", str(exc), (key,)))
        for lo, hi, what in (("hbond_min", "hbond_max", "hydrogen-bond window"),
                             ("link_min", "link_max", "O3′–P limits"), ("pp_min", "pp_max", "P–P range")):
            if s[lo] >= s[hi]:
                out.append(Issue("error", f"{what}: the minimum must be below the maximum", (lo, hi)))
        if s["link_fail"] < s["link_max"]:
            out.append(Issue("error", "the O3′–P break length must not be below the longest unstrained link",
                             ("link_fail", "link_max")))
        reactivity = s["kink_reactivity"].strip()
        if reactivity and not os.path.isfile(ctx.abspath(reactivity)):
            out.append(Issue("warning", f"the reactivity file {reactivity} does not exist yet (it is read "
                                        "when the script runs)", ("kink_reactivity",)))
        chain = s["kink_reactivity_chain"].strip()
        if chain and not re.fullmatch(r"[A-Za-z0-9]", chain):
            out.append(Issue("error", "the chain of the reactivity positions is one letter or digit",
                             ("kink_reactivity_chain",)))
        try:
            limits = modelcheck.parse_score_limits(s["score_limits"])
        except ValueError as exc:
            limits = []
            out.append(Issue("error", str(exc), ("score_limits",)))
        scores = ctx.has("scores")
        if limits and not scores:
            out.append(Issue("error", "score limits need the decoy table: connect the scores input",
                             ("score_limits", "input:scores")))
        if s["order"] == "score" and not scores:
            out.append(Issue("error", "ranking by a score term needs the decoy table: connect the scores "
                                      "input", ("order", "input:scores")))
        if s["order"] == "score" and not s["rank_term"].strip():
            out.append(Issue("error", "give the score term to rank by", ("rank_term",)))
        n = self._n_models(ctx)
        if s["on_fail"] == "filter" and n and s["n_pass"] > n:
            out.append(Issue("info", f"{n} model(s) connected: {n} passed on at most", ("n_pass",)))
        if s["on_fail"] == "filter" and n and s["min_pass"] > n:
            out.append(Issue("error", f"{n} model(s) connected: {s['min_pass']} can never pass",
                             ("min_pass",)))
        if s["on_fail"] == "report" and (s["max_clashes"] is not None or limits):
            out.append(Issue("info", "the gates only mark models in the report ('Models that fail': report)",
                             ("on_fail",)))
        if not s["segments"].strip() and not (s["ss_segments"] and ctx.has("secstruct")):
            out.append(Issue("info", "no loop segments: type ranges or connect secondary structures for the "
                                     "compaction check", ("segments",)))
        if not any(i.level == "error" for i in out):
            exp_l, exp_d, source = self.expected_hands(s, ctx)
            if s["hands"] == "auto" and not source:
                out.append(Issue("info", "no model exists yet to measure the hands on: they are reported, "
                                         "not checked (choose Ranges to type them)", ("hands",)))
            elif s["hands"] == "auto":
                out.append(Issue("info", f"expected hands as measured on {source}: "
                                         + "; ".join(t for t in (f"L {exp_l}" if exp_l else "",
                                                                  f"D {exp_d}" if exp_d else "") if t),
                                 ("hands",)))
            elif s["hands"] == "ranges" and not (exp_l or exp_d):
                out.append(Issue("warning", "type the residues expected L or D (or choose automatic / none)",
                                 ("expect_l", "expect_d")))
        if not s["python"].strip():
            out.append(Issue("error", "give the Python interpreter", ("python",)))
        return out

    # ------------------------------------------------------------ files
    def arguments(self, s, ctx) -> List[str]:
        """Command-line arguments of the check (after the helper), each option with its value, file names
        relative to the step folder."""
        n = ctx.name
        q = shell_quote
        out = [q(p) for p in ctx.inps("models")]

        def opt(flag: str, value: Optional[str] = None) -> None:
            out.append(flag if value is None else f"{flag} {value}")

        for key, flag in (("clash_tol", "--clash-tol"), ("hbond_min", "--hbond-min"),
                          ("hbond_max", "--hbond-max")):
            opt(flag, f"{s[key]:g}")
        opt("--ions", s["ions"])
        if s["waters"]:
            opt("--waters")
        if s["hydrogen_check"]:
            opt("--hydrogen-check")
        for key, flag in (("protonated", "--protonated"), ("segments", "--segments"),
                          ("kink_allow", "--kink-allow")):
            text = self._ranges(s, key, ctx)
            if text:
                opt(flag, q(text))
        for key, flag in (("link_min", "--link-min"), ("link_max", "--link-max"),
                          ("link_fail", "--link-fail"), ("pp_min", "--pp-min"), ("pp_max", "--pp-max"),
                          ("compaction_flag", "--compaction-flag"), ("kink_min", "--kink-min")):
            opt(flag, f"{s[key]:g}")
        if s["kink_reactivity"].strip():
            opt("--kink-reactivity", q(ctx.rel(s["kink_reactivity"].strip())))
            opt("--kink-reactivity-min", f"{s['kink_reactivity_min']:g}")
            if s["kink_reactivity_chain"].strip():
                opt("--kink-reactivity-chain", s["kink_reactivity_chain"].strip())
        if s["ss_segments"]:
            for path in ctx.inps("secstruct"):
                opt("--secstruct", q(path))
        exp_l, exp_d, _source = self.expected_hands(s, ctx)
        if exp_l:
            opt("--expect-l", q(exp_l))
        if exp_d:
            opt("--expect-d", q(exp_d))
        if ctx.has("scores"):
            root = posixpath.relpath(".", ctx.folder) if ctx.folder else "."
            terms = ",".join(t for t in re.split(r"[,\s]+", s["score_terms"]) if t)
            opt("--scores", q(ctx.inp("scores")))
            opt("--plan-root", q(root))
            opt("--score-terms", q(terms))
        if s["max_clashes"] is not None:
            opt("--max-clashes", str(s["max_clashes"]))
        for key, flag, on in (("gate_links", "--no-gate-links", False),
                              ("gate_hands", "--no-gate-hands", False), ("gate_pp", "--gate-pp", True),
                              ("gate_kinks", "--gate-kinks", True)):
            if bool(s[key]) == on:
                opt(flag)
        if s["score_limits"].strip():
            opt("--score-limits", q(s["score_limits"].strip()))
        opt("--on-fail", s["on_fail"])
        if s["on_fail"] == "filter":
            opt("--pass", str(self._n_passed(s, ctx)))
            opt("--min-pass", str(s["min_pass"]))
        opt("--order", s["order"])
        if s["order"] == "score":
            opt("--rank-term", q(s["rank_term"].strip()))
        opt("--passed-out", q(n("passed-MODELNUM", "pdb").replace("MODELNUM", "{n:02d}")))
        opt("--best-out", n("best", "pdb"))
        opt("--tsv -o", n("report", "tsv"))
        return out

    def generate(self, s, ctx):
        n = ctx.name
        script, helper = n(self.tag, "sh"), n("modelcheck", "py")
        report, log, best = n("report", "tsv"), n("report", "log"), n("best", "pdb")
        sh = script_header("Model check: clashes, O3'-P links, loop compaction, sugar hands", ctx, s, script)
        passed = [n(f"passed-{k:02d}", "pdb") for k in range(1, self._n_passed(s, ctx) + 1)]
        stale = n("passed-MODELNUM", "pdb").replace("MODELNUM", "[0-9]*")
        sh += ["# Outputs of an earlier run go first: a failed check must leave nothing that looks finished",
               f"rm -f {report} {best} {stale}", ""]
        what = {"report": "every model is passed on; the report marks those that fail",
                "filter": f"up to {len(passed)} models that pass are passed on (the slots left over are "
                          f"written as empty slots); the step fails when fewer than {s['min_pass']} pass",
                "stop": "the step fails when any model fails"}[s["on_fail"]]
        lines, cur = [], ""
        for w in [shell_quote(s["python"].strip() or "python3"), helper] + self.arguments(s, ctx):
            if cur and len(cur) + 1 + len(w) > 100:  # one option per continued line, roughly
                lines.append(cur + " \\")
                cur = "    " + w
            else:
                cur = f"{cur} {w}" if cur else w
        lines.append(f"{cur} 2>&1 | tee {log}")
        sh += [f"# Check {self._n_models(ctx)} model(s): {what}"] + lines + [
            require_file(report, "the model check"), require_file(best, "the model check"),
            f'echo "Report: {report}  Best model: {best}"', ""]
        doc = (f"Model checks of RNA models - written by {PROJECT_ABBREV} {__version__} (mdatk.pdbtools and "
               "mdatk.modelcheck, standard library only).\n\nUsage: python3 THIS_SCRIPT MODEL.pdb ... "
               "[--tsv -o REPORT.tsv] [--expect-l C:45-73] [--on-fail report|filter|stop] ...  (--help lists "
               "all)")
        return [GeneratedFile(script, "\n".join(sh), "run", f"checks the models, writes {report}"),
                GeneratedFile(helper, helper_source(doc), "helper",
                              "clash, backbone, compaction and chirality checks (mdatk pdb qc)")]
