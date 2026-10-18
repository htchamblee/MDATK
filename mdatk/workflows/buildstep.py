"""Planner step of the in-house starting-structure builder (mdatk.builder): an ideal A-form helix (D or
L), base mutations with compensatory partners, or a rigid-body fit into a density map.  The step writes
a short bash script that calls ``python3 -m mdatk build ...``."""

from __future__ import annotations

from typing import List, Optional

from .. import builder, pdbtools
from . import rna
from .base import PortValue, Tool, env_section, require_file, script_header, shell_quote
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

MODES = [("helix", "Ideal A-form helix"), ("mutate", "Mutate bases"), ("fit", "Fit into a density map")]
SCHEME_CHOICES = [("", "default (A/C/G/U, L: 0A/0C/0G/0U)"), ("amber", "Amber (L: LA … + 5'/3')"),
                  ("ccd", "CCD for Coot / Phenix (L: 0A … + 5'/3')"), ("rosetta", "Rosetta"),
                  ("pdb", "PDB deposition")]


@register
class StructureBuilderTool(Tool):
    name = "structure-builder"
    label = "Structure builder (MDATK)"
    category = "Structure building"
    tag = "build"
    description = ("Starting structures without external programs: ideal A-form RNA helices (D or L), "
                   "base mutations keeping the glycosidic frame (with compensatory partners), and "
                   "rigid-body fitting into a density map.")
    inputs = [Port("pdb", "Model (PDB)", ("pdb",), tooltip="The structure to mutate or to fit."),
              Port("map", "Density map", ("map",), tooltip="MRC / CCP4 map for the rigid-body fit.")]
    outputs = [Port("pdb", "Structure (PDB)", ("pdb",)),
               Port("secstruct", "Secondary structure", ("secstruct",),
                    tooltip="Duplex pairs of the ideal helix (Watson-Crick and G·U), e.g. for helix restraints.")]
    sections = [
        Section("mode", "Mode", [
            Option("mode", "Build", "radio", "helix", "Ideal helix: a new A-form duplex or single strand. "
                   "Mutate: swap bases of the connected PDB. Fit: place the connected PDB in the map.",
                   choices=MODES),
        ]),
        Section("helix", "Ideal helix", [
            Option("sequence", "Sequence (5'->3')", "text", "", "Strand 1; T is read as U.",
                   placeholder="GGCAUCGAUGCC"),
            Option("partner_mode", "Partner strand", "radio", "auto",
                   "Watson-Crick complement, none (single strand) or a sequence typed below.",
                   choices=[("auto", "complement"), ("none", "none"), ("explicit", "explicit")]),
            Option("partner", "Partner (5'->3')", "text", "", "Same length as strand 1; mismatches and G·U "
                   "wobbles are allowed.", requires={"partner_mode": "explicit"}),
            Option("chirality", "Chirality", "radio", "D", "D: right-handed. L: the mirror image (left-handed, "
                   "L-nucleotides).", choices=[("D", "D-RNA"), ("L", "L-RNA")]),
            Option("chains", "Chain ids", "text", "A,B", "Strand 1, strand 2."),
            Option("start1", "First residue (strand 1)", "int", 1, "", minimum=-999, maximum=9999),
            Option("start2", "First residue (strand 2)", "int", 1, "", minimum=-999, maximum=9999,
                   requires={"partner_mode": ("auto", "explicit")}),
            Option("scheme", "Residue names", "choice", "", "Name scheme of mdatk.pdbtools.",
                   choices=SCHEME_CHOICES),
            Option("terminals", "5'/3' terminal names", "bool", True, "For the schemes with terminal "
                   "variants (Amber, CCD)."),
            Option("phosphate5", "5'-phosphate", "bool", False, "Keep P/OP1/OP2/OP3 on the 5' ends (the "
                   "Amber 5' units have none)."),
            Option("hydrogens", "Hydrogens", "bool", False, "Add all hydrogens (default: heavy atoms only)."),
        ], requires={"mode": "helix"}),
        Section("mutate", "Mutations", [
            Option("mutations", "Mutations", "text", "", "Chain:residue + new base, comma separated.",
                   placeholder="A:5G, B:20C"),
            Option("compensatory", "Compensatory partner mutations", "bool", True,
                   "Change the Watson-Crick / G·U partner of every mutated base to the complement unless "
                   "it is mutated explicitly."),
            Option("wobble", "Keep G·U pairs", "bool", False, "A G or U facing U / G keeps its partner (wobble) "
                   "instead of the Watson-Crick complement.", requires={"compensatory": True}),
        ], requires={"mode": "mutate"}),
        Section("fit", "Rigid-body fit", [
            Option("resolution", "Map resolution", "float", None, "Width of the simulated model map.",
                   minimum=0.5, maximum=50.0, step=0.5, digits=1, optional=True, unit="Å"),
            Option("search", "Search", "radio", "global", "Global: all orientations and positions. Local: "
                   "refine the model as placed.", choices=[("global", "global"), ("local", "local")]),
            Option("angular_step", "Angular step", "float", 30.0, "Rotation sampling of the global search.",
                   minimum=5.0, maximum=45.0, step=5.0, digits=0, unit="°", requires={"search": "global"}),
        ], requires={"mode": "fit"}),
        env_section("conda activate mdatk", [
            Option("python", "Python", "text", "python3", "Interpreter that can import MDATK."),
        ]),
    ]

    def input_ports(self, s):
        if s.get("mode") == "mutate":
            return [self.inputs[0]]
        if s.get("mode") == "fit":
            return list(self.inputs)
        return []

    def output_ports(self, s):
        return list(self.outputs) if s.get("mode") == "helix" and s.get("partner_mode") != "none" \
            else [self.outputs[0]]

    def ss_model(self, s) -> Optional[rna.RNAModel]:
        """Secondary structure of a helix duplex: both strands, Watson-Crick and G·U pairs bracketed."""
        if s["mode"] != "helix":
            return None
        try:
            seq, other, _notes = builder.resolve_partner(s["sequence"], self._partner(s))
        except builder.BuildError:
            return None
        if other is None:
            return None
        kinds = builder.pair_types(seq, other)
        first = "".join("." if k == "mismatch" else "(" for k in kinds)
        second = "".join("." if k == "mismatch" else ")" for k in reversed(kinds))
        try:
            return rna.build_model(f"{seq} {other}", f"{first} {second}")
        except ValueError:
            return None

    def output_values(self, s, ctx):
        out = {"pdb": PortValue([ctx.own(self._out_tag(s), "pdb")], "pdb",
                                {"chirality": s["chirality"]} if s["mode"] == "helix" else {})}
        model = self.ss_model(s)
        if model is not None:
            out["secstruct"] = PortValue([ctx.own(self._out_tag(s), "secstruct")], "secstruct",
                                         {"rna_model": model})
        return out

    def resources(self, s):
        # the Arnott A-RNA fiber monomer of the helix builder (cited once mdatk.citations lists it)
        from ..citations import RESOURCES
        return [r for r in ("arnott-arna",) if r in RESOURCES] if s["mode"] == "helix" else []

    def describe(self, s):
        if s["mode"] == "helix":
            seq = (s["sequence"] or "").strip()
            short = seq if len(seq) <= 14 else seq[:12] + "…"
            return f"{s['chirality']}-RNA helix {short or '(no sequence)'}"
        if s["mode"] == "mutate":
            return f"mutate {s['mutations'].strip() or '(no mutations)'}"
        return f"fit into map ({s['search']})"

    @staticmethod
    def _out_tag(s) -> str:
        return {"helix": "helix", "mutate": "mutant", "fit": "fitted"}[s["mode"]]

    def _partner(self, s) -> str:
        return s["partner"] if s["partner_mode"] == "explicit" else s["partner_mode"]

    def check(self, s, ctx):
        out: List[Issue] = []
        if s["mode"] == "helix":
            if not s["sequence"].strip():
                out.append(Issue("error", "enter the sequence", ("sequence",)))
            elif s["partner_mode"] == "explicit" and not s["partner"].strip():
                out.append(Issue("error", "type the partner sequence (or choose complement / none)",
                                 ("partner",)))
            else:
                for level, text in builder.check_helix(s["sequence"], self._partner(s)):
                    out.append(Issue(level, text, ("partner",) if "partner" in text else ("sequence",)))
            chains = [c.strip() for c in s["chains"].split(",")]
            two = s["partner_mode"] != "none"
            if not chains or any(len(c) > 1 for c in chains) or len(chains) > 2:
                out.append(Issue("error", "chain ids: one or two single characters, e.g. A,B", ("chains",)))
            elif two and (len(chains) < 2 or chains[0] == chains[1]):
                out.append(Issue("error", "a duplex needs two different chain ids, e.g. A,B", ("chains",)))
            if s["phosphate5"] and s["scheme"] == "amber":
                out.append(Issue("warning", "the Amber 5' units have no 5'-phosphate; tleap cannot type it",
                                 ("phosphate5", "scheme")))
        elif s["mode"] == "mutate":
            try:
                builder.parse_mutations(s["mutations"])
            except pdbtools.PDBError as exc:
                out.append(Issue("error", str(exc), ("mutations",)))
            path = ctx.existing("pdb")
            if path and not out:
                out += self._check_residues(s, path)
        else:
            if s["resolution"] is None:
                out.append(Issue("error", "enter the map resolution", ("resolution",)))
            if s["search"] == "local":
                out.append(Issue("info", "a local fit needs the model roughly placed in the map already",
                                 ("search",)))
        if not s["python"].strip():
            out.append(Issue("error", "give the Python interpreter", ("python",)))
        return out

    def _check_residues(self, s, path) -> List[Issue]:
        try:
            struct = pdbtools.read_pdb(path)
        except (OSError, pdbtools.PDBError) as exc:
            return [Issue("warning", f"cannot read the model: {exc}", ("input:pdb",))]
        have = {(r.chain, r.resseq, r.icode) for r in struct.residues()}
        missing = [m.where for m in builder.parse_mutations(s["mutations"])
                   if (m.chain, m.resseq, m.icode) not in have]
        if missing:
            return [Issue("error", "residue(s) not in the model: " + ", ".join(missing), ("mutations",))]
        return []

    # ------------------------------------------------------------ files
    def command(self, s, ctx) -> str:
        py = s["python"].strip() or "python3"
        out = ctx.name(self._out_tag(s), "pdb")
        q = shell_quote
        if s["mode"] == "helix":
            words = [py, "-m", "mdatk", "build", "helix", "--seq", q(s["sequence"].strip()),
                     "--partner", q(self._partner(s).strip()), "--chirality", s["chirality"],
                     "--chains", q(s["chains"].replace(" ", ""))]
            two = s["partner_mode"] != "none"
            # one word: argparse would read a leading '-5,1' as an option
            words += [f"--start={s['start1']},{s['start2'] if two else s['start1']}"]
            if s["scheme"]:
                words += ["--scheme", s["scheme"]]
            if not s["terminals"]:
                words.append("--no-terminal-names")
            if s["phosphate5"]:
                words.append("--phosphate5")
            if s["hydrogens"]:
                words.append("--hydrogens")
        elif s["mode"] == "mutate":
            words = [py, "-m", "mdatk", "build", "mutate", q(ctx.inp("pdb")), q(s["mutations"].strip())]
            if not s["compensatory"]:
                words.append("--no-compensatory")
            elif s["wobble"]:
                words.append("--wobble")
        else:
            words = [py, "-m", "mdatk", "build", "fit", q(ctx.inp("pdb")), q(ctx.inp("map")),
                     "--resolution", f"{s['resolution']:g}", "--search", s["search"]]
            if s["search"] == "global":
                words += ["--step", f"{s['angular_step']:g}"]
        return " ".join(words + ["-o", out])

    def generate(self, s, ctx):
        script = ctx.name(self.tag, "sh")
        title = {"helix": "Ideal A-form helix", "mutate": "Base mutations",
                 "fit": "Rigid-body fit into a density map"}[s["mode"]]
        sh = script_header(f"MDATK structure builder: {title}", ctx, s, script,
                           ["MDATK must be importable by the Python below (installed, or its folder on "
                            "PYTHONPATH)."])
        out = ctx.name(self._out_tag(s), "pdb")
        sh += [self.command(s, ctx), require_file(out, "mdatk build"), ""]
        files = [GeneratedFile(script, "\n".join(sh), "run", f"builds {out}")]
        model = self.ss_model(s)
        if model is not None:
            files.append(GeneratedFile(ctx.name(self._out_tag(s), "secstruct"), rna.secstruct_text(model),
                                       "input", "duplex pairs of the helix (dot-bracket, sequence)"))
        return files
