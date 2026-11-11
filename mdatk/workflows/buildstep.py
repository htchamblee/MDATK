"""Planner step of the in-house starting-structure builder (mdatk.builder, mdatk.chimera): an ideal A-form RNA or
B-form DNA helix (D or L), a D/L chimera of such helices joined through strand 1, base mutations with
compensatory partners, or a rigid-body fit into a density map.  The step writes a short bash script that calls
``python3 -m mdatk build ...``."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from .. import builder, chimera, pdbtools
from . import rna
from .base import PortValue, Tool, env_section, require_file, script_header, shell_quote
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

MODES = [("helix", "Ideal helix"), ("chimera", "D/L chimera"), ("mutate", "Mutate bases"),
         ("fit", "Fit into a density map")]
NUCLEIC_ACID_CHOICES = [("rna", "RNA (A-form)"), ("dna", "DNA (B-form)")]
PAIRING_CHOICES = [("wc", "Watson-Crick (cWW), antiparallel A-form"),
                   ("reverse-wc", "reverse Watson-Crick (tWW), parallel"),
                   ("hoogsteen", "Hoogsteen (cWH: A·U, G·C+), parallel"),
                   ("reverse-hoogsteen", "reverse Hoogsteen (tWH), antiparallel"),
                   ("hh", "Hoogsteen/Hoogsteen A·A (tHH, PDB 4JRD), parallel")]
TRIPLEX_CHOICES = [("none", "none (duplex)"), ("hoogsteen", "pyrimidine motif: Hoogsteen, parallel (U·A-U, C+·G-C)"),
                   ("reverse-hoogsteen", "purine motif: reverse Hoogsteen, antiparallel (A·A-U, G·G-C)")]
SCHEME_CHOICES = [("", "default (A/C/G/U, DNA DA …; L: 0A/0C/0G/0U, 0DA …)"),
                  ("amber", "Amber (L: LA …, LDA … + 5'/3')"),
                  ("ccd", "CCD for Coot / Phenix (L: 0A … + 5'/3', 0DA …)"), ("rosetta", "Rosetta"),
                  ("pdb", "PDB deposition")]
FIBERS = {"rna": "arnott-arna", "dna": "arnott-bdna"}  # mdatk.citations: the fiber model of each helix
DESIGN_EXAMPLE = "D-RNA:GGCAUC 3'-5' L-RNA:GAUGCC"
DESIGN_HELP = ("Segments joined by links, separated by spaces: HAND[-NA]:SEQUENCE[/PARTNER] LINK "
               "HAND[-NA]:SEQUENCE ... HAND is D or L, NA RNA or DNA (default below), SEQUENCE strand 1 5'->3', "
               "PARTNER auto (the complement, a duplex: the default), none (a single strand) or a sequence 5'->3'. "
               "A link is 3'-5' or 5'-3' (a phosphodiester) or 3'-3' or 5'-5' (a bridging phosphate); its first "
               f"end is the free end of the strand built so far. {DESIGN_EXAMPLE} is 5'-D-RNA(6)-p-L-RNA(6)-3'.")


@register
class StructureBuilderTool(Tool):
    name = "structure-builder"
    label = "Structure builder (MDATK)"
    category = "Structure building"
    tag = "build"
    description = ("Starting structures without external programs: ideal A-form RNA and B-form DNA helices (D or "
                   "L), D/L chimeras of such helices, base mutations keeping the glycosidic frame (with "
                   "compensatory partners), and rigid-body fitting into a density map.")
    inputs = [Port("pdb", "Model (PDB)", ("pdb",), tooltip="The structure to mutate or to fit."),
              Port("map", "Density map", ("map",), tooltip="MRC / CCP4 map for the rigid-body fit.")]
    outputs = [Port("pdb", "Structure (PDB)", ("pdb",)),
               Port("secstruct", "Secondary structure", ("secstruct",),
                    tooltip="Duplex pairs of the ideal helix (Watson-Crick and G·U, DNA G·T, antiparallel only), "
                            "or of a chimera with linked partner strands of one nucleic acid, e.g. for helix "
                            "restraints.")]
    sections = [
        Section("mode", "Mode", [
            Option("mode", "Build", "radio", "helix", "Ideal helix: a new A-form RNA or B-form DNA duplex or single "
                   "strand. D/L chimera: ideal helices of either hand joined through strand 1. Mutate: swap bases "
                   "of the connected PDB. Fit: place the connected PDB in the map.", choices=MODES),
        ]),
        Section("helix", "Ideal helix", [
            Option("nucleic_acid", "Nucleic acid", "radio", "rna", "RNA: the Arnott A-form, with the pairings and "
                   "triplexes below. DNA: the Arnott B-form, a Watson-Crick duplex or a single strand.",
                   choices=NUCLEIC_ACID_CHOICES),
            Option("sequence", "Sequence (5'->3')", "text", "", "Strand 1; RNA reads T as U, DNA reads U as T.",
                   placeholder="GGCAUCGAUGCC"),
            Option("partner_mode", "Partner strand", "radio", "auto",
                   "Complement (the canonical partners of the pairing: Watson-Crick for wc), none (single "
                   "strand) or a sequence typed below.",
                   choices=[("auto", "complement"), ("none", "none"), ("explicit", "explicit")]),
            Option("partner", "Partner (5'->3')", "text", "", "Same length as strand 1; mismatches and G·U (DNA: "
                   "G·T) wobbles are allowed (reported).", requires={"partner_mode": "explicit"}),
            Option("pairing", "Pairing", "choice", "wc", "Base-pair geometry of the RNA duplex: Watson-Crick "
                   "(antiparallel A-form, default), reverse Watson-Crick / Hoogsteen / Hoogsteen-Hoogsteen "
                   "(parallel strands) or reverse Hoogsteen (antiparallel). For the Hoogsteen pairings strand 1 "
                   "is the purine strand.", choices=PAIRING_CHOICES,
                   requires={"partner_mode": ("auto", "explicit"), "nucleic_acid": "rna"}),
            Option("triplex", "Triplex", "choice", "none", "A third strand in the major groove on the Hoogsteen "
                   "edge of the strand-1 purines (RNA Watson-Crick duplex only).", choices=TRIPLEX_CHOICES,
                   requires={"partner_mode": ("auto", "explicit"), "pairing": "wc", "nucleic_acid": "rna"}),
            Option("third_mode", "Third strand", "radio", "auto", "The canonical motif bases (U / C+ or A / G "
                   "on A / G) or a sequence typed below.", choices=[("auto", "motif"), ("explicit", "explicit")],
                   requires={"partner_mode": ("auto", "explicit"), "pairing": "wc",
                             "triplex": ("hoogsteen", "reverse-hoogsteen"), "nucleic_acid": "rna"}),
            Option("third", "Third strand (5'->3')", "text", "", "Same length as strand 1; mismatches are "
                   "allowed (reported).", requires={"third_mode": "explicit", "partner_mode": ("auto", "explicit"),
                                                    "pairing": "wc", "triplex": ("hoogsteen", "reverse-hoogsteen"),
                                                    "nucleic_acid": "rna"}),
            Option("chirality", "Chirality", "radio", "D", "D: right-handed. L: the mirror image (left-handed, "
                   "L-nucleotides).", choices=[("D", "D (right-handed)"), ("L", "L (mirror image)")]),
            Option("chains", "Chain ids", "text", "A,B", "Strand 1, strand 2 (and 3 of a triplex; default C)."),
            Option("start1", "First residue (strand 1)", "int", 1, "", minimum=-999, maximum=9999),
            Option("start2", "First residue (strand 2)", "int", 1, "", minimum=-999, maximum=9999,
                   requires={"partner_mode": ("auto", "explicit")}),
            Option("start3", "First residue (strand 3)", "int", 1, "", minimum=-999, maximum=9999,
                   requires={"partner_mode": ("auto", "explicit"), "pairing": "wc",
                             "triplex": ("hoogsteen", "reverse-hoogsteen"), "nucleic_acid": "rna"}),
        ], requires={"mode": "helix"}),
        Section("chimera", "D/L chimera", [
            Option("design", "Design", "text", "", DESIGN_HELP, placeholder=DESIGN_EXAMPLE),
            Option("chimera_na", "Default nucleic acid", "radio", "rna", "Of the segments that name none "
                   "(D:GGCAUC); D-RNA:... and L-DNA:... name their own.", choices=NUCLEIC_ACID_CHOICES),
            Option("chimera_chains", "Chain ids", "text", "", "Optional, comma separated, one per chain: the "
                   "strand-1 chains in design order (strand 1 is split at 3'-3' / 5'-5' links), then the partner "
                   "strands in segment order. Empty: A, B, C ...", placeholder="A,B,C"),
            Option("chimera_link_partners", "Link the partner strands", "bool", False, "Join the partner strands "
                   "too, by a phosphodiester at every junction: one continuous partner chain (two chains in all) "
                   "instead of one nicked chain per segment. Every link 3'-5' or 5'-3', every segment paired."),
        ], requires={"mode": "chimera"}),
        # residue names and atoms of both builds, shown once: the helix options of before, keys and all, so that
        # projects keep their values
        Section("names", "Names & atoms", [
            Option("scheme", "Residue names", "choice", "", "Name scheme of mdatk.pdbtools.",
                   choices=SCHEME_CHOICES),
            Option("terminals", "5'/3' terminal names", "bool", True, "For the schemes with terminal "
                   "variants (Amber, CCD)."),
            Option("phosphate5", "5'-phosphate", "bool", False, "Keep P/OP1/OP2/OP3 on the 5' ends (the "
                   "Amber 5' units have none)."),
            Option("hydrogens", "Hydrogens", "bool", False, "Add all hydrogens (default: heavy atoms only)."),
        ], requires={"mode": ("helix", "chimera")}),
        Section("mutate", "Mutations", [
            Option("mutations", "Mutations", "text", "", "Chain:residue + new base, comma separated.",
                   placeholder="A:5G, B:20C"),
            Option("compensatory", "Compensatory partner mutations", "bool", True,
                   "Change the Watson-Crick / G·U partner of every mutated base to the complement unless "
                   "it is mutated explicitly."),
            Option("wobble", "Keep G·U pairs", "bool", False, "A G or U facing U / G keeps its partner (wobble) "
                   "instead of the Watson-Crick complement (DNA: G·T).", requires={"compensatory": True}),
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
            Option("python", "Python", "text", "python3", "Interpreter that can import MDATK. From the standalone "
                   "AppImage / Windows build the script calls that program itself instead."),
        ]),
    ]

    def input_ports(self, s):
        if s.get("mode") == "mutate":
            return [self.inputs[0]]
        if s.get("mode") == "fit":
            return list(self.inputs)
        return []

    def output_ports(self, s):
        return list(self.outputs) if self._duplex(s) or self._linked_chimera(s) else [self.outputs[0]]

    @staticmethod
    def _na(s) -> str:
        """Nucleic acid of the helix: 'rna' (also of projects written before DNA helices) or 'dna'."""
        return "dna" if s.get("nucleic_acid") == "dna" else "rna"

    @staticmethod
    def _pairing(s) -> str:
        """The pairing built: the chosen one for an RNA duplex; wc for a single strand and for DNA (a Watson-Crick
        B-form duplex: a pairing left over from RNA is ignored)."""
        if s.get("partner_mode") == "none" or s.get("nucleic_acid") == "dna":
            return "wc"
        return s.get("pairing", "wc")

    @staticmethod
    def _triplex(s) -> Optional[str]:
        """The triplex motif built (RNA Watson-Crick duplexes only; None for DNA whatever the triplex option)."""
        return s["triplex"] if s.get("nucleic_acid") != "dna" and s.get("pairing", "wc") == "wc" \
            and s.get("triplex") not in (None, "", "none") and s.get("partner_mode") != "none" else None

    @classmethod
    def _duplex(cls, s) -> bool:
        """Whether the build is a Watson-Crick helix duplex, RNA or DNA: the builds with a secondary structure."""
        return s.get("mode") == "helix" and s.get("partner_mode") != "none" and cls._pairing(s) == "wc"

    def ss_model(self, s) -> Optional[rna.RNAModel]:
        """Secondary structure of a Watson-Crick helix duplex: both strands, Watson-Crick and G·U pairs
        bracketed (a triplex: the third strand unpaired).  None for the other pairings: the helix
        restraints assume antiparallel Watson-Crick H-bonds.

        A B-DNA duplex has one as well (G·T wobbles bracketed): mdatk.workflows.rna reads its T as U, and the
        steps that restrain the pairs (Amber, NAMD, Phenix) find the hydrogen-bond atoms of U in T (N3, O4, O2)
        and take A·T for the A·U class; its .secstruct file keeps the T (:meth:`secstruct_text`).

        A chimera with linked partner strands of one nucleic acid has one as well: strand 1 and the continuous
        partner chain (:func:`mdatk.chimera.linked_strands`), the hand of the residues aside."""
        third = ""
        if self._linked_chimera(s):
            try:
                seq, other, kinds = chimera.linked_strands(self._design(s), s.get("chimera_na") or "rna")
            except builder.BuildError:
                return None
            other_kinds = kinds
        elif not self._duplex(s):
            return None
        else:
            na = self._na(s)
            try:
                seq, other, _notes = builder.resolve_partner(s["sequence"], self._partner(s), na=na)
                third = builder.resolve_third(seq, self._third(s), self._triplex(s))[0] if self._triplex(s) else ""
            except builder.BuildError:
                return None
            other_kinds = None
        if other is None:
            return None
        kinds = other_kinds if other_kinds is not None else builder.pair_types(seq, other, na)
        first = "".join("." if k == "mismatch" else "(" for k in kinds)
        second = "".join("." if k == "mismatch" else ")" for k in reversed(kinds))
        seqs, dots = f"{seq} {other}", f"{first} {second}"
        if third:
            seqs, dots = f"{seqs} {third}", f"{dots} {'.' * len(third)}"
        try:
            return rna.build_model(seqs, dots)
        except ValueError:
            return None

    def secstruct_text(self, s, model: rna.RNAModel) -> str:
        """The .secstruct file of the duplex (:func:`mdatk.workflows.rna.secstruct_text`): dot-bracket, then the
        sequence for reference, of a DNA duplex with its T (read as U again by whatever reads the file)."""
        text = rna.secstruct_text(model)
        linked = self._linked_chimera(s)
        if (linked or self._na(s)) != "dna":
            return text
        dots, letters = text.split("\n")[:2]
        return f"{dots}\n{letters.replace('u', 't')}\n"

    @classmethod
    def _linked_chimera(cls, s) -> Optional[str]:
        """The nucleic acid ('rna' or 'dna') of a chimera with linked partner strands whose segments are all of
        it (it has a secondary structure, :meth:`ss_model`); None otherwise, or while the design does not parse
        or its partners cannot be linked."""
        if s.get("mode") != "chimera" or not s.get("chimera_link_partners"):
            return None
        design = cls._chimera(s)
        if design is None:
            return None
        segments, links = design
        nas = {seg.na for seg in segments}
        if len(nas) != 1 or any(link in chimera.BRIDGES for link in links) or \
                any(seg.partner is None for seg in segments):
            return None
        return nas.pop()

    @staticmethod
    def _design(s) -> str:
        """The chimera design on one line (its tokens are separated by whitespace: nothing it says changes)."""
        return " ".join((s.get("design") or "").split())

    @classmethod
    def _chimera(cls, s) -> Optional[Tuple[List[chimera.Segment], List[str]]]:
        """(segments, links) of the chimera design; None while it does not parse."""
        try:
            segments, links, _notes = chimera.parse_design(cls._design(s), s.get("chimera_na") or "rna")
        except builder.BuildError:
            return None
        return segments, links

    def chimera_hand(self, s) -> Optional[str]:
        """The hand of the chimera for the steps it feeds (port data 'chirality'): 'D' or 'L' when every segment
        has it, 'mixed' otherwise; None while the design does not parse."""
        design = self._chimera(s)
        if design is None:
            return None
        hands = {seg.chirality for seg in design[0]}
        return hands.pop() if len(hands) == 1 else "mixed"

    def model_data(self, s) -> dict:
        """What the steps fed by the PDB port know of the model before it exists (its port data): 'chirality'
        (D or L, of every residue; 'mixed' for a chimera of both hands), 'l_rna' (whether it has L-RNA residues:
        the L residues that need the L-RNA restraint dictionaries and Amber's L-RNA library; L-DNA needs neither)
        and 'build' ('helix' or 'chimera').  Empty for the other modes and while a chimera design does not
        parse."""
        if s["mode"] == "helix":
            dna = self._na(s) == "dna"
            return {"chirality": s["chirality"], "l_rna": s["chirality"] == "L" and not dna,
                    "dna": dna, "l_dna": s["chirality"] == "L" and dna, "rna": not dna, "build": "helix"}
        design = self._chimera(s) if s["mode"] == "chimera" else None
        if design is None:
            return {}
        segs = design[0]
        return {"chirality": self.chimera_hand(s),
                "l_rna": any(seg.chirality == "L" and seg.na == "rna" for seg in segs),
                "dna": any(seg.na == "dna" for seg in segs),
                "l_dna": any(seg.chirality == "L" and seg.na == "dna" for seg in segs),
                "rna": any(seg.na == "rna" for seg in segs), "build": "chimera"}

    def output_values(self, s, ctx):
        data = self.model_data(s)
        out = {"pdb": PortValue([ctx.own(self._out_tag(s), "pdb")], "pdb", data)}
        model = self.ss_model(s)
        if model is not None:
            out["secstruct"] = PortValue([ctx.own(self._out_tag(s), "secstruct")], "secstruct",
                                         {"rna_model": model})
        return out

    def resources(self, s):
        # the Arnott fiber monomers of the helices built (A-RNA, B-DNA; cited once mdatk.citations lists them) and
        # PDB 4JRD, the source of the other RNA pairings
        from ..citations import RESOURCES
        if s["mode"] == "helix":
            if self._na(s) == "dna":
                used = [FIBERS["dna"]]
            else:
                used = ["arnott-arna"] + (["pdb-4jrd"] if self._pairing(s) != "wc" or self._triplex(s) else [])
        elif s["mode"] == "chimera":
            design = self._chimera(s)
            used = [FIBERS[na] for na in chimera.nucleic_acids(design)] if design else []
        else:
            return []
        return [r for r in used if r in RESOURCES]

    def describe(self, s):
        if s["mode"] == "helix":
            seq = (s["sequence"] or "").strip()
            short = seq if len(seq) <= 14 else seq[:12] + "…"
            if self._na(s) == "dna":  # a Watson-Crick duplex or a single strand
                return f"{s['chirality']}-DNA helix {short or '(no sequence)'}"
            kind = "" if s["pairing"] == "wc" or s["partner_mode"] == "none" else f" ({s['pairing']})"
            if self._triplex(s):
                kind = f" ({self._triplex(s)} triplex)"
            return f"{s['chirality']}-RNA helix{kind} {short or '(no sequence)'}"
        if s["mode"] == "chimera":
            design = self._chimera(s)
            if design is not None:
                return f"chimera {chimera.describe(design)}"
            text = self._design(s)
            return f"chimera {text if len(text) <= 24 else text[:22] + '…'}" if text else "chimera (no design)"
        if s["mode"] == "mutate":
            return f"mutate {s['mutations'].strip() or '(no mutations)'}"
        return f"fit into map ({s['search']})"

    @staticmethod
    def _out_tag(s) -> str:
        return {"helix": "helix", "chimera": "chimera", "mutate": "mutant", "fit": "fitted"}[s["mode"]]

    def _partner(self, s) -> str:
        return s["partner"] if s["partner_mode"] == "explicit" else s["partner_mode"]

    @staticmethod
    def _third(s) -> str:
        return s["third"] if s["third_mode"] == "explicit" else "auto"

    def check(self, s, ctx):
        out: List[Issue] = []
        if s["mode"] == "helix":
            if not s["sequence"].strip():
                out.append(Issue("error", "enter the sequence", ("sequence",)))
            elif s["partner_mode"] == "explicit" and not s["partner"].strip():
                out.append(Issue("error", "type the partner sequence (or choose complement / none)",
                                 ("partner",)))
            elif self._triplex(s) and s["third_mode"] == "explicit" and not s["third"].strip():
                out.append(Issue("error", "type the third-strand sequence (or choose the motif bases)",
                                 ("third",)))
            else:
                for level, text in builder.check_helix(s["sequence"], self._partner(s), self._pairing(s),
                                                       self._triplex(s), self._third(s), na=self._na(s)):
                    keys = ("third",) if "third strand" in text or text.startswith(("third", "triplex")) else (
                        ("partner",) if "partner" in text else ("sequence",))
                    out.append(Issue(level, text, keys))
            if s["partner_mode"] != "none" and self._pairing(s) != "wc":
                out.append(Issue("info", "no secondary-structure output: the helix restraints (dot-bracket) "
                                         "assume antiparallel Watson-Crick pairs", ("pairing",)))
            chains = [c.strip() for c in s["chains"].split(",")]
            two = s["partner_mode"] != "none"
            most = 3 if self._triplex(s) else 2
            if not chains or any(len(c) > 1 for c in chains) or len(chains) > most:
                out.append(Issue("error", "chain ids: one or two single characters, e.g. A,B" if most == 2 else
                                 "chain ids: two or three single characters, e.g. A,B,C", ("chains",)))
            elif two and (len(chains) < 2 or chains[0] == chains[1]):
                out.append(Issue("error", "a duplex needs two different chain ids, e.g. A,B", ("chains",)))
            elif most == 3 and len(chains) == 3 and chains[2] in chains[:2]:
                out.append(Issue("error", "a triplex needs three different chain ids, e.g. A,B,C", ("chains",)))
            out += self._phosphate5_issues(s)
        elif s["mode"] == "chimera":
            out += self._check_chimera(s) + self._phosphate5_issues(s)
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

    @staticmethod
    def _phosphate5_issues(s) -> List[Issue]:
        if s["phosphate5"] and s["scheme"] == "amber":
            return [Issue("warning", "the Amber 5' units have no 5'-phosphate; tleap cannot type it",
                          ("phosphate5", "scheme"))]
        return []

    @staticmethod
    def _chimera_chains(s) -> str:
        """The chain ids of a chimera as the command passes them ('' for A, B, C ...): whitespace dropped."""
        return "".join(s["chimera_chains"].split())

    def _check_chimera(self, s) -> List[Issue]:
        """The builder's checks of the design (on the design option), then of the chain ids against the chains
        the design writes: :func:`mdatk.chimera.check_chimera`, the rule of 'mdatk build chimera' and of the
        RNAdo window."""
        text = self._design(s)
        if not text:
            return [Issue("error", f"enter the design, e.g. {DESIGN_EXAMPLE}", ("design",))]
        args = (text, s["chimera_na"], s["hydrogens"], s["phosphate5"])
        linked = bool(s.get("chimera_link_partners"))
        out = [Issue(level, message, ("design",)) for level, message in chimera.check_chimera(*args)]
        if linked and not any(i.level == "error" for i in out):
            # the design reads: an error now is the linked partners' (a bridge, a single strand)
            out = [Issue(level, message, ("chimera_link_partners",) if message.startswith(("link partners",
                         "partner strands linked")) else ("design",))
                   for level, message in chimera.check_chimera(*args, link_partners=True)]
        chains = self._chimera_chains(s)
        if chains and not any(i.level == "error" for i in out):
            # the design reads: an error now is the chain ids' (checked last, after the design)
            out += [Issue(level, message, ("chimera_chains",))
                    for level, message in chimera.check_chimera(*args, chains=chains, link_partners=linked)
                    if level == "error"]
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
    @staticmethod
    def mdatk(s) -> List[str]:
        """How the script starts MDATK: the Linux AppImage that wrote it, or ``<python> -m mdatk``.  The
        Windows build writes bash scripts that run on Linux / WSL / a cluster, where its own .exe path
        means nothing, so it keeps ``python3 -m mdatk`` (MDATK installed there)."""
        py = s["python"].strip() or "python3"
        from .. import frozen
        appimage = os.environ.get("APPIMAGE", "")
        if frozen.is_frozen() and py == "python3" and appimage and os.name != "nt":
            return [shell_quote(appimage)]
        return [py, "-m", "mdatk"]

    @staticmethod
    def _name_flags(s) -> List[str]:
        """Residue names and atoms of a helix or a chimera: scheme, terminal names, 5'-phosphate, hydrogens."""
        words = ["--scheme", s["scheme"]] if s["scheme"] else []
        if not s["terminals"]:
            words.append("--no-terminal-names")
        if s["phosphate5"]:
            words.append("--phosphate5")
        if s["hydrogens"]:
            words.append("--hydrogens")
        return words

    @staticmethod
    def _option(flag: str, value: str) -> List[str]:
        """``flag value`` as two words, or one word ``flag=value`` for a value that starts with '-', which
        argparse would read as an option (a chain id '-'; as --start below)."""
        return [f"{flag}={shell_quote(value)}"] if value.startswith("-") else [flag, shell_quote(value)]

    def command(self, s, ctx) -> str:
        out = ctx.name(self._out_tag(s), "pdb")
        q = shell_quote
        if s["mode"] == "helix":
            words = self.mdatk(s) + ["build", "helix", "--seq", q(s["sequence"].strip()),
                                     "--partner", q(self._partner(s).strip())]
            if self._na(s) == "dna":  # the flag only for DNA: RNA command lines stay as they were
                words += ["--na", "dna"]
            words += ["--chirality", s["chirality"]] + self._option("--chains", s["chains"].replace(" ", ""))
            if self._pairing(s) != "wc":
                words += ["--pairing", self._pairing(s)]
            if self._triplex(s):
                words += ["--triplex", self._triplex(s), "--third-strand", q(self._third(s).strip())]
            two = s["partner_mode"] != "none"
            # one word: argparse would read a leading '-5,1' as an option
            words += [f"--start={s['start1']},{s['start2'] if two else s['start1']}"
                      + (f",{s['start3']}" if self._triplex(s) else "")]
            words += self._name_flags(s)
        elif s["mode"] == "chimera":
            # one quoted word: the links carry apostrophes (3'-5'), the segments are separated by spaces
            words = self.mdatk(s) + ["build", "chimera", "--design", q(self._design(s))]
            if s["chimera_na"] == "dna":
                words += ["--na", "dna"]
            chains = self._chimera_chains(s)
            if chains:
                words += self._option("--chains", chains)
            if s.get("chimera_link_partners"):  # the flag only when on: other command lines stay as they were
                words.append("--link-partners")
            words += self._name_flags(s)
        elif s["mode"] == "mutate":
            words = self.mdatk(s) + ["build", "mutate", q(ctx.inp("pdb")), q(s["mutations"].strip())]
            if not s["compensatory"]:
                words.append("--no-compensatory")
            elif s["wobble"]:
                words.append("--wobble")
        else:
            words = self.mdatk(s) + ["build", "fit", q(ctx.inp("pdb")), q(ctx.inp("map")),
                     "--resolution", f"{s['resolution']:g}", "--search", s["search"]]
            if s["search"] == "global":
                words += ["--step", f"{s['angular_step']:g}"]
        return " ".join(words + ["-o", out])

    def generate(self, s, ctx):
        script = ctx.name(self.tag, "sh")
        title = {"helix": "Ideal A-form helix", "chimera": "Chimera", "mutate": "Base mutations",
                 "fit": "Rigid-body fit into a density map"}[s["mode"]]
        design = self._chimera(s) if s["mode"] == "chimera" else None
        if s["mode"] == "helix" and self._na(s) == "dna":
            title = f"Ideal {builder.helix_kind(na='dna')} DNA helix"
        elif s["mode"] == "helix" and (self._pairing(s) != "wc" or self._triplex(s)):
            title = f"Ideal helix ({builder.helix_kind(self._pairing(s), self._triplex(s))})"
        elif design is not None:
            title = f"Chimera {chimera.describe(design)}"  # as 'mdatk build chimera' logs it
        sh = script_header(f"MDATK structure builder: {title}", ctx, s, script,
                           ["MDATK must be importable by the Python below (installed, or its folder on "
                            "PYTHONPATH)."])
        out = ctx.name(self._out_tag(s), "pdb")
        sh += [self.command(s, ctx), require_file(out, "mdatk build"), ""]
        files = [GeneratedFile(script, "\n".join(sh), "run", f"builds {out}")]
        model = self.ss_model(s)
        if model is not None:
            files.append(GeneratedFile(ctx.name(self._out_tag(s), "secstruct"), self.secstruct_text(s, model),
                                       "input", "duplex pairs of the helix (dot-bracket, sequence)"))
        return files
