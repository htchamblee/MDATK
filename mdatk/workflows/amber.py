"""Amber generators: system build (tleap / ParmEd / L-RNA renaming), helix
distance restraints and MD stages (pmemd, pmemd.cuda, pmemd.MPI).

The defaults reproduce a working RNA protocol: OL3 + CP force field, NBfix19 and
0BPh Lennard-Jones pair modifications applied with ParmEd, SPC/E water in a
15 Å truncated octahedron, helix-preserving DISANG restraints, and the stage
sequence min1 (solute restrained) -> min2 -> heat (Langevin, 0 -> 300 K) ->
NPT equilibration -> NPT equilibration without positional restraints ->
production.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import PROJECT_ABBREV, __version__
from . import mg_params, rna
from .base import (PortValue, StepContext, Tool, env_section, extra_lines, fortran_float,
                   require_file, script_header)
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section
from .scripts import embed
from .scripts import helix_dist as _helix_dist
from .scripts import prep_pdb as _prep_pdb

# ---------------------------------------------------------------------------
# force fields, water models, ions
# ---------------------------------------------------------------------------

# id -> (label, leaprc, short ASCII name used in generated files)
FORCE_FIELDS: Dict[str, Tuple[str, str, str]] = {
    "ol3cp": ("OL3 + CP (bsc0 χOL3, Steinbrecher phosphate vdW)", "leaprc.RNA.OL3", "OL3+CP"),
    "ol3": ("OL3 (bsc0 χOL3)", "leaprc.RNA.OL3", "OL3"),
    "ljbb": ("LJbb (Bergonzo & Cheatham 2015)", "leaprc.RNA.LJbb", "LJbb"),
    "roc": ("ROC-RNA (Aytenfisu et al. 2017)", "leaprc.RNA.ROC", "ROC-RNA"),
    "yil": ("YIL (Yildirim et al. 2010)", "leaprc.RNA.YIL", "YIL"),
    "shaw": ("DESRES RNA (Tan et al. 2018)", "leaprc.RNA.Shaw", "DESRES-RNA"),
}

# id -> (label, leaprc, solvent box, ParmEd add12_6_4 water model, 12-6-4 ion frcmod)
# (box and frcmod names checked against AmberTools 26 dat/leap)
WATERS: Dict[str, Tuple[str, str, str, str, str]] = {
    "spce": ("SPC/E", "leaprc.water.spce", "SPCBOX", "SPCE", "frcmod.ions234lm_1264_spce"),
    "tip3p": ("TIP3P", "leaprc.water.tip3p", "TIP3PBOX", "TIP3P", "frcmod.ions234lm_1264_tip3p"),
    "opc": ("OPC", "leaprc.water.opc", "OPCBOX", "OPC", "frcmod.ionslm_1264_opc"),
    "opc3": ("OPC3", "leaprc.water.opc3", "OPC3BOX", "OPC3", "frcmod.ionslm_1264_opc3"),
    "tip4pew": ("TIP4P-Ew", "leaprc.water.tip4pew", "TIP4PEWBOX", "TIP4PEW", "frcmod.ions234lm_1264_tip4pew"),
    "fb3": ("TIP3P-FB", "leaprc.water.fb3", "FB3BOX", "FB3", "frcmod.ionslm_1264_fb3"),
    "tip4pd": ("TIP4P-D", "leaprc.water.tip4pd", "TIP4PDBOX", "", ""),
    "none": ("none (no solvent box)", "", "", "", ""),
}
FOUR_SITE_WATERS = ("opc", "tip4pew", "tip4pd")

SOLVATED = tuple(k for k in WATERS if k != "none")
DEFAULT_STRIP = ":WAT,HOH,Na+,Cl-,K+,MG"
SOLVENT_MASK = "!:WAT,HOH,Na+,Cl-,K+,MG"

NBFIX19_LINES = [
    "#",
    "# Off-diagonal Lennard-Jones parameters replacing the external gHBfix19 H-bond correction",
    "# potential (Kuehrova & Mlynsky et al., JCTC 2019) with the NBfix19 modification;",
    "# for details, see Mlynsky V., Kuehrova P., Bussi G., Otyepka M., Sponer J., and Banas P., 2026",
    "#",
    "# IMPORTANT: this assumes that the topology was built with the OL3 RNA force field (Zgarbova et al.,",
    "# JCTC 2011) further adjusted by the phosphate oxygen van der Waals correction (CP) by Steinbrecher",
    "# et al., JCTC 2012; other force fields may have different atom types and need different atom masks",
    "#",
    "# NBfix modification of -NH...N- interactions according to the gHBfix19 potential",
    "# all A_N6(H61/H62), C_N4(H41/H42), G_N2(H21/H22), G_N1(H1), and U_N3(H3) hydrogens share same and",
    "# unique atom type H",
    "# A(N1), A(N3), A(N7), C(N3), G(N3), and G(N7) share same NB and NC atom types, but these are also share",
    "# with, e.g., N, NA, N2 .., so we need to split it",
    "addLJType @%NB,NC",
    "changeLJPair @%NB,NC @%H 2.1 0.63",
    "#",
    "# NBfix modification of -OH...bO/nbO- interactions according to the gHBfix19 potential",
    "# all 2'-OH hydrogens (i.e., HO2', HO3', and HO5') share same and unique atom type HO",
    "# all bridging (bO; O5' and O3') and nonbridging (nbO; O1P and O2P) phosphate oxygens share same and",
    "# unique atom types OR and OP",
    "changeLJPair @%HO @%O2 2.73 0.01",
    "changeLJPair @%HO @%OS 3.03 0.01",
]
BPH0_LINES = [
    "#",
    "# Additional parameters from 0BPh",
    "changeLJPair @%H5 @%OS 2.8808 0.0505",
    "changeLJPair @%H4 @%OS 2.9308 0.0505",
]
LRNA_NOTE = [
    "#",
    "# L-RNA pairs (e.g. changeLJPair @%HO @%LD 3.03 0.01) are not included; add them under",
    "# 'Extra ParmEd commands' when the topology contains L-RNA residues.",
]

# ---------------------------------------------------------------------------
# shared option builders
# ---------------------------------------------------------------------------


def restraint_options() -> List[Option]:
    return [
        Option("sequence", "Sequence", "multiline", "",
               "RNA sequence in topology order; chains separated by spaces, '&' or new lines "
               "(optionally 'A: GGCAC...'). Not needed when a secondary-structure file is connected.",
               placeholder="GGGCAC... GUGCCC..."),
        Option("dotbracket", "Secondary structure", "multiline", "",
               "Dot-bracket notation for the whole sequence; '()', '[]', '{}' and '<>' pairs are "
               "restrained. Strand separators (spaces, '&') are ignored.",
               placeholder="((((((....)))))) ..."),
        Option("res_offset", "Residue offset", "int", 0,
               "Topology residue number = position in the sequence + offset (use when the "
               "topology has extra residues before the RNA).", minimum=-100000, maximum=100000),
        Option("wobble", "Restrain G·U wobble pairs", "bool", False,
               "Also restrain G·U pairs (G O6–U N3, G N1–U O2). Off: only Watson–Crick G–C and A–U "
               "pairs are restrained."),
        Option("r1", "r1", "float", 2.3, "Lower distance where the linear restraint wall starts (Å).",
               minimum=0.0, maximum=50.0, step=0.1, digits=2, unit="Å"),
        Option("r2", "r2", "float", 2.7, "Start of the flat bottom (Å).", minimum=0.0, maximum=50.0,
               step=0.1, digits=2, unit="Å"),
        Option("r3", "r3", "float", 3.2, "End of the flat bottom (Å).", minimum=0.0, maximum=50.0,
               step=0.1, digits=2, unit="Å"),
        Option("r4", "r4", "float", 3.7, "Upper distance where the restraint turns linear (Å).",
               minimum=0.0, maximum=50.0, step=0.1, digits=2, unit="Å"),
        Option("rk2", "rk2", "float", 10.0, "Force constant below r2 (kcal/mol/Å²).", minimum=0.0,
               maximum=1000.0, step=0.5, digits=2),
        Option("rk3", "rk3", "float", 10.0, "Force constant above r3 (kcal/mol/Å²).", minimum=0.0,
               maximum=1000.0, step=0.5, digits=2),
    ]


def restraint_window(s: Dict[str, Any]) -> str:
    return (f"r1={fortran_float(s['r1'])}, r2={fortran_float(s['r2'])}, r3={fortran_float(s['r3'])}, "
            f"r4={fortran_float(s['r4'])}, rk2={fortran_float(s['rk2'])}, rk3={fortran_float(s['rk3'])},")


def secondary_structure(s: Dict[str, Any], ctx: StepContext,
                        port: str = "secstruct") -> Tuple[Optional[rna.RNAModel], str, List[Issue]]:
    """Secondary structure for restraints: a connected / chosen file wins over typed fields.

    Returns (model or None, source description, issues)."""
    issues: List[Issue] = []
    value = ctx.value(port)
    if value is not None and value.paths:
        model = value.data.get("rna_model")
        if model is not None:
            return model, f"{value.source_label or 'connected file'}", issues
        path = ctx.existing(port)
        if not path:
            issues.append(Issue("error", "the secondary-structure file must exist when the files are "
                                         "written (or connect a step that knows it)", (f"input:{port}",)))
            return None, "", issues
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith(">")]
            seq_text = s.get("sequence", "")
            if len(lines) >= 2 and not seq_text.strip():
                seq_text = lines[1]
            model = rna.build_model(seq_text, lines[0] if lines else "")
        except (OSError, ValueError) as exc:
            issues.append(Issue("error", f"cannot read {os.path.basename(path)}: {exc}", (f"input:{port}",)))
            return None, "", issues
        return model, os.path.basename(path), issues
    if not (s.get("dotbracket") or "").strip():
        return None, "", issues
    try:
        model = rna.build_model(s.get("sequence", ""), s.get("dotbracket", ""))
    except ValueError as exc:
        issues.append(Issue("error", str(exc), ("sequence", "dotbracket")))
        return None, "", issues
    if not model.chains:
        issues.append(Issue("error", "enter the sequence for the secondary structure", ("sequence",)))
        return None, "", issues
    return model, "typed secondary structure", issues


def restraint_pairs(model: rna.RNAModel, s: Dict[str, Any]) -> Tuple[List[Tuple[int, int, str]], List[Issue]]:
    keep, skip = rna.classify_pairs(model, s.get("wobble", False))
    issues: List[Issue] = []
    if skip:
        kinds = sorted({k for _i, _j, k in skip})
        wob = [p for p in skip if p[2] in rna.WOBBLE]
        msg = f"{len(skip)} base pair(s) not restrained ({', '.join(kinds)})"
        if wob and not s.get("wobble"):
            msg += "; tick 'Restrain G·U wobble pairs' to include G·U"
        issues.append(Issue("info", msg, ("wobble",) if wob else ("dotbracket",)))
    if not keep:
        issues.append(Issue("warning", "the secondary structure contains no restrainable base pairs",
                            ("dotbracket",)))
    r = [s["r1"], s["r2"], s["r3"], s["r4"]]
    if any(r[k] > r[k + 1] for k in range(3)):
        issues.append(Issue("error", "restraint distances must satisfy r1 ≤ r2 ≤ r3 ≤ r4",
                            ("r1", "r2", "r3", "r4")))
    return keep, issues


def helix_script(pairs, s: Dict[str, Any], source: str) -> str:
    bonds = {k: v for k, v in rna.HBOND_ATOMS.items() if any(p[2] == k for p in pairs)}
    doc = (f"Helix-preserving distance restraints (Amber DISANG) - written by {PROJECT_ABBREV} "
           f"{__version__}.\n\nPairs come from: {source}.  Usage:\n"
           "    python3 THIS_SCRIPT TOPOLOGY.parm7 OUTPUT.dist\n"
           "Needs only Python 3 (no ParmEd).  Atom indices are looked up by residue number and atom "
           "name, so run it on the final topology (after solvation / ParmEd).")
    return embed(_helix_dist, {"PAIRS": [tuple(p) for p in pairs], "BONDS": bonds,
                               "WINDOW": restraint_window(s), "OFFSET": int(s.get("res_offset", 0))},
                 "main(sys.argv[1:], PAIRS, BONDS, WINDOW, OFFSET)", doc)


def count_solute_residues(pdb_path: str) -> Optional[int]:
    """Residues in a PDB that are not water or ions (None if unreadable)."""
    skip = {"HOH", "WAT", "SOL", "NA", "NA+", "CL", "CL-", "K", "K+", "MG", "MG2", "ZN", "CA"}
    seen = set()
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
                    if line[17:21].strip().upper() in skip:
                        continue
                    seen.add((line[21], line[22:27]))
    except OSError:
        return None
    return len(seen) or None


# ---------------------------------------------------------------------------
# Amber system build
# ---------------------------------------------------------------------------


@register
class AmberBuildTool(Tool):
    name = "amber-build"
    label = "Amber · system build"
    category = "Amber"
    tag = "amber-build"
    description = ("tleap topology and coordinates from a PDB: RNA force field, water box and ions, "
                   "ParmEd NBfix19 / 0BPh / HMR / 12-6-4, optional two-pass build with L-RNA renaming.")
    inputs = [Port("structure", "Starting structure (PDB)", ("pdb",),
                   tooltip="RNA model to parameterise, e.g. the best FARFAR2 cluster centre.")]
    sections = [
        Section("prep", "Input structure", [
            Option("prep", "Prepare the PDB for tleap", "bool", True,
                   "Write a cleaned copy of the input first: hydrogens removed (tleap rebuilds them), "
                   "TER records between chains, first MODEL only. Rosetta / FARFAR2 models need this."),
            Option("strip_5p", "Remove 5′-terminal phosphates", "bool", True,
                   "Drop P/OP1/OP2 of each chain's first residue: Amber's 5′-terminal residues (G5, "
                   "A5 ...) have no phosphate.", requires={"prep": True}),
            Option("gap_ter", "Break chains at numbering gaps", "bool", True,
                   "Insert a TER record where residue numbers jump, so tleap does not bond across "
                   "a gap.", requires={"prep": True}),
            Option("strip_water", "Remove waters from the input", "bool", True,
                   "Crystal waters are dropped; the solvent box is built by tleap.",
                   requires={"prep": True}),
        ], tooltip="How the starting PDB is cleaned before tleap reads it."),
        Section("forcefield", "Force field", [
            Option("rna_ff", "RNA force field", "choice", "ol3cp",
                   "leaprc file for RNA. 'OL3 + CP' adds the phosphate oxygen vdW correction of "
                   "Steinbrecher et al. (JCTC 2012) from the frcmod below; NBfix19 / 0BPh assume it.",
                   choices=[(k, v[0]) for k, v in FORCE_FIELDS.items()]),
            Option("cp_frcmod", "CP frcmod file", "file", "frcmod.99bsc0-chiol3-CaseP",
                   "Parameter file with the CP phosphate correction: a name in leap's search path / "
                   "the step folder, or an absolute path.", requires={"rna_ff": "ol3cp"}),
            Option("extra_frcmod", "Extra parameter files", "multiline", "",
                   "One frcmod file per line, loaded with loadamberparams after the force field "
                   "(e.g. frcmod.LRNA).", placeholder="frcmod.LRNA"),
            Option("extra_libs", "Extra residue libraries", "multiline", "",
                   "One library (.lib / .off) per line, loaded with loadOff (e.g. LRNA.lib).",
                   placeholder="LRNA.lib"),
            Option("leap_before_params", "Leap commands before parameter files", "multiline", "",
                   "Inserted right after the leaprc files, e.g. addAtomTypes for new atom types.",
                   placeholder='addAtomTypes { { "LD" "O" "sp3" } }'),
            Option("leap_before_load", "Leap commands before loadpdb", "multiline", "",
                   "Inserted before the PDB is read, e.g. addPdbResMap entries.",
                   placeholder='addPdbResMap { { "LG" "G0" } }'),
            Option("leap_after_load", "Leap commands after loadpdb", "multiline", "",
                   "Inserted after the PDB is read and before solvation, e.g. bond commands."),
        ], tooltip="RNA force field and extra parameters / libraries / leap commands."),
        Section("solvent", "Solvent & ions", [
            Option("water", "Water model", "choice", "spce",
                   "Water model; its leaprc also loads the matching monovalent / divalent ion "
                   "parameters.", choices=[(k, v[0]) for k, v in WATERS.items()]),
            Option("box", "Box shape", "radio", "oct",
                   "Truncated octahedron (solvateoct) needs ~30 % less water than a rectangular box.",
                   choices=[("oct", "Truncated octahedron"), ("rect", "Rectangular"), ("cubic", "Cubic")],
                   requires={"water": SOLVATED}),
            Option("buffer", "Buffer", "float", 15.0,
                   "Minimum distance between the RNA and the box edge.", minimum=5.0, maximum=50.0,
                   step=0.5, digits=1, unit="Å",
                   requires={"water": SOLVATED}),
            Option("neutralize", "Neutralise with monovalent cations", "bool", True,
                   "addIonsRand <cation> 0 neutralises the RNA's phosphate charge.",
                   requires={"water": SOLVATED}),
            Option("cation", "Cation", "radio", "Na+", "Monovalent cation for neutralisation and salt.",
                   choices=[("Na+", "Na⁺"), ("K+", "K⁺")],
                   requires={"water": SOLVATED}),
            Option("mg_count", "Mg²⁺ ions", "int", 0, "Number of Mg²⁺ ions (residue MG) added at random "
                   "positions.", minimum=0, maximum=10000,
                   requires={"water": SOLVATED}),
            Option("mg_balance", "Balance Mg²⁺ with Cl⁻", "bool", True,
                   "Add two Cl⁻ per Mg²⁺ so the box stays neutral.",
                   requires={"water": SOLVATED}),
            Option("salt_pairs", "Extra salt pairs", "int", 0,
                   "Additional cation/Cl⁻ pairs (≈ concentration [M] × number of waters / 55.5).",
                   minimum=0, maximum=100000,
                   requires={"water": SOLVATED}),
            Option("mg_model", "Mg²⁺ Lennard-Jones model", "radio", "126",
                   "12-6: the Li & Merz parameters loaded with the water model. 12-6-4 (Li & Merz): "
                   "loads frcmod.ions234lm_1264_<water> and adds the C4 terms with ParmEd add12_6_4 "
                   "(Amber only; GROMACS cannot represent C4). microMg / nanoMg (Grotz, Cruz-León & "
                   "Schwierz): optimised 12-6 Mg²⁺ with explicit Mg²⁺-RNA pair terms tuned to phosphate "
                   "binding - written as a frcmod plus ParmEd changeLJPair lines, and carried into "
                   "GROMACS as [ nonbond_params ]. microMg keeps the experimental (µs) water exchange; "
                   "nanoMg exchanges water faster for better sampling of ion binding.",
                   choices=[("126", "12-6 (default)"), ("1264", "12-6-4 (Li & Merz)"),
                            ("micro", "microMg"), ("nano", "nanoMg")],
                   requires={"water": SOLVATED}),
            Option("mg_cl_pair", "Mg²⁺-Cl⁻ pair term", "bool", True,
                   "Apply the authors' Mg²⁺-Cl⁻ pair term (optimised with their Cl⁻ model; Amber's Cl⁻ "
                   "parameters are kept for everything else). Without it Mg²⁺-Cl⁻ follows the combination "
                   "rule with the optimised Mg²⁺, which is not a published combination and gives much too "
                   "strong Mg²⁺-Cl⁻ contacts.",
                   requires={"water": SOLVATED, "mg_model": ("micro", "nano")}),
        ], tooltip="Water box and ions (tleap solvateoct / solvatebox and addIonsRand)."),
        Section("parmed", "ParmEd", [
            Option("nbfix19", "NBfix19 (gHBfix19 as LJ pairs)", "bool", True,
                   "Off-diagonal Lennard-Jones terms for -NH...N- and -OH...O(phosphate) pairs "
                   "(Kuehrova & Mlynsky JCTC 2019; Mlynsky et al. 2026). The masks assume OL3 + CP."),
            Option("bph0", "0BPh pairs", "bool", True,
                   "Extra H5/H4 – OS Lennard-Jones pairs of the 0BPh modification (OL3 + CP)."),
            Option("split_types", "Separate HO and O2 Lennard-Jones types first", "bool", False,
                   "addLJType @%HO and @%O2 before the pair changes. tleap merges atom types with equal "
                   "LJ parameters: HO shares its LJ type with water HW and O2 with base carbonyl O, so "
                   "without this the HO-O2 / HO-OS pairs also change water-H and carbonyl-O interactions.",
                   requires={"nbfix19": True}),
            Option("hmr", "Hydrogen mass repartitioning", "bool", False,
                   "ParmEd HMassRepartition (hydrogens 3.024 Da) so MD can use a 4 fs time step "
                   "with SHAKE."),
            Option("parmed_extra", "Extra ParmEd commands", "multiline", "",
                   "Added before outparm, e.g. the L-RNA changeLJPair lines.",
                   placeholder="changeLJPair @%HO @%LD 3.03 0.01"),
        ], tooltip="Topology edits applied with ParmEd after tleap (leap cannot reload a prmtop)."),
        Section("twopass", "Two-pass / L-RNA", [
            Option("two_pass", "Two-pass build", "bool", False,
                   "Dry tleap -> ParmEd -> ambpdb -> (to_lrna.sh) -> solvated tleap -> ParmEd. Needed "
                   "when residues must be renamed (e.g. back to L-RNA) between the passes."),
            Option("lrna", "Rename residues to L-RNA (to_lrna.sh)", "bool", False,
                   "Call to_lrna.sh on the dry PDB: G/A/C/U in the given ranges become LG/LA/LC/LU "
                   "(ambpdb writes them as D-RNA).", requires={"two_pass": True}),
            Option("lrna_script", "to_lrna.sh", "file", "to_lrna.sh",
                   "The renaming script: a name on PATH or a path (relative to the step folder or "
                   "absolute).", requires={"two_pass": True, "lrna": True}),
            Option("lrna_chain", "Chain", "text", "A", "Chain identifier passed to to_lrna.sh.",
                   requires={"two_pass": True, "lrna": True}),
            Option("lrna_ranges", "Residue ranges", "text", "",
                   "Residue range(s) to rename, e.g. '45-73' or '45-73, 80-90'.",
                   placeholder="45-73", requires={"two_pass": True, "lrna": True}),
        ], tooltip="Leap cannot reload a prmtop, so renaming residues needs a dry first pass."),
        Section("outputs", "Outputs", [
            Option("save_pdb", "Also save a PDB of the built system", "bool", True,
                   "tleap savepdb after solvation - handy for a visual check."),
        ]),
        env_section("source /opt/amber26/amber.sh   or   module load Amber/24"),
    ]

    def output_ports(self, s):
        ports = [Port("topology", "Topology (parm7)", ("parm7",)),
                 Port("coordinates", "Coordinates (rst7)", ("rst7",))]
        if s.get("save_pdb"):
            ports.append(Port("pdb", "System PDB", ("pdb",)))
        if s.get("two_pass"):
            ports.append(Port("dry_topology", "Dry topology (parm7)", ("parm7",)))
            ports.append(Port("dry_pdb", "Dry PDB (after renaming)", ("pdb",)))
        return ports

    # names ------------------------------------------------------------
    @staticmethod
    def _solvated(s) -> bool:
        return s["water"] != "none"

    @staticmethod
    def _lj_mods(s) -> bool:
        return bool(s["nbfix19"] or s["bph0"] or extra_lines(s["parmed_extra"]))

    @staticmethod
    def _mg_pairs(s) -> bool:
        """microMg / nanoMg: frcmod for the ion + ParmEd pair terms (solvated builds only)."""
        return (s["water"] != "none" and s["mg_model"] in mg_params.MODELS
                and (s["mg_model"], s["water"]) in mg_params.MG_SETS)

    def _wet_parmed(self, s) -> bool:
        return bool(self._lj_mods(s) or s["hmr"] or self._mg_pairs(s)
                    or (self._solvated(s) and s["mg_model"] == "1264"))

    def _final_topology(self, s, ctx) -> str:
        return ctx.name("parmed" if self._wet_parmed(s) else "tleap", "parm7")

    def describe(self, s):
        parts = [FORCE_FIELDS[s["rna_ff"]][2]]
        if self._solvated(s):
            parts.append(WATERS[s["water"]][0])
        mods = [m for m, on in (("NBfix19", s["nbfix19"]), ("0BPh", s["bph0"]), ("HMR", s["hmr"]))
                if on]
        if mods:
            parts.append("+".join(mods))
        if self._solvated(s) and s["mg_model"] != "126":
            parts.append(mg_params.MODELS.get(s["mg_model"], "12-6-4 Mg"))
        if s["two_pass"]:
            parts.append("two-pass" + (" L-RNA" if s["lrna"] else ""))
        return ", ".join(parts)

    # checks -----------------------------------------------------------
    def check(self, s, ctx):
        out: List[Issue] = []
        if (s["nbfix19"] or s["bph0"]) and s["rna_ff"] != "ol3cp":
            out.append(Issue("error", "NBfix19 / 0BPh masks assume the OL3 + CP force field; choose "
                                      "'OL3 + CP' or switch the modifications off",
                             ("rna_ff", "nbfix19", "bph0")))
        if s["rna_ff"] == "ol3cp" and not s["cp_frcmod"].strip():
            out.append(Issue("error", "OL3 + CP needs the CP frcmod file", ("cp_frcmod",)))
        if s["rna_ff"] == "ol3cp" and not extra_lines(s["extra_libs"]):
            new_types = _frcmod_new_types(s["cp_frcmod"].strip(), ctx)
            if new_types:
                out.append(Issue("warning", f"the CP frcmod defines atom types {', '.join(new_types)} that the "
                                            "standard RNA library (RNA.lib: O2 / OS) does not use, so the CP "
                                            "correction has no effect; add a CP residue library under 'Extra "
                                            "residue libraries'", ("cp_frcmod", "extra_libs")))
        if s["rna_ff"] == "shaw" and s["water"] not in ("tip4pd", "none"):
            out.append(Issue("warning", "the DESRES RNA force field was parameterised with TIP4P-D "
                                        "water", ("rna_ff", "water")))
        solvated = self._solvated(s)
        if not solvated:
            out.append(Issue("warning", "no solvent box: the topology is for gas-phase / implicit "
                                        "solvent use; periodic MD stages will not run", ("water",)))
        else:
            if s["mg_model"] == "1264" and not WATERS[s["water"]][4]:
                out.append(Issue("error", f"AmberTools has no 12-6-4 ion parameters for "
                                          f"{WATERS[s['water']][0]}", ("mg_model", "water")))
            if s["mg_model"] == "1264" and s["mg_count"] == 0:
                out.append(Issue("warning", "12-6-4 is selected but no Mg²⁺ is added; it only matters "
                                            "when the input already contains MG ions",
                                 ("mg_model", "mg_count")))
            if s["mg_model"] in mg_params.MODELS:
                out += self._check_mg_pairs(s)
            if s["buffer"] < 10.0:
                out.append(Issue("warning", "a buffer below 10 Å can let periodic images of the RNA "
                                            "interact", ("buffer",)))
            if not s["neutralize"] and s["mg_count"] and not s["mg_balance"]:
                out.append(Issue("warning", "the box is not neutralised and Mg²⁺ is not balanced: "
                                            "PME will use a net-charge background", ("neutralize",)))
        if s["two_pass"] and not s["lrna"]:
            out.append(Issue("warning", "the two-pass build only matters when residues are renamed "
                                        "between the passes; a single pass gives the same topology",
                             ("two_pass",)))
        if s["two_pass"] and s["lrna"]:
            if not s["lrna_script"].strip():
                out.append(Issue("error", "give the path of to_lrna.sh", ("lrna_script",)))
            if not s["lrna_chain"].strip():
                out.append(Issue("error", "give the chain passed to to_lrna.sh", ("lrna_chain",)))
            if not _ranges(s["lrna_ranges"]):
                out.append(Issue("error", "give the residue range(s) to rename, e.g. 45-73",
                                 ("lrna_ranges",)))
            if not (extra_lines(s["extra_libs"]) or extra_lines(s["extra_frcmod"])):
                out.append(Issue("warning", "L-RNA residues need their library / parameters: add "
                                            "e.g. LRNA.lib and frcmod.LRNA under Force field",
                                 ("extra_libs", "extra_frcmod")))
        if s["lrna_ranges"].strip() and not _ranges(s["lrna_ranges"]):
            out.append(Issue("error", "residue ranges look like '45-73' or '45-73, 80-90'",
                             ("lrna_ranges",)))
        if not s["prep"] and ctx.value("structure") and ctx.value("structure").source:
            src = ctx.value("structure").source_label
            out.append(Issue("info", f"the PDB from {src} is used as is; models from Rosetta usually "
                                     "need 'Prepare the PDB for tleap'", ("prep",)))
        if s["hmr"]:
            out.append(Issue("info", "with hydrogen mass repartitioning the MD stages can use "
                                     "dt = 0.004 ps", ("hmr",)))
        if s["nbfix19"] and not s["split_types"] and self._solvated(s):
            out.append(Issue("info", "tleap gives HO the same Lennard-Jones type as water HW (both zero) and "
                                     "O2 the same as carbonyl O, so 'changeLJPair @%HO ...' also changes "
                                     "water-H and carbonyl-O pairs; tick 'Separate HO and O2 Lennard-Jones "
                                     "types first' if that is not intended", ("split_types",)))
        return out

    def _check_mg_pairs(self, s) -> List[Issue]:
        out: List[Issue] = []
        model, water = s["mg_model"], s["water"]
        name, wlabel = mg_params.MODELS[model], WATERS[water][0]
        if (model, water) in mg_params.UNAVAILABLE:
            out.append(Issue("error", f"{name} is not available for {wlabel}: "
                                      f"{mg_params.UNAVAILABLE[(model, water)]}", ("mg_model", "water")))
            return out
        if (model, water) not in mg_params.MG_SETS:
            have = sorted({WATERS[w][0] for m, w in mg_params.MG_SETS if m == model})
            out.append(Issue("error", f"no published {name} parameters for {wlabel} (available: "
                                      f"{', '.join(have)})", ("mg_model", "water")))
            return out
        if s["mg_count"] == 0:
            out.append(Issue("warning", f"{name} is selected but no Mg²⁺ is added; it only matters when "
                                        "the input already contains MG ions", ("mg_model", "mg_count")))
        if s["rna_ff"] not in ("ol3", "ol3cp"):
            out.append(Issue("warning", f"the {name} Mg²⁺-RNA pair terms were fitted with parmBSC0χOL3 "
                                        "atom types; RNA atom types of this force field that are not "
                                        "in that set keep the plain combination rule", ("mg_model", "rna_ff")))
        mg_extra = [ln for ln in extra_lines(s["parmed_extra"])
                    if re.search(r":MG\b|Mg2\+", ln, re.IGNORECASE)]
        if mg_extra:
            out.append(Issue("warning", f"the {name} pair terms are applied after the extra ParmEd commands "
                                        "and replace any Mg²⁺ pair terms set there",
                             ("parmed_extra",)))
        p = mg_params.get_set(model, water)
        has_cl = bool(s["mg_balance"] and s["mg_count"] or s["salt_pairs"])
        if s["mg_cl_pair"] and has_cl:
            out.append(Issue("info", f"{name} for {wlabel} was optimised with the {p.chloride} Cl⁻ model; "
                                     "the Mg²⁺-Cl⁻ pair term is applied, other Cl⁻ interactions use Amber's "
                                     "Cl⁻", ("mg_cl_pair",)))
        if not s["mg_cl_pair"] and has_cl:
            out.append(Issue("warning", f"without the pair term, Mg²⁺-Cl⁻ uses the combination rule with the "
                                        f"optimised {name} parameters - not a published combination (the "
                                        f"authors' pair: σ {p.mg_cl[0]:.3f} nm, ε {p.mg_cl[1]:.2f} kJ/mol) and "
                                        "far too strong Mg²⁺-Cl⁻ contacts", ("mg_cl_pair",)))
        if extra_lines(s["extra_libs"]) or extra_lines(s["extra_frcmod"]) or s["lrna"]:
            out.append(Issue("warning", f"the {name} pair terms cover the 27 parmBSC0χOL3 RNA atom types; atom "
                                        "types from extra libraries or parameter files (e.g. L-RNA) keep the "
                                        "plain combination rule with Mg²⁺. Phosphate oxygens retyped by a CP "
                                        "library (OP, OR) get the authors' O2 / OS terms - a choice by MDATK, "
                                        "since the authors fitted Mg²⁺ against plain OL3 phosphate oxygens",
                             ("mg_model", "extra_libs")))
        out.append(Issue("info", f"{name} ({mg_params.CITATIONS[water]}): Mg²⁺-water uses the "
                                 "Lorentz-Berthelot rule, Mg²⁺-RNA the authors' pair terms; GROMACS "
                                 "conversion keeps them as [ nonbond_params ]", ("mg_model",)))
        return out

    def output_values(self, s, ctx):
        nres = None
        path = ctx.existing("structure")
        if path:
            nres = count_solute_residues(path)
        data = {"nbfix": bool(s["nbfix19"] or s["bph0"]), "hmr": bool(s["hmr"]),
                "c4": bool(self._solvated(s) and s["mg_model"] == "1264"),
                "mg_pairs": mg_params.MODELS[s["mg_model"]] if self._mg_pairs(s) else "",
                "box": self._solvated(s), "solute_residues": nres,
                "forcefield": s["rna_ff"], "water": s["water"],
                "four_site_water": s["water"] in FOUR_SITE_WATERS}
        out = {"topology": PortValue([ctx.from_own(self._final_topology(s, ctx))], "parm7", dict(data)),
               "coordinates": PortValue([ctx.own("tleap", "rst7")], "rst7", dict(data, velocities=False))}
        if s["save_pdb"]:
            out["pdb"] = PortValue([ctx.own("tleap", "pdb")], "pdb", dict(data))
        if s["two_pass"]:
            dry_top = "parmed-dry" if self._lj_mods(s) else "tleap-dry"
            out["dry_topology"] = PortValue([ctx.own(dry_top, "parm7")], "parm7",
                                            dict(data, box=False, c4=False, hmr=False, mg_pairs=""))
            out["dry_pdb"] = PortValue([ctx.own("lrna" if s["lrna"] else "ambpdb", "pdb")], "pdb",
                                       dict(data, box=False))
        return out

    # files ------------------------------------------------------------
    def _leap(self, s, ctx, title, pdb_in, stem, wet, save_pdb) -> str:
        lines = [f"# {title}", f"# written by {PROJECT_ABBREV} {__version__}",
                 f"source {FORCE_FIELDS[s['rna_ff']][1]}"]
        solvated = wet and self._solvated(s)
        if solvated:
            lines.append(f"source {WATERS[s['water']][1]}")
        lines += extra_lines(s["leap_before_params"])
        if s["rna_ff"] == "ol3cp":
            lines.append(f"loadamberparams {s['cp_frcmod'].strip()}")
        lines += [f"loadamberparams {f.strip()}" for f in extra_lines(s["extra_frcmod"])]
        if solvated and s["mg_model"] == "1264" and WATERS[s["water"]][4]:
            lines.append(f"loadamberparams {WATERS[s['water']][4]}")
        if solvated and self._mg_pairs(s):
            lines.append(f"loadamberparams {self._mg_frcmod(ctx)}")
        lines += [f"loadOff {f.strip()}" for f in extra_lines(s["extra_libs"])]
        lines += extra_lines(s["leap_before_load"])
        lines.append(f'mol = loadpdb "{pdb_in}"')
        lines += extra_lines(s["leap_after_load"])
        if solvated:
            box = WATERS[s["water"]][2]
            buf = fortran_float(s["buffer"])
            if s["box"] == "oct":
                lines.append(f"solvateoct mol {box} {buf}")
            elif s["box"] == "rect":
                lines.append(f"solvatebox mol {box} {buf}")
            else:
                lines.append(f"solvatebox mol {box} {buf} iso")
            lines.append("")
            ions = []
            if s["neutralize"]:
                ions.append(f"addIonsRand mol {s['cation']} 0")
            if s["mg_count"]:
                ions.append(f"addIonsRand mol MG {s['mg_count']}")
            n_cl = (2 * s["mg_count"] if s["mg_balance"] else 0) + s["salt_pairs"]
            if s["salt_pairs"]:
                ions.append(f"addIonsRand mol {s['cation']} {s['salt_pairs']}")
            if n_cl:
                ions.append(f"addIonsRand mol Cl- {n_cl}")
            lines += ions
            lines.append("")
        lines.append("check mol")
        lines.append(f"saveamberparm mol {stem}.parm7 {stem}.rst7")
        if save_pdb:
            lines.append(f"savepdb mol {stem}.pdb")
        lines.append("quit")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _mg_frcmod(ctx) -> str:
        return ctx.name("mg", "frcmod")

    def _parmed(self, s, title, out_parm, wet, ctx=None) -> str:
        lines = ["#", f"# {title}", f"# ParmEd input written by {PROJECT_ABBREV} {__version__}"]
        if s["nbfix19"] and s["split_types"]:
            lines += ["#", "# Give HO and O2 their own Lennard-Jones types so the pair changes below do not also",
                      "# apply to atom types tleap merged with them (water HW, carbonyl O)",
                      "addLJType @%HO", "addLJType @%O2"]
        if s["nbfix19"]:
            lines += NBFIX19_LINES
        if s["bph0"]:
            lines += BPH0_LINES
        if s["nbfix19"] or s["bph0"]:
            lines += LRNA_NOTE
        if wet and self._solvated(s) and s["mg_model"] == "1264":
            lines += ["#", "# 12-6-4 Lennard-Jones terms for Mg2+ (Li & Merz)",
                      f"add12_6_4 :MG watermodel {WATERS[s['water']][3]}"]
        if wet and s["hmr"]:
            lines += ["#", "# Hydrogen mass repartitioning (4 fs time steps with SHAKE)",
                      "HMassRepartition"]
        extra = extra_lines(s["parmed_extra"])
        if extra:
            lines += ["#", "# Extra commands"] + extra
        if wet and self._mg_pairs(s):
            lines += mg_params.parmed_lines(s["mg_model"], s["water"], WATERS[s["water"]][0],
                                            self._mg_frcmod(ctx), s["mg_cl_pair"])
        lines += ["#", f"outparm {out_parm}"]
        return "\n".join(lines) + "\n"

    def generate(self, s, ctx):
        files: List[GeneratedFile] = []
        n = ctx.name
        script_name = n(self.tag, "sh")
        ff = FORCE_FIELDS[s["rna_ff"]][2]
        sh = script_header(f"Amber system build ({ff}) - tleap / ParmEd", ctx, s, script_name)
        pdb_in = ctx.inp("structure")
        if s["prep"]:
            prep_py, prep_pdb = n("prep", "py"), n("prep", "pdb")
            files.append(GeneratedFile(prep_py, embed(
                _prep_pdb, {}, "main(sys.argv[1:], strip_h=True, strip_5p=%r, gap_ter=%r, "
                "strip_water=%r)" % (s["strip_5p"], s["gap_ter"], s["strip_water"]),
                f"Prepare an RNA PDB for tleap - written by {PROJECT_ABBREV} {__version__}.\n\n"
                "Usage: python3 THIS_SCRIPT INPUT.pdb OUTPUT.pdb"), "helper",
                "cleans the starting PDB for tleap"))
            sh += ["# Prepare the starting structure for tleap (hydrogens, 5'-phosphates, TER records)",
                   f"python3 {prep_py} {pdb_in} {prep_pdb}", ""]
            pdb_in = prep_pdb
        lj = self._lj_mods(s)
        mods = " + ".join(m for m, on in (("NBfix19", s["nbfix19"]), ("0BPh", s["bph0"]),
                                          ("extra commands", bool(extra_lines(s["parmed_extra"]))))
                          if on)
        if s["two_pass"]:
            dry = n("tleap-dry")
            files.append(GeneratedFile(n("tleap-dry", "leap"), self._leap(
                s, ctx, f"Dry build ({ff}) for project {ctx.project}, step {ctx.step:02d}", pdb_in, dry,
                wet=False, save_pdb=False), "input", "dry tleap input"))
            sh += [f"# First parameterisation (dry), {ff}",
                   f"rm -f {dry}.parm7 {dry}.rst7",
                   f"tleap -f {dry}.leap | tee {dry}.log",
                   require_file(f"{dry}.parm7", "tleap (dry)"), ""]
            dry_top = f"{dry}.parm7"
            if lj:
                pm = n("parmed-dry")
                files.append(GeneratedFile(n("parmed-dry", "in"), self._parmed(
                    s, f"{mods} on the dry topology", f"{pm}.parm7", wet=False), "input"))
                sh += [f"# Edit parameters: {mods}",
                       f"parmed -O {dry}.parm7 -i {pm}.in 2>&1 | tee {pm}.log",
                       require_file(f"{pm}.parm7", "ParmEd"), ""]
                dry_top = f"{pm}.parm7"
            amb = n("ambpdb", "pdb")
            sh += ["# PDB of the dry system (ambpdb writes standard D-RNA residue names)",
                   f"ambpdb -p {dry_top} -c {dry}.rst7 > {amb}", require_file(amb, "ambpdb"), ""]
            pdb_in = amb
            if s["lrna"]:
                ranges = _ranges(s["lrna_ranges"])
                out_pdb = n("lrna", "pdb")
                src = amb
                for k, (lo, hi) in enumerate(ranges):
                    dst = out_pdb if k == len(ranges) - 1 else n(f"lrna{k + 1}", "pdb")
                    sh.append(f"# Convert residues {lo}-{hi} of chain {s['lrna_chain']} back to L-RNA")
                    sh.append(f"{s['lrna_script'].strip()} {src} {s['lrna_chain'].strip()} {lo},{hi} > {dst}")
                    src = dst
                sh += [require_file(out_pdb, "to_lrna.sh"), ""]
                pdb_in = out_pdb
        if self._mg_pairs(s):
            name = mg_params.MODELS[s["mg_model"]]
            files.append(GeneratedFile(self._mg_frcmod(ctx), mg_params.frcmod_text(
                s["mg_model"], s["water"], WATERS[s["water"]][0]), "input",
                f"{name} Mg2+ Lennard-Jones parameters (replace Li & Merz 12-6)"))
        wet = n("tleap")
        what = "Solvate and add ions" if self._solvated(s) else "Build the topology"
        files.append(GeneratedFile(n("tleap", "leap"), self._leap(
            s, ctx, f"{what} ({ff}) for project {ctx.project}, step {ctx.step:02d}", pdb_in, wet,
            wet=True, save_pdb=s["save_pdb"]), "input", "tleap input"))
        sh += [f"# {what}", f"rm -f {wet}.parm7 {wet}.rst7",
               f"tleap -f {wet}.leap | tee {wet}.log", require_file(f"{wet}.parm7", "tleap"), ""]
        final = f"{wet}.parm7"
        if self._wet_parmed(s):
            pm = n("parmed")
            mg = ("12-6-4" if self._solvated(s) and s["mg_model"] == "1264" else
                  mg_params.MODELS[s["mg_model"]] + " pairs" if self._mg_pairs(s) else "")
            wet_mods = [m for m in (mods, mg, "HMR" if s["hmr"] else "") if m]
            files.append(GeneratedFile(n("parmed", "in"), self._parmed(
                s, f"{' + '.join(wet_mods)} on the solvated topology", f"{pm}.parm7", wet=True, ctx=ctx),
                "input"))
            sh += [f"# {'Re-apply' if s['two_pass'] and lj else 'Apply'} {' + '.join(wet_mods)}",
                   f"parmed -O {wet}.parm7 -i {pm}.in 2>&1 | tee {pm}.log",
                   require_file(f"{pm}.parm7", "ParmEd"), ""]
            final = f"{pm}.parm7"
        sh += [f'echo "Topology: {final}  Coordinates: {wet}.rst7"', ""]
        files.append(GeneratedFile(script_name, "\n".join(sh), "run", "runs tleap / ParmEd"))
        return files


RNA_LIB_TYPES = {"C", "C4", "C5", "CA", "CB", "CI", "CP", "CQ", "CS", "CT", "H", "H1", "H2", "H4", "H5",
                 "HA", "HO", "N*", "N2", "NA", "NB", "NC", "O", "O2", "OH", "OS", "P"}


def _frcmod_new_types(path: str, ctx) -> List[str]:
    """Atom types with NONB parameters in a frcmod that RNA.lib does not use ([] if unreadable)."""
    if not path:
        return []
    for cand in (path, os.path.join(ctx.folder, path) if getattr(ctx, "folder", "") else ""):
        if cand and os.path.isfile(cand):
            break
    else:
        return []
    found, section = [], ""
    try:
        with open(cand, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()[1:]
    except OSError:
        return []
    for line in lines:
        head = line.strip().upper()
        if head in ("MASS", "BOND", "ANGLE", "ANGL", "DIHE", "IMPROPER", "IMPR", "NONB", "NONBON", "HBON",
                    "HBOND", "CMAP"):
            section = head
            continue
        if section.startswith("NONB") and line.strip():
            t = line.split()[0]
            if t not in RNA_LIB_TYPES and t not in found:
                found.append(t)
    return found


def _ranges(text: str) -> List[Tuple[int, int]]:
    out = []
    for tok in re.split(r"[,\s]+", (text or "").strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
        if not m:
            return []
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        if hi < lo:
            return []
        out.append((lo, hi))
    return out


# ---------------------------------------------------------------------------
# helix restraints
# ---------------------------------------------------------------------------


@register
class AmberRestraintsTool(Tool):
    name = "amber-restraints"
    label = "Amber · helix restraints"
    category = "Amber"
    tag = "amber-restraints"
    description = ("Helix-preserving DISANG distance restraints for the hydrogen bonds of every "
                   "base pair in the secondary structure (looked up in the final topology).")
    inputs = [Port("topology", "Topology (parm7)", ("parm7",),
                   tooltip="The solvated (and ParmEd-modified) topology the MD runs with."),
              Port("secstruct", "Secondary structure", ("secstruct",), optional=True,
                   tooltip="Optional: a dot-bracket file or a FARFAR2 step; otherwise type it below.")]
    outputs = [Port("restraints", "DISANG restraints", ("dist",))]
    sections = [Section("restraints", "Base pairs", restraint_options(),
                        tooltip="Each G–C pair gets three, each A–U pair two distance restraints "
                                "(flat-bottom window r1–r4)."),
                env_section("module load python   (only python3 is needed)")]

    def check(self, s, ctx):
        model, _src, out = secondary_structure(s, ctx)
        if model is None:
            if not out:
                out.append(Issue("error", "enter the secondary structure (dot-bracket and sequence) or "
                                          "connect a secondary-structure file", ("dotbracket",)))
            return out
        _pairs, more = restraint_pairs(model, s)
        out += more
        if ctx.existing("topology"):
            try:
                self._direct(s, ctx, model)
            except (OSError, ValueError) as exc:
                out.append(Issue("error", f"topology check: {exc}", ("res_offset", "sequence")))
        return out

    def _direct(self, s, ctx, model) -> str:
        pairs, _ = restraint_pairs(model, s)
        names, labels, pointers = _helix_dist.read_parm7_names(ctx.existing("topology"))
        bonds = rna.HBOND_ATOMS
        return _helix_dist.restraint_text(names, labels, pointers, pairs, bonds, restraint_window(s),
                                          int(s["res_offset"]))

    def describe(self, s):
        if not (s.get("dotbracket") or "").strip():
            return "pairs from the secondary-structure input"
        try:
            model = rna.build_model(s.get("sequence", ""), s.get("dotbracket", ""))
            return f"{len(model.pairs)} base pairs"
        except ValueError:
            return "check the secondary structure"

    def output_values(self, s, ctx):
        model, _src, _ = secondary_structure(s, ctx)
        data = {"pairs": len(model.pairs)} if model else {}
        return {"restraints": PortValue([ctx.own("helix", "dist")], "dist", data)}

    def generate(self, s, ctx):
        model, source, _ = secondary_structure(s, ctx)
        pairs, _ = restraint_pairs(model, s)
        n = ctx.name
        files = [GeneratedFile(n("helix", "py"), helix_script(pairs, s, source), "helper",
                               "writes the DISANG file from the topology")]
        top = ctx.inp("topology")
        dist = n("helix", "dist")
        if ctx.existing("topology"):
            files.append(GeneratedFile(dist, self._direct(s, ctx, model), "input",
                                       "restraints (topology already exists)"))
        script_name = n(self.tag, "sh")
        sh = script_header("Helix-preserving distance restraints (DISANG)", ctx, s, script_name)
        sh += [f"# {len(pairs)} base pairs from the {source}",
               f"python3 {n('helix', 'py')} {top} {dist}", require_file(dist, "the restraint script"), ""]
        files.append(GeneratedFile(script_name, "\n".join(sh), "run", "writes the DISANG file"))
        return files


# ---------------------------------------------------------------------------
# Amber MD stages
# ---------------------------------------------------------------------------

_MD_KIND = {"kind": "md"}
_MIN_KIND = {"kind": "min"}

STAGE_OPTIONS = [
    Option("name", "Name", "text", "stage", "Short stage name used in file names (letters, digits, "
           "hyphens), e.g. min1, heat, eq, prod."),
    Option("title", "Title", "text", "", "First line of the mdin file (a free comment)."),
    Option("kind", "Type", "radio", "md", "Energy minimisation (imin=1) or molecular dynamics (imin=0).",
           choices=[("min", "Minimisation"), ("md", "MD")]),
    # minimisation
    Option("maxcyc", "Cycles (maxcyc)", "int", 2000, "Total minimisation cycles.", minimum=1,
           maximum=10_000_000, requires=_MIN_KIND),
    Option("ncyc", "Steepest-descent cycles (ncyc)", "int", 500,
           "With ntmin=1: cycles of steepest descent before switching to conjugate gradient.",
           minimum=0, maximum=10_000_000, requires=_MIN_KIND),
    Option("ntmin", "Method (ntmin)", "radio", "1", "Minimisation algorithm.",
           choices=[("1", "SD then CG"), ("2", "SD only"), ("0", "CG only")], requires=_MIN_KIND),
    # dynamics
    Option("nstlim", "Steps (nstlim)", "int", 50000, "Number of MD steps.", minimum=1,
           maximum=10_000_000_000, requires=_MD_KIND),
    Option("dt", "Time step (dt)", "float", 0.002, "Time step in ps (0.002 with SHAKE; 0.004 needs "
           "hydrogen mass repartitioning).", minimum=0.0001, maximum=0.01, step=0.001, digits=4,
           unit="ps", requires=_MD_KIND),
    Option("restart", "Velocities", "radio", "auto",
           "auto: new velocities (irest=0, ntx=1) after a minimisation or at the start, otherwise "
           "continue the previous stage (irest=1, ntx=5).",
           choices=[("auto", "Auto"), ("new", "New (irest=0)"), ("continue", "Continue (irest=1)")],
           requires=_MD_KIND),
    Option("ensemble", "Ensemble", "radio", "nvt", "Constant volume (ntb=1) or constant pressure "
           "(ntb=2, ntp>0).", choices=[("nvt", "NVT"), ("npt", "NPT")], requires=_MD_KIND),
    Option("thermostat", "Thermostat", "radio", "langevin", "Temperature control (ntt).",
           choices=[("langevin", "Langevin (ntt=3)"), ("berendsen", "Berendsen (ntt=1)"),
                    ("andersen", "Andersen (ntt=2)")], requires=_MD_KIND),
    Option("gamma_ln", "Collision frequency (gamma_ln)", "float", 1.0, "Langevin collision "
           "frequency in ps⁻¹.", minimum=0.0, maximum=100.0, step=0.5, digits=2,
           requires={"kind": "md", "thermostat": "langevin"}),
    Option("tautp", "Coupling time (tautp)", "float", 1.0, "Berendsen thermostat time constant (ps).",
           minimum=0.0, maximum=100.0, step=0.1, digits=2,
           requires={"kind": "md", "thermostat": "berendsen"}),
    Option("vrand", "Randomisation interval (vrand)", "int", 1000, "Andersen: steps between velocity "
           "randomisations.", minimum=1, maximum=10_000_000,
           requires={"kind": "md", "thermostat": "andersen"}),
    Option("tempi", "Initial temperature (tempi)", "float", 0.0, "Temperature of the new velocities (K).",
           minimum=0.0, maximum=1000.0, step=10.0, digits=1, unit="K", requires=_MD_KIND),
    Option("temp0", "Temperature (temp0)", "float", 300.0, "Reference temperature (K).",
           minimum=0.0, maximum=1000.0, step=1.0, digits=1, unit="K", requires=_MD_KIND),
    Option("ramp", "Ramp the temperature", "bool", False,
           "Raise temp0 linearly from tempi with an nmropt TEMP0 schedule instead of jumping to "
           "temp0 at once.", requires=_MD_KIND),
    Option("ramp_steps", "Ramp length (steps)", "int", 20000, "Steps over which the temperature rises; "
           "the rest of the stage runs at temp0.", minimum=1, maximum=10_000_000_000,
           requires={"kind": "md", "ramp": True}),
    Option("barostat", "Barostat", "radio", "berendsen", "Pressure control in NPT.",
           choices=[("berendsen", "Berendsen (default)"), ("mc", "Monte Carlo (barostat=2)")],
           requires={"kind": "md", "ensemble": "npt"}),
    Option("pres0", "Pressure (pres0)", "float", 1.0, "Reference pressure (bar).", minimum=0.0,
           maximum=10000.0, step=0.1, digits=2, unit="bar", requires={"kind": "md", "ensemble": "npt"}),
    Option("taup", "Relaxation time (taup)", "float", 1.0, "Berendsen barostat time constant (ps).",
           minimum=0.0, maximum=100.0, step=0.1, digits=2,
           requires={"kind": "md", "ensemble": "npt", "barostat": "berendsen"}),
    Option("mcbarint", "MC volume moves every (mcbarint)", "int", 100, "Steps between Monte Carlo "
           "volume moves.", minimum=1, maximum=100000,
           requires={"kind": "md", "ensemble": "npt", "barostat": "mc"}),
    Option("ntp", "Pressure scaling (ntp)", "radio", "1", "Isotropic (ntp=1) or anisotropic (ntp=2).",
           choices=[("1", "Isotropic"), ("2", "Anisotropic")], requires={"kind": "md", "ensemble": "npt"}),
    Option("shake", "SHAKE bonds to hydrogen", "bool", True, "ntc=2, ntf=2 (needed for dt > 0.001 ps).",
           requires=_MD_KIND),
    Option("tol", "SHAKE tolerance (tol)", "float", None, "Relative SHAKE tolerance; empty uses "
           "Amber's default (1.0e-5).", optional=True, digits=8, requires={"kind": "md", "shake": True}),
    Option("ig", "Random seed (ig)", "int", None, "Empty: a new seed each run (ig=-1).", optional=True,
           minimum=-1, maximum=2**31 - 1, requires=_MD_KIND),
    # shared
    Option("cut", "Cutoff (cut)", "float", 10.0, "Direct-space non-bonded cutoff (Å).", minimum=6.0,
           maximum=30.0, step=0.5, digits=1, unit="Å"),
    Option("posres", "Positional restraints", "bool", False,
           "Harmonic restraints to the reference coordinates (ntr=1, restraint_wt, restraintmask)."),
    Option("posres_mask", "Restraint mask", "text", "",
           "Amber mask of restrained atoms; empty restrains the solute (':1-N' when the residue count "
           "is known, otherwise everything except water and ions).", requires={"posres": True},
           placeholder=":1-73"),
    Option("posres_wt", "Restraint weight", "float", 10.0, "restraint_wt in kcal/mol/Å².", minimum=0.0,
           maximum=10000.0, step=1.0, digits=2, requires={"posres": True}),
    Option("distres", "Helix distance restraints", "bool", False,
           "nmropt=1 with DISANG=<restraint file> (the step's restraint input)."),
    Option("listout", "Print restraint violations (LISTOUT=POUT)", "bool", False,
           "Writes the restraint violation summary to the output file.", requires={"distres": True}),
    Option("ntpr", "Energy output every (ntpr)", "int", 500, "Steps between energy printouts.",
           minimum=1, maximum=100_000_000),
    Option("ntwx", "Trajectory frame every (ntwx)", "int", 500, "Steps between trajectory frames "
           "(0: no trajectory).", minimum=0, maximum=100_000_000, requires=_MD_KIND),
    Option("ntwr", "Restart file every (ntwr)", "int", 5000, "Steps between restart-file updates.",
           minimum=1, maximum=1_000_000_000, requires=_MD_KIND),
    Option("netcdf", "NetCDF trajectory (ioutfm=1)", "bool", True, "Binary NetCDF instead of ASCII.",
           requires=_MD_KIND),
    Option("iwrap", "Wrap into the box (iwrap=1)", "bool", False, "Wrap molecules into the primary box "
           "in the output.", requires=_MD_KIND),
    Option("segments", "Split into segments", "int", 1, "Run this stage as N consecutive pmemd runs "
           "(e.g. to fit cluster wall-time limits); file names get -001, -002 ...", minimum=1,
           maximum=1000, requires=_MD_KIND),
    Option("production", "Production stage", "bool", False, "Its trajectory is passed on to analysis / "
           "visualisation steps and goes into the separate production script.", requires=_MD_KIND),
    Option("extra_cntrl", "Extra &cntrl settings", "multiline", "", "Additional name=value settings "
           "added inside &cntrl (they override generated values with the same name).",
           placeholder="nscm=1000,"),
    Option("extra_namelists", "Extra namelists", "multiline", "", "Text added after &cntrl, e.g. an "
           "&ewald namelist.", placeholder="&ewald\n  skinnb=2.0,\n/"),
]

_STAGE_TEMPLATES = [
    ("min1", "Minimisation, solute restrained", dict(
        name="min1", title="Minimize water/ions, RNA restrained", kind="min", maxcyc=2000, ncyc=500,
        posres=True, posres_wt=10.0, ntpr=100)),
    ("min2", "Minimisation, helix pairs held", dict(
        name="min2", title="Unrestrained min, helix pairs held", kind="min", maxcyc=2000, ncyc=500,
        distres=True, ntpr=100)),
    ("heat", "Heating (NVT, Langevin)", dict(
        name="heat", title="Heat 0 -> 300 K, helix pairs held", kind="md", nstlim=25000, dt=0.002,
        ensemble="nvt", tempi=0.0, temp0=300.0, tol=1.0e-6, distres=True)),
    ("heat-ramp", "Heating with a temperature ramp", dict(
        name="heat", title="Heat 0 -> 300 K (ramp), helix pairs held", kind="md", nstlim=25000,
        ensemble="nvt", tempi=0.0, temp0=300.0, ramp=True, ramp_steps=20000, tol=1.0e-6,
        distres=True)),
    ("eq", "NPT equilibration", dict(
        name="eq", title="NPT eq, light restraint", kind="md", nstlim=50000, ensemble="npt",
        distres=True)),
    ("eq-free", "NPT equilibration (no positional restraints)", dict(
        name="eq-free", title="Unrestrained NPT eq", kind="md", nstlim=50000, ensemble="npt",
        distres=True, listout=True)),
    ("prod", "Production (NPT, helix pairs held)", dict(
        name="prod", title="Production, helix pairs held", kind="md", nstlim=150_000_000,
        ensemble="npt", distres=True, ntwx=5000, ntwr=50000, production=True)),
    ("prod-free", "Production (NPT, unrestrained)", dict(
        name="prod", title="Production", kind="md", nstlim=150_000_000, ensemble="npt",
        ntwx=5000, ntwr=50000, production=True)),
    ("min-all", "Minimisation, no restraints", dict(
        name="min", title="Minimisation", kind="min", maxcyc=2000, ncyc=500, ntpr=100)),
]


@register
class AmberMDTool(Tool):
    name = "amber-md"
    label = "Amber · MD stages"
    category = "Amber"
    tag = "amber-md"
    description = ("pmemd input files (minimisation, heating, equilibration, production) and the "
                   "scripts that run them with pmemd, pmemd.cuda or pmemd.MPI.")
    inputs = [Port("topology", "Topology (parm7)", ("parm7",)),
              Port("coordinates", "Coordinates (rst7)", ("rst7",)),
              Port("restraints", "Restraints (DISANG)", ("dist",), optional=True,
                   tooltip="Needed by stages with 'Helix distance restraints'.")]
    sections = [
        Section("engine", "Engine", [
            Option("amber_version", "Amber version", "radio", "26",
                   "Version the files target (local Amber 26 or cluster Amber 24); the input format is "
                   "the same, the version is recorded in the scripts.",
                   choices=[("26", "Amber 26 (local)"), ("24", "Amber 24 (cluster)")]),
            Option("engine", "MD engine", "radio", "pmemd.cuda", "Program for the MD stages.",
                   choices=[("pmemd.cuda", "pmemd.cuda (GPU)"), ("pmemd", "pmemd (1 CPU core)"),
                            ("pmemd.MPI", "pmemd.MPI (CPU cores)")]),
            Option("nproc", "MPI processes", "int", 8, "Processes for pmemd.MPI.", minimum=1,
                   maximum=100000, requires={"engine": "pmemd.MPI"}),
            Option("mpi_launcher", "MPI launcher", "text", "mpirun -np {n}",
                   "Command placed before pmemd.MPI; {n} is replaced by the process count (srun on "
                   "SLURM clusters).", requires={"engine": "pmemd.MPI"}),
            Option("gpu", "GPU (CUDA_VISIBLE_DEVICES)", "text", "", "Optional GPU index for pmemd.cuda.",
                   requires={"engine": "pmemd.cuda"}, placeholder="0"),
            Option("min_engine", "Minimisation engine", "radio", "same",
                   "pmemd.cuda can stop on bad initial contacts; the CPU pmemd is more forgiving.",
                   choices=[("same", "Same as MD"), ("pmemd", "pmemd (CPU)")]),
            Option("input_velocities", "Input coordinates have velocities", "bool", False,
                   "Tick when the starting rst7 is a restart of an earlier MD run, so the first MD "
                   "stage can continue it (irest=1)."),
            Option("restraint_ref", "Positional restraint reference", "radio", "previous",
                   "Coordinates used as -ref for positional restraints.",
                   choices=[("previous", "Previous stage"), ("initial", "Starting coordinates")]),
            Option("separate_production", "Separate production script", "bool", True,
                   "Production stages go into their own script (e.g. to run them on a cluster)."),
            Option("resume", "Skip finished stages", "bool", True,
                   "When the script is re-run, stages whose output file already has pmemd's final "
                   "timing lines are skipped (resumes an interrupted run)."),
        ], tooltip="Which pmemd binary runs the stages and how."),
        Section("stages", "Stages", STAGE_OPTIONS, repeat=True, item_label="name",
                templates=_STAGE_TEMPLATES,
                default_items=("min1", "min2", "heat", "eq", "eq-free", "prod"),
                tooltip="Stages run in order; each starts from the previous stage's restart file."),
        Section("final", "Final structure", [
            Option("dry_pdb", "Dry PDB before production", "bool", True,
                   "cpptraj strips water and ions from the last restart of the main script (before "
                   "the production script) - for a visual check of the equilibrated structure."),
            Option("strip_mask", "Strip mask", "text", DEFAULT_STRIP, "cpptraj mask of atoms removed "
                   "in the dry PDB / trajectory.", requires={"dry_pdb": True}),
            Option("dry_traj", "Dry, imaged production trajectory", "bool", False,
                   "cpptraj autoimage + strip of the production trajectories (and the matching dry "
                   "topology) for visualisation."),
            Option("dry_mask", "Strip mask (trajectory)", "text", DEFAULT_STRIP,
                   "cpptraj mask removed from the dry trajectory.", requires={"dry_traj": True}),
        ]),
        env_section("source /opt/amber26/amber.sh   or   module load Amber/24"),
    ]

    # ----------------------------------------------------------- helpers
    @staticmethod
    def stage_tag(stage: Dict[str, Any], k: int) -> str:
        from .naming import sanitize_tag
        return sanitize_tag(stage.get("name") or f"stage{k + 1}")

    def output_ports(self, s):
        ports = [Port("topology", "Topology (parm7)", ("parm7",)),
                 Port("restart", "Final restart (rst7)", ("rst7",))]
        if any(st["kind"] == "md" and st["ntwx"] > 0 for st in s.get("stages", [])):
            ports.append(Port("trajectory", "Trajectory", ("nc",), multiple=True))
        if s.get("dry_pdb"):
            ports.append(Port("dry_pdb", "Dry final PDB", ("pdb",)))
        if s.get("dry_traj"):
            ports.append(Port("dry_topology", "Dry topology", ("parm7",)))
            ports.append(Port("dry_trajectory", "Dry trajectory", ("nc",)))
        return ports

    def _segments(self, s, ctx) -> List[Dict[str, Any]]:
        """Expand stages into runs: name, tag, stage, index, segment, files."""
        runs = []
        for k, st in enumerate(s["stages"]):
            tag = self.stage_tag(st, k)
            nseg = st["segments"] if st["kind"] == "md" else 1
            for seg in range(1, nseg + 1):
                rtag = f"{tag}-{seg:03d}" if nseg > 1 else tag
                runs.append({"stage": st, "index": k, "seg": seg, "nseg": nseg, "tag": tag, "rtag": rtag,
                             "out": ctx.name(rtag, "out"), "rst": ctx.name(rtag, "rst7"),
                             "nc": ctx.name(rtag, "nc") if st["kind"] == "md" and st["ntwx"] > 0 else "",
                             "inf": ctx.name(rtag, "mdinfo")})
        return runs

    def _restart_modes(self, s, ctx) -> List[Tuple[int, int]]:
        """(irest, ntx) for every stage (minimisation stages get (0, 1))."""
        has_vel = bool(ctx.data("coordinates").get("velocities", s["input_velocities"]))
        modes = []
        for st in s["stages"]:
            if st["kind"] == "min":
                modes.append((0, 1))
                has_vel = False
                continue
            mode = st["restart"]
            if mode == "auto":
                mode = "continue" if has_vel else "new"
            modes.append((1, 5) if mode == "continue" else (0, 1))
            has_vel = True
        return modes

    def _solute_mask(self, ctx) -> str:
        nres = ctx.data("topology").get("solute_residues")
        return f":1-{nres}" if nres else SOLVENT_MASK

    # ------------------------------------------------------------ checks
    def check(self, s, ctx):
        out: List[Issue] = []
        stages = s["stages"]
        if not stages:
            return [Issue("error", "add at least one stage", ("stages",))]
        tags = [self.stage_tag(st, k) for k, st in enumerate(stages)]
        for k, st in enumerate(stages):
            if tags.count(tags[k]) > 1:
                out.append(Issue("error", f"stage name '{tags[k]}' is used twice", (f"stages.{k}.name",)))
            if st["name"] and tags[k] != st["name"]:
                out.append(Issue("info", f"stage '{st['name']}' is written as '{tags[k]}' in file names",
                                 (f"stages.{k}.name",)))
        top = ctx.data("topology")
        if top.get("box") is False:
            out.append(Issue("error", "the topology has no periodic box; these stages need one (ntb>0)",
                             ("input:topology",)))
        needs = [k for k, st in enumerate(stages) if st["distres"]]
        if needs and not ctx.has("restraints"):
            out.append(Issue("error", f"stages {', '.join(tags[k] for k in needs)} use helix distance "
                                      "restraints but no DISANG file is connected or chosen (an 'Amber · "
                                      "helix restraints' step writes one)",
                             tuple(f"stages.{k}.distres" for k in needs) + ("input:restraints",)))
        modes = self._restart_modes(s, ctx)
        prev_kind = None
        temps = []
        for k, st in enumerate(stages):
            key = f"stages.{k}."
            label = f"stage {tags[k]}"
            if st["posres"] and st["posres_wt"] <= 0:
                out.append(Issue("error", f"{label}: positional restraints need a positive weight",
                                 (key + "posres_wt",)))
            if st["kind"] == "min":
                if st["ncyc"] > st["maxcyc"] and st["ntmin"] == "1":
                    out.append(Issue("warning", f"{label}: ncyc exceeds maxcyc, so conjugate gradient never "
                                                "starts", (key + "ncyc",)))
                prev_kind = "min"
                continue
            irest, _ntx = modes[k]
            if st["restart"] == "continue" and prev_kind == "min":
                out.append(Issue("error", f"{label}: cannot continue velocities after a minimisation "
                                          "(use Auto or New)", (key + "restart",)))
            if st["restart"] == "continue" and prev_kind is None and not (
                    ctx.data("coordinates").get("velocities", s["input_velocities"])):
                out.append(Issue("error", f"{label}: the starting coordinates have no velocities to "
                                          "continue (tick 'Input coordinates have velocities' or use New)",
                                 (key + "restart", "input_velocities")))
            if st["dt"] > 0.001 and not st["shake"]:
                out.append(Issue("error", f"{label}: dt > 0.001 ps needs SHAKE", (key + "dt", key + "shake")))
            if st["dt"] > 0.0025 and not top.get("hmr"):
                lvl = "warning" if top.get("hmr") is None else "error"
                out.append(Issue(lvl, f"{label}: dt = {st['dt']:g} ps needs hydrogen mass "
                                      "repartitioning in the topology", (key + "dt",)))
            if st["dt"] > 0.0045:
                out.append(Issue("error", f"{label}: dt above 0.004 ps is unstable", (key + "dt",)))
            if st["thermostat"] == "langevin" and st["gamma_ln"] <= 0:
                out.append(Issue("error", f"{label}: Langevin needs gamma_ln > 0", (key + "gamma_ln",)))
            if st["ensemble"] == "npt" and irest == 0 and st["tempi"] < 50:
                out.append(Issue("warning", f"{label}: heating from {st['tempi']:g} K at constant "
                                            "pressure can collapse the box; heat in NVT first",
                                 (key + "ensemble",)))
            if st["ramp"]:
                if st["ramp_steps"] > st["nstlim"]:
                    out.append(Issue("error", f"{label}: the ramp is longer than the stage",
                                     (key + "ramp_steps",)))
                if irest == 1:
                    out.append(Issue("warning", f"{label}: a ramp from tempi on a continued run starts "
                                                "below the current temperature", (key + "ramp",)))
            if irest == 1 and st["tempi"] not in (0.0, st["temp0"]) and st["restart"] != "new":
                out.append(Issue("info", f"{label}: tempi is ignored when velocities are continued",
                                 (key + "tempi",)))
            if st["ntwx"] == 0 and st["production"]:
                out.append(Issue("warning", f"{label}: production stage without trajectory output (ntwx=0)",
                                 (key + "ntwx",)))
            if st["ntwx"] and st["nstlim"] % st["ntwx"]:
                out.append(Issue("info", f"{label}: nstlim is not a multiple of ntwx", (key + "ntwx",)))
            if not st["netcdf"] and st["ntwx"]:
                out.append(Issue("warning", f"{label}: ASCII trajectories are large and slow; NetCDF "
                                            "(ioutfm=1) is recommended", (key + "netcdf",)))
            if st["ig"] not in (None, -1) and st["segments"] > 1 and st["thermostat"] == "langevin":
                out.append(Issue("warning", f"{label}: a fixed Langevin seed repeats the same random "
                                            "forces in every segment (use ig=-1)", (key + "ig",)))
            if st["ensemble"] == "npt" and st["barostat"] == "mc" and st["taup"] != 1.0:
                out.append(Issue("info", f"{label}: taup has no effect with the Monte Carlo barostat",
                                 (key + "taup",)))
            temps.append((k, tags[k], st["temp0"]))
            prev_kind = "md"
            dup = _duplicated_keys(st["extra_cntrl"])
            if dup:
                out.append(Issue("warning", f"{label}: extra &cntrl settings override generated ones: "
                                            f"{', '.join(dup)}", (key + "extra_cntrl",)))
        if len({t for _k, _n, t in temps}) > 1:
            listing = ", ".join(f"{n} {t:g} K" for _k, n, t in temps)
            out.append(Issue("warning", f"MD stages use different temperatures ({listing})",
                             tuple(f"stages.{k}.temp0" for k, _n, _t in temps)))
        if s["engine"] == "pmemd.MPI" and "{n}" not in s["mpi_launcher"] and str(s["nproc"]) not in s["mpi_launcher"]:
            out.append(Issue("warning", "the MPI launcher does not contain {n}; the process count is not "
                                        "passed", ("mpi_launcher",)))
        if s["engine"] == "pmemd" and any(st["kind"] == "md" and st["nstlim"] * st["dt"] > 10_000
                                          for st in stages):
            out.append(Issue("info", "single-core pmemd is slow for long runs; consider pmemd.cuda or "
                                     "pmemd.MPI", ("engine",)))
        if not any(st.get("production") for st in stages if st["kind"] == "md") and s["separate_production"]:
            out.append(Issue("info", "no stage is marked as production; everything runs in one script",
                             ("separate_production",)))
        if not env_has_amber(s):
            out.append(Issue("info", "no environment setup: pmemd and cpptraj must be on PATH when the "
                                     "script runs", ("env_setup",)))
        return out

    def describe(self, s):
        names = [self.stage_tag(st, k) for k, st in enumerate(s.get("stages", []))]
        return f"{s.get('engine')}: " + " → ".join(names)

    def output_values(self, s, ctx):
        runs = self._segments(s, ctx)
        out: Dict[str, PortValue] = {}
        top = ctx.value("topology")
        if top is not None and top.paths:
            out["topology"] = PortValue(list(top.paths), "parm7", dict(top.data))
        if runs:
            out["restart"] = PortValue([ctx.own(runs[-1]["rtag"], "rst7")], "rst7", {"velocities":
                                       runs[-1]["stage"]["kind"] == "md"})
        prod = [r for r in runs if r["nc"] and r["stage"]["production"]]
        if not prod:
            md = [r for r in runs if r["nc"]]
            prod = [r for r in md if r["index"] == md[-1]["index"]] if md else []
        if prod:
            out["trajectory"] = PortValue([ctx.from_own(r["nc"]) for r in prod], "nc")
        main_runs = [r for r in runs if not (s["separate_production"] and r["stage"]["production"])]
        if s["dry_pdb"] and main_runs:
            out["dry_pdb"] = PortValue([ctx.own("dry", "pdb")], "pdb")
        if s["dry_traj"] and prod:
            out["dry_topology"] = PortValue([ctx.own("dry", "parm7")], "parm7", {"box": False})
            out["dry_trajectory"] = PortValue([ctx.own("dry-prod", "nc")], "nc")
        return out

    # ------------------------------------------------------------- files
    def mdin(self, st: Dict[str, Any], irest: int, ntx: int, disang: str, solute_mask: str) -> str:
        cntrl: List[str] = []
        if st["kind"] == "min":
            cntrl.append(f"imin=1, maxcyc={st['maxcyc']}, ncyc={st['ncyc']},")
            cntrl.append(f"ntmin={st['ntmin']},")
            cntrl.append(f"ntb=1, ntp=0, cut={fortran_float(st['cut'])},")
            cntrl.append("ntc=1, ntf=1,")
        else:
            cntrl.append(f"imin=0, irest={irest}, ntx={ntx},")
            cntrl.append(f"nstlim={st['nstlim']}, dt={fortran_float(st['dt'])},")
            if st["shake"]:
                tol = f", tol={fortran_float(st['tol'])}" if st["tol"] is not None else ""
                cntrl.append(f"ntc=2, ntf=2{tol},")
            else:
                cntrl.append("ntc=1, ntf=1,")
            if st["ensemble"] == "npt":
                baro = f", barostat=2, mcbarint={st['mcbarint']}" if st["barostat"] == "mc" else ""
                taup = f", taup={fortran_float(st['taup'])}" if st["barostat"] == "berendsen" else ""
                wrap = ", iwrap=1" if st["iwrap"] else ""
                cntrl.append(f"ntb=2, ntp={st['ntp']}{baro}, pres0={fortran_float(st['pres0'])}{taup}, "
                             f"cut={fortran_float(st['cut'])}{wrap},")
            else:
                wrap = ", iwrap=1" if st["iwrap"] else ""
                cntrl.append(f"ntb=1, ntp=0, cut={fortran_float(st['cut'])}{wrap},")
            thermo = {"langevin": f"ntt=3, gamma_ln={fortran_float(st['gamma_ln'])}",
                      "berendsen": f"ntt=1, tautp={fortran_float(st['tautp'])}",
                      "andersen": f"ntt=2, vrand={st['vrand']}"}[st["thermostat"]]
            tempi = f", tempi={fortran_float(st['tempi'])}" if irest == 0 else ""
            temp0 = st["tempi"] if st["ramp"] else st["temp0"]
            seed = f", ig={st['ig']}" if st["ig"] is not None else ""
            cntrl.append(f"{thermo}{tempi}, temp0={fortran_float(temp0)}{seed},")
        if st["posres"]:
            mask = (st["posres_mask"] or "").strip() or solute_mask
            cntrl.append(f"ntr=1, restraint_wt={fortran_float(st['posres_wt'])},")
            cntrl.append(f"restraintmask='{mask}',")
        else:
            cntrl.append("ntr=0,")
        nmr = st["distres"] or (st["kind"] == "md" and st["ramp"])
        if nmr:
            cntrl.append("nmropt=1,")
        if st["kind"] == "min":
            cntrl.append(f"ntpr={st['ntpr']},")
        else:
            cntrl.append(f"ntpr={st['ntpr']}, ntwx={st['ntwx']}, ntwr={st['ntwr']},")
            if st["netcdf"]:
                cntrl.append("ioutfm=1,")
        for line in extra_lines(st["extra_cntrl"]):
            line = line.strip()
            cntrl.append(line if line.endswith(",") else line + ",")
        lines = [st["title"] or st["name"] or "stage", "&cntrl"] + [f"  {c}" for c in cntrl] + ["/"]
        lines += extra_lines(st["extra_namelists"])
        if nmr:
            if st["kind"] == "md" and st["ramp"]:
                n1 = st["ramp_steps"]
                lines.append(f"&wt type='TEMP0', istep1=0, istep2={n1}, value1={fortran_float(st['tempi'])}, "
                             f"value2={fortran_float(st['temp0'])} /")
                if n1 < st["nstlim"]:
                    lines.append(f"&wt type='TEMP0', istep1={n1 + 1}, istep2={st['nstlim']}, "
                                 f"value1={fortran_float(st['temp0'])}, value2={fortran_float(st['temp0'])} /")
            lines.append("&wt type='END' /")
            if st["distres"]:
                lines.append(f"DISANG={disang}")
                if st["listout"]:
                    lines.append("LISTOUT=POUT")
        return "\n".join(lines) + "\n"

    def generate(self, s, ctx):
        files: List[GeneratedFile] = []
        runs = self._segments(s, ctx)
        modes = self._restart_modes(s, ctx)
        disang = ctx.inp("restraints")
        solute = self._solute_mask(ctx)
        top, crd = ctx.inp("topology"), ctx.inp("coordinates")
        # one mdin per stage (plus one for the first segment when its restart mode differs)
        mdin_of: Dict[Tuple[int, int], str] = {}
        for k, st in enumerate(s["stages"]):
            irest, ntx = modes[k]
            tag = self.stage_tag(st, k)
            name = ctx.name(tag, "in")
            files.append(GeneratedFile(name, self.mdin(st, irest, ntx, disang, solute), "input",
                                       f"pmemd input of stage {tag}"))
            nseg = st["segments"] if st["kind"] == "md" else 1
            for seg in range(1, nseg + 1):
                mdin_of[(k, seg)] = name
            if nseg > 1 and irest == 0:
                cont = ctx.name(f"{tag}-cont", "in")
                files.append(GeneratedFile(cont, self.mdin(st, 1, 5, disang, solute), "input",
                                           f"pmemd input of stage {tag}, segments 2-{nseg}"))
                for seg in range(2, nseg + 1):
                    mdin_of[(k, seg)] = cont
        engine = s["engine"]
        if engine == "pmemd.MPI":
            launcher = s["mpi_launcher"].replace("{n}", str(s["nproc"])).strip()
            pmemd = f"{launcher} pmemd.MPI".strip()
        else:
            pmemd = engine
        min_pmemd = "pmemd" if s["min_engine"] == "pmemd" else "$PMEMD"
        prod_split = s["separate_production"] and any(r["stage"]["production"] for r in runs)

        def preamble(title, script):
            lines = script_header(title, ctx, s, script, [
                f"Amber {s['amber_version']}; engine {engine}."])
            if engine == "pmemd.cuda" and s["gpu"].strip():
                lines += [f"export CUDA_VISIBLE_DEVICES={s['gpu'].strip()}", ""]
            lines += [f'PMEMD="{pmemd}"', f'TOP="{top}"', ""]
            if s["resume"]:
                lines += ["# a stage counts as finished when its output has the final timing lines",
                          '# (pmemd: "Total wall time", sander: "Run   done at")',
                          'finished() { [ -s "$1" ] && grep -qE "Total wall time|Run +done at" "$1"; }', ""]
            return lines

        main_name = ctx.name(self.tag, "sh")
        prod_name = ctx.name("amber-prod", "sh")
        main = preamble("Amber MD stages", main_name)
        prod = preamble("Amber production", prod_name) if prod_split else main
        prev_rst = crd
        for r in runs:
            st = r["stage"]
            target = prod if (prod_split and st["production"]) else main
            ref = crd if s["restraint_ref"] == "initial" else prev_rst
            exe = min_pmemd if st["kind"] == "min" else "$PMEMD"
            seg = f" (segment {r['seg']}/{r['nseg']})" if r["nseg"] > 1 else ""
            cmd = (f'{exe} -O -i {mdin_of[(r["index"], r["seg"])]} -p "$TOP" -c {prev_rst} -ref {ref} '
                   f'-o {r["out"]} -r {r["rst"]}' + (f" -x {r['nc']}" if r["nc"] else "") +
                   f' -inf {r["inf"]}')
            target.append(f"# {r['rtag']}{seg}: {st['title'] or st['name']}")
            if s["resume"]:
                target += [f"if finished {r['out']}; then",
                           f'  echo "{r["rtag"]}: already finished - skipping"',
                           "else", f"  {cmd}", "fi"]
            else:
                target.append(cmd)
            target += [require_file(r["rst"], f"pmemd ({r['rtag']})"), ""]
            prev_rst = r["rst"]
        main_runs = [r for r in runs if not (prod_split and r["stage"]["production"])]
        if main_runs and s["dry_pdb"]:
            dry = ctx.name("dry", "pdb")
            main += [f"# Strip a dry PDB of {main_runs[-1]['rtag']} for visual checks",
                     "cpptraj << EOF", 'parm "$TOP"', f"trajin {main_runs[-1]['rst']}",
                     f"strip {re.sub(r'[ ]+', '', s['strip_mask'])}", f"trajout {dry} pdb", "run",
                     "EOF", require_file(dry, "cpptraj"), ""]
        prod_runs = [r for r in runs if r["nc"] and r["stage"]["production"]]
        if s["dry_traj"] and prod_runs:
            last = prod if prod_split else main
            mask = re.sub(r"[ ]+", "", s["dry_mask"])
            last += ["# Dry, imaged copy of the production trajectory (and its topology)",
                     "cpptraj << EOF", 'parm "$TOP"'] + [f"trajin {r['nc']}" for r in prod_runs] + [
                "autoimage", f"strip {mask} parmout {ctx.name('dry', 'parm7')}",
                f"trajout {ctx.name('dry-prod', 'nc')} netcdf", "run", "EOF",
                require_file(ctx.name("dry-prod", "nc"), "cpptraj"), ""]
        files.append(GeneratedFile(main_name, "\n".join(main), "run", "runs the MD stages"))
        if prod_split:
            files.append(GeneratedFile(prod_name, "\n".join(prod), "run-separate",
                                       "runs the production stages"))
        return files


def env_has_amber(s: Dict[str, Any]) -> bool:
    return bool(s.get("conda_env", "").strip() or s.get("env_setup", "").strip())


def _duplicated_keys(extra: str) -> List[str]:
    generated = {"imin", "irest", "ntx", "nstlim", "dt", "ntc", "ntf", "tol", "ntb", "ntp", "pres0",
                 "taup", "barostat", "mcbarint", "cut", "ntt", "gamma_ln", "tautp", "vrand", "tempi",
                 "temp0", "ig", "ntr", "restraint_wt", "restraintmask", "nmropt", "ntpr", "ntwx", "ntwr",
                 "ioutfm", "iwrap", "maxcyc", "ncyc", "ntmin"}
    keys = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", extra or "")
    return sorted({k for k in keys if k.lower() in generated})
