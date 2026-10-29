"""NAMD MDFF on the Amber topology: flexible fitting into a density map without converting the force field.

NAMD reads the parm7 written by the Amber build natively (``amber on``), so the CP / L-RNA atom types,
the L-RNA dihedrals, every off-diagonal Lennard-Jones pair (NBfix19, 0BPh, microMg / nanoMg) and HMR
masses are used exactly as Amber uses them.  Before fitting, a single-point energy comparison with
sander proves it for the topology at hand.

Restraints are L-RNA-safe and written by MDATK (scripts/namd_mdff.py): chirality impropers with the
target of each residue's own hand, flat-bottom base-pair colvars, positional restraints.  VMD's
``chirality`` / ``ssrestraints`` / ``cispeptide`` tools assume D-nucleotides and are never used.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .. import PROJECT_ABBREV, __version__
from . import rna
from .amber import WATERS, secondary_structure
from .base import PortValue, Tool, env_section, fortran_float, require_file, script_header, shell_quote
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section
from .scripts import embed
from .scripts import namd_energy as _namd_energy
from .scripts import namd_mdff as _namd_mdff

# Improper dihedral (centre, partner 1-3; scripts/namd_mdff.SUGAR_CENTRES) of each ribose centre of a
# D-nucleotide, degrees, measured on MDATK's C3'-endo nucleotide from PDB 4JRD (mdatk.builder); an L
# residue gets the mirror image (the negative).  The same for every base.
D_IMPROPERS = {"C1'": -35.01, "C2'": 40.03, "C3'": 38.30, "C4'": 33.31}
FOUR_SITE = {"opc", "tip4pew", "tip4pd", "tip4p2005"}
_FILE_CHECKS: Dict[tuple, List[Issue]] = {}

STAGE_OPTIONS = [
    Option("name", "Name", "text", "stage", "Short stage name used in file names (letters, digits, hyphens)."),
    Option("kind", "Type", "radio", "md", "Energy minimisation (NAMD minimize) or molecular dynamics (run).",
           choices=[("min", "Minimisation"), ("md", "MD")]),
    Option("steps", "Steps", "int", 50000, "Minimisation steps, or MD steps of the whole stage (all segments).",
           minimum=1, maximum=1_000_000_000),
    Option("gscale", "Map coupling ξ", "float", 0.3,
           "mgridforcescale ξ (kcal/mol per amu): the force on an atom is ξ × its mass × the gradient of the MDFF "
           "potential (0 at the strongest density, 1 at the threshold). 0.3 is the default of VMD's mdff setup; a "
           "strong coupling such as 10 in a final minimisation pulls the atoms into the density.",
           minimum=0.0, maximum=10000.0, step=0.05, digits=3),
    Option("gscale_end", "Map coupling ξ at the end", "float", None,
           "MD only: ξ changes linearly from the start value to this one over the segments (empty: constant).",
           optional=True, minimum=0.0, maximum=10000.0, digits=3, requires={"kind": "md"}),
    Option("segments", "Segments", "int", 1,
           "MD only: NAMD runs the stage is split into, each restarting from the previous one; a ξ ramp moves "
           "one step per segment.", minimum=1, maximum=1000, requires={"kind": "md"}),
    Option("chirality", "Chirality restraints", "bool", True,
           "L-RNA-safe impropers at C1' C2' C3' C4' (each residue keeps its own hand)."),
    Option("ssdihedrals", "Secondary-structure dihedrals", "bool", True,
           "α β γ δ ε ζ χ of the base-paired residues held at their starting values, the RNA restraints of "
           "Trabuco et al. (2008); measured on the model itself, so L residues keep their own values."),
    Option("basepairs", "Base-pair restraints", "bool", True,
           "Flat-bottom hydrogen-bond distance restraints from the secondary structure (Restraints page)."),
    Option("positional", "Positional restraints", "bool", False,
           "Harmonic restraints to the starting coordinates (atoms and force constant on the Restraints page)."),
    Option("dcdfreq", "Trajectory frame every", "int", 5000, "dcdfreq (MD steps); 0: no trajectory.",
           minimum=0, maximum=100_000_000, requires={"kind": "md"}),
    Option("energyfreq", "Energies every", "int", 1000, "outputEnergies (steps).", minimum=1, maximum=100_000_000),
]
_STAGE_TEMPLATES = [
    ("min", "Minimisation (ξ 0.3)", dict(name="min", kind="min", steps=2000, gscale=0.3)),
    ("ramp", "MD, map coupling ramped 0.05 → 0.3", dict(name="ramp", kind="md", steps=50000, gscale=0.05,
                                                        gscale_end=0.3, segments=5)),
    ("fit", "MDFF (ξ 0.3)", dict(name="fit", kind="md", steps=250000, gscale=0.3)),
    ("final-min", "Final minimisation (ξ 10)", dict(name="final-min", kind="min", steps=2000, gscale=10.0)),
]


@register
class NamdMDFFTool(Tool):
    name = "namd-mdff"
    label = "NAMD · MDFF (Amber topology)"
    category = "Refinement"
    tag = "namd-mdff"
    description = ("Flexible fitting into a density map with NAMD reading the Amber topology natively: CP, "
                   "NBfix19, 0BPh, L-RNA types and dihedrals, Mg²⁺ pair terms and HMR stay exactly as built. "
                   "L-RNA-safe chirality / base-pair restraints (never VMD's D-only stereochemistry tools); "
                   "an energy comparison with sander runs first. Drop it after an Amber build (or MD) step; "
                   "steps already in the plan are not changed.")
    inputs = [Port("topology", "Topology (parm7)", ("parm7",),
                   tooltip="The Amber topology as built (solvated for PME runs, the dry one for vacuum runs)."),
              Port("coordinates", "Coordinates (rst7 / PDB)", ("rst7", "pdb"),
                   tooltip="Coordinates of the same system: tleap output or a pmemd restart (NetCDF restarts are "
                           "converted with cpptraj), or a PDB in topology atom order - for vacuum runs the dry PDB "
                           "of the Amber build."),
              Port("map", "Density map", ("map",), tooltip="MRC / CCP4 map (or an OpenDX density)."),
              Port("model", "Map-frame model (PDB)", ("pdb",), optional=True,
                   tooltip="The PDB that was fitted into the map (e.g. the input of the Amber build). tleap moves "
                           "the solute when it solvates it; the model gives the map's frame back."),
              Port("secstruct", "Secondary structure", ("secstruct",), optional=True,
                   tooltip="Dot-bracket for the base-pair restraints (or type it on the Restraints page).")]
    outputs = [Port("fitted", "Fitted model (PDB, map frame)", ("pdb",))]
    sections = [
        Section("system", "System", [
            Option("solvent", "Environment", "radio", "solvated",
                   "Solvated: the periodic box of the Amber build with PME, as in the Amber MD runs. Vacuum: the "
                   "dry topology without a box (the classic MDFF setting; optionally GBIS implicit solvent).",
                   choices=[("solvated", "Solvated (periodic, PME)"), ("vacuum", "Vacuum (dry topology)")]),
            Option("cutoff", "Cutoff", "float", 10.0,
                   "Non-bonded cutoff (Å); 10 Å matches MDATK's Amber MD stages. Amber settings: no switching, "
                   "scaled 1-4 terms (1/1.2 electrostatics, 1/2 van der Waals), long-range LJ correction with PME.",
                   minimum=6.0, maximum=999.0, step=0.5, digits=1, unit="Å"),
            Option("timestep", "Time step", "radio", "auto",
                   "Automatic: 4 fs when the topology has repartitioned hydrogen masses (HMR), else 2 fs. All bonds "
                   "to hydrogen are rigid (SHAKE / SETTLE).",
                   choices=[("auto", "Automatic"), ("1", "1 fs"), ("2", "2 fs"), ("4", "4 fs (needs HMR)")]),
            Option("temperature", "Temperature", "float", 300.0, "Langevin thermostat temperature (K).",
                   minimum=1.0, maximum=1000.0, digits=1, unit="K"),
            Option("damping", "Langevin damping", "float", 1.0, "langevinDamping (1/ps).", minimum=0.01,
                   maximum=100.0, digits=2),
            Option("gbis", "GBIS implicit solvent", "bool", False,
                   "Vacuum runs only: NAMD's generalised Born implicit solvent instead of plain vacuum.",
                   requires={"solvent": "vacuum"}),
            Option("ion_conc", "Ion concentration (GBIS)", "float", 0.15, "ionConcentration (M).", minimum=0.0,
                   maximum=5.0, digits=3, requires={"solvent": "vacuum", "gbis": True}),
        ], tooltip="How NAMD treats the Amber system."),
        Section("map", "Density map", [
            Option("threshold", "Density threshold", "float", 0.0,
                   "Densities at or below this value give no force (VMD mdff griddx -threshold, default 0). Trabuco "
                   "et al. (2008) take the minimum between the solvent and the biomolecule peaks of the density "
                   "histogram, or the map's average when the peaks are not resolved.", step=0.01, digits=4),
            Option("couple", "Atoms coupled to the map", "radio", "heavy",
                   "Every coupled atom feels the map in proportion to its (standard) mass.",
                   choices=[("heavy", "RNA heavy atoms"), ("all", "All RNA atoms, with hydrogens")]),
            Option("couple_ions", "Couple ions too", "bool", False, "Also couple Mg²⁺ and other ions to the map."),
            Option("pad", "Margin around the RNA", "float", 8.0,
                   "The potential is cropped to the coupled atoms plus this margin (Å); it must fit into the "
                   "periodic cell.", minimum=0.0, maximum=100.0, digits=1, unit="Å"),
            Option("min_overlap", "Required overlap", "float", 0.3,
                   "Share of coupled atoms that must lie on density above the map's mean + 1 σ, a check that the "
                   "model is in the map's frame.", minimum=0.0, maximum=1.0, step=0.05, digits=2),
            Option("max_frame_rmsd", "Largest model / topology RMSD", "float", 2.0,
                   "RMSD (Å) allowed between the map-frame model and the coordinates after superposition.",
                   minimum=0.01, maximum=50.0, digits=2, unit="Å"),
            Option("resample_divisor", "Resampling refinement", "int", 2,
                   "If the solute was rotated (not only moved) the potential is resampled on a grid this many "
                   "times finer than the map.", minimum=1, maximum=4),
        ], tooltip="The MDFF potential made from the map."),
        Section("restraints", "Restraints", [
            Option("chirality_k", "Chirality force constant", "float", 200.0,
                   "Improper restraints at C1' C2' C3' C4' (kcal/mol/rad²); the targets have the sign of each "
                   "residue's own hand, so L sugars are kept L.", minimum=0.0, maximum=10000.0, digits=1),
            Option("ss_k", "Dihedral force constant", "float", 200.0,
                   "Secondary-structure dihedral restraints (kcal/mol/rad², E = k (θ − θ₀)²). Trabuco et al. (2008) "
                   "use 200 and discourage softening them: they are what prevents overfitting.",
                   minimum=0.0, maximum=10000.0, digits=1),
            Option("sequence", "Sequence", "multiline", "",
                   "For a typed secondary structure: RNA sequence in topology order, chains separated by spaces.",
                   placeholder="GGGCAC GUGCCC"),
            Option("dotbracket", "Secondary structure", "multiline", "",
                   "Dot-bracket; ignored when a secondary structure is connected.", placeholder="(((((( ))))))"),
            Option("res_offset", "Residue offset", "int", 0,
                   "Topology residue number = position in the sequence + offset.", minimum=-100000, maximum=100000),
            Option("wobble", "Restrain G·U wobble pairs", "bool", False, "Also restrain G·U pairs."),
            Option("bp_lower", "Flat bottom from", "float", 2.7, "Lower wall of each H-bond distance (Å).",
                   minimum=0.0, maximum=20.0, digits=2, unit="Å"),
            Option("bp_upper", "Flat bottom to", "float", 3.2, "Upper wall (Å).", minimum=0.0, maximum=20.0,
                   digits=2, unit="Å"),
            Option("bp_k", "Wall force constant", "float", 10.0,
                   "k in E = k (d − wall)² beyond the walls (kcal/mol/Å², as rk2 / rk3 of Amber restraints).",
                   minimum=0.0, maximum=1000.0, digits=2),
            Option("positional_atoms", "Positional restraints on", "radio", "backbone",
                   "Atoms restrained to the starting coordinates in stages with positional restraints.",
                   choices=[("backbone", "RNA backbone (P O5' C5' C4' C3' O3')"), ("heavy", "RNA heavy atoms")]),
            Option("positional_k", "Positional force constant", "float", 1.0, "kcal/mol/Å².", minimum=0.0,
                   maximum=1000.0, digits=2),
        ], tooltip="L-RNA-safe restraints written by MDATK (never VMD's chirality / ssrestraints)."),
        Section("stages", "Stages", STAGE_OPTIONS, repeat=True, item_label="name", templates=_STAGE_TEMPLATES,
                default_items=("min", "ramp", "fit", "final-min"),
                tooltip="Stages run in order, each from the previous one's coordinates."),
        Section("engine", "Engine", [
            Option("namd", "NAMD program", "text", "namd3",
                   "NAMD 3.0 (namd3) or 2.14 (namd2). The runs use the classic (not the GPU-resident, "
                   "CUDASOAintegrate) mode: grid forces are not known to work there.", placeholder="namd3"),
            Option("cores", "CPU cores", "int", 8, "+p for multicore NAMD builds.", minimum=1, maximum=100000),
            Option("gpu", "GPU (+devices)", "text", "", "GPU index for CUDA builds (empty: no +devices).",
                   placeholder="0"),
            Option("namd_args", "Extra NAMD arguments", "text", "", "Added to every NAMD command line.",
                   placeholder="+setcpuaffinity"),
            Option("energy_check", "Compare energies with sander first", "bool", True,
                   "Single-point energies of the starting coordinates from sander and NAMD must agree term by term "
                   "(bonds, angles, torsions, van der Waals incl. the NBfix / Mg²⁺ pairs, electrostatics) before "
                   "the fitting starts."),
            Option("sander", "sander program", "text", "sander", "AmberTools sander for the energy comparison.",
                   requires={"energy_check": True}),
            Option("resume", "Skip finished runs", "bool", True,
                   "When the script is run again, runs whose log shows NAMD finished are skipped."),
        ], tooltip="How the NAMD runs are started."),
        env_section("module load NAMD/3.0   and   module load Amber/24   (sander, cpptraj)"),
    ]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def stage_tag(stage: Dict[str, Any], k: int) -> str:
        from .naming import sanitize_tag
        return sanitize_tag(stage.get("name") or f"stage{k + 1}")

    def runs(self, s, ctx) -> List[Dict[str, Any]]:
        """Stages expanded into NAMD runs (segments), with the coupling of each run."""
        out = []
        for k, st in enumerate(s["stages"]):
            tag = self.stage_tag(st, k)
            nseg = st["segments"] if st["kind"] == "md" else 1
            end = st["gscale_end"] if st["kind"] == "md" and st["gscale_end"] is not None else st["gscale"]
            steps = max(1, st["steps"] // nseg)
            for seg in range(nseg):
                frac = seg / (nseg - 1) if nseg > 1 else 0.0
                rtag = f"{tag}-{seg + 1:03d}" if nseg > 1 else tag
                out.append({"stage": st, "index": k, "seg": seg, "tag": rtag, "steps": steps,
                            "gscale": st["gscale"] + (end - st["gscale"]) * frac,
                            "conf": ctx.name(rtag, "namd"), "out": ctx.name(rtag), "log": ctx.name(rtag, "log")})
        return out

    def timestep(self, s, ctx) -> float:
        if s["timestep"] != "auto":
            return float(s["timestep"])
        return 4.0 if ctx.data("topology").get("hmr") else 2.0

    def pairs(self, s, ctx) -> Tuple[List[Tuple[int, int, str]], List[Issue], str]:
        model, source, issues = secondary_structure(s, ctx)
        if model is None:
            return [], issues, source
        keep, skip = rna.classify_pairs(model, s["wobble"])
        if skip:
            issues.append(Issue("info", f"{len(skip)} base pair(s) of the secondary structure are not restrained "
                                        f"({', '.join(sorted({k for _i, _j, k in skip}))})", ("wobble",)))
        return keep, issues, source

    def paired_residues(self, s, ctx) -> List[int]:
        """Topology residue numbers of every base-paired residue (all pair kinds) of the secondary structure."""
        model, _source, _issues = secondary_structure(s, ctx)
        if model is None:
            return []
        keep, skip = rna.classify_pairs(model, True)
        return sorted({p + s["res_offset"] for i, j, _k in keep + skip for p in (i, j)})

    def uses(self, s, key: str) -> bool:
        return any(st[key] for st in s["stages"])

    def prep_settings(self, s, ctx) -> Dict[str, Any]:
        n = ctx.name
        pairs, _issues, _src = self.pairs(s, ctx)
        paired = self.paired_residues(s, ctx)
        bonds = {k: v for k, v in rna.HBOND_ATOMS.items() if any(p[2] == k for p in pairs)}
        k2 = 2.0 * s["bp_k"]  # Colvars walls: E = k/2 (d - wall)^2, Amber-style input: E = k (d - wall)^2
        return {
            "topology": ctx.inp("topology"), "coordinates": n("start", "rst7"), "model": ctx.inp("model"),
            "map": ctx.inp("map"), "periodic": s["solvent"] == "solvated",
            "four_site": s["solvent"] == "solvated" and bool(ctx.data("topology").get("four_site_water")),
            "timestep": self.timestep(s, ctx), "threshold": s["threshold"], "pad": s["pad"],
            "couple": s["couple"], "couple_ions": s["couple_ions"], "min_overlap": s["min_overlap"],
            "max_frame_rmsd": s["max_frame_rmsd"], "resample_divisor": s["resample_divisor"],
            "potential": n("potential", "dx"), "grid_pdb": n("coupling", "pdb"), "restraint_pdb": n("posres", "pdb"),
            "extrabonds": n("chirality", "txt"), "colvars": n("basepairs", "colvars"), "xsc": n("cell", "xsc"),
            "prep_json": n("prep", "json"), "final_pdb": n("fitted", "pdb"),
            "final_coor": self.runs(s, ctx)[-1]["out"] + ".coor" if s["stages"] else "",
            "chirality": {"enabled": self.uses(s, "chirality"), "k": s["chirality_k"], "targets": dict(D_IMPROPERS)},
            "basepairs": {"enabled": self.uses(s, "basepairs") and bool(pairs), "pairs": [list(p) for p in pairs],
                          "bonds": {k: [list(b) for b in v] for k, v in bonds.items()}, "offset": s["res_offset"],
                          "walls": [s["bp_lower"], s["bp_upper"], k2, k2], "traj_frequency": 0},
            "positional": {"selection": s["positional_atoms"] if self.uses(s, "positional") else "none",
                           "k": s["positional_k"]},
            "ssdihedrals": {"enabled": self.uses(s, "ssdihedrals") and bool(paired), "k": s["ss_k"],
                            "residues": paired, "file": n("ssdihedrals", "txt")},
        }

    # ----------------------------------------------------------------- planner
    def output_values(self, s, ctx):
        return {"fitted": PortValue([ctx.own("fitted", "pdb")], "pdb", {"map_frame": True})}

    def resources(self, s):
        out = ["namd", "mdff"]
        if self.uses(s, "basepairs") and (s.get("dotbracket") or "").strip():
            out.append("colvars")
        if s["energy_check"]:
            out.append("ambertools")
        return out

    def step_resources(self, s, ctx):
        out = [r for r in self.resources(s) if r != "colvars"]
        if ctx is not None and self.uses(s, "basepairs") and self.pairs(s, ctx)[0]:
            out.insert(2, "colvars")  # Colvars only when base-pair restraints are written
        return out

    def describe(self, s):
        gs = [st["gscale"] for st in s["stages"]]
        env = "solvated" if s["solvent"] == "solvated" else ("GBIS" if s["gbis"] else "vacuum")
        span = f", ξ {min(gs):g}–{max(gs):g}" if gs else ""
        return f"NAMD MDFF on the Amber topology, {env}, {len(s['stages'])} stage(s){span}"

    def check(self, s, ctx):
        out: List[Issue] = []
        data = ctx.data("topology")
        solvated = s["solvent"] == "solvated"
        if not s["stages"]:
            out.append(Issue("error", "add at least one stage", ("stages",)))
        tags = [self.stage_tag(st, k) for k, st in enumerate(s["stages"])]
        dup = sorted({t for t in tags if tags.count(t) > 1})
        if dup:
            out.append(Issue("error", f"stage names must differ ({', '.join(dup)})", ("stages",)))
        if data.get("c4"):
            out.append(Issue("error", "the topology carries 12-6-4 C4 terms (Li & Merz 12-6-4 Mg²⁺); NAMD cannot apply "
                                      "them. Build with the 12-6, microMg or nanoMg model for NAMD.", ("input:topology",)))
        crd = ctx.data("coordinates")
        if "box" in crd and "box" in data and crd["box"] != data["box"]:
            out.append(Issue("error", "the coordinates belong to the "
                                      + ("solvated" if crd["box"] else "dry") + " system, the topology to the "
                                      + ("solvated" if data["box"] else "dry") + " one"
                                      + ("; for vacuum runs connect the dry PDB of the Amber build" if not solvated
                                         else ""), ("input:coordinates",)))
        if "box" in data:
            if solvated and not data["box"]:
                out.append(Issue("error", "the connected topology has no box: choose 'Vacuum' or connect the "
                                          "solvated topology", ("solvent",)))
            if not solvated and data["box"]:
                out.append(Issue("error", "vacuum runs need the dry topology (Amber build: 'Two-pass build' gives "
                                          "it) - the connected one is solvated", ("solvent",)))
        if solvated and data.get("four_site_water"):
            water = WATERS.get(data.get("water", ""), (data.get("water", "four-site water"),))[0]
            out.append(Issue("warning", f"{water} is a four-site water: NAMD runs it with 'watermodel tip4' from the "
                                        "Amber extra points, which MDATK has not verified - only trust the run if the "
                                        "energy comparison with sander passes", ("energy_check",)))
        if data.get("hmr") and s["timestep"] in ("1", "2"):
            out.append(Issue("info", "the topology has HMR masses; 4 fs would be possible", ("timestep",)))
        if s["timestep"] == "4" and "hmr" in data and not data["hmr"]:
            out.append(Issue("error", "4 fs needs hydrogen mass repartitioning (HMR) in the Amber build",
                             ("timestep",)))
        if s["timestep"] == "auto" and data:
            out.append(Issue("info", f"time step {self.timestep(s, ctx):g} fs "
                                     f"({'HMR topology' if data.get('hmr') else 'no HMR'})", ("timestep",)))
        if data.get("nbfix") or data.get("mg_pairs"):
            parts = [p for p, on in (("NBfix19 / 0BPh pairs", data.get("nbfix")),
                                     (f"{data.get('mg_pairs')} Mg²⁺ pair terms", data.get("mg_pairs"))) if on]
            out.append(Issue("info", f"NAMD reads the {' and '.join(parts)} from the topology's Lennard-Jones table; "
                                     "nothing is converted", ("energy_check",)))
        if not s["energy_check"]:
            out.append(Issue("warning", "without the energy comparison nothing confirms that NAMD reproduces the "
                                        "Amber force field of this topology", ("energy_check",)))
        if not ctx.has("model"):
            if solvated:
                out.append(Issue("warning", "no map-frame model: tleap moves the solute when it solvates it, so the "
                                            "coordinates are usually not in the map's frame. Connect the PDB that was "
                                            "fitted into the map (the input of the Amber build); the script stops if "
                                            "the atoms miss the density", ("input:model",)))
            else:
                out.append(Issue("info", "no map-frame model: the coordinates are used in their own frame, which "
                                         "must be the map's (true for a dry tleap build of a fitted PDB)",
                                 ("input:model",)))
        if self.uses(s, "ssdihedrals"):
            paired = self.paired_residues(s, ctx)
            if paired:
                out.append(Issue("info", f"{len(paired)} base-paired residues: their backbone and χ dihedrals are held "
                                         "at their starting values (Trabuco et al. 2008)", ("ss_k",)))
            elif not self.uses(s, "basepairs"):
                out.append(Issue("warning", "stages use secondary-structure dihedral restraints but there is no "
                                            "secondary structure: none are written", ("dotbracket",)))
            if 0 < s["ss_k"] < 200.0:
                out.append(Issue("info", "Trabuco et al. (2008) discourage softening their 200 kcal/mol/rad² "
                                         "restraints: weaker ones let the model overfit the map", ("ss_k",)))
        if self.uses(s, "basepairs"):
            pairs, issues, source = self.pairs(s, ctx)
            out += issues
            if not pairs and not any(i.level == "error" for i in issues):
                out.append(Issue("warning", "stages use base-pair restraints but there is no secondary structure "
                                            "(connect one or type it on the Restraints page): none are written",
                                 ("dotbracket",)))
            elif pairs:
                out.append(Issue("info", f"{len(pairs)} base pairs restrained ({source})", ("dotbracket",)))
            if s["bp_lower"] > s["bp_upper"]:
                out.append(Issue("error", "the lower wall must not be above the upper wall", ("bp_lower", "bp_upper")))
        if not self.uses(s, "chirality") and any(st["gscale"] > 1.0 for st in s["stages"]):
            out.append(Issue("warning", "strong map coupling without chirality restraints can invert sugar centres "
                                        "(of D and L residues alike)", ("stages",)))
        for k, st in enumerate(s["stages"]):
            if st["kind"] == "md" and st["segments"] > st["steps"]:
                out.append(Issue("error", f"stage {k + 1}: more segments than steps", (f"stages.{k}.segments",)))
        out.append(Issue("info", "restraints are written by MDATK and keep every residue's own hand; do not add "
                                 "VMD's chirality / ssrestraints / cispeptide restraints, which assume D-nucleotides "
                                 "and would push L sugars towards the inverted centre", ("chirality_k",)))
        out += self._file_checks(s, ctx)
        return out

    def _file_checks(self, s, ctx) -> List[Issue]:
        """When topology and coordinates already exist: the script's own topology checks, now (cached per
        file version: the planner re-checks after every edit and a solvated parm7 takes a moment to read)."""
        top_path, crd_path = ctx.existing("topology"), ctx.existing("coordinates")
        if not top_path or not crd_path or getattr(ctx, "preview", False):
            return []
        cfg = self.prep_settings(s, ctx)
        cfg.update(topology=top_path, coordinates=crd_path)
        try:
            stamp = tuple((os.path.getmtime(p), os.path.getsize(p)) for p in (top_path, crd_path))
        except OSError:
            return []
        key = (top_path, crd_path, stamp, cfg["periodic"], cfg["four_site"], cfg["timestep"],
               cfg["chirality"]["enabled"], cfg["ssdihedrals"]["enabled"], tuple(cfg["ssdihedrals"]["residues"]))
        if key not in _FILE_CHECKS:
            if len(_FILE_CHECKS) > 32:
                _FILE_CHECKS.clear()
            _FILE_CHECKS[key] = self._read_and_check(cfg, top_path, crd_path)
        return list(_FILE_CHECKS[key])

    @staticmethod
    def _read_and_check(cfg, top_path, crd_path) -> List[Issue]:
        try:
            top = _namd_mdff.Topology(top_path)
            problems, _notes = _namd_mdff.topology_problems(top, cfg)
            xyz = None
            if _namd_mdff.is_netcdf(crd_path):
                pass  # converted by cpptraj when the script runs
            elif _namd_mdff.is_pdb(crd_path):
                atoms = _namd_mdff.read_pdb_atoms(crd_path)
                if len(atoms) != top.natom:
                    problems.append(f"{os.path.basename(crd_path)} has {len(atoms)} atoms, the topology {top.natom}")
                else:
                    xyz = [a[3] for a in atoms]
            else:
                xyz, _box, _vel = _namd_mdff.read_rst7(crd_path, top.natom)
            if xyz is not None:
                if cfg["chirality"]["enabled"]:
                    _lines, bad, _more = _namd_mdff.chirality_lines(top, xyz, 1.0, D_IMPROPERS)
                    problems += bad
                if cfg["ssdihedrals"]["enabled"]:
                    problems += _namd_mdff.ss_dihedral_lines(top, xyz, cfg["ssdihedrals"]["residues"], 1.0)[1]
        except (OSError, ValueError, _namd_mdff.PrepError) as exc:
            return [Issue("error", f"cannot check the topology: {exc}", ("input:topology",))]
        return [Issue("error", p, ("input:topology",)) for p in problems]

    # ---------------------------------------------------------------- files
    def _common(self, s, ctx, periodic: bool, for_energy: bool = False) -> List[str]:
        n = ctx.name
        # the vacuum energy comparison uses an effectively infinite cutoff: NAMD shifts electrostatics
        # without PME, sander truncates them, and only a cutoff beyond the molecule makes them comparable
        cut = 999.0 if for_energy and not periodic else s["cutoff"]
        lines = ["# Amber force field read natively from the parm7 (nothing converted)",
                 "amber            on",
                 f"parmfile         {ctx.inp('topology')}",
                 f"ambercoor        {n('start', 'rst7')}",
                 "readexclusions   yes",
                 "exclude          scaled1-4",
                 "1-4scaling       0.833333",
                 "scnb             2.0",
                 "switching        off",
                 f"cutoff           {fortran_float(cut)}",
                 f"pairlistdist     {fortran_float(cut + 2.0)}"]
        if periodic:
            lines += ["", "# periodic cell of the Amber box, PME",
                      "PME              yes", "PMEGridSpacing   1.0",
                      f"LJcorrection     {'off' if for_energy else 'on'}"]
            if ctx.data("topology").get("four_site_water"):
                lines.append("watermodel       tip4")
        elif s["gbis"] and not for_energy:
            lines += ["", "# generalised Born implicit solvent", "GBIS             on",
                      f"ionConcentration {fortran_float(s['ion_conc'])}",
                      f"alphaCutoff      {fortran_float(max(cut - 2.0, 1.0))}"]
        if for_energy:
            lines += ["", "rigidBonds       none"]
        else:
            lines += ["", "rigidBonds       all", "rigidTolerance   1.0e-8", "useSettle        on",
                      f"timestep         {fortran_float(self.timestep(s, ctx))}",
                      "nonbondedFreq    1", "fullElectFrequency 1", "stepspercycle    10"]
        return lines

    def _conf(self, s, ctx, run, prev) -> str:
        n = ctx.name
        st = run["stage"]
        periodic = s["solvent"] == "solvated"
        lines = [f"# NAMD MDFF stage '{run['tag']}' - written by {PROJECT_ABBREV} {__version__}",
                 f"# {st['kind'].upper()}, {run['steps']} steps, map coupling {run['gscale']:g}", ""]
        lines += self._common(s, ctx, periodic)
        lines += ["", "# thermostat", "langevin         on", f"langevinDamping  {fortran_float(s['damping'])}",
                  f"langevinTemp     {fortran_float(s['temperature'])}", "langevinHydrogen off", ""]
        if prev is None:
            lines.append(f"temperature      {fortran_float(s['temperature']) if st['kind'] == 'md' else '0'}")
            if periodic:
                lines.append(f"extendedSystem   {n('cell', 'xsc')}")
        else:
            lines.append(f"bincoordinates   {prev['out']}.coor")
            if prev["stage"]["kind"] == "md" and st["kind"] == "md":
                lines.append(f"binvelocities    {prev['out']}.vel")
            else:
                lines.append(f"temperature      {fortran_float(s['temperature']) if st['kind'] == 'md' else '0'}")
            if periodic:
                lines.append(f"extendedSystem   {prev['out']}.xsc")
        lines += ["", f"outputName       {run['out']}", "binaryOutput     yes",
                  f"restartfreq      {max(1, min(run['steps'], 5000))}",
                  f"outputEnergies   {st['energyfreq']}"]
        if periodic:
            lines.append(f"xstFreq          {max(1, min(run['steps'], 5000))}")
        if st["kind"] == "md" and st["dcdfreq"]:
            lines.append(f"dcdfreq          {st['dcdfreq']}")
        g = f"{run['gscale']:.6g}"
        lines += ["", "# MDFF (as VMD's mdff template): occupancy 1 couples an atom, beta = its mass; force = -scale x "
                  "mass x grad(potential)",
                  "mgridforce       on",
                  f"mgridforcefile   mdff {n('coupling', 'pdb')}",
                  "mgridforcecol    mdff O",
                  "mgridforcechargecol mdff B",
                  f"mgridforcepotfile mdff {n('potential', 'dx')}",
                  f"mgridforcescale  mdff {g} {g} {g}",
                  "mgridforcecont1  mdff no", "mgridforcecont2  mdff no", "mgridforcecont3  mdff no"]
        settings = self.prep_settings(s, ctx)
        extra = []
        if st["chirality"] and settings["chirality"]["enabled"]:
            extra.append(f"extraBondsFile   {n('chirality', 'txt')}")
        if st["ssdihedrals"] and settings["ssdihedrals"]["enabled"]:
            extra.append(f"extraBondsFile   {n('ssdihedrals', 'txt')}")
        if extra:
            lines += ["", "# L-RNA-safe restraints written by MDATK (chirality impropers, secondary-structure dihedrals)",
                      "extraBonds       on", *extra]
        if st["basepairs"] and settings["basepairs"]["enabled"]:
            lines += ["", "# base-pair hydrogen-bond restraints (Colvars)", "colvars          on",
                      f"colvarsConfig    {n('basepairs', 'colvars')}"]
        if st["positional"]:
            lines += ["", "# positional restraints to the starting coordinates", "constraints      on",
                      f"consref          {n('posres', 'pdb')}", f"conskfile        {n('posres', 'pdb')}",
                      "conskcol         B"]
        lines += ["", f"{'minimize' if st['kind'] == 'min' else 'run'} {run['steps']}", ""]
        return "\n".join(lines)

    def _energy_conf(self, s, ctx) -> str:
        n = ctx.name
        periodic = s["solvent"] == "solvated"
        lines = [f"# Single-point energy for the comparison with sander - written by {PROJECT_ABBREV} {__version__}",
                 "# no map, no restraints, no rigid bonds, no long-range LJ correction: the force field alone", ""]
        lines += self._common(s, ctx, periodic, for_energy=True)
        lines += ["", "temperature      0"]
        if periodic:
            lines.append(f"extendedSystem   {n('cell', 'xsc')}")
        lines += [f"outputName       {n('energy')}", "outputEnergies   1", "", "run 0", ""]
        return "\n".join(lines)

    def _energy_mdin(self, s) -> str:
        periodic = s["solvent"] == "solvated"
        cut = fortran_float(s["cutoff"]) if periodic else "999.0"
        return "\n".join([
            "Single-point energy for the comparison with NAMD (MDATK)",
            " &cntrl",
            "  imin=1, maxcyc=1, ncyc=1, ntpr=1,",
            f"  ntb={1 if periodic else 0}, cut={cut}, igb=0,",
            "  ntc=1, ntf=1,",
            " /",
            *([" &ewald", "  vdwmeth=0,", " /"] if periodic else []),
            ""])

    def generate(self, s, ctx):
        n = ctx.name
        files: List[GeneratedFile] = []
        settings = self.prep_settings(s, ctx)
        prep_py, energy_py = n("mdff", "py"), n("energy", "py")
        files.append(GeneratedFile(prep_py, embed(
            _namd_mdff, {"SETTINGS": settings}, "main(sys.argv[1:], SETTINGS)",
            f"MDFF preparation and final model for NAMD on an Amber topology - written by {PROJECT_ABBREV} "
            f"{__version__}.\n\nUsage: python3 THIS_SCRIPT prep | final | check-only | is-netcdf FILE"),
            "helper", "potential, coupling PDB, L-RNA-safe restraints, periodic cell; final model"))
        runs = self.runs(s, ctx)
        prev = None
        for run in runs:
            files.append(GeneratedFile(run["conf"], self._conf(s, ctx, run, prev), "input",
                                       f"NAMD stage {run['tag']}"))
            prev = run
        if s["energy_check"]:
            tolerances = dict(_namd_energy.TOLERANCES)
            if s["solvent"] != "solvated":  # NAMD shifts cutoff electrostatics, sander truncates: 1 %
                tolerances["ELECT"] = (2.0, 1e-2)
            files.append(GeneratedFile(energy_py, embed(
                _namd_energy, {"TOLERANCES": tolerances}, "main(sys.argv[1:], TOLERANCES)",
                f"sander / NAMD single-point energy comparison - written by {PROJECT_ABBREV} {__version__}.\n\n"
                "Usage: python3 THIS_SCRIPT SANDER.mdout NAMD.log"), "helper", "compares the energy terms"))
            files.append(GeneratedFile(n("energy", "namd"), self._energy_conf(s, ctx), "input",
                                       "NAMD single point"))
            files.append(GeneratedFile(n("energy", "mdin"), self._energy_mdin(s), "input", "sander single point"))
        files.append(GeneratedFile(n(self.tag, "sh"), self._script(s, ctx, runs), "run",
                                   "prepares and runs the MDFF stages"))
        return files

    def _script(self, s, ctx, runs) -> str:
        n = ctx.name
        script = n(self.tag, "sh")
        sh = script_header("NAMD MDFF on the Amber topology", ctx, s, script, notes=[
            "NAMD reads the Amber parm7 natively (amber on): CP / L-RNA types and dihedrals, NBfix19 / 0BPh /",
            "Mg2+ pair terms and HMR masses are used exactly as built. Restraints are MDATK's L-RNA-safe ones;",
            "never add VMD's chirality / ssrestraints / cispeptide restraints (they assume D-nucleotides)."])
        cmd = [s["namd"].strip() or "namd3", f"+p{s['cores']}"]
        if s["gpu"].strip():
            cmd.append(f"+devices {s['gpu'].strip()}")
        if s["namd_args"].strip():
            cmd.append(s["namd_args"].strip())
        top, crd, start = ctx.inp("topology"), ctx.inp("coordinates"), n("start", "rst7")
        sh += [f'NAMD="{" ".join(cmd)}"', "",
               "# Coordinates as ASCII Amber restart (pmemd writes NetCDF restarts: converted with cpptraj)",
               f"if python3 {n('mdff', 'py')} is-netcdf {crd}; then",
               f"    cpptraj -p {top} -y {crd} -x {start} > {n('start', 'log')}",
               "else",
               f"    python3 {n('mdff', 'py')} coordinates {crd}",
               "fi",
               require_file(start, "the coordinate conversion"), "",
               "# Checks, MDFF potential, coupling and restraint files, periodic cell",
               f"python3 {n('mdff', 'py')} prep | tee {n('prep', 'log')}",
               require_file(n("potential", "dx"), "the MDFF preparation"), ""]
        if s["energy_check"]:
            e = n("energy")
            sh += ["# Same energies in sander and NAMD? (else NAMD does not reproduce the topology's force field)",
                   f"{s['sander'].strip() or 'sander'} -O -i {e}.mdin -o {e}.mdout -p {top} -c {start} "
                   f"-r {e}.rst7 -inf {e}.mdinfo || {{ echo \"ERROR: sander failed; see {e}.mdout\" >&2; exit 1; }}",
                   f"$NAMD {e}.namd > {e}.namd.log 2>&1 || {{ echo \"ERROR: NAMD failed on the Amber topology; see "
                   f"{e}.namd.log\" >&2; exit 1; }}",
                   f"python3 {n('energy', 'py')} {e}.mdout {e}.namd.log | tee {n('energy-check', 'log')}", ""]
        for run in runs:
            done = f'[ -s "{run["out"]}.coor" ] && grep -q "WallClock:" "{run["log"]}" 2>/dev/null'
            body = [f"$NAMD {run['conf']} > {run['log']} 2>&1 || {{ echo \"ERROR: NAMD failed on {run['conf']}; see "
                     f"{run['log']}\" >&2; exit 1; }}",
                    f'grep -q "WallClock:" "{run["log"]}" || {{ echo "ERROR: NAMD did not finish {run["conf"]}; see '
                    f'{run["log"]}" >&2; exit 1; }}']
            sh.append(f"# {run['tag']}: {run['stage']['kind']} {run['steps']} steps, map coupling {run['gscale']:g}")
            if s["resume"]:
                sh += [f"if {done}; then", f'    echo "{run["tag"]} already finished - skipped"', "else"]
                sh += ["    " + b for b in body] + ["fi"]
            else:
                sh += body
            sh.append("")
        sh += ["# The fitted model (solute only) in the map's frame",
               f"python3 {n('mdff', 'py')} final", require_file(n("fitted", "pdb"), "the final conversion"),
               f'echo "Fitted model: {n("fitted", "pdb")}"', ""]
        return "\n".join(sh)
