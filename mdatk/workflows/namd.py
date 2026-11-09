"""NAMD MDFF on the Amber topology: flexible fitting into a density map without converting the force field.

NAMD reads the parm7 written by the Amber build natively (``amber on``), so the CP / L-RNA atom types,
the L-RNA dihedrals, every off-diagonal Lennard-Jones pair (NBfix19, 0BPh, microMg / nanoMg) and HMR
masses are used exactly as Amber uses them.  Before fitting, a single-point energy comparison with
sander proves it for the topology at hand.

Restraints are L-RNA-safe and written by MDATK (scripts/namd_mdff.py), all as NAMD extraBonds or
positional restraints: chirality impropers with the target of each residue's own hand, the MDFF paper's
dihedral restraints of paired residues, flat-bottom base-pair walls, the plan's own distance walls
(``extra_bonds``) and positional restraints, optionally on some residues only.  VMD's ``chirality`` /
``ssrestraints`` / ``cispeptide`` tools assume D-nucleotides and are never used.

Every restraint a fit applies is a setting of the step, so the planner's checks list it.  A step can
continue another NAMD MDFF step's fit (its ``restart`` and ``topology`` outputs, its fitted model as the
map-frame model) with restraints of its own: a two-part protocol is one plan.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import PROJECT_ABBREV, __version__
from . import rna
from .amber import WATERS, secondary_structure
from .base import PortValue, Tool, env_section, fortran_float, require_file, script_header, shell_quote
from .budget import env_gpus, gpu_ids, gpu_step
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section
from .scripts import embed
from .scripts import namd_energy as _namd_energy
from .scripts import namd_mdff as _namd_mdff

# Improper dihedral (centre, partner 1-3; scripts/namd_mdff.SUGAR_CENTRES) of each ribose centre of a
# D-nucleotide, degrees, measured on the C3'-endo Arnott A-RNA fiber monomer of mdatk.builder (the
# nucleotide of its default Watson-Crick helices); an L residue gets the mirror image (the negative).
# The same for every base.
D_IMPROPERS = {"C1'": -35.01, "C2'": 40.03, "C3'": 38.30, "C4'": 33.31}
FOUR_SITE = {"opc", "tip4pew", "tip4pd", "tip4p2005"}
# file tags of the step's own files (ctx.name): a stage or segment run of the same name would overwrite them
RESERVED_TAGS = ("start", "prep", "potential", "coupling", "posres", "chirality", "ssdihedrals", "basepairs",
                 "extrabonds", "cell", "fitted", "fitted-system", "mdff", "energy", "energy-check", "namd-mdff")
_FILE_CHECKS: Dict[tuple, List[Issue]] = {}
_TOPOLOGY_FACTS: Dict[tuple, Dict[str, Any]] = {}


def topology_facts(path: str) -> Dict[str, Any]:
    """Four-site water and HMR read from a parm7 on disk - what the Amber build reports for its own
    topologies (port data), for a topology chosen as a file.  Cached per file version."""
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return {}
    if key not in _TOPOLOGY_FACTS:
        if len(_TOPOLOGY_FACTS) > 32:
            _TOPOLOGY_FACTS.clear()
        try:
            top = _namd_mdff.Topology(path)
            h_mass = [m for i, (m, z) in enumerate(zip(top.mass, top.number)) if z == 1 and not top.solvent(i)]
            heavy_h = sum(1 for m in h_mass if m > 2.0)
            beta = top.box_dimensions[0] if top.box_dimensions else 90.0
            _TOPOLOGY_FACTS[key] = {"four_site_water": top.numextra > 0,
                                    "hmr": bool(h_mass) and heavy_h >= 0.5 * len(h_mass),
                                    "octahedral": top.ifbox == 2 or abs(beta - 90.0) > 0.1}
        except (OSError, ValueError, _namd_mdff.PrepError):
            _TOPOLOGY_FACTS[key] = {}
    return dict(_TOPOLOGY_FACTS[key])


def namd_file(path: str) -> str:
    """A file name as a NAMD config value: in braces, which NAMD strips as Tcl does, when it holds spaces
    or characters Tcl would substitute."""
    return "{" + path + "}" if any(c.isspace() or c in '$[]";\\#' for c in path) else path

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
    Option("extrabonds", "Extra distance restraints", "bool", True,
           "The step's own flat-bottom distance walls between named atoms (Extra distance restraints page)."),
    Option("positional", "Positional restraints", "bool", False,
           "Harmonic restraints to the starting coordinates (atoms, residues and force constant on the Restraints "
           "page)."),
    Option("dcdfreq", "Trajectory frame every", "int", 5000, "dcdfreq (MD steps); 0: no trajectory.",
           minimum=0, maximum=100_000_000, requires={"kind": "md"}),
    Option("energyfreq", "Energies every", "int", 1000, "outputEnergies (steps).", minimum=1, maximum=100_000_000),
]
EXTRA_BOND_OPTIONS = [
    Option("name", "Name", "text", "wall", "Names the restraint in the restraint file and the planner's checks."),
    Option("atom1", "First atom", "text", "",
           "Residue and atom: a topology residue number (16:O3') or a chain and residue number as in the fitted "
           "model, i.e. those of the map-frame model (A:16:O3').", placeholder="16:O3'"),
    Option("atom2", "Second atom", "text", "", "Residue and atom, as the first.", placeholder="17:P"),
    Option("lower", "Flat bottom from", "float", 1.4,
           "Lower wall (Å). At least 0.1 Å: NAMD reads a lower wall of 0 as a bond of zero length.", minimum=0.1,
           maximum=100.0, digits=2, unit="Å"),
    Option("upper", "Flat bottom to", "float", 1.8, "Upper wall (Å).", minimum=0.1, maximum=100.0, digits=2,
           unit="Å"),
    Option("k", "Wall force constant", "float", 20.0,
           "k in E = k (d − wall)² beyond the walls (kcal/mol/Å², NAMD extraBonds 'wall').", minimum=0.0,
           maximum=10000.0, digits=2),
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
    outputs = [Port("fitted", "Fitted model (PDB, map frame)", ("pdb",)),
               Port("restart", "Final coordinates, whole system (rst7)", ("rst7",), turns_on="continuation",
                    tooltip="NAMD's last coordinates of every atom in the simulation frame, with the box: the "
                            "coordinates of a continuation step (with the topology output)."),
               Port("topology", "Topology (passed on)", ("parm7",), turns_on="continuation",
                    tooltip="The topology this step ran on, for a continuation step: the final coordinates belong "
                            "to it (a rebuilt topology numbers the solvent anew).")]
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
            Option("couple_ions", "Couple ions too", "bool", False,
                   "Also couple the Mg²⁺ and other ions that start within the margin around the RNA (bulk "
                   "counterions are left alone)."),
            Option("pad", "Margin around the RNA", "float", 8.0,
                   "The potential is cropped to the coupled atoms plus this margin (Å); it must fit into the "
                   "periodic cell as NAMD checks it, which a truncated octahedron allows only for small margins.",
                   minimum=0.0, maximum=100.0, digits=1, unit="Å"),
            Option("min_overlap", "Required overlap", "float", 0.3,
                   "Share of the coupled RNA atoms that must lie on density above the map's mean + 1 σ, a check "
                   "that the model is in the map's frame.", minimum=0.0, maximum=1.0, step=0.05, digits=2),
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
            Option("l_residues", "L residues", "text", "",
                   "Topology residue numbers of the L residues, e.g. 45-73 - for topologies whose residue names "
                   "do not tell the hand (an L-RNA library that writes D names). Every other nucleotide is then D, "
                   "and each residue's sugar geometry must agree, or the run stops. Empty: the hand comes from the "
                   "residue names (LA, LG5 ...).", placeholder="45-73"),
            Option("sequence", "Sequence", "multiline", "",
                   "For a typed secondary structure: RNA sequence in topology order, chains separated by spaces.",
                   placeholder="GGGCAC GUGCCC"),
            Option("dotbracket", "Secondary structure", "multiline", "",
                   "Dot-bracket; ignored when a secondary structure is connected.", placeholder="(((((( ))))))"),
            Option("res_offset", "Residue offset", "int", 0,
                   "Topology residue number = position in the sequence + offset.", minimum=-100000, maximum=100000),
            Option("wobble", "Restrain G·U wobble pairs", "bool", False, "Also restrain G·U pairs."),
            Option("bp_lower", "Flat bottom from", "float", 2.7,
                   "Lower wall of each H-bond distance (Å). At least 0.1 Å: NAMD reads a lower wall of 0 as a bond "
                   "of zero length (or, on GPUs, drops it).", minimum=0.1, maximum=20.0, digits=2, unit="Å"),
            Option("bp_upper", "Flat bottom to", "float", 3.2, "Upper wall (Å).", minimum=0.0, maximum=20.0,
                   digits=2, unit="Å"),
            Option("bp_k", "Wall force constant", "float", 10.0,
                   "k in E = k (d − wall)² beyond the walls (kcal/mol/Å², as rk2 / rk3 of Amber restraints; NAMD "
                   "extraBonds 'wall').", minimum=0.0, maximum=1000.0, digits=2),
            Option("positional_atoms", "Positional restraints on", "radio", "backbone",
                   "Atoms restrained to the starting coordinates in stages with positional restraints.",
                   choices=[("backbone", "RNA backbone (P O5' C5' C4' C3' O3')"), ("heavy", "RNA heavy atoms")]),
            Option("positional_residues", "Positional restraints on residues", "text", "",
                   "Only these residues are held (the B column of the restraint PDB is zero for every other atom): "
                   "topology residue numbers such as 16-17, or chains and numbers as in the fitted model (those of "
                   "the map-frame model) such as A:16-17, B:3. Empty: every RNA residue. A range that names no "
                   "residue is an error, never an empty selection.", placeholder="16-17"),
            Option("positional_k", "Positional force constant", "float", 1.0, "kcal/mol/Å².", minimum=0.0,
                   maximum=1000.0, digits=2),
        ], tooltip="L-RNA-safe restraints written by MDATK (never VMD's chirality / ssrestraints)."),
        Section("extra_bonds", "Extra distance restraints", EXTRA_BOND_OPTIONS, repeat=True, item_label="name",
                templates=[("wall", "Distance wall between two atoms", {})],
                tooltip="Flat-bottom distance restraints of your own between two atoms, e.g. across a chain break "
                        "the map does not cover: NAMD extraBonds walls in the step's own file, applied in the "
                        "stages that ask for them."),
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
                   "When the script is run again, finished runs are skipped while their inputs and stage files are "
                   "unchanged; a change reruns that stage and every one after it."),
            Option("continuation", "Outputs for a continuation step", "bool", False,
                   "Also write NAMD's last coordinates of every atom (solvent included) as an Amber rst7 in the "
                   "simulation frame, with the box, and pass the topology on: a following NAMD MDFF step takes them "
                   "as its coordinates and topology (and this step's fitted model as its map-frame model) and "
                   "continues the fit with restraints of its own. Connecting either output turns this on."),
        ], tooltip="How the NAMD runs are started."),
        env_section("module load NAMD/3.0   and   module load Amber/24   (sander, cpptraj)"),
    ]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def stage_tag(stage: Dict[str, Any], k: int) -> str:
        from .naming import sanitize_tag
        return sanitize_tag(stage.get("name") or f"stage{k + 1}")

    @staticmethod
    def stage_scales(st: Dict[str, Any]) -> List[float]:
        """Map coupling ξ of each NAMD run (segment) of a stage: constant, or a linear ramp over the
        segments of an MD stage."""
        nseg = st["segments"] if st["kind"] == "md" else 1
        end = st["gscale_end"] if st["kind"] == "md" and st["gscale_end"] is not None else st["gscale"]
        return [st["gscale"] + (end - st["gscale"]) * (seg / (nseg - 1) if nseg > 1 else 0.0)
                for seg in range(nseg)]

    def runs(self, s, ctx) -> List[Dict[str, Any]]:
        """Stages expanded into NAMD runs (segments), with the coupling of each run."""
        out = []
        for k, st in enumerate(s["stages"]):
            tag = self.stage_tag(st, k)
            scales = self.stage_scales(st)
            steps = max(1, st["steps"] // len(scales))
            for seg, gscale in enumerate(scales):
                rtag = f"{tag}-{seg + 1:03d}" if len(scales) > 1 else tag
                out.append({"stage": st, "index": k, "seg": seg, "tag": rtag, "steps": steps, "gscale": gscale,
                            "conf": ctx.name(rtag, "namd"), "out": ctx.name(rtag), "log": ctx.name(rtag, "log")})
        return out

    def topology_data(self, ctx) -> Dict[str, Any]:
        """The topology's port data; for a topology chosen as a file, four-site water and HMR are read from
        the file (the Amber build reports them for the topologies it writes)."""
        data = dict(ctx.data("topology"))
        if "four_site_water" not in data or "hmr" not in data:
            path = ctx.existing("topology")
            for key, value in (topology_facts(path) if path else {}).items():
                data.setdefault(key, value)
        return data

    def four_site(self, s, ctx) -> bool:
        return s["solvent"] == "solvated" and bool(self.topology_data(ctx).get("four_site_water"))

    def timestep(self, s, ctx) -> float:
        if s["timestep"] != "auto":
            return float(s["timestep"])
        return 4.0 if self.topology_data(ctx).get("hmr") else 2.0

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

    @staticmethod
    def l_residues(s) -> Optional[List[int]]:
        try:
            found = _namd_mdff.parse_residues(s["l_residues"])
        except _namd_mdff.PrepError:
            return None  # reported by check()
        return sorted(found) if found is not None else None

    def uses(self, s, key: str) -> bool:
        return any(st[key] for st in s["stages"])

    @staticmethod
    def extra_bonds(s) -> List[list]:
        """The step's own distance restraints as the helper reads them: [name, atom 1, atom 2, lower, upper, k]
        (white space collapsed: the name goes into a comment line of the restraint file)."""
        return [[" ".join(b["name"].split()) or f"wall{k + 1}", b["atom1"].strip(), b["atom2"].strip(), b["lower"],
                 b["upper"], b["k"]] for k, b in enumerate(s["extra_bonds"])]

    @staticmethod
    def positional_residues(s) -> str:
        """The positional residue selection on one line ("" for every RNA residue)."""
        return " ".join(s["positional_residues"].split())

    def stages_using(self, s, key: str) -> str:
        return ", ".join(self.stage_tag(st, k) for k, st in enumerate(s["stages"]) if st[key])

    def restraint_lines(self, s) -> List[str]:
        """What the plan restrains beyond the secondary structure and chirality, in words: the extra distance
        walls and the positional restraints (planner checks, plan summary)."""
        out = []
        where = self.stages_using(s, "extrabonds")
        for name, a1, a2, lower, upper, k in self.extra_bonds(s):
            out.append(f"extra restraint {name}: {a1} - {a2}, flat {lower:g}-{upper:g} Å, k {k:g} kcal/mol/Å²"
                       + (f" (stages {where})" if where else " (no stage applies it)"))
        if self.uses(s, "positional"):
            which = self.positional_residues(s)
            out.append(f"positional restraints, k {s['positional_k']:g} kcal/mol/Å², on the "
                       f"{'backbone' if s['positional_atoms'] == 'backbone' else 'heavy'} atoms of "
                       + (f"residues {which} only" if which else "every RNA residue")
                       + f" (stages {self.stages_using(s, 'positional')})")
        return out

    @staticmethod
    def continuation_source(ctx) -> Optional[PortValue]:
        """The coordinates when they are another NAMD MDFF step's final coordinates (a continuation)."""
        value = ctx.value("coordinates")
        return value if value is not None and value.paths and "mdff_topology" in value.data else None

    def prep_settings(self, s, ctx) -> Dict[str, Any]:
        n = ctx.name
        pairs, _issues, _src = self.pairs(s, ctx)
        paired = self.paired_residues(s, ctx)
        bonds = {k: v for k, v in rna.HBOND_ATOMS.items() if any(p[2] == k for p in pairs)}
        out = {
            "topology": ctx.inp("topology"), "coordinates": n("start", "rst7"), "model": ctx.inp("model"),
            "map": ctx.inp("map"), "periodic": s["solvent"] == "solvated",
            "four_site": self.four_site(s, ctx),
            "timestep": self.timestep(s, ctx), "threshold": s["threshold"], "pad": s["pad"],
            "couple": s["couple"], "couple_ions": s["couple_ions"], "min_overlap": s["min_overlap"],
            "max_frame_rmsd": s["max_frame_rmsd"], "resample_divisor": s["resample_divisor"],
            "potential": n("potential", "dx"), "grid_pdb": n("coupling", "pdb"), "restraint_pdb": n("posres", "pdb"),
            "extrabonds": n("chirality", "txt"), "xsc": n("cell", "xsc"),
            "prep_json": n("prep", "json"), "final_pdb": n("fitted", "pdb"),
            "final_coor": self.runs(s, ctx)[-1]["out"] + ".coor" if s["stages"] else "",
            "chirality": {"enabled": self.uses(s, "chirality"), "k": s["chirality_k"], "targets": dict(D_IMPROPERS),
                          "l_residues": self.l_residues(s)},
            "basepairs": {"enabled": self.uses(s, "basepairs") and bool(pairs), "pairs": [list(p) for p in pairs],
                          "bonds": {k: [list(b) for b in v] for k, v in bonds.items()}, "offset": s["res_offset"],
                          "walls": [s["bp_lower"], s["bp_upper"], s["bp_k"]], "file": n("basepairs", "txt")},
            "positional": {"selection": s["positional_atoms"] if self.uses(s, "positional") else "none",
                           "k": s["positional_k"]},
            "ssdihedrals": {"enabled": self.uses(s, "ssdihedrals") and bool(paired), "k": s["ss_k"],
                            "residues": paired, "file": n("ssdihedrals", "txt")},
        }
        # the settings below appear only when used, so the helper of a step without them stays as it was
        if self.positional_residues(s):
            out["positional"]["residues"] = self.positional_residues(s)
        if s["extra_bonds"]:
            out["extra_bonds"] = {"enabled": self.uses(s, "extrabonds"), "bonds": self.extra_bonds(s),
                                  "file": n("extrabonds", "txt")}
        if s["continuation"]:
            out["final_rst7"] = n("fitted-system", "rst7")
        return out

    # ----------------------------------------------------------------- planner
    def output_values(self, s, ctx):
        out = {"fitted": PortValue([ctx.own("fitted", "pdb")], "pdb", {"map_frame": True})}
        if s["continuation"]:
            top = ctx.value("topology")
            out["restart"] = PortValue([ctx.own("fitted-system", "rst7")], "rst7",
                                       {"box": s["solvent"] == "solvated", "map_frame": False,
                                        "mdff_topology": top.path if top is not None else ""})
            if top is not None and top.paths:  # passed on: the final coordinates belong to this very file
                out["topology"] = PortValue([top.path], top.kind, dict(top.data))
        return out

    def programs(self, s):
        # cpptraj runs only for NetCDF coordinates: not checked beforehand (bash names it if it is missing)
        namd = (s["namd"].strip() or "namd3").split()[0]
        sander = (s["sander"].strip() or "sander").split()[0]
        out = [(namd, r"(^|[\s;(&|])\$NAMD\s"), ("python3", r"(^|[\s;(&|])python3\s")]
        if s["energy_check"]:
            out.append((sander, r"(^|[\s;(&|])" + re.escape(sander) + r"\s+-O\b"))
        return out

    def resources(self, s):
        # cpptraj: the script converts NetCDF restarts (pmemd's default) with it
        return ["namd", "mdff", "cpptraj"] + (["ambertools"] if s["energy_check"] else [])

    def run_resources(self, s):
        # +p cores and one GPU (+devices names it; it counts the GPUs the Environment leaves visible)
        devices = gpu_ids(s["gpu"])
        visible = env_gpus(s)
        if devices and visible and all(d.isdigit() and int(d) < len(visible) for d in devices):
            devices = tuple(visible[int(d)] for d in devices)
        return gpu_step(int(s["cores"]), 1, s, devices)

    def describe(self, s):
        gs = [g for st in s["stages"] for g in self.stage_scales(st)]
        env = "solvated" if s["solvent"] == "solvated" else ("GBIS" if s["gbis"] else "vacuum")
        span = f", ξ {min(gs):g}–{max(gs):g}" if gs else ""
        extra = ""
        if s["extra_bonds"]:
            extra += f", {len(s['extra_bonds'])} extra restraint{'s' * (len(s['extra_bonds']) > 1)}"
        if self.positional_residues(s) and self.uses(s, "positional"):
            extra += f", positional on {self.positional_residues(s)}"
        return f"NAMD MDFF on the Amber topology, {env}, {len(s['stages'])} stage(s){span}{extra}"

    def summary_lines(self, s, ctx):
        # positional restraints of every RNA residue are what the stages' check boxes always meant: the summary
        # of such a plan stays as it was
        lines = [ln for ln in self.restraint_lines(s)
                 if not ln.startswith("positional") or self.positional_residues(s)]
        return ["Restraints of the plan:"] + [f"- {ln}" for ln in lines] if lines else []

    def check(self, s, ctx):
        out: List[Issue] = []
        data = self.topology_data(ctx)
        solvated = s["solvent"] == "solvated"
        if not s["stages"]:
            out.append(Issue("error", "add at least one stage", ("stages",)))
        # every run writes <tag>.namd / .log / .coor / .xsc ...: names must differ without regard to letter case
        # (macOS file systems ignore it), and segment names (<stage>-001 ...) must not meet another run's or the
        # step's own files
        tags = [self.stage_tag(st, k) for k, st in enumerate(s["stages"])]
        low_tags = [t.lower() for t in tags]
        dup = sorted({t for t in tags if low_tags.count(t.lower()) > 1})
        if dup:
            out.append(Issue("error", f"stage names must differ, letter case aside ({', '.join(dup)})", ("stages",)))
        twice = {t.lower() for t in dup}
        runs = self.runs(s, ctx)
        low = [r["tag"].lower() for r in runs]
        clash = sorted({r["tag"] for r in runs
                        if low.count(r["tag"].lower()) > 1 and tags[r["index"]].lower() not in twice})
        run_tags = [r["tag"] for r in runs]
        if clash:
            out.append(Issue("error", f"run names repeat ({', '.join(clash)}): a stage name matches another stage's "
                                      "segment runs (<stage>-001 ...)", ("stages",)))
        reserved = sorted({t for t in run_tags if t.lower() in RESERVED_TAGS})
        if reserved:
            several = len(reserved) > 1
            out.append(Issue("error", f"stage name{'s' * several} {', '.join(reserved)} {'are' if several else 'is'} "
                                      f"taken by the step's own files ({', '.join(RESERVED_TAGS)})", ("stages",)))
        if any(c in ctx.inp("topology") for c in "{}\n"):
            out.append(Issue("error", "the topology's path contains a brace, which a NAMD configuration cannot "
                                      "hold; move or rename the file", ("input:topology",)))
        try:
            _namd_mdff.parse_residues(s["l_residues"])
        except _namd_mdff.PrepError as exc:
            out.append(Issue("error", f"L residues: {exc}", ("l_residues",)))
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
        path = ctx.existing("topology")
        if solvated and (data.get("box_shape") == "oct" or (path and topology_facts(path).get("octahedral"))):
            out.append(Issue("warning", "the periodic box is a truncated octahedron: NAMD fits the MDFF potential only "
                                        "into a region much smaller than the box's width, so the usual margin rarely "
                                        "fits (the script checks it and stops before NAMD runs). A rectangular box "
                                        "(Amber build: Box shape 'Rectangular') avoids this", ("input:topology",)))
        if solvated and data.get("four_site_water"):
            water = (f"{WATERS[data['water']][0]} is a four-site water" if data.get("water") in WATERS
                     else "the topology has four-site water (extra points)")
            out.append(Issue("warning", f"{water}: NAMD runs it with 'watermodel tip4' from the Amber extra points, "
                                        "which MDATK has not verified - only trust the run if the energy comparison "
                                        "with sander passes", ("energy_check",)))
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
        if not ctx.has("model") and self.continuation_source(ctx) is None:  # a continuation: _restraint_checks
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
        strong = [k for k, st in enumerate(s["stages"])
                  if (not st["chirality"] or s["chirality_k"] <= 0) and max(self.stage_scales(st)) > 1.0]
        if strong:
            plural = "s" * (len(strong) > 1)
            out.append(Issue("warning", f"strong map coupling (ξ > 1) without chirality restraints in stage{plural} "
                                        f"{', '.join(tags[k] for k in strong)} can invert sugar centres (of D and L "
                                        "residues alike)", tuple(f"stages.{k}.chirality" for k in strong)))
        for k, st in enumerate(s["stages"]):
            if st["kind"] == "md" and st["segments"] > st["steps"]:
                out.append(Issue("error", f"stage {k + 1}: more segments than steps", (f"stages.{k}.segments",)))
            if (st["kind"] == "md" and st["gscale_end"] is not None and st["gscale_end"] != st["gscale"]
                    and st["segments"] < 2):
                out.append(Issue("warning", f"stage {k + 1}: ξ stays at {st['gscale']:g} - a ramp to "
                                            f"{st['gscale_end']:g} needs at least 2 segments",
                                 (f"stages.{k}.segments",)))
        out += self._restraint_checks(s, ctx)
        out.append(Issue("info", "restraints are written by MDATK and keep every residue's own hand; do not add "
                                 "VMD's chirality / ssrestraints / cispeptide restraints, which assume D-nucleotides "
                                 "and would push L sugars towards the inverted centre", ("chirality_k",)))
        out += self._file_checks(s, ctx)
        return out

    def _restraint_checks(self, s, ctx) -> List[Issue]:
        """The plan's own restraints (extra distance walls, positional residues) and a continuation's wiring,
        as far as the settings tell; _file_checks resolves the atoms and residues once the files exist."""
        out: List[Issue] = []
        for k, b in enumerate(s["extra_bonds"]):
            name = " ".join(b["name"].split()) or f"wall{k + 1}"
            for key, which in (("atom1", "first"), ("atom2", "second")):
                if not b[key].strip():
                    out.append(Issue("error", f"extra restraint {name}: name its {which} atom (residue:atom such as "
                                              "16:O3', or chain:residue:atom such as A:16:O3')",
                                     (f"extra_bonds.{k}.{key}",)))
                    continue
                try:
                    _namd_mdff.parse_atom(b[key])
                except _namd_mdff.PrepError as exc:
                    out.append(Issue("error", f"extra restraint {name}: {exc}", (f"extra_bonds.{k}.{key}",)))
            if b["atom1"].strip() and b["atom1"].strip() == b["atom2"].strip():
                out.append(Issue("error", f"extra restraint {name}: both ends are {b['atom1'].strip()}",
                                 (f"extra_bonds.{k}.atom2",)))
            if b["lower"] > b["upper"]:
                out.append(Issue("error", f"extra restraint {name}: the lower wall must not be above the upper wall",
                                 (f"extra_bonds.{k}.lower", f"extra_bonds.{k}.upper")))
        if s["extra_bonds"] and not self.uses(s, "extrabonds"):
            out.append(Issue("warning", f"{len(s['extra_bonds'])} extra distance restraint(s) are defined but no "
                                        "stage applies them (tick 'Extra distance restraints' in a stage)",
                             ("extra_bonds",)))
        residues = self.positional_residues(s)
        try:
            _namd_mdff.parse_selection(residues)
        except _namd_mdff.PrepError as exc:
            out.append(Issue("error", f"positional restraints on residues: {exc}", ("positional_residues",)))
        if residues and not self.uses(s, "positional"):
            out.append(Issue("warning", f"positional restraints are limited to residues {residues}, but no stage "
                                        "uses positional restraints", ("positional_residues",)))
        for line in self.restraint_lines(s):
            out.append(Issue("info", line, ("extra_bonds",) if line.startswith("extra") else ("positional_residues",)))
        cont = self.continuation_source(ctx)
        if cont is not None:
            src = cont.source_label or "the step before"
            top, model = ctx.value("topology"), ctx.value("model")
            if top is None or not top.paths or top.path != cont.data["mdff_topology"]:
                out.append(Issue("error", f"the coordinates are the final coordinates of {src} and belong to its "
                                          f"topology: connect {src}'s topology output (a rebuilt topology numbers the "
                                          "solvent anew, so the coordinates would not fit it)", ("input:topology",)))
            if model is None or not model.paths:
                out.append(Issue("warning", f"connect {src}'s fitted model as the map-frame model: the continued "
                                            "coordinates are in the simulation frame, and the fitted model gives the "
                                            "map's frame back", ("input:model",)))
            elif model.source != cont.source:
                out.append(Issue("warning", f"the map-frame model is not {src}'s fitted model: the continued "
                                            "coordinates are superposed on it, and a model from before the fit differs "
                                            "from them by what the fit moved", ("input:model",)))
            else:
                out.append(Issue("info", f"continues the fit of {src} from its final coordinates and topology; the "
                                         "restraints of this step (positional ones included) start from where that "
                                         "fit ended", ("input:coordinates",)))
        return out

    def _file_checks(self, s, ctx) -> List[Issue]:
        """When topology and coordinates already exist: the script's own topology checks, now (cached per
        file version: the planner re-checks after every edit and a solvated parm7 takes a moment to read).
        The plan's own restraints are resolved as soon as the topology exists (residues given by chain once the
        coordinates do too)."""
        top_path, crd_path = ctx.existing("topology"), ctx.existing("coordinates")
        if not top_path or getattr(ctx, "preview", False):
            return []
        own = bool(s["extra_bonds"]) or bool(self.positional_residues(s) and self.uses(s, "positional"))
        if not crd_path and not own:
            return []
        cfg = self.prep_settings(s, ctx)
        cfg.update(topology=top_path, coordinates=crd_path)
        key, files = (), [top_path, crd_path]
        if own:  # residues given by chain are numbered as in the map-frame model
            model = ctx.existing("model")
            cfg.update(model=model, model_missing=ctx.has("model") and not model)
            key = (repr(cfg.get("extra_bonds")), repr(cfg["positional"]), model, cfg["model_missing"])
            files.append(model)
        try:
            stamp = tuple((os.path.getmtime(p), os.path.getsize(p)) for p in files if p)
        except OSError:
            return []
        key = (top_path, crd_path, stamp, cfg["periodic"], cfg["four_site"], cfg["timestep"],
               cfg["chirality"]["enabled"], cfg["ssdihedrals"]["enabled"], tuple(cfg["ssdihedrals"]["residues"]),
               tuple(cfg["chirality"]["l_residues"] or ()), cfg["chirality"]["l_residues"] is None) + key
        if key not in _FILE_CHECKS:
            if len(_FILE_CHECKS) > 32:
                _FILE_CHECKS.clear()
            _FILE_CHECKS[key] = self._read_and_check(cfg, top_path, crd_path)
        return list(_FILE_CHECKS[key])

    @staticmethod
    def _own_restraint_issues(cfg, top, xyz) -> List[Issue]:
        """The extra distance walls and the positional residue selection resolved on the topology (and the
        coordinates, for residues given by chain), as the script will: what does not exist is an error."""
        out: List[Issue] = []
        chains = xyz is not None and not cfg.get("model_missing")
        numbers = _namd_mdff.ResidueNumbers(top, xyz if chains else None, cfg["model"] or None)
        later = []
        extra = cfg.get("extra_bonds")
        if extra:
            bonds = []
            for bond in extra["bonds"]:
                try:
                    by_chain = any(_namd_mdff.parse_atom(a)[0] for a in bond[1:3])
                except _namd_mdff.PrepError:
                    continue  # reported by _restraint_checks, as are equal ends and crossed walls
                if bond[1] == bond[2] or bond[3] > bond[4]:
                    continue
                if by_chain and not chains:  # the end given by topology number can be checked already
                    for spec in bond[1:3]:
                        try:
                            if _namd_mdff.parse_atom(spec)[0] is None:
                                numbers.atom(spec)
                        except _namd_mdff.PrepError as exc:
                            out.append(Issue("error", f"extra restraint {bond[0]}: {exc}", ("extra_bonds",)))
                    later.append(f"extra restraint {bond[0]}")
                    continue
                bonds.append(bond)
            _lines, problems, notes = _namd_mdff.extra_bond_lines(top, bonds, numbers, xyz)
            out += [Issue("error", p, ("extra_bonds",)) for p in problems]
            out += [Issue("info", n, ("extra_bonds",)) for n in notes]
        pos = cfg["positional"]
        if pos.get("residues") and pos["selection"] != "none":
            try:
                ranges = _namd_mdff.parse_selection(pos["residues"])
            except _namd_mdff.PrepError:
                ranges = None  # reported by _restraint_checks
            if ranges and any(chain for chain, _a, _b in ranges) and not chains:
                for chain, first, last in ranges:  # the ranges given by topology number can be checked already
                    if chain is None and not numbers.find(None, first, last):
                        out.append(Issue("error", f"positional restraints: {_namd_mdff.selection_text(None, first, last)}"
                                                  f" names no residue (the topology has {top.nres})",
                                         ("positional_residues",)))
                later.append("the positional residues")
            elif ranges:
                chosen, problems = _namd_mdff.selected_residues(top, numbers, pos["residues"], pos["selection"])
                out += [Issue("error", p, ("positional_residues",)) for p in problems]
                if not problems:
                    held = sum(1 for r in chosen for i in top.residue_atoms(r)
                               if _namd_mdff.positional(top, i, pos["selection"]))
                    out.append(Issue("info", f"positional restraints hold {held} atoms of {len(chosen)} residues "
                                             f"({pos['residues']}); every other atom is free",
                                     ("positional_residues",)))
        if later:
            out.append(Issue("info", f"{', '.join(later)}: residues given by chain are found from the coordinates "
                                     "and the map-frame model, which the planner cannot read yet; the script checks "
                                     "them before NAMD runs", ("extra_bonds",)))
        return out

    @staticmethod
    def _read_and_check(cfg, top_path, crd_path) -> List[Issue]:
        notes: List[Issue] = []
        try:
            top = _namd_mdff.Topology(top_path)
            if not crd_path:  # the plan's own restraints only, on the topology
                return NamdMDFFTool._own_restraint_issues(cfg, top, None)
            problems, _notes = _namd_mdff.topology_problems(top, cfg)
            xyz = None
            if _namd_mdff.is_netcdf(crd_path):  # converted by cpptraj when the script runs
                notes.append(Issue("info", "the coordinates are a NetCDF restart: the D/L and dihedral checks of the "
                                           "coordinates run when the script has converted them",
                                   ("input:coordinates",)))
            elif _namd_mdff.is_pdb(crd_path):
                atoms = _namd_mdff.read_pdb_atoms(crd_path)
                order = _namd_mdff.pdb_order_problem(top, atoms, os.path.basename(crd_path))
                if order:  # coordinates by position would be assigned to the wrong atoms: check nothing else
                    problems.append(order)
                else:
                    xyz = [a[3] for a in atoms]
                    if top.ifbox and _namd_mdff.pdb_box(crd_path) is None:
                        problems.append(f"{os.path.basename(crd_path)} has no CRYST1 record, so the periodic box is "
                                        "unknown: connect the rst7 of the Amber build or an MD restart")
            else:
                xyz, _box, _vel = _namd_mdff.read_rst7(crd_path, top.natom)
            if xyz is not None:
                problems += _namd_mdff.bond_problems(top, xyz)
                if cfg["chirality"]["enabled"]:
                    declared = cfg["chirality"].get("l_residues")
                    _lines, bad, _more = _namd_mdff.chirality_lines(top, xyz, 1.0, D_IMPROPERS,
                                                                    set(declared) if declared is not None else None)
                    problems += bad
                if cfg["ssdihedrals"]["enabled"]:
                    problems += _namd_mdff.ss_dihedral_lines(top, xyz, cfg["ssdihedrals"]["residues"], 1.0)[1]
            if cfg.get("extra_bonds") or cfg["positional"].get("residues"):
                notes += NamdMDFFTool._own_restraint_issues(cfg, top, xyz)
        except (OSError, ValueError, EOFError, _namd_mdff.PrepError) as exc:
            return [Issue("error", f"cannot check the topology: {exc}", ("input:topology",))]
        return [Issue("error", p, ("input:topology",)) for p in problems] + notes

    # ---------------------------------------------------------------- files
    def _common(self, s, ctx, periodic: bool, for_energy: bool = False) -> List[str]:
        n = ctx.name
        # the vacuum energy comparison runs sander without a cutoff (cut=999); with 'switching off' NAMD
        # truncates plain Coulomb and LJ sums at its cutoff, so the same 999 A makes them the same sums
        cut = 999.0 if for_energy and not periodic else s["cutoff"]
        lines = ["# Amber force field read natively from the parm7 (nothing converted)",
                 "amber            on",
                 f"parmfile         {namd_file(ctx.inp('topology'))}",
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
            if self.four_site(s, ctx):
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
        if st["basepairs"] and settings["basepairs"]["enabled"]:
            extra.append(f"extraBondsFile   {n('basepairs', 'txt')}")
        own = st["extrabonds"] and settings.get("extra_bonds", {}).get("enabled")
        if own:
            extra.append(f"extraBondsFile   {n('extrabonds', 'txt')}")
        if extra:
            lines += ["", "# L-RNA-safe restraints written by MDATK: chirality impropers, secondary-structure dihedrals,",
                      "# base-pair walls (extraBonds; never VMD's chirality / ssrestraints)"
                      + (", and the plan's extra distance walls" if own else ""),
                      "extraBonds       on", *extra]
        if st["positional"]:
            residues = settings["positional"].get("residues")
            lines += ["", "# positional restraints to the starting coordinates"
                      + (f" (residues {residues} only)" if residues else ""), "constraints      on",
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
            files.append(GeneratedFile(energy_py, embed(
                _namd_energy, {"TOLERANCES": dict(_namd_energy.TOLERANCES)}, "main(sys.argv[1:], TOLERANCES)",
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
        qtop, qcrd = shell_quote(top), shell_quote(crd)
        sh += [f'NAMD="{" ".join(cmd)}"', "",
               "# Coordinates as ASCII Amber restart (pmemd writes NetCDF restarts: converted with cpptraj)",
               f"if python3 {n('mdff', 'py')} is-netcdf {qcrd}; then",
               f"    cpptraj -p {qtop} -y {qcrd} -x {start} > {n('start', 'log')}",
               "else",
               f"    python3 {n('mdff', 'py')} coordinates {qcrd}",
               "fi",
               require_file(start, "the coordinate conversion"), "",
               "# Checks, MDFF potential, coupling and restraint files, periodic cell",
               f"python3 {n('mdff', 'py')} prep | tee {n('prep', 'log')}",
               require_file(n("potential", "dx"), "the MDFF preparation"), ""]
        if s["energy_check"]:
            e = n("energy")
            sh += ["# Same energies in sander and NAMD? (else NAMD does not reproduce the topology's force field)",
                   f"{s['sander'].strip() or 'sander'} -O -i {e}.mdin -o {e}.mdout -p {qtop} -c {start} "
                   f"-r {e}.rst7 -inf {e}.mdinfo || {{ echo \"ERROR: sander failed; see {e}.mdout\" >&2; exit 1; }}",
                   f"$NAMD {e}.namd > {e}.namd.log 2>&1 || {{ echo \"ERROR: NAMD failed on the Amber topology; see "
                   f"{e}.namd.log\" >&2; exit 1; }}",
                   f"python3 {n('energy', 'py')} {e}.mdout {e}.namd.log | tee {n('energy-check', 'log')}", ""]
        if runs:
            # a finished run belongs to the inputs it ran from: its key chains the start files with every stage
            # file up to it, so a changed input or stage setting reruns that stage and all after it.  Keys are
            # written whether or not runs are skipped, so they always describe the outputs on disk.  The
            # topology's %VERSION line holds the time it was written and is left out (a solvated rebuild still
            # changes the start coordinates: tleap places ions at random)
            settings = self.prep_settings(s, ctx)
            inputs = [start, n("prep", "json"), n("potential", "dx"), n("coupling", "pdb")]
            for path, used in ((n("posres", "pdb"), settings["positional"]["selection"] != "none"),
                               (n("chirality", "txt"), settings["chirality"]["enabled"]),
                               (n("ssdihedrals", "txt"), settings["ssdihedrals"]["enabled"]),
                               (n("basepairs", "txt"), settings["basepairs"]["enabled"]),
                               (n("extrabonds", "txt"), settings.get("extra_bonds", {}).get("enabled")),
                               (n("cell", "xsc"), s["solvent"] == "solvated")):
                if used:
                    inputs.append(path)
            sh += ["# Finished runs are " + ("skipped only if they ran from these inputs and the same stage files"
                                             if s["resume"] else "not skipped (keys record what each run used)"),
                   f"KEY=$( {{ sed '/^%VERSION/d' {qtop}; cat {' '.join(shell_quote(p) for p in inputs)}; }} | cksum )",
                   "FRESH=0", ""]
        for run in runs:
            key = f"{run['out']}.key"
            body = ["FRESH=1", f'rm -f "{key}"',
                    f"$NAMD {run['conf']} > {run['log']} 2>&1 || {{ echo \"ERROR: NAMD failed on {run['conf']}; see "
                    f"{run['log']}\" >&2; exit 1; }}",
                    f'grep -q "WallClock:" "{run["log"]}" || {{ echo "ERROR: NAMD did not finish {run["conf"]}; see '
                    f'{run["log"]}" >&2; exit 1; }}',
                    f'echo "$KEY" > "{key}"']
            sh.append(f"# {run['tag']}: {run['stage']['kind']} {run['steps']} steps, map coupling {run['gscale']:g}")
            sh.append(f'KEY=$( {{ echo "$KEY"; cat "{run["conf"]}"; }} | cksum )')
            if s["resume"]:
                done = (f'[ "$FRESH" = 0 ] && [ -s "{run["out"]}.coor" ] && grep -q "WallClock:" "{run["log"]}" '
                        f'2>/dev/null && [ "$(cat "{key}" 2>/dev/null)" = "$KEY" ]')
                sh += [f"if {done}; then", f'    echo "{run["tag"]} already finished - skipped"', "else"]
                sh += ["    " + b for b in body] + ["fi"]
            else:
                sh += body
            sh.append("")
        sh += ["# The fitted model (solute only) in the map's frame",
               f"python3 {n('mdff', 'py')} final", require_file(n("fitted", "pdb"), "the final conversion"),
               f'echo "Fitted model: {n("fitted", "pdb")}"', ""]
        if s["continuation"]:
            sh[-1:] = ["# ... and every atom in the simulation frame, for a continuation step",
                       require_file(n("fitted-system", "rst7"), "the final conversion"),
                       f'echo "Final coordinates of the whole system: {n("fitted-system", "rst7")}"', ""]
        return "\n".join(sh)
