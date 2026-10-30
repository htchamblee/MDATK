"""GROMACS generator: Amber topology conversion (ParmEd or acpype), mdp files
for minimisation / NVT / NPT / production, and a grompp + mdrun script for
GROMACS 2024+ (workstation) or 2023.3 (cluster module)."""

from __future__ import annotations

from typing import Any, Dict, List

from .. import PROJECT_ABBREV, __version__
from .base import (PortValue, Tool, env_section, extra_lines, fortran_float, require_file,
                   script_header, separate_check)
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section
from .scripts import embed
from .scripts import amber2gmx as _amber2gmx
from .scripts import gmx_posre as _gmx_posre

_EM = {"integrator": "steep"}
_DYN = {"integrator": ("md", "sd")}

STAGE_OPTIONS = [
    Option("name", "Name", "text", "stage", "Stage name used in file names (em, nvt, npt, md ...)."),
    Option("integrator", "Integrator", "radio", "md",
           "steep: energy minimisation; md: leap-frog MD; sd: stochastic (Langevin) dynamics.",
           choices=[("steep", "Minimisation (steep)"), ("md", "MD (md)"), ("sd", "Langevin (sd)")]),
    Option("nsteps", "Steps (nsteps)", "int", 50000, "Number of steps.", minimum=1, maximum=10_000_000_000),
    Option("emtol", "Force tolerance (emtol)", "float", 1000.0, "Stop when the maximum force is below "
           "this (kJ/mol/nm).", minimum=0.1, maximum=1e6, step=10.0, digits=1, requires=_EM),
    Option("emstep", "Initial step (emstep)", "float", 0.01, "nm", minimum=0.0001, maximum=1.0, step=0.001,
           digits=4, requires=_EM),
    Option("dt", "Time step (dt)", "float", 0.002, "ps (0.004 needs hydrogen mass repartitioning).",
           minimum=0.0005, maximum=0.005, step=0.001, digits=4, unit="ps", requires=_DYN),
    Option("velocities", "Velocities", "radio", "auto",
           "auto: generate after a minimisation, otherwise continue from the previous checkpoint.",
           choices=[("auto", "Auto"), ("new", "Generate"), ("continue", "Continue")], requires=_DYN),
    Option("tcoupl", "Thermostat (tcoupl)", "radio", "V-rescale", "Ignored by the sd integrator.",
           choices=[("V-rescale", "V-rescale"), ("Nose-Hoover", "Nose-Hoover"), ("Berendsen", "Berendsen")],
           requires={"integrator": "md"}),
    Option("tc_grps", "Coupling groups (tc-grps)", "text", "System",
           "Thermostat groups; anything other than 'System' needs an index file.", requires=_DYN),
    Option("tau_t", "tau-t", "float", 0.1, "Thermostat time constant (ps); for sd the inverse friction.",
           minimum=0.001, maximum=100.0, step=0.1, digits=3, requires=_DYN),
    Option("ref_t", "Temperature (ref-t)", "float", 300.0, "K", minimum=0.0, maximum=1000.0, step=1.0,
           digits=1, unit="K", requires=_DYN),
    Option("pcoupl", "Barostat (pcoupl)", "radio", "no", "C-rescale works for equilibration and "
           "production; Parrinello-Rahman needs an equilibrated box.",
           choices=[("no", "None (NVT)"), ("C-rescale", "C-rescale"), ("Parrinello-Rahman", "Parrinello-Rahman"),
                    ("Berendsen", "Berendsen")], requires=_DYN),
    Option("tau_p", "tau-p", "float", 2.0, "Barostat time constant (ps).", minimum=0.1, maximum=100.0,
           step=0.5, digits=2, requires={"integrator": ("md", "sd"),
                                          "pcoupl": ("C-rescale", "Parrinello-Rahman", "Berendsen")}),
    Option("ref_p", "Pressure (ref-p)", "float", 1.0, "bar", minimum=0.0, maximum=10000.0, step=0.1,
           digits=2, requires={"integrator": ("md", "sd"),
                               "pcoupl": ("C-rescale", "Parrinello-Rahman", "Berendsen")}),
    Option("compressibility", "Compressibility", "float", 4.5e-5, "bar⁻¹ (water: 4.5e-5).", minimum=0.0,
           maximum=1.0, digits=8, requires={"integrator": ("md", "sd"),
                                            "pcoupl": ("C-rescale", "Parrinello-Rahman", "Berendsen")}),
    Option("constraints", "Constraints", "radio", "h-bonds", "Bonds constrained with LINCS.",
           choices=[("h-bonds", "Bonds to H"), ("all-bonds", "All bonds"), ("none", "None")]),
    Option("cutoff", "Cutoff (rvdw = rcoulomb)", "float", 1.0, "nm (Amber force fields: 0.8-1.2).",
           minimum=0.6, maximum=3.0, step=0.1, digits=2, unit="nm"),
    Option("posres", "RNA position restraints (-DPOSRES)", "bool", False,
           "Restrain nucleic-acid heavy atoms to the stage's starting coordinates."),
    Option("nstxout_compressed", "Trajectory frame every", "int", 5000, "nstxout-compressed (0: none).",
           minimum=0, maximum=100_000_000, requires=_DYN),
    Option("nstenergy", "Energies every", "int", 5000, "nstenergy / nstlog.", minimum=1, maximum=100_000_000),
    Option("production", "Production stage", "bool", False,
           "Its trajectory is passed on to analysis / visualisation steps.", requires=_DYN),
    Option("extra_mdp", "Extra mdp lines", "multiline", "", "Added at the end of the mdp file "
           "(later lines override earlier ones).", placeholder="nstcalcenergy = 100"),
]

_TEMPLATES = [
    ("em", "Energy minimisation", dict(name="em", integrator="steep", nsteps=50000, emtol=1000.0,
                                       constraints="none")),
    ("nvt", "NVT equilibration (RNA restrained)", dict(name="nvt", integrator="md", nsteps=50000,
                                                        velocities="auto", posres=True)),
    ("npt", "NPT equilibration (RNA restrained)", dict(name="npt", integrator="md", nsteps=50000,
                                                        pcoupl="C-rescale", posres=True)),
    ("md", "Production (NPT)", dict(name="md", integrator="md", nsteps=50_000_000, pcoupl="C-rescale",
                                    production=True)),
    ("md-sd", "Production (Langevin, NPT)", dict(name="md", integrator="sd", nsteps=50_000_000,
                                                 tau_t=1.0, pcoupl="C-rescale", production=True)),
]


@register
class GromacsTool(Tool):
    name = "gromacs"
    label = "GROMACS · MD"
    category = "GROMACS"
    tag = "gromacs"
    description = ("Converts an Amber topology to GROMACS (ParmEd or acpype) and writes mdp files and "
                   "a grompp / mdrun script for minimisation, equilibration and production.")
    inputs = [Port("topology", "Topology (parm7/top)", ("parm7", "top")),
              Port("coordinates", "Coordinates (rst7/gro)", ("rst7", "gro"))]
    sections = [
        Section("setup", "Version & conversion", [
            Option("gmx_version", "GROMACS version", "radio", "2024",
                   "2024+ on the workstation, 2023.3 on the cluster (loaded as a module).",
                   choices=[("2024", "2024+ (local)"), ("2023", "2023.3 (cluster)")]),
            Option("module", "Module to load", "text", "GROMACS/2023.3-CUDA-11.8.0",
                   "'module load' line for the cluster version (empty: none).",
                   requires={"gmx_version": "2023"}),
            Option("gmx", "gmx executable", "text", "gmx", "gmx, gmx_mpi or a full path."),
            Option("convert", "Convert Amber files with", "radio", "parmed",
                   "ParmEd keeps off-diagonal Lennard-Jones pairs (NBfix) as [ nonbond_params ]; acpype is an "
                   "alternative; 'none' uses GROMACS files as they are.",
                   choices=[("parmed", "ParmEd"), ("acpype", "acpype"), ("none", "None (top + gro)")]),
            Option("mass_repartition", "Hydrogen mass repartitioning (mdp)", "bool", False,
                   "mass-repartition-factor = 3 in every mdp (GROMACS 2024+ only) so dt = 0.004 ps works "
                   "with constraints = h-bonds. Not needed when the Amber topology already has HMR."),
            Option("posres_fc", "Restraint force constant", "float", 1000.0,
                   "kJ/mol/nm² for stages with RNA position restraints.", minimum=1.0, maximum=1e6,
                   step=100.0, digits=1),
            Option("allow_nbfix_loss", "Allow losing NBfix pairs", "bool", False,
                   "Normally the conversion stops when the Amber topology has off-diagonal LJ pairs that "
                   "do not appear in the GROMACS topology."),
        ], tooltip="Which GROMACS and how the Amber topology is converted."),
        Section("stages", "Stages", STAGE_OPTIONS, repeat=True, item_label="name", templates=_TEMPLATES,
                default_items=("em", "nvt", "npt", "md"),
                tooltip="Each stage starts from the previous stage's coordinates and checkpoint."),
        Section("mdrun", "mdrun", [
            Option("gpu", "Offload to the GPU", "bool", True, "-nb gpu -pme gpu -bonded gpu"),
            Option("update_gpu", "Update on the GPU (-update gpu)", "bool", False,
                   "Fastest on one GPU; not available with every setting.", requires={"gpu": True}),
            Option("ntomp", "OpenMP threads (-ntomp)", "int", 0, "0: let GROMACS decide.", minimum=0,
                   maximum=1024),
            Option("maxwarn", "grompp -maxwarn", "int", 0, "Number of grompp warnings to accept.",
                   minimum=0, maximum=100),
            Option("extra_mdrun", "Extra mdrun flags", "text", "", "", placeholder="-pin on"),
            Option("separate_production", "Separate production script", "bool", True,
                   "Production stages go into their own script."),
        ]),
        env_section("module load GROMACS/2023.3-CUDA-11.8.0"),
    ]

    def output_ports(self, s):
        ports = [Port("topology", "GROMACS topology", ("top",)), Port("structure", "Final structure", ("gro",))]
        if any(st["integrator"] != "steep" and st["nstxout_compressed"] for st in s.get("stages", [])):
            ports.append(Port("trajectory", "Trajectory (xtc)", ("xtc",), multiple=True))
        return ports

    # names
    def _tags(self, s) -> List[str]:
        from .naming import sanitize_tag
        return [sanitize_tag(st["name"] or f"stage{k + 1}") for k, st in enumerate(s["stages"])]

    def _top(self, s, ctx) -> str:
        if s["convert"] == "none":
            return ctx.inp("topology")
        return ctx.name("convert", "top")

    def _gro(self, s, ctx) -> str:
        if s["convert"] == "none":
            return ctx.inp("coordinates")
        return ctx.name("convert", "gro")

    def output_values(self, s, ctx):
        tags = self._tags(s)
        out: Dict[str, PortValue] = {}
        top = self._top(s, ctx)
        if top:
            out["topology"] = PortValue([ctx.from_own(top) if s["convert"] != "none"
                                         else ctx.value("topology").path], "top")
        if tags:
            out["structure"] = PortValue([ctx.own(tags[-1], "gro")], "gro")
        prod = [k for k, st in enumerate(s["stages"]) if st["production"] and st["integrator"] != "steep"
                and st["nstxout_compressed"]]
        if not prod:
            dyn = [k for k, st in enumerate(s["stages"]) if st["integrator"] != "steep" and st["nstxout_compressed"]]
            prod = dyn[-1:]
        if prod:
            out["trajectory"] = PortValue([ctx.own(tags[k], "xtc") for k in prod], "xtc")
        return out

    def resources(self, s):
        return ["gromacs"] + {"parmed": ["parmed"], "acpype": ["acpype"]}.get(s["convert"], [])

    def describe(self, s):
        return f"GROMACS {s['gmx_version']}: " + " → ".join(self._tags(s))

    def check(self, s, ctx):
        out: List[Issue] = []
        stages = s["stages"]
        if not stages:
            return [Issue("error", "add at least one stage", ("stages",))]
        tags = self._tags(s)
        top_kind = ctx.value("topology").kind if ctx.value("topology") else ""
        crd_kind = ctx.value("coordinates").kind if ctx.value("coordinates") else ""
        if s["convert"] == "none" and "parm7" in (top_kind,):
            out.append(Issue("error", "Amber files need a conversion (ParmEd or acpype)", ("convert",)))
        if s["convert"] != "none" and top_kind == "top":
            out.append(Issue("error", "the topology is already a GROMACS file: set conversion to 'None'",
                             ("convert",)))
        if s["convert"] != "none" and crd_kind == "gro":
            out.append(Issue("error", "the conversion needs Amber coordinates (rst7)", ("convert",)))
        data = ctx.data("topology")
        if data.get("c4"):
            out.append(Issue("error", "the Amber topology uses 12-6-4 ion terms, which GROMACS cannot "
                                      "represent", ("input:topology",)))
        if data.get("nbfix") and s["convert"] == "acpype":
            out.append(Issue("error" if not s["allow_nbfix_loss"] else "warning",
                             "acpype drops off-diagonal Lennard-Jones pairs (NBfix19 / 0BPh) without a "
                             "warning; convert with ParmEd (MDATK carries them over as [ nonbond_params ])",
                             ("convert", "allow_nbfix_loss")))
        if data.get("nbfix") and s["convert"] == "parmed":
            out.append(Issue("info", "NBfix19 / 0BPh pairs are written as [ nonbond_params ] and 1-4 "
                                     "[ pairtypes ] by the conversion script", ("convert",)))
        if data.get("mg_pairs") and s["convert"] == "acpype":
            out.append(Issue("error" if not s["allow_nbfix_loss"] else "warning",
                             f"acpype drops the {data['mg_pairs']} Mg²⁺ pair terms (they are off-diagonal "
                             "Lennard-Jones pairs); convert with ParmEd", ("convert", "allow_nbfix_loss")))
        if data.get("mg_pairs") and s["convert"] == "parmed":
            out.append(Issue("info", f"the {data['mg_pairs']} Mg²⁺-RNA pair terms are written as "
                                     "[ nonbond_params ] by the conversion script", ("convert",)))
        if data.get("box") is False:
            out.append(Issue("error", "the topology has no periodic box", ("input:topology",)))
        for k, st in enumerate(stages):
            key, label = f"stages.{k}.", f"stage {tags[k]}"
            if tags.count(tags[k]) > 1:
                out.append(Issue("error", f"stage name '{tags[k]}' is used twice", (key + "name",)))
            if st["integrator"] == "steep":
                if st["posres"] and k == 0 and s["convert"] == "none":
                    out.append(Issue("info", f"{label}: position restraints need POSRES includes in your "
                                             "topology", (key + "posres",)))
                continue
            if st["dt"] > 0.0025 and not data.get("hmr") and not s["mass_repartition"]:
                lvl = "warning" if data.get("hmr") is None else "error"
                out.append(Issue(lvl, f"{label}: dt = {st['dt']:g} ps needs hydrogen mass repartitioning",
                                 (key + "dt",)))
            if st["constraints"] == "none" and st["dt"] > 0.001:
                out.append(Issue("error", f"{label}: dt > 0.001 ps needs constraints on bonds to H",
                                 (key + "constraints", key + "dt")))
            if "Berendsen" in (st["tcoupl"] if st["integrator"] == "md" else "", st["pcoupl"]):
                out.append(Issue("error" if s["maxwarn"] == 0 else "warning",
                                 f"{label}: grompp warns about Berendsen coupling and stops unless -maxwarn "
                                 "≥ 1; use V-rescale / C-rescale", (key + "tcoupl", key + "pcoupl", "maxwarn")))
            gen_vel = st["velocities"] == "new" or (st["velocities"] == "auto" and
                                                     (k == 0 or stages[k - 1]["integrator"] == "steep"))
            if st["pcoupl"] == "Parrinello-Rahman" and gen_vel:
                out.append(Issue("error" if s["maxwarn"] == 0 else "warning",
                                 f"{label}: grompp warns about Parrinello-Rahman with new velocities "
                                 "(gen-vel = yes) and stops; equilibrate with C-rescale first",
                                 (key + "pcoupl", key + "velocities")))
            if s["gpu"] and s["update_gpu"]:
                if st["tcoupl"] == "Nose-Hoover" and st["integrator"] == "md":
                    out.append(Issue("error", f"{label}: -update gpu does not support Nose-Hoover",
                                     (key + "tcoupl", "update_gpu")))
                if st["integrator"] == "sd":
                    out.append(Issue("info", f"{label}: -update gpu needs integrator md; this stage runs "
                                             "the update on the CPU", (key + "integrator",)))
            if st["integrator"] == "sd" and st["tcoupl"] != "V-rescale":
                out.append(Issue("info", f"{label}: the sd integrator is its own thermostat; tcoupl is "
                                         "ignored", (key + "tcoupl",)))
            if st["tc_grps"].strip() not in ("System", ""):
                out.append(Issue("warning", f"{label}: tc-grps other than System need an index file "
                                            "(add -n index.ndx under 'Extra mdrun flags' / grompp)",
                                 (key + "tc_grps",)))
            if st["pcoupl"] == "Parrinello-Rahman" and k > 0 and stages[k - 1]["pcoupl"] == "no":
                out.append(Issue("warning", f"{label}: Parrinello-Rahman straight after NVT can oscillate; "
                                            "equilibrate with C-rescale first", (key + "pcoupl",)))
            if st["velocities"] == "continue" and (k == 0 or stages[k - 1]["integrator"] == "steep"):
                out.append(Issue("error", f"{label}: nothing to continue after a minimisation",
                                 (key + "velocities",)))
        if s["gpu"] and s["update_gpu"] and data.get("four_site_water"):
            out.append(Issue("error", "-update gpu does not support virtual sites (4-site water such as "
                                      "OPC / TIP4P-Ew)", ("update_gpu",)))
        if s["mass_repartition"] and s["gmx_version"] == "2023":
            out.append(Issue("error", "mass-repartition-factor exists only in GROMACS 2024+",
                             ("mass_repartition", "gmx_version")))
        if s["mass_repartition"] and data.get("hmr"):
            out.append(Issue("error", "the Amber topology already has repartitioned hydrogen masses; do not "
                                      "repartition twice", ("mass_repartition",)))
        if s["convert"] == "none" and any(st["posres"] for st in stages):
            out.append(Issue("warning", "position restraints are generated only when converting from "
                                        "Amber; your topology must include its own POSRES blocks",
                             ("convert",)))
        if s["maxwarn"] > 0:
            out.append(Issue("warning", f"-maxwarn {s['maxwarn']} hides grompp warnings", ("maxwarn",)))
        temps = {st["ref_t"] for st in stages if st["integrator"] != "steep"}
        if len(temps) > 1:
            out.append(Issue("warning", "stages use different temperatures: " +
                             ", ".join(f"{t:g} K" for t in sorted(temps)), ("stages",)))
        return out

    # files ------------------------------------------------------------
    def mdp(self, st: Dict[str, Any], gen_vel: bool, continuation: bool, title: str,
            hmr: bool = False) -> str:
        L = [f"; {title}", f"; written by {PROJECT_ABBREV} {__version__}"]
        if st["posres"]:
            L.append("define                  = -DPOSRES")
        L.append(f"integrator              = {st['integrator']}")
        L.append(f"nsteps                  = {st['nsteps']}")
        if st["integrator"] == "steep":
            L += [f"emtol                   = {fortran_float(st['emtol'])}",
                  f"emstep                  = {fortran_float(st['emstep'])}"]
        else:
            L += [f"dt                      = {fortran_float(st['dt'])}",
                  f"continuation            = {'yes' if continuation else 'no'}"]
        L += ["", "; output", f"nstenergy               = {st['nstenergy']}",
              f"nstlog                  = {st['nstenergy']}"]
        if st["integrator"] != "steep":
            L.append(f"nstxout-compressed      = {st['nstxout_compressed']}")
        L += ["", "; interactions", "cutoff-scheme           = Verlet", "nstlist                 = 10",
              "pbc                     = xyz", "coulombtype             = PME",
              f"rcoulomb                = {fortran_float(st['cutoff'])}",
              f"rvdw                    = {fortran_float(st['cutoff'])}",
              "DispCorr                = EnerPres",
              f"constraints             = {st['constraints']}"]
        if st["constraints"] != "none":
            L.append("constraint-algorithm    = lincs")
        if hmr:
            L.append("mass-repartition-factor = 3")
        if st["integrator"] != "steep":
            L += ["", "; temperature"]
            if st["integrator"] == "md":
                L.append(f"tcoupl                  = {st['tcoupl']}")
            groups = st["tc_grps"].split() or ["System"]
            L += [f"tc-grps                 = {' '.join(groups)}",
                  f"tau-t                   = {' '.join(fortran_float(st['tau_t']) for _ in groups)}",
                  f"ref-t                   = {' '.join(fortran_float(st['ref_t']) for _ in groups)}"]
            L += ["", "; pressure", f"pcoupl                  = {st['pcoupl']}"]
            if st["pcoupl"] != "no":
                L += ["pcoupltype              = isotropic",
                      f"tau-p                   = {fortran_float(st['tau_p'])}",
                      f"ref-p                   = {fortran_float(st['ref_p'])}",
                      f"compressibility         = {st['compressibility']:g}"]
                if st["posres"]:
                    L.append("refcoord-scaling        = com")
            L += ["", "; velocities", f"gen-vel                 = {'yes' if gen_vel else 'no'}"]
            if gen_vel:
                L += [f"gen-temp                = {fortran_float(st['ref_t'])}", "gen-seed                = -1"]
        extra = extra_lines(st["extra_mdp"])
        if extra:
            L += ["", "; extra"] + extra
        return "\n".join(L) + "\n"

    @staticmethod
    def _prod_split(s) -> bool:
        return bool(s["separate_production"] and any(st["production"] and st["integrator"] != "steep"
                                                     for st in s["stages"]))

    def generate(self, s, ctx):
        files: List[GeneratedFile] = []
        n = ctx.name
        tags = self._tags(s)
        top, gro = self._top(s, ctx), self._gro(s, ctx)
        script = n(self.tag, "sh")
        prod_script = n("gromacs-prod", "sh")
        notes = [f"GROMACS {'2024+' if s['gmx_version'] == '2024' else '2023.3'}."]

        def preamble(title: str, name: str) -> List[str]:
            lines = script_header(title, ctx, s, name, notes)
            if s["gmx_version"] == "2023" and s["module"].strip():
                lines += [f"module load {s['module'].strip()}", ""]
            return lines + [f'GMX="{s["gmx"].strip() or "gmx"}"', ""]

        main = preamble("GROMACS: conversion and MD stages", script)
        waiting: List[str] = []
        if self._prod_split(s):  # the main script must not report success while production has not run
            # (mdrun writes a stage's .gro last); without per-stage resume, the first lines also keep a re-run
            # from redoing the conversion and stages production started from
            done = [f"[ -s {n(tags[k], 'gro')} ]" for k, st in enumerate(s["stages"])
                    if st["production"] and st["integrator"] != "steep"]
            head, waiting = separate_check(prod_script, done, "the production stages")
            main += head
        need_posres = any(st["posres"] for st in s["stages"])
        if s["convert"] == "parmed":
            conv = n("convert", "py")
            files.append(GeneratedFile(conv, self._convert_script(s, ctx), "helper",
                                       "ParmEd conversion Amber -> GROMACS"))
            main += ["# Convert the Amber topology / coordinates with ParmEd",
                     f"python3 {conv} {ctx.inp('topology')} {ctx.inp('coordinates')} {top} {gro}",
                     require_file(top, "ParmEd"), ""]
        elif s["convert"] == "acpype":
            base = n("acpype")
            main += ["# Convert the Amber topology / coordinates with acpype",
                     f"rm -rf {base}.amb2gmx",
                     f"acpype -p {ctx.inp('topology')} -x {ctx.inp('coordinates')} -b {base}",
                     f"cp {base}.amb2gmx/{base}_GMX.top {top}",
                     f"cp {base}.amb2gmx/{base}_GMX.gro {gro}",
                     require_file(top, "acpype"), ""]
        if need_posres and s["convert"] != "none":
            posre = n("posre", "py")
            pattern = ctx.name("posre-MOLNAME", "itp").replace("MOLNAME", "{mol}")
            files.append(GeneratedFile(posre, embed(
                _gmx_posre, {"ITP_PATTERN": pattern, "FORCE": float(s["posres_fc"])},
                "main(sys.argv[1:], ITP_PATTERN, FORCE)",
                f"Add RNA position restraints to a GROMACS topology - written by {PROJECT_ABBREV} "
                f"{__version__}.\n\nUsage: python3 THIS_SCRIPT TOPOLOGY.top"), "helper",
                "adds POSRES includes"))
            main += ["# Position restraints for the RNA (used by stages with define = -DPOSRES)",
                     f"python3 {posre} {top}", ""]
        prod_split = self._prod_split(s)
        prod = preamble("GROMACS production", prod_script) if prod_split else main
        mdrun_flags = []
        if s["gpu"]:
            mdrun_flags.append("-nb gpu -pme gpu -bonded gpu")
        if s["ntomp"]:
            mdrun_flags.append(f"-ntomp {s['ntomp']}")
        if s["extra_mdrun"].strip():
            mdrun_flags.append(s["extra_mdrun"].strip())
        prev_gro, prev_cpt, prev_kind = gro, "", "start"
        for k, st in enumerate(s["stages"]):
            tag = tags[k]
            dyn = st["integrator"] != "steep"
            if dyn:
                mode = st["velocities"]
                if mode == "auto":
                    mode = "continue" if prev_kind == "md" else "new"
                gen_vel, cont = mode == "new", mode == "continue"
            else:
                gen_vel, cont = False, False
            files.append(GeneratedFile(n(tag, "mdp"), self.mdp(st, gen_vel, cont, f"{tag}: stage {k + 1}",
                                                               s["mass_repartition"]),
                                       "input", f"mdp of stage {tag}"))
            target = prod if (prod_split and st["production"] and dyn) else main
            ref = f" -r {prev_gro}" if st["posres"] else ""
            cpt = f" -t {prev_cpt}" if (dyn and cont and prev_cpt) else ""
            target += [f"# {tag}",
                       f"$GMX grompp -f {n(tag, 'mdp')} -c {prev_gro}{ref}{cpt} -p {top} -o {n(tag, 'tpr')} "
                       f"-maxwarn {s['maxwarn']}",
                       f"$GMX mdrun -deffnm {n(tag)}" + (" " + " ".join(mdrun_flags) if mdrun_flags else "")
                       + (" -update gpu" if s["gpu"] and s["update_gpu"] and st["integrator"] == "md" else ""),
                       require_file(n(tag, "gro"), f"mdrun ({tag})"), ""]
            prev_gro, prev_cpt = n(tag, "gro"), n(tag, "cpt")
            prev_kind = "md" if dyn else "min"
        main += waiting
        files.append(GeneratedFile(script, "\n".join(main), "run", "runs the GROMACS stages"))
        if prod_split:
            files.append(GeneratedFile(prod_script, "\n".join(prod), "run-separate", "production stages"))
        return files

    def _convert_script(self, s, ctx) -> str:
        return embed(_amber2gmx, {"ALLOW_NBFIX_LOSS": bool(s["allow_nbfix_loss"])},
                     "main(sys.argv[1:], ALLOW_NBFIX_LOSS)",
                     f"Amber -> GROMACS conversion with ParmEd - written by {PROJECT_ABBREV} {__version__}.\n\n"
                     "Usage: python3 THIS_SCRIPT TOPOLOGY.parm7 COORDINATES.rst7 OUT.top OUT.gro\n"
                     "Needs ParmEd (pip install parmed or the AmberTools environment).  Off-diagonal\n"
                     "Lennard-Jones pairs (NBfix19 / 0BPh) are carried over as [ nonbond_params ];\n"
                     "12-6-4 terms stop the conversion.")
