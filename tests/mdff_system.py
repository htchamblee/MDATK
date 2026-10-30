"""A small D-RNA + L-RNA test system for the NAMD MDFF step: helices built by MDATK, a simulated
map and a *names-and-masses-only* parm7.

The parm7 written here holds atom names, residue labels, atomic numbers, masses, atom types and a
trivial one-type Lennard-Jones table - enough for every file MDATK prepares around a topology, but
no force field.  Its title says so; it must never be simulated.  Real topologies come from the
Amber build step (tleap + ParmEd).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mdatk  # noqa: F401
except ImportError:  # run as a script from a source checkout
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mdatk import builder, density, pdbtools
from mdatk.io import mrc
from mdatk.io.parm7 import write_parm7

TEST_TITLE = "MDATK TEST TOPOLOGY - NAMES AND MASSES ONLY, NO FORCE FIELD - DO NOT SIMULATE"
NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "MG": 12, "NA": 11, "CL": 17}
MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 15: 30.974, 12: 24.305, 11: 22.990, 17: 35.45, 0: 0.0}
L_TYPES = {"O3'": "LD", "O5'": "LD", "O4'": "LE", "O2'": "LH"}


def mixed_pdb(seq: str = "GCGC", gap: float = 26.0) -> str:
    """A D duplex (chains A/B) next to an L duplex (chains C/D), Amber residue names, hydrogens."""
    d = builder.build_helix(seq, chirality="D", chains=("A", "B"), scheme="amber", hydrogens=True)
    l_ = builder.build_helix(seq, chirality="L", chains=("C", "D"), scheme="amber", hydrogens=True,
                             start=(1, 1))
    out = []
    for text, dx in ((d, gap / 2), (l_, -gap / 2)):
        for line in text.splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                x = float(line[30:38]) + dx
                out.append(line[:30] + f"{x:8.3f}" + line[38:])
            elif line.startswith("TER"):
                out.append("TER")
    return "\n".join(out) + "\nEND\n"


def atoms_of(pdb_text: str):
    """[(chain, resseq, resname, atom name, element, xyz)] in file order."""
    s = pdbtools.read_pdb(pdb_text)
    out = []
    for res in s.residues():
        for a in res.atoms:
            el = (a.element or a.name.lstrip("0123456789")[:1]).upper()
            out.append((res.chain, res.resseq, res.name, a.name, el, a.xyz))
    return out


def with_solvent(atoms, waters: int = 6, mg: int = 1, spacing: float = 3.1, start=(0.0, 0.0, 14.0)):
    """Append a few waters (O, H1, H2) and Mg2+ ions as separate residues."""
    out = list(atoms)
    x0, y0, z0 = start
    for k in range(waters):
        o = (x0 + k * spacing, y0, z0)
        out += [("W", k + 1, "WAT", "O", "O", o), ("W", k + 1, "WAT", "H1", "H", (o[0] + 0.96, o[1], o[2])),
                ("W", k + 1, "WAT", "H2", "H", (o[0] - 0.24, o[1] + 0.93, o[2]))]
    for k in range(mg):
        out.append(("M", k + 1, "MG", "MG", "MG", (x0 + k * spacing, y0 + 6.0, z0)))
    return out


def write_topology(path: str, atoms, box: Optional[Sequence[float]] = None, hmr: bool = False,
                   extra_flags: Sequence[Tuple[str, str, list]] = (), l_types: bool = True, numextra: int = 0,
                   ifbox: Optional[int] = None) -> None:
    labels: List[str] = []
    pointers: List[int] = []
    last = None
    names, numbers, masses, types = [], [], [], []
    for k, (chain, num, resname, name, el, _xyz) in enumerate(atoms):
        key = (chain, num, resname)
        if key != last:
            labels.append(resname)
            pointers.append(k + 1)
            last = key
        z = NUMBERS.get(el, 6)
        names.append(name)
        numbers.append(z)
        masses.append(MASSES[z])
        lres = resname.startswith("L") and len(resname) <= 3 and resname not in ("LI",)
        types.append(L_TYPES.get(name, el[:2]) if (l_types and lres) else el[:2])
    if hmr:  # ParmEd HMassRepartition: solute H 3.024, the heavy atom gives the difference (water untouched)
        for k, (chain, num, resname, name, el, _xyz) in enumerate(atoms):
            if el == "H" and resname != "WAT":
                masses[k] = 3.024
    n = len(names)
    pointers_section = [0] * 31
    pointers_section[0] = n
    pointers_section[1] = 1
    pointers_section[11] = len(labels)
    pointers_section[27] = ifbox if ifbox is not None else (1 if box else 0)
    pointers_section[30] = numextra  # NUMEXTRA: extra points of four-site water
    sections = [("POINTERS", "i", pointers_section), ("ATOM_NAME", "a", names), ("CHARGE", "e", [0.0] * n),
                ("ATOMIC_NUMBER", "i", numbers), ("MASS", "e", masses), ("ATOM_TYPE_INDEX", "i", [1] * n),
                ("NONBONDED_PARM_INDEX", "i", [1]), ("RESIDUE_LABEL", "a", labels),
                ("RESIDUE_POINTER", "i", pointers), ("LENNARD_JONES_ACOEF", "e", [1.0e6]),
                ("LENNARD_JONES_BCOEF", "e", [1.0e3]), ("AMBER_ATOM_TYPE", "a", types)]
    if box:
        sections.append(("BOX_DIMENSIONS", "e", [box[3], box[0], box[1], box[2]]))
    sections += list(extra_flags)
    write_parm7(path, sections, title=TEST_TITLE)


def write_rst7(path: str, xyz, box: Optional[Sequence[float]] = None, velocities=None) -> None:
    values = [c for p in xyz for c in p] + ([c for v in velocities for c in v] if velocities else [])
    with open(path, "w", encoding="ascii") as fh:
        fh.write("MDATK test coordinates\n")
        fh.write(f"{len(xyz):6d}\n")
        for k in range(0, len(values), 6):
            fh.write("".join(f"{v:12.7f}" for v in values[k:k + 6]) + "\n")
        if box:
            fh.write("".join(f"{v:12.7f}" for v in box) + "\n")


def write_map(path: str, atoms, resolution: float = 4.0, spacing: float = 1.5) -> mrc.DensityMap:
    xyz = np.array([a[5] for a in atoms if a[4] != "H" and a[2] not in ("WAT", "MG")])
    els = [a[4] for a in atoms if a[4] != "H" and a[2] not in ("WAT", "MG")]
    dmap = density.simulate_map(xyz, density.atom_weights(els), resolution, spacing=spacing)
    mrc.write_map(path, dmap)
    return dmap


def rotation(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    t = math.radians(degrees)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(t) * kx + (1 - math.cos(t)) * (kx @ kx)


def moved(atoms, rot: np.ndarray, shift) -> list:
    """The atoms after x -> rot x + shift (what tleap does to a solute when it solvates it)."""
    out = []
    for a in atoms:
        p = rot @ np.array(a[5]) + np.asarray(shift)
        out.append(a[:5] + (tuple(float(v) for v in p),))
    return out


def model_pdb(atoms) -> str:
    """PDB text of the solute (the map-frame model)."""
    lines = []
    for k, (chain, num, resname, name, el, (x, y, z)) in enumerate(atoms):
        if resname in ("WAT", "MG"):
            continue
        nm = f" {name:<3s}" if len(name) < 4 else name
        lines.append(f"ATOM  {k + 1:5d} {nm} {resname:>3s} {chain}{num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00          {el:>2s}")
    return "\n".join(lines) + "\nEND\n"


# ------------------------------------------------------------------------------------------ example
EXAMPLE_README = """# NAMD MDFF example: a D-RNA and an L-RNA duplex

Files for the `namd-mdff` planner step (NAMD reading the Amber topology natively), made by
`python3 tests/mdff_system.py --example examples/namd_mdff` from the MDATK source tree.

| File | What it is |
|---|---|
| `mixed_dl_model.pdb` | the model in the map's frame: a D duplex (chains A/B, GCGC) next to its L mirror image (chains C/D, LG/LC), MDATK helices |
| `mixed_dl_4A.mrc` | a 4 Å map simulated from that model |
| `namd_mdff.mdatk.toml` | a project: model → Amber build (OL3 + CP, NBfix19, 0BPh, L-RNA fixes, nanoMg, TIP3P in a rectangular box, HMR) → NAMD MDFF, map and model connected |
| `generated/02_namd-mdff/` | what `mdatk workflow write` writes for the MDFF step: one NAMD configuration per stage (`amber on`, `mgridforce`, `extraBonds`), the energy comparison with sander, the preparation helper and the run script |
| `prepared/` | what the preparation helper writes around a topology: the MDFF potential (`.dx`), the coupling PDB, the `extraBonds` restraint files (L-RNA-safe chirality impropers, secondary-structure dihedrals of the paired residues, base-pair hydrogen-bond walls), the periodic cell (`.xsc`) and its log |
| `prepared/stand-in/` | the topology and coordinates `prepared/` was made from |

**`prepared/stand-in/` is not a force field.** Without AmberTools in the build environment the
example cannot contain a real tleap topology, so `prepared/` was computed from a parm7 that holds
only atom names, residue labels, atomic numbers, masses and atom types (its title says so), solvated
and moved the way tleap moves a solute, to show the frame recovery. Never simulate it.

To run the example for real you need AmberTools, NAMD and an L-RNA residue library with its
parameters, which AmberTools does not include (e.g. Dudek & Trylska's `nucleic12_LRNA.lib` and
`parm10_LRNA.dat`). Add the library and parameter files to the Amber build step (Force field:
extra residue libraries and parameter files) - until then the planner reports that the L-RNA
residues need them. Then write the project (`mdatk workflow write -p namd_mdff.mdatk.toml`) and run
the two step scripts: the Amber build makes the real topology, the MDFF script prepares the same
files from it, compares sander and NAMD energies and runs the stages. The box is rectangular because
NAMD fits the MDFF potential into a truncated octahedron only with a small margin.

In `prepared/*_chirality_*.txt` every D residue has the D targets (C1' -35.01, C2' 40.03, C3' 38.30,
C4' 33.31 degrees) and every L residue (LG / LC) their mirror image: nothing pulls an L sugar towards
the D centre, which is what VMD's `chirality` / `ssrestraints` restraints would do (they assume
D-nucleotides; MDATK never uses them).
"""


def make_example(out: str) -> None:
    """examples/namd_mdff: model, map, project, generated step files and prepared files."""
    import io
    import shutil
    from contextlib import redirect_stdout

    from mdatk.config import PlanEdge, PlanNode, Project
    from mdatk.workflows import TOOLS
    from mdatk.workflows.plan import Plan
    from mdatk.workflows.scripts import namd_mdff as nm

    out = os.path.abspath(out)  # every path below is made relative to the example folder
    os.makedirs(out, exist_ok=True)
    for sub in ("generated", "prepared"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    atoms = atoms_of(mixed_pdb())
    heavy = [a for a in atoms if a[4] != "H"]
    model = os.path.join(out, "mixed_dl_model.pdb")
    with open(model, "w") as fh:
        fh.write(model_pdb(heavy))
    mapfile = os.path.join(out, "mixed_dl_4A.mrc")
    write_map(mapfile, atoms, resolution=4.0, spacing=1.5)
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(EXAMPLE_README)

    # the project: model -> Amber build -> NAMD MDFF
    p = Project()
    wf = p.workflow
    wf.project_name, wf.stamp = "mixeddl", "20261029-1200"
    seq, db = "GCGC GCGC GCGC GCGC", "(((( )))) (((( ))))"
    nodes = [
        PlanNode(id="model", tool="input-file", step=0, x=40, y=40,
                 settings={"path": "mixed_dl_model.pdb", "kind": "pdb"}),
        PlanNode(id="map", tool="input-file", step=0, x=40, y=260,
                 settings={"path": "mixed_dl_4A.mrc", "kind": "map"}),
        PlanNode(id="build", tool="amber-build", step=1, x=320, y=40, settings=TOOLS["amber-build"].normalise(
            {"water": "tip3p", "box": "rect", "mg_model": "nano", "mg_count": 4, "hmr": True})),
        PlanNode(id="mdff", tool="namd-mdff", step=2, x=620, y=40, settings=TOOLS["namd-mdff"].normalise(
            {"sequence": seq, "dotbracket": db})),
    ]
    wf.nodes = nodes
    wf.edges = [PlanEdge("model", "file", "build", "structure"), PlanEdge("build", "topology", "mdff", "topology"),
                PlanEdge("build", "coordinates", "mdff", "coordinates"), PlanEdge("map", "file", "mdff", "map"),
                PlanEdge("model", "file", "mdff", "model")]
    project_path = os.path.join(out, "namd_mdff.mdatk.toml")
    p.save(project_path)
    loaded = Project.load(project_path)
    plan = Plan(loaded.workflow, os.path.join(out, "generated"))
    plan.write(["mdff"], master=False)
    for root, _dirs, files in os.walk(os.path.join(out, "generated")):
        for f in files:  # absolute paths of this machine -> relative to the example folder
            path = os.path.join(root, f)
            text = open(path).read()
            new = text.replace(os.path.abspath(out) + "/", "../../")
            if new != text:
                open(path, "w").write(new)

    # prepared files, from a stand-in topology solvated and moved as tleap would
    prep = os.path.join(out, "prepared")
    stand = os.path.join(prep, "stand-in")
    os.makedirs(stand)
    rot = rotation((0.2, 0.3, 1.0), 23.0)
    solvated = with_solvent(moved(atoms, rot, (36.0, 33.0, 31.0)), waters=8, mg=4, start=(18.0, 12.0, 58.0))
    box = (84.0, 84.0, 84.0, 90.0, 90.0, 90.0)
    write_topology(os.path.join(stand, "mixed_dl_stand-in.parm7"), solvated, box=box, hmr=True)
    write_rst7(os.path.join(stand, "mixed_dl_stand-in.rst7"), [a[5] for a in solvated], box=box)
    step = plan.steps["mdff"]
    cfg = TOOLS["namd-mdff"].prep_settings(step.settings, step.ctx)
    names = {k: os.path.basename(v) for k, v in cfg.items() if isinstance(v, str) and v}
    cfg.update({k: os.path.join(prep, names[k]) for k in ("potential", "grid_pdb", "restraint_pdb", "extrabonds",
                                                          "xsc", "prep_json", "coordinates")})
    for key in ("ssdihedrals", "basepairs"):  # restraint files named inside their own settings
        cfg[key]["file"] = os.path.join(prep, os.path.basename(cfg[key]["file"]))
    cfg["topology"] = os.path.join(stand, "mixed_dl_stand-in.parm7")
    cfg["model"], cfg["map"] = model, mapfile
    buf = io.StringIO()
    with redirect_stdout(buf):
        nm.start_coordinates(cfg, os.path.join(stand, "mixed_dl_stand-in.rst7"))
        nm.prep(cfg)
    log = buf.getvalue().replace(os.path.abspath(out) + "/", "")
    with open(os.path.join(prep, names["prep_json"].replace(".json", ".log")), "w") as fh:
        fh.write(log)
    for f in os.listdir(prep):  # no absolute paths of this machine in the example
        path = os.path.join(prep, f)
        if os.path.isfile(path):
            text = open(path).read()
            new = text.replace(os.path.abspath(out) + "/", "")
            if new != text:
                open(path, "w").write(new)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--example":
        make_example(sys.argv[2])
    else:
        print("usage: python3 tests/mdff_system.py --example OUTPUT_FOLDER", file=sys.stderr)
        sys.exit(2)
