# NAMD MDFF example: a D-RNA and an L-RNA duplex

Files for the `namd-mdff` planner step (NAMD reading the Amber topology natively), made by
`python3 tests/mdff_system.py --example examples/namd_mdff` from the MDATK source tree.

| File | What it is |
|---|---|
| `mixed_dl_model.pdb` | the model in the map's frame: a D duplex (chains A/B, GCGC) next to its L mirror image (chains C/D, LG/LC), MDATK helices |
| `mixed_dl_4A.mrc` | a 4 Å map simulated from that model |
| `namd_mdff.mdatk.toml` | a project: model → Amber build (OL3 + CP, NBfix19, 0BPh, L-RNA fixes, nanoMg, TIP3P, HMR) → NAMD MDFF, map and model connected |
| `generated/02_namd-mdff/` | what `mdatk workflow write` writes for the MDFF step: one NAMD configuration per stage (`amber on`, `mgridforce`, `extraBonds`), the energy comparison with sander, the preparation helper and the run script |
| `prepared/` | what the preparation helper writes around a topology: the MDFF potential (`.dx`), the coupling PDB, the `extraBonds` restraint files (L-RNA-safe chirality impropers, secondary-structure dihedrals of the paired residues, base-pair hydrogen-bond walls), the periodic cell (`.xsc`) and its log |
| `prepared/stand-in/` | the topology and coordinates `prepared/` was made from |

**`prepared/stand-in/` is not a force field.** Without AmberTools in the build environment the
example cannot contain a real tleap topology, so `prepared/` was computed from a parm7 that holds
only atom names, residue labels, atomic numbers, masses and atom types (its title says so), solvated
and moved the way tleap moves a solute, to show the frame recovery. Never simulate it. With
AmberTools and NAMD installed, write the project (`mdatk workflow write -p namd_mdff.mdatk.toml`)
and run the two step scripts: the Amber build makes the real topology, the MDFF script prepares the
same files from it, compares sander and NAMD energies and runs the stages.

In `prepared/*_chirality_*.txt` every D residue has the D targets (C1' -35.01, C2' 40.03, C3' 38.30,
C4' 33.31 degrees) and every L residue (LG / LC) their mirror image: nothing pulls an L sugar towards
the D centre, which is what VMD's `chirality` / `ssrestraints` restraints would do (they assume
D-nucleotides; MDATK never uses them).
