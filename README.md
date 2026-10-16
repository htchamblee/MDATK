# Molecular Dynamic Analysis Tool Kit (MDATK)

MDATK is an offline, Linux-native GTK 3 desktop application (with a companion command line
interface) for analysing molecular dynamics trajectories of RNA systems and turning them into
multi-panel MP4 videos, high-resolution time charts and selected-frame PDB files.

It reads Amber topologies (`.parm7`) and NetCDF trajectories (`.nc`) directly, can strip and
clean solvated trajectories with **cpptraj**, computes the analyses you select in a single
streaming pass, caches every result, and renders a configurable grid of panels: a rotatable
structural view with proximity glow (a 2D backbone projection, or VMD-style 3D representations:
New Cartoon with base slabs, ribbon, tube, licorice, VDW spheres, surface), time series with a
moving time cursor, per-residue heatmaps and contact maps. A cryo-EM / crystallographic density
map can be fitted to the model and drawn as a projected shadow.

Everything runs locally; MDATK makes no network calls.

Since 0.0.2b MDATK also **writes the inputs and scripts that come before the analysis**: a script
wizard and a free-form **Project Planner** generate consistently named control files for Rosetta
FARFAR2, SimRNA, RNAMake, Amber (tleap / ParmEd / pmemd), GROMACS and VMD, from sequence to MD
ensemble. MDATK only writes these files; it never runs those programs.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [Script wizard and Project Planner](#script-wizard-and-project-planner)
- [Analyses](#analyses)
- [Structural view, colouring and glow](#structural-view-colouring-and-glow)
- [3D representations](#3d-representations)
- [Density maps](#density-maps)
- [Panels, data keys and layouts](#panels-data-keys-and-layouts)
- [Outputs](#outputs)
- [Caching and reproducibility](#caching-and-reproducibility)
- [Command line reference](#command-line-reference)
- [Parameter presets](#parameter-presets)
- [Architecture and extending MDATK](#architecture-and-extending-mdatk)
- [Testing](#testing)
- [Notes and limitations](#notes-and-limitations)

---

## Installation

MDATK needs Python ≥ 3.9 on Linux. It is installed from a clone of this repository:
`pip install .` only works from **inside the cloned project directory** (the one containing
`pyproject.toml`).

**1. Install the system packages** (Debian/Ubuntu names):

```bash
sudo apt install git python3-venv python3-gi python3-gi-cairo gir1.2-gtk-3.0 ffmpeg
```

- `python3-gi`, `python3-gi-cairo` and `gir1.2-gtk-3.0` provide PyGObject, its Cairo bridge and
  GTK 3 (needed for the GUI only; the CLI works without them).
- `ffmpeg` encodes the videos.

**2. Clone the repository and enter it:**

```bash
git clone https://github.com/htchamblee/MDATK.git
cd MDATK
```

**3. Create a virtual environment and install MDATK** (still inside `MDATK/`):

```bash
# PyGObject comes from the system packages, so the venv needs --system-site-packages
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install .            # installs MDAnalysis, NumPy, SciPy, Matplotlib
pip install '.[test]'    # optional: adds pytest for running the test suite
pip install '.[vtk]'     # optional: VTK as an alternative 3D renderer (see 3D representations)
```

This installs two commands: `mdatk` (CLI) and `mdatk-gui` (GUI); `python -m mdatk` works too.
Activate the venv (`. .venv/bin/activate`) in each new shell before using them. A desktop
entry is provided in `data/mdatk.desktop`.

**Optional: cpptraj.** cpptraj (part of [AmberTools](https://ambermd.org/AmberTools.php), e.g.
`conda install -c conda-forge ambertools`) is only needed for automated solvent stripping and
cleaning. Everything else works without it, e.g. on trajectories you have already cleaned. If
you use it, set the **conda environment** that provides it (e.g. `AmberTools26`) in the cleaning
options or with `--conda-env`; every cpptraj call then runs through `conda run -n <env>`, which is
equivalent to `conda activate <env>`. When no environment is set, MDATK uses cpptraj from `PATH`
or `$AMBERHOME/bin`, and otherwise finds an environment by looking for `bin/cpptraj` in
`~/miniforge3`, `~/mambaforge`, `~/miniconda3`, `~/anaconda3`, `/opt/conda`, … (the GUI
pre-fills the field with what it finds; `mdatk list` prints the candidates).

## Quick start

```bash
mdatk demo ./demo                        # synthetic solvated RNA aptamer/target system + preset
mdatk gui ./demo/demo_preset.json        # open it in the GUI and press "Render"
# or headless:
mdatk render -p ./demo/demo_preset.json  # add --clean to strip solvent with cpptraj first
```

The demo writes a genuine Amber `demo.parm7` + `demo.nc` pair (a 12-bp aptamer/target duplex,
three Mg²⁺ ions, Na⁺ and ~4 000 waters, 120 frames) in which one end of the duplex frays and
re-anneals, a Mg²⁺ bridge between the strands breaks and reforms, and another ion visits a
phosphate. It exists to exercise every feature without real data; its geometry is schematic.

For your own system start from `examples/aptamer_target_preset.json` (relative paths in a
preset are resolved against the preset's folder) or from the GUI.

## Workflow

1. **Load** a topology (`.parm7` preferred; cleaned PDB or any MDAnalysis topology) and one or
   more trajectories (`.nc` preferred; DCD, XTC, … are concatenated in order).
2. **Clean** (optional): MDATK writes a cpptraj script that strips solvent and ions (Mg²⁺ is
   kept by default), images/re-centres (`autoimage`), optionally fits to the first frame, a
   numbered frame or a reference file, and writes a cleaned trajectory plus the matching stripped
   topology and a reference PDB. The cpptraj log streams into the GUI. Results are cached and
   reused while inputs and cleaning parameters are unchanged.
3. **Configure** residue groups, analysis parameters, colouring, proximity glow, orientation,
   grid layout and panel contents.
4. **Render**: MDATK computes the analyses the enabled outputs need (with progress), caches them,
   then writes the MP4, the charts and the PDB files, plus a parameter log.
5. **Reproduce**: save/load presets, re-render from the cache, or replay a run from its
   `MDATK_params_log.json`.

### The GUI

The main window *is* the figure: it shows the selected layout drawn by the same code that renders
the video, at the video's aspect ratio, filled with **example data** (sine-like traces, residue
heatmaps sized from your groups and a schematic duplex) so the layout can be arranged without
computing anything. *Load real frame* replaces the schematic with the reference frame of your
trajectory.

- **Drag a panel** onto another slot to swap them; panels snap to the slots of the grid preset.
- **Drag the structure panel** to rotate the model (drag its title strip, or hold Shift, to move
  it instead). The X/Y/Z sliders follow and show the exact angles in degrees and radians.
- **Double-click a panel** to edit its settings.

The settings live in drop-down menus that slide out a side panel (close it to refresh the
preview):

| Menu | Contents |
|---|---|
| **Input** | topology/trajectory/external-data choosers, frame range, time-step override, display time unit, cpptraj cleaning options (conda environment, masks, imaging, fitting), *Run cleaning now*, *Inspect system* |
| **Groups** | table of named groups (residue ranges or selections) with colour swatches and a colour picker |
| **Analyses** | enable switches and parameters for RMSD, contacts, H-bonds, O–Mg–O / Mg occupancy, π–π stacking and intermolecular energy |
| **Structure** | orientation sliders, residue-number tags, colouring, proximity and glow, appearance |
| **Layout** | grid preset and, per slot, panel type, data, residue groups, extra keys, title (follows the data until you type your own) and options; *Render still* renders one frame from real data |
| **Outputs** | output/cache folders, video, charts, PDB export, timestamps, *Re-render from cache* |

The bottom bar holds **Render**, **Cancel**, the progress bar, **Open output folder** (shown in
your file manager) and **Open video** (the most recent video); the log of the pipeline and of
cpptraj is in the expander above it. Presets are opened/saved from the header bar (Ctrl+O /
Ctrl+S, Ctrl+R renders). Long jobs run in a background thread and can be cancelled.

## Script wizard and Project Planner

The **Workflow** menu in the header bar opens two windows (they also work from the command
line, see below). Both only **write** files and scripts; nothing is executed.

- **Script wizard** (Ctrl+Shift+W): pick a program on the left, fill in its options (check boxes,
  radio buttons, text fields; every option has a tooltip), choose input files, and *Write files*.
  The options are the RNA-relevant ones for each program plus a free-text box for anything else.
  Logic checks run on every change: **errors** (conflicting or impossible settings) block writing,
  **warnings** flag likely mistakes and **infos** point out redundant or noteworthy choices; the
  offending fields are highlighted and clicking a message jumps to the field. A live preview shows
  every file that will be written.
- **Project Planner** (Ctrl+P): a canvas on which you place boxes (steps) and connect them by
  dragging from an output port (right edge) to an input port (left edge); only compatible files
  connect, and loops are refused. Double-click a box to edit it in the wizard. *Templates* insert
  pre-connected chains (FARFAR2 → models → Amber build → helix restraints → MD → MDATK analysis +
  VMD; Amber protocol; Amber → GROMACS). *Write all files* writes every step, a master script that
  runs the steps in order, and a Markdown summary of the plan. *Use for analysis* (on an MDATK
  analysis box) puts the planned topology / trajectory into Input & cleaning. The plan is saved in
  the preset (`workflow` section), so one JSON file holds the plan and the analysis settings.
  `examples/rna_plan_preset.json` is a complete example plan (open it, set the output folder, and
  open the planner).

### File names

Every file is named `<project>_<step>_<tool>_<time>.<ext>`, e.g. `apt1_03_tleap_20261015-1432.parm7`:
the project name, the step number of the box, the program or stage that writes the file, and one
time stamp per plan. The stamp is fixed the first time files are written (and stored in the preset),
so later re-generations keep the same names; *New stamp* starts a new set, and the stamp can be
switched off. Each step writes into its own folder (`03_amber-build/`), and scripts refer to the files
of other steps by relative paths (`../03_amber-build/apt1_03_parmed_….parm7`), so a whole plan folder
can be copied to a cluster. Every script starts with `cd "$(dirname "$0")"`, so it can be started from
anywhere.

### Step types

| Step | Writes | Notes |
|---|---|---|
| **Rosetta · FARFAR2** | FASTA + secstruct from a sequence and dot-bracket, a flags file, a script running *N* seeds in parallel (`xargs -P`), one silent file per run | fixed helices (`-s`, `-input_res` typed or all helix residues), cryo-EM map (`-edensity:*`), score weights, constant seeds; checks that paired bases are complementary (rna_denovo stops otherwise) |
| **Rosetta · FARFAR2 models** | script: `rna_cluster` → `extract_pdbs` → `x3dna-dssr` | ranked models `…_model-01_….pdb` (best first) feed the Amber build |
| **SimRNA** | `config.dat`, sequence, secondary-structure (pseudoknots on extra lines) and restraint files, parallel replica-exchange runs, clustering + all-atom rebuild (`clustering`, `SimRNA_trafl2pdbs … AA`) | links SimRNA's `data/` folder into every run folder |
| **RNAMake** | script for `design_rna_scaffold` (current, `--options`) or the legacy `design_rna` (`-options`): motif-graph search, sequence optimisation, thermal-fluctuation scoring | needs `RNAMAKE`; current RNAMake has no separate builder / optimiser programs |
| **Amber · system build** | tleap inputs (force field, water, box, ions, extra frcmod / libraries / leap commands), ParmEd inputs, a PDB-preparation helper and the build script | see below |
| **Amber · helix restraints** | DISANG file of the hydrogen bonds of every base pair (G–C ×3, A–U ×2, optional G·U), found by residue number and atom name in the final topology | written directly when the topology exists, otherwise by a small Python script (no ParmEd needed) the step runs |
| **Amber · MD stages** | one mdin per stage and the run scripts | see below |
| **GROMACS · MD** | ParmEd (or acpype) conversion, RNA position restraints, mdp files, grompp / mdrun scripts | 2024+ (workstation) or 2023.3 (cluster module `GROMACS/2023.3-CUDA-11.8.0`) |
| **VMD** | Tcl script + start script | one representation per residue group in the MDATK colours, bases as licorice, Mg²⁺ spheres, frame alignment, smoothing, snapshot or movie (Tachyon + ffmpeg) |
| **MDATK analysis**, **Existing file**, **Note** | nothing | link a plan to this project's analysis, feed an existing file into steps, annotate the canvas |

**Amber.** The defaults reproduce a working RNA protocol: OL3 + CP (`frcmod.99bsc0-chiol3-CaseP`),
the NBfix19 and 0BPh Lennard-Jones pair modifications applied with ParmEd (only offered with OL3 + CP,
whose atom types their masks assume), SPC/E water in a 15 Å truncated octahedron, neutralising Na⁺,
optional Mg²⁺ balanced with Cl⁻, and the stages min1 (solute restrained) → min2 → heat (Langevin from
0 K) → NPT equilibration → NPT equilibration without positional restraints → production; the mdin files
match the hand-written reference files line for line. Other choices: RNA force fields OL3, LJbb,
ROC-RNA, YIL, DESRES; waters TIP3P, OPC, OPC3, TIP4P-Ew, TIP3P-FB, TIP4P-D; four Mg²⁺ models (see
below); hydrogen mass repartitioning;
positional restraints (mask, weight, reference); helix distance restraints with `LISTOUT=POUT`;
temperature ramps; Berendsen / Monte Carlo barostat; stages split into segments for wall-time limits;
a separate production script for the cluster; resuming (finished stages are skipped); a dry PDB and a
dry, imaged production trajectory. The **two-pass build** (dry tleap → ParmEd → ambpdb → `to_lrna.sh`
→ solvated tleap → ParmEd) is there for L-RNA: leap cannot reload a prmtop, and ambpdb writes D-RNA
residue names, so `to_lrna.sh` renames the given ranges before the second pass. The engine is pmemd,
pmemd.cuda or pmemd.MPI (with an MPI launcher such as `mpirun -np {n}` or `srun`); Amber 24 and 26 use
the same input, the version is recorded in the scripts. The *Environment* section adds
`conda activate` / `module load` / `source amber.sh` lines.

> tleap merges atom types with identical Lennard-Jones parameters: HO (2′-OH hydrogen) shares its LJ
> type with water HW (both have zero LJ terms) and O2 with base carbonyl O. ParmEd's
> `changeLJPair @%HO @%O2` / `@%OS` therefore also changes water-H…O2/OS and HO…O interactions. The
> wizard says so, and *Separate HO and O2 Lennard-Jones types first* (`addLJType @%HO`,
> `addLJType @%O2`) restricts the pairs to the intended atoms. It is off by default so the reference
> protocol is reproduced unchanged.

**Mg²⁺ models.** *Mg²⁺ Lennard-Jones model* in the build step:

| Model | What MDATK writes | GROMACS |
|---|---|---|
| 12-6 (default) | nothing extra: the Li & Merz 12-6 parameters loaded with the water model | yes |
| 12-6-4 (Li & Merz) | loads `frcmod.ions…_1264_<water>` and runs ParmEd `add12_6_4 :MG` | no (C4 terms) |
| microMg / nanoMg (Grotz, Cruz-León & Schwierz) | a frcmod with the optimised Mg²⁺ Lennard-Jones terms, and ParmEd `changeLJPair` lines with the authors' Mg²⁺–RNA pair terms for all 27 parmBSC0χOL3 atom types, plus their Mg²⁺–Cl⁻ pair | yes, as `[ nonbond_params ]` |

microMg and nanoMg are plain 12-6 models, so they work in GROMACS. microMg keeps the experimental
(µs) water exchange. nanoMg exchanges water on the nanosecond scale, which helps sample ion binding.
Their Mg²⁺–RNA pair terms (scaled combination rules, the same scaling for every RNA atom type)
reproduce binding to phosphate oxygens. Mg²⁺–water uses the Lorentz-Berthelot rule. The numbers come
from the authors' GROMACS parameter files and are cited in `mdatk/workflows/mg_params.py` with commit
links:

- TIP3P: JCTC 17, 2530 (2021), doi:10.1021/acs.jctc.0c01281
- SPC/E, TIP3P-FB, TIP4P-Ew, TIP4P-D: JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791
- OPC (the SPC/E values, shown to transfer): J. Chem. Phys. 156, 114501 (2022), doi:10.1063/5.0087292

MDATK does not offer two combinations:

- microMg with TIP4P-D: the authors' parameter file and paper disagree on the Mg²⁺–RNA σ (0.2598 nm
  vs 0.2197 nm for Mg–O2).
- OPC3: there are no published sets.

The authors publish no Amber files, so MDATK converts their values: Rmin/2 = 2^(1/6)·σ/2,
Rmin_ij = 2^(1/6)·σ_ij, 1 kcal = 4.184 kJ. An Amber build with these sets, converted to GROMACS, gives
the published σ/ε in `[ nonbond_params ]` (checked by the opt-in integration test).

- **Order.** The pair lines come after every `addLJType`, including NBfix19's `addLJType @%NB,NC` and
  *Separate HO and O2 types*, because `addLJType` resets the pair terms of the atoms it moves.
- **CP library.** If a CP residue library retypes the phosphate oxygens (O2 → OP, OS → OR), those types
  get the authors' O2 / OS terms. This is MDATK's choice: the authors fitted against plain OL3 oxygens.
- **Other types.** Atom types outside the authors' table, for example from an L-RNA library, keep the
  plain combination rule; the build step warns about this.
- **ParmEd output.** ParmEd's messages, including skipped masks, go into the step's log.
- **Cl⁻.** Amber's Cl⁻ is kept; only the Mg²⁺–Cl⁻ pair takes the authors' value, which was fitted with
  their Cl⁻ model. Without that pair (*Mg²⁺–Cl⁻ pair term* unticked), Mg²⁺–Cl⁻ follows the combination
  rule with the optimised Mg²⁺. That is not a published combination and gives far too strong Mg²⁺–Cl⁻
  contacts, so the build step warns about it.

> In the default *OL3 + CP*, `frcmod.99bsc0-chiol3-CaseP` is loaded after `leaprc.RNA.OL3`. If that
> frcmod defines new atom types (OP / OR, as the published file does) and no CP residue library is
> loaded, no atom uses them and the CP correction has no effect. The build step warns when it can read
> the frcmod and finds such types.

**GROMACS.** ParmEd's GROMACS writer (and acpype) keep only the diagonal Lennard-Jones terms, so the
NBfix19 / 0BPh pairs would be lost silently. MDATK's conversion script therefore gives every LJ type
its own name and adds the modified pairs from the Amber tables as `[ nonbond_params ]` (and, for 1-4
interactions, `[ pairtypes ]` scaled by fudgeLJ); the microMg / nanoMg pair terms are carried over the
same way. Converting such a topology with acpype is an error, and 12-6-4 topologies cannot be
converted. Checks cover grompp warnings that stop a run with
`-maxwarn 0` (Berendsen coupling, Parrinello-Rahman with new velocities), `-update gpu` restrictions
(no Nose-Hoover, no virtual-site waters such as OPC / TIP4P-Ew), `mass-repartition-factor`
(GROMACS 2024+ only), time steps that need hydrogen mass repartitioning, and more.

### Command line

```bash
mdatk workflow tools                       # step types; `mdatk workflow tools amber-md` lists options
mdatk workflow generate amber-md -o out --step 4 --project-name apt1 \
    -i topology=sys.parm7 -i coordinates=sys.rst7 -i restraints=helix.dist \
    --set engine=pmemd.MPI --set nproc=32 --set stages.5.nstlim=150000000
mdatk workflow template farfar2-amber -p plan.json   # add a template to a preset's plan
mdatk workflow check -p plan.json                    # list the plan's problems
mdatk workflow write -p plan.json [--step 3]         # write files, master script, summary
mdatk workflow restraints --top sys.parm7 --sequence "GGGCAC GUGCCC" \
    --dotbracket "(((((( ))))))" -o helix.dist       # DISANG file for an existing topology
```

### What was tested

The Amber chain was run end to end with AmberTools 26: tleap, ParmEd 4.3.1, ambpdb and cpptraj, with
sander standing in for pmemd. The restraints, segments, a 4 fs HMR step and the dry trajectory were
included. The GROMACS output was run through ParmEd conversion, grompp and mdrun with GROMACS 2023.3,
including microMg (TIP3P) and nanoMg (SPC/E) builds with NBfix19.
For Rosetta, SimRNA, RNAMake, VMD and x3dna-dssr, the flags were checked against each program's
source or documentation. The scripts were tested with stand-in executables. Review the generated
scripts before running them on your machines.

## Analyses

All analyses run over the selected frames in **one streaming pass** (MDAnalysis keeps a single
frame in memory). Neighbour searches use `scipy.spatial.cKDTree` (or MDAnalysis' cell-list grid
when periodic distances are enabled), so the cost grows roughly linearly with system size.

| Analysis | What is computed | Key parameters |
|---|---|---|
| **RMSD** | RMSD of each target selection to a reference frame, optionally after superposition on a separate fit selection; default targets are all heavy atoms (`rmsd/Complex`) and the C4′ backbone (`rmsd/Backbone`) | targets (label, selection, fit selection), reference frame, superposition, mass weighting |
| **Pairwise contacts** | heavy-atom pairs between group A and B within the cutoff; residue–residue contact counts per frame (sparse), per-residue contact heatmaps, fraction of native residue contacts *Q* | groups, cutoff (4.5 Å) |
| **H-bonds** | inter-group H-bonds: donor–acceptor distance ≤ cutoff and D–H···A angle ≥ cutoff. Donor hydrogens come from topology bonds (or are assigned by distance); N atoms bearing H or three heavy neighbours are not acceptors; without hydrogens a distance-only criterion is used | groups, distance (3.5 Å), angle (135°) |
| **O–Mg–O / Mg occupancy** | Mg–O inner-sphere (≤ 2.6 Å) and outer-sphere (≤ 4.5 Å) contacts with oxygens of the selected ranges; O–Mg–O bridges (pairs of oxygens on one ion), bridges between group A and B and the ions forming them; optional octahedral geometry check (O–Mg–O ≈ 90°/180° ± tolerance); per-residue occupancy class per frame (none / outer / inner sphere) | Mg selection, ligand selection/elements, cutoffs, bridge shell, geometry |
| **π–π stacking** | ring centroids, least-squares normals and planarity per base (purine: 9 ring atoms, pyrimidine: 6; aromatic amino acids too); a pair is stacked when both rings are planar, centroids are within the distance cutoff, normals within the angle cutoff and the lateral offset is small | distance (4.5 Å), angle (30°), offset (2.5 Å), planarity (0.25 Å RMS), scope (all pairs or A vs B) |
| **Intermolecular energy** | Coulomb + Lennard-Jones energy between group A and B within a cutoff (exact Amber LJ tables when the topology is parm7; approximate element parameters otherwise), plus the magnitude of the net force of B on A; a target value can be drawn on charts/panels | cutoff (12 Å), dielectric (optionally distance dependent, ε·r), bonded-pair exclusions, target |

Two internal analyses feed the structure panel: **trace** (one point per residue — the first
available atom of the *trace atoms* list such as `C4'` or `P`, or the heavy-atom centroid for
ligands — superposed onto the reference frame) and **proximity** (per residue minimum distance
to the proximity set, stored up to the *proximity search radius* so the render-time cutoff can be
changed without recomputation).

### Residue groups and selections

Groups are named residue ranges such as `1-40, 45, 50-60` (chain/segment qualified: `A:1-40`)
or raw [MDAnalysis selections](https://docs.mdanalysis.org/stable/documentation_pages/selections.html).
Wherever a selection is expected you can write `group:<Name>`, e.g.
`group:Aptamer and name P C4'`. Fields that ask for a group accept its bare name. Amber
topologies number residues 1…N in file order. Defaults are provided for the reference system
(`Aptamer` = 1–12, `Target` = 13–24) — edit them for yours.

## Structural view, colouring and glow

- 2D projection of the backbone trace of the selected groups (consecutive residues of the same
  chain are linked), Mg²⁺ ions and a time label, with depth cueing (size, width and fog).
- **Orientation**: drag the preview of the reference frame with the mouse, or use the X/Y/Z
  sliders. The angles (degrees and radians) are stored in the preset (`visual.rotation`,
  `R = Rz·Ry·Rx` about the reference-frame centroid) and applied to every frame. Frames are
  superposed onto the reference frame first (*superpose frames*), so the model does not drift.
- **Colouring modes**: `chain` (chainID, segment or bonded fragment), `groups` (group colours) and
  `proximity` (residues within the cutoff of the proximity set blend from the base colour to the
  proximity colour; set members keep their group colour).
- **Proximity glow** around residues near the proximity set:
  1. `ever` — glows for the entire movie if the residue ever meets the cutoff;
  2. `fade` — glows while in proximity, then fades over *fade frames* after separation;
  3. `current` — glows only while in proximity (no fade).
- **Residue-number tags** on every N-th residue (`visual.label_every`) and/or listed residues
  (`visual.label_residues`, e.g. `1, 12, 24, 40-45`), placed just outside the trace, in the video
  and in the GUI.
- The view is framed by the model's bounding sphere, so changing the orientation never zooms.

## 3D representations

Instead of the 2D trace, the structural view can draw residue groups as VMD-style
representations (*Visual → Representations (3D)*, `visual.representations`). Each row is a
style, the groups it draws (empty: every shown group), a colouring, a size factor, an opacity
and, for cartoons, a *sugars* switch. Representations intersect correctly by depth, so a cartoon
of the aptamer and licorice of a binding pocket can share the view.

| Style | What it draws |
|---|---|
| `cartoon` | New Cartoon for nucleic acids: an elliptical tube along a spline through P (or C4′, `visual.backbone_atom`), a rung from the backbone through C1′ to the glycosidic N (N9 / N1; C5 for pseudouridine), and each base as a closed slab (purines as the whole fused 9-atom outline). With *sugars*, the ribose ring is a lighter slab and the rung runs backbone → C4′, C1′ → N. |
| `ribbon` | a flat band through the backbone points, facing the bases |
| `tube` | a round tube through the backbone points |
| `licorice` | bonds as sticks with atom balls (r = 0.3 Å); a group next to a cartoon joins it at the backbone |
| `vdw` | van der Waals spheres (Bondi radii) |
| `surf` | a smooth molecular surface (sum of atomic Gaussians, like VMD QuickSurf), coloured by the nearest residue |
| `trace` | a ball at each backbone point and sticks to its neighbours (the 2D trace, but in 3D) |

- **Base positioning.** The cartoon deliberately differs from VMD where it hides base geometry:
  the rung bends at C1′, so the glycosidic angle (*syn*/*anti*) shows; the full purine outline
  shows inclination, propeller twist, buckle and stacking; the optional sugar slab shows the
  pucker; the backbone keeps the analysis colours (groups / proximity / glow) while bases are
  coloured by nucleotide (A blue, G green, C gold, U/T red). The spline is computed along whole
  strands, so a representation of part of a strand has the same shape as the full one.
- **Colouring.** `auto`: residue colours for backbones and carbons, nucleotide colours for cartoon
  bases, element colours for other atoms; or `residue`, `element`, `nucleotide` throughout.
- **Other residues.** Residues no representation draws are shown as a `trace` (default) or hidden
  (`visual.rep_rest: hide`). Mg²⁺ ions are spheres in the scene.
- **Look.** Lit surfaces, the same depth cue (fog toward the panel colour) as the 2D trace, ink
  outlines at depth steps (they separate stacked bases of the same colour) and anti-aliased edges.
  Glowing residues get a soft halo and a tint in the glow colour. Translucent representations show
  what is behind them.
- **Renderer.** `visual.renderer: auto` (default) uses the built-in renderer: a numpy z-buffer
  rasteriser that needs no OpenGL, so the same video comes out of a laptop, the GUI and a
  headless cluster node. It costs roughly 50–80 ms per 1080p frame for cartoon or licorice of a
  ~500-atom RNA, about 150 ms for surfaces, on top of the figure itself; while you drag the
  preview it renders at half resolution. `renderer: vtk` uses VTK (`pip install '.[vtk]'`) where it
  can render (a display, or EGL / OSMesa libraries on a headless machine, e.g. `libegl1
  libegl-mesa0`); this is checked once in a separate process, and where VTK cannot render, the
  built-in renderer is used with a note in the log. VTK keeps the 2D glow halos.

## Density maps

A cryo-EM or crystallographic map (MRC2014 / CCP4: `.mrc`, `.map`, `.ccp4`) can be drawn behind the
structural view as a projected **shadow**: the density above the contour level, summed along the
viewing direction, in one colour with adjustable opacity. The model stays where it is and the map is
placed on it by a rigid-body fit.

- **Fit.** MDATK simulates a map from the model (one Gaussian per heavy atom, weighted by atomic
  number, σ = 0.225 × resolution and a grid spacing of resolution / 3, as ChimeraX `molmap` does). It
  then maximises the correlation about the mean between that map and yours, using the grid points
  inside the model's envelope.
  - *Global search*: about 550 orientations at 30° (`density.angular_step`, 5–45°).
    - For each orientation, an FFT finds the best translations on a coarse grid. The score is a
      locally normalised correlation, so other molecules in the map (protein in an RNP, for example)
      do not attract the fit.
    - The 12 best distinct poses, spread over several sites, are refined against a map smoothed to
      8 Å.
    - The best poses and their 180° flips about the model's principal axes are then refined at the
      real resolution.
  - *Refine*: a local optimisation from the current placement.
  - Points of the model outside the map box are ignored. A pose needs at least half of the model
    inside the map to get its full score.
  - A search takes 10–60 s, depending on the map size and resolution. On synthetic maps with noise,
    with unrelated density next to the model, or with half the model outside the box, it recovers
    the true pose to within about a degree.
- **Frame.** The fit uses the reference frame (the first analysed frame, which the video is
  superposed on) or any trajectory frame number (`density.fit_frame`). That frame is superposed
  exactly like the video frames, so the placement applies to every frame.
- **Manual start.**
  1. Choose *Centre on model*, which puts the density centre on the model.
  2. Toggle *Move the map with the mouse* and drag the shadow in the structure panel; Shift + drag
     rotates it about its centre.
  3. Choose *Refine*.
- **Automatic fitting.** With *Fit automatically before rendering* (`density.auto_fit`), a render
  without a stored placement runs the global search first. The placement (`density.transform`: a 3×3
  rotation and a translation from model to map), its correlation and a summary are stored in the
  preset.
- **Appearance.**
  - Contour level `density.level`, in σ above the map mean; default 2. Lower it for faint maps.
  - Colour, opacity (0–1) and *show*.
  - The shadow sits under the molecule, glow and labels. With 3D representations, *in front*
    (`density.front`) draws it over them instead, since opaque cartoons hide much of it. In the
    video it is part of the static background (no per-frame cost). In the GUI it follows
    rotations, coarser while dragging.

```bash
mdatk density info map.mrc                               # grid, voxel size, origin, statistics
mdatk density fit -p run.json --map map.mrc --resolution 4.5 --save   # store the placement
mdatk density fit -p run.json --search manual --frame 250             # refine on another frame
mdatk density simulate -p run.json --resolution 6 -O model_6A.mrc [--placed]
```

## Panels, data keys and layouts

Every analysis output is addressed by a data key `<analysis>/<output>`:

| Kind | Keys |
|---|---|
| time series | `rmsd/<label>`, `contacts/atom_pairs`, `contacts/residue_pairs`, `contacts/q_native`, `hbonds/count`, `mg/contacts`, `mg/outer_contacts`, `mg/bridges`, `mg/inter_bridges`, `mg/bridging_ions`, `mg/inner_residues`, `stacking/count`, `stacking/inter`, `stacking/intra`, `energy/total`, `energy/elec`, `energy/vdw`, `energy/net_force`, `external/<label>.<column>` |
| per-residue matrices | `contacts/per_residue_a`, `contacts/per_residue_b`, `hbonds/per_residue_a`, `hbonds/per_residue_b`, `mg/occupancy`, `proximity/min_distance` |
| residue maps | `contacts/map`, `hbonds/map`, `stacking/map` |

**Residue groups per panel.** By default an analysis compares the groups set in its settings
(e.g. contacts between *Aptamer* and *Target*). A data key can name its own groups after `@`:
`contacts@Target+Mg/atom_pairs`, `hbonds@Aptamer+Target/count`, `energy@Aptamer+Target/total`,
or a single group for RMSD (`rmsd@Aptamer/Heavy`, `rmsd@Aptamer/Backbone`). Each group choice is
computed on demand in the same pass over the trajectory and cached on its own; in the GUI the
*Groups* drop-downs of a panel build these keys. Group names may contain letters, digits, `_`,
`.` and `-`.

Panel titles default to the label of the data shown. Count series (contacts, H-bonds, Mg
bridges, stacked pairs …) always get whole-number axis ticks, in panels and charts.

Panel types (`mdatk list` prints them):

| Type | Shows | Options (JSON) |
|---|---|---|
| `structure` | structural view described above | — |
| `timeseries` | one or more series vs time, moving cursor and value markers, optional target line | `progressive` (reveal over time), `smooth` (frames), `target`, `ylim` |
| `dual_axis` | two series on independent left/right axes | — |
| `heatmap` | residues × time (counts, values or occupancy classes) with cursor | `vmin`, `vmax`, `max_ticks` |
| `contact_map` | residue × residue map: `instant` (evolves with the movie), `frame` (fixed) or `frequency` (fraction of frames) | `mode`, `frame`, `max_ticks` |

Layout presets: `single`; `large_plus_2`, `large_plus_3`, `large_plus_4` (one large panel and 2–4
small ones on the right); `large_top_2`, `large_top_3` (small panels below); `equal_2` …
`equal_6` (2–6 equal panels). Panels fill the slots in order; the first slot is the large one.
The pipeline computes exactly the analyses that the chosen panels and charts need.

## Outputs

All outputs go to the output directory (default `MDATK_output/`):

| File | Content |
|---|---|
| `MDATK_video_<timestamp>.mp4` | H.264 video of the multi-panel figure (FPS, resolution, DPI, frame stride, codec and CRF configurable). Rendering is headless (Matplotlib Agg) and frames are piped straight into FFmpeg; static content is drawn once and only moving artists are redrawn per frame. |
| `MDATK_chart_<key>_<timestamp>.png/svg/pdf` | standalone time charts (all series or a chosen subset), optionally per-residue heatmaps; DPI, size, Matplotlib style (or MDATK's own chrome), colour, labels, title template, running-mean smoothing and target line are configurable |
| `MDATK_frame_<n>_<timestamp>.pdb` | selected frames by number (`0, 10, 20-50:10, last`) and/or time (`1.5ns, 2000ps`, nearest frame), optionally restricted to groups or a selection |
| `MDATK_params_log.json` | full parameters, software versions, input fingerprints, cache keys, the run's timestamp and the list of outputs |
| `MDATK_cache/` | analysis cache (see below) and cleaned trajectories |

All outputs of one run share a timestamp such as `2026-10-15_143205`, so runs never overwrite
each other; the cache and the parameter log keep fixed names. Set
`output.timestamp_files` to `false` for plain names.

## Caching and reproducibility

- Each analysis result is stored as a compressed `<analysis>-<key>.npz` with a JSON sidecar; the
  manifest `MDATK_cache.json` records the system, frame selection, time axis, residue groups,
  available data keys and the entries of the latest run.
- The key hashes everything that influences the numbers: input file fingerprints (path, size,
  modification time), frame selection, the analysis parameters and the definitions of the
  residue groups it uses. Changing a contact cutoff recomputes only contacts; changing colours,
  orientation, layout, glow or chart style recomputes nothing.
- `mdatk render --from-cache` (or *Re-render from cache* in the GUI) re-renders video and charts
  from the manifest without touching the trajectory.
- Cleaned trajectories live in `MDATK_cache/cleaned/<key>/` with the generated `cpptraj.in`,
  the full `cpptraj.log` and a completion marker.
- Any preset or `MDATK_params_log.json` can be passed back with `--params` to reproduce a run.

## Command line reference

```text
mdatk gui [PRESET]                 start the GUI
mdatk demo DIR [--n-frames N] [--no-solvent] [--run]
mdatk init-params OUT.json         write defaults (accepts the overrides below)
mdatk info      -p PRESET          system summary and residue-group check (JSON)
mdatk list                         analyses, panel types, layout presets
mdatk clean     -p PRESET [--force]
mdatk analyze   -p PRESET [--only rmsd,contacts] [--force]
mdatk render    -p PRESET [--no-video|--no-charts|--no-pdb] [--force] [--from-cache]
mdatk charts    -p PRESET [--series k1,k2] [--formats png,svg,pdf] [--dpi N] [--from-cache]
mdatk export-pdb -p PRESET [--frame-list "0, last"] [--times "1ns"] [--selection "Aptamer"]
mdatk preview   -p PRESET --frame N [--output still.png] [--from-cache]
mdatk density info MAP | fit -p PRESET [--map M] [--resolution R] [--search global|manual] [--save]
mdatk density simulate -p PRESET --resolution R -O OUT.mrc [--frame F] [--placed]
mdatk workflow tools|generate|template|check|write|restraints   (see Script wizard and Project Planner)
```

Common options: `--top`, `--traj …`, `--frames start:stop:step`, `--clean/--no-clean`,
`--cpptraj PATH`, `--conda-env NAME`, `-o/--out-dir`, `--cache-dir`, `-q/--quiet`, `-v/--verbose` and
`-s/--set KEY=VALUE` (repeatable; values are JSON when they parse as JSON), for example:

```bash
mdatk render -p run.json --set analyses.contacts.cutoff=5.0 \
    --set 'visual.rotation=[-90, 30, 0]' --set visual.glow_mode=current --no-pdb
```

Set `MDATK_DEBUG=1` to get full tracebacks.

## Parameter presets

A preset is a JSON file mirroring `mdatk.config.Project`:

```text
inputs     topology, trajectories, start/stop/step, dt_ps, time_unit, external[]
cleaning   enabled, cpptraj_path, solvent/ion/extra masks, autoimage(+anchor), fit, fit_mask,
           fit_reference (first|frame|file), reference_frame/file, remove_box_after_fit,
           start/stop/offset, extra_commands
groups[]   name, residues, selection, color
analyses   rmsd, contacts, hbonds, mg, stacking, energy, use_pbc, proximity_search_radius
visual     structure_groups, trace_atoms, align_frames, rotation, color_mode, colours,
           proximity_set/cutoff, glow_mode/color/fade_frames/strength, theme, sizes, title
layout     preset, panels[] {type, series[], title, options{}}
output     out_dir, cache_dir, video{...}, charts{...}, pdb{...}, write_params_log
workflow   project_name, stamp, use_stamp, out_dir, step_folders,
           nodes[] {id, tool, step, title, x, y, settings{}, files{}},
           edges[] {source, output, target, input}, scratch[] (stand-alone wizard settings)
```

Unknown keys are ignored with a warning and missing keys take their defaults, so presets stay
compatible across versions.

## Architecture and extending MDATK

```text
mdatk/
  config.py          parameter model, JSON presets, hashing
  selections.py      residue ranges, group:<Name> expansion, frame/time specs, elements
  spatial.py         cKDTree / cell-list neighbour search, Kabsch superposition
  cleaning/          cpptraj script generation, streaming execution, cached cleaned outputs;
                     conda environment detection / `conda run`
  io/                trajectory loading, parm7 parser/writer, cache store, external data, PDB export
  analysis/          registry + single-pass runner; rmsd, contacts, hbonds, mg, stacking, energy,
                     structure (trace, proximity)
  data.py            standardised results: series, matrices, sparse residue maps; Dataset
  render/            layouts, panels, colouring/glow, orientation, theme, video (FFmpeg), charts,
                     example data for the GUI preview; molecule/ holds the 3D representations
                     (topology roles, geometry per style, surface nets, scene colours, the
                     built-in z-buffer renderer and the optional VTK one)
  density.py, io/mrc.py   density maps: MRC / CCP4 I/O, simulated maps, rigid-body fitting
  pipeline.py        clean -> analyse (cached) -> outputs; shared by GUI and CLI
  workflows/         script generators (option schemas, checks, file generation) for FARFAR2,
                     SimRNA, RNAMake, Amber, GROMACS, VMD; plan.py resolves the Project Planner
                     graph and names files; scripts/ holds self-contained helper programs that
                     are embedded in generated workflows
  cli.py, gui/       command line and GTK 3 front ends (gui/canvas.py: WYSIWYG figure canvas,
                     gui/wizard.py: script wizard, gui/planner.py: Project Planner)
  demo.py            synthetic RNA system generator
```

**Adding an analysis**: subclass `mdatk.analysis.base.Analysis`, implement `enabled`,
`params_for`, `selection_texts`, `expected_outputs`, `setup`, `process(k, ts)` and `finalize`,
describe the outputs on the returned `AnalysisResult` (`add_series`, `add_matrix`,
`add_pairmap`) and decorate the class with `@register`. Set `group_slots` (and `OUTPUT_LABELS`)
to let panels choose residue groups for it. Its data keys immediately work in panels, charts,
the cache and the GUI.

**Adding a panel type**: subclass `mdatk.render.panels.Panel`, implement `setup` /
`draw_frame` (and `animated_artists` for fast blitting) and decorate it with `@register_panel`.

**Adding a step type (script generator)**: subclass `mdatk.workflows.base.Tool`, describe the options
as `Section`s of `Option`s (the wizard builds its forms from them), declare `inputs` / `outputs`
ports with file kinds, and implement `check` (return `Issue`s), `output_values` (the planned output
names, from `ctx.name(tag, ext)`) and `generate` (return `GeneratedFile`s); decorate it with
`@register`. It appears in the wizard, the planner and `mdatk workflow`.

## Testing

```bash
pip install '.[test]'
xvfb-run -a pytest                                # GUI tests need a display (Xvfb works)
MDATK_TEST_CPPTRAJ=/path/to/cpptraj pytest        # also run the real-cpptraj test
MDATK_TEST_AMBERHOME=$AMBERHOME pytest tests/test_workflow_integration.py
                                                  # run generated Amber scripts with tleap, ParmEd,
                                                  # ambpdb, sander, cpptraj (add MDATK_TEST_GROMACS=1
                                                  # with gmx on PATH for the GROMACS conversion + run)
```

The suite covers exact geometric criteria of every analysis on hand-built systems, cache keys
and reuse, cpptraj script generation (with a stand-in executable, and with real cpptraj when
available), the full pipeline including FFmpeg output, the CLI and GUI smoke tests. For the 3D
view it checks the rasteriser against analytic spheres and cylinders, closed meshes without
pinholes, depth order beyond the panel's range, registration and handedness in the final figure,
fog, glow and translucency, thread safety and a render without a display.

## Notes and limitations

- MDATK produces finished files; there is no interactive playback or frame scrubbing (the GUI
  preview shows example data, or one real frame, for arranging the layout).
- Interaction energies are non-bonded pair sums within a cutoff (no PME / long-range
  electrostatics, no solvent); use them for trends and comparisons against a target value.
- Without a parm7 topology, Lennard-Jones parameters are approximate element defaults and the
  result metadata records this (`lj_source`).
- When fitting, cpptraj rotates the unit cell, which NetCDF cannot store, so the box is removed
  after fitting by default (`cleaning.remove_box_after_fit`).
- Very large systems: restrict groups and the structure selection to the region of interest;
  only selected atoms are processed per frame and cached results are compact (sparse maps,
  float32 traces).
- The script wizard and planner only write files. Generated scripts call the programs by name
  (`tleap`, `parmed`, `pmemd.cuda`, `rna_denovo`, `gmx` …); use the *Environment* section of each
  step (conda environment, `module load`, `source amber.sh`) when they are not on PATH.
- Dual-axis plots are available because they are useful for juxtaposing e.g. contacts with
  another metric, but two y scales on one plot invite reading a correlation into an arbitrary
  alignment; prefer two stacked time-series panels when in doubt.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
