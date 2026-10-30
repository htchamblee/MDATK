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
ensemble. You review and run these scripts yourself, or ask MDATK to run them (*Run all…* or *Run
selected steps*), which logs
every run in the project file.

Since 0.0.3a a project is **one commented TOML file** (`*.mdatk.toml`) holding the settings, the
planner workflow, notes and a log of every step that ran; MDATK repairs and renames **PDB files**
(ChimeraX damage, L-RNA detected from the sugar chirality, Amber / CCD / Rosetta names, chain ids,
numbering, SEQRES); and every script, run log and the *About* dialog **cite** the programs, force
fields and parameter sets the project actually uses (BibTeX export).

Since 0.0.3b MDATK finds **base pairs** in every *n*-th frame (builtin canonical-pair detection or
the optional [Barnaba](https://github.com/srnas/barnaba)) and shows them in a **secondary-structure
arc panel**; a **sequence panel** spans the width of the video; both glow and colour with the
structure view. The structure view can be **zoomed and centred** on a residue or the origin, saved
in the project; and `mdatk pdb annotate` writes base pairs, stacking and dot-bracket into a PDB
file as REMARK 950 records.

In 0.0.5 the Amber build **checks the finished topology** and stops when a box meant to be neutral is not, a
Mg²⁺ does not carry +2, or (microMg / nanoMg) the ion lacks its published Lennard-Jones terms or a
solute atom type has no Mg²⁺ pair term. L-RNA and other non-standard atom types now get the pair
terms of their D-RNA twins (see *Mg²⁺ models*).

In 0.0.8 a new planner step runs **MDFF in NAMD on the Amber topology** (`namd-mdff`): NAMD reads the
parm7 as built, so CP, NBfix19, 0BPh, the L-RNA types and dihedrals, the Mg²⁺ pair terms and HMR are kept,
and a sander / NAMD energy comparison proves it before fitting. Its chirality and base-pair restraints
keep L-RNA sugars L, and it never uses VMD's D-only stereochemistry tools. Steps that existed in 0.0.7
write the same files as before (tested against the 0.0.7 output).

In 0.0.6 the Project Planner can **run a selection of steps**: tick the boxes, then *Run selected
steps*. In 0.0.7 independent steps can **run at the same time** (*Parallel*; each step waits for its
inputs), and a box shows an animated marker while its script runs.

Since 0.0.4 the builder also makes **parallel and Hoogsteen-paired duplexes and triplexes**, MDATK runs
as a **stand-alone AppImage or Windows program** without Python, and value fields no longer change
when the mouse wheel scrolls past them.

Since 0.0.3c MDATK **builds starting structures** (ideal A-form helices of either chirality, base
mutations with compensatory partners, rigid-body fits into a map), writes **restraint dictionaries**
for the L-RNA terminal residues Coot and Phenix do not know (acedrg / phenix.elbow from built-in L
SMILES, renamed to the model's atom names), and adds planner steps for **real-space refinement** with
Phenix 1.21 / 2.x and Coot 0.9 / 1.x (batch). microMg is now available with TIP4P-D as well.

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
- [PDB clean-up](#pdb-clean-up)
- [Structure builder](#structure-builder)
- [Restraint dictionaries, real-space refinement and MDFF](#restraint-dictionaries-real-space-refinement-and-mdff)
- [Citations](#citations)
- [Command line reference](#command-line-reference)
- [Project files](#project-files)
- [Architecture and extending MDATK](#architecture-and-extending-mdatk)
- [Testing](#testing)
- [Notes and limitations](#notes-and-limitations)

---

## Installation

### Stand-alone downloads (no Python needed)

Each [GitHub release](https://github.com/htchamblee/MDATK/releases) has ready-to-run builds
that bring their own Python, GTK 3, NumPy / SciPy / Matplotlib, MDAnalysis, Barnaba and FFmpeg.
No virtual environment, conda or system packages are needed for analysis and videos.

**Linux** (x86-64; Ubuntu 22.04, RHEL / Rocky / Alma 9 or newer): download
`MDATK-<version>-x86_64.AppImage`, then

```bash
chmod +x MDATK-<version>-x86_64.AppImage
./MDATK-<version>-x86_64.AppImage                          # the GUI
./MDATK-<version>-x86_64.AppImage project.mdatk.toml       # the GUI with a project
./MDATK-<version>-x86_64.AppImage --help                   # the command line (any mdatk command)
./MDATK-<version>-x86_64.AppImage demo ~/mdatk-demo        # e.g. the synthetic demo system
```

With no argument (or a project file) the AppImage starts the GUI; with a command it is the
`mdatk` command line. For a shorter name, link it: `ln -s ~/Apps/MDATK-<version>-x86_64.AppImage
~/.local/bin/mdatk` (a link named `mdatk-gui` always starts the GUI). If the AppImage does not
start because FUSE is missing (some containers and minimal installs), run it with
`--appimage-extract-and-run` or install `fuse`/`libfuse2`.

**Windows 10 / 11** (64-bit) - **beta, not yet tested on real machines**: run
`MDATK-<version>-windows-x64-setup.exe` (installs per user, no administrator rights needed;
Start-menu entry "MDATK", optional PATH entry for `mdatk.exe`) or unzip
`MDATK-<version>-windows-x64.zip` anywhere and start `mdatk-gui.exe` (`mdatk.exe` is the command
line). On Windows:

- the generated workflow scripts (Amber, GROMACS, Rosetta, Phenix, Coot ...) are bash scripts:
  MDATK writes them but does not run them - run them on a Linux machine, a cluster or in
  [WSL](https://learn.microsoft.com/windows/wsl/); the GUI's *Run* says so;
- cpptraj (AmberTools) does not run natively, so clean trajectories on Linux / WSL first, or
  analyse trajectories that are already cleaned.

**What you still install yourself** (both platforms):

- **cpptraj / AmberTools** - only for automatic solvent stripping and cleaning (see *Optional:
  cpptraj* below); the stand-alone builds detect it exactly like the source installation
  (`PATH`, `$AMBERHOME`, conda environments such as `AmberTools26`).
- The programs the generated workflow scripts call (Amber, GROMACS, Rosetta, Phenix, Coot, VMD
  ...), on the machine where you run those scripts.
- The optional VTK renderer is not included (the built-in 3D renderer is used).

FFmpeg is bundled: the default "FFmpeg executable" setting (`ffmpeg`) uses the bundled one;
set a path there to use another build. The licences of all bundled components are in the
`licenses` folder of each build (`licenses/THIRD_PARTY_NOTICES.txt`); the FFmpeg build is GPL v3
(`licenses/ffmpeg/FFMPEG-NOTICE.txt` says where its source is). Building the stand-alone
versions yourself: [packaging/README.md](packaging/README.md).

### Installing from source

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
pip install .            # installs MDAnalysis, NumPy, SciPy, Matplotlib (+ tomli on Python < 3.11)
pip install '.[test]'    # optional: adds pytest for running the test suite
pip install '.[vtk]'     # optional: VTK as an alternative 3D renderer (see 3D representations)
pip install '.[barnaba]' # optional: Barnaba for Leontis-Westhof base-pair classes and stacking
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
mdatk demo ./demo                        # synthetic solvated RNA aptamer/target system + project
mdatk gui ./demo/demo.mdatk.toml         # open it in the GUI and press "Render"
# or headless:
mdatk render -p ./demo/demo.mdatk.toml   # add --clean to strip solvent with cpptraj first
```

The demo writes a genuine Amber `demo.parm7` + `demo.nc` pair (a 12-bp aptamer/target duplex,
three Mg²⁺ ions, Na⁺ and ~4 000 waters, 120 frames) in which one end of the duplex frays and
re-anneals, a Mg²⁺ bridge between the strands breaks and reforms, and another ion visits a
phosphate. It exists to exercise every feature without real data; its geometry is schematic.

For your own system start from `examples/aptamer_target.mdatk.toml` (relative paths in a
project file are resolved against its folder) or from the GUI. The GUI opens empty: nothing is
loaded at start-up; *Open project* has a drop-down with the recent projects, the demo project and
*Browse…*.

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
5. **Reproduce**: save/open project files, re-render from the cache, or replay a run from its
   `MDATK_params_log.toml`.

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

The settings live in a permanent side bar on the left (fixed 520 px wide, never scrolled
sideways), opened at the Input page; the icon bar above it switches pages (hover for the name):

| Page | Contents |
|---|---|
| **Input** | project notes, topology/trajectory/external-data choosers, frame range, time-step override, display time unit, cpptraj cleaning options (conda environment, masks, imaging, fitting), *Run cleaning now*, *Inspect system* |
| **Groups** | table of named groups (residue ranges or selections) with colour swatches and a colour picker |
| **Analyses** | enable switches and parameters for RMSD, contacts, H-bonds, O–Mg–O / Mg occupancy, π–π stacking and intermolecular energy |
| **Structure** | orientation sliders, residue-number tags, colouring, proximity and glow, appearance |
| **Layout** | grid preset and, per slot, panel type, data, residue groups, extra keys, title (follows the data until you type your own) and options; *Render still* renders one frame from real data |
| **Outputs** | output/cache folders, video, charts, PDB export, timestamps, *Re-render from cache* |

The bottom bar holds **Render**, **Cancel**, the progress bar, **Open output folder** (shown in
your file manager) and **Open video** (the most recent video); the log of the pipeline and of
cpptraj is in the expander above it. Projects are opened/saved from the header bar (Ctrl+O /
Ctrl+S, Ctrl+R renders); the main menu has *Project log…* (the steps that ran), *Export citations
(BibTeX)…* and *About and citations*. File choosers name the extensions they accept, e.g.
"Trajectory files (.nc, .ncdf, .mdcrd, .crd, .trj, .dcd, .trr, .gro, .xtc, .pdb, .xyz, .binpos)".
Long jobs run in a background thread and can be cancelled.

## Script wizard and Project Planner

The **Workflow** menu in the header bar opens two windows (they also work from the command
line, see below) and the PDB clean-up window. They write files and scripts; nothing is executed
unless you ask for it with *Run all…* / *Run selected steps* (planner), *Run step…* (a planner box's
details) or *Write and run…* (wizard). MDATK then asks for
confirmation and runs each step's main script with `bash`. Each step starts once the steps it
reads from (through the planner's connections) have finished. Otherwise steps go in plan order,
up to *Parallel* scripts at a time (planner toolbar, saved with the project; default 1).

- **Output and logging:** the output streams into the log, each line marked with its step number
  when scripts run side by side. Every run is recorded in the project file.
- **Failures:** a failed script stops every step that depends on it; independent steps carry on.
- **Cancel** stops every running script.
- **Running markers:** a box whose script is running shows a turning arc in its title bar, and a
  waiting box a dashed ring. When a script ends, its box shows a tick (done), a cross (failed) or a
  dash (cancelled / not started) until the next run.
- **GPU and cores:** GPU steps that run at the same time share the GPU unless each step's
  *Environment* section sets `CUDA_VISIBLE_DEVICES`; CPU steps share the cores.

Scripts marked *run separately* (production runs meant for a cluster queue) are never started.

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
  VMD; Amber protocol; Amber → GROMACS; ideal helix → Amber build → restraints → MD; model + map →
  restraint dictionaries → Phenix RSR → Coot RSR). *Write all files* writes every step, a master script that
  runs the steps in order, and a Markdown summary of the plan. Each step box has a check box in its
  title bar (also Space on the selected box, or the right-click menu; *Tick all steps* / *Untick all
  steps* on the canvas): *Run selected steps (N)* writes the ticked steps' files and runs their main
  scripts in plan order, whatever order you ticked them in. Steps they read from are not written or
  run again, so their outputs must exist already. The ticks are not saved with the project.
  *Use for analysis* (on an MDATK
  analysis box) puts the planned topology / trajectory into Input & cleaning. The plan is saved in
  the project file (`workflow` section), so one file holds the plan and the analysis settings.
  `examples/rna_plan.mdatk.toml` is a complete example plan (open it, set the output folder, and
  open the planner).

### File names

Every file is named `<project>_<step>_<tool>_<time>.<ext>`, e.g. `apt1_03_tleap_20261015-1432.parm7`:
the project name, the step number of the box, the program or stage that writes the file, and one
time stamp per plan. The stamp is fixed the first time files are written (and stored in the project),
so later re-generations keep the same names; *New stamp* starts a new set, and the stamp can be
switched off. Each step writes into its own folder (`03_amber-build/`), and scripts refer to the files
of other steps by relative paths (`../03_amber-build/apt1_03_parmed_….parm7`), so a whole plan folder
can be copied to a cluster. Every script starts with `cd "$(dirname "$0")"`, so it can be started from
anywhere. Each script's header cites what that step uses (see [Citations](#citations)).

### Step types

| Step | Writes | Notes |
|---|---|---|
| **Rosetta · FARFAR2** | FASTA + secstruct from a sequence and dot-bracket, a flags file, a script running *N* seeds in parallel (`xargs -P`), one silent file per run | fixed helices (`-s`, `-input_res` typed or all helix residues), cryo-EM map (`-edensity:*`), score weights, constant seeds; checks that paired bases are complementary (rna_denovo stops otherwise) |
| **Rosetta · FARFAR2 models** | script: `rna_cluster` → `extract_pdbs` → `x3dna-dssr` | ranked models `…_model-01_….pdb` (best first) feed the Amber build |
| **SimRNA** | `config.dat`, sequence, secondary-structure (pseudoknots on extra lines) and restraint files, parallel replica-exchange runs, clustering + all-atom rebuild (`clustering`, `SimRNA_trafl2pdbs … AA`) | links SimRNA's `data/` folder into every run folder |
| **RNAMake** | script for `design_rna_scaffold` (current, `--options`) or the legacy `design_rna` (`-options`): motif-graph search, sequence optimisation, thermal-fluctuation scoring | needs `RNAMAKE`; current RNAMake has no separate builder / optimiser programs |
| **Amber · system build** | tleap inputs (force field, water, box, ions, extra frcmod / libraries / leap commands), ParmEd inputs, helpers (PDB preparation, L-RNA naming, CP library, TIP4P/2005 files) and the build script | see below |
| **Amber · helix restraints** | DISANG file of the hydrogen bonds of every base pair (G–C ×3, A–U ×2, optional G·U), found by residue number and atom name in the final topology | written directly when the topology exists, otherwise by a small Python script (no ParmEd needed) the step runs |
| **Amber · MD stages** | one mdin per stage and the run scripts | see below |
| **GROMACS · MD** | ParmEd (or acpype) conversion, RNA position restraints, mdp files, grompp / mdrun scripts | 2024+ (workstation) or 2023.3 (cluster module `GROMACS/2023.3-CUDA-11.8.0`) |
| **VMD** | Tcl script + start script | one representation per residue group in the MDATK colours, bases as licorice, Mg²⁺ spheres, frame alignment, smoothing, snapshot or movie (Tachyon + ffmpeg) |
| **Structure builder (MDATK)** | a script calling `python3 -m mdatk build`: ideal A-form helix (D or L), base mutations with compensatory partners, or a rigid-body fit into a map | see Structure builder |
| **Restraint dictionaries · acedrg / eLBOW** | per residue code a SMILES file, the acedrg or phenix.elbow call and MDATK's renaming to the model's atom names | L-RNA 5'/3' variants (0A5, 0G3 …) for Coot and Phenix; see Restraint dictionaries |
| **Phenix · real-space refinement** | `phenix.real_space_refine` script (1.21 / 2.x), optional base-pair `.eff` from a dot-bracket or the model's REMARK 950 annotation | see Real-space refinement |
| **Coot · real-space refinement (batch)** | headless Coot 0.9 / 1.x script refining each chain in overlapping windows | see Real-space refinement |
| **MDATK analysis**, **Existing file**, **Note** | nothing | link a plan to this project's analysis, feed an existing file into steps, annotate the canvas |

**Amber.** The defaults reproduce a working RNA protocol: OL3 + CP (phosphate oxygens retyped, see
below), the NBfix19 and 0BPh Lennard-Jones pair modifications applied with ParmEd (only offered with
OL3 + CP, whose atom types their masks assume), SPC/E water in a 15 Å truncated octahedron, neutralising Na⁺,
optional Mg²⁺ balanced with Cl⁻, and the stages min1 (solute restrained) → min2 → heat (Langevin from
0 K) → NPT equilibration → NPT equilibration without positional restraints → production; the mdin files
match the hand-written reference files line for line. Other choices: RNA force fields OL3, LJbb,
ROC-RNA, YIL, DESRES; waters TIP3P, OPC, OPC3, TIP4P-Ew, TIP3P-FB, TIP4P-D, TIP4P/2005; four Mg²⁺
models (see below); hydrogen mass repartitioning;
positional restraints (mask, weight, reference); helix distance restraints with `LISTOUT=POUT`;
temperature ramps; Berendsen / Monte Carlo barostat; stages split into segments for wall-time limits;
a separate production script for the cluster; resuming (finished stages are skipped); a dry PDB and a
dry, imaged production trajectory. The engine is pmemd,
pmemd.cuda or pmemd.MPI (with an MPI launcher such as `mpirun -np {n}` or `srun`); Amber 24 and 26 use
the same input, the version is recorded in the scripts. The *Environment* section adds
`conda activate` / `module load` / `source amber.sh` lines.

**L-RNA.** *Name L-nucleotides for leap* (on by default) runs MDATK's PDB tools, copied into the step
folder, on the starting structure before tleap: residues whose sugar chirality (C1′, C3′, C4′) is L
get the Dudek & Trylska names (LA LC LG LU, LA5/LA3 … at chain ends), the others Amber names; *Force L
for* (e.g. `C:45-73`) overrides the detection, and *5′/3′ terminal names* can be switched off for
libraries that map chain ends with `addPdbResMap` (plain LG, as `to_lrna.sh` wrote). This replaces
`to_lrna.sh`. The L-RNA library and parameters themselves go under *Force field* (Dudek & Trylska's
`nucleic12_LRNA.lib` + `parm10_LRNA.dat`, or your own `LRNA.lib` + `frcmod.LRNA`); when the input PDB
exists the build step checks it and reports L residues without a library as an error. The optional
**two-pass build** (L-RNA naming → dry tleap → ParmEd → ambpdb, keeping the dry topology's residue
names → solvated tleap → ParmEd) is kept for set-ups that need the dry intermediate files.

**OL3 + CP library.** The CP correction of Steinbrecher, Latzer & Case (JCTC 2012) gives the phosphate
oxygens their own Lennard-Jones types: non-bridging OP (Rmin/2 1.7493 Å, ε 0.21 kcal/mol) and
bridging OR (1.7718 Å, 0.17 kcal/mol). AmberTools' `RNA.lib` types them O2 / OS like other oxygens
(O4′ is also OS), so by default (*CP parameters: Retyped library*) the build script writes
`RNA_CP.lib` (OP1/OP2 → OP; bridging O5′/O3′ → OR; terminal hydroxyls and O4′ unchanged) and
`frcmod.cp` (the CP values plus every parm10.dat bonded term of O2 / OS repeated for OP / OR) from
the AmberTools installation, and loads them after `leaprc.RNA.OL3`. The NBfix19 and 0BPh masks then
address the phosphate oxygens as `@%OP` / `@%OR`, as their authors describe (0BPh: "H5−OR/OS and
H4−OR/OS atom types", OR in OL3CP). *Own frcmod file* loads a CP parameter file you provide instead
(masks stay `@%O2` / `@%OS`). Only standard (D-) residues are retyped; L-RNA libraries keep their own
types (Dudek & Trylska type O5′ and O4′ as LD; other libraries give O4′ its own type, e.g. LE). With
*NBfix19 / 0BPh / CP for L-RNA* (on by default, also when the input PDB already has L names) the build
declares the L-RNA library's types for tleap
(`addAtomTypes { { "LD" "O" "sp3" } { "LH" "O" "sp3" } { "LE" "O" "sp3" } }`)
and adds the same pairs for its oxygen type: `changeLJPair @%HO @%LD 3.03 0.01` (NBfix19),
`@%H5 @%LD 2.8808 0.0505` and `@%H4 @%LD 2.9308 0.0505` (0BPh). ParmEd skips them when the structure has
no L residues. The CP correction of L residues comes with the L-RNA parameter file (e.g. `frcmod.LRNA`);
the types are editable (*L-RNA atom types*) for libraries that name them differently.

**TIP4P/2005.** AmberTools ships no TIP4P/2005 files. The build script derives `frcmod.tip4p2005` and
`tip4p2005box.off` from AmberTools' TIP4P-Ew files with the Abascal & Vega (2005) values (ε/k 93.2 K,
σ 3.1589 Å, q_H 0.5564 e, d_OM 0.1546 Å; Rmin/2 1.772873 Å, ε 0.185207 kcal/mol), keeps residue
WAT and the atom names (so `:WAT` masks and SETTLE work) and maps crystal waters to it. A 200 ps NPT
run of 608 waters (sander, 298 K, 1 bar, 9 Å cutoff, PME) gave 0.9964 ± 0.002 g/cm³ against the
paper's 0.9979; sander keeps the extra point at 0.1546 Å. There are no Na⁺/K⁺ or 12-6-4 parameters
for TIP4P/2005: neutralising ions come from the TIP4P-Ew (Joung–Cheatham) set, with a warning, and
12-6-4 is not offered. microMg / nanoMg include the authors' TIP4P/2005 Cl⁻ (Grotz & Schwierz 2022,
Table 2).

> tleap merges atom types with identical Lennard-Jones parameters: HO (2′-OH hydrogen) shares its LJ
> type with water HW (both have zero LJ terms) and O2 with base carbonyl O. ParmEd's
> `changeLJPair @%HO @%O2` / `@%OS` therefore also changes water-H…O2/OS and HO…O interactions. The
> wizard warns, and *Separate HO and O2 Lennard-Jones types first* (`addLJType @%HO`,
> `addLJType @%O2`; with the CP library only `@%HO`, since the phosphate oxygens are already OP)
> restricts the pairs to the intended atoms. It is on by default for new projects since 0.0.5.1;
> untick it only to reproduce a protocol that did not separate them (projects saved earlier keep their
> setting). In a real TIP3P build the HW Lennard-Jones type held 18,716 atoms:
> 18,700 water hydrogens and 16 hydroxyl hydrogens. The topology check's log lists every
> Lennard-Jones type shared by solute and water atoms.

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
- SPC/E, TIP3P-FB, TIP4P/2005, TIP4P-Ew, TIP4P-D: JCTC 18, 526 (2022), doi:10.1021/acs.jctc.1c00791.
  For TIP4P/2005, Table 1 of the paper repeats the microMg Mg–OP pair (0.2217 nm / 6.6266 kJ/mol) in
  the nanoMg row; the nanoMg λ factors of that row and the authors' parameter file both give
  0.2197 nm / 7.6257 kJ/mol, which MDATK uses.
- OPC (the SPC/E values, shown to transfer): J. Chem. Phys. 156, 114501 (2022), doi:10.1063/5.0087292

Where a paper's table and the parameter file differ, the file is right: N. Schwierz confirmed that
the files on GitHub match the published scaling factors and the tables carry transcription errors.
For TIP4P-D microMg, σ(Mg–OP) is 0.2598 nm (λσ = 1.3255; Table 1 prints 0.2197 nm, and nanoMg is
affected likewise); for TIP3P nanoMg σ(Mg–OP) is 0.2279 nm (not 0.2277); for OPC microMg σ(Mg–Cl)
is 0.4318 nm (not 0.4316). MDATK uses the file values throughout, so every water model above has
both microMg and nanoMg. OPC3 has no published set.

The authors publish no Amber files, so MDATK converts their values: Rmin/2 = 2^(1/6)·σ/2,
Rmin_ij = 2^(1/6)·σ_ij, 1 kcal = 4.184 kJ. An Amber build with these sets, converted to GROMACS, gives
the published σ/ε in `[ nonbond_params ]` (checked by the opt-in integration test).

- **Order.** The pair lines come after every `addLJType`, including NBfix19's `addLJType @%NB,NC` and
  *Separate HO and O2 types*, because `addLJType` resets the pair terms of the atoms it moves.
- **CP library.** With the retyped CP library (the default) the phosphate oxygens are OP / OR; they
  get the authors' O2 / OS terms, and the build step says so. This is MDATK's choice: the authors fitted
  against plain OL3 oxygens.
- **Ion atom type.** The frcmod defines the ion's terms for both type names in use: `Mg2+` (the
  type of the `MG` unit in AmberTools' `atomic_ions.lib`) and `MG` (the older type of `parm10.dat`
  and `leaprc.RNA.OL3`). On AmberTools 26 the `MG` unit carries `Mg2+` with charge +2, so the
  published terms applied in earlier MDATK versions too (checked on a built nanoMg / TIP3P topology).
  The second name only guards against a release that types the unit differently, and the topology
  check (below) shows the ion's type, charge and terms in every build. tleap's message
  `MG has a charge of 0. Adding the ions anyway.` during `addIonsRand mol MG` is misleading: the
  ions carry +2 and MDATK's 2 Cl⁻ per Mg²⁺ leave the box neutral.
- **Other types.** Atom types outside the authors' table get no pair term from the table, and the
  combination rule with these ions binds Mg²⁺ several times too strongly. Before 0.0.5 that was
  the case for the oxygens of an L-RNA library. Now each L-RNA atom type gets the term of its D-RNA
  twin (*L-RNA atom types*: LD and LE as OS, LH as OH; `TYPE:REF` for others, e.g. `LX:OS`), and
  *Extra Mg²⁺ pair types* does the same for any library (`XO:OS XH:OH`). References: the 27
  parmBSC0χOL3 types and the CP types OP / OR. Choose the reference with the same Lennard-Jones
  terms as the type (its LJ twin): tleap merges types with equal LJ terms, and a pair set for one
  applies to all of them (the topology check reports conflicting maps). The planner reads the extra
  residue libraries and warns about atom types without a reference.
- **Topology check.** *Check the final topology* (on by default) runs a small standard-library
  Python script on the final parm7. It stops the build script when:
  - the box is not neutral, when it should be (neutralised, and Mg²⁺ balanced with Cl⁻ or none added);
  - fewer Mg²⁺ than requested are present;
  - a Mg²⁺ does not carry +2;
  - with microMg / nanoMg, the ion's own Lennard-Jones terms are not the published ones;
  - with microMg / nanoMg, a solute atom type with Lennard-Jones terms has no explicit Mg²⁺ pair term.

  The report is written to `…_check_….log`. It reads the stored coefficients, so it also catches a
  misspelt mask or a later `addLJType` that reset a pair.
- **ParmEd output.** ParmEd's messages, including skipped masks, go into the step's log.
- **Cl⁻.** Amber's Cl⁻ is kept (except with TIP4P/2005, where Amber has none and the authors' Cl⁻ is
  used); only the Mg²⁺–Cl⁻ pair takes the authors' value, which was fitted with
  their Cl⁻ model. Without that pair (*Mg²⁺–Cl⁻ pair term* unticked), Mg²⁺–Cl⁻ follows the combination
  rule with the optimised Mg²⁺. That is not a published combination and gives far too strong Mg²⁺–Cl⁻
  contacts, so the build step warns about it.

> With *CP parameters: Own frcmod file*, the file is loaded after `leaprc.RNA.OL3`. If it defines new
> atom types (OP / OR) and no CP residue library is loaded, no atom uses them and the CP correction
> has no effect. The build step warns when it can read the frcmod and finds such types.

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
mdatk workflow template farfar2-amber -p plan.mdatk.toml   # add a template to a project's plan
mdatk workflow check -p plan.mdatk.toml                    # list the plan's problems
mdatk workflow write -p plan.mdatk.toml [--step 3] [--run] # write files, master script, summary
mdatk workflow run -p plan.mdatk.toml [--step 3] [--parallel 2]  # run the written main scripts (logged)
mdatk workflow restraints --top sys.parm7 --sequence "GGGCAC GUGCCC" \
    --dotbracket "(((((( ))))))" -o helix.dist       # DISANG file for an existing topology
```

### What was tested

The Amber chain was run end to end with AmberTools 26: tleap, ParmEd 4.3.1, ambpdb and cpptraj, with
sander standing in for pmemd. The restraints, segments, a 4 fs HMR step and the dry trajectory were
included. The GROMACS output was run through ParmEd conversion, grompp and mdrun with GROMACS 2023.3,
including microMg (TIP3P) and nanoMg (SPC/E) builds with NBfix19. In 0.0.3a the default build (CP
library, NBfix19, 0BPh, HMR) and a CP + TIP4P/2005 + nanoMg build were run with tleap and ParmEd, and
the default build was converted to GROMACS and run with grompp / mdrun.
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

A third internal analysis, **basepairs**, feeds the secondary-structure panel (it runs only when a
`secondary` panel needs it). Every *stride*-th frame (10) it finds the base pairs; the frames in
between reuse the last result. Two engines are available (*Analyses → Base pairs*):

- **builtin** (default, no extra dependency): canonical pairs from donor–acceptor distances
  ≤ *cutoff* (3.5 Å). G–C needs 2 of its 3 Watson–Crick H-bonds, A–U / A–T both; G–U wobble
  pairs (both H-bonds) are optional. The C1′ atoms must be 8.5–12.5 Å apart (about 10.5 Å in a
  real pair), which keeps stacked neighbours such as a 5′-GpC-3′ step out. Each base pairs at most
  once (the pair with the shortest summed distance wins; near-ties go to the C1′–C1′ distance closer
  to 10.5 Å). It works for L-RNA too, because it uses
  distances only.
- **barnaba** (optional, `pip install barnaba`): Leontis–Westhof classes of all pairs, including
  non-canonical ones (*Non-canonical pairs* draws them too). Barnaba's classification assumes
  D-nucleotide geometry; for L-RNA the edges and cis/trans may be mirrored, so the builtin engine
  is the safer choice there.

The result is a sparse residue × residue map per frame (1 = Watson–Crick, 2 = wobble, 3 = other),
cached like the other analyses.

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
  sliders. The angles (degrees and radians) are stored in the project (`visual.rotation`,
  `R = Rz·Ry·Rx` about the reference-frame centroid) and applied to every frame. Frames are
  superposed onto the reference frame first (*superpose frames*), so the model does not drift.
- **Zoom and centre** (`visual.zoom`, `visual.center_on`, `visual.center_follow`): the mouse wheel
  over the structure panel (or *Zoom* on the Structure page) magnifies the view; *Centre on* puts a
  residue (`12`, `B:12`) or the coordinate origin (`origin`) in the middle instead of the model's
  centre, and *Follow the centre residue* keeps it there in every frame. All three are saved in the
  project and used in the video, e.g. to stay on one interface.
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
| `licorice` | bonds as sticks with atom balls (r = 0.3 Å); bonds leaving the group run to the neighbouring atom, so it reaches a neighbouring cartoon |
| `vdw` | van der Waals spheres (Bondi radii) |
| `surf` | a smooth molecular surface (sum of atomic Gaussians, like VMD QuickSurf), coloured by the nearest residue |
| `trace` | a ball at each backbone point and sticks to its neighbours (the 2D trace, but in 3D) |

- **Base positioning.** The cartoon deliberately differs from VMD where it hides base geometry:
  the rung bends at C1′, so the glycosidic angle (*syn*/*anti*) shows; the full purine outline
  shows inclination, propeller twist, buckle and stacking; the optional sugar slab shows the
  pucker; the backbone keeps the analysis colours (groups / proximity / glow) while bases are
  coloured by nucleotide (A blue, G green, C gold, U/T red; Amber, CHARMM and other names, or the
  base's own atoms for unknown names). The spline is computed along whole strands, so a
  representation of part of a strand has the same shape as the full one.
- **Colouring.** `auto`: residue colours for backbones, carbons and surfaces, nucleotide colours
  for cartoon bases, element colours for other atoms; or `residue`, `element` (surfaces by the
  nearest atom), `nucleotide` throughout.
- **Other residues.** Residues no representation draws (including those a style cannot draw, such
  as a small-molecule target under a cartoon) are shown as a `trace`, ligands without a backbone
  atom as `licorice` (default), or hidden (`visual.rep_rest: hide`). Mg²⁺ ions are spheres in the
  scene.
- **Look.** Lit surfaces, the same depth cue (fog toward the panel colour) as the 2D trace, ink
  outlines at depth steps (they separate stacked bases of the same colour) and anti-aliased edges.
  Glowing residues get a soft halo and a tint in the glow colour. Translucent representations show
  what is behind them (the front-most translucent surface over the opaque ones).
- **Renderer.** `visual.renderer: auto` (default) uses the built-in renderer: a numpy z-buffer
  rasteriser that needs no OpenGL, so the same video comes out of a laptop, the GUI and a
  headless cluster node. It costs roughly 50–80 ms per 1080p frame for cartoon or licorice of a
  ~500-atom RNA, about 150 ms for surfaces, on top of the figure itself; while you drag the
  preview it renders at half resolution. `renderer: vtk` uses VTK (`pip install '.[vtk]'`) where it
  can render (a display, or EGL / OSMesa libraries on a headless machine, e.g. `libegl1
  libegl-mesa0`); this is checked once in a separate process, and where VTK cannot render, the
  built-in renderer is used with a note in the log. With VTK the glow is drawn the same way;
  translucent representations use VTK's own blending (all layers show).

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
  project file.
- **Appearance.**
  - Contour level `density.level`, in σ above the map mean; default 2. Lower it for faint maps.
  - Colour, opacity (0–1) and *show*.
  - The shadow sits under the molecule, glow and labels. With 3D representations, *in front*
    (`density.front`) draws it over them instead, since opaque cartoons hide much of it. In the
    video it is part of the static background (no per-frame cost). In the GUI it follows
    rotations, coarser while dragging.

```bash
mdatk density info map.mrc                               # grid, voxel size, origin, statistics
mdatk density fit -p run.mdatk.toml --map map.mrc --resolution 4.5 --save   # store the placement
mdatk density fit -p run.mdatk.toml --search manual --frame 250             # refine on another frame
mdatk density simulate -p run.mdatk.toml --resolution 6 -O model_6A.mrc [--placed]
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
| `contact_map` | residue × residue map: `instant` (evolves with the movie), `frame` (fixed) or `frequency` (fraction of frames); `transpose` puts the first group along x | `mode`, `frame`, `max_ticks`, `transpose` |
| `sequence` | one-letter sequence of the structure-view residues (chains separated, chain letters shown) in the structure panel's colours, proximity glow and number tags; wraps into rows, and more than two rows log a warning to choose a smaller range | `residues` (e.g. `A:1-20, B:5-30`), `per_row` |
| `secondary` | arc diagram on a fixed layout: residues along a line (same colours, glow and tags), base pairs within a chain as arcs above it, pairs between chains as transparent yellow arcs below it | `source`: `dynamic` (pairs re-detected every *n* frames by the base-pair analysis), `first` (first frame) or `dotbracket` (with `dotbracket`, `&` or a space between chains); `residues` |

Layout presets: `single`; `large_plus_2`, `large_plus_3`, `large_plus_4` (one large panel and 2–4
small ones on the right); `large_top_2`, `large_top_3` (small panels below); `equal_2` …
`equal_6` (2–6 equal panels); `single_strip`, `large_plus_2_strip`, `large_plus_3_strip`,
`large_top_3_strip`, `equal_4_strip` add a thin full-width strip at the bottom (their last slot) for
a sequence or secondary-structure panel. Panels fill the slots in order; the first slot is the
large one.
The pipeline computes exactly the analyses that the chosen panels and charts need.

## Outputs

All outputs go to the output directory (default `MDATK_output/`):

| File | Content |
|---|---|
| `MDATK_video_<timestamp>.mp4` | H.264 video of the multi-panel figure (FPS, resolution, DPI, frame stride, codec and CRF configurable). Rendering is headless (Matplotlib Agg) and frames are piped straight into FFmpeg; static content is drawn once and only moving artists are redrawn per frame. |
| `MDATK_chart_<key>_<timestamp>.png/svg/pdf` | standalone time charts (all series or a chosen subset), optionally per-residue heatmaps; DPI, size, Matplotlib style (or MDATK's own chrome), colour, labels, title template, running-mean smoothing and target line are configurable |
| `MDATK_frame_<n>_<timestamp>.pdb` | selected frames by number (`0, 10, 20-50:10, last`) and/or time (`1.5ns, 2000ps`, nearest frame), optionally restricted to groups or a selection |
| `MDATK_params_log.toml` | full parameters, software versions, input fingerprints, cache keys, the run's timestamp, the list of outputs and the project's citations; open it as a project to repeat the run |
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
- Any project file or `MDATK_params_log.toml` can be passed back with `-p` to reproduce a run.

## PDB clean-up

*Workflow → PDB clean-up…* in the GUI and `mdatk pdb` on the command line repair and rewrite PDB
files with fixed-column output that keeps coordinates, occupancies and B-factors verbatim:

- **ChimeraX damage** such as a residue name fused with the chain id (`GC:` = residue G of chain C)
  is repaired automatically; only the first MODEL is kept.
- **L-nucleotides** are detected from the signed chiral volumes at C1′, C3′ and C4′ (all usable
  centres must agree; flat or contradicting centres leave the residue to its name) and ranges such as
  `C:45-73` force L. A warning lists L-named residues with D geometry and vice versa.
- **Residue names**: `amber` (D: A C G U, A5/A3 …; L: LA LC LG LU, LA5/LA3 … as in Dudek & Trylska's
  library; a single-residue L segment gets the 5′ unit), `ccd` for Coot / Phenix (L: 0A 0C 0G 0U and
  0A5/0A3 … - these terminal variants are not CCD entries; a 5′ end that keeps its phosphate is named
  after the component itself, e.g. 0G), `rosetta` (L: 0A …, no terminal variants,
  every polymer residue as ATOM because Rosetta ignores HETATM residues) and `pdb` (non-standard
  residues as HETATM). Any of these, plus common aliases (RA, G5, ADE …), is recognised on input;
  DNA and unknown residues pass through. *5′/3′ terminal names* can be switched off.
- **CONECT** records are stripped by default (until final deposition processing).
- **5′ phosphates**: `--strip-5p` (GUI: *Remove 5′-terminal phosphates*, ticked automatically for
  the Amber scheme) removes P/OP1/OP2/OP3 from every chain's 5′ nucleotide, as Amber's 5′ units
  (G5, LG5 …) expect; a residue with hydrogens gets HO5′ where the phosphorus was.
- **Chains**: restore the chain ids of a cpptraj PDB by copying them residue by residue from a
  reference (e.g. the structure the topology was built from), from a list per TER segment, or as
  A, B, C …; split chains per TER segment or at a residue (new ids); combine chains; sort them.
- **Numbering**: continuous across chains (default) or restarting per chain, optional start number
  and kept gaps; **SEQRES** records and an optional **FASTA** file (L chains are marked).

```bash
mdatk pdb check model.pdb                                   # chains, sequences, names, chirality (JSON)
mdatk pdb clean rosetta.pdb -o amber.pdb --scheme amber     # L names from the geometry
mdatk pdb clean cpptraj.pdb -o fixed.pdb --chain-reference start.pdb --scheme amber
mdatk pdb clean chimerax.pdb -o rosetta.pdb --scheme rosetta --l-ranges C --renumber --keep-gaps
mdatk pdb to-lrna in.pdb A:45-73 -o out.pdb [--no-terminal-names]   # the to_lrna.sh replacement
```

Add `-p project.mdatk.toml` to record the clean-up in the project's step log. The same module is
copied into Amber build steps, so generated scripts need only `python3`.

### Base-pair annotation

`mdatk pdb annotate` (and *Annotation* in the PDB clean-up window, applied after the clean-up)
writes the base pairs of a PDB file into its header as REMARK 950 records. Pairs come from the
first model; every other line (all models, atom serials, CONECT) is copied unchanged, an earlier
annotation is replaced and a MASTER record is dropped. `mdatk pdb clean` drops the records (they name
the input's residues), so annotate after cleaning; the clean-up window does this when *Annotation* is on. Strands are TER segments, so files with blank chain ids (Amber, cpptraj)
still get a `&` between strands:

```text
REMARK 950 MDATK BASE-PAIR ANNOTATION
REMARK 950 ENGINE barnaba 0.1.9
REMARK 950 PAIRS 20 (WATSON-CRICK 18, WOBBLE 2, OTHER 0)
REMARK 950 PAIR  A   1 G   B  24 C   WCc        Leontis-Westhof class (builtin: WCc / GUc)
REMARK 950 STACK A   1 G   A   2 C   >>         stacking (Barnaba only)
REMARK 950 CHAINS A,B                           one block per group of paired chains
REMARK 950 SEQ   GCAUGGCUAGCU&AGCUAGCCAUGC
REMARK 950 DBN   ((((((((((((&))))))))))))      Watson-Crick and wobble pairs
```

```bash
mdatk pdb annotate model.pdb -o annotated.pdb                   # Barnaba if installed, else builtin
mdatk pdb annotate model.pdb -o annotated.pdb --engine builtin --no-stacking
```

`--no-pairs`, `--no-stacking` and `--no-dotbracket` leave record types out; `-p PROJECT` logs the
step (with the Barnaba citation when it was used).

## Structure builder

*Workflow → Structure builder…*, `mdatk build` and the planner step *Structure builder (MDATK)* make
starting structures without external programs.

**Ideal helices.** An A-form RNA duplex (Watson-Crick complement, or an explicit partner 5'→3' of the
same length with mismatches / G·U wobbles) or a single strand. Every nucleotide is the Arnott A-RNA fiber
monomer (the `$ARNA` table of AmberTools' `dat/leap/parm/nucgen.dat`, embedded in MDATK): C3'-endo
sugars, χ ≈ −166°, 11 bp per turn (twist 32.7°, rise 2.81 Å), Watson-Crick pairs with C1'–C1' 10.9 Å and
O3'–P 1.59 Å. The **L** helix is the exact mirror image (left-handed, L-nucleotides) with L names from
the PDB clean-up schemes: by default A/C/G/U and 0A/0C/0G/0U (all ATOM, no terminal variants), or
`--scheme amber` (LA5 … LU3), `ccd`, `rosetta`, `pdb`. Chains start with a 5'-OH unless
`--phosphate5`; `--hydrogens` adds all hydrogens (heavy atoms only by default). G·U pairs get an
approximate wobble geometry (bases turned in their plane about C1'); other mismatches keep the
Watson-Crick frame and colliding bases are reported. The helix axis is z, the centre at the origin.

**Other pairings, parallel duplexes and triplexes.** `--pairing` (GUI: *Pairing*; planner option
*Pairing*) chooses the base-pair geometry of the duplex; `wc` (the default) is the A-form duplex above,
unchanged. The Leontis-Westhof rules fix the strand orientation:

| `--pairing` | pairs (strand 1 · strand 2) | strands | Leontis-Westhof |
|---|---|---|---|
| `wc` | A·U, G·C (G·U wobble) | antiparallel | cis WC/WC |
| `reverse-wc` | A·U, G·C (either strand) | parallel | trans WC/WC (barnaba `WWt`) |
| `hoogsteen` | A·U, G·C+ (U / C on the Hoogsteen edge of the strand-1 purine) | parallel | cis WC/Hoogsteen (`HWc`) |
| `reverse-hoogsteen` | A·U, A·A, G·G (partner WC edge on the strand-1 purine Hoogsteen edge) | antiparallel | trans WC/Hoogsteen (`HWt`) |
| `hh` | A·A (N6–N7 twice) | parallel | trans Hoogsteen/Hoogsteen (`HHt`), as PDB 4JRD |

`--partner auto` gives the canonical partner of each strand-1 base (the complement for `reverse-wc` and
`hoogsteen`, A·U / G·G for `reverse-hoogsteen`, the same base for `hh`); a parallel partner is read
5'→3' in the same direction as strand 1 (residue *i* pairs with residue *i*). For the Hoogsteen pairings
strand 1 must be the purine strand. `--triplex hoogsteen` (pyrimidine motif: U·A-U, C+·G-C, third
strand parallel to strand 1) or `--triplex reverse-hoogsteen` (purine motif: A·A-U, G·G-C, also U·A-U,
antiparallel) adds a third strand in the major groove on the Hoogsteen edge of the strand-1 purines of a
Watson-Crick duplex; `--third-strand` gives its sequence 5'→3' (default `auto`: the canonical motif
bases), `--chains A,B,C` / `--start=1,1,1` its chain id and numbering (default: the next free chain id).
Mismatches are allowed: they are placed in the frame of the reference pair and reported (strand-1
pyrimidines have no Hoogsteen edge for the Hoogsteen pairings and the triplex). Protonated C+ is
written as plain C with a note (protonate N3 or restrain the pair for the simulation).

How they are built: every nucleotide is the C3'-endo / anti nucleotide of PDB 4JRD (the poly(rA) parallel
duplex; its coordinates are embedded, PDB data are CC0) carrying the Arnott base, and each partner base
is placed coplanar on the H-bond donors / acceptors of the two edges (2.9 Å). `hh` is the 4JRD helix
itself (twist 41.8°, rise 3.79 Å; a poly(A) 10-mer has a phosphorus RMSD of 0.8 Å to 4JRD). For the
other pairings and the triplexes MDATK fits the helical screw and the backbone torsions that repeat a
reference level (A·U; A·U-U / A·U-A for the triplexes) with closed O3'–P links and no close contacts; a
partner strand with other pairs (G·C, G·G, mismatches) is then closed residue by residue (small rigid
shifts of each nucleoside, H-bonds kept). The Watson-Crick duplex of a triplex is fitted together with
the third strand (the Arnott A-form duplex leaves no room for it). These are **starting models to be
minimised**: the parallel and reverse pairings are not A-form RNA, and the notes say how far off the
model is — O3'–P link range and gaps, heavy-atom contacts closer than 2.4 Å, the H-bond distances, the
mismatches and the mean backbone torsions of each strand next to the 4JRD values. Long mixed sequences
can take some seconds (reverse-Hoogsteen G·G pairs the longest). The secondary-structure output of the
planner step (the dot-bracket for helix restraints) is written for Watson-Crick duplexes only (a
triplex: the third strand unpaired); the other pairings write none, since the Amber helix restraints
assume Watson-Crick H-bonds. L builds are the mirror images, as for the A-form helix.

```bash
mdatk build helix --seq AAAAAAAAAA --pairing hh -o polyA_hh.pdb          # parallel A·A, as PDB 4JRD
mdatk build helix --seq AAGAGAAGGA --pairing hoogsteen -o hoog.pdb       # parallel U·A / C+·G
mdatk build helix --seq GAUCAGUAGC --pairing reverse-wc --chirality L -o rwc_L.pdb
mdatk build helix --seq AAGAGAAGGA --triplex hoogsteen -o triplex.pdb    # third strand UUCUCUUCCU
mdatk build helix --seq AAGAG --triplex reverse-hoogsteen --third-strand GAGAA --chains A,B,T -o t.pdb
```

**Mutations.** `A:5G, B:20C`: the new base is superposed on the old glycosidic frame (C1', N9/N1 and the
two ring atoms bonded to it: purine N9/C4/C8 ↔ pyrimidine N1/C2/C6); sugar and backbone atoms stay as
they are, the residue keeps its name scheme, chirality and terminal variant (LG5 → LC5, 0G → 0C,
G5 → C5) and record type. Hydrogens of the old base are dropped (the new base has heavy atoms only).
With *compensatory* mutations (default) every mutated base that is Watson-Crick / G·U paired in the
input (builtin pair detection) gets its partner changed to the complement unless the partner is mutated
explicitly; `--wobble` keeps a G·U pair where the new base allows it. Every change is listed.

**Rigid-body fit.** The density-map fit of `mdatk density fit` (global search: rotations × FFT
translations, then local refinement; or local refinement of the model as placed) applied to a PDB
file: only the coordinates change (ANISOU records are rotated), every other line is copied. An ideal
self-complementary duplex is two-fold symmetric, so the fit may swap its strands.

```bash
mdatk build helix --seq GGCAUCGAUGCC -o helix.pdb                      # D duplex, A/C/G/U
mdatk build helix --seq GGCAU --chirality L --scheme amber --hydrogens -o lhelix.pdb
mdatk build helix --seq GGCAU --partner AAGCU -o wobble.pdb            # G·U + A·A mismatch
mdatk build helix --seq GGACU --partner none --chains C --start 45 -o strand.pdb
mdatk build mutate lhelix.pdb "A:2C, B:3G" [--no-compensatory] [--wobble] -o mutant.pdb
mdatk build fit model.pdb map.mrc --resolution 4 [--search local] [--step 30] -o fitted.pdb
```

Add `-p project.mdatk.toml` to log the step (the fit logs its correlation). The planner step writes a
short script calling `python3 -m mdatk build …` (MDATK must be importable by that Python); its inputs
are the model (mutate, fit) and the map (fit), its output the PDB.

## Restraint dictionaries, real-space refinement and MDFF

MDATK writes the scripts; acedrg, Phenix and Coot run on your machine. These command lines were
written from the programs' documentation and checked with stand-in executables only (none of the three
is installed where MDATK was tested), so check the first run - every version-dependent detail sits in
one table at the top of `mdatk/workflows/dictstep.py` and `mdatk/workflows/refine.py`.

### Restraint dictionaries for L-RNA (Coot / Phenix)

Coot and Phenix need an mmCIF restraint dictionary for every residue name they do not know, with atom names
identical to the model's. The CCD has the L-ribonucleotides 0A, 0C, 0G and 0U, but not the terminal variants
0A5/0A3, 0C5/0C3, 0G5/0G3 and 0U5/0U3 that `mdatk pdb clean --scheme ccd` writes at L-RNA chain ends.
acedrg (CCP4) and phenix.elbow (Phenix) build restraints from a SMILES string but invent their own atom names;
MDATK supplies the SMILES and renames the result:

* `mdatk dict list` - the twelve codes with the isomeric SMILES given to the programs (all four sugar centres
  L; `--json` for scripts). `mdatk dict smiles 0G5` prints one.
* `mdatk dict finish work_0G5/ --code 0G5 -o 0G5.cif` - finds the dictionary acedrg / eLBOW wrote (a file,
  a folder or a glob), maps every atom to the CCD names by graph matching (element + bonds; hydrogens by
  their heavy atom: H5'/H5'', H61/H62 ... from the generated coordinates), checks the sugar chirality and sets
  `_chem_comp.id`, name and `group` (`RNA` by default so the residue is linked into the chain;
  `--group non-polymer` for a free ligand). `--project` records the step in the project log.
* `mdatk dict pubchem CID` - the only command that uses the network: fetches the isomeric SMILES of a PubChem
  compound and reports which code it matches (wrong stereochemistry or compound is refused).
* Planner step **Restraint dictionaries · acedrg / eLBOW** (Refinement): program acedrg or phenix.elbow
  (Phenix 1.21 / 2.x, sourcing `phenix_env.sh`; or a CCP4 setup script), the codes (by default the 5'/3'
  variants the connected model needs - detected from the sugar geometry, so Amber or cpptraj names work - or
  all eight), custom `code = SMILES` lines and, off by default, "Fetch SMILES from PubChem when writing
  (network)" with `code = CID` lines. The step writes one `.smi` file per code, runs the program in
  `work_<code>/`, then a copy of `dictionaries.py` (`python3 ..._dictionaries_....py finish ...`) writes
  `<project>_<step>_<code>_<time>.cif`; the output port carries these dictionaries.

Conventions: `<code>5` = no 5'-phosphate, HO5' on O5' (a 5' end that keeps its phosphate is named
after the CCD component itself, `0G` etc., by `mdatk pdb clean --scheme ccd`, so it needs no extra
dictionary); `<code>3` = P, OP1, OP2 without the leaving OP3 and a
free 3'-OH (generated from the 5'-monophosphate; `finish` deletes OP3 and every restraint that uses it); the
parents keep OP3 and HO3' as leaving atoms of the polymer link, as in the CCD.

Note on the CCD files of 0A/0C/0G/0U (monomer-library form): their SMILES / InChI descriptors and ideal
coordinates describe the **D** enantiomer (CIP R,R,S,R at C1'/C2'/C3'/C4'; InChI .../m1/s1 like natural AMP);
only their `_chem_comp_chir` restraints are L. Do not feed those descriptors to acedrg / eLBOW - MDATK's table
has L SMILES (S,S,R,S). `mdatk dict finish` on such a file warns about the D coordinates.

### Real-space refinement (Phenix / Coot)

Two planner steps (category *Refinement*) refine a model against a cryo-EM map (MRC / CCP4). Both take the
model (PDB), the map and, optionally, restraint dictionaries (mmCIF, e.g. from the *Restraint dictionaries*
step) and produce the refined model (PDB) and a log. MDATK only writes the scripts; Phenix and Coot are not
run when the files are written.

**Phenix · real-space refinement** (`phenix-rsr`) writes a script running `phenix.real_space_refine` from
Phenix 1.21 or 2.x (the script sources `phenix_env.sh` of the installation folder you give, or uses the
conda / module lines of *Environment*): resolution (required), macro cycles, the run strategy as check
boxes (global minimisation, rigid body, local grid search, morphing, simulated annealing, ADP), nproc, an
optional random seed, the output prefix and extra parameters appended verbatim. *Nucleic-acid
secondary-structure restraints* switch on Phenix's automatic base-pair / stacking search. *Explicit base
pairs* are written to a parameter file (`…_basepairs_….eff`) from a connected or typed dot-bracket
(positions mapped to the nucleotides of the model in file order, or numbered continuously / per chain when
the model does not exist yet) or from the REMARK 950 annotation of the model (`mdatk pdb annotate`;
Watson-Crick and G·U pairs, optionally the STACK records as stacking restraints). Phenix names its output
`<prefix>_real_space_refined_NNN.pdb`; the script copies the newest one to the step's own
`…_refined_….pdb` (and the mmCIF to `…_refined_….cif`).

**Coot · real-space refinement (batch)** (`coot-rsr`) runs Coot 0.9 or 1.x without graphics
(`coot --no-graphics [--no-guano] --script …py`). The generated Python script works in both Coot
generations: every Coot function is looked up by name (the `coot` module first, then the global functions
of 0.9), and it is Python 2.7 compatible. Each chain - or only the residue ranges you give, e.g.
`A:1-20, B` - is refined in overlapping windows of N residues (`refine_zone` + accept), for a number of
cycles, with Coot's default map weight, `auto_weight_for_refinement` or a fixed `set_matrix` weight, and
optional torsion restraints. When the model exists while the files are written the windows are listed in
the script; otherwise the script finds the chains and residues in the model when it runs. To refine
interactively, open the model and map in Coot yourself.

Checks: a missing model / map / resolution, a resolution outside 1–20 Å, map files that do not exist,
L-RNA 5'/3' variants (0A5, 0G3 …) without restraint dictionaries (the CCD defines only 0A/0C/0G/0U),
dictionaries that miss a code of the model, Amber-style names (LA/LA5, A5/A3) and L sugars with D names
(cpptraj output) that Phenix and Coot would not recognise or would restrain as D - convert those with
`mdatk pdb clean --scheme ccd` first.

```bash
mdatk workflow generate phenix-rsr -i model=model_ccd.pdb -i map=emd_1234.map \
    -i dictionaries="0G5.cif;0C3.cif" --set resolution=3.4 --set phenix_version=1.21 \
    --set run_simulated_annealing=true -o refine
mdatk workflow generate coot-rsr -i model=model_ccd.pdb -i map=emd_1234.map --set coot_version=0.9 \
    --set residues="A:1-40" --set zone_size=8 -o refine
```

### MDFF with NAMD on the Amber topology (`namd-mdff`)

**NAMD · MDFF (Amber topology)** (category *Refinement*) fits the Amber system into a density map by
molecular dynamics flexible fitting (MDFF; Trabuco et al. 2008) in NAMD. NAMD reads the parm7 of the
*Amber build* natively (`amber on`, `parmfile`, `ambercoor`), so nothing is converted or
re-parameterised. Everything the build put into the topology is used exactly as built:

- the OL3 + CP and L-RNA atom types (LD / LE / LH ...) and the L-RNA dihedrals;
- every off-diagonal Lennard-Jones pair of the parm7 table: NBfix19, 0BPh, the microMg / nanoMg
  Mg²⁺ pair terms and the `addLJType` splits;
- the HMR masses.

A CHARMM PSF conversion would lose exactly these terms, so MDATK does not offer one.

**Inputs and output.**

- **Topology and coordinates:** the topology and coordinates of the Amber build, or a restart of
  *Amber · MD stages*. NetCDF restarts are converted with cpptraj; vacuum runs take the dry PDB. A
  PDB of a solvated system must be in the topology's atom order and carry the box (CRYST1).
- **Map:** the density map (MRC / CCP4, or an OpenDX density).
- **Map-frame model:** the PDB that was fitted into the map, usually the input of the Amber build.
- **Secondary structure** (optional), for the dihedral and base-pair restraints.
- **Output:** the fitted model, solute only, in the map's frame, ready for *Phenix* / *Coot*
  real-space refinement. It keeps the chains and residue numbers of the map-frame model (without
  one, every strand gets its own chain), with TER between strands, so renaming tools such as
  `mdatk pdb clean --scheme ccd` find every strand end.

Drop the step after an existing *Amber build* (or MD) step. It adds new files only; the steps
already in the plan and their files do not change.

**What the script does.**

1. **Start coordinates** (`…_start_….rst7`): NetCDF restarts are converted with cpptraj.
2. **Preparation** (`…_mdff_….py prep`, standard-library Python):
   - **Checks, before anything runs:**
     - 12-6-4 C4 terms (NAMD cannot apply them; use 12-6, microMg or nanoMg for NAMD);
     - per-dihedral 1-4 scaling different from 1.2 / 2.0;
     - a polarisable topology;
     - box, four-site water and time step that do not fit the topology (4 fs needs HMR).
   - **Map frame:** tleap moves (and may rotate) the solute when it solvates it. The script
     superposes the coordinates on the map-frame model and moves the potential with them; a
     rotation is resampled on a grid twice as fine as the map. It stops if less than 30 % of the
     coupled RNA atoms lie on density above the map's mean + 1σ (read from the map itself, whatever the
     threshold), which catches a missing or wrong frame.
   - **MDFF potential** (`…_potential_….dx`): V = (ρmax − max(ρ, threshold)) / (ρmax − threshold),
     Equation 3 of Trabuco et al. (2008) and the formula of VMD's `mdff griddx`. The paper sets the
     threshold at the minimum between the solvent and biomolecule peaks of the density histogram, or at
     the map's average when they are not resolved; the default is 0, as in `mdff griddx`. It is cropped to
     the RNA plus a margin and must fit into the periodic cell the way NAMD checks it: padded by one
     voxel and wrapped about its centre, no corner may reach half a cell vector. A truncated
     octahedron, the Amber build's default box, takes a grid only about half as wide as the box, so
     the usual 8 Å margin rarely fits; the *amber-mdff* template therefore builds a rectangular box.
     The script stops with an explanation before NAMD would.
   - **Coupling** (`…_coupling_….pdb`, as VMD's `mdff gridpdb`: occupancy 1 couples an atom, beta is its
     mass): the RNA heavy atoms by default, weighted by standard masses, so HMR does not change the map
     forces. *Couple ions too* adds the ions that start within the margin around the RNA; the
     counterions spread through the box are left alone.
   - **Restraints** (below) and the periodic cell (`…_cell_….xsc`, triclinic boxes included).
3. **Energy comparison** (on by default). sander and NAMD evaluate the start coordinates once, with
   the same cutoff, no rigid bonds, no long-range LJ correction and no map or restraints. The terms
   must agree:
   - bonds, angles and torsions within 0.01 %;
   - van der Waals within 0.1 %; this term holds every off-diagonal pair;
   - electrostatics within 0.2 %. In vacuum both programs sum plain Coulomb terms without a cutoff
     (NAMD with `switching off` and a 999 Å cutoff, sander with `cut=999`).

   On a mismatch nothing is fitted. This is how the step proves, for each topology, that NAMD kept
   the force field.
4. **Stages.** Each stage is a NAMD run, or several segments, restarting from the previous one:
   - minimisation at ξ = 0.3;
   - MD with the map coupling ξ ramped 0.05 → 0.3 over five segments;
   - MDFF at ξ = 0.3 (the default of VMD's `mdff setup`);
   - a final minimisation with strong coupling (ξ = 10).

   ξ is in kcal/mol per amu: each atom feels ξ × its mass × the potential gradient. A ramp needs at
   least two segments. Stage names must differ from each other and from the step's own files
   (`energy`, `prep`, `cell` ...).

   When the script is run again, finished runs are skipped as long as their inputs (topology, start
   coordinates, potential, coupling and restraint files) and the stage files up to them are unchanged.
   A change reruns that stage and every stage after it. The date line that tleap and ParmEd write into
   a topology does not count. A rebuild of a solvated system still restarts the fit: tleap places the
   ions anew at random (`addIonsRand`), which gives new start coordinates. To keep finished runs, run
   only the MDFF step (*Run selected steps*, or its own script), not the whole plan.
5. **Final model** (`…_mdff_….py final`): NAMD's last coordinates, back in the map's frame.

**Amber settings in NAMD.**

- Always: `exclude scaled1-4`, `1-4scaling 0.833333`, `scnb 2.0`, `readexclusions yes`,
  `switching off`, `cutoff` 10 Å (as MDATK's Amber MD), `rigidBonds all` with SETTLE.
- Time step: 4 fs when the topology has HMR masses, else 2 fs.
- Solvated runs (the default): PME and `LJcorrection on` in the build's periodic box, NVT with a
  Langevin thermostat. The map is fixed in space, so the box must not change.
- Vacuum runs: the dry topology, optionally with NAMD's GBIS implicit solvent.
- Four-site waters (OPC, TIP4P-Ew, TIP4P-D, TIP4P/2005) get `watermodel tip4`. For a topology chosen
  as a file, the step reads the extra points (and HMR masses) from the parm7. MDATK has not verified
  NAMD's handling of Amber extra points, so the planner warns: trust such a run only when the energy
  comparison passes.

**L-RNA-safe restraints.** VMD's stereochemistry plugins (`chirality`, `cispeptide`, `ssrestraints`,
whose restraint files `mdff setup -extrab` and the MDFF GUI add to a run) validate against D-nucleotide
libraries. On an
L-RNA they push every L sugar towards the inverted centre, and the model still looks fine
afterwards: the error would be silent. **Do not use them on these systems.** The step never calls
VMD and writes its own restraints:

- **Chirality** (`…_chirality_….txt`, NAMD `extraBonds`): one harmonic improper per ribose centre
  (C1′ O4′ N C2′, C2′ C1′ O2′ C3′, C3′ C2′ O3′ C4′, C4′ C3′ O4′ C5′), default
  200 kcal/mol/rad². N is the base atom bonded to C1′: N9 or N1, or C5 for C-nucleosides such as
  pseudouridine.
  - The targets come from the C3'-endo Arnott A-RNA fiber nucleotide of MDATK's helix builder:
    −35.0°, +40.0°, +38.3° and +33.3° for D residues, and the mirror image (opposite signs) for L
    residues.
  - Each residue's hand comes from its name (LA/LC/LG/LU and terminal forms, CCD 0A…) and must
    agree with its coordinates. A residue named D whose sugar is L, or the reverse, stops the run.
    So does any clearly defined sugar centre of the other hand, even in a residue whose overall
    geometry is ambiguous: its restraint would invert it. The planner reports these as soon as the
    files exist (for a NetCDF restart, the script does once it has converted the restart).
- **Secondary-structure dihedrals** (`…_ssdihedrals_….txt`, NAMD `extraBonds`), the RNA restraints
  of the MDFF paper (Trabuco et al. 2008).
  - α, β, γ, δ, ε, ζ and χ of every base-paired residue in the secondary structure are held at their
    starting values, k = 200 kcal/mol/rad².
  - The targets are measured on the model itself, so an L residue keeps its own mirror-image values.
  - α, ε and ζ reach into a neighbour only when it is bonded, never across a chain break.
  - The paper's authors discourage softening these restraints: they are what prevents overfitting.
  - The paper also restrains two inter-base distances per pair and stacked residues. MDATK uses the
    flat-bottom hydrogen-bond walls below for the pairs and does not restrain stacking.
- **Base pairs** (`…_basepairs_….txt`, NAMD `extraBonds` `wall`): the hydrogen-bond atom pairs of
  *Amber · helix restraints* from the connected or typed secondary structure.
  - Each distance is flat between 2.7 and 3.2 Å, with U = k (d − wall)² outside, k = 10 kcal/mol/Å²:
    the form of Amber's restraints (rk2 / rk3). The lower wall must be at least 0.1 Å: NAMD reads a
    lower wall of 0 as a bond of zero length.
  - Unlike Amber's restraints, the walls stay harmonic far from the window; they have no linear
    part.
- **Positional** (`…_posres_….pdb`, NAMD `constraints`): the RNA backbone or all heavy atoms,
  restrained to the start coordinates, in stages that ask for it.

All restraint files are NAMD `extraBonds` files (comments on lines of their own, 0-based atoms) plus
the positional-restraint PDB. Every setting is stored with the project, and the step cites NAMD, MDFF,
CPPTRAJ (it converts NetCDF restarts) and AmberTools (for the energy comparison).

**What NAMD does with an Amber topology.** These settings were checked against the NAMD 3.0 user's guide
(section 3.3, AMBER file and force field support; grid forces; extra bonded terms; GPU-resident features)
and the NAMD 2.14 guide and source code.

- NAMD copies the parm7 Lennard-Jones A/B table pair by pair. Every off-diagonal pair is kept, and 1-4
  pairs are divided by `scnb`.
- NAMD silently ignores 12-6-4 C4 terms and per-dihedral 1-4 scaling factors, which is why the step
  refuses topologies that have them.
- The box is never read from the parm7 or coordinates; the step writes it as an `.xsc` file.
- NAMD finds waters (for SETTLE) by an oxygen of 14–18 amu. A mass-repartitioned water would not be
  rigid, so the step stops if it finds one; repartition the solute only (ParmEd's default).
- NAMD 3.0 has a new parm7 reader (`oldParmReader` selects the old one). The energy comparison checks
  each topology with the NAMD that will run it.
- `1-4scaling` is written as in the guide's own examples; NAMD 3.0 also calls it `oneFourScaling`. The
  old name is deprecated only for NAMD's Python interface.
- The runs use NAMD's classic mode. The 3.0 guide's table of GPU-resident features lists neither grid
  forces nor extra bonded terms, so the GPU-resident mode (`CUDASOAintegrate`) would not run MDFF.

**Limits.**

- 12-6-4 topologies cannot be used; per-dihedral 1-4 scaling is refused.
- Four-site water is unverified, as described above.
- NPT is not offered.
- The potential must fit into the periodic cell as NAMD checks it (see above): use a rectangular box,
  a larger buffer or a smaller margin.
- If NAMD ever disagrees with sander on a topology (energy comparison), the fallback would be a
  conversion that keeps the Lennard-Jones pair table. ParmEd's `chamber` / CHARMM writers do not
  do that, so MDATK does not generate one; ask before planning around it.

`examples/namd_mdff/` holds a small D + L test case: the model, a simulated map, a project, the
generated step files and the files the preparation writes. Its stand-in topology carries names and
masses only and is not a force field (see its README).

```bash
mdatk workflow generate namd-mdff -i topology=sys.parm7 -i coordinates=sys.rst7 -i map=emd_1234.map \
    -i model=fitted_model.pdb --set sequence="GGGCAC GUGCCC" --set dotbracket="(((((( ))))))" -o mdff
```

## Citations

MDATK cites only what a project uses. Each planner step reports the programs, force fields,
modifications, water models and ion parameters its settings turn on (an Amber build with OL3 + CP,
NBfix19, TIP4P/2005 and nanoMg cites those papers, not the ones for options left off); the analysis
and video pipeline adds MDAnalysis, NumPy, SciPy, Matplotlib, and cpptraj / FFmpeg / VTK when they
are used. The citations appear

- in the header of every generated script (that step only),
- in the step log entries and in `MDATK_params_log.toml` (the whole project),
- in *About and citations* (the whole project), with *Export citations (BibTeX)…*.

The reference data live in `mdatk/_citation_data.py`; each entry records where its details were
checked (the paper's PDF, the project's own citation file, or a reference list).

## Command line reference

```text
mdatk gui [PROJECT]                start the GUI (opens PROJECT if given; nothing otherwise)
mdatk demo DIR [--n-frames N] [--no-solvent] [--run]
mdatk init OUT.mdatk.toml          write a project with the defaults (accepts the overrides below)
mdatk info      -p PROJECT         system summary and residue-group check (JSON)
mdatk list                         analyses, panel types, layout presets
mdatk clean     -p PROJECT [--force]
mdatk analyze   -p PROJECT [--only rmsd,contacts] [--force]
mdatk render    -p PROJECT [--no-video|--no-charts|--no-pdb] [--force] [--from-cache]
mdatk charts    -p PROJECT [--series k1,k2] [--formats png,svg,pdf] [--dpi N] [--from-cache]
mdatk export-pdb -p PROJECT [--frame-list "0, last"] [--times "1ns"] [--selection "Aptamer"]
mdatk preview   -p PROJECT --frame N [--output still.png] [--from-cache]
mdatk density info MAP | fit -p PROJECT [--map M] [--resolution R] [--search global|manual] [--save]
mdatk density simulate -p PROJECT --resolution R -O OUT.mrc [--frame F] [--placed]
mdatk workflow tools|generate|template|check|write|run|restraints   (see Script wizard and Project Planner)
mdatk pdb check|clean|to-lrna|annotate   (see PDB clean-up)
mdatk build helix|mutate|fit       (see Structure builder)
mdatk dict list|smiles|finish|pubchem   (see Restraint dictionaries; only 'pubchem' uses the network)
```

Common options: `--top`, `--traj …`, `--frames start:stop:step`, `--clean/--no-clean`,
`--cpptraj PATH`, `--conda-env NAME`, `-o/--out-dir`, `--cache-dir`, `-q/--quiet`, `-v/--verbose` and
`-s/--set KEY=VALUE` (repeatable; values are JSON when they parse as JSON), for example:

```bash
mdatk render -p run.mdatk.toml --set analyses.contacts.cutoff=5.0 \
    --set 'visual.rotation=[-90, 30, 0]' --set visual.glow_mode=current --no-pdb
```

Set `MDATK_DEBUG=1` to get full tracebacks.

## Project files

A project is a TOML file (`*.mdatk.toml`) mirroring `mdatk.config.Project`. Every entry has a
comment above it saying what it does (planner step settings carry their option's description), and
settings that are not set are written as commented lines (`# stop =   (not set)`). Below the settings
the file keeps a step log: one `[[log]]` entry per analysis run, render, PDB clean-up, plan write or
generated script that MDATK ran (time, status, duration, command, folder, files, summary, citations
and the tail of the program output). Saving the settings keeps the log; when the file would grow
beyond 50 MB the log moves to `<name>.log.toml` next to it. JSON presets of earlier versions are not
read.

```text
notes      free text (aims, decisions, to-dos)
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

Unknown keys are ignored with a warning and missing keys take their defaults, so project files stay
compatible across versions. The recent-projects list is kept in `~/.config/mdatk/settings.toml`.

## Architecture and extending MDATK

```text
mdatk/
  config.py          parameter model, hashing
  projectfile.py, tomlio.py   commented TOML project files, step log, parameter logs, recent projects
  pdbtools.py        PDB hygiene (ChimeraX repair, L-RNA detection and names, chains, numbering)
  citations.py       per-use citations (data in _citation_data.py)
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
- The script wizard and planner write files; they run them only on request (*Run all…*, *Run selected
  steps*). Generated
  scripts call the programs by name
  (`tleap`, `parmed`, `pmemd.cuda`, `rna_denovo`, `gmx` …); use the *Environment* section of each
  step (conda environment, `module load`, `source amber.sh`) when they are not on PATH.
- 3D representations are made for nucleic acids. Protein chains get a round tube through CA in
  cartoons and ribbons (no helix / sheet detection), and licorice, VDW, surface and trace work for
  any residue. PDF / SVG previews embed the 3D layer as an image. Surfaces are rebuilt every frame (tens of ms for a few thousand atoms; the grid
  coarsens to 1.5 Å above 5 000 atoms).
- The builtin base-pair engine knows canonical Watson–Crick and G–U wobble pairs only; use
  Barnaba for non-canonical pairs and stacking. The arc diagram keeps one fixed layout (residues
  in file order), so pseudoknots appear as crossing arcs.
- The refinement and dictionary steps were not run against real Phenix, Coot or acedrg / eLBOW
  installations (only stand-in executables); the Phenix parameter scopes for explicit base pairs,
  the Coot function names that differ between 0.9 and 1.x, and the output file names of acedrg / eLBOW
  are assumptions listed in the version tables of `mdatk/workflows/refine.py` and
  `mdatk/workflows/dictstep.py`.
- The CCD files of 0A / 0C / 0G / 0U describe the D enantiomer in their SMILES, InChI and ideal
  coordinates; only their chirality restraints are L. MDATK's dictionary step uses its own L SMILES.
- The parallel, Hoogsteen and reverse-Hoogsteen helices and the triplexes are clash-free starting
  models, not fibre models: their helix parameters are fitted so that the backbone closes without close
  contacts (e.g. reverse-wc twist ≈ 22°, rise ≈ 4.2 Å; triplex rise ≈ 5 Å, so stacked bases are
  ≈ 4.4 Å apart), and some backbone torsions are unusual. Minimise or restrain them before production
  MD; long sequences take a few seconds per 10 nt to build.
- Ideal helices are starting models: hydrogens, OP3 and G·U wobble placements are idealised, and a
  self-complementary duplex fitted into a map may come out with its strands swapped.
- Dual-axis plots are available because they are useful for juxtaposing e.g. contacts with
  another metric, but two y scales on one plot invite reading a correlation into an arbitrary
  alignment; prefer two stacked time-series panels when in doubt.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
