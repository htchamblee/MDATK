<p align="center"><img src="mdatk/assets/rnado-wordmark.png" alt="RNAdo, MDATK's workflow planner" width="600"></p>

<p align="center"><b>RNAdo</b>, MDATK's workflow planner, ships with MDATK as a program of its own.</p>

# MDATK

MDATK, the Molecular Dynamic Analysis Tool Kit, is a Linux-native GTK 3 desktop toolkit, with a
companion command line, for molecular dynamics (MD) simulations of RNA systems. It is two programs that
work on the same project file (`*.mdatk.toml`):

- **The analysis window** (`mdatk-gui`, and the `mdatk` command line) analyses MD trajectories and turns
  them into multi-panel MP4 videos, high-resolution time charts and selected-frame PDB files.
- **RNAdo**, MDATK's workflow planner (`rnado`), lays out the steps from sequence to MD ensemble on a
  canvas and writes, and on request runs, consistently named control files and scripts for Rosetta
  FARFAR2, SimRNA, RNAMake, Amber, GROMACS, NAMD, VMD, Phenix, Coot and scripts of your own. Its script
  wizard, PDB clean-up and structure builder prepare the files that go in.

Everything runs locally; MDATK makes no network calls of its own, with one exception you ask for:
PubChem look-ups of SMILES for restraint dictionaries (`mdatk dict pubchem`, or *Fetch SMILES from
PubChem when writing (network)* in the restraint-dictionary step, off by default). The downloads in
`packaging/` (FFmpeg, appimagetool, the MSYS2 and pip packages that the Windows build's
`--install-deps` installs, and `pin_ffmpeg_url.py` asking GitHub which release holds the FFmpeg build)
happen only when the stand-alone versions are built, never when MDATK runs. Generated
scripts run only when you start them or ask RNAdo to; what they do is up to the programs they call.

---

## Contents

- [What's new in 0.0.12](#whats-new-in-0012)
- [Installation](#installation)
  - [Stand-alone downloads (no Python needed)](#stand-alone-downloads-no-python-needed)
  - [Installing from source](#installing-from-source)
  - [Applications menu (Linux)](#applications-menu-linux)
  - [Optional: cpptraj](#optional-cpptraj)
- [Quick start](#quick-start)
- [The GUI](#the-gui)
  - [The analysis window](#the-analysis-window)
  - [RNAdo's window](#rnados-window)
  - [One project in both programs](#one-project-in-both-programs)
- [Analysis workflow](#analysis-workflow)
- [Analyses](#analyses)
  - [Residue groups and selections](#residue-groups-and-selections)
- [Structural view, colouring and glow](#structural-view-colouring-and-glow)
- [3D representations](#3d-representations)
- [Density maps](#density-maps)
- [Panels, data keys and layouts](#panels-data-keys-and-layouts)
- [Outputs](#outputs)
- [Caching and reproducibility](#caching-and-reproducibility)
- [RNAdo, MDATK's workflow planner](#rnado-mdatks-workflow-planner)
  - [Starting RNAdo](#starting-rnado)
  - [The plan](#the-plan)
  - [Script wizard](#script-wizard)
  - [Running steps](#running-steps)
  - [File names](#file-names)
  - [Previous runs](#previous-runs)
  - [Step types](#step-types)
  - [FARFAR2 decoys: scores, run audit and frames](#farfar2-decoys-scores-run-audit-and-frames)
  - [Custom scripts](#custom-scripts)
  - [Decoy interfaces](#decoy-interfaces)
  - [Command line](#command-line)
  - [What was tested](#what-was-tested)
- [PDB clean-up](#pdb-clean-up)
  - [Base-pair annotation](#base-pair-annotation)
  - [Model checks](#model-checks)
- [Structure builder](#structure-builder)
- [Restraint dictionaries, real-space refinement and MDFF](#restraint-dictionaries-real-space-refinement-and-mdff)
  - [Restraint dictionaries for L-RNA (Coot / Phenix)](#restraint-dictionaries-for-l-rna-coot--phenix)
  - [Real-space refinement (Phenix / Coot)](#real-space-refinement-phenix--coot)
  - [MDFF with NAMD on the Amber topology (`namd-mdff`)](#mdff-with-namd-on-the-amber-topology-namd-mdff)
- [Decoy ensembles, map-model scores and statistics](#decoy-ensembles-map-model-scores-and-statistics)
- [Citations](#citations)
- [Command line reference](#command-line-reference)
- [Project files](#project-files)
- [Architecture and extending MDATK](#architecture-and-extending-mdatk)
- [Testing](#testing)
- [Notes and limitations](#notes-and-limitations)
- [License](#license)

---

## What's new in 0.0.12

- **Your analysis scripts as RNAdo steps.** New steps and commands, each generalised so that ranges,
  chains, registers and paths are options: *Model check* (`mdatk pdb qc`: clashes, O3′–P links with the
  strain a stretched link would put into tleap, sugar hands, kinks, a filter that passes on up to N
  models and fails only below a minimum), *Decoy interfaces* (`mdatk decoys`: census, clusters, segment
  support; builtin, barnaba or DSSR pairs), *Register census* and *Ensemble comparison* (`mdatk
  ensemble`), *Map-model score* (`mdatk map score`: per-residue Q-score with ChimeraX's `qscore`,
  ranked on dQ), *Body hybrids* (`mdatk pdb hybrids`), *Rosetta · run audit* (`mdatk rosetta audit`:
  flags, Rosetta build and seeds of every run) and *MDATK · frame check* (`mdatk frame`). See [Model
  checks](#model-checks), [FARFAR2 decoys](#farfar2-decoys-scores-run-audit-and-frames), [Decoy
  interfaces](#decoy-interfaces) and [Decoy ensembles](#decoy-ensembles-map-model-scores-and-statistics).
- **FARFAR2 decoys:** a decoy table (`run`, `tag`, `pdb`, every score term by name); validity filters
  (hands, bases, closure, defining pairs) before clustering and quality filters after it; *Every decoy
  (PDB)*, a large dot on the right of the FARFAR2 box, passes the whole set to the steps that take
  several models. A template, *Decoy validation (FARFAR2)*, chains them.
- **Applications menu:** `mdatk desktop install` (or *Add MDATK and RNAdo to the applications menu…*)
  adds MDATK and RNAdo as two entries, from a source install or the AppImage (see [Applications menu
  (Linux)](#applications-menu-linux)).
- **Completed runs move aside:** writing a step again first moves its completed run, with the scripts
  that made it, into `<step folder>/superseded/<stamp>_<time>/` (see [Previous runs](#previous-runs)).
- **CPU and GPU budgets** replace *Parallel*: scripts start while their cores and GPUs fit (default:
  half the cores, every GPU found); NAMD counts its `+p` cores and a GPU; a GPU step gets a free GPU in
  `CUDA_VISIBLE_DEVICES` unless it names its own (see [Running steps](#running-steps)).
- **Stop buttons:** *Stop scripts* in RNAdo's toolbar stops every script; a stop button on each running
  box stops that one.
- **The commands a step runs:** the script wizard has a *Commands* tab with the commands the step's
  script will run.
- **Fixes:** MDATK no longer stays running after its windows are closed (each program quits with its main
  window, and the programs it starts no longer inherit the AppImage's descriptors); the NAMD MDFF result
  PDB's REMARK is numbered, so ChimeraX reads it; DSSR 2.9 output with blank chains is read.
- **Generated workflow files** of the earlier step types are unchanged unless a new option is turned on,
  apart from the version in their headers and the PDB tools and PDB preparation copied into Amber builds (see
  [CHANGELOG.md](CHANGELOG.md)).

Earlier releases: see [CHANGELOG.md](CHANGELOG.md).

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
./MDATK-<version>-x86_64.AppImage                          # the analysis window
./MDATK-<version>-x86_64.AppImage project.mdatk.toml       # the analysis window with a project
./MDATK-<version>-x86_64.AppImage rnado project.mdatk.toml # RNAdo, the workflow planner
./MDATK-<version>-x86_64.AppImage rnado write -p plan.mdatk.toml  # = mdatk workflow write
./MDATK-<version>-x86_64.AppImage --help                   # the command line (any mdatk command)
./MDATK-<version>-x86_64.AppImage demo ~/mdatk-demo        # e.g. the synthetic demo system
./MDATK-<version>-x86_64.AppImage desktop install          # MDATK and RNAdo in the applications menu
```

With no argument (or a project file) the AppImage starts the analysis window; with `rnado` it is
`rnado` (RNAdo, or with an action `mdatk workflow ACTION`); with any other command it is the `mdatk`
command line. For shorter names, link it: `ln -s ~/Apps/MDATK-<version>-x86_64.AppImage
~/.local/bin/mdatk`; a link named `mdatk-gui` always starts the analysis window, and one named `rnado`
(or `rnado-gui`) is `rnado` with all its arguments. Once the AppImage is integrated into the desktop
(AppImageLauncher, appimaged), the MDATK launcher has an *RNAdo (workflow planner)* action
(right-click). If the AppImage does not start because FUSE is missing (some containers and minimal
installs), run it with `--appimage-extract-and-run` or install `fuse`/`libfuse2`; *Open in RNAdo…* then
starts RNAdo the same way.

**An RNAdo entry in the applications menu:** desktop integration takes only one desktop entry from an
AppImage (MDATK's), so RNAdo gets no menu entry of its own that way. *Add MDATK and RNAdo to the
applications menu…* (main menu of either window) or `./MDATK-<version>-x86_64.AppImage desktop install`
adds both, with their icons (see [Applications menu (Linux)](#applications-menu-linux)). The entries start
the AppImage at its current path: after moving or renaming it, add them again from the new place.

**Windows 10 / 11** (64-bit) - **beta, not yet tested on real machines**: run
`MDATK-<version>-windows-x64-setup.exe` (installs per user, no administrator rights needed;
Start-menu entries "MDATK" and "RNAdo", optional desktop shortcuts, optional PATH entry for
`mdatk.exe`) or unzip `MDATK-<version>-windows-x64.zip` anywhere and start `mdatk-gui.exe` (the
analysis window), `rnado-gui.exe` (RNAdo, the workflow planner) or `mdatk.exe` (the command line). On
Windows:

- the generated workflow scripts (Amber, GROMACS, Rosetta, Phenix, Coot ...) are bash scripts:
  MDATK writes them but does not run them - run them on a Linux machine, a cluster or in
  [WSL](https://learn.microsoft.com/windows/wsl/); RNAdo's *Run* says so;
- cpptraj (AmberTools) does not run natively, so clean trajectories on Linux / WSL first, or
  analyse trajectories that are already cleaned.

**What you still install yourself** (both platforms):

- **cpptraj / AmberTools** - only for automatic solvent stripping and cleaning (see [Optional:
  cpptraj](#optional-cpptraj)); the stand-alone builds detect it exactly like the source installation
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

This installs four commands: `mdatk` (CLI), `mdatk-gui` (the analysis window), `rnado` and
`rnado-gui` (RNAdo, the workflow planner); `python -m mdatk` works too. Activate the venv
(`. .venv/bin/activate`) in each new shell before using them. `mdatk desktop install` (or *Add MDATK
and RNAdo to the applications menu…* in either window) puts both programs into your applications menu,
started from this installation without activating the venv (see [Applications menu
(Linux)](#applications-menu-linux)).

### Applications menu (Linux)

*Add MDATK and RNAdo to the applications menu…* in the main menu of the analysis window and of RNAdo,
`mdatk desktop install` or `rnado --install-launchers` writes two desktop entries, `mdatk.desktop` (with
its *RNAdo* action) and `rnado.desktop`, into `~/.local/share/applications`, and their icons into
`~/.local/share/icons/hicolor/<size>/apps` (`mdatk.png` and `rnado.png` in every size, `rnado.svg` in
`scalable`); `$XDG_DATA_HOME` replaces `~/.local/share` when it is set. The window first lists the files
and what the entries start, and afterwards what it wrote. The entries start what is running now:

| MDATK runs from | MDATK's entry | RNAdo's entry |
|---|---|---|
| the AppImage | `"<AppImage>" gui %f` | `"<AppImage>" rnado %f` |
| the AppImage without FUSE (extracted) | `env APPIMAGE_EXTRACT_AND_RUN=1 "<AppImage>" gui %f` | the same with `rnado` |
| a bundle folder | `<folder>/mdatk gui %f` (`mdatk-gui` where there is one) | `<folder>/rnado-gui %f` |
| an extracted AppImage folder (`squashfs-root`) | `<folder>/AppRun gui %f` | `<folder>/AppRun rnado %f` |
| a pip / venv installation | the `mdatk-gui` script that runs this Python (the venv's `bin`, the user's `~/.local/bin` or `PATH`), else `<python> -m mdatk gui %f` | `rnado-gui`, else `<python> -m mdatk.gui.rnado %f` |
| a source checkout that is not installed | `env PYTHONPATH=<checkout> <python> -m mdatk gui %f` | the same with `-m mdatk.gui.rnado` |

Paths are quoted as the Desktop Entry Specification asks (spaces, `$`, quotes, backslashes). The entries
hold these paths: after moving the AppImage, the bundle folder or the venv, add them again.
`mdatk desktop status` lists the files, which are there, which start something else (`outdated`) and what
the entries start now.

Every file MDATK writes carries the mark `X-MDATK-Installed=<version>` (a key of the desktop entry, a text
chunk of the PNG, a comment of the SVG). Adding again rewrites only MDATK's own files; an entry or icon of
the same name that MDATK did not write (one you made yourself, or `xdg-icon-resource`'s) is left alone and
listed, and `mdatk desktop install --replace` moves it to `<file>.bak` first. *Remove them* (offered by
the same question once MDATK's entries are there) or `mdatk desktop uninstall` removes the marked files
(of any MDATK version) and nothing else.
Afterwards `update-desktop-database` runs on the applications folder and `gtk-update-icon-cache` on the
icon folder when that has a cache (without one, GTK reads the folders; no cache is made), each only when
installed; a failure of either is reported, never fatal. Some desktops show new entries only after the
next log-in. An AppImage that AppImageLauncher integrated keeps its own MDATK entry, so MDATK can then be
in the menu twice. On Windows the installer makes the Start-menu entries *MDATK* and *RNAdo*; the menu
item says so and writes nothing.

### Optional: cpptraj

cpptraj (part of [AmberTools](https://ambermd.org/AmberTools.php), e.g.
`conda install -c conda-forge ambertools`) is only needed for automated solvent stripping and
cleaning. Everything else works without it, e.g. on trajectories you have already cleaned. If
you use it, set the **conda environment** that provides it (e.g. `AmberTools26`) in the cleaning
options or with `--conda-env`; every cpptraj call then runs through `conda run -n <env>`, which is
equivalent to `conda activate <env>`. When no environment is set, MDATK uses cpptraj from `PATH`
or `$AMBERHOME/bin`, and otherwise finds an environment by looking for `bin/cpptraj` in
`~/miniforge3`, `~/mambaforge`, `~/miniconda3`, `~/anaconda3`, `/opt/conda`, … (the analysis
window pre-fills the field with what it finds; `mdatk list` prints the candidates).

## Quick start

```bash
mdatk demo ./demo                        # synthetic solvated RNA aptamer/target system + project
mdatk gui ./demo/demo.mdatk.toml         # open it in the analysis window and press "Render"
rnado ./demo/demo.mdatk.toml             # plan the steps before the analysis in RNAdo
# or headless:
mdatk render -p ./demo/demo.mdatk.toml   # add --clean to strip solvent with cpptraj first
```

The demo writes a genuine Amber `demo.parm7` + `demo.nc` pair (a 12-bp aptamer/target duplex,
three Mg²⁺ ions, Na⁺ and ~4 000 waters, 120 frames) in which one end of the duplex frays and
re-anneals, a Mg²⁺ bridge between the strands breaks and reforms, and another ion visits a
phosphate. It exists to exercise every feature without real data; its geometry is schematic.

For your own system start from `examples/aptamer_target.mdatk.toml` (relative paths in a
project file are resolved against its folder) or from the analysis window, which opens empty: nothing
is loaded at start-up; *Open project* has a drop-down with the recent projects, the demo project and
*Browse…*. RNAdo without a project opens an empty plan; `examples/rna_plan.mdatk.toml` is a complete
example plan.

## The GUI

Each program has one main window. Both open, save and remember the same project files, run long jobs in
a background thread that can be cancelled, and name the extensions their file choosers accept, e.g.
"Trajectory files (.nc, .mdcrd, .crd, .trj, .dcd, .trr, .gro, .xtc, .pdb, .xyz, .binpos, .ncdf)". Value
fields (spin buttons, drop-downs, sliders) ignore the mouse wheel, so scrolling a form never changes a
value.

### The analysis window

`mdatk-gui` (or `mdatk gui [PROJECT]`) opens the analysis window. It *is* the figure: it shows the
selected layout drawn by the same code that renders the video, at the video's aspect ratio, filled with
**example data** (sine-like traces, residue heatmaps sized from your groups and a schematic duplex) so
the layout can be arranged without computing anything. *Load real frame* replaces the schematic with the
reference frame of your trajectory.

- **Drag a panel** onto another slot to swap them; panels snap to the slots of the grid preset.
- **Drag the structure panel** to rotate the model (drag its title strip, or hold Shift, to move
  it instead). The X/Y/Z sliders follow and show the exact angles in degrees and radians.
- **Double-click a panel** to edit its settings.

The settings live in a permanent side bar on the left (fixed 520 px wide, never scrolled
sideways), opened at the *Input & cleaning* page; the icon bar above it switches pages (hover for the
name):

| Page | Contents |
|---|---|
| **Input & cleaning** | project notes, topology/trajectory/external-data choosers, *Use the plan's files*, frame range, time-step override, display time unit, cpptraj cleaning options (conda environment, masks, imaging, fitting), *Run cleaning now*, *Inspect system* |
| **Residue groups** | table of named groups (residue ranges or selections) with colour swatches and a colour picker; renames follow every use; *Groups from chains* |
| **Analyses** | enable switches and parameters for RMSD, contacts, H-bonds, O–Mg–O / Mg occupancy, π–π stacking, base pairs and intermolecular energy |
| **Structure view** | orientation sliders, residue-number tags, colouring, proximity and glow, appearance, 3D representations, density map |
| **Layout & panels** | grid preset and, per slot, panel type, data, residue groups, extra keys, title (follows the data until you type your own) and options; *Render still* renders one frame from real data |
| **Outputs** | output/cache folders, video, charts, PDB export, timestamps, *Re-render video + charts from cache* |

The settings are checked as you edit them (a moment after the last change). A page with problems gets
a **red badge with their count** on its button (amber when they are only warnings); clicking the button
also scrolls to the page's next problem. The field itself is **flagged**: red frame and text, an error
icon at the end of entries, and the message as its tooltip; in the group, RMSD-target and external-data
tables the offending cell is tinted. The **problems button** next to the page title lists every
problem with its location; click one to go to its field. Flags and badges disappear when the setting
is fixed. A setting that *Render* would not use is flagged as a warning, not an error (e.g. the proximity
set while the video is off, or a draft group nothing uses), and does not stop it; *Render still*,
*Export PDBs now* and the density map's *Global search* / *Refine* check what they use themselves (see
[Residue groups and selections](#residue-groups-and-selections)). A selection that fails during a run
(e.g. a group that matches no atom of the structure) is flagged the same way until that setting or the
residue groups change.

The bottom bar holds **Render** (Ctrl+R), **Cancel**, the progress bar, **Open output folder** (shown
in your file manager) and **Open video** (the most recent video); the log of the pipeline and of
cpptraj is in the expander above it. The header bar has *Open project* (Ctrl+O; its drop-down lists the
recent projects, the demo project and *Browse…*), *Save* (Ctrl+S) and *Save as…* (Ctrl+Shift+S). The
main menu has *New project (defaults)*, *Create demo system…*, *Project log…* (the steps that ran),
*Export citations (BibTeX)…*, *Add MDATK and RNAdo to the applications menu…* (see [Applications menu
(Linux)](#applications-menu-linux)) and *About and citations*. The **RNAdo** menu next to them starts
RNAdo on the project:

- *Open in RNAdo…* (Ctrl+P) opens the plan;
- *Script wizard: open in RNAdo…* (Ctrl+Shift+W), *PDB clean-up: open in RNAdo…* and *Structure
  builder: open in RNAdo…* open the plan and that tool;
- *Use the plan's files as inputs* is the *Use the plan's files* button of the Input & cleaning page
  (see [The plan](#the-plan)).

RNAdo reads the project file, so the window first offers to save unsaved changes (a project that was
never saved can also be left out: RNAdo then opens without it). A start of RNAdo that fails within a few
seconds is reported in the analysis window.

### RNAdo's window

`rnado` opens RNAdo's window (see [Starting RNAdo](#starting-rnado) for the other ways). Top to bottom:

- **Header bar:** *Open* (Ctrl+O; its drop-down lists the recent projects and *Browse…*), *Save* (Ctrl+S;
  it saves the plan into the project file), the tools *Script wizard* (Ctrl+Shift+W), *PDB clean-up* and
  *Structure builder*, and a menu with *New project* (Ctrl+N), *Open project…*, *Save*, *Save as…*
  (Ctrl+Shift+S), the three tools, *Show the output log*, *Project log…*, *Add MDATK and RNAdo to the
  applications menu…* (see [Applications menu (Linux)](#applications-menu-linux)) and *About RNAdo* (with
  the citations of the plan and *Export BibTeX…*).
- **The planner:** its toolbar (*Add step*, *Templates*, *Delete*, *Arrange*, zoom, *Fit*, *Check
  previous runs*, *Write all files*, *Run all…*, *Run selected steps*, *Stop scripts*, *Open folder*),
  the naming bar (project name, *Time stamp*, *New stamp*, *Step folders*, *Move completed runs aside*,
  the plan folder and *Change…*, then *CPU cores*, *GPUs* and *At once* for running steps), the canvas
  and the details pane of the selected box (see [The plan](#the-plan) and [Running steps](#running-steps)).
- **Output log panel:** drag its edge to resize it; the arrow or Ctrl+L folds it away, and a run
  unfolds it. It shows the planner's messages (files written, completed runs moved aside with their files,
  previous runs checked, runs used or dropped), MDATK's warnings, and every line the scripts print, marked
  with their step (`[03] …`).
  Standard error is red, lines starting with ERROR (and failed steps) bold red, lines starting with
  WAITING or WARNING (and waiting or stopped steps) amber. Each step gets a start line (with the cores
  and GPU it got), a finished / failed / cancelled / stopped line, and a waiting or not-started line
  when it does not run to the end. *Step* shows the lines of one step
  only; *Last run* shows the stored log of the selected box's last run, from the project's step log. The
  other buttons follow new lines (auto-scroll), copy or save the lines shown as text, or clear them.
  Lines are inserted in batches and the panel keeps the last 5 000–6 000 lines; a script that prints
  faster than the panel can show them gets lines skipped, with a note per step of how many, so the
  window stays responsive and *Cancel* reachable. Every run is also recorded in the project file.
- **Status bar:** the planner's status line (steps, connections, errors, warnings, plan folder), the
  progress bar and *Cancel*, which stops every running script (as *Stop scripts* does). Closing RNAdo
  during a run asks first and stops the scripts.

### One project in both programs

Both programs can have one project open at the same time; each owns one part of the file.

- RNAdo writes only the plan (the file's `workflow` section) and leaves the analysis settings as the
  file has them. When the file brings new analysis settings it takes them; renamed residue groups are
  followed in its unsaved plan, so a plan changed only by those renames is no conflict.
- The analysis window writes the analysis settings and the plan it finds in the file, with its own
  residue-group renames applied to it.
- When the file changed on disk (the window gets the focus, or the file monitor reports it), each
  program takes the other's part, and reloads its own part when it has no unsaved changes there.
  Otherwise it asks in a bar above its content: *Reload it* / *Keep mine* in RNAdo (the plan), *Reload
  them* / *Keep mine* in the analysis window (the analysis settings). After *Keep mine*, saving replaces
  the other program's version without asking again.
- Saving asks first when the other program saved the same part meanwhile (*Replace with mine* /
  *Reload theirs* in RNAdo, *Replace them* in the analysis window), then re-reads the file and writes
  it under the file's lock after any question, so nothing the other program saved meanwhile is lost.
  *Save* without changes of your own takes the newer version instead ("Nothing to save").
- Opening a script wizard is no change to the plan.

## Analysis workflow

The analysis window and the `mdatk` command line read Amber topologies (`.parm7`) and NetCDF
trajectories (`.nc`) directly, can strip and clean solvated trajectories with **cpptraj**, compute the
analyses you select in a single streaming pass, cache every result, and render a configurable grid of
panels: a rotatable structural view with proximity glow (a 2D backbone projection, or VMD-style 3D
representations: New Cartoon with base slabs, ribbon, tube, licorice, VDW spheres, surface), time
series with a moving time cursor, per-residue heatmaps and contact maps. A cryo-EM / crystallographic
density map can be fitted to the model and drawn as a projected shadow.

1. **Load** a topology (`.parm7` preferred; cleaned PDB or any MDAnalysis topology) and one or
   more trajectories (`.nc` preferred; DCD, XTC, … are concatenated in order).
2. **Clean** (optional): MDATK writes a cpptraj script that strips solvent and ions (Mg²⁺ is
   kept by default), images/re-centres (`autoimage`), optionally fits to the first frame, a
   numbered frame or a reference file, and writes a cleaned trajectory plus the matching stripped
   topology and a reference PDB. The cpptraj log streams into the analysis window. Results are cached and
   reused while inputs and cleaning parameters are unchanged.
3. **Configure** residue groups, analysis parameters, colouring, proximity glow, orientation,
   grid layout and panel contents.
4. **Render**: MDATK computes the analyses the enabled outputs need (with progress), caches them,
   then writes the MP4, the charts and the PDB files, plus a parameter log.
5. **Reproduce**: save/open project files, re-render from the cache, or replay a run from its
   `MDATK_params_log.toml`.

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

Three internal analyses feed the structure panel: **trace** (one point per residue — the first
available atom of the *trace atoms* list such as `C4'` or `P`, or the heavy-atom centroid for
ligands — superposed onto the reference frame), **proximity** (per residue minimum distance
to the proximity set, stored up to the *proximity search radius* so the render-time cutoff can be
changed without recomputation) and **atoms** (per-atom coordinates of the structure-view residues,
superposed like the trace, for the 3D representations and the density fit).

A fourth internal analysis, **basepairs**, feeds the secondary-structure panel (it runs only when a
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
  non-canonical ones (*Barnaba: keep non-canonical pairs* draws them too). Barnaba's classification
  assumes D-nucleotide geometry; for L-RNA the edges and cis/trans may be mirrored, so the builtin
  engine is the safer choice there.

The result is a sparse residue × residue map per frame (1 = Watson–Crick, 2 = wobble, 3 = other),
cached like the other analyses.

### Residue groups and selections

Groups are named residue ranges such as `1-40, 45, 50-60` (chain/segment qualified: `A:1-40`)
or raw [MDAnalysis selections](https://docs.mdanalysis.org/stable/documentation_pages/selections.html).
Wherever a selection is expected you can write `group:<Name>`, e.g.
`group:Aptamer and name P C4'`. Fields that ask for a group accept its bare name. Amber
topologies number residues 1…N in file order. Defaults are provided for the reference system
(`Aptamer` = 1–12, `Target` = 13–24) — edit them for yours.

**Renaming a group** in the Groups table (confirm the name with Enter or by leaving the cell; Escape
cancels) renames it everywhere the project uses it: the group choices of the analyses, selections that
are just its name or contain `group:<Name>`, *Groups shown*, the groups of 3D representations, the
proximity set, the per-panel group choices and data keys (`contacts@Aptamer+Target/map`), chart
series, the PDB export subset, other groups' selections and the planner steps' settings. A token that
only starts with the name (`group:Aptamers`, `group:Aptamer_2`) is another group and stays; project
notes are never rewritten. Automatic panel titles follow. The status line says how many places
changed (hover it for the list). Empty names, names with spaces or other characters than letters,
digits, `_`, `.` and `-`, and names another group has are refused, and flagged red while you type.
Deleting a group that is still used names the settings that still refer to it; a panel keeps showing
it as *(missing)* until you choose another group. `Project.rename_group(old, new)` does the same from
Python and returns the changed locations; `Project.group_references(name)` lists them.

**Groups from chains.** One group per polymer chain of the topology, water and ions left out: from
the chain IDs when there are several (PDB, or a parm7 written with `RESIDUE_CHAINID`), else from
segment ids that name several chains, else from bonded fragments numbered in order (`Chain_1`,
`Chain_2` …; consecutive residues within covalent distance count as bonded when the file has no
bonds). Groups are named `Chain_<ID>` and get distinct colours. Each group's residue spec is checked
to select exactly its chain's atoms in that topology: plain ranges (`1-44`) when the residue numbers
are unique, chain-qualified ranges (`A:1-44`) when chains repeat them, else a selection (as a last
resort `index …`, which only holds for topologies with the same atom order).

- Choosing a topology on the Input & cleaning page (or taking the plan's files with *Use the plan's
  files*) fills the groups
  **automatically only when nothing you made is overwritten** (opening a project never does): when there
  are no groups, or the two untouched defaults. The defaults are replaced when the topology has at least
  two chains, unless they already select one chain each (as in the demo); every reference to `Aptamer`
  then names the first chain's group and every reference to `Target` the second's, so the analyses keep
  comparing two groups, and further chains are added. With a single chain the defaults stay.
- Otherwise nothing changes by itself; the status line lists the chains without a group and
  **Groups from chains** adds them (a chain that a group already selects exactly is skipped, an existing
  name gets a suffix). Your groups are never changed, and neither are the untouched defaults: the button
  only adds.

**When a group or selection is wrong, the message says where and what.** Every problem starts with
the place of its setting, as the analysis window shows it, and names the offending part:

| Problem | Example message |
|---|---|
| unknown group | `Analyses › Hydrogen bonds › Group B: unknown residue group 'Tagret' (defined: Aptamer, Target); did you mean 'Target'?` |
| MDAnalysis cannot parse it | `Analyses › RMSD › Target 3 (X) › Selection: invalid selection 'group:Aptamer and nme P' (expanded: '((resid 1:12)) and nme P'): MDAnalysis says: Unknown selection token: 'nme' (did you mean 'name'?)` |
| nothing matches (each group, then each and / or part, is tried on its own) | `Analyses › Pairwise contacts › Group B: selection 'group:Target' matched no atoms: group 'Target' (residues '100-112') matches no atoms - the structure's residues are numbered 1-24` |
| a group's own definition | `Residue groups › Group 3 (Loop) › Selection (overrides): unknown residue group 'Aptamr' (defined: Aptamer, Target, Loop); did you mean 'Aptamer'?` |
| a panel's group pair | `Layout & panels › Panel 4 (Contact map) › Groups: …` |

The settings are checked before every run (`mdatk render` refuses with the list; warnings, e.g. a group
in *Groups shown* that does not exist and is left out, are logged); a selection that fails while
analysing (it can only be matched against the structure then) names its setting the same way.
**Only what is about to run can block it**: a wrong setting is an error if the run uses it, and only a
warning otherwise. The structure view's settings (*Groups shown*, the proximity set, the
representations' groups, the Mg selection while the Mg analysis is off) are errors only when the
structure view is rendered: the video is on (and the layout has a structure, sequence or
secondary-structure panel), or *Render still*. `mdatk render --no-video`, `mdatk analyze` with the
video off, `mdatk export-pdb` (and *Export PDBs now*) and the density fit are not refused for them;
the density fit does use *Groups shown* and the Mg selection. The PDB atom subset is an error only when
PDBs are exported; one made only of commas names no group and is an error too (an empty subset exports
every atom). A selection that uses a broken group is not blamed for it: the group's definition is. That
definition is an error when something about to run selects the group (a setting that is an error, the
groups the rendered structure view shows, a panel or chart that is drawn), directly or through other
groups, and a warning for a draft group nothing uses. Circular definitions (`Stem` uses `group:Loop`,
`Loop` uses `group:Stem`) are reported at every group of the circle. A run of an analysis on a panel's
or chart's groups (`rmsd@Loop/Heavy`) that fails is reported at that panel or chart, else at the
group's definition, never at the analysis' own settings. In Python, `Project.validate()` (errors) and
`Project.problems()` (errors and warnings) take what is about to run (`video`, `charts`, `pdb`: default
the output switches; `density`) and return `mdatk.problems.Problem` strings whose text is the message
above and which carry the setting's `field` path (`analyses.hbonds.group_b`), GUI page (`tab`), readable
`location` and offending `token`; `SelectionError` carries the same plus `kind`, the selection as written
and expanded and MDAnalysis' own message.

## Structural view, colouring and glow

- 2D projection of the backbone trace of the selected groups (consecutive residues of the same
  chain are linked), Mg²⁺ ions and a time label, with depth cueing (size, width and fog).
- **Orientation**: drag the preview of the reference frame with the mouse, or use the X/Y/Z
  sliders. The angles (degrees and radians) are stored in the project (`visual.rotation`,
  `R = Rz·Ry·Rx` about the reference-frame centroid) and applied to every frame. Frames are
  superposed onto the reference frame first (*superpose frames*), so the model does not drift.
- **Zoom and centre** (`visual.zoom`, `visual.center_on`, `visual.center_follow`): the mouse wheel
  over the structure panel (or *Zoom* on the Structure view page) magnifies the view; *Centre on* puts a
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
  and in the analysis window.
- The view is framed by the model's bounding sphere, so changing the orientation never zooms.

## 3D representations

Instead of the 2D trace, the structural view can draw residue groups as VMD-style
representations (*Structure view → Representations (3D)*, `visual.representations`). Each row is a
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
  rasteriser that needs no OpenGL, so the same video comes out of a laptop, the analysis window and
  a headless cluster node. It costs roughly 50–80 ms per 1080p frame for cartoon or licorice of a
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
    video it is part of the static background (no per-frame cost). In the analysis window it
    follows rotations, coarser while dragging.

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
computed on demand in the same pass over the trajectory and cached on its own; in the analysis
window the *Groups* drop-downs of a panel build these keys. Group names may contain letters, digits,
`_`, `.` and `-`.

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
- `mdatk render --from-cache` (or *Re-render video + charts from cache* in the analysis window)
  re-renders video and charts from the manifest without touching the trajectory.
- Cleaned trajectories live in `MDATK_cache/cleaned/<key>/` with the generated `cpptraj.in`,
  the full `cpptraj.log` and a completion marker.
- Any project file or `MDATK_params_log.toml` can be passed back with `-p` to reproduce a run.

## RNAdo, MDATK's workflow planner

RNAdo lays the steps from sequence to MD ensemble out on a canvas (the plan), writes their control files
and scripts with consistent names and, on request, runs them. Its script wizard writes the files of one
program at a time; its [PDB clean-up](#pdb-clean-up) and [structure builder](#structure-builder)
prepare the structures that go in. The window is described in [RNAdo's window](#rnados-window);
everything it writes can also be written from the command line (`mdatk workflow`, see [Command
line](#command-line)).

### Starting RNAdo

- `rnado` (an empty plan) or `rnado project.mdatk.toml`; `rnado --open wizard|pdb|builder` opens one of
  its tools as well, and `rnado-gui` is the same without a terminal;
- the analysis window's *RNAdo* menu (see [The analysis window](#the-analysis-window));
- the stand-alone builds: `MDATK-<version>-x86_64.AppImage rnado [PROJECT]` or a link to the AppImage
  named `rnado`; on Windows `rnado-gui.exe` or the Start-menu entry *RNAdo* (see
  [Installation](#installation));
- the applications menu (Linux): the entry *RNAdo*, or the *RNAdo* action of the entry *MDATK*, once
  *Add MDATK and RNAdo to the applications menu…* or `mdatk desktop install` (`rnado
  --install-launchers`) added them (see [Applications menu (Linux)](#applications-menu-linux));
- `mdatk rnado [PROJECT]`, `mdatk gui --rnado [PROJECT]` (or `--planner`; `--open TOOL` alone starts
  RNAdo too), or `python -m mdatk.gui.rnado [PROJECT]` from a source checkout.

`rnado ACTION …` with an action of `mdatk workflow` (`write`, `run`, `status`, `check`, `template`,
`generate`, `tools`, `restraints`) is the same as `mdatk workflow ACTION …`.

### The plan

Place boxes (steps) with *Add step* or the canvas's right-click menu, and connect them by dragging from
an output port (right edge) to an input port (left edge); only compatible files connect, and loops are
refused. A large, stacked dot is a **set**: the full set of N files of one kind, such as *Every decoy* on
the right of a FARFAR2 box, whose files are known only once that step has run (see [Every decoy from the
FARFAR2 step](#farfar2-decoys-scores-run-audit-and-frames)). A set connects only to inputs drawn as large
dots, on the left of the steps that take many models (the models of *Model check*, *Decoy interfaces*,
*Register census*, *Ensemble comparison* (both arms), *MDATK · frame check* and *Map-model score*); they
take files connected one by one as well. Dropped on an input that takes a decoy table, a set connects its
step's decoy table instead; connections of a set are drawn thicker, and an output that needs an option of
its step (here *Extract every decoy after the runs*) switches it on when connected. Double-click a box to
edit it in the wizard. The details pane of the selected box has *Edit in
wizard…*, *Write step*, *Run step…*, the previous-run buttons (see [Previous runs](#previous-runs)) and,
on an MDATK analysis box, *Use for analysis…*. *Templates* insert pre-connected chains (FARFAR2 → models
→ Amber build → helix restraints → MD → MDATK analysis + VMD; Amber protocol; Amber → GROMACS; ideal
helix → Amber build → restraints → MD; model + map → restraint dictionaries → Phenix RSR → Coot RSR;
model + map → Amber build → NAMD MDFF; and *Decoy validation (FARFAR2)*, the order for a decoy campaign
against a map: FARFAR2 → models (closure filter before clustering) → model check (clashes) → decoy
interfaces (segment support) → map-model score (Q-score, interface vs scaffold dQ), with notes on the
validity filters, the rule "filter on validity before clustering; filter on quality after, when choosing
representatives" and the kink test still to come). *Write all files* writes every step, a master script
that runs the steps in order, and a Markdown summary of the plan.

Each step box has a check box in its title bar (also Space on the selected box, or the right-click
menu; *Tick all steps* / *Untick all steps* on the canvas): *Run selected steps (N)* writes the ticked
steps' files and runs their main scripts in plan order, whatever order you ticked them in. Steps they
read from are not written or run again, so their outputs must exist already. The ticks are not saved
with the project.

The plan is saved in the project file (`workflow` section), so one file holds the plan and the analysis
settings. The analysis window's *Use the plan's files* (Input & cleaning, also in its *RNAdo* menu) puts
the topology and trajectories an MDATK analysis box reads into Input & cleaning, from the plan saved in
the project file (*Use for analysis…* on the box in RNAdo saves the plan for it); with several analysis
boxes it asks which. `examples/rna_plan.mdatk.toml` is a complete example plan (open it in RNAdo and set
the output folder).

### Script wizard

*Script wizard* in RNAdo's header bar (Ctrl+Shift+W) writes the files of one step on its own; a box's
*Edit in wizard…* opens the same form for that box. Pick a program on the left, fill in its options
(check boxes, radio buttons, text fields; every option has a tooltip), choose input files, and *Write
files*, or *Write and run…*. The options are the RNA-relevant ones for each program plus a free-text box
for anything else. Logic checks run on every change: **errors** (conflicting or impossible settings)
block writing, **warnings** flag likely mistakes and **infos** point out redundant or noteworthy
choices; the offending fields are highlighted and clicking a message jumps to the field. A live preview
shows every file that will be written. The **Commands** tab (next to *Files*) shows what the step will
run, from the settings as they are and updated with the checks: the program and helper calls of each run
script under the script's name, without the set-up (conda, modules, `cd`), checks, messages and error
handling, with continued lines joined, here-documents kept and a variable the script sets once shown with
its value (`pmemd.cuda -O -i …` rather than `$PMEMD -O -i …`); a script run separately (production) is
marked as such. *Whole scripts* shows the scripts as written, *Copy* copies what is shown, and a step with
errors lists its errors there instead. On the command line, `mdatk workflow generate` is the wizard.

### Running steps

The plan writes files and scripts; nothing is executed unless you ask for it with *Run all…* /
*Run selected steps* (planner toolbar), *Run step…* (a box's details) or *Write and run…* (wizard).
RNAdo then asks for confirmation and runs each step's main script with `bash`. Each step starts once the
steps it reads from (through the planner's connections) have finished and the CPU cores and GPUs it
uses are free; otherwise steps go in plan order.

- **CPU cores and GPUs** (naming bar, saved with the project): the scripts that run at the same time
  share two budgets, *CPU cores* (default *auto*: half of this computer's cores) and *GPUs* (default
  *auto*: the GPUs `nvidia-smi -L` lists, or those in RNAdo's own `CUDA_VISIBLE_DEVICES`). Each box
  shows what one run of its step uses, e.g. `8c+1g`, and its details say it in words:

  | Step | Uses |
  |---|---|
  | Amber MD stages | pmemd.cuda: 1 core and 1 GPU (its *GPU* option names it); pmemd.MPI: its *MPI processes*; pmemd: 1 core |
  | NAMD MDFF | its *CPU cores* (`+p`) and 1 GPU (`+devices` names it) |
  | GROMACS | its OpenMP threads (`-ntomp`, times `-ntmpi` among the extra mdrun flags, or `-nt` there; without `-ntomp` every core, as mdrun takes them all, so no other step runs beside it) and 1 GPU when the work is offloaded |
  | FARFAR2, SimRNA | their parallel runs (at most the number of runs); FARFAR2 with *Extract every decoy* its parallel extractions when there are more |
  | FARFAR2 models | its parallel extractions when every model is extracted, else 1 core |
  | Phenix RSR | its processors (`nproc`) |
  | Register census, frame check | their worker processes |
  | Custom script | its *Cores and GPUs* section (*CPU cores*, 0: all of them; *GPUs*) |
  | every other step (Amber build, model check, map-model score with ChimeraX, ...) | 1 core |

  A step that uses a GPU gets a free one in `CUDA_VISIBLE_DEVICES` (the start line in the log says
  which, and how many cores the step got; the step log records it in the command), unless the step
  names its own: the Amber MD step's *GPU*, NAMD's `+devices` or `CUDA_VISIBLE_DEVICES` in the step's
  *Environment* setup commands; it then runs when no other step uses that GPU. A later step that fits
  may start while an earlier one waits for room, except behind a step that needs every core: that one
  starts once nothing runs, before the steps after it. A step larger than the whole budget runs alone
  the same way, with a warning; with no GPU found (and none set), the steps that use a GPU run one at a
  time, with a warning. *At once* caps the number of scripts (default *no limit*: the budgets decide);
  projects saved before 0.0.12 keep their earlier *Parallel* value there (1: one script at a time, as
  before). On the command line: `mdatk workflow run --cores N --gpus N [--parallel N]`.
- **Output:** every line goes to the [output log panel](#rnados-window), marked with its step, and
  every run is recorded in the project file.
- **Failures:** a failed script stops every step that depends on it; independent steps carry on.
- **Stop scripts** (planner toolbar, red while scripts run; also *Cancel* in the status bar) stops
  every running script.
- **Stopping one step:** the small square stop button in the title bar of a running box (or *Stop this
  step* in its right-click menu, also for a step that has not started) stops only that step's script
  and the programs it started (its process group). The log says so, the box shows a square, the steps
  that read from it are not started (`not started (step 03 was stopped)`) and the other scripts carry
  on. The step log records the run as *stopped*, and the run leaves a note beside the step's files
  (`<project>_<step>_stopped_<time>.txt`), so *Check previous runs* calls it **partial** until the step
  runs again (or a file of the run is written after the note).
- **Running markers:** a box whose script is running shows a turning arc in its title bar, and a
  waiting box a dashed ring. When a script ends, its box shows a tick (done), a cross (failed), two
  bars (waiting for a separately run script, see below), a square (stopped) or a dash (cancelled / not
  started) until the next run.
- **Windows:** the generated scripts are bash scripts for Linux; RNAdo on Windows writes them but does
  not run them (*Run* says so), see [Installation](#installation).
- **Writing again:** *Run all…*, *Run selected steps* and *Run step…* write the steps first, so a step
  that has a completed run is moved aside before it runs again (see [Previous runs](#previous-runs)).

Scripts marked *run separately* (production runs meant for a cluster queue: *Separate production
script* of the Amber MD and GROMACS steps) are not started unless you tick *Also run the scripts marked
'run separately'* in the confirmation (or give `--include-separate`); they then run right after their
step's main script. Such a step is complete only when its separate script has finished, and its main
script checks that:

- it runs its own stages before production (equilibration; for GROMACS also the conversion) only while
  the separate script has not finished - once it has, they are not redone, as production started from
  them;
- while production has not finished, it then exits with status **75**, naming the script to run
  (`WAITING: the production stages have not finished. Run them … with: bash …/<name>_amber-prod_….sh`).
  MDATK reports such a step as **waiting**, not as failed, and does not start the steps that read
  from it; `mdatk workflow run` exits with 75 too. Run the separate script (e.g. on the cluster), then
  the step again;
- once production has finished, the stages after it (e.g. a cooling stage) and the dry PDB run, and the
  step succeeds.

Main scripts written before 0.0.10 do not make this check; MDATK treats such a step as waiting and
asks you to write it again.

### File names

Every file is named `<project>_<step>_<tool>_<time>.<ext>`, e.g. `apt1_03_tleap_20261015-1432.parm7`:
the project name, the step number of the box, the program or stage that writes the file, and one
time stamp per plan. The stamp is fixed the first time files are written (and stored in the project),
so later re-generations keep the same names; *New stamp* starts a new set, and the stamp can be
switched off. Each step writes into its own folder (`03_amber-build/`), and scripts refer to the files
of other steps by relative paths (`../03_amber-build/apt1_03_parmed_….parm7`), so a whole plan folder
can be copied to a cluster. Every script starts with `cd "$(dirname "$0")"`, so it can be started from
anywhere. Each script's header cites what that step uses (see [Citations](#citations)).

### Previous runs

Each iteration of a project may get a new time stamp (*New stamp*), and the stamp is part of every file
name, so after an update the plan no longer sees the outputs of its earlier runs. *Check previous runs*
(planner toolbar; for one box or the ticked boxes in the right-click menu and the details pane) looks
in each step's folder for the files the step's outputs name: under the plan's stamp first, otherwise
under the newest earlier stamp with all of them. Only the files the step's scripts make count; for
Amber MD every stage's output must also end with pmemd's closing lines, so a run still in progress is
*partial*, and a custom script's run counts only with the marker its run script writes last (see
[Custom scripts](#custom-scripts)). Each box then shows a footer: **done** with the stamp and the time
its files were last written, **partial** with how many files are there (the details list the missing
ones), or **missing**.

A step with a complete run can **use it** (*Use this run* in the details, the right-click menu, or the
offer after a check): the project file stores the run's stamp in the step (`adopted`), and the planner
points the step's outputs at that run's files. The steps after it then read existing files, so the
checks that look into their inputs run, and the scripts they write refer to the right files; the step
itself is not written or run again (the master script lists it as used). *Run selected steps* / *Run
all…* ask whether to skip such steps or run them again; *Stop using it* returns a step to normal. If a
file of the run disappears, the planner warns. On the command line:

```bash
mdatk workflow status -p plan.mdatk.toml              # done / partial / missing for every step
mdatk workflow status -p plan.mdatk.toml --adopt      # use every complete run found (saved)
mdatk workflow status -p plan.mdatk.toml --forget --step 3   # step 3 is written and run again
```

`workflow run` skips steps that use an earlier run and says so.

**Completed runs move aside before a step is written again.** Writing a step again under the same stamp
would replace the scripts and inputs its last run was made with, and running it again that run's outputs.
So when a step is about to be written (*Write all files*, *Write step*, the wizard's *Write files*, and
every *Run* action, which writes first) and its folder holds a **complete** run of the plan's current
stamp (*done*, as above), that run first moves into `<step folder>/superseded/<stamp>_<YYYYMMDD-HHMMSS>/`,
even if the step is never run again. The scripts go with it, as the step may be written with changed
settings, so the outputs keep the scripts that made them and the run stays repeatable. What moves, only
from the step's own folder and only of that stamp:

- the step's outputs, and every file the step writes for the stamp (scripts, inputs, helpers);
- its logs: the files a step type checks for closing lines and every other file named
  `<project>_<NN>_*_<stamp>.*` in the folder;
- outputs whose names carry no stamp (a custom script's fixed file names, plans without time stamps,
  which use `superseded/no-stamp_<time>/`) with the completion marker that ties them to the run;
- the stamp-free folders a run fills: `runs/` of FARFAR2 and SimRNA, `models/` of FARFAR2 (with *Extract
  every decoy after the runs*) and of the FARFAR2 models step (with *Extract every model* or validity
  filters), moved whole so `models/r01/S_000001.pdb` keeps its place; likewise the frame check's folder of
  moved decoys, `<project>_<NN>_reframed_<stamp>/`.

A `manifest.toml` in the run's folder says what moved, from which step and stamp, when and why; to use the
run again, move its files back. Nothing else moves: partial runs stay where they are (resuming relies on
them), steps that use an earlier run are never written, files of other stamps and your own files stay, and
nothing in `superseded/` is ever replaced (a second move in the same second gets `…-2`). In a plan without
step folders a step's runs go to `superseded/<NN>_<tool>/` in the plan folder, the `<project>_<NN>_*` rule
applies only when no other step has the number, and `runs/` (steps share it) and the extracted models
(`<project>_<NN>_models/`) are left alone.
A step's move is all or nothing: if a file cannot be moved, the files moved so far go back, the step is not
written and the write reports it. *Check previous runs* never looks inside `superseded/`. Steps that skip
finished stages when run again (Amber MD *Skip finished stages*, NAMD MDFF *Skip finished runs*) reuse
them from a run left in place, i.e. a partial run, or any run with the option below off; a complete run
moves aside, so the step runs again from the start.

RNAdo asks first: *Write all files*, *Write step*, the *Run* actions and the wizard list the steps whose
completed runs would move and where (*Cancel*, or *Move aside and write*); the output log then lists what
moved, and a saved project records each move in its step log (`workflow supersede`). While a step that
reads those outputs runs or waits to run (or the wizard's run, which RNAdo does not follow), RNAdo refuses
such a write: write again once the scripts have finished. To keep a run in place, let the step use it
instead (*Use this run*). The naming bar's *Move completed runs aside* (project
option `workflow.supersede_runs`, on by default) turns moving off: writing then replaces the scripts, and
the next run the outputs, as before 0.0.12. On the command line `workflow write` (also with `--run`) and
`workflow generate` print the runs they moved on standard error and under `superseded` in their JSON;
`workflow run` only runs scripts already written, so it moves nothing:

```bash
mdatk workflow write -p plan.mdatk.toml --dry-run   # what would be written, and which runs would move
mdatk workflow write -p plan.mdatk.toml             # moved    03 · Amber · MD stages: moved its …
```

### Step types

| Step | Writes | Notes |
|---|---|---|
| **Rosetta · FARFAR2** | FASTA + secstruct from a sequence and dot-bracket, a flags file, a script running *N* seeds in parallel (`xargs -P`), one silent file per run; optionally, after the runs, every decoy (`models/rNN/<tag>.pdb`), the decoy table and the list of the decoys | fixed helices (`-s`, `-input_res` typed or all helix residues), cryo-EM map (`-edensity:*`), score weights, constant seeds; checks that paired bases are complementary (rna_denovo stops otherwise); output *Every decoy* (a large dot: the full set for the steps that take many models) and *Decoy table*, see [FARFAR2 decoys](#farfar2-decoys-scores-run-audit-and-frames) |
| **Rosetta · FARFAR2 models** | script: `rna_cluster` → `extract_pdbs` → `x3dna-dssr`; optionally validity filters before clustering (hands, base identity, closure, required pairs), a quality filter after it (score limits on the ranked centres), every model of every run (`models/rNN/<tag>.pdb`) and the decoy table | ranked models `…_model-01_….pdb` (best first) feed the Amber build; see [FARFAR2 decoys](#farfar2-decoys-scores-run-audit-and-frames) |
| **Model check (clashes · backbone · chirality)** | a script running MDATK's PDB tools and model checks (copied into the step folder, standard library only) on every connected model: a report table `…_report_….tsv`, the passed models and the best one | report, filter or stop; see [Model checks](#model-checks) |
| **Rosetta · run audit** | script calling `python3 -m mdatk rosetta audit` | what each rna_denovo run was given (logged command lines, flags files, input fingerprints), the Rosetta build and the seed of every run (duplicate seeds are an error), conditions compared in pairs; see [FARFAR2 decoys](#farfar2-decoys-scores-run-audit-and-frames) |
| **MDATK · frame check** | script calling `python3 -m mdatk frame check` (and `reframe`) | fit one body, measure the others: one frame or sampled placements; models moved into a re-fitted reference's frame when one rigid transform suffices |
| **SimRNA** | `config.dat`, sequence, secondary-structure (pseudoknots on extra lines) and restraint files, parallel replica-exchange runs, clustering + all-atom rebuild (`clustering`, `SimRNA_trafl2pdbs … AA`) | links SimRNA's `data/` folder into every run folder |
| **RNAMake** | script for `design_rna_scaffold` (current, `--options`) or the legacy `design_rna` (`-options`): motif-graph search, sequence optimisation, thermal-fluctuation scoring | needs `RNAMAKE`; current RNAMake has no separate builder / optimiser programs |
| **Amber · system build** | tleap inputs (force field, water, box, ions, extra frcmod / libraries / leap commands), ParmEd inputs, helpers (PDB preparation, L-RNA naming and name check, CP library, TIP4P/2005 files) and the build script | see below |
| **Amber · helix restraints** | DISANG file of the hydrogen bonds of every base pair (G–C ×3, A–U ×2, optional G·U), found by residue number and atom name in the final topology | written directly when the topology exists, otherwise by a small Python script (no ParmEd needed) the step runs |
| **Amber · MD stages** | one mdin per stage and the run scripts | see below |
| **GROMACS · MD** | ParmEd (or acpype) conversion, RNA position restraints, mdp files, grompp / mdrun scripts | 2024+ (workstation) or 2023.3 (cluster module `GROMACS/2023.3-CUDA-11.8.0`) |
| **VMD** | Tcl script + start script | one representation per residue group in the MDATK colours (chain groups keep to their chain), bases as licorice, Mg²⁺ spheres, frame alignment, smoothing, snapshot or movie (Tachyon + ffmpeg) |
| **Structure builder (MDATK)** | a script calling `python3 -m mdatk build` (the step's *Python*; from the Linux AppImage, the AppImage itself): ideal A-form helix (D or L), base mutations with compensatory partners, or a rigid-body fit into a map | see [Structure builder](#structure-builder) |
| **Restraint dictionaries · acedrg / eLBOW** | per residue code a SMILES file, the acedrg or phenix.elbow call and MDATK's renaming to the model's atom names | L-RNA 5'/3' variants (0A5, 0G3 …) for Coot and Phenix; see [Restraint dictionaries](#restraint-dictionaries-for-l-rna-coot--phenix) |
| **Phenix · real-space refinement** | `phenix.real_space_refine` script (1.21 / 2.x), optional base-pair `.eff` from a dot-bracket or the model's REMARK 950 annotation | see [Real-space refinement](#real-space-refinement-phenix--coot) |
| **Coot · real-space refinement (batch)** | headless Coot 0.9 / 1.x script refining each chain in overlapping windows | see [Real-space refinement](#real-space-refinement-phenix--coot) |
| **NAMD · MDFF (Amber topology)** | NAMD configurations per stage, the sander / NAMD energy comparison, a preparation helper and the run script | NAMD reads the Amber topology natively; see [MDFF with NAMD](#mdff-with-namd-on-the-amber-topology-namd-mdff) |
| **Register census (MDATK)** | a script calling `python3 -m mdatk ensemble census` | which register (named set of defining base pairs) each decoy formed or approached, by folder and build condition; see [Decoy ensembles](#decoy-ensembles-map-model-scores-and-statistics) |
| **Ensemble comparison (MDATK)** | a script calling `mdatk ensemble compare` | two ensembles: closure, defining pairs, map occupancy, exact statistics and the detectable difference (modes compare / sweep / pilot) |
| **Map-model score (Q-score / CC)** | a script calling `mdatk map score` | per-residue Q-score (ChimeraX's `qscore`, on the models as they are) or MDATK's per-residue CC, and the models ranked by the dQ of named regions against a reference (scaffold) |
| **Body hybrids** | a script calling `mdatk pdb hybrids` | every old/new combination of up to six re-fitted bodies, optionally scored, with each body's main effect |
| **Custom script** | your bash, Python or other script (typed in, or copied from a file) and a run script that passes it the file names and checks its outputs | input and output files you declare; see [Custom scripts](#custom-scripts) |
| **Decoy interfaces (MDATK)** | a script calling `python3 -m mdatk decoys census / clusters / support` (from the Linux AppImage, the AppImage itself) | interface census of an ensemble between two sides, clusters of the contact fingerprints with medoid representatives (PDB outputs), support of rebuilt segments; see [Decoy interfaces](#decoy-interfaces) |
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
a separate production script for the cluster (the main script then waits for it, see above); resuming
(finished stages are skipped); a dry PDB and a
dry, imaged production trajectory. The engine is pmemd,
pmemd.cuda or pmemd.MPI (with an MPI launcher such as `mpirun -np {n}` or `srun`); Amber 24 and 26 use
the same input, the version is recorded in the scripts. The *Environment* section adds
`conda activate` / `module load` / `source amber.sh` lines.

**Input structure.** *Remove Rosetta virtual residues (VRT/XXX)* (on, part of *Prepare the PDB for
tleap*) drops the virtual residues of density-scored Rosetta models, which tleap cannot type. *Check
O3′–P links before tleap* (off by default; warn or stop) measures O3′(i)–P(i+1) of consecutive
nucleotides in every TER segment of the PDB tleap reads (after the preparation and the L-RNA naming)
with MDATK's model checks, copied into the step folder. tleap bonds them by name whatever the distance,
so a chain break left in a model (Rosetta numbers continuously across its cutpoints) becomes a bond
that the first minimisation has to relax, with a strain E = k(r − r0)² (Amber's k, no ½). Up to
*Longest unstrained O3′–P link* (2.0 Å) a link is a bond; from there to *Longest O3′–P link before a
break* (3.0 Å) it is listed with its strain, a warning; longer it is a break (as are links shorter than
1.4 Å or unmeasurable), and *Stop* ends the script before tleap. k and r0 come from the force field the
first tleap input loads: the check follows its `source` and `loadamberparams` lines into
`$AMBERHOME/dat/leap` (leaprc → parm10.dat) and the frcmods, a later term replacing an earlier one, O3′
typed OR for OL3 + CP, else OS; parm10.dat's OS–P is 230 kcal/mol/Å² and 1.61 Å (about 35 kcal/mol at
2.0 Å, 180 at 2.5 Å, 440 at 3.0 Å), unchanged by the frcmods of OL3, LJbb, ROC, YIL and DESRES and
copied for OR by the CP helper, and MDATK's default when no file can be read (the report names the
source). With the check on, the PDB is prepared, named and checked first, before the CP, L-RNA library
and TIP4P/2005 helpers and before every tleap run (the dry one of a two-pass build included). With an
existing input PDB the planner measures the links when the plan is written. A structure from a model
check's *Passed models* may be an empty slot (see [Model checks](#model-checks)): the script then stops
at its first line, before any helper or tleap, with the slot's message.

**L-RNA.** *Name L-nucleotides for leap* (on by default) runs MDATK's PDB tools, copied into the step
folder, on the starting structure before tleap: residues whose sugar chirality (C1′, C3′, C4′) is L
get the Dudek & Trylska names (LA LC LG LU, LA5/LA3 … at chain ends), the others Amber names; *Force L
for* (e.g. `C:45-73`) overrides the detection, and *5′/3′ terminal names* can be switched off for
libraries that map chain ends with `addPdbResMap` (plain LG, as `to_lrna.sh` wrote). This replaces
`to_lrna.sh`.

*L-RNA library* decides where the L units and their parameters come from:

- **Generated by MDATK.** The build script writes `LRNA.lib` and `frcmod.lrna` from the force field in
  use (OL3, or OL3 + CP with the retyped library or your own CP frcmod) with a small standard-library
  helper. An L-nucleotide is the mirror image of a D-nucleotide, and every Amber term is unchanged by
  mirroring except a dihedral or improper whose phase is not 0° or 180°: that one needs the phase −γ.
  In OL3 every such term contains O5′ or O4′ (α O3′-P-O5′-C5′, γ O5′-C5′-C4′-C3′, χ O4′-C1′-N-C).
  So, as in Dudek & Trylska's scheme, the L units give these two oxygens their own types: LD for a
  bridging O5′, LH for a 5′-OH, and, with CP, LE for O4′ (OR and OS are different types there).
  `frcmod.lrna` repeats every term of their D twins for the new types (masses, Lennard-Jones, bonds,
  angles, and dihedrals and impropers with mirrored phases). `LRNA.lib` holds the D units mirrored
  (tleap builds missing hydrogens from these templates) and named the Dudek way inside and out: LA LC
  LG LU, LA5 / LA3 / LAN …. Because the names live inside the units, ambpdb writes L names back out
  and a two-pass build keeps the L units. The tests check what must hold: built with tleap, a D duplex
  and its mirror image have the same sander energy, term by term, for OL3 and OL3 + CP. Without the
  mirrored phases the dihedral energy of an 8-bp duplex is off by about 3 kcal/mol.
- **My own files.** The libraries and parameters listed under *Force field* (Dudek & Trylska's
  `nucleic12_LRNA.lib` + `parm10_LRNA.dat`, or your own `LRNA.lib` + `frcmod.LRNA`).
- **Automatic** (default). Generated when the structure is known to have L residues when the plan is
  made (an existing input PDB, *Force L for*, or an L helix straight from the builder), the force field
  is OL3 or OL3 + CP, and no own L-RNA files are listed or loaded (extra libraries, a parameter file
  defining LD / LH / LE, or load commands in the leap fields); own files otherwise. For a structure
  that an earlier step makes (mutation, fitting, FARFAR2 …), choose *Generated by MDATK*.

The generator refuses a parameter set with a dihedral or improper whose phase is not 0° / 180° and
contains none of the twinned types (e.g. a 2′-OH torsion from an extra frcmod): the generated library
could not make such a molecule's mirror image equal in energy. With CP from an own frcmod and an own
CP residue library, use your own L-RNA files too (the generated units mirror AmberTools' `RNA.lib`).

When the input PDB exists, the build step checks it and reports L residues without a library as an
error.

> **L residue names must match the library.** MDATK writes the Dudek names (LA LC LG LU, LA5/LA3 …).
> A library whose units have other names (e.g. A0/C0/G0/U0 with A05/A03 terminals) needs
> `addPdbResMap` lines under *Leap commands before loadpdb*. Without them tleap invents
> parameterless units for LC and LG, and, without any warning, builds LA and LU as the lanthanum
> and lutetium ions of AmberTools' `atomic_ions.lib`; the atoms it cannot place land near the origin,
> and the solvent box grows to fill the gap (in one case 13 times the volume and an hour of tleap).
> The planner reads the library's unit names and the map, reports every unmapped L name as an
> error, and suggests the map lines; when an earlier step makes the structure, it checks every name
> MDATK may write (LA LC LG LU, with *5′/3′ terminal names* also LA5 / LA3 …) and warns. The build
> script also checks before the first tleap, for builds with own libraries (listed under
> *Extra residue libraries* or loaded with `loadOff` in the leap commands): a small standard-library
> helper reads the L names of the renamed PDB, the units of every library the tleap input loads
> before `loadpdb` (a name in leap's search path is looked up in `$AMBERHOME/dat/leap`) and its
> `addPdbResMap` lines, and stops the script on a name that is neither a unit nor mapped. A library
> it cannot read gives a warning and the build goes on. The topology check (below) is the last net.
> The warning for a structure from an earlier step and the check in the build script come with an
> L-RNA set-up only: *My own files* chosen under *L-RNA library*, a structure known to have L
> residues (*Force L for*, an L helix, an input PDB with L sugars), or an own library that looks like
> an L-RNA one (it uses the atom types LD / LH / LE, or has nucleotide units named the L-RNA ways:
> LA…, A0…, 0A…, with their 5′ / 3′ / N forms; D libraries of modified nucleotides such as
> `all_modrna08.lib` do not count). A D build with, say, a ligand library is written as before.

The optional
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
  - with microMg / nanoMg, a solute atom type with Lennard-Jones terms has no explicit Mg²⁺ pair term;
  - a residue named as a nucleotide (A, G5, LA, LU3 …) has fewer than 10 atoms: tleap matched the name to
    another unit, such as the lanthanum ion LA (also for dry L-RNA builds).
  - with the generated L-RNA library, the number of topology residues carrying LD / LH / LE differs from
    the number of L residues in the PDB that the final tleap read: D units were loaded for L residues,
    and the L strand would be simulated as D.

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

### FARFAR2 decoys: scores, run audit and frames

**Scores by column name.** Every silent file has header lines `SCORE: score … linear_chainbreak …
description`; MDATK looks each term up by its name in the header that applies, never by a fixed column,
which points at another term as soon as the score function emits one term more or less.

**Decoy table.** The FARFAR2 models step writes `<project>_<step>_scores_<time>.tsv` (output *scores*,
file kind *table*) with *Write the decoy table*, and always with a filter or *Extract every model*. Its
columns are `run`, `tag`, `pdb` and then every SCORE term by name (with validity filters also their columns,
below); one row per decoy of every run:

```text
run  tag       pdb                                         score    …  linear_chainbreak  …
r01  S_000001  02_farfar2-models/models/r01/S_000001.pdb  -412.31     0.074
r02  S_000001  02_farfar2-models/models/r02/S_000001.pdb  -398.85     6.210
```

- `run` is `rNN` from the silent file's name (`…_farfar2-rNN_<time>.out`; with two FARFAR2 steps feeding
  one models step `01-r01`, `05-r01` …), so the tags that every run reuses never collide.
- `pdb` is relative to the plan folder, empty when the decoy was not extracted; ranked models of an
  unclustered ranking fill it too.
- Steps that take a decoy table (input *scores*) and `--scores TSV` on the command line read terms by name
  and accept extra columns (`mdatk.rosetta.read_decoy_table`).

**Filtering order: validity before clustering, quality after it.** A register census is a statement about
cluster populations, so the populations must be those of every decoy that is a model of the construct at
all. A score cut before clustering reshapes them by a quantity that need not track quality (the best-scoring
decoys can be the clashing ones), so the step separates the two:

- *Validity filters (before clustering)*, judged on every decoy and counted in this order:
  1. *Hands*: every sugar's hand from its geometry (C4′, C3′, C1′; `mdatk.pdbtools.residue_hands`) against
     expected counts (`44 D / 29 L`) and / or ranges (*Residues expected L* `C:*`, *… D* `A:*, B:*`,
     `group:NAME`); a residue of the other hand, an epimerised or flat sugar and, with counts, any other
     count or an unmeasurable sugar make a decoy invalid.
  2. *Base identity against the sequence*: every nucleotide's base (from its atoms) at its chain and number
     must be the sequence's (the connected FARFAR2 step's, numbered by the `A:1-16` tags of its *Existing
     FASTA* when it uses one, or a typed one: FASTA with `A:1-16` tags or `A: GGGC…`); a missing or extra
     residue is invalid too.
  3. *Chain closure*: a score term by name (`linear_chainbreak <= 5` by default), every O3′–P link of
     consecutive nucleotides at most 2.0 Å (`mdatk.modelcheck.check_backbone`), or both.
  4. *Required pairs*: the defining pairs of a register (`pres=B:22-B:37, B:23-B:36, B:24-B:35`, as
     `mdatk ensemble census` reads it), each formed (Watson-Crick N1/N3 < 3.5 Å) or, if chosen, approached
     (N1–N3 < 6 Å with C1′–C1′ 8–13 Å) - the distances of `mdatk.ensemble`.

  They need every decoy as a PDB, so they extract every model of every run (below). A helper written into the
  step folder (`…_validity_….py`: MDATK's PDB tools, model checks and silent-file scores, standard library
  only) judges each decoy and writes `…_validity_….tsv` (a row per decoy: `valid` 1 / 0, `invalid_by` - the
  first criterion it fails - `invalid_reason`, and per criterion its columns: `hands_D`, `hands_L`,
  `hands_undetermined`, `hands_ok`, `base_errors`, `bases_ok`, `o3p_max`, `closure_ok`, `pairs_formed`,
  `pairs_approached`, `pairs_ok`) and the report `…_validity_….txt` (output *validity*):

  ```text
    decoys                12
    after hands           10   (2 fail)
    after bases            8   (2 fail)
    after closure          6   (2 fail)
    after pairs            4   (2 fail)
  ```

  with every criterion also counted on its own, an example reason each and the valid decoys of each run.
  The valid decoys of each run are written into `…_filtered-rNN_….out` (every line of a structure ends with
  its tag, so the file keeps whole structures), and only those are clustered (or ranked). The decoy table
  gets the validity columns, so every decoy keeps its row and its reason.
- *Quality filters (after clustering, choosing representatives)*: *Score limits* such as
  `score <= -300, elec_dens_fast <= -40` (`<=`, `<`, `>=`, `>`; terms by column name) skip the ranked
  cluster centres outside them when the representatives are extracted. None by default. *Before clustering
  (changes cluster populations)* is still offered, with a planner warning; the step then clusters only the
  (valid) decoys within the limits. A term missing from a SCORE header stops the step with the header's
  columns. (Plans of earlier 0.0.12 builds: their score filter becomes a limit `TERM <= MAX`.)

No filter is on by default, and the step's files are then those of 0.0.11.

**Extract every model.** *Extract every model of every run* (implied by the validity filters) writes
`models/<run>/<tag>.pdb` (in a plan without step folders `<project>_<NN>_models/<run>/<tag>.pdb`, a folder
per step, so two steps never extract into one). `extract_pdbs` writes `<tag>.pdb` into the folder it runs
in, so each run is extracted in a `mktemp` folder of its own (*Parallel extractions* at a time), the run's
earlier models are removed first, and afterwards every run must have exactly one PDB per SCORE line: a
missing decoy stops the step and is named. A new run of the step replaces `models/`.

**Every decoy from the FARFAR2 step.** The FARFAR2 box has two more outputs: *Every decoy (PDB)*, the large
dot on its right (a set, see [The plan](#the-plan)), and *Decoy table*. Connecting either switches on
*Extract every decoy after the runs* (section *Decoys*; it can be ticked without a connection too). After
the rna_denovo runs, the step's script runs `<project>_<step>_decoys_<time>.sh`, which extracts every
model of every run into `models/<run>/<tag>.pdb` of the FARFAR2 folder (`<project>_<NN>_models/` without
step folders) as *Extract every model* does
(*Parallel extractions*, `extract_pdbs` from the box's *extract_pdbs executable* with its Rosetta database,
one PDB per SCORE line checked) and writes the decoy table `<project>_<step>_scores_<time>.tsv` (columns as
above) and the list of the decoys `<project>_<step>_decoys_<time>.txt` (one PDB per line, relative to the
plan folder, in the table's order). Run that script alone to extract again without running rna_denovo. The
decoys are known only when the runs have finished (`-autofilter`, the chain-closure filters and failed runs
leave fewer than runs × models per run), so the set passes its list: the steps that take it (model check,
decoy interfaces, register census, ensemble comparison, frame check, map-model score) read the list when
their scripts run, beside any files connected one by one, and stop when it is missing or empty; what they
plan by count (the model check's passed models, the frame check's moved models) assumes at most runs ×
models per run, and the numbered files no decoy fills are written as empty slots (no atoms), so the step
writes every file it promised. The decoys come straight from the runs: the cluster-centre models of
*FARFAR2 models* are another output of another step. A register census groups decoys by folder: with *Group by folder level* 1
each run (`models/rNN`) is a group, with 2 every run of the FARFAR2 step together; its *Rosetta log
folders* take the FARFAR2 step's folder, where the runs' logs are. The set also carries FARFAR2's
`-input_res` (the residues of the fixed-helix PDB; none without one), which the segment support of *Decoy
interfaces* reads. Without the option the step's files are unchanged.

**Run audit** (step *Rosetta · run audit*, `mdatk rosetta audit`). The authoritative record of what a run was
given is the command line Rosetta logs (`core.init: command: …`). MDATK reads every such line of every
rna_denovo log. A fully expanded command line is the normal case; `@flags` files are expanded recursively too
(MDATK's FARFAR2 step passes its flags that way; `#` starts a comment). MDATK reads `::` as `:`, and treats
short and long spellings as one flag (`-s` = `-in:file:s`, `-obligate_pair_explicit` =
`-rna:denovo:obligate_pair_explicit`; a repeated flag counts with its last value). The input files the flags
name (`-fasta`, `-secstruct_file`, `-s`, `-native`, `-edensity:mapfile`) are fingerprinted by their whole
content, since two runs can name the same path and still have read different files. A condition is a folder of
runs (a FARFAR2 step folder, or an experiment folder of a campaign run outside RNAdo); its runs must agree,
and the report lists what differs between conditions. The completed runs moved aside into a step folder's
`superseded/` are never read (give such a run's own folder to audit it). *Pairs* (`--pair CONTROL=TREATMENT`, or `--pair-suffix`)
should differ only in the pairing: `-secstruct_file`, `-obligate_pair(_explicit)`, `-filter_lores_base_pairs`,
the secondary structure itself and what *Controlled residues* derive (`--allow` adds flags); seeds, output
names, `-nstruct` and `-database` are ignored in the comparison. Any other difference makes the pair
*confounded*. With *Controlled residues* (`--controlled A:22-24,A:35-37`, numbered as in the FASTA) the report
says how each condition gives their pairing: paired in the secondary structure, as obligate pairs, or fixed in
the `-s` structure or `-input_res` (chain by chain) - a pairing imposed through `-obligate_pair_explicit`
while the secondary structure leaves the residues unpaired is noted. The output is a text report and a table
with one row per flag and one column per condition.

*Which Rosetta.* The flags cannot say which build made a run (a custom L-RNA build with its own step library
has the flags of any other). The provenance is the block Rosetta writes before the command line:
`core.init: Rosetta version: <package> r<revision> <version> <commit> <repository> <date>`,
`core.init: Rosetta extras: [...]`, the block that `-version` writes instead, and other `core.init` lines
naming a commit, checkout or branch. They are kept raw and parsed tolerantly (older `… from <repository>`
lines too) into the derived entries `[rosetta version]`, `[rosetta extras]` and `[rosetta provenance]`,
compared like flags: runs of a condition must agree, and a pair built with different versions is
confounded. The report lists every build (B1, B2 … with commit, repository, date, extras, the raw lines and
its runs per condition) and warns when the runs were built with more than one.

*Seeds.* The seed is a run's identity and is never normalised away: the value Rosetta logs after the
command line (`Constant seed mode, seed=… real_seed=…`, `-seed_offset` included), else `-constant_seed`
with `-jran` (Rosetta's default 1111111 without one). The report gives the seed and build of every run (and
of every restart), the conditions their seed ranges, the flags table a `[seeds]` row, and `--runs-tsv`
(output *runs* of the step) a row per start of every run: seed, `-constant_seed`, `-jran` as given, where
the seed was read, the build's fields and its raw lines. Two runs, or two starts of one run, with the same
seed under `-constant_seed` and the same inputs (flags and input contents; output names, `-nstruct`, the
executable's path aside) write the same decoys: they are listed under *DUPLICATE SEEDS* as an error (and on
standard error; `--strict` fails). The same seed with other inputs or another build is a matched-seed design
and is only noted; without `-constant_seed` Rosetta ignores `-jran`.

**Frame check** (step *MDATK · frame check*, `mdatk frame check`). Models built around pre-placed parts
(helices given with `-s`, a fit into a map) inherit their relative placement. Each model is superposed on
the reference with the atoms of one body (*Fit on*, default P, C1′, C4′ of `group:Aptamer`), and the RMSD of
each other body (*Measure*, one or more) under that superposition says how far it moved relative to the fit
body. A model whose fit body is more than 1 Å from the reference's does not contain it: it is listed, not
counted. With the 95th percentile of every measured body below 1 Å the models share one frame; otherwise
the placement was sampled. Atoms are matched by chain, residue number and insertion code; *Model numbering*
pairs differently numbered residues (`C:1-29=B:45-73`). The report table has a row per model (run and tag
from a decoy table), also for models that could not be measured. Every file is read once: a model given that
is also a decoy of the connected table (the set of a FARFAR2 step together with its decoy table) is that
decoy, with its run and tag, and the table's cut (*Decoys of the table*) leaves it out. Folders given are
searched for PDB files, never inside `superseded/`.

**Re-fitted reference** (`mdatk frame reframe`). The old reference is superposed onto the re-fitted one on
the fit body; if every measured body is carried to its new place within 1 Å, one rigid transform
describes the re-fit and the models are moved with it (ANISOU tensors rotated, decoys of a table written
as `<run>/<tag>.pdb`, a manifest and a decoy table pointing at the moved files). The step names the moved
models `<project>_<NN>_reframed-01_<time>.pdb` … (`--pattern`) and writes exactly as many as its output
*Moved models* lists (`--slots`): with fewer models (a set of decoys, a cut of the table) the rest are empty
slots. Otherwise the bodies moved relative to each other and the step fails: the models must be built
again on the new reference. Models sampled with the map in the score (`elec_dens_fast`) were drawn towards
the old placement; moving them keeps their geometry, which is not what sampling in the new frame would
build, and the step warns.

For a construct with an aptamer of chains A and B (1-16, 17-44) and a target in chain C (45-73):

```bash
mdatk rosetta scores 01_farfar2/ --max linear_chainbreak=5 --tsv scores.tsv
mdatk rosetta audit runs/ --pair-suffix _nointernalpairs --controlled B:22-24,B:35-37 -o audit.txt \
    --runs-tsv runs.tsv
mdatk frame check helices.pdb --scores 02_farfar2-models/apt1_02_scores_20261015-1432.tsv \
    --fit A:1-16,B:17-44 --measure Target=C:45-73 --tsv frame.tsv
mdatk frame reframe helices.pdb helices_sharpened.pdb models/ --fit A:1-16,B:17-44 \
    --measure C:45-73 -O reframed --manifest reframed.tsv
```

Flags given to `--allow` / `--ignore` start with a dash: write them with `=` (`--allow=-cycles`).

### Custom scripts

A **Custom script** step (`user-script`, *Add step → Custom*) runs a script of your own as part of a
plan, for whatever the built-in steps do not cover: renaming residues, a quick analysis, converting a
file for another program.

- **Script.** Type it into the step (*typed here*) or choose a file (*from a file*). Either way its text
  is copied into the step folder each time the step's files are written, so the plan folder stays
  self-contained; after editing the original file, write the step again. *Name in the step folder*
  names the copy (default `<project>_<step>_script_<time>.sh` / `.py`, or the file's own name without
  a leading `.`, `-` or `+`), e.g. when a program wants an extension (`.pml` for PyMOL).
- **Run with** bash, python3 or another command (`Rscript`, `python3.12 -u`,
  `vmd -dispdev text -e {script} -args`: the script's name goes where `{script}` is, otherwise at the
  end). *Arguments* follow, in shell syntax.
- **Input files** and **Output files** are lists (*+* adds one from a template): a name and a file
  type each, *Optional* and *Several connections* for inputs, and a file name for outputs (empty:
  `<project>_<step>_<name>_<time>.<ext>`). Each one becomes a port of the box. An input of type *any
  file* accepts every output; a typed input only outputs of its type, as for other steps.
- **Cores and GPUs.** What the script uses when RNAdo runs it beside other steps (default 1 core, no
  GPU; see [Running steps](#running-steps)); a script that asks for GPUs gets free ones in
  `CUDA_VISIBLE_DEVICES`, unless its *Environment* sets `CUDA_VISIBLE_DEVICES`. The run script does not
  change.
- **Environment.** A conda environment and setup lines (`module load …`), as for the other steps.

The step's run script (`<project>_<step>_custom_<time>.sh`, the one *Run* starts) changes into the
step folder and exports these variables (`<NAME>` is the declared name in upper case, `-` as `_`):

| Variable | Value |
|---|---|
| `MDATK_IN_<NAME>` | the input's path, relative to the step folder; one path per line when it carries several files, empty for an optional input without a file |
| `MDATK_OUT_<NAME>` | the name of the file the script must write into the step folder |
| `MDATK_PROJECT`, `MDATK_STEP`, `MDATK_STAMP` | the parts of the plan's file names |

It then replaces `{in:<name>}`, `{out:<name>}` and `{script}` in the arguments with the names, quoted
for the shell (one word per file; do not put quotes around them). An optional input without a file
becomes `''`, an empty word that keeps the place of a positional argument, or no word at all with
*Several connections*. The run script removes the declared outputs of an earlier run, runs the script,
checks that every declared output exists and is not empty, and only then writes the marker
`<project>_<step>_custom_<time>.done`. A script that fails (non-zero exit status) or misses an output
fails the step, so the steps that read from it are not started; a file left from an earlier run does
not count. *Check previous runs* counts a run as done only with its marker, so a run that failed or was
stopped half-way, perhaps after writing part of an output, is *partial*. An output with a fixed file
name is the same file under every stamp: it belongs to the stamp whose marker is there, and a new run
removes the markers of the step's earlier stamps too. For example, a python3 step with the input
`structure` (PDB), the output `fixed` (PDB, file `fixed.pdb`) and the arguments
`--model {in:structure} -o {out:fixed}` writes

```bash
export MDATK_IN_STRUCTURE=../01_build/apt1_01_helix_20261015-1432.pdb
export MDATK_OUT_FIXED=fixed.pdb

rm -f -- apt1_02_custom_20261015-1432.done fixed.pdb
rm -f -- apt1_02_custom.done apt1_02_custom_????????-????.done

python3 apt1_02_script_20261015-1432.py --model ../01_build/apt1_01_helix_20261015-1432.pdb -o fixed.pdb

[ -s "fixed.pdb" ] || { echo "ERROR: apt1_02_script_20261015-1432.py did not write fixed.pdb" >&2; exit 1; }

echo "MDATK: custom script finished $(date '+%Y-%m-%d %H:%M:%S')" > apt1_02_custom_20261015-1432.done
```

and an Amber build connected to `fixed` reads `../02_custom/fixed.pdb`.

The planner reports as errors: a missing, unreadable or empty script; a placeholder that names no
declared input or output, or a misspelt one (`{input:x}`); two inputs or two outputs with the same
name; output file names that are not plain file names (letters, digits, `.`, `_`, `-` and `+`,
starting with a letter, digit or `_`; no folders or spaces); an output that is one of the step's own
input files (the run would remove it before the script reads it); an output file that another step
writes too, which can happen in a plan without step folders. A `#!` line that does not match *Run
with* is a warning. The plan summary shows the command line. MDATK cites nothing for this step and
downloads nothing for it; what the script itself does is up to you, so review it like any other script
before running it.

### Decoy interfaces

The **Decoy interfaces** step (`decoy-interface`, *Add step → Analysis*) describes, model by model, the
interface between two parts of an ensemble (e.g. every decoy of a FARFAR2 step), groups the models
by the contacts they make and checks what holds their rebuilt segments. It assumes no register, construct
or numbering: the two **sides** are residue selections, as chain-qualified ranges (`A:1-16,B:17-44`),
whole chains (`C`), ranges for every chain (`1-44`) or the project's residue groups (`group:Aptamer`,
written into the script as ranges). Side B may be left empty: every nucleotide not in side A. Residues are
keyed by chain, number and insertion code, so chains numbered alike (Amber or tleap output) stay apart;
only nucleotides count (names in any scheme, also L names such as LA or 0A, and unknown names such as
Rosetta's `0` or `LA0` whose atoms show a sugar and a base); ions, water, Rosetta's virtual residues and
hydrogens (by element, also `1H5'`) never do. The sugar's hand (D / L) comes from its chiral volumes.

**Census** (the `census` table, one row per model), across the boundary only:

| Column | What is counted (defaults) |
|---|---|
| `contacts` | heavy-atom pairs closer than 4.5 Å |
| `stack` | base stacking: ring centroids < 4.5 Å apart, ring normals within 30°, at least 2.5 Å of the offset along the mean normal (a coplanar base pair has its offset in the base plane) |
| `wc`, `gu` | Watson-Crick and G·U pairs from the chosen engine: builtin (the pair's hydrogen bonds by base identity and a C1′–C1′ window, so an A·C mismatch or a trans pair is no Watson-Crick pair), barnaba or DSSR |
| `other` | non-canonical pairs (barnaba, DSSR; empty with builtin) |
| `bph` | base N-H donors (A N6, G N1/N2, C N4, U N3) within 3.5 Å of OP1/OP2 of the other side, per donor-acceptor atom pair; optionally also the C-H donors of the base-phosphate classes (A C2/C8, G C8, C/U C5/C6) |
| `ohph` | 2′-OH to OP1/OP2 within 3.5 Å |
| `mgbr` | phosphate pairs, one per side, with a P–P distance of 5–7 Å that one ion could bridge (an opportunity, not an observed ion); every P–P pair is tested |
| `residues_a`, `residues_b`, `fingerprint` | the residues in contact and the residue pairs in contact (`A:8/C:60 …`) |

With a **scores table** connected (the `scores` output of the FARFAR2 models step: columns `run`, `tag`,
`pdb` relative to the plan folder, then every Rosetta score term by name; extra columns are ignored),
models are matched to their rows by path (then file name, then a tag unique across runs), the table's
`score` and the filtered terms are copied into the census, and **score filters** keep the models whose
terms pass, one condition per line (default `linear_chainbreak <= 5`: a severed backbone lets the loops
collapse onto each other and inflates the contacts). A filtered term missing from the table is an error,
not a silent pass. *Models: every decoy in the scores table* takes the models from the table's `pdb`
column instead of the connected ones.

**Clusters** (`clusters` table, `representatives` PDB outputs): leader clustering of the fingerprints by
Jaccard distance (cut 0.5), at the level of the residues of each side (default), residues binned into
blocks per chain, or residue pairs. Seeds go in order of fingerprint size, ties by model name, so the
input order does not matter; the representative of each cluster is its medoid over all members, and the
medoids of the largest clusters are copied as `…_rep-01_….pdb`, `rep-02` … (*Representatives*, default
5), ready for the Amber build, Phenix or NAMD MDFF steps. Clusters with at least *Recurring from* members
(default 5) give the coverage and a neutral summary of the pattern in the log (dispersed / few / several
recurring interfaces). **Reference interfaces** are typed residue sets per side, or the census of
connected *reference models* (measured with the same cut-offs as the ensemble). Each cluster is scored on
the residues exclusive to each reference, on both sides, as the mean share of those residues its members
touch, normalised per reference so that a reference with more exclusive residues does not win by size; the
log gives the size-weighted shares over all recurring clusters, also without *broad* clusters (whose
representative touches more than 10 side-B residues).

**Segment support** (`support` table, plus per-residue and per-residue-share tables): for every rebuilt
residue, its heavy-atom contacts (4.5 Å) beyond a sequence separation of 3, counted within a chain only
(residues of other chains are never neighbours), by source: the partner side (`#`), its own side outside
the rebuilt segments (`o`, scaffold), optionally base pairs between rebuilt residues (`=`, two distinct
base-edge hydrogen bonds within 3.3 Å, or canonical pairs) and its own rebuilt segments only (`:`, which
is no support). A model is flagged when it has a run of 4 unsupported residues (or *auto*: the largest gap
of the ensemble's longest runs). The rebuilt residues are the rest of the sides outside FARFAR2's fixed
residues when the connected models carry them (*Every decoy* of a FARFAR2 step does), or typed (fixed or
rebuilt residues).

For the 73-nt construct the scripts came from (an aptamer in chains A and B, a target in chain C), the
sides are `A:1-16,B:17-44` and `C:45-73`, and FARFAR2's fixed helices `A:1-6 A:13-16 B:17-20 B:39-44
C:45-57 C:64-73`. The same work on the command line:

```bash
mdatk decoys census models/*.pdb --side-a A:1-16,B:17-44 --side-b C:45-73 \
    --scores scores.tsv --filter 'linear_chainbreak<=5' -o census.tsv
mdatk decoys clusters census.tsv --level residues --cut 0.5 --min-size 5 \
    --reference 'pres=A:8-9,B:25-28,B:33-35/C:58-64' \
    --representatives 'rep-{n}.pdb' --n-representatives 5 -o clusters.tsv
mdatk decoys support --census census.tsv --side-a A:1-16,B:17-44 --side-b C:45-73 \
    --fixed 'A:1-6 A:13-16 B:17-20 B:39-44 C:45-57 C:64-73' --run auto --pairs contacts -o support.tsv
mdatk pdb interface model.pdb --side-a group:Aptamer --side-b group:Target -p project.mdatk.toml \
    --engine dssr -o interface.tsv      # pairs, stacks, base / 2'-OH - phosphate H-bonds, hands
```

`-p PROJECT` resolves `group:NAME` and records the command in the project's step log; without `-o` the
table goes to standard output and the summary to standard error.

### Command line

Every `mdatk workflow ACTION` is also `rnado ACTION` (e.g. `rnado write -p plan.mdatk.toml --run`).

```bash
mdatk workflow tools                       # step types; `mdatk workflow tools amber-md` lists options
mdatk workflow generate amber-md -o out --step 4 --project-name apt1 \
    -i topology=sys.parm7 -i coordinates=sys.rst7 -i restraints=helix.dist \
    --set engine=pmemd.MPI --set nproc=32 --set stages.5.nstlim=150000000
mdatk workflow generate user-script -o out --step 2 -i model=model.pdb \
    --set source=file --set script_file=fix.py --set interpreter=python3 \
    --set 'inputs=[{"name": "model", "kind": "pdb"}]' \
    --set 'outputs=[{"name": "fixed", "kind": "pdb", "file": "fixed.pdb"}]' \
    --set 'arguments={in:model} {out:fixed}'          # a custom script step
mdatk workflow template farfar2-amber -p plan.mdatk.toml   # add a template to a project's plan
mdatk workflow check -p plan.mdatk.toml                    # list the plan's problems
mdatk workflow write -p plan.mdatk.toml [--step 3] [--run] # write files, master script, summary
mdatk workflow write -p plan.mdatk.toml --dry-run    # ... list them, and the completed runs that would move
mdatk workflow run -p plan.mdatk.toml [--step 3] [--parallel 2]  # run the written main scripts (logged)
mdatk workflow run -p plan.mdatk.toml --cores 16 --gpus 2   # within 16 CPU cores and 2 GPUs
mdatk workflow run -p plan.mdatk.toml --include-separate  # ... and the scripts marked 'run separately'
mdatk workflow status -p plan.mdatk.toml [--adopt|--forget]  # earlier runs of the steps (see above)
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

## PDB clean-up

*PDB clean-up* in RNAdo's header bar (or *RNAdo → PDB clean-up: open in RNAdo…* in the analysis
window) and `mdatk pdb` on the command line repair and rewrite PDB files with fixed-column output that
keeps coordinates, occupancies and B-factors verbatim:

- **ChimeraX damage** such as a residue name fused with the chain id (`GC:` = residue G of chain C)
  is repaired automatically; only the first MODEL is kept.
- **L-nucleotides** are detected from the signed chiral volumes at C1′, C3′ and C4′ (all usable
  centres must agree; flat or contradicting centres leave the residue to its name) and ranges such as
  `C:45-73` force L. A warning lists L-named residues with D geometry and vice versa.
- **Residue names**: `amber` (D: A C G U, A5/A3 …; L: LA LC LG LU, LA5/LA3 … as in Dudek & Trylska's
  library; a single-residue L segment gets the 5′ unit), `ccd` for Coot / Phenix (L: 0A 0C 0G 0U and
  0A5/0A3 … - these terminal variants are not CCD entries; a 5′ end that keeps its phosphate is named
  after the component itself, e.g. 0G), `rosetta` (L: 0A …, no terminal variants,
  every polymer residue as ATOM because Rosetta ignores HETATM residues), `pdb` (non-standard
  residues as HETATM) and `standard` (plain A C G U for both hands and every record ATOM: a copy for
  programs that read standard RNA only and ignore chirality, such as DSSR; ChimeraX's Q-score reads L
  names as they are and needs no copy; a warning says that the names no longer record the hand, so the
  copy is never meant for tleap, refinement or name-based analyses). Any of these, plus common aliases
  (RA, G5, ADE …), is recognised on input; DNA and unknown residues pass through. *5′/3′ terminal names*
  can be switched off.
- **Hydrogens and Rosetta's virtual residues**: `--strip-h` removes hydrogens (element column, else the
  name), `--drop-virtual` the VRT / XXX residues Rosetta writes into density-scored models (no atoms of
  the molecule: tleap cannot type them, and a map-model score would read them as atoms).
- **CONECT** records are stripped by default (until final deposition processing).
- **5′ phosphates**: `--strip-5p` (window: *Remove 5′-terminal phosphates*, ticked automatically for
  the Amber scheme) removes P/OP1/OP2/OP3 from every chain's 5′ nucleotide, as Amber's 5′ units
  (G5, LG5 …) expect; a residue with hydrogens gets HO5′ where the phosphorus was.
- **Chains**: restore the chain ids of a cpptraj PDB by copying them residue by residue from a
  reference (e.g. the structure the topology was built from), from a list per TER segment, or as
  A, B, C …; split chains per TER segment or at a residue (new ids); combine chains; sort them.
- **Numbering**: continuous across chains (default) or restarting per chain, optional start number
  and kept gaps; **SEQRES** records and an optional **FASTA** file (L chains are marked).

`mdatk pdb check` also measures every sugar: `--expect-l` / `--expect-d RANGES` give the hands
expected (`C:45-73`, `A,B`, `*:45-73` for every chain), `--strict` exits with status 1 when a residue
has the other hand, cannot be measured where a hand is expected, or has an epimerised (C3′ or C1′
against the other centres), contradicting or flat centre, and `--table` prints one line per residue
(name, the hand the name implies, the three-centre call, the C4′ hand alone, the C4′ / C3′ / C1′
volumes, the expectation and a note).

```bash
mdatk pdb check model.pdb                                   # chains, sequences, names, chirality (JSON)
mdatk pdb check refined.pdb --expect-l C --expect-d A,B --strict --table   # every sugar, per residue
mdatk pdb clean decoy.pdb -o amber.pdb --scheme amber --strip-5p --strip-h --drop-virtual  # decoy for tleap
mdatk pdb clean decoy.pdb -o std.pdb --scheme standard --strip-h --drop-virtual   # chirality-blind tools
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
the input's residues), so annotate after cleaning; the clean-up window does this when *Annotation* is
on. Strands are TER segments, so files with blank chain ids (Amber, cpptraj) still get a `&` between
strands:

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

### Model checks

`mdatk pdb qc MODEL.pdb …` and the planner step *Model check (clashes · backbone · chirality)* look
for what a score, a map fit and the base pairs do not show: atoms pressed together, a chain break that
tleap would bond anyway, a loop coiled into a knot, a ribose of the wrong hand. Both run the same code
(`mdatk/modelcheck.py` on top of the PDB tools, standard library only); residues are addressed by
chain, number and insertion code, and every range is written as for the PDB tools (`A:7-12, B:25-34`,
`C` for a whole chain, `*:45-73` for every chain; in the step also `group:NAME` for a residue group).

- **Clashes**: heavy atoms closer than their van der Waals contact (Bondi radii) by at least 0.4 Å
  (`--clash-tol`), listed with the overlap. An N/O pair with a donor on either side is a hydrogen bond
  between 2.2 and 3.2 Å (`--hbond-min` / `--hbond-max`), not a clash, and closer than 2.2 Å a clash
  (no room for the hydrogen). Donors follow the base (2′-OH; A N6; G N1, N2; C N4; U N3), plus terminal
  5′/3′ hydroxyls, A N1 / C N3 of residues given with `--protonated`, waters, and the N/O atoms of
  other residues; two pure acceptors (a phosphate oxygen and O4′, say) never H-bond. A carbon one bond
  from an H-bonded atom, against its partner and no closer than the H-bond itself, is the same contact
  and is not counted either. Atoms up to three bonds apart are left out, along the bonds of the
  nucleotide templates and the O3′–P links of each TER segment, so neighbouring residues (stacking,
  sugar–base) and all chains are checked. Waters and ions are skipped unless asked for (`--waters`,
  `--ions check`: ions against their first-shell distances, so Mg²⁺–O at 2.1 Å is coordination).
  Hydrogens are never part of it: their positions differ between programs, and the builder's ideal
  A-form helix (the Arnott fiber) already has H2′(i)…H6/H8(i+1) at 1.57–1.70 Å. That H2′ is placed
  from the sugar's own geometry (1.09 Å from C2′, within 5° of the exact tetrahedral direction from C1′,
  C3′ and O2′; putting it exactly there moves the contact by under 0.03 Å): the short contact is the
  fiber model's C2′(i)…C8/C6(i+1) of 3.5 Å, so smaller hydrogen radii would hide a geometry rather than
  fix a threshold.
- **Hydrogen contacts** (`--hydrogen-check`, off by default; reported in a section and columns of their
  own, `h_atoms h_clashes h_worst_overlap h_clashscore h_radii`, and never a gate or a filter): the
  model's explicit hydrogens on MolProbity's conventions, none added. Radii from Probe (`atomprops.h`):
  for hydrogens at nuclear positions (C–H 1.09 Å, as Amber, Rosetta and the builder place them) 1.17 Å
  on carbon and 1.00 Å for a polar or aromatic (base C–H) hydrogen, for electron-cloud positions (C–H
  0.97 Å, Reduce's default for X-ray models) 1.22 and 1.05 Å, chosen from the model's median C–H
  (`--hydrogen-radii auto|nuclear|ecloud`); heavy atoms C 1.70, N 1.55, O 1.40, P 1.80 Å. An overlap
  of 0.4 Å or more is a clash; a polar hydrogen may overlap an acceptor by 0.6 Å (a hydrogen bond,
  Probe's `-HBRegular`) before the excess counts; pairs up to four bonds apart are left out; the
  clashscore is clashes per 1000 atoms. On the fiber helix this flags every H2′(i)…H6/H8(i+1).
- **Backbone**: O3′(i)–P(i+1) of consecutive nucleotides within each TER segment, as tleap bonds them
  (by name, whatever the distance): up to 2.0 Å a bond (`--link-max`); from 2.0 to 3.0 Å
  (`--link-fail`) a strained bond, a warning that gives the strain the first minimisation has to relax,
  E = k(r − r0)² with Amber's k (no ½): 230 kcal/mol/Å² and 1.61 Å (OS–P of parm99 / parm10.dat), about
  35 kcal/mol at 2.0 Å, 180 at 2.5 Å, 440 at 3.0 Å, or read from the force field with `--link-params`
  (a tleap input or leaprc followed through its `source` / `loadamberparams` lines, parm*.dat, frcmod;
  `--link-types OR,OS` for the CP force field, `--amberhome`) or given (`--link-k`, `--link-r0`);
  longer than 3.0 Å a break (the chain is usually in two pieces), shorter than 1.4 Å an overlap, a
  missing atom an unmeasurable link, which fail the links gate. Report columns `strained_links` and
  `link_strain` (kcal/mol, every link outside the limits); consecutive P–P outside 5.0–7.5 Å are
  reported.
- **Loop compaction**: end-to-end distance over contour length along C1′ for `--segments` and for the
  runs of three or more residues unpaired in every `--secstruct` file given (two registers of one
  sequence, say: a segment that is a hairpin in one and a strand in the other is left out). Below 0.20
  (`--compaction-flag`) a loop is flagged for inspection; there is no verdict: a loop is meant to turn.
- **Sugar hands** from the C4′, C3′ and C1′ geometry (never from the names, which a Rosetta silent file
  does not keep), against `--expect-l` / `--expect-d`; epimers and flat centres are named.
- **Kinks**: the virtual angle C1′(i−1)–C1′(i)–C1′(i+1) of every residue between two consecutive
  nucleotides of a TER segment (A-form 152°); below 60° (`--kink-min`) a sharp kink is flagged, except
  where one is allowed: `--kink-allow RANGES` (the flexible positions of in-line probing, say) and the
  positions of a two-column reactivity file (`--kink-reactivity FILE`: position, reactivity; tabs,
  commas or spaces; NaN skipped) that reach `--kink-reactivity-min`, numbered in
  `--kink-reactivity-chain` (default: every chain). Report columns `kinks`, `sharpest_kink`,
  `sharpest_kink_at`; a gate only with `--gate-kinks`.
- **Decoy table** (`--scores`): Rosetta score terms per decoy (columns `run`, `tag`, `pdb` relative to
  the plan root, then the terms by name), matched to the models by path, file name or tag:
  `--score-terms` copies terms into the report, `--score-limits 'linear_chainbreak<=1'` gates on them,
  `--order score` ranks by `--rank-term`.
- **Gates** (`--max-clashes`, links and hands on by default, `--gate-pp`, `--gate-kinks`, score limits;
  strained links and ungated kinks are warnings, column `warnings`, and hydrogen contacts never gate)
  and what happens to failing models: `--on-fail report` (default; all models passed on), `filter` or
  `stop` (status 1 when any fails). `filter` passes on the models that pass, up to `--pass N` of them,
  and fails only when fewer than `--min-pass M` pass (default 1: when none does; 0: never), since how
  many pass can be the finding itself. With `--pass N` the models fill N slots in order and the slots
  left over are written as empty-slot files (`REMARK 951 empty slot 4 of 5: only 3 models passed`,
  `END`, no atoms), so a plan keeps its fixed number of files
  without pretending that N models passed; the report states `M of K models passed the gates` (and how
  many slots were written empty, once they are: a filter below its minimum writes nothing). `report`
  and `stop` take `--slots N` for the same fixed number of files: the models fill the slots in order,
  the slots left over (inputs that were empty slots, a set of decoys that listed fewer than planned) are
  written as empty slots that say why (`empty slot 6 of 6: only 5 models were given`, `only 1 of 3
  inputs were models (2 empty slots)`), and more models than slots is an error.
  `--passed-out 'passed-{n:02d}.pdb'` and `--best-out` copy the models
  passed on (in input order, `--order clashes` or `--order score`; the best is an empty slot when none
  passed and `--min-pass 0`). The report is text, `--tsv` (one row per model) or `--json`, with `-o FILE`
  for the table while the text goes to the screen.
- **Empty slots downstream**: `pdbtools.is_empty_slot(path)` (and `empty_slot`, its note) tells them
  apart; the model check itself, the decoy interfaces (`mdatk decoys`, `mdatk pdb interface`), the
  register census and ensemble comparison (`mdatk ensemble`), the frame check (`mdatk frame`) and the
  map-model score (`mdatk map score`) skip them with a note, `read_pdb` refuses one by name, and an
  Amber build whose structure is one stops before any helper or tleap runs. Other steps that read one
  model (SimRNA, FARFAR2's `-s`, RNAMake, the real-space refinements, MDFF, restraint dictionaries) do
  not check: they take the first slot, which is empty only when a filter with `--min-pass 0` passed none.

```bash
mdatk pdb qc model-*.pdb --segments 'A:7-12, B:25-34, C:58-63' --expect-d A,B --expect-l C
mdatk pdb qc decoys/*.pdb --secstruct presentation.secstruct --secstruct invasion.secstruct \
    --scores decoys.tsv --score-terms score,linear_chainbreak --on-fail filter --pass 5 --order score \
    --passed-out 'passed-{n:02d}.pdb' --best-out best.pdb --tsv -o qc.tsv
mdatk pdb qc decoys/*.pdb --kink-reactivity inline.tsv --kink-reactivity-min 0.5 \
    --kink-reactivity-chain C --gate-kinks --hydrogen-check --link-params tleap.leap --tsv -o qc.tsv
```

The planner step takes the models (several, or the set *Every decoy* of a FARFAR2 step: a large dot),
optional secondary structures and an optional decoy table (the FARFAR2 models step's `scores`, or the
FARFAR2 step's *Decoy table*), and writes the report (table), the passed models and the
best one (the first passed). With *Models that fail: filter*, *Models passed on (up to)* is the number
of slots and *Fewest passing models* the minimum (1); its *Passed models* then carry a mark that the
slots may be empty, and an Amber build fed one writes the guard that stops it before tleap. In every
mode the step writes one passed file per connected model (of a set of decoys, as many as it can hold),
so the next step finds every file it was planned with: with *report* and *stop* the slots no model fills
(inputs that were empty slots, decoys the set did not list) are empty slots too. *Kinks* and
*Report hydrogen contacts* set the checks above. *Expected hands* is automatic by default: the hands
measured on the first connected model when the plan is written (when it exists then; of a set of decoys
the first it lists once its step has run), or the hand of a builder helix; *Ranges* takes them typed.
Its script copies MDATK's PDB tools and model checks into the step folder, so the machine that runs it
needs `python3` only, and removes the outputs of an earlier run
first, so a check that fails leaves nothing that looks finished. This replaces `model_sanity.py`,
`chirality.py` and `bbcheck.py`; `mdatk pdb clean --scheme amber --strip-5p --strip-h --drop-virtual`
replaces `to_amber.py`, and `--scheme standard` the renaming of `q_prep.py`.
`--engine dssr` runs x3dna-dssr (licensed separately, free for academic use; never bundled; `--dssr-exe`
names the executable) on a copy in a temporary folder, so its auxiliary `dssr-*` files never land next to
your files. The copy gives every nucleotide its canonical name (A C G U, DA …; L names and names DSSR does
not know are mapped from the residue or its atoms), leaves ions, water and Rosetta's virtual residues out,
and numbers the residues itself, so DSSR's answer maps back by residue index and never depends on how a
DSSR version writes its residue ids (`A.G1`, `1:A.G1`, `A.0A/45`, `A.G1^B`, atom ids `N1@A.G1`). Its
Leontis-Westhof classes are written in the same form as Barnaba's (`cWW` Watson-Crick → `WCc`, wobble →
`GUc`, `tHS` → `HSt`), stacks as `--`. DSSR, like Barnaba, does not check chirality: it classifies a D–L
pair like any other. `mdatk pdb interface` (see [Decoy interfaces](#decoy-interfaces)) lists the pairs,
stacks and base / 2′-OH to phosphate hydrogen bonds between two sides of each model, with the hand of each
partner, from the builtin engine, Barnaba or DSSR.

## Structure builder

*Structure builder* in RNAdo's header bar (or *RNAdo → Structure builder: open in RNAdo…* in the
analysis window), `mdatk build` and the planner step *Structure builder (MDATK)* make starting
structures without external programs.

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

**Other pairings, parallel duplexes and triplexes.** `--pairing` (*Pairing* in the builder window and
in the planner step) chooses the base-pair geometry of the duplex; `wc` (the default) is the A-form
duplex above, unchanged. The Leontis-Westhof rules fix the strand orientation:

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
short script calling `python3 -m mdatk build …` (the step's *Python*, which must be able to import
MDATK; with the default `python3`, a step written from the Linux AppImage calls the AppImage itself);
its inputs are the model (mutate, fit) and the map (fit), its output the PDB.

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

- `mdatk dict list` - the twelve codes with the isomeric SMILES given to the programs (all four sugar centres
  L; `--json` for scripts). `mdatk dict smiles 0G5` prints one.
- `mdatk dict finish work_0G5/ --code 0G5 -o 0G5.cif` - finds the dictionary acedrg / eLBOW wrote (a file,
  a folder or a glob), maps every atom to the CCD names by graph matching (element + bonds; hydrogens by
  their heavy atom: H5'/H5'', H61/H62 ... from the generated coordinates), checks the sugar chirality and sets
  `_chem_comp.id`, name and `group` (`RNA` by default so the residue is linked into the chain;
  `--group non-polymer` for a free ligand). `--project` records the step in the project log.
- `mdatk dict pubchem CID` - the only command that uses the network: fetches the isomeric SMILES of a PubChem
  compound and reports which code it matches (wrong stereochemistry or compound is refused).
- Planner step **Restraint dictionaries · acedrg / eLBOW** (Refinement): program acedrg or phenix.elbow
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
  PDB of a solvated system must be in the topology's atom order and carry the box (CRYST1; the
  angles are taken from the topology, so a truncated octahedron keeps its exact 109.4712°).
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
     - box, four-site water and time step that do not fit the topology (4 fs needs HMR);
     - bonds longer than 3 Å: strands a PDB numbered as one chain, which tleap joined.
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
    agree with its coordinates. Some L-RNA libraries write D names (G5, C3 …) into their units; for
    such a topology, list the L residues under *L residues* (topology numbers, e.g. `45-73`). Every
    other nucleotide is then D, and the geometry must still agree with the declaration. A residue
    named D whose sugar is L, or the reverse, stops the run. So does any clearly defined sugar centre
    of the other hand, even in a residue whose overall geometry is ambiguous: its restraint would
    invert it. The planner reports these as soon as the files exist (for a NetCDF restart, the
    script does once it has converted the restart).
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

## Decoy ensembles, map-model scores and statistics

Four planner steps (and their commands) look at many models at once: which pairing each decoy formed and
under which input, how two ensembles differ, how well each residue fits a map, and which re-fitted body
earned its move. The library is `mdatk/ensemble.py` (NumPy), `mdatk/ensemble_stats.py` (standard library
only) and `mdatk/mapscore.py`; the steps write a short script calling `python3 -m mdatk …`, like the
structure builder, so MDATK must be installed where they run.

**Residues and ranges.** Residues are keyed by chain, number and insertion code throughout (read with
`mdatk.pdbtools`: first model, first alternate location), so two chains numbered from 1 never merge.
Ranges are chain-qualified as elsewhere in MDATK (`A:1-16,B:17-44`, `C` for a whole chain); in the steps
they may also name a project residue group (`group:Aptamer`; its unqualified numbers mean any chain).

**Registers** are named sets of defining base pairs, `pres=B:22-B:37, B:23-B:36, B:24-B:35`. A pair is
*formed* when its Watson-Crick N1/N3 distance (purine N1, pyrimidine N3) is below 3.5 Å, and *in register*
(approached) when it is formed or its N1-N3 is below 6 Å with C1'-C1' between 8 and 13 Å: textbook pair
geometry whose bases have not closed yet, which refinement or a short MD would close. Bases come from the
residue name (any MDATK scheme: A, 0A, LA5 …), else from one diagnostic atom (O6 G, N6 A, N4 C, O4 U),
which also reads names such as `0` or `LG0`.

**Numbering.** The registers and ranges are written in the numbering of a reference construct. With a
reference sequence (typed, a FASTA file, or the plan's FARFAR2 sequence), each chain of a decoy is located in
it by its sequence (gaps in its numbering become wildcards), so per-chain, shifted or renumbered decoys
still match; the bases of the defining residues are checked in every decoy, and a decoy that does not match
is reported, not counted. Explicit per-chain offsets (`B:-367,C:44`) do the same by hand. A plain sequence
leaves the chain ids open (the decoys keep theirs); FASTA records with Rosetta's tags (`>x A:1-16`) and
`A:GGGC` prefixes give them.

**Register census** (`register-census`, `mdatk ensemble census`). Counts, per group of decoys, how many
formed and approached each register (and every combination), the pairs per decoy, score deciles, the
scores of the decoys forming exactly one register (Mann-Whitney between registers) and, optionally, the
L-handed riboses per decoy. A group is a folder (or the folder `--group-depth` levels up); in the planner
each connection of decoys from another step is its own group. Its **build condition** is read, per
register, from a Rosetta log beside the decoys or in the given log folders (`core.init: command:`, with
`@flags` files expanded, relative paths tried from the log's folder and the run folders below it): the
secondary structure it names (`-secstruct_file`, `-secstruct`; none given: all unpaired) and
`-obligate_pair(_explicit)`; failing a log, from a `*.secstruct` beside the decoys, then from a connected
secondary structure. FORCED: every defining pair is imposed; UNFORCED: none of the register's residues is
paired by the input; PARTIAL otherwise; a group without a source, or whose logs disagree, is UNKNOWN.
Defining residues held by the `-s` structure (`-input_res`) are noted. Groups are reported separately and
pooled only with groups of the same condition, never across conditions; `--pair-suffix` (e.g.
`_nointernalpairs`) sets each group X beside the group X<suffix> with Fisher tests. Scores come from the
decoy table (below), else from the pose energies table in the PDB, else from a `denovo_<100000+10·score>_`
file name; deciles are formed within one weight set only. Output: one row per decoy (`.tsv`) and a report.

**Ensemble comparison** (`ensemble-compare`, `mdatk ensemble compare`). Two arms, each with its register,
per decoy: closure, all defining pairs formed, both, and the share of interface atoms in the density.

- *Closure*: a score term of the decoy table below a cut (default `linear_chainbreak < 5`), or every
  O3'(i)-P(i+1) of consecutive nucleotides of one chain segment below 2.0 Å.
- *Occupancy*: heavy atoms of the interface regions above a contour set at the 5th percentile of the density
  at the atoms of regions held fixed; per decoy (its own fixed atoms), or one common level for both arms
  (pilot mode: the regions that did not move). Map values are interpolated trilinearly (`mdatk.density`);
  `nearest` takes the nearest voxel and reproduces the original scripts' numbers exactly. Atoms outside the
  map count as not in density and are reported.
- *Statistics* (`mdatk.ensemble_stats`, checked against SciPy in the tests): rates with exact
  Clopper-Pearson intervals (summed in log space, so ensembles of any size work) or Wilson intervals,
  two-sided Fisher exact tests, Mann-Whitney U (exact or normal approximation), and the **power
  statement**: the rates of arm B that a two-sided Fisher test would tell from arm A's with the given power
  (default 80%), computed exactly up to 4 000 decoys in both arms and from the normal approximation above.
  Both arms' sampling enters, so the uncertainty of the baseline rate is part of it.
- *Modes*: `compare` (the headline rates, best closed scores, median occupancy), `sweep` (also over a range
  of closure cuts - a sensitivity display; its p-values are not corrected for the number of cuts) and
  `pilot` (two runs of one register: common contour, Wilson intervals, optionally only the runs both arms
  have, with their seeds compared when log folders are given).

The report gives numbers only; deciding what they mean (e.g. whether to regenerate an ensemble) is left to
you.

**The decoy table.** The FARFAR2 models step's `scores` output (kind *table*) is a TSV with the columns
`run`, `tag`, `pdb` (relative to the plan folder; empty for a decoy not extracted) and every Rosetta SCORE
term by its name. The census and the comparison take it as input `scores` (`--scores TSV`, `--root PLAN`),
find each decoy's row by its path (else by file name, else by tag against the file name) and look terms up
by name; extra columns are ignored, comma-separated files are read too.

**Map-model score** (`map-model-score`, `mdatk map score`). Per residue of every model:

- `qscore` (default): the Q-score (Pintilie et al., Nat. Methods 17, 328-334 (2020)) computed by
  ChimeraX's `qscore` command, which comes with the *QScore* bundle of the ChimeraX Toolshed
  (ChimeraX-QScore by T. I. Croll; in ChimeraX: `toolshed install QScore`). ChimeraX reads the model as it
  is: L residue names (LG5, 0A …) and HETATM records are scored like any other residue, so no renamed copy
  is made. Each model is one ChimeraX run, the command you would type yourself:
  `chimerax --nogui --cmd "open MODEL id 1 maxModels 1; open MAP id 2; qscore #1 toVolume #2 useGui false
  logDetails true [options]; exit"` (`maxModels 1`: the first model of a multi-model file, as MDATK reads
  it). *Models per ChimeraX run* (`--batch N`) opens the map once and the models one after another, each
  opened as #2 (`id 2`, so the numbering never drifts), scored and closed, with a marker line (`log text
  MDATK-QSCORE k`) logged before each so that the tables are told apart; ChimeraX ends a `--nogui` run at the
  first command that fails, and MDATK then resumes after that model. Further qscore options are typed as
  keyword value pairs (`pointsPerShell 16 shellRadiusStep 0.05`; also `referenceGaussianSigma`,
  `maxShellRadius`); `useGui false logDetails true` are always set, and the command has no resolution or
  hydrogen option (hydrogens are never scored). Rosetta's virtual residues (VRT / XXX) are left out of the
  selection (`#1 & ~:VRT`) and of the table. The per-residue table that `logDetails` prints (Chain, Number,
  Name, Qavg, Qworst, Qbb, Qsc) is read from ChimeraX's log tolerantly: blank lines between the rows, the
  dashed rule, tabs or spaces, the `INFO:` lines and `<pre>` tags of a plain-text log, insertion codes
  after the number, `N/A` cells, warnings such as `Ignored bad PDB record` (counted in the report); the
  `Overall mean Q-Score` line closes the table and is reported per model (a mean over atoms, termini
  included). The per-residue output has all four Q columns plus `score`, the one the regions use: Qavg by
  default, or Qbb (backbone) / Qsc (side chain, i.e. the base) as ChimeraX classifies the atoms. ChimeraX's
  log of every run stays in the step's `qscore/` folder. A model ChimeraX cannot score is listed in the
  report (with the reason and its log) and not ranked; the others are written and the command ends with
  status 1. Known limit of ChimeraX-QScore 1.2: a residue with an atom outside the map box makes its
  per-residue table stop there, so such a model fails with "atoms outside the map box".
- `cc`: MDATK's per-residue correlation, no external program: the model's map is simulated at the given
  resolution and, for each residue, correlated with the map over the grid points within 2.5 Å of its atoms.

Named regions (`interface=B:25-34,C:58-63`) are then averaged per model, with their difference to a reference
region (default the whole model; typically the scaffold, e.g. the fixed helices every decoy shares) from
which every named region is removed, so no residue counts on both sides. The first and last residue of every
chain segment are left out of every mean unless kept (a terminal residue scores lower for lack of
neighbours); residues without a score (outside the map) are counted, never averaged. Models are ranked by
**dQ** = Q(interface) − Q(scaffold) of the first region unless another order is given (`--sort NAME` for a
region's mean, `delta:NAME`, `reference`, `whole`): the scaffold is the same in every decoy and the
best-ordered part of the map, so dQ follows the interface and survives a change of map scaling or
sharpening, where a whole-model mean mostly measures the shared scaffold. The report gives, per model, the
mean of each region, its `dQ:NAME`, the reference and the whole model, then the spread of each over the
models. Outputs: the per-residue table, the region table (`delta` per region) and the ranking report.
`mdatk ensemble regions TABLE…` ranks per-residue tables written elsewhere (also ChimeraX logs with qscore
tables, one model per table, qscore's CSV and headless `chain,resnum,base,Q` CSV files).

**Body hybrids** (`body-hybrids`, `mdatk pdb hybrids`). Up to six bodies (name and ranges, a body may span
chains) of an old and a re-fitted model; every one of the 2^k combinations is written, over the residues and
atoms both files have, with coordinates copied verbatim from each body's source (residues in no body stay
old). Residues are matched by chain and number, or by number alone for files whose chain ids differ (the
numbers must then be unique). With a map, every hybrid is scored (CC or Q-score) and each body's main effect
reported: the mean score of the hybrids with its new coordinates minus the mean of those with its old ones,
over the whole model and over the body itself.

Example, for a 73-nt construct numbered A 1-16, B 17-44, C 45-73 (a nicked aptamer and its target):

```bash
mdatk ensemble census runs/ -r \
    --reference GGGCAAAGAAACAGCUAGCUAGGGCGCGAGCAAACCCGUUGCCCGCCAGAUUUGAGCCUGGGAGCUCUCUGGC \
    --register 'pres=B:22-B:37,B:23-B:36,B:24-B:35' --register 'inv=B:35-C:60,B:36-C:61,B:37-C:62' \
    --exclude 'helices*' --exclude '*_native.pdb' --pair-suffix _nointernalpairs --jobs 8 -o census.tsv
mdatk ensemble compare --mode sweep --arm-a run/pres --arm-b run/inv --labels presentation,invasion \
    --register-a 'pres=B:22-B:37,B:23-B:36,B:24-B:35' --register-b 'inv=B:35-C:60,B:36-C:61,B:37-C:62' \
    --scores models_scores.tsv --map map.mrc --calibration A:1-6,A:13-16,B:17-20,B:39-44,C:45-57,C:64-73 \
    --regions B:25-34,C:58-63 --report sweep.txt
mdatk ensemble compare --mode pilot --register 'pres=B:22-B:37,B:23-B:36,B:24-B:35' --arm-a run/pres \
    --arm-b pilot/pres --labels baseline,pilot --scores scores.tsv --map map.mrc \
    --calibration A:13-16,B:17-20,C:45-57,C:64-73 --regions B:25-34,C:58-63 --matched-runs
mdatk map score decoy_*.pdb --map map.mrc --method qscore --chimerax /usr/bin/chimerax --batch 20 \
    --region pres-loops=B:25-34,C:58-63 --reference-region A:1-6,A:13-16,B:17-20,B:39-44,C:45-57,C:64-73
mdatk pdb hybrids helices_old.pdb helices_new.pdb --body terminal=A:1-6,B:39-44 \
    --body junction=A:13-16,B:17-20 --body L-TAR=C:45-57,C:64-73 --match number \
    --map map.mrc --resolution 4.8 --effects-out effects.tsv -o hybrids
```

Tested on synthetic D- and L-RNA duplexes (`tests/test_ensemble.py`): the register counts and the
nearest-voxel occupancy equal those of the scripts they replace. The Q-score runner was tested with a
stand-in `chimerax` that prints the log format of a real run (`tests/test_chimerax_qscore.py`), not with
ChimeraX itself; the command options (`open … id`, `maxModels`, `log text`, the qscore keywords) were checked
against the ChimeraX and ChimeraX-QScore sources.

## Citations

MDATK cites only what a project uses. Each planner step reports the programs, force fields,
modifications, water models and ion parameters its settings turn on (an Amber build with OL3 + CP,
NBfix19, TIP4P/2005 and nanoMg cites those papers, not the ones for options left off); the analysis
and video pipeline adds MDAnalysis, NumPy, SciPy, Matplotlib, and cpptraj / FFmpeg / VTK when they
are used. The citations appear

- in the header of every generated script (that step only),
- in the step log entries and in `MDATK_params_log.toml` (the whole project),
- in the analysis window's *About and citations* (the whole project), with *Export citations
  (BibTeX)…*, and in RNAdo's *About RNAdo* (the plan), with *Export BibTeX…*.

The reference data live in `mdatk/_citation_data.py`; each entry records where its details were
checked (the paper's PDF, the project's own citation file, or a reference list).

## Command line reference

The analysis window and the analysis commands:

```text
mdatk gui [PROJECT]                the analysis window (opens PROJECT if given; nothing otherwise);
                                   mdatk-gui [PROJECT] is the same without a terminal
mdatk demo DIR [--n-frames N] [--dt PS] [--no-solvent] [--run]
mdatk init OUT.mdatk.toml [--overwrite]   write a project with the defaults (accepts the overrides below)
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
mdatk desktop install [--replace] | uninstall | status   MDATK and RNAdo in the applications menu
                                   (Linux; see Applications menu)
```

`mdatk render` also takes `--video`, `--charts` and `--pdb` to switch an output on that the project has
off.

RNAdo and its tools:

```text
rnado [PROJECT] [--open wizard|pdb|builder]   RNAdo, the workflow planner (rnado-gui: no terminal)
rnado --install-launchers          the same as mdatk desktop install
rnado ACTION ...                   the same as mdatk workflow ACTION ... (tools, check, write, run,
                                   status, generate, template, restraints)
mdatk rnado ...                    the same as rnado ...
mdatk gui --rnado [PROJECT] [--open wizard|pdb|builder]   RNAdo instead of the analysis window
                                   (--planner is the same as --rnado; --open alone starts RNAdo too)
mdatk workflow tools [TOOL]        step types, or the options of one
mdatk workflow template [NAME] [-p PROJECT]   list templates, or add one to a project's plan
mdatk workflow check  -p PROJECT   resolve the plan and list its problems
mdatk workflow write  -p PROJECT [--step N] [--force] [--no-save] [--dry-run] [--run [--parallel N]
                      [--cores N] [--gpus N] [--include-separate]]
mdatk workflow run    -p PROJECT [--step N] [--parallel N] [--cores N] [--gpus N] [--include-separate]
                                   CPU cores (0: half of them) and GPUs (-1: those found) the scripts
                                   share; --parallel caps the scripts at once (0: no limit)
mdatk workflow status -p PROJECT [--step N] [--adopt | --forget] [--json]
mdatk workflow generate TOOL [-o DIR] [--step N] [-i PORT=PATH ...] [-s KEY=VALUE ...] [--dry-run] [--run]
mdatk workflow generate user-script -i NAME=PATH --set source=file --set script_file=SCRIPT \
    --set interpreter=bash|python3|custom [--set command=CMD] --set 'inputs=[...]' \
    --set 'outputs=[...]' --set 'arguments=...'      a custom script step (see Custom scripts)
mdatk workflow restraints --top PARM7 --sequence SEQ --dotbracket DB [--wobble] -o OUT.dist
mdatk pdb check|clean|to-lrna|annotate|qc   (see PDB clean-up and Model checks)
mdatk pdb check|clean|to-lrna|annotate|interface   (see PDB clean-up, Decoy interfaces)
mdatk decoys census|clusters|support   interfaces of an ensemble of models (see Decoy interfaces)
mdatk build helix|mutate|fit       (see Structure builder)
mdatk dict list|smiles|finish|pubchem   (see Restraint dictionaries; only 'pubchem' uses the network)
mdatk rosetta scores SILENT|DIR ... [--term T] [--max T=X] [--tsv OUT] [--extracted DIR] [--legacy-index N]
mdatk rosetta audit ROOT ... [--pair C=T] [--pair-suffix S] [--controlled RANGES] [--allow=-FLAG]
    [--ignore=-FLAG] [--fast-hash] [--shallow] [-o REPORT] [--tsv TABLE] [--runs-tsv RUNS] [--strict]
mdatk frame check REF [MODELS|DIR ...] [--scores TSV [--root DIR] [--max T=X]] --fit RANGES
    --measure [NAME=]RANGES ... [--atoms P,C1',C4'] [--model-map REF=MODEL] [--tsv OUT] [--strict]
mdatk frame reframe OLD NEW [MODELS|DIR ...] [--scores TSV] --fit RANGES --measure RANGES ... [--tol 1]
    [-O DIR] [--manifest TSV] [--scores-out TSV]      (see FARFAR2 decoys)
mdatk ensemble census|compare|regions   (see Decoy ensembles, map-model scores and statistics)
mdatk map score MODEL... --map MAP [--method cc|qscore] [--chimerax EXE] [--batch N] [--q-column qavg|qbb|qsc]
    [--region NAME=RANGES ...] [--reference-region RANGES]   per-residue map-model scores, ranked on dQ
mdatk pdb hybrids OLD NEW --body NAME=RANGES ... [--map MAP]   old/new body combinations
```

`run` and `write --run` exit with status 75 when a step waits for a separately run script (see [Running
steps](#running-steps)); `status` without `--adopt` / `--forget` only reports. The planner commands
share `-p/--project`, `-o/--out-dir` (the plan folder), `--project-name`, `--stamp`, `--no-stamp` and
`-q/--quiet`; `mdatk workflow ACTION --help` lists the rest. Examples are in [Command
line](#command-line) and the sections on the tools.

Common options of the analysis commands: `--top`, `--traj …`, `--frames start:stop:step`,
`--clean/--no-clean`, `--cpptraj PATH`, `--conda-env NAME`, `-o/--out-dir`, `--cache-dir`,
`-q/--quiet`, `-v/--verbose` and `-s/--set KEY=VALUE` (repeatable; values are JSON when they parse as
JSON), for example:

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
analyses   rmsd, contacts, hbonds, mg, stacking, energy, basepairs, use_pbc, proximity_search_radius
visual     structure_groups, trace_atoms, representations[], renderer, rep_rest, align_frames, rotation,
           zoom, center_on, center_follow, color_mode, colours, proximity_set/cutoff,
           glow_mode/color/fade_frames/strength, theme, sizes, label_every, label_residues, title
density    map_path, show, resolution, level, color, opacity, front, fit_frame, search, angular_step,
           auto_fit, transform, anchor, fit_cc, fit_info (the stored placement, its correlation)
layout     preset, panels[] {type, series[], title, options{}}
output     out_dir, cache_dir, video{...}, charts{...}, pdb{...}, write_params_log, timestamp_files
workflow   project_name, stamp, use_stamp, out_dir, step_folders, supersede_runs, parallel, cpu_cores, gpus,
           nodes[] {id, tool, step, title, x, y, settings{}, files{}, adopted},
           edges[] {source, output, target, input}, scratch[] (stand-alone wizard settings)
```

RNAdo writes only the `workflow` section; the analysis window writes the rest and keeps the plan RNAdo
saved (see [One project in both programs](#one-project-in-both-programs)).

Unknown keys are ignored with a warning and missing keys take their defaults, so project files stay
compatible across versions. The recent-projects list is kept in `~/.config/mdatk/settings.toml`.

## Architecture and extending MDATK

```text
mdatk/
  config.py          parameter model, hashing, validation; group renames and references
  problems.py        located problems: field path, GUI page and readable location of a setting
  projectfile.py, tomlio.py   commented TOML project files, the file lock, step log, parameter logs,
                     recent projects
  selections.py      residue ranges, group:<Name> expansion, frame/time specs, elements
  chaingroups.py     residue groups from the chains of a topology
  spatial.py         cKDTree / cell-list neighbour search, Kabsch superposition
  progress.py        progress reporting and cancellation shared by the CLI and the GUI
  cleaning/          cpptraj script generation, streaming execution, cached cleaned outputs;
                     conda environment detection / `conda run`
  io/                trajectory loading, parm7 parser/writer, cache store, external data, PDB export,
                     MRC / CCP4 maps (mrc.py)
  analysis/          registry + single-pass runner; rmsd, contacts, hbonds, mg, stacking, energy,
                     basepairs, structure (trace, proximity)
  secstruct.py       dot-bracket notation and base-pair detection
  data.py            standardised results: series, matrices, sparse residue maps; Dataset
  render/            layouts, panels (seqpanels.py: sequence and arc panels), colouring/glow,
                     orientation, theme, video (FFmpeg), charts, example data for the GUI preview;
                     molecule/ holds the 3D representations (topology roles, geometry per style,
                     surface nets, scene colours, the built-in z-buffer renderer and the optional
                     VTK one)
  density.py         density maps: simulated maps, rigid-body fitting, projected shadows
  pipeline.py        clean -> analyse (cached) -> outputs; shared by GUI and CLI
  pdbtools.py        PDB hygiene (ChimeraX repair, L-RNA detection and names, chains, numbering)
  annotate.py        REMARK 950 base-pair / stacking / dot-bracket annotation
  builder.py         structure builder: ideal and non-canonical helices (D / L), mutations, map fits
  dictionaries.py    restraint dictionaries for L-RNA residues (SMILES, renaming, PubChem lookup)
  citations.py       per-use citations (data in _citation_data.py)
  workflows/         script generators: schema.py (option schemas, shared with the wizard),
                     base.py, registry.py, naming.py, rna.py; farfar2.py, simrna.py, rnamake.py,
                     amber.py (+ mg_params.py), gromacs.py, vmd.py, namd.py (MDFF), buildstep.py,
                     dictstep.py, refine.py (Phenix, Coot), userscript.py (custom scripts), misc.py
                     (notes, existing files, MDATK analysis); plan.py resolves the plan graph and
                     names files; runner.py runs generated scripts on request, within the CPU and
                     GPU budgets of budget.py; runstate.py finds earlier runs; commands.py reads the
                     commands out of a script (the wizard's Commands tab)
    scripts/         self-contained helper programs embedded in generated workflows: prep_pdb.py,
                     cp_retype.py, tip4p2005.py, lrna_lib.py (generated L-RNA library),
                     lrna_names.py (L names before tleap), topcheck.py (topology check),
                     helix_dist.py, amber2gmx.py, gmx_posre.py, namd_mdff.py, namd_energy.py
  cli.py             the mdatk and rnado commands
  gui/               GTK 3 front ends:
    app.py           the analysis window (canvas.py: WYSIWYG figure canvas; problemview.py: problem
                     badges, field flags and the problems list; cursors.py: cursors that show on
                     every platform)
    rnado.py         RNAdo's window (planner.py: the plan canvas; logpanel.py: the output log panel;
                     wizard.py, pdbdialog.py, builderdialog.py: its tools, which use only host.py)
    host.py          what the planner and the tools need from the window that hosts them
    jobs.py          background jobs, progress, Cancel and the run confirmation of both windows
    launch.py        starting RNAdo from the analysis window (source, bundle, AppImage)
    projectui.py     project files in both programs: open menu, file choosers, co-editing
    about.py         About dialogs and BibTeX export; widgets.py, worker.py, desktop.py: helpers
  assets/            RNAdo's logo, wordmark and icons (drawn by packaging/make_rnado_art.py)
  frozen.py          stand-alone builds: bundled FFmpeg, environments of the programs MDATK starts
  demo.py            synthetic RNA system generator
packaging/           stand-alone builds (see packaging/README.md)
tests/               the test suite (see Testing); examples/: example projects; data/: desktop entries
```

**Adding an analysis**: subclass `mdatk.analysis.base.Analysis`, implement `enabled`,
`params_for`, `selection_texts`, `expected_outputs`, `setup`, `process(k, ts)` and `finalize`,
describe the outputs on the returned `AnalysisResult` (`add_series`, `add_matrix`,
`add_pairmap`) and decorate the class with `@register`. Set `group_slots` (and `OUTPUT_LABELS`)
to let panels choose residue groups for it. Its data keys immediately work in panels, charts,
the cache and the GUI. Select through `self.ctx.select(text, field=self.field("selection"))` (or
`ctx.group_atoms(name, field=...)`) so a failing selection names its setting.

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
xvfb-run -a python -m pytest                    # the whole suite; the GUI tests need a display (Xvfb)
python -m pytest tests/test_golden.py           # only the regression test of released planner output
MDATK_TEST_CPPTRAJ=/path/to/cpptraj python -m pytest tests/test_cleaning.py   # with real cpptraj
MDATK_TEST_AMBERHOME=$AMBERHOME python -m pytest tests/test_workflow_integration.py
                                                # run generated Amber scripts with tleap, ParmEd,
                                                # ambpdb, sander, cpptraj (add MDATK_TEST_GROMACS=1
                                                # with gmx on PATH for the GROMACS conversion + run)
```

Run the suite with the Python MDATK is installed in (`python` in the venv; for example `xvfb-run -a
python3.12 -m pytest` with a system Python 3.12). Without a display the GUI tests (both windows) are
skipped, and tests that need cpptraj, AmberTools or GROMACS are skipped without them.

- **Optional AmberTools tests.** `MDATK_TEST_AMBERHOME` names an AmberTools installation:
  `tests/test_workflow_integration.py` runs the generated Amber chain end to end (sander standing in
  for pmemd), `tests/test_lrna_library.py` builds a D duplex and its mirror image with tleap and
  compares sander energies term by term for OL3 and OL3 + CP, and `tests/test_workflows.py` retypes
  AmberTools' `RNA.lib` for CP. The last two also take `$AMBERHOME` (the library test also the `tleap`
  on PATH).
- **Golden regression test.** `tests/test_golden.py` regenerates, with the current code, every file the
  planner writes and every planner message for every template, every step type (defaults, then each
  check box flipped and each radio / combo choice, one at a time), typed secondary structures and the
  example projects, and compares their hashes with `tests/data/golden_v0.0.7.json`, recorded from the
  0.0.7 release (`python3 tests/golden.py --write OUT.json` with that release importable). Only the
  "written by MDATK x.y.z" version and dates are normalised. A step type of 0.0.7 must not change its
  output unless the change is intended; the intended changes are listed in the test with their reasons
  (see [CHANGELOG.md](CHANGELOG.md)).
- **Shipped example.** `examples/namd_mdff` must be what `python3 tests/mdff_system.py --example
  examples/namd_mdff` makes now, apart from the date in script headers.

The suite covers exact geometric criteria of every analysis on hand-built systems, cache keys
and reuse, cpptraj script generation (with a stand-in executable, and with real cpptraj when
available), the full pipeline including FFmpeg output, the CLI and GUI smoke tests. For the 3D
view it checks the rasteriser against analytic spheres and cylinders, closed meshes without
pinholes, depth order beyond the panel's range, registration and handedness in the final figure,
fog, glow and translucency, thread safety and a render without a display. For RNAdo it covers the
script generators and their checks with stand-in executables, the plan, previous runs, custom scripts,
the run machinery (parallel runs, waiting steps, Cancel), the output log panel under a flood of lines,
how RNAdo is started, both programs saving one project, both programs ending with their windows and the
programs they start inheriting no AppImage descriptors, and the stand-alone packaging (spec, AppRun,
desktop entries, assets). The menu entries (`tests/test_launchers.py`) are checked for every way MDATK
runs, through GLib's own parser and a real launch of a program whose path has a space and a `$`, and
installed, updated and removed in a temporary data folder next to files MDATK did not write.

## Notes and limitations

- MDATK produces finished files; there is no interactive playback or frame scrubbing (the analysis
  window's preview shows example data, or one real frame, for arranging the layout).
- Interaction energies are non-bonded pair sums within a cutoff (no PME / long-range
  electrostatics, no solvent); use them for trends and comparisons against a target value.
- Without a parm7 topology, Lennard-Jones parameters are approximate element defaults and the
  result metadata records this (`lj_source`).
- When fitting, cpptraj rotates the unit cell, which NetCDF cannot store, so the box is removed
  after fitting by default (`cleaning.remove_box_after_fit`).
- Very large systems: restrict groups and the structure selection to the region of interest;
  only selected atoms are processed per frame and cached results are compact (sparse maps,
  float32 traces).
- RNAdo's planner and script wizard write files; they run them only on request (*Run all…*, *Run
  selected steps*, *Run step…*, *Write and run…*). Generated scripts call the programs by name
  (`tleap`, `parmed`, `pmemd.cuda`, `rna_denovo`, `gmx` …); use the *Environment* section of each
  step (conda environment, `module load`, `source amber.sh`) when they are not on PATH. Scripts get
  your own `PYTHONPATH`, also when the analysis window started RNAdo.
- 3D representations are made for nucleic acids. Protein chains get a round tube through CA in
  cartoons and ribbons (no helix / sheet detection), and licorice, VDW, surface and trace work for
  any residue. PDF / SVG previews embed the 3D layer as an image. Surfaces are rebuilt every frame
  (tens of ms for a few thousand atoms; the grid coarsens to 1.5 Å above 5 000 atoms).
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
