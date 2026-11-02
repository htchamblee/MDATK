# Changelog

All releases of MDATK, newest first. The dates are those of the version commits. There was no 0.0.9:
0.0.10 followed 0.0.8. Since 0.0.8 a regression test (`tests/test_golden.py`) compares the files that
the step types of 0.0.7 write with the 0.0.7 output; the intended changes are listed in the test and in
the notes below.

- [0.0.11](#0011-2026-10-31) · [0.0.10](#0010-2026-10-30) · [0.0.8](#008-2026-10-29) ·
  [0.0.7](#007-2026-10-25) · [0.0.6](#006-2026-10-25) · [0.0.5.1](#0051-2026-10-25) ·
  [0.0.5](#005-2026-10-25) · [0.0.4.1](#0041-2026-10-22) · [0.0.4](#004-2026-10-21) ·
  [0.0.3d](#003d-2026-10-20) · [0.0.3c](#003c-2026-10-18) · [0.0.3b](#003b-2026-10-18) ·
  [0.0.3a](#003a-2026-10-18) · [0.0.2c](#002c-2026-10-16) · [0.0.2b](#002b-2026-10-15) ·
  [0.0.2a](#002a-2026-10-15) · [0.0.1](#001-2026-10-12)

## 0.0.12 (unreleased)

### Model checks

- **`mdatk pdb qc MODEL.pdb …`** (`mdatk/modelcheck.py`, standard library only): clashes as overlap inside
  Bondi contact (0.4 Å), N/O pairs with a donor counted as hydrogen bonds from 2.2 to 3.2 Å (donors by
  base, terminal hydroxyls, protonated A⁺ / C⁺ on request; two acceptors never), carbon contacts beside
  an H-bond excused only when no closer than the H-bond, exclusions along bonds (up to three, four for
  hydrogens) instead of residue numbers, every chain and neighbouring residue checked, waters and ions
  skipped or (ions) checked against first-shell distances, hydrogens optional; O3′–P links within every
  TER segment (1.4–2.0 Å) and P–P spacing; loop compaction for typed ranges or the residues unpaired in
  every given dot-bracket; sugar hands from the geometry against expected ranges, epimers named; a
  decoy table (`--scores`: run, tag, pdb, score terms) for report columns, limits and ranking; gates with
  `--on-fail report|filter|stop`, text, `--tsv` or `--json` reports, copies of the passed and best models.
- **Planner step *Model check (clashes · backbone · chirality)*** (Structure prediction): models (several),
  optional secondary structures and decoy table in; a report table, the passed models and the best one
  out. Expected hands are measured on the first model when the plan is written (or typed, or none); range
  options take `group:NAME`. The script runs a copy of the PDB tools and the model checks (`python3`
  only) and removes the outputs of an earlier run before it starts.
- **`mdatk pdb check`**: `--expect-l` / `--expect-d RANGES`, `--strict` (status 1 for a residue of the
  other hand, an unmeasurable expected one, or an epimerised, contradicting or flat centre) and `--table`
  (per residue: the three-centre call, the C4′ hand alone, the volumes, the expectation, a note).
- **PDB tools**: `infer_base` (the base from the atom names, for residue names that are not
  recognised) and `nucleotide(..., infer=True)`; `--strip-h`, `--drop-virtual` (Rosetta's VRT / XXX) and
  the scheme `standard` (plain A C G U for both hands, every record ATOM, with a warning that the names no
  longer record the hand), so `mdatk pdb clean --scheme amber --strip-5p --strip-h --drop-virtual`
  turns a Rosetta decoy into a tleap input and `--scheme standard` makes the copy for chirality-blind
  programs; the record policy `atom`; ranges accept `*` for every chain where hands are expected.
- **Amber build**: *Remove Rosetta virtual residues (VRT/XXX)* in the PDB preparation, on by default
  (tleap cannot type them, so only builds that failed change), and *Check O3′–P links before tleap*
  (off, warn or stop; off by default), also measured on an existing input PDB when the plan is written.
  Generated files are unchanged apart from the two helpers that copy library code: the PDB tools and the
  PDB preparation (listed as intended in `tests/test_golden.py`).
- **FARFAR2 decoy table and score filter** (step *Rosetta · FARFAR2 models*, `mdatk rosetta scores`): SCORE
  terms are looked up by column name in the header that applies (`mdatk/workflows/scripts/silent_scores.py`,
  standard library, embedded in the step's folder). New output *scores* (file kind *table*):
  `<project>_<step>_scores_<time>.tsv` with `run` (`rNN` from the silent file's name, `NN-rNN` when two
  FARFAR2 steps feed the step), `tag`, `pdb` (plan-relative, empty when not extracted) and every SCORE term,
  one row per decoy, written with *Write the decoy table*, a filter or *Extract every model*;
  `mdatk.rosetta.read_decoy_table` reads it (TSV or CSV, extra columns kept). *Filter* section: a term
  (`linear_chainbreak`) and a maximum (5), applied before clustering (filtered silent files per run) or
  after it (ranked centres skipped). *Extract every model of every run*: `models/<run>/<tag>.pdb`, each run
  in a `mktemp` folder of its own, earlier models of the run removed first, and a check that every run has
  one PDB per SCORE line. All new options are off by default; the step's files are unchanged without them.
- **Run audit** (`mdatk rosetta audit`, step *Rosetta · run audit*): what each rna_denovo run was given,
  from the command lines in its logs, with `@flags` files expanded recursively, `::` read as `:`, short and
  long flag spellings folded, every command line of a log read (restarts), input files fingerprinted by
  their whole content (same content under MDATK's step-numbered names is no difference), conditions
  compared and checked in pairs (`--pair`, `--pair-suffix`, `--allow`, `--ignore`), and, with
  `--controlled RANGES`, how each condition gives the pairing of those residues (secondary structure,
  obligate pairs, `-s` / `-input_res`, chain by chain). Report (text) and a flags-by-condition table.
- **Frames of multi-body models** (`mdatk/frames.py`, `mdatk frame check|reframe`, step *MDATK · frame
  check*): fit on one body, measure one or more others (P, C1′, C4′ by default; chain-qualified ranges or
  residue groups; atoms keyed by chain, residue number and insertion code); one frame when the 95th
  percentile of every body stays below 1 Å, models without the fit body listed but not counted; a re-fitted
  reference applied to the models when one rigid transform suffices (carried RMSD ≤ 1 Å), otherwise the
  step fails; moved decoys written as `<run>/<tag>.pdb` with ANISOU rotated, a manifest and a decoy table
  of the moved files; a warning for models sampled with the map in the score. Models come from PDB files
  or a decoy table's `pdb` column.
- Bugs of the user scripts these replace, fixed: runs of MDATK-named silent files collided on the stamp;
  a missing score term crashed the audit; duplicate tags overwrote each other silently; problems went to
  standard error only; extracted decoys of earlier runs stayed beside new ones and failed extractions went
  unnoticed; `::` flags and `@flags` files hid confounds; `-r` could not be switched off; relative paths
  were resolved for the first run only; flag aliases and restarts were missed; directory keys depended on
  the working folder; inputs were sampled, not hashed; atoms and output files were keyed without the chain
  or the run (same-named decoys of different runs overwrote each other or shared a label); a set-wide
  median let models without the fit body into the statistics; unusable models vanished from the table;
  ANISOU records were not rotated.

## 0.0.11 (2026-10-31)

### RNAdo, a program of its own

- The workflow planner, until now the *Project Planner* of the analysis window, becomes **RNAdo, MDATK's
  workflow planner**, a program of its own (`mdatk/gui/rnado.py`). The **script wizard**, the **PDB
  clean-up** and the **structure builder** moved with it from the analysis window into its header bar.
- RNAdo's window holds the plan canvas with its toolbar and details pane, the output log panel (below)
  and a status bar with the planner's status line, the progress bar and *Cancel*. Header bar: *Open*
  with the recent projects, *Save*, the three tools; its menu has *New project*, *Open project…*,
  *Save*, *Save as…*, the tools, *Show the output log*, *Project log…* and *About RNAdo*. Shortcuts:
  Ctrl+N, Ctrl+O, Ctrl+S, Ctrl+Shift+S, Ctrl+Shift+W (wizard), Ctrl+L (log panel).
- RNAdo opens, saves and remembers the same project files (`*.mdatk.toml`) as the analysis window and
  runs without it. Closing RNAdo during a run asks first and stops the scripts.
- The planner, the wizard, the PDB clean-up and the builder use only a small host interface
  (`mdatk/gui/host.py`) and no longer know the analysis window. Background jobs and the run
  confirmation (`mdatk/gui/jobs.py`) are shared by both windows; the runner can mark every line with
  its step and pass standard error apart (command-line output unchanged).
- **Output log panel** (`mdatk/gui/logpanel.py`) at the bottom of RNAdo's window: the planner's
  messages (files written, previous runs checked, runs used or dropped) and every line of the scripts
  it runs, each marked with its step (`[03] …`), with start, finished, failed, cancelled, waiting and
  not-started lines. Standard error is red, lines starting with ERROR (and failed steps) bold red,
  lines starting with WAITING (and waiting steps) amber. *Step* filters by step, *Last run* shows the
  stored log of the selected step's last run (from the project's step log); the panel follows new
  lines, copies, saves or clears them, folds away (Ctrl+L) and unfolds for a run. Lines are inserted in
  batches, the panel keeps the last 5 000–6 000 lines, and a script that prints faster than the panel
  can show gets lines skipped, with a note of how many, so the window stays responsive.
- **Starting RNAdo:**
  - `rnado [PROJECT] [--open wizard|pdb|builder]` and `rnado-gui` (the same without a terminal), new
    entry points of the source installation; `rnado ACTION …` is `mdatk workflow ACTION …` (`tools`,
    `check`, `write`, `run`, `status`, `generate`, `template`, `restraints`);
  - `mdatk rnado …` (the same as `rnado …`), `mdatk gui --rnado` (alias `--planner`) and
    `mdatk gui --open TOOL`, `python -m mdatk.gui.rnado`;
  - in the analysis window, the header-bar menu *RNAdo*: *Open in RNAdo…* (Ctrl+P), *Script wizard /
    PDB clean-up / Structure builder: open in RNAdo…* (Ctrl+Shift+W for the wizard). It saves the
    project first when it has unsaved changes (it asks), then starts RNAdo on the project file
    (`mdatk/gui/launch.py`).
- **The plan's files for the analysis:** *Use the plan's files* (analysis window, Input & cleaning;
  *Use the plan's files as inputs* in its *RNAdo* menu) takes the topology and trajectories that the
  plan's MDATK analysis step reads, from the plan saved in the project file; with several analysis
  steps it asks which. *Use for analysis…* on such a step in RNAdo saves the plan for it.
- **Both programs on one project** (`mdatk/gui/projectui.py`): RNAdo writes only the file's plan (the
  `workflow` section, `projectfile.edit_workflow`) and leaves the analysis settings as the file has
  them; the analysis window saves the plan it finds in the file with its own residue-group renames
  applied, and RNAdo follows those renames in its unsaved plan. Each program notices when the other
  saved (focus, file monitor), takes the other's part, reloads its own when it has no unsaved changes
  there, and otherwise asks in a bar above its content.
- Generated workflow files are the same as in 0.0.10, apart from the version in their headers.

### Logo, icons and stand-alone builds

- RNAdo's logo, wordmark banner and icons (`mdatk/assets`: `rnado-logo.png`, `rnado-wordmark.png`,
  `rnado-icon.svg` and PNGs from 16 to 512 px, 0.8 MB in all) are drawn by
  `packaging/make_rnado_art.py` from a real A-form RNA model: the ideal duplex of `mdatk.builder`,
  read back with `pdbtools`, drawn as P-atom ribbons and C4′–C1′–N1/N3 rungs with beads. The helix
  keeps its geometry: 11 bp per turn, right-handed, pairs 5 Å off the axis and inclined, a narrow, deep
  major groove (11.3 Å) and a wide, shallow minor groove (19.6 Å); the tornado funnel only tapers the
  radius (twist, rise and grooves kept). Reds, red-orange highlights, white and black; the wordmark's
  letters are outlines of Inter Black Italic (SIL Open Font License 1.1, no font file shipped). The
  script is seeded and reproducible; the images are committed, so neither the GUI nor the builds draw
  anything. The wordmark heads the README.
- The stand-alone builds have RNAdo as a second program: `rnado-gui` (windowed; `rnado.ico` on Windows)
  from `packaging/entry_rnado.py`, with `mdatk/assets` bundled. AppImage: `AppImage rnado ARGS` and a
  link named `rnado` start it; the MDATK desktop entry has an *RNAdo (workflow planner)* action, and
  `rnado.desktop` and the hicolor icons are in the AppDir. Windows installer: Start-menu entry *RNAdo*,
  an optional desktop shortcut and an optional launch. `data/rnado.desktop` for source installations;
  `assets/*` is package data. `packaging/smoke_test.py --rnado` starts RNAdo, and the release workflow
  runs it on the Linux jobs.

### Fixes from the release review

- Co-editing:
  - the analysis window's *Save* asks its questions first, then re-reads the file, replays its
    residue-group renames on the plan RNAdo saved and writes, all under one file lock (re-entrant
    within the process). A plan RNAdo saves while *Replace them?* is open is kept; when the analysis
    settings change again meanwhile, the question is asked again;
  - RNAdo's save re-reads the file, takes newer analysis settings, writes the plan and records the
    file state under one lock, so the state it keeps is the one it merged with and wrote;
  - *Reload them* in the analysis window's change bar loads the file as it is: the window's pending
    group renames go with its other unsaved edits instead of being replayed on the plan;
  - RNAdo works out the group renames from the old and new group lists when it takes new analysis
    settings, follows them in its unsaved plan, in the plan last read and in the saved one, and reloads
    the forms of open wizards. A plan changed only by those renames is no conflict;
  - *Save* without edits in the analysis window, when another program saved newer analysis settings,
    reloads them ("Nothing to save") instead of writing the older copy back; *Save as* gives the copy
    the newest settings;
  - *Use the plan's files* keeps the file state, the pending change bar and the settings it offers; a
    save after a shown bar replaces those without asking again, newer ones are asked about;
  - opening a script wizard (stand-alone or a step's) and choosing a tool no longer marks the plan as
    changed, although the wizard fills in defaults.
- Launch and runs:
  - scripts and programs that RNAdo runs get **your own `PYTHONPATH`**: *Open in RNAdo…* from a source
    checkout puts MDATK's folder on RNAdo's `PYTHONPATH` only when it is not a site-packages folder and
    records your value (`MDATK_ORIG_PYTHONPATH`), which `frozen.child_env` gives back, so another Python
    a step runs (ParmEd, a conda environment) no longer imports MDATK's packages;
  - the **output log panel is bounded**: a flush inserts at most 2 000 lines, the newest, with a note
    per step of how many were not shown, and a flooding script pauses 1 ms per 2 000 lines. `seq 1
    1000000` run from RNAdo stalls the window for about 0.1 s at most (was 24 s), and *Cancel* is
    handled on time;
  - **AppImage `rnado` actions**: `AppRun` runs `mdatk rnado …` for a first argument `rnado` and for
    links named `rnado*`, so `AppImage rnado write|run|status|tools …` work; the window opens in that
    process with the program name `rnado`, so its WM_CLASS is RNAdo's;
  - **how RNAdo is started** follows how MDATK runs: the AppImage again only when MDATK itself runs
    from it (its executable lies in `$APPDIR`), with `APPIMAGE_EXTRACT_AND_RUN=1` (and `NO_CLEANUP=1`,
    so RNAdo leaves the shared extracted folder in place) when the AppImage runs extracted, without
    FUSE; otherwise the `rnado-gui` next to the program, or `python -m mdatk.gui.rnado`. The start is
    watched for 3 s: an error status is shown in the analysis window instead of "Started RNAdo", and a
    window closed meanwhile is left alone;
  - **run tokens**: the planner gives a run its token only once the run has started, so a refused
    *Run step…* no longer orphans the markers and spinner of the run in progress.
- The analysis window cancels its pending settings check when it is destroyed (Gtk-CRITICAL warnings).
- Tests: the example test (`test_example_generator`) compares the shipped `examples/namd_mdff` without
  the date in script headers, so it **no longer fails on every day after the example was generated**;
  the test of chain reads replaced while running waits for the read to start.
- `examples/namd_mdff` regenerated (version in the headers).
- Help texts, docstrings and project-file comments say RNAdo for the planner: `mdatk workflow` is "the
  plan of RNAdo, MDATK's workflow planner, and its script generators", and the `workflow` key of a
  project file is commented "RNAdo's plan and script wizard".

## 0.0.10 (2026-10-30)

### L-RNA names before tleap; Windows cursor

- With an own L-RNA library, the Amber build script checks the L residue names **before tleap runs**:
  a standard-library helper (`scripts/lrna_names.py`) reads the L names of the renamed PDB, the units of
  every library the tleap input loads before `loadpdb` (a name in leap's search path is looked up in
  `$AMBERHOME/dat/leap`) and its `addPdbResMap` lines (also over several lines, comments ignored), and
  stops on a name that is neither a unit nor mapped, which tleap would read as the lanthanum or
  lutetium ion (LA / LU). A library it cannot read gives a warning. The planner uses the same parser
  and warns about the names even when an earlier step makes the structure.
- These checks come only with an L-RNA set-up: *My own files* under *L-RNA library*, a structure known
  to have L residues (*Force L for*, an L helix, an input PDB with L sugars), or an own library that
  looks like an L-RNA one (atom types LD / LH / LE, or nucleotide units named the L-RNA ways: LA…,
  A0…, 0A…). D builds with, say, a ligand library, and modified-nucleotide D libraries such as
  `all_modrna08.lib`, are written as before (none of AmberTools' 86 libraries triggers the check;
  Dudek & Trylska's `nucleic12_LRNA.lib` does, and passes).
- In the Windows build the mouse pointer no longer disappears over the structure panel: GDK there maps
  only some cursor names to system cursors and the build leaves the cursor theme out, so
  `mdatk/gui/cursors.py` asks for cursor types with Win32 system cursors (elsewhere the named cursors,
  with the type as fallback).

### Previous runs and separately run production

- **Check previous runs** (planner toolbar, right-click menu, details pane, `mdatk workflow status`)
  finds the outputs of a step's earlier run, also under an earlier time stamp (each iteration of a
  project may get a new one), and marks the box **done**, **partial** or **missing**. Only the files
  the step's scripts make count; Amber MD stage outputs must also end with pmemd's closing lines, so a
  run in progress is partial (`mdatk/workflows/runstate.py`).
- A step can then **use those outputs** (*Use this run*; `adopted` in the project file): the steps after
  it read them, and it is not written or run again (the master script and the summary say so). *Stop
  using it* undoes it; *Run selected steps* / *Run all…* ask whether to skip such steps; the planner
  warns when a file of the run disappears. `mdatk workflow status [--step N] [--adopt | --forget]
  [--json]`; `workflow run` skips steps that use an earlier run and says so.
- A step whose production runs in a separate script no longer reports success before that script has
  finished. Before 0.0.10 the main script of an Amber MD step whose stages were all production held
  only the preamble and "succeeded" without running anything. Now the main script runs the stages
  before production (for GROMACS also the conversion) only while production has not finished, and
  **waits** (exit status 75, naming the script to run); the steps after it do not start, and
  `mdatk workflow run` exits with 75. Once production has finished, the stages after it and the dry
  PDB run and the step succeeds. A main script written by an older version is treated as waiting.
- `mdatk workflow run / write --run --include-separate` and a check box in the run confirmation run the
  separate scripts too, each right after its step's main script (not again once finished).

### Residue groups and located errors

- **Residue groups follow their names and the topology.** Renaming a group renames it everywhere the
  project uses it (analysis groups, selections, panels, charts, representations, PDB export, planner
  steps; `group:Older` and `group:Old_2` stay; notes are never rewritten); empty, invalid and taken
  names are refused. Deleting a group that is still used says where. Choosing a topology fills empty
  or untouched default groups with one group per chain (`mdatk/chaingroups.py`), and *Groups from
  chains* adds the chains no group covers yet. `Project.rename_group` and `Project.group_references`.
- **Selection and group errors say exactly where they are.** Every problem names its setting
  ("Analyses › Hydrogen bonds › Group B: unknown residue group 'Tagret' (defined: Aptamer, Target);
  did you mean 'Target'?"), shows MDAnalysis' own message next to the selection as written and as
  expanded, and says which group or part of a selection matched no atoms (`mdatk/problems.py`:
  `Problem`, `SelectionError` with field path, page, location and token; `Project.problems()`).
- The GUI checks the settings as you type: red count badges on the page buttons, flagged fields with
  the message as tooltip, tinted table cells, and a problems list that jumps to the field.
- **Only what is about to run can block it**: a wrong setting is an error if the run uses it and a
  warning otherwise. The structure view's settings are errors only while the view is rendered, so a
  0.0.7 project with a stale proximity set renders charts and PDBs with the video off again. A group's
  own definition is an error only when something about to run selects it; circular definitions are
  reported at every group of the circle; a failing run on a panel's or chart's groups is reported at
  that panel or chart; a PDB atom subset of only commas is a located error instead of a crash.

### Custom scripts

- **Custom scripts as planner steps** (`user-script`, *Custom*): type a bash or Python script into a
  step, or pick a script file, declare the files it reads and writes, and connect them to other steps
  like any program's files. MDATK copies the script into the step folder, passes the file names as
  `MDATK_IN_<NAME>` / `MDATK_OUT_<NAME>` and `{in:…}` / `{out:…}` arguments, and stops the step when a
  declared output is missing (`mdatk/workflows/userscript.py`).
- The run script removes the declared outputs before the script starts and writes a stamped completion
  marker last, so a stale or partial output never passes; a fixed-name output belongs to the stamp
  whose marker is there. Errors for two steps that write the same output file and for an output that is
  one of the step's own inputs; plain file names may not start with `+`; an optional input with
  several connections and no file expands to no word.
- Wizard: taller boxes for multi-line options (`Option.rows`); the *Files* page follows input ports
  that change through a record list.

### Other fixes

- Chain groups: a chain read or topology timer no longer reaches a project loaded since; renames and
  chain fills retitle only panels with automatic titles (default titles such as "RMSD" stay); *Groups
  from chains* only adds groups; group lists with empty items are renamed and found. VMD keeps chain
  qualifiers and translates chain-group selections, so chain groups colour their own chain.
- `examples/namd_mdff` regenerated (headers, `adopted` field).

## 0.0.8 (2026-10-29)

### MDFF in NAMD on the Amber topology

- New planner step **NAMD · MDFF (Amber topology)** (`namd-mdff`, category *Refinement*): MDFF in NAMD
  on the Amber topology. NAMD reads the parm7 as built (`amber on`, `parmfile`, `ambercoor`), so CP,
  NBfix19, 0BPh, the L-RNA types and dihedrals, the Mg²⁺ pair terms and HMR are kept. A sander / NAMD
  energy comparison proves it before fitting; a mismatch stops the script.
- Solvated (periodic, PME, LJ correction, NVT; default) or vacuum (dry topology, optional GBIS) runs;
  stages: minimisation, MD with a ramp of the map coupling, MDFF and a final minimisation with strong
  coupling. Finished runs are skipped while their inputs are unchanged.
- Its chirality and base-pair restraints **keep L-RNA sugars L**; it never uses VMD's D-only
  stereochemistry tools. The secondary-structure dihedral restraints of the MDFF paper (Trabuco et al.
  2008) hold the torsions of base-paired residues; base pairs are flat-bottom `extraBonds` walls. All
  restraint files are NAMD `extraBonds` files.
- Preparation helper `scripts/namd_mdff.py` (standard library only): reads parm7 / rst7, MRC / CCP4 and
  OpenDX maps, finds the map's frame after tleap moved the solute, writes the potential (cropped and
  tested against the periodic cell exactly as NAMD tests it), the coupling PDB, the restraints and the
  cell; the final model returns to the map's frame with the chains and residue numbers of the
  map-frame model.
- Template *Model + map → Amber build → NAMD MDFF* (rectangular box; NAMD fits the potential into a
  truncated octahedron only with a small margin, and the planner warns about one).
  `examples/namd_mdff/` holds a D + L test case.
- Citations: NAMD (Phillips et al. 2020); MDFF (Trabuco et al. 2008) verified from the paper.

### Generated L-RNA library

- The Amber build can **generate the L-RNA library** from the force field in use (OL3 or OL3 + CP;
  option *L-RNA library*: automatic, generated or own files). The D units are mirrored and named LA LC
  LG LU (LA5 / LA3 / LAN …) inside and out, O5′ and O4′ get their own types (LD, LH; LE with CP), and
  every term of their D twins is repeated with dihedral and improper phases mirrored. A D duplex and
  its mirror image are tested to the same energy. The generator refuses odd-phase terms without a
  twinned type and the generated library with an own CP residue library; *Automatic* leaves own L-RNA
  set-ups alone, so 0.0.7 projects build as before.
- The topology check counts the residues carrying the L types against the L residues of the PDB that
  the final tleap read: D units loaded for L residues stop the build (tleap really loaded L units).
- **Lanthanide trap:** a library whose units are named otherwise (A0 / C0 …) and no `addPdbResMap`
  made tleap build LA and LU as lanthanum and lutetium ions without any warning. The planner reads the
  extra libraries' unit names and the map lines and reports every unmapped L name as an error with the
  map lines to add; the topology check stops on a residue named as a nucleotide with fewer than 10
  atoms, now also for dry L-RNA builds.
- `namd-mdff`: *L residues* declares the L residues by topology number for libraries that write D
  names into their units.

### Regression test and examples

- Steps that existed in 0.0.7 write the same files as before: `tests/golden.py` records every file the
  planner writes and every planner message for every template, every step type (each option changed
  one at a time), typed secondary structures and the example projects (193 cases), and
  `tests/test_golden.py` compares them with the 0.0.7 output. Intended changes: the topology check
  (and the dry L-RNA build script that now runs it).
- `examples/rna_plan.mdatk.toml` and `examples/aptamer_target.mdatk.toml` had been committed empty in
  0.0.4 (and shipped empty since); restored from 0.0.3d.

## 0.0.7 (2026-10-25)

- Independent steps can **run at the same time** (*Parallel* in the planner toolbar, 1–64, saved as
  `workflow.parallel`, default 1; `mdatk workflow run / write --run --parallel N`, checked before any
  file is written). Each step starts when the steps it reads from (through the planner's connections)
  have succeeded; plan order otherwise.
- A failure stops the steps that depend on it; independent steps carry on. Cancel or Ctrl-C stops every
  running script; the final error names each failed script and why. An error of the output itself
  (e.g. `| head` closing the pipe) ends the run instead of hanging. Output lines carry the step number
  when scripts run side by side; the command line writes each line in one piece.
- A box shows an **animated marker** while its script runs: a turning arc, a dashed ring while it waits,
  then a tick, a cross or a dash until the next run. Marks of a run started before a project reload, or
  after the planner closed, are ignored.

## 0.0.6 (2026-10-25)

- The Project Planner can **run a selection of steps**: tick the boxes (check box in the title bar,
  Space on the selected box, the right-click menu, *Tick all steps* / *Untick all steps*), then *Run
  selected steps (N)*: it writes the ticked steps' files and runs their main scripts in plan order,
  whatever order they were ticked in. Notes and the MDATK analysis box have no check box; the ticks are
  not saved. *Run…* is renamed *Run all…*.

## 0.0.5.1 (2026-10-25)

- *Separate HO and O2 Lennard-Jones types first* is on by default for new projects: tleap merges HO with
  water HW, so the NBfix19 HO pairs otherwise apply to every water hydrogen. Projects saved earlier keep
  their setting; turned off, the planner warns.
- On AmberTools 26 the `MG` unit carries type `Mg2+` with charge +2 (checked on a built nanoMg / TIP3P
  topology), so earlier versions applied the published ion terms too; tleap's "MG has a charge of 0"
  message is misleading.
- Two-pass builds leave Mg²⁺ lines under extra ParmEd commands out of the dry pass.
- Topology check: solvent is classified by residue, never by atom type name; the log lists the
  Lennard-Jones types shared by solute and water atoms.

## 0.0.5 (2026-10-25)

- The Amber build **checks the finished topology** (*Check the final topology*, on by default) and stops
  when a box meant to be neutral is not, fewer Mg²⁺ than requested are present, a Mg²⁺ does not carry
  +2, or (microMg / nanoMg) the ion lacks its published Lennard-Jones terms or a solute atom type has no
  Mg²⁺ pair term.
- L-RNA and other non-standard atom types get the Mg²⁺ pair terms of their D-RNA twins (*L-RNA atom
  types*: LD and LE as OS, LH as OH; *Extra Mg²⁺ pair types* for any library). Before 0.0.5, with
  microMg / nanoMg, the combination rule bound Mg²⁺ to the oxygens of an L-RNA library several times
  too strongly.
- The microMg / nanoMg frcmod defines the ion's terms for both type names in use (`Mg2+` and `MG`).
- The L-RNA fixes (atom types, NBfix19 / 0BPh and Mg²⁺ pairs) no longer depend on MDATK naming the L
  residues: an input PDB may already have L names.
- The Mg²⁺ block only adds `changeLJPair` lines, so pair terms set earlier under extra ParmEd commands
  are not reset; planner errors and warnings for types without a reference.
- Documentation: the plan for a future C++ version (`docs/ARCHITECTURE_CPP.md`) updated.

## 0.0.4.1 (2026-10-22)

- The Linux AppImage is built on Rocky Linux 9 (glibc 2.34) with Rocky's Python 3.12 and GTK 3; the 0.0.4
  build needed glibc 2.35 and did not start on RHEL 9. The glibc check now fails the build.
- The AppImage carries the Wayland client libraries in `_internal/fallback` and uses them only on hosts
  without them (minimal or server installs).
- Smoke tests on a clean Rocky Linux 9 and on Ubuntu 22.04; the Windows build keeps MDAnalysis' C
  diagnostics as warnings for GCC 14+. Releases get files only when every build and test passed.
- Planner: selecting a step no longer scrolls the canvas back to the top.
- Citation of PDB 4JRD (Safaee et al. 2013) verified against the paper.

## 0.0.4 (2026-10-21)

- The structure builder also makes **parallel and Hoogsteen-paired duplexes and triplexes**:
  reverse Watson-Crick, Hoogsteen and Hoogsteen/Hoogsteen (4JRD-type) parallel duplexes,
  reverse-Hoogsteen antiparallel duplexes, and triplexes with a Hoogsteen (U·A-U, C+·G-C) or
  reverse-Hoogsteen (A·A-U, G·G-C) third strand; D or L, mismatches reported, C+ written as C.
- PDB clean-up: `--strip-5p` (*Remove 5′-terminal phosphates*) for Amber.
- MDATK runs as a **stand-alone AppImage or Windows program** without Python (PyInstaller, with Barnaba
  and FFmpeg bundled; the Windows build is a beta).
- Value fields (spin buttons, drop-downs, sliders) no longer change when the mouse wheel scrolls past
  them; the canvas keeps wheel zoom.

## 0.0.3d (2026-10-20)

- The Amber build declares the L-RNA library's atom types (LD, LH) for tleap and adds the NBfix19 / 0BPh
  pairs for its bridging-oxygen type, as in the validated hand-written L-RNA ParmEd input; ParmEd skips
  them without L residues.
- Citations of Arnott 1972, Phenix 2019, Coot 2010, AceDRG 2017, eLBOW 2009, Afonine 2018, Casañal 2020,
  CCP4 2023, CCD 2015 and PubChem 2023 verified against the papers.
- `docs/ARCHITECTURE_CPP.md` and `LICENSING.md`: the plan for a future C++ version and its licensing.

## 0.0.3c (2026-10-18)

- MDATK **builds starting structures** (`mdatk build`, the structure builder window, a planner step):
  ideal A-form helices of either chirality from the Arnott fibre monomer, base mutations with
  compensatory partners, rigid-body fits into a map.
- It writes **restraint dictionaries** for the L-RNA terminal residues Coot and Phenix do not know
  (acedrg / phenix.elbow from built-in L SMILES, renamed to the model's atom names; `mdatk dict`;
  PubChem SMILES only on explicit request).
- Planner steps for **real-space refinement** with Phenix 1.21 / 2.x and Coot 0.9 / 1.x (batch), with
  checks for names, terminal variants and dictionaries; templates *ideal helix → Amber MD* and *model +
  map → dictionaries → Phenix → Coot*.
- microMg is available with TIP4P-D as well (the authors' parameter-file values; N. Schwierz confirmed
  that the paper tables carry transcription errors).
- `ccd` scheme: a 5′ end that keeps its phosphate keeps the CCD component name (0G).

## 0.0.3b (2026-10-18)

- MDATK finds **base pairs** in every *n*-th frame (builtin canonical-pair detection or the optional
  [Barnaba](https://github.com/srnas/barnaba)) and shows them in a **secondary-structure arc panel**; a
  **sequence panel** spans the width of the video; both glow and colour with the structure view.
- The structure view can be **zoomed and centred** on a residue or the origin, saved in the project
  (mouse-wheel zoom in the GUI).
- `mdatk pdb annotate` writes base pairs, stacking and dot-bracket into a PDB file as REMARK 950
  records.
- The real-frame preview shows real residue names, chain ids and base pairs.

## 0.0.3a (2026-10-18)

- A project is **one commented TOML file** (`*.mdatk.toml`) holding the settings, the planner workflow,
  notes and a log of every step that ran (moved to `<name>.log.toml` beyond 50 MB); JSON presets are no
  longer read. Generated scripts can be run on request, each run logged in the project file.
- MDATK repairs and renames **PDB files** (ChimeraX damage, L-RNA detected from the sugar chirality,
  Amber / CCD / Rosetta names, chain ids, numbering, SEQRES); this replaces `to_lrna.sh` in the Amber
  build.
- Every script, run log and the *About* dialog **cite** the programs, force fields and parameter sets the
  project actually uses (BibTeX export).
- Amber: OL3 + CP retyped residue library; TIP4P/2005 water with microMg / nanoMg and the authors' Cl⁻.
  The default build (CP library, NBfix19, 0BPh, HMR) and a CP + TIP4P/2005 + nanoMg build were run with
  tleap and ParmEd, and the default build was converted to GROMACS and run with grompp / mdrun.
- GUI: icon bar and a permanent 520 px side bar; *Open project* drop-down with recent projects and the
  demo; nothing loaded at start-up; file choosers name their extensions; PDB clean-up window, project
  log, About with the project's citations.

## 0.0.2c (2026-10-16)

- **3D representations** of residue groups: VMD-style cartoon (New Cartoon with base slabs), ribbon,
  tube, licorice, VDW spheres, surface and trace, drawn by a built-in numpy z-buffer renderer (no OpenGL,
  headless too) or the optional VTK renderer; glow, fog and translucency.
- **Density maps** (MRC / CCP4): a simulated map, a rigid-body fit of the model (locally normalised FFT
  search, refinement) and a projected shadow behind the structure view; `mdatk density`.
- Optimised 12-6 Mg²⁺ models **microMg / nanoMg** in the Amber build, carried over to GROMACS.

## 0.0.2b (2026-10-15)

- MDATK also **writes the inputs and scripts that come before the analysis**: a script wizard and a
  free-form **Project Planner** generate consistently named control files for Rosetta FARFAR2, SimRNA,
  RNAMake, Amber (tleap / ParmEd / pmemd), GROMACS and VMD, from sequence to MD ensemble, with logic
  checks (errors, warnings, infos); files are named `<project>_<step>_<tool>_<time>.<ext>`; command
  line `mdatk workflow`.
- Amber defaults reproduce the reference protocol (OL3 + CP, NBfix19 / 0BPh, SPC/E, min1 → production);
  the GROMACS conversion keeps the off-diagonal Lennard-Jones pairs that ParmEd and acpype drop.

## 0.0.2a (2026-10-15)

Package version 0.0.2.

- The main window is the figure: the same panel code as the video, at its aspect ratio, filled with
  example data or the real reference frame; drag panels to swap them, drag the structure to rotate it.
- Data keys may name their own groups (`contacts@Target+Mg/atom_pairs`); C4′ backbone RMSD; whole-number
  ticks for counts; residue-number tags; framing that never zooms on rotation.
- cpptraj inside a conda environment (`--conda-env`); one timestamp per run on output names.

## 0.0.1 (2026-10-12)

- First release: an offline, Linux-native GTK 3 application with a companion command line that analyses
  MD trajectories of RNA systems (cpptraj cleaning; RMSD, contacts, H-bonds, O–Mg–O bridges and Mg
  occupancy, π–π stacking, intermolecular energy in one streaming pass; cached results) and renders
  multi-panel MP4 videos, time charts and selected-frame PDB files; synthetic demo system.
