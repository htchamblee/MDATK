# Molecular Dynamic Analysis Tool Kit (MDATK)

MDATK is an offline, Linux-native GTK 3 desktop application (with a companion command line
interface) for analysing molecular dynamics trajectories of RNA systems and turning them into
multi-panel MP4 videos, high-resolution time charts and selected-frame PDB files.

It reads Amber topologies (`.parm7`) and NetCDF trajectories (`.nc`) directly, can strip and
clean solvated trajectories with **cpptraj**, computes the analyses you select in a single
streaming pass, caches every result, and renders a configurable grid of panels: a rotatable 2D
backbone projection with proximity glow, time series with a moving time cursor, per-residue
heatmaps and contact maps.

Everything runs locally; MDATK makes no network calls.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [Analyses](#analyses)
- [Structural view, colouring and glow](#structural-view-colouring-and-glow)
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
                     example data for the GUI preview
  pipeline.py        clean -> analyse (cached) -> outputs; shared by GUI and CLI
  cli.py, gui/       command line and GTK 3 front ends (gui/canvas.py: WYSIWYG figure canvas)
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

## Testing

```bash
pip install '.[test]'
xvfb-run -a pytest                                # GUI tests need a display (Xvfb works)
MDATK_TEST_CPPTRAJ=/path/to/cpptraj pytest        # also run the real-cpptraj test
```

The suite covers exact geometric criteria of every analysis on hand-built systems, cache keys
and reuse, cpptraj script generation (with a stand-in executable, and with real cpptraj when
available), the full pipeline including FFmpeg output, the CLI and GUI smoke tests.

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
- Dual-axis plots are available because they are useful for juxtaposing e.g. contacts with
  another metric, but two y scales on one plot invite reading a correlation into an arbitrary
  alignment; prefer two stacked time-series panels when in doubt.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
