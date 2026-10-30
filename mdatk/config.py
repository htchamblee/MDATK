"""MDATK parameter model.

All user-adjustable settings live in one :class:`Project` tree of dataclasses.
The tree is stored in a human-readable TOML project file (see
:mod:`mdatk.projectfile`) that the GUI and the CLI share, and every render
leaves a parameter log of the same form next to its outputs.  The comment on
each field below becomes the explanation written above that entry in the file.

Unknown keys are ignored with a warning (forward compatibility); missing keys
fall back to the defaults below (backward compatibility).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
import logging
import os
import re
import typing
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Sequence

from . import PROJECT_ABBREV
from .problems import Problem, problem

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupUse:
    """How a string field uses residue-group names, so a renamed group can be followed everywhere
    (:meth:`Project.rename_group`): ``"name"`` holds a group name, ``"selection"`` a selection (a bare
    group name, comma-separated group names or ``group:<name>`` tokens), ``"key"`` a data key whose
    analysis instance may name groups (``contacts@Aptamer+Target/atom_pairs``) and ``"text"`` free text
    that is never rewritten.  Every other string is searched for ``group:<name>`` tokens only."""

    kind: str


GroupName = Annotated[str, GroupUse("name")]
Selection = Annotated[str, GroupUse("selection")]
DataKey = Annotated[str, GroupUse("key")]
FreeText = Annotated[str, GroupUse("text")]


# ---------------------------------------------------------------------------
# Leaf parameter blocks
# ---------------------------------------------------------------------------


@dataclass
class ResidueGroup:
    """A named set of residues used throughout analyses and colouring.

    ``residues`` takes ranges such as ``"1-40, 45, 50-60"`` (optionally chain
    or segment qualified: ``"A:1-40"``).  ``selection`` is a raw MDAnalysis
    selection string that, when given, overrides ``residues``.
    """

    name: str = "Group"  # Group name, used in selections (group:<name>), panels and legends
    residues: str = ""  # Residue numbers or ranges, e.g. "1-40, 45"; may be chain-qualified ("A:1-40")
    selection: str = ""  # MDAnalysis selection that replaces 'residues' when given
    color: str = "#2a78d6"  # Colour of the group in the structure view, legends and charts (#rrggbb)


@dataclass
class ExternalSeries:
    """An external CSV / whitespace-delimited data file to show as traces."""

    path: str = ""  # CSV or whitespace-delimited text file (e.g. a cpptraj .dat file)
    label: str = ""  # Name of the trace in panels and charts ("": the file name)
    x_column: str = ""  # Column holding the time axis: name or 0-based index ("": the first column)
    x_unit: str = "ps"  # Unit of that column: "ps", "ns", "us", "frame" (0-based) or "frame1" (1-based)
    # Columns to show, by name or index (empty: all other columns)
    columns: List[str] = field(default_factory=list)
    delimiter: str = ""  # Column separator ("": detect comma, tab or whitespace)


@dataclass
class InputParams:
    topology: str = ""  # Topology: Amber .parm7 / .prmtop (preferred), PDB, PSF, GRO, ...
    # Trajectory files in order; several files are concatenated
    trajectories: List[str] = field(default_factory=list)
    start: int = 0  # First analysed frame (0-based; of the cleaned trajectory when cleaning is on)
    stop: Optional[int] = None  # Stop before this frame (0-based, exclusive; not set: up to the last frame)
    step: int = 1  # Analyse every n-th frame
    # Time between stored frames in ps (not set: from the trajectory's time stamps)
    dt_ps: Optional[float] = None
    time_unit: str = "ns"  # Unit shown on time axes: "ps", "ns", "us" or "frame"
    # External data files shown as extra traces (interpolated onto the analysed frames)
    external: List[ExternalSeries] = field(default_factory=list)


@dataclass
class CleaningParams:
    """cpptraj based stripping / imaging / fitting."""

    enabled: bool = False  # Strip, image and fit the trajectory with cpptraj before the analyses
    cpptraj_path: str = "cpptraj"  # cpptraj executable: a name on PATH or an absolute path
    # Conda environment providing cpptraj (a name such as AmberTools26 or a prefix path); every call runs
    # through 'conda run -n <env>'
    conda_env: str = ""
    conda_exe: str = ""  # conda / mamba / micromamba executable ("": found automatically)
    solvent_mask: str = ":WAT,HOH,TIP3,SOL,T3P,T4P,SPC"  # cpptraj mask of the solvent residues to strip
    # cpptraj mask of the ions to strip (Mg2+ is kept by default)
    ion_mask: str = ":Na+,Cl-,K+,NA,CL,K,Na,Cl,SOD,CLA,POT"
    extra_strip_mask: str = ""  # Further atoms to strip (cpptraj mask; optional)
    autoimage: bool = True  # Re-image molecules into the primary box (cpptraj autoimage)
    autoimage_anchor: str = ""  # Anchor mask for autoimage ("": cpptraj's default, the first molecule)
    fit: bool = True  # RMS-fit every frame onto the reference
    fit_mask: str = "@P,C4'"  # cpptraj mask of the atoms used for the RMS fit
    fit_reference: str = "first"  # Fit reference: the "first" frame, a given "frame" number or a "file"
    # Remove the box after fitting (a rotated box cannot be stored in NetCDF)
    remove_box_after_fit: bool = True
    reference_frame: int = 1  # Reference frame for fit_reference = "frame" (1-based, cpptraj convention)
    reference_file: str = ""  # Reference structure for fit_reference = "file"
    start: int = 1  # First frame read by cpptraj (1-based trajin argument)
    stop: Optional[int] = None  # Last frame read by cpptraj (1-based, inclusive; not set: the last frame)
    offset: int = 1  # Read every n-th frame (cpptraj trajin offset)
    extra_commands: str = ""  # Advanced: cpptraj commands inserted before trajout
    output_dir: str = ""  # Folder for the cleaned files ("": <cache_dir>/cleaned)


@dataclass
class RMSDTarget:
    label: str = "Complex"  # Name of the series (data key rmsd/<label>)
    # Atoms whose RMSD is computed (selection; group:<name> allowed)
    selection: Selection = "(group:Aptamer or group:Target) and not name H*"
    fit_selection: Selection = ""  # Atoms superposed before computing the RMSD ("": the same atoms)


def _default_rmsd_targets() -> List[RMSDTarget]:
    return [RMSDTarget(),
            RMSDTarget("Backbone", "(group:Aptamer or group:Target) and name C4'")]


@dataclass
class RMSDParams:
    enabled: bool = True  # Compute RMSD time series
    targets: List[RMSDTarget] = field(default_factory=_default_rmsd_targets)  # One RMSD series per target
    reference_frame: int = 0  # Reference structure: index into the analysed frames (0: the first)
    superposition: bool = True  # Superpose each frame onto the reference before computing the RMSD
    mass_weighted: bool = False  # Mass-weighted superposition and RMSD


@dataclass
class ContactParams:
    enabled: bool = True  # Count heavy-atom contacts between two groups
    group_a: GroupName = "Aptamer"  # First residue group
    group_b: GroupName = "Target"  # Second residue group
    cutoff: float = 4.5  # Contact distance cutoff (Å)
    heavy_only: bool = True  # Ignore hydrogens


@dataclass
class HBondParams:
    enabled: bool = True  # Count hydrogen bonds between two groups
    group_a: GroupName = "Aptamer"  # First residue group
    group_b: GroupName = "Target"  # Second residue group
    distance_cutoff: float = 3.5  # Maximum donor-acceptor distance (Å)
    use_angle: bool = True  # Also require a minimum D-H...A angle (needs hydrogens in the topology)
    angle_cutoff: float = 135.0  # Minimum D-H...A angle (degrees)
    donor_elements: str = "N,O"  # Elements that can donate (comma separated)
    acceptor_elements: str = "N,O"  # Elements that can accept (comma separated)


@dataclass
class MgParams:
    enabled: bool = True  # Analyse Mg2+ coordination: inner/outer-sphere contacts, bridges and occupancy
    mg_selection: Selection = "resname MG Mg MG2 or name MG Mg"  # Selection of the Mg2+ ions
    ligand_selection: Selection = "group:Aptamer or group:Target"  # Atoms that can coordinate Mg2+
    ligand_elements: str = "O"  # Elements of the coordinating atoms (comma separated)
    group_a: GroupName = "Aptamer"  # First group of a Mg2+ bridge
    group_b: GroupName = "Target"  # Second group of a Mg2+ bridge
    inner_cutoff: float = 2.6  # Inner-sphere Mg-O distance cutoff (Å)
    outer_cutoff: float = 4.5  # Outer-sphere (water-mediated) Mg-O distance cutoff (Å)
    bridge_shell: str = "inner"  # Shell that counts as a bridge: "inner" or "outer"
    geometry: bool = False  # Require octahedral O-Mg-O angles (about 90 or 180 degrees)
    angle_tolerance: float = 25.0  # Allowed deviation from the octahedral angles (degrees)
    occupancy_selection: Selection = ""  # Atoms whose Mg2+ occupancy is reported ("": the ligand selection)
    occupancy_elements: str = "N,O"  # Elements counted for the occupancy (comma separated)


@dataclass
class StackingParams:
    enabled: bool = True  # Detect base stacking from ring geometry
    selection: Selection = "group:Aptamer or group:Target"  # Residues whose rings are considered
    scope: str = "all"  # "all" ring pairs, or "inter" (group A against group B only)
    group_a: GroupName = "Aptamer"  # First group for scope = "inter"
    group_b: GroupName = "Target"  # Second group for scope = "inter"
    distance_cutoff: float = 4.5  # Maximum ring-centroid distance (Å)
    angle_cutoff: float = 30.0  # Maximum angle between the ring normals (degrees)
    offset_cutoff: float = 2.5  # Maximum lateral displacement of the centroids (Å)
    planarity_cutoff: float = 0.25  # Maximum RMS deviation of ring atoms from their plane (Å)


@dataclass
class BasePairParams:
    # Base pairs of the residues in the structure view, for the secondary-structure panel and
    # PDB annotation; computed whenever a panel needs them.
    engine: str = "builtin"  # Engine: "builtin" (canonical pairs from H-bond geometry) or "barnaba" (optional)
    stride: int = 10  # Re-detect the pairs every n-th analysed frame (pairs hold in between)
    cutoff: float = 3.5  # Builtin: maximum donor-acceptor distance of a base-pair H-bond (Å)
    wobble: bool = True  # Builtin: also count G-U wobble pairs
    noncanonical: bool = False  # Barnaba: also keep non-canonical (non Watson-Crick) pairs


@dataclass
class EnergyParams:
    enabled: bool = True  # Non-bonded interaction energy between two groups
    group_a: GroupName = "Aptamer"  # First residue group
    group_b: GroupName = "Target"  # Second residue group
    cutoff: float = 12.0  # Pair distance cutoff (Å)
    dielectric: float = 1.0  # Relative dielectric constant
    distance_dependent: bool = False  # Distance-dependent dielectric (epsilon = dielectric x r)
    use_parm7: bool = True  # Use the exact Lennard-Jones parameters of a parm7 topology
    # Reference energy drawn on charts and panels (kcal/mol; not set: none)
    target_value: Optional[float] = None
    exclude_bonded: int = 3  # Exclude pairs within n bonds (0: exclude none)


@dataclass
class AnalysisParams:
    rmsd: RMSDParams = field(default_factory=RMSDParams)  # RMSD time series
    contacts: ContactParams = field(default_factory=ContactParams)  # Heavy-atom contacts between two groups
    hbonds: HBondParams = field(default_factory=HBondParams)  # Hydrogen bonds between two groups
    mg: MgParams = field(default_factory=MgParams)  # Mg2+ coordination, bridges and occupancy
    stacking: StackingParams = field(default_factory=StackingParams)  # Base stacking
    energy: EnergyParams = field(default_factory=EnergyParams)  # Non-bonded interaction energy
    basepairs: BasePairParams = field(default_factory=BasePairParams)  # Base pairs (secondary structure)
    use_pbc: bool = False  # Minimum-image distances (cleaned, imaged systems rarely need it)
    proximity_search_radius: float = 8.0  # Largest proximity cutoff usable later without re-analysing (Å)


REP_STYLES = ("cartoon", "ribbon", "tube", "licorice", "vdw", "surf", "trace")
REP_REST = ("trace", "hide")
REP_COLORS = ("auto", "residue", "element", "nucleotide")


@dataclass
class RepSpec:
    """One 3D representation drawn in the structural view (VMD-like)."""

    style: str = "cartoon"  # Representation: cartoon, ribbon, tube, licorice, vdw, surf or trace
    groups: List[GroupName] = field(default_factory=list)  # Residue groups drawn (empty: every shown group)
    color: str = "auto"  # Colouring: auto, residue, element or nucleotide
    size: float = 1.0  # Scales widths and radii
    opacity: float = 1.0  # Opacity, 0 (invisible) to 1
    sugars: bool = False  # Cartoon only: draw the ribose rings (shows the pucker)


@dataclass
class VisualParams:
    # Residue groups shown in the structure view (empty: all)
    structure_groups: List[GroupName] = field(default_factory=list)
    # Atom used for each residue's 2D trace point, first one found wins (P, C4', C1', CA ...)
    trace_atoms: str = "C4',CA"
    # 3D representations (empty: the 2D backbone trace)
    representations: List[RepSpec] = field(default_factory=list)
    renderer: str = "auto"  # 3D renderer: "auto" / "software" (built-in) or "vtk" (optional)
    rep_rest: str = "trace"  # Residues no representation draws: "trace" (3D trace) or "hide"
    backbone_atom: str = "P"  # Spline control atom of ribbons, cartoons and traces: "P" or "C4'"
    rep_hydrogens: bool = False  # Include hydrogens in all-atom representations
    show_ions: bool = True  # Show Mg2+ ions in the structure view
    align_frames: bool = True  # Superpose each frame's trace onto the reference frame
    # View rotation in degrees about X, Y and Z (R = Rz Ry Rx)
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    zoom: float = 1.0  # Magnification of the structure view (1: the whole model fits; 2: twice as large)
    # View centre: "" (centre of the model), "origin" (0, 0, 0 of the coordinates) or a residue,
    # e.g. "12" or "B:12"
    center_on: str = ""
    center_follow: bool = False  # Keep the centre residue in the middle in every frame (not only the first)
    color_mode: str = "groups"  # Residue colouring: "chain", "groups" or "proximity"
    base_color: str = "#b5b3ad"  # Colour of residues no group or proximity colour applies to
    chain_palette: str = "tab10"  # Matplotlib palette for chain colouring
    # Group (or sel:<selection>) whose proximity colours and glows residues
    proximity_set: Selection = "Target"
    proximity_cutoff: float = 4.0  # Distance that counts as proximity (Å)
    proximity_color: str = "#e34948"  # Colour of residues in proximity
    # Glow of residues near the proximity set: "none", "ever" (whole movie), "fade" or "current"
    glow_mode: str = "fade"
    glow_color: str = "#ffb000"  # Glow colour
    fade_frames: int = 15  # Frames over which the glow fades (glow_mode = "fade")
    glow_strength: float = 1.0  # Glow intensity factor
    ion_color: str = "#008300"  # Colour of Mg2+ ions
    line_width: float = 2.2  # Line width of the 2D trace (points)
    point_size: float = 22.0  # Marker size of the 2D trace points
    depth_cue: bool = True  # Fade distant parts toward the background colour
    theme: str = "light"  # "light" or "dark"
    font_size: float = 11.0  # Base font size (points)
    show_time_label: bool = True  # Show the time label in the structure view
    label_every: int = 0  # Residue-number tags on every n-th residue (0: off)
    label_residues: str = ""  # Explicit residue numbers / ranges to tag, e.g. "1, 12, 24"
    title: str = ""  # Figure title ("": none)


@dataclass
class DensityParams:
    """A density map (MRC / CCP4) drawn behind the structural view as a projected
    "shadow", placed by a rigid-body fit of the model."""

    map_path: str = ""  # Density map file (.mrc / .map / .ccp4)
    show: bool = True  # Draw the map shadow
    resolution: float = 5.0  # Resolution in Å; width of the simulated map used for fitting
    level: float = 2.0  # Contour level in map standard deviations above the mean
    color: str = "#2a78d6"  # Shadow colour
    opacity: float = 0.45  # Shadow opacity, 0 (invisible) to 1
    front: bool = False  # Draw the shadow in front of 3D representations (else behind them)
    fit_frame: str = "ref"  # "ref" (the first analysed frame) or a trajectory frame number to fit on
    search: str = "global"  # Automatic fit: "global" search or "manual" (refine the stored placement)
    angular_step: float = 30.0  # Degrees between sampled orientations of the global search
    auto_fit: bool = True  # Fit before rendering when no placement is stored
    # Placement: 3x3 rotation (row-major) + translation, model -> map
    transform: List[float] = field(default_factory=list)
    # Backbone points of the reference frame the placement was made on (the map follows the molecule when
    # the reference changes: other start frame, cleaned vs raw files)
    anchor: List[List[float]] = field(default_factory=list)
    fit_cc: float = 0.0  # Correlation of the stored placement (0: not fitted)
    fit_info: str = ""  # Summary of the last fit


@dataclass
class PanelSpec:
    """One cell of the figure grid.

    ``type`` is a registered panel type (see :mod:`mdatk.render.panels`).
    ``series`` lists data keys such as ``"rmsd/Complex"``,
    ``"contacts/atom_pairs"`` or ``"contacts/per_residue_a"``.
    """

    type: str = "structure"  # Panel type: structure, timeseries, dual_axis, heatmap or contact_map
    # Data keys shown, e.g. rmsd/Complex or contacts/per_residue_a
    series: List[DataKey] = field(default_factory=list)
    title: str = ""  # Panel title ("": automatic)
    # Panel-specific options, e.g. axis limits or the contact-map mode
    options: Dict[str, Any] = field(default_factory=dict)


# titles of the default / demo layouts: they count as automatic, so a panel that changes type
# gets the new type's title instead of keeping one that no longer fits
DEFAULT_PANEL_TITLES = ("Structure", "RMSD", "Inter-strand H-bonds", "Aptamer residue contacts",
                        "Mg occupancy")


def layout_warnings(project: "Project") -> List[str]:
    """Layout choices that are allowed but probably unintended."""
    out = []
    n_struct = sum(1 for sp in project.layout.panels if sp.type == "structure")
    if n_struct > 1:
        out.append(f"{n_struct} structure panels in the layout; usually one is intended (check each "
                   "slot's panel type)")
    return out


def _default_panels() -> List[PanelSpec]:
    return [
        PanelSpec("structure", [], "Structure"),
        PanelSpec("timeseries", ["rmsd/Complex"], "RMSD"),
        PanelSpec("timeseries", ["hbonds/count"], "Inter-strand H-bonds"),
        # aptamer residues along x, target residues along y, evolving with the movie
        PanelSpec("contact_map", ["contacts/map"], "", {"mode": "instant", "transpose": True}),
    ]


@dataclass
class LayoutParams:
    preset: str = "large_plus_3"  # Grid layout (see 'mdatk list')
    panels: List[PanelSpec] = field(default_factory=_default_panels)  # Panels in slot order


@dataclass
class VideoParams:
    enabled: bool = True  # Write the MP4 video
    # Video file name (time-stamped when output.timestamp_files is on)
    filename: str = f"{PROJECT_ABBREV}_video.mp4"
    fps: int = 24  # Frames per second
    width: int = 1920  # Width in pixels
    height: int = 1080  # Height in pixels
    dpi: int = 120  # Figure resolution; sets text and line sizes relative to the frame
    codec: str = "libx264"  # FFmpeg video codec
    crf: int = 20  # Constant rate factor: lower means better quality and larger files (libx264: 18-28)
    frame_stride: int = 1  # Use every n-th analysed frame
    ffmpeg_path: str = "ffmpeg"  # FFmpeg executable


@dataclass
class ChartParams:
    enabled: bool = True  # Write high-resolution time charts
    series: List[DataKey] = field(default_factory=list)  # Data keys to chart (empty: every scalar series)
    include_heatmaps: bool = False  # Also chart per-residue heatmaps
    formats: List[str] = field(default_factory=lambda: ["png"])  # File formats, e.g. png, pdf, svg
    dpi: int = 300  # Raster resolution
    width: float = 8.0  # Width in inches
    height: float = 4.5  # Height in inches
    style: str = "mdatk"  # "mdatk" (built-in look) or any matplotlib style name
    color: str = ""  # Line colour ("": the first palette colour)
    line_width: float = 1.5  # Line width (points)
    title: str = ""  # Chart title ("": the series label; "{label}" placeholders allowed)
    xlabel: str = ""  # x-axis label ("": automatic)
    ylabel: str = ""  # y-axis label ("": automatic)
    smoothing: int = 0  # Running-average window in frames (0: off)
    show_target: bool = True  # Draw the energy target value when one is set
    prefix: str = f"{PROJECT_ABBREV}_chart"  # File name prefix


@dataclass
class PDBParams:
    enabled: bool = False  # Export selected frames as PDB files
    frames: str = ""  # Frames to export, e.g. "0, 10, 20-50:10, last"
    times: str = ""  # Times to export, e.g. "1.5ns, 2000ps" (no unit: inputs.time_unit)
    selection: Selection = ""  # Group names (comma separated) or a selection ("": everything)
    prefix: str = f"{PROJECT_ABBREV}_frame"  # File name prefix


@dataclass
class OutputParams:
    # Output folder (a relative path is relative to the project file)
    out_dir: str = f"{PROJECT_ABBREV}_output"
    cache_dir: str = ""  # Cache of analyses and cleaned trajectories ("": <out_dir>/MDATK_cache)
    video: VideoParams = field(default_factory=VideoParams)  # MP4 video
    charts: ChartParams = field(default_factory=ChartParams)  # Time charts
    pdb: PDBParams = field(default_factory=PDBParams)  # PDB export of selected frames
    # Write MDATK_params_log.toml (settings, versions, cache keys, citations) next to the outputs
    write_params_log: bool = True
    # Time-stamp output file names, e.g. MDATK_video_2026-10-15_143205.mp4 (cache and log keep fixed names)
    timestamp_files: bool = True


@dataclass
class PlanNode:
    """One box of the Project Planner (a step that writes files for one program).

    ``tool`` names a generator in :mod:`mdatk.workflows`; ``settings`` holds its
    options (see ``mdatk workflow tools``) and ``files`` hand-picked input files
    for ports that no connection feeds (``{port: path}``).
    """

    id: str = ""  # Unique id of the step
    tool: str = "note"  # Step type: the script generator (see 'mdatk workflow tools')
    step: int = 1  # Step number; orders the steps and their file names
    title: str = ""  # Box title ("": the step type's label)
    x: float = 40.0  # Position of the box on the planner canvas
    y: float = 40.0  # Position of the box on the planner canvas
    # The step's options (each one is explained by its comment)
    settings: Dict[str, Any] = field(default_factory=dict)
    # Hand-picked input files for ports that no connection feeds ({port = path})
    files: Dict[str, str] = field(default_factory=dict)
    # Time stamp of an earlier run whose output files the plan uses for this step, which is then neither
    # written nor run again ("": none; set by 'Check previous runs' or 'mdatk workflow status --adopt')
    adopted: str = ""


@dataclass
class PlanEdge:
    """A connection: output ``output`` of node ``source`` feeds input ``input`` of ``target``."""

    source: str = ""  # Id of the producing step
    output: str = ""  # Output port of the producing step
    target: str = ""  # Id of the receiving step
    input: str = ""  # Input port of the receiving step


@dataclass
class WorkflowParams:
    """Project Planner and script-wizard settings (saved with the preset).

    Generated files are named ``<project>_<step>_<tool>_<stamp>.<ext>``, e.g.
    ``apt1_03_tleap_20261015-1432.parm7``.  ``stamp`` is fixed the first time
    files are written so later re-generations keep matching names.
    """

    project_name: str = "project"  # Project name used in every generated file name
    stamp: str = ""  # Time stamp in generated file names (fixed once files are written)
    use_stamp: bool = True  # Put the time stamp into file names
    out_dir: str = ""  # Folder of the generated files ("": <output.out_dir>/<project_name>_workflow)
    step_folders: bool = True  # One sub-folder per step, e.g. 03_amber-build/
    parallel: int = 1  # Scripts MDATK runs at the same time (Run all / Run selected); steps wait for their inputs
    nodes: List[PlanNode] = field(default_factory=list)  # Planner steps
    edges: List[PlanEdge] = field(default_factory=list)  # Planner connections
    # Stand-alone script wizard settings, one entry per step type
    scratch: List[PlanNode] = field(default_factory=list)


def _default_groups() -> List[ResidueGroup]:
    # Editable defaults for the reference aptamer/target system.
    return [
        ResidueGroup("Aptamer", "1-12", "", "#2a78d6"),
        ResidueGroup("Target", "13-24", "", "#eb6834"),
    ]


@dataclass
class Project:
    """Complete MDATK parameter set."""

    notes: FreeText = ""  # Free-text notes about the project (what, why, decisions, to-dos)
    inputs: InputParams = field(default_factory=InputParams)  # Input files and frame range
    cleaning: CleaningParams = field(default_factory=CleaningParams)  # Trajectory cleaning with cpptraj
    groups: List[ResidueGroup] = field(default_factory=_default_groups)  # Residue groups
    analyses: AnalysisParams = field(default_factory=AnalysisParams)  # Analyses
    # Structure view: colouring, glow, 3D representations, labels
    visual: VisualParams = field(default_factory=VisualParams)
    density: DensityParams = field(default_factory=DensityParams)  # Density map shadow and fit
    layout: LayoutParams = field(default_factory=LayoutParams)  # Figure layout: grid and panels
    output: OutputParams = field(default_factory=OutputParams)  # Outputs: folders, video, charts, PDB frames
    workflow: WorkflowParams = field(default_factory=WorkflowParams)  # Project Planner and script wizard

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "Project":
        data = dict(data)
        for meta_key in ("format", "format_version", "mdatk_version", "saved", "log", "log_file"):
            data.pop(meta_key, None)
        project = _from_plain(cls, data, "project")
        if base_dir:
            project.resolve_paths(base_dir)
        return project

    def save(self, path: str) -> None:
        """Write the settings to a TOML project file (an existing step log is kept)."""
        from . import projectfile

        projectfile.save(path, self)

    @classmethod
    def load(cls, path: str) -> "Project":
        """Read a project file (or a parameter log, to replay that run)."""
        from . import projectfile

        data = projectfile.read(path)
        return cls.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))

    def copy(self) -> "Project":
        return copy.deepcopy(self)

    # -------------------------------------------------------------- helpers
    def resolve_paths(self, base_dir: str) -> None:
        """Make relative file paths absolute with respect to ``base_dir``."""

        def fix(p: str) -> str:
            if not p:
                return p
            p = os.path.expanduser(p)
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

        self.inputs.topology = fix(self.inputs.topology)
        self.inputs.trajectories = [fix(p) for p in self.inputs.trajectories]
        for ext in self.inputs.external:
            ext.path = fix(ext.path)
        self.cleaning.reference_file = fix(self.cleaning.reference_file)
        self.cleaning.output_dir = fix(self.cleaning.output_dir)
        if self.cleaning.cpptraj_path and os.sep in self.cleaning.cpptraj_path:
            self.cleaning.cpptraj_path = fix(self.cleaning.cpptraj_path)
        self.output.out_dir = fix(self.output.out_dir)
        self.output.cache_dir = fix(self.output.cache_dir)
        self.density.map_path = fix(self.density.map_path)
        self.workflow.out_dir = fix(self.workflow.out_dir)
        from .workflows import resolve_node_paths  # lazy: workflows imports this module
        for node in list(self.workflow.nodes) + list(self.workflow.scratch):
            resolve_node_paths(node, fix)

    def group(self, name: str) -> ResidueGroup:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"no residue group named {name!r} "
                       f"(defined: {', '.join(g.name for g in self.groups) or 'none'})")

    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    def rename_group(self, old: str, new: str) -> List[str]:
        """Rename residue group ``old`` to ``new`` and every reference to it in the project.

        The whole parameter tree is searched, planner steps included (see :class:`GroupUse`): a field
        holding a group name and a selection that is just the name change when they equal ``old``,
        analysis instances in data keys (``contacts@old+B/atom_pairs``) and ``group:old`` tokens in any
        text change too, while ``group:older`` or ``group:old_2`` stay.  Returns the changed locations,
        e.g. ``analyses.contacts.group_a`` (the group's own name not counted).  Raises KeyError for an
        unknown group and ValueError for an empty, invalid or taken new name."""
        group = self.group(old)
        new = new.strip()
        if new == old:
            return []
        names = self.group_names()
        if names.count(old) > 1:
            raise ValueError(f"several residue groups are named {old!r}; give them distinct names first")
        problem = group_name_problem(new, [n for n in names if n != old])
        if problem:
            raise ValueError(problem)
        changed: List[str] = []
        known = set(names)

        def rename(path: str, kind: str, text: str) -> str:
            out = _renamed_text(kind, text, old, new, known)
            if out != text:
                changed.append(path)
            return out

        _walk_strings(self, "", "", rename)
        group.name = new
        return changed

    def group_references(self, name: str) -> List[str]:
        """Locations that refer to residue group ``name``, found by the rules of :meth:`rename_group`
        (e.g. what still uses a group that was deleted)."""
        found: List[str] = []
        names = set(self.group_names()) | {name}

        def visit(path: str, kind: str, text: str) -> str:
            if _renamed_text(kind, text, name, "\0", names) != text:
                found.append(path)
            return text

        _walk_strings(self, "", "", visit)
        return found

    @property
    def out_dir(self) -> str:
        return os.path.abspath(self.output.out_dir or f"{PROJECT_ABBREV}_output")

    @property
    def workflow_dir(self) -> str:
        """Folder that receives Project Planner / wizard files."""
        if self.workflow.out_dir:
            return os.path.abspath(self.workflow.out_dir)
        from .workflows.naming import sanitize_project
        return os.path.join(self.out_dir, f"{sanitize_project(self.workflow.project_name)}_workflow")

    @property
    def cache_dir(self) -> str:
        if self.output.cache_dir:
            return os.path.abspath(self.output.cache_dir)
        return os.path.join(self.out_dir, f"{PROJECT_ABBREV}_cache")

    def density_active(self) -> bool:
        """A density map is set and shown in a structure panel."""
        d = self.density
        return bool(d.map_path and d.show and any(sp.type == "structure" for sp in self.layout.panels))

    def density_problems(self) -> List[Problem]:
        d = self.density
        if not d.map_path:
            return []
        out: List[Problem] = []

        def add(field: str, message: str) -> None:
            out.append(problem(f"density.{field}", message, self))

        if self.density_active() and not os.path.exists(d.map_path):
            add("map_path", f"Density map not found: {d.map_path} (untick 'show' to render without it)")
        if d.resolution <= 0:
            add("resolution", "Density map resolution must be positive.")
        if not 0.0 <= d.opacity <= 1.0:
            add("opacity", "Density map opacity must be between 0 and 1.")
        if not 5.0 <= d.angular_step <= 45.0:
            add("angular_step", "Density search angular step must be between 5 and 45 degrees.")
        if d.search not in ("global", "manual"):
            add("search", f"Unknown density fit start {d.search!r} (global or manual).")
        if d.transform:
            from .density import RigidTransform

            try:
                RigidTransform.from_list(d.transform)
            except (TypeError, ValueError) as exc:
                add("transform", f"The stored density map placement is invalid ({exc}); clear it or fit "
                    "again.")
        frame = d.fit_frame.strip().lower()
        if frame not in ("", "ref") and not frame.isdigit():
            add("fit_frame", f"Density fit frame must be 'ref' or a frame number, not {d.fit_frame!r}.")
        return out

    def validate(self) -> List[Problem]:
        """Return a list of human readable problems (empty when valid).

        The items are :class:`~mdatk.problems.Problem` strings: each text starts with where the problem
        is ("Analyses › Hydrogen bonds › Group A: unknown residue group ...") and each item carries the
        setting's field path and GUI page.  Warnings (see :meth:`problems`) are left out."""
        return [p for p in self.problems() if p.level == "error"]

    def problems(self) -> List[Problem]:
        """Every problem of the settings: errors (a run would fail; :meth:`validate`) and warnings (the
        setting is allowed but ignored or probably unintended), each naming its setting."""
        out: List[Problem] = []

        def add(field: str, message: str, level: str = "error", token: str = "") -> None:
            out.append(problem(field, message, self, level, token))

        if not self.inputs.topology:
            add("inputs.topology", "No topology file selected.")
        elif not os.path.exists(self.inputs.topology):
            add("inputs.topology", f"Topology file not found: {self.inputs.topology}")
        if not self.inputs.trajectories:
            add("inputs.trajectories", "No trajectory file selected.")
        for t in self.inputs.trajectories:
            if not os.path.exists(t):
                add("inputs.trajectories", f"Trajectory file not found: {t}")
        if self.inputs.step < 1:
            add("inputs.step", "Frame step must be >= 1.")
        names = [g.name for g in self.groups]
        for k, g in enumerate(self.groups):
            if not g.name.strip():
                add(f"groups.{k}.name", "Residue groups need a name.")
            elif not re.fullmatch(r"[A-Za-z0-9_.\-]+", g.name):
                add(f"groups.{k}.name", f"Group name {g.name!r} may only contain letters, digits, "
                    "'_', '.' and '-' (no spaces).", token=g.name)
            elif g.name in names[:k]:
                add(f"groups.{k}.name", f"Residue group names must be unique: {g.name!r} is also the name of "
                    f"group {names.index(g.name) + 1}.", token=g.name)
            if not (g.residues.strip() or g.selection.strip()):
                add(f"groups.{k}.residues", f"Group {g.name!r} has neither residues nor a selection.")
        out += self.selection_problems()
        a = self.analyses
        if a.contacts.cutoff <= 0:
            add("analyses.contacts.cutoff", "Contact cutoff must be positive.")
        if a.mg.inner_cutoff > a.mg.outer_cutoff:
            add("analyses.mg.inner_cutoff", "Mg inner-sphere cutoff must not exceed the outer-sphere cutoff.")
        v = self.visual
        if v.color_mode not in ("chain", "groups", "proximity"):
            add("visual.color_mode", f"Unknown colour mode {v.color_mode!r}.")
        if v.glow_mode not in ("none", "ever", "fade", "current"):
            add("visual.glow_mode", f"Unknown glow mode {v.glow_mode!r}.")
        for k, rs in enumerate(v.representations):
            rep = f"visual.representations.{k}"
            if rs.style not in REP_STYLES:
                add(f"{rep}.style", f"Unknown style {rs.style!r} ({', '.join(REP_STYLES)}).")
            if rs.color not in REP_COLORS:
                add(f"{rep}.color", f"Unknown colouring {rs.color!r}.")
            if rs.size <= 0 or not 0 < rs.opacity <= 1:
                add(f"{rep}.size", "Size must be positive and opacity in (0, 1].")
        if v.renderer not in ("auto", "software", "vtk"):
            add("visual.renderer", f"Unknown renderer {v.renderer!r} (auto, software or vtk).")
        if not 0.05 <= v.zoom <= 50:
            add("visual.zoom", "Structure zoom must be between 0.05 and 50.")
        bp = a.basepairs
        if bp.engine not in ("builtin", "barnaba"):
            add("analyses.basepairs.engine", f"Unknown base-pair engine {bp.engine!r} (builtin or barnaba).")
        if bp.stride < 1:
            add("analyses.basepairs.stride", "Base-pair stride must be at least 1.")
        if not 2.0 <= bp.cutoff <= 5.0:
            add("analyses.basepairs.cutoff", "Base-pair H-bond cutoff must be between 2.0 and 5.0 Å.")
        if v.rep_rest not in REP_REST:
            add("visual.rep_rest", f"Unknown setting for residues without a representation {v.rep_rest!r} "
                "(trace or hide).")
        if v.backbone_atom not in ("P", "C4'"):
            add("visual.backbone_atom", "The backbone atom of ribbons and cartoons must be P or C4'.")
        if v.proximity_cutoff > a.proximity_search_radius:
            add("visual.proximity_cutoff", "Proximity cutoff exceeds the proximity search radius "
                "(raise analyses.proximity_search_radius).")
        out += self.density_problems()
        if self.output.video.fps <= 0:
            add("output.video.fps", "Video FPS must be positive.")
        for text in layout_warnings(self):
            add("layout.panels", text, "warning")
        return out

    def selection_problems(self) -> List[Problem]:
        """Problems of the residue groups and of every setting that names a group or holds a selection:
        unknown groups (with the closest defined name), circular group definitions, residue ranges and
        selections MDAnalysis cannot parse.  Settings of disabled analyses are not checked; settings that
        are ignored when wrong, or not used with the current options, only get warnings."""
        from .selections import (SelectionError, check_selection, parse_residue_ranges, referenced_groups,
                                 split_csv, unknown_group_error)

        out: List[Problem] = []
        names = self.group_names()
        broken: set = set()  # groups whose own definition has a problem (reported there only)
        a, v = self.analyses, self.visual

        def add(field: str, message: str, level: str = "error", token: str = "") -> None:
            out.append(problem(field, message, self, level, token))

        def unknown(field: str, gname: str, level: str = "error", note: str = "") -> None:
            message = unknown_group_error(gname, names).message
            if note:  # a second sentence: what happens with the unknown name
                message += ("" if message.endswith("?") else ".") + f" {note}"
            add(field, message, level, gname)

        def check(field: str, text: str, level: str = "error", required: bool = True,
                  group: str = "") -> bool:
            """``text``: a group name, ``sel:<selection>`` or a selection with group:<Name> tokens.  A
            selection is not blamed for a broken group it uses: that group's definition is."""
            text = (text or "").strip()
            if not text and not required:
                return True
            try:
                check_selection(text, self.groups)
            except SelectionError as exc:
                if (exc.group and exc.group != group) or (broken & set(referenced_groups(text, self.groups))):
                    return True
                if exc.kind == "syntax" and exc.token == text:  # a lone unknown word: a group name?
                    exc = unknown_group_error(text, names)
                add(field, exc.message, level, exc.token)
                return False
            return True

        def ranges(field: str, text: str, level: str = "error") -> bool:
            try:
                parse_residue_ranges(text)
            except SelectionError as exc:
                add(field, f"{exc.message} (ranges look like 1-40, 45, A:50-60)", level, exc.token)
                return False
            return True

        for k, g in enumerate(self.groups):
            if g.selection.strip():
                ok = check(f"groups.{k}.selection", g.selection, group=g.name)
            else:
                ok = ranges(f"groups.{k}.residues", g.residues)
            if not ok or not (g.selection.strip() or g.residues.strip()):
                broken.add(g.name)

        if a.rmsd.enabled:
            for k, t in enumerate(a.rmsd.targets):
                check(f"analyses.rmsd.targets.{k}.selection", t.selection)
                check(f"analyses.rmsd.targets.{k}.fit_selection", t.fit_selection, required=False)
        for key in ("contacts", "hbonds", "mg", "stacking", "energy"):
            params = getattr(a, key)
            if params.enabled:
                for slot in ("group_a", "group_b"):
                    gname = getattr(params, slot)
                    if gname and gname not in names:
                        unknown(f"analyses.{key}.{slot}", gname)
        panel_types = {sp.type for sp in self.layout.panels}
        structure_used = bool(panel_types & {"structure", "sequence", "secondary"})
        if a.mg.enabled or structure_used:  # the structure view draws the ions of this selection
            check("analyses.mg.mg_selection", a.mg.mg_selection)
        if a.mg.enabled:
            check("analyses.mg.ligand_selection", a.mg.ligand_selection)
            check("analyses.mg.occupancy_selection", a.mg.occupancy_selection, required=False)
        if a.stacking.enabled:
            check("analyses.stacking.selection", a.stacking.selection)

        shown = [n for n in v.structure_groups if n in names]
        for gname in v.structure_groups:
            if gname not in names:
                unknown("visual.structure_groups", gname, "warning", "It is left out of the structure view.")
        if structure_used and not (shown if v.structure_groups else names):
            why = "none of the groups shown exists" if v.structure_groups else "define a residue group"
            add("visual.structure_groups", f"The structure view has no residue group to show ({why}).")
        for k, rs in enumerate(v.representations):
            for gname in rs.groups:
                if gname not in names:
                    unknown(f"visual.representations.{k}.groups", gname)
        proximity = structure_used and (v.color_mode == "proximity" or v.glow_mode != "none")
        check("visual.proximity_set", v.proximity_set, "error" if proximity else "warning",
              required=proximity)
        if v.center_on.strip() and v.center_on.strip().lower() != "origin":
            ranges("visual.center_on", v.center_on, "warning")
        ranges("visual.label_residues", v.label_residues, "warning")

        for k, sp in enumerate(self.layout.panels):
            for j, key in enumerate(sp.series):
                for gname in _key_groups(key):
                    if gname not in names:
                        unknown(f"layout.panels.{k}.{'groups' if j == 0 else 'series'}", gname, "warning",
                                f"The panel has no data for {key!r}.")
            residues = (sp.options or {}).get("residues") if sp.type in ("sequence", "secondary") else None
            if isinstance(residues, str):
                ranges(f"layout.panels.{k}.options", residues, "warning")
        pdb = self.output.pdb
        items = split_csv(pdb.selection)
        level = "error" if pdb.enabled else "warning"
        if "," in pdb.selection and all(re.fullmatch(r"[A-Za-z0-9_.\-]+", n) for n in items):
            for gname in items:  # a list of group names
                if gname not in names:
                    unknown("output.pdb.selection", gname, level)
        elif not (items and all(n in names for n in items)):
            check("output.pdb.selection", pdb.selection, level, required=False)
        if self.output.charts.enabled:
            for key in self.output.charts.series:
                for gname in _key_groups(key):
                    if gname not in names:
                        unknown("output.charts.series", gname, "warning", f"No chart is drawn for {key!r}.")
        return out


def _key_groups(key: str) -> List[str]:
    """Residue groups of a data key: ``"contacts@A+B/atom_pairs"`` -> ``["A", "B"]``."""
    instance = key.split("/", 1)[0]
    return [g for g in instance.partition("@")[2].split("+") if g]


# ---------------------------------------------------------------------------
# Generic dataclass <-> plain data conversion
# ---------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):  # free-form settings: an unset (None) option is simply absent
        return {str(k): _to_plain(v) for k, v in obj.items() if v is not None}
    return obj


_ALIASES = {
    # convenience spellings accepted in hand-written project files
    ("InputParams", "trajectory"): "trajectories",
    ("Project", "input"): "inputs",
}


def _from_plain(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Union:  # Optional[X]
        if value is None:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _from_plain(non_none[0], value, where)

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected an object, got {type(value).__name__}")
        hints = typing.get_type_hints(tp)
        names = {f.name for f in dataclasses.fields(tp)}
        kwargs = {}
        for key, val in value.items():
            key = _ALIASES.get((tp.__name__, key), key)
            if key not in names:
                log.warning("%s: ignoring unknown key %r", where, key)
                continue
            if tp is InputParams and key == "trajectories" and isinstance(val, str):
                val = [val]
            kwargs[key] = _from_plain(hints[key], val, f"{where}.{key}")
        return tp(**kwargs)

    if origin in (list, List):
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list")
        (item_tp,) = args or (Any,)
        return [_from_plain(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected an object")
        return dict(value)

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return int(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if value is None:
            return ""
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Residue-group references (rename, find)
# ---------------------------------------------------------------------------

_GROUP_NAME = re.compile(r"[A-Za-z0-9_.\-]+")  # the names validate() accepts
# an analysis instance in a data key: base@GroupA+GroupB/output (see mdatk.analysis.base.parse_instance)
_INSTANCE_KEY = re.compile(r"(?P<base>[^@/]+)@(?P<groups>[^/]*)(?P<rest>/.*)?", re.DOTALL)


def group_name_problem(name: str, taken: Sequence[str] = ()) -> str:
    """Why ``name`` cannot name a residue group ("" when it can); ``taken``: the other groups' names."""
    if not name.strip():
        return "A residue group needs a name."
    if not _GROUP_NAME.fullmatch(name):
        return f"Group name {name!r} may only contain letters, digits, '_', '.' and '-' (no spaces)."
    if name in taken:
        return f"There is already a residue group named {name!r}."
    return ""


def _use_of(tp: Any) -> str:
    """The :class:`GroupUse` kind of a type hint, looking inside List[...] / Optional[...] ("" for none)."""
    if typing.get_origin(tp) is Annotated:
        for meta in tp.__metadata__:
            if isinstance(meta, GroupUse):
                return meta.kind
        tp = typing.get_args(tp)[0]
    for arg in typing.get_args(tp):
        kind = _use_of(arg)
        if kind:
            return kind
    return ""


@functools.lru_cache(maxsize=None)
def _field_uses(cls: type) -> Dict[str, str]:
    hints = typing.get_type_hints(cls, include_extras=True)
    return {f.name: _use_of(hints.get(f.name)) for f in dataclasses.fields(cls)}


def _walk_strings(obj: Any, path: str, kind: str, fn) -> None:
    """Replace every string below ``obj`` by ``fn(location, kind, text)``: dataclass fields (with the
    kind of their :class:`GroupUse` marker), list items and the values of free-form tables (planner
    step settings, panel options: kind "")."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        uses = _field_uses(type(obj))
        for f in dataclasses.fields(obj):
            where = f"{path}.{f.name}" if path else f.name
            value = getattr(obj, f.name)
            if isinstance(value, str):
                new = fn(where, uses[f.name], value)
                if new != value:
                    setattr(obj, f.name, new)
            else:
                _walk_strings(value, where, uses[f.name], fn)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, str):
                obj[i] = fn(f"{path}[{i}]", kind, value)
            else:
                _walk_strings(value, f"{path}[{i}]", kind, fn)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                obj[key] = fn(f"{path}.{key}", "", value)
            else:
                _walk_strings(value, f"{path}.{key}", "", fn)


def _renamed_text(kind: str, text: str, old: str, new: str, names) -> str:
    """``text`` with residue group ``old`` renamed to ``new``, read as a field of ``kind`` uses it;
    ``names``: the project's group names (a comma-separated list of them counts as group names)."""
    from .selections import rename_group_tokens  # lazy: keeps this module free of numpy

    if kind == "text":
        return text
    if kind == "name":
        return new if text.strip() == old else text
    if kind == "key":
        m = _INSTANCE_KEY.fullmatch(text)
        groups = m.group("groups").split("+") if m else []
        if old not in groups:
            return text
        return f"{m.group('base')}@{'+'.join(new if g == old else g for g in groups)}{m.group('rest') or ''}"
    if kind == "selection":
        items = text.split(",")
        if any(t.strip() == old for t in items) and all(t.strip() in names for t in items):
            return ",".join(t.replace(old, new) if t.strip() == old else t for t in items)
    return rename_group_tokens(text, old, new)


# ---------------------------------------------------------------------------
# Hashing helpers (cache keys)
# ---------------------------------------------------------------------------


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic short hash of a JSON-serialisable object."""
    blob = json.dumps(_to_plain(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: str) -> Dict[str, Any]:
    """Cheap identity of a file: absolute path, size and modification time."""
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
