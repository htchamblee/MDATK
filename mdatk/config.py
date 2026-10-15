"""MDATK parameter model.

All user-adjustable settings live in one :class:`Project` tree of dataclasses.
The tree serialises to and from plain JSON so that a parameter preset written
by the GUI can be replayed by the CLI (and vice versa) and so that every render
can leave a reproducible parameter log next to its outputs.

Unknown keys in a preset are ignored with a warning (forward compatibility);
missing keys fall back to the defaults below (backward compatibility).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import PROJECT_ABBREV, __version__

log = logging.getLogger(__name__)

PRESET_FORMAT = "mdatk-preset"
PRESET_FORMAT_VERSION = 1


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

    name: str = "Group"
    residues: str = ""
    selection: str = ""
    color: str = "#2a78d6"


@dataclass
class ExternalSeries:
    """An external CSV / whitespace-delimited data file to show as traces."""

    path: str = ""
    label: str = ""
    x_column: str = ""  # column name or 0-based index; "" -> first column
    x_unit: str = "ps"  # "ps", "ns", "us" or "frame"
    columns: List[str] = field(default_factory=list)  # empty -> all others
    delimiter: str = ""  # "" -> auto (comma, tab or whitespace)


@dataclass
class InputParams:
    topology: str = ""
    trajectories: List[str] = field(default_factory=list)
    start: int = 0  # 0-based frame of the (cleaned) trajectory
    stop: Optional[int] = None  # exclusive, None -> last frame
    step: int = 1
    dt_ps: Optional[float] = None  # override time between stored frames
    time_unit: str = "ns"  # display unit: "ps", "ns", "us" or "frame"
    external: List[ExternalSeries] = field(default_factory=list)


@dataclass
class CleaningParams:
    """cpptraj based stripping / imaging / fitting."""

    enabled: bool = False
    cpptraj_path: str = "cpptraj"
    conda_env: str = ""  # e.g. "AmberTools26" or a prefix path; run as `conda run -n <env> cpptraj`
    conda_exe: str = ""  # conda / mamba / micromamba executable ("" = auto-detect)
    solvent_mask: str = ":WAT,HOH,TIP3,SOL,T3P,T4P,SPC"
    ion_mask: str = ":Na+,Cl-,K+,NA,CL,K,Na,Cl,SOD,CLA,POT"  # Mg2+ is kept
    extra_strip_mask: str = ""
    autoimage: bool = True
    autoimage_anchor: str = ""  # cpptraj mask, "" -> cpptraj default
    fit: bool = True
    fit_mask: str = "@P,C4'"
    fit_reference: str = "first"  # "first", "frame" or "file"
    remove_box_after_fit: bool = True  # a fitted (rotated) box cannot be stored in NetCDF
    reference_frame: int = 1  # 1-based (cpptraj convention)
    reference_file: str = ""
    start: int = 1  # 1-based (cpptraj trajin convention)
    stop: Optional[int] = None  # None -> last
    offset: int = 1
    extra_commands: str = ""  # advanced: inserted before trajout
    output_dir: str = ""  # "" -> <cache_dir>/cleaned


@dataclass
class RMSDTarget:
    label: str = "Complex"
    selection: str = "(group:Aptamer or group:Target) and not name H*"
    fit_selection: str = ""  # "" -> same as selection


def _default_rmsd_targets() -> List[RMSDTarget]:
    return [RMSDTarget(),
            RMSDTarget("Backbone", "(group:Aptamer or group:Target) and name C4'")]


@dataclass
class RMSDParams:
    enabled: bool = True
    targets: List[RMSDTarget] = field(default_factory=_default_rmsd_targets)
    reference_frame: int = 0  # index into the analysed frames
    superposition: bool = True
    mass_weighted: bool = False


@dataclass
class ContactParams:
    enabled: bool = True
    group_a: str = "Aptamer"
    group_b: str = "Target"
    cutoff: float = 4.5
    heavy_only: bool = True


@dataclass
class HBondParams:
    enabled: bool = True
    group_a: str = "Aptamer"
    group_b: str = "Target"
    distance_cutoff: float = 3.5  # donor - acceptor
    use_angle: bool = True
    angle_cutoff: float = 135.0  # minimum D-H...A angle (degrees)
    donor_elements: str = "N,O"
    acceptor_elements: str = "N,O"


@dataclass
class MgParams:
    enabled: bool = True
    mg_selection: str = "resname MG Mg MG2 or name MG Mg"
    ligand_selection: str = "group:Aptamer or group:Target"
    ligand_elements: str = "O"
    group_a: str = "Aptamer"
    group_b: str = "Target"
    inner_cutoff: float = 2.6  # inner-sphere Mg-O
    outer_cutoff: float = 4.5  # outer-sphere (water mediated) Mg-O
    bridge_shell: str = "inner"  # "inner" or "outer"
    geometry: bool = False  # require octahedral O-Mg-O angles (~90 / 180 deg)
    angle_tolerance: float = 25.0
    occupancy_selection: str = ""  # "" -> ligand_selection
    occupancy_elements: str = "N,O"


@dataclass
class StackingParams:
    enabled: bool = True
    selection: str = "group:Aptamer or group:Target"
    scope: str = "all"  # "all" (every ring pair) or "inter" (group A vs group B only)
    group_a: str = "Aptamer"
    group_b: str = "Target"
    distance_cutoff: float = 4.5  # ring centroid distance
    angle_cutoff: float = 30.0  # angle between ring normals
    offset_cutoff: float = 2.5  # lateral centroid displacement
    planarity_cutoff: float = 0.25  # RMS deviation of ring atoms from plane


@dataclass
class EnergyParams:
    enabled: bool = True
    group_a: str = "Aptamer"
    group_b: str = "Target"
    cutoff: float = 12.0
    dielectric: float = 1.0
    distance_dependent: bool = False
    use_parm7: bool = True  # exact Amber LJ parameters when the topology is parm7
    target_value: Optional[float] = None  # kcal/mol, drawn on charts / panels
    exclude_bonded: int = 3  # exclude pairs within n bonds (0 disables)


@dataclass
class AnalysisParams:
    rmsd: RMSDParams = field(default_factory=RMSDParams)
    contacts: ContactParams = field(default_factory=ContactParams)
    hbonds: HBondParams = field(default_factory=HBondParams)
    mg: MgParams = field(default_factory=MgParams)
    stacking: StackingParams = field(default_factory=StackingParams)
    energy: EnergyParams = field(default_factory=EnergyParams)
    use_pbc: bool = False  # minimum-image distances (cleaned systems rarely need it)
    proximity_search_radius: float = 8.0  # max proximity cutoff usable at render time


@dataclass
class VisualParams:
    structure_groups: List[str] = field(default_factory=list)  # [] -> all groups
    trace_atoms: str = "C4',CA"  # per-residue priority list (P, C4', C1', CA ...)
    show_ions: bool = True
    align_frames: bool = True  # superpose each frame's trace onto the reference
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # deg X,Y,Z
    color_mode: str = "groups"  # "chain", "groups" or "proximity"
    base_color: str = "#b5b3ad"
    chain_palette: str = "tab10"
    proximity_set: str = "Target"  # group name or "sel:<selection>"
    proximity_cutoff: float = 4.0
    proximity_color: str = "#e34948"
    glow_mode: str = "fade"  # "none", "ever", "fade" or "current"
    glow_color: str = "#ffb000"
    fade_frames: int = 15
    glow_strength: float = 1.0
    ion_color: str = "#008300"
    line_width: float = 2.2
    point_size: float = 22.0
    depth_cue: bool = True
    theme: str = "light"  # "light" or "dark"
    font_size: float = 11.0
    show_time_label: bool = True
    label_every: int = 0  # residue-number tags on every N-th residue (0: off)
    label_residues: str = ""  # explicit residue numbers / ranges to tag, e.g. "1, 12, 24"
    title: str = ""


@dataclass
class PanelSpec:
    """One cell of the figure grid.

    ``type`` is a registered panel type (see :mod:`mdatk.render.panels`).
    ``series`` lists data keys such as ``"rmsd/Complex"``,
    ``"contacts/atom_pairs"`` or ``"contacts/per_residue_a"``.
    """

    type: str = "structure"
    series: List[str] = field(default_factory=list)
    title: str = ""
    options: Dict[str, Any] = field(default_factory=dict)


def _default_panels() -> List[PanelSpec]:
    return [
        PanelSpec("structure", [], "Structure"),
        PanelSpec("timeseries", ["rmsd/Complex"], "RMSD"),
        PanelSpec("timeseries", ["hbonds/count"], "Inter-strand H-bonds"),
        PanelSpec("heatmap", ["contacts/per_residue_a"], "Aptamer residue contacts"),
    ]


@dataclass
class LayoutParams:
    preset: str = "large_plus_3"
    panels: List[PanelSpec] = field(default_factory=_default_panels)


@dataclass
class VideoParams:
    enabled: bool = True
    filename: str = f"{PROJECT_ABBREV}_video.mp4"
    fps: int = 24
    width: int = 1920
    height: int = 1080
    dpi: int = 120
    codec: str = "libx264"
    crf: int = 20
    frame_stride: int = 1
    ffmpeg_path: str = "ffmpeg"


@dataclass
class ChartParams:
    enabled: bool = True
    series: List[str] = field(default_factory=list)  # [] -> every scalar series
    include_heatmaps: bool = False
    formats: List[str] = field(default_factory=lambda: ["png"])
    dpi: int = 300
    width: float = 8.0  # inches
    height: float = 4.5
    style: str = "mdatk"  # "mdatk" (built-in chrome) or any matplotlib style name
    color: str = ""  # "" -> first palette colour
    line_width: float = 1.5
    title: str = ""  # "" -> series label; "{label}" placeholders allowed
    xlabel: str = ""
    ylabel: str = ""
    smoothing: int = 0  # running-average window in frames (0 disables)
    show_target: bool = True
    prefix: str = f"{PROJECT_ABBREV}_chart"


@dataclass
class PDBParams:
    enabled: bool = False
    frames: str = ""  # e.g. "0, 10, 20-50:10, last"
    times: str = ""  # e.g. "1.5ns, 2000ps" (unitless -> inputs.time_unit)
    selection: str = ""  # group names (comma separated) or a selection; "" -> all
    prefix: str = f"{PROJECT_ABBREV}_frame"


@dataclass
class OutputParams:
    out_dir: str = f"{PROJECT_ABBREV}_output"
    cache_dir: str = ""  # "" -> <out_dir>/MDATK_cache
    video: VideoParams = field(default_factory=VideoParams)
    charts: ChartParams = field(default_factory=ChartParams)
    pdb: PDBParams = field(default_factory=PDBParams)
    write_params_log: bool = True
    timestamp_files: bool = True  # e.g. MDATK_video_2026-10-15_143205.mp4 (cache/log keep fixed names)


def _default_groups() -> List[ResidueGroup]:
    # Editable defaults for the reference aptamer/target system.
    return [
        ResidueGroup("Aptamer", "1-12", "", "#2a78d6"),
        ResidueGroup("Target", "13-24", "", "#eb6834"),
    ]


@dataclass
class Project:
    """Complete MDATK parameter set."""

    inputs: InputParams = field(default_factory=InputParams)
    cleaning: CleaningParams = field(default_factory=CleaningParams)
    groups: List[ResidueGroup] = field(default_factory=_default_groups)
    analyses: AnalysisParams = field(default_factory=AnalysisParams)
    visual: VisualParams = field(default_factory=VisualParams)
    layout: LayoutParams = field(default_factory=LayoutParams)
    output: OutputParams = field(default_factory=OutputParams)

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> Dict[str, Any]:
        data = {"format": PRESET_FORMAT, "format_version": PRESET_FORMAT_VERSION,
                "mdatk_version": __version__}
        data.update(_to_plain(self))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "Project":
        data = dict(data)
        for meta_key in ("format", "format_version", "mdatk_version"):
            data.pop(meta_key, None)
        project = _from_plain(cls, data, "project")
        if base_dir:
            project.resolve_paths(base_dir)
        return project

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: preset must contain a JSON object")
        if data.get("format") == "mdatk-params-log":  # replay a previous run
            data = data.get("parameters") or {}
        if data.get("format") not in (None, PRESET_FORMAT):
            raise ValueError(f"{path}: not an {PROJECT_ABBREV} preset (format={data.get('format')!r})")
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

    def group(self, name: str) -> ResidueGroup:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"no residue group named {name!r} "
                       f"(defined: {', '.join(g.name for g in self.groups) or 'none'})")

    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    @property
    def out_dir(self) -> str:
        return os.path.abspath(self.output.out_dir or f"{PROJECT_ABBREV}_output")

    @property
    def cache_dir(self) -> str:
        if self.output.cache_dir:
            return os.path.abspath(self.output.cache_dir)
        return os.path.join(self.out_dir, f"{PROJECT_ABBREV}_cache")

    def validate(self) -> List[str]:
        """Return a list of human readable problems (empty when valid)."""
        problems: List[str] = []
        if not self.inputs.topology:
            problems.append("No topology file selected.")
        elif not os.path.exists(self.inputs.topology):
            problems.append(f"Topology file not found: {self.inputs.topology}")
        if not self.inputs.trajectories:
            problems.append("No trajectory file selected.")
        for t in self.inputs.trajectories:
            if not os.path.exists(t):
                problems.append(f"Trajectory file not found: {t}")
        if self.inputs.step < 1:
            problems.append("Frame step must be >= 1.")
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            problems.append("Residue group names must be unique.")
        for g in self.groups:
            if not g.name.strip():
                problems.append("Residue groups need a name.")
            elif not re.fullmatch(r"[A-Za-z0-9_.\-]+", g.name):
                problems.append(f"Group name {g.name!r} may only contain letters, digits, "
                                "'_', '.' and '-' (no spaces).")
            if not (g.residues.strip() or g.selection.strip()):
                problems.append(f"Group {g.name!r} has neither residues nor a selection.")
        a = self.analyses
        for label, params in (("contacts", a.contacts), ("H-bonds", a.hbonds),
                              ("Mg", a.mg), ("stacking", a.stacking), ("energy", a.energy)):
            if not params.enabled:
                continue
            for gname in (params.group_a, params.group_b):
                if gname and gname not in names:
                    problems.append(f"{label}: unknown residue group {gname!r}.")
        if a.contacts.cutoff <= 0:
            problems.append("Contact cutoff must be positive.")
        if a.mg.inner_cutoff > a.mg.outer_cutoff:
            problems.append("Mg inner-sphere cutoff must not exceed the outer-sphere cutoff.")
        v = self.visual
        if v.color_mode not in ("chain", "groups", "proximity"):
            problems.append(f"Unknown colour mode {v.color_mode!r}.")
        if v.glow_mode not in ("none", "ever", "fade", "current"):
            problems.append(f"Unknown glow mode {v.glow_mode!r}.")
        if v.proximity_cutoff > a.proximity_search_radius:
            problems.append("Proximity cutoff exceeds the proximity search radius "
                            "(raise analyses.proximity_search_radius).")
        if self.output.video.fps <= 0:
            problems.append("Video FPS must be positive.")
        return problems


# ---------------------------------------------------------------------------
# Generic dataclass <-> plain JSON conversion
# ---------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    return obj


_ALIASES = {
    # convenience spellings accepted in hand-written presets
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
