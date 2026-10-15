"""End-to-end orchestration shared by the GUI and the CLI.

    clean (optional, cached) -> load -> analyse (cached per analysis) -> outputs

Each stage reports through a :class:`~mdatk.progress.Reporter`, so the GUI can
show one progress bar plus the cpptraj / pipeline log, and every stage checks
for cancellation.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import PROJECT_ABBREV, PROJECT_TITLE, __version__
from .analysis import ANALYSES, AnalysisContext, run_analyses
from .analysis.base import analysis_class, instance_project, parse_instance, split_key
from .cleaning import CleanResult, clean, cleaning_key
from .cleaning.cpptraj import MARKER, cpptraj_command
from .config import Project, file_fingerprint
from .data import Dataset
from .io.cache import CacheStore, write_json
from .io.external import load_external
from .io.loader import frame_indices, load_universe
from .io.pdb_export import export_frames, frames_from_specs, resolve_export_selection
from .progress import Reporter

log = logging.getLogger(__name__)

PARAMS_LOG_NAME = f"{PROJECT_ABBREV}_params_log.json"
STAMP_FORMAT = "%Y-%m-%d_%H%M%S"


def stamped(name: str, stamp: str) -> str:
    """``MDATK_video.mp4`` -> ``MDATK_video_2026-10-15_143205.mp4``."""
    if not stamp:
        return name
    root, ext = os.path.splitext(name)
    return f"{root}_{stamp}{ext}"


class PipelineError(RuntimeError):
    pass


@dataclass
class ResolvedInputs:
    topology: str
    trajectories: List[str]
    cleaned: Optional[CleanResult] = None


@dataclass
class RunOutputs:
    video: Optional[str] = None
    charts: List[str] = field(default_factory=list)
    pdbs: List[str] = field(default_factory=list)
    manifest: Optional[str] = None
    params_log: Optional[str] = None
    dataset: Optional[Dataset] = None

    def summary(self) -> Dict[str, Any]:
        return {"video": self.video, "charts": self.charts, "pdbs": self.pdbs,
                "cache_manifest": self.manifest, "params_log": self.params_log}


class Pipeline:
    def __init__(self, project: Project, reporter: Optional[Reporter] = None):
        self.project = project
        self.reporter = reporter or Reporter()
        self.cache = CacheStore(project.cache_dir)
        self._inputs: Optional[ResolvedInputs] = None
        self._universe = None
        self.analysis_keys: Dict[str, str] = {}
        self._stamp: Optional[str] = None
        self.last_video: Optional[str] = None

    # ------------------------------------------------------------ file names
    def stamp(self) -> str:
        """Timestamp shared by all outputs of one run ('' when disabled)."""
        if not self.project.output.timestamp_files:
            return ""
        if self._stamp is None:
            self._stamp = _dt.datetime.now().strftime(STAMP_FORMAT)
        return self._stamp

    def new_stamp(self) -> None:
        self._stamp = None

    # ---------------------------------------------------------------- inputs
    def check(self) -> None:
        problems = self.project.validate()
        if problems:
            raise PipelineError("Invalid parameters:\n  - " + "\n  - ".join(problems))

    def resolve_inputs(self, reporter: Optional[Reporter] = None, force_clean: bool = False) -> ResolvedInputs:
        if self._inputs is not None and not force_clean:
            return self._inputs
        rep = reporter or self.reporter
        p = self.project
        if p.cleaning.enabled:
            rep.log("Cleaning trajectory with cpptraj ...")
            res = clean(p.cleaning, p.inputs.topology, p.inputs.trajectories, p.cache_dir,
                        log_cb=rep.raw_log, progress_cb=lambda f: rep.progress(f, "cpptraj"),
                        cancel_event=rep.cancel_event, force=force_clean)
            self._inputs = ResolvedInputs(res.topology, [res.trajectory], res)
        else:
            self._inputs = ResolvedInputs(p.inputs.topology, list(p.inputs.trajectories), None)
        self._universe = None
        rep.progress(1.0, "inputs ready")
        return self._inputs

    def cleaned_if_available(self) -> ResolvedInputs:
        """Cleaned files when a matching cleaning result is cached, else the raw inputs."""
        p = self.project
        if p.cleaning.enabled:
            try:
                key = cleaning_key(p.cleaning, p.inputs.topology, p.inputs.trajectories)
                out = p.cleaning.output_dir or os.path.join(p.cache_dir, "cleaned", key)
                marker = os.path.join(out, MARKER)
                from .cleaning.cpptraj import CLEAN_PARM, CLEAN_REF, CLEAN_TRAJ
                top = os.path.join(out, CLEAN_PARM)
                if not os.path.exists(top):
                    top = os.path.join(out, CLEAN_REF)
                traj = os.path.join(out, CLEAN_TRAJ)
                if os.path.exists(marker) and os.path.exists(top) and os.path.exists(traj):
                    return ResolvedInputs(top, [traj], None)
            except OSError:
                pass
        return ResolvedInputs(p.inputs.topology, list(p.inputs.trajectories), None)

    def universe(self):
        if self._universe is None:
            inputs = self.resolve_inputs()
            self._universe = load_universe(inputs.topology, inputs.trajectories, self.project.inputs.dt_ps)
        return self._universe

    def input_fingerprint(self, inputs: ResolvedInputs) -> Dict[str, Any]:
        p = self.project.inputs
        return {"topology": file_fingerprint(inputs.topology),
                "trajectories": [file_fingerprint(t) for t in inputs.trajectories],
                "frames": {"start": p.start, "stop": p.stop, "step": p.step}, "dt_ps": p.dt_ps}

    # -------------------------------------------------------------- analyses
    def required_analyses(self, video: Optional[bool] = None, charts: Optional[bool] = None) -> List[str]:
        from .render.panels import required_analyses as panel_requirements

        p = self.project
        video = p.output.video.enabled if video is None else video
        charts = p.output.charts.enabled if charts is None else charts
        names = {n for n, cls in ANALYSES.items() if not cls.internal and cls.enabled(p)}
        if video:
            names |= panel_requirements(p)
        if charts:
            names |= {split_key(k)[0] for k in p.output.charts.series if "/" in k}
        names.discard("external")
        usable = []
        for inst in names:
            try:  # the affected panels / charts report "No data"
                instance_project(p, inst)
            except (KeyError, ValueError) as exc:
                self.reporter.log(f"WARNING: skipping data source {inst!r}: {exc}")
                continue
            usable.append(inst)
        order = list(ANALYSES)
        return sorted(usable, key=lambda i: (order.index(parse_instance(i)[0]), i))

    def analyze(self, names: Optional[Sequence[str]] = None, force: bool = False,
                reporter: Optional[Reporter] = None) -> Optional[Dataset]:
        rep = reporter or self.reporter
        p = self.project
        names = list(self.required_analyses() if names is None else names)
        inputs = self.resolve_inputs(rep.sub(0.0, 0.02))
        fp = self.input_fingerprint(inputs)
        projects = {n: instance_project(p, n) for n in names}
        keys = {n: analysis_class(n).cache_key(projects[n], fp) for n in names}
        self.analysis_keys = dict(keys)
        results = {}
        frames = time_ps = None
        for n in names:
            if force:
                break
            loaded = self.cache.load_result(n, keys[n])
            if loaded is not None:
                res, fr, tm = loaded
                results[n] = res
                frames, time_ps = fr, tm
                rep.log(f"{analysis_class(n).label}: loaded from cache ({n}-{keys[n]})")
        missing = [n for n in names if n not in results]
        if missing:
            u = self.universe()
            idx = frame_indices(u.trajectory.n_frames, p.inputs.start, p.inputs.stop, p.inputs.step)
            objs = [analysis_class(n)(AnalysisContext(u, projects[n], inputs.topology, rep), n)
                    for n in missing]
            new, times = run_analyses(objs[0].ctx, objs, idx, rep.sub(0.02, 1.0))
            for res in new:
                res.key = keys[res.name]
                self.cache.save_result(res, idx, times)
                results[res.name] = res
            frames, time_ps = idx, times
        if frames is None:
            # nothing requested: still provide the frame axis
            u = self.universe()
            frames = frame_indices(u.trajectory.n_frames, p.inputs.start, p.inputs.stop, p.inputs.step)
            time_ps = np.array([u.trajectory[int(f)].time for f in frames])
        dataset = Dataset(frames, time_ps, {n: results[n] for n in names if n in results},
                          time_unit=p.inputs.time_unit)
        self.apply_display_overrides(dataset)
        if p.inputs.external:
            try:
                dataset.add(load_external(p.inputs.external, dataset))
            except (OSError, ValueError) as exc:
                raise PipelineError(f"external data: {exc}") from exc
        self.cache.write_manifest(
            system={"inputs": fp, "cleaning_key": inputs.cleaned.key if inputs.cleaned else None,
                    "topology": inputs.topology, "trajectories": inputs.trajectories},
            frame_selection={"start": p.inputs.start, "stop": p.inputs.stop, "step": p.inputs.step},
            dataset=dataset, groups=[vars(g) for g in p.groups],
            entries={n: keys[n] for n in names})
        rep.progress(1.0, "analysis done")
        return dataset

    def apply_display_overrides(self, dataset: Dataset) -> None:
        """Parameters that change presentation but not the cached numbers."""
        for name, res in dataset.results.items():
            if parse_instance(name)[0] == "energy" and "total" in res.outputs.get("series", {}):
                res.outputs["series"]["total"]["target"] = self.project.analyses.energy.target_value

    def load_cached_dataset(self) -> Dataset:
        """Dataset of the most recent run, straight from the cache manifest."""
        ds = self.cache.load_dataset_from_manifest(self.project.inputs.time_unit)
        self.apply_display_overrides(ds)
        return ds

    # --------------------------------------------------------------- outputs
    def output_path(self, name: str) -> str:
        return name if os.path.isabs(name) else os.path.join(self.project.out_dir, name)

    def render_video(self, dataset: Dataset, reporter: Optional[Reporter] = None) -> str:
        from .render.video import render_video

        path = self.output_path(stamped(self.project.output.video.filename, self.stamp()))
        self.last_video = render_video(dataset, self.project, path, reporter or self.reporter)
        return self.last_video

    def render_charts(self, dataset: Dataset, reporter: Optional[Reporter] = None) -> List[str]:
        from .render.charts import render_charts

        return render_charts(dataset, self.project, self.project.out_dir, reporter or self.reporter,
                             self.stamp())

    def export_pdbs(self, reporter: Optional[Reporter] = None) -> List[str]:
        rep = reporter or self.reporter
        p = self.project
        pp = p.output.pdb
        u = self.universe()
        frames = frames_from_specs(u, pp.frames, pp.times, p.inputs.time_unit, p.inputs.dt_ps)
        if not frames:
            raise PipelineError("PDB export enabled but no frames or times were given")
        atoms = resolve_export_selection(u, pp.selection, p.groups)
        rep.log(f"Exporting {len(frames)} frame(s) ({atoms.n_atoms} atoms) as PDB")
        return export_frames(u, frames, atoms, p.out_dir, pp.prefix,
                             progress=lambda f, m: rep.progress(f, m), stamp=self.stamp())

    def render_preview(self, dataset: Dataset, pos: int, path: str) -> str:
        from .render.video import render_frame

        return render_frame(dataset, self.project, pos, path)

    def write_params_log(self, outputs: RunOutputs) -> str:
        path = os.path.join(self.project.out_dir, PARAMS_LOG_NAME)
        inputs = self._inputs
        write_json(path, {
            "format": "mdatk-params-log",
            "project": PROJECT_TITLE,
            "mdatk_version": __version__,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "environment": environment_info(self.project),
            "inputs": {
                "topology": inputs.topology if inputs else None,
                "trajectories": inputs.trajectories if inputs else None,
                "cleaning": ({"key": inputs.cleaned.key, "script": inputs.cleaned.script,
                              "log": inputs.cleaned.log, "reused": inputs.cleaned.reused}
                             if inputs and inputs.cleaned else None),
            },
            "cache_dir": self.project.cache_dir,
            "analysis_keys": self.analysis_keys,
            "timestamp": self._stamp,
            "outputs": outputs.summary(),
            "parameters": self.project.to_dict(),
        })
        return path

    def run(self, video: Optional[bool] = None, charts: Optional[bool] = None,
            pdb: Optional[bool] = None, force_analysis: bool = False) -> RunOutputs:
        """Run everything that is enabled (or explicitly requested)."""
        p = self.project
        video = p.output.video.enabled if video is None else video
        charts = p.output.charts.enabled if charts is None else charts
        pdb = p.output.pdb.enabled if pdb is None else pdb
        self.check()
        rep = self.reporter
        os.makedirs(p.out_dir, exist_ok=True)
        outputs = RunOutputs()
        start = _dt.datetime.now()
        self.new_stamp()
        rep.log(f"{PROJECT_TITLE} {__version__} - output directory {p.out_dir}")
        self.resolve_inputs(rep.sub(0.0, 0.15))
        names = self.required_analyses(video, charts)
        dataset = None
        if names or video or charts:
            dataset = self.analyze(names, force=force_analysis, reporter=rep.sub(0.15, 0.55))
            outputs.dataset = dataset
            outputs.manifest = self.cache.manifest_path
        if video:
            outputs.video = self.render_video(dataset, rep.sub(0.55, 0.92))
        if charts:
            outputs.charts = self.render_charts(dataset, rep.sub(0.92, 0.97))
        if pdb:
            outputs.pdbs = self.export_pdbs(rep.sub(0.97, 1.0))
        if p.output.write_params_log:
            outputs.params_log = self.write_params_log(outputs)
        rep.progress(1.0, "done")
        rep.log(f"Finished in {(_dt.datetime.now() - start).total_seconds():.1f} s")
        return outputs

    # --------------------------------------------------------------- preview
    def structure_preview(self) -> Dict[str, Any]:
        """Backbone trace of the first selected frame (for the orientation preview).

        Uses cleaned files when a matching cpptraj result is cached; never runs
        cpptraj itself.
        """
        from .analysis.structure import TraceAnalysis
        from .render.coloring import static_colors

        p = self.project
        inputs = self.cleaned_if_available()
        u = load_universe(inputs.topology, inputs.trajectories, p.inputs.dt_ps)
        idx = frame_indices(u.trajectory.n_frames, p.inputs.start, p.inputs.stop, p.inputs.step)[:1]
        ctx = AnalysisContext(u, p, inputs.topology, Reporter())
        ana = TraceAnalysis(ctx)
        (res,), _ = run_analyses(ctx, [ana], idx, Reporter())
        v = p.visual
        colors = static_colors(v.color_mode, res.arrays["chain"], res.arrays["group_index"],
                               [g.color for g in p.groups], v.base_color, v.chain_palette)
        return {"coords": res.arrays["coords"][0].astype(float),
                "ions": res.arrays["ion_coords"][0].astype(float),
                "links": res.arrays["links"], "colors": colors,
                "chain": res.arrays["chain"], "group_index": res.arrays["group_index"],
                "resid": res.arrays["resid"],
                "labels": res.meta.get("residue_labels", []),
                "cleaned": inputs.topology != p.inputs.topology}


def _tool_version(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        text = (out.stdout or out.stderr).strip().splitlines()
        return text[0] if text else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info(project: Optional[Project] = None) -> Dict[str, Any]:
    import matplotlib
    import MDAnalysis
    import scipy

    info: Dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "MDAnalysis": MDAnalysis.__version__,
    }
    ffmpeg = project.output.video.ffmpeg_path if project else "ffmpeg"
    info["ffmpeg"] = _tool_version([ffmpeg, "-version"])
    if project and project.cleaning.enabled:
        try:
            info["cpptraj"] = {"command": " ".join(cpptraj_command(project.cleaning))}
        except Exception as exc:  # noqa: BLE001 - informational only
            info["cpptraj"] = {"error": str(exc)}
    return info
