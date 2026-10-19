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
from .citations import keys_for, pipeline_resources, project_resources, text_lines
from .cleaning import CleanResult, clean, cleaning_key
from .cleaning.cpptraj import MARKER, cpptraj_command
from .config import Project, file_fingerprint, layout_warnings
from .data import Dataset
from .io.cache import CacheStore
from .io.external import load_external
from .io.loader import frame_indices, load_universe
from .io.pdb_export import export_frames, frames_from_specs, resolve_export_selection
from .progress import Reporter

log = logging.getLogger(__name__)

PARAMS_LOG_NAME = f"{PROJECT_ABBREV}_params_log.toml"
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
    def __init__(self, project: Project, reporter: Optional[Reporter] = None,
                 project_path: Optional[str] = None):
        self.project = project
        self.project_path = project_path  # project file whose step log records the runs
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
        for text in layout_warnings(self.project):
            self.reporter.log(f"WARNING: {text}")

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

        rep = reporter or self.reporter
        path = self.output_path(stamped(self.project.output.video.filename, self.stamp()))
        fit = self.project.density_active() and len(self.project.density.transform) != 12
        if fit:
            self.ensure_density_fit(rep.sub(0.0, 0.1))
        try:
            self.last_video = render_video(dataset, self.project, path, rep.sub(0.1, 1.0) if fit else rep)
        finally:
            _release_gl()
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

        self.ensure_density_fit()
        try:
            return render_frame(dataset, self.project, pos, path)
        finally:
            _release_gl()

    def write_params_log(self, outputs: RunOutputs) -> str:
        from .projectfile import write_params_log

        path = os.path.join(self.project.out_dir, PARAMS_LOG_NAME)
        inputs = self._inputs
        run = {
            "project": PROJECT_TITLE,
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
        }
        keys = keys_for(project_resources(self.project))  # the whole project: every planner step + this run
        run["citation_keys"] = keys
        run["citations"] = text_lines(keys)
        write_params_log(path, self.project, run)
        return path

    def run(self, video: Optional[bool] = None, charts: Optional[bool] = None,
            pdb: Optional[bool] = None, force_analysis: bool = False) -> RunOutputs:
        """Run everything that is enabled (or explicitly requested)."""
        from .projectfile import logged

        p = self.project
        video = p.output.video.enabled if video is None else video
        charts = p.output.charts.enabled if charts is None else charts
        pdb = p.output.pdb.enabled if pdb is None else pdb
        self.check()
        with logged(self.project_path, "render", "Analyses and outputs",
                    keys_for(pipeline_resources(p))) as entry:
            outputs = self._run(video, charts, pdb, force_analysis)
            entry.files = [f for f in [outputs.video, *outputs.charts, *outputs.pdbs, outputs.params_log] if f]
            entry.summary = (f"{outputs.dataset.n_frames} frames analysed" if outputs.dataset else "no analyses") + \
                f"; {len(entry.files)} file(s) written to {p.out_dir}"
        return outputs

    def _run(self, video: bool, charts: bool, pdb: bool, force_analysis: bool) -> RunOutputs:
        p = self.project
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

    # --------------------------------------------------------------- density
    def density_frame_atoms(self, frame: str = "ref", reporter: Optional[Reporter] = None):
        """Superposed atom coordinates (structural-view frame) of one trajectory frame.

        Returns (xyz, elements, frame number, reference backbone points).  "ref" is the
        first analysed frame; any other frame is superposed onto it exactly like the video
        frames.  The reference points anchor a placement made in this frame.
        """
        from .analysis.structure import AtomsAnalysis, TraceAnalysis

        p = self.project
        u = self.universe()
        idx = frame_indices(u.trajectory.n_frames, p.inputs.start, p.inputs.stop, p.inputs.step)
        if not len(idx):
            raise PipelineError("no frames selected")
        ref = int(idx[0])
        spec = (frame or "ref").strip().lower()
        f = ref if spec in ("", "ref") else int(spec)
        if not 0 <= f < u.trajectory.n_frames:
            raise PipelineError(f"density fit frame {f} is outside the trajectory "
                                f"(0-{u.trajectory.n_frames - 1})")
        frames = np.array([ref] if f == ref else [ref, f], dtype=np.int64)
        inputs = self.resolve_inputs()
        ctx = AnalysisContext(u, p, inputs.topology, reporter or Reporter())
        (res, tr), _ = run_analyses(ctx, [AtomsAnalysis(ctx), TraceAnalysis(ctx)], frames, Reporter())
        return (res.arrays["coords"][-1].astype(np.float64), list(res.meta["elements"]), f,
                tr.arrays["coords"][0].astype(np.float64))

    def fit_density(self, search: Optional[str] = None, start: Optional[Sequence[float]] = None,
                    reporter: Optional[Reporter] = None):
        """Rigid-body fit of the structure into the density map; stores the placement
        in ``project.density`` and returns the :class:`~mdatk.density.FitResult`."""
        from .density import (FitError, RigidTransform, anchor_points, anchored, centre_transform, fit_model,
                              load_map_cached, map_level)
        from .io.mrc import MapError

        rep = reporter or self.reporter
        d = self.project.density
        if not d.map_path:
            raise PipelineError("no density map selected")
        try:
            dmap = load_map_cached(d.map_path)
        except (OSError, MapError) as exc:
            raise PipelineError(f"density map: {exc}") from exc
        search = search or d.search
        rep.log(f"Density fit ({'global search' if search == 'global' else 'local refinement'}) of "
                f"frame {d.fit_frame or 'ref'} into {os.path.basename(d.map_path)} at {d.resolution:g} Å")
        xyz, elements, frame, reference = self.density_frame_atoms(d.fit_frame, rep)
        t0 = None
        if search != "global":
            if start is not None:
                t0 = RigidTransform.from_list(start)
            elif len(d.transform) == 12:
                try:
                    t0 = anchored(d.transform, d.anchor, reference)
                except ValueError as exc:
                    raise PipelineError(f"density map: stored placement: {exc}") from exc
            else:
                t0 = centre_transform(dmap, map_level(dmap, d.level), xyz.mean(axis=0))
        try:
            res = fit_model(dmap, xyz, elements, d.resolution, start=t0, search=search,
                            angular_step=d.angular_step, progress=lambda f, m: rep.progress(f, m),
                            cancelled=rep.check_cancel)
        except FitError as exc:
            raise PipelineError(f"density fit: {exc}") from exc
        d.transform = res.transform.to_list()
        d.anchor = anchor_points(reference)
        d.fit_cc = round(float(res.cc), 4)
        d.fit_info = (f"{'global search' if search == 'global' else 'local refinement'}, frame {frame}, "
                      f"{len(xyz)} atoms, {d.resolution:g} Å: CC {res.cc:.3f}"
                      + (f" ({res.message})" if res.message else ""))
        rep.log(f"Density fit: {d.fit_info}")
        return res

    def ensure_density_fit(self, reporter: Optional[Reporter] = None) -> None:
        """Fit before rendering when a map is shown but not yet placed.  The map is an
        optional overlay: a failed fit is logged and the video is drawn without it."""
        from .progress import Cancelled

        p = self.project
        rep = reporter or self.reporter
        if p.density_active() and len(p.density.transform) != 12:
            if not p.density.auto_fit:
                rep.log("WARNING: the density map has no placement and automatic fitting is off; "
                        "it is not drawn")
                return
            try:
                self.fit_density(reporter=reporter)
            except Cancelled:
                raise
            except (PipelineError, OSError, ValueError) as exc:
                rep.log(f"WARNING: density map not drawn: {exc}")

    # --------------------------------------------------------------- preview
    def structure_preview(self) -> Dict[str, Any]:
        """Backbone trace of the first selected frame (for the orientation preview).

        Uses cleaned files when a matching cpptraj result is cached; never runs
        cpptraj itself.
        """
        from .analysis.structure import AtomsAnalysis, TraceAnalysis
        from .render.coloring import static_colors

        p = self.project
        inputs = self.cleaned_if_available()
        u = load_universe(inputs.topology, inputs.trajectories, p.inputs.dt_ps)
        idx = frame_indices(u.trajectory.n_frames, p.inputs.start, p.inputs.stop, p.inputs.step)[:1]
        ctx = AnalysisContext(u, p, inputs.topology, Reporter())
        objs = [TraceAnalysis(ctx), AtomsAnalysis(ctx)]
        (res, atoms), _ = run_analyses(ctx, objs, idx, Reporter())
        v = p.visual
        colors = static_colors(v.color_mode, res.arrays["chain"], res.arrays["group_index"],
                               [g.color for g in p.groups], v.base_color, v.chain_palette)
        return {"coords": res.arrays["coords"][0].astype(float),
                "ions": res.arrays["ion_coords"][0].astype(float),
                "links": res.arrays["links"], "colors": colors,
                "chain": res.arrays["chain"], "group_index": res.arrays["group_index"],
                "group_member": res.arrays["group_member"],
                "resid": res.arrays["resid"],
                "resname": res.arrays["resname"], "chain_name": res.arrays["chain_name"],
                "labels": res.meta.get("residue_labels", []),
                "atoms": {"coords": atoms.arrays["coords"][0].astype(float),
                          "names": atoms.meta["atom_names"], "elements": atoms.meta["elements"],
                          "resnames": atoms.meta["resnames"], "row": atoms.arrays["row"],
                          "bonds": atoms.arrays["bonds"]},
                "cleaned": inputs.topology != p.inputs.topology}


def _release_gl() -> None:
    """Free this thread's VTK scene (OpenGL contexts belong to their thread)."""
    import sys

    mod = sys.modules.get("mdatk.render.molecule.vtk_backend")
    if mod is not None:
        mod.release_thread()


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
