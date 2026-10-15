"""Companion command line interface for MDATK.

Every GUI action is available here, driven by the same parameter preset::

    mdatk demo ./demo                      # synthetic RNA system + preset
    mdatk render --params ./demo/demo_preset.json
    mdatk render --params run.json --set analyses.contacts.cutoff=5.0 --no-pdb
    mdatk charts --params MDATK_output/MDATK_params_log.json --from-cache
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import warnings
from typing import Any, List, Optional

from . import PROJECT_ABBREV, PROJECT_TITLE, __version__
from .config import Project
from .progress import Cancelled, Reporter

log = logging.getLogger("mdatk")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_override(project: Project, assignment: str) -> None:
    """Apply ``dotted.path=value`` (value parsed as JSON when possible).

    List elements are addressed by index (``groups.0.residues``, ``layout.panels.2``).
    Every assignment goes through the validating preset loader, so nested objects
    given as JSON become proper parameter objects and type errors are reported.
    """
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    path, raw = assignment.split("=", 1)
    parts = [p for p in path.strip().split(".") if p]
    if not parts:
        raise ValueError(f"empty key in {assignment!r}")
    obj: Any = project
    current: Any = None
    for n, part in enumerate(parts):
        if isinstance(obj, list):
            if not part.lstrip("-").isdigit() or not -len(obj) <= int(part) < len(obj):
                raise ValueError(f"{path}: no list element {part!r}")
            current = obj[int(part)]
        else:
            if not dataclasses.is_dataclass(obj) or not hasattr(obj, part):
                raise ValueError(f"unknown parameter {path!r}")
            current = getattr(obj, part)
        if n < len(parts) - 1:
            obj = current
    value = _parse_value(raw)
    if isinstance(current, bool) and not isinstance(value, bool):
        raise ValueError(f"{path} expects true/false")
    if isinstance(current, str) and not isinstance(value, str):
        value = raw
    data = project.to_dict()
    node: Any = data
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    fresh = Project.from_dict(data)
    for f in dataclasses.fields(Project):
        setattr(project, f.name, getattr(fresh, f.name))


def load_project(args) -> Project:
    if getattr(args, "params", None):
        project = Project.load(args.params)
    else:
        project = Project()
    if getattr(args, "top", None):
        project.inputs.topology = os.path.abspath(args.top)
    if getattr(args, "traj", None):
        project.inputs.trajectories = [os.path.abspath(t) for t in args.traj]
    if getattr(args, "out_dir", None):
        project.output.out_dir = os.path.abspath(args.out_dir)
    if getattr(args, "cache_dir", None):
        project.output.cache_dir = os.path.abspath(args.cache_dir)
    if getattr(args, "cpptraj", None):
        project.cleaning.cpptraj_path = args.cpptraj
    if getattr(args, "conda_env", None):
        project.cleaning.conda_env = args.conda_env
    if getattr(args, "clean", None) is not None:
        project.cleaning.enabled = args.clean
    if getattr(args, "frames", None):
        parts = (args.frames.split(":") + ["", "", ""])[:3]
        project.inputs.start = int(parts[0]) if parts[0] else 0
        project.inputs.stop = int(parts[1]) if parts[1] else None
        project.inputs.step = int(parts[2]) if parts[2] else 1
    for assignment in getattr(args, "set", None) or []:
        apply_override(project, assignment)
    return project


class ConsoleProgress:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.tty = sys.stderr.isatty()
        self.last = -1.0
        self.last_t = 0.0

    def __call__(self, fraction: float, message: str) -> None:
        if self.quiet:
            return
        now = time.time()
        if fraction < 1.0 and fraction - self.last < 0.01 and now - self.last_t < 2.0:
            return
        self.last, self.last_t = fraction, now
        line = f"[{fraction * 100:5.1f}%] {message}"
        if self.tty:
            sys.stderr.write("\r\033[K" + line + ("\n" if fraction >= 1.0 else ""))
        elif fraction >= 1.0 or int(fraction * 10) != int(self.last * 10):
            sys.stderr.write(line + "\n")
        sys.stderr.flush()


def make_reporter(args) -> Reporter:
    quiet = getattr(args, "quiet", False)
    verbose = getattr(args, "verbose", False)

    def log_line(text: str) -> None:
        if verbose or not text.startswith(("    ", "\t", "|")):
            if not quiet:
                sys.stderr.write(("\r\033[K" if sys.stderr.isatty() else "") + text + "\n")

    return Reporter(log_cb=log_line, progress_cb=ConsoleProgress(quiet))


def _pipeline(args):
    from .pipeline import Pipeline

    project = load_project(args)
    return project, Pipeline(project, make_reporter(args))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_gui(args) -> int:
    from .gui.app import main as gui_main

    return gui_main([args.params] if args.params else [])


def cmd_demo(args) -> int:
    from .demo import build_demo

    paths = build_demo(args.out_dir, n_frames=args.n_frames, dt_ps=args.dt,
                       solvent=not args.no_solvent)
    print(f"Synthetic system written to {os.path.abspath(args.out_dir)}:")
    for k, v in paths.items():
        print(f"  {k:10s} {v}")
    if args.run:
        ns = argparse.Namespace(params=paths["preset"], set=[], quiet=args.quiet, verbose=False)
        return cmd_render(_render_namespace(ns))
    print(f"Next: mdatk render --params {paths['preset']}   (or: mdatk gui {paths['preset']})")
    return 0


def _render_namespace(ns):
    for k in ("top", "traj", "out_dir", "cache_dir", "cpptraj", "clean", "frames"):
        setattr(ns, k, None)
    ns.no_video = ns.no_charts = ns.no_pdb = False
    ns.video = ns.charts = ns.pdb = False
    ns.force = False
    ns.from_cache = False
    return ns


def cmd_init(args) -> int:
    project = load_project(args)
    if os.path.exists(args.output) and not args.overwrite:
        print(f"{args.output} exists (use --overwrite)", file=sys.stderr)
        return 1
    project.save(args.output)
    print(f"Wrote default parameters to {args.output}")
    return 0


def cmd_info(args) -> int:
    from .io.loader import load_universe, system_summary
    from .selections import select

    project = load_project(args)
    if not project.inputs.topology:
        print("info needs --params or --top", file=sys.stderr)
        return 2
    u = load_universe(project.inputs.topology, project.inputs.trajectories, project.inputs.dt_ps)
    info = system_summary(u)
    groups = {}
    for g in project.groups:
        try:
            ag = select(u, f"group:{g.name}", project.groups)
            groups[g.name] = {"atoms": int(ag.n_atoms), "residues": int(ag.residues.n_residues),
                              "first": f"{ag.residues[0].resname}{ag.residues[0].resid}",
                              "last": f"{ag.residues[-1].resname}{ag.residues[-1].resid}"}
        except Exception as exc:  # noqa: BLE001 - report every problem
            groups[g.name] = {"error": str(exc)}
    info["groups"] = groups
    info["problems"] = project.validate()
    print(json.dumps(info, indent=2))
    return 0


def cmd_list(args) -> int:
    from .analysis import ANALYSES
    from .render.layout import PRESET_LABELS
    from .render.panels import PANEL_TYPES

    print("Analyses:")
    for name, cls in ANALYSES.items():
        tag = " (internal, used by the structure panel)" if cls.internal else ""
        print(f"  {name:10s} {cls.label}{tag}")
    print("\nConda environments with cpptraj:")
    from .cleaning import conda
    envs = conda.find_cpptraj_envs()
    for env in envs:
        print(f"  {env.name:14s} {env.prefix}")
    if not envs:
        print("  (none found)")
    print("\nPanel types:")
    for name, cls in PANEL_TYPES.items():
        print(f"  {name:12s} {cls.label} - {cls.description}")
    print("\nLayout presets:")
    for name, label in PRESET_LABELS.items():
        print(f"  {name:14s} {label}")
    print("\nData keys for panels/charts look like '<analysis>/<output>', e.g. rmsd/Complex, "
          "contacts/atom_pairs,\ncontacts/per_residue_a (matrix), contacts/map (contact map). "
          "Add residue groups after '@' to\nrun an analysis on other groups: "
          "contacts@Target+Mg/atom_pairs, rmsd@Aptamer/Backbone.\n"
          "Run 'mdatk analyze' to list the keys of a system.")
    return 0


def cmd_clean(args) -> int:
    project, pipe = _pipeline(args)
    if not project.cleaning.enabled:
        project.cleaning.enabled = True
    res = pipe.resolve_inputs(force_clean=args.force)
    c = res.cleaned
    print(json.dumps({"topology": res.topology, "trajectory": res.trajectories,
                      "script": c.script if c else None, "log": c.log if c else None,
                      "reused": c.reused if c else None}, indent=2))
    return 0


def cmd_analyze(args) -> int:
    project, pipe = _pipeline(args)
    pipe.check()
    names = [n.strip() for n in args.only.split(",")] if args.only else None
    ds = pipe.analyze(names, force=args.force)
    print(json.dumps({"cache_manifest": pipe.cache.manifest_path, "n_frames": ds.n_frames,
                      "data_keys": ds.describe()}, indent=2))
    return 0


def _dataset(pipe, args):
    if getattr(args, "from_cache", False):
        return pipe.load_cached_dataset()
    pipe.check()
    return pipe.analyze()


def cmd_render(args) -> int:
    project, pipe = _pipeline(args)
    if args.from_cache:
        os.makedirs(project.out_dir, exist_ok=True)
        ds = pipe.load_cached_dataset()
        out = {}
        if not args.no_video and (project.output.video.enabled or args.video):
            out["video"] = pipe.render_video(ds)
        if not args.no_charts and (project.output.charts.enabled or args.charts):
            out["charts"] = pipe.render_charts(ds)
        print(json.dumps(out, indent=2))
        return 0
    video = False if args.no_video else (True if args.video else None)
    charts = False if args.no_charts else (True if args.charts else None)
    pdb = False if args.no_pdb else (True if args.pdb else None)
    outputs = pipe.run(video=video, charts=charts, pdb=pdb, force_analysis=args.force)
    print(json.dumps(outputs.summary(), indent=2))
    return 0


def cmd_charts(args) -> int:
    project, pipe = _pipeline(args)
    if args.series:
        project.output.charts.series = [s.strip() for s in args.series.split(",") if s.strip()]
    if args.formats:
        project.output.charts.formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.dpi:
        project.output.charts.dpi = args.dpi
    ds = _dataset(pipe, args)
    print(json.dumps({"charts": pipe.render_charts(ds)}, indent=2))
    return 0


def cmd_export_pdb(args) -> int:
    project, pipe = _pipeline(args)
    pp = project.output.pdb
    if args.frame_list is not None:
        pp.frames = args.frame_list
    if args.times is not None:
        pp.times = args.times
    if args.selection is not None:
        pp.selection = args.selection
    pipe.check()
    os.makedirs(project.out_dir, exist_ok=True)
    print(json.dumps({"pdbs": pipe.export_pdbs()}, indent=2))
    return 0


def cmd_preview(args) -> int:
    project, pipe = _pipeline(args)
    ds = _dataset(pipe, args)
    pos = min(max(0, args.frame), ds.n_frames - 1)
    out = args.output or os.path.join(project.out_dir, f"{PROJECT_ABBREV}_preview_{pos:05d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(pipe.render_preview(ds, pos, out))
    return 0


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------


def _common(p: argparse.ArgumentParser, inputs: bool = True) -> None:
    p.add_argument("--params", "-p", help="parameter preset (JSON) or a previous params log")
    if inputs:
        p.add_argument("--top", help="topology file (.parm7, .pdb, ...)")
        p.add_argument("--traj", nargs="+", help="trajectory file(s) (.nc, .dcd, ...)")
        p.add_argument("--frames", help="frame selection start:stop:step (0-based, stop exclusive)")
        p.add_argument("--clean", dest="clean", action="store_true", default=None,
                       help="enable cpptraj cleaning")
        p.add_argument("--no-clean", dest="clean", action="store_false", help="disable cleaning")
        p.add_argument("--cpptraj", help="path to the cpptraj executable")
        p.add_argument("--conda-env", help="conda environment providing cpptraj (e.g. AmberTools26); "
                       "cpptraj then runs via `conda run`")
    p.add_argument("--out-dir", "-o", help="output directory")
    p.add_argument("--cache-dir", help="cache directory (default <out-dir>/MDATK_cache)")
    p.add_argument("--set", "-s", action="append", metavar="KEY=VALUE",
                   help="override a parameter, e.g. analyses.contacts.cutoff=5 (repeatable)")
    p.add_argument("--quiet", "-q", action="store_true", help="only print the result")
    p.add_argument("--verbose", "-v", action="store_true", help="also print full cpptraj output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdatk", description=f"{PROJECT_TITLE} - analysis and video rendering of MD "
        "trajectories (RNA systems).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("gui", help="start the GTK 3 interface")
    p.add_argument("params", nargs="?", help="preset to open")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("demo", help="write a synthetic RNA aptamer/target system and preset")
    p.add_argument("out_dir")
    p.add_argument("--n-frames", type=int, default=120)
    p.add_argument("--dt", type=float, default=10.0, help="ps between frames")
    p.add_argument("--no-solvent", action="store_true", help="omit water / Na+ (no cleaning needed)")
    p.add_argument("--run", action="store_true", help="render the demo immediately")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("init-params", help="write a parameter preset with defaults")
    p.add_argument("output")
    p.add_argument("--overwrite", action="store_true")
    _common(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("info", help="summarise a system and check residue groups")
    _common(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("list", help="list analyses, panel types and layout presets")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("clean", help="run (or reuse) cpptraj stripping/imaging/fitting")
    _common(p)
    p.add_argument("--force", action="store_true", help="re-run cpptraj even if cached")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("analyze", help="compute (or load cached) analyses")
    _common(p)
    p.add_argument("--only", help="comma separated analysis names")
    p.add_argument("--force", action="store_true", help="ignore cached results")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("render", help="full run: clean, analyse, video, charts, PDBs")
    _common(p)
    for name in ("video", "charts", "pdb"):
        p.add_argument(f"--{name}", action="store_true", help=f"force {name} output on")
        p.add_argument(f"--no-{name}", action="store_true", help=f"skip {name} output")
    p.add_argument("--force", action="store_true", help="recompute analyses")
    p.add_argument("--from-cache", action="store_true",
                   help="re-render from the cache manifest without touching trajectories")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("charts", help="write standalone time charts")
    _common(p)
    p.add_argument("--series", help="comma separated data keys (default: all)")
    p.add_argument("--formats", help="comma separated: png,svg,pdf")
    p.add_argument("--dpi", type=int)
    p.add_argument("--from-cache", action="store_true")
    p.set_defaults(func=cmd_charts)

    p = sub.add_parser("export-pdb", help="write selected frames as PDB files")
    _common(p)
    p.add_argument("--frame-list", help='frames, e.g. "0, 10, 20-50:10, last"')
    p.add_argument("--times", help='times, e.g. "1.5ns, 2000ps"')
    p.add_argument("--selection", help="group names (comma separated) or a selection")
    p.set_defaults(func=cmd_export_pdb)

    p = sub.add_parser("preview", help="render one composite frame as an image")
    _common(p)
    p.add_argument("--frame", type=int, default=0, help="position in the analysed frames")
    p.add_argument("--output", help="image path (.png/.svg/.pdf)")
    p.add_argument("--from-cache", action="store_true")
    p.set_defaults(func=cmd_preview)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore", module="MDAnalysis")
    try:
        return int(args.func(args) or 0)
    except Cancelled:
        print("cancelled", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        if os.environ.get("MDATK_DEBUG"):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
