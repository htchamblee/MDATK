"""Companion command line interface for MDATK.

Every GUI action is available here, driven by the same TOML project file::

    mdatk demo ./demo                      # synthetic RNA system + project file
    mdatk render -p ./demo/demo.mdatk.toml
    mdatk render -p run.mdatk.toml --set analyses.contacts.cutoff=5.0 --no-pdb
    mdatk charts -p MDATK_output/MDATK_params_log.toml --from-cache
    mdatk workflow generate amber-md -i topology=sys.parm7 -i coordinates=sys.rst7 --set engine=pmemd.MPI
    mdatk workflow write -p plan.mdatk.toml
    mdatk workflow run -p plan.mdatk.toml --step 3
    mdatk workflow status -p plan.mdatk.toml --adopt
    mdatk pdb clean model.pdb -o model_amber.pdb --scheme amber --chain-reference start.pdb
    mdatk build helix --seq GGCAU --chirality L -o helix.pdb
    mdatk dict finish acedrg_out/ --code 0A5 -o 0A5.cif

``rnado`` (RNAdo, MDATK's workflow planner) opens the planner on a project; ``rnado write -p plan.mdatk.toml``
and the other actions are the same as ``mdatk workflow write ...`` (see :func:`rnado_main`).
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
    Every assignment goes through the validating project loader, so nested objects
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


def load_project(args, resolve: bool = True) -> Project:
    """The project of ``-p`` (or the defaults) with the command-line overrides; with ``resolve`` the
    paths still relative after loading (defaults, --set values) are taken relative to the working
    directory, so a parameter log of the run records where the files really are."""
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
    if resolve:
        project.resolve_paths(os.getcwd())
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
    return project, Pipeline(project, make_reporter(args), project_path=_project_file(args))


def _project_file(args) -> Optional[str]:
    """The project file named by --project (not a parameter log), whose step log records runs."""
    from .projectfile import is_params_log
    path = getattr(args, "params", None)
    if not path or is_params_log(path):
        return None
    return path


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_gui(args) -> int:
    if args.rnado or args.open:
        argv = [args.params] if args.params else []
        return _rnado_window(argv + (["--open", args.open] if args.open else []))
    from .gui.app import main as gui_main

    return gui_main([args.params] if args.params else [])


def cmd_desktop(args) -> int:
    """'mdatk desktop install|uninstall|status': MDATK and RNAdo in the applications menu
    (:mod:`mdatk.launchers`)."""
    from . import launchers
    return launchers.run(args.desktop_command, replace=getattr(args, "replace", False))


def cmd_demo(args) -> int:
    from .demo import build_demo

    paths = build_demo(args.out_dir, n_frames=args.n_frames, dt_ps=args.dt,
                       solvent=not args.no_solvent)
    print(f"Synthetic system written to {os.path.abspath(args.out_dir)}:")
    for k, v in paths.items():
        print(f"  {k:10s} {v}")
    if args.run:
        ns = argparse.Namespace(params=paths["project"], set=[], quiet=args.quiet, verbose=False)
        return cmd_render(_render_namespace(ns))
    print(f"Next: mdatk render -p {paths['project']}   (or: mdatk gui {paths['project']})")
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
    project = load_project(args, resolve=False)  # relative paths stay relative to the new file
    if os.path.exists(args.output) and not args.overwrite:
        print(f"{args.output} exists (use --overwrite)", file=sys.stderr)
        return 1
    project.save(args.output)
    print(f"Wrote a project file with default settings to {args.output}")
    return 0


def cmd_info(args) -> int:
    from .io.loader import load_universe, system_summary
    from .selections import select

    project = load_project(args)
    if not project.inputs.topology:
        print("info needs --project or --top", file=sys.stderr)
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
    pipe.check(pdb=False)  # analyze() runs what the video and the charts need
    names = [n.strip() for n in args.only.split(",")] if args.only else None
    ds = pipe.analyze(names, force=args.force)
    print(json.dumps({"cache_manifest": pipe.cache.manifest_path, "n_frames": ds.n_frames,
                      "data_keys": ds.describe()}, indent=2))
    return 0


def _dataset(pipe, args):
    if getattr(args, "from_cache", False):
        return pipe.load_cached_dataset()
    pipe.check(pdb=False)  # analyze() runs what the video and the charts need
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
    pipe.check(video=False, charts=False, pdb=True)
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


# ---------------------------------------------------------------------------
# workflow: script generators and RNAdo's plan
# ---------------------------------------------------------------------------


def apply_tool_override(tool, settings: dict, assignment: str) -> None:
    """KEY=VALUE for a generator setting; records of repeated sections use
    ``section.index.key`` (e.g. ``stages.5.nstlim=1000``)."""
    if "=" not in assignment:
        raise ValueError(f"expected KEY=VALUE, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    key = key.strip()
    parts = key.split(".")
    flat = tool.flat_options()
    if len(parts) == 1 and key in flat:
        settings[key] = flat[key].coerce(_parse_value(raw) if flat[key].kind not in ("text", "multiline", "file")
                                         else raw)
        return
    sections = {sec.key: sec for sec in tool.sections if sec.repeat}
    if parts[0] in sections:
        sec = sections[parts[0]]
        if len(parts) == 1:
            value = _parse_value(raw)
            if not isinstance(value, list):
                raise ValueError(f"{key}: expected a JSON list of records")
            settings[key] = tool.normalise({key: value})[key]
            return
        if len(parts) == 3 and parts[1].isdigit():
            records = settings.setdefault(sec.key, sec.default_records())
            k = int(parts[1])
            if not 0 <= k < len(records):
                raise ValueError(f"{key}: there is no record {k} (0-{len(records) - 1})")
            opt = sec.option(parts[2])
            records[k][opt.key] = opt.coerce(_parse_value(raw) if opt.kind not in ("text", "multiline", "file")
                                             else raw)
            return
    raise ValueError(f"{tool.name} has no setting {key!r} (see: mdatk workflow tools {tool.name})")


def _print_issues(plan, quiet: bool = False) -> int:
    n_err = 0
    for nid, issue in plan.all_issues():
        if issue.level == "error":
            n_err += 1
        if not quiet or issue.level == "error":
            label = plan.steps[nid].label if nid in plan.steps else "plan"
            print(f"{issue.level.upper():8s} {label}: {issue.message}", file=sys.stderr)
    return n_err


def cmd_density_info(args) -> int:
    from .density import map_level
    from .io.mrc import read_map

    dmap = read_map(args.map)
    st = dmap.stats()
    info = {"map": os.path.abspath(args.map), "grid_xyz": list(dmap.shape_xyz),
            "voxel_A": [round(float(v), 4) for v in dmap.voxel],
            "origin_A": [round(float(v), 3) for v in dmap.origin],
            "extent_A": [round(float(v), 3) for v in dmap.extent],
            **{k: round(v, 6) for k, v in st.items()},
            "level_2sigma": round(map_level(dmap, 2.0), 6)}
    print(json.dumps(info, indent=2))
    return 0


def _density_overrides(project, args) -> None:
    d = project.density
    if args.map:
        if os.path.abspath(args.map) != d.map_path:
            d.transform, d.fit_cc, d.fit_info = [], 0.0, ""
        d.map_path = os.path.abspath(args.map)
    for name in ("resolution", "level"):
        if getattr(args, name, None) is not None:
            setattr(d, name, getattr(args, name))
    if getattr(args, "frame", None) is not None:
        d.fit_frame = str(args.frame)
    if getattr(args, "step", None) is not None:
        d.angular_step = args.step


def cmd_density_fit(args) -> int:
    project, pipe = _pipeline(args)
    _density_overrides(project, args)
    problems = project.density_problems()
    if not project.density.map_path:
        problems.append("no density map: give --map or set density.map_path in the project")
    if problems:
        raise ValueError("; ".join(problems))
    pipe.check(video=False, charts=False, pdb=False, density=True)
    res = pipe.fit_density(search=args.search)
    d = project.density
    out = {"cc": round(float(res.cc), 4), "start": res.start, "info": d.fit_info,
           "transform": d.transform}
    if args.save:
        target = args.save if isinstance(args.save, str) else args.params
        if not target:
            raise ValueError("--save needs a path when no --project was given")
        project.save(target)
        out["saved"] = os.path.abspath(target)
    print(json.dumps(out, indent=2))
    return 0


def cmd_density_simulate(args) -> int:
    from .density import anchored, atom_weights, simulate_map
    from .io.mrc import write_map

    project, pipe = _pipeline(args)
    pipe.check(video=False, charts=False, pdb=False, density=True)
    xyz, elements, frame, reference = pipe.density_frame_atoms(args.frame or "ref")
    if args.placed:
        if len(project.density.transform) != 12:
            raise ValueError("--placed needs a stored placement (run 'mdatk density fit --save' first)")
        xyz = anchored(project.density.transform, project.density.anchor, reference).apply(xyz)
    dmap = simulate_map(xyz, atom_weights(elements), args.resolution, args.spacing or None)
    write_map(args.output, dmap)
    print(json.dumps({"map": os.path.abspath(args.output), "frame": frame, "atoms": len(xyz),
                      "grid_xyz": list(dmap.shape_xyz), "voxel_A": round(float(dmap.voxel[0]), 4),
                      "frame_of_reference": "map (placed)" if args.placed else "model (superposed)"},
                     indent=2))
    return 0


def cmd_pdb(args) -> int:
    from . import pdbtools
    from .projectfile import logged
    if args.pdb_command == "check":
        return pdbtools.run(args)[0]
    name = os.path.basename(args.input)
    if args.pdb_command == "annotate":
        return _pdb_annotate(args, name)
    with logged(_project_file(args), f"pdb {args.pdb_command}", f"PDB clean-up of {name}") as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["pdb"] else ""
        code, result = pdbtools.run(args)
        entry.files = list(result.get("files", []))
        entry.summary = "\n".join(result.get("notes", []))
    return code


def cmd_pdb_qc(args) -> int:
    """'mdatk pdb qc': only reads (--passed-out / --best-out copy models), so the step log is left alone."""
    from . import modelcheck
    return modelcheck.run_qc(args)[0]


def _pdb_annotate(args, name: str) -> int:
    from .annotate import annotate
    from .projectfile import logged
    with logged(_project_file(args), "pdb annotate", f"Base-pair annotation of {name}") as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["pdb"] else ""
        res = annotate(args.input, pairs=not args.no_pairs, stacking=not args.no_stacking,
                       dotbracket=not args.no_dotbracket, engine=args.engine,
                       dssr_exe=getattr(args, "dssr_exe", None) or "x3dna-dssr")
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(res.text)
        for note in res.notes:
            print(f"annotate: {note}", file=sys.stderr)
        entry.files = [args.output]
        entry.summary = "\n".join(res.notes)
        if res.engine in ("barnaba", "dssr"):
            from .citations import keys_for
            entry.citations = keys_for([res.engine])
    return 0


def _outputs(args, *names: str) -> List[str]:
    return [p for p in (getattr(args, n, None) for n in names) if p]


def cmd_ensemble(args) -> int:
    """'mdatk ensemble census | compare | regions' (mdatk.ensemble, mdatk.mapscore)."""
    from . import ensemble
    from .projectfile import logged
    title = {"census": "Register census", "compare": f"Ensemble comparison ({getattr(args, 'mode', '')})",
             "regions": "Regions ranked by per-residue scores"}[args.ensemble_command]
    with logged(_project_file(args), f"ensemble {args.ensemble_command}", title) as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["ensemble"] else ""
        code = ensemble.run(args)
        entry.files = _outputs(args, "output", "report")
    return code


def cmd_map(args) -> int:
    """'mdatk map score' (mdatk.mapscore)."""
    from . import mapscore
    from .projectfile import logged
    with logged(_project_file(args), "map score", f"Map-model scores ({args.method})") as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["map"] else ""
        code = mapscore.run_score(args)
        entry.files = _outputs(args, "output", "regions_out", "report")
        if code:  # the tables of the others are written; the report lists the models not scored
            entry.status, entry.summary = "failed", "some models were not scored (see the report)"
        if args.method == "qscore":
            from .citations import keys_for
            entry.citations = keys_for(["chimerax", "qscore"])
    return code


def cmd_pdb_hybrids(args) -> int:
    """'mdatk pdb hybrids' (mdatk.mapscore)."""
    from . import mapscore
    from .projectfile import logged
    with logged(_project_file(args), "pdb hybrids", f"Body hybrids of {os.path.basename(args.old)} and "
                f"{os.path.basename(args.new)}") as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["pdb"] else ""
        code = mapscore.run_hybrids(args)
        entry.files = _outputs(args, "scores_out", "effects_out", "report")
    return code


def cmd_pdb_interface(args) -> int:
    from . import interfaces
    from .projectfile import logged
    names = ", ".join(os.path.basename(m) for m in args.models[:3]) + (" ..." if len(args.models) > 3 else "")
    with logged(_project_file(args), "pdb interface", f"Interface records of {names}") as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["pdb"] else ""
        code, result = interfaces.run(args)
        entry.files = list(result.get("files", []))
        entry.summary = "\n".join(result.get("notes", []))
        if args.engine in ("barnaba", "dssr"):
            from .citations import keys_for
            entry.citations = keys_for([args.engine])
    return code


def cmd_decoys(args) -> int:
    from . import interfaces
    from .projectfile import logged
    titles = {"census": "Interface census", "clusters": "Interface clusters", "support": "Segment support"}
    title = titles.get(args.decoys_command, args.decoys_command)
    with logged(_project_file(args), f"decoys {args.decoys_command}", title) as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["decoys"] else ""
        code, result = interfaces.run(args)
        entry.files = list(result.get("files", []))
        entry.summary = "\n".join(result.get("notes", []))
        if args.decoys_command == "census" and args.pairs in ("barnaba", "dssr"):
            from .citations import keys_for
            entry.citations = keys_for([args.pairs])
    return code


def cmd_build(args) -> int:
    from . import builder
    from .projectfile import logged
    na = "rna"
    if args.build_command == "helix":
        na = builder.check_na(getattr(args, "na", "rna"))  # the step-log title says D-DNA / L-DNA
        kind = builder.helix_kind(getattr(args, "pairing", "wc"), getattr(args, "triplex", None), na)
        title = f"Ideal {kind} {args.chirality}-{na.upper()} helix {args.seq.strip()}"
    elif args.build_command == "chimera":
        from . import chimera
        try:  # a design that does not parse fails inside the logged step, with its own message
            title = f"Chimera {chimera.describe(args.design, getattr(args, 'na', 'rna'))}"
        except builder.BuildError:
            title = f"Chimera {args.design.strip()}"
    elif args.build_command == "mutate":
        title = f"Mutation of {os.path.basename(args.input)}: {args.mutations.strip()}"
    else:
        title = f"Rigid-body fit of {os.path.basename(args.input)} into {os.path.basename(args.map)}"
    with logged(_project_file(args), f"build {args.build_command}", title) as entry:
        entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["build"] else ""
        res = builder.run(args)
        for note in res.notes:
            print(f"build: {note}", file=sys.stderr)
        entry.files = [args.output]
        entry.summary = "\n".join(res.notes)
        if args.build_command == "helix":
            from .citations import keys_for
            other = getattr(args, "pairing", "wc") != "wc" or getattr(args, "triplex", "none") not in (None, "", "none")
            fiber = "arnott-bdna" if na == "dna" else "arnott-arna"
            entry.citations = keys_for([fiber] + (["pdb-4jrd"] if other else []))  # the template sources
        elif args.build_command == "chimera":
            from .citations import keys_for
            fibers = {"rna": "arnott-arna", "dna": "arnott-bdna"}  # the helices of the segments, in order of use
            used = chimera.nucleic_acids(args.design, getattr(args, "na", "rna"))
            entry.citations = keys_for([fibers[n] for n in used])
    if res.cc is not None:
        print(json.dumps({"output": os.path.abspath(args.output), "cc": round(res.cc, 4)}, indent=2))
    return 0


def cmd_dict(args) -> int:
    from . import dictionaries
    from .projectfile import logged
    try:
        if args.dict_command != "finish":  # list / smiles / pubchem write no files
            return dictionaries.run(args)[0]
        with logged(_project_file(args), "dict finish", f"Restraint dictionary {args.code}") as entry:
            entry.command = " ".join(sys.argv[1:]) if sys.argv[1:2] == ["dict"] else ""
            code, result = dictionaries.run(args)
            entry.files = list(result.get("files", []))
            entry.summary = "\n".join(result.get("notes", []))
            from .citations import keys_for
            entry.citations = keys_for(["ccd"])
        return code
    except dictionaries.DictionaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_rosetta(args) -> int:
    from . import rosetta
    return rosetta.run(args)


def cmd_frame(args) -> int:
    from . import frames
    return frames.run(args)


def cmd_workflow_tools(args) -> int:
    from .workflows import TOOLS, tools_by_category
    from .workflows.schema import kind_label
    if not args.tool:
        for category, tools in tools_by_category():
            print(category)
            for t in tools:
                print(f"  {t.name:18s} {t.label} - {t.description}")
        return 0
    tool = TOOLS.get(args.tool)
    if tool is None:
        raise KeyError(f"unknown step type {args.tool!r}")
    print(f"{tool.label} ({tool.name})\n{tool.description}\n")
    defaults = tool.normalise({})
    for port in tool.input_ports(defaults):
        print(f"input  {port.name:14s} {', '.join(kind_label(k) for k in port.kinds)}"
              f"{' (optional)' if port.optional else ''}")
    for port in tool.output_ports(defaults):
        print(f"output {port.name:14s} {kind_label(port.kind)}")
    for sec in tool.sections:
        print(f"\n[{sec.title}]" + (f"  records: {sec.key}.<n>.<key>, templates: "
                                     f"{', '.join(t[0] for t in sec.templates)}" if sec.repeat else ""))
        for o in sec.options:
            default = o.default if not sec.repeat else ""
            choices = f" {{{', '.join(c[0] for c in o.choices)}}}" if o.choices else ""
            print(f"  {o.key:28s} {o.kind:9s}{choices} default={default!r}\n      {o.tooltip}")
    return 0


def _workflow_project(args) -> Project:
    project = Project.load(args.params) if getattr(args, "params", None) else Project()
    if getattr(args, "out_dir", None):
        project.workflow.out_dir = os.path.abspath(args.out_dir)
    if getattr(args, "project_name", None):
        project.workflow.project_name = args.project_name
    if getattr(args, "no_stamp", False):
        project.workflow.use_stamp = False
    if getattr(args, "stamp", None):
        project.workflow.stamp = args.stamp
    return project


def cmd_workflow_check(args) -> int:
    from .workflows.plan import Plan
    project = _workflow_project(args)
    plan = Plan(project.workflow, project.workflow_dir, project.groups)
    n_err = _print_issues(plan)
    print(json.dumps({"steps": [plan.steps[n].label for n in plan.order], "errors": n_err,
                      "folder": project.workflow_dir}, indent=2, ensure_ascii=False))
    return 1 if n_err else 0


def _print_superseded(items, quiet: bool = False, dry_run: bool = False) -> int:
    """Say on standard error which completed runs moved aside (would move, with ``dry_run``: always said)
    before their steps were written; returns how many could not be moved (always said: those steps were not
    written)."""
    failed = 0
    for item in items:
        failed += bool(item.error)
        if quiet and not item.error and not dry_run:
            continue
        done = "deleted" if item.deleted else "moved"
        head = "ERROR" if item.error else ("would move" if dry_run else done)
        lines = item.lines(limit=10 if dry_run else 20)
        print(f"{head:8s} {lines[0]}", file=sys.stderr)
        for line in lines[1:]:
            print(f"         {line}", file=sys.stderr)
    return failed


def cmd_workflow_write(args) -> int:
    from .workflows import supersede
    from .workflows.plan import Plan
    project = _workflow_project(args)
    if args.dry_run and args.run:
        raise ValueError("--dry-run writes nothing, so there is nothing to --run")
    # a bad --parallel / --cores / --gpus fails before writing
    parallel, budget = (_parallel(args, project), _budget(args, project)) if args.run else (1, None)
    plan = Plan(project.workflow, project.workflow_dir, project.groups)
    ids = None
    if args.step is not None:
        ids = [n.id for n in project.workflow.nodes if n.step == args.step]
        if not ids:
            raise ValueError(f"no step number {args.step} in the plan")
    n_err = _print_issues(plan, quiet=args.quiet)
    if n_err and not args.force:
        print("error: the plan has errors (fix them or use --force to write the other steps)", file=sys.stderr)
        return 1
    if args.dry_run:  # what would be written, and which completed runs would move aside first
        from .workflows.manifest import IDENTITY, MANIFEST
        moving = supersede.pending(plan, ids)
        _print_superseded(moving.values(), args.quiet, dry_run=True)
        targets = [n for n in (ids if ids is not None else plan.order) if not plan.steps[n].adopted]
        generated = plan.generate(targets)
        files = [os.path.join(plan.root, plan.steps[n].ctx.from_own(f.name))
                 for n, fs in generated.items() for f in fs]
        records = [os.path.join(plan.root, plan.steps[n].ctx.folder, IDENTITY)
                   for n, fs in generated.items() if fs and plan.steps[n].ctx.folder]
        out = {"folder": plan.root, "dry_run": True, "write": files,
               "records": [os.path.join(plan.root, MANIFEST)] + records,
               "superseded": [i.plain() for i in moving.values()]}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    stamp_was = project.workflow.stamp
    from .projectfile import logged
    with logged(_project_file(args), "workflow write", "Generated the plan's files") as entry:
        report = plan.write(ids, project_path=_project_file(args), discard=args.delete_old_runs)
        entry.folder = report.root
        entry.files = list(report.written)
        entry.summary = f"{len(report.written)} file(s) written, {len(report.skipped)} step(s) skipped"
        if report.superseded:
            entry.summary += f", {len(report.superseded)} completed run(s) " + (
                "deleted" if args.delete_old_runs else "moved aside")
    failed = _print_superseded(report.superseded.values(), args.quiet)
    if args.params and not stamp_was and project.workflow.stamp and not args.no_save:
        stamp = project.workflow.stamp  # remember it so later writes keep the same names
        _edit_project_workflow(args.params, lambda wf: setattr(wf, "stamp", stamp))
    out = {"folder": report.root, "written": report.written, "skipped": report.skipped,
           "master": report.master, "summary": report.summary, "manifest": report.manifest}
    if report.kept:  # steps that use the outputs of an earlier run: not written
        out["kept"] = report.kept
    if report.superseded:  # completed runs moved aside before their steps were written
        out["superseded"] = [i.plain() for i in report.superseded.values()]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.run:
        ok_ids = [n for n in (ids if ids is not None else plan.order) if n not in report.skipped]
        code = _run_plan_scripts(plan, ok_ids, _project_file(args), parallel, args.include_separate, budget)
        return code or (1 if failed else 0)
    return 1 if failed else 0


def cmd_workflow_run(args) -> int:
    from .workflows.plan import Plan
    project = _workflow_project(args)
    plan = Plan(project.workflow, project.workflow_dir, project.groups)
    ids = None
    if args.step is not None:
        ids = [n.id for n in project.workflow.nodes if n.step == args.step]
        if not ids:
            raise ValueError(f"no step number {args.step} in the plan")
    return _run_plan_scripts(plan, ids, _project_file(args), _parallel(args, project), args.include_separate,
                             _budget(args, project))


def cmd_workflow_status(args) -> int:
    """Look for the outputs of earlier runs of the plan's steps; --adopt / --forget store the choice."""
    from .workflows import runstate
    from .workflows.plan import Plan
    project = _workflow_project(args)
    ids = None
    if args.step is not None:
        ids = [n.id for n in project.workflow.nodes if n.step == args.step]
        if not ids:
            raise ValueError(f"no step number {args.step} in the plan")
    if (args.adopt or args.forget) and not args.params:
        raise ValueError("give the project file (--project) that stores which runs the steps use")
    if args.adopt and args.forget:
        raise ValueError("--adopt and --forget exclude each other")
    plan = Plan(project.workflow, project.workflow_dir, project.groups)
    changed: List[str] = []
    if args.forget:
        changed = runstate.forget(project.workflow, ids)
    runs = runstate.check_runs(plan, ids)
    if args.adopt:
        changed = runstate.adopt(plan, runs)
    if changed:
        use = {n.id: n.adopted for n in project.workflow.nodes if n.id in changed}

        def store(wf):
            for node in wf.nodes:
                if node.id in use:
                    node.adopted = use[node.id]

        _edit_project_workflow(args.params, store)
        plan = Plan(project.workflow, project.workflow_dir, project.groups)
        runs = runstate.check_runs(plan, ids)
    if args.json:
        steps = [{"step": plan.steps[n].node.step, "title": plan.steps[n].label, "state": r.state,
                  "stamp": r.stamp, "current": r.current, "adopted": r.adopted, "finished": r.date,
                  "found": r.found, "missing": r.missing} for n, r in runs.items()]
        print(json.dumps({"folder": plan.root, "stamp": plan.stamp, "changed": changed, "steps": steps},
                         indent=2, ensure_ascii=False))
        return 0
    print(f"plan folder {plan.root}, time stamp {plan.stamp or '(none)'}")
    for nid, r in runs.items():
        print(f"{plan.steps[nid].label}: {r.summary()}")
        if r.state == "partial" and not args.quiet:
            for path in r.missing[:5]:
                print(f"    missing or unfinished: {path}")
            if len(r.missing) > 5:
                print(f"    ... and {len(r.missing) - 5} more")
    for nid in changed:
        stamp = next(n.adopted for n in project.workflow.nodes if n.id == nid)
        print(f"{plan.steps[nid].label}: " + (f"now uses the outputs of its run of {stamp}" if stamp
                                               else "runs again (no longer uses an earlier run)"))
    offer = [n for n, r in runs.items() if r.state == "done" and r.stamp and not r.adopted]
    if offer and not args.adopt:
        print(f"{len(offer)} step(s) have a complete run they do not use yet: --adopt lets them use it "
              "(they are then not written or run again)")
    return 0


def cmd_workflow_manifest(args) -> int:
    """The plan's steps against the step folders of an existing tree (by default the plan folder): which
    folder holds which step's files, by the identity file each write leaves in a step folder (node id,
    then folder tag, then title) or by the folder tag in a folder's name.  1 when a step's folder holds
    another step's files (writing that step would put its files beside them)."""
    from .workflows.manifest import manifest_text, match
    from .workflows.plan import Plan
    project = _workflow_project(args)
    plan = Plan(project.workflow, project.workflow_dir, project.groups)
    if args.table:  # what a write puts into <plan folder>/MANIFEST
        sys.stdout.write(manifest_text(plan))
        return 0
    tree = os.path.abspath(args.tree) if args.tree else plan.root
    if not os.path.isdir(tree):
        raise ValueError(f"{tree} is not a folder" + ("" if args.tree else
                                                      " (nothing written yet; --tree names another one)"))
    result = match(plan, tree)
    if args.json:
        print(json.dumps(result.plain(), indent=2, ensure_ascii=False))
    else:
        print(result.text())
    return 1 if result.conflicts else 0


def _parallel(args, project) -> int:
    """--parallel, else the project's setting: the most scripts at once (0: no limit)."""
    if getattr(args, "parallel", None) is not None:
        if args.parallel < 0:
            raise ValueError("--parallel must be 0 (no limit) or more")
        return args.parallel
    return max(0, int(project.workflow.parallel))


def _budget(args, project):
    """The CPU cores and GPUs the scripts may use together: --cores / --gpus, else the project's settings."""
    from .workflows.budget import Budget
    cores, gpus = getattr(args, "cores", None), getattr(args, "gpus", None)
    if cores is not None and cores < 0:
        raise ValueError("--cores must be 0 (half the cores) or more")
    if gpus is not None and gpus < -1:
        raise ValueError("--gpus must be -1 (the GPUs found), 0 or more")
    return Budget.of(project.workflow, cores, gpus)


def _run_plan_scripts(plan, ids, project_path: Optional[str], parallel: int = 1,
                      include_separate: bool = False, budget=None) -> int:
    """Run the main scripts of ``ids`` (all steps when None), with ``include_separate`` also the scripts
    marked 'run separately', at most ``parallel`` at once (0: no limit) within ``budget`` (CPU cores and
    GPUs; None: no limit); 0 when every script succeeded, WAITING_STATUS when a step waits for a
    separately run script (its dependents were not started), 1 when a script failed."""
    from .workflows.base import WAITING_STATUS
    from .workflows.runner import ScriptFailed, adopted_steps, plan_jobs, run_jobs
    kept = adopted_steps(plan, ids)
    for nid in kept:
        step = plan.steps[nid]
        print(f"{step.label} uses the outputs of its run of {step.adopted}: "
              f"not run again (to run it: mdatk workflow status --forget --step {step.node.step})")
    jobs = plan_jobs(plan, ids, include_separate)
    if not jobs:
        if kept:
            return 0
        print("no scripts to run", file=sys.stderr)
        return 1
    import threading
    out_lock = threading.Lock()

    def show(line: str) -> None:  # one write per line: lines of parallel scripts never merge
        with out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    if budget is not None and len(jobs) > 1:
        cap = f", at most {parallel} at once" if parallel else ""
        show(f"running {len(jobs)} script(s) with {budget.text()}{cap}")
    try:
        results = run_jobs(jobs, project_path, on_line=show, parallel=parallel, budget=budget)
    except ScriptFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    waiting = [r for r in results if r.status == "waiting"]
    for r in waiting:
        print(f"waiting: {r.job.title} is complete once its separately run script "
              "has finished; run it, then this step again (or use --include-separate): "
              + " ".join(r.job.separate))
    if waiting:
        skipped = sum(1 for r in results if r.status == "skipped")
        if skipped:
            print(f"{skipped} script(s) of later steps not started")
        return WAITING_STATUS
    return 0


def cmd_workflow_generate(args) -> int:
    from .workflows import get_tool
    from .workflows.plan import scratch_node, single_step_plan
    project = _workflow_project(args)
    tool = get_tool(args.tool)
    node = scratch_node(project.workflow, tool.name)
    settings = tool.normalise(node.settings)
    for assignment in args.set or []:
        apply_tool_override(tool, settings, assignment)
    node.settings = settings
    node.step = args.step
    for item in args.input or []:
        if "=" not in item:
            raise ValueError(f"--input expects PORT=PATH, got {item!r}")
        port, path = item.split("=", 1)
        node.files[port] = ";".join(os.path.abspath(p) for p in path.split(";") if p)
    plan = single_step_plan(project.workflow, node, project.workflow_dir, project.groups)
    n_err = _print_issues(plan, quiet=args.quiet)
    if n_err:
        return 1
    from .workflows import supersede
    if args.dry_run:
        _print_superseded(supersede.pending(plan, [node.id]).values(), args.quiet, dry_run=True)
        for f in plan.generate([node.id]).get(node.id, []):
            print(f"===== {f.name}\n{f.content}")
        return 0
    report = plan.write([node.id], master=False, project_path=_project_file(args))
    failed = _print_superseded(report.superseded.values(), args.quiet)
    out = {"folder": os.path.join(report.root, plan.steps[node.id].ctx.folder),
           "written": report.written, "stamp": plan.stamp}
    if report.superseded:
        out["superseded"] = [i.plain() for i in report.superseded.values()]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if failed:
        return 1
    if args.run:
        return _run_plan_scripts(plan, [node.id], _project_file(args))
    return 0


def cmd_workflow_template(args) -> int:
    from .workflows.plan import TEMPLATES, apply_template
    if not args.name:
        for name, (label, _s, _e) in TEMPLATES.items():
            print(f"{name:16s} {label}")
        return 0
    if not args.params:
        raise ValueError("give the project to add the template to (--project)")
    added = []

    def add(wf):
        y = max((n.y for n in wf.nodes), default=-180.0) + 220.0
        added.extend(apply_template(wf, args.name, 40.0, y))

    _edit_project_workflow(args.params, add)
    print(json.dumps({"added": [f"{n.step:02d} {n.tool}" for n in added], "project": args.params}, indent=2))
    return 0


def _edit_project_workflow(path: str, edit) -> None:
    """Change the workflow part of a project file (paths stay as written; the step log is kept)."""
    from .projectfile import edit_workflow

    edit_workflow(path, edit)


def cmd_workflow_restraints(args) -> int:
    from .workflows import rna
    from .workflows.amber import restraint_pairs, restraint_window
    from .workflows.scripts import helix_dist
    model = rna.build_model(args.sequence, args.dotbracket)
    s = {"wobble": args.wobble, "r1": args.r[0], "r2": args.r[1], "r3": args.r[2], "r4": args.r[3],
         "rk2": args.rk[0], "rk3": args.rk[1], "res_offset": args.offset}
    pairs, issues = restraint_pairs(model, s)
    for issue in issues:
        print(f"{issue.level.upper()}: {issue.message}", file=sys.stderr)
    names, labels, pointers = helix_dist.read_parm7_names(args.top)
    text = helix_dist.restraint_text(names, labels, pointers, pairs, rna.HBOND_ATOMS, restraint_window(s),
                                     args.offset)
    with open(args.output, "w") as fh:
        fh.write(text)
    print(json.dumps({"output": os.path.abspath(args.output), "pairs": len(pairs),
                      "restraints": text.count("&rst")}, indent=2))
    return 0


def _common(p: argparse.ArgumentParser, inputs: bool = True) -> None:
    p.add_argument("--project", "-p", "--params", dest="params",
                   help=f"project file (.mdatk.toml) or a run's {PROJECT_ABBREV}_params_log.toml")
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

    p = sub.add_parser("gui", help="start the GTK 3 interface (the analysis window, or RNAdo)")
    p.add_argument("params", nargs="?", help="project file to open")
    p.add_argument("--rnado", "--planner", action="store_true",
                   help="start RNAdo, the workflow planner, instead of the analysis window (same as: rnado)")
    p.add_argument("--open", choices=RNADO_TOOLS, help="start RNAdo with this tool open as well")
    p.set_defaults(func=cmd_gui)
    sub.add_parser("rnado", help="RNAdo, the workflow planner: 'mdatk rnado [PROJECT]' opens it, 'mdatk "
                                 "rnado ACTION ...' is 'mdatk workflow ACTION ...' (see: rnado --help)")
    p = sub.add_parser("desktop", help="MDATK and RNAdo in the applications menu: desktop entries and icons "
                                       "in ~/.local/share (Linux), starting this installation or AppImage")
    dsub = p.add_subparsers(dest="desktop_command", metavar="action")
    d = dsub.add_parser("install", help="write mdatk.desktop, rnado.desktop and their icons (files MDATK did "
                                        "not write are left alone)")
    d.add_argument("--replace", action="store_true",
                   help="move entries and icons MDATK did not write aside (FILE.bak) and write MDATK's")
    d.set_defaults(func=cmd_desktop)
    dsub.add_parser("uninstall", help="remove the entries and icons MDATK wrote (marked X-MDATK-Installed), "
                                      "nothing else").set_defaults(func=cmd_desktop)
    dsub.add_parser("status", help="what is installed and what the entries start").set_defaults(
        func=cmd_desktop)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    p = sub.add_parser("demo", help="write a synthetic RNA aptamer/target system and project file")
    p.add_argument("out_dir")
    p.add_argument("--n-frames", type=int, default=120)
    p.add_argument("--dt", type=float, default=10.0, help="ps between frames")
    p.add_argument("--no-solvent", action="store_true", help="omit water / Na+ (no cleaning needed)")
    p.add_argument("--run", action="store_true", help="render the demo immediately")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("init", aliases=["init-params"], help="write a project file with default settings")
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

    p = sub.add_parser("density", help="density maps (MRC / CCP4): header info, rigid-body fit of the "
                                       "model, simulated maps")
    dsub = p.add_subparsers(dest="dcommand", metavar="action")
    d = dsub.add_parser("info", help="print a map's grid, voxel size, origin and statistics")
    d.add_argument("map")
    d.set_defaults(func=cmd_density_info)
    d = dsub.add_parser("fit", help="fit the structure into the project's (or --map) density map")
    _common(d)
    d.add_argument("--map", help="MRC / CCP4 map (default: density.map_path of the project)")
    d.add_argument("--resolution", type=float, help="map resolution in Å (simulated-map width)")
    d.add_argument("--level", type=float, help="contour level in σ (shadow and centring)")
    d.add_argument("--frame", help="'ref' (first analysed frame) or a trajectory frame number")
    d.add_argument("--search", choices=("global", "manual"), help="global search, or refine the stored "
                   "placement (default: density.search)")
    d.add_argument("--step", type=float, help="angular step of the global search (degrees)")
    d.add_argument("--save", nargs="?", const=True, default=False, metavar="PRESET",
                   help="store the placement in the project (--project, or the given path)")
    d.set_defaults(func=cmd_density_fit)
    d = dsub.add_parser("simulate", help="write a map simulated from one frame (molmap-style Gaussians)")
    _common(d)
    d.add_argument("--resolution", type=float, required=True, help="resolution in Å")
    d.add_argument("--spacing", type=float, help="grid spacing in Å (default resolution / 3)")
    d.add_argument("--frame", help="'ref' or a trajectory frame number")
    d.add_argument("--placed", action="store_true", help="apply the stored placement (map frame)")
    d.add_argument("--output", "-O", required=True, help="output .mrc")
    d.set_defaults(func=cmd_density_simulate)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .pdbtools import add_commands as pdb_commands
    p = sub.add_parser("pdb", help="PDB hygiene: ChimeraX repair, L-RNA detection and naming (Amber / CCD / "
                                   "Rosetta), chain ids, renumbering, SEQRES / FASTA")
    psub = p.add_subparsers(dest="pdb_command", metavar="action")
    for c in pdb_commands(psub):
        c.add_argument("--project", "-p", "--params", dest="params",
                       help="project file whose step log records the clean-up")
        c.set_defaults(func=cmd_pdb)
    c = psub.add_parser("annotate", help="add REMARK 950 base-pair / stacking / dot-bracket records "
                                         "(barnaba when installed, else builtin canonical pairs)")
    c.add_argument("input", help="PDB file")
    c.add_argument("--output", "-o", required=True, help="output PDB")
    c.add_argument("--no-pairs", action="store_true", help="no PAIR records")
    c.add_argument("--no-stacking", action="store_true", help="no STACK records")
    c.add_argument("--no-dotbracket", action="store_true", help="no sequence / dot-bracket records")
    c.add_argument("--engine", choices=("auto", "builtin", "barnaba", "dssr"), default="auto",
                   help="auto: barnaba if importable, else builtin; dssr: x3dna-dssr (licensed separately)")
    c.add_argument("--dssr-exe", default="x3dna-dssr", help="x3dna-dssr executable (--engine dssr)")
    c.add_argument("--project", "-p", "--params", dest="params",
                   help="project file whose step log records the annotation")
    c.set_defaults(func=cmd_pdb)
    from .modelcheck import add_qc_arguments
    c = psub.add_parser("qc", help="plausibility of models: clashes, O3'-P links, loop compaction, sugar "
                                   "hands (text, --json or --tsv; gates, filtering and ranking of models)")
    add_qc_arguments(c)
    c.set_defaults(func=cmd_pdb_qc)
    from .mapscore import add_hybrid_command
    add_hybrid_command(psub).set_defaults(func=cmd_pdb_hybrids)
    from .interfaces import add_interface_command
    c = add_interface_command(psub)
    c.add_argument("--project", "-p", "--params", dest="params",
                   help="project file: its residue groups (group:NAME) and the step log")
    c.set_defaults(func=cmd_pdb_interface)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .interfaces import add_commands as decoys_commands
    p = sub.add_parser("decoys", help="ensembles of models (e.g. FARFAR2 decoys): interface census between "
                                      "two sides, clusters of the contact fingerprints, support of rebuilt "
                                      "segments")
    dsub = p.add_subparsers(dest="decoys_command", metavar="action")
    for c in decoys_commands(dsub):
        c.add_argument("--project", "-p", "--params", dest="params",
                       help="project file: its residue groups (group:NAME) and the step log")
        c.set_defaults(func=cmd_decoys)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .builder import add_commands as build_commands
    p = sub.add_parser("build", help="starting structures: ideal A-form RNA / B-form DNA helices (D / L), D/L "
                                     "chimeras, base mutations, rigid-body fit into a density map")
    bsub = p.add_subparsers(dest="build_command", metavar="action")
    for c in build_commands(bsub):
        c.add_argument("--project", "-p", "--params", dest="params",
                       help="project file whose step log records the step")
        c.set_defaults(func=cmd_build)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .dictionaries import add_commands as dict_commands
    p = sub.add_parser("dict", help="restraint dictionaries for L-RNA residues: SMILES for acedrg / eLBOW, "
                                    "renaming to the CCD atom names, PubChem lookup (network, on request)")
    dsub = p.add_subparsers(dest="dict_command", metavar="action")
    for c in dict_commands(dsub):
        if c.prog.endswith(" finish"):
            c.add_argument("--project", "-p", "--params", dest="params",
                           help="project file whose step log records the step")
        c.set_defaults(func=cmd_dict)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .rosetta import add_commands as rosetta_commands
    p = sub.add_parser("rosetta", help="Rosetta FARFAR2 bookkeeping: SCORE terms by column name and the "
                                       "decoy table, and an audit of what each rna_denovo run was given")
    rsub = p.add_subparsers(dest="rosetta_command", metavar="action")
    for c in rosetta_commands(rsub):
        c.set_defaults(func=cmd_rosetta)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .frames import add_commands as frame_commands
    p = sub.add_parser("frame", help="rigid frames of multi-body models: fit on one body, measure the "
                                     "others; move models into a re-fitted reference's frame")
    fsub = p.add_subparsers(dest="frame_command", metavar="action")
    for c in frame_commands(fsub):
        c.set_defaults(func=cmd_frame)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .ensemble import add_commands as ensemble_commands
    p = sub.add_parser("ensemble", help="decoy ensembles: register census by build condition, two-ensemble "
                                        "comparison (closure, defining pairs, map occupancy, statistics), "
                                        "regions ranked by per-residue scores")
    esub = p.add_subparsers(dest="ensemble_command", metavar="action")
    for c in ensemble_commands(esub):
        c.set_defaults(func=cmd_ensemble)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    from .mapscore import add_commands as map_commands
    p = sub.add_parser("map", help="map-model scores per residue (MDATK correlation or ChimeraX Q-score) and "
                                   "the regions ranked by them")
    msub = p.add_subparsers(dest="map_command", metavar="action")
    for c in map_commands(msub):
        c.set_defaults(func=cmd_map)
    p.set_defaults(func=lambda a, _p=p: (_p.print_help(), 2)[1])

    p = sub.add_parser("workflow", help="the plan of RNAdo, MDATK's workflow planner, and its script "
                                        "generators (FARFAR2, SimRNA, RNAMake, Amber, GROMACS, NAMD, VMD, "
                                        "Phenix, Coot, custom scripts ...)")
    wsub = p.add_subparsers(dest="wcommand", metavar="action")
    w = wsub.add_parser("tools", help="list step types, or the options of one")
    w.add_argument("tool", nargs="?")
    w.set_defaults(func=cmd_workflow_tools)

    def naming(w):
        w.add_argument("--project", "-p", "--params", dest="params",
                       help="project file holding the plan / wizard settings")
        w.add_argument("--out-dir", "-o", help="plan folder (default <output dir>/<project>_workflow)")
        w.add_argument("--project-name", help="the <project> part of file names")
        w.add_argument("--stamp", help="the <time> part of file names (default: the plan's stamp or now)")
        w.add_argument("--no-stamp", action="store_true", help="names without a time stamp")
        w.add_argument("--quiet", "-q", action="store_true", help="only print errors")

    def budget_args(w, prefix=""):
        w.add_argument("--cores", type=int, metavar="N",
                       help=prefix + "CPU cores the scripts may use together; a step starts when the cores "
                       "it uses are free (0: half the cores; default: the project's setting)")
        w.add_argument("--gpus", type=int, metavar="N",
                       help=prefix + "GPUs they may use; a step that uses one gets a free GPU in "
                       "CUDA_VISIBLE_DEVICES unless it names its own (-1: the GPUs found, 0: none; default: "
                       "the project's setting)")

    w = wsub.add_parser("check", help="resolve the project's plan and list its problems")
    naming(w)
    w.set_defaults(func=cmd_workflow_check)
    w = wsub.add_parser("write", help="write the plan's files, master script and summary")
    naming(w)
    w.add_argument("--step", type=int, help="only this step number")
    w.add_argument("--force", action="store_true", help="write the steps without errors even if others "
                                                       "have errors")
    w.add_argument("--no-save", action="store_true", help="do not store the new time stamp in the project")
    w.add_argument("--dry-run", action="store_true",
                   help="write and move nothing: list the files that would be written and the completed runs "
                        "that would move into their step's superseded/ folder first")
    w.add_argument("--delete-old-runs", action="store_true",
                   help="delete the completed runs that would move into superseded/ instead (trial runs not "
                        "worth the disk space; cannot be undone)")
    w.add_argument("--run", action="store_true", help="then run the written steps' main scripts (each when "
                                                     "the steps it reads from succeeded)")
    w.add_argument("--parallel", type=int, metavar="N", help="with --run: run up to N scripts at the same "
                                                            "time (0: no limit; default: the project's "
                                                            "setting)")
    budget_args(w, "with --run: ")
    w.add_argument("--include-separate", action="store_true",
                   help="with --run: also run the scripts marked 'run separately' (e.g. production), each "
                        "after its step's main script, which then runs again for the rest of the step")
    w.set_defaults(func=cmd_workflow_write)
    w = wsub.add_parser("run", help="run the plan's main scripts (already written), each when the steps it "
                                    "reads from succeeded; each run is logged in the project file. A step "
                                    "whose separately run script has not finished is reported as waiting "
                                    "(exit status 75) and the steps after it are not started")
    naming(w)
    w.add_argument("--step", type=int, help="only this step number")
    w.add_argument("--parallel", type=int, metavar="N", help="run up to N scripts at the same time (0: no "
                                                            "limit; default: the project's setting)")
    budget_args(w)
    w.add_argument("--include-separate", action="store_true",
                   help="also run the scripts marked 'run separately' (e.g. production runs meant for a "
                        "cluster), each after its step's main script, which then runs again for the rest of the "
                        "step, and before the steps that read from it")
    w.set_defaults(func=cmd_workflow_run)
    w = wsub.add_parser("status", help="look in each step's folder for the output files of earlier runs "
                                       "(under the plan's time stamp or an earlier one): done, partial or "
                                       "missing")
    naming(w)
    w.add_argument("--step", type=int, help="only this step number")
    w.add_argument("--adopt", action="store_true",
                   help="let steps with a complete run use its outputs: later steps read them and the step "
                        "is not written or run again (stored in the project file)")
    w.add_argument("--forget", action="store_true",
                   help="stop using earlier runs (all steps, or --step): they are written and run again")
    w.add_argument("--json", action="store_true", help="print the result as JSON")
    w.set_defaults(func=cmd_workflow_status)
    w = wsub.add_parser("manifest", help="match the plan's steps to the step folders of an output tree (e.g. "
                                         "one written before the steps were renumbered): by the MDATK_step.toml "
                                         "every write leaves in a step folder (node id, folder tag, title) or "
                                         "the folder tag in a folder's name. Exit status 1 when a step's folder "
                                         "holds another step's files")
    naming(w)
    w.add_argument("--tree", metavar="DIR", help="the output tree (default: the plan folder)")
    w.add_argument("--table", action="store_true",
                   help="print the plan's MANIFEST instead (one line per step: step, folder, node id, type, "
                        "title, folder tag, input files), as 'workflow write' writes it into the plan folder")
    w.add_argument("--json", action="store_true", help="print the result as JSON")
    w.set_defaults(func=cmd_workflow_manifest)
    w = wsub.add_parser("generate", help="stand-alone wizard: write one step's files")
    w.add_argument("tool", help="step type (see: mdatk workflow tools)")
    naming(w)
    w.add_argument("--step", type=int, default=1, help="step number in the file names")
    w.add_argument("--set", "-s", action="append", metavar="KEY=VALUE",
                   help="option value, e.g. engine=pmemd.MPI or stages.5.nstlim=1000 (repeatable)")
    w.add_argument("--input", "-i", action="append", metavar="PORT=PATH",
                   help="input file, e.g. structure=model.pdb (repeatable; ';' separates several)")
    w.add_argument("--dry-run", action="store_true", help="print the files instead of writing them")
    w.add_argument("--run", action="store_true", help="then run the step's main script")
    w.set_defaults(func=cmd_workflow_generate)
    w = wsub.add_parser("template", help="list templates, or add one to a project's plan")
    w.add_argument("name", nargs="?")
    w.add_argument("--project", "-p", "--params", dest="params",
                   help="project to add the template to (saved in place)")
    w.set_defaults(func=cmd_workflow_template)
    w = wsub.add_parser("restraints", help="write helix-preserving DISANG restraints for an existing "
                                           "topology")
    w.add_argument("--top", required=True, help="Amber topology (parm7)")
    w.add_argument("--sequence", required=True, help="RNA sequence (chains separated by spaces)")
    w.add_argument("--dotbracket", required=True, help="secondary structure")
    w.add_argument("--offset", type=int, default=0, help="topology residue = sequence position + offset")
    w.add_argument("--wobble", action="store_true", help="also restrain G·U pairs")
    w.add_argument("--r", type=float, nargs=4, default=[2.3, 2.7, 3.2, 3.7], metavar=("R1", "R2", "R3", "R4"))
    w.add_argument("--rk", type=float, nargs=2, default=[10.0, 10.0], metavar=("RK2", "RK3"))
    w.add_argument("--output", "-o", default="helix.dist")
    w.set_defaults(func=cmd_workflow_restraints)
    p.set_defaults(func=lambda a: (p.print_help(), 2)[1])
    return parser


# ---------------------------------------------------------------------------
# rnado: the workflow planner's own command
# ---------------------------------------------------------------------------

RNADO_TOOLS = ("wizard", "pdb", "builder")  # mdatk.gui.launch.TOOLS (that module stays free of the CLI)
WORKFLOW_ACTIONS = ("tools", "check", "write", "run", "status", "manifest", "generate", "template",
                    "restraints")


def rnado_gui_parser() -> argparse.ArgumentParser:
    """Arguments of the RNAdo program (``rnado`` without an action, ``rnado-gui``)."""
    parser = argparse.ArgumentParser(
        prog="rnado", description="RNAdo, MDATK's workflow planner: lay out the steps from sequence to MD "
        "ensemble, write their scripts and run them. 'rnado ACTION ...' is 'mdatk workflow ACTION ...' "
        f"(actions: {', '.join(WORKFLOW_ACTIONS)}).")
    parser.add_argument("--version", action="version", version=f"RNAdo ({PROJECT_ABBREV}) {__version__}")
    parser.add_argument("project", nargs="?", help="project file (.mdatk.toml) to open")
    parser.add_argument("--open", choices=RNADO_TOOLS,
                        help="also open the script wizard, the PDB clean-up or the structure builder")
    parser.add_argument("--install-launchers", action="store_true",
                        help="add MDATK and RNAdo to the applications menu and quit (the same as: mdatk "
                        "desktop install)")
    return parser


def rnado_main(argv: Optional[List[str]] = None) -> int:
    """``rnado``: without arguments or with a project file the RNAdo window; ``rnado ACTION ...`` (an
    action of ``mdatk workflow``: write, run, status ...) is ``mdatk workflow ACTION ...``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in WORKFLOW_ACTIONS:
        return main(["workflow"] + argv)
    args = rnado_gui_parser().parse_args(argv)  # --help, --version and wrong arguments end here, without GTK
    if args.install_launchers:
        return main(["desktop", "install"])
    return _rnado_window(argv)


def _rnado_window(argv: List[str]) -> int:
    """The RNAdo window in this process, also in the stand-alone ``mdatk`` program (the AppImage runs
    ``mdatk rnado``: that process has the AppImage's library path set up already)."""
    try:  # before GTK starts: the windows' WM_CLASS is RNAdo's, not the analysis window's ("mdatk")
        from gi.repository import GLib

        GLib.set_prgname("rnado")
    except ImportError:  # mdatk.gui.rnado says what is missing
        pass
    from .gui.rnado import main as rnado_gui

    return rnado_gui(argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["rnado"]:  # everything after it belongs to rnado (the AppImage runs 'mdatk rnado ...')
        return rnado_main(argv[1:])
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
