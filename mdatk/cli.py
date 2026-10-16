"""Companion command line interface for MDATK.

Every GUI action is available here, driven by the same parameter preset::

    mdatk demo ./demo                      # synthetic RNA system + preset
    mdatk render --params ./demo/demo_preset.json
    mdatk render --params run.json --set analyses.contacts.cutoff=5.0 --no-pdb
    mdatk charts --params MDATK_output/MDATK_params_log.json --from-cache
    mdatk workflow generate amber-md -i topology=sys.parm7 -i coordinates=sys.rst7 --set engine=pmemd.MPI
    mdatk workflow write --params plan_preset.json
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


# ---------------------------------------------------------------------------
# workflow: script generators and the Project Planner
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
        problems.append("no density map: give --map or set density.map_path in the preset")
    if problems:
        raise ValueError("; ".join(problems))
    pipe.check()
    res = pipe.fit_density(search=args.search)
    d = project.density
    out = {"cc": round(float(res.cc), 4), "start": res.start, "info": d.fit_info,
           "transform": d.transform}
    if args.save:
        target = args.save if isinstance(args.save, str) else args.params
        if not target:
            raise ValueError("--save needs a path when no --params preset was given")
        project.save(target)
        out["saved"] = os.path.abspath(target)
    print(json.dumps(out, indent=2))
    return 0


def cmd_density_simulate(args) -> int:
    from .density import RigidTransform, atom_weights, simulate_map
    from .io.mrc import write_map

    project, pipe = _pipeline(args)
    pipe.check()
    xyz, elements, frame = pipe.density_frame_atoms(args.frame or "ref")
    if args.placed:
        if len(project.density.transform) != 12:
            raise ValueError("--placed needs a stored placement (run 'mdatk density fit --save' first)")
        xyz = RigidTransform.from_list(project.density.transform).apply(xyz)
    dmap = simulate_map(xyz, atom_weights(elements), args.resolution, args.spacing or None)
    write_map(args.output, dmap)
    print(json.dumps({"map": os.path.abspath(args.output), "frame": frame, "atoms": len(xyz),
                      "grid_xyz": list(dmap.shape_xyz), "voxel_A": round(float(dmap.voxel[0]), 4),
                      "frame_of_reference": "map (placed)" if args.placed else "model (superposed)"},
                     indent=2))
    return 0


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


def cmd_workflow_write(args) -> int:
    from .workflows.plan import Plan
    project = _workflow_project(args)
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
    stamp_was = project.workflow.stamp
    report = plan.write(ids)
    if args.params and not stamp_was and project.workflow.stamp and not args.no_save:
        stamp = project.workflow.stamp  # remember it so later writes keep the same names
        _edit_preset_workflow(args.params, lambda wf: setattr(wf, "stamp", stamp))
    print(json.dumps({"folder": report.root, "written": report.written, "skipped": report.skipped,
                      "master": report.master, "summary": report.summary}, indent=2))
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
    if args.dry_run:
        for f in plan.generate([node.id]).get(node.id, []):
            print(f"===== {f.name}\n{f.content}")
        return 0
    report = plan.write([node.id], master=False)
    print(json.dumps({"folder": os.path.join(report.root, plan.steps[node.id].ctx.folder),
                      "written": report.written, "stamp": plan.stamp}, indent=2))
    return 0


def cmd_workflow_template(args) -> int:
    from .workflows.plan import TEMPLATES, apply_template
    if not args.name:
        for name, (label, _s, _e) in TEMPLATES.items():
            print(f"{name:16s} {label}")
        return 0
    if not args.params:
        raise ValueError("give the preset to add the template to (--params)")
    added = []

    def add(wf):
        y = max((n.y for n in wf.nodes), default=-180.0) + 220.0
        added.extend(apply_template(wf, args.name, 40.0, y))

    _edit_preset_workflow(args.params, add)
    print(json.dumps({"added": [f"{n.step:02d} {n.tool}" for n in added], "preset": args.params}, indent=2))
    return 0


def _edit_preset_workflow(path: str, edit) -> None:
    """Change only the "workflow" part of a preset file (other settings stay byte-for-byte)."""
    from .config import WorkflowParams, _from_plain, _to_plain
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get("format") not in (None, "mdatk-preset"):
        raise ValueError(f"{path}: not an {PROJECT_ABBREV} preset")
    wf = _from_plain(WorkflowParams, data.get("workflow") or {}, "workflow")
    edit(wf)
    data["workflow"] = _to_plain(wf)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


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

    p = sub.add_parser("density", help="density maps (MRC / CCP4): header info, rigid-body fit of the "
                                       "model, simulated maps")
    dsub = p.add_subparsers(dest="dcommand", metavar="action")
    d = dsub.add_parser("info", help="print a map's grid, voxel size, origin and statistics")
    d.add_argument("map")
    d.set_defaults(func=cmd_density_info)
    d = dsub.add_parser("fit", help="fit the structure into the preset's (or --map) density map")
    _common(d)
    d.add_argument("--map", help="MRC / CCP4 map (default: density.map_path of the preset)")
    d.add_argument("--resolution", type=float, help="map resolution in Å (simulated-map width)")
    d.add_argument("--level", type=float, help="contour level in σ (shadow and centring)")
    d.add_argument("--frame", help="'ref' (first analysed frame) or a trajectory frame number")
    d.add_argument("--search", choices=("global", "manual"), help="global search, or refine the stored "
                   "placement (default: density.search)")
    d.add_argument("--step", type=float, help="angular step of the global search (degrees)")
    d.add_argument("--save", nargs="?", const=True, default=False, metavar="PRESET",
                   help="store the placement in the preset (--params, or the given path)")
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

    p = sub.add_parser("workflow", help="script generators (FARFAR2, SimRNA, RNAMake, Amber, GROMACS, "
                                        "VMD) and the Project Planner")
    wsub = p.add_subparsers(dest="wcommand", metavar="action")
    w = wsub.add_parser("tools", help="list step types, or the options of one")
    w.add_argument("tool", nargs="?")
    w.set_defaults(func=cmd_workflow_tools)

    def naming(w):
        w.add_argument("--params", "-p", help="preset holding the plan / wizard settings")
        w.add_argument("--out-dir", "-o", help="plan folder (default <output dir>/<project>_workflow)")
        w.add_argument("--project-name", help="the <project> part of file names")
        w.add_argument("--stamp", help="the <time> part of file names (default: the plan's stamp or now)")
        w.add_argument("--no-stamp", action="store_true", help="names without a time stamp")
        w.add_argument("--quiet", "-q", action="store_true", help="only print errors")

    w = wsub.add_parser("check", help="resolve the preset's plan and list its problems")
    naming(w)
    w.set_defaults(func=cmd_workflow_check)
    w = wsub.add_parser("write", help="write the plan's files, master script and summary")
    naming(w)
    w.add_argument("--step", type=int, help="only this step number")
    w.add_argument("--force", action="store_true", help="write the steps without errors even if others "
                                                       "have errors")
    w.add_argument("--no-save", action="store_true", help="do not store the new time stamp in the preset")
    w.set_defaults(func=cmd_workflow_write)
    w = wsub.add_parser("generate", help="stand-alone wizard: write one step's files")
    w.add_argument("tool", help="step type (see: mdatk workflow tools)")
    naming(w)
    w.add_argument("--step", type=int, default=1, help="step number in the file names")
    w.add_argument("--set", "-s", action="append", metavar="KEY=VALUE",
                   help="option value, e.g. engine=pmemd.MPI or stages.5.nstlim=1000 (repeatable)")
    w.add_argument("--input", "-i", action="append", metavar="PORT=PATH",
                   help="input file, e.g. structure=model.pdb (repeatable; ';' separates several)")
    w.add_argument("--dry-run", action="store_true", help="print the files instead of writing them")
    w.set_defaults(func=cmd_workflow_generate)
    w = wsub.add_parser("template", help="list templates, or add one to a preset's plan")
    w.add_argument("name", nargs="?")
    w.add_argument("--params", "-p", help="preset to add the template to (saved in place)")
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
