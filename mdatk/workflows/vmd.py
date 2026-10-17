"""VMD generator: a Tcl script that loads a topology / trajectory, draws the
RNA with one representation per MDATK residue group (in the group colours),
optionally aligns frames, and renders a snapshot or movie frames."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
from .base import PortValue, Tool, env_section, script_header
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

STYLES = [
    ("NewCartoon", "New Cartoon"), ("Licorice", "Licorice"), ("Ribbons", "Ribbons"), ("Tube", "Tube"),
    ("Surf", "Surf"), ("QuickSurf", "QuickSurf"), ("VDW", "VDW"), ("CPK", "CPK"), ("Lines", "Lines"),
]
STYLE_PARAMS = {
    "NewCartoon": "NewCartoon 0.300000 10.000000 4.100000 0",
    "Licorice": "Licorice 0.300000 12.000000 12.000000",
    "Ribbons": "Ribbons 0.300000 10.000000 2.000000",
    "Tube": "Tube 0.300000 10.000000",
    "Surf": "Surf 1.400000 0.000000",
    "QuickSurf": "QuickSurf 1.000000 0.500000 1.000000 1.000000",
    "VDW": "VDW 1.000000 12.000000",
    "CPK": "CPK 1.000000 0.300000 12.000000 12.000000",
    "Lines": "Lines 1.000000",
}
MATERIALS = [("Opaque", "Opaque"), ("AOChalky", "AOChalky"), ("AOShiny", "AOShiny"), ("Glossy", "Glossy"),
             ("Transparent", "Transparent"), ("Diffuse", "Diffuse")]
TRAJ_TYPES = {".nc": "netcdf", ".ncdf": "netcdf", ".mdcrd": "crdbox", ".crd": "crdbox", ".xtc": "xtc",
              ".trr": "trr", ".dcd": "dcd", ".pdb": "pdb"}
TOP_TYPES = {".parm7": "parm7", ".prmtop": "parm7", ".top": "parm7", ".gro": "gro", ".pdb": "pdb",
             ".psf": "psf", ".mol2": "mol2"}
COLOR_IDS = list(range(23, 33))  # free VMD colour ids redefined to the group colours


def residue_selection(residues: str) -> str:
    """'1-12, 20, A:30-40' -> 'resid 1 to 12 20 30 to 40' (chain prefixes dropped)."""
    parts = []
    for tok in re.split(r"[,\s]+", (residues or "").strip()):
        if not tok:
            continue
        tok = tok.split(":")[-1]
        m = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", tok)
        if not m:
            return ""
        parts.append(f"{m.group(1)} to {m.group(2)}" if m.group(2) else m.group(1))
    return f"resid {' '.join(parts)}" if parts else ""


def _hex_rgb(color: str) -> Tuple[float, float, float]:
    c = (color or "#888888").lstrip("#")
    if len(c) != 6:
        return (0.5, 0.5, 0.5)
    return tuple(int(c[k:k + 2], 16) / 255.0 for k in (0, 2, 4))  # type: ignore[return-value]


def _tcl_quote(text: str) -> str:
    return "{" + text.replace("{", "(").replace("}", ")") + "}"


@register
class VMDTool(Tool):
    name = "vmd"
    label = "VMD · visualisation"
    category = "Visualisation"
    tag = "vmd"
    description = ("A VMD Tcl script: RNA representations per residue group (MDATK colours), Mg²⁺ "
                   "spheres, frame alignment, smoothing and rendered snapshots or movie frames.")
    inputs = [Port("topology", "Topology", ("parm7", "pdb", "gro", "top"),
                   tooltip="parm7 (Amber), gro / pdb; must match the trajectory atoms."),
              Port("trajectory", "Trajectory", ("nc", "xtc"), optional=True, multiple=True)]
    outputs = [Port("image", "Rendered image / movie", ("image",))]
    sections = [
        Section("load", "Loading", [
            Option("first", "First frame", "int", 0, "First trajectory frame loaded (0-based).",
                   minimum=0, maximum=100_000_000),
            Option("last", "Last frame", "int", -1, "Last frame (-1: until the end).", minimum=-1,
                   maximum=100_000_000),
            Option("stride", "Stride", "int", 1, "Load every n-th frame.", minimum=1, maximum=100000),
        ]),
        Section("look", "Representations", [
            Option("rna_style", "RNA style", "choice", "NewCartoon",
                   "Representation of every residue group.", choices=STYLES),
            Option("bases", "Bases as licorice", "bool", True,
                   "Overlay the nucleobases (and ribose) as licorice so base positions are visible."),
            Option("group_colors", "Colour by residue group", "bool", True,
                   "One representation per MDATK residue group in its colour; off: colour by "
                   "residue name."),
            Option("selection", "Extra selection", "text", "",
                   "Also show this VMD selection (e.g. 'resname MG and within 5 of nucleic').",
                   placeholder="within 5 of resid 30"),
            Option("selection_style", "Extra selection style", "choice", "Licorice", "",
                   choices=STYLES),
            Option("ions", "Mg²⁺ as spheres", "bool", True, "Show Mg²⁺ (resname MG) as VDW spheres."),
            Option("ion_scale", "Ion sphere scale", "float", 0.6, "VDW radius scale for ions.",
                   minimum=0.1, maximum=3.0, step=0.1, digits=2, requires={"ions": True}),
            Option("material", "Material", "choice", "AOChalky", "VMD material of all representations.",
                   choices=MATERIALS),
            Option("smooth", "Smoothing window", "int", 0, "Trajectory smoothing of the representations "
                   "(frames, 0: off).", minimum=0, maximum=100),
        ], tooltip="How the RNA is drawn."),
        Section("display", "Display", [
            Option("background", "Background", "radio", "white", "", choices=[("white", "White"),
                                                                           ("black", "Black")]),
            Option("projection", "Projection", "radio", "Orthographic", "",
                   choices=[("Orthographic", "Orthographic"), ("Perspective", "Perspective")]),
            Option("depthcue", "Depth cueing", "bool", False, "Fade distant atoms."),
            Option("align", "Align frames", "bool", True, "Superpose every frame on the first using the "
                   "alignment selection (measure fit)."),
            Option("align_sel", "Alignment selection", "text", "nucleic and name P C4'",
                   "Atoms used to superpose frames.", requires={"align": True}),
            Option("width", "Window width", "int", 1280, "Display / image width in pixels.", minimum=64,
                   maximum=8192),
            Option("height", "Window height", "int", 960, "", minimum=64, maximum=8192),
        ]),
        Section("render", "Rendering", [
            Option("render", "Output", "radio", "interactive",
                   "Interactive: open VMD with the scene. Snapshot: render the last frame. Movie: render "
                   "every loaded frame and join them with ffmpeg.",
                   choices=[("interactive", "Interactive"), ("snapshot", "Snapshot"), ("movie", "Movie")]),
            Option("renderer", "Renderer", "radio", "TachyonInternal",
                   "TachyonInternal works without a display (batch); snapshot captures the OpenGL window.",
                   choices=[("TachyonInternal", "Tachyon (internal)"), ("snapshot", "OpenGL snapshot")],
                   requires={"render": ("snapshot", "movie")}),
            Option("fps", "Movie frame rate", "int", 24, "", minimum=1, maximum=120,
                   requires={"render": "movie"}),
            Option("vmd_exe", "VMD executable", "text", "vmd", ""),
            Option("ffmpeg_exe", "ffmpeg executable", "text", "ffmpeg", "", requires={"render": "movie"}),
            Option("extra_tcl", "Extra Tcl commands", "multiline", "",
                   "Run after the scene is built (before rendering).",
                   placeholder="rotate y by 90"),
        ]),
        env_section("module load vmd"),
    ]

    def output_ports(self, s):
        if s.get("render") == "interactive":
            return []
        return [Port("image", "Movie (mp4)" if s.get("render") == "movie" else "Image (tga)", ("image",))]

    def output_values(self, s, ctx):
        if s["render"] == "snapshot":
            return {"image": PortValue([ctx.own(self.tag, "tga")], "image")}
        if s["render"] == "movie":
            return {"image": PortValue([ctx.own("movie", "mp4")], "image")}
        return {}

    def describe(self, s):
        return f"{s['rna_style']}, {s['render']}"

    def check(self, s, ctx):
        out: List[Issue] = []
        top = ctx.inp("topology")
        trajs = ctx.inps("trajectory")
        if top and os.path.splitext(top)[1].lower() not in TOP_TYPES:
            out.append(Issue("warning", f"unknown topology type {os.path.splitext(top)[1]}; VMD may need "
                                        "an explicit type", ("input:topology",)))
        for t in trajs:
            if os.path.splitext(t)[1].lower() not in TRAJ_TYPES:
                out.append(Issue("warning", f"unknown trajectory type {os.path.splitext(t)[1]}",
                                 ("input:trajectory",)))
        if not trajs and s["render"] == "movie":
            out.append(Issue("error", "a movie needs a trajectory", ("render", "input:trajectory")))
        if not trajs and (s["align"] or s["smooth"]):
            out.append(Issue("info", "alignment and smoothing need a trajectory", ("align", "smooth")))
        if s["render"] in ("snapshot", "movie") and s["renderer"] == "snapshot":
            out.append(Issue("info", "the OpenGL snapshot renderer needs a visible VMD window; the script "
                                     "then starts VMD with a display", ("renderer",)))
        if s["last"] != -1 and s["last"] < s["first"]:
            out.append(Issue("error", "the last frame comes before the first", ("first", "last")))
        if s["group_colors"] and not [g for g in ctx.groups if residue_selection(_g(g, "residues"))]:
            out.append(Issue("info", "no residue groups with residue ranges; colouring by residue name",
                             ("group_colors",)))
        if s["rna_style"] in ("Surf",) and trajs:
            out.append(Issue("info", "Surf is slow on trajectories; QuickSurf is much faster",
                             ("rna_style",)))
        if s["bases"] and s["rna_style"] == "Licorice":
            out.append(Issue("info", "bases are already shown by the Licorice style", ("bases",)))
        if trajs and ctx.data("topology").get("box"):
            out.append(Issue("info", "a solvated trajectory loads slowly in VMD; a dry, imaged copy "
                                     "(Amber MD 'Dry, imaged production trajectory') is lighter",
                             ("input:trajectory",)))
        return out

    # ------------------------------------------------------------ files
    def tcl(self, s: Dict[str, Any], ctx) -> str:
        top = ctx.inp("topology")
        trajs = ctx.inps("trajectory")
        L = [f"# VMD script written by {PROJECT_ABBREV} {__version__} for project {ctx.project}, "
             f"step {ctx.step:02d}", f"# run: vmd -e {ctx.name(self.tag, 'tcl')}", ""]
        ttype = TOP_TYPES.get(os.path.splitext(top)[1].lower(), "parm7")
        L.append(f"set mol [mol new {_tcl_quote(top)} type {ttype} waitfor all]")
        for t in trajs:
            typ = TRAJ_TYPES.get(os.path.splitext(t)[1].lower(), "netcdf")
            L.append(f"mol addfile {_tcl_quote(t)} type {typ} first {s['first']} last {s['last']} "
                     f"step {s['stride']} waitfor all")
        L += ["", "mol delrep 0 $mol", f"display resize {s['width']} {s['height']}",
              f"display projection {s['projection']}", f"display depthcue {'on' if s['depthcue'] else 'off'}",
              f"color Display Background {'white' if s['background'] == 'white' else 'black'}",
              "axes location Off", ""]
        reps = []  # (selection, representation with parameters, colour)
        groups = [g for g in ctx.groups if residue_selection(_g(g, "residues"))]
        style = STYLE_PARAMS[s["rna_style"]]
        if s["group_colors"] and groups:
            for k, g in enumerate(groups[:len(COLOR_IDS)]):
                cid = COLOR_IDS[k]
                r, gg, b = _hex_rgb(_g(g, "color"))
                L.append(f"color change rgb {cid} {r:.3f} {gg:.3f} {b:.3f} ;# {_g(g, 'name')}")
                reps.append((residue_selection(_g(g, "residues")), style, f"ColorID {cid}"))
        else:
            reps.append(("nucleic", style, "ResName"))
        if s["bases"] and s["rna_style"] != "Licorice":
            for sel, _style, color in list(reps):
                reps.append((f"({sel}) and nucleic and not backbone and noh",
                             "Licorice 0.200000 12.000000 12.000000", color))
        if s["ions"]:
            reps.append(("resname MG Mg MG2", f"VDW {s['ion_scale']:.6f} 12.000000", "ColorID 19"))
        if s["selection"].strip():
            reps.append((s["selection"].strip(), STYLE_PARAMS[s["selection_style"]], "Name"))
        L.append("")
        for k, (sel, params, color) in enumerate(reps):
            L += [f"mol representation {params}", f"mol color {color}", f"mol selection {{{sel}}}",
                  f"mol material {s['material']}", "mol addrep $mol"]
            if s["smooth"] and trajs:
                L.append(f"mol smoothrep $mol {k} {s['smooth']}")
        if s["align"] and trajs:
            L += ["", "# superpose every frame on the first",
                  f"set ref [atomselect $mol {{{s['align_sel']}}} frame 0]",
                  f"set cur [atomselect $mol {{{s['align_sel']}}}]",
                  "set all [atomselect $mol all]",
                  "set n [molinfo $mol get numframes]",
                  "for {set f 0} {$f < $n} {incr f} {",
                  "  $cur frame $f",
                  "  $all frame $f",
                  "  $all move [measure fit $cur $ref]",
                  "}"]
        L += ["", "display resetview"]
        extra = [ln.rstrip() for ln in s["extra_tcl"].splitlines() if ln.strip()]
        if extra:
            L += ["", "# extra commands"] + extra
        renderer = s["renderer"]
        if s["render"] == "snapshot":
            img = ctx.name(self.tag, "tga")
            L += ["", "animate goto end", f"render {renderer} {img}", "quit"]
        elif s["render"] == "movie":
            L += ["", "file mkdir frames", "set n [molinfo $mol get numframes]",
                  "for {set f 0} {$f < $n} {incr f} {",
                  "  animate goto $f",
                  f"  render {renderer} [format \"frames/frame.%05d.tga\" $f]",
                  "}", "quit"]
        return "\n".join(L) + "\n"

    def generate(self, s, ctx):
        files = [GeneratedFile(ctx.name(self.tag, "tcl"), self.tcl(s, ctx), "input", "VMD script")]
        script = ctx.name(self.tag, "sh")
        sh = script_header("VMD", ctx, s, script)
        tcl = ctx.name(self.tag, "tcl")
        exe = s["vmd_exe"].strip() or "vmd"
        if s["render"] == "interactive":
            sh.append(f"{exe} -e {tcl}")
        else:
            batch = "-dispdev text " if s["renderer"] == "TachyonInternal" else ""
            sh.append(f"{exe} {batch}-e {tcl}")
            if s["render"] == "movie":
                movie = ctx.name("movie", "mp4")
                sh += ["", "# join the frames into a movie",
                       f"{s['ffmpeg_exe'].strip() or 'ffmpeg'} -y -framerate {s['fps']} -i frames/frame.%05d.tga "
                       f"-c:v libx264 -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {movie}"]
        sh.append("")
        files.append(GeneratedFile(script, "\n".join(sh), "run", "starts VMD"))
        return files


def _g(group: Any, key: str) -> str:
    if isinstance(group, dict):
        return str(group.get(key, ""))
    return str(getattr(group, key, ""))


def group_dicts(groups: Sequence[Any]) -> List[Dict[str, str]]:
    return [{"name": _g(g, "name"), "residues": _g(g, "residues"), "color": _g(g, "color")} for g in groups]
