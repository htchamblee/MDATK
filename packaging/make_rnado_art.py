"""Draw the RNAdo logo, wordmark banner and application icon from a real A-form RNA model.

    python3 packaging/make_rnado_art.py [--out DIR] [--font FILE] [--sheet DIR] [--only WHAT]

Writes into ``mdatk/assets`` (or ``--out``); the results are committed, so the source tree and the GUI
need no build step:

* ``rnado-logo.png``       portrait logo (784 x 1168): the helix tornado over the "RNAdo" wordmark
* ``rnado-wordmark.png``   wide banner (1200 x 360): helix tornado and "RNAdo" side by side
* ``rnado-icon.svg``       square icon, vector, no text
* ``rnado-icon-N.png``     the icon at N = 16, 24, 32, 48, 64, 128, 256 and 512 px; up to 64 px a
                           simplified, bolder drawing (about one turn, thick strands, every second
                           pair, no beads), 16 and 24 px without the swirl

``--sheet DIR`` also writes comparison sheets (every icon size side by side on dark and light
backgrounds, and the small sizes enlarged) for checking the art; they do not belong in the repository.
``packaging/make_icon.py`` turns the PNGs into the Windows ``rnado.ico`` at build time.

The helix is not a generic ladder: it is the ideal A-form RNA duplex of MDATK's own builder
(:func:`mdatk.builder.build_helix`, Arnott fibre parameters: 11 bp per turn, twist 32.7 degrees, rise
2.81 A, right-handed), read back with :func:`mdatk.pdbtools.read_pdb`.  The two backbones are smooth
ribbons (cubic splines) through the P atoms; every base pair is a rung from the ribbon next to C4'
over C1' and the hydrogen-bonded ring nitrogens (N1 of the purine, N3 of the pyrimidine) to C1' and the
other ribbon, with bead nodes on the two C1' atoms and the two nitrogens.  So the drawing keeps the
A-form geometry: the pairs sit about 5 A off the helix axis (the hollow core) and are inclined by 17
to 19 degrees, and between the P ribbons the deep, narrow major groove (11.3 A vertical gap) alternates
with the wide, shallow minor groove (19.6 A); ``VIEW_AZIMUTH`` says which features the view shows.

The tornado comes from a funnel taper (:class:`Funnel`): the distance of every atom from the helix
axis is multiplied by a smooth function of its height, wide at the top and narrow at the bottom tip.
Height and azimuth of every atom are kept, so the twist, the rise, the handedness, the 11-bp repeat
and the azimuth and vertical width of both grooves are exactly those of the A-form model; only the
radii (and with them the groove depths) change with height.  The flame-like swirl strokes, the motion
streaks and the embers around the funnel are decoration, seeded so every run draws the same.  Colours
are those of the reference artwork only: reds, red-orange highlights, white text, black.

Needs pycairo, Pillow, NumPy, SciPy and matplotlib (glyph outlines: the text is drawn as paths, so
neither the PNGs nor the SVG need the font).  The font is the first one found of Inter Black Italic
(SIL Open Font License 1.1), Liberation Sans Bold Italic (OFL 1.1) and the DejaVu Sans Bold Oblique
that ships with matplotlib (Bitstream Vera licence); all three licences allow their use in artwork
such as a logo.  The font used is printed; the committed images were drawn with Inter Black Italic.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cairo
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # the mdatk package of this source tree
from mdatk import builder, pdbtools  # noqa: E402

RGB = Tuple[float, float, float]
Draw = Callable[[cairo.Context], None]
Prim = Tuple[float, Draw]  # (depth, drawing): painted far to near

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
SMALL_ICON = 64  # this size and below: the simplified drawing
TINY_ICON = 24  # this size and below: simpler still
LOGO_SIZE = (784, 1168)
BANNER_SIZE = (1200, 360)
FONTS = [
    "/usr/share/fonts/opentype/inter/Inter-BlackItalic.otf",
    "/usr/share/fonts/inter/Inter-BlackItalic.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-BoldItalic.ttf",
    "/usr/share/fonts/liberation/LiberationSans-BoldItalic.ttf",
]

# --- colours of the reference artwork: reds, red-orange highlights, white text, black ---------------------
BLACK: RGB = (0.0, 0.0, 0.0)
DEEP: RGB = (0.20, 0.0, 0.0)
DARK: RGB = (0.48, 0.02, 0.01)
RED: RGB = (0.82, 0.05, 0.02)
BRIGHT: RGB = (1.0, 0.15, 0.05)
FLAME: RGB = (1.0, 0.33, 0.09)  # red-orange highlight
HOT: RGB = (1.0, 0.52, 0.30)  # the hottest highlight: still red-orange, never yellow
WHITE: RGB = (1.0, 1.0, 1.0)
# nearness (0 far side of the helix, 1 near side) -> colour of backbones and rungs
DEPTH_RAMP = [(0.0, DEEP), (0.35, DARK), (0.7, RED), (0.93, BRIGHT), (1.0, FLAME)]

# --- the RNA model ----------------------------------------------------------------------------------------
SEQUENCE = "GGCGCAGUCAUCGAUGACUGCGCC"  # 24 bp: a little over two turns
ICON_SEQUENCE = "GCGCAGUCAUCGAUGACUGCGC"  # 22 bp, two turns: the icon from 64 px
SMALL_SEQUENCE = "GGCGUCAGCCGC"  # 12 bp for the 32 and 48 px icons: a little over one turn
TINY_SEQUENCE = "GGCGUCAGCC"  # 10 bp for the 16 and 24 px icons: about one turn

# The view.  The helix axis stays vertical; VIEW_AZIMUTH turns the model about it, VIEW_ELEVATION looks
# down on it from slightly above.  Along any vertical line on the near side the P ribbons alternate
# between a narrow gap of 11.3 A (the deep, narrow major groove: the two near strands run close together
# and the dark pairs on the far side show through it) and a wide gap of 19.6 A (the wide, shallow minor
# groove, which the bright near pairs fill almost to the backbones).  At this azimuth the wide top of
# the logo's funnel shows a near minor groove (85 % of the height) with its inclined rungs, then a major
# groove at 64 %, a minor groove at 44 % and a major groove at 23 %; the elevation adds the inclination
# of the pairs and, at the open top, the hollow core (pairs about 5 A off the axis).
VIEW_AZIMUTH = 160.0
VIEW_ELEVATION = 9.0
P_RADIUS = 8.71  # A: radius of the P atoms in the Arnott model (the backbone ribbons)


@dataclass
class Model:
    """Coordinates (A) of the A-form duplex, helix axis along z, per strand 5'->3'."""
    trace: List[np.ndarray]  # backbone ribbon: every P atom, then the O3' of the 3'-terminal residue
    c4: List[np.ndarray]  # (n, 3): where the rungs leave the ribbon
    c1: List[np.ndarray]
    hb: List[np.ndarray]  # N1 of purines, N3 of pyrimidines: the central Watson-Crick hydrogen bond
    pairs: List[Tuple[int, int]]  # (index on strand 1, index on strand 2)

    @property
    def z_range(self) -> Tuple[float, float]:
        zs = np.concatenate([c[:, 2] for c in self.trace])
        return float(zs.min()), float(zs.max())


def load_model(sequence: str) -> Model:
    """Build the ideal A-form duplex with MDATK's builder and read the atoms back from its PDB text.

    The chains get 5'-phosphates (``phosphate5``) so every nucleotide has its P atom on the ribbon.
    """
    text = builder.build_helix(sequence, chirality="D", scheme="amber", phosphate5=True)
    structure = pdbtools.read_pdb(text)
    if len(structure.segments) != 2:
        raise RuntimeError("the builder did not return a duplex")
    trace, c4, c1, hb = [], [], [], []
    for strand in structure.segments:
        rows = {"P": [], "C4'": [], "C1'": [], "hb": []}
        for res in strand:
            xyz = {name: np.array(v, dtype=float) for name, v in res.coords().items()}
            purine = pdbtools.parse_residue_name(res.name).base in "AG"
            for name in ("P", "C4'", "C1'"):
                rows[name].append(xyz[name])
            rows["hb"].append(xyz["N1"] if purine else xyz["N3"])
        trace.append(np.array(rows["P"] + [xyz["O3'"]]))
        c4.append(np.array(rows["C4'"]))
        c1.append(np.array(rows["C1'"]))
        hb.append(np.array(rows["hb"]))
    n = len(sequence)
    # residue k of strand 1 pairs residue n-1-k of strand 2 (antiparallel, at the same helix level)
    return Model(trace, c4, c1, hb, [(k, n - 1 - k) for k in range(n)])


# --- funnel taper and view --------------------------------------------------------------------------------
@dataclass
class Funnel:
    """Radial taper of the helix: radius factor ``tip`` at the bottom and ``top`` at the top.

    Only the distance from the axis changes, so azimuth and height of every atom (twist, rise, handedness,
    the 11-bp repeat, azimuth and vertical width of the grooves) are those of the A-form model.
    """
    z_lo: float
    z_hi: float
    tip: float = 0.22
    top: float = 1.35
    power: float = 1.5  # > 1: a trumpet-like funnel, flaring towards the top

    def scale(self, z):
        h = np.clip((np.asarray(z, dtype=float) - self.z_lo) / (self.z_hi - self.z_lo), 0.0, 1.0)
        return self.tip + (self.top - self.tip) * h ** self.power

    def radius(self, z) -> float:
        return P_RADIUS * float(self.scale(z))

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        out = np.array(xyz, dtype=float)
        s = self.scale(out[..., 2])
        out[..., 0] *= s
        out[..., 1] *= s
        return out


class View:
    """Orthographic view: model (A) -> screen (px) and depth (A, larger = nearer the viewer).

    The viewer looks along -toward, with toward = (sin a, cos a, 0) tilted up by the elevation, and the
    screen axes right = (-cos a, sin a, 0) and up = z: a proper rotation (right x up = toward), so a
    right-handed helix stays right-handed on screen (the near strands run up to the right).
    """

    def __init__(self, azimuth: float, elevation: float, scale: float, cx: float, cy: float, zc: float):
        a, e = math.radians(azimuth), math.radians(elevation)
        self.ca, self.sa, self.ce, self.se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
        self.k, self.cx, self.cy, self.zc = scale, cx, cy, zc

    def basis(self) -> np.ndarray:
        """Rows right, up, toward (model coordinates); its determinant is +1 (no mirror image)."""
        right = np.array([-self.ca, self.sa, 0.0])
        toward0 = np.array([self.sa, self.ca, 0.0])
        up = np.array([0.0, 0.0, self.ce]) - toward0 * self.se
        toward = toward0 * self.ce + np.array([0.0, 0.0, self.se])
        return np.array([right, up, toward])

    def project(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2] - self.zc
        right = -self.ca * x + self.sa * y
        d = self.sa * x + self.ca * y  # towards the viewer, before the elevation tilt
        up = z * self.ce - d * self.se
        depth = d * self.ce + z * self.se
        return self.cx + self.k * right, self.cy - self.k * up, depth

    def axial_depth(self, xyz: np.ndarray) -> np.ndarray:
        """Depth relative to the helix axis at the same height (> 0: in front of the axis)."""
        return self.sa * xyz[..., 0] + self.ca * xyz[..., 1]


def mix(a: RGB, b: RGB, t: float) -> RGB:
    t = min(1.0, max(0.0, float(t)))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def ramp(stops: Sequence[Tuple[float, RGB]], t: float) -> RGB:
    if t <= stops[0][0]:
        return stops[0][1]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return mix(c0, c1, (t - t0) / (t1 - t0))
    return stops[-1][1]


# --- drawing primitives -----------------------------------------------------------------------------------
LIGHT = (-0.5, -0.866)  # screen direction towards the light: upper left (y points down)


def quad(ctx: cairo.Context, p0, p1, w0: float, w1: float, colour: RGB, shade: bool = False,
         lean: bool = False) -> None:
    """One piece of a variable-width stroke from p0 to p1, overlapping its neighbours a little.

    ``shade``: a tube-like gradient across the stroke, dark edges and a highlight on the side facing
    ``LIGHT``; the highlight moves smoothly with the direction of the stroke, so the pieces join seamlessly.
    ``lean``: three gradient stops instead of five (a smaller SVG).
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1e-9
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    ext = min(0.5, 0.3 * max(w0, w1))
    ax, ay = p0[0] - ux * ext, p0[1] - uy * ext
    bx, by = p1[0] + ux * ext, p1[1] + uy * ext
    ctx.move_to(ax + nx * w0 / 2, ay + ny * w0 / 2)
    ctx.line_to(bx + nx * w1 / 2, by + ny * w1 / 2)
    ctx.line_to(bx - nx * w1 / 2, by - ny * w1 / 2)
    ctx.line_to(ax - nx * w0 / 2, ay - ny * w0 / 2)
    ctx.close_path()
    if not shade:
        ctx.set_source_rgb(*colour)
    else:
        w = (w0 + w1) / 2
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        lit = nx * LIGHT[0] + ny * LIGHT[1]  # +1: the +n edge faces the light
        hl = 0.5 - 0.36 * lit
        g = cairo.LinearGradient(mx + nx * w / 2, my + ny * w / 2, mx - nx * w / 2, my - ny * w / 2)
        g.add_color_stop_rgb(0.0, *mix(colour, BLACK, 0.45 - 0.25 * lit))
        if not lean:
            g.add_color_stop_rgb(max(0.04, hl - 0.24), *colour)
        g.add_color_stop_rgb(hl, *mix(colour, HOT, 0.42))
        if not lean:
            g.add_color_stop_rgb(min(0.96, hl + 0.26), *colour)
        g.add_color_stop_rgb(1.0, *mix(colour, BLACK, 0.45 + 0.25 * lit))
        ctx.set_source(g)
    ctx.fill()


def bead(ctx: cairo.Context, x: float, y: float, r: float, colour: RGB, flat: bool = False) -> None:
    """A shaded bead; ``flat``: a disc with a highlight dot instead of a gradient (a smaller SVG)."""
    if flat:
        ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.set_source_rgb(*colour)
        ctx.fill()
        ctx.arc(x - 0.3 * r, y - 0.32 * r, 0.38 * r, 0, 2 * math.pi)
        ctx.set_source_rgb(*mix(colour, HOT, 0.6))
        ctx.fill()
        return
    g = cairo.RadialGradient(x - 0.35 * r, y - 0.4 * r, 0.05 * r, x, y, r)
    g.add_color_stop_rgb(0.0, *mix(colour, HOT, 0.7))
    g.add_color_stop_rgb(0.55, *colour)
    g.add_color_stop_rgb(1.0, *mix(colour, BLACK, 0.5))
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.set_source(g)
    ctx.fill()


def stroke_outline(ctx: cairo.Context, xs: np.ndarray, ys: np.ndarray, widths: np.ndarray) -> None:
    """Path of a smooth variable-width stroke along a polyline (left side out, right side back)."""
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    length = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / length, dx / length
    left = list(zip(xs + nx * widths / 2, ys + ny * widths / 2))
    right = list(zip(xs - nx * widths / 2, ys - ny * widths / 2))
    ctx.move_to(*left[0])
    for p in left[1:]:
        ctx.line_to(*p)
    for p in reversed(right):
        ctx.line_to(*p)
    ctx.close_path()


@dataclass
class Style:
    """Sizes (A, before the taper) and level of detail of one helix drawing."""
    ribbon: float = 2.3  # backbone width
    rung: float = 0.85
    bead: float = 0.8  # bead radius
    beads: bool = True
    shade: bool = True  # tube-like shading of the backbones
    samples: int = 14  # spline samples per residue
    rung_step: float = 0.6  # A per rung piece (depth sorting)
    pair_step: int = 1  # draw every pair_step-th base pair (the small icons: every second)
    vector: bool = False  # fewer gradients (the SVG)


def helix_prims(model: Model, funnel: Funnel, view: View, style: Style) -> List[Prim]:
    """The two backbone ribbons, the rungs and the beads as depth-sorted pieces."""
    prims: List[Prim] = []
    k = view.k

    def width_at(z: float, base: float) -> float:
        return k * base * (0.4 + 0.6 * float(funnel.scale(z)) / funnel.top)

    def nearness(xyz: np.ndarray) -> float:
        r = float(np.hypot(xyz[0], xyz[1])) or 1.0
        return 0.5 + 0.5 * max(-1.0, min(1.0, float(view.axial_depth(xyz)) / r))

    anchors = []  # per strand and residue: the ribbon point next to the C4' atom, where the rung starts
    for strand, c4 in zip(model.trace, model.c4):
        t = np.arange(len(strand))
        ts = np.linspace(0, len(strand) - 1, (len(strand) - 1) * style.samples + 1)
        raw = CubicSpline(t, strand, axis=0)(ts)
        near = np.argmin(np.linalg.norm(raw[None, :, :] - c4[:, None, :], axis=-1), axis=1)
        anchors.append(raw[near])
        pts = funnel.apply(raw)
        sx, sy, dep = view.project(pts)
        m = len(ts)
        for i in range(m - 1):
            # both chain ends narrow to a point over half a residue: flame tips
            end = min(i, m - 2 - i) / (style.samples * 0.5)
            cap = min(1.0, 0.2 + 0.8 * end)
            f = nearness(pts[i])
            grow = (0.78 + 0.32 * f) * cap
            w0, w1 = width_at(pts[i, 2], style.ribbon) * grow, width_at(pts[i + 1, 2], style.ribbon) * grow
            colour = ramp(DEPTH_RAMP, f)

            def draw(ctx, p0=(sx[i], sy[i]), p1=(sx[i + 1], sy[i + 1]), w0=w0, w1=w1, colour=colour):
                quad(ctx, p0, p1, w0, w1, colour, shade=style.shade, lean=style.vector)
            prims.append((float(dep[i] + dep[i + 1]) / 2, draw))

    for i, j in model.pairs[::style.pair_step]:
        chain = [anchors[0][i], model.c1[0][i], model.hb[0][i], model.hb[1][j], model.c1[1][j], anchors[1][j]]
        pts = []
        for a, b in zip(chain, chain[1:]):
            n = max(1, int(round(np.linalg.norm(b - a) / style.rung_step)))
            pts.extend(a + (b - a) * u for u in np.linspace(0, 1, n, endpoint=False))
        pts.append(chain[-1])
        pts = funnel.apply(np.array(pts))
        sx, sy, dep = view.project(pts)
        for q in range(len(pts) - 1):
            f = nearness(pts[q])
            w = width_at(pts[q, 2], style.rung) * (0.8 + 0.3 * f)
            colour = mix(ramp(DEPTH_RAMP, f), DEEP, 0.1)

            def draw(ctx, p0=(sx[q], sy[q]), p1=(sx[q + 1], sy[q + 1]), w=w, colour=colour):
                quad(ctx, p0, p1, w, w, colour)
            prims.append((float(dep[q] + dep[q + 1]) / 2 - 0.05, draw))
        if style.beads:
            for xyz in funnel.apply(np.array(chain[1:5])):
                bx, by, bd = view.project(xyz)
                f = nearness(xyz)
                r = width_at(xyz[2], style.bead) * (0.85 + 0.25 * f)
                colour = ramp(DEPTH_RAMP, min(1.0, f + 0.08))

                def draw(ctx, x=float(bx), y=float(by), r=r, colour=colour):
                    bead(ctx, x, y, r, colour, flat=style.vector)
                prims.append((float(bd) + 0.05, draw))
    return prims


# --- the tornado around the helix -------------------------------------------------------------------------
@dataclass
class Decor:
    """Swirl strokes, motion streaks and embers around the funnel (counts and sizes in A)."""
    swirls: int = 22
    swirl_width: Tuple[float, float] = (0.35, 1.0)
    streaks: int = 12
    embers: int = 110
    ember_size: Tuple[float, float] = (0.12, 0.42)
    seed: int = 11
    samples: int = 90  # points per swirl stroke
    vector: bool = False  # embers without gradients (the SVG)
    spread: float = 1.0  # radial reach of the decoration, relative
    crown: float = 1.0  # extra swirls at the top (the cloud base the funnel hangs from)


@dataclass
class Tornado:
    """The pieces of one drawing, in the order they are painted (each list sorted by depth)."""
    back: List[Prim] = field(default_factory=list)  # behind the helix
    helix: List[Prim] = field(default_factory=list)
    front: List[Prim] = field(default_factory=list)  # in front of the helix
    embers: List[Prim] = field(default_factory=list)


def _swirl(t: Tornado, view: View, xyz: np.ndarray, widths: np.ndarray, colours: Tuple[RGB, RGB],
           alpha: float, back_dim: float = 0.55, fade: float = 0.35) -> None:
    """Split a 3D stroke into its parts behind and in front of the axis; each part one gradient fill.

    The colour runs from ``colours[0]`` at the tail to ``colours[1]`` at the head; the first ``fade`` of
    the stroke fades in from transparent (a wisp, not a band).
    """
    sx, sy, _ = view.project(xyz)
    front = view.axial_depth(xyz) >= 0
    n = len(xyz)
    start = 0
    for i in range(1, n + 1):
        if i < n and front[i] == front[start]:
            continue
        lo, hi = max(0, start - 1), min(n, i + 1)  # overlap one sample so the parts join
        if hi - lo >= 2:
            is_front = bool(front[start])
            dim = 1.0 if is_front else back_dim
            stops = []
            for u in np.linspace(lo, hi - 1, 4) / (n - 1):
                colour = mix(BLACK, mix(*colours, u), dim)
                stops.append((colour, alpha * min(1.0, u / fade) if fade else alpha))

            def draw(ctx, xs=sx[lo:hi], ys=sy[lo:hi], ws=widths[lo:hi], stops=stops):
                stroke_outline(ctx, xs, ys, ws)
                g = cairo.LinearGradient(xs[0], ys[0], xs[-1], ys[-1])
                for q, (colour, a) in enumerate(stops):
                    g.add_color_stop_rgba(q / (len(stops) - 1), *colour, a)
                ctx.set_source(g)
                ctx.fill()
            depth = float(np.mean(view.axial_depth(xyz[lo:hi])))
            (t.front if is_front else t.back).append((depth, draw))
        start = i


def tornado(model: Model, funnel: Funnel, view: View, style: Style, decor: Optional[Decor]) -> Tornado:
    t = Tornado(helix=helix_prims(model, funnel, view, style))
    if decor is None:
        return t
    rng = np.random.default_rng(decor.seed)
    z_lo, z_hi = funnel.z_lo, funnel.z_hi
    height = z_hi - z_lo
    k = view.k

    def funnel_r(z: float) -> float:
        return funnel.radius(min(max(z, z_lo), z_hi))

    # flame-like swirl strokes: counter-clockwise seen from above (as the helix winds), rising slowly; a
    # thin tail, a flickering body and a head that ends in a point, each with a thinner companion stroke
    # beside it like a brush stroke; a few wider ones at the top (the crown the funnel hangs from)
    n_crown = int(round(decor.swirls * 0.25 * decor.crown))
    for s in range(decor.swirls + n_crown):
        crown = s >= decor.swirls
        h = rng.uniform(0.98, 1.06) if crown else 0.97 - 1.02 * rng.random() ** 1.1
        z0 = z_lo + h * height
        reach = rng.uniform(1.3, 1.65) if crown else rng.uniform(1.25, 1.7)
        r0 = funnel_r(z0) * reach * decor.spread + 1.0
        phi0 = rng.uniform(0, 2 * math.pi)
        span = rng.uniform(0.8, 1.5) * math.pi
        rise = rng.uniform(0.2, 0.6)
        wob, wph = rng.uniform(0.03, 0.07), rng.uniform(0, 2 * math.pi)
        wmax = rng.uniform(*decor.swirl_width) * (0.45 + 0.55 * float(funnel.scale(z0)) / funnel.top)
        head = FLAME if rng.random() < 0.6 else BRIGHT
        alpha = rng.uniform(0.75, 0.95)
        for part, (dr, dz, wf, sf) in enumerate(((0.0, 0.0, 1.0, 1.0), (0.9, -0.5, 0.4, 0.7))):
            if part and wmax * wf * k < 1.0:
                continue
            phi = np.linspace(phi0 + (1 - sf) * span, phi0 + span, decor.samples)
            u = (phi - phi[0]) / (phi[-1] - phi[0])
            r = (r0 + dr) * (1 + wob * np.sin(2.5 * phi + wph)) * (1 + 0.1 * u)
            z = z0 + dz + rise * (phi - phi0) + 0.4 * np.sin(3.1 * phi + wph)
            xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)
            flicker = 1 + 0.22 * np.sin(2 * np.pi * (2.5 * u) + wph)
            widths = k * wmax * wf * np.clip(np.sin(np.pi * u ** 1.4), 0, 1) ** 0.8 * flicker
            _swirl(t, view, xyz, widths, (DARK, head), alpha=alpha * (1 - 0.25 * part))

    # motion streaks: thin, faint, long arcs further out
    for _ in range(decor.streaks):
        h = rng.uniform(0.0, 1.05)
        z0 = z_lo + h * height
        r0 = funnel_r(z0) * rng.uniform(1.7, 2.4) * decor.spread + 2.0
        phi0 = rng.uniform(0, 2 * math.pi)
        span = rng.uniform(0.6, 1.3) * math.pi
        phi = np.linspace(phi0, phi0 + span, decor.samples)
        u = (phi - phi0) / span
        xyz = np.stack([r0 * np.cos(phi), r0 * np.sin(phi), z0 + 0.3 * (phi - phi0)], axis=-1)
        widths = k * rng.uniform(0.12, 0.3) * np.clip(np.sin(np.pi * u), 0, 1) ** 0.6
        _swirl(t, view, xyz, widths, (DEEP, RED), alpha=rng.uniform(0.35, 0.6), back_dim=0.7, fade=0.0)

    # embers: glowing sparks, some drawn out along their path
    for _ in range(decor.embers):
        h = rng.uniform(-0.08, 1.12)
        z = z_lo + h * height
        r = funnel_r(z) * (1.1 + 1.9 * rng.random() ** 1.6) * decor.spread + 1.0
        phi = rng.uniform(0, 2 * math.pi)
        xyz = np.array([r * math.cos(phi), r * math.sin(phi), z])
        size = k * rng.uniform(*decor.ember_size) * (1.6 if rng.random() < 0.08 else 1.0)
        trail = rng.random() < 0.45
        colour = (FLAME, BRIGHT, RED, DARK)[min(3, int(rng.random() ** 1.1 * 4))]
        sx, sy, dep = view.project(xyz)
        if trail:  # a short streak back along the swirl (clockwise from the spark)
            back = np.array([r * math.cos(phi - 0.25), r * math.sin(phi - 0.25), z - 0.15])
            tx, ty, _ = view.project(back)
        else:
            tx, ty = sx, sy

        def draw(ctx, x=float(sx), y=float(sy), tx=float(tx), ty=float(ty), size=size, colour=colour):
            if (tx, ty) != (x, y):
                if decor.vector:
                    ctx.set_source_rgba(*colour, 0.45)
                else:
                    g = cairo.LinearGradient(tx, ty, x, y)
                    g.add_color_stop_rgba(0.0, *colour, 0.0)
                    g.add_color_stop_rgba(1.0, *colour, 0.9)
                    ctx.set_source(g)
                ctx.set_line_width(size * 1.1)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.move_to(tx, ty)
                ctx.line_to(x, y)
                ctx.stroke()
            if decor.vector:
                ctx.arc(x, y, size * 0.8, 0, 2 * math.pi)
                ctx.set_source_rgb(*mix(colour, HOT, 0.35))
                ctx.fill()
                return
            g = cairo.RadialGradient(x, y, 0, x, y, size * 1.6)
            g.add_color_stop_rgba(0.0, *mix(colour, HOT, 0.35), 1.0)
            g.add_color_stop_rgba(0.35, *colour, 0.95)
            g.add_color_stop_rgba(1.0, *colour, 0.0)
            ctx.set_source(g)
            ctx.arc(x, y, size * 1.6, 0, 2 * math.pi)
            ctx.fill()
        t.embers.append((float(dep), draw))
    return t


def fit_view(model: Model, funnel: Funnel, box: Tuple[float, float, float, float],
             azimuth: float = VIEW_AZIMUTH, elevation: float = VIEW_ELEVATION, margin: float = 0.0) -> View:
    """A view that puts the tapered helix (plus ``margin`` A around it) into box (x0, y0, x1, y1)."""
    pts = funnel.apply(np.concatenate(model.trace + model.c1))
    probe = View(azimuth, elevation, 1.0, 0.0, 0.0, 0.0)
    sx, sy, _ = probe.project(pts)
    w = float(sx.max() - sx.min()) + 2 * margin
    h = float(sy.max() - sy.min()) + 2 * margin
    x0, y0, x1, y1 = box
    k = min((x1 - x0) / w, (y1 - y0) / h)
    cx = (x0 + x1) / 2 - k * float(sx.max() + sx.min()) / 2
    cy = (y0 + y1) / 2 - k * float(sy.max() + sy.min()) / 2
    return View(azimuth, elevation, k, cx, cy, 0.0)


# --- text -------------------------------------------------------------------------------------------------
def find_font(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    for path in FONTS:
        if os.path.isfile(path):
            return path
    import matplotlib
    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-BoldOblique.ttf")


@dataclass
class Text:
    """Glyph outlines of a string (font units of a 100-unit font, y up) from matplotlib's TextPath."""
    segments: list
    x0: float
    x1: float
    cap: float  # height of the capitals

    @classmethod
    def of(cls, text: str, font: str) -> "Text":
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
        prop = FontProperties(fname=font)
        path = TextPath((0, 0), text, size=100, prop=prop)
        cap = TextPath((0, 0), "N", size=100, prop=prop).get_extents()
        ext = path.get_extents()
        return cls(list(path.iter_segments(curves=True, simplify=False)), ext.x0, ext.x1, cap.y1)


STRETCH = 1.18  # the wordmark is set wider than the font (a wide, extended look)


def text_path(ctx: cairo.Context, text: Text, x: float, baseline: float, cap: float, stretch: float = STRETCH,
              shear: float = 0.08) -> float:
    """Add the outlines to ``ctx``: left edge at x, capitals ``cap`` px high, widened by ``stretch`` and
    slanted a little further by ``shear``; returns the right edge."""
    s = cap / text.cap

    def tr(vx: float, vy: float) -> Tuple[float, float]:
        return x + s * stretch * (vx - text.x0 + shear * vy), baseline - s * vy

    current = (0.0, 0.0)
    for verts, code in text.segments:
        v = list(verts)
        if code == 1:  # MOVETO
            ctx.move_to(*tr(v[0], v[1]))
            current = (v[0], v[1])
        elif code == 2:  # LINETO
            ctx.line_to(*tr(v[0], v[1]))
            current = (v[0], v[1])
        elif code == 3:  # CURVE3: quadratic -> cubic
            (qx, qy), (ex, ey) = (v[0], v[1]), (v[2], v[3])
            c1 = (current[0] + 2 / 3 * (qx - current[0]), current[1] + 2 / 3 * (qy - current[1]))
            c2 = (ex + 2 / 3 * (qx - ex), ey + 2 / 3 * (qy - ey))
            ctx.curve_to(*tr(*c1), *tr(*c2), *tr(ex, ey))
            current = (ex, ey)
        elif code == 4:  # CURVE4
            ctx.curve_to(*tr(v[0], v[1]), *tr(v[2], v[3]), *tr(v[4], v[5]))
            current = (v[4], v[5])
        elif code == 79:  # CLOSEPOLY
            ctx.close_path()
    return x + s * stretch * (text.x1 - text.x0)


def swoosh(ctx: cairo.Context, x0: float, x1: float, y: float, h: float) -> None:
    """The red swoosh underline: a brush stroke from a point on the left, swelling under the first
    letters, sweeping up to a sharp point on the right."""
    L = x1 - x0
    ctx.move_to(x0, y)
    ctx.curve_to(x0 + 0.25 * L, y + 0.9 * h, x0 + 0.6 * L, y + 0.8 * h, x1, y - 0.45 * h)
    ctx.curve_to(x0 + 0.6 * L, y + 0.25 * h, x0 + 0.3 * L, y + 0.15 * h, x0, y)
    ctx.close_path()
    g = cairo.LinearGradient(x0, y, x1, y)
    g.add_color_stop_rgb(0.0, *DARK)
    g.add_color_stop_rgb(0.35, *RED)
    g.add_color_stop_rgb(0.8, *BRIGHT)
    g.add_color_stop_rgb(1.0, *FLAME)
    ctx.set_source(g)
    ctx.fill()


def wordmark(ctx: cairo.Context, text: Text, x: float, baseline: float, cap: float) -> float:
    """White "RNAdo" with the swoosh below it; returns the right edge of the text."""
    ctx.new_path()
    right = text_path(ctx, text, x, baseline, cap)
    ctx.set_source_rgb(*WHITE)
    ctx.fill()
    swoosh(ctx, x - 0.12 * cap, right + 0.06 * cap, baseline + 0.2 * cap, 0.42 * cap)
    return right


def wordmark_glow_path(ctx: cairo.Context, text: Text, x: float, baseline: float, cap: float) -> None:
    ctx.new_path()
    right = text_path(ctx, text, x, baseline, cap)
    ctx.set_source_rgb(*RED)
    ctx.fill()
    swoosh(ctx, x - 0.12 * cap, right + 0.06 * cap, baseline + 0.2 * cap, 0.42 * cap)


# --- raster compositing (premultiplied RGBA floats) -------------------------------------------------------
def layer(w: int, h: int, prims: Sequence[Prim], extra: Optional[Draw] = None) -> np.ndarray:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    for _, draw in sorted(prims, key=lambda p: p[0]):
        draw(ctx)
    if extra:
        extra(ctx)
    surface.flush()
    stride = surface.get_stride()
    raw = np.ndarray((h, stride // 4, 4), dtype=np.uint8, buffer=surface.get_data())[:, :w]
    arr = raw.astype(np.float32) / 255.0
    return arr[..., [2, 1, 0, 3]] if sys.byteorder == "little" else arr[..., [1, 2, 3, 0]]


def over(dst: np.ndarray, src: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    src = src * alpha
    return src + dst * (1.0 - src[..., 3:4])


def bloom(src: np.ndarray, sigmas: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Light of a layer: blurred copies (sigma px, strength), added to what is below."""
    out = np.zeros_like(src)
    for sigma, strength in sigmas:
        blurred = [gaussian_filter(src[..., c], sigma, mode="constant") for c in range(4)]
        out += strength * np.stack(blurred, axis=-1)
    return out


def add_light(dst: np.ndarray, light: np.ndarray) -> np.ndarray:
    out = dst.copy()
    out[..., :3] += light[..., :3]
    out[..., 3] = np.maximum(dst[..., 3], np.minimum(1.0, light[..., 3]))
    return np.clip(out, 0.0, 1.0)


def palette_clamp(arr: np.ndarray) -> np.ndarray:
    """Keep added light in the reds: green and blue never outgrow the red-orange highlight (no yellow)."""
    out = arr.copy()
    out[..., 1] = np.minimum(out[..., 1], 0.56 * out[..., 0])
    out[..., 2] = np.minimum(out[..., 2], 0.36 * out[..., 0])
    return out


def to_image(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr[..., 3:4], 0.0, 1.0)
    rgb = np.where(a > 1e-6, arr[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([np.clip(rgb, 0.0, 1.0), a], axis=-1)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA")


def render_scene(w: int, h: int, t: Tornado, background: Draw, foreground: Optional[Draw] = None,
                 glow: float = 1.0, foreground_glow: Optional[Draw] = None) -> np.ndarray:
    """Background, then (each with its own glow) back decoration, helix, front decoration, embers, text."""
    bg = layer(w, h, [], background)
    scene = bg
    unit = min(w, h)
    for prims, sigmas, alpha in (
            (t.back, [(unit / 120, 0.5), (unit / 40, 0.3)], 1.0),
            (t.helix, [(unit / 160, 0.45), (unit / 45, 0.5), (unit / 14, 0.25)], 1.0),
            (t.front, [(unit / 120, 0.5), (unit / 40, 0.3)], 0.85),
            (t.embers, [(unit / 300, 0.5), (unit / 90, 0.35)], 1.0)):
        if not prims:
            continue
        sharp = layer(w, h, prims)
        if glow > 0:
            scene = add_light(scene, bloom(sharp, [(s, g * glow) for s, g in sigmas]))
        scene = over(scene, sharp, alpha)
    scene = palette_clamp(scene)
    if foreground:
        if foreground_glow and glow > 0:
            scene = add_light(scene, bloom(layer(w, h, [], foreground_glow), [(unit / 60, 0.35 * glow)]))
        scene = over(scene, layer(w, h, [], foreground))
    return scene


def black(ctx: cairo.Context) -> None:
    ctx.set_source_rgb(*BLACK)
    ctx.paint()


def icon_background(size: float) -> Draw:
    """Black rounded square with a faint dark-red core light; transparent corners."""
    def draw(ctx: cairo.Context) -> None:
        r = 0.2 * size
        inset = 0.02 * size if size > 20 else 0.0
        x0, y0, x1, y1 = inset, inset, size - inset, size - inset
        ctx.new_path()
        ctx.arc(x1 - r, y0 + r, r, -math.pi / 2, 0)
        ctx.arc(x1 - r, y1 - r, r, 0, math.pi / 2)
        ctx.arc(x0 + r, y1 - r, r, math.pi / 2, math.pi)
        ctx.arc(x0 + r, y0 + r, r, math.pi, 3 * math.pi / 2)
        ctx.close_path()
        g = cairo.RadialGradient(size / 2, size * 0.45, 0, size / 2, size * 0.45, size * 0.75)
        g.add_color_stop_rgb(0.0, 0.13, 0.0, 0.0)
        g.add_color_stop_rgb(1.0, *BLACK)
        ctx.set_source(g)
        ctx.fill()
    return draw


# --- the assets -------------------------------------------------------------------------------------------
def make_logo(font: str) -> Image.Image:
    w, h = LOGO_SIZE
    model = load_model(SEQUENCE)
    funnel = Funnel(*model.z_range)
    view = fit_view(model, funnel, (0.16 * w, 0.09 * h, 0.84 * w, 0.76 * h))
    t = tornado(model, funnel, view, Style(), Decor())
    text = Text.of("RNAdo", font)
    cap = 0.097 * h
    width = cap / text.cap * STRETCH * (text.x1 - text.x0)
    x = (w - width) / 2 - 0.02 * w
    base = 0.872 * h

    def fg(ctx):
        wordmark(ctx, text, x, base, cap)

    def fg_glow(ctx):
        wordmark_glow_path(ctx, text, x, base, cap)
    return to_image(render_scene(w, h, t, black, fg, foreground_glow=fg_glow)).convert("RGB")


def make_banner(font: str) -> Image.Image:
    w, h = BANNER_SIZE
    model = load_model(ICON_SEQUENCE)
    funnel = Funnel(*model.z_range, top=1.65)
    view = fit_view(model, funnel, (0, 0.13 * h, 1e6, 0.95 * h), margin=3.0)  # as tall as the banner allows
    helix_w = view.k * helix_width(model, funnel)
    text = Text.of("RNAdo", font)
    cap = 0.33 * h
    text_w = cap / text.cap * STRETCH * (text.x1 - text.x0)
    gap = 0.75 * helix_w  # room for the swirls right of the funnel
    x0 = (w - (helix_w + gap + text_w)) / 2 + 0.15 * helix_w  # the swirls reach further left than the text
    view.cx += x0 + helix_w / 2 - 0.5e6
    decor = Decor(swirls=18, streaks=8, embers=70, seed=5, spread=0.9, crown=0.6)
    t = tornado(model, funnel, view, Style(ribbon=2.6, rung=1.0, bead=0.9), decor)
    x = x0 + helix_w + gap
    base = 0.6 * h

    def fg(ctx):
        wordmark(ctx, text, x, base, cap)

    def fg_glow(ctx):
        wordmark_glow_path(ctx, text, x, base, cap)
    return to_image(render_scene(w, h, t, black, fg, foreground_glow=fg_glow)).convert("RGB")


def helix_width(model: Model, funnel: Funnel) -> float:
    """Projected width (A) of the tapered helix in the standard view."""
    probe = View(VIEW_AZIMUTH, VIEW_ELEVATION, 1.0, 0, 0, 0)
    sx, _, _ = probe.project(funnel.apply(np.concatenate(model.trace)))
    return float(sx.max() - sx.min())


def icon_scene(size: int, px: float, vector: bool = False) -> Tuple[Tornado, Draw, float]:
    """Tornado, background and glow strength of the ``size`` px icon, drawn ``px`` px large.

    Up to TINY_ICON: one turn, every second pair, no decoration; up to SMALL_ICON: a little over one turn
    and one swirl; both with thick strands and rungs and no beads, so the twisted funnel stays readable.
    Larger: two turns with beads, swirls, streaks and embers.  ``vector``: fewer, larger pieces (the SVG).
    """
    if size <= SMALL_ICON:
        tiny = size <= TINY_ICON
        model = load_model(TINY_SEQUENCE if tiny else SMALL_SEQUENCE)
        funnel = Funnel(*model.z_range, tip=0.3, top=1.6, power=1.2)
        top = 0.1 if tiny else 0.15  # room for the swirl above the funnel
        view = fit_view(model, funnel, (0.12 * px, top * px, 0.88 * px, 0.9 * px), elevation=12.0)
        style = Style(ribbon=6.5 if tiny else 6.0, rung=3.0 if tiny else 2.8, beads=False, shade=not tiny,
                      samples=10, rung_step=1.5, pair_step=2)
        decor = None if tiny else Decor(swirls=1, swirl_width=(2.4, 2.8), streaks=0, embers=0, seed=3,
                                        spread=0.85, crown=0.0)
        glow = 0.3 if tiny else 0.4
    else:
        model = load_model(ICON_SEQUENCE)
        funnel = Funnel(*model.z_range, top=2.0, power=1.6)  # wider than the logo's: fills the square
        view = fit_view(model, funnel, (0.1 * px, 0.13 * px, 0.9 * px, 0.92 * px), margin=4.0)
        style = Style(ribbon=3.4, rung=1.3, bead=1.1, samples=4 if vector else 14,
                      rung_step=2.0 if vector else 0.6, vector=vector)
        decor = Decor(swirls=16, streaks=6, embers=28 if vector else 50, seed=7, spread=1.1, crown=1.0,
                      samples=24 if vector else 90, vector=vector)
        glow = 0.8
    return tornado(model, funnel, view, style, decor), icon_background(px), glow


def make_icon(size: int) -> Image.Image:
    # drawn larger (at least 256 px) and scaled down: smooth edges, and the glow keeps its proportions
    scale = max(1, int(math.ceil(1024 / size)) if size > SMALL_ICON else int(math.ceil(256 / size)))
    big = size * scale
    t, bg, glow = icon_scene(size, big)
    arr = render_scene(big, big, t, bg, glow=glow)
    # the glow must not leak out of the rounded square: clip to the background's alpha
    arr = arr * np.clip(layer(big, big, [], bg)[..., 3:4], 0.0, 1.0)
    img = to_image(arr)
    return img.resize((size, size), Image.LANCZOS) if scale > 1 else img


def make_svg(path: str) -> None:
    """The large icon as vector art: a soft radial halo instead of the blurred glow, no text."""
    size = 512
    t, bg, _ = icon_scene(size, size, vector=True)
    buf = io.BytesIO()
    surface = cairo.SVGSurface(buf, size, size)
    surface.set_document_unit(cairo.SVGUnit.PX)
    ctx = cairo.Context(surface)
    bg(ctx)
    halo = cairo.RadialGradient(size / 2, size * 0.4, 0, size / 2, size * 0.4, size * 0.45)
    halo.add_color_stop_rgba(0.0, *RED, 0.3)
    halo.add_color_stop_rgba(1.0, *RED, 0.0)
    ctx.set_source(halo)
    ctx.paint()
    for prims in (t.back, t.helix, t.front, t.embers):
        for _, draw in sorted(prims, key=lambda p: p[0]):
            draw(ctx)
    surface.finish()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(compact_svg(buf.getvalue().decode("utf-8")))


def compact_svg(text: str) -> str:
    """Smaller SVG text, same picture: cairo writes six decimals (one is plenty in a 512 px icon), the
    default fill rule and a redundant move after every closed path."""
    head, sep, text = text.partition("?>")  # the XML declaration stays as it is
    text = text.replace(' fill-rule="nonzero"', "")
    text = re.sub(r"Z M [-\d.]+ [-\d.]+ ", "Z ", text)
    text = re.sub(r"(\d+\.\d)\d+", r"\1", text)
    text = re.sub(r"(\d+)\.0(?=\D)", r"\1", text)
    return head + sep + text


def save_png(img: Image.Image, path: str) -> None:
    img.save(path, optimize=True)


def make_sheets(out: str, sheet_dir: str) -> None:
    """Comparison sheets: every icon size on dark and light backgrounds, actual size and enlarged."""
    os.makedirs(sheet_dir, exist_ok=True)
    icons = [Image.open(os.path.join(out, f"rnado-icon-{n}.png")).convert("RGBA") for n in ICON_SIZES]
    pad = 16
    width = sum(i.width for i in icons) + pad * (len(icons) + 1)
    height = max(i.height for i in icons) + 2 * pad
    sheet = Image.new("RGBA", (width, 2 * height), (0, 0, 0, 0))
    for row, colour in enumerate(((48, 48, 52, 255), (238, 238, 236, 255))):
        sheet.paste(Image.new("RGBA", (width, height), colour), (0, row * height))
        x = pad
        for icon in icons:
            sheet.alpha_composite(icon, (x, row * height + pad + (height - 2 * pad - icon.height)))
            x += icon.width + pad
    sheet.save(os.path.join(sheet_dir, "icons_all_sizes.png"))
    zoom = 8  # the small sizes enlarged (nearest neighbour: every pixel visible)
    small = [i for i in icons if i.width <= 64]
    width = sum(i.width * zoom for i in small) + pad * (len(small) + 1)
    height = max(i.height for i in small) * zoom + 2 * pad
    sheet = Image.new("RGBA", (width, 2 * height), (0, 0, 0, 0))
    for row, colour in enumerate(((48, 48, 52, 255), (238, 238, 236, 255))):
        sheet.paste(Image.new("RGBA", (width, height), colour), (0, row * height))
        x = pad
        for icon in small:
            big = icon.resize((icon.width * zoom, icon.height * zoom), Image.NEAREST)
            sheet.alpha_composite(big, (x, row * height + pad))
            x += big.width + pad
    sheet.save(os.path.join(sheet_dir, "icons_small_zoomed.png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "mdatk", "assets"), help="output folder")
    ap.add_argument("--font", help="font file of the wordmark (default: the first of FONTS found)")
    ap.add_argument("--sheet", metavar="DIR", help="also write comparison sheets into DIR")
    ap.add_argument("--only", choices=("logo", "banner", "icons", "svg"), action="append",
                    help="draw only these assets (repeatable)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    font = find_font(args.font)
    only = set(args.only or ("logo", "banner", "icons", "svg"))
    print(f"font: {font}")
    if "logo" in only:
        save_png(make_logo(font), os.path.join(args.out, "rnado-logo.png"))
    if "banner" in only:
        save_png(make_banner(font), os.path.join(args.out, "rnado-wordmark.png"))
    if "icons" in only:
        for n in ICON_SIZES:
            save_png(make_icon(n), os.path.join(args.out, f"rnado-icon-{n}.png"))
    if "svg" in only:
        make_svg(os.path.join(args.out, "rnado-icon.svg"))
    for name in sorted(os.listdir(args.out)):
        if name.startswith("rnado-"):
            print(f"  {name:24s} {os.path.getsize(os.path.join(args.out, name)) / 1e3:7.1f} kB")
    if args.sheet and "icons" in only:
        make_sheets(args.out, args.sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
