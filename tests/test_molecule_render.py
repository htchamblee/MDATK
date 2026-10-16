"""The built-in 3D renderer: exact rasterisation, depth, registration with the axes,
compositing (fog, glow, translucency), threads, headless use and the pipeline."""

import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from mdatk.render.molecule.geometry import Mesh, Spheres, Sticks
from mdatk.render.molecule.scene import Colored, Shading
from mdatk.render.molecule.software import SoftwareRenderer, _Cam, _ZPass
from mdatk.render.molecule.stage import Viewport
from mdatk.render.molecule.surface import gaussian_surface
from mdatk.render.orientation import euler_to_matrix

WHITE = np.ones(3)
PLAIN = Shading(WHITE, 0.0)


def vp(R=None, lim=10.0, size=100, zrange=(-10.0, 10.0)):
    return Viewport(np.eye(3) if R is None else R, (-lim, lim), (-lim, lim), size, size, zrange)


def spheres(xyz, r, rgb=(0.8, 0.2, 0.2), owner=None):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    n = len(xyz)
    own = np.arange(n) if owner is None else np.asarray(owner)
    return Colored(Spheres(xyz, np.broadcast_to(np.asarray(r, float), (n,)).copy(), own, np.full(n, -1)),
                   np.tile(np.asarray(rgb, float), (n, 1)))


def resolved(items, view):
    cam = _Cam(view, items)
    zp = SoftwareRenderer._draw(cam, items)
    return cam, zp.resolve()


def test_sphere_area_and_depth_are_exact():
    cam, (pix, rgb, depth, owner) = resolved([spheres([0, 0, 0], 5.0)], vp())
    rp = 5.0 * cam.s  # 25 px
    assert len(pix) == pytest.approx(np.pi * rp * rp, rel=0.02)
    centre = (cam.H // 2) * cam.W + cam.W // 2
    k = np.flatnonzero(pix == centre)[0]
    assert depth[k] == pytest.approx(5.0, abs=0.02)  # front of the sphere, toward the viewer
    assert set(owner) == {0}


def test_stick_width_and_ends():
    st = Sticks(np.array([[-6.0, 0, 0]]), np.array([[6.0, 0, 0]]), np.array([1.0]), np.array([0]), np.array([-1]))
    cam, (pix, *_rest) = resolved([Colored(st, np.array([[0.2, 0.4, 0.8]]))], vp())
    py, px = pix // cam.W, pix % cam.W
    column = py[px == cam.W // 2]
    assert column.max() - column.min() + 1 == pytest.approx(2 * 1.0 * cam.s, abs=1)
    assert px.min() >= (4 - 0.1) * cam.s - 1 and px.max() <= (16 + 0.1) * cam.s + 1  # open, unextended ends


@pytest.mark.parametrize("seed", range(4))
def test_closed_mesh_has_no_pinholes(seed):
    rng = np.random.default_rng(seed)
    atoms = rng.normal(scale=2.5, size=(12, 3))
    verts, normals, tris, _near = gaussian_surface(atoms)
    m = Mesh(verts, normals, tris, np.zeros(len(verts), np.int64), np.zeros(len(verts), np.int8))
    R = euler_to_matrix(rng.uniform(-180, 180, 3))
    cam, (pix, *_rest) = resolved([Colored(m, np.full((len(verts), 3), 0.5))], vp(R, lim=12, size=160))
    from scipy import ndimage

    cover = np.zeros(cam.W * cam.H, bool)
    cover[pix] = True
    cover = cover.reshape(cam.H, cam.W)
    assert np.array_equal(ndimage.binary_fill_holes(cover), cover)


def test_primitive_order_does_not_change_the_image():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-6, 6, (40, 3))
    rgb = rng.uniform(0.2, 0.9, (40, 3))
    r = SoftwareRenderer()
    items = [Colored(Spheres(xyz, np.full(40, 1.6), np.arange(40), np.full(40, -1)), rgb)]
    perm = rng.permutation(40)
    shuffled = [Colored(Spheres(xyz[perm], np.full(40, 1.6), perm, np.full(40, -1)), rgb[perm])]
    a = r.render(items, vp(), PLAIN)
    b = r.render(shuffled, vp(), PLAIN)
    assert np.array_equal(a, b)


def test_geometry_beyond_the_panel_depth_range_keeps_its_order():
    # the panel's range comes from the trace atoms; atoms and radii may reach beyond it
    front = spheres([0, 0, 60.0], 3.0, (1, 0, 0))
    back = spheres([1.0, 0, 57.0], 3.0, (0, 0, 1))
    img = SoftwareRenderer().render([back, front], vp(zrange=(-10, 10)), PLAIN)
    cam = _Cam(vp(), [front])
    y, x = cam.H // 2, cam.W // 2
    assert img[y, x, 0] > 2 * img[y, x, 2]  # the front (red) sphere wins at the centre


def test_image_registers_with_the_axes_and_keeps_handedness():
    """A sphere at model (x, y) lands on the axes' data point (x, y) in the final figure
    (rows of the image run downward, PixelImage flips them for Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mdatk.render.molecule.stage import PixelImage

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    R = euler_to_matrix([0, 0, 90])  # rotation about z: model +x goes to view +y (right-handed)
    items = [spheres([4.0, 0.0, 0.0], 1.2, (1, 0, 0))]

    def render(w, h):
        return SoftwareRenderer().render(items, Viewport(R, ax.get_xlim(), ax.get_ylim(), w, h, (-10, 10)), PLAIN)

    img = PixelImage(ax, render)
    ax.add_artist(img)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    red = (buf[..., 1] < 120) & (buf[..., 2] < 120)  # the sphere, lit or not, and its outline
    ys, xs = np.nonzero(red)
    view = np.array([4.0, 0.0, 0.0]) @ R.T
    assert view[1] == pytest.approx(4.0)
    px, py = ax.transData.transform(view[:2])
    assert xs.mean() == pytest.approx(px, abs=1.5)
    assert ys.mean() == pytest.approx(buf.shape[0] - py, abs=1.5)  # display y is up, rows down
    plt.close(fig)


def test_fog_fades_distant_parts_toward_the_panel_colour():
    near, far = spheres([-5, 0, 8.0], 2.0, (0.9, 0.1, 0.1)), spheres([5, 0, -8.0], 2.0, (0.9, 0.1, 0.1))
    img = SoftwareRenderer().render([near, far], vp(), Shading(WHITE, 0.45))
    cam = _Cam(vp(), [near])
    y = cam.H // 2
    xn, xf = int(round(5 * cam.s)), int(round(15 * cam.s))
    assert img[y, xf, 1] > img[y, xn, 1] + 40  # the far sphere is paler (closer to white)


def test_glow_halo_only_with_glow():
    items = [spheres([0, 0, 0], 2.0, (0.3, 0.3, 0.8), owner=[0])]
    r = SoftwareRenderer()
    plain = r.render(items, vp(), Shading(WHITE, 0.0, np.zeros(1, np.float32)))
    glow = r.render(items, vp(), Shading(WHITE, 0.0, np.ones(1, np.float32), np.array([1.0, 0.7, 0.0])))
    cam = _Cam(vp(), items)
    y, x = cam.H // 2, int(round((10 + 3.2) * cam.s))  # beyond the sphere and its outline
    assert plain[y, x, 3] == 0
    assert glow[y, x, 3] > 60 and tuple(glow[y, x, :3]) == (255, 179, 0)
    c = cam.W // 2
    assert glow[c, c, 0] > plain[c, c, 0]  # tinted toward the glow colour


def test_translucent_representation_blends_over_the_opaque_one():
    opaque = spheres([0, 0, 0], 3.0, (1, 0, 0))
    glass = spheres([0, 0, 4.0], 3.0, (0, 0, 1))
    glass.opacity = 0.5
    img = SoftwareRenderer().render([opaque, glass], vp(), PLAIN)
    c = img.shape[0] // 2
    r, g, b, a = (int(v) for v in img[c, c])
    assert a == 255 and r > 40 and b > 40  # both show
    alone = SoftwareRenderer().render([glass], vp(), PLAIN)
    assert 100 < alone[c, c, 3] < 160  # straight alpha of the translucent surface


def test_fast_quality_returns_the_requested_size():
    img = SoftwareRenderer().render([spheres([0, 0, 0], 4.0)], Viewport(np.eye(3), (-10, 10), (-10, 10), 101, 77,
                                                                        (-10, 10)), PLAIN, quality="fast")
    assert img.shape == (77, 101, 4) and img[..., 3].any()


def test_threads_render_like_serial():
    rng = np.random.default_rng(5)
    scenes = [[spheres(rng.uniform(-6, 6, (30, 3)), 1.5, rng.uniform(0, 1, 3))] for _ in range(4)]
    r = SoftwareRenderer()
    serial = [r.render(s, vp(size=150), PLAIN) for s in scenes]
    out = [None] * 4

    def work(k):
        for _ in range(3):
            out[k] = r.render(scenes[k], vp(size=150 + 0), PLAIN)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(np.array_equal(a, b) for a, b in zip(serial, out))


def test_surface_of_one_atom_is_a_closed_outward_sphere():
    verts, normals, tris, near = gaussian_surface(np.zeros((1, 3)), spacing=0.5)
    radius = np.linalg.norm(verts, axis=1)
    expect = 1.4 * np.sqrt(2 * np.log(2))  # Gaussian sigma 1.4 at half height
    assert radius.mean() == pytest.approx(expect, rel=0.03)
    face = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    assert np.all((face * verts[tris].mean(1)).sum(1) > 0)
    vol = np.einsum("ij,ij->i", verts[tris[:, 0]], np.cross(verts[tris[:, 1]], verts[tris[:, 2]])).sum() / 6
    assert vol == pytest.approx(4 / 3 * np.pi * radius.mean() ** 3, rel=0.05)  # closed
    assert set(near) == {0}


def test_cartoon_shape_does_not_depend_on_the_selection(demo_dry):
    from mdatk.config import Project
    from mdatk.pipeline import Pipeline
    from mdatk.render.molecule.geometry import STYLES
    from mdatk.render.molecule.topology import Topology

    import tempfile

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = tempfile.mkdtemp(prefix="mdatk-cart-")
    ds = Pipeline(p).analyze(["trace", "atoms"])
    a, tr = ds.results["atoms"], ds.results["trace"]
    xyz = np.asarray(a.arrays["coords"][2], dtype=np.float64)
    topo = Topology(a.meta["atom_names"], a.meta["elements"], a.meta["resnames"], a.arrays["row"],
                    len(tr.arrays["resid"]), a.arrays["bonds"], xyz, tr.arrays["chain"], tr.arrays["resindex"])
    every = np.ones(topo.n_res, bool)
    one = np.zeros(topo.n_res, bool)
    one[5] = True
    full = STYLES["cartoon"](topo, every).build(xyz)
    single = STYLES["cartoon"](topo, one).build(xyz)
    back_full, back_one = full.meshes[0], single.meshes[0]
    assert set(back_one.owner) == {5} and len(back_one.tris)  # one residue still gets its piece
    ring = len(back_one.verts) - 2
    mine = np.flatnonzero(back_full.owner[:-2] == 5)
    # residue 5's stretch of the full cartoon matches the single-residue piece (the last ring of
    # the piece belongs to the next residue's stretch, joining the two without a gap)
    np.testing.assert_allclose(back_full.verts[mine], back_one.verts[:ring][:len(mine)], atol=1e-9)
    assert len(single.sticks[0].a) == 2  # its rung: backbone -> C1' -> glycosidic N


def test_build_scene_draws_other_residues_as_a_trace(demo_dry):
    from mdatk.config import Project, RepSpec
    from mdatk.pipeline import Pipeline
    from mdatk.render.molecule.scene import build_scene

    import tempfile

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = tempfile.mkdtemp(prefix="mdatk-scene-")
    ds = Pipeline(p).analyze(["trace", "atoms"])
    names = p.group_names()
    specs = [RepSpec("cartoon", [names[0]])]
    sc = build_scene(ds.results["atoms"], ds.results["trace"], specs, names)
    assert [s.style for s in sc.specs] == ["cartoon", "trace"]
    assert not (sc.styles[0].mask & sc.styles[1].mask).any() and (sc.styles[0].mask | sc.styles[1].mask).all()
    hidden = build_scene(ds.results["atoms"], ds.results["trace"], specs, names, rest="hide")
    assert [s.style for s in hidden.specs] == ["cartoon"] and len(specs) == 1


def _headless_env():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["MPLBACKEND"] = "Agg"
    return env


@pytest.mark.parametrize("renderer", ["auto", "vtk"])
def test_headless_render_without_a_display(demo_dry, tmp_path, renderer):
    """No display (cluster node): the built-in renderer draws the representations; asking for
    VTK there falls back to it instead of crashing or dropping to the 2D trace."""
    code = f"""
import sys
from mdatk.config import Project, RepSpec
from mdatk.pipeline import Pipeline
p = Project.load({demo_dry['preset']!r})
p.output.out_dir = {str(tmp_path)!r}
p.visual.representations = [RepSpec("cartoon")]
p.visual.renderer = {renderer!r}
pipe = Pipeline(p)
ds = pipe.analyze()
from mdatk.render.video import build_figure
fig, canvas, panels = build_figure(ds, p, 640, 360, 80)
st = [x for x in panels if getattr(x, "stage", None) is not None]
assert st, "no 3D stage"
st[0].draw_frame(3)
fig.canvas.draw()
assert st[0].stage.image.rgba is not None and st[0].stage.image.rgba[..., 3].any()
print("OK", type(st[0].stage.renderer).__name__, st[0].stage_note)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         env=_headless_env())
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK " in out.stdout
    if renderer == "auto":
        assert "SoftwareRenderer" in out.stdout


def test_render_video_with_representations(demo_dry, tmp_path):
    import glob

    from mdatk.config import Project, RepSpec
    from mdatk.pipeline import Pipeline

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = str(tmp_path)
    p.output.charts.enabled = False
    p.output.video.width, p.output.video.height, p.output.video.dpi = 640, 360, 80
    p.output.video.frame_stride = 8
    names = p.group_names()
    p.visual.representations = [RepSpec("cartoon", [names[0]], sugars=True),
                                RepSpec("surf", [names[1]], opacity=0.5)]
    pipe = Pipeline(p)
    ds = pipe.analyze()
    pipe.render_video(ds)
    assert glob.glob(str(tmp_path / "MDATK_video_*.mp4"))


def test_sphere_culling_leaves_the_image_unchanged(monkeypatch):
    import mdatk.render.molecule.software as SW

    rng = np.random.default_rng(11)
    xyz = rng.normal(scale=4.0, size=(400, 3))
    items = [Colored(Spheres(xyz, rng.choice([1.52, 1.7, 1.8], 400), np.arange(400), np.full(400, -1)),
                     rng.uniform(0.2, 0.9, (400, 3)))]
    r = SoftwareRenderer()
    for angles in ([0, 0, 0], [40, -70, 10], [120, 30, -60]):
        view = vp(euler_to_matrix(angles), lim=14, size=240)
        cam = _Cam(view, items)
        X, Y, Z = cam.screen(xyz)
        assert not SW._unoccluded(cam, X, Y, Z, items[0].prim.radius.astype(np.float32)).all()  # some culled
        culled = r.render(items, view, PLAIN)
        monkeypatch.setattr(SW, "CULL_MIN", 10 ** 9)
        assert np.array_equal(culled, r.render(items, view, PLAIN))
        monkeypatch.setattr(SW, "CULL_MIN", 64)
