"""Density maps: MRC I/O, simulated maps, rigid-body fitting and the projected shadow."""

import json
import math
import os

import numpy as np
import pytest

from mdatk import density as D
from mdatk.config import Project
from mdatk.io.mrc import DensityMap, MapError, read_map, write_map
from mdatk.render.orientation import euler_to_matrix


def _raw_map(path, data_crs, mapcrs, voxel=(1.0, 1.0, 1.0), origin=(0, 0, 0), nstart=(0, 0, 0),
             endian="<", mode=2, nsymbt=0):
    """Write an MRC file by hand: ``data_crs`` is indexed [section, row, column]."""
    ns, nr, nc = data_crs.shape
    ints = np.zeros(256, dtype=f"{endian}i4")
    fl = ints.view(f"{endian}f4")
    ints[0:4] = (nc, nr, ns, mode)
    ints[4:7] = nstart
    n_cart = [0, 0, 0]
    for k, m in enumerate(mapcrs):
        n_cart[m - 1] = (nc, nr, ns)[k]
    ints[7:10] = n_cart
    fl[10:13] = np.array(n_cart) * np.array(voxel)
    fl[13:16] = 90.0
    ints[16:19] = mapcrs
    ints[23] = nsymbt
    ints[27] = 20140
    fl[49:52] = origin
    raw = bytearray(ints.tobytes())
    raw[208:212] = b"MAP "
    raw[212:214] = b"\x44\x44" if endian == "<" else b"\x11\x11"
    dtype = {2: "f4", 1: "i2", 0: "i1", 6: "u2"}[mode]
    with open(path, "wb") as fh:
        fh.write(bytes(raw))
        fh.write(b"\0" * nsymbt)
        fh.write(np.ascontiguousarray(data_crs, dtype=f"{endian}{dtype}").tobytes())


def _blob(shape_zyx, centre_xyz, width=1.5):
    z, y, x = np.indices(shape_zyx)
    cx, cy, cz = centre_xyz
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * width ** 2)).astype(np.float32)


@pytest.mark.parametrize("mapcrs", [(1, 2, 3), (3, 1, 2), (2, 3, 1), (3, 2, 1)])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_read_map_axis_orders_and_byte_order(tmp_path, mapcrs, endian):
    cart = _blob((14, 12, 10), (3, 7, 9))  # [z, y, x]: blob at x=3, y=7, z=9
    # file axes (column, row, section) hold Cartesian axes mapcrs - 1
    axes_zyx = {0: 2, 1: 1, 2: 0}  # Cartesian axis -> axis of the [z, y, x] array
    file_arr = np.transpose(cart, [axes_zyx[mapcrs[2] - 1], axes_zyx[mapcrs[1] - 1], axes_zyx[mapcrs[0] - 1]])
    path = tmp_path / "m.mrc"
    _raw_map(path, file_arr, mapcrs, voxel=(0.8, 1.1, 1.3), nstart=(0, 0, 0), endian=endian,
             origin=(10.0, -5.0, 2.0), nsymbt=64)
    m = read_map(str(path))
    assert m.data.shape == (14, 12, 10)
    np.testing.assert_allclose(m.data, cart, rtol=1e-6)
    peak = np.unravel_index(np.argmax(m.data), m.data.shape)
    xyz = m.index_to_xyz(np.array(peak[::-1], float))
    np.testing.assert_allclose(xyz, [10 + 3 * 0.8, -5 + 7 * 1.1, 2 + 9 * 1.3], atol=1e-5)


def test_read_map_nstart_origin_modes_and_errors(tmp_path):
    cart = (_blob((8, 8, 8), (4, 4, 4)) * 100).astype(np.int16)
    path = tmp_path / "s.map"
    _raw_map(path, cart, (1, 2, 3), voxel=(2.0, 2.0, 2.0), nstart=(-4, 2, 10), mode=1)
    m = read_map(str(path))
    np.testing.assert_allclose(m.origin, [-8.0, 4.0, 20.0])  # NSTART x voxel when ORIGIN is zero
    assert m.data.dtype == np.float32 and m.data.max() == 100
    bad = tmp_path / "bad.mrc"
    bad.write_bytes(b"\0" * 100)
    with pytest.raises(MapError):
        read_map(str(bad))
    trunc = tmp_path / "trunc.mrc"
    trunc.write_bytes(path.read_bytes()[:1100])
    with pytest.raises(MapError, match="shorter"):
        read_map(str(trunc))


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    m = DensityMap(rng.normal(size=(5, 6, 7)).astype(np.float32), np.array([1.5, -2.0, 3.25]),
                   np.array([1.1, 1.1, 1.1]))
    write_map(str(tmp_path / "r.mrc"), m)
    back = read_map(str(tmp_path / "r.mrc"))
    np.testing.assert_array_equal(back.data, m.data)
    np.testing.assert_allclose(back.origin, m.origin)
    np.testing.assert_allclose(back.voxel, m.voxel, rtol=1e-6)


def test_simulated_map_peak_and_width():
    xyz = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    m = D.simulate_map(xyz, np.array([6.0, 6.0]), resolution=4.0, spacing=0.5)
    sigma = D.SIGMA_FACTOR * 4.0
    for atom in xyz:
        idx = np.round(m.xyz_to_index(atom)).astype(int)
        local = m.data[idx[2] - 3:idx[2] + 4, idx[1] - 3:idx[1] + 4, idx[0] - 3:idx[0] + 4]
        assert local.max() == m.data[idx[2], idx[1], idx[0]]
    # Gaussian width along x through the first atom
    i0 = np.round(m.xyz_to_index(xyz[0])).astype(int)
    line = m.data[i0[2], i0[1], :]
    xs = m.origin[0] + m.voxel[0] * np.arange(len(line))
    near = np.abs(xs) < 4  # amplitude = weight (atomic number), centred on the atom
    expected = 6.0 * np.exp(-xs[near] ** 2 / (2 * sigma ** 2))
    np.testing.assert_allclose(line[near], expected, rtol=1e-4, atol=1e-5)


def test_rotation_samples_cover_so3():
    for step in (20.0, 30.0):
        rots = D.rotation_samples(step)
        assert np.allclose(np.einsum("nij,nkj->nik", rots, rots), np.eye(3), atol=1e-9)
        assert np.allclose(np.linalg.det(rots), 1.0)
        rng = np.random.default_rng(1)
        worst = 0.0
        for q in rng.normal(size=(200, 4)):
            r = D.quaternion_matrix(q)
            c = (np.einsum("nij,ij->n", rots, r) - 1) / 2
            worst = max(worst, float(np.degrees(np.arccos(np.clip(c.max(), -1, 1)))))
        assert worst < 1.05 * step, (step, worst)


def test_rigid_transform_algebra():
    t = D.RigidTransform(D.rotvec_matrix([0.3, -0.2, 0.5]), np.array([1.0, 2.0, 3.0]))
    p = np.random.default_rng(2).normal(size=(5, 3))
    np.testing.assert_allclose(t.inverse().apply(t.apply(p)), p, atol=1e-12)
    back = D.RigidTransform.from_list(t.to_list(digits=12))
    np.testing.assert_allclose(back.apply(p), t.apply(p), atol=1e-9)
    with pytest.raises(ValueError):
        D.RigidTransform.from_list([1, 2, 3])


@pytest.fixture(scope="module")
def model():
    """Demo-like RNA fragment: the synthetic system's first-frame heavy atoms."""
    from mdatk.demo import build_demo
    from mdatk.pipeline import Pipeline
    import tempfile

    d = tempfile.mkdtemp(prefix="mdatk-dens-")
    demo = build_demo(d, n_frames=3, solvent=False)
    p = Project.load(demo["preset"])
    p.output.out_dir = os.path.join(d, "out")
    xyz, elements, _f, _ref = Pipeline(p).density_frame_atoms("ref")
    return {"xyz": xyz, "elements": elements, "preset": demo["preset"], "dir": d}


def _target_map(xyz, elements, true, res=6.0, pad=15, noise=0.1, seed=3):
    sim = D.simulate_map(true.apply(xyz), D.atom_weights(elements), res, spacing=1.5)
    big = np.zeros(tuple(np.array(sim.data.shape) + 2 * pad), np.float32)
    big[pad:-pad, pad:-pad, pad:-pad] = sim.data
    big += np.random.default_rng(seed).normal(scale=noise * sim.data.max(), size=big.shape).astype(np.float32)
    return DensityMap(big, sim.origin - pad * 1.5, sim.voxel)


def test_global_search_recovers_the_pose(model):
    xyz, el = model["xyz"], model["elements"]
    true = D.RigidTransform(D.quaternion_matrix([0.3, -0.8, 0.4, 0.2]), np.array([25.0, -40.0, 12.0]))
    emap = _target_map(xyz, el, true)
    res = D.fit_model(emap, xyz, el, 6.0, search="global", angular_step=30)
    c = xyz.mean(axis=0)
    assert res.cc > 0.9
    assert D.rotation_angle(res.transform.rotation @ true.rotation.T) < 3.0
    assert np.linalg.norm(res.transform.apply(c) - true.apply(c)) < 1.0
    # local refinement from a nearby placement converges to the same pose
    near = D.RigidTransform(D.rotvec_matrix([0.15, -0.1, 0.1]), np.array([2.0, -1.5, 1.0])).about(
        true.apply(c)).compose(true)
    loc = D.fit_model(emap, xyz, el, 6.0, start=near, search="manual")
    assert loc.start == "manual" and loc.cc > 0.9
    assert D.rotation_angle(loc.transform.rotation @ true.rotation.T) < 3.0


def test_projected_shadow_lies_under_the_model(model):
    xyz, el = model["xyz"], model["elements"]
    true = D.RigidTransform(D.quaternion_matrix([0.9, 0.1, 0.3, -0.2]), np.array([5.0, 8.0, -3.0]))
    emap = _target_map(xyz, el, true, noise=0.0)
    centre = xyz.mean(axis=0)
    w = D.atom_weights(el)
    for angles in ([0, 0, 0], [40, -30, 10], [90, 0, 45]):
        v = euler_to_matrix(angles)
        sh = D.project_density(emap, true, v, centre, level=0.02 * float(emap.data.max()))
        img = sh.image.astype(np.float64)
        rows, cols = img.shape
        u = np.linspace(sh.extent[0], sh.extent[1], cols + 1)[:-1] + (sh.extent[1] - sh.extent[0]) / cols / 2
        vv = np.linspace(sh.extent[2], sh.extent[3], rows + 1)[:-1] + (sh.extent[3] - sh.extent[2]) / rows / 2
        tot = img.sum()
        cu, cv = (img.sum(axis=0) * u).sum() / tot, (img.sum(axis=1) * vv).sum() / tot
        proj = (xyz - centre) @ v.T
        mu = (proj[:, :2] * w[:, None]).sum(axis=0) / w.sum()
        assert math.hypot(cu - mu[0], cv - mu[1]) < 1.0, angles
    rgba = D.shadow_rgba(sh, (0.1, 0.2, 0.9), 0.5)
    assert rgba.shape == sh.image.shape + (4,) and 0 < rgba[..., 3].max() <= 0.5


def test_config_density_validation(tmp_path):
    p = Project()
    assert p.density_problems() == [] and not p.density_active()
    p.density.map_path = str(tmp_path / "missing.mrc")
    p.density.fit_frame = "first"
    p.density.transform = [1.0, 2.0]
    probs = " | ".join(p.density_problems())
    assert "not found" in probs and "'ref' or a frame number" in probs and "12 numbers" in probs
    data = p.to_dict()
    assert Project.from_dict(data).density == p.density


def test_pipeline_fit_and_video_with_density(model, tmp_path):
    from mdatk.config import PanelSpec
    from mdatk.pipeline import Pipeline

    xyz, el = model["xyz"], model["elements"]
    true = D.RigidTransform(D.quaternion_matrix([0.5, 0.5, -0.5, 0.5]), np.array([-10.0, 30.0, 4.0]))
    path = str(tmp_path / "target.mrc")
    write_map(path, _target_map(xyz, el, true))
    p = Project.load(model["preset"])
    p.output.out_dir = str(tmp_path / "out")
    p.output.video.width, p.output.video.height, p.output.video.dpi = 320, 180, 40
    p.output.charts.enabled = False
    p.layout.preset = "single"
    p.layout.panels = [PanelSpec("structure", [], "Structure")]
    p.density.map_path = path
    p.density.resolution = 6.0
    out = Pipeline(p).run()
    assert os.path.exists(out.video)
    assert len(p.density.transform) == 12 and p.density.fit_cc > 0.9
    assert "global search" in p.density.fit_info
    # the panel draws the shadow on the real data
    from mdatk.render.video import build_figure
    from mdatk.render.panels import StructurePanel
    _fig, _canvas, panels = build_figure(out.dataset, p, 320, 180, dpi=40)
    st = [x for x in panels if isinstance(x, StructurePanel)][0]
    assert st.has_density() and st.density_img.get_visible()
    before = st.density_img.get_array().copy()
    st.set_rotation([60, 20, 0])
    assert not np.array_equal(before, st.density_img.get_array())
    moved = st.nudge_density(shift=(3.0, 0.0))
    t0, t1 = D.RigidTransform.from_list(p.density.transform), D.RigidTransform.from_list(moved)
    c = xyz.mean(axis=0)
    shift_world = t0.apply(c) - t1.apply(c)  # the map moved by +3 Å along screen x
    np.testing.assert_allclose(st.view_matrix @ t1.rotation.T @ shift_world, [3.0, 0.0, 0.0], atol=1e-6)


def test_cli_density(model, tmp_path, capsys):
    from mdatk.cli import main

    xyz, el = model["xyz"], model["elements"]
    true = D.RigidTransform(D.quaternion_matrix([0.2, 0.9, 0.1, -0.3]), np.array([3.0, 4.0, 5.0]))
    path = str(tmp_path / "t.mrc")
    write_map(path, _target_map(xyz, el, true))
    assert main(["density", "info", path]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["voxel_A"] == [1.5, 1.5, 1.5] and info["grid_xyz"][0] > 10
    saved = str(tmp_path / "fitted.json")
    assert main(["density", "fit", "-p", model["preset"], "--map", path, "--resolution", "6", "-q",
                 "-o", str(tmp_path / "out"), "--save", saved]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["cc"] > 0.9 and len(res["transform"]) == 12 and os.path.exists(saved)
    assert Project.load(saved).density.transform == res["transform"]
    sim = str(tmp_path / "sim.mrc")
    assert main(["density", "simulate", "-p", saved, "--resolution", "6", "--placed", "-O", sim, "-q"]) == 0
    capsys.readouterr()
    assert read_map(sim).data.max() > 0
    assert main(["density", "fit", "-p", saved, "--search", "manual", "-q"]) == 0
    assert json.loads(capsys.readouterr().out)["cc"] > 0.9


def test_gzip_imod_bytes_and_nstart(tmp_path):
    import gzip

    cart = _blob((6, 6, 6), (2, 3, 4))
    path = tmp_path / "g.mrc"
    write_map(str(path), DensityMap(cart, np.array([-3.0, 1.5, 6.0]), np.array([1.5, 1.5, 1.5])))
    head = np.frombuffer(path.read_bytes()[:64], dtype="<i4")
    assert list(head[4:7]) == [-2, 1, 4]  # NSTART written when the origin lies on whole voxels
    gz = tmp_path / "g.map.gz"
    gz.write_bytes(gzip.compress(path.read_bytes()))
    m = read_map(str(gz))
    np.testing.assert_array_equal(m.data, cart)
    np.testing.assert_allclose(m.origin, [-3.0, 1.5, 6.0])
    # IMOD mode 0 maps hold unsigned bytes
    raw = tmp_path / "imod.mrc"
    vals = np.full((2, 2, 2), 200, dtype=np.uint8)
    _raw_map(raw, vals.view(np.int8), (1, 2, 3), mode=0)
    b = bytearray(raw.read_bytes())
    b[152:160] = np.array([1146047817, 0], dtype="<i4").tobytes()
    raw.write_bytes(bytes(b))
    assert read_map(str(raw)).data.min() == 200
    sim = D.simulate_map(np.array([[0.3, 0.7, -1.1]]), np.array([6.0]), 3.0)
    assert np.allclose(sim.origin / sim.voxel, np.round(sim.origin / sim.voxel))


def test_global_search_ignores_unrelated_density_and_validates_steps(model):
    """A protein-like blob next to the RNA must not attract the search (locally normalised
    correlation); bad angular steps are rejected; tiny maps do not crash."""
    xyz, el = model["xyz"], model["elements"]
    rng = np.random.default_rng(5)
    true = D.RigidTransform(D.quaternion_matrix([0.6, -0.2, 0.7, 0.3]), np.array([60.0, 50.0, 55.0]))
    placed = true.apply(xyz)
    blob = rng.normal(size=(2500, 3))
    blob = blob / np.linalg.norm(blob, axis=1)[:, None] * 22.0 * rng.uniform(0, 1, (2500, 1)) ** (1 / 3)
    blob += placed.mean(axis=0) + np.array([42.0, 0.0, 0.0])
    pts = np.concatenate([placed, blob])
    w = np.concatenate([D.atom_weights(el), np.full(len(blob), 6.7)])
    sim = D.simulate_map(pts, w, 8.0, spacing=1.6)
    data = sim.data + np.random.default_rng(1).normal(scale=0.05 * sim.data.max(), size=sim.data.shape)
    emap = DensityMap(data.astype(np.float32), sim.origin, sim.voxel)
    res = D.fit_model(emap, xyz, el, 8.0, search="global", angular_step=30)
    c = xyz.mean(axis=0)
    assert np.linalg.norm(res.transform.apply(c) - true.apply(c)) < 1.5
    assert D.rotation_angle(res.transform.rotation @ true.rotation.T) < 5.0
    with pytest.raises(D.FitError):
        D.rotation_samples(0)
    p = Project()
    p.density.map_path, p.density.angular_step = __file__, 90.0
    assert any("angular step" in m for m in p.density_problems())
    tiny = DensityMap(np.random.default_rng(0).random((2, 30, 30)).astype(np.float32), np.zeros(3), np.ones(3))
    try:
        D.fit_model(tiny, xyz, el, 20.0, search="global")
    except D.FitError:
        pass


def test_placement_follows_the_molecule_and_failures_degrade(model, tmp_path):
    from mdatk.config import PanelSpec
    from mdatk.pipeline import Pipeline
    from mdatk.render.panels import StructurePanel
    from mdatk.render.video import build_figure

    xyz, el = model["xyz"], model["elements"]
    true = D.RigidTransform(D.quaternion_matrix([0.4, 0.3, -0.8, 0.2]), np.array([20.0, -15.0, 8.0]))
    path = str(tmp_path / "t.mrc")
    write_map(path, _target_map(xyz, el, true))
    p = Project.load(model["preset"])
    p.output.out_dir = str(tmp_path / "out")
    p.layout.preset = "large_top_3"
    p.layout.panels = [PanelSpec("structure", [], "S"), PanelSpec("timeseries", ["rmsd/Complex"], ""),
                       PanelSpec("timeseries", ["rmsd/Complex"], ""), PanelSpec("timeseries", ["rmsd/Complex"], "")]
    p.density.map_path, p.density.resolution = path, 6.0
    Pipeline(p).fit_density()
    assert len(p.density.anchor) > 3 and p.density.fit_cc > 0.9
    # a later start frame is a different reference: the anchored placement still fits the model there
    p.inputs.start = 2
    pipe = Pipeline(p)
    xyz2, el2, _f, ref2 = pipe.density_frame_atoms("ref")
    prob = D.FitProblem(read_map(path), xyz2, D.atom_weights(el2), 6.0)
    raw = D.RigidTransform.from_list(p.density.transform)
    moved = D.anchored(p.density.transform, p.density.anchor, ref2)
    assert prob.cc(moved) > 0.85 and prob.cc(moved) > prob.cc(raw) + 0.05
    ds = pipe.analyze()
    _fig, _c, panels = build_figure(ds, p, 1920, 1080, dpi=60)
    st = [x for x in panels if isinstance(x, StructurePanel)][0]
    np.testing.assert_allclose(st._to_map.to_list(), moved.to_list(), atol=1e-4)
    # the shadow window follows the axes limits after layout (wide 16:9 panel), not a fixed square
    st.after_layout()
    (x0, x1), (y0, y1) = st.ax.get_xlim(), st.ax.get_ylim()
    assert (x1 - x0) > 1.5 * (y1 - y0)
    ext = st.density_img.get_extent()
    pad = 0.15 * (x1 - x0)
    assert x0 - pad <= ext[0] < ext[1] <= x1 + pad
    # a hidden missing map does not block runs; a failed automatic fit does not stop the video
    p.density.show = False
    p.density.map_path = str(tmp_path / "gone.mrc")
    assert p.density_problems() == []
    bad = tmp_path / "bad.mrc"
    bad.write_bytes(b"\0" * 2000)
    p.density.show, p.density.map_path, p.density.transform, p.density.anchor = True, str(bad), [], []
    p.output.video.width, p.output.video.height, p.output.video.dpi = 320, 180, 40
    p.output.charts.enabled = False
    out = Pipeline(p).run()
    assert os.path.exists(out.video) and p.density.transform == []
    p.density.transform = [1, 0, 0, 0, 1, 0, 0, 0, 1, float("inf"), 0, 0]
    assert any("invalid" in m for m in p.density_problems())
