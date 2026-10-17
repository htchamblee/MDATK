"""Renderer-independent geometry of the 3D representations."""

import numpy as np
import pytest

from mdatk.render.molecule.geometry import (STYLES, catmull_rom, frames_along, ring_slabs,
                                            sweep_faces)
from mdatk.render.molecule.topology import Topology, base_letter, normalise_name


@pytest.fixture(scope="module")
def demo_atoms(demo_dry):
    from mdatk.config import Project
    from mdatk.pipeline import Pipeline

    import tempfile

    p = Project.load(demo_dry["preset"])
    p.output.out_dir = tempfile.mkdtemp(prefix="mdatk-geom-")
    ds = Pipeline(p).analyze(["trace", "atoms"])
    a, tr = ds.results["atoms"], ds.results["trace"]
    xyz = np.asarray(a.arrays["coords"], dtype=np.float64)
    topo = Topology(a.meta["atom_names"], a.meta["elements"], a.meta["resnames"], a.arrays["row"],
                    len(tr.arrays["resid"]), a.arrays["bonds"], xyz[0], tr.arrays["chain"],
                    tr.arrays["resindex"])
    return topo, xyz


def test_names_and_letters():
    assert normalise_name("C1*") == "C1'" and normalise_name("O1P") == "OP1"
    assert [base_letter(r) for r in ("G", "RA5", "C3", "DT", "PSU", "U5", "MG", "LYS")] == \
        ["G", "A", "C", "T", "U", "U", "?", "?"]


def test_catmull_rom_interpolates_and_differentiates():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(6, 3))
    a, d, owner = catmull_rom(len(p), 8)
    s = a @ p
    np.testing.assert_allclose(s[::8], p, atol=1e-12)  # passes through the control points
    # derivative matches finite differences of a densely sampled spline
    a2, d2, _o = catmull_rom(len(p), 200)
    s2, ds2 = a2 @ p, d2 @ p
    fd = (s2[2:] - s2[:-2]) / (2 / 200)
    inner = np.ones(len(fd), bool)
    inner[199::200] = False  # skip the kinks at control points (C1 only)
    np.testing.assert_allclose(ds2[1:-1][inner], fd[inner], rtol=1e-3, atol=1e-3)
    assert owner[0] == 0 and owner[-1] == 5 and list(owner[8:16]) == [1] * 4 + [2] * 4


def test_sweep_faces_close_and_frames_are_orthonormal():
    tris = sweep_faces(5, 6)
    assert tris.max() == 5 * 6 + 1 and len(tris) == 2 * 4 * 6 + 2 * 6
    t = np.linspace(0, 4 * np.pi, 50)
    pos = np.stack([np.cos(t), np.sin(t), 0.2 * t], 1)
    tan = np.gradient(pos, axis=0)
    for side in (None, -pos * np.array([1, 1, 0])):
        tt, n, b = frames_along(pos, tan, side)
        for u, v in ((tt, n), (tt, b), (n, b)):
            np.testing.assert_allclose((u * v).sum(1), 0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-9)


def test_ring_slab_is_closed_with_outward_normals():
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(6)], 1)[None] * 1.4
    v, n, t, owner = ring_slabs(ring, 0.45)
    face = np.cross(v[t][:, 1] - v[t][:, 0], v[t][:, 2] - v[t][:, 0])
    assert np.all((face * n[t].sum(1)).sum(1) > 0)
    # closed surface: the signed volume equals area * thickness
    vol = np.einsum("ij,ij->i", v[t][:, 0], np.cross(v[t][:, 1], v[t][:, 2])).sum() / 6
    area = 3 * np.sqrt(3) / 2 * 1.4 ** 2
    assert vol == pytest.approx(area * 0.9, rel=1e-6)
    assert set(owner) == {0}


def test_topology_roles(demo_atoms):
    topo, _xyz = demo_atoms
    assert topo.n_res == 24 and [len(s) for s in topo.segments] == [12, 12]  # two strands
    assert len(topo.purine_res) + len(topo.pyrimidine_res) == 24 and len(topo.sugar_res) == 24
    for rows, ring, gly in ((topo.purine_res, topo.purine_ring, topo.purine_gly),
                            (topo.pyrimidine_res, topo.pyrimidine_ring, topo.pyrimidine_gly)):
        for r, ids, g in zip(rows, ring, gly):
            assert all(topo.row[i] == r for i in ids) and g in ids
            assert topo.names[g] == ("N9" if len(ids) == 9 else "N1")
    assert all(topo.names[c] == "P" for c in topo.control[topo.control >= 0])


@pytest.mark.parametrize("style", sorted(STYLES))
def test_styles_build_outward_meshes_for_selected_residues(demo_atoms, style):
    topo, xyz = demo_atoms
    mask = np.zeros(topo.n_res, bool)
    mask[:12] = True  # first strand only
    dl = STYLES[style](topo, mask, size=1.0, sugars=True).build(xyz[3])
    owners = [m.owner for m in dl.meshes] + [s.owner for s in dl.spheres] + [s.owner for s in dl.sticks]
    assert owners and all(len(o) == 0 or (o.max() < 12 and o.min() >= 0) for o in owners)
    for m in dl.meshes:
        face = np.cross(m.verts[m.tris][:, 1] - m.verts[m.tris][:, 0], m.verts[m.tris][:, 2] - m.verts[m.tris][:, 0])
        big = np.linalg.norm(face, axis=1) > 1e-9
        assert np.all((face * m.normals[m.tris].sum(1)).sum(1)[big] > 0)
    if style == "cartoon":
        assert any((m.role == 1).all() for m in dl.meshes)  # base slabs
        assert dl.sticks and len(dl.sticks[0].a) >= 12
    if style in ("vdw", "licorice"):
        n_atoms = int((topo.row < 12).sum())
        assert len(dl.spheres[0].xyz) == n_atoms


def test_banded_spline_matches_dense():
    from mdatk.render.molecule.geometry import catmull_rom_banded, extend_controls

    p = np.random.default_rng(4).normal(size=(7, 3))
    a, d, owner = catmull_rom(7, 8)
    idx, w, dw, owner2 = catmull_rom_banded(7, 8)
    ext = extend_controls(p)[idx]
    np.testing.assert_allclose(np.einsum("sk,skc->sc", w, ext), a @ p, atol=1e-12)
    np.testing.assert_allclose(np.einsum("sk,skc->sc", dw, ext), d @ p, atol=1e-12)
    assert np.array_equal(owner, owner2)


def test_charmm_names_and_bases_told_by_their_atoms():
    from mdatk.render.molecule.topology import PURINE_RING, PYRIMIDINE_RING

    assert [base_letter(r) for r in ("ADE", "GUA", "CYT", "URA", "THY", "URI")] == ["A", "G", "C", "U", "T", "U"]
    names, res = [], []
    for r, extra in enumerate([("O6",), ("N6",), ("N4",), ("O4",), ("O4", "C7")]):
        ring = PURINE_RING if r < 2 else PYRIMIDINE_RING
        for n in ring + extra:
            names.append(n)
            res.append(r)
    n = len(names)
    topo = Topology(names, ["C"] * n, ["XXX"] * n, np.array(res), 5, np.zeros((0, 2)), np.zeros((n, 3)),
                    np.zeros(5, int), np.arange(5))
    assert topo.letter == ["G", "A", "C", "U", "T"]


def test_undrawable_residues_fall_back(demo_atoms):
    """Residues no representation actually draws get the trace; residues without a backbone
    atom (ligands, which backbone styles cannot draw) get licorice."""
    from mdatk.config import RepSpec
    from mdatk.render.molecule.scene import MoleculeScene

    topo, xyz = demo_atoms
    mask = np.zeros(topo.n_res, bool)
    mask[[0, 2, 3]] = True  # residue 0: its stretch exists; all three are drawn
    sc = MoleculeScene(topo, [RepSpec("tube")], [mask])
    assert sc.drawn()[[0, 2, 3]].all() and not sc.drawn()[1]
    sc.fill_rest()
    assert [s.style for s in sc.specs] == ["tube", "trace"] and sc.styles[1].mask[1] and not sc.styles[1].mask[0]
    saved = topo.control
    topo.control = topo.control.copy()
    topo.control[5] = -1  # pretend residue 5 has no backbone atom
    try:
        sc = MoleculeScene(topo, [RepSpec("cartoon")], [np.arange(topo.n_res) < 3])
        sc.fill_rest()
        lic = [st for sp, st in zip(sc.specs, sc.styles) if sp.style == "licorice"]
        assert lic and lic[0].mask[5] and lic[0].mask.sum() == 1
    finally:
        topo.control = saved


def test_protein_segments_get_a_round_tube():
    from mdatk.render.molecule.geometry import Backbone

    class T:
        n_res = 6
        segments = [np.arange(6)]
        nucleic = np.zeros(0, np.int64)
        control = np.arange(6)
        side = np.arange(6) + 6

    xyz = np.zeros((12, 3))
    xyz[:6, 0] = np.arange(6) * 3.8
    xyz[6:, 1] = np.array([1, -1, 1, -1, 1, -1]) * 1.5  # side atoms alternate (CB on a strand)
    xyz[6:, 0] = xyz[:6, 0]
    meshes, _ = Backbone(T(), np.ones(6, bool), "ellipse", 1.6, 0.45, round_radius=0.6).build(xyz)
    v = meshes[0].verts[:-2].reshape(-1, 12, 3)
    centre = v.mean(1, keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(v - centre, axis=2), 0.6, atol=1e-6)


def test_cartoon_rungs_are_closed_by_balls(demo_atoms):
    topo, xyz = demo_atoms
    mask = np.zeros(topo.n_res, bool)
    mask[:4] = True
    for sugars, per_res in ((False, 2), (True, 3)):
        dl = STYLES["cartoon"](topo, mask, sugars=sugars).build(xyz[0])
        balls = dl.spheres[0].xyz
        ends = np.concatenate([dl.sticks[0].a, dl.sticks[0].b])
        assert len(balls) == 4 * per_res
        # every stick end except the backbone anchor sits in a ball
        d = np.linalg.norm(ends[:, None] - balls[None], axis=2).min(1)
        assert (d < 1e-9).sum() >= len(ends) - 4
