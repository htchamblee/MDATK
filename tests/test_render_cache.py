import json

import numpy as np
import pytest

from mdatk.data import AnalysisResult, Dataset, CSRBuilder
from mdatk.io.cache import CacheStore
from mdatk.io.external import load_external, read_table
from mdatk.config import ExternalSeries
from mdatk.render.coloring import glow_intensity, static_colors
from mdatk.render.layout import PRESETS, n_slots
from mdatk.render.orientation import drag, euler_to_matrix, matrix_to_euler


def test_glow_modes():
    d = np.array([[9, 9, 2, 2, 9, 9, 9, 9, 9, 9],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    member = np.array([False, True])
    cur = glow_intensity(d, member, 4.0, "current")
    np.testing.assert_array_equal(cur[0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    assert not cur[1].any()  # members of the proximity set never glow
    np.testing.assert_array_equal(glow_intensity(d, member, 4.0, "ever")[0], np.ones(10))
    fade = glow_intensity(d, member, 4.0, "fade", fade_frames=3)
    np.testing.assert_allclose(fade[0], [0, 0, 1, 1, 0.75, 0.5, 0.25, 0, 0, 0])
    assert not glow_intensity(d, member, 4.0, "none").any()
    with pytest.raises(ValueError):
        glow_intensity(d, member, 4.0, "disco")


def test_static_colors_modes():
    chain = np.array([0, 0, 1])
    gi = np.array([0, -1, 1])
    groups = static_colors("groups", chain, gi, ["#ff0000", "#0000ff"], "#808080", "tab10")
    np.testing.assert_allclose(groups[0, :3], [1, 0, 0])
    np.testing.assert_allclose(groups[1, :3], [128 / 255] * 3, atol=1e-3)
    chains = static_colors("chain", chain, gi, [], "#808080", "tab10")
    assert not np.allclose(chains[0], chains[2])


def test_euler_roundtrip_and_drag():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ang = rng.uniform(-179, 179, 3)
        ang[1] = rng.uniform(-85, 85)
        np.testing.assert_allclose(matrix_to_euler(euler_to_matrix(ang)), ang, atol=1e-6)
    new = drag([10, 20, 30], 40, -15)
    r = euler_to_matrix(new)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert drag([0, 0, 0], 0, 0) == pytest.approx((0, 0, 0))


@pytest.mark.parametrize("name", list(PRESETS))
def test_layout_slots_tile_without_overlap(name):
    rows, cols, slots = PRESETS[name]
    used = np.zeros((rows, cols), int)
    for r0, r1, c0, c1 in slots:
        assert 0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols
        used[r0:r1, c0:c1] += 1
    assert used.max() == 1
    assert n_slots(name) == len(slots)
    if name.startswith("equal_"):
        panels = slots[:-1] if name.endswith("_strip") else slots
        sizes = {(r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in panels}
        assert len(sizes) == 1  # equal-sized panels (besides the strip)


def test_cache_roundtrip_and_manifest(tmp_path):
    store = CacheStore(str(tmp_path / "cache"))
    res = AnalysisResult("demo", "abc123", {"cutoff": 4.5}, {"v": np.arange(5.0)})
    res.add_series("value", "v", "Value", "count")
    b = CSRBuilder()
    b.add([0, 1], [1, 0], [2, 3])
    b.add([], [], [])
    res.arrays.update(b.arrays("m"))
    res.meta.update(rows=["A1", "A2"], cols=["B1", "B2"])
    res.add_pairmap("map", "m", "rows", "cols", "Map")
    frames, time = np.arange(5), np.arange(5) * 10.0
    store.save_result(res, frames, time)
    loaded, fr, tm = store.load_result("demo", "abc123")
    np.testing.assert_array_equal(loaded.arrays["v"], np.arange(5.0))
    ds = Dataset(fr, tm, {"demo": loaded}, time_unit="ns")
    np.testing.assert_allclose(ds.time(), time / 1000)
    assert ds.series("demo/value").label == "Value"
    pm = ds.pairmap("demo/map")
    np.testing.assert_array_equal(pm.dense(0), [[0, 2], [3, 0]])
    assert pm.dense(1).sum() == 0
    store.write_manifest(system={}, frame_selection={"start": 0}, dataset=ds, groups=[],
                         entries={"demo": "abc123"})
    manifest = json.loads((tmp_path / "cache" / "MDATK_cache.json").read_text())
    assert manifest["available"]["series"] == ["demo/value"]
    assert store.load_dataset_from_manifest().series_keys() == ["demo/value"]
    # a corrupt entry is ignored, not fatal
    (tmp_path / "cache" / "demo-abc123.npz").write_bytes(b"garbage")
    assert store.load_result("demo", "abc123") is None


def test_external_table_and_interpolation(tmp_path):
    dat = tmp_path / "rmsd.dat"
    dat.write_text("#Frame  rmsd  other\n1 0.0 5\n2 1.0 6\n3 2.0 7\n")
    header, data = read_table(str(dat))
    assert header == ["Frame", "rmsd", "other"] and data.shape == (3, 3)
    ds = Dataset(np.array([0, 1, 2]), np.array([0.0, 10.0, 20.0]))
    res = load_external([ExternalSeries(path=str(dat), label="cpp", x_column="Frame",
                                        x_unit="frame1", columns=["rmsd"])], ds)
    ds.add(res)
    np.testing.assert_allclose(ds.series("external/cpp.rmsd").values, [0, 1, 2])
    csv = tmp_path / "e.csv"
    csv.write_text("time_ns,energy\n0,1\n0.02,3\n")
    res = load_external([ExternalSeries(path=str(csv), x_unit="ns")], ds)
    np.testing.assert_allclose(res.arrays[next(iter(res.arrays))], [1, 2, 3])
