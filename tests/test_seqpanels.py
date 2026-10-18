import warnings

import numpy as np
import pytest

from mdatk.config import PanelSpec, Project
from mdatk.render import layout as layouts
from mdatk.render.example import example_dataset
from mdatk.render.residues import one_letter, residue_rows
from mdatk.render.video import build_figure
from mdatk.selections import SelectionError

warnings.filterwarnings("ignore")


def figure(project, size=(1600, 900)):
    ds = example_dataset(project)
    return ds, build_figure(ds, project, size[0], size[1], 100)[2]


def project_with(panels, preset):
    p = Project()
    p.layout.preset = preset
    p.layout.panels = panels
    return p


def test_one_letter_codes():
    assert [one_letter(n) for n in ("G5", "RA", "LG3", "0C", "U", "DT", "HIS", "XYZ")] == \
        ["G", "A", "G", "C", "U", "T", "H", "X"]


def test_residue_rows_by_chain_and_number():
    p = project_with([PanelSpec("sequence", [], "")], "single")
    tr = example_dataset(p).results["trace"]
    rows = residue_rows(tr, "B:13-15")
    assert list(np.asarray(tr.arrays["resid"])[rows]) == [13, 14, 15]
    assert len(residue_rows(tr, "")) == len(tr.arrays["chain"])
    with pytest.raises(SelectionError):
        residue_rows(tr, "C:1")


@pytest.mark.parametrize("preset", [k for k in layouts.PRESETS if k.endswith("_strip")])
def test_strip_layouts_put_the_last_slot_across_the_bottom(preset):
    rows, cols, slots = layouts.preset(preset)
    ratios = layouts.height_ratios(preset)
    assert len(ratios) == rows and slots[-1] == (rows - 1, rows, 0, cols) and ratios[-1] < 1.0
    specs = [PanelSpec("structure", [], "")] + [PanelSpec("timeseries", ["rmsd/Complex"], "")] * (len(slots) - 2)
    specs.append(PanelSpec("sequence", [], ""))
    _ds, panels = figure(project_with(specs, preset))
    assert all(pn.ok for pn in panels)
    strip, big = panels[-1].ax.get_position(), panels[0].ax.get_position()
    assert strip.height < big.height and strip.width > 0.8


def test_sequence_panel_wraps_and_warns(caplog):
    from mdatk.render import residues
    residues._WARNED.clear()  # warnings are logged once per process
    p = project_with([PanelSpec("structure", [], ""), PanelSpec("sequence", [], "", {"per_row": 5})],
                     "single_strip")
    _ds, panels = figure(p)
    seq = panels[1]
    assert seq.ok and seq.n_rows > 2 and "choose a smaller range" in seq.note.get_text()
    assert "smaller range" in caplog.text
    p.layout.panels[1] = PanelSpec("sequence", [], "", {"residues": "A:1-6"})
    _ds, panels = figure(p)
    seq = panels[1]
    assert seq.ok and seq.n_rows == 1 and seq.note.get_text() == "" and len(seq.texts) == 6
    assert "".join(t.get_text() for t in seq.texts) == "GCAUGG"
    for pos in (0, 5):  # colours and glow follow the frames
        seq.update(pos)
    assert seq.boxes.get_facecolors().shape[0] == 6


def test_secondary_panel_pairs_change_with_the_frames():
    p = project_with([PanelSpec("secondary", [], "", {"source": "dynamic"})], "single")
    _ds, panels = figure(p)
    arc = panels[0]
    assert arc.ok
    arc.update(0)
    n0 = len(arc.inter.get_segments())
    arc.update(6)  # the example opens the duplex end every 6 frames
    n6 = len(arc.inter.get_segments())
    assert n0 == 12 and n6 == 10 and len(arc.intra.get_segments()) == 0  # pairs between the two chains
    first = figure(project_with([PanelSpec("secondary", [], "", {"source": "first"})], "single"))[1][0]
    first.update(6)
    assert len(first.inter.get_segments()) == 12


def test_secondary_panel_from_dotbracket():
    db = "((((....))))" + "&" + "." * 12
    p = project_with([PanelSpec("secondary", [], "", {"source": "dotbracket", "dotbracket": db})], "single")
    arc = figure(p)[1][0]
    assert arc.ok
    arc.update(0)
    assert len(arc.intra.get_segments()) == 4 and len(arc.inter.get_segments()) == 0
    bad = project_with([PanelSpec("secondary", [], "", {"source": "dotbracket", "dotbracket": "((..))"})],
                       "single")
    assert not figure(bad)[1][0].ok  # 6 positions for 24 residues: a message instead of arcs


def test_secondary_panel_requires_the_basepair_analysis():
    from mdatk.render.panels import required_analyses
    p = project_with([PanelSpec("secondary", [], "")], "single")
    assert {"trace", "basepairs"} <= required_analyses(p)
    p.layout.panels[0].options = {"source": "dotbracket", "dotbracket": "."}
    assert "basepairs" not in required_analyses(p)


def test_structure_zoom_and_centre():
    p = project_with([PanelSpec("structure", [], "")], "single")
    st = figure(p, (800, 800))[1][0]
    w1 = np.diff(st.ax.get_xlim())[0]
    p.visual.zoom = 2.0
    st2 = figure(p, (800, 800))[1][0]
    assert np.diff(st2.ax.get_xlim())[0] < 0.6 * w1
    p.visual.center_on = "origin"
    assert np.allclose(figure(p, (800, 800))[1][0].center, 0.0)
    p.visual.center_on = "A:6"
    st3 = figure(p, (800, 800))[1][0]
    tr = st3.ctx.dataset.results["trace"]
    assert np.allclose(st3.center, tr.arrays["coords"][0][5])
    p.visual.center_on = "Z:99"  # not found: the model centre, with a note
    st4 = figure(p, (800, 800))[1][0]
    assert st4.ok and "centre" in st4.view_note


def test_followed_centre_moves_the_model_with_the_residue():
    p = project_with([PanelSpec("structure", [], "")], "single")
    ds = example_dataset(p)
    tr = ds.results["trace"]
    coords = np.array(tr.arrays["coords"], dtype=np.float64)
    coords[:, :, 0] += np.arange(len(coords))[:, None] * 0.5  # the whole model drifts along x
    tr.arrays["coords"] = coords
    p.visual.center_on, p.visual.center_follow = "A:3", True
    st = build_figure(ds, p, 600, 600, 100)[2][0]
    assert np.allclose(st.centred[:, 2], st.centred[0, 2])  # the residue stays in the middle
    assert np.allclose(st.frame_shift[-1, 0], 0.5 * (len(coords) - 1))


def test_long_chain_names_are_kept():
    import MDAnalysis as mda
    from mdatk.analysis.structure import chain_ids, chain_names
    u = mda.Universe.empty(4, n_residues=4, atom_resindex=[0, 1, 2, 3], trajectory=False)
    u.add_TopologyAttr("chainIDs", ["RNA_chain_A"] * 2 + ["RNA_chain_B"] * 2)
    u.add_TopologyAttr("segids", ["SYSTEM"])
    assert list(chain_names(u.residues)) == ["RNA_chain_A"] * 2 + ["RNA_chain_B"] * 2
    assert list(chain_ids(u.residues)) == [0, 0, 1, 1]


def test_labels_drawn_over_animated_layers_are_animated():
    p = project_with([PanelSpec("structure", [], ""), PanelSpec("secondary", [], "", {"source": "dynamic"})],
                     "single_strip")
    _ds, panels = figure(p)
    arc = panels[1]
    assert arc.labels and set(arc.labels) <= set(arc.animated_artists())
    p.layout.panels[1] = PanelSpec("sequence", [], "")
    seq = figure(p)[1][1]
    anim = set(seq.animated_artists())
    assert set(seq.chain_texts) <= anim and set(seq.tag_texts.values()) <= anim and seq.note in anim


def test_sequence_chain_label_never_ends_a_row():
    p = project_with([PanelSpec("structure", [], ""), PanelSpec("sequence", [], "")], "single_strip")
    seq = figure(p)[1][1]
    for per_row in range(2, 20):
        slots, n_rows = seq._slots(per_row)
        for slot, (idx, _lab) in zip(slots, seq.cells):
            if idx == -2:
                assert slot % per_row != per_row - 1
            if idx == -1:
                assert slot < 0 or slot % per_row != 0
        assert n_rows * per_row >= max(slots) + 1
