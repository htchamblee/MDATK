"""GTK tests of RNAdo's structure builder window (run under xvfb-run -a): its pages, the DNA option of the ideal
helix, and the chimera page: Check (the chain ids too), Write PDB in the window's worker with its progress bar,
and Cancel (also of the density fit); a suggested output name per design, the question before Write PDB
replaces a suggested file that exists, and the design in the step log.

The window is hosted by a stand-in for RNAdo (:class:`mdatk.gui.host.Host`: a project file that logs the builds
and an output log).  The chimeras have one or two base pairs per segment, so a fit takes seconds; the fits are
cached by design, so the tests that fit give the cache a fresh dictionary."""

import os
import time
from collections import OrderedDict

import pytest

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
except (ValueError, ImportError):  # pragma: no cover
    pytest.skip("GTK 3 introspection data not installed", allow_module_level=True)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    pytest.skip("no display (run under xvfb-run)", allow_module_level=True)
if not Gtk.init_check([])[0]:  # pragma: no cover
    pytest.skip("cannot initialise GTK", allow_module_level=True)

from mdatk import builder as B  # noqa: E402
from mdatk import chimera as C  # noqa: E402
from mdatk import pdbtools as pt  # noqa: E402
from mdatk.config import Project  # noqa: E402
from mdatk.gui.builderdialog import PAGES, BuilderWindow, chimera_file_name  # noqa: E402
from mdatk.projectfile import read_log  # noqa: E402


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.005)


def wait_for(cond, timeout=120.0):
    """Run the GTK main loop (the worker reports through it) until ``cond()`` or the timeout."""
    end = time.time() + timeout
    while time.time() < end and not cond():
        pump(0.02)
    return cond()


class Host:
    """What the builder window uses of RNAdo: the project file that logs the builds, and the output log."""

    def __init__(self, project_path):
        self.project = Project.load(project_path)
        self.project_path = project_path
        self.lines, self.messages = [], []

    def append_log(self, text):
        self.lines.append(text)

    def error(self, text, detail=""):
        self.messages.append(("error", text, detail))

    def info(self, text, detail=""):
        self.messages.append(("info", text, detail))


@pytest.fixture
def window(tmp_path):
    path = str(tmp_path / "build.mdatk.toml")
    Project().save(path)
    win = BuilderWindow(Host(path))
    win.show_all()
    pump(0.3)
    yield win
    win.destroy()
    pump(0.1)


@pytest.fixture
def fresh_fits(monkeypatch):
    """An empty cache of junction fits, so that a design is fitted here (not taken from another test's fit)."""
    monkeypatch.setattr(C, "_FITS", OrderedDict())


def report(win):
    buf = win.report.get_buffer()
    return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)


def labels(combo):
    return [row[0] for row in combo.get_model()]


def progress_seen(win, monkeypatch):
    """The (fraction, message) updates the window's progress bar receives."""
    seen = []
    real = win._on_progress
    monkeypatch.setattr(win, "_on_progress", lambda f, m: seen.append((f, m)) or real(f, m))
    return seen


def test_the_pages_in_order_and_their_buttons(window):
    nb = window.notebook
    assert [nb.get_tab_label_text(nb.get_nth_page(k)) for k in range(nb.get_n_pages())] == [
        "Ideal helix", "Chimera", "Mutate", "Fit to map"]
    assert PAGES == ("helix", "chimera", "mutate", "fit")
    for name in PAGES:
        window.set_page(name)
        pump(0.05)
        assert window.page() == name
        assert window.check_button.get_sensitive() == (name != "fit")  # the fit has nothing to check
        assert window.write_button.get_sensitive()
    assert window.cancel_button.get_label() == "Cancel"  # one button for both long jobs, shown while they run
    assert not window.cancel_button.get_visible() and not window.progress.get_visible()


def test_dna_leaves_the_pairing_controls_insensitive_and_builds_the_b_form(window, tmp_path):
    w = window
    assert w.na.get_active_id() == "rna" and labels(w.na) == ["RNA (A-form)", "DNA (B-form)"]
    assert not any("RNA" in text for text in labels(w.chirality))
    w.seq.set_text("GGCAT")
    w.triplex.set_active_id("hoogsteen")  # an RNA triplex chosen first
    controls = (w.pairing, w.triplex, w.third, w.chain3, w.start3)
    assert all(c.get_sensitive() for c in controls)
    w.scheme.set_active_id("amber")
    w.na.set_active_id("dna")
    assert not any(c.get_sensitive() for c in controls)
    # they show what is built: the Watson-Crick B-form duplex, no triplex
    assert w.pairing.get_active_text() == "Watson-Crick (cWW), antiparallel B-form"
    assert w.triplex.get_active_id() == "none"
    assert w.scheme.get_active_id() == "amber" and "LDA/LDC/LDG/LDT" in w.scheme.get_active_text()  # DNA names
    opts = w.helix_options()
    assert (opts["na"], opts["pairing"], opts["triplex"], opts["chains"]) == ("dna", "wc", None, ("A", "B"))
    w.check_current()
    assert "INFO: 5 base pairs: 5 Watson-Crick, 0 G·T wobble, 0 mismatched" in report(w).splitlines()
    assert "ERROR" not in report(w)  # the triplex and pairing chosen for RNA are not asked of the DNA helix
    w.chirality.set_active_id("L")
    out = tmp_path / "ldna.pdb"
    w.helix_out.entry.set_text(str(out))
    w.run_current()
    assert w.worker is None  # a Watson-Crick duplex is built at once
    s = pt.read_pdb(str(out))
    # three-character names at the chain ends too (0.0.17)
    assert s.chain_ids() == ["A", "B"] and [r.name for r in s.segments[0]] == ["LDG", "LDG", "LDC", "LDA", "LDT"]
    assert s.header[0].startswith("REMARK 951 MDATK IDEAL B-FORM DNA (left-handed")
    assert report(w).startswith(f"Wrote {out}\n") and "Phosphates turned" in report(w)
    assert w.host.lines == [f"Structure builder: wrote {out}"]
    entry = read_log(w.host.project_path)[-1]
    assert (entry["step"], entry["title"], entry["files"]) == ("build helix", "Ideal B-form L-DNA helix GGCAT",
                                                               [str(out)])
    assert entry["citations"] == ["arnott1972dna"]
    # back to RNA: the pairing controls with the RNA choices, and the RNA names again
    w.na.set_active_id("rna")
    assert all(c.get_sensitive() for c in controls) and "LA/LC/LG/LU" in w.scheme.get_active_text()
    assert w.pairing.get_active_text() == "Watson-Crick (cWW), antiparallel A-form"
    assert (w.helix_options()["na"], w.helix_options()["triplex"]) == ("rna", "hoogsteen")
    w.triplex.set_active_id("none")
    w.pairing.set_active_id("hh")
    w.na.set_active_id("dna")
    assert (w.pairing.get_active_id(), w.helix_options()["pairing"]) == ("wc", "wc")
    w.na.set_active_id("rna")
    assert w.pairing.get_active_id() == "hh" and not w.triplex.get_sensitive()


def test_check_reports_a_bad_design_and_the_output_name_follows_the_design(window, tmp_path):
    w = window
    w.set_page("chimera")
    w.design.set_text("D:GC 3'-3' L:GC 3'-5' D:GC")  # after a 3'-3' link the free end is a 5' end
    w.check_current()
    text = report(w)
    assert text.startswith("ERROR: link 2 (3'-5') starts at a 3' end") and "use 5'-3' or 5'-5'" in text
    assert w.chim_out.entry.get_text() == ""  # nothing to name yet
    w.design.set_text("D:GX 3'-5' L:GC")
    w.check_current()
    assert report(w) == "ERROR: segment 1 sequence: invalid letter(s) X (RNA: A, C, G, U)"
    w.chim_na.set_active_id("dna")  # the default nucleic acid of the segments that name none
    w.check_current()
    assert report(w) == "ERROR: segment 1 sequence: invalid letter(s) X (DNA: A, C, G, T)"
    w.design.set_text("D:GC 3'-3' L-RNA:GC")
    w.check_current()
    text = report(w)
    assert "ERROR" not in text and text.startswith("INFO: 5'-D-DNA(2)-3'p3'-L-RNA(2)-5': 2 segments, 1 link(s)")
    assert "segment 1 (D-DNA): 2 base pairs: 2 Watson-Crick" in text
    # the suggested output: next to the project file, named after the design (its layout and a digest of the
    # design itself), until the user types one
    assert w.chim_out.entry.get_text() == str(tmp_path / "chimera_D-DNA2_3p3_L-RNA2_6783ad.pdb")
    w.design.set_text("D:GC 3'-3' L-RNA:G")
    assert w.chim_out.entry.get_text() == str(tmp_path / "chimera_D-DNA2_3p3_L-RNA1_092506.pdb")
    w.design.set_text("D:GC 3'-3' L-RNA:")  # being typed: the last suggestion stays
    assert w.chim_out.entry.get_text() == str(tmp_path / "chimera_D-DNA2_3p3_L-RNA1_092506.pdb")
    w.chim_na.set_active_id("rna")
    w.design.set_text("D:GC 5'-5' L:GC")
    assert w.chim_out.entry.get_text() == str(tmp_path / "chimera_D-RNA2_5p5_L-RNA2_f3dcfc.pdb")
    w.chim_out.entry.set_text(str(tmp_path / "mine.pdb"))
    w.design.set_text("D:GC 3'-5' L:GC")
    assert w.chim_out.entry.get_text() == str(tmp_path / "mine.pdb")
    assert chimera_file_name("D-RNA:GGCAUC 3'-5' L-RNA:GAUGCC") == "chimera_D-RNA6_p_L-RNA6_c1e9cc.pdb"
    assert chimera_file_name("D:GC 3'-3'") == ""


def test_a_suggested_name_per_design_sequences_and_partners_included():
    """Designs of one layout (hands, nucleic acids, lengths, links) get names of their own; one design typed two
    ways gets one name (the digest is of the parsed design)."""
    same = chimera_file_name("D-RNA:GG 3'-5' L-RNA:CC")
    assert same == "chimera_D-RNA2_p_L-RNA2_f9d38d.pdb"
    assert chimera_file_name("d-rna:gg/auto 3-5 L:cc") == same == chimera_file_name("D:GG/CC 3'5' L-RNA:CC")
    others = [chimera_file_name(d) for d in ("D-RNA:AU 3'-5' L-RNA:UA", "D-RNA:GG/none 3'-5' L-RNA:CC",
                                             "D-RNA:GG/CU 3'-5' L-RNA:CC", "D-RNA:GG 3'-5' L-RNA:CC/none")]
    assert len(set(others + [same])) == 5 and all(n.startswith("chimera_D-RNA2_p_L-RNA2_") for n in others)
    assert chimera_file_name("D:GC 3'-5' L:GC", "dna") == "chimera_D-DNA2_p_L-DNA2_64c3d3.pdb"


def test_check_validates_the_chain_ids_as_the_build_and_the_planner_do(window):
    from mdatk.workflows import TOOLS
    from mdatk.workflows.base import StepContext

    tool = TOOLS["structure-builder"]
    ctx = StepContext("p", 1, "s", tool.tag)
    w = window
    w.set_page("chimera")
    design = "D-RNA:GGCAUC 3'-5' L-RNA:GAUGCC"  # 3 chains: strand 1, then the two partner strands
    w.design.set_text(design)
    for chains, error in (("A,B", "ERROR: chains: the design writes 3 chains (the strand-1 chains between bridges in "
                                  "design order, then the partner strands in segment order); 2 chain id(s) given"),
                          ("A,A,B", "ERROR: chains: every chain needs its own id, got A, A, B"),
                          ("AB,C,D", "ERROR: chains: chain ids must be single characters, got 'AB', 'C', 'D'"),
                          ("P, Q ,R", None), ("", None), ("-,B,C", None)):
        w.chim_chains.set_text(chains)
        w.check_current()
        planner = [i.message for i in tool.check(tool.normalise({"mode": "chimera", "design": design,
                                                                 "chimera_chains": chains}), ctx)
                   if i.level == "error"]
        if error is None:
            assert "ERROR" not in report(w) and report(w).startswith("INFO: 5'-D-RNA(6)-p-L-RNA(6)-3'"), chains
            assert planner == [], chains
        else:
            assert report(w) == error  # what Write PDB would stop with, before any fit
            assert planner == [error[len("ERROR: "):]], chains  # the planner step's message, word for word
    w.chim_chains.set_text("A,B")
    w.design.set_text("D:GX 3'-5' L:GC")  # a design with errors: those first (its chains are not counted)
    w.check_current()
    assert report(w) == "ERROR: segment 1 sequence: invalid letter(s) X (RNA: A, C, G, U)"


def answering(monkeypatch, responses):
    """Answer the window's message dialogs with ``responses`` in turn; returns what each asked: (question,
    detail, whether every label of the dialog is selectable)."""
    asked, queue = [], list(responses)

    def run(dlg):
        labels = [c for c in dlg.get_message_area().get_children() if isinstance(c, Gtk.Label)]
        asked.append((dlg.props.text, dlg.props.secondary_text or "", all(lb.get_selectable() for lb in labels)))
        return queue.pop(0)

    monkeypatch.setattr(Gtk.MessageDialog, "run", run)
    return asked


def residues(path):
    return [(r.chain, r.resseq, r.name) for r in pt.read_pdb(str(path)).residues()]


def test_write_pdb_asks_before_it_replaces_a_suggested_file_and_logs_the_design(window, tmp_path, monkeypatch):
    w = window
    asked = answering(monkeypatch, [Gtk.ResponseType.NO, Gtk.ResponseType.YES])
    w.set_page("chimera")
    w.design.set_text("D-RNA:GG 3'-5' L-RNA:CC")
    first = tmp_path / "chimera_D-RNA2_p_L-RNA2_f9d38d.pdb"
    assert w.chim_out.entry.get_text() == str(first)
    w.run_current()
    assert wait_for(lambda: w.worker is None, 180) and first.exists() and not asked
    notes = report(w).splitlines()[2:]  # 'Wrote ...', '', then the build notes
    # another design of the same layout: a file of its own, the first model stays
    w.design.set_text("D-RNA:AU 3'-5' L-RNA:UA")
    second = tmp_path / "chimera_D-RNA2_p_L-RNA2_19cf5e.pdb"
    assert w.chim_out.entry.get_text() == str(second)
    w.chim_chains.set_text("P,Q,R")
    w.chim_scheme.set_active_id("amber")
    w.run_current()
    assert wait_for(lambda: w.worker is None, 180) and not asked
    assert residues(first)[:4] == [("A", 1, "G"), ("A", 2, "G"), ("A", 3, "0C"), ("A", 4, "0C")]
    assert residues(second)[:4] == [("P", 1, "A5"), ("P", 2, "U"), ("P", 3, "LU"), ("P", 4, "LA3")]
    # the step log holds each design, so that it can rebuild the models (the titles name no sequences)
    entries = read_log(w.host.project_path)
    assert [(e["title"], e["files"]) for e in entries] == [("Chimera 5'-D-RNA(2)-p-L-RNA(2)-3'", [str(first)]),
                                                           ("Chimera 5'-D-RNA(2)-p-L-RNA(2)-3'", [str(second)])]
    assert [e["summary"].splitlines()[0] for e in entries] == [
        "Design: D-RNA:GG 3'-5' L-RNA:CC (default nucleic acid RNA; chains A, B, C ...; residue names default)",
        "Design: D-RNA:AU 3'-5' L-RNA:UA (default nucleic acid RNA; chains P,Q,R; residue names amber)"]
    assert entries[0]["summary"].splitlines()[1:] == notes and notes  # then the notes, as in the report
    # back to the first design: its file exists, Write PDB asks first; No writes nothing
    w.chim_chains.set_text("")
    w.chim_scheme.set_active_id("ccd")
    w.design.set_text("D-RNA:GG 3'-5' L-RNA:CC")
    assert w.chim_out.entry.get_text() == str(first)
    before = first.read_text()
    w.run_current()
    assert w.worker is None and first.read_text() == before and len(read_log(w.host.project_path)) == 2
    assert asked == [(f"Replace {first.name}?", f"The suggested output file {first} exists already (an earlier "
                      "build?). Yes replaces it; No writes nothing, so that you can choose another file.", True)]
    assert report(w) == (f"Nothing written: {first} exists. Choose another output file, or Write PDB again and "
                         "replace it.")
    # Yes replaces it
    w.run_current()
    assert wait_for(lambda: w.worker is None, 180) and len(asked) == 2
    assert first.read_text() != before and residues(first)[3] == ("A", 4, "0C3")  # the CCD names now
    assert read_log(w.host.project_path)[-1]["summary"].splitlines()[0] == (
        "Design: D-RNA:GG 3'-5' L-RNA:CC (default nucleic acid RNA; chains A, B, C ...; residue names ccd)")
    # a file chosen by the user is not asked about (the browse dialog asks), whether it exists or not
    w.chim_out.entry.set_text(str(second))
    w.run_current()
    assert wait_for(lambda: w.worker is None, 180) and len(asked) == 2 and residues(second)[0] == ("A", 1, "G")


def test_the_mutate_page_asks_before_it_replaces_its_suggestion(window, tmp_path, monkeypatch):
    w = window
    asked = answering(monkeypatch, [Gtk.ResponseType.NO])
    model = tmp_path / "helix.pdb"
    model.write_text(B.build_helix("GGCAU"))
    w.set_page("mutate")
    w.mut_input.entry.set_text(str(model))
    w.mutations.set_text("A:2C")
    out = tmp_path / "helix_mut.pdb"
    assert w.mut_out.entry.get_text() == str(out)
    w.run_current()
    assert out.exists() and not asked
    out.write_text("an earlier mutant\n")
    w.run_current()
    assert len(asked) == 1 and asked[0][0] == "Replace helix_mut.pdb?" and asked[0][2]
    assert out.read_text() == "an earlier mutant\n" and report(w).startswith("Nothing written:")


def test_write_pdb_fits_a_chimera_in_the_worker(window, tmp_path, monkeypatch, fresh_fits):
    w = window
    seen = progress_seen(w, monkeypatch)
    w.set_page("chimera")
    w.design.set_text("D:GC 3'-3' L:GC")
    w.chim_chains.set_text("P,Q,R,S")
    w.chim_scheme.set_active_id("amber")
    out = tmp_path / "chimera_D-RNA2_3p3_L-RNA2_b31a74.pdb"
    assert w.chim_out.entry.get_text() == str(out)
    w.run_current()
    assert w.worker is not None  # the window stays responsive while the junctions are fitted
    assert w.progress.get_visible() and w.cancel_button.get_visible() and not w.write_button.get_sensitive()
    assert report(w).startswith("Fitting the junctions of 5'-D-RNA(2)-3'p3'-L-RNA(2)-5'")
    w.run_current()  # Write PDB again meanwhile: nothing starts
    assert wait_for(lambda: w.worker is None, 180)
    assert not w.progress.get_visible() and not w.cancel_button.get_visible() and w.write_button.get_sensitive()
    fractions = [f for f, _m in seen]
    assert len(fractions) > 10 and fractions == sorted(fractions) and fractions[-1] == 1.0
    assert seen[0][1] == "the design: junction 1 (3'-3'): start 1 of 126"
    # a readable PDB with the chains asked for: the strand-1 runs, then the partners; the PO2 bridge a TER
    # segment of its own after the run it follows (its chain id)
    s = pt.read_pdb(str(out))
    assert [(seg[0].chain, [r.name for r in seg]) for seg in s.segments] == [
        ("P", ["G5", "C"]), ("P", ["PO2"]), ("Q", ["LG5", "LC"]), ("R", ["G5", "C3"]), ("S", ["LG5", "LC3"])]
    assert [(k.first.name, k.first.chain, k.first.resseq, k.second.name, k.second.chain, k.second.resseq)
            for k in pt.link_records(s)] == [("O3'", "P", 2, "P", "P", 3), ("P", "P", 3, "O3'", "Q", 2)]
    # the notes in the report, the output log and the step log (as the helix page and 'mdatk build chimera')
    text = report(w)
    assert text.startswith(f"Wrote {out}\n") and "Junction 1 (3'-3', P:2 O3' - P:3 P - Q:2 O3'" in text
    assert w.host.lines == [f"Structure builder: wrote {out}"]
    entry = read_log(w.host.project_path)[-1]
    assert (entry["step"], entry["title"], entry["status"]) == ("build chimera",
                                                                "Chimera 5'-D-RNA(2)-3'p3'-L-RNA(2)-5'", "ok")
    assert entry["files"] == [str(out)] and entry["citations"] == ["arnott1972"]
    assert "Junction 1 (3'-3'" in entry["summary"]
    assert entry["summary"].splitlines()[0] == ("Design: D:GC 3'-3' L:GC (default nucleic acid RNA; chains "
                                                "P,Q,R,S; residue names amber)")
    # chain ids that do not fit the design: reported at once (before any fit), logged as failed; the model
    # written above stays (the window asks before it replaces the file it suggested)
    asked = answering(monkeypatch, [Gtk.ResponseType.YES])
    before = out.read_text()
    w.chim_chains.set_text("P,Q")
    w.run_current()
    assert wait_for(lambda: w.worker is None, 30)
    assert report(w).startswith("ERROR: chains: the design writes 4 chains")
    assert read_log(w.host.project_path)[-1]["status"] == "failed"
    assert [q for q, _d, _s in asked] == [f"Replace {out.name}?"] and out.read_text() == before


def test_cancel_stops_a_running_chimera_fit(window, tmp_path, monkeypatch, fresh_fits):
    w = window
    seen = progress_seen(w, monkeypatch)
    design = "D-DNA:GGCATGCA 3'-3' L-DNA:GGCATGCA 5'-3' D-DNA:GCA"
    w.set_page("chimera")
    w.design.set_text(design)
    out = tmp_path / "cancelled.pdb"
    w.chim_out.entry.set_text(str(out))
    w.run_current()
    assert wait_for(lambda: seen, 60)  # the fit runs
    t0 = time.time()
    w.cancel_button.clicked()
    assert wait_for(lambda: w.worker is None, 30)
    assert time.time() - t0 < 5.0
    assert max(f for f, _m in seen) < 0.5  # stopped within the design's own fit (its mirror image comes second)
    segments, links, _notes = C.parse_design(design)
    assert C._fit_key(segments, links, False) not in C._FITS  # no fit was finished
    assert report(w) == "Cancelled: the junction fit stopped, nothing was written." and not out.exists()
    assert w.write_button.get_sensitive() and not w.cancel_button.get_visible() and not w.progress.get_visible()
    assert w.host.lines == []
    entry = read_log(w.host.project_path)[-1]
    assert (entry["step"], entry["status"]) == ("build chimera", "cancelled") and "summary" not in entry
    # closing the window stops a running fit as well
    del seen[:]
    w.run_current()
    assert wait_for(lambda: seen, 60)
    worker = w.worker
    w.destroy()
    worker.thread.join(10)
    assert not worker.running and not out.exists()


def test_cancel_stops_a_running_density_fit(window, tmp_path, monkeypatch):
    from mdatk.density import atom_weights, simulate_map
    from mdatk.io.mrc import write_map

    w = window
    seen = progress_seen(w, monkeypatch)
    model = tmp_path / "helix.pdb"
    model.write_text(B.build_helix("GAAGAGAAGGC"))
    xyz, elements = B.model_atoms(pt.read_pdb(str(model)))
    density = tmp_path / "sim.mrc"
    write_map(str(density), simulate_map(xyz + [30.0, -12.0, 55.0], atom_weights(elements), 4.0))
    w.set_page("fit")
    w.fit_input.entry.set_text(str(model))
    w.fit_map.entry.set_text(str(density))
    w.step.set_value(10.0)  # a fine global search: long enough to cancel
    assert w.fit_out.entry.get_text() == str(tmp_path / "helix_fit.pdb")
    w.run_current()
    assert w.cancel_button.get_visible() and w.progress.get_visible()
    assert wait_for(lambda: seen, 60)
    w.cancel_button.clicked()
    assert wait_for(lambda: w.worker is None, 60)
    assert report(w) == "Cancelled: the density fit stopped, nothing was written."
    assert not (tmp_path / "helix_fit.pdb").exists() and not w.cancel_button.get_visible()
    entry = read_log(w.host.project_path)[-1]
    assert (entry["step"], entry["status"]) == ("build fit", "cancelled")
