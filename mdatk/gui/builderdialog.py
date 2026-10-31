"""Structure builder window of RNAdo: the GUI of :mod:`mdatk.builder` ('mdatk build helix' / 'mutate' /
'fit').

It uses only RNAdo's host interface (:class:`mdatk.gui.host.Host`): the project file that logs the
builds and the output log.
"""

from __future__ import annotations

import os
from typing import Optional

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from .. import builder, pdbtools  # noqa: E402
from ..citations import keys_for  # noqa: E402
from ..projectfile import logged  # noqa: E402
from . import widgets as W  # noqa: E402
from .host import Host  # noqa: E402

PDB_FILES = ("PDB files", ["*.pdb", "*.ent"])
MAP_FILES = ("Density maps", ["*.mrc", "*.map", "*.ccp4"])
SCHEME_LABELS = [
    ("", "default (A/C/G/U; L: 0A/0C/0G/0U)"),
    ("amber", "Amber (L: LA/LC/LG/LU + 5'/3', Dudek & Trylska)"),
    ("ccd", "CCD for Coot / Phenix (L: 0A … + 0A5/0A3)"),
    ("rosetta", "Rosetta (L: 0A/0C/0G/0U, all ATOM)"),
    ("pdb", "PDB deposition (L: 0A …, non-standard as HETATM)"),
]
PARTNER_LABELS = [("auto", "complement (canonical partners)"), ("none", "none (single strand)"),
                  ("explicit", "explicit sequence")]
PAIRING_LABELS = [("wc", "Watson-Crick (cWW), antiparallel A-form"),
                  ("reverse-wc", "reverse Watson-Crick (tWW), parallel"),
                  ("hoogsteen", "Hoogsteen (cWH: A·U, G·C+), parallel"),
                  ("reverse-hoogsteen", "reverse Hoogsteen (tWH), antiparallel"),
                  ("hh", "Hoogsteen/Hoogsteen A·A (tHH, PDB 4JRD), parallel")]
TRIPLEX_LABELS = [("none", "none (duplex)"), ("hoogsteen", "pyrimidine motif: Hoogsteen, parallel"),
                  ("reverse-hoogsteen", "purine motif: reverse Hoogsteen, antiparallel")]
PAGES = ("helix", "mutate", "fit")


class BuilderWindow(Gtk.Window):
    def __init__(self, host: Host):
        super().__init__(title="Structure builder · RNAdo",
                         transient_for=host if isinstance(host, Gtk.Window) else None)
        self.host = host
        self.worker = None
        self._auto = {"mutate": "", "fit": ""}
        self.set_default_size(980, 680)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(outer)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        outer.pack_start(paned, True, True, 0)
        self.notebook = Gtk.Notebook()
        self.notebook.append_page(self._helix_page(), Gtk.Label(label="Ideal helix"))
        self.notebook.append_page(self._mutate_page(), Gtk.Label(label="Mutate"))
        self.notebook.append_page(self._fit_page(), Gtk.Label(label="Fit to map"))
        self.notebook.connect("switch-page", lambda _nb, _p, n: self._sync(n))
        self.notebook.set_size_request(500, -1)
        paned.pack1(self.notebook, False, False)
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=8)
        right.pack_start(W.label("<b>Report</b>", markup=True), False, False, 0)
        self.report = Gtk.TextView(monospace=True, editable=False)
        self.report.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        sw = Gtk.ScrolledWindow()
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.add(self.report)
        right.pack_start(sw, True, True, 0)
        self.progress = Gtk.ProgressBar(show_text=True)
        self.progress.set_no_show_all(True)
        right.pack_start(self.progress, False, False, 0)
        paned.pack2(right, True, False)
        paned.set_position(520)
        bar = Gtk.Box(spacing=6, margin=8)
        bar.pack_end(W.button("Close", self.destroy), False, False, 0)
        self.write_button = W.button("Write PDB", self.run_current, "Run the current page and write its "
                                     "output file", "document-save-symbolic")
        self.write_button.get_style_context().add_class("suggested-action")
        bar.pack_end(self.write_button, False, False, 0)
        self.check_button = W.button("Check", self.check_current, "Validate the inputs of the current page "
                                     "(nothing is written)")
        bar.pack_end(self.check_button, False, False, 0)
        self.cancel_button = W.button("Cancel fit", self._cancel, "Stop the running density fit")
        self.cancel_button.set_no_show_all(True)
        bar.pack_end(self.cancel_button, False, False, 0)
        outer.pack_start(bar, False, False, 0)
        self.connect("destroy", lambda *_: self._cancel())
        self._sync(0)

    # ------------------------------------------------------------------ pages
    def _helix_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        f = W.Form()
        self.seq = W.entry("e.g. GGCAUCGAUGCC", "Strand 1, 5'->3' (T is read as U)")
        f.add("Sequence (5'->3')", self.seq)
        self.partner_mode = W.combo(PARTNER_LABELS)
        self.partner_mode.set_active_id("auto")
        self.partner_mode.connect("changed", lambda *_: self._sync())
        f.add("Partner strand", self.partner_mode, "An explicit partner (5'->3', same length) may contain "
              "mismatches and G·U wobbles")
        self.partner = W.entry("partner 5'->3'")
        f.add("Partner sequence", self.partner)
        self.chirality = W.combo([("D", "D-RNA (right-handed)"), ("L", "L-RNA (mirror image, left-handed)")])
        self.chirality.set_active_id("D")
        f.add("Chirality", self.chirality)
        self.pairing = W.combo(PAIRING_LABELS)
        self.pairing.set_active_id("wc")
        self.pairing.connect("changed", lambda *_: self._sync())
        f.add("Pairing", self.pairing, "Base-pair geometry of the duplex; for the Hoogsteen pairings strand 1 is "
              "the purine strand. Mismatches are allowed and reported.")
        self.triplex = W.combo(TRIPLEX_LABELS)
        self.triplex.set_active_id("none")
        self.triplex.connect("changed", lambda *_: self._sync())
        f.add("Triplex", self.triplex, "A third strand in the major groove on the Hoogsteen edge of the strand-1 "
              "purines (Watson-Crick duplex only)")
        self.third = W.entry("third strand 5'->3' (empty: motif bases)", "Empty: U / C+ (pyrimidine motif) or "
                             "A / G (purine motif) on the strand-1 A / G")
        f.add("Third strand", self.third)
        box.pack_start(W.frame("Sequence", f), False, False, 0)
        n = W.Form()
        self.chain1, self.chain2, self.chain3 = W.entry("A", width=3), W.entry("B", width=3), W.entry("C", width=3)
        self.chain1.set_text("A")
        self.chain2.set_text("B")
        self.chain3.set_text("C")
        n.add_row("Chains (strand 1 / 2 / 3)", [self.chain1, self.chain2, self.chain3])
        self.start1, self.start2, self.start3 = W.spin(-999, 9999, 1), W.spin(-999, 9999, 1), W.spin(-999, 9999, 1)
        self.start1.set_value(1)
        self.start2.set_value(1)
        self.start3.set_value(1)
        n.add_row("First residue (1 / 2 / 3)", [self.start1, self.start2, self.start3])
        self.scheme = W.combo(SCHEME_LABELS)
        self.scheme.set_active_id("")
        n.add("Residue names", self.scheme)
        self.terminals = W.check("5'/3' terminal names (Amber, CCD)")
        self.terminals.set_active(True)
        n.add_wide(self.terminals)
        self.phosphate5 = W.check("Keep a 5'-phosphate (P, OP1, OP2, OP3)")
        n.add_wide(self.phosphate5)
        self.hydrogens = W.check("Add hydrogens (default: heavy atoms only)")
        n.add_wide(self.hydrogens)
        box.pack_start(W.frame("Names and atoms", n), False, False, 0)
        o = W.Form()
        self.helix_out = W.file_chooser_button("Output PDB", Gtk.FileChooserAction.SAVE, patterns=[PDB_FILES])
        o.add("Output PDB", self.helix_out)
        box.pack_start(W.frame("Output", o), False, False, 0)
        box.pack_start(W.hint("Watson-Crick: Arnott A-RNA fiber geometry, C3'-endo sugars, 11 bp per turn (twist "
                              "32.7°, rise 2.81 Å). Other pairings and triplexes: the PDB 4JRD nucleotide with "
                              "fitted helices and backbones (starting models: minimise; the report lists link "
                              "gaps, clashes and mismatches). The L helix is the exact mirror image of the D "
                              "helix."), False, False, 0)
        return W.scrolled(box)

    def _mutate_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        f = W.Form()
        self.mut_input = W.file_chooser_button("Select a PDB file", patterns=[PDB_FILES])
        self.mut_input.entry.connect("changed", lambda *_: self._suggest("mutate"))
        f.add("Input PDB", self.mut_input)
        self.mutations = W.entry("e.g. A:5G, B:20C")
        f.add("Mutations", self.mutations, "Chain:residue number followed by the new base")
        self.compensatory = W.check("Compensatory mutations of the paired partners")
        self.compensatory.set_active(True)
        self.compensatory.set_tooltip_text("Every mutated base that is Watson-Crick / G·U paired in the input "
                                           "gets its partner changed to the complement, unless you mutate the "
                                           "partner yourself")
        self.compensatory.connect("toggled", lambda *_: self._sync())
        f.add_wide(self.compensatory)
        self.wobble = W.check("Keep G·U pairs where possible")
        f.add_wide(self.wobble)
        self.mut_out = W.file_chooser_button("Output PDB", Gtk.FileChooserAction.SAVE, patterns=[PDB_FILES])
        f.add("Output PDB", self.mut_out)
        box.pack_start(W.frame("Mutations", f), False, False, 0)
        box.pack_start(W.hint("The new base is superposed on the old glycosidic frame (C1', N9/N1 and its two "
                              "ring neighbours); sugar and backbone atoms are kept as they are. Residue "
                              "names keep their scheme and chirality (LG -> LC, 0G -> 0C)."), False, False, 0)
        return W.scrolled(box)

    def _fit_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        f = W.Form()
        self.fit_input = W.file_chooser_button("Select a PDB file", patterns=[PDB_FILES])
        self.fit_input.entry.connect("changed", lambda *_: self._suggest("fit"))
        f.add("Model PDB", self.fit_input)
        self.fit_map = W.file_chooser_button("Select a density map", patterns=[MAP_FILES])
        f.add("Density map", self.fit_map)
        self.resolution = W.spin(0.5, 50.0, 0.1, 1)
        self.resolution.set_value(4.0)
        f.add("Resolution (Å)", self.resolution, "Width of the map simulated from the model")
        self.search = W.combo([("global", "global search (all orientations)"),
                               ("local", "local refinement of the model as placed")])
        self.search.set_active_id("global")
        self.search.connect("changed", lambda *_: self._sync())
        f.add("Search", self.search)
        self.step = W.spin(5.0, 45.0, 5.0, 0)
        self.step.set_value(30.0)
        f.add("Angular step (°)", self.step)
        self.fit_out = W.file_chooser_button("Output PDB", Gtk.FileChooserAction.SAVE, patterns=[PDB_FILES])
        f.add("Output PDB", self.fit_out)
        box.pack_start(W.frame("Rigid-body fit", f), False, False, 0)
        box.pack_start(W.hint("Only coordinates change: every other line of the model is written as it was."),
                       False, False, 0)
        return W.scrolled(box)

    def _sync(self, page: Optional[int] = None) -> None:
        explicit = self.partner_mode.get_active_id() == "explicit"
        self.partner.set_sensitive(explicit)
        duplex = self.partner_mode.get_active_id() != "none"
        self.chain2.set_sensitive(duplex)
        self.start2.set_sensitive(duplex)
        self.pairing.set_sensitive(duplex)
        self.triplex.set_sensitive(duplex and self.pairing.get_active_id() == "wc")
        triple = self.triplex.get_sensitive() and self.triplex.get_active_id() != "none"
        for widget in (self.third, self.chain3, self.start3):
            widget.set_sensitive(triple)
        self.wobble.set_sensitive(self.compensatory.get_active())
        self.step.set_sensitive(self.search.get_active_id() == "global")
        page = self.notebook.get_current_page() if page is None else page
        self.check_button.set_sensitive(page != 2)

    def _suggest(self, page: str) -> None:
        """<input>_mut.pdb / <input>_fit.pdb while the output is empty or still the previous suggestion."""
        src = self.mut_input if page == "mutate" else self.fit_input
        dst = self.mut_out if page == "mutate" else self.fit_out
        path = src.entry.get_text().strip()
        current = dst.entry.get_text().strip()
        if current and current != self._auto[page]:
            return
        stem, ext = os.path.splitext(path)
        self._auto[page] = f"{stem}_{'mut' if page == 'mutate' else 'fit'}{ext or '.pdb'}" if path else ""
        dst.entry.set_text(self._auto[page])

    # --------------------------------------------------------------- actions
    def page(self) -> str:
        return PAGES[self.notebook.get_current_page()]

    def helix_options(self) -> dict:
        mode = self.partner_mode.get_active_id()
        partner = self.partner.get_text().strip() if mode == "explicit" else mode
        c1 = self.chain1.get_text().strip() or "A"
        c2 = self.chain2.get_text().strip() or "B"
        pairing = self.pairing.get_active_id() if mode != "none" else "wc"
        triplex = self.triplex.get_active_id() if mode != "none" and pairing == "wc" else "none"
        chains, start = (c1, c2), (int(self.start1.get_value()), int(self.start2.get_value()))
        if triplex != "none":
            chains += (self.chain3.get_text().strip() or "C",)
            start += (int(self.start3.get_value()),)
        return dict(sequence=self.seq.get_text(), partner=partner, chirality=self.chirality.get_active_id(),
                    chains=chains, start=start,
                    scheme=self.scheme.get_active_id() or None, terminals=self.terminals.get_active(),
                    hydrogens=self.hydrogens.get_active(), phosphate5=self.phosphate5.get_active(),
                    pairing=pairing, triplex=None if triplex == "none" else triplex,
                    third=self.third.get_text().strip() or "auto")

    def _show(self, text: str) -> None:
        self.report.get_buffer().set_text(text)

    def _output(self, chooser, source: str = "") -> Optional[str]:
        out = chooser.entry.get_text().strip()
        if not out:
            self._show("Choose an output file.")
            return None
        if source and os.path.abspath(out) == os.path.abspath(source):
            self._show("The output must not overwrite the input file.")
            return None
        return out

    def _input(self, chooser, what: str) -> Optional[str]:
        path = chooser.entry.get_text().strip()
        if not path or not os.path.isfile(path):
            self._show(f"Choose an existing {what} first.")
            return None
        return path

    def check_current(self) -> None:
        if self.page() == "helix":
            opts = self.helix_options()
            msgs = builder.check_helix(opts["sequence"], opts["partner"], opts["pairing"], opts["triplex"],
                                       opts["third"])
            self._show("\n".join(f"{level.upper()}: {text}" for level, text in msgs))
        elif self.page() == "mutate":
            path = self._input(self.mut_input, "input PDB")
            if not path:
                return
            try:
                muts = builder.parse_mutations(self.mutations.get_text())
                s = pdbtools.read_pdb(path)
            except (OSError, pdbtools.PDBError) as exc:
                self._show(f"ERROR: {exc}")
                return
            have = {(r.chain, r.resseq, r.icode): r for r in s.residues()}
            lines = []
            for m in muts:
                res = have.get((m.chain, m.resseq, m.icode))
                lines.append(f"{m.where} -> {m.base}: " + (f"{res.name}" if res else "NOT FOUND"))
            self._show("\n".join(lines))

    def run_current(self) -> None:
        {"helix": self.run_helix, "mutate": self.run_mutate, "fit": self.run_fit}[self.page()]()

    def _project(self):
        return self.host.project_path

    def _log(self, text: str) -> None:
        self.host.append_log(text)

    def run_helix(self) -> None:
        out = self._output(self.helix_out)
        if not out:
            return
        opts = self.helix_options()
        other = opts["pairing"] != "wc" or opts["triplex"] not in (None, "", "none")
        project = self._project()

        def job(_rep=None):
            notes = []
            kind = builder.helix_kind(opts["pairing"], opts["triplex"])
            title = f"Ideal {kind} {opts['chirality']}-RNA helix {opts['sequence'].strip()}"
            with logged(project, "build helix", title) as entry:
                text = builder.build_helix(notes=notes, **opts)
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(text)
                entry.files = [out]
                entry.summary = "\n".join(notes)
                entry.citations = keys_for(["arnott-arna"] + (["pdb-4jrd"] if other else []))
            return out, notes

        if other:  # the fitted pairings can take tens of seconds: build in the background
            from .worker import Worker
            self._show("Building the helix (other pairings take a few seconds per 10 nucleotides) …")
            self.progress.show()
            self.progress.pulse()
            self.write_button.set_sensitive(False)
            self.worker = Worker(job, lambda _t: None, lambda *_: self.progress.pulse(), self._on_helix_done)
            self.worker.start()
            return
        try:
            result = job()
        except (OSError, ValueError) as exc:
            self._show(f"ERROR: {exc}")
            return
        self._on_helix_done(result, None, None)

    def _on_helix_done(self, result, error, trace) -> None:
        self.worker = None
        self.progress.hide()
        self.write_button.set_sensitive(True)
        if error is not None:
            self._show(f"ERROR: {error}")
            return
        out, notes = result
        self._show("\n".join([f"Wrote {out}", ""] + notes))
        self._log(f"Structure builder: wrote {out}")

    def run_mutate(self) -> None:
        path = self._input(self.mut_input, "input PDB")
        if not path:
            return
        out = self._output(self.mut_out, path)
        if not out:
            return
        spec = self.mutations.get_text().strip()
        try:
            with logged(self._project(), "build mutate", f"Mutation of {os.path.basename(path)}: {spec}") as entry:
                text, notes = builder.mutate(path, spec, compensatory=self.compensatory.get_active(),
                                             wobble=self.wobble.get_active())
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(text)
                entry.files = [out]
                entry.summary = "\n".join(notes)
        except (OSError, ValueError) as exc:
            self._show(f"ERROR: {exc}")
            return
        self._show("\n".join([f"Wrote {out}", ""] + notes))
        self._log(f"Structure builder: wrote {out}")

    def run_fit(self) -> None:
        if self.worker is not None and self.worker.running:
            return
        path = self._input(self.fit_input, "model PDB")
        mpath = path and self._input(self.fit_map, "density map")
        if not path or not mpath:
            return
        out = self._output(self.fit_out, path)
        if not out:
            return
        from .worker import Worker
        resolution, search, step = self.resolution.get_value(), self.search.get_active_id(), self.step.get_value()
        project = self._project()

        def job(rep):
            title = f"Rigid-body fit of {os.path.basename(path)} into {os.path.basename(mpath)}"
            with logged(project, "build fit", title) as entry:
                text, cc, notes = builder.fit_to_map(
                    path, mpath, resolution, search, step, progress=rep.progress,
                    cancelled=lambda: rep.cancel_event.is_set())
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(text)
                entry.files = [out]
                entry.summary = "\n".join(notes)
            return out, cc, notes

        self._show(f"Fitting {os.path.basename(path)} into {os.path.basename(mpath)} …")
        self.progress.show()
        self.cancel_button.show()
        self.write_button.set_sensitive(False)
        self.worker = Worker(job, lambda _t: None, self._on_progress, self._on_fit_done)
        self.worker.start()

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.set_fraction(fraction)
        self.progress.set_text(message)

    def _on_fit_done(self, result, error, trace) -> None:
        self.worker = None
        self.progress.hide()
        self.cancel_button.hide()
        self.write_button.set_sensitive(True)
        if error is not None:
            self._show(f"ERROR: {error}")
            return
        out, cc, notes = result
        self._show("\n".join([f"Wrote {out} (CC {cc:.3f})", ""] + notes))
        self._log(f"Structure builder: wrote {out} (density fit CC {cc:.3f})")

    def _cancel(self) -> None:
        if self.worker is not None and self.worker.running:
            self.worker.cancel()
