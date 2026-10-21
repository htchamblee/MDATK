"""PDB clean-up window: the GUI of :mod:`mdatk.pdbtools` ('mdatk pdb clean' / 'check')."""

from __future__ import annotations

import json
import os
from typing import Optional

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from .. import pdbtools  # noqa: E402
from ..annotate import annotate  # noqa: E402
from ..citations import keys_for  # noqa: E402
from ..projectfile import logged  # noqa: E402
from . import widgets as W  # noqa: E402

PDB_FILES = ("PDB files", ["*.pdb", "*.ent", "*.pdb1"])
FASTA_FILES = ("FASTA files", ["*.fasta", "*.fa", "*.fas"])
SCHEME_LABELS = [
    ("", "keep the names"),
    ("amber", "Amber (L: LA/LC/LG/LU + 5'/3', Dudek & Trylska)"),
    ("ccd", "CCD for Coot / Phenix (L: 0A/0C/0G/0U + 0A5/0A3)"),
    ("rosetta", "Rosetta (L: 0A/0C/0G/0U, all ATOM)"),
    ("pdb", "PDB deposition (L: 0A..., non-standard as HETATM)"),
]
RECORD_LABELS = [("", "follow the name scheme"), ("keep", "keep ATOM / HETATM"),
                 ("rosetta", "all polymer residues as ATOM (Rosetta)"),
                 ("pdb", "non-standard residues as HETATM (PDB)")]
ENGINE_LABELS = [("auto", "Barnaba when installed, else builtin"),
                 ("builtin", "builtin (canonical pairs only)"),
                 ("barnaba", "Barnaba (Leontis-Westhof classes, stacking)")]
CHAIN_LABELS = [("keep", "keep the chain ids"), ("reference", "copy from a reference PDB"),
                ("list", "ids per TER segment (list)"), ("assign", "A, B, C … per TER segment")]


class PDBToolsWindow(Gtk.Window):
    def __init__(self, main):
        super().__init__(title="PDB clean-up", transient_for=main)
        self.main = main
        self._auto_output = ""
        self.set_default_size(980, 720)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(outer)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        outer.pack_start(paned, True, True, 0)
        paned.pack1(self._form(), False, False)
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=8)
        right.pack_start(W.label("<b>Report</b>", markup=True), False, False, 0)
        self.report = Gtk.TextView(monospace=True, editable=False)
        self.report.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        sw = Gtk.ScrolledWindow()
        sw.set_shadow_type(Gtk.ShadowType.IN)
        sw.add(self.report)
        right.pack_start(sw, True, True, 0)
        paned.pack2(right, True, False)
        paned.set_position(520)
        bar = Gtk.Box(spacing=6, margin=8)
        bar.pack_end(W.button("Close", self.destroy), False, False, 0)
        run = W.button("Write cleaned PDB", self.run_clean, "Apply the options and write the output file",
                       "document-save-symbolic")
        run.get_style_context().add_class("suggested-action")
        bar.pack_end(run, False, False, 0)
        bar.pack_end(W.button("Check input", self.run_check, "Chains, sequences, residue names and the "
                              "sugar chirality of every nucleotide (nothing is written)"), False, False, 0)
        outer.pack_start(bar, False, False, 0)
        self._sync()

    # ------------------------------------------------------------------ form
    def _form(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        f = W.Form()
        self.input = W.file_chooser_button("Select a PDB file", patterns=[PDB_FILES])
        self.input.entry.connect("changed", self._on_input)
        f.add("Input PDB", self.input, "PDB to repair (ChimeraX-damaged lines are fixed automatically)")
        self.output = W.file_chooser_button("Output PDB", Gtk.FileChooserAction.SAVE, patterns=[PDB_FILES])
        f.add("Output PDB", self.output)
        box.pack_start(W.frame("Files", f), False, False, 0)

        n = W.Form()
        self.scheme = W.combo(SCHEME_LABELS)
        self.scheme.set_active_id("")
        n.add("Residue names", self.scheme, "Base from the name, chirality from the sugar geometry (or the "
              "ranges below), 5'/3' variants from the segment ends")
        self.terminals = W.check("5'/3' terminal names (LA5/LA3, A5/A3 …)")
        self.terminals.set_active(True)
        self.terminals.set_tooltip_text("Off: chain ends keep the internal names (LA, A), for leap libraries "
                                        "that map terminals with addPdbResMap")
        n.add_wide(self.terminals)
        self.detect = W.check("Detect L-nucleotides from the sugar chirality (C1', C3', C4')")
        self.detect.set_active(True)
        n.add_wide(self.detect)
        self.l_ranges = W.entry("e.g. C:45-73 or C (whole chain)")
        n.add("Force L for", self.l_ranges, "Residue ranges that are L whatever the geometry says "
              "(overrides the detection)")
        self.records = W.combo(RECORD_LABELS)
        self.records.set_active_id("")
        n.add("ATOM / HETATM", self.records)
        self.keep_conect = W.check("Keep CONECT records (stripped until final deposition processing)")
        n.add_wide(self.keep_conect)
        self.strip_5p = W.check("Remove 5'-terminal phosphates (for Amber)")
        self.strip_5p.set_tooltip_text("Drop P/OP1/OP2/OP3 of every chain's first nucleotide: Amber's 5' units "
                                       "(G5, LG5 ...) have a 5'-OH. Hydrogenated residues get HO5'.")
        self.scheme.connect("changed", lambda *_: self.strip_5p.set_active(
            self.strip_5p.get_active() or self.scheme.get_active_id() == "amber"))
        n.add_wide(self.strip_5p)
        box.pack_start(W.frame("Names and records", n), False, False, 0)

        c = W.Form()
        self.chain_mode = W.combo(CHAIN_LABELS)
        self.chain_mode.set_active_id("keep")
        self.chain_mode.connect("changed", lambda *_: self._sync())
        c.add("Chain ids", self.chain_mode, "cpptraj writes PDBs without chain ids: copy them from the "
              "structure the topology was built from, give one id per TER segment, or assign A, B, C …")
        self.reference = W.file_chooser_button("Reference PDB", patterns=[PDB_FILES])
        c.add("Reference PDB", self.reference)
        self.copy_numbers = W.check("Also copy the residue numbers")
        c.add_wide(self.copy_numbers)
        self.chain_ids = W.entry("e.g. A,B,C")
        c.add("Ids per segment", self.chain_ids)
        self.split_segments = W.check("Give every TER segment its own new chain id")
        c.add_wide(self.split_segments)
        self.split_at = W.entry("e.g. A:31")
        c.add("Split before", self.split_at, "Start a new chain (new id) at these residues")
        self.combine = W.entry("e.g. A,B")
        self.combine_into = W.entry("first", width=4)
        c.add_row("Combine chains / into", [self.combine, self.combine_into])
        self.sort_chains = W.check("Write the chains in alphabetical order")
        c.add_wide(self.sort_chains)
        box.pack_start(W.frame("Chains", c), False, False, 0)

        r = W.Form()
        self.renumber = W.check("Renumber residues (continuous across chains)")
        self.renumber.connect("toggled", lambda *_: self._sync())
        r.add_wide(self.renumber)
        self.start = W.spin(-999, 9999, 1)
        self.start.set_value(1)
        r.add("First number", self.start)
        self.per_chain = W.check("Restart the numbering in every chain")
        r.add_wide(self.per_chain)
        self.keep_gaps = W.check("Keep numbering gaps")
        r.add_wide(self.keep_gaps)
        self.seqres = W.check("Write SEQRES records")
        r.add_wide(self.seqres)
        self.fasta = W.file_chooser_button("FASTA file", Gtk.FileChooserAction.SAVE, patterns=[FASTA_FILES])
        r.add("FASTA (optional)", self.fasta)
        box.pack_start(W.frame("Numbering and sequence", r), False, False, 0)

        a = W.Form()
        self.annotate = W.check("Annotate base pairs as REMARK 950 records")
        self.annotate.set_tooltip_text("Base pairs, stacking and dot-bracket written into the header of the "
                                       "output (as 'mdatk pdb annotate')")
        self.annotate.connect("toggled", lambda *_: self._sync())
        a.add_wide(self.annotate)
        from ..secstruct import barnaba_available
        labels = ENGINE_LABELS if barnaba_available() else [
            (k, text + " - not installed" if k == "barnaba" else text) for k, text in ENGINE_LABELS]
        self.ann_engine = W.combo(labels)
        self.ann_engine.set_active_id("auto")
        a.add("Engine", self.ann_engine, "Barnaba is optional (pip install barnaba); the builtin engine finds "
              "Watson-Crick and G-U wobble pairs from H-bond distances")
        self.ann_pairs = W.check("Base pairs (PAIR)")
        self.ann_stacking = W.check("Stacking (STACK, Barnaba only)")
        self.ann_dbn = W.check("Sequence and dot-bracket (SEQ / DBN)")
        for w in (self.ann_pairs, self.ann_stacking, self.ann_dbn):
            w.set_active(True)
            a.add_wide(w)
        box.pack_start(W.frame("Annotation", a), False, False, 0)
        sw = W.scrolled(box)
        sw.set_size_request(500, -1)
        return sw

    def _sync(self) -> None:
        mode = self.chain_mode.get_active_id()
        self.reference.set_sensitive(mode == "reference")
        self.copy_numbers.set_sensitive(mode == "reference")
        self.chain_ids.set_sensitive(mode == "list")
        on = self.renumber.get_active()
        for w in (self.start, self.per_chain, self.keep_gaps):
            w.set_sensitive(on)
        on = self.annotate.get_active()
        for w in (self.ann_engine, self.ann_pairs, self.ann_stacking, self.ann_dbn):
            w.set_sensitive(on)

    def _on_input(self, *_):
        """Suggest <input>_clean.pdb while the output is empty or still the previous suggestion."""
        path = self.input.entry.get_text().strip()
        current = self.output.entry.get_text().strip()
        if current and current != self._auto_output:
            return  # chosen by hand
        stem, ext = os.path.splitext(path)
        self._auto_output = f"{stem}_clean{ext or '.pdb'}" if path else ""
        self.output.entry.set_text(self._auto_output)

    # --------------------------------------------------------------- actions
    def options(self) -> dict:
        def ids(text):
            return [c.strip() for c in text.split(",") if c.strip()] or None

        mode = self.chain_mode.get_active_id()
        return dict(
            scheme=self.scheme.get_active_id() or None, records=self.records.get_active_id() or None,
            strip_conect=not self.keep_conect.get_active(), detect_l=self.detect.get_active(),
            l_ranges=self.l_ranges.get_text().strip() or None,
            chain_reference=self.reference.entry.get_text().strip() or None if mode == "reference" else None,
            chain_ids=ids(self.chain_ids.get_text()) if mode == "list" else None,
            assign_chains=mode == "assign", copy_numbers=mode == "reference" and self.copy_numbers.get_active(),
            split_segments=self.split_segments.get_active(), split_at=self.split_at.get_text().strip() or None,
            combine=ids(self.combine.get_text()),
            combine_into=self.combine_into.get_text().strip() or None,
            sort_chains=self.sort_chains.get_active(), renumber=self.renumber.get_active(),
            start=int(self.start.get_value()), per_chain=self.per_chain.get_active(),
            keep_gaps=self.keep_gaps.get_active(), seqres=self.seqres.get_active(),
            make_fasta=bool(self.fasta.entry.get_text().strip()), terminal_names=self.terminals.get_active(),
            strip_5p=self.strip_5p.get_active())

    def annotation_options(self) -> Optional[dict]:
        """Keyword arguments of :func:`mdatk.annotate.annotate`, or None when annotation is off."""
        if not self.annotate.get_active():
            return None
        return dict(pairs=self.ann_pairs.get_active(), stacking=self.ann_stacking.get_active(),
                    dotbracket=self.ann_dbn.get_active(), engine=self.ann_engine.get_active_id() or "auto")

    def _input(self) -> Optional[str]:
        path = self.input.entry.get_text().strip()
        if not path or not os.path.isfile(path):
            self._show("Choose an existing input PDB file first.")
            return None
        return path

    def _show(self, text: str) -> None:
        self.report.get_buffer().set_text(text)

    def run_check(self) -> None:
        path = self._input()
        if not path:
            return
        try:
            s = pdbtools.read_pdb(path)
            notes = pdbtools.assign_chirality(s, detect=True, l_ranges=self.l_ranges.get_text().strip() or None)
        except (OSError, pdbtools.PDBError) as exc:
            self._show(f"ERROR: {exc}")
            return
        info = pdbtools.summary(s)
        lines = [f"{os.path.basename(path)}: {info['segments']} TER segment(s), "
                 f"{info['conect_records']} CONECT record(s)", ""]
        for chain, c in info["chains"].items():
            chir = ", ".join(f"{n} {k}" for k, n in sorted(c["chirality"].items()))
            lines.append(f"Chain {chain}: {c['residues']} residues ({c['numbers']}), names "
                         f"{' '.join(c['names'])}; chirality {chir or '-'}")
            lines.append(f"  {c['sequence']}")
        if info["other_residues"]:
            lines.append("Other residues: " + json.dumps(info["other_residues"]))
        lines += [""] + list(info["notes"]) + notes
        self._show("\n".join(lines))

    def run_clean(self) -> None:
        path = self._input()
        if not path:
            return
        out = self.output.entry.get_text().strip()
        if not out:
            self._show("Choose an output file.")
            return
        if os.path.abspath(out) == os.path.abspath(path):
            self._show("The output must not overwrite the input file.")
            return
        opts = self.options()
        ann_opts = self.annotation_options()
        fasta_path = self.fasta.entry.get_text().strip()
        project_path = getattr(self.main, "project_path", None)
        try:
            with logged(project_path, "pdb clean", f"PDB clean-up of {os.path.basename(path)}") as entry:
                res = pdbtools.clean(path, **opts)
                text, notes = res.text, list(res.notes)
                if ann_opts is not None:
                    ann = annotate(text, **ann_opts)
                    text = ann.text
                    notes += [n for n in ann.notes if n not in notes]
                    if ann.engine == "barnaba":
                        entry.citations = keys_for(["barnaba"])
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(text)
                entry.files = [out]
                if fasta_path and res.fasta:
                    with open(fasta_path, "w", encoding="utf-8") as fh:
                        fh.write(res.fasta)
                    entry.files.append(fasta_path)
                entry.summary = "\n".join(notes)
        except (OSError, ValueError, ImportError, RuntimeError, pdbtools.PDBError) as exc:
            self._show(f"ERROR: {exc}")
            return
        self._show("\n".join([f"Wrote {out}"] + ([f"Wrote {fasta_path}"] if fasta_path else []) + [""]
                             + notes))
        if hasattr(self.main, "append_log"):
            self.main.append_log(f"PDB clean-up: wrote {out}")
