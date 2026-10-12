"""GTK 3 desktop interface for MDATK."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import warnings
from typing import List, Optional

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # noqa: E402

from .. import PROJECT_ABBREV, PROJECT_TITLE, __version__  # noqa: E402
from ..analysis import expected_keys  # noqa: E402
from ..config import ExternalSeries, PanelSpec, Project, ResidueGroup, RMSDTarget  # noqa: E402
from ..progress import Cancelled  # noqa: E402
from ..render.coloring import COLOR_MODES, static_colors  # noqa: E402
from ..render.layout import PRESET_LABELS, PRESETS, n_slots  # noqa: E402
from ..render.orientation import format_angles  # noqa: E402
from ..render.panels import PANEL_TYPES  # noqa: E402
from . import widgets as W  # noqa: E402
from .preview import OrientationView  # noqa: E402
from .worker import Worker  # noqa: E402

log = logging.getLogger(__name__)

APP_ID = "org.mdatk.MDATK"
TIME_UNITS = [("ps", "ps"), ("ns", "ns"), ("us", "µs"), ("frame", "frames")]
X_UNITS = ["ps", "ns", "us", "frame", "frame1"]


class GuiLogHandler(logging.Handler):
    def __init__(self, window: "MainWindow"):
        super().__init__(level=logging.WARNING)
        self.window = window

    def emit(self, record):
        msg = f"{record.levelname}: {record.getMessage()}"
        from .worker import call_soon
        call_soon(self.window.append_log, msg)


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, project: Optional[Project] = None,
                 preset_path: Optional[str] = None):
        super().__init__(application=app, title=PROJECT_TITLE)
        self.set_default_size(1320, 920)
        self.project = project or Project()
        self.preset_path = preset_path
        self.binder = W.Binder()
        self.worker: Optional[Worker] = None
        self.preview_data = None
        self._syncing = False
        self.group_combos: List[Gtk.ComboBoxText] = []
        self.slot_rows: List[dict] = []
        self.busy_widgets: List[Gtk.Widget] = []

        self._build_header()
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(outer)
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        outer.pack_start(paned, True, True, 0)
        self.notebook = Gtk.Notebook()
        self.notebook.set_scrollable(True)
        paned.pack1(self.notebook, resize=True, shrink=False)
        paned.pack2(self._build_log(), resize=False, shrink=True)
        paned.set_position(660)
        outer.pack_start(self._build_action_bar(), False, False, 0)

        self.notebook.append_page(self._tab_input(), Gtk.Label(label="Input & cleaning"))
        self.notebook.append_page(self._tab_groups(), Gtk.Label(label="Residue groups"))
        self.notebook.append_page(self._tab_analyses(), Gtk.Label(label="Analyses"))
        self.notebook.append_page(self._tab_visual(), Gtk.Label(label="Structure view"))
        self.notebook.append_page(self._tab_layout(), Gtk.Label(label="Layout & panels"))
        self.notebook.append_page(self._tab_outputs(), Gtk.Label(label="Outputs"))
        self._install_actions()
        self.load_project(self.project, preset_path)
        self.connect("delete-event", self._on_close)
        self.append_log(f"{PROJECT_TITLE} {__version__} ready. Everything runs locally and offline.")

    # ================================================================ chrome
    def _build_header(self):
        hb = Gtk.HeaderBar()
        hb.set_show_close_button(True)
        hb.set_title(PROJECT_TITLE)
        self.header = hb
        self.set_titlebar(hb)
        hb.pack_start(W.button("Open preset", self.open_preset, "Load a parameter preset (Ctrl+O)",
                               "document-open-symbolic"))
        hb.pack_start(W.button("Save preset", self.save_preset, "Save parameters (Ctrl+S)",
                               "document-save-symbolic"))
        hb.pack_start(W.button("Save as…", lambda: self.save_preset(True),
                               "Save parameters to a new file (Ctrl+Shift+S)"))
        menu = Gtk.Menu()
        for text, cb in (("New (defaults)", self.new_project),
                         ("Create demo system…", self.create_demo),
                         ("About", self.show_about)):
            item = Gtk.MenuItem(label=text)
            item.connect("activate", lambda _w, c=cb: c())
            menu.append(item)
        menu.show_all()
        mb = Gtk.MenuButton()
        mb.set_image(Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON))
        mb.set_popup(menu)
        hb.pack_end(mb)

    def _build_log(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6, margin=4)
        head.pack_start(W.label("<b>Log</b> (pipeline and cpptraj output)", markup=True), False, False, 4)
        head.pack_end(W.button("Clear", lambda: self.log_buffer.set_text("")), False, False, 0)
        box.pack_start(head, False, False, 0)
        self.log_view = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log_buffer = self.log_view.get_buffer()
        self.log_end = self.log_buffer.create_mark("end", self.log_buffer.get_end_iter(), False)
        sw = Gtk.ScrolledWindow()
        sw.set_min_content_height(120)
        sw.add(self.log_view)
        box.pack_start(sw, True, True, 0)
        return box

    def _build_action_bar(self):
        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, margin=8)
        self.render_button = W.button("Render", self.render, "Run cleaning, analyses and all "
                                      "enabled outputs (Ctrl+R)", "media-playback-start-symbolic")
        self.render_button.get_style_context().add_class("suggested-action")
        self.cancel_button = W.button("Cancel", self.cancel, "Stop the running job",
                                      "process-stop-symbolic")
        self.cancel_button.set_sensitive(False)
        self.progress = Gtk.ProgressBar(show_text=True)
        self.progress.set_text("Idle")
        self.progress.set_hexpand(True)
        self.progress.set_valign(Gtk.Align.CENTER)
        bar.pack_start(self.render_button, False, False, 0)
        bar.pack_start(self.cancel_button, False, False, 0)
        bar.pack_start(self.progress, True, True, 0)
        bar.pack_end(W.button("Open output folder", self.open_output_dir, icon="folder-open-symbolic"),
                     False, False, 0)
        self.busy_widgets.append(self.render_button)
        return bar

    def _install_actions(self):
        app = self.get_application()
        for name, cb, accels in (("open", self.open_preset, ["<Control>o"]),
                                 ("save", self.save_preset, ["<Control>s"]),
                                 ("saveas", lambda: self.save_preset(True), ["<Control><Shift>s"]),
                                 ("render", self.render, ["<Control>r"])):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, c=cb: c())
            self.add_action(action)
            if app is not None:
                app.set_accels_for_action(f"win.{name}", accels)

    # ================================================================== tabs
    def _tab_input(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        files = W.Form()
        top = W.file_chooser_button("Select topology", patterns=[
            ("Topologies", ["*.parm7", "*.prmtop", "*.top", "*.pdb", "*.psf", "*.gro"])])
        b.path_entry("inputs.topology", files.add("Topology", top,
                     "Amber .parm7 (preferred), cleaned PDB or another MDAnalysis topology"))
        self.traj_editor = W.ListEditor([("Trajectory file(s), in order", 500)], height=70)
        self.traj_editor.buttons.pack_start(
            W.button("Browse…", self._browse_trajectories, icon="document-open-symbolic"),
            False, False, 0)
        files.add("Trajectories", self.traj_editor,
                  "Amber NetCDF .nc (preferred), DCD, XTC, ... (several files are concatenated)")
        b._add("inputs.trajectories", self.traj_editor,
               lambda: [r[0].strip() for r in self.traj_editor.rows() if r[0].strip()],
               lambda v: self.traj_editor.set_rows([[x] for x in v]))
        box.pack_start(W.frame("Input files", files), False, False, 0)

        fr = W.Form()
        start = b.spin_int("inputs.start", W.spin(0, 10 ** 9, 1))
        stop = b.optional_int("inputs.stop", W.entry("last", width=8))
        step = b.spin_int("inputs.step", W.spin(1, 10 ** 6, 1))
        fr.add_row("Start / stop / step", [start, stop, step],
                   "0-based frames of the (cleaned) trajectory; stop is exclusive (blank = last)")
        fr.add("Time step override (ps)", b.optional_float("inputs.dt_ps", W.entry("from file", width=10)),
               "Leave blank to use the trajectory's time stamps")
        fr.add("Display time unit", b.combo("inputs.time_unit", W.combo(TIME_UNITS)))
        self.ext_editor = W.ListEditor([("Path", 260), ("Label", 80), ("x column", 70),
                                        ("x unit", 60), ("Columns (comma)", 120)], height=60,
                                       defaults=["", "", "", "ps", ""], combos={3: X_UNITS})
        self.ext_editor.buttons.pack_start(W.button("Browse…", self._browse_external),
                                           False, False, 0)
        fr.add("External data (CSV/text)", self.ext_editor,
               "Extra traces (e.g. cpptraj .dat files) interpolated onto the analysed frames")
        b._add("inputs.external", self.ext_editor, self._get_external, self._set_external)
        box.pack_start(W.frame("Frames, time and external data", fr), False, False, 0)

        cl = W.Form()
        self.clean_enable = b.check("cleaning.enabled", W.check(
            "Strip and clean the trajectory with cpptraj before analysis"))
        cl.add_wide(self.clean_enable)
        cl.add("cpptraj executable", b.path_entry("cleaning.cpptraj_path",
               W.file_chooser_button("Select cpptraj")), "Name on PATH or absolute path")
        cl.add("Solvent mask", b.entry("cleaning.solvent_mask", W.entry()))
        cl.add("Ion mask", b.entry("cleaning.ion_mask", W.entry()), "Mg2+ is kept by default")
        cl.add("Extra strip mask", b.entry("cleaning.extra_strip_mask", W.entry("optional")))
        cl.add_row("Imaging", [b.check("cleaning.autoimage", W.check("autoimage (recentre)")),
                               W.label("anchor"), b.entry("cleaning.autoimage_anchor",
                                                          W.entry("default: first molecule", width=18))])
        cl.add_row("Fitting", [b.check("cleaning.fit", W.check("rms fit")), W.label("mask"),
                               b.entry("cleaning.fit_mask", W.entry(width=14)),
                               W.label("reference"),
                               b.combo("cleaning.fit_reference", W.combo(
                                   [("first", "first frame"), ("frame", "frame #"),
                                    ("file", "file")])),
                               b.spin_int("cleaning.reference_frame", W.spin(1, 10 ** 9, 1))])
        cl.add("Reference file", b.path_entry("cleaning.reference_file",
               W.file_chooser_button("Select reference structure")))
        cl.add_wide(b.check("cleaning.remove_box_after_fit", W.check(
            "Remove box information after fitting (a rotated box cannot be stored in NetCDF)")))
        cl.add_row("trajin start / stop / offset",
                   [b.spin_int("cleaning.start", W.spin(1, 10 ** 9, 1)),
                    b.optional_int("cleaning.stop", W.entry("last", width=8)),
                    b.spin_int("cleaning.offset", W.spin(1, 10 ** 6, 1))],
                   "1-based cpptraj trajin arguments")
        extra = Gtk.TextView(monospace=True)
        extra.set_size_request(-1, 50)
        swx = Gtk.ScrolledWindow()
        swx.set_shadow_type(Gtk.ShadowType.IN)
        swx.add(extra)
        b.text_view("cleaning.extra_commands", extra)
        cl.add("Extra cpptraj commands", swx, "Advanced: inserted before trajout")
        btns = Gtk.Box(spacing=6)
        btns.pack_start(W.button("Run cleaning now", self.run_cleaning, icon="system-run-symbolic"),
                        False, False, 0)
        btns.pack_start(W.button("Inspect system", self.inspect_system,
                                 "Print atom/residue/frame counts and check residue groups"),
                        False, False, 0)
        cl.add_wide(btns)
        box.pack_start(W.frame("Trajectory cleaning (cpptraj)", cl), False, False, 0)
        return W.scrolled(box)

    def _get_external(self):
        out = []
        for path, lab, xcol, xunit, cols in self.ext_editor.rows():
            if path.strip():
                out.append(ExternalSeries(path=path.strip(), label=lab.strip(), x_column=xcol.strip(),
                                          x_unit=xunit.strip() or "ps",
                                          columns=[c.strip() for c in cols.split(",") if c.strip()]))
        return out

    def _set_external(self, specs):
        self.ext_editor.set_rows([[s.path, s.label, s.x_column, s.x_unit, ", ".join(s.columns)]
                                  for s in specs])

    def _browse_trajectories(self):
        dlg = Gtk.FileChooserDialog(title="Add trajectory files", transient_for=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        dlg.set_select_multiple(True)
        flt = Gtk.FileFilter()
        flt.set_name("Trajectories")
        for p in ("*.nc", "*.ncdf", "*.dcd", "*.xtc", "*.trr", "*.mdcrd", "*.crd", "*.pdb"):
            flt.add_pattern(p)
        dlg.add_filter(flt)
        if dlg.run() == Gtk.ResponseType.OK:
            for f in dlg.get_filenames():
                self.traj_editor.add_row([f])
        dlg.destroy()

    def _browse_external(self):
        dlg = Gtk.FileChooserDialog(title="Add external data file", transient_for=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        if dlg.run() == Gtk.ResponseType.OK:
            path = dlg.get_filename()
            self.ext_editor.add_row([path, os.path.splitext(os.path.basename(path))[0], "", "ps", ""])
        dlg.destroy()

    # ----------------------------------------------------------------- groups
    def _tab_groups(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        form = W.Form()
        self.group_editor = W.ListEditor([("Name", 110), ("Residues", 200), ("Selection (overrides residues)", 260),
                                          ("Colour", 80)], height=180,
                                         defaults=["NewGroup", "", "", "#6250d6"])
        swatch = Gtk.CellRendererText()
        col = Gtk.TreeViewColumn("", swatch)
        col.set_cell_data_func(swatch, self._swatch)
        col.set_min_width(28)
        self.group_editor.view.append_column(col)
        self.group_editor.buttons.pack_start(W.button("Pick colour…", self._pick_group_color,
                                                      icon="applications-graphics-symbolic"),
                                             False, False, 0)
        self.group_editor.on_change = self._groups_changed
        form.add_wide(self.group_editor)
        form.add_wide(W.hint(
            "Residues accept ranges such as “1-40, 45, 50-60” or chain/segment-qualified "
            "“A:1-40”. A selection is a raw MDAnalysis selection string. Anywhere a selection is "
            "expected you can refer to a group as group:<Name> (e.g. “group:Aptamer and name P”). "
            "Group names must not contain spaces."))
        self.binder._add("groups", self.group_editor, self._get_groups, self._set_groups)
        box.pack_start(W.frame("Residue groups", form), False, False, 0)
        return W.scrolled(box)

    def _swatch(self, _col, cell, model, it, _data):
        rgba = Gdk.RGBA()
        if rgba.parse(model[it][3] or ""):
            cell.set_property("cell-background-rgba", rgba)
        else:
            cell.set_property("cell-background-set", False)

    def _pick_group_color(self):
        k = self.group_editor.selected_index()
        if k is None:
            self.error("Select a group first.")
            return
        dlg = Gtk.ColorChooserDialog(title="Group colour", transient_for=self)
        dlg.set_rgba(W.hex_to_rgba(self.group_editor.store[k][3]))
        if dlg.run() == Gtk.ResponseType.OK:
            self.group_editor.store[k][3] = W.rgba_to_hex(dlg.get_rgba())
            self._groups_changed()
        dlg.destroy()

    def _get_groups(self):
        return [ResidueGroup(name=n.strip(), residues=r.strip(), selection=s.strip(),
                             color=c.strip() or "#2a78d6")
                for n, r, s, c in self.group_editor.rows() if n.strip()]

    def _set_groups(self, groups):
        self.group_editor.set_rows([[g.name, g.residues, g.selection, g.color] for g in groups])
        self._groups_changed()

    def _groups_changed(self):
        names = [r[0].strip() for r in self.group_editor.rows() if r[0].strip()]
        for c in self.group_combos:
            W.set_combo_entry_items(c, names)
        self._recolor_preview()

    def _group_combo(self, path):
        c = W.combo_entry()
        self.group_combos.append(c)
        return self.binder.combo_text(path, c)

    # --------------------------------------------------------------- analyses
    def _tab_analyses(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        gen = W.Form()
        gen.add_wide(b.check("analyses.use_pbc", W.check(
            "Use periodic (minimum-image) distances (not needed after cpptraj imaging)")))
        gen.add("Proximity search radius (Å)", b.spin_float(
            "analyses.proximity_search_radius", W.spin(1, 30, 0.5, 1)),
            "Largest proximity cutoff usable at render time without recomputing")
        box.pack_start(W.frame("General", gen), False, False, 0)

        r = W.Form()
        r.add_wide(b.check("analyses.rmsd.enabled", W.check("Compute RMSD")))
        self.rmsd_editor = W.ListEditor([("Label", 100), ("Selection", 300), ("Fit selection (optional)", 200)],
                                        height=70, defaults=["Target", "group:Aptamer and not name H*", ""])
        r.add("Targets", self.rmsd_editor)
        b._add("analyses.rmsd.targets", self.rmsd_editor,
               lambda: [RMSDTarget(l.strip(), s.strip(), f.strip())
                        for l, s, f in self.rmsd_editor.rows() if s.strip()],
               lambda v: self.rmsd_editor.set_rows([[t.label, t.selection, t.fit_selection] for t in v]))
        r.add_row("Reference frame", [b.spin_int("analyses.rmsd.reference_frame", W.spin(0, 10 ** 9)),
                                      b.check("analyses.rmsd.superposition", W.check("superpose")),
                                      b.check("analyses.rmsd.mass_weighted", W.check("mass weighted"))],
                  "Index into the analysed frames")
        box.pack_start(W.frame("RMSD", r), False, False, 0)

        c = W.Form()
        c.add_wide(b.check("analyses.contacts.enabled", W.check("Compute heavy-atom contacts")))
        c.add_row("Groups", [self._group_combo("analyses.contacts.group_a"), W.label("vs"),
                             self._group_combo("analyses.contacts.group_b")])
        c.add_row("Cutoff (Å)", [b.spin_float("analyses.contacts.cutoff", W.spin(0.5, 20, 0.1, 2)),
                                     b.check("analyses.contacts.heavy_only", W.check("heavy atoms only"))])
        box.pack_start(W.frame("Pairwise contacts", c), False, False, 0)

        h = W.Form()
        h.add_wide(b.check("analyses.hbonds.enabled", W.check("Count inter-group hydrogen bonds")))
        h.add_row("Groups", [self._group_combo("analyses.hbonds.group_a"), W.label("vs"),
                             self._group_combo("analyses.hbonds.group_b")])
        h.add_row("D–A distance (Å) / angle", [
            b.spin_float("analyses.hbonds.distance_cutoff", W.spin(2.0, 5.0, 0.05, 2)),
            b.check("analyses.hbonds.use_angle", W.check("use D–H···A angle ≥")),
            b.spin_float("analyses.hbonds.angle_cutoff", W.spin(90, 180, 1, 0))])
        h.add_row("Donor / acceptor elements", [b.entry("analyses.hbonds.donor_elements", W.entry(width=8)),
                                                b.entry("analyses.hbonds.acceptor_elements", W.entry(width=8))])
        box.pack_start(W.frame("Hydrogen bonds", h), False, False, 0)

        m = W.Form()
        m.add_wide(b.check("analyses.mg.enabled", W.check("Analyse Mg²⁺ coordination")))
        m.add("Mg selection", b.entry("analyses.mg.mg_selection", W.entry()))
        m.add_row("Ligand selection / elements", [b.entry("analyses.mg.ligand_selection", W.entry(width=30)),
                                                  b.entry("analyses.mg.ligand_elements", W.entry(width=6))],
                  "Oxygens of these residue ranges count for Mg–O and O–Mg–O")
        m.add_row("Bridging groups", [self._group_combo("analyses.mg.group_a"), W.label("vs"),
                                      self._group_combo("analyses.mg.group_b")])
        m.add_row("Inner / outer sphere (Å)", [
            b.spin_float("analyses.mg.inner_cutoff", W.spin(1.5, 4.0, 0.05, 2)),
            b.spin_float("analyses.mg.outer_cutoff", W.spin(2.0, 8.0, 0.1, 2)),
            W.label("bridges from"),
            b.combo("analyses.mg.bridge_shell", W.combo([("inner", "inner sphere"),
                                                          ("outer", "inner + outer sphere")]))])
        m.add_row("Coordination geometry", [
            b.check("analyses.mg.geometry", W.check("require octahedral O–Mg–O angles ±")),
            b.spin_float("analyses.mg.angle_tolerance", W.spin(5, 45, 1, 0))])
        m.add_row("Occupancy residues / atoms", [
            b.entry("analyses.mg.occupancy_selection", W.entry("default: ligand selection", width=30)),
            b.entry("analyses.mg.occupancy_elements", W.entry(width=6))])
        box.pack_start(W.frame("O–Mg–O contacts and Mg occupancy", m), False, False, 0)

        s = W.Form()
        s.add_wide(b.check("analyses.stacking.enabled", W.check("Detect π–π stacking")))
        s.add("Selection", b.entry("analyses.stacking.selection", W.entry()))
        s.add_row("Scope", [b.combo("analyses.stacking.scope", W.combo([("all", "all ring pairs"),
                                                                        ("inter", "group A vs group B")])),
                            self._group_combo("analyses.stacking.group_a"), W.label("/"),
                            self._group_combo("analyses.stacking.group_b")])
        s.add_row("Centroid distance / angle / offset / planarity", [
            b.spin_float("analyses.stacking.distance_cutoff", W.spin(2.5, 8, 0.1, 2)),
            b.spin_float("analyses.stacking.angle_cutoff", W.spin(5, 90, 1, 0)),
            b.spin_float("analyses.stacking.offset_cutoff", W.spin(0.5, 5, 0.1, 2)),
            b.spin_float("analyses.stacking.planarity_cutoff", W.spin(0.01, 2, 0.01, 2))],
            "Å, degrees between ring normals, Å lateral offset, Å RMS from plane")
        box.pack_start(W.frame("π–π stacking", s), False, False, 0)

        e = W.Form()
        e.add_wide(b.check("analyses.energy.enabled", W.check(
            "Compute intermolecular interaction energy and net force")))
        e.add_row("Groups", [self._group_combo("analyses.energy.group_a"), W.label("vs"),
                             self._group_combo("analyses.energy.group_b")])
        e.add_row("Cutoff (Å) / dielectric", [
            b.spin_float("analyses.energy.cutoff", W.spin(4, 50, 0.5, 1)),
            b.spin_float("analyses.energy.dielectric", W.spin(1, 100, 0.5, 1)),
            b.check("analyses.energy.distance_dependent", W.check("distance-dependent (εr)"))])
        e.add_row("Parameters", [
            b.check("analyses.energy.use_parm7", W.check("Amber LJ from parm7 when available")),
            W.label("exclude pairs within"),
            b.spin_int("analyses.energy.exclude_bonded", W.spin(0, 3)), W.label("bonds")])
        e.add("Target value (kcal/mol)", b.optional_float("analyses.energy.target_value",
              W.entry("none", width=10)), "Drawn as a reference line on charts and panels")
        box.pack_start(W.frame("Intermolecular energy", e), False, False, 0)
        return W.scrolled(box)

    # ----------------------------------------------------------------- visual
    def _tab_visual(self):
        b = self.binder
        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10, margin=10)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.preview = OrientationView(self._on_preview_rotate)
        pf = Gtk.Frame()
        pf.add(self.preview)
        left.pack_start(pf, True, True, 0)
        row = Gtk.Box(spacing=6)
        lp = W.button("Load preview (reference frame)", self.load_preview,
                      "Reads the first selected frame (cleaned files are used when cached)",
                      "view-refresh-symbolic")
        self.busy_widgets.append(lp)
        row.pack_start(lp, False, False, 0)
        row.pack_start(W.button("Reset orientation", lambda: self._set_rotation([0, 0, 0])), False, False, 0)
        left.pack_start(row, False, False, 0)
        self.angle_label = W.label("")
        self.angle_label.set_selectable(True)
        left.pack_start(self.angle_label, False, False, 0)
        left.pack_start(W.hint("Drag with the left mouse button to rotate. The orientation is set "
                               "on the reference frame and applied to every frame of the video."),
                        False, False, 0)
        outer.pack_start(left, True, True, 0)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        rot = W.Form()
        self.rot_scales = []
        for k, axis in enumerate("XYZ"):
            sc = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -180, 180, 0.5)
            sc.set_digits(1)
            sc.set_value_pos(Gtk.PositionType.RIGHT)
            sc.set_size_request(260, -1)
            sc.connect("value-changed", self._on_scale)
            self.rot_scales.append(sc)
            rot.add(f"Rotation {axis} (°)", sc)
        b._add("visual.rotation", self.rot_scales[0],
               lambda: [round(s.get_value(), 3) for s in self.rot_scales], self._set_rotation)
        right.pack_start(W.frame("Orientation", rot), False, False, 0)

        f = W.Form()
        f.add("Groups shown", b.csv("visual.structure_groups", W.entry("all groups", width=24)),
              "Comma separated group names")
        f.add("Trace atoms (priority)", b.entry("visual.trace_atoms", W.entry(width=16)),
              "First atom found per residue, e.g. P or C4' (CA for proteins)")
        f.add_row("Options", [b.check("visual.align_frames", W.check("superpose frames")),
                              b.check("visual.show_ions", W.check("show Mg²⁺")),
                              b.check("visual.depth_cue", W.check("depth cue"))])
        self.color_mode = b.combo("visual.color_mode", W.combo([(m, m) for m in COLOR_MODES]))
        self.color_mode.connect("changed", lambda *_: self._recolor_preview())
        f.add("Colouring", self.color_mode)
        self.base_color = b.color("visual.base_color", W.color_button())
        self.base_color.connect("color-set", lambda *_: self._recolor_preview())
        f.add_row("Base / ion colour", [self.base_color, b.color("visual.ion_color", W.color_button())])
        self.chain_palette = b.combo_text("visual.chain_palette", W.combo_entry(
            ["tab10", "Set2", "Dark2", "viridis"]))
        self.chain_palette.connect("changed", lambda *_: self._recolor_preview())
        f.add("Chain palette", self.chain_palette)
        f.add("Proximity set", b.entry("visual.proximity_set", W.entry(width=20)),
              "Group name, or sel:<selection>")
        f.add_row("Proximity cutoff (Å) / colour", [
            b.spin_float("visual.proximity_cutoff", W.spin(1, 30, 0.1, 1)),
            b.color("visual.proximity_color", W.color_button())])
        f.add_row("Glow", [b.combo("visual.glow_mode", W.combo([
            ("none", "none"), ("ever", "whole movie if ever in proximity"),
            ("fade", "while in proximity, then fade"), ("current", "only while in proximity")])),
            b.color("visual.glow_color", W.color_button())])
        f.add_row("Fade frames / strength", [b.spin_int("visual.fade_frames", W.spin(0, 10000)),
                                             b.spin_float("visual.glow_strength", W.spin(0, 3, 0.1, 1))])
        f.add_row("Line width / point size", [b.spin_float("visual.line_width", W.spin(0.2, 10, 0.1, 1)),
                                              b.spin_float("visual.point_size", W.spin(1, 200, 1, 0))])
        self.theme_combo = b.combo("visual.theme", W.combo([("light", "light"), ("dark", "dark")]))
        self.theme_combo.connect("changed", lambda *_: self.preview.set_theme(
            self.theme_combo.get_active_id() or "light"))
        f.add_row("Theme / font size", [self.theme_combo, b.spin_float("visual.font_size",
                                                                       W.spin(6, 30, 0.5, 1))])
        f.add_row("Labels", [b.check("visual.show_time_label", W.check("time label")),
                             b.entry("visual.title", W.entry("figure title (optional)", width=20))])
        right.pack_start(W.frame("Appearance", f), False, False, 0)
        sw = W.scrolled(right)
        sw.set_size_request(470, -1)
        outer.pack_start(sw, False, False, 0)
        return outer

    def _set_rotation(self, angles):
        self._syncing = True
        try:
            for sc, a in zip(self.rot_scales, angles):
                sc.set_value(float(a))
        finally:
            self._syncing = False
        self._on_scale()

    def _on_scale(self, *_):
        angles = [s.get_value() for s in self.rot_scales]
        self.angle_label.set_text(format_angles(angles))
        self.preview.set_angles(angles)

    def _on_preview_rotate(self, angles):
        self._syncing = True
        try:
            for sc, a in zip(self.rot_scales, angles):
                sc.set_value(float(a))
        finally:
            self._syncing = False
        self.angle_label.set_text(format_angles(angles))

    def _recolor_preview(self):
        if not self.preview_data or not hasattr(self, "color_mode"):
            return
        groups = self._get_groups()
        mode = self.color_mode.get_active_id() or "groups"
        colors = static_colors(mode, self.preview_data["chain"], self.preview_data["group_index"],
                               [g.color for g in groups], W.rgba_to_hex(self.base_color.get_rgba()),
                               self.chain_palette.get_child().get_text().strip() or "tab10")
        d = self.preview_data
        self.preview.set_structure(d["coords"], d["ions"], d["links"], colors)

    # ----------------------------------------------------------------- layout
    def _tab_layout(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        form = W.Form()
        self.preset_combo = W.combo([(k, PRESET_LABELS.get(k, k)) for k in PRESETS])
        self.preset_combo.connect("changed", lambda *_: self._rebuild_slots())
        form.add("Grid preset", self.preset_combo)
        form.add_wide(W.hint(
            "Data keys look like <analysis>/<output>: time-series panels take one or more series "
            "(comma separated), dual-axis panels exactly two, heatmaps a per-residue matrix, contact "
            "maps a residue map. Options are JSON, e.g. {\"mode\": \"frequency\"} for a contact map, "
            "{\"progressive\": true} or {\"target\": -50} for a time series."))
        self.slot_grid = Gtk.Grid(column_spacing=8, row_spacing=6, margin=8)
        form.add_wide(self.slot_grid)
        prev = Gtk.Box(spacing=6)
        self.preview_pos = W.spin(0, 10 ** 9)
        prev.pack_start(W.label("Preview composite frame"), False, False, 0)
        prev.pack_start(self.preview_pos, False, False, 0)
        pb = W.button("Render still", self.preview_frame,
                      "Analyse (or load cached results) and render one frame of the video",
                      "image-x-generic-symbolic")
        self.busy_widgets.append(pb)
        prev.pack_start(pb, False, False, 0)
        form.add_wide(prev)
        box.pack_start(W.frame("Figure layout", form), False, False, 0)
        self.binder._add("layout", self.preset_combo, self._get_layout, self._set_layout)
        return W.scrolled(box)

    def _key_suggestions(self, panel_type: str) -> List[str]:
        try:
            project = self.collect(validate=False)
        except Exception:  # noqa: BLE001 - suggestions are best effort
            project = self.project
        keys = expected_keys(project)
        kind = PANEL_TYPES[panel_type].data_kind if panel_type in PANEL_TYPES else "series"
        if panel_type == "structure":
            return []
        return keys.get(kind, [])

    def _rebuild_slots(self, specs: Optional[List[PanelSpec]] = None):
        if specs is None:
            specs = self._read_slots()
        for child in self.slot_grid.get_children():
            self.slot_grid.remove(child)
        self.slot_rows = []
        preset = self.preset_combo.get_active_id() or "single"
        for col, text in enumerate(("Slot", "Panel type", "Data keys", "Title", "Options (JSON)")):
            self.slot_grid.attach(W.label(f"<b>{text}</b>", markup=True), col, 0, 1, 1)
        for k in range(n_slots(preset)):
            spec = specs[k] if k < len(specs) else PanelSpec("timeseries", [], "")
            tcombo = W.combo([(t, cls.label) for t, cls in PANEL_TYPES.items()])
            tcombo.set_active_id(spec.type)
            keys = W.combo_entry(self._key_suggestions(spec.type))
            keys.get_child().set_text(", ".join(spec.series))
            keys.set_size_request(260, -1)
            title = W.entry(width=18)
            title.set_text(spec.title)
            opts = W.entry("{}", width=18)
            opts.set_text(json.dumps(spec.options) if spec.options else "")
            tcombo.connect("changed", lambda c, kc=keys: W.set_combo_entry_items(
                kc, self._key_suggestions(c.get_active_id() or "timeseries")))
            r = k + 1
            self.slot_grid.attach(W.label(f"{k + 1}{' (large)' if k == 0 and 'large' in preset else ''}"),
                                  0, r, 1, 1)
            self.slot_grid.attach(tcombo, 1, r, 1, 1)
            self.slot_grid.attach(keys, 2, r, 1, 1)
            self.slot_grid.attach(title, 3, r, 1, 1)
            self.slot_grid.attach(opts, 4, r, 1, 1)
            self.slot_rows.append({"type": tcombo, "keys": keys, "title": title, "opts": opts})
        self.slot_grid.show_all()

    def _read_slots(self) -> List[PanelSpec]:
        specs = []
        for row in self.slot_rows:
            text = row["opts"].get_text().strip()
            try:
                options = json.loads(text) if text else {}
                if not isinstance(options, dict):
                    raise ValueError("options must be a JSON object")
            except ValueError as exc:
                raise ValueError(f"panel options {text!r}: {exc}") from exc
            keys = [k.strip() for k in row["keys"].get_child().get_text().split(",") if k.strip()]
            specs.append(PanelSpec(row["type"].get_active_id() or "timeseries", keys,
                                   row["title"].get_text().strip(), options))
        return specs

    def _get_layout(self):
        from ..config import LayoutParams
        return LayoutParams(preset=self.preset_combo.get_active_id() or "single",
                            panels=self._read_slots())

    def _set_layout(self, layout):
        self.preset_combo.set_active_id(layout.preset)  # "changed" rebuilds from old rows
        self._rebuild_slots(list(layout.panels))

    # ---------------------------------------------------------------- outputs
    def _tab_outputs(self):
        b = self.binder
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        g = W.Form()
        g.add("Output directory", b.path_entry("output.out_dir", W.file_chooser_button(
            "Select output directory", Gtk.FileChooserAction.SELECT_FOLDER)))
        g.add("Cache directory", b.path_entry("output.cache_dir", W.file_chooser_button(
            "Select cache directory", Gtk.FileChooserAction.SELECT_FOLDER)),
            f"Blank: <output>/{PROJECT_ABBREV}_cache (analysis .npz + JSON, cleaned trajectories)")
        g.add_wide(b.check("output.write_params_log", W.check(
            f"Write {PROJECT_ABBREV}_params_log.json (parameters, versions, cache keys) for reproducibility")))
        box.pack_start(W.frame("Locations", g), False, False, 0)

        v = W.Form()
        v.add_wide(b.check("output.video.enabled", W.check("Write MP4 video")))
        v.add("File name", b.entry("output.video.filename", W.entry()))
        v.add_row("Resolution (px) / DPI", [b.spin_int("output.video.width", W.spin(64, 7680, 2)),
                                            W.label("×"),
                                            b.spin_int("output.video.height", W.spin(64, 4320, 2)),
                                            b.spin_int("output.video.dpi", W.spin(10, 600, 1))])
        v.add_row("FPS / frame stride", [b.spin_int("output.video.fps", W.spin(1, 240)),
                                         b.spin_int("output.video.frame_stride", W.spin(1, 10000))])
        v.add_row("Codec / quality (CRF)", [b.combo_text("output.video.codec", W.combo_entry(
            ["libx264", "libx265", "mpeg4"])), b.spin_int("output.video.crf", W.spin(0, 51))])
        v.add("FFmpeg executable", b.path_entry("output.video.ffmpeg_path",
                                                W.file_chooser_button("Select ffmpeg")))
        box.pack_start(W.frame("Video", v), False, False, 0)

        c = W.Form()
        c.add_wide(b.check("output.charts.enabled", W.check("Write standalone time charts")))
        c.add("Series", b.csv("output.charts.series", W.entry("all series", width=40)),
              "Comma separated data keys, blank = every time series")
        c.add_wide(b.check("output.charts.include_heatmaps", W.check("Also chart per-residue heatmaps")))
        fmt_box = Gtk.Box(spacing=6)
        self.fmt_checks = {f: W.check(f.upper()) for f in ("png", "svg", "pdf")}
        for w in self.fmt_checks.values():
            fmt_box.pack_start(w, False, False, 0)
        c.add("Formats", fmt_box)
        b._add("output.charts.formats", fmt_box,
               lambda: [f for f, w in self.fmt_checks.items() if w.get_active()] or ["png"],
               lambda v: [w.set_active(f in v) for f, w in self.fmt_checks.items()])
        c.add_row("DPI / size (in)", [b.spin_int("output.charts.dpi", W.spin(10, 1200, 10)),
                                      b.spin_float("output.charts.width", W.spin(2, 30, 0.5, 1)),
                                      W.label("×"),
                                      b.spin_float("output.charts.height", W.spin(2, 30, 0.5, 1))])
        styles = ["mdatk"]
        try:
            import matplotlib.style as mstyle
            styles += sorted(mstyle.available)
        except Exception:  # noqa: BLE001
            pass
        c.add_row("Style / line colour / width", [
            b.combo_text("output.charts.style", W.combo_entry(styles)),
            b.entry("output.charts.color", W.entry("palette", width=9)),
            b.spin_float("output.charts.line_width", W.spin(0.2, 8, 0.1, 1))])
        c.add("Title template", b.entry("output.charts.title", W.entry("{label}")))
        c.add_row("Axis labels", [b.entry("output.charts.xlabel", W.entry("auto", width=18)),
                                  b.entry("output.charts.ylabel", W.entry("auto", width=18))])
        c.add_row("Smoothing (frames)", [b.spin_int("output.charts.smoothing", W.spin(0, 100000)),
                                         b.check("output.charts.show_target", W.check("show target line"))])
        c.add("File prefix", b.entry("output.charts.prefix", W.entry()))
        box.pack_start(W.frame("Time charts", c), False, False, 0)

        p = W.Form()
        p.add_wide(b.check("output.pdb.enabled", W.check("Export frames as PDB")))
        p.add("Frames", b.entry("output.pdb.frames", W.entry("e.g. 0, 10, 20-50:10, last")))
        p.add("Times", b.entry("output.pdb.times", W.entry("e.g. 1.5ns, 2000ps")),
              "Nearest frame is used; unitless values use the display time unit")
        p.add("Atom subset", b.entry("output.pdb.selection", W.entry("all atoms; or e.g. Aptamer, Target")),
              "Group names (comma separated) or an MDAnalysis selection")
        p.add("File prefix", b.entry("output.pdb.prefix", W.entry()))
        row = Gtk.Box(spacing=6)
        eb = W.button("Export PDBs now", self.export_pdbs, icon="document-save-symbolic")
        rb = W.button("Re-render video + charts from cache", self.rerender_from_cache,
                      "Uses the last cached analyses (no trajectory access); handy after changing "
                      "colours, layout or chart style")
        self.busy_widgets += [eb, rb]
        row.pack_start(eb, False, False, 0)
        row.pack_start(rb, False, False, 0)
        p.add_wide(row)
        box.pack_start(W.frame("Frame PDBs", p), False, False, 0)
        return W.scrolled(box)

    # ============================================================ model sync
    def load_project(self, project: Project, path: Optional[str] = None) -> None:
        self.project = project
        self.preset_path = path
        self.binder.load(project)
        self._on_scale()
        self.preview.set_theme(project.visual.theme)
        self.header.set_subtitle(path or "unsaved parameters")

    def collect(self, validate: bool = True) -> Project:
        project = self.project.copy()
        errors = self.binder.store(project)
        if errors:
            raise ValueError("\n".join(errors))
        if validate:
            problems = project.validate()
            if problems:
                raise ValueError("\n".join(problems))
        return project

    # ================================================================ actions
    def new_project(self):
        self.load_project(Project(), None)
        self.preview_data = None
        self.preview.coords = None
        self.preview.set_message("Load a topology and trajectory, then click “Load preview”.")

    def open_preset(self):
        dlg = Gtk.FileChooserDialog(title="Open parameter preset", transient_for=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        flt = Gtk.FileFilter()
        flt.set_name("Presets / parameter logs (*.json)")
        flt.add_pattern("*.json")
        dlg.add_filter(flt)
        resp = dlg.run()
        path = dlg.get_filename()
        dlg.destroy()
        if resp == Gtk.ResponseType.OK and path:
            self.open_path(path)

    def open_path(self, path: str):
        try:
            project = Project.load(path)
        except (OSError, ValueError) as exc:
            self.error(f"Could not open {path}", str(exc))
            return
        self.load_project(project, path)
        self.append_log(f"Loaded parameters from {path}")

    def save_preset(self, save_as: bool = False):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Some parameters are invalid", str(exc))
            return
        path = self.preset_path
        if save_as or not path or not path.endswith(".json"):
            dlg = Gtk.FileChooserDialog(title="Save parameter preset", transient_for=self,
                                        action=Gtk.FileChooserAction.SAVE)
            dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
            dlg.set_do_overwrite_confirmation(True)
            dlg.set_current_name(f"{PROJECT_ABBREV}_preset.json")
            resp = dlg.run()
            path = dlg.get_filename()
            dlg.destroy()
            if resp != Gtk.ResponseType.OK or not path:
                return
        try:
            project.save(path)
        except OSError as exc:
            self.error("Could not save preset", str(exc))
            return
        self.project = project
        self.preset_path = path
        self.header.set_subtitle(path)
        self.append_log(f"Saved parameters to {path}")

    def create_demo(self):
        dlg = Gtk.FileChooserDialog(title="Folder for the synthetic demo system", transient_for=self,
                                    action=Gtk.FileChooserAction.SELECT_FOLDER)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Create", Gtk.ResponseType.OK)
        resp = dlg.run()
        folder = dlg.get_filename()
        dlg.destroy()
        if resp != Gtk.ResponseType.OK or not folder:
            return

        def job(rep):
            from ..demo import build_demo
            rep.log(f"Writing synthetic RNA aptamer/target system to {folder} ...")
            return build_demo(folder)

        def done(result):
            self.append_log(f"Demo system ready: {result['preset']}")
            self.open_path(result["preset"])

        self.start_job("Creating demo", job, done)

    def show_about(self):
        dlg = Gtk.AboutDialog(transient_for=self, modal=True)
        dlg.set_program_name(PROJECT_TITLE)
        dlg.set_version(__version__)
        dlg.set_comments("Offline analysis and multi-panel video rendering of molecular dynamics "
                         "trajectories of RNA systems.")
        dlg.set_license_type(Gtk.License.GPL_3_0)
        dlg.run()
        dlg.destroy()

    def open_output_dir(self):
        try:
            project = self.collect(validate=False)
        except ValueError:
            project = self.project
        path = project.out_dir
        if not os.path.isdir(path):
            self.error("The output directory does not exist yet.", path)
            return
        try:
            Gio.AppInfo.launch_default_for_uri(GLib.filename_to_uri(path, None), None)
        except GLib.Error as exc:
            self.error("Could not open the folder", str(exc))

    # --------------------------------------------------------------- jobs
    def _pipeline(self, project, rep):
        from ..pipeline import Pipeline
        return Pipeline(project, rep)

    def render(self):
        try:
            project = self.collect()
        except ValueError as exc:
            self.error("Please fix these parameters before rendering", str(exc))
            return

        def job(rep):
            return self._pipeline(project, rep).run()

        def done(outputs):
            s = outputs.summary()
            lines = []
            if s["video"]:
                lines.append(f"Video: {s['video']}")
            if s["charts"]:
                lines.append(f"Charts: {len(s['charts'])} file(s)")
            if s["pdbs"]:
                lines.append(f"PDB files: {len(s['pdbs'])}")
            if s["params_log"]:
                lines.append(f"Parameter log: {s['params_log']}")
            self.info("Render finished", "\n".join(lines) or "Nothing was enabled.")

        self.start_job("Rendering", job, done)

    def run_cleaning(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return
        project.cleaning.enabled = True

        def job(rep):
            return self._pipeline(project, rep).resolve_inputs()

        def done(inputs):
            c = inputs.cleaned
            self.info("Cleaning finished" if not c.reused else "Cleaned files reused",
                      f"Topology: {inputs.topology}\nTrajectory: {inputs.trajectories[0]}\n"
                      f"cpptraj script: {c.script}")

        self.start_job("Cleaning", job, done)

    def inspect_system(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return

        def job(rep):
            from ..io.loader import load_universe, system_summary
            from ..selections import select
            pipe = self._pipeline(project, rep)
            inputs = pipe.cleaned_if_available()
            u = load_universe(inputs.topology, inputs.trajectories, project.inputs.dt_ps)
            info = system_summary(u)
            rep.log(f"System ({'cleaned' if inputs.topology != project.inputs.topology else 'raw'} "
                    f"files): {json.dumps(info)}")
            for g in project.groups:
                try:
                    ag = select(u, f"group:{g.name}", project.groups)
                    rep.log(f"  group {g.name}: {ag.n_atoms} atoms, {ag.residues.n_residues} residues "
                            f"({ag.residues[0].resname}{ag.residues[0].resid}–"
                            f"{ag.residues[-1].resname}{ag.residues[-1].resid})")
                except Exception as exc:  # noqa: BLE001
                    rep.log(f"  group {g.name}: ERROR {exc}")
            return info

        self.start_job("Inspecting", job, lambda _r: None)

    def load_preview(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return
        self.preview.set_message("Loading reference frame …")

        def job(rep):
            return self._pipeline(project, rep).structure_preview()

        def done(data):
            self.preview_data = data
            self._recolor_preview()
            self.preview.set_angles([s.get_value() for s in self.rot_scales])
            self.append_log(f"Preview loaded: {len(data['coords'])} residues "
                            f"({'cleaned' if data['cleaned'] else 'raw'} files).")

        self.start_job("Loading preview", job, done)

    def preview_frame(self):
        try:
            project = self.collect()
        except ValueError as exc:
            self.error("Please fix these parameters first", str(exc))
            return
        pos = int(self.preview_pos.get_value())

        def job(rep):
            pipe = self._pipeline(project, rep)
            ds = pipe.analyze(pipe.required_analyses(video=True, charts=False),
                              reporter=rep.sub(0, 0.9))
            path = os.path.join(tempfile.gettempdir(), f"{PROJECT_ABBREV}_still_{os.getpid()}.png")
            return pipe.render_preview(ds, min(pos, ds.n_frames - 1), path)

        self.start_job("Rendering still", job, self.show_image)

    def export_pdbs(self):
        try:
            project = self.collect()
        except ValueError as exc:
            self.error("Please fix these parameters first", str(exc))
            return

        def job(rep):
            os.makedirs(project.out_dir, exist_ok=True)
            return self._pipeline(project, rep).export_pdbs()

        self.start_job("Exporting PDBs", job,
                       lambda paths: self.info("PDB export finished", "\n".join(paths)))

    def rerender_from_cache(self):
        try:
            project = self.collect(validate=False)
        except ValueError as exc:
            self.error("Invalid parameters", str(exc))
            return

        def job(rep):
            pipe = self._pipeline(project, rep)
            os.makedirs(project.out_dir, exist_ok=True)
            ds = pipe.load_cached_dataset()
            out = []
            if project.output.video.enabled:
                out.append(pipe.render_video(ds, rep.sub(0, 0.85)))
            if project.output.charts.enabled:
                out += pipe.render_charts(ds, rep.sub(0.85, 1.0))
            return out

        self.start_job("Re-rendering from cache", job,
                       lambda paths: self.info("Re-render finished", f"{len(paths)} file(s) written"))

    # ------------------------------------------------------------ job runner
    def start_job(self, title: str, job, on_success):
        if self.worker is not None and self.worker.running:
            self.error("Another job is still running.")
            return
        self.progress.set_fraction(0.0)
        self.progress.set_text(f"{title} …")
        for w in self.busy_widgets:
            w.set_sensitive(False)
        self.cancel_button.set_sensitive(True)
        self.append_log(f"--- {title} ---")

        def done(result, exc, tb):
            for w in self.busy_widgets:
                w.set_sensitive(True)
            self.cancel_button.set_sensitive(False)
            self.worker = None
            if isinstance(exc, Cancelled):
                self.progress.set_text("Cancelled")
                self.append_log("Cancelled.")
                return
            if exc is not None:
                self.progress.set_text("Failed")
                if tb:
                    self.append_log(tb.rstrip())
                self.error(f"{title} failed", str(exc))
                return
            self.progress.set_fraction(1.0)
            self.progress.set_text(f"{title}: done")
            on_success(result)

        self.worker = Worker(job, self.append_log, self._on_progress, done)
        self.worker.start()

    def _on_progress(self, fraction: float, message: str):
        self.progress.set_fraction(min(1.0, max(0.0, fraction)))
        if message:
            self.progress.set_text(f"{fraction * 100:4.0f}%  {message}")

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.progress.set_text("Cancelling …")

    def _on_close(self, *_):
        if self.worker is not None and self.worker.running:
            self.worker.cancel()
        return False

    # -------------------------------------------------------------- feedback
    def append_log(self, text: str):
        buf = self.log_buffer
        buf.insert(buf.get_end_iter(), text + "\n")
        lines = buf.get_line_count()
        if lines > 6000:
            buf.delete(buf.get_start_iter(), buf.get_iter_at_line(lines - 5000))
        buf.move_mark(self.log_end, buf.get_end_iter())
        self.log_view.scroll_mark_onscreen(self.log_end)

    def error(self, text: str, detail: str = ""):
        self.append_log(f"ERROR: {text}{(': ' + detail) if detail else ''}")
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.ERROR,
                                buttons=Gtk.ButtonsType.OK, text=text)
        if detail:
            dlg.format_secondary_text(detail[:3000])
        dlg.run()
        dlg.destroy()

    def info(self, text: str, detail: str = ""):
        self.append_log(f"{text}{(': ' + detail) if detail else ''}")
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.INFO,
                                buttons=Gtk.ButtonsType.OK, text=text)
        if detail:
            dlg.format_secondary_text(detail[:3000])
        dlg.run()
        dlg.destroy()

    def show_image(self, path: str):
        dlg = Gtk.Dialog(title=f"{PROJECT_ABBREV} preview – {os.path.basename(path)}",
                         transient_for=self, modal=False)
        dlg.add_button("Close", Gtk.ResponseType.CLOSE)
        from gi.repository import GdkPixbuf
        pix = GdkPixbuf.Pixbuf.new_from_file(path)
        screen_w = 1200
        if pix.get_width() > screen_w:
            h = int(pix.get_height() * screen_w / pix.get_width())
            pix = pix.scale_simple(screen_w, h, GdkPixbuf.InterpType.BILINEAR)
        dlg.get_content_area().pack_start(Gtk.Image.new_from_pixbuf(pix), True, True, 0)
        dlg.connect("response", lambda d, _r: d.destroy())
        dlg.show_all()
        self.append_log(f"Still frame written to {path}")


class MDATKApplication(Gtk.Application):
    def __init__(self, preset: Optional[str] = None):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.preset = preset
        self.window: Optional[MainWindow] = None

    def do_activate(self):
        if self.window is None:
            project, path = None, None
            if self.preset:
                try:
                    project, path = Project.load(self.preset), self.preset
                except (OSError, ValueError) as exc:
                    print(f"warning: could not load {self.preset}: {exc}", file=sys.stderr)
            self.window = MainWindow(self, project, path)
            logging.getLogger("mdatk").addHandler(GuiLogHandler(self.window))
        self.window.show_all()
        self.window.present()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    warnings.filterwarnings("ignore", module="MDAnalysis")
    preset = argv[0] if argv else None
    app = MDATKApplication(preset)
    return app.run([sys.argv[0]])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
