"""RNAdo's Cluster profiles window: the per-user cluster profiles (mdatk.clusters) that the Amber MD and GROMACS
steps' cluster job scripts use - header, GPU, queues, cushion, how to submit, and the speeds of earlier runs (learnt
from pmemd.cuda .out files).  Kept on this computer in <config>/mdatk/clusters.toml, for every project; saved when
the window closes, and with Save."""

from __future__ import annotations

import copy
from typing import List, Optional

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Pango  # noqa: E402

from .. import clusters  # noqa: E402
from . import widgets as W  # noqa: E402

HEADER_TIP = ("The lines every job script of this cluster starts with (after #!/bin/bash): the scheduler's "
              "directives (#SBATCH, #PBS, #BSUB or #$) and the commands that set up its programs (module "
              "load ...). Leave out the wall time and the queue: MDATK adds them per job. {job_name}, {project}, "
              "{step} and {folder} are filled in, e.g. #SBATCH --output={job_name}.%j")
GPU_TIP = ("The GPU the header asks for, as part of the name pmemd gives the device (e.g. A100 for NVIDIA "
           "A100-SXM4-40GB): only earlier runs on it set the speed. Empty: any GPU.")
QUEUES_TIP = ("The cluster's queues: a job goes into the shortest whose limit holds its estimated wall time (with "
              "the cushion); one longer than every limit is submitted as several jobs in a row (finished stages and "
              "segments are skipped). Directives: the lines that choose the queue, e.g. #SBATCH --partition=gpu "
              "(several separated by ';').")
SPEEDS_TIP = ("Earlier runs: GPU, atoms, time step and seconds/ns from pmemd.cuda .out files. A job's speed is "
              "the median of the nearest runs in size on the profile's GPU, scaled by atoms and time step. Write "
              "cluster jobs (menu) learns the runs of the open plan's cluster jobs; Learn from runs reads any "
              "folder.")


def _number(_column, cell, model, it, data) -> None:
    col, fmt = data
    cell.set_property("text", fmt.format(model[it][col]))


class ClusterProfilesWindow(Gtk.Window):
    def __init__(self, host=None, path: str = ""):
        super().__init__(title="Cluster profiles")
        self.host = host
        self.path = path
        if host is not None and isinstance(host, Gtk.Window):
            self.set_transient_for(host)
        self.set_default_size(980, 640)
        self.profiles: List[clusters.ClusterProfile] = clusters.load_profiles(path)
        self.current: Optional[int] = None
        self.dirty = False
        self._loading = False

        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, margin=8)
        self.add(outer)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        outer.pack_start(left, False, False, 0)
        self.listbox = Gtk.ListBox()
        self.listbox.set_size_request(200, -1)
        self.listbox.connect("row-selected", self._on_row)
        left.pack_start(W.scrolled(self.listbox), True, True, 0)
        buttons = Gtk.Box(spacing=4)
        for text, cb, tip in (("New", self.new_profile, "A new profile (a Slurm starting point)"),
                              ("Duplicate", self.duplicate, "A copy of this profile, e.g. a similar cluster"),
                              ("Delete", self.delete, "Delete this profile")):
            buttons.pack_start(W.button(text, cb, tip), False, False, 0)
        left.pack_start(buttons, False, False, 0)
        left.pack_start(W.hint(f"Kept in {clusters.profiles_path() if not path else path}, for every project."),
                        False, False, 0)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        outer.pack_start(right, True, True, 0)
        self.editor = right
        form = W.Form()
        self.name = form.add("Name", W.entry("e.g. Grace", "The name a step's Cluster profile option gives"))
        self.gpu = form.add("GPU", W.entry("e.g. A100", GPU_TIP))
        self.cushion = form.add("Cushion (%)", W.spin(clusters.CUSHION_RANGE[0], clusters.CUSHION_RANGE[1], 1.0, 0,
                                                      "Added to every estimated wall time (1-20%)"))
        self.time_directive = form.add("Wall-time line", W.entry(
            "from the scheduler, e.g. #SBATCH --time={time}",
            "How the job asks for its wall time; empty: the scheduler's ({time} HH:MM:SS, {hhmm}, {minutes})", 36))
        self.submit = form.add("Submit command", W.entry(
            "from the scheduler, e.g. sbatch --job-name={name} {file}",
            "How the plan's submit script submits a job ({name} job name, {file} job file); empty: from the "
            "header's directives", 36))
        right.pack_start(form, False, False, 0)
        self.header = Gtk.TextView(monospace=True, wrap_mode=Gtk.WrapMode.NONE)
        self.header.set_tooltip_text(HEADER_TIP)
        self.header.get_buffer().connect("changed", self._changed)
        sw = W.scrolled(self.header)
        sw.set_size_request(-1, 150)
        right.pack_start(W.frame("Job header", sw), False, False, 0)

        self.queues = Gtk.ListStore(str, float, str)
        qview = Gtk.TreeView(model=self.queues)
        qview.set_tooltip_text(QUEUES_TIP)
        for col, (title, kind) in enumerate((("Queue", str), ("Max hours", float), ("Directives", str))):
            cell = Gtk.CellRendererText(editable=True)
            cell.connect("edited", self._queue_edited, col)
            column = Gtk.TreeViewColumn(title, cell, text=col)
            if kind is float:
                column.set_cell_data_func(cell, _number, (col, "{:g}"))
            column.set_resizable(True)
            column.set_expand(col == 2)
            qview.append_column(column)
        self.queue_view = qview
        qbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        qsw = W.scrolled(qview)
        qsw.set_size_request(-1, 90)
        qbox.pack_start(qsw, True, True, 0)
        qb = Gtk.Box(spacing=4)
        qb.pack_start(W.button("Add queue", self.add_queue), False, False, 0)
        qb.pack_start(W.button("Remove queue", self.remove_queue), False, False, 0)
        qbox.pack_start(qb, False, False, 0)
        right.pack_start(W.frame("Queues", qbox), False, False, 0)

        self.speeds = Gtk.ListStore(str, str, int, float, float, str)
        sview = Gtk.TreeView(model=self.speeds)
        sview.set_tooltip_text(SPEEDS_TIP)
        for col, title in enumerate(("Date", "GPU", "Atoms", "dt (ps)", "s/ns", "File")):
            cell = Gtk.CellRendererText()
            if col == 5:
                cell.set_property("ellipsize", Pango.EllipsizeMode.START)
            column = Gtk.TreeViewColumn(title, cell, text=col)
            if col in (2, 3, 4):
                column.set_cell_data_func(cell, _number, (col, ("{:,}", "{:g}", "{:.1f}")[col - 2]))
            column.set_resizable(True)
            column.set_expand(col == 5)
            sview.append_column(column)
        sview.get_selection().set_mode(Gtk.SelectionMode.MULTIPLE)
        self.speed_view = sview
        sbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        ssw = W.scrolled(sview)
        ssw.set_size_request(-1, 120)
        sbox.pack_start(ssw, True, True, 0)
        sb = Gtk.Box(spacing=4)
        sb.pack_start(W.button("Learn from runs…", self.learn_from_folder,
                               "Read the speeds of the finished pmemd.cuda runs (.out files) in a folder"),
                      False, False, 0)
        sb.pack_start(W.button("Remove", self.remove_speeds, "Forget the selected runs"), False, False, 0)
        self.speed_note = W.hint("")
        sb.pack_start(self.speed_note, False, False, 6)
        sbox.pack_start(sb, False, False, 0)
        right.pack_start(W.frame("Run speeds", sbox), True, True, 0)

        bottom = Gtk.Box(spacing=6)
        self.status = W.hint("")
        bottom.pack_start(self.status, True, True, 0)
        bottom.pack_end(W.button("Close", self.close_window), False, False, 0)
        bottom.pack_end(W.button("Save", self.save, "Save the profiles now (closing the window saves them too)"),
                        False, False, 0)
        right.pack_end(bottom, False, False, 0)

        for w in (self.name, self.gpu, self.time_directive, self.submit):
            w.connect("changed", self._changed)
        self.cushion.connect("value-changed", self._changed)
        self.connect("delete-event", self._on_delete)
        self._fill_list()
        if self.profiles:
            self.listbox.select_row(self.listbox.get_row_at_index(0))
        self._sensitive()

    # ------------------------------------------------------------ list
    def _fill_list(self) -> None:
        for row in self.listbox.get_children():
            self.listbox.remove(row)
        for p in self.profiles:
            self.listbox.add(W.label(p.name or "(no name)"))
        self.listbox.show_all()

    def _on_row(self, _box, row) -> None:
        self._store_current()
        self.current = row.get_index() if row is not None else None
        self._load_current()
        self._sensitive()

    def _select(self, index: int) -> None:
        self._fill_list()
        self.current = None
        row = self.listbox.get_row_at_index(index)
        if row is not None:
            self.listbox.select_row(row)

    def _sensitive(self) -> None:
        self.editor.set_sensitive(self.current is not None)

    def new_profile(self) -> None:
        self._store_current()
        p = clusters.grace_example()
        p.name = self._unique("New cluster")
        self.profiles.append(p)
        self.dirty = True
        self._select(len(self.profiles) - 1)

    def duplicate(self) -> None:
        if self.current is None:
            return
        self._store_current()
        p = copy.deepcopy(self.profiles[self.current])
        p.name = self._unique(p.name + " copy")
        self.profiles.insert(self.current + 1, p)
        self.dirty = True
        self._select(self.current + 1)

    def delete(self) -> None:
        if self.current is None:
            return
        del self.profiles[self.current]
        index = min(self.current, len(self.profiles) - 1)
        self.current = None
        self.dirty = True
        self._fill_list()
        if index >= 0:
            self.listbox.select_row(self.listbox.get_row_at_index(index))
        else:
            self._load_current()
        self._sensitive()

    def _unique(self, name: str) -> str:
        names = {p.name for p in self.profiles}
        out, k = name, 2
        while out in names:
            out, k = f"{name} {k}", k + 1
        return out

    # ------------------------------------------------------------ editor
    def _changed(self, *_args) -> None:
        if not self._loading:
            self.dirty = True

    def _load_current(self) -> None:
        self._loading = True
        try:
            p = self.profiles[self.current] if self.current is not None else clusters.ClusterProfile("")
            self.name.set_text(p.name)
            self.gpu.set_text(p.gpu)
            self.cushion.set_value(p.cushion)
            self.time_directive.set_text(p.time_directive)
            self.submit.set_text(p.submit)
            self.header.get_buffer().set_text(p.header)
            self.queues.clear()
            for q in p.queues:
                self.queues.append([q.name, float(q.max_hours), q.directives.replace("\n", "; ")])
            self._fill_speeds(p)
        finally:
            self._loading = False

    def _fill_speeds(self, p: clusters.ClusterProfile) -> None:
        self.speeds.clear()
        for b in p.benchmarks:
            self.speeds.append([b.when, b.gpu, int(b.atoms), float(b.dt), float(b.s_per_ns), b.source])
        on = len(p.speeds())
        self.speed_note.set_text(f"{on} on its GPU, {len(p.benchmarks)} in all" if p.benchmarks else "none yet")

    def _store_current(self) -> None:
        """The editor's values into the selected profile."""
        if self.current is None or self.current >= len(self.profiles):
            return
        p = self.profiles[self.current]
        buf = self.header.get_buffer()
        old_name = p.name
        p.name = self.name.get_text().strip()
        p.gpu = self.gpu.get_text().strip()
        p.cushion = clusters.clamp_cushion(self.cushion.get_value())
        p.time_directive = self.time_directive.get_text().strip()
        p.submit = self.submit.get_text().strip()
        p.header = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).rstrip()
        p.queues = [clusters.Queue(row[0].strip(), float(row[1]), "\n".join(
            x.strip() for x in row[2].split(";") if x.strip())) for row in self.queues if row[0].strip()]
        if p.name != old_name:
            row = self.listbox.get_row_at_index(self.current)
            if row is not None:
                row.get_child().set_text(p.name or "(no name)")

    def _queue_edited(self, _cell, path, text, col) -> None:
        if col == 1:
            try:
                value = float(text)
            except ValueError:
                return
            if value <= 0:
                return
            self.queues[path][col] = value
        else:
            self.queues[path][col] = text
        self.dirty = True

    def add_queue(self) -> None:
        self.queues.append(["queue", 48.0, ""])
        self.dirty = True

    def remove_queue(self) -> None:
        model, it = self.queue_view.get_selection().get_selected()
        if it is not None:
            model.remove(it)
            self.dirty = True

    def remove_speeds(self) -> None:
        if self.current is None:
            return
        model, paths = self.speed_view.get_selection().get_selected_rows()
        drop = {model[p][5] + "\0" + str(model[p][4]) for p in paths}
        p = self.profiles[self.current]
        p.benchmarks = [b for b in p.benchmarks if b.source + "\0" + str(b.s_per_ns) not in drop]
        self.dirty = True
        self._fill_speeds(p)

    def learn_from_folder(self, folder: str = "") -> List[clusters.Benchmark]:
        if self.current is None:
            return []
        if not folder:
            dlg = Gtk.FileChooserDialog(title="Learn run speeds from a folder", parent=self,
                                        action=Gtk.FileChooserAction.SELECT_FOLDER)
            dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Learn", Gtk.ResponseType.ACCEPT)
            try:
                if dlg.run() != Gtk.ResponseType.ACCEPT:
                    return []
                folder = dlg.get_filename() or ""
            finally:
                dlg.destroy()
        self._store_current()
        p = self.profiles[self.current]
        new = clusters.learn(p, [folder])
        self.dirty = self.dirty or bool(new)
        self._fill_speeds(p)
        self.status.set_text(f"{len(new)} run speed(s) from {folder}"
                             + ("" if new else " (finished pmemd.cuda .out files only)"))
        return new

    # ------------------------------------------------------------ saving
    def problems(self) -> List[str]:
        names = [p.name for p in self.profiles]
        out = []
        if any(not n for n in names):
            out.append("a profile has no name")
        dup = sorted({n for n in names if n and names.count(n) > 1})
        if dup:
            out.append("names used twice: " + ", ".join(dup))
        for p in self.profiles:
            if not p.queues:
                out.append(f"{p.name}: no queue")
        return out

    def save(self) -> bool:
        self._store_current()
        bad = self.problems()
        if bad:
            self.status.set_text("Not saved: " + "; ".join(bad))
            return False
        try:
            clusters.save_profiles(self.profiles, self.path)
        except OSError as exc:
            self.status.set_text(f"Not saved: {exc}")
            return False
        self.dirty = False
        self.status.set_text(f"Saved {len(self.profiles)} profile(s)")
        if self.host is not None and hasattr(self.host, "append_log"):
            self.host.append_log(f"Cluster profiles saved: {', '.join(p.name for p in self.profiles) or 'none'}")
        return True

    def _on_delete(self, *_args) -> bool:
        self._store_current()
        if self.dirty and not self.save():
            return True  # stays open: the status line says why
        return False

    def close_window(self) -> None:
        if not self._on_delete():
            self.destroy()
