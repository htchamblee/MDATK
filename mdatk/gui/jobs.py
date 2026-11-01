"""Background jobs with a progress bar and a Cancel button, and the questions around running generated
scripts.  The analysis window (renders, cleaning) and RNAdo (generated scripts) share them."""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from ..progress import Cancelled  # noqa: E402
from .worker import Worker  # noqa: E402


class JobControl:
    """One background job at a time: the progress bar shows it, the Cancel button stops it and the
    ``busy_widgets`` are greyed out while it runs."""

    def __init__(self, progress: Gtk.ProgressBar, cancel_button: Gtk.Button, log: Callable[[str], None],
                 error: Callable[[str, str], None], busy_widgets: Optional[List[Gtk.Widget]] = None):
        self.progress = progress
        self.cancel_button = cancel_button
        self.log = log
        self.error = error
        self.busy_widgets: List[Gtk.Widget] = busy_widgets if busy_widgets is not None else []
        self.worker: Optional[Worker] = None
        cancel_button.set_sensitive(False)

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.running

    def start(self, title: str, job: Callable[[Any], Any], on_success: Callable[[Any], None],
              on_failure: Optional[Callable[[BaseException], None]] = None,
              on_log: Optional[Callable[[str], None]] = None, expected: Tuple[type, ...] = ()) -> bool:
        """Run ``job(reporter)`` in the background; ``on_success(result)`` on the main loop.  A failure
        is logged with its traceback (not for the ``expected`` exceptions, e.g. a failed script) and
        reported (``on_failure(exc)`` first).  False when another job still runs."""
        if self.running:
            self.error("Another job is still running.", "")
            return False
        self.progress.set_fraction(0.0)
        self.progress.set_text(f"{title} …")
        for w in self.busy_widgets:
            w.set_sensitive(False)
        self.cancel_button.set_sensitive(True)
        self.log(f"--- {title} ---")

        def done(result, exc, tb):
            for w in self.busy_widgets:
                w.set_sensitive(True)
            self.cancel_button.set_sensitive(False)
            self.worker = None
            if isinstance(exc, Cancelled):
                self.progress.set_text("Cancelled")
                self.log("Cancelled.")
                return
            if exc is not None:
                self.progress.set_text("Failed")
                if tb and not isinstance(exc, expected):
                    self.log(tb.rstrip())
                if on_failure is not None:
                    on_failure(exc)
                self.error(f"{title} failed", str(exc))
                return
            self.progress.set_fraction(1.0)
            self.progress.set_text(f"{title}: done")
            on_success(result)

        self.worker = Worker(job, on_log or self.log, self.on_progress, done)
        self.worker.start()
        return True

    def on_progress(self, fraction: float, message: str) -> None:
        self.progress.set_fraction(min(1.0, max(0.0, fraction)))
        if message:
            self.progress.set_text(f"{fraction * 100:4.0f}%  {message}")

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.progress.set_text("Cancelling …")

    def stop(self, wait: float = 0.0) -> None:
        """Cancel the job (window closing) and give it up to ``wait`` seconds to stop its scripts."""
        worker = self.worker
        if worker is not None and worker.running:
            worker.cancel()
            if wait > 0:
                worker.thread.join(wait)


# ---------------------------------------------------------------------------
# running generated scripts
# ---------------------------------------------------------------------------


def confirm_run(parent: Optional[Gtk.Window], jobs: Sequence, separate: Sequence[str] = (),
                project_path: Optional[str] = None, parallel: int = 1,
                with_separate: Optional[Sequence] = None) -> Optional[List]:
    """Ask before running generated scripts; returns the jobs to run (``with_separate`` when the user
    also runs the scripts marked 'run separately'), or None when the user cancels."""
    lines = [f"• step {j.step:02d} ({j.title}): {os.path.basename(j.path)}" for j in jobs]
    detail = "\n".join(lines[:20]) + (f"\n… and {len(lines) - 20} more" if len(lines) > 20 else "")
    if parallel > 1 and len(jobs) > 1:
        detail += (f"\n\nUp to {parallel} scripts run at the same time on this computer; each step starts "
                   "when the steps it reads from have finished. A failure stops the steps that depend "
                   "on it, Cancel stops every running script. Their output appears in the output log, each "
                   "line marked with its step number.")
    else:
        detail += ("\n\nThe scripts run one after the other on this computer; a failure stops the steps "
                   "that depend on it, Cancel stops the running script. Their output appears in the output "
                   "log, each line marked with its step number.")
    if separate:
        detail += (f"\n\n{len(separate)} script(s) marked 'run separately' (e.g. production) are not "
                   "started: until they have finished, their step waits and the steps that read from it "
                   "do not start.")
    if project_path:
        detail += f"\n\nEach run is logged in {os.path.basename(project_path)}."
    else:
        detail += ("\n\nThe project has not been saved, so the runs are not logged (save it first to keep a "
                   "log).")
    dlg = Gtk.MessageDialog(transient_for=parent, modal=True, message_type=Gtk.MessageType.QUESTION,
                            buttons=Gtk.ButtonsType.OK_CANCEL, text=f"Run {len(jobs)} generated script(s)?")
    dlg.format_secondary_text(detail)
    include = None
    if separate and with_separate:
        include = Gtk.CheckButton(label="Also run the scripts marked 'run separately' here, each after "
                                        "its step's main script")
        include.set_tooltip_text("For production runs that are meant for this computer rather than a "
                                 "cluster queue; finished ones are not run again")
        dlg.get_message_area().pack_start(include, False, False, 0)
        include.show()
    ok = dlg.run() == Gtk.ResponseType.OK
    chosen = list(with_separate) if ok and include is not None and include.get_active() else list(jobs)
    dlg.destroy()
    return chosen if ok else None


def waiting_text(results: Sequence) -> str:
    """What a run's waiting steps need ("" when no step waits for a separately run script)."""
    waiting = [r for r in results if r.status == "waiting"]
    if not waiting:  # not a failure: the user runs those scripts, then the step again
        return ""
    text = "\n".join(f"Step {r.job.step:02d} ({r.job.title}) is complete once its separately run script "
                     f"has finished: run {', '.join(r.job.separate) or 'it'} (e.g. on a cluster), then the "
                     "step again." for r in waiting)
    skipped = sum(1 for r in results if r.status == "skipped")
    if skipped:
        text += f"\n\n{skipped} script(s) of the steps that read from it were not started."
    return text
