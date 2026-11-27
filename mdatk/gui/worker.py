"""Run pipeline work in a background thread and marshal callbacks to the GTK loop."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, List, Optional

from gi.repository import GLib

from ..progress import Cancelled, Reporter


def call_soon(fn: Callable, *args) -> None:
    """Run ``fn(*args)`` once on the GTK main loop (never repeated)."""

    def once():
        fn(*args)
        return False

    GLib.idle_add(once)


FLUSH_MS = 40  # log lines that arrive within this time reach the window together
MAX_BATCH = 2000  # lines one batch holds at most, the newest (a note counts the others)
FLOOD_PAUSE_S = 0.001  # a thread that floods the log pauses this long after every MAX_BATCH lines


class LogBatch:
    """Log lines from any thread, handed to ``sink(text)`` on the GTK main loop in batches (the design of
    RNAdo's :class:`~mdatk.gui.logpanel.LogPanel`): the lines of FLUSH_MS as one text, so a chatty job
    costs the loop one callback per batch instead of one per line.  A batch holds at most MAX_BATCH
    lines, the newest; when a job prints faster than that, a note says how many lines were skipped."""

    def __init__(self, sink: Callable[[str], None]):
        self.sink = sink
        self._pending: List[str] = []
        self._skipped = 0
        self._lock = threading.Lock()
        self._source = 0
        self._closed = False

    def feed(self, text: str) -> None:
        """Queue a message (several lines allowed); safe from any thread."""
        with self._lock:
            if self._closed:
                return
            self._pending.extend(str(text).split("\n"))
            flood = len(self._pending) > 2 * MAX_BATCH
            if flood:
                self._skip(len(self._pending) - MAX_BATCH)
            if not self._source:
                self._source = GLib.timeout_add(FLUSH_MS, self._flush)
        if flood and threading.current_thread() is not threading.main_thread():
            time.sleep(FLOOD_PAUSE_S)  # the main loop gets the interpreter for a moment

    def _skip(self, n: int) -> None:
        del self._pending[:n]
        self._skipped += n

    def flush(self) -> None:
        """Hand the queued lines over now (main loop): what follows them in time follows them in the log."""
        with self._lock:
            if self._source:
                GLib.source_remove(self._source)
                self._source = 0
        self._flush()

    def _flush(self) -> bool:
        with self._lock:
            if len(self._pending) > MAX_BATCH:
                self._skip(len(self._pending) - MAX_BATCH)
            lines, self._pending = self._pending, []
            skipped, self._skipped = self._skipped, 0
            self._source = 0
        if skipped:
            lines.insert(0, f"… {skipped} line(s) not shown: they came faster than the log can show them")
        if lines and not self._closed:
            self.sink("\n".join(lines))
        return False

    def close(self) -> None:
        """The window is gone: nothing more is handed over."""
        with self._lock:
            self._closed = True
            self._pending = []
            if self._source:
                GLib.source_remove(self._source)
                self._source = 0


class Worker:
    """Runs ``job(reporter)`` in a thread; callbacks run on the GTK main loop.  The job's log lines reach
    ``on_log`` in batches (:class:`LogBatch`; ``log_batch``: the one of the window, shared with its other
    sources of lines), all of them before ``on_done``."""

    def __init__(self, job: Callable[[Reporter], Any],
                 on_log: Callable[[str], None],
                 on_progress: Callable[[float, str], None],
                 on_done: Callable[[Any, Optional[BaseException], str], None],
                 log_batch: Optional[LogBatch] = None):
        self.job = job
        self.log_batch = log_batch or LogBatch(on_log)

        def done(result, exc, tb):
            self.log_batch.flush()  # the job's last lines first
            on_done(result, exc, tb)

        self.on_done = done
        self.reporter = Reporter(
            log_cb=self.log_batch.feed,
            progress_cb=self._throttled(on_progress))
        self.thread = threading.Thread(target=self._run, name="mdatk-worker", daemon=True)

    @staticmethod
    def _throttled(cb):
        state = {"last": -1.0}

        def progress(fraction: float, message: str) -> None:
            if fraction >= 1.0 or fraction - state["last"] >= 0.005 or fraction < state["last"]:
                state["last"] = fraction
                call_soon(cb, fraction, message)

        return progress

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.reporter.cancel()

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def _run(self) -> None:
        try:
            result = self.job(self.reporter)
        except Cancelled as exc:
            call_soon(self.on_done, None, exc, "")
        except BaseException as exc:  # noqa: BLE001 - report everything to the UI
            call_soon(self.on_done, None, exc, traceback.format_exc())
        else:
            call_soon(self.on_done, result, None, "")
