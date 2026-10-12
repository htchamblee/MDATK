"""Run pipeline work in a background thread and marshal callbacks to the GTK loop."""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Optional

from gi.repository import GLib

from ..progress import Cancelled, Reporter


def call_soon(fn: Callable, *args) -> None:
    """Run ``fn(*args)`` once on the GTK main loop (never repeated)."""

    def once():
        fn(*args)
        return False

    GLib.idle_add(once)


class Worker:
    """Runs ``job(reporter)`` in a thread; callbacks run on the GTK main loop."""

    def __init__(self, job: Callable[[Reporter], Any],
                 on_log: Callable[[str], None],
                 on_progress: Callable[[float, str], None],
                 on_done: Callable[[Any, Optional[BaseException], str], None]):
        self.job = job
        self.on_done = on_done
        self.reporter = Reporter(
            log_cb=lambda text: call_soon(on_log, text),
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
