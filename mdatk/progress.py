"""Progress reporting and cancellation shared by the CLI and the GUI."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

log = logging.getLogger("mdatk")


class Cancelled(Exception):
    """Raised when the user cancels a long running operation."""


class Reporter:
    """Forwards log lines and progress fractions to optional callbacks.

    ``sub(start, end)`` returns a reporter whose 0..1 progress maps onto the
    ``start..end`` part of its parent, so pipeline stages can report
    independently while the GUI shows one overall progress bar.
    """

    def __init__(self, log_cb: Optional[Callable[[str], None]] = None,
                 progress_cb: Optional[Callable[[float, str], None]] = None,
                 cancel_event: Optional[threading.Event] = None,
                 _span=(0.0, 1.0), _parent: Optional["Reporter"] = None):
        self._log_cb = log_cb
        self._progress_cb = progress_cb
        self.cancel_event = cancel_event or threading.Event()
        self._span = _span
        self._parent = _parent

    def log(self, message: str) -> None:
        log.info(message)
        if self._log_cb:
            self._log_cb(message)

    def raw_log(self, message: str) -> None:
        """Forward a line (e.g. cpptraj output) without re-logging it to python logging."""
        log.debug(message)
        if self._log_cb:
            self._log_cb(message)

    def progress(self, fraction: float, message: str = "") -> None:
        fraction = min(1.0, max(0.0, float(fraction)))
        a, b = self._span
        if self._progress_cb:
            self._progress_cb(a + (b - a) * fraction, message)

    def sub(self, start: float, end: float) -> "Reporter":
        a, b = self._span
        return Reporter(self._log_cb, self._progress_cb, self.cancel_event,
                        (a + (b - a) * start, a + (b - a) * end), self)

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("operation cancelled")

    def cancel(self) -> None:
        self.cancel_event.set()
