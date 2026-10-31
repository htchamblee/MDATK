"""What the planner, the script wizard, the PDB clean-up and the structure builder need from the window
that hosts them.

RNAdo (:mod:`mdatk.gui.rnado`) is that window.  The tools use nothing else of it, so a test or another
program can host them with any object that has these members.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Sequence

from ..config import Project


class Host(Protocol):
    """The project and the services a hosted tool uses."""

    project: Project  # the open project (the planner and the wizard edit project.workflow in place)
    project_path: Optional[str]  # its file, whose step log records what runs (None: unsaved)

    def append_log(self, text: str) -> None:
        """A message for the output log (planner writes, checks, PDB clean-ups ...)."""

    def error(self, text: str, detail: str = "") -> None:
        """Report a problem (logged and shown)."""

    def info(self, text: str, detail: str = "") -> None:
        """Report a result (logged and shown)."""

    def save_project(self, save_as: bool = False) -> bool:
        """Save the project (its plan); False when it was not saved."""

    def without_edit(self, change: Callable[[], None]) -> None:
        """Run ``change``, which adjusts the plan without the user's edit (a script wizard opens and fills
        in its step's defaults); a plan without unsaved edits still has none afterwards."""

    def run_scripts(self, jobs: Sequence, separate: Sequence[str] = (), parent=None, parallel: int = 1,
                    on_state: Optional[Callable] = None, on_finished: Optional[Callable] = None,
                    with_separate: Optional[Sequence] = None) -> None:
        """Run generated scripts after a confirmation; ``on_state(job, state)`` and ``on_finished()`` are
        called on the GTK main loop, ``with_separate`` (the jobs including the scripts marked 'run
        separately') adds the choice to run those too."""

    def use_for_analysis(self, topology: str, trajectories: List[str], clean: bool = True) -> None:
        """Hand the files an MDATK analysis step reads to the analysis window."""
