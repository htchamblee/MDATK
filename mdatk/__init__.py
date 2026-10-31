"""Molecular Dynamic Analysis Tool Kit (MDATK).

An offline, Linux-native toolkit for analysing and visualising molecular
dynamics trajectories of RNA systems.  The package is organised as a strict
pipeline::

    cleaning (cpptraj wrapper) -> analysis modules -> cache I/O -> rendering / GUI

Every stage is importable on its own and is driven by a single
:class:`mdatk.config.Project` parameter object, so the same work can be done
from the GTK 3 GUI (``mdatk gui``) or the companion command line interface
(``mdatk --help``).
"""

PROJECT_NAME = "Molecular Dynamic Analysis Tool Kit"
PROJECT_ABBREV = "MDATK"
PROJECT_TITLE = f"{PROJECT_NAME} ({PROJECT_ABBREV})"

__version__ = "0.0.10"

__all__ = ["PROJECT_NAME", "PROJECT_ABBREV", "PROJECT_TITLE", "__version__"]
