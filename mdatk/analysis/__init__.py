"""Analysis modules.

Importing this package registers every built-in analysis in
:data:`mdatk.analysis.base.ANALYSES`.
"""

from . import contacts, energy, hbonds, mg, rmsd, stacking, structure  # noqa: F401 (registration)
from .base import ANALYSES, Analysis, AnalysisContext, referenced_groups, register, run_analyses

__all__ = ["ANALYSES", "Analysis", "AnalysisContext", "referenced_groups", "register",
           "run_analyses"]


def expected_keys(project, include_disabled: bool = True):
    """Data keys ``{"series": [...], "matrices": [...], "pairmaps": [...]}`` that the
    registered analyses will provide for ``project`` (without computing anything)."""
    out = {"series": [], "matrices": [], "pairmaps": []}
    for name, cls in ANALYSES.items():
        if not include_disabled and not cls.internal and not cls.enabled(project):
            continue
        for kind, names in cls.expected_outputs(project).items():
            out[kind].extend(f"{name}/{n}" for n in names)
    return out
