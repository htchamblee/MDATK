"""Control-file / script generators and the Project Planner (MDATK 0.0.2b).

The generators *write* input files and shell scripts for Rosetta FARFAR2,
SimRNA, RNAMake, Amber, GROMACS and VMD; they never run those programs.  Each
generator is a :class:`~mdatk.workflows.base.Tool` with a declarative option
schema (used by the GTK wizard and the CLI), logic checks and planner ports.
The Project Planner (:mod:`mdatk.workflows.plan`) chains steps so that input
and output names stay consistent: ``<project>_<step>_<tool>_<time>.<ext>``.
"""

from __future__ import annotations

from typing import Callable

from . import amber, buildstep, dictstep, farfar2, gromacs, misc, refine, rnamake, simrna, vmd  # noqa: F401
from .base import PortValue, StepContext, Tool
from .registry import TOOLS, get_tool, tools_by_category
from .schema import GeneratedFile, Issue, Option, Port, Section

__all__ = ["TOOLS", "get_tool", "tools_by_category", "Tool", "StepContext", "PortValue",
           "GeneratedFile", "Issue", "Option", "Port", "Section", "resolve_node_paths"]


def resolve_node_paths(node, fix: Callable[[str], str]) -> None:
    """Make hand-picked file paths of a planner step absolute (preset loading)."""
    node.files = {port: ";".join(fix(p.strip()) for p in path.split(";") if p.strip())
                  for port, path in node.files.items()}
    tool = TOOLS.get(node.tool)
    if tool is not None:
        for key in tool.path_settings:
            value = node.settings.get(key)
            if isinstance(value, str) and value:
                node.settings[key] = fix(value)
