"""Planner helpers that write no files: notes, existing input files and the
MDATK analysis step that links a plan's trajectory to this project."""

from __future__ import annotations

import os
from typing import List

from .base import PortValue, Tool
from .registry import register
from .schema import FILE_KINDS, Issue, Option, Port, Section


@register
class NoteTool(Tool):
    name = "note"
    label = "Note"
    category = "Planning"
    tag = "note"
    generates = False
    description = "A free-text note on the planner canvas."
    sections = [Section("note", "Note", [
        Option("text", "Text", "multiline", "", "Anything worth remembering about the plan.")])]

    def describe(self, s):
        first = (s.get("text") or "").strip().splitlines()
        return first[0] if first else "(empty note)"


@register
class InputFileTool(Tool):
    name = "input-file"
    label = "Existing file"
    category = "Planning"
    tag = "input"
    generates = False
    path_settings = ("path",)
    description = ("A file that already exists (starting PDB, density map, topology ...) "
                   "feeding later steps.")
    sections = [Section("file", "File", [
        Option("path", "File", "file", "", "The existing file. Relative paths are read from the "
               "folder of the saved preset."),
        Option("kind", "Type", "choice", "pdb", "What the file contains; decides which inputs it "
               "can feed.", choices=[(k, v[0]) for k, v in FILE_KINDS.items()]),
    ])]

    def output_ports(self, s):
        return [Port("file", FILE_KINDS.get(s.get("kind"), (s.get("kind"),))[0], (s.get("kind") or "pdb",))]

    def check(self, s, ctx):
        path = (s.get("path") or "").strip()
        if not path:
            return [Issue("error", "choose the file", ("path",))]
        if not os.path.exists(os.path.abspath(os.path.expanduser(path))):
            return [Issue("warning", f"file not found (yet): {path}", ("path",))]
        return []

    def output_values(self, s, ctx):
        path = (s.get("path") or "").strip()
        if not path:
            return {}
        return {"file": PortValue([os.path.abspath(os.path.expanduser(path))], s.get("kind") or "pdb")}

    def describe(self, s):
        return os.path.basename(s.get("path") or "") or "(no file chosen)"


@register
class AnalysisTool(Tool):
    name = "mdatk"
    label = "MDATK analysis"
    category = "Analysis"
    tag = "mdatk"
    generates = False
    description = ("Marks the topology and trajectory this project analyses; 'Use for analysis' in "
                   "the planner copies their paths into Input & cleaning.")
    inputs = [
        Port("topology", "Topology", ("parm7", "top", "gro", "pdb")),
        Port("trajectory", "Trajectory", ("nc", "xtc"), multiple=True),
    ]
    sections = [Section("analysis", "Analysis", [
        Option("clean", "Clean with cpptraj first", "bool", True,
               "Enable cpptraj stripping / imaging / fitting (Input & cleaning) when the paths are "
               "applied; solvated MD output normally needs it."),
    ])]

    def check(self, s, ctx):
        out: List[Issue] = []
        for port in ("topology", "trajectory"):
            value = ctx.value(port)
            if value and value.source:
                out.append(Issue("info", f"{port} comes from {value.source_label}; the file exists "
                                         "once that step has run"))
        return out

    def describe(self, s):
        return "analyse in this project"
