"""Registry of script generators (Project Planner step types)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import Tool

TOOLS: Dict[str, Tool] = {}

CATEGORIES = ("Planning", "Structure prediction", "Amber", "GROMACS", "Visualisation", "Analysis")


def register(cls):
    tool = cls()
    if not tool.name or tool.name in TOOLS:
        raise ValueError(f"bad or duplicate tool name {tool.name!r}")
    keys = [o.key for s in tool.sections if not s.repeat for o in s.options]
    keys += [s.key for s in tool.sections if s.repeat]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"{tool.name}: duplicate option keys {dupes}")
    TOOLS[tool.name] = tool
    return cls


def get_tool(name: str) -> Tool:
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown step type {name!r} (known: {', '.join(sorted(TOOLS))})") from None


def tools_by_category() -> List[Tuple[str, List[Tool]]]:
    out = []
    for cat in CATEGORIES:
        tools = [t for t in TOOLS.values() if t.category == cat]
        if tools:
            out.append((cat, tools))
    rest = [t for t in TOOLS.values() if t.category not in CATEGORIES]
    if rest:
        out.append(("Other", rest))
    return out
