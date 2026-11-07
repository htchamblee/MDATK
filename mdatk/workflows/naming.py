"""File-name convention of generated workflow files.

Every file a step produces is named ``<project>_<step>_<tool>_<time>.<ext>``,
for example ``apt1_03_tleap_20261015-1432.parm7``:

``project``
    the plan's project name (letters, digits, ``.``, ``-`` and ``_``);
``step``
    the planner step number, zero padded to two digits;
``tool``
    the program or stage that writes the file (``tleap``, ``parmed``,
    ``min1``, ``heat`` ...).  Only letters, digits and hyphens are kept so
    names split unambiguously (``eq_free`` -> ``eq-free``);
``time``
    one time stamp per plan (``%Y%m%d-%H%M``); it is fixed the first time
    files are written so later re-generations keep matching names.  Plans can
    switch it off.

Each step writes into its own folder ``<step>_<tool>`` (e.g. ``03_amber-build``).  A step may carry a folder
tag (``PlanNode.folder_tag``, ASCII letters and digits with hyphens between them), which is added to its
folder name: ``03_amber-build_model12``.  File names do not carry it: :func:`parse_name` reads exactly the
four parts above, and the tag is in the folder name, the step's title (``03 · MD · model12``) and the
``MDATK_step.toml`` identity file instead (:mod:`mdatk.workflows.manifest`).
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, Optional

STAMP_FORMAT = "%Y%m%d-%H%M"
# sub-folder of a step folder that holds the completed runs moved aside before the step was written again
# (mdatk.workflows.supersede); never searched for runs
SUPERSEDED = "superseded"

_PROJECT_BAD = re.compile(r"[^A-Za-z0-9._-]+")
_TAG_BAD = re.compile(r"[^A-Za-z0-9-]+")


def new_stamp(now: Optional[_dt.datetime] = None) -> str:
    return (now or _dt.datetime.now()).strftime(STAMP_FORMAT)


def sanitize_project(name: str) -> str:
    name = _PROJECT_BAD.sub("-", (name or "").strip()).strip("-._")
    return name or "project"


def sanitize_tag(tag: str) -> str:
    tag = _TAG_BAD.sub("-", (tag or "").strip()).strip("-")
    return tag or "step"


def step_label(step: int) -> str:
    return f"{int(step):02d}"


def file_name(project: str, step: int, tag: str, stamp: str = "", ext: str = "") -> str:
    parts = [sanitize_project(project), step_label(step), sanitize_tag(tag)]
    if stamp:
        parts.append(stamp)
    name = "_".join(parts)
    ext = (ext or "").lstrip(".")
    return f"{name}.{ext}" if ext else name


def step_folder(step: int, tag: str, folder_tag: str = "") -> str:
    """``<step>_<tag>`` (``03_amber-build``), with a folder tag ``<step>_<tag>_<folder tag>``
    (``03_amber-build_model12``)."""
    name = f"{step_label(step)}_{sanitize_tag(tag)}"
    return f"{name}_{sanitize_tag(folder_tag)}" if (folder_tag or "").strip() else name


def folder_tag_problem(tag: str) -> str:
    """Why ``tag`` cannot be a step's folder tag ("" when it can, or is empty): only ASCII letters, digits
    and inner hyphens, the characters :func:`sanitize_tag` keeps, so the folder name splits unambiguously."""
    tag = (tag or "").strip()
    if not tag or sanitize_tag(tag) == tag:
        return ""
    fixed = _TAG_BAD.sub("-", tag).strip("-")
    return (f"folder tag {tag!r}: use ASCII letters and digits, with '-' only between them"
            + (f" (e.g. {fixed!r})" if fixed else ""))


_FOLDER = re.compile(r"^(?P<step>\d{2,})_(?P<tag>[A-Za-z0-9-]+)(?:_(?P<folder_tag>[A-Za-z0-9-]+))?$")


def parse_folder(name: str) -> Optional[Dict[str, str]]:
    """Split a step folder name (:func:`step_folder`) into ``step``, ``tag`` and ``folder_tag`` ("" when it
    has none); None if ``name`` is not one."""
    m = _FOLDER.match(name)
    return {k: (v or "") for k, v in m.groupdict().items()} if m else None


_NAME = re.compile(r"^(?P<project>.+)_(?P<step>\d{2,})_(?P<tag>[A-Za-z0-9-]+)"
                   r"(?:_(?P<stamp>\d{8}-\d{4}))?(?:\.(?P<ext>[A-Za-z0-9]+))?$")


def parse_name(name: str) -> Optional[Dict[str, str]]:
    """Split a conventional file name into its parts (None if it does not match)."""
    m = _NAME.match(name)
    if not m:
        return None
    parts = {k: (v or "") for k, v in m.groupdict().items()}
    return parts
