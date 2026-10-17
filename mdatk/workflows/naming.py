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
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, Optional

STAMP_FORMAT = "%Y%m%d-%H%M"

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


def step_folder(step: int, tag: str) -> str:
    return f"{step_label(step)}_{sanitize_tag(tag)}"


_NAME = re.compile(r"^(?P<project>.+)_(?P<step>\d{2,})_(?P<tag>[A-Za-z0-9-]+)"
                   r"(?:_(?P<stamp>\d{8}-\d{4}))?(?:\.(?P<ext>[A-Za-z0-9]+))?$")


def parse_name(name: str) -> Optional[Dict[str, str]]:
    """Split a conventional file name into its parts (None if it does not match)."""
    m = _NAME.match(name)
    if not m:
        return None
    parts = {k: (v or "") for k, v in m.groupdict().items()}
    return parts
