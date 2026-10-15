"""Locate conda environments that provide cpptraj and run commands inside them.

People install AmberTools under different environment names (``AmberTools26``,
``ambertools``, ``amber24`` ...), so environments are recognised by what they
contain - an executable ``bin/cpptraj`` - rather than by their name.
Commands run through ``conda run`` (or ``mamba run`` / ``micromamba run``),
which is equivalent to ``conda activate <env>`` for the child process.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

COMMON_ROOTS = (
    "~/miniforge3", "~/mambaforge", "~/miniconda3", "~/anaconda3", "~/micromamba",
    "~/.local/share/mamba", "~/.conda", "/opt/conda", "/opt/miniforge3", "/opt/mambaforge",
    "/opt/miniconda3", "/opt/anaconda3", "/usr/local/miniconda3", "/usr/local/anaconda3",
)
_AMBER_NAME = re.compile(r"amber", re.IGNORECASE)


@dataclass(frozen=True)
class CondaEnv:
    name: str  # environment name ("base" for a root installation)
    prefix: str  # absolute environment directory
    root: str  # installation the environment belongs to

    @property
    def cpptraj(self) -> str:
        return os.path.join(self.prefix, "bin", "cpptraj")

    @property
    def has_cpptraj(self) -> bool:
        return os.path.isfile(self.cpptraj) and os.access(self.cpptraj, os.X_OK)


def candidate_roots(extra: Iterable[str] = ()) -> List[str]:
    """Conda/mamba installation directories worth searching (existing ones only)."""
    roots: List[str] = list(extra)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:  # <root>/bin/conda or <root>/condabin/conda
        roots.append(os.path.dirname(os.path.dirname(conda_exe)))
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        parent = os.path.dirname(prefix)
        roots.append(os.path.dirname(parent) if os.path.basename(parent) == "envs" else prefix)
    for var in ("MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        if os.environ.get(var):
            roots.append(os.environ[var])
    roots.extend(COMMON_ROOTS)
    out: List[str] = []
    seen = set()
    for r in roots:
        path = os.path.realpath(os.path.expanduser(r))
        if path not in seen and os.path.isdir(path):
            seen.add(path)
            out.append(path)
    return out


def _registered_prefixes() -> List[str]:
    """Environments conda recorded in ``~/.conda/environments.txt``."""
    path = os.path.expanduser("~/.conda/environments.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    except OSError:
        return []


def find_envs(roots: Optional[Sequence[str]] = None, include_registered: bool = True) -> List[CondaEnv]:
    """All environments under ``roots`` (default: :func:`candidate_roots`)."""
    roots = candidate_roots() if roots is None else [os.path.realpath(os.path.expanduser(r)) for r in roots]
    envs: List[CondaEnv] = []
    seen = set()

    def add(name: str, prefix: str, root: str) -> None:
        real = os.path.realpath(prefix)
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            envs.append(CondaEnv(name, real, root))

    for root in roots:
        add("base", root, root)
        env_dir = os.path.join(root, "envs")
        if os.path.isdir(env_dir):
            for name in sorted(os.listdir(env_dir)):
                add(name, os.path.join(env_dir, name), root)
    if include_registered:
        for prefix in _registered_prefixes():
            parent = os.path.dirname(prefix)
            root = os.path.dirname(parent) if os.path.basename(parent) == "envs" else prefix
            add(os.path.basename(prefix) if prefix != root else "base", prefix, root)
    return envs


def find_cpptraj_envs(roots: Optional[Sequence[str]] = None) -> List[CondaEnv]:
    """Environments containing an executable cpptraj, AmberTools-like names first."""
    envs = [e for e in find_envs(roots, include_registered=roots is None) if e.has_cpptraj]
    return sorted(envs, key=lambda e: (not _AMBER_NAME.search(e.name), e.name == "base", e.name))


def detect_ambertools_env(roots: Optional[Sequence[str]] = None) -> Optional[CondaEnv]:
    envs = find_cpptraj_envs(roots)
    return envs[0] if envs else None


def resolve_env(spec: str, roots: Optional[Sequence[str]] = None) -> Optional[CondaEnv]:
    """Find an environment given by name (``AmberTools26``) or by path."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if os.sep in spec:
        prefix = os.path.realpath(os.path.expanduser(spec))
        if not os.path.isdir(prefix):
            return None
        parent = os.path.dirname(prefix)
        root = os.path.dirname(parent) if os.path.basename(parent) == "envs" else prefix
        return CondaEnv(os.path.basename(prefix), prefix, root)
    for env in find_envs(roots, include_registered=roots is None):
        if env.name == spec:
            return env
    return None


def find_conda_exe(explicit: str = "", env: Optional[CondaEnv] = None) -> Optional[str]:
    """conda / mamba / micromamba executable able to ``run`` inside an environment."""
    if explicit:
        path = os.path.expanduser(explicit)
        if os.sep in path:
            return path if os.access(path, os.X_OK) else None
        return shutil.which(path)
    candidates = []
    if env is not None:
        for rel in ("condabin/conda", "bin/conda", "bin/mamba", "bin/micromamba"):
            candidates.append(os.path.join(env.root, rel))
    if os.environ.get("CONDA_EXE"):
        candidates.append(os.environ["CONDA_EXE"])
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found
    return None


def run_command(conda_exe: str, env_spec: str, argv: Sequence[str]) -> List[str]:
    """Command line running ``argv`` inside ``env_spec`` (a name or a prefix path)."""
    flag = "-p" if os.sep in env_spec else "-n"
    tool = os.path.basename(conda_exe)
    if tool.startswith("micromamba"):
        return [conda_exe, "run", flag, env_spec, *argv]
    # --no-capture-output streams the child's output live (conda >= 4.9)
    return [conda_exe, "run", "--no-capture-output", flag, env_spec, *argv]
