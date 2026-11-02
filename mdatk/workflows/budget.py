"""CPU and GPU budgets for the scripts MDATK runs at the same time.

Each step type declares what one run of its scripts uses, from the step's settings
(:meth:`mdatk.workflows.base.Tool.run_resources`): CPU cores, GPUs, and the GPU ids the step names
itself (*pinned*: the Amber MD step's GPU option, NAMD's ``+devices``, ``CUDA_VISIBLE_DEVICES`` in the
step's Environment setup commands).  The runner (:func:`mdatk.workflows.runner.run_jobs`) starts a ready
script when the cores and GPUs it needs are free (:class:`Slots`, :func:`schedule`) and gives it free GPUs
in ``CUDA_VISIBLE_DEVICES`` unless the step names its own.  The budget is the project's (``cpu_cores``:
half the cores by default, ``gpus``: the GPUs found); ``parallel`` remains as an optional cap on the
number of scripts at once.

The bookkeeping here is pure (no processes, no clock), so the order in which scripts start depends only
on the plan order, what each step needs and which scripts have finished.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

ALL_CORES = 0  # Resources.cores of a program that takes every core it finds (e.g. gmx mdrun without -ntomp)
MAX_GPUS_LISTED = 16  # GPU counts RNAdo's GPUs control offers besides "auto"


class Resources(NamedTuple):
    """What one run of a step's scripts uses."""

    cores: int = 1  # CPU cores (ALL_CORES: every core of the budget)
    gpus: int = 0  # GPUs
    pinned: Tuple[str, ...] = ()  # the GPU ids the step names itself (empty: MDATK hands out free ones)

    @property
    def uses_gpu(self) -> bool:
        return self.gpus > 0 or bool(self.pinned)

    def text(self) -> str:
        """'8 cores, 1 GPU', 'all cores', '1 core, GPU 1 (named by the step)'."""
        cores = "all cores" if self.cores == ALL_CORES else f"{self.cores} core{'s' * (self.cores != 1)}"
        if self.pinned:
            return f"{cores}, GPU {', '.join(self.pinned)} (named by the step)"
        if self.gpus:
            return f"{cores}, {self.gpus} GPU{'s' * (self.gpus != 1)}"
        return cores

    def badge(self) -> str:
        """Short form for a planner box: '1c', '8c+1g', 'all cores+1g'."""
        cores = "all cores" if self.cores == ALL_CORES else f"{self.cores}c"
        n = len(self.pinned) or self.gpus
        return f"{cores}+{n}g" if n else cores


def gpu_ids(text: str) -> Tuple[str, ...]:
    """The ids in a ``CUDA_VISIBLE_DEVICES`` / ``+devices`` value: '0', '0,1', '0 1'."""
    return tuple(p for p in re.split(r"[,\s]+", (text or "").strip()) if p)


_CUDA_SET = re.compile(r"(?:^|[\s;&|(])(?:export\s+)?CUDA_VISIBLE_DEVICES=(\"[^\"]*\"|'[^']*'|[^\s;&|)]*)")


def env_gpus(settings: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    """GPU ids that a step's Environment setup commands put into ``CUDA_VISIBLE_DEVICES`` (the last
    assignment; comments ignored), or None when they do not set it."""
    found: Optional[Tuple[str, ...]] = None
    for line in str(settings.get("env_setup") or "").splitlines():
        line = re.split(r"(?:^|\s)#", line, maxsplit=1)[0]
        for m in _CUDA_SET.finditer(line):
            found = gpu_ids(m.group(1).strip("'\""))
    return found


def gpu_step(cores: int, gpus: int, settings: Mapping[str, Any], named: Sequence[str] = ()) -> Resources:
    """Resources of a step that uses ``gpus`` GPUs: the ids the step names (``named``, e.g. its GPU
    option, else its Environment's ``CUDA_VISIBLE_DEVICES``) are pinned and set the count."""
    pinned = tuple(named) or env_gpus(settings) or ()
    return Resources(cores, len(pinned) if pinned else gpus, pinned)


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------


def auto_cores() -> int:
    """The default CPU budget: half the cores (at least one)."""
    return max(1, (os.cpu_count() or 2) // 2)


@functools.lru_cache(maxsize=1)
def _smi_listing() -> str:
    """``nvidia-smi -L`` ("" when it is missing or fails); asked once per process."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""
    from ..frozen import child_env
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=5, env=child_env())
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def detected_gpus(environ: Optional[Mapping[str, str]] = None, listing: Optional[str] = None) -> List[str]:
    """The GPUs MDATK may hand out on this computer, as ``CUDA_VISIBLE_DEVICES`` ids.

    ``CUDA_VISIBLE_DEVICES`` of MDATK's own environment, when set, limits what its scripts can see, so
    its entries are the GPUs (empty or ``-1``: none).  Otherwise one id per GPU that ``nvidia-smi -L``
    lists (``0``, ``1``, ...; ``listing`` stands in for its output), else none."""
    env = os.environ if environ is None else environ
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = gpu_ids(visible)
        return [] if not ids or ids[0].startswith("-") else list(ids)
    text = _smi_listing() if listing is None else listing
    count = sum(1 for line in text.splitlines() if line.startswith("GPU "))
    return [str(k) for k in range(count)]


@dataclass(frozen=True)
class Budget:
    """What the scripts of one run may use together."""

    cores: int  # CPU cores
    gpus: Tuple[str, ...] = ()  # the GPUs (CUDA_VISIBLE_DEVICES ids) handed out to steps that use one

    @classmethod
    def resolve(cls, cores: int = 0, gpus: int = -1, found: Optional[Sequence[str]] = None) -> "Budget":
        """The budget of the settings ``cores`` (0: :func:`auto_cores`) and ``gpus`` (-1: the GPUs found,
        ``found`` or :func:`detected_gpus`).  A count above the GPUs found adds ids 0, 1, ... not found."""
        ids = list(detected_gpus() if found is None else found)
        if gpus >= 0:
            extra = [str(k) for k in range(gpus + len(ids)) if str(k) not in ids]
            ids = (ids + extra)[:gpus]
        return cls(int(cores) if cores and cores > 0 else auto_cores(), tuple(ids))

    @classmethod
    def of(cls, workflow, cores: Optional[int] = None, gpus: Optional[int] = None) -> "Budget":
        """The budget of a plan's settings (``WorkflowParams``), ``cores`` / ``gpus`` overriding them."""
        return cls.resolve(workflow.cpu_cores if cores is None else cores,
                           workflow.gpus if gpus is None else gpus)

    def text(self) -> str:
        """'8 CPU cores and 2 GPUs (0, 1)'."""
        gpus = (f"{len(self.gpus)} GPU{'s' * (len(self.gpus) != 1)} ({', '.join(self.gpus)})" if self.gpus
                else "no GPU")
        return f"{self.cores} CPU core{'s' * (self.cores != 1)} and {gpus}"


class Grant(NamedTuple):
    """What a starting script gets."""

    cores: int
    gpus: Tuple[str, ...] = ()  # the GPU ids it uses
    assigned: bool = False  # MDATK chose them: the script gets them in CUDA_VISIBLE_DEVICES
    alone: bool = False  # larger than the whole budget: nothing else runs beside it
    uses_gpu: bool = False


class Slots:
    """The free part of a budget while scripts run: which needs fit now (:meth:`grant`), and taking and
    giving back what a script got.  ``budget`` None: no limits but ``limit`` (scripts at once, 0: none);
    GPUs are then neither counted nor handed out."""

    def __init__(self, budget: Optional[Budget], limit: int = 0):
        self.budget = budget
        self.limit = max(0, int(limit))
        self.running = 0
        self.free_cores = budget.cores if budget is not None else 0
        self.in_use: Set[str] = set()  # GPU ids of running scripts (handed out or named by the step)
        self.gpu_jobs = 0  # running scripts that use a GPU
        self.alone = False  # a script larger than the budget runs

    def need(self, res: Resources) -> int:
        """CPU cores ``res`` takes from the budget."""
        if self.budget is None:
            return 0
        return self.budget.cores if res.cores == ALL_CORES else max(1, res.cores)

    def oversized(self, res: Resources) -> bool:
        """Larger than the whole budget: it runs only when nothing else does (instead of never)."""
        b = self.budget
        if b is None:
            return False
        return self.need(res) > b.cores or (not res.pinned and bool(b.gpus) and res.gpus > len(b.gpus))

    def whole(self, res: Resources) -> bool:
        """Needs every core of the budget (or more): it can start only when nothing runs, so the scripts
        after it wait for it instead of starving it."""
        return self.budget is not None and (self.oversized(res) or self.need(res) >= self.budget.cores)

    def grant(self, res: Resources) -> Optional[Grant]:
        """What a script with needs ``res`` gets if it starts now, or None when it does not fit."""
        if self.alone or (self.limit and self.running >= self.limit):
            return None
        b = self.budget
        if b is None:
            return Grant(0)
        uses_gpu = res.uses_gpu
        if self.oversized(res):
            if self.running:
                return None
            ids = res.pinned or (b.gpus[:res.gpus] if uses_gpu else ())
            return Grant(b.cores, tuple(ids), bool(ids) and not res.pinned, True, uses_gpu)
        need = self.need(res)
        if need > self.free_cores:
            return None
        if not uses_gpu:
            return Grant(need)
        if not b.gpus and self.gpu_jobs:  # no GPU known: one script that uses a GPU at a time
            return None
        if res.pinned:
            if self.in_use & set(res.pinned):
                return None
            return Grant(need, res.pinned, False, False, True)
        if not b.gpus:
            return Grant(need, (), False, False, True)
        free = [g for g in b.gpus if g not in self.in_use]
        if len(free) < res.gpus:
            return None
        return Grant(need, tuple(free[:res.gpus]), True, False, True)

    def take(self, grant: Grant) -> None:
        self.running += 1
        self.free_cores -= grant.cores
        self.in_use |= set(grant.gpus)
        self.gpu_jobs += grant.uses_gpu
        self.alone = self.alone or grant.alone

    def give_back(self, grant: Grant) -> None:
        self.running -= 1
        self.free_cores += grant.cores
        self.in_use -= set(grant.gpus)
        self.gpu_jobs -= grant.uses_gpu
        if grant.alone:
            self.alone = False


def schedule(ready: Sequence[Tuple[int, Resources]], slots: Slots) -> List[Tuple[int, Grant]]:
    """The scripts among ``ready`` ((index, needs) of the scripts whose inputs are done, in plan order) that
    start now, each with what it gets; ``slots`` takes their share.  Every one that fits starts, in plan
    order (a later, smaller script may start before an earlier one that waits for room); a script that
    needs every core of the budget, or more than the budget, waits until nothing runs, and while it waits
    nothing after it starts."""
    started: List[Tuple[int, Grant]] = []
    for index, res in ready:
        grant = slots.grant(res)
        if grant is not None:
            slots.take(grant)
            started.append((index, grant))
        elif slots.whole(res):
            break
    return started


def budget_notes(needs: Sequence[Tuple[str, Resources]], budget: Optional[Budget]) -> List[str]:
    """Warnings before a run: steps (``needs``: (label, resources) per step) larger than the budget,
    steps that use a GPU when no GPU is known, GPU ids that steps name outside the budget."""
    if budget is None:
        return []
    out: List[str] = []
    slots = Slots(budget)
    seen: Dict[str, Resources] = {}
    for label, res in needs:
        seen.setdefault(label, res)
    gpu_steps = [label for label, res in seen.items() if res.uses_gpu]
    for label, res in seen.items():
        if slots.oversized(res):
            what = (f"{slots.need(res)} CPU cores" if slots.need(res) > budget.cores
                    else f"{res.gpus} GPUs")
            out.append(f"WARNING: {label} asks for {what}, more than the {budget.text()} MDATK may use: it "
                       "runs alone, once no other script runs")
        outside = [g for g in res.pinned if budget.gpus and g not in budget.gpus]
        if outside:
            out.append(f"WARNING: {label} names GPU {', '.join(outside)}, which is not among the GPUs "
                       f"MDATK hands out ({', '.join(budget.gpus)}); it runs when no other step uses "
                       "that GPU")
    if gpu_steps and not budget.gpus:
        out.append(f"WARNING: no GPU found (nvidia-smi -L, CUDA_VISIBLE_DEVICES) and none set, so the steps "
                   f"that use a GPU run one at a time: {'; '.join(gpu_steps)}. Set the number of GPUs "
                   "(RNAdo: GPUs; command line: --gpus N) if this computer has some.")
    return out
