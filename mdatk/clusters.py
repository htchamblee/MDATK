"""Cluster profiles: the per-user description of a batch cluster that the steps' cluster job scripts use
(mdatk.workflows.cluster), kept outside project files in ``<config>/mdatk/clusters.toml`` so that every project on
this computer uses them.

A profile holds the job header (the scheduler's directives and module loads), the GPU the jobs ask for, the
cluster's queues (the lines that choose one and its wall-time limit), a cushion and the speeds of earlier runs:
GPU, atoms, time step and seconds/ns read from finished pmemd ``.out`` files (:func:`learn`).  A job's wall time
(:func:`estimate`) is the ns of the stages it runs times the seconds/ns of the nearest earlier runs on that GPU,
scaled to the system's atoms and time step, plus a few minutes, plus the cushion; the job goes into the shortest
queue whose limit holds it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import PROJECT_ABBREV, PROJECT_TITLE, tomlio

CUSHION_RANGE = (1.0, 20.0)  # percent
MAX_BENCHMARKS = 400  # per profile: the newest are kept
OVERHEAD_MIN = 5.0  # minutes per job (start-up, the main script's checks, the dry copies)
PER_MIN_STAGE = 2.0  # minutes per energy minimisation the job runs
ROUND_MIN = 5  # wall times are rounded up to this many minutes
NEAREST = 3  # earlier runs that set a speed (the nearest in size)


@dataclass
class Queue:
    name: str = "gpu"
    max_hours: float = 48.0
    directives: str = ""  # the lines that choose it, e.g. #SBATCH --partition=gpu


@dataclass
class Benchmark:
    gpu: str  # as pmemd names the device, e.g. NVIDIA A100-PCIE-40GB
    atoms: int
    dt: float  # ps
    s_per_ns: float
    when: str = ""  # date of the .out file
    source: str = ""  # its path


@dataclass
class ClusterProfile:
    name: str
    header: str = ""  # directives and module loads, without the wall time and the queue
    gpu: str = ""  # the GPU the jobs ask for, as part of pmemd's device name (e.g. A100); "": any
    queues: List[Queue] = field(default_factory=lambda: [Queue()])
    cushion: float = 10.0  # percent added to the estimate
    time_directive: str = ""  # "": from the scheduler ({time} HH:MM:SS, {hhmm} HH:MM, {minutes})
    submit: str = ""  # "": from the scheduler ({name}, {file})
    benchmarks: List[Benchmark] = field(default_factory=list)

    def speeds(self) -> List[Benchmark]:
        """The earlier runs on the profile's GPU."""
        want = self.gpu.strip().lower()
        return [b for b in self.benchmarks if not want or want in b.gpu.lower()]


# ---------------------------------------------------------------------------- storage
def profiles_path() -> str:
    root = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(root, PROJECT_ABBREV.lower(), "clusters.toml")


def _from_dict(d: Dict[str, Any]) -> ClusterProfile:
    def pick(cls, data):
        names = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in data.items() if k in names}
    p = ClusterProfile(**{k: v for k, v in pick(ClusterProfile, d).items() if k not in ("queues", "benchmarks")})
    p.queues = [Queue(**pick(Queue, q)) for q in d.get("queues", []) if isinstance(q, dict)] or [Queue()]
    p.benchmarks = [Benchmark(**pick(Benchmark, b)) for b in d.get("benchmarks", []) if isinstance(b, dict)
                    and {"gpu", "atoms", "dt", "s_per_ns"} <= set(b)]
    p.cushion = clamp_cushion(p.cushion)
    return p


def load_profiles(path: str = "") -> List[ClusterProfile]:
    try:
        data = tomlio.load(path or profiles_path())
    except (OSError, tomlio.TOMLDecodeError):
        return []
    return [_from_dict(d) for d in data.get("profile", []) if isinstance(d, dict) and d.get("name")]


def save_profiles(profiles: Sequence[ClusterProfile], path: str = "") -> None:
    target = path or profiles_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    data = {"profile": [dataclasses.asdict(p) for p in profiles]}
    text = tomlio.dumps(data, lambda p, s: None, [
        f"{PROJECT_TITLE} cluster profiles (RNAdo: menu > Cluster profiles…). A step's Cluster job section names",
        "one; its job script takes the header, a queue and a wall time estimated from the speeds below."])
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, target)


def find(name: str, profiles: Optional[Sequence[ClusterProfile]] = None) -> Optional[ClusterProfile]:
    name = (name or "").strip()
    for p in (load_profiles() if profiles is None else profiles):
        if p.name == name:
            return p
    return None


def clamp_cushion(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 10.0
    return min(max(v, CUSHION_RANGE[0]), CUSHION_RANGE[1])


# ---------------------------------------------------------------------------- speeds from pmemd output
_GPU = re.compile(r"CUDA Device Name:\s*(.+)")
_NATOM = re.compile(r"NATOM\s*=\s*(\d+)")
_DT = re.compile(r"\bdt\s*=\s*([0-9.]+(?:[eE][-+]?\d+)?)")
_ALL = re.compile(r"Average timings for all steps:.*?seconds/ns\s*=\s*([0-9.]+)", re.S)


def parse_out(path: str) -> Optional[Benchmark]:
    """The speed of a finished pmemd.cuda MD run (its .out file); None for anything else (a minimisation, a CPU
    run, a run that has not finished)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    gpu, natom, timing = _GPU.search(text), _NATOM.search(text), _ALL.findall(text)
    if not (gpu and natom and timing):
        return None
    dt = _DT.search(text)
    try:
        s_per_ns = float(timing[-1])
        bench = Benchmark(gpu.group(1).strip(), int(natom.group(1)), float(dt.group(1)) if dt else 0.002, s_per_ns,
                          _dt.date.fromtimestamp(os.path.getmtime(path)).isoformat(), os.path.abspath(path))
    except (ValueError, OSError):
        return None
    return bench if bench.s_per_ns > 0 and bench.atoms > 0 and bench.dt > 0 else None


def out_files(paths: Iterable[str]) -> List[str]:
    found = []
    for p in paths:
        if os.path.isfile(p):
            found.append(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d != "superseded"]
                found += [os.path.join(root, f) for f in sorted(files) if f.endswith(".out")]
    return found


def learn(profile: ClusterProfile, paths: Iterable[str]) -> List[Benchmark]:
    """Adds the speeds of the finished pmemd.cuda runs in ``paths`` (.out files, or folders searched for them) to the
    profile (a file read before is replaced); returns the new ones.  Not saved: :func:`save_profiles`."""
    added = []
    for path in out_files(paths):
        bench = parse_out(path)
        if bench is None:
            continue
        profile.benchmarks = [b for b in profile.benchmarks if b.source != bench.source]
        profile.benchmarks.append(bench)
        added.append(bench)
    profile.benchmarks = profile.benchmarks[-MAX_BENCHMARKS:]
    return added


# ---------------------------------------------------------------------------- wall time
@dataclass
class Estimate:
    seconds: float  # the wall time asked for (cushion included, rounded up)
    queue: Queue
    jobs: int  # > 1: longer than the longest queue - that many jobs one after the other
    basis: str  # how it was worked out, for the job file and the planner


def speed(profile: ClusterProfile, atoms: int, dt: float) -> Optional[Tuple[float, List[Benchmark]]]:
    """Seconds per ns of a system of ``atoms`` atoms at time step ``dt`` on the profile's GPU: the median of the
    nearest earlier runs in size, each scaled by atoms and time step; None without one."""
    runs = profile.speeds()
    if not runs or atoms <= 0 or dt <= 0:
        return None
    near = sorted(runs, key=lambda b: abs(math.log(atoms / b.atoms)))[:NEAREST]
    scaled = sorted(b.s_per_ns * (atoms / b.atoms) * (b.dt / dt) for b in near)
    mid = len(scaled) // 2
    value = scaled[mid] if len(scaled) % 2 else (scaled[mid - 1] + scaled[mid]) / 2
    return value, near


def queue_for(profile: ClusterProfile, seconds: float) -> Tuple[Queue, int]:
    queues = sorted(profile.queues, key=lambda q: q.max_hours)
    for q in queues:
        if seconds <= q.max_hours * 3600:
            return q, 1
    longest = queues[-1]
    return longest, max(1, math.ceil(seconds / (longest.max_hours * 3600)))


def estimate(profile: ClusterProfile, atoms: int, md: Sequence[Tuple[float, float]], minimisations: int = 0
             ) -> Optional[Estimate]:
    """The wall time of a job that runs MD stages ``md`` ((ns, dt in ps) each) and ``minimisations`` energy
    minimisations on a system of ``atoms`` atoms; None without an earlier run on the profile's GPU."""
    total_ns, md_s, used = 0.0, 0.0, []
    for ns, dt in md:
        got = speed(profile, atoms, dt)
        if got is None:
            return None
        rate, near = got
        md_s += ns * rate
        total_ns += ns
        used = near
    overhead = (OVERHEAD_MIN + PER_MIN_STAGE * minimisations) * 60
    cushion = clamp_cushion(profile.cushion)
    raw = (md_s + overhead) * (1 + cushion / 100)
    seconds = math.ceil(raw / (ROUND_MIN * 60)) * ROUND_MIN * 60
    queue, jobs = queue_for(profile, seconds)
    rate = md_s / total_ns if total_ns else 0.0
    gpu = used[0].gpu if used else profile.gpu
    basis = (f"{atoms:,} atoms, {total_ns:g} ns at {rate:.1f} s/ns ({gpu}; from {len(used)} earlier run"
             f"{'s' if len(used) != 1 else ''} of {', '.join(f'{b.atoms:,}' for b in used)} atoms) "
             f"+ {overhead / 60:g} min, + {cushion:g}% = {hms(seconds)}")
    if jobs > 1:
        return Estimate(queue.max_hours * 3600, queue, jobs, basis + f": over the longest queue ({queue.name}, "
                                                                    f"{queue.max_hours:g} h): {jobs} jobs in a row")
    return Estimate(seconds, queue, 1, basis)


def hms(seconds: float) -> str:
    minutes = int(math.ceil(seconds / 60))
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def time_line(profile: ClusterProfile, scheduler: str, seconds: float) -> str:
    """The wall-time directive: the profile's own ({time}, {hhmm}, {minutes}) or the scheduler's."""
    minutes = int(math.ceil(seconds / 60))
    values = {"{time}": hms(seconds), "{hhmm}": f"{minutes // 60}:{minutes % 60:02d}", "{minutes}": str(minutes)}
    template = profile.time_directive.strip() or {
        "Slurm": "#SBATCH --time={time}", "PBS / Torque": "#PBS -l walltime={time}", "LSF": "#BSUB -W {hhmm}",
        "Grid Engine": "#$ -l h_rt={time}"}.get(scheduler, "")
    for k, v in values.items():
        template = template.replace(k, v)
    return template


# a wall-time line the header already has: MDATK writes its own
TIME_LINE = re.compile(r"^\s*(#SBATCH\s+(--time[= ]|-t\s)|#PBS\s+-l\s+walltime|#BSUB\s+-W\s|#\$\s+-l\s+h_rt)")


def without_time(lines: Sequence[str]) -> Tuple[List[str], List[str]]:
    keep, dropped = [], []
    for ln in lines:
        (dropped if TIME_LINE.match(ln) else keep).append(ln)
    return keep, dropped


def grace_example() -> ClusterProfile:
    """A starting point for a new profile (Slurm, one GPU, a 48 h queue and a longer one)."""
    return ClusterProfile(
        name="New cluster",
        header="#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem=8G\n#SBATCH --gres=gpu:a100:1\n"
               "#SBATCH --output={job_name}.%j\n\nmodule purge\nmodule load Amber/24",
        gpu="A100",
        queues=[Queue("gpu", 48.0, "#SBATCH --partition=gpu"), Queue("long", 168.0, "#SBATCH --partition=long")])


def atoms_in(path: str) -> Optional[int]:
    """The atom count of a system file: an Amber topology (POINTERS), a GROMACS .gro, a PDB; None if unknown."""
    try:
        if path.endswith((".parm7", ".prmtop")):
            with open(path, encoding="utf-8", errors="replace") as fh:
                seen = False
                for line in fh:
                    if line.startswith("%FLAG POINTERS"):
                        seen = True
                    elif seen and not line.startswith("%"):
                        return int(line.split()[0])
            return None
        if path.endswith(".gro"):
            with open(path, encoding="utf-8", errors="replace") as fh:
                fh.readline()
                return int(fh.readline().split()[0])
        if path.endswith(".pdb"):
            with open(path, encoding="utf-8", errors="replace") as fh:
                return sum(1 for ln in fh if ln.startswith(("ATOM", "HETATM"))) or None
    except (OSError, ValueError, IndexError):
        return None
    return None
