"""Cluster job scripts for the steps with a separately run script (Amber MD, GROMACS production).

A step whose *Cluster job* section names a cluster profile (:mod:`mdatk.clusters`; or gives a job header of its own)
gets one more file, ``<project>_<NN>_job_<stamp>.<ext>``: the profile's header, the line that chooses the queue and
the wall time MDATK estimates for the job (the stages it runs, the system's atoms, the speeds of earlier runs on the
profile's GPU, plus the profile's cushion), then the commands that run the step there - its main script (the stages
before the separate ones; finished stages are skipped), its separate script(s), and its main script again (the
stages after them and the step's last files, e.g. the dry PDB) - so the step is complete when the job ends.  The
plan writes ``<project>_00_submit_<stamp>.sh`` beside its master script: it submits every step's job from the
step's folder, in plan order (Slurm: each job after the jobs of the steps it reads from, ``--dependency=afterok``;
a job longer than the longest queue as that many jobs in a row, ``--dependency=afterany``).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__, clusters
from .base import JOB_VAR, WAITING_STATUS, shell_quote
from .schema import GeneratedFile, Issue, Option, Section

JOB_TAG = "job"

# scheduler: directive prefix, file extension, how a job file is submitted ({name} job name, {file} job file)
SCHEDULERS: Sequence[Tuple[str, str, str, str]] = (
    ("Slurm", "#SBATCH", "slurm", "sbatch --job-name={name} {file}"),
    ("PBS / Torque", "#PBS", "pbs", "qsub -N {name} {file}"),
    ("LSF", "#BSUB", "lsf", "bsub -J {name} < {file}"),
    ("Grid Engine", "#$", "sge", "qsub -N {name} {file}"),
)
PLACEHOLDERS = ("{project}", "{step}", "{job_name}", "{folder}")

HEADER_EXAMPLE = """\
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output={job_name}.%j
module purge
module load Amber/24"""

CLUSTER_SECTION = Section("cluster", "Cluster job", [
    Option("cluster", "Cluster profile", "text", "",
           "Write a cluster job script for the step's separately run script(s), for this cluster profile (RNAdo: "
           "menu > Cluster profiles…; kept on this computer, for every project): its header, the queue and a "
           "wall time estimated from the stages the job runs, the system's atoms and the seconds/ns of earlier "
           "runs on the profile's GPU, plus the profile's cushion. The plan's submit script "
           "(<project>_00_submit_<stamp>.sh) submits every step's job from its folder.",
           placeholder="e.g. Grace"),
    Option("job_header", "Job header / extra lines", "multiline", "",
           "With a cluster profile: lines added after its header (e.g. an account). Without one: the whole header "
           "of the job script, written as it is - the scheduler's directives (#SBATCH, #PBS, #BSUB or #$) and the "
           "commands that set up the cluster's programs (module load ...); empty: no job script. {project}, "
           "{step}, {job_name} and {folder} are replaced, e.g. #SBATCH --output={job_name}.%j.",
           placeholder=HEADER_EXAMPLE, rows=6),
    Option("job_scope", "The job runs", "radio", "step",
           "The whole step: the main script (the stages before the separate ones; finished stages are skipped, "
           "so it costs nothing when they ran here), the separate script(s), then the main script again (the "
           "stages after them and the step's last files, e.g. the dry PDB): the step is complete when the job "
           "ends. The separate script(s) only: the main script runs here, before and after the job (RNAdo's Run, "
           "which waits for the separate script until its files are back).",
           choices=(("step", "The whole step"), ("separate", "The separate script(s) only"))),
    Option("job_submit", "Submit command", "text", "",
           "How the submit script submits the job ({name}: the job name <project>_<NN>, {file}: the job file). "
           "Empty: the cluster profile's, else from the header's directives - sbatch --job-name={name} {file} "
           "(#SBATCH), qsub -N {name} {file} (#PBS, #$), bsub -J {name} < {file} (#BSUB); bash {file} without "
           "any.", placeholder="sbatch --job-name={name} {file}"),
], tooltip="A job script for a batch scheduler that runs the step on a cluster.")


def _lines(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("#!"):
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def split_header(lines: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(directives, commands): a scheduler reads its directives only before the first command (Slurm stops at the
    first line that is neither blank nor a comment), so the leading comment lines are the directives."""
    for k, ln in enumerate(lines):
        if ln.strip() and not ln.lstrip().startswith("#"):
            head = list(lines[:k])
            while head and not head[-1].strip():
                head.pop()
            return head, list(lines[k:])
    return list(lines), []


def merge_header(*parts: Sequence[str]) -> List[str]:
    """Header parts in order, every part's directives before every command: [directives], blank, [commands]."""
    directives: List[str] = []
    commands: List[str] = []
    for part in parts:
        d, c = split_header(part)
        directives += d
        commands += c
    return directives + ([""] if directives and commands else []) + commands


def header_lines(s: Dict[str, Any]) -> List[str]:
    """The step's own job header lines (a leading #! line dropped)."""
    return _lines(s.get("job_header") or "")


def wants_job(s: Dict[str, Any]) -> bool:
    return bool((s.get("cluster") or "").strip() or header_lines(s))


def _scheduler_of(lines: Sequence[str]) -> Tuple[str, str, str]:
    for name, prefix, ext, submit in SCHEDULERS:
        if any(ln.lstrip().startswith(prefix) for ln in lines):
            return name, ext, submit
    return "", "job", "bash {file}"


def job_name(ctx) -> str:
    return f"{ctx.project}_{ctx.step:02d}"


@dataclass
class JobPlan:
    """What a step's job script holds: header lines (queue and wall time included), the scheduler, how to
    submit it, the estimate (or None) and the notes for the planner."""

    lines: List[str]
    scheduler: str
    ext: str
    submit: str
    profile: Optional[clusters.ClusterProfile] = None
    estimate: Optional[clusters.Estimate] = None
    jobs: int = 1
    issues: List[Issue] = field(default_factory=list)
    comment: List[str] = field(default_factory=list)


def _system_atoms(tool, s, ctx) -> Optional[int]:
    port = getattr(tool, "cluster_atoms_port", "")
    value = ctx.value(port) if port and ctx is not None else None
    if not value or not value.paths:
        return None
    path = value.paths[0]
    if not os.path.isabs(path):
        path = os.path.join(ctx.root or "", path)
    return clusters.atoms_in(path) if os.path.isfile(path) else None


def job_plan(s: Dict[str, Any], ctx, tool=None, profiles: Optional[Sequence[clusters.ClusterProfile]] = None
             ) -> Optional[JobPlan]:
    """The step's job script plan; None when the step asks for none."""
    if not wants_job(s):
        return None
    own = header_lines(s)
    name = (s.get("cluster") or "").strip()
    issues: List[Issue] = []
    profile = None
    if name:
        profile = clusters.find(name, profiles)
        if profile is None:
            known = [p.name for p in (clusters.load_profiles() if profiles is None else profiles)]
            issues.append(Issue("warning", f"no cluster profile {name!r} on this computer"
                                + (f" (there are: {', '.join(known)})" if known else " (none yet: RNAdo menu > "
                                                                                     "Cluster profiles…)")
                                + (": the job script has the step's own header only" if own else
                                   ": no job script"), ("cluster",)))
            if not own:
                return JobPlan([], "", "job", "", issues=issues)
    base = merge_header(_lines(profile.header), own) if profile else own
    lines, dropped = clusters.without_time(base) if profile else (base, [])
    if dropped:
        issues.append(Issue("info", "the wall time comes from the estimate: left out of the header: "
                            + "; ".join(ln.strip() for ln in dropped), ("job_header", "cluster")))
    sched, ext, submit = _scheduler_of(lines + (_lines(profile.queues[0].directives) if profile and
                                                profile.queues else []))
    submit = (s.get("job_submit") or "").strip() or (profile.submit.strip() if profile else "") or submit
    plan = JobPlan(lines, sched, ext, submit, profile, issues=issues)
    if profile is not None:
        _add_queue_and_time(plan, s, ctx, tool)
    if not sched and not (s.get("job_submit") or "").strip() and not (profile and profile.submit.strip()):
        plan.issues.append(Issue("info", "no scheduler directives (#SBATCH, #PBS, #BSUB, #$) in the job header: "
                                         "the submit script runs the job with bash (give a Submit command for "
                                         "another scheduler)", ("job_header", "job_submit")))
    if "{file}" not in plan.submit:
        plan.issues.append(Issue("error", "the submit command needs {file}, the job file", ("job_submit",)))
    unknown = sorted(set(re.findall(r"\{[a-z_]+\}", "\n".join(plan.lines))) - set(PLACEHOLDERS))
    if unknown:
        plan.issues.append(Issue("warning", f"left as written in the job header: {', '.join(unknown)} (replaced: "
                                            f"{', '.join(PLACEHOLDERS)})", ("job_header",)))
    if sched == "Slurm" and not any("--job-name" in ln or re.search(r"#SBATCH\s+-J\b", ln) for ln in plan.lines) \
            and "{name}" not in plan.submit:
        plan.issues.append(Issue("info", "no job name: Slurm names the job after the file", ("job_header",)))
    return plan


def _add_queue_and_time(plan: JobPlan, s, ctx, tool) -> None:
    """The queue line(s) and the wall time of a job with a cluster profile."""
    profile = plan.profile
    work = tool.cluster_work(s, s.get("job_scope", "step") == "separate") if hasattr(tool, "cluster_work") \
        else None
    atoms = _system_atoms(tool, s, ctx) if work is not None else None
    queues = sorted(profile.queues, key=lambda q: q.max_hours)
    est = None
    if work is None:
        why = "MDATK estimates the wall time of Amber pmemd.cuda steps only"
    elif atoms is None:
        why = ("the system's atoms are not known until its topology exists (the build has run): the job file is "
               "written again when the step waits for its separate script, or with Write cluster jobs")
    else:
        est = clusters.estimate(profile, atoms, *work)
        why = "" if est else (f"no earlier run on the profile's GPU{' (' + profile.gpu + ')' if profile.gpu else ''}"
                              f" to take the speed from: Cluster profiles… > Learn from runs, or a finished run "
                              f"of this plan (Write cluster jobs learns from it)")
    if est is None:
        queue = queues[-1] if queues else clusters.Queue()
        seconds = queue.max_hours * 3600
        plan.issues.append(Issue("info", f"wall time: the limit of queue {queue.name} ({queue.max_hours:g} h), as "
                                         f"{why}", ("cluster",)))
        plan.comment.append(f"# Wall time: the limit of queue {queue.name} ({why.split(':')[0]})")
    else:
        queue, seconds, plan.jobs = est.queue, est.seconds, est.jobs
        plan.estimate = est
        plan.comment.append(f"# Wall time: {est.basis}")
        level = "warning" if est.jobs > 1 else "info"
        plan.issues.append(Issue(level, f"wall time {clusters.hms(est.seconds)} in queue {queue.name}: "
                                        f"{est.basis}" + (" (finished stages and segments are skipped)"
                                                          if est.jobs > 1 else ""), ("cluster",)))
    plan.comment.insert(0, f"# Cluster profile {profile.name}, queue {queue.name} (up to {queue.max_hours:g} h)")
    extra = _lines(queue.directives)
    timeline = clusters.time_line(profile, plan.scheduler or _scheduler_of(extra)[0], seconds)
    # the queue and the wall time among the directives, before the first command (or the scheduler ignores them)
    plan.lines = merge_header(plan.lines, extra + ([timeline] if timeline else []))


def job_file(s: Dict[str, Any], ctx, files: Sequence[GeneratedFile], title: str = "", tool=None,
             plan: Optional[JobPlan] = None) -> Optional[GeneratedFile]:
    """The step's cluster job script, or None (no job asked for, or no separately run script)."""
    separate = [f.name for f in files if f.role == "run-separate"]
    plan = plan or job_plan(s, ctx, tool)
    if plan is None or not plan.lines or not separate:
        return None
    lines = list(plan.lines)
    values = {"{project}": ctx.project, "{step}": f"{ctx.step:02d}", "{job_name}": job_name(ctx),
              "{folder}": ctx.folder or "."}
    for key, value in values.items():
        lines = [ln.replace(key, value) for ln in lines]
    main = [f.name for f in files if f.role == "run"]
    name = ctx.name(JOB_TAG, plan.ext)
    whole = s.get("job_scope", "step") != "separate" and bool(main)
    first = shell_quote(main[0]) if main else ""
    out = ["#!/bin/bash"] + lines + [
        "",
        f"# Cluster job of project {ctx.project}, step {ctx.step:02d}" + (f" ({title})" if title else "")
        + f" - written by {PROJECT_ABBREV} {__version__} on {_dt.date.today().isoformat()}.",
        *plan.comment,
        "# Submit it from its step folder (the plan's submit script does): "
        + (f"cd {ctx.folder} && " if ctx.folder else "") + submit_line(plan, ctx, name),
        ("# Runs the main script (the stages before the separate ones; finished stages are skipped), the separate"
         " script(s), then the main script again (the stages after them and the step's last files)."
         if whole else "# Runs the separate script(s); the main script runs before and after the job (MDATK, "
                       "RNAdo's Run)."),
        "",
        "# the step folder: where the job was submitted from (a scheduler runs a copy of this file elsewhere)",
        'cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-${LS_SUBCWD:-${SGE_O_WORKDIR:-$(dirname "$0")}}}}"',
        "# the step's scripts skip their conda lines in the job (the modules above set up the programs)",
        f"export {JOB_VAR}=1",
        f"[ -f {shell_quote(separate[0])} ] || {{ echo \"ERROR: {separate[0]} is not in $PWD: submit the job "
        f"from its step folder{(' ' + ctx.folder) if ctx.folder else ''}\" >&2; exit 2; }}",
        "# the first job of the step here leaves this: runs at least as new ran on the cluster (their speeds are",
        "# learnt for the cluster profile, not those of runs on another computer)",
        f"[ -e {shell_quote(started_name(name))} ] || hostname > {shell_quote(started_name(name))} 2>/dev/null || true",
        "",
    ]
    if whole:
        out += [f"bash {first}",
                "status=$?",
                f"# {WAITING_STATUS}: waiting for the separate script (next)",
                f'if [ "$status" -ne 0 ] && [ "$status" -ne {WAITING_STATUS} ]; then',
                f'  echo "ERROR: {main[0]} failed (exit status $status)" >&2; exit "$status"',
                "fi"]
    for sep in separate:
        out.append(f"bash {shell_quote(sep)} || exit $?")
    if whole:
        out += ["# the stages after the separate ones and the step's last files", f"bash {first}"]
    what = plan.profile.name if plan.profile else (plan.scheduler or "no scheduler directives")
    return GeneratedFile(name, "\n".join(out) + "\n", "job",
                         f"cluster job ({what}): " + ("the whole step" if whole else "the separate script(s)"))


def started_name(job_file_name: str) -> str:
    """The note a step's first cluster job leaves in its folder (the host it ran on)."""
    return job_file_name + ".started"


def cluster_runs(step, root: str) -> List[str]:
    """The step's .out files that ran in its cluster job: of its separate script's MD runs, those at least as new
    as the note its first job left; [] when no job has run here."""
    if not hasattr(step.tool, "cluster_outputs") or step.ctx is None:
        return []
    folder = os.path.join(root, step.ctx.folder or ".")
    try:
        notes = [os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(".started")
                 and n.startswith(step.ctx.name(JOB_TAG))]
    except OSError:
        return []
    if not notes:
        return []
    since = min(os.path.getmtime(n) for n in notes) - 1
    paths = [os.path.join(folder, name) for name in step.tool.cluster_outputs(step.settings, step.ctx)]
    return [p for p in paths if os.path.isfile(p) and os.path.getmtime(p) >= since]


def submit_line(plan: JobPlan, ctx, name: str) -> str:
    return plan.submit.replace("{name}", job_name(ctx)).replace("{file}", shell_quote(name))


def job_issues(s: Dict[str, Any], ctx=None, tool=None) -> List[Issue]:
    """Planner notes on the Cluster job section."""
    plan = job_plan(s, ctx, tool) if ctx is not None else None
    return plan.issues if plan else []


def submit_script(plan) -> Optional[str]:
    """``<project>_00_submit_<stamp>.sh``: submits every step's cluster job from its folder, in plan order; None
    when no step has one.  Slurm jobs wait for the jobs of the steps they read from (--dependency=afterok); a job
    longer than the longest queue is submitted that many times in a row (--dependency=afterany)."""
    entries = []  # (node id, step, folder, job file, job plan, slurm)
    for nid in plan.order:
        step = plan.steps[nid]
        if step.tool is None or not step.tool.generates or step.adopted or not step.ok or step.ctx is None:
            continue
        job = next((f for f in step.files if f.role == "job"), None)
        if job is None:
            continue
        jp = job_plan(step.settings, step.ctx, step.tool)
        slurm = jp.scheduler == "Slurm" and jp.submit.startswith("sbatch ")
        entries.append((nid, step, step.ctx.folder or ".", job.name, jp, slurm))
    if not entries:
        return None
    ids = {nid: f"JOB_{step.node.step:02d}" for nid, step, *_rest in entries}
    parents: Dict[str, set] = {}
    for e in plan.workflow.edges:
        parents.setdefault(e.target, set()).add(e.source)

    def ancestors(nid: str) -> List[str]:
        seen, todo = set(), list(parents.get(nid, ()))
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo += parents.get(n, ())
        return [n for n in plan.order if n in seen]

    lines = ["#!/bin/bash",
             f"# Submits the cluster jobs of plan {plan.project} - written by {PROJECT_ABBREV} {__version__} on "
             f"{_dt.date.today().isoformat()}.",
             f"# Run it on the cluster, in the plan folder: bash {os.path.basename(submit_path(plan))}",
             "# Each job is submitted from its step folder. Steps without a job (structures, builds, restraints "
             "...) must have",
             "# run before (here, or copied over with the plan folder).",
             "", "set -eo pipefail", 'cd "$(dirname "$0")"', ""]
    for nid, step, folder, name, jp, slurm in entries:
        before = [ids[a] for a in ancestors(nid) if a in ids]
        lines.append(f"# {step.label}" + (f" ({jp.profile.name})" if jp.profile else ""))
        cmd = submit_line(jp, step.ctx, name)
        if slurm:
            var = ids[nid]
            dep = (" --dependency=afterok:" + ":".join(f"${b}" for b in before)) if before else ""
            lines += [f"{var}=$(cd {shell_quote(folder)} && sbatch --parsable{dep}{cmd[len('sbatch'):]})",
                      f'echo "{step.node.step:02d}: job ${var} ({folder}/{name})"']
            for k in range(2, jp.jobs + 1):  # longer than the longest queue: the same job again after it
                lines += [f"{var}=$(cd {shell_quote(folder)} && sbatch --parsable --dependency=afterany:${var}"
                          f"{cmd[len('sbatch'):]})",
                          f'echo "{step.node.step:02d}: job ${var}, part {k} of {jp.jobs} (after the one before)"']
        else:
            if before:
                lines.append("# runs after the jobs of the steps it reads from: submit it once they have finished")
            lines.append(f"(cd {shell_quote(folder)} && {cmd})")
            if jp.jobs > 1:
                lines.append(f"# longer than the longest queue: submit it again {jp.jobs - 1} more time(s), each "
                             "after the one before (finished stages and segments are skipped)")
        lines.append("")
    return "\n".join(lines)


def submit_path(plan) -> str:
    from .naming import file_name
    return os.path.join(plan.root, file_name(plan.project, 0, "submit", plan.stamp, "sh"))


def refresh(plan, node_ids: Optional[Sequence[str]] = None, learn: bool = True) -> List[str]:
    """Writes the job files of the plan's steps (``node_ids``: those only) and its submit script again - only
    those, so a completed run never moves - with what is known now (the topology a build has written, the speeds
    of runs that finished).  ``learn``: first adds the speeds of the finished runs in those steps' folders to their
    cluster profiles.  Returns lines for the output log."""
    from .plan import _write_text, generate_step
    out: List[str] = []
    targets = [n for n in plan.order if (node_ids is None or n in node_ids)]
    if learn:
        profiles = clusters.load_profiles()
        changed = False
        for nid in targets:
            step = plan.steps[nid]
            prof = clusters.find(step.settings.get("cluster", ""), profiles) if step.settings else None
            if prof is None or step.ctx is None or not plan.root:
                continue
            new = clusters.learn(prof, cluster_runs(step, plan.root))
            if new:
                changed = True
                out.append(f"{step.label}: {len(new)} run speed(s) for cluster profile {prof.name}")
        if changed:
            clusters.save_profiles(profiles)
    for nid in targets:
        step = plan.steps[nid]
        if step.tool is None or not step.tool.generates or not step.ok or step.adopted or step.ctx is None \
                or not wants_job(step.settings):
            continue
        step.files = generate_step(step)
        job = next((f for f in step.files if f.role == "job"), None)
        if job is None:
            continue
        main = next((f for f in step.files if f.role == "run"), None)
        if main is None or not os.path.isfile(os.path.join(plan.root, step.ctx.from_own(main.name))):
            out.append(f"{step.label}: not written yet - write the plan (or the step) first")
            continue
        path = os.path.join(plan.root, step.ctx.from_own(job.name))
        _write_text(path, job.content, True)
        est = job_plan(step.settings, step.ctx, step.tool)
        out.append(f"{step.label}: wrote {os.path.relpath(path, plan.root)}"
                   + (f" ({est.estimate.basis})" if est and est.estimate else ""))
    text = submit_script(plan)
    if text is not None and plan.root:
        _write_text(submit_path(plan), text, True)
        out.append(f"wrote {os.path.basename(submit_path(plan))}")
    return out
