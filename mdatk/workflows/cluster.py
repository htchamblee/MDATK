"""Cluster job scripts for the steps with a separately run script (Amber MD, GROMACS production).

A step whose *Cluster job* section has a job header gets one more file, ``<project>_<NN>_job_<stamp>.<ext>``:
the header (the scheduler's directives and the module loads of the cluster), then the commands that run the step
there - its main script (the stages before the separate ones; finished stages are skipped), its separate
script(s), and its main script again (the stages after them and the step's last files, e.g. the dry PDB) - so the
step is complete when the job ends.  The plan writes ``<project>_00_submit_<stamp>.sh`` beside its master
script: it submits every step's job from the step's folder, in plan order (Slurm: each job after the jobs of the
steps it reads from, ``--dependency=afterok``).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import PROJECT_ABBREV, __version__
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
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output={job_name}.%j
module purge
module load Amber/24"""

CLUSTER_SECTION = Section("cluster", "Cluster job", [
    Option("job_header", "Job header", "multiline", "",
           "Write a cluster job script for the step's separately run script(s): these lines at its top (after "
           "#!/bin/bash) - the scheduler's directives (#SBATCH, #PBS, #BSUB or #$) and the commands that set up "
           "the cluster's programs (module load ...). Empty: no job script. {project}, {step}, {job_name} and "
           "{folder} are replaced, e.g. #SBATCH --output={job_name}.%j. The plan's submit script "
           "(<project>_00_submit_<stamp>.sh) submits every step's job from its folder.",
           placeholder=HEADER_EXAMPLE, rows=8),
    Option("job_scope", "The job runs", "radio", "step",
           "The whole step: the main script (the stages before the separate ones; finished stages are skipped, "
           "so it costs nothing when they ran here), the separate script(s), then the main script again (the "
           "stages after them and the step's last files, e.g. the dry PDB): the step is complete when the job "
           "ends. The separate script(s) only: the main script runs here, before and after the job (RNAdo's Run, "
           "which waits for the separate script until its files are back).",
           choices=(("step", "The whole step"), ("separate", "The separate script(s) only"))),
    Option("job_submit", "Submit command", "text", "",
           "How the submit script submits the job ({name}: the job name <project>_<NN>, {file}: the job file). "
           "Empty: from the header's directives - sbatch --job-name={name} {file} (#SBATCH), qsub -N {name} "
           "{file} (#PBS, #$), bsub -J {name} < {file} (#BSUB); bash {file} without any.",
           placeholder="sbatch --job-name={name} {file}"),
], tooltip="A job script for a batch scheduler that runs the step on a cluster.")


def header_lines(s: Dict[str, Any]) -> List[str]:
    """The job header's lines (trailing spaces and a leading #! line dropped); [] for no job script."""
    lines = [ln.rstrip() for ln in (s.get("job_header") or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("#!"):
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def scheduler(s: Dict[str, Any]) -> Tuple[str, str, str]:
    """(name, file extension, submit command template) of the step's job header."""
    lines = header_lines(s)
    for name, prefix, ext, submit in SCHEDULERS:
        if any(ln.lstrip().startswith(prefix) for ln in lines):
            return name, ext, (s.get("job_submit") or "").strip() or submit
    return "", "job", (s.get("job_submit") or "").strip() or "bash {file}"


def job_name(ctx) -> str:
    return f"{ctx.project}_{ctx.step:02d}"


def job_file_name(s: Dict[str, Any], ctx) -> str:
    return ctx.name(JOB_TAG, scheduler(s)[1])


def job_file(s: Dict[str, Any], ctx, files: Sequence[GeneratedFile], title: str = "") -> Optional[GeneratedFile]:
    """The step's cluster job script, or None (no job header, or no separately run script)."""
    lines = header_lines(s)
    separate = [f.name for f in files if f.role == "run-separate"]
    if not lines or not separate:
        return None
    main = [f.name for f in files if f.role == "run"]
    values = {"{project}": ctx.project, "{step}": f"{ctx.step:02d}", "{job_name}": job_name(ctx),
              "{folder}": ctx.folder or "."}
    for key, value in values.items():
        lines = [ln.replace(key, value) for ln in lines]
    name = job_file_name(s, ctx)
    sched, _ext, _submit = scheduler(s)
    whole = s.get("job_scope", "step") != "separate" and bool(main)
    first = shell_quote(main[0]) if main else ""
    out = ["#!/bin/bash"] + lines + [
        "",
        f"# Cluster job of project {ctx.project}, step {ctx.step:02d}" + (f" ({title})" if title else "")
        + f" - written by {PROJECT_ABBREV} {__version__} on {_dt.date.today().isoformat()}.",
        "# Submit it from its step folder (the plan's submit script does): "
        + (f"cd {ctx.folder} && " if ctx.folder else "") + _submit_line(s, ctx, name),
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
    return GeneratedFile(name, "\n".join(out) + "\n", "job",
                         f"cluster job ({sched or 'no scheduler directives'}): " +
                         ("the whole step" if whole else "the separate script(s)"))


def _submit_line(s: Dict[str, Any], ctx, name: str) -> str:
    return scheduler(s)[2].replace("{name}", job_name(ctx)).replace("{file}", shell_quote(name))


def job_issues(s: Dict[str, Any]) -> List[Issue]:
    """Planner notes on the Cluster job section."""
    lines = header_lines(s)
    if not lines:
        return []
    keys = ("job_header",)
    out = []
    sched, _ext, submit = scheduler(s)
    if not sched and not (s.get("job_submit") or "").strip():
        out.append(Issue("info", "no scheduler directives (#SBATCH, #PBS, #BSUB, #$) in the job header: the submit "
                                 "script runs the job with bash (give a Submit command for another scheduler)",
                         keys + ("job_submit",)))
    if (s.get("job_submit") or "").strip() and "{file}" not in s["job_submit"]:
        out.append(Issue("error", "the submit command needs {file}, the job file", ("job_submit",)))
    unknown = sorted(set(re.findall(r"\{[a-z_]+\}", "\n".join(lines))) - set(PLACEHOLDERS))
    if unknown:
        out.append(Issue("warning", f"left as written in the job header: {', '.join(unknown)} (replaced: "
                                    f"{', '.join(PLACEHOLDERS)})", keys))
    if sched == "Slurm" and not any("--job-name" in ln or re.search(r"#SBATCH\s+-J\b", ln) for ln in lines) \
            and "{name}" not in submit:
        out.append(Issue("info", "no job name: Slurm names the job after the file", keys))
    return out


def submit_script(plan) -> Optional[str]:
    """``<project>_00_submit_<stamp>.sh``: submits every step's cluster job from its folder, in plan order; None
    when no step has one.  Slurm jobs wait for the jobs of the steps they read from (--dependency=afterok)."""
    entries = []  # (node id, step, folder, job file, submit template, slurm)
    for nid in plan.order:
        step = plan.steps[nid]
        if step.tool is None or not step.tool.generates or step.adopted or not step.ok or step.ctx is None:
            continue
        job = next((f for f in step.files if f.role == "job"), None)
        if job is None:
            continue
        sched, _ext, submit = scheduler(step.settings)
        slurm = sched == "Slurm" and submit.startswith("sbatch ")
        entries.append((nid, step, step.ctx.folder or ".", job.name, submit, slurm))
    if not entries:
        return None
    ids = {nid: f"JOB_{step.node.step:02d}_{k}" for k, (nid, step, *_rest) in enumerate(entries)}
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
             "# Run it on the cluster, in the plan folder: bash " + os.path.basename(
                 _submit_name(plan)),
             "# Each job is submitted from its step folder. Steps without a job (structures, builds, restraints "
             "...) must have",
             "# run before (here, or copied over with the plan folder).",
             "", "set -eo pipefail", 'cd "$(dirname "$0")"', ""]
    for nid, step, folder, name, submit, slurm in entries:
        before = [ids[a] for a in ancestors(nid) if a in ids]
        lines.append(f"# {step.label}")
        cmd = submit.replace("{name}", f"{plan.project}_{step.node.step:02d}").replace("{file}", shell_quote(name))
        if slurm:
            dep = (" --dependency=afterok:" + ":".join(f"${b}" for b in before)) if before else ""
            cmd = "sbatch --parsable" + dep + cmd[len("sbatch"):]
            lines += [f"{ids[nid]}=$(cd {shell_quote(folder)} && {cmd})",
                      f'echo "{step.node.step:02d}: job ${ids[nid]} ({folder}/{name})"']
        else:
            if before:
                lines.append("# runs after the jobs of the steps it reads from: submit it once they have finished")
            lines.append(f"(cd {shell_quote(folder)} && {cmd})")
        lines.append("")
    return "\n".join(lines)


def _submit_name(plan) -> str:
    from .naming import file_name
    return os.path.join(plan.root, file_name(plan.project, 0, "submit", plan.stamp, "sh"))


def submit_path(plan) -> str:
    return _submit_name(plan)
