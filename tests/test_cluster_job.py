"""Cluster job scripts (mdatk.workflows.cluster): a step with a separately run script and a job header gets
<project>_<NN>_job_<stamp>.<ext> - the header, then its main script, its separate script(s) and its main script
again - and the plan a submit script that submits every job from its step folder (Slurm: after the jobs of the
steps it reads from)."""
import os
import shutil
import stat
import subprocess

import pytest

from mdatk.config import PlanEdge, PlanNode, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan

STAMP = "20261113-1200"
needs_bash = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")

SLURM = """#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output={job_name}.%j

module purge
module load Amber/24"""


def md(name, **kw):
    return dict(dict(name=name, kind="md", nstlim=1000, ntwx=500), **kw)


def md_node(nid, step, files=None, **settings):
    s = dict(engine="pmemd.cuda", stages=[md("eq1"), md("prod1", separate="yes")], dry_pdb=False,
             input_velocities=True,
             separate_production=True)
    s.update(settings)
    return PlanNode(id=nid, tool="amber-md", step=step, settings=TOOLS["amber-md"].normalise(s), files=files or {})


def plan_of(tmp_path, nodes, edges=()):
    (tmp_path / "in").mkdir(exist_ok=True)
    for name in ("sys.parm7", "sys.rst7"):
        (tmp_path / "in" / name).write_text("x\n")
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=nodes, edges=list(edges)),
                str(tmp_path / "plan"))
    plan.write()
    return plan


def inputs(tmp_path):
    return {"topology": str(tmp_path / "in" / "sys.parm7"), "coordinates": str(tmp_path / "in" / "sys.rst7")}


def test_no_job_header_writes_what_it_wrote_before(tmp_path):
    plan = plan_of(tmp_path, [md_node("md", 1, inputs(tmp_path), conda_env="AmberTools26")])
    folder = tmp_path / "plan" / "01_amber-md"
    assert not [n for n in os.listdir(folder) if "_job_" in n]
    assert not [n for n in os.listdir(tmp_path / "plan") if "_submit_" in n]
    main = (folder / f"apt1_01_amber-md_{STAMP}.sh").read_text()
    assert 'eval "$(conda shell.bash hook)"\nconda activate AmberTools26\n' in main and "MDATK_JOB" not in main
    assert not [i for i in plan.steps["md"].issues if "job_header" in i.keys]


def test_a_slurm_job_runs_the_step_and_the_submit_script_submits_it(tmp_path):
    plan = plan_of(tmp_path, [md_node("md", 1, inputs(tmp_path), job_header=SLURM, conda_env="AmberTools26")])
    folder = tmp_path / "plan" / "01_amber-md"
    job = (folder / f"apt1_01_job_{STAMP}.slurm").read_text()
    assert job.startswith("#!/bin/bash\n#SBATCH --time=12:00:00\n")
    assert "#SBATCH --output=apt1_01.%j\n\nmodule purge\nmodule load Amber/24\n" in job  # placeholders filled
    assert 'cd "${SLURM_SUBMIT_DIR:-' in job and "export MDATK_JOB=1" in job
    main, prod = f"apt1_01_amber-md_{STAMP}.sh", f"apt1_01_amber-prod_{STAMP}.sh"
    body = job[job.index("export MDATK_JOB=1"):]
    assert body.index(f"bash {main}") < body.index(f"bash {prod}") < body.rindex(f"bash {main}")
    assert os.access(folder / f"apt1_01_job_{STAMP}.slurm", os.X_OK)
    # the main script keeps conda except in the job; the separate one as before (only when MDATK runs it)
    text = (folder / main).read_text()
    assert 'if [ -z "${MDATK_JOB:-}" ]; then\n  eval "$(conda shell.bash hook)"\n  conda activate AmberTools26\nfi' \
        in text
    assert 'if [ -n "${MDATK_LOCAL:-}" ]; then' in (folder / prod).read_text()
    submit = (tmp_path / "plan" / f"apt1_00_submit_{STAMP}.sh").read_text()
    assert f"JOB_01=$(cd 01_amber-md && sbatch --parsable --job-name=apt1_01 apt1_01_job_{STAMP}.slurm)" in submit
    master = (tmp_path / "plan" / f"apt1_00_plan_{STAMP}.sh").read_text()
    assert f"#   cluster job: 01_amber-md/apt1_01_job_{STAMP}.slurm" in master
    assert [i.level for i in plan.steps["md"].issues if "job_header" in i.keys] == []


def test_other_schedulers_scope_and_checks(tmp_path):
    pbs = "#PBS -l walltime=12:00:00\n#PBS -q gpu\nmodule load amber"
    plan = plan_of(tmp_path, [
        md_node("a", 1, inputs(tmp_path), job_header=pbs, job_scope="separate"),
        md_node("b", 2, inputs(tmp_path), job_header="module load amber\n{nodes}"),
        md_node("c", 3, inputs(tmp_path), job_header=SLURM, job_submit="sbatch --account=x"),
        md_node("d", 4, inputs(tmp_path), job_header=SLURM, separate_production=False,
                stages=[md("eq1"), md("prod1")])])
    root = tmp_path / "plan"
    job = (root / "01_amber-md" / f"apt1_01_job_{STAMP}.pbs").read_text()
    assert 'cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-' in job
    assert f"bash apt1_01_amber-md_{STAMP}.sh" not in job and f"bash apt1_01_amber-prod_{STAMP}.sh" in job
    assert (root / "02_amber-md" / f"apt1_02_job_{STAMP}.job").exists()
    assert not [n for n in os.listdir(root / "04_amber-md") if "_job_" in n]  # no separate script: no job
    submit = (root / f"apt1_00_submit_{STAMP}.sh").read_text()
    assert f"(cd 01_amber-md && qsub -N apt1_01 apt1_01_job_{STAMP}.pbs)" in submit
    assert f"(cd 02_amber-md && bash apt1_02_job_{STAMP}.job)" in submit
    notes = {nid: [(i.level, i.message) for i in plan.steps[nid].issues if set(i.keys) & {"job_header", "job_submit"}]
             for nid in "abcd"}
    assert notes["a"] == []
    assert [lv for lv, _ in notes["b"]] == ["info", "warning"] and "{nodes}" in notes["b"][1][1]
    assert notes["c"] == [("error", "the submit command needs {file}, the job file"),
                          ("info", "no job name: Slurm names the job after the file")]


def _stub(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@needs_bash
@pytest.mark.parametrize("main_status, expect", [(75, ["main", "prod", "main"]), (0, ["main", "prod", "main"]),
                                                 (3, ["main"])])
def test_the_job_runs_main_separate_main_from_the_submit_folder(tmp_path, main_status, expect):
    plan_of(tmp_path, [md_node("md", 1, inputs(tmp_path), job_header=SLURM)])
    folder = tmp_path / "plan" / "01_amber-md"
    log = tmp_path / "calls.log"
    # stand-ins for the step's scripts: the main script waits (75) the first time, then finishes
    _stub(folder / f"apt1_01_amber-md_{STAMP}.sh",
          f'echo "main $MDATK_JOB" >> {log}\n[ -e {tmp_path}/once ] && exit 0\ntouch {tmp_path}/once\n'
          f"exit {main_status}\n")
    _stub(folder / f"apt1_01_amber-prod_{STAMP}.sh", f'echo "prod $MDATK_JOB" >> {log}\n')
    elsewhere = tmp_path / "spool"
    elsewhere.mkdir()
    shutil.copy(folder / f"apt1_01_job_{STAMP}.slurm", elsewhere / "job")  # Slurm runs a copy of the file
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(folder))
    env.pop("MDATK_JOB", None)
    res = subprocess.run(["bash", str(elsewhere / "job")], env=env, capture_output=True, text=True, timeout=60)
    assert [ln.split()[0] for ln in log.read_text().splitlines()] == expect
    assert all(ln.split()[1] == "1" for ln in log.read_text().splitlines())  # MDATK_JOB=1 in every script
    assert res.returncode == (3 if main_status == 3 else 0), res.stderr
    # submitted from the wrong folder: stops at once, naming the folder
    log.unlink()
    res = subprocess.run(["bash", str(elsewhere / "job")], env=dict(env, SLURM_SUBMIT_DIR=str(elsewhere)),
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 2 and "submit the job from its step folder 01_amber-md" in res.stderr
    assert not log.exists()


@needs_bash
def test_the_submit_script_chains_dependent_slurm_jobs(tmp_path):
    first = md_node("a", 1, inputs(tmp_path), job_header=SLURM)
    second = md_node("b", 2, job_header=SLURM)
    third = md_node("c", 3, inputs(tmp_path), job_header=SLURM)
    edges = [PlanEdge("a", "topology", "b", "topology"), PlanEdge("a", "restart", "b", "coordinates")]
    plan_of(tmp_path, [first, second, third], edges)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    _stub(bin_dir / "sbatch", f'n=$(( $(wc -l < {log} 2>/dev/null || echo 0) + 101 ))\n'
                              f'echo "$PWD $*" >> {log}\necho "$n"\n')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    res = subprocess.run(["bash", str(tmp_path / "plan" / f"apt1_00_submit_{STAMP}.sh")], env=env,
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    calls = log.read_text().splitlines()
    assert calls[0].endswith(f"01_amber-md --parsable --job-name=apt1_01 apt1_01_job_{STAMP}.slurm")
    assert calls[1].endswith(f"02_amber-md --parsable --dependency=afterok:101 --job-name=apt1_02 "
                             f"apt1_02_job_{STAMP}.slurm")
    assert "--dependency" not in calls[2]
    assert "02: job 102 (02_amber-md/" in res.stdout
