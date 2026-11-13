"""Cluster profiles (mdatk.clusters): kept per user outside project files; speeds learnt from pmemd.cuda .out files;
a job's wall time from the stages it runs, the system's atoms and the nearest earlier runs, plus a cushion, in the
shortest queue that holds it - and the job scripts, planner notes and submit script that use them."""
import os
import shutil
import subprocess

import pytest

from mdatk import clusters
from mdatk.config import PlanNode, WorkflowParams
from mdatk.workflows import TOOLS, cluster
from mdatk.workflows.plan import Plan

STAMP = "20261113-1200"
needs_bash = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home


def pmemd_out(path, gpu="NVIDIA A100-SXM4-40GB", natom=50000, dt=0.004, s_per_ns=160.0, finished=True):
    """A pmemd.cuda .out file: the parts the speed is read from."""
    text = [" Amber 24 PMEMD", "|------------------- GPU DEVICE INFO --------------------",
            f"|                         CUDA Device Name: {gpu}" if gpu else "",
            "", f" NATOM  = {natom:8d} NTYPES =      18 NBONH =   47000 MBONA  =    2400",
            "Molecular dynamics:", "     nstlim  =   5000000, nscm    =      1000, nrespa  =         1",
            f"     t       =   0.00000, dt      =   {dt:.5f}, vlimit  =  -1.00000", "",
            "|     Average timings for last    5000 steps:",
            "|         Elapsed(s) =      10.00 Per Step(ms) =       2.00",
            "|             ns/day =      99.00   seconds/ns =     999.00"]
    if finished:
        text += ["|     Average timings for all steps:",
                 "|         Elapsed(s) =    3200.00 Per Step(ms) =       0.64",
                 f"|             ns/day =     540.00   seconds/ns =     {s_per_ns:.2f}",
                 "|  Final Performance Info:"]
    path.write_text("\n".join(text) + "\n")
    return path


def grace(**kw):
    p = clusters.ClusterProfile(
        name="Grace", header="#SBATCH --nodes=1\n#SBATCH --time=12:00:00\n#SBATCH --gres=gpu:a100:1\n"
                             "#SBATCH --output={job_name}.%j\nmodule load Amber/24",
        gpu="A100", queues=[clusters.Queue("long", 168.0, "#SBATCH --partition=long"),
                            clusters.Queue("gpu", 48.0, "#SBATCH --partition=gpu")],
        benchmarks=[clusters.Benchmark("NVIDIA A100-SXM4-40GB", 50000, 0.004, 160.0, "2026-11-12", "/x/a.out")])
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_profiles_are_kept_per_user_and_outside_projects(config_home):
    assert clusters.load_profiles() == []
    faster = grace(name="FASTER", gpu="A40", cushion=55.0, benchmarks=[])
    clusters.save_profiles([grace(), faster])
    assert os.path.isfile(config_home / "mdatk" / "clusters.toml")
    back = clusters.load_profiles()
    assert [p.name for p in back] == ["Grace", "FASTER"]
    assert back[0] == grace()  # queues and run speeds kept
    assert back[1].cushion == 20.0  # the cushion stays within 1-20%
    assert clusters.find("FASTER").gpu == "A40" and clusters.find("Nope") is None


def test_speeds_come_from_finished_pmemd_cuda_runs(tmp_path):
    run = tmp_path / "runs"
    run.mkdir()
    pmemd_out(run / "prod-001.out", natom=52345, dt=0.004, s_per_ns=159.73)
    pmemd_out(run / "heat.out", finished=False)  # not finished: no 'all steps' timing
    pmemd_out(run / "cpu.out", gpu="")  # a CPU run
    (run / "min1.out").write_text("NATOM = 100\n minimization\n")
    (run / "superseded").mkdir()
    pmemd_out(run / "superseded" / "old.out")
    b = clusters.parse_out(str(run / "prod-001.out"))
    assert (b.gpu, b.atoms, b.dt, b.s_per_ns) == ("NVIDIA A100-SXM4-40GB", 52345, 0.004, 159.73)
    p = grace(benchmarks=[])
    assert [x.source for x in clusters.learn(p, [str(run)])] == [str(run / "prod-001.out")]
    clusters.learn(p, [str(run / "prod-001.out")])  # the same file again: replaced, not added
    assert len(p.benchmarks) == 1


def test_the_wall_time_scales_with_atoms_time_step_and_ns():
    p = grace()
    # 200 ns + 6 ns at 4 fs on 50,000 atoms: 206 x 160 s, + 5 min + 2 x 2 min (minimisations), + 10%
    est = clusters.estimate(p, 50000, [(6.0, 0.004), (200.0, 0.004)], minimisations=2)
    raw = (206 * 160 + 9 * 60) * 1.10
    assert est.seconds == pytest.approx(-(-raw // 300) * 300)
    assert clusters.hms(est.seconds) == "10:15:00" and est.queue.name == "gpu" and est.jobs == 1
    assert "50,000 atoms, 206 ns at 160.0 s/ns (NVIDIA A100-SXM4-40GB" in est.basis and "+ 10%" in est.basis
    # twice the atoms and half the time step: four times the seconds/ns
    assert clusters.speed(p, 100000, 0.002)[0] == pytest.approx(640.0)
    # the nearest runs in size set the speed (median), not a far one
    p.benchmarks += [clusters.Benchmark("NVIDIA A100-SXM4-40GB", 52000, 0.004, 170.0),
                     clusters.Benchmark("NVIDIA A100-SXM4-40GB", 49000, 0.004, 150.0),
                     clusters.Benchmark("NVIDIA A100-SXM4-40GB", 400000, 0.004, 9000.0)]
    assert clusters.speed(p, 50000, 0.004)[0] == pytest.approx(160.0 * 1.0, rel=0.03)
    # another GPU's runs do not count; none of the profile's GPU: no estimate
    assert clusters.estimate(grace(gpu="H100"), 50000, [(200.0, 0.004)]) is None


def test_queue_choice_and_jobs_in_a_row():
    p = grace()
    long_run = clusters.estimate(p, 50000, [(1200.0, 0.004)])  # ~59 h: over 48 h, into the long queue
    assert long_run.queue.name == "long" and long_run.jobs == 1
    huge = clusters.estimate(p, 50000, [(5000.0, 0.004)])  # ~245 h: over the longest queue (168 h)
    assert huge.jobs == 2 and huge.seconds == 168 * 3600 and "2 jobs in a row" in huge.basis


@pytest.mark.parametrize("sched, line", [("Slurm", "#SBATCH --time=10:15:00"), ("PBS / Torque", "#PBS -l walltime=10:15:00"),
                                         ("LSF", "#BSUB -W 10:15"), ("Grid Engine", "#$ -l h_rt=10:15:00")])
def test_time_directives(sched, line):
    assert clusters.time_line(grace(), sched, 36900) == line
    assert clusters.time_line(grace(time_directive="#SBATCH -t {minutes}"), sched, 36900) == "#SBATCH -t 615"


def parm7(path, natom):
    path.write_text(f"%VERSION\n%FLAG TITLE\n%FORMAT(20a4)\nsys\n%FLAG POINTERS\n%FORMAT(10I8)\n"
                    f"{natom:8d}      18\n")
    return path


def md_plan(tmp_path, profiles=None, atoms=50000, **settings):
    if profiles is not None:
        clusters.save_profiles(profiles)
    (tmp_path / "in").mkdir(parents=True, exist_ok=True)
    top = parm7(tmp_path / "in" / "sys.parm7", atoms) if atoms else tmp_path / "in" / "missing.parm7"
    (tmp_path / "in" / "sys.rst7").write_text("x\n")
    stages = [dict(name="min1", kind="min"), dict(name="eq", kind="md", nstlim=1500000, dt=0.004, ntwx=12500),
              dict(name="prod", kind="md", nstlim=5000000, dt=0.004, segments=10, ntwx=12500, separate="yes")]
    s = dict(engine="pmemd.cuda", stages=stages, dry_pdb=False, input_velocities=True, cluster="Grace")
    s.update(settings)
    node = PlanNode(id="md", tool="amber-md", step=4, settings=TOOLS["amber-md"].normalise(s),
                    files={"topology": str(top), "coordinates": str(tmp_path / "in" / "sys.rst7")})
    plan = Plan(WorkflowParams(project_name="dlval", stamp=STAMP, nodes=[node]), str(tmp_path / "plan"))
    plan.write()
    return plan


def notes(plan):
    return [(i.level, i.message) for i in plan.steps["md"].issues if "cluster" in i.keys]


def test_a_profile_job_gets_the_queue_and_the_estimated_wall_time(tmp_path):
    plan = md_plan(tmp_path, [grace()], conda_env="AmberTools26")
    assert "chem123" not in (tmp_path / "plan" / "04_amber-md" / f"dlval_04_job_{STAMP}.slurm").read_text()
    plan = md_plan(tmp_path, [grace()], job_header="#SBATCH --account=chem123", conda_env="AmberTools26")
    job = (tmp_path / "plan" / "04_amber-md" / f"dlval_04_job_{STAMP}.slurm").read_text()
    head = job[:job.index("\n\n# Cluster job")].splitlines()
    # the profile's header without its own --time, the step's extra line, the queue, the estimate
    # every directive before the first command: Slurm reads none after it
    assert head == ["#!/bin/bash", "#SBATCH --nodes=1", "#SBATCH --gres=gpu:a100:1", "#SBATCH --output=dlval_04.%j",
                    "#SBATCH --account=chem123", "#SBATCH --partition=gpu", "#SBATCH --time=10:15:00", "",
                    "module load Amber/24"]
    assert "# Cluster profile Grace, queue gpu (up to 48 h)" in job
    assert "# Wall time: 50,000 atoms, 206 ns at 160.0 s/ns" in job
    levels = notes(plan)
    assert [lv for lv, _ in levels] == ["info", "info"]
    assert "left out of the header: #SBATCH --time=12:00:00" in levels[0][1]
    assert levels[1][1].startswith("wall time 10:15:00 in queue gpu: 50,000 atoms")
    main = (tmp_path / "plan" / "04_amber-md" / f"dlval_04_amber-md_{STAMP}.sh").read_text()
    assert 'if [ -z "${MDATK_JOB:-}" ]; then\n  eval "$(conda shell.bash hook)"' in main  # a profile alone too
    submit = (tmp_path / "plan" / f"dlval_00_submit_{STAMP}.sh").read_text()
    assert "# 04 · " in submit and "(Grace)\nJOB_04=$(cd 04_amber-md && sbatch --parsable --job-name=dlval_04" in submit


def test_without_atoms_or_speeds_the_job_asks_for_the_queue_limit(tmp_path):
    plan = md_plan(tmp_path, [grace()], atoms=0)
    job = (tmp_path / "plan" / "04_amber-md" / f"dlval_04_job_{STAMP}.slurm").read_text()
    assert "#SBATCH --partition=long\n#SBATCH --time=168:00:00\n\nmodule load Amber/24\n" in job
    assert any("the system's atoms are not known until its topology exists" in m for _, m in notes(plan))
    plan = md_plan(tmp_path / "b", [grace(benchmarks=[])])
    assert any("no earlier run on the profile's GPU (A100)" in m for _, m in notes(plan))


def test_a_missing_profile_writes_no_job(tmp_path):
    plan = md_plan(tmp_path, [grace(name="FASTER")])
    assert not [n for n in os.listdir(tmp_path / "plan" / "04_amber-md") if "_job_" in n]
    assert notes(plan) == [("warning", "no cluster profile 'Grace' on this computer (there are: FASTER): "
                                       "no job script")]


def test_a_job_longer_than_the_longest_queue_is_submitted_in_parts(tmp_path):
    plan = md_plan(tmp_path, [grace(queues=[clusters.Queue("gpu", 4.0, "#SBATCH --partition=gpu")])])
    assert any(lv == "warning" and "3 jobs in a row" in m for lv, m in notes(plan))
    submit = (tmp_path / "plan" / f"dlval_00_submit_{STAMP}.sh").read_text()
    assert submit.count("--dependency=afterany:$JOB_04") == 2 and "part 3 of 3" in submit


@needs_bash
def test_refresh_learns_the_cluster_runs_and_writes_the_job_again(tmp_path):
    plan = md_plan(tmp_path, [grace(benchmarks=[])])
    folder = tmp_path / "plan" / "04_amber-md"
    job_path = folder / f"dlval_04_job_{STAMP}.slurm"
    assert "--time=168:00:00" in job_path.read_text()  # no speed yet: the queue limit
    # a run of this step on another computer is not the cluster's: no note from a job, nothing learnt
    pmemd_out(folder / f"dlval_04_prod-001_{STAMP}.out", s_per_ns=100.0)
    assert cluster.refresh(plan, ["md"])[-1].startswith("wrote dlval_00_submit")
    assert clusters.find("Grace").benchmarks == []
    # the job ran there (its note), then its production segments
    (folder / f"dlval_04_job_{STAMP}.slurm.started").write_text("grace-c123\n")
    os.utime(folder / f"dlval_04_job_{STAMP}.slurm.started", (1, 1))
    pmemd_out(folder / f"dlval_04_prod-002_{STAMP}.out", s_per_ns=160.0)
    pmemd_out(folder / f"dlval_04_eq_{STAMP}.out", s_per_ns=10.0)  # main script's stage: not the separate script's
    lines = cluster.refresh(plan, ["md"])
    assert "2 run speed(s) for cluster profile Grace" in lines[0]  # prod-001, -002; not eq
    assert sorted(b.s_per_ns for b in clusters.find("Grace").benchmarks) == [100.0, 160.0]
    assert "--time=" in job_path.read_text() and "--time=168:00:00" not in job_path.read_text()


@needs_bash
def test_the_job_leaves_its_note_once(tmp_path):
    md_plan(tmp_path, [grace()])
    folder = tmp_path / "plan" / "04_amber-md"
    for name in (f"dlval_04_amber-md_{STAMP}.sh", f"dlval_04_amber-prod_{STAMP}.sh"):
        (folder / name).write_text("#!/bin/bash\nexit 0\n")
    job = folder / f"dlval_04_job_{STAMP}.slurm"
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(folder))
    assert subprocess.run(["bash", str(job)], env=env, capture_output=True, timeout=60).returncode == 0
    note = folder / f"dlval_04_job_{STAMP}.slurm.started"
    first = note.read_text()
    assert first.strip()
    os.utime(note, (1, 1))
    subprocess.run(["bash", str(job)], env=env, capture_output=True, timeout=60)
    assert os.path.getmtime(note) == 1  # kept: the first job's time


def test_the_runner_writes_the_job_again_when_the_step_waits(tmp_path, monkeypatch):
    from mdatk.workflows import runner
    plan = md_plan(tmp_path, [grace()], atoms=0)
    job = next(j for j in runner.plan_jobs(plan) if j.role == "run")
    assert job.on_wait is not None
    calls = []
    monkeypatch.setattr(cluster, "refresh", lambda p, ids, learn=True: calls.append(ids) or ["wrote x"])
    assert job.on_wait() == ["wrote x"] and calls == [["md"]]


def test_cli_cluster_learn_and_list(tmp_path, capsys):
    from mdatk.cli import main
    clusters.save_profiles([grace(benchmarks=[])])
    pmemd_out(tmp_path / "prod.out", natom=60000, s_per_ns=190.0)
    assert main(["cluster", "learn", "Grace", str(tmp_path)]) == 0
    assert "1 run speed(s) added to Grace (1 on its GPU)" in capsys.readouterr().out
    assert main(["cluster", "list", "-v"]) == 0
    out = capsys.readouterr().out
    assert "Grace: GPU A100, queues gpu (≤ 48 h), long (≤ 168 h), cushion 10%, 1 run speed(s)" in out
    assert "60,000 atoms" in out
