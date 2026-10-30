"""0.0.10: earlier runs of planner steps (mdatk.workflows.runstate, PlanNode.adopted, 'workflow status') and
steps whose production runs in a separate script (WAITING_STATUS, 'workflow run --include-separate')."""

import json
import os
import shutil
import subprocess
import threading

import pytest

from mdatk.cli import main
from mdatk.config import PlanNode, Project, WorkflowParams
from mdatk.projectfile import read_log
from mdatk.workflows import TOOLS, runstate
from mdatk.workflows.base import WAITING_STATUS
from mdatk.workflows.plan import Plan, add_node, apply_template, connect
from mdatk.workflows.runner import ScriptJob, plan_jobs, run_jobs

OLD, OLDER, NEW = "20261015-1432", "20261001-0900", "20261101-0815"
needs_bash = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")

FAKE_PMEMD = """#!/bin/bash
# stand-in pmemd: writes the restart, trajectory and an output file with pmemd's closing lines
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift ;;
    -r) rst="$2"; shift ;;
    -x) nc="$2"; shift ;;
    -c) crd="$2"; shift ;;
  esac
  shift
done
[ -s "$crd" ] || { echo "pmemd: no coordinates $crd" >&2; exit 1; }  # like pmemd
echo "pmemd $out" >> "$FAKE_LOG"
echo restart > "$rst"
if [ -n "$nc" ]; then echo trajectory > "$nc"; fi
printf '|  Total wall time:           1    seconds\\n' > "$out"
"""
FAKE_CPPTRAJ = """#!/bin/bash
# stand-in cpptraj: creates the files named after trajout / parmout in its input
echo cpptraj >> "$FAKE_LOG"
while read -r line; do
  set -- $line
  while [ $# -gt 0 ]; do
    case "$1" in trajout|parmout) echo x > "$2"; shift ;; esac
    shift
  done
done
"""
FAKE_GMX = """#!/bin/bash
# stand-in gmx: grompp writes the -o file, mdrun the final -deffnm .gro
echo "gmx $*" >> "$FAKE_LOG"
cmd="$1"
shift
while [ $# -gt 0 ]; do
  case "$1" in
    -o) if [ "$cmd" = grompp ]; then echo x > "$2"; fi; shift ;;
    -deffnm) echo x > "$2.gro"; shift ;;
  esac
  shift
done
"""


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("pmemd", FAKE_PMEMD), ("cpptraj", FAKE_CPPTRAJ), ("gmx", FAKE_GMX)):
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    log = tmp_path / "programs.log"
    log.write_text("")
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    return log


def calls(log):
    return [line for line in log.read_text().splitlines() if line]


def touch(path, text="x\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def amber_plan(tmp_path, stamp=OLD):
    wf = WorkflowParams(project_name="apt1", stamp=stamp)
    build, restraints, md, analysis = apply_template(wf, "amber")
    build.files["structure"] = str(tmp_path / "model.pdb")
    restraints.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    return wf, build, restraints, md, analysis


def fake_run(plan, nid, stamp, finished=True):
    """The output files (and pmemd outputs) of a run of step ``nid`` with ``stamp``."""
    step = plan.steps[nid]
    for path in runstate.expected(plan, nid, stamp):
        touch(plan.node_path(path))
    for path, _pattern in step.tool.finish_marks(step.settings, step.ctx.with_stamp(stamp)):
        touch(plan.node_path(path), "   Total wall time: 1 s\n" if finished else "NSTEP = 500\n")


# ---------------------------------------------------------------------------
# finding earlier runs
# ---------------------------------------------------------------------------


def test_status_with_the_current_and_earlier_stamps(tmp_path):
    wf, build, restraints, md, analysis = amber_plan(tmp_path)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    runs = runstate.check_runs(plan)
    assert set(runs) == {build.id, restraints.id, md.id}  # the analysis box writes nothing
    assert {r.state for r in runs.values()} == {"missing"}
    # the MD step's outputs are only its own files: the topology it passes on belongs to the build
    assert runstate.expected(plan, md.id, OLD) == [f"03_amber-md/apt1_03_prod_{OLD}.rst7",
                                                   f"03_amber-md/apt1_03_prod_{OLD}.nc",
                                                   f"03_amber-md/apt1_03_dry_{OLD}.pdb"]
    fake_run(plan, build.id, OLD)
    fake_run(plan, restraints.id, OLD)
    fake_run(plan, md.id, OLD, finished=False)  # e.g. production still running: no closing lines
    runs = runstate.check_runs(plan)
    assert runs[build.id].state == "done" and runs[build.id].current and runs[build.id].stamp == OLD
    assert runs[build.id].summary().startswith(f"done: run {OLD}, finished 20")
    assert runs[md.id].state == "partial" and f"03_amber-md/apt1_03_prod_{OLD}.out" in runs[md.id].missing
    fake_run(plan, md.id, OLD)
    assert runstate.check_step(plan, md.id).state == "done"
    # a new stamp: nothing under it, the newest complete earlier run is found
    wf.stamp = NEW
    plan = Plan(wf, root)
    assert runstate.run_stamps(plan, md.id) == [OLD]
    found = runstate.check_step(plan, md.id)
    assert found.state == "done" and found.stamp == OLD and not found.current and not found.adopted
    assert "(an earlier stamp)" in found.summary()
    # an even older complete run and a newer partial one: the newest complete one wins
    fake_run(plan, md.id, OLDER)
    fake_run(plan, md.id, "20261020-1000", finished=False)
    assert runstate.run_stamps(plan, md.id) == ["20261020-1000", OLD, OLDER]
    assert runstate.check_step(plan, md.id).stamp == OLD
    # nothing complete at all: the partial run is reported, with what it lacks
    for name in os.listdir(os.path.join(root, "03_amber-md")):
        if OLD in name or OLDER in name:
            os.remove(os.path.join(root, "03_amber-md", name))
    partial = runstate.check_step(plan, md.id)
    assert partial.state == "partial" and partial.stamp == "20261020-1000" and partial.found


def test_files_written_with_the_scripts_are_not_a_run(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=OLD)
    f = add_node(wf, "farfar2", settings={"sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))"})
    plan = Plan(wf, str(tmp_path / "plan"))
    assert {"fasta", "secstruct"} <= set(plan.steps[f.id].outputs)
    plan.write()  # writes the FASTA and secondary-structure files the step passes on, but runs nothing
    paths = runstate.expected(plan, f.id, OLD)
    assert paths and all(p.endswith(".out") for p in paths)  # the silent files rna_denovo writes
    assert runstate.check_step(plan, f.id).state == "missing"


def test_adopting_points_later_steps_at_the_existing_files(tmp_path):
    wf, build, restraints, md, analysis = amber_plan(tmp_path)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    for nid in (build.id, restraints.id):
        fake_run(plan, nid, OLD)
    wf.stamp = NEW  # the next iteration of the project
    plan = Plan(wf, root)
    st = plan.steps[md.id]
    assert st.ctx.inp("topology") == f"../01_amber-build/apt1_01_parmed_{NEW}.parm7"
    assert not st.ctx.existing("topology")  # the new build has not run
    assert not any("topology check" in i.message for i in plan.steps[restraints.id].issues)
    runs = runstate.check_runs(plan)
    assert runstate.adopt(plan, {build.id: runs[build.id]}) == [build.id] and build.adopted == OLD
    plan = Plan(wf, root)
    # the restraints step now reads the existing topology (a stand-in here, which its check rejects)
    assert any("topology check" in i.message for i in plan.steps[restraints.id].issues)
    assert sorted(runstate.adopt(plan, runstate.check_runs(plan))) == [restraints.id]
    plan = Plan(wf, root)
    st = plan.steps[md.id]
    assert st.ctx.inp("topology") == f"../01_amber-build/apt1_01_parmed_{OLD}.parm7"
    assert st.ctx.existing("topology") and st.ctx.existing("restraints")
    script = {f.name: f.content for f in plan.generate([md.id])[md.id]}[f"apt1_03_amber-md_{NEW}.sh"]
    assert f'TOP="../01_amber-build/apt1_01_parmed_{OLD}.parm7"' in script
    assert f"-c ../01_amber-build/apt1_01_tleap_{OLD}.rst7" in script
    assert f"DISANG=../02_amber-restraints/apt1_02_helix_{OLD}.dist" in \
        {f.name: f.content for f in plan.steps[md.id].files}[f"apt1_03_heat_{NEW}.in"]
    assert f"-o apt1_03_min1_{NEW}.out" in script  # its own files carry the new stamp
    assert any(f"uses the outputs of its run of {OLD}" in i.message and i.level == "info"
               for i in plan.steps[build.id].issues)
    # adopted steps are neither written nor run again; the others are
    report = plan.write()
    assert report.kept == {build.id: OLD, restraints.id: OLD} and build.id not in report.skipped
    assert not os.path.exists(os.path.join(root, "01_amber-build", f"apt1_01_amber-build_{NEW}.sh"))
    assert os.path.exists(os.path.join(root, "03_amber-md", f"apt1_03_amber-md_{NEW}.sh"))
    master = open(report.master).read()
    assert f"#   uses the outputs of its run of {OLD} (not run again)" in master
    assert "bash \"01_amber-build" not in master and f'bash "03_amber-md/apt1_03_amber-md_{NEW}.sh"' in master
    assert f"Uses the outputs of its run of `{OLD}`" in open(report.summary).read()
    assert [j.node_id for j in plan_jobs(plan)] == [md.id]
    # a file of the adopted run disappears: the planner says so
    os.remove(os.path.join(root, "01_amber-build", f"apt1_01_parmed_{OLD}.parm7"))
    plan = Plan(wf, root)
    warn = [i.message for i in plan.steps[build.id].issues if i.level == "warning"]
    assert any(f"run of {OLD}, but 01_amber-build/apt1_01_parmed_{OLD}.parm7 is missing" in w for w in warn)
    assert runstate.check_step(plan, build.id).state == "partial"
    assert runstate.check_step(plan, build.id).adopted
    # forgetting: the step is written and run again, under the plan's stamp
    assert runstate.forget(wf, [build.id]) == [build.id] and build.adopted == ""
    plan = Plan(wf, root)
    assert plan.steps[md.id].ctx.inp("topology") == f"../01_amber-build/apt1_01_parmed_{NEW}.parm7"
    assert [j.node_id for j in plan_jobs(plan)] == [build.id, md.id]


def test_without_adoption_nothing_changes_and_the_field_round_trips(tmp_path):
    wf, build, restraints, md, analysis = amber_plan(tmp_path)
    p = Project()
    p.workflow = wf
    path = str(tmp_path / "p.mdatk.toml")
    p.save(path)
    text = open(path).read()
    assert 'adopted = ""' in text and "Time stamp of an earlier run whose output files" in text
    # a project file of an earlier version (no such entry) loads and resolves exactly the same
    old_text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("adopted ="))
    (tmp_path / "old.mdatk.toml").write_text(old_text)
    a = Plan(Project.load(path).workflow, str(tmp_path / "plan"))
    b = Plan(Project.load(str(tmp_path / "old.mdatk.toml")).workflow, str(tmp_path / "plan"))
    for nid in a.order:
        assert a.steps[nid].outputs == b.steps[nid].outputs
        assert [str(i) for i in a.steps[nid].issues] == [str(i) for i in b.steps[nid].issues]
    build.adopted = OLD
    p.save(path)
    assert {n.id: n.adopted for n in Project.load(path).workflow.nodes}[build.id] == OLD


def test_cli_status_adopt_run_and_forget(tmp_path, capsys):
    wf, build, restraints, md, analysis = amber_plan(tmp_path)
    wf.out_dir = str(tmp_path / "plan")
    p = Project()
    p.workflow = wf
    proj = str(tmp_path / "p.mdatk.toml")
    p.save(proj)
    plan = Plan(Project.load(proj).workflow, wf.out_dir)
    fake_run(plan, build.id, OLD)
    fake_run(plan, restraints.id, OLD)
    assert main(["workflow", "status", "-p", proj, "--stamp", NEW]) == 0
    out = capsys.readouterr().out
    assert f"01 · Amber · system build: done: run {OLD} (an earlier stamp)" in out
    assert "03 · Amber · MD stages: missing" in out and "2 step(s) have a complete run" in out
    assert main(["workflow", "status", "--stamp", NEW, "--adopt"]) == 1  # needs the project file
    capsys.readouterr()
    assert main(["workflow", "status", "-p", proj, "--stamp", NEW, "--adopt", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert sorted(data["changed"]) == sorted([build.id, restraints.id])
    assert [(s["step"], s["state"], s["adopted"]) for s in data["steps"]] == [
        (1, "done", True), (2, "done", True), (3, "missing", False)]
    nodes = {n.id: n for n in Project.load(proj).workflow.nodes}
    assert nodes[build.id].adopted == OLD and nodes[md.id].adopted == ""
    # adopted steps are not run again (nothing else is selected here: nothing to do, which is fine)
    assert main(["workflow", "run", "-p", proj, "--stamp", NEW, "--step", "1"]) == 0
    assert f"uses the outputs of its run of {OLD}: not run again" in capsys.readouterr().out
    assert main(["workflow", "status", "-p", proj, "--forget", "--step", "1"]) == 0
    assert "runs again" in capsys.readouterr().out
    nodes = {n.id: n for n in Project.load(proj).workflow.nodes}
    assert nodes[build.id].adopted == "" and nodes[restraints.id].adopted == OLD


# ---------------------------------------------------------------------------
# production in a separate script: the main script waits for it
# ---------------------------------------------------------------------------


def md_plan(tmp_path, stages, **settings):
    """An amber-md step (number 1) on hand-picked inputs, written; returns the plan and the two scripts."""
    for name in ("sys.parm7", "sys.rst7"):
        touch(str(tmp_path / "in" / name))
    tool = TOOLS["amber-md"]
    s = dict(engine="pmemd", stages=stages, dry_pdb=False)
    s.update(settings)
    node = PlanNode(id="md", tool="amber-md", step=1, settings=tool.normalise(s),
                    files={"topology": str(tmp_path / "in" / "sys.parm7"),
                           "coordinates": str(tmp_path / "in" / "sys.rst7")})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=OLD, nodes=[node]), str(tmp_path / "plan"))
    assert plan.steps["md"].ok, plan.steps["md"].issues
    plan.write()
    folder = tmp_path / "plan" / "01_amber-md"
    return plan, str(folder / f"apt1_01_amber-md_{OLD}.sh"), str(folder / f"apt1_01_amber-prod_{OLD}.sh")


def bash(script):
    return subprocess.run(["bash", script], capture_output=True, text=True, timeout=60)


PROD = dict(name="prod", kind="md", production=True, nstlim=1000, ntwx=500)
EQ = dict(name="eq", kind="md", nstlim=1000, ntwx=0)


@needs_bash
def test_all_production_step_waits_for_its_production_script(tmp_path, fake_bin):
    # report hazard 4: the main script of an all-production step used to hold only the preamble and exit 0
    plan, main_sh, prod_sh = md_plan(tmp_path, [PROD])
    res = bash(main_sh)
    assert res.returncode == WAITING_STATUS, res.stdout + res.stderr
    assert "WAITING: the production stages have not finished" in res.stderr
    assert f"  bash {prod_sh}" in res.stderr.splitlines()  # exactly which script to run
    assert calls(fake_bin) == []
    res = bash(prod_sh)
    assert res.returncode == 0, res.stderr
    assert calls(fake_bin) == [f"pmemd apt1_01_prod_{OLD}.out"]
    res = bash(main_sh)
    assert res.returncode == 0 and len(calls(fake_bin)) == 1, res.stdout + res.stderr
    assert runstate.check_step(plan, "md").state == "done"


@needs_bash
def test_stages_after_production_run_once_it_has_finished(tmp_path, fake_bin):
    # review: an early 'production finished: exit 0' skipped the stages that continue from production
    cool = dict(name="cool", kind="md", nstlim=1000, ntwx=0)
    plan, main_sh, prod_sh = md_plan(tmp_path, [EQ, PROD, cool], resume=False)
    res = bash(main_sh)
    assert res.returncode == WAITING_STATUS and calls(fake_bin) == [f"pmemd apt1_01_eq_{OLD}.out"]
    assert bash(prod_sh).returncode == 0
    res = bash(main_sh)  # equilibration is not redone (production started from it); cool runs
    assert res.returncode == 0, res.stdout + res.stderr
    assert calls(fake_bin)[2:] == [f"pmemd apt1_01_cool_{OLD}.out"]
    assert (tmp_path / "plan" / "01_amber-md" / f"apt1_01_cool_{OLD}.rst7").exists()
    assert runstate.check_step(plan, "md").state == "done"


@needs_bash
def test_dry_pdb_asked_for_after_production_is_still_written(tmp_path, fake_bin):
    plan, main_sh, prod_sh = md_plan(tmp_path, [EQ, PROD], dry_pdb=False)
    assert bash(main_sh).returncode == WAITING_STATUS and bash(prod_sh).returncode == 0
    assert bash(main_sh).returncode == 0
    plan, main_sh, prod_sh = md_plan(tmp_path, [EQ, PROD], dry_pdb=True)  # written again, same stamp
    n = len(calls(fake_bin))
    assert bash(main_sh).returncode == 0
    assert calls(fake_bin)[n:] == ["cpptraj"]  # only the dry PDB: equilibration is not redone
    assert (tmp_path / "plan" / "01_amber-md" / f"apt1_01_dry_{OLD}.pdb").exists()


@needs_bash
@pytest.mark.parametrize("resume", [True, False])
def test_mixed_step_runs_its_part_then_waits_and_never_redoes_it(tmp_path, fake_bin, resume):
    plan, main_sh, prod_sh = md_plan(tmp_path, [EQ, dict(PROD, segments=2)], resume=resume, dry_traj=True)
    res = bash(main_sh)
    assert res.returncode == WAITING_STATUS and calls(fake_bin) == [f"pmemd apt1_01_eq_{OLD}.out"]
    res = bash(main_sh)  # again before production: the finished stage is skipped (with 'resume')
    assert res.returncode == WAITING_STATUS
    assert len(calls(fake_bin)) == (1 if resume else 2)
    n = len(calls(fake_bin))
    assert bash(prod_sh).returncode == 0
    assert calls(fake_bin)[n:] == [f"pmemd apt1_01_prod-001_{OLD}.out", f"pmemd apt1_01_prod-002_{OLD}.out",
                                   "cpptraj"]
    n = len(calls(fake_bin))
    res = bash(main_sh)  # production has finished: the step is complete, nothing runs again
    assert res.returncode == 0 and len(calls(fake_bin)) == n
    os.remove(str(tmp_path / "plan" / "01_amber-md" / f"apt1_01_dry-prod_{OLD}.nc"))
    assert bash(main_sh).returncode == WAITING_STATUS  # the separate script did not get to its end


@needs_bash
def test_gromacs_step_waits_for_its_production_script(tmp_path, fake_bin):
    for name in ("sys.top", "sys.gro"):
        touch(str(tmp_path / "in" / name))
    tool = TOOLS["gromacs"]
    node = PlanNode(id="g", tool="gromacs", step=2,
                    settings=tool.normalise({"convert": "none", "gpu": False}),
                    files={"topology": str(tmp_path / "in" / "sys.top"),
                           "coordinates": str(tmp_path / "in" / "sys.gro")})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=OLD, nodes=[node]), str(tmp_path / "plan"))
    plan.write()
    folder = tmp_path / "plan" / "02_gromacs"
    main_sh = str(folder / f"apt1_02_gromacs_{OLD}.sh")
    prod_sh = str(folder / f"apt1_02_gromacs-prod_{OLD}.sh")
    res = bash(main_sh)
    assert res.returncode == WAITING_STATUS and f"  bash {prod_sh}" in res.stderr.splitlines()
    n = len(calls(fake_bin))
    assert n and not any("md_" in c for c in calls(fake_bin))
    assert bash(prod_sh).returncode == 0
    n = len(calls(fake_bin))
    res = bash(main_sh)
    assert res.returncode == 0 and len(calls(fake_bin)) == n  # equilibration is not redone


# ---------------------------------------------------------------------------
# the runner, the CLI
# ---------------------------------------------------------------------------


def script(path, body):
    path.write_text(f'cd "$(dirname "$0")"\n{body}\n')
    return str(path)


def run(jobs, **kw):
    lines, states = [], []
    lock = threading.Lock()

    def on_line(line):
        with lock:
            lines.append(line)

    def on_state(j, s):
        states.append((j.node_id, j.role, s))

    results = run_jobs(jobs, None, on_line=on_line, on_state=on_state, **kw)
    return results, lines, states


@needs_bash
def test_runner_reports_waiting_and_runs_separate_scripts_on_request(tmp_path):
    sep = script(tmp_path / "a-prod.sh", "echo production >> ran.txt\ntouch prod.done")
    main_body = f"echo main >> ran.txt\nif [ -f prod.done ]; then exit 0; fi\nexit {WAITING_STATUS}"
    a = ScriptJob(1, "a", script(tmp_path / "a.sh", main_body), node_id="a", separate=[sep])
    b = ScriptJob(2, "b", script(tmp_path / "b.sh", "echo b >> ran.txt"), node_id="b", after=["a"])
    results, lines, states = run([a, b])  # no exception: waiting is not a failure
    assert [(r.job.node_id, r.status) for r in results] == [("a", "waiting"), ("b", "skipped")]
    assert results[0].message == "waiting for a separately run script: a-prod.sh"
    assert ("a", "run", "waiting") in states
    assert any("not started (step 01 waits for a separately run script)" in ln for ln in lines)
    assert (tmp_path / "ran.txt").read_text() == "main\n"
    # --include-separate: the separate script runs after the main one, then the dependent step
    s = ScriptJob(1, "a", sep, node_id="a", separate=[sep], role="run-separate")
    results, lines, _states = run([a, s, b], parallel=2)
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    assert (tmp_path / "ran.txt").read_text() == "main\nmain\nproduction\nb\n"
    # once it has finished, the main script says so and the separate script is not run again
    results, lines, _states = run([a, s, b])
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    assert any("a-prod.sh has already finished" in ln for ln in lines)
    assert (tmp_path / "ran.txt").read_text().count("production") == 1
    # a main script written before the check existed exits 0 whatever production did: not trusted
    legacy = ScriptJob(1, "a", script(tmp_path / "old.sh", "exit 0"), node_id="a", separate=[sep])
    results, lines, _states = run([legacy, b])
    assert [r.status for r in results] == ["waiting", "skipped"] and "older MDATK" in results[0].message


@needs_bash
def test_cli_run_waits_then_include_separate_runs_everything(tmp_path, fake_bin, capsys):
    plan, _main, _prod = md_plan(tmp_path, [PROD])
    wf = plan.workflow
    wf.out_dir = str(tmp_path / "plan")
    tool = TOOLS["amber-md"]
    after = PlanNode(id="next", tool="amber-md", step=2, settings=tool.normalise(
        dict(engine="pmemd", dry_pdb=False, stages=[dict(name="min", kind="min")])))
    wf.nodes.append(after)
    connect(wf, "md", "topology", "next", "topology")
    connect(wf, "md", "restart", "next", "coordinates")
    p = Project()
    p.workflow = wf
    proj = str(tmp_path / "p.mdatk.toml")
    p.save(proj)
    assert main(["workflow", "write", "-p", proj, "-q"]) == 0
    capsys.readouterr()
    assert main(["workflow", "run", "-p", proj]) == WAITING_STATUS
    out = capsys.readouterr().out
    assert "waiting: 01 · Amber · MD stages is complete once" in out and f"apt1_01_amber-prod_{OLD}.sh" in out
    assert "not started (step 01 waits for a separately run script)" in out
    assert calls(fake_bin) == []
    log = [e for e in read_log(proj) if e["step"] == "01"]
    assert log[-1]["status"] == "waiting"
    assert main(["workflow", "run", "-p", proj, "--include-separate"]) == 0
    assert calls(fake_bin) == [f"pmemd apt1_01_prod_{OLD}.out", f"pmemd apt1_02_min_{OLD}.out"]
    capsys.readouterr()
    assert main(["workflow", "run", "-p", proj, "--include-separate"]) == 0
    assert "has already finished" in capsys.readouterr().out
    assert calls(fake_bin)[2:] == []  # production and the finished minimisation are not run again
    assert main(["workflow", "run", "-p", proj]) == 0  # complete now: the plain run succeeds too
