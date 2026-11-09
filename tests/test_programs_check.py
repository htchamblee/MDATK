"""A step whose program is missing says which program and why, before its script runs, the same way whether
MDATK was started from a terminal or from the applications menu; the Setup file option (sourced if it
exists); a script that exits 127 ("command not found") is explained too (the feature request 'per-stage
control of the dry trajectory', item 8)."""
import os
import shutil

import pytest

from mdatk.config import PlanNode, WorkflowParams
from mdatk.workflows import TOOLS, runner
from mdatk.workflows.base import env_lines
from mdatk.workflows.plan import Plan
from mdatk.workflows.runner import ScriptFailed, plan_jobs, run_jobs

STAMP = "20261015-1432"
needs_bash = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def bare_path(tmp_path, monkeypatch):
    """PATH with bash and coreutils only (no pmemd or cpptraj), MDATK started from a terminal."""
    b = tmp_path / "bare-bin"
    b.mkdir()
    for prog in ("bash", "echo", "cat", "mkdir", "rm", "mv", "date", "touch", "env", "true"):
        where = shutil.which(prog)
        if where:
            (b / prog).symlink_to(where)
    monkeypatch.setenv("PATH", str(b))
    monkeypatch.setenv("MDATK_SHELL_ENV", "0")
    return b


def md(name, **kw):
    return dict(dict(name=name, kind="md", nstlim=1000, ntwx=500), **kw)


def written(tmp_path, stages, **settings):
    (tmp_path / "in").mkdir(exist_ok=True)
    for name in ("sys.parm7", "sys.rst7"):
        (tmp_path / "in" / name).write_text("x\n")
    tool = TOOLS["amber-md"]
    s = dict(engine="pmemd", stages=stages, dry_pdb=False, input_velocities=True)
    s.update(settings)
    node = PlanNode(id="md", tool="amber-md", step=1, settings=tool.normalise(s),
                    files={"topology": str(tmp_path / "in" / "sys.parm7"),
                           "coordinates": str(tmp_path / "in" / "sys.rst7")})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), str(tmp_path / "plan"))
    plan.write()
    return plan


def test_amber_md_declares_its_engine_and_the_tools_it_calls():
    tool = TOOLS["amber-md"]
    progs = [p for p, _ in tool.programs(tool.normalise(dict(engine="pmemd.cuda")))]
    assert progs[0] == "pmemd.cuda" and "cpptraj" in progs
    mpi = tool.normalise(dict(engine="pmemd.MPI", nproc=4, mpi_launcher="mpirun -np {n}"))
    assert [p for p, _ in tool.programs(mpi)][:2] == ["mpirun", "pmemd.MPI"]
    assert {p for p, _ in TOOLS["amber-build"].programs(TOOLS["amber-build"].normalise({}))} >= {"tleap"}


@needs_bash
def test_a_missing_engine_stops_the_step_before_its_script_runs(tmp_path, bare_path):
    plan = write_two_stage(tmp_path, engine="pmemd.cuda")
    jobs = plan_jobs(plan)
    assert jobs[0].programs and jobs[0].setup == []
    lines = []
    with pytest.raises(ScriptFailed):
        run_jobs(jobs, on_line=lines.append)
    text = "\n".join(lines)
    assert "not started: program not found: pmemd.cuda" in text, text
    assert "the environment it was started from (a terminal)" in text
    assert "Setup file" in text
    # the script never ran: no stage output
    assert not any(n.endswith(".out") for n in os.listdir(tmp_path / "plan" / "01_amber-md"))


def write_two_stage(tmp_path, **settings):
    return written(tmp_path, [md("eq1"), md("prod1")], **settings)


@needs_bash
def test_found_programs_are_reported_and_the_script_runs(tmp_path, bare_path):
    plan = write_two_stage(tmp_path)
    for prog in ("pmemd",):
        p = bare_path / prog
        p.write_text("#!/bin/bash\nexit 3\n")
        p.chmod(0o755)
    lines = []
    with pytest.raises(ScriptFailed):
        run_jobs(plan_jobs(plan), on_line=lines.append)
    text = "\n".join(lines)
    assert "Programs found: pmemd" in text and "not started" not in text


@needs_bash
def test_the_setup_file_puts_the_engine_on_path(tmp_path, bare_path):
    amber = tmp_path / "amber26"
    (amber / "bin").mkdir(parents=True)
    (amber / "bin" / "pmemd.cuda").write_text("#!/bin/bash\nexit 0\n")
    (amber / "bin" / "pmemd.cuda").chmod(0o755)
    setup = amber / "amber.sh"
    setup.write_text(f'export PATH="{amber}/bin:$PATH"\n')
    plan = write_two_stage(tmp_path, engine="pmemd.cuda", env_file=str(setup))
    main = (tmp_path / "plan" / "01_amber-md" / f"apt1_01_amber-md_{STAMP}.sh").read_text()
    assert f"if [ -f {setup} ]; then . {setup}; else" in main
    job = plan_jobs(plan)[0]
    assert job.setup == env_lines(plan.steps["md"].settings)
    needed = runner.called_programs(job.path, job.programs)
    assert "pmemd.cuda" in needed
    missing, _ = runner.missing_programs(job.path, needed, job.setup, dict(os.environ))
    assert "pmemd.cuda" not in missing
    missing, _ = runner.missing_programs(job.path, needed, [], dict(os.environ))
    assert "pmemd.cuda" in missing


def test_no_setup_file_leaves_the_environment_lines_as_they_were():
    assert env_lines({"env_conda": "", "env_setup": "module load amber"}) == ["module load amber"]
    lines = env_lines({"env_file": "/opt/my amber/amber.sh"})
    assert lines == ["if [ -f '/opt/my amber/amber.sh' ]; then . '/opt/my amber/amber.sh'; else echo "
                     "\"WARNING: setup file /opt/my amber/amber.sh not found\" >&2; fi"]


def test_the_planner_warns_about_a_setup_file_that_is_not_here(tmp_path):
    plan = write_two_stage(tmp_path, env_file="/nonexistent/amber.sh")
    issues = [i for i in plan.steps["md"].issues if "env_file" in i.keys]
    assert [i.level for i in issues] == ["warning"] and "does not exist on this computer" in issues[0].message
    plan = write_two_stage(tmp_path, env_file="")
    assert not [i for i in plan.steps["md"].issues if "env_file" in i.keys]


def test_a_main_script_whose_stages_all_run_separately_needs_no_engine(tmp_path):
    stages = [md(n, separate="yes") for n in ("heat1", "hold1", "cool1", "relax1")]
    plan = written(tmp_path, stages, engine="pmemd.cuda", separate_production=False)
    jobs = plan_jobs(plan, include_separate=True)
    main = next(j for j in jobs if j.role == "run")
    sep = next(j for j in jobs if j.role == "run-separate")
    assert "pmemd.cuda" not in runner.called_programs(main.path, main.programs)
    assert "pmemd.cuda" in runner.called_programs(sep.path, sep.programs)


def test_exit_127_names_the_command_that_was_not_found():
    out = "starting\n/x/step.sh: line 12: pmemd.cuda: command not found\n"
    assert runner.not_found(out) == ["pmemd.cuda"]
    terminal = runner.program_message(["pmemd.cuda"], False)
    menu = runner.program_message(["pmemd.cuda"], True)
    assert terminal.startswith("program not found: pmemd.cuda") and menu.startswith("program not found: pmemd.cuda")
    assert "a terminal" in terminal and "~/.bashrc" in menu
    assert "'module' is a shell function" in runner.program_message(["module"], False)


@needs_bash
def test_a_script_that_exits_127_reports_the_program(tmp_path, bare_path):
    from mdatk.workflows.runner import ScriptJob
    path = tmp_path / "step.sh"
    path.write_text("#!/bin/bash\nnot-a-program-xyz --help\n")
    lines = []
    with pytest.raises(ScriptFailed) as err:
        run_jobs([ScriptJob(1, "test", str(path))], on_line=lines.append)
    text = "\n".join(lines) + str(err.value)
    assert "exited with status 127: program not found: not-a-program-xyz" in text, text


def test_namd_mdff_declares_namd_python_and_sander(tmp_path):
    tool = TOOLS["namd-mdff"]
    s = tool.normalise(dict(namd="/opt/namd/namd3 ", energy_check=True))
    progs = dict(tool.programs(s))
    assert set(progs) == {"/opt/namd/namd3", "python3", "sander"}
    script = tmp_path / "s.sh"
    script.write_text('NAMD="/opt/namd/namd3 +p8"\n# sander -O in a comment\npython3 x.py prep\n'
                      "$NAMD e.namd > e.log 2>&1\nsander -O -i e.mdin\n")
    assert runner.called_programs(str(script), tool.programs(s)) == ["/opt/namd/namd3", "python3", "sander"]
    assert "sander" not in dict(tool.programs(tool.normalise(dict(energy_check=False))))
