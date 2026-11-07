"""Amber MD stages: per-stage dry trajectories and dry PDBs, 'Production stage' unbundled into its Auto
settings (separate script, trajectory passed on, dry production trajectory), the waiting exit of a step with a
separate script, and the ramp warnings (the feature request 'per-stage control of the dry trajectory').

Scripts are run with stand-in pmemd / cpptraj programs: cpptraj is not installed here, so a dry trajectory's
frames are checked by the cpptraj input the script feeds (its trajin lines), not by counting frames."""

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import threading

import pytest

from mdatk.cli import main
from mdatk.config import PlanNode, Project, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.base import WAITING_STATUS
from mdatk.workflows.plan import Plan, add_node, connect
from mdatk.workflows.runner import ScriptJob, plan_jobs, run_jobs

STAMP = "20261015-1432"
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
[ -s "$crd" ] || { echo "pmemd: no coordinates $crd" >&2; exit 1; }
echo "pmemd $out" >> "$FAKE_LOG"
echo restart > "$rst"
if [ -n "$nc" ]; then echo "trajectory of $out" > "$nc"; fi
printf '|  Total wall time:           1    seconds\\n' > "$out"
"""
FAKE_CPPTRAJ = """#!/bin/bash
# stand-in cpptraj: keeps its input as cpptraj-NNN.in in $FAKE_INPUTS, logs the files it writes (trajout) and
# creates them and the parmout files; every trajin file must exist
n=$(ls "$FAKE_INPUTS" | wc -l)
input="$FAKE_INPUTS/cpptraj-$(printf %03d $((n + 1))).in"
cat > "$input"
echo "cpptraj $(awk '$1 == "trajout" {printf "%s ", $2}' "$input")" | sed 's/ $//' >> "$FAKE_LOG"
while read -r line; do
  set -- $line
  case "$1" in
    trajin) [ -s "$2" ] || { echo "cpptraj: no $2" >&2; exit 1; } ;;
  esac
  while [ $# -gt 0 ]; do
    case "$1" in trajout|parmout) echo "made from $input" > "$2"; shift ;; esac
    shift
  done
done < "$input"
"""


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("pmemd", FAKE_PMEMD), ("cpptraj", FAKE_CPPTRAJ)):
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    log = tmp_path / "programs.log"
    log.write_text("")
    inputs = tmp_path / "cpptraj-inputs"
    inputs.mkdir()
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_INPUTS", str(inputs))
    return log


def calls(log):
    return [ln for ln in log.read_text().splitlines() if ln]


def copy(name):
    """The stand-in cpptraj's log line for a stage's dry copy ``name``: cpptraj writes ``name``.part, which the
    script renames once it is complete."""
    return f"cpptraj {name}.part"


def cpptraj_inputs(tmp_path):
    """The inputs the stand-in cpptraj got, in order: (trajout file, trajin files, the whole input)."""
    out = []
    for name in sorted(os.listdir(tmp_path / "cpptraj-inputs")):
        text = (tmp_path / "cpptraj-inputs" / name).read_text()
        words = [ln.split() for ln in text.splitlines() if ln.strip()]
        out.append((next(w[1] for w in words if w[0] == "trajout"), [w[1] for w in words if w[0] == "trajin"],
                    text))
    return out


def md(name, **kw):
    return dict(dict(name=name, kind="md", nstlim=1000, ntwx=500), **kw)


def anneal(**per_stage):
    """The request's annealing cycle: heat 300 -> 340 K, hold, cool 340 -> 300 K, relax at 300 K; ``per_stage``
    maps a stage name to extra options."""
    stages = [md("heat1", tempi=300.0, temp0=340.0, ramp=True, ramp_steps=1000, dt=0.003),
              md("hold1", tempi=340.0, temp0=340.0, dt=0.003),
              md("cool1", tempi=340.0, temp0=300.0, ramp=True, ramp_steps=1000, dt=0.003),
              md("relax1", temp0=300.0, dt=0.004)]
    for st in stages:
        st.update(per_stage.get(st["name"], {}))
    return stages


def md_step(stages, step_no=1, **settings):
    """An amber-md step on hand-picked inputs (not written); returns (plan, resolved step)."""
    tool = TOOLS["amber-md"]
    s = dict(engine="pmemd", stages=stages, dry_pdb=False, input_velocities=True)
    s.update(settings)
    node = PlanNode(id="md", tool="amber-md", step=step_no, settings=tool.normalise(s),
                    files={"topology": "/data/sys.parm7", "coordinates": "/data/sys.rst7"})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), "/tmp/mdatk-amber-stages")
    return plan, plan.steps["md"]


def scripts(plan):
    files = {f.name: f for f in plan.generate()["md"]}
    return (files[f"apt1_01_amber-md_{STAMP}.sh"].content,
            files[f"apt1_01_amber-prod_{STAMP}.sh"].content if f"apt1_01_amber-prod_{STAMP}.sh" in files else None)


def messages(st, level=None):
    return [i.message for i in st.issues if level is None or i.level == level]


def written(tmp_path, stages, **settings):
    """The amber-md step (number 1) on hand-picked inputs that exist, written; returns (plan, main script,
    separate script, step folder)."""
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
    assert plan.steps["md"].ok, plan.steps["md"].issues
    plan.write()
    folder = tmp_path / "plan" / "01_amber-md"
    return plan, str(folder / f"apt1_01_amber-md_{STAMP}.sh"), str(folder / f"apt1_01_amber-prod_{STAMP}.sh"), folder


def bash(script):
    return subprocess.run(["bash", script], capture_output=True, text=True, timeout=60)


def pmemd_lines(text):
    return [ln for ln in text.splitlines() if ln.strip().startswith(("$PMEMD -O", "pmemd -O"))]


# ---------------------------------------------------------------------------
# the request's acceptance criteria
# ---------------------------------------------------------------------------


@needs_bash
def test_criterion_1_a_dry_label_on_the_last_stage_dries_that_stage_only(tmp_path, fake_bin):
    plan, main_sh, _prod, folder = written(tmp_path, anneal(relax1={"dry": "relax1"}))
    step = plan.steps["md"]
    assert [p for p in step.outputs if p.startswith("dry")] == ["dry_topology", "dry_relax1"]
    assert step.outputs["dry_relax1"].paths == [f"01_amber-md/apt1_01_dry-relax1_{STAMP}.nc"]
    assert step.outputs["dry_relax1"].data["stages"] == ["relax1"]
    res = bash(main_sh)
    assert res.returncode == 0, res.stdout + res.stderr
    assert [c for c in calls(fake_bin) if c.startswith("cpptraj")] == [copy(f"apt1_01_dry-relax1_{STAMP}.nc")]
    # the one dry trajectory holds the frames of relax1 and no others: its cpptraj input reads only them (frame
    # counts would need a real cpptraj, which is not installed here)
    (out, trajins, text), = cpptraj_inputs(tmp_path)
    assert out == f"apt1_01_dry-relax1_{STAMP}.nc.part" and trajins == [f"apt1_01_relax1_{STAMP}.nc"]
    assert "autoimage" in text and f"parmout apt1_01_dry_{STAMP}.parm7" in text
    assert sorted(p.name for p in folder.iterdir() if "_dry" in p.name) == [
        f"apt1_01_dry-relax1_{STAMP}.nc", f"apt1_01_dry_{STAMP}.parm7"]  # renamed: no .part is left


def test_criterion_2_all_stages_separate():
    every = {name: {"separate": "yes"} for name in ("heat1", "hold1", "cool1", "relax1")}
    plan, st = md_step(anneal(**every), separate_production=False)
    main_sh, prod_sh = scripts(plan)
    assert len(pmemd_lines(prod_sh)) == 4 and pmemd_lines(main_sh) == []
    assert [ln.split(" -o ")[1].split()[0] for ln in pmemd_lines(prod_sh)] == [
        f"apt1_01_{n}_{STAMP}.out" for n in ("heat1", "hold1", "cool1", "relax1")]
    assert "WAITING: the separately run stages have not finished" in main_sh and f"exit {WAITING_STATUS}" in main_sh
    # its title says what it runs (as its description and the main script's message do)
    assert prod_sh.splitlines()[1] == "# Amber MD stages run separately"
    # with 'Production stage' on all four (and the step's separate script) the split is the same
    plan2, _st = md_step(anneal(**{n: {"production": True} for n in every}))
    main2, prod2 = scripts(plan2)
    assert pmemd_lines(prod2) == pmemd_lines(prod_sh) and pmemd_lines(main2) == []
    assert prod2.splitlines()[1] == "# Amber production"  # as before


# MDATK 0.0.12's scripts of [eq, prod (2 segments)] with dry_traj and dry_pdb (from 'set -eo pipefail' on): the
# separate production script and the main script, and the main script without a separate one
SCRIPTS_0_0_12 = {(True, "amber-md"): "ffa5afa9f6f879a48fed", (True, "amber-prod"): "7787e602dbd59bf889a1",
                  (False, "amber-md"): "941caf915f040d263301"}


@pytest.mark.parametrize("separate", [True, False])
@pytest.mark.parametrize("explicit", [False, True])
def test_criterion_3_dry_traj_without_stage_options_writes_the_scripts_of_0_0_12(separate, explicit):
    defaults = dict(separate="auto", export="auto", dry="", dry_pdb=False) if explicit else {}
    stages = [md("eq", **defaults), md("prod", production=True, segments=2, **defaults)]
    tool = TOOLS["amber-md"]
    node = PlanNode(id="md", tool="amber-md", step=1, settings=tool.normalise(dict(
        engine="pmemd", stages=stages, dry_traj=True, dry_pdb=True, separate_production=separate)),
        files={"topology": "/data/sys.parm7", "coordinates": "/data/sys.rst7"})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), "/tmp/mdatk-amber-stages")
    got = {}
    for f in plan.generate()["md"]:
        if f.name.endswith(".sh"):
            text = f.content[f.content.index("set -eo pipefail"):]
            got[(separate, f.name.split("_")[2])] = hashlib.sha256(text.encode()).hexdigest()[:20]
    assert got == {k: v for k, v in SCRIPTS_0_0_12.items() if k[0] == separate}


def _anneal_project(tmp_path, anneal_stages, **anneal_settings):
    """build (HMR) -> equilibration at 300 K -> one annealing cycle continuing it; saved as a project."""
    p = Project()
    wf = p.workflow
    wf.project_name, wf.stamp, wf.out_dir = "apt1", STAMP, str(tmp_path / "plan")
    build = add_node(wf, "amber-build", settings={"hmr": True})
    build.files["structure"] = str(tmp_path / "model.pdb")
    eq = add_node(wf, "amber-md", settings={"stages": [
        dict(name="min", kind="min"), md("heat", tempi=0.0, temp0=300.0, ensemble="nvt"),
        md("eq", temp0=300.0, ensemble="npt", ntwx=0)], "dry_pdb": False})
    cyc = add_node(wf, "amber-md", settings=dict(dict(stages=anneal_stages, dry_pdb=False), **anneal_settings))
    for src, out, dst, inp in ((build, "topology", eq, "topology"), (build, "coordinates", eq, "coordinates"),
                               (eq, "topology", cyc, "topology"), (eq, "restart", cyc, "coordinates")):
        connect(wf, src.id, out, dst.id, inp)
    path = str(tmp_path / "anneal.mdatk.toml")
    p.save(path)
    return path, cyc


def test_criterion_4_check_has_no_errors_for_an_annealing_cycle(tmp_path, capsys):
    path, _cyc = _anneal_project(tmp_path, anneal(relax1={"dry": "relax1", "export": "yes"}), separate_production=False)
    code = main(["workflow", "check", "-p", path])
    captured = capsys.readouterr()
    assert code == 0 and json.loads(captured.out)["errors"] == 0, captured.err
    # stages that differ in temperature and time step are fine; the ramps continue from matching temperatures
    # (heat1 from the equilibration at 300 K, cool1 from hold1 at 340 K): no ramp warning, and since every change
    # of temperature is such a ramp, no 'different temperatures' warning either
    assert "ramp" not in captured.err and "tempi is ignored" not in captured.err
    cycle = [ln for ln in captured.err.splitlines() if "03 · Amber · MD stages" in ln]
    assert cycle and not any(ln.startswith(("WARNING", "ERROR")) for ln in cycle), captured.err


def test_criterion_5_a_dry_label_without_trajectory_frames_is_an_error():
    _plan, st = md_step(anneal(relax1={"dry": "relax1", "ntwx": 0}))
    assert any("the dry label 'relax1' needs a trajectory" in m for m in messages(st, "error"))
    assert not st.ok
    # a stage's dry PDB is made from its restart: no frames needed
    _plan, st = md_step(anneal(relax1={"dry_pdb": True, "ntwx": 0}))
    assert st.ok and "dry_pdb_relax1" in st.outputs


# ---------------------------------------------------------------------------
# precedence: 'Production stage' and its Auto settings, the step's options
# ---------------------------------------------------------------------------


def test_separate_follows_production_unless_set():
    eq, prod = md("eq"), md("prod", production=True)
    # Auto: production stages, with the step's 'Separate production script'
    _p, st = md_step([eq, prod])
    assert TOOLS["amber-md"].roles(st.settings).separate == [False, True]
    _p, st = md_step([eq, prod], separate_production=False)
    assert TOOLS["amber-md"].roles(st.settings).separate == [False, False]
    # Yes: also without that option and for a stage that is no production stage; No: in the main script
    _p, st = md_step([dict(eq, separate="yes"), prod], separate_production=False)
    assert TOOLS["amber-md"].roles(st.settings).separate == [True, False]
    plan, st = md_step([eq, dict(prod, separate="no")])
    roles = TOOLS["amber-md"].roles(st.settings)
    assert roles.separate == [False, False] and not roles.split
    main_sh, prod_sh = scripts(plan)
    assert prod_sh is None and len(pmemd_lines(main_sh)) == 2
    assert any("every production stage runs in the main script" in m for m in messages(st, "info"))
    # minimisation stages have no such option (it is greyed out): theirs is ignored
    _p, st = md_step([dict(name="min", kind="min", separate="yes"), eq], separate_production=False)
    assert TOOLS["amber-md"].roles(st.settings).separate == [False, False]


def test_export_follows_production_unless_set():
    eq, prod = md("eq"), md("prod", production=True)
    tool = TOOLS["amber-md"]
    _p, st = md_step([eq, prod])
    assert st.outputs["trajectory"].paths == [f"01_amber-md/apt1_01_prod_{STAMP}.nc"]
    assert st.outputs["trajectory"].data == {"stages": ["prod"]}
    _p, st = md_step([dict(eq, export="yes"), prod])
    assert st.outputs["trajectory"].data["stages"] == ["eq", "prod"]
    # No on the production stage: as when no stage is a production stage, the last one with Auto goes out
    _p, st = md_step([eq, dict(prod, export="no")])
    assert st.outputs["trajectory"].data["stages"] == ["eq"]
    _p, st = md_step([dict(eq, export="no"), dict(prod, export="no")])
    assert "trajectory" not in st.outputs and "trajectory" not in [p.name for p in tool.output_ports(st.settings)]
    # the trajectory goes out from its script's stage, wherever that runs
    _p, st = md_step([eq, dict(prod, separate="no")])
    assert st.outputs["trajectory"].data["stages"] == ["prod"]
    # Yes without frames: warned
    _p, st = md_step([dict(eq, export="yes", ntwx=0), prod])
    assert any("stage eq: its trajectory is to be passed on, but it writes none" in m for m in messages(st, "warning"))


def between_errors(st):
    return [i for i in st.issues if "between" in i.message and i.level == "error"]


def test_a_main_stage_between_separately_run_stages_is_an_error():
    """The separate script runs its stages one after the other, each from the previous one's restart: a stage of
    the main script between two of them would make the later one wait for a restart that the main script writes
    only once the separate script has finished, so neither script could ever finish."""
    stages = [md("eq"), md("p1", separate="yes"), md("cool"), md("p2", separate="yes")]
    _p, st = md_step(stages, separate_production=False)
    err, = between_errors(st)
    assert not st.ok and err.keys == ("stages.2.separate",)
    assert err.message.startswith("stage cool runs in the main script between stages of the separate script: "
                                  "stage p2 would start from a restart that the main script writes only once the "
                                  "separate script has finished")
    assert "set 'Run in the separate script' to Yes for it, or move it" in err.message
    # 'Production stage' on two stages that are not next to each other: the same (0.0.12 wrote such scripts)
    _p, st = md_step([md("eq"), md("p1", production=True), md("cool"), md("hold"), md("p2", production=True)])
    err, = between_errors(st)
    assert err.message.startswith("stages cool, hold run in the main script") and err.keys == (
        "stages.2.separate", "stages.3.separate")
    # a minimisation always runs in the main script: it can only move
    _p, st = md_step([md("eq"), md("p1", separate="yes"), dict(name="min", kind="min"), md("p2", separate="yes")],
                     separate_production=False)
    err, = between_errors(st)
    assert "move it before or after the separately run stages (a minimisation always runs in the main script)" \
        in err.message and err.keys == ("stages.2.kind",)
    # stages before and after the separately run ones are fine
    _p, st = md_step([md("eq"), md("p1", separate="yes"), md("p2", separate="yes"), md("cool")],
                     separate_production=False)
    assert st.ok and between_errors(st) == [], messages(st, "error")


def test_gromacs_a_stage_between_production_stages_is_an_error():
    tool = TOOLS["gromacs"]
    stages = tool.normalise({})["stages"]  # em, nvt, npt, md (production)
    mid = dict(stages[2], name="mid")
    for extra, ok in (([], True), ([mid, dict(stages[3], name="md2")], False)):
        node = PlanNode(id="g", tool="gromacs", step=2, settings=tool.normalise(
            {"convert": "none", "stages": stages + extra}), files={"topology": "/data/sys.top",
                                                                    "coordinates": "/data/sys.gro"})
        st = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), "/tmp/mdatk-amber-stages").steps["g"]
        errors = between_errors(st)
        assert (errors == []) == ok, messages(st, "error")
    assert errors[0].message.startswith("stage mid runs in the main script between production stages: stage md2 "
                                        "would start from a structure") and errors[0].keys == ("stages.4.production",)


def test_dry_labels_and_the_dry_production_trajectory():
    tool = TOOLS["amber-md"]
    a, b = md("a", production=True), md("b", production=True)
    # the step's dry_traj: the production stages without a label of their own
    plan, st = md_step([md("eq"), a, dict(b, dry="b")], dry_traj=True)
    assert tool.roles(st.settings).dry == {"prod": [1], "b": [2]}
    assert st.outputs["dry_trajectory"].data["stages"] == ["a"]
    assert st.outputs["dry_b"].paths == [f"01_amber-md/apt1_01_dry-b_{STAMP}.nc"]
    _main, prod_sh = scripts(plan)
    block = prod_sh[prod_sh.index("# Dry, imaged copy of the production"):]
    assert [ln for ln in block.splitlines() if ln.startswith("trajin")] == [f"trajin apt1_01_a_{STAMP}.nc"]
    # 'prod' joins that file (in stage order), also from a stage that is no production stage
    _p, st = md_step([dict(md("eq"), dry="prod"), a, b], dry_traj=True)
    assert tool.roles(st.settings).dry == {"prod": [0, 1, 2]}
    # ... and makes it without the step option
    _p, st = md_step([dict(md("eq"), dry="prod"), a, b])
    assert tool.roles(st.settings).dry == {"prod": [0]} and "dry_trajectory" in st.outputs
    # without production stages dry_traj has nothing to dry: no dry trajectory output (it used to name one)
    _p, st = md_step([md("eq"), md("eq2")], dry_traj=True)
    assert "dry_trajectory" not in st.outputs and "dry_topology" not in st.outputs
    # labels are shared by name, in stage order; a stage's dry PDB is independent of the step's
    _p, st = md_step(anneal(heat1={"dry": "hot", "dry_pdb": True}, hold1={"dry": "hot"}, relax1={"dry": "relax1"}))
    assert tool.roles(st.settings).dry == {"hot": [0, 1], "relax1": [3]}
    assert st.outputs["dry_hot"].data == {"stages": ["heat1", "hold1"], "label": "hot"}
    assert st.outputs["dry_pdb_heat1"].paths == [f"01_amber-md/apt1_01_dry-heat1_{STAMP}.pdb"]
    assert "dry_pdb" not in st.outputs  # the step's own dry PDB is off here
    assert "cpptraj" in tool.resources(st.settings)


def test_dry_label_checks():
    def errors(**per_stage):
        return messages(md_step(anneal(**per_stage))[1], "error")

    assert any("may hold only letters, digits" in m for m in errors(relax1={"dry": "relax 1"}))
    for reserved in ("pdb", "topology", "trajectory", "pdb_relax1"):
        assert any(f"the dry label '{reserved}' is reserved" in m for m in errors(relax1={"dry": reserved})), reserved
    clash = errors(heat1={"dry": "hot_1"}, relax1={"dry": "hot-1"})
    assert any("would write the same file" in m and "dry-hot-1" in m for m in clash)
    _p, st = md_step([md("a", production=True), md("b", dry="prod_")], dry_traj=True)
    assert any("'prod', 'prod_' would write the same file" in m for m in messages(st, "error"))
    assert errors(relax1={"dry": "relax_1"}) == [] and errors(relax1={"dry": " relax1 "}) == []
    # minimisation stages have no dry label (the option is greyed out): ignored
    _p, st = md_step([dict(name="min", kind="min", dry="x y", ntwx=0), md("eq")])
    assert st.ok and not any(p.startswith("dry_") for p in st.outputs if p != "dry_pdb")
    # a label that reads as true / false (dry = "false" in a project file) is no name
    assert any("the dry label 'False' reads as a true/false value" in m for m in errors(relax1={"dry": "False"}))


def test_a_dry_copy_never_takes_the_name_of_a_stage_s_trajectory():
    """The dry copy of label X is <project>_<step>_dry-X.nc, the trajectory of a stage named dry-X too."""
    def errors(stages, **settings):
        return [m for m in messages(md_step(stages, **settings)[1], "error") if "the trajectory of stage" in m]

    # cpptraj would write over the trajectory it reads ...
    assert errors([md("dry-relax", dry="relax")]) == [
        f"the dry label 'relax' would write apt1_01_dry-relax_{STAMP}.nc, the trajectory of stage dry-relax: "
        "rename one of them"]
    # ... or stage dry-relax over the dry copy made before it
    assert len(errors([md("relax", dry="relax"), md("dry-relax")])) == 1
    # the dry production trajectory (MDATK 0.0.12 wrote this one too)
    assert errors([md("eq"), md("dry-prod", production=True)], dry_traj=True) == [
        f"the dry production trajectory would write apt1_01_dry-prod_{STAMP}.nc, the trajectory of stage dry-prod: "
        "rename one of them"]
    # a segment's trajectory; a stage without trajectory frames writes none
    assert errors([md("dry-hot", segments=2), md("x", dry="hot-001")]) == [
        f"the dry label 'hot-001' would write apt1_01_dry-hot-001_{STAMP}.nc, the trajectory of stage dry-hot "
        "(segment 1): rename one of them"]
    assert errors([md("dry-hot", ntwx=0), md("heat", dry="hot")]) == []
    _p, st = md_step([md("relax", dry="relax"), md("dry-relax")])
    assert {k for i in st.issues if "the trajectory of stage" in i.message for k in i.keys} == {
        "stages.0.dry", "stages.1.name"}


@pytest.mark.parametrize("label", ["pdb", "topology", "trajectory", "pdb_prod", "relax 1", "False"])
def test_a_label_the_check_refuses_is_no_label(label):
    """Its output would take the name of one of the step's own (dry_pdb, dry_topology, ...): the ports, the values
    and the step's own dry outputs stay as without it, while the check reports the label."""
    stages = [md("eq"), md("prod", production=True, dry=label)]
    _p, st = md_step(stages, dry_pdb=True)
    assert not st.ok and [p.name for p in TOOLS["amber-md"].output_ports(st.settings)] == list(st.outputs)
    assert list(st.outputs) == ["topology", "restart", "trajectory", "dry_pdb"]
    dry_pdb = st.outputs["dry_pdb"]
    assert dry_pdb.paths == [f"01_amber-md/apt1_01_dry_{STAMP}.pdb"] and dry_pdb.kind == "pdb"
    # with the step's dry production trajectory the stage goes there, as without a label
    _p, st = md_step(stages, dry_traj=True)
    assert TOOLS["amber-md"].roles(st.settings).dry == {"prod": [1]}


def test_two_stages_of_one_name_give_one_port_per_output():
    _p, st = md_step([md("a", dry_pdb=True), md("a", dry_pdb=True)])
    assert not st.ok and [p.name for p in TOOLS["amber-md"].output_ports(st.settings)] == list(st.outputs)


def test_true_and_false_in_a_project_file_read_as_the_stage_options(tmp_path, capsys):
    """The request proposed boolean stage flags (dry, separate, export), and its plan generators write project
    files: true / false mean a dry label named after the stage (its own file) or none, and Yes / No."""
    tool = TOOLS["amber-md"]
    issues = []
    s = tool.normalise({"stages": [md("heat1", dry=False, separate=True), md("hold1", dry=False, export=False),
                                   "not a stage", md("relax1", dry=True, export=True)]}, issues)
    assert [(i.message, i.keys) for i in issues] == [("expected an object - using the default", ("stages.2",))]
    assert [(x["name"], x["dry"], x["separate"], x["export"]) for x in s["stages"]] == [
        ("heat1", "", "yes", "auto"), ("hold1", "", "auto", "no"), ("relax1", "relax1", "auto", "yes")]
    # through a project file and 'mdatk workflow check'
    (tmp_path / "in").mkdir()
    for name in ("sys.parm7", "sys.rst7"):
        (tmp_path / "in" / name).write_text("x\n")
    p = Project()
    p.workflow.project_name, p.workflow.stamp = "apt1", STAMP
    m = add_node(p.workflow, "amber-md", settings={"engine": "pmemd", "dry_pdb": False, "input_velocities": True,
                                                   "separate_production": False})
    m.settings["stages"] = [md("heat1", dry=False, separate=True), md("relax1", dry=True, separate=True)]
    m.files.update({"topology": str(tmp_path / "in" / "sys.parm7"), "coordinates": str(tmp_path / "in" / "sys.rst7")})
    path = str(tmp_path / "flags.mdatk.toml")
    p.save(path)
    text = open(path).read()
    assert "dry = false" in text and "dry = true" in text and "separate = true" in text
    capsys.readouterr()
    assert main(["workflow", "check", "-p", path]) == 0
    assert "WARNING" not in capsys.readouterr().err
    step = Plan(Project.load(path).workflow, str(tmp_path / "plan")).steps[m.id]
    assert [p for p in step.outputs if p.startswith("dry")] == ["dry_topology", "dry_relax1"]
    assert TOOLS["amber-md"].roles(step.settings).separate == [True, True]


# ---------------------------------------------------------------------------
# ports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stages,settings", [
    ([md("eq"), md("prod", production=True)], {}),
    ([md("eq"), md("prod", production=True)], {"dry_traj": True, "dry_pdb": True}),
    ([md("prod", production=True)], {"dry_pdb": True, "dry_traj": True}),  # no main stage: no dry PDB
    ([md("eq", ntwx=0), md("prod", production=True, ntwx=0)], {"dry_traj": True}),
    (anneal(heat1={"dry": "hot", "dry_pdb": True}, hold1={"dry": "hot"}, relax1={"dry": "relax1"}), {}),
    (anneal(**{n: {"separate": "yes", "dry_pdb": True} for n in ("heat1", "hold1", "cool1", "relax1")}),
     {"dry_pdb": True}),
    ([dict(name="min", kind="min"), md("eq", dry="prod", export="no")], {"dry_pdb": True}),
])
def test_output_ports_agree_with_output_values(stages, settings):
    _plan, st = md_step(stages, **settings)
    assert [p.name for p in TOOLS["amber-md"].output_ports(st.settings)] == list(st.outputs)


def test_a_dry_label_feeds_a_visualisation_step(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    m = add_node(wf, "amber-md", settings={"stages": anneal(relax1={"dry": "relax1"}), "dry_pdb": False})
    m.files.update({"topology": "/data/sys.parm7", "coordinates": "/data/sys.rst7"})
    v = add_node(wf, "vmd")
    ports = {p.name: p for p in TOOLS["amber-md"].output_ports(TOOLS["amber-md"].normalise(m.settings))}
    assert ports["dry_relax1"].label == "Dry trajectory relax1" and ports["dry_relax1"].kind == "nc"
    connect(wf, m.id, "dry_topology", v.id, "topology")
    connect(wf, m.id, "dry_relax1", v.id, "trajectory")
    plan = Plan(wf, str(tmp_path / "plan"))
    vmd = plan.steps[v.id]
    assert vmd.ctx.inps("trajectory") == [f"../01_amber-md/apt1_01_dry-relax1_{STAMP}.nc"]
    assert vmd.ctx.inp("topology") == f"../01_amber-md/apt1_01_dry_{STAMP}.parm7"
    assert not any("does not produce" in i.message or "no longer produces" in i.message for i in vmd.issues)


# ---------------------------------------------------------------------------
# where the dry copies are made
# ---------------------------------------------------------------------------


def test_each_dry_copy_follows_its_last_stage_in_that_stage_s_script():
    stages = [md("eq1", dry="eq"), md("eq2", dry="eq", dry_pdb=True), md("prod", production=True, dry="p"),
              md("cool", dry="cool")]
    plan, st = md_step(stages, dry_traj=True)
    main_sh, prod_sh = scripts(plan)
    # main script: the stages before the separate script, then their dry copies (made when those stages' pmemd
    # runs ran, and when missing or out of date), the wait, then the stages after it with theirs
    order = [main_sh.index(t) for t in (
        "  # eq2: eq2", "    STAGES_RAN=1\n  fi\n  [ -s \"apt1_01_eq2", "dry_out_of_date() {",
        f'if [ -n "${{STAGES_RAN:-}}" ] || dry_out_of_date apt1_01_dry-eq_{STAMP}.nc apt1_01_eq1_{STAMP}.out '
        f"apt1_01_eq2_{STAMP}.out; then",
        f"dry_out_of_date apt1_01_dry-eq2_{STAMP}.pdb apt1_01_eq2_{STAMP}.out; then", "if ! separate_done; then",
        "# cool: cool", f"trajout apt1_01_dry-cool_{STAMP}.nc.part netcdf")]
    assert order == sorted(order)
    # a run marks that it ran right after its pmemd command, not when it is skipped as finished
    assert main_sh.count("STAGES_RAN=1") == 2 and main_sh.count(
        f"-inf apt1_01_eq1_{STAMP}.mdinfo\n    STAGES_RAN=1\n  fi") == 1
    # every stage's dry copy is written as <file>.part and renamed once complete
    for name in (f"apt1_01_dry-eq_{STAMP}.nc", f"apt1_01_dry-eq2_{STAMP}.pdb", f"apt1_01_dry-cool_{STAMP}.nc"):
        assert (f'[ -s "{name}.part" ] || {{ echo "ERROR: cpptraj did not write {name}.part" >&2; exit 1; }}\n'
                f"mv -f {name}.part {name}\n") in main_sh, name
    # the first dry trajectory of each script writes the dry topology
    assert main_sh.count("parmout") == 1 and main_sh.index("parmout") < main_sh.index("dry-cool")
    assert prod_sh.count("parmout") == 1 and f"strip :WAT,HOH,Na+,Cl-,K+,MG parmout apt1_01_dry_{STAMP}.parm7\n" \
        f"trajout apt1_01_dry-p_{STAMP}.nc.part netcdf" in prod_sh
    # the separate script's dry copies belong to it: the main script waits for them too
    start = main_sh.index("separate_done() {")
    done = main_sh[start:main_sh.index("}\n", start)]
    assert f"[ -s apt1_01_dry-p_{STAMP}.nc ]" in done and "dry-eq" not in done and "dry-cool" not in done
    # dry_traj with the production stage labelled: no dry production trajectory at all
    assert "dry-prod" not in main_sh + prod_sh
    # the dry production trajectory keeps its place (the end of its last stage's script) and its parmout
    plan, _st = md_step([md("eq"), md("prod", production=True), md("prod2", production=True, dry="p2")],
                        dry_traj=True)
    _main, prod_sh = scripts(plan)
    assert prod_sh.rstrip().endswith(f'[ -s "apt1_01_dry-prod_{STAMP}.nc" ] || {{ echo "ERROR: cpptraj did not '
                                     f'write apt1_01_dry-prod_{STAMP}.nc" >&2; exit 1; }}')
    assert prod_sh.count("parmout") == 2


@needs_bash
def test_resume_never_keeps_a_dry_copy_of_a_stage_that_ran_again(tmp_path, fake_bin):
    stages = [md("eq", dry="eq", dry_pdb=True), md("s1", separate="yes", dry="s"), md("s2", separate="yes", dry="s")]
    _plan, main_sh, prod_sh, folder = written(tmp_path, stages)
    res = bash(main_sh)
    assert res.returncode == WAITING_STATUS, res.stdout + res.stderr
    eq_copies = [copy(f"apt1_01_dry-eq_{STAMP}.nc"), copy(f"apt1_01_dry-eq_{STAMP}.pdb")]
    assert calls(fake_bin) == [f"pmemd apt1_01_eq_{STAMP}.out"] + eq_copies
    # run again before the separate script: the finished stage is skipped, and so are its dry copies (up to date)
    assert bash(main_sh).returncode == WAITING_STATUS
    assert calls(fake_bin)[3:] == []
    # ... but when it has to run again (its output is gone), they are made again after it
    os.remove(folder / f"apt1_01_eq_{STAMP}.out")
    assert bash(main_sh).returncode == WAITING_STATUS
    assert calls(fake_bin)[3:] == [f"pmemd apt1_01_eq_{STAMP}.out"] + eq_copies
    res = bash(prod_sh)
    assert res.returncode == 0, res.stderr
    assert calls(fake_bin)[6:] == [f"pmemd apt1_01_s1_{STAMP}.out", f"pmemd apt1_01_s2_{STAMP}.out",
                                   copy(f"apt1_01_dry-s_{STAMP}.nc")]
    # s1 has to run again (its output is gone): the dry copy is made again after it, from both stages
    os.remove(folder / f"apt1_01_s1_{STAMP}.out")
    assert bash(prod_sh).returncode == 0
    assert calls(fake_bin)[9:] == [f"pmemd apt1_01_s1_{STAMP}.out", copy(f"apt1_01_dry-s_{STAMP}.nc")]
    *_, (out, trajins, _text) = cpptraj_inputs(tmp_path)
    assert out == f"apt1_01_dry-s_{STAMP}.nc.part"
    assert trajins == [f"apt1_01_s1_{STAMP}.nc", f"apt1_01_s2_{STAMP}.nc"]
    assert (folder / f"apt1_01_dry-s_{STAMP}.nc").read_text() == f"made from {_text_path(tmp_path, -1)}\n"
    # the separate script has finished: the main script's earlier stages and their dry copies are not redone ...
    n = len(calls(fake_bin))
    res = bash(main_sh)
    assert res.returncode == 0 and len(calls(fake_bin)) == n, res.stdout + res.stderr
    # ... unless a dry copy is missing (asked for after the run, or removed): only that one is made
    os.remove(folder / f"apt1_01_dry-eq_{STAMP}.nc")
    assert bash(main_sh).returncode == 0
    assert calls(fake_bin)[n:] == [copy(f"apt1_01_dry-eq_{STAMP}.nc")]
    # ... or older than its stage's output (the stage ran again in a run stopped before its dry copies)
    out = folder / f"apt1_01_eq_{STAMP}.out"
    later = os.path.getmtime(folder / f"apt1_01_dry-eq_{STAMP}.nc") + 10
    os.utime(out, (later, later))
    n = len(calls(fake_bin))
    assert bash(main_sh).returncode == 0
    assert calls(fake_bin)[n:] == eq_copies


def _text_path(tmp_path, k):
    return str(tmp_path / "cpptraj-inputs" / sorted(os.listdir(tmp_path / "cpptraj-inputs"))[k])


@needs_bash
def test_a_waiting_main_script_does_not_redo_the_held_copies_of_skipped_stages(tmp_path, fake_bin):
    """Run again while the separate script has not finished, the stages before it are skipped as finished: their
    dry copies are up to date and not made again, so the step still waits (exit 0 and its .waiting file) when the
    raw trajectory they were made from has been removed since."""
    stages = [md("eq", dry="eq"), md("p", separate="yes")]
    _plan, main_sh, prod_sh, folder = written(tmp_path, stages, separate_production=False, waiting_exit="file")
    flag = main_sh + ".waiting"
    assert bash(main_sh).returncode == 0 and os.path.exists(flag)
    assert calls(fake_bin) == [f"pmemd apt1_01_eq_{STAMP}.out", copy(f"apt1_01_dry-eq_{STAMP}.nc")]
    res = bash(main_sh)
    assert res.returncode == 0 and os.path.exists(flag) and "eq: already finished - skipping" in res.stdout
    assert len(calls(fake_bin)) == 2
    os.remove(folder / f"apt1_01_eq_{STAMP}.nc")  # the raw trajectory, once its dry copy exists
    res = bash(main_sh)
    assert res.returncode == 0 and os.path.exists(flag) and len(calls(fake_bin)) == 2, res.stderr
    assert bash(prod_sh).returncode == 0
    res = bash(main_sh)
    assert res.returncode == 0 and not os.path.exists(flag), res.stderr
    assert calls(fake_bin)[2:] == [f"pmemd apt1_01_p_{STAMP}.out"]


@needs_bash
def test_the_dry_production_trajectory_of_stages_before_the_separate_script_is_held(tmp_path, fake_bin):
    """A production stage kept in the main script before the separate one: its dry production trajectory comes
    with the other dry copies of that part (made when the stage ran, when missing or out of date), not on every
    run of the main script - also not once the step is complete and the raw trajectory is gone."""
    stages = [md("eq", production=True, separate="no"), md("p", separate="yes")]
    plan, main_sh, prod_sh, folder = written(tmp_path, stages, dry_traj=True)
    assert plan.steps["md"].outputs["dry_trajectory"].data == {"stages": ["eq"]}
    text = open(main_sh).read()
    guard = (f'if [ -n "${{STAGES_RAN:-}}" ] || dry_out_of_date apt1_01_dry-prod_{STAMP}.nc apt1_01_eq_{STAMP}.out; '
             "then\n# Dry, imaged copy of the production trajectory (and its topology)\n")
    assert guard in text and text.index(guard) < text.index("if ! separate_done; then")
    assert text.count("cpptraj << EOF") == 1 and f"mv -f apt1_01_dry-prod_{STAMP}.nc.part apt1_01_dry-prod" in text
    assert bash(main_sh).returncode == WAITING_STATUS
    assert calls(fake_bin) == [f"pmemd apt1_01_eq_{STAMP}.out", copy(f"apt1_01_dry-prod_{STAMP}.nc")]
    assert bash(prod_sh).returncode == 0 and bash(main_sh).returncode == 0
    n = len(calls(fake_bin))
    assert bash(main_sh).returncode == 0 and len(calls(fake_bin)) == n  # complete: nothing is made again
    os.remove(folder / f"apt1_01_eq_{STAMP}.nc")
    res = bash(main_sh)
    assert res.returncode == 0 and len(calls(fake_bin)) == n, res.stderr


def cut_off_cpptraj(tmp_path):
    """Replace the stand-in cpptraj by one that, with $CUT_OFF set, writes part of its trajout file and dies (a
    stopped run, a wall-time limit); without it, it does what the stand-in did."""
    b = tmp_path / "bin"
    (b / "cpptraj").rename(b / "cpptraj-whole")
    (b / "cpptraj").write_text('#!/bin/bash\ninput=$(cat)\nif [ -n "$CUT_OFF" ]; then\n'
                               '  out=$(printf "%s\\n" "$input" | awk \'$1 == "trajout" {print $2}\')\n'
                               '  echo "PARTIAL frames" > "$out"\n  exit 1\nfi\n'
                               'printf "%s\\n" "$input" | cpptraj-whole\n')
    (b / "cpptraj").chmod(0o755)


def cut_off(script):
    return subprocess.run(["bash", script], capture_output=True, text=True, timeout=60,
                          env=dict(os.environ, CUT_OFF="1"))


@needs_bash
def test_a_dry_copy_cut_off_while_written_never_counts_as_made(tmp_path, fake_bin):
    """cpptraj writes <file>.part, renamed once complete: a copy cut off is neither kept as up to date (a held copy
    of the main script) nor taken for the end of the separate script (its waiting check, 'Check previous
    runs', the runner, which then runs that script again)."""
    from mdatk.workflows import runstate
    cut_off_cpptraj(tmp_path)
    # a held copy: dry label hot on the stages before the separate script
    stages = [md("heat1", dry="hot"), md("hold1", dry="hot"), md("relax1", production=True)]
    _plan, main_sh, prod_sh, folder = written(tmp_path, stages, waiting_exit="file")
    hot = folder / f"apt1_01_dry-hot_{STAMP}.nc"
    assert cut_off(main_sh).returncode == 1
    assert not hot.exists() and (folder / f"apt1_01_dry-hot_{STAMP}.nc.part").read_text() == "PARTIAL frames\n"
    assert bash(prod_sh).returncode == 0
    n = len(calls(fake_bin))
    res = bash(main_sh)
    assert res.returncode == 0, res.stderr
    assert calls(fake_bin)[n:] == [copy(hot.name)] and hot.read_text().startswith("made from ")
    assert not (folder / f"apt1_01_dry-hot_{STAMP}.nc.part").exists()
    # a copy the separate script makes (relax1's, the analysis pool)
    shutil.rmtree(tmp_path / "plan")
    stages = [md("eq"), md("heat1", separate="yes"), md("relax1", separate="yes", export="yes", dry="relax1")]
    plan, main_sh, prod_sh, folder = written(tmp_path, stages, waiting_exit="file", separate_production=False)
    pool = folder / f"apt1_01_dry-relax1_{STAMP}.nc"
    assert bash(main_sh).returncode == 0 and cut_off(prod_sh).returncode == 1 and not pool.exists()
    res = bash(main_sh)  # the separate script has not finished: still waiting
    assert res.returncode == 0 and os.path.exists(main_sh + ".waiting"), res.stderr
    assert runstate.check_step(plan, "md").state == "partial"
    results, _lines = run(plan_jobs(plan, include_separate=True))
    assert [(r.job.role, r.status) for r in results] == [("run", "ok"), ("run-separate", "ok"), ("run", "ok")]
    assert pool.read_text().startswith("made from ") and not os.path.exists(main_sh + ".waiting")
    assert runstate.check_step(plan, "md").state == "done"


# ---------------------------------------------------------------------------
# the ramp warning
# ---------------------------------------------------------------------------


def ramp_warnings(st):
    return [m for m in messages(st, "warning") if "ramp" in m]


def ramp(name, tempi, temp0, **kw):
    return md(name, tempi=tempi, temp0=temp0, ramp=True, ramp_steps=1000, **kw)


def test_ramp_warning_compares_tempi_with_the_previous_temperature():
    eq300 = md("eq", temp0=300.0)
    # continuing from a stage at the ramp's start temperature: deliberate
    _p, st = md_step([eq300, ramp("heat", 300.0, 340.0)])
    assert ramp_warnings(st) == [] and st.ok
    # ... a cooling one as well
    _p, st = md_step([md("hold", temp0=340.0), ramp("cool", 340.0, 300.0)])
    assert ramp_warnings(st) == []
    # starting elsewhere than the previous stage ends: warned, with both temperatures (the thermostat's target
    # jumps there), a heating ramp or a cooling one
    _p, st = md_step([eq300, ramp("heat", 310.0, 340.0)])
    assert ramp_warnings(st) == ["stage heat: the ramp starts at tempi = 310 K, but stage eq ended at 300 K"]
    _p, st = md_step([md("hot", temp0=360.0), ramp("cool", 340.0, 300.0)])
    assert ramp_warnings(st) == ["stage cool: the ramp starts at tempi = 340 K, but stage hot ended at 360 K"]
    _p, st = md_step([eq300, ramp("cool", 340.0, 300.0)])
    assert ramp_warnings(st) == ["stage cool: the ramp starts at tempi = 340 K, but stage eq ended at 300 K"]
    # the first stage continuing hand-picked coordinates: the temperature is not known, as before; a cooling ramp
    # starts above its target by design
    _p, st = md_step([ramp("heat", 300.0, 340.0)])
    assert ramp_warnings(st) == ["stage heat: a ramp from tempi on a continued run starts below the current "
                                 "temperature"]
    _p, st = md_step([ramp("cool", 340.0, 300.0)])
    assert ramp_warnings(st) == []
    # new velocities: no continued run
    _p, st = md_step([eq300, ramp("heat", 300.0, 340.0, restart="new")])
    assert ramp_warnings(st) == []
    # tempi is the start of a ramp, so it is not 'ignored'; for a plain continued stage it is
    _p, st = md_step([eq300, ramp("heat", 310.0, 340.0), md("x", tempi=200.0, temp0=340.0)])
    infos = [m for m in messages(st, "info") if "tempi is ignored" in m]
    assert infos == ["stage x: tempi is ignored when velocities are continued"]


def test_ramp_warning_knows_the_temperature_of_an_upstream_md_step(tmp_path):
    path, cyc = _anneal_project(tmp_path, anneal())
    plan = Plan(Project.load(path).workflow, str(tmp_path / "plan"))
    step = plan.steps[cyc.id]
    assert step.ctx.data("coordinates") == {"velocities": True, "temp0": 300.0}
    assert ramp_warnings(step) == []
    # the cycle after a cycle continues at relax1's 300 K; a cycle starting hotter is warned
    path, cyc = _anneal_project(tmp_path, anneal(heat1={"tempi": 320.0, "temp0": 340.0}))
    step = Plan(Project.load(path).workflow, str(tmp_path / "plan")).steps[cyc.id]
    assert ramp_warnings(step) == ["stage heat1: the ramp starts at tempi = 320 K, but the input restart ended "
                                   "at 300 K"]


def test_different_temperatures_are_warned_unless_each_change_is_a_ramp_from_the_last():
    def warned(stages):
        return [m for m in messages(md_step(stages)[1], "warning") if "different temperatures" in m]

    # the annealing cycle: heat1 ramps from 300 K (the input), cool1 from hold1's 340 K; the rest keep theirs
    assert warned(anneal()) == []
    assert warned([md("eq", temp0=300.0)] + anneal()) == []
    # a change without a ramp, or a ramp from another temperature: warned, as before
    assert warned([md("eq", temp0=300.0), md("prod", temp0=310.0)]) == [
        "MD stages use different temperatures (eq 300 K, prod 310 K)"]
    assert len(warned([md("eq", temp0=300.0), ramp("heat", 310.0, 340.0)])) == 1
    assert len(warned(anneal(relax1={"temp0": 310.0}))) == 1
    assert len(warned([md("eq", temp0=300.0), dict(name="min", kind="min"), ramp("heat", 0.0, 340.0)])) == 1


# ---------------------------------------------------------------------------
# the waiting exit: exit status 75, or exit 0 and <main script>.waiting
# ---------------------------------------------------------------------------


def run(jobs, **kw):
    lines = []
    lock = threading.Lock()

    def on_line(line):
        with lock:
            lines.append(line)

    return run_jobs(jobs, None, on_line=on_line, **kw), lines


@needs_bash
def test_waiting_file_mode_end_to_end(tmp_path, fake_bin):
    stages = anneal(cool1={"separate": "yes"}, relax1={"separate": "yes", "dry": "relax1"})
    plan, main_sh, prod_sh, folder = written(tmp_path, stages, waiting_exit="file")
    flag = main_sh + ".waiting"
    res = bash(main_sh)
    assert res.returncode == 0 and os.path.exists(flag), res.stderr
    assert "WAITING: the separately run stages have not finished" in res.stderr
    assert f"bash apt1_01_amber-prod_{STAMP}.sh" in open(flag).read()
    assert calls(fake_bin) == [f"pmemd apt1_01_{n}_{STAMP}.out" for n in ("heat1", "hold1")]
    # MDATK's runner: exit 0 with the file present is waiting, like 75, and the steps after it do not start
    job, = plan_jobs(plan)
    after = ScriptJob(2, "after", str(tmp_path / "after.sh"), node_id="after", after=["md"])
    (tmp_path / "after.sh").write_text("touch ran-after\n")
    results, lines = run([job, after])
    assert [(r.job.node_id, r.status) for r in results] == [("md", "waiting"), ("after", "skipped")]
    assert results[0].message == f"waiting for a separately run script: apt1_01_amber-prod_{STAMP}.sh"
    assert os.path.exists(flag) and not (tmp_path / "ran-after").exists()
    # with the separate script asked for, it runs next, then the main script again: complete, it removes the file
    jobs = plan_jobs(plan, include_separate=True)
    assert [(j.role, j.again) for j in jobs] == [("run", False), ("run-separate", False), ("run", True)]
    results, lines = run(jobs)
    assert [r.status for r in results] == ["ok", "ok", "ok"] and [r.returncode for r in results] == [0, 0, 0]
    assert (folder / f"apt1_01_dry-relax1_{STAMP}.nc").exists() and not os.path.exists(flag)
    # the main script again by hand: complete, it exits 0 without the file; the runner says ok
    res = bash(main_sh)
    assert res.returncode == 0 and not os.path.exists(flag), res.stderr
    results, _lines = run(plan_jobs(plan))
    assert [r.status for r in results] == ["ok"]


@needs_bash
@pytest.mark.parametrize("mode", ["status", "file"])
def test_a_run_with_the_separate_scripts_runs_the_main_script_again(tmp_path, fake_bin, mode):
    """RNAdo's 'Also run the scripts marked run separately' / --include-separate: after the separate script, the
    main script runs again - the stages after the separate ones, its end (no .waiting file left) - before the
    steps that read from the step start."""
    (tmp_path / "in").mkdir()
    for name in ("sys.parm7", "sys.rst7"):
        (tmp_path / "in" / name).write_text("x\n")
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = add_node(wf, "amber-md", settings={"engine": "pmemd", "dry_pdb": False, "waiting_exit": mode,
                                           "stages": [md("eq"), md("prod", production=True), md("post")]})
    a.files.update({"topology": str(tmp_path / "in" / "sys.parm7"), "coordinates": str(tmp_path / "in" / "sys.rst7")})
    b = add_node(wf, "amber-md", settings={"engine": "pmemd", "dry_pdb": False,
                                           "stages": [dict(name="min", kind="min")]})
    connect(wf, a.id, "topology", b.id, "topology")
    connect(wf, a.id, "restart", b.id, "coordinates")
    plan = Plan(wf, str(tmp_path / "plan"))
    plan.write()
    main_sh = str(tmp_path / "plan" / "01_amber-md" / f"apt1_01_amber-md_{STAMP}.sh")
    jobs = plan_jobs(plan, include_separate=True)
    assert [(j.node_id, j.role, j.again) for j in jobs] == [
        (a.id, "run", False), (a.id, "run-separate", False), (a.id, "run", True), (b.id, "run", False)]
    results, lines = run(jobs)
    assert [(r.job.node_id, r.status) for r in results] == [(a.id, "ok")] * 3 + [(b.id, "ok")]
    assert calls(fake_bin) == [f"pmemd apt1_01_{n}_{STAMP}.out" for n in ("eq", "prod", "post")] + [
        f"pmemd apt1_02_min_{STAMP}.out"]
    assert not os.path.exists(main_sh + ".waiting")
    log = open(main_sh + ".log").read()  # the output log of its last run: the one that completed the step
    assert "the work before them is not run again" in log and "Step completed successfully" in log
    # complete: neither the separate script nor the main script is run a second time
    results, lines = run(plan_jobs(plan, include_separate=True))
    assert [r.status for r in results] == ["ok"] * 4 and len(calls(fake_bin)) == 4
    assert any("has already finished" in ln for ln in lines)
    assert any("not run again (it found its separately run scripts finished and ran to its end)" in ln
               for ln in lines)


@needs_bash
def test_the_main_script_run_again_decides_whether_its_step_waits(tmp_path):
    sep = tmp_path / "a-prod.sh"
    sep.write_text('cd "$(dirname "$0")"\necho prod >> ran.txt\n')  # does not make what the main script waits for
    (tmp_path / "a.sh").write_text(f'cd "$(dirname "$0")"\necho main >> ran.txt\n'
                                   f'if [ ! -f prod.done ]; then exit {WAITING_STATUS}; fi\n')
    a = ScriptJob(1, "a", str(tmp_path / "a.sh"), node_id="a", separate=[str(sep)])
    s = ScriptJob(1, "a", str(sep), node_id="a", separate=[str(sep)], role="run-separate")
    again = ScriptJob(1, "a", str(tmp_path / "a.sh"), node_id="a", separate=[str(sep)], again=True)
    b = ScriptJob(2, "b", str(tmp_path / "b.sh"), node_id="b", after=["a"])
    (tmp_path / "b.sh").write_text('cd "$(dirname "$0")"\necho b >> ran.txt\n')
    results, _lines = run([a, s, again, b])
    assert [r.status for r in results] == ["ok", "ok", "waiting", "skipped"]
    assert (tmp_path / "ran.txt").read_text() == "main\nprod\nmain\n"
    # a main script written before the waiting check (it cannot tell): not run a second time, as before
    (tmp_path / "old.sh").write_text('cd "$(dirname "$0")"\necho old >> ran.txt\n')
    old = ScriptJob(1, "a", str(tmp_path / "old.sh"), node_id="a", separate=[str(sep)])
    results, lines = run([old, s, dataclasses.replace(old, again=True)])
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    assert (tmp_path / "ran.txt").read_text().endswith("main\nold\nprod\n")
    assert any("written by an older MDATK" in ln for ln in lines)


@needs_bash
def test_the_master_script_stops_after_a_step_that_waits_with_exit_0(tmp_path, fake_bin):
    (tmp_path / "in").mkdir()
    for name in ("sys.parm7", "sys.rst7"):
        (tmp_path / "in" / name).write_text("x\n")
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = add_node(wf, "amber-md", settings={"engine": "pmemd", "dry_pdb": False, "waiting_exit": "file",
                                           "stages": [md("eq"), md("prod", production=True)]})
    a.files.update({"topology": str(tmp_path / "in" / "sys.parm7"), "coordinates": str(tmp_path / "in" / "sys.rst7")})
    b = add_node(wf, "amber-md", settings={"engine": "pmemd", "dry_pdb": False,
                                           "stages": [dict(name="min", kind="min")]})
    connect(wf, a.id, "topology", b.id, "topology")
    connect(wf, a.id, "restart", b.id, "coordinates")
    report = Plan(wf, str(tmp_path / "plan")).write()
    res = bash(report.master)
    assert res.returncode == 0, res.stderr
    assert f"WAITING: 01_amber-md/apt1_01_amber-md_{STAMP}.sh waits for its separately run script" in res.stderr
    assert calls(fake_bin) == [f"pmemd apt1_01_eq_{STAMP}.out"]  # step 02 did not start
    master = open(report.master).read()
    assert master.count(".waiting") == 1  # only step 01 waits that way
    # the status mode (the default) leaves the master script as it was
    wf.nodes[0].settings["waiting_exit"] = "status"
    report = Plan(wf, str(tmp_path / "plan2")).write()
    assert ".waiting" not in open(report.master).read()


@needs_bash
def test_runner_treats_exit_0_with_the_waiting_file_as_waiting(tmp_path):
    sep = tmp_path / "a-prod.sh"
    sep.write_text('cd "$(dirname "$0")"\ntouch prod.done\n')
    body = ('cd "$(dirname "$0")"\nrm -f a.sh.waiting\n'
            'if [ ! -f prod.done ]; then echo waiting > a.sh.waiting; exit 0; fi\nrm -f a.sh.waiting\n')
    (tmp_path / "a.sh").write_text(body)
    a = ScriptJob(1, "a", str(tmp_path / "a.sh"), node_id="a", separate=[str(sep)])
    b_path = tmp_path / "b.sh"
    b_path.write_text('cd "$(dirname "$0")"\ntouch b.ran\n')
    b = ScriptJob(2, "b", str(b_path), node_id="b", after=["a"])
    results, lines = run([a, b])
    assert [(r.job.node_id, r.status) for r in results] == [("a", "waiting"), ("b", "skipped")]
    assert any("not started (step 01 waits for a separately run script)" in ln for ln in lines)
    assert not (tmp_path / "b.ran").exists()
    s = ScriptJob(1, "a", str(sep), node_id="a", separate=[str(sep)], role="run-separate")
    results, _lines = run([a, s, b])
    assert [r.status for r in results] == ["ok", "ok", "ok"] and (tmp_path / "b.ran").exists()
    results, _lines = run([a, b])  # finished: the script removes its file
    assert [r.status for r in results] == ["ok", "ok"] and not (tmp_path / "a.sh.waiting").exists()
    # a stale file beside a script that exits 75 instead (written again in that mode) does not make it wait
    (tmp_path / "c.sh").write_text(f'cd "$(dirname "$0")"\nif [ ! -f prod.done ]; then exit {WAITING_STATUS}; fi\n')
    (tmp_path / "c.sh.waiting").write_text("old\n")
    c = ScriptJob(1, "c", str(tmp_path / "c.sh"), node_id="c", separate=[str(sep)])
    results, _lines = run([c])
    assert [r.status for r in results] == ["ok"]


def test_waiting_exit_option_of_the_steps_with_a_separate_script():
    # amber-md: the main script exits 0 and leaves its file; the default is exit status 75
    for tool in ("amber-md", "gromacs"):
        opt = TOOLS[tool].flat_options()["waiting_exit"]
        assert opt.default == "status" and opt.choice_values == ["status", "file"]
    plan, _st = md_step([md("eq"), md("prod", production=True)], waiting_exit="file")
    main_sh, _prod = scripts(plan)
    flag = f"apt1_01_amber-md_{STAMP}.sh.waiting"
    assert f"exit {WAITING_STATUS}" not in main_sh and main_sh.count(f"rm -f {flag}") == 2
    assert main_sh.rstrip().endswith(f"rm -f {flag}")
    assert f"> {flag}\n  exit 0\nfi" in main_sh
    # without a separate script the option has nothing to do
    plan, _st = md_step([md("eq"), md("prod", production=True)], waiting_exit="file", separate_production=False)
    assert ".waiting" not in scripts(plan)[0]
    # GROMACS: the same handshake
    tool = TOOLS["gromacs"]
    node = PlanNode(id="g", tool="gromacs", step=2,
                    settings=tool.normalise({"convert": "none", "waiting_exit": "file"}),
                    files={"topology": "/data/sys.top", "coordinates": "/data/sys.gro"})
    gplan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), "/tmp/mdatk-amber-stages")
    files = {f.name: f.content for f in gplan.generate()["g"]}
    gmain = files[f"apt1_02_gromacs_{STAMP}.sh"]
    gflag = f"apt1_02_gromacs_{STAMP}.sh.waiting"
    assert f"exit {WAITING_STATUS}" not in gmain and f"> {gflag}\n  exit 0\nfi" in gmain
    assert gmain.rstrip().endswith(f"rm -f {gflag}")
    assert tool.flat_options()["waiting_exit"].requires == {"separate_production": True}


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


@needs_bash
def test_gromacs_waits_with_exit_0_and_its_file(tmp_path, fake_bin):
    (tmp_path / "bin" / "gmx").write_text(FAKE_GMX)
    (tmp_path / "bin" / "gmx").chmod(0o755)
    (tmp_path / "in").mkdir()
    for name in ("sys.top", "sys.gro"):
        (tmp_path / "in" / name).write_text("x\n")
    tool = TOOLS["gromacs"]
    node = PlanNode(id="g", tool="gromacs", step=2,
                    settings=tool.normalise({"convert": "none", "gpu": False, "waiting_exit": "file"}),
                    files={"topology": str(tmp_path / "in" / "sys.top"),
                           "coordinates": str(tmp_path / "in" / "sys.gro")})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), str(tmp_path / "plan"))
    plan.write()
    folder = tmp_path / "plan" / "02_gromacs"
    main_sh, prod_sh = str(folder / f"apt1_02_gromacs_{STAMP}.sh"), str(folder / f"apt1_02_gromacs-prod_{STAMP}.sh")
    res = bash(main_sh)
    assert res.returncode == 0 and os.path.exists(main_sh + ".waiting"), res.stderr
    results, _lines = run(plan_jobs(plan))
    assert [r.status for r in results] == ["waiting"]
    assert bash(prod_sh).returncode == 0
    res = bash(main_sh)
    assert res.returncode == 0 and not os.path.exists(main_sh + ".waiting"), res.stderr
    results, _lines = run(plan_jobs(plan))
    assert [r.status for r in results] == ["ok"]
