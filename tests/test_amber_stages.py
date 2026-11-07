"""Amber MD stages: per-stage dry trajectories and dry PDBs, 'Production stage' unbundled into its Auto
settings (separate script, trajectory passed on, dry production trajectory), the waiting exit of a step with a
separate script, and the ramp warnings (the feature request 'per-stage control of the dry trajectory').

Scripts are run with stand-in pmemd / cpptraj programs: cpptraj is not installed here, so a dry trajectory's
frames are checked by the cpptraj input the script feeds (its trajin lines), not by counting frames."""

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
    assert [c for c in calls(fake_bin) if c.startswith("cpptraj")] == [f"cpptraj apt1_01_dry-relax1_{STAMP}.nc"]
    # the one dry trajectory holds the frames of relax1 and no others: its cpptraj input reads only them (frame
    # counts would need a real cpptraj, which is not installed here)
    (out, trajins, text), = cpptraj_inputs(tmp_path)
    assert out == f"apt1_01_dry-relax1_{STAMP}.nc" and trajins == [f"apt1_01_relax1_{STAMP}.nc"]
    assert "autoimage" in text and f"parmout apt1_01_dry_{STAMP}.parm7" in text
    assert sorted(p.name for p in folder.iterdir() if "_dry" in p.name) == [
        f"apt1_01_dry-relax1_{STAMP}.nc", f"apt1_01_dry_{STAMP}.parm7"]


def test_criterion_2_all_stages_separate():
    every = {name: {"separate": "yes"} for name in ("heat1", "hold1", "cool1", "relax1")}
    plan, st = md_step(anneal(**every), separate_production=False)
    main_sh, prod_sh = scripts(plan)
    assert len(pmemd_lines(prod_sh)) == 4 and pmemd_lines(main_sh) == []
    assert [ln.split(" -o ")[1].split()[0] for ln in pmemd_lines(prod_sh)] == [
        f"apt1_01_{n}_{STAMP}.out" for n in ("heat1", "hold1", "cool1", "relax1")]
    assert "WAITING: the separately run stages have not finished" in main_sh and f"exit {WAITING_STATUS}" in main_sh
    # with 'Production stage' on all four (and the step's separate script) the split is the same
    plan2, _st = md_step(anneal(**{n: {"production": True} for n in every}))
    main2, prod2 = scripts(plan2)
    assert pmemd_lines(prod2) == pmemd_lines(prod_sh) and pmemd_lines(main2) == []


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
    # (heat1 from the equilibration at 300 K, cool1 is a cooling ramp): no ramp warning
    assert "ramp" not in captured.err and "tempi is ignored" not in captured.err
    assert "MD stages use different temperatures" in captured.err  # a warning, not an error


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
    # main script: the stages before the separate script, then their dry copies (made whenever those stages ran,
    # and when missing), the wait, then the stages after it with theirs
    order = [main_sh.index(t) for t in (
        "  # eq2: eq2", "STAGES_RAN=1\nfi", "dry_out_of_date() {",
        f'if [ -n "${{STAGES_RAN:-}}" ] || dry_out_of_date apt1_01_dry-eq_{STAMP}.nc apt1_01_eq1_{STAMP}.out '
        f"apt1_01_eq2_{STAMP}.out; then",
        f"dry_out_of_date apt1_01_dry-eq2_{STAMP}.pdb apt1_01_eq2_{STAMP}.out; then", "if ! separate_done; then",
        "# cool: cool", f"trajout apt1_01_dry-cool_{STAMP}.nc netcdf")]
    assert order == sorted(order)
    # the first dry trajectory of each script writes the dry topology
    assert main_sh.count("parmout") == 1 and main_sh.index("parmout") < main_sh.index("dry-cool")
    assert prod_sh.count("parmout") == 1 and f"strip :WAT,HOH,Na+,Cl-,K+,MG parmout apt1_01_dry_{STAMP}.parm7\n" \
        f"trajout apt1_01_dry-p_{STAMP}.nc" in prod_sh
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
    eq_copies = [f"cpptraj apt1_01_dry-eq_{STAMP}.nc", f"cpptraj apt1_01_dry-eq_{STAMP}.pdb"]
    assert calls(fake_bin) == [f"pmemd apt1_01_eq_{STAMP}.out"] + eq_copies
    # run again before the separate script: the finished stage is skipped, its dry copies are made again
    assert bash(main_sh).returncode == WAITING_STATUS
    assert calls(fake_bin)[3:] == eq_copies
    res = bash(prod_sh)
    assert res.returncode == 0, res.stderr
    assert calls(fake_bin)[5:] == [f"pmemd apt1_01_s1_{STAMP}.out", f"pmemd apt1_01_s2_{STAMP}.out",
                                   f"cpptraj apt1_01_dry-s_{STAMP}.nc"]
    # s1 has to run again (its output is gone): the dry copy is made again after it, from both stages
    os.remove(folder / f"apt1_01_s1_{STAMP}.out")
    assert bash(prod_sh).returncode == 0
    assert calls(fake_bin)[8:] == [f"pmemd apt1_01_s1_{STAMP}.out", f"cpptraj apt1_01_dry-s_{STAMP}.nc"]
    *_, (out, trajins, _text) = cpptraj_inputs(tmp_path)
    assert out == f"apt1_01_dry-s_{STAMP}.nc" and trajins == [f"apt1_01_s1_{STAMP}.nc", f"apt1_01_s2_{STAMP}.nc"]
    assert (folder / f"apt1_01_dry-s_{STAMP}.nc").read_text() == f"made from {_text_path(tmp_path, -1)}\n"
    # the separate script has finished: the main script's earlier stages and their dry copies are not redone ...
    n = len(calls(fake_bin))
    res = bash(main_sh)
    assert res.returncode == 0 and len(calls(fake_bin)) == n, res.stdout + res.stderr
    # ... unless a dry copy is missing (asked for after the run, or removed): only that one is made
    os.remove(folder / f"apt1_01_dry-eq_{STAMP}.nc")
    assert bash(main_sh).returncode == 0
    assert calls(fake_bin)[n:] == [f"cpptraj apt1_01_dry-eq_{STAMP}.nc"]
    # ... or older than its stage's output (the stage ran again in a run stopped before its dry copies)
    out = folder / f"apt1_01_eq_{STAMP}.out"
    later = os.path.getmtime(folder / f"apt1_01_dry-eq_{STAMP}.nc") + 10
    os.utime(out, (later, later))
    n = len(calls(fake_bin))
    assert bash(main_sh).returncode == 0
    assert calls(fake_bin)[n:] == eq_copies


def _text_path(tmp_path, k):
    return str(tmp_path / "cpptraj-inputs" / sorted(os.listdir(tmp_path / "cpptraj-inputs"))[k])


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
    # a cooling ramp starts above its target by design
    _p, st = md_step([md("hold", temp0=340.0), ramp("cool", 340.0, 300.0)])
    assert ramp_warnings(st) == []
    _p, st = md_step([md("hot", temp0=360.0), ramp("cool", 340.0, 300.0)])
    assert ramp_warnings(st) == []
    # starting elsewhere than the previous stage ends: warned, with both temperatures
    _p, st = md_step([eq300, ramp("heat", 310.0, 340.0)])
    assert ramp_warnings(st) == ["stage heat: the ramp starts at tempi = 310 K, but stage eq ended at 300 K"]
    # the first stage continuing hand-picked coordinates: the temperature is not known, as before
    _p, st = md_step([ramp("heat", 300.0, 340.0)])
    assert ramp_warnings(st) == ["stage heat: a ramp from tempi on a continued run starts below the current "
                                 "temperature"]
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
    # with the separate script asked for, it runs next and the step completes
    jobs = plan_jobs(plan, include_separate=True)
    results, lines = run(jobs)
    assert [r.status for r in results] == ["ok", "ok"]
    assert (folder / f"apt1_01_dry-relax1_{STAMP}.nc").exists()
    # the main script again: complete, so it removes the file and exits 0; the runner says ok
    res = bash(main_sh)
    assert res.returncode == 0 and not os.path.exists(flag), res.stderr
    results, _lines = run(plan_jobs(plan))
    assert [r.status for r in results] == ["ok"]


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
