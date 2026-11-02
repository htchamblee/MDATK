"""0.0.12: a step that is written again first moves its completed run of the plan's stamp aside, into
<step folder>/superseded/<stamp>_<time>/ (mdatk.workflows.supersede, Plan.write, 'workflow write')."""

import datetime as dt
import json
import os

from mdatk.cli import main
from mdatk.config import Project, WorkflowParams
from mdatk.projectfile import read_log
from mdatk.tomlio import load as load_toml
from mdatk.workflows import runstate, supersede
from mdatk.workflows.plan import Plan, add_node, apply_template, connect
from mdatk.workflows.userscript import DONE_TEXT

OLD, OLDER = "20261015-1432", "20261001-0900"
WHEN = dt.datetime(2026, 11, 2, 10, 15, 30)


def touch(path, text="x\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def fake_run(plan, nid, stamp, finished=True):
    """The output files (and finished logs) of a run of step ``nid`` with ``stamp``."""
    step = plan.steps[nid]
    for path in runstate.expected(plan, nid, stamp):
        touch(plan.node_path(path))
    for path, _pattern in step.tool.finish_marks(step.settings, step.ctx.with_stamp(stamp)):
        touch(plan.node_path(path),
              ("   Total wall time: 1 s\n" + DONE_TEXT + "\n") if finished else "NSTEP = 500\n")


def amber_plan(tmp_path, stamp=OLD, **wf_settings):
    wf = WorkflowParams(project_name="apt1", stamp=stamp, **wf_settings)
    build, restraints, md, analysis = apply_template(wf, "amber")
    build.files["structure"] = str(tmp_path / "model.pdb")
    restraints.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    return wf, build, restraints, md


def tree(folder):
    """Every file below ``folder``, relative to it."""
    out = set()
    for dirpath, _dirs, files in os.walk(folder):
        out |= {os.path.relpath(os.path.join(dirpath, f), folder).replace(os.sep, "/") for f in files}
    return out


def superseded(folder):
    """The run folders in <folder>/superseded/."""
    path = os.path.join(folder, "superseded")
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


# ---------------------------------------------------------------------------
# what moves, and what stays
# ---------------------------------------------------------------------------


def test_a_complete_run_moves_with_its_scripts_and_logs_and_nothing_else(tmp_path):
    wf, build, restraints, md = amber_plan(tmp_path)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    b, r = os.path.join(root, "01_amber-build"), os.path.join(root, "02_amber-restraints")
    fake_run(plan, build.id, OLD)
    touch(os.path.join(b, f"apt1_01_tleap_{OLD}.log"), "tleap log\n")  # a log of the run, by its name
    touch(os.path.join(b, f"apt1_01_parmed_{OLDER}.parm7"))  # another stamp's run
    touch(os.path.join(b, f"apt1_01_amber-build_{OLDER}.sh"))
    touch(os.path.join(b, "notes.txt"))  # the user's own file
    touch(os.path.join(root, "02_amber-restraints", f"apt1_02_helix_{OLD}.log"))  # step 02: partial
    os.makedirs(os.path.join(b, "superseded", "unrelated"))
    before_build, before_r = tree(b), tree(r)
    plan = Plan(wf, root)  # step 02 reads the (stand-in) topology now, and refuses it: not written
    old_script = open(os.path.join(b, f"apt1_01_amber-build_{OLD}.sh")).read()
    assert runstate.run_status(plan, build.id, OLD).state == "done"
    assert runstate.run_status(plan, restraints.id, OLD).state == "missing"
    # what would move: listed before anything is written
    moving = supersede.pending(plan)
    assert list(moving) == [build.id]
    item = moving[build.id]
    assert item.destination() == f"01_amber-build/superseded/{OLD}_<time>"
    assert "01 · Amber · system build: completed run 20261015-1432, finished " in item.summary()
    assert set(item.files) == {f"01_amber-build/{n}" for n in before_build
                               if OLD in n and not n.startswith("superseded/")}
    report = plan.write()
    assert list(report.superseded) == [build.id] and list(report.skipped) == [restraints.id]
    item = report.superseded[build.id]
    assert item.target.startswith(f"01_amber-build/superseded/{OLD}_") and not item.error
    target = os.path.join(root, item.target)
    moved = tree(target)
    assert {f"apt1_01_parmed_{OLD}.parm7", f"apt1_01_tleap_{OLD}.rst7", f"apt1_01_tleap_{OLD}.pdb",
            f"apt1_01_tleap_{OLD}.log", f"apt1_01_amber-build_{OLD}.sh", f"apt1_01_tleap_{OLD}.leap",
            f"apt1_01_parmed_{OLD}.in", "manifest.toml"} <= moved
    assert open(os.path.join(target, f"apt1_01_amber-build_{OLD}.sh")).read() == old_script
    # other stamps, the user's files and other steps stay; the step's new files are written
    left = tree(b)
    assert {f"apt1_01_parmed_{OLDER}.parm7", f"apt1_01_amber-build_{OLDER}.sh", "notes.txt"} <= left
    assert f"apt1_01_amber-build_{OLD}.sh" in left and f"apt1_01_parmed_{OLD}.parm7" not in left
    assert tree(r) == before_r and restraints.id in report.skipped
    assert os.listdir(os.path.join(b, "superseded", "unrelated")) == []
    # the manifest
    man = load_toml(os.path.join(target, "manifest.toml"))
    assert (man["format"], man["step"], man["stamp"], man["from"], man["tool"]) == (
        "mdatk-superseded-run", 1, OLD, "01_amber-build", "amber-build")
    assert man["title"] == "01 · Amber · system build" and "written again" in man["reason"]
    assert sorted(man["files"]) == sorted(moved - {"manifest.toml"}) and man["folders"] == []
    # runstate never looks inside superseded/: the run of OLD is gone from the step folder
    assert runstate.run_status(plan, build.id, OLD).state == "missing"
    assert OLD in runstate.run_stamps(plan, build.id)  # its new scripts
    assert runstate.check_step(plan, build.id).state == "partial"  # OLDER's lone file
    # written again: nothing to move any more
    assert supersede.pending(Plan(wf, root)) == {}
    assert Plan(wf, root).write().superseded == {}


def test_a_partial_run_stays_and_the_option_turns_moving_off(tmp_path):
    wf, build, restraints, md = amber_plan(tmp_path)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, md.id, OLD, finished=False)  # pmemd still running: resuming needs these files
    assert runstate.run_status(plan, md.id, OLD).state == "partial"
    before = tree(os.path.join(root, "03_amber-md"))
    assert plan.write().superseded == {}
    assert tree(os.path.join(root, "03_amber-md")) == before
    fake_run(plan, md.id, OLD)  # finished now
    wf.supersede_runs = False
    plan = Plan(wf, root)
    assert supersede.pending(plan) == {}
    assert plan.write().superseded == {} and superseded(os.path.join(root, "03_amber-md")) == []
    assert plan.write(supersede=True).superseded  # explicitly on


def test_adopted_steps_are_never_written_or_moved(tmp_path):
    wf, build, restraints, md = amber_plan(tmp_path)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, build.id, OLD)
    assert runstate.adopt(plan, runstate.check_runs(plan, [build.id])) == [build.id]
    plan = Plan(wf, root)
    before = tree(os.path.join(root, "01_amber-build"))
    assert supersede.pending(plan) == {}
    report = plan.write()
    assert report.kept == {build.id: OLD} and report.superseded == {}
    assert tree(os.path.join(root, "01_amber-build")) == before


def test_stamp_free_custom_outputs_move_with_their_marker(tmp_path):
    wf = WorkflowParams(project_name="run", stamp=OLD)
    a = add_node(wf, "user-script", settings={
        "script": 'echo hi > "$MDATK_OUT_T"\n', "script_name": "go.sh",
        "outputs": [{"name": "t", "kind": "text", "file": "result.txt"}]})
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    folder = os.path.join(root, "01_custom")
    touch(os.path.join(folder, "result.txt"), "hi\n")
    assert runstate.run_status(plan, a.id, OLD).state == "partial"  # no marker: not a finished run
    assert supersede.pending(plan) == {}
    fake_run(plan, a.id, OLD)
    touch(os.path.join(folder, "result.txt"), "hi\n")
    touch(os.path.join(folder, "other.txt"))  # not the step's
    report = plan.write()
    target = os.path.join(root, report.superseded[a.id].target)
    assert tree(target) == {"result.txt", "go.sh", f"run_01_custom_{OLD}.sh", f"run_01_custom_{OLD}.done",
                            "manifest.toml"}
    assert open(os.path.join(target, "result.txt")).read() == "hi\n"
    assert tree(folder) - {p for p in tree(folder) if p.startswith("superseded/")} == {
        "go.sh", f"run_01_custom_{OLD}.sh", "other.txt"}


def test_a_plan_without_time_stamps_or_step_folders(tmp_path):
    # no stamp: the run's unstamped names move, a stamped file of the step stays
    wf, build, restraints, md = amber_plan(tmp_path, stamp="", use_stamp=False)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    assert plan.stamp == ""
    plan.write()
    fake_run(plan, build.id, "")
    b = os.path.join(root, "01_amber-build")
    touch(os.path.join(b, f"apt1_01_parmed_{OLD}.parm7"))
    touch(os.path.join(b, "apt1_01_tleap.log"))
    report = plan.write([build.id])
    [run] = superseded(b)
    assert run.startswith("no-stamp_")
    assert report.superseded[build.id].target == f"01_amber-build/superseded/{run}"
    assert {"apt1_01_parmed.parm7", "apt1_01_tleap.log", "apt1_01_amber-build.sh"} <= \
        tree(os.path.join(b, "superseded", run))
    assert os.path.exists(os.path.join(b, f"apt1_01_parmed_{OLD}.parm7"))
    # no step folders: every step writes into the plan folder; only step 01's files move, into
    # superseded/01_amber-build/, and the other steps' files and the master script stay
    wf, build, restraints, md = amber_plan(tmp_path, step_folders=False)
    root = str(tmp_path / "flat")
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, build.id, OLD)
    fake_run(plan, restraints.id, OLD)
    touch(os.path.join(root, f"apt1_01_tleap_{OLD}.log"))
    others = {n for n in os.listdir(root) if not n.startswith("apt1_01_")}
    report = plan.write([build.id])
    assert list(report.superseded) == [build.id]
    [run] = os.listdir(os.path.join(root, "superseded", "01_amber-build"))
    moved = tree(os.path.join(root, "superseded", "01_amber-build", run))
    assert f"apt1_01_tleap_{OLD}.log" in moved and f"apt1_01_parmed_{OLD}.parm7" in moved
    assert not any(n.startswith(("apt1_02_", "apt1_03_", "apt1_00_")) for n in moved)
    assert others <= set(os.listdir(root))


def test_files_chosen_by_hand_that_a_step_passes_on_stay(tmp_path):
    # FARFAR2 with the user's own FASTA / secondary structure: its outputs, but not its files
    wf = WorkflowParams(project_name="apt1", stamp=OLD, step_folders=False)
    f = add_node(wf, "farfar2", settings={"make_inputs": False, "fasta_file": "mine.fasta",
                                          "secstruct_file": "mine.secstruct", "runs": 1})
    root = str(tmp_path / "plan")
    for name in ("mine.fasta", "mine.secstruct"):
        touch(os.path.join(root, name))
    plan = Plan(wf, root)
    assert plan.steps[f.id].outputs["fasta"].paths == ["mine.fasta"]
    plan.write()
    fake_run(plan, f.id, OLD)
    report = plan.write()
    assert sorted(os.path.basename(p) for p in report.superseded[f.id].moved) == [
        f"apt1_01_farfar2-r01_{OLD}.out", f"apt1_01_farfar2_{OLD}.flags", f"apt1_01_farfar2_{OLD}.sh"]
    assert os.path.exists(os.path.join(root, "mine.fasta"))
    assert os.path.exists(os.path.join(root, "mine.secstruct"))


def test_farfar2_sub_folders_keep_their_layout(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=OLD)
    f = add_node(wf, "farfar2", settings={"sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))",
                                          "runs": 2})
    m = add_node(wf, "farfar2-models", settings={"extract_all": True, "n_models": 2})
    connect(wf, f.id, "silent", m.id, "silent")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    assert plan.steps[f.id].ok and plan.steps[m.id].ok
    plan.write()
    fake_run(plan, f.id, OLD)
    fake_run(plan, m.id, OLD)
    fd, md_ = os.path.join(root, "01_farfar2"), os.path.join(root, "02_farfar2-models")
    for run in ("r01", "r02"):
        touch(os.path.join(fd, "runs", run, "default.out"))
        touch(os.path.join(fd, f"apt1_01_farfar2-{run}_{OLD}.log"))
        touch(os.path.join(md_, "models", run, "S_000001.pdb"))
        touch(os.path.join(md_, "models", f"extract-{run}.log"))
    report = plan.write()
    assert set(report.superseded) == {f.id, m.id}
    item = report.superseded[m.id]
    assert item.folders == ["02_farfar2-models/models"]
    moved = tree(os.path.join(root, item.target))
    assert {"models/r01/S_000001.pdb", "models/r02/S_000001.pdb", "models/extract-r01.log",
            f"apt1_02_model-01_{OLD}.pdb", f"apt1_02_scores_{OLD}.tsv", f"apt1_02_farfar2-models_{OLD}.sh",
            f"apt1_02_scores_{OLD}.py"} <= moved
    assert not os.path.exists(os.path.join(md_, "models"))
    moved = tree(os.path.join(root, report.superseded[f.id].target))
    assert {"runs/r01/default.out", "runs/r02/default.out", f"apt1_01_farfar2-r01_{OLD}.out",
            f"apt1_01_farfar2-r02_{OLD}.log", f"apt1_01_farfar2_{OLD}.flags", f"apt1_01_farfar2_{OLD}.fasta",
            f"apt1_01_farfar2_{OLD}.sh"} <= moved
    # the step's inputs passed on to later steps are written again
    assert os.path.exists(os.path.join(fd, f"apt1_01_farfar2_{OLD}.secstruct"))
    assert "  models/" in "\n".join(item.lines())
    man = load_toml(os.path.join(root, item.target, "manifest.toml"))
    assert man["folders"] == ["models"]


def test_farfar2_extracted_decoys_move_with_their_run(tmp_path):
    # with 'Extract every decoy' only runs/ moved: the decoy list and table went to superseded/ while the
    # decoys they name stayed in models/, where the next extraction deletes and replaces them
    wf = WorkflowParams(project_name="apt1", stamp=OLD)
    f = add_node(wf, "farfar2", settings={"sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))",
                                          "runs": 2, "extract_decoys": True})
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, f.id, OLD)
    fd = os.path.join(root, "01_farfar2")
    for run in ("r01", "r02"):
        touch(os.path.join(fd, "runs", run, "default.out"))
        touch(os.path.join(fd, "models", run, "S_000001.pdb"), f"{run}\n")
        touch(os.path.join(fd, "models", f"extract-{run}.log"))
    (item,) = supersede.pending(plan).values()
    assert item.folders == ["01_farfar2/runs", "01_farfar2/models"]
    report = plan.write()
    item = report.superseded[f.id]
    moved = tree(os.path.join(root, item.target))
    assert {"models/r01/S_000001.pdb", "models/r02/S_000001.pdb", "models/extract-r01.log",
            f"apt1_01_decoys_{OLD}.txt", f"apt1_01_scores_{OLD}.tsv", f"apt1_01_decoys_{OLD}.sh"} <= moved
    assert open(os.path.join(root, item.target, "models", "r02", "S_000001.pdb")).read() == "r02\n"
    assert not os.path.exists(os.path.join(fd, "models"))
    assert load_toml(os.path.join(root, item.target, "manifest.toml"))["folders"] == ["runs", "models"]


def test_frame_check_moved_decoys_move_with_their_run(tmp_path):
    # the frame check writes the moved decoys of a decoy table into <project>_<NN>_reframed_<stamp>/<run>/;
    # supersede skips folders found by name, so that folder stayed while the manifest and the decoy table
    # pointing into it moved, and the next run overwrote it
    wf = WorkflowParams(project_name="apt1", stamp=OLD)
    f = add_node(wf, "farfar2", settings={"sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))",
                                          "runs": 2})
    fc = add_node(wf, "frame-check", settings={"reframe": True, "fit": "A:1-3", "measure": "B=A:4-6"})
    fc.files = {"reference": str(tmp_path / "ref.pdb"), "new_reference": str(tmp_path / "new.pdb")}
    connect(wf, f.id, "decoy_table", fc.id, "scores")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    assert plan.steps[fc.id].ok, plan.steps[fc.id].issues
    plan.write()
    fake_run(plan, fc.id, OLD)
    reframed = os.path.join(root, "02_frame-check", f"apt1_02_reframed_{OLD}")
    touch(os.path.join(reframed, "r01", "S_000001.pdb"))
    (item,) = supersede.pending(plan).values()
    assert item.folders == [f"02_frame-check/apt1_02_reframed_{OLD}"]
    item = plan.write().superseded[fc.id]
    moved = tree(os.path.join(root, item.target))
    assert {f"apt1_02_reframed_{OLD}/r01/S_000001.pdb", f"apt1_02_reframed-scores_{OLD}.tsv"} <= moved
    assert not os.path.exists(reframed)


def test_repeated_writes_get_folders_of_their_own(tmp_path):
    wf, build, restraints, md = amber_plan(tmp_path)
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    names = []
    for k in range(3):
        fake_run(plan, build.id, OLD)
        touch(os.path.join(root, "01_amber-build", f"apt1_01_tleap_{OLD}.log"), f"run {k}\n")
        item = supersede.move_aside(plan, build.id, WHEN)  # the same second every time
        assert item is not None and not item.error
        names.append(os.path.basename(item.target))
    stem = f"{OLD}_20261102-101530"
    assert names == [stem, f"{stem}-2", f"{stem}-3"]
    for k, name in enumerate(names):
        path = os.path.join(root, "01_amber-build", "superseded", name, f"apt1_01_tleap_{OLD}.log")
        assert open(path).read() == f"run {k}\n"


def custom_chain(tmp_path):
    """Step 01 (a custom script, output t) feeds step 02 (output c in c.txt); a saved project."""
    wf = WorkflowParams(project_name="run", stamp=OLD, out_dir=str(tmp_path / "plan"))
    a = add_node(wf, "user-script", settings={"script": 'echo a > "$MDATK_OUT_T"\n',
                                               "outputs": [{"name": "t", "kind": "text"}]})
    b = add_node(wf, "user-script", settings={"script": 'cp "$MDATK_IN_T" "$MDATK_OUT_C"\n',
                                               "inputs": [{"name": "t", "kind": "text"}],
                                               "outputs": [{"name": "c", "kind": "text", "file": "c.txt"}]})
    connect(wf, a.id, "t", b.id, "t")
    p = Project()
    p.workflow = wf
    proj = str(tmp_path / "p.mdatk.toml")
    p.save(proj)
    return wf, a, b, proj


def test_a_failed_move_puts_everything_back_and_skips_the_step(tmp_path, monkeypatch):
    wf, a, b, proj = custom_chain(tmp_path)
    root = wf.out_dir
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, a.id, OLD)
    fake_run(plan, b.id, OLD)
    one = os.path.join(root, "01_custom")
    before = {n: open(os.path.join(one, n)).read() for n in tree(one)}
    assert len(before) == 4  # output, marker, the script's copy, the run script
    a.settings["script"] = 'echo changed > "$MDATK_OUT_T"\n'  # the step's new script differs
    plan = Plan(wf, root)
    real = os.rename
    calls = []

    def flaky(src, dst):
        calls.append(src)
        if "01_custom" in src and len([c for c in calls if "01_custom" in c]) == 3:
            raise OSError(28, "No space left on device")
        return real(src, dst)

    monkeypatch.setattr(supersede.os, "rename", flaky)
    report = plan.write(project_path=proj)
    item = report.superseded[a.id]
    assert "could not move 01_custom/" in item.error and "No space left on device" in item.error
    assert a.id in report.skipped and "could not be moved aside" in report.skipped[a.id]
    # everything is back, unchanged (the old script too: the step was not written), and nothing is left
    # in superseded/
    assert {n: open(os.path.join(one, n)).read() for n in tree(one)} == before
    assert not os.path.exists(os.path.join(one, "superseded"))
    # the other step was written (its run moved aside first)
    assert b.id in report.superseded and not report.superseded[b.id].error and b.id not in report.skipped
    assert os.path.exists(os.path.join(root, "02_custom", f"run_02_custom_{OLD}.sh"))
    log = [e for e in read_log(proj) if e["step"] == supersede.LOG_STEP]
    assert [(e["status"], e["title"]) for e in log] == [
        ("failed", "Could not move the completed run of 01 · Custom script aside"),
        ("ok", "Moved the completed run of 02 · Custom script aside")]
    assert "could not move 01_custom/" in log[0]["summary"]
    assert log[1]["folder"] == os.path.join(root, report.superseded[b.id].target)
    assert any(f.endswith("/c.txt") for f in log[1]["files"])


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def test_cli_write_dry_run_then_write_and_the_step_log(tmp_path, capsys):
    wf, a, b, proj = custom_chain(tmp_path)
    assert main(["workflow", "write", "-p", proj, "-q"]) == 0
    plan = Plan(Project.load(proj).workflow, wf.out_dir)
    fake_run(plan, a.id, OLD)
    one = os.path.join(wf.out_dir, "01_custom")
    before = tree(one)
    capsys.readouterr()
    # --dry-run: lists what would be written and moved, and touches nothing
    assert main(["workflow", "write", "-p", proj, "--dry-run"]) == 0
    out = capsys.readouterr()
    data = json.loads(out.out)
    assert data["dry_run"] and any(f.endswith(f"run_02_custom_{OLD}.sh") for f in data["write"])
    [sup] = data["superseded"]
    assert (sup["step"], sup["stamp"], sup["to"]) == (1, OLD, f"01_custom/superseded/{OLD}_<time>")
    assert sorted(sup["files"]) == sorted(before)
    assert f"would move 01 · Custom script: completed run {OLD}" in out.err
    assert f"  run_01_t_{OLD}.txt" in out.err
    assert tree(one) == before
    assert not [e for e in read_log(proj) if e["step"] == supersede.LOG_STEP]
    assert main(["workflow", "write", "-p", proj, "--dry-run", "--run"]) != 0
    assert "nothing to --run" in capsys.readouterr().err
    # the write moves it, says so and logs it
    assert main(["workflow", "write", "-p", proj]) == 0
    out = capsys.readouterr()
    data = json.loads(out.out)
    [sup] = data["superseded"]
    assert sup["to"].startswith(f"01_custom/superseded/{OLD}_") and not sup["error"]
    assert f"moved    01 · Custom script: moved its completed run {OLD} aside (4 files)" in out.err
    assert f"run_01_custom_{OLD}.done" in out.err
    [entry] = [e for e in read_log(proj) if e["step"] == supersede.LOG_STEP]
    assert entry["status"] == "ok" and entry["folder"] == os.path.join(wf.out_dir, sup["to"])
    assert "Reason: the step's files were written again" in entry["summary"]
    write = [e for e in read_log(proj) if e["step"] == "workflow write"][-1]
    assert write["summary"].endswith("1 completed run(s) moved aside")
    assert os.path.exists(os.path.join(wf.out_dir, sup["to"], f"run_01_t_{OLD}.txt"))
    assert sorted(tree(one)) == sorted([f"run_01_custom_{OLD}.sh", f"run_01_script_{OLD}.sh"] +
                                       [f"{sup['to'][len('01_custom/'):]}/{n}" for n in before] +
                                       [f"{sup['to'][len('01_custom/'):]}/manifest.toml"])


def test_cli_generate_moves_the_steps_completed_run(tmp_path, capsys):
    args = ["workflow", "generate", "user-script", "-o", str(tmp_path / "gen"), "--step", "4",
            "--project-name", "apt1", "--stamp", OLD, "--set", 'script=echo hi > "$MDATK_OUT_R"',
            "--set", 'outputs=[{"name": "r", "kind": "text", "file": "r.txt"}]', "-q"]
    assert main(args) == 0
    folder = tmp_path / "gen" / "04_custom"
    (folder / "r.txt").write_text("hi\n")
    (folder / f"apt1_04_custom_{OLD}.done").write_text(DONE_TEXT + "\n")
    capsys.readouterr()
    assert main(args + ["--dry-run"]) == 0
    err = capsys.readouterr().err
    assert f"would move 04 · Custom script: completed run {OLD}" in err and "  r.txt" in err
    assert (folder / "r.txt").exists() and not (folder / "superseded").exists()
    assert main(args) == 0
    out = capsys.readouterr()
    [sup] = json.loads(out.out)["superseded"]
    assert sup["to"].startswith(f"04_custom/superseded/{OLD}_") and "r.txt" in sup["files"]
    assert not (folder / "r.txt").exists()
    assert (tmp_path / "gen" / sup["to"] / "r.txt").read_text() == "hi\n"
