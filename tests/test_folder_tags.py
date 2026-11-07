"""Steps carry their identity into their folder names (PlanNode.folder_tag) and every write records what the
folders are (MANIFEST, MDATK_step.toml: mdatk.workflows.manifest), so a plan whose steps were renumbered can
still be matched to the tree it wrote ('mdatk workflow manifest').  Steps that write no files have no
folder.  Without folder tags, names, files and project files stay as they were."""

import glob
import json
import os

from mdatk import tomlio
from mdatk.cli import main
from mdatk.config import PlanNode, Project, WorkflowParams, _to_plain
from mdatk.workflows import TOOLS, runstate, supersede
from mdatk.workflows.base import Tool
from mdatk.workflows.manifest import (COLUMNS, IDENTITY, MANIFEST, manifest_text, match, read_identity,
                                      scan_tree)
from mdatk.workflows.naming import folder_tag_problem, parse_folder, parse_name, step_folder
from mdatk.workflows.plan import (Plan, add_node, apply_template, connect, generate_step, remove_node,
                                  single_step_plan, step_title)
from mdatk.workflows.userscript import DONE_TEXT

STAMP = "20261015-1432"
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")
NODE_KEYS = {"id", "tool", "step", "title", "x", "y", "settings", "files", "adopted"}  # as saved before


def branch(wf, tag="", title=""):
    """The Amber template (build -> restraints -> MD -> analysis) as one branch of a plan; its build,
    restraints and MD steps get ``tag`` as their folder tag and titles naming ``title``."""
    build, restraints, md, analysis = apply_template(wf, "amber")
    build.files["structure"] = f"/data/{title or 'model'}.pdb"
    restraints.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    for node, what in ((build, "build"), (restraints, "restraints"), (md, "MD")):
        node.folder_tag = tag
        node.title = f"{what} {title}" if title else ""
    return build, restraints, md, analysis


def folders(plan):
    return {nid: s.ctx.folder for nid, s in plan.steps.items() if s.tool.generates}


def errors(plan):
    return [(nid, i.message) for nid, i in plan.all_issues() if i.level == "error"]


def touch(path, text="x\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def fake_run(plan, nid):
    """The output files and finished logs of a completed run of step ``nid`` (the plan's stamp)."""
    step = plan.steps[nid]
    for path in runstate.expected(plan, nid, plan.stamp):
        touch(plan.node_path(path))
    for path, _pattern in step.tool.finish_marks(step.settings, step.ctx):
        touch(plan.node_path(path), "   Total wall time: 1 s\n" + DONE_TEXT + "\n")
    assert runstate.run_status(plan, nid, plan.stamp).state == "done"


def tree(folder):
    out = set()
    for dirpath, _dirs, files in os.walk(folder):
        out |= {os.path.relpath(os.path.join(dirpath, f), folder).replace(os.sep, "/") for f in files}
    return out


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def test_folder_names_with_and_without_a_tag():
    assert step_folder(3, "amber-build") == "03_amber-build"  # as before
    assert step_folder(3, "amber-build", "") == step_folder(3, "amber-build", "  ") == "03_amber-build"
    assert step_folder(2, "amber-build", "model12") == "02_amber-build_model12"
    assert parse_folder("02_amber-build_model12") == {"step": "02", "tag": "amber-build",
                                                      "folder_tag": "model12"}
    assert parse_folder("03_amber-build") == {"step": "03", "tag": "amber-build", "folder_tag": ""}
    assert parse_folder("superseded") is None and parse_folder("02_a_b_c") is None
    assert folder_tag_problem("") == folder_tag_problem("model-12") == folder_tag_problem("M12") == ""
    rule = "use ASCII letters and digits, with '-' only between them"
    assert folder_tag_problem("model 12") == f"folder tag 'model 12': {rule} (e.g. 'model-12')"
    assert folder_tag_problem("model_12").endswith("(e.g. 'model-12')")
    assert folder_tag_problem("-x") == f"folder tag '-x': {rule} (e.g. 'x')"
    assert folder_tag_problem("__") == f"folder tag '__': {rule}"
    # the text fits every tag it refuses: one that only ends in '-', one with a letter that is not ASCII
    assert folder_tag_problem("a-") == f"folder tag 'a-': {rule} (e.g. 'a')"
    assert folder_tag_problem("modèle12") == f"folder tag 'modèle12': {rule} (e.g. 'mod-le12')"
    # file names keep their four parts, so every reader of them works as before
    assert parse_name("apt1_02_tleap_20261015-1432.parm7")["tag"] == "tleap"


def test_two_tagged_branches_get_folders_of_their_own_and_a_third_changes_neither(tmp_path):
    # acceptance criterion 6: two branches' folders differ, and adding a branch renames neither
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = branch(wf, "model12", "model 12")
    b = branch(wf, "model13", "model 13")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    assert errors(plan) == []
    before = folders(plan)
    assert [before[n.id] for n in a[:3]] == ["01_amber-build_model12", "02_amber-restraints_model12",
                                            "03_amber-md_model12"]
    assert [before[n.id] for n in b[:3]] == ["04_amber-build_model13", "05_amber-restraints_model13",
                                            "06_amber-md_model13"]
    assert len(set(before.values())) == len(before)
    plan.write()
    written = sorted(os.listdir(root))
    # a third branch from the template, and a step added by hand: numbered after the existing steps
    c = branch(wf, "model14", "model 14")
    extra = add_node(wf, "amber-md")
    for port in ("topology", "coordinates"):
        connect(wf, c[0].id, port, extra.id, port)
    connect(wf, c[1].id, "restraints", extra.id, "restraints")
    plan = Plan(wf, root)
    assert errors(plan) == []
    assert {nid: plan.steps[nid].ctx.folder for nid in before} == before
    assert [n.step for n in a[:3] + b[:3]] == [1, 2, 3, 4, 5, 6]
    assert plan.steps[c[0].id].ctx.folder == "07_amber-build_model14"
    assert plan.steps[extra.id].ctx.folder == "10_amber-md"  # untagged: today's name
    plan.write()
    assert set(written) <= set(os.listdir(root))
    # one step number for both branches: the tags keep the folders apart (with step folders)
    b[0].step = a[0].step
    plan = Plan(wf, root)
    assert errors(plan) == []
    assert plan.steps[b[0].id].ctx.folder == "01_amber-build_model13"
    assert any("step number 01 is used by" in i.message for i in plan.issues)


def test_untagged_plans_keep_their_names_and_steps_that_write_nothing_have_no_folder(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    model = tmp_path / "model.pdb"
    model.write_text("END\n")
    src = add_node(wf, "input-file", settings={"path": str(model), "kind": "pdb"})
    note = add_node(wf, "note", settings={"text": "a note"})
    build, restraints, md, analysis = apply_template(wf, "amber")
    connect(wf, src.id, "file", build.id, "structure")
    restraints.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    assert errors(plan) == []
    assert folders(plan) == {build.id: "01_amber-build", restraints.id: "02_amber-restraints",
                             md.id: "03_amber-md"}
    assert plan.steps[src.id].ctx.folder == plan.steps[analysis.id].ctx.folder == ""  # no folder
    assert plan.steps[note.id].ctx is None or plan.steps[note.id].ctx.folder == ""
    report = plan.write()
    # the tree is what it was, plus the records of what each folder is (never among the written files)
    plan_files = [f"apt1_00_plan_{STAMP}.md", f"apt1_00_plan_{STAMP}.sh"]
    assert sorted(os.listdir(root)) == sorted(["01_amber-build", "02_amber-restraints", "03_amber-md",
                                               MANIFEST] + plan_files)
    records = {report.manifest} | set(report.identity.values())
    assert not records & set(report.written)
    assert {os.path.join(root, p) for p in tree(root)} == set(report.written) | records
    assert set(report.identity) == {build.id, restraints.id, md.id}
    assert "Folder: `01_amber-build`" in open(report.summary).read()
    # MANIFEST: the existing file has no folder, the note is left out; each folder holds its step's files
    lines = open(report.manifest).read().splitlines()
    assert lines[0].split("\t") == list(COLUMNS) == ["step", "folder", "node", "tool", "title", "folder_tag",
                                                     "inputs", "holds"]
    rows = {line.split("\t")[2]: line.split("\t") for line in lines[1:]}
    assert note.id not in rows and len(rows) == 5
    assert rows[src.id] == ["-", "-", src.id, "input-file", "Existing file", "-", f"file={model}", "-"]
    assert rows[build.id] == ["01", "01_amber-build", build.id, "amber-build", "Amber · system build", "-",
                              f"structure={model}", "this step"]
    assert rows[md.id][:2] == ["03", "03_amber-md"] and rows[analysis.id][:2] == ["-", "-"]
    assert rows[md.id][7] == "this step" and rows[analysis.id][7] == "-"
    assert rows[md.id][6] == (f"topology=01_amber-build/apt1_01_parmed_{STAMP}.parm7; "
                              f"coordinates=01_amber-build/apt1_01_tleap_{STAMP}.rst7; "
                              f"restraints=02_amber-restraints/apt1_02_helix_{STAMP}.dist")


def test_a_step_whose_generator_writes_nothing_leaves_no_folder(tmp_path, monkeypatch):
    class Nothing(Tool):
        name, label, tag = "nothing", "Writes nothing", "nothing"

        def generate(self, s, ctx):
            return []

    monkeypatch.setitem(TOOLS, "nothing", Nothing())
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    quiet = add_node(wf, "nothing")
    quiet.folder_tag = "x"
    custom = add_node(wf, "user-script", settings={"script": "true\n"})
    root = str(tmp_path / "plan")
    report = Plan(wf, root).write()
    assert not os.path.exists(os.path.join(root, "01_nothing_x"))
    assert set(report.identity) == {custom.id} and os.path.isdir(os.path.join(root, "02_custom"))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def test_folder_tag_checks():
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = add_node(wf, "user-script", step=2, settings={"script": "true\n"})
    a.folder_tag = "model 12"
    issues = [i for i in Plan(wf).steps[a.id].issues if i.keys == ("folder_tag",)]
    assert [(i.level, i.message) for i in issues] == [
        ("error", "folder tag 'model 12': use ASCII letters and digits, with '-' only between them (e.g. "
                  "'model-12'); it is added to the step's folder name")]
    a.folder_tag = " model-12 "  # spaces around it are not part of it
    assert errors(Plan(wf)) == [] and Plan(wf).steps[a.id].ctx.folder == "02_custom_model-12"
    # two steps, one folder (the step titles name the folder tags)
    b = add_node(wf, "user-script", step=2, settings={"script": "true\n"})
    b.folder_tag = "model-12"
    pair = "02 · Custom script · model-12 and 02 · Custom script · model-12 would write into"
    assert errors(Plan(wf)) == [(b.id, f"{pair} the same folder 02_custom_model-12/ with the same names: "
                                       "give one of them another folder tag or step number")]
    b.folder_tag = "Model-12"  # one folder where case does not count
    pair = "02 · Custom script · Model-12 and 02 · Custom script · model-12 would write into"
    assert errors(Plan(wf)) == [(b.id, f"{pair} 02_custom_Model-12/ and 02_custom_model-12/, one folder on a "
                                       "file system that ignores case (macOS, Windows): give one of them "
                                       "another folder tag")]
    b.folder_tag = "model-13"
    assert errors(Plan(wf)) == []
    b.folder_tag = ""
    assert errors(Plan(wf)) == []  # 02_custom and 02_custom_model-12
    # untagged steps: the message of before
    a.folder_tag = ""
    pair = "02 · Custom script and 02 · Custom script would write into"
    before = f"{pair} the same folder with the same names: give one of them another step number"
    assert errors(Plan(wf)) == [(b.id, before)]
    # without step folders a tag names no step folder: the steps still share the plan folder and the file
    # names; it names the folder their completed runs move into, and the step titles show it
    wf.step_folders = False
    a.folder_tag, b.folder_tag = "model-12", "model-13"
    plan = Plan(wf)
    pair = "02 · Custom script · model-13 and 02 · Custom script · model-12 would write into"
    assert errors(plan) == [(b.id, f"{pair} the same folder with the same names: give one of them another step "
                                   "number")]
    assert [i.message for i in plan.steps[a.id].issues if "folder tag" in i.message] == [
        "folder tag 'model-12': the plan writes no step folders, so the tag names only the folder its "
        "completed runs move into (superseded/02_custom_model-12/), and it is in the step's title and "
        "MANIFEST line"]
    assert [i.level for i in plan.steps[a.id].issues if "folder tag" in i.message] == ["info"]
    # a step that writes no files has no folder for a tag
    wf.step_folders = True
    f = add_node(wf, "input-file", settings={"path": "/x.pdb"})
    f.folder_tag = "model12"
    assert [(i.level, i.message) for i in Plan(wf).steps[f.id].issues if "folder tag" in i.message] == [
        ("warning", "folder tag 'model12': this step writes no files, so it has no folder and the tag is not "
                    "used")]


# ---------------------------------------------------------------------------
# the planner's own searches find tagged folders
# ---------------------------------------------------------------------------


def test_previous_and_superseded_runs_are_found_in_tagged_folders(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    build, restraints, md, _analysis = branch(wf, "model12")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, md.id)
    folder = os.path.join(root, "03_amber-md_model12")
    assert runstate.check_runs(plan)[md.id].state == "done"
    assert STAMP in runstate.run_stamps(plan, md.id)
    moving = supersede.pending(plan)
    assert list(moving) == [md.id] and moving[md.id].folder == "03_amber-md_model12"
    assert moving[md.id].destination() == f"03_amber-md_model12/superseded/{STAMP}_<time>"
    assert moving[md.id].held == "" and "held" not in moving[md.id].plain()  # the step's own run
    report = plan.write()
    assert report.superseded[md.id].target.startswith(f"03_amber-md_model12/superseded/{STAMP}_")
    assert os.path.isfile(os.path.join(folder, IDENTITY))  # stays with the folder, not with the run
    assert not glob.glob(os.path.join(root, report.superseded[md.id].target, IDENTITY))
    # without step folders the runs of a tagged step move into superseded/<NN>_<tool>_<tag>/
    flat = WorkflowParams(project_name="apt1", stamp=STAMP, step_folders=False)
    fb, _fr, fmd, _fa = branch(flat, "model12")
    froot = str(tmp_path / "flat")
    fplan = Plan(flat, froot)
    fplan.write()
    fake_run(fplan, fmd.id)
    assert supersede.candidate(fplan, fmd.id).store == "superseded/03_amber-md_model12"


# ---------------------------------------------------------------------------
# MANIFEST and the identity files
# ---------------------------------------------------------------------------


def test_manifest_and_identity_files_are_rewritten_on_every_write(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    build, restraints, md, analysis = branch(wf, "model12", "model 12")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    report = plan.write()
    assert report.manifest == os.path.join(root, MANIFEST)
    ident = read_identity(os.path.join(root, "03_amber-md_model12"))
    assert {k: ident[k] for k in ("format", "project", "step", "folder", "node", "tool", "title", "label",
                                  "folder_tag", "stamp")} == {
        "format": "mdatk-step-folder", "project": "apt1", "step": 3, "folder": "03_amber-md_model12",
        "node": md.id, "tool": "amber-md", "title": "MD model 12", "label": "03 · MD model 12 · model12",
        "folder_tag": "model12", "stamp": STAMP}
    text = open(os.path.join(root, "03_amber-md_model12", IDENTITY)).read()
    assert text.startswith("# This folder holds the files of step 03 · MD model 12 · model12 of plan apt1, "
                           "written by")
    assert "# The step's id in the plan (n1, n2 ...). RNAdo never gives a deleted step's id to a new step" in text
    assert "\n# steps that agree on all of them.\nnode = " in text
    assert "by type,\n# folder tag and title (the node id only confirms them)." in text
    # a write of one step: MANIFEST lists the plan as it is now; only that step's identity file is rewritten
    restraints.title = "helices of model 12"
    md.title = "production model 12"
    before = open(os.path.join(root, "03_amber-md_model12", IDENTITY)).read()
    report = Plan(wf, root).write([restraints.id])
    assert set(report.identity) == {restraints.id} and report.written and report.master == ""
    assert "helices of model 12" in open(report.identity[restraints.id]).read()
    assert open(os.path.join(root, "03_amber-md_model12", IDENTITY)).read() == before
    assert "\tproduction model 12\t" in open(report.manifest).read()
    assert open(report.manifest).read() == manifest_text(Plan(wf, root))
    # the MD step's folder still says "MD model 12": by its title another step's until the step is written
    # again (a plan made again from scratch gives its ids out afresh, so the id alone does not tell)
    plan = Plan(wf, root)
    rows = {line.split("\t")[2]: line.split("\t") for line in open(report.manifest).read().splitlines()[1:]}
    assert rows[restraints.id][7] == "this step" and rows[build.id][7] == "this step"
    assert rows[md.id][7].startswith(f"{md.id} 'MD model 12' (step 03, written 20")
    [warning] = [i for i in plan.steps[md.id].issues if i.keys == ("folder_tag",)]
    assert warning.level == "warning" and warning.message.startswith(
        f"03_amber-md_model12/ holds the files of 03 · MD model 12 · model12 (node {md.id}, written 20")
    assert warning.message.endswith("writing this step puts its files beside them and moves a completed run "
                                    "there aside as this step's; give it a folder tag or another step number, "
                                    "or, if it is that step under a new title, write it again")
    Plan(wf, root).write([md.id])
    assert not [i for i in Plan(wf, root).steps[md.id].issues if i.keys == ("folder_tag",)]
    # a step that is not written (errors) gets no identity file and no folder
    other = add_node(wf, "amber-md")
    report = Plan(wf, root).write(None)
    assert other.id in report.skipped and other.id not in report.identity
    assert not os.path.exists(os.path.join(root, "04_amber-md"))
    assert f"\t{other.id}\tamber-md\t" in open(report.manifest).read()  # MANIFEST lists every step
    # the stand-alone wizard's one-step plan is not the folder's plan: it leaves MANIFEST alone
    manifest = open(report.manifest).read()
    solo = PlanNode(id="wizard-user-script", tool="user-script", step=7,
                    settings=TOOLS["user-script"].normalise({"script": "true\n"}))
    report = single_step_plan(wf, solo, root).write([solo.id], master=False)
    assert report.manifest == "" and open(os.path.join(root, MANIFEST)).read() == manifest
    assert read_identity(os.path.join(root, "07_custom"))["node"] == "wizard-user-script"
    # without step folders: no identity files, MANIFEST names the plan folder
    flat = WorkflowParams(project_name="apt1", stamp=STAMP, step_folders=False)
    branch(flat, "model12")
    report = Plan(flat, str(tmp_path / "flat")).write()
    assert report.identity == {} and not glob.glob(str(tmp_path / "flat" / "**" / IDENTITY), recursive=True)
    lines = open(report.manifest).read().splitlines()[1:]
    assert [line.split("\t")[1] for line in lines] == [".", ".", ".", "-"]


def test_cells_of_the_manifest_stay_on_their_line(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    node = add_node(wf, "user-script", title="a\ttitle\nwith breaks", settings={"script": "true\n"})
    text = manifest_text(Plan(wf, str(tmp_path)))
    assert text.splitlines()[1].split("\t") == ["01", "01_custom", node.id, "user-script",
                                                "a title with breaks", "-", "-", "-"]


# ---------------------------------------------------------------------------
# a renumbered plan against the tree it wrote (acceptance criterion 7)
# ---------------------------------------------------------------------------


def _project(tmp_path, wf):
    project = Project()
    project.workflow = wf
    project.workflow.out_dir = str(tmp_path / "plan")
    path = str(tmp_path / "plan.mdatk.toml")
    project.save(path)
    return path


def _manifest(capsys, *args):
    capsys.readouterr()
    code = main(["workflow", "manifest", *args, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_a_renumbered_plan_is_matched_to_the_tree_it_wrote(tmp_path, capsys):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = branch(wf, "model12", "model 12")
    plain = branch(wf, "", "model 13")  # no folder tags: only the identity files know it
    proj = _project(tmp_path, wf)
    assert main(["workflow", "write", "-p", proj, "-q"]) == 0
    written = json.loads(capsys.readouterr().out)
    root = written["folder"]
    assert written["manifest"] == os.path.join(root, MANIFEST)
    plan = Plan(Project.load(proj).workflow, root)
    fake_run(plan, a[2].id)  # model 12's MD has run: 03_amber-md_model12
    fake_run(plan, plain[2].id)  # model 13's: 06_amber-md
    code, data = _manifest(capsys, "-p", proj)
    assert code == 0 and data["conflicts"] == [] and data["others"] == []
    assert {s["folder"]: s["state"] for s in data["steps"]} == {
        "01_amber-build_model12": "ours", "02_amber-restraints_model12": "ours",
        "03_amber-md_model12": "ours", "04_amber-build": "ours", "05_amber-restraints": "ours",
        "06_amber-md": "ours"}
    # the user's own plan generator adds two branches in front of the others: every step is renumbered
    project = Project.load(proj)
    wf2 = project.workflow
    for node in wf2.nodes:
        if node.step:
            node.step += 6
    front = branch(wf2, "inv1", "invasion 1") + branch(wf2, "", "invasion 2")
    for k, node in enumerate(n for n in front if n.tool != "mdatk"):
        node.step = k + 1
    project.save(proj)
    code, data = _manifest(capsys, "-p", proj)
    steps = {s["node"]: s for s in data["steps"]}
    # model 12's MD (now step 09) is not in 09_amber-md_model12/, its files are in 03_amber-md_model12/
    md12 = steps[a[2].id]
    assert (md12["step"], md12["folder"], md12["state"]) == (9, "09_amber-md_model12", "absent")
    # (by its folder tag; the node id only confirms it)
    assert [(f["folder"], f["step"], f["matched_by"], f["node_id_agrees"]) for f in md12["found"]] == [
        ("03_amber-md_model12", 3, "folder tag", True)]
    # model 13's MD (now 12): found in 06_amber-md/ by its title although no name says model 13
    md13 = steps[plain[2].id]
    assert md13["state"] == "absent" and [(f["folder"], f["matched_by"]) for f in md13["found"]] == [
        ("06_amber-md", "title")]
    assert data["by_node_id_only"] == []
    # the untagged invasion branch now owns the numbers 04-06: its folders hold model 13's files
    inv2_md = steps[front[6].id]
    assert (inv2_md["folder"], inv2_md["state"]) == ("06_amber-md", "other")
    assert inv2_md["here"]["node"] == plain[2].id and inv2_md["here"]["title"] == "MD model 13"
    assert code == 1 and sorted(data["conflicts"]) == sorted(n.id for n in front[4:7])
    # the text says the same
    capsys.readouterr()
    assert main(["workflow", "manifest", "-p", proj]) == 1
    text = capsys.readouterr().out
    line = next(ln for ln in text.splitlines() if ln.startswith("09 "))
    assert "09_amber-md_model12/" in line
    assert "not there; its files are in 03_amber-md_model12/ (step 03, by folder tag and node id)" in line
    line = next(ln for ln in text.splitlines() if ln.startswith("06 "))
    assert "holds another step's files: 06_amber-md/ (step 06, node " in line and "MD model 13" in line
    assert "3 step(s) would write into a folder that holds another step's files" in text
    # --table: the MANIFEST a write of every step would leave, renumbered (each folder then holds its step's
    # files); MANIFEST as the folders are now says whose files the renumbered steps' folders hold
    capsys.readouterr()
    assert main(["workflow", "manifest", "-p", proj, "--table"]) == 0
    table = capsys.readouterr().out
    plan = Plan(Project.load(proj).workflow, root)
    assert table == manifest_text(plan, [n for n in plan.order if plan.steps[n].tool.generates])
    assert f"09\t09_amber-md_model12\t{a[2].id}\tamber-md\tMD model 12\tmodel12\t" in table
    assert all(line.endswith("\tthis step") for line in table.splitlines()[1:] if line[:2] != "-\t")
    now = {line.split("\t")[2]: line.split("\t")[7] for line in manifest_text(plan).splitlines()[1:]}
    assert now[a[2].id] == "-" and now[front[6].id].startswith(f"{plain[2].id} 'MD model 13' (step 06, written ")


def test_matching_without_node_ids_by_folder_tag_title_or_names_alone(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = branch(wf, "model12", "model 12")
    b = branch(wf, "", "model 13")
    root = str(tmp_path / "plan")
    Plan(wf, root).write()
    # a plan generator that makes new node ids as well as new numbers: folder tag, then title
    wf2 = WorkflowParams(project_name="apt1", stamp=STAMP)
    for node in wf.nodes:
        wf2.nodes.append(PlanNode(id="g" + node.id, tool=node.tool, step=node.step + 20 if node.step else 0,
                                  title=node.title, settings=dict(node.settings), files=dict(node.files),
                                  folder_tag=node.folder_tag))
    result = match(Plan(wf2, root))
    found = {s.node_id: [(f.name, c.how) for f, c in s.found] for s in result.steps}
    assert found["g" + a[0].id] == [("01_amber-build_model12", "folder tag")]
    assert found["g" + b[2].id] == [("06_amber-md", "title")]
    assert not any(c.confirmed for s in result.steps for _f, c in s.found)  # the node ids are new
    assert result.others == [] and result.conflicts == [] and result.weak == []
    # from the names alone (no identity files): the folder tag in a folder's name
    for path in glob.glob(os.path.join(root, "*", IDENTITY)):
        os.remove(path)
    result = match(Plan(wf2, root))
    found = {s.node_id: [(f.name, c.how) for f, c in s.found] for s in result.steps}
    assert found["g" + a[2].id] == [("03_amber-md_model12", "the folder tag in its name")]
    assert found["g" + b[2].id] == []  # nothing on disk says which step 06_amber-md was
    assert sorted(f.name for f in result.others) == ["04_amber-build", "05_amber-restraints", "06_amber-md"]
    assert [f.identity for f in scan_tree(root)] == [False] * 6
    # the same plan, not renumbered: an untagged folder of its own name cannot be told from another's
    result = match(Plan(wf, root))
    assert {s.folder: s.state for s in result.steps}["06_amber-md"] == "unknown"
    assert {s.folder: s.state for s in result.steps}["03_amber-md_model12"] == "ours"
    assert {s.folder: s.how for s in result.steps}["03_amber-md_model12"] == "the folder tag in its name"


def test_cli_write_dry_run_lists_the_records_and_check_and_status_see_tags(tmp_path, capsys):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a = branch(wf, "model12", "model 12")
    proj = _project(tmp_path, wf)
    capsys.readouterr()
    assert main(["workflow", "write", "-p", proj, "--dry-run"]) == 0
    data = json.loads(capsys.readouterr().out)
    root = data["folder"]
    assert data["records"] == [os.path.join(root, MANIFEST)] + [
        os.path.join(root, f, IDENTITY) for f in ("01_amber-build_model12", "02_amber-restraints_model12",
                                                  "03_amber-md_model12")]
    assert not os.path.exists(root)
    assert main(["workflow", "manifest", "-p", proj]) == 1
    assert "is not a folder (nothing written yet; --tree names another one)" in capsys.readouterr().err
    assert main(["workflow", "write", "-p", proj, "-q"]) == 0
    capsys.readouterr()
    plan = Plan(Project.load(proj).workflow, root)
    fake_run(plan, a[2].id)
    assert main(["workflow", "status", "-p", proj, "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert [s["state"] for s in status["steps"] if s["step"] == 3] == ["done"]
    assert main(["rnado", "manifest", "-p", proj, "--tree", root]) == 0  # rnado ACTION is workflow ACTION


# ---------------------------------------------------------------------------
# node ids are not identities: plans made again from scratch, steps deleted and added again
# ---------------------------------------------------------------------------


def generated(models, tagged=False):
    """A plan made from scratch the way a plan generator of one's own makes it (the user's
    make_free_md_plan.py): a fresh WorkflowParams and one Amber branch per model, so node ids (n1, n2 ...)
    and step numbers both follow a branch's position.  Returns it and {model: node id of its MD step}."""
    wf = WorkflowParams(project_name="free", stamp=STAMP)
    mds = {}
    for model in models:
        build, restraints, md, _analysis = apply_template(wf, "amber")
        build.files["structure"] = f"/data/{model}.pdb"
        restraints.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
        for node, what in ((build, "build"), (restraints, "restraints"), (md, "production")):
            node.title = f"{what} {model}"
            node.folder_tag = model if tagged else ""
        mds[model] = md.id
    return wf, mds


def test_a_plan_made_again_with_branches_in_front_is_not_matched_by_its_recycled_node_ids(tmp_path, capsys):
    root = str(tmp_path / "plain" / "plan")
    wf1, md1 = generated(["model12", "model13"])
    first = Plan(wf1, root)
    first.write()
    fake_run(first, md1["model12"])
    wf2, md2 = generated(["inv1", "inv2", "model12", "model13"])
    assert md2["inv1"] == md1["model12"] == "n3"  # the same id names another step now
    plan = Plan(wf2, root)
    result = match(plan)
    steps = {s.node_id: s for s in result.steps}
    # invasion 1's MD would write into 03_amber-md/, which holds model 12's production: a conflict
    inv1 = steps["n3"]
    assert (inv1.folder, inv1.state, inv1.owner, inv1.differs) == ("03_amber-md", "other",
                                                                   "09 · production model12", ["title"])
    assert inv1.where().startswith("holds another step's files: 03_amber-md/ (step 03, node n3: 03 · production "
                                   "model12, written 20")
    assert inv1.where().endswith(", now 09 · production model12; this step has its node id with another title")
    # model 12's production (now step 09) is found where its files are, by its title
    m12 = steps[md2["model12"]]
    assert m12.state == "absent" and [(f.name, c.how, c.confirmed) for f, c in m12.found] == [
        ("03_amber-md", "title", False)]
    assert m12.where() == "not there; its files are in 03_amber-md/ (step 03, by title)"
    assert sorted(s.node_id for s in result.conflicts) == ["n1", "n2", "n3", "n5", "n6", "n7"]
    assert result.others == [] and result.weak == []
    # the check says so before anything is written over them (a warning: writing stays possible)
    [warning] = [i for i in plan.steps["n3"].issues if i.keys == ("folder_tag",)]
    assert warning.level == "warning" and not errors(plan)
    assert warning.message.startswith("03_amber-md/ holds the files of 03 · production model12 (node n3, "
                                      "written 20")
    assert warning.message.endswith(", now 09 · production model12: writing this step puts its files beside "
                                    "them and moves a completed run there aside as this step's; give it a "
                                    "folder tag or another step number")
    assert plan.steps["n3"].held == warning.message.split(": writing this step")[0]
    assert not [i for i in plan.steps[md2["model12"]].issues if i.keys == ("folder_tag",)]  # no folder yet
    # 'Check previous runs' finds model 12's run in 03_amber-md/: not invasion 1's, so it is not offered
    runs = runstate.check_runs(plan)
    assert runs["n3"].state == "done" and runs["n3"].held == plan.steps["n3"].held
    assert not runstate.offered(runs["n3"]) and runstate.adopt(plan, runs) == []
    assert "- not this step's run: 03_amber-md/ holds the files of 03 · production model12" in runs["n3"].summary()
    # writing moves the run aside: its manifest names the step that wrote it, so it is found as model 12's
    moving = supersede.pending(plan)
    assert list(moving) == ["n3"] and moving["n3"].held == plan.steps["n3"].held
    assert moving["n3"].lines()[1] == f"  not this step's run: {plan.steps['n3'].held}; it would move aside as " \
                                      "this step's"
    assert moving["n3"].plain()["held"] == plan.steps["n3"].held
    report = plan.write()
    run = report.superseded["n3"].target
    manifest = tomlio.load(os.path.join(root, run, "manifest.toml"))
    assert (manifest["node"], manifest["title"]) == ("n3", "03 · production inv1")  # the step written
    assert {k: manifest["written_by"][k] for k in ("node", "title", "label", "step", "folder_tag", "tool")} == {
        "node": "n3", "title": "production model12", "label": "03 · production model12", "step": 3,
        "folder_tag": "", "tool": "amber-md"}
    text = open(os.path.join(root, run, "manifest.toml")).read()
    assert "# Not that step's run: 03_amber-md/ holds the files of 03 · production model12 (node n3" in text
    tomlio.loads(text)
    result = match(Plan(wf2, root))
    steps = {s.node_id: s for s in result.steps}
    assert steps["n3"].state == "ours" and steps["n3"].found == []
    assert [(f.name, c.how, f.moved != "") for f, c in steps[md2["model12"]].found] == [(run, "title", True)]
    assert steps[md2["model12"]].where().startswith(f"this step's; and in {run}/ (step 03, moved 20")
    assert result.conflicts == [] and result.others == []
    # the CLI on a fresh tree: exit status 1 for the conflicts
    proj = _project(tmp_path, generated(["model12", "model13"])[0])
    assert main(["workflow", "write", "-p", proj, "-q"]) == 0
    project = Project.load(proj)
    project.workflow = generated(["inv1", "model12", "model13"])[0]
    project.workflow.out_dir = str(tmp_path / "plan")
    project.save(proj)
    capsys.readouterr()
    assert main(["workflow", "manifest", "-p", proj]) == 1
    assert ("6 step(s) would write into a folder that holds another step's files: 01 · build inv1, 02 · "
            "restraints inv1, 03 · production inv1, 04 · build model12, 05 · restraints model12, 06 · production "
            "model12") in capsys.readouterr().out


def test_tagged_branches_made_again_match_by_folder_tag_not_by_recycled_ids(tmp_path):
    root = str(tmp_path / "plan")
    wf1, md1 = generated(["model12", "model13"], tagged=True)
    Plan(wf1, root).write()
    wf2, md2 = generated(["inv1", "inv2", "model12", "model13"], tagged=True)
    result = match(Plan(wf2, root))
    steps = {s.node_id: s for s in result.steps}
    # n3 is invasion 1's MD now: 03_amber-md_model12/ (whose identity file names n3) is not its folder
    assert (steps["n3"].folder, steps["n3"].state, steps["n3"].found) == ("03_amber-md_inv1", "absent", [])
    m12 = steps[md2["model12"]]
    assert (m12.folder, m12.state) == ("09_amber-md_model12", "absent")
    assert [(f.name, c.how) for f, c in m12.found] == [("03_amber-md_model12", "folder tag")]
    assert m12.where() == "not there; its files are in 03_amber-md_model12/ (step 03, by folder tag)"
    assert result.conflicts == [] and result.others == []
    # tagged folders never collide, so nothing warns
    assert not [i for _n, i in Plan(wf2, root).all_issues() if i.keys == ("folder_tag",)]


def test_folder_tag_then_title_tell_a_folder_the_node_id_only_confirms_and_disagreements_are_ambiguous(
        tmp_path, capsys):
    # steps without titles or folder tags: only their node ids tell them apart, and the answer says so
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    first, second = branch(wf), branch(wf)
    proj = _project(tmp_path, wf)
    assert main(["workflow", "write", "-p", proj, "-q"]) == 0
    root = json.loads(capsys.readouterr().out)["folder"]
    plan = Plan(Project.load(proj).workflow, root)
    result = match(plan)
    assert {s.state for s in result.steps} == {"ours"} and result.conflicts == []
    assert {s.how for s in result.steps} == {"node id"} and all(s.weak for s in result.steps)
    assert all(s.where() == "this step's (by node id only)" for s in result.steps)
    assert result.text().endswith("\n\nby node id only: nothing else on record tells these steps from the others of "
                                  "their type, and a plan made again from scratch gives node ids out afresh; titles "
                                  "or folder tags tell them for certain")
    code, data = _manifest(capsys, "-p", proj)
    assert code == 0 and sorted(data["by_node_id_only"]) == sorted(n.id for n in first[:3] + second[:3])
    assert not [i for _n, i in plan.all_issues() if i.keys == ("folder_tag",)]  # the planner does not warn
    # MANIFEST after a write of one step: the others hold their files by node id only
    report = Plan(Project.load(proj).workflow, root).write([first[0].id], master=False)
    holds = {line.split("\t")[2]: line.split("\t")[7] for line in open(report.manifest).read().splitlines()[1:]}
    assert holds[first[0].id] == "this step" and holds[second[2].id] == "this step (node id only)"
    # a title tells a folder whose node id another step has now, and the node id confirms what the titles tell
    root = str(tmp_path / "titled")
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    a, b = branch(wf, "", "model 12"), branch(wf, "", "model 13")
    Plan(wf, root).write()
    result = match(Plan(wf, root))
    assert [s.where() for s in result.steps] == ["this step's"] * 6  # by title, confirmed by the node id
    assert {(s.how, s.told.confirmed) for s in result.steps} == {("title", True)}
    # the same titles twice: the node id tells the two apart ...
    c = branch(wf, "", "model 13")
    Plan(wf, root).write()
    result = match(Plan(wf, root))
    told = {s.node_id: s for s in result.steps}
    assert told[b[2].id].how == told[c[2].id].how == "node id" and told[a[2].id].how == "title"
    # ... and when a plan made again gives them new ids, nothing does: no step is picked
    wf2 = WorkflowParams(project_name="apt1", stamp=STAMP)
    for node in wf.nodes:
        wf2.nodes.append(PlanNode(id="g" + node.id, tool=node.tool, step=node.step, title=node.title,
                                  settings=dict(node.settings), files=dict(node.files)))
    plan = Plan(wf2, root)
    result = match(plan)
    md = {s.node_id: s for s in result.steps}["g" + b[2].id]
    assert md.state == "ambiguous" and md in result.conflicts
    assert md.where() == (f"holds files no step can be told for: 06_amber-md/ (step 06, node {b[2].id}: 06 · MD "
                          f"model 13, written {_written(root, '06_amber-md')}); its type, folder tag and title fit 2 "
                          f"steps (06 · MD model 13, 09 · MD model 13), its node id none of them")
    assert {s.state for s in result.steps if s.title.endswith("model 12")} == {"ours"}  # by title
    [warning] = [i for i in plan.steps["g" + b[2].id].issues if i.keys == ("folder_tag",)]
    assert warning.message.startswith(f"06_amber-md/ holds the files of 06 · MD model 13 (node {b[2].id}, written ")
    assert "; its type, folder tag and title fit 2 steps" in warning.message
    # the folder tag first: its one step is told by it, whatever its node id; one with another title is no step
    root = str(tmp_path / "tagged")
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    x = branch(wf, "model12", "model 12")
    Plan(wf, root).write()
    fake_run(Plan(wf, root), x[2].id)
    x[2].title = "production model 12"  # renamed since, or another step with the tag: no step is picked
    y = add_node(wf, "amber-md", title="MD model 12", step=7)  # the old title, without the tag
    plan = Plan(wf, root)
    result = match(plan)
    md = {s.node_id: s for s in result.steps}[x[2].id]
    assert (md.state, md.told.node, md.differs) == ("ambiguous", "", ["title"])
    assert md.where().endswith("; 03 · production model 12 · model12 has its folder tag and node id but not its "
                               "title 'MD model 12'")
    assert {s.node_id: s.found for s in result.steps}[y.id] == []  # the title alone does not overrule the tag
    [warning] = [i for i in plan.steps[x[2].id].issues if i.keys == ("folder_tag",)]
    assert warning.message.endswith("or, if it is that step under a new title, write it again")
    runs = runstate.check_runs(plan)  # its completed run is not offered while that is not told
    assert runs[x[2].id].state == "done" and runs[x[2].id].held and not runstate.offered(runs[x[2].id])
    # another step with the folder tag but no title does not take the folder its node id's step has a claim to
    z = add_node(wf, "amber-md", step=8)
    z.folder_tag = "model12"
    md = {s.node_id: s for s in match(Plan(wf, root)).steps}[x[2].id]
    assert (md.state, md.told.doubt) == ("ambiguous", "03 · production model 12 · model12 has its folder tag and "
                                                      "node id but not its title 'MD model 12'")
    wf.nodes.remove(z)
    # the records and the folder's name disagree (a folder renamed by hand): no step is picked
    x[2].title = "MD model 12"
    os.rename(os.path.join(root, "03_amber-md_model12"), os.path.join(root, "05_amber-md_model12"))
    x[2].step = 5
    result = match(Plan(wf, root))
    md = {s.node_id: s for s in result.steps}[x[2].id]
    assert md.state == "ambiguous" and md.where().endswith(
        "; its records were written for 03_amber-md_model12/, not 05_amber-md_model12/ (renamed or copied since)")
    capsys.readouterr()
    assert main(["workflow", "manifest", "-p", _project(tmp_path, wf), "--tree", root]) == 1
    assert "1 step(s) would write into a folder that holds another step's files (1 of them: files no step can be " \
           "told for): 05 · MD model 12 · model12" in capsys.readouterr().out


def _written(root, folder):
    return read_identity(os.path.join(root, folder))["written"].strftime("%Y-%m-%d %H:%M")


def test_a_deleted_step_never_gives_its_id_to_a_new_one_and_its_folder_warns(tmp_path, capsys):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    m12 = branch(wf, "", "model 12")
    m13 = branch(wf, "", "model 13")
    assert [n.id for n in m12 + m13] == [f"n{k}" for k in range(1, 9)] and wf.last_node == 8
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    plan.write()
    fake_run(plan, m13[2].id)  # model 13's MD has run in 06_amber-md/
    for node in m13:  # its branch is deleted; the folders stay
        remove_node(wf, node.id)
    m14 = branch(wf, "", "model 14")
    assert [n.id for n in m14] == ["n9", "n10", "n11", "n12"]  # not n5-n8 again
    assert [n.step for n in m14[:3]] == [4, 5, 6]  # the numbers after the highest one: 04-06 again
    plan = Plan(wf, root)
    md14 = m14[2].id
    [warning] = [i for i in plan.steps[md14].issues if i.keys == ("folder_tag",)]
    assert warning.message.startswith(f"06_amber-md/ holds the files of 06 · MD model 13 (node {m13[2].id}, "
                                      "written 20")
    assert "under a new title" not in warning.message  # another node id: not this step renamed
    result = match(plan)
    assert sorted(s.node_id for s in result.conflicts) == sorted(n.id for n in m14[:3])
    [md] = [s for s in result.steps if s.node_id == md14]
    assert md.where().startswith(f"holds another step's files: 06_amber-md/ (step 06, node {m13[2].id}: "
                                 "06 · MD model 13, written 20")
    runs = runstate.check_runs(plan)
    assert runs[md14].state == "done" and runs[md14].held and not runstate.offered(runs[md14])
    # a project file keeps the counter (only once set), so a reloaded plan goes on from it
    proj = _project(tmp_path, wf)
    loaded = Project.load(proj)
    assert loaded.workflow.last_node == 12 and "last_node = 12" in open(proj).read()
    remove_node(loaded.workflow, m14[3].id)
    assert add_node(loaded.workflow, "note").id == "n13"
    # the CLI: --adopt refuses the run, and says why
    capsys.readouterr()
    assert main(["workflow", "status", "-p", proj, "--adopt", "--step", "06_amber-md"]) == 0
    out = capsys.readouterr().out
    assert f"06 · MD model 14: not adopted: the run is not this step's (06_amber-md/ holds the files of 06 · MD " \
           f"model 13 (node {m13[2].id}" in out
    assert Project.load(proj).workflow.nodes[-2].adopted == ""
    assert main(["workflow", "status", "-p", proj, "--json"]) == 0
    steps = json.loads(capsys.readouterr().out)["steps"]
    assert [s["title"] for s in steps if "held" in s] == ["04 · build model 14", "05 · restraints model 14",
                                                         "06 · MD model 14"]
    # an old project file without the counter goes on after its highest id
    old = WorkflowParams(nodes=[PlanNode(id="n2", tool="note"), PlanNode(id="n7", tool="note"),
                                PlanNode(id="wizard-x", tool="note")])
    assert add_node(old, "note").id == "n8" and old.last_node == 8


def test_folders_written_under_another_project_name_are_another_plans(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    build, restraints, md, _analysis = branch(wf, "", "model 12")
    root = str(tmp_path / "plan")
    Plan(wf, root).write()
    wf.project_name = "apt2"  # renamed (or a copy of the project writing into the same folder)
    plan = Plan(wf, root)
    [warning] = [i for i in plan.steps[md.id].issues if i.keys == ("folder_tag",)]
    assert warning.message.startswith(f"03_amber-md/ holds the files of 03 · MD model 12 (node {md.id} of plan "
                                      "apt1, written 20")
    assert warning.message.endswith(": writing this step puts this plan's files beside them; give it a folder "
                                    "tag or another step number to keep the two apart")
    result = match(plan)
    [step] = [s for s in result.steps if s.node_id == md.id]
    assert step.state == "other" and step.differs == ["project"]
    assert step.where().endswith("; this step has its node id with another project")
    assert len(result.conflicts) == 3
    Plan(wf, root).write()  # written again: the folders are this plan's now (the files of apt1 stay)
    assert match(Plan(wf, root)).conflicts == []
    assert any(n.startswith("apt1_03_") for n in os.listdir(os.path.join(root, "03_amber-md")))


def test_manifest_says_what_each_folder_holds_after_a_partial_write(tmp_path):
    root = str(tmp_path / "plan")
    wf1, md1 = generated(["model12", "model13"])
    Plan(wf1, root).write()
    wf2, md2 = generated(["inv1", "inv2", "model12", "model13"])
    plan = Plan(wf2, root)
    report = plan.write([plan.order[0]], master=False)  # 'Write' on one new box: invasion 1's build
    assert set(report.identity) == {"n1"}
    rows = {line.split("\t")[2]: line.split("\t") for line in open(report.manifest).read().splitlines()[1:]}
    assert rows["n1"][7] == "this step"
    when = read_identity(os.path.join(root, "03_amber-md"))["written"].strftime("%Y-%m-%d %H:%M")
    assert rows["n3"][:5] == ["03", "03_amber-md", "n3", "amber-md", "production inv1"]
    assert rows["n3"][7] == f"n3 'production model12' (step 03, written {when})"  # not invasion 1's files
    assert rows[md2["model12"]][7] == "-"  # 09_amber-md/: not there yet
    assert rows["n4"][7] == "-"  # the analysis box has no folder
    # a folder without an identity file: nothing says whose it is
    os.remove(os.path.join(root, "06_amber-md", IDENTITY))
    rows = {line.split("\t")[2]: line.split("\t") for line in manifest_text(Plan(wf2, root)).splitlines()[1:]}
    assert rows["n7"][7] == "unknown (no MDATK_step.toml)"


def test_titles_with_line_breaks_keep_the_records_valid(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    title = 'MD "model" 12\nsecond line\tand tab é ü \\ end\r\x0b\x01!'
    node = add_node(wf, "user-script", title=title, settings={"script": "true\n"})
    node.folder_tag = "model12"
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    report = plan.write()
    ident = read_identity(os.path.join(root, "01_custom_model12"))
    assert ident is not None and ident["title"] == title and ident["label"] == f"01 · {title} · model12"
    assert [(f.name, f.identity, f.node) for f in scan_tree(root)] == [("01_custom_model12", True, node.id)]
    text = open(report.identity[node.id]).read()
    assert text.splitlines()[0] == ("# This folder holds the files of step 01 · MD \"model\" 12 second line and "
                                    "tab é ü \\ end ! · model12 of plan apt1, written by MDATK " +
                                    text.splitlines()[0].rsplit("MDATK ", 1)[1])
    # the plan without its tag: the folder is told by its identity file (no longer by its name)
    node.folder_tag = ""
    result = match(Plan(wf, root))
    assert [f.name for f in result.others] == ["01_custom_model12"]
    assert all("\n" not in line and "\t" not in line for line in result.text().splitlines()[1:])
    assert "the plan's n1 (01 · MD \"model\" 12 second line and tab é ü \\ end !) has another folder tag" \
        in result.text()
    assert not [c for c in result.text() if ord(c) < 32 and c != "\n"]
    # header comments of every TOML file MDATK writes: one comment line per line of a header item
    assert tomlio.loads(tomlio.dumps({"a": 1}, None, ["one\ntwo", "tab\there \x07bell", ""])) == {"a": 1}
    assert tomlio.dumps({"a": 1}, None, ["one\ntwo", "plain"]).splitlines()[:4] == ["# one", "# two", "# plain", ""]
    # a superseded run's manifest with such a label
    node.folder_tag = "model12"
    fake_run(Plan(wf, root), node.id)
    report = Plan(wf, root).write()
    moved = tomlio.load(os.path.join(root, report.superseded[node.id].target, "manifest.toml"))
    assert moved["written_by"]["title"] == title and moved["title"] == f"01 · {title} · model12"


def test_steps_sharing_a_number_have_titles_and_a_step_selection_of_their_own(tmp_path, capsys):
    from mdatk.workflows.runner import plan_jobs

    project = Project()
    wf = project.workflow
    wf.project_name, wf.stamp, wf.out_dir = "apt1", STAMP, str(tmp_path / "plan")
    a = add_node(wf, "user-script", settings={"script": "true\n"})
    b = add_node(wf, "user-script", step=1, settings={"script": "true\n"})
    c = add_node(wf, "user-script", settings={"script": "true\n"})
    a.folder_tag, b.folder_tag = "model12", "model13"
    plan = Plan(wf, wf.out_dir)
    assert errors(plan) == []
    # the tag is in the step's title (logs, messages, the step log), not in the title on its box
    assert [plan.steps[n.id].label for n in (a, b, c)] == ["01 · Custom script · model12",
                                                           "01 · Custom script · model13", "02 · Custom script"]
    assert step_title(a, tag=False) == "01 · Custom script" and plan.steps[a.id].ctx.title == "01 · Custom script"
    assert [(j.step, j.title) for j in plan_jobs(plan, [a.id, b.id])] == [(1, "01 · Custom script · model12"),
                                                                       (1, "01 · Custom script · model13")]
    assert any(i.message == "step number 01 is used by 01 · Custom script · model12, 01 · Custom script · model13"
               for i in plan.issues)
    # a step's files do not change with the tag (the custom script's heading is the box title's)
    untagged = generate_step(Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[
        PlanNode(id="x", tool="user-script", step=1, settings=dict(a.settings))]), "").steps["x"])
    assert [f.content for f in generate_step(plan.steps[a.id])] == [f.content for f in untagged]
    path = str(tmp_path / "two.mdatk.toml")
    project.save(path)
    # --step: a number selects every step with it; a folder or a node id one step
    capsys.readouterr()
    for step, folders_written in (("1", ["01_custom_model12", "01_custom_model13"]),
                                  ("01_custom_model13", ["01_custom_model13"]),
                                  ("01_custom_model13/", ["01_custom_model13"]), (a.id, ["01_custom_model12"]),
                                  ("02_custom", ["02_custom"])):
        assert main(["workflow", "write", "-p", path, "--step", step, "--dry-run", "-q"]) == 0
        written = json.loads(capsys.readouterr().out)["write"]
        assert sorted({os.path.basename(os.path.dirname(p)) for p in written}) == folders_written, step
    assert main(["workflow", "status", "-p", path, "--step", "03_custom"]) == 1
    assert "no step '03_custom' in the plan: give a step number, a step folder (e.g. 03_amber-md) or a node id " \
           "(e.g. n3)" in capsys.readouterr().err
    assert main(["workflow", "run", "-p", path, "--step", "7"]) == 1
    assert "no step number 7 in the plan" in capsys.readouterr().err
    # an adopted step's hint names the one step
    from mdatk.cli import _step_arg
    assert (_step_arg(plan, a.id), _step_arg(plan, c.id)) == ("01_custom_model12", "2")


def test_a_record_that_cannot_be_written_does_not_fail_the_write(tmp_path, capsys):
    project = Project()
    project.workflow.project_name = "apt1"
    project.workflow.out_dir = str(tmp_path / "plan")
    add_node(project.workflow, "user-script", settings={"script": "true\n"})
    os.makedirs(os.path.join(project.workflow.out_dir, MANIFEST))  # MANIFEST cannot be replaced
    path = str(tmp_path / "p.mdatk.toml")
    project.save(path)
    capsys.readouterr()
    assert main(["workflow", "write", "-p", path, "-q"]) == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["manifest"] == "" and out["records_error"].startswith(os.path.join(out["folder"], MANIFEST) + ": ")
    assert "WARNING  MANIFEST / MDATK_step.toml not written: " in captured.err
    stamp = Project.load(path).workflow.stamp
    assert stamp  # the write went through: the project keeps its stamp, so the next write keeps the names
    root = out["folder"]
    assert sorted(os.listdir(os.path.join(root, "01_custom"))) == [
        IDENTITY, f"apt1_01_custom_{stamp}.sh", f"apt1_01_script_{stamp}.sh"]
    assert not glob.glob(os.path.join(root, "**", "*.tmp"), recursive=True)
    # an identity file that cannot be written: the same
    plan = Plan(Project.load(path).workflow, root)
    os.remove(os.path.join(root, "01_custom", IDENTITY))
    os.makedirs(os.path.join(root, "01_custom", IDENTITY))
    os.rmdir(os.path.join(root, MANIFEST))
    report = plan.write()
    assert report.identity == {} and report.manifest == os.path.join(root, MANIFEST)
    assert report.records_error.startswith(os.path.join(root, "01_custom", IDENTITY) + ": ")
    assert report.written and not glob.glob(os.path.join(root, "**", "*.tmp"), recursive=True)


# ---------------------------------------------------------------------------
# project files
# ---------------------------------------------------------------------------


def test_project_files_without_folder_tags_load_and_save_as_before(tmp_path):
    paths = sorted(glob.glob(os.path.join(EXAMPLES, "*.mdatk.toml")))
    assert paths
    for k, path in enumerate(paths):
        project = Project.load(path)
        assert all(n.folder_tag == "" for n in project.workflow.nodes) and project.workflow.last_node == 0
        out = str(tmp_path / f"{k}.mdatk.toml")
        project.save(out)
        text = open(out).read()
        assert "folder_tag" not in text and "last_node" not in text  # the id counter: once a step is added
        for node in tomlio.load(out)["workflow"]["nodes"]:
            assert set(node) <= NODE_KEYS  # nothing that was not saved before
        again = str(tmp_path / f"{k}-again.mdatk.toml")
        Project.load(out).save(again)
        strip = lambda t: [ln for ln in t.splitlines() if not ln.startswith("saved = ")]  # noqa: E731
        assert strip(open(again).read()) == strip(text)
    assert set(_to_plain(PlanNode(id="n1"))) == NODE_KEYS


def test_a_folder_tag_is_saved_with_its_comment_and_read_back(tmp_path):
    project = Project()
    node = add_node(project.workflow, "amber-build")
    node.folder_tag = "model12"
    add_node(project.workflow, "amber-md")
    path = str(tmp_path / "p.mdatk.toml")
    project.save(path)
    text = open(path).read()
    assert text.count("folder_tag") == 1
    assert "# Added to the step's folder name, e.g. \"model12\": 02_amber-build_model12 instead of" in text
    assert 'folder_tag = "model12"' in text
    loaded = Project.load(path)
    assert [n.folder_tag for n in loaded.workflow.nodes] == ["model12", ""]
    assert Plan(loaded.workflow).steps[node.id].ctx.folder == "01_amber-build_model12"
