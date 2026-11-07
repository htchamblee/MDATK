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
from mdatk.workflows.plan import Plan, add_node, apply_template, connect, single_step_plan
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
    assert folder_tag_problem("model 12") == ("folder tag 'model 12': use letters, digits and '-' only "
                                              "(e.g. 'model-12')")
    assert folder_tag_problem("model_12").endswith("(e.g. 'model-12')")
    assert folder_tag_problem("-x") == "folder tag '-x': use letters, digits and '-' only (e.g. 'x')"
    assert folder_tag_problem("__") == "folder tag '__': use letters, digits and '-' only"
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
    # MANIFEST: the existing file has no folder, the note is left out
    lines = open(report.manifest).read().splitlines()
    assert lines[0].split("\t") == list(COLUMNS) == ["step", "folder", "node", "tool", "title", "folder_tag",
                                                     "inputs"]
    rows = {line.split("\t")[2]: line.split("\t") for line in lines[1:]}
    assert note.id not in rows and len(rows) == 5
    assert rows[src.id] == ["-", "-", src.id, "input-file", "Existing file", "-", f"file={model}"]
    assert rows[build.id] == ["01", "01_amber-build", build.id, "amber-build", "Amber · system build", "-",
                              f"structure={model}"]
    assert rows[md.id][:2] == ["03", "03_amber-md"] and rows[analysis.id][:2] == ["-", "-"]
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
        ("error", "folder tag 'model 12': use letters, digits and '-' only (e.g. 'model-12'); it is added to "
                  "the step's folder name")]
    a.folder_tag = " model-12 "  # spaces around it are not part of it
    assert errors(Plan(wf)) == [] and Plan(wf).steps[a.id].ctx.folder == "02_custom_model-12"
    # two steps, one folder
    b = add_node(wf, "user-script", step=2, settings={"script": "true\n"})
    b.folder_tag = "model-12"
    pair = "02 · Custom script and 02 · Custom script would write into"
    assert errors(Plan(wf)) == [(b.id, f"{pair} the same folder 02_custom_model-12/ with the same names: "
                                       "give one of them another folder tag or step number")]
    b.folder_tag = "Model-12"  # one folder where case does not count
    assert errors(Plan(wf)) == [(b.id, f"{pair} 02_custom_Model-12/ and 02_custom_model-12/, one folder on a "
                                       "file system that ignores case (macOS, Windows): give one of them "
                                       "another folder tag")]
    b.folder_tag = "model-13"
    assert errors(Plan(wf)) == []
    b.folder_tag = ""
    assert errors(Plan(wf)) == []  # 02_custom and 02_custom_model-12
    # untagged steps: the message of before
    a.folder_tag = ""
    before = f"{pair} the same folder with the same names: give one of them another step number"
    assert errors(Plan(wf)) == [(b.id, before)]
    # without step folders a tag names no folder: the steps still share the plan folder and the file names
    wf.step_folders = False
    a.folder_tag, b.folder_tag = "model-12", "model-13"
    plan = Plan(wf)
    assert errors(plan) == [(b.id, before)]
    assert [i.message for i in plan.steps[a.id].issues if "folder tag" in i.message] == [
        "folder tag 'model-12': the plan writes no step folders, so the tag names no folder (MANIFEST lists "
        "it)"]
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
        "node": md.id, "tool": "amber-md", "title": "MD model 12", "label": "03 · MD model 12",
        "folder_tag": "model12", "stamp": STAMP}
    text = open(os.path.join(root, "03_amber-md_model12", IDENTITY)).read()
    assert text.startswith("# This folder holds the files of step 03 · MD model 12 of plan apt1, written by")
    assert "# The step's id in the plan; it stays when steps are added or renumbered.\nnode = " in text
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
                                                "a title with breaks", "-", "-"]


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
    assert [(f["folder"], f["step"], f["matched_by"]) for f in md12["found"]] == [
        ("03_amber-md_model12", 3, "node id")]
    # model 13's MD (now 12): found in 06_amber-md/ by its node id although no name says model 13
    md13 = steps[plain[2].id]
    assert md13["state"] == "absent" and [f["folder"] for f in md13["found"]] == ["06_amber-md"]
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
    assert "not there; its files are in 03_amber-md_model12/ (step 03, by node id)" in line
    line = next(ln for ln in text.splitlines() if ln.startswith("06 "))
    assert "holds another step's files: 06_amber-md/ (step 06, node " in line and "MD model 13" in line
    assert "3 step(s) would write into a folder that holds another step's files" in text
    # --table: the MANIFEST a write would leave, renumbered
    capsys.readouterr()
    assert main(["workflow", "manifest", "-p", proj, "--table"]) == 0
    table = capsys.readouterr().out
    assert table == manifest_text(Plan(Project.load(proj).workflow, root))
    assert f"09\t09_amber-md_model12\t{a[2].id}\tamber-md\tMD model 12\tmodel12\t" in table


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
    found = {s.node_id: [(f.name, how) for f, how in s.found] for s in result.steps}
    assert found["g" + a[0].id] == [("01_amber-build_model12", "folder tag")]
    assert found["g" + b[2].id] == [("06_amber-md", "title")]
    assert result.others == [] and result.conflicts == []
    # from the names alone (no identity files): the folder tag in a folder's name
    for path in glob.glob(os.path.join(root, "*", IDENTITY)):
        os.remove(path)
    result = match(Plan(wf2, root))
    found = {s.node_id: [(f.name, how) for f, how in s.found] for s in result.steps}
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
# project files
# ---------------------------------------------------------------------------


def test_project_files_without_folder_tags_load_and_save_as_before(tmp_path):
    paths = sorted(glob.glob(os.path.join(EXAMPLES, "*.mdatk.toml")))
    assert paths
    for k, path in enumerate(paths):
        project = Project.load(path)
        assert all(n.folder_tag == "" for n in project.workflow.nodes)
        out = str(tmp_path / f"{k}.mdatk.toml")
        project.save(out)
        text = open(out).read()
        assert "folder_tag" not in text
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
