"""Custom script steps (``user-script``): ports, the run script, checks, real runs and project files."""

import json
import os
import shlex
import shutil
import stat
import subprocess

import pytest

from mdatk import tomlio
from mdatk.citations import step_citations
from mdatk.config import PlanNode, Project, WorkflowParams
from mdatk.workflows import TOOLS, runstate, tools_by_category
from mdatk.workflows.plan import Plan, add_node, apply_template, connect, generate_step, single_step_plan
from mdatk.workflows.runner import ScriptFailed, plan_jobs, run_jobs, run_script

STAMP = "20261015-1432"
NEEDS_BASH = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")
NEEDS_PYTHON3 = pytest.mark.skipif(shutil.which("python3") is None, reason="needs python3 on PATH")


def step(settings=None, files=None, step_no=2, root="/tmp/mdatk-plan", **wf_kw):
    """Resolve a single custom-script step; returns (plan, resolved step)."""
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, **wf_kw)
    node = PlanNode(id="x", tool="user-script", step=step_no,
                    settings=TOOLS["user-script"].normalise(settings or {}), files=dict(files or {}))
    plan = single_step_plan(wf, node, root)
    return plan, plan.steps["x"]


def messages(st, level=None):
    return [i.message for i in st.issues if level is None or i.level == level]


def files_of(plan, nid="x"):
    return {f.name: f for f in generate_step(plan.steps[nid])}


def run_sh(plan, nid="x"):
    return [f for f in files_of(plan, nid).values() if f.role == "run"][0].content


# ---------------------------------------------------------------------------
# registration, ports, planning
# ---------------------------------------------------------------------------


def test_registered_in_its_own_category_without_citations():
    tool = TOOLS["user-script"]
    assert tool.label == "Custom script" and tool.generates and tool.resources(tool.defaults()) == []
    assert ("Custom", [tool]) in tools_by_category()
    s = tool.defaults()
    assert s["inputs"] == [] and s["outputs"] == [] and s["interpreter"] == "bash" and s["source"] == "inline"
    assert tool.input_ports(s) == [] and tool.output_ports(s) == []
    plan, st = step({"script": "echo hi"})
    assert step_citations(st) == [] and "Please cite" not in run_sh(plan)


def test_declared_ports_and_any_type():
    tool = TOOLS["user-script"]
    s = tool.normalise({"inputs": [{"name": "model", "kind": "pdb"},
                                   {"name": "anything", "kind": "any", "optional": True, "multiple": True}],
                        "outputs": [{"name": "fixed", "kind": "pdb"}, {"name": "table", "kind": "text"}]})
    ins = {p.name: p for p in tool.input_ports(s)}
    assert ins["model"].kinds == ("pdb",) and not ins["model"].optional
    assert ins["anything"].optional and ins["anything"].multiple
    assert all(ins["anything"].accepts(k) for k in ("pdb", "parm7", "nc", "map", "text", "image"))
    assert [(p.name, p.kind) for p in tool.output_ports(s)] == [("fixed", "pdb"), ("table", "text")]
    wf = WorkflowParams()
    b = add_node(wf, "amber-build")
    u = add_node(wf, "user-script", settings=s)
    connect(wf, b.id, "topology", u.id, "anything")  # any type: a topology connects
    with pytest.raises(ValueError, match="accepts"):
        connect(wf, b.id, "topology", u.id, "model")  # typed input: only a PDB
    md = add_node(wf, "amber-md")
    with pytest.raises(ValueError, match="accepts"):
        connect(wf, u.id, "table", md.id, "topology")
    connect(wf, u.id, "fixed", add_node(wf, "vmd").id, "topology")


def test_builder_to_custom_script_to_amber_build(tmp_path):
    """The custom step reads the builder's helix and writes the PDB that the Amber build reads."""
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    b = add_node(wf, "structure-builder", settings={"sequence": "GGCAUCGAUGCC"})
    u = add_node(wf, "user-script", settings={
        "source": "inline", "interpreter": "python3",
        "script": "import os, shutil\n"
                  "shutil.copy(os.environ['MDATK_IN_STRUCTURE'], os.environ['MDATK_OUT_FIXED'])\n",
        "arguments": "--model {in:structure} -o {out:fixed}",
        "inputs": [{"name": "structure", "kind": "pdb"}],
        "outputs": [{"name": "fixed", "kind": "pdb", "file": "fixed.pdb"}]})
    a = add_node(wf, "amber-build")
    connect(wf, b.id, "pdb", u.id, "structure")
    connect(wf, u.id, "fixed", a.id, "structure")
    plan = Plan(wf, str(tmp_path / "plan"))
    errors = [(plan.steps[n].label, i.message) for n, i in plan.all_issues() if i.level == "error"]
    assert errors == []
    assert plan.order == [b.id, u.id, a.id]
    assert plan.steps[u.id].outputs["fixed"].paths == ["02_custom/fixed.pdb"]
    amber_in = plan.steps[a.id].ctx.value("structure")
    assert amber_in.paths == ["02_custom/fixed.pdb"] and amber_in.source == u.id
    report = plan.write()
    folder = tmp_path / "plan" / "02_custom"
    run = folder / f"apt1_02_custom_{STAMP}.sh"
    copy = folder / f"apt1_02_script_{STAMP}.py"
    assert run.exists() and copy.exists() and os.access(run, os.X_OK)
    sh = run.read_text()
    assert f"export MDATK_IN_STRUCTURE=../01_build/apt1_01_helix_{STAMP}.pdb" in sh
    assert (f"\npython3 apt1_02_script_{STAMP}.py --model ../01_build/apt1_01_helix_{STAMP}.pdb "
            "-o fixed.pdb\n") in sh
    build_sh = (tmp_path / "plan" / "03_amber-build" / f"apt1_03_amber-build_{STAMP}.sh").read_text()
    assert "../02_custom/fixed.pdb" in build_sh
    master = open(report.master).read()
    order = [ln for ln in master.splitlines() if ln.startswith("bash ")]
    assert order == [f'bash "01_build/apt1_01_build_{STAMP}.sh"',
                     f'bash "02_custom/apt1_02_custom_{STAMP}.sh"',
                     f'bash "03_amber-build/apt1_03_amber-build_{STAMP}.sh"']
    summary = open(report.summary).read()
    assert "## 02 · Custom script" in summary and "Type: Custom script" in summary
    assert f"Runs: `python3 apt1_02_script_{STAMP}.py --model ../01_build/" in summary
    assert "- fixed: `02_custom/fixed.pdb`" in summary and "from 02 · Custom script" in summary
    proc = subprocess.run(["bash", "-n", str(run)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    jobs = plan_jobs(plan)
    assert [j.node_id for j in jobs] == [b.id, u.id, a.id] and jobs[2].after == [b.id, u.id]


# ---------------------------------------------------------------------------
# the run script
# ---------------------------------------------------------------------------


def test_run_script_exports_placeholders_and_output_checks():
    plan, st = step({
        "script": 'echo "$@"\n',
        "arguments": "-i {in:model} -t {in:traj} -x {in:extra} -o {out:result} --log {out:log} ${HOME:-x} "
                     "'{print $1}' {script} ${script}",
        "inputs": [{"name": "model", "kind": "pdb"}, {"name": "traj", "kind": "nc", "multiple": True},
                   {"name": "extra", "kind": "any", "optional": True}],
        "outputs": [{"name": "result", "kind": "pdb", "file": "result.pdb"}, {"name": "log", "kind": "text"}],
        "conda_env": "myenv", "env_setup": "module load R/4.3"},
        files={"model": "/data/model 1.pdb", "traj": "/data/a.nc;/data/b.nc"})
    assert messages(st, "error") == []
    files = files_of(plan)
    copy, run = files[f"apt1_02_script_{STAMP}.sh"], files[f"apt1_02_custom_{STAMP}.sh"]
    assert copy.role == "helper" and copy.content == 'echo "$@"\n'
    assert run.role == "run" and run.executable
    sh = run.content
    assert sh.startswith("#!/bin/bash\n# Custom script\n")
    assert "conda activate myenv\nmodule load R/4.3" in sh
    assert f"export MDATK_PROJECT=apt1 MDATK_STEP=02 MDATK_STAMP={STAMP}" in sh
    assert "export MDATK_IN_MODEL='/data/model 1.pdb'" in sh
    assert "export MDATK_IN_TRAJ='/data/a.nc\n/data/b.nc'" in sh  # one path per line
    assert "export MDATK_IN_EXTRA=''" in sh  # optional, not connected
    assert "export MDATK_OUT_RESULT=result.pdb" in sh
    assert f"export MDATK_OUT_LOG=apt1_02_log_{STAMP}.txt" in sh
    line = [ln for ln in sh.splitlines() if ln.startswith("bash apt1_02_script")][0]
    assert line == (f"bash apt1_02_script_{STAMP}.sh -i '/data/model 1.pdb' -t /data/a.nc /data/b.nc -x '' "
                    f"-o result.pdb --log apt1_02_log_{STAMP}.txt ${{HOME:-x}} '{{print $1}}' "
                    f"apt1_02_script_{STAMP}.sh ${{script}}")  # shell variables stay as they are
    assert sh.index(line) < sh.index('[ -s "result.pdb" ]')
    assert (f'[ -s "result.pdb" ] || {{ echo "ERROR: apt1_02_script_{STAMP}.sh did not write result.pdb" '
            ">&2; exit 1; }") in sh
    assert f'[ -s "apt1_02_log_{STAMP}.txt" ]' in sh
    assert st.outputs["result"].paths == ["02_custom/result.pdb"] and st.outputs["result"].kind == "pdb"
    assert st.outputs["log"].paths == [f"02_custom/apt1_02_log_{STAMP}.txt"]


def test_interpreters_custom_command_and_script_file(tmp_path):
    src = tmp_path / "my tools" / "fix names.py"
    src.parent.mkdir()
    src.write_text("#!/usr/bin/env python3\r\nprint('fixed')\r\n")
    plan, st = step({"source": "file", "script_file": str(src), "interpreter": "python3"})
    assert messages(st, "error") == [] and messages(st, "warning") == []
    files = files_of(plan)
    assert files["fix_names.py"].content == "#!/usr/bin/env python3\nprint('fixed')\n"  # Windows line ends go
    sh = files[f"apt1_02_custom_{STAMP}.sh"].content
    assert "\npython3 fix_names.py\n" in sh and f"(a copy of {src})" in sh
    assert TOOLS["user-script"].describe(st.settings) == "python3: fix names.py"
    # another program, with and without {script}; a chosen name for the copy
    plan, st = step({"script": "load {x}\n", "interpreter": "custom", "script_name": "view.pml",
                     "command": "pymol -cq {script} --", "arguments": "{in:model}",
                     "inputs": [{"name": "model", "kind": "pdb"}]}, files={"model": "/m.pdb"})
    assert messages(st, "error") == []
    files = files_of(plan)
    assert "view.pml" in files
    assert "\npymol -cq view.pml -- /m.pdb\n" in files[f"apt1_02_custom_{STAMP}.sh"].content
    plan, st = step({"script": "x <- 1\n", "interpreter": "custom", "command": "Rscript --vanilla"})
    sh = run_sh(plan)
    assert f"\nRscript --vanilla apt1_02_script_{STAMP}.txt\n" in sh
    assert TOOLS["user-script"].describe(st.settings) == "Rscript: typed script, 1 line"
    # no step folders: everything in the plan folder
    plan, st = step({"script": "true", "outputs": [{"name": "r", "kind": "text", "file": "r.txt"}]},
                    step_folders=False)
    assert st.outputs["r"].paths == ["r.txt"]


# ---------------------------------------------------------------------------
# planner checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("settings, key, expected", [
    ({"script": "  \n"}, "script", "the script is empty"),
    ({"source": "file"}, "script_file", "choose the script file"),
    ({"source": "file", "script_file": "/no/such/dir/x.sh"}, "script_file", "script file not found"),
    ({"script": "true", "arguments": "{in:model}"}, "arguments",
     "{in:model} names no declared input (declared: none)"),
    ({"script": "true", "arguments": "-o {out:reslt}", "outputs": [{"name": "result", "kind": "pdb"}]},
     "arguments", "{out:reslt} names no declared output (declared: result)"),
    ({"script": "true", "arguments": "{input:model}", "inputs": [{"name": "model", "kind": "pdb"}]},
     "arguments", "unknown placeholder {input:model}; write {in:model}"),
    ({"script": "true", "interpreter": "custom", "command": "vmd -e {script} -args {in:map}"}, "command",
     "{in:map} names no declared input"),
    ({"script": "true", "inputs": [{"name": "model", "kind": "pdb"}, {"name": "Model", "kind": "any"}]},
     "inputs.1.name", "two inputs are named 'model' / 'Model' (both would be $MDATK_IN_MODEL)"),
    ({"script": "true", "outputs": [{"name": "a-b", "kind": "pdb"}, {"name": "a_b", "kind": "pdb"}]},
     "outputs.1.name", "two outputs are named 'a-b' / 'a_b'"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "pdb"}, {"name": "r", "kind": "text"}]},
     "outputs.1.name", "two outputs are named 'r' (both would be $MDATK_OUT_R)"),
    ({"script": "true", "inputs": [{"name": "1st", "kind": "pdb"}]}, "inputs.0.name",
     "input name '1st': use letters"),
    ({"script": "true", "inputs": [{"name": "", "kind": "pdb"}]}, "inputs.0.name", "input 1 has no name"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "pdb", "file": "out/r.pdb"}]}, "outputs.0.file",
     "output 'r': 'out/r.pdb' is not a plain file name"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "pdb", "file": "my model.pdb"}]}, "outputs.0.file",
     "'my model.pdb' is not a plain file name"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "pdb", "file": "../r.pdb"}]}, "outputs.0.file",
     "'../r.pdb' is not a plain file name"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "pdb", "file": "$HOME.pdb"}]}, "outputs.0.file",
     "is not a plain file name"),
    ({"script": "true", "outputs": [{"name": "a", "kind": "pdb", "file": "x.pdb"},
                                    {"name": "b", "kind": "pdb", "file": "x.pdb"}]},
     "outputs.1.file", "output 'b': x.pdb is also output 'a's file"),
    ({"script": "true", "script_name": "s.sh", "outputs": [{"name": "a", "kind": "text", "file": "s.sh"}]},
     "outputs.0.file", "s.sh is also the script's copy"),
    ({"script": "true", "script_name": f"apt1_02_custom_{STAMP}.sh"}, "script_name",
     "is the name of the step's run script"),
    ({"script": "true", "script_name": "sub/s.sh"}, "script_name", "'sub/s.sh' is not a plain file name"),
    ({"script": "true", "interpreter": "custom", "command": " "}, "command",
     "give the command that runs the script"),
    # bash reads a name after its own that starts with '+' as options ('bash +go.sh' fails)
    ({"script": "true", "script_name": "+go.sh"}, "script_name", "'+go.sh' is not a plain file name"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "text", "file": "+r.txt"}]}, "outputs.0.file",
     "'+r.txt' is not a plain file name: use letters, digits, '.', '_', '-' and '+', starting with"),
    ({"script": "true", "script_name": f"apt1_02_custom_{STAMP}.done"}, "script_name",
     f"apt1_02_custom_{STAMP}.done is the name of the file that marks a finished run"),
    ({"script": "true", "outputs": [{"name": "r", "kind": "text", "file": f"apt1_02_custom_{STAMP}.done"}]},
     "outputs.0.file", "is also the file that marks a finished run"),
])
def test_planner_errors(settings, key, expected):
    _plan, st = step(settings)
    hits = [i for i in st.issues if i.level == "error" and expected in i.message]
    assert hits, messages(st)
    assert key in hits[0].keys


def test_planner_checks_files_warnings_and_infos(tmp_path):
    empty = tmp_path / "empty.sh"
    empty.write_text("\n\n")
    _plan, st = step({"source": "file", "script_file": str(empty)})
    assert any(m.startswith("the script file is empty") for m in messages(st, "error"))
    binary = tmp_path / "prog"
    binary.write_bytes(b"\x7fELF\x00\x01\xff\xfe")
    _plan, st = step({"source": "file", "script_file": str(binary)})
    assert any("not a UTF-8 text file" in m for m in messages(st, "error"))
    _plan, st = step({"source": "file", "script_file": str(tmp_path)})
    assert any(m.startswith("not a file") for m in messages(st, "error"))
    # a python script run with bash, an output whose name does not fit its type, no outputs at all
    _plan, st = step({"script": "#!/usr/bin/env python3\nprint(1)\n",
                      "outputs": [{"name": "r", "kind": "pdb", "file": "r.txt"}]})
    assert messages(st, "error") == []
    assert any("starts with '#!/usr/bin/env python3' but runs with bash" in m
               for m in messages(st, "warning"))
    assert any("r.txt does not end in .pdb" in m for m in messages(st, "info"))
    _plan, st = step({"script": "#!/bin/bash\necho\n", "interpreter": "python3"})
    assert any("runs with python3" in m for m in messages(st, "warning"))
    assert any("no output files declared" in m for m in messages(st, "info"))
    # a required input without a file; shell ${var:-default} is not a placeholder
    _plan, st = step({"script": "true", "arguments": "${OUT:-x} {in:model}",
                      "inputs": [{"name": "model", "kind": "pdb"}]})
    assert messages(st, "error") == ["model: connect a step that produces it or choose a file"]


def test_wrong_records_do_not_break_the_ports():
    tool = TOOLS["user-script"]
    s = tool.normalise({"inputs": [{"name": "a"}, {"name": "A"}, {"name": "bad name"}, {"name": "b"}],
                        "outputs": [{"name": "r", "file": "x/y"}]})
    assert [p.name for p in tool.input_ports(s)] == ["a", "b"]  # duplicates and bad names left out
    assert [p.name for p in tool.output_ports(s)] == ["r"]
    plan, st = step(s)
    assert "r" not in st.outputs and st.ok is False
    assert "Runs: `bash" in plan.summary_text()  # a step with errors still gets a summary


# ---------------------------------------------------------------------------
# real runs (bash and python3 only, no other programs)
# ---------------------------------------------------------------------------

BASH_SCRIPT = """\
set -e
# the input arrives as $1 and as $MDATK_IN_DATA; the output name as $2 and $MDATK_OUT_UPPER
[ "$1" = "$MDATK_IN_DATA" ] && [ "$2" = "$MDATK_OUT_UPPER" ]
tr a-z A-Z < "$1" > "$2"
echo "step $MDATK_STEP of $MDATK_PROJECT wrote $2"
"""
PYTHON_SCRIPT = """\
import os, sys
text = open(os.environ["MDATK_IN_TEXT"]).read()
with open(os.environ["MDATK_OUT_COUNT"], "w") as fh:
    fh.write(f"{len(text.split())} words, args {sys.argv[1:]}\\n")
print("counted in", os.getcwd())
"""


def _two_step_plan(tmp_path, python_script=PYTHON_SCRIPT):
    """An existing text file -> bash step (steps 01) -> python3 step (02); files written."""
    data = tmp_path / "data.txt"
    data.write_text("hello custom steps\n")
    wf = WorkflowParams(project_name="run", stamp=STAMP)
    inp = add_node(wf, "input-file", settings={"path": str(data), "kind": "text"})
    sh = add_node(wf, "user-script", settings={
        "script": BASH_SCRIPT, "arguments": "{in:data} {out:upper}",
        "inputs": [{"name": "data", "kind": "any"}],
        "outputs": [{"name": "upper", "kind": "text", "file": "upper.txt"}]})
    py = add_node(wf, "user-script", settings={
        "script": python_script, "interpreter": "python3", "arguments": "--flag {in:text}",
        "inputs": [{"name": "text", "kind": "text"}], "outputs": [{"name": "count", "kind": "text"}]})
    connect(wf, inp.id, "file", sh.id, "data")
    connect(wf, sh.id, "upper", py.id, "text")
    assert (sh.step, py.step) == (1, 2)
    plan = Plan(wf, str(tmp_path / "plan"))
    assert [i for _n, i in plan.all_issues() if i.level == "error"] == []
    plan.write()
    return plan, sh, py


@NEEDS_BASH
@NEEDS_PYTHON3
def test_bash_and_python_steps_run_end_to_end(tmp_path):
    plan, sh, py = _two_step_plan(tmp_path)
    jobs = plan_jobs(plan)
    assert [j.node_id for j in jobs] == [sh.id, py.id] and sh.id in jobs[1].after
    lines = []
    results = run_jobs(jobs, None, on_line=lines.append)
    assert [r.status for r in results] == ["ok", "ok"]
    upper = tmp_path / "plan" / "01_custom" / "upper.txt"
    count = tmp_path / "plan" / "02_custom" / f"run_02_count_{STAMP}.txt"
    assert upper.read_text() == "HELLO CUSTOM STEPS\n"
    assert count.read_text() == "3 words, args ['--flag', '../01_custom/upper.txt']\n"
    assert "step 01 of run wrote upper.txt" in lines
    assert f"counted in {os.path.realpath(tmp_path / 'plan' / '02_custom')}" in lines


@NEEDS_BASH
@NEEDS_PYTHON3
def test_a_missing_output_fails_the_step_and_stops_its_dependents(tmp_path):
    plan, sh, py = _two_step_plan(tmp_path, python_script="print('forgot to write it')\n")
    folder = os.path.join(plan.root, "02_custom")
    run, copy = (os.path.join(folder, f"run_02_{tag}_{STAMP}.{ext}") for tag, ext in (("custom", "sh"),
                                                                                       ("script", "py")))
    lines = []
    with pytest.raises(ScriptFailed, match=r"^run_02_custom_\d{8}-\d{4}\.sh \(step 02\) failed: exit status 1$"):
        run_jobs(plan_jobs(plan), None, on_line=lines.append)
    assert "forgot to write it" in lines
    assert f"ERROR: run_02_script_{STAMP}.py did not write run_02_count_{STAMP}.txt" in lines
    # an empty output counts as missing; a failing script stops the step before the check, with its status
    for body, status in (('open(__import__("os").environ["MDATK_OUT_COUNT"], "w")\n', 1),
                         ("raise SystemExit(7)\n", 7)):
        with open(copy, "w") as fh:
            fh.write(body)
        code, output = run_script(run, lambda line: None)
        assert code == status
        assert ("did not write" in output) == (status == 1)


@NEEDS_BASH
def test_a_failed_output_check_stops_the_steps_that_read_it(tmp_path):
    wf = WorkflowParams(project_name="run", stamp=STAMP)
    a = add_node(wf, "user-script", settings={"script": "exit 0\n",
                                               "outputs": [{"name": "r", "kind": "text", "file": "r.txt"}]})
    b = add_node(wf, "user-script", settings={"script": "touch started\n",
                                               "inputs": [{"name": "r", "kind": "text"}]})
    connect(wf, a.id, "r", b.id, "r")
    plan = Plan(wf, str(tmp_path / "plan"))
    plan.write()
    states = []
    with pytest.raises(ScriptFailed, match="1 dependent script"):
        run_jobs(plan_jobs(plan), None, on_line=lambda line: None, on_state=lambda j, s: states.append(s))
    assert "skipped" in states and not (tmp_path / "plan" / "02_custom" / "started").exists()


# ---------------------------------------------------------------------------
# earlier runs: stale and partial outputs, fixed names, shared names (review of 0.0.10)
# ---------------------------------------------------------------------------

OLD, NEW = STAMP, "20261101-0900"


def _quiet_run(plan):
    return [r.status for r in run_jobs(plan_jobs(plan), None, on_line=lambda line: None)]


def _chain(tmp_path, script, outputs=None, stamp=STAMP):
    """Step 01 runs ``script`` (output t) and step 02 copies t to c.txt; returns (workflow, a, b, root)."""
    wf = WorkflowParams(project_name="run", stamp=stamp)
    a = add_node(wf, "user-script", settings={"script": script,
                                               "outputs": outputs or [{"name": "t", "kind": "text"}]})
    b = add_node(wf, "user-script", settings={"script": 'cp "$MDATK_IN_T" "$MDATK_OUT_C"\n',
                                               "inputs": [{"name": "t", "kind": "text"}],
                                               "outputs": [{"name": "c", "kind": "text", "file": "c.txt"}]})
    connect(wf, a.id, "t", b.id, "t")
    return wf, a, b, str(tmp_path / "plan")


def _written(wf, root):
    plan = Plan(wf, root)
    assert [i.message for _n, i in plan.all_issues() if i.level == "error"] == []
    plan.write()
    return plan


@NEEDS_BASH
def test_an_output_left_by_an_earlier_run_does_not_pass(tmp_path):
    wf, a, _b, root = _chain(tmp_path, 'echo "run 1" > "$MDATK_OUT_T"\n')
    assert _quiet_run(_written(wf, root)) == ["ok", "ok"]
    table = os.path.join(root, "01_custom", f"run_01_t_{STAMP}.txt")
    assert open(table).read() == "run 1\n"
    # the script no longer writes its output (a bug) but exits 0: run 1's file must not pass the check
    a.settings["script"] = 'echo "nothing to write"\n'
    plan = _written(wf, root)
    sh = run_sh(plan, a.id)
    remove = f"\nrm -f -- run_01_custom_{STAMP}.done run_01_t_{STAMP}.txt\n"
    assert sh.index(remove) < sh.index("\nbash run_01_")
    lines = []
    with pytest.raises(ScriptFailed, match="exit status 1; 1 dependent script"):
        run_jobs(plan_jobs(plan), None, on_line=lines.append)
    assert f"ERROR: run_01_script_{STAMP}.sh did not write run_01_t_{STAMP}.txt" in lines
    assert not os.path.exists(table)


@NEEDS_BASH
def test_only_a_run_that_passed_its_checks_counts_as_done(tmp_path):
    wf, a, b, root = _chain(tmp_path, 'echo partial > "$MDATK_OUT_T"\necho crash >&2; exit 3\n')
    plan = _written(wf, root)
    with pytest.raises(ScriptFailed, match="exit status 3"):
        _quiet_run(plan)
    folder = os.path.join(root, "01_custom")
    done = os.path.join(folder, f"run_01_custom_{STAMP}.done")
    assert open(os.path.join(folder, f"run_01_t_{STAMP}.txt")).read() == "partial\n"
    assert not os.path.exists(done)
    runs = runstate.check_runs(plan)
    assert runs[a.id].state == "partial" and runs[a.id].missing == [f"01_custom/run_01_custom_{STAMP}.done"]
    assert runstate.adopt(plan, runs) == []
    # a run that passed every check writes the marker last
    a.settings["script"] = 'echo whole > "$MDATK_OUT_T"\n'
    plan = _written(wf, root)
    sh = run_sh(plan, a.id)
    mark = ("\necho \"MDATK: custom script finished $(date '+%Y-%m-%d %H:%M:%S')\" > "
            f"run_01_custom_{STAMP}.done\n")
    assert sh.index("\nbash run_01_script") < sh.index('\n[ -s "run_01_t_') < sh.index(mark)
    assert sh.endswith(mark)  # the last line
    assert _quiet_run(plan) == ["ok", "ok"]
    assert open(done).read().startswith("MDATK: custom script finished 20")
    assert TOOLS["user-script"].finish_marks(plan.steps[a.id].settings, plan.steps[a.id].ctx) == \
        [(f"01_custom/run_01_custom_{STAMP}.done", r"MDATK:\ custom\ script\ finished")]
    runs = runstate.check_runs(plan)
    assert (runs[a.id].state, runs[b.id].state) == ("done", "done")
    assert runs[a.id].finished == os.path.getmtime(done)
    # killed while writing the marker: not finished
    with open(done, "w") as fh:
        fh.write("MDATK: cust")
    assert runstate.check_runs(plan)[a.id].state == "partial"
    # a step without outputs is done once its script ran
    wf2 = WorkflowParams(project_name="run", stamp=STAMP)
    c = add_node(wf2, "user-script", settings={"script": "true\n"})
    plan = _written(wf2, str(tmp_path / "plan2"))
    assert runstate.check_runs(plan)[c.id].state == "missing"
    assert _quiet_run(plan) == ["ok"]
    assert runstate.check_runs(plan)[c.id].state == "done"


@NEEDS_BASH
def test_a_fixed_output_name_belongs_to_the_stamp_of_its_run(tmp_path):
    fixed = [{"name": "t", "kind": "text", "file": "result.txt"}]
    wf, a, b, root = _chain(tmp_path, 'echo "$MDATK_STAMP" > "$MDATK_OUT_T"\n', fixed, OLD)
    assert _quiet_run(_written(wf, root)) == ["ok", "ok"]
    wf.stamp = NEW  # 'New stamp' for the next iteration; nothing written or run yet
    plan = Plan(wf, root)
    run = runstate.check_runs(plan)[a.id]
    assert (run.state, run.stamp, run.current) == ("done", OLD, False)
    assert runstate.run_status(plan, a.id, NEW).state == "partial"  # result.txt, but no marker of NEW
    # NEW's run fails after writing part of result.txt: that is no run of OLD either
    a.settings["script"] = 'echo partial > "$MDATK_OUT_T"\nexit 3\n'
    plan = _written(wf, root)
    assert "\nrm -f -- run_01_custom.done run_01_custom_????????-????.done\n" in run_sh(plan, a.id)
    with pytest.raises(ScriptFailed):
        _quiet_run(plan)
    assert runstate.check_runs(plan)[a.id].state == "partial"
    assert runstate.run_status(plan, a.id, OLD).state == "partial"
    # a complete run of NEW; step 02 has automatic (stamped) names, so its run of OLD stays complete
    a.settings["script"] = 'echo "$MDATK_STAMP" > "$MDATK_OUT_T"\n'
    b.settings["outputs"] = [{"name": "c", "kind": "text"}]
    plan = _written(wf, root)
    assert "????????-????" not in run_sh(plan, b.id)
    assert _quiet_run(plan) == ["ok", "ok"]
    run = runstate.check_runs(plan)[a.id]
    assert (run.state, run.stamp, run.current) == ("done", NEW, True)
    assert sorted(f for f in os.listdir(os.path.join(root, "01_custom")) if f.endswith(".done")) == \
        [f"run_01_custom_{NEW}.done"]
    assert sorted(f for f in os.listdir(os.path.join(root, "02_custom")) if f.endswith(".done")) == \
        [f"run_02_custom_{OLD}.done", f"run_02_custom_{NEW}.done"]


@NEEDS_BASH
def test_two_steps_must_not_write_the_same_file(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, step_folders=False)
    out = [{"name": "r", "kind": "pdb", "file": "result.pdb"}]
    a = add_node(wf, "user-script", title="model A", settings={"script": 'echo A > "$MDATK_OUT_R"\n',
                                                                "outputs": out})
    b = add_node(wf, "user-script", title="model B", settings={"script": 'echo B > "$MDATK_OUT_R"\n',
                                                                "outputs": out})
    c = add_node(wf, "user-script", title="reads A", settings={
        "script": 'cp "$MDATK_IN_M" "$MDATK_OUT_O"\n', "inputs": [{"name": "m", "kind": "pdb"}],
        "outputs": [{"name": "o", "kind": "text", "file": "o.txt"}]})
    connect(wf, a.id, "r", c.id, "m")
    root = str(tmp_path / "plan")
    plan = Plan(wf, root)
    assert [(n, i.message) for n, i in plan.all_issues() if i.level == "error"] == [
        (b.id, "output 'r': result.pdb is also written by 01 · model A; give one of them another file name")]
    assert b.id in plan.write().skipped
    assert _quiet_run(plan) == ["ok", "ok"]
    assert open(os.path.join(root, "o.txt")).read() == "A\n"
    # in step folders the same name is two files; one folder for both is reported once, as before
    wf.step_folders = True
    assert [i for _n, i in Plan(wf, root).all_issues() if i.level == "error"] == []
    b.step = a.step
    assert [i.message for _n, i in Plan(wf, root).all_issues() if i.level == "error"] == [
        "01 · model B and 01 · model A would write into the same folder with the same names: give one of "
        "them another step number"]
    # built-in steps: files passed on from an input or chosen by hand are not written by the step
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, step_folders=False)
    for _k in range(2):
        add_node(wf, "farfar2", settings={"make_inputs": False, "fasta_file": "x.fasta",
                                          "secstruct_file": "x.secstruct"})
    apply_template(wf, "amber")
    for node in apply_template(wf, "amber"):
        node.step += 10
    assert not [i for _n, i in Plan(wf, root).all_issues() if "also written by" in i.message]


def test_an_output_must_not_be_one_of_the_steps_inputs(tmp_path):
    root = tmp_path / "plan"
    model = root / "02_custom" / "model.pdb"  # a hand-picked file in the step's own folder
    settings = {"script": "true", "inputs": [{"name": "m", "kind": "pdb"}],
                "outputs": [{"name": "r", "kind": "pdb", "file": "model.pdb"}]}
    _plan, st = step(settings, files={"m": str(model)}, root=str(root))
    hits = [i for i in st.issues if i.level == "error" and "model.pdb is also an input file" in i.message]
    assert hits and hits[0].keys == ("outputs.0.file",)
    settings["outputs"][0]["file"] = "fixed.pdb"
    _plan, st = step(settings, files={"m": str(model)}, root=str(root))
    assert messages(st, "error") == []


def test_names_that_bash_would_read_as_options(tmp_path):
    src = tmp_path / "+fix.sh"
    src.write_text("echo fixed\n")
    plan, st = step({"source": "file", "script_file": str(src)})
    assert messages(st, "error") == [] and "fix.sh" in files_of(plan)
    assert "\nbash fix.sh\n" in run_sh(plan)


@NEEDS_PYTHON3
def test_an_optional_input_for_several_files_without_any_gives_no_word(tmp_path):
    settings = {"interpreter": "python3", "arguments": "{out:args} {in:extra}",
                "script": 'import sys\nopen(sys.argv[1], "w").write(repr(sys.argv[2:]))\n',
                "inputs": [{"name": "extra", "kind": "any", "optional": True, "multiple": True}],
                "outputs": [{"name": "args", "kind": "text", "file": "args.txt"}]}

    def words(plan):
        return shlex.split([ln for ln in run_sh(plan).splitlines() if ln.startswith("python3")][0])[2:]

    plan, st = step(settings, root=str(tmp_path / "plan"))
    assert messages(st, "error") == [] and words(plan) == ["args.txt"]
    plan.write()
    code, _output = run_script(os.path.join(plan.root, "02_custom", f"apt1_02_custom_{STAMP}.sh"),
                               lambda line: None)
    assert code == 0 and (tmp_path / "plan" / "02_custom" / "args.txt").read_text() == "[]"
    plan, _st = step(settings, files={"extra": "/d/a.txt;/d/b c.txt"})
    assert words(plan) == ["args.txt", "/d/a.txt", "/d/b c.txt"]
    # one optional file keeps its place as an empty word
    settings["inputs"][0]["multiple"] = False
    plan, _st = step(settings)
    assert words(plan) == ["args.txt", ""]


# ---------------------------------------------------------------------------
# project files and the command line
# ---------------------------------------------------------------------------

TRICKY = ('#!/usr/bin/env python3\n'
          '"""Doc string with "quotes", a backslash \\ and \\n escapes."""\n'
          "print(f\"{os.environ['MDATK_IN_MODEL']!r}\\t\u2192 done\")\t# unicode and a tab\n"
          "s = r'C:\\path\\' + \"'''\" + '\"\"\"'\n"
          "trailing spaces   \n"
          "\n"
          "\x01 a control character and a \r carriage return\n")


def test_project_file_round_trip(tmp_path):
    project = Project()
    project.workflow.project_name = "apt1"
    f = add_node(project.workflow, "input-file", settings={"path": "inputs/model.pdb", "kind": "pdb"})
    u = add_node(project.workflow, "user-script", title="Rename residues", settings={
        "script": TRICKY, "interpreter": "python3", "arguments": "-m {in:model} -o {out:fixed}",
        "inputs": [{"name": "model", "kind": "pdb"},
                   {"name": "extra", "kind": "any", "optional": True, "multiple": True}],
        "outputs": [{"name": "fixed", "kind": "pdb", "file": "fixed.pdb"}], "env_setup": "module load X"})
    add_node(project.workflow, "user-script", settings={"source": "file", "script_file": "scripts/go.sh"})
    connect(project.workflow, f.id, "file", u.id, "model")
    connect(project.workflow, u.id, "fixed", add_node(project.workflow, "amber-build").id, "structure")
    path = tmp_path / "plan.mdatk.toml"
    project.save(str(path))
    text = path.read_text()
    assert "# Script text: The script." in text and "# Run with: The program that runs the script." in text
    assert "[[workflow.nodes.settings.inputs]]" in text and "\n[[workflow.nodes.settings.outputs]]\n" in text
    data = tomlio.load(str(path))
    assert data["workflow"]["nodes"][1]["settings"]["script"] == TRICKY
    loaded = Project.load(str(path))
    lu = loaded.workflow.nodes[1]
    assert lu.tool == "user-script" and lu.title == "Rename residues"
    assert lu.settings == TOOLS["user-script"].normalise(u.settings)
    assert loaded.workflow.nodes[2].settings["script_file"] == str(tmp_path / "scripts" / "go.sh")
    assert [(e.source, e.output, e.target, e.input) for e in loaded.workflow.edges] == \
        [(e.source, e.output, e.target, e.input) for e in project.workflow.edges]
    plan = Plan(loaded.workflow, str(tmp_path / "plan"))
    assert plan.steps[lu.id].outputs["fixed"].paths == ["01_custom/fixed.pdb"]  # the file box has no number
    # projects saved before custom scripts existed load as before
    old = Project.from_dict({"workflow": {"nodes": [{"id": "n1", "tool": "amber-build"}]}})
    assert [n.tool for n in old.workflow.nodes] == ["amber-build"]


def test_cli_lists_and_generates_a_custom_step(tmp_path, capsys):
    from mdatk.cli import main
    assert main(["workflow", "tools"]) == 0
    out = capsys.readouterr().out
    assert "Custom\n  user-script" in out
    assert main(["workflow", "tools", "user-script"]) == 0
    out = capsys.readouterr().out
    assert "records: inputs.<n>.<key>" in out and "records: outputs.<n>.<key>" in out and "interpreter" in out
    model = tmp_path / "model.pdb"
    model.write_text("END\n")
    args = ["workflow", "generate", "user-script", "-o", str(tmp_path / "gen"), "--step", "5",
            "--project-name", "apt1", "--stamp", STAMP, "-i", f"model={model}",
            "--set", "script=cp \"$MDATK_IN_MODEL\" \"$MDATK_OUT_COPY\"", "--set", "arguments={in:model}",
            "--set", 'inputs=[{"name": "model", "kind": "pdb"}]',
            "--set", 'outputs=[{"name": "copy", "kind": "pdb", "file": "copy.pdb"}]', "-q"]
    assert main(args + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"===== apt1_05_custom_{STAMP}.sh" in out and f"bash apt1_05_script_{STAMP}.sh {model}" in out
    assert main(args) == 0
    res = json.loads(capsys.readouterr().out)
    run = os.path.join(res["folder"], f"apt1_05_custom_{STAMP}.sh")
    assert run in res["written"] and os.stat(run).st_mode & stat.S_IXUSR
    assert main(["workflow", "generate", "user-script", "--set", "arguments={in:nothing}", "--set",
                 "script=true", "-o", str(tmp_path / "bad")]) == 1
    assert "names no declared input" in capsys.readouterr().err


@NEEDS_BASH
def test_cli_generate_and_run(tmp_path):
    from mdatk.cli import main
    assert main(["workflow", "generate", "user-script", "-o", str(tmp_path / "gen"), "--stamp", STAMP,
                 "--project-name", "apt1", "--set", "script=echo made > \"$MDATK_OUT_R\"",
                 "--set", 'outputs=[{"name": "r", "kind": "text", "file": "r.txt"}]', "-q", "--run"]) == 0
    assert (tmp_path / "gen" / "01_custom" / "r.txt").read_text() == "made\n"
