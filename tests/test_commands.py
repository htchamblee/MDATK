"""The commands a generated script runs (mdatk.workflows.commands, the script wizard's Commands tab): what
command_lines keeps and leaves out, on scripts written for the purpose and on the real scripts of several
step types."""

import copy
import time

import pytest

from mdatk.config import PlanNode, Project, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.commands import command_lines
from mdatk.workflows.plan import Plan, add_node, connect, generate_step

STAMP = "20261015-1432"
ENV = {"conda_env": "AmberTools25", "env_setup": "module load amber/24\nsource /opt/amber24/amber.sh"}
# what no command may show: the header, the environment, checks, messages and error handling
LEFT_OUT = ("#!/bin/bash", "set -eo pipefail", 'cd "$(dirname', "conda", "module load", "source /opt",
            "[ -s", "echo ", "ERROR", "exit ", "rm -f", "mkdir", "|| {", "|| true")
CONTROL = {"if", "then", "else", "fi", "do", "done"}


def run_scripts(name, settings=None, inputs=None):
    """{file name: (role, text)} of the run scripts of step type ``name``, generated as the wizard previews
    them (ctx.preview), with its required inputs (and ``inputs``) chosen as stand-in paths."""
    tool = TOOLS[name]
    s = tool.normalise(copy.deepcopy(settings or {}))
    files = {p.name: f"/golden/inputs/{p.name}.{p.kinds[0]}" for p in tool.input_ports(s) if not p.optional}
    files.update(inputs or {})
    node = PlanNode(id="x", tool=name, step=3, settings=s, files=files)
    plan = Plan(WorkflowParams(project_name="golden", stamp=STAMP, nodes=[node]), "/plan", Project().groups)
    step = plan.steps["x"]
    assert not [i.message for i in step.issues if i.level == "error"]
    step.ctx.preview = True
    return {f.name: (f.role, f.content) for f in generate_step(step) if f.role in ("run", "run-separate")}


def commands(name, settings=None, inputs=None):
    """{role: commands} of the step type's run scripts (one script per role in these cases)."""
    out = {}
    for role, text in run_scripts(name, settings, inputs).values():
        assert role not in out
        out[role] = command_lines(text)
    return out


def assert_clean(cmds):
    for c in cmds:
        first = c.split("\n")[0]  # a here-document's lines may hold anything
        assert not any(bad in first for bad in LEFT_OUT) and not CONTROL & set(first.split()), c


# ---------------------------------------------------------------------------
# the rules, on scripts written for them
# ---------------------------------------------------------------------------


def test_header_environment_checks_and_messages_are_left_out():
    script = "\n".join([
        "#!/bin/bash", "# Title", "", "set -eo pipefail", 'cd "$(dirname "$0")"', "",
        "# Environment", 'eval "$(conda shell.bash hook)"', "conda activate amber", "module load cuda/12",
        "spack load gromacs", "export OMP_NUM_THREADS=4", "",
        "command -v tleap >/dev/null || { echo \"ERROR: tleap not found\" >&2; exit 1; }",
        "rm -f old.parm7 && mkdir -p work", "# Solvate", "tleap -f build.leap | tee build.log",
        '[ -s "sys.parm7" ] || { echo "ERROR: tleap did not write sys.parm7" >&2; exit 1; }',
        'echo "Topology: sys.parm7"', "exit 0", ""])
    assert command_lines(script) == ["tleap -f build.leap | tee build.log"]


def test_continued_lines_are_joined_but_not_after_a_comment():
    # an escaped backslash at the end of a line does not continue it
    script = "prog --a 1 \\\n    --b 'x y' \\\n    --c 3\n# a comment \\\nother --flag\nprintf x\\\\\nlast"
    assert command_lines(script) == ["prog --a 1 --b 'x y' --c 3", "other --flag", "last"]


def test_quotes_brackets_and_pipelines_spanning_lines():
    script = "\n".join(['python3 -c "', "import sys", 'print(sys.argv)"', "MODELS=(", "  a.pdb  # first",
                        "  b.pdb", ")", 'score "${MODELS[@]}" |', "  sort -k2"])
    assert command_lines(script) == ['python3 -c "\nimport sys\nprint(sys.argv)"',
                                     'score "${MODELS[@]}" | sort -k2']


def test_here_documents_stay_with_their_command():
    script = "\n".join([
        'TOP="sys.parm7"', "cpptraj -p $TOP << EOF", "parm $TOP", "trajin 'md.nc'", "run", "EOF",
        "cat > notes.txt <<'END'", "$TOP stays as written", "END",
        "tclsh <<-DONE", "\tputs $TOP", "\tDONE", "after"])
    assert command_lines(script) == [
        "cpptraj -p sys.parm7 << EOF\nparm sys.parm7\ntrajin 'md.nc'\nrun\nEOF",
        "cat > notes.txt <<'END'\n$TOP stays as written\nEND",
        "tclsh <<-DONE\n\tputs sys.parm7\nDONE", "after"]
    # the here-document of a line left out goes with it
    assert command_lines("read -r A B <<EOF\nprog 1\nEOF\nrun.sh") == ["run.sh"]


def test_control_structure_functions_and_error_handling():
    script = "\n".join([
        "N=1", "N=2", 'ENGINE="pmemd.cuda"', "FLAGS=run.flags",
        'finished() { [ -s "$1" ] && grep -qE "done|Total" "$1"; }',
        "work() {", "  local x=$1",
        '  $ENGINE -i "$x" @$FLAGS > "${x}.log" 2>&1 || { echo "ERROR: $x failed" >&2; exit 1; }', "}",
        "if finished a.out; then", '  echo "a: done"', "else", "  work a.in", "fi",
        "if ! python3 check.py --n $N; then python3 fix.py; fi",
        "for f in *.pdb; do", '  [ -e "$f" ] || continue', "  analyse '$ENGINE' \"$f\" | tee -a log.txt",
        "done",
        "while read -r A B; do", "  score $A ${FLAGS} $B", "done < pairs.txt",
        "( cd work && acedrg -i ../x.smi ) > acedrg.log 2>&1 \\", '    || { echo "ERROR" >&2; exit 1; }',
        "x3dna-dssr --cleanup > /dev/null 2>&1 || true", "server --port 8000 &", "wait"])
    assert command_lines(script) == [
        'pmemd.cuda -i "$x" @run.flags > "${x}.log" 2>&1',  # set once: shown with their values
        "python3 check.py --n $N", "python3 fix.py",  # N is set twice: as written
        "analyse '$ENGINE' \"$f\" | tee -a log.txt",  # nothing expands in single quotes
        "score $A run.flags $B",
        "( cd work && acedrg -i ../x.smi ) > acedrg.log 2>&1",
        "x3dna-dssr --cleanup > /dev/null 2>&1",
        "server --port 8000 &"]


def test_a_long_list_of_models_stays_quick():
    """The wizard reads the scripts after every change; a list of thousands of models (one per line) or a
    long continued command must not be read again for each of its lines."""
    models = "\n".join(f"  ../02_models/golden_02_model-{k:04d}_{STAMP}.pdb" for k in range(3000))
    flags = "".join(f"    --opt{k} {k} \\\n" for k in range(1000))
    script = f'MODELS=(\n{models}\n)\nscore "${{MODELS[@]}}" \\\n{flags}    -o out.tsv\n'
    start = time.perf_counter()
    cmds = command_lines(script)
    assert time.perf_counter() - start < 2.0  # about 0.1 s; reading it again per line took a minute
    assert len(cmds) == 1 and cmds[0].startswith('score "${MODELS[@]}" --opt0 0 --opt1 1 ')
    assert cmds[0].endswith("--opt999 999 -o out.tsv")


def test_nothing_to_run():
    assert command_lines("") == []
    assert command_lines("#!/bin/bash\nset -e\n# nothing\necho done\n") == []


def test_case_patterns_are_not_commands():
    # a pattern's ')' ends no subshell: '/*) MODELS+=("$f")' is a pattern and an assignment, not a program
    script = "\n".join([
        'while IFS= read -r f || [ -n "$f" ]; do',
        '  case "$f" in "") ;; /*) MODELS+=("$f") ;; *) MODELS+=("../$f") ;; esac',
        "done < list.txt",
        'case "$x" in', '  a|b) prog-a "$x" ;;', "  (c) prog-c ;;", "  d)", "    prog-d", "    ;;",
        "  *) other; another", "esac",
        "(", "  cd work", "  inner", ") > log 2>&1", "{ case $y in q) prog-q ;; esac; }", "last"])
    assert command_lines(script) == ['prog-a "$x"', "prog-c", "prog-d", "other", "another", "inner",
                                     "prog-q", "last"]


def test_steps_that_take_a_set_show_only_their_program():
    # the members of a set are read from its lists with a case statement (listed_files)
    wf = WorkflowParams(project_name="golden", stamp=STAMP)
    fold = add_node(wf, "farfar2", settings={"sequence": "GGGCAC", "runs": 2, "nstruct": 3})
    check = add_node(wf, "model-check")
    compare = add_node(wf, "ensemble-compare", settings={"register_a": "p=A:1-A:6", "register_b": "x=A:2-A:5",
                                                         "closure": "geometry"})
    for target, port in ((check, "models"), (compare, "arm_a"), (compare, "arm_b")):
        connect(wf, fold.id, "decoys", target.id, port)
    plan = Plan(wf, "/plan", Project().groups)
    for node, program in ((check, "python3 golden_02_modelcheck_"),
                          (compare, "python3 -m mdatk ensemble compare")):
        [script] = [f.content for f in generate_step(plan.steps[node.id]) if f.role == "run"]
        assert "case \"$f\" in" in script
        cmds = command_lines(script)
        assert_clean(cmds)
        assert len(cmds) == 1 and cmds[0].startswith(program), cmds


# ---------------------------------------------------------------------------
# real scripts
# ---------------------------------------------------------------------------


def test_amber_build():
    cmds = commands("amber-build", ENV)["run"]
    assert_clean(cmds)
    assert [c.split()[0] for c in cmds] == ["python3", "python3", "python3", "tleap", "parmed", "python3"]
    assert cmds[0] == f'python3 golden_03_cp-lib_{STAMP}.py "$AMBERHOME" .'  # set from the environment
    assert cmds[3] == f"tleap -f golden_03_tleap_{STAMP}.leap | tee golden_03_tleap_{STAMP}.log"
    assert cmds[5].startswith(f"python3 golden_03_check_{STAMP}.py golden_03_parmed_{STAMP}.parm7 | tee ")
    assert not any("AMBERHOME=" in c for c in cmds)


def test_amber_md_with_separate_production():
    restraints = {"restraints": "/golden/inputs/helix.dist"}
    scripts = run_scripts("amber-md", ENV, restraints)
    cmds = commands("amber-md", ENV, restraints)
    assert set(cmds) == {"run", "run-separate"}
    main, prod = cmds["run"], cmds["run-separate"]
    assert_clean(main + prod)
    pmemd = [c for c in main if c.startswith("pmemd.cuda -O -i ")]  # $PMEMD shown as what it is
    assert len(pmemd) == len(main) - 1 == 5
    assert all('-p "/golden/inputs/topology.parm7"' in c for c in pmemd)  # "$TOP" too
    assert [c.split(" -i ")[1].split()[0] for c in pmemd] == [
        f"golden_03_{s}_{STAMP}.in" for s in ("min1", "min2", "heat", "eq", "eq-free")]
    dry = main[-1].split("\n")
    assert dry[0] == "cpptraj << EOF" and dry[-1] == "EOF" and 'parm "/golden/inputs/topology.parm7"' in dry
    assert len(prod) == 1 and prod[0].startswith(f"pmemd.cuda -O -i golden_03_prod_{STAMP}.in ")
    # the scripts themselves hold what the commands leave out
    text = "\n".join(t for _r, t in scripts.values())
    assert "conda activate AmberTools25" in text and "module load amber/24" in text and "$PMEMD" in text
    mpi = commands("amber-md", {"engine": "pmemd.MPI", "nproc": 4}, restraints)["run"]
    assert mpi[0].startswith("mpirun -np 4 pmemd.MPI -O -i ")


def test_farfar2():
    cmds = commands("farfar2", {"sequence": "GGCUCUUCGGGGUC", "dotbracket": "(((((....)))))"})["run"]
    assert_clean(cmds)
    denovo = [c for c in cmds if "rna_denovo" in c]
    assert len(denovo) == 1  # the body of the run function; its runs start through xargs
    assert denovo[0].startswith(f"nice rna_denovo @../../golden_03_farfar2_{STAMP}.flags ")  # $FLAGS
    assert "${N}" in denovo[0] and "||" not in denovo[0]  # set in each run: as written
    assert any(c.startswith("seq 1 ") and "xargs -P " in c and "run_one" in c for c in cmds)
    assert not any(c.startswith(("run_one", "export", "I=", "N=")) for c in cmds)


def test_namd_mdff():
    cmds = commands("namd-mdff")["run"]
    assert_clean(cmds)
    assert not any(bad in c for c in cmds for bad in ("KEY=", "FRESH=", "cksum", "grep -q", "$NAMD"))
    assert cmds[0] == "python3 golden_03_mdff_20261015-1432.py is-netcdf /golden/inputs/coordinates.rst7"
    namd = [c for c in cmds if c.startswith("namd3 ")]  # $NAMD with its value
    assert len(namd) >= 4 and all(c.startswith("namd3 +p") for c in namd)
    assert any(c.startswith("sander -O -i ") for c in cmds)
    assert cmds[-1] == f"python3 golden_03_mdff_{STAMP}.py final"


@pytest.mark.parametrize("interpreter, command, expected", [
    ("bash", "", f"bash golden_03_script_{STAMP}.sh --model /golden/inputs/structure.pdb -o out.pdb"),
    ("custom", "Rscript --vanilla",
     f"Rscript --vanilla golden_03_script_{STAMP}.txt --model /golden/inputs/structure.pdb -o out.pdb"),
])
def test_user_script(interpreter, command, expected):
    settings = {"script": "echo hello", "interpreter": interpreter, "command": command,
                "arguments": "--model {in:structure} -o {out:result}",
                "inputs": [{"name": "structure", "kind": "pdb"}],
                "outputs": [{"name": "result", "kind": "pdb", "file": "out.pdb"}]}
    assert commands("user-script", settings) == {"run": [expected]}
