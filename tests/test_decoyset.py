"""0.0.12: the full set of decoys of a FARFAR2 step as a collection port (Port.collection): the port flag and
its connection rules, the FARFAR2 step's extraction of every decoy with the decoy table, and the steps that
take many models reading the set from its list when their scripts run."""

import os
import stat
import subprocess
import textwrap

import pytest

from mdatk import builder, rosetta
from mdatk.config import PlanEdge, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan, add_node, connect, generate_step
from mdatk.workflows.schema import Port

STAMP = "20261015-1432"
# the multi-file model inputs of the steps that classify many models: drawn as large dots, they take the set
CLASSIFIERS = {"model-check": ["models"], "decoy-interface": ["models"], "register-census": ["models"],
               "ensemble-compare": ["arm_a", "arm_b"], "frame-check": ["models"],
               "map-model-score": ["model"]}


def _plan(**fold):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    node = add_node(wf, "farfar2", settings=dict({"sequence": "GGGCAC", "runs": 2, "nstruct": 3}, **fold))
    return wf, node


def _issues(plan, nid, level="error"):
    return [i.message for i in plan.steps[nid].issues if i.level == level]


# ---------------------------------------------------------------------------
# the port flag
# ---------------------------------------------------------------------------


def test_collection_flag():
    p = Port("decoys", "Every decoy", ("pdb",), collection=True)
    assert p.collection and p.multiple  # a set carries several files
    q = Port("model", "Model", ("pdb",))
    assert not q.collection and not q.multiple and not q.turns_on and not q.table
    ports = {p.name: p for p in TOOLS["farfar2"].output_ports(TOOLS["farfar2"].normalise({}))}
    assert ports["decoys"].collection and ports["decoys"].kind == "pdb"
    assert ports["decoys"].turns_on == "extract_decoys" and ports["decoys"].table == "decoy_table"
    assert ports["decoy_table"].kind == "table" and not ports["decoy_table"].collection
    assert not any(ports[k].collection for k in ("silent", "fasta", "secstruct"))
    for tool, names in CLASSIFIERS.items():
        t = TOOLS[tool]
        ins = {p.name: p for p in t.input_ports(t.normalise({}))}
        assert all(ins[n].collection and ins[n].multiple for n in names), tool
        assert not any(p.collection for n, p in ins.items() if n not in names), tool
    t = TOOLS["decoy-interface"]  # models from the scores table: the input stays a (now optional) set
    models = t.input_ports(t.normalise({"models_from": "scores"}))[0]
    assert models.collection and models.optional
    # ordinary multi-file inputs are no sets
    assert not TOOLS["vmd"].inputs[1].collection and not TOOLS["phenix-rsr"].inputs[2].collection


# ---------------------------------------------------------------------------
# connection rules
# ---------------------------------------------------------------------------


def test_connecting_the_set_switches_the_extraction_on():
    wf, fold = _plan()
    mc = add_node(wf, "model-check")
    assert fold.settings["extract_decoys"] is False
    edge = connect(wf, fold.id, "decoys", mc.id, "models")
    assert edge == PlanEdge(fold.id, "decoys", mc.id, "models") and fold.settings["extract_decoys"] is True
    plan = Plan(wf, "/tmp/plan")
    value = plan.steps[mc.id].ctx.value("models")
    listing = f"01_farfar2/apt1_01_decoys_{STAMP}.txt"
    assert value.paths == [listing] and value.lists == {listing: 6} and value.path == ""
    assert plan.steps[mc.id].ctx.inps("models") == [] and plan.steps[mc.id].ctx.count("models") == 6
    assert not _issues(plan, mc.id)
    # the decoy table connects too, and switches the extraction on as well
    wf, fold = _plan()
    mc = add_node(wf, "model-check")
    connect(wf, fold.id, "decoy_table", mc.id, "scores")
    assert fold.settings["extract_decoys"] is True


def test_the_set_into_a_table_input_connects_the_decoy_table():
    wf, fold = _plan()
    for tool, port in (("model-check", "scores"), ("frame-check", "scores"), ("register-census", "scores"),
                       ("decoy-interface", "scores"), ("ensemble-compare", "scores")):
        node = add_node(wf, tool)
        edge = connect(wf, fold.id, "decoys", node.id, port)
        assert (edge.output, edge.input) == ("decoy_table", port), tool
    plan = Plan(wf, "/tmp/plan")
    for nid in plan.order[1:]:
        assert plan.steps[nid].ctx.inp("scores") == f"../01_farfar2/apt1_01_scores_{STAMP}.tsv"


def test_the_set_connects_only_to_inputs_that_take_a_set():
    wf, fold = _plan()
    build = add_node(wf, "amber-build")
    with pytest.raises(ValueError, match="takes one file"):
        connect(wf, fold.id, "decoys", build.id, "structure")
    di = add_node(wf, "decoy-interface")
    with pytest.raises(ValueError, match="one by one"):  # its reference models are no set
        connect(wf, fold.id, "decoys", di.id, "references")
    with pytest.raises(ValueError, match="accepts"):  # a set of PDB files into a map input
        connect(wf, fold.id, "decoys", add_node(wf, "map-model-score").id, "map")
    assert not wf.edges and fold.settings["extract_decoys"] is False  # a refused connection changes nothing
    # ordinary files still connect to an input that takes a set, beside the set
    models = add_node(wf, "farfar2-models")
    mc = add_node(wf, "model-check")
    connect(wf, fold.id, "silent", models.id, "silent")
    connect(wf, models.id, "models", mc.id, "models")
    connect(wf, fold.id, "decoys", mc.id, "models")
    plan = Plan(wf, "/tmp/plan")
    ctx = plan.steps[mc.id].ctx
    assert len(ctx.inps("models")) == 10
    assert ctx.lists("models") == [f"../01_farfar2/apt1_01_decoys_{STAMP}.txt"]
    assert ctx.count("models") == 16
    # an edge written into the project file by hand: the step says what is wrong
    wf.edges.append(PlanEdge(fold.id, "decoys", di.id, "references"))
    plan = Plan(wf, "/tmp/plan")
    assert any("cannot take the set of files" in m for m in _issues(plan, di.id))
    # the extraction switched off again: the steps that read the set cannot have it
    fold.settings["extract_decoys"] = False
    plan = Plan(wf, "/tmp/plan")
    assert any("does not produce this file" in m for m in _issues(plan, mc.id))


# ---------------------------------------------------------------------------
# the FARFAR2 step's files
# ---------------------------------------------------------------------------


def test_farfar2_without_the_set_is_unchanged():
    wf, fold = _plan()
    plan = Plan(wf, "/tmp/plan")
    st = plan.steps[fold.id]
    assert set(st.outputs) == {"silent", "fasta", "secstruct"}
    files = generate_step(st)
    assert [f.name for f in files] == [f"apt1_01_farfar2_{STAMP}.fasta", f"apt1_01_farfar2_{STAMP}.flags",
                                       f"apt1_01_farfar2_{STAMP}.sh"]
    assert "decoys" not in files[-1].content and "extract" not in files[-1].content
    assert not any("decoy" in i.message for i in st.issues)


def test_farfar2_with_the_set_writes_the_extraction_and_the_table():
    wf, fold = _plan(input_res="A:1-3", database="/opt/rosetta/database", extract_parallel=2)
    fold.files["structure"] = "/data/helices.pdb"
    mc = add_node(wf, "model-check")
    di = add_node(wf, "decoy-interface", settings={"side_a": "A:1-3"})
    connect(wf, fold.id, "decoys", mc.id, "models")
    connect(wf, fold.id, "decoys", mc.id, "scores")
    connect(wf, fold.id, "decoys", di.id, "models")
    plan = Plan(wf, "/tmp/plan")
    st = plan.steps[fold.id]
    assert st.outputs["decoy_table"].paths == [f"01_farfar2/apt1_01_scores_{STAMP}.tsv"]
    assert st.outputs["decoys"].data["input_res"] == "A:1-3"  # for the support check of decoy interfaces
    assert any("up to 6 PDB files" in i.message for i in st.issues if i.level == "info")
    files = {f.name: f for f in generate_step(st)}
    assert set(files) == {f"apt1_01_farfar2_{STAMP}.fasta", f"apt1_01_farfar2_{STAMP}.flags",
                          f"apt1_01_farfar2_{STAMP}.sh", f"apt1_01_decoys_{STAMP}.sh",
                          f"apt1_01_scores_{STAMP}.py"}
    main = files[f"apt1_01_farfar2_{STAMP}.sh"]
    assert main.role == "run"
    assert main.content.rstrip().splitlines()[-1].startswith(f"bash apt1_01_decoys_{STAMP}.sh || ")
    ext = files[f"apt1_01_decoys_{STAMP}.sh"]
    assert ext.role == "helper" and ext.executable
    text = ext.content
    for line in (f"rm -f apt1_01_decoys_{STAMP}.txt apt1_01_scores_{STAMP}.tsv",
                 f"extract_run r01 ../../apt1_01_farfar2-r01_{STAMP}.out &",
                 "extract_pdbs -database /opt/rosetta/database -in:file:silent",
                 f"python3 apt1_01_scores_{STAMP}.py check-extracted models",
                 f"python3 apt1_01_scores_{STAMP}.py table apt1_01_scores_{STAMP}.tsv --extracted models "
                 f"--term score --list apt1_01_decoys_{STAMP}.txt"):
        assert line in text, line
    helper = files[f"apt1_01_scores_{STAMP}.py"]
    assert helper.role == "helper" and "RUNS = [('r01', " in helper.content
    assert "PREFIX = '01_farfar2'" in helper.content
    compile(helper.content, "helper", "exec")
    # the steps that take the set read its list when they run
    sh = generate_step(plan.steps[mc.id])[0].content
    assert f"done < ../01_farfar2/apt1_01_decoys_{STAMP}.txt" in sh and '"${MODELS[@]}" --clash-tol' in sh
    assert f"--scores ../01_farfar2/apt1_01_scores_{STAMP}.tsv" in sh and "# Check up to 6 model(s)" in sh
    sh = generate_step(plan.steps[di.id])[0].content
    assert f"done < ../01_farfar2/apt1_01_decoys_{STAMP}.txt" in sh and 'census "${MODELS[@]}"' in sh


def test_every_step_that_takes_the_set_reads_its_list():
    settings = {"decoy-interface": {"side_a": "A:1-3"},
                "register-census": {"registers": [{"name": "pres", "pairs": "A:1-A:6"}]},
                "ensemble-compare": {"register_a": "pres=A:1-A:6"},
                "frame-check": {"fit": "A:1-3", "measure": "B=A:4-6"}}
    files = {"frame-check": {"reference": "/data/ref.pdb"}, "map-model-score": {"map": "/data/map.mrc"}}
    wf, fold = _plan()
    nodes = {}
    for tool, ports in CLASSIFIERS.items():
        nodes[tool] = add_node(wf, tool, settings=settings.get(tool, {}))
        nodes[tool].files.update(files.get(tool, {}))
        for port in ports:
            connect(wf, fold.id, "decoys", nodes[tool].id, port)
    connect(wf, fold.id, "decoys", nodes["ensemble-compare"].id, "scores")  # closure by a score term
    plan = Plan(wf, "/tmp/plan")
    listing = f"../01_farfar2/apt1_01_decoys_{STAMP}.txt"
    for tool, node in nodes.items():
        assert not _issues(plan, node.id), tool
        sh = generate_step(plan.steps[node.id])[0].content
        assert sh.count(f"done < {listing}") == len(CLASSIFIERS[tool]), tool
        assert "../$f" in sh, tool  # plan-relative members, seen from the step folder


# ---------------------------------------------------------------------------
# running it: fake Rosetta programs, the real model check
# ---------------------------------------------------------------------------

# rna_denovo: three decoys per run into -out:file:silent (SCORE lines only; nothing reads their coordinates)
FAKE_DENOVO = textwrap.dedent('''\
    out=""
    for ((i=1;i<=$#;i++)); do j=$((i+1)); [ "${!i}" = -out:file:silent ] && out="${!j}"; done
    {
      echo "SEQUENCE: gggcac"
      echo "SCORE: score rna_vdw linear_chainbreak description"
      for k in 1 2 3; do echo "SCORE: -1$k.5 1.0 0.$k S_00000$k"; done
    } > "$out"
    ''')
# extract_pdbs: every decoy of the silent file as <tag>.pdb (a copy of $TEMPLATE) in the current folder
FAKE_EXTRACT = textwrap.dedent('''\
    silent=""
    for ((i=1;i<=$#;i++)); do j=$((i+1)); [ "${!i}" = -in:file:silent ] && silent="${!j}"; done
    awk '$1=="SCORE:" && $NF!="description" {print $NF}' "$silent" | while read -r t; do
      [ "$t" = "${DROP:-}" ] || cp "$TEMPLATE" "$t.pdb"
    done
    ''')


def _fake_tools(folder, scripts, **env):
    bindir = folder / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for name, body in scripts.items():
        p = bindir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", **env)


def test_farfar2_decoys_feed_a_model_check(tmp_path):
    template = tmp_path / "helix.pdb"
    template.write_text(builder.build_helix("GGCGCC"))
    wf, fold = _plan(make_inputs=True, fixed_stems=False)
    mc = add_node(wf, "model-check", settings={"hands": "none"})
    connect(wf, fold.id, "decoys", mc.id, "models")
    connect(wf, fold.id, "decoys", mc.id, "scores")
    plan = Plan(wf, str(tmp_path))
    assert not plan.has_errors(), plan.all_issues()
    plan.write()
    env = _fake_tools(tmp_path, {"rna_denovo": FAKE_DENOVO, "extract_pdbs": FAKE_EXTRACT},
                      TEMPLATE=str(template))
    folder = tmp_path / "01_farfar2"
    proc = subprocess.run(["bash", str(folder / f"apt1_01_farfar2_{STAMP}.sh")], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert sorted(os.listdir(folder / "models" / "r02")) == ["S_000001.pdb", "S_000002.pdb", "S_000003.pdb"]
    listed = (folder / f"apt1_01_decoys_{STAMP}.txt").read_text().split()
    assert listed[0] == "01_farfar2/models/r01/S_000001.pdb" and len(listed) == 6
    table = rosetta.read_decoy_table(str(folder / f"apt1_01_scores_{STAMP}.tsv"))
    assert [r.pdb for r in table.rows] == listed and table.rows[4].values["score"] == "-12.5"
    # the model check reads the set from the list and the scores from the table
    check = tmp_path / "02_model-check"
    proc = subprocess.run(["bash", str(check / f"apt1_02_model-check_{STAMP}.sh")], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    rows = (check / f"apt1_02_report_{STAMP}.tsv").read_text().splitlines()
    header = rows[0].split("\t")
    assert len(rows) == 7 and "score" in header
    assert {r.split("\t")[header.index("score")] for r in rows[1:]} == {"-11.5", "-12.5", "-13.5"}
    # the extraction runs alone too (the runs are not repeated); a run one decoy short stops it
    env["DROP"] = "S_000002"
    proc = subprocess.run(["bash", str(folder / f"apt1_01_decoys_{STAMP}.sh")], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 1 and "S_000002" in proc.stderr
    assert not (folder / f"apt1_01_decoys_{STAMP}.txt").exists()  # no list that looks finished
    proc = subprocess.run(["bash", str(check / f"apt1_02_model-check_{STAMP}.sh")], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 1 and "is missing: run the step that lists the set first" in proc.stderr
