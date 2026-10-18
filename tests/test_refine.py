"""0.0.3c: real-space refinement planner steps (phenix-rsr, coot-rsr)."""

import os
import re
import subprocess
import sys
import types

import pytest

from mdatk.config import WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows import refine as R
from mdatk.workflows.plan import Plan, add_node, connect
from test_pdbtools import make_pdb
from test_workflows import STAMP, generated, messages, step


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def lrna_model(tmp_path, names=("0G5", "0C", "0A3"), mirror=True, name="model.pdb", header=(),
               phosphate5=False):
    """One chain A of L (mirrored) nucleotides with the given residue names (5' variants without their
    5'-phosphate unless ``phosphate5``)."""
    bases = [n.lstrip("0LR")[0] for n in names]
    residues = [(n, k + 1, b, mirror) for k, (n, b) in enumerate(zip(names, bases))]
    text = make_pdb([("A", residues)], header=header)
    if not phosphate5:
        text = "".join(ln for ln in text.splitlines(True)
                       if not (ln.startswith("ATOM") and ln[17:20].strip().endswith("5")
                               and ln[12:16].strip() in ("P", "OP1", "OP2", "OP3")))
    return write(tmp_path, name, text)


def phenix(settings=None, files=None, **kw):
    base = {"resolution": 3.5}
    base.update(settings or {})
    return step("phenix-rsr", base, files or {"model": "/tmp/missing/model.pdb", "map": "/tmp/missing/map.mrc"},
                **kw)


# ---------------------------------------------------------------------------
# registry and schema
# ---------------------------------------------------------------------------


def test_registered_in_refinement_category():
    for name in ("phenix-rsr", "coot-rsr"):
        tool = TOOLS[name]
        assert tool.category == "Refinement" and tool.generates
        assert [p.name for p in tool.outputs] == ["refined", "log"]
        assert tool.outputs[0].kind == "pdb" and tool.outputs[1].kind == "text"
        ports = {p.name: p for p in tool.inputs}
        assert ports["map"].kinds == ("map",) and ports["dictionaries"].multiple
        assert ports["dictionaries"].optional and ports["dictionaries"].kinds == ("cif",)
    assert set(R.PHENIX_VERSIONS) == {"1.21", "2.x"} and set(R.COOT_VERSIONS) == {"0.9", "1.x"}


# ---------------------------------------------------------------------------
# Phenix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["1.21", "2.x"])
def test_phenix_script_for_both_versions(version):
    plan, st = phenix({"phenix_version": version, "phenix_dir": f"/opt/phenix-{version}", "macro_cycles": 3,
                       "run_rigid_body": True, "run_adp": False, "nproc": 8, "random_seed": 42,
                       "extra": "# a comment\nramachandran_plot_restraints.enabled=False\n"
                                "write_initial_geo_file=False"},
                      files={"model": "/data/m.pdb", "map": "/data/emd_1.map",
                             "dictionaries": "/data/0A5.cif;/data/0G3.cif"})
    assert not messages(st, "error")
    files = generated(plan)
    sh = files["apt1_01_phenix-rsr_20261015-1432.sh"]
    assert sh.role == "run"
    text = sh.content
    assert f"source /opt/phenix-{version}/phenix_env.sh" in text
    assert f"# {R.PHENIX_VERSIONS[version]['label']}" in text
    assert "phenix.real_space_refine \\\n    /data/m.pdb \\\n    /data/emd_1.map \\\n" in text
    assert "    /data/0A5.cif \\\n    /data/0G3.cif \\\n" in text
    assert "resolution=3.5 \\" in text and "macro_cycles=3 \\" in text
    assert "run=minimization_global+rigid_body+local_grid_search \\" in text
    assert "pdb_interpretation.secondary_structure.enabled=True \\" in text
    assert "nproc=8 \\" in text and "random_seed=42 \\" in text
    assert "output.prefix=apt1_01_phenix-rsr_20261015-1432 \\" in text
    assert "    ramachandran_plot_restraints.enabled=False \\\n    write_initial_geo_file=False \\\n" in text
    assert "a comment" not in text
    assert "ls -t apt1_01_phenix-rsr_20261015-1432*real_space_refined*.pdb" in text
    assert "real_space_refined*.cif" in text
    assert 'cp "$OUT" apt1_01_refined_20261015-1432.pdb' in text
    assert not any(n.endswith(".eff") for n in files)  # no dot-bracket given
    assert st.outputs["refined"].paths == ["01_phenix-rsr/apt1_01_refined_20261015-1432.pdb"]
    assert st.outputs["log"].paths == ["01_phenix-rsr/apt1_01_phenix-rsr_20261015-1432.log"]


def test_phenix_defaults_and_options_off():
    plan, st = phenix({"na_ss": False, "prefix": "my run"})
    text = generated(plan)["apt1_01_phenix-rsr_20261015-1432.sh"].content
    assert "source " not in text and "secondary_structure" not in text
    assert "nproc=" not in text and "random_seed" not in text
    assert "run=minimization_global+local_grid_search+adp" in text
    assert "output.prefix=my_run" in text and "ls -t my_run*real_space_refined*.pdb" in text
    assert any("output prefix written as my_run" in m for m in messages(st, "info"))
    assert any("must be on PATH" in m for m in messages(st, "info"))


def test_phenix_script_passes_bash_syntax_check(tmp_path):
    plan, _ = phenix({"phenix_dir": "/opt/phenix", "sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))",
                      "numbering": "continuous"})
    for name, f in generated(plan).items():
        if name.endswith(".sh"):
            path = write(tmp_path, name, f.content)
            proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr


def test_phenix_eff_from_dot_bracket():
    plan, st = phenix({"sequence": "GGGCAC GUGCCU", "dotbracket": "(((((( ))))))", "numbering": "continuous"})
    assert not messages(st, "error")
    files = generated(plan)
    eff = files["apt1_01_basepairs_20261015-1432.eff"].content
    assert eff.startswith("# Base-pair restraints for phenix.real_space_refine")
    lines = [ln.strip() for ln in eff.splitlines()]
    assert lines[2:6] == ["pdb_interpretation {", "secondary_structure {", "enabled = True", "nucleic_acid {"]
    assert eff.count("base_pair {") == 6 and "stacking_pair" not in eff
    # G1-U12 is a wobble pair (Saenger 28); the rest are Watson-Crick G-C (19) / A-U (20)
    first = eff.split("base_pair {")[1]
    assert 'base1 = "chain \'A\' and resid 1"' in first and 'base2 = "chain \'B\' and resid 12"' in first
    assert "saenger_class = 28" in first
    assert 'base1 = "chain \'A\' and resid 5"' in eff and 'base2 = "chain \'B\' and resid 8"' in eff
    assert eff.count("saenger_class = 19") == 4 and eff.count("saenger_class = 20") == 1
    assert eff.count("{") == eff.count("}")
    sh = files["apt1_01_phenix-rsr_20261015-1432.sh"].content
    assert "    apt1_01_basepairs_20261015-1432.eff \\" in sh
    # per-chain numbering restarts at 1; no wobble pairs without the option
    plan, st = phenix({"sequence": "GGGCAC GUGCCU", "dotbracket": "(((((( ))))))", "numbering": "chain",
                       "wobble": False})
    eff = generated(plan)["apt1_01_basepairs_20261015-1432.eff"].content
    assert eff.count("base_pair {") == 5 and "saenger_class = 28" not in eff
    assert 'base2 = "chain \'B\' and resid 2"' in eff
    assert any("1 base pair(s) not restrained (GU)" in m for m in messages(st, "info"))


def test_phenix_eff_numbering_from_model(tmp_path):
    model = write(tmp_path, "m.pdb", make_pdb([("C", [("G", 45, "G", False), ("G", 46, "G", False)]),
                                               ("D", [("C", 70, "C", False), ("C", 71, "C", False)])]))
    plan, st = phenix({"sequence": "GG CC", "dotbracket": "(( ))"}, files={"model": model, "map": model})
    assert not messages(st, "error")
    eff = generated(plan)["apt1_01_basepairs_20261015-1432.eff"].content
    assert 'base1 = "chain \'C\' and resid 45"' in eff and 'base2 = "chain \'D\' and resid 71"' in eff
    assert 'base1 = "chain \'C\' and resid 46"' in eff and 'base2 = "chain \'D\' and resid 70"' in eff
    _, st = phenix({"sequence": "GGG CCC", "dotbracket": "((( )))"}, files={"model": model, "map": model})
    assert any("6 nucleotides, the model 4" in m for m in messages(st, "error"))
    _, st = phenix({"sequence": "GA UC", "dotbracket": "(( ))"}, files={"model": model, "map": model})
    assert any("sequence differs from the model at 2 position(s)" in m for m in messages(st, "warning"))


def test_phenix_eff_from_annotation(tmp_path):
    header = ["REMARK 950 MDATK BASE-PAIR ANNOTATION",
              "REMARK 950 PAIR  A   1 0G  B   4 0C  WCc",
              "REMARK 950 PAIR  A   2 0G  B   3 0U  GUc",
              "REMARK 950 PAIR  A   1 0G  B   3 0U  cWH",
              "REMARK 950 STACK A   1 0G  A   2 0G  >>"]
    model = write(tmp_path, "m.pdb", make_pdb([("A", [("0G", 1, "G", True), ("0G", 2, "G", True)]),
                                               ("B", [("0U", 3, "U", True), ("0C", 4, "C", True)])],
                                              header=header))
    plan, st = phenix({"pairs_source": "annotation", "stacking": True}, files={"model": model, "map": model})
    assert not messages(st, "error")
    assert any("1 annotated pair(s) are not Watson-Crick" in m for m in messages(st, "info"))
    eff = generated(plan)["apt1_01_basepairs_20261015-1432.eff"].content
    assert "# Source: REMARK 950 annotation of m.pdb" in eff
    assert eff.count("base_pair {") == 2 and eff.count("stacking_pair {") == 1
    assert "saenger_class = 19" in eff and "saenger_class = 28" in eff
    plain = write(tmp_path, "plain.pdb", make_pdb([("A", [("G", 1, "G", False)])]))
    _, st = phenix({"pairs_source": "annotation"}, files={"model": plain, "map": plain})
    assert any("mdatk pdb annotate" in m for m in messages(st, "warning"))
    _, st = phenix({"pairs_source": "annotation"})
    assert any("must exist when the files are written" in m for m in messages(st, "error"))


def test_phenix_checks():
    _, st = phenix({"resolution": None})
    assert any("give the resolution" in m for m in messages(st, "error"))
    _, st = phenix({"resolution": 25.0})
    assert any("outside the usual 1-20 Å" in m for m in messages(st, "warning"))
    _, st = phenix({"resolution": 0.8})
    assert any("outside the usual 1-20 Å" in m for m in messages(st, "warning"))
    _, st = phenix({"run_minimization_global": False, "run_local_grid_search": False, "run_adp": False})
    assert any("at least one run strategy" in m for m in messages(st, "error"))
    _, st = phenix({"resolution": 6.0, "na_ss": False})
    assert any("RNA helices tend to deform" in m for m in messages(st, "info"))
    _, st = step("phenix-rsr", {"resolution": 3.0})
    errs = messages(st, "error")
    assert any(m.startswith("Model (PDB): connect a step") for m in errs)
    assert any(m.startswith("Density map: connect a step") for m in errs)
    _, st = phenix()
    assert any("Density map: file not found (yet)" in m for m in messages(st, "warning"))
    _, st = phenix({"pairs_source": "none", "dotbracket": "((..))"})
    assert not messages(st, "error")


# ---------------------------------------------------------------------------
# residue names and dictionaries (shared checks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["phenix-rsr", "coot-rsr"])
def test_lrna_variants_need_dictionaries(tmp_path, tool):
    model = lrna_model(tmp_path)
    program = "Phenix" if tool == "phenix-rsr" else "Coot"
    _, st = step(tool, {"resolution": 3.0}, {"model": model, "map": model})
    warn = " | ".join(messages(st, "warning"))
    assert "0A3, 0G5" in warn and "connect a 'restraint-dicts' step" in warn
    assert "it knows 0A/0C/0G/0U" in warn
    assert any(f"0C are Chemical Component Dictionary entries; {program}" in m for m in messages(st, "info"))
    # dictionaries chosen by hand: the codes are read from the files
    d1 = write(tmp_path, "0G5.cif", "data_comp_list\nloop_\n_chem_comp.id\n0G5\ndata_comp_0G5\n")
    _, st = step(tool, {"resolution": 3.0}, {"model": model, "map": model, "dictionaries": d1})
    warn = " | ".join(messages(st, "warning"))
    assert "no restraint dictionary for 0A3 among the connected dictionaries" in warn
    d2 = write(tmp_path, "0A3.cif", "data_comp_0A3\n_chem_comp.id 0A3\n")
    _, st = step(tool, {"resolution": 3.0}, {"model": model, "map": model, "dictionaries": f"{d1};{d2}"})
    assert not [m for m in messages(st, "warning") if "dictionar" in m]


@pytest.mark.parametrize("tool", ["phenix-rsr", "coot-rsr"])
def test_amber_style_and_cpptraj_names_are_flagged(tmp_path, tool):
    amber = lrna_model(tmp_path, ("LG5", "LC", "LA3"), name="amber.pdb")
    _, st = step(tool, {"resolution": 3.0}, {"model": amber, "map": amber})
    warn = " | ".join(messages(st, "warning"))
    assert "does not recognise the residue names" in warn and "LG5 x1" in warn and "LA3 x1" in warn
    assert "mdatk pdb clean --scheme ccd" in warn
    dterm = write(tmp_path, "d.pdb", make_pdb([("A", [("G5", 1, "G", False), ("C3", 2, "C", False)])]))
    _, st = step(tool, {"resolution": 3.0}, {"model": dterm, "map": dterm})
    assert any("G5 x1, C3 x1" in m for m in messages(st, "warning"))
    # cpptraj output: L sugars with plain D names
    cpptraj = lrna_model(tmp_path, ("G", "C", "A"), name="cpptraj.pdb")
    _, st = step(tool, {"resolution": 3.0}, {"model": cpptraj, "map": cpptraj})
    assert any("3 residues have L sugars but D names" in m for m in messages(st, "warning"))
    # D-RNA with CCD names: nothing to report about names
    plain = write(tmp_path, "plain.pdb", make_pdb([("A", [("G", 1, "G", False), ("C", 2, "C", False)])]))
    _, st = step(tool, {"resolution": 3.0}, {"model": plain, "map": plain})
    assert not messages(st, "warning")


def test_model_from_a_step_and_lrna_builder_data():
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    src = add_node(wf, "phenix-rsr", settings={"resolution": 3.0})
    src.files.update(model="/tmp/missing/m.pdb", map="/tmp/missing/m.mrc")
    coot = add_node(wf, "coot-rsr")
    coot.files["map"] = "/tmp/missing/m.mrc"
    connect(wf, src.id, "refined", coot.id, "model")
    plan = Plan(wf, "/tmp/mdatk-plan")
    st = plan.steps[coot.id]
    assert st.ctx.inp("model") == "../01_phenix-rsr/apt1_01_refined_20261015-1432.pdb"
    assert any("residue names are checked once the model exists" in m for m in messages(st, "info"))
    assert any("found in the model when the script runs" in m for m in messages(st, "info"))
    py = generated(plan, coot.id)["apt1_02_coot-rsr_20261015-1432.py"].content
    assert "MODEL = '../01_phenix-rsr/apt1_01_refined_20261015-1432.pdb'" in py
    assert "ZONES = None" in py


# ---------------------------------------------------------------------------
# Coot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version,flags", [("0.9", "--no-graphics --no-guano --script"),
                                           ("1.x", "--no-graphics --script")])
def test_coot_scripts_for_both_versions(tmp_path, version, flags):
    model = lrna_model(tmp_path, ("0G5",) + ("0C",) * 22 + ("0A3",))
    plan, st = step("coot-rsr", {"coot_version": version, "coot_exe": "/opt/coot/bin/coot", "zone_size": 10,
                                 "overlap": 3, "cycles": 3, "weight_mode": "fixed", "weight": 45.0,
                                 "torsions": True},
                    {"model": model, "map": "/data/emd.map", "dictionaries": "/d/0G5.cif;/d/0A3.cif"})
    assert not messages(st, "error")
    files = generated(plan)
    sh = files["apt1_01_coot-rsr_20261015-1432.sh"]
    py = files["apt1_01_coot-rsr_20261015-1432.py"]
    assert sh.role == "run" and py.role == "input"
    assert (f"/opt/coot/bin/coot {flags} apt1_01_coot-rsr_20261015-1432.py < /dev/null "
            "> apt1_01_coot-rsr_20261015-1432.log 2>&1") in sh.content
    assert ("--no-guano" in sh.content) == (version == "0.9")
    assert f"# Coot batch real-space refinement ({R.COOT_VERSIONS[version]['label']})" in sh.content
    path = write(tmp_path, "coot.sh", sh.content)
    assert subprocess.run(["bash", "-n", path], capture_output=True).returncode == 0
    compile(py.content, py.name, "exec")
    text = py.content
    assert text.splitlines()[3] == "from __future__ import print_function"
    assert "DICTIONARIES = ['/d/0G5.cif', '/d/0A3.cif']" in text
    assert "WEIGHT_MODE = 'fixed'" in text and "WEIGHT = 45.0" in text and "TORSIONS = True" in text
    assert "    ('A', 1, 10),\n    ('A', 8, 17),\n    ('A', 15, 24),\n]" in text
    assert "'accept': ('accept_regularizement', 'accept_moving_atoms')," in text
    assert not re.search(r"(?<![\w'])f['\"]", text)  # no f-strings: Python 2.7 compatible (Coot 0.9 builds)


def test_coot_script_compiles_as_python2_syntax_subset():
    text = R.coot_script("m.pdb", "m.map", [], "out.pdb", [("A", 1, 5), (None, 7, 9)], 5, 1, 1, "auto", None,
                         False, "0.9")
    compile(text, "coot.py", "exec")
    assert "RANGES = [('A', 1, 5), (None, 7, 9)]" in text and "ZONES = None" in text


def _fake_coot(calls, tmp_path, accept_name):
    def record(name, result=None):
        def fn(*args):
            calls.append((name,) + args)
            return result
        return fn

    funcs = {name: record(name, 0) for name in (
        "read_cif_dictionary", "set_imol_refinement_map", "set_refinement_immediate_replacement", "refine_zone",
        "set_matrix", "set_refine_with_torsion_restraints", accept_name)}
    funcs["read_pdb"] = record("read_pdb", 0)
    funcs["handle_read_ccp4_map"] = record("handle_read_ccp4_map", 1)

    def write_pdb_file(imol, path):
        calls.append(("write_pdb_file", imol, path))
        with open(os.path.join(str(tmp_path), path), "w") as fh:
            fh.write("END\n")
    funcs["write_pdb_file"] = write_pdb_file
    funcs["coot_real_exit"] = record("coot_real_exit")
    return funcs


@pytest.mark.parametrize("style", ["0.9-globals", "1.x-module"])
def test_coot_script_runs_with_a_fake_coot(tmp_path, monkeypatch, style):
    model = lrna_model(tmp_path, ("0G5",) + ("0C",) * 6 + ("0A3",))
    with open(model, "a") as fh:  # water and an ion are not refined
        fh.write("HETATM  999  O   HOH W   1       0.000   0.000   0.000  1.00  0.00           O\n"
                 "HETATM 1000 MG    MG M   1       1.000   0.000   0.000  1.00  0.00          MG\n")
    text = R.coot_script(model, "map.mrc", ["d.cif"], "refined.pdb", [], 4, 1, 2, "fixed", 50.0, True,
                         style[:3])
    calls = []
    accept = "accept_regularizement" if style == "0.9-globals" else "accept_moving_atoms"
    funcs = _fake_coot(calls, tmp_path, accept)
    namespace = {"__name__": "__main__"}
    if style == "0.9-globals":
        namespace.update(funcs)
        monkeypatch.setitem(sys.modules, "coot", None)  # no coot module: import fails
    else:
        monkeypatch.setitem(sys.modules, "coot", types.SimpleNamespace(**funcs))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        exec(compile(text, "coot.py", "exec"), namespace)
    assert exc.value.code == 0
    names = [c[0] for c in calls]
    assert names[:2] == ["read_cif_dictionary", "read_pdb"]
    assert ("set_matrix", 50.0) in calls and ("set_refine_with_torsion_restraints", 1) in calls
    zones = [c[2:5] for c in calls if c[0] == "refine_zone"]
    assert zones == [("A", 1, 4), ("A", 4, 7), ("A", 5, 8)] * 2
    assert names.count(accept) == 6
    assert ("write_pdb_file", 0, "refined.pdb") in calls and calls[-1] == ("coot_real_exit", 0)
    assert (tmp_path / "refined.pdb").exists()


def test_coot_script_reports_missing_functions(tmp_path, monkeypatch, capsys):
    text = R.coot_script("m.pdb", "m.map", [], "out.pdb", [], 5, 1, 1, "default", None, False, "1.x")
    monkeypatch.setitem(sys.modules, "coot", types.SimpleNamespace(read_pdb=lambda p: 0))
    with pytest.raises(SystemExit) as exc:
        exec(compile(text, "coot.py", "exec"), {"__name__": "__main__"})
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "Coot functions not found: handle_read_ccp4_map" in out.err
    assert "ERROR: Coot refinement failed" in out.out


def test_coot_zones_and_ranges(tmp_path):
    residues = [("A", list(range(1, 24))), ("B", [5, 6, 7])]
    assert R.refinement_zones(residues, [], 10, 3) == [("A", 1, 10), ("A", 8, 17), ("A", 14, 23), ("B", 5, 7)]
    assert R.refinement_zones(residues, [("A", 5, 12), (None, 20, 22)], 4, 1) == [
        ("A", 5, 8), ("A", 8, 11), ("A", 9, 12), ("A", 20, 22)]
    assert R.refinement_zones(residues, [("B", 1, 100)], 1, 0) == [("B", 5, 5), ("B", 6, 6), ("B", 7, 7)]
    model = lrna_model(tmp_path, ("0G5",) + ("0C",) * 10 + ("0A3",))
    plan, st = step("coot-rsr", {"residues": "A : 3 - 8"}, {"model": model, "map": model})
    assert not messages(st, "error")
    assert "ZONES = [\n    ('A', 3, 8),\n]" in generated(plan)["apt1_01_coot-rsr_20261015-1432.py"].content
    plan, _ = step("coot-rsr", {"residues": "A"}, {"model": model, "map": model})
    assert "    ('A', 1, 10),\n    ('A', 3, 12),\n]" in generated(plan)["apt1_01_coot-rsr_20261015-1432.py"].content


def test_coot_checks(tmp_path):
    model = lrna_model(tmp_path, ("G", "C"), mirror=False)
    _, st = step("coot-rsr", {"zone_size": 5, "overlap": 5}, {"model": model, "map": model})
    assert any("overlap must be smaller" in m for m in messages(st, "error"))
    _, st = step("coot-rsr", {"residues": "A:1-x"}, {"model": model, "map": model})
    assert any(m.startswith("Residues: invalid residue range") for m in messages(st, "error"))
    _, st = step("coot-rsr", {"residues": "B:1-5"}, {"model": model, "map": model})
    assert any("no residue of the model matches" in m for m in messages(st, "error"))
    _, st = step("coot-rsr", {"residues": "1-5"}, {"model": model, "map": model})
    assert any("without a chain id apply to every chain" in m for m in messages(st, "info"))
    _, st = step("coot-rsr", {"coot_exe": " "}, {"model": model, "map": model})
    assert any("Coot executable" in m for m in messages(st, "error"))
    _, st = step("coot-rsr", {}, {})
    errs = messages(st, "error")
    assert any(m.startswith("Model (PDB)") for m in errs) and any(m.startswith("Density map") for m in errs)


# ---------------------------------------------------------------------------
# planner chaining
# ---------------------------------------------------------------------------


def test_planner_input_files_to_phenix_and_coot(tmp_path):
    model = lrna_model(tmp_path, ("0G5", "0C", "0A3"))
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    pdb = add_node(wf, "input-file", settings={"path": model, "kind": "pdb"})
    emap = add_node(wf, "input-file", settings={"path": str(tmp_path / "emd_1234.map"), "kind": "map"})
    rsr = add_node(wf, "phenix-rsr", settings={"resolution": 3.2})
    coot = add_node(wf, "coot-rsr")
    connect(wf, pdb.id, "file", rsr.id, "model")
    connect(wf, emap.id, "file", rsr.id, "map")
    connect(wf, rsr.id, "refined", coot.id, "model")
    connect(wf, emap.id, "file", coot.id, "map")
    if "restraint-dicts" in TOOLS:
        dicts = add_node(wf, "restraint-dicts", settings={"codes_mode": "variants"})
        connect(wf, dicts.id, "dictionaries", rsr.id, "dictionaries")
    plan = Plan(wf, str(tmp_path / "plan"))
    assert plan.order.index(rsr.id) < plan.order.index(coot.id)
    st = plan.steps[rsr.id]
    assert st.ctx.inp("model") == model and st.ctx.inp("map") == str(tmp_path / "emd_1234.map")
    assert any("map file not found (yet)" in m for m in messages(st, "warning"))
    if "restraint-dicts" in TOOLS:
        assert not [m for m in messages(st, "warning") if "dictionar" in m]
        assert all(p.startswith("../") and p.endswith(".cif") for p in st.ctx.inps("dictionaries"))
    else:
        assert any("connect a 'restraint-dicts' step" in m for m in messages(st, "warning"))
    assert not messages(st, "error")
    files = generated(plan, rsr.id)
    sh = files[f"apt1_0{rsr.step}_phenix-rsr_20261015-1432.sh"].content
    assert f"    {model} \\\n    {tmp_path / 'emd_1234.map'} \\\n" in sh
    for p in st.ctx.inps("dictionaries"):
        assert f"    {p} \\\n" in sh
    cst = plan.steps[coot.id]
    assert cst.ctx.inp("model") == f"../0{rsr.step}_phenix-rsr/apt1_0{rsr.step}_refined_20261015-1432.pdb"
    report = plan.write([rsr.id, coot.id])
    for path in report.written:
        if path.endswith(".sh"):
            assert subprocess.run(["bash", "-n", path], capture_output=True).returncode == 0
        if path.endswith(".py"):
            proc = subprocess.run([sys.executable, "-I", "-m", "py_compile", path], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr


def test_describe_and_resources():
    tool = TOOLS["phenix-rsr"]
    s = tool.normalise({"resolution": 3.5, "phenix_version": "1.21"})
    assert tool.describe(s) == "Phenix 1.21: 3.5 Å, 5 macro cycles"
    assert set(tool.resources(s)) <= {"phenix", "phenix_rsr"}
    coot = TOOLS["coot-rsr"]
    assert coot.describe(coot.defaults()) == "Coot 1.x: 10-residue zones × 2"
    assert set(coot.resources(coot.defaults())) <= {"coot"}


def test_5prime_variant_with_a_phosphate_is_flagged(tmp_path):
    model = lrna_model(tmp_path, phosphate5=True)
    _, st = step("phenix-rsr", {"resolution": 3.0}, {"model": model, "map": model})
    warn = " | ".join(messages(st, "warning"))
    assert "carry a phosphate: A:1 0G5" in warn and "mdatk pdb clean --scheme ccd" in warn


def test_inline_comments_mmcif_numbering_and_coot_mmcif(tmp_path):
    from mdatk.workflows.base import extra_flags
    assert extra_flags("a=1  # off for RNA\n# note\nb=2\nc=x#y") == ["a=1", "b=2", "c=x#y"]
    cif = write(tmp_path, "model.cif", "data_x\n")
    _, st = phenix({"pairs_source": "secstruct", "sequence": "GGC GCC", "dotbracket": "((( )))",
                    "numbering": "model"}, {"model": cif, "map": cif})
    assert any("numbering from the model needs a PDB model" in m for m in messages(st, "error"))
    _, st = step("coot-rsr", {}, {"model": cif, "map": cif})
    assert any("give a PDB file" in m for m in messages(st, "error"))
