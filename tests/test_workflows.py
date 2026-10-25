"""0.0.2b: script generators, logic checks and the Project Planner (no GUI)."""

import json
import os
import re
import stat
import subprocess
import sys
import textwrap

import pytest

from mdatk import tomlio
from mdatk.config import PlanEdge, PlanNode, Project, WorkflowParams
from mdatk.workflows import TOOLS, rna
from mdatk.workflows.amber import AmberMDTool
from mdatk.workflows.base import fortran_float
from mdatk.workflows.naming import file_name, parse_name, sanitize_project, sanitize_tag, step_folder
from mdatk.workflows.plan import (Plan, add_node, analysis_inputs, apply_template, connect,
                                  scratch_node, single_step_plan)
from mdatk.workflows import mg_params
from mdatk.workflows.schema import Option, requirements_met
from mdatk.workflows.scripts import amber2gmx, gmx_posre, prep_pdb

STAMP = "20261015-1432"


def step(tool, settings=None, files=None, step_no=1, **wf_kw):
    """Resolve a single step; returns (plan, resolved step)."""
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, **wf_kw)
    node = PlanNode(id="x", tool=tool, step=step_no, settings=TOOLS[tool].normalise(settings or {}),
                    files=dict(files or {}))
    plan = single_step_plan(wf, node, "/tmp/mdatk-plan")
    return plan, plan.steps["x"]


def messages(st, level=None):
    return [i.message for i in st.issues if level is None or i.level == level]


def generated(plan, name="x"):
    return {f.name: f for f in plan.generate([name]).get(name, [])}


# ---------------------------------------------------------------------------
# naming, schema, RNA helpers
# ---------------------------------------------------------------------------


def test_naming_convention():
    assert file_name("apt1", 3, "tleap", STAMP, "parm7") == "apt1_03_tleap_20261015-1432.parm7"
    assert file_name("apt1", 3, "tleap", "", "parm7") == "apt1_03_tleap.parm7"
    assert sanitize_tag("eq_free") == "eq-free" and sanitize_tag("a b/c") == "a-b-c"
    assert sanitize_project("my apt 1") == "my-apt-1" and sanitize_project("") == "project"
    assert step_folder(3, "amber-build") == "03_amber-build"
    parts = parse_name("my_apt_03_eq-free_20261015-1432.rst7")
    assert parts == {"project": "my_apt", "step": "03", "tag": "eq-free", "stamp": STAMP, "ext": "rst7"}
    assert parse_name("apt1_03_tleap.parm7")["tag"] == "tleap"


def test_option_coercion_and_requirements():
    o = Option("n", "N", "int", 1, minimum=0, maximum=5)
    assert o.coerce("3") == 3
    with pytest.raises(ValueError):
        o.coerce(7)
    with pytest.raises(ValueError):
        o.coerce(1.5)
    assert Option("t", "T", "float", None, optional=True).coerce("") is None
    assert Option("b", "B", "bool", False).coerce("yes") is True
    with pytest.raises(ValueError):
        Option("c", "C", "choice", "a", choices=[("a", "A")]).coerce("b")
    assert requirements_met({"engine": ("pmemd.MPI",), "shake": True}, {"engine": "pmemd.MPI", "shake": 1})
    assert not requirements_met({"engine": "pmemd"}, {"engine": "pmemd.cuda"})


def test_normalise_defaults_bad_values_and_records():
    tool = TOOLS["amber-md"]
    issues = []
    s = tool.normalise({"engine": "nonsense", "nproc": 4, "unknown_key": 1,
                        "stages": [{"name": "only", "kind": "min", "maxcyc": "oops"}]}, issues)
    assert s["engine"] == "pmemd.cuda" and s["nproc"] == 4
    assert [st["name"] for st in s["stages"]] == ["only"] and s["stages"][0]["maxcyc"] == 2000
    assert len(issues) == 2 and all(i.level == "warning" for i in issues)
    assert [st["name"] for st in tool.normalise({})["stages"]] == ["min1", "min2", "heat", "eq", "eq-free",
                                                                      "prod"]


def test_sequence_and_dotbracket_parsing():
    chains = rna.parse_sequence("A: ggcac\nB: GUGCC")
    assert [(c.id, c.sequence, c.start, c.end) for c in chains] == [("A", "GGCAC", 1, 5), ("B", "GUGCC", 6, 10)]
    assert [c.id for c in rna.parse_sequence("GGG & CCC")] == ["A", "B"]
    assert [c.id for c in rna.parse_sequence(">x A:1-3\nggg\n>x C:4-6\nccc")] == ["A", "C"]
    assert rna.parse_sequence("GGT")[0].sequence == "GGU"
    with pytest.raises(ValueError):
        rna.parse_sequence("GGX")
    db, pairs, spacers = rna.parse_dotbracket("((..[[)) ]]")
    assert db == "((..[[))]]" and pairs == [(1, 8), (2, 7), (5, 10), (6, 9)] and spacers == [8]
    for bad in ("((.)", "(.))", "(.x)"):
        with pytest.raises(ValueError):
            rna.parse_dotbracket(bad)
    with pytest.raises(ValueError, match="positions"):
        rna.build_model("GGGCCC", "((..))..")
    model = rna.build_model("GGGAAACCC UUU", "(((...)))...")
    assert model.stems() == [[(1, 9), (2, 8), (3, 7)]]
    assert model.ranges([1, 2, 3, 7, 8, 9, 10, 11]) == "A:1-3 A:7-9 B:10-11"


def test_fasta_and_secstruct_files():
    model = rna.build_model("A: GGCAC\nB: GUGCC", "((((( )))))")
    assert rna.fasta_text(model, "apt1") == ">apt1 A:1-5\nggcac\n>apt1 B:6-10\ngugcc\n"
    assert rna.secstruct_text(model) == "((((( )))))\nggcac gugcc\n"


def test_fortran_float_format():
    assert [fortran_float(v) for v in (10, 0.002, 1e-6, 300.0, 0.0, 2.8808, 1.5e-5)] == \
        ["10.0", "0.002", "1.0e-6", "300.0", "0.0", "2.8808", "1.5e-5"]


# ---------------------------------------------------------------------------
# Amber
# ---------------------------------------------------------------------------

USER_HEAT = """\
Heat 0 -> 300 K, RNA restrained
&cntrl
  imin=0, irest=0, ntx=1,
  nstlim=25000, dt=0.002,
  ntc=2, ntf=2, tol=1.0e-6,
  ntb=1, ntp=0, cut=10.0,
  ntt=3, gamma_ln=1.0, tempi=0.0, temp0=300.0,
  ntr=0,
  nmropt=1,
  ntpr=500, ntwx=500, ntwr=5000,
  ioutfm=1,
/
&wt type='END' /
DISANG=helix.dist
"""

USER_EQ_FREE = """\
Unrestrained NPT eq
&cntrl
  imin=0, irest=1, ntx=5,
  nstlim=50000, dt=0.002,
  ntc=2, ntf=2,
  ntb=2, ntp=1, pres0=1.0, taup=1.0, cut=10.0,
  ntt=3, gamma_ln=1.0, temp0=300.0,
  ntr=0,
  nmropt=1,
  ntpr=500, ntwx=500, ntwr=5000,
  ioutfm=1,
/
&wt type='END' /
DISANG=helix.dist
LISTOUT=POUT
"""

USER_MIN1 = """\
Minimize water/ions, RNA restrained
&cntrl
  imin=1, maxcyc=2000, ncyc=500,
  ntmin=1,
  ntb=1, ntp=0, cut=10.0,
  ntc=1, ntf=1,
  ntr=1, restraint_wt=10.0,
  restraintmask=':1-73',
  ntpr=100,
/
"""


def test_amber_stage_files_reproduce_the_reference_protocol():
    tool = AmberMDTool()
    stages = {st["name"]: st for st in tool.defaults()["stages"]}
    heat = dict(stages["heat"], title="Heat 0 -> 300 K, RNA restrained")
    assert tool.mdin(heat, 0, 1, "helix.dist", ":1-73") == USER_HEAT
    assert tool.mdin(stages["eq-free"], 1, 5, "helix.dist", ":1-73") == USER_EQ_FREE
    assert tool.mdin(stages["min1"], 0, 1, "helix.dist", ":1-73") == USER_MIN1
    ramp = dict(stages["heat"], ramp=True, ramp_steps=20000, distres=False)
    text = tool.mdin(ramp, 0, 1, "", ":1-10")
    assert "temp0=0.0" in text and "&wt type='TEMP0', istep1=0, istep2=20000, value1=0.0, value2=300.0 /" in text
    assert "nmropt=1," in text and "DISANG" not in text
    mc = dict(stages["eq"], barostat="mc", ig=42, extra_cntrl="nscm=1000")
    text = tool.mdin(mc, 1, 5, "h.dist", ":1-10")
    assert "barostat=2, mcbarint=100" in text and "taup" not in text and "ig=42" in text and "nscm=1000," in text


def amber_plan(build=None, md=None, restraints=None, structure="/tmp/missing/model.pdb"):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    b = add_node(wf, "amber-build", settings=build or {})
    b.files["structure"] = structure
    r = add_node(wf, "amber-restraints", settings=restraints or {
        "sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))"})
    m = add_node(wf, "amber-md", settings=md or {})
    connect(wf, b.id, "topology", r.id, "topology")
    connect(wf, b.id, "topology", m.id, "topology")
    connect(wf, b.id, "coordinates", m.id, "coordinates")
    connect(wf, r.id, "restraints", m.id, "restraints")
    return wf, b, r, m


def test_amber_plan_chains_file_names():
    wf, b, r, m = amber_plan()
    plan = Plan(wf, "/tmp/mdatk-plan")
    md = plan.steps[m.id]
    assert md.ctx.inp("topology") == "../01_amber-build/apt1_01_parmed_20261015-1432.parm7"
    assert md.ctx.inp("coordinates") == "../01_amber-build/apt1_01_tleap_20261015-1432.rst7"
    assert md.ctx.inp("restraints") == "../02_amber-restraints/apt1_02_helix_20261015-1432.dist"
    files = generated(plan, m.id)
    heat = files["apt1_03_heat_20261015-1432.in"].content
    assert "DISANG=../02_amber-restraints/apt1_02_helix_20261015-1432.dist" in heat
    script = files["apt1_03_amber-md_20261015-1432.sh"].content
    assert 'TOP="../01_amber-build/apt1_01_parmed_20261015-1432.parm7"' in script
    assert "-c ../01_amber-build/apt1_01_tleap_20261015-1432.rst7 -ref ../01_amber-build/" in script
    assert "-c apt1_03_min1_20261015-1432.rst7 -ref apt1_03_min1_20261015-1432.rst7" in script
    assert "prod" not in script.split("cpptraj")[0].split("# eq-free")[0]
    prod = files["apt1_03_amber-prod_20261015-1432.sh"]
    assert prod.role == "run-separate" and "-c apt1_03_eq-free_20261015-1432.rst7" in prod.content
    assert plan.steps[m.id].outputs["trajectory"].paths == ["03_amber-md/apt1_03_prod_20261015-1432.nc"]


def test_amber_build_files():
    plan, st = step("amber-build", {"mg_count": 6, "two_pass": True, "lrna": True, "lrna_ranges": "A:45-73",
                                    "cp_source": "frcmod",
                                    "extra_libs": "LRNA.lib", "extra_frcmod": "frcmod.LRNA",
                                    "leap_before_params": 'addAtomTypes { { "LD" "O" "sp3" } }',
                                    "hmr": True, "mg_model": "1264", "water": "opc"},
                    {"structure": "/data/model.pdb"})
    assert not messages(st, "error")
    files = generated(plan)
    wet = files["apt1_01_tleap_20261015-1432.leap"].content.splitlines()
    assert wet[2:9] == ["source leaprc.RNA.OL3", "source leaprc.water.opc", 'addAtomTypes { { "LD" "O" "sp3" } }',
                        "loadamberparams frcmod.99bsc0-chiol3-CaseP", "loadamberparams frcmod.LRNA",
                        "loadamberparams frcmod.ionslm_1264_opc", "loadOff LRNA.lib"]
    assert 'mol = loadpdb "apt1_01_lrna-dry_20261015-1432.pdb"' in wet
    assert "solvateoct mol OPCBOX 15.0" in wet
    assert wet.index("addIonsRand mol Na+ 0") < wet.index("addIonsRand mol MG 6") < wet.index(
        "addIonsRand mol Cl- 12")
    dry = files["apt1_01_tleap-dry_20261015-1432.leap"].content
    assert "solvate" not in dry and "water" not in dry and 'loadpdb "apt1_01_lrna_20261015-1432.pdb"' in dry
    pm = files["apt1_01_parmed_20261015-1432.in"].content
    assert "addLJType @%NB,NC\nchangeLJPair @%NB,NC @%H 2.1 0.63" in pm
    assert "changeLJPair @%H4 @%OS 2.9308 0.0505" in pm and "add12_6_4 :MG watermodel OPC" in pm
    assert "HMassRepartition" in pm and pm.rstrip().endswith("outparm apt1_01_parmed_20261015-1432.parm7")
    assert "HMassRepartition" not in files["apt1_01_parmed-dry_20261015-1432.in"].content
    sh = files["apt1_01_amber-build_20261015-1432.sh"].content
    order = [sh.index(x) for x in ("python3 apt1_01_prep", "python3 apt1_01_pdbtools_20261015-1432.py clean apt1_01_prep",
                                   "tleap -f apt1_01_tleap-dry", "parmed -O apt1_01_tleap-dry",
                                   "ambpdb -p apt1_01_parmed-dry", "pdbtools_20261015-1432.py clean apt1_01_ambpdb",
                                   "tleap -f apt1_01_tleap_", "parmed -O apt1_01_tleap_")]
    assert order == sorted(order)
    assert ("clean apt1_01_ambpdb_20261015-1432.pdb -o apt1_01_lrna-dry_20261015-1432.pdb --scheme amber "
            "--no-detect") in sh
    assert "to_lrna" not in sh and "def main(" in files["apt1_01_pdbtools_20261015-1432.py"].content
    out = st.outputs
    assert out["topology"].data["hmr"] and out["topology"].data["c4"] and out["topology"].data["nbfix"]
    split = generated(step("amber-build", {"split_types": True, "cp_source": "frcmod"}, {"structure": "/x.pdb"})[0])
    assert "addLJType @%HO\naddLJType @%O2\n#" in split["apt1_01_parmed_20261015-1432.in"].content


@pytest.mark.parametrize("settings, expected", [
    ({"rna_ff": "ol3"}, "NBfix19 / 0BPh masks assume the OL3 + CP"),
    ({"water": "tip4pd", "mg_model": "1264", "mg_count": 2}, "no 12-6-4 ion parameters"),
    ({"lrna_ranges": "45-x"}, "L ranges: cannot read residue range"),
])
def test_amber_build_errors(settings, expected):
    _plan, st = step("amber-build", settings, {"structure": "/x.pdb"})
    assert any(expected in m for m in messages(st, "error")), messages(st)


def test_amber_build_warnings():
    _plan, st = step("amber-build", {"two_pass": True, "buffer": 8.0, "rna_ff": "shaw", "nbfix19": False,
                                     "bph0": False}, {"structure": "/x.pdb"})
    warn = " ".join(messages(st, "warning"))
    assert "below 10" in warn and "TIP4P-D" in warn
    _plan, st = step("amber-build", {}, {"structure": "/x.pdb"})
    assert any("HO the same Lennard-Jones type as water HW" in m for m in messages(st, "info"))



# parmBSC0chiOL3 RNA atom types as listed in the authors' ff_Mg*.itp files (sigma nm, epsilon kJ/mol)
OL3_LJ = {"C": (0.339967, 0.359824), "C4": (0.339967, 0.359824), "C5": (0.339967, 0.359824),
          "CA": (0.339967, 0.359824), "CB": (0.339967, 0.359824), "CI": (0.339967, 0.45773),
          "CP": (0.339967, 0.359824), "CQ": (0.339967, 0.359824), "CS": (0.339967, 0.359824),
          "CT": (0.339967, 0.45773), "H": (0.106908, 0.0656888), "H1": (0.247135, 0.0656888),
          "H2": (0.229317, 0.0656888), "H4": (0.251055, 0.06276), "H5": (0.242146, 0.06276),
          "HA": (0.259964, 0.06276), "HO": (0.0, 0.0), "N*": (0.325, 0.71128), "N2": (0.325, 0.71128),
          "NA": (0.325, 0.71128), "NB": (0.325, 0.71128), "NC": (0.325, 0.71128), "O": (0.295992, 0.87864),
          "O2": (0.295992, 0.87864), "OH": (0.306647, 0.880314), "OS": (0.300001, 0.71128),
          "P": (0.374177, 0.8368)}


def test_mg_parameter_table_is_internally_consistent():
    """Each set scales the Lorentz-Berthelot pair of every RNA type by one (lambda_sigma, lambda_eps):
    a transcription error in any of the copied numbers would break that."""
    from mdatk.workflows import mg_params

    assert set(mg_params.MG_SETS) == {(m, w) for m in ("micro", "nano")
                                      for w in ("tip3p", "spce", "opc", "fb3", "tip4pew", "tip4pd", "tip4p2005")}
    assert not mg_params.UNAVAILABLE
    # the values confirmed by the authors (the papers' tables differ)
    assert round(mg_params.MG_SETS[("micro", "tip4pd")].rna["O2"][0], 4) == 0.2598
    assert round(mg_params.MG_SETS[("nano", "tip3p")].rna["O2"][0], 4) == 0.2279
    assert round(mg_params.MG_SETS[("micro", "opc")].mg_cl[0], 4) == 0.4318
    for key, p in mg_params.MG_SETS.items():
        assert set(p.rna) == set(OL3_LJ), key
        lam_s, lam_e = set(), set()
        for ty, (sig, eps) in p.rna.items():
            s0, e0 = OL3_LJ[ty]
            lam_s.add(round(sig / ((p.ion[0] + s0) / 2), 4))
            if e0:
                lam_e.add(round(eps / (p.ion[1] * e0) ** 0.5, 4))
            else:
                assert eps == 0.0
        assert len(lam_s) == 1 and len(lam_e) == 1, (key, lam_s, lam_e)
        assert 1.09 < lam_s.pop() < 1.33 and 0.2 < lam_e.pop() < 0.5, key  # 1.3255: TIP4P-D microMg
    # published scalings quoted in the papers (TIP4P-D microMg: lambda_sigma 1.3255, confirmed by N. Schwierz)
    p = mg_params.get_set("micro", "tip4pd")
    assert p.rna["O2"][0] / ((p.ion[0] + 0.295992) / 2) == pytest.approx(1.3255, abs=1e-4)
    p = mg_params.get_set("micro", "tip3p")
    assert p.rna["O2"][0] / ((p.ion[0] + 0.295992) / 2) == pytest.approx(1.1375, abs=1e-4)
    assert mg_params.MG_SETS[("micro", "opc")] == mg_params.MG_SETS[("micro", "spce")].__class__(
        **{**mg_params.MG_SETS[("micro", "spce")].__dict__, "source": mg_params.MG_SETS[("micro", "opc")].source})
    assert p.rmin_half == pytest.approx(0.572085, abs=1e-6) and p.eps_kcal == pytest.approx(56.35856, abs=1e-5)
    # TIP4P/2005 (Grotz & Schwierz 2022, Table 1): lambda_RNA 1.1484 / 0.2648 (micro), 1.1345 / 0.2923 (nano);
    # the table's nanoMg Mg-OP pair repeats the microMg value, the lambdas and the authors' file do not
    for model, lam in (("micro", (1.1484, 0.2648)), ("nano", (1.1345, 0.2923))):
        p = mg_params.get_set(model, "tip4p2005")
        s0, e0 = OL3_LJ["O2"]
        assert p.rna["O2"][0] / ((p.ion[0] + s0) / 2) == pytest.approx(lam[0], abs=1e-4)
        assert p.rna["O2"][1] / (p.ion[1] * e0) ** 0.5 == pytest.approx(lam[1], abs=1e-4)
    assert mg_params.get_set("nano", "tip4p2005").rna["O2"] == (0.2196970595, 7.62567188774)


def test_amber_build_optimised_mg_files():
    plan, st = step("amber-build", {"mg_count": 4, "mg_model": "nano", "water": "tip3p", "split_types": True,
                                    "parmed_extra": "addLJType @%P"}, {"structure": "/x.pdb"})
    assert not messages(st, "error"), messages(st)
    assert st.outputs["topology"].data["mg_pairs"] == "nanoMg" and not st.outputs["topology"].data["c4"]
    files = generated(plan)
    frc = files["apt1_01_mg_20261015-1432.frcmod"].content
    assert "MASS\nMg2+ 24.305\nMG   24.305\n" in frc and frc.isascii()
    leap = files["apt1_01_tleap_20261015-1432.leap"].content.splitlines()
    assert leap.index("loadamberparams apt1_01_mg_20261015-1432.frcmod") < leap.index(
        'mol = loadpdb "apt1_01_lrna_20261015-1432.pdb"')
    assert leap.index("source leaprc.water.tip3p") < leap.index("loadamberparams apt1_01_mg_20261015-1432.frcmod")
    pm = files["apt1_01_parmed_20261015-1432.in"].content
    lines = pm.splitlines()
    first_mg = next(k for k, ln in enumerate(lines) if ln.startswith("changeLJPair :MG"))
    assert all(k < first_mg for k, ln in enumerate(lines) if ln.startswith("addLJType"))
    assert "changeLJPair :MG @%O2 2.55761105 1.10579359" in pm
    assert "changeLJPair :MG @%N\\* " in pm and "changeLJPair :MG :Cl- 5.48171" in pm
    # 27 OL3 types + CP types OP, OR + the default L-RNA types LD, LH, LE
    assert sum(ln.startswith("changeLJPair :MG @%") for ln in lines) == 27 + 2 + 3
    os_line = next(ln for ln in lines if ln.startswith("changeLJPair :MG @%OS ")).split()[3:]
    oh_line = next(ln for ln in lines if ln.startswith("changeLJPair :MG @%OH ")).split()[3:]
    for t, ref in (("LD", os_line), ("LE", os_line), ("LH", oh_line)):
        assert next(ln for ln in lines if ln.startswith(f"changeLJPair :MG @%{t} ")).split()[3:] == ref
    assert "NONB\n  Mg2+     0.57545262  93.16414946\n  MG       0.57545262  93.16414946\n" in frc
    assert "doi:10.1021/acs.jctc.0c01281" in pm and pm.isascii()
    no_cl = generated(step("amber-build", {"mg_count": 4, "mg_model": "micro", "water": "spce",
                                           "mg_cl_pair": False}, {"structure": "/x.pdb"})[0])
    assert ":Cl-" not in no_cl["apt1_01_parmed_20261015-1432.in"].content
    # the plain 12-6 default writes no frcmod and no Mg pair terms
    plain = generated(step("amber-build", {"mg_count": 4}, {"structure": "/x.pdb"})[0])
    assert not any(n.endswith(".frcmod") for n in plain)
    assert ":MG" not in plain["apt1_01_parmed_20261015-1432.in"].content


@pytest.mark.parametrize("settings, level, expected", [
    ({"mg_model": "nano", "water": "opc3", "mg_count": 2}, "error", "no published nanoMg parameters for OPC3"),
    ({"mg_model": "nano", "mg_count": 0}, "warning", "no Mg²⁺ is added"),
    ({"mg_model": "micro", "mg_count": 2, "rna_ff": "ljbb", "nbfix19": False, "bph0": False}, "warning",
     "fitted with parmBSC0χOL3"),
    ({"mg_model": "micro", "mg_count": 2, "parmed_extra": "changeLJPair :MG @%O2 2.5 1.0"}, "warning",
     "applied after the extra ParmEd commands"),
    ({"mg_model": "micro", "mg_count": 2}, "info", "Smith-Dang"),
])
def test_amber_build_optimised_mg_checks(settings, level, expected):
    _plan, st = step("amber-build", settings, {"structure": "/x.pdb"})
    assert any(expected in m for m in messages(st, level)), messages(st)


def test_gromacs_keeps_optimised_mg_pairs():
    wf, b, _r, _m = amber_plan(build={"water": "tip3p", "mg_model": "micro", "mg_count": 2})
    g = add_node(wf, "gromacs", settings={"convert": "acpype"})
    connect(wf, b.id, "topology", g.id, "topology")
    connect(wf, b.id, "coordinates", g.id, "coordinates")
    plan = Plan(wf)
    assert any("acpype drops the microMg" in m for m in messages(plan.steps[g.id], "error"))
    g.settings["convert"] = "parmed"
    plan = Plan(wf)
    assert not messages(plan.steps[g.id], "error")
    assert any("microMg Mg²⁺-RNA pair terms are written" in m for m in messages(plan.steps[g.id], "info"))

def test_amber_md_checks():
    wf, b, r, m = amber_plan(md={"stages": [{"name": "min", "kind": "min"},
                                             {"name": "go", "kind": "md", "restart": "continue",
                                              "dt": 0.004, "temp0": 310.0},
                                             {"name": "go2", "kind": "md", "dt": 0.002, "temp0": 300.0,
                                              "shake": False, "ramp": True, "ramp_steps": 10 ** 9}]})
    msgs = " | ".join(messages(Plan(wf).steps[m.id]))
    for text in ("cannot continue velocities after a minimisation", "needs hydrogen mass repartitioning",
                 "dt > 0.001 ps needs SHAKE", "the ramp is longer than the stage",
                 "MD stages use different temperatures"):
        assert text in msgs, text
    _plan, st = step("amber-md", {}, {"topology": "/x.parm7", "coordinates": "/x.rst7"})
    assert any("min2, heat, eq, eq-free, prod use helix distance restraints" in t for t in messages(st, "error"))
    _plan, st = step("amber-md", {"engine": "pmemd.MPI", "mpi_launcher": "srun"},
                     {"topology": "/x.parm7", "coordinates": "/x.rst7", "restraints": "/x.dist"})
    assert any("does not contain {n}" in t for t in messages(st, "warning"))


def test_amber_md_segments_and_engines():
    stages = AmberMDTool().defaults()["stages"]
    stages[-1]["segments"] = 3
    plan, st = step("amber-md", {"stages": stages, "engine": "pmemd.MPI", "nproc": 16, "min_engine": "pmemd",
                                 "resume": False},
                    {"topology": "/x.parm7", "coordinates": "/x.rst7", "restraints": "/x.dist"}, step_no=4)
    files = generated(plan)
    prod = files["apt1_04_amber-prod_20261015-1432.sh"].content
    assert 'PMEMD="mpirun -np 16 pmemd.MPI"' in prod
    assert "-o apt1_04_prod-002_20261015-1432.out" in prod and "-c apt1_04_prod-002_20261015-1432.rst7" in prod
    assert "finished" not in prod
    main = files["apt1_04_amber-md_20261015-1432.sh"].content
    assert "\npmemd -O -i apt1_04_min1" in main and "$PMEMD -O -i apt1_04_heat" in main
    assert st.outputs["trajectory"].paths == [f"04_amber-md/apt1_04_prod-00{k}_20261015-1432.nc" for k in (1, 2, 3)]


def test_restraints_from_topology_and_generated_script(demo_dry, tmp_path):
    seq = "GCAUGGCAUCGA UCGAUGCCAUGC"
    s = {"sequence": seq, "dotbracket": "(" * 12 + " " + ")" * 12}
    plan, st = step("amber-restraints", s, {"topology": demo_dry["topology"]})
    assert not messages(st, "error")
    files = generated(plan)
    direct = files["apt1_01_helix_20261015-1432.dist"].content
    assert direct.count("&rst") == 31  # 7 G-C pairs x 3 + 5 A-U pairs x 2
    assert direct.startswith("# 1 O6 -- 24 N4\n&rst\n iat=")
    script = tmp_path / "helix.py"
    script.write_text(files["apt1_01_helix_20261015-1432.py"].content)
    out = tmp_path / "out.dist"
    proc = subprocess.run([sys.executable, "-I", str(script), demo_dry["topology"], str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out.read_text() == direct
    _plan, bad = step("amber-restraints", dict(s, res_offset=1), {"topology": demo_dry["topology"]})
    assert any("expects" in m or "does not exist" in m for m in messages(bad, "error"))
    _plan, wob = step("amber-restraints", {"sequence": "GGGG UCCC", "dotbracket": "(((( ))))"},
                      {"topology": "/nonexistent.parm7"})
    assert any("G·U" in m for m in messages(wob, "info"))


def test_prep_pdb_script(tmp_path):
    pdb = textwrap.dedent("""\
        ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P
        ATOM      2  OP1   G A   1       1.000   0.000   0.000  1.00  0.00           O
        ATOM      3  O5'   G A   1       2.000   0.000   0.000  1.00  0.00           O
        ATOM      4  H5'   G A   1       3.000   0.000   0.000  1.00  0.00           H
        ATOM      5  P     C A   2       4.000   0.000   0.000  1.00  0.00           P
        ATOM      6  O5'   C A   2       5.000   0.000   0.000  1.00  0.00           O
        ATOM      7  P    rG B   3       6.000   0.000   0.000  1.00  0.00           P
        ATOM      8  C1'  rG B   3       7.000   0.000   0.000  1.00  0.00           C
        HETATM    9  O   HOH B   4       8.000   0.000   0.000  1.00  0.00           O
        ATOM     10  C1'   U B   9       9.000   0.000   0.000  1.00  0.00           C
        END
        """)
    lines, stats = prep_pdb.prepare(pdb.splitlines())
    names = [ln[12:16].strip() for ln in lines if ln.startswith("ATOM")]
    assert names == ["O5'", "P", "O5'", "C1'", "C1'"]
    assert lines.count("TER") == 3 and lines[-1] == "END"
    assert " G B   3" in "\n".join(lines)  # rG renamed
    assert stats == {"atoms": 5, "hydrogens": 1, "phosphates": 3, "waters": 1, "chains": 3}


# ---------------------------------------------------------------------------
# FARFAR2 / SimRNA / RNAMake
# ---------------------------------------------------------------------------


def test_farfar2_files_and_flags():
    s = {"sequence": "A: GGGCAC\nB: GUGCCC", "dotbracket": "(((((( ))))))", "input_res_mode": "stems"}
    plan, st = step("farfar2", s, {"structure": "/data/helices.pdb", "map": "/data/map.mrc"})
    assert not messages(st, "error")
    files = generated(plan)
    assert files["apt1_01_farfar2_20261015-1432.fasta"].content == ">apt1 A:1-6\ngggcac\n>apt1 B:7-12\ngugccc\n"
    flags = files["apt1_01_farfar2_20261015-1432.flags"].content.splitlines()
    for flag in ("-fasta ../../apt1_01_farfar2_20261015-1432.fasta", "-cycles 50000", "-fixed_stems",
                 "-rna:denovo:bps_moves false", "-autofilter false", "-filter_chain_closure_distance 15.0",
                 "-s /data/helices.pdb", "-input_res A:1-6 B:7-12", "-edensity:mapfile /data/map.mrc",
                 "-edensity:mapreso 4.3", "-set_weights elec_dens_fast 20.0 coordinate_constraint 1 "
                 "linear_chainbreak 10.0", "-run:constant_seed"):
        assert flag in flags, flag
    sh = files["apt1_01_farfar2_20261015-1432.sh"].content
    assert "-out:file:silent ../../apt1_01_farfar2-r${N}_20261015-1432.out -run:jran $((3000 + I))" in sh
    assert "seq 1 12 | xargs -P 6" in sh and "set -eo pipefail" not in sh
    assert len(st.outputs["silent"].paths) == 12
    assert st.outputs["secstruct"].data["rna_model"].pairs[0] == (1, 12)
    _plan, bad = step("farfar2", {"sequence": "GGGAAA", "dotbracket": "((()))"})
    assert any("not complementary" in m for m in messages(bad, "error"))
    _plan, st = step("farfar2", {"sequence": "GGGCCC", "dotbracket": "((()))", "input_res": "A:1-3"})
    assert any("-input_res is set but no fixed-helix PDB" in m for m in messages(st, "warning"))


def _fake_tools(tmp_path, scripts):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in scripts.items():
        p = bindir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    return env


SILENT = """SEQUENCE: gggcac
SCORE:     score    rna_vdw   description
REMARK BINARY SILENTFILE
SCORE:   -10.5    1.0   S_000001
SCORE:   -30.0    1.0   S_000002
SCORE:   -20.0    1.0   S_000003
"""


def test_farfar2_models_script_runs_with_fake_rosetta(tmp_path):
    for clustering in (True, False):
        wf = WorkflowParams(project_name="apt1", stamp=STAMP)
        work = tmp_path / f"c{int(clustering)}"
        fold = add_node(wf, "farfar2", settings={"sequence": "GGGCAC", "runs": 2})
        models = add_node(wf, "farfar2-models", settings={"cluster": clustering, "n_models": 2})
        connect(wf, fold.id, "silent", models.id, "silent")
        plan = Plan(wf, str(work))
        assert not messages(plan.steps[models.id], "error")
        plan.write([models.id])
        for k in (1, 2):
            (work / "01_farfar2" / f"apt1_01_farfar2-r0{k}_{STAMP}.out").parent.mkdir(exist_ok=True)
            (work / "01_farfar2" / f"apt1_01_farfar2-r0{k}_{STAMP}.out").write_text(SILENT)
        env = _fake_tools(work, {
            "rna_cluster": 'out=""; for ((i=1;i<=$#;i++)); do [ "${!i}" = -out:file:silent ] && j=$((i+1)) && '
                           'out="${!j}"; done; printf "SCORE: score description\\nSCORE: -30 S_0\\n'
                           'SCORE: -20 S_1\\n" > "$out"',
            "extract_pdbs": 'for ((i=1;i<=$#;i++)); do [ "${!i}" = -in:file:tags ] && j=$((i+1)) && '
                            'echo "ATOM" > "${!j}.pdb"; done; exit 1',
            "x3dna-dssr": 'for a in "$@"; do case "$a" in -o=*) echo dssr > "${a#-o=}";; esac; done'})
        script = work / "02_farfar2-models" / f"apt1_02_farfar2-models_{STAMP}.sh"
        proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        folder = work / "02_farfar2-models"
        assert (folder / f"apt1_02_model-01_{STAMP}.pdb").exists()
        assert (folder / f"apt1_02_model-02_{STAMP}.dssr").exists()
        ranking = (folder / f"apt1_02_models_{STAMP}.txt").read_text().split("\n")[0].split()
        if clustering:
            assert ranking[:3] == ["01", "-30", "S_0"]
        else:
            assert ranking[:3] == ["01", "-30.0", "S_000002"]


def test_simrna_files():
    s = {"sequence": "GGGCACUUCG GUGCCCCG", "dotbracket": "((((((..[[ ))))))]]",
         "restraints": "SLOPE A/1/P A/16/P 0 10 1", "ss_weight": 2.0}
    plan, st = step("simrna", s)
    assert not messages(st, "error")
    files = generated(plan)
    assert files["apt1_01_simrna_20261015-1432.seq"].content == "gggcacuucg gugccccg\n"
    assert files["apt1_01_simrna_20261015-1432.ss"].content == \
        "((((((.... ))))))..\n........(( ......))\n"
    cfg = files["apt1_01_simrna_20261015-1432.config"].content
    assert "NUMBER_OF_ITERATIONS 16000000" in cfg and "SECOND_STRC_RESTRAINTS_WEIGHT 2.0" in cfg
    run = files["apt1_01_simrna_20261015-1432.sh"].content
    assert "-S ../../apt1_01_simrna_20261015-1432.ss -r ../../apt1_01_simrna_20261015-1432.restraints -E 10" in run
    assert 'ln -sfn "$SIMRNA_DIR/data" data' in run
    clus = files["apt1_01_simrna-cluster_20261015-1432.sh"].content
    assert "clustering apt1_01_simrna-all_20261015-1432.trafl 0.01 1.8" in clus
    assert 'SimRNA_trafl2pdbs "$REF" "$C" 1 AA' in clus
    _plan, st = step("simrna", {"sequence": "GGG", "write_every": 10 ** 9, "iterations": 1000})
    assert any("trajectory interval exceeds" in m for m in messages(st, "error"))


def test_rnamake_commands():
    base = {"start_bp": "A222-A251", "end_bp": "A149-A154", "search_type": "mc", "thermo_fluc": True}
    plan, st = step("rnamake", base, {"structure": "/data/scaffold.pdb"})
    assert not messages(st, "error")
    sh = generated(plan)["apt1_01_rnamake_20261015-1432.sh"].content
    for flag in ("design_rna_scaffold", "--pdb /data/scaffold.pdb", "--start_bp A222-A251", "--designs 10",
                 "--search_type mc", "--thermo_fluc --thermo_fluc_steps 1000000", "--dump_pdbs"):
        assert flag in sh, flag
    plan, st = step("rnamake", dict(base, app="design_rna", skip_seq_opt=True), {"structure": "/data/s.pdb"})
    sh = generated(plan)["apt1_01_rnamake_20261015-1432.sh"].content
    assert "design_rna \\\n  -pdb /data/s.pdb" in sh and "-mc" in sh and "-only_ideal" in sh and "-pdbs" in sh
    assert any("ignores: thermo_fluc" in m for m in messages(st, "warning"))
    _plan, st = step("rnamake", {"start_bp": "A1", "end_bp": "A1"}, {"structure": "/s.pdb"})
    assert len(messages(st, "error")) >= 2


# ---------------------------------------------------------------------------
# GROMACS / VMD
# ---------------------------------------------------------------------------


def test_gromacs_mdp_scripts_and_checks():
    plan, st = step("gromacs", {}, {"topology": "/x.parm7", "coordinates": "/x.rst7"})
    assert not messages(st, "error")
    files = generated(plan)
    npt = files["apt1_01_npt_20261015-1432.mdp"].content
    assert "define                  = -DPOSRES" in npt and "pcoupl                  = C-rescale" in npt
    assert "refcoord-scaling        = com" in npt and "continuation            = yes" in npt
    assert "gen-vel                 = yes" in files["apt1_01_nvt_20261015-1432.mdp"].content
    sh = files["apt1_01_gromacs_20261015-1432.sh"].content
    assert "python3 apt1_01_convert_20261015-1432.py /x.parm7 /x.rst7" in sh
    assert "-c apt1_01_em_20261015-1432.gro -r apt1_01_em_20261015-1432.gro" in sh
    assert "-t apt1_01_nvt_20261015-1432.cpt" in sh and "-update gpu" not in sh
    prod = files["apt1_01_gromacs-prod_20261015-1432.sh"].content
    assert "-t apt1_01_npt_20261015-1432.cpt" in prod
    bad = {"gmx_version": "2023", "mass_repartition": True, "update_gpu": True,
           "stages": [{"name": "em", "integrator": "steep"},
                      {"name": "md", "integrator": "md", "tcoupl": "Berendsen", "pcoupl": "Parrinello-Rahman"}]}
    _plan, st = step("gromacs", bad, {"topology": "/x.parm7", "coordinates": "/x.rst7"})
    errs = " | ".join(messages(st, "error"))
    for text in ("mass-repartition-factor exists only in GROMACS 2024", "Berendsen",
                 "Parrinello-Rahman with new velocities"):
        assert text in errs, text
    wf, b, _r, _m = amber_plan(build={"water": "opc"})
    g = add_node(wf, "gromacs", settings={"convert": "acpype", "update_gpu": True})
    connect(wf, b.id, "topology", g.id, "topology")
    connect(wf, b.id, "coordinates", g.id, "coordinates")
    errs = " | ".join(messages(Plan(wf).steps[g.id], "error"))
    assert "acpype drops off-diagonal" in errs and "virtual sites" in errs


def test_amber2gmx_pair_transfer():
    # three LJ types; pair (0, 2) modified, everything else follows the combining rule
    rad, eps = [1.0, 1.5, 0.0], [0.1, 0.2, 0.0]
    nt = 3
    idx = []
    pairs = {}
    for i in range(nt):
        for j in range(i, nt):
            r, e = rad[i] + rad[j], (eps[i] * eps[j]) ** 0.5
            if (i, j) == (0, 2):
                r, e = 2.73, 0.01
            pairs[(i, j)] = (e * r ** 12, 2 * e * r ** 6)
    order = list(pairs)
    for i in range(nt):
        for j in range(nt):
            idx.append(order.index((min(i, j), max(i, j))) + 1)
    acoef = [pairs[k][0] for k in order]
    bcoef = [pairs[k][1] for k in order]
    data = {"POINTERS": [0, nt], "NONBONDED_PARM_INDEX": idx, "LENNARD_JONES_ACOEF": acoef,
            "LENNARD_JONES_BCOEF": bcoef}
    found = amber2gmx.modified_pairs(data)
    assert len(found) == 1 and found[0][:2] == (0, 2)
    assert found[0][2] == pytest.approx(2.73) and found[0][3] == pytest.approx(0.01)
    top = "[ defaults ]\n1 2 yes 0.5 0.8333\n\n[ atomtypes ]\nHO 0 1.0 0 A 0 0\n\n[ moleculetype ]\nX 3\n"
    text, with_pairs = amber2gmx.insert_pair_sections(top, found, {0: ["HO", "HW"], 2: ["O2"]})
    assert with_pairs
    nb = text[text.index("[ nonbond_params ]"):text.index("[ moleculetype ]")]
    assert nb.count("1  2.43215") == 4  # HO-O2 and HW-O2 in both sections
    assert text.index("[ atomtypes ]") < text.index("[ nonbond_params ]") < text.index("[ pairtypes ]") < \
        text.index("[ moleculetype ]")
    sigma = 2.73 / 2 ** (1 / 6) / 10
    line = [ln for ln in text.splitlines() if ln.startswith("HO") and "O2" in ln]
    assert float(line[0].split()[3]) == pytest.approx(sigma) and float(line[0].split()[4]) == pytest.approx(0.04184)
    assert float(line[1].split()[4]) == pytest.approx(0.02092)


def test_gmx_posre_insertion():
    top = textwrap.dedent("""\
        [ moleculetype ]
        ; name nrexcl
        system1   3

        [ atoms ]
          1  P   1  G5  O5'  1 -0.6 16.0
          2  HO  1  G5  HO5' 1  0.4 1.0
          3  CT  2  A3  C1'  2  0.0 12.0

        [ bonds ]
          1 2 1

        [ moleculetype ]
        WAT 2

        [ atoms ]
          1 OW 1 WAT O 1 -0.8 16.0

        [ system ]
        x
        """)
    text, itps = gmx_posre.add_posres(top, "apt1_04_posre-{mol}.itp", 1000.0)
    assert list(itps) == ["apt1_04_posre-system1.itp"]
    body = itps["apt1_04_posre-system1.itp"]
    assert "     1     1  1000.0" in body and "     3     1" in body and "     2     1" not in body
    assert text.index('#include "apt1_04_posre-system1.itp"') < text.index("WAT 2")


def test_vmd_script_uses_groups(tmp_path):
    from mdatk.workflows.vmd import residue_selection
    assert residue_selection("1-12, 20, A:30-40") == "resid 1 to 12 20 30 to 40"
    project = Project()
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    node = PlanNode(id="x", tool="vmd", step=6, settings=TOOLS["vmd"].normalise({"render": "movie"}),
                    files={"topology": "/d/dry.parm7", "trajectory": "/d/a.nc;/d/b.nc"})
    plan = single_step_plan(wf, node, str(tmp_path), project.groups)
    files = generated(plan)
    tcl = files["apt1_06_vmd_20261015-1432.tcl"].content
    assert "mol new {/d/dry.parm7} type parm7 waitfor all" in tcl
    assert "mol addfile {/d/b.nc} type netcdf first 0 last -1 step 1 waitfor all" in tcl
    assert "color change rgb 23 0.165 0.471 0.839 ;# Aptamer" in tcl and "mol selection {resid 13 to 24}" in tcl
    assert "render TachyonInternal" in tcl and "measure fit" in tcl
    sh = files["apt1_06_vmd_20261015-1432.sh"].content
    assert "vmd -dispdev text -e apt1_06_vmd_20261015-1432.tcl" in sh and "-c:v libx264" in sh


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def test_connect_validation_and_order():
    wf = WorkflowParams()
    md = add_node(wf, "amber-md")
    b = add_node(wf, "amber-build")
    vmd = add_node(wf, "vmd")
    with pytest.raises(ValueError, match="accepts"):
        connect(wf, b.id, "coordinates", md.id, "topology")
    connect(wf, b.id, "topology", md.id, "topology")
    connect(wf, b.id, "topology", md.id, "topology")  # replaces, no duplicate
    assert len(wf.edges) == 1
    connect(wf, md.id, "trajectory", vmd.id, "trajectory")
    md2 = add_node(wf, "amber-md")
    connect(wf, md.id, "restart", md2.id, "coordinates")
    with pytest.raises(ValueError, match="loop"):
        connect(wf, md2.id, "restart", md.id, "coordinates")
    with pytest.raises(ValueError, match="itself"):
        connect(wf, md.id, "restart", md.id, "coordinates")
    plan = Plan(wf)
    assert plan.order.index(b.id) < plan.order.index(md.id) < plan.order.index(vmd.id)
    assert plan.order.index(md.id) < plan.order.index(md2.id)


def test_plan_collisions_cycles_and_unknown_tools():
    wf = WorkflowParams()
    a = add_node(wf, "amber-md", step=3)
    add_node(wf, "amber-md", step=3)
    wf.nodes.append(PlanNode(id="zz", tool="does-not-exist", step=9))
    plan = Plan(wf)
    assert any("same folder" in m for m in messages(plan.steps[wf.nodes[1].id], "error"))
    assert any("unknown step type" in m for m in messages(plan.steps["zz"], "error"))
    wf2 = WorkflowParams()
    x = add_node(wf2, "amber-md")
    y = add_node(wf2, "amber-md")
    wf2.edges += [PlanEdge(x.id, "restart", y.id, "coordinates"), PlanEdge(y.id, "restart", x.id, "coordinates")]
    plan2 = Plan(wf2)
    assert all(any("loop" in m for m in messages(plan2.steps[n], "error")) for n in (x.id, y.id))
    assert a.step == 3


def test_plan_write_master_summary_and_stamp(tmp_path):
    wf = WorkflowParams(project_name="apt1")
    nodes = apply_template(wf, "amber")
    build, restraints, md, analysis = nodes
    build.files["structure"] = str(tmp_path / "model.pdb")
    restraints.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    assert not wf.stamp
    plan = Plan(wf, str(tmp_path / "plan"))
    report = plan.write()
    assert wf.stamp == plan.stamp and re.fullmatch(r"\d{8}-\d{4}", wf.stamp)
    master = open(report.master).read()
    lines = [ln for ln in master.splitlines() if ln.startswith("bash ")]
    assert lines == [f'bash "01_amber-build/apt1_01_amber-build_{wf.stamp}.sh"',
                     f'bash "02_amber-restraints/apt1_02_amber-restraints_{wf.stamp}.sh"',
                     f'bash "03_amber-md/apt1_03_amber-md_{wf.stamp}.sh"']
    assert f"run separately: bash 03_amber-md/apt1_03_amber-prod_{wf.stamp}.sh" in master
    summary = open(report.summary).read()
    assert "## 03 · Amber · MD stages" in summary and "from 01 · Amber · system build" in summary
    for path in report.written:
        assert os.path.exists(path)
    sh = tmp_path / "plan" / "03_amber-md" / f"apt1_03_amber-md_{wf.stamp}.sh"
    assert os.access(sh, os.X_OK)
    proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    top, trajs = analysis_inputs(plan, analysis.id)
    assert top.endswith(f"01_amber-build/apt1_01_parmed_{wf.stamp}.parm7")
    assert trajs == [str(tmp_path / "plan" / "03_amber-md" / f"apt1_03_prod_{wf.stamp}.nc")]
    again = Plan(wf, str(tmp_path / "plan"))
    assert again.stamp == wf.stamp  # stable names on re-generation


def test_all_generated_scripts_pass_bash_syntax_check(tmp_path):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    apply_template(wf, "farfar2-amber")
    for n in wf.nodes:
        if n.tool == "farfar2":
            n.settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    simrna = add_node(wf, "simrna", settings={"sequence": "GGGCAC GUGCCC", "dotbracket": "(((((( ))))))"})
    rm = add_node(wf, "rnamake", settings={"start_bp": "A1-A12", "end_bp": "A6-A7"})
    connect(wf, simrna.id, "best", rm.id, "structure")
    gmx = add_node(wf, "gromacs")
    build = [n for n in wf.nodes if n.tool == "amber-build"][0]
    connect(wf, build.id, "topology", gmx.id, "topology")
    connect(wf, build.id, "coordinates", gmx.id, "coordinates")
    plan = Plan(wf, str(tmp_path))
    errors = [(plan.steps[n].label, i.message) for n, i in plan.all_issues() if i.level == "error"]
    assert errors == []
    report = plan.write()
    scripts = [p for p in report.written if p.endswith(".sh")]
    assert len(scripts) >= 11
    for path in scripts:
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, (path, proc.stderr)
    for path in [p for p in report.written if p.endswith(".py")]:
        proc = subprocess.run([sys.executable, "-I", "-m", "py_compile", path], capture_output=True, text=True)
        assert proc.returncode == 0, (path, proc.stderr)


def test_workflow_saved_in_project(tmp_path):
    project = Project()
    project.workflow.project_name = "apt1"
    b = add_node(project.workflow, "amber-build", settings={"mg_count": 4})
    b.files["structure"] = "inputs/model.pdb"
    f = add_node(project.workflow, "input-file", settings={"path": "maps/map.mrc", "kind": "map"})
    connect(project.workflow, b.id, "topology", add_node(project.workflow, "amber-md").id, "topology")
    scratch_node(project.workflow, "vmd").settings["rna_style"] = "Tube"
    path = tmp_path / "plan.mdatk.toml"
    project.save(str(path))
    data = tomlio.load(str(path))
    assert data["workflow"]["nodes"][0]["settings"]["mg_count"] == 4
    loaded = Project.load(str(path))
    assert loaded.workflow.nodes[0].files["structure"] == str(tmp_path / "inputs" / "model.pdb")
    assert loaded.workflow.nodes[1].settings["path"] == str(tmp_path / "maps" / "map.mrc")
    assert len(loaded.workflow.edges) == 1 and loaded.workflow.scratch[0].settings["rna_style"] == "Tube"
    assert f.id == loaded.workflow.nodes[1].id


def test_cli_workflow(tmp_path, capsys, demo_dry):
    from mdatk.cli import main
    preset = tmp_path / "p.mdatk.toml"
    assert main(["init", str(preset)]) == 0
    raw_before = tomlio.load(str(preset))
    assert main(["workflow", "template", "amber", "-p", str(preset)]) == 0
    data = tomlio.load(str(preset))
    assert [n["tool"] for n in data["workflow"]["nodes"]] == ["amber-build", "amber-restraints", "amber-md",
                                                              "mdatk"]
    assert data["inputs"] == raw_before["inputs"]  # only the workflow part changed
    capsys.readouterr()
    assert main(["workflow", "check", "-p", str(preset)]) == 1
    assert main(["workflow", "tools", "amber-md"]) == 0
    out = capsys.readouterr().out
    assert "stages.<n>.<key>" in out and "pmemd.MPI" in out
    assert main(["workflow", "generate", "amber-md", "-o", str(tmp_path / "gen"), "--step", "4",
                 "--project-name", "apt1", "--stamp", STAMP, "-i", "topology=/x.parm7",
                 "-i", "coordinates=/x.rst7", "-i", "restraints=/x.dist", "--set", "engine=pmemd.MPI",
                 "--set", "stages.5.nstlim=1000", "-q"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert any(p.endswith("apt1_04_prod_20261015-1432.in") for p in res["written"])
    prod = open(os.path.join(res["folder"], "apt1_04_prod_20261015-1432.in")).read()
    assert "nstlim=1000," in prod
    assert main(["workflow", "generate", "amber-md", "--set", "nonsense=1"]) == 1
    out_dist = tmp_path / "helix.dist"
    assert main(["workflow", "restraints", "--top", demo_dry["topology"], "--sequence",
                 "GCAUGGCAUCGA UCGAUGCCAUGC", "--dotbracket", "(" * 12 + ")" * 12, "-o", str(out_dist)]) == 0
    assert out_dist.read_text().count("&rst") == 31


def test_mg_pairs_cover_cp_types_and_warn(tmp_path):
    plan, st = step("amber-build", {"mg_count": 2, "mg_model": "micro", "water": "spce"}, {"structure": "/x.pdb"})
    pm = generated(plan)["apt1_01_parmed_20261015-1432.in"].content
    o2 = [ln for ln in pm.splitlines() if ln.startswith("changeLJPair :MG @%O2 ")][0].split()[3:]
    op = [ln for ln in pm.splitlines() if ln.startswith("changeLJPair :MG @%OP ")][0].split()[3:]
    assert o2 == op and "changeLJPair :MG @%OR " in pm
    sh = generated(plan)["apt1_01_amber-build_20261015-1432.sh"].content
    assert "2>&1 | tee apt1_01_parmed_20261015-1432.log" in sh
    _plan, st = step("amber-build", {"mg_count": 2, "mg_model": "micro", "extra_libs": "LRNA.lib"},
                     {"structure": "/x.pdb"})
    assert any("need Mg²⁺ pair terms" in m for m in messages(st, "info"))
    _plan, st = step("amber-build", {"mg_count": 2, "mg_model": "nano", "mg_cl_pair": False},
                     {"structure": "/x.pdb"})
    assert any("not a published combination" in m for m in messages(st, "warning"))
    cp = tmp_path / "frcmod.cp"
    cp.write_text("CP correction\nMASS\nOP 16.00\nOR 16.00\n\nNONB\n  OP   1.7493  0.2100\n  OR   1.7718  0.1700\n")
    _plan, st = step("amber-build", {"cp_source": "frcmod", "cp_frcmod": str(cp)}, {"structure": "/x.pdb"})
    assert any("OP, OR" in m and "no effect" in m for m in messages(st, "warning"))


def test_amber_build_cp_library_and_tip4p2005():
    plan, st = step("amber-build", {"mg_count": 4, "mg_model": "nano", "water": "tip4p2005", "split_types": True},
                    {"structure": "/x.pdb"})
    assert not messages(st, "error"), messages(st)
    assert any("Na+ parameters exist for TIP4P/2005" in m for m in messages(st, "warning"))
    files = generated(plan)
    leap = files["apt1_01_tleap_20261015-1432.leap"].content.splitlines()
    assert leap[2:12] == ["source leaprc.RNA.OL3", "source leaprc.water.tip4pew", "loadamberparams frcmod.tip4p2005",
                          "loadOff tip4p2005box.off", "WAT = TIP4P2005", "HOH = TIP4P2005",
                          'addAtomTypes { { "LD" "O" "sp3" } { "LH" "O" "sp3" } { "LE" "O" "sp3" } }',
                          'addAtomTypes { { "OR" "O" "sp3" } }', "loadamberparams frcmod.cp", "loadOff RNA_CP.lib"]
    assert leap.index("loadOff RNA_CP.lib") < leap.index("loadamberparams apt1_01_mg_20261015-1432.frcmod")
    assert "solvateoct mol TIP4P2005BOX 15.0" in leap
    frc = files["apt1_01_mg_20261015-1432.frcmod"].content
    assert "  Cl-      2.82559616  0.01025024" in frc  # the authors' TIP4P/2005 Cl- (Table 2)
    pm = files["apt1_01_parmed_20261015-1432.in"].content
    assert "changeLJPair @%HO @%OP 2.73 0.01" in pm and "changeLJPair @%HO @%OR 3.03 0.01" in pm
    assert "changeLJPair @%H5 @%OR 2.8808 0.0505" in pm and "changeLJPair @%H4 @%OR 2.9308 0.0505" in pm
    assert "changeLJPair @%HO @%OS" not in pm and "addLJType @%O2" not in pm
    sh = files["apt1_01_amber-build_20261015-1432.sh"].content
    order = [sh.index(x) for x in ('AMBERHOME="${AMBERHOME:-', 'python3 apt1_01_cp-lib_20261015-1432.py "$AMBERHOME" .',
                                   'python3 apt1_01_tip4p2005_20261015-1432.py "$AMBERHOME" .', "tleap -f")]
    assert order == sorted(order)
    for name in ("apt1_01_cp-lib_20261015-1432.py", "apt1_01_tip4p2005_20261015-1432.py"):
        compile(files[name].content, name, "exec")
    _plan, st = step("amber-build", {"mg_count": 2, "water": "tip4p2005"}, {"structure": "/x.pdb"})
    warn = " ".join(messages(st, "warning"))
    assert "TIP4P-Ew Li & Merz" in warn and "Joung-Cheatham" in warn


def test_cp_retype_helper_on_ambertools(tmp_path):
    from mdatk.workflows.scripts import cp_retype
    home = os.environ.get("MDATK_TEST_AMBERHOME") or os.environ.get("AMBERHOME", "")
    lib = os.path.join(home, "dat", "leap", "lib", "RNA.lib")
    if not home or not os.path.isfile(lib):
        pytest.skip("AmberTools data not available (set MDATK_TEST_AMBERHOME)")
    assert cp_retype.main([home, str(tmp_path)]) == 0
    text = (tmp_path / "RNA_CP.lib").read_text()
    assert '"OP1" "OP"' in text and '"O5\'" "OR"' in text and '"O4\'" "OS"' in text
    assert '"O5\'" "OH"' in text  # 5'-terminal hydroxyl untouched
    frc = (tmp_path / "frcmod.cp").read_text()
    assert "OP-P   525.0    1.480" in frc and "OR-P   230.0    1.610" in frc
    assert "  OP       1.7493  0.2100" in frc and "  OR       1.7718  0.1700" in frc


def test_amber_1264_with_cp_library_and_two_pass_ranges():
    plan, st = step("amber-build", {"mg_count": 2, "mg_model": "1264", "water": "tip3p"}, {"structure": "/x.pdb"})
    pm = generated(plan)["apt1_01_parmed_20261015-1432.in"].content
    assert "add12_6_4 :MG watermodel TIP3P polfile lj_1264_pol_cp.dat" in pm
    from mdatk.workflows.scripts import cp_retype
    pol = cp_retype.pol_table("O2    0.569    x\nOS    0.637    y\n")
    assert "OP    0.569" in pol and "OR    0.637" in pol
    plan, st = step("amber-build", {"two_pass": True, "lrna_ranges": "C:50-52", "extra_libs": "LRNA.lib"},
                    {"structure": "/x.pdb"})
    sh = generated(plan)["apt1_01_amber-build_20261015-1432.sh"].content
    dry = [ln for ln in sh.splitlines() if "clean apt1_01_ambpdb" in ln][0]
    assert "--no-detect" in dry and "--l-ranges" not in dry
    md = TOOLS["amber-md"]
    assert "cpptraj" not in md.resources(md.normalise({"dry_pdb": False, "dry_traj": False}))


def test_builder_and_cryoem_templates_write_valid_plans(tmp_path):
    from mdatk.builder import build_helix
    from mdatk.density import simulate_map
    from mdatk.io.mrc import write_map
    from mdatk import pdbtools
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    helix, build, restraints, md = apply_template(wf, "builder-amber")
    helix.settings.update(sequence="GGCAUC")  # the restraints get the duplex pairs from the builder
    pdb = tmp_path / "model.pdb"
    pdb.write_text(build_helix("GGCAUC", chirality="L", scheme="ccd"))
    s = pdbtools.read_pdb(str(pdb))
    xyz = [a.xyz for r in s.residues() for a in r.atoms]
    import numpy as np
    mp = tmp_path / "map.mrc"
    write_map(str(mp), simulate_map(np.array(xyz), np.ones(len(xyz)), 6.0))
    model, emap, dicts, phenix, coot = apply_template(wf, "cryoem-refine")
    model.settings["path"], emap.settings["path"] = str(pdb), str(mp)
    phenix.settings["resolution"] = 6.0
    plan = Plan(wf, str(tmp_path / "plan"))
    errors = [(plan.steps[n].label, i.message) for n, i in plan.all_issues() if i.level == "error"]
    assert errors == []
    report = plan.write()
    scripts = [p for p in report.written if p.endswith(".sh")]
    assert len(scripts) >= 7
    dist = [p for p in report.written if p.endswith(".dist")] or [p for p in report.written if "restraint" in p]
    assert dist and any(p.endswith(".secstruct") for p in report.written)
    for path in scripts:
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, (path, proc.stderr)



def test_lrna_force_field_fixes_are_automatic():
    """NBfix19 / 0BPh pairs for the L-RNA bridging-oxygen type and its atom types for tleap, as in the
    validated hand-written L-parmed_0BPh_NBfix19.in."""
    plan, st = step("amber-build", {"mg_count": 2}, {"structure": "/x.pdb"})
    files = generated(plan)
    parmed = next(f.content for n, f in files.items() if n.endswith(".in") and "parmed" in n)
    for line in ("changeLJPair @%HO @%LD 3.03 0.01", "changeLJPair @%H5 @%LD 2.8808 0.0505",
                 "changeLJPair @%H4 @%LD 2.9308 0.0505"):
        assert line in parmed
    assert "are not included" not in parmed
    leap = next(f.content for n, f in files.items() if n.endswith(".leap"))
    assert 'addAtomTypes { { "LD" "O" "sp3" } { "LH" "O" "sp3" } { "LE" "O" "sp3" } }' in leap
    # only NBfix19: no 0BPh pairs; fixes off: the old note, no declaration
    plan, _ = step("amber-build", {"bph0": False}, {"structure": "/x.pdb"})
    parmed = next(f.content for n, f in generated(plan).items() if n.endswith(".in") and "parmed" in n)
    assert "@%HO @%LD" in parmed and "@%H5 @%LD" not in parmed
    plan, _ = step("amber-build", {"lrna_fixes": False}, {"structure": "/x.pdb"})
    files = generated(plan)
    parmed = next(f.content for n, f in files.items() if n.endswith(".in") and "parmed" in n)
    assert "\nchangeLJPair @%HO @%LD" not in parmed and "are not included" in parmed
    assert "LD" not in next(f.content for n, f in files.items() if n.endswith(".leap"))
    _, st = step("amber-build", {"lrna_types": "L?D"}, {"structure": "/x.pdb"})
    assert any("L-RNA atom types" in m for m in messages(st, "error"))


@pytest.mark.parametrize("settings, level, expected", [
    ({"lrna_types": "LD LH LX"}, "error", "LX has no Mg²⁺ pair reference"),
    ({"lrna_types": "LD LH LX:OS"}, "info", "LX as OS"),
    ({"lrna_types": "LD LH", "mg_types": "LX:QQ"}, "error", "LX:QQ"),
    ({"lrna_types": "LD LH LX", "mg_types": "LX:OR"}, "info", "LX as OR"),
    ({"mg_types": "XO:OS XH:OH"}, "info", "XO as OS, XH as OH"),
    ({"parmed_extra": "changeLJPair :MG @%LH 2.6 1.1"}, "warning", "Mg²⁺ pairs of LH itself"),
])
def test_mg_pair_type_mapping_checks(settings, level, expected):
    _plan, st = step("amber-build", dict({"mg_count": 2, "mg_model": "micro"}, **settings), {"structure": "/x.pdb"})
    assert any(expected in m for m in messages(st, level)), messages(st)


def test_mg_pair_type_mapping_lines():
    plan, st = step("amber-build", {"mg_count": 2, "mg_model": "micro", "water": "tip3p",
                                    "lrna_types": "LD LH LE LX:OR", "mg_types": "XO:OS"}, {"structure": "/x.pdb"})
    assert not messages(st, "error"), messages(st)
    files = generated(plan)
    pm = files["apt1_01_parmed_20261015-1432.in"].content
    ref = mg_params.amber_pair(*mg_params.get_set("micro", "tip3p").rna["OS"])
    for t in ("LD", "LE", "LX", "XO"):
        assert f"changeLJPair :MG @%{t} {ref[0]:.8f} {ref[1]:.8f}" in pm
    assert all("#" not in ln for ln in pm.splitlines() if ln.startswith("changeLJPair"))
    leap = files["apt1_01_tleap_20261015-1432.leap"].content
    assert 'addAtomTypes { { "LD" "O" "sp3" } { "LH" "O" "sp3" } { "LE" "O" "sp3" } { "LX" "O" "sp3" } }' in leap
    assert "changeLJPair @%HO @%LD 3.03 0.01" in pm  # NBfix19 still uses the first (bridging) type


def test_amber_build_topology_check_step():
    plan, st = step("amber-build", {"mg_count": 4, "mg_model": "nano", "water": "tip3p"}, {"structure": "/x.pdb"})
    files = generated(plan)
    chk = files["apt1_01_check_20261015-1432.py"].content
    compile(chk, "check.py", "exec")
    p = mg_params.get_set("nano", "tip3p")
    assert f"ION_LJ = ({round(p.rmin_half, 6)!r}, {round(p.eps_kcal, 6)!r})" in chk
    assert "PAIRS = True" in chk and "NEUTRAL = True" in chk and "MG_COUNT = 4" in chk
    sh = files["apt1_01_amber-build_20261015-1432.sh"].content
    assert ("python3 apt1_01_check_20261015-1432.py apt1_01_parmed_20261015-1432.parm7 | tee "
            "apt1_01_check_20261015-1432.log ||") in sh
    assert sh.index("parmed -O") < sh.index("python3 apt1_01_check")
    plain = generated(step("amber-build", {"mg_count": 4, "mg_balance": False}, {"structure": "/x.pdb"})[0])
    chk = plain["apt1_01_check_20261015-1432.py"].content
    assert "PAIRS = False" in chk and "ION_LJ = None" in chk and "NEUTRAL = False" in chk
    off = generated(step("amber-build", {"check_topology": False}, {"structure": "/x.pdb"})[0])
    assert not any("_check_" in n for n in off)
    dry = generated(step("amber-build", {"water": "none"}, {"structure": "/x.pdb"})[0])
    assert not any("_check_" in n for n in dry)


def test_mg_pairs_without_lrna_naming_and_library_types(tmp_path):
    # the input PDB already has L names: the L-RNA types still get their Mg2+ pairs
    plan, st = step("amber-build", {"mg_count": 2, "mg_model": "micro", "lrna": False}, {"structure": "/x.pdb"})
    pm = generated(plan)["apt1_01_parmed_20261015-1432.in"].content
    assert "changeLJPair :MG @%LD " in pm and "changeLJPair :MG @%LE " in pm
    assert "\nchangeLJPair @%HO @%LD 3.03 0.01" in pm  # NBfix19 L pairs too
    lib = tmp_path / "LRNA.lib"
    lib.write_text('!!index array str\n "LA"\n!entry.LA.unit.atoms table  str name  str type  int typex  '
                   'int resx  int flags  int seq  int elmnt  dbl chg\n "P" "P" 0 1 131072 1 15 1.1662\n'
                   ' "O5\'" "LD" 0 1 131072 2 8 -0.4989\n "O4\'" "LQ" 0 1 131072 3 8 -0.3548\n'
                   ' "H1\'" "H2" 0 1 131072 4 1 0.2\n!entry.LA.unit.atomspertinfo table  str pname\n "P" "P"\n')
    _p, st = step("amber-build", {"mg_count": 2, "mg_model": "micro", "extra_libs": str(lib)}, {"structure": "/x.pdb"})
    warn = [m for m in messages(st, "warning") if "no Mg²⁺ pair reference" in m]
    assert warn and "LQ" in warn[0] and "LD" not in warn[0] and "pname" not in warn[0], messages(st)
    _p, st = step("amber-build", {"mg_count": 2, "mg_model": "micro", "extra_libs": str(lib), "mg_types": "LQ:OS"},
                  {"structure": "/x.pdb"})
    assert not [m for m in messages(st, "warning") if "no Mg²⁺ pair reference" in m]
    chk = generated(step("amber-build", {"mg_count": 2, "mg_model": "micro", "mg_types": "LQ:OS"},
                         {"structure": "/x.pdb"})[0])["apt1_01_check_20261015-1432.py"].content
    ref = [round(v, 6) for v in mg_params.amber_pair(*mg_params.get_set("micro", "spce").rna["OS"])]
    assert "PAIR_REFS = {" in chk and f"'LQ': {ref!r}" in chk and f"'OS': {ref!r}" in chk
