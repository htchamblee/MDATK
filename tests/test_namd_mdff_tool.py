"""The 'namd-mdff' planner step: generated files, planner checks and a full run of the generated script
with stand-in programs (namd3, sander, cpptraj) - no NAMD or AmberTools installation needed."""

import json
import os
import re
import shutil
import stat
import subprocess
import sys

import pytest

import mdff_system as ms
from mdatk.config import PlanEdge, PlanNode, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan
from mdatk.workflows.scripts import namd_mdff as nm

STAMP = "20261015-1432"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# stock VMD stereochemistry / MDFF plugin commands that assume D-nucleotides (or would re-derive what
# MDATK writes): none of them may appear in anything the step generates
STOCK_VMD = re.compile(r"^\s*(package\s+require\s+(mdff|chirality|cispeptide|ssrestraints|autopsf)|"
                       r"(chirality|cispeptide|ssrestraints)\s+\w+|mdff\s+(setup|gridpdb|griddx|check|sim)\b)",
                       re.MULTILINE)


def single(settings=None, files=None, data=None):
    tool = TOOLS["namd-mdff"]
    node = PlanNode(id="m", tool="namd-mdff", step=2, settings=tool.normalise(settings or {}),
                    files=dict(files or {"topology": "/d/sys.parm7", "coordinates": "/d/sys.rst7",
                                         "map": "/d/map.mrc", "model": "/d/model.pdb"}))
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node])
    plan = Plan(wf, "/tmp/mdatk-plan")
    if data is not None:  # what an Amber build would report about its topology
        plan.steps["m"].ctx.inputs["topology"].data.update(data)
        plan.steps["m"].issues = tool.check(plan.steps["m"].settings, plan.steps["m"].ctx)
    return plan, plan.steps["m"]


def messages(step, level=None):
    return [i.message for i in step.issues if level is None or i.level == level]


def built_after_amber(build_settings, mdff_settings=None, two_pass=False, coordinates=None):
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    b = PlanNode(id="b", tool="amber-build", step=1, settings=TOOLS["amber-build"].normalise(
        dict(build_settings, two_pass=two_pass)), files={"structure": "/d/model.pdb"})
    m = PlanNode(id="m", tool="namd-mdff", step=2, settings=TOOLS["namd-mdff"].normalise(mdff_settings or {}),
                 files={"map": "/d/map.mrc", "model": "/d/model.pdb"})
    wf.nodes = [b, m]
    out = "dry_topology" if two_pass else "topology"
    crd = coordinates or ("dry_pdb" if two_pass else "coordinates")
    wf.edges = [PlanEdge(source="b", output=out, target="m", input="topology"),
                PlanEdge(source="b", output=crd, target="m", input="coordinates")]
    plan = Plan(wf, "/tmp/mdatk-plan")
    return plan, plan.steps["m"]


def test_generated_configuration_uses_the_amber_topology_natively():
    plan, st = built_after_amber({"mg_count": 2, "mg_model": "nano", "water": "tip3p", "hmr": True},
                                 {"sequence": "GCGC GCGC", "dotbracket": "(((( ))))"})
    assert not messages(st, "error"), messages(st)
    files = {f.name: f for f in plan.generate(["m"])["m"]}
    fit = files[f"apt1_02_fit_{STAMP}.namd"].content
    for line in ("amber            on", "parmfile         ../01_amber-build/apt1_01_parmed_" + STAMP + ".parm7",
                 "readexclusions   yes", "exclude          scaled1-4", "1-4scaling       0.833333",
                 "scnb             2.0", "switching        off", "PME              yes", "LJcorrection     on",
                 "rigidBonds       all", "timestep         4.0", "mgridforce       on",
                 f"mgridforcepotfile mdff apt1_02_potential_{STAMP}.dx", "mgridforcescale  mdff 0.3 0.3 0.3",
                 "extraBonds       on", f"extraBondsFile   apt1_02_chirality_{STAMP}.txt",
                 f"extraBondsFile   apt1_02_ssdihedrals_{STAMP}.txt", "colvars          on", "run 250000"):
        assert line in fit, line
    assert "structure" not in fit.split("\n# MDFF")[0].lower().replace("structure.", "")  # no PSF / CHARMM files
    assert "paraTypeCharmm" not in fit and "parameters " not in fit
    ramp = [files[f"apt1_02_ramp-{k:03d}_{STAMP}.namd"].content for k in range(1, 6)]
    assert [re.search(r"mgridforcescale  mdff (\S+)", r).group(1) for r in ramp] == ["0.05", "0.1125", "0.175",
                                                                                     "0.2375", "0.3"]
    assert "binvelocities    apt1_02_ramp-001" in ramp[1] and "temperature      300.0" in ramp[0]
    final = files[f"apt1_02_final-min_{STAMP}.namd"].content
    assert "mgridforcescale  mdff 10 10 10" in final and "minimize 2000" in final
    energy = files[f"apt1_02_energy_{STAMP}.namd"].content
    assert "LJcorrection     off" in energy and "rigidBonds       none" in energy and "mgridforce" not in energy
    assert "vdwmeth=0" in files[f"apt1_02_energy_{STAMP}.mdin"].content
    helper = files[f"apt1_02_mdff_{STAMP}.py"].content
    compile(helper, "mdff.py", "exec")
    compile(files[f"apt1_02_energy_{STAMP}.py"].content, "energy.py", "exec")
    sh = files[f"apt1_02_namd-mdff_{STAMP}.sh"].content
    assert sh.index("prep |") < sh.index("sander -O") < sh.index("_min_") < sh.index("_fit_") < sh.index(" final")
    head = sh.split("set -eo pipefail")[0]
    assert "Phillips J. C." in head and "Trabuco L. G." in head and "Fiorin G., Klein M. L." in head
    for f in files.values():  # never VMD's D-only stereochemistry tools (nor its MDFF plugin)
        assert not STOCK_VMD.search(f.content), f.name


def test_no_colvars_citation_without_base_pairs():
    plan, st = built_after_amber({"water": "tip3p"})
    sh = [f for f in plan.generate(["m"])["m"] if f.role == "run"][0].content
    head = sh.split("set -eo pipefail")[0]
    assert "Trabuco" in head and "Klein M. L." not in head
    assert any("no secondary structure" in m for m in messages(st, "warning"))
    files = {f.name: f.content for f in plan.generate(["m"])["m"]}
    assert "colvars" not in files[f"apt1_02_fit_{STAMP}.namd"]


@pytest.mark.parametrize("build, mdff, level, text", [
    ({"water": "tip3p", "mg_model": "1264", "mg_count": 2}, {}, "error", "12-6-4 C4 terms"),
    ({"water": "opc", "mg_count": 0}, {}, "warning", "four-site water"),
    ({"water": "tip3p"}, {"timestep": "4"}, "error", "4 fs needs hydrogen mass repartitioning"),
    ({"water": "tip3p"}, {"solvent": "vacuum"}, "error", "vacuum runs need the dry topology"),
    ({"water": "tip3p"}, {"energy_check": False}, "warning", "without the energy comparison"),
    ({"water": "tip3p", "nbfix19": True}, {}, "info", "NBfix19 / 0BPh pairs"),
    ({"water": "tip3p"}, {"stages": [{"name": "a", "kind": "min"}, {"name": "a", "kind": "md"}]}, "error",
     "stage names must differ"),
])
def test_planner_checks(build, mdff, level, text):
    _plan, st = built_after_amber(build, mdff)
    assert any(text in m for m in messages(st, level)), messages(st)


def test_vacuum_run_on_the_dry_topology():
    plan, st = built_after_amber({"water": "tip3p"}, {"solvent": "vacuum", "gbis": True}, two_pass=True)
    assert not messages(st, "error"), messages(st)
    files = {f.name: f.content for f in plan.generate(["m"])["m"]}
    fit = files[f"apt1_02_fit_{STAMP}.namd"]
    assert "PME" not in fit and "extendedSystem" not in fit and "GBIS             on" in fit
    energy = files[f"apt1_02_energy_{STAMP}.namd"]
    assert "cutoff           999.0" in energy and "GBIS" not in energy
    assert "ntb=0, cut=999.0" in files[f"apt1_02_energy_{STAMP}.mdin"]


def test_vacuum_needs_dry_coordinates_too():
    _plan, st = built_after_amber({"water": "tip3p"}, {"solvent": "vacuum"}, two_pass=True,
                                  coordinates="coordinates")
    assert any("the coordinates belong to the solvated system" in m and "dry PDB" in m
               for m in messages(st, "error")), messages(st)


def test_pdb_coordinates_are_converted_in_topology_order(tmp_path):
    atoms = ms.atoms_of(ms.mixed_pdb())
    ms.write_topology(str(tmp_path / "t.parm7"), atoms)
    (tmp_path / "dry.pdb").write_text(ms.model_pdb(atoms))
    cfg = {"topology": str(tmp_path / "t.parm7"), "coordinates": str(tmp_path / "start.rst7")}
    assert nm.start_coordinates(cfg, str(tmp_path / "dry.pdb")) == 0
    xyz, box, _v = nm.read_rst7(str(tmp_path / "start.rst7"), len(atoms))
    assert box is None and max(abs(a - b) for p, q in zip(xyz, atoms) for a, b in zip(p, q[5])) < 0.001
    swapped = atoms[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    (tmp_path / "bad.pdb").write_text(ms.model_pdb(swapped))
    with pytest.raises(nm.PrepError, match="not in the topology's atom order"):
        nm.start_coordinates(cfg, str(tmp_path / "bad.pdb"))


def test_missing_frame_model_is_flagged_for_solvated_runs():
    _plan, st = single(files={"topology": "/d/sys.parm7", "coordinates": "/d/sys.rst7", "map": "/d/map.mrc"})
    assert any("tleap moves the solute" in m for m in messages(st, "warning"))


# ------------------------------------------------------------------------------- full run, stand-ins
FAKE_NAMD = r'''#!/usr/bin/env python3
import json, os, re, struct, sys
sys.path.insert(0, os.environ["MDATK_SRC"])
from mdatk.workflows.scripts import namd_mdff as nm
conf = [a for a in sys.argv[1:] if not a.startswith("+")][-1]
text = open(conf).read()
stock = re.compile(r"^\s*(package\s+require|chirality|ssrestraints|cispeptide|mdff)\b", re.M)
assert not stock.search(text), "stock VMD command in " + conf
opts = {}
for line in text.splitlines():
    line = line.split("#")[0].strip()
    if line:
        key, _, value = line.partition(" ")
        opts.setdefault(key.lower(), []).append(value.strip())
assert opts["amber"] == ["on"] and os.path.isfile(opts["parmfile"][0]) and os.path.isfile(opts["ambercoor"][0])
for key in ("extendedsystem", "bincoordinates", "binvelocities", "extrabondsfile", "colvarsconfig", "consref"):
    for v in opts.get(key, []):
        assert os.path.isfile(v), (key, v)
for v in opts.get("mgridforcepotfile", []) + opts.get("mgridforcefile", []):
    assert os.path.isfile(v.split()[1]), v
top = nm.Topology(opts["parmfile"][0])
out = opts["outputname"][0]
if "run" in opts and opts["run"][0] == "0":
    e = json.loads(os.environ["FAKE_NAMD_ENERGY"])
    print("ETITLE:      TS           BOND          ANGLE          DIHED          IMPRP               ELECT            VDW       BOUNDARY           MISC        KINETIC               TOTAL")
    print("ENERGY:       0 " + " ".join(str(e[k]) for k in ("BOND", "ANGLE", "DIHED", "IMPRP", "ELECT", "VDW")) + " 0 0 0 0")
    print("WallClock: 0.1  CPUTime: 0.1  Memory: 1 MB")
    sys.exit(0)
if "bincoordinates" in opts:
    xyz = nm.read_namd_coor(opts["bincoordinates"][0], top.natom)
else:
    xyz, _box, _vel = nm.read_rst7(opts["ambercoor"][0], top.natom)
with open(out + ".coor", "wb") as fh:
    fh.write(struct.pack("<i", top.natom) + struct.pack("<%dd" % (3 * top.natom), *[c for p in xyz for c in p]))
with open(out + ".vel", "wb") as fh:
    fh.write(struct.pack("<i", top.natom) + struct.pack("<%dd" % (3 * top.natom), *[0.0] * (3 * top.natom)))
if "extendedsystem" in opts:
    open(out + ".xsc", "w").write(open(opts["extendedsystem"][0]).read())
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(conf + "\n")
print("WallClock: 0.1  CPUTime: 0.1  Memory: 1 MB")
'''

FAKE_SANDER = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
e = json.loads(os.environ["FAKE_SANDER_ENERGY"])
with open(out, "w") as fh:
    fh.write("   NSTEP       ENERGY          RMS            GMAX         NAME    NUMBER\n")
    fh.write("      1      -1.0000E+03     1.0000E+00     1.0000E+01     C1'         5\n\n")
    fh.write(" BOND    = %14.4f  ANGLE   = %14.4f  DIHED      = %14.4f\n" % (e["BOND"], e["ANGLE"], e["DIHED"]))
    fh.write(" VDWAALS = %14.4f  EEL     = %14.4f  HBOND      =        0.0000\n" % (e["VDWAALS"], e["EEL"]))
    fh.write(" 1-4 VDW = %14.4f  1-4 EEL = %14.4f  RESTRAINT  =        0.0000\n\n" % (e["1-4 VDW"], e["1-4 EEL"]))
'''

FAKE_CPPTRAJ = r'''#!/usr/bin/env python3
import os, shutil, sys
args = sys.argv[1:]
shutil.copy(os.environ["FAKE_ASCII_RESTART"], args[args.index("-x") + 1])
print("cpptraj stand-in: converted")
'''

SANDER = {"BOND": 120.5, "ANGLE": 330.25, "DIHED": 410.0, "VDWAALS": -812.0, "EEL": -15020.0, "1-4 VDW": 95.5,
          "1-4 EEL": 4021.0}
NAMD_SAME = {"BOND": 120.5001, "ANGLE": 330.2501, "DIHED": 400.0, "IMPRP": 10.0, "VDW": -716.6, "ELECT": -10999.5}


def _bin(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("namd3", FAKE_NAMD), ("sander", FAKE_SANDER), ("cpptraj", FAKE_CPPTRAJ)):
        p = b / name
        p.write_text(text.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return b


def _system(tmp_path, netcdf=False):
    d = tmp_path / "data"
    d.mkdir()
    atoms = ms.atoms_of(ms.mixed_pdb())
    rot = ms.rotation((0.0, 0.0, 1.0), 0.0)
    moved = ms.with_solvent(ms.moved(atoms, rot, (30.0, 28.0, 26.0)), start=(10.0, 10.0, 55.0))
    box = (80.0, 80.0, 80.0, 90.0, 90.0, 90.0)
    ms.write_topology(str(d / "sys.parm7"), moved, box=box, hmr=True)
    ms.write_rst7(str(d / "ascii.rst7"), [a[5] for a in moved], box=box)
    if netcdf:
        (d / "sys.rst7").write_bytes(b"CDF\x02" + b"\0" * 32)
    else:
        shutil.copy(d / "ascii.rst7", d / "sys.rst7")
    ms.write_map(str(d / "map.mrc"), atoms)
    (d / "model.pdb").write_text(ms.model_pdb(atoms))
    return d, atoms


def _write(tmp_path, d, settings=None):
    tool = TOOLS["namd-mdff"]
    s = {"sequence": "GCGC GCGC", "dotbracket": "(((( ))))", "timestep": "4",
         "stages": [dict(name="min", kind="min", steps=100, gscale=0.3),
                    dict(name="ramp", kind="md", steps=40, gscale=0.1, gscale_end=0.3, segments=2,
                         positional=True),
                    dict(name="final-min", kind="min", steps=100, gscale=10.0)]}
    s.update(settings or {})
    node = PlanNode(id="m", tool="namd-mdff", step=1, settings=tool.normalise(s),
                    files={"topology": str(d / "sys.parm7"), "coordinates": str(d / "sys.rst7"),
                           "map": str(d / "map.mrc"), "model": str(d / "model.pdb")})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), str(tmp_path / "plan"))
    st = plan.steps["m"]
    assert not messages(st, "error"), messages(st)
    report = plan.write(["m"])
    return [p for p in report.written if p.endswith("namd-mdff_" + STAMP + ".sh")][0]


def _run(script, tmp_path, d, namd_energy):
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}", MDATK_SRC=ROOT,
               FAKE_NAMD_ENERGY=json.dumps(namd_energy), FAKE_SANDER_ENERGY=json.dumps(SANDER),
               FAKE_ASCII_RESTART=str(d / "ascii.rst7"), FAKE_LOG=str(tmp_path / "namd-runs.log"))
    return subprocess.run(["bash", script], capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("netcdf", [False, True])
def test_full_run_with_stand_in_programs(tmp_path, netcdf):
    _bin(tmp_path)
    d, atoms = _system(tmp_path, netcdf)
    script = _write(tmp_path, d)
    res = _run(script, tmp_path, d, NAMD_SAME)
    assert res.returncode == 0, res.stdout + res.stderr
    folder = os.path.dirname(script)
    assert "NAMD reproduces the sander energies" in res.stdout
    runs = (tmp_path / "namd-runs.log").read_text().split()
    assert [os.path.basename(r).split("_")[2] for r in runs] == ["min", "ramp-001", "ramp-002", "final-min"]
    fitted = os.path.join(folder, f"apt1_01_fitted_{STAMP}.pdb")
    back = [ln for ln in open(fitted).read().splitlines() if ln.startswith("ATOM")]
    assert len(back) == len(atoms)  # solute only, in the map's frame (the stand-in NAMD moved nothing)
    for ln, a in zip(back[::37], atoms[::37]):
        assert abs(float(ln[30:38]) - a[5][0]) < 0.002 and abs(float(ln[46:54]) - a[5][2]) < 0.002
    chir = open(os.path.join(folder, f"apt1_01_chirality_{STAMP}.txt")).read()
    assert chir.count("\nimproper ") == 64
    prep_log = open(os.path.join(folder, f"apt1_01_prep_{STAMP}.log")).read()
    assert "8 L" in prep_log and "repartitioned masses" in prep_log
    if netcdf:
        assert "cpptraj stand-in: converted" in open(os.path.join(folder, f"apt1_01_start_{STAMP}.log")).read()
    # run again: every NAMD run is skipped
    res = _run(script, tmp_path, d, NAMD_SAME)
    assert res.returncode == 0 and res.stdout.count("already finished - skipped") == 4


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_energy_mismatch_stops_before_fitting(tmp_path):
    _bin(tmp_path)
    d, _atoms = _system(tmp_path)
    script = _write(tmp_path, d)
    wrong = dict(NAMD_SAME, VDW=NAMD_SAME["VDW"] + 25.0)  # e.g. the off-diagonal LJ pairs not applied
    res = _run(script, tmp_path, d, wrong)
    assert res.returncode != 0
    assert "disagree on VDW" in res.stderr and "MISMATCH" in res.stdout
    assert not (tmp_path / "namd-runs.log").exists()  # no MDFF stage was started


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_preparation_problems_stop_the_script(tmp_path):
    _bin(tmp_path)
    d, atoms = _system(tmp_path)
    # an L residue of the topology named as D: its restraints would invert it -> stop
    top_atoms = ms.with_solvent(ms.moved(atoms, ms.rotation((0, 0, 1), 0.0), (30.0, 28.0, 26.0)),
                                start=(10.0, 10.0, 55.0))
    top_atoms = [a if not (a[0] == "C" and a[1] == 2) else a[:2] + ("G",) + a[3:] for a in top_atoms]
    script = _write(tmp_path, d)  # written while the topology was fine
    ms.write_topology(str(d / "sys.parm7"), top_atoms, box=(80.0, 80.0, 80.0, 90.0, 90.0, 90.0), hmr=True)
    # the planner sees the problem as soon as the files exist ...
    tool = TOOLS["namd-mdff"]
    node = PlanNode(id="m", tool="namd-mdff", step=1, settings=tool.normalise({}),
                    files={"topology": str(d / "sys.parm7"), "coordinates": str(d / "sys.rst7"),
                           "map": str(d / "map.mrc"), "model": str(d / "model.pdb")})
    st = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), str(tmp_path / "x")).steps["m"]
    assert any("residue 10 (G) is named as D but its sugar has L geometry" in m for m in messages(st, "error"))
    # ... and the script stops before NAMD runs
    res = _run(script, tmp_path, d, NAMD_SAME)
    assert res.returncode != 0 and "is named as D but its sugar has L geometry" in res.stderr
    assert not (tmp_path / "namd-runs.log").exists()


def test_energy_parsers_and_comparison(tmp_path):
    from mdatk.workflows.scripts import namd_energy as ne
    mdout = tmp_path / "e.mdout"
    mdout.write_text("""
   NSTEP       ENERGY          RMS            GMAX         NAME    NUMBER
      1      -3.6046E+04     1.6386E+01     1.1347E+02     O2P       215

 BOND    =      123.1234  ANGLE   =      234.5678  DIHED      =      345.6789
 VDWAALS =     1234.5678  EEL     =   -45678.1234  HBOND      =        0.0000
 1-4 NB  =      123.4567  1-4 EEL =     1234.5678  RESTRAINT  =        0.0000


   NSTEP       ENERGY          RMS            GMAX         NAME    NUMBER
      2      -3.7000E+04     1.0E+01     1.0E+02     O2P       215

 BOND    =      999.0000  ANGLE   =      234.5678  DIHED      =      345.6789
""")
    t = ne.sander_terms(str(mdout))
    assert t["BOND"] == 123.1234 and t["1-4 VDW"] == 123.4567 and t["EEL"] == -45678.1234  # the first step
    log = tmp_path / "namd.log"
    log.write_text("Info: something\nETITLE:      TS           BOND          ANGLE          DIHED          IMPRP"
                   "               ELECT            VDW\nENERGY:       0       123.1235       234.5679       340.0"
                   "            5.6789         -44443.5556      1358.0235\n")
    n = ne.namd_terms(str(log))
    rows = ne.compare(t, n, ne.TOLERANCES)
    assert [r[0] for r in rows] == ["BOND", "ANGLE", "DIHED", "VDW", "ELECT"] and all(r[5] for r in rows)
    rows = ne.compare(t, dict(n, VDW=n["VDW"] + 5.0), ne.TOLERANCES)
    assert [r[0] for r in rows if not r[5]] == ["VDW"]
    mdout.write_text(mdout.read_text().replace("1234.5678  EEL", "**********  EEL", 1))
    with pytest.raises(ValueError, match=r"\*\*\*\*\* for VDWAALS"):
        ne.sander_terms(str(mdout))
