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
                 f"extraBondsFile   apt1_02_ssdihedrals_{STAMP}.txt",
                 f"extraBondsFile   apt1_02_basepairs_{STAMP}.txt", "run 250000"):
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
    assert "Phillips J. C." in head and "Trabuco L. G." in head  # NAMD, MDFF
    assert "colvars" not in fit.lower()  # base pairs are extraBonds walls
    for f in files.values():  # never VMD's D-only stereochemistry tools (nor its MDFF plugin)
        assert not STOCK_VMD.search(f.content), f.name


def test_no_base_pair_files_without_secondary_structure():
    plan, st = built_after_amber({"water": "tip3p"})
    assert any("no secondary structure" in m for m in messages(st, "warning"))
    files = {f.name: f.content for f in plan.generate(["m"])["m"]}
    fit = files[f"apt1_02_fit_{STAMP}.namd"]
    assert "basepairs" not in fit and "ssdihedrals" not in fit and f"apt1_02_chirality_{STAMP}.txt" in fit


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
    assert "cutoff           999.0" in energy and "GBIS" not in energy and "switching        off" in energy
    assert "ntb=0, cut=999.0" in files[f"apt1_02_energy_{STAMP}.mdin"]
    # with switching off NAMD sums plain truncated Coulomb terms, as sander does without a cutoff: the
    # vacuum comparison is held to the same tolerances as the periodic one
    ns = {"__name__": "energy"}
    exec(compile(files[f"apt1_02_energy_{STAMP}.py"], "energy.py", "exec"), ns)
    from mdatk.workflows.scripts import namd_energy as ne
    assert ns["TOLERANCES"] == ne.TOLERANCES


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
import json, math, os, re, struct, sys
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
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1]  # NAMD strips braces, as Tcl does
        opts.setdefault(key.lower(), []).append(value)
assert opts["amber"] == ["on"] and os.path.isfile(opts["parmfile"][0]) and os.path.isfile(opts["ambercoor"][0])
for key in ("extendedsystem", "bincoordinates", "binvelocities", "extrabondsfile", "consref"):
    for v in opts.get(key, []):
        assert os.path.isfile(v), (key, v)
for v in opts.get("mgridforcepotfile", []) + opts.get("mgridforcefile", []):
    assert os.path.isfile(v.split()[1]), v
for v in opts.get("mgridforcepotfile", []):  # gridforcechecksize (GridForceGrid.C fits_lattice), written anew
    if "extendedsystem" not in opts:
        continue
    head = open(v.split()[1]).read(4000)
    n = [int(x) for x in re.search(r"counts\s+(\d+)\s+(\d+)\s+(\d+)", head).groups()]
    o = [float(x) for x in re.search(r"\norigin\s+(\S+)\s+(\S+)\s+(\S+)", head).groups()]
    d = [float(row.split()[k + 1]) for k, row in enumerate(re.findall(r"\ndelta[^\n]*", head))]
    c = [float(x) for x in open(opts["extendedsystem"][0]).read().split("\n")[2].split()[1:10]]
    a1, a2, a3 = c[0:3], c[3:6], c[6:9]
    cross = lambda u, w: (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
    vol = sum(x * y for x, y in zip(a1, cross(a2, a3)))
    recip = [[x / vol for x in cross(a2, a3)], [x / vol for x in cross(a3, a1)], [x / vol for x in cross(a1, a2)]]
    centre = [o[k] + d[k] * (n[k] - 1) / 2 for k in range(3)]
    ends = [(o[k] - d[k], o[k] + n[k] * d[k]) for k in range(3)]  # one border voxel on every side
    for corner in ((x, y, z) for x in ends[0] for y in ends[1] for z in ends[2]):
        diff = [corner[k] - centre[k] for k in range(3)]
        if any(math.floor(0.5 + sum(b[k] * diff[k] for k in range(3))) for b in recip):
            print("FATAL ERROR: Periodic cell basis too small for Gridforce grid 0.")
            sys.exit(1)
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


def _system(tmp_path, netcdf=False, folder="data"):
    d = tmp_path / folder
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
    # completed runs stay in place: the tests check which finished runs the script reuses (since 0.0.12 a
    # completed run moves aside when its step is written again, unless the project option is off)
    report = plan.write(["m"], supersede=False)
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_resume_reuses_only_runs_made_from_the_same_inputs(tmp_path):
    _bin(tmp_path)
    d, atoms = _system(tmp_path)
    script = _write(tmp_path, d)
    folder = os.path.dirname(script)
    log = tmp_path / "namd-runs.log"

    def run():
        if log.exists():
            log.unlink()
        res = _run(script, tmp_path, d, NAMD_SAME)
        assert res.returncode == 0, res.stdout + res.stderr
        return [os.path.basename(r).split("_")[2] for r in log.read_text().split()] if log.exists() else []

    def first_atom():
        line = next(ln for ln in open(os.path.join(folder, f"apt1_01_fitted_{STAMP}.pdb")) if ln.startswith("ATOM"))
        return float(line[30:38]), float(line[38:46]), float(line[46:54])

    every = ["min", "ramp-001", "ramp-002", "final-min"]
    assert run() == every
    assert run() == []  # nothing changed: every run is reused
    # new start coordinates of the same system (e.g. a new MD restart), 6 A further along x: the finished runs
    # belong to the old coordinates and the old map frame, so all of them run again - and the fitted model
    # stays on the map
    top = nm.Topology(str(d / "sys.parm7"))
    xyz, box, _vel = nm.read_rst7(str(d / "sys.rst7"), top.natom)
    ms.write_rst7(str(d / "sys.rst7"), [(x + 6.0, y, z) for x, y, z in xyz], box=box)
    assert run() == every
    assert first_atom() == pytest.approx(atoms[0][5], abs=0.002)
    # a changed stage reruns from that stage on
    stages = [dict(name="min", kind="min", steps=100, gscale=0.3),
              dict(name="ramp", kind="md", steps=40, gscale=0.1, gscale_end=0.3, segments=2, positional=True),
              dict(name="final-min", kind="min", steps=200, gscale=10.0)]
    script = _write(tmp_path, d, {"stages": stages})
    assert run() == ["final-min"]
    # a changed restraint file reruns everything
    script = _write(tmp_path, d, {"stages": stages, "chirality_k": 150.0})
    assert run() == every
    # a run whose output is gone runs again, and so does every run after it
    os.remove(os.path.join(folder, f"apt1_01_ramp-001_{STAMP}.coor"))
    assert run() == ["ramp-001", "ramp-002", "final-min"]
    assert run() == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_input_paths_with_spaces(tmp_path):
    _bin(tmp_path)
    d, atoms = _system(tmp_path, netcdf=True, folder="my project data")
    script = _write(tmp_path, d)
    conf = open(script.replace(f"namd-mdff_{STAMP}.sh", f"min_{STAMP}.namd")).read()
    assert f"parmfile         {{{d / 'sys.parm7'}}}" in conf  # braces: one NAMD (Tcl) word
    res = _run(script, tmp_path, d, NAMD_SAME)
    assert res.returncode == 0, res.stdout + res.stderr
    assert os.path.getsize(script.replace(f"namd-mdff_{STAMP}.sh", f"fitted_{STAMP}.pdb")) > 0


@pytest.mark.parametrize("stages, text", [
    ([dict(name="min", kind="min"), dict(name="energy", kind="min")], "stage name energy is taken by the step's own"),
    ([dict(name="Prep", kind="min")], "stage name Prep is taken"),
    ([dict(name="cell", kind="md", steps=100), dict(name="start", kind="min")], "stage names cell, start are taken"),
    ([dict(name="ramp", kind="md", steps=100, segments=2), dict(name="ramp-002", kind="min")],
     "run names repeat (ramp-002)"),
])
def test_run_names_must_not_meet_other_files(stages, text):
    _plan, st = single({"stages": stages})
    assert any(text in m for m in messages(st, "error")), messages(st)


def test_hand_picked_topology_is_read_for_four_site_water_and_hmr(tmp_path):
    atoms = ms.with_solvent(ms.atoms_of(ms.mixed_pdb()))
    box = (80.0, 80.0, 80.0, 90.0, 90.0, 90.0)
    ms.write_topology(str(tmp_path / "opc.parm7"), atoms, box=box, hmr=True, numextra=6)
    ms.write_rst7(str(tmp_path / "opc.rst7"), [a[5] for a in atoms], box=box)
    files = {"topology": str(tmp_path / "opc.parm7"), "coordinates": str(tmp_path / "opc.rst7"),
             "map": "/d/map.mrc", "model": "/d/model.pdb"}
    plan, st = single(files=files)
    assert not any("four-site" in m for m in messages(st, "error")), messages(st)
    assert any("the topology has four-site water (extra points)" in m for m in messages(st, "warning"))
    assert any("time step 4 fs (HMR topology)" in m for m in messages(st, "info"))
    fit = {f.name: f.content for f in plan.generate(["m"])["m"]}[f"apt1_02_fit_{STAMP}.namd"]
    assert "watermodel       tip4" in fit and "timestep         4.0" in fit
    # three-site water, no HMR: neither
    ms.write_topology(str(tmp_path / "opc.parm7"), atoms, box=box)
    plan, st = single(files=files)
    fit = {f.name: f.content for f in plan.generate(["m"])["m"]}[f"apt1_02_fit_{STAMP}.namd"]
    assert "watermodel" not in fit and "timestep         2.0" in fit
    _plan, st = single({"timestep": "4"}, files=files)
    assert any("4 fs needs hydrogen mass repartitioning" in m for m in messages(st, "error"))


def test_coupling_warnings_follow_each_stage():
    tool = TOOLS["namd-mdff"]
    stages = tool.normalise({})["stages"]
    stages[-1]["chirality"] = False  # final-min at xi = 10 without chirality restraints
    _plan, st = single({"stages": stages})
    assert any("without chirality restraints in stage final-min can invert" in m for m in messages(st, "warning"))
    ramp = [dict(name="ramp", kind="md", steps=1000, gscale=0.05, gscale_end=5.0, segments=5, chirality=False)]
    _plan, st = single({"stages": ramp})
    assert any("in stage ramp can invert" in m for m in messages(st, "warning")), messages(st)
    assert tool.describe(st.settings).endswith("ξ 0.05–5")
    one = [dict(name="fit", kind="md", steps=1000, gscale=0.1, gscale_end=0.5)]
    _plan, st = single({"stages": one})
    assert any("ξ stays at 0.1 - a ramp to 0.5 needs at least 2 segments" in m for m in messages(st, "warning"))
    assert tool.describe(st.settings).endswith("ξ 0.1–0.1")
    _plan, st = single()
    assert not any("can invert" in m for m in messages(st, "warning"))


def test_cpptraj_is_cited():
    plan, st = single({"energy_check": False})
    assert TOOLS["namd-mdff"].resources(st.settings) == ["namd", "mdff", "cpptraj"]
    sh = {f.name: f.content for f in plan.generate(["m"])["m"]}[f"apt1_02_namd-mdff_{STAMP}.sh"]
    assert "cpptraj -p" in sh and "Roe D. R." in sh.split("set -eo pipefail")[0]


def test_example_generator(tmp_path, monkeypatch):
    # examples/namd_mdff is made by this generator: it must run, keep everything in its folder and leave
    # no path of the machine it ran on
    from mdatk.config import Project
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "example"
    ms.make_example("example")  # relative, as documented
    assert os.listdir(tmp_path) == ["example"]
    prepared = os.listdir(out / "prepared")
    for tag in ("potential", "coupling", "chirality", "ssdihedrals", "basepairs", "cell", "prep", "start"):
        assert any(f"_{tag}_" in f for f in prepared), tag
    assert not any("_posres_" in f for f in prepared)  # no stage of the example project uses them
    for root, _dirs, files in os.walk(out):
        for f in files:
            if not f.endswith(".mrc"):
                assert str(tmp_path) not in open(os.path.join(root, f), errors="replace").read(), f
    log = next(f for f in prepared if f.endswith(".log"))
    assert "16 nucleotides, 8 L" in (out / "prepared" / log).read_text()
    project = Project.load(str(out / "namd_mdff.mdatk.toml"))
    plan = Plan(project.workflow, str(out / "generated"))
    assert not messages(plan.steps["mdff"], "error"), messages(plan.steps["mdff"])
    # the L-RNA library and parameters are generated by the build step, as the example's README says
    assert not messages(plan.steps["build"], "error"), messages(plan.steps["build"])
    assert any("generates the L-RNA library" in m for m in messages(plan.steps["build"], "info"))
    assert "L-RNA residue library" in (out / "README.md").read_text()
    # the shipped example is what the generator makes now: the same files, and the same text wherever it does
    # not depend on floating-point results (maps, potential, coordinates may differ in the last digit elsewhere)
    shipped = os.path.join(ROOT, "examples", "namd_mdff")

    def listing(top):
        return sorted(os.path.relpath(os.path.join(r, f), top) for r, _d, fs in os.walk(top) for f in fs
                      if not f.endswith(".lock"))

    assert listing(shipped) == listing(str(out)), "regenerate: python3 tests/mdff_system.py --example examples/namd_mdff"
    def undated(text):  # script headers say when they were written ("... written by MDATK x on 2026-10-30")
        return re.sub(r"(written by \S+ \S+) on \d{4}-\d{2}-\d{2}", r"\1", text)

    for rel in listing(str(out)):
        if rel.startswith("generated") or rel == "README.md" or "_chirality_" in rel or "_basepairs_" in rel:
            assert undated(open(os.path.join(shipped, rel)).read()) == undated((out / rel).read_text()), \
                f"examples/namd_mdff/{rel} is out of date: python3 tests/mdff_system.py --example examples/namd_mdff"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_keys_follow_the_outputs_whatever_the_resume_setting(tmp_path):
    _bin(tmp_path)
    d, _atoms = _system(tmp_path)
    log = tmp_path / "namd-runs.log"

    def run(settings=None):
        script = _write(tmp_path, d, settings)
        if log.exists():
            log.unlink()
        res = _run(script, tmp_path, d, NAMD_SAME)
        assert res.returncode == 0, res.stdout + res.stderr
        return [os.path.basename(r).split("_")[2] for r in log.read_text().split()] if log.exists() else []

    def stages(final_xi):
        return [dict(name="min", kind="min", steps=100, gscale=0.3),
                dict(name="ramp", kind="md", steps=40, gscale=0.1, gscale_end=0.3, segments=2, positional=True),
                dict(name="final-min", kind="min", steps=100, gscale=final_xi)]

    every = ["min", "ramp-001", "ramp-002", "final-min"]
    assert run({"stages": stages(10.0)}) == every
    # 'Skip finished runs' off with another coupling: everything runs, and the keys record it
    assert run({"stages": stages(5.0), "resume": False}) == every
    # back on, with the first coupling: the final-min outputs on disk were made with 5, so it runs again
    assert run({"stages": stages(10.0)}) == ["final-min"]
    # a rebuilt topology differs only in the time stamp tleap / ParmEd write: finished runs are kept
    top = (d / "sys.parm7").read_text()
    stamped = re.sub(r"DATE = \S+\s+\S+", "DATE = 10/30/26  14:02:11", top, count=1)
    assert stamped != top
    (d / "sys.parm7").write_text(stamped)
    assert run({"stages": stages(10.0)}) == []


def test_planner_checks_pdb_atom_order_before_reading_coordinates(tmp_path):
    atoms = ms.atoms_of(ms.mixed_pdb())
    ms.write_topology(str(tmp_path / "t.parm7"), atoms)
    swapped = atoms[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    (tmp_path / "order.pdb").write_text(ms.model_pdb(swapped))
    files = {"topology": str(tmp_path / "t.parm7"), "coordinates": str(tmp_path / "order.pdb"),
             "map": "/d/map.mrc", "model": "/d/model.pdb"}
    _plan, st = single({"solvent": "vacuum"}, files=files)
    errors = messages(st, "error")
    assert any("atom 1 is " in m and "not in the topology's atom order" in m for m in errors), errors
    assert not any("hand" in m or "geometry" in m for m in errors), errors  # nothing read from wrong atoms
    (tmp_path / "restart.rst7").write_bytes(b"CDF\x02" + b"\0" * 32)
    files["coordinates"] = str(tmp_path / "restart.rst7")
    _plan, st = single({"solvent": "vacuum"}, files=files)
    assert any("NetCDF restart" in m for m in messages(st, "info"))


def test_stage_names_differing_in_letter_case():
    _plan, st = single({"stages": [dict(name="Fit", kind="md", steps=100), dict(name="fit", kind="min")]})
    errors = messages(st, "error")
    assert "stage names must differ, letter case aside (Fit, fit)" in errors and not any("repeat" in m for m in errors)
    ramp = dict(name="ramp", kind="md", steps=100, segments=5)
    _plan, st = single({"stages": [ramp, dict(ramp)]})
    assert [m for m in messages(st, "error") if "stage names" in m or "repeat" in m] == [
        "stage names must differ, letter case aside (ramp)"]


def test_planner_warns_about_octahedral_boxes_and_needs_cryst1_for_solvated_pdbs(tmp_path):
    atoms = ms.atoms_of(ms.mixed_pdb())
    oct_box = (64.0, 64.0, 64.0, 109.4712206, 109.4712206, 109.4712206)
    ms.write_topology(str(tmp_path / "oct.parm7"), atoms, box=oct_box)
    ms.write_rst7(str(tmp_path / "oct.rst7"), [a[5] for a in atoms], box=oct_box)
    files = {"topology": str(tmp_path / "oct.parm7"), "coordinates": str(tmp_path / "oct.rst7"),
             "map": "/d/map.mrc", "model": "/d/model.pdb"}
    _plan, st = single(files=files)
    assert any("truncated octahedron" in m and "Rectangular" in m for m in messages(st, "warning")), messages(st)
    # the same system as a PDB in topology order: a periodic run needs its CRYST1 box
    (tmp_path / "plain.pdb").write_text(ms.model_pdb(atoms))
    _plan, st = single(files=dict(files, coordinates=str(tmp_path / "plain.pdb")))
    assert any("has no CRYST1 record" in m for m in messages(st, "error")), messages(st)
    (tmp_path / "boxed.pdb").write_text("CRYST1   64.000   64.000   64.000 109.47 109.47 109.47 P 1           1\n"
                                        + ms.model_pdb(atoms))
    _plan, st = single(files=dict(files, coordinates=str(tmp_path / "boxed.pdb")))
    assert not any("CRYST1" in m for m in messages(st, "error"))


def test_amber_mdff_template_builds_a_rectangular_box():
    from mdatk.workflows.plan import apply_template
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    apply_template(wf, "amber-mdff")
    build = next(n for n in wf.nodes if n.tool == "amber-build")
    assert TOOLS["amber-build"].normalise(build.settings)["box"] == "rect"


def test_planned_octahedral_build_is_flagged_before_anything_runs():
    _plan, st = built_after_amber({"water": "tip3p", "box": "oct"})
    assert any("truncated octahedron" in m for m in messages(st, "warning")), messages(st)
    _plan, st = built_after_amber({"water": "tip3p", "box": "rect"})
    assert not any("truncated octahedron" in m for m in messages(st, "warning"))


# ------------------------------------------------- restraints and continuations that are plan settings
JUNCTION = {"name": "junction", "atom1": "16:O3'", "atom2": "17:P", "lower": 1.4, "upper": 1.8, "k": 20.0}


def test_defaults_add_nothing_to_the_generated_files():
    plan, st = single()
    cfg = TOOLS["namd-mdff"].prep_settings(st.settings, st.ctx)
    assert "extra_bonds" not in cfg and "residues" not in cfg["positional"] and "final_rst7" not in cfg
    assert set(st.outputs) == {"fitted"}  # nothing passed on: the plan summary stays as it was
    for f in plan.generate(["m"])["m"]:
        if not f.name.endswith(".py"):
            assert "extrabonds" not in f.content and "fitted-system" not in f.content, f.name
    assert "Restraints of the plan" not in plan.summary_text()


def test_extra_restraints_and_positional_residues_are_plan_settings():
    stages = [dict(name="fit", kind="md", steps=1000, positional=True),
              dict(name="free", kind="min", steps=100, extrabonds=False)]
    plan, st = single({"extra_bonds": [JUNCTION], "positional_residues": "16-17", "stages": stages})
    assert not messages(st, "error"), messages(st)
    # what will be applied is in the checks, not only in a generated file
    info = messages(st, "info")
    assert "extra restraint junction: 16:O3' - 17:P, flat 1.4-1.8 Å, k 20 kcal/mol/Å² (stages fit)" in info
    assert ("positional restraints, k 1 kcal/mol/Å², on the backbone atoms of residues 16-17 only (stages fit)"
            in info)
    assert TOOLS["namd-mdff"].describe(st.settings).endswith(", 1 extra restraint, positional on 16-17")
    summary = plan.summary_text()
    assert "Restraints of the plan:" in summary and "- extra restraint junction: 16:O3' - 17:P" in summary
    files = {f.name: f.content for f in plan.generate(["m"])["m"]}
    settings = files[f"apt1_02_mdff_{STAMP}.py"].split("# --- settings written by MDATK ---")[1]
    assert ("'extra_bonds': {'enabled': True, 'bonds': [['junction', \"16:O3'\", '17:P', 1.4, 1.8, 20.0]], "
            f"'file': 'apt1_02_extrabonds_{STAMP}.txt'}}") in settings
    assert "'positional': {'selection': 'backbone', 'k': 1.0, 'residues': '16-17'}" in settings
    fit, free = files[f"apt1_02_fit_{STAMP}.namd"], files[f"apt1_02_free_{STAMP}.namd"]
    assert f"extraBondsFile   apt1_02_extrabonds_{STAMP}.txt" in fit and "extrabonds" not in free
    assert "# positional restraints to the starting coordinates (residues 16-17 only)" in fit
    key = next(ln for ln in files[f"apt1_02_namd-mdff_{STAMP}.sh"].splitlines() if ln.startswith("KEY=$( { sed"))
    assert f"apt1_02_extrabonds_{STAMP}.txt" in key  # a changed restraint reruns the stages


@pytest.mark.parametrize("settings, level, text", [
    ({"extra_bonds": [dict(JUNCTION, atom1="16-O3'")]}, "error", "extra restraint junction: cannot read the atom"),
    ({"extra_bonds": [dict(JUNCTION, atom2="")]}, "error", "extra restraint junction: name its second atom"),
    ({"extra_bonds": [dict(JUNCTION, atom2="16:O3'")]}, "error", "both ends are 16:O3'"),
    ({"extra_bonds": [dict(JUNCTION, lower=2.0)]}, "error", "the lower wall must not be above the upper wall"),
    ({"extra_bonds": [JUNCTION], "stages": [dict(name="a", kind="min", steps=10, extrabonds=False)]}, "warning",
     "extra distance restraint(s) are defined but no stage applies them"),
    ({"positional_residues": "17-16", "stages": [dict(name="a", kind="min", steps=10, positional=True)]}, "error",
     "positional restraints on residues: cannot read the residue range '17-16'"),
    ({"positional_residues": "16-17"}, "warning", "no stage uses positional restraints"),
])
def test_own_restraint_settings_are_checked(settings, level, text):
    _plan, st = single(settings)
    assert any(text in m for m in messages(st, level)), messages(st)


def test_planner_resolves_the_restraints_on_the_files(tmp_path):
    d, _atoms = _system(tmp_path)
    files = {"topology": str(d / "sys.parm7"), "coordinates": str(d / "sys.rst7"), "map": str(d / "map.mrc"),
             "model": str(d / "model.pdb")}
    stages = [dict(name="fit", kind="min", steps=10, positional=True)]
    bad = {"extra_bonds": [dict(JUNCTION, atom1="40:O3'", atom2="A:4:XX")], "positional_residues": "30-35",
           "stages": stages}
    _plan, st = single(bad, files)
    errors = messages(st, "error")
    assert "extra restraint junction: residue 40 does not exist (the topology has 23 residues)" in errors
    assert any("30-35 names no residue" in m for m in errors), errors
    good = {"extra_bonds": [dict(JUNCTION, atom1="4:O3'", atom2="B:1:O5'")], "positional_residues": "A:4, B:1",
            "stages": stages}
    _plan, st = single(good, files)
    assert not messages(st, "error"), messages(st)
    info = messages(st, "info")
    assert any("extra restraint junction (4:O3' - B:1:O5') starts at" in m for m in info), info
    assert "positional restraints hold 11 atoms of 2 residues (A:4, B:1); every other atom is free" in info
    # before the coordinates exist (e.g. a continuation's), topology numbers are checked on the topology alone
    _plan, st = single(dict(bad, positional_residues="A:30, 40"), dict(files, coordinates=str(d / "later.rst7")))
    errors = messages(st, "error")
    assert "extra restraint junction: residue 40 does not exist (the topology has 23 residues)" in errors
    assert "positional restraints: 40 names no residue (the topology has 23)" in errors
    assert any("extra restraint junction, the positional residues: residues given by chain are found from the "
               "coordinates" in m for m in messages(st, "info")), messages(st)


def _continuation_plan(root, d, first=None, second=None, connect_model=True):
    from mdatk.workflows.plan import add_node, connect
    wf = WorkflowParams(project_name="apt1", stamp=STAMP)
    base = {"sequence": "GCGC GCGC", "dotbracket": "(((( ))))", "timestep": "4"}
    p1 = add_node(wf, "namd-mdff", settings=dict(base, stages=[dict(name="fit", kind="min", steps=100)],
                                                 **(first or {})))
    p1.files = {"topology": str(d / "sys.parm7"), "coordinates": str(d / "sys.rst7"), "map": str(d / "map.mrc"),
                "model": str(d / "model.pdb")}
    p2 = add_node(wf, "namd-mdff", settings=dict(base, **(second or {})))
    p2.files = {"map": str(d / "map.mrc")}
    assert not TOOLS["namd-mdff"].normalise(p1.settings)["continuation"]
    connect(wf, p1.id, "restart", p2.id, "coordinates")
    assert p1.settings["continuation"] is True  # connecting the output turns it on
    connect(wf, p1.id, "topology", p2.id, "topology")
    if connect_model:
        connect(wf, p1.id, "fitted", p2.id, "model")
    return wf, p1, p2, Plan(wf, str(root))


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_continuation_step_continues_the_fit_of_the_step_before(tmp_path):
    _bin(tmp_path)
    d, atoms = _system(tmp_path)
    second = {"extra_bonds": [dict(JUNCTION, atom1="A:4:O3'", atom2="B:1:O5'")], "positional_residues": "A:4 B:1",
              "stages": [dict(name="pin", kind="min", steps=100, positional=True)]}
    wf, p1, p2, plan = _continuation_plan(tmp_path / "plan", d, second=second)
    st1, st2 = plan.steps[p1.id], plan.steps[p2.id]
    assert not messages(st1, "error") and not messages(st2, "error"), messages(st2)
    assert set(st1.outputs) == {"fitted", "restart", "topology"}
    assert st2.ctx.value("topology").path == str(d / "sys.parm7")  # passed on, the very file
    assert any(m.startswith("continues the fit of") for m in messages(st2, "info")), messages(st2)
    # the restraints are visible before anything ran; residues by chain are found once the coordinates exist
    assert any("extra restraint junction: A:4:O3' - B:1:O5'" in m for m in messages(st2, "info"))
    report = plan.write([p1.id, p2.id], supersede=False)
    scripts = sorted(p for p in report.written if re.search(r"_namd-mdff_.*\.sh$", p))
    assert len(scripts) == 2
    first_sh = open(scripts[0]).read()
    assert f"apt1_01_fitted-system_{STAMP}.rst7" in first_sh
    for script in scripts:
        res = _run(script, tmp_path, d, NAMD_SAME)
        assert res.returncode == 0, res.stdout + res.stderr
    one, two = (os.path.dirname(s) for s in scripts)
    whole = os.path.join(one, f"apt1_01_fitted-system_{STAMP}.rst7")
    top = nm.Topology(str(d / "sys.parm7"))
    start = nm.read_rst7(os.path.join(two, f"apt1_02_start_{STAMP}.rst7"), top.natom)
    assert start[0] == nm.read_rst7(whole, top.natom)[0] and start[1] is not None  # every atom, with the box
    prep_log = open(os.path.join(two, f"apt1_02_prep_{STAMP}.log")).read()
    assert re.search(r"map frame: \d+ heavy atoms of \.\./01_namd-mdff/apt1_01_fitted_\S+ superposed, RMSD 0\.00",
                     prep_log), prep_log
    walls = [ln for ln in open(os.path.join(two, f"apt1_02_extrabonds_{STAMP}.txt")) if ln.startswith("wall")]
    o3 = next(i for i, a in enumerate(atoms) if a[:2] == ("A", 4) and a[3] == "O3'")
    o5 = next(i for i, a in enumerate(atoms) if a[:2] == ("B", 1) and a[3] == "O5'")
    assert walls == [f"wall {o3} {o5} 20.000 1.400 1.800\n"]
    ref = [ln for ln in open(os.path.join(two, f"apt1_02_posres_{STAMP}.pdb")) if ln.startswith("ATOM")]
    assert {int(ln[22:26]) for ln in ref if float(ln[60:66])} == {4, 5}
    # planned again once the first part has run: the residues given by chain are resolved on its files
    st2 = Plan(wf, str(tmp_path / "plan")).steps[p2.id]
    assert "positional restraints hold 11 atoms of 2 residues (A:4 B:1); every other atom is free" in messages(st2)


def test_continuation_wiring_is_checked(tmp_path):
    d, _atoms = _system(tmp_path)
    wf, p1, p2, plan = _continuation_plan("/tmp/mdatk-plan", d, connect_model=False)
    st2 = plan.steps[p2.id]
    assert any("fitted model as the map-frame model" in m for m in messages(st2, "warning")), messages(st2)
    assert not any("tleap moves the solute" in m for m in messages(st2))
    # another topology than the one the coordinates belong to (a rebuild numbers the solvent anew)
    wf.edges = [e for e in wf.edges if e.input != "topology"]
    p2.files.update(topology=str(d / "sys.parm7").replace("sys", "rebuilt"), model=str(d / "model.pdb"))
    st2 = Plan(wf, "/tmp/mdatk-plan").steps[p2.id]
    assert any("belong to its topology: connect" in m for m in messages(st2, "error")), messages(st2)
    assert any("the map-frame model is not" in m for m in messages(st2, "warning")), messages(st2)
    # an edge from an output the step no longer writes
    p1.settings["continuation"] = False
    st2 = Plan(wf, "/tmp/mdatk-plan").steps[p2.id]
    assert any("does not produce this file" in m for m in messages(st2, "error")), messages(st2)
