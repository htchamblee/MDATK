"""Opt-in end-to-end runs of generated scripts with the real programs.

Set MDATK_TEST_AMBERHOME to an AmberTools installation (tleap, ambpdb, sander,
cpptraj; ParmEd importable by python3) to run the Amber chain; sander stands
in for pmemd.  Set MDATK_TEST_GROMACS=1 with gmx on PATH to also convert the
NBfix topology to GROMACS and run grompp + a few steps.
"""

import os
import shutil
import stat
import subprocess
import warnings

import pytest

from mdatk.config import WorkflowParams
from mdatk.workflows.amber import AmberMDTool
from mdatk.workflows.plan import Plan, add_node, connect

AMBERHOME = os.environ.get("MDATK_TEST_AMBERHOME", "")
pytestmark = pytest.mark.skipif(not (AMBERHOME and os.path.exists(os.path.join(AMBERHOME, "bin", "tleap"))),
                                reason="set MDATK_TEST_AMBERHOME to an AmberTools installation")
STAMP = "20261015-1432"


def _wrappers(tmp_path):
    """tleap/ambpdb/sander/cpptraj wrappers (+ pmemd -> sander) with AmberTools' libraries."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    lib = os.path.join(AMBERHOME, "lib")
    for name, target in (("tleap", "tleap"), ("ambpdb", "ambpdb"), ("sander", "sander"),
                         ("cpptraj", "cpptraj"), ("pmemd", "sander")):
        p = bindir / name
        p.write_text(f'#!/bin/sh\nexport AMBERHOME="{AMBERHOME}"\n'
                     f'LD_LIBRARY_PATH="{lib}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}" '
                     f'exec "{AMBERHOME}/bin/{target}" "$@"\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    if not shutil.which("parmed"):
        p = bindir / "parmed"
        p.write_text('#!/bin/sh\nexec python3 -c "import sys; from parmed.scripts import clapp; '
                     'sys.argv[0] = \'parmed\'; clapp()" "$@"\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")


def _duplex_pdb(tmp_path):
    import MDAnalysis as mda
    from mdatk.demo import build_demo
    warnings.filterwarnings("ignore")
    demo = build_demo(str(tmp_path / "demo"), n_frames=2, solvent=False)
    u = mda.Universe(demo["topology"], demo["trajectory"])
    if not hasattr(u.atoms, "chainIDs"):
        u.add_TopologyAttr("chainIDs")
    rna = u.select_atoms("resname G C A U")
    rna.chainIDs = ["A" if a.resid <= 12 else "B" for a in rna]
    path = tmp_path / "duplex.pdb"
    rna.write(str(path))
    return str(path)


def test_amber_chain_runs(tmp_path):
    env = _wrappers(tmp_path)
    cp = tmp_path / "frcmod.cp"
    cp.write_text("placeholder CP frcmod\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n\nIMPROPER\n\nNONBON\n\n")
    wf = WorkflowParams(project_name="duplex", stamp=STAMP)
    b = add_node(wf, "amber-build", settings={"cp_frcmod": str(cp), "buffer": 8.0, "mg_count": 2,
                                              "hmr": True, "two_pass": True})
    b.files["structure"] = _duplex_pdb(tmp_path)
    r = add_node(wf, "amber-restraints", settings={"sequence": "GCAUGGCAUCGA UCGAUGCCAUGC",
                                                   "dotbracket": "(" * 12 + ")" * 12})
    stages = AmberMDTool().defaults()["stages"]
    for st in stages:
        if st["kind"] == "min":
            st["maxcyc"], st["ncyc"] = 20, 10
        else:
            st["nstlim"], st["ntpr"], st["ntwx"], st["ntwr"] = 20, 10, 10, 10
    stages[-1]["segments"] = 2
    m = add_node(wf, "amber-md", settings={"engine": "pmemd", "stages": stages, "separate_production": False,
                                           "dry_traj": True})
    connect(wf, b.id, "topology", r.id, "topology")
    connect(wf, b.id, "topology", m.id, "topology")
    connect(wf, b.id, "coordinates", m.id, "coordinates")
    connect(wf, r.id, "restraints", m.id, "restraints")
    gmx = None
    if os.environ.get("MDATK_TEST_GROMACS") and shutil.which("gmx"):
        gmx = add_node(wf, "gromacs", settings={"gmx_version": "2023", "module": "", "gpu": False,
                                                "separate_production": False})
        for st in gmx.settings["stages"]:
            st["nsteps"] = 20
            st["nstxout_compressed"] = st["nstenergy"] = 10
        connect(wf, b.id, "topology", gmx.id, "topology")
        connect(wf, b.id, "coordinates", gmx.id, "coordinates")
    plan = Plan(wf, str(tmp_path / "plan"))
    assert not [i for _n, i in plan.all_issues() if i.level == "error"]
    report = plan.write()
    proc = subprocess.run(["bash", report.master], capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    root = tmp_path / "plan"
    dist = (root / "02_amber-restraints" / f"duplex_02_helix_{STAMP}.dist").read_text()
    assert dist.count("&rst") == 31
    parmed_log = (root / "01_amber-build" / f"duplex_01_parmed_{STAMP}.log").read_text()
    assert "Repartitioning hydrogen masses" in parmed_log and "@%NB,NC-@%H pairwise" in parmed_log
    md = root / "03_amber-md"
    assert (md / f"duplex_03_prod-002_{STAMP}.rst7").exists()
    assert "RESTRAINT" in (md / f"duplex_03_heat_{STAMP}.out").read_text()
    assert (md / f"duplex_03_dry-prod_{STAMP}.nc").exists()
    if gmx is not None:
        top = (root / "04_gromacs" / f"duplex_04_convert_{STAMP}.top").read_text()
        assert "[ nonbond_params ]" in top and "[ pairtypes ]" in top and "#ifdef POSRES" in top
        assert (root / "04_gromacs" / f"duplex_04_md_{STAMP}.gro").exists()
