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
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", AMBERHOME=AMBERHOME)


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
        if st["kind"] == "min":  # enough minimisation that the random-seeded heating is stable
            st["maxcyc"], st["ncyc"] = 300, 100
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


def _nonbond_rows(top_text):
    rows, state = [], ""
    for line in top_text.splitlines():
        s = line.split(";")[0].strip()
        if s.startswith("["):
            state = s.strip("[] ").lower()
            continue
        if state == "nonbond_params" and s:
            f = s.split()
            rows.append((f[0], f[1], float(f[3]), float(f[4])))
    return rows


@pytest.mark.parametrize("model,water", [("nano", "spce"), ("micro", "tip3p")])
def test_optimised_mg_pairs_reach_gromacs(tmp_path, model, water):
    """microMg / nanoMg: frcmod + ParmEd pair terms survive NBfix19's addLJType and
    the GROMACS conversion carries them as [ nonbond_params ] with the authors' values."""
    from mdatk.workflows import mg_params

    env = _wrappers(tmp_path)
    cp = tmp_path / "frcmod.cp"
    cp.write_text("placeholder CP frcmod\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n\nIMPROPER\n\nNONBON\n\n")
    wf = WorkflowParams(project_name="mg", stamp=STAMP)
    b = add_node(wf, "amber-build", settings={"cp_frcmod": str(cp), "buffer": 8.0, "mg_count": 2,
                                              "water": water, "mg_model": model, "split_types": True,
                                              "save_pdb": False})
    b.files["structure"] = _duplex_pdb(tmp_path)
    run_gmx = bool(os.environ.get("MDATK_TEST_GROMACS") and shutil.which("gmx"))
    g = add_node(wf, "gromacs", settings={"gmx_version": "2023", "module": "", "gpu": False,
                                          "separate_production": False, "maxwarn": 1})
    for st in g.settings["stages"]:
        st["nsteps"] = 500 if st["integrator"] == "steep" else 10
        st["nstxout_compressed"] = st["nstenergy"] = 10
    connect(wf, b.id, "topology", g.id, "topology")
    connect(wf, b.id, "coordinates", g.id, "coordinates")
    plan = Plan(wf, str(tmp_path / "plan"))
    assert not [i for _n, i in plan.all_issues() if i.level == "error"]
    report = plan.write()
    root = tmp_path / "plan"
    build = root / "01_amber-build"
    script = report.master if run_gmx else str(next(build.glob("*_amber-build_*.sh")))
    proc = subprocess.run(["bash", os.path.basename(script)], cwd=os.path.dirname(script), capture_output=True,
                          text=True, env=env, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    leap_log = (build / f"mg_01_tleap_{STAMP}.log").read_text()
    assert f"mg_01_mg_{STAMP}.frcmod" in leap_log
    p = mg_params.get_set(model, water)
    parm = build / f"mg_01_parmed_{STAMP}.parm7"
    gro_dir = root / "02_gromacs"
    if run_gmx:
        top = (gro_dir / f"mg_02_convert_{STAMP}.top").read_text()
    else:  # convert in-process with the embedded script's functions
        from mdatk.workflows.scripts import amber2gmx
        out_top = tmp_path / "conv.top"
        assert amber2gmx.main([str(parm), str(build / f"mg_01_tleap_{STAMP}.rst7"), str(out_top),
                               str(tmp_path / "conv.gro")]) == 0
        top = out_top.read_text()
    rows = _nonbond_rows(top)
    mg_rows = [(s, e) for a, b2, s, e in rows if "Mg" in a or "Mg" in b2]

    def present(pair):
        return any(abs(s - pair[0]) <= 1e-6 * pair[0] and abs(e - pair[1]) <= 1e-6 * pair[1]
                   for s, e in mg_rows)

    for ty in ("O2", "NB", "OS", "P", "C"):
        assert present(p.rna[ty]), (ty, p.rna[ty], sorted(mg_rows))
    assert present(p.mg_cl)
    # NBfix19 pairs are still there next to the Mg2+ terms
    assert any(abs(s - 2.1 / 2 ** (1 / 6) / 10) < 1e-6 for _a, _b, s, _e in rows)
    atomtypes = top.split("[ atomtypes ]")[1].split("[")[0]
    mg_line = [ln for ln in atomtypes.splitlines() if ln.strip().startswith("Mg")][0].split()
    assert float(mg_line[-2]) == pytest.approx(p.ion[0], rel=1e-5)
    assert float(mg_line[-1]) == pytest.approx(p.ion[1], rel=1e-5)
    if run_gmx:
        assert (gro_dir / f"mg_02_md_{STAMP}.gro").exists()


def test_amber_build_cp_library_tip4p2005_nanomg(tmp_path):
    env = _wrappers(tmp_path)
    wf = WorkflowParams(project_name="duplex", stamp=STAMP)
    b = add_node(wf, "amber-build", settings={"water": "tip4p2005", "mg_model": "nano", "mg_count": 2,
                                              "buffer": 8.0})
    b.files["structure"] = _duplex_pdb(tmp_path)
    plan = Plan(wf, str(tmp_path / "plan"))
    assert not [i for _n, i in plan.all_issues() if i.level == "error"]
    report = plan.write()
    proc = subprocess.run(["bash", report.master], capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    step = tmp_path / "plan" / "01_amber-build"
    log = (step / f"duplex_01_tleap_{STAMP}.log").read_text()
    assert "Could not find" not in log and "FATAL" not in log
    top = (step / f"duplex_01_parmed_{STAMP}.parm7").read_text()
    types = " ".join(top.split("%FLAG AMBER_ATOM_TYPE")[1].split("%FLAG")[0].splitlines()[1:])
    assert " OP " in f" {types} " and " OR " in f" {types} " and " EP " in f" {types} "
    pm_log = (step / f"duplex_01_parmed_{STAMP}.log").read_text()
    assert "@%OP-@%HO" in pm_log.replace(" ", "") or "OP" in pm_log


def test_amber_build_1264_with_cp_library(tmp_path):
    env = _wrappers(tmp_path)
    wf = WorkflowParams(project_name="duplex", stamp=STAMP)
    b = add_node(wf, "amber-build", settings={"water": "tip3p", "mg_model": "1264", "mg_count": 2, "buffer": 8.0})
    b.files["structure"] = _duplex_pdb(tmp_path)
    plan = Plan(wf, str(tmp_path / "plan"))
    report = plan.write()
    proc = subprocess.run(["bash", report.master], capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    step = tmp_path / "plan" / "01_amber-build"
    assert "LENNARD_JONES_CCOEF" in (step / f"duplex_01_parmed_{STAMP}.parm7").read_text()
