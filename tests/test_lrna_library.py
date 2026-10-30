"""The L-RNA library MDATK generates from the D force field (scripts/lrna_lib.py), its wiring into the
Amber build and the topology check that the L residues got it: the generator on a small force field, the
planner's automatic choice, the build script, and - with AmberTools - the test that matters: a D duplex
and its mirror image built with the generated library have the same energy, term by term."""

import os
import re
import shutil
import subprocess

import pytest

import mdff_system as ms
from mdatk import builder
from mdatk.config import PlanNode, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan
from mdatk.workflows.scripts import cp_retype, lrna_lib, topcheck
from test_lrna_naming import _lib
from test_topcheck import run, write_parm7

# a small parm*.dat: title, MASS, BOND (with its title line), ANGL, DIHE, IMPR, equivalences, MOD4 NONB
DAT = """\
mini force field
OS 16.00         0.465
OH 16.00         0.465
CT 12.01         0.878
CI 12.01         0.878
P  30.97         1.538

title of the bond block
OS-CT  301.0    1.425
OH-CI  320.0    1.410

OS-CT-CI   50.0     109.50
OH-CI-CT   50.0     109.50

CT-CI-OS-P     1    0.20        31.79698      -1.
CT-CI-OS-P     1    0.40       180.0           2.
OH-CI-CT-CT    1    0.15         0.0           3.

CT-CT-OS-P          1.1         180.0         2.

OS  OH

MOD4      RE
  OS          1.6837  0.1700
  OH          1.7210  0.2104
  CT          1.9080  0.1094
END
"""


def _unit(name, o5_type):
    return [f'!entry.{name}.unit.atoms table  str name  str type  int typex  int resx  int flags  int seq  '
            'int elmnt  dbl chg',
            f' "O5\'" "{o5_type}" 0 1 131072 1 8 -0.5', ' "O4\'" "OS" 0 1 131072 2 8 -0.35',
            f'!entry.{name}.unit.name single str', f' "{name}"',
            f'!entry.{name}.unit.positions table  dbl x  dbl y  dbl z', ' 1.0 2.0 3.0', ' -1.5 0.5 -2.25',
            f'!entry.{name}.unit.residues table  str name  int seq  int childseq  int startatomx  str restype  '
            'int imagingx', f' "{name}" 1 3 1 "?" 0']


LIB = "\n".join(['!!index array str', ' "G"', ' "G5"', ' "MG"'] + _unit("G", "OS") + _unit("G5", "OH")
                + ['!entry.MG.unit.name single str', ' "MG"']) + "\n"


def test_generator_mirrors_units_and_dihedral_phases(tmp_path):
    (tmp_path / "RNA.lib").write_text(LIB)
    (tmp_path / "mini.dat").write_text(DAT)
    assert lrna_lib.main([str(tmp_path / "RNA.lib"), str(tmp_path / "mini.dat"), str(tmp_path / "out")]) == 0
    lib = (tmp_path / "out" / "LRNA.lib").read_text()
    # nucleotide units only, Dudek names inside and out; O5' / O4' retyped; z mirrored
    assert lib.startswith('!!index array str\n "LG"\n "LG5"\n') and "MG" not in lib
    assert '!entry.LG5.unit.name single str\n "LG5"' in lib and ' "LG" 1 3 1 "?" 0' in lib
    assert ' "O5\'" "LD" 0' in lib and ' "O5\'" "LH" 0' in lib and ' "O4\'" "LD" 0' in lib
    assert " 1 2 -3\n -1.5 0.5 2.25" in lib
    frc = (tmp_path / "out" / "frcmod.lrna").read_text()
    sections = dict(re.findall(r"^(MASS|BOND|ANGL|DIHE|IMPR|NONB)\n(.*?)\n\n", frc, re.S | re.M))
    assert sections["MASS"].splitlines() == ["LD 16.00         0.465", "LH 16.00         0.465"]
    # LH: OH's own Lennard-Jones terms, not those of the type it is listed as equivalent to
    assert sections["NONB"].split() == ["LD", "1.6837", "0.1700", "LH", "1.7210", "0.2104"]
    assert sections["BOND"].splitlines() == ["LD-CT  301.0    1.425", "LH-CI  320.0    1.410"]
    dihe = sections["DIHE"].splitlines()
    # a multi-term dihedral stays together; 31.79698 -> 328.20302, 180 / 0 unchanged
    assert [ln.split()[:4] for ln in dihe[:2]] == [["CT-CI-LD-P", "1", "0.20", "328.20302"],
                                                   ["CT-CI-LD-P", "1", "0.40", "180"]]
    assert dihe[2].split()[:4] == ["LH-CI-CT-CT", "1", "0.15", "0.0"]
    assert sections["IMPR"].split()[:2] == ["CT-CT-LD-P", "1.1"]


def test_generator_refuses_types_the_parameters_already_have(tmp_path, capsys):
    (tmp_path / "RNA.lib").write_text(LIB)
    (tmp_path / "mini.dat").write_text(DAT)
    (tmp_path / "frcmod.x").write_text("own L types\nMASS\nLD 16.00\n\nNONB\n  LD  1.6837  0.17\n")
    args = [str(tmp_path / "RNA.lib"), str(tmp_path / "mini.dat"), str(tmp_path / "frcmod.x"), str(tmp_path)]
    assert lrna_lib.main(args) == 1 and "already define atom type LD" in capsys.readouterr().err
    assert not (tmp_path / "LRNA.lib").exists()


def test_twin_map_adds_le_when_o4_differs_from_o5():
    assert lrna_lib.twin_map({"O5'": "OS", "O5'5": "OH", "O4'": "OS"}) == {"OS": "LD", "OH": "LH"}
    assert lrna_lib.twin_map({"O5'": "OR", "O5'5": "OH", "O4'": "OS"}) == {"OR": "LD", "OH": "LH", "OS": "LE"}


# ------------------------------------------------------------------------------------ planner / script
def _plan(tmp_path, pdb, **settings):
    (tmp_path / "in.pdb").write_text(pdb)
    s = TOOLS["amber-build"].normalise(dict({"water": "tip3p", "lrna": True}, **settings))
    node = PlanNode(id="b", tool="amber-build", step=1, settings=s, files={"structure": str(tmp_path / "in.pdb")})
    return Plan(WorkflowParams(project_name="apt1", stamp="20261015-1432", nodes=[node]), str(tmp_path / "p"))


def _files(plan):
    return {f.name.split("_", 2)[2]: f.content for f in plan.generate(["b"])["b"]}


def test_automatic_library_for_a_structure_with_l_residues(tmp_path):
    plan = _plan(tmp_path, ms.mixed_pdb())
    st = plan.steps["b"]
    assert not [i for i in st.issues if i.level == "error"], st.issues
    assert any("generates the L-RNA library" in i.message and "LD, LH, LE" in i.message for i in st.issues)
    files = _files(plan)
    sh = files["amber-build_20261015-1432.sh"]
    assert 'python3 apt1_01_lrna-lib_20261015-1432.py RNA_CP.lib "$AMBERHOME/dat/leap/parm/parm10.dat" ' \
           'frcmod.cp .' in sh
    assert sh.index("cp-lib_20261015-1432.py") < sh.index("lrna-lib_20261015-1432.py RNA_CP.lib")
    # the check counts the L residues of the PDB the final tleap reads
    assert re.search(r"python3 apt1_01_check_\S+\.py \S+\.parm7 apt1_01_lrna_20261015-1432\.pdb \|", sh), sh
    leap = files["tleap_20261015-1432.leap"]
    assert '{ "LD" "O" "sp3" } { "LH" "O" "sp3" } { "LE" "O" "sp3" }' in leap
    assert leap.index("loadOff LRNA.lib") < leap.index("mol = loadpdb")
    assert "L_TYPES = ['LD', 'LH', 'LE']" in files["check_20261015-1432.py"]
    assert "lrna" in __import__("mdatk.citations", fromlist=["x"]).step_resources(st)


def test_plain_ol3_twins_and_own_parameter_files(tmp_path):
    plan = _plan(tmp_path, ms.mixed_pdb(), rna_ff="ol3", nbfix19=False, bph0=False,
                 extra_frcmod="frcmod.mine")
    sh = _files(plan)["amber-build_20261015-1432.sh"]
    assert "P0=frcmod.mine; [ -f \"$P0\" ] || P0=\"$AMBERHOME/dat/leap/parm/$P0\"" in sh
    assert ('lrna-lib_20261015-1432.py "$AMBERHOME/dat/leap/lib/RNA.lib" '
            '"$AMBERHOME/dat/leap/parm/parm10.dat" "$P0" .') in sh
    assert "L_TYPES = ['LD', 'LH']" in _files(plan)["check_20261015-1432.py"]


def test_own_library_or_d_structure_is_not_generated(tmp_path):
    lib = _lib(tmp_path / "nucleic12_LRNA.lib", [f"L{b}{t}" for b in "ACGU" for t in ("", "5", "3")])
    own = _plan(tmp_path, ms.mixed_pdb(), extra_libs=lib, extra_frcmod="parm10_LRNA.dat")
    assert own.steps["b"].ok, own.steps["b"].issues
    assert "lrna-lib" not in _files(own)["amber-build_20261015-1432.sh"]
    d_only = builder.build_helix("GCGC", scheme="amber", hydrogens=True)
    assert "lrna-lib" not in _files(_plan(tmp_path, d_only))["amber-build_20261015-1432.sh"]


def test_generated_library_needs_ol3_and_terminal_names(tmp_path):
    errors = [i.message for i in _plan(tmp_path, ms.mixed_pdb(), rna_ff="ljbb", lrna_library="generated")
              .steps["b"].issues if i.level == "error"]
    assert any("for OL3 and OL3 + CP" in m for m in errors), errors
    errors = [i.message for i in _plan(tmp_path, ms.mixed_pdb(), lrna_terminals=False).steps["b"].issues
              if i.level == "error"]
    assert any("turn '5'/3' terminal names' on" in m for m in errors), errors


# ---------------------------------------------------------------------------------- topology check
def test_topology_check_counts_residues_with_the_l_types(tmp_path):
    pdb = tmp_path / "in.pdb"
    pdb.write_text(ms.mixed_pdb())
    assert topcheck.l_residues_in_pdb(str(pdb)) == 8
    atoms = [("OS", -1.0)] * 12 + [("LD", -1.0)] + [("OS", 1.0)] * 11
    path = write_parm7(tmp_path / "t.parm7", atoms, [("G5", 1), ("LG5", 13)], {})
    opts = dict(neutral=False, mg_count=0, ion_lj=None, pairs=False, l_types=("LD", "LH"))
    assert run(path, expected_l=1, **opts) == []
    problems = run(path, expected_l=2, **opts)
    assert len(problems) == 1 and "tleap loaded D units for L residues" in problems[0]


# ----------------------------------------------------------------------------- AmberTools energies
def _amberhome():
    home = os.environ.get("MDATK_TEST_AMBERHOME") or os.environ.get("AMBERHOME")
    if not home:
        exe = shutil.which("tleap")
        home = os.path.dirname(os.path.dirname(os.path.realpath(exe))) if exe else ""
    ok = home and all(os.path.isfile(os.path.join(home, "bin", x)) for x in ("tleap", "sander"))
    return home if ok and os.path.isfile(os.path.join(home, "dat", "leap", "lib", "RNA.lib")) else None


TERMS = ("BOND", "ANGLE", "DIHED", "VDWAALS", "EEL", "1-4 VDW", "1-4 EEL")


def _energies(home, d, name, leap):
    (d / f"{name}.leap").write_text(leap + f"m = loadpdb {name}.pdb\nsaveamberparm m {name}.parm7 {name}.rst7\n"
                                    "quit\n")
    env = dict(os.environ, AMBERHOME=home)
    subprocess.run([os.path.join(home, "bin", "tleap"), "-f", f"{name}.leap"], cwd=d, env=env,
                   capture_output=True, timeout=600, check=True)
    (d / "sp.in").write_text("single point\n &cntrl\n  imin=1, maxcyc=1, ncyc=1, ntb=0, cut=999.0, igb=0, "
                             "ntpr=1,\n /\n")
    subprocess.run([os.path.join(home, "bin", "sander"), "-O", "-i", "sp.in", "-p", f"{name}.parm7", "-c",
                    f"{name}.rst7", "-o", f"{name}.out", "-r", f"{name}.restrt"], cwd=d, env=env,
                   capture_output=True, timeout=600, check=True)
    text = (d / f"{name}.out").read_text()
    block = text[text.index("NSTEP"):]
    return {t: float(re.search(re.escape(t) + r"\s*=\s*(-?\d+\.\d+)", block).group(1)) for t in TERMS}


@pytest.mark.skipif(_amberhome() is None, reason="AmberTools (tleap, sander) not available")
@pytest.mark.parametrize("cp", [False, True], ids=["OL3", "OL3+CP"])
def test_mirror_image_has_the_same_energy(tmp_path, cp):
    """GACUGCAU duplex, D and its mirror image (heavy atoms: tleap builds the H from the templates)."""
    home = _amberhome()
    leap = "source leaprc.RNA.OL3\n"
    if cp:
        assert cp_retype.main([home, str(tmp_path)]) == 0
        leap += 'addAtomTypes { { "OR" "O" "sp3" } }\nloadamberparams frcmod.cp\nloadOff RNA_CP.lib\n'
        d_lib, parms = str(tmp_path / "RNA_CP.lib"), [str(tmp_path / "frcmod.cp")]
    else:
        d_lib, parms = os.path.join(home, "dat", "leap", "lib", "RNA.lib"), []
    parms.insert(0, os.path.join(home, "dat", "leap", "parm", "parm10.dat"))
    assert lrna_lib.main([d_lib] + parms + [str(tmp_path)]) == 0
    for chirality in "DL":
        (tmp_path / f"{chirality}.pdb").write_text(builder.build_helix("GACUGCAU", chirality=chirality,
                                                                       scheme="amber"))
    twins = ("LD", "LH", "LE") if cp else ("LD", "LH")
    l_leap = (leap + "addAtomTypes { " + " ".join(f'{{ "{t}" "O" "sp3" }}' for t in twins) + " }\n"
              "loadamberparams frcmod.lrna\nloadOff LRNA.lib\n")
    e_d, e_l = _energies(home, tmp_path, "D", leap), _energies(home, tmp_path, "L", l_leap)
    for term in TERMS:  # equal to the rounding of the 4-decimal coordinates (rst7 / PDB)
        assert e_l[term] == pytest.approx(e_d[term], abs=0.01 + 1e-5 * abs(e_d[term])), (term, e_d, e_l)
    parm = (tmp_path / "L.parm7").read_text()
    assert all(t in parm for t in twins)


# ------------------------------------------------------------------------------- review regressions
def test_generator_refuses_odd_phases_it_cannot_mirror(tmp_path, capsys):
    (tmp_path / "RNA.lib").write_text(LIB)
    (tmp_path / "mini.dat").write_text(DAT)
    (tmp_path / "frcmod.odd").write_text("2'-OH torsion\nDIHE\nHO-OH-CT-CT   1    0.50        40.0           1.\n")
    (tmp_path / "frcmod.even").write_text("even\nDIHE\nCT-CT-CT-CT   1    0.50       180.0           2.\n")
    out = tmp_path / "out"
    lib, dat = str(tmp_path / "RNA.lib"), str(tmp_path / "mini.dat")
    assert lrna_lib.main([lib, dat, str(tmp_path / "frcmod.even"), str(out)]) == 0
    # OH is twinned here (the 5'-OH): a term without OS / OH with an odd phase is refused
    (tmp_path / "frcmod.odd").write_text("x\nDIHE\nHO-CT-CT-CT   1    0.50        40.0           1.\n")
    assert lrna_lib.main([lib, dat, str(tmp_path / "frcmod.odd"), str(tmp_path / "o2")]) == 1
    assert "HO-CT-CT-CT" in capsys.readouterr().err and not (tmp_path / "o2" / "LRNA.lib").exists()


def test_equivalences_come_from_the_block_before_the_lennard_jones_terms():
    dat = DAT.replace("OS  OH\n", "N   NA  N2\n").replace("title of the bond block", "OS OH CT\n\ntitle")
    params = lrna_lib.read_dat(DAT.replace("OS  OH\n", "CT  CI\n"))
    assert params["EQUIV"] == {"CI": "CT"}
    assert "OS" not in lrna_lib.read_dat(dat)["EQUIV"]


def test_l_residues_are_counted_like_tleap_reads_them(tmp_path):
    blank = "\n".join(ln[:21] + " " + ln[22:] if ln.startswith(("ATOM", "HETATM")) else ln
                      for ln in ms.mixed_pdb().splitlines())
    (tmp_path / "b.pdb").write_text(blank)
    assert topcheck.l_residues_in_pdb(str(tmp_path / "b.pdb")) == 8


def test_automatic_leaves_own_l_rna_set_ups_alone(tmp_path):
    frc = tmp_path / "frcmod.mylrna"
    frc.write_text("own L-RNA\nMASS\nLD 16.00\n\nNONB\n  LD  1.6837  0.17\n")
    for settings in ({"extra_frcmod": str(frc)}, {"leap_before_load": "loadOff /x/mylrna.lib"},
                     {"leap_before_params": "loadAmberParams /x/frcmod.LRNA"}):
        plan = _plan(tmp_path, ms.mixed_pdb(), **settings)
        assert not plan.steps["b"].tool.lrna_generated(plan.steps["b"].settings, plan.steps["b"].ctx), settings


def test_generated_library_refuses_an_own_cp_residue_library(tmp_path):
    lib = _lib(tmp_path / "RNA_CaseP.lib", ["A", "C", "G", "U"])
    (tmp_path / "frcmod.casep").write_text("x\n")
    errors = [i.message for i in _plan(tmp_path, ms.mixed_pdb(), lrna_library="generated", cp_source="frcmod",
                                       cp_frcmod=str(tmp_path / "frcmod.casep"), extra_libs=lib)
              .steps["b"].issues if i.level == "error"]
    assert any("mirrors AmberTools' RNA.lib" in m for m in errors), errors


def test_no_stale_addatomtypes_hint_with_the_generated_library(tmp_path):
    st = _plan(tmp_path, ms.mixed_pdb(), lrna_fixes=False).steps["b"]
    assert not any("if leap reports unknown types" in i.message for i in st.issues), st.issues
