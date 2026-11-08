"""L-DNA: the library MDATK mirrors from the D-DNA force field (scripts/lrna_lib.py with the L-DNA settings of
mdatk.workflows.amber), its wiring into the Amber build (DNA force field, residue names, twin types for the
12-6-4 polarizabilities and the Mg2+ pair terms), and - with AmberTools - the energy of a D-DNA duplex and its
mirror image."""
import math
import os
import re
import shutil
import subprocess

import pytest

from mdatk import builder
from mdatk.config import PlanNode, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows import amber
from mdatk.workflows.plan import Plan
from mdatk.workflows.scripts import embed, lrna_lib

DNA = dict(amber.LDNA_SETTINGS)

# a small parm*.dat: the DNA types of C3' (C7) and C5' (CJ), an epsilon-like term through O3' (C7-OS-P-O2)
# and a delta-like one (OS-C7-CT-CJ) with phases that are not 0 / 180
DAT = """\
mini DNA force field
OS 16.00         0.465
OH 16.00         0.465
O2 16.00         0.434
CT 12.01         0.878
C7 12.01         0.878
CJ 12.01         0.878
P  30.97         1.538

title of the bond block
OS-C7  320.0    1.410
OH-C7  320.0    1.410

C7-OS-P    100.0    120.50

C7-OS-P-O2     1    0.30        31.79698       3.
OS-C7-CT-CJ    1    0.20       155.0          -1.
OS-C7-CT-CJ    1    0.40       180.0           2.

X -X -C7-OS         1.1         180.0         2.

MOD4      RE
  OS          1.6837  0.1700
  OH          1.7210  0.2104
  O2          1.6612  0.2100
  CT          1.9080  0.1094
END
"""


def _unit(name, o3_type):
    return [f'!entry.{name}.unit.atoms table  str name  str type  int typex  int resx  int flags  int seq  '
            'int elmnt  dbl chg',
            ' "O5\'" "OS" 0 1 131072 1 8 -0.5', ' "O4\'" "OS" 0 1 131072 2 8 -0.35',
            f' "O3\'" "{o3_type}" 0 1 131072 3 8 -0.5', ' "C3\'" "C7" 0 1 131072 4 6 0.1',
            f'!entry.{name}.unit.name single str', f' "{name}"',
            f'!entry.{name}.unit.positions table  dbl x  dbl y  dbl z', ' 1.0 2.0 3.0', ' -1.5 0.5 -2.25',
            ' 0.5 0.5 0.5', ' 2.0 1.0 -1.0',
            f'!entry.{name}.unit.residues table  str name  int seq  int childseq  int startatomx  str restype  '
            'int imagingx', f' "{name}" 1 5 1 "?" 0']


LIB = "\n".join(['!!index array str', ' "DT"', ' "DT3"', ' "A"'] + _unit("DT", "OS") + _unit("DT3", "OH")
                + _unit("A", "OS")) + "\n"


@pytest.fixture
def dna_mode(monkeypatch):
    """lrna_lib with the L-DNA settings the build script writes after its body."""
    for key, value in DNA.items():
        monkeypatch.setattr(lrna_lib, key, value)


def test_generator_in_dna_mode_twins_o3_and_mirrors_phases(tmp_path, dna_mode):
    (tmp_path / "DNA.lib").write_text(LIB)
    (tmp_path / "mini.dat").write_text(DAT)
    assert lrna_lib.main([str(tmp_path / "DNA.lib"), str(tmp_path / "mini.dat"), str(tmp_path / "out")]) == 0
    lib = (tmp_path / "out" / "LDNA.lib").read_text()
    # the DNA units only (not the RNA unit A), MDATK's L-DNA names, every sugar / backbone oxygen retyped
    assert lib.startswith('!!index array str\n "LDT"\n "LDT3"\n') and '"LA"' not in lib
    assert ' "O3\'" "LS" 0' in lib and ' "O3\'" "LT" 0' in lib and ' "O5\'" "LS" 0' in lib
    assert ' "O4\'" "LS" 0' in lib and ' "C3\'" "C7" 0' in lib  # carbons keep their types
    assert " 1 2 -3\n -1.5 0.5 2.25" in lib  # z mirrored
    frc = (tmp_path / "out" / "frcmod.ldna").read_text()
    assert frc.startswith("L-DNA atom types LT (OH) LS (OS)")
    sections = dict(re.findall(r"^(MASS|BOND|ANGL|DIHE|IMPR|NONB)\n(.*?)\n\n", frc, re.S | re.M))
    dihe = [[ln[:11].replace(" ", "")] + ln[11:].split()[:3] for ln in sections["DIHE"].splitlines()]
    # the epsilon-like term through O3' (C7-OS-P-O2): 31.79698 -> 328.20302
    assert ["C7-LS-P-O2", "1", "0.30", "328.20302"] in dihe, dihe
    # the delta-like term through O3': phase 155 -> 205, its 180 term unchanged, kept together
    delta = [d for d in dihe if d[0] == "LS-C7-CT-CJ"]
    assert [d[3] for d in delta] == ["205", "180"]


def test_dna_twin_map_and_refusals(dna_mode, tmp_path, capsys):
    assert lrna_lib.twin_map({"O5'": "OS", "O4'": "OS", "O3'3": "OH"}) == {"OH": "LT", "OS": "LS"}
    with pytest.raises(ValueError, match="without an L-DNA twin: OR"):
        lrna_lib.twin_map({"O5'": "OR", "O4'": "OS"})
    # an odd phase on a term no twinned type is in: refused, naming the twinned oxygens
    (tmp_path / "DNA.lib").write_text(LIB)
    (tmp_path / "odd.dat").write_text(DAT.replace(
        "OS-C7-CT-CJ    1    0.40       180.0           2.",
        "OS-C7-CT-CJ    1    0.40       180.0           2.\nCT-CT-C7-CJ    1    0.10        33.0           1."))
    assert lrna_lib.main([str(tmp_path / "DNA.lib"), str(tmp_path / "odd.dat"), str(tmp_path / "o")]) == 1
    assert "mirrors only O5' / O4' / O3'; use your own L-DNA files" in capsys.readouterr().err


def test_embedded_helper_uses_the_l_dna_settings(tmp_path):
    """The module has no __main__ block of its own: the settings MDATK writes after its body apply."""
    script = tmp_path / "ldna.py"
    script.write_text(embed(lrna_lib, DNA, "main(sys.argv[1:])", "L-DNA helper"))
    (tmp_path / "DNA.lib").write_text(LIB)
    (tmp_path / "mini.dat").write_text(DAT)
    res = subprocess.run(["python3", "-I", str(script), str(tmp_path / "DNA.lib"), str(tmp_path / "mini.dat"),
                          str(tmp_path / "out")], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "out" / "LDNA.lib").exists() and (tmp_path / "out" / "frcmod.ldna").exists()
    assert "LT as OH, LS as OS" in res.stderr


# ----------------------------------------------------------------------------------------- the build step

STAMP = "20261015-1432"


def _plan(tmp_path, pdb_text, **settings):
    (tmp_path / "in.pdb").write_text(pdb_text)
    s = TOOLS["amber-build"].normalise(dict({"water": "opc"}, **settings))
    node = PlanNode(id="b", tool="amber-build", step=1, settings=s, files={"structure": str(tmp_path / "in.pdb")})
    plan = Plan(WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[node]), str(tmp_path / "plan"))
    return plan, plan.steps["b"]


def _files(plan):
    return {f.name.replace(f"_{STAMP}", "").replace("apt1_01_", ""): f.content for f in plan.generate()["b"]}


def test_build_with_l_dna_sources_dna_mirrors_and_names(tmp_path):
    chimera = builder.build_helix("GACT", na="dna", chirality="L", scheme="ccd")  # CCD names 0DG ...
    plan, st = _plan(tmp_path, chimera, mg_count=2, mg_model="1264")
    assert st.ok, st.issues
    files = _files(plan)
    leap = [ln for ln in files["tleap.leap"].splitlines() if not ln.startswith("#")]
    assert leap[:2] == ["source leaprc.RNA.OL3", "source leaprc.DNA.OL24"]
    assert any('{ "LS" "O" "sp3" } { "LT" "O" "sp3" }' in ln for ln in leap)
    assert leap.index("loadamberparams frcmod.ldna") + 1 == leap.index("loadOff LDNA.lib")
    sh = files["amber-build.sh"]
    assert ('python3 apt1_01_ldna-lib_20261015-1432.py "$AMBERHOME/dat/leap/lib/ff-nucleic-OL24.lib" '
            '"$AMBERHOME/dat/leap/parm/parm10.dat" "$AMBERHOME/dat/leap/parm/frcmod.ff14SB" '
            '"$AMBERHOME/dat/leap/parm/ff-nucleic-OL24.frcmod" .') in sh
    assert "--scheme amber --dna" in sh  # 0DG ... -> LDG5 ... before tleap
    assert "for pair in LS=OS LT=OH; do" in sh and f"cp lj_1264_pol_cp.dat {amber.POL_1264}" in sh
    assert f"add12_6_4 :MG watermodel OPC polfile {amber.POL_1264}" in files["parmed.in"]
    assert "ldna-lib.py" in files and "TWIN_NAMES = {'OS': 'LS', 'OH': 'LT'}" in files["ldna-lib.py"]
    infos = [i.message for i in st.issues if i.level == "info"]
    assert any(m.startswith("MDATK generates the L-DNA library") for m in infos)
    # OL21 instead: its own library and parameter files
    plan, _st = _plan(tmp_path, chimera, dna_ff="ol21")
    sh = _files(plan)["amber-build.sh"]
    assert '"$AMBERHOME/dat/leap/lib/ff-nucleic-OL21.lib" "$AMBERHOME/dat/leap/parm/parm10.dat" ' \
           '"$AMBERHOME/dat/leap/parm/ff-nucleic-OL21.frcmod"' in sh


def test_microMg_pair_terms_cover_the_l_dna_types(tmp_path):
    plan, st = _plan(tmp_path, builder.build_helix("GACT", na="dna", chirality="L", scheme="amber"),
                     mg_count=2, mg_model="micro")
    pm = _files(plan)["parmed.in"]
    for twin, d in (("LS", "OS"), ("LT", "OH")):
        d_lines = [ln.split()[3:] for ln in pm.splitlines() if ln.startswith(f"changeLJPair :MG @%{d} ")]
        assert d_lines and d_lines == [ln.split()[3:] for ln in pm.splitlines()
                                       if ln.startswith(f"changeLJPair :MG @%{twin} ")], (twin, pm)


def test_d_dna_loads_the_dna_force_field_without_mirroring(tmp_path):
    plan, st = _plan(tmp_path, builder.build_helix("GACT", na="dna", scheme="amber"), mg_count=2, mg_model="1264")
    files = _files(plan)
    assert "source leaprc.DNA.OL24" in files["tleap.leap"] and "LDNA.lib" not in files["tleap.leap"]
    assert "ldna-lib.py" not in files and "polfile lj_1264_pol_cp.dat" in files["parmed.in"]  # no twins: as before


def test_rna_only_builds_are_unchanged_and_dna_needs_a_force_field(tmp_path):
    plan, _st = _plan(tmp_path, builder.build_helix("GACU", scheme="amber"))
    leap = _files(plan)["tleap.leap"]
    assert "leaprc.DNA" not in leap and "LDNA" not in leap
    _plan_, st = _plan(tmp_path, builder.build_helix("GACT", na="dna", scheme="amber"), dna_ff="none")
    assert ("error", "the structure has DNA but no DNA force field is chosen") in [(i.level, i.message)
                                                                                for i in st.issues]


def test_dna_from_the_structure_builder_is_known_before_the_pdb_exists(tmp_path):
    from mdatk.config import PlanEdge

    b = PlanNode(id="s", tool="structure-builder", step=1,
                 settings=TOOLS["structure-builder"].normalise({"mode": "chimera", "chimera_na": "dna",
                                                                "design": "D-DNA:GCAC 3'-5' L-DNA:CCCCCC"}))
    a = PlanNode(id="b", tool="amber-build", step=2, settings=TOOLS["amber-build"].normalise({"water": "opc"}))
    wf = WorkflowParams(project_name="apt1", stamp=STAMP, nodes=[b, a],
                        edges=[PlanEdge(source="s", output="pdb", target="b", input="structure")])
    plan = Plan(wf, str(tmp_path / "plan"))
    leap = next(f.content for f in plan.generate()["b"] if f.name.endswith(".leap") and "tleap" in f.name)
    assert "source leaprc.DNA.OL24" in leap and "loadOff LDNA.lib" in leap


# ------------------------------------------------------------------------------- with AmberTools only

def _amberhome():
    home = os.environ.get("AMBERHOME")
    if not home and shutil.which("tleap"):
        home = os.path.dirname(os.path.dirname(os.path.realpath(shutil.which("tleap"))))
    return home if home and os.path.isdir(os.path.join(home, "dat", "leap")) and shutil.which("sander") else None


def _units(text):
    out = {}
    for head, lines in lrna_lib._entries(text):
        m = re.match(r"!entry\.([^.]+)\.unit\.(atoms|positions|connectivity)", head)
        if not m:
            continue
        u = out.setdefault(m.group(1), {"atoms": [], "pos": [], "bonds": []})
        if m.group(2) == "atoms":
            u["atoms"] = [ln.split('"')[3] for ln in lines if ln.count('"') >= 4]
        elif m.group(2) == "positions":
            u["pos"] = [tuple(float(v) for v in ln.split()) for ln in lines if ln.strip()]
        else:
            u["bonds"] = [(int(ln.split()[0]) - 1, int(ln.split()[1]) - 1) for ln in lines if ln.strip()]
    return out


def _dihedral_energy(types, pos, bonds, params):
    adj = {i: set() for i in range(len(types))}
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)

    def phi(a, b, c, d):
        def sub(p, q): return [p[i] - q[i] for i in range(3)]
        def cross(p, q): return [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]
        def dot(p, q): return sum(p[i] * q[i] for i in range(3))
        b0, b1, b2 = sub(a, b), sub(c, b), sub(d, c)
        n1, n2 = cross(b0, b1), cross(b1, b2)
        m1 = cross(n1, [x / math.sqrt(dot(b1, b1)) for x in b1])
        return math.atan2(dot(m1, n2), dot(n1, n2))

    e, seen, missing = 0.0, set(), 0
    for b, c in bonds:
        for bb, cc in ((b, c), (c, b)):
            for a in adj[bb] - {cc}:
                for d in adj[cc] - {bb, a}:
                    if (a, bb, cc, d) in seen or (d, cc, bb, a) in seen:
                        continue
                    seen.add((a, bb, cc, d))
                    key = tuple(types[i] for i in (a, bb, cc, d))
                    terms = next((params["DIHE"][k] for k in (key, key[::-1], ("X", key[1], key[2], "X"),
                                                               ("X", key[2], key[1], "X")) if k in params["DIHE"]),
                                 None)
                    if terms is None:
                        missing += 1
                        continue
                    x = phi(*(pos[i] for i in (a, bb, cc, d)))
                    for ln in terms:
                        f = ln[lrna_lib.WIDTH["DIHE"]:].split()
                        e += float(f[1]) / float(f[0]) * (1 + math.cos(abs(float(f[3])) * x - math.radians(float(f[2]))))
    return e, missing


@pytest.mark.skipif(_amberhome() is None, reason="AmberTools not available")
@pytest.mark.parametrize("ff", list(amber.DNA_FORCE_FIELDS))
def test_every_l_unit_has_the_dihedral_energy_of_its_d_unit(tmp_path, ff, dna_mode):
    """At the exact mirror coordinates; and without the twin types (the control) it would not."""
    leap = os.path.join(_amberhome(), "dat", "leap")
    label, _rc, _short, lib, parms = amber.DNA_FORCE_FIELDS[ff]
    paths = [os.path.join(leap, "parm", p) for p in parms if os.path.exists(os.path.join(leap, "parm", p))]
    assert lrna_lib.main([os.path.join(leap, "lib", lib)] + paths + [str(tmp_path)]) == 0
    sets = [lrna_lib.read_dat(open(paths[0], errors="replace").read())]
    sets += [lrna_lib.read_frcmod(open(p, errors="replace").read()) for p in paths[1:] + [str(tmp_path / "frcmod.ldna")]]
    params = lrna_lib.merge(sets)
    d_units = _units(open(os.path.join(leap, "lib", lib)).read())
    l_units = _units((tmp_path / "LDNA.lib").read_text())
    worst_control = 0.0
    for name, u in d_units.items():
        if not re.match(DNA["NUCLEOTIDE_PATTERN"], name):
            continue
        mirror = [(x, y, -z) for x, y, z in u["pos"]]
        e_d, miss_d = _dihedral_energy(u["atoms"], u["pos"], u["bonds"], params)
        e_l, miss_l = _dihedral_energy(l_units["L" + name]["atoms"], mirror, u["bonds"], params)
        assert miss_d == miss_l == 0 and e_l == pytest.approx(e_d, abs=1e-9), (name, e_d, e_l)
        worst_control = max(worst_control, abs(_dihedral_energy(u["atoms"], mirror, u["bonds"], params)[0] - e_d))
    assert worst_control > 1.0, label  # the twins matter


TERMS = ("BOND", "ANGLE", "DIHED", "VDWAALS", "EEL", "1-4 VDW", "1-4 EEL")


@pytest.mark.skipif(_amberhome() is None, reason="AmberTools (tleap, sander) not available")
@pytest.mark.parametrize("ff", list(amber.DNA_FORCE_FIELDS))
def test_l_dna_duplex_has_the_energy_of_its_mirror_image(tmp_path, ff, dna_mode):
    home = _amberhome()
    leap_dir = os.path.join(home, "dat", "leap")
    label, leaprc, _short, lib, parms = amber.DNA_FORCE_FIELDS[ff]
    paths = [os.path.join(leap_dir, "parm", p) for p in parms]
    assert lrna_lib.main([os.path.join(leap_dir, "lib", lib)] + paths + [str(tmp_path)]) == 0
    base = f"source {leaprc}\n"
    l_leap = (base + 'addAtomTypes { { "LS" "O" "sp3" } { "LT" "O" "sp3" } }\n'
              "loadamberparams frcmod.ldna\nloadOff LDNA.lib\n")
    energies = {}
    for hand, leap in (("D", base), ("L", l_leap)):
        (tmp_path / f"{hand}.pdb").write_text(builder.build_helix("GACTGCAT", na="dna", chirality=hand,
                                                                  scheme="amber"))
        (tmp_path / f"{hand}.leap").write_text(leap + f'm = loadpdb {hand}.pdb\nsaveamberparm m {hand}.parm7 '
                                               f'{hand}.rst7\nquit\n')
        env = dict(os.environ, AMBERHOME=home)
        subprocess.run(["tleap", "-f", f"{hand}.leap"], cwd=tmp_path, env=env, capture_output=True, timeout=600,
                       check=True)
        (tmp_path / "min.in").write_text("single point\n &cntrl\n  imin=1, maxcyc=0, ntb=0, igb=0, cut=999.0,\n /\n")
        subprocess.run(["sander", "-O", "-i", "min.in", "-p", f"{hand}.parm7", "-c", f"{hand}.rst7", "-o",
                        f"{hand}.out", "-r", f"{hand}.restrt"], cwd=tmp_path, env=env, capture_output=True,
                       timeout=600, check=True)
        text = (tmp_path / f"{hand}.out").read_text()
        block = text[text.index("NSTEP"):]
        energies[hand] = {t: float(re.search(re.escape(t) + r"\s*=\s*(-?\d+\.\d+)", block).group(1)) for t in TERMS}
    for term in TERMS:
        assert energies["L"][term] == pytest.approx(energies["D"][term], abs=0.01 + 1e-5 * abs(energies["D"][term])), \
            (label, term, energies)
    assert all(t in (tmp_path / "L.parm7").read_text() for t in ("LS",))
