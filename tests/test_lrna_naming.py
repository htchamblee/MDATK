"""L-RNA residue names: the Amber build's checks - when planning, and in the build script before tleap -
that the names MDATK writes (LA LC LG LU, LA5 LA3 ...) are units of the L-RNA library or mapped onto them -
AmberTools' atomic_ions.lib defines LA (lanthanum) and LU (lutetium), so an unmapped LA / LU builds silently
as an ion - and namd-mdff's declared L residues for topologies whose residue names do not tell the hand."""

import os
import subprocess
import sys

import pytest

import mdff_system as ms
from mdatk.config import PlanEdge, PlanNode, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan
from mdatk.workflows.scripts import lrna_names
from mdatk.workflows.scripts import namd_mdff as nm

D_TARGETS = {"C1'": -35.01, "C2'": 40.03, "C3'": 38.30, "C4'": 33.31}


def _lib(path, units):
    """A minimal OFF library: the unit index and an atoms table per unit (all the check reads)."""
    lines = ["!!index array str"] + [f' "{u}"' for u in units]
    for u in units:
        lines += [f'!entry.{u}.unit.atoms table  str name  str type  int typex  int resx  int flags  int seq  '
                  'int elmnt  dbl chg', ' "P" "P" 0 1 131072 1 15 1.166200']
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _build(tmp_path, **settings):
    (tmp_path / "dl.pdb").write_text(ms.mixed_pdb())
    s = TOOLS["amber-build"].normalise(dict({"water": "tip3p", "lrna": True, "extra_frcmod": "frcmod.LRNA"},
                                            **settings))
    node = PlanNode(id="b", tool="amber-build", step=1, settings=s, files={"structure": str(tmp_path / "dl.pdb")})
    st = Plan(WorkflowParams(project_name="apt1", stamp="20261015-1432", nodes=[node]), str(tmp_path / "p")).steps["b"]
    return [(i.level, i.message) for i in st.issues]


A0_UNITS = [f"{b}0{t}" for b in "ACGU" for t in ("", "5", "3")]  # an A0-style library (A0 A05 A03 ...)


def test_unmapped_l_names_are_an_error_naming_the_lanthanide_trap(tmp_path):
    lib = _lib(tmp_path / "LRNA_cpsym.lib", A0_UNITS)
    issues = _build(tmp_path, extra_libs=lib)
    errors = [m for lv, m in issues if lv == "error"]
    assert len(errors) == 1, issues
    assert "does not define LC, LC3, LG, LG5" in errors[0] or "does not define" in errors[0]
    # the test system's L strands are LG5 LC LG LC3: no A or U, so no lanthanide; with an A it is named
    assert 'addPdbResMap { { "LC" "C0" } { "LC3" "C03" } { "LG" "G0" } { "LG5" "G05" } }' in errors[0]


def test_lanthanum_and_lutetium_are_named(tmp_path):
    (tmp_path / "au.pdb").write_text(ms.mixed_pdb("GAUC"))
    lib = _lib(tmp_path / "L.lib", A0_UNITS)
    s = TOOLS["amber-build"].normalise({"water": "tip3p", "lrna": True, "extra_libs": lib, "extra_frcmod": "x"})
    node = PlanNode(id="b", tool="amber-build", step=1, settings=s, files={"structure": str(tmp_path / "au.pdb")})
    st = Plan(WorkflowParams(project_name="apt1", stamp="20261015-1432", nodes=[node]), str(tmp_path / "p")).steps["b"]
    errors = [i.message for i in st.issues if i.level == "error"]
    assert any("silently read LA and LU as the lanthanum and lutetium ion" in m for m in errors), errors


def test_mapped_or_matching_libraries_pass(tmp_path):
    lib = _lib(tmp_path / "LRNA_cpsym.lib", A0_UNITS)
    mapping = 'addPdbResMap { { "LC" "C0" } { "LC3" "C03" } { "LG" "G0" } { "LG5" "G05" } }'
    assert not [m for lv, m in _build(tmp_path, extra_libs=lib, leap_before_load=mapping) if lv == "error"]
    dudek = _lib(tmp_path / "nucleic12_LRNA.lib", [f"L{b}{t}" for b in "ACGU" for t in ("", "5", "3")])
    assert not [m for lv, m in _build(tmp_path, extra_libs=dudek) if lv == "error"]
    # a library only leap's search path knows cannot be read: a warning, not an error
    issues = _build(tmp_path, extra_libs="not_here_LRNA.lib")
    assert any(lv == "warning" and "cannot check that the L residue names" in m for lv, m in issues), issues


# -------------------------------------------------- a structure from an earlier step: plan time and run time
A0_MAP = ("addPdbResMap {\n" + "\n".join(f'  {{ "L{u[0]}{u[2:]}" "{u}" }}' for u in A0_UNITS)
          + "\n}  # the library's A0-style units")
DUDEK_UNITS = [f"L{b}{t}" for b in "ACGU" for t in ("", "5", "3")]


def _upstream(tmp_path, **settings):
    """The build with its structure from an earlier step (a helix the plan has not built yet)."""
    s = TOOLS["amber-build"].normalise(dict({"water": "tip3p", "lrna": True, "extra_frcmod": "frcmod.LRNA"},
                                            **settings))
    helix = PlanNode(id="h", tool="structure-builder", step=1,
                     settings=TOOLS["structure-builder"].normalise({"sequence": "GGCAUC"}))
    build = PlanNode(id="b", tool="amber-build", step=2, settings=s)
    wf = WorkflowParams(project_name="apt1", stamp="20261015-1432", nodes=[helix, build],
                        edges=[PlanEdge("h", "pdb", "b", "structure")])
    return Plan(wf, str(tmp_path / "p"))


def _name_warnings(plan):
    st = plan.steps["b"]
    assert not [i for i in st.issues if i.level == "error"], st.issues
    return [i.message for i in st.issues
            if i.level == "warning" and "names MDATK writes for L residues" in i.message]


def test_names_are_checked_at_plan_time_when_an_earlier_step_makes_the_structure(tmp_path):
    lib = _lib(tmp_path / "LRNA_cpsym.lib", A0_UNITS)
    warned = _name_warnings(_upstream(tmp_path, extra_libs=lib))
    assert len(warned) == 1
    msg = warned[0]
    # every name the L naming may write (a single-residue L chain is written LA5: there is no LAN)
    assert "does not define LA, LA3, LA5, LC, LC3, LC5, LG, LG3, LG5, LU, LU3, LU5," in msg
    assert "LAN" not in msg
    assert "silently read LA and LU as the lanthanum and lutetium ion of AmberTools' atomic_ions.lib" in msg
    assert '{ "LA" "A0" } { "LA3" "A03" } { "LA5" "A05" } { "LC" "C0" }' in msg
    assert "before tleap runs" in msg
    plain = _name_warnings(_upstream(tmp_path, extra_libs=lib, lrna_terminals=False))
    assert len(plain) == 1 and "does not define LA, LC, LG, LU," in plain[0]
    # mapped (one addPdbResMap over several lines), Dudek-named, generated or not readable: no warning
    assert not _name_warnings(_upstream(tmp_path, extra_libs=lib, leap_before_load=A0_MAP))
    dudek = _lib(tmp_path / "nucleic12_LRNA.lib", DUDEK_UNITS)
    assert not _name_warnings(_upstream(tmp_path, extra_libs=dudek))
    assert not _name_warnings(_upstream(tmp_path, leap_before_load=f"loadOff {dudek}\n" + A0_MAP))
    assert not _name_warnings(_upstream(tmp_path, extra_libs="LRNA_in_the_search_path.lib"))
    assert not _name_warnings(_upstream(tmp_path, extra_libs=lib, lrna=False))
    # an existing PDB keeps the error (the names it really gets), also for a library loaded with loadOff
    issues = _build(tmp_path, extra_frcmod="x", leap_before_load=f"loadOff {lib}")
    errors = [m for lv, m in issues if lv == "error"]
    assert len(errors) == 1 and "does not define LC, LC3, LG, LG5" in errors[0], errors


NAMED_LEAP = ("source leaprc.RNA.OL3\nsource leaprc.water.tip3p\n{libs}\n{before}\n"
              'mol = loadpdb "named.pdb"\n{after}\n')


def _runtime(tmp_path, units, before="", after="", lib_line=None):
    """The run-time check in a step folder: the named PDB (L residues LG5 LA LU LC3), a library and the
    tleap input."""
    (tmp_path / "named.pdb").write_text(ms.mixed_pdb("GAUC"))
    if units is not None:
        _lib(tmp_path / "LRNA_cpsym.lib", units)
    libs = "loadOff LRNA_cpsym.lib" if lib_line is None else lib_line
    (tmp_path / "t.leap").write_text(NAMED_LEAP.format(libs=libs, before=before, after=after))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return lrna_names.check("named.pdb", "t.leap")
    finally:
        os.chdir(cwd)


def test_runtime_check_missing_mapped_and_present_names(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AMBERHOME", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no-amber"))  # no tleap: names in the search path stay unread
    problems, warnings = _runtime(tmp_path, A0_UNITS)
    assert len(problems) == 1 and not warnings
    assert "the L residue names LA, LC3, LG5, LU are neither units" in problems[0]
    assert "silently read LA and LU as the lanthanum and lutetium ion" in problems[0]
    assert 'addPdbResMap { { "LA" "A0" } { "LC3" "C03" } { "LG5" "G05" } { "LU" "U0" } }' in problems[0]
    assert "L-RNA name check FAILED" in capsys.readouterr().out
    assert _runtime(tmp_path, A0_UNITS, before=A0_MAP) == ([], [])  # mapped
    assert _runtime(tmp_path, A0_UNITS, before='lig = loadPdb "lig.pdb"\n' + A0_MAP) == ([], [])
    # a map after loadpdb comes too late; a partial map leaves the rest (here only LA, the lanthanum)
    assert _runtime(tmp_path, A0_UNITS, after=A0_MAP)[0]
    partial = A0_MAP.replace('{ "LA" "A0" }', "")
    assert "names LA are" in _runtime(tmp_path, A0_UNITS, before=partial)[0][0]
    assert "lanthanum ion of" in _runtime(tmp_path, A0_UNITS, before=partial)[0][0]
    assert _runtime(tmp_path, DUDEK_UNITS) == ([], [])  # present
    assert _runtime(tmp_path, A0_UNITS, before="LA = copy A0\nLU = copy U0\nLG5 = copy G05\nLC3 = copy C03") \
        == ([], [])  # units assigned in the leap commands
    # a library that cannot be read: a warning, and the build goes on
    problems, warnings = _runtime(tmp_path, None, lib_line="loadOff not_there.lib")
    assert not problems and "cannot read not_there.lib" in warnings[0] and "may define them" in warnings[1]
    # a lanthanum ion named LA (no sugar or phosphate atoms) is not an L residue
    (tmp_path / "ion.pdb").write_text("HETATM    1 LA    LA I   1       0.000   0.000   0.000  1.00  0.00"
                                      "          LA\nEND\n")
    assert lrna_names.l_names(str(tmp_path / "ion.pdb")) == []


def test_runtime_check_finds_libraries_in_leaps_search_path(tmp_path, monkeypatch):
    home = tmp_path / "amber"
    (home / "dat" / "leap" / "lib").mkdir(parents=True)
    step = tmp_path / "step"
    step.mkdir()
    _lib(home / "dat" / "leap" / "lib" / "nucleic12_LRNA.lib", DUDEK_UNITS)
    monkeypatch.setenv("AMBERHOME", str(home))
    assert _runtime(step, None, lib_line='loadOff "nucleic12_LRNA.lib"') == ([], [])
    # AMBERHOME from tleap on PATH, as the build script finds it
    (home / "bin").mkdir()
    (home / "bin" / "tleap").write_text("#!/bin/sh\n")
    (home / "bin" / "tleap").chmod(0o755)
    monkeypatch.delenv("AMBERHOME")
    monkeypatch.setenv("PATH", str(home / "bin"))
    assert lrna_names.amberhome() == os.path.realpath(home)
    assert _runtime(step, None, lib_line="loadoff nucleic12_LRNA.lib") == ([], [])


def _script(plan, step="b"):
    return {f.name.split("_", 2)[2]: f for f in plan.generate([step])[step]}


def test_build_script_checks_the_names_before_tleap_only_with_own_libraries(tmp_path):
    lib = _lib(tmp_path / "LRNA_cpsym.lib", A0_UNITS)
    files = _script(_upstream(tmp_path, extra_libs=lib))
    sh = files["amber-build_20261015-1432.sh"].content
    line = ("python3 apt1_02_lrna-names_20261015-1432.py apt1_02_lrna_20261015-1432.pdb "
            "apt1_02_tleap_20261015-1432.leap || {")
    assert line in sh and sh.index("--scheme amber") < sh.index(line) < sh.index("tleap -f")
    assert files["lrna-names_20261015-1432.py"].role == "helper"
    two = _script(_upstream(tmp_path, leap_before_load=f"loadOff {lib}", two_pass=True))
    sh = two["amber-build_20261015-1432.sh"].content
    assert "apt1_02_tleap-dry_20261015-1432.leap || {" in sh
    assert sh.index("lrna-names_20261015-1432.py") < sh.index("tleap -f apt1_02_tleap-dry")
    for settings in ({}, {"extra_libs": lib, "lrna": False}, {"extra_libs": lib, "lrna_library": "generated"},
                     {"leap_before_load": A0_MAP}):
        files = _script(_upstream(tmp_path, **settings))
        assert "lrna-names_20261015-1432.py" not in files, settings
        assert "lrna-names" not in files["amber-build_20261015-1432.sh"].content, settings


def test_generated_check_runs_on_its_own(tmp_path, monkeypatch):
    """The embedded script, as written into the step folder, with nothing but python3."""
    monkeypatch.delenv("AMBERHOME", raising=False)
    lib = _lib(tmp_path / "LRNA_cpsym.lib", A0_UNITS)
    files = _script(_upstream(tmp_path, extra_libs=lib))
    folder = tmp_path / "step"
    folder.mkdir()
    (folder / "check.py").write_text(files["lrna-names_20261015-1432.py"].content)
    (folder / "named.pdb").write_text(ms.mixed_pdb("GAUC"))
    _lib(folder / "RNA_CP.lib", ["A", "A5", "A3", "C", "G", "U"])  # the CP helper writes it earlier
    leap = files["tleap_20261015-1432.leap"].content
    assert "loadOff RNA_CP.lib" in leap and f"loadOff {lib}" in leap
    (folder / "t.leap").write_text(leap)
    env = {"PATH": os.path.dirname(sys.executable)}
    run = subprocess.run([sys.executable, "-I", "check.py", "named.pdb", "t.leap"], cwd=folder, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 1 and "lanthanum and lutetium" in run.stdout, run.stdout + run.stderr
    (folder / "t.leap").write_text(leap.replace("mol = loadpdb", A0_MAP + "\nmol = loadpdb"))
    run = subprocess.run([sys.executable, "-I", "check.py", "named.pdb", "t.leap"], cwd=folder, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0 and "L-RNA name check passed" in run.stdout, run.stdout + run.stderr


def test_declared_l_residues_replace_the_names_and_the_geometry_must_agree(tmp_path):
    # an L-RNA library that writes D names into its units: the L strands come out as G5 C G C3
    atoms = [a[:2] + (a[2][1:] if a[2].startswith("L") else a[2],) + a[3:] for a in ms.atoms_of(ms.mixed_pdb())]
    ms.write_topology(str(tmp_path / "t.parm7"), atoms, l_types=False)
    top = nm.Topology(str(tmp_path / "t.parm7"))
    xyz = [a[5] for a in atoms]
    _lines, problems, _notes = nm.chirality_lines(top, xyz, 200.0, D_TARGETS)
    assert len(problems) == 8 and "residue 9 (G5) is named as D but its sugar has L geometry" in problems[0]
    lines, problems, _notes = nm.chirality_lines(top, xyz, 200.0, D_TARGETS, nm.parse_residues("9-16"))
    assert not problems and sum(ln.startswith("improper") for ln in lines) == 64
    _lines, problems, _notes = nm.chirality_lines(top, xyz, 200.0, D_TARGETS, nm.parse_residues("5-12"))
    assert any("residue 5 (G5) is declared L but its sugar has D geometry" in p for p in problems)
    with pytest.raises(nm.PrepError, match="cannot read the residue range"):
        nm.parse_residues("9-x")


def test_declared_l_residues_reach_the_planner_and_the_script(tmp_path):
    from test_namd_mdff_prep import config
    atoms = [a[:2] + (a[2][1:] if a[2].startswith("L") else a[2],) + a[3:] for a in ms.atoms_of(ms.mixed_pdb())]
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms, l_types=False)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in atoms])
    files = {"topology": str(tmp_path / "sys.parm7"), "coordinates": str(tmp_path / "sys.rst7"),
             "map": "/d/map.mrc", "model": "/d/model.pdb"}

    def errors(settings):
        node = PlanNode(id="m", tool="namd-mdff", step=2, settings=TOOLS["namd-mdff"].normalise(settings), files=files)
        st = Plan(WorkflowParams(project_name="apt1", stamp="20261015-1432", nodes=[node]), str(tmp_path / "p")).steps["m"]
        return [i.message for i in st.issues if i.level == "error"]

    assert any("named as D but its sugar has L geometry" in m for m in errors({"solvent": "vacuum"}))
    assert not errors({"solvent": "vacuum", "l_residues": "9-16"})
    assert any(m.startswith("L residues: cannot read") for m in errors({"solvent": "vacuum", "l_residues": "9-"}))
    cfg = config(tmp_path)
    cfg["chirality"]["l_residues"] = list(range(9, 17))
    assert nm.prep(cfg, write=False) == 0
    cfg["chirality"]["l_residues"] = None
    with pytest.raises(nm.PrepError, match="named as D but its sugar has L geometry"):
        nm.prep(cfg, write=False)
