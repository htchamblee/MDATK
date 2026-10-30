"""L-RNA residue names: the Amber build's planner check that the names MDATK writes (LA LC LG LU,
LA5 LA3 ...) are units of the L-RNA library or mapped onto them - AmberTools' atomic_ions.lib defines LA
(lanthanum) and LU (lutetium), so an unmapped LA / LU builds silently as an ion - and namd-mdff's declared
L residues for topologies whose residue names do not tell the hand."""

import pytest

import mdff_system as ms
from mdatk.config import PlanNode, WorkflowParams
from mdatk.workflows import TOOLS
from mdatk.workflows.plan import Plan
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
