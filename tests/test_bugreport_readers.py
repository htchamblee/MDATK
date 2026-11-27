"""Regression tests for the reader, selection and command-line items of the external bug report on
MDATK 0.1.1 (MDATK-BUG-028, 030, 039-056, 065, 066, 072, 083-085)."""

import numpy as np
import pytest

from conftest import make_universe
from mdatk.selections import SelectionError, check_selection, select


# ---------------------------------------------------------------------------
# selections (052-056)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def masses_universe():
    """Six atoms with the masses 10, 12, 14, 14, 16, 18 and the charges -1 ... 1.5."""
    masses = [10.0, 12.0, 14.0, 14.0, 16.0, 18.0]
    atoms = [{"name": f"A{k}", "resid": k + 1, "resname": "RES", "element": "C", "mass": m,
              "charge": -1.0 + 0.5 * k} for k, m in enumerate(masses)]
    coords = np.zeros((1, len(atoms), 3))
    coords[0, :, 0] = np.arange(len(atoms)) * 3.0
    return make_universe(atoms, coords)


def _indices(u, text):
    return list(select(u, text, allow_empty=True).indices)


def test_bug052_reversed_prop_comparison_is_mirrored_not_negated(masses_universe):
    u = masses_universe
    pairs = [("prop mass > 14", "prop 14 < mass"), ("prop mass >= 14", "prop 14 <= mass"),
             ("prop mass < 14", "prop 14 > mass"), ("prop mass <= 14", "prop 14 >= mass"),
             ("prop mass == 14", "prop 14 == mass"), ("prop mass != 14", "prop 14 != mass")]
    for plain, reversed_ in pairs:
        assert _indices(u, plain) == _indices(u, reversed_), (plain, reversed_)
    assert _indices(u, "prop 14 < mass") == [4, 5] and _indices(u, "prop 14 <= mass") == [2, 3, 4, 5]
    # every spacing MDAnalysis accepts, and 'abs'
    for text in ("prop 14<mass", "prop 14 <mass", "prop 14< mass", "not (prop 14 >= mass)"):
        assert _indices(u, text) == [4, 5], text
    assert _indices(u, "prop abs 1 <= charge") == _indices(u, "prop abs charge >= 1") == [0, 4, 5]
    assert _indices(u, "prop mass>14") == _indices(u, "prop mass > 14") == [4, 5]  # unchanged


@pytest.mark.parametrize("text, operator", [("prop mass ><14", "><"), ("prop mass <> 14", "<>"),
                                            ("prop mass => 14", "=>"), ("prop mass >> 14", ">>")])
def test_bug053_malformed_prop_operator_is_an_error(masses_universe, text, operator):
    for run in (lambda: select(masses_universe, text, allow_empty=True), lambda: check_selection(text)):
        with pytest.raises(SelectionError) as info:
            run()
        message = str(info.value)
        assert f"invalid operator {operator!r} in prop (use <, >, <=, >=, ==, !=)" in message
        assert info.value.kind == "syntax" and "float" not in message


@pytest.mark.parametrize("text", ["atom SYS 1 and", "atom and 1 A0", "atom SYS 1 )", "atom SYS 1",
                                  "atom SYS x A0", "name A1 or atom SYS 1 or"])
def test_bug054_atom_does_not_take_keywords_as_arguments(masses_universe, text):
    for run in (lambda: select(masses_universe, text, allow_empty=True), lambda: check_selection(text)):
        with pytest.raises(SelectionError, match="atom: expected segid resid name"):
            run()


def test_bug054_atom_still_selects(masses_universe):
    u = masses_universe
    assert _indices(u, "atom SYS 2 A1") == [1]
    assert _indices(u, "atom SYS 2 A1 or atom SYS 4 A3") == [1, 3]
    assert _indices(u, "(atom SYS 2 A1) and name A1") == [1]
    assert _indices(u, r"atom \SYS 2 \A1") == [1]  # the escape for names that are keywords


def test_bug055_negative_distance_is_an_error_whatever_the_atom_counts(masses_universe):
    u = masses_universe
    for text in ("around -2 resid 1", "around -2 (resid 1 and name A0)", "resid 2 and around -0.5 resid 1",
                 "point 0 0 0 -3"):
        with pytest.raises(SelectionError, match="distance must not be negative"):
            select(u, text, allow_empty=True)
        with pytest.raises(SelectionError, match="distance must not be negative"):
            check_selection(text)
    assert _indices(u, "around 3.5 resid 1") == [1] and _indices(u, "around 0 resid 1") == []
    assert _indices(u, "point 0 0 0 3.5") == [0, 1]


def test_bug056_missing_arguments_name_the_keyword(masses_universe):
    u = masses_universe
    for text, expected in (("around", "around: expected a distance and a selection, e.g. 'around 3.5 resname MG'"),
                           ("around 3", "around: expected a distance and a selection"),
                           ("around x resid 1", "around: expected a distance and a selection"),
                           ("point 1 2 3", "point: expected x y z distance"),
                           ("prop mass >", "prop: expected a property, an operator and a value")):
        with pytest.raises(SelectionError) as info:
            select(u, text, allow_empty=True)
        assert expected in str(info.value) and "NoneType" not in str(info.value), text
        with pytest.raises(SelectionError):
            check_selection(text)


def test_bug056_float_range_bounds_take_any_float(masses_universe):
    u = masses_universe
    assert _indices(u, "mass 12:14") == [1, 2, 3]  # as before
    for text in ("mass 12:+14", "mass 1.2E1:1.4E1", "mass 1.2e1:1.4E1", "mass 12 to +14", "mass 12 : +14"):
        assert _indices(u, text) == [1, 2, 3], text
    assert _indices(u, "mass 16:inf") == [4, 5] and _indices(u, "mass -inf:12") == [0, 1]
    assert _indices(u, "mass -inf:inf") == [0, 1, 2, 3, 4, 5]
    assert _indices(u, "charge -1.0:-0.5") == [0, 1] and _indices(u, "charge -1--0.5") == [0, 1]
    assert _indices(u, "mass 10 12:14 18") == [0, 1, 2, 3, 5]
    for text in ("charge :1", "mass 12:14abc", "mass 12:13:14"):
        with pytest.raises(SelectionError) as info:
            select(u, text, allow_empty=True)
        assert "cannot read the range" in str(info.value) and "float" not in str(info.value), text


def test_bug056_upper_case_keywords_get_a_hint(masses_universe):
    for text, keyword in (("resid 1 AND name A0", "and"), ("name A0 AND resid 1", "and"),
                          ("resid 1 OR resid 2", "or")):
        with pytest.raises(SelectionError) as info:
            select(masses_universe, text, allow_empty=True)
        assert f"keywords are lower case: did you mean {keyword!r}?" in str(info.value), text
        with pytest.raises(SelectionError, match="keywords are lower case"):
            check_selection(text)


# ---------------------------------------------------------------------------
# file readers and the loader (039-051, 072)
# ---------------------------------------------------------------------------

TWO_PDB = ("ATOM      1  O1  HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
           "ATOM      2  H1  HOH A   1       0.960   0.000   0.000  1.00  0.00           H\n"
           "END\n")
ONE_PDB = "ATOM      1  O1  HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n"
MDCRD_FRAME = "%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n" % (0, 0, 0, 0.96, 0, 0)
GRO_FRAME = "frame\n    1\n    1SOL     OW    1   1.000   2.000   3.000\n   3.00000   3.00000   3.00000\n"
NO_TIMES = "no time information in the file: frames are taken as 1 ps apart (set the time step, inputs.dt_ps)"


def _write(tmp_path, name, text):
    path = tmp_path / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    return str(path)


def _load(top, *trajectories, **kwargs):
    from mdatk.io.loader import load_universe
    return load_universe(top, trajectories, **kwargs)


def _reader_error(top, *trajectories):
    """The message of the ReaderError that loading raises (it is a ValueError: the CLI prints it)."""
    from mdatk.io.loader import ReaderError
    with pytest.raises(ReaderError) as info:
        _load(top, *trajectories)
    assert isinstance(info.value, ValueError)
    message = str(info.value)
    assert "<class" not in message and "MDAnalysis.topology" not in message
    return message


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and r.name == "mdatk.io.loader"]


def _netcdf(path, lengths, n_atoms=2):
    from scipy.io import netcdf_file
    f = netcdf_file(str(path), "w", version=2)
    f.Conventions, f.ConventionVersion, f.program, f.programVersion, f.title = "AMBER", "1.0", "x", "1", "t"
    for name, size in (("frame", None), ("spatial", 3), ("atom", n_atoms), ("cell_spatial", 3),
                       ("cell_angular", 3), ("label", 5)):
        f.createDimension(name, size)
    f.createVariable("spatial", "c", ("spatial",))[:] = b"xyz"
    f.createVariable("cell_spatial", "c", ("cell_spatial",))[:] = b"abc"
    f.createVariable("cell_angular", "c", ("cell_angular", "label"))[:] = [b"alpha", b"beta ", b"gamma"]
    time = f.createVariable("time", "f", ("frame",))
    time.units = "picosecond"
    xyz = f.createVariable("coordinates", "f", ("frame", "atom", "spatial"))
    xyz.units = "angstrom"
    cell = f.createVariable("cell_lengths", "d", ("frame", "cell_spatial"))
    cell.units = "angstrom"
    angles = f.createVariable("cell_angles", "d", ("frame", "cell_angular"))
    angles.units = "degree"
    for k in range(2):
        time[k], cell[k], angles[k] = 2.0 * k, lengths, [90, 90, 90]
        xyz[k] = np.arange(3 * n_atoms).reshape(n_atoms, 3) + k
    f.close()
    return str(path)


def test_bug039_pdb_conect_records_with_hybrid36_serials(tmp_path):
    pdb = _write(tmp_path, "b01.pdb",
                 "ATOM  A0000  O1  HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
                 "ATOM  A0001  H1  HOH A   1       0.960   0.000   0.000  1.00  0.00           H\n"
                 "ATOM  A0002  H2  HOH A   1      -0.240   0.930   0.000  1.00  0.00           H\n"
                 "CONECTA0000A0001A0002\nCONECTA0001A0000\nEND\n")
    u = _load(pdb)
    assert list(u.atoms.ids) == [100000, 100001, 100002]
    assert sorted(map(tuple, u.bonds.indices)) == [(0, 1), (0, 2)]
    assert u.trajectory.n_frames == 1 and u.atoms.positions[1, 0] == pytest.approx(0.96)
    # decimal CONECT records are read by MDAnalysis' own parser, as before
    plain = _write(tmp_path, "plain.pdb", TWO_PDB.replace("END\n", "CONECT    1    2\nEND\n"))
    assert sorted(map(tuple, _load(plain).bonds.indices)) == [(0, 1)]
    import MDAnalysis as mda
    assert mda._PARSERS["PDB"].__module__ == "MDAnalysis.topology.PDBParser"  # nothing registered globally


def test_bug040_one_frame_mdcrd_without_box_is_a_clear_error(tmp_path):
    pdb = _write(tmp_path, "two.pdb", TWO_PDB)
    crd = _write(tmp_path, "one_frame.mdcrd", "title\n" + MDCRD_FRAME)
    message = _reader_error(pdb, crd)
    assert message.startswith(f"{crd}: cannot read the Amber mdcrd trajectory: the file ends with its first frame")
    assert "single frame without a box line" in message


def test_bug041_netcdf_with_zero_cell_lengths_has_no_cell(tmp_path, caplog):
    pdb = _write(tmp_path, "two.pdb", TWO_PDB)
    nc = _netcdf(tmp_path / "zero_box.nc", [0, 0, 0])
    with caplog.at_level("WARNING"):
        u = _load(pdb, nc)
    assert u.trajectory.n_frames == 2
    for k, ts in enumerate(u.trajectory):
        assert ts.dimensions is None and ts.time == pytest.approx(2.0 * k)
        assert np.allclose(u.atoms.positions, np.arange(6).reshape(2, 3) + k)
    assert _warnings(caplog) == [f"{nc}: the cell lengths are 0 0 0: read as a trajectory without a periodic cell"]
    caplog.clear()
    boxed = _netcdf(tmp_path / "box.nc", [20, 21, 22])  # a real cell is read as before, without a warning
    with caplog.at_level("WARNING"):
        u = _load(pdb, boxed)
    assert np.allclose(u.dimensions, [20, 21, 22, 90, 90, 90]) and _warnings(caplog) == []


def test_bug042_parm7_unknown_section_after_pointers_is_named(demo_dry, tmp_path):
    text = open(demo_dry["topology"]).read()
    i = text.index("%FLAG POINTERS")
    j = text.index("%FLAG", i + 5)
    top = _write(tmp_path, "unknown_first.parm7",
                 text[:j] + "%FLAG MY_EXTRA_SECTION\n%FORMAT(10I8)\n       1\n" + text[j:])
    message = _reader_error(top, demo_dry["trajectory"])
    assert message.startswith(f"{top}: cannot read the Amber parm7 topology: the section %FLAG MY_EXTRA_SECTION "
                              "directly after POINTERS")
    assert "numlines" not in message


def test_bug043_dcd_with_frame_interval_zero(tmp_path):
    import struct

    import MDAnalysis as mda
    pdb = _write(tmp_path, "two.pdb", TWO_PDB)
    dcd = str(tmp_path / "ok.dcd")
    u = mda.Universe(pdb)
    with mda.Writer(dcd, 2) as w:
        w.write(u.atoms)
        w.write(u.atoms)
    assert _load(pdb, dcd).trajectory.n_frames == 2
    raw = bytearray(open(dcd, "rb").read())
    struct.pack_into("<i", raw, 16, 0)  # NSAVC
    bad = _write(tmp_path, "nsavc0.dcd", bytes(raw))
    assert _reader_error(pdb, bad) == f"{bad}: DCD header has frame interval (NSAVC) 0"


def test_bug044_chain_of_trajectories_with_different_extensions(tmp_path):
    pdb = _write(tmp_path, "two.pdb", TWO_PDB)
    a = _write(tmp_path, "two.mdcrd", "title\n" + MDCRD_FRAME * 2)
    b = _write(tmp_path, "two.crd", "title\n" + MDCRD_FRAME * 2)
    u = _load(pdb, a, b)
    assert u.trajectory.n_frames == 4 and u.atoms.positions[1, 0] == pytest.approx(0.96)
    assert _load(pdb, a, b, dt_ps=2.0).trajectory[3].time == pytest.approx(6.0)
    nc = _netcdf(tmp_path / "box.nc", [20, 21, 22])  # formats mixed in one chain: each file with its own
    u = _load(pdb, a, nc)
    assert u.trajectory.n_frames == 4 and np.allclose(u.trajectory[3].positions, np.arange(6).reshape(2, 3) + 1)


def test_bug045_gro_with_coordinates_wider_than_their_columns(tmp_path):
    gro = _write(tmp_path, "wide.gro", "wide\n    1\n    1SOL     OW    1 10000.000   2.000   3.000\n"
                                       "   3.00000   3.00000   3.00000\n")
    assert _reader_error(gro).startswith(f"{gro} line 3: cannot determine coordinate field width")
    ok = _load(_write(tmp_path, "ok.gro", GRO_FRAME))
    assert np.allclose(ok.atoms.positions, [[10, 20, 30]]) and np.allclose(ok.dimensions[:3], 30)
    precise = _load(_write(tmp_path, "precise.gro", "p\n    1\n    1SOL     OW    1   1.00000   2.00000   3.00000\n"
                                                    "   3.00000   3.00000   3.00000\n"))
    assert np.allclose(precise.atoms.positions, [[10, 20, 30]])


def test_bug046_multi_frame_gro_warns_that_one_frame_is_read(tmp_path, caplog):
    gro = _write(tmp_path, "three_frames.gro", GRO_FRAME * 3)
    with caplog.at_level("WARNING"):
        u = _load(gro)
    assert u.trajectory.n_frames == 1
    assert _warnings(caplog) == [f"{gro}: only the first of 3 frames is read (a GRO file is read as one structure)"]
    caplog.clear()
    with caplog.at_level("WARNING"):
        _load(_write(tmp_path, "one.gro", GRO_FRAME + "\n"))
    assert _warnings(caplog) == []


def test_bug047_one_atom_mdcrd_warns_that_box_lines_are_read_as_frames(tmp_path, caplog):
    pdb = _write(tmp_path, "one_atom.pdb", ONE_PDB)
    crd = _write(tmp_path, "n1.mdcrd", "title\n" + ("%8.3f%8.3f%8.3f\n%8.3f%8.3f%8.3f\n" % (1, 2, 3, 30, 30, 30)) * 3)
    with caplog.at_level("WARNING"):
        _load(pdb, crd)
    assert any(w.startswith(f"{crd}: an mdcrd trajectory of one atom is read without box lines")
               for w in _warnings(caplog))


def test_bug048_pdb_atoms_without_element_get_a_guessed_mass(tmp_path, caplog):
    pdb = _write(tmp_path, "b10.pdb",
                 "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                 "ATOM      2  N2  LIG A   1       0.960   0.000   0.000  1.00  0.00            \n"
                 "ATOM      3  Q9  LIG A   1       1.960   0.000   0.000  1.00  0.00            \nEND\n")
    with caplog.at_level("WARNING"):
        u = _load(pdb)
    assert u.atoms.masses[:2] == pytest.approx([12.011, 14.007]) and u.atoms.masses[2] == 0
    assert _warnings(caplog) == [f"{pdb}: 2 atom(s) have no element in the file; their masses are guessed from "
                                 "the atom names (1 could not be guessed and keep the mass 0)"]


def test_bug048_masses_read_from_a_topology_are_left_alone(demo_dry, caplog):
    import MDAnalysis as mda
    with caplog.at_level("WARNING"):
        u = _load(demo_dry["topology"], demo_dry["trajectory"])
    assert np.array_equal(u.atoms.masses, mda.Universe(demo_dry["topology"]).atoms.masses)
    assert _warnings(caplog) == []  # and the demo system loads without a warning


def test_bug049_reader_warnings_are_logged_with_the_file(tmp_path, caplog):
    pdb = _write(tmp_path, "noelements.pdb", "\n".join(line[:66] for line in TWO_PDB.splitlines()) + "\n")
    crd = _write(tmp_path, "two.mdcrd", "title\n" + MDCRD_FRAME * 2)
    missing = f"{pdb}: Element information is missing, elements attribute will not be populated."
    with caplog.at_level("WARNING"):
        _load(pdb, crd)
        _load(pdb, crd, crd)
    assert _warnings(caplog) == [missing, f"{crd}: {NO_TIMES}", missing, f"{crd}, {crd}: {NO_TIMES}"]
    caplog.clear()
    with caplog.at_level("WARNING"):
        _load(pdb, crd, dt_ps=2.0)  # the time step is given
        _load(_write(tmp_path, "two.pdb", TWO_PDB))  # a single structure has no times
    assert _warnings(caplog) == [missing]


def test_bug050_gromacs_top_is_not_read_as_parm7(tmp_path):
    crd = _write(tmp_path, "two.mdcrd", "title\n" + MDCRD_FRAME * 2)
    top = _write(tmp_path, "gmx.top", "[ defaults ]\n1 2\n")
    assert _reader_error(top, crd) == f"{top}: not an Amber parm7 topology (GROMACS .top files are not supported)"
    prmtop = _write(tmp_path, "x.prmtop", "garbage\n")
    assert _reader_error(prmtop, crd) == f"{prmtop}: not an Amber parm7 topology"


def test_bug051_reader_errors_name_the_file_and_the_problem(demo_dry, tmp_path):
    crd = _write(tmp_path, "two.mdcrd", "title\n" + MDCRD_FRAME * 2)
    pdb = _write(tmp_path, "two.pdb", TWO_PDB)
    for name in ("empty.pdb", "empty.gro"):
        path = _write(tmp_path, name, "")
        assert _reader_error(path) == f"{path}: file is empty"
    path = _write(tmp_path, "empty.mdcrd", "")
    assert _reader_error(pdb, path) == f"{path}: file is empty"
    path = _write(tmp_path, "remark.pdb", "REMARK x\nEND\n")
    assert _reader_error(path) == f"{path}: no ATOM or HETATM records"
    path = _write(tmp_path, "latin.pdb", "REMARK caf\xe9 here\n".encode("latin-1") + TWO_PDB.encode())
    assert _reader_error(path) == f"{path} line 1: not UTF-8 text (byte 0xe9); save the file as UTF-8 or plain ASCII"
    path = _write(tmp_path, "no_box.gro", GRO_FRAME.rsplit("\n", 2)[0] + "\n")
    assert _reader_error(path).startswith(f"{path}: missing box line after atom 1")
    path = _write(tmp_path, "short.gro", GRO_FRAME.replace("    1\n", "    3\n", 1))
    assert _reader_error(path) == f"{path}: the header says 3 atoms but the file ends after 2 atom lines"
    # what MDAnalysis itself rejects: the file, the format in words, its message without the parser class
    path = _write(tmp_path, "bad.pdb", TWO_PDB.replace("0.960", "0.9x0"))
    assert _reader_error(path).startswith(f"{path}: cannot read the PDB coordinates: could not convert")
    path = _write(tmp_path, "cut.parm7", open(demo_dry["topology"]).read()[:4000])
    assert _reader_error(path, crd).startswith(f"{path}: cannot read the Amber parm7 topology: the file is "
                                               "damaged or not in this format (")
    # the trajectory of a chain that cannot be read is the one named
    nc = _netcdf(tmp_path / "three_atoms.nc", [20, 20, 20], n_atoms=3)
    ok = _netcdf(tmp_path / "ok.nc", [20, 20, 20])
    message = _reader_error(pdb, ok, nc)
    assert message.startswith(f"{nc}: cannot read the Amber NetCDF trajectory: ")
    assert "n_atoms can be None" not in message


def test_bug072_gro_box_line_with_run_together_negative_values(tmp_path):
    box = "   3.00000   3.00000   3.00000"
    gro = _write(tmp_path, "neg_box.gro", GRO_FRAME.replace(box, "-1.00000-2.00000-3.00000"))
    assert _reader_error(gro) == (f"{gro} line 4: GRO box line has neither 3 nor 9 values: "
                                  "'-1.00000-2.00000-3.00000'")
    # the forms MDAnalysis reads correctly stay readable: a triclinic box with negative components, and
    # unsigned values without spaces between them
    tri = GRO_FRAME.replace(box, box + "   0.00000   0.00000  -1.00000   0.00000   0.50000   0.50000")
    assert _load(_write(tmp_path, "tri.gro", tri)).dimensions is not None
    joined = _load(_write(tmp_path, "joined.gro", GRO_FRAME.replace(box, "1000.000002000.000003000.00000")))
    assert np.allclose(joined.dimensions, [10000, 20000, 30000, 90, 90, 90])


# ---------------------------------------------------------------------------
# command line and messages (028, 030, 065, 066, 083)
# ---------------------------------------------------------------------------


def _main(capsys, *argv):
    from mdatk.cli import main
    capsys.readouterr()
    code = main(list(argv))
    return code, capsys.readouterr().err


def test_bug028_reader_failures_are_one_line_errors_on_the_command_line(demo_dry, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MDATK_DEBUG", raising=False)
    code, err = _main(capsys, "info", "--top", demo_dry["topology"])
    assert code == 1 and "Traceback" not in err
    assert f"error: {demo_dry['topology']} has no coordinates; give a trajectory with --traj" in err
    pdb = _write(tmp_path, "two.pdb", TWO_PDB)
    cases = [(["--top", _write(tmp_path, "no_box.gro", GRO_FRAME.rsplit("\n", 2)[0] + "\n")], "missing box line"),
             (["--top", _write(tmp_path, "empty.pdb", "")], "file is empty"),
             (["--top", _write(tmp_path, "remark.pdb", "REMARK x\nEND\n")], "no ATOM or HETATM records"),
             (["--top", pdb, "--traj", _write(tmp_path, "one_frame.mdcrd", "title\n" + MDCRD_FRAME)],
              "ends with its first frame"),
             (["--top", pdb, "--traj", _netcdf(tmp_path / "three.nc", [20, 20, 20], n_atoms=3)],
              "cannot read the Amber NetCDF trajectory")]
    for args, expected in cases:
        code, err = _main(capsys, "info", *args)
        assert code == 1 and "Traceback" not in err, args
        assert err.startswith("error: ") and expected in err and args[-1] in err, err


def test_bug028_unexpected_errors_are_one_line_unless_debugging(capsys, monkeypatch):
    from mdatk import cli

    def broken(_args):
        raise TypeError("'NoneType' object is not subscriptable")

    monkeypatch.setattr(cli, "cmd_density_info", broken)
    monkeypatch.delenv("MDATK_DEBUG", raising=False)
    code, err = _main(capsys, "density", "info", "x.mrc")
    assert code == 1
    assert err == "error: TypeError: 'NoneType' object is not subscriptable (set MDATK_DEBUG=1 for the traceback)\n"
    monkeypatch.setenv("MDATK_DEBUG", "1")
    with pytest.raises(TypeError):
        cli.main(["density", "info", "x.mrc"])


def _mrc_header(n, order="<", stamp=(0x44, 0x44, 0, 0), mode=2):
    import struct
    h = bytearray(1024)
    struct.pack_into(order + "3i", h, 0, n, n, n)
    struct.pack_into(order + "i", h, 12, mode)
    struct.pack_into(order + "3i", h, 28, n, n, n)
    struct.pack_into(order + "6f", h, 40, 10, 10, 10, 90, 90, 90)
    struct.pack_into(order + "3i", h, 64, 1, 2, 3)
    h[208:212] = b"MAP "
    h[212:216] = bytes(stamp)
    return bytes(h)


def test_bug030_density_info_on_damaged_maps(tmp_path, capsys, monkeypatch):
    import gzip

    from mdatk.io.mrc import MapError, read_map
    monkeypatch.delenv("MDATK_DEBUG", raising=False)
    huge = _write(tmp_path, "huge.mrc", _mrc_header(100000) + b"\0" * 16)
    cut = _write(tmp_path, "trunc.mrc.gz", gzip.compress(_mrc_header(4) + b"\0" * 256)[:-30])
    notgz = _write(tmp_path, "notgz.mrc.gz", b"\x1f\x8b" + b"garbage" * 300)
    swapped = _write(tmp_path, "swapped.mrc", _mrc_header(4, order=">") + b"\0" * 256)
    nostamp = _write(tmp_path, "nostamp.mrc", _mrc_header(4, stamp=(0, 0, 0, 0), mode=77) + b"\0" * 256)
    for path, expected in ((huge, "file is shorter than its header says"),
                           (cut, "the gzip-compressed file is cut off or damaged"),
                           (notgz, "the gzip-compressed file is cut off or damaged"),
                           (swapped, "cannot determine the byte order of the map header (mode 33554432"),
                           (nostamp, "cannot determine the byte order of the map header")):
        code, err = _main(capsys, "density", "info", path)
        assert code == 1 and "Traceback" not in err, path
        assert err.startswith(f"error: {path}: {expected}"), err
        with pytest.raises(MapError):
            read_map(path)
    good = _write(tmp_path, "good.mrc.gz", gzip.compress(_mrc_header(4) + np.arange(64, dtype="<f4").tobytes()))
    assert read_map(good).data.shape == (4, 4, 4) and read_map(good).data[3, 3, 3] == 63


def test_bug065_keyerror_messages_are_printed_without_quotes(demo_dry, tmp_path, capsys, monkeypatch):
    from mdatk.config import Project
    from mdatk.pipeline import Pipeline
    from mdatk.progress import Reporter
    monkeypatch.delenv("MDATK_DEBUG", raising=False)
    code, err = _main(capsys, "analyze", "-p", demo_dry["project"], "-o", str(tmp_path / "o"), "--only", "nope",
                      "-q")
    assert code == 1 and err.startswith("error: unknown analysis 'nope' (known: ")
    from mdatk.progress import error_text
    assert error_text(KeyError("name")) == "name" and error_text(KeyError(3)) == "3"
    assert error_text(ValueError("a 'b'")) == "a 'b'"
    project = Project.load(demo_dry["project"])
    project.output.charts.series = ["bogus@Nope/x"]
    logs = []
    Pipeline(project, Reporter(log_cb=logs.append)).required_analyses(video=False, charts=True)
    warning = [line for line in logs if "skipping data source" in line]
    assert len(warning) == 1 and ": unknown analysis " in warning[0] and '"' not in warning[0]


def test_bug066_frames_and_list_index_errors_name_the_option(demo_dry, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MDATK_DEBUG", raising=False)
    out = str(tmp_path / "x.toml")
    for frames in ("x", "1.5", "0:y", "0:10:z"):
        code, err = _main(capsys, "init", out, "--frames", frames)
        assert code == 1
        assert err == f"error: --frames: {frames!r} is not a frame number or range (start:stop:step, e.g. 0:100:2)\n"
    code, err = _main(capsys, "init", out, "-p", demo_dry["project"], "--set", "groups.--1.name=x")
    assert code == 1 and err == "error: groups.--1.name: no list element '--1'\n"
    from mdatk.cli import apply_override
    from mdatk.config import Project
    project = Project.load(demo_dry["project"])
    last = project.groups[-1].name
    apply_override(project, "groups.-1.name=Renamed")  # negative indices still count from the end
    assert project.groups[-1].name == "Renamed" != last
    code, err = _main(capsys, "init", out, "--frames", "2:10:2")
    assert code == 0
    loaded = Project.load(out).inputs
    assert (loaded.start, loaded.stop, loaded.step) == (2, 10, 2)


def test_bug083_empty_frame_selection_says_end_not_none():
    from mdatk.io.loader import frame_indices
    with pytest.raises(ValueError) as info:
        frame_indices(6, 30, None, 1)
    assert str(info.value) == "frame selection start=30 stop=end step=1 selects no frames (trajectory has 6)"
    with pytest.raises(ValueError, match="start=30 stop=40 step=1"):
        frame_indices(6, 30, 40, 1)
    assert list(frame_indices(6, 4, None, 1)) == [4, 5]


# ---------------------------------------------------------------------------
# pipeline (084, 085)
# ---------------------------------------------------------------------------


def test_bug084_external_data_files_are_read_before_the_analyses(demo_dry, tmp_path, monkeypatch):
    from mdatk import pipeline
    from mdatk.config import ExternalSeries, Project
    project = Project.load(demo_dry["project"])
    project.output.out_dir = str(tmp_path / "out")
    project.output.cache_dir = str(tmp_path / "cache")
    calls = []
    real = pipeline.run_analyses
    monkeypatch.setattr(pipeline, "run_analyses", lambda *a, **k: calls.append(1) or real(*a, **k))
    missing = str(tmp_path / "missing.dat")
    bad = tmp_path / "bad.dat"
    bad.write_text("#Frame RMSD\n1 0.5\n2 oops\n")
    for path, expected in ((missing, "external data: "), (str(bad), f"external data: {bad}: non-numeric data line")):
        project.inputs.external = [ExternalSeries(path=path, x_unit="frame1")]
        with pytest.raises(pipeline.PipelineError) as info:
            pipeline.Pipeline(project).analyze(["rmsd"])
        assert str(info.value).startswith(expected) and path in str(info.value)
        assert calls == []  # found before the frames were read
    good = tmp_path / "good.dat"
    good.write_text("#Frame RMSD\n" + "".join(f"{k + 1} {0.1 * k}\n" for k in range(24)))
    project.inputs.external = [ExternalSeries(path=str(good), x_unit="frame1")]
    dataset = pipeline.Pipeline(project).analyze(["rmsd"])
    assert calls == [1]
    assert np.allclose(dataset.results["external"].arrays["s0_1"], 0.1 * np.arange(24))


def test_bug085_pdb_export_checks_every_frame_before_writing(tmp_path):
    from mdatk.io.pdb_export import export_frames
    atoms = [{"name": "C1", "resid": 1, "resname": "LIG", "element": "C"},
             {"name": "C2", "resid": 1, "resname": "LIG", "element": "C"}]
    coords = np.zeros((4, 2, 3))
    coords[:, 1, 0] = 1.5
    coords[2, 1, 0] = 123456.0
    coords[3, 0, 2] = -5000.0
    u = make_universe(atoms, coords)
    out = tmp_path / "pdb"
    with pytest.raises(ValueError) as info:
        export_frames(u, [0, 1, 2, 3], u.atoms, str(out), "x")
    assert str(info.value) == ("PDB files hold coordinates from -999.999 to 9999.999 Å; frames 2, 3 are outside "
                               "this range; no files were written.")
    assert not out.exists() or list(out.iterdir()) == []
    with pytest.raises(ValueError, match="frame 2 is outside this range; no files were written"):
        export_frames(u, [1, 2], u.atoms, str(out), "x")
    assert not out.exists() or list(out.iterdir()) == []
    paths = export_frames(u, [0, 1], u.atoms, str(out), "x")
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["x_00000.pdb", "x_00001.pdb"]
    assert "1.500" in open(paths[1]).read()


# ---------------------------------------------------------------------------
# addendum: command line (087, 092, 094, 095)
# ---------------------------------------------------------------------------


def _settings(path, *keys):
    """The lines of a project file that set ``keys`` (first occurrence of each)."""
    lines = open(path, encoding="utf-8").read().splitlines()
    return [next(line for line in lines if line.startswith(f"{key} = ")) for key in keys]


def test_bug087_density_fit_save_keeps_the_paths_as_written(tmp_path, capsys, monkeypatch):
    from mdatk.cli import main
    from mdatk.config import Project
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "demo", "--n-frames", "2", "--no-solvent", "--quiet"]) == 0
    project = "demo/demo.mdatk.toml"
    keys = ("topology", "trajectories", "out_dir", "cache_dir")
    before = _settings(project, *keys)
    assert before[:3] == ['topology = "demo.parm7"', 'trajectories = ["demo.nc"]', 'out_dir = "MDATK_output"']
    assert main(["density", "simulate", "-p", project, "--resolution", "8", "--output", "demo/sim.mrc", "-q"]) == 0
    capsys.readouterr()
    assert main(["density", "fit", "-p", project, "--map", "demo/sim.mrc", "--resolution", "8", "--step",
                 "45", "--save", "-q"]) == 0
    assert _settings(project, *keys) == before
    assert _settings(project, "map_path") == ['map_path = "sim.mrc"']
    saved = Project.load(project)  # the placement is stored, and the file loads as before
    assert len(saved.density.transform) == 12 and saved.density.resolution == 8
    assert saved.density.map_path == str(tmp_path / "demo" / "sim.mrc")
    assert saved.inputs.topology == str(tmp_path / "demo" / "demo.parm7")
    # a second fit without --map keeps the map as the file has it; other settings given on the command
    # line are saved as before
    assert main(["density", "fit", "-p", project, "--search", "manual", "--save", "-q", "--set",
                 "inputs.step=2"]) == 0
    assert _settings(project, *keys, "map_path") == before + ['map_path = "sim.mrc"']
    assert Project.load(project).inputs.step == 2
    # saved to another folder the paths must be absolute to stay valid
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert main(["density", "fit", "-p", project, "--search", "manual", "--save", str(other / "copy.toml"),
                 "-q"]) == 0
    assert Project.load(str(other / "copy.toml")).inputs.topology == str(tmp_path / "demo" / "demo.parm7")


@pytest.mark.parametrize("options, message", [
    (["--n-frames", "0"], "the demo needs at least one frame (--n-frames 0)"),
    (["--n-frames", "-2"], "the demo needs at least one frame (--n-frames -2)"),
    (["--n-frames", "3", "--dt", "-5"], "the time between frames must be a positive number of ps (--dt -5.0)"),
    (["--n-frames", "2", "--dt", "nan"], "the time between frames must be a positive number of ps (--dt nan)"),
    (["--n-frames", "2", "--dt", "0"], "the time between frames must be a positive number of ps (--dt 0.0)")])
def test_bug092_demo_refuses_impossible_arguments_before_writing(tmp_path, capsys, monkeypatch, options, message):
    monkeypatch.delenv("MDATK_DEBUG", raising=False)
    out = tmp_path / "d"
    code, err = _main(capsys, "demo", str(out), "--no-solvent", "--quiet", *options)
    assert code == 1 and err == f"error: {message}\n"
    assert not out.exists()
