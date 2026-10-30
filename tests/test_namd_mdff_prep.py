"""The NAMD MDFF preparation helper (mdatk/workflows/scripts/namd_mdff.py) on a D + L test system."""

import array
import json
import math
import os
import struct

import numpy as np
import pytest

import mdff_system as ms
from mdatk.io import mrc
from mdatk.workflows import rna
from mdatk.workflows.scripts import namd_mdff as nm

D_TARGETS = {"C1'": -35.01, "C2'": 40.03, "C3'": 38.30, "C4'": 33.31}


def config(tmp_path, **kw):
    cfg = {"topology": str(tmp_path / "sys.parm7"), "coordinates": str(tmp_path / "sys.rst7"), "model": "",
           "map": str(tmp_path / "map.mrc"), "periodic": False, "four_site": False, "timestep": 2.0,
           "threshold": 0.0, "pad": 6.0, "couple": "heavy", "couple_ions": False, "min_overlap": 0.3,
           "max_frame_rmsd": 2.0, "potential": str(tmp_path / "pot.dx"), "grid_pdb": str(tmp_path / "grid.pdb"),
           "restraint_pdb": str(tmp_path / "ref.pdb"), "extrabonds": str(tmp_path / "chir.txt"),
           "xsc": str(tmp_path / "cell.xsc"),
           "prep_json": str(tmp_path / "prep.json"), "final_pdb": str(tmp_path / "final.pdb"),
           "final_coor": str(tmp_path / "final.coor"),
           "chirality": {"enabled": True, "k": 200.0, "targets": D_TARGETS},
           "basepairs": {"enabled": False, "pairs": [], "bonds": {}, "offset": 0, "walls": (2.7, 3.2, 10.0),
                         "file": str(tmp_path / "bp.txt")},
           "positional": {"selection": "none", "k": 1.0},
           "ssdihedrals": {"enabled": False, "k": 200.0, "residues": [], "file": str(tmp_path / "ssd.txt")}}
    cfg.update(kw)
    return cfg


@pytest.fixture
def system(tmp_path):
    atoms = ms.atoms_of(ms.mixed_pdb())
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in atoms])
    ms.write_map(str(tmp_path / "map.mrc"), atoms)
    (tmp_path / "model.pdb").write_text(ms.model_pdb(atoms))
    return atoms


def test_topology_reader_and_l_residues(tmp_path, system):
    top = nm.Topology(str(tmp_path / "sys.parm7"))
    assert top.natom == len(system) and top.nres == 16 and not top.ifbox
    nts = nm.nucleotides(top, [a[5] for a in system])
    hands = [(lab, h, g) for _r, lab, h, g, _a in nts]
    assert len(hands) == 16
    assert all(h == g == "D" for lab, h, g in hands[:8]) and all(h == g == "L" for lab, h, g in hands[8:])
    assert {"LD", "LE", "LH"} <= set(top.types)


def test_chirality_targets_follow_each_residue_hand(tmp_path, system):
    top = nm.Topology(str(tmp_path / "sys.parm7"))
    xyz = [a[5] for a in system]
    lines, problems, _notes = nm.chirality_lines(top, xyz, 200.0, D_TARGETS)
    assert not problems and len(lines) == 16 * 5  # a comment line and four impropers per nucleotide
    assert all(ln.startswith("# ") or len(ln.split()) == 7 for ln in lines)  # nothing after the values
    for line in (ln for ln in lines if ln.startswith("improper")):
        parts = line.split()
        i, j, k, l = (int(v) for v in parts[1:5])
        target = float(parts[6])
        centre = top.names[i]  # 0-based atom numbers: the first atom is the centre
        hand = "L" if top.labels[top.residue[i]].startswith("L") else "D"
        # D residues get the D value, L residues its mirror image ...
        assert target == pytest.approx(D_TARGETS[centre] * (1 if hand == "D" else -1)), line
        # ... which is what their own coordinates show: nothing pushes a sugar to the other hand
        measured = nm.dihedral(xyz[i], xyz[j], xyz[k], xyz[l])
        assert math.copysign(1, measured) == math.copysign(1, target) and abs(measured - target) < 8.0, line


def test_name_geometry_mismatch_stops(tmp_path, system):
    atoms = [a if a[0] != "C" or a[1] != 2 else a[:2] + ("G",) + a[3:] for a in system]  # an L residue named D
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms)
    with pytest.raises(nm.PrepError, match=r"residue 10 \(G\) is named as D but its sugar has L geometry"):
        nm.prep(config(tmp_path))


def test_prep_without_model_writes_every_file(tmp_path, system):
    cfg = config(tmp_path, positional={"selection": "backbone", "k": 2.5})
    assert nm.prep(cfg) == 0
    pot = nm.read_dx(cfg["potential"])
    assert 0.0 <= min(pot.data) < 0.2 and max(pot.data) == pytest.approx(1.0)
    grid = (tmp_path / "grid.pdb").read_text().splitlines()
    atoms = [ln for ln in grid if ln.startswith("ATOM")]
    assert len(atoms) == len(system)
    coupled = [ln for ln in atoms if float(ln[54:60]) == 1.0]
    assert len(coupled) == sum(1 for a in system if a[4] != "H")
    assert {round(float(ln[60:66]), 2) for ln in coupled} == {12.01, 14.01, 16.0, 30.97}
    ref = [ln for ln in (tmp_path / "ref.pdb").read_text().splitlines() if ln.startswith("ATOM")]
    assert {ln[12:16].strip() for ln in ref if float(ln[60:66]) == 2.5} == set(nm.BACKBONE)
    assert (tmp_path / "chir.txt").read_text().count("\nimproper ") == 64
    info = json.loads((tmp_path / "prep.json").read_text())
    assert info["on_density"] > 0.8 and info["frame_rmsd"] is None


def test_potential_is_the_mdff_griddx_formula(tmp_path):
    g = nm.Grid(3, 1, 1, (0, 0, 0), (1, 1, 1), array.array("f", [-1.0, 1.0, 3.0]))
    pot, rho_max = nm.mdff_potential(g, 0.0)
    assert rho_max == 3.0 and list(pot.data) == pytest.approx([1.0, 2 / 3, 0.0])
    pot, _ = nm.mdff_potential(g, 1.5)
    assert list(pot.data) == pytest.approx([1.0, 1.0, 0.0])
    with pytest.raises(nm.PrepError, match="no density above"):
        nm.mdff_potential(g, 5.0)


def test_mrc_reader_matches_mdatk_for_permuted_axes(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.normal(size=(5, 4, 3)).astype(np.float32)  # [z, y, x]
    dmap = mrc.DensityMap(data, np.array([1.5, -2.0, 3.0]), np.array([0.5, 0.7, 0.9]))
    path = str(tmp_path / "m.mrc")
    mrc.write_map(path, dmap)
    g = nm.read_mrc(path)
    assert g.n == (3, 4, 5) and g.origin == pytest.approx((1.5, -2.0, 3.0)) and g.delta == pytest.approx((0.5, 0.7, 0.9))
    assert g.value(2, 1, 3) == pytest.approx(float(data[3, 1, 2]))
    # the same map stored with sections along x, rows along z, columns along y (MAPC/R/S = 2, 3, 1)
    raw = bytearray(open(path, "rb").read())
    ints = list(struct.unpack("<256i", raw[:1024]))
    ints[0:3] = [4, 5, 3]
    ints[16:19] = [2, 3, 1]
    raw[:1024] = struct.pack("<256i", *ints)[:1024]
    raw[1024:] = np.ascontiguousarray(np.transpose(data, (2, 0, 1))).astype("<f4").tobytes()  # [x][z][y]
    raw[196:208] = struct.pack("<3f", 1.5, -2.0, 3.0)
    open(path, "wb").write(bytes(raw))
    ref = mrc.read_map(path)
    g = nm.read_mrc(path)
    assert g.n == ref.shape_xyz
    for (i, j, k) in ((0, 0, 0), (2, 3, 4), (1, 2, 3)):
        assert g.value(i, j, k) == pytest.approx(float(ref.data[k, j, i]))


def test_dx_roundtrip(tmp_path):
    g = nm.Grid(2, 3, 4, (1, 2, 3), (0.5, 0.5, 0.5), array.array("f", [float(v) for v in range(24)]))
    path = str(tmp_path / "g.dx")
    nm.write_dx(path, g, "test")
    h = nm.read_dx(path)
    assert h.n == g.n and h.origin == pytest.approx(g.origin) and list(h.data) == list(g.data)
    text = open(path).read()
    assert "object 1 class gridpositions counts 2 3 4" in text
    # OpenDX: z runs fastest
    assert text.split("data follows\n")[1].split()[:4] == ["0", "6", "12", "18"]


def _mean_potential(cfg, atoms):
    pot = nm.read_dx(cfg["potential"])
    heavy = [a[5] for a in atoms if a[4] != "H" and a[2] not in ("WAT", "MG")]
    return sum(pot.interpolate(*p, 1.0) for p in heavy) / len(heavy)


def test_frame_recovered_after_tleap_style_move(tmp_path, system):
    # reference: the potential the atoms see where they were fitted
    base = config(tmp_path)
    assert nm.prep(base) == 0
    expected = _mean_potential(base, system)
    assert expected < 0.75
    # tleap moves (here also rotates) the solute when it solvates it: the coordinates are no longer in
    # the map's frame; the map-frame model (the PDB fitted into the map) gives the transformation back
    rot = ms.rotation((0.3, 1.0, 0.2), 37.0)
    shift = (20.0, -15.0, 32.0)
    solvated = ms.with_solvent(ms.moved(system, rot, shift), start=(shift[0] - 8, shift[1] - 8, shift[2] + 25))
    box = (90.0, 90.0, 90.0, 90.0, 90.0, 90.0)
    ms.write_topology(str(tmp_path / "sys.parm7"), solvated, box=box)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in solvated], box=box)
    cfg = config(tmp_path, periodic=True)
    with pytest.raises(nm.PrepError, match="not in the map's frame"):
        nm.prep(cfg)  # without the model the atoms miss the density
    cfg["model"] = str(tmp_path / "model.pdb")
    assert nm.prep(cfg) == 0
    info = json.loads((tmp_path / "prep.json").read_text())
    assert info["frame_rmsd"] < 0.01 and info["on_density"] > 0.8
    assert np.allclose(info["rotation"], rot, atol=1e-4) and np.allclose(info["translation"], shift, atol=1e-3)
    xyz = [a[5] for a in solvated]
    # the potential moved with the molecule: the moved atoms see what they saw in the map's frame
    assert _mean_potential(cfg, solvated) == pytest.approx(expected, abs=0.03)
    xsc = (tmp_path / "cell.xsc").read_text().splitlines()
    assert xsc[1].split()[1:] == ["step", "a_x"][1:] or xsc[1].startswith("#$LABELS step a_x")
    assert [float(v) for v in xsc[2].split()[1:10]] == pytest.approx([90, 0, 0, 0, 90, 0, 0, 0, 90])
    # final: NAMD's binary coordinates back into the map's frame (solute only)
    n = len(xyz)
    with open(cfg["final_coor"], "wb") as fh:
        fh.write(struct.pack("<i", n) + struct.pack(f"<{3 * n}d", *[c for p in xyz for c in p]))
    assert nm.final(cfg) == 0
    out = [ln for ln in (tmp_path / "final.pdb").read_text().splitlines() if ln.startswith("ATOM")]
    assert len(out) == len(system)
    back = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in out])
    assert np.abs(back - np.array([a[5] for a in system])).max() < 0.002


def test_pure_translation_moves_the_grid_without_resampling(tmp_path, system):
    shift = (31.0, 27.5, 40.25)
    moved = ms.moved(system, np.eye(3), shift)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in moved])
    cfg = config(tmp_path, model=str(tmp_path / "model.pdb"))
    assert nm.prep(cfg) == 0
    pot = nm.read_dx(cfg["potential"])
    dmap = mrc.read_map(str(tmp_path / "map.mrc"))
    # origin on the original voxel lattice, shifted: no interpolation happened
    frac = (np.array(pot.origin) - shift - dmap.origin) / dmap.voxel
    assert np.allclose(frac, np.round(frac), atol=1e-3)


def test_topology_problems(tmp_path, system):
    atoms = system
    cfg = config(tmp_path)
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms, extra_flags=[("LENNARD_JONES_CCOEF", "e", [0.0])])
    with pytest.raises(nm.PrepError, match="12-6-4 C4 terms"):
        nm.prep(cfg, write=False)
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms, extra_flags=[("SCEE_SCALE_FACTOR", "e", [1.2, 1.0])])
    with pytest.raises(nm.PrepError, match="per-dihedral SCEE"):
        nm.prep(cfg, write=False)
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms, extra_flags=[("SCEE_SCALE_FACTOR", "e", [1.2, 1.2])])
    assert nm.prep(cfg, write=False) == 0
    with pytest.raises(nm.PrepError, match="no box"):
        nm.prep(config(tmp_path, periodic=True), write=False)
    with pytest.raises(nm.PrepError, match="normal masses"):
        nm.prep(config(tmp_path, timestep=4.0), write=False)
    ms.write_topology(str(tmp_path / "sys.parm7"), ms.with_solvent(atoms), hmr=True)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in ms.with_solvent(atoms)])
    assert nm.prep(config(tmp_path, timestep=4.0), write=False) == 0  # water H stay light


def test_mass_repartitioned_water_is_refused(tmp_path, system):
    # NAMD finds waters (for SETTLE) by an oxygen of 14-18 amu: repartitioned water would not be rigid
    solvated = ms.with_solvent(system)
    ms.write_topology(str(tmp_path / "sys.parm7"), solvated)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in solvated])
    assert nm.prep(config(tmp_path), write=False) == 0
    text = (tmp_path / "sys.parm7").read_text()
    head, mass = text.split("%FLAG MASS\n")
    fmt, rest = mass.split("\n", 1)
    body, tail = rest.split("%FLAG", 1)
    values = body.split()
    oxygens = [k for k, a in enumerate(solvated) if a[2] == "WAT" and a[4] == "O"]
    for k in oxygens:
        values[k] = "1.19750000E+01"
    lines = ["".join(f"{float(v):16.8E}" for v in values[i:i + 5]) for i in range(0, len(values), 5)]
    (tmp_path / "sys.parm7").write_text(head + "%FLAG MASS\n" + fmt + "\n" + "\n".join(lines) + "\n%FLAG" + tail)
    with pytest.raises(nm.PrepError, match="water was mass-repartitioned"):
        nm.prep(config(tmp_path), write=False)


def test_netcdf_restart_is_refused_with_a_hint(tmp_path, system):
    (tmp_path / "sys.rst7").write_bytes(b"CDF\x02" + b"\0" * 64)
    with pytest.raises(nm.PrepError, match="cpptraj -p TOPOLOGY -y sys.rst7"):
        nm.prep(config(tmp_path), write=False)


def test_basepair_walls(tmp_path, system):
    top = nm.Topology(str(tmp_path / "sys.parm7"))
    model = rna.build_model("GCGC GCGC", "(((( ))))")
    pairs = [p for p in rna.classify_pairs(model, False)[0]]
    lines, count = nm.basepair_walls(top, pairs, rna.HBOND_ATOMS, 0, (2.7, 3.2, 10.0))
    assert count == 12  # four G-C pairs x three hydrogen bonds
    walls = [ln.split() for ln in lines if ln.startswith("wall")]
    assert len(walls) == 12 and all(len(w) == 6 for w in walls)  # nothing after the values
    a, b = int(walls[0][1]), int(walls[0][2])  # 0-based atom numbers
    assert top.names[a] == "O6" and top.residue[a] == 0 and top.names[b] == "N4" and top.residue[b] == 7
    assert walls[0][3:] == ["10.000", "2.700", "3.200"]  # k lower upper
    with pytest.raises(nm.PrepError, match="residue 9 is LG5"):
        nm.basepair_walls(top, [(1, 9, "GC")], rna.HBOND_ATOMS, 0, (2.7, 3.2, 10.0))


def test_superpose_and_cell_helpers():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20, 3)) * 5
    rot = ms.rotation((1, 2, 3), 63.0)
    moved = pts @ rot.T + np.array([1.0, -2.0, 3.5])
    r, t, rmsd = nm.superpose([tuple(p) for p in pts], [tuple(p) for p in moved])
    assert rmsd < 1e-6 and np.allclose(r, rot, atol=1e-8) and np.allclose(t, [1.0, -2.0, 3.5], atol=1e-8)
    oct_box = (60.0, 60.0, 60.0, 109.4712206, 109.4712206, 109.4712206)
    v = nm.cell_vectors(oct_box)
    assert [math.sqrt(nm.dot(x, x)) for x in v] == pytest.approx([60.0] * 3)
    widths = nm.cell_widths(v)
    assert widths == pytest.approx([60 * math.sqrt(2 / 3) * 1.0] * 3, rel=0.02) or min(widths) > 45
    assert nm.residue_hand("LG5") == "L" and nm.residue_hand("0C") == "L" and nm.residue_hand("RA3") == "D"
    assert nm.residue_hand("G") == "D" and nm.residue_hand("WAT") is None


def test_secondary_structure_dihedrals_keep_each_hand(tmp_path, system):
    top = nm.Topology(str(tmp_path / "sys.parm7"))
    xyz = [a[5] for a in system]
    lines, problems = nm.ss_dihedral_lines(top, xyz, range(1, 17), 200.0)
    assert not problems
    torsions = [ln for ln in lines if ln.startswith("dihedral")]
    # gamma delta chi for all 16; alpha and beta need the 5' phosphate (the four chain starts are 5'-OH);
    # epsilon and zeta need a bonded next residue (not the four 3' ends)
    assert len(torsions) == 16 * 3 + 12 * 2 + 12 * 2
    values = {}
    for ln in torsions:
        a, b, c, d = (int(v) for v in ln.split()[1:5])
        target = float(ln.split()[6])
        assert target == pytest.approx(nm.dihedral(xyz[a], xyz[b], xyz[c], xyz[d]), abs=0.01)
        assert len(ln.split()) == 7  # nothing after the values (extraBonds would read a periodic form)
        values[(top.residue[b], top.names[a], top.names[b], top.names[c], top.names[d])] = target
        # never across a chain break: all four atoms in this residue or bonded neighbours
        assert max(top.residue[i] for i in (a, b, c, d)) - min(top.residue[i] for i in (a, b, c, d)) <= 1
    # the L duplex (residues 9-16) is the mirror image of the D duplex (1-8): every dihedral flips sign
    for (r, *names), v in values.items():
        if r < 8:
            assert values[(r + 8, *names)] == pytest.approx(-v, abs=0.02)
    # chain starts have no alpha into the previous chain's last residue
    starts = [0, 4, 8, 12]
    assert not any(k[0] in starts and k[1] == "O3'" for k in values)
    lines, problems = nm.ss_dihedral_lines(top, xyz, [99], 200.0)
    assert problems and "does not exist" in problems[0]


def test_prep_writes_secondary_structure_dihedrals(tmp_path, system):
    cfg = config(tmp_path, ssdihedrals={"enabled": True, "k": 200.0, "residues": [1, 2, 9, 10],
                                        "file": str(tmp_path / "ssd.txt")})
    assert nm.prep(cfg) == 0
    text = (tmp_path / "ssd.txt").read_text()
    assert text.count("\ndihedral ") > 4 * 4 and "Trabuco" in text


def _standard_columns(line):
    """An ATOM record read at the standard PDB columns, as NAMD and VMD read it."""
    return {"name": line[12:16].strip(), "alt": line[16], "resname": line[17:21].strip(), "chain": line[21],
            "resseq": int(line[22:26]), "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            "occupancy": float(line[54:60]), "beta": float(line[60:66]), "element": line[76:78].strip()}


def test_pdb_records_use_the_standard_columns(tmp_path, system):
    # far from the origin (x < -100) and with a force constant >= 100 every field starts with a digit or a
    # sign: a field written one column off is then read with a wrong sign or value
    far = ms.moved(system, np.eye(3), (-150.0, 0.0, 0.0))
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in far])
    ms.write_map(str(tmp_path / "map.mrc"), far)
    cfg = config(tmp_path, positional={"selection": "heavy", "k": 250.0})
    assert nm.prep(cfg) == 0
    top = nm.Topology(cfg["topology"])
    ref = [_standard_columns(ln) for ln in (tmp_path / "ref.pdb").read_text().splitlines() if ln.startswith("ATOM")]
    assert len(ref) == top.natom
    for i, rec in enumerate(ref):
        assert rec["xyz"] == pytest.approx(far[i][5], abs=0.001)
        assert rec["alt"] == " " and rec["chain"] == "A" and rec["resname"] == top.labels[top.residue[i]]
        assert rec["name"] == top.names[i] and rec["resseq"] == top.residue[i] + 1
        assert rec["beta"] == (250.0 if top.number[i] > 1 else 0.0)
        assert rec["element"] == {1: "H", 6: "C", 7: "N", 8: "O", 15: "P"}[top.number[i]]
    # a force constant of 1000 still fits the six beta columns
    line = nm.pdb_line(0, top, [p[5] for p in far], 1.0, 1000.0)
    assert _standard_columns(line)["beta"] == 1000.0 and len(line) == 79
    # the fitted model (the step's output) keeps residue names, chain and coordinates for any reader
    n = top.natom
    with open(cfg["final_coor"], "wb") as fh:
        fh.write(struct.pack("<i", n) + struct.pack(f"<{3 * n}d", *[c for a in far for c in a[5]]))
    assert nm.final(cfg) == 0
    from mdatk import pdbtools
    residues = list(pdbtools.read_pdb((tmp_path / "final.pdb").read_text()).residues())
    assert [res.name for res in residues] == top.labels and {res.chain for res in residues} == {"A"}
    assert residues[0].atoms[0].xyz[0] == pytest.approx(far[0][5][0], abs=0.001)
    assert residues[0].atoms[0].xyz[0] < -100


def test_frame_check_reads_the_density_whatever_the_threshold(tmp_path, system):
    # with a threshold above the map's mean + 1 sigma the potential is 1 wherever the density is not
    # above the threshold: the check must still see atoms that miss the density
    d = nm.read_map(str(tmp_path / "map.mrc"))
    mean = sum(d.data) / len(d.data)
    sigma = math.sqrt(sum(v * v for v in d.data) / len(d.data) - mean * mean)
    high = 1.5 * (mean + sigma)
    assert nm.prep(config(tmp_path, threshold=high, min_overlap=0.5)) == 0  # in the map's frame
    ms.write_rst7(str(tmp_path / "sys.rst7"), [tuple(np.array(a[5]) + (9.0, 0.0, 0.0)) for a in system])
    for threshold in (0.0, high):
        with pytest.raises(nm.PrepError, match="not in the map's frame"):
            nm.prep(config(tmp_path, threshold=threshold, min_overlap=0.5))


def test_only_ions_near_the_rna_are_coupled(tmp_path, system):
    # counterions of a solvated build are spread through the box: they must neither size the potential
    # nor be coupled far from the RNA
    near, far = (0.0, 0.0, 14.0), (40.0, 40.0, 40.0)
    ions = [("M", 1, "MG", "MG", "MG", near), ("M", 2, "MG", "MG", "MG", far),
            ("N", 1, "Na+", "Na+", "NA", (-40.0, -38.0, 35.0))]
    atoms = list(system) + ions
    box = (90.0, 90.0, 90.0, 90.0, 90.0, 90.0)
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms, box=box)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in atoms], box=box)
    cfg = config(tmp_path, periodic=True, couple_ions=True, pad=6.0)
    assert nm.prep(cfg) == 0
    records = [_standard_columns(ln) for ln in (tmp_path / "grid.pdb").read_text().splitlines()
               if ln.startswith("ATOM")]
    occ = [r["occupancy"] for r in records[-3:]]
    assert occ == [1.0, 0.0, 0.0]
    assert records[-3]["beta"] == pytest.approx(24.305, abs=0.01)
    pot = nm.read_dx(cfg["potential"])
    heavy = np.array([a[5] for a in system if a[4] != "H"])
    size = np.array(pot.delta) * (np.array(pot.n) - 1)
    assert np.all(size <= heavy.max(axis=0) - heavy.min(axis=0) + 2 * 6.0 + 2 * 1.5)


def test_translation_branch_applies_and_records_the_shift_it_used(tmp_path):
    # a model far from the map's origin, coordinates rotated by less than the resampling limit: the grid
    # is only moved, by the shift that is exact at the molecule, and final() undoes exactly that shift
    atoms = ms.moved(ms.atoms_of(ms.mixed_pdb()), np.eye(3), (330.0, 150.0, 60.0))
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms)
    ms.write_map(str(tmp_path / "map.mrc"), atoms)
    (tmp_path / "model.pdb").write_text(ms.model_pdb(atoms))
    centre = np.mean([a[5] for a in atoms], axis=0)
    rot = ms.rotation((0.2, 0.5, 1.0), 0.045)
    sim = [tuple(rot @ (np.array(a[5]) - centre) + centre + (-300.0, -140.0, -50.0)) for a in atoms]
    ms.write_rst7(str(tmp_path / "sys.rst7"), sim)
    cfg = config(tmp_path, model=str(tmp_path / "model.pdb"))
    assert nm.prep(cfg) == 0
    info = json.loads((tmp_path / "prep.json").read_text())
    assert info["rotation"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # the potential sits where the fitted transform puts the density
    pot = nm.read_dx(cfg["potential"])
    reference, _ = nm.mdff_potential(nm.read_map(str(tmp_path / "map.mrc")), 0.0)
    heavy = [i for i, a in enumerate(atoms) if a[4] != "H"]
    assert max(abs(pot.interpolate(*sim[i], 1.0) - reference.interpolate(*atoms[i][5], 1.0)) for i in heavy) < 0.02
    n = len(sim)
    with open(cfg["final_coor"], "wb") as fh:
        fh.write(struct.pack("<i", n) + struct.pack(f"<{3 * n}d", *[c for p in sim for c in p]))
    assert nm.final(cfg) == 0
    out = [ln for ln in (tmp_path / "final.pdb").read_text().splitlines() if ln.startswith("ATOM")]
    back = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in out])
    assert np.abs(back - np.array([a[5] for a in atoms])).max() < 0.05  # 0.25 A with the fitted t


def test_gzipped_dx_and_non_ascii_file_names(tmp_path, system):
    import gzip
    import shutil
    g = nm.Grid(2, 3, 4, (1, 2, 3), (0.5, 0.5, 0.5), array.array("f", [float(v) for v in range(24)]))
    nm.write_dx(str(tmp_path / "g.dx"), g, "test")
    with open(tmp_path / "g.dx", "rb") as src, gzip.open(tmp_path / "g.dx.gz", "wb") as dst:
        dst.write(src.read())
    h = nm.read_map(str(tmp_path / "g.dx.gz"))
    assert h.n == g.n and list(h.data) == list(g.data)
    # a map or coordinate file named with non-ASCII characters: the names end up in ASCII file headers
    shutil.copy(tmp_path / "map.mrc", tmp_path / "carte_résolue_3.2Å.mrc")
    shutil.copy(tmp_path / "sys.rst7", tmp_path / "coordonnées.rst7")
    cfg = config(tmp_path, map=str(tmp_path / "carte_résolue_3.2Å.mrc"), coordinates=str(tmp_path / "start.rst7"))
    assert nm.main(["coordinates", str(tmp_path / "coordonnées.rst7")], cfg) == 0
    assert nm.main(["prep"], cfg) == 0
    assert "carte_r?solue_3.2?.mrc" in (tmp_path / "pot.dx").read_text().splitlines()[0]
    assert nm.read_rst7(cfg["coordinates"], len(system))[0] == pytest.approx([a[5] for a in system], abs=1e-6)


def _flatten(atoms, chain, resseq, centre, partners, factor):
    """``centre`` moved along the normal of its partners' plane: factor 1 puts it into the plane, 2
    mirrors it through the plane (inverts that one centre)."""
    pos = {a[3]: np.array(a[5]) for a in atoms if a[0] == chain and a[1] == resseq}
    p = [pos[n] for n in partners]
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    normal /= np.linalg.norm(normal)
    new = pos[centre] - factor * np.dot(pos[centre] - p[0], normal) * normal
    return [a[:5] + (tuple(new),) if a[0] == chain and a[1] == resseq and a[3] == centre else a for a in atoms]


@pytest.mark.parametrize("change", ["flat C1'", "mirrored C1'"])
def test_ambiguous_sugar_does_not_hide_centres_of_the_other_hand(tmp_path, system, change):
    # an L residue named as D (residue 10, chain C 2) whose geometry gives no clear hand: one flat or
    # one inverted centre leaves no unanimous vote, but its other centres are clearly L - D targets
    # would invert them, so the run must stop
    factor = 1.0 if change.startswith("flat") else 2.0
    atoms = _flatten(system, "C", 2, "C1'", ("O4'", "N1", "C2'"), factor)
    atoms = [a[:2] + ("C",) + a[3:] if a[0] == "C" and a[1] == 2 else a for a in atoms]
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in atoms])
    top = nm.Topology(str(tmp_path / "sys.parm7"))
    lines, problems, _notes = nm.chirality_lines(top, [a[5] for a in atoms], 200.0, D_TARGETS)
    assert len(problems) == 1
    assert problems[0].startswith("residue 10 (C) is named as D but its ") and "C3', C4'" in problems[0]
    assert "# residue 10 " not in "\n".join(lines)
    with pytest.raises(nm.PrepError, match="residue 10 .C. is named as D"):
        nm.prep(config(tmp_path), write=False)
    # the same residue named L: the flat / single inverted centre is the only odd one; no clearly
    # defined centre is pushed through inversion, so nothing stops
    atoms = [a[:2] + ("LC",) + a[3:] if a[0] == "C" and a[1] == 2 else a for a in atoms]
    ms.write_topology(str(tmp_path / "sys.parm7"), atoms)
    top = nm.Topology(str(tmp_path / "sys.parm7"))
    _lines, problems, _notes = nm.chirality_lines(top, [a[5] for a in atoms], 200.0, D_TARGETS)
    assert problems == ([] if change.startswith("flat") else
                        ["residue 10 (LC) is named as L but its C1' has the other hand: the restraints would "
                         "invert it - correct the model or the residue name"])


def test_inverted_c2_of_a_d_residue_stops(tmp_path, system):
    # the C2' centre does not vote on the hand, but its restraint would still invert a clear C2'
    atoms = _flatten(system, "A", 2, "C2'", ("C1'", "O2'", "C3'"), 2.0)
    ms.write_rst7(str(tmp_path / "sys.rst7"), [a[5] for a in atoms])
    with pytest.raises(nm.PrepError, match=r"residue 2 \(C\) is named as D but its C2' has the other hand"):
        nm.prep(config(tmp_path), write=False)
