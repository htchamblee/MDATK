"""MDFF with NAMD on an Amber topology - preparation and final model (standard library only).

NAMD reads the Amber parm7 natively (``amber on``), so every parameter the Amber build put
into it is used as is: the Lennard-Jones A/B table with all off-diagonal pairs (NBfix19, 0BPh,
microMg / nanoMg pair terms, ``addLJType`` splits), the CP and L-RNA atom types and the L-RNA
dihedrals.  Nothing is converted or re-parameterised.  This script prepares the files MDFF
needs around that topology:

``prep``
    * checks the topology for what NAMD cannot reproduce (12-6-4 C4 terms, per-dihedral 1-4
      scaling, polarisable topologies) and against the step's settings (periodic box,
      four-site water, time step vs. hydrogen masses);
    * finds the map's frame in the simulation: tleap moves the solute when it solvates it, so
      the coordinates are superposed onto the map-frame model (the PDB that was fitted into
      the map) and the potential is moved (or resampled) into the simulation frame;
    * converts the density into the MDFF potential exactly as VMD's ``mdff griddx`` does,
      V = (rho_max - max(rho, threshold)) / (rho_max - threshold), cropped around the solute
      and written as OpenDX;
    * writes the coupling PDB (B = standard mass of every coupled atom, O = 1), the positional
      restraint PDB, the periodic cell (.xsc) and the restraints:
        - chirality: one harmonic improper per ribose centre (C1', C2', C3', C4') with the
          target sign of the residue's own hand - D residues the D value, L residues its mirror
          image.  The hand comes from the residue name and must agree with the coordinates;
          a residue whose geometry contradicts its name - or any clearly defined centre of the
          other hand, which the restraint would invert - stops the script.  VMD's ``chirality``
          / ``ssrestraints`` / ``cispeptide`` tools assume D-nucleotides and are never used:
          on an L-RNA they would push every sugar towards the inverted centre, silently;
        - base pairs: flat-bottom extraBonds walls on the hydrogen-bond distances of the secondary
          structure (same pairs, atoms and window as MDATK's Amber helix restraints).
``final``
    writes the solute of NAMD's last binary coordinates as a PDB in the map's frame.
``check-only``
    the topology checks of ``prep`` without writing anything.
"""

from __future__ import annotations

import array
import gzip
import json
import math
import os
import re
import struct
import sys
import zlib

WATER = {"WAT", "HOH", "TIP3", "TIP4", "TIP5", "SOL", "SPC", "OPC", "T3P", "T4P", "T4E"}
STANDARD_MASS = {0: 0.0, 1: 1.008, 3: 6.94, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 11: 22.990,
                 12: 24.305, 15: 30.974, 16: 32.06, 17: 35.45, 19: 39.098, 20: 40.078, 30: 65.38,
                 35: 79.904, 37: 85.468, 53: 126.904, 55: 132.905}
ELEMENT_NUMBER = {"H": 1, "LI": 3, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12, "P": 15, "S": 16,
                  "CL": 17, "K": 19, "CA": 20, "ZN": 30, "BR": 35, "RB": 37, "I": 53, "CS": 55}
SYMBOL = {z: s for s, z in ELEMENT_NUMBER.items()}
SUGAR_CENTRES = (("C1'", ("O4'", "N", "C2'")), ("C2'", ("C1'", "O2'", "C3'")),
                 ("C3'", ("C2'", "O3'", "C4'")), ("C4'", ("C3'", "O4'", "C5'")))
# signs of the chiral volumes of an L-ribose (MDATK pdbtools); D has the opposite signs
L_CHIRAL_SIGNS = (("C4'", ("O4'", "C3'", "C5'"), 1.0), ("C3'", ("O3'", "C4'", "C2'"), -1.0),
                  ("C1'", ("O4'", "N", "C2'"), 1.0))
MIN_CHIRAL_VOLUME = 0.5
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
BACKBONE = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")
_FMT_RE = re.compile(r"\(\s*(\d+)\s*([aAiIeEfF])\s*(\d+)")
_NT_RE = re.compile(r"^([RL0D]?)([ACGTU])([53N]?)$")


class PrepError(Exception):
    """A problem that stops the preparation (printed as ERROR)."""


# ---------------------------------------------------------------------------------------------- input
def read_parm7(path, wanted):
    """{flag: values} for the ``wanted`` flags of an Amber parm7, and the set of all flags present."""
    sections, flags = {}, set()
    current, kind, width = None, "a", 80
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                name = line[5:].strip()
                flags.add(name)
                current = name if name in wanted else None
                if current:
                    sections[current] = []
                continue
            if line.startswith("%FORMAT"):
                m = _FMT_RE.search(line)
                if m is None:
                    raise PrepError(f"{path}: cannot read the format line {line.strip()!r}")
                kind, width = m.group(2).lower(), int(m.group(3))
                continue
            if current is None or line.startswith("%"):
                continue
            text = line.rstrip("\n")
            for k in range(0, len(text), width):
                chunk = text[k:k + width]
                if kind == "a":
                    sections[current].append(chunk.strip())
                elif chunk.strip():
                    value = chunk.strip().replace("D", "E").replace("d", "e")
                    sections[current].append(int(value) if kind == "i" else float(value))
    return sections, flags


class Topology:
    """The parts of a parm7 the preparation needs."""

    def __init__(self, path):
        want = {"POINTERS", "ATOM_NAME", "RESIDUE_LABEL", "RESIDUE_POINTER", "ATOMIC_NUMBER", "MASS",
                "AMBER_ATOM_TYPE", "SCEE_SCALE_FACTOR", "SCNB_SCALE_FACTOR", "BOX_DIMENSIONS"}
        s, self.flags = read_parm7(path, want)
        missing = [k for k in ("POINTERS", "ATOM_NAME", "RESIDUE_LABEL", "RESIDUE_POINTER", "MASS") if k not in s]
        if missing:
            raise PrepError(f"{path}: not an Amber topology (missing {', '.join(missing)})")
        ptr = s["POINTERS"]
        self.path = path
        if len(ptr) < 12:
            raise PrepError(f"{path}: truncated topology (POINTERS)")
        self.natom, self.nres = ptr[0], ptr[11]
        short = [k for k, n in (("ATOM_NAME", self.natom), ("MASS", self.natom), ("RESIDUE_LABEL", self.nres),
                                ("RESIDUE_POINTER", self.nres)) if len(s[k]) < n]
        if short:
            raise PrepError(f"{path}: truncated topology ({', '.join(short)} shorter than the atom / residue counts)")
        self.ifbox = ptr[27] if len(ptr) > 27 else 0
        self.numextra = ptr[30] if len(ptr) > 30 else 0
        self.names = s["ATOM_NAME"][:self.natom]
        self.labels = s["RESIDUE_LABEL"][:self.nres]
        self.pointers = s["RESIDUE_POINTER"][:self.nres]
        self.mass = s["MASS"][:self.natom]
        self.types = s.get("AMBER_ATOM_TYPE", [""] * self.natom)[:self.natom]
        if "ATOMIC_NUMBER" in s and len(s["ATOMIC_NUMBER"]) >= self.natom:
            self.number = s["ATOMIC_NUMBER"][:self.natom]
        else:  # old topologies: from the mass
            self.number = [_number_from_mass(m, n) for m, n in zip(self.mass, self.names)]
        self.scee = s.get("SCEE_SCALE_FACTOR", [])
        self.scnb = s.get("SCNB_SCALE_FACTOR", [])
        self.box_dimensions = s.get("BOX_DIMENSIONS", [])
        self.residue = [0] * self.natom
        bounds = self.pointers + [self.natom + 1]
        for r in range(self.nres):
            for i in range(bounds[r] - 1, bounds[r + 1] - 1):
                self.residue[i] = r
        sizes = [bounds[r + 1] - bounds[r] for r in range(self.nres)]
        self.solvent_residue = [self.labels[r].upper() in WATER or sizes[r] == 1 for r in range(self.nres)]

    def residue_atoms(self, r):
        end = self.pointers[r + 1] - 1 if r + 1 < self.nres else self.natom
        return range(self.pointers[r] - 1, end)

    def solvent(self, i):
        return self.solvent_residue[self.residue[i]]

    def heavy(self, i):
        return self.number[i] > 1


def _number_from_mass(mass, name):
    if mass < 0.5:
        return 0  # extra point
    best = min(STANDARD_MASS.items(), key=lambda kv: abs(kv[1] - mass) if kv[0] else 1e9)
    if abs(best[1] - mass) < 0.6:
        return best[0]
    return ELEMENT_NUMBER.get(name[:1].upper(), 6)  # repartitioned masses: the name decides


def is_netcdf(path):
    with open(path, "rb") as fh:
        head = fh.read(4)
    return head[:3] == b"CDF" or head == b"\x89HDF"


def read_rst7(path, natom):
    """Coordinates [(x, y, z)], box (a, b, c, alpha, beta, gamma) or None, has-velocities."""
    if is_netcdf(path):
        raise PrepError(f"{path} is a NetCDF restart; NAMD reads ASCII coordinates (convert it with "
                        f"cpptraj -p TOPOLOGY -y {os.path.basename(path)} -x start.rst7)")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 3:
        raise PrepError(f"{path}: not an Amber coordinate file")
    try:
        n = int(lines[1].split()[0])
    except (IndexError, ValueError):
        raise PrepError(f"{path}: line 2 must start with the atom count") from None
    if n != natom:
        raise PrepError(f"{path} has {n} atoms, the topology {natom}: they do not belong together")
    values = []
    for line in lines[2:]:
        text = line.rstrip()
        values += [float(text[k:k + 12]) for k in range(0, len(text), 12) if text[k:k + 12].strip()]
    count = len(values)
    if count not in (3 * n, 3 * n + 6, 6 * n, 6 * n + 6):
        raise PrepError(f"{path}: {count} numbers after the header do not fit {n} atoms")
    xyz = [tuple(values[3 * i:3 * i + 3]) for i in range(n)]
    box = tuple(values[-6:]) if count in (3 * n + 6, 6 * n + 6) else None
    return xyz, box, count >= 6 * n


def is_pdb(path):
    if path.lower().endswith((".pdb", ".ent")):
        return True
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    return any(ln.startswith(("ATOM  ", "HETATM", "CRYST1", "REMARK", "MODEL ")) for ln in head.splitlines())


def write_rst7(path, xyz, box=None, title="MDATK"):
    values = [c for p in xyz for c in p]
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(f"{title[:80]}\n{len(xyz):6d}\n")
        for k in range(0, len(values), 6):
            fh.write("".join(f"{v:12.7f}" for v in values[k:k + 6]) + "\n")
        if box:
            fh.write("".join(f"{v:12.7f}" for v in box) + "\n")


def start_coordinates(cfg, source):
    """The ASCII Amber coordinates NAMD starts from (``ambercoor``): an ASCII rst7 is checked and copied,
    a PDB in topology atom order (e.g. the dry PDB of the Amber build) converted."""
    top = Topology(cfg["topology"])
    if is_pdb(source):
        atoms = read_pdb_atoms(source)
        problem = pdb_order_problem(top, atoms, source)
        if problem:
            raise PrepError(problem)
        box = pdb_box(source) if top.ifbox else None  # a periodic system takes its box from CRYST1
        if top.ifbox and box is None:
            raise PrepError(f"{source} has no CRYST1 record, so the periodic box is unknown: connect the rst7 of "
                            "the Amber build or an MD restart")
        write_rst7(cfg["coordinates"], [a[3] for a in atoms], box, title=f"from {os.path.basename(source)} (MDATK)")
    else:
        xyz, box, _vel = read_rst7(source, top.natom)
        write_rst7(cfg["coordinates"], xyz, box, title=f"from {os.path.basename(source)} (MDATK)")
    print(f"start coordinates: {cfg['coordinates']} from {source}")
    return 0


def pdb_order_problem(top, atoms, source):
    """Why a PDB's atoms are not the topology's atoms in the topology's order (None when they are)."""
    if len(atoms) != top.natom:
        return (f"{source} has {len(atoms)} atoms, the topology {top.natom}: use the PDB written from this "
                "topology (Amber build: dry PDB)")
    for k, a in enumerate(atoms):
        if _canonical(a[2]) != _canonical(top.names[k]):
            return (f"atom {k + 1} is {a[2]} in {source} but {top.names[k]} in the topology: the PDB is not in "
                    "the topology's atom order")
    return None


def _atom_fields(line):
    """(chain, resSeq + iCode (5 columns), residue name, (x, y, z), element) of an ATOM / HETATM line, with
    ChimeraX's damage repaired as mdatk.pdbtools does: the residue name fused with the chain id and a colon
    ('GC:' = residue G, chain C) and the rest of the line shifted (element then unknown: '')."""
    colon = line.find(":", 18, 30)
    if colon > 0 and not line[colon - 1].isspace():
        words = line[16:colon - 1].split()
        if len(words) == 1 or (len(words) == 2 and words[0] == line[16]):
            rest = line[colon + 1:]
            m = re.match(r"\s*(-?\d+)([A-Za-z]?)\s", rest)
            coords = re.findall(r"-?\d+\.\d+", rest[m.end():]) if m else []
            if len(coords) < 3:
                raise PrepError(f"cannot read the damaged line {line.strip()!r}")
            number = f"{int(m.group(1)) % 10000:4d}{m.group(2) or ' '}"
            return line[colon - 1], number, words[-1], tuple(float(v) for v in coords[:3]), ""
    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return line[21], line[22:27], line[17:21].strip(), xyz, line[76:78].strip().upper()


def read_pdb_atoms(path):
    """[(residue key, residue name, atom name, (x, y, z), element)] of the first model.  The key is (ordinal,
    chain, resSeq + iCode): a new residue starts wherever chain, number or name change or a TER intervenes,
    so strands that share a chain id and numbering (as VMD writes segments) stay apart."""
    out = []
    ordinal, last = 0, None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("TER"):
                last = None
                continue
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            chain, number, resname, xyz, element = _atom_fields(line.rstrip("\n"))
            if (chain, number, resname) != last:
                ordinal += 1
                last = (chain, number, resname)
            out.append(((ordinal, chain, number), resname, line[12:16].strip(), xyz, element))
    if not out:
        raise PrepError(f"{path}: no atoms")
    return out


def pdb_box(path):
    """The periodic box (a, b, c, alpha, beta, gamma) of a PDB's CRYST1 record, None without one."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                try:
                    box = tuple(float(line[k:k + w]) for k, w in ((6, 9), (15, 9), (24, 9), (33, 7), (40, 7), (47, 7)))
                except ValueError:
                    return None
                return box if all(v > 1.0 for v in box) else None  # 1 1 1 90 90 90: no box
            if line.startswith(("ATOM  ", "HETATM")):
                return None
    return None


# ------------------------------------------------------------------------------------------------ maps
class Grid:
    """A scalar grid with x running fastest: value(i, j, k) = data[i + nx * (j + ny * k)]."""

    def __init__(self, nx, ny, nz, origin, delta, data):
        self.n = (nx, ny, nz)
        self.origin = tuple(float(v) for v in origin)
        self.delta = tuple(float(v) for v in delta)
        self.data = data

    def value(self, i, j, k):
        nx, ny, _ = self.n
        return self.data[i + nx * (j + ny * k)]

    def corner(self):
        return tuple(self.origin[a] + (self.n[a] - 1) * self.delta[a] for a in range(3))

    def interpolate(self, x, y, z, outside):
        """Trilinear value at Cartesian (x, y, z); ``outside`` beyond the grid."""
        f = [(p - o) / d for p, o, d in zip((x, y, z), self.origin, self.delta)]
        i0 = [int(math.floor(v)) for v in f]
        for a in range(3):
            if f[a] < 0 or f[a] > self.n[a] - 1:
                return outside
            i0[a] = min(i0[a], self.n[a] - 2) if self.n[a] > 1 else 0
        t = [f[a] - i0[a] for a in range(3)]
        total = 0.0
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    w = ((t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1]) * (t[2] if dk else 1 - t[2]))
                    if w:
                        total += w * self.value(min(i0[0] + di, self.n[0] - 1), min(i0[1] + dj, self.n[1] - 1),
                                                min(i0[2] + dk, self.n[2] - 1))
        return total


MRC_MODES = {0: "b", 1: "h", 2: "f", 6: "H"}


def read_mrc(path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if gz else open(path, "rb")) as fh:
        raw = fh.read(1024)
        if len(raw) < 1024:
            raise PrepError(f"{path}: file too short for an MRC header")
        e = "<" if raw[212] == 0x44 else ">" if raw[212] == 0x11 else None
        if e is None:  # no machine stamp: the byte order whose mode looks valid
            e = "<" if struct.unpack("<i", raw[12:16])[0] in (0, 1, 2, 6) else ">"
        ints = struct.unpack(e + "256i", raw)
        floats = struct.unpack(e + "256f", raw)
        nc, nr, ns, mode = ints[0:4]
        if mode not in MRC_MODES:
            raise PrepError(f"{path}: MRC mode {mode} is not supported (0, 1, 2, 6)")
        if min(nc, nr, ns) <= 0:
            raise PrepError(f"{path}: invalid dimensions {nc} x {nr} x {ns}")
        nstart = ints[4:7]
        mxyz = [m if m > 0 else 0 for m in ints[7:10]]
        cell = floats[10:13]
        angles = floats[13:16]
        mapcrs = list(ints[16:19])
        if mapcrs == [0, 0, 0]:
            mapcrs = [1, 2, 3]
        if sorted(mapcrs) != [1, 2, 3]:
            raise PrepError(f"{path}: invalid axis order {mapcrs}")
        if all(a > 0 for a in angles) and any(abs(a - 90.0) > 0.01 for a in angles):
            raise PrepError(f"{path}: non-orthogonal map cells are not supported")
        nsymbt = ints[23]
        if nsymbt > 0:
            fh.read(nsymbt)
        data = array.array(MRC_MODES[mode])
        count = nc * nr * ns
        buf = fh.read(count * data.itemsize)
        if len(buf) < count * data.itemsize:
            raise PrepError(f"{path}: file is shorter than its header says")
        data.frombytes(buf)
        if gz:
            fh.read()  # to the end of the stream, where gzip checks its CRC
    if (e == "<") != (sys.byteorder == "little") and data.itemsize > 1:
        data.byteswap()
    values = array.array("f", data) if mode != 2 else data
    axis_of = [m - 1 for m in mapcrs]  # Cartesian axis of the file's columns, rows, sections
    n_cart, start_cart = [0, 0, 0], [0, 0, 0]
    for k, cart in enumerate(axis_of):
        n_cart[cart] = (nc, nr, ns)[k]
        start_cart[cart] = nstart[k]
    if axis_of != [0, 1, 2]:  # reorder so that x runs fastest
        out = array.array("f", bytes(4 * count))
        idx = [0, 0, 0]
        p = 0
        for s_ in range(ns):
            idx[axis_of[2]] = s_
            for r_ in range(nr):
                idx[axis_of[1]] = r_
                for c_ in range(nc):
                    idx[axis_of[0]] = c_
                    out[idx[0] + n_cart[0] * (idx[1] + n_cart[1] * idx[2])] = values[p]
                    p += 1
        values = out
    mx = [mxyz[a] or n_cart[a] for a in range(3)]
    voxel = [cell[a] / mx[a] if cell[a] > 0 else 1.0 for a in range(3)]
    origin_field = floats[49:52]
    if any(v != 0 for v in origin_field) and all(math.isfinite(v) for v in origin_field):
        origin = origin_field
    else:
        origin = [start_cart[a] * voxel[a] for a in range(3)]
    return Grid(n_cart[0], n_cart[1], n_cart[2], origin, voxel, values)


def read_dx(path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rt", encoding="utf-8", errors="replace") if gz
          else open(path, "r", encoding="utf-8", errors="replace")) as fh:
        text = fh.read()
    m = re.search(r"gridpositions\s+counts\s+(\d+)\s+(\d+)\s+(\d+)", text)
    o = re.search(r"\norigin\s+(\S+)\s+(\S+)\s+(\S+)", text)
    deltas = re.findall(r"\ndelta\s+(\S+)\s+(\S+)\s+(\S+)", text)
    start = text.find("data follows")
    if not (m and o and len(deltas) == 3 and start >= 0):
        raise PrepError(f"{path}: not an OpenDX scalar grid")
    nx, ny, nz = (int(v) for v in m.groups())
    d = [[float(v) for v in row] for row in deltas]
    if any(abs(d[a][b]) > 1e-6 for a in range(3) for b in range(3) if a != b):
        raise PrepError(f"{path}: rotated DX grids are not supported")
    body = text[start + len("data follows"):]
    body = body[:body.find("attribute")] if "attribute" in body else body
    body = body.split("object")[0]
    vals = [float(v) for v in body.split()]
    if len(vals) < nx * ny * nz:
        raise PrepError(f"{path}: {len(vals)} values for a {nx} x {ny} x {nz} grid")
    data = array.array("f", bytes(4 * nx * ny * nz))
    p = 0
    for i in range(nx):  # DX: z runs fastest
        for j in range(ny):
            for k in range(nz):
                data[i + nx * (j + ny * k)] = vals[p]
                p += 1
    return Grid(nx, ny, nz, [float(v) for v in o.groups()], (d[0][0], d[1][1], d[2][2]), data)


def read_map(path):
    lower = path.lower()
    return read_dx(path) if lower.endswith(".dx") or lower.endswith(".dx.gz") else read_mrc(path)


def write_dx(path, grid, comment):
    nx, ny, nz = grid.n
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(f"# {comment}\n")
        fh.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        fh.write("origin {:.6f} {:.6f} {:.6f}\n".format(*grid.origin))
        fh.write(f"delta {grid.delta[0]:.6f} 0 0\ndelta 0 {grid.delta[1]:.6f} 0\ndelta 0 0 {grid.delta[2]:.6f}\n")
        fh.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        fh.write(f"object 3 class array type double rank 0 items {nx * ny * nz} data follows\n")
        row = []
        for i in range(nx):
            for j in range(ny):
                base = i + nx * j
                for k in range(nz):
                    row.append(f"{grid.data[base + nx * ny * k]:.6g}")
                    if len(row) == 3:
                        fh.write(" ".join(row) + "\n")
                        row = []
        if row:
            fh.write(" ".join(row) + "\n")
        fh.write('attribute "dep" string "positions"\n')  # the trailer VMD writes (mdff griddx files)
        fh.write('object "regular positions regular connections" class field\n')
        fh.write('component "positions" value 1\ncomponent "connections" value 2\ncomponent "data" value 3\n')


def mdff_potential(density, threshold):
    """The MDFF potential of ``mdff griddx``: 0 at the highest density, 1 at and below the threshold."""
    rho_max = max(density.data)
    if rho_max <= threshold:
        raise PrepError(f"the map has no density above the threshold {threshold:g} (maximum {rho_max:g})")
    scale = 1.0 / (rho_max - threshold)
    data = array.array("f", ((rho_max - (v if v > threshold else threshold)) * scale for v in density.data))
    return Grid(*density.n, density.origin, density.delta, data), rho_max


def crop(grid, lo, hi):
    """The part of ``grid`` covering the box lo..hi (Cartesian, in the grid's frame)."""
    a0, a1 = [], []
    for a in range(3):
        i0 = int(math.floor((lo[a] - grid.origin[a]) / grid.delta[a]))
        i1 = int(math.ceil((hi[a] - grid.origin[a]) / grid.delta[a]))
        i0, i1 = max(i0, 0), min(i1, grid.n[a] - 1)
        if i1 < i0:
            raise PrepError("the map does not overlap the solute at all: check the map and the frame")
        a0.append(i0)
        a1.append(i1)
    n = [a1[a] - a0[a] + 1 for a in range(3)]
    data = array.array("f", bytes(4 * n[0] * n[1] * n[2]))
    p = 0
    for k in range(a0[2], a1[2] + 1):
        for j in range(a0[1], a1[1] + 1):
            start = a0[0] + grid.n[0] * (j + grid.n[1] * k)
            data[p:p + n[0]] = grid.data[start:start + n[0]]
            p += n[0]
    origin = [grid.origin[a] + a0[a] * grid.delta[a] for a in range(3)]
    return Grid(n[0], n[1], n[2], origin, grid.delta, data)


# ------------------------------------------------------------------------------------- geometry helpers
def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dihedral(p0, p1, p2, p3):
    """Dihedral angle p0-p1-p2-p3 in degrees (IUPAC sign)."""
    b0, b1, b2 = sub(p0, p1), sub(p2, p1), sub(p3, p2)
    norm = math.sqrt(dot(b1, b1)) or 1.0
    b1n = (b1[0] / norm, b1[1] / norm, b1[2] / norm)
    v = sub(b0, tuple(dot(b0, b1n) * c for c in b1n))
    w = sub(b2, tuple(dot(b2, b1n) * c for c in b1n))
    return math.degrees(math.atan2(dot(cross(b1n, v), w), dot(v, w)))


def chiral_volume(centre, a, b, c):
    return dot(sub(a, centre), cross(sub(b, centre), sub(c, centre)))


def mat_vec(m, v):
    return tuple(m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] for r in range(3))


def transpose(m):
    return [[m[c][r] for c in range(3)] for r in range(3)]


def _jacobi4(a):
    """Eigen-decomposition of a symmetric 4x4 matrix: (eigenvalues, eigenvectors as columns)."""
    a = [row[:] for row in a]
    v = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    for _sweep in range(100):
        off = sum(a[r][c] ** 2 for r in range(4) for c in range(4) if r != c)
        if off < 1e-22:
            break
        for p in range(3):
            for q in range(p + 1, 4):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(4):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(4):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(4):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(4)], v


def superpose(mobile, target):
    """Rigid (R, t, rmsd) with target ~ R mobile + t (Horn's quaternion method)."""
    n = len(mobile)
    cm = tuple(sum(p[a] for p in mobile) / n for a in range(3))
    ct = tuple(sum(p[a] for p in target) / n for a in range(3))
    s = [[0.0] * 3 for _ in range(3)]
    for p, q in zip(mobile, target):
        x, y = sub(p, cm), sub(q, ct)
        for r in range(3):
            for c in range(3):
                s[r][c] += x[r] * y[c]
    (sxx, sxy, sxz), (syx, syy, syz), (szx, szy, szz) = s
    k = [[sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
         [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
         [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
         [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz]]
    vals, vecs = _jacobi4(k)
    best = max(range(4), key=lambda i: vals[i])
    q0, q1, q2, q3 = (vecs[r][best] for r in range(4))
    rot = [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
           [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
           [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
    rc = mat_vec(rot, cm)
    t = sub(ct, rc)
    msd = sum(dot(d, d) for d in (sub(sub(q, t), mat_vec(rot, p)) for p, q in zip(mobile, target))) / n
    return rot, t, math.sqrt(max(msd, 0.0))


def rotation_angle(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, (rot[0][0] + rot[1][1] + rot[2][2] - 1.0) / 2.0))))


def cell_vectors(box):
    """Amber box (a, b, c, alpha, beta, gamma) -> three cell vectors (a along x, b in the xy plane)."""
    a, b, c, al, be, ga = box
    ca, cb, cg = (math.cos(math.radians(v)) for v in (al, be, ga))
    sg = math.sin(math.radians(ga))
    v1 = (a, 0.0, 0.0)
    v2 = (b * cg, b * sg, 0.0)
    cy = c * (ca - cb * cg) / sg
    v3 = (c * cb, cy, math.sqrt(max(c * c - (c * cb) ** 2 - cy * cy, 0.0)))
    return v1, v2, v3


def grid_reach(grid, vectors):
    """NAMD's test of a grid against the periodic cell (gridforcechecksize, GridForceGrid.C fits_lattice):
    the grid is padded by one voxel on every non-continuous side and its corners are wrapped about the
    grid's centre; NAMD stops if any corner moves, i.e. if |b_i . (corner - centre)| >= 0.5 for a reciprocal
    vector b_i.  Returns the largest of these over the corners (it must stay below 0.5)."""
    v1, v2, v3 = vectors
    volume = dot(v1, cross(v2, v3))
    recip = [tuple(c / volume for c in cross(v2, v3)), tuple(c / volume for c in cross(v3, v1)),
             tuple(c / volume for c in cross(v1, v2))]
    half = [(grid.n[a] + 1) * grid.delta[a] / 2.0 for a in range(3)]  # padded corners about the unpadded centre
    return max(sum(abs(b[a]) * half[a] for a in range(3)) for b in recip)


# ------------------------------------------------------------------------------------- nucleotides
def residue_hand(label):
    """'D' / 'L' from an Amber residue label (LA, LA5, 0A, RA, A3 ...), None if not a nucleotide name."""
    m = _NT_RE.match(label.strip().upper())
    if not m:
        return None
    return "L" if m.group(1) in ("L", "0") else "D"


def residue_base(label):
    m = _NT_RE.match(label.strip().upper())
    return m.group(2) if m else None


def glycosidic(coords, base):
    """The base atom bonded to C1': N9 (purines), N1 (pyrimidines) or C5 (C-nucleosides such as
    pseudouridine), found by distance; from the base letter when none is within bonding distance."""
    c1 = coords.get("C1'")
    if c1 is not None:
        for name in ("N9", "N1", "C5"):
            if name in coords:
                d = sub(coords[name], c1)
                if dot(d, d) < 1.7 * 1.7:
                    return name
    return "N9" if base in ("A", "G") or (base not in ("C", "U", "T") and "N9" in coords) else "N1"


def geometry_hand(coords, base):
    """'D', 'L' or None (no usable or contradicting centres) from the ribose chiral volumes."""
    glyco = glycosidic(coords, base)
    votes = {"D": 0, "L": 0, "flat": 0}
    for centre, partners, l_sign in L_CHIRAL_SIGNS:
        names = [glyco if p == "N" else p for p in partners]
        if centre not in coords or any(n not in coords for n in names):
            continue
        v = chiral_volume(coords[centre], *(coords[n] for n in names))
        if abs(v) < MIN_CHIRAL_VOLUME:
            votes["flat"] += 1
        else:
            votes["L" if v * l_sign > 0 else "D"] += 1
    if votes["flat"] or (votes["L"] and votes["D"]) or not (votes["L"] or votes["D"]):
        return None
    return "L" if votes["L"] else "D"


def nucleotides(top, xyz):
    """[(residue index, label, hand from the name, hand from the coordinates, {name: atom index})]."""
    out = []
    for r in range(top.nres):
        if top.solvent_residue[r]:
            continue
        atoms = {top.names[i]: i for i in top.residue_atoms(r)}
        if not all(n in atoms for n in ("C1'", "C3'", "C4'", "O4'")):
            continue
        base = residue_base(top.labels[r])
        hand = residue_hand(top.labels[r])
        geo = geometry_hand({n: xyz[i] for n, i in atoms.items()}, base)
        out.append((r, top.labels[r], hand, geo, atoms))
    return out


def chirality_lines(top, xyz, k, targets):
    """extraBonds impropers (0-based atoms) restraining every ribose centre to its own hand."""
    lines, problems, notes = [], [], []
    for r, label, hand, geo, atoms in nucleotides(top, xyz):
        if hand is None and geo is None:
            problems.append(f"residue {r + 1} ({label}): neither its name nor its geometry tells D from L")
            continue
        if hand is not None and geo is not None and hand != geo:
            problems.append(f"residue {r + 1} ({label}) is named as {hand} but its sugar has {geo} geometry")
            continue
        use = hand or geo
        if hand is None:
            notes.append(f"residue {r + 1} ({label}): hand {geo} taken from the geometry (name not recognised)")
        elif geo is None:
            notes.append(f"residue {r + 1} ({label}): geometry gives no clear hand; {hand} from the name")
        glyco = glycosidic({n: xyz[i] for n, i in atoms.items()}, residue_base(label))
        sign = 1.0 if use == "D" else -1.0
        quads, other_hand = [], []
        for centre, partners in SUGAR_CENTRES:
            names = [centre] + [glyco if p == "N" else p for p in partners]
            if any(n not in atoms for n in names):
                continue
            idx = [atoms[n] for n in names]
            target = sign * targets[centre]
            # the chiral volume over the restraint's own atoms has the sign of the improper: a clearly
            # defined centre of the other sign would be driven through inversion by the restraint
            v = chiral_volume(*(xyz[i] for i in idx))
            if v * target < 0 and abs(v) >= MIN_CHIRAL_VOLUME:
                other_hand.append(centre)
            quads.append((idx, target))
        if other_hand:
            centres = f"{', '.join(other_hand)} {'has' if len(other_hand) == 1 else 'have'} the other hand"
            if hand:
                problems.append(f"residue {r + 1} ({label}) is named as {use} but its {centres}: the restraints "
                                "would invert it - correct the model or the residue name")
            else:  # the hand is the vote of C1' C3' C4', so only C2' can disagree and no name would help
                problems.append(f"residue {r + 1} ({label}): its sugar is {use} by C1' C3' C4' but its {centres}: "
                                "the restraints would invert it - correct the model")
            continue
        # comments on lines of their own: extraBonds reads "improper a b c d k n ref" when a line has
        # one field more, so trailing text after the values must never appear
        lines.append(f"# residue {r + 1} {label} ({use}): C1' C2' C3' C4'")
        for idx, target in quads:
            lines.append(f"improper {idx[0]} {idx[1]} {idx[2]} {idx[3]} {k:.3f} {target:.2f}")
    return lines, problems, notes


def _bonded(xyz, a, b, limit=2.0):
    d = sub(xyz[a], xyz[b])
    return dot(d, d) < limit * limit


def ss_dihedral_lines(top, xyz, residues, k):
    """extraBonds dihedrals (0-based atoms) holding alpha .. zeta and chi of the given residues (1-based
    topology numbers) at their values in ``xyz``, as the secondary-structure restraints of Trabuco et al.
    (2008).  The targets are measured on the model itself, so L residues keep their own (mirror-image)
    values.  alpha, epsilon and zeta reach into the neighbouring residue only when it is bonded
    (O3'-P below 2 A), never across a chain break."""
    lines, problems = [], []
    names = [{top.names[i]: i for i in top.residue_atoms(r)} for r in range(top.nres)]
    for resnum in sorted(set(residues)):
        r = resnum - 1
        if not 0 <= r < top.nres:
            problems.append(f"residue {resnum} of the secondary structure does not exist (the topology has "
                            f"{top.nres} residues): check the residue offset")
            continue
        at = names[r]
        if not all(n in at for n in ("C1'", "C4'", "O4'", "C3'")):
            problems.append(f"residue {resnum} ({top.labels[r]}) is not a nucleotide")
            continue
        prev = names[r - 1] if r > 0 else {}
        nxt = names[r + 1] if r + 1 < top.nres else {}
        has_prev = "O3'" in prev and "P" in at and _bonded(xyz, prev["O3'"], at["P"])
        has_next = "O3'" in at and "P" in nxt and _bonded(xyz, at["O3'"], nxt["P"])
        glyco = glycosidic({n: xyz[i] for n, i in at.items()}, residue_base(top.labels[r]))
        chi = ("O4'", "C1'", glyco, {"N9": "C4", "N1": "C2", "C5": "C4"}[glyco])
        torsions = []
        if has_prev:
            torsions.append(("alpha", [(prev, "O3'"), (at, "P"), (at, "O5'"), (at, "C5'")]))
        torsions += [("beta", [(at, "P"), (at, "O5'"), (at, "C5'"), (at, "C4'")]),
                     ("gamma", [(at, "O5'"), (at, "C5'"), (at, "C4'"), (at, "C3'")]),
                     ("delta", [(at, "C5'"), (at, "C4'"), (at, "C3'"), (at, "O3'")])]
        if has_next:
            torsions += [("epsilon", [(at, "C4'"), (at, "C3'"), (at, "O3'"), (nxt, "P")]),
                         ("zeta", [(at, "C3'"), (at, "O3'"), (nxt, "P"), (nxt, "O5'")])]
        torsions.append(("chi", [(at, n) for n in chi]))
        kept = []
        for name, quad in torsions:
            if any(n not in res for res, n in quad):
                continue  # e.g. a 5'-OH end without P
            idx = [res[n] for res, n in quad]
            value = dihedral(*(xyz[i] for i in idx))
            kept.append(f"dihedral {idx[0]} {idx[1]} {idx[2]} {idx[3]} {k:.3f} {value:.2f}")
        if kept:
            lines.append(f"# residue {resnum} {top.labels[r]}: alpha beta gamma delta epsilon zeta chi as built")
            lines += kept
    return lines, problems


def basepair_walls(top, pairs, bonds, offset, walls):
    """extraBonds flat-bottom walls (0-based atoms), one per hydrogen bond of every base pair:
    U = k (d - upper)^2 above the window, k (d - lower)^2 below it, 0 inside - the form of Amber's
    restraints (rk2 / rk3) without their linear continuation."""
    lower, upper, k = walls
    out, count = [], 0
    index = {}
    for i in range(top.natom):
        index[(top.residue[i], top.names[i])] = i
    for a, b, kind in pairs:
        ra, rb = a + offset, b + offset
        for res, letter in ((ra, kind[0]), (rb, kind[1])):
            if not 1 <= res <= top.nres:
                raise PrepError(f"residue {res} does not exist (the topology has {top.nres} residues): "
                                "check the residue offset")
            if residue_base(top.labels[res - 1]) != letter:
                raise PrepError(f"residue {res} is {top.labels[res - 1]} in the topology but the secondary "
                                f"structure expects {letter}: check the sequence and residue offset")
        out.append(f"# pair {ra} {top.labels[ra - 1]} - {rb} {top.labels[rb - 1]} ({kind})")
        for an, bn in bonds[kind]:
            ia, ib = index.get((ra - 1, an)), index.get((rb - 1, bn))
            if ia is None or ib is None:
                raise PrepError(f"atom {an} of residue {ra} or {bn} of residue {rb} is missing")
            count += 1
            out.append(f"wall {ia} {ib} {k:.3f} {lower:.3f} {upper:.3f}")
    return out, count


# ------------------------------------------------------------------------------------------- output
def _fixed(value, width, decimals):
    """``value`` in ``width`` columns, with fewer decimals when it would not fit (a wider field shifts
    every later column of a PDB line)."""
    for d in range(decimals, -1, -1):
        text = f"{value:{width}.{d}f}"
        if len(text) <= width:
            return text
    raise PrepError(f"{value:g} does not fit a {width}-column PDB field")


def pdb_line(i, top, xyz, occupancy, beta, chain="A", number=None):
    """ATOM record in the standard columns (1-based): name 13-16, residue 18-21, chain 22, number 23-26 (+ the
    insertion code 27), x y z 31-54, occupancy 55-60, beta 61-66, element 77-78 - where NAMD and VMD read
    them.  ``number``: residue number and insertion code (5 columns); the topology's residue number if None."""
    r = top.residue[i]
    name = top.names[i][:4]
    name = f" {name:<3s}" if len(name) < 4 else name
    x, y, z = xyz[i]
    number = number if number is not None else f"{(r + 1) % 10000:4d} "
    return (f"ATOM  {(i + 1) % 100000:5d} {name} {top.labels[r][:4]:<4s}{chain[:1] or ' '}{number[:5]:>5s}   "
            f"{_fixed(x, 8, 3)}{_fixed(y, 8, 3)}{_fixed(z, 8, 3)}{_fixed(occupancy, 6, 2)}{_fixed(beta, 6, 2)}"
            f"{'':10s}{SYMBOL.get(top.number[i], ''):>2s}\n")


def write_pdb(path, top, xyz, columns, remark):
    """A PDB of every atom; ``columns(i)`` -> (occupancy, beta)."""
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(f"REMARK   {remark}\n")
        for i in range(top.natom):
            occ, beta = columns(i)
            fh.write(pdb_line(i, top, xyz, occ, beta))
        fh.write("END\n")


def write_xsc(path, vectors, origin):
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("# NAMD extended system configuration (written by MDATK from the Amber box)\n")
        fh.write("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n")
        values = [v for vec in vectors for v in vec] + list(origin)
        fh.write("0 " + " ".join(f"{v:.6f}" for v in values) + "\n")


def read_namd_coor(path, natom):
    """NAMD binary coordinates: int32 atom count, then x, y, z float64 per atom (native byte order)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise PrepError(f"{path}: not NAMD binary coordinates for {natom} atoms")
    for e in ("<", ">"):
        n = struct.unpack(e + "i", raw[:4])[0]
        if n == natom and len(raw) >= 4 + 24 * n:
            v = struct.unpack(e + f"{3 * n}d", raw[4:4 + 24 * n])
            return [tuple(v[3 * i:3 * i + 3]) for i in range(n)]
    raise PrepError(f"{path}: not NAMD binary coordinates for {natom} atoms")


# -------------------------------------------------------------------------------------- the steps
def topology_problems(top, cfg):
    """Problems that stop the run (list of str) and notes."""
    problems, notes = [], []
    if "LENNARD_JONES_CCOEF" in top.flags:
        problems.append("the topology has 12-6-4 C4 terms (LENNARD_JONES_CCOEF, e.g. the Li & Merz 12-6-4 "
                        "Mg2+ model); NAMD cannot apply them - build with the 12-6, microMg or nanoMg model")
    if "IPOL" in top.flags and "POLARIZABILITY" in top.flags:
        problems.append("the topology is polarisable (IPOL / POLARIZABILITY); NAMD's Amber reader is not")
    for label, values, ref in (("SCEE", top.scee, 1.2), ("SCNB", top.scnb, 2.0)):
        odd = sorted({round(v, 4) for v in values if v and abs(v - ref) > 1e-4})
        if odd:
            problems.append(f"per-dihedral {label} values {odd[:4]} differ from {ref}; NAMD uses one global 1-4 "
                            "scaling (1-4scaling 0.833333, scnb 2.0)")
    if cfg["periodic"] and not top.ifbox:
        problems.append("the step is set to run solvated (periodic, PME) but the topology has no box; connect the "
                        "solvated topology or choose vacuum")
    if not cfg["periodic"] and top.ifbox:
        problems.append("the step is set to run in vacuum but the topology is periodic (solvated); connect the "
                        "dry topology of the Amber build or choose solvated")
    if top.numextra and not cfg["four_site"]:
        problems.append(f"the topology has {top.numextra} extra points (four-site water) but the step does not "
                        "expect four-site water")
    if cfg["four_site"] and not top.numextra:
        notes.append("four-site water was expected, but the topology has no extra points")
    # solute hydrogens only: HMR leaves water alone
    h_mass = [m for i, (m, z) in enumerate(zip(top.mass, top.number)) if z == 1 and not top.solvent(i)]
    heavy_h = sum(1 for m in h_mass if m > 2.0)
    if cfg["timestep"] > 2.0 + 1e-9 and h_mass and heavy_h < 0.5 * len(h_mass):
        problems.append(f"the time step is {cfg['timestep']:g} fs but the hydrogens have normal masses (no "
                        "hydrogen mass repartitioning); use 2 fs or repartition the masses")
    if heavy_h:
        notes.append(f"{heavy_h} of {len(h_mass)} hydrogens carry repartitioned masses (HMR)")
    light_o = sorted({round(top.mass[i], 3) for i in range(top.natom) if top.solvent(i) and top.number[i] == 8
                      and not 14.0 <= top.mass[i] <= 18.0})
    if light_o:
        problems.append(f"water oxygens with masses {light_o[:3]}: the water was mass-repartitioned; NAMD recognises "
                        "waters (for SETTLE) by an oxygen of 14-18 amu - repartition the solute only")
    return problems, notes


def model_residues(model_path):
    """[(residue key (chain, resSeq + iCode), {heavy atom name: position})] of the model's solute, in order:
    without water, ions and lone atoms - the residues the topology's solute residues are matched to."""
    residues, order = {}, []
    for key, resname, name, pos, element in read_pdb_atoms(model_path):
        if resname.upper() in WATER:
            continue
        if key not in residues:
            residues[key] = {}
            order.append(key)
        heavy = element != "H" and not (element == "" and name.lstrip("0123456789").startswith("H"))
        if heavy:
            residues[key][_canonical(name)] = pos
    return [(k, residues[k]) for k in order if len(residues[k]) > 1]


def frame_transform(top, xyz, model_path):
    """(R, t, rmsd, matched) with simulation = R map + t; identity without a model."""
    if not model_path:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], (0.0, 0.0, 0.0), None, 0
    model_res = [names for _key, names in model_residues(model_path)]
    sim_res = [r for r in range(top.nres) if not top.solvent_residue[r]]
    if len(model_res) != len(sim_res):
        raise PrepError(f"the map-frame model has {len(model_res)} residues (without water and ions), the "
                        f"topology {len(sim_res)}: they are not the same molecule")
    mobile, target = [], []
    for r, names in zip(sim_res, model_res):
        for i in top.residue_atoms(r):
            n = _canonical(top.names[i])
            if top.number[i] > 1 and n in names:
                mobile.append(names[n])
                target.append(xyz[i])
    heavy_total = sum(1 for i in range(top.natom) if top.number[i] > 1 and not top.solvent(i))
    if len(mobile) < 3 or len(mobile) < 0.5 * heavy_total:
        raise PrepError(f"only {len(mobile)} of {heavy_total} heavy atoms of the topology match the map-frame "
                        "model by name: it is not the model the topology was built from")
    rot, t, rmsd = superpose(mobile, target)
    return rot, t, rmsd, len(mobile)


def _canonical(name):
    n = name.strip().replace("*", "'")
    return {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3"}.get(n, n)


def potential_in_simulation_frame(density, cfg, rot, t, region):
    """The MDFF potential cropped to ``region`` (simulation frame), moved / resampled from the map frame;
    also the transform actually applied (simulation = R map + t), which final() undoes."""
    potential, rho_max = mdff_potential(density, cfg["threshold"])
    rot_t = transpose(rot)
    corners = [(x, y, z) for x in (region[0][0], region[1][0]) for y in (region[0][1], region[1][1])
               for z in (region[0][2], region[1][2])]
    in_map = [mat_vec(rot_t, sub(c, t)) for c in corners]
    lo = [min(c[a] for c in in_map) for a in range(3)]
    hi = [max(c[a] for c in in_map) for a in range(3)]
    part = crop(potential, lo, hi)
    angle = rotation_angle(rot)
    if angle < 0.05:  # a translation (tleap centring): move the grid by t' = t + (R - I) c, the shift that
        # is exact at the region's centre c (map frame) - t alone belongs to the rotation fitted with it
        mid = [(region[0][a] + region[1][a]) / 2.0 for a in range(3)]
        shift = sub(mid, mat_vec(rot_t, sub(mid, t)))
        part.origin = tuple(part.origin[a] + shift[a] for a in range(3))
        return part, rho_max, angle, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], shift
    # rotated frames: resample on an axis-parallel grid in the simulation frame, finer than the map so
    # that the second interpolation (NAMD's) adds little blur
    step = min(part.delta) / max(1, int(cfg.get("resample_divisor", 2)))
    n = [max(2, int(math.ceil((region[1][a] - region[0][a]) / step)) + 1) for a in range(3)]
    data = array.array("f", bytes(4 * n[0] * n[1] * n[2]))
    p = 0
    for k in range(n[2]):
        z = region[0][2] + k * step
        for j in range(n[1]):
            y = region[0][1] + j * step
            for i in range(n[0]):
                q = mat_vec(rot_t, sub((region[0][0] + i * step, y, z), t))
                data[p] = part.interpolate(q[0], q[1], q[2], 1.0)
                p += 1
    return Grid(n[0], n[1], n[2], region[0], (step, step, step), data), rho_max, angle, rot, t


def coupled(top, i, cfg):
    if top.solvent(i):
        return cfg["couple_ions"] and top.number[i] > 1 and top.labels[top.residue[i]].upper() not in WATER
    return top.number[i] > 1 or cfg["couple"] == "all"


def near(xyz, i, atoms, distance):
    """Whether atom ``i`` lies within ``distance`` of any of ``atoms``."""
    d2 = distance * distance
    x, y, z = xyz[i]
    return any((xyz[j][0] - x) ** 2 + (xyz[j][1] - y) ** 2 + (xyz[j][2] - z) ** 2 <= d2 for j in atoms)


def positional(top, i, selection):
    if selection == "none" or top.solvent(i) or top.number[i] <= 1:
        return False
    return selection == "heavy" or top.names[i] in BACKBONE


def prep(cfg, write=True):
    top = Topology(cfg["topology"])
    problems, notes = topology_problems(top, cfg)
    xyz, box, _vel = read_rst7(cfg["coordinates"], top.natom)
    if cfg["periodic"] and box is None:
        problems.append(f"{cfg['coordinates']} has no box line; the solvated run needs one")
    nts = nucleotides(top, xyz)
    if cfg["chirality"]["enabled"]:
        _lines, bad, more = chirality_lines(top, xyz, cfg["chirality"]["k"], cfg["chirality"]["targets"])
        problems += bad
        notes += more
    ssd = cfg.get("ssdihedrals") or {"enabled": False}
    if ssd["enabled"]:
        problems += ss_dihedral_lines(top, xyz, ssd["residues"], ssd["k"])[1]
    for n in notes:
        print(f"note: {n}")
    if problems:
        raise PrepError("\n  ".join(["the run would not reproduce the Amber force field or the model:"] + problems))
    nl = sum(1 for nt in nts if (nt[2] or nt[3]) == "L")
    print(f"topology: {top.natom} atoms, {top.nres} residues ({len(nts)} nucleotides, {nl} L), "
          f"{'periodic' if top.ifbox else 'no box'}{', ' + str(top.numextra) + ' extra points' if top.numextra else ''}")
    if not write:
        return 0
    rot, t, rmsd, matched = frame_transform(top, xyz, cfg["model"])
    if cfg["model"]:
        print(f"map frame: {matched} heavy atoms of {cfg['model']} superposed, RMSD {rmsd:.3f} A, rotation "
              f"{rotation_angle(rot):.3f} deg, shift {t[0]:.3f} {t[1]:.3f} {t[2]:.3f} A")
        if rmsd > cfg["max_frame_rmsd"]:
            raise PrepError(f"the coordinates differ from the map-frame model by {rmsd:.2f} A RMSD after "
                            "superposition: is this the model the topology was built from?")
    else:
        print("map frame: no map-frame model connected; the coordinates are taken to be in the map's frame")
    solute = [i for i in range(top.natom) if coupled(top, i, cfg) and not top.solvent(i)]
    if not solute:
        raise PrepError("no atoms to couple to the map")
    # the potential covers the RNA plus the margin; of the ions only those within the margin of an RNA atom
    # are coupled (the counterions of a solvated build are spread through the whole box)
    lo = [min(xyz[i][a] for i in solute) - cfg["pad"] for a in range(3)]
    hi = [max(xyz[i][a] for i in solute) + cfg["pad"] for a in range(3)]
    ions = [i for i in range(top.natom) if top.solvent(i) and coupled(top, i, cfg)
            and all(lo[a] <= xyz[i][a] <= hi[a] for a in range(3)) and near(xyz, i, solute, cfg["pad"])]
    sel = solute + ions
    density = read_map(cfg["map"])
    grid, rho_max, angle, rot, t = potential_in_simulation_frame(density, cfg, rot, t, (lo, hi))
    print(f"map {cfg['map']}: {density.n[0]} x {density.n[1]} x {density.n[2]} voxels, maximum {rho_max:g}; "
          f"potential {grid.n[0]} x {grid.n[1]} x {grid.n[2]} voxels"
          + (f" (resampled for a {angle:.2f} deg rotation)" if angle >= 0.05 else ""))
    # sanity check of the frame: a fitted model sits mostly on density above the map's mean + 1 sigma -
    # read from the density itself, in the map's frame (the potential is flat above a high threshold)
    count = len(density.data)
    mean = sum(density.data) / count
    sigma = math.sqrt(max(sum(v * v for v in density.data) / count - mean * mean, 0.0))
    level = mean + sigma
    rot_t = transpose(rot)
    corner = grid.corner()
    outside = sum(1 for i in sel if any(not (grid.origin[a] <= xyz[i][a] <= corner[a]) for a in range(3)))
    on_density = sum(1 for i in solute if density.interpolate(*mat_vec(rot_t, sub(xyz[i], t)), -math.inf) >= level)
    share = on_density / len(solute)
    print(f"coupled atoms: {len(solute)} RNA" + (f" + {len(ions)} ion{'s' * (len(ions) > 1)}" if ions else "")
          + f"; {share:.0%} of the RNA's on density above mean + 1 sigma ({level:g}), "
          f"{outside} outside the potential grid")
    if share < cfg["min_overlap"]:
        raise PrepError(f"only {share:.0%} of the coupled atoms lie on density above the map's mean + 1 sigma: "
                        "the model is not in the map's frame (connect the map-frame model) or not fitted into "
                        f"this map (the check needs {cfg['min_overlap']:.0%}; it can be lowered in the step)")
    vectors = None
    if cfg["periodic"]:
        vectors = cell_vectors(box)
        reach = grid_reach(grid, vectors)
        if reach >= 0.5 - 1e-6:
            size = [(grid.n[a] - 1) * grid.delta[a] for a in range(3)]
            shape = "truncated-octahedron" if any(abs(v - 90.0) > 0.1 for v in box[3:]) else "rectangular"
            raise PrepError(
                "the potential grid ({:.1f} x {:.1f} x {:.1f} A) does not fit the periodic cell as NAMD requires: "
                "padded by one voxel and wrapped about its centre it reaches {:.2f} of a cell vector, where NAMD "
                "allows less than 0.5 (it would stop with 'Periodic cell basis too small for Gridforce grid'). "
                .format(*size, reach)
                + ("A truncated-octahedron cell holds a much smaller grid than its width suggests: build a rectangular "
                   "box (Amber build: Box shape 'Rectangular'), " if shape != "rectangular" else "Build a larger box, ")
                + "or lower the margin around the RNA")
        print(f"the potential fits the periodic cell as NAMD checks it ({reach:.2f} of a cell vector, limit 0.5)")
    write_dx(cfg["potential"], grid, f"MDFF potential (mdff griddx formula, threshold {cfg['threshold']:g}) "
                                     f"from {os.path.basename(cfg['map'])}, in the simulation frame - MDATK")
    masses = {i: STANDARD_MASS.get(top.number[i], top.mass[i]) for i in sel}
    write_pdb(cfg["grid_pdb"], top, xyz, lambda i: (1.0, masses[i]) if i in masses else (0.0, 0.0),
              "MDFF coupling: occupancy 1 = coupled, beta = standard atomic mass")
    print(f"wrote {cfg['potential']} and {cfg['grid_pdb']}")
    if cfg["positional"]["selection"] != "none":
        k = cfg["positional"]["k"]
        chosen = [i for i in range(top.natom) if positional(top, i, cfg["positional"]["selection"])]
        write_pdb(cfg["restraint_pdb"], top, xyz, lambda i: (1.0, k) if positional(
            top, i, cfg["positional"]["selection"]) else (0.0, 0.0),
                  "positional restraints: beta = force constant (kcal/mol/A^2), reference = these coordinates")
        print(f"wrote {cfg['restraint_pdb']} ({len(chosen)} restrained atoms)")
    if cfg["chirality"]["enabled"]:
        lines, _bad, _more = chirality_lines(top, xyz, cfg["chirality"]["k"], cfg["chirality"]["targets"])
        with open(cfg["extrabonds"], "w", encoding="ascii", errors="replace") as fh:
            fh.write("# L-RNA-safe chirality restraints (MDATK): harmonic impropers at C1' C2' C3' C4' with the\n"
                     "# target of each residue's own hand (D value, or its mirror image for L residues).\n"
                     "# Atom numbers are 0-based. Never replace these with VMD's chirality / ssrestraints tools,\n"
                     "# which assume D-nucleotides.\n")
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {cfg['extrabonds']} ({sum(1 for ln in lines if ln.startswith('improper'))} chirality "
              "impropers)")
    if ssd["enabled"]:
        lines, _bad = ss_dihedral_lines(top, xyz, ssd["residues"], ssd["k"])
        with open(ssd["file"], "w", encoding="ascii", errors="replace") as fh:
            fh.write("# Secondary-structure restraints (MDATK, after Trabuco et al. 2008): backbone and glycosidic\n"
                     "# dihedrals of the base-paired residues held at their starting values (L residues keep their\n"
                     "# own mirror-image values). Atom numbers are 0-based; angles in degrees, k in kcal/mol/rad^2.\n")
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {ssd['file']} ({sum(1 for ln in lines if ln.startswith('dihedral'))} dihedrals of "
              f"{len(set(ssd['residues']))} paired residues)")
    if cfg["basepairs"]["enabled"]:
        bp = cfg["basepairs"]
        lines, count = basepair_walls(top, bp["pairs"], bp["bonds"], bp["offset"], bp["walls"])
        with open(bp["file"], "w", encoding="ascii", errors="replace") as fh:
            fh.write("# Base-pair hydrogen-bond restraints (MDATK): flat-bottom walls, U = k (d - upper)^2 above,\n"
                     "# k (d - lower)^2 below, 0 inside (extraBonds 'wall'). Atom numbers are 0-based.\n")
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {bp['file']} ({count} hydrogen-bond distances)")
    if vectors is not None:
        cx = tuple(sum(p[a] for p in xyz) / len(xyz) for a in range(3))
        write_xsc(cfg["xsc"], vectors, cx)
        print(f"wrote {cfg['xsc']} (cell vectors from the Amber box)")
    with open(cfg["prep_json"], "w", encoding="utf-8") as fh:
        json.dump({"rotation": rot, "translation": list(t), "frame_rmsd": rmsd, "coupled_atoms": len(sel),
                   "on_density": share, "rho_max": rho_max}, fh, indent=1)
    return 0


def solute_chains(top, xyz, model_path):
    """{solute residue: (chain id, resSeq + iCode, starts a segment)}.  A segment starts wherever the
    backbone is not bonded to the previous residue (O3'-P, or C-N of a protein).  Chain ids and numbers
    are those of the map-frame model when it has the same residues, else A, B ... per segment from 1."""
    sim_res = [r for r in range(top.nres) if not top.solvent_residue[r]]
    keys = [key for key, _names in model_residues(model_path)] if model_path else []
    out, chain, cid, number, prev, used = {}, -1, "", 0, None, {}
    for k, r in enumerate(sim_res):
        names = {top.names[i]: i for i in top.residue_atoms(r)}
        linked = prev is not None and any(a in prev and b in names and _bonded(xyz, prev[a], names[b])
                                          for a, b in (("O3'", "P"), ("C", "N")))
        if not linked:
            chain += 1
            cid = CHAIN_IDS[chain % len(CHAIN_IDS)]
            number = used.get(cid, 0)  # beyond 62 strands an id comes back: its numbers continue
        number += 1
        used[cid] = number
        if len(keys) == len(sim_res):
            out[r] = (keys[k][1], keys[k][2], not linked)
        else:
            out[r] = (cid, f"{number % 10000:4d} ", not linked)
        prev = names
    return out


def final(cfg):
    """The solute of NAMD's final coordinates as a PDB in the map's frame, with the chains (TER between
    segments) and residue numbers of the map-frame model - what refinement and renaming tools need."""
    top = Topology(cfg["topology"])
    xyz = read_namd_coor(cfg["final_coor"], top.natom)
    with open(cfg["prep_json"], "r", encoding="utf-8") as fh:
        info = json.load(fh)
    rot_t, t = transpose(info["rotation"]), info["translation"]
    back = [mat_vec(rot_t, sub(p, t)) for p in xyz]
    chains = solute_chains(top, back, cfg.get("model"))
    with open(cfg["final_pdb"], "w", encoding="ascii", errors="replace") as fh:
        fh.write("REMARK   MDFF result (NAMD), solute only, in the frame of the map - MDATK\n")
        last = None
        for r in range(top.nres):
            if r not in chains:
                continue
            chain, number, starts = chains[r]
            if last is not None and (starts or chain != last):
                fh.write("TER\n")
            for i in top.residue_atoms(r):
                fh.write(pdb_line(i, top, back, 1.0, 0.0, chain, number))
            last = chain
        fh.write("TER\nEND\n")
    print(f"wrote {cfg['final_pdb']}")
    return 0


def main(argv, cfg):
    for stream in (sys.stdout, sys.stderr):  # file names with non-ASCII characters on an ASCII terminal
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if argv[:1] == ["is-netcdf"] and len(argv) == 2:  # exit status 0: NetCDF, 1: not (ASCII)
        return 0 if is_netcdf(argv[1]) else 1
    if argv[:1] == ["coordinates"] and len(argv) == 2:
        try:
            return start_coordinates(cfg, argv[1])
        except (PrepError, OSError, ValueError, EOFError, zlib.error) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if not argv or argv[0] not in ("prep", "final", "check-only"):
        print("usage: python3 THIS_SCRIPT prep | final | check-only | is-netcdf FILE | coordinates FILE",
              file=sys.stderr)
        return 2
    try:
        if argv[0] == "final":
            return final(cfg)
        return prep(cfg, write=argv[0] == "prep")
    except (PrepError, OSError, ValueError, EOFError, zlib.error) as exc:  # unreadable numbers, encodings,
        print(f"ERROR: {exc}", file=sys.stderr)                           # damaged or truncated gzip files
        return 1


SETTINGS = {}
