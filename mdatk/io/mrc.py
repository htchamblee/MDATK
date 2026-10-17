"""Reader / writer for MRC (MRC2014) and CCP4 density maps (.mrc, .map, .ccp4, also gzipped).

The map is returned as a :class:`DensityMap` whose ``data`` array is indexed
``[z, y, x]`` (C order) in Cartesian axis order regardless of the file's
MAPC/MAPR/MAPS axis order, with ``origin`` (Å, coordinates of voxel (0,0,0))
and ``voxel`` (Å per voxel along x, y, z).  Only orthogonal cells are
supported (cryo-EM maps always are).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import Tuple

import numpy as np

HEADER_BYTES = 1024
MODES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}
IMOD_STAMP = 1146047817


class MapError(ValueError):
    pass


@dataclass
class DensityMap:
    data: np.ndarray  # [z, y, x] float32
    origin: np.ndarray  # (3,) Å, position of voxel index (0, 0, 0) as (x, y, z)
    voxel: np.ndarray  # (3,) Å per voxel along x, y, z
    path: str = ""

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        nz, ny, nx = self.data.shape
        return nx, ny, nz

    @property
    def extent(self) -> np.ndarray:
        """(xmax, ymax, zmax) of the grid in Å (centre of the last voxel)."""
        return self.origin + (np.array(self.shape_xyz) - 1) * self.voxel

    def stats(self) -> dict:
        d = self.data
        return {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean()),
                "rms": float(d.std())}

    def index_to_xyz(self, ijk: np.ndarray) -> np.ndarray:
        """Fractional (x, y, z) voxel indices -> Cartesian Å."""
        return self.origin + np.asarray(ijk, dtype=np.float64) * self.voxel

    def xyz_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """Cartesian Å -> fractional (x, y, z) voxel indices."""
        return (np.asarray(xyz, dtype=np.float64) - self.origin) / self.voxel


def _endian(raw: bytes) -> str:
    machst = raw[212:214]
    if machst[0] == 0x44:
        return "<"
    if machst[0] == 0x11:
        return ">"
    # no usable machine stamp: pick the byte order whose MODE and MAPC look valid
    for e in ("<", ">"):
        mode = int(np.frombuffer(raw, dtype=f"{e}i4", count=1, offset=12)[0])
        mapcrs = np.frombuffer(raw, dtype=f"{e}i4", count=3, offset=64)
        if mode in MODES and sorted(mapcrs.tolist()) in ([1, 2, 3], [0, 0, 0]):
            return e
    raise MapError("cannot determine the byte order of the map header")


def _open(path: str):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if gz else open(path, "rb")


def read_map(path: str) -> DensityMap:
    """Read an MRC / CCP4 map (optionally gzip-compressed) into Cartesian [z, y, x] order."""
    with _open(path) as fh:
        raw = fh.read(HEADER_BYTES)
        if len(raw) < HEADER_BYTES:
            raise MapError(f"{path}: file too short for an MRC header")
        e = _endian(raw)
        ints = np.frombuffer(raw, dtype=f"{e}i4", count=256)
        floats = np.frombuffer(raw, dtype=f"{e}f4", count=256)
        nc, nr, ns, mode = (int(v) for v in ints[0:4])
        nstart = np.array(ints[4:7], dtype=np.float64)  # along columns, rows, sections
        mxyz = np.array(ints[7:10], dtype=np.float64)
        cell = np.array(floats[10:13], dtype=np.float64)
        angles = np.array(floats[13:16], dtype=np.float64)
        mapcrs = [int(v) for v in ints[16:19]]
        nsymbt = int(ints[23])
        origin_field = np.array(floats[49:52], dtype=np.float64)
        if mode not in MODES:
            raise MapError(f"{path}: unsupported MRC mode {mode} (supported: 0, 1, 2, 6, 12)")
        if min(nc, nr, ns) <= 0:
            raise MapError(f"{path}: invalid dimensions {nc} x {nr} x {ns}")
        if sorted(mapcrs) != [1, 2, 3]:
            if mapcrs == [0, 0, 0]:  # some writers leave the field empty: default order
                mapcrs = [1, 2, 3]
            else:
                raise MapError(f"{path}: invalid axis order MAPC/MAPR/MAPS = {mapcrs}")
        if np.any(np.abs(angles - 90.0) > 0.01) and np.all(angles > 0):
            raise MapError(f"{path}: non-orthogonal cell angles {angles.tolist()} are not supported")
        dtype = np.dtype(MODES[mode]).newbyteorder(e)
        if mode == 0 and (raw[208:212] != b"MAP " or
                          (int(ints[38]) == IMOD_STAMP and not int(ints[39]) & 1)):
            dtype = np.dtype(np.uint8)  # IMOD (or pre-2000) maps store unsigned bytes
        count = nc * nr * ns
        if nsymbt > 0:
            fh.read(nsymbt)
        buf = fh.read(count * dtype.itemsize)
        if len(buf) < count * dtype.itemsize:
            raise MapError(f"{path}: file is shorter than its header says")
        data = np.frombuffer(buf, dtype=dtype, count=count)
    data = data.reshape(ns, nr, nc).astype(np.float32)  # [section, row, column]
    # file axis k (0 = column, 1 = row, 2 = section) holds Cartesian axis mapcrs[k] - 1
    axis_of = [m - 1 for m in mapcrs]  # cartesian index per file axis (c, r, s)
    # current array axes are (s, r, c) -> cartesian (axis_of[2], axis_of[1], axis_of[0]);
    # we want (z, y, x) = cartesian (2, 1, 0)
    current = [axis_of[2], axis_of[1], axis_of[0]]
    data = np.ascontiguousarray(np.transpose(data, [current.index(c) for c in (2, 1, 0)]))
    n_cart = np.zeros(3)
    start_cart = np.zeros(3)
    for k, cart in enumerate(axis_of):
        n_cart[cart] = (nc, nr, ns)[k]
        start_cart[cart] = nstart[k]
    mxyz = np.where(mxyz > 0, mxyz, n_cart)
    if np.any(cell <= 0):
        voxel = np.ones(3)
    else:
        voxel = cell / mxyz
    if np.any(origin_field != 0) and np.all(np.isfinite(origin_field)):
        origin = origin_field.copy()
    else:
        origin = start_cart * voxel
    return DensityMap(data, origin, voxel, path)


def write_map(path: str, dmap: DensityMap) -> None:
    """Write a float32 MRC2014 map (x fastest).  The origin goes into ORIGIN and, when it
    lies on whole voxels, also into NXSTART/NYSTART/NZSTART for CCP4-style readers."""
    data = np.ascontiguousarray(dmap.data, dtype="<f4")
    nz, ny, nx = data.shape
    words = np.zeros(256, dtype="<i4")
    fl = words.view("<f4")
    words[0:3] = (nx, ny, nz)
    words[3] = 2
    start = np.asarray(dmap.origin, dtype=np.float64) / np.asarray(dmap.voxel, dtype=np.float64)
    if np.allclose(start, np.round(start), atol=1e-4):
        words[4:7] = np.round(start).astype(np.int32)
    words[7:10] = (nx, ny, nz)
    fl[10:13] = np.array((nx, ny, nz)) * np.asarray(dmap.voxel)
    fl[13:16] = 90.0
    words[16:19] = (1, 2, 3)
    fl[19], fl[20], fl[21] = data.min(), data.max(), data.mean()
    words[22] = 1
    words[27] = 20140
    fl[49:52] = np.asarray(dmap.origin, dtype=np.float64)
    fl[54] = data.std()
    raw = bytearray(words.tobytes())
    raw[208:212] = b"MAP "
    raw[212:216] = bytes([0x44, 0x44, 0, 0])
    with open(path, "wb") as fh:
        fh.write(bytes(raw))
        fh.write(data.tobytes())
