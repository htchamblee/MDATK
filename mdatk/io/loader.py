"""Loading topologies and trajectories through MDAnalysis.

Frames are always streamed: MDAnalysis keeps a single frame in memory and the
analysis runner visits each selected frame exactly once.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import struct
import sys
import traceback
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

_TOPOLOGY_FORMATS = {".parm7": "PARM7", ".prmtop": "PARM7", ".top": "PARM7"}
_TRAJECTORY_FORMATS = {".nc": "NCDF", ".ncdf": "NCDF", ".netcdf": "NCDF",
                       ".mdcrd": "TRJ", ".crd": "TRJ", ".trj": "TRJ",
                       ".rst7": "INPCRD", ".inpcrd": "INPCRD", ".ncrst": "NCRST"}
# the formats in words, for messages (other formats are named by their extension)
_FORMAT_WORDS = {"PARM7": "Amber parm7", "NCDF": "Amber NetCDF", "TRJ": "Amber mdcrd", "INPCRD": "Amber restart",
                 "NCRST": "Amber NetCDF restart", "ENT": "PDB", "GRO": "GROMACS GRO"}


class _FrameTimes:
    """Trajectory transformation of the time-step override: the time of frame k is k * dt.  MDAnalysis' own
    ``dt`` only fills in for trajectories without time stamps; a NetCDF file keeps the times stored in it (and
    a second file starts again at its own first time)."""

    def __init__(self, dt_ps: float):
        self.dt_ps = dt_ps

    def __call__(self, ts):
        ts.time = ts.frame * self.dt_ps
        return ts


def is_parm7(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _TOPOLOGY_FORMATS


class ReaderError(ValueError):
    """A topology or trajectory file that cannot be read.  The message starts with the file as it was
    given and says what is wrong with it; the exception MDAnalysis raised, if any, is its cause."""


# ---------------------------------------------------------------------------
# Checks before MDAnalysis reads a file
# ---------------------------------------------------------------------------
# MDAnalysis reads every format.  A few inputs it reads wrongly without a word, and others stop it with a
# Python error that names neither the file nor the problem; these are looked for here first.


def _extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _format_words(path: str) -> str:
    ext = _extension(path)
    fmt = _TOPOLOGY_FORMATS.get(ext) or _TRAJECTORY_FORMATS.get(ext) or ext[1:].upper()
    return _FORMAT_WORDS.get(fmt, fmt or "structure")


def _compressed(path: str) -> bool:
    """gzip or bzip2 data (MDAnalysis unpacks it whatever the file is called); not checked beforehand."""
    with open(path, "rb") as fh:
        return fh.read(2) in (b"\x1f\x8b", b"BZ")


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _check_parm7(path: str) -> None:
    with open(path, "rb") as fh:
        first = fh.readline()
    if not first.startswith(b"%VE"):  # %VERSION, as MDAnalysis tests it
        hint = " (GROMACS .top files are not supported)" if _extension(path) == ".top" else ""
        raise ReaderError(f"{path}: not an Amber parm7 topology{hint}")


def _check_gro(path: str) -> None:
    """The coordinates of a GRO file as MDAnalysis reads them: it takes the width of the coordinate
    columns from the first atom line on the assumption that x is 8 or more characters wide with its
    decimal point in column 25 (a wider x makes it read single characters), reads a box line it cannot
    split by its unsigned numbers only, and reads the first frame only."""
    with open(path, "rb") as fh:
        fh.readline()  # title
        count = fh.readline().decode("latin-1").strip()
        if not count.isdigit():
            raise ReaderError(f"{path} line 2: expected the number of atoms, found {count!r}")
        n = int(count)
        if n < 1:
            return
        first = fh.readline().decode("latin-1").rstrip("\r\n")
        if not first.strip():
            raise ReaderError(f"{path}: the header says {n} atoms but the file ends after line 2")
        dots = [k for k, c in enumerate(first) if c == "." and k >= 20][:3]
        if len(dots) < 3 or dots[0] != 24 or dots[1] - dots[0] != dots[2] - dots[1]:
            raise ReaderError(f"{path} line 3: cannot determine coordinate field width (the decimal points "
                              "of x, y and z are not one column width apart from column 25; a coordinate too "
                              "large for its column?)")
        lines = 1 + sum(1 for _ in itertools.islice(fh, n - 1))
        box = fh.readline().decode("latin-1")
        if lines < n:
            raise ReaderError(f"{path}: the header says {n} atoms but the file ends after {lines} atom lines")
        if not box.strip():
            raise ReaderError(f"{path}: missing box line after atom {n} (or the number of atoms in line 2 "
                              "is too large)")
        fields = box.split()
        if not (len(fields) in (3, 9) and all(_number(f) is not None for f in fields)) and not re.fullmatch(
                r"\s*(?:\d+\.\d{5}\s*){3}(?:(?:\d+\.\d{5}\s*){6})?", box):  # values run together: unsigned only
            raise ReaderError(f"{path} line {n + 3}: GRO box line has neither 3 nor 9 values: {box.strip()!r}")
        more, tail = 0, b"\n"
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            more, tail = more + chunk.count(b"\n"), chunk[-1:]
        frames = 1 + (more + (tail != b"\n")) // (n + 3)
    if frames > 1:
        log.warning("%s: only the first of %d frames is read (a GRO file is read as one structure)",
                    path, frames)


def _hybrid36_field(field: str) -> bool:
    from MDAnalysis.topology.PDBParser import hy36decode
    try:
        int(field)
        return False
    except ValueError:
        pass
    try:
        hy36decode(5, field)
        return True
    except (ValueError, KeyError):
        return False


def _scan_pdb(path: str) -> bool:
    """Raise for a PDB topology without atoms; True when CONECT records hold hybrid-36 serials."""
    atoms = ended = hybrid = False
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"CONECT"):
                if not hybrid:
                    record = line.strip().decode("latin-1")
                    hybrid = any(_hybrid36_field(record[k:k + 5]) for k in range(6, len(record), 5))
            elif not atoms and not ended:  # as MDAnalysis' parser: up to the first END / ENDMDL
                record = line.strip()
                atoms = record.startswith((b"ATOM", b"HETATM"))
                ended = record.startswith(b"END")
    if not atoms:
        raise ReaderError(f"{path}: no ATOM or HETATM records")
    return hybrid


def _hybrid36_pdb_parser():
    """MDAnalysis' PDB parser, reading the atom serials of CONECT records as it reads those of ATOM
    records: decimal or hybrid-36 (files of 100 000 atoms and more).  MDAnalysis itself reads CONECT
    serials as decimal numbers only and fails on such a file."""
    from MDAnalysis.core.topologyattrs import Bonds
    from MDAnalysis.lib import util
    from MDAnalysis.topology import PDBParser as pdb

    def serial(field: str) -> int:
        try:
            return int(field)
        except ValueError:
            return pdb.hy36decode(5, field)

    class Parser(pdb.PDBParser):  # without a 'format' of its own it is not registered for any file type
        def _parsebonds(self, serials):
            if self._wrapped_serials:
                return super()._parsebonds(serials)
            index = {s: i for i, s in enumerate(serials)}
            bonds = set()
            with util.openany(self.filename) as fh:
                for line in fh:
                    if line[:6] != "CONECT":
                        continue
                    record = line.strip()
                    if (len(record) - 6) % 5:
                        raise RuntimeError(f"CONECT record is not in columns of 5: {record}")
                    ids = [serial(record[k:k + 5]) for k in range(6, len(record), 5)]
                    bonds.update(tuple(sorted((index[ids[0]], index[a]))) for a in ids[1:]
                                 if ids[0] in index and a in index)  # serials of TER records: no atoms
            return Bonds(tuple(sorted(bonds)))

    return Parser


def _check_dcd(path: str) -> None:
    """MDAnalysis divides by the frame interval of the header (NSAVC) for the time of every frame."""
    with open(path, "rb") as fh:
        head = fh.read(20)
    for order in "<>":
        if len(head) == 20 and head[4:8] == b"CORD" and struct.unpack(order + "i", head[:4])[0] == 84:
            if struct.unpack(order + "i", head[16:20])[0] == 0:
                raise ReaderError(f"{path}: DCD header has frame interval (NSAVC) 0")


def _zero_cell(path: str) -> bool:
    """An Amber NetCDF trajectory whose first frame has the cell lengths 0 0 0 (written by some programs
    for runs without a periodic cell)."""
    try:
        from scipy.io import netcdf_file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fh = netcdf_file(path, "r", mmap=True)
            try:
                lengths = fh.variables.get("cell_lengths")
                zero = lengths is not None and lengths.shape[0] > 0 and not np.array(lengths[0]).any()
                del lengths
            finally:
                fh.close()
        return bool(zero)
    except Exception:  # noqa: BLE001 - not NetCDF 3, damaged ...: MDAnalysis reports it
        return False


# ---------------------------------------------------------------------------
# What MDAnalysis says while reading
# ---------------------------------------------------------------------------

_WRAPPED = re.compile(r"Failed to (?:construct topology from file|load from the topology file) .*? "
                      r"with parser <class '[^']*'>\.\s*Error:\s*", re.S)
_ADVICE = re.compile(r"\s*If needed,? \w+ can be\s+guessed using\s+universe\.guess_TopologyAttrs\(.*\)\.?", re.S)
_SILENT = ("No coordinate reader found",  # a topology without coordinates: an error of its own below
           "Unknown element found")  # blank element fields: see _guess_missing_masses
_REWORDED = {
    "Reader has no dt information": "no time information in the file: frames are taken as 1 ps apart "
                                    "(set the time step, inputs.dt_ps)",
    "Trajectory contains a single atom": "an mdcrd trajectory of one atom is read without box lines; if "
                                         "the file has box lines, each is read as a frame",
}


def _chain(exc: Optional[BaseException]):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _release(exc: Optional[BaseException]) -> None:
    """Drop, quietly, the readers MDAnalysis left half-built in the frames of ``exc``'s traceback.  Their
    ``__del__`` fails on the attributes they never got, which Python prints as 'Exception ignored in ...'
    with a traceback at whatever moment they are collected - after the error message of the command line."""
    hook, sys.unraisablehook = sys.unraisablehook, lambda _unraisable: None
    try:
        for e in _chain(exc):
            traceback.clear_frames(e.__traceback__)
    finally:
        sys.unraisablehook = hook


def _raised_in(exc: BaseException, package: str) -> bool:
    """True when ``exc`` (or the error it replaced) was raised below MDAnalysis' ``package``."""
    part = f"/MDAnalysis/{package}/"
    return any(part in frame.f_code.co_filename.replace(os.sep, "/")
               for e in _chain(exc) for frame, _ in traceback.walk_tb(e.__traceback__))


def _undecodable_line(path: str) -> Tuple[int, int]:
    """(line number, byte) of the first line that is not UTF-8 text, or (0, 0)."""
    try:
        with open(path, "rb") as fh:
            for n, line in enumerate(fh, 1):
                try:
                    line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    return n, line[exc.start]
    except OSError:
        pass
    return 0, 0


def _first_flag_after_pointers(path: str) -> str:
    try:
        with open(path, "r", encoding="latin-1") as fh:
            pointers = False
            for line in fh:
                if line.startswith("%FLAG"):
                    if pointers:
                        return line[5:].strip()
                    pointers = line.startswith("%FLAG POINTERS")
    except OSError:
        pass
    return ""


def _reader_error(exc: BaseException, path: str, kind: str) -> ReaderError:
    """``exc``, raised by MDAnalysis while it read ``path``, as an error that names the file, its format
    in words and the problem (Python's own exceptions are kept in brackets, for a bug report)."""
    what = f"cannot read the {_format_words(path)} {kind}"
    fmt = _TOPOLOGY_FORMATS.get(_extension(path)) or _TRAJECTORY_FORMATS.get(_extension(path))
    if any(isinstance(e, UnicodeDecodeError) for e in _chain(exc)):
        line, byte = _undecodable_line(path)
        if line:
            return ReaderError(f"{path} line {line}: not UTF-8 text (byte 0x{byte:02x}); save the file as "
                               "UTF-8 or plain ASCII")
    if fmt == "PARM7" and isinstance(exc, UnboundLocalError):
        flag = _first_flag_after_pointers(path)
        return ReaderError(f"{path}: {what}: the section %FLAG {flag} directly after POINTERS is not one "
                           "MDAnalysis knows, and it cannot skip a section in that place (move the section "
                           "to the end of the file)")
    if fmt == "TRJ" and isinstance(exc, StopIteration):
        return ReaderError(f"{path}: {what}: the file ends with its first frame. A single frame without a "
                           "box line cannot be read (MDAnalysis looks at the line after the first frame to "
                           "see whether the file has box lines)")
    text = re.sub(r"<class '(?:[\w.]*\.)?(\w+)'>", r"\1", _WRAPPED.sub("", str(exc)).strip())
    text = re.sub(r"\s*Note: n_atoms can be None.*", "", text, flags=re.S)  # advice for MDAnalysis' API
    if isinstance(exc, (ValueError, OSError, NotImplementedError)) and text:
        return ReaderError(f"{path}: {what}: {text}")
    return ReaderError(f"{path}: {what}: the file is damaged or not in this format "
                       f"({type(exc).__name__}{': ' + text if text else ''})")


def _forward_warnings(caught, topology: str, trajectories: Sequence[str], timed: bool) -> None:
    """Log what MDAnalysis warned about while reading, once per message, with the file it is about."""
    done = set()
    here = os.path.abspath(__file__)
    for w in caught:
        text = _ADVICE.sub("", " ".join(str(w.message).split()))
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning,
                                   ImportWarning, ResourceWarning)) or text.startswith(_SILENT):
            continue
        if text.startswith("Reader has no dt information") and not timed:
            continue  # a single structure has no times
        text = next((new for old, new in _REWORDED.items() if text.startswith(old)), text)
        source = str(w.filename).replace(os.sep, "/")
        about_frames = "/MDAnalysis/coordinates/" in source or os.path.abspath(str(w.filename)) == here
        path = ", ".join(trajectories) if trajectories and about_frames else topology
        if (path, text) not in done:
            done.add((path, text))
            log.warning("%s: %s", path, text)


def _guess_missing_masses(u, topology: str) -> None:
    """A PDB file whose element column is filled for some atoms only: MDAnalysis takes the masses from
    that column and gives the atoms without an element the mass 0, where it guesses every mass from the
    atom names when the column is empty.  The masses of those atoms are guessed from their names too."""
    try:
        if not u._topology.masses.is_guessed:  # masses read from the file (a parm7 extra point has mass 0)
            return
        blank = (u.atoms.masses == 0) & (np.asarray(u.atoms.elements, dtype=object) == "")
    except Exception:  # noqa: BLE001 - no masses or no elements
        return
    if not blank.any():
        return
    from MDAnalysis.guesser.default_guesser import DefaultGuesser
    guesser = DefaultGuesser(u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        masses = guesser.guess_masses(guesser.guess_types(u.atoms.names[blank]))
    u.atoms[blank].masses = masses
    unknown = int(np.count_nonzero(masses == 0))
    log.warning("%s: %d atom(s) have no element in the file; their masses are guessed from the atom names%s",
                topology, int(blank.sum()),
                f" ({unknown} could not be guessed and keep the mass 0)" if unknown else "")


def _failing_trajectory(topology: str, trajectories: Sequence[str], top_kwargs: Dict[str, object],
                        dt: Dict[str, object]):
    """(file, error) of the first trajectory of a chain that cannot be read on its own, else (None, None)."""
    import MDAnalysis as mda
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(topology, **top_kwargs)
            for t in trajectories:
                try:
                    u.load_new(t, format=_TRAJECTORY_FORMATS.get(_extension(t)), **dt)
                except Exception as exc:  # noqa: BLE001
                    return t, exc
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    return None, None


def load_universe(topology: str, trajectories: Sequence[str] = (), dt_ps: Optional[float] = None):
    """Create an MDAnalysis Universe, choosing formats from file extensions.

    A file that cannot be read raises :class:`ReaderError` (a ``ValueError``) naming the file; what
    MDAnalysis warns about while reading is logged as a warning with the file name."""
    import MDAnalysis as mda

    if not topology:
        raise ValueError("no topology file given")
    if not os.path.exists(topology):
        raise FileNotFoundError(f"topology not found: {topology}")
    for t in trajectories:
        if not os.path.exists(t):
            raise FileNotFoundError(f"trajectory not found: {t}")
    trajectories = list(trajectories)
    kwargs: Dict[str, object] = {}
    for path in [topology] + trajectories:
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            raise ReaderError(f"{path}: file is empty")
    ext = _extension(topology)
    if ext in _TOPOLOGY_FORMATS:
        kwargs["topology_format"] = _TOPOLOGY_FORMATS[ext]
        if not _compressed(topology):
            _check_parm7(topology)
    elif ext in (".pdb", ".ent") and not _compressed(topology):
        if _scan_pdb(topology):
            kwargs["topology_format"] = _hybrid36_pdb_parser()
    for path in trajectories or [topology]:  # the files the coordinates are read from
        if _compressed(path):
            continue
        if _extension(path) == ".gro":
            _check_gro(path)
        elif _extension(path) == ".dcd":
            _check_dcd(path)
    top_kwargs = dict(kwargs)
    formats = [_TRAJECTORY_FORMATS.get(_extension(t)) for t in trajectories]
    files: List[object] = list(trajectories)
    if len(set(formats)) == 1:
        if formats[0]:
            kwargs["format"] = formats[0]
    elif trajectories:  # a chain of several formats: each file with its own
        files = [list(zip(trajectories, formats))]
    zero_cell = [t for t, fmt in zip(trajectories, formats) if fmt == "NCDF" and _zero_cell(t)]
    if zero_cell:
        if set(formats) != {"NCDF"}:
            raise ReaderError(f"{zero_cell[0]}: the NetCDF trajectory has the cell lengths 0 0 0, which "
                              "cannot be read in a chain with trajectories of other formats")
        # MDAnalysis fails on such a frame when it converts the units of the cell; Amber NetCDF is in
        # MDAnalysis' own units (Å, ps), so nothing MDATK reads (positions, cell, time) needs converting
        kwargs["convert_units"] = False
        log.warning("%s: the cell lengths are 0 0 0: read as a trajectory without a periodic cell",
                    ", ".join(zero_cell))
    dt: Dict[str, object] = {"dt": float(dt_ps)} if dt_ps else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            u = mda.Universe(topology, *files, **kwargs, **dt)
        except Exception as exc:  # noqa: BLE001 - MDAnalysis raises whatever its readers run into
            if not trajectories or _raised_in(exc, "topology") or not _raised_in(exc, "coordinates"):
                kind = "coordinates" if _raised_in(exc, "coordinates") else "topology"
                error = _reader_error(exc, topology, kind)
            else:
                path, cause = (trajectories[0], exc) if len(trajectories) == 1 else _failing_trajectory(
                    topology, trajectories, top_kwargs, dt)
                error = (_reader_error(cause, path, "trajectory") if path else
                         ReaderError(f"cannot read the trajectories {', '.join(trajectories)} as one: {exc}"))
                _release(cause)
            _release(exc)
            raise error from exc
        if not hasattr(u, "trajectory"):
            raise ReaderError(f"{topology} has no coordinates; give a trajectory with --traj "
                              "(inputs.trajectories)")
        timed = bool(trajectories) and u.trajectory.n_frames > 1
        if timed and not dt_ps:
            try:
                u.trajectory.dt  # MDAnalysis warns here when the file has no times
            except Exception:  # noqa: BLE001 - found when the frames are read
                pass
    _forward_warnings(caught, topology, trajectories, timed)
    _guess_missing_masses(u, topology)
    if dt_ps and trajectories:
        u.trajectory.add_transformations(_FrameTimes(float(dt_ps)))
    log.info("loaded %s: %d atoms, %d residues, %d frames", os.path.basename(topology),
             u.atoms.n_atoms, u.residues.n_residues, u.trajectory.n_frames)
    return u


def frame_indices(n_frames: int, start: int = 0, stop: Optional[int] = None,
                  step: int = 1) -> np.ndarray:
    """0-based frame indices selected by a python-style slice."""
    if step < 1:
        raise ValueError("frame step must be >= 1")
    idx = np.arange(n_frames)[slice(start, stop, step)]
    if len(idx) == 0:
        raise ValueError(f"frame selection start={start} stop={'end' if stop is None else stop} step={step} "
                         f"selects no frames (trajectory has {n_frames})")
    return idx


def system_summary(u) -> Dict[str, object]:
    """Small JSON-friendly description of a loaded system."""
    resnames, counts = np.unique(u.residues.resnames, return_counts=True)
    order = np.argsort(-counts)
    traj = u.trajectory
    return {
        "n_atoms": int(u.atoms.n_atoms),
        "n_residues": int(u.residues.n_residues),
        "n_segments": int(u.segments.n_segments),
        "n_frames": int(traj.n_frames),
        "dt_ps": float(traj.dt) if traj.n_frames > 1 else None,
        "has_box": bool(traj.ts.dimensions is not None),
        "top_resnames": {str(resnames[k]): int(counts[k]) for k in order[:12]},
    }


def residue_table(u, max_rows: Optional[int] = None) -> List[Dict[str, object]]:
    rows = []
    for res in u.residues[: max_rows or None]:
        rows.append({"resid": int(res.resid), "resname": str(res.resname),
                     "segid": str(res.segid), "n_atoms": int(res.atoms.n_atoms)})
    return rows
