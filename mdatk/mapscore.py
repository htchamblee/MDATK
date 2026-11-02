"""Map-model scores per residue, regions ranked by them, and old/new body hybrids.

Scores
    ``qscore``: the map-model Q-score (Pintilie et al., Nat. Methods 2020) computed by ChimeraX's
    ``qscore`` command, which comes with the QScore bundle of the ChimeraX Toolshed (ChimeraX-QScore,
    T. I. Croll).  ChimeraX reads the model as it is: L residue names (LG5, 0A ...) and HETATM records
    are scored like any other residue, so no renamed copy is made.  Each model is scored by
    ``chimerax --nogui --cmd "open MODEL id 1 maxModels 1; open MAP id 2; qscore #1 toVolume #2 useGui
    false logDetails true [options]; exit"`` (``maxModels 1``: the first model of a multi-model file, as
    MDATK reads it), or several models in one ChimeraX run: the map opened once as #1, each model opened as
    #2, scored and closed, with a marker line logged before it.  The per-residue table that ``logDetails``
    prints (Chain, Number, Name, Qavg, Qworst, Qbb, Qsc) is read from the log tolerantly: blank lines,
    the dashed rule, tabs or spaces, ``<pre>`` tags or ChimeraX's ``INFO:`` lines around each row,
    insertion codes after the number and ``N/A`` cells; the ``Overall mean Q-Score`` line (a mean over
    atoms) closes it.  Qavg ranks the regions, or Qbb / Qsc.
    ``cc``: MDATK's own per-residue correlation, no external program: the model's map is simulated
    (:func:`mdatk.density.simulate_map`, molmap-style Gaussians at the given resolution) and, for each
    residue, correlated with the experimental map (trilinear) over the grid points within a radius of the
    residue's atoms.
Regions
    Named regions (``interface=B:25-34,C:58-63``) are averaged per model, with the difference to a reference
    region (default: the whole model) from which every named region is removed, so no residue counts on
    both sides.  Chain termini (the first and last nucleotide of every chain segment, from the model) are
    left out of every mean unless kept: a terminal residue has fewer neighbours and a lower Q for that
    reason alone.  Residues without a score (NaN, e.g. outside the map) are counted, never averaged.
    Models are ranked by default on the first region's difference to the reference, dQ = Q(interface) -
    Q(scaffold): the scaffold every decoy shares would dominate a whole-model mean, and dQ survives a change
    of map scaling or sharpening where an absolute Q does not.
Hybrids
    Every old/new combination of up to six bodies (residue ranges of an old and a re-fitted model), over the
    atoms both files have, coordinates copied verbatim; scored against a map, each body's main effect is the
    mean score of the hybrids with its new coordinates minus the mean of those with its old ones.
"""

from __future__ import annotations

import copy
import csv
import html
import itertools
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import pdbtools
from .ensemble import (EnsembleError, Range, ResKey, in_selection, is_hydrogen, key_text, parse_selection,
                       residue_base, residue_key, selection_residues, selection_text, write_table)
from .pdbtools import Residue, Structure

CHIMERAX = "chimerax"
MASK_RADIUS = 2.5  # Å around a residue's atoms for its correlation
MAX_BODIES = 6
_VIRTUAL = frozenset({"VRT", "XXX"})
_VIRTUAL_ATOMS = frozenset({"ORIG", "X", "Y", "Z"})
ResidueScores = Dict[ResKey, Tuple[str, float, int]]  # key -> (residue name, score, atoms / points)

# "qavg" last: MDATK's Q-score table has both it and 'score' (the column the regions were ranked on)
_Q_COLS = ("q", "qscore", "q_score", "q-score", "qscore_mean", "q_mean", "mean_q", "qscore_per_residue",
           "q_residue", "q-residue", "score", "cc", "qavg")
_CHAIN_COLS = ("chain", "chain_id", "chainid", "chn", "auth_asym_id", "label_asym_id", "asym_id")
_RESSEQ_COLS = ("resseq", "resnum", "resid", "resi", "res_num", "resno", "res_seq", "residue_number",
                "auth_seq_id", "seq_id", "residue_seq", "number")
_ICODE_COLS = ("icode", "ins_code", "insertion_code", "pdbx_pdb_ins_code", "inscode")
_RESNAME_COLS = ("resname", "resn", "res_name", "comp_id", "auth_comp_id", "residue_name", "base", "name")
_MODEL_COLS = ("model",)
_NOT_A_NUMBER = ("nan", "na", "n/a", "", "none")


def _is_virtual(res: Residue) -> bool:
    names = {a.name for a in res.atoms}
    return res.name.strip().upper() in _VIRTUAL or (bool(names) and names <= _VIRTUAL_ATOMS)


# --- reading score output ---------------------------------------------------------------------------------

def _split(line: str, delim: Optional[str]) -> List[str]:
    if delim is None:
        return line.split()
    return [v.strip() for v in next(csv.reader([line], delimiter=delim))]


def _find(header: Sequence[str], names: Sequence[str]) -> Optional[int]:
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    return None


def parse_score_table(text: str, column: str = "") -> Optional[Dict[str, ResidueScores]]:
    """Per-residue scores from a table of per-atom or per-residue values: {model: {key: (name, mean, n)}}
    ('' when the table has no model column).  The header is the first line with a score column and a
    residue-number column; per-atom rows are averaged per residue.  None when no such table is found."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    for h, line in enumerate(lines):
        delim = "\t" if "\t" in line else ("," if "," in line else None)
        header = _split(line, delim)
        qcol = _find(header, (column.lower(),) if column else _Q_COLS)
        rcol = _find(header, _RESSEQ_COLS)
        if qcol is None or rcol is None:
            continue
        ccol, icol = _find(header, _CHAIN_COLS), _find(header, _ICODE_COLS)
        ncol, mcol = _find(header, _RESNAME_COLS), _find(header, _MODEL_COLS)
        sums: Dict[str, Dict[ResKey, List]] = {}
        for row in lines[h + 1:]:
            vals = _split(row, delim)
            if len(vals) <= max(qcol, rcol):
                continue
            m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", vals[rcol].strip())
            if not m:
                continue
            try:
                q = float(vals[qcol])
            except ValueError:  # ChimeraX writes "N/A", quotes included
                q = math.nan if vals[qcol].strip().strip('"').lower() in _NOT_A_NUMBER else None
                if q is None:
                    continue
            chain = vals[ccol].strip() if ccol is not None and ccol < len(vals) else ""
            icode = (vals[icol].strip() if icol is not None and icol < len(vals) else "") or m.group(2)
            name = vals[ncol].strip() if ncol is not None and ncol < len(vals) else ""
            model = vals[mcol].strip() if mcol is not None and mcol < len(vals) else ""
            rec = sums.setdefault(model, {}).setdefault((chain, int(m.group(1)), icode), [name, [], 0])
            rec[1].append(q)
            rec[2] += 1
        if not sums:
            continue
        out: Dict[str, ResidueScores] = {}
        for model, recs in sums.items():
            res = {}
            for k, (name, qs, n) in recs.items():
                finite = [v for v in qs if v == v]
                res[k] = (name, sum(finite) / len(finite) if finite else math.nan, n)
            out[model] = res
        return out
    return None


# --- Q-score with ChimeraX ----------------------------------------------------------------------------------

# fixed qscore options: no GUI (ChimeraX runs with --nogui) and the per-residue table in the log
QSCORE_FIXED = "useGui false logDetails true"
# keywords the qscore command declares (ChimeraX-QScore 1.2, src/cmd.py); toVolume, useGui, logDetails and
# outputFile are MDATK's (an outputFile would not change the log, but each model would overwrite it)
QSCORE_KEYWORDS = ("referenceGaussianSigma", "pointsPerShell", "maxShellRadius", "shellRadiusStep",
                   "assignAttr")
QSCORE_RESERVED = ("toVolume", "useGui", "logDetails", "outputFile")
Q_COLUMNS = ("qavg", "qbb", "qsc")
QSCORE_MARKER = "MDATK-QSCORE"  # logged ('log text') before each model of a several-model ChimeraX run
MAX_COMMAND = 100_000  # characters in one --cmd argument: Linux limits a single argument to 128 KiB


@dataclass
class QRow:
    """One residue of ChimeraX's qscore table; N/A cells are NaN."""
    name: str
    qavg: float
    qworst: float
    qbb: float = math.nan
    qsc: float = math.nan


@dataclass
class QTable:
    """One per-residue table of a ChimeraX log: rows keyed (chain, number, insertion code) as ChimeraX prints
    them, the 'Overall mean Q-Score' that follows it (a mean over atoms; None while it is missing, e.g. when
    ChimeraX stopped inside the table) and the model index of the MDATK marker logged before it."""
    rows: Dict[ResKey, QRow] = field(default_factory=dict)
    overall: Optional[float] = None
    marker: Optional[int] = None

    @property
    def complete(self) -> bool:
        return self.overall is not None


def check_qscore_options(text: str) -> Tuple[str, List[str]]:
    """Further qscore options ('pointsPerShell 16 shellRadiusStep 0.05') as one line, and warnings for
    keywords the command does not declare.  EnsembleError for a ';' (another ChimeraX command), a keyword
    without a value, or one MDATK sets."""
    if ";" in (text or ""):
        raise EnsembleError("qscore options: no ';' (it would start another ChimeraX command)")
    words = (text or "").split()
    if len(words) % 2:
        raise EnsembleError(f"qscore options: {words[-1]!r} has no value (keyword value pairs, e.g. "
                            "'pointsPerShell 16')")
    known = {k.lower() for k in QSCORE_KEYWORDS}
    reserved = {k.lower() for k in QSCORE_RESERVED}
    warnings = []
    for kw in words[::2]:
        if kw.lower() in reserved:
            raise EnsembleError(f"qscore options: {kw} is set by MDATK ('{QSCORE_FIXED}', toVolume the map)")
        if kw.lower() not in known:
            warnings.append(f"qscore options: {kw} is not a keyword of the qscore command (ChimeraX-QScore "
                            f"1.2 has {', '.join(QSCORE_KEYWORDS)}); ChimeraX stops at a keyword it does not "
                            "know")
    return " ".join(words), warnings


_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LEVEL = re.compile(r"(?:INFO|WARNING|ERROR|BUG|STATUS):\s*$")  # ChimeraX's --nogui log prefixes
_OVERALL = re.compile(r"overall\s+mean\s+q-?score\s*[:=]?\s*(\S+)", re.I)
_RULE = re.compile(r"-{5,}")
_Q_HEADER = ["chain", "number", "name", "qavg", "qworst"]
_RES_NUMBER = re.compile(r"(-?\d+)([A-Za-z]?)")


def _q_value(text: str) -> float:
    t = text.strip().strip("\"'")
    return math.nan if t.lower() in _NOT_A_NUMBER else float(t)


def _q_cells(line: str) -> List[str]:
    """Tab-separated (the log), else comma-separated (qscore's outputFile), else split on spaces."""
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    if "," in line:
        return [c.strip() for c in line.split(",")]
    return line.split()


def _q_row(cells: List[str], width: int) -> Optional[Tuple[ResKey, QRow]]:
    if len(cells) == width - 1 and _RES_NUMBER.fullmatch(cells[0]):  # a blank chain, lost in a space split
        cells = [""] + cells
    if len(cells) < 5:
        return None
    m = _RES_NUMBER.fullmatch(cells[1])
    if not m:
        return None
    try:
        values = [_q_value(c) for c in cells[3:7]]
    except ValueError:
        return None
    values += [math.nan] * (4 - len(values))
    return (cells[0], int(m.group(1)), m.group(2)), QRow(cells[2], *values)


def parse_chimerax_qscore(text: str) -> List[QTable]:
    """Every per-residue table in a ChimeraX log, in order.

    Tolerant of what ChimeraX's plain-text log puts around the rows: blank lines, ``INFO:`` lines (or the
    prefix glued to the end of a row), ``<pre>`` tags when html2text is not installed, ANSI colours, the
    dashed rule, tabs or runs of spaces, warnings between the tables.  The number may carry an insertion
    code ('12A'); 'N/A' (with or without quotes) is NaN.  A table ends at its 'Overall mean Q-Score'
    line, the next header or the next MDATK marker; one without its overall line is not complete."""
    tables: List[QTable] = []
    cur: Optional[QTable] = None
    marker: Optional[int] = None
    width = 7
    for raw in (text or "").splitlines():
        line = _LEVEL.sub("", html.unescape(_TAG.sub("", _ANSI.sub("", raw)))).rstrip()
        body = line.lstrip(" ")  # html2text indents <pre> blocks; a blank chain is an empty cell before a tab
        if not body.strip():
            continue
        m = re.match(rf"{QSCORE_MARKER}\s+(\d+)\b", body.strip())
        if m:
            cur, marker = None, int(m.group(1))
            continue
        m = _OVERALL.search(body)
        if m:
            if cur is not None:
                try:
                    cur.overall = float(m.group(1))
                except ValueError:
                    cur.overall = math.nan
            cur = None
            continue
        cells = _q_cells(body)
        if [c.lower() for c in cells[:5]] == _Q_HEADER:
            cur = QTable(marker=marker)
            tables.append(cur)
            marker, width = None, len(cells)
            continue
        if cur is None or _RULE.fullmatch(body.strip()):
            continue
        row = _q_row(cells, width)
        if row is not None:
            cur.rows[row[0]] = row[1]
    return tables


def qscore_residues(table: QTable, structure: Optional[Structure] = None, column: str = "qavg"
                    ) -> Tuple[ResidueScores, Dict[ResKey, QRow], int]:
    """(scores by the chosen column, the ChimeraX rows, virtual residues left out), keyed as the model's
    residues: ChimeraX prints a blank chain as an empty cell.  The atom count is the residue's heavy atoms
    in the model (ChimeraX scores no hydrogens)."""
    if column not in Q_COLUMNS + ("qworst",):
        raise EnsembleError(f"unknown Q column {column!r} ({', '.join(Q_COLUMNS)})")
    index: Dict[Tuple[str, int, str], Residue] = {}
    for res in (structure.residues() if structure is not None else []):
        index.setdefault((res.chain.strip(), res.resseq, res.icode.strip()), res)
    scores: ResidueScores = {}
    rows: Dict[ResKey, QRow] = {}
    virtual = 0
    for (chain, number, icode), row in table.rows.items():
        res = index.get((chain.strip(), number, icode.strip()))
        if row.name.strip().upper() in _VIRTUAL or (res is not None and _is_virtual(res)):
            virtual += 1
            continue
        key = residue_key(res) if res is not None else (chain, number, icode)
        n = len({a.name for a in res.atoms if not is_hydrogen(a)}) if res is not None else 0
        scores[key] = (row.name, getattr(row, column), n)
        rows[key] = row
    return scores, rows, virtual


def _cx_path(path: str) -> str:
    """A file name in a ChimeraX command: absolute, in double quotes."""
    p = os.path.abspath(path)
    if '"' in p or ";" in p:
        raise EnsembleError(f"{p}: a ChimeraX command cannot name a file with '\"' or ';' in its path "
                            "(rename or link it)")
    return f'"{p}"'


def _virtual_names(structure: Optional[Structure]) -> List[str]:
    if structure is None:
        return []
    return sorted({r.name.strip() for r in structure.residues() if _is_virtual(r) and r.name.strip()})


def _qscore_target(model_id: int, virtual: Sequence[str]) -> str:
    """The residues to score: the whole model, minus Rosetta's virtual residues when it has any (their
    atoms sit near the origin, often outside the map box, where ChimeraX-QScore 1.2 stops)."""
    return f"#{model_id}" + "".join(f" & ~:{name}" for name in virtual)


def qscore_command(models: Sequence[Tuple[str, Sequence[str]]], map_path: str, options: str = "") -> str:
    """The ``--cmd`` text scoring ``models`` ([(path, its virtual residue names)]) against the map.

    One model: the user's command, model #1 and map #2.  Several: the map is opened once as #1 and each
    model in turn as #2 (``id 2``, so the numbering never drifts), scored and closed, a marker line logged
    before it so that each table can be told apart.  ``maxModels 1``: the first model of a multi-model PDB,
    as MDATK reads it (several submodels would make qscore refuse the selection)."""
    tail = QSCORE_FIXED + (f" {options}" if options else "")
    if len(models) == 1:
        path, virtual = models[0]
        return (f"open {_cx_path(path)} id 1 maxModels 1; open {_cx_path(map_path)} id 2; "
                f"qscore {_qscore_target(1, virtual)} toVolume #2 {tail}; exit")
    parts = [f"open {_cx_path(map_path)} id 1"]
    for k, (path, virtual) in enumerate(models, 1):
        parts += [f"log text {QSCORE_MARKER} {k}", f"open {_cx_path(path)} id 2 maxModels 1",
                  f"qscore {_qscore_target(2, virtual)} toVolume #1 {tail}", "close #2"]
    return "; ".join(parts + ["exit"])


def _failure(text: str, code: Optional[int]) -> str:
    """Why ChimeraX did not score a model, from its output."""
    m = re.search(r"have atoms outside the bounds of volume[^\n]*\n?([^\n]*)", text)
    if m:
        where = m.group(1).strip()[:200]
        return ("atoms outside the map box: ChimeraX-QScore stops its per-residue table at the first such "
                f"residue{' (' + where + ')' if where else ''}; enlarge or re-centre the map box")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not _LEVEL.fullmatch(ln.strip())]
    tail = lines[-1][:300] if lines else ""
    if code == 0:
        return "ChimeraX printed no complete per-residue Q-score table for it" + (
            f" ({tail})" if tail else "")
    return f"ChimeraX ended with status {code}" + (f": {tail}" if tail else "")


def _end_group(proc: "subprocess.Popen", grace: float = 5.0) -> None:
    """Terminate the process group that ``proc`` leads (a wrapper such as 'xvfb-run -a chimerax' with the
    ChimeraX it started): SIGTERM, then, once ``proc`` has ended or after ``grace`` seconds, SIGKILL to what
    is left of the group."""
    if os.name != "posix":
        if proc.poll() is None:
            proc.kill()
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:  # the group is gone
            return
        if sig == signal.SIGTERM:
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                pass


class _Stopped(BaseException):
    """MDATK was told to stop (SIGTERM, SIGHUP) while ChimeraX ran."""


def _run_chimerax(argv: List[str], cwd: str, limit: Optional[float]) -> Tuple[str, str, Optional[int]]:
    """Run ChimeraX: (standard output, standard error, exit status; None when stopped at the time limit).

    ChimeraX runs in a process group of its own, which the time limit ends whole: killing only the program
    MDATK started would leave ChimeraX running behind a wrapper ('xvfb-run -a chimerax', a launcher script).
    The script's group, which RNAdo's Stop terminates, no longer holds ChimeraX, so while it runs a SIGTERM
    or SIGHUP to MDATK (and Ctrl-C) ends ChimeraX's group too before MDATK ends."""
    posix = os.name == "posix"
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, errors="replace", start_new_session=posix)
    caught: List[int] = []

    def stop(signum, _frame):
        if not caught:  # a second signal must not cut the clean-up short
            caught.append(signum)
            raise _Stopped(signum)

    handlers = {}
    if posix and threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(sig) == signal.SIG_DFL:  # one that would end MDATK (not an ignored SIGHUP)
                handlers[sig] = signal.signal(sig, stop)
    try:
        try:
            out, err = proc.communicate(timeout=limit)
            return out or "", err or "", proc.returncode
        except subprocess.TimeoutExpired:
            _end_group(proc)
            try:
                out, err = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:  # a program that left the group still holds the output
                out, err = "", "(the output was not read: a program ChimeraX started still holds it)"
            return out or "", err or "", None
    finally:
        if proc.returncode is None:  # stopped while ChimeraX ran (a signal, Ctrl-C)
            _end_group(proc)
            proc.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if caught:  # end as the signal would have ended MDATK, now that ChimeraX has gone
            os.kill(os.getpid(), caught[0])


def run_chimerax_qscore(models: Sequence[str], map_path: str, work_dir: str, chimerax: str = CHIMERAX,
                        options: str = "", batch: int = 1, timeout: Optional[float] = None,
                        structures: Optional[Dict[str, Structure]] = None
                        ) -> Tuple[Dict[str, QTable], Dict[str, str], List[str]]:
    """Score ``models`` with ChimeraX's qscore, ChimeraX's log of each run kept in ``work_dir``:
    ({model: its table}, {model: why it was not scored}, notes).

    ``batch`` models share one ChimeraX run (1: one run per model, the safe default).  A run that stops at
    a model (ChimeraX ends a --nogui run at the first failing command) is resumed after it, so a bad model
    costs only its own score.  ``timeout`` is per model.  EnsembleError when ChimeraX cannot be started or
    has no qscore command."""
    opts, notes = check_qscore_options(options)
    exe = (chimerax or CHIMERAX).strip()
    try:  # a program path as it is (it may hold spaces), else a command line such as 'xvfb-run -a chimerax'
        argv0 = [exe] if os.path.isfile(exe) else shlex.split(exe)
    except ValueError as exc:
        raise EnsembleError(f"cannot read the ChimeraX executable {chimerax!r}: {exc}") from None
    if not argv0:
        raise EnsembleError("give the ChimeraX executable")
    os.makedirs(work_dir, exist_ok=True)
    structures = structures or {}
    virtual = {m: _virtual_names(structures.get(m)) for m in models}
    tables: Dict[str, QTable] = {}
    failed: Dict[str, str] = {}
    ignored = runs = 0
    used: Set[str] = set()
    pending = list(models)
    while pending:
        chunk = pending[:max(1, batch)]
        cmd = qscore_command([(m, virtual[m]) for m in chunk], map_path, opts)
        while len(chunk) > 1 and len(cmd) > MAX_COMMAND:
            chunk = chunk[:len(chunk) // 2]
            cmd = qscore_command([(m, virtual[m]) for m in chunk], map_path, opts)
        runs += 1
        stem = os.path.splitext(os.path.basename(chunk[0]))[0] if len(chunk) == 1 else f"run-{runs:03d}"
        name, same = f"{stem}_chimerax.log", 1
        while name in used:  # decoys of different runs share their tags (file names)
            same += 1
            name = f"{stem}-{same}_chimerax.log"
        used.add(name)
        log_path = os.path.join(work_dir, name)
        argv = argv0 + ["--nogui", "--cmd", cmd]
        limit = None if timeout is None else timeout * len(chunk)
        try:
            out, err, code = _run_chimerax(argv, work_dir, limit)
        except FileNotFoundError:
            raise EnsembleError(f"ChimeraX not found: {argv0[0]} (give the chimerax program, e.g. "
                                "/usr/bin/chimerax or /opt/UCSF/ChimeraX/bin/ChimeraX)") from None
        if code is None:
            err += f"\nstopped after {limit:g} s (time limit)"
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"$ {shlex.join(argv)}\n{out}\n{err}")
        both = out + "\n" + err
        ignored += len(re.findall(r"Ignored bad PDB record", both))
        if re.search(r"unknown command:?\s*qscore", both, re.I):
            raise EnsembleError("ChimeraX has no qscore command: install the QScore bundle from the ChimeraX "
                                f"Toolshed (in ChimeraX: toolshed install QScore); see {log_path}")
        found = parse_chimerax_qscore(out)
        got: Dict[str, QTable] = {}
        if len(chunk) == 1:
            first = next((t for t in found if t.complete), None)
            if first is not None:
                got[chunk[0]] = first
        else:
            for t in found:
                if t.complete and t.marker is not None and 1 <= t.marker <= len(chunk):
                    got.setdefault(chunk[t.marker - 1], t)
        tables.update(got)
        rest = pending[len(chunk):]
        missing = [m for m in chunk if m not in got]
        if not missing:
            if code != 0:
                notes.append(f"ChimeraX ended with status {code} after scoring every model of "
                             f"{os.path.basename(log_path)}")
        elif code == 0:  # ChimeraX ran to the end: the models without a table will not get one
            for m in missing:
                failed[m] = _failure(both, code) + f" (log: {log_path})"
        else:  # it stopped at the first model without a table; those after it never ran
            k = chunk.index(missing[0])
            failed[chunk[k]] = _failure(both, code) + f" (log: {log_path})"
            rest = chunk[k + 1:] + rest
        pending = rest
    if ignored:
        notes.append(f"ChimeraX ignored {ignored} PDB record(s) it could not read (e.g. free-text REMARK "
                     "lines); the scores do not depend on them")
    return tables, failed, notes


# --- MDATK's per-residue correlation -----------------------------------------------------------------------

def _element(atom: pdbtools.Atom) -> str:
    if atom.element:
        return atom.element.strip().upper()
    letters = re.sub(r"^\d+", "", atom.name.strip())
    return letters[:1].upper()


def residue_cc(structure: Structure, dmap, resolution: float, radius: float = MASK_RADIUS,
               keep_h: bool = False) -> ResidueScores:
    """Per residue: the correlation of the experimental map with the model's simulated map over the grid
    points within ``radius`` of the residue's atoms (NaN with fewer than 4 points inside the map)."""
    from .density import atom_weights, sample_map, simulate_map
    from .spatial import pairs_within
    if not resolution or resolution <= 0:
        raise EnsembleError("the per-residue correlation needs the map resolution")
    atoms: List[Tuple[ResKey, str, Tuple[float, float, float], str]] = []
    for res in structure.residues():
        if _is_virtual(res):
            continue
        seen = set()
        for a in res.atoms:
            if a.name in seen or (not keep_h and is_hydrogen(a)):
                continue
            seen.add(a.name)
            atoms.append((residue_key(res), res.name, a.xyz, _element(a)))
    if not atoms:
        raise EnsembleError("the model has no atoms to score")
    xyz = np.array([a[2] for a in atoms], dtype=np.float64)
    spacing = min(float(np.min(dmap.voxel)), resolution / 3.0)
    sim = simulate_map(xyz, atom_weights([a[3] for a in atoms]), resolution, spacing=spacing)
    h = float(sim.voxel[0])
    nz, ny, nx = sim.data.shape
    out: ResidueScores = {}
    by_res: Dict[ResKey, List[int]] = {}
    names: Dict[ResKey, str] = {}
    for k, (key, name, _xyz, _el) in enumerate(atoms):
        by_res.setdefault(key, []).append(k)
        names[key] = name
    for key, idx in by_res.items():
        pts = xyz[idx]
        lo = np.maximum(np.floor((pts.min(axis=0) - radius - sim.origin) / h).astype(int), 0)
        top = np.ceil((pts.max(axis=0) + radius - sim.origin) / h).astype(int)
        hi = np.minimum(top, [nx - 1, ny - 1, nz - 1])
        gx, gy, gz = (np.arange(lo[i], hi[i] + 1) for i in range(3))
        grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
        coords = sim.origin + grid * h
        near = np.unique(pairs_within(coords, pts, radius)[0])
        grid, coords = grid[near], coords[near]
        s_vals = sim.data[grid[:, 2], grid[:, 1], grid[:, 0]].astype(np.float64)
        e_vals = sample_map(dmap, coords, cval=np.nan)
        ok = np.isfinite(e_vals)
        cc = math.nan
        if ok.sum() >= 4:
            s, e = s_vals[ok] - s_vals[ok].mean(), e_vals[ok] - e_vals[ok].mean()
            den = math.sqrt(float(np.dot(s, s)) * float(np.dot(e, e)))
            if den > 0:
                cc = float(np.dot(s, e) / den)
        out[key] = (names[key], cc, int(ok.sum()))
    return out


# --- scoring several models ---------------------------------------------------------------------------------

@dataclass
class ModelScores:
    """One model's per-residue scores; for Q-scores also ChimeraX's rows and its overall mean."""
    model: str
    structure: Structure
    scores: ResidueScores
    q_rows: Optional[Dict[ResKey, QRow]] = None
    overall: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def score_models(models: Sequence[str], map_path: str, method: str = "cc", *,
                 resolution: Optional[float] = None, radius: float = MASK_RADIUS, keep_h: bool = False,
                 chimerax: str = CHIMERAX, options: str = "", batch: int = 1, column: str = "qavg",
                 work_dir: str = "", timeout: Optional[float] = None
                 ) -> Tuple[List[ModelScores], Dict[str, str], List[str]]:
    """Per-residue scores of every model by 'cc' or 'qscore': (the scored models in order, {model: why it
    was not scored} (qscore), notes)."""
    if method not in ("cc", "qscore"):
        raise EnsembleError(f"unknown scoring method {method!r} (cc or qscore)")
    models = list(dict.fromkeys(models))  # a file given twice is scored (and ranked) once
    structures = {m: pdbtools.read_pdb(m) for m in models}
    if method == "cc":
        from .density import load_map_cached
        dmap = load_map_cached(map_path)
        return [ModelScores(m, s, residue_cc(s, dmap, resolution or 0.0, radius, keep_h))
                for m, s in structures.items()], {}, []
    if column not in Q_COLUMNS:
        raise EnsembleError(f"unknown Q column {column!r} ({', '.join(Q_COLUMNS)})")
    if not os.path.isfile(map_path):
        raise EnsembleError(f"map not found: {map_path}")
    work = work_dir or os.path.join(os.path.dirname(os.path.abspath(models[0])), "qscore")
    tables, failed, notes = run_chimerax_qscore(models, map_path, work, chimerax, options, batch, timeout,
                                                structures)
    if keep_h:
        notes.append("ChimeraX's qscore never scores hydrogens: --keep-h has no effect on Q-scores")
    out = []
    for m in models:
        if m not in tables:
            continue
        scores, rows, virtual = qscore_residues(tables[m], structures[m], column)
        more = [f"ChimeraX's overall mean Q-Score {tables[m].overall:.2f} (all atoms, termini included)"]
        if virtual:
            more.append(f"{virtual} virtual residue(s) (Rosetta VRT / XXX) left out")
        if not scores:
            failed[m] = "ChimeraX's table has no residue of the model"
            continue
        out.append(ModelScores(m, structures[m], scores, rows, tables[m].overall, more))
    return out, failed, notes


# --- regions ------------------------------------------------------------------------------------------------

@dataclass
class Region:
    name: str
    selection: List[Range]


def parse_region(text: str, groups: Sequence[Any] = ()) -> Region:
    """'NAME=RANGES' (ranges as in :func:`mdatk.ensemble.parse_selection`)."""
    name, sep, ranges = (text or "").partition("=")
    if not sep or not name.strip():
        raise EnsembleError(f"cannot read the region {text!r} (expected NAME=RANGES, e.g. loops=B:25-34)")
    name = name.strip()
    if name in ("reference", "whole") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name):
        raise EnsembleError(f"region name {name!r}: letters, digits, '.', '_', '-', '+'; not 'reference' or "
                            "'whole'")
    sel = parse_selection(ranges, groups)
    if not sel:
        raise EnsembleError(f"region {name} has no residues")
    return Region(name, sel)


def structure_termini(structure: Structure) -> Set[ResKey]:
    """First and last nucleotide (else residue) of every chain segment."""
    out: Set[ResKey] = set()
    for seg in structure.segments:
        by_chain: Dict[str, List[Residue]] = {}
        for res in seg:
            by_chain.setdefault(res.chain, []).append(res)
        for residues in by_chain.values():
            nts = [r for r in residues if residue_base(r)] or residues
            out.add(residue_key(nts[0]))
            out.add(residue_key(nts[-1]))
    return out


def key_termini(keys: Sequence[ResKey]) -> Set[ResKey]:
    """Termini from the residue numbers alone: the lowest and highest of every chain."""
    by_chain: Dict[str, List[ResKey]] = {}
    for k in keys:
        by_chain.setdefault(k[0], []).append(k)
    out: Set[ResKey] = set()
    for ks in by_chain.values():
        ks = sorted(ks, key=lambda k: (k[1], k[2]))
        out.update((ks[0], ks[-1]))
    return out


def region_rows(model: str, scores: ResidueScores, regions: Sequence[Region],
                reference: Optional[Sequence[Range]] = None, termini: Set[ResKey] = frozenset(),
                keep_termini: bool = False) -> List[Dict[str, Any]]:
    """One row per named region, then 'reference' (its ranges, or the whole model, minus every named region)
    and 'whole': residues, scored, NaN and missing counts, the mean and its difference to the reference."""
    drop = set() if keep_termini else set(termini)
    keys = [k for k in scores if k not in drop]
    have = {(k[0], k[1]) for k in scores}
    have_any = {k[1] for k in scores}
    named = [reg.selection for reg in regions]

    def absent(sel: Optional[Sequence[Range]], skip: Sequence[Sequence[Range]] = ()) -> int:
        """Residues a selection names explicitly that the model has no score row for."""
        if sel is None:
            return 0
        n = 0
        for c, r in selection_residues(sel):
            if any(in_selection((c or "", r, ""), s) for s in skip):
                continue  # counted with its named region
            if (r not in have_any) if c is None else ((c, r) not in have):
                n += 1
        return n

    def summary(name: str, members: List[ResKey], missing: int) -> Dict[str, Any]:
        vals = [scores[k][1] for k in members]
        finite = [v for v in vals if v == v]
        return {"model": model, "region": name, "residues": len(members), "scored": len(finite),
                "nan": len(vals) - len(finite), "missing": missing,
                "mean": sum(finite) / len(finite) if finite else math.nan}

    rows = [summary(reg.name, [k for k in keys if in_selection(k, reg.selection)], absent(reg.selection))
            for reg in regions]
    base = [k for k in keys if (reference is None or in_selection(k, reference))
            and not any(in_selection(k, sel) for sel in named)]
    ref_row = summary("reference", base, absent(reference, named))
    whole = summary("whole", keys, 0)
    for row in rows + [ref_row, whole]:
        row["delta"] = row["mean"] - ref_row["mean"] if row is not ref_row else 0.0
    return rows + [ref_row, whole]


REGION_COLUMNS = ["model", "region", "residues", "scored", "nan", "missing", "mean", "delta", "rank"]


def rank_models(rows: List[Dict[str, Any]], sort: str) -> List[str]:
    """Models best first by a region's mean ('NAME'), its difference to the reference ('delta:NAME'),
    'reference' or 'whole'; NaN last.  Sets each row's 'rank'."""
    region, which = (sort.split(":", 1)[1], "delta") if sort.startswith("delta:") else (sort, "mean")
    key = {}
    for r in rows:
        if r["region"] == region:
            v = r[which]
            key[r["model"]] = -v if v == v else math.inf
    models = list(dict.fromkeys(r["model"] for r in rows))
    if region not in {r["region"] for r in rows}:
        raise EnsembleError(f"cannot sort by {sort!r}: no region {region!r}")
    order = sorted(models, key=lambda m: (key.get(m, math.inf), models.index(m)))
    rank = {m: i + 1 for i, m in enumerate(order)}
    for r in rows:
        r["rank"] = rank[r["model"]]
    return order


def sort_text(sort: str, label: str = "") -> str:
    """'delta:interface' -> 'dQ:interface (Q of interface minus Q of the reference)' and so on."""
    if sort.startswith("delta:"):
        name = sort.split(":", 1)[1]
        what = label or "the score"
        return f"d{label}:{name} ({what} of {name} minus {what} of the reference)"
    return f"{label}:{sort}" if label else sort


def regions_report(rows: List[Dict[str, Any]], order: Sequence[str], regions: Sequence[Region],
                   reference: Optional[Sequence[Range]], keep_termini: bool, sort: str, what: str,
                   label: str = "") -> str:
    """The ranking: per model the mean of every region, its difference to the reference ('dQ:NAME' with
    ``label`` 'Q'), the reference and the whole model, best first by ``sort``."""
    names = [r.name for r in regions]
    cols = names + [f"d:{n}" for n in names] + ["reference", "whole"]
    heads = [(f"d{label}:{c[2:]}" if c.startswith("d:") else c) for c in cols]
    widths = [max(10, len(h)) for h in heads]
    by = {(r["model"], r["region"]): r for r in rows}
    out = [f"{len(order)} models ranked by {sort_text(sort, label)}, best first ({what}); "
           f"{'chain termini kept' if keep_termini else 'chain termini left out'}"]
    for reg in regions:
        out.append(f"  {reg.name}: {selection_text(reg.selection)}")
    out.append(f"  reference: {selection_text(reference) if reference else 'the whole model'} minus the "
               "named regions")
    out.append("")
    out.append(f"{'rank':>4} {'model':<32} " + " ".join(f"{h:>{w}}" for h, w in zip(heads, widths))
               + "  notes")
    for i, m in enumerate(order, 1):
        cells, notes = [], []
        for c, w in zip(cols, widths):
            if c.startswith("d:"):
                v = by.get((m, c[2:]), {}).get("delta", math.nan)
                cells.append(f"{v:+{w}.3f}" if v == v else f"{'-':>{w}}")
            else:
                r = by.get((m, c), {})
                v = r.get("mean", math.nan)
                cells.append(f"{v:{w}.3f}" if v == v else f"{'-':>{w}}")
                if r.get("missing"):
                    notes.append(f"{c}: {r['missing']} missing")
                if r.get("nan"):
                    notes.append(f"{c}: {r['nan']} without a score")
        out.append(f"{i:4d} {os.path.basename(m)[:32]:<32} " + " ".join(cells) + ("  " + "; ".join(notes)
                                                                                   if notes else ""))
    if len(order) > 1:
        out.append("")
        spreads = [(c, "mean", c) for c in names] + [(n, "delta", f"d{label}:{n}") for n in names]
        for c, which, head in spreads + [("reference", "mean", "reference"), ("whole", "mean", "whole")]:
            v = [by[(m, c)][which] for m in order if (m, c) in by]
            v = [x for x in v if x == x]
            if v:
                out.append(f"{head:<16} spread {min(v):+.3f} to {max(v):+.3f} (range {max(v) - min(v):.3f})")
    return "\n".join(out) + "\n"


_CHIMERAX_OUTPUT = re.compile(r"UCSF ChimeraX|Traceback \(most recent call last\)|outside the bounds of "
                              r"volume|^(?:INFO|WARNING|ERROR):", re.M)


def _is_chimerax_log(text: str, tables: Sequence[QTable]) -> bool:
    """Whether qscore tables come from a ChimeraX log, where a table without its 'Overall mean Q-Score' line
    is one ChimeraX stopped inside (not qscore's CSV outputFile, which has no such line)."""
    return (any(t.complete for t in tables) or text.startswith("$ ") or QSCORE_MARKER in text
            or bool(_CHIMERAX_OUTPUT.search(text)))


def read_residue_scores(path: str, column: str = "", skipped: Optional[List[str]] = None
                        ) -> Dict[str, ResidueScores]:
    """Per-residue tables: MDATK's (model, chain, resseq, ...), a ChimeraX log with qscore tables (one model
    per table: ``path#1``, ``path#2`` ... when it has several) or qscore's CSV, or headed / headless CSV with
    the columns chain, residue number, base, Q (as written per decoy by Q-score scripts).  Keys are
    chain-qualified.

    A table of a ChimeraX log without its 'Overall mean Q-Score' line is one ChimeraX stopped inside (the
    log of a model that failed, which ``mdatk map score`` keeps too): it holds the residues printed before
    the failure only, so it is not read: why goes into ``skipped`` (without that list: EnsembleError)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    q_tables = [t for t in parse_chimerax_qscore(text) if t.rows]
    if q_tables:
        q_column = (column or "qavg").lower()
        named = len(q_tables) > 1
        tables: Dict[str, ResidueScores] = {}
        stopped = []
        log = _is_chimerax_log(text, q_tables)
        for k, t in enumerate(q_tables, 1):
            name = f"{path}#{k}" if named else path
            if log and not t.complete:
                stopped.append(f"{name}: ChimeraX stopped inside its per-residue table (no 'Overall mean "
                               f"Q-Score' line after {len(t.rows)} residues), e.g. at atoms outside the map "
                               "box: not ranked")
            else:
                tables[name] = qscore_residues(t, None, q_column)[0]
        if stopped and skipped is None:
            raise EnsembleError("; ".join(stopped))
        if skipped is not None:
            skipped += stopped
        return tables
    got = parse_score_table(text, column)
    if got:
        return {(m or path): s for m, s in got.items()}
    out: ResidueScores = {}  # headless: chain, resnum, base, Q
    for row in csv.reader(text.splitlines()):
        if len(row) < 4:
            continue
        m = re.fullmatch(r"\s*(-?\d+)([A-Za-z]?)\s*", row[1])
        try:
            q = float(row[3])
        except ValueError:
            continue
        if m:
            out[(row[0].strip(), int(m.group(1)), m.group(2))] = (row[2].strip(), q, 1)
    if not out:
        raise EnsembleError(f"{path}: no per-residue scores found")
    return {path: out}


# --- per-residue table -------------------------------------------------------------------------------------

RESIDUE_COLUMNS = ["model", "chain", "resseq", "icode", "resname", "score", "n", "method"]
# Q-scores: ChimeraX's four columns, then 'score' (the one the regions are ranked on) and the heavy atoms
QSCORE_COLUMNS = ["model", "chain", "resseq", "icode", "resname", "qavg", "qworst", "qbb", "qsc", "score",
                  "n", "method"]


def residue_table_rows(model: str, scores: ResidueScores, method: str,
                       q_rows: Optional[Dict[ResKey, QRow]] = None) -> List[List[Any]]:
    """Rows of the per-residue table (QSCORE_COLUMNS with ``q_rows``, else RESIDUE_COLUMNS)."""
    def num(v: float, digits: int) -> str:
        return "nan" if v != v else f"{v:.{digits}f}"

    rows = []
    for k in sorted(scores, key=lambda k: (k[0], k[1], k[2])):
        name, v, n = scores[k]
        row = [model, k[0], k[1], k[2], name]
        if q_rows is not None:  # ChimeraX prints three decimals
            q = q_rows.get(k)
            row += [num(x, 3) for x in (q.qavg, q.qworst, q.qbb, q.qsc)] if q else ["nan"] * 4
            row.append(num(v, 3))
        else:
            row.append(num(v, 4))
        rows.append(row + [n, method])
    return rows


# --- body hybrids -----------------------------------------------------------------------------------------

@dataclass
class Body:
    name: str
    selection: List[Range]


def parse_body(text: str, groups: Sequence[Any] = ()) -> Body:
    reg = parse_region(text, groups)
    return Body(reg.name, reg.selection)


def _residue_index(structure: Structure, match: str) -> Dict[Tuple, Residue]:
    out: Dict[Tuple, Residue] = {}
    dupes = []
    for res in structure.residues():
        if _is_virtual(res):
            continue
        k = residue_key(res) if match == "chain" else (res.resseq, res.icode)
        if k in out:
            dupes.append(k)
            continue
        out[k] = res
    if dupes:
        what = ", ".join(str(d[0]) + d[1] for d in dupes[:6]) if match == "number" else \
            ", ".join(key_text(d) for d in dupes[:6])
        raise EnsembleError(f"residue numbers used twice ({what})" + (
            ": matching by number alone needs numbers unique across chains; match by chain and number"
            if match == "number" else ""))
    return out


def hybrid_codes(n: int) -> List[str]:
    """'ooo', 'oon', ... 'nnn': o = old coordinates, n = new, one letter per body in order."""
    return ["".join(c) for c in itertools.product("on", repeat=n)]


def make_hybrids(old: Structure, new: Structure, bodies: Sequence[Body], match: str = "chain"
                 ) -> Tuple[Dict[str, Structure], Dict[str, List[ResKey]], List[str]]:
    """Every old/new combination of the bodies: (code -> structure, body -> its residues (old keys), notes).

    Residues are matched by chain and number (``match='number'``: by number alone, for files whose chain
    ids differ; the numbers must then be unique).  Only residues in both files and, per residue, only the
    atoms both have are written, so every hybrid has the same atoms; residues in no body keep their old
    coordinates.  Coordinates are copied verbatim from the source of each body."""
    if not bodies:
        raise EnsembleError("define at least one body")
    if len(bodies) > MAX_BODIES:
        raise EnsembleError(f"at most {MAX_BODIES} bodies ({2 ** MAX_BODIES} combinations)")
    names = [b.name for b in bodies]
    if len(set(names)) != len(names):
        raise EnsembleError("body names must differ")
    io, inew = _residue_index(old, match), _residue_index(new, match)
    shared = [k for k in io if k in inew]
    notes = []
    only = len(io) + len(inew) - 2 * len(shared)
    if only:
        notes.append(f"{only} residues are in only one of the files and are left out")
    which: Dict[Tuple, Optional[int]] = {}
    members: Dict[str, List[ResKey]] = {b.name: [] for b in bodies}
    overlaps = []
    for k in shared:
        rk = residue_key(io[k])
        hit = [i for i, b in enumerate(bodies) if in_selection(rk, b.selection)]
        if len(hit) > 1:
            overlaps.append(f"{key_text(rk)} ({' and '.join(names[i] for i in hit)})")
        which[k] = hit[0] if hit else None
        if hit:
            members[bodies[hit[0]].name].append(rk)
    if overlaps:
        raise EnsembleError("bodies overlap: " + ", ".join(overlaps[:6]))
    empty = [b.name for b in bodies if not members[b.name]]
    if empty:
        raise EnsembleError(f"no residue of the two files lies in body {', '.join(empty)}: check its ranges")
    loose = sum(1 for k in shared if which[k] is None)
    if loose:
        notes.append(f"{loose} residues lie in no body and keep their old coordinates in every combination")
    common = {k: [a.name for a in io[k].atoms if a.name in {b.name for b in inew[k].atoms}] for k in shared}
    dropped = sum(len({a.name for a in io[k].atoms} | {a.name for a in inew[k].atoms}) - len(set(common[k]))
                  for k in shared)
    if dropped:
        notes.append(f"{dropped} atoms are not in both files and are left out of every combination")
    out: Dict[str, Structure] = {}
    for code in hybrid_codes(len(bodies)):
        segments = []
        for seg in old.segments:
            kept = []
            for res in seg:
                k = residue_key(res) if match == "chain" else (res.resseq, res.icode)
                if k not in which:
                    continue
                b = which[k]
                src = inew[k] if b is not None and code[b] == "n" else io[k]
                atoms, seen = [], set()
                wanted = set(common[k])
                for a in src.atoms:
                    if a.name in wanted and a.name not in seen:  # first alternate location
                        seen.add(a.name)
                        atoms.append(copy.copy(a))
                r = Residue(src.name, res.chain, res.resseq, res.icode, atoms)
                kept.append(r)
            if kept:
                segments.append(kept)
        label = " ".join(f"{b.name}={c}" for b, c in zip(bodies, code))
        header = [ln for ln in old.header if not ln.startswith("REMARK 951 MDATK BODY HYBRID")]
        header.append(f"REMARK 951 MDATK BODY HYBRID {label} (o = old, n = new coordinates)"[:80])
        out[code] = Structure(header=header, segments=segments)
    return out, members, notes


def body_effects(bodies: Sequence[Body], scores: Dict[str, ResidueScores], members: Dict[str, List[ResKey]]
                 ) -> Tuple[List[List[Any]], List[List[Any]]]:
    """(per-hybrid rows, per-body rows).  A hybrid's score is the mean of its per-residue scores (over all
    residues, and per body over that body's residues); a body's main effect is the mean over the hybrids
    with its new coordinates minus the mean over those with its old ones, for the whole model and for the
    body itself."""
    def mean(vals):
        f = [v for v in vals if v == v]
        return sum(f) / len(f) if f else math.nan

    whole, local = {}, {}
    hyb_rows = []
    for code, sc in scores.items():
        whole[code] = mean(v for _, v, _ in sc.values())
        local[code] = {b.name: mean(sc[k][1] for k in members[b.name] if k in sc) for b in bodies}
        hyb_rows.append([code, f"{whole[code]:.4f}"] + [f"{local[code][b.name]:.4f}" for b in bodies])
    body_rows = []
    for i, b in enumerate(bodies):
        n_codes = [c for c in scores if c[i] == "n"]
        o_codes = [c for c in scores if c[i] == "o"]
        n_w, o_w = mean(whole[c] for c in n_codes), mean(whole[c] for c in o_codes)
        n_l, o_l = mean(local[c][b.name] for c in n_codes), mean(local[c][b.name] for c in o_codes)
        body_rows.append([b.name, len(members[b.name]), f"{n_w - o_w:+.4f}", f"{o_w:.4f}", f"{n_w:.4f}",
                          f"{n_l - o_l:+.4f}", f"{o_l:.4f}", f"{n_l:.4f}"])
    return hyb_rows, body_rows


HYBRID_COLUMNS = ["hybrid", "whole"]
EFFECT_COLUMNS = ["body", "residues", "effect_whole", "old_whole", "new_whole", "effect_body", "old_body",
                  "new_body"]


# --- command line -------------------------------------------------------------------------------------------

def _add_qscore_arguments(p) -> None:
    p.add_argument("--chimerax", default=CHIMERAX, help="qscore: the ChimeraX program (with the QScore "
                   "bundle of the ChimeraX Toolshed)")
    p.add_argument("--qscore-options", default="", metavar="OPTIONS",
                   help=f"qscore: further options of the qscore command, e.g. 'pointsPerShell 16' (always "
                        f"'{QSCORE_FIXED}')")
    p.add_argument("--batch", type=int, default=1, help="qscore: models per ChimeraX run (default 1: one run "
                   "per model; more: the map is opened once)")
    p.add_argument("--q-column", choices=Q_COLUMNS, default="qavg",
                   help="qscore: the column the regions are ranked on (Qavg, Qbb backbone, Qsc base)")
    p.add_argument("--work", default="", help="qscore: work folder for ChimeraX's logs (default: qscore/ "
                   "beside the first output file)")
    p.add_argument("--timeout", type=float, help="qscore: seconds per model")


def add_score_arguments(p) -> None:
    p.add_argument("--map", required=True, help="density map (MRC / CCP4)")
    p.add_argument("--method", choices=("cc", "qscore"), default="cc",
                   help="cc: MDATK's per-residue correlation (no external program); qscore: ChimeraX's "
                        "Q-score")
    p.add_argument("--resolution", type=float, help="map resolution (Å; needed by cc)")
    p.add_argument("--radius", type=float, default=MASK_RADIUS, help="cc: grid points this close to a "
                   "residue's atoms (Å)")
    _add_qscore_arguments(p)
    p.add_argument("--keep-h", action="store_true", help="cc: score hydrogens too (qscore never does)")


def add_region_arguments(p) -> None:
    p.add_argument("--region", action="append", default=[], metavar="NAME=RANGES",
                   help="a named region, e.g. 'loops=B:25-34,C:58-63' (repeat for each)")
    p.add_argument("--reference-region", default="", help="reference ranges (default: the whole model); the "
                   "named regions are always removed from it")
    p.add_argument("--keep-termini", action="store_true", help="keep chain termini in the means")
    p.add_argument("--sort", default="", help="rank by a region's mean ('NAME'), its difference to the "
                   "reference ('delta:NAME', dQ for Q-scores), 'reference' or 'whole' (default: delta of the "
                   "first region, else whole)")


def add_commands(sub) -> list:
    """'mdatk map score'."""
    c = sub.add_parser("score", help="per-residue map-model scores (MDATK correlation or Q-score) and "
                       "regions")
    c.add_argument("models", nargs="+", help="model PDB files")
    add_score_arguments(c)
    add_region_arguments(c)
    c.add_argument("--project", "-p", dest="params", help="project file for group:NAME regions")
    c.add_argument("-o", "--output", help="per-residue table (.tsv / .csv)")
    c.add_argument("--regions-out", help="region table (.tsv / .csv)")
    c.add_argument("--report", help="report text file (default: standard output)")
    return [c]


def add_hybrid_command(sub):
    """'mdatk pdb hybrids'."""
    c = sub.add_parser("hybrids", help="every old/new combination of re-fitted bodies, optionally scored "
                       "against a map")
    c.add_argument("old", help="the model before the re-fit")
    c.add_argument("new", help="the re-fitted model")
    c.add_argument("--body", action="append", default=[], required=True, metavar="NAME=RANGES",
                   help=f"a body, e.g. 'terminal=A:1-6,B:39-44' (repeat, at most {MAX_BODIES})")
    c.add_argument("--match", choices=("chain", "number"), default="chain",
                   help="match residues by chain and number, or by number alone (chain ids differ)")
    c.add_argument("--out-dir", "-o", default=".", help="folder of the hybrids")
    c.add_argument("--names", default="hyb_{code}.pdb", help="file names, {code} = o/n per body")
    c.add_argument("--map", help="score every hybrid against this map")
    c.add_argument("--method", choices=("cc", "qscore"), default="cc")
    c.add_argument("--resolution", type=float, help="map resolution (Å; needed by cc)")
    c.add_argument("--radius", type=float, default=MASK_RADIUS)
    _add_qscore_arguments(c)
    c.add_argument("--keep-h", action="store_true")
    c.add_argument("--scores-out", help="per-hybrid scores (.tsv / .csv)")
    c.add_argument("--effects-out", help="main effect of every body (.tsv / .csv)")
    c.add_argument("--project", "-p", dest="params", help="project file for group:NAME ranges")
    c.add_argument("--report", help="report text file (default: standard output)")
    return c


def _groups(args) -> List[Any]:
    if not getattr(args, "params", None):
        return []
    from .config import Project
    return list(Project.load(args.params).groups)


def _emit(text: str, path: Optional[str]) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def _region_setup(args, groups) -> Tuple[List[Region], Optional[List[Range]], str]:
    """Regions, reference and ranking; by default the models are ranked on the first region's difference to
    the reference (dQ = Q(interface) - Q(scaffold)), which the scaffold shared by every decoy normalises."""
    regions = [parse_region(t, groups) for t in args.region]
    reference = parse_selection(args.reference_region, groups) if args.reference_region.strip() else None
    sort = args.sort or (f"delta:{regions[0].name}" if regions else "whole")
    return regions, reference, sort


def _scored(args, models: Sequence[str], work: str) -> Tuple[List[ModelScores], Dict[str, str], List[str]]:
    if args.method == "cc" and not args.resolution:
        raise EnsembleError("--method cc needs --resolution")
    if args.batch < 1:
        raise EnsembleError("--batch: at least 1 model per ChimeraX run")
    kept, skipped = [], []
    for model in models:
        slot = pdbtools.empty_slot(model)
        if slot:  # an unfilled slot of a model check's filter: no model to score
            skipped.append(f"skipped {os.path.basename(model)}: {slot} (an empty slot of a model check, "
                           "no model)")
        else:
            kept.append(model)
    if not kept:
        return [], {}, skipped
    scored, failed, notes = score_models(kept, args.map, args.method, resolution=args.resolution,
                                         radius=args.radius, keep_h=args.keep_h, chimerax=args.chimerax,
                                         options=args.qscore_options, batch=args.batch, column=args.q_column,
                                         work_dir=work, timeout=args.timeout)
    return scored, failed, skipped + notes


def _what(args) -> Tuple[str, str]:
    """(the score in words, its label in the report: Q or CC)."""
    if args.method == "qscore":
        return f"ChimeraX Q-score, {args.q_column.replace('q', 'Q', 1)}", "Q"
    return f"per-residue CC at {args.resolution:g} Å", "CC"


def run_score(args) -> int:
    """'mdatk map score'.  Every model is tried; one ChimeraX could not score is listed in the report and
    the command then ends with status 1, after writing the tables of the others."""
    groups = _groups(args)
    regions, reference, sort = _region_setup(args, groups)
    beside = args.output or args.regions_out or args.report or "x"  # x: the current folder
    work = args.work or os.path.join(os.path.dirname(os.path.abspath(beside)), "qscore")
    scored, failed, notes = _scored(args, args.models, work)
    if not scored and not failed:  # nothing was tried: each input is an empty slot of a model check
        raise EnsembleError("no model to score" + (": every input is an empty slot of a model check ("
                                                   + "; ".join(notes[:5]) + (" ..." if len(notes) > 5 else "")
                                                   + ")" if notes else ""))
    if not scored:
        raise EnsembleError("no model was scored: " + "; ".join(f"{os.path.basename(m)}: {why}"
                                                               for m, why in list(failed.items())[:5]))
    method = f"qscore:{args.q_column}" if args.method == "qscore" else "cc"
    res_rows, reg_rows = [], []
    for ms in scored:
        notes += [f"{os.path.basename(ms.model)}: {n}" for n in ms.notes]
        res_rows += residue_table_rows(ms.model, ms.scores, method, ms.q_rows)
        reg_rows += region_rows(ms.model, ms.scores, regions, reference, structure_termini(ms.structure),
                                args.keep_termini)
    order = rank_models(reg_rows, sort)
    if args.output:
        write_table(args.output, QSCORE_COLUMNS if args.method == "qscore" else RESIDUE_COLUMNS, res_rows)
    if args.regions_out:
        write_table(args.regions_out, REGION_COLUMNS,
                    [[_cell(r[c]) for c in REGION_COLUMNS] for r in reg_rows])
    what, label = _what(args)
    text = regions_report(reg_rows, order, regions, reference, args.keep_termini, sort, what, label)
    if failed:
        text += f"\n{len(failed)} model(s) not scored (not ranked):\n" + "".join(
            f"  {os.path.basename(m)}: {why}\n" for m, why in failed.items())
    if notes:
        text += "\n" + "\n".join(notes[:40]) + ("\n..." if len(notes) > 40 else "") + "\n"
    _emit(text, args.report)
    if failed and args.report:
        sys.stderr.write(f"mdatk map score: {len(failed)} model(s) not scored; see {args.report}\n")
    return 1 if failed else 0


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.4f}"
    return v


def run_regions(args) -> int:
    """'mdatk ensemble regions': rank models from per-residue tables written elsewhere."""
    groups = _groups(args)
    regions, reference, sort = _region_setup(args, groups)
    reg_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []  # tables ChimeraX stopped inside: their models failed
    for path in args.tables:
        for model, scores in read_residue_scores(path, args.column, skipped).items():
            reg_rows += region_rows(model, scores, regions, reference, key_termini(list(scores)),
                                    args.keep_termini)
    if not reg_rows:
        raise EnsembleError("no per-residue scores read" + (": " + "; ".join(skipped[:5]) if skipped else ""))
    order = rank_models(reg_rows, sort)
    if args.output:
        write_table(args.output, REGION_COLUMNS, [[_cell(r[c]) for c in REGION_COLUMNS] for r in reg_rows])
    what = "per-residue tables; termini from each chain's lowest and highest number"
    text = regions_report(reg_rows, order, regions, reference, args.keep_termini, sort, what)
    if skipped:
        text += f"\n{len(skipped)} table(s) not ranked:\n" + "".join(f"  {why}\n" for why in skipped)
    _emit(text, args.report)
    if skipped and args.report:
        sys.stderr.write(f"mdatk ensemble regions: {len(skipped)} table(s) not ranked; see {args.report}\n")
    return 1 if skipped else 0


def run_hybrids(args) -> int:
    groups = _groups(args)
    bodies = [parse_body(t, groups) for t in args.body]
    old, new = pdbtools.read_pdb(args.old), pdbtools.read_pdb(args.new)
    hybrids, members, notes = make_hybrids(old, new, bodies, args.match)
    if "{code}" not in args.names:
        raise EnsembleError("--names must contain {code}")
    os.makedirs(args.out_dir, exist_ok=True)
    lines = [f"{len(hybrids)} combinations of {', '.join(b.name for b in bodies)} "
             f"(o = {os.path.basename(args.old)}, n = {os.path.basename(args.new)})"]
    lines += [f"  {b.name}: {selection_text(b.selection)} ({len(members[b.name])} residues)" for b in bodies]
    lines += [f"  {n}" for n in notes]
    paths = {}
    for code, s in hybrids.items():
        path = os.path.join(args.out_dir, args.names.replace("{code}", code))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(pdbtools.write_pdb(s))
        paths[code] = path
        lines.append(f"  wrote {os.path.basename(path)} ({sum(len(r.atoms) for r in s.residues())} atoms)")
    if args.map:
        codes = list(paths)
        scored, failed, more = _scored(args, [paths[c] for c in codes],
                                       args.work or os.path.join(args.out_dir, "qscore"))
        if failed:  # a main effect needs every combination
            raise EnsembleError("hybrids not scored: " + "; ".join(f"{os.path.basename(m)}: {why}"
                                                                   for m, why in failed.items()))
        by_path = {ms.model: ms.scores for ms in scored}
        scores = {c: by_path[paths[c]] for c in codes}
        lines += [f"  {n}" for n in more]
        hyb_rows, body_rows = body_effects(bodies, scores, members)
        if args.scores_out:
            write_table(args.scores_out, HYBRID_COLUMNS + [b.name for b in bodies], hyb_rows)
        if args.effects_out:
            write_table(args.effects_out, EFFECT_COLUMNS, body_rows)
        what = _what(args)[0]
        lines += ["", f"scores ({what}; mean over all residues, and over each body's own):",
                  f"  {'hybrid':<10} {'whole':>8} " + " ".join(f"{b.name[:10]:>10}" for b in bodies)]
        for row in hyb_rows:
            lines.append(f"  {row[0]:<10} {row[1]:>8} " + " ".join(f"{v:>10}" for v in row[2:]))
        lines += ["", "main effects (mean of the n hybrids minus mean of the o hybrids):",
                  f"  {'body':<14} {'residues':>8} {'whole':>9} {'own':>9}"]
        for row in body_rows:
            lines.append(f"  {row[0]:<14} {row[1]:>8} {row[2]:>9} {row[5]:>9}")
    _emit("\n".join(lines) + "\n", args.report)
    return 0
