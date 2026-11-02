"""Map-model scores per residue, regions ranked by them, and old/new body hybrids.

Scores
    ``qscore``: an external Q-score program (Pintilie et al., Nat. Methods 17, 328 (2020)), by default
    Phenix's ``phenix.qscore`` (cctbx's ``mmtbx.qscore``), run on a copy of the model with standard
    residue names: plain A / C / G / U for D and L residues alike, every nucleotide an ATOM record,
    hydrogens and Rosetta's virtual residues dropped.  Q-score samples the radial density profile around
    each atom, which does not depend on handedness, so the plain names change nothing it measures, while
    programs that look residues up in their library would otherwise skip or reject L names (0A, LG ...)
    and HETATM records.  The copy never leaves the score's work folder.  The command is a template
    (``{model}``, ``{map}``, ``{stem}``, ``{resolution}``); its per-atom or per-residue output is read
    tolerantly: any table in the files it writes or on its standard output with a Q column and residue
    columns (chain, residue number), else a PDB it writes with Q in the B-factor column.  The command
    name and its output format could not be checked against a Phenix installation here; adjust the
    template to the installed version.
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
Hybrids
    Every old/new combination of up to six bodies (residue ranges of an old and a re-fitted model), over the
    atoms both files have, coordinates copied verbatim; scored against a map, each body's main effect is the
    mean score of the hybrids with its new coordinates minus the mean of those with its old ones.
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import pdbtools
from .ensemble import (EnsembleError, Range, ResKey, in_selection, is_hydrogen, key_text, parse_selection,
                       residue_base, residue_key, selection_residues, selection_text, write_table)
from .pdbtools import Residue, Structure

QSCORE_COMMAND = "phenix.qscore {model} {map}"
MASK_RADIUS = 2.5  # Å around a residue's atoms for its correlation
MAX_BODIES = 6
_VIRTUAL = frozenset({"VRT", "XXX"})
_VIRTUAL_ATOMS = frozenset({"ORIG", "X", "Y", "Z"})
ResidueScores = Dict[ResKey, Tuple[str, float, int]]  # key -> (residue name, score, atoms / points)

_Q_COLS = ("q", "qscore", "q_score", "q-score", "qscore_mean", "q_mean", "mean_q", "qscore_per_residue",
           "q_residue", "q-residue", "score", "cc")
_CHAIN_COLS = ("chain", "chain_id", "chainid", "chn", "auth_asym_id", "label_asym_id", "asym_id")
_RESSEQ_COLS = ("resseq", "resnum", "resid", "resi", "res_num", "resno", "res_seq", "residue_number",
                "auth_seq_id", "seq_id", "residue_seq")
_ICODE_COLS = ("icode", "ins_code", "insertion_code", "pdbx_pdb_ins_code", "inscode")
_RESNAME_COLS = ("resname", "resn", "res_name", "comp_id", "auth_comp_id", "residue_name", "base")
_MODEL_COLS = ("model",)


def _is_virtual(res: Residue) -> bool:
    names = {a.name for a in res.atoms}
    return res.name.strip().upper() in _VIRTUAL or (bool(names) and names <= _VIRTUAL_ATOMS)


# --- the standard-name copy -------------------------------------------------------------------------------

def standard_copy(structure: Structure, keep_h: bool = False) -> Tuple[Structure, List[str]]:
    """A copy for chirality-blind scoring: plain nucleotide names (A C G U; DNA DA ...), nucleotides as ATOM,
    hydrogens (``keep_h`` keeps them) and virtual residues dropped; other residues unchanged.

    Private stand-in for the pdbtools 'standard' scheme of the model-check work; the names here no longer
    show the chirality, so the copy is for scoring only."""
    s = copy.deepcopy(structure)
    renamed = hydrogens = virtual = 0
    segments = []
    for seg in s.segments:
        kept = []
        for res in seg:
            if _is_virtual(res):
                virtual += 1
                continue
            info = pdbtools.parse_residue_name(res.name)
            base = residue_base(res)
            if base:  # every nucleotide as ATOM: scoring programs commonly skip HETATM as ligands
                name = ("D" + base) if info is not None and info.sugar == "dna" else base
                if name != res.name:
                    renamed += 1
                res.name = name
                for a in res.atoms:
                    a.record = "ATOM"
            if not keep_h:
                before = len(res.atoms)
                res.atoms = [a for a in res.atoms if not is_hydrogen(a)]
                hydrogens += before - len(res.atoms)
            if res.atoms:
                kept.append(res)
        if kept:
            segments.append(kept)
    s.segments = segments
    s.conect = []
    notes = []
    if renamed:
        notes.append(f"{renamed} residues given plain names for scoring (the copy no longer shows chirality)")
    if hydrogens:
        notes.append(f"{hydrogens} hydrogens left out")
    if virtual:
        notes.append(f"{virtual} virtual residues left out")
    return s, notes


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
            except ValueError:
                q = math.nan if vals[qcol].strip().lower() in ("nan", "na", "") else None
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


def scores_from_bfactors(structure: Structure) -> Optional[ResidueScores]:
    """Per-residue mean of the B-factor column when every value lies in [-1, 1] (Q written as B)."""
    out: ResidueScores = {}
    for res in structure.residues():
        vals = []
        for a in res.atoms:
            if is_hydrogen(a):
                continue
            try:
                b = float(a.tail[33:39])
            except ValueError:
                continue
            if not -1.0 <= b <= 1.0:
                return None
            vals.append(b)
        if vals:
            out[residue_key(res)] = (res.name, sum(vals) / len(vals), len(vals))
    return out or None


def _command(template: str, **values: str) -> str:
    """Placeholders replaced by shell-quoted values; other braces (e.g. in a ChimeraX command) stay."""
    text = template
    for k, v in values.items():
        text = text.replace("{" + k + "}", shlex.quote(v))
    return text


def run_qscore(model: str, map_path: str, work_dir: str, command: str = QSCORE_COMMAND,
               keep_h: bool = False, resolution: Optional[float] = None, timeout: Optional[float] = None
               ) -> Tuple[ResidueScores, List[str]]:
    """Run the Q-score program on the standard-name copy of ``model`` in ``work_dir`` and read its scores
    (keys are the model's own: the copy keeps chains and residue numbers)."""
    os.makedirs(work_dir, exist_ok=True)
    structure = pdbtools.read_pdb(model)
    std, notes = standard_copy(structure, keep_h)
    stem = os.path.splitext(os.path.basename(model))[0]
    copy_path = os.path.join(work_dir, f"{stem}_standard.pdb")
    with open(copy_path, "w", encoding="utf-8") as fh:
        fh.write(pdbtools.write_pdb(std))
    before = set(os.listdir(work_dir))
    cmd = _command(command, model=os.path.basename(copy_path), map=os.path.abspath(map_path), stem=stem,
                   resolution="" if resolution is None else f"{resolution:g}")
    try:
        proc = subprocess.run(cmd, shell=True, cwd=work_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise EnsembleError(f"the Q-score command did not finish within {timeout:g} s: {cmd}") from None
    log = os.path.join(work_dir, f"{stem}_qscore.log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"$ {cmd}\n{proc.stdout}\n{proc.stderr}")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-5:])
        raise EnsembleError(f"the Q-score command failed (exit {proc.returncode}): {cmd}\n{tail}")
    made = sorted(set(os.listdir(work_dir)) - before - {os.path.basename(log)})
    for name in made:  # a table first
        if name.lower().endswith((".csv", ".tsv", ".txt", ".dat", ".log")):
            with open(os.path.join(work_dir, name), encoding="utf-8", errors="replace") as fh:
                got = parse_score_table(fh.read())
            if got:
                notes.append(f"Q-scores read from {name}")
                return _one_model(got), notes
    got = parse_score_table(proc.stdout)
    if got:
        notes.append("Q-scores read from the program's output")
        return _one_model(got), notes
    for name in made:  # then a PDB with Q in the B-factors
        if name.lower().endswith((".pdb", ".ent")):
            try:
                scores = scores_from_bfactors(pdbtools.read_pdb(os.path.join(work_dir, name)))
            except pdbtools.PDBError:
                continue
            if scores:
                notes.append(f"Q-scores read from the B-factors of {name}")
                return scores, notes
    raise EnsembleError(f"no per-atom or per-residue Q-scores found in the output of: {cmd} (see {log}); "
                        "adjust the command so that it writes them")


def _one_model(tables: Dict[str, ResidueScores]) -> ResidueScores:
    out: ResidueScores = {}
    for scores in tables.values():
        out.update(scores)
    return out


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


def score_model(model: str, map_path: str, method: str = "cc", *, resolution: Optional[float] = None,
                radius: float = MASK_RADIUS, keep_h: bool = False, command: str = QSCORE_COMMAND,
                work_dir: str = "", timeout: Optional[float] = None
                ) -> Tuple[Structure, ResidueScores, List[str]]:
    """(structure, per-residue scores, notes) of one model by 'cc' or 'qscore'."""
    structure = pdbtools.read_pdb(model)
    if method == "qscore":
        work = work_dir or os.path.join(os.path.dirname(os.path.abspath(model)), "qscore")
        scores, notes = run_qscore(model, map_path, work, command, keep_h, resolution, timeout)
        return structure, scores, notes
    if method != "cc":
        raise EnsembleError(f"unknown scoring method {method!r} (cc or qscore)")
    from .density import load_map_cached
    return structure, residue_cc(structure, load_map_cached(map_path), resolution or 0.0, radius, keep_h), []


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
    region, field = (sort.split(":", 1)[1], "delta") if sort.startswith("delta:") else (sort, "mean")
    key = {}
    for r in rows:
        if r["region"] == region:
            v = r[field]
            key[r["model"]] = -v if v == v else math.inf
    models = list(dict.fromkeys(r["model"] for r in rows))
    if region not in {r["region"] for r in rows}:
        raise EnsembleError(f"cannot sort by {sort!r}: no region {region!r}")
    order = sorted(models, key=lambda m: (key.get(m, math.inf), models.index(m)))
    rank = {m: i + 1 for i, m in enumerate(order)}
    for r in rows:
        r["rank"] = rank[r["model"]]
    return order


def regions_report(rows: List[Dict[str, Any]], order: Sequence[str], regions: Sequence[Region],
                   reference: Optional[Sequence[Range]], keep_termini: bool, sort: str, what: str) -> str:
    names = [r.name for r in regions]
    cols = names + [f"d:{n}" for n in names] + ["reference", "whole"]
    by = {(r["model"], r["region"]): r for r in rows}
    out = [f"{len(order)} models ranked by {sort} ({what}); "
           f"{'chain termini kept' if keep_termini else 'chain termini left out'}"]
    for reg in regions:
        out.append(f"  {reg.name}: {selection_text(reg.selection)}")
    out.append(f"  reference: {selection_text(reference) if reference else 'the whole model'} minus the "
               "named regions")
    out.append("")
    out.append(f"{'rank':>4} {'model':<32} " + " ".join(f"{c[:10]:>10}" for c in cols) + "  notes")
    for i, m in enumerate(order, 1):
        cells, notes = [], []
        for c in cols:
            if c.startswith("d:"):
                v = by.get((m, c[2:]), {}).get("delta", math.nan)
                cells.append(f"{v:+10.3f}" if v == v else f"{'-':>10}")
            else:
                r = by.get((m, c), {})
                v = r.get("mean", math.nan)
                cells.append(f"{v:10.3f}" if v == v else f"{'-':>10}")
                if r.get("missing"):
                    notes.append(f"{c}: {r['missing']} missing")
                if r.get("nan"):
                    notes.append(f"{c}: {r['nan']} without a score")
        out.append(f"{i:4d} {os.path.basename(m)[:32]:<32} " + " ".join(cells) + ("  " + "; ".join(notes)
                                                                                   if notes else ""))
    if len(order) > 1:
        out.append("")
        for c in names + ["reference", "whole"]:
            v = [by[(m, c)]["mean"] for m in order if (m, c) in by]
            v = [x for x in v if x == x]
            if v:
                out.append(f"{c:<16} spread {min(v):+.3f} to {max(v):+.3f} (range {max(v) - min(v):.3f})")
    return "\n".join(out) + "\n"


def read_residue_scores(path: str, column: str = "") -> Dict[str, ResidueScores]:
    """Per-residue tables: MDATK's (model, chain, resseq, ...), or headed / headless CSV with the columns
    chain, residue number, base, Q (as written per decoy by Q-score scripts).  Keys are chain-qualified."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
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


def residue_table_rows(model: str, scores: ResidueScores, method: str) -> List[List[Any]]:
    rows = []
    for k in sorted(scores, key=lambda k: (k[0], k[1], k[2])):
        name, v, n = scores[k]
        rows.append([model, k[0], k[1], k[2], name, "nan" if v != v else f"{v:.4f}", n, method])
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

def add_score_arguments(p) -> None:
    p.add_argument("--map", required=True, help="density map (MRC / CCP4)")
    p.add_argument("--method", choices=("cc", "qscore"), default="cc",
                   help="cc: MDATK's per-residue correlation (no external program); qscore: the Q-score "
                        "program")
    p.add_argument("--resolution", type=float, help="map resolution (Å; needed by cc)")
    p.add_argument("--radius", type=float, default=MASK_RADIUS, help="cc: grid points this close to a "
                   "residue's atoms (Å)")
    p.add_argument("--command", default=QSCORE_COMMAND, help="qscore: command template with {model} {map} "
                   "{stem} {resolution} (run in the work folder)")
    p.add_argument("--work", default="", help="qscore: work folder (default: qscore/ beside the output)")
    p.add_argument("--timeout", type=float, help="qscore: seconds per model")
    p.add_argument("--keep-h", action="store_true", help="score hydrogens too")


def add_region_arguments(p) -> None:
    p.add_argument("--region", action="append", default=[], metavar="NAME=RANGES",
                   help="a named region, e.g. 'loops=B:25-34,C:58-63' (repeat for each)")
    p.add_argument("--reference-region", default="", help="reference ranges (default: the whole model); the "
                   "named regions are always removed from it")
    p.add_argument("--keep-termini", action="store_true", help="keep chain termini in the means")
    p.add_argument("--sort", default="", help="rank by a region ('NAME'), 'delta:NAME', 'reference' or "
                   "'whole' (default: the first region, else whole)")


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
    c.add_argument("--command", default=QSCORE_COMMAND)
    c.add_argument("--work", default="")
    c.add_argument("--timeout", type=float)
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
    regions = [parse_region(t, groups) for t in args.region]
    reference = parse_selection(args.reference_region, groups) if args.reference_region.strip() else None
    sort = args.sort or (regions[0].name if regions else "whole")
    return regions, reference, sort


def run_score(args) -> int:
    groups = _groups(args)
    regions, reference, sort = _region_setup(args, groups)
    if args.method == "cc" and not args.resolution:
        raise EnsembleError("--method cc needs --resolution")
    work = args.work or os.path.join(os.path.dirname(os.path.abspath(args.output or "x")), "qscore")
    res_rows, reg_rows, notes = [], [], []
    for model in args.models:
        slot = pdbtools.empty_slot(model)
        if slot:  # an unfilled slot of a model check's filter: no model to score
            notes.append(f"skipped {os.path.basename(model)}: {slot} (an empty slot of a model check, "
                         "no model)")
            continue
        structure, scores, more = score_model(model, args.map, args.method, resolution=args.resolution,
                                              radius=args.radius, keep_h=args.keep_h, command=args.command,
                                              work_dir=work, timeout=args.timeout)
        notes += [f"{os.path.basename(model)}: {n}" for n in more]
        res_rows += residue_table_rows(model, scores, args.method)
        reg_rows += region_rows(model, scores, regions, reference, structure_termini(structure),
                                args.keep_termini)
    order = rank_models(reg_rows, sort)
    if args.output:
        write_table(args.output, RESIDUE_COLUMNS, res_rows)
    if args.regions_out:
        write_table(args.regions_out, REGION_COLUMNS,
                    [[_cell(r[c]) for c in REGION_COLUMNS] for r in reg_rows])
    what = "Q-score" if args.method == "qscore" else f"per-residue CC at {args.resolution:g} Å"
    text = regions_report(reg_rows, order, regions, reference, args.keep_termini, sort, what)
    if notes:
        text += "\n" + "\n".join(notes[:20]) + "\n"
    _emit(text, args.report)
    return 0


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.4f}"
    return v


def run_regions(args) -> int:
    """'mdatk ensemble regions': rank models from per-residue tables written elsewhere."""
    groups = _groups(args)
    regions, reference, sort = _region_setup(args, groups)
    reg_rows = []
    for path in args.tables:
        for model, scores in read_residue_scores(path, args.column).items():
            reg_rows += region_rows(model, scores, regions, reference, key_termini(list(scores)),
                                    args.keep_termini)
    if not reg_rows:
        raise EnsembleError("no per-residue scores read")
    order = rank_models(reg_rows, sort)
    if args.output:
        write_table(args.output, REGION_COLUMNS, [[_cell(r[c]) for c in REGION_COLUMNS] for r in reg_rows])
    what = "per-residue tables; termini from each chain's lowest and highest number"
    _emit(regions_report(reg_rows, order, regions, reference, args.keep_termini, sort, what), args.report)
    return 0


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
        if args.method == "cc" and not args.resolution:
            raise EnsembleError("--method cc needs --resolution")
        scores = {}
        for code, path in paths.items():
            _s, sc, more = score_model(path, args.map, args.method, resolution=args.resolution,
                                       radius=args.radius, keep_h=args.keep_h, command=args.command,
                                       work_dir=args.work or os.path.join(args.out_dir, "qscore"),
                                       timeout=args.timeout)
            scores[code] = sc
        hyb_rows, body_rows = body_effects(bodies, scores, members)
        if args.scores_out:
            write_table(args.scores_out, HYBRID_COLUMNS + [b.name for b in bodies], hyb_rows)
        if args.effects_out:
            write_table(args.effects_out, EFFECT_COLUMNS, body_rows)
        what = "Q-score" if args.method == "qscore" else f"per-residue CC at {args.resolution:g} Å"
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
