"""Rigid frames of multi-body models: fit on one body, measure the others (``mdatk frame``).

Models built around pre-placed parts (helices given to rna_denovo with ``-s``, a rigid-body fit into a map)
inherit a frame: where the bodies sit relative to each other.  Two questions follow, and both are answered
by superposing on one body (the *fit* selection) and measuring what that transform does to the others (the
*measure* bodies), with the same atoms of the same residues, matched by chain, residue number and
insertion code:

``check``
    Did every model keep the reference's placement?  Each model is superposed onto the reference on the
    fit body.  Its fit RMSD says whether it contains that body at all (above ``valid_fit``, 1 Å by default,
    it does not, and its other numbers mean nothing: such models are reported but left out of the
    statistics); the RMSD of each measure body under that transform says how far the body moved relative to
    the fit body.  With the 95th percentile of every body below ``same_frame`` (1 Å), the models share one
    frame: the relative placement was fixed before sampling.  Otherwise the placement was explored.

``reframe``
    The reference is re-fitted (a new map, a sharpened map): can the existing models simply be moved into
    the new frame?  The old reference is superposed onto the new one on the fit body; if every measure body
    is then carried to its new place (RMSD within ``tol``, 1 Å), one rigid transform describes the re-fit,
    and applying it to every model moves them into the new frame.  Otherwise the bodies moved relative to
    each other, and the models have to be built again on the new reference.  A caveat: models sampled with
    the map in the score (Rosetta's ``elec_dens_fast``) were also drawn towards the old placement, so moving
    them reproduces their geometry in the new frame, not what sampling in that frame would have built.

Selections are chain-qualified residue ranges (``A:1-16,B:17-44``, ``C`` for a whole chain, ``1-44`` in any
chain) or residue groups (``group:Aptamer``).  When the models are numbered differently from the reference,
``model_map`` pairs the residues (``C:1-29=B:45-73``: model residues on the right, reference on the left).
"""

from __future__ import annotations

import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pdbtools
from .density import RigidTransform, rotation_angle
from .spatial import kabsch

DEFAULT_ATOMS = ("P", "C1'", "C4'")
MIN_ATOMS = 6
VALID_FIT = 1.0  # Å: a model whose fit body is further than this from the reference's does not contain it
SAME_FRAME = 1.0  # Å: 95th percentile of the measured bodies below this: one shared frame
TOLERANCE = 1.0  # Å: carried RMSD within this: one rigid transform describes a re-fit
HISTOGRAM = (0.5, 1.0, 2.0, 4.0, 8.0)

Range = Tuple[Optional[str], Optional[int], Optional[int]]  # chain (None: any), first, last (None: all)
AtomKey = Tuple[str, int, str, str]  # chain, residue number, insertion code, atom name


class FrameError(ValueError):
    pass


# ---------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------


def _g(group: Any, key: str) -> str:
    return str(group.get(key, "") if isinstance(group, dict) else getattr(group, key, "") or "")


_UNQUALIFIED = re.compile(r"^(-?\d+)(?:-(-?\d+))?$")
_GROUP_SELECTION = re.compile(r"^\s*(?:(?:chainID|segid|chain)\s+(\S+)\s+and\s+)?resid\s+(.+?)\s*$")


def _group_ranges(group: Any) -> List[Range]:
    residues = _g(group, "residues").strip()
    if residues:
        return parse_selection(residues)
    sel = _g(group, "selection")
    m = _GROUP_SELECTION.match(sel)
    if m:  # the selections MDATK writes for chain groups: 'chainID A and resid 1:40'
        out: List[Range] = []
        for item in m.group(2).split():
            mm = re.fullmatch(r"(-?\d+)(?::(-?\d+))?", item)
            if not mm:
                break
            a = int(mm.group(1))
            out.append((m.group(1), a, int(mm.group(2)) if mm.group(2) else a))
        else:
            return out
    raise FrameError(f"residue group {_g(group, 'name')!r} is defined by the selection {sel!r}, which cannot "
                     "be read as residue ranges; give its residues instead")


def parse_selection(spec: str, groups: Sequence[Any] = ()) -> List[Range]:
    """Residue ranges: ``'A:1-16,B:17-44'`` (mdatk.pdbtools.parse_ranges), ``'C'`` (a whole chain), ``'1-44'``
    (those numbers in any chain), ``'group:Aptamer'`` or a group's bare name."""
    by_name = {_g(g, "name"): g for g in groups}
    out: List[Range] = []
    for tok in re.split(r"[,;\s]+", (spec or "").strip()):
        if not tok:
            continue
        name = tok[6:] if tok.startswith("group:") else tok
        if name in by_name:
            out += _group_ranges(by_name[name])
            continue
        if tok.startswith("group:"):
            known = ", ".join(by_name) or "none"
            raise FrameError(f"unknown residue group {name!r} (defined: {known})")
        m = _UNQUALIFIED.match(tok)
        if m:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) is not None else a
            out.append((None, min(a, b), max(a, b)))
            continue
        try:
            out += pdbtools.parse_ranges(tok)
        except pdbtools.PDBError as exc:
            raise FrameError(str(exc)) from None
    return out


def selection_text(spec: str, groups: Sequence[Any] = ()) -> str:
    """The selection with groups replaced by their ranges (for scripts that run without the project)."""
    parts = []
    for chain, a, b in parse_selection(spec, groups):
        rng = "" if a is None else (f"{a}" if a == b else f"{a}-{b}")
        parts.append(rng if chain is None else (f"{chain}:{rng}" if rng else chain))
    return ",".join(parts)


@dataclass
class Body:
    name: str
    ranges: List[Range]


def parse_bodies(text: str, groups: Sequence[Any] = ()) -> List[Body]:
    """Measure bodies, one per line or separated by ';': ``Target=C:1-29``, ``group:Target`` or ``C:1-29``
    (named after the group or the ranges)."""
    out: List[Body] = []
    for item in re.split(r"[;\n]+", text or ""):
        item = item.strip()
        if not item:
            continue
        name, sep, spec = item.partition("=")
        if not sep:
            spec = item
            bare = item[6:] if item.startswith("group:") else item
            name = bare if bare in {_g(g, "name") for g in groups} else item
        name = re.sub(r"[^A-Za-z0-9._:-]+", "_", name.strip()) or f"body{len(out) + 1}"
        ranges = parse_selection(spec, groups)
        if not ranges:
            raise FrameError(f"body {name!r} selects no residues")
        out.append(Body(name, ranges))
    if len({b.name for b in out}) != len(out):
        raise FrameError("two measure bodies have the same name")
    return out


def bodies_text(text: str, groups: Sequence[Any] = ()) -> List[str]:
    """``NAME=RANGES`` of every body, groups replaced by their ranges."""
    out = []
    for b in parse_bodies(text, groups):
        parts = []
        for chain, a, c in b.ranges:
            rng = "" if a is None else (f"{a}" if a == c else f"{a}-{c}")
            parts.append(rng if chain is None else (f"{chain}:{rng}" if rng else chain))
        out.append(f"{b.name}={','.join(parts)}")
    return out


def in_selection(chain: str, resseq: int, ranges: Sequence[Range]) -> bool:
    for c, a, b in ranges:
        if (c is None or c == chain) and (a is None or a <= resseq <= b):
            return True
    return False


def parse_model_map(text: str) -> Dict[Tuple[str, int], Tuple[str, int]]:
    """``'A:1-44=A:1-44, C:1-29=B:45-73'`` (reference = model) -> {(model chain, number): (reference chain,
    number)}; the residues of each side pair up in order."""
    out: Dict[Tuple[str, int], Tuple[str, int]] = {}
    for item in re.split(r"[,;\s]+", (text or "").strip()):
        if not item:
            continue
        ref, sep, model = item.partition("=")
        if not sep:
            raise FrameError(f"model map item {item!r}: expected REFERENCE=MODEL, e.g. C:1-29=B:45-73")
        sides = []
        for side in (ref, model):
            rr = pdbtools.parse_ranges(side)
            if len(rr) != 1 or rr[0][1] is None:
                raise FrameError(f"model map item {item!r}: one chain-qualified range on each side")
            c, a, b = rr[0]
            step = 1 if b >= a else -1
            sides.append([(c, k) for k in range(a, b + step, step)])
        if len(sides[0]) != len(sides[1]):
            raise FrameError(f"model map item {item!r}: {len(sides[0])} reference and {len(sides[1])} model "
                             "residues")
        out.update({m: r for r, m in zip(*sides)})
    return out


# ---------------------------------------------------------------------------
# atoms and superposition
# ---------------------------------------------------------------------------


def atoms_of(source, atoms: Sequence[str] = DEFAULT_ATOMS,
             model_map: Optional[Dict[Tuple[str, int], Tuple[str, int]]] = None
             ) -> Tuple[Dict[AtomKey, np.ndarray], List[str]]:
    """{(chain, residue number, insertion code, atom name): xyz} of the named atoms (first alternate
    location, first model; old-style '*' read as a prime).  Residues are told apart by chain, so two chains
    numbered alike never mix; a residue that appears twice keeps its first copy.  Returns (atoms, notes)."""
    struct = source if isinstance(source, pdbtools.Structure) else pdbtools.read_pdb(source)
    notes = list(struct.notes)
    out: Dict[AtomKey, np.ndarray] = {}
    seen = set()
    wanted = set(atoms)
    for res in struct.residues():
        chain, num = res.chain, res.resseq
        if model_map and (chain, num) in model_map:
            chain, num = model_map[(chain, num)]
        rid = (chain, num, res.icode)
        if rid in seen:
            notes.append(f"residue {chain}:{num}{res.icode} appears twice; its first copy is used")
            continue
        seen.add(rid)
        for name, xyz in res.coords().items():
            if name in wanted:
                out[(chain, num, res.icode, name)] = np.asarray(xyz, dtype=np.float64)
    return out, notes


def keys_in(atoms: Dict[AtomKey, np.ndarray], ranges: Sequence[Range]) -> List[AtomKey]:
    return sorted(k for k in atoms if in_selection(k[0], k[1], ranges))


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt((d * d).sum() / max(1, len(d))))


def superpose(mobile: np.ndarray, target: np.ndarray) -> Tuple[RigidTransform, float]:
    """The proper rotation + translation taking ``mobile`` onto ``target`` (Kabsch) and the RMSD after it."""
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(mobile) < 3:
        raise FrameError(f"{len(mobile)} atoms cannot define a superposition")
    centred = mobile - mobile.mean(axis=0)
    if np.linalg.matrix_rank(centred, tol=1e-3) < 2:
        raise FrameError("the fit atoms lie on a line: the superposition is not defined")
    r, mc, rc = kabsch(mobile, target)
    t = RigidTransform(r, rc - r @ mc)
    return t, rmsd(t.apply(mobile), target)


def shared(ref: Dict[AtomKey, np.ndarray], other: Dict[AtomKey, np.ndarray],
           ranges: Sequence[Range]) -> List[AtomKey]:
    return [k for k in keys_in(ref, ranges) if k in other]


# ---------------------------------------------------------------------------
# check: one frame or several
# ---------------------------------------------------------------------------


@dataclass
class ModelRef:
    path: str  # file to read
    label: str  # as shown in tables (plan-relative path, or as given)
    run: str = ""
    tag: str = ""


@dataclass
class CheckRow:
    model: ModelRef
    n_fit: int = 0
    fit_rmsd: float = float("nan")
    angle: float = float("nan")  # of the superposition onto the reference (degrees)
    shift: float = float("nan")  # of the fit body's centre (Å)
    bodies: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    error: str = ""
    valid: bool = False


def measure(ref: Dict[AtomKey, np.ndarray], mdl: Dict[AtomKey, np.ndarray], fit: Sequence[Range],
            bodies: Sequence[Body], min_atoms: int = MIN_ATOMS) -> Tuple[RigidTransform, CheckRow]:
    """Superpose ``mdl`` on ``ref`` with the fit body's shared atoms and measure each body under it (the
    model is the mobile side).  Raises FrameError when a selection has fewer than ``min_atoms`` shared
    atoms."""
    fk = shared(ref, mdl, fit)
    if len(fk) < min_atoms:
        raise FrameError(f"{len(fk)} fit atoms shared with the reference (at least {min_atoms} needed)")
    mob = np.array([mdl[k] for k in fk])
    tgt = np.array([ref[k] for k in fk])
    t, fit_rmsd = superpose(mob, tgt)
    row = CheckRow(ModelRef("", ""), len(fk), fit_rmsd, rotation_angle(t.rotation),
                   float(np.linalg.norm(tgt.mean(axis=0) - mob.mean(axis=0))))
    for b in bodies:
        bk = shared(ref, mdl, b.ranges)
        if len(bk) < min_atoms:
            raise FrameError(f"{len(bk)} atoms of {b.name} shared with the reference (at least {min_atoms} "
                             "needed)")
        moved = t.apply(np.array([mdl[k] for k in bk]))
        row.bodies[b.name] = (len(bk), rmsd(moved, np.array([ref[k] for k in bk])))
    return t, row


def _check_one(job) -> CheckRow:
    model, ref, fit, bodies, atoms, model_map, min_atoms, valid_fit = job
    try:
        mdl, _notes = atoms_of(model.path, atoms, model_map)
        _t, row = measure(ref, mdl, fit, bodies, min_atoms)
    except (OSError, ValueError) as exc:  # FrameError and PDBError are ValueErrors
        return CheckRow(model, error=str(getattr(exc, "strerror", None) or exc))
    row.model = model
    row.valid = row.fit_rmsd <= valid_fit
    return row


def check_models(reference: str, models: Sequence[ModelRef], fit: Sequence[Range], bodies: Sequence[Body],
                 atoms: Sequence[str] = DEFAULT_ATOMS, min_atoms: int = MIN_ATOMS,
                 valid_fit: float = VALID_FIT, model_map=None,
                 jobs: int = 1) -> Tuple[List[CheckRow], List[str]]:
    """A row per model (also for models that could not be measured: their ``error`` says why)."""
    ref, notes = atoms_of(reference, atoms)
    fk = keys_in(ref, fit)
    if len(fk) < min_atoms:
        raise FrameError(f"the reference has {len(fk)} fit atoms ({', '.join(atoms)}) in the fit selection; "
                         f"at least {min_atoms} are needed")
    for b in bodies:
        n = len(keys_in(ref, b.ranges))
        if n < min_atoms:
            raise FrameError(f"the reference has {n} atoms of {b.name}; at least {min_atoms} are needed")
        overlap = set(keys_in(ref, b.ranges)) & set(fk)
        if overlap:
            notes.append(f"{b.name} shares {len(overlap)} atoms with the fit selection")
    jobs_list = [(m, ref, list(fit), list(bodies), tuple(atoms), model_map, min_atoms, valid_fit)
                 for m in models]
    if jobs > 1 and len(models) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(_check_one, jobs_list, chunksize=8))
    else:
        rows = [_check_one(j) for j in jobs_list]
    return rows, notes


def _stats(values: np.ndarray) -> str:
    return (f"median {np.median(values):.2f} Å  [p5 {np.percentile(values, 5):.2f}, p95 "
            f"{np.percentile(values, 95):.2f}]  max {values.max():.2f}")


@dataclass
class CheckSummary:
    verdict: str  # "one-frame", "moved", "fit-not-preserved", "nothing-measured"
    lines: List[str]


def summarise(rows: Sequence[CheckRow], bodies: Sequence[Body], valid_fit: float = VALID_FIT,
              same_frame: float = SAME_FRAME) -> CheckSummary:
    measured = [r for r in rows if not r.error]
    valid = [r for r in measured if r.valid]
    lines = [f"{len(rows)} models: {len(measured)} measured, {len(rows) - len(measured)} not measurable, "
             f"{len(measured) - len(valid)} without the fit body (fit RMSD > {valid_fit:g} Å)"]
    for r in [r for r in rows if r.error][:4]:
        lines.append(f"  not measured: {r.model.label}: {r.error}")
    if not measured:
        return CheckSummary("nothing-measured", lines + ["-> Nothing could be measured."])
    fits = np.array([r.fit_rmsd for r in measured])
    lines.append(f"fit body onto the reference: {_stats(fits)}")
    if not valid:
        lines += ["-> No model contains the reference's fit body at this numbering: nothing else is "
                  "meaningful. Check the selections (or the model map)."]
        return CheckSummary("fit-not-preserved", lines)
    one = True
    for b in bodies:
        vals = np.array([r.bodies[b.name][1] for r in valid])
        lines.append(f"{b.name} moved relative to the fit body: {_stats(vals)}")
        edges = (0.0,) + HISTOGRAM + (float("inf"),)
        labels = [f"<{HISTOGRAM[0]:g}"] + [f"{a:g}-{c:g}" for a, c in zip(HISTOGRAM, HISTOGRAM[1:])] + \
            [f">{HISTOGRAM[-1]:g}"]
        for k, lab in enumerate(labels):
            n = int(((vals >= edges[k]) & (vals < edges[k + 1])).sum())
            lines.append(f"    {lab:>7} Å {n:6d}  {'#' * int(round(50 * n / len(vals)))}")
        one = one and float(np.percentile(vals, 95)) < same_frame
    if np.median(fits) > valid_fit:
        lines.append(f"Most models do not contain the fit body (median fit RMSD {np.median(fits):.2f} Å): "
                     "the statistics above come from the few that do.")
    if one:
        lines.append(f"-> ONE FRAME: every measured body stays within {same_frame:g} Å (95th percentile) of "
                     "its place relative to the fit body; the relative placement was fixed before sampling, "
                     "and the models differ in what was built around it. Whether that placement is unique is "
                     "a question for the fit that made it (e.g. a rigid-body fit from many starts).")
        return CheckSummary("one-frame", lines)
    lines.append("-> THE FRAME MOVED: the relative placement of the bodies differs between models, so it was "
                 "sampled; its spread is a result (the modes are candidate placements), not noise.")
    return CheckSummary("moved", lines)


def _fmt(x: float, digits: int = 3) -> str:
    return "" if x != x else f"{x:.{digits}f}"


def write_check_table(path: str, rows: Sequence[CheckRow], bodies: Sequence[Body]) -> None:
    """One row per model, also for those that could not be measured (with the reason)."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        head = ["model", "run", "tag", "n_fit", "fit_rmsd", "valid", "angle", "shift"]
        for b in bodies:
            head += [f"n_{b.name}", f"rmsd_{b.name}"]
        w.writerow(head + ["error"])
        for r in rows:
            rec = [r.model.label, r.model.run, r.model.tag, r.n_fit or "", _fmt(r.fit_rmsd),
                   ("yes" if r.valid else "no") if not r.error else "", _fmt(r.angle, 2), _fmt(r.shift)]
            for b in bodies:
                n, v = r.bodies.get(b.name, (0, float("nan")))
                rec += [n or "", _fmt(v)]
            w.writerow(rec + [r.error])


# ---------------------------------------------------------------------------
# reframe: one rigid transform, or rebuild
# ---------------------------------------------------------------------------


@dataclass
class Reframe:
    transform: RigidTransform  # old frame -> new frame
    n_fit: int
    fit_rmsd: float
    carried: Dict[str, Tuple[int, float]]
    whole_rmsd: float  # every shared atom, without superposition
    centroid_shift: float
    angle: float
    verdict: str  # "rigid", "moved", "not-same-atoms"
    lines: List[str]


def reframe_decision(old: str, new: str, fit: Sequence[Range], bodies: Sequence[Body],
                     atoms: Sequence[str] = DEFAULT_ATOMS, min_atoms: int = MIN_ATOMS,
                     tol: float = TOLERANCE) -> Reframe:
    """Superpose the old reference onto the new one on the fit body and see where that leaves each body."""
    a, _n1 = atoms_of(old, atoms)
    b, _n2 = atoms_of(new, atoms)
    t, row = measure(b, a, fit, bodies, min_atoms)
    common = sorted(set(a) & set(b))
    pa, pb = np.array([a[k] for k in common]), np.array([b[k] for k in common])
    whole = rmsd(pa, pb)
    shift = float(np.linalg.norm(pb.mean(axis=0) - pa.mean(axis=0)))
    lines = [f"the re-fit moved the shared atoms by {whole:.2f} Å RMSD (centroid {shift:.2f} Å, rotation "
             f"{row.angle:.1f}°)",
             f"  fit body onto its re-fitted self: {row.fit_rmsd:.3f} Å ({row.n_fit} atoms)"]
    for name, (n, v) in row.bodies.items():
        lines.append(f"  {name} carried by that transform: {v:.3f} Å ({n} atoms; tolerance {tol:g} Å)")
    if row.fit_rmsd > tol:
        verdict = "not-same-atoms"
        lines.append("-> The fit body does not map onto itself: the two references do not hold the same "
                     "atoms at this numbering. Nothing else is meaningful; check the selections.")
    elif any(v > tol for _n, v in row.bodies.values()):
        verdict = "moved"
        lines.append("-> THE BODIES MOVED RELATIVE TO EACH OTHER: the re-fit changed the assembly itself, "
                     "not only its placement, so no single transform reproduces it. The models must be built "
                     "again on the new reference.")
    else:
        verdict = "rigid"
        lines.append("-> ONE RIGID TRANSFORM: the re-fit is a change of placement; applying it moves every "
                     "model into the new frame with its geometry unchanged.")
        if whole < 0.3:
            lines.append(f"   The re-fit barely moved ({whole:.2f} Å): there may be little to recover.")
    return Reframe(t, row.n_fit, row.fit_rmsd, row.bodies, whole, shift, row.angle, verdict, lines)


def _rotate_anisou(text: str, r: np.ndarray) -> str:
    """ANISOU columns 28-80 (as kept by pdbtools) with the tensor rotated: U' = R U R^T."""
    t = text.ljust(53)
    try:
        u11, u22, u33, u12, u13, u23 = (int(t[k:k + 7]) for k in range(1, 43, 7))
    except ValueError:
        return text
    u = np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]], dtype=np.float64)
    v = r @ u @ r.T
    vals = (v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 2])
    return t[0] + "".join(f"{int(round(x)):7d}" for x in vals) + t[43:]


def transform_structure(struct: pdbtools.Structure, t: RigidTransform) -> None:
    """Move every atom (and rotate its ANISOU tensor) in place; nothing else changes."""
    for res in struct.residues():
        for atom in res.atoms:
            x, y, z = t.apply(np.array(atom.xyz))
            tail = atom.tail.ljust(27)
            atom.tail = f"{tail[:3]}{x:8.3f}{y:8.3f}{z:8.3f}{tail[27:]}"
            if atom.anisou:
                atom.anisou = _rotate_anisou(atom.anisou, t.rotation)


def output_names(models: Sequence[ModelRef], pattern: str = "", decoy_pattern: str = "{run}/{tag}.pdb"
                 ) -> List[str]:
    """Unique output names: ``decoy_pattern`` (``{run}``, ``{tag}``) for decoys of a table, ``pattern`` with
    ``{n}`` (01, 02 ...) and ``{stem}`` for the others, else the path below the inputs' common folder; '-2',
    '-3' ... on clashes, so models of different runs with the same tag never overwrite each other."""
    plain = [m for m in models if not (m.run and m.tag)]
    common = ""
    if len(plain) > 1:
        common = os.path.commonpath([os.path.dirname(os.path.abspath(m.path)) for m in plain])
    names: List[str] = []
    k = 0
    for m in models:
        stem = os.path.splitext(os.path.basename(m.path))[0]
        if m.run and m.tag:
            name = decoy_pattern.replace("{run}", m.run).replace("{tag}", m.tag)
        else:
            k += 1
            if pattern:
                name = pattern.replace("{n}", f"{k:02d}").replace("{stem}", stem)
            elif common:
                name = os.path.relpath(os.path.abspath(m.path), common).replace(os.sep, "/")
            else:
                name = os.path.basename(m.path)
        base, ext = os.path.splitext(name)
        cand, j = name, 1
        while cand in names:
            j += 1
            cand = f"{base}-{j}{ext}"
        names.append(cand)
    return names


def apply_transform(models: Sequence[ModelRef], t: RigidTransform, out_dir: str, pattern: str = "",
                    decoy_pattern: str = "{run}/{tag}.pdb") -> List[Tuple[ModelRef, str, str]]:
    """Write every model moved by ``t`` into ``out_dir``.  Returns (model, output path, '' or the error)."""
    out = []
    for m, name in zip(models, output_names(models, pattern, decoy_pattern)):
        dst = os.path.join(out_dir, name)
        try:
            struct = pdbtools.read_pdb(m.path)
            transform_structure(struct, t)
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(pdbtools.write_pdb(struct, strip_conect=False))
            out.append((m, dst, ""))
        except (OSError, ValueError) as exc:
            out.append((m, "", str(getattr(exc, "strerror", None) or exc)))
    return out


# ---------------------------------------------------------------------------
# models from files and decoy tables
# ---------------------------------------------------------------------------


def gather_models(paths: Sequence[str], scores: str = "", root: str = "", maxima=None,
                  exclude: Sequence[str] = ()) -> Tuple[List[ModelRef], List[str], Any]:
    """Models given as files or folders (their PDB files, recursively) and the extracted decoys of a decoy
    table (``pdb`` relative to ``root``).  Returns (models, notes, table or None)."""
    from .rosetta import read_decoy_table

    skip = {os.path.abspath(p) for p in exclude}
    models: List[ModelRef] = []
    notes: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(os.path.join(r, f) for r, _d, fs in os.walk(p) for f in fs
                           if f.lower().endswith(".pdb"))
            models += [ModelRef(f, f) for f in found if os.path.abspath(f) not in skip]
        elif os.path.abspath(p) not in skip:
            models.append(ModelRef(p, p))
    table = None
    if scores:
        table = read_decoy_table(scores)
        rows, sel_notes = table.select(maxima or {})
        notes += sel_notes
        missing = 0
        for row in rows:
            path = table.pdb_path(row, root or None)
            if not path:
                missing += 1
                continue
            models.append(ModelRef(path, row.pdb, row.run, row.tag))
        if missing:
            notes.append(f"{missing} decoys of the table were not extracted (empty pdb column) and are "
                         "skipped")
        dens = [r for r in table.rows if (r.value("elec_dens_fast") or 0.0) != 0.0]
        if dens:
            notes.append("the decoys carry an elec_dens_fast score: they were sampled with the map in the "
                         "score, so they were drawn towards the old placement")
    return models, notes, table


# ---------------------------------------------------------------------------
# command line: mdatk frame check | reframe
# ---------------------------------------------------------------------------


def add_commands(sub) -> list:
    def common(p):
        p.add_argument("models", nargs="*", help="model PDB files or folders (searched for *.pdb)")
        p.add_argument("--scores", help="decoy table (run, tag, pdb, terms): its extracted models")
        p.add_argument("--root", default="", help="folder the table's pdb paths are relative to (default: "
                       "the table's step folder's parent)")
        p.add_argument("--max", action="append", default=[], metavar="TERM=VALUE",
                       help="only the table's decoys with TERM <= VALUE (repeatable)")
        p.add_argument("--fit", required=True, help="residues to superpose on, e.g. 'A:1-16,B:17-44' or "
                       "group:Aptamer")
        p.add_argument("--measure", action="append", required=True, metavar="[NAME=]RANGES",
                       help="a body measured under that superposition (repeatable), e.g. Target=C:1-29")
        p.add_argument("--atoms", default=",".join(DEFAULT_ATOMS), help="atom names used (default P,C1',C4')")
        p.add_argument("--min-atoms", type=int, default=MIN_ATOMS, help="fewest shared atoms per selection")
        p.add_argument("--model-map", default="", help="residue pairs when the models are numbered "
                       "differently: REFERENCE=MODEL ranges, e.g. C:1-29=B:45-73")
        p.add_argument("--project", "-p", dest="params", help="project file whose residue groups the "
                       "selections may name")
    c = sub.add_parser("check", help="did every model keep the reference's placement? Fit on one body, "
                                     "measure the others")
    c.add_argument("reference", help="reference PDB (the placed bodies)")
    common(c)
    c.add_argument("--valid-fit", type=float, default=VALID_FIT, help="fit RMSD above which a model does not "
                   "contain the fit body (default 1.0 Å)")
    c.add_argument("--same-frame", type=float, default=SAME_FRAME, help="95th percentile of the measured "
                   "bodies below which the models share one frame (default 1.0 Å)")
    c.add_argument("--jobs", type=int, default=1, help="parallel processes (default 1)")
    c.add_argument("--tsv", help="write a row per model (also the ones that could not be measured)")
    c.add_argument("--output", "-o", help="also write the summary to this file")
    c.add_argument("--strict", action="store_true",
                   help="exit with status 1 unless the models share one frame")
    r = sub.add_parser("reframe", help="can a re-fit of the reference be applied to the models? If one rigid "
                                       "transform suffices, move them")
    r.add_argument("old", help="reference the models were built on")
    r.add_argument("new", help="the re-fitted reference")
    common(r)
    r.add_argument("--tol", type=float, default=TOLERANCE, help="carried RMSD within which one rigid "
                   "transform describes the re-fit (default 1.0 Å)")
    r.add_argument("--out-dir", "-O", default="", help="folder for the moved models (decoys of a table as "
                   "<run>/<tag>.pdb)")
    r.add_argument("--pattern", default="", help="names of the moved model files ({n}: 01, 02 ...; {stem}: "
                   "the input's name); default: as below the inputs' common folder")
    r.add_argument("--decoy-pattern", default="{run}/{tag}.pdb", help="names of the moved decoys of the "
                   "table (default {run}/{tag}.pdb)")
    r.add_argument("--manifest", help="write a table: source, output, run, tag")
    r.add_argument("--scores-out", help="write the decoy table again with pdb pointing at the moved models")
    r.add_argument("--output", "-o", help="also write the summary to this file")
    return [c, r]


def _shown(path: str, root: str) -> str:
    """``path`` relative to ``root`` (the plan folder) when it lies inside it, else absolute."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return os.path.abspath(path) if rel.startswith("..") else rel.replace(os.sep, "/")


def _groups(args) -> list:
    if not getattr(args, "params", None):
        return []
    from .config import Project
    return list(Project.load(args.params).groups)


def run(args, stdout=None, stderr=None) -> int:
    """``mdatk frame check | reframe``.  Exit status: check 0 (1 with --strict unless one frame); reframe 0
    when one rigid transform suffices, 1 when the bodies moved relative to each other, 2 when the fit body
    does not map onto itself."""
    from .rosetta import parse_maxima

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    groups = _groups(args)
    fit = parse_selection(args.fit, groups)
    bodies = parse_bodies("\n".join(args.measure), groups)
    atoms = tuple(a for a in re.split(r"[,\s]+", args.atoms) if a)
    model_map = parse_model_map(args.model_map) if args.model_map else None
    exclude = [args.reference] if args.frame_command == "check" else [args.old, args.new]
    models, notes, table = gather_models(args.models, args.scores or "", args.root, parse_maxima(args.max),
                                         exclude)
    lines: List[str] = []
    if args.frame_command == "check":
        if not models:
            raise FrameError("no models: give PDB files, folders or a decoy table with extracted models")
        rows, ref_notes = check_models(args.reference, models, fit, bodies, atoms, args.min_atoms,
                                       args.valid_fit, model_map, max(1, args.jobs))
        summary = summarise(rows, bodies, args.valid_fit, args.same_frame)
        lines = [f"reference {args.reference}: fit on {args.fit}, measured "
                 + ", ".join(b.name for b in bodies) + f"; atoms {' '.join(atoms)}"]
        lines += [f"note: {n}" for n in notes + ref_notes] + summary.lines
        if args.tsv:
            write_check_table(args.tsv, rows, bodies)
            lines.append(f"wrote {len(rows)} rows to {args.tsv}")
        code = 1 if args.strict and summary.verdict != "one-frame" else 0
    else:
        res = reframe_decision(args.old, args.new, fit, bodies, atoms, args.min_atoms, args.tol)
        lines = [f"old {args.old} -> new {args.new}: fit on {args.fit}; atoms {' '.join(atoms)}"]
        lines += [f"note: {n}" for n in notes] + res.lines
        code = {"rigid": 0, "moved": 1, "not-same-atoms": 2}[res.verdict]
        if res.verdict == "rigid":
            if models and not args.out_dir:
                raise FrameError("models given but no --out-dir for the moved models")
            if models and any("elec_dens_fast" in n for n in notes):
                lines.append("   Caveat: these models were sampled with the map in the score; moved rigidly "
                             "they keep the loops that the old placement drew them into.")
            written = apply_transform(models, res.transform, args.out_dir, args.pattern, args.decoy_pattern) \
                if models else []
            ok = [w for w in written if not w[2]]
            lines.append(f"wrote {len(ok)} moved models" + (f" to {args.out_dir}" if ok else ""))
            for m, _dst, why in [w for w in written if w[2]][:4]:
                lines.append(f"  not written: {m.label}: {why}")
            root = table.root(args.root or None) if table is not None else os.getcwd()
            if args.manifest:
                with open(args.manifest, "w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                    w.writerow(["source", "output", "run", "tag", "error"])
                    for m, dst, why in written:
                        w.writerow([m.label, _shown(dst, root) if dst else "", m.run, m.tag, why])
            if args.scores_out and table is not None:
                moved = {(m.run, m.tag): _shown(dst, root) for m, dst, why in written if dst and m.run}
                with open(args.scores_out, "w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                    w.writerow(table.columns if "pdb" in table.columns else table.columns + ["pdb"])
                    for row in table.rows:
                        d = dict(row.values, run=row.run, tag=row.tag, pdb=moved.get((row.run, row.tag), ""))
                        w.writerow([d.get(c, "") for c in (table.columns if "pdb" in table.columns
                                                           else table.columns + ["pdb"])])
            if len(ok) < len(written):
                code = 1
        elif models and res.verdict != "rigid":
            lines.append("No model was moved.")
    text = "\n".join(lines) + "\n"
    out.write(text)
    if getattr(args, "output", None):
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    if code and args.frame_command == "reframe":
        print({1: "ERROR: the bodies moved relative to each other (or a model could not be written)",
               2: "ERROR: the fit body does not map onto itself"}[code], file=err)
    return code
