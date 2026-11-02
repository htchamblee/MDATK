"""Validity of FARFAR2 decoys, judged on every decoy before the decoys are clustered.

A cut on a score before clustering reshapes the cluster populations by a quantity that need not track quality,
and a register census is a statement about those populations.  What may go before clustering is what makes a
decoy not a model of the construct at all; quality is judged afterwards, on the cluster centres.  Four
criteria, each judged on every decoy and counted in this order:

hands
    The sugar hand of every nucleotide from its geometry (:func:`mdatk.pdbtools.residue_hands`), against
    expected counts (``44 D / 29 L``) and / or expected ranges (L in ``C``, D in ``A, B``).  A residue
    measured against its expected hand fails, so does an epimerised, contradicting or flat sugar and, with
    counts, any other count or a sugar whose hand cannot be measured.
bases
    The base of every nucleotide (from its atoms, else its name: :func:`mdatk.modelcheck.residue_kind`)
    against the sequence at the residue's chain and number; a different base, a missing residue and a residue
    the sequence does not have all fail.
closure
    A score term looked up by name in the decoy's SCORE line (``linear_chainbreak <= 5``) and / or every
    O3'(i)-P(i+1) of consecutive nucleotides of a TER segment at most a distance
    (:func:`mdatk.modelcheck.check_backbone`, 2.0 Å); a missing value or an unmeasurable link fails.
pairs
    The defining pairs of a register, every one formed (Watson-Crick N1/N3 below 3.5 Å) or approached
    (formed, or N1-N3 below 6 Å with C1'-C1' 8-13 Å), as :mod:`mdatk.ensemble` counts them.  The planner
    reads the register with ``mdatk.ensemble.parse_register`` and writes the pairs and the distances into
    the settings, so the two cannot drift apart.

Every decoy gets a row (``run``, ``tag``, ``valid`` 1 / 0, the first criterion it fails and why, and the
columns of each criterion); the report counts the decoys left after each criterion in order.

Standard library only: the FARFAR2 models step copies MDATK's PDB tools, model checks and silent-file
scores in front of this module into the script it writes, so the machine that runs Rosetta needs
``python3`` and nothing else.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

try:
    from ... import modelcheck, pdbtools
    from . import silent_scores
except ImportError:  # copied into a generated script after those modules: they are this module
    pdbtools = modelcheck = silent_scores = sys.modules[__name__]

CRITERIA = ("hands", "bases", "closure", "pairs")
VALIDITY_ID_COLUMNS = ("run", "tag", "valid", "invalid_by", "invalid_reason")
_HAND_COUNT = re.compile(r"\s*(?:(\d+)\s*([DLdl])(?![A-Za-z])|([DLdl])\s*[=:]?\s*(\d+))\s*(?:[,;/&+]|and\b)?")


# --- settings ---------------------------------------------------------------------------------------------

def parse_hand_counts(text):
    """'44 D / 29 L', 'D=44, L=29', '29L' -> {'D': 44, 'L': 29} ({} for empty text); ValueError otherwise."""
    text = (text or "").strip()
    out = {}
    pos = 0
    while pos < len(text):
        m = _HAND_COUNT.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read the hand counts {text!r} (expected e.g. '44 D / 29 L')")
        number, hand = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        hand = hand.upper()
        if hand in out:
            raise ValueError(f"the hand counts {text!r} give {hand} twice")
        out[hand] = int(number)
        pos = m.end()
    return out


def whole_chains(text):
    """Residue ranges with 'C:*' read as the whole chain C (pdbtools ranges spell it 'C')."""
    return re.sub(r"(^|[\s,;])([^:\s,;]):\*(?=$|[\s,;])", r"\1\2", (text or "").strip())


def describe(settings):
    """One line per criterion that is switched on, in order."""
    lines = []
    h = settings.get("hands")
    if h:
        what = []
        if h.get("counts"):
            what.append(" / ".join(f"{n} {k}" for k, n in sorted(h["counts"].items())))
        if h.get("expect_l"):
            what.append(f"L in {h['expect_l']}")
        if h.get("expect_d"):
            what.append(f"D in {h['expect_d']}")
        lines.append("hands: " + "; ".join(what) + " (sugar geometry: C4', C3', C1')")
    b = settings.get("bases")
    if b:
        lines.append(f"bases: against the sequence ({len(b['reference'])} nt)")
    c = settings.get("closure")
    if c:
        what = []
        if c.get("term"):
            what.append(f"{c['term']} <= {c['max']:g}")
        if c.get("o3p") is not None:
            what.append(f"every O3'-P <= {c['o3p']:g} A")
        lines.append("closure: " + " and ".join(what))
    p = settings.get("pairs")
    if p:
        pairs = ", ".join(f"{_key(a)}-{_key(b)}" for a, b in p["pairs"])
        if p.get("mode") == "approached":
            how = (f"formed or approached (N1/N3 < {p['near_n']:g} A, C1'-C1' {p['near_c1'][0]:g}-"
                   f"{p['near_c1'][1]:g} A)")
        else:
            how = f"formed (N1/N3 < {p['cut']:g} A)"
        lines.append(f"pairs: register {p['name']} ({pairs}), every pair {how}")
    return lines


def _key(key):
    chain, num, icode = key
    return f"{chain}:{num}{icode}"


def _label(res):
    return f"{res.chain or '_'}:{res.resseq}{res.icode}"


def _listed(items, limit=4):
    return ", ".join(items[:limit]) + (f" and {len(items) - limit} more" if len(items) > limit else "")


# --- the criteria -----------------------------------------------------------------------------------------

def judge_hands(structure, spec):
    """(ok, columns, reason) of the sugar hands."""
    rows = pdbtools.residue_hands(structure, spec.get("expect_l") or None, spec.get("expect_d") or None)
    n = {"D": 0, "L": 0, "?": 0}
    for r in rows:
        n[r.call if r.call in ("D", "L") else "?"] += 1
    cols = {"hands_D": n["D"], "hands_L": n["L"], "hands_undetermined": n["?"]}
    why = []
    for hand, want in sorted((spec.get("counts") or {}).items()):
        if n[hand] != want:
            why.append(f"{n[hand]} {hand} where {want} are expected")
    for want in ("L", "D"):
        wrong = [r.residue for r in rows if r.expected == want and r.call and r.call != want]
        if wrong:
            why.append(f"{len(wrong)} expected {want} measured {'D' if want == 'L' else 'L'} "
                       f"({pdbtools.ranges_text(wrong)})")
    defects = [r.residue for r in rows if r.defect]
    if defects:
        why.append(f"{len(defects)} sugar(s) with an epimerised, contradicting or flat centre "
                   f"({pdbtools.ranges_text(defects)})")
    unmeasured = [r.residue for r in rows if not r.call and not r.defect
                  and (r.expected or spec.get("counts"))]
    if unmeasured:
        why.append(f"{len(unmeasured)} sugar(s) whose hand cannot be measured "
                   f"({pdbtools.ranges_text(unmeasured)})")
    return not why, cols, "; ".join(why)


def judge_bases(structure, spec):
    """(ok, columns, reason) of the bases against the sequence: [(chain or None, number, base)]."""
    exact, loose = {}, {}
    for chain, num, base in spec["reference"]:
        (loose if chain is None else exact)[(chain, int(num))] = str(base).upper().replace("T", "U")
    seen = set()
    other, extra, twice = [], [], []
    for res in structure.residues():
        kind, base = modelcheck.residue_kind(res)
        if kind != "nt":
            continue  # waters, ions and Rosetta's virtual residues are not part of the sequence
        key = (res.chain, res.resseq)
        want = exact.get(key) if not res.icode else None
        if want is None and not res.icode and (None, res.resseq) in loose:
            key = (None, res.resseq)
            want = loose[key]
        if want is None:
            extra.append(_label(res))
            continue
        if key in seen:
            twice.append(_label(res))
            continue
        seen.add(key)
        got = (base or "?").upper().replace("T", "U")
        if want != "N" and got != want:
            other.append(f"{_label(res)} {got} for {want}")
    missing = [f"{c or '*'}:{num}" for (c, num) in list(exact) + list(loose) if (c, num) not in seen]
    why = []
    if other:
        why.append(f"{len(other)} base(s) differ from the sequence ({_listed(other)})")
    if missing:
        why.append(f"{len(missing)} residue(s) of the sequence missing ({_listed(missing)})")
    if extra:
        why.append(f"{len(extra)} nucleotide(s) the sequence does not have ({_listed(extra)})")
    if twice:
        why.append(f"{len(twice)} residue(s) numbered twice ({_listed(twice)})")
    errors = len(other) + len(missing) + len(extra) + len(twice)
    return not errors, {"base_errors": errors}, "; ".join(why)


def judge_closure(structure, values, spec):
    """(ok, columns, reason) of the chain closure: a score term by name and / or the O3'-P links."""
    why = []
    cols = {}
    term = spec.get("term")
    if term:
        x = silent_scores.number((values or {}).get(term))
        if x is None:
            why.append(f"no value of {term}")
        elif x > spec["max"]:
            why.append(f"{term} {x:g} > {spec['max']:g}")
    if spec.get("o3p") is not None:
        if structure is None:
            why.append("O3'-P not measured")
            cols["o3p_max"] = ""
        else:
            bb = modelcheck.check_backbone(structure, link_max=spec["o3p"])
            measured = [k for k in bb.links if k.distance is not None]
            longest = max((k.distance for k in measured), default=None)
            cols["o3p_max"] = "" if longest is None else f"{longest:.3f}"
            broken = [k for k in measured if k.distance > spec["o3p"]]
            unmeasured = [k for k in bb.links if k.distance is None]
            if broken:
                why.append(f"{len(broken)} O3'-P link(s) longer than {spec['o3p']:g} A ("
                           + _listed([f"{k.first}-{k.second} {k.distance:.2f}" for k in broken]) + ")")
            if unmeasured:
                why.append(f"{len(unmeasured)} O3'-P link(s) not measurable ("
                           + _listed([f"{k.first}-{k.second} {k.status}" for k in unmeasured]) + ")")
    return not why, cols, "; ".join(why)


def judge_pairs(structure, spec):
    """(ok, columns, reason) of a register's defining pairs, measured as :func:`mdatk.ensemble.scan_decoy`
    does: Watson-Crick N1/N3 below ``cut`` is formed; below ``near_n`` with C1'-C1' in ``near_c1`` is
    approached."""
    keyed = {}
    for res in structure.residues():
        keyed.setdefault((res.chain, res.resseq, res.icode), res)
    formed = approached = 0
    states, why = [], []
    lo, hi = spec["near_c1"]
    for a, b in spec["pairs"]:
        ra, rb = keyed.get(tuple(a)), keyed.get(tuple(b))
        if ra is None or rb is None:
            why.append(f"no residue {_key(a if ra is None else b)}")
            states.append("missing")
            continue
        ka, ba = modelcheck.residue_kind(ra)
        kb, bb = modelcheck.residue_kind(rb)
        if ka != "nt" or kb != "nt":
            why.append(f"{_label(ra if ka != 'nt' else rb)} is not a nucleotide")
            states.append("missing")
            continue
        ca, cb = ra.coords(), rb.coords()
        wa, wb = ca.get(spec["wc_atom"].get(ba or "", "")), cb.get(spec["wc_atom"].get(bb or "", ""))
        if wa is None or wb is None:
            why.append(f"no N1/N3 atom in {_label(ra)}-{_label(rb)}")
            states.append("missing")
            continue
        wc = math.dist(wa, wb)
        c1 = math.dist(ca["C1'"], cb["C1'"]) if "C1'" in ca and "C1'" in cb else None
        if wc < spec["cut"]:
            formed += 1
            approached += 1
            states.append("formed")
        elif wc < spec["near_n"] and c1 is not None and lo <= c1 <= hi:
            approached += 1
            states.append("approached")
        else:
            states.append("apart")
    n = len(spec["pairs"])
    need = approached if spec.get("mode") == "approached" else formed
    if need < n and not why:
        word = "formed or approached" if spec.get("mode") == "approached" else "formed"
        apart = [f"{_key(a)}-{_key(b)}" for (a, b), st in zip(spec["pairs"], states)
                 if st == "apart" or (st == "approached" and spec.get("mode") != "approached")]
        why.append(f"{need} of {n} pairs {word} (not: {_listed(apart)})")
    cols = {"pairs_formed": formed, "pairs_approached": approached}
    return not why and need == n, cols, "; ".join(why)


def judge_decoy(path, values, settings):
    """The row of one decoy: ``valid``, ``invalid_by`` (the first criterion it fails, in order),
    ``invalid_reason`` and every criterion's columns with ``<criterion>_ok`` (1 / 0)."""
    row = {"valid": 1, "invalid_by": "", "invalid_reason": ""}
    try:
        structure = pdbtools.read_pdb(path)
        unread = ""
    except (OSError, pdbtools.PDBError) as exc:
        structure, unread = None, f"cannot read the model ({exc})"
    for name in CRITERIA:
        spec = settings.get(name)
        if not spec:
            continue
        if structure is None and not (name == "closure" and spec.get("o3p") is None):
            ok, cols, why = False, {}, unread
        elif name == "hands":
            ok, cols, why = judge_hands(structure, spec)
        elif name == "bases":
            ok, cols, why = judge_bases(structure, spec)
        elif name == "closure":
            ok, cols, why = judge_closure(structure, values, spec)
        else:
            ok, cols, why = judge_pairs(structure, spec)
        row.update(cols)
        row[f"{name}_ok"] = 1 if ok else 0
        row[f"_{name}_why"] = why  # not a column: the report quotes it
        if not ok and not row["invalid_by"]:
            row["invalid_by"], row["invalid_reason"] = name, why
    row["valid"] = 0 if row["invalid_by"] else 1
    return row


def columns_of(settings):
    """The validity columns of the switched-on criteria, in order (after run and tag)."""
    out = list(VALIDITY_ID_COLUMNS[2:])
    if settings.get("hands"):
        out += ["hands_D", "hands_L", "hands_undetermined", "hands_ok"]
    if settings.get("bases"):
        out += ["base_errors", "bases_ok"]
    if settings.get("closure"):
        out += (["o3p_max"] if settings["closure"].get("o3p") is not None else []) + ["closure_ok"]
    if settings.get("pairs"):
        out += ["pairs_formed", "pairs_approached", "pairs_ok"]
    return out


def enabled(settings):
    return [c for c in CRITERIA if settings.get(c)]


def cascade(rows, settings):
    """[(label, decoys left)]: all decoys, then after each switched-on criterion in order."""
    alive = list(rows)
    out = [("decoys", len(alive))]
    for name in enabled(settings):
        alive = [r for r in alive if r.get(f"{name}_ok")]
        out.append((f"after {name}", len(alive)))
    return out


def report_lines(rows, settings):
    """The validity report: the criteria, the counts after each in order, every criterion on its own, and the
    valid decoys of each run."""
    lines = ["Validity filters (before clustering), applied in this order:"]
    lines += [f"  {k}. {d}" for k, d in enumerate(describe(settings), 1)]
    lines.append("")
    steps = cascade(rows, settings)
    prev = None
    for label, n in steps:
        lost = f"   ({prev - n} fail)" if prev is not None and prev != n else ""
        lines.append(f"  {label:16s} {n:7d}{lost}")
        prev = n
    valid = sum(1 for r in rows if r["valid"])
    lines += ["", f"{valid} of {len(rows)} decoys are valid: only these are clustered", "",
              "each criterion on every decoy (a decoy can fail several):"]
    for name in enabled(settings):
        bad = [r for r in rows if not r.get(f"{name}_ok")]
        eg = f"   e.g. {bad[0]['run']}/{bad[0]['tag']}: {bad[0].get(f'_{name}_why', '')}" if bad else ""
        lines.append(f"  {name:8s} {len(bad):7d} fail{eg}")
    by_run = {}
    for r in rows:
        k = by_run.setdefault(r["run"], [0, 0])
        k[0] += r["valid"]
        k[1] += 1
    if by_run:
        lines += ["", "valid decoys per run: "
                  + ", ".join(f"{lab} {v}/{n}" for lab, (v, n) in by_run.items())]
    empty = [lab for lab, (v, _n) in by_run.items() if not v]
    if empty:
        lines.append(f"runs without a valid decoy: {', '.join(empty)}")
    return lines


def write_rows(path, rows, settings):
    columns = list(VALIDITY_ID_COLUMNS[:2]) + columns_of(settings)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(silent_scores.tsv_field(r.get(c, "")) for c in columns) + "\n")
    return columns


def judge_runs(silents, settings, extracted):
    """A row per decoy of every run (``extracted/<run>/<tag>.pdb``), in file order."""
    rows = []
    for s in silents:
        for tag, values, _h in s.rows:
            row = judge_decoy(os.path.join(extracted, s.label, f"{tag}.pdb"), values, settings)
            row["run"], row["tag"] = s.label, tag
            rows.append(row)
    return rows


# --- command line -----------------------------------------------------------------------------------------

def validity_main(argv, runs=None, settings=None, extracted="models"):
    """``runs``: (label, silent file, ...) as written into the step's script; ``settings``: the criteria."""
    p = argparse.ArgumentParser(prog="decoy_validity",
                                description="Validity of FARFAR2 decoys before clustering.")
    sub = p.add_subparsers(dest="cmd")
    j = sub.add_parser("judge", help="judge every decoy; write a row per decoy and the report")
    j.add_argument("output", help="TSV: run, tag, valid, the first criterion failed and why, every "
                   "criterion's columns")
    j.add_argument("--report", default="", help="also write the report to this file")
    j.add_argument("--extracted", default=extracted, help="folder holding <run>/<tag>.pdb")
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help(sys.stderr)
        return 2
    settings = settings or {}
    if not enabled(settings):
        print("ERROR: no validity criterion is switched on", file=sys.stderr)
        return 2
    silents = silent_scores.read_runs([tuple(r) for r in (runs or [])])
    problems = [q for s in silents for q in s.problems]
    term = (settings.get("closure") or {}).get("term")
    if term:
        bad = silent_scores.missing_term(silents, term)
        if bad:
            print(f"ERROR: {bad[0]}", file=sys.stderr)
            return 1
    rows = judge_runs(silents, settings, a.extracted)
    write_rows(a.output, rows, settings)
    lines = report_lines(rows, settings)
    if problems:
        lines += ["", f"{len(problems)} problem(s) reading the silent files:"]
        lines += [f"  {q}" for q in problems[:20]]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if a.report:
        with open(a.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    if not any(r["valid"] for r in rows):
        print(f"ERROR: none of the {len(rows)} decoys is valid - nothing to cluster", file=sys.stderr)
        return 1
    return 0
