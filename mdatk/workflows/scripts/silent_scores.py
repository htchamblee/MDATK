"""Rosetta silent files: score terms by column name, the decoy table (and the list of the extracted decoys),
score filters, the decoys kept for clustering and the extraction check.

Every silent file carries header lines ``SCORE: <term> <term> ... description``; the data lines after a header
hold the values in the same columns and end with the decoy's tag.  Terms are looked up by name in the header
that applies to each line, never by a fixed column index: an index points at another term as soon as the
score function emits one term more or less, and nothing would fail - the decoys would be filtered on the
wrong number.

The decoy table (TSV) has the columns ``run``, ``tag``, ``pdb`` (the extracted model, relative to the plan
folder; empty when the decoy was not extracted) and then every SCORE term by its name, one row per decoy.
A run is labelled ``rNN`` from MDATK's file names (``<project>_<step>_farfar2-rNN_<stamp>.out``), otherwise by
the file name; runs and tags together are unique, so the tags that every run reuses (``S_000001`` ...) never
collide.  With validity filters (:mod:`decoy_validity`) the table also gets their columns and a ``valid``
flag (1 / 0) for every decoy.

Standard library only: MDATK embeds this module in the script of the FARFAR2 models step, and
:mod:`mdatk.rosetta` uses it for ``mdatk rosetta scores``.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import sys

TABLE_COLUMNS = ("run", "tag", "pdb")
_MDATK_NAME = re.compile(r"^(?P<project>.+)_(?P<step>\d{2,})_(?P<tag>[A-Za-z0-9-]+)"
                         r"(?:_(?P<stamp>\d{8}-\d{4}))?(?:\.(?P<ext>[A-Za-z0-9]+))?$")
_RUN_SUFFIX = re.compile(r"(?:^|-)(r\d+)$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Silent:
    """The SCORE lines of one silent file.

    ``headers``: the distinct column lists (terms between ``SCORE:`` and ``description``), in order of
    appearance; ``rows``: ``(tag, {term: text}, header index)`` per decoy, in file order, the first line of a
    tag kept (``duplicates`` lists the tags seen again); ``problems``: what could not be read."""

    def __init__(self, path, label=""):
        self.path = path
        self.label = label or run_label(path)
        self.headers = []
        self.rows = []
        self.duplicates = []
        self.problems = []
        self.read = False

    @property
    def tags(self):
        return [r[0] for r in self.rows]

    def values(self):
        """{tag: {term: text}}"""
        return {tag: vals for tag, vals, _h in self.rows}


def read_silent(path, label=""):
    """Read the SCORE lines of a silent file (a missing or unreadable file gives a problem, no rows)."""
    out = Silent(path, label)
    header = None
    seen = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.startswith("SCORE:"):
                    continue
                f = line.split()
                if len(f) < 2:
                    out.problems.append(f"{path}:{lineno}: empty SCORE line")
                    continue
                if f[-1] == "description":
                    header = tuple(f[1:-1])
                    if header not in out.headers:
                        out.headers.append(header)
                    continue
                tag = f[-1]
                if header is None:
                    out.problems.append(f"{path}:{lineno}: SCORE line of {tag} before any header line")
                    continue
                values = f[1:-1]
                if tag in seen:
                    # keep the first: a later line with the same tag is reported, never silently swapped in
                    out.duplicates.append(tag)
                    out.problems.append(f"{path}:{lineno}: decoy {tag} appears again (the first is kept)")
                    continue
                seen.add(tag)
                if len(values) != len(header):
                    out.problems.append(f"{path}:{lineno}: {len(values)} values for the {len(header)} "
                                        f"columns of the header ({tag}): its scores are not used")
                    out.rows.append((tag, {}, out.headers.index(header)))
                    continue
                out.rows.append((tag, dict(zip(header, values)), out.headers.index(header)))
        out.read = True
    except OSError as exc:
        out.problems.append(f"{path}: cannot read ({exc.strerror or exc})")
    return out


def run_label(path):
    """``rNN`` for MDATK's run files (``..._farfar2-r03_<stamp>.out``, ``..._filtered-r03_...``), else the
    file name without its extension; characters other than letters, digits, '.', '_' and '-' become '-'."""
    name = os.path.basename(path)
    m = _MDATK_NAME.match(name)
    if m:
        r = _RUN_SUFFIX.search(m.group("tag"))
        if r:
            return r.group(1)
    stem = name[:-4] if name.lower().endswith(".out") else os.path.splitext(name)[0]
    return _UNSAFE.sub("-", stem).strip("-") or "run"


def unique_labels(paths):
    """Run labels for ``paths``, unique: ``rNN``; ``NN-rNN`` (step number first) when two steps' runs share a
    number; the parent folder's name in front of a plain name; then a counter."""
    base = [run_label(p) for p in paths]
    out = list(base)
    for i, p in enumerate(paths):
        if base.count(base[i]) < 2:
            continue
        m = _MDATK_NAME.match(os.path.basename(p))
        if m and _RUN_SUFFIX.search(m.group("tag")):
            out[i] = f"{m.group('step')}-{base[i]}"
        else:
            parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
            out[i] = _UNSAFE.sub("-", f"{parent}-{base[i]}").strip("-")
    seen = {}
    for i, label in enumerate(out):
        if out.count(label) > 1:
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                out[i] = f"{label}-{seen[label]}"
    return out


def read_runs(runs):
    """[Silent] for ``runs``: (label, path) pairs or bare paths (labelled with :func:`unique_labels`)."""
    runs = list(runs)
    if runs and isinstance(runs[0], str):
        runs = list(zip(unique_labels(runs), runs))
    return [read_silent(path, label) for label, path, *_rest in runs]


def terms_of(silents):
    """Every score term of the files, in order of first appearance."""
    seen = {}
    for s in silents:
        for header in s.headers:
            for t in header:
                seen.setdefault(t, None)
    return list(seen)


def number(text):
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN counts as unreadable


def passes(values, term, maximum):
    """True / False for a decoy's ``{term: text}``; None when the term is missing or not a number."""
    x = number(values.get(term))
    return None if x is None else x <= maximum


# score limits as (term, operator, value); the planner reads the text with mdatk.modelcheck.parse_score_limits
LIMIT_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
             ">": lambda a, b: a > b}


def limits_text(limits):
    return ", ".join(f"{t} {op} {x:g}" for t, op, x in limits)


def within(values, limits):
    """True / False for a decoy's ``{term: text}`` against every limit; None when a term has no number."""
    for term, op, x in limits:
        v = number(values.get(term))
        if v is None:
            return None
        if not LIMIT_OPS[op](v, x):
            return False
    return True


def missing_term(silents, term):
    """Problems for headers that lack ``term`` (the column a filter or an audit asks for)."""
    out = []
    for s in silents:
        for header in s.headers:
            if term not in header:
                out.append(f"{s.path}: the SCORE header has no column {term!r} (columns: {' '.join(header)})")
                break
    return out


# --- the decoy table --------------------------------------------------------------------------------------

def tsv_field(text):
    return str(text).replace("\t", " ").replace("\n", " ")


def write_table(out_path, silents, pdb_of=None, extra=None):
    """Write the decoy table; ``pdb_of(label, tag)`` gives the plan-relative PDB of a decoy ('' if none);
    ``extra``: (columns, {(label, tag): {column: text}}) appended after the score terms (the validity
    columns), empty for a decoy without a row.  Returns the number of rows."""
    terms = terms_of(silents)
    more, rows = extra or ((), {})
    more = [c for c in more if c not in TABLE_COLUMNS and c not in terms]
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(TABLE_COLUMNS + tuple(terms) + tuple(more)) + "\n")
        for s in silents:
            for tag, values, _h in s.rows:
                pdb = pdb_of(s.label, tag) if pdb_of else ""
                fields = [s.label, tag, pdb or ""] + [values.get(t, "") for t in terms]
                if more:
                    got = rows.get((s.label, tag), {})
                    fields += [got.get(c, "") for c in more]
                fh.write("\t".join(tsv_field(f) for f in fields) + "\n")
                n += 1
    return n


def write_list(out_path, silents, pdb_of):
    """Write the PDB file of every decoy that has one, one per line in the order of the decoy table (the list
    the steps that take the full set of decoys read).  Returns the number of files."""
    paths = [p for s in silents for tag, _v, _h in s.rows for p in [pdb_of(s.label, tag)] if p]
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(p + "\n" for p in paths))
    return len(paths)


def read_validity(path):
    """(columns after run and tag, {(run, tag): {column: text}}) of a validity table (TSV)."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines:
        raise ValueError(f"{path}: empty validity table")
    header = lines[0].split("\t")
    if header[:2] != ["run", "tag"] or "valid" not in header:
        raise ValueError(f"{path}: not a validity table (columns: {' '.join(header[:6])})")
    rows = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        f = ln.split("\t")
        f += [""] * (len(header) - len(f))
        d = dict(zip(header, f))
        rows[(d.pop("run"), d.pop("tag"))] = d
    return header[2:], rows


def percentile(sorted_values, q):
    """Nearest-rank percentile of sorted numbers (``q`` in 0..100)."""
    k = int(round(q / 100.0 * (len(sorted_values) - 1)))
    return sorted_values[min(len(sorted_values) - 1, max(0, k))]


def audit_lines(silents, terms=(), maximum=None):
    """The column audit: where each requested term sits in each header layout, how many decoys there are,
    and the percentiles (and, with ``maximum``, the decoys at or below it) of each term."""
    lines = []
    n = sum(len(s.rows) for s in silents)
    layouts = {}
    for s in silents:
        for h in s.headers:
            layouts.setdefault(h, []).append(s)
    lines.append(f"{n} decoys in {len(silents)} silent file(s), {len(layouts)} SCORE header layout(s)")
    for header, files in layouts.items():
        lines.append(f"  {len(header)} terms in {len(files)} file(s), e.g. {os.path.basename(files[0].path)}")
        for t in terms:
            where = f"column {header.index(t) + 1}" if t in header else "MISSING"
            lines.append(f"    {t}: {where}")
    for s in silents:
        if not s.headers and s.read:
            lines.append(f"  {s.path}: no SCORE header line")
    for t in terms:
        vals = sorted(x for s in silents for _tag, v, _h in s.rows
                      for x in [number(v.get(t))] if x is not None)
        if not vals:
            lines.append(f"{t}: no values")
            continue
        unread = n - len(vals)
        qs = "  ".join(f"p{q}={percentile(vals, q):.2f}" for q in (0, 10, 25, 50, 75, 90, 100))
        lines.append(f"{t}: {qs}" + (f"  ({unread} decoys without a value)" if unread else ""))
        if maximum is not None:
            keep = sum(1 for x in vals if x <= maximum)
            lines.append(f"  {t} <= {maximum:g}: {keep} pass, {len(vals) - keep} fail"
                         + (f", {unread} without a value (fail)" if unread else ""))
    return lines


# --- filters ----------------------------------------------------------------------------------------------

def filter_silent(src, dst, term, maximum):
    """Copy the decoys of ``src`` whose ``term`` is at most ``maximum`` into ``dst`` (with the file's header
    lines).  A decoy line is a line that ends with the decoy's tag, as Rosetta writes every line of a
    structure.  Decoys without a readable value fail.  Returns (kept, failed, without a value)."""
    s = read_silent(src)
    if not s.read:
        raise ValueError("; ".join(s.problems))
    bad = missing_term([s], term)
    if bad:
        raise ValueError(bad[0])
    verdict = {tag: passes(values, term, maximum) for tag, values, _h in s.rows}
    copy_decoys(src, dst, verdict, [t for t, v in verdict.items() if v])
    kept = sum(1 for v in verdict.values() if v)
    unread = sum(1 for v in verdict.values() if v is None)
    return kept, len(verdict) - kept - unread, unread


def copy_decoys(src, dst, tags, keep):
    """Copy ``src`` into ``dst`` without the lines of the decoys in ``tags`` that are not in ``keep``.  A
    decoy line is a line that ends with the decoy's tag, as Rosetta writes every line of a structure; the
    header lines stay."""
    drop = set(tags) - set(keep)
    tmp = dst + ".tmp"
    with open(src, encoding="utf-8", errors="replace") as fi, open(tmp, "w", encoding="utf-8") as fo:
        for line in fi:
            f = line.split()
            if f and f[-1] in drop:
                continue
            fo.write(line)
    os.replace(tmp, dst)


def keep_runs(runs, valid=None, limits=()):
    """Write each run's filtered silent file (``runs``: (label, silent file, filtered file)) with the decoys
    to cluster: those flagged valid in ``valid`` ({(run, tag): row}, None: every decoy) that are also within
    the score ``limits``.  Returns [(label, kept, invalid, outside the limits, without a value)]."""
    out = []
    for label, src, dst, *_rest in runs:
        s = read_silent(src, label)
        if not s.read:
            raise ValueError("; ".join(s.problems))
        for term, _op, _x in limits:
            bad = missing_term([s], term)
            if bad:
                raise ValueError(bad[0])
        keep, invalid, outside, unread = [], 0, 0, 0
        for tag, values, _h in s.rows:
            if valid is not None and str(valid.get((label, tag), {}).get("valid", "")).strip() != "1":
                invalid += 1
                continue
            ok = within(values, limits) if limits else True
            if ok:
                keep.append(tag)
            elif ok is None:
                unread += 1
            else:
                outside += 1
        copy_decoys(src, dst, s.tags, keep)
        out.append((label, len(keep), invalid, outside, unread))
    return out


def filter_ranking(path, term=None, maximum=None, limits=()):
    """Keep the lines ``score tag file`` of a ranking whose decoy (looked up in ``file``) passes ``term`` <=
    ``maximum`` and every one of ``limits`` ((term, operator, value)).  Returns (kept, failed, without a
    value)."""
    limits = list(limits) + ([(term, "<=", maximum)] if term else [])
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    cache = {}
    kept, failed, unread = [], 0, 0
    for ln in lines:
        f = ln.split()
        if len(f) < 3:
            continue
        tag, src = f[1], f[2]
        if src not in cache:
            s = read_silent(src)
            bad = [b for t, _op, _x in limits for b in missing_term([s], t)] if s.read else s.problems[:1]
            if bad:
                raise ValueError(bad[0])
            cache[src] = s.values()
        ok = within(cache[src].get(tag, {}), limits)
        if ok:
            kept.append(ln)
        elif ok is None:
            unread += 1
        else:
            failed += 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(ln + "\n" for ln in kept))
    return len(kept), failed, unread


# --- extracted models -------------------------------------------------------------------------------------

def check_extracted(silents, folder):
    """Compare the tags of every run with ``folder/<run>/<tag>.pdb``.  Returns (problems, counts) where
    counts is {label: (decoys, files)}; a decoy without its file or a file without a decoy is a problem."""
    problems, counts = [], {}
    for s in silents:
        sub = os.path.join(folder, s.label)
        files = {n[:-4] for n in os.listdir(sub) if n.endswith(".pdb")} if os.path.isdir(sub) else set()
        tags = set(s.tags)
        counts[s.label] = (len(tags), len(files))
        missing = sorted(tags - files)
        extra = sorted(files - tags)
        if missing:
            problems.append(f"run {s.label}: {len(missing)} of {len(tags)} decoys were not extracted into "
                            f"{sub}/ (e.g. {', '.join(missing[:4])})")
        if extra:
            problems.append(f"run {s.label}: {len(extra)} PDB files in {sub}/ have no SCORE line in "
                            f"{os.path.basename(s.path)} (e.g. {', '.join(extra[:4])})")
    return problems, counts


def ranked_models(models_txt, pattern, runs):
    """{(label, tag): model file} from a models list (lines 'N score tag silent-file') whose silent file is
    one of the runs' files (``runs``: (label, path, other paths of the same run ...))."""
    if not models_txt or not os.path.isfile(models_txt):
        return {}
    by_file = {}
    for label, *paths in runs:
        for p in paths:
            if p:
                by_file[os.path.normpath(p)] = label
    out = {}
    with open(models_txt, encoding="utf-8") as fh:
        for ln in fh:
            f = ln.split()
            if len(f) < 4:
                continue
            label = by_file.get(os.path.normpath(f[3]))
            if label is not None:
                out.setdefault((label, f[2]), pattern.replace("{N}", f[0]))
    return out


def pdb_finder(prefix, extracted="", models=None):
    """``pdb_of(label, tag)`` for :func:`write_table`: ``extracted/<run>/<tag>.pdb`` when it exists, else the
    ranked model file; paths relative to the current folder, written relative to the plan folder
    (``prefix`` is the step folder, '' without step folders)."""
    models = models or {}

    def pdb_of(label, tag):
        rel = ""
        if extracted:
            cand = posixpath.join(extracted, label, f"{tag}.pdb")
            if os.path.isfile(cand):
                rel = cand
        if not rel:
            rel = models.get((label, tag), "")
            if rel and not os.path.isfile(rel):
                rel = ""
        if not rel:
            return ""
        return posixpath.normpath(posixpath.join(prefix, rel)) if prefix else posixpath.normpath(rel)
    return pdb_of


# --- command line -----------------------------------------------------------------------------------------

def _report(problems, limit=10):
    for p in problems[:limit]:
        print(f"  {p}", file=sys.stderr)
    if len(problems) > limit:
        print(f"  ... and {len(problems) - limit} more", file=sys.stderr)


def main(argv, runs=None, prefix="", model_pattern=""):
    """``runs``: (label, silent file, other files of the run ...) as written into the step's script; without
    them the silent files are given on the command line."""
    p = argparse.ArgumentParser(prog="silent_scores",
                                description="Rosetta silent-file scores by column name.")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("table", help="write the decoy table (TSV) and the column audit")
    t.add_argument("output")
    t.add_argument("silent", nargs="*", help="silent files (default: the step's runs)")
    t.add_argument("--extracted", default="", help="folder holding <run>/<tag>.pdb")
    t.add_argument("--models", default="", help="list of ranked models ('N score tag silent-file')")
    t.add_argument("--term", action="append", default=[], help="term to audit (repeatable)")
    t.add_argument("--validity", default="", help="validity table: its columns are appended to the table")
    t.add_argument("--list", default="", help="also write the PDB file of every decoy that has one, one per "
                                              "line (relative to the plan folder); none is an error")
    f = sub.add_parser("filter", help="copy the decoys of silent files that pass a score cut")
    f.add_argument("files", nargs="+", help="SOURCE OUTPUT [SOURCE OUTPUT ...]")
    f.add_argument("--term", required=True)
    f.add_argument("--max", type=float, required=True)
    r = sub.add_parser("filter-ranking", help="drop the lines of a ranking whose decoy fails score limits")
    r.add_argument("ranking")
    r.add_argument("--term", help="a term with --max (term <= max)")
    r.add_argument("--max", type=float)
    k = sub.add_parser("keep", help="write the filtered silent file of every run of the step: the decoys to "
                                    "cluster (valid, and within the limits)")
    k.add_argument("--valid", default="", help="validity table: only decoys with valid 1 are kept")
    for c in (r, k):
        c.add_argument("--limit", nargs=3, action="append", default=[], metavar=("TERM", "OP", "VALUE"),
                       help="a score limit, e.g. --limit score '<=' -300 (repeatable)")
    c = sub.add_parser("check-extracted", help="every decoy of every run has its <run>/<tag>.pdb")
    c.add_argument("folder")
    c.add_argument("silent", nargs="*")
    a = p.parse_args(argv)
    if a.cmd is None:
        p.print_help(sys.stderr)
        return 2
    try:
        limits = []
        for term, op, value in getattr(a, "limit", []):
            if op not in LIMIT_OPS:
                print(f"ERROR: unknown operator {op!r} in a limit on {term} (use <=, <, >= or >)",
                      file=sys.stderr)
                return 2
            limits.append((term, op, float(value)))
        if a.cmd == "keep":
            valid = read_validity(a.valid)[1] if a.valid else None
            total = 0
            for label, kept, invalid, outside, unread in keep_runs([tuple(r) for r in (runs or [])], valid,
                                                                   limits):
                total += kept
                why = ([f"{invalid} invalid"] if valid is not None else []) + (
                    [f"{outside} outside {limits_text(limits)}"] if limits else []) + (
                    [f"{unread} without a value"] if unread else [])
                print(f"run {label}: {kept} decoys kept for clustering"
                      + (f" ({', '.join(why)})" if why else ""))
                if not kept:
                    print(f"WARNING: run {label} has no decoy left to cluster", file=sys.stderr)
            if not total:
                print("ERROR: no decoy of any run is left to cluster", file=sys.stderr)
                return 1
            return 0
        if a.cmd == "filter-ranking" and (not (a.term or limits) or (a.term and a.max is None)):
            print("ERROR: filter-ranking needs --term with --max, or --limit", file=sys.stderr)
            return 2
        if a.cmd in ("filter", "filter-ranking"):
            cut = limits_text(limits + ([(a.term, "<=", a.max)] if a.term else []))
            if a.cmd == "filter":
                if len(a.files) % 2:
                    print("ERROR: filter takes pairs of files: SOURCE OUTPUT ...", file=sys.stderr)
                    return 2
                jobs = list(zip(a.files[::2], a.files[1::2]))
            else:
                jobs = [(a.ranking, "")]
            total = 0
            for src, dst in jobs:
                if dst:
                    kept, failed, unread = filter_silent(src, dst, a.term, a.max)
                else:
                    kept, failed, unread = filter_ranking(src, a.term, a.max, limits)
                total += kept
                print(f"{cut}: {kept} kept, {failed} dropped"
                      + (f", {unread} dropped without a value" if unread else "")
                      + f" ({src}{' -> ' + dst if dst else ''})")
            if not total:
                print(f"ERROR: no decoy passes {cut}", file=sys.stderr)
                return 1
            return 0
        run_list = [(lab, path) for lab, path in zip(unique_labels(a.silent), a.silent)] if a.silent \
            else [tuple(r) for r in (runs or [])]
        silents = read_runs(run_list)
        problems = [q for s in silents for q in s.problems]
        if a.cmd == "check-extracted":
            bad, counts = check_extracted(silents, a.folder)
            for label, (want, have) in counts.items():
                print(f"run {label}: {have} of {want} decoys extracted")
            if bad:
                print("ERROR: the extracted models do not match the SCORE lines:", file=sys.stderr)
                _report(bad)
                return 1
            return 0
        models = ranked_models(a.models, model_pattern, run_list) if a.models and model_pattern else {}
        extra = read_validity(a.validity) if a.validity else None
        pdb_of = pdb_finder(prefix, a.extracted, models)
        n = write_table(a.output, silents, pdb_of, extra)
        for line in audit_lines(silents, a.term):
            print(line)
        print(f"wrote {n} decoys to {a.output}")
        if problems:
            print(f"{len(problems)} problem(s) reading the silent files:", file=sys.stderr)
            _report(problems)
        if a.list:
            listed = write_list(a.list, silents, pdb_of)
            print(f"listed {listed} PDB files in {a.list}")
            if not listed:
                print(f"ERROR: no decoy has a PDB file to list in {a.list}", file=sys.stderr)
                return 1
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
