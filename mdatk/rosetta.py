"""Rosetta FARFAR2 bookkeeping: the decoy table, the silent-file score audit and the run audit.

Decoy table
    The FARFAR2 models step writes one row per decoy (:mod:`mdatk.workflows.scripts.silent_scores`):
    ``run``, ``tag``, ``pdb`` (plan-relative, empty when not extracted) and every SCORE term by name.
    :func:`read_decoy_table` reads it (TSV or CSV, extra columns allowed) for the steps and commands that
    take ``--scores``.

Score audit (``mdatk rosetta scores``)
    Where each term really sits in each SCORE header, how many decoys pass a cut, and whether the
    extracted models (``models/<run>/<tag>.pdb``) and the silent files describe the same decoys.

Run audit (``mdatk rosetta audit``)
    What each rna_denovo run was actually given.  The authoritative record is the command line Rosetta
    logs (``core.init: command: ...``); flags files named with ``@file`` (MDATK passes every flag that way)
    are expanded recursively, ``::`` is read as ``:``, and short and long spellings of a flag
    (``-s`` / ``-in:file:s``) are one flag.  Input files the flags name are fingerprinted by content, since
    two runs can name the same path and still have read different files (each folder its own copy of a
    map).  Runs are grouped into conditions (folders); conditions can be compared in pairs, where only the
    pairing flags and the secondary structure may differ (``--controlled`` names the residues whose
    pairing is the variable under test).  Fully expanded command lines are the normal case; ``@flags``
    files are read as well.

    Which Rosetta built a run is not answerable from its flags: the provenance comes from the lines Rosetta
    writes before the command line (``core.init: Rosetta version: <package> r<revision> <version> <commit>
    <url> <date>``, ``core.init: Rosetta extras: [...]``, the ``-version`` block, commit and checkout lines),
    kept raw and parsed tolerantly.  Runs compared with each other that were built with different versions
    are flagged.  The random seed is a run's identity: ``-constant_seed`` with ``-jran`` (and
    ``-seed_offset``), or the ``seed mode, seed=... real_seed=...`` line Rosetta logs after the command line,
    is reported for every run and never normalised away, and two runs (or two starts of one run) with the
    same seed under ``-constant_seed`` and the same inputs are reported as duplicates (an error: they write
    the same decoys).
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .workflows.naming import SUPERSEDED, parse_name
from .workflows.scripts import silent_scores

# ---------------------------------------------------------------------------
# the decoy table
# ---------------------------------------------------------------------------

TABLE_COLUMNS = silent_scores.TABLE_COLUMNS


@dataclass
class DecoyRow:
    run: str
    tag: str
    pdb: str  # plan-relative ('' when the decoy was not extracted)
    values: Dict[str, str] = field(default_factory=dict)  # every other column by name

    @property
    def key(self) -> str:
        return f"{self.run}/{self.tag}"

    def value(self, term: str) -> Optional[float]:
        return silent_scores.number(self.values.get(term))


@dataclass
class DecoyTable:
    path: str
    columns: List[str]
    rows: List[DecoyRow]

    @property
    def terms(self) -> List[str]:
        """The columns after run / tag / pdb (the score terms, and any extra columns)."""
        return [c for c in self.columns if c not in TABLE_COLUMNS]

    def select(self, maxima: Optional[Dict[str, float]] = None) -> Tuple[List[DecoyRow], List[str]]:
        """Rows whose terms are at most the given maxima; a row without a readable value fails.  Returns
        (rows, notes)."""
        maxima = maxima or {}
        notes = [f"the table has no column {t!r}" for t in maxima if t not in self.columns]
        if notes:
            raise ValueError(f"{self.path}: " + "; ".join(notes))
        keep = []
        for r in self.rows:
            if all((r.value(t) is not None and r.value(t) <= m) for t, m in maxima.items()):
                keep.append(r)
        if maxima:
            cut = ", ".join(f"{t} <= {m:g}" for t, m in maxima.items())
            notes.append(f"{len(keep)} of {len(self.rows)} decoys pass {cut}")
        return keep, notes

    def root(self, root: Optional[str] = None) -> str:
        """The plan folder the ``pdb`` paths are relative to: ``root`` when given, else the table's step
        folder's parent when the first extracted model is there, else the table's own folder."""
        if root:
            return os.path.abspath(root)
        here = os.path.dirname(os.path.abspath(self.path))
        first = next((r.pdb for r in self.rows if r.pdb), "")
        for cand in (os.path.dirname(here), here):
            if not first or os.path.isfile(os.path.join(cand, first)):
                return cand
        return os.path.dirname(here)

    def pdb_path(self, row: DecoyRow, root: Optional[str] = None) -> str:
        if not row.pdb:
            return ""
        return row.pdb if os.path.isabs(row.pdb) else os.path.normpath(os.path.join(self.root(root), row.pdb))


def read_decoy_table(path: str) -> DecoyTable:
    """Read a decoy table (TSV, or CSV for a ``.csv`` file or a header with commas and no tabs).  It needs
    the columns ``run`` and ``tag``; ``pdb`` may be missing (no models extracted); other columns are kept
    by name."""
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    first = text.split("\n", 1)[0]
    delim = "," if path.lower().endswith(".csv") or ("\t" not in first and "," in first) else "\t"
    reader = csv.reader(text.splitlines(), delimiter=delim)
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        raise ValueError(f"{path}: empty table") from None
    missing = [c for c in ("run", "tag") if c not in header]
    if missing:
        raise ValueError(f"{path}: not a decoy table (no column {', '.join(missing)}; columns: "
                         f"{', '.join(header[:8])})")
    rows: List[DecoyRow] = []
    seen = set()
    for k, rec in enumerate(reader, 2):
        if not any(x.strip() for x in rec):
            continue
        rec = rec + [""] * (len(header) - len(rec))
        d = dict(zip(header, (x.strip() for x in rec)))
        row = DecoyRow(d.pop("run"), d.pop("tag"), d.pop("pdb", ""), d)
        if row.key in seen:
            raise ValueError(f"{path}:{k}: decoy {row.key} appears twice")
        seen.add(row.key)
        rows.append(row)
    return DecoyTable(path, header, rows)


def parse_maxima(items: Iterable[str]) -> Dict[str, float]:
    """``['linear_chainbreak=5', 'score=-100']`` -> {term: maximum}."""
    out: Dict[str, float] = {}
    for item in items or ():
        term, sep, value = item.partition("=")
        if not sep or not term.strip():
            raise ValueError(f"expected TERM=VALUE, got {item!r}")
        try:
            out[term.strip()] = float(value)
        except ValueError:
            raise ValueError(f"{item!r}: {value!r} is not a number") from None
    return out


# ---------------------------------------------------------------------------
# silent-file score audit
# ---------------------------------------------------------------------------


def silent_files(paths: Sequence[str]) -> List[str]:
    """Silent files given directly or as folders (their ``*.out`` files, sorted)."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".out"))
        else:
            out.append(p)
    return out


def score_report(paths: Sequence[str], terms: Sequence[str] = ("score", "linear_chainbreak"),
                 maxima: Optional[Dict[str, float]] = None, extracted: str = "",
                 legacy_index: Optional[int] = None) -> Tuple[List[str], List[str], List[str], list]:
    """The column audit of ``mdatk rosetta scores``.  Returns (report lines, problems, failures, silents):
    failures are a cut's term missing from a header and extracted models that do not match the decoys."""
    files = silent_files(paths)
    silents = silent_scores.read_runs(files)
    problems = [p for s in silents for p in s.problems]
    maxima = maxima or {}
    wanted = list(dict.fromkeys(list(terms) + list(maxima)))
    lines = silent_scores.audit_lines(silents, wanted)
    lines.append("runs: " + ", ".join(f"{s.label} = {os.path.basename(s.path)} ({len(s.rows)})"
                                      for s in silents))
    failures = [p for t in maxima for p in silent_scores.missing_term(silents, t)]
    if maxima:
        n = sum(len(s.rows) for s in silents)
        keep = sum(1 for s in silents for _tag, v, _h in s.rows
                   if all(silent_scores.passes(v, t, m) for t, m in maxima.items()))
        cut = ", ".join(f"{t} <= {m:g}" for t, m in maxima.items())
        lines.append(f"{cut}: {keep} of {n} decoys pass (a decoy without a value fails)")
    if legacy_index is not None:
        # an old filter read one fixed column of the split SCORE line (index 0 is 'SCORE:')
        layouts = list(dict.fromkeys(h for s in silents for h in s.headers))
        for header in layouts:
            cols = ("SCORE:",) + header + ("description",)
            held = cols[legacy_index] if 0 <= legacy_index < len(cols) else "<past the end>"
            lines.append(f"index {legacy_index} of the split SCORE lines holds {held!r} in the layout with "
                         f"{len(header)} terms")
            for t in maxima:
                if held != t and t in header:
                    lines.append(f"  a filter on {t} by that index reads another column: {t} is index "
                                 f"{cols.index(t)}")
    if extracted:
        bad, counts = silent_scores.check_extracted(silents, extracted)
        lines.append("extracted: " + ", ".join(f"{k} {have}/{want}" for k, (want, have) in counts.items()))
        failures += bad
    return lines, problems, failures, silents


# ---------------------------------------------------------------------------
# rna_denovo command lines
# ---------------------------------------------------------------------------

# leaves that name different options under different namespaces: kept with their namespace
AMBIGUOUS_LEAVES = frozenset({"silent", "silent_struct_type", "pdb", "path", "prefix", "suffix", "tags", "l",
                              "list", "o", "file", "output", "input", "all"})
# how a flag is shown: the spelling MDATK's FARFAR2 step writes; other flags by their last component
DISPLAY = {"bps_moves": "-rna:denovo:bps_moves", "dump": "-rna:denovo:out:dump",
           "minimizer_use_coordinate_constraints":
               "-rna:denovo:minimize:minimizer_use_coordinate_constraints",
           "cryoem_scatterers": "-edensity:cryoem_scatterers", "mapfile": "-edensity:mapfile",
           "mapreso": "-edensity:mapreso", "jran": "-run:jran", "constant_seed": "-run:constant_seed",
           "seed_offset": "-run:seed_offset"}

# flags that differ between the runs of one condition by design (seeds, output names); left out of the
# comparisons only: the seeds are reported per run and checked for duplicates
IGNORED = ("-run:jran", "-out:file:silent", "-out:prefix", "-out:path", "-database", "-nstruct",
           "-out:file:silent_struct_type", "-run:constant_seed", "-run:seed_offset")
DEFAULT_JRAN = 1111111  # Rosetta's -run:jran when -constant_seed is given without one
# derived entries of the build (from the log lines before the command line), compared like flags
BUILD_ENTRIES = ("[rosetta version]", "[rosetta extras]", "[rosetta provenance]")
# input files worth fingerprinting: the path may be the same while the file is not
FILE_FLAGS = ("-secstruct_file", "-edensity:mapfile", "-fasta", "-s", "-native")
# flags that may differ in a pair whose variable is the pairing
PAIR_FLAGS = ("-secstruct_file", "-obligate_pair_explicit", "-obligate_pair", "-filter_lores_base_pairs")
# derived entries that belong to the pairing
PAIR_DERIVED = ("[secstruct]", "[content] -secstruct_file", "[controlled paired in secstruct]",
                "[controlled in obligate pairs]", "[effective condition]")
OBLIGATE = ("-obligate_pair_explicit", "-obligate_pair")


def flag_key(flag: str) -> str:
    """One name per option: one leading '-', '::' read as ':', and the namespace dropped unless the last
    component is ambiguous (``-in:file:s`` = ``-s``; ``-in:file:silent`` and ``-out:file:silent`` stay)."""
    name = re.sub(r":{2,}", ":", flag.strip().lstrip("-")).strip(":")
    parts = [p for p in name.split(":") if p]
    if not parts:
        return flag
    leaf = parts[-1]
    if leaf in AMBIGUOUS_LEAVES:
        return "-" + ":".join(parts)
    return DISPLAY.get(leaf, "-" + leaf)


def is_flag(token: str) -> bool:
    """'-cycles', '--cycles', '-in::file::s'; not a negative number or a lone '-'."""
    t = token[2:] if token.startswith("--") else token[1:] if token.startswith("-") else ""
    return bool(t) and (t[0].isalpha() or t[0] in "_:")


def norm_value(values: Sequence[str]) -> str:
    """Values compared by meaning: a bare flag is 'true', '5' = '5.0', 'True' = 'true'."""
    if not values:
        return "true"
    out = []
    for v in values:
        low = v.lower()
        if low in ("true", "false"):
            out.append(low)
            continue
        try:
            out.append(f"{float(v):g}")
        except ValueError:
            out.append(v)
    return " ".join(out)


def _strip_comment(line: str) -> str:
    return re.split(r"(?:^|\s)#", line, maxsplit=1)[0]


@dataclass
class Command:
    """One rna_denovo command line with its @flags files expanded."""
    exe: str = ""
    values: Dict[str, List[str]] = field(default_factory=dict)  # flag key -> raw values (last occurrence)
    repeated: List[str] = field(default_factory=list)
    flag_files: List[str] = field(default_factory=list)  # files read for @file, resolved
    problems: List[str] = field(default_factory=list)

    @property
    def flags(self) -> Dict[str, str]:
        return {k: norm_value(v) for k, v in self.values.items()}


def resolve(path: str, bases: Sequence[str]) -> Tuple[str, List[str]]:
    """The first existing ``path`` among ``bases`` (absolute paths as they are); also every other base where
    a different file of that name exists.  Returns (resolved or '', other candidates)."""
    if os.path.isabs(path):
        return (path if os.path.exists(path) else ""), []
    found = []
    for b in bases:
        q = os.path.normpath(os.path.join(b, path))
        if os.path.isfile(q) and q not in found:
            found.append(q)
    if not found:
        return "", []
    return found[0], found[1:]


def parse_command(text: str, bases: Sequence[str] = (), max_depth: int = 8) -> Command:
    """Parse a logged command line (the part after ``command:`` when present).  ``@file`` arguments, also
    nested ones inside flags files, are read from the first of ``bases`` (the run's working folder first)
    that has them; '#' starts a comment in a flags file."""
    if "command:" in text:
        text = text.split("command:", 1)[1]
    tokens = text.split()
    cmd = Command()
    if tokens and not is_flag(tokens[0]) and not tokens[0].startswith("@"):
        cmd.exe = tokens.pop(0)
    current: Optional[str] = None
    seen_keys: set = set()

    def expand(toks: List[str], depth: int, stack: Tuple[str, ...]) -> None:
        nonlocal current
        for tok in toks:
            if tok.startswith("@") and len(tok) > 1:
                path, _others = resolve(tok[1:], bases)
                if not path:
                    cmd.problems.append(f"flags file {tok[1:]} not found")
                    cmd.values.setdefault("[unread flags file]", []).append(tok[1:])
                    continue
                if path in stack or depth >= max_depth:
                    cmd.problems.append(f"flags file {tok[1:]} includes itself")
                    continue
                cmd.flag_files.append(path)
                try:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        inner = [t for ln in fh for t in _strip_comment(ln).split()]
                except OSError as exc:
                    cmd.problems.append(f"flags file {path}: {exc.strerror or exc}")
                    continue
                current = None  # a flags file starts with a flag; values never continue across it
                expand(inner, depth + 1, stack + (path,))
                current = None
            elif is_flag(tok):
                current = flag_key(tok)
                if current in seen_keys and current not in cmd.repeated:
                    cmd.repeated.append(current)
                seen_keys.add(current)
                cmd.values[current] = []  # a repeated flag: the last occurrence is the one Rosetta uses
            elif current is not None:
                cmd.values[current].append(tok)
            else:
                cmd.problems.append(f"argument {tok!r} before any flag")

    expand(tokens, 0, ())
    return cmd


_PRINTS: Dict[Tuple, str] = {}  # (real path, size, mtime, full) -> fingerprint


def fingerprint(path: str, full: bool = True) -> str:
    """``size:sha256[:12]`` of a file (``full`` False: 1 MiB at the start, middle and end of large files)."""
    try:
        st = os.stat(path)
    except OSError:
        return "?"
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns, full)
    if key not in _PRINTS:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                if full or st.st_size <= 3 << 20:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                else:
                    h.update(fh.read(1 << 20))
                    fh.seek(st.st_size // 2)
                    h.update(fh.read(1 << 20))
                    fh.seek(max(0, st.st_size - (1 << 20)))
                    h.update(fh.read(1 << 20))
        except OSError:
            return "?"
        _PRINTS[key] = f"{st.st_size}:{h.hexdigest()[:12]}"
    return _PRINTS[key]


# ---------------------------------------------------------------------------
# residues: controlled ranges, FASTA numbering, -input_res, -s, obligate pairs
# ---------------------------------------------------------------------------

Res = Tuple[Optional[str], int]  # (chain or None for "any chain", number)
_RES_ITEM = re.compile(r"^(?:([A-Za-z0-9]):)?(-?\d+)(?:-(-?\d+))?$")


def parse_residues(text: str) -> List[Tuple[Optional[str], int, int]]:
    """``'A:22-24,A:35-37'``, ``'22-24 35-37'``, ``'A:1-6,8'`` -> [(chain or None, first, last)]."""
    out = []
    for tok in re.split(r"[\s;]+", (text or "").strip()):
        if not tok:
            continue
        chain = None
        m = re.match(r"^([A-Za-z0-9]):(.*)$", tok)
        if m and not m.group(2).startswith("-") and m.group(2):
            chain, tok = m.group(1), m.group(2)
        for item in tok.split(","):
            if not item:
                continue
            mm = _RES_ITEM.match(item)
            if not mm:
                raise ValueError(f"cannot read residue range {item!r} (expected e.g. A:22-24 or 22-24)")
            c = mm.group(1) or chain
            a = int(mm.group(2))
            b = int(mm.group(3)) if mm.group(3) is not None else a
            out.append((c, min(a, b), max(a, b)))
    return out


def in_ranges(res: Res, ranges) -> bool:
    chain, num = res
    for c, a, b in ranges:
        if a <= num <= b and (c is None or chain is None or c == chain):
            return True
    return False


def fasta_numbering(path: str) -> List[Res]:
    """(chain, number) of every sequence position of a Rosetta FASTA file, from header tokens such as
    ``A:1-16`` (MDATK writes one per chain); without them chain None and 1, 2, 3 ... across records."""
    out: List[Res] = []
    spans: List[Tuple[str, int, int]] = []
    seq: List[str] = []

    def flush():
        n = len("".join(seq))
        given = [(c, k) for c, a, b in spans for k in range(a, b + 1)]
        if len(given) == n:
            out.extend(given)
        else:
            start = (out[-1][1] + 1) if out else 1
            out.extend((None, start + k) for k in range(n))

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    flush()
                seq = []
                spans = [(m.group(1), int(m.group(2)), int(m.group(3)))
                         for m in re.finditer(r"\b([A-Za-z0-9]):(-?\d+)-(-?\d+)\b", line)]
            elif line:
                seq.append(re.sub(r"\s+", "", line))
    if seq:
        flush()
    return out


def secstruct_pairs(path: str) -> Tuple[str, List[Tuple[int, int]]]:
    """(dot-bracket, 1-based pairs) of a Rosetta secstruct file (line 1; ()[]{}<> brackets)."""
    from .workflows.rna import parse_dotbracket
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    db, pairs, _sp = parse_dotbracket(text)
    return db, pairs


def pdb_residues(path: str) -> List[Res]:
    from . import pdbtools
    return [(r.chain, r.resseq) for r in pdbtools.read_pdb(path).residues()]


def residues_in(text: str) -> List[Res]:
    """Residues named in an obligate-pair value (``22 37 W W A`` or ``A:22 A:37``): every number, with its
    chain when written ``X:n``; edge and orientation letters are skipped."""
    return [(m.group(1), int(m.group(2)))
            for m in re.finditer(r"(?:\b([A-Za-z0-9]):)?(?<![\w.])(-?\d+)\b", text)]


def _ranges_text(res: Sequence[Res]) -> str:
    by: Dict[Optional[str], List[int]] = {}
    for c, n in res:
        by.setdefault(c, []).append(n)
    parts = []
    for c, nums in by.items():
        nums = sorted(set(nums))
        runs, start = [], nums[0]
        for a, b in zip(nums, nums[1:] + [None]):
            if b is None or b != a + 1:
                runs.append(f"{start}-{a}" if a != start else f"{start}")
                start = b
        parts += [f"{c}:{r}" if c else r for r in runs]
    return ",".join(parts)


# ---------------------------------------------------------------------------
# the log around a command line: the Rosetta build before it, the random seed after it
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # Rosetta colours some tracer output
_TRACER = re.compile(r"^\s*[A-Za-z_][\w.]*(?:\s*\{\d+\})?\s*:\s*")  # 'core.init: ', 'core.init {0}: '
_VERSION = re.compile(r"\b(?:Mini-)?Rosetta\s+version\b(?!\s+information)\s*:?\s*(?P<text>.*)$", re.I)
_EXTRAS = re.compile(r"\bRosetta\s+extras\s*:?\s*(?P<text>.*)$", re.I)
# the block that -version writes instead of the one-line version
_VERSION_FIELDS = {"package": "package", "testing server revision": "revision", "pep-440 version": "version",
                   "git commit": "commit", "url": "repo", "date": "date"}
_FIELD_LINE = re.compile(r"^(?P<key>Package|Testing Server Revision|PEP-440 Version|Git Commit|URL|Date)"
                         r"\s*:\s*(?P<value>.*)$", re.I)
_TIME_TOKEN = re.compile(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?|[+-]\d{4}|[A-Z]{3,4}")  # after a date
# other provenance a custom build or a run script may log on core.init before the command line
_INIT_DETAIL = re.compile(r"^\s*core\.init\S*\s*:.*\b(?:commit|checkout|branch|revision)\b", re.I)
_INIT = re.compile(r"^\s*core\.init\b")
_PROVENANCE = re.compile(r"\b(?:Mini-)?Rosetta\s+(?:version|extras)\b", re.I)  # also the -version heading
_SEED_LINE = re.compile(r"(?P<mode>constant|'?rng device'?|'?time'?)\s+seed\s+mode\b.*?\bseed\s*=\s*"
                        r"(?P<seed>-?\d+)(?:.*?\bseed_offset\s*=\s*(?P<offset>-?\d+))?"
                        r"(?:.*?\breal_seed\s*=\s*(?P<real>-?\d+))?", re.I)
_FIELD_ORDER = ("package", "revision", "version", "commit", "repo", "date", "other")


def parse_version(text: str) -> Dict[str, str]:
    """Fields of a version line's text (after 'Rosetta version:'), read tolerantly: ``package``
    (rosetta.binary.linux.release-371), ``revision`` (r371), ``version`` (2024.39+release.59628fb),
    ``commit`` (a hex SHA-1), ``repo`` (a URL, git@..., ...git, or the word after 'from'), ``date`` (ISO, with
    a following time and zone) and ``other`` (the rest, e.g. 'exported')."""
    out: Dict[str, str] = {}
    rest: List[str] = []
    tokens = text.split()
    i = 0
    while i < len(tokens):
        tok, low = tokens[i], tokens[i].lower()
        if low == "from" and i + 1 < len(tokens):
            out.setdefault("repo", tokens[i + 1])
            i += 2
            continue
        if re.fullmatch(r"r\d+", tok):
            out.setdefault("revision", tok[1:])
        elif re.fullmatch(r"[0-9a-f]{7,40}", low) and not tok.isdigit():
            out.setdefault("commit", tok)
        elif re.match(r"^(?:[a-z][\w+.-]*://|git@|ssh:)", low) or low.endswith(".git"):
            out.setdefault("repo", tok)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T_][\d:.]+)?(?:Z|[+-]\d{2}:?\d{2})?", tok):
            date = tok
            while i + 1 < len(tokens) and _TIME_TOKEN.fullmatch(tokens[i + 1]):
                i += 1
                date += " " + tokens[i]
            out.setdefault("date", date)
        elif re.fullmatch(r"\d{4}\.\d{1,2}(?:[.+]\S*)?|\d+\.\d+(?:\.\d+)*\S*", tok):
            out.setdefault("version", tok)
        elif "release" in low or low.startswith(("rosetta.", "pyrosetta")):
            out.setdefault("package", tok)
        else:
            rest.append(tok)
        i += 1
    if rest:
        out["other"] = " ".join(rest)
    return out


@dataclass
class Build:
    """The Rosetta build a run names in its log before its command line."""
    lines: List[str] = field(default_factory=list)  # the raw provenance lines, in log order
    version_line: str = ""  # the text after 'Rosetta version', whitespace collapsed
    extras: str = ""
    fields: Dict[str, str] = field(default_factory=dict)  # see parse_version
    details: List[str] = field(default_factory=list)  # other provenance lines (commit, checkout ...)

    @property
    def known(self) -> bool:
        return bool(self.version_line or self.fields or self.details)

    @property
    def version(self) -> str:
        """The version in a canonical form: the parsed fields in a fixed order (so the one-line version and
        the -version block of one build agree), else the raw text."""
        if not self.fields:
            return self.version_line
        f = self.fields
        return " ".join(("r" + f[k] if k == "revision" else f[k]) for k in _FIELD_ORDER if f.get(k))

    @property
    def identity(self) -> str:
        """What tells two builds apart: the version, the extras and the other provenance lines."""
        return " | ".join(x for x in [self.version, f"extras {self.extras}" if self.extras else ""]
                          + self.details if x)

    def describe(self) -> str:
        if not self.known:
            return "no Rosetta version line in the log"
        f = self.fields
        head = f.get("version") or f.get("package") or f.get("other") or self.version_line
        bits = [head] + ([f"r{f['revision']}"] if f.get("revision") else [])
        bits += [f"commit {f['commit']}"] if f.get("commit") else []
        bits += [f"from {f['repo']}"] if f.get("repo") else []
        bits += [f"({f['date']})"] if f.get("date") else []
        bits += [f"extras {self.extras}"] if self.extras else []
        bits += self.details
        return " ".join(b for b in bits if b)

    def entries(self) -> Dict[str, str]:
        """The derived entries compared like flags (absent when the log has no such line)."""
        values = (self.version, self.extras, " | ".join(self.details))
        return {key: v for key, v in zip(BUILD_ENTRIES, values) if v}


def _tracer_text(line: str) -> str:
    return _TRACER.sub("", line, count=1).strip()


def parse_build(lines: Sequence[str]) -> Build:
    """The build from the provenance lines before a command line (raw lines kept)."""
    b = Build(lines=[ln.strip() for ln in lines])
    for ln in lines:
        text = _tracer_text(ln)
        m = _FIELD_LINE.match(text)
        if m:
            b.fields.setdefault(_VERSION_FIELDS[m.group("key").lower()], " ".join(m.group("value").split()))
            continue
        m = _EXTRAS.search(text)
        if m:
            b.extras = " ".join(m.group("text").split())
            continue
        m = _VERSION.search(text)
        if m:
            if not b.version_line:
                b.version_line = " ".join(m.group("text").split())
                for k, v in parse_version(b.version_line).items():
                    b.fields.setdefault(k, v)
            continue
        if re.search(r"\bRosetta\s+version\s+information\b", text, re.I):
            continue  # the heading of the -version block
        b.details.append(" ".join(text.split()))
    # the one-line version leaves out a 'devel' package and a 'None' revision: so does the canonical form
    if b.fields.get("package", "").lower() == "devel":
        del b.fields["package"]
    if b.fields.get("revision", "").lower() in ("none", ""):
        b.fields.pop("revision", None)
    return b


@dataclass
class Seed:
    """The random seed of one start of a run."""
    constant: Optional[bool]  # -constant_seed in effect (None: not known)
    value: Optional[int]  # the seed the run used: Rosetta's logged real_seed, else -jran (+ -seed_offset)
    jran: str = ""  # -jran as given ('' when absent)
    source: str = ""  # "log" (the seed line after the command line) or "flags"
    mode: str = ""  # constant / rng device / time (from the log)
    line: str = ""  # the raw seed line

    def describe(self) -> str:
        if self.constant and self.value is not None:
            how = "logged" if self.source == "log" else (
                "Rosetta's default -jran" if not self.jran else "from -jran")
            return f"{self.value} (-constant_seed, {how})"
        if self.constant is False:
            got = f", real_seed {self.value}" if self.value is not None else ""
            ignored = f"; -jran {self.jran} is ignored without -constant_seed" if self.jran else ""
            return f"random ({self.mode or 'no -constant_seed'}{got}{ignored})"
        return "unknown"


def seed_of(cmd: Command, seed_line: str = "") -> Seed:
    """The seed of a start: the 'seed mode' line Rosetta logs after the command line when there is one (its
    real_seed includes -seed_offset), else -constant_seed / -jran / -seed_offset from the command line."""
    jran = " ".join(cmd.values.get("-run:jran", []))
    m = _SEED_LINE.search(seed_line or "")
    if m:
        mode = m.group("mode").strip("'").lower()
        value = int(m.group("real") if m.group("real") is not None else m.group("seed"))
        return Seed(mode == "constant", value, jran, "log", mode, seed_line.strip())
    constant = cmd.flags.get("-run:constant_seed") == "true"
    value: Optional[int] = None
    if constant:
        try:
            value = int(float(jran)) if jran else DEFAULT_JRAN
            value += int(float(" ".join(cmd.values.get("-run:seed_offset", [])) or 0))
        except ValueError:
            value = None
    return Seed(constant, value, jran, "flags")


@dataclass
class Start:
    """One start of a run in its log: the command line, the build named before it, the seed line after it."""
    command: str
    build: Build
    seed_line: str = ""


def read_log(log: str, exe_pattern: str = "rna_denovo") -> List[Start]:
    """Every start of an rna_denovo run in ``log``: its logged command line, the provenance lines since the
    previous command line and the first seed line after it.  A command line of another program
    (extract_pdbs, rna_cluster ...) ends a start; its header is not a run's."""
    starts: List[Start] = []
    pending: List[str] = []
    current: Optional[Start] = None
    try:
        with open(log, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                ln = _ANSI.sub("", raw).rstrip("\r\n")
                if "command:" in ln:
                    head = ln.split("command:", 1)[1].split()
                    if head and exe_pattern in os.path.basename(head[0]):
                        current = Start(ln.split("command:", 1)[1].strip(), parse_build(pending))
                        starts.append(current)
                    else:
                        current = None
                    pending = []
                    continue
                if current is not None and not current.seed_line and _SEED_LINE.search(ln):
                    current.seed_line = ln.strip()
                    continue
                if _PROVENANCE.search(ln) or _INIT_DETAIL.search(ln) or (
                        pending and _INIT.match(ln) and _FIELD_LINE.match(_tracer_text(ln))):
                    pending.append(ln.strip())
    except OSError:
        pass
    return starts


# ---------------------------------------------------------------------------
# collecting runs and conditions
# ---------------------------------------------------------------------------


@dataclass
class Thread:
    """One rna_denovo run: its log, working folder and command line (with the derived entries), and per
    start (a restart appends one) the Rosetta build and the random seed."""
    log: str
    cwd: str
    command: Command
    flags: Dict[str, str]
    starts: int = 1  # command lines in the log (a restart appends one)
    notes: List[str] = field(default_factory=list)
    builds: List[Build] = field(default_factory=list)
    seeds: List[Seed] = field(default_factory=list)

    @property
    def build(self) -> Build:
        return self.builds[0] if self.builds else Build()

    @property
    def seed(self) -> Seed:
        return self.seeds[0] if self.seeds else Seed(None, None)


@dataclass
class Condition:
    label: str
    path: str
    threads: List[Thread] = field(default_factory=list)
    flags: Dict[str, str] = field(default_factory=dict)  # agreed values (or a 'varies' marker)
    notes: List[str] = field(default_factory=list)


def _is_log(name: str) -> bool:
    return name.endswith(".log") or (name.startswith("log_") and name.endswith(".txt"))


def _searched(dirs: Iterable[str]) -> List[str]:
    """The subfolders searched for runs, sorted: never the completed runs MDATK moved aside into a step
    folder's superseded/ (they are not runs of the step's current condition)."""
    return sorted(d for d in dirs if d != SUPERSEDED)


def find_logs(folder: str, recursive: bool = True) -> List[str]:
    if not recursive:
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                      if _is_log(f) and os.path.isfile(os.path.join(folder, f)))
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = _searched(dirs)
        out += [os.path.join(root, f) for f in files if _is_log(f)]
    return sorted(out)


def command_lines(log: str, exe_pattern: str = "rna_denovo") -> List[str]:
    """Every logged command line of an rna_denovo run in ``log`` (a log of extract_pdbs, rna_cluster ... has
    none: its flags are not those of the runs)."""
    return [s.command for s in read_log(log, exe_pattern)]


def run_folders(log: str, condition: str, silent: str = "") -> List[str]:
    """Where the run may have worked, most likely first: MDATK's ``runs/rNN`` beside a
    ``..._farfar2-rNN_<stamp>.log``; folders from which the run's own silent output (``silent``, as named on
    its command line) exists; folders holding PDB or silent files; the log's folder; the condition folder's
    other subfolders (two levels, e.g. ``run_<time>_<n>/``); the condition folder."""
    first: List[str] = []
    parts = parse_name(os.path.basename(log))
    if parts:
        m = re.search(r"(?:^|-)(r\d+)$", parts["tag"])
        if m:
            first.append(os.path.join(os.path.dirname(log), "runs", m.group(1)))
    subs: List[str] = []
    with_output: List[str] = []
    for root, dirs, files in os.walk(condition):
        dirs[:] = _searched(dirs)
        rel = os.path.relpath(root, condition)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth:
            subs.append(root)
            if any(f.endswith((".pdb", ".out")) for f in files):
                with_output.append(root)
        if depth >= 2:
            dirs[:] = []
    candidates = first + with_output + [os.path.dirname(log)] + subs + [condition]
    if silent and not os.path.isabs(silent):
        own = [d for d in candidates if os.path.isfile(os.path.normpath(os.path.join(d, silent)))]
        candidates = own + candidates
    seen, uniq = set(), []
    for d in candidates:
        d = os.path.normpath(d)
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            uniq.append(d)
    return uniq


def derive(cmd: Command, bases: Sequence[str], controlled,
           full_hash: bool) -> Tuple[Dict[str, str], List[str]]:
    """The flags plus derived entries: input fingerprints, the secondary structure and, with ``controlled``
    residues, how their pairing was given (secstruct, obligate pairs, fixed in -s / -input_res)."""
    flags = cmd.flags
    notes = list(cmd.problems)
    found: Dict[str, List[str]] = {}
    for fl in FILE_FLAGS:
        if fl not in cmd.values:
            continue
        prints, paths = [], []
        for val in cmd.values[fl]:
            path, others = resolve(val, bases)
            if not path:
                notes.append(f"{fl} names {val}, which is not found")
                prints.append("missing")
                continue
            fp = fingerprint(path, full_hash)
            if any(fingerprint(o, full_hash) != fp for o in others):
                notes.append(f"{fl} {val}: different files of that name in {os.path.dirname(path)} and "
                             f"{os.path.dirname(others[0])}; the first is used")
            prints.append(fp)
            paths.append(path)
        flags[f"[content] {fl}"] = " ".join(prints)
        found[fl] = paths
    # (the flags files themselves need no fingerprint: every flag they hold is compared)
    ss = found.get("-secstruct_file", [])
    db, pairs = "", []
    if ss:
        try:
            db, pairs = secstruct_pairs(ss[0])
            flags["[secstruct]"] = db
        except (OSError, ValueError) as exc:
            notes.append(f"secstruct {ss[0]}: {exc}")
    if not controlled:
        return flags, notes
    numbering: List[Res] = []
    if found.get("-fasta"):
        try:
            numbering = fasta_numbering(found["-fasta"][0])
        except OSError as exc:
            notes.append(f"FASTA {found['-fasta'][0]}: {exc}")
    wanted = [r for r in numbering if in_ranges(r, controlled)] if numbering else []
    why = []
    if db and numbering:
        if len(numbering) != len(db):
            notes.append(f"the FASTA has {len(numbering)} residues and the secstruct {len(db)}")
        index = {k + 1: numbering[k] for k in range(min(len(numbering), len(db)))}
        paired = sorted({index[k] for p in pairs for k in p
                         if k in index and in_ranges(index[k], controlled)})
        state = ("paired" if wanted and len(paired) == len(wanted) else "unpaired" if not paired
                 else f"partly paired ({len(paired)} of {len(wanted)})")
        flags["[controlled paired in secstruct]"] = state + (f" {_ranges_text(paired)}" if paired else "")
        if paired:
            why.append(f"secstruct pairs {_ranges_text(paired)}")
    elif db:
        notes.append("no FASTA numbering: the secondary structure cannot be matched to the controlled "
                     "residues")
    def placed(res: Res) -> Res:
        """A number without a chain is a full-model number: its (chain, number) in the FASTA when unique."""
        if res[0] is not None or not numbering:
            return res
        hits = [r for r in numbering if r[1] == res[1]]
        return hits[0] if len(hits) == 1 else res

    obligate = " ".join(" ".join(cmd.values[k]) for k in OBLIGATE if k in cmd.values)
    if obligate:
        hit = sorted({placed(r) for r in residues_in(obligate) if in_ranges(placed(r), controlled)},
                     key=lambda r: (r[0] or "", r[1]))
        flags["[controlled in obligate pairs]"] = _ranges_text(hit) if hit else "no"
        if hit:
            why.append(f"obligate pairs {_ranges_text(hit)}")
    fixed: List[str] = []
    for path in found.get("-s", []):
        try:
            hit = [r for r in pdb_residues(path) if in_ranges(r, controlled)]
        except (OSError, ValueError) as exc:
            notes.append(f"-s {path}: {exc}")
            continue
        flags[f"[controlled in -s] {os.path.basename(path)}"] = _ranges_text(hit) if hit else "no"
        if hit:
            fixed.append(f"{os.path.basename(path)} {_ranges_text(hit)}")
    if "-input_res" in cmd.values:
        try:
            given = parse_residues(" ".join(cmd.values["-input_res"]))
            hit = sorted({r for r in (numbering or []) if in_ranges(r, given) and in_ranges(r, controlled)},
                         key=lambda r: (r[0] or "", r[1]))
            if not numbering:  # numbers only
                hit = [(c, n) for c, a, b in given for n in range(a, b + 1) if in_ranges((c, n), controlled)]
            flags["[controlled in -input_res]"] = _ranges_text(hit) if hit else "no"
            if hit:
                fixed.append(f"-input_res {_ranges_text(hit)}")
        except ValueError as exc:
            notes.append(f"-input_res: {exc}")
    if fixed:
        why.append("fixed in the input structure (" + "; ".join(fixed) + ")")
    flags["[effective condition]"] = ("imposed: " + "; ".join(why)) if why else "free"
    if why and flags.get("[controlled paired in secstruct]", "").startswith("unpaired"):
        notes.append("the secstruct leaves the controlled residues unpaired, but their pairing is given "
                     "anyway (" + "; ".join(why) + ")")
    return flags, notes


def collect(folder: str, label: str = "", controlled=None, full_hash: bool = True, recursive: bool = True,
            exe_pattern: str = "rna_denovo", ignored: Sequence[str] = IGNORED) -> Condition:
    """The runs of one condition folder."""
    cond = Condition(label or os.path.basename(os.path.normpath(folder)), folder)
    for log in find_logs(folder, recursive):
        starts = read_log(log, exe_pattern)
        lines = [s.command for s in starts]
        if not lines:
            continue
        # the silent file named on the command line itself tells which folder the run worked in
        bare = parse_command(lines[0], ())
        bases = run_folders(log, folder, " ".join(bare.values.get("-out:file:silent", [])[:1]))
        cmd = parse_command(lines[0], bases)
        flags, notes = derive(cmd, bases, controlled, full_hash)
        if cmd.exe:
            flags["[executable]"] = cmd.exe
        flags.update(starts[0].build.entries())
        t = Thread(log, bases[0] if bases else folder, cmd, flags, len(lines), notes,
                   [s.build for s in starts])
        t.seeds = [seed_of(cmd if k == 0 else parse_command(s.command, bases), s.seed_line)
                   for k, s in enumerate(starts)]
        if len(lines) > 1:
            later = parse_command(lines[-1], bases).flags
            later.update(starts[-1].build.entries())
            first = cmd.flags
            first.update(starts[0].build.entries())
            diff = sorted(k for k in set(first) | set(later)
                          if k not in ignored and first.get(k) != later.get(k))
            t.notes.append(f"{len(lines)} command lines (restarts)" +
                           (f"; the last differs in {', '.join(diff)}" if diff else ""))
        if cmd.flag_files:
            newer = [f for f in cmd.flag_files if os.path.getmtime(f) > os.path.getmtime(log)]
            if newer:
                t.notes.append(f"{os.path.basename(newer[0])} was written after this run's log: the audit "
                               "reads the file as it is now, which may not be what the run read")
        cond.threads.append(t)
    if not cond.threads:
        cond.notes.append("no logged rna_denovo command line ('core.init: command: ...') in any log")
        return cond
    keys = sorted({k for t in cond.threads for k in t.flags if k not in ignored})
    for k in keys:
        vals = [t.flags.get(k) for t in cond.threads]
        distinct = sorted({v if v is not None else "(absent)" for v in vals})
        if len(distinct) == 1:
            if vals[0] is not None:
                cond.flags[k] = vals[0]
        else:
            cond.flags[k] = "(varies: " + " | ".join(distinct) + ")"
            cond.notes.append(f"the runs disagree on {k} (e.g. {os.path.basename(cond.threads[-1].log)})")
    for t in cond.threads:
        cond.notes += [f"{os.path.basename(t.log)}: {n}" for n in dict.fromkeys(t.notes)]
    return cond


def find_conditions(roots: Sequence[str], recursive: bool = True,
                    exe_pattern: str = "rna_denovo") -> List[Tuple[str, str]]:
    """(label, folder) of every condition: a root whose own folder holds rna_denovo logs is one condition,
    otherwise each of its subfolders is (never ``superseded/``, see :func:`_searched`).  Labels are paths relative to the root (the root's name for the root
    itself), with the root's name in front when two roots give the same label."""
    out: List[Tuple[str, str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            raise ValueError(f"not a folder: {root}")
        root = os.path.normpath(root)
        name = os.path.basename(os.path.abspath(root))
        if any(command_lines(f, exe_pattern) for f in find_logs(root, recursive=False)):
            out.append((name, root, name))
            continue
        subs = _searched(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        out += [(d, os.path.join(root, d), name) for d in subs] or [(name, root, name)]
    labels = [lab for lab, _p, _r in out]
    return [(f"{r}/{lab}" if labels.count(lab) > 1 and lab != r else lab, p) for lab, p, r in out]


# ---------------------------------------------------------------------------
# comparing conditions
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    control: str
    treatment: str
    expected: List[str]
    other: List[str]

    @property
    def clean(self) -> bool:
        return not self.other


def compare(a: Condition, b: Condition, allowed: Sequence[str] = PAIR_FLAGS,
            ignored: Sequence[str] = IGNORED) -> PairResult:
    """Differences of two conditions: those expected when the pairing is the variable (``allowed`` flags and
    the derived pairing entries) and all others, which confound the comparison."""
    allowed = set(allowed) | set(PAIR_DERIVED)
    diff = sorted(k for k in set(a.flags) | set(b.flags)
                  if k not in ignored and a.flags.get(k) != b.flags.get(k) and not same_content(k, [a, b]))
    expected = [k for k in diff if k in allowed or any(k == f"[content] {f}" for f in allowed)]
    return PairResult(a.label, b.label, expected, [k for k in diff if k not in expected])


def same_content(flag: str, conds: Sequence[Condition]) -> bool:
    """An input-file flag whose paths differ while every condition read the same content (MDATK's file names
    carry the step number): no difference."""
    if flag not in FILE_FLAGS:
        return False
    prints = {c.flags.get(f"[content] {flag}") for c in conds}
    return len(prints) == 1 and None not in prints and "missing" not in next(iter(prints)).split()


def match_label(name: str, labels: Sequence[str]) -> str:
    """The label ``name`` means: itself, or the one label whose last path components are ``name``."""
    if name in labels:
        return name
    hits = [lab for lab in labels if lab.endswith("/" + name)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"{name!r} is " + ("ambiguous: " + ", ".join(hits) if hits else
                                         "not one of the conditions: " + ", ".join(labels)))


def pairs_from(specs: Sequence[str], suffix: str,
               labels: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Pairs from 'CONTROL=TREATMENT' specs and from a name suffix (X and X<suffix>).  Returns (pairs,
    problems)."""
    out, problems = [], []
    for spec in specs or ():
        c, sep, t = spec.partition("=")
        if not sep:
            problems.append(f"pair {spec!r}: expected CONTROL=TREATMENT")
            continue
        try:
            out.append((match_label(c.strip(), labels), match_label(t.strip(), labels)))
        except ValueError as exc:
            problems.append(f"pair {spec!r}: {exc}")
    if suffix:
        out += [(lab[:-len(suffix)], lab) for lab in labels
                if lab.endswith(suffix) and lab[:-len(suffix)] in labels]
    return list(dict.fromkeys(out)), problems


def _short(v: Optional[str], width: int = 100) -> str:
    if v is None:
        return "(absent)"
    return v if len(v) <= width else v[:width - 3] + "..."


def _diff_lines(k: str, a: Optional[str], b: Optional[str]) -> List[str]:
    if a is None or b is None or max(len(a), len(b)) <= 90:
        return [f"      {k}", f"        {_short(a)}", f"        {_short(b)}"]
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    i = max(0, i - 8)  # long values (-set_weights): show where they part
    pre = "..." if i else ""
    return [f"      {k}", f"        {pre}{a[i:i + 90]}", f"        {pre}{b[i:i + 90]}"]


# ---------------------------------------------------------------------------
# builds and seeds across the runs
# ---------------------------------------------------------------------------


def _seed_ranges(values: Sequence[int]) -> str:
    vals = sorted(set(values))
    out, k = [], 0
    while k < len(vals):
        j = k
        while j + 1 < len(vals) and vals[j + 1] == vals[j] + 1:
            j += 1
        out.append(f"{vals[k]}-{vals[j]}" if j > k else f"{vals[k]}")
        k = j + 1
    return ", ".join(out)


def seeds_text(threads: Sequence[Thread]) -> str:
    """The seeds of runs (first starts): '3001-3060 (-constant_seed)', 'random (no -constant_seed)' ..."""
    seeds = [t.seed for t in threads]
    fixed = [s.value for s in seeds if s.constant and s.value is not None]
    parts = []
    if fixed:
        repeated = len(fixed) - len(set(fixed))
        parts.append(f"{_seed_ranges(fixed)} (-constant_seed" + (f", {repeated} repeated" if repeated else "")
                     + ")")
    random = sum(1 for s in seeds if s.constant is False)
    unknown = len(seeds) - len(fixed) - random
    if random:
        parts.append(f"{random} random (no -constant_seed)")
    if unknown:
        parts.append(f"{unknown} unknown")
    return "; ".join(parts)


def build_labels(conds: Sequence[Condition]) -> Dict[str, Tuple[str, Build]]:
    """Build identity -> (B1, B2 ..., the build) for every distinct build of every start, in order."""
    out: Dict[str, Tuple[str, Build]] = {}
    for c in conds:
        for t in c.threads:
            for b in t.builds:
                if b.known:
                    out.setdefault(b.identity, (f"B{len(out) + 1}", b))
    return out


def _run_name(c: Condition, t: Thread) -> str:
    return f"{c.label}/{os.path.relpath(t.log, c.path)}"


def provenance_lines(conds: Sequence[Condition]) -> List[str]:
    """Which Rosetta built the runs: every distinct build with its runs per condition and its raw lines, and a
    warning when the runs were built with more than one."""
    builds = build_labels(conds)
    lines = ["", "Rosetta builds (from the lines each log writes before its command line):"]
    unknown = [_run_name(c, t) for c in conds for t in c.threads if not t.build.known]
    if not builds:
        return lines + ["  no log names its Rosetta version: which build made the runs is not recorded"]
    for ident, (label, b) in builds.items():
        who: Dict[str, int] = {}
        for c in conds:
            for t in c.threads:
                if any(x.identity == ident for x in t.builds):
                    who[c.label] = who.get(c.label, 0) + 1
        lines.append(f"  {label}: {b.describe()}")
        lines.append(f"      {sum(who.values())} run(s): " + ", ".join(f"{k} {n}" for k, n in who.items()))
        lines += [f"      | {raw}" for raw in b.lines[:6]]
    if unknown:
        lines.append(f"  {len(unknown)} run(s) without a version line: {', '.join(unknown[:4])}"
                     + (" ..." if len(unknown) > 4 else ""))
    if len(builds) > 1:
        names = ", ".join(lab for lab, _b in builds.values())
        lines.append(f"  WARNING: these runs were built with {len(builds)} different Rosetta builds "
                     f"({names}): runs of different builds are not one experiment, and a comparison between "
                     "them is confounded by the build")
    return lines


def _inputs_key(t: Thread, ignored: Sequence[str]) -> Tuple:
    """What a run's decoys depend on besides the seed: its flags and derived entries without the ignored flags
    (seeds, output names, -nstruct), the executable's path and the input paths (their content counts)."""
    skip = set(ignored) | {"[executable]"} | set(FILE_FLAGS)
    return tuple(sorted((k, v) for k, v in t.flags.items() if k not in skip))


def seed_report(conds: Sequence[Condition], ignored: Sequence[str] = IGNORED
                ) -> Tuple[List[str], List[str], List[str]]:
    """(report lines, duplicates, notes): the seed and build of every run, then the seeds used twice under
    -constant_seed.  Two runs, or two starts of one run, with the same seed and the same inputs write the
    same decoys: a duplicate (an error).  The same seed with other inputs is a note (a matched-seed
    design)."""
    labels = {ident: lab for ident, (lab, _b) in build_labels(conds).items()}
    lines = ["", "seed and build of every run (the seed is a run's identity; it is never left out):"]
    width = max((len(_run_name(c, t)) for c in conds for t in c.threads), default=10)
    entries = []
    for c in conds:
        for t in c.threads:
            for k, sd in enumerate(t.seeds or [t.seed]):
                b = t.builds[k] if k < len(t.builds) else Build()
                start = f" (start {k + 1})" if k else ""
                lines.append(f"  {(_run_name(c, t) + start).ljust(width)}  seed {sd.describe()}  "
                             f"build {labels.get(b.identity, '?') if b.known else '?'}")
                if sd.constant and sd.value is not None:
                    entries.append((c, t, k, sd))
    by_value: Dict[int, list] = {}
    for e in entries:
        by_value.setdefault(e[3].value, []).append(e)
    duplicates, notes = [], []
    for value, items in sorted(by_value.items()):
        if len(items) < 2:
            continue
        groups: Dict[Tuple, list] = {}
        for e in items:
            groups.setdefault(_inputs_key(e[1], ignored), []).append(e)
        for es in groups.values():
            if len(es) < 2:
                continue
            names = [_run_name(c, t) + (f" (start {k + 1})" if k or len(t.seeds) > 1 else "")
                     for c, t, k, _sd in es]
            restart = len({id(e[1]) for e in es}) < len(es)
            more = " ..." if len(names) > 6 else ""
            duplicates.append(f"seed {value} under -constant_seed with the same inputs in {len(es)} runs: "
                              f"{', '.join(names[:6])}{more} - they write the same decoys (duplicates)"
                              + ("; a restart under -constant_seed starts the random stream again"
                                 if restart else ""))
        if len(groups) > 1:
            names = sorted({_run_name(c, t) for c, t, _k, _sd in items})
            more = " ..." if len(names) > 4 else ""
            notes.append(f"seed {value} is shared by runs whose inputs or build differ "
                         f"({', '.join(names[:4])}{more}): not duplicates (a matched-seed design)")
    if duplicates:
        lines += ["", f"DUPLICATE SEEDS (error, {len(duplicates)}):"] + [f"  {d}" for d in duplicates]
    if notes:
        lines += ["", "seeds shared by runs whose inputs or build differ:"] + [f"  {n}" for n in notes]
    return lines, duplicates, notes


@dataclass
class AuditResult:
    conditions: List[Condition]
    pairs: List[PairResult]
    problems: List[str]
    lines: List[str]
    duplicates: List[str] = field(default_factory=list)  # runs with the same seed and inputs (errors)


def audit(roots: Sequence[str], pairs: Sequence[str] = (), pair_suffix: str = "", controlled: str = "",
          allowed: Sequence[str] = (), ignored: Sequence[str] = (), full_hash: bool = True,
          recursive: bool = True, show_all: bool = False, per_condition: bool = False,
          exe_pattern: str = "rna_denovo") -> AuditResult:
    """Collect every condition under ``roots``, compare the flags across them and check the pairs."""
    ranges = parse_residues(controlled) if controlled.strip() else None
    ign = tuple(dict.fromkeys(list(IGNORED) + [flag_key(f) for f in ignored]))
    allow = tuple(dict.fromkeys(list(PAIR_FLAGS) + [flag_key(f) for f in allowed]))
    found = find_conditions(roots, recursive, exe_pattern)
    conds = [collect(p, lab, ranges, full_hash, recursive, exe_pattern, ign) for lab, p in found]
    used = [c for c in conds if c.threads]
    lines = [f"{len(used)} condition(s) with a logged rna_denovo command line, "
             f"{len(conds) - len(used)} without"]
    given = {os.path.normpath(r) for r in roots}
    problems: List[str] = [f"{c.label}: {c.notes[0]}" for c in conds if not c.threads and
                           os.path.normpath(c.path) in given]
    for c in used:
        files = sorted({os.path.relpath(f, c.path) for t in c.threads for f in t.command.flag_files})
        lines.append(f"  {c.label}: {len(c.threads)} run(s), seeds {seeds_text(c.threads)}"
                     + (f", flags files {', '.join(files)}" if files else ""))
    lines += provenance_lines(used)
    seed_lines, duplicates, _notes = seed_report(used, ign)
    lines += seed_lines
    problems += [f"duplicate {d}" for d in duplicates]
    labels = [c.label for c in used]
    pair_list, bad = pairs_from(pairs, pair_suffix, labels)
    problems += bad
    keys = sorted({k for c in used for k in c.flags})
    varying, presence, renamed = [], [], []
    for k in keys:
        vals = [c.flags.get(k) for c in used]
        real = {v for v in vals if v is not None}
        if len(real) > 1:
            (renamed if same_content(k, used) else varying).append(k)
        elif None in vals and real:
            presence.append(k)
    lines += ["", f"{len(keys)} flags and derived entries: {len(varying)} differ in value between "
                  f"conditions, {len(presence)} only by being present or absent"]
    if renamed:
        lines.append("same file content under other names: " + ", ".join(renamed))
    if varying:
        lines.append("values that differ (they can confound a comparison):")
        for k in varying:
            groups: Dict[str, List[str]] = {}
            for c in used:
                if k in c.flags:
                    groups.setdefault(c.flags[k], []).append(c.label)
            lines.append(f"  {k}: {len(groups)} values")
            for v, who in list(groups.items())[:6]:
                lines.append(f"      {_short(v, 70)}  ({', '.join(who[:4])}{' ...' if len(who) > 4 else ''})")
            if len(groups) > 6:
                lines.append(f"      ... {len(groups) - 6} more")
    if presence:
        lines.append("present in some conditions only: " + ", ".join(presence))
    if show_all:
        lines.append("every flag:")
        for k in keys:
            lines.append(f"  {k}: " + "; ".join(f"{c.label}={_short(c.flags.get(k), 40)}" for c in used))
    if per_condition:
        for c in used:
            lines.append(f"\n  {c.label} ({c.path})")
            for k in sorted(c.flags):
                if k.startswith("[") or k in FILE_FLAGS:
                    lines.append(f"      {k:34s} {_short(c.flags[k], 80)}")
    noted = [(c.label, n) for c in conds for n in c.notes]
    if noted:
        lines += ["", f"provenance notes ({len(noted)}):"]
        lines += [f"  {lab}: {n}" for lab, n in noted[:40]]
        if len(noted) > 40:
            lines.append(f"  ... and {len(noted) - 40} more")
    results = []
    by_label = {c.label: c for c in used}
    if pair_list:
        lines += ["", f"{len(pair_list)} pair(s), checked flag by flag:"]
    for ctl, trt in pair_list:
        r = compare(by_label[ctl], by_label[trt], allow, ign)
        results.append(r)
        fa, fb = by_label[ctl].flags, by_label[trt].flags
        lines += ["", f"  {ctl}  vs  {trt}"]
        for k in ("[controlled paired in secstruct]", "[effective condition]"):
            if k in fa or k in fb:
                lines.append(f"    {k[1:-1]}: {fa.get(k, '?')}  vs  {fb.get(k, '?')}")
        if r.expected:
            lines.append("    differences expected for the pairing: " + ", ".join(r.expected))
        if r.clean:
            lines.append("    CLEAN: the only differences are the secondary structure and the pairing flags")
        else:
            lines.append(f"    CONFOUNDED by {len(r.other)} other difference(s):")
            for k in r.other:
                lines += _diff_lines(k, fa.get(k), fb.get(k))
            lines.append("    Treat the two as separate conditions, not as a controlled comparison.")
    if problems:
        lines += ["", "problems:"] + [f"  {p}" for p in problems]
    return AuditResult(conds, results, problems, lines, duplicates)


def write_matrix(path: str, conds: Sequence[Condition]) -> None:
    """The flags of every condition as a table: one row per flag, one column per condition."""
    used = [c for c in conds if c.threads]
    keys = sorted({k for c in used for k in c.flags})
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(["flag"] + [c.label for c in used]) + "\n")
        fh.write("\t".join(["[runs]"] + [str(len(c.threads)) for c in used]) + "\n")
        fh.write("\t".join(["[seeds]"] + [silent_scores.tsv_field(seeds_text(c.threads)) for c in used])
                 + "\n")
        for k in keys:
            fields = [k] + [c.flags.get(k, "") for c in used]
            fh.write("\t".join(silent_scores.tsv_field(x) for x in fields) + "\n")


RUN_COLUMNS = ("condition", "log", "start", "seed", "constant_seed", "jran", "seed_source", "build",
               "rosetta_version", "commit", "repo", "date", "extras", "provenance_lines", "executable")


def write_runs(path: str, conds: Sequence[Condition]) -> None:
    """One row per start of every run: its seed (the value Rosetta used, -constant_seed, -jran as given, where
    the seed was read) and its build (label, parsed fields, the raw provenance lines)."""
    used = [c for c in conds if c.threads]
    labels = {ident: lab for ident, (lab, _b) in build_labels(used).items()}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(RUN_COLUMNS) + "\n")
        for c in used:
            for t in c.threads:
                for k, sd in enumerate(t.seeds or [t.seed]):
                    b = t.builds[k] if k < len(t.builds) else Build()
                    const = "" if sd.constant is None else ("yes" if sd.constant else "no")
                    fields = [c.label, os.path.relpath(t.log, c.path), k + 1,
                              "" if sd.value is None else sd.value, const, sd.jran, sd.source,
                              labels.get(b.identity, "") if b.known else "", b.version,
                              b.fields.get("commit", ""), b.fields.get("repo", ""), b.fields.get("date", ""),
                              b.extras, " | ".join(b.lines), t.command.exe]
                    fh.write("\t".join(silent_scores.tsv_field(x) for x in fields) + "\n")


# ---------------------------------------------------------------------------
# command line: mdatk rosetta scores | audit
# ---------------------------------------------------------------------------


def add_commands(sub) -> list:
    s = sub.add_parser("scores", help="SCORE terms of silent files by column name: column audit, cuts, decoy "
                                      "table, extracted-model check")
    s.add_argument("silent", nargs="+", help="silent files, or folders holding *.out files")
    s.add_argument("--term", action="append", default=[], help="term to audit (repeatable; default score and "
                   "linear_chainbreak)")
    s.add_argument("--max", action="append", default=[], metavar="TERM=VALUE",
                   help="count the decoys with TERM <= VALUE (repeatable), e.g. linear_chainbreak=5")
    s.add_argument("--tsv", help="write the decoy table (run, tag, pdb, every term)")
    s.add_argument("--extracted", default="", help="folder of extracted models <run>/<tag>.pdb: compare them "
                   "with the SCORE lines and fill the table's pdb column")
    s.add_argument("--root", default="", help="folder the table's pdb paths are relative to (default: the "
                   "current folder)")
    s.add_argument("--legacy-index", type=int, help="also say which term a fixed index of the split SCORE "
                   "line holds (an old filter's column)")
    a = sub.add_parser("audit", help="what each rna_denovo run was given: logged command lines with @flags "
                                     "files expanded, inputs fingerprinted, conditions compared in pairs")
    a.add_argument("roots", nargs="+", help="condition folders (a FARFAR2 step folder, a campaign's "
                   "experiment folder), or folders holding them")
    a.add_argument("--pair", action="append", default=[], metavar="CONTROL=TREATMENT",
                   help="two conditions that should differ only in the pairing (repeatable)")
    a.add_argument("--pair-suffix", default="", help="pair X with X<SUFFIX> (e.g. _nointernalpairs)")
    a.add_argument("--controlled", default="", help="residues whose pairing is the variable, e.g. "
                   "'A:22-24,A:35-37' (numbers as in the FASTA)")
    a.add_argument("--allow", action="append", default=[], metavar="FLAG",
                   help="another flag that may differ in a pair (repeatable)")
    a.add_argument("--ignore", action="append", default=[], metavar="FLAG",
                   help="another flag to ignore everywhere (repeatable)")
    a.add_argument("--fast-hash", action="store_true", help="fingerprint large files from three 1 MiB "
                   "samples instead of their whole content")
    a.add_argument("--shallow", action="store_true", help="look for logs only in the condition folders "
                   "themselves, not below them")
    a.add_argument("--exe", default="rna_denovo", help="text in the executable's name of the runs to read "
                   "(default rna_denovo)")
    a.add_argument("--show-all", action="store_true", help="list every flag of every condition")
    a.add_argument("--per-condition", action="store_true",
                   help="list each condition's inputs and derived entries")
    a.add_argument("--output", "-o", help="also write the report to this file")
    a.add_argument("--tsv", help="write the flags of every condition as a table")
    a.add_argument("--runs-tsv", help="write the seed and Rosetta build of every run (one row per start)")
    a.add_argument("--strict", action="store_true", help="exit with status 1 when a pair is confounded or "
                   "cannot be resolved, a condition given has no command line, or two runs are duplicates "
                   "(the same seed under -constant_seed with the same inputs)")
    return [s, a]


def run(args, stdout=None, stderr=None) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    if args.rosetta_command == "scores":
        maxima = parse_maxima(args.max)
        lines, problems, failures, silents = score_report(
            args.silent, args.term or ["score", "linear_chainbreak"], maxima, args.extracted,
            args.legacy_index)
        if not any(s.rows for s in silents):
            print("ERROR: no SCORE data lines in " + ", ".join(args.silent), file=err)
            for p in problems[:5]:
                print(f"  {p}", file=err)
            return 1
        for ln in lines:
            print(ln, file=out)
        if args.tsv:
            root = os.path.abspath(args.root or os.getcwd())

            def pdb_of(label, tag):
                cand = os.path.join(args.extracted, label, f"{tag}.pdb") if args.extracted else ""
                return os.path.relpath(os.path.abspath(cand), root).replace(os.sep, "/") \
                    if cand and os.path.isfile(cand) else ""
            n = silent_scores.write_table(args.tsv, silents, pdb_of)
            print(f"wrote {n} decoys to {args.tsv}", file=out)
        # problems go into the report itself, not only to standard error
        for title, items in (("problem(s)", problems), ("failure(s)", failures)):
            if items:
                print(f"\n{len(items)} {title}:", file=out)
                for p in items[:20]:
                    print(f"  {p}", file=out)
                if len(items) > 20:
                    print(f"  ... and {len(items) - 20} more", file=out)
        for p in failures:
            print(f"ERROR: {p}", file=err)
        return 1 if failures else 0
    if args.rosetta_command == "audit":
        res = audit(args.roots, args.pair, args.pair_suffix, args.controlled, args.allow, args.ignore,
                    not args.fast_hash, not args.shallow, args.show_all, args.per_condition, args.exe)
        text = "\n".join(res.lines) + "\n"
        out.write(text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        if args.tsv:
            write_matrix(args.tsv, res.conditions)
        if args.runs_tsv:
            write_runs(args.runs_tsv, res.conditions)
        for d in res.duplicates:  # an error in the report, and said where a script's log shows it
            print(f"ERROR: duplicate {d}", file=err)
        if not any(c.threads for c in res.conditions):
            print("ERROR: no condition folder has a logged rna_denovo command line", file=err)
            return 1
        if args.strict and (res.problems or any(not r.clean for r in res.pairs)):
            return 1
        return 0
    raise ValueError(f"unknown rosetta command {args.rosetta_command!r}")
