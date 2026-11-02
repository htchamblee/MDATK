"""Decoy ensembles: which register each decoy formed, under which build condition, whether its backbone
closed and how much of its interface sits in the density - and the comparison of two such ensembles.

Residues are keyed by ``(chain, resseq, icode)`` throughout and read with :mod:`mdatk.pdbtools` (first
model, first alternate location, ChimeraX repair), never by residue number alone: two chains numbered
from 1 must not collapse onto each other.

Registers
    A register is a named set of defining base pairs, ``'pres=B:22-B:37, B:23-B:36, B:24-B:35'``.  A pair
    is *formed* when the Watson-Crick N1/N3 distance (purine N1, pyrimidine N3) is below the cut (3.5 Å),
    and *approached* when it is formed or sits in register without the hydrogen bond: N1-N3 below 6 Å
    with the C1'-C1' distance between 8 and 13 Å (a pair at textbook geometry whose bases have not yet
    closed, which refinement or a short MD would close).
Numbering
    Register pairs, regions and calibration ranges are written in the numbering of a reference
    construct.  With a reference sequence (typed, a FASTA file such as the FARFAR2 step's, or the plan's
    sequence) each chain of a decoy is located in it by its sequence (bases from MDATK's residue-name
    parser, else from a diagnostic base atom: O6 G, N6 A, N4 C, O4 U), and its residues take the
    reference's numbers; explicit per-chain offsets do the same by hand.  The base each register residue
    has in the reference is checked in every decoy, so a numbering slip cannot pass silently.
Build condition
    A register can be an input or an outcome.  The condition of every group of decoys is read from a
    Rosetta log's command line (``core.init: command:``, ``@flags`` files expanded): the secondary
    structure it names (``-secstruct_file`` or ``-secstruct``) and ``-obligate_pair(_explicit)``; failing
    a log, from a ``*.secstruct`` beside the decoys, then from a given secondary structure.  Per register:
    FORCED (every defining pair imposed), UNFORCED (none of its residues paired by the input), PARTIAL;
    a group without any source is UNKNOWN.  Groups are never pooled across conditions.
Scores
    Rosetta score terms are looked up by name in the decoy table of the FARFAR2 models step (TSV: ``run``,
    ``tag``, ``pdb`` relative to the plan folder, then every SCORE term), else read from the pose energies
    table inside the PDB, else from a ``denovo_<100000 + 10 * score>_`` file name.  Score deciles are only
    formed within one weight set.
Closure
    A score term below a cut (default ``linear_chainbreak < 5``), or the geometric check: every O3'(i)-P(i+1)
    of consecutive nucleotides in one chain segment at most 2.0 Å.
Map occupancy
    The share of interface / loop heavy atoms above a contour calibrated on regions held fixed: the 5th
    percentile of the density at their atoms (per decoy, or one common level over both arms when the
    calibration regions did not move).  The map is read by :mod:`mdatk.density` and sampled by trilinear
    interpolation; ``nearest`` reproduces the nearest-voxel sampling of the scripts this replaces (at 0.8
    Å voxels the two agree closely, but only nearest gives their numbers exactly).  Atoms outside the map
    count as not in density and are reported.

The comparison (:func:`compare_arms`) reports numbers: rates with exact intervals, Fisher p-values, a
closure-threshold sweep (a sensitivity display, not a set of tests) and the rates the second arm would
need for a difference to be detected with a stated power.  It draws no conclusion.
"""

from __future__ import annotations

import csv
import fnmatch
import glob
import math
import os
import re
import shlex
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import ensemble_stats as stats
from . import pdbtools
from .pdbtools import Residue, Structure

ResKey = Tuple[str, int, str]  # (chain, resseq, icode)
Range = Tuple[Optional[str], Optional[int], Optional[int]]  # (chain, start, end); chain None: any chain

WC_ATOM = {"G": "N1", "A": "N1", "C": "N3", "U": "N3", "T": "N3"}  # Watson-Crick N1/N3 across a pair
BASE_ATOMS = (("O6", "G"), ("N6", "A"), ("N4", "C"), ("O4", "U"))  # one atom identifies each base
PAIR_CUT = 3.5  # Å: a formed Watson-Crick N1-N3 hydrogen bond
NEAR_N = 6.0  # Å: N1-N3 of a pair in register but not bonded ...
NEAR_C1 = (8.0, 13.0)  # ... with C1'-C1' in this window (about 10.4 Å in a canonical pair)
CLOSURE_TERM, CLOSURE_CUT = "linear_chainbreak", 5.0
MAX_O3P = 2.0  # Å: O3'-P longer than this is a chain break (the bond is 1.6 Å)
CONTOUR_PERCENTILE = 5.0
DEFAULT_SWEEP = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0, math.inf)
GEOMETRY_SWEEP = (1.7, 1.8, 2.0, 2.5, 3.0, 5.0, math.inf)
VERDICTS = ("FORCED", "UNFORCED", "PARTIAL")
TABLE_ID_COLUMNS = ("run", "tag", "pdb")
_VIRTUAL = frozenset({"VRT", "XXX"})  # Rosetta virtual residues (ORIG / X / Y atoms)
_FNAME_SCORE = re.compile(r"denovo_(\d+)_")
_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_RES = r"([A-Za-z0-9]?):(-?\d+)([A-Za-z]?)"
_PAIR_RE = re.compile(rf"^{_RES}-{_RES}$")
_RUN_ID = re.compile(r"(?:^|[-_./])(r\d+)(?=$|[-_.])")


class EnsembleError(ValueError):
    """Unusable input or settings."""


# --- residue keys, pairs and selections -------------------------------------------------------------------

def key_text(key: ResKey) -> str:
    return f"{key[0]}:{key[1]}{key[2]}"


def residue_key(res: Residue) -> ResKey:
    return (res.chain, res.resseq, res.icode)


def parse_pairs(text: str) -> List[Tuple[ResKey, ResKey]]:
    """'A:22-A:37, A:23-A:36' -> [(('A', 22, ''), ('A', 37, '')), ...] (chain-qualified, ':22' = blank)."""
    out = []
    for item in re.split(r"[,;\s]+", (text or "").strip()):
        if not item:
            continue
        m = _PAIR_RE.match(item)
        if not m:
            raise EnsembleError(f"cannot read the pair {item!r} (expected chain:residue-chain:residue, "
                                f"e.g. A:22-A:37)")
        c1, r1, i1, c2, r2, i2 = m.groups()
        a, b = (c1, int(r1), i1), (c2, int(r2), i2)
        if a == b:
            raise EnsembleError(f"the pair {item!r} pairs a residue with itself")
        out.append((a, b))
    return out


@dataclass
class Register:
    """A named set of defining base pairs."""
    name: str
    pairs: List[Tuple[ResKey, ResKey]]

    @property
    def residues(self) -> List[ResKey]:
        return list(dict.fromkeys(k for p in self.pairs for k in p))

    def text(self) -> str:
        return ", ".join(f"{key_text(a)}-{key_text(b)}" for a, b in self.pairs)


def parse_register(text: str, default_name: str = "register") -> Register:
    """'NAME=A:22-A:37, A:23-A:36' (the name is optional: ``default_name``)."""
    name, sep, pairs = (text or "").partition("=")
    if not sep:
        name, pairs = default_name, text
    name = name.strip()
    if not _NAME_OK.match(name):
        raise EnsembleError(f"register name {name!r}: use letters, digits, '.', '_', '-' and '+'")
    reg = Register(name, parse_pairs(pairs))
    if not reg.pairs:
        raise EnsembleError(f"register {name!r} has no defining pairs")
    return reg


def check_registers(registers: Sequence[Register]) -> None:
    names = [r.name for r in registers]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise EnsembleError(f"register names used twice: {', '.join(dupes)}")
    for reg in registers:
        seen = Counter(k for p in reg.pairs for k in p)
        twice = [key_text(k) for k, n in seen.items() if n > 1]
        if twice:
            raise EnsembleError(f"register {reg.name}: {', '.join(twice)} is in more than one defining pair")


def _group_residues(group: Any) -> Tuple[str, str]:
    if isinstance(group, dict):
        return str(group.get("residues", "")), str(group.get("selection", ""))
    return str(getattr(group, "residues", "") or ""), str(getattr(group, "selection", "") or "")


def group_ranges(name: str, groups: Sequence[Any]) -> str:
    """A project residue group as selection text of this module (unqualified ranges as ``*:a-b``)."""
    for g in groups or ():
        gname = g.get("name") if isinstance(g, dict) else getattr(g, "name", "")
        if gname == name:
            residues, selection = _group_residues(g)
            if not residues.strip():
                raise EnsembleError(f"residue group {name!r} has no residue ranges"
                                    + (" (only an MDAnalysis selection)" if selection.strip() else ""))
            items = []
            for tok in re.split(r"[,;\s]+", residues.strip()):
                if not tok:
                    continue
                if re.fullmatch(r"-?\d+(?:-(-?\d+))?", tok):
                    items.append("*:" + tok)  # the analysis' residue numbers: any chain
                else:
                    items.append(tok)
            return ",".join(items)
    raise EnsembleError(f"no residue group named {name!r}")


def parse_selection(text: str, groups: Sequence[Any] = ()) -> List[Range]:
    """Residue ranges in :func:`mdatk.pdbtools.parse_ranges` syntax (``'A:1-16,B:17-44'``, ``'C'`` for a
    whole chain), plus ``*:12-30`` (these numbers in any chain) and ``group:NAME`` (a project residue
    group)."""
    out: List[Range] = []
    for tok in re.split(r"[,;\s]+", (text or "").strip()):
        if not tok:
            continue
        if tok.startswith("group:"):
            out += parse_selection(group_ranges(tok[6:], groups))
            continue
        m = re.fullmatch(r"\*:(-?\d+)(?:-(-?\d+))?", tok)
        if m:
            a = int(m.group(1))
            out.append((None, a, int(m.group(2)) if m.group(2) else a))
            continue
        try:
            out += pdbtools.parse_ranges(tok)
        except pdbtools.PDBError as exc:
            raise EnsembleError(str(exc)) from None
    return out


def selection_text(sel: Sequence[Range]) -> str:
    """Canonical text of a selection (what the planner passes to the command line)."""
    items = []
    for chain, a, b in sel:
        c = "*" if chain is None else chain
        if a is None:
            items.append(c)
        else:
            items.append(f"{c}:{a}" if a == b else f"{c}:{a}-{b}")
    return ",".join(items)


def in_selection(key: ResKey, sel: Sequence[Range]) -> bool:
    chain, resseq = key[0], key[1]
    for c, a, b in sel:
        if (c is None or c == chain) and (a is None or a <= resseq <= b):
            return True
    return False


def selection_residues(sel: Sequence[Range]) -> List[Tuple[Optional[str], int]]:
    """The residues a selection names explicitly (whole chains name none)."""
    out = []
    for c, a, b in sel:
        if a is not None:
            out += [(c, r) for r in range(min(a, b), max(a, b) + 1)]
    return list(dict.fromkeys(out))


# --- bases ------------------------------------------------------------------------------------------------

def residue_base(res: Residue) -> str:
    """A / C / G / U (T) of a nucleotide residue: MDATK's name parser, else one diagnostic base atom (for
    names such as '0', 'LG0' or 'A0' that no scheme defines); '' for anything else."""
    info = pdbtools.parse_residue_name(res.name)
    names = {a.name.replace("*", "'") for a in res.atoms}
    sugar = "C1'" in names or "C4'" in names
    if info is not None and (sugar or "P" in names):
        return info.base
    if not sugar or res.name.strip().upper() in _VIRTUAL:
        return ""
    for atom, base in BASE_ATOMS:
        if atom in names:
            return base
    return ""


def is_hydrogen(atom: pdbtools.Atom) -> bool:
    """From the element column when the file has one; only without it from the name ('1H5'', 'H2'')."""
    if atom.element:
        return atom.element.strip().upper() in ("H", "D")
    return re.sub(r"^\d+", "", atom.name.strip()).upper().startswith("H")


# --- the reference construct -------------------------------------------------------------------------------

@dataclass
class Reference:
    """A numbered reference sequence: per position the chain (None: any chain) and residue number."""
    chains: List[Optional[str]]
    numbers: List[int]
    bases: str

    def __len__(self) -> int:
        return len(self.bases)

    def position(self, key: Tuple[Optional[str], int]) -> Optional[int]:
        """0-based position of (chain, resseq); None when absent or ambiguous."""
        chain, resseq = key[0], key[1]
        hits = [i for i, (c, n) in enumerate(zip(self.chains, self.numbers)) if n == resseq]
        exact = [i for i in hits if self.chains[i] == chain]
        if len(exact) == 1:
            return exact[0]
        loose = [i for i in hits if self.chains[i] is None or chain is None or chain == ""]
        return loose[0] if len(loose) == 1 and not exact else None

    def base_at(self, key: ResKey) -> Optional[str]:
        i = self.position(key)
        return self.bases[i] if i is not None else None

    def key(self, i: int, model_chain: str) -> ResKey:
        chain = self.chains[i]
        return (model_chain if chain is None else chain, self.numbers[i], "")

    def describe(self) -> str:
        if all(c is None for c in self.chains):
            return f"{len(self)} nt, numbered 1-{len(self)}"
        parts = []
        for c in dict.fromkeys(self.chains):
            nums = [n for cc, n in zip(self.chains, self.numbers) if cc == c]
            parts.append(f"{c or '*'}:{nums[0]}-{nums[-1]}")
        return f"{len(self)} nt, " + " ".join(parts)


def _clean_bases(seq: str) -> str:
    seq = re.sub(r"[\s\d&+,.*-]+", "", seq).upper().replace("T", "U")
    bad = sorted(set(seq) - set("ACGUN"))
    if bad:
        raise EnsembleError(f"reference sequence: unexpected letters {''.join(bad)}")
    return seq


def parse_reference(text: str) -> Reference:
    """A reference construct from sequence text.

    * FASTA: ``>name A:1-16`` headers (Rosetta's chain:range tags, several per header allowed) number the
      record's residues when the tags cover it; otherwise its residues continue the numbering of the
      records before it, chain unknown.
    * Otherwise one or more sequences, optionally ``A:GGGC`` / ``A: GGGC``; strands separated by spaces,
      ``&``, ``+`` or ``,`` are numbered continuously from 1, and without a chain prefix the chain is
      left open (a decoy keeps its own chain ids).
    """
    text = (text or "").strip()
    chains: List[Optional[str]] = []
    numbers: List[int] = []
    bases: List[str] = []
    nxt = 1
    if text.startswith(">"):
        records: List[Tuple[str, List[str]]] = []
        for line in text.splitlines():
            if line.startswith(">"):
                records.append((line, []))
            elif records and line.strip():
                records[-1][1].append(line.strip())
        for header, lines in records:
            seq = _clean_bases("".join(lines))
            tags = [(c, int(a), int(b)) for c, a, b in
                    re.findall(r"(?:^|[\s>])([A-Za-z0-9]):(-?\d+)-(-?\d+)(?=\s|$)", header)]
            if tags and sum(abs(b - a) + 1 for _, a, b in tags) == len(seq):
                for c, a, b in tags:
                    for r in range(a, b + 1) if b >= a else range(a, b - 1, -1):
                        chains.append(c)
                        numbers.append(r)
                nxt = max(numbers) + 1
            else:
                for _ in seq:
                    chains.append(None)
                    numbers.append(nxt)
                    nxt += 1
            bases.append(seq)
    else:
        prefix: Optional[str] = None  # 'A:' before the next strand
        for tok in re.split(r"[\s&+,]+", text):
            if not tok:
                continue
            m = re.fullmatch(r"([A-Za-z0-9]):([A-Za-z]*)", tok)
            if m:
                prefix, tok = m.group(1), m.group(2)
                if not tok:
                    continue
            seq = _clean_bases(tok)
            for _ in seq:
                chains.append(prefix)
                numbers.append(nxt)
                nxt += 1
            bases.append(seq)
            prefix = None  # a strand without a prefix of its own leaves the chain open
    ref = Reference(chains, numbers, "".join(bases))
    if not len(ref):
        raise EnsembleError("the reference sequence is empty")
    if len(set(zip(chains, numbers))) != len(ref):
        raise EnsembleError("the reference numbers a residue twice (check the FASTA chain:range tags)")
    return ref


def read_reference(path: str) -> Reference:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_reference(fh.read())


def reference_from_model(model: Any) -> Optional[Reference]:
    """The planner's sequence (``mdatk.workflows.rna.RNAModel``: chains numbered continuously)."""
    chains = getattr(model, "chains", None)
    if not chains:
        return None
    ch: List[Optional[str]] = []
    nums: List[int] = []
    for c in chains:
        for k in range(len(c.sequence)):
            ch.append(c.id)
            nums.append(c.start + k)
    return Reference(ch, nums, "".join(c.sequence for c in chains))


def reference_text(ref: Reference) -> str:
    """FASTA text that :func:`parse_reference` reads back as ``ref``."""
    if all(c is None for c in ref.chains) and ref.numbers == list(range(1, len(ref) + 1)):
        return ">reference\n" + ref.bases + "\n"
    lines, start = [], 0
    for i in range(1, len(ref) + 1):
        if i == len(ref) or ref.chains[i] != ref.chains[start] or ref.numbers[i] != ref.numbers[i - 1] + 1:
            c = ref.chains[start] or "A"
            lines += [f">reference {c}:{ref.numbers[start]}-{ref.numbers[i - 1]}", ref.bases[start:i]]
            start = i
    return "\n".join(lines) + "\n"


# --- mapping a decoy's numbering onto the reference ------------------------------------------------------

def parse_offsets(text: str) -> Dict[str, int]:
    """'B:-367,C:44' -> {'B': -367, 'C': 44}."""
    out: Dict[str, int] = {}
    for tok in re.split(r"[,;\s]+", (text or "").strip()):
        if not tok:
            continue
        m = re.fullmatch(r"([A-Za-z0-9]?):([+-]?\d+)", tok)
        if not m:
            raise EnsembleError(f"cannot read the offset {tok!r} (expected chain:offset, e.g. B:-367)")
        out[m.group(1)] = int(m.group(2))
    return out


def nucleotide_chains(structure: Structure) -> "OrderedDict[str, List[Tuple[Residue, str]]]":
    """Chain id -> its nucleotides with their bases, in file order (virtual residues left out)."""
    out: "OrderedDict[str, List[Tuple[Residue, str]]]" = OrderedDict()
    for res in structure.residues():
        base = residue_base(res)
        if base:
            out.setdefault(res.chain, []).append((res, base))
    return out


def _locate(items: Sequence[Tuple[Residue, str]], ref: Reference, chain: str
            ) -> Tuple[Optional[List[int]], str]:
    """Reference positions of a chain's residues, from its sequence; (None, reason) when not unique."""
    nums = [r.resseq for r, _ in items]
    numbered = all(not r.icode for r, _ in items) and all(b > a for a, b in zip(nums, nums[1:]))
    if numbered:  # numbering gaps become wildcards (missing residues)
        length = nums[-1] - nums[0] + 1
        pattern = ["?"] * length
        for r, base in items:
            pattern[r.resseq - nums[0]] = base.replace("T", "U")
        offsets = [r.resseq - nums[0] for r, _ in items]
    else:
        pattern = [b.replace("T", "U") for _, b in items]
        length = len(pattern)
        offsets = list(range(length))
    hits = []
    for i in range(len(ref) - length + 1):
        if ref.chains[i] is not None and any(ref.chains[i + j] != ref.chains[i] for j in range(length)):
            continue
        if all(p in ("?", "N") or ref.bases[i + j] in (p, "N") for j, p in enumerate(pattern)):
            hits.append(i)
    if len(hits) > 1:
        own = [i for i in hits if ref.chains[i] == chain]
        if len(own) == 1:
            hits = own
    if len(hits) == 1:
        return [hits[0] + o for o in offsets], ""
    seq = "".join(pattern).replace("?", "-")
    if not hits:
        more = "..." if len(seq) > 30 else ""
        return None, f"its {len(items)}-nt sequence {seq[:30]}{more} is not in the reference"
    return None, f"its sequence occurs at {len(hits)} places in the reference"


def map_keys(structure: Structure, reference: Optional[Reference] = None,
             offsets: Optional[Dict[str, int]] = None) -> Tuple[Dict[ResKey, ResKey], List[str]]:
    """Model residue key -> key in the reference numbering, and notes (one per chain).  Explicit offsets
    win; chains the reference cannot place keep their own numbers (and are reported)."""
    keymap: Dict[ResKey, ResKey] = {}
    notes: List[str] = []
    for res in structure.residues():
        keymap[residue_key(res)] = residue_key(res)
    for chain, items in nucleotide_chains(structure).items():
        label = chain or "_"
        first, last = items[0][0], items[-1][0]
        span = f"{first.resseq}{first.icode}-{last.resseq}{last.icode}"
        if offsets and chain in offsets:
            off = offsets[chain]
            for res, _ in items:
                keymap[residue_key(res)] = (chain, res.resseq + off, res.icode)
            notes.append(f"chain {label}: {len(items)} nt at {span}, offset {off:+d} as given")
        elif reference is not None:
            pos, why = _locate(items, reference, chain)
            if pos is None:
                notes.append(f"chain {label}: {why}; its numbering is kept")
                continue
            for (res, _), i in zip(items, pos):
                keymap[residue_key(res)] = reference.key(i, chain)
            a, b = reference.key(pos[0], chain), reference.key(pos[-1], chain)
            off = a[1] - first.resseq
            moved = a[0] != chain or off
            notes.append(f"chain {label}: {len(items)} nt at {span} matches the reference "
                         f"{a[0] or '_'}:{a[1]}-{b[1]}" + (f" (offset {off:+d})" if moved else ""))
    return keymap, notes


def keyed_residues(structure: Structure, keymap: Dict[ResKey, ResKey]
                   ) -> Tuple[Dict[ResKey, Residue], List[str]]:
    """Mapped key -> residue; two residues mapped onto one key are reported (the first is kept)."""
    out: Dict[ResKey, Residue] = {}
    clashes = []
    for res in structure.residues():
        k = keymap.get(residue_key(res), residue_key(res))
        if k in out:
            clashes.append(key_text(k))
            continue
        out[k] = res
    return out, clashes


# --- the decoy table (scores by name) -------------------------------------------------------------------

@dataclass
class DecoyTable:
    path: str
    columns: List[str]
    rows: List[Dict[str, str]]

    @property
    def terms(self) -> List[str]:
        return [c for c in self.columns if c not in TABLE_ID_COLUMNS]


def read_decoy_table(path: str) -> DecoyTable:
    """The decoy table: a header row, then one row per decoy; tab- or comma-separated (whitespace also
    read).  ``run``, ``tag`` and ``pdb`` are found case-insensitively; extra columns are kept."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n\r") for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise EnsembleError(f"{path}: the table is empty")
    head = lines[0]
    if "\t" in head:
        rows = list(csv.reader(lines, delimiter="\t"))
    elif "," in head:
        rows = list(csv.reader(lines))
    else:
        rows = [ln.split() for ln in lines]
    header = [h.strip() for h in rows[0]]
    canon = {h: h.lower() if h.lower() in TABLE_ID_COLUMNS else h for h in header}
    columns = [canon[h] for h in header]
    out = []
    for values in rows[1:]:
        values = [v.strip() for v in values] + [""] * (len(columns) - len(values))
        out.append(dict(zip(columns, values)))
    return DecoyTable(path, columns, out)


def number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_id(value: str) -> str:
    """'r01' from 'r01', 'apt1_04_farfar2-r01_20261015-1432.out' ...; the value itself otherwise."""
    m = _RUN_ID.search(value or "")
    return m.group(1) if m else (value or "")


class ScoreIndex:
    """Rows of decoy tables, found by a model's path.  A row's ``pdb`` is relative to the plan folder:
    ``root`` when given, else the parent of the table's folder (the table sits in a step folder), the
    table's folder, then the working folder.  Models whose path is in no row are matched by file name,
    then by ``tag`` against the file name without extension."""

    def __init__(self, tables: Sequence[DecoyTable] = (), root: str = ""):
        self.tables = list(tables)
        self.notes: List[str] = []
        self.by_path: Dict[str, Dict[str, str]] = {}
        self.by_name: Dict[str, List[Dict[str, str]]] = {}
        self.by_tag: Dict[str, List[Dict[str, str]]] = {}
        self.paths: List[Tuple[str, Dict[str, str]]] = []  # (absolute pdb path, row), table order
        self.roots: Dict[str, List[str]] = {}  # table path -> folders its relative paths may start from
        for t in self.tables:
            roots = [root] if root else []
            tdir = os.path.dirname(os.path.abspath(t.path))
            roots = list(dict.fromkeys(roots + [os.path.dirname(tdir), tdir, os.getcwd()]))
            self.roots[t.path] = roots
            for row in t.rows:
                row.setdefault("_table", t.path)
                tag = row.get("tag", "")
                if tag:
                    self.by_tag.setdefault(tag, []).append(row)
                pdb = row.get("pdb", "")
                if not pdb:
                    continue
                cands = [pdb] if os.path.isabs(pdb) else [os.path.join(r, pdb) for r in roots]
                cands = [os.path.normpath(os.path.abspath(c)) for c in cands]
                hit = next((c for c in cands if os.path.exists(c)), cands[0])
                for c in cands:
                    if c in self.by_path and self.by_path[c] is not row:
                        note = f"{pdb} is listed twice; the first row is used"
                        if note not in self.notes:
                            self.notes.append(note)
                        continue
                    self.by_path.setdefault(c, row)
                self.by_name.setdefault(os.path.basename(pdb), []).append(row)
                self.paths.append((hit, row))

    def __bool__(self) -> bool:
        return bool(self.tables)

    def lookup(self, path: str) -> Optional[Dict[str, str]]:
        p = os.path.normpath(os.path.abspath(path))
        row = self.by_path.get(p) or self.by_path.get(os.path.realpath(p))
        if row is not None:
            return row
        named = self.by_name.get(os.path.basename(path), [])
        if len(named) == 1:
            return named[0]
        tagged = self.by_tag.get(os.path.splitext(os.path.basename(path))[0], [])
        return tagged[0] if len(tagged) == 1 else None

    def models(self) -> List[str]:
        """The decoys the tables list with a PDB file."""
        return list(dict.fromkeys(p for p, _ in self.paths))

    def run_folder(self, row: Dict[str, str]) -> Optional[str]:
        """The folder of a row's run when ``run`` names a file (e.g. the silent file), else None: the
        Rosetta log of the run sits beside it."""
        run = row.get("run", "")
        if not run:
            return None
        for r in self.roots.get(row.get("_table", ""), [os.getcwd()]):
            for cand in (run, run + ".out"):
                p = cand if os.path.isabs(cand) else os.path.join(r, cand)
                if os.path.isfile(p):
                    return os.path.dirname(os.path.abspath(p))
        return None


def read_score_index(paths: Sequence[str], root: str = "") -> ScoreIndex:
    return ScoreIndex([read_decoy_table(p) for p in paths], root)


def embedded_score(text: str, path: str = "") -> Tuple[Optional[float], str]:
    """(total score, weights row) from a Rosetta PDB's pose energies table ('label' / 'weights' / 'pose'
    lines) or a 'score' line; else the score coded in a ``denovo_<100000 + 10 * score>_`` file name."""
    total, weights, label = None, "", None
    for ln in text.splitlines():
        if ln.startswith("label "):
            label = ln.split()[1:]
        elif ln.startswith("weights "):
            weights = " ".join(ln.split()[1:])
        elif ln.startswith("pose "):
            f = ln.split()[1:]
            try:
                if label and "total" in label and len(f) == len(label):
                    total = float(f[label.index("total")])
                elif f:
                    total = float(f[-1])
            except ValueError:
                pass
        elif ln[:6].lower() == "score " and total is None:
            try:
                total = float(ln.split()[1])
            except (IndexError, ValueError):
                pass
    if total is None and path:
        m = _FNAME_SCORE.search(os.path.basename(path))
        if m:
            total = (int(m.group(1)) - 100000) / 10.0
    return total, weights


# --- pair geometry and closure ---------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.dist(a, b)


def pair_distances(ri: Residue, rj: Residue, bi: str, bj: str) -> Tuple[Optional[float], Optional[float]]:
    """(Watson-Crick N1/N3 distance, C1'-C1' distance); None for a missing atom."""
    ci, cj = ri.coords(), rj.coords()
    ai, aj = ci.get(WC_ATOM.get(bi, "")), cj.get(WC_ATOM.get(bj, ""))
    wc = _dist(ai, aj) if ai is not None and aj is not None else None
    c1i, c1j = ci.get("C1'"), cj.get("C1'")
    c1 = _dist(c1i, c1j) if c1i is not None and c1j is not None else None
    return wc, c1


def backbone_closure(structure: Structure) -> Tuple[Optional[float], int, int]:
    """(longest O3'(i)-P(i+1), junctions measured, junctions without the atoms) over consecutive
    nucleotides of each chain segment (TER-delimited, same chain): never across chains or segments."""
    worst: Optional[float] = None
    measured = missing = 0
    for seg in structure.segments:
        nts = [r for r in seg if residue_base(r)]
        for a, b in zip(nts, nts[1:]):
            if a.chain != b.chain:
                continue
            o3, p = a.coords().get("O3'"), b.coords().get("P")
            if o3 is None or p is None:
                missing += 1
                continue
            d = _dist(o3, p)
            measured += 1
            worst = d if worst is None else max(worst, d)
    return worst, measured, missing


@dataclass
class Closure:
    """How a decoy counts as closed: a score term below ``cut``, or (``mode='geometry'``) every O3'-P below
    ``cut`` Å; ``mode='none'`` turns closure off."""
    mode: str = "term"
    term: str = CLOSURE_TERM
    cut: float = CLOSURE_CUT

    def describe(self, cut: Optional[float] = None) -> str:
        c = self.cut if cut is None else cut
        if self.mode == "geometry":
            return f"O3'-P < {c:g} Å" if math.isfinite(c) else "any O3'-P"
        if self.mode == "none":
            return "not tested"
        return f"{self.term} < {c:g}" if math.isfinite(c) else f"any {self.term}"

    def value(self, row: Optional[Dict[str, str]], structure: Optional[Structure]) -> Optional[float]:
        if self.mode == "geometry":
            return backbone_closure(structure)[0] if structure is not None else None
        if self.mode == "none" or row is None:
            return None
        return number(row.get(self.term))

    @staticmethod
    def closed(value: Optional[float], cut: float) -> Optional[bool]:
        if value is None or value != value:
            return None
        return value < cut


# --- density at atoms -------------------------------------------------------------------------------------

def load_map(path: str):
    from .density import load_map_cached
    return load_map_cached(path)


def map_values(dmap, xyz: np.ndarray, interpolation: str = "linear") -> np.ndarray:
    """Density at Cartesian points: trilinear ('linear') or the nearest voxel ('nearest'); NaN outside."""
    from .density import sample_map
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not len(xyz):
        return np.empty(0)
    order = 0 if interpolation == "nearest" else 1
    vals = sample_map(dmap, xyz, cval=np.nan, order=order)
    ijk = dmap.xyz_to_index(xyz)
    if order == 0:  # round as the scripts did; outside the grid after rounding is outside
        idx = np.rint(ijk)
        inside = np.all((idx >= 0) & (idx <= np.array(dmap.shape_xyz) - 1), axis=1)
    else:
        inside = np.all((ijk >= 0) & (ijk <= np.array(dmap.shape_xyz) - 1), axis=1)
    vals = np.where(inside, vals, np.nan)
    return vals


def heavy_atom_coords(residues: Dict[ResKey, Residue], sel: Sequence[Range]) -> np.ndarray:
    xyz = []
    for k, res in residues.items():
        if not in_selection(k, sel):
            continue
        seen = set()
        for a in res.atoms:  # first alternate location only
            if a.name in seen or is_hydrogen(a):
                continue
            seen.add(a.name)
            xyz.append(a.xyz)
    return np.array(xyz, dtype=np.float64).reshape(-1, 3)


def contour_level(values: np.ndarray, percentile: float = CONTOUR_PERCENTILE) -> Optional[float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, percentile)) if len(vals) else None


def occupancy(values: np.ndarray, level: Optional[float]) -> Tuple[Optional[float], int]:
    """(share of atoms above ``level``, atoms outside the map); atoms outside count as not in density."""
    vals = np.asarray(values, dtype=np.float64)
    outside = int(np.sum(~np.isfinite(vals)))
    if level is None or not len(vals):
        return None, outside
    return float(np.sum(np.where(np.isfinite(vals), vals > level, False)) / len(vals)), outside


# --- build conditions -------------------------------------------------------------------------------------

def _flag_name(raw: str) -> str:
    """'-rna:denovo:secstruct_file' / '-rna::denovo::secstruct_file' -> 'secstruct_file'."""
    return raw.lstrip("-").replace("::", ":").strip(":").split(":")[-1]


def _is_flag(tok: str) -> bool:
    return len(tok) > 1 and tok[0] == "-" and (tok[1].isalpha() or tok[1] in ":-")


def _tokens(text: str) -> List[str]:
    try:
        return shlex.split(text, comments=True)
    except ValueError:
        return text.split()


def _resolve(path: str, bases: Sequence[str]) -> Optional[str]:
    if os.path.isabs(path):
        return path if os.path.isfile(path) else None
    for b in bases:
        q = os.path.normpath(os.path.join(b, path))
        if os.path.isfile(q):
            return q
    return None


def expand_flags(tokens: Sequence[str], bases: Sequence[str], notes: List[str], depth: int = 0) -> List[str]:
    """Rosetta ``@file`` arguments replaced by the flags in the file (comments dropped)."""
    out: List[str] = []
    for tok in tokens:
        if tok.startswith("@") and len(tok) > 1 and depth < 4:
            found = _resolve(tok[1:], bases)
            if found is None:
                notes.append(f"flags file {tok[1:]} not found")
                continue
            with open(found, encoding="utf-8", errors="replace") as fh:
                out += expand_flags(_tokens(fh.read()), [os.path.dirname(found)] + list(bases), notes,
                                    depth + 1)
        else:
            out.append(tok)
    return out


def parse_flags(tokens: Sequence[str]) -> Dict[str, List[str]]:
    """Flag (last part of its name) -> values; a repeated flag keeps its last values, as Rosetta does."""
    out: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    for tok in tokens:
        if _is_flag(tok):
            name, eq, value = tok.partition("=")
            cur = _flag_name(name)
            out[cur] = [value] if eq else []
        elif cur is not None:
            out[cur].append(tok)
    return out


_ROSETTA_COMMAND = re.compile(r"core\.init\S*:\s*command:"
                              r"|command:\s*\S*(?:denovo|rosetta|\.linux|\.mac|\.default)")


def command_line(log_path: str, max_lines: int = 400) -> Optional[str]:
    """The command line a Rosetta log records ('core.init: command: ...'), from its first lines; other logs
    that merely contain 'command:' are not taken for a Rosetta run."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for k, line in enumerate(fh):
                if k >= max_lines:
                    break
                if _ROSETTA_COMMAND.search(line):
                    return line[line.find("command:") + len("command:"):].strip()
    except OSError:
        return None
    return None


def _numbering(fasta: Optional[str], reference: Optional[Reference]) -> Optional[Reference]:
    if fasta:
        try:
            return read_reference(fasta)
        except (OSError, EnsembleError):
            pass
    return reference


def _pose_index(key: Tuple[Optional[str], int], numbering: Optional[Reference], length: int) -> Optional[int]:
    """1-based position in the run's sequence of a residue in the reference numbering."""
    if numbering is None:  # no numbering known: residue numbers are positions (one chain, from 1)
        return key[1] if 1 <= key[1] <= length else None
    i = numbering.position(key)
    return None if i is None else i + 1


def _obligate(values: Sequence[str], numbering: Optional[Reference], length: int, explicit: bool
              ) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Pairs of -obligate_pair_explicit (res1 res2 edge1 edge2 orientation ...) or -obligate_pair
    (res1 res2 ...); residues as 'N' or 'C:N' in the run's numbering."""
    pairs, notes = [], []
    toks = list(values)
    step = 5 if explicit and len(toks) % 5 == 0 and all(not re.fullmatch(r"[A-Za-z0-9]?:?-?\d+", t)
                                                       for t in toks[2::5]) else 2
    for k in range(0, len(toks) - 1, step):
        ends = []
        for tok in toks[k:k + 2]:
            m = re.fullmatch(r"(?:([A-Za-z0-9]):)?(-?\d+)", tok)
            if not m:
                ends = []
                break
            ends.append(_pose_index((m.group(1), int(m.group(2))), numbering, length) if m.group(1)
                        else _pose_index((None, int(m.group(2))), numbering, length))
        if len(ends) == 2 and None not in ends:
            pairs.append(tuple(sorted(ends)))
        else:
            notes.append(f"obligate pair {' '.join(toks[k:k + 2])} not placed in the sequence")
    return pairs, notes


@dataclass
class Condition:
    """The build condition of a group of decoys: per register FORCED / UNFORCED / PARTIAL (empty: unknown)."""
    verdicts: Dict[str, str] = field(default_factory=dict)
    detail: str = ""
    source: str = ""

    @property
    def known(self) -> bool:
        return bool(self.verdicts)

    def label(self) -> str:
        if not self.verdicts:
            return "UNKNOWN"
        return ", ".join(f"{k} {v}" for k, v in self.verdicts.items())

    @property
    def signature(self) -> Tuple:
        return tuple(sorted(self.verdicts.items())) if self.verdicts else ("UNKNOWN",)


def judge(dotbracket: Optional[str], obligate: Sequence[Tuple[int, int]], registers: Sequence[Register],
          numbering: Optional[Reference]) -> Tuple[Dict[str, str], List[str]]:
    """Verdict per register from a run's secondary structure (None: none given, all unpaired) and its
    obligate pairs, with the defining pairs placed by ``numbering``."""
    from .secstruct import dotbracket_to_pairs
    notes: List[str] = []
    if dotbracket is not None:
        try:
            ss_pairs, length, _ = dotbracket_to_pairs(dotbracket)
        except ValueError as exc:
            return {}, [f"unreadable secondary structure: {exc}"]
        pairs = {(i + 1, j + 1) for i, j in ss_pairs}
    else:
        pairs, length = set(), len(numbering) if numbering is not None else 10 ** 6
    if numbering is not None and dotbracket is not None and length != len(numbering):
        return {}, [f"the secondary structure has {length} positions, the sequence {len(numbering)}"]
    paired = {i for p in pairs for i in p}
    forced = set(pairs) | set(obligate)
    paired_ob = {i for p in obligate for i in p}
    out: Dict[str, str] = {}
    for reg in registers:
        states = []
        for a, b in reg.pairs:
            ia, ib = _pose_index(a, numbering, length), _pose_index(b, numbering, length)
            if ia is None or ib is None:
                notes.append(f"{reg.name}: {key_text(a if ia is None else b)} is not in the run's sequence")
                states = []
                break
            if tuple(sorted((ia, ib))) in forced:
                states.append("imposed")
            elif ia not in paired | paired_ob and ib not in paired | paired_ob:
                states.append("free")
            else:
                states.append("other")
        if not states:
            continue
        out[reg.name] = ("FORCED" if all(s == "imposed" for s in states) else
                         "UNFORCED" if all(s == "free" for s in states) else "PARTIAL")
    return out, notes


def _read_dotbracket(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                return line.strip()
    return ""


def condition_from_command(cmd: str, bases: Sequence[str], registers: Sequence[Register],
                           reference: Optional[Reference]) -> Tuple[Dict[str, str], str]:
    notes: List[str] = []
    flags = parse_flags(expand_flags(_tokens(cmd), bases, notes))
    fasta = _resolve(flags["fasta"][0], bases) if flags.get("fasta") else None
    numbering = _numbering(fasta, reference)
    where = []
    dotbracket: Optional[str] = None
    if flags.get("secstruct_file"):
        ss = _resolve(flags["secstruct_file"][0], bases)
        if ss is None:
            return {}, f"the command line names {flags['secstruct_file'][0]}, which was not found"
        dotbracket = _read_dotbracket(ss)
        where.append(f"-secstruct_file {flags['secstruct_file'][0]}")
    elif flags.get("secstruct"):
        dotbracket = " ".join(flags["secstruct"])
        where.append("-secstruct")
    else:
        where.append("no secondary structure on the command line (all unpaired)")
    obligate: List[Tuple[int, int]] = []
    length = len(numbering) if numbering is not None else 10 ** 6
    for flag, explicit in (("obligate_pair_explicit", True), ("obligate_pair", False)):
        if flags.get(flag):
            got, more = _obligate(flags[flag], numbering, length, explicit)
            obligate += got
            notes += more
            where.append(f"-{flag} ({len(got)} pairs)")
    verdicts, more = judge(dotbracket, obligate, registers, numbering)
    notes += more
    fixed = _input_residues(flags.get("input_res", []), numbering, length) if flags.get("s") else set()
    held = sorted({key_text(k) for reg in registers for k in reg.residues
                   if _pose_index(k, numbering, length) in fixed})
    if held:  # paired or not, these come from the -s structure: the secondary structure does not decide them
        notes.append(f"{', '.join(held[:6])}{' ...' if len(held) > 6 else ''} in -input_res (held by the "
                     "-s structure)")
    return verdicts, "; ".join(where + notes)


def _input_residues(values: Sequence[str], numbering: Optional[Reference], length: int) -> set:
    """Positions of -input_res ('A:1-6 A:13-16' or '1-6 13-16')."""
    out = set()
    for tok in values:
        m = re.fullmatch(r"(?:([A-Za-z0-9]):)?(-?\d+)(?:-(-?\d+))?", tok)
        if not m:
            continue
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        for r in range(min(a, b), max(a, b) + 1):
            i = _pose_index((m.group(1), r), numbering, length)
            if i is not None:
                out.add(i)
    return out


def _logs(dirs: Sequence[str], limit: int = 60) -> List[str]:
    out: List[str] = []
    for d in dirs:
        for pat in ("log_*.txt", "*.log"):
            for f in sorted(glob.glob(os.path.join(glob.escape(d), pat))):
                if f not in out:
                    out.append(f)
    return out[:limit]


def _run_dirs(d: str, limit: int = 40) -> List[str]:
    """Folders below a log folder where Rosetta may have run (MDATK's FARFAR2: runs/rNN)."""
    subs = [p for p in sorted(glob.glob(os.path.join(glob.escape(d), "*", "*"))) if os.path.isdir(p)]
    subs += [p for p in sorted(glob.glob(os.path.join(glob.escape(d), "*"))) if os.path.isdir(p)]
    return subs[:limit]


def detect_condition(group_dir: str, decoy_dirs: Sequence[str], registers: Sequence[Register],
                     reference: Optional[Reference] = None, log_dirs: Sequence[str] = (),
                     secstructs: Sequence[str] = ()) -> Condition:
    """Build condition of one group (see the module docstring for the order of the sources)."""
    near = list(dict.fromkeys([group_dir, os.path.dirname(group_dir), *decoy_dirs]))
    search = list(dict.fromkeys(near + list(log_dirs)))
    found: Dict[Tuple, Tuple[Dict[str, str], str]] = {}
    unplaced = []
    for log in _logs(search):
        cmd = command_line(log)
        if not cmd:
            continue
        ld = os.path.dirname(log)
        bases = [ld, *_run_dirs(ld), *near, os.getcwd()]
        verdicts, detail = condition_from_command(cmd, bases, registers, reference)
        if not verdicts:
            unplaced.append(f"{os.path.basename(log)}: {detail}")
            continue
        found.setdefault(tuple(sorted(verdicts.items())), (verdicts, f"{detail} [{os.path.basename(log)}]"))
    if len(found) == 1:
        verdicts, detail = next(iter(found.values()))
        return Condition(verdicts, detail, "log")
    if len(found) > 1:
        return Condition({}, "the logs disagree: " + " | ".join(f"{Condition(v).label()} ({d})"
                                                                for v, d in found.values()), "")
    sss = []
    for d in search:
        sss += sorted(glob.glob(os.path.join(glob.escape(d), "*.secstruct")))
    seen: Dict[Tuple, Tuple[Dict[str, str], str]] = {}
    for ss in dict.fromkeys(sss):
        fastas = sorted(glob.glob(os.path.join(glob.escape(os.path.dirname(ss)), "*.fasta")))
        numbering = _numbering(fastas[0] if len(fastas) == 1 else None, reference)
        verdicts, notes = judge(_read_dotbracket(ss), [], registers, numbering)
        if verdicts:
            seen.setdefault(tuple(sorted(verdicts.items())),
                            (verdicts, f"{os.path.basename(ss)} beside the decoys (no log)"
                             + (f"; {'; '.join(notes)}" if notes else "")))
    if len(seen) == 1:
        verdicts, detail = next(iter(seen.values()))
        return Condition(verdicts, detail, "secstruct")
    if len(seen) > 1:
        return Condition({}, "several *.secstruct files disagree", "")
    for ss in secstructs:
        verdicts, notes = judge(_read_dotbracket(ss), [], registers, reference)
        if verdicts:
            return Condition(verdicts, f"{os.path.basename(ss)} (given)" + (f"; {'; '.join(notes)}"
                                                                            if notes else ""), "given")
    why = "; ".join(unplaced[:2]) if unplaced else "no Rosetta log and no *.secstruct found"
    return Condition({}, why, "")


# --- one decoy -------------------------------------------------------------------------------------------

@dataclass
class Decoy:
    """What one decoy file holds for the census and the comparison."""
    path: str
    status: str = ""  # "" when every defining pair could be measured, else why not
    formed: Dict[str, int] = field(default_factory=dict)
    approached: Dict[str, int] = field(default_factory=dict)
    in_register: Dict[str, List[str]] = field(default_factory=dict)  # per pair: formed/approached/apart
    score: Optional[float] = None
    weights: str = ""
    l_count: Optional[int] = None
    d_count: Optional[int] = None
    closure: Optional[float] = None  # geometry: the longest O3'-P
    notes: List[str] = field(default_factory=list)
    keyed: Optional[Dict[ResKey, Residue]] = None  # kept for map sampling (not pickled back by workers)


def scan_decoy(path: str, registers: Sequence[Register], reference: Optional[Reference] = None,
               offsets: Optional[Dict[str, int]] = None, cut: float = PAIR_CUT, near_n: float = NEAR_N,
               near_c1: Tuple[float, float] = NEAR_C1, chirality: bool = False, geometry: bool = False,
               keep: bool = False) -> Decoy:
    """Read one decoy and measure its registers (and optionally chirality / O3'-P closure)."""
    out = Decoy(path)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        structure = pdbtools.read_pdb(text)
    except (OSError, pdbtools.PDBError) as exc:
        out.status = f"unreadable: {exc}"
        return out
    out.score, out.weights = embedded_score(text, path)
    keymap, notes = map_keys(structure, reference, offsets)
    out.notes = notes
    keyed, clashes = keyed_residues(structure, keymap)
    if clashes:
        out.status = f"two residues numbered {clashes[0]} after mapping"
    if chirality:
        calls = Counter()
        for res in keyed.values():
            base = residue_base(res)
            if base:
                calls[pdbtools.sugar_chirality(res.coords(), base)[0]] += 1
        out.l_count, out.d_count = calls.get("L", 0), calls.get("D", 0)
    if geometry:
        out.closure = backbone_closure(structure)[0]
    if keep:
        out.keyed = keyed
    if out.status:
        return out
    for reg in registers:
        formed = approached = 0
        states = []
        for a, b in reg.pairs:
            ra, rb = keyed.get(a), keyed.get(b)
            if ra is None or rb is None:
                out.status = f"no residue {key_text(a if ra is None else b)}"
                return out
            ba, bb = residue_base(ra), residue_base(rb)
            for k, r, base in ((a, ra, ba), (b, rb, bb)):
                want = reference.base_at(k) if reference is not None else None
                if not base:
                    out.status = f"{key_text(k)} ({r.name}) is not a nucleotide"
                    return out
                if want and want != "N" and want != base.replace("T", "U"):
                    out.status = (f"base mismatch: {key_text(k)} is {r.name} (read as {base}), "
                                  f"the reference has {want}")
                    return out
            wc, c1 = pair_distances(ra, rb, ba, bb)
            if wc is None:
                out.status = f"missing N1/N3 atoms in {key_text(a)}-{key_text(b)}"
                return out
            if wc < cut:
                formed += 1
                approached += 1
                states.append("formed")
            elif wc < near_n and c1 is not None and near_c1[0] <= c1 <= near_c1[1]:
                approached += 1
                states.append("approached")
            else:
                states.append("apart")
        out.formed[reg.name] = formed
        out.approached[reg.name] = approached
        out.in_register[reg.name] = states
    return out


def _scan_job(args) -> Decoy:
    path, kwargs = args
    d = scan_decoy(path, **kwargs)
    d.keyed = None
    return d


def scan_all(paths: Sequence[str], jobs: int = 1, **kwargs) -> List[Decoy]:
    """:func:`scan_decoy` over many files, in ``jobs`` processes."""
    if jobs > 1 and len(paths) > 8 and not kwargs.get("keep"):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # spawned workers: forking a process that already runs threads (NumPy's BLAS, a GUI) can deadlock
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
            return list(ex.map(_scan_job, ((p, kwargs) for p in paths), chunksize=16))
    return [scan_decoy(p, **kwargs) for p in paths]


def expand_models(items: Sequence[str], recursive: bool = False, exclude: Sequence[str] = ()) -> List[str]:
    """PDB files from files, folders (``*.pdb``, ``recursive``: the whole tree) and glob patterns, without
    the names matching an ``exclude`` pattern (e.g. a run folder's own input helices)."""
    files: List[str] = []
    for p in items:
        if os.path.isdir(p):
            if recursive:
                for root, _dirs, fs in os.walk(p):
                    files += [os.path.join(root, f) for f in fs if f.lower().endswith(".pdb")]
            else:
                files += glob.glob(os.path.join(glob.escape(p), "*.pdb"))
        elif os.path.isfile(p):
            files.append(p)
        else:
            files += glob.glob(p)
    files = sorted(dict.fromkeys(files))
    if exclude:
        files = [f for f in files if not any(fnmatch.fnmatch(os.path.basename(f).lower(), e.lower())
                                             for e in exclude)]
    kept = []
    for f in files:  # the unfilled slots of a model check's filter hold no decoy: not one of the ensemble
        slot = pdbtools.empty_slot(f)
        if slot:
            print(f"skipped {f}: {slot} (an empty slot of a model check, no model)", file=sys.stderr)
        else:
            kept.append(f)
    return kept


# --- the register census ----------------------------------------------------------------------------------

@dataclass
class CensusRow:
    decoy: Decoy
    group: str
    run: str = ""
    tag: str = ""
    score: Optional[float] = None
    weights: str = ""


@dataclass
class CensusResult:
    registers: List[Register]
    rows: List[CensusRow]
    conditions: Dict[str, Condition]
    report: str
    weight_sets: List[str]


def _pct(k: int, n: int) -> str:
    return f"{100.0 * k / n:6.2f}%" if n else "    - "


def _interval(k: int, n: int, alpha: float = 0.05) -> str:
    if not n:
        return ""
    lo, hi = stats.clopper_pearson(k, n, alpha)
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def _fmt(v: Optional[float], spec: str = ".1f") -> str:
    return "-" if v is None or v != v else format(v, spec)


def _census_block(rows: Sequence[CensusRow], registers: Sequence[Register], title: str, detail: str,
                  weight_ids: Dict[str, str], chirality: bool, expect_l: Optional[int],
                  near_n: float, near_c1: Tuple[float, float], cut: float) -> List[str]:
    n = len(rows)
    names = [r.name for r in registers]
    size = {r.name: len(r.pairs) for r in registers}
    out = [f"=== {title}  ({n} decoys) ===", f"  condition: {detail}"]
    if not n:
        return out
    full = {nm: [r for r in rows if r.decoy.formed.get(nm) == size[nm]] for nm in names}
    out.append(f"  all defining pairs formed (N1-N3 < {cut:g} Å), with 95% Clopper-Pearson intervals:")
    for nm in names:
        k = len(full[nm])
        out.append(f"    {nm:<16} {k:7d} {_pct(k, n)}  {_interval(k, n)}")
    if len(names) > 1:
        out.append("  combinations (registers fully formed):")
        combos = Counter(tuple(r.decoy.formed.get(nm) == size[nm] for nm in names) for r in rows)
        for flags, k in sorted(combos.items(), key=lambda kv: (-kv[1], kv[0])):
            label = " + ".join(nm for nm, f in zip(names, flags) if f) or "none"
            only = " only" if sum(flags) == 1 else ""
            out.append(f"    {label + only:<30} {k:7d} {_pct(k, n)}")
    out.append(f"  in register (formed, or N1-N3 < {near_n:g} Å at C1'-C1' "
               f"{near_c1[0]:g}-{near_c1[1]:g} Å):")
    for nm in names:
        k = sum(1 for r in rows if r.decoy.approached.get(nm) == size[nm])
        hist = Counter(r.decoy.approached.get(nm) for r in rows)
        hist_f = Counter(r.decoy.formed.get(nm) for r in rows)
        out.append(f"    {nm:<16} {k:7d} {_pct(k, n)}   pairs in register per decoy: "
                   + "  ".join(f"{j}/{size[nm]}:{hist.get(j, 0)}" for j in range(size[nm] + 1))
                   + "   formed: " + "  ".join(f"{j}/{size[nm]}:{hist_f.get(j, 0)}"
                                               for j in range(size[nm] + 1)))
    # scores: deciles and Mann-Whitney only within one weight set (-set_weights moves the total)
    by_w: Dict[str, List[CensusRow]] = {}
    for r in rows:
        if r.score is not None and r.score == r.score:
            by_w.setdefault(r.weights, []).append(r)
    for w, scored in sorted(by_w.items(), key=lambda kv: weight_ids.get(kv[0], "")):
        scored.sort(key=lambda r: r.score)
        wl = f"weight set {weight_ids[w]}" if w else "weights not recorded"
        out.append(f"  by score decile, best first ({len(scored)} scored, {wl}):")
        k = max(1, len(scored) // 10)
        for d in range(10):
            chunk = scored[d * k:(d + 1) * k] if d < 9 else scored[9 * k:]
            if not chunk:
                continue
            cells = []
            for nm in names:
                hit = sum(1 for r in chunk if r.decoy.formed.get(nm) == size[nm])
                cells.append(f"{nm} {hit:5d} {100.0 * hit / len(chunk):5.1f}%")
            out.append(f"    {d + 1:2d}  {chunk[0].score:10.1f} to {chunk[-1].score:10.1f}   "
                       + "   ".join(cells))
        if len(names) > 1:
            only = {nm: [r.score for r in scored if r.decoy.formed.get(nm) == size[nm]
                         and all(r.decoy.formed.get(o) != size[o] for o in names if o != nm)] for nm in names}
            out.append("  score of decoys forming exactly one register:")
            for nm in names:
                v = only[nm]
                out.append(f"    {nm:<16} n={len(v):6d}  best {_fmt(min(v) if v else None):>10}  "
                           f"median {_fmt(stats.median(v)):>10}")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if only[a] and only[b]:
                        u, p, how = stats.mann_whitney(only[a], only[b])
                        gap = stats.median(only[b]) - stats.median(only[a])
                        out.append(f"    {b} - {a}: median gap {gap:+.1f}, Mann-Whitney U = {u:g}, "
                                   f"p = {p:.3g} ({how})")
    if chirality:
        ls = [r.decoy.l_count for r in rows if r.decoy.l_count is not None]
        if ls:
            line = (f"  L-handed riboses per decoy: {min(ls)}-{max(ls)} "
                    f"(most often {Counter(ls).most_common(1)[0][0]})")
            if expect_l is not None:
                line += f"; {sum(1 for v in ls if v != expect_l)} of {len(ls)} decoys differ from {expect_l}"
            out.append(line)
    return out


def census(paths: Sequence[str], registers: Sequence[Register], *, scores: Optional[ScoreIndex] = None,
           reference: Optional[Reference] = None, offsets: Optional[Dict[str, int]] = None,
           secstructs: Sequence[str] = (), log_dirs: Sequence[str] = (), group_depth: int = 1,
           pair_suffix: str = "", cut: float = PAIR_CUT, near_n: float = NEAR_N,
           near_c1: Tuple[float, float] = NEAR_C1, chirality: bool = False, expect_l: Optional[int] = None,
           jobs: int = 1, alpha: float = 0.05) -> CensusResult:
    """Count which registers each decoy formed, grouped by folder and build condition (never pooled across
    conditions); see the module docstring.  Returns the rows, the conditions and the report text."""
    check_registers(registers)
    if not paths:
        raise EnsembleError("no decoys given")
    decoys = scan_all(list(paths), jobs=jobs, registers=list(registers), reference=reference, offsets=offsets,
                      cut=cut, near_n=near_n, near_c1=near_c1, chirality=chirality)
    rows: List[CensusRow] = []
    hint_dirs: Dict[str, List[str]] = {}
    for d in decoys:
        g = os.path.dirname(os.path.abspath(d.path))
        for _ in range(max(0, group_depth - 1)):
            g = os.path.dirname(g)
        row = CensusRow(d, g, score=d.score, weights=d.weights)
        rec = scores.lookup(d.path) if scores else None
        if rec is not None:
            row.run, row.tag = rec.get("run", ""), rec.get("tag", "")
            if number(rec.get("score")) is not None:
                row.score = number(rec.get("score"))
            folder = scores.run_folder(rec)
            if folder and folder not in hint_dirs.setdefault(g, []):
                hint_dirs[g].append(folder)
        rows.append(row)
    counted = [r for r in rows if not r.decoy.status]
    bad = [r for r in rows if r.decoy.status]
    weight_sets = sorted({r.weights for r in counted if r.weights})
    weight_ids = {w: f"W{k + 1}" for k, w in enumerate(weight_sets)}
    lines = [f"Register census of {len(rows)} decoys"]
    lines.append("registers: " + "; ".join(f"{r.name} = {r.text()}" for r in registers))
    if reference is not None:
        lines.append(f"reference: {reference.describe()}")
    if offsets:
        lines.append("offsets: " + ", ".join(f"{c}:{v:+d}" for c, v in offsets.items()))
    notes = Counter(n for r in counted for n in r.decoy.notes)
    for note, k in notes.most_common(8):
        lines.append(f"  numbering: {note}" + (f"  ({k} decoys)" if k != len(counted) else ""))
    if bad:
        lines.append(f"{len(bad)} decoys could not be counted:")
        for why, k in Counter(r.decoy.status for r in bad).most_common(8):
            lines.append(f"  {k:7d}  {why}")
        sample = next((r for r in bad if r.decoy.notes), None)
        if sample is not None:
            lines.append(f"  numbering of {os.path.basename(sample.decoy.path)}: "
                         + "; ".join(sample.decoy.notes))
    for w, wid in weight_ids.items():
        lines.append(f"weight set {wid}: {w[:200]}")
    groups: "OrderedDict[str, List[CensusRow]]" = OrderedDict()
    for r in sorted(counted, key=lambda r: r.group):
        groups.setdefault(r.group, []).append(r)
    conditions: Dict[str, Condition] = {}
    for g, grows in groups.items():
        ddirs = sorted({os.path.dirname(os.path.abspath(r.decoy.path)) for r in grows})
        conditions[g] = detect_condition(g, ddirs, registers, reference,
                                         list(log_dirs) + hint_dirs.get(g, []), secstructs)
    lines.append(f"{len(counted)} decoys counted in {len(groups)} group(s)")
    rel = _relabel(list(groups))
    for g, grows in groups.items():
        c = conditions[g]
        lines.append("")
        lines += _census_block(grows, registers, rel[g], f"{c.label()} - {c.detail}", weight_ids, chirality,
                               expect_l, near_n, near_c1, cut)
    if pair_suffix:
        lines += _matched_groups(groups, conditions, registers, pair_suffix, rel)
    by_cond: "OrderedDict[Tuple, List[CensusRow]]" = OrderedDict()
    for g, grows in groups.items():
        by_cond.setdefault(conditions[g].signature, []).extend(grows)
    if len(groups) > 1 and any(len([g for g in groups if conditions[g].signature == sig]) > 1
                               for sig in by_cond):
        lines += ["", "########## pooled WITHIN each condition only ##########"]
        for sig, crows in by_cond.items():
            members = [g for g in groups if conditions[g].signature == sig]
            if len(members) < 2:
                continue
            label = Condition(dict(sig)).label() if sig != ("UNKNOWN",) else "UNKNOWN"
            lines.append("")
            lines += _census_block(crows, registers, f"all {label} groups ({len(members)})",
                                   "pooled across groups sharing this condition", weight_ids, chirality,
                                   expect_l, near_n, near_c1, cut)
        if ("UNKNOWN",) in by_cond:
            lines.append("UNKNOWN groups are only pooled with each other: without the input that built them, "
                         "a register cannot be told apart as an input or an outcome.")
    return CensusResult(list(registers), rows, conditions, "\n".join(lines) + "\n", weight_sets)


def _relabel(groups: Sequence[str]) -> Dict[str, str]:
    """Short group labels: paths relative to their common folder."""
    if not groups:
        return {}
    if len(groups) == 1:
        return {groups[0]: os.path.basename(groups[0].rstrip(os.sep)) or groups[0]}
    common = os.path.commonpath(groups)
    return {g: os.path.relpath(g, os.path.dirname(common) if g == common else common) for g in groups}


def _matched_groups(groups, conditions, registers, suffix: str, rel: Dict[str, str]) -> List[str]:
    """Rows of the groups X and X<suffix> side by side: the same set-up but for what the suffix marks."""
    names = {os.path.basename(g.rstrip(os.sep)): g for g in groups}
    pairs = sorted((b, n) for n in names for b in [n[:-len(suffix)]] if n.endswith(suffix) and b in names)
    if not pairs:
        return []
    size = {r.name: len(r.pairs) for r in registers}
    out = ["", f"########## matched groups on '{suffix}' ##########"]
    for b, n in pairs:
        gb, gn = names[b], names[n]
        sets = {}
        for g in (gb, gn):
            rows = groups[g]
            sets[g] = {reg.name: sum(1 for r in rows if r.decoy.formed.get(reg.name) == size[reg.name]
                                     and all(r.decoy.formed.get(o.name) != size[o.name]
                                             for o in registers if o is not reg)) for reg in registers}
            cells = "   ".join(f"{nm} only {k:6d} {_pct(k, len(rows))}" for nm, k in sets[g].items())
            out.append(f"  {rel[g][:50]:<50} n={len(rows):6d}  {cells}  [{conditions[g].label()}]")
        for reg in registers:
            a, nb = sets[gb][reg.name], len(groups[gb])
            c, nn = sets[gn][reg.name], len(groups[gn])
            p = stats.fisher_exact(a, nb - a, c, nn - c)
            out.append(f"    {reg.name} only, {b} vs {n}: Fisher p = {p:.3g}")
    return out


def census_table_rows(result: CensusResult, rel: Optional[Dict[str, str]] = None
                      ) -> Tuple[List[str], List[List[str]]]:
    rel = rel or _relabel(sorted({r.group for r in result.rows}))
    wid = {w: f"W{k + 1}" for k, w in enumerate(result.weight_sets)}
    cols = ["group", "condition", "model", "run", "tag", "score", "weights", "status"]
    for reg in result.registers:
        cols += [f"{reg.name}_formed", f"{reg.name}_approached", f"{reg.name}_pairs",
                 f"{reg.name}_all_formed", f"{reg.name}_in_register"]
    chir = any(r.decoy.l_count is not None for r in result.rows)
    if chir:
        cols += ["l_riboses", "d_riboses"]
    out = []
    for r in result.rows:
        d = r.decoy
        cond = result.conditions.get(r.group)
        vals = [rel.get(r.group, r.group), cond.label() if cond else "", d.path, r.run, r.tag,
                _fmt(r.score, "g") if r.score is not None else "", wid.get(r.weights, ""),
                d.status or "counted"]
        for reg in result.registers:
            n = len(reg.pairs)
            f, a = d.formed.get(reg.name), d.approached.get(reg.name)
            vals += ["" if f is None else str(f), "" if a is None else str(a), str(n),
                     "" if f is None else str(int(f == n)), "" if a is None else str(int(a == n))]
        if chir:
            vals += ["" if d.l_count is None else str(d.l_count), "" if d.d_count is None else str(d.d_count)]
        out.append(vals)
    return cols, out


def write_table(path: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    delim = "," if path.lower().endswith(".csv") else "\t"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=delim, lineterminator="\n")
        w.writerow(columns)
        for row in rows:
            w.writerow(["" if v is None else v for v in row])


# --- comparing two ensembles ------------------------------------------------------------------------------

@dataclass
class ArmDecoy:
    path: str
    run: str = ""
    tag: str = ""
    score: Optional[float] = None
    closure: Optional[float] = None
    formed: Optional[int] = None
    pairs: int = 0
    occupancy: Optional[float] = None
    level: Optional[float] = None
    calibration_atoms: int = 0
    region_atoms: int = 0
    outside: int = 0
    status: str = ""
    calib_values: Optional[np.ndarray] = None
    region_values: Optional[np.ndarray] = None

    @property
    def all_formed(self) -> Optional[bool]:
        return None if self.formed is None else self.formed == self.pairs


@dataclass
class Arm:
    label: str
    register: Register
    regions: List[Range]
    decoys: List[ArmDecoy]
    notes: List[str] = field(default_factory=list)


def collect_arm(label: str, paths: Sequence[str], register: Register, *, scores: Optional[ScoreIndex] = None,
                closure: Optional[Closure] = None, dmap=None, calibration: Sequence[Range] = (),
                regions: Sequence[Range] = (), interpolation: str = "linear",
                percentile: float = CONTOUR_PERCENTILE, reference: Optional[Reference] = None,
                offsets: Optional[Dict[str, int]] = None, cut: float = PAIR_CUT) -> Arm:
    """Per decoy of one arm: score, closure value, defining pairs formed and the density at the
    calibration and region atoms (the occupancy itself is set by :func:`apply_contour`)."""
    closure = closure or Closure()
    arm = Arm(label, register, list(regions), [])
    no_row = 0
    for path in paths:
        d = scan_decoy(path, [register], reference, offsets, cut=cut, geometry=closure.mode == "geometry",
                       keep=dmap is not None)
        rec = scores.lookup(path) if scores else None
        a = ArmDecoy(path, pairs=len(register.pairs), score=d.score)
        if rec is not None:
            a.run, a.tag = run_id(rec.get("run", "")), rec.get("tag", "")
            if number(rec.get("score")) is not None:
                a.score = number(rec.get("score"))
        elif scores:
            no_row += 1
        if closure.mode == "geometry":
            a.closure = d.closure
        elif closure.mode == "term":
            a.closure = number(rec.get(closure.term)) if rec is not None else None
        if d.status:
            a.status = d.status
        else:
            a.formed = d.formed[register.name]
        if dmap is not None and d.keyed is not None:
            a.calib_values = map_values(dmap, heavy_atom_coords(d.keyed, calibration), interpolation)
            a.region_values = map_values(dmap, heavy_atom_coords(d.keyed, regions), interpolation)
            a.calibration_atoms, a.region_atoms = len(a.calib_values), len(a.region_values)
            if not a.calibration_atoms or not a.region_atoms:
                missing = "no calibration atoms" if not a.calibration_atoms else "no region atoms"
                a.status = a.status or missing
        arm.decoys.append(a)
    if no_row:
        arm.notes.append(f"{no_row} decoys have no row in the decoy table")
    return arm


def apply_contour(arms: Sequence[Arm], mode: str = "per-decoy", percentile: float = CONTOUR_PERCENTILE
                  ) -> Optional[float]:
    """Set every decoy's contour and occupancy: its own calibration atoms ('per-decoy') or one level from
    the calibration atoms of every decoy of both arms ('common'; returned)."""
    common = None
    if mode == "common":
        vals = [d.calib_values for a in arms for d in a.decoys
                if d.calib_values is not None and len(d.calib_values)]
        common = contour_level(np.concatenate(vals), percentile) if vals else None
    for arm in arms:
        for d in arm.decoys:
            if d.region_values is None:
                continue
            d.level = common if mode == "common" else (
                contour_level(d.calib_values, percentile) if d.calib_values is not None else None)
            d.occupancy, d.outside = occupancy(d.region_values, d.level)
    return common


def _rate_cell(k: int, n: int, interval: str, alpha: float) -> str:
    if not n:
        return "-"
    lo, hi = (stats.wilson if interval == "wilson" else stats.clopper_pearson)(k, n, alpha)
    return f"{k:>4}/{n:<5} {100.0 * k / n:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]"


def _fisher(ka: int, na: int, kb: int, nb: int) -> float:
    return stats.fisher_exact(ka, na - ka, kb, nb - kb) if na and nb else float("nan")


def _counts(arm: Arm, pred) -> Tuple[int, int]:
    """(decoys for which ``pred`` holds, decoys for which it is known)."""
    known = [v for v in (pred(d) for d in arm.decoys) if v is not None]
    return sum(1 for v in known if v), len(known)


def _both(d: ArmDecoy, closure: Closure, cut: float) -> Optional[bool]:
    c = closure.closed(d.closure, cut)
    f = d.all_formed
    if c is False or f is False:
        return False
    if c is None or f is None:
        return None
    return True


def _is_closed(d: ArmDecoy, closure: Closure, cut: float) -> bool:
    return closure.mode == "none" or bool(closure.closed(d.closure, cut))


def _power_lines(label_a: str, label_b: str, ka: int, na: int, nb: int, alpha: float, power: float
                 ) -> List[str]:
    """The rates of arm B a two-sided Fisher test would tell from arm A's with the given power."""
    if not na or not nb:
        return []
    p0 = ka / na
    det = stats.detectable_rates(p0, na, nb, alpha, power)
    sides = []
    if det["lower"] is not None:
        sides.append(f"below {100 * det['lower']:.1f}%")
    if det["upper"] is not None:
        sides.append(f"above {100 * det['upper']:.1f}%")
    what = " or ".join(sides) if sides else "no rate at all"
    method = "exact" if det["method"] == "exact" else "normal approximation"
    return [f"  power: against {label_a}'s {100 * p0:.1f}% ({ka}/{na}), a rate in {label_b} (n={nb}) {what}",
            f"         is detected with {100 * power:.0f}% probability (two-sided Fisher test, "
            f"alpha {alpha:g}, {method});",
            "         smaller differences can go undetected at these numbers of decoys."]


def _median_pct(values: Sequence[float]) -> str:
    m = stats.median(values)
    return "-" if m is None else f"median {100 * m:.0f}%"


def compare_arms(a: Arm, b: Arm, *, mode: str = "compare", closure: Optional[Closure] = None,
                 sweep: Sequence[float] = (), alpha: float = 0.05, power: float = 0.8,
                 interval: str = "clopper-pearson", common_level: Optional[float] = None,
                 matched: Optional[str] = None, percentile: float = CONTOUR_PERCENTILE) -> str:
    """The comparison report: numbers, intervals, Fisher p-values and the power statement; no verdict."""
    closure = closure or Closure()
    cut = closure.cut
    iv = "Wilson" if interval == "wilson" else "Clopper-Pearson"
    out = [f"Ensemble comparison ({mode}): {a.label} against {b.label}"]
    for arm in (a, b):
        regions = f"; occupancy regions {selection_text(arm.regions)}" if arm.regions else ""
        out.append(f"  {arm.label}: register {arm.register.name} ({arm.register.text()}), "
                   f"{len(arm.decoys)} decoys{regions}")
        out += [f"    {note}" for note in arm.notes]
        for why, k in Counter(d.status for d in arm.decoys if d.status).most_common(5):
            out.append(f"    {k} decoys: {why}")
    if matched:
        out.append(f"  {matched}")
    if a.regions and b.regions and selection_text(a.regions) != selection_text(b.regions):
        out.append("  note: the arms' occupancy is measured over different residues, so over different atoms")
    if common_level is not None:
        out.append(f"  common contour {common_level:.4g}: the {percentile:g}th percentile of the density at "
                   "the calibration atoms of every decoy of both arms")
    out += ["", f"  {'':<30} {a.label[:32]:<32} {b.label[:32]:<32} {'Fisher p':>9}"]
    pairs_label = (f"defining pairs formed ({len(a.register.pairs)}/{len(a.register.pairs)})"
                   if a.register.name == b.register.name else "defining pairs formed (all)")
    rows = [(pairs_label, lambda d: d.all_formed)]
    if closure.mode != "none":
        rows = [(f"closed ({closure.describe()})", lambda d: closure.closed(d.closure, cut))] + rows
        rows.append(("closed AND pairs formed", lambda d: _both(d, closure, cut)))
    counts = {}
    for name, pred in rows:
        ka, na = _counts(a, pred)
        kb, nb = _counts(b, pred)
        counts[name] = (ka, na, kb, nb)
        out.append(f"  {name:<30} {_rate_cell(ka, na, interval, alpha):<32} "
                   f"{_rate_cell(kb, nb, interval, alpha):<32} {_fisher(ka, na, kb, nb):9.4f}")
    out.append(f"  ({iv} {100 * (1 - alpha):.0f}% intervals)")
    if any(d.occupancy is not None for arm in (a, b) for d in arm.decoys):
        cells = [_median_pct([d.occupancy for d in arm.decoys
                              if d.occupancy is not None and _is_closed(d, closure, cut)]) for arm in (a, b)]
        out.append(f"  {'region atoms in density, closed':<30} {cells[0]:<32} {cells[1]}")
    for arm in (a, b):
        closed = [d for d in arm.decoys if d.score is not None and _is_closed(d, closure, cut)]
        if not closed:
            continue
        best = min(closed, key=lambda d: d.score)
        bits = [os.path.basename(best.path)]
        if closure.mode != "none":
            bits.append(f"closure {_fmt(best.closure, '.2f')}")
        bits.append(f"pairs {'-' if best.formed is None else best.formed}/{best.pairs}")
        if best.occupancy is not None:
            bits.append(f"region atoms in density {100 * best.occupancy:.0f}%")
        out.append(f"  best closed score, {arm.label}: {best.score:.1f} ({', '.join(bits)})")
    ka, na, kb, nb = counts[rows[-1][0]]
    out.append("")
    out += _power_lines(a.label, b.label, ka, na, nb, alpha, power)
    if mode == "sweep" and closure.mode != "none":
        values = list(sweep) or list(GEOMETRY_SWEEP if closure.mode == "geometry" else DEFAULT_SWEEP)
        what = "longest O3'-P" if closure.mode == "geometry" else closure.term
        out += ["", f"--- closure threshold sweep: closed ({what} < X) AND defining pairs formed ---",
                "  (a sensitivity display: the p-values are not corrected for the number of cuts)",
                f"  {'X':>6} {a.label[:16]:>16} {b.label[:16]:>16} {'Fisher p':>9}"]
        for c in values:
            ka, na = _counts(a, lambda d, c=c: _both(d, closure, c))
            kb, nb = _counts(b, lambda d, c=c: _both(d, closure, c))
            ra = 100.0 * ka / na if na else 0.0
            rb = 100.0 * kb / nb if nb else 0.0
            lbl = "any" if not math.isfinite(c) else f"{c:g}"
            out.append(f"  {lbl:>6} {ka:8d} {ra:6.1f}% {kb:8d} {rb:6.1f}% {_fisher(ka, na, kb, nb):9.4f}")
        out += ["", "--- score and density among decoys that close AND pair ---"]
        for arm in (a, b):
            sel = [d for d in arm.decoys if _both(d, closure, cut)]
            sc = [d.score for d in sel if d.score is not None]
            oc = [d.occupancy for d in sel if d.occupancy is not None]
            best = _fmt(min(sc) if sc else None, ".0f")
            out.append(f"  {arm.label:<16} n={len(sel):5d}  best {best:>8}  "
                       f"median {_fmt(stats.median(sc), '.0f'):>8}   region atoms in density "
                       f"{_median_pct(oc)}")
        out += ["", "--- density among all closed decoys, regardless of pairing ---"]
        for arm in (a, b):
            oc = [d.occupancy for d in arm.decoys
                  if d.occupancy is not None and closure.closed(d.closure, cut)]
            if oc:
                out.append(f"  {arm.label:<16} n={len(oc):5d}  {_median_pct(oc)}  "
                           f"mean {100 * sum(oc) / len(oc):.0f}%")
    return "\n".join(out) + "\n"


def compare_table_rows(arms: Sequence[Arm], closure: Closure) -> Tuple[List[str], List[List[Any]]]:
    cols = ["arm", "register", "model", "run", "tag", "score", "closure_value", "closed", "pairs_formed",
            "pairs", "all_formed", "occupancy", "contour", "calibration_atoms", "region_atoms",
            "outside_atoms", "status"]
    out = []
    for arm in arms:
        for d in arm.decoys:
            closed = closure.closed(d.closure, closure.cut)
            out.append([arm.label, arm.register.name, d.path, d.run, d.tag,
                        _fmt(d.score, "g") if d.score is not None else "",
                        _fmt(d.closure, ".4g") if d.closure is not None else "",
                        "" if closed is None else int(closed), "" if d.formed is None else d.formed, d.pairs,
                        "" if d.all_formed is None else int(d.all_formed),
                        "" if d.occupancy is None else f"{d.occupancy:.4f}",
                        "" if d.level is None else f"{d.level:.5g}", d.calibration_atoms, d.region_atoms,
                        d.outside, d.status])
    return cols, out


def matched_runs(a: Arm, b: Arm) -> str:
    """Keep only the decoys of the runs both arms have (same run number: the same seed, when the seeds are
    set from the run number); returns a note for the report."""
    ra = {d.run for d in a.decoys if d.run}
    rb = {d.run for d in b.decoys if d.run}
    keep = ra & rb
    lost = sum(1 for arm in (a, b) for d in arm.decoys if d.run not in keep)
    for arm in (a, b):
        arm.decoys = [d for d in arm.decoys if d.run in keep]
    return (f"restricted to the {len(keep)} runs both arms have ({', '.join(sorted(keep)[:12])}"
            f"{' ...' if len(keep) > 12 else ''}); {lost} decoys of other runs or without a run left out")


def run_seeds(log_dirs: Sequence[str]) -> Dict[str, str]:
    """Run id -> random seed (-run:jran / -jran) from the command lines of the Rosetta logs in folders."""
    out: Dict[str, str] = {}
    for log in _logs(log_dirs, limit=10000):
        rid = run_id(os.path.basename(log))
        cmd = command_line(log)
        if not cmd or not rid:
            continue
        m = re.search(r"(?:^|\s)-(?:[\w:]*:)?jran[\s=]+(-?\d+)", cmd)
        if m:
            out[rid] = m.group(1)
    return out


def seed_note(a_dirs: Sequence[str], b_dirs: Sequence[str], runs: Iterable[str]) -> str:
    sa, sb = run_seeds(a_dirs), run_seeds(b_dirs)
    runs = sorted(set(runs))
    same = [r for r in runs if r in sa and r in sb and sa[r] == sb[r]]
    differ = [r for r in runs if r in sa and r in sb and sa[r] != sb[r]]
    unknown = [r for r in runs if r not in sa or r not in sb]
    out = f"seeds: {len(same)} runs share their seed"
    if differ:
        out += f", {len(differ)} do NOT ({', '.join(differ[:6])})"
    if unknown:
        out += f", {len(unknown)} not found in the logs"
    return out


# --- command line -----------------------------------------------------------------------------------------

def _common_numbering(p) -> None:
    p.add_argument("--reference", help="reference sequence the registers and ranges are numbered in: "
                   "plain ('GGGC...', numbered from 1, chains as in the decoys), chain-prefixed "
                   "('A:GGGC B:...') or FASTA text")
    p.add_argument("--fasta", help="FASTA file with the reference sequence (Rosetta's chain:range tags "
                   "number it, e.g. '>x A:1-16')")
    p.add_argument("--offset", default="", help="per-chain residue-number offsets instead of a reference, "
                   "e.g. 'B:-367,C:44'")
    p.add_argument("--scores", action="append", default=[], metavar="TSV",
                   help="decoy table(s) of the FARFAR2 models step (run, tag, pdb, score terms by name)")
    p.add_argument("--root", default="", help="plan folder the decoy table's 'pdb' paths are relative to "
                   "(default: the parent of the table's folder)")
    p.add_argument("--project", "-p", dest="params", help="project file: its residue groups can be named as "
                   "group:NAME in the ranges")


def _reference_of(args) -> Optional[Reference]:
    if getattr(args, "reference", None):
        return parse_reference(args.reference)
    if getattr(args, "fasta", None):
        return read_reference(args.fasta)
    return None


def _groups_of(args) -> List[Any]:
    path = getattr(args, "params", None)
    if not path:
        return []
    from .config import Project
    return list(Project.load(path).groups)


def _cut_window(text: str) -> Tuple[float, float]:
    m = re.fullmatch(r"\s*([\d.]+)\s*[-,:]\s*([\d.]+)\s*", text or "")
    if not m:
        raise EnsembleError(f"cannot read the C1'-C1' window {text!r} (expected e.g. 8-13)")
    return float(m.group(1)), float(m.group(2))


def sweep_values(text: str) -> List[float]:
    out = []
    for tok in re.split(r"[,;\s]+", (text or "").strip()):
        if tok:
            out.append(math.inf if tok.lower() in ("inf", "any") else float(tok))
    return out


def add_commands(sub) -> list:
    """Register 'census', 'compare' and 'regions' on an argparse sub-parser collection."""
    c = sub.add_parser("census", help="which register each decoy formed, grouped by build condition")
    c.add_argument("models", nargs="*", help="decoy PDB files or folders (with -r: whole trees)")
    c.add_argument("--register", action="append", default=[], required=True, metavar="NAME=PAIRS",
                   help="a register: name and defining pairs, e.g. 'pres=B:22-B:37,B:23-B:36' (repeat for "
                        "each)")
    _common_numbering(c)
    c.add_argument("--models-from-scores", action="store_true",
                   help="also count every decoy the decoy table lists with a PDB file")
    c.add_argument("-r", "--recursive", action="store_true", help="look for *.pdb in whole folder trees")
    c.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                   help="leave out PDB files whose name matches (e.g. 'helices*', '*_native.pdb')")
    c.add_argument("--secstruct", action="append", default=[], help="secondary structure the decoys were "
                   "built with, for groups without a Rosetta log or *.secstruct of their own")
    c.add_argument("--log-dir", action="append", default=[], help="also look for Rosetta logs here")
    c.add_argument("--group-depth", type=int, default=1, help="folder levels above a decoy that form its "
                   "group (1: its own folder)")
    c.add_argument("--pair-suffix", default="", help="report the groups X and X<SUFFIX> side by side")
    c.add_argument("--cut", type=float, default=PAIR_CUT, help="N1-N3 distance of a formed pair (Å)")
    c.add_argument("--near", type=float, default=NEAR_N, help="N1-N3 limit of a pair in register (Å)")
    c.add_argument("--near-c1", default=f"{NEAR_C1[0]:g}-{NEAR_C1[1]:g}", help="C1'-C1' window of a pair in "
                   "register (Å)")
    c.add_argument("--chirality", action="store_true", help="count L-handed riboses per decoy")
    c.add_argument("--expect-l", type=int, help="the number of L riboses every decoy should have")
    c.add_argument("--jobs", type=int, default=1, help="worker processes")
    c.add_argument("-o", "--output", help="per-decoy table (.tsv / .csv)")
    c.add_argument("--report", help="report text file (default: standard output)")

    k = sub.add_parser("compare", help="two decoy ensembles: closure, defining pairs, map occupancy, "
                       "statistics (modes compare / sweep / pilot)")
    k.add_argument("--arm-a", nargs="+", required=True, metavar="PDB", help="decoys (files or folders) of "
                                                                            "arm A")
    k.add_argument("--arm-b", nargs="+", required=True, metavar="PDB", help="decoys of arm B")
    k.add_argument("--labels", default="A,B", help="names of the two arms")
    k.add_argument("--mode", choices=("compare", "sweep", "pilot"), default="compare",
                   help="compare: the headline rates; sweep: also the closure-threshold sweep; pilot: "
                   "two runs of one register, a common contour and optionally matched runs")
    k.add_argument("--register", help="register of both arms (NAME=PAIRS), e.g. in pilot mode")
    k.add_argument("--register-a", help="register of arm A (NAME=PAIRS)")
    k.add_argument("--register-b", help="register of arm B (NAME=PAIRS)")
    _common_numbering(k)
    k.add_argument("--closure", choices=("term", "geometry", "none"), default="term",
                   help="term: a score term below --cut; geometry: every O3'-P below --cut Å")
    k.add_argument("--term", default=CLOSURE_TERM, help="score term of the closure test")
    k.add_argument("--cut", type=float, help=f"closure cut (default {CLOSURE_CUT:g} for a term, "
                   f"{MAX_O3P:g} Å for the geometry)")
    k.add_argument("--pair-cut", type=float, default=PAIR_CUT, help="N1-N3 distance of a formed pair (Å)")
    k.add_argument("--map", help="density map for the occupancy")
    k.add_argument("--calibration", default="", help="regions held fixed (unmoved) whose atoms calibrate "
                   "the contour, e.g. 'A:1-6,B:39-44'")
    k.add_argument("--regions", default="", help="interface / loop regions of both arms")
    k.add_argument("--regions-a", default="", help="regions of arm A (instead of --regions)")
    k.add_argument("--regions-b", default="", help="regions of arm B (instead of --regions)")
    k.add_argument("--percentile", type=float, default=CONTOUR_PERCENTILE,
                   help="percentile of the calibration density that sets the contour")
    k.add_argument("--contour", choices=("per-decoy", "common"), help="per-decoy: each decoy's own "
                   "calibration atoms; common: one level for both arms (default: common in pilot mode)")
    k.add_argument("--interpolation", choices=("linear", "nearest"), default="linear",
                   help="trilinear, or the nearest voxel (as the original scripts)")
    k.add_argument("--sweep", default="", help="closure cuts of the sweep (default: 1 2 3 5 8 10 15 25 inf "
                   "for a term)")
    k.add_argument("--matched-runs", action="store_true", help="keep only the runs both arms have")
    k.add_argument("--log-dir-a", action="append", default=[], help="Rosetta logs of arm A (seed check)")
    k.add_argument("--log-dir-b", action="append", default=[], help="Rosetta logs of arm B (seed check)")
    k.add_argument("--alpha", type=float, default=0.05)
    k.add_argument("--power", type=float, default=0.8, help="power of the detectable-difference statement")
    k.add_argument("--interval", choices=("clopper-pearson", "wilson"),
                   help="interval of the rates (default: Clopper-Pearson; Wilson in pilot mode)")
    k.add_argument("-o", "--output", help="per-decoy table (.tsv / .csv)")
    k.add_argument("--report", help="report text file (default: standard output)")

    g = sub.add_parser("regions", help="rank models by per-residue scores (Q-score, CC) over named regions")
    from .mapscore import add_region_arguments
    g.add_argument("tables", nargs="+", help="per-residue tables (mdatk map score output, or CSV with "
                   "chain, resnum, base, Q)")
    g.add_argument("--column", default="", help="score column (default: score, Q, qscore or cc)")
    add_region_arguments(g)
    g.add_argument("--project", "-p", dest="params", help="project file for group:NAME regions")
    g.add_argument("-o", "--output", help="region table (.tsv / .csv)")
    g.add_argument("--report", help="report text file (default: standard output)")
    return [c, k, g]


def _emit(text: str, path: Optional[str]) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


def run(args) -> int:
    """Execute a parsed 'mdatk ensemble ...' command."""
    if args.ensemble_command == "regions":
        from .mapscore import run_regions
        return run_regions(args)
    reference = _reference_of(args)
    offsets = parse_offsets(args.offset) if args.offset else None
    scores = read_score_index(args.scores, args.root) if args.scores else None
    groups = _groups_of(args)
    if args.ensemble_command == "census":
        regs = [parse_register(t, f"register{k + 1}") for k, t in enumerate(args.register)]
        models = expand_models(args.models, args.recursive, args.exclude)
        if args.models_from_scores and scores:
            models = list(dict.fromkeys(models + [p for p in scores.models() if os.path.isfile(p)]))
        if not models:
            raise EnsembleError("no decoy PDB files found" + ("" if args.recursive else " (try -r)"))
        res = census(models, regs, scores=scores, reference=reference, offsets=offsets,
                     secstructs=args.secstruct, log_dirs=args.log_dir, group_depth=args.group_depth,
                     pair_suffix=args.pair_suffix, cut=args.cut, near_n=args.near,
                     near_c1=_cut_window(args.near_c1), chirality=args.chirality, expect_l=args.expect_l,
                     jobs=max(1, args.jobs))
        if args.output:
            cols, rows = census_table_rows(res)
            write_table(args.output, cols, rows)
        _emit(res.report, args.report)
        return 0 if any(not r.decoy.status for r in res.rows) else 1
    # compare
    labels = [s.strip() for s in args.labels.split(",")]
    if len(labels) != 2 or not all(labels):
        raise EnsembleError("--labels: two names, e.g. baseline,pilot")
    ra = args.register_a or args.register
    rb = args.register_b or args.register
    if not ra or not rb:
        raise EnsembleError("give --register (both arms) or --register-a and --register-b")
    reg_a, reg_b = parse_register(ra, labels[0]), parse_register(rb, labels[1])
    cut = args.cut if args.cut is not None else (MAX_O3P if args.closure == "geometry" else CLOSURE_CUT)
    closure = Closure(args.closure, args.term, cut)
    if closure.mode == "term" and not scores:
        raise EnsembleError("closure by a score term needs the decoy table (--scores), or use --closure "
                            "geometry / none")
    if closure.mode == "term" and not any(closure.term in t.columns for t in scores.tables):
        cols = ", ".join(scores.tables[0].terms[:12])
        raise EnsembleError(f"the decoy table has no column {closure.term!r} (score terms: {cols})")
    dmap = load_map(args.map) if args.map else None
    calib = parse_selection(args.calibration, groups)
    regions_a = parse_selection(args.regions_a or args.regions, groups)
    regions_b = parse_selection(args.regions_b or args.regions, groups)
    if dmap is not None and (not calib or not regions_a or not regions_b):
        raise EnsembleError("with a map, give the calibration regions and the occupancy regions")
    arms = []
    for label, items, reg, regions in ((labels[0], args.arm_a, reg_a, regions_a),
                                       (labels[1], args.arm_b, reg_b, regions_b)):
        paths = expand_models(items)
        if not paths:
            raise EnsembleError(f"arm {label}: no decoy PDB files found")
        arms.append(collect_arm(label, paths, reg, scores=scores, closure=closure, dmap=dmap,
                                calibration=calib, regions=regions, interpolation=args.interpolation,
                                percentile=args.percentile, reference=reference, offsets=offsets,
                                cut=args.pair_cut))
    note = None
    if args.matched_runs:
        note = matched_runs(arms[0], arms[1])
        if args.log_dir_a and args.log_dir_b:
            note += "; " + seed_note(args.log_dir_a, args.log_dir_b, {d.run for d in arms[0].decoys})
        else:
            note += "; seeds not checked (no --log-dir-a / --log-dir-b)"
    contour = args.contour or ("common" if args.mode == "pilot" else "per-decoy")
    level = apply_contour(arms, contour, args.percentile) if dmap is not None else None
    interval = args.interval or ("wilson" if args.mode == "pilot" else "clopper-pearson")
    text = compare_arms(arms[0], arms[1], mode=args.mode, closure=closure, sweep=sweep_values(args.sweep),
                        alpha=args.alpha, power=args.power, interval=interval, common_level=level,
                        matched=note, percentile=args.percentile)
    if args.output:
        cols, rows = compare_table_rows(arms, closure)
        write_table(args.output, cols, rows)
    _emit(text, args.report)
    return 0
