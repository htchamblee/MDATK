"""RNA sequence / secondary-structure helpers for the script generators.

Residues are numbered continuously across chains starting at 1 (the
numbering tleap gives a multi-chain PDB and the one used in FARFAR2
``-input_res`` lists such as ``A:1-6 B:39-44``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

BASES = "ACGU"
SPACERS = " &+,"  # strand separators accepted in sequences and dot-brackets
OPENERS = "([{<"
CLOSERS = ")]}>"

# Hydrogen-bonded atom pairs used for helix-preserving distance restraints
# (pair type is <base of residue i><base of residue j>).
HBOND_ATOMS: Dict[str, List[Tuple[str, str]]] = {
    "GC": [("O6", "N4"), ("N1", "N3"), ("N2", "O2")],
    "CG": [("N4", "O6"), ("N3", "N1"), ("O2", "N2")],
    "AU": [("N1", "N3"), ("N6", "O4")],
    "UA": [("N3", "N1"), ("O4", "N6")],
    "GU": [("O6", "N3"), ("N1", "O2")],  # G.U wobble
    "UG": [("N3", "O6"), ("O2", "N1")],
}
CANONICAL = ("GC", "CG", "AU", "UA")
WOBBLE = ("GU", "UG")


@dataclass
class Chain:
    id: str
    sequence: str
    start: int  # global number of the first residue

    @property
    def end(self) -> int:
        return self.start + len(self.sequence) - 1


@dataclass
class RNAModel:
    chains: List[Chain]
    dotbracket: str = ""  # without strand separators; "" when unknown
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # 1-based, i < j

    @property
    def sequence(self) -> str:
        return "".join(c.sequence for c in self.chains)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def base(self, resnum: int) -> str:
        return self.sequence[resnum - 1]

    def pair_type(self, i: int, j: int) -> str:
        return self.base(i) + self.base(j)

    def chain_of(self, resnum: int) -> Chain:
        for c in self.chains:
            if c.start <= resnum <= c.end:
                return c
        raise KeyError(resnum)

    def stems(self) -> List[List[Tuple[int, int]]]:
        return stems(self.pairs)

    def split(self, text: str, sep: str = " ") -> str:
        """Insert ``sep`` between chains of a per-residue string."""
        return sep.join(text[c.start - 1:c.end] for c in self.chains)

    def ranges(self, resnums: Sequence[int]) -> str:
        """Chain-qualified ranges, e.g. ``A:1-6 B:39-44`` (global numbering)."""
        out: List[str] = []
        for chain in self.chains:
            mine = sorted(r for r in set(resnums) if chain.start <= r <= chain.end)
            for lo, hi in _runs(mine):
                out.append(f"{chain.id}:{lo}-{hi}" if hi > lo else f"{chain.id}:{lo}")
        return " ".join(out)


def _runs(values: Sequence[int]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    for v in values:
        if runs and v == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], v)
        else:
            runs.append((v, v))
    return runs


_CHAIN_PREFIX = re.compile(r"^\s*([A-Za-z0-9])\s*:\s*(.*)$")


def parse_sequence(text: str, first_chain: str = "A") -> List[Chain]:
    """Parse one or more RNA chains.

    Accepted forms: a single sequence; several sequences separated by
    spaces, ``&``, ``+``, ``,`` or new lines; lines prefixed with a chain id
    (``A: GGCAC``); or FASTA text (``>`` headers, chain taken from a
    ``X:1-12`` token in the header when present).  Letters are upper-cased
    and T is read as U.  Raises ValueError for anything else.
    """
    text = (text or "").strip()
    if not text:
        return []
    pieces: List[Tuple[Optional[str], str]] = []
    if text.startswith(">"):
        chain_id: Optional[str] = None
        buf: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if buf:
                    pieces.append((chain_id, "".join(buf)))
                    buf = []
                m = re.search(r"\b([A-Za-z0-9]):\d+", line)
                chain_id = m.group(1) if m else None
            elif line:
                buf.append(line)
        if buf:
            pieces.append((chain_id, "".join(buf)))
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            m = _CHAIN_PREFIX.match(line)
            if m and len(line.split(":")[0].strip()) == 1:
                # "A: GGCA" or "A:GGCA B:UGCC" on one line
                tokens = re.split(r"\s+(?=[A-Za-z0-9]\s*:)", line.strip())
                for tok in tokens:
                    mm = _CHAIN_PREFIX.match(tok)
                    if mm:
                        pieces.append((mm.group(1), mm.group(2)))
                    else:
                        pieces.append((None, tok))
            else:
                for tok in re.split(r"[\s&+,]+", line.strip()):
                    if tok:
                        pieces.append((None, tok))
    chains: List[Chain] = []
    used = set()
    next_id = ord(first_chain.upper()) if first_chain else ord("A")
    start = 1
    for cid, seq in pieces:
        seq = re.sub(r"[\s&+,]+", "", seq).upper().replace("T", "U")
        bad = sorted(set(seq) - set(BASES))
        if bad:
            raise ValueError(f"sequence contains characters other than A, C, G, U: {''.join(bad)}")
        if not seq:
            continue
        if cid is None:
            while chr(next_id) in used:
                next_id += 1
            cid = chr(next_id)
        if cid in used:
            raise ValueError(f"chain {cid} is defined twice")
        used.add(cid)
        chains.append(Chain(cid, seq, start))
        start += len(seq)
    return chains


def strip_spacers(text: str) -> Tuple[str, List[int]]:
    """Remove strand separators; return the string and the 0-based positions
    (in the stripped string) before which a separator stood."""
    out: List[str] = []
    spacers: List[int] = []
    for ch in text:
        if ch in SPACERS:
            if out and (not spacers or spacers[-1] != len(out)):
                spacers.append(len(out))
        else:
            out.append(ch)
    return "".join(out), spacers


def parse_dotbracket(text: str) -> Tuple[str, List[Tuple[int, int]], List[int]]:
    """Parse dot-bracket notation (first non-empty line; ``()[]{}<>``).

    Returns (dot-bracket without separators, 1-based pairs, separator
    positions).  Raises ValueError for unbalanced brackets or unknown
    characters.
    """
    line = ""
    for raw in (text or "").splitlines():
        if raw.strip():
            line = raw.strip()
            break
    db, spacers = strip_spacers(line)
    bad = sorted(set(db) - set(".") - set(OPENERS) - set(CLOSERS))
    if bad:
        raise ValueError(f"dot-bracket contains unexpected characters: {''.join(bad)}")
    stacks: Dict[str, List[int]] = {o: [] for o in OPENERS}
    pairs: List[Tuple[int, int]] = []
    for k, ch in enumerate(db, start=1):
        if ch in OPENERS:
            stacks[ch].append(k)
        elif ch in CLOSERS:
            opener = OPENERS[CLOSERS.index(ch)]
            if not stacks[opener]:
                raise ValueError(f"unmatched '{ch}' at position {k}")
            pairs.append((stacks[opener].pop(), k))
    for opener, stack in stacks.items():
        if stack:
            raise ValueError(f"unmatched '{opener}' at position {stack[-1]}")
    return db, sorted(pairs), spacers


def build_model(sequence_text: str, dotbracket_text: str = "") -> RNAModel:
    """Combine a sequence and an optional dot-bracket; raise ValueError on mismatch."""
    chains = parse_sequence(sequence_text)
    model = RNAModel(chains)
    if not (dotbracket_text or "").strip():
        return model
    db, pairs, _spacers = parse_dotbracket(dotbracket_text)
    if chains and len(db) != model.length:
        raise ValueError(f"dot-bracket has {len(db)} positions but the sequence has "
                         f"{model.length} residues")
    model.dotbracket = db
    model.pairs = pairs
    return model


def stems(pairs: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Group stacked pairs (i, j), (i+1, j-1) into helices."""
    pair_set = set(pairs)
    seen = set()
    out: List[List[Tuple[int, int]]] = []
    for i, j in sorted(pairs):
        if (i, j) in seen:
            continue
        stem = [(i, j)]
        seen.add((i, j))
        a, b = i + 1, j - 1
        while (a, b) in pair_set and a < b:
            stem.append((a, b))
            seen.add((a, b))
            a, b = a + 1, b - 1
        out.append(stem)
    return out


def classify_pairs(model: RNAModel, wobble: bool) -> Tuple[List[Tuple[int, int, str]],
                                                           List[Tuple[int, int, str]]]:
    """Split pairs into (restrainable, skipped) with their pair types."""
    keep, skip = [], []
    allowed = CANONICAL + (WOBBLE if wobble else ())
    for i, j in model.pairs:
        kind = model.pair_type(i, j)
        (keep if kind in allowed else skip).append((i, j, kind))
    return keep, skip


def fasta_text(model: RNAModel, name: str) -> str:
    """Rosetta-style FASTA: one lower-case entry per chain with its numbering."""
    lines = []
    for c in model.chains:
        lines.append(f">{name} {c.id}:{c.start}-{c.end}")
        lines.append(c.sequence.lower())
    return "\n".join(lines) + "\n"


def secstruct_text(model: RNAModel) -> str:
    """Rosetta secstruct file: dot-bracket on line 1 (strands separated by
    spaces), sequence on line 2 for reference (Rosetta reads only line 1)."""
    return model.split(model.dotbracket) + "\n" + model.split(model.sequence.lower()) + "\n"
