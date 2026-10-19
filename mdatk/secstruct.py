"""Nucleic-acid secondary structure: dot-bracket notation and base-pair detection.

Dot-bracket
    :func:`dotbracket_to_pairs` reads ``()``, ``[]``, ``{}``, ``<>`` and ``Aa`` ... ``Zz`` pseudoknot
    levels with ``&`` or whitespace between chains; :func:`pairs_to_dotbracket` writes it, putting each
    helix on the lowest bracket level it does not cross (longest helices first).

Builtin detection (:class:`PairDetector`)
    Canonical pairs from the donor-acceptor distances of their base-pair hydrogen bonds:

    =========  ==========================================  =====================
    pair       distances (each <= cutoff)                  required
    =========  ==========================================  =====================
    G-C        G N1-C N3, G N2-C O2, G O6-C N4             at least 2 of 3
    A-U / A-T  A N6-U O4, A N1-U N3                         both
    G-U / G-T  G O6-U N3, G N1-U O2 (wobble, optional)      both
    =========  ==========================================  =====================

    The two C1' atoms must also be 8.5-12.5 A apart (about 10.5 A in a canonical pair), which rules
    out stacked neighbours: a 5'-GpC-3' step has two of the G-C distances below 3.5 A as well, with
    its C1' atoms about 5.5 A apart.  Residues without a C1' atom skip this test.

    Every base takes part in at most one pair; competing candidates are resolved greedily by the
    smallest sum of the two shortest H-bond distances (comparable between 2- and 3-bond pairs; sums
    within 0.05 A go to the C1'-C1' distance closer to 10.5 A).  The
    criteria use heavy atoms only and do not depend on sugar chirality, so D- and L-RNA (and D/L
    mixtures that cannot pair canonically anyway) are handled alike.

Barnaba (:class:`BarnabaRunner`)
    Optional engine (``pip install barnaba``; Bottaro et al., NAR 2014) giving Leontis-Westhof classes
    (``WCc`` Watson-Crick, ``GUc`` wobble, e.g. ``WHt``, ``SWc`` ...) and stacking.  Barnaba only
    recognises standard residue names, so residues are written with canonical names (A C G U, DA DC
    DG DT) to a temporary PDB, all frames of one call as models of a single file.

Only the standard library and numpy are needed; barnaba is imported lazily.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import warnings
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .pdbtools import CHAIN_IDS, parse_residue_name

#: pair kinds (pairmap values)
WATSON_CRICK, WOBBLE, OTHER = 1, 2, 3
KIND_LABELS = {WATSON_CRICK: "Watson-Crick", WOBBLE: "wobble", OTHER: "non-canonical"}

OPENERS = "([{<" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLOSERS = ")]}>" + "abcdefghijklmnopqrstuvwxyz"
UNPAIRED = ".-,:_"
CHAIN_SEPARATOR = "&"

# (base 1, partner bases, kind, ((atom of base 1, atom of partner), ...), bonds required)
PAIR_RULES = (
    ("G", ("C",), WATSON_CRICK, (("N1", "N3"), ("N2", "O2"), ("O6", "N4")), 2),
    ("A", ("U", "T"), WATSON_CRICK, (("N6", "O4"), ("N1", "N3")), 2),
    ("G", ("U", "T"), WOBBLE, (("O6", "N3"), ("N1", "O2")), 2),
)
#: C1'-C1' distance window of a base pair (Å)
C1_WINDOW = (8.5, 12.5)
C1_IDEAL = 10.5
SCORE_TIE = 0.05  # H-bond score differences below this are ties (Å)
BASE_PAIR_ATOMS = frozenset(a for rule in PAIR_RULES for pair in rule[3] for a in pair)


# --- dot-bracket ---------------------------------------------------------------------------------------

def dotbracket_to_pairs(text: str) -> Tuple[List[Tuple[int, int]], int, List[int]]:
    """Parse dot-bracket text.

    Returns ``(pairs, n_residues, chain_breaks)``: pairs ``(i, j)`` with ``i < j`` as 0-based residue
    indices (separators do not count), sorted by ``i``; ``chain_breaks`` are the indices of the first
    residue of every chain after the first.
    """
    stacks: Dict[str, List[Tuple[int, int]]] = {}
    pairs: List[Tuple[int, int]] = []
    breaks: List[int] = []
    n = 0
    pending_break = False
    for pos, ch in enumerate(text, 1):
        if ch == CHAIN_SEPARATOR or ch.isspace():
            pending_break = n > 0
            continue
        if pending_break:
            breaks.append(n)
            pending_break = False
        if ch in OPENERS:
            stacks.setdefault(ch, []).append((n, pos))
        elif ch in CLOSERS:
            opener = OPENERS[CLOSERS.index(ch)]
            if not stacks.get(opener):
                raise ValueError(f"unbalanced dot-bracket: {ch!r} at position {pos} (residue {n + 1}) "
                                 f"has no matching {opener!r}")
            i, _ = stacks[opener].pop()
            pairs.append((i, n))
        elif ch not in UNPAIRED:
            raise ValueError(f"invalid dot-bracket character {ch!r} at position {pos}")
        n += 1
    open_left = sorted((p, o) for o, st in stacks.items() for _, p in st)
    if open_left:
        pos, opener = open_left[0]
        more = f" (and {len(open_left) - 1} more)" if len(open_left) > 1 else ""
        raise ValueError(f"unbalanced dot-bracket: {opener!r} at position {pos} is never closed{more}")
    return sorted(pairs), n, breaks


def _stems(pairs: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Group pairs into helices of directly stacked pairs (i, j), (i + 1, j - 1) ..."""
    partner = {i: j for i, j in pairs}
    seen = set()
    stems = []
    for i, j in sorted(pairs):
        if i in seen:
            continue
        stem = []
        while partner.get(i) == j and i not in seen:
            stem.append((i, j))
            seen.add(i)
            i, j = i + 1, j - 1
        stems.append(stem)
    return stems


def _crosses(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[0] < a[1] < b[1] or b[0] < a[0] < b[1] < a[1]


def pairs_to_dotbracket(pairs: Sequence[Tuple[int, int]], n: int, chain_breaks: Sequence[int] = ()) -> str:
    """Dot-bracket text of ``n`` residues; ``&`` is written before every index in ``chain_breaks``.

    Pseudoknots are assigned greedily: helices, longest first, take the first level of ``()``, ``[]``,
    ``{}``, ``<>``, ``Aa`` ... whose pairs they do not cross.
    """
    norm = []
    used: Dict[int, Tuple[int, int]] = {}
    for p in pairs:
        i, j = sorted((int(p[0]), int(p[1])))
        if i == j or i < 0 or j >= n:
            raise ValueError(f"invalid base pair ({p[0]}, {p[1]}) for {n} residues")
        for k in (i, j):
            if k in used and used[k] != (i, j):
                raise ValueError(f"residue {k} is in more than one base pair ({used[k]} and {(i, j)})")
            used[k] = (i, j)
        if (i, j) not in norm:
            norm.append((i, j))
    levels: List[List[Tuple[int, int]]] = []
    chars = ["."] * n
    for stem in sorted(_stems(norm), key=lambda s: (-len(s), s[0])):
        for lev, members in enumerate(levels):
            if not any(_crosses(a, b) for a in stem for b in members):
                break
        else:
            lev = len(levels)
            if lev >= len(OPENERS):
                raise ValueError("too many pseudoknot levels for dot-bracket notation")
            levels.append([])
        levels[lev].extend(stem)
        for i, j in stem:
            chars[i], chars[j] = OPENERS[lev], CLOSERS[lev]
    breaks = {int(b) for b in chain_breaks if 0 < int(b) < n}
    return "".join((CHAIN_SEPARATOR if k in breaks else "") + c for k, c in enumerate(chars))


# --- builtin detection ---------------------------------------------------------------------------------

def base_letter(resname: str) -> Optional[str]:
    """Base (A C G U T) of a nucleotide residue name in any supported scheme, else None."""
    info = parse_residue_name(resname)
    return info.base if info else None


def canonical_name(resname: str) -> Optional[str]:
    """Standard PDB name (A C G U; DA DC DG DT) of a nucleotide residue name, else None."""
    info = parse_residue_name(resname)
    if info is None:
        return None
    return ("D" + info.base) if info.sugar == "dna" else info.base


def _c1(index: Mapping[str, int]) -> int:
    return index.get("C1'", index.get("C1*", -1))


class PairDetector:
    """Canonical base pairs from coordinates (see the module docstring for the criteria).

    ``resnames``: residue name per residue; ``atom_index``: per residue, atom name -> row of the
    coordinate array later passed to :meth:`detect`.  Residues that are no nucleotide, or lack the
    atoms of a rule, never pair by that rule.
    """

    def __init__(self, resnames: Sequence[str], atom_index: Sequence[Mapping[str, int]],
                 cutoff: float = 3.5, wobble: bool = True):
        self.n = len(resnames)
        self.cutoff = float(cutoff)
        bases = [base_letter(r) for r in resnames]
        self.rules = []
        for base, partners, kind, atoms, need in PAIR_RULES:
            if kind == WOBBLE and not wobble:
                continue
            rows_p = [r for r, b in enumerate(bases) if b == base]
            rows_q = [r for r, b in enumerate(bases) if b in partners]
            ip = np.array([[atom_index[r].get(a, -1) for a, _ in atoms] for r in rows_p], dtype=np.int64)
            iq = np.array([[atom_index[r].get(b, -1) for _, b in atoms] for r in rows_q], dtype=np.int64)
            ip, iq = ip.reshape(-1, len(atoms)), iq.reshape(-1, len(atoms))
            keep_p = (ip >= 0).sum(axis=1) >= need  # residues that can still reach `need` bonds
            keep_q = (iq >= 0).sum(axis=1) >= need
            if keep_p.any() and keep_q.any():
                c1_p = np.array([_c1(atom_index[r]) for r in rows_p], dtype=np.int64)
                c1_q = np.array([_c1(atom_index[r]) for r in rows_q], dtype=np.int64)
                self.rules.append((kind, need, np.array(rows_p)[keep_p], ip[keep_p], c1_p[keep_p],
                                   np.array(rows_q)[keep_q], iq[keep_q], c1_q[keep_q]))

    def candidates(self, xyz: np.ndarray) -> np.ndarray:
        """All pairs meeting a rule: float array [m, 5] of (row 1, row 2, kind, score, |C1'-C1' - 10.5|)."""
        xyz = np.asarray(xyz, dtype=np.float64)
        out = []
        lo, hi = C1_WINDOW
        for kind, need, rows_p, ip, c1_p, rows_q, iq, c1_q in self.rules:
            q = xyz[np.maximum(iq, 0)]  # [n_q, k, 3]; missing atoms masked below
            q_missing = iq < 0
            chunk = max(1, 2_000_000 // max(1, iq.size))  # bound the [chunk, n_q, k] temporaries
            for s in range(0, len(ip), chunk):
                pi = ip[s:s + chunk]
                p = xyz[np.maximum(pi, 0)]
                d = np.linalg.norm(p[:, None, :, :] - q[None, :, :, :], axis=-1)
                d[(pi < 0)[:, None, :] | q_missing[None, :, :]] = np.inf
                a, b = np.nonzero((d <= self.cutoff).sum(axis=-1) >= need)
                ca, cb = c1_p[s + a], c1_q[b]
                both = (ca >= 0) & (cb >= 0)
                dc = np.linalg.norm(xyz[np.maximum(ca, 0)] - xyz[np.maximum(cb, 0)], axis=-1)
                ok = ~both | ((dc >= lo) & (dc <= hi))
                a, b = a[ok], b[ok]
                if len(a) == 0:
                    continue
                score = np.sort(d[a, b], axis=-1)[:, :2].sum(axis=-1)
                dev = np.where(both[ok], np.abs(dc[ok] - C1_IDEAL), C1_IDEAL)
                out.append(np.column_stack([rows_p[s + a], rows_q[b], np.full(len(a), kind), score, dev]))
        return np.concatenate(out) if out else np.zeros((0, 5))

    def detect(self, xyz: np.ndarray) -> np.ndarray:
        """Pairs as an int array [m, 3] of (i, j, kind), i < j, sorted by i."""
        cand = self.candidates(xyz)
        used = np.zeros(self.n, dtype=bool)
        keep = []
        # shortest H-bonds first (to 0.05 A); near-ties go to the more ideal C1'-C1' distance, so a
        # base between two partners does not take the one that leaves both neighbours unpaired
        for k in np.lexsort((cand[:, 4], np.round(cand[:, 3] / SCORE_TIE) * SCORE_TIE)):
            i, j = int(cand[k, 0]), int(cand[k, 1])
            if not used[i] and not used[j]:
                used[i] = used[j] = True
                keep.append((min(i, j), max(i, j), int(cand[k, 2])))
        out = np.array(sorted(keep), dtype=np.int64).reshape(-1, 3)
        return out


# --- barnaba -------------------------------------------------------------------------------------------

class FrameAnnotation(NamedTuple):
    """Barnaba result of one frame; residue indices are those given to :class:`BarnabaRunner`."""
    pairs: List[Tuple[int, int, str]]  # (i, j, Leontis-Westhof class), i < j
    stacks: List[Tuple[int, int, str]]  # (i, j, stacking class such as '>>'), i < j


def lw_kind(code: str) -> int:
    """Pair kind of a barnaba Leontis-Westhof class."""
    return WATSON_CRICK if code == "WCc" else WOBBLE if code == "GUc" else OTHER


def import_barnaba():
    try:
        import barnaba
    except ImportError as exc:
        raise RuntimeError("the barnaba base-pair engine needs the optional package barnaba "
                           f"(pip install barnaba): {exc}") from None
    return barnaba


@contextlib.contextmanager
def _numpy_compat():
    """barnaba 0.1.9 calls ``np.chararray``, which numpy 2.5 no longer exports at the top level
    (``np.char.chararray`` remains, deprecated): provide it for the duration of the call."""
    added = False
    if not hasattr(np, "chararray"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            chararray = getattr(getattr(np, "char", None), "chararray", None)
        if chararray is not None:
            np.chararray = chararray
            added = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield
    finally:
        if added:
            del np.chararray


def barnaba_available() -> bool:
    try:
        import_barnaba()
    except RuntimeError:
        return False
    return True


def _element(name: str) -> str:
    letters = name.lstrip("0123456789")
    return letters[:1].upper() if letters else "X"


def _name_field(name: str) -> str:
    return name[:4].ljust(4) if len(name) >= 4 else " " + name.ljust(3)


class BarnabaRunner:
    """Runs ``barnaba.annotate`` on chosen residues.

    ``resnames``: residue name per residue (non-nucleotides are left out); ``atoms``: per residue the
    (atom name, coordinate row) pairs to write - heavy atoms suffice.  Residue k is written as residue
    ``k % 9999 + 1`` of chain ``CHAIN_IDS[k // 9999]`` so that barnaba's residue labels
    (``NAME_RESSEQ_CHAININDEX``) map back to k without relying on which residues barnaba skips.
    """

    BLOCK = 9999

    def __init__(self, resnames: Sequence[str], atoms: Sequence[Sequence[Tuple[str, int]]]):
        self.residues = [(k, canonical_name(r), list(a)) for k, (r, a) in enumerate(zip(resnames, atoms))
                         if canonical_name(r) is not None and a]
        if len(self.residues) > self.BLOCK * len(CHAIN_IDS):
            raise ValueError("too many nucleotides for the barnaba engine")

    def _model_lines(self, xyz: np.ndarray) -> List[str]:
        lines = []
        serial = 0
        for slot, (_, name, atoms) in enumerate(self.residues):
            chain, resseq = CHAIN_IDS[slot // self.BLOCK], slot % self.BLOCK + 1
            for atom, row in atoms:
                serial += 1
                x, y, z = xyz[row]
                lines.append(f"ATOM  {serial % 100000:>5} {_name_field(atom)} {name:>3} {chain}{resseq:>4}"
                             f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {_element(atom):>2}")
        return lines

    def annotate(self, frames: Sequence[np.ndarray], stacking: bool = False) -> List[FrameAnnotation]:
        """One :class:`FrameAnnotation` per coordinate array in ``frames``."""
        frames = list(frames)
        if not frames:
            return []
        empty = [FrameAnnotation([], []) for _ in frames]
        if len(self.residues) < 2:
            return empty
        barnaba = import_barnaba()
        lines = []
        for m, xyz in enumerate(frames, 1):
            lines.append(f"MODEL     {m:>4}")
            lines += self._model_lines(np.asarray(xyz, dtype=np.float64))
            lines.append("ENDMDL")
        lines.append("END")
        fd, path = tempfile.mkstemp(prefix="mdatk_barnaba_", suffix=".pdb")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            # barnaba reports progress on stderr and calls sys.exit() when no residue is usable
            with contextlib.redirect_stderr(io.StringIO()), _numpy_compat():
                try:
                    stackings, pairings, seq = barnaba.annotate(path)
                except SystemExit:
                    return empty
        finally:
            os.unlink(path)
        index = [self._original(label) for label in seq]
        out = []
        for st, pa in zip(stackings, pairings):
            pairs = self._records(pa, index, pairing=True)
            stacks = self._records(st, index, pairing=False) if stacking else []
            out.append(FrameAnnotation(pairs, stacks))
        return out + empty[len(out):]

    def _original(self, label: str) -> int:
        _, resseq, chain = str(label).rsplit("_", 2)
        return self.residues[int(chain) * self.BLOCK + int(resseq) - 1][0]

    @staticmethod
    def _records(frame, index, pairing: bool) -> List[Tuple[int, int, str]]:
        """``[[pairs], [classes]]`` of one frame -> sorted (i, j, class) with i < j.

        Barnaba lists residue pairs in file order, which is our order, so i < j already holds; the
        swap only guards against that changing (edge letters follow their residue)."""
        out = []
        for (a, b), code in zip(frame[0], frame[1]):
            code = str(code)
            if pairing and "X" in code:  # 'XXX': bases close in space but no pairing criteria met
                continue
            i, j = index[int(a)], index[int(b)]
            if i > j:
                i, j = j, i
                if pairing and code not in ("WCc", "GUc"):
                    code = code[1] + code[0] + code[2:]
            out.append((i, j, code))
        return sorted(out)


def unique_pairs(pairs: Sequence[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split pairs into a set where each residue occurs once (first come) and the rejected rest."""
    used = set()
    keep, dropped = [], []
    for i, j in pairs:
        if i in used or j in used:
            dropped.append((i, j))
            continue
        used.update((i, j))
        keep.append((i, j))
    return keep, dropped
