"""Base-pair / stacking / dot-bracket annotation of a PDB file as REMARK 950 records.

:func:`annotate` reads a PDB file (:func:`mdatk.pdbtools.read_pdb`; coordinates are kept verbatim),
finds base pairs in its first model with barnaba or the builtin canonical-pair detection
(:mod:`mdatk.secstruct`, the same code as the ``basepairs`` analysis) and inserts the records below
before the first MODEL/ATOM/HETATM record.  Every other line of the file (all models, atom serials,
CONECT ...) is copied unchanged; REMARK 950 lines of an earlier annotation are replaced and a MASTER
record (whose counts no longer apply) is dropped.  'mdatk pdb clean' drops REMARK 950 lines, which
name the input residues.

Record layout (fixed columns; residue fields are chain id, residue number in 4 columns, insertion
code, residue name left-justified in 3 columns)::

    REMARK 950 MDATK BASE-PAIR ANNOTATION
    REMARK 950 ENGINE barnaba 0.1.9                      (or: builtin, canonical pairs ...)
    REMARK 950 PAIRS 20 (WATSON-CRICK 18, WOBBLE 2, OTHER 0)
    REMARK 950 PAIR  A  12 G   B  33 C   WCc             one per base pair (Leontis-Westhof class;
                                                         the builtin engine writes WCc / GUc)
    REMARK 950 STACK A  12 G   A  13 C   >>              one per stacking (barnaba only)
    REMARK 950 CHAINS A,B                                one block per group of chains joined by pairs
    REMARK 950 SEQ   GGGCAAAGCU&AGCUUUGCCC              sequence, '&' between chains
    REMARK 950 DBN   ((((...(((&)))...))))              Watson-Crick and wobble pairs as dot-bracket

SEQ and DBN are wrapped to 60 characters per line (continuation lines repeat the keyword).  Only
nucleotides (a known name plus backbone atoms) appear in SEQ / DBN; chains without nucleotides are left
out.  A strand is a TER-delimited segment of one chain, so files with blank chain ids (Amber, cpptraj)
still get a '&' between strands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .pdbtools import Residue, Structure, _nucleotide, _read_text, parse_residue_name, read_pdb, write_pdb
from .secstruct import (OTHER, WATSON_CRICK, WOBBLE, BarnabaRunner, PairDetector, barnaba_available,
                        base_letter, import_barnaba, lw_kind, pairs_to_dotbracket, unique_pairs)

ENGINES = ("auto", "builtin", "barnaba")
REMARK = "REMARK 950"
WRAP = 60


@dataclass
class AnnotationResult:
    text: str  # the annotated PDB file
    notes: List[str] = field(default_factory=list)
    records: List[str] = field(default_factory=list)  # the REMARK 950 lines written
    engine: str = ""  # the engine used ("barnaba" or "builtin")


def _res_field(res: Residue) -> str:
    return f"{(res.chain or ' ')[:1]}{res.resseq:>4}{(res.icode or ' ')[:1]}{res.name:<3}"


def _wrapped(keyword: str, text: str) -> List[str]:
    return [f"{REMARK} {keyword:<5} {text[k:k + WRAP]}" for k in range(0, max(1, len(text)), WRAP)]


def _barnaba_version() -> str:
    try:
        from importlib.metadata import version
        return version("barnaba")
    except Exception:  # metadata missing for a source checkout
        return ""


def _strands(s: Structure) -> List[List[int]]:
    """Residue indices (into ``s.residues()``) of the nucleotides of every TER segment that has any."""
    out, k = [], 0
    for seg in s.segments:
        rows = [k + n for n, r in enumerate(seg) if _nucleotide(r) is not None]
        if rows:
            out.append(rows)
        k += len(seg)
    return out


def _strand_groups(strands: Sequence[List[int]], pairs: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Strand numbers grouped by the pairs between them (file order within and between groups)."""
    parent = list(range(len(strands)))
    strand_of = {k: n for n, rows in enumerate(strands) for k in rows}

    def root(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for i, j in pairs:
        if i in strand_of and j in strand_of:
            a, b = root(strand_of[i]), root(strand_of[j])
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: Dict[int, List[int]] = {}
    for c in range(len(strands)):
        groups.setdefault(root(c), []).append(c)
    return list(groups.values())


def _dotbracket_records(residues: Sequence[Residue], strands: Sequence[List[int]],
                        pairs: Sequence[Tuple[int, int]]) -> List[str]:
    out = []
    for group in _strand_groups(strands, pairs):
        rows, breaks = [], []
        for c in group:
            if rows:
                breaks.append(len(rows))
            rows += strands[c]
        local = {k: n for n, k in enumerate(rows)}
        group_pairs = [(local[i], local[j]) for i, j in pairs if i in local and j in local]
        seq = "".join(("&" if n in breaks else "") + base_letter(residues[k].name)
                      for n, k in enumerate(rows))
        chains = dict.fromkeys(residues[strands[c][0]].chain or "_" for c in group)
        out.append(f"{REMARK} CHAINS {','.join(chains)}")
        out += _wrapped("SEQ", seq)
        out += _wrapped("DBN", pairs_to_dotbracket(group_pairs, len(rows), breaks))
    return out


def _gap_notes(residues: Sequence[Residue], strands: Sequence[List[int]]) -> List[str]:
    gaps = []
    for rows in strands:
        for a, b in zip(rows, rows[1:]):
            ra, rb = residues[a], residues[b]
            if rb.resseq - ra.resseq > 1:
                gaps.append(f"{ra.chain or '_'}:{ra.resseq}-{rb.resseq}")
    if not gaps:
        return []
    return [f"Residue numbering gaps ({', '.join(gaps)}): missing residues are not in SEQ / DBN, so "
            "the residues on either side are written next to each other"]


def _insert_records(text: str, records: Sequence[str]) -> str:
    """``text`` with its REMARK 950 lines replaced by ``records`` before the first MODEL/ATOM/HETATM
    (and without a MASTER record, whose REMARK count no longer applies)."""
    lines = [line for line in text.splitlines() if not line.startswith((REMARK, "MASTER"))]
    at = next((k for k, line in enumerate(lines) if line[:6].strip().upper() in ("MODEL", "ATOM", "HETATM")),
              len(lines))
    return "\n".join(lines[:at] + list(records) + lines[at:]) + "\n"


def annotate(source, pairs: bool = True, stacking: bool = True, dotbracket: bool = True,
             engine: str = "auto", cutoff: float = 3.5, wobble: bool = True) -> AnnotationResult:
    """Annotate PDB text or a path; see the module docstring for the records.

    ``engine``: "barnaba", "builtin" (canonical pairs from H-bond distances, ``cutoff`` / ``wobble``
    as in the basepairs analysis; no stacking) or "auto" (barnaba when importable).
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown annotation engine {engine!r} (choose from {', '.join(ENGINES)})")
    original = None if isinstance(source, Structure) else _read_text(source)
    s = read_pdb(source)
    notes = []
    for note in s.notes:
        if note.startswith("Kept model 1 of"):
            note = f"Annotated model 1 of {note.rsplit(' ', 1)[-1]} (all models are copied unchanged)"
        elif note.startswith("Repaired") and original is not None:
            note = ("The input has ChimeraX-damaged ATOM/HETATM lines; they are copied unchanged (repair "
                    "them with 'mdatk pdb clean')")
        elif note.startswith(("Dropped", "Ignored")) and original is not None:
            if note.startswith("Dropped the MASTER"):
                notes.append(note)  # _insert_records drops it too
            continue  # anything else is copied verbatim
        notes.append(note)
    residues = list(s.residues())
    strands = _strands(s)
    nucleotide = {k for rows in strands for k in rows}
    rows: List[np.ndarray] = []
    atom_index: List[Dict[str, int]] = []
    heavy_atoms: List[List[Tuple[str, int]]] = []
    n_atoms = 0
    for res in residues:
        coords = res.coords()
        index = {name: n_atoms + k for k, name in enumerate(coords)}
        rows.append(np.array(list(coords.values()), dtype=np.float64).reshape(-1, 3))
        atom_index.append(index)
        heavy_atoms.append([(n, k) for n, k in index.items() if not n.lstrip("0123456789").startswith("H")])
        n_atoms += len(coords)
    xyz = np.concatenate(rows) if rows else np.zeros((0, 3))
    if engine == "auto":
        engine = "barnaba" if barnaba_available() else "builtin"
    if engine == "barnaba":
        import_barnaba()  # the installation hint, even for files with nothing to pair
    names = [r.name if k in nucleotide else "" for k, r in enumerate(residues)]  # '' never pairs
    found: List[Tuple[int, int, str]] = []
    stacks: List[Tuple[int, int, str]] = []
    if engine == "barnaba":
        frame = BarnabaRunner(names, heavy_atoms).annotate([xyz], stacking=stacking)[0]
        found, stacks = frame.pairs, frame.stacks
        version = _barnaba_version()
        header = [f"{REMARK} ENGINE barnaba{' ' + version if version else ''}"]
        if any((parse_residue_name(residues[k].name) or (0, 0, "D"))[2] == "L" for k in nucleotide):
            notes.append("barnaba classifies pairs from base geometry made for D-nucleotides; Leontis-"
                         "Westhof classes of L-RNA pairs may be mirrored (edges / cis-trans)")
    else:
        code = {WATSON_CRICK: "WCc", WOBBLE: "GUc"}
        detected = PairDetector(names, atom_index, cutoff=cutoff, wobble=wobble).detect(xyz)
        found = [(int(i), int(j), code[int(k)]) for i, j, k in detected]
        header = [f"{REMARK} ENGINE builtin, canonical pairs from H-bond distances <= {cutoff:g} A"]
        if stacking:
            notes.append("The builtin engine finds canonical base pairs only (no stacking); "
                         "install barnaba (pip install barnaba) for stacking and non-canonical pairs")
    kinds = [lw_kind(c) for _, _, c in found]
    records = [f"{REMARK} MDATK BASE-PAIR ANNOTATION"] + header
    records.append(f"{REMARK} PAIRS {len(found)} (WATSON-CRICK {kinds.count(WATSON_CRICK)}, "
                   f"WOBBLE {kinds.count(WOBBLE)}, OTHER {kinds.count(OTHER)})")
    if pairs:
        records += [f"{REMARK} PAIR  {_res_field(residues[i])} {_res_field(residues[j])} {c}"
                    for i, j, c in found]
    if stacking and engine == "barnaba":
        records += [f"{REMARK} STACK {_res_field(residues[i])} {_res_field(residues[j])} {c}"
                    for i, j, c in stacks]
    if dotbracket:
        canonical = [(i, j) for (i, j, c), k in zip(found, kinds) if k in (WATSON_CRICK, WOBBLE)]
        canonical, dropped = unique_pairs(canonical)
        if dropped:
            notes.append(f"Left {len(dropped)} canonical pair(s) out of the dot-bracket: their residues "
                         "already pair with another base")
        records += _dotbracket_records(residues, strands, canonical)
        notes += _gap_notes(residues, strands)
    stacked = f" and {len(stacks)} stacking(s)" if stacks else ""
    notes.append(f"Annotated {len(found)} base pair(s){stacked} with the {engine} engine")
    if original is None:
        s.header = [line for line in s.header if not line.startswith(REMARK)] + records
        text = write_pdb(s, strip_conect=False)
    else:
        text = _insert_records(original, records)
    return AnnotationResult(text, notes, records, engine)
