"""D/L chimeras for the structure builder: ideal helices of either hand and either nucleic acid joined through
strand 1, by phosphodiesters (3'-5', 5'-3') or by a phosphate bridging two 3' or two 5' ends (3'-3', 5'-5').

Design (:func:`parse_design`)
    One string of whitespace-separated tokens, SEGMENT LINK SEGMENT [LINK SEGMENT ...].  A segment is
    ``HAND[-NA]:SEQUENCE[/PARTNER]``: HAND D or L, NA RNA or DNA (either case; default the build's ``na``),
    SEQUENCE strand 1 of the segment 5'->3' (:func:`mdatk.builder.parse_sequence`), PARTNER 'auto' or nothing
    (the Watson-Crick complement: an A-form RNA / B-form DNA duplex), 'none' or '-' (a single strand) or the
    partner strand 5'->3' as :func:`mdatk.builder.build_helix` takes it.  A link is 3'-5', 5'-3', 3'-3' or 5'-5'
    (also written 3-5, 35 or 3'5'); its first end is the free end of the strand built so far.  Segment 1 is
    read 5'->3' unless the first link starts at 5'; after a link that ends at 3' (3'-3', 5'-3') the next
    segment is read 3'->5', so its free end is its 5' end, and after one that ends at 5' it is read 5'->3'.
    "D-RNA:AAAAAA 3'-5' L-RNA:AAAAAA" is 5'-D-RNA-p-L-RNA-3', "D-DNA:AAAAAA 3'-3' L-DNA:AAAAAA"
    5'-D-DNA-3'p3'-L-DNA-5' (:func:`describe`), "D-RNA:GGCC 3'-5' D-DNA:GGCC" an RNA-DNA strand of one hand.

Geometry (the junction fit)
    Every segment is the builder's ideal helix (or single strand) of its nucleic acid and hand.  Segment 1
    stays where the builder puts it (helix axis z); every next segment is placed as a rigid body on the
    structure built so far.  A 3'-5' / 5'-3' link is a phosphodiester: O3'-P 1.60 Å, C3'-O3'-P 119.7 and
    O3'-P-O5' 104 degrees (the targets of the builder's backbone closure, Parkinson et al. 1996).  The residue
    that carries the P (its 5' end, no OP3) gets its OP1 / OP2 rebuilt in every step of the fit from P, O3' and
    O5', as the builder rebuilds a phosphate (``builder._GEOM``: P-OP 1.485 Å, OP1-P-OP2 119.6 degrees in the
    plane that bisects O3'-P-O5', away from both, so O3'-P-OP about 108 degrees; OP1 named by the residue's
    hand: vol(P; O3', O5', OP1) is positive for a D and negative for an L residue, as in the builder's helices).
    When the residue that carries the P is DNA, its gamma (about C4'-C5': O5', H5', H5'' and everything beyond
    O5') and beta (about C5'-O5': P and everything beyond it) may turn as well, pulled back to the B-form by a
    prior of 0.3 per radian, and an RNA one stays rigid.  For a 5'-3' link that residue is the 5' end of the
    structure built so far, whose turning atoms and phosphate oxygens then belong to the junction.  A 3'-3' /
    5'-5' link adds a free phosphorus bonded to both O3' (O5'): O-P 1.60 Å on both sides, C-O-P 119.7 and O-P-O
    104 degrees, OP1 / OP2 built the same way opposite the two ester oxygens (OP1 on the side of
    (O(a) - P) x (O(b) - P)).  The end pair of the new segment (its junction residue and that residue's partner;
    the residue alone in a single strand) stacks on the end pair of the structure so far, centroids 3.6 Å
    apart (a strong wish for a phosphodiester, a weak one for the head-to-head ends of a bridge); the O-P
    bonds are stiffer than either wish.  Contacts, the segments' hydrogens included whether or not the model
    has them (so the heavy atoms do not depend on that option), for pairs four or more bonds apart through the
    junction: heavy atoms of the new segment stay 3.2 Å from every atom already placed (CLASH), 3.0 Å for
    pairs four or five bonds apart through it (CLOSE; both residues DNA: 3.2 Å), and never closer than the
    model check's clash distance plus 0.1 Å (QC_MARGIN: ``modelcheck.VDW`` of the two elements less
    ``modelcheck.CLASH_TOL``, so C-C 3.1, P-C 3.2 Å); hydrogens 2.1 Å from hydrogens and 2.4 Å from heavy
    atoms (``builder.CONTACT_FIT`` hh / hx).  Atoms that a gamma / beta turn or the phosphate rebuild moves come
    no closer to the rest of their own segment than in the ideal helix (or those distances).  A bridge stays
    2.9 Å from every atom but the two oxygens it is bonded to and their carbons (BRIDGE_CLASH; four bonds or
    more from it, at least the model check's distance plus the margin).  The fit: many starts (18 turns about
    the helix axis of the previous segment times 2 tilts, 7 for a turning or bridged link), least squares of
    the junction terms alone, then the 8 starts lowest in the full cost (junction and contacts) at their
    solution, one per pose, refitted with the contacts (atom pairs within reach, refreshed when the atoms
    move); the closure of the scratch models of MDATK's first D/L chimeras, generalised to every link and
    segment.  The design and its mirror image (every hand flipped) are both fitted and the lower cost is
    written, as it is or as its exact mirror image (x -> -x, the bridging oxygens keeping their names), so a
    design and its twin give exact mirror images.  Every junction of the written model is then compared with
    its targets (``CLOSED``: the O-P bonds within 0.05 Å, the angles within 10 degrees, a phosphodiester's end
    pairs within 2.5 Å of the stacking, no pair four or more bonds apart more than 0.1 Å inside its contact
    distance); one that misses any is named in a WARNING note with the numbers and the fit cost.

Output (:func:`build_chimera`)
    Strand 1 is split into runs at the bridging links; each run is one chain written 5'->3' (a segment read
    3'->5' is written 5'->3' like any chain), so consecutive residues of a chain are always O3'(i)-P(i+1)
    linked.  Chains: the runs in design order, then the partner strands in segment order (A, B, C ...);
    residues numbered from 1 in each chain (:func:`chain_count`).  A bridging phosphate is a HETATM residue PO2
    (P, OP1, OP2; ATOM in the standard scheme) written as a TER segment of its own after the run before it,
    with that run's chain id and the number after its last residue, so that sequences (SEQRES, FASTA) leave it
    out; LINK records for its two O-P bonds and CONECT records for its four bonds.  Residue names follow each
    residue's hand and nucleic acid in the chosen scheme (the atoms of the CCD's 0DA / 0DT with the
    components' own names, :func:`mdatk.pdbtools.component_atom_name`); terminal variants only at free strand
    ends (the residues bonded through a junction get internal names), OP3 only on free 5' ends with
    ``phosphate5`` (not on a partner strand's 5' end at a junction, which would point at the next segment's end
    pair: it keeps a 5'-OH), HO5' / HO3' only on free hydroxyl ends.  The model is centred on its heavy atoms (the
    middle of their bounding box at the origin); :func:`check_chimera` reports a design too long for the PDB
    coordinate columns before any fit.  The REMARK 951 header names the design, the segments and the
    junctions.  These are starting models: minimise before production.

Linked partner strands (``link_partners``, off by default)
    The partner strands are joined as well: at every junction the partner 3' end that faces it is bonded by a
    phosphodiester to the other partner's 5' end, which keeps P / OP1 / OP2 (no OP3) instead of a 5'-OH, so the
    partners make one continuous chain written 5'->3' after strand 1 (the reverse of strand 1's segment order,
    :func:`describe_partner`; two chains, :func:`chain_count`), its junction residues with internal names.  Every
    link must be a phosphodiester and every segment paired (:func:`check_chimera` and :func:`build_chimera` name
    the bridge or the single strand otherwise).  The junction fit (:class:`_LinkedJunction`) closes both
    phosphodiesters at once with the targets above, one P on either side (3'-5': strand 1's on the new segment,
    the partners' on the structure so far), both phosphates rebuilt, the gamma / beta of both P residues turning
    (RNA too: a rigid body cannot close two links and stack), the contacts scored through whichever link is
    the shorter way; every start is fitted with the rigid body alone and with the turns
    (``_LINKED_STARTS``), and the REFITS best poses of all are refitted with the contacts.  The notes, the header
    (a JUNCTION line of its own ending PARTNER STRANDS) and the closure warning give the partners'
    phosphodiester too.

Everything runs locally; nothing here calls an external program.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import builder, pdbtools
from .builder import BuildError
from .modelcheck import CLASH_TOL, VDW, template_bonds
from .pdbtools import Atom, Residue, Structure

LINK_KINDS = ("3'-5'", "5'-3'", "3'-3'", "5'-5'")
BRIDGES = ("3'-3'", "5'-5'")  # a free phosphate bonded to two O3' / two O5'
HANDS = ("D", "L")
# junction targets: the backbone closure of the builder (builder._GEOM; Parkinson et al. 1996)
LINK_BOND = 1.60  # Å, O3'-P of a phosphodiester, O-P of both bonds of a bridge
C_O_P = 119.7  # degrees, C3'-O3'-P (a 5'-5' bridge: C5'-O5'-P)
O_P_O = 104.0  # degrees, O3'-P-O5' (bridges: O3'-P-O3', O5'-P-O5')
O_P_OP = 109.5  # degrees, O3'-P-OP1 / OP2 of the builder's ideal helices (the rebuilt phosphate: about 108)
STACK = 3.6  # Å, centroids of the two end pairs
CLASH = 3.2  # Å, heavy atoms of a new segment from every atom already placed
CLOSE = {"rna": 3.0, "dna": 3.2}  # Å, pairs four or five bonds apart through the junction (DNA: both sides DNA)
BRIDGE_CLASH = 2.9  # Å, a bridging phosphate from every atom but its two oxygens and their carbons
QC_MARGIN = 0.1  # Å beyond the model check's clash distance (modelcheck.VDW, CLASH_TOL) of pairs >= 4 bonds apart
H_CONTACT = {"hh": builder.CONTACT_FIT["hh"], "hx": builder.CONTACT_FIT["hx"]}  # Å, H-H and H-heavy contacts
OP_BOND = builder._GEOM["P-OP"]  # Å, P-OP1 / P-OP2 of a rebuilt phosphate (a bridge, a junction's P residue)
OP_ANGLE = builder._GEOM["OP-P-OP"]  # degrees, OP1-P-OP2 of a rebuilt phosphate
# a junction of the written model off its targets by more than this is named in a WARNING note: Å for the O-P
# bonds, degrees for the angles, Å beyond STACK for the end pairs of a phosphodiester (an ideal helix puts a
# single-strand end 4.0-5.2 Å from the next pair), Å inside the contact distance of a pair four or more bonds apart
# (the model check's clash distance for C-C pairs four or five bonds apart and every pair with a P)
CLOSED = {"bond": 0.05, "angle": 10.0, "stack": 2.5, "contact": 0.1}
PDB_REACH = 999.999  # Å: the PDB coordinate columns (8.3f) hold -999.999 .. 9999.999: a centred model reaches this
FLEX_PRIOR = 0.3  # per radian of a gamma / beta turn (DNA phosphodiesters)
SPINS = tuple(float(a) for a in range(0, 360, 20))  # degrees about the helix axis of the previous segment
TILTS_RIGID = (0.0, 180.0)  # degrees about x: a rigid phosphodiester
TILTS = (0.0, 20.0, 40.0, 160.0, 180.0, 200.0, 140.0)  # a turning phosphodiester or a bridge
REFITS = 8  # junction fits refitted with the contacts (the lowest in the full cost, one per pose)
MAX_NFEV = (300, 600)  # function evaluations of the two phases
MAX_ATOMS = 99_999  # atoms and TER records: the PDB serial field, so that the CONECT records stay unambiguous
# weights: per Å or radian of the junction terms, per Å of a contact.  The O3'-P of a phosphodiester is as stiff
# as a bridge's O-P bonds (the first chimeras weighted it 1 per Å: a D/L B-DNA junction then stretched it to
# 1.7-1.8 Å for its stacking wish; the A-form RNA ones closed anyway), and so are its angles: at 2 per radian a
# junction that cannot stack gave up 12-25 degrees of C3'-O3'-P or O3'-P-O5' for it, at 4 it gives up the stacking
_W_LINK = (10.0, 4.0, 4.0)  # O3'-P, C3'-O3'-P, O3'-P-O5'
_W_BRIDGE = (10.0, 4.0, 4.0)  # both O-P, both C-O-P, O-P-O
_W_STACK = {"link": 0.5, "bridge": 0.15}  # head-to-head ends of a bridge: stacking is a wish
_W_CONTACT = 3.0
_MARGIN = 2.0  # Å beyond the contact distance: atom pairs scored by a refit (refreshed past half of it)
_REFRESHES = 6
# linked partners: the junction-only fits of every start move the rigid body alone (turns at zero) and then the turns
# too; the two sets land in different poses (neither finds the best stacking every time), the refits choose from both.
# The fits with the turns stop early: short ones propose poses as good (lower fit costs on the designs tried, D/L
# RNA and DNA, 2 and 3 segments) in a third of the time
_LINKED_STARTS = ("rigid", "free")
_LINKED_NFEV = {"free": 40}  # function evaluations of those fits by mode (otherwise MAX_NFEV[0])
_PHASE1_TOL = 1e-5  # tolerances of the junction-only fits: they propose poses, the refits converge (1e-8)
_POSE = 0.5  # Å: junction fits whose new junction residue lies within this of a chosen one are the same pose
_CACHE_SIZE = 64  # fitted designs kept (a few arrays each): a page that renames or adds hydrogens refits nothing
_BACKBONE = frozenset({"P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"})
_GAMMA = ("O5'", "H5'", "H5''", "P", "OP1", "OP2", "OP3")  # substituents of C5' but C4', and beyond O5'
_BETA = ("P", "OP1", "OP2", "OP3")  # P and beyond
_FAR = 99  # bond separation of atoms not reached by a search
_ELEMENTS = {"C": 0, "N": 1, "O": 2, "P": 3, "H": 4}  # element codes of the contact model (others: as carbon)
_H = _ELEMENTS["H"]
_RADII = np.array([VDW[e] for e in _ELEMENTS])
# the closest a pair of heavy atoms (codes) four or more bonds apart may come in a fit: the model check's clash
# distance (it counts a pair VDW[a] + VDW[b] - CLASH_TOL or closer) plus QC_MARGIN
_QC_LIMIT = _RADII[:, None] + _RADII[None, :] - CLASH_TOL + QC_MARGIN
_REACH = max(CLASH, max(CLOSE.values()), BRIDGE_CLASH, float(_QC_LIMIT[:4, :4].max()), *H_CONTACT.values())

Progress = Optional[Callable[[float, str], None]]
Cancel = Optional[Callable[[], bool]]


# --- the design ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """One segment of a chimera: strand 1 (5'->3'), its hand, nucleic acid and partner strand (5'->3', None:
    a single strand)."""
    sequence: str
    chirality: str = "D"
    na: str = "rna"
    partner: Optional[str] = None

    @property
    def kind(self) -> str:
        """'D-RNA', 'L-DNA' ..."""
        return f"{self.chirality}-{self.na.upper()}"

    def flipped(self) -> "Segment":
        return replace(self, chirality="L" if self.chirality == "D" else "D")


Design = Tuple[List[Segment], List[str]]
_SEGMENT_RE = re.compile(r"^([^:\s\-]+)(?:-([^:\s]*))?:(.*)$")
_PRIMES = str.maketrans({"’": "'", "′": "'", "‘": "'", "`": "'"})


def link_kind(token: str) -> Optional[str]:
    """The link of a token in its LINK_KINDS spelling (3-5, 35, 3'5', 3'-5' -> "3'-5'"), None otherwise."""
    t = (token or "").strip().translate(_PRIMES)
    if not re.fullmatch(r"[35]'?-?[35]'?", t):
        return None
    ends = re.sub(r"[^35]", "", t)
    return f"{ends[0]}'-{ends[1]}'"


def _link_list() -> str:
    return "3'-5', 5'-3', 3'-3' or 5'-5'"


def _segment(token: str, k: int, na: str) -> Tuple[Segment, List[str]]:
    """Segment ``k`` (1-based) of a design from its token, and notes."""
    m = _SEGMENT_RE.match(token)
    if not m:
        raise BuildError(f"segment {k} {token!r}: write HAND[-NA]:SEQUENCE[/PARTNER], e.g. D-RNA:GGCAUC, "
                         "L:GAUGCC/none or D-DNA:GGCAT/ATGCC")
    hand, kind, rest = m.groups()
    if hand.upper() not in HANDS:
        raise BuildError(f"segment {k} {token!r}: the hand must be D or L, got {hand!r}")
    if kind is not None and kind.strip().lower() not in builder.NUCLEIC_ACIDS:
        raise BuildError(f"segment {k} {token!r}: unknown nucleic acid {kind!r} (RNA or DNA)")
    seg_na = builder.check_na(kind if kind is not None else na)
    seq_text, slash, partner_text = rest.partition("/")
    seq, notes = builder.parse_sequence(seq_text, f"segment {k} sequence", seg_na)
    mode = partner_text.strip() if slash else "auto"
    if not mode:
        raise BuildError(f"segment {k} {token!r}: nothing after '/': give the partner strand 5'->3', auto (the "
                         "Watson-Crick complement) or none (a single strand)")
    if mode in ("-",) or mode.lower() == "none":
        partner = None
    elif mode.lower() == "auto":
        partner = builder.complement(seq, seg_na)
    else:
        partner, more = builder.parse_sequence(mode, f"segment {k} partner", seg_na)
        notes += more
        if len(partner) != len(seq):
            raise BuildError(f"segment {k}: the partner has {len(partner)} nucleotides but the sequence has "
                             f"{len(seq)}; an explicit partner must have the same length (5'->3')")
    return Segment(seq, hand.upper(), seg_na, partner), notes


def _read53(links: Sequence[str]) -> List[bool]:
    """Per segment: read 5'->3' (True) or 3'->5' as the design is read."""
    out = [not links or links[0][0] == "3"]
    out += [link[3] == "5" for link in links]  # after a link that ends at 5' the next segment is read 5'->3'
    return out


def _fitting(end: str) -> str:
    return "3'-5' or 3'-3'" if end == "3" else "5'-3' or 5'-5'"


def parse_design(text: str, na: str = "rna") -> Tuple[List[Segment], List[str], List[str]]:
    """(segments, links in their LINK_KINDS spelling, notes) of a design string (see the module docstring);
    ``na``: the nucleic acid of the segments that name none.  Raises :class:`BuildError` for anything else,
    naming the token or segment."""
    na = builder.check_na(na)
    tokens = (text or "").split()
    if not tokens:
        raise BuildError("the design is empty: write SEGMENT LINK SEGMENT ..., e.g. D-RNA:GGCAUC 3'-5' L-RNA:GAUGCC")
    segments: List[Segment] = []
    links: List[str] = []
    notes: List[str] = []
    for i, token in enumerate(tokens):
        if i % 2:
            kind = link_kind(token)
            if kind is None:
                if _SEGMENT_RE.match(token):
                    raise BuildError(f"segments {len(segments)} and {len(segments) + 1} need a link between them "
                                     f"({_link_list()})")
                raise BuildError(f"unknown link {token} after segment {len(segments)} ({_link_list()}; also "
                                 "written 3-5, 35 or 3'5')")
            links.append(kind)
            continue
        if link_kind(token) is not None:
            where = "the design starts with" if not i else f"link {len(links)} is followed by"
            raise BuildError(f"{where} the link {token} where a segment belongs")
        seg, more = _segment(token, len(segments) + 1, na)
        segments.append(seg)
        notes += more
    if len(tokens) % 2 == 0:
        raise BuildError(f"the design ends with the link {tokens[-1]}: a segment must follow it")
    if len(segments) < 2:
        raise BuildError("a chimera needs at least two segments joined by a link, e.g. D-RNA:GGCAUC 3'-5' "
                         "L-RNA:GAUGCC (one helix: mdatk build helix)")
    read = _read53(links)
    for j in range(1, len(links)):
        free = "3" if read[j] else "5"
        if links[j][0] != free:
            how = "5'->3'" if read[j] else "3'->5'"
            raise BuildError(f"link {j + 1} ({links[j]}) starts at a {links[j][0]}' end, but the free end of segment "
                             f"{j + 1} ({_token(segments[j])}, read {how} after link {j} {links[j - 1]}) is its "
                             f"{free}' end: use {_fitting(free)}")
    return segments, links, notes


def _token(seg: Segment) -> str:
    return f"{seg.kind}:{seg.sequence}"


def _design(design, na: str = "rna") -> Design:
    """(segments, links) of a design string or of the result of :func:`parse_design`."""
    if isinstance(design, str):
        segments, links, _notes = parse_design(design, na)
        return segments, links
    segments, links = list(design[0]), list(design[1])
    if len(segments) != len(links) + 1 or len(segments) < 2 or any(k not in LINK_KINDS for k in links):
        raise BuildError("a design is segments with one link fewer between them")
    return segments, links


def describe(design, na: str = "rna") -> str:
    """The strand as the design reads it: "5'-D-RNA(6)-p-L-RNA(6)-3'", "5'-D-DNA(6)-3'p3'-L-DNA(6)-5'"."""
    segments, links = _design(design, na)
    read = _read53(links)
    out = ["5'" if read[0] else "3'"]
    for k, seg in enumerate(segments):
        if k:
            link = links[k - 1]
            out.append("p" if link not in BRIDGES else f"{link[0]}'p{link[0]}'")
        out.append(f"{seg.kind}({len(seg.sequence)})")
    out.append("3'" if read[-1] else "5'")
    return "-".join(out)


def linked_strands(design, na: str = "rna") -> Tuple[str, str, List[str]]:
    """Strand 1 and the continuous partner strand of a design with linked partners as written (both 5'->3', the
    letters of every segment) and the pair type of every strand-1 position (``builder.pair_types``: 'wc',
    'wobble', 'mismatch'; strand-1 residue i pairs with partner residue n - 1 - i); raises :class:`BuildError`
    for a design whose partners cannot be linked."""
    segments, links = _design(design, na)
    _check_linked(segments, links)
    order = _partner_order(segments, links)
    strand1 = "".join(segments[k].sequence for k in order[::-1])
    partner = "".join(segments[k].partner for k in order)
    kinds = [t for k in order[::-1] for t in builder.pair_types(segments[k].sequence, segments[k].partner,
                                                                segments[k].na)]
    return strand1, partner, kinds


def _partner_end5(end3: str) -> str:
    """The 5' end of a segment's partner strand: 'nick' where it faces a junction (it pairs with strand 1's 3' end,
    so it does when that end is bonded), 'free' otherwise.  A nick end keeps a 5'-OH even with ``phosphate5``: a
    phosphate there points at the next segment's end pair, where the stacking puts it in the way (a one base
    pair segment between two of them could not close)."""
    return "nick" if end3 == "bonded" else "free"


def _partner_ends(ends: Tuple[str, str], linked: bool = False) -> Tuple[str, str]:
    """(5' end, 3' end) of a segment's partner strand from the ends of its strand 1 (:func:`_ends`).  Unlinked: the
    5' end 'nick' or 'free' (:func:`_partner_end5`), the 3' end 'free'.  Linked partner strands (``link_partners``):
    the 5' end that faces a junction carries the P of the partners' phosphodiester ('p'), the 3' end that faces
    one is 'bonded'; the ends of the chimera stay 'free'."""
    end5, end3 = ends
    if not linked:
        return _partner_end5(end3), "free"
    return ("p" if end3 == "bonded" else "free"), ("bonded" if end5 != "free" else "free")


def _ends(links: Sequence[str], read: Sequence[bool], k: int) -> Tuple[str, str]:
    """(5' end, 3' end) of strand 1 of segment ``k``: 5' 'free', 'p' (it carries the P of a phosphodiester)
    or 'bridge' (its O5' is bonded to a bridging phosphate); 3' 'free' or 'bonded'."""
    entry = links[k - 1] if k else None
    exit_ = links[k] if k < len(links) else None
    end5 = end3 = "free"
    if read[k]:  # entry at the 5' end, exit at the 3' end
        if entry:
            end5 = "p" if entry == "3'-5'" else "bridge"
        if exit_:
            end3 = "bonded"
    else:
        if entry:
            end3 = "bonded"
        if exit_:
            end5 = "p" if exit_ == "5'-3'" else "bridge"
    return end5, end3


def _atom_count(segments: Sequence[Segment], links: Sequence[str], hydrogens: bool, phosphate5: bool,
                linked: bool = False) -> int:
    """Atoms of the model (the bridging phosphates included) without building it."""
    read = _read53(links)
    total = 3 * sum(link in BRIDGES for link in links)
    for k, seg in enumerate(segments):
        end5, end3 = _ends(links, read, k)
        strands = [(seg.sequence, end5, end3)] + ([(seg.partner, *_partner_ends((end5, end3), linked))]
                                                   if seg.partner else [])
        for seq, e5, e3 in strands:
            total += sum(len(builder.template(b, hydrogens, seg.na)) for b in seq)
            if e5 == "free":
                total += 1 if phosphate5 else -3 + int(hydrogens)  # OP3, or a 5'-OH (HO5')
            elif e5 == "nick":
                total += -3 + int(hydrogens)  # a 5'-OH even with phosphate5
            elif e5 == "bridge":
                total -= 3
            if e3 == "free" and hydrogens:
                total += 1  # HO3'
    return total


def _chain_count(segments: Sequence[Segment], links: Sequence[str], linked: bool = False) -> int:
    partners = sum(s.partner is not None for s in segments)
    return 1 + sum(link in BRIDGES for link in links) + (min(partners, 1) if linked else partners)


def chain_count(design, na: str = "rna", link_partners: bool = False) -> int:
    """The chains a design writes (the chain ids ``chains`` of :func:`build_chimera` must name): the strand-1 runs
    between the 3'-3' / 5'-5' links, then the partner strands (``link_partners``: the one partner chain).  A
    bridge's PO2 is a TER segment of its own with the chain id of the run before it, so it adds none.  ``design``:
    as :func:`build_chimera` takes it; raises :class:`BuildError` for a design that does not parse."""
    segments, links = _design(design, na)
    return _chain_count(segments, links, link_partners)


def _ter_count(segments: Sequence[Segment], links: Sequence[str], linked: bool = False) -> int:
    """TER records of the model: one per chain and one per PO2 segment."""
    return _chain_count(segments, links, linked) + sum(link in BRIDGES for link in links)


def _check_linked(segments: Sequence[Segment], links: Sequence[str]) -> None:
    """Partner strands can be linked (``link_partners``) only when every link is a phosphodiester (3'-5', 5'-3') and
    every segment has a partner strand: raises :class:`BuildError` naming the first that is not."""
    for j, link in enumerate(links):
        if link in BRIDGES:
            raise BuildError(f"link partners: link {j + 1} ({link}) is a phosphate bridging two {link[0]}' ends; "
                             "the partner strands are joined only across 3'-5' / 5'-3' phosphodiesters (build "
                             "without linked partners)")
    for k, seg in enumerate(segments, 1):
        if seg.partner is None:
            raise BuildError(f"link partners: segment {k} ({_token(seg)}) is a single strand (partner none); "
                             "every segment needs a partner strand to join them (auto or a sequence), or build "
                             "without linked partners")


def _partner_order(segments: Sequence[Segment], links: Sequence[str]) -> List[int]:
    """The segments in the order of their partner strands in the linked partner chain (5'->3'): the reverse of
    strand 1's written order."""
    order = list(range(len(segments)))
    return order[::-1] if _read53(links)[0] else order


def describe_partner(design, na: str = "rna") -> str:
    """The continuous partner strand of a design with linked partners, 5'->3' as written:
    "5'-L-DNA(10)-p-D-DNA(10)-3'" for "D-DNA:... 3'-5' L-DNA:..."; raises :class:`BuildError` for a design whose
    partners cannot be linked (:func:`_check_linked`)."""
    segments, links = _design(design, na)
    _check_linked(segments, links)
    return "5'-" + "-p-".join(f"{segments[k].kind}({len(segments[k].partner)})"
                              for k in _partner_order(segments, links)) + "-3'"


@lru_cache(maxsize=None)
def _level_span(na: str, paired: bool) -> float:
    """Å along the helix axis of one level of a segment (a base pair or a nucleotide), hydrogens and a
    5'-phosphate included: a segment of n levels spans this plus n - 1 rises."""
    s = pdbtools.read_pdb(builder.build_helix("G", "auto" if paired else "none", na=na, hydrogens=True,
                                              phosphate5=True))
    z = [a.xyz[2] for r in s.residues() for a in r.atoms]
    return float(max(z) - min(z))


def _extent(segments: Sequence[Segment], links: Sequence[str]) -> float:
    """Å from end to end of the chimera with its segments stacked in a straight line (more than a fit gives):
    every segment's span along its axis, and per junction 12 Å for the end pairs (a bridge's may lie 11 Å
    apart)."""
    out = 12.0 * len(links)
    for seg in segments:
        rise = builder.DNA_RISE if seg.na == "dna" else builder.RISE
        out += (len(seg.sequence) - 1) * rise + _level_span(seg.na, seg.partner is not None)
    return out


def _check_extent(segments: Sequence[Segment], links: Sequence[str]) -> None:
    extent = _extent(segments, links)
    if extent > 2.0 * (PDB_REACH - 1.0):
        raise BuildError(f"too long: the segments end to end reach about {extent:.0f} Å, and a PDB file holds "
                         f"{2.0 * PDB_REACH:.0f} Å of a model centred at the origin (coordinates from -999.999 Å): "
                         "shorten the segments")


def _check_chains(chains, count: int) -> None:
    if chains is not None and not (isinstance(chains, str) and not chains.strip()):
        _chain_ids(chains, count)


def check_chimera(text: str, na: str = "rna", hydrogens: bool = False, phosphate5: bool = False, chains=None,
                  link_partners: bool = False) -> List[Tuple[str, str]]:
    """Design-level messages [(level, text)] (levels 'error', 'warning', 'info'), as
    :func:`mdatk.builder.check_helix`: the design as read, the base pairs of every segment, the size and the
    extent (a model too long for the PDB coordinate columns is an error before any fit).  ``chains``: chain ids
    as :func:`build_chimera` takes them (None or '': A, B, C ...), checked with the same rule (one single
    character per chain of :func:`chain_count`, all different).  ``link_partners``: the partner strands joined
    too (:func:`build_chimera`): an error for a bridging link or a single-strand segment."""
    try:
        segments, links, notes = parse_design(text, na)
        if link_partners:
            _check_linked(segments, links)
        atoms = _atom_count(segments, links, hydrogens, phosphate5, link_partners)
        count = _chain_count(segments, links, link_partners)
        _check_size(atoms, _ter_count(segments, links, link_partners), count)
        _check_extent(segments, links)
        _check_chains(chains, count)
    except BuildError as exc:
        return [("error", str(exc))]
    out = [("info", f"{describe((segments, links))}: {len(segments)} segments, {len(links)} link(s) "
                    f"({', '.join(links)})")]
    if link_partners:
        out.append(("info", f"partner strands linked: one continuous partner chain {describe_partner((segments, links))}"
                            ", its phosphodiesters fitted together with strand 1's at every junction"))
    out += [("info", n) for n in notes]
    for k, seg in enumerate(segments, 1):
        out += [(level, f"segment {k} ({seg.kind}): {msg}") for level, msg in _pair_messages(seg)]
    out.append(("info", f"{atoms} atoms ({'hydrogens included' if hydrogens else 'heavy atoms only'}); the "
                        "junctions are fitted as rigid bodies (seconds per junction): minimise before production"))
    return out


def _pair_messages(seg: Segment) -> List[Tuple[str, str]]:
    n, dna = len(seg.sequence), seg.na == "dna"
    if seg.partner is None:
        return [("info", f"single strand of {n} nucleotides in the {'B' if dna else 'A'}-form helix geometry")]
    kinds = builder.pair_types(seg.sequence, seg.partner, seg.na)
    mism = [f"{k + 1}:{seg.sequence[k]}·{seg.partner[n - 1 - k]}" for k, t in enumerate(kinds) if t == "mismatch"]
    out = [("info", f"{n} base pairs: {kinds.count('wc')} Watson-Crick, {kinds.count('wobble')} "
                    f"G·{'T' if dna else 'U'} wobble, {len(mism)} mismatched")]
    if mism:
        out.append(("warning", "mismatched pairs keep the Watson-Crick frame (bases may collide; minimise or "
                               "remodel): " + ", ".join(mism[:8]) + (" ..." if len(mism) > 8 else "")))
    return out


def _check_size(atoms: int, ters: int, chains: int) -> None:
    if atoms + ters > MAX_ATOMS:
        raise BuildError(f"too many atoms: {atoms} atoms and {ters} TER records do not fit the {MAX_ATOMS} serial "
                         "numbers of a PDB file (the CONECT records of the bridges need them unambiguous)")
    if chains > len(pdbtools.CHAIN_IDS):
        raise BuildError(f"too many chains: {chains} (at most {len(pdbtools.CHAIN_IDS)} chain ids)")


# --- segments as built ----------------------------------------------------------------------------------

@dataclass
class _Res:
    """A nucleotide of a segment: strand 0 (strand 1 of the design) or 1 (the partner), index 5'->3'."""
    seg: int
    strand: int
    index: int
    base: str
    hand: str
    na: str
    names: List[str]
    xyz: np.ndarray  # local frame of the segment, then world

    def at(self, name: str) -> np.ndarray:
        return self.xyz[self.names.index(name)]


def _segment_residues(k: int, seg: Segment, ends: Tuple[str, str], phosphate5: bool, linked: bool = False
                      ) -> List[List[_Res]]:
    """The strands of segment ``k`` as the builder makes them, hydrogens included, atoms by their standard names
    (its helix frame: axis z, centred on the heavy atoms), with the junction ends: a 5' end that carries the P of
    a phosphodiester keeps P, OP1, OP2 (no OP3), a bridged 5' end has none (and no HO5'), a bonded 3' end no
    HO3', the partner's 5' end at a junction a 5'-OH (:func:`_partner_end5`; ``linked``: the partner's junction
    ends as :func:`_partner_ends` gives them); free ends as asked for (``phosphate5``: P, OP1, OP2, OP3; otherwise
    a 5'-OH with the HO5' the builder gives one).  One build with every 5'-phosphate, centred by the builder (a
    long segment fits the PDB columns); the 5'-OH ends drop theirs."""
    partner = seg.partner if seg.partner is not None else "none"
    built = pdbtools.read_pdb(builder.build_helix(seg.sequence, partner, seg.chirality, na=seg.na, hydrogens=True,
                                                  phosphate5=True))
    end5, end3 = ends
    p5, p3 = _partner_ends(ends, linked)
    strands: List[List[_Res]] = []
    for si, residues in enumerate(built.segments):
        seq = seg.sequence if si == 0 else seg.partner
        out = []
        for j, res in enumerate(residues):
            e5 = (end5 if si == 0 else p5) if j == 0 else None
            e3 = (end3 if si == 0 else p3) if j == len(residues) - 1 else None
            hydroxyl = e5 == "nick" or (e5 == "free" and not phosphate5)  # a 5'-OH
            drop = set()
            if e5 == "p":
                drop = {"OP3"}
            elif e5 == "bridge" or hydroxyl:
                drop = {"P", "OP1", "OP2", "OP3"}
            if e3 == "bonded":
                drop.add("HO3'")
            atoms = [(pdbtools.standard_atom_name(res.name, a.name), np.array(a.xyz, dtype=np.float64))
                     for a in res.atoms]
            atoms = [(name, xyz) for name, xyz in atoms if name not in drop]
            if hydroxyl:  # HO5' as builder._strand_residue places it, before HO3'
                at = dict(atoms)
                ho5 = builder._place(at["C4'"], at["C5'"], at["O5'"], builder.BOND_H["oh"], 109.5, 180.0)
                where = next((i for i, (name, _x) in enumerate(atoms) if name == "HO3'"), len(atoms))
                atoms.insert(where, ("HO5'", ho5))
            out.append(_Res(k, si, j, seq[j], seg.chirality, seg.na, [name for name, _x in atoms],
                            np.array([xyz for _n, xyz in atoms], dtype=np.float64)))
        strands.append(out)
    heavy = np.vstack([r.xyz[[not n.startswith("H") for n in r.names]] for st in strands for r in st])
    mid = 0.5 * (heavy[:, 2].min() + heavy[:, 2].max())
    for st in strands:
        for r in st:
            r.xyz[:, 2] -= mid
    return strands


def _heavy_only(strands: List[List[_Res]]) -> None:
    """Drop the hydrogens of built strands (in place)."""
    for st in strands:
        for r in st:
            keep = [i for i, n in enumerate(r.names) if not n.startswith("H")]
            r.names = [r.names[i] for i in keep]
            r.xyz = r.xyz[keep]


def _codes(names: Sequence[str]) -> np.ndarray:
    """Element codes (``_ELEMENTS``) of atom names."""
    return np.array([_ELEMENTS.get(builder._element(n), 0) for n in names], dtype=np.int64)


class _Part:
    """The atoms of one segment (its local frame), hydrogens included when it has them, with their bond graph
    (each hydrogen bonded to the nearest heavy atom of its residue), for the fit and the reports."""

    def __init__(self, k: int, strands: List[List[_Res]]):
        self.k = k
        self.n = len(strands[0])
        self.paired = len(strands) > 1
        self.keys: List[Tuple[int, int, int, str]] = []  # (segment, strand, index, atom name)
        rows = []
        for st in strands:
            for r in st:
                for name, xyz in zip(r.names, r.xyz):
                    self.keys.append((k, r.strand, r.index, name))
                    rows.append(xyz)
        self.local = np.array(rows, dtype=np.float64).reshape(-1, 3)
        self.code = _codes([key[3] for key in self.keys])
        self.index = {key[1:]: i for i, key in enumerate(self.keys)}
        self.adj: List[List[int]] = [[] for _ in self.keys]
        for st in strands:
            for r in st:
                for a, b in template_bonds(r.base):
                    self._bond((r.strand, r.index, a), (r.strand, r.index, b))
                self._bond_hydrogens(r)
            for r1, r2 in zip(st, st[1:]):
                self._bond((r1.strand, r1.index, "O3'"), (r2.strand, r2.index, "P"))
        heavy = self.local[self.code != _H]
        self.centre = heavy.mean(axis=0)  # the rotation centre of a rigid placement
        self.span = float(heavy[:, 2].max() - heavy[:, 2].min())
        self.base_rows = {}
        for st in strands:
            for r in st:
                self.base_rows[(r.strand, r.index)] = [self.index[(r.strand, r.index, n)] for n in r.names
                                                       if not n.startswith("H") and n not in _BACKBONE]

    def _bond_hydrogens(self, r: _Res) -> None:
        hyd = [i for i, n in enumerate(r.names) if n.startswith("H")]
        heavy = [i for i, n in enumerate(r.names) if not n.startswith("H")]
        if not hyd or not heavy:
            return
        d = np.linalg.norm(r.xyz[hyd][:, None] - r.xyz[heavy][None], axis=-1)
        for h, row in zip(hyd, d):
            if row.min() < 1.35:
                self._bond((r.strand, r.index, r.names[h]), (r.strand, r.index, r.names[heavy[int(np.argmin(row))]]))

    def _bond(self, a, b) -> None:
        i, j = self.index.get(a), self.index.get(b)
        if i is not None and j is not None:
            self.adj[i].append(j)
            self.adj[j].append(i)

    def pair_rows(self, index: int) -> List[List[int]]:
        """Base atoms of strand-1 residue ``index`` and of its partner (the residue alone in a single strand)."""
        out = [self.base_rows[(0, index)]]
        if self.paired:
            out.append(self.base_rows[(1, self.n - 1 - index)])
        return out


def _pair_centroid(xyz: np.ndarray, rows: List[List[int]]) -> np.ndarray:
    return np.mean([xyz[r].mean(axis=0) for r in rows], axis=0)


def _bfs(adj: Sequence[Sequence[int]], start: int, depth: int = 5) -> Dict[int, int]:
    found = {start: 0}
    shell = [start]
    for d in range(1, depth + 1):
        shell = [b for a in shell for b in adj[a] if b not in found]
        for b in shell:
            found.setdefault(b, d)
    return found


def _separations(adj, start: int, size: int, depth: int = 5) -> np.ndarray:
    out = np.full(size, _FAR, dtype=np.int64)
    for i, d in _bfs(adj, start, depth).items():
        out[i] = d
    return out


# --- geometry helpers -----------------------------------------------------------------------------------

def _norms(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v * v).sum(axis=-1))


def _turned(points: np.ndarray, start: np.ndarray, end: np.ndarray, theta) -> np.ndarray:
    """``points`` (B, k, 3) turned by ``theta`` (B,) radians about the axes start -> end (B, 3 each) through ``end``
    (right-handed; Rodrigues)."""
    k = end - start
    k = (k / _norms(k)[:, None])[:, None, :]
    v = points - end[:, None, :]
    c, s = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    return end[:, None, :] + v * c + builder._cross(k, v) * s + k * (v * k).sum(axis=-1, keepdims=True) * (1.0 - c)


def _angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angles a-b-c in radians (points (3,) or (B, 3), broadcast)."""
    u, v = a - b, c - b
    return np.arccos(np.clip((u * v).sum(axis=-1) / (_norms(u) * _norms(v)), -1.0, 1.0))


def _phosphate_oxygens(oa: np.ndarray, ob: np.ndarray, p: np.ndarray, sign: float = 1.0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """OP1 / OP2 of a phosphate at ``p`` bonded to the ester oxygens ``oa`` and ``ob`` (points (3,) or (B, 3)),
    as :func:`mdatk.builder._phosphate_oxygens` rebuilds one: OP_BOND from P, OP_ANGLE apart in the plane that
    bisects oa-P-ob, pointing away from both.  ``sign`` +1: OP1 on the side of (oa - P) x (ob - P), the OP1 of a
    D nucleotide for oa = O3' (of the residue before) and ob = O5' (vol(P; O3', O5', OP1) > 0), and of a
    bridge; -1: the other side (an L nucleotide)."""
    u1 = (oa - p) / _norms(oa - p)[..., None]
    u2 = (ob - p) / _norms(ob - p)[..., None]
    bis = -(u1 + u2)
    bis = bis / _norms(bis)[..., None]
    n = builder._cross(u1, u2)
    n = n / _norms(n)[..., None]
    half = math.radians(OP_ANGLE / 2.0)
    along, across = OP_BOND * math.cos(half), sign * OP_BOND * math.sin(half)
    return p + along * bis + across * n, p + along * bis - across * n


def _contact_limits(sep: np.ndarray, ca: np.ndarray, cb: np.ndarray, close: float) -> np.ndarray:
    """Contact distances (Å; NaN: not scored) of atom pairs ``sep`` bonds apart through a junction with the
    element codes ``ca`` and ``cb``: heavy atoms ``close`` (four or five bonds) or CLASH (six or more), at least
    the model check's clash distance plus QC_MARGIN; H-H and H-heavy pairs H_CONTACT; pairs up to three bonds
    apart are not scored."""
    sep, ca, cb = np.asarray(sep), np.asarray(ca), np.asarray(cb)
    ha, hb = ca == _H, cb == _H
    heavy = np.maximum(np.where(sep >= 6, CLASH, close), _QC_LIMIT[ca, cb])
    out = np.where(ha & hb, H_CONTACT["hh"], np.where(ha | hb, H_CONTACT["hx"], heavy))
    return np.where(sep <= 3, np.nan, out)


def _bridge_limits(bsep: np.ndarray, cb: np.ndarray, cx: np.ndarray) -> np.ndarray:
    """Contact distances of the atoms of a bridge (P, OP1, OP2: codes ``cb``) and atoms ``bsep`` bonds from them
    (codes ``cx``): not scored up to two bonds; heavy atoms BRIDGE_CLASH, from four bonds at least the model
    check's clash distance plus QC_MARGIN; hydrogens H-heavy from four bonds."""
    bsep, cb, cx = np.asarray(bsep), np.asarray(cb), np.asarray(cx)
    heavy = np.where(bsep >= 4, np.maximum(BRIDGE_CLASH, _QC_LIMIT[cb, cx]), BRIDGE_CLASH)
    out = np.where(cx == _H, np.where(bsep >= 4, H_CONTACT["hx"], np.nan), heavy)
    return np.where(bsep <= 2, np.nan, out)


_STEP = float(np.finfo(np.float64).eps) ** 0.5


def _jacobian(batch: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Forward-difference Jacobian (m, n) of ``batch`` at ``x`` from one batched call, with the steps of scipy's
    '2-point' scheme (sqrt(eps) * max(1, |x|), away from zero)."""
    h = _STEP * np.where(x >= 0.0, 1.0, -1.0) * np.maximum(1.0, np.abs(x))
    h = (x + h) - x
    f = batch(np.vstack([x[None, :], x[None, :] + np.diag(h)]))
    return ((f[1:] - f[0]) / h[:, None]).T


def _dihedral(a, b, c, d) -> float:
    return float(builder._dihedral(np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(d)))


# --- the structure built so far -------------------------------------------------------------------------

class _Placed:
    """Atoms placed so far (world frame; hydrogens included when the parts have them) with their bond graph and
    element codes; keys (segment, strand, index, name) for nucleotides, ('PO2', junction, name) for a bridge."""

    def __init__(self):
        self.xyz = np.zeros((0, 3))
        self.keys: List[tuple] = []
        self.adj: List[List[int]] = []
        self.index: Dict[tuple, int] = {}
        self.code = np.zeros(0, dtype=np.int64)

    def add(self, keys: Sequence[tuple], xyz: np.ndarray, adj: Sequence[Sequence[int]] = ()) -> int:
        off = len(self.keys)
        self.xyz = np.vstack([self.xyz, np.asarray(xyz, dtype=np.float64).reshape(-1, 3)])
        self.keys += list(keys)
        self.adj += [[off + j for j in nb] for nb in adj] if adj else [[] for _ in keys]
        self.code = np.concatenate([self.code, _codes([key[-1] for key in keys])])
        for i, key in enumerate(keys):
            self.index[key] = off + i
        return off

    def bond(self, a: tuple, b: tuple) -> None:
        i, j = self.index[a], self.index[b]
        self.adj[i].append(j)
        self.adj[j].append(i)


@dataclass
class _Report:
    """What a junction looks like in the written model (distances in Å, angles in degrees)."""
    bonds: List[Tuple[tuple, tuple, float]]  # (atom key a, atom key b, length) as keys of _Placed
    angles: List[Tuple[str, float, float]]  # (label, degrees, target)
    stack: float
    contact: Tuple[float, tuple, tuple]  # closest heavy-atom contact between the segments: (distance, key, key)
    contact_h: Optional[Tuple[float, tuple, tuple]] = None  # ... with a hydrogen (a model with hydrogens)
    bridge_contact: Optional[Tuple[float, tuple, tuple]] = None
    # the heavy-atom pair farthest inside its contact distance (the fit's): (excess, distance, limit, key, key)
    worst: Optional[Tuple[float, float, float, tuple, tuple]] = None
    flex: Optional[Tuple[float, float, float, float]] = None  # gamma turn, beta turn, gamma, beta (degrees)
    flexed: Optional[tuple] = None  # (segment, strand, index) of the residue whose gamma / beta turned
    # the partner strands' phosphodiester (linked partners): as ``bonds`` and ``angles``, ``flex`` and ``flexed``
    partner_bond: Optional[Tuple[tuple, tuple, float]] = None
    partner_angles: Optional[List[Tuple[str, float, float]]] = None
    partner_flex: Optional[Tuple[float, float, float, float]] = None
    partner_flexed: Optional[tuple] = None


class _Junction:
    """The fit of segment ``j + 1`` onto the structure built so far (``placed``) through link ``j``; ``nas`` and
    ``hands``: the nucleic acids and hands of the two segments."""

    def __init__(self, j: int, link: str, placed: _Placed, prev: _Part, new: _Part, read: Sequence[bool],
                 frame: Tuple[np.ndarray, np.ndarray, np.ndarray], nas: Tuple[str, str],
                 hands: Tuple[str, str] = ("D", "D")):
        self.j, self.link, self.placed, self.prev, self.new = j, link, placed, prev, new
        self.bridge = link in BRIDGES
        self.ia = prev.n - 1 if read[j] else 0  # junction residue of the previous segment (strand 1)
        self.ib = 0 if read[j + 1] else new.n - 1  # ... of the new one
        ka = lambda name: placed.index[(prev.k, 0, self.ia, name)]  # noqa: E731
        kb = lambda name: new.index[(0, self.ib, name)]  # noqa: E731
        self.close = CLOSE["dna" if nas == ("dna", "dna") else "rna"]
        self.flex: Optional[str] = None
        self.p_side: Optional[str] = None
        self.gamma_rows: List[int] = []
        self.beta_rows: List[int] = []
        # rows that the fit moves within their own segment (the gamma / beta turn, the rebuilt phosphate oxygens):
        # rows of placed for a 5'-3' link (p_side 'a'), of the new segment for a 3'-5' link
        self.own_rows: List[int] = []
        # which side carries what
        if self.bridge:
            o, c = ("O3'", "C3'") if link == "3'-3'" else ("O5'", "C5'")
            self.oa, self.ca, self.ob, self.cb = ka(o), ka(c), kb(o), kb(c)
            self.ja, self.jb, self.span_bonds = self.oa, self.ob, 2
        else:
            p_side = "b" if link == "3'-5'" else "a"
            get = kb if p_side == "b" else ka
            other = ka if p_side == "b" else kb
            self.p_side = p_side
            self.o3, self.c3 = other("O3'"), other("C3'")
            self.p, self.o5, self.c5, self.c4 = get("P"), get("O5'"), get("C5'"), get("C4'")
            self.op = (get("OP1"), get("OP2"))
            self.ja, self.jb = (self.o3, self.p) if p_side == "b" else (self.p, self.o3)
            self.span_bonds = 1
            p_na, p_hand = (nas[1], hands[1]) if p_side == "b" else (nas[0], hands[0])
            self.op_sign = 1.0 if p_hand == "D" else -1.0  # OP1 / OP2 named by the hand of the residue with the P
            self.flex = p_side if p_na == "dna" else None  # an RNA phosphodiester stays rigid
            table, res = (new.index, (0, self.ib)) if p_side == "b" else (placed.index, (prev.k, 0, self.ia))
            if self.flex:  # the residue that carries the P
                self.gamma_rows = [table[res + (n,)] for n in _GAMMA if res + (n,) in table]
                self.beta_rows = [table[res + (n,)] for n in _BETA if res + (n,) in table]
            self.own_rows = sorted(set(self.gamma_rows) | set(self.op))
        # the frame of the previous segment: the start places the new one beyond its free end
        rot_prev, centre_prev, shift_prev = frame
        self.rot_prev = rot_prev
        origin = -rot_prev @ centre_prev + centre_prev + shift_prev  # the previous segment's local origin
        end_dir = rot_prev @ np.array([0.0, 0.0, -1.0 if read[j] else 1.0])  # strand 1 runs 5'->3' towards -z
        self.t0 = origin + end_dir * (0.5 * (prev.span + new.span) + STACK)
        self.pivot = new.local[self.jb].copy()  # the rotation centre of the fit: the new junction atom
        self.end_dir = end_dir
        # end pairs
        self.stack1 = _pair_centroid(placed.xyz, [[placed.index[prev.keys[r]] for r in rows]
                                                  for rows in prev.pair_rows(self.ia)])
        self.stack_rows = new.pair_rows(self.ib)
        self.stack2_local = _pair_centroid(new.local, self.stack_rows)
        # the contacts: static atoms (all placed but the ones a 5'-3' link moves), mobile atoms
        nf = len(placed.keys)
        self.side_rows = list(self.own_rows) if self.p_side == "a" else []  # rows of placed that move
        self.side_index = {row: k for k, row in enumerate(self.side_rows)}
        static = np.ones(nf, dtype=bool)
        static[self.side_rows] = False
        self.static_rows = np.nonzero(static)[0]
        from scipy.spatial import cKDTree
        self.tree = cKDTree(placed.xyz[self.static_rows])
        self.d_fixed = _separations(placed.adj, self.ja, nf)
        self.d_new = _separations(new.adj, self.jb, len(new.keys))
        nm = len(new.keys)
        self.n_mobile = nm + len(self.side_rows) + (3 if self.bridge else 0)
        # per mobile row: what it is (0 new segment, 1 moving atom of the structure so far, 2 bridge P, 3 OP) and
        # its element code
        self.kind = np.zeros(self.n_mobile, dtype=np.int64)
        self.kind[nm:nm + len(self.side_rows)] = 1
        bridge_codes = [_ELEMENTS["P"], _ELEMENTS["O"], _ELEMENTS["O"]] if self.bridge else []
        self.code = np.concatenate([new.code, placed.code[self.side_rows], np.array(bridge_codes, dtype=np.int64)])
        if self.bridge:
            self.kind[-3] = 2
            self.kind[-2:] = 3
        self.within: Dict[int, np.ndarray] = {}  # bond separations of the moving atoms within their own side
        if self.p_side == "a":
            for i in self.side_rows:
                self.within[i] = _separations(placed.adj, i, nf)
        elif self.p_side == "b":
            for i in self.own_rows:
                self.within[i] = _separations(new.adj, i, nm)
        # the mobile rows that do not move with the new segment as a rigid body (``_special``)
        self.special_rows = (np.array(self.own_rows, dtype=np.int64) if self.p_side == "b" else
                             np.arange(nm, self.n_mobile, dtype=np.int64))
        self.evaluations = 0
        self.cancel: Cancel = None

    # x: rotation vector, translation, then the (gamma, beta) turns or the bridge's P.  The methods below take a
    # batch X (B, n) of them, so that one call evaluates a point and all its finite-difference steps (_jacobian)
    def start(self, spin: float, tilt: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(self.rot_prev) * Rotation.from_euler("zx", [spin, tilt], degrees=True)
        # the start pose: the new segment turned about its centroid, the centroid moved by t0; the fit turns it
        # about its junction atom (``pivot``), which keeps the junction terms apart from the turn
        shift = self.t0 + (np.eye(3) - rot.as_matrix()) @ (self.new.centre - self.pivot)
        x = [rot.as_rotvec(), shift]
        if self.flex:
            x.append(np.zeros(2))
        elif self.bridge:
            x.append(self.placed.xyz[self.oa] + self.end_dir)
        return np.concatenate(x)

    def _moved(self, X: np.ndarray, local: np.ndarray) -> np.ndarray:
        """Local points (k, 3) of the new segment placed by every x of X: (B, k, 3)."""
        c = self.pivot
        return np.einsum("bij,kj->bki", builder._rotvecs(X[:, :3]), local - c) + c + X[:, None, 3:6]

    def _check_cancel(self) -> None:
        self.evaluations += 1
        if self.cancel is not None and self.evaluations % 100 == 0 and self.cancel():
            raise BuildError("cancelled")

    @staticmethod
    def _turn(xyz: np.ndarray, c4: np.ndarray, c5: np.ndarray, gamma_rows, beta_rows, X: np.ndarray) -> np.ndarray:
        """``xyz`` (B, k, 3) with the gamma turn about C4'-C5' (B, 3 each) and then the beta turn about C5'-O5'
        (O5' the first gamma row)."""
        out = np.array(xyz)
        out[:, gamma_rows] = _turned(out[:, gamma_rows], c4, c5, X[:, 6])
        out[:, beta_rows] = _turned(out[:, beta_rows], c5, out[:, gamma_rows[0]], X[:, 7])
        return out

    def _junction_points(self, X: np.ndarray):
        """The atoms of the junction terms ((3,) fixed or (B, 3)) and the new end pair's centroid (B, 3)."""
        new, xyz = self.new, self.placed.xyz
        if self.bridge:
            pts = self._moved(X, np.vstack([new.local[[self.cb, self.ob]], self.stack2_local]))
            return ({"ca": xyz[self.ca], "oa": xyz[self.oa], "cb": pts[:, 0], "ob": pts[:, 1], "p": X[:, 6:9]},
                    pts[:, 2])
        if self.p_side == "b":
            pts = self._moved(X, np.vstack([new.local[[self.c4, self.c5, self.o5, self.p]], self.stack2_local]))
            four = pts[:, :4]
            if self.flex:
                four = self._turn(four, four[:, 0], four[:, 1], [2, 3], [3], X)
            return {"o3": xyz[self.o3], "c3": xyz[self.c3], "o5": four[:, 2], "p": four[:, 3]}, pts[:, 4]
        pts = self._moved(X, np.vstack([new.local[[self.c3, self.o3]], self.stack2_local]))
        o5, p = xyz[self.o5], xyz[self.p]
        if self.flex:
            four = np.broadcast_to(xyz[[self.c4, self.c5, self.o5, self.p]], (len(X), 4, 3))
            four = self._turn(four, four[:, 0], four[:, 1], [2, 3], [3], X)
            o5, p = four[:, 2], four[:, 3]
        return {"o3": pts[:, 1], "c3": pts[:, 0], "o5": o5, "p": p}, pts[:, 2]

    def junction_terms(self, X: np.ndarray) -> np.ndarray:
        """Residuals of the junction: bond lengths and angles (and the turns' prior), the stacking (B, m)."""
        self._check_cancel()
        a, stack2 = self._junction_points(X)
        theta1, theta2 = math.radians(C_O_P), math.radians(O_P_O)
        p = a["p"]
        if self.bridge:
            w_bond, w_c, w_o = _W_BRIDGE
            cols = [w_bond * (_norms(p - a["oa"]) - LINK_BOND), w_bond * (_norms(p - a["ob"]) - LINK_BOND),
                    w_c * (_angles(a["ca"], a["oa"], p) - theta1), w_c * (_angles(a["cb"], a["ob"], p) - theta1),
                    w_o * (_angles(a["oa"], p, a["ob"]) - theta2)]
            weight = _W_STACK["bridge"]
        else:
            w_bond, w_c, w_o = _W_LINK
            cols = [w_bond * (_norms(p - a["o3"]) - LINK_BOND), w_c * (_angles(a["c3"], a["o3"], p) - theta1),
                    w_o * (_angles(a["o3"], p, a["o5"]) - theta2)]
            if self.flex:
                cols += [FLEX_PRIOR * X[:, 6], FLEX_PRIOR * X[:, 7]]
            weight = _W_STACK["link"]
        cols.append(weight * (_norms(stack2 - self.stack1) - STACK))
        out = np.empty((len(X), len(cols)))
        for k, col in enumerate(cols):
            out[:, k] = col
        return out

    def mobile(self, X: np.ndarray) -> np.ndarray:
        """Every atom the fit moves (B, N, 3): the new segment, then the moving atoms of the structure so far
        (5'-3') or the bridge (P, OP1, OP2).  A phosphodiester's P residue turns its gamma / beta (DNA) and gets
        its OP1 / OP2 rebuilt from P, O3' and O5' (``_phosphate_oxygens``)."""
        moved = self._moved(X, self.new.local)
        rows, special = self._special(X)
        if self.p_side == "b":
            moved[:, rows] = special
            return moved
        return np.concatenate([moved, special], axis=1)

    def mobile_rows(self, X: np.ndarray, plan) -> np.ndarray:
        """``mobile(X)[:, rows]`` for the rows of a ``plan`` (``_plan``), moving only those: the refits score the
        pairs near the junction, a small part of a long segment."""
        rows, rigid, at, found = plan
        y = np.empty((len(X), len(rows), 3))
        y[:, rigid] = self._moved(X, self.new.local[rows[rigid]])
        if len(at):
            y[:, at] = self._special(X)[1][:, found]
        return y

    def _special(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The mobile rows that do not move with the new segment as a rigid body, and their places (B, s, 3): a 3'-5'
        link's turning rows and rebuilt phosphate oxygens of the new segment, a 5'-3' link's of the structure so far
        (rows after the new segment's), or the bridge (P, OP1, OP2)."""
        new, xyz = self.new, self.placed.xyz
        if self.bridge:
            p = X[:, 6:9]
            op1, op2 = _phosphate_oxygens(xyz[self.oa], self._moved(X, new.local[[self.ob]])[:, 0], p)
            return self.special_rows, np.stack([p, op1, op2], axis=1)
        if self.p_side == "b":
            rows = list(self.own_rows)
            order = rows + [r for r in (self.c4, self.c5, self.o5, self.p) if r not in rows]
            local = {row: k for k, row in enumerate(order)}
            pts = self._moved(X, new.local[order])
            if self.flex:
                pts = self._turn(pts, pts[:, local[self.c4]], pts[:, local[self.c5]],
                                 [local[r] for r in self.gamma_rows], [local[r] for r in self.beta_rows], X)
            op1, op2 = _phosphate_oxygens(xyz[self.o3], pts[:, local[self.o5]], pts[:, local[self.p]], self.op_sign)
            pts[:, local[self.op[0]]], pts[:, local[self.op[1]]] = op1, op2
            return self.special_rows, pts[:, :len(rows)]
        rows, local, b = self.side_rows, self.side_index, len(X)
        side = np.array(np.broadcast_to(xyz[rows], (b, len(rows), 3)))
        if self.flex:
            side = self._turn(side, np.broadcast_to(xyz[self.c4], (b, 3)), np.broadcast_to(xyz[self.c5], (b, 3)),
                              [local[r] for r in self.gamma_rows], [local[r] for r in self.beta_rows], X)
        p = side[:, local[self.p]] if self.p in local else xyz[self.p]
        o5 = side[:, local[self.o5]] if self.o5 in local else xyz[self.o5]
        op1, op2 = _phosphate_oxygens(self._moved(X, new.local[[self.o3]])[:, 0], o5, p, self.op_sign)
        side[:, local[self.op[0]]], side[:, local[self.op[1]]] = op1, op2
        return self.special_rows, side

    def _plan(self, rows: np.ndarray):
        """What ``mobile_rows`` needs for the sorted mobile rows ``rows``: the rows, which of them move rigidly, and
        where the special rows (``_special``) go: (rows, rigid, positions in rows, indices into the special rows)."""
        nm = len(self.new.keys)
        special = self.special_rows
        at = np.searchsorted(rows, special)
        found = (at < len(rows)) & (rows[np.minimum(at, len(rows) - 1)] == special) if len(rows) else at < 0
        rigid = rows < nm
        rigid[at[found]] = False
        return rows, rigid, at[found], np.nonzero(found)[0]

    def _limits(self, sep: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
        """Contact distances for bond separations through the junction (NaN: not checked)."""
        return _contact_limits(sep, ca, cb, self.close)

    def candidates(self, x: np.ndarray):
        """The atom pairs a refit scores near ``x``: the mobile atoms there (N, 3), (mobile rows, static rows,
        contact distances) and (mobile rows, mobile rows, contact distances), pairs within their contact distance
        + _MARGIN."""
        y = self.mobile(x[None])[0]
        new, placed = self.new, self.placed
        nm = len(new.keys)
        lists = self.tree.query_ball_point(y, _REACH + _MARGIN)
        yi = np.repeat(np.arange(len(y)), [len(v) for v in lists])
        sj = np.fromiter((s for v in lists for s in v), dtype=np.int64, count=len(yi))
        fj = self.static_rows[sj]  # rows of placed
        thr = np.full(len(yi), np.nan)
        kind = self.kind[yi]
        sel = kind == 0  # the new segment against the structure so far: separations through the junction
        sep = self.d_new[yi[sel]] + self.span_bonds + self.d_fixed[fj[sel]]
        thr[sel] = self._limits(sep, self.code[yi[sel]], placed.code[fj[sel]])
        for k, row in enumerate(self.side_rows):  # moving atoms against the rest of their own structure
            sel = yi == nm + k
            d0 = np.linalg.norm(placed.xyz[fj[sel]] - placed.xyz[row], axis=1)  # as in the ideal helix
            thr[sel] = np.minimum(self._limits(self.within[row][fj[sel]], placed.code[row], placed.code[fj[sel]]),
                                  d0)  # NaN stays NaN
        if self.bridge:
            sel = kind >= 2
            bsep = np.where(kind[sel] == 2, 1, 2) + self.d_fixed[fj[sel]]
            thr[sel] = _bridge_limits(bsep, self.code[yi[sel]], placed.code[fj[sel]])
        dist = np.linalg.norm(y[yi] - placed.xyz[fj], axis=1)
        keep = ~np.isnan(thr) & (dist < np.nan_to_num(thr, nan=0.0) + _MARGIN)
        static = (yi[keep], fj[keep], thr[keep])
        # mobile-mobile pairs whose distance the fit changes
        pairs_i, pairs_k, pairs_t = [], [], []
        if self.p_side == "a":
            for k, row in enumerate(self.side_rows):
                # against the new segment, through the junction
                pairs_i.append(np.full(nm, nm + k))
                pairs_k.append(np.arange(nm))
                pairs_t.append(self._limits(self.d_new + self.span_bonds + self.d_fixed[row], placed.code[row],
                                            new.code))
                # against the moving atoms after it, as in the ideal helix
                later = np.array(self.side_rows[k + 1:], dtype=np.int64)
                if len(later):
                    d0 = np.linalg.norm(placed.xyz[later] - placed.xyz[row], axis=1)
                    pairs_i.append(np.full(len(later), nm + k))
                    pairs_k.append(nm + np.arange(k + 1, len(self.side_rows)))
                    pairs_t.append(np.minimum(self._limits(self.within[row][later], placed.code[row],
                                                           placed.code[later]), d0))
        elif self.p_side == "b":  # the moving atoms of the new segment against the rest of it (each pair once)
            own = np.array(self.own_rows, dtype=np.int64)
            for k, row in enumerate(self.own_rows):
                d0 = np.linalg.norm(new.local - new.local[row], axis=1)
                t = np.minimum(self._limits(self.within[row], new.code[row], new.code), d0)
                t[own[:k + 1]] = np.nan
                pairs_i.append(np.full(nm, row))
                pairs_k.append(np.arange(nm))
                pairs_t.append(t)
        elif self.bridge:
            for k in range(3):  # P, OP1, OP2 against the new segment
                bsep = (1 if k == 0 else 2) + self.d_new
                pairs_i.append(np.full(nm, nm + k))
                pairs_k.append(np.arange(nm))
                pairs_t.append(_bridge_limits(bsep, self.code[nm + k], new.code))
        if pairs_i:
            mi, mk, mt = (np.concatenate(v) for v in (pairs_i, pairs_k, pairs_t))
            d = np.linalg.norm(y[mi] - y[mk], axis=1)
            sel = ~np.isnan(mt) & (d < np.nan_to_num(mt, nan=0.0) + _MARGIN)
            mobile = (mi[sel], mk[sel], mt[sel])
        else:
            empty = np.zeros(0, dtype=np.int64)
            mobile = (empty, empty, np.zeros(0))
        # the rows the pairs use, and the pairs by their positions among them
        rows = np.unique(np.concatenate([static[0], mobile[0], mobile[1]]))
        where = lambda r: np.searchsorted(rows, r)  # noqa: E731
        return y, static, mobile, (self._plan(rows), where(static[0]), where(mobile[0]), where(mobile[1]))

    def residuals(self, X: np.ndarray, cand) -> np.ndarray:
        """Junction terms and one contact term per scored pair (B, m)."""
        _y0, (_si, sj, st), (_mi, _mk, mt), (plan, si, mi, mk) = cand
        y = self.mobile_rows(X, plan)
        over = np.concatenate([st - _norms(y[:, si] - self.placed.xyz[sj]), mt - _norms(y[:, mi] - y[:, mk])],
                              axis=1)
        return np.concatenate([self.junction_terms(X), _W_CONTACT * np.clip(over, 0.0, None)], axis=1)

    def fit(self, tick: Callable[[str], None]) -> Tuple[np.ndarray, float]:
        from scipy.optimize import least_squares

        tilts = TILTS if (self.flex or self.bridge) else TILTS_RIGID
        starts = []
        # A turning link keeps its turns at zero here: the rigid placement alone closes the junction terms exactly
        # (and the prior would pull the turns back), so the two turns would only lengthen the path.  The refits turn.
        if self.flex:
            def terms(X):
                return self.junction_terms(np.concatenate([X, np.zeros((len(X), 2))], axis=1))
        else:
            terms = self.junction_terms
        free = 6 if self.flex else None
        for spin in SPINS:
            for tilt in tilts:
                tick(f"junction {self.j + 1} ({self.link}): start {len(starts) + 1} of {len(SPINS) * len(tilts)}")
                x0 = self.start(spin, tilt)
                sol = least_squares(lambda x: terms(x[None])[0], x0[:free], jac=lambda x: _jacobian(terms, x),
                                    method="trf", max_nfev=MAX_NFEV[0], ftol=_PHASE1_TOL, xtol=_PHASE1_TOL,
                                    gtol=_PHASE1_TOL)
                starts.append(np.concatenate([sol.x, x0[len(sol.x):]]))
        best: Optional[Tuple[np.ndarray, float]] = None
        for k, x in enumerate(self.select(starts)):
            tick(f"junction {self.j + 1} ({self.link}): contacts, fit {k + 1} of {REFITS}")
            x, cost = self.refine(x)
            if best is None or cost < best[1]:
                best = (x, cost)
        assert best is not None
        return best

    def select(self, starts) -> List[np.ndarray]:
        """The junction fits to refit with the contacts: the REFITS lowest in the full cost (junction terms and
        contacts) at their solution, one per pose.  Most starts close the junction exactly (the junction terms
        alone leave the placement underdetermined), so their own cost does not tell them apart; the contacts do.
        A pose is the new junction residue's place (its atoms within _POSE of one already chosen: the same)."""
        scored = []
        for x in starts:
            res = self.residuals(x[None], self.candidates(x))[0]
            scored.append((0.5 * float(res @ res), len(scored), x))
        rows = [self.new.index[key[1:]] for key in self.new.keys if key[1:3] == (0, self.ib)]
        chosen: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        for _cost, _k, x in sorted(scored, key=lambda t: t[:2]):
            pose = self.mobile(x[None])[0][rows]
            if any(np.max(_norms(pose - other)) < _POSE for other in poses):
                continue
            chosen.append(x)
            poses.append(pose)
            if len(chosen) == REFITS:
                break
        return chosen

    def refine(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """The fit with the contacts from ``x``: pairs refreshed until no atom has moved past half the margin."""
        from scipy.optimize import least_squares

        for _ in range(_REFRESHES):
            cand = self.candidates(x)
            batch = lambda X: self.residuals(X, cand)  # noqa: E731
            sol = least_squares(lambda v: batch(v[None])[0], x, jac=lambda v: _jacobian(batch, v), method="trf",
                                max_nfev=MAX_NFEV[1])
            moved = float(np.max(np.linalg.norm(self.mobile(sol.x[None])[0] - cand[0], axis=1)))
            x = sol.x
            if moved <= 0.5 * _MARGIN:
                break
        res = self.residuals(x[None], self.candidates(x))[0]
        return x, 0.5 * float(res @ res)


def _turn_points(xyz: np.ndarray, c4: np.ndarray, c5: np.ndarray, gamma_rows, beta_rows, gamma: np.ndarray,
                 beta: np.ndarray) -> np.ndarray:
    """``xyz`` (B, k, 3) with the gamma turn ``gamma`` (B,) about C4'-C5' (B, 3 each) and then the beta turn
    ``beta`` about C5'-O5' (O5' the first gamma row), as ``_Junction._turn``."""
    out = np.array(xyz)
    out[:, gamma_rows] = _turned(out[:, gamma_rows], c4, c5, gamma)
    out[:, beta_rows] = _turned(out[:, beta_rows], c5, out[:, gamma_rows[0]], beta)
    return out


@dataclass
class _Diester:
    """One phosphodiester of a junction with linked partners: the side of its P ('a' the structure built so far,
    'b' the new segment), its atoms (rows of ``placed`` on side a, of the new segment's local atoms on side b; the
    O3' and C3' on the other side), the hand sign of its OP1 and its turns (``var``: the column of gamma in x, beta
    the next; -1 when it does not turn)."""
    p_side: str
    o3: int
    c3: int
    p: int
    o5: int
    c5: int
    c4: int
    op: Tuple[int, int]
    sign: float
    var: int
    gamma_rows: List[int]
    beta_rows: List[int]

    @property
    def flex(self) -> bool:
        return self.var >= 0


class _LinkedJunction(_Junction):
    """The fit of segment ``j + 1`` through phosphodiester link ``j`` with the partner strands joined as well
    (``link_partners``): two phosphodiesters, strand 1's and the partners', one with its P on each side (3'-5':
    strand 1's P on the new segment, the partners' on the partner 5' end of the structure so far; 5'-3' the other
    way round).  Both P residues turn their gamma / beta (RNA too: a rigid body alone cannot close two links and
    stack), and both phosphates get their OP1 / OP2 rebuilt.  x: rotation vector, translation, then (gamma, beta)
    of each diester in ``bonds`` order (strand 1's, the partners').  The contacts are scored through whichever
    link is the shorter way; everything else is :class:`_Junction`'s."""

    def __init__(self, j: int, link: str, placed: _Placed, prev: _Part, new: _Part, read: Sequence[bool],
                 frame: Tuple[np.ndarray, np.ndarray, np.ndarray], nas: Tuple[str, str], hands: Tuple[str, str],
                 bonds: Sequence[Tuple[tuple, tuple]]):
        self.j, self.link, self.placed, self.prev, self.new = j, link, placed, prev, new
        self.bridge = False
        self.flex = None
        self.p_side = None
        self.ia = prev.n - 1 if read[j] else 0
        self.ib = 0 if read[j + 1] else new.n - 1
        self.close = CLOSE["dna" if nas == ("dna", "dna") else "rna"]

        def row(key: tuple, name: str) -> int:
            return new.index[key[1:] + (name,)] if key[0] == new.k else placed.index[key + (name,)]

        self.bonds: List[_Diester] = []
        var = 6
        for o_key, p_key in bonds:
            p_side = "b" if p_key[0] == new.k else "a"
            table, res = (new.index, p_key[1:]) if p_side == "b" else (placed.index, p_key)
            gamma = [table[res + (n,)] for n in _GAMMA if res + (n,) in table]
            beta = [table[res + (n,)] for n in _BETA if res + (n,) in table]
            hand = hands[1] if p_side == "b" else hands[0]
            self.bonds.append(_Diester(p_side, row(o_key, "O3'"), row(o_key, "C3'"), row(p_key, "P"), row(p_key, "O5'"),
                                       row(p_key, "C5'"), row(p_key, "C4'"), (row(p_key, "OP1"), row(p_key, "OP2")),
                                       1.0 if hand == "D" else -1.0, var, gamma, beta))
            var += 2
        self.n_var = var
        side = [b for b in self.bonds if b.p_side == "a"]
        own = [b for b in self.bonds if b.p_side == "b"]
        assert len(side) == len(own) == 1
        self.bond_a, self.bond_b = side[0], own[0]
        a, b = self.bond_a, self.bond_b
        # the junction atoms of each diester (the P or O3' on either side): placed rows, new rows
        self.ends = [(d.o3, d.p) if d.p_side == "b" else (d.p, d.o3) for d in self.bonds]
        self.ja, self.jb = self.ends[0]
        self.span_bonds = 1
        self.own_rows = sorted(set(b.gamma_rows) | set(b.op))  # rows of the new segment that move within it
        self.side_rows = sorted(set(a.gamma_rows) | set(a.op))  # rows of placed that move
        self.side_index = {r: k for k, r in enumerate(self.side_rows)}
        rot_prev, centre_prev, shift_prev = frame
        self.rot_prev = rot_prev
        origin = -rot_prev @ centre_prev + centre_prev + shift_prev
        end_dir = rot_prev @ np.array([0.0, 0.0, -1.0 if read[j] else 1.0])
        self.t0 = origin + end_dir * (0.5 * (prev.span + new.span) + STACK)
        self.pivot = new.local[self.jb].copy()
        self.end_dir = end_dir
        self.stack1 = _pair_centroid(placed.xyz, [[placed.index[prev.keys[r]] for r in rows]
                                                  for rows in prev.pair_rows(self.ia)])
        self.stack_rows = new.pair_rows(self.ib)
        self.stack2_local = _pair_centroid(new.local, self.stack_rows)
        nf, nm = len(placed.keys), len(new.keys)
        static = np.ones(nf, dtype=bool)
        static[self.side_rows] = False
        self.static_rows = np.nonzero(static)[0]
        from scipy.spatial import cKDTree
        self.tree = cKDTree(placed.xyz[self.static_rows])
        self.d_fixed_all = [_separations(placed.adj, fa, nf) for fa, _fb in self.ends]
        self.d_new_all = [_separations(new.adj, fb, nm) for _fa, fb in self.ends]
        self.d_fixed, self.d_new = self.d_fixed_all[0], self.d_new_all[0]
        self.n_mobile = nm + len(self.side_rows)
        self.kind = np.zeros(self.n_mobile, dtype=np.int64)
        self.kind[nm:] = 1
        self.code = np.concatenate([new.code, placed.code[self.side_rows]])
        self.within = {i: _separations(placed.adj, i, nf) for i in self.side_rows}
        self.within_new = {i: _separations(new.adj, i, nm) for i in self.own_rows}
        self.special_rows = np.concatenate([np.array(self.own_rows, dtype=np.int64),
                                            np.arange(nm, self.n_mobile, dtype=np.int64)])
        self.evaluations = 0
        self.cancel: Cancel = None

    def _through(self, i_new: np.ndarray, j_fixed: np.ndarray) -> np.ndarray:
        """Bond separations through the junction of new atoms ``i_new`` and placed atoms ``j_fixed`` (broadcast):
        the shorter way, through either diester."""
        return np.minimum(*[dn[i_new] + 1 + df[j_fixed] for dn, df in zip(self.d_new_all, self.d_fixed_all)])

    def start(self, spin: float, tilt: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(self.rot_prev) * Rotation.from_euler("zx", [spin, tilt], degrees=True)
        shift = self.t0 + (np.eye(3) - rot.as_matrix()) @ (self.new.centre - self.pivot)
        return np.concatenate([rot.as_rotvec(), shift, np.zeros(self.n_var - 6)])

    def _junction_points(self, X: np.ndarray):
        """Per diester (in ``bonds`` order) its O3', C3', O5' and P ((3,) fixed or (B, 3)), and the new end pair's
        centroid (B, 3)."""
        new, xyz, a, b = self.new, self.placed.xyz, self.bond_a, self.bond_b
        pts = self._moved(X, np.vstack([new.local[[b.c4, b.c5, b.o5, b.p, a.c3, a.o3]], self.stack2_local]))
        four = _turn_points(pts[:, :4], pts[:, 0], pts[:, 1], [2, 3], [3], X[:, b.var], X[:, b.var + 1])
        at_b = {"o3": xyz[b.o3], "c3": xyz[b.c3], "o5": four[:, 2], "p": four[:, 3]}
        side = np.broadcast_to(xyz[[a.c4, a.c5, a.o5, a.p]], (len(X), 4, 3))
        side = _turn_points(side, side[:, 0], side[:, 1], [2, 3], [3], X[:, a.var], X[:, a.var + 1])
        at_a = {"o3": pts[:, 5], "c3": pts[:, 4], "o5": side[:, 2], "p": side[:, 3]}
        return [at_a if d is a else at_b for d in self.bonds], pts[:, 6]

    def junction_terms(self, X: np.ndarray) -> np.ndarray:
        """Residuals of both diesters (bond, C3'-O3'-P, O3'-P-O5', then the turns' priors) and the stacking."""
        self._check_cancel()
        points, stack2 = self._junction_points(X)
        theta1, theta2 = math.radians(C_O_P), math.radians(O_P_O)
        w_bond, w_c, w_o = _W_LINK
        cols = []
        for at in points:
            p = at["p"]
            cols += [w_bond * (_norms(p - at["o3"]) - LINK_BOND), w_c * (_angles(at["c3"], at["o3"], p) - theta1),
                     w_o * (_angles(at["o3"], p, at["o5"]) - theta2)]
        cols += [FLEX_PRIOR * X[:, k] for k in range(6, self.n_var)]
        cols.append(_W_STACK["link"] * (_norms(stack2 - self.stack1) - STACK))
        out = np.empty((len(X), len(cols)))
        for k, col in enumerate(cols):
            out[:, k] = col
        return out

    def mobile(self, X: np.ndarray) -> np.ndarray:
        """Every atom the fit moves (B, N, 3): the new segment, then the moving atoms of the structure so far."""
        nm = len(self.new.keys)
        out = np.concatenate([self._moved(X, self.new.local), np.zeros((len(X), self.n_mobile - nm, 3))], axis=1)
        rows, special = self._special(X)
        out[:, rows] = special
        return out

    def _special(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The turning rows and rebuilt phosphate oxygens of both diesters: the new segment's (side b), then the
        structure so far's (side a, rows after the new segment's)."""
        new, xyz, a, b = self.new, self.placed.xyz, self.bond_a, self.bond_b
        own = list(self.own_rows)
        order = own + [r for r in (b.c4, b.c5, b.o5, b.p, a.o3) if r not in own]
        local = {row: k for k, row in enumerate(order)}
        pts = self._moved(X, new.local[order])
        pts = _turn_points(pts, pts[:, local[b.c4]], pts[:, local[b.c5]], [local[r] for r in b.gamma_rows],
                           [local[r] for r in b.beta_rows], X[:, b.var], X[:, b.var + 1])
        op1, op2 = _phosphate_oxygens(xyz[b.o3], pts[:, local[b.o5]], pts[:, local[b.p]], b.sign)
        pts[:, local[b.op[0]]], pts[:, local[b.op[1]]] = op1, op2
        rows, sl, n = self.side_rows, self.side_index, len(X)
        side = np.array(np.broadcast_to(xyz[rows], (n, len(rows), 3)))
        side = _turn_points(side, np.broadcast_to(xyz[a.c4], (n, 3)), np.broadcast_to(xyz[a.c5], (n, 3)),
                            [sl[r] for r in a.gamma_rows], [sl[r] for r in a.beta_rows], X[:, a.var], X[:, a.var + 1])
        op1, op2 = _phosphate_oxygens(pts[:, local[a.o3]], side[:, sl[a.o5]], side[:, sl[a.p]], a.sign)
        side[:, sl[a.op[0]]], side[:, sl[a.op[1]]] = op1, op2
        return self.special_rows, np.concatenate([pts[:, :len(own)], side], axis=1)

    def candidates(self, x: np.ndarray):
        """As :meth:`_Junction.candidates`, with the separations through either diester and both kinds of moving
        atoms: the structure so far's (against the rest of it as in the ideal helix, against the new segment
        through the junction) and the new segment's (against the rest of it as in the ideal helix)."""
        y = self.mobile(x[None])[0]
        new, placed = self.new, self.placed
        nm = len(new.keys)
        lists = self.tree.query_ball_point(y, _REACH + _MARGIN)
        yi = np.repeat(np.arange(len(y)), [len(v) for v in lists])
        sj = np.fromiter((s for v in lists for s in v), dtype=np.int64, count=len(yi))
        fj = self.static_rows[sj]
        thr = np.full(len(yi), np.nan)
        sel = self.kind[yi] == 0
        thr[sel] = self._limits(self._through(yi[sel], fj[sel]), self.code[yi[sel]], placed.code[fj[sel]])
        for k, row in enumerate(self.side_rows):
            sel = yi == nm + k
            d0 = np.linalg.norm(placed.xyz[fj[sel]] - placed.xyz[row], axis=1)
            thr[sel] = np.minimum(self._limits(self.within[row][fj[sel]], placed.code[row], placed.code[fj[sel]]), d0)
        dist = np.linalg.norm(y[yi] - placed.xyz[fj], axis=1)
        keep = ~np.isnan(thr) & (dist < np.nan_to_num(thr, nan=0.0) + _MARGIN)
        static = (yi[keep], fj[keep], thr[keep])
        pairs_i, pairs_k, pairs_t = [], [], []
        everything = np.arange(nm)
        for k, row in enumerate(self.side_rows):
            pairs_i.append(np.full(nm, nm + k))
            pairs_k.append(everything)
            pairs_t.append(self._limits(self._through(everything, np.full(nm, row)), placed.code[row], new.code))
            later = np.array(self.side_rows[k + 1:], dtype=np.int64)
            if len(later):
                d0 = np.linalg.norm(placed.xyz[later] - placed.xyz[row], axis=1)
                pairs_i.append(np.full(len(later), nm + k))
                pairs_k.append(nm + np.arange(k + 1, len(self.side_rows)))
                pairs_t.append(np.minimum(self._limits(self.within[row][later], placed.code[row],
                                                       placed.code[later]), d0))
        own = np.array(self.own_rows, dtype=np.int64)
        for k, row in enumerate(self.own_rows):
            d0 = np.linalg.norm(new.local - new.local[row], axis=1)
            t = np.minimum(self._limits(self.within_new[row], new.code[row], new.code), d0)
            t[own[:k + 1]] = np.nan
            pairs_i.append(np.full(nm, row))
            pairs_k.append(everything)
            pairs_t.append(t)
        mi, mk, mt = (np.concatenate(v) for v in (pairs_i, pairs_k, pairs_t))
        d = np.linalg.norm(y[mi] - y[mk], axis=1)
        sel = ~np.isnan(mt) & (d < np.nan_to_num(mt, nan=0.0) + _MARGIN)
        mobile = (mi[sel], mk[sel], mt[sel])
        rows = np.unique(np.concatenate([static[0], mobile[0], mobile[1]]))
        where = lambda r: np.searchsorted(rows, r)  # noqa: E731
        return y, static, mobile, (self._plan(rows), where(static[0]), where(mobile[0]), where(mobile[1]))

    def fit(self, tick: Callable[[str], None]) -> Tuple[np.ndarray, float]:
        from scipy.optimize import least_squares

        starts = []
        total = len(SPINS) * len(TILTS) * len(_LINKED_STARTS)
        for mode in _LINKED_STARTS:  # the rigid body alone (turns at zero), then the turns as well
            free = 6 if mode == "rigid" else self.n_var
            turns = np.zeros(self.n_var - free)

            def terms(X, turns=turns):
                return self.junction_terms(np.concatenate([X, np.broadcast_to(turns, (len(X), len(turns)))], axis=1))
            for spin in SPINS:
                for tilt in TILTS:
                    tick(f"junction {self.j + 1} ({self.link}, partners linked): start {len(starts) + 1} of {total}")
                    x0 = self.start(spin, tilt)
                    sol = least_squares(lambda x: terms(x[None])[0], x0[:free], jac=lambda x: _jacobian(terms, x),
                                        method="trf", max_nfev=_LINKED_NFEV.get(mode, MAX_NFEV[0]), ftol=_PHASE1_TOL,
                                        xtol=_PHASE1_TOL, gtol=_PHASE1_TOL)
                    starts.append(np.concatenate([sol.x, turns]))
        best: Optional[Tuple[np.ndarray, float]] = None
        for k, x in enumerate(self.select(starts)):
            tick(f"junction {self.j + 1} ({self.link}, partners linked): contacts, fit {k + 1} of {REFITS}")
            x, cost = self.refine(x)
            if best is None or cost < best[1]:
                best = (x, cost)
        assert best is not None
        return best


# --- the fit of a design ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Fit:
    """A fitted design: per segment its rigid placement world = R (local - c) + c + t, per junction its x."""
    frames: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    xs: Tuple[np.ndarray, ...]
    costs: Tuple[float, ...]

    @property
    def cost(self) -> float:
        return float(sum(self.costs))


_FITS: "OrderedDict[tuple, _Fit]" = OrderedDict()


def _fit_key(segments: Sequence[Segment], links: Sequence[str], phosphate5: bool, linked: bool = False) -> tuple:
    return tuple(segments), tuple(links), bool(phosphate5), bool(linked)


def _fit(segments: Sequence[Segment], links: Sequence[str], phosphate5: bool, tick: Callable[[str], None],
         cancel: Cancel, linked: bool = False) -> _Fit:
    """Fit every junction in design order (cached: the fit depends on the design, the 5'-phosphates and whether
    the partners are linked only; the segments' hydrogens are always in its contacts, written or not)."""
    key = _fit_key(segments, links, phosphate5, linked)
    if key in _FITS:
        _FITS.move_to_end(key)
        return _FITS[key]
    read = _read53(links)
    parts = [_Part(k, _segment_residues(k, seg, _ends(links, read, k), phosphate5, linked))
             for k, seg in enumerate(segments)]
    placed = _Placed()
    placed.add(parts[0].keys, parts[0].local, parts[0].adj)
    frames = [(np.eye(3), parts[0].centre.copy(), np.zeros(3))]
    xs, costs = [], []
    for j, link in enumerate(links):
        prev, new = parts[j], parts[j + 1]
        nas, hands = (segments[j].na, segments[j + 1].na), (segments[j].chirality, segments[j + 1].chirality)
        if linked:
            junction = _LinkedJunction(j, link, placed, prev, new, read, frames[j], nas, hands,
                                       _junction_bonds(segments, links, j, True))
        else:
            junction = _Junction(j, link, placed, prev, new, read, frames[j], nas, hands)
        junction.cancel = cancel
        x, cost = junction.fit(tick)
        _place(placed, junction, x)
        frames.append((builder._rotvecs(x[None, :3])[0], junction.pivot.copy(), np.array(x[3:6])))
        xs.append(np.array(x))
        costs.append(cost)
    fit = _Fit(tuple(frames), tuple(xs), tuple(costs))
    _FITS[key] = fit
    while len(_FITS) > _CACHE_SIZE:
        _FITS.popitem(last=False)
    return fit


def _place(placed: _Placed, junction: _Junction, x: np.ndarray) -> None:
    """Add the new segment (and a bridge) to the structure built so far, with the junction bonds."""
    y = junction.mobile(x[None])[0]
    new = junction.new
    nm = len(new.keys)
    placed.add(new.keys, y[:nm], new.adj)
    if junction.side_rows:
        placed.xyz[junction.side_rows] = y[nm:nm + len(junction.side_rows)]
    if isinstance(junction, _LinkedJunction):
        for d in junction.bonds:  # O3' - P of both diesters
            o3 = placed.keys[d.o3] if d.p_side == "b" else new.keys[d.o3]
            p = new.keys[d.p] if d.p_side == "b" else placed.keys[d.p]
            placed.bond(o3, p)
    elif junction.bridge:
        keys = [("PO2", junction.j, n) for n in ("P", "OP1", "OP2")]
        placed.add(keys, y[nm:])
        for other in (keys[1], keys[2], placed.keys[junction.oa], new.keys[junction.ob]):
            placed.bond(keys[0], other)
    elif junction.p_side == "b":  # O3' placed before, the P on the new segment
        placed.bond(placed.keys[junction.o3], new.keys[junction.p])
    else:  # the P placed before (5'-3'), the O3' on the new segment
        placed.bond(placed.keys[junction.p], new.keys[junction.o3])


# --- the written model ----------------------------------------------------------------------------------

def _junction_residues(segments: Sequence, links: Sequence[str], j: int) -> Tuple[int, int]:
    """Strand-1 indices of the residues joined by link ``j``: the free end of segment j as read, the end of
    segment j + 1 the link names."""
    read = _read53(links)
    ia = len(segments[j].sequence) - 1 if read[j] else 0
    ib = 0 if read[j + 1] else len(segments[j + 1].sequence) - 1
    return ia, ib


def _junction_bonds(segments: Sequence, links: Sequence[str], j: int, linked: bool = False
                    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """The phosphodiesters of link ``j`` as (O3' residue, P residue), residues (segment, strand, index): strand
    1's, then (``linked``) the partner strands' (none for a bridge).  The partners' runs the other way: its P is
    on the side of strand 1's O3'."""
    if links[j] in BRIDGES:
        return []
    ia, ib = _junction_residues(segments, links, j)
    a, b = (j, 0, ia), (j + 1, 0, ib)
    out = [(a, b) if links[j] == "3'-5'" else (b, a)]
    if linked:
        pa = (j, 1, len(segments[j].sequence) - 1 - ia)  # the partner residues of the junction pairs
        pb = (j + 1, 1, len(segments[j + 1].sequence) - 1 - ib)
        out.append((pb, pa) if links[j] == "3'-5'" else (pa, pb))
    return out


def _bond_turns(segments: Sequence[Segment], links: Sequence[str], j: int, x: np.ndarray, linked: bool = False,
                sign: float = 1.0) -> List[Optional[Tuple[float, float]]]:
    """(gamma, beta) turns (radians, times ``sign``) of the P residue of every phosphodiester of link ``j``
    (:func:`_junction_bonds`) in the fit ``x``, None for one that stays rigid: unlinked a DNA P residue turns
    (x[6:8]); linked, every P residue (x[6:8] strand 1's, x[8:10] the partners')."""
    if not linked:
        return [(sign * float(x[6]), sign * float(x[7])) if _flexed(segments, links, j) else None
                for _bond in _junction_bonds(segments, links, j)]
    return [(sign * float(x[6 + 2 * k]), sign * float(x[7 + 2 * k]))
            for k, _bond in enumerate(_junction_bonds(segments, links, j, True))]


def _flexed(segments: Sequence[Segment], links: Sequence[str], j: int) -> Optional[Tuple[int, int]]:
    """(segment, strand-1 index) of the residue whose gamma / beta link ``j`` turns (a DNA residue carrying the P
    of a phosphodiester), or None."""
    link = links[j]
    if link in BRIDGES:
        return None
    ia, ib = _junction_residues(segments, links, j)
    k, i = (j + 1, ib) if link == "3'-5'" else (j, ia)
    return (k, i) if segments[k].na == "dna" else None


def _turn_residue(res: _Res, gamma: float, beta: float) -> None:
    """Gamma (about C4'-C5': O5', H5', H5'' and beyond) and beta (about C5'-O5': P and beyond) turns, in place."""
    g = [i for i, n in enumerate(res.names) if n in _GAMMA]
    b = [i for i, n in enumerate(res.names) if n in _BETA]
    c4, c5 = res.at("C4'")[None], res.at("C5'")[None]
    res.xyz[g] = _turned(res.xyz[g][None], c4, c5, np.array([gamma]))[0]
    res.xyz[b] = _turned(res.xyz[b][None], c5, res.at("O5'")[None], np.array([beta]))[0]


def _coordinates(segments: Sequence[Segment], links: Sequence[str], fit: _Fit, hydrogens: bool, phosphate5: bool,
                 linked: bool = False) -> Tuple[List[List[List[_Res]]], Dict[int, Dict[str, np.ndarray]]]:
    """Every residue of the design placed as ``fit`` places it (the gamma / beta turns, the rebuilt phosphate
    oxygens of every phosphodiester junction, the partner strands' too when ``linked``; hydrogens carried with
    their atoms, or dropped) and the atoms of the bridges by junction."""
    read = _read53(links)
    builds = [_segment_residues(k, seg, _ends(links, read, k), phosphate5, linked) for k, seg in enumerate(segments)]
    for k, strands in enumerate(builds):
        rot, centre, shift = fit.frames[k]
        for st in strands:
            for r in st:
                r.xyz = (r.xyz - centre) @ rot.T + centre + shift
    bridges: Dict[int, Dict[str, np.ndarray]] = {}
    for j, link in enumerate(links):
        x = fit.xs[j]
        ia, ib = _junction_residues(segments, links, j)
        a, b = builds[j][0][ia], builds[j + 1][0][ib]
        if link in BRIDGES:
            o = "O3'" if link == "3'-3'" else "O5'"
            p = np.array(x[6:9])
            op1, op2 = _phosphate_oxygens(a.at(o), b.at(o), p)
            bridges[j] = {"P": p, "OP1": op1, "OP2": op2}
            continue
        for (o_key, p_key), turns in zip(_junction_bonds(segments, links, j, linked),
                                         _bond_turns(segments, links, j, x, linked)):
            o_res, p_res = builds[o_key[0]][o_key[1]][o_key[2]], builds[p_key[0]][p_key[1]][p_key[2]]
            if turns is not None:
                _turn_residue(p_res, *turns)
            sign = 1.0 if p_res.hand == "D" else -1.0  # OP1 / OP2 named by the hand of the residue with the P
            op1, op2 = _phosphate_oxygens(o_res.at("O3'"), p_res.at("O5'"), p_res.at("P"), sign)
            p_res.xyz[p_res.names.index("OP1")], p_res.xyz[p_res.names.index("OP2")] = op1, op2
    if not hydrogens:
        for strands in builds:
            _heavy_only(strands)
    return builds, bridges


def _mirror(builds, bridges) -> None:
    """x -> -x for every atom, in place (the names stay: an L residue is the mirror image of the D residue)."""
    for strands in builds:
        for st in strands:
            for r in st:
                r.xyz[:, 0] *= -1.0
    for atoms in bridges.values():
        for name in atoms:
            atoms[name] = atoms[name] * np.array([-1.0, 1.0, 1.0])


def _centre(builds, bridges) -> Tuple[np.ndarray, np.ndarray]:
    """Move the model so that the bounding box of its heavy atoms is centred at the origin (in place; a model and
    its mirror image move by mirrored shifts, so they stay exact mirror images); returns the lowest and highest
    coordinates (x y z) of all its atoms."""
    rows = [r.xyz[[not n.startswith("H") for n in r.names]] for strands in builds for st in strands for r in st]
    rows += [np.vstack(list(atoms.values())) for atoms in bridges.values()]
    heavy = np.vstack(rows)
    mid = 0.5 * (heavy.min(axis=0) + heavy.max(axis=0))
    every = []
    for strands in builds:
        for st in strands:
            for r in st:
                r.xyz = r.xyz - mid
                every.append(r.xyz)
    for atoms in bridges.values():
        for name in atoms:
            atoms[name] = atoms[name] - mid
            every.append(atoms[name][None])
    every = np.vstack(every)
    return every.min(axis=0), every.max(axis=0)


def _layout(segments: Sequence[Segment], links: Sequence[str], linked: bool = False
            ) -> List[Tuple[List[Tuple[int, int, int]], Optional[int]]]:
    """The chains as written: per chain its residues (segment, strand, index) 5'->3' and the junction whose PO2
    follows it as a TER segment of its own with the chain's id (None).  The runs of strand 1 between bridges in
    design order, then the partner strands in segment order (``linked``: one partner chain, the partner strands
    in the reverse of strand 1's written order, :func:`_partner_order`)."""
    if linked:
        strand1 = [(k, 0, i) for k in _partner_order(segments, links)[::-1] for i in range(len(segments[k].sequence))]
        partner = [(k, 1, i) for k in _partner_order(segments, links) for i in range(len(segments[k].partner))]
        return [(strand1, None), (partner, None)]
    read = _read53(links)
    runs = [[0]]
    for j, link in enumerate(links):
        if link in BRIDGES:
            runs.append([j + 1])
        else:
            runs[-1].append(j + 1)
    chains = []
    for run in runs:
        order = run if read[run[0]] else run[::-1]  # a run read 3'->5' is written 5'->3' as well
        items = [(k, 0, i) for k in order for i in range(len(segments[k].sequence))]
        last = run[-1]
        chains.append((items, last if last < len(links) and links[last] in BRIDGES else None))
    for k, seg in enumerate(segments):
        if seg.partner is not None:
            chains.append(([(k, 1, i) for i in range(len(seg.partner))], None))
    return chains


def _chain_ids(chains, count: int) -> List[str]:
    if chains is None or (isinstance(chains, str) and not chains.strip()):
        return list(pdbtools.CHAIN_IDS[:count])
    ids = [c.strip() for c in chains.split(",")] if isinstance(chains, str) else [str(c).strip() for c in chains]
    what = (f"the design writes {count} chains (the strand-1 chains between bridges in design order, then the "
            "partner strands in segment order)")
    if len(ids) != count:
        raise BuildError(f"chains: {what}; {len(ids)} chain id(s) given")
    if any(len(c) != 1 or c.isspace() for c in ids):
        raise BuildError(f"chains: chain ids must be single characters, got {', '.join(repr(c) for c in ids)}")
    if len(set(ids)) != len(ids):
        raise BuildError(f"chains: every chain needs its own id, got {', '.join(ids)}")
    return ids


def _terminal(seg_ends: Tuple[str, str], strand: int, index: int, n: int, linked: bool = False) -> str:
    """Terminal state of a residue from its free ends only (a junction end is no terminal; the partner strands'
    ends are all free unless ``linked``)."""
    e5, e3 = seg_ends if strand == 0 else ("free", "free") if not linked else _partner_ends(seg_ends, True)
    five, three = index == 0 and e5 == "free", index == n - 1 and e3 == "free"
    return "N" if five and three else "5" if five else "3" if three else ""


_DEFAULT_NAMES = {("D", "rna"): "A, C, G, U", ("L", "rna"): "0A, 0C, 0G, 0U (the CCD L-ribonucleotides)",
                  ("D", "dna"): "DA, DC, DG, DT", ("L", "dna"): "0DA, 0DC, 0DG, 0DT (the CCD L-deoxyribonucleotides)"}


def _label(chain: str, resseq: int) -> str:
    return f"{chain}:{resseq}"


def _assemble(segments, links, builds, bridges, scheme: str, terminals: bool, ids: Sequence[str],
              linked: bool = False):
    """The Structure (chains, names, records, PO2 residues as TER segments of their own), the labels (segment,
    strand, index) -> (chain, resseq, name), the PO2 of every bridge (junction -> (chain, resseq)) and the
    residues the name notes list ('terminal': those with a terminal variant of their name)."""
    read = _read53(links)
    ends = [_ends(links, read, k) for k in range(len(segments))]
    s = Structure()
    labels: Dict[Tuple[int, int, int], Tuple[str, int, str]] = {}
    po2: Dict[int, Tuple[str, int]] = {}
    special: Dict[str, List[Residue]] = {"kept_phosphate": [], "single_l": [], "l_dna": [], "plain_l": [],
                                         "terminal": []}
    for (items, bridge), chain in zip(_layout(segments, links, linked), ids):
        seg_res: List[Residue] = []
        for number, (k, strand, i) in enumerate(items, 1):
            r = builds[k][strand][i]
            n = len(segments[k].sequence)
            term = _terminal(ends[k], strand, i, n, linked) if terminals else ""
            kept = scheme == "ccd" and r.na == "rna" and r.hand == "L" and term in ("5", "N") and "P" in r.names
            if kept:  # 0A5 ... stand for a 5'-OH end; a 5' end with its phosphate is the CCD component itself
                term = ""
            name = pdbtools.scheme_name(r.base, r.hand, term, scheme, r.na)
            res = Residue(name, chain, number, chirality=r.hand)
            for what, listed in (("kept_phosphate", kept),
                                 ("single_l", scheme == "amber" and term == "N" and r.hand == "L" and r.na == "rna"),
                                 ("l_dna", scheme == "amber" and r.hand == "L" and r.na == "dna"),
                                 ("plain_l", scheme == "standard" and r.hand == "L"),
                                 ("terminal", name != pdbtools.scheme_name(r.base, r.hand, "", scheme, r.na))):
                if listed:
                    special[what].append(res)
            for atom_name, xyz in zip(r.names, r.xyz):  # a CCD component's atoms by its own names (0DA: N9A ...)
                res.atoms.append(Atom("ATOM", pdbtools.component_atom_name(name, atom_name), "", builder._tail(xyz),
                                      builder._element(atom_name)))
            labels[(k, strand, i)] = (chain, number, res.name)
            seg_res.append(res)
        s.segments.append(seg_res)
        if bridge is not None:  # a TER segment of its own: no residue of the chain's sequence
            res = Residue("PO2", chain, len(items) + 1)
            for atom_name in ("P", "OP1", "OP2"):
                res.atoms.append(Atom("HETATM", atom_name, "", builder._tail(bridges[bridge][atom_name]),
                                      builder._element(atom_name)))
            po2[bridge] = (chain, len(items) + 1)
            s.segments.append([res])
    return s, labels, po2, special


def _listed(residues: Sequence[Residue]) -> str:
    return pdbtools._listing([_label(r.chain, r.resseq) for r in residues])


def _name_notes(segments, s: Structure, scheme: str, chosen: Optional[str], special, phosphate5: bool,
                bridges: bool) -> List[str]:
    kinds = list(dict.fromkeys((seg.chirality, seg.na) for seg in segments))
    record = "ATOM" if scheme == "standard" else "HETATM"  # the standard scheme writes every record ATOM
    po2 = f"; the bridging phosphates PO2 {record} (TER segments of their own)" if bridges else ""
    if chosen is None:
        if len(kinds) == 1:
            return [f"Residue names {_DEFAULT_NAMES[kinds[0]]}, ATOM records{po2}"]
        parts = [f"{_DEFAULT_NAMES[k]} ({k[0]}-{k[1].upper()})" for k in kinds]
        return [f"Residue names {'; '.join(parts)}, ATOM records{po2}"]
    names: Dict[str, List[str]] = {}
    for res in s.residues():
        if res.name != "PO2":
            nt = pdbtools.parse_residue_name(res.name)
            kind = f"{res.chirality}-{(nt.sugar if nt else 'rna').upper()}"
            names.setdefault(kind, [])
            if res.name not in names[kind]:
                names[kind].append(res.name)
    terminal = (" - terminal variants at the free strand ends only (the residues bonded through a junction have "
                "internal names)") if special["terminal"] else ""
    out = [f"Residue names of the {scheme} scheme: " + "; ".join(f"{', '.join(v)} ({k})" for k, v in names.items())
           + f"{terminal}{po2}"]
    if special["kept_phosphate"]:
        out.append("5'-terminal L residues with a 5'-phosphate keep the CCD component names (0A ..., which include "
                   "P/OP1/OP2/OP3; the 5' variants 0A5 ... have a 5'-OH): " + _listed(special["kept_phosphate"]))
    if special["single_l"]:
        out.append("nucleic12_LRNA.lib has no unit for a single-residue L segment; wrote the 5' unit (its O3' lacks "
                   "a hydrogen): " + _listed(special["single_l"]))
    if special["l_dna"]:
        out.append(f"L-DNA: {len(special['l_dna'])} residue(s) ({pdbtools.ranges_text(special['l_dna'])}) have MDATK's "
                   "Amber-style names LDA, LDC, LDG, LDT (three characters at the chain ends too: the Amber build maps "
                   "the ends to the terminal units); AmberTools ships no L-DNA library, so tleap needs residue units "
                   "of these names (the Amber build generates them)")
    if special["plain_l"]:
        out.append(f"WARNING: the standard scheme gives {len(special['plain_l'])} L residue(s) plain D names "
                   f"({pdbtools.ranges_text(special['plain_l'])}): the names no longer record the hand. Use this copy "
                   "only for programs that ignore chirality (map-model scores such as Q-score, DSSR), never for "
                   "tleap, refinement or analyses that read residue names")
    if phosphate5 and scheme == "amber":
        dna = any(seg.na == "dna" for seg in segments)
        out.append("WARNING: the Amber 5' units have no 5'-phosphate (tleap cannot type P/OP1/OP2/OP3 of "
                   f"{'A5/LA5, DA5 / LDA (L-DNA 5 prime end)' if dna else 'A5/LA5'} ...); build without the 5'-phosphate for tleap")
    return out


def _model_graph(segments, links, builds, bridges, linked: bool = False) -> Tuple[_Placed, List[_Part]]:
    """The atoms of the written model (hydrogens when it has them) with every bond (the junctions' too, the linked
    partner strands' as well), for the reports."""
    placed = _Placed()
    parts = [_Part(k, strands) for k, strands in enumerate(builds)]
    for part in parts:
        placed.add(part.keys, part.local, part.adj)
    for j, link in enumerate(links):
        ia, ib = _junction_residues(segments, links, j)
        if link in BRIDGES:
            o = "O3'" if link == "3'-3'" else "O5'"
            keys = [("PO2", j, n) for n in ("P", "OP1", "OP2")]
            placed.add(keys, np.vstack([bridges[j][n] for n in ("P", "OP1", "OP2")]))
            for other in (keys[1], keys[2], (j, 0, ia, o), (j + 1, 0, ib, o)):
                placed.bond(keys[0], other)
        else:
            for o_key, p_key in _junction_bonds(segments, links, j, linked):
                placed.bond(o_key + ("O3'",), p_key + ("P",))
    return placed, parts


def _closest(xyz: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray, sep: Callable[[np.ndarray, np.ndarray],
             np.ndarray], minimum: int) -> Tuple[float, int, int]:
    """Closest pair (distance, row a, row b) of rows_a x rows_b at least ``minimum`` bonds apart (within 8 Å)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz[rows_a])
    best = (math.inf, -1, -1)
    for k, near in enumerate(tree.query_ball_point(xyz[rows_b], 8.0)):
        if not near:
            continue
        a = rows_a[np.array(near)]
        b = np.full(len(a), rows_b[k])
        ok = sep(a, b) >= minimum
        if not np.any(ok):
            continue
        d = np.linalg.norm(xyz[a[ok]] - xyz[b[ok]], axis=1)
        i = int(np.argmin(d))
        if d[i] < best[0]:
            best = (float(d[i]), int(a[ok][i]), int(b[ok][i]))
    return best


def _worst(xyz: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray,
           limits: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Optional[Tuple[float, float, float, int, int]]:
    """The pair of rows_a x rows_b farthest inside its contact distance ``limits(a, b)`` (NaN: not scored):
    (excess, distance, contact distance, row a, row b), None if no pair is inside."""
    from scipy.spatial import cKDTree

    if not len(rows_a) or not len(rows_b):
        return None
    tree = cKDTree(xyz[rows_a])
    best: Optional[Tuple[float, float, float, int, int]] = None
    for k, near in enumerate(tree.query_ball_point(xyz[rows_b], _REACH)):
        if not near:
            continue
        a = rows_a[np.array(near)]
        b = np.full(len(a), rows_b[k])
        limit = limits(a, b)
        d = np.linalg.norm(xyz[a] - xyz[b], axis=1)
        excess = np.where(np.isnan(limit), -np.inf, np.nan_to_num(limit, nan=0.0) - d)
        i = int(np.argmax(excess))
        if excess[i] > 0.0 and (best is None or excess[i] > best[0]):
            best = (float(excess[i]), float(d[i]), float(limit[i]), int(a[i]), int(b[i]))
    return best


def _diester(at, o_key: tuple, p_key: tuple) -> Tuple[Tuple[tuple, tuple, float], List[Tuple[str, float, float]]]:
    """The O3'-P bond (O3' key, P key, length) and the angles (label, degrees, target) of a phosphodiester of the
    written model (``at(residue key, name)``: an atom's place)."""
    deg = math.degrees
    p, o3 = at(p_key, "P"), at(o_key, "O3'")
    bond = (o_key + ("O3'",), p_key + ("P",), float(_norms(o3 - p)))
    angles = [("C3'-O3'-P", deg(float(_angles(at(o_key, "C3'"), o3, p))), C_O_P),
              ("O3'-P-O5'", deg(float(_angles(o3, p, at(p_key, "O5'")))), O_P_O),
              ("O3'-P-OP1", deg(float(_angles(o3, p, at(p_key, "OP1")))), O_P_OP),
              ("O3'-P-OP2", deg(float(_angles(o3, p, at(p_key, "OP2")))), O_P_OP)]
    return bond, angles


def _turned_now(at, key: tuple, turns: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """(gamma turn, beta turn, gamma, beta) in degrees of a P residue of the written model."""
    gamma = _dihedral(at(key, "O5'"), at(key, "C5'"), at(key, "C4'"), at(key, "C3'"))
    beta = _dihedral(at(key, "P"), at(key, "O5'"), at(key, "C5'"), at(key, "C4'"))
    return math.degrees(turns[0]), math.degrees(turns[1]), gamma, beta


def _report(segments, links, placed: _Placed, parts: List[_Part], j: int,
            turns: Sequence[Optional[Tuple[float, float]]], linked: bool = False) -> _Report:
    """What junction ``j`` of the written model looks like; ``turns``: per phosphodiester (:func:`_bond_turns`)."""
    link = links[j]
    ia, ib = _junction_residues(segments, links, j)
    xyz, index, code = placed.xyz, placed.index, placed.code
    ka, kb = (j, 0, ia), (j + 1, 0, ib)
    at = lambda key, name: xyz[index[key + (name,)]]  # noqa: E731
    deg = math.degrees
    if link in BRIDGES:
        o, c = ("O3'", "C3'") if link == "3'-3'" else ("O5'", "C5'")
        pkey = ("PO2", j, "P")
        p = xyz[index[pkey]]
        bonds = [(ka + (o,), pkey, float(_norms(at(ka, o) - p))), (pkey, kb + (o,), float(_norms(p - at(kb, o))))]
        angles = [(f"{c}-{o}-P", deg(float(_angles(at(ka, c), at(ka, o), p))), C_O_P),
                  (f"P-{o}-{c}", deg(float(_angles(p, at(kb, o), at(kb, c)))), C_O_P),
                  (f"{o}-P-{o}", deg(float(_angles(at(ka, o), p, at(kb, o)))), O_P_O)]
        ja, jb, span = index[ka + (o,)], index[kb + (o,)], 2
    else:
        diesters = _junction_bonds(segments, links, j, linked)
        bond, angles = _diester(at, *diesters[0])
        bonds = [bond]
        # the junction atoms of every diester: (on segment j's side, on segment j + 1's side)
        ends = [(index[o + ("O3'",)], index[p + ("P",)]) if o[0] == j else (index[p + ("P",)], index[o + ("O3'",)])
                for o, p in diesters]
        ja, jb = ends[0]
        span = 1
    stack1 = _pair_centroid(parts[j].local, parts[j].pair_rows(ia))
    stack2 = _pair_centroid(parts[j + 1].local, parts[j + 1].pair_rows(ib))
    n = len(placed.keys)
    d_a, d_b = _separations(placed.adj, ja, n), _separations(placed.adj, jb, n)
    side = np.array([(key[0] <= j) if key[0] != "PO2" else (key[1] < j) for key in placed.keys])
    new = np.array([key[0] == j + 1 for key in placed.keys])
    heavy = code != _H
    through = lambda a, b: d_a[a] + span + d_b[b]  # noqa: E731
    if linked and link not in BRIDGES:  # the shorter way, through either diester
        seps = [(_separations(placed.adj, ea, n), _separations(placed.adj, eb, n)) for ea, eb in ends]
        through = lambda a, b: np.minimum(*[sa[a] + 1 + sb[b] for sa, sb in seps])  # noqa: E731
    contact = _closest(xyz, np.nonzero(side & heavy)[0], np.nonzero(new & heavy)[0], through, 4)
    report = _Report(bonds, angles, float(_norms(stack2 - stack1)),
                     (contact[0], placed.keys[contact[1]], placed.keys[contact[2]]))
    if not heavy.all():  # a model with hydrogens: the closest contact of a hydrogen too
        best = min(_closest(xyz, np.nonzero(side & ~heavy)[0], np.nonzero(new)[0], through, 4),
                   _closest(xyz, np.nonzero(side)[0], np.nonzero(new & ~heavy)[0], through, 4))
        if best[1] >= 0:
            report.contact_h = (best[0], placed.keys[best[1]], placed.keys[best[2]])
    # the heavy-atom pair farthest inside its contact distance (the same fit, so the same pair, with or without
    # hydrogens; the hydrogens' closest contact is ``contact_h``)
    close = CLOSE["dna" if (segments[j].na, segments[j + 1].na) == ("dna", "dna") else "rna"]
    worst = [_worst(xyz, np.nonzero(side & heavy)[0], np.nonzero(new & heavy)[0],
                    lambda a, b: _contact_limits(through(a, b), code[a], code[b], close))]
    if link in BRIDGES:
        rows = np.array([index[("PO2", j, name)] for name in ("P", "OP1", "OP2")])
        others = np.array([i for i, key in enumerate(placed.keys) if not (key[0] == "PO2" and key[1] == j)])
        near = np.minimum(d_a, d_b)  # bonds from the nearer of the two ester oxygens
        bsep = lambda a, b: near[a] + np.where(b == rows[0], 1, 2)  # noqa: E731
        best = _closest(xyz, others[heavy[others]], rows, bsep, 3)
        report.bridge_contact = (best[0], placed.keys[best[2]], placed.keys[best[1]])
        worst.append(_worst(xyz, others[heavy[others]], rows, lambda a, b: np.where(bsep(a, b) >= 4, _bridge_limits(
            bsep(a, b), code[b], code[a]), np.nan)))  # pairs four or more bonds apart, as the model check
    worst = [w for w in worst if w is not None]
    if worst:
        excess, d, limit, a, b = max(worst)
        report.worst = (excess, d, limit, placed.keys[a], placed.keys[b])
    if link not in BRIDGES:
        for k, ((_o_key, p_key), turn) in enumerate(zip(diesters, turns)):
            if k:  # the partner strands' diester
                report.partner_bond, report.partner_angles = _diester(at, _o_key, p_key)
            if turn is None:
                continue
            if k:
                report.partner_flex, report.partner_flexed = _turned_now(at, p_key, turn), p_key
            else:
                report.flex, report.flexed = _turned_now(at, p_key, turn), p_key
    return report


def _atom_label(key: tuple, labels, po2) -> str:
    if key[0] == "PO2":
        chain, number = po2[key[1]]
        return f"{chain}:{number} {key[2]}"
    chain, number, _name = labels[key[:3]]
    return f"{chain}:{number} {key[3]}"


def _res_label(key: tuple, labels) -> str:
    chain, number, _name = labels[key[:3]]
    return f"{chain}:{number}"


def _junction_note(j: int, link: str, rep: _Report, labels, po2) -> str:
    name = lambda key: _atom_label(key, labels, po2)  # noqa: E731
    if link in BRIDGES:
        (a, p, la), (_p, b, lb) = rep.bonds
        what = (f"Junction {j + 1} ({link}, {name(a)} - {name(p)} - {name(b)}, a bridging phosphate PO2): O-P "
                f"{la:.2f} / {lb:.2f} Å, {rep.angles[0][0]} {rep.angles[0][1]:.1f}° / {rep.angles[1][1]:.1f}°, "
                f"{rep.angles[2][0]} {rep.angles[2][1]:.1f}°")
    else:
        (o, p, length), = rep.bonds
        (c_label, c_value, _t), (o_label, o_value, _t), (_l, op1, _t), (_l, op2, _t) = rep.angles
        what = (f"Junction {j + 1} ({link}, {name(o)} - {name(p)}): O3'-P {length:.2f} Å, {c_label} {c_value:.1f}°, "
                f"{o_label} {o_value:.1f}°, O3'-P-OP1 / OP2 {op1:.1f}° / {op2:.1f}°")
    if rep.flex is not None:
        g, b, gamma, beta = rep.flex
        what += (f"; gamma of {_res_label(rep.flexed, labels)} turned {g:+.1f}° (now {gamma:.0f}°), beta {b:+.1f}° "
                 f"(now {beta:.0f}°)")
    if rep.partner_bond is not None:
        o, p, length = rep.partner_bond
        (c_label, c_value, _t), (o_label, o_value, _t), (_l, op1, _t), (_l, op2, _t) = rep.partner_angles
        what += (f"; partner strands joined ({name(o)} - {name(p)}): O3'-P {length:.2f} Å, {c_label} {c_value:.1f}°, "
                 f"{o_label} {o_value:.1f}°, O3'-P-OP1 / OP2 {op1:.1f}° / {op2:.1f}°")
        if rep.partner_flex is not None:
            g, b, gamma, beta = rep.partner_flex
            what += (f"; gamma of {_res_label(rep.partner_flexed, labels)} turned {g:+.1f}° (now {gamma:.0f}°), beta "
                     f"{b:+.1f}° (now {beta:.0f}°)")
    what += (f"; end pairs {rep.stack:.2f} Å apart (target {STACK:g}); closest heavy-atom contact between the "
             f"segments {rep.contact[0]:.2f} Å ({name(rep.contact[1])} - {name(rep.contact[2])})")
    if rep.contact_h is not None:
        what += (f"; closest contact of a hydrogen between the segments {rep.contact_h[0]:.2f} Å "
                 f"({name(rep.contact_h[1])} - {name(rep.contact_h[2])})")
    if rep.bridge_contact is not None:
        what += (f"; closest contact of the bridging phosphate {rep.bridge_contact[0]:.2f} Å "
                 f"({name(rep.bridge_contact[1])} - {name(rep.bridge_contact[2])})")
    return what


def _span(chain: str, lo: int, hi: int) -> str:
    return f"{chain}:{lo}" + (f"-{hi}" if hi != lo else "")


def _header(segments, links, reps: List[_Report], labels, po2) -> List[str]:
    out = [f"REMARK 951 MDATK CHIMERA {describe((segments, links))}"]
    partner_chain = {}
    for (k, strand, i), (chain, number, _n) in labels.items():
        if strand == 1:
            partner_chain.setdefault(k, (chain, []))[1].append(number)
    for k, seg in enumerate(segments):
        first, last = labels[(k, 0, 0)], labels[(k, 0, len(seg.sequence) - 1)]
        lo, hi = sorted((first[1], last[1]))
        form = "B-FORM" if seg.na == "dna" else "A-FORM"
        if k in partner_chain:
            chain, numbers = partner_chain[k]
            partner = f"PARTNER {_span(chain, min(numbers), max(numbers))}"
        else:
            partner = "SINGLE STRAND"
        out.append(f"REMARK 951 SEGMENT {k + 1} {_span(first[0], lo, hi)} {seg.kind} {form}, {partner}, {seg.sequence}")
    for j, (link, rep) in enumerate(zip(links, reps)):
        name = lambda key: _atom_label(key, labels, po2).replace(" ", ":")  # noqa: E731
        if link in BRIDGES:
            (a, p, la), (_p, b, lb) = rep.bonds
            (label, a1, _t), (_l, a2, _t), (label_o, a3, _t) = rep.angles
            what = (f"{name(a)}-{name(p)}-{name(b)} (PO2): O-P {la:.2f}/{lb:.2f} A, {label} {a1:.1f}/{a2:.1f}, "
                    f"{label_o} {a3:.1f} DEG")
        else:
            (o, p, length), = rep.bonds
            (c_label, c_value, _t), (o_label, o_value, _t), (_l, op1, _t), (_l, op2, _t) = rep.angles
            what = (f"{name(o)}-{name(p)}: O3'-P {length:.2f} A, {c_label} {c_value:.1f}, {o_label} {o_value:.1f}, "
                    f"O3'-P-OP1/OP2 {op1:.1f}/{op2:.1f} DEG")
        if rep.flex:
            what += f", GAMMA/BETA TURNED {rep.flex[0]:+.1f}/{rep.flex[1]:+.1f}"
        what += f", END PAIRS {rep.stack:.2f} A, CLOSEST {rep.contact[0]:.2f} A"
        if rep.bridge_contact:
            what += f", BRIDGE {rep.bridge_contact[0]:.2f} A"
        out.append(f"REMARK 951 JUNCTION {j + 1} {link} {what}")
        if rep.partner_bond is not None:  # the partner strands' phosphodiester, written the same way
            o, p, length = rep.partner_bond
            (c_label, c_value, _t), (o_label, o_value, _t), (_l, op1, _t), (_l, op2, _t) = rep.partner_angles
            what = (f"{name(o)}-{name(p)}: O3'-P {length:.2f} A, {c_label} {c_value:.1f}, {o_label} {o_value:.1f}, "
                    f"O3'-P-OP1/OP2 {op1:.1f}/{op2:.1f} DEG")
            if rep.partner_flex:
                what += f", GAMMA/BETA TURNED {rep.partner_flex[0]:+.1f}/{rep.partner_flex[1]:+.1f}"
            out.append(f"REMARK 951 JUNCTION {j + 1} {link} {what}, PARTNER STRANDS")
    out.append("REMARK 951 RIGID-BODY JUNCTION: MINIMISE BEFORE PRODUCTION")
    return out


def _segment_notes(segments, builds, labels) -> List[str]:
    """The builder's notes of every segment: base pairs, clashing mismatches, G·U / G·T wobbles."""
    out = []
    for k, seg in enumerate(segments):
        if seg.partner is None:
            continue
        kinds = builder.pair_types(seg.sequence, seg.partner, seg.na)
        strands = []
        for strand in (0, 1):
            residues = []
            for r in builds[k][strand]:
                chain, number, name = labels[(k, strand, r.index)]
                res = Residue(name, chain, number)
                res.atoms = [Atom("ATOM", n, "", builder._tail(x)) for n, x in zip(r.names, r.xyz)]
                residues.append(res)
            strands.append(residues)
        out += [f"Segment {k + 1}: {n}" for n in builder._mismatch_notes(Structure(segments=strands), seg.sequence,
                                                                           seg.partner, kinds, seg.na)]
        if kinds.count("wobble"):
            out.append(f"Segment {k + 1}: {kinds.count('wobble')} G·{'T' if seg.na == 'dna' else 'U'} pair(s) built "
                       "with an approximate wobble geometry (bases turned in their plane about C1'); minimise before "
                       "production")
    return out


def _amber_notes(segments, links, s: Structure, labels, po2) -> List[str]:
    """What tleap needs for the bridges: a PO2 unit and bond commands (tleap numbers residues in file order)."""
    bridged = [j for j, link in enumerate(links) if link in BRIDGES]
    if not bridged:
        return []
    order = {(r.chain, r.resseq): n for n, r in enumerate(s.residues(), 1)}
    bonds = []
    for j in bridged:
        o = "O3'" if links[j] == "3'-3'" else "O5'"
        ia, ib = _junction_residues(segments, links, j)
        p = po2[j]
        for key in ((j, 0, ia), (j + 1, 0, ib)):
            chain, number, _name = labels[key]
            bonds.append(f"bond m.{order[p]}.P m.{order[(chain, number)]}.{o} ({p[0]}:{p[1]} P - {chain}:{number} {o})")
    out = ["WARNING: tleap has no unit for the bridging phosphate PO2 (P, OP1, OP2) and does not read LINK records: "
           "load a PO2 unit before loadpdb (m = loadpdb ...) and bond the LINK atoms after it (residues numbered in "
           "file order across the chains): " + "; ".join(bonds)]
    five = []
    for j in bridged:
        if links[j] == "5'-5'":
            ia, ib = _junction_residues(segments, links, j)
            five += [labels[(j, 0, ia)], labels[(j + 1, 0, ib)]]
    if five:
        out.append("WARNING: the residues bonded to a PO2 through O5' (" + ", ".join(f"{c}:{n}" for c, n, _x in five)
                   + ") have internal names but no 5'-phosphate of their own: their units must not add one (tleap "
                   "adds the atoms a unit has and the file lacks)")
    return out


def _closure_warning(j: int, link: str, rep: _Report, cost: float, labels, po2) -> Optional[str]:
    """A WARNING note for a junction of the written model off its targets by more than ``CLOSED`` (the O-P
    bonds, the angles, a phosphodiester's end pairs, the pair farthest inside its contact distance), or None."""
    misses = [f"O-P {length:.2f} Å (target {LINK_BOND:.2f})" for _a, _b, length in rep.bonds
              if abs(length - LINK_BOND) > CLOSED["bond"]]
    misses += [f"{label} {value:.1f}° (target {target:g}°)" for label, value, target in rep.angles
               if abs(value - target) > CLOSED["angle"]]
    if rep.partner_bond is not None:  # the partner strands' phosphodiester
        o, p, length = rep.partner_bond
        where = f"partner strands {_atom_label(o, labels, po2)} - {_atom_label(p, labels, po2)}"
        if abs(length - LINK_BOND) > CLOSED["bond"]:
            misses.append(f"{where} O-P {length:.2f} Å (target {LINK_BOND:.2f})")
        misses += [f"{where} {label} {value:.1f}° (target {target:g}°)" for label, value, target in rep.partner_angles
                   if abs(value - target) > CLOSED["angle"]]
    if link not in BRIDGES and rep.stack > STACK + CLOSED["stack"]:
        misses.append(f"end pairs {rep.stack:.2f} Å apart (target {STACK:g})")
    if rep.worst is not None and rep.worst[0] > CLOSED["contact"]:
        _excess, d, limit, a, b = rep.worst
        misses.append(f"{_atom_label(a, labels, po2)} - {_atom_label(b, labels, po2)} {d:.2f} Å (contact "
                      f"distance {limit:.2f})")
    if not misses:
        return None
    return (f"WARNING: junction {j + 1} ({link}) did not close: " + ", ".join(misses) + f"; fit cost {cost:.4f}: "
            "minimise before use, or change the design (other sequences, longer segments)")


def _units(segments: Sequence[Segment], links: Sequence[str], linked: bool = False) -> int:
    """Progress steps of one fit: every start and refit of every junction."""
    out = 0
    for j, link in enumerate(links):
        free = linked or link in BRIDGES or _flexed(segments, links, j) is not None
        out += len(SPINS) * len(TILTS if free else TILTS_RIGID) * (len(_LINKED_STARTS) if linked else 1) + REFITS
    return out


def build_chimera(design, na: str = "rna", scheme: Optional[str] = None, terminals: bool = True,
                  hydrogens: bool = False, phosphate5: bool = False, chains=None, notes: Optional[List[str]] = None,
                  progress: Progress = None, cancel: Cancel = None, link_partners: bool = False) -> str:
    """PDB text of a D/L chimera (see the module docstring).

    ``design``: a design string (:func:`parse_design`; ``na`` the nucleic acid of the segments that name none) or
    the (segments, links[, notes]) it parses to.  ``scheme``: residue names as in :mod:`mdatk.pdbtools` (None:
    ``builder.DEFAULT_SCHEME``); ``terminals``: 5'/3' variants at the free strand ends for the schemes that have
    them; ``hydrogens`` adds all hydrogens (carried with their atoms; the fit keeps them clear whether or not
    they are written, so the heavy atoms are the same either way); ``phosphate5`` keeps a 5'-phosphate (P, OP1,
    OP2, OP3) on the free 5' ends, but for a partner strand's 5' end at a junction, which keeps a 5'-OH.
    ``chains``: chain ids ('A,B,C,D' or a list) for the strand-1 chains in design order and then the partner
    strands in segment order (:func:`chain_count` of them).  ``link_partners`` joins the partner strands as well, by
    a phosphodiester at every junction, into one continuous partner chain (:func:`describe_partner`); every link
    must be a phosphodiester and every segment paired (:class:`BuildError` otherwise), and both diesters of a
    junction are fitted together (:class:`_LinkedJunction`).  ``notes`` (a list) receives the build notes;
    ``progress(fraction, message)`` is called during the fit and ``cancel()`` returning True stops it with a
    :class:`BuildError` 'cancelled'.  A junction that misses its targets is named in a 'WARNING: junction ...'
    note."""
    if isinstance(design, str):
        segments, links, parse_notes = parse_design(design, na)
    else:
        segments, links = _design(design, na)
        parse_notes = list(design[2]) if len(design) > 2 else []
    chosen = scheme or None
    scheme = scheme or builder.DEFAULT_SCHEME
    if scheme not in builder.SCHEMES:
        raise BuildError(f"unknown residue name scheme {scheme!r} (choose from {', '.join(builder.SCHEMES)})")
    linked = bool(link_partners)
    if linked:
        _check_linked(segments, links)
    layout = _layout(segments, links, linked)
    ids = _chain_ids(chains, len(layout))
    atoms = _atom_count(segments, links, hydrogens, phosphate5, linked)
    _check_size(atoms, _ter_count(segments, links, linked), len(layout))
    _check_extent(segments, links)
    twin = [seg.flipped() for seg in segments]
    units = _units(segments, links, linked)
    done = [0]

    def ticker(offset: int, what: str):
        def tick(message: str) -> None:
            if cancel is not None and cancel():
                raise BuildError("cancelled")
            done[0] = min(done[0] + 1, offset + units)
            if progress is not None:
                progress(done[0] / (2 * units), f"{what}: {message}")
        return tick

    fit = _fit(segments, links, phosphate5, ticker(0, "the design"), cancel, linked)
    done[0] = units
    fit_twin = _fit(twin, links, phosphate5, ticker(units, "its mirror image"), cancel, linked)
    if progress is not None:
        progress(1.0, "junctions fitted")
    mirrored = fit_twin.cost < fit.cost or (fit_twin.cost == fit.cost and [s.chirality for s in twin]
                                            < [s.chirality for s in segments])
    chosen_fit, chosen_segments = (fit_twin, twin) if mirrored else (fit, segments)
    builds, bridges = _coordinates(chosen_segments, links, chosen_fit, hydrogens, phosphate5, linked)
    if mirrored:
        _mirror(builds, bridges)
        for k, strands in enumerate(builds):
            for st in strands:
                for r in st:
                    r.hand = segments[k].chirality
    lo, hi = _centre(builds, bridges)
    if float(lo.min()) <= -999.9995 or float(hi.max()) >= 9999.9995:  # what builder._tail writes (8.3f)
        raise BuildError(f"too long: the fitted chimera spans {float((hi - lo).max()):.0f} Å, and a PDB file holds "
                         f"{2.0 * PDB_REACH:.0f} Å of a model centred at the origin: shorten the segments")
    placed, parts = _model_graph(segments, links, builds, bridges, linked)
    sign = -1.0 if mirrored else 1.0  # a turn of the mirror image is the opposite turn here
    reps = [_report(segments, links, placed, parts, j, _bond_turns(segments, links, j, chosen_fit.xs[j], linked, sign),
                    linked) for j in range(len(links))]
    s, labels, po2, special = _assemble(segments, links, builds, bridges, scheme, terminals, ids, linked)
    out = list(parse_notes)
    out += _segment_notes(segments, builds, labels)
    if any(seg.na == "dna" for seg in segments):
        c2_op2, c3_o3_p, p_o5_c5 = builder._dna_link_geometry()
        out.append(f"B-DNA phosphates turned {builder.DNA_PHOSPHATE_TURN:g}° about O3'-O5' from the Arnott fiber "
                   f"model to clear C2'(i)-OP2(i+1): {c2_op2:.2f} Å, C3'-O3'-P {c3_o3_p:.1f}°, P-O5'-C5' "
                   f"{p_o5_c5:.1f}°")
    out += _name_notes(segments, s, scheme, chosen, special, phosphate5, bool(po2))
    read = _read53(links)
    nicks = [labels[(k, 1, 0)] for k, seg in enumerate(segments)
             if seg.partner is not None and _partner_ends(_ends(links, read, k), linked)[0] == "nick"]
    if phosphate5 and nicks:
        out.append("5'-phosphates on the free 5' ends, but not on the partner strands' 5' ends at the junctions ("
                   + ", ".join(f"{c}:{n}" for c, n, _x in nicks) + "): a phosphate there would point at the next "
                   "segment's end pair and stand in the way of its stacking, so these keep a 5'-OH")
    if scheme == "pdb":
        out += pdbtools.set_records(s, "pdb")
    elif scheme == "standard":  # every record ATOM, the PO2 too (as 'mdatk pdb clean --scheme standard' writes)
        pdbtools.set_records(s, "atom")
    if linked:
        first, last = s.segments[1][0], s.segments[1][-1]
        out.append(f"Partner strands linked: {first.chain}:{first.resseq}-{last.resseq} is one continuous chain "
                   f"{describe_partner((segments, links))} (no nick, no TER inside it); each junction's two "
                   "phosphodiesters were fitted together, the gamma / beta of both P residues turning")
    out += [_junction_note(j, link, rep, labels, po2) for j, (link, rep) in enumerate(zip(links, reps))]
    out.append(f"Junction fit: the design and its mirror image (every hand flipped) were both fitted (cost "
               f"{fit.cost:.4f} and {fit_twin.cost:.4f}); written: "
               + ("the mirror image of the twin's fit (x -> -x)" if mirrored else "the design's own fit")
               + ", so that a design and its twin give exact mirror images; the model is centred on its heavy "
               "atoms")
    for j, (link, rep) in enumerate(zip(links, reps)):
        warning = _closure_warning(j, link, rep, chosen_fit.costs[j], labels, po2)
        if warning:
            out.append(warning)
    if scheme == "amber":
        out += _amber_notes(segments, links, s, labels, po2)
    # header, LINK and CONECT records of the bridges
    s.header = _header(segments, links, reps, labels, po2)
    serial, serials = 0, {}
    for seg_res in s.segments:
        for res in seg_res:
            for atom in res.atoms:
                serial += 1
                serials[(res.chain, res.resseq, atom.name)] = (serial, atom)
        serial += 1  # TER
    conect: List[Tuple[int, List[int]]] = []
    for j, rep in enumerate(reps):
        if links[j] not in BRIDGES:
            continue
        chain, number = po2[j]
        phosphate = pdbtools.LinkAtom("P", "PO2", chain, number)
        p_serial, p_atom = serials[(chain, number, "P")]
        p_atom.serial = p_serial  # write_pdb maps these serials to the ones it writes (the same)
        o_serials = []
        for side, (key, length) in enumerate(((rep.bonds[0][0], rep.bonds[0][2]), (rep.bonds[1][1], rep.bonds[1][2]))):
            c, n, name = labels[key[:3]]
            oxygen = pdbtools.LinkAtom(key[3], name, c, n)
            pair = (oxygen, phosphate) if side == 0 else (phosphate, oxygen)  # O(a) - P, then P - O(b)
            s.header.append(pdbtools.link_line(*pair, length=round(length, 2)))
            o_serial, o_atom = serials[(c, n, key[3])]
            o_atom.serial = o_serial
            o_serials.append(o_serial)
            conect.append((o_serial, [p_serial]))
        for op in ("OP1", "OP2"):  # the HET group's own bonds (PDB 3.3: CONECT for every bond of a HET group)
            op_serial, op_atom = serials[(chain, number, op)]
            op_atom.serial = op_serial
            o_serials.append(op_serial)
            conect.append((op_serial, [p_serial]))
        conect.append((p_serial, o_serials))  # P: O(a), O(b), OP1, OP2
    s.conect = ["CONECT" + "".join(f"{i:5d}" for i in [a] + bs) for a, bs in sorted(conect)]
    heavy = sum(1 for res in s.residues() for a in res.atoms if not a.name.startswith("H"))
    total = sum(len(res.atoms) for res in s.residues())
    out.append(f"Built a chimera {describe((segments, links))}: {len(segments)} segments, {len(layout)} chains"
               + (f" and {len(po2)} PO2 segment(s)" if po2 else "") + f", {total} atoms"
               + (f" ({heavy} heavy)" if hydrogens else ", heavy atoms only")
               + "; a starting model: minimise before production")
    if notes is not None:
        notes.extend(out)
    return pdbtools.write_pdb(s, strip_conect=False)


# --- command line ('mdatk build chimera') --------------------------------------------------------------

def add_command(sub):
    """Register 'chimera' on the 'mdatk build' sub-parser collection (returns the parser)."""
    c = sub.add_parser("chimera", help="D/L chimera: ideal helices of either hand (A-form RNA, B-form DNA) joined "
                                       "through strand 1 by 3'-5', 5'-3', 3'-3' or 5'-5' links")
    c.add_argument("--design", required=True,
                   help="SEGMENT LINK SEGMENT ...: a segment HAND[-NA]:SEQUENCE[/PARTNER] (HAND D or L, NA RNA or DNA, "
                        "PARTNER auto (complement), none / - (single strand) or 5'->3'), a link 3'-5', 5'-3', 3'-3' "
                        "or 5'-5'; e.g. \"D-RNA:GGCAUC 3'-5' L-RNA:GAUGCC\" or \"D-DNA:GGCAT 3'-3' L-DNA:GGCAT\"")
    c.add_argument("--na", "--nucleic-acid", dest="na", choices=builder.NUCLEIC_ACIDS, default="rna",
                   help="nucleic acid of the segments that name none (default rna)")
    c.add_argument("--scheme", choices=builder.SCHEMES, help="residue names (default: A/C/G/U and 0A/0C/0G/0U, DNA: "
                   "DA/DC/DG/DT and 0DA/0DC/0DG/0DT, no terminal variants; the bridging phosphate PO2)")
    c.add_argument("--no-terminal-names", action="store_true", help="no 5'/3' residue name variants")
    c.add_argument("--phosphate5", action="store_true", help="keep a 5'-phosphate (P, OP1, OP2, OP3) on the free 5' "
                                                             "ends (a partner strand's 5' end at a junction keeps "
                                                             "a 5'-OH)")
    c.add_argument("--hydrogens", action="store_true", help="add all hydrogens (default: heavy atoms only)")
    c.add_argument("--chains", help="chain ids, comma-separated: the strand-1 chains in design order (split at 3'-3' "
                                    "/ 5'-5' links), then the partner strands in segment order (default A, B, C ...)")
    c.add_argument("--link-partners", action="store_true",
                   help="join the partner strands too, by a phosphodiester at every junction: one continuous partner "
                        "chain (every link 3'-5' or 5'-3', every segment paired; two chains in all)")
    c.add_argument("--output", "-o", required=True, help="output PDB")
    return c


def run_command(args, progress: Progress = None) -> Tuple[str, List[str]]:
    """Build the chimera of a parsed 'chimera' command: (PDB text, notes)."""
    notes: List[str] = []
    text = build_chimera(args.design, na=getattr(args, "na", "rna"), scheme=args.scheme or None,
                         terminals=not args.no_terminal_names, hydrogens=args.hydrogens, phosphate5=args.phosphate5,
                         chains=args.chains or None, notes=notes, progress=progress,
                         link_partners=getattr(args, "link_partners", False))
    return text, notes


def nucleic_acids(design, na: str = "rna") -> List[str]:
    """The nucleic acids of a design's segments in order of first use ('rna', 'dna')."""
    segments, _links = _design(design, na)
    return list(dict.fromkeys(seg.na for seg in segments))
