"""x3dna-dssr (3DNA-DSSR) as a base-pair engine: runs the program on a renamed copy and reads its JSON.

DSSR is an external program with its own licence (free for academic use); MDATK never bundles or
downloads it and runs it only when asked (``engine="dssr"``, ``--engine dssr``), as ``x3dna-dssr`` on
PATH or the executable given.

Two properties of DSSR make its raw output hard to use on MDATK's structures, and this module avoids
both instead of parsing around them:

* **Residue names.**  DSSR recognises nucleotides by name.  L-RNA names (LA, 0A, LG0, or a bare "0" in
  older Rosetta output) and modified names are not standard names, so their pairs could be missed.  The
  caller gives every residue a canonical name (A C G U, DA DC DG DT; :mod:`mdatk.interfaces` infers the
  base from the atoms when the name is unknown); everything else (ions, water, Rosetta's virtual
  residues) is left out of the copy rather than renamed.
* **Nucleotide ids.**  DSSR names residues ``chain.nameNUMBER`` - ``A.G1``, ``1:A.G1`` with a model
  number, ``A.0A/45`` where the name ends in a digit, ``A.G1^B`` with an insertion code - and names
  atoms ``atom@nucleotide`` (``N1@A.G1``) in its hydrogen-bond list.  Chains of several letters, blank
  chains, numbering restarted per chain and changes between DSSR versions all break a reader that
  splits these ids by hand.  Here residue ``k`` of the copy is written as residue ``k % 9999 + 1`` of
  chain ``CHAIN_IDS[k // 9999]`` with its canonical name (as :class:`mdatk.secstruct.BarnabaRunner`
  does), so an id maps back to ``k`` through the JSON's ``nts`` table (``chain_name``, ``nt_resnum``)
  or, without one, through a tolerant reading of the id that only has to recover a one-letter chain
  and a number MDATK wrote itself.

JSON keys read: ``nts`` (``nt_id``, ``chain_name``, ``nt_resnum``), ``pairs`` (``nt1``, ``nt2``,
``LW``, ``name``), ``stacks`` (``nts_long``) and ``hbonds`` (``atom1_id``, ``atom2_id``,
``distance``), as written by DSSR 1.x with ``--json``.  The same keys are expected from DSSR 2.x; that
could not be checked against a DSSR 2.x file here, so the reader also accepts ``hbonds_all`` /
``hBonds``, atom ids in either order (``N1@A.G1`` or ``A.G1@N1``) and ids without an ``nts`` table,
and it reports a hydrogen-bond list it could not read instead of counting it as empty.

DSSR's classification does not depend on sugar chirality: it annotates a D-L pair like a D-D pair.
Use it for what the contacts are, not as evidence that a heterochiral pair is plausible.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pdbtools import CHAIN_IDS

DEFAULT_EXE = "x3dna-dssr"
BLOCK = 9999  # residues per synthetic chain (residue numbers 1-9999)
TIMEOUT = 600  # seconds for one DSSR run

# 'model:chain.name[/]number[^insertion]' - the name is matched lazily, so 'A.5MC40' is name 5MC,
# residue 40; the chain may be empty or several characters (DSSR's own ids for other files)
_NT_ID = re.compile(r"^(?:(?P<model>\d+):)?(?P<chain>[^.]*)\.(?P<name>.*?)/?(?P<num>-?\d+)"
                    r"(?:\^(?P<icode>.))?$")


class DssrError(RuntimeError):
    """x3dna-dssr is missing, failed, or wrote no readable JSON."""


@dataclass
class DssrResult:
    """DSSR's annotation in terms of the residue indices given to :class:`DssrRunner`."""
    pairs: List[Tuple[int, int, str, str]] = field(default_factory=list)  # (i, j, LW e.g. 'cWW', name)
    stacks: List[Tuple[int, int]] = field(default_factory=list)  # neighbours within DSSR's stacks
    hbonds: List[Tuple[int, str, int, str, Optional[float]]] = field(default_factory=list)
    hbonds_seen: int = 0  # entries of the hydrogen-bond list in the JSON
    hbonds_read: int = 0  # ... whose two atom ids could be mapped back
    has_hbonds: bool = False  # the JSON has a hydrogen-bond list at all
    version: str = ""
    notes: List[str] = field(default_factory=list)


def available(exe: str = DEFAULT_EXE) -> bool:
    return bool(shutil.which(exe) or (os.path.isfile(exe) and os.access(exe, os.X_OK)))


def lw_code(lw: str, name: str = "") -> str:
    """MDATK's REMARK 950 class of a DSSR pair: 'WCc' (Watson-Crick), 'GUc' (wobble), otherwise the
    Leontis-Westhof edges and cis/trans in barnaba's order ('cHS' -> 'HSc'); '' when DSSR gave none."""
    lw = (lw or "").strip()
    name = (name or "").strip()
    if lw == "cWW" and name == "WC":
        return "WCc"
    if lw == "cWW" and name == "Wobble":
        return "GUc"
    m = re.fullmatch(r"([ct])([WHS.])([WHS.])", lw)
    return f"{m.group(2)}{m.group(3)}{m.group(1)}" if m else lw


def parse_nt_id(text: str) -> Optional[Tuple[str, int, str]]:
    """(chain, number, insertion code) of a DSSR nucleotide id, or None."""
    m = _NT_ID.match((text or "").strip())
    if not m:
        return None
    return m.group("chain"), int(m.group("num")), m.group("icode") or ""


def _slot(chain: str, number: int) -> Optional[int]:
    """Residue index of a synthetic (chain, number) written by :class:`DssrRunner`."""
    if len(chain) != 1 or chain not in CHAIN_IDS or not 1 <= number <= BLOCK:
        return None
    return CHAIN_IDS.index(chain) * BLOCK + number - 1


def _element(name: str) -> str:
    letters = name.lstrip("0123456789")
    return letters[:1].upper() if letters else "X"


def _name_field(name: str) -> str:
    return name[:4].ljust(4) if len(name) >= 4 else " " + name.ljust(3)


class DssrRunner:
    """Runs x3dna-dssr on chosen residues.

    ``resnames``: the canonical name to write per residue ("" leaves the residue out); ``atoms``: per
    residue the (atom name, (x, y, z)) pairs to write (heavy atoms suffice: DSSR finds hydrogen bonds from
    donor-acceptor distances).  Results refer to the indices of these lists.
    """

    def __init__(self, resnames: Sequence[str], atoms: Sequence[Sequence[Tuple[str, Sequence[float]]]],
                 exe: str = DEFAULT_EXE, timeout: float = TIMEOUT):
        self.exe = (exe or DEFAULT_EXE).strip()
        self.timeout = timeout
        self.residues = [(k, name, list(a)) for k, (name, a) in enumerate(zip(resnames, atoms)) if name and a]
        if len(self.residues) > BLOCK * len(CHAIN_IDS):
            raise ValueError("too many nucleotides for the DSSR engine")
        # slot in the copy -> residue index of the caller
        self.original = {slot: k for slot, (k, _n, _a) in enumerate(self.residues)}

    def pdb_text(self) -> str:
        """The renamed copy: one model, canonical names, synthetic chains and numbers, TER per chain."""
        lines, serial, chain = [], 0, None
        for slot, (_k, name, atoms) in enumerate(self.residues):
            cid, resseq = CHAIN_IDS[slot // BLOCK], slot % BLOCK + 1
            if chain is not None and cid != chain:
                lines.append("TER")
            chain = cid
            for atom, xyz in atoms:
                serial += 1
                x, y, z = (float(v) for v in xyz)
                lines.append(f"ATOM  {serial % 100000:>5} {_name_field(atom)} {name:>3} {cid}{resseq:>4}"
                             f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {_element(atom):>2}")
        lines += ["TER", "END"]
        return "\n".join(lines) + "\n"

    def run(self, keep: str = "", stem: str = "dssr") -> DssrResult:
        """Run DSSR in a temporary folder (its auxiliary dssr-* files stay there) and read its JSON.
        ``keep``: a folder that receives the renamed copy and the JSON as ``<stem>_dssr.pdb`` /
        ``<stem>.json``."""
        if len(self.residues) < 2:
            return DssrResult(notes=["fewer than two nucleotides: DSSR not run"])
        with tempfile.TemporaryDirectory(prefix="mdatk_dssr_") as tmp:
            with open(os.path.join(tmp, "in.pdb"), "w", encoding="utf-8") as fh:
                fh.write(self.pdb_text())
            cmd = [self.exe, "-i=in.pdb", "-o=out.json", "--json"]
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=self.timeout)
            except FileNotFoundError:
                raise DssrError(f"{self.exe} not found: the DSSR engine needs x3dna-dssr (its own licence, "
                                "free for academic use), on PATH or given as the executable") from None
            except subprocess.TimeoutExpired:
                raise DssrError(f"{self.exe} did not finish within {self.timeout:g} s") from None
            out = os.path.join(tmp, "out.json")
            if keep:
                os.makedirs(keep, exist_ok=True)
                shutil.copyfile(os.path.join(tmp, "in.pdb"), os.path.join(keep, f"{stem}_dssr.pdb"))
                if os.path.isfile(out):
                    shutil.copyfile(out, os.path.join(keep, f"{stem}.json"))
            if proc.returncode != 0 or not os.path.isfile(out):
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
                raise DssrError(f"{self.exe} failed (exit status {proc.returncode})"
                                + (": " + " / ".join(tail) if tail else ""))
            try:
                with open(out, encoding="utf-8", errors="replace") as fh:
                    doc = json.load(fh)
            except ValueError as exc:
                raise DssrError(f"unreadable DSSR JSON: {exc}") from None
        return self.read(doc)

    def read(self, doc: Dict[str, Any]) -> DssrResult:
        """Map a DSSR JSON document back to the caller's residue indices."""
        return parse_json(doc, self.original)


def parse_json(doc: Dict[str, Any], original: Optional[Dict[int, int]] = None) -> DssrResult:
    """Pairs, stacks and hydrogen bonds of a DSSR JSON document.  ``original`` maps the synthetic slot of
    a residue (see :class:`DssrRunner`) to the caller's index (identity when None)."""
    res = DssrResult()
    if not isinstance(doc, dict):
        res.notes.append("the DSSR JSON is not an object")
        return res
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    res.version = str(doc.get("version") or meta.get("version") or meta.get("dssr") or "")
    by_id: Dict[str, int] = {}
    for nt in doc.get("nts") or []:
        if not isinstance(nt, dict):
            continue
        try:
            slot = _slot(str(nt.get("chain_name", "")), int(nt.get("nt_resnum")))
        except (TypeError, ValueError):
            slot = None
        if slot is not None and nt.get("nt_id"):
            by_id[str(nt["nt_id"])] = slot

    def index(text: Any) -> Optional[int]:
        text = str(text or "").strip()
        slot = by_id.get(text)
        if slot is None:
            parsed = parse_nt_id(text)
            slot = _slot(parsed[0], parsed[1]) if parsed else None
        if slot is None:
            return None
        return slot if original is None else original.get(slot)

    unread = 0
    for p in doc.get("pairs") or []:
        if not isinstance(p, dict):
            continue
        i, j = index(p.get("nt1")), index(p.get("nt2"))
        if i is None or j is None or i == j:
            unread += 1
            continue
        res.pairs.append((min(i, j), max(i, j), str(p.get("LW") or p.get("lw") or ""),
                          str(p.get("name") or "")))
    for s in doc.get("stacks") or []:
        if not isinstance(s, dict):
            continue
        members = s.get("nts_long") or s.get("nts") or ""
        members = members.split(",") if isinstance(members, str) else list(members)
        rows = [index(m) for m in members if str(m).strip()]
        for a, b in zip(rows, rows[1:]):
            if a is None or b is None or a == b:
                unread += 1
                continue
            res.stacks.append((min(a, b), max(a, b)))
    if unread:
        res.notes.append(f"{unread} DSSR pair / stacking entries named residues that could not be mapped "
                         "back")

    hbonds = None
    for key in ("hbonds", "hbonds_all", "hBonds"):
        if isinstance(doc.get(key), list):
            hbonds = doc[key]
            break
    res.has_hbonds = hbonds is not None
    for h in hbonds or []:
        if not isinstance(h, dict):
            continue
        res.hbonds_seen += 1
        a = _atom(h.get("atom1_id"), index)
        b = _atom(h.get("atom2_id"), index)
        if a is None or b is None:
            continue
        res.hbonds_read += 1
        try:
            dist = float(h.get("distance"))
        except (TypeError, ValueError):
            dist = None
        res.hbonds.append((a[1], a[0], b[1], b[0], dist))
    if hbonds is None:
        res.notes.append("the DSSR JSON has no hydrogen-bond list (looked for hbonds, hbonds_all, hBonds): "
                         "hydrogen-bond contacts were not assessed")
    elif res.hbonds_seen and not res.hbonds_read:
        res.notes.append(f"none of the {res.hbonds_seen} hydrogen bonds in the DSSR JSON had atom ids that "
                         "could be read (atom1_id / atom2_id as atom@nucleotide): hydrogen-bond contacts "
                         "were not assessed")
    res.pairs.sort()
    res.stacks = sorted(set(res.stacks))
    return res


def _atom(text: Any, index) -> Optional[Tuple[str, int]]:
    """(atom name, residue index) of a DSSR atom id 'N1@A.G1' (or 'A.G1@N1'); None for atoms of residues
    that are not in the copy (ions, water) or ids that cannot be read."""
    text = str(text or "").strip()
    if "@" not in text:
        return None
    left, _, right = text.partition("@")
    k = index(right)
    if k is not None and left:
        return left.strip(), k
    k = index(left)
    if k is not None and right:
        return right.strip(), k
    return None
