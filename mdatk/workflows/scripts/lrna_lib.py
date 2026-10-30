"""L-RNA residue library and parameters generated from the D-RNA force field in use.

An L-nucleotide is the mirror image of a D-nucleotide.  Every force-field term is unchanged by mirroring
except a dihedral (or improper) whose phase is not 0 or 180 degrees: mirroring turns phi into -phi, so
the L molecule needs the phase -gamma.  In OL3 (parm10.dat: bsc0 alpha / gamma, chiOL3) - and in
OL3 + CP - every such term contains O5' or O4' (alpha O3'-P-O5'-C5', gamma O5'-C5'-C4'-C3', chi
O4'-C1'-N-C).  So, as in Dudek & Trylska's L-RNA parameters, an L residue gives these two oxygens
their own atom types (LD for a bridging O5', LH for a 5'-OH, and LE for O4' when its D type differs
from O5''s - the CP case), and this script writes

* LRNA.lib: the nucleotide units of the D library, mirrored (z -> -z: tleap builds missing hydrogens
  from these templates), named the Dudek way inside and out (LA LC LG LU, LA5 / LA3, LAN ...), with
  O5' and O4' retyped;
* frcmod.lrna: masses and Lennard-Jones terms of the new types (those of their D twins) and every
  bond, angle, dihedral and improper term of the D parameters that involves a twinned type, repeated
  for the new types in all combinations - dihedral and improper phases mirrored (-gamma).

The result is checked the way it must hold: a D molecule and its mirror image have the same energy,
term by term (MDATK's tests build both with tleap and compare sander single points).  Standard
library only (it runs where only AmberTools' python3 is available).
"""

from __future__ import annotations

import itertools
import os
import re
import sys
from typing import Dict, List, Tuple

NUCLEOTIDES = re.compile(r"^[ACGU][35N]?$")
WIDTH = {"BOND": 5, "ANGL": 8, "DIHE": 11, "IMPR": 11}
TWIN_ATOMS = ("O5'", "O4'")


# ---------------------------------------------------------------------------------- parameters
def _blocks(text: str) -> List[List[str]]:
    out: List[List[str]] = [[]]
    for line in text.splitlines():
        if line.strip():
            out[-1].append(line)
        elif out[-1]:
            out.append([])
    return [b for b in out if b]


def _types(line: str, section: str) -> Tuple[str, ...]:
    return tuple(t.strip() for t in line[:WIDTH[section]].split("-"))


def _grouped(lines: List[str], section: str) -> List[Tuple[Tuple[str, ...], List[str]]]:
    """Consecutive lines with the same key belong together (multi-term dihedrals)."""
    groups: List[Tuple[Tuple[str, ...], List[str]]] = []
    for line in lines:
        key = _types(line, section)
        if groups and groups[-1][0] == key and section == "DIHE":
            groups[-1][1].append(line)
        else:
            groups.append((key, [line]))
    return groups


def read_dat(text: str) -> Dict[str, object]:
    """MASS / BOND / ANGL / DIHE / IMPR / NONB of an Amber parm*.dat file."""
    blocks = _blocks("\n".join(text.splitlines()[1:]))
    nonb: Dict[str, Tuple[float, float]] = {}
    equiv: Dict[str, str] = {}
    lines = text.splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.startswith("MOD4")), None)
    if start is not None:
        for ln in lines[start + 1:]:
            parts = ln.split()
            if not parts or parts[0] == "END":
                break
            try:
                nonb[parts[0]] = (float(parts[1]), float(parts[2]))
            except (IndexError, ValueError):
                continue
        for ln in lines[:start]:  # equivalence lines just before MOD4: "N   NA  N2 ..." share N's terms
            parts = ln.split()
            if len(parts) > 1 and all(len(p) <= 2 for p in parts) and parts[0] in nonb:
                for p in parts[1:]:
                    equiv.setdefault(p, parts[0])
    return {"MASS": {ln.split()[0]: ln[2:].strip() for ln in blocks[0]},
            "BOND": _grouped(blocks[1][1:], "BOND"), "ANGL": _grouped(blocks[2], "ANGL"),
            "DIHE": _grouped(blocks[3], "DIHE"), "IMPR": _grouped(blocks[4], "IMPR"),
            "NONB": nonb, "EQUIV": equiv}


def read_frcmod(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {"MASS": {}, "BOND": [], "ANGL": [], "DIHE": [], "IMPR": [], "NONB": {}, "EQUIV": {}}
    section, lines = None, text.splitlines()[1:]
    collected: Dict[str, List[str]] = {k: [] for k in WIDTH}
    for ln in lines:
        head = ln.strip()[:4].upper()
        if head in ("MASS", "BOND", "ANGL", "DIHE", "IMPR", "NONB") and len(ln.split()) <= 2:
            section = head
            continue
        if not ln.strip():
            section = None if section != "NONB" else section
            continue
        if section == "MASS":
            out["MASS"][ln.split()[0]] = ln[2:].strip()
        elif section in WIDTH:
            collected[section].append(ln)
        elif section == "NONB":
            parts = ln.split()
            try:
                out["NONB"][parts[0]] = (float(parts[1]), float(parts[2]))
            except (IndexError, ValueError):
                pass
    for sec in WIDTH:
        out[sec] = _grouped(collected[sec], sec)
    return out


def merge(params: List[Dict[str, object]]) -> Dict[str, object]:
    """Parameter sets in load order: a later term for the same atom types replaces the earlier one."""
    out: Dict[str, object] = {"MASS": {}, "NONB": {}, "EQUIV": {}}
    for sec in WIDTH:
        out[sec] = {}
    for p in params:
        out["MASS"].update(p["MASS"])
        out["NONB"].update(p["NONB"])
        out["EQUIV"].update(p["EQUIV"])
        for sec in WIDTH:
            for key, lines in p[sec]:
                rev = tuple(reversed(key))
                out[sec].pop(rev, None)
                out[sec][key] = lines
    return out


def _mirror(line: str, section: str) -> str:
    """The same term with the phase -gamma (dihedral: IDIVF PK PHASE PN; improper: PK PHASE PN)."""
    width = WIDTH[section]
    fields = line[width:].split()
    k = 2 if section == "DIHE" else 1
    phase = float(fields[k])
    fields[k] = f"{(-phase) % 360.0:.5f}".rstrip("0").rstrip(".") if (-phase) % 360.0 else "0.0"
    return line[:width] + "  " + "  ".join(fields)


def twin_terms(params: Dict[str, object], twins: Dict[str, str]) -> Dict[str, List[str]]:
    """Every term with a twinned type repeated for the twins in all combinations; phases mirrored."""
    out: Dict[str, List[str]] = {}
    for sec in WIDTH:
        lines, seen = [], set()
        for key, glines in params[sec].items():
            idx = [i for i, t in enumerate(key) if t in twins]
            for n in range(1, len(idx) + 1):
                for chosen in itertools.combinations(idx, n):
                    var = tuple(twins[t] if i in chosen else t for i, t in enumerate(key))
                    if var in seen or tuple(reversed(var)) in seen:
                        continue
                    seen.add(var)
                    head = "-".join(f"{t:<2}" for t in var)
                    for gl in glines:
                        body = head + gl[WIDTH[sec]:]
                        lines.append(_mirror(body, sec) if sec in ("DIHE", "IMPR") else body)
        out[sec] = lines
    return out


def frcmod_text(params: Dict[str, object], twins: Dict[str, str], source: str) -> str:
    terms = twin_terms(params, twins)
    lines = [f"L-RNA atom types {' '.join(f'{v} ({k})' for k, v in twins.items())}: the terms of their D "
             f"twins, dihedral phases mirrored - generated by MDATK from {source}", "MASS"]
    for base, twin in twins.items():
        if base not in params["MASS"]:
            raise ValueError(f"no mass for atom type {base}")
        lines.append(f"{twin:<2} {params['MASS'][base]}")
    lines.append("")
    for sec in WIDTH:
        lines += [sec] + terms[sec] + [""]
    lines.append("NONB")
    for base, twin in twins.items():
        ref = base if base in params["NONB"] else params["EQUIV"].get(base, base)  # own entry first
        if ref not in params["NONB"]:
            raise ValueError(f"no Lennard-Jones terms for atom type {base}")
        r, e = params["NONB"][ref]
        lines.append(f"  {twin:<2}          {r:.4f}  {e:.4f}")
    return "\n".join(lines + ["", ""])


# ------------------------------------------------------------------------------------ library
def _entries(text: str) -> List[Tuple[str, List[str]]]:
    """(header, lines) of every '!' block of an OFF file."""
    out: List[Tuple[str, List[str]]] = []
    for line in text.splitlines():
        if line.startswith("!"):
            out.append((line, []))
        elif out:
            out[-1][1].append(line)
    return out


def library_types(text: str) -> Dict[str, str]:
    """D type of O5' (bridging and 5'-OH) and O4' in the library's nucleotide units: {atom name/kind: type}."""
    found: Dict[str, str] = {}
    for head, lines in _entries(text):
        m = re.match(r"!entry\.([^.]+)\.unit\.atoms table", head)
        if not m or not NUCLEOTIDES.match(m.group(1)):
            continue
        for ln in lines:
            parts = ln.split('"')
            if len(parts) >= 5 and parts[1] in TWIN_ATOMS:
                kind = parts[1] + ("5" if m.group(1).endswith(("5", "N")) and parts[1] == "O5'" else "")
                found.setdefault(kind, parts[3])
    return found


def twin_map(types: Dict[str, str]) -> Dict[str, str]:
    """D type -> L type: LD for the bridging O5', LH for a 5'-OH, LE for O4' when its type differs."""
    if "O5'" not in types or "O4'" not in types:
        raise ValueError("the library's nucleotides have no O5' / O4'")
    twins = {types["O5'"]: "LD"}
    if types.get("O5'5") and types["O5'5"] not in twins:
        twins[types["O5'5"]] = "LH"
    if types["O4'"] not in twins:
        twins[types["O4'"]] = "LE"
    return twins


def l_library(text: str, twins: Dict[str, str]) -> Tuple[str, List[str]]:
    """The nucleotide units mirrored, renamed L... and with O5' / O4' retyped; returns (text, units)."""
    units = [m for m in re.findall(r"^!entry\.([^.]+)\.unit\.name", text, re.MULTILINE) if NUCLEOTIDES.match(m)]
    out = ["!!index array str"] + [f' "L{u}"' for u in units]
    for head, lines in _entries(text):
        m = re.match(r"!entry\.([^.]+)\.(\S+)(.*)", head)
        if not m or m.group(1) not in units:
            continue
        unit, field = m.group(1), m.group(2)
        out.append(f"!entry.L{unit}.{field}{m.group(3)}")
        for ln in lines:
            parts = ln.split('"')
            if field in ("unit.atoms", "unit.atomspertinfo") and len(parts) >= 5:
                if parts[1] in TWIN_ATOMS and parts[3] in twins:
                    parts[3] = twins[parts[3]]
                ln = '"'.join(parts)
            elif field in ("unit.name", "unit.residues") and len(parts) >= 3 and parts[1] == unit:
                parts[1] = "L" + unit
                ln = '"'.join(parts)
            elif field == "unit.positions":
                x, y, z = (float(v) for v in ln.split())
                ln = f" {x:.6g} {y:.6g} {-z:.6g}"
            out.append(ln)
    return "\n".join(out) + "\n", ["L" + u for u in units]


def main(argv: List[str]) -> int:
    """argv: D library, then parameter files in load order (parm*.dat first, then frcmods), then the
    output directory; writes LRNA.lib and frcmod.lrna there."""
    if len(argv) < 3 or argv[0] in ("-h", "--help"):
        print("usage: python3 THIS_SCRIPT D_LIBRARY PARM.dat [FRCMOD ...] OUTPUT_DIR", file=sys.stderr)
        return 2
    lib_path, parm_paths, out_dir = argv[0], argv[1:-1], argv[-1]
    try:
        with open(lib_path, encoding="utf-8", errors="replace") as fh:
            lib = fh.read()
        sets = []
        for k, path in enumerate(parm_paths):
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            sets.append(read_dat(text) if k == 0 else read_frcmod(text))
        params = merge(sets)
        twins = twin_map(library_types(lib))
        for twin in twins.values():
            if twin in params["MASS"]:
                raise ValueError(f"the parameters already define atom type {twin}")
        lib_text, units = l_library(lib, twins)
        frcmod = frcmod_text(params, twins, ", ".join(os.path.basename(p) for p in [lib_path] + parm_paths))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "LRNA.lib"), "w", encoding="ascii") as fh:
            fh.write(lib_text)
        with open(os.path.join(out_dir, "frcmod.lrna"), "w", encoding="ascii") as fh:
            fh.write(frcmod)
    except (OSError, ValueError, IndexError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{out_dir}: LRNA.lib ({len(units)} units: {' '.join(units)}) and frcmod.lrna "
          f"({', '.join(f'{v} as {k}' for k, v in twins.items())}) written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
