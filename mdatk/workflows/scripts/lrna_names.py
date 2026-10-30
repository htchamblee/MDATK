"""L residue names against the residue libraries tleap loads: run before tleap, so a mismatch costs seconds.

MDATK names L nucleotides the Dudek & Trylska way (LA LC LG LU, LA5 / LA3 ...).  tleap resolves a PDB
residue name to any unit of that name, and AmberTools' atomic_ions.lib (loaded with every water model)
defines LA (lanthanum) and LU (lutetium).  With an L-RNA library whose units have other names (A0 / A05 /
A03 ...) and no addPdbResMap onto them, tleap builds every L-adenosine and L-uridine as a one-atom ion
without a warning and invents the atoms it cannot place near the origin; the solvent box then fills the
gap between them and the solute (13 times the volume and 57 minutes of tleap in the case that prompted
this check).  The Amber build runs this script right after the L residues are named and before the first
tleap, on the named PDB and the tleap input, and stops when an L name is neither a unit loaded before
loadpdb (loadOff libraries, loadAmberPrep files, units assigned with NAME = ...) nor mapped onto one with
addPdbResMap.  A library given as a name in leap's search path is read from $AMBERHOME/dat/leap (AMBERHOME
from the environment, else from tleap on PATH); one that cannot be read is reported and the check
continues.  Files run with ``source`` are not followed: AmberTools' leaprc files load atomic_ions.lib,
whose LA / LU are exactly what must not count (a command file of one's own makes the result a warning).

The planner (mdatk.workflows.amber) uses the same parsing for its checks.  Standard library only (it runs
where only AmberTools' python3 is available).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Dict, List, Sequence, Set, Tuple

L_NAME = re.compile(r"L[ACGU][35N]?")  # the names MDATK writes for L nucleotides
IONS = {"LA": "lanthanum", "LU": "lutetium"}  # units of AmberTools' atomic_ions.lib with L-nucleotide names
LEAP_DIRS = ("prep", "lib", "parm", "cmd")  # tleap's default search path under $AMBERHOME/dat/leap
_LOAD = re.compile(r'^\s*(loadoff|loadamberprep|source)\s+("[^"]*"|\S+)', re.IGNORECASE | re.MULTILINE)
_ASSIGN = re.compile(r'^\s*([^\s=#{}"]+)\s*=', re.MULTILINE)
_MAP_ENTRY = re.compile(r'\{\s*(?:\d+\s+)?"?([^"\s{}]+)"?\s+"?([^"\s{}]+)"?\s*\}')


# ---------------------------------------------------------------------------------- parsing
def _code(text: str) -> str:
    """Leap commands without comments ('#' to the end of the line)."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def before_pdb(text: str, pdb: str = "") -> str:
    """The commands that take effect before the PDB is read: everything above the loadpdb line that reads
    ``pdb`` (the first loadpdb line when no line names it)."""
    code = _code(text)
    found = list(re.finditer(r"^.*\bloadpdb\b.*$", code, re.IGNORECASE | re.MULTILINE))
    named = [m for m in found if pdb and os.path.basename(pdb) in m.group(0)]
    m = (named or found or [None])[0]
    return code[:m.start()] if m else code


def leap_sources(text: str, pdb: str = "") -> Tuple[List[str], List[str], List[str], Set[str]]:
    """What leap commands load before the PDB is read: OFF libraries (loadOff), prep files (loadAmberPrep),
    command files (source), and the names that assignments (NAME = ...) give to units."""
    code = before_pdb(text, pdb)
    libs: List[str] = []
    preps: List[str] = []
    sourced: List[str] = []
    for cmd, arg in _LOAD.findall(code):
        {"loadoff": libs, "loadamberprep": preps, "source": sourced}[cmd.lower()].append(arg.strip('"'))
    return libs, preps, sourced, set(_ASSIGN.findall(code))


def lib_units(text: str) -> Set[str]:
    """Unit names an OFF library defines."""
    return set(re.findall(r"^!entry\.([^.\s]+)\.unit\.atoms table", text, re.MULTILINE))


def prep_units(text: str) -> Set[str]:
    """Residue names an Amber prep file defines (the 'NAME INT 0' line of each residue)."""
    return set(re.findall(r"^\s*(\S+)\s+INT\s+\d+\s*$", text, re.MULTILINE))


def pdb_res_maps(text: str) -> Dict[str, Set[str]]:
    """PDB residue name -> unit names from addPdbResMap commands ({ "LG" "G0" } and { 0 "LG" "G05" }),
    also when one spans several lines."""
    out: Dict[str, Set[str]] = {}
    code = _code(text)
    for m in re.finditer(r"\baddpdbresmap\b", code, re.IGNORECASE):
        start = code.find("{", m.end())
        if start < 0:
            continue
        depth, end = 0, len(code)
        for k in range(start, len(code)):
            depth += {"{": 1, "}": -1}.get(code[k], 0)
            if depth == 0:
                end = k
                break
        for name, unit in _MAP_ENTRY.findall(code[start + 1:end]):
            out.setdefault(name, set()).add(unit)
    return out


def l_names(path: str) -> List[str]:
    """Names of the L nucleotides in a PDB.  A residue counts when it has a sugar or phosphate atom, so a
    lanthanum ion named LA is not taken for an L-adenosine."""
    found: Set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")) and L_NAME.fullmatch(line[17:21].strip()):
                atom = line[12:16].strip()
                if "'" in atom or "*" in atom or atom == "P":
                    found.add(line[17:21].strip())
    return sorted(found)


# ---------------------------------------------------------------------------------- verdict
def unmapped(names: Sequence[str], units: Set[str], maps: Dict[str, Set[str]]) -> List[str]:
    """The names that are neither units nor mapped onto one."""
    return [m for m in names if m not in units and not (maps.get(m, set()) & units)]


def consequence(missing: Sequence[str]) -> str:
    """What tleap makes of L names it has no unit for."""
    silent = [m for m in missing if m in IONS]
    rest = [m for m in missing if m not in IONS]
    text = "tleap would "
    if silent:
        ions = " and ".join(IONS[m] for m in silent)
        text += (f"silently read {' and '.join(silent)} as the {ions} ion of AmberTools' atomic_ions.lib"
                 + (" and " if rest else ""))
    if rest:
        text += "invent parameterless units for " + ("the rest" if silent else ", ".join(rest))
    return text


def map_suggestion(missing: Sequence[str], units: Set[str]) -> str:
    """addPdbResMap line for the missing names: onto the units of an A0-style library (LG5 -> G05, LA -> A0)
    where they exist, else a template."""
    entries = []
    for m in missing:
        unit = f"{m[1]}0{m[2:]}"
        if unit in units:
            entries.append(f'{{ "{m}" "{unit}" }}')
    if entries:
        return f"addPdbResMap {{ {' '.join(entries)} }}"
    return 'addPdbResMap { { "LG" "<the library\'s L-G unit>" } ... }'


def unit_list(units: Set[str], limit: int = 12) -> str:
    return ", ".join(sorted(units)[:limit]) + (" ..." if len(units) > limit else "")


# ---------------------------------------------------------------------------------- script
def amberhome() -> str:
    """$AMBERHOME, else the installation of tleap on PATH (as the build script finds it)."""
    home = os.environ.get("AMBERHOME", "")
    if not home:
        tleap = shutil.which("tleap")
        home = os.path.dirname(os.path.dirname(os.path.realpath(tleap))) if tleap else ""
    return home


def resolve(name: str, home: str) -> str:
    """A file here (or an absolute path), else the file of that name in leap's search path."""
    if os.path.isfile(name) or not home:
        return name
    for sub in LEAP_DIRS:
        cand = os.path.join(home, "dat", "leap", sub, name)
        if os.path.isfile(cand):
            return cand
    return name


def check(pdb: str, leap: str) -> Tuple[List[str], List[str]]:
    """(problems, warnings) of the L residue names in ``pdb`` against the tleap input ``leap``; prints a
    report."""
    names = l_names(pdb)
    with open(leap, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    print(f"L-RNA residue names of {pdb} against the residue libraries of {leap}")
    if not names:
        print("  no L residues")
        return [], []
    print(f"  L residue names: {', '.join(names)}")
    libs, preps, sourced, units = leap_sources(text, pdb)
    home = amberhome()
    warnings: List[str] = []
    unread: List[str] = []
    for name, read in [(f, lib_units) for f in libs] + [(f, prep_units) for f in preps]:
        path = resolve(name, home)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                found = read(fh.read())
        except OSError as exc:
            unread.append(name)
            warnings.append(f"cannot read {name} ({exc.strerror or exc}): its units are not checked")
            continue
        units |= found
        print(f"  {name}: {len(found)} units")
    for name in sourced:
        if os.path.isfile(name):  # a command file of one's own (AmberTools' leaprc files are not followed)
            unread.append(name)
            warnings.append(f"{name} is run with 'source' and not followed: units it loads are not checked")
    missing = unmapped(names, units, pdb_res_maps(before_pdb(text, pdb)))
    problems: List[str] = []
    if missing and unread:
        warnings.append(f"{', '.join(missing)} are neither units of the libraries read nor mapped onto "
                        f"them with addPdbResMap; {', '.join(unread)} could not be checked and may define "
                        f"them. If they do not, {consequence(missing)}")
    elif missing:
        problems.append(f"the L residue names {', '.join(missing)} are neither units of the residue "
                        f"libraries loaded before loadpdb ({unit_list(units) or 'none'}) nor mapped onto "
                        f"them with addPdbResMap. {consequence(missing)}; tleap puts the atoms it invents "
                        "near the origin, and the solvent box grows to fill the gap. Add under 'Leap "
                        f"commands before loadpdb' (or before loadpdb in {leap}): "
                        f"{map_suggestion(missing, units)}")
    for msg in warnings:
        print(f"  WARNING: {msg}")
    for msg in problems:
        print(f"  PROBLEM: {msg}")
    if not missing:
        print("  every L residue name is a unit or mapped onto one")
    print("  L-RNA name check " + ("FAILED" if problems else "passed"))
    return problems, warnings


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 THIS_SCRIPT NAMED.pdb TLEAP_INPUT.leap", file=sys.stderr)
        return 2
    try:
        problems, _warnings = check(argv[0], argv[1])
    except OSError as exc:
        print(f"L-RNA name check: {exc}", file=sys.stderr)
        return 1
    return 1 if problems else 0
