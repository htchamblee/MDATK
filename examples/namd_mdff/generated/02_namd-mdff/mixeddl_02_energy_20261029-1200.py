#!/usr/bin/env python3
"""sander / NAMD single-point energy comparison - written by MDATK 0.0.10.

Usage: python3 THIS_SCRIPT SANDER.mdout NAMD.log
"""

import re
import sys

PAIR_RE = re.compile(r"(\d-\d [A-Z]+|[A-Z][A-Z0-9]*)\s*=\s*([-+]?\d+\.\d+(?:[eE][-+]?\d+)?|\*+)")
ALIASES = {"1-4 NB": "1-4 VDW"}


def sander_terms(path):
    """Energy terms of the first step (the input coordinates) of a sander minimisation output."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.split()[:2] == ["NSTEP", "ENERGY"]), None)
    if start is None:
        raise ValueError(f"{path}: no energy block (did sander run?)")
    terms = {}
    for line in lines[start + 2:start + 12]:
        if not line.strip() and "BOND" in terms:
            break
        for name, value in PAIR_RE.findall(line):
            name = ALIASES.get(name, name)
            if value.startswith("*"):
                raise ValueError(f"{path}: sander printed ***** for {name} (overflow: bad contacts?)")
            terms[name] = float(value)
    missing = [k for k in ("BOND", "ANGLE", "DIHED", "VDWAALS", "EEL", "1-4 VDW", "1-4 EEL") if k not in terms]
    if missing:
        raise ValueError(f"{path}: energy terms {', '.join(missing)} not found")
    return terms


def namd_terms(path):
    """Energy terms of the first ENERGY: line of a NAMD log (step 0 of 'run 0')."""
    title = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("ETITLE:") and title is None:
                title = line.split()[1:]
            elif line.startswith("ENERGY:") and title is not None:
                values = line.split()[1:]
                return {k: float(v) for k, v in zip(title, values) if k != "TS"}
    raise ValueError(f"{path}: no ETITLE / ENERGY lines (did NAMD run?)")


def compare(sander, namd, tolerances):
    """[(term, sander value, NAMD value, difference, allowed, ok)]."""
    pairs = [("BOND", sander["BOND"], namd["BOND"]),
             ("ANGLE", sander["ANGLE"], namd["ANGLE"]),
             ("DIHED", sander["DIHED"], namd["DIHED"] + namd.get("IMPRP", 0.0)),
             ("VDW", sander["VDWAALS"] + sander["1-4 VDW"], namd["VDW"]),
             ("ELECT", sander["EEL"] + sander["1-4 EEL"], namd["ELECT"])]
    rows = []
    for term, a, b in pairs:
        absolute, relative = tolerances[term]
        allowed = max(absolute, relative * abs(a))
        rows.append((term, a, b, b - a, allowed, abs(b - a) <= allowed))
    return rows


def main(argv, tolerances):
    if len(argv) != 2:
        print("usage: python3 THIS_SCRIPT SANDER.mdout NAMD.log", file=sys.stderr)
        return 2
    try:
        rows = compare(sander_terms(argv[0]), namd_terms(argv[1]), tolerances)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{'term':8s}{'sander':>16s}{'NAMD':>16s}{'difference':>14s}{'allowed':>12s}")
    for term, a, b, d, allowed, ok in rows:
        print(f"{term:8s}{a:16.4f}{b:16.4f}{d:14.4f}{allowed:12.4f}  {'ok' if ok else 'MISMATCH'}")
    bad = [r[0] for r in rows if not r[5]]
    if bad:
        print(f"ERROR: NAMD and sander disagree on {', '.join(bad)}: NAMD does not reproduce the Amber force "
              "field of this topology; the MDFF run is not started", file=sys.stderr)
        return 1
    print("NAMD reproduces the sander energies of this topology")
    return 0


TOLERANCES = {"BOND": (0.05, 1e-4), "ANGLE": (0.05, 1e-4), "DIHED": (0.05, 1e-4), "VDW": (0.5, 1e-3),
              "ELECT": (2.0, 2e-3)}


# --- settings written by MDATK ---
TOLERANCES = {'BOND': (0.05, 0.0001), 'ANGLE': (0.05, 0.0001), 'DIHED': (0.05, 0.0001), 'VDW': (0.5, 0.001), 'ELECT': (2.0, 0.002)}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], TOLERANCES))
