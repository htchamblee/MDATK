"""Released behaviour stays put: every file the planner writes, and every planner message, matches
what MDATK 0.0.7 produced for the same cases (see tests/golden.py for the cases and for the two
things normalised before comparing: the "written by MDATK x.y.z" version and dates).

New step types are not part of the comparison; a step type that existed in 0.0.7 must not change
its output unless that change is intended and the reference is re-recorded from the release that
introduces it."""

import json
import os
import re

import pytest

import golden

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "data", "golden_v0.0.7.json")
# intended changes of 0.0.7 output, each with its reason; everything else must match 0.0.7 byte for byte
INTENDED = [
    # 0.0.8: the Amber build's topology check also checks that residues named as nucleotides have a
    # nucleotide's atoms (an L-RNA library without the names MDATK writes: LA / LU became La / Lu ions)
    (re.compile(r"(^|/|:)[^/]*_check_\d{8}-\d{4}\.py$"), None),
    # ... and runs for dry L-RNA builds too (they had no topology check)
    (re.compile(r"_amber-build_\d{8}-\d{4}\.sh$|_00_plan_\d{8}-\d{4}\.md$"), {"tool:amber-build:water=none"}),
    # 0.0.10: the main script of a step whose production runs in a separate script stops at once when that
    # script has finished and otherwise ends with exit status 75 ("waiting"), never with success (an
    # all-production step used to "succeed" having run nothing); only lines are added.  Every case with a
    # separate production script, i.e. all but separate_production=False
    (re.compile(r"_amber-md_\d{8}-\d{4}\.sh$"),
     {"example:rna_plan.mdatk.toml", "template:amber", "template:builder-amber", "template:farfar2-amber"}
     | {f"tool:amber-md:{v}" for v in ("default", "amber_version=24", "dry_pdb=False", "dry_traj=True",
                                       "engine=pmemd", "engine=pmemd.MPI", "input_velocities=True",
                                       "min_engine=pmemd", "restraint_ref=initial", "resume=False")}),
    (re.compile(r"_gromacs_\d{8}-\d{4}\.sh$"),
     {"template:amber-gromacs"}
     | {f"tool:gromacs:{v}" for v in ("default", "allow_nbfix_loss=True", "convert=acpype", "convert=none",
                                      "gmx_version=2023", "gpu=False", "mass_repartition=True",
                                      "update_gpu=True")}),
]


def intended(case, key):
    return any(rx.search(key) and (cases is None or case in cases) for rx, cases in INTENDED)


@pytest.fixture(scope="module")
def reference():
    with open(REFERENCE, encoding="utf-8") as fh:
        return json.load(fh)


def test_released_output_unchanged(reference):
    from mdatk.workflows import TOOLS
    assert reference["mdatk_version"] == "0.0.7"
    assert set(reference["tools"]) <= set(TOOLS), "a step type of 0.0.7 was removed"
    now = golden.collect(os.path.join(os.path.dirname(HERE), "examples"), reference["tools"])
    ref = reference["cases"]
    assert set(ref) <= set(now), f"cases no longer generated: {sorted(set(ref) - set(now))[:10]}"
    changed = {}
    for case, files in ref.items():
        diff = sorted(k for k in set(files) | set(now[case])
                      if files.get(k) != now[case].get(k) and not intended(case, k))
        if diff:
            changed[case] = diff
    assert not changed, (f"{len(changed)} case(s) differ from 0.0.7, e.g. "
                         + "; ".join(f"{c}: {', '.join(d[:4])}" for c, d in list(changed.items())[:6]))
