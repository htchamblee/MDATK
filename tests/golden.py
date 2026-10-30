"""Generated workflow output, reduced to hashes, for the regression test of released behaviour.

``python3 tests/golden.py --write tests/data/golden_v0.0.7.json`` records, with whatever ``mdatk``
is importable (e.g. ``PYTHONPATH=<checkout of v0.0.7>``), every file the planner writes and every
planner message for:

* every built-in template (default settings, dummy input paths);
* every step type on its own: default settings, then each check box flipped and each radio /
  combo choice selected, one at a time (all inputs connected to dummy paths);
* the example projects in ``examples/``.

Every case is planned with residue groups, as the app plans it: the example projects with their own,
the others with those of a new project.  One VMD case without groups keeps the fallback pinned.  Steps
that take a typed secondary structure get extra cases with one (a hairpin with a G·U pair, wobble pairs
off and on; optional inputs left unconnected), so the secondary-structure code they share with later
step types is pinned too.

``test_released_output_unchanged`` (tests/test_golden.py) regenerates the same cases with the
current code and compares.  Two things are normalised before hashing, because they change with
every release or every day by design: the MDATK version in "written by MDATK x.y.z" texts and
dates (YYYY-MM-DD).  Everything else must be byte-identical.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from typing import Dict, Iterator, List, Tuple

STAMP = "20261015-1432"
VERSION_RE = re.compile(r"\b(MDATK|mdatk) \d+\.\d+\.\d+(?:\.\d+)?\b")
DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def normalise(text: str, root: str) -> str:
    text = text.replace(root, "<ROOT>")
    text = VERSION_RE.sub(lambda m: f"{m.group(1)} <VERSION>", text)
    return DATE_RE.sub("<DATE>", text)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()[:20]


def _dummy(port) -> str:
    return f"/golden/inputs/{port.name}.{port.kinds[0]}"


TYPED_SS = {"sequence": "GGGCUUCGGUCC", "dotbracket": "((((....))))"}  # G3·U10 is a wobble pair
EXTRA_CASES = {  # (label, settings) beyond the one-change-at-a-time variants
    "amber-restraints": [("typed-ss", TYPED_SS), ("typed-ss+wobble", dict(TYPED_SS, wobble=True))],
    "phenix-rsr": [("typed-ss", dict(TYPED_SS, resolution=3.5)),
                   ("typed-ss+wobble", dict(TYPED_SS, resolution=3.5, wobble=True))],
    "simrna": [("typed-ss", TYPED_SS)],
    "farfar2": [("typed-ss", TYPED_SS)],
    "structure-builder": [("mode=fit+resolution", {"mode": "fit", "resolution": 4.0})],
}


def _variants(tool) -> Iterator[Tuple[str, Dict]]:
    """(label, settings) - the defaults, then one changed value at a time."""
    yield "default", {}
    for section in tool.sections:
        if section.repeat:
            continue
        for opt in section.options:
            if opt.kind == "bool":
                yield f"{opt.key}={not opt.default}", {opt.key: not opt.default}
            elif opt.kind in ("radio", "choice"):
                for value, _label in opt.choices:
                    if value != opt.default:
                        yield f"{opt.key}={value}", {opt.key: value}


def cases(tool_names=None) -> Iterator[Tuple[str, object, list]]:
    """(case id, WorkflowParams, residue groups) for every case; ``tool_names`` limits the per-tool cases."""
    from mdatk.config import PlanNode, Project, WorkflowParams
    from mdatk.workflows import TOOLS
    from mdatk.workflows.plan import TEMPLATES, apply_template

    groups = Project().groups  # what every new project starts with
    for name in sorted(TEMPLATES):
        wf = WorkflowParams(project_name="golden", stamp=STAMP)
        apply_template(wf, name)
        for node in wf.nodes:
            tool = TOOLS[node.tool]
            for port in tool.input_ports(tool.normalise(node.settings)):
                if not port.optional and not any(e.target == node.id and e.input == port.name
                                                 for e in wf.edges):
                    node.files.setdefault(port.name, _dummy(port))
        yield f"template:{name}", wf, groups
    for name in sorted(tool_names if tool_names is not None else TOOLS):
        tool = TOOLS[name]
        if not tool.generates:
            continue
        for label, change in _variants(tool):
            settings = tool.normalise(copy.deepcopy(change))
            node = PlanNode(id="x", tool=name, step=3, settings=settings,
                            files={p.name: _dummy(p) for p in tool.input_ports(settings)})
            wf = WorkflowParams(project_name="golden", stamp=STAMP, nodes=[node])
            yield f"tool:{name}:{label}", wf, groups
            if name == "vmd" and label == "default":  # the fallback when a project has no groups
                yield f"tool:{name}:{label}:no-groups", copy.deepcopy(wf), []
        for label, change in EXTRA_CASES.get(name, []):
            settings = tool.normalise(copy.deepcopy(change))
            node = PlanNode(id="x", tool=name, step=3, settings=settings,
                            files={p.name: _dummy(p) for p in tool.input_ports(settings) if not p.optional})
            yield f"tool:{name}:{label}", WorkflowParams(project_name="golden", stamp=STAMP, nodes=[node]), groups


def run_case(wf, root: str, groups=()) -> Dict[str, str]:
    """Hashes of the planner messages and of every written file (paths relative to the plan folder)."""
    from mdatk.workflows.plan import Plan

    plan = Plan(wf, root, list(groups))
    out: Dict[str, str] = {}
    messages = [f"plan {i.level}: {i.message}" for i in plan.issues]
    for nid in plan.order:
        step = plan.steps[nid]
        messages += [f"{nid} {i.level}: {i.message}" for i in step.issues]
    out[":issues"] = _sha(normalise("\n".join(messages), root))
    report = plan.write(None)
    for path in sorted(report.written):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8", errors="surrogateescape") as fh:
            out[rel] = _sha(normalise(fh.read(), root))
    # steps the planner refuses to write (errors, e.g. dummy inputs): their generators still run
    from mdatk.workflows.plan import generate_step
    for nid in sorted(report.skipped):
        step = plan.steps[nid]
        try:
            files = generate_step(step)
        except Exception as exc:  # noqa: BLE001 - recorded, compared like any output
            out[f":forced:{nid}"] = f"exception {type(exc).__name__}"
            continue
        for f in files:
            out[f":forced:{nid}:{f.name}"] = _sha(normalise(f.content, root))
    return out


def example_cases(examples_dir: str) -> Iterator[Tuple[str, object, list]]:
    from mdatk.config import Project
    for fname in sorted(os.listdir(examples_dir)):
        if fname.endswith(".mdatk.toml"):
            project = Project.load(os.path.join(examples_dir, fname))
            project.workflow.stamp = project.workflow.stamp or STAMP
            yield f"example:{fname}", project.workflow, project.groups


def collect(examples_dir: str, tool_names=None) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for case, wf, groups in list(cases(tool_names)) + list(example_cases(examples_dir)):
        with tempfile.TemporaryDirectory(prefix="mdatk-golden-") as tmp:
            root = os.path.join(tmp, "plan")
            result[case] = run_case(wf, root, groups)
    return result


def main(argv: List[str]) -> int:
    try:
        import mdatk
    except ImportError:  # run from a source checkout: PYTHONPATH (e.g. an older release) wins
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import mdatk
    from mdatk.workflows import TOOLS
    if len(argv) != 2 or argv[0] != "--write":
        print("usage: python3 tests/golden.py --write OUT.json", file=sys.stderr)
        return 2
    here = os.path.dirname(os.path.abspath(__file__))
    data = {"mdatk_version": mdatk.__version__, "tools": sorted(TOOLS),
            "cases": collect(os.path.join(os.path.dirname(here), "examples"))}
    with open(argv[1], "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"{len(data['cases'])} cases from MDATK {mdatk.__version__} ({mdatk.__file__}) -> {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
