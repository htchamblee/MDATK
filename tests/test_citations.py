import re

from mdatk import citations as C
from mdatk.config import Project
from mdatk.workflows import TOOLS, mg_params
from mdatk.workflows.plan import Plan, add_node, generate_step


def test_registry_is_consistent_and_bibtex_parses():
    assert C.REFERENCES and C.RESOURCES
    for res in C.RESOURCES.values():
        assert res.keys and all(k in C.REFERENCES for k in res.keys), res.id
    for key, ref in C.REFERENCES.items():
        assert ref.text and ref.verified, key
        m = re.match(r"@(\w+)\{([^,]+),", ref.bibtex.strip())
        assert m and m.group(2) == key and ref.bibtex.count("{") == ref.bibtex.count("}"), key
    text = C.bibtex(C.REFERENCES)
    assert text.count("@") >= len(C.REFERENCES)


def test_every_resource_a_tool_can_report_is_known():
    ids = set(mg_params.RESOURCES.values())
    for tool in TOOLS.values():
        if tool.generates:
            ids |= set(tool.resources(tool.defaults()))
    amber = TOOLS["amber-build"]
    for water in ("tip3p", "spce", "opc", "opc3", "tip4pew", "fb3", "tip4pd", "tip4p2005"):
        for mg in ("126", "1264", "micro", "nano"):
            s = amber.normalise({"water": water, "mg_model": mg, "mg_count": 2, "hmr": True,
                                 "lrna_ranges": "C:1-3"})
            ids |= set(amber.resources(s))
    ids |= set(C.pipeline_resources(Project()))
    assert not sorted(i for i in ids if i not in C.RESOURCES)


def test_only_used_resources_are_cited():
    amber = TOOLS["amber-build"]
    s = amber.normalise({"rna_ff": "ol3", "nbfix19": False, "bph0": False, "water": "tip4p2005",
                         "mg_model": "nano", "mg_count": 2})
    ids = amber.resources(s)
    assert "cp" not in ids and "nbfix19" not in ids and "water_tip4p2005" in ids and "mg_tip4p_family" in ids
    p = Project()
    p.cleaning.enabled = False
    p.output.video.enabled = False
    assert "cpptraj" not in C.pipeline_resources(p) and "ffmpeg" not in C.pipeline_resources(p)


def test_script_header_cites_its_own_step_only(tmp_path):
    p = Project()
    b = add_node(p.workflow, "amber-build", settings={"water": "tip4p2005"})
    b.files["structure"] = "/x.pdb"
    g = add_node(p.workflow, "gromacs")
    plan = Plan(p.workflow, str(tmp_path))
    files = generate_step(plan.steps[b.id])
    sh = [f for f in files if f.role == "run"][0].content
    head = sh.split("set -eo pipefail")[0]
    assert head.startswith("#!/bin/bash") and "Please cite" in head and "Abascal" in head
    assert "Abraham" not in head  # GROMACS belongs to the other step
    assert C.step_citations(plan.steps[g.id]) and "abraham2015" in C.step_citations(plan.steps[g.id])
    keys = C.keys_for(C.project_resources(p, plan))
    assert "abascal2005" in keys and "abraham2015" in keys and "michaudagrawal2011" in keys
