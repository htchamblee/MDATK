"""What the README and the command-line help say matches the code: the decoy-validation template's chain,
the project layout's folders, the leftover-process note of 0.0.12 and the program that computes Q-scores."""

import os
import re

import pytest

from mdatk.cli import main
from mdatk.workflows.plan import TEMPLATES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def readme() -> str:
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        return fh.read()


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_decoy_validation_template_as_the_readme_describes_it():
    label, steps, _edges = TEMPLATES["decoy-validation"]
    assert label.endswith("→ kinks") and ("kinks", "model-check") in [(k, t) for k, t, *_r in steps]
    text = flat(readme())
    assert "→ map-model score (Q-score, interface vs scaffold dQ) → model check (kinks)" in text
    assert "still to come" not in text  # the kink test is in the template


def test_the_project_layout_names_folders_that_exist():
    section = readme().split("## Architecture and extending MDATK", 1)[1]
    block = section.split("```text", 1)[1].split("```", 1)[0]
    top = re.findall(r"^([\w.-]+)/", block, re.M) + re.findall(r"\b([\w.-]+)/:", block)
    package = re.findall(r"^  ([\w.-]+)/", block, re.M)  # the folders of mdatk/
    assert "tests" in top and "assets" in package
    for name in top:
        assert os.path.isdir(os.path.join(ROOT, name)), f"README's project layout names {name}/"
    for name in package:
        assert os.path.isdir(os.path.join(ROOT, "mdatk", name)), f"the layout names mdatk/{name}/"
    assets = flat(block.split("  assets/", 1)[1].split("\n  frozen.py", 1)[0])
    for name in ("mdatk.desktop", "rnado.desktop"):  # the desktop-entry templates live in mdatk/assets
        assert name in assets and os.path.isfile(os.path.join(ROOT, "mdatk", "assets", name))


def test_whats_new_does_not_call_the_appimage_leftover_process_fixed():
    new = flat(readme().split("## What's new in 0.0.12", 1)[1].split("\n---", 1)[0])
    assert "no longer stays running" not in new and "no longer inherit" not in new
    assert "quits when its main window closes" in new and "close_fds" in new
    assert "still being investigated" in new


def test_the_help_names_chimerax_for_q_scores(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = flat(capsys.readouterr().out)
    assert "map-model scores per residue (MDATK correlation or ChimeraX Q-score)" in text
    assert "Phenix Q-score" not in text
