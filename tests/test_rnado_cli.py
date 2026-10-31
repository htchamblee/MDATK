"""RNAdo without a display: the command that starts it, ``rnado`` / ``mdatk rnado`` dispatching to the
workflow actions, and the program's independence from the analysis window."""

import argparse
import json
import os
import subprocess
import sys

import pytest

from mdatk import cli
from mdatk.config import Project
from mdatk.gui import launch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_rnado_command_in_its_three_modes(monkeypatch, tmp_path):
    project = str(tmp_path / "p.mdatk.toml")
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    # from source or an installation
    assert launch.rnado_command(project) == [sys.executable, "-m", "mdatk.gui.rnado", project]
    assert launch.rnado_command(project, "pdb") == [sys.executable, "-m", "mdatk.gui.rnado", project,
                                                    "--open", "pdb"]
    assert launch.rnado_command(None, "wizard") == [sys.executable, "-m", "mdatk.gui.rnado",
                                                    "--open", "wizard"]
    # a source run in a terminal some AppImage started: APPIMAGE alone does not count
    monkeypatch.setenv("APPIMAGE", "/apps/Other.AppImage")
    assert launch.rnado_command(project)[:3] == [sys.executable, "-m", "mdatk.gui.rnado"]
    # PyInstaller one-dir build: the rnado-gui program next to this one
    monkeypatch.delenv("APPIMAGE")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/mdatk/mdatk")
    exe = "rnado-gui.exe" if os.name == "nt" else "rnado-gui"
    assert launch.rnado_command(project, "builder") == [os.path.join("/opt/mdatk", exe), project,
                                                        "--open", "builder"]
    # the AppImage
    monkeypatch.setenv("APPIMAGE", "/apps/MDATK-x86_64.AppImage")
    assert launch.rnado_command(project) == ["/apps/MDATK-x86_64.AppImage", "rnado", project]
    assert launch.rnado_command(project, "wizard")[-2:] == ["--open", "wizard"]
    with pytest.raises(ValueError):
        launch.rnado_command(project, "planner")


def test_launch_passes_this_mdatk_on(monkeypatch, tmp_path):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))

    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setattr(launch.subprocess, "Popen", FakePopen)
    launch.launch_rnado(str(tmp_path / "p.mdatk.toml"), "pdb")
    (cmd, kw), = calls
    assert cmd == launch.rnado_command(str(tmp_path / "p.mdatk.toml"), "pdb")
    assert kw["env"]["PYTHONPATH"].split(os.pathsep) == [ROOT, "/somewhere/else"]
    assert kw["stdin"] == subprocess.DEVNULL


def test_workflow_actions_match_the_parser():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    workflow = sub.choices["workflow"]
    wsub = next(a for a in workflow._actions if isinstance(a, argparse._SubParsersAction))
    assert set(cli.WORKFLOW_ACTIONS) == set(wsub.choices)
    assert set(cli.RNADO_TOOLS) == set(launch.TOOLS)


def test_rnado_main_dispatches_workflow_actions(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "p.mdatk.toml")
    p = Project()
    p.output.out_dir = str(tmp_path / "out")
    p.save(path)
    # 'rnado template NAME -p FILE' is 'mdatk workflow template NAME -p FILE'
    assert cli.rnado_main(["template", "amber", "-p", path]) == 0
    added = json.loads(capsys.readouterr().out)["added"]
    assert added and Project.load(path).workflow.nodes
    assert cli.rnado_main(["check", "-p", path]) == cli.main(["workflow", "check", "-p", path])
    out = capsys.readouterr().out
    assert out.count('"steps"') == 2
    # 'mdatk rnado ACTION' too (the AppImage runs 'mdatk rnado ...')
    assert cli.main(["rnado", "tools"]) == 0
    assert "user-script" in capsys.readouterr().out
    # everything else opens the window, with the arguments as given
    seen = []
    import types
    fake = types.ModuleType("mdatk.gui.rnado")
    fake.main = lambda argv=None: seen.append(argv) or 0
    monkeypatch.setitem(sys.modules, "mdatk.gui.rnado", fake)
    monkeypatch.setattr("mdatk.gui.rnado", fake, raising=False)
    assert cli.rnado_main([]) == 0 and cli.rnado_main([path, "--open", "pdb"]) == 0
    assert cli.main(["rnado", path]) == 0
    assert cli.main(["gui", path, "--rnado"]) == 0 and cli.main(["gui", "--open", "wizard"]) == 0
    assert seen == [[], [path, "--open", "pdb"], [path], [path], ["--open", "wizard"]]
    with pytest.raises(SystemExit) as stop:
        cli.rnado_main(["--open", "planner"])  # wrong arguments end before any window
    assert stop.value.code == 2
    with pytest.raises(SystemExit) as stop:
        cli.rnado_main(["--help"])
    assert stop.value.code == 0 and "usage: rnado" in capsys.readouterr().out


def test_rnado_help_needs_no_gtk_and_rnado_never_loads_the_analysis_window():
    env = dict(os.environ, PYTHONPATH=ROOT)
    code = ("import sys, mdatk.cli as c\n"
            "try:\n    c.rnado_main(['--help'])\nexcept SystemExit:\n    pass\n"
            "assert 'gi' not in sys.modules, 'rnado --help imported GTK'\n"
            "import mdatk.gui.launch\n"
            "assert 'gi' not in sys.modules\n")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT, capture_output=True)
    pytest.importorskip("gi")
    code = ("import sys\n"
            "try:\n    import mdatk.gui.rnado\nexcept (ImportError, ValueError):\n    sys.exit(0)\n"
            "assert 'mdatk.gui.app' not in sys.modules, 'RNAdo imported the analysis window'\n"
            "assert 'mdatk.gui.canvas' not in sys.modules\n")
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
