"""RNAdo without a display: the command that starts it (and the environment it and its scripts get),
``rnado`` / ``mdatk rnado`` dispatching to the workflow actions, and the program's independence from
the analysis window."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from mdatk import cli
from mdatk.config import Project
from mdatk.gui import launch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_rnado_command_from_source(monkeypatch, tmp_path):
    project = str(tmp_path / "p.mdatk.toml")
    monkeypatch.delattr(sys, "frozen", raising=False)
    # a source run in a terminal some AppImage started: its variables do not count
    monkeypatch.setenv("APPIMAGE", "/apps/Other.AppImage")
    monkeypatch.setenv("APPDIR", "/tmp/.mount_Other")
    assert launch.launch_mode() == "source"
    assert launch.rnado_command(project) == [sys.executable, "-m", "mdatk.gui.rnado", project]
    assert launch.rnado_command(project, "pdb") == [sys.executable, "-m", "mdatk.gui.rnado", project,
                                                    "--open", "pdb"]
    assert launch.rnado_command(None, "wizard") == [sys.executable, "-m", "mdatk.gui.rnado",
                                                    "--open", "wizard"]
    with pytest.raises(ValueError):
        launch.rnado_command(project, "planner")


def _launched(monkeypatch, project, tool=None):
    """(command, Popen keywords) launch_rnado uses, without starting anything."""
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))

    with monkeypatch.context() as m:  # subprocess itself: undone at once
        m.setattr(launch.subprocess, "Popen", FakePopen)
        launch.launch_rnado(project, tool)
    (cmd, kw), = calls
    return cmd, kw


@pytest.mark.skipif(os.name == "nt", reason="the AppImage is Linux only")
@pytest.mark.parametrize("appimage, appdir, inside, mounted, extract, mode", [
    (None, None, False, False, None, "bundle"),  # a bundle folder
    ("/apps/MDATK.AppImage", None, False, False, None, "bundle"),  # APPIMAGE alone, passed on by a terminal
    ("/apps/Other.AppImage", "/tmp/.mount_Other", False, True, None, "bundle"),  # another AppImage's
    ("/apps/MDATK.AppImage", "@appdir", True, True, None, "appimage"),  # this program from its FUSE mount
    ("/apps/MDATK.AppImage", "@appdir", True, False, None, "appimage-extracted"),  # extracted: no FUSE
    ("/apps/MDATK.AppImage", "@appdir", True, True, "1", "appimage-extracted"),  # APPIMAGE_EXTRACT_AND_RUN=1
])
def test_rnado_command_of_a_frozen_build(monkeypatch, tmp_path, appimage, appdir, inside, mounted, extract,
                                         mode):
    mount = tmp_path / "mount"
    folder = (mount / "usr" / "lib" / "mdatk") if inside else (tmp_path / "dist" / "mdatk")
    folder.mkdir(parents=True)
    appdir = str(mount) if appdir == "@appdir" else appdir
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / "mdatk"))
    for name, value in (("APPIMAGE", appimage), ("APPDIR", appdir), ("APPIMAGE_EXTRACT_AND_RUN", extract)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("NO_CLEANUP", raising=False)
    ismount = os.path.ismount
    monkeypatch.setattr(launch.os.path, "ismount", lambda p: mounted if p == appdir else ismount(p))
    assert launch.launch_mode() == mode
    project = str(tmp_path / "p.mdatk.toml")
    cmd, kw = _launched(monkeypatch, project, "wizard")
    if mode == "bundle":
        assert cmd == [str(folder / "rnado-gui"), project, "--open", "wizard"]
    else:  # the AppImage again: RNAdo gets a mount (or extraction) of its own
        assert cmd == [appimage, "rnado", project, "--open", "wizard"]
    env = kw["env"]
    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1" and "MDATK_ORIG_PYTHONPATH" not in env
    # without FUSE the child runs extracted too, and leaves the folder this program runs from in place
    extracted = mode == "appimage-extracted"
    assert env.get("APPIMAGE_EXTRACT_AND_RUN") == ("1" if extracted else None)
    assert env.get("NO_CLEANUP") == ("1" if extracted else None)


def test_launch_passes_this_mdatk_on(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.delenv("MDATK_ORIG_PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    project = str(tmp_path / "p.mdatk.toml")
    cmd, kw = _launched(monkeypatch, project, "pdb")
    assert cmd == launch.rnado_command(project, "pdb")
    # a source checkout: in front of the user's PYTHONPATH, which is recorded for the scripts RNAdo runs
    assert kw["env"]["PYTHONPATH"].split(os.pathsep) == [ROOT, "/somewhere/else"]
    assert kw["env"]["MDATK_ORIG_PYTHONPATH"] == "/somewhere/else"
    assert kw["stdin"] == subprocess.DEVNULL
    # installed (site-packages, a venv): importable anyway, so PYTHONPATH stays as it is
    monkeypatch.setattr(launch.site, "getsitepackages", lambda: ["/usr/lib/python3/dist-packages", ROOT])
    _cmd, kw = _launched(monkeypatch, project)
    assert kw["env"]["PYTHONPATH"] == "/somewhere/else" == kw["env"]["MDATK_ORIG_PYTHONPATH"]
    monkeypatch.delenv("PYTHONPATH")
    _cmd, kw = _launched(monkeypatch, project)
    assert "PYTHONPATH" not in kw["env"] and kw["env"]["MDATK_ORIG_PYTHONPATH"] == ""


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("user", [None, "/home/me/python"])
def test_scripts_rnado_runs_see_the_users_pythonpath(monkeypatch, tmp_path, user):
    """RNAdo started by the analysis window from source has MDATK on its PYTHONPATH; the scripts it runs get
    the user's own back (another Python they start, ParmEd from AmberTools say, must not import this
    Python's packages)."""
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delenv("MDATK_ORIG_PYTHONPATH", raising=False)
    if user is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", user)
    _cmd, kw = _launched(monkeypatch, str(tmp_path / "p.mdatk.toml"))
    env = kw["env"]
    assert ROOT in env["PYTHONPATH"].split(os.pathsep)  # RNAdo itself imports this mdatk
    script = tmp_path / "step.sh"
    script.write_text('echo "PYTHONPATH=${PYTHONPATH-unset}"\necho "ORIG=${MDATK_ORIG_PYTHONPATH-unset}"\n')
    # RNAdo's process (the environment launch_rnado gave it) runs the script the way run_jobs does
    code = ("import sys\nfrom mdatk.workflows.runner import run_script\n"
            "sys.exit(run_script(sys.argv[1], print)[0])\n")
    out = subprocess.run([sys.executable, "-c", code, str(script)], env=env, cwd=str(tmp_path),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"PYTHONPATH={user or 'unset'}", "ORIG=unset"]


def test_watch_start_reports_a_start_that_fails_at_once():
    try:
        from gi.repository import GLib
    except ImportError:  # pragma: no cover
        pytest.skip("PyGObject not installed")

    def watch(code: str):
        proc = subprocess.Popen([sys.executable, "-c", code])
        got = []
        launch.watch_start(proc, lambda: got.append("started"), got.append, seconds=1.0, every_ms=20)
        ctx, end = GLib.MainContext.default(), time.monotonic() + 20
        while not got and time.monotonic() < end:
            ctx.iteration(False)
            time.sleep(0.01)
        proc.kill()
        proc.wait()
        return got

    (failed,) = watch("import sys; sys.exit(3)")
    assert failed.startswith("RNAdo ended right after it started (exit status 3)")
    assert f"Command: {sys.executable} -c 'import sys; sys.exit(3)'" in failed
    assert watch("import time; time.sleep(30)") == ["started"]  # still running after the time
    assert watch("pass") == ["started"]  # ended without an error (closed at once)


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


def test_the_window_of_mdatk_rnado_is_named_rnado(tmp_path):
    """The AppImage runs 'mdatk rnado [PROJECT]': the window opens in that process, whose program name (the
    windows' WM_CLASS, by which desktops group them) is RNAdo's, not the analysis window's."""
    pytest.importorskip("gi")
    code = ("import sys, types\n"
            "fake = types.ModuleType('mdatk.gui.rnado')\n"
            "def main(argv=None):\n"
            "    import gi\n"
            "    gi.require_version('Gtk', '3.0')\n"
            "    from gi.repository import GLib, Gtk\n"
            "    print(GLib.get_prgname(), *argv)\n"
            "    return 0\n"
            "fake.main = main\n"
            "import mdatk.gui\n"
            "sys.modules['mdatk.gui.rnado'] = fake\n"
            "sys.argv[0] = '/tmp/.mount_MDATK/usr/lib/mdatk/mdatk'  # the stand-alone program\n"
            "from mdatk.cli import main\n"
            "sys.exit(main(['rnado', 'p.mdatk.toml', '--open', 'pdb']))\n")
    out = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=ROOT),
                         cwd=str(tmp_path), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["rnado", "p.mdatk.toml", "--open", "pdb"]


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
