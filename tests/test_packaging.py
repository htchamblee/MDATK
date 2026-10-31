"""Stand-alone builds: frozen-app helpers, the script-runner guard, AppRun and the trim lists.

The builds themselves (PyInstaller, AppImage, Windows) are made and smoke-tested by
packaging/linux/build_appimage.sh, packaging/windows/build_windows.sh and packaging/smoke_test.py.
"""

import os
import subprocess
import sys

import pytest

from mdatk import frozen
from mdatk.workflows import runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGING = os.path.join(ROOT, "packaging")


def test_frozen_helpers_from_source(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert not frozen.is_frozen()
    assert frozen.bundle_dir() is None and frozen.tools_dir() is None
    assert frozen.bundled_tool("ffmpeg") is None
    assert frozen.child_env() is None
    frozen.setup()
    assert os.environ["PATH"] == "/usr/bin"


@pytest.fixture
def fake_bundle(tmp_path, monkeypatch):
    """A PyInstaller-like onedir layout: dist/mdatk (executable) + _internal (_MEIPASS) with tools/ffmpeg."""
    internal = tmp_path / "mdatk" / "_internal"
    tools = internal / "tools"
    tools.mkdir(parents=True)
    ffmpeg = tools / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\n")
    ffmpeg.chmod(0o755)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "mdatk" / "mdatk"))
    return internal


@pytest.mark.skipif(os.name == "nt", reason="POSIX paths")
def test_frozen_setup_and_child_env(fake_bundle, monkeypatch):
    internal = str(fake_bundle)
    tools = os.path.join(internal, "tools")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{internal}:/opt/amber/lib")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.setenv("GI_TYPELIB_PATH", os.path.join(internal, "gi_typelibs"))
    monkeypatch.setenv("GDK_PIXBUF_MODULE_FILE", "/tmp/tmpab12cd")  # PyInstaller's temporary loader cache
    monkeypatch.setenv("FONTCONFIG_FILE", "/etc/fonts/my.conf")  # the user's own: kept
    monkeypatch.setenv("XDG_DATA_DIRS", f"{internal}/share:/usr/share")
    assert frozen.is_frozen() and frozen.bundle_dir() == internal
    assert frozen.tools_dir() == tools and frozen.bundled_tool("ffmpeg") == os.path.join(tools, "ffmpeg")

    frozen.setup()
    assert os.environ["PATH"] == f"{tools}:/usr/local/bin:/usr/bin"  # bundled FFmpeg first
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/amber/lib"  # programs started later: the user's value
    frozen.setup()  # idempotent
    assert os.environ["PATH"].count(tools) == 1

    env = frozen.child_env()
    assert "GI_TYPELIB_PATH" not in env
    assert "GDK_PIXBUF_MODULE_FILE" not in env and env["FONTCONFIG_FILE"] == "/etc/fonts/my.conf"
    assert env["XDG_DATA_DIRS"] == "/usr/share"
    assert env["PATH"] == f"{tools}:/usr/local/bin:/usr/bin"

    monkeypatch.setenv("LD_LIBRARY_PATH", internal)
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/home/me/lib")  # saved by the PyInstaller bootloader
    frozen.setup()
    assert os.environ["LD_LIBRARY_PATH"] == "/home/me/lib" and "LD_LIBRARY_PATH_ORIG" not in os.environ

    # AppRun's fallback folder (inside the bundle) is not passed on either
    monkeypatch.setenv("LD_LIBRARY_PATH", internal)
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", f"{internal}/fallback:/home/me/lib")
    frozen.setup()
    assert os.environ["LD_LIBRARY_PATH"] == "/home/me/lib"
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", f"{internal}/fallback")
    frozen.setup()
    assert "LD_LIBRARY_PATH" not in os.environ


def test_runner_guard(monkeypatch, tmp_path):
    script = tmp_path / "step.sh"
    script.write_text("echo hi\n")
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    problem = runner.runner_problem()
    assert problem and "bash" in problem
    with pytest.raises(runner.ScriptFailed, match="bash"):
        runner.run_script(str(script), lambda line: None)


def test_runner_guard_windows(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(runner.os, "name", "nt")
        problem = runner.runner_problem()
    assert "WSL" in problem and "Linux" in problem


@pytest.mark.skipif(not runner.shutil.which("bash") or os.name == "nt", reason="needs bash")
def test_runner_runs_with_bash(tmp_path):
    assert runner.runner_problem() is None
    script = tmp_path / "step.sh"
    script.write_text("echo hello from bash\n")
    lines = []
    code, out = runner.run_script(str(script), lines.append)
    assert code == 0 and lines == ["hello from bash"]


@pytest.mark.skipif(os.name == "nt", reason="AppRun is a POSIX shell script")
@pytest.mark.parametrize("args, argv0, expected", [
    ([], None, "gui"),
    (["--version"], None, "--version"),
    (["demo", "out"], None, "demo out"),
    (["@project"], None, "gui @project"),
    (["--help"], "/somewhere/mdatk-gui", "gui --help"),
])
def test_apprun_dispatch(tmp_path, args, argv0, expected):
    appdir = tmp_path / "AppDir"
    exe = appdir / "usr" / "lib" / "mdatk" / "mdatk"
    exe.parent.mkdir(parents=True)
    exe.write_text('#!/bin/sh\necho "$@"\n')
    exe.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_text(open(os.path.join(PACKAGING, "linux", "AppRun"), encoding="utf-8").read())
    apprun.chmod(0o755)
    project = tmp_path / "x.mdatk.toml"
    project.write_text("")
    args = [str(project) if a == "@project" else a for a in args]
    env = {k: v for k, v in os.environ.items() if k != "ARGV0"}
    if argv0:
        env["ARGV0"] = argv0
    out = subprocess.run([str(apprun), *args], capture_output=True, text=True, env=env, check=True).stdout
    assert out.strip() == expected.replace("@project", str(project))


@pytest.mark.skipif(os.name == "nt", reason="AppRun is a POSIX shell script")
@pytest.mark.parametrize("libname, used", [("libmdatk-not-on-any-host.so.0", True), ("libc.so.6", False)])
def test_apprun_fallback_libraries(tmp_path, libname, used):
    appdir = tmp_path / "AppDir"
    exe = appdir / "usr" / "lib" / "mdatk" / "mdatk"
    fallback = exe.parent / "_internal" / "fallback"
    fallback.mkdir(parents=True)
    (fallback / libname).write_text("")
    exe.write_text('#!/bin/sh\necho "${LD_LIBRARY_PATH:-}"\n')
    exe.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_text(open(os.path.join(PACKAGING, "linux", "AppRun"), encoding="utf-8").read())
    apprun.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("ARGV0", "LD_LIBRARY_PATH")}
    env["LD_LIBRARY_PATH"] = "/opt/amber/lib"
    out = subprocess.run([str(apprun), "--version"], capture_output=True, text=True, env=env, check=True).stdout
    assert out.strip() == (f"{fallback}:/opt/amber/lib" if used else "/opt/amber/lib")


def test_trim_lists():
    sys.path.insert(0, PACKAGING)
    try:
        import trim
    finally:
        sys.path.remove(PACKAGING)
    for needed in ("numpy", "numpy.f2py", "scipy", "matplotlib", "MDAnalysis", "barnaba", "mdtraj", "gi",
                   "cairo", "PIL", "mdatk", "unittest"):
        assert needed not in trim.EXCLUDES
    assert not trim.keep_entry("numpy/_core/tests/data/umath-validation-set-arccos.csv")
    assert not trim.keep_entry("share/locale/de/LC_MESSAGES/gtk30.mo")
    assert not trim.keep_entry("share/icons/hicolor/48x48/apps/libreoffice-calc.png")
    assert not trim.keep_entry("matplotlib/mpl-data/sample_data/goog.npz")
    assert not trim.keep_entry("libgcc_s.so.1")
    for kept in ("mdtraj/formats/pdb/data/residues.xml", "share/icons/hicolor/index.theme",
                 "share/icons/Adwaita/16x16/legacy/document-open.png",
                 "matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf", "share/glib-2.0/schemas/gschemas.compiled",
                 "libgcc_s_seh-1.dll", "lib/gdk-pixbuf-2.0/2.10.0/loaders.cache"):
        assert trim.keep_entry(kept), kept


def test_spec_and_scripts_parse():
    compile(open(os.path.join(PACKAGING, "mdatk.spec"), encoding="utf-8").read(), "mdatk.spec", "exec")
    if os.name == "nt":
        return
    for script in ("linux/build_appimage.sh", "linux/check_glibc.sh", "windows/build_windows.sh"):
        subprocess.run(["bash", "-n", os.path.join(PACKAGING, script)], check=True)
    subprocess.run(["sh", "-n", os.path.join(PACKAGING, "linux", "AppRun")], check=True)


def test_gui_run_explains_where_scripts_run(monkeypatch):
    """GUI 'Run' on Windows (or without bash): an explanation instead of a run."""
    pytest.importorskip("gi")
    try:
        from mdatk.gui.rnado import RNAdoWindow
    except (ImportError, ValueError):  # pragma: no cover - no GTK 3 typelibs
        pytest.skip("GTK 3 not available")

    class Fake:
        project_path = None

        def __init__(self):
            self.messages = []
            self.info = lambda t, d="": self.messages.append(("info", t, d))
            self.error = lambda t, d="": self.messages.append(("error", t, d))

    monkeypatch.setattr(runner.os, "name", "nt")
    try:
        fake = Fake()
        RNAdoWindow.run_scripts(fake, [runner.ScriptJob(1, "Amber build", "/p/01_build.sh")])
    finally:
        monkeypatch.undo()
    ((kind, title, detail),) = fake.messages
    assert kind == "info" and "not run" in title and "WSL" in detail and "/p/01_build.sh" in detail
