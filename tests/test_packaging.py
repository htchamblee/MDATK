"""Stand-alone builds: frozen-app helpers, the script-runner guard, AppRun and the trim lists, and
RNAdo as the builds' second program (spec, AppRun, desktop files, icons) with its art in mdatk/assets.

The builds themselves (PyInstaller, AppImage, Windows) are made and smoke-tested by
packaging/linux/build_appimage.sh, packaging/windows/build_windows.sh and packaging/smoke_test.py.
"""

import glob
import math
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
                 "mdatk/assets/rnado-icon-16.png", "mdatk/assets/rnado-icon.svg",
                 "share/icons/Adwaita/16x16/legacy/document-open.png",
                 "matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf", "share/glib-2.0/schemas/gschemas.compiled",
                 "libgcc_s_seh-1.dll", "lib/gdk-pixbuf-2.0/2.10.0/loaders.cache"):
        assert trim.keep_entry(kept), kept


def test_spec_and_scripts_parse():
    compile(open(os.path.join(PACKAGING, "mdatk.spec"), encoding="utf-8").read(), "mdatk.spec", "exec")
    for script in ("entry_cli.py", "entry_gui.py", "entry_rnado.py", "make_icon.py", "make_rnado_art.py"):
        compile(open(os.path.join(PACKAGING, script), encoding="utf-8").read(), script, "exec")
    if os.name == "nt":
        return
    for script in ("linux/build_appimage.sh", "linux/check_glibc.sh", "windows/build_windows.sh"):
        subprocess.run(["bash", "-n", os.path.join(PACKAGING, script)], check=True)
    subprocess.run(["sh", "-n", os.path.join(PACKAGING, "linux", "AppRun")], check=True)


def test_gui_run_explains_where_scripts_run(monkeypatch):
    """GUI 'Run' on Windows (or without bash): an explanation instead of a run."""
    pytest.importorskip("gi")
    try:
        from mdatk.gui.app import MainWindow
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
        MainWindow.run_scripts(fake, [runner.ScriptJob(1, "Amber build", "/p/01_build.sh")])
    finally:
        monkeypatch.undo()
    ((kind, title, detail),) = fake.messages
    assert kind == "info" and "not run" in title and "WSL" in detail and "/p/01_build.sh" in detail


# --- RNAdo: the builds' second program and its art in mdatk/assets ----------------------------------------
ASSETS = os.path.join(ROOT, "mdatk", "assets")
RNADO_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def _packaging_module(name):
    sys.path.insert(0, PACKAGING)
    try:
        return __import__(name)
    finally:
        sys.path.remove(PACKAGING)


@pytest.mark.skipif(os.name == "nt", reason="AppRun is a POSIX shell script")
@pytest.mark.parametrize("args, argv0, expected", [
    (["rnado"], None, "rnado-gui"),
    (["rnado", "@project"], None, "rnado-gui @project"),
    (["rnado", "@project", "--open", "wizard"], None, "rnado-gui @project --open wizard"),  # launch.py
    (["@project"], "/home/me/bin/rnado", "rnado-gui @project"),  # a symbolic link named rnado
    ([], "/home/me/bin/rnado-gui", "rnado-gui"),
    (["--version"], None, "mdatk --version"),  # everything else as before
    (["@project"], None, "mdatk gui @project"),
    (["workflow", "list"], None, "mdatk workflow list"),
])
def test_apprun_dispatches_rnado(tmp_path, args, argv0, expected):
    appdir = tmp_path / "AppDir"
    lib = appdir / "usr" / "lib" / "mdatk"
    lib.mkdir(parents=True)
    for name in ("mdatk", "rnado-gui"):  # fake programs: print their name and arguments
        exe = lib / name
        exe.write_text(f'#!/bin/sh\necho {name} "$@"\n')
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


def _run_spec(monkeypatch, platform, icon_dir):
    """Execute mdatk.spec with recording stand-ins for PyInstaller: (Analysis, {name: EXE}, namespace)."""
    import types

    made = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            made.append(self)

    class Analysis(Recorder):
        def __init__(self, scripts, **kwargs):
            super().__init__(scripts, **kwargs)
            boot = [("pyiboot01_bootstrap", "/x/pyiboot01_bootstrap.py", "PYSOURCE")]
            self.scripts = boot + [(os.path.basename(s)[:-3], s, "PYSOURCE") for s in scripts]
            self.pure, self.binaries, self.datas = [], [], []

    class EXE(Recorder):
        pass

    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_submodules = lambda name, filter=None: [name]
    hooks.collect_data_files = lambda name, **kwargs: []
    hooks.copy_metadata = lambda name: []
    for mod in ("PyInstaller", "PyInstaller.utils"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the spec puts packaging/ in front
    monkeypatch.setenv("MDATK_ICON_DIR", str(icon_dir))
    monkeypatch.delenv("MDATK_NO_BARNABA", raising=False)
    source = open(os.path.join(PACKAGING, "mdatk.spec"), encoding="utf-8").read()
    ns = {"SPECPATH": PACKAGING, "Analysis": Analysis, "PYZ": Recorder, "EXE": EXE, "COLLECT": Recorder}
    with monkeypatch.context() as m:
        m.setattr(sys, "platform", platform)
        exec(compile(source, "mdatk.spec", "exec"), ns)
    analysis = next(r for r in made if isinstance(r, Analysis))
    return analysis, {r.kwargs["name"]: r for r in made if isinstance(r, EXE)}, ns


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_spec_builds_rnado_gui(monkeypatch, tmp_path, platform):
    for name in ("mdatk.ico", "rnado.ico"):
        (tmp_path / name).write_bytes(b"")
    analysis, exes, ns = _run_spec(monkeypatch, platform, tmp_path)
    assert set(exes) == {"mdatk", "rnado-gui"} | ({"mdatk-gui"} if platform == "win32" else set())
    rnado = exes["rnado-gui"]
    assert rnado.kwargs["console"] is False and exes["mdatk"].kwargs["console"] is True
    # every program runs PyInstaller's bootstrap and its own entry script, not the others'
    scripts = {name: [s[0] for s in exe.args[1]] for name, exe in exes.items()}
    assert scripts["rnado-gui"] == ["pyiboot01_bootstrap", "entry_rnado"]
    assert scripts["mdatk"] == ["pyiboot01_bootstrap", "entry_cli"]
    if platform == "win32":
        assert scripts["mdatk-gui"] == ["pyiboot01_bootstrap", "entry_gui"]
        assert rnado.kwargs["icon"] == str(tmp_path / "rnado.ico")
        assert exes["mdatk"].kwargs["icon"] == str(tmp_path / "mdatk.ico")
    else:
        assert rnado.kwargs["icon"] is None
    assert os.path.join(PACKAGING, "entry_rnado.py") in analysis.args[0]
    # the art travels as package data (mdatk/assets/<file> in the bundle) and survives the trimming
    src, dest = next(d for d in analysis.kwargs["datas"] if "assets" in d[1])
    assert dest.replace("\\", "/") == "mdatk/assets"
    found = {os.path.basename(p) for p in glob.glob(src)}
    assert {"rnado-icon.svg", "rnado-logo.png", "rnado-wordmark.png", "rnado-icon-16.png"} <= found
    assert all(ns["keep_entry"](f"mdatk/assets/{name}") for name in found)


def test_entry_rnado_starts_the_rnado_gui():
    source = open(os.path.join(PACKAGING, "entry_rnado.py"), encoding="utf-8").read()
    assert "from mdatk.gui.rnado import main" in source and "main(sys.argv[1:])" in source
    assert source.index("setup()") < source.index("from mdatk.gui.rnado import main")  # frozen set-up first


def test_smoke_test_starts_rnado():
    smoke_test = _packaging_module("smoke_test")
    appimage = "/d/MDATK-0.0.11-x86_64.AppImage"
    assert smoke_test.rnado_command(appimage) == [appimage, "rnado"]
    program = "rnado-gui.exe" if os.name == "nt" else "rnado-gui"
    exe = os.path.join("d", "dist", "mdatk", "mdatk")
    assert smoke_test.rnado_command(exe) == [os.path.join("d", "dist", "mdatk", program)]


def test_rnado_assets_exist_with_their_sizes():
    Image = pytest.importorskip("PIL.Image")
    import xml.etree.ElementTree as ET

    total = 0
    for n in RNADO_ICON_SIZES:
        path = os.path.join(ASSETS, f"rnado-icon-{n}.png")
        with Image.open(path) as img:
            assert img.format == "PNG" and img.size == (n, n), path
            rgba = img.convert("RGBA")
            assert rgba.getpixel((0, 0))[3] < 40, "transparent corners (rounded square)"
            assert rgba.getpixel((n // 2, n // 2))[3] == 255
        total += os.path.getsize(path)
    for name, size in (("rnado-logo.png", (784, 1168)), ("rnado-wordmark.png", (1200, 360))):
        path = os.path.join(ASSETS, name)
        with Image.open(path) as img:
            assert img.format == "PNG" and img.size == size, name
        total += os.path.getsize(path)
    svg = os.path.join(ASSETS, "rnado-icon.svg")
    root = ET.parse(svg).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg" and root.get("viewBox") == "0 0 512 512"
    tags = {el.tag.split("}")[-1] for el in root.iter()}
    assert not tags & {"text", "image", "font", "glyph", "use"}  # vector paths only: no text, no raster
    total += os.path.getsize(svg)
    assert total < 3_000_000


def test_rnado_icons_stay_in_the_reference_colours():
    """Reds, red-orange highlights and black (the white text is in the logo, not in the icon)."""
    Image = pytest.importorskip("PIL.Image")
    import numpy as np

    for n in RNADO_ICON_SIZES:
        with Image.open(os.path.join(ASSETS, f"rnado-icon-{n}.png")) as img:
            px = np.asarray(img.convert("RGBA")).reshape(-1, 4).astype(float)
        px = px[px[:, 3] > 128]
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        assert (g <= 0.62 * r + 6).all() and (b <= 0.42 * r + 6).all(), n  # no yellow, green or blue


def test_rnado_ico_from_the_drawn_pngs(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    make_icon = _packaging_module("make_icon")
    make_icon.rnado(str(tmp_path))
    with Image.open(tmp_path / "rnado.ico") as ico:
        assert set(ico.info["sizes"]) == {(n, n) for n in make_icon.ICO_SIZES}
    with Image.open(tmp_path / "rnado.png") as png:
        assert png.size == (256, 256)


def test_rnado_desktop_entries():
    import configparser

    def read(name):
        cp = configparser.ConfigParser(interpolation=None)
        cp.optionxform = str
        cp.read(os.path.join(ROOT, "data", name), encoding="utf-8")
        return cp

    rnado = read("rnado.desktop")["Desktop Entry"]
    assert rnado["Name"] == "RNAdo" and rnado["Icon"] == "rnado" and rnado["Type"] == "Application"
    assert rnado["Exec"].split()[0] == "rnado-gui"  # the package's gui-script
    mdatk = read("mdatk.desktop")
    assert "rnado" in mdatk["Desktop Entry"]["Actions"].split(";")
    assert mdatk["Desktop Action rnado"]["Exec"].split()[0] == "rnado-gui"


def test_pyproject_installs_the_assets():
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        config = tomllib.load(fh)
    assert "assets/*" in config["tool"]["setuptools"]["package-data"]["mdatk"]


def test_rnado_art_keeps_the_a_form_geometry():
    """The art generator's taper keeps twist, rise and grooves, and its view is a rotation (no mirror)."""
    for mod in ("cairo", "PIL", "scipy", "matplotlib"):
        pytest.importorskip(mod)
    import numpy as np
    from scipy.interpolate import CubicSpline

    art = _packaging_module("make_rnado_art")
    for azimuth, elevation in ((0, 0), (art.VIEW_AZIMUTH, art.VIEW_ELEVATION), (250, 30)):
        assert np.linalg.det(art.View(azimuth, elevation, 1, 0, 0, 0).basis()) == pytest.approx(1.0)

    model = art.load_model(art.SEQUENCE)
    p = model.trace[0][:-1]  # the P atoms of strand 1, 5'->3'
    phi = np.degrees(np.unwrap(np.arctan2(p[:, 1], p[:, 0])))
    # right-handed: 5'->3' runs down (-z) turning clockwise seen from above, 32.7 degrees and 2.81 A a step
    assert np.diff(phi) == pytest.approx(-32.7, abs=0.05)
    assert np.diff(p[:, 2]) == pytest.approx(-2.81, abs=0.01)
    funnel = art.Funnel(*model.z_range)
    q = funnel.apply(p)
    assert np.arctan2(q[:, 1], q[:, 0]) == pytest.approx(np.arctan2(p[:, 1], p[:, 0]))
    assert q[:, 2] == pytest.approx(p[:, 2])
    assert np.hypot(q[:, 0], q[:, 1]) == pytest.approx(np.hypot(p[:, 0], p[:, 1]) * funnel.scale(p[:, 2]))

    # the pairs sit off the axis (the hollow core) and are inclined, as in A-RNA
    centre = (model.hb[0] + model.hb[1][::-1]) / 2
    off_axis = np.hypot(centre[:, 0], centre[:, 1])
    assert ((off_axis > 4.0) & (off_axis < 6.0)).all()
    c1 = model.c1[0] - model.c1[1][::-1]
    tilt = np.degrees(np.arcsin(np.abs(c1[:, 2]) / np.linalg.norm(c1, axis=1)))
    assert ((tilt > 14) & (tilt < 24)).all()

    # along a vertical line on the near side the P ribbons leave the narrow major groove (11.3 A) and the
    # wide minor groove (19.6 A) between them, as VIEW_AZIMUTH documents
    near = math.radians(90 - art.VIEW_AZIMUTH)
    crossings = []
    for strand in model.trace:
        s = CubicSpline(np.arange(len(strand)), strand, axis=0)(np.linspace(0, len(strand) - 1, 6000))
        d = np.angle(np.exp(1j * (np.arctan2(s[:, 1], s[:, 0]) - near)))
        hit = (np.sign(d[:-1]) != np.sign(d[1:])) & (np.abs(d[:-1]) < 1.0)
        crossings += list(s[:-1][hit][:, 2])
    gaps = np.diff(sorted(crossings))
    assert len(gaps) >= 3
    assert all(abs(g - 11.3) < 0.6 or abs(g - 19.6) < 0.6 for g in gaps)
    assert any(abs(g - 11.3) < 0.6 for g in gaps) and any(abs(g - 19.6) < 0.6 for g in gaps)
