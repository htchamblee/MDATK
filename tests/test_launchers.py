"""Menu entries for MDATK and RNAdo (mdatk/launchers.py, 'mdatk desktop', the GUI's menu item): the Exec
lines of every way MDATK runs, their quoting, install / status / uninstall in a temporary XDG data folder
(only files MDATK wrote are replaced or removed), and the menu item of both windows."""

import configparser
import os
import stat
import struct
import sys
import time

import pytest

from mdatk import __version__, cli, launchers
from mdatk.gui import launch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "mdatk", "assets")
APPIMAGE = "/opt/My Apps/MDATK $1 (x86_64).AppImage"  # a space, a dollar sign and parentheses
pytestmark = pytest.mark.skipif(os.name == "nt", reason="desktop entries are written on Linux only")


@pytest.fixture(autouse=True)
def data_home(tmp_path, monkeypatch):
    """Every test writes into its own XDG data folder, never into ~/.local/share; no cache tools run."""
    folder = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(folder))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(launchers.shutil, "which", lambda name, *a, **k: None)
    return folder


def appimage_mode(monkeypatch, path=APPIMAGE, mode="appimage"):
    monkeypatch.setenv("APPIMAGE", path)
    monkeypatch.setattr(launch, "launch_mode", lambda: mode)


def read_entry(path):
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(path, encoding="utf-8")
    return cp


def make_exe(path, text="#!/bin/sh\nexit 0\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


# ---------------------------------------------------------------------------
# Exec lines and their quoting
# ---------------------------------------------------------------------------


def test_exec_arguments_are_quoted_per_the_specification():
    assert launchers.exec_arg("/usr/bin/mdatk-gui") == "/usr/bin/mdatk-gui"
    assert launchers.exec_arg("/opt/My Apps/x") == '"/opt/My Apps/x"'
    assert launchers.exec_arg('a"b`c$d\\e') == '"a\\"b\\`c\\$d\\\\e"'
    assert launchers.exec_arg("~/x") == '"~/x"' and launchers.exec_arg("") == '""'
    assert launchers.exec_arg("100%/x") == "100%%/x"  # a literal % is no field code
    assert launchers.exec_arg("PYTHONPATH=/a b") == '"PYTHONPATH=/a b"'
    # the string escapes of the file come on top: a literal backslash is four in a quoted argument
    assert launchers.exec_value(["/opt/My Apps/MDATK.AppImage", "gui"]) == \
        '"/opt/My Apps/MDATK.AppImage" gui %f'
    assert launchers.exec_value(["/a\\b c"]) == '"/a\\\\\\\\b c" %f'
    assert launchers.exec_value(["/a\tb"], code="") == '"/a\\tb"'


@pytest.mark.parametrize("argv", [
    ["/opt/My Apps/MDATK.AppImage", "gui"],
    [APPIMAGE, "rnado"],
    ["env", "PYTHONPATH=/home/u/src/My MDATK", "/usr/bin/python3", "-m", "mdatk.gui.rnado"],
    ["/x/it's a \"test\"/back\\slash/100%/~tilde/a;b|c&d<e>f*g?h#i", "gui"],
])
def test_exec_lines_round_trip_through_glib(tmp_path, argv):
    """GLib (GTK desktops) reads the value as a string and splits it with shell quoting: the same argv."""
    gi = pytest.importorskip("gi")
    from gi.repository import GLib

    kf = GLib.KeyFile()
    data = f"[Desktop Entry]\nExec={launchers.exec_value(argv)}\n"
    kf.load_from_data(data, len(data.encode()), GLib.KeyFileFlags.NONE)
    ok, parsed = GLib.shell_parse_argv(kf.get_string("Desktop Entry", "Exec"))
    assert ok and gi
    assert [a.replace("%%", "%") for a in parsed] == argv + ["%f"]


def test_a_desktop_launch_passes_paths_with_spaces_as_one_argument(tmp_path):
    """The whole way: Gio starts the entry's program, with a space and a dollar sign in its path, and
    hands it the file (%f) as one argument."""
    pytest.importorskip("gi")
    from gi.repository import Gio, GLib

    record = tmp_path / "argv.txt"
    program = make_exe(tmp_path / "My Apps $HOME" / "fake mdatk",
                       f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > "{record}"\n')
    project = tmp_path / "a project (1).mdatk.toml"
    project.write_text("")
    entry = tmp_path / "test-mdatk.desktop"
    entry.write_text("[Desktop Entry]\nType=Application\nName=T\n"
                     f"Exec={launchers.exec_value([program, 'gui'])}\n", encoding="utf-8")
    info = Gio.DesktopAppInfo.new_from_filename(str(entry))
    assert info is not None
    assert info.launch([Gio.File.new_for_path(str(project))], None)
    end = time.time() + 10
    while time.time() < end and not record.exists():
        GLib.MainContext.default().iteration(False)
        time.sleep(0.05)
    time.sleep(0.1)
    assert record.read_text().splitlines() == ["gui", str(project)]


@pytest.mark.parametrize("folder", ["pct 100%", "a%b", "50%off=x"])
def test_a_program_path_with_a_percent_sign_still_makes_an_entry(tmp_path, monkeypatch, folder):
    """GLib looks the program up before it turns %% back into %: an AppImage in 'pct 100%' made entries
    that GLib dropped (no menu entry, no message).  They start through env (sh for a path with '=')."""
    pytest.importorskip("gi")
    from gi.repository import Gio, GLib

    record = tmp_path / "argv.txt"
    program = make_exe(tmp_path / folder / "MDATK.AppImage",
                       f'#!/bin/sh\nprintf "%s\\n" "$0" "$@" > "{record}"\n')
    appimage_mode(monkeypatch, program)
    cmds = launchers.commands()
    project = tmp_path / "p.mdatk.toml"
    project.write_text("")
    entry = tmp_path / "test-rnado.desktop"
    entry.write_text(launchers.entry_text("rnado", cmds), encoding="utf-8")
    try:
        info = Gio.DesktopAppInfo.new_from_filename(str(entry))
    except TypeError:  # "constructor returned NULL": GLib refused the entry
        info = None
    assert info is not None, launchers.exec_value(cmds["rnado"])
    assert cmds["rnado"][-2:] == [program, "rnado"] and cmds["rnado"][0] in ("env", "sh")
    assert info.launch([Gio.File.new_for_path(str(project))], None)
    end = time.time() + 10
    while time.time() < end and not record.exists():
        GLib.MainContext.default().iteration(False)
        time.sleep(0.05)
    time.sleep(0.1)
    assert record.read_text().splitlines() == [program, "rnado", str(project)]


def test_commands_from_the_appimage(monkeypatch):
    monkeypatch.setenv("APPIMAGE", APPIMAGE)
    assert launchers.commands("appimage") == {"mdatk": [APPIMAGE, "gui"], "rnado": [APPIMAGE, "rnado"]}
    # without FUSE MDATK runs extracted: so do the entries
    assert launchers.commands("appimage-extracted")["rnado"] == \
        ["env", "APPIMAGE_EXTRACT_AND_RUN=1", APPIMAGE, "rnado"]
    assert launchers.exec_value(launchers.commands("appimage")["mdatk"]) == \
        '"/opt/My Apps/MDATK \\\\$1 (x86_64).AppImage" gui %f'
    assert "current path" in launchers.warning("appimage") and APPIMAGE in launchers.warning("appimage")


@pytest.mark.parametrize("appdir, inside, mounted, extract, mode", [
    ("@appdir", True, True, None, "appimage"),
    ("@appdir", True, False, None, "appimage-extracted"),
    ("@appdir", True, True, "1", "appimage-extracted"),
    (None, False, False, None, "bundle"),
    ("/tmp/.mount_Other", False, True, None, "bundle"),  # a terminal another AppImage started
])
def test_commands_follow_the_launch_mode(monkeypatch, tmp_path, appdir, inside, mounted, extract, mode):
    """sys.frozen, APPIMAGE / APPDIR and sys.executable decide, as for 'Open in RNAdo…'."""
    mount = tmp_path / "mount"
    folder = (mount / "usr" / "lib" / "mdatk") if inside else (tmp_path / "dist" / "mdatk")
    make_exe(folder / "mdatk")
    make_exe(folder / "rnado-gui")
    appdir = str(mount) if appdir == "@appdir" else appdir
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / "rnado-gui"))  # RNAdo asks: the same entries
    for name, value in (("APPIMAGE", APPIMAGE), ("APPDIR", appdir), ("APPIMAGE_EXTRACT_AND_RUN", extract)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    ismount = os.path.ismount
    monkeypatch.setattr(launch.os.path, "ismount", lambda p: mounted if p == appdir else ismount(p))
    cmds = launchers.commands()
    if mode == "bundle":
        assert cmds == {"mdatk": [str(folder / "mdatk"), "gui"], "rnado": [str(folder / "rnado-gui")]}
        assert str(folder) in launchers.warning()
    else:
        env = ["env", "APPIMAGE_EXTRACT_AND_RUN=1"] if mode == "appimage-extracted" else []
        assert cmds == {"mdatk": env + [APPIMAGE, "gui"], "rnado": env + [APPIMAGE, "rnado"]}


def test_commands_of_a_bundle_folder(monkeypatch, tmp_path):
    folder = tmp_path / "MDATK 0.0.12" / "mdatk"
    make_exe(folder / "mdatk")
    make_exe(folder / "rnado-gui")
    monkeypatch.setattr(sys, "executable", str(folder / "mdatk"))
    assert launchers.commands("bundle") == {"mdatk": [str(folder / "mdatk"), "gui"],
                                            "rnado": [str(folder / "rnado-gui")]}
    make_exe(folder / "mdatk-gui")  # the Windows layout: a windowed program of its own
    assert launchers.commands("bundle")["mdatk"] == [str(folder / "mdatk-gui")]
    # an extracted AppImage (squashfs-root/usr/lib/mdatk): its AppRun, which sets up the library path
    folder = tmp_path / "squashfs-root" / "usr" / "lib" / "mdatk"
    make_exe(folder / "mdatk")
    apprun = make_exe(tmp_path / "squashfs-root" / "AppRun")
    monkeypatch.setattr(sys, "executable", str(folder / "mdatk"))
    assert launchers.commands("bundle") == {"mdatk": [apprun, "gui"], "rnado": [apprun, "rnado"]}


def test_commands_of_a_pip_installation(monkeypatch, tmp_path):
    """The gui scripts that run this Python (venv bin, user bin, PATH); else python -m."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python3"
    os.symlink(sys.executable, python)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(launchers, "_installed", lambda: True)
    monkeypatch.setattr(launchers, "_scripts_dirs", lambda: [str(venv)])
    # nothing installed as scripts: python -m
    assert launchers.commands("source") == {"mdatk": [str(python), "-m", "mdatk", "gui"],
                                            "rnado": [str(python), "-m", "mdatk.gui.rnado"]}
    # pip's scripts: a #! line, or the /bin/sh form pip writes when the interpreter's path has a space
    gui = make_exe(venv / "mdatk-gui", f"#!{python}\nfrom mdatk.gui.app import main\n")
    rnado = make_exe(venv / "rnado-gui", f"#!/bin/sh\n'''exec' \"{python}\" \"$0\" \"$@\"\n' '''\n")
    assert launchers.commands("source") == {"mdatk": [gui], "rnado": [rnado]}
    assert launchers.warning("source") is None
    # a script of another Python (the system's, which the venv's python links to) is not this installation
    make_exe(venv / "mdatk-gui", f"#!{os.path.realpath(sys.executable)}\n")
    make_exe(venv / "rnado-gui", "#!/usr/bin/env python3\n")
    assert launchers.commands("source")["mdatk"] == [str(python), "-m", "mdatk", "gui"]
    assert launchers.commands("source")["rnado"] == [str(python), "-m", "mdatk.gui.rnado"]
    # one on PATH that runs this Python counts too
    other = tmp_path / "bin"
    make_exe(other / "rnado-gui", f"#!{python}\n")
    monkeypatch.setattr(launchers.shutil, "which", lambda name, *a, **k: str(other / name)
                        if (other / name).exists() else None)
    assert launchers.commands("source")["rnado"] == [str(other / "rnado-gui")]


def test_the_interpreter_of_a_script(tmp_path):
    spaced = "/home/u/My Envs/mdatk/bin/python3"
    sh_form = f"#!/bin/sh\n'''exec' \"{spaced}\" \"$0\" \"$@\"\n' '''\nimport sys\n"
    assert launchers._interpreter(make_exe(tmp_path / "a", "#!/usr/bin/python3 -E\n")) == "/usr/bin/python3"
    assert launchers._interpreter(make_exe(tmp_path / "b", sh_form)) == spaced
    assert launchers._interpreter(make_exe(tmp_path / "c", "#!/usr/bin/env python3\n")) is None
    assert launchers._interpreter(make_exe(tmp_path / "d", "#!/bin/sh\nexec python3 -m mdatk\n")) == "/bin/sh"
    assert launchers._interpreter(make_exe(tmp_path / "e", "\x7fELF")) is None
    assert launchers._interpreter(str(tmp_path / "missing")) is None


def test_commands_of_a_source_checkout_that_is_not_installed(monkeypatch):
    monkeypatch.setattr(launchers, "_installed", lambda: False)
    cmds = launchers.commands("source")
    assert cmds["mdatk"] == ["env", f"PYTHONPATH={ROOT}", sys.executable, "-m", "mdatk", "gui"]
    assert cmds["rnado"][-2:] == ["-m", "mdatk.gui.rnado"]
    assert ROOT in launchers.warning("source")


def test_installed_recognises_an_editable_installation(monkeypatch, tmp_path):
    from importlib import metadata

    from mdatk.gui import launch as launch_mod

    class Dist:
        def __init__(self, url):
            self.url = url

        def read_text(self, name):
            if name != "direct_url.json":
                return None
            return '{"url": "%s", "dir_info": {"editable": true}}' % self.url

    monkeypatch.setattr(launch_mod, "_site_dir", lambda path: False)
    monkeypatch.setattr(metadata, "distribution", lambda name: Dist("file://" + ROOT.replace(" ", "%20")))
    assert launchers._installed()
    monkeypatch.setattr(metadata, "distribution", lambda name: Dist("file:///somewhere/else"))
    assert not launchers._installed()


# ---------------------------------------------------------------------------
# the files: install, status, uninstall
# ---------------------------------------------------------------------------


def test_install_status_and_uninstall(monkeypatch, data_home):
    appimage_mode(monkeypatch)
    assert launchers.data_home() == str(data_home)
    found = launchers.status()
    assert {t.state for t in found} == {launchers.MISSING}
    result = launchers.install(update_caches=False)
    apps = data_home / "applications"
    hicolor = data_home / "icons" / "hicolor"
    assert sorted(os.listdir(apps)) == ["mdatk.desktop", "rnado.desktop"]
    sizes = sorted(int(n.split("x")[0]) for n in os.listdir(hicolor) if n != "scalable")
    assert sizes == [16, 24, 32, 48, 64, 128, 256, 512]
    for n in (16, 24, 32, 48, 64, 128, 256):
        assert (hicolor / f"{n}x{n}" / "apps" / "mdatk.png").is_file()
    for n in sizes:
        assert (hicolor / f"{n}x{n}" / "apps" / "rnado.png").is_file()
    assert (hicolor / "scalable" / "apps" / "rnado.svg").is_file()
    assert len(result.written) == 2 + 7 + 8 + 1 and not result.kept
    # the entries: the templates' fields, Exec lines starting the AppImage, the icons, the mark
    mdatk = read_entry(apps / "mdatk.desktop")
    rnado = read_entry(apps / "rnado.desktop")
    quoted = launchers.exec_arg(APPIMAGE).replace("\\", "\\\\")
    assert mdatk["Desktop Entry"]["Exec"] == f"{quoted} gui %f"
    assert mdatk["Desktop Action rnado"]["Exec"] == rnado["Desktop Entry"]["Exec"] == f"{quoted} rnado %f"
    assert mdatk["Desktop Entry"]["Icon"] == "mdatk" and rnado["Desktop Entry"]["Icon"] == "rnado"
    assert mdatk["Desktop Entry"][launchers.MARK] == rnado["Desktop Entry"][launchers.MARK] == __version__
    assert mdatk["Desktop Entry"]["Name"] == "MDATK" and rnado["Desktop Entry"]["Name"] == "RNAdo"
    assert rnado["Desktop Entry"]["MimeType"] == "application/toml;"
    assert "xdg-icon-resource" not in (apps / "rnado.desktop").read_text()  # the template's comments are gone
    for path in result.written:
        assert launchers.is_marked(path), path
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o644
    # icons stay valid images
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None
    if Image is not None:
        with Image.open(hicolor / "48x48" / "apps" / "mdatk.png") as img:
            assert img.size == (48, 48) and img.info.get(launchers.MARK) == __version__
            img.load()
    assert open(hicolor / "scalable" / "apps" / "rnado.svg", "rb").read().startswith(b"<?xml")
    # status: all there; again: nothing written
    assert {t.state for t in launchers.status()} == {launchers.INSTALLED}
    again = launchers.install(update_caches=False)
    assert not again.written and len(again.unchanged) == len(result.written)
    # the AppImage moved: the entries are MDATK's but outdated; install rewrites only them
    appimage_mode(monkeypatch, "/home/u/Applications/MDATK.AppImage")
    states = {os.path.basename(t.path): t.state for t in launchers.status()}
    assert states["mdatk.desktop"] == states["rnado.desktop"] == launchers.OUTDATED
    outdated = [t for t in launchers.status() if t.state == launchers.OUTDATED]
    assert outdated[0].exec_now == f"{quoted} gui %f"
    assert [os.path.basename(p) for p in launchers.install(update_caches=False).written] == \
        ["mdatk.desktop", "rnado.desktop"]
    assert read_entry(apps / "rnado.desktop")["Desktop Entry"]["Exec"] == \
        "/home/u/Applications/MDATK.AppImage rnado %f"
    # uninstall: everything MDATK wrote, nothing else
    removed = launchers.uninstall(update_caches=False)
    assert sorted(removed.removed) == sorted(result.written) and not removed.kept
    assert os.listdir(apps) == [] and not [f for _d, _s, files in os.walk(hicolor) for f in files]


def test_files_mdatk_did_not_write_are_left_alone(monkeypatch, data_home):
    appimage_mode(monkeypatch)
    apps = data_home / "applications"
    icon = data_home / "icons" / "hicolor" / "256x256" / "apps" / "rnado.png"
    other_icon = data_home / "icons" / "hicolor" / "48x48" / "apps" / "other.png"
    for path in (apps / "x.desktop", icon, other_icon):
        path.parent.mkdir(parents=True, exist_ok=True)
    own = "[Desktop Entry]\nType=Application\nName=My RNAdo\nExec=/home/u/bin/rnado-gui %f\nIcon=rnado\n"
    (apps / "rnado.desktop").write_text(own)
    (apps / "x.desktop").write_text("[Desktop Entry]\nName=X\nExec=x\n")
    with open(os.path.join(ASSETS, "rnado-icon-256.png"), "rb") as fh:
        icon.write_bytes(fh.read())  # e.g. 'xdg-icon-resource install … rnado' by hand
    other_icon.write_bytes(b"\x89PNG\r\n\x1a\nnot really")
    states = {t.path: t.state for t in launchers.status()}
    assert states[str(apps / "rnado.desktop")] == states[str(icon)] == launchers.FOREIGN
    result = launchers.install(update_caches=False)
    assert sorted(result.kept) == sorted([str(apps / "rnado.desktop"), str(icon)])
    assert (apps / "rnado.desktop").read_text() == own and (apps / "mdatk.desktop").is_file()
    assert "Left alone" in launchers.plan_text(launchers.status(), "appimage")
    # an older version's icon of a size this one does not write: MDATK's too
    old = data_home / "icons" / "hicolor" / "22x22" / "apps" / "mdatk.png"
    old.parent.mkdir(parents=True)
    with open(os.path.join(ASSETS, "mdatk-icon-24.png"), "rb") as fh:
        old.write_bytes(launchers.mark_png(fh.read(), "0.0.11"))
    removed = launchers.uninstall(update_caches=False)
    assert str(old) in removed.removed and str(apps / "mdatk.desktop") in removed.removed
    assert str(apps / "rnado.desktop") in removed.kept and str(icon) in removed.kept
    assert (apps / "rnado.desktop").read_text() == own and icon.is_file() and other_icon.is_file()
    assert (apps / "x.desktop").is_file() and not old.exists() and not (apps / "mdatk.desktop").exists()
    # --replace: the user's files are moved aside, not lost
    result = launchers.install(replace=True, update_caches=False)
    assert (apps / "rnado.desktop.bak").read_text() == own
    assert (str(apps / "rnado.desktop"), str(apps / "rnado.desktop.bak")) in result.backups
    assert launchers.is_marked(str(apps / "rnado.desktop")) and launchers.is_marked(str(icon))
    assert (icon.parent / "rnado.png.bak").is_file()
    launchers.uninstall(update_caches=False)
    assert (apps / "rnado.desktop.bak").read_text() == own and not (apps / "rnado.desktop").exists()


def test_the_marks():
    with open(os.path.join(ASSETS, "rnado-icon-16.png"), "rb") as fh:
        png = fh.read()
    marked = launchers.mark_png(png, "9.9")
    assert marked[:8] == png[:8] and len(marked) == len(png) + 12 + len(launchers.MARK) + 1 + 3
    n = struct.unpack(">I", marked[33:37])[0]
    assert marked[37:41] == b"tEXt" and marked[41:41 + n] == launchers.MARK.encode() + b"\0" + b"9.9"
    with pytest.raises(ValueError):
        launchers.mark_png(b"GIF89a")
    svg = launchers.mark_svg(b'<?xml version="1.0"?>\n<svg/>\n', "9.9")
    assert svg.startswith(b'<?xml version="1.0"?>\n<!-- X-MDATK-Installed=9.9')
    assert launchers.mark_svg(b"<svg/>").startswith(b"<!-- X-MDATK-Installed=")


def test_entry_text_from_a_template():
    template = ("[Desktop Entry]\nType=Application\nName=MDATK\nExec=mdatk-gui %f\n"
                "Icon=applications-science\nActions=rnado;\n# a comment\n\n[Desktop Action rnado]\n"
                "Name=RNAdo\nExec=rnado-gui %f\nIcon=rnado\n")
    text = launchers.entry_text("mdatk", {"mdatk": ["/a b/mdatk-gui"], "rnado": ["/c/rnado-gui"]}, "mdatk",
                                template, version="1.2")
    lines = text.splitlines()
    assert lines[0].startswith("# Written by MDATK 1.2") and "# a comment" not in text
    assert lines[1:4] == ["[Desktop Entry]", "X-MDATK-Installed=1.2", "Type=Application"]
    assert 'Exec="/a b/mdatk-gui" %f' in lines and "Exec=/c/rnado-gui %f" in lines and "Icon=mdatk" in lines
    assert "Icon=rnado" in lines and text.endswith("Icon=rnado\n")
    # no icon of its own installed: the template's
    assert "Icon=applications-science" in launchers.entry_text(
        "mdatk", {"mdatk": ["x"], "rnado": ["y"]}, None, template)


def test_xdg_data_home_must_be_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", "relative/share")
    assert launchers.data_home() == os.path.join(str(tmp_path / "home"), ".local", "share")
    monkeypatch.delenv("XDG_DATA_HOME")
    assert launchers.data_home() == os.path.join(str(tmp_path / "home"), ".local", "share")


def test_cache_updates_run_when_present_and_never_fail(monkeypatch, tmp_path, data_home):
    appimage_mode(monkeypatch)
    calls = tmp_path / "calls.txt"
    tools = tmp_path / "tools"
    make_exe(tools / "update-desktop-database", f'#!/bin/sh\necho "udd $*" >> "{calls}"\necho broken >&2\n'
                                                "exit 3\n")
    make_exe(tools / "gtk-update-icon-cache", f'#!/bin/sh\necho "guic $*" >> "{calls}"\n')
    monkeypatch.setattr(launchers.shutil, "which",
                        lambda name, *a, **k: str(tools / name) if (tools / name).exists() else None)
    result = launchers.install()
    assert result.tools == ["update-desktop-database: exit status 3 (broken)"]  # no icon cache: none made
    assert calls.read_text() == f"udd -q {data_home / 'applications'}\n"
    (data_home / "icons" / "hicolor" / "icon-theme.cache").write_bytes(b"")
    result = launchers.uninstall()
    assert result.tools[1] == "gtk-update-icon-cache: done"
    assert calls.read_text().splitlines()[-1] == f"guic -q -f -t {data_home / 'icons' / 'hicolor'}"
    (tools / "gtk-update-icon-cache").unlink()
    (tools / "gtk-update-icon-cache").mkdir()  # cannot be run
    result = launchers.install()
    assert result.written and result.tools[1].startswith("gtk-update-icon-cache: not run (")
    assert launchers.install(replace=True).tools == []  # nothing written: nothing to tell


def test_windows_gets_a_message_instead_of_files(monkeypatch, data_home):
    lines = []
    with monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        why = launchers.unsupported()
        code = launchers.run("install", out=lines.append)
    assert "Start-menu" in why and code == 0 and lines == [why]
    assert not data_home.exists()
    with monkeypatch.context() as m:
        m.setattr(launchers, "unsupported", lambda: "no")
        with pytest.raises(RuntimeError):
            launchers.install()


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def test_mdatk_desktop_install_status_uninstall(monkeypatch, capsys, data_home):
    appimage_mode(monkeypatch)
    assert cli.main(["desktop", "install"]) == 0
    out = capsys.readouterr().out
    assert f"written    {data_home / 'applications' / 'rnado.desktop'}" in out
    assert "current path" in out and APPIMAGE in out
    assert cli.main(["desktop", "status"]) == 0
    out = capsys.readouterr().out
    assert f"installed    {data_home / 'applications' / 'mdatk.desktop'}" in out
    assert "MDATK and RNAdo are in the applications menu." in out and "RNAdo starts as: " in out
    assert cli.main(["desktop", "uninstall"]) == 0
    assert f"removed    {data_home / 'applications' / 'mdatk.desktop'}" in capsys.readouterr().out
    assert cli.main(["desktop", "uninstall"]) == 0
    assert "Nothing of MDATK's" in capsys.readouterr().out
    assert cli.main(["desktop"]) == 2
    # rnado --install-launchers (the AppImage: 'MDATK-*.AppImage rnado --install-launchers')
    assert cli.rnado_main(["--install-launchers"]) == 0
    assert "written" in capsys.readouterr().out and (data_home / "applications" / "mdatk.desktop").is_file()


def test_mdatk_desktop_needs_no_gtk(tmp_path):
    import subprocess

    code = ("import sys; from mdatk import cli; code = cli.main(['desktop', 'status']); "
            "assert 'gi' not in sys.modules, 'GTK loaded'; sys.exit(code)")
    env = dict(os.environ, PYTHONPATH=ROOT, XDG_DATA_HOME=str(tmp_path / "share"))
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert f"missing      {tmp_path / 'share' / 'applications' / 'mdatk.desktop'}" in proc.stdout


def test_rnado_gui_install_launchers_opens_no_window(monkeypatch, data_home):
    pytest.importorskip("gi")
    try:
        from mdatk.gui import rnado
    except (ImportError, ValueError):  # pragma: no cover - no GTK 3
        pytest.skip("GTK 3 not available")
    appimage_mode(monkeypatch)
    monkeypatch.setattr(rnado, "RNAdoApplication", lambda *a: pytest.fail("a window was opened"))
    assert rnado.main(["--install-launchers"]) == 0
    assert (data_home / "applications" / "rnado.desktop").is_file()


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------


def test_the_templates_and_mdatk_icons_are_package_data():
    for name in ("mdatk.desktop", "rnado.desktop"):  # templates: shipped wherever mdatk/assets is
        assert os.path.isfile(os.path.join(ASSETS, name))
    Image = pytest.importorskip("PIL.Image")
    for n in (16, 24, 32, 48, 64, 128, 256):
        with Image.open(os.path.join(ASSETS, f"mdatk-icon-{n}.png")) as img:
            assert img.format == "PNG" and img.size == (n, n)
            assert img.convert("RGBA").getpixel((0, 0))[3] < 40  # transparent corners (rounded square)


def test_mdatk_icons_are_the_drawn_icon():
    """packaging/make_icon.py --assets wrote them: the drawing at that size (Pillow versions may differ in
    the last bit of the resampling)."""
    Image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    import importlib.util

    path = os.path.join(ROOT, "packaging", "make_icon.py")
    spec = importlib.util.spec_from_file_location("make_icon", path)
    make_icon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_icon)
    drawn = np.asarray(make_icon.draw().resize((64, 64), Image.LANCZOS), dtype=float)
    with Image.open(os.path.join(ASSETS, "mdatk-icon-64.png")) as img:
        committed = np.asarray(img.convert("RGBA"), dtype=float)
    assert np.abs(drawn - committed).mean() < 2.0


# ---------------------------------------------------------------------------
# the menu item of both windows
# ---------------------------------------------------------------------------


@pytest.fixture
def gtk():
    gi = pytest.importorskip("gi")
    try:
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
    except (ValueError, ImportError):  # pragma: no cover
        pytest.skip("GTK 3 introspection data not installed")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no display (run under xvfb-run)")
    if not Gtk.init_check([])[0]:  # pragma: no cover
        pytest.skip("cannot initialise GTK")
    return Gtk


def _windows(Gtk, tmp_path):
    from mdatk.config import Project
    from mdatk.gui.app import MainWindow
    from mdatk.gui.rnado import RNAdoWindow

    app = Gtk.Application(application_id=None)
    app.register(None)
    path = str(tmp_path / "p.mdatk.toml")
    Project().save(path)
    made = [MainWindow(app, Project.load(path), path), RNAdoWindow(app, None, None)]
    for win in made:
        win.show_all()
        win.messages = []
        win.info = lambda t, d="", w=win: w.messages.append(("info", t, d))
        win.error = lambda t, d="", w=win: w.messages.append(("error", t, d))
    return made


def test_the_menu_item_in_both_windows_calls_the_installer(gtk, monkeypatch, tmp_path, data_home):
    from mdatk.gui import desktop

    Gtk = gtk
    appimage_mode(monkeypatch)
    asked, calls = [], []

    def ask(parent, text, detail, buttons, default, *a):
        asked.append((parent, text, detail, [label for label, _r in buttons]))
        return Gtk.ResponseType.ACCEPT

    def install(*a, **k):
        calls.append("install")
        return launchers.Result(str(data_home), written=[str(data_home / "applications" / "rnado.desktop")])

    real_install = launchers.install
    monkeypatch.setattr(desktop, "ask", ask)
    monkeypatch.setattr(launchers, "install", install)
    windows = _windows(Gtk, tmp_path)
    try:
        for win in windows:
            items = [i for i in win.main_menu.get_children() if isinstance(i, Gtk.MenuItem)
                     and i.get_label() == "Add MDATK and RNAdo to the applications menu…"]
            assert len(items) == 1, type(win).__name__
            items[0].activate()
            assert calls and calls.pop() == "install"
            parent, text, detail, buttons = asked.pop()
            assert parent is win and text == "Add MDATK and RNAdo to the applications menu?"
            assert buttons == ["Cancel", "Add to the menu"]  # nothing installed yet: nothing to remove
            assert "applications/: mdatk.desktop, rnado.desktop" in detail
            assert "icons/hicolor/scalable/apps/: rnado.svg" in detail
            assert "current path" in detail and "Moving or renaming the file breaks them" in detail
            assert launchers.exec_value([APPIMAGE, "rnado"]) in detail
            (kind, title, said), = win.messages
            assert kind == "info" and title == "MDATK and RNAdo are in the applications menu"
            assert "rnado.desktop" in said and "current path" in said
            win.messages.clear()
        # cancelled: nothing written
        monkeypatch.setattr(desktop, "ask", lambda *a, **k: Gtk.ResponseType.CANCEL)
        assert windows[1].add_launchers() == "cancelled" and not calls and not windows[1].messages
        # installed for real, then removed through the same item
        monkeypatch.setattr(launchers, "install", real_install)
        monkeypatch.setattr(desktop, "ask", lambda *a, **k: Gtk.ResponseType.ACCEPT)
        assert windows[0].add_launchers() == "installed"
        assert (data_home / "applications" / "rnado.desktop").is_file()
        buttons = []
        monkeypatch.setattr(desktop, "ask", lambda _p, _t, _d, b, *a: buttons.extend(x for x, _ in b)
                            or Gtk.ResponseType.REJECT)
        assert windows[1].add_launchers() == "removed" and "Remove them" in buttons
        assert not (data_home / "applications" / "rnado.desktop").exists()
        (kind, title, said), = windows[1].messages
        assert title == "MDATK and RNAdo were removed from the applications menu"
        assert "applications/: mdatk.desktop, rnado.desktop" in said and "Left alone" not in said
        # Windows: a message, no question, no files
        monkeypatch.setattr(launchers, "unsupported", lambda: "The installer made Start-menu entries.")
        monkeypatch.setattr(desktop, "ask", lambda *a, **k: pytest.fail("asked on Windows"))
        windows[0].messages.clear()
        assert windows[0].add_launchers() == "unsupported"
        assert windows[0].messages == [("info", "MDATK and RNAdo have Start-menu entries already",
                                        "The installer made Start-menu entries.")]
    finally:
        for win in windows:
            win.destroy()
