# -*- mode: python -*-
# PyInstaller spec of the stand-alone MDATK builds (Linux AppImage and Windows).
#
#   pyinstaller --noconfirm --distpath DIST --workpath WORK packaging/mdatk.spec
#
# Produces the folder DIST/mdatk with
#   mdatk      the command line interface (console program)
#   mdatk-gui  Windows only: the GTK 3 interface without a console window, same as "mdatk gui"
#              (on Linux the AppImage's AppRun starts "mdatk gui")
#   rnado-gui  RNAdo, the project planner, as a program of its own (no console window on Windows):
#              "rnado-gui [PROJECT] [ARGS]" = "python -m mdatk.gui.rnado [PROJECT] [ARGS]"; the
#              analysis window starts it from here (mdatk.gui.launch.rnado_command)
# All three share one Analysis, so the bundle holds every library once. The RNAdo logo and icons
# (mdatk/assets) are bundled as package data. The build scripts then add FFmpeg
# (DIST/mdatk/_internal/tools) and the licence notices.
#
# Environment:
#   MDATK_ICON_DIR       folder with mdatk.ico and rnado.ico (Windows program icons), from
#                        packaging/make_icon.py
#   MDATK_NO_BARNABA=1   build without the barnaba base-pair engine
#   MDATK_NO_TRIM=1      keep everything PyInstaller collects (to measure what trimming saves)
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

sys.path.insert(0, SPECPATH)
from trim import EXCLUDES, keep_entry  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(SPECPATH))
WINDOWS = sys.platform == "win32"
TRIM = not os.environ.get("MDATK_NO_TRIM")


def program_icon(name):
    """The .ico of a Windows program (MDATK_ICON_DIR); None elsewhere or when it was not made."""
    path = os.path.join(os.environ.get("MDATK_ICON_DIR", ""), name)
    return path if WINDOWS and os.path.isfile(path) else None


ICON = program_icon("mdatk.ico")
RNADO_ICON = program_icon("rnado.ico")
ENTRIES = {"mdatk": "entry_cli", "mdatk-gui": "entry_gui", "rnado-gui": "entry_rnado"}  # program -> script

# Every MDATK module (the CLI imports most of them lazily); the VTK backend's optional
# "vtkmodules" is not bundled, so it reports VTK as unavailable.
hiddenimports = collect_submodules("mdatk")
# Compiled (Cython) modules import each other in ways PyInstaller cannot see.
hiddenimports += collect_submodules(
    "MDAnalysis", filter=lambda n: not n.startswith(("MDAnalysis.analysis.", "MDAnalysis.visualization",
                                                     "MDAnalysis.tests")) or n == "MDAnalysis.analysis.rms")
# the RNAdo logo and icons, read by the GUI next to the package (mdatk/assets/...)
datas = [(os.path.join(ROOT, "mdatk", "assets", "*"), os.path.join("mdatk", "assets"))]
if not os.environ.get("MDATK_NO_BARNABA"):
    hiddenimports += ["barnaba", "barnaba.functions"]
    hiddenimports += collect_submodules(
        "mdtraj", filter=lambda n: not n.startswith(("mdtraj.html", "mdtraj.testing", "mdtraj.scripts")))
    datas += copy_metadata("barnaba")  # the PDB annotation records barnaba's version
    datas += collect_data_files("mdtraj", excludes=["**/tests/**", "**/testing/**"])  # PDB residue tables

a = Analysis(
    [os.path.join(SPECPATH, script + ".py") for script in ENTRIES.values()],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "gi": {
            "icons": ["Adwaita", "hicolor"],
            "themes": ["Adwaita"],
            "languages": [],  # no translations: the interface is English
            "module-versions": {"Gtk": "3.0", "Gdk": "3.0"},
        },
        "matplotlib": {"backends": ["Agg", "PDF", "PS", "SVG"]},  # charts; the GUI draws with cairo itself
    },
    runtime_hooks=[],
    excludes=(EXCLUDES if TRIM else []) + (["barnaba", "mdtraj"] if os.environ.get("MDATK_NO_BARNABA") else []),
    noarchive=False,
    # MDATK embeds the source of some of its modules in generated scripts (inspect.getsource),
    # so its .py files are bundled next to the compiled ones; mdtraj.core finds its PDB tables
    # through "core/../formats", which needs a real mdtraj/core folder.
    module_collection_mode={"mdatk": "pyz+py", "mdtraj.core": "pyz+py"},
    optimize=0,
)

# trimming: tests, translations, documentation, headers, sample data, large icon sizes
if TRIM:
    a.datas = [e for e in a.datas if keep_entry(e[0])]
    a.binaries = [e for e in a.binaries if keep_entry(e[0])]

pyz = PYZ(a.pure)


def scripts_for(program):
    """PyInstaller's bootstrap and runtime hooks plus the program's own entry script (not the others')."""
    others = {script for name, script in ENTRIES.items() if name != program}
    return [s for s in a.scripts if s[0] not in others]


exe_cli = EXE(
    pyz,
    scripts_for("mdatk"),
    [],
    exclude_binaries=True,
    name="mdatk",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    icon=ICON,
)
exes = [exe_cli]
if WINDOWS:
    exes.append(EXE(
        pyz,
        scripts_for("mdatk-gui"),
        [],
        exclude_binaries=True,
        name="mdatk-gui",
        debug=False,
        strip=False,
        upx=False,
        console=False,
        icon=ICON,
    ))
# RNAdo on every platform: the AppImage's AppRun and mdatk.gui.launch start it from the bundle
exes.append(EXE(
    pyz,
    scripts_for("rnado-gui"),
    [],
    exclude_binaries=True,
    name="rnado-gui",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=RNADO_ICON,
))

coll = COLLECT(
    *exes,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mdatk",
)
