# -*- mode: python -*-
# PyInstaller spec of the stand-alone MDATK builds (Linux AppImage and Windows).
#
#   pyinstaller --noconfirm --distpath DIST --workpath WORK packaging/mdatk.spec
#
# Produces the folder DIST/mdatk with
#   mdatk      the command line interface (console program)
#   mdatk-gui  Windows only: the GTK 3 interface without a console window, same as "mdatk gui"
#              (on Linux the AppImage's AppRun starts "mdatk gui")
# The build scripts then add FFmpeg (DIST/mdatk/_internal/tools) and the licence notices.
#
# Environment:
#   MDATK_ICON_DIR       folder with mdatk.ico (Windows program icon), from packaging/make_icon.py
#   MDATK_NO_BARNABA=1   build without the barnaba base-pair engine
#   MDATK_NO_TRIM=1      keep everything PyInstaller collects (to measure what trimming saves)
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

sys.path.insert(0, SPECPATH)
from trim import EXCLUDES, keep_entry  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(SPECPATH))
WINDOWS = sys.platform == "win32"
ICON = os.path.join(os.environ.get("MDATK_ICON_DIR", ""), "mdatk.ico")
ICON = ICON if WINDOWS and os.path.isfile(ICON) else None
TRIM = not os.environ.get("MDATK_NO_TRIM")

# Every MDATK module (the CLI imports most of them lazily); the VTK backend's optional
# "vtkmodules" is not bundled, so it reports VTK as unavailable.
hiddenimports = collect_submodules("mdatk")
# Compiled (Cython) modules import each other in ways PyInstaller cannot see.
hiddenimports += collect_submodules(
    "MDAnalysis", filter=lambda n: not n.startswith(("MDAnalysis.analysis.", "MDAnalysis.visualization",
                                                     "MDAnalysis.tests")) or n == "MDAnalysis.analysis.rms")
datas = []
if not os.environ.get("MDATK_NO_BARNABA"):
    hiddenimports += ["barnaba", "barnaba.functions"]
    hiddenimports += collect_submodules(
        "mdtraj", filter=lambda n: not n.startswith(("mdtraj.html", "mdtraj.testing", "mdtraj.scripts")))
    datas += copy_metadata("barnaba")  # the PDB annotation records barnaba's version
    datas += collect_data_files("mdtraj", excludes=["**/tests/**", "**/testing/**"])  # PDB residue tables

a = Analysis(
    [os.path.join(SPECPATH, "entry_cli.py"), os.path.join(SPECPATH, "entry_gui.py")],
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


def scripts_without(name):
    return [s for s in a.scripts if s[0] != name]


exe_cli = EXE(
    pyz,
    scripts_without("entry_gui"),
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
        scripts_without("entry_cli"),
        [],
        exclude_binaries=True,
        name="mdatk-gui",
        debug=False,
        strip=False,
        upx=False,
        console=False,
        icon=ICON,
    ))

coll = COLLECT(
    *exes,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mdatk",
)
