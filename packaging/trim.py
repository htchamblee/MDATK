"""What the stand-alone builds leave out (used by ``mdatk.spec``; also run as a size report).

Two levels:

* ``EXCLUDES``: Python packages PyInstaller must not follow (optional imports of
  numpy / scipy / matplotlib / MDAnalysis / mdtraj that MDATK never uses, GUI
  toolkits other than GTK, test frameworks).
* ``keep_entry``: data files and libraries dropped from the collected bundle
  (test folders, translations, documentation, headers, matplotlib sample data,
  large icon sizes).

``python packaging/trim.py DIST_DIR`` prints the size of the biggest folders of
a built bundle, which is how the lists below were chosen.
"""

from __future__ import annotations

import os
import re
import sys

EXCLUDES = [
    # other GUI toolkits and interactive shells
    "tkinter", "_tkinter", "turtle", "idlelib", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "IPython",
    "ipykernel", "jupyter_client", "notebook", "jedi",
    *(f"matplotlib.backends.{b}" for b in (
        "backend_qt", "backend_qtagg", "backend_qtcairo", "backend_qt5", "backend_qt5agg", "backend_tkagg",
        "backend_tkcairo", "_backend_tk", "backend_wx", "backend_wxagg", "backend_webagg", "backend_nbagg",
        "backend_macosx")),
    # optional scientific stacks MDATK does not use (MDAnalysis / mdtraj / barnaba import them when present)
    "pandas", "sklearn", "numba", "llvmlite", "sympy", "networkx", "tables", "h5py", "dask", "pyarrow",
    "netCDF4", "cftime", "vtk", "vtkmodules", "rdkit", "openmm", "parmed", "Bio", "nglview", "astropy",
    "skimage", "statsmodels", "seaborn", "plotly", "bokeh", "zarr", "numcodecs", "torch", "tensorflow",
    "cupy", "pyopencl", "pooch", "scipy.datasets",
    # build / test tooling
    "pytest", "_pytest", "py", "hypothesis", "Cython", "pyximport", "setuptools", "pkg_resources", "pip",
    "distutils", "numpy.distutils", "sphinx", "docutils", "pydoc_data", "lib2to3",  # (scipy needs numpy.f2py)
    # barnaba / mdtraj parts MDATK does not call (MDATK uses barnaba.annotate only)
    "barnaba.cluster", "barnaba.smm", "barnaba.smm_old", "barnaba.sec_str_svg", "barnaba.commandline",
    "mdtraj.html", "mdtraj.testing", "mdtraj.scripts",
]

_DROP = [re.compile(p) for p in (
    r"(^|/)tests?/",  # test suites and their data
    r"(^|/)testing/_private/",
    r"(^|/)share/locale/", r"(^|/)locale/[^/]+/LC_MESSAGES/",  # translations (the interface is English)
    r"(^|/)share/(doc|man|info|gtk-doc|bash-completion|zsh|fish|aclocal|pkgconfig|gettext|installed-tests)/",
    r"(^|/)lib/pkgconfig/", r"(^|/)include/",
    r"\.(a|la|h|hpp|pxd|pyx|pyi|c|cpp|cxx|f|f90)$",
    r"(^|/)mpl-data/sample_data/", r"(^|/)mpl-data/images/",  # toolbar images: no matplotlib toolbar
    r"(^|/)mpl-data/fonts/afm/", r"(^|/)mpl-data/fonts/pdfcorefonts/",  # core-font metrics: TTF fonts are used
    r"(^|/)icons/Adwaita/(64x64|96x96|128x128|256x256|512x512)/",  # icon sizes GTK 3 widgets never request
    r"(^|/)icons/Adwaita/cursors/",
    r"(^|/)icons/hicolor/.+/",  # other applications' icons from the build machine (index.theme stays)
    r"(^|/)scipy/misc/", r"(^|/)scipy/datasets/",
    # Linux: every system has libgcc_s; the build machine's copy can need a newer glibc than the user's
    r"^libgcc_s\.so\.1$",
)]


def keep_entry(dest: str) -> bool:
    """False for a bundle entry (destination path) the builds leave out."""
    path = dest.replace("\\", "/")
    return not any(p.search(path) for p in _DROP)


def report(root: str, depth: int = 3, top: int = 25) -> None:
    sizes = {}
    total = 0
    for base, _dirs, files in os.walk(root):
        rel = os.path.relpath(base, root).replace("\\", "/")
        key = "/".join(rel.split("/")[:depth]) if rel != "." else "."
        for f in files:
            path = os.path.join(base, f)
            if os.path.islink(path):  # PyInstaller links some libraries; count each file once
                continue
            try:
                n = os.path.getsize(path)
            except OSError:
                continue
            sizes[key] = sizes.get(key, 0) + n
            total += n
    print(f"{root}: {total / 1e6:.1f} MB")
    for key, n in sorted(sizes.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {n / 1e6:8.1f} MB  {key}")


if __name__ == "__main__":
    report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
