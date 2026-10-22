# Stand-alone builds (Linux AppImage, Windows)

This folder builds MDATK as programs that need no Python, conda or virtual environment:

| Build | File | Status |
|---|---|---|
| Linux x86-64 | `MDATK-<version>-x86_64.AppImage` | built on Rocky Linux 9, runs on Ubuntu 22.04 / RHEL 9 and newer |
| Windows 10/11 x64 | `MDATK-<version>-windows-x64.zip`, `MDATK-<version>-windows-x64-setup.exe` | **beta, untested** on real machines (CI smoke test only) |

Both are [PyInstaller](https://pyinstaller.org) "one-folder" bundles of the MDATK source tree
with Python 3.12, GTK 3 + PyGObject, NumPy, SciPy, Matplotlib, MDAnalysis, Barnaba (+ mdtraj)
and a static GPL build of FFmpeg. The GitHub Actions workflow `.github/workflows/release.yml`
builds, tests and publishes them (see [CI](#ci-and-releases)).

## Files

| File | Purpose |
|---|---|
| `mdatk.spec` | PyInstaller spec: one Analysis, the `mdatk` console program (+ `mdatk-gui.exe` without console on Windows) |
| `entry_cli.py`, `entry_gui.py` | entry scripts; call `mdatk.frozen.setup()` first, then `mdatk.cli.main` |
| `trim.py` | what the builds leave out (module excludes, data/library filters); `python packaging/trim.py DIR` prints a size report |
| `make_icon.py` | draws the application icon (PNG + ICO) with Pillow at build time |
| `collect_licenses.py` | writes `licenses/` into the bundle: licence files of every bundled Python distribution, system package copyright files (dpkg / rpm / pacman), FFmpeg notice, `THIRD_PARTY_NOTICES.txt` |
| `notices/FFMPEG.txt` | FFmpeg licence / source notice template |
| `smoke_test.py` | tests a finished build (`--version`, `demo`, `analyze`, `render`, barnaba, GUI start); stdlib only |
| `linux/build_appimage.sh` | the Linux build: PyInstaller → FFmpeg → licences → AppDir → `appimagetool` |
| `linux/AppRun` | AppImage entry point: no argument or a project file → GUI, otherwise the command line; adds `_internal/fallback` (Wayland libraries) to the library path on hosts without them |
| `linux/check_glibc.sh` | newest glibc symbol version the bundle needs (RHEL 9 = 2.34) |
| `windows/build_windows.sh` | the Windows build, in an MSYS2 UCRT64 shell (optionally installs the dependencies) |
| `windows/mdatk.iss` | Inno Setup installer (per-user by default, optional PATH entry) |
| `windows/README-Windows.txt` | `README.txt` of the Windows build |

Frozen-app support inside the package is in `mdatk/frozen.py`: the bundled FFmpeg
(`_internal/tools/ffmpeg[.exe]`) goes in front of `PATH`, programs MDATK starts (generated
scripts, the file manager, cpptraj) get an environment without the bundle's GTK / library
settings, and MDAnalysis' warning about the (deliberately left out) netCDF4 is silenced. The GI
typelib path, GDK pixbuf loaders and GSettings schemas are set up by PyInstaller's own runtime
hooks. On Windows (`mdatk/workflows/runner.py`, `runner_problem()`), MDATK writes workflow
scripts but does not run them; the GUI's *Run* explains that they run on Linux, a cluster or WSL.

## Building locally

### Linux AppImage

Build on the **oldest** distribution the AppImage must support (glibc is not bundled): releases
are built in a `rockylinux:9` container (glibc 2.34) with Rocky's own Python 3.12 and GTK 3. A
build on a newer system works on that system but not on older ones (`check_glibc.sh` tells).

```bash
# Rocky / Alma / RHEL 9 (as root, or with sudo)
dnf install -y dnf-plugins-core && dnf config-manager --set-enabled crb
dnf install -y python3.12 python3.12-devel python3.12-pip gcc pkgconf-pkg-config \
    gobject-introspection-devel cairo-devel cairo-gobject-devel gtk3 librsvg2 \
    adwaita-icon-theme xorg-x11-server-Xvfb xorg-x11-xauth binutils file tar xz
# (Ubuntu / Debian instead: apt install libgirepository1.0-dev libcairo2-dev pkg-config
#  gir1.2-gtk-3.0 librsvg2-common adwaita-icon-theme xvfb binutils)
python3.12 -m venv /tmp/mdatk-build && . /tmp/mdatk-build/bin/activate
pip install "PyGObject<3.51" pycairo pyinstaller        # PyGObject 3.51+ needs GLib 2.80
pip install . && pip uninstall -y mdatk                  # MDATK's dependencies only
pip install mdtraj future && pip install --no-deps barnaba
PYTHON=python packaging/linux/build_appimage.sh          # -> dist/MDATK-<version>-x86_64.AppImage
packaging/linux/check_glibc.sh build/appimage/dist/mdatk 2.34
xvfb-run -a python packaging/smoke_test.py dist/MDATK-*-x86_64.AppImage --render --bundled-ffmpeg --barnaba --gui 15
```

Options: `--work DIR` (default `build/appimage`, holds the unpacked bundle `dist/mdatk`),
`--out DIR` (default `dist`), `--no-ffmpeg`; environment `FFMPEG_ARCHIVE` (use a local FFmpeg
archive), `FFMPEG_URL`, `APPIMAGETOOL`, `APPIMAGETOOL_VERSION` (default 1.9.1),
`MDATK_NO_BARNABA=1`, `MDATK_NO_TRIM=1` (keep everything, to measure the trimming).

### Windows

In an [MSYS2](https://www.msys2.org) **UCRT64** shell, from the repository folder:

```bash
packaging/windows/build_windows.sh --install-deps   # first time: pacman + pip dependencies
packaging/windows/build_windows.sh                  # later builds
```

`--install-deps` installs Python, GTK 3, PyGObject, NumPy, SciPy, Matplotlib and PyInstaller
from MSYS2 and builds MDAnalysis, mdtraj and barnaba with pip (MSYS2's Python has no wheels; if
mdtraj or barnaba fail to build, the bundle is made without barnaba). MSYS2's GCC 14+ treats some
old C as errors that MDAnalysis still contains, so the script relaxes them for these builds
(`CFLAGS=-Wno-error=implicit-function-declaration ...`). The installer is made when
Inno Setup 6 (`ISCC.exe`) is installed. Test from PowerShell or cmd, **not** from the MSYS2 shell
(its `PATH` would hide DLLs missing from the bundle):
`python packaging\smoke_test.py build\windows\dist\mdatk\mdatk.exe --render --bundled-ffmpeg`.

## CI and releases

`.github/workflows/release.yml` runs on a pushed tag `v*` (publishing a GitHub release with a
new tag pushes one), on pushes to any branch that change `packaging/` or the workflow, and on
manual dispatch:

1. **appimage** (`rockylinux:9` container, Rocky's Python 3.12 and GTK 3): build, glibc check
   (fails if anything needs more than 2.34), smoke test including the GUI under Xvfb.
2. **rhel9** and **ubuntu2204**: the same smoke test of the AppImage on a clean `rockylinux:9`
   container (no GTK installed) and on Ubuntu 22.04.
3. **windows** (windows-2022, MSYS2 UCRT64): build zip + installer, smoke test (`--version`,
   `demo`, `analyze`, headless `render`) with only the bundle's DLLs.
4. **release** (tags only): attaches the AppImage, the Windows files and `SHA256SUMS.txt` to the
   GitHub release; the notes mark Windows as *beta, untested*. It runs only when all jobs above
   passed, so a failed build leaves the release without files instead of half-filled: fix the
   build and publish a new tag.

Branch and manual runs only upload the files as workflow artifacts. The actions are pinned to commit SHAs
(versions in the comments); the FFmpeg download is BtbN's current `n8.1` "gpl" build (falls back
to the `master` build when BtbN drops the branch) - its sha256 is printed in the build log.

## What is bundled and what is not

Bundled: MDATK (with the `.py` sources: workflow generators embed some of them), Python 3.12,
GTK 3, PyGObject / pycairo, NumPy, SciPy, Matplotlib (Agg/PDF/PS/SVG back-ends), MDAnalysis,
Barnaba + mdtraj (only what `barnaba.annotate` needs), FFmpeg (`_internal/tools`), the Adwaita
icon theme and the licence texts (`licenses/`). The AppImage also carries the host's Wayland
client libraries in `_internal/fallback`: PyInstaller leaves them out because they must match the
host's graphics drivers, so `AppRun` uses the copies only on hosts that have none (minimal or
server installs, e.g. a cluster node used over `ssh -X`); desktops keep their own.

Left out (`trim.py`): test suites and their data, translations, documentation and headers,
Matplotlib sample data and toolbar images, Adwaita icons above 48 px and cursors, the build
machine's other hicolor icons, Qt / Tk / wx back-ends, optional stacks MDAnalysis and mdtraj
import when present (pandas, scikit-learn, netCDF4 + HDF5, numba, ...), `libgcc_s` (always on the
host). Not bundled on purpose: **cpptraj / AmberTools** (the user's own installation; MDATK
detects it as before), the optional **VTK** renderer, and every program the generated workflow
scripts call (Amber, GROMACS, Rosetta, Phenix, Coot, VMD ...).

Sizes measured on Ubuntu 24.04 (Python 3.12, NumPy 2.5, SciPy 1.18, FFmpeg 8.1 is ~172 MB of
the unpacked bundle):

| | unpacked bundle | AppImage (zstd) |
|---|---|---|
| untrimmed (`MDATK_NO_TRIM=1`) | 584 MB | 199 MB |
| trimmed | 514 MB | 180 MB |

## Licences

MDATK is GPL v3, so bundling GPL components is fine for these builds:

- **Barnaba** (GPL v3) is bundled; mdtraj is LGPL 2.1+.
- **FFmpeg**: BtbN's "gpl" variant (GPL v3: x264, x265 ...). `licenses/ffmpeg/FFMPEG-NOTICE.txt`
  names the exact version and where its corresponding source is (FFmpeg git commit + the
  FFmpeg-Builds scripts that pin every library). If a stricter reading of GPL §6 is wanted, attach
  the FFmpeg source tarball to the release as well.
- **GTK 3, GLib, Pango, PyGObject, pycairo** (LGPL): shipped as separate shared libraries, so
  users can replace them (AppImage: `--appimage-extract`; Windows: the install folder).
- **PyInstaller bootloader**: GPL 2+ with the bootloader exception.
- Everything else: see `licenses/THIRD_PARTY_NOTICES.txt` in a build (generated from the
  distribution metadata, `dpkg` and `pacman`; entries marked "[no licence file found]" need a look).

`LICENSING.md` (plan for a future C++ version) says Barnaba is never bundled and FFmpeg must be
LGPL-only; that applies to the C++ version, not to these GPL v3 Python builds.

## Known limitations

- The Windows build is untested on real machines; MDAnalysis / mdtraj are compiled from source
  with MSYS2's GCC in CI.
- cpptraj cannot run on Windows (AmberTools has no Windows build): clean trajectories on Linux /
  WSL, or use already-cleaned ones.
- AppImages need FUSE 2 or 3 (`fusermount`) to mount; without it run
  `./MDATK-*.AppImage --appimage-extract-and-run` or set `APPIMAGE_EXTRACT_AND_RUN=1`.
- Matplotlib's font cache is rebuilt at every start (PyInstaller keeps it in a temporary
  folder); this costs a second or two.
