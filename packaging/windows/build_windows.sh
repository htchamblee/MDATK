#!/usr/bin/env bash
# Build the stand-alone Windows 10/11 (64-bit) MDATK bundle - BETA, untested on real machines.
#
# Run in an MSYS2 UCRT64 shell (https://www.msys2.org) from the repository folder:
#
#   packaging/windows/build_windows.sh [--install-deps] [--out DIR] [--work DIR] [--no-ffmpeg]
#
# --install-deps installs the build dependencies first: Python, GTK 3, PyGObject, NumPy, SciPy,
# Matplotlib and PyInstaller from MSYS2 (pacman); MDAnalysis, barnaba and mdtraj with pip.
#
# Result (default OUT: dist/):
#   MDATK-<version>-windows-x64.zip   the folder MDATK-<version>/ with mdatk-gui.exe (interface),
#                                     mdatk.exe (command line), tools/ffmpeg.exe and the licences
#   MDATK-<version>-windows-x64-setup.exe   Inno Setup installer, when ISCC.exe is found
#                                     (ISCC environment variable, PATH or the default folder)
#
# Environment: FFMPEG_URL (static FFmpeg .zip with bin/ffmpeg.exe), FFMPEG_ARCHIVE (local copy),
# MDATK_NO_BARNABA=1 (build without barnaba).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
ROOT="$(dirname "$PKG")"
PYTHON="${PYTHON:-python}"
OUT="$ROOT/dist"
WORK="$ROOT/build/windows"
WITH_FFMPEG=1
INSTALL_DEPS=0
FFMPEG_BRANCH="${FFMPEG_BRANCH:-8.1}"
FFMPEG_URL="${FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n${FFMPEG_BRANCH}-latest-win64-gpl-${FFMPEG_BRANCH}.zip}"

while [ $# -gt 0 ]; do
    case "$1" in
        --install-deps) INSTALL_DEPS=1; shift ;;
        --out) OUT="$(realpath -m "$2")"; shift 2 ;;
        --work) WORK="$(realpath -m "$2")"; shift 2 ;;
        --no-ffmpeg) WITH_FFMPEG=0; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "${MSYSTEM:-}" != "UCRT64" ] && [ "${MSYSTEM:-}" != "MINGW64" ]; then
    echo "warning: run this in an MSYS2 UCRT64 shell (MSYSTEM is '${MSYSTEM:-unset}')" >&2
fi
P="${MINGW_PACKAGE_PREFIX:-mingw-w64-ucrt-x86_64}"

if [ "$INSTALL_DEPS" = 1 ]; then
    echo "== dependencies (pacman)"
    pacman -S --needed --noconfirm zip unzip \
        "$P-python" "$P-python-pip" "$P-python-setuptools" "$P-python-wheel" "$P-cython" "$P-gcc" \
        "$P-pkgconf" "$P-gtk3" "$P-adwaita-icon-theme" "$P-python-gobject" "$P-python-cairo" \
        "$P-python-numpy" "$P-python-scipy" "$P-python-matplotlib" "$P-python-pillow"
    # compiled helpers MSYS2 packages too (pip installs the pure-Python rest)
    for extra in python-msgpack python-packaging python-pyparsing python-tqdm python-threadpoolctl \
                 python-joblib python-filelock; do
        pacman -S --needed --noconfirm "$P-$extra" || echo "   ($P-$extra not available: pip)"
    done
    pacman -S --needed --noconfirm "$P-pyinstaller" "$P-pyinstaller-hooks-contrib" ||
        "$PYTHON" -m pip install --break-system-packages pyinstaller pyinstaller-hooks-contrib
    echo "== dependencies (pip)"
    export PIP_BREAK_SYSTEM_PACKAGES=1  # MSYS2's Python is "externally managed"
    # no wheels for MSYS2's Python: compiled packages build against the MSYS2 NumPy / Cython.
    # GCC 14+ turns old-C warnings into errors; MDAnalysis' encore/spe.c calls getpid() without
    # declaring it on Windows ("implicit declaration"), so keep those warnings warnings.
    export CFLAGS="${CFLAGS:+$CFLAGS }-Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-pointer-types -Wno-error=int-conversion"
    "$PYTHON" -m pip install --no-build-isolation "MDAnalysis>=2.4"
    if [ -z "${MDATK_NO_BARNABA:-}" ]; then
        if ! "$PYTHON" -m pip install --no-build-isolation mdtraj future ||
           ! "$PYTHON" -m pip install --no-deps barnaba; then
            echo "warning: barnaba / mdtraj could not be installed; building without barnaba" >&2
            export MDATK_NO_BARNABA=1
        fi
    fi
fi

cd "$ROOT"
VERSION="$("$PYTHON" -c 'import mdatk; print(mdatk.__version__)')"
DIST="$WORK/dist/mdatk"
NAME="MDATK-$VERSION-windows-x64"
echo "== MDATK $VERSION, work folder $WORK"
mkdir -p "$WORK" "$OUT"

echo "== icon"
"$PYTHON" "$PKG/make_icon.py" "$WORK/icons"

echo "== PyInstaller"
rm -rf "$WORK/dist" "$WORK/pyinstaller"
MDATK_ICON_DIR="$WORK/icons" "$PYTHON" -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath "$WORK/dist" --workpath "$WORK/pyinstaller" "$PKG/mdatk.spec"
"$PYTHON" "$PKG/trim.py" "$DIST" 1 | head -n 1

FFMPEG_VERSION=""
FFMPEG_DIR=""
FFMPEG_SHA256=""
if [ "$WITH_FFMPEG" = 1 ]; then
    echo "== FFmpeg"
    archive="${FFMPEG_ARCHIVE:-}"
    if [ -z "$archive" ]; then
        archive="$WORK/ffmpeg.zip"
        # a dated autobuild instead of the moving "latest" link, so the licence notice stays valid
        FFMPEG_URL="$("$PYTHON" "$PKG/pin_ffmpeg_url.py" "$FFMPEG_URL")"
        if ! curl -fsSL --retry 3 -o "$archive" "$FFMPEG_URL"; then
            FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
            echo "   n$FFMPEG_BRANCH build not available, using $FFMPEG_URL"
            curl -fsSL --retry 3 -o "$archive" "$FFMPEG_URL"
        fi
    fi
    FFMPEG_DIR="$WORK/ffmpeg"
    rm -rf "$FFMPEG_DIR" && mkdir -p "$FFMPEG_DIR"
    unzip -q "$archive" -d "$FFMPEG_DIR"
    exe="$(find "$FFMPEG_DIR" -type f -path '*/bin/ffmpeg.exe' | head -n 1)"
    [ -n "$exe" ] || { echo "no bin/ffmpeg.exe in $archive" >&2; exit 1; }
    mkdir -p "$DIST/_internal/tools"
    cp "$exe" "$DIST/_internal/tools/ffmpeg.exe"
    FFMPEG_VERSION="$("$DIST/_internal/tools/ffmpeg.exe" -hide_banner -version | head -n 1 | tr -d '\r')"
    echo "   $FFMPEG_VERSION"
    FFMPEG_SHA256="$(sha256sum "$archive" | cut -d' ' -f1)"
    echo "   sha256 $FFMPEG_SHA256"
fi

echo "== licences"
"$PYTHON" "$PKG/collect_licenses.py" --dist "$DIST" --work "$WORK/pyinstaller/mdatk" \
    ${FFMPEG_DIR:+--ffmpeg-dir "$FFMPEG_DIR"} --ffmpeg-version "$FFMPEG_VERSION" \
    --ffmpeg-url "${FFMPEG_ARCHIVE:-$FFMPEG_URL}${FFMPEG_SHA256:+ (sha256 $FFMPEG_SHA256)}" > "$WORK/licenses.log"
tail -n 3 "$WORK/licenses.log"
sed "s/@VERSION@/$VERSION/g" "$HERE/README-Windows.txt" | sed 's/$/\r/' > "$DIST/README.txt"

echo "== zip"
rm -rf "$WORK/stage" && mkdir -p "$WORK/stage"
cp -a "$DIST" "$WORK/stage/MDATK-$VERSION"
rm -f "$OUT/$NAME.zip"
(cd "$WORK/stage" && zip -qr9 "$OUT/$NAME.zip" "MDATK-$VERSION")

echo "== installer"
iscc="${ISCC:-}"
[ -n "$iscc" ] || iscc="$(command -v iscc || command -v ISCC.exe || true)"
for cand in "/c/Program Files (x86)/Inno Setup 6/ISCC.exe" "/c/Program Files/Inno Setup 6/ISCC.exe"; do
    [ -n "$iscc" ] || { [ -f "$cand" ] && iscc="$cand"; } || true
done
if [ -n "$iscc" ]; then
    # MSYS2_ARG_CONV_EXCL: pass the /D options to the Windows program unchanged
    MSYS2_ARG_CONV_EXCL='*' "$iscc" /Q "/DAppVersion=$VERSION" "/DSourceDir=$(cygpath -w "$DIST")" \
        "/DOutputDir=$(cygpath -w "$OUT")" "/DOutputName=$NAME-setup" \
        "/DIconFile=$(cygpath -w "$WORK/icons/mdatk.ico")" "$(cygpath -w "$HERE/mdatk.iss")"
else
    echo "   Inno Setup (ISCC.exe) not found: zip only"
fi

echo "== done"
du -sh "$DIST" | sed 's/^/   bundle (unpacked): /'
ls -l "$OUT"/"$NAME"* | sed 's/^/   /'
