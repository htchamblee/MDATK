#!/usr/bin/env bash
# Build the stand-alone Linux AppImage of MDATK.
#
#   packaging/linux/build_appimage.sh [--out DIR] [--work DIR] [--no-ffmpeg]
#
# Steps: application icons (MDATK, RNAdo) -> PyInstaller bundle (packaging/mdatk.spec: mdatk,
# rnado-gui) -> Wayland client libraries into _internal/fallback (used only where the host has
# none) -> static FFmpeg into _internal/tools -> licence notices -> AppDir (AppRun, desktop files,
# icons) -> appimagetool. "MDATK-*.AppImage rnado [PROJECT]" starts RNAdo (see AppRun).
# Result: OUT/MDATK-<version>-x86_64.AppImage (default OUT: dist/) plus the unpacked bundle in
# WORK/dist/mdatk (default WORK: build/appimage/).
#
# Requirements: the Python that runs MDATK with its dependencies, PyGObject + GTK 3 (typelibs:
# gir1.2-gtk-3.0) and PyInstaller; barnaba + mdtraj (unless MDATK_NO_BARNABA=1); curl, tar, xz.
# Build on the oldest distribution the AppImage must run on (glibc): Rocky Linux 9 for releases.
#
# Environment:
#   PYTHON          interpreter to use (default: python3)
#   FFMPEG_URL      static FFmpeg archive (.tar.xz with bin/ffmpeg); default: BtbN "gpl" build
#   FFMPEG_ARCHIVE  local copy of that archive instead of downloading it
#   APPIMAGETOOL    appimagetool to use instead of downloading release APPIMAGETOOL_VERSION (1.9.1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
ROOT="$(dirname "$PKG")"
PYTHON="${PYTHON:-python3}"
OUT="$ROOT/dist"
WORK="$ROOT/build/appimage"
WITH_FFMPEG=1
FFMPEG_BRANCH="${FFMPEG_BRANCH:-8.1}"
FFMPEG_URL="${FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n${FFMPEG_BRANCH}-latest-linux64-gpl-${FFMPEG_BRANCH}.tar.xz}"
APPIMAGETOOL_VERSION="${APPIMAGETOOL_VERSION:-1.9.1}"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-x86_64.AppImage"

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$(realpath -m "$2")"; shift 2 ;;
        --work) WORK="$(realpath -m "$2")"; shift 2 ;;
        --no-ffmpeg) WITH_FFMPEG=0; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

cd "$ROOT"
VERSION="$("$PYTHON" -c 'import mdatk; print(mdatk.__version__)')"
ARCH="$(uname -m)"
DIST="$WORK/dist/mdatk"
echo "== MDATK $VERSION ($ARCH), work folder $WORK"
mkdir -p "$WORK" "$OUT"

echo "== icons"
"$PYTHON" "$PKG/make_icon.py" "$WORK/icons"

echo "== PyInstaller"
rm -rf "$WORK/dist" "$WORK/pyinstaller"
MDATK_ICON_DIR="$WORK/icons" "$PYTHON" -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath "$WORK/dist" --workpath "$WORK/pyinstaller" "$PKG/mdatk.spec"
"$PYTHON" "$PKG/trim.py" "$DIST" 1 | head -n 1

echo "== fallback libraries"
# PyInstaller never bundles the Wayland client libraries (they belong to the host's graphics
# stack); AppRun puts these copies on the library path only on hosts that lack them.
FALLBACK_SOURCES=()
gdk="$(find "$DIST/_internal" -maxdepth 1 -name 'libgdk-3.so*' | head -n 1)"
if [ -n "$gdk" ]; then
    mkdir -p "$DIST/_internal/fallback"
    while read -r name path; do
        cp -L "$path" "$DIST/_internal/fallback/$name"
        FALLBACK_SOURCES+=("$path")
        echo "   $name ($path)"
    done < <(ldd "$gdk" | awk '$1 ~ /^libwayland-/ && $3 ~ /^\// {print $1, $3}')
fi

FFMPEG_VERSION=""
FFMPEG_DIR=""
FFMPEG_SHA256=""
if [ "$WITH_FFMPEG" = 1 ]; then
    echo "== FFmpeg"
    archive="${FFMPEG_ARCHIVE:-}"
    if [ -z "$archive" ]; then
        archive="$WORK/ffmpeg.tar.xz"
        # a dated autobuild instead of the moving "latest" link, so the licence notice stays valid
        FFMPEG_URL="$("$PYTHON" "$PKG/pin_ffmpeg_url.py" "$FFMPEG_URL")"
        if ! curl -fsSL --retry 3 -o "$archive" "$FFMPEG_URL"; then
            # BtbN keeps only the newest release branches: fall back to the master build
            FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
            echo "   n$FFMPEG_BRANCH build not available, using $FFMPEG_URL"
            curl -fsSL --retry 3 -o "$archive" "$FFMPEG_URL"
        fi
    fi
    FFMPEG_DIR="$WORK/ffmpeg"
    rm -rf "$FFMPEG_DIR" && mkdir -p "$FFMPEG_DIR"
    tar -C "$FFMPEG_DIR" -xf "$archive"
    exe="$(find "$FFMPEG_DIR" -type f -path '*/bin/ffmpeg' | head -n 1)"
    [ -n "$exe" ] || { echo "no bin/ffmpeg in $archive" >&2; exit 1; }
    mkdir -p "$DIST/_internal/tools"
    install -m 755 "$exe" "$DIST/_internal/tools/ffmpeg"
    FFMPEG_VERSION="$("$DIST/_internal/tools/ffmpeg" -hide_banner -version | head -n 1)"
    echo "   $FFMPEG_VERSION"
    FFMPEG_SHA256="$(sha256sum "$archive" | cut -d' ' -f1)"
    echo "   sha256 $FFMPEG_SHA256"
fi

echo "== licences"
"$PYTHON" "$PKG/collect_licenses.py" --dist "$DIST" --work "$WORK/pyinstaller/mdatk" \
    ${FFMPEG_DIR:+--ffmpeg-dir "$FFMPEG_DIR"} --ffmpeg-version "$FFMPEG_VERSION" \
    ${FALLBACK_SOURCES[@]+"${FALLBACK_SOURCES[@]/#/--extra-binary=}"} \
    --ffmpeg-url "${FFMPEG_ARCHIVE:-$FFMPEG_URL}${FFMPEG_SHA256:+ (sha256 $FFMPEG_SHA256)}" > "$WORK/licenses.log"
tail -n 3 "$WORK/licenses.log"

echo "== AppDir"
APPDIR="$WORK/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -a "$DIST" "$APPDIR/usr/lib/mdatk"
install -m 755 "$HERE/AppRun" "$APPDIR/AppRun"
# The desktop file of the source tree, with the bundled icon. Its "RNAdo" action becomes
# "mdatk-gui rnado %f": desktop integration (AppImageLauncher, appimaged) replaces the program of every
# Exec line with the AppImage, and AppRun starts "mdatk rnado" for a first argument "rnado". The
# X-AppImage-Version line goes into the main group, before the action group.
sed -e '/^Icon=rnado$/!s/^Icon=.*/Icon=mdatk/' -e 's/^Exec=mdatk-gui .*/Exec=mdatk-gui %f/' \
    -e 's/^Exec=rnado-gui /Exec=mdatk-gui rnado /' \
    -e "/^Actions=/i X-AppImage-Version=$VERSION" "$ROOT/data/mdatk.desktop" \
    > "$APPDIR/usr/share/applications/mdatk.desktop"
grep -q '^X-AppImage-Version=' "$APPDIR/usr/share/applications/mdatk.desktop" ||
    sed -i "/^\[Desktop Entry\]/a X-AppImage-Version=$VERSION" "$APPDIR/usr/share/applications/mdatk.desktop"
cp "$APPDIR/usr/share/applications/mdatk.desktop" "$APPDIR/mdatk.desktop"
cp "$WORK/icons/mdatk.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/mdatk.png"
cp "$WORK/icons/mdatk.png" "$APPDIR/mdatk.png"
ln -sf mdatk.png "$APPDIR/.DirIcon"
# RNAdo: its icon in every size (the action's Icon=rnado) and its own desktop file for reference;
# only the root desktop file (MDATK) is integrated, so RNAdo is started through the action,
# "MDATK-*.AppImage rnado [PROJECT]" or a symbolic link named "rnado" to the AppImage (see AppRun)
for png in "$ROOT"/mdatk/assets/rnado-icon-*.png; do
    size="$(basename "$png" .png)"
    size="${size#rnado-icon-}"
    install -D -m 644 "$png" "$APPDIR/usr/share/icons/hicolor/${size}x${size}/apps/rnado.png"
done
install -D -m 644 "$ROOT/mdatk/assets/rnado-icon.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/rnado.svg"
sed -e 's/^Exec=rnado-gui /Exec=mdatk-gui rnado /' -e '/^#/d' "$ROOT/data/rnado.desktop" \
    > "$APPDIR/usr/share/applications/rnado.desktop"

echo "== appimagetool"
tool="${APPIMAGETOOL:-}"
if [ -z "$tool" ]; then
    tool="$WORK/appimagetool-$APPIMAGETOOL_VERSION-$ARCH.AppImage"
    [ -x "$tool" ] || { curl -fsSL --retry 3 -o "$tool" "$APPIMAGETOOL_URL"; chmod +x "$tool"; }
fi
target="$OUT/MDATK-$VERSION-$ARCH.AppImage"
rm -f "$target"
# no FUSE in containers / CI runners: let the tool unpack itself
APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" "$tool" --no-appstream "$APPDIR" "$target"
chmod +x "$target"

echo "== done"
du -sh "$DIST" | sed 's/^/   bundle (unpacked): /'
du -sh "$target" | sed 's/^/   AppImage:          /'
