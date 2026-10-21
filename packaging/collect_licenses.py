"""Collect the licence texts of everything a stand-alone build bundles.

    python packaging/collect_licenses.py --dist DIST/mdatk --work WORK/mdatk \\
        [--ffmpeg-dir EXTRACTED_FFMPEG] [--ffmpeg-version "n8.1.3 ..."] [--ffmpeg-url URL]

Writes ``DIST/mdatk/licenses/``:

* ``LICENSE-MDATK.txt`` (GNU GPL v3; also copied to ``DIST/mdatk/LICENSE.txt``)
* ``python/<distribution>-<version>/``: the licence files of every bundled Python
  distribution (found through PyInstaller's module lists in the work folder)
* ``system/<package>/``: copyright files of bundled system libraries (Debian /
  Ubuntu: ``dpkg -S`` and ``/usr/share/doc/<pkg>/copyright``; MSYS2: ``pacman -Qo``
  and ``share/licenses/<pkg>``)
* ``ffmpeg/``: FFmpeg's licence and where its source code is
* ``THIRD_PARTY_NOTICES.txt``: one line per component with its licence

Nothing here is fatal: a component whose licence cannot be found is listed as
such in the notices, so the build can still be inspected and fixed by hand.
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from typing import Dict, Iterable, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LICENSE_NAME = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE|AUTHORS|COPYRIGHT)[^/]*$", re.IGNORECASE)


def _read_toc(path: str) -> List[Tuple[str, str, str]]:
    """Entries (dest, source, type) of a PyInstaller TOC file (a Python literal)."""
    try:
        data = ast.literal_eval(open(path, encoding="utf-8").read())
    except (OSError, ValueError, SyntaxError):
        return []
    out: List[Tuple[str, str, str]] = []

    def walk(obj):
        if isinstance(obj, (list, tuple)):
            if len(obj) == 3 and all(isinstance(x, str) for x in obj):
                out.append(tuple(obj))  # type: ignore[arg-type]
            else:
                for x in obj:
                    walk(x)

    walk(data)
    return out


def _site_dirs() -> List[str]:
    dirs = {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}
    dirs.update(p for p in sys.path if p.endswith(("site-packages", "dist-packages")) or os.path.isdir(p))
    return sorted({os.path.abspath(d) for d in dirs if d and os.path.isdir(d)}, key=len, reverse=True)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _top_level_sources(work: str) -> Dict[str, str]:
    """Top-level module/package name -> one source file, from the PYZ and COLLECT lists."""
    out: Dict[str, str] = {}
    for toc in glob.glob(os.path.join(work, "PYZ-*.toc")):
        for name, src, _typ in _read_toc(toc):
            out.setdefault(name.split(".")[0], src)
    for toc in glob.glob(os.path.join(work, "COLLECT-*.toc")):
        for dest, src, typ in _read_toc(toc):
            if typ == "EXTENSION":
                top = dest.replace("\\", "/").split("/")[0]
                out.setdefault(top.split(".")[0], src)
    return out


def _distribution_for(top: str, src: str, by_top: Dict[str, List[str]]) -> Optional[md.Distribution]:
    for name in by_top.get(top, []):
        try:
            return md.distribution(name)
        except md.PackageNotFoundError:
            pass
    src = os.path.abspath(src)
    for site in _site_dirs():
        if src.startswith(site + os.sep):
            for dist in md.distributions(path=[site]):
                tops = (dist.read_text("top_level.txt") or "").split()
                if top in tops or _norm(dist.metadata["Name"] or "") == _norm(top):
                    return dist
    return None


def _licence_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    text = (meta.get("License") or "").strip()
    return text.splitlines()[0][:80] if text else "see licence files"


def _licence_files(dist: md.Distribution) -> List[str]:
    found: List[str] = []
    base = getattr(dist, "_path", None)  # the .dist-info / .egg-info folder
    for rel in dist.metadata.get_all("License-File") or []:
        for cand in ((os.path.join(str(base), "licenses", rel), os.path.join(str(base), rel)) if base else ()):
            if os.path.isfile(cand):
                found.append(cand)
                break
    if not found:
        for f in dist.files or []:
            if LICENSE_NAME.search(str(f).replace("\\", "/")):
                path = str(dist.locate_file(f))
                if os.path.isfile(path):
                    found.append(path)
    return sorted(set(found))


def _copy(files: Iterable[str], dest: str) -> int:
    n = 0
    for path in files:
        os.makedirs(dest, exist_ok=True)
        name = os.path.basename(path)
        target = os.path.join(dest, name)
        k = 1
        while os.path.exists(target):  # several files called LICENSE
            k += 1
            target = os.path.join(dest, f"{k}-{name}")
        shutil.copyfile(path, target)
        n += 1
    return n


# --------------------------------------------------------------------------- system packages
def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _dpkg_owner(path: str) -> Optional[str]:
    real = os.path.realpath(path)
    unmerged = [re.sub(r"^/usr(?=/lib)", "", p) for p in (real, path)]  # /lib vs /usr/lib (merged /usr)
    for cand in dict.fromkeys((real, path, *unmerged)):
        out = _run(["dpkg", "-S", cand]).strip()
        if out and ":" in out:
            return out.split(":")[0].split(",")[0].strip()
    return None


def _pacman_owner(path: str) -> Optional[str]:
    unix = _run(["cygpath", "-u", path]).strip() or path
    out = _run(["pacman", "-Qqo", unix]).strip()
    return out.splitlines()[0] if out else None


def _owner_function():
    """Function file -> owning system package (MSYS2 pacman or Debian dpkg), or None."""
    return _pacman_owner if shutil.which("pacman") else _dpkg_owner if shutil.which("dpkg") else None


def _system_packages(work: str, sites: List[str]) -> Dict[str, Set[str]]:
    """Package -> bundled file names, for bundled files that came from outside the Python site folders."""
    owner = _owner_function()
    pkgs: Dict[str, Set[str]] = {}
    if owner is None:
        return pkgs
    cache: Dict[str, Optional[str]] = {}
    for toc in glob.glob(os.path.join(work, "COLLECT-*.toc")):
        for dest, src, typ in _read_toc(toc):
            if typ not in ("BINARY", "EXTENSION", "DATA") or not os.path.isabs(src):
                continue
            if any(os.path.abspath(src).startswith(s + os.sep) for s in sites):
                continue  # a Python distribution's own file (licensed with it)
            folder = os.path.dirname(src)
            key = src if typ != "DATA" else folder  # data files: one lookup per folder
            if key not in cache:
                cache[key] = owner(src)
            if cache[key]:
                pkgs.setdefault(cache[key], set()).add(dest)
    return pkgs


def _system_licence(pkg: str, dest: str) -> Tuple[str, int]:
    if shutil.which("pacman"):
        info = _run(["pacman", "-Qi", pkg])
        m = re.search(r"^Licenses\s*:\s*(.+)$", info, re.MULTILINE)
        prefix = _run(["cygpath", "-m", "/"]).strip() or "/"
        n = 0
        for root in ("ucrt64", "mingw64", "clang64", "usr"):
            folder = os.path.join(prefix, root, "share", "licenses", re.sub(r"^mingw-w64-[a-z0-9_]+-", "", pkg))
            if os.path.isdir(folder):
                n += _copy([os.path.join(folder, f) for f in os.listdir(folder)
                            if os.path.isfile(os.path.join(folder, f))], dest)
        return (m.group(1).strip() if m else "see package"), n
    copyright_file = f"/usr/share/doc/{pkg.split(':')[0]}/copyright"
    if os.path.isfile(copyright_file):
        _copy([copyright_file], dest)
        text = open(copyright_file, encoding="utf-8", errors="replace").read()
        names = sorted(set(re.findall(r"^License:\s*(\S+)", text, re.MULTILINE)))
        return (", ".join(names[:6]) if names else "see copyright file"), 1
    return "not found", 0


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", required=True, help="the bundle folder (DIST/mdatk)")
    ap.add_argument("--work", required=True, help="PyInstaller's work folder of the spec (WORK/mdatk)")
    ap.add_argument("--ffmpeg-dir", help="extracted FFmpeg build (for its LICENSE file)")
    ap.add_argument("--ffmpeg-version", default="", help="first line of 'ffmpeg -version'")
    ap.add_argument("--ffmpeg-url", default="", help="where the FFmpeg build was downloaded from")
    args = ap.parse_args(argv)

    out = os.path.join(args.dist, "licenses")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    lines = ["Third-party components bundled with this MDATK build", "=" * 52, "",
             "MDATK itself is free software under the GNU General Public License v3 (LICENSE.txt).",
             "Its source code: https://github.com/htchamblee/MDATK", "",
             "Python distributions (licence files in python/):"]
    shutil.copyfile(os.path.join(ROOT, "LICENSE"), os.path.join(out, "LICENSE-MDATK.txt"))
    shutil.copyfile(os.path.join(ROOT, "LICENSE"), os.path.join(args.dist, "LICENSE.txt"))

    by_top = md.packages_distributions()
    seen: Dict[str, md.Distribution] = {}
    unknown: List[Tuple[str, str]] = []
    for top, src in sorted(_top_level_sources(args.work).items()):
        if top == "mdatk" or top.startswith(("_pyi", "pyimod")) or not src or not os.path.isabs(src):
            continue
        stdlib = sysconfig.get_paths()["stdlib"]
        if os.path.abspath(src).startswith(os.path.abspath(stdlib) + os.sep) and "-packages" not in src:
            continue  # the standard library (Python's own licence below)
        dist = _distribution_for(top, src, by_top)
        if dist is None:
            unknown.append((top, src))
            continue
        seen.setdefault(_norm(dist.metadata["Name"]), dist)
    for key, dist in sorted(seen.items()):
        name, version = dist.metadata["Name"], dist.version
        n = _copy(_licence_files(dist), os.path.join(out, "python", f"{name}-{version}"))
        lines.append(f"  {name} {version}: {_licence_of(dist)}" + ("" if n else "  [no licence file found]"))
    owner = _owner_function()
    system_modules: Dict[str, Set[str]] = {}
    for top, src in unknown:  # e.g. PyGObject / pycairo from the distribution's packages
        pkg = owner(src) if owner else None
        if pkg:
            system_modules.setdefault(pkg, set()).add(top)
        else:
            lines.append(f"  {top}: no distribution metadata found  [no licence file found]")

    pylicence = os.path.join(sysconfig.get_paths()["stdlib"], "LICENSE.txt")
    if os.path.isfile(pylicence):
        shutil.copyfile(pylicence, os.path.join(out, "LICENSE-Python.txt"))
    lines += ["", f"Python {sys.version.split()[0]} (interpreter and standard library): "
                  "Python Software Foundation License (LICENSE-Python.txt)",
              "PyInstaller bootloader (the mdatk program launcher): GPL-2.0-or-later with the "
              "PyInstaller bootloader exception (https://github.com/pyinstaller/pyinstaller)", ""]

    sites = _site_dirs()
    system = _system_packages(args.work, sites)
    for pkg, tops in system_modules.items():
        system.setdefault(pkg, set()).update(tops)
    if system:
        lines.append("System packages (GTK 3, PyGObject and their dependencies; copyright files in system/):")
        for pkg in sorted(system):
            lic, n = _system_licence(pkg, os.path.join(out, "system", pkg))
            lines.append(f"  {pkg}: {lic}" + ("" if n else "  [no copyright file found]"))
        lines.append("")

    if args.ffmpeg_dir:
        fdir = os.path.join(out, "ffmpeg")
        os.makedirs(fdir, exist_ok=True)
        _copy([p for p in glob.glob(os.path.join(args.ffmpeg_dir, "**", "*"), recursive=True)
               if os.path.isfile(p) and LICENSE_NAME.search(p.replace("\\", "/"))], fdir)
        notice = open(os.path.join(HERE, "notices", "FFMPEG.txt"), encoding="utf-8").read()
        notice = notice.replace("@FFMPEG_VERSION@", args.ffmpeg_version or "unknown")
        notice = notice.replace("@FFMPEG_URL@", args.ffmpeg_url or "unknown")
        open(os.path.join(fdir, "FFMPEG-NOTICE.txt"), "w", encoding="utf-8").write(notice)
        lines += [f"FFmpeg ({args.ffmpeg_version or 'version unknown'}): GNU GPL v3 build "
                  "(_internal/tools; licence and source code: ffmpeg/FFMPEG-NOTICE.txt)", ""]

    lines += ["External programs MDATK writes scripts for or calls (cpptraj / AmberTools, GROMACS,",
              "Rosetta, Phenix, Coot, CCP4, VMD ...) are NOT included and keep their own licences."]
    open(os.path.join(out, "THIRD_PARTY_NOTICES.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
