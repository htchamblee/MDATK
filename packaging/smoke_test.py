"""Smoke test of a stand-alone MDATK build (AppImage, bundle folder or Windows mdatk.exe).

    python packaging/smoke_test.py EXE [--work DIR] [--render] [--barnaba] [--gui SECONDS]
                                       [--bundled-ffmpeg] [--version X.Y.Z]

EXE is the AppImage, ``dist/mdatk/mdatk`` or ``dist\\mdatk\\mdatk.exe``.  Runs, in order:

* ``EXE --version``
* ``EXE demo DIR --no-solvent`` (a small synthetic system) and ``EXE analyze`` on it
* ``--render``: ``EXE render`` (video + charts + PDBs, headless) and checks the video;
  with ``--bundled-ffmpeg`` no other ffmpeg is reachable on PATH, so the bundled one must work
* ``--barnaba``: ``EXE build helix`` + ``EXE pdb annotate --engine barnaba`` (bundled barnaba)
  and ``EXE workflow generate amber-build --dry-run`` (embedded module sources)
* ``--gui N``: ``EXE gui PROJECT`` must still be running after N seconds without errors
  (run under ``xvfb-run -a`` on a headless Linux machine)

Uses the standard library only; exits with 1 when a check fails.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

FAILED: List[str] = []


def _env(no_ffmpeg: bool) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # the build must not need the build machine's packages
    env.setdefault("APPIMAGE_EXTRACT_AND_RUN", "1")  # AppImage without FUSE (containers, CI)
    if no_ffmpeg:
        exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        dirs = [d for d in env.get("PATH", "").split(os.pathsep) if d]
        keep = [d for d in dirs if not os.path.isfile(os.path.join(d, exe))]
        if os.name != "nt" and len(keep) < len(dirs):
            # the AppImage's AppRun needs a few utilities that live next to the system ffmpeg
            tools = tempfile.mkdtemp(prefix="mdatk-smoke-path-")
            for name in ("sh", "readlink", "dirname", "basename"):
                found = shutil.which(name)
                if found:
                    os.symlink(found, os.path.join(tools, name))
            keep.append(tools)
        env["PATH"] = os.pathsep.join(keep)
    return env


def run(label: str, cmd: List[str], env: Dict[str, str], timeout: float = 900,
        expect: Optional[str] = None) -> Optional[str]:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        FAILED.append(label)
        print(f"FAIL {label}: {exc}")
        return None
    out = proc.stdout + proc.stderr
    ok = proc.returncode == 0 and (expect is None or expect in out)
    print(f"{'ok  ' if ok else 'FAIL'} {label} ({time.time() - t0:.1f} s)")
    if not ok:
        FAILED.append(label)
        print("     " + " ".join(cmd))
        print("     " + out[-3000:].replace("\n", "\n     "))
    return out if ok else None


def gui(exe: str, project: str, seconds: float, env: Dict[str, str]) -> None:
    label = f"gui runs for {seconds:g} s"
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen([exe, "gui", project], stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            proc.wait(seconds)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.seek(0)
        out = log.read().decode("utf-8", errors="replace")
    bad = [line for line in out.splitlines()
           if "Traceback" in line or "CRITICAL" in line or line.startswith(("ERROR", "error:"))]
    if exited or bad:
        FAILED.append(label)
        print(f"FAIL {label}: " + ("exited early" if exited else "errors in the output"))
        print("     " + out[-3000:].replace("\n", "\n     "))
    else:
        print(f"ok   {label}" + (f" (output: {len(out.splitlines())} line(s))" if out.strip() else ""))
        if out.strip():
            print("     " + out[-1500:].replace("\n", "\n     "))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("exe")
    ap.add_argument("--work", help="folder for the test files (default: a temporary folder)")
    ap.add_argument("--version", dest="want_version", help="expected version string")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--bundled-ffmpeg", action="store_true")
    ap.add_argument("--barnaba", action="store_true")
    ap.add_argument("--gui", type=float, default=0.0, metavar="SECONDS")
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args(argv)

    exe = os.path.abspath(args.exe)
    work = os.path.abspath(args.work or tempfile.mkdtemp(prefix="mdatk-smoke-"))
    os.makedirs(work, exist_ok=True)
    env = _env(args.bundled_ffmpeg)
    print(f"smoke test of {exe} in {work}")

    run("--version", [exe, "--version"], env, 120, expect=args.want_version or "mdatk")
    demo = os.path.join(work, "demo")
    shutil.rmtree(demo, ignore_errors=True)
    run("demo", [exe, "demo", demo, "--n-frames", str(args.frames), "--no-solvent"], env)
    project = os.path.join(demo, "demo.mdatk.toml")
    if os.path.isfile(project):
        run("analyze", [exe, "analyze", "-p", project, "-q"], env)
        if args.render:
            run("render (headless)", [exe, "render", "-p", project, "-q"], env)
            videos = glob.glob(os.path.join(demo, "**", "*.mp4"), recursive=True)
            if videos and os.path.getsize(videos[0]) > 1000:
                print(f"ok   video {os.path.basename(videos[0])} ({os.path.getsize(videos[0]) / 1e3:.0f} kB)")
            else:
                FAILED.append("video")
                print("FAIL video: no .mp4 written")
    if args.barnaba:
        helix = os.path.join(work, "helix.pdb")
        run("build helix", [exe, "build", "helix", "--seq", "GGGCGAAAGCCC", "--partner", "GGGCUUUCGCCC",
                            "-o", helix], env)
        run("pdb annotate (barnaba)", [exe, "pdb", "annotate", "--engine", "barnaba", "-o",
                                       os.path.join(work, "helix_annotated.pdb"), helix], env,
            expect="barnaba")
        run("workflow generate (embedded sources)", [exe, "workflow", "generate", "amber-build", "--dry-run",
                                                     "--out-dir", os.path.join(work, "wf"),
                                                     "-i", f"structure={helix}"], env)
    if args.gui and os.path.isfile(project):
        gui(exe, project, args.gui, env)

    print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
