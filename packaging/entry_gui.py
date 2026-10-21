"""Entry script of the stand-alone ``mdatk-gui`` program (PyInstaller; no console on Windows).

``mdatk-gui [PROJECT]`` is the same as ``mdatk gui [PROJECT]``.
"""

import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # A windowed Windows program has no console: give print() and logging somewhere to write.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    from mdatk.frozen import setup

    setup()
    from mdatk.cli import main

    sys.exit(main(["gui", *sys.argv[1:]]))
