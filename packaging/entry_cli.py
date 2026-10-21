"""Entry script of the stand-alone ``mdatk`` command (PyInstaller)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mdatk.frozen import setup

    setup()
    from mdatk.cli import main

    sys.exit(main())
