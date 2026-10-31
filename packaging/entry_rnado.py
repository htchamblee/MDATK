"""Entry script of the stand-alone ``rnado-gui`` program (PyInstaller; no console on Windows).

``rnado-gui [PROJECT] [ARGS ...]`` starts RNAdo, the project planner, the same as
``python -m mdatk.gui.rnado``: every argument is passed on (``mdatk.gui.launch.rnado_command`` adds
``--open TOOL``).  The AppImage's AppRun starts it for ``MDATK-*.AppImage rnado ...``.
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
    from mdatk.gui.rnado import main

    sys.exit(main(sys.argv[1:]))
