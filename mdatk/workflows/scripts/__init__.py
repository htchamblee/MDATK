"""Self-contained helper programs embedded in generated workflows.

Each module here uses only the Python standard library so that MDATK can copy
its source into a generated script (followed by a few constants and a
``__main__`` call) that runs anywhere ``python3`` exists - no MDATK, ParmEd or
NumPy needed on the machine that runs the simulation.
"""

from __future__ import annotations

import inspect
from typing import Dict, Iterable


def embed(module, constants: Dict[str, object], call: str, doc: str, imports: Iterable[str] = ()) -> str:
    """Source of a standalone script: shebang + docstring + module body + constants + main call."""
    body = inspect.getsource(module)
    # drop the module docstring and __future__ import; the generated script gets its own header
    lines = body.splitlines()
    k = 0
    if lines and lines[0].startswith('"""'):
        k = 1
        if not (lines[0].count('"""') >= 2 and len(lines[0]) > 3):
            while k < len(lines) and '"""' not in lines[k]:
                k += 1
            k += 1
    rest = [ln for ln in lines[k:] if not ln.startswith("from __future__")]
    while rest and not rest[0].strip():
        rest.pop(0)
    out = ["#!/usr/bin/env python3", f'"""{doc.strip()}', '"""', ""]
    out += list(imports)
    out += rest
    out += ["", "", "# --- settings written by MDATK ---"]
    for key, value in constants.items():
        out.append(f"{key} = {value!r}")
    out += ["", "", 'if __name__ == "__main__":', f"    sys.exit({call})", ""]
    return "\n".join(out)
