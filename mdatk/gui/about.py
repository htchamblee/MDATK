"""Citations for the About dialogs of the analysis window and RNAdo, and their BibTeX export."""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

from ..citations import bibtex, grouped, keys_for  # noqa: E402
from . import widgets as W  # noqa: E402


def citation_text(resources: Sequence[str], what: str = "This project") -> str:
    """The references of ``resources`` (resource ids), grouped by resource."""
    keys = keys_for(resources)
    if not keys:
        return f"{what} uses no external resource with a reference yet."
    lines = [f"Resources {what.lower()} uses ({len(keys)} references; each generated script lists only "
             "its own):", ""]
    for res, refs in grouped(resources):
        if not refs:
            continue
        lines.append(res.name + (f" - {res.note}" if res.note else ""))
        lines += [f"    {r.text}" for r in refs]
        lines.append("")
    return "\n".join(lines)


def export_bibtex(parent: Optional[Gtk.Window], resources: Sequence[str], folder: str, name: str,
                  log: Callable[[str], None], info: Callable[[str, str], None],
                  error: Callable[[str, str], None]) -> Optional[str]:
    """Ask for a file and write the BibTeX entries of ``resources``; returns the file written."""
    keys = keys_for(resources)
    if not keys:
        info("Nothing to cite yet", "The project uses no external resource with a reference.")
        return None
    dlg = Gtk.FileChooserDialog(title="Export citations (BibTeX)", transient_for=parent,
                                action=Gtk.FileChooserAction.SAVE)
    dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dlg.set_do_overwrite_confirmation(True)
    W.add_filters(dlg, [("BibTeX files", ["*.bib"])])
    if folder and os.path.isdir(folder):
        dlg.set_current_folder(folder)
    dlg.set_current_name(f"{name}_citations.bib")
    path = dlg.get_filename() if dlg.run() == Gtk.ResponseType.OK else None
    dlg.destroy()
    if not path:
        return None
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(bibtex(keys))
    except OSError as exc:
        error("Could not write the BibTeX file", str(exc))
        return None
    log(f"Wrote {len(keys)} references to {path}")
    return path
