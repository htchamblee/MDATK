"""Planner steps for real-space refinement: Phenix (1.21 / 2.x) and Coot (0.9 / 1.x).

* ``phenix-rsr`` writes a shell script running ``phenix.real_space_refine`` on a model and a cryo-EM map
  (resolution, macro cycles, run strategy, nucleic-acid secondary-structure restraints) plus, on request,
  a parameter (.eff) file with explicit base-pair restraints from a dot-bracket or from the REMARK 950
  annotation of the model (``mdatk pdb annotate``).  The refined model is found by globbing Phenix's
  output names and copied to the step's own file.
* ``coot-rsr`` writes one Python script that runs in Coot 0.9 and Coot 1.x without graphics: every Coot
  function is looked up through a small shim (the ``coot`` module, then the global functions of 0.9), and
  each chain is refined in overlapping windows of residues.  Interactive refinement is left to the user.

L-RNA models need restraint dictionaries for the 5'/3' variants 0A5/0A3 ... (planner step
``restraint-dicts``); the checks name missing codes and residue names the programs will not recognise.
Everything that may differ between program versions is kept in :data:`PHENIX_VERSIONS`,
:data:`COOT_VERSIONS` and :data:`COOT_API`.
"""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .. import PROJECT_ABBREV, __version__, pdbtools
from ..selections import SelectionError, parse_residue_ranges
from . import rna
from .base import (PortValue, StepContext, Tool, env_section, extra_flags, fortran_float, require_file,
                   script_header, shell_quote)
from .registry import register
from .schema import GeneratedFile, Issue, Option, Port, Section

# ---------------------------------------------------------------------------
# version-dependent details
# ---------------------------------------------------------------------------

# Phenix: program, environment script and the PHIL parameter paths written by phenix-rsr.  The command-line
# parameters resolution / run / macro_cycles / nproc / output.prefix are the well-established
# real_space_refine ones; the entries marked "unverified" are listed in the 0.0.3c notes.  No difference
# between 1.21 and 2.x is known to MDATK; the table keeps one place to change if one turns up.
_PHENIX_COMMON: Dict[str, str] = {
    "program": "phenix.real_space_refine",
    "env_script": "phenix_env.sh",  # in the top folder of the installation
    "resolution": "resolution",
    "run": "run",  # run=minimization_global+adp ...
    "macro_cycles": "macro_cycles",
    "nproc": "nproc",
    "random_seed": "random_seed",  # unverified: only written when a seed is set
    "prefix": "output.prefix",  # output: <prefix>_real_space_refined_NNN.pdb / .cif
    # nucleic-acid secondary structure (unverified scope path): enabled=True turns on the automatic search
    # for base pairs and stacking; explicit pairs go into base_pair / stacking_pair scopes below it
    "ss_scope": "pdb_interpretation.secondary_structure",
    "ss_enabled": "enabled",
    "na_scope": "nucleic_acid",
    "base_pair": "base_pair",
    "stacking_pair": "stacking_pair",
    "saenger_class": "saenger_class",  # unverified field name; 19 G-C, 20 A-U, 28 G-U wobble
    "cif_as_pdb": "phenix.cif_as_pdb",  # unverified: only used when Phenix wrote no PDB file
}
PHENIX_VERSIONS: Dict[str, Dict[str, str]] = {
    "1.21": dict(_PHENIX_COMMON, label="Phenix 1.21", example="/usr/local/phenix-1.21.2-5419"),
    "2.x": dict(_PHENIX_COMMON, label="Phenix 2.x", example="/usr/local/phenix-2.0-5936"),
}
# real_space_refine 'run' strategies offered as check boxes: (option key, PHIL value, label, default, tooltip)
RUN_STRATEGIES: Sequence[Tuple[str, str, str, bool, str]] = (
    ("run_minimization_global", "minimization_global", "Global minimisation", True,
     "Gradient-driven minimisation of the whole model against the map with geometry restraints."),
    ("run_rigid_body", "rigid_body", "Rigid body", False,
     "Rigid-body refinement of the chains first (a model that is not yet placed well)."),
    ("run_local_grid_search", "local_grid_search", "Local grid search", True,
     "Local real-space fitting of side chains / residues by grid search."),
    ("run_morphing", "morphing", "Morphing", False,
     "Morphs the model towards the map: large, smooth moves of models from another state."),
    ("run_simulated_annealing", "simulated_annealing", "Simulated annealing", False,
     "Real-space simulated annealing: larger radius of convergence, slower."),
    ("run_adp", "adp", "ADP (B-factors)", True, "Refines atomic displacement parameters at the end."),
)
# Saenger classes of the base pairs written as explicit restraints
SAENGER = {"GC": 19, "CG": 19, "AU": 20, "UA": 20, "GU": 28, "UG": 28}

# Coot: headless command lines.  {exe} executable, {script} the generated Python script.  --no-guano stops
# 0.9 from writing its state / history files into the step folder.
COOT_VERSIONS: Dict[str, Dict[str, str]] = {
    "0.9": {"label": "Coot 0.9", "command": "{exe} --no-graphics --no-guano --script {script}"},
    "1.x": {"label": "Coot 1.x", "command": "{exe} --no-graphics --script {script}"},  # unverified flags
}
# Coot API names tried in order (first in the 'coot' module, then among the global functions of 0.9).
# Required functions stop the script when none of the names exists; optional ones are skipped.
COOT_API: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "read_pdb": (("read_pdb",), True),
    "read_map": (("handle_read_ccp4_map",), True),
    "set_refinement_map": (("set_imol_refinement_map",), True),
    "read_dictionary": (("read_cif_dictionary",), True),
    "immediate_replacement": (("set_refinement_immediate_replacement",), True),
    "refine_zone": (("refine_zone",), True),
    "accept": (("accept_regularizement", "accept_moving_atoms"), True),
    "write_pdb": (("write_pdb_file",), True),
    "set_weight": (("set_matrix",), False),
    "auto_weight": (("auto_weight_for_refinement",), False),
    "torsions": (("set_refine_with_torsion_restraints",), False),
    "exit": (("coot_real_exit",), False),
}

# Nucleotide names Phenix and Coot know from the Chemical Component Dictionary: D-RNA / DNA and the four
# L-ribonucleotides.  The L 5'/3' variants (0A5, 0A3 ...) need restraint dictionaries.
CCD_NUCLEOTIDES = frozenset({"A", "C", "G", "U", "I", "N", "DA", "DC", "DG", "DT", "DI", "DN"})
L_PARENTS = ("0A", "0C", "0G", "0U")
L_VARIANTS = tuple(p + t for p in L_PARENTS for t in "53")


# ---------------------------------------------------------------------------
# model inspection (residue names, chirality, residues) - cached per file
# ---------------------------------------------------------------------------


@dataclass
class ModelReport:
    """What the checks need to know about a model PDB file."""
    l_parents: List[str] = field(default_factory=list)  # 0A/0C/0G/0U present
    l_variants: List[str] = field(default_factory=list)  # 0A5/0A3 ... present
    foreign: Dict[str, int] = field(default_factory=dict)  # nucleotide names unknown to the CCD -> count
    l_with_d_names: List[str] = field(default_factory=list)  # labels of L sugars named as D residues
    phosphate5: List[str] = field(default_factory=list)  # labels of 0A5 ... residues that keep a 5'-phosphate
    nucleotides: List[Tuple[str, int, str, str]] = field(default_factory=list)  # (chain, resseq, icode, base)
    residues: List[Tuple[str, int]] = field(default_factory=list)  # (chain, resseq) of every residue
    pairs: List[Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str], str]] = field(default_factory=list)
    stacks: List[Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str], str]] = field(default_factory=list)
    annotated: bool = False  # REMARK 950 records present
    problem: str = ""


_REPORTS: Dict[Tuple[str, float], ModelReport] = {}


def _remark_residue(text: str) -> Tuple[str, int, str, str]:
    """A REMARK 950 residue field (chain, resseq in 4 columns, icode, name in 3) -> (chain, resseq, icode, name)."""
    return (text[0:1].strip(), int(text[1:5]), text[5:6].strip(), text[6:9].strip())


def read_annotation(header: Sequence[str]) -> Tuple[list, list]:
    """PAIR and STACK records of an MDATK base-pair annotation (see :mod:`mdatk.annotate`)."""
    pairs, stacks = [], []
    for line in header:
        if not line.startswith("REMARK 950 PAIR ") and not line.startswith("REMARK 950 STACK"):
            continue
        try:
            item = (_remark_residue(line[17:26]), _remark_residue(line[27:36]), line[37:].strip())
        except ValueError:
            continue
        (pairs if line.startswith("REMARK 950 PAIR") else stacks).append(item)
    return pairs, stacks


def model_report(path: str) -> ModelReport:
    """Residue names, chirality and annotation of a PDB file (cached by path and modification time)."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError as exc:
        return ModelReport(problem=str(exc))
    if key in _REPORTS:
        return _REPORTS[key]
    report = ModelReport()
    try:
        s = pdbtools.read_pdb(path)
    except (OSError, pdbtools.PDBError) as exc:
        report.problem = f"could not read {os.path.basename(path)}: {exc}"
        _REPORTS[key] = report
        return report
    report.annotated = any(line.startswith("REMARK 950") for line in s.header)
    report.pairs, report.stacks = read_annotation(s.header)
    names_d = {}
    for res in s.residues():
        report.residues.append((res.chain, res.resseq))
        nt = pdbtools._nucleotide(res)
        if nt is None:
            continue
        name = res.name.strip().upper()
        report.nucleotides.append((res.chain, res.resseq, res.icode.strip(), nt.base))
        if name in L_PARENTS:
            report.l_parents.append(name)
        elif name in L_VARIANTS:
            report.l_variants.append(name)
            if name.endswith("5") and any(a.name in ("P", "OP1", "OP2", "OP3") for a in res.atoms):
                report.phosphate5.append(res.label)
        elif name not in CCD_NUCLEOTIDES:
            report.foreign[name] = report.foreign.get(name, 0) + 1
        names_d[id(res)] = nt.chirality
    pdbtools.assign_chirality(s, detect=True)
    for res in s.residues():
        if names_d.get(id(res)) == "D" and res.chirality == "L":
            report.l_with_d_names.append(res.label)
    report.l_parents = sorted(set(report.l_parents))
    report.l_variants = sorted(set(report.l_variants))
    _REPORTS[key] = report
    return report


def dictionary_codes(ctx: StepContext, port: str = "dictionaries") -> Tuple[Set[str], bool]:
    """Residue codes covered by the connected / chosen dictionaries; (codes, complete).

    ``complete`` is False when a dictionary file does not exist yet and its producer did not say which
    codes it writes."""
    value = ctx.value(port)
    if value is None or not value.paths:
        return set(), True
    codes = {str(c).upper() for c in value.data.get("codes", ())}
    if codes:
        return codes, True
    complete = True
    for path in value.paths:
        full = ctx.abspath(path)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            complete = False
            continue
        codes |= {m.upper() for m in re.findall(r"^data_comp_(\S+)", text, re.M) if m.lower() != "list"}
        codes |= {m.upper().strip("'\"") for m in re.findall(r"^_chem_comp\.id\s+(\S+)", text, re.M)}
    return codes, complete


def _listing(items: Sequence[str], limit: int = 8) -> str:
    items = list(items)
    return ", ".join(items[:limit]) + (f" and {len(items) - limit} more" if len(items) > limit else "")


def model_issues(s: Dict[str, Any], ctx: StepContext, program: str) -> List[Issue]:
    """Checks shared by both refinement steps: files, residue names and restraint dictionaries."""
    out: List[Issue] = []
    for port, what in (("model", "model"), ("map", "map")):
        value = ctx.value(port)
        # hand-picked files are checked by the planner; files of an 'Existing file' step are absolute
        if value and value.paths and value.source and os.path.isabs(value.path) \
                and not os.path.exists(value.path):
            out.append(Issue("warning", f"{what} file not found (yet): {value.path}", (f"input:{port}",)))
    if not ctx.has("model"):
        return out
    dict_codes, complete = dictionary_codes(ctx)
    have_dicts = ctx.has("dictionaries")
    path = ctx.existing("model")
    if not path:
        if not have_dicts and ctx.data("model").get("chirality") == "L":
            out.append(Issue("warning", f"the model is L-RNA: {program} needs restraint dictionaries for its 5'/3' "
                                        "variants (0A5/0A3 ...) - connect a 'restraint-dicts' step",
                             ("input:dictionaries",)))
        elif ctx.value("model").source:
            out.append(Issue("info", "residue names are checked once the model exists", ("input:model",)))
        return out
    if path.lower().endswith((".cif", ".mmcif")):
        out.append(Issue("info", "mmCIF model: residue names are not checked", ("input:model",)))
        return out
    report = model_report(path)
    if report.problem:
        out.append(Issue("warning", report.problem, ("input:model",)))
        return out
    fix = "convert with 'mdatk pdb clean --scheme ccd' first"
    if report.foreign:
        names = [f"{n} x{c}" for n, c in sorted(report.foreign.items(), key=lambda kv: -kv[1])]
        out.append(Issue("warning", f"{program} does not recognise the residue names {_listing(names)} "
                                    f"(Amber-style names); {fix}", ("input:model",)))
    if report.l_with_d_names:
        out.append(Issue("warning", f"{len(report.l_with_d_names)} residues have L sugars but D names (e.g. "
                                    f"cpptraj output) and would be restrained as D: "
                                    f"{_listing(report.l_with_d_names, 4)}; {fix}", ("input:model",)))
    if report.phosphate5:
        out.append(Issue("warning", f"5'-terminal residues named as 5'-OH variants carry a phosphate: "
                                    f"{_listing(report.phosphate5, 4)}; the 0A5 ... dictionaries have no "
                                    f"P/OP1/OP2/OP3 - rename them to the CCD components (0A ...) with "
                                    f"'mdatk pdb clean --scheme ccd'", ("input:model",)))
    if report.l_variants:
        if not have_dicts:
            out.append(Issue("warning", f"the model contains {', '.join(report.l_variants)}, which the Chemical "
                                        "Component Dictionary does not define (it knows 0A/0C/0G/0U): connect "
                                        "a 'restraint-dicts' step with their dictionaries",
                             ("input:dictionaries",)))
        elif complete:
            missing = [c for c in report.l_variants if c not in dict_codes]
            if missing:
                out.append(Issue("warning", f"no restraint dictionary for {', '.join(missing)} among the "
                                            "connected dictionaries", ("input:dictionaries",)))
    if report.l_parents and not have_dicts:
        out.append(Issue("info", f"{', '.join(report.l_parents)} are Chemical Component Dictionary entries; "
                                 f"{program} reads them from its monomer library (connect dictionaries if "
                                 "yours lacks them)", ("input:dictionaries",)))
    return out


def resolution_issues(value: Optional[float], key: str = "resolution") -> List[Issue]:
    if value is None:
        return [Issue("error", "give the resolution of the map (Å)", (key,))]
    if not 1.0 <= value <= 20.0:
        return [Issue("warning", f"a resolution of {value:g} Å is outside the usual 1-20 Å of cryo-EM maps",
                      (key,))]
    return []


def _guarded(ids: Sequence[str]) -> List[str]:
    # resources are reported once mdatk.citations lists them
    from ..citations import RESOURCES
    return [r for r in ids if r in RESOURCES]


# ---------------------------------------------------------------------------
# Phenix: explicit base pairs (.eff)
# ---------------------------------------------------------------------------

ResidueId = Tuple[str, int, str]  # chain, resseq, insertion code


def phil_selection(res: ResidueId) -> str:
    chain, resseq, icode = res
    return f"chain '{chain or ' '}' and resid {resseq}{icode}"


def eff_text(pairs: Sequence[Tuple[ResidueId, ResidueId, int]], stacks: Sequence[Tuple[ResidueId, ResidueId]],
             version: str, source: str) -> str:
    """Phenix parameter file with explicit base-pair (and stacking) restraints."""
    ver = PHENIX_VERSIONS[version]
    scopes = ver["ss_scope"].split(".")
    lines = [f"# Base-pair restraints for {ver['program']} ({ver['label']}) - written by {PROJECT_ABBREV} "
             f"{__version__}", f"# Source: {source}"]
    lines += [f"{'  ' * k}{name} {{" for k, name in enumerate(scopes)]
    ind = "  " * len(scopes)
    lines += [f"{ind}{ver['ss_enabled']} = True", f"{ind}{ver['na_scope']} {{"]
    for a, b, saenger in pairs:
        lines += [f"{ind}  {ver['base_pair']} {{",
                  f'{ind}    base1 = "{phil_selection(a)}"',
                  f'{ind}    base2 = "{phil_selection(b)}"',
                  f"{ind}    {ver['saenger_class']} = {saenger}",
                  f"{ind}  }}"]
    for a, b in stacks:
        lines += [f"{ind}  {ver['stacking_pair']} {{",
                  f'{ind}    base1 = "{phil_selection(a)}"',
                  f'{ind}    base2 = "{phil_selection(b)}"',
                  f"{ind}  }}"]
    lines.append(f"{ind}}}")
    lines += [f"{'  ' * k}}}" for k in reversed(range(len(scopes)))]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Coot: residue windows (these two functions are copied into the generated script; they must stay
# plain Python 2.7 / 3 without annotations or imports)
# ---------------------------------------------------------------------------


def pdb_residues(text):
    """[(chain, [residue numbers in file order])] of the polymer residues of PDB text (first model).

    Water and single-atom residues (ions) are left out."""
    water = ("HOH", "WAT", "H2O", "DOD", "SOL", "TIP", "TIP3", "T3P", "T4P", "SPC", "OPC")
    chains, counts, order = {}, {}, []
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM") or len(line) < 26:
            continue
        name, chain = line[17:20].strip(), line[21:22]
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        key = (chain, resseq, line[26:27])
        if key not in counts:
            counts[key] = 0
            order.append((key, name))
        counts[key] += 1
    for key, name in order:
        if name.upper() in water or counts[key] < 2:
            continue
        numbers = chains.setdefault(key[0], [])
        if key[1] not in numbers:
            numbers.append(key[1])
    seen = []
    for key, _name in order:
        if key[0] in chains and key[0] not in seen:
            seen.append(key[0])
    return [(c, chains[c]) for c in seen]


def refinement_zones(residues, ranges, size, overlap):
    """Overlapping windows [(chain, first, last)] of ``size`` residues sharing ``overlap`` residues.

    ``residues`` is the output of pdb_residues; ``ranges`` a list of (chain or None, first, last) limiting
    the refined residues (empty: everything; chain None: every chain).  Windows never span residues that
    are not refined; the last window of a stretch ends at its last residue."""
    step = max(1, size - overlap)
    zones = []
    for chain, numbers in residues:
        stretches, current = [], []
        for n in numbers:
            wanted = not ranges or any((c is None or c == chain) and a <= n <= b for c, a, b in ranges)
            if wanted:
                current.append(n)
            elif current:
                stretches.append(current)
                current = []
        if current:
            stretches.append(current)
        for stretch in stretches:
            start = 0
            while True:
                window = stretch[start:start + size]
                zones.append((chain, window[0], window[-1]))
                if start + size >= len(stretch):
                    break
                start += step
                if start + size > len(stretch):
                    start = max(0, len(stretch) - size)
    return zones


def coot_script(model: str, map_path: str, dictionaries: Sequence[str], output: str,
                ranges: Sequence[Tuple[Optional[str], int, int]], size: int, overlap: int, cycles: int,
                weight_mode: str, weight: Optional[float], torsions: bool, version: str,
                zones: Optional[Sequence[Tuple[str, int, int]]] = None) -> str:
    """The Coot Python script (Python 2.7 and 3; Coot 0.9 and 1.x)."""
    def pylist(items):
        return "[" + ", ".join(repr(i) for i in items) + "]"

    api = "{\n" + "".join(f"    {k!r}: ({', '.join(repr(n) for n in names)},),\n" if len(names) == 1 else
                          f"    {k!r}: ({', '.join(repr(n) for n in names)}),\n"
                          for k, (names, _req) in COOT_API.items()) + "}"
    required = pylist(k for k, (_names, req) in COOT_API.items() if req)
    zone_text = "None  # found in the model when the script runs" if zones is None else \
        "[\n" + "".join(f"    {z!r},\n" for z in zones) + "]"
    lines = [
        f"# Coot batch real-space refinement - written by {PROJECT_ABBREV} {__version__} for "
        f"{COOT_VERSIONS[version]['label']} (runs in Coot 0.9 and 1.x).",
        "# Started without graphics by the step's shell script; for interactive refinement open the model",
        "# and the map in Coot yourself.",
        "from __future__ import print_function",
        "",
        "import sys",
        "import traceback",
        "",
        f"MODEL = {model!r}",
        f"MAP = {map_path!r}",
        f"DICTIONARIES = {pylist(dictionaries)}",
        f"OUTPUT = {output!r}",
        f"RANGES = {pylist(ranges)}  # (chain or None, first, last); empty: every residue",
        f"ZONE_SIZE = {size}",
        f"OVERLAP = {overlap}",
        f"CYCLES = {cycles}",
        f"WEIGHT_MODE = {weight_mode!r}  # 'default', 'auto' or 'fixed'",
        f"WEIGHT = {weight!r}",
        f"TORSIONS = {torsions!r}",
        f"ZONES = {zone_text}",
        "",
        "# Coot functions: tried in this order in the 'coot' module (Coot 1.x; also present in 0.9) and",
        "# among the global functions Coot 0.9 defines for scripts.",
        f"API_NAMES = {api}",
        f"REQUIRED = {required}",
        "",
        "",
        "def _namespaces():",
        "    spaces = []",
        "    for module in ('coot', 'coot_utils'):",
        "        try:",
        "            spaces.append(vars(__import__(module)))",
        "        except ImportError:",
        "            pass",
        "    spaces.append(globals())",
        "    try:",
        "        import __main__",
        "        spaces.append(vars(__main__))",
        "    except ImportError:",
        "        pass",
        "    return spaces",
        "",
        "",
        "def coot_api():",
        '    """{role: function} for the Coot functions this script uses (None for missing optional ones)."""',
        "    spaces = _namespaces()",
        "    api, missing = {}, []",
        "    for role, names in API_NAMES.items():",
        "        api[role] = None",
        "        for name in names:",
        "            found = [space[name] for space in spaces if callable(space.get(name))]",
        "            if found:",
        "                api[role] = found[0]",
        "                break",
        "        if api[role] is None and role in REQUIRED:",
        "            missing.append('/'.join(names))",
        "    if missing:",
        "        raise RuntimeError('Coot functions not found: ' + ', '.join(missing))",
        "    return api",
        "",
        "",
        inspect.getsource(pdb_residues).rstrip(),
        "",
        "",
        inspect.getsource(refinement_zones).rstrip(),
        "",
        "",
        "def _mdatk_main(api):",
        "    for path in DICTIONARIES:",
        "        print('Reading dictionary', path)",
        "        api['read_dictionary'](path)",
        "    imol = api['read_pdb'](MODEL)",
        "    if imol is None or imol < 0:",
        "        raise RuntimeError('Coot could not read the model ' + MODEL)",
        "    imap = api['read_map'](MAP, 0)",
        "    if imap is None or imap < 0:",
        "        raise RuntimeError('Coot could not read the map ' + MAP)",
        "    api['set_refinement_map'](imap)",
        "    api['immediate_replacement'](1)",
        "    if TORSIONS:",
        "        if api['torsions'] is None:",
        "            print('WARNING: this Coot has no set_refine_with_torsion_restraints; torsions not restrained')",
        "        else:",
        "            api['torsions'](1)",
        "    if WEIGHT_MODE == 'fixed' and WEIGHT is not None:",
        "        if api['set_weight'] is None:",
        "            print('WARNING: this Coot has no set_matrix; keeping its default map weight')",
        "        else:",
        "            api['set_weight'](float(WEIGHT))",
        "            print('Map weight', WEIGHT)",
        "    zones = ZONES",
        "    if zones is None:",
        "        with open(MODEL) as fh:",
        "            zones = refinement_zones(pdb_residues(fh.read()), RANGES, ZONE_SIZE, OVERLAP)",
        "    if not zones:",
        "        raise RuntimeError('no residues to refine')",
        "    if WEIGHT_MODE == 'auto':",
        "        if api['auto_weight'] is None:",
        "            print('WARNING: this Coot has no auto_weight_for_refinement; keeping its default map weight')",
        "        else:",
        "            api['auto_weight']()",
        "    for cycle in range(CYCLES):",
        "        for k, (chain, first, last) in enumerate(zones):",
        "            print('Cycle %d/%d, zone %d/%d: chain %s residues %d-%d'",
        "                  % (cycle + 1, CYCLES, k + 1, len(zones), chain or '_', first, last))",
        "            sys.stdout.flush()",
        "            api['refine_zone'](imol, chain, first, last, '')",
        "            api['accept']()",
        "    api['write_pdb'](imol, OUTPUT)",
        "    print('Wrote', OUTPUT)",
        "",
        "",
        "def _mdatk_finish(status):",
        "    sys.stdout.flush()",
        "    try:",
        "        exit_coot = _mdatk_exit_function()",
        "    except Exception:",
        "        exit_coot = None",
        "    if exit_coot is not None:",
        "        exit_coot(status)",
        "    sys.exit(status)",
        "",
        "",
        "def _mdatk_exit_function():",
        "    for space in _namespaces():",
        "        for name in API_NAMES['exit']:",
        "            if callable(space.get(name)):",
        "                return space[name]",
        "    return None",
        "",
        "",
        "try:",
        "    _mdatk_main(coot_api())",
        "except Exception:",
        "    traceback.print_exc()",
        "    print('ERROR: Coot refinement failed')",
        "    _mdatk_finish(1)",
        "_mdatk_finish(0)",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# shared option helpers
# ---------------------------------------------------------------------------


def _phenix_source_lines(s: Dict[str, Any]) -> List[str]:
    ver = PHENIX_VERSIONS[s["phenix_version"]]
    if not s["phenix_dir"].strip():
        return []
    return [f"# {ver['label']}", f"source {shell_quote(os.path.join(s['phenix_dir'].strip(), ver['env_script']))}",
            ""]


def _found_on_path(exe: str) -> str:
    return f'command -v {exe} >/dev/null || {{ echo "ERROR: {exe} not found on PATH" >&2; exit 1; }}'


# ---------------------------------------------------------------------------
# Phenix real-space refinement
# ---------------------------------------------------------------------------


@register
class PhenixRSRTool(Tool):
    name = "phenix-rsr"
    label = "Phenix · real-space refinement"
    category = "Refinement"
    tag = "phenix-rsr"
    description = ("phenix.real_space_refine (Phenix 1.21 / 2.x) of a model against a cryo-EM map, with "
                   "restraint dictionaries and nucleic-acid base-pair / stacking restraints.")
    inputs = [Port("model", "Model (PDB)", ("pdb",), tooltip="The model to refine (PDB; an mmCIF file chosen "
                                                             "by hand works as well)."),
              Port("map", "Density map", ("map",), tooltip="MRC / CCP4 map."),
              Port("dictionaries", "Restraint dictionaries", ("cif",), optional=True, multiple=True,
                   tooltip="mmCIF dictionaries of residues Phenix does not know (L-RNA 5'/3' variants: "
                           "'restraint-dicts' step)."),
              Port("secstruct", "Secondary structure", ("secstruct",), optional=True,
                   tooltip="Dot-bracket written as explicit base-pair restraints (.eff).")]
    outputs = [Port("refined", "Refined model (PDB)", ("pdb",)),
               Port("log", "Log", ("text",))]
    sections = [
        Section("phenix", "Phenix", [
            Option("phenix_version", "Phenix version", "radio", "2.x", "Installed Phenix release.",
                   choices=[(k, v["label"]) for k, v in PHENIX_VERSIONS.items()]),
            Option("phenix_dir", "Phenix installation", "text", "",
                   "Top folder of the Phenix installation; the script sources its phenix_env.sh. Leave empty "
                   "when phenix.real_space_refine is already on PATH (or set it up below).",
                   placeholder=PHENIX_VERSIONS["2.x"]["example"]),
        ], tooltip="Which Phenix runs the refinement."),
        Section("refinement", "Refinement", [
            Option("resolution", "Resolution", "float", None, "Resolution of the map (required).", optional=True,
                   minimum=0.1, maximum=100.0, unit="Å"),
            Option("macro_cycles", "Macro cycles", "int", 5, "macro_cycles of real_space_refine.", minimum=1,
                   maximum=100),
            *[Option(key, label, "bool", default, f"run={value}: {tip}")
              for key, value, label, default, tip in RUN_STRATEGIES],
            Option("nproc", "Processors", "int", 1, "nproc (parallel parts of the refinement).", minimum=1,
                   maximum=256),
            Option("random_seed", "Random seed", "int", None, "Empty: Phenix's default.", optional=True),
            Option("prefix", "Output prefix", "text", "", "output.prefix of Phenix's files (empty: "
                   "<project>_<step>_phenix-rsr_<time>); the refined model is copied to the step's own file.",
                   placeholder="refine1"),
            Option("extra", "Extra parameters", "multiline", "",
                   "Appended verbatim to the command line, one or more per line (# starts a comment).",
                   placeholder="ramachandran_plot_restraints.enabled=False"),
        ], tooltip="real_space_refine parameters."),
        Section("secondary", "Base pairs", [
            Option("na_ss", "Nucleic-acid secondary-structure restraints", "bool", True,
                   "Phenix searches the model for base pairs and stacking and restrains them "
                   "(secondary_structure.enabled=True)."),
            Option("pairs_source", "Explicit base pairs", "radio", "secstruct",
                   "Written as base_pair restraints into a parameter (.eff) file. Dot-bracket: the connected "
                   "secondary structure or the one typed below (nothing is written when there is none). "
                   "Annotation: the REMARK 950 PAIR records of the model ('mdatk pdb annotate'), read when the "
                   "files are written.",
                   choices=[("none", "None"), ("secstruct", "Dot-bracket"), ("annotation", "Model annotation")]),
            Option("sequence", "Sequence", "multiline", "", "For a typed dot-bracket (chains separated by spaces).",
                   placeholder="GGGCAC GUGCCC", requires={"pairs_source": "secstruct"}),
            Option("dotbracket", "Dot-bracket", "multiline", "", "Ignored when a secondary structure is connected.",
                   placeholder="(((((( ))))))", requires={"pairs_source": "secstruct"}),
            Option("numbering", "Residue numbering", "radio", "model",
                   "How dot-bracket positions map to residues: the nucleotides of the model in file order "
                   "(when it exists; otherwise continuous numbering), numbers continuous across chains from 1, "
                   "or restarting at 1 in every chain.",
                   choices=[("model", "From the model"), ("continuous", "Continuous"), ("chain", "Per chain")],
                   requires={"pairs_source": "secstruct"}),
            Option("wobble", "Restrain G·U wobble pairs", "bool", True, "Saenger class 28.",
                   requires={"pairs_source": ("secstruct", "annotation")}),
            Option("stacking", "Stacking from the annotation", "bool", False,
                   "Also write the STACK records as stacking_pair restraints.",
                   requires={"pairs_source": "annotation"}),
        ], tooltip="Base-pair and stacking restraints."),
        env_section("module load phenix   or   source /usr/local/phenix-2.0-5936/phenix_env.sh"),
    ]

    # ------------------------------------------------------------- helpers
    def strategies(self, s) -> List[str]:
        return [value for key, value, _l, _d, _t in RUN_STRATEGIES if s[key]]

    def prefix(self, s, ctx) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "_", s["prefix"].strip()) or ctx.name(self.tag)

    def explicit_pairs(self, s, ctx) -> Tuple[List[Tuple[ResidueId, ResidueId, int]],
                                              List[Tuple[ResidueId, ResidueId]], str, List[Issue]]:
        """(base pairs with Saenger class, stacking pairs, source description, issues)."""
        issues: List[Issue] = []
        mode = s["pairs_source"]
        if mode == "none":
            if ctx.has("secstruct"):
                issues.append(Issue("info", "the connected secondary structure is not used (explicit base pairs: "
                                            "none)", ("pairs_source",)))
            return [], [], "", issues
        if mode == "annotation":
            return self._annotation_pairs(s, ctx)
        from .amber import secondary_structure
        model, source, issues = secondary_structure(s, ctx)
        if model is None or not model.pairs:
            if model is not None and not issues:
                issues.append(Issue("warning", "the secondary structure has no base pairs", ("dotbracket",)))
            return [], [], "", issues
        residues: List[ResidueId] = []
        path = ctx.existing("model")
        is_cif = bool(path) and path.lower().endswith((".cif", ".mmcif"))
        report = model_report(path) if path and not is_cif else None
        if s["numbering"] == "model" and is_cif:
            issues.append(Issue("error", "numbering from the model needs a PDB model (mmCIF is not read here); "
                                         "choose continuous or per-chain numbering", ("numbering", "input:model")))
            return [], [], "", issues
        if s["numbering"] == "model" and report is not None and not report.problem:
            nts = report.nucleotides
            if len(nts) != model.length:
                issues.append(Issue("error", f"the secondary structure has {model.length} nucleotides, the model "
                                             f"{len(nts)}", ("dotbracket", "input:secstruct")))
                return [], [], "", issues
            residues = [(c, n, i) for c, n, i, _b in nts]
            wrong = [k + 1 for k, (_c, _n, _i, b) in enumerate(nts) if b != model.base(k + 1)]
            if wrong:
                issues.append(Issue("warning", f"the sequence differs from the model at {len(wrong)} position(s), "
                                               f"e.g. {wrong[0]}", ("sequence", "input:secstruct")))
            source += ", residues of the model in file order"
        elif s["numbering"] == "chain":
            residues = [(c.id, k + 1, "") for c in model.chains for k in range(len(c.sequence))]
            source += ", numbered from 1 in every chain"
        else:
            residues = [(c.id, c.start + k, "") for c in model.chains for k in range(len(c.sequence))]
            source += ", numbered continuously from 1"
        keep, skip = rna.classify_pairs(model, s["wobble"])
        if skip:
            kinds = sorted({k for _i, _j, k in skip})
            issues.append(Issue("info", f"{len(skip)} base pair(s) not restrained ({', '.join(kinds)})",
                                ("wobble",) if any(k in rna.WOBBLE for k in kinds) else ("dotbracket",)))
        pairs = [(residues[i - 1], residues[j - 1], SAENGER[kind]) for i, j, kind in keep]
        return pairs, [], source, issues

    def _annotation_pairs(self, s, ctx):
        from ..secstruct import OTHER, lw_kind
        issues: List[Issue] = []
        path = ctx.existing("model")
        if not path:
            issues.append(Issue("error", "the model must exist when the files are written: its REMARK 950 "
                                         "annotation is read then", ("pairs_source", "input:model")))
            return [], [], "", issues
        report = model_report(path)
        if report.problem:
            issues.append(Issue("error", report.problem, ("input:model",)))
            return [], [], "", issues
        if not report.pairs:
            issues.append(Issue("warning", "the model has no REMARK 950 base-pair records; annotate it with "
                                           "'mdatk pdb annotate'", ("pairs_source", "input:model")))
            return [], [], "", issues
        pairs, skipped = [], 0
        for a, b, code in report.pairs:
            na, nb = pdbtools.parse_residue_name(a[3]), pdbtools.parse_residue_name(b[3])
            kind = (na.base + nb.base) if na and nb else ""
            if lw_kind(code) == OTHER or kind not in SAENGER or (kind in rna.WOBBLE and not s["wobble"]):
                skipped += 1
                continue
            pairs.append((a[:3], b[:3], SAENGER[kind]))
        if skipped:
            issues.append(Issue("info", f"{skipped} annotated pair(s) are not Watson-Crick / G·U pairs (or G·U is "
                                        "off) and are not restrained", ("wobble",)))
        stacks = [(a[:3], b[:3]) for a, b, _c in report.stacks] if s["stacking"] else []
        if s["stacking"] and not report.stacks:
            issues.append(Issue("info", "the annotation has no STACK records (they come from the barnaba engine)",
                                ("stacking",)))
        if not pairs and not stacks:
            issues.append(Issue("warning", "no restrainable pairs in the annotation", ("pairs_source",)))
        return pairs, stacks, f"REMARK 950 annotation of {os.path.basename(path)}", issues

    # ------------------------------------------------------------- planner
    def resources(self, s):
        return _guarded(["phenix", "phenix_rsr"])

    def describe(self, s):
        res = f"{s['resolution']:g} Å" if s["resolution"] is not None else "resolution?"
        return f"{PHENIX_VERSIONS[s['phenix_version']]['label']}: {res}, {s['macro_cycles']} macro cycles"

    def output_values(self, s, ctx):
        return {"refined": PortValue([ctx.own("refined", "pdb")], "pdb", dict(ctx.data("model"))),
                "log": PortValue([ctx.own(self.tag, "log")], "text")}

    def check(self, s, ctx):
        out = model_issues(s, ctx, "Phenix")
        out += resolution_issues(s["resolution"])
        if not self.strategies(s):
            out.append(Issue("error", "tick at least one run strategy", tuple(k for k, *_ in RUN_STRATEGIES)))
        pairs, stacks, _src, issues = self.explicit_pairs(s, ctx)
        out += issues
        if s["resolution"] is not None and s["resolution"] > 4.0 and not s["na_ss"] and not pairs:
            out.append(Issue("info", "no base-pair restraints: at resolutions worse than 4 Å RNA helices tend to "
                                     "deform", ("na_ss", "pairs_source")))
        if pairs and s["na_ss"]:
            out.append(Issue("info", "explicit base pairs are given: Phenix restrains those; whether it also "
                                     "searches for more depends on the version", ("na_ss", "pairs_source")))
        if s["run_simulated_annealing"] and s["run_morphing"]:
            out.append(Issue("info", "morphing and simulated annealing both move the model a lot; use them for "
                                     "models far from the map", ("run_morphing", "run_simulated_annealing")))
        if not s["phenix_dir"].strip() and not s["env_setup"].strip() and not s["conda_env"].strip():
            out.append(Issue("info", "phenix.real_space_refine must be on PATH (or set the Phenix installation)",
                             ("phenix_dir",)))
        if s["prefix"].strip() and self.prefix(s, ctx) != s["prefix"].strip():
            out.append(Issue("info", f"output prefix written as {self.prefix(s, ctx)}", ("prefix",)))
        return out

    # ------------------------------------------------------------- files
    def generate(self, s, ctx):
        ver = PHENIX_VERSIONS[s["phenix_version"]]
        files: List[GeneratedFile] = []
        script = ctx.name(self.tag, "sh")
        log = ctx.name(self.tag, "log")
        refined = ctx.name("refined", "pdb")
        refined_cif = ctx.name("refined", "cif")
        prefix = self.prefix(s, ctx)
        pairs, stacks, source, _ = self.explicit_pairs(s, ctx)
        eff = ""
        if pairs or stacks:
            eff = ctx.name("basepairs", "eff")
            files.append(GeneratedFile(eff, eff_text(pairs, stacks, s["phenix_version"], source), "input",
                                       f"{len(pairs)} base-pair / {len(stacks)} stacking restraints"))
        notes = [f"{ver['label']}: {', '.join(self.strategies(s))}; {s['macro_cycles']} macro cycles."]
        sh = script_header(f"Phenix real-space refinement ({ver['label']})", ctx, s, script, notes)
        sh += _phenix_source_lines(s)
        sh += [_found_on_path(ver["program"]), ""]
        args = [shell_quote(ctx.inp("model")), shell_quote(ctx.inp("map"))]
        args += [shell_quote(p) for p in ctx.inps("dictionaries")]
        if eff:
            args.append(eff)
        if s["resolution"] is not None:
            args.append(f"{ver['resolution']}={fortran_float(s['resolution'])}")
        args.append(f"{ver['run']}={'+'.join(self.strategies(s))}")
        args.append(f"{ver['macro_cycles']}={s['macro_cycles']}")
        if s["na_ss"]:
            args.append(f"{ver['ss_scope']}.{ver['ss_enabled']}=True")
        if s["nproc"] > 1:
            args.append(f"{ver['nproc']}={s['nproc']}")
        if s["random_seed"] is not None:
            args.append(f"{ver['random_seed']}={s['random_seed']}")
        args.append(f"{ver['prefix']}={prefix}")
        args += extra_flags(s["extra"])
        sh += [f"# real-space refinement (Phenix's own log: {prefix}*real_space_refined*.log)",
               f"{ver['program']} \\"]
        sh += [f"    {a} \\" for a in args]
        sh += [f'    > {log} 2>&1 || {{ echo "ERROR: {ver["program"]} failed (see {log})" >&2; exit 1; }}', ""]
        sh += ["# the refined model: Phenix names it <prefix>_real_space_refined_NNN.pdb (newest first)",
               f'OUT=$(ls -t {prefix}*real_space_refined*.pdb 2>/dev/null | head -n 1 || true)',
               f'CIF=$(ls -t {prefix}*real_space_refined*.cif 2>/dev/null | head -n 1 || true)',
               f'[ -n "$CIF" ] && cp "$CIF" {refined_cif}',
               'if [ -z "$OUT" ] && [ -n "$CIF" ]; then',
               "  # only mmCIF output (e.g. names that do not fit the PDB format): convert it",
               f'  {ver["cif_as_pdb"]} "$CIF" >> {log} 2>&1 || true',
               '  [ -s "${CIF%.cif}.pdb" ] && OUT="${CIF%.cif}.pdb"',
               "fi",
               f'[ -n "$OUT" ] || {{ echo "ERROR: no {prefix}*real_space_refined*.pdb found (see {log})" >&2; '
               "exit 1; }",
               f'cp "$OUT" {refined}',
               require_file(refined, ver["program"]), "",
               "# final model statistics",
               f'grep -E "CC_mask|Clashscore|MolProbity score" {log} | tail -n 6 || true',
               f'echo "Refined model: {refined}"', ""]
        files.append(GeneratedFile(script, "\n".join(sh), "run", f"runs {ver['program']}"))
        return files


# ---------------------------------------------------------------------------
# Coot batch real-space refinement
# ---------------------------------------------------------------------------


@register
class CootRSRTool(Tool):
    name = "coot-rsr"
    label = "Coot · real-space refinement (batch)"
    category = "Refinement"
    tag = "coot-rsr"
    description = ("Coot 0.9 / 1.x without graphics: every chain refined against the map in overlapping "
                   "residue windows (refine_zone), with restraint dictionaries.")
    inputs = [Port("model", "Model (PDB)", ("pdb",)),
              Port("map", "Density map", ("map",), tooltip="MRC / CCP4 map."),
              Port("dictionaries", "Restraint dictionaries", ("cif",), optional=True, multiple=True,
                   tooltip="mmCIF dictionaries of residues Coot does not know (L-RNA 5'/3' variants: "
                           "'restraint-dicts' step).")]
    outputs = [Port("refined", "Refined model (PDB)", ("pdb",)),
               Port("log", "Log", ("text",))]
    sections = [
        Section("coot", "Coot", [
            Option("coot_version", "Coot version", "radio", "1.x",
                   "Installed Coot generation; the script works in both, the version selects the command-line "
                   "flags.", choices=[(k, v["label"]) for k, v in COOT_VERSIONS.items()]),
            Option("coot_exe", "Coot executable", "text", "coot", "Name or path of the coot program."),
        ], tooltip="Which Coot runs the refinement (batch mode; run Coot yourself to refine interactively)."),
        Section("refinement", "Refinement", [
            Option("residues", "Residues", "text", "",
                   "Residue ranges to refine, e.g. 'A:1-20, B:5-12' (empty: every polymer residue; a range "
                   "without a chain applies to every chain).", placeholder="A:1-20, B"),
            Option("zone_size", "Zone size", "int", 10, "Residues refined together (refine_zone window).",
                   minimum=1, maximum=1000),
            Option("overlap", "Zone overlap", "int", 3, "Residues shared by neighbouring windows.", minimum=0,
                   maximum=999),
            Option("cycles", "Cycles", "int", 2, "Passes over all windows.", minimum=1, maximum=100),
            Option("weight_mode", "Map weight", "radio", "default",
                   "Default: Coot's own weight. Auto: auto_weight_for_refinement (when this Coot has it). "
                   "Fixed: set_matrix with the value below.",
                   choices=[("default", "Default"), ("auto", "Auto"), ("fixed", "Fixed")]),
            Option("weight", "Weight", "float", 60.0, "set_matrix value (larger: the map pulls harder).",
                   minimum=0.01, maximum=10000.0, step=5.0, digits=2, requires={"weight_mode": "fixed"}),
            Option("torsions", "Torsion restraints", "bool", False,
                   "set_refine_with_torsion_restraints: also restrain torsions from the dictionaries."),
        ], tooltip="What Coot refines and how."),
        env_section("module load ccp4   or   source /opt/xtal/ccp4-9/bin/ccp4.setup-sh"),
    ]

    # ------------------------------------------------------------- helpers
    def ranges(self, s) -> Tuple[List[Tuple[Optional[str], int, int]], List[Issue]]:
        """Residue ranges: 'A:1-20' and 'A' (whole chain) -> (chain or None, first, last)."""
        out: List[Tuple[Optional[str], int, int]] = []
        whole: List[str] = []
        text = re.sub(r"\s*([-:])\s*", r"\1", s["residues"].strip())
        for tok in re.split(r"[,;\s]+", text):
            if re.fullmatch(r"[A-Za-z0-9]", tok or "") and not tok.isdigit():
                whole.append(tok)  # a whole chain
        rest = " ".join(t for t in re.split(r"[,;\s]+", text) if t and t not in whole)
        try:
            out = [(c, a, b) for c, a, b in parse_residue_ranges(rest)]
        except SelectionError as exc:
            return [], [Issue("error", f"Residues: {exc}", ("residues",))]
        out += [(c, -99999, 99999) for c in whole]
        return out, []

    def zones(self, s, ctx) -> Optional[List[Tuple[str, int, int]]]:
        """Windows computed now when the model exists (None: the script finds them)."""
        path = ctx.existing("model")
        if not path or path.lower().endswith((".cif", ".mmcif")):
            return None
        ranges, issues = self.ranges(s)
        if issues:
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                residues = pdb_residues(fh.read())
        except OSError:
            return None
        return refinement_zones(residues, ranges, s["zone_size"], min(s["overlap"], s["zone_size"] - 1))

    # ------------------------------------------------------------- planner
    def resources(self, s):
        return _guarded(["coot"])

    def describe(self, s):
        return (f"{COOT_VERSIONS[s['coot_version']]['label']}: {s['zone_size']}-residue zones × "
                f"{s['cycles']}")

    def output_values(self, s, ctx):
        return {"refined": PortValue([ctx.own("refined", "pdb")], "pdb", dict(ctx.data("model"))),
                "log": PortValue([ctx.own(self.tag, "log")], "text")}

    def check(self, s, ctx):
        out = model_issues(s, ctx, "Coot")
        model = ctx.inp("model")
        if model.lower().endswith((".cif", ".mmcif")):
            out.append(Issue("error", "the Coot step reads the model as PDB (residues and refinement zones); "
                                      "give a PDB file", ("input:model",)))
        ranges, issues = self.ranges(s)
        out += issues
        if s["overlap"] >= s["zone_size"]:
            out.append(Issue("error", "the zone overlap must be smaller than the zone size",
                             ("overlap", "zone_size")))
        if not s["coot_exe"].strip():
            out.append(Issue("error", "give the Coot executable", ("coot_exe",)))
        if any(c is None for c, _a, _b in ranges):
            out.append(Issue("info", "ranges without a chain id apply to every chain", ("residues",)))
        if s["zone_size"] > 30:
            out.append(Issue("info", "large zones refine slowly in Coot; 5-15 residues is usual", ("zone_size",)))
        zones = self.zones(s, ctx) if not issues and s["overlap"] < s["zone_size"] else None
        if zones is not None and not zones:
            out.append(Issue("error", "no residue of the model matches the residues to refine", ("residues",)))
        elif zones is None and ctx.has("model") and not issues:
            out.append(Issue("info", "the refinement zones are found in the model when the script runs",
                             ("residues",)))
        return out

    # ------------------------------------------------------------- files
    def generate(self, s, ctx):
        ver = COOT_VERSIONS[s["coot_version"]]
        script = ctx.name(self.tag, "sh")
        pyscript = ctx.name(self.tag, "py")
        log = ctx.name(self.tag, "log")
        refined = ctx.name("refined", "pdb")
        ranges, _ = self.ranges(s)
        overlap = min(s["overlap"], s["zone_size"] - 1)
        zones = self.zones(s, ctx)
        weight = s["weight"] if s["weight_mode"] == "fixed" else None
        py = coot_script(ctx.inp("model"), ctx.inp("map"), ctx.inps("dictionaries"), refined, ranges,
                         s["zone_size"], overlap, s["cycles"], s["weight_mode"], weight, s["torsions"],
                         s["coot_version"], zones)
        exe = s["coot_exe"].strip() or "coot"
        what = f"{len(zones)} zones" if zones is not None else "zones found at run time"
        sh = script_header(f"Coot batch real-space refinement ({ver['label']})", ctx, s, script,
                           [f"{s['zone_size']}-residue zones overlapping by {overlap}, {s['cycles']} cycle(s), "
                            f"{what}."])
        sh += [_found_on_path(shell_quote(exe)), "",
               f"rm -f {refined}",
               ver["command"].format(exe=shell_quote(exe), script=pyscript) + f" < /dev/null > {log} 2>&1 \\",
               f'    || {{ echo "ERROR: Coot failed (see {log})" >&2; exit 1; }}',
               require_file(refined, "Coot"),
               f'echo "Refined model: {refined}"', ""]
        return [GeneratedFile(pyscript, py, "input", "Coot Python script (0.9 and 1.x)"),
                GeneratedFile(script, "\n".join(sh), "run", "runs Coot without graphics")]
