"""Planner step that runs a script of the user's own (``user-script``, "Custom script").

The script is typed into the step or read from a file, and its text is copied into the step folder,
so the plan folder stays self-contained.  The step declares its own input and output files as
planner ports.  The run script MDATK writes exports their names as ``MDATK_IN_<NAME>`` /
``MDATK_OUT_<NAME>``, replaces ``{in:<name>}`` / ``{out:<name>}`` / ``{script}`` in the arguments,
runs the script with bash, python3 or a command of the user's choice, and stops with an error when a
declared output is missing, so later steps never start on files that were not written.  Like every
step, it only writes files; MDATK downloads nothing for it.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from .base import PortValue, Tool, env_section, require_file, script_header, shell_quote
from .registry import register
from .schema import FILE_KINDS, GeneratedFile, Issue, Option, Port, Section, kind_label

INTERPRETERS = [("bash", "bash"), ("python3", "python3"), ("custom", "other command")]
EXTENSIONS = {"bash": "sh", "python3": "py", "custom": "txt"}  # of a typed script's copy
# an input of type "any" accepts every file type; "text" first: the type of a hand-picked unknown file
ANY_KINDS = ("text",) + tuple(k for k in FILE_KINDS if k != "text")
MAX_SCRIPT_BYTES = 1 << 20  # a script, not data: data files come in through input ports

_PORT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_PLAIN_FILE = re.compile(r"^[A-Za-z0-9_+][A-Za-z0-9._+-]*$")
PLAIN_RULE = "use letters, digits, '.', '_', '-' and '+' (no folders or spaces)"
# {in:name} / {out:name} / {script}; '${var:-x}' and '${script}' are shell syntax, not placeholders
_PLACEHOLDER = re.compile(r"(?<!\$)\{(?:(in|out):([^{}]*)|script)\}")
_SCRIPT = re.compile(r"(?<!\$)\{script\}")
_MISSPELT = re.compile(r"(?<!\$)\{((?i:in|out|input|output|inp|outp?)):([^{}]*)\}")

INPUT_OPTIONS = [
    Option("name", "Name", "text", "input",
           "Name of the input: letters, digits, '_' and '-'. The script finds the file in $MDATK_IN_<NAME> "
           "(upper case, '-' as '_') and the arguments as {in:<name>}."),
    Option("kind", "File type", "choice", "any",
           "Which outputs of other steps can connect to it ('any file': every type).",
           choices=[("any", "any file")] + [(k, v[0]) for k, v in FILE_KINDS.items()]),
    Option("optional", "Optional", "bool", False,
           "The step also runs without this file; $MDATK_IN_<NAME> is then empty."),
    Option("multiple", "Several connections", "bool", False,
           "Several steps may feed this input. $MDATK_IN_<NAME> holds one path per line and "
           "{in:<name>} becomes one word per file."),
]
INPUT_TEMPLATES = [
    ("any", "Any file", {"name": "input", "kind": "any"}),
    ("pdb", "Structure (PDB)", {"name": "structure", "kind": "pdb"}),
    ("parm7", "Amber topology", {"name": "topology", "kind": "parm7"}),
    ("rst7", "Amber coordinates", {"name": "coordinates", "kind": "rst7"}),
    ("nc", "Amber trajectory", {"name": "trajectory", "kind": "nc"}),
    ("map", "Density map", {"name": "map", "kind": "map"}),
    ("secstruct", "Secondary structure", {"name": "secstruct", "kind": "secstruct"}),
]
OUTPUT_OPTIONS = [
    Option("name", "Name", "text", "result",
           "Name of the output: letters, digits, '_' and '-'. The script finds the file name to write in "
           "$MDATK_OUT_<NAME> (upper case, '-' as '_') and the arguments as {out:<name>}."),
    Option("kind", "File type", "choice", "pdb", "What the file contains; decides which inputs of later "
           "steps it can feed.", choices=[(k, v[0]) for k, v in FILE_KINDS.items()]),
    Option("file", "File name", "text", "",
           "A plain file name in the step folder, e.g. result.pdb. Empty: <project>_<step>_<name>_<time> "
           "with the type's extension.", placeholder="(automatic)"),
]
OUTPUT_TEMPLATES = [
    ("pdb", "Structure (PDB)", {"name": "structure", "kind": "pdb"}),
    ("text", "Text file", {"name": "result", "kind": "text"}),
    ("secstruct", "Secondary structure", {"name": "secstruct", "kind": "secstruct"}),
    ("map", "Density map", {"name": "map", "kind": "map"}),
    ("image", "Image / movie", {"name": "image", "kind": "image"}),
]


def env_name(name: str) -> str:
    """``structure-fixed`` -> ``STRUCTURE_FIXED`` (the part after MDATK_IN_ / MDATK_OUT_)."""
    return name.strip().upper().replace("-", "_")


def plain_file_name(name: str) -> bool:
    """A file name without folders, spaces or shell characters (no leading '.' or '-')."""
    return bool(_PLAIN_FILE.match(name)) and len(name) <= 200


def read_script(path: str) -> Tuple[str, str]:
    """(text, problem) of a script file; ``problem`` is "" when the file can be copied."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        return "", f"script file not found: {path}"
    if not os.path.isfile(path):
        return "", f"not a file: {path}"
    try:
        if os.path.getsize(path) > MAX_SCRIPT_BYTES:
            return "", (f"{os.path.basename(path)} is larger than {MAX_SCRIPT_BYTES >> 20} MB; feed data "
                        "files through an input instead")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        return "", f"{os.path.basename(path)} is not a UTF-8 text file"
    except OSError as exc:
        return "", f"cannot read the script file: {exc}"
    if "\0" in text:
        return "", f"{os.path.basename(path)} is not a text file"
    return text, ""


@register
class UserScriptTool(Tool):
    name = "user-script"
    label = "Custom script"
    category = "Custom"
    tag = "custom"
    path_settings = ("script_file",)
    description = ("Your own bash, Python or other script as a step: declare the files it reads and writes, "
                   "connect them like any other step's, and MDATK passes their names to the script and stops "
                   "when a declared output is missing.")
    sections = [
        Section("script", "Script", [
            Option("source", "Script", "radio", "inline", "Type the script here, or copy a script file into "
                   "the step folder each time the step's files are written (edit the original, then write "
                   "the step again).", choices=[("inline", "typed here"), ("file", "from a file")]),
            Option("script", "Script text", "multiline", "",
                   "The script. It runs in the step folder; $MDATK_IN_<NAME> holds each input's path "
                   "(relative to the step folder, one per line for several files) and $MDATK_OUT_<NAME> each "
                   "output's file name.",
                   placeholder='cp "$MDATK_IN_STRUCTURE" "$MDATK_OUT_RESULT"', requires={"source": "inline"},
                   rows=14),
            Option("script_file", "Script file", "file", "", "The script to copy into the step folder. "
                   "Relative paths are read from the folder of the saved project.",
                   requires={"source": "file"}),
            Option("script_name", "Name in the step folder", "text", "",
                   "File name of the copy. Empty: <project>_<step>_script_<time> with .sh / .py (.txt for "
                   "another command) for a typed script, the file's own name otherwise. Set it when the "
                   "program needs an extension, e.g. .pml for PyMOL or .cxc for ChimeraX.",
                   placeholder="(automatic)"),
            Option("interpreter", "Run with", "radio", "bash", "The program that runs the script.",
                   choices=INTERPRETERS),
            Option("command", "Command", "text", "",
                   "Program and options that run the script, e.g. 'Rscript' or 'python3.12 -u'. The script's "
                   "name goes where {script} is, otherwise at the end; the arguments follow.",
                   placeholder="vmd -dispdev text -e {script} -args", requires={"interpreter": "custom"}),
            Option("arguments", "Arguments", "text", "",
                   "Command-line arguments (shell syntax). {in:<name>} becomes the input's path (one word "
                   "per file), {out:<name>} the output's file name and {script} the script's name; do not "
                   "put quotes around them.", placeholder="--model {in:structure} -o {out:result}"),
        ], tooltip="What runs. The step's run script exports the file names, runs this script in the step "
                   "folder and checks that every declared output exists and is not empty."),
        Section("inputs", "Input files", INPUT_OPTIONS, repeat=True, item_label="name",
                templates=INPUT_TEMPLATES,
                tooltip="Files the script reads; each one is an input port that other steps connect to."),
        Section("outputs", "Output files", OUTPUT_OPTIONS, repeat=True, item_label="name",
                templates=OUTPUT_TEMPLATES,
                tooltip="Files the script must write into the step folder; each one is an output port."),
        env_section("conda activate myenv   or   module load R/4.3"),
    ]

    # ------------------------------------------------------------ ports
    @staticmethod
    def _records(s, key: str) -> List[Tuple[int, Dict]]:
        """(index, record) of the declared ports with a usable, not repeated name."""
        out, seen = [], set()
        for k, rec in enumerate(s.get(key) or []):
            name = str(rec.get("name") or "").strip()
            if _PORT_NAME.match(name) and env_name(name) not in seen:
                seen.add(env_name(name))
                out.append((k, rec))
        return out

    def input_ports(self, s):
        ports = []
        for _k, rec in self._records(s, "inputs"):
            name = rec["name"].strip()
            kinds = ANY_KINDS if rec.get("kind") == "any" else (rec.get("kind") or "text",)
            ports.append(Port(name, name, kinds, optional=bool(rec.get("optional")),
                              multiple=bool(rec.get("multiple")),
                              tooltip=f"The script reads it as $MDATK_IN_{env_name(name)} or {{in:{name}}}."))
        return ports

    def output_ports(self, s):
        return [Port(rec["name"].strip(), rec["name"].strip(), (rec.get("kind") or "text",),
                     tooltip=f"The script writes it to $MDATK_OUT_{env_name(rec['name'])} or "
                             f"{{out:{rec['name'].strip()}}}.")
                for _k, rec in self._records(s, "outputs")]

    # ------------------------------------------------------------ names
    @staticmethod
    def _output_file(rec, ctx) -> str:
        """File name (in the step folder) of a declared output."""
        given = str(rec.get("file") or "").strip()
        if given:
            return given
        exts = FILE_KINDS.get(rec.get("kind") or "text", ("", (".txt",)))[1]
        return ctx.name(rec["name"].strip(), exts[0] if exts else "")

    def _outputs(self, s, ctx) -> Dict[str, str]:
        """Output name -> file name, for the outputs with a plain file name."""
        out = {}
        for _k, rec in self._records(s, "outputs"):
            fname = self._output_file(rec, ctx)
            if plain_file_name(fname):
                out[rec["name"].strip()] = fname
        return out

    def output_values(self, s, ctx):
        kinds = {p.name: p.kind for p in self.output_ports(s)}
        return {name: PortValue([ctx.from_own(fname)], kinds[name])
                for name, fname in self._outputs(s, ctx).items()}

    @staticmethod
    def _source_path(s) -> str:
        path = (s.get("script_file") or "").strip()
        return os.path.abspath(os.path.expanduser(path)) if path else ""

    def script_name(self, s, ctx) -> str:
        """Name of the script's copy in the step folder."""
        given = (s.get("script_name") or "").strip()
        if given:
            return given
        if s.get("source") == "file" and self._source_path(s):
            base = re.sub(r"[^A-Za-z0-9._+-]", "_", os.path.basename(self._source_path(s))).lstrip(".-")
            return base or ctx.name("script", EXTENSIONS["custom"])
        return ctx.name("script", EXTENSIONS.get(s.get("interpreter"), "txt"))

    def _text(self, s) -> Tuple[str, str]:
        """(script text, problem)."""
        if s.get("source") == "file":
            path = self._source_path(s)
            return read_script(path) if path else ("", "choose the script file")
        return s.get("script") or "", ""

    def _runner(self, s) -> str:
        if s.get("interpreter") == "custom":
            words = (s.get("command") or "").split()
            return words[0] if words else "command"
        return s.get("interpreter") or "bash"

    # ------------------------------------------------------------ checks
    def check(self, s, ctx):
        out: List[Issue] = []
        text, problem = self._text(s)
        if problem:
            out.append(Issue("error", problem, ("script_file",)))
        elif not text.strip():
            key = "script_file" if s.get("source") == "file" else "script"
            out.append(Issue("error", "the script is empty" if key == "script" else
                             f"the script file is empty: {self._source_path(s)}", (key,)))
        else:
            out += self._shebang(s, text)
        if s.get("interpreter") == "custom" and not (s.get("command") or "").strip():
            out.append(Issue("error", "give the command that runs the script", ("command",)))
        given = (s.get("script_name") or "").strip()
        if given and not plain_file_name(given):
            out.append(Issue("error", f"'{given}' is not a plain file name: {PLAIN_RULE}", ("script_name",)))
        names = {"inputs": set(), "outputs": set()}
        for key, what, var in (("inputs", "input", "IN"), ("outputs", "output", "OUT")):
            seen: Dict[str, str] = {}
            for k, rec in enumerate(s.get(key) or []):
                name = str(rec.get("name") or "").strip()
                where = (f"{key}.{k}.name",)
                if not name:
                    out.append(Issue("error", f"{what} {k + 1} has no name", where))
                elif not _PORT_NAME.match(name):
                    out.append(Issue("error", f"{what} name '{name}': use letters, digits, '_' and '-', "
                                              "starting with a letter", where))
                elif env_name(name) in seen:
                    out.append(Issue("error", f"two {what}s are named '{seen[env_name(name)]}'"
                                     + ("" if seen[env_name(name)] == name else f" / '{name}'")
                                     + f" (both would be $MDATK_{var}_{env_name(name)})", where))
                else:
                    seen[env_name(name)] = name
                    names[key].add(name)
        out += self._check_outputs(s, ctx)
        out += self._check_placeholders(s, names)
        if not (s.get("outputs") or []):
            out.append(Issue("info", "no output files declared: later steps cannot connect to this step, and "
                                     "only the script's exit status tells whether it worked"))
        return out

    def _shebang(self, s, text: str) -> List[Issue]:
        first = text.lstrip("\ufeff").split("\n", 1)[0]
        if not first.startswith("#!") or s.get("interpreter") == "custom":
            return []
        python = "python" in first
        shell = not python and any(w in first for w in ("bash", "/sh", " sh"))
        if (s["interpreter"] == "bash" and python) or (s["interpreter"] == "python3" and shell):
            return [Issue("warning", f"the script starts with '{first.strip()}' but runs with "
                                     f"{s['interpreter']}", ("interpreter",))]
        return []

    def _check_outputs(self, s, ctx) -> List[Issue]:
        out: List[Issue] = []
        run, script = ctx.name(self.tag, "sh"), self.script_name(s, ctx)
        if script == run:
            out.append(Issue("error", f"{script} is the name of the step's run script; choose another name",
                             ("script_name",)))
        taken = {run: "the step's run script", script: "the script's copy"}
        for k, rec in self._records(s, "outputs"):
            name = rec["name"].strip()
            fname = self._output_file(rec, ctx)
            given = bool(str(rec.get("file") or "").strip())
            where = (f"outputs.{k}.file",) if given else (f"outputs.{k}.name",)
            if not plain_file_name(fname):
                out.append(Issue("error", f"output '{name}': '{fname}' is not a plain file name: "
                                          f"{PLAIN_RULE}", where))
                continue
            if fname in taken:
                out.append(Issue("error", f"output '{name}': {fname} is also {taken[fname]}", where))
                continue
            taken[fname] = f"output '{name}'s file"
            exts = FILE_KINDS.get(rec.get("kind"), ("", ()))[1]
            if given and exts and not fname.lower().endswith(tuple(e.lower() for e in exts)):
                out.append(Issue("info", f"output '{name}': {fname} does not end in {' / '.join(exts)} "
                                         f"({kind_label(rec['kind'])})", where))
        return out

    def _check_placeholders(self, s, names: Dict[str, set]) -> List[Issue]:
        out: List[Issue] = []
        fields = [("arguments", "arguments")]
        if s.get("interpreter") == "custom":
            fields.append(("command", "command"))
        for key, what in fields:
            text = s.get(key) or ""
            for m in _MISSPELT.finditer(text):
                side, name = m.group(1), m.group(2)
                if side in ("in", "out"):
                    declared = names["inputs" if side == "in" else "outputs"]
                    if name not in declared:
                        kind = "input" if side == "in" else "output"
                        out.append(Issue("error", f"{what}: {m.group(0)} names no declared {kind} (declared: "
                                                  f"{', '.join(sorted(declared)) or 'none'})", (key,)))
                else:
                    right = f"{{{'in' if side.lower().startswith('in') else 'out'}:{name}}}"
                    out.append(Issue("error", f"{what}: unknown placeholder {m.group(0)}; write {right}",
                                     (key,)))
        return out

    # ------------------------------------------------------------ files
    def _expand(self, text: str, s, ctx, script: str) -> str:
        outputs = self._outputs(s, ctx)

        def sub(m):
            if m.group(0) == "{script}":
                return shell_quote(script)
            if m.group(1) == "in":
                paths = ctx.inps(m.group(2))
                return " ".join(shell_quote(p) for p in paths) if paths else "''"
            name = m.group(2)
            return shell_quote(outputs[name]) if name in outputs else m.group(0)  # an error of check()

        return _PLACEHOLDER.sub(sub, text)

    def command(self, s, ctx) -> str:
        """The line of the run script that starts the user's script."""
        script = self.script_name(s, ctx)
        if s.get("interpreter") == "custom":
            cmd = (s.get("command") or "").strip()
            words = self._expand(cmd, s, ctx, script) if _SCRIPT.search(cmd) else \
                f"{self._expand(cmd, s, ctx, script)} {shell_quote(script)}"
        else:
            words = f"{s.get('interpreter') or 'bash'} {shell_quote(script)}"
        args = (s.get("arguments") or "").strip()
        return f"{words} {self._expand(args, s, ctx, script)}" if args else words

    def generate(self, s, ctx):
        text, problem = self._text(s)
        if problem:  # the file changed after check(): writing reports it like any other unreadable file
            raise OSError(problem)
        script = self.script_name(s, ctx)
        run = ctx.name(self.tag, "sh")
        origin = f"a copy of {self._source_path(s)}" if s.get("source") == "file" else "typed into the step"
        title = ctx.title.split(" · ", 1)[-1] if ctx.title else ""  # the box title without the step number
        heading = "Custom script" + (f": {title}" if title and title != self.label else "")
        sh = script_header(heading, ctx, s, run, [
            f"Runs {script} ({origin}) with {self._runner(s)}, then checks its declared outputs.",
            "Inputs: $MDATK_IN_<NAME>, paths relative to this folder (one per line for several files).",
            "Outputs: $MDATK_OUT_<NAME>, file names in this folder."])
        sh += ["# Names of this plan and step",
               f"export MDATK_PROJECT={shell_quote(ctx.project)} MDATK_STEP={ctx.step:02d} "
               f"MDATK_STAMP={shell_quote(ctx.stamp)}"]
        inputs = self.input_ports(s)
        if inputs:
            sh += ["", "# Input files (relative to this folder)"]
            sh += [f"export MDATK_IN_{env_name(p.name)}={shell_quote(chr(10).join(ctx.inps(p.name)))}"
                   for p in inputs]
        outputs = self._outputs(s, ctx)
        if outputs:
            sh += ["", "# Output files the script must write (in this folder)"]
            sh += [f"export MDATK_OUT_{env_name(name)}={shell_quote(fname)}"
                   for name, fname in outputs.items()]
        sh += ["", "# The script", self.command(s, ctx)]
        if outputs:
            sh += ["", "# Every declared output must exist and not be empty"]
            sh += [require_file(fname, script) for fname in outputs.values()]
        sh.append("")
        what = f"your script ({origin})"
        return [GeneratedFile(script, text, "helper", what),
                GeneratedFile(run, "\n".join(sh), "run", f"runs {script} and checks its outputs")]

    def summary_lines(self, s, ctx) -> List[str]:
        return [f"Runs: `{self.command(s, ctx)}`"]

    def describe(self, s):
        if s.get("source") == "file":
            what = os.path.basename((s.get("script_file") or "").strip()) or "(no file chosen)"
        else:
            n = len([ln for ln in (s.get("script") or "").splitlines() if ln.strip()])
            what = f"typed script, {n} line{'s' if n != 1 else ''}" if n else "(no script)"
        return f"{self._runner(s)}: {what}"

