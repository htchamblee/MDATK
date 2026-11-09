"""Started without a terminal (the applications menu), the scripts MDATK runs get the environment of an
interactive bash, so what ~/.bashrc sets up after its "not interactive: return" guard (pmemd26's amber.sh,
for pmemd.cuda) reaches them, as when MDATK is started from a terminal."""
import os
import shutil

import pytest

from mdatk.workflows import runner

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

BASHRC = """# ~/.bashrc as distributions write it: nothing for non-interactive shells
case $- in
    *i*) ;;
      *) return;;
esac
echo "welcome text that must not reach the environment"
source "$HOME/pmemd26/amber.sh"
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A HOME whose ~/.bashrc puts a fake pmemd.cuda on PATH (interactive shells only)."""
    amber = tmp_path / "pmemd26"
    (amber / "bin").mkdir(parents=True)
    prog = amber / "bin" / "pmemd.cuda"
    prog.write_text("#!/bin/bash\necho fake pmemd.cuda\n")
    prog.chmod(0o755)
    (amber / "amber.sh").write_text(f'export AMBERHOME="{amber}"\nexport PATH="$AMBERHOME/bin:$PATH"\n')
    (tmp_path / ".bashrc").write_text(BASHRC)
    monkeypatch.setenv("HOME", str(tmp_path))
    runner._shell_env_cache.clear()
    yield tmp_path
    runner._shell_env_cache.clear()


def script(tmp_path):
    path = tmp_path / "step.sh"
    path.write_text("#!/bin/bash\ncommand -v pmemd.cuda || echo 'pmemd.cuda: not found'\n"
                    'echo "AMBERHOME=${AMBERHOME:-}"\n')
    return str(path)


def test_started_without_terminal():
    assert runner.started_without_terminal({"MDATK_SHELL_ENV": "1", "TERM": "xterm"})
    assert not runner.started_without_terminal({"MDATK_SHELL_ENV": "0"})
    assert not runner.started_without_terminal({"TERM": "xterm-256color"})


@needs_bash
def test_a_script_gets_what_bashrc_sets_up_when_mdatk_has_no_terminal(home, monkeypatch):
    lines = []
    monkeypatch.setenv("MDATK_SHELL_ENV", "1")
    code, out = runner.run_script(script(home), lines.append)
    assert code == 0, out
    assert os.path.join(str(home), "pmemd26", "bin", "pmemd.cuda") in out
    assert f"AMBERHOME={home / 'pmemd26'}" in out
    assert "welcome text" not in out  # the shell's own output never reaches the environment or the script
    assert any("started without a terminal" in ln for ln in lines)


@needs_bash
def test_from_a_terminal_the_script_keeps_mdatk_s_environment(home, monkeypatch):
    lines = []
    monkeypatch.setenv("MDATK_SHELL_ENV", "0")
    monkeypatch.delenv("AMBERHOME", raising=False)
    code, out = runner.run_script(script(home), lines.append)
    assert code == 0, out
    assert "pmemd.cuda: not found" in out and "AMBERHOME=\n" in out + "\n"
    assert not any("started without a terminal" in ln for ln in lines)


@needs_bash
def test_variables_mdatk_sets_win_over_the_shell_s(home, monkeypatch):
    monkeypatch.setenv("MDATK_SHELL_ENV", "1")
    path = home / "gpu.sh"
    path.write_text('#!/bin/bash\necho "GPU=$CUDA_VISIBLE_DEVICES LOCAL=$MDATK_LOCAL"\n')
    code, out = runner.run_script(str(path), lambda _l: None, env={"CUDA_VISIBLE_DEVICES": "1"})
    assert code == 0 and "GPU=1 LOCAL=1" in out


@needs_bash
def test_a_shell_that_cannot_answer_warns_and_keeps_the_environment(home, monkeypatch):
    (home / ".bashrc").write_text(BASHRC.replace('source "$HOME/pmemd26/amber.sh"', "exit 3"))
    monkeypatch.setenv("MDATK_SHELL_ENV", "1")
    lines = []
    code, out = runner.run_script(script(home), lines.append)
    assert code == 0 and "pmemd.cuda: not found" in out
    assert any(ln.startswith("WARNING: MDATK was started without a terminal") for ln in lines)
