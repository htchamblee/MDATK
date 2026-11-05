"""Scheduling of generated scripts: dependencies, parallel runs, failures (mdatk.workflows.runner)."""

import os
import shutil
import threading
import time

import pytest

from mdatk.progress import Cancelled
from mdatk.workflows.runner import ScriptFailed, ScriptJob, run_jobs, run_script

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash")


def job(tmp_path, step, node, after=(), body="sleep 0.6", name=None):
    path = tmp_path / f"{name or node}.sh"
    path.write_text(f'cd "$(dirname "$0")"\necho begin {node} $(date +%s.%N)\n{body}\n'
                    f'echo end {node} $(date +%s.%N)\n')
    return ScriptJob(step, node, str(path), node_id=node, after=list(after))


def times(lines):
    out = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] in ("begin", "end"):
            out[(parts[0], parts[1])] = float(parts[2])
    return out


def run(jobs, parallel, **kw):
    lines, states = [], []
    lock = threading.Lock()

    def on_line(line):
        with lock:
            lines.append(line.split("] ", 1)[-1])

    def on_state(j, s):
        with lock:
            states.append((j.node_id, s))

    try:
        return run_jobs(jobs, None, on_line=on_line, parallel=parallel, on_state=on_state, **kw), lines, states
    finally:
        run.lines, run.states = lines, states


def test_independent_steps_run_concurrently_and_dependents_wait(tmp_path):
    # a and b are independent; c reads from both
    jobs = [job(tmp_path, 1, "a"), job(tmp_path, 2, "b"), job(tmp_path, 3, "c", after=["a", "b"])]
    t0 = time.monotonic()
    results, lines, states = run(jobs, parallel=2)
    assert [r.status for r in results].count("ok") == 3
    t = times(lines)
    assert t[("begin", "b")] < t[("end", "a")]  # a and b overlapped
    assert t[("begin", "c")] >= max(t[("end", "a")], t[("end", "b")])
    assert time.monotonic() - t0 < 1.6  # 0.6 + 0.6, not 3 x 0.6
    assert ("a", "queued") in states and states.index(("a", "running")) < states.index(("a", "ok"))


def test_parallel_one_keeps_plan_order(tmp_path):
    jobs = [job(tmp_path, 1, "a", body="sleep 0.2"), job(tmp_path, 2, "b", body="sleep 0.2")]
    _results, lines, _states = run(jobs, parallel=1)
    t = times(lines)
    assert t[("begin", "b")] >= t[("end", "a")]
    assert not any(line.startswith("[0") for line in run.lines)


def test_failure_skips_only_dependent_steps(tmp_path):
    jobs = [job(tmp_path, 1, "a", body="exit 3"), job(tmp_path, 2, "b", after=["a"]),
            job(tmp_path, 3, "c", body="sleep 0.2")]
    with pytest.raises(ScriptFailed, match=r"a\.sh \(step 01\) failed: exit status 3; 1 dependent"):
        run(jobs, parallel=1)
    assert ("b", "skipped") in run.states and ("c", "ok") in run.states and ("a", "failed") in run.states
    assert not any(line.startswith("begin b") for line in run.lines)


def test_jobs_without_step_ids_run_strictly_in_order(tmp_path):
    a = job(tmp_path, 1, "a", body="exit 1")
    b = job(tmp_path, 2, "b")
    a.node_id = b.node_id = ""
    with pytest.raises(ScriptFailed):
        run([a, b], parallel=4)
    assert not any(line.startswith("begin b") for line in run.lines)


def test_cancel_stops_every_running_script(tmp_path):
    jobs = [job(tmp_path, 1, "a", body="sleep 20"), job(tmp_path, 2, "b", body="sleep 20"),
            job(tmp_path, 3, "c", after=["a"])]
    cancel = threading.Event()
    threading.Timer(0.8, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run(jobs, parallel=2, cancel=cancel)
    assert time.monotonic() - t0 < 12
    assert {s for n, s in run.states if n in ("a", "b")} >= {"cancelled"}
    assert ("c", "cancelled") in run.states and not any(line.startswith("begin c") for line in run.lines)


def test_plan_jobs_dependencies():
    from mdatk.workflows.plan import Plan
    from mdatk.workflows.runner import plan_jobs
    from mdatk.config import WorkflowParams
    from mdatk.workflows.plan import apply_template
    wf = WorkflowParams(project_name="apt1", stamp="20261015-1432")
    apply_template(wf, "amber")
    nodes = {n.tool: n for n in wf.nodes}
    nodes["amber-build"].files["structure"] = "/x.pdb"
    nodes["amber-restraints"].settings.update(sequence="GGGCAC GUGCCC", dotbracket="(((((( ))))))")
    plan = Plan(wf, "/tmp/mdatk-plan")
    jobs = {j.node_id: j for j in plan_jobs(plan)}
    b, r, md = nodes["amber-build"].id, nodes["amber-restraints"].id, nodes["amber-md"].id
    assert jobs[b].after == [] and jobs[r].after == [b] and set(jobs[md].after) == {b, r}
    # with only build and MD selected, MD still waits for the build (through the restraints step)
    sub = {j.node_id: j for j in plan_jobs(plan, [b, md])}
    assert sub[md].after == [b]


def test_broken_output_stops_every_script_and_does_not_hang(tmp_path):
    # e.g. `mdatk workflow run ... | head -3`: printing raises BrokenPipeError once head exits
    jobs = [job(tmp_path, 1, "a", body="sleep 4; touch a.done"), job(tmp_path, 2, "b", body="sleep 4; touch b.done"),
            job(tmp_path, 3, "c", after=["a"])]
    count = {"n": 0}

    def on_line(line):
        count["n"] += 1
        if count["n"] > 3:
            raise BrokenPipeError(32, "Broken pipe")

    t0 = time.monotonic()
    with pytest.raises(BrokenPipeError):
        run_jobs(jobs, None, on_line=on_line, parallel=2)
    assert time.monotonic() - t0 < 12
    time.sleep(4.5)
    assert not (tmp_path / "a.done").exists() and not (tmp_path / "b.done").exists()  # both stopped


def test_failure_message_keeps_the_reason(tmp_path):
    missing = ScriptJob(1, "gone", str(tmp_path / "gone.sh"), node_id="g")
    with pytest.raises(ScriptFailed, match=r"gone\.sh \(step 01\) failed: .*does not exist"):
        run_jobs([missing], None, on_line=lambda line: None)


def test_step_marks_and_stderr_apart(tmp_path):
    """RNAdo's log panel: every line marked with its step, stderr apart (and still in the step log)."""
    from mdatk.projectfile import read_log
    from mdatk.config import Project

    project = tmp_path / "p.mdatk.toml"
    Project().save(str(project))
    script = tmp_path / "s.sh"
    script.write_text('echo "to stdout"\necho "to stderr" >&2\necho "ERROR: bad" >&2\nexit 0\n')
    out, err = [], []
    results = run_jobs([ScriptJob(7, "Step seven", str(script), node_id="n7")], str(project),
                       on_line=out.append, on_stderr=err.append, step_marks=True)
    assert [r.status for r in results] == ["ok"]
    assert "[07] to stdout" in out and "[07] to stderr" not in out
    assert err == ["[07] to stderr", "[07] ERROR: bad"]
    assert any(line.startswith("==== step 07: Step seven - bash s.sh") for line in out)
    entry = read_log(str(project))[-1]
    assert entry["step"] == "07"
    assert all(t in entry["output"] for t in ("to stdout", "to stderr", "ERROR: bad"))
    # by default (one script at a time) nothing changes: no marks, stderr merged into on_line
    lines = []
    run_jobs([ScriptJob(7, "Step seven", str(script), node_id="n7")], None, on_line=lines.append)
    assert "to stdout" in lines and "to stderr" in lines and "ERROR: bad" in lines


def test_a_broken_stderr_callback_stops_the_script(tmp_path):
    script = tmp_path / "s.sh"
    script.write_text('echo "first" >&2\nsleep 5\necho "never"\n')

    def broken(_line):
        raise BrokenPipeError("closed")

    t0 = time.monotonic()
    with pytest.raises(BrokenPipeError):
        run_jobs([ScriptJob(1, "s", str(script), node_id="n")], None, on_line=lambda _l: None,
                 on_stderr=broken)
    assert time.monotonic() - t0 < 4


# ---------------------------------------------------------------------------
# CPU / GPU budgets and stopping one step (0.0.12)
# ---------------------------------------------------------------------------


def test_gpus_are_handed_out_in_cuda_visible_devices(tmp_path, monkeypatch):
    from mdatk.workflows.budget import Budget, Resources
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    body = 'echo "gpu ${CUDA_VISIBLE_DEVICES-unset}"; sleep 0.6'
    gpu = Resources(1, 1)
    jobs = [job(tmp_path, k + 1, n, body=body) for k, n in enumerate("abcde")]
    for j in jobs[:3]:
        j.resources = gpu
    jobs[3].resources = Resources(1, 1, ("7",))  # names its own GPU: the environment is left alone
    jobs[4].resources = Resources(2)  # a CPU step
    out = []
    results = run_jobs(jobs, None, on_line=out.append, parallel=0, budget=Budget(8, ("0", "1")))
    assert all(r.status == "ok" for r in results)
    seen = {}
    for line in out:
        mark, _, text = line.partition("] ")
        if text.startswith("gpu "):
            seen[int(mark.strip("["))] = text.split()[1]
    assert seen[1] == "0" and seen[2] == "1"  # plan order, the first free GPU
    assert seen[3] in ("0", "1")  # waited for one of them
    assert seen[4] == "unset" and seen[5] == "unset"
    assert any(line.startswith("==== step 01: a - bash a.sh (1 core, GPU 0)") for line in out)
    assert any("step 04: d - bash d.sh (1 core, GPU 7 (named by the step))" in line for line in out)
    assert any(line.endswith("e.sh (2 cores)") for line in out)
    assert any(line.startswith("WARNING: step 04 (d) names GPU 7") for line in out)
    t = times([line.split("] ", 1)[-1] for line in out])
    assert t[("begin", "c")] >= min(t[("end", "a")], t[("end", "b")])  # two GPUs: c waited for one


def test_a_budget_of_one_core_runs_the_steps_in_plan_order(tmp_path):
    from mdatk.workflows.budget import Budget
    jobs = [job(tmp_path, k + 1, n, body="sleep 0.2") for k, n in enumerate("abc")]
    states = []
    run_jobs(jobs, None, on_line=lambda _l: None, parallel=0, budget=Budget(1),
             on_state=lambda j, s: states.append((j.node_id, s)))
    assert [n for n, s in states if s == "running"] == ["a", "b", "c"]


def ended(pid, timeout=5.0):
    """The process is gone, or a zombie its new parent has not reaped yet (an orphaned child)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            with open(f"/proc/{pid}/stat") as fh:
                if fh.read().rsplit(")", 1)[-1].split()[0] == "Z":
                    return True
        except OSError:  # gone meanwhile (or no /proc: ask again)
            pass
        time.sleep(0.05)
    return False


def test_a_stopped_step_skips_its_dependents_and_the_others_finish(tmp_path):
    from mdatk.workflows.budget import Budget
    from mdatk.workflows.runner import StepStops
    # a starts a child process too: stopping the step ends its whole process group
    a = job(tmp_path, 1, "a", body='echo $$ > a.pid\nsleep 30 &\necho $! > a.child\nwait')
    a.stop_mark = str(tmp_path / "01_a" / "apt1_01_stopped.txt")
    b = job(tmp_path, 2, "b", after=["a"])
    c = job(tmp_path, 3, "c", body="sleep 1.5; touch c.done")
    stops = StepStops()

    def stop_when_running():
        end = time.monotonic() + 10
        while time.monotonic() < end and not (tmp_path / "a.child").exists():
            time.sleep(0.05)
        stops.stop("a")

    threading.Thread(target=stop_when_running, daemon=True).start()
    t0 = time.monotonic()
    results, lines, states = run([a, b, c], parallel=0, budget=Budget(4), stops=stops)
    assert time.monotonic() - t0 < 12
    status = {r.job.node_id: r.status for r in results}
    assert status == {"a": "stopped", "b": "skipped", "c": "ok"}
    assert (tmp_path / "c.done").exists() and ("a", "stopped") in states
    assert "==== step 02: b - not started (step 01 was stopped)" in lines
    for pid_file in ("a.pid", "a.child"):
        assert ended(int((tmp_path / pid_file).read_text())), pid_file
    note = (tmp_path / "01_a" / "apt1_01_stopped.txt").read_text()
    assert "step 01 (a) was stopped by the user" in note
    # the next run of the step removes the note
    a2 = job(tmp_path, 1, "a", body="true", name="a2")
    a2.stop_mark = a.stop_mark
    run_jobs([a2], None, on_line=lambda _l: None)
    assert not os.path.exists(a.stop_mark)


def test_a_step_stopped_before_it_starts(tmp_path):
    from mdatk.workflows.budget import Budget
    from mdatk.workflows.runner import StepStops
    a = job(tmp_path, 1, "a", body="sleep 0.8")
    b = job(tmp_path, 2, "b", after=["a"], body="touch b.ran")
    c = job(tmp_path, 3, "c", after=["b"])
    stops = StepStops()
    stops.stop("b")
    results, lines, _states = run([a, b, c], parallel=0, budget=Budget(4), stops=stops)
    assert {r.job.node_id: r.status for r in results} == {"a": "ok", "b": "stopped", "c": "skipped"}
    assert not (tmp_path / "b.ran").exists()
    assert "==== step 03: c - not started (step 02 was stopped)" in lines


def test_a_stopped_run_counts_as_partial(tmp_path):
    from mdatk.config import WorkflowParams
    from mdatk.workflows import runstate
    from mdatk.workflows.plan import Plan, add_node
    from mdatk.workflows.runner import StepStops, plan_jobs
    wf = WorkflowParams(project_name="apt1", stamp="20261015-1432")
    node = add_node(wf, "user-script", settings={
        "script": 'echo partial > "$MDATK_OUT_RESULT"\nsleep 30\n',
        "outputs": [{"name": "result", "kind": "text", "file": "result.txt"}]})
    plan = Plan(wf, str(tmp_path / "plan"))
    plan.write()
    jobs = plan_jobs(plan)
    assert jobs[0].stop_mark.endswith("01_custom/apt1_01_stopped_20261015-1432.txt")
    stops = StepStops()
    folder = os.path.join(plan.root, "01_custom")

    def stop_when_written():
        end = time.monotonic() + 10
        while time.monotonic() < end and not os.path.exists(os.path.join(folder, "result.txt")):
            time.sleep(0.05)
        time.sleep(0.2)
        stops.stop(node.id)

    threading.Thread(target=stop_when_written, daemon=True).start()
    results = run_jobs(jobs, None, on_line=lambda _l: None, parallel=0, stops=stops)
    assert [r.status for r in results] == ["stopped"]
    found = runstate.check_runs(Plan(wf, plan.root))[node.id]
    assert found.state == "partial" and found.stopped and "stopped before it finished" in found.summary()
    # files written after the stop (the step ran again by hand) count as they are
    with open(os.path.join(folder, "apt1_01_custom_20261015-1432.done"), "w") as fh:
        fh.write("MDATK: custom script finished\n")
    later = time.time() + 5
    for name in os.listdir(folder):
        os.utime(os.path.join(folder, name), (later, later))
    os.utime(jobs[0].stop_mark, (later - 60, later - 60))
    found = runstate.check_runs(Plan(wf, plan.root))[node.id]
    assert found.state == "done" and not found.stopped


# ---------------------------------------------------------------------------
# conda in scripts run from RNAdo (conda init sets PATH for interactive shells only)
# ---------------------------------------------------------------------------

CONDA_STAND_IN = """#!/bin/bash
if [ "$1" = shell.bash ] && [ "$2" = hook ]; then
  echo 'conda() { if [ "$1" = activate ]; then echo "activated $2"; fi; }'
fi
"""


def _conda_script(tmp_path):
    script = tmp_path / "s.sh"
    script.write_text('set -e\neval "$(conda shell.bash hook)"\nconda activate AmberTools26\necho done\n')
    return str(script)


def test_a_script_finds_conda_that_is_not_on_path(tmp_path, monkeypatch):
    # RNAdo started from the applications menu has no conda on PATH; a step with a conda environment
    # failed with "conda: command not found" (exit status 127)
    home = tmp_path / "home"
    conda = home / "miniforge3" / "condabin" / "conda"
    conda.parent.mkdir(parents=True)
    conda.write_text(CONDA_STAND_IN)
    conda.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    for var in ("CONDA_EXE", "CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    lines = []
    code, out = run_script(_conda_script(tmp_path), lines.append)
    assert code == 0, out
    assert "activated AmberTools26" in lines and "done" in lines
    assert any("using " + str(conda) in line for line in lines)


def test_without_conda_the_script_fails_with_a_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    for var in ("CONDA_EXE", "CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    import mdatk.cleaning.conda as cc
    monkeypatch.setattr(cc, "COMMON_ROOTS", ())
    lines = []
    code, _out = run_script(_conda_script(tmp_path), lines.append)
    assert code != 0
    assert any(line.startswith("WARNING: the script uses conda") for line in lines)


def test_scripts_without_conda_keep_their_environment(tmp_path):
    from mdatk.workflows.runner import conda_on_path
    script = tmp_path / "s.sh"
    script.write_text("# conda activate is only mentioned here\necho hi\n")
    assert conda_on_path(None, str(script)) is None


# ---------------------------------------------------------------------------
# the per-step output log (<script>.log)
# ---------------------------------------------------------------------------

import re  # noqa: E402

STAMPED = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


def test_each_run_leaves_an_output_log_with_a_timestamped_end(tmp_path):
    from mdatk.workflows.runner import DONE_FLAG, step_log_path
    ok = tmp_path / "ok.sh"
    ok.write_text('echo "stage 1"\necho "a warning" >&2\necho "stage 2"\n')
    bad = tmp_path / "bad.sh"
    bad.write_text('echo "starting"\nexit 3\n')
    run_jobs([ScriptJob(1, "one", str(ok), node_id="a")], None, on_line=lambda _l: None,
             on_stderr=lambda _l: None)
    text = open(step_log_path(str(ok))).read().splitlines()
    assert re.fullmatch(rf"==== ok\.sh - started {STAMPED} by MDATK \(bash ok\.sh, in .*\)", text[0])
    body = text[1:-1]  # every line; each stream's in order (standard error is read alongside)
    assert sorted(body) == ["a warning", "stage 1", "stage 2"]
    assert body.index("stage 1") < body.index("stage 2")
    assert re.fullmatch(rf"==== {DONE_FLAG} - {STAMPED} \(exit status 0\)", text[-1])
    with pytest.raises(ScriptFailed):
        run_jobs([ScriptJob(2, "two", str(bad), node_id="b")], None, on_line=lambda _l: None)
    text = open(step_log_path(str(bad))).read().splitlines()
    assert "starting" in text and DONE_FLAG not in "\n".join(text)
    assert re.fullmatch(rf"==== Step failed - {STAMPED} \(exit status 3\): exit status 3", text[-1])
    # the next run starts a new log
    run_jobs([ScriptJob(1, "one", str(ok), node_id="a")], None, on_line=lambda _l: None)
    assert open(step_log_path(str(ok))).read().count("started") == 1


def test_a_stopped_step_says_so_in_its_output_log(tmp_path):
    import threading as _th

    from mdatk.workflows.runner import StepStops, step_log_path
    script = tmp_path / "slow.sh"
    script.write_text('echo "begun"\nsleep 20\n')
    stops = StepStops()
    _th.Timer(0.8, lambda: stops.stop("n")).start()
    results = run_jobs([ScriptJob(1, "slow", str(script), node_id="n")], None, on_line=lambda _l: None,
                       stops=stops)
    assert results[0].status == "stopped"
    last = open(step_log_path(str(script))).read().splitlines()[-1]
    assert re.fullmatch(rf"==== Step stopped by the user - {STAMPED}", last)
