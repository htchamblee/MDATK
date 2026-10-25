"""Scheduling of generated scripts: dependencies, parallel runs, failures (mdatk.workflows.runner)."""

import os
import shutil
import threading
import time

import pytest

from mdatk.progress import Cancelled
from mdatk.workflows.runner import ScriptFailed, ScriptJob, run_jobs

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
