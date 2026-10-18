import os
import re
import threading

import pytest

from mdatk import projectfile as PF
from mdatk import tomlio
from mdatk.config import PanelSpec, Project


def _entries(text):
    """(line number, key) of every value line outside multi-line strings / arrays."""
    out, depth, in_str = [], 0, False
    for k, line in enumerate(text.splitlines()):
        if in_str:
            in_str = line.count('"""') % 2 == 0
            continue
        if depth:
            depth += line.count("[") - line.count("]")
            continue
        m = re.match(r'^([A-Za-z0-9_"-]+) = (.*)$', line)
        if m:
            out.append((k, m.group(1)))
            rest = m.group(2)
            if rest.startswith('"""') and rest.count('"""') == 1:
                in_str = True
            elif rest.startswith("[") and rest.count("[") > rest.count("]"):
                depth = rest.count("[") - rest.count("]")
    return out


def test_every_entry_is_commented(tmp_path):
    p = Project()
    p.notes = "Aim: compare\nthree \"quoted\" runs"
    from mdatk.workflows.plan import add_node
    node = add_node(p.workflow, "amber-build", settings={"mg_count": 4})
    node.files["structure"] = "model.pdb"
    p.layout.panels[1].options = {"smooth": 5, "progressive": True, "custom": 1}
    path = tmp_path / "x.mdatk.toml"
    p.save(str(path))
    text = path.read_text()
    lines = text.splitlines()
    missing = []
    for k, key in _entries(text):
        prev = lines[k - 1] if k else ""
        if not prev.startswith("#"):
            missing.append(key)
    assert not missing, missing
    assert "# key =" not in text
    for line in lines:  # tables carry a comment too (first occurrence)
        if line.startswith("[") and not line.startswith("[["):
            assert lines[lines.index(line) - 1].startswith("#"), line
    assert Project.load(str(path)).notes == p.notes


def test_optional_values_and_round_trip(tmp_path):
    p = Project()
    p.inputs.stop = None
    p.inputs.dt_ps = 2.5
    p.analyses.energy.target_value = None
    p.layout.panels.append(PanelSpec("contact_map", ["contacts/map"], "Map", {"transpose": True}))
    path = tmp_path / "x.mdatk.toml"
    p.save(str(path))
    text = path.read_text()
    assert re.search(r"^# stop =   \(not set\)$", text, re.M)
    q = Project.load(str(path))
    assert q.inputs.stop is None and q.inputs.dt_ps == 2.5
    assert q.to_dict() == Project.load(str(path)).to_dict()
    q.save(str(path))
    assert Project.load(str(path)).to_dict() == q.to_dict()


def test_examples_load_and_rewrite_identically(tmp_path):
    root = os.path.join(os.path.dirname(__file__), "..", "examples")
    for name in ("aptamer_target.mdatk.toml", "rna_plan.mdatk.toml"):
        q = Project.load(os.path.join(root, name))
        out = tmp_path / name
        q.save(str(out))
        assert Project.load(str(out)).to_dict() == q.to_dict()


def test_step_log_append_keep_and_split(tmp_path, monkeypatch):
    path = str(tmp_path / "p.mdatk.toml")
    Project().save(path)
    with PF.logged(path, "02", "tleap build", ["key1"]) as entry:
        entry.command = "bash build.sh"
        entry.output = "x" * (PF.LOG_OUTPUT_CHARS + 50)
    with pytest.raises(RuntimeError):
        with PF.logged(path, "03", "md"):
            raise RuntimeError("pmemd crashed")
    log = PF.read_log(path)
    assert [e["step"] for e in log] == ["02", "03"]
    assert log[0]["status"] == "ok" and log[0]["citations"] == ["key1"]
    assert log[0]["output"].startswith("[...]") and len(log[0]["output"]) < PF.LOG_OUTPUT_CHARS + 10
    assert log[1]["status"] == "failed" and "pmemd crashed" in log[1]["summary"]
    p = Project.load(path)
    p.notes = "edited"
    p.save(path)  # saving the settings keeps the log
    assert len(PF.read_log(path)) == 2 and Project.load(path).notes == "edited"
    # beyond the size limit the log moves to <name>.log.toml
    monkeypatch.setattr(PF, "LOG_SPLIT_BYTES", 1000)
    with PF.logged(path, "04", "analysis"):
        pass
    side = str(tmp_path / "p.log.toml")
    assert os.path.exists(side) and 'log_file = "p.log.toml"' in open(path).read()
    assert PF.LOG_MARKER not in open(path).read()
    assert [e["step"] for e in PF.read_log(path)] == ["02", "03", "04"]
    Project.load(path).save(path)
    with PF.logged(path, "05", "video"):
        pass
    assert [e["step"] for e in PF.read_log(path)] == ["02", "03", "04", "05"]
    assert 'log_file = "p.log.toml"' in open(path).read()


def test_params_log_is_never_logged_into(tmp_path):
    path = str(tmp_path / "MDATK_params_log.toml")
    PF.write_params_log(path, Project(), {"timestamp": "t"})
    before = open(path).read()
    assert PF.is_params_log(path)
    with PF.logged(path, "render"):
        pass
    assert open(path).read() == before


def test_recent_projects(tmp_path):
    paths = []
    for k in range(PF.RECENT_MAX + 2):
        p = tmp_path / f"p{k}.mdatk.toml"
        Project().save(str(p))
        PF.add_recent(str(p))
        paths.append(str(p))
    recent = PF.recent_projects()
    assert recent[0] == paths[-1] and len(recent) == PF.RECENT_MAX
    os.remove(paths[-1])
    assert PF.recent_projects()[0] == paths[-2]  # missing files are skipped


def test_tomlio_values_round_trip():
    data = {"s": 'a "b" \\ c\ttab', "m": "line 1\nline \"2\"\n", "f": 1e-7, "i": -3, "b": True,
            "l": [1, 2.5, "x"], "long": list(range(60)), "t": {"k": None, "v": [{"a": 1}, {"a": 2}]},
            "weird key": 1}
    back = tomlio.loads(tomlio.dumps(data))
    assert back["s"] == data["s"] and back["m"] == data["m"] and back["f"] == data["f"]
    assert back["long"] == data["long"] and back["t"]["v"] == data["t"]["v"] and "k" not in back["t"]
    assert back["weird key"] == 1


def test_runner_runs_logs_and_stops(tmp_path):
    from mdatk.progress import Cancelled
    from mdatk.workflows.runner import ScriptFailed, ScriptJob, run_jobs
    proj = str(tmp_path / "p.mdatk.toml")
    Project().save(proj)
    ok = tmp_path / "ok.sh"
    ok.write_text('cd "$(dirname "$0")"\necho hello; touch made.txt\n')
    bad = tmp_path / "bad.sh"
    bad.write_text("echo oops >&2; exit 3\n")
    never = tmp_path / "never.sh"
    never.write_text("touch never.txt\n")
    lines = []
    with pytest.raises(ScriptFailed, match="status 3"):
        run_jobs([ScriptJob(1, "first", str(ok)), ScriptJob(2, "second", str(bad)),
                  ScriptJob(3, "third", str(never))], proj, on_line=lines.append)
    assert "hello" in lines and "oops" in lines
    assert (tmp_path / "made.txt").exists() and not (tmp_path / "never.txt").exists()
    log = PF.read_log(proj)
    assert [(e["step"], e["status"]) for e in log] == [("01", "ok"), ("02", "failed")]
    assert log[1]["summary"] == "exit status 3" and "oops" in log[1]["output"]
    slow = tmp_path / "slow.sh"
    slow.write_text("echo start; sleep 30; echo end\n")
    cancel = threading.Event()

    def on_line(line):
        if line == "start":
            cancel.set()

    with pytest.raises(Cancelled):
        run_jobs([ScriptJob(4, "slow", str(slow))], proj, on_line=on_line, cancel=cancel)
    assert PF.read_log(proj)[-1]["status"] == "cancelled"


def test_params_log_is_never_overwritten_by_save(tmp_path):
    path = str(tmp_path / "run_log.toml")
    PF.write_params_log(path, Project(), {"timestamp": "t"})
    with pytest.raises(ValueError, match="parameter log"):
        Project().save(path)
    assert PF.is_params_log(path)


def test_concurrent_saves_and_log_appends_keep_every_entry(tmp_path):
    path = str(tmp_path / "p.mdatk.toml")
    Project().save(path)
    p = Project()

    def saver():
        for _ in range(40):
            p.save(path)

    t = threading.Thread(target=saver)
    t.start()
    for i in range(40):
        PF.append_log(path, PF.LogEntry(step=str(i)))
    t.join()
    assert [e["step"] for e in PF.read_log(path)] == [str(i) for i in range(40)]


def test_undecodable_file_names_do_not_break_the_log(tmp_path):
    path = str(tmp_path / "p.mdatk.toml")
    Project().save(path)
    with PF.logged(path, "pdb clean") as entry:
        entry.files = ["caf\udce9.pdb"]
    assert PF.read_log(path)[-1]["files"] == ["caf�.pdb"]


def test_runner_stops_the_script_on_keyboard_interrupt_and_keeps_output(tmp_path):
    from mdatk.workflows.runner import ScriptJob, run_jobs
    from mdatk.progress import Cancelled
    proj = str(tmp_path / "p.mdatk.toml")
    Project().save(proj)
    script = tmp_path / "slow.sh"
    script.write_text('cd "$(dirname "$0")"\necho start\nsleep 3\ntouch finished.txt\n')

    def on_line(line):
        if line == "start":
            raise KeyboardInterrupt

    with pytest.raises(Cancelled):
        run_jobs([ScriptJob(1, "slow", str(script))], proj, on_line=on_line)
    import time
    time.sleep(3.5)
    assert not (tmp_path / "finished.txt").exists()
    entry = PF.read_log(proj)[-1]
    assert entry["status"] == "cancelled" and "start" in entry["output"]


def test_cli_replay_of_a_params_log_uses_the_run_folders(tmp_path, monkeypatch, demo_dry):
    from mdatk.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["render", "-q", "--top", demo_dry["topology"], "--traj", demo_dry["trajectory"],
                 "--no-video", "--no-pdb", "--no-charts"]) == 0
    log = tmp_path / "MDATK_output" / "MDATK_params_log.toml"
    data = tomlio.load(str(log))
    assert data["parameters"]["output"]["out_dir"] == str(tmp_path / "MDATK_output")
    assert main(["charts", "-q", "-p", str(log), "--from-cache", "--formats", "svg"]) == 0
