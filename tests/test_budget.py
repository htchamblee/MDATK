"""CPU and GPU budgets of the runner (mdatk.workflows.budget): what each step type uses, the scheduling of
fake jobs on a simulated clock, GPU detection and the project-file settings."""

import heapq

import pytest

from mdatk.config import Project
from mdatk.workflows import TOOLS
from mdatk.workflows.budget import (ALL_CORES, Budget, Resources, Slots, budget_notes, detected_gpus,
                                    env_gpus, gpu_ids, schedule)

CPU = Resources(1)
GPU = Resources(1, 1)


def simulate(needs, budget, durations=None, deps=None, limit=0):
    """Run fake jobs on a simulated clock with the runner's scheduling: returns [(index, start, grant)] in
    the order they started.  Asserts that the running jobs never exceed the budget or share a GPU."""
    durations = durations or [1.0] * len(needs)
    deps = deps or [set() for _ in needs]
    slots = Slots(budget, limit)
    done, started, running = set(), [], []  # running: heap of (end, index, grant)
    now = 0.0
    while len(done) < len(needs):
        waiting = [i for i in range(len(needs)) if i not in done and all(i != s[0] for s in started)]
        ready = [(i, needs[i]) for i in waiting if deps[i] <= done]
        for i, grant in schedule(ready, slots):
            started.append((i, now, grant))
            heapq.heappush(running, (now + durations[i], i, grant))
        # the budget holds at every moment
        if budget is not None:
            assert sum(g.cores for _e, _i, g in running) <= budget.cores
            ids = [x for _e, _i, g in running for x in g.gpus]
            assert len(ids) == len(set(ids)), ids
        assert running, "nothing runs and nothing starts: deadlock"
        end, i, grant = heapq.heappop(running)
        now = end
        slots.give_back(grant)
        done.add(i)
        while running and running[0][0] == now:  # jobs that end at the same time
            _e, k, g = heapq.heappop(running)
            slots.give_back(g)
            done.add(k)
    return started


def starts(result):
    return {i: t for i, t, _g in result}


# ---------------------------------------------------------------------------
# scheduling (pure, simulated clock)
# ---------------------------------------------------------------------------


def test_cpu_only_jobs_fill_the_cores_in_plan_order():
    result = simulate([Resources(2), Resources(2), Resources(2), Resources(1)], Budget(4))
    assert starts(result) == {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
    assert [g.cores for _i, _t, g in result] == [2, 2, 2, 1]
    # a later, smaller job starts while an earlier one waits for room
    result = simulate([Resources(3), Resources(2), Resources(1)], Budget(4), durations=[2.0, 1.0, 1.0])
    assert starts(result) == {0: 0.0, 2: 0.0, 1: 2.0}


def test_gpu_jobs_get_their_own_gpu():
    result = simulate([GPU] * 4, Budget(8, ("0", "1")), durations=[1.0, 2.0, 1.0, 1.0])
    assert [(i, t, g.gpus, g.assigned) for i, t, g in result] == [
        (0, 0.0, ("0",), True), (1, 0.0, ("1",), True), (2, 1.0, ("0",), True), (3, 2.0, ("0",), True)]


def test_namd_uses_its_cores_and_a_gpu():
    namd = TOOLS["namd-mdff"].run_resources(TOOLS["namd-mdff"].normalise({}))
    assert namd == Resources(8, 1, ())
    # two NAMD steps on 8 cores: one at a time although a second GPU is free
    assert starts(simulate([namd, namd], Budget(8, ("0", "1")))) == {0: 0.0, 1: 1.0}
    # with 12 cores, NAMD and pmemd.cuda run side by side on different GPUs
    result = simulate([namd, GPU], Budget(12, ("0", "1")))
    assert [(i, t, g.cores, g.gpus) for i, t, g in result] == [(0, 0.0, 8, ("0",)), (1, 0.0, 1, ("1",))]
    # on 8 cores pmemd.cuda waits for a core, not for a GPU
    assert starts(simulate([namd, GPU], Budget(8, ("0", "1")))) == {0: 0.0, 1: 1.0}


def test_a_pinned_gpu_is_used_as_named_and_kept_from_others():
    pinned = Resources(1, 1, ("1",))
    result = simulate([pinned, GPU, pinned], Budget(8, ("0", "1")))
    by = {i: (t, g) for i, t, g in result}
    assert by[0][1].gpus == ("1",) and not by[0][1].assigned  # the step's own export decides
    assert by[1][1].gpus == ("0",) and by[1][0] == 0.0  # the free one, not GPU 1
    assert by[2][0] == 1.0  # waits for GPU 1
    # a GPU outside the budget is only kept from other steps that name it
    other = Resources(1, 1, ("3",))
    assert starts(simulate([other, other, GPU], Budget(8, ("0",)))) == {0: 0.0, 2: 0.0, 1: 1.0}
    notes = budget_notes([("step 01 (X)", other)], Budget(8, ("0",)))
    assert any("names GPU 3" in n for n in notes)


def test_a_job_larger_than_the_budget_runs_alone():
    big = Resources(16)
    result = simulate([CPU, big, CPU, Resources(1, 2)], Budget(4, ("0",)))
    # 0 runs; 1 waits for it and holds back 2 and 3; then 1 alone; then 2; then 3 (2 GPUs of 1: alone)
    assert [(i, t) for i, t, _g in result] == [(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)]
    by = {i: g for i, _t, g in result}
    assert by[1].alone and by[1].cores == 4 and by[3].alone and by[3].gpus == ("0",)
    notes = budget_notes([("step 02 (Big)", big), ("step 04 (Two GPUs)", Resources(1, 2))], Budget(4, ("0",)))
    assert "step 02 (Big) asks for 16 CPU cores, more than the 4 CPU cores and 1 GPU (0)" in notes[0]
    assert "step 04 (Two GPUs) asks for 2 GPUs" in notes[1] and "runs alone" in notes[1]


def test_gpu_steps_without_a_gpu_run_one_at_a_time():
    result = simulate([GPU, GPU, CPU], Budget(4))
    assert starts(result) == {0: 0.0, 2: 0.0, 1: 1.0}
    assert all(g.gpus == () and not g.assigned for _i, _t, g in result)
    notes = budget_notes([("step 01 (MD)", GPU), ("step 02 (MD)", GPU), ("step 03 (CPU)", CPU)], Budget(4))
    assert len(notes) == 1 and "no GPU found" in notes[0] and "step 01 (MD); step 02 (MD)" in notes[0]


def test_all_cores_steps_and_the_scripts_at_once_cap():
    gmx = Resources(ALL_CORES, 1)
    result = simulate([gmx, CPU, GPU], Budget(6, ("0", "1")))
    assert starts(result) == {0: 0.0, 1: 1.0, 2: 1.0}  # every core: nothing beside it, and no warning
    assert result[0][2].cores == 6 and not result[0][2].alone
    assert budget_notes([("step 01 (GROMACS)", gmx)], Budget(6, ("0",))) == []
    # waiting for every core, it holds back the steps after it (they would keep it waiting)
    result = simulate([CPU, gmx, CPU, CPU], Budget(4, ("0",)), durations=[2.0, 1.0, 1.0, 1.0])
    assert starts(result) == {0: 0.0, 1: 2.0, 2: 3.0, 3: 3.0}
    # at most two at once although four cores are free
    assert starts(simulate([CPU] * 3, Budget(4), limit=2)) == {0: 0.0, 1: 0.0, 2: 1.0}
    # no budget: only the cap counts, GPUs are not handed out
    result = simulate([GPU, GPU, GPU], None, limit=2)
    assert starts(result) == {0: 0.0, 1: 0.0, 2: 1.0} and all(g.gpus == () for _i, _t, g in result)


def test_dependencies_and_determinism():
    needs = [Resources(2), Resources(1, 1), Resources(3), Resources(1), Resources(2, 1),
             Resources(1, 1, ("1",))]
    deps = [set(), set(), {0}, set(), {1}, set()]
    durations = [2.0, 1.0, 1.0, 3.0, 1.0, 2.0]
    first = simulate(needs, Budget(4, ("0", "1")), durations, deps)
    for _ in range(5):
        assert simulate(needs, Budget(4, ("0", "1")), durations, deps) == first
    t = starts(first)
    assert t[2] >= 2.0 and t[4] >= 1.0  # after their inputs
    # the order depends on the plan order, not on chance
    assert [i for i, _t, _g in first] == [0, 1, 3, 5, 4, 2]


# ---------------------------------------------------------------------------
# what each step type uses
# ---------------------------------------------------------------------------


def res(tool, **settings):
    t = TOOLS[tool]
    return t.run_resources(t.normalise(settings))


def test_run_resources_of_every_step_type():
    assert res("amber-md") == Resources(1, 1, ())  # pmemd.cuda
    assert res("amber-md", gpu="1") == Resources(1, 1, ("1",))
    assert res("amber-md", engine="pmemd.MPI", nproc=16) == Resources(16, 0, ())
    assert res("amber-md", engine="pmemd") == Resources(1, 0, ())
    # CUDA_VISIBLE_DEVICES in the Environment names the GPU; the step's GPU option is exported after it
    env = "module load cuda\nexport CUDA_VISIBLE_DEVICES=2  # the free one"
    assert res("amber-md", env_setup=env) == Resources(1, 1, ("2",))
    assert res("amber-md", env_setup=env, gpu="0") == Resources(1, 1, ("0",))
    assert res("amber-md", engine="pmemd.MPI", nproc=4, env_setup=env) == Resources(4, 0, ())
    assert res("gromacs") == Resources(ALL_CORES, 1, ())  # mdrun without -ntomp takes every core
    assert res("gromacs", ntomp=4) == Resources(4, 1, ())
    assert res("gromacs", ntomp=4, gpu=False) == Resources(4, 0, ())
    assert res("gromacs", ntomp=4, extra_mdrun="-ntmpi 2 -pin on") == Resources(8, 1, ())
    assert res("gromacs", extra_mdrun="-nt 12") == Resources(12, 1, ())
    assert res("gromacs", ntomp=2, env_setup="CUDA_VISIBLE_DEVICES=1 gmx --version") == Resources(2, 1,
                                                                                                    ("1",))
    assert res("namd-mdff") == Resources(8, 1, ())
    assert res("namd-mdff", cores=16, gpu="1") == Resources(16, 1, ("1",))
    # +devices counts the GPUs the Environment leaves visible
    assert res("namd-mdff", gpu="1", env_setup="export CUDA_VISIBLE_DEVICES=2,3") == Resources(8, 1, ("3",))
    assert res("farfar2") == Resources(6) and res("farfar2", runs=4, parallel=20) == Resources(4)
    assert res("farfar2-models") == Resources(1)
    assert res("farfar2-models", extract_all=True, extract_parallel=4) == Resources(4)
    assert res("farfar2-models", extract_parallel=4) == Resources(1)  # nothing extracted side by side
    assert res("simrna") == Resources(4) and res("simrna", runs=2) == Resources(2)
    assert res("phenix-rsr", nproc=8) == Resources(8)
    assert res("register-census", jobs=4) == Resources(4)
    assert res("frame-check", jobs=3) == Resources(3)
    assert res("user-script") == Resources(1, 0, ())
    assert res("user-script", cores=4, gpus=1) == Resources(4, 1, ())
    assert res("user-script", cores=0) == Resources(ALL_CORES, 0, ())
    assert res("user-script", env_setup="export CUDA_VISIBLE_DEVICES=0") == Resources(1, 1, ("0",))
    one_core = {"amber-build", "amber-restraints", "structure-builder", "restraint-dicts", "coot-rsr",
                "rnamake", "vmd", "model-check", "decoy-interface", "rosetta-audit", "ensemble-compare",
                "map-model-score", "body-hybrids"}
    covered = one_core | {"amber-md", "gromacs", "namd-mdff", "farfar2", "farfar2-models", "simrna",
                          "phenix-rsr", "register-census", "frame-check", "user-script"}
    assert covered == {n for n, t in TOOLS.items() if t.generates}, "a step type without a tested budget"
    for name in one_core:
        assert res(name) == Resources(1, 0, ()), name


def test_resource_texts_and_gpu_settings():
    assert Resources(8, 1).badge() == "8c+1g" and Resources(1).badge() == "1c"
    assert Resources(ALL_CORES, 1).badge() == "all cores+1g"
    assert Resources(1, 1, ("2", "3")).badge() == "1c+2g"
    assert Resources(8, 1).text() == "8 cores, 1 GPU"
    assert Resources(1, 1, ("1",)).text() == "1 core, GPU 1 (named by the step)"
    assert gpu_ids("0, 1") == ("0", "1") and gpu_ids("") == ()
    assert env_gpus({"env_setup": "export CUDA_VISIBLE_DEVICES='0,1'"}) == ("0", "1")
    assert env_gpus({"env_setup": "# export CUDA_VISIBLE_DEVICES=3\nmodule load x"}) is None
    assert env_gpus({"env_setup": "export CUDA_VISIBLE_DEVICES=1\nexport CUDA_VISIBLE_DEVICES=2"}) == ("2",)
    assert env_gpus({}) is None


def test_gpu_detection_and_budget():
    smi = ("GPU 0: NVIDIA A100 (UUID: GPU-1)\n  MIG 1g.10gb Device 0: (UUID: MIG-1)\n"
           "GPU 1: NVIDIA A100 (UUID: GPU-2)\n")
    assert detected_gpus({}, smi) == ["0", "1"]
    assert detected_gpus({}, "") == []
    # MDATK's own CUDA_VISIBLE_DEVICES limits what its scripts may use
    assert detected_gpus({"CUDA_VISIBLE_DEVICES": "2,3"}, smi) == ["2", "3"]
    assert detected_gpus({"CUDA_VISIBLE_DEVICES": ""}, smi) == []
    assert detected_gpus({"CUDA_VISIBLE_DEVICES": "-1"}, smi) == []
    b = Budget.resolve(0, -1, found=["0", "1"])
    assert b.cores >= 1 and b.gpus == ("0", "1")
    assert Budget.resolve(6, 1, found=["2", "3"]) == Budget(6, ("2",))
    assert Budget.resolve(6, 3, found=["0"]) == Budget(6, ("0", "1", "2"))  # more than found: as told
    assert Budget.resolve(6, 0, found=["0"]).gpus == ()
    assert Budget(8, ("0", "1")).text() == "8 CPU cores and 2 GPUs (0, 1)"


# ---------------------------------------------------------------------------
# project files
# ---------------------------------------------------------------------------


def test_old_project_files_keep_their_parallel_setting(tmp_path):
    old = tmp_path / "old.mdatk.toml"
    old.write_text('[workflow]\nproject_name = "apt1"\nparallel = 3\n')
    wf = Project.load(str(old)).workflow
    assert (wf.parallel, wf.cpu_cores, wf.gpus) == (3, 0, -1)
    older = tmp_path / "older.mdatk.toml"  # before 'parallel' existed: one script at a time, as then
    older.write_text('[workflow]\nproject_name = "apt1"\n')
    assert Project.load(str(older)).workflow.parallel == 1
    new = Project()
    assert (new.workflow.parallel, new.workflow.cpu_cores, new.workflow.gpus) == (0, 0, -1)
    new.workflow.cpu_cores, new.workflow.gpus = 12, 2
    path = tmp_path / "new.mdatk.toml"
    new.save(str(path))
    text = path.read_text()
    assert "# CPU cores the scripts MDATK runs at the same time may use together" in text
    assert "cpu_cores = 12" in text and "gpus = 2" in text and "parallel = 0" in text
    back = Project.load(str(path)).workflow
    assert (back.parallel, back.cpu_cores, back.gpus) == (0, 12, 2)
    assert Budget.of(back, gpus=1).cores == 12


def test_cli_budget_options(tmp_path):
    from mdatk.cli import _budget, _parallel, build_parser
    p = Project()
    p.workflow.parallel, p.workflow.cpu_cores, p.workflow.gpus = 3, 6, 0
    parser = build_parser()
    args = parser.parse_args(["workflow", "run", "--cores", "2", "--gpus", "-1", "--parallel", "0"])
    assert _parallel(args, p) == 0 and _budget(args, p).cores == 2
    args = parser.parse_args(["workflow", "write", "--run"])
    assert _parallel(args, p) == 3 and _budget(args, p) == Budget(6, ())
    with pytest.raises(ValueError, match="--cores"):
        _budget(parser.parse_args(["workflow", "run", "--cores", "-2"]), p)
    with pytest.raises(ValueError, match="--parallel"):
        _parallel(parser.parse_args(["workflow", "run", "--parallel", "-1"]), p)
