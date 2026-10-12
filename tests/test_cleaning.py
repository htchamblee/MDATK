import os
import stat
import sys
import textwrap

import pytest

from conftest import cpptraj_executable
from mdatk.cleaning import CpptrajError, build_script, clean, cleaning_key
from mdatk.config import CleaningParams

FAKE_CPPTRAJ = textwrap.dedent("""\
    #!{python}
    # Minimal stand-in for cpptraj: writes the files named in the script.
    import re, sys
    script = open(sys.argv[sys.argv.index("-i") + 1]).read()
    print("CPPTRAJ: fake")
    for path in re.findall(r'(?:trajout|parmout|parmwrite out) "([^"]+)"', script):
        open(path, "w").write("fake")
    print(" 0% 50% 100% Complete.")
    """)


@pytest.fixture
def fake_cpptraj(tmp_path):
    path = tmp_path / "cpptraj"
    path.write_text(FAKE_CPPTRAJ.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_script_generation(tmp_path):
    top, traj = tmp_path / "s.parm7", tmp_path / "s.nc"
    top.write_text("x")
    traj.write_text("x")
    p = CleaningParams(solvent_mask=":WAT, HOH", ion_mask=":Na+,Cl-", fit_mask="@P,C4'",
                       autoimage_anchor=":1-12", offset=2)
    s = build_script(p, str(top), [str(traj)], str(tmp_path / "out"))
    lines = s.splitlines()
    assert lines[1] == f'parm "{top}"'
    assert f'trajin "{traj}" 1 last 2' in lines
    strip = next(l for l in lines if l.startswith("strip"))
    assert strip.startswith("strip (:WAT,HOH)|(:Na+,Cl-) parmout")
    assert lines.index("autoimage anchor :1-12") < lines.index("box nobox") < lines.index(
        "rms MDATKfit @P,C4' first mass")
    assert any(l.startswith("trajout") and l.endswith("netcdf") for l in lines)
    assert lines[-2:] == ["run", "quit"]
    p.fit_reference, p.reference_frame = "frame", 5
    s = build_script(p, str(top), [str(traj)], str(tmp_path / "out"))
    assert f'reference "{traj}" 5 [mdatkref]' in s and "ref [mdatkref] mass" in s
    p.fit = False
    assert "rms" not in build_script(p, str(top), [str(traj)], str(tmp_path / "out"))


def test_clean_runs_once_and_reuses(tmp_path, fake_cpptraj):
    top, traj = tmp_path / "s.parm7", tmp_path / "s.nc"
    top.write_text("x")
    traj.write_text("x")
    p = CleaningParams(enabled=True, cpptraj_path=fake_cpptraj)
    lines, progress = [], []
    r1 = clean(p, str(top), [str(traj)], str(tmp_path / "cache"), log_cb=lines.append,
               progress_cb=progress.append)
    assert not r1.reused and os.path.exists(r1.trajectory) and os.path.exists(r1.topology)
    assert any("CPPTRAJ: fake" in l for l in lines)
    assert progress[-1] == 1.0
    r2 = clean(p, str(top), [str(traj)], str(tmp_path / "cache"))
    assert r2.reused and r2.key == r1.key
    p.fit_mask = "@CA"
    assert cleaning_key(p, str(top), [str(traj)]) != r1.key
    traj.write_text("changed input")
    assert cleaning_key(CleaningParams(), str(top), [str(traj)]) != cleaning_key(
        CleaningParams(), str(top), [str(tmp_path / "s.parm7")])


def test_missing_cpptraj_is_reported(tmp_path):
    top = tmp_path / "s.parm7"
    top.write_text("x")
    p = CleaningParams(enabled=True, cpptraj_path="definitely-not-cpptraj")
    with pytest.raises(CpptrajError, match="not found"):
        clean(p, str(top), [str(top)], str(tmp_path))


@pytest.mark.skipif(cpptraj_executable() is None,
                    reason="cpptraj not available (set MDATK_TEST_CPPTRAJ to its path)")
def test_real_cpptraj_strips_solvent(demo_wet, tmp_path):
    import MDAnalysis as mda

    p = CleaningParams(enabled=True, cpptraj_path=cpptraj_executable())
    res = clean(p, demo_wet["topology"], [demo_wet["trajectory"]], str(tmp_path))
    raw = mda.Universe(demo_wet["topology"], demo_wet["trajectory"])
    u = mda.Universe(res.topology, res.trajectory)
    assert u.atoms.n_atoms < raw.atoms.n_atoms
    assert set(u.residues.resnames) == {"A", "C", "G", "U", "MG"}
    assert u.trajectory.n_frames == raw.trajectory.n_frames
    assert os.path.exists(res.reference_pdb)
