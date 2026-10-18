import os
import shutil
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", module="MDAnalysis")


@pytest.fixture(autouse=True)
def _private_settings(tmp_path_factory, monkeypatch):
    """Keep the recent-projects list of the tests out of the user's ~/.config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config")))


def make_universe(atoms, coords, bonds=(), dt=1.0):
    """Small in-memory Universe.

    ``atoms``: list of dicts with name, resid, resname, element and optional
    charge / mass.  ``coords``: (n_frames, n_atoms, 3).
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    resids = []
    for a in atoms:
        if a["resid"] not in resids:
            resids.append(a["resid"])
    resnames = {}
    for a in atoms:
        resnames.setdefault(a["resid"], a["resname"])
    u = mda.Universe.empty(len(atoms), n_residues=len(resids),
                           atom_resindex=[resids.index(a["resid"]) for a in atoms], trajectory=True)
    u.add_TopologyAttr("names", [a["name"] for a in atoms])
    u.add_TopologyAttr("resids", resids)
    u.add_TopologyAttr("resnames", [resnames[r] for r in resids])
    u.add_TopologyAttr("elements", [a["element"] for a in atoms])
    u.add_TopologyAttr("masses", [a.get("mass", 12.0) for a in atoms])
    u.add_TopologyAttr("charges", [a.get("charge", 0.0) for a in atoms])
    u.add_TopologyAttr("segids", ["SYS"])
    if bonds:
        u.add_TopologyAttr("bonds", list(bonds))
    u.load_new(np.asarray(coords, dtype=np.float32), format=MemoryReader, dt=dt)
    return u


def micro_project(groups):
    """Project whose groups are ``{name: residue range}``."""
    from mdatk.config import Project, ResidueGroup

    p = Project()
    p.groups = [ResidueGroup(name, res, "", "#2a78d6") for name, res in groups.items()]
    return p


def run_one(analysis_name, universe, project, topology_path=""):
    from mdatk.analysis import ANALYSES, AnalysisContext, run_analyses

    ctx = AnalysisContext(universe, project, topology_path)
    (res,), _ = run_analyses(ctx, [ANALYSES[analysis_name](ctx)],
                             np.arange(universe.trajectory.n_frames))
    return res


@pytest.fixture(scope="session")
def demo_dry(tmp_path_factory):
    """Solvent-free synthetic system (fast; no cleaning needed)."""
    from mdatk.demo import build_demo

    d = tmp_path_factory.mktemp("demo_dry")
    return build_demo(str(d), n_frames=24, solvent=False)


@pytest.fixture(scope="session")
def demo_wet(tmp_path_factory):
    """Solvated synthetic system for cpptraj tests."""
    from mdatk.demo import build_demo

    d = tmp_path_factory.mktemp("demo_wet")
    return build_demo(str(d), n_frames=12, solvent=True)


def cpptraj_executable():
    path = os.environ.get("MDATK_TEST_CPPTRAJ") or shutil.which("cpptraj")
    return path if path and os.access(path, os.X_OK) else None
