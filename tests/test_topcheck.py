"""The topology check run after the Amber build (mdatk/workflows/scripts/topcheck.py)."""

import math

import pytest

from mdatk.workflows import mg_params
from mdatk.workflows.scripts import topcheck

SET = mg_params.get_set("micro", "tip3p")
ION = (SET.rmin_half, SET.eps_kcal)
OWN = {"MG": ION, "OS": (1.6837, 0.17), "LD": (1.6837, 0.17), "OW": (1.7683, 0.1520), "HO": (0.0, 0.0),
       "NA": (1.824, 0.17), "Na+": (1.369, 0.0874)}


def _fmt(values, per, width, fmt):
    out = []
    for k in range(0, len(values), per):
        out.append("".join(format(v, fmt).rjust(width) for v in values[k:k + per]))
    return out or [""]


def write_parm7(path, atoms, residues, pairs):
    """atoms: [(type, charge)]; residues: [(label, first atom index 1-based)];
    pairs: {(type_i, type_j): (Rmin_ij, eps_ij)} overriding the combination rule."""
    types = sorted({t for t, _q in atoms}, key=list(OWN).index)
    n = len(types)
    idx = {t: k + 1 for k, t in enumerate(types)}
    acoef, bcoef, ico = [], [], [0] * (n * n)
    for j in range(1, n + 1):
        for i in range(1, j + 1):
            ti, tj = types[i - 1], types[j - 1]
            (ri, ei), (rj, ej) = OWN[ti], OWN[tj]
            r, e = pairs.get((ti, tj), pairs.get((tj, ti), (ri + rj, math.sqrt(ei * ej))))
            k = len(acoef) + 1
            acoef.append(e * r ** 12)
            bcoef.append(2 * e * r ** 6)
            ico[n * (i - 1) + (j - 1)] = ico[n * (j - 1) + (i - 1)] = k
    pointers = [len(atoms), n] + [0] * 29
    sections = [
        ("POINTERS", "10I8", _fmt(pointers, 10, 8, "d")),
        ("AMBER_ATOM_TYPE", "20a4", ["".join(t.ljust(4) for t, _q in atoms)]),
        ("ATOM_TYPE_INDEX", "10I8", _fmt([idx[t] for t, _q in atoms], 10, 8, "d")),
        ("NONBONDED_PARM_INDEX", "10I8", _fmt(ico, 10, 8, "d")),
        ("LENNARD_JONES_ACOEF", "5E16.8", _fmt(acoef, 5, 16, ".8E")),
        ("LENNARD_JONES_BCOEF", "5E16.8", _fmt(bcoef, 5, 16, ".8E")),
        ("CHARGE", "5E16.8", _fmt([q * topcheck.CHARGE_SCALE for _t, q in atoms], 5, 16, ".8E")),
        ("RESIDUE_LABEL", "20a4", ["".join(lab.ljust(4) for lab, _p in residues)]),
        ("RESIDUE_POINTER", "10I8", _fmt([p for _l, p in residues], 10, 8, "d")),
    ]
    lines = ["%VERSION  VERSION_STAMP = V0001.000", "%FLAG TITLE", "%FORMAT(20a4)", "test"]
    for flag, fmt, body in sections:
        lines += [f"%FLAG {flag}", f"%FORMAT({fmt})"] + body
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def system(tmp_path, ld_pair=True, mg_charge=2.0, cl_missing=0.0, ion=ION):
    OWN["MG"] = ion
    atoms = [("OS", -1.0), ("LD", -1.0), ("HO", 0.0), ("OW", cl_missing), ("MG", mg_charge)]
    residues = [("LIG", 1), ("WAT", 4), ("MG", 5)]  # a small solute residue (not a nucleotide name)
    ref = mg_params.amber_pair(*SET.rna["OS"])
    pairs = {("MG", "OS"): ref}
    if ld_pair:
        pairs[("MG", "LD")] = ref
    try:
        return write_parm7(tmp_path / "sys.parm7", atoms, residues, pairs)
    finally:
        OWN["MG"] = ION


def run(path, **kw):
    opts = dict(mg_res="MG", neutral=True, mg_count=1, ion_lj=ION, pairs=True, model="microMg")
    opts.update(kw)
    return topcheck.check(path, **opts)


def test_good_system_passes(tmp_path, capsys):
    assert run(system(tmp_path)) == []
    out = capsys.readouterr().out
    assert "explicit Mg2+ pair term" in out and "passed" in out


def test_missing_pair_term_is_reported(tmp_path):
    problems = run(system(tmp_path, ld_pair=False))
    assert len(problems) == 1 and "LD (1)" in problems[0] and "OS" not in problems[0].split(":")[1].split(".")[0]


def test_uncharged_ion_and_net_charge(tmp_path):
    problems = run(system(tmp_path, mg_charge=0.0))
    assert any("not neutral" in p and "-2.0000" in p for p in problems)
    assert any("charge +0.0000 e instead of +2" in p for p in problems)
    assert run(system(tmp_path, mg_charge=0.0), neutral=False) != []  # the ion charge is checked anyway


def test_ion_without_published_lj(tmp_path):
    problems = run(system(tmp_path, ion=(1.395, 0.0103)))  # Li-Merz-like 12-6 values
    assert any("not the microMg values" in p for p in problems)
    # the plain 12-6 path checks only charges
    assert run(system(tmp_path, ion=(1.395, 0.0103), ld_pair=False), ion_lj=None, pairs=False) == []


def test_ion_count(tmp_path):
    assert any("4 Mg2+ ions were requested" in p for p in run(system(tmp_path), mg_count=4))


def test_main_exit_codes(tmp_path, monkeypatch):
    good = system(tmp_path)
    monkeypatch.setattr(topcheck, "ION_LJ", ION)
    monkeypatch.setattr(topcheck, "PAIRS", True)
    monkeypatch.setattr(topcheck, "MG_COUNT", 1)
    assert topcheck.main([good]) == 0
    assert topcheck.main([str(tmp_path / "missing.parm7")]) == 1
    (tmp_path / "x.parm7").write_text("%FLAG TITLE\n")
    assert topcheck.main([str(tmp_path / "x.parm7")]) == 1
    assert topcheck.main([]) == 2


def test_embedded_script_runs(tmp_path):
    import subprocess
    import sys
    from mdatk.workflows.scripts import embed
    script = tmp_path / "check.py"
    script.write_text(embed(topcheck, {"MG_RESIDUE": "MG", "NEUTRAL": True, "MG_COUNT": 1, "ION_LJ": ION,
                                       "PAIRS": True, "MODEL": "microMg"}, "main(sys.argv[1:])", "check"))
    good = system(tmp_path)
    assert subprocess.run([sys.executable, "-I", str(script), good], capture_output=True).returncode == 0
    bad = system(tmp_path, ld_pair=False)
    res = subprocess.run([sys.executable, "-I", str(script), bad], capture_output=True, text=True)
    assert res.returncode == 1 and "LD (1)" in res.stdout


def test_mapped_type_sharing_an_lj_type_is_reported(tmp_path):
    # XO reuses the OS Lennard-Jones terms, so tleap merges it into the OS LJ type; mapping it as OH
    # would overwrite the OS pair
    atoms = [("OS", -1.0), ("OS", 0.0), ("MG", 2.0), ("OW", -1.0)]
    path = write_parm7(tmp_path / "s.parm7", atoms, [("LIG", 1), ("MG", 3), ("WAT", 4)],
                       {("MG", "OS"): mg_params.amber_pair(*SET.rna["OS"])})
    os_ref = [round(v, 6) for v in mg_params.amber_pair(*SET.rna["OS"])]
    oh_ref = [round(v, 6) for v in mg_params.amber_pair(*SET.rna["OH"])]
    assert run(path, pair_refs={"OS": os_ref}) == []
    # the parm7 type names: rename the second atom XO but keep its LJ index
    text = open(path).read().replace("OS  OS  MG", "OS  XO  MG")
    open(path, "w").write(text)
    assert run(path, pair_refs={"OS": os_ref, "XO": os_ref}) == []
    problems = run(path, pair_refs={"OS": os_ref, "XO": oh_ref})
    assert any("OS, XO share one Lennard-Jones type" in p for p in problems)


def test_shared_solute_water_lj_type_is_noted(tmp_path, capsys):
    atoms = [("OS", -1.0), ("HO", 0.0), ("HO", 0.0), ("HO", 0.0), ("MG", 2.0), ("OW", -1.0)]
    # HO of the RNA and the water hydrogens (also zero LJ, typed HW in a real topology) share an LJ type
    path = write_parm7(tmp_path / "s.parm7", atoms, [("LIG", 1), ("WAT", 3), ("MG", 5), ("WAT", 6)],
                       {("MG", "OS"): mg_params.amber_pair(*SET.rna["OS"])})
    text = open(path).read().replace("OS  HO  HO  HO  MG", "OS  HO  HW  HW  MG")
    open(path, "w").write(text)
    assert run(path, pairs=False, ion_lj=None) == []  # a note, not a failure
    out = capsys.readouterr().out
    assert "shared by solute and solvent atoms: HO (1), HW (2)" in out


def test_solvent_is_classified_by_residue(tmp_path):
    # NA is the aromatic ring nitrogen of the RNA (G N1), Na+ the ion: the RNA's NA needs a Mg2+ pair,
    # the sodium ion (a single-atom residue, whatever its name) does not
    atoms = [("OS", -1.0), ("NA", 0.0), ("MG", 2.0), ("Na+", 1.0), ("OW", -2.0)]
    refs = {("MG", "OS"): mg_params.amber_pair(*SET.rna["OS"])}
    path = write_parm7(tmp_path / "s.parm7", atoms, [("LIG", 1), ("MG", 3), ("SOD", 4), ("WAT", 5)], refs)
    problems = run(path)
    assert len(problems) == 1 and "NA (1)" in problems[0] and "Na+" not in problems[0]
    refs[("MG", "NA")] = mg_params.amber_pair(*SET.rna["NA"])
    assert run(write_parm7(tmp_path / "t.parm7", atoms, [("LIG", 1), ("MG", 3), ("SOD", 4), ("WAT", 5)], refs)) == []


def test_residue_named_as_a_nucleotide_must_have_a_nucleotides_atoms(tmp_path):
    # an L-RNA library without the names MDATK writes: tleap resolves LA to lanthanum (atomic_ions.lib)
    atoms = [("OS", -1.0)] * 12 + [("NA", 1.0), ("MG", 2.0), ("Na+", -2.0)]
    path = write_parm7(tmp_path / "la.parm7", atoms, [("LG5", 1), ("LA", 13), ("MG", 14), ("LIG", 15)], {})
    problems = run(path, neutral=False, mg_count=0, ion_lj=None, pairs=False)
    assert len(problems) == 1 and "residue 2 LA with 1" in problems[0] and "lanthanum" in problems[0]
    path = write_parm7(tmp_path / "ok.parm7", atoms, [("LG5", 1), ("LIG", 13), ("MG", 14), ("LIG2", 15)], {})
    assert run(path, neutral=False, mg_count=0, ion_lj=None, pairs=False) == []
