"""Base pairs of the structural-view residues (internal; feeds secondary-structure panels).

Rows are exactly the residues of the ``trace`` analysis (:func:`structure_residues`, same order), so
row k of the pair map is row k of the trace.  Only nucleotides pair.  Pairs are detected on every
``stride``-th analysed frame and held in between (base pairs live far longer than a frame).

Engines (``analyses.basepairs.engine``):

``builtin``
    canonical Watson-Crick (G-C, A-U, A-T) and optional G-U wobble pairs from H-bond donor-acceptor
    distances, each base in at most one pair (:class:`mdatk.secstruct.PairDetector`).
``barnaba``
    optional package; Watson-Crick (``WCc``) and wobble (``GUc``) pairs, plus every other
    Leontis-Westhof class when ``noncanonical`` is set (a base may then be in several pairs).

Outputs: pair map ``map`` (value 1 Watson-Crick, 2 wobble, 3 other; each pair once with i < j),
series ``count`` (pairs per frame), arrays ``resindex`` (aligned with the trace) and meta ``engine``
plus, for barnaba, ``lw_first_frame`` ([i, j, class] of the first analysed frame).  Distances ignore
periodic images: paired bases belong to one molecule, which cleaned trajectories keep whole.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data import CSRBuilder
from ..secstruct import BarnabaRunner, PairDetector, import_barnaba, lw_kind
from ..selections import heavy
from .base import Analysis, register
from .common import labels
from .structure import VIEW_GROUPS, structure_groups, structure_residues

BARNABA_BATCH = 100  # stride frames written to one temporary multi-model PDB per barnaba call


@register
class BasePairAnalysis(Analysis):
    name = "basepairs"
    label = "Base pairs"
    internal = True
    version = 1
    OUTPUT_LABELS = {"map": "Base pairs", "count": "Base pairs"}

    @classmethod
    def params_for(cls, project):
        bp = project.analyses.basepairs
        return {"groups": structure_groups(project), "ions": project.analyses.mg.mg_selection,
                "engine": bp.engine, "stride": int(bp.stride), "cutoff": float(bp.cutoff),
                "wobble": bool(bp.wobble), "noncanonical": bool(bp.noncanonical)}

    @classmethod
    def selection_texts(cls, project):
        return structure_groups(project) + [project.analyses.mg.mg_selection]

    @classmethod
    def expected_outputs(cls, project):
        return {"series": ["count"], "matrices": [], "pairmaps": ["map"]}

    def setup(self, frame_indices):
        super().setup(frame_indices)
        bp = self.project.analyses.basepairs
        residues, _ = structure_residues(self.ctx)
        if len(residues) == 0:
            raise self.ctx.fail("base pairs: no residues selected for the structural view", VIEW_GROUPS)
        self.residues = residues
        self.engine = bp.engine
        self.stride = max(1, int(bp.stride))
        self.noncanonical = bool(bp.noncanonical)
        if self.engine == "barnaba":
            import_barnaba()  # fail early with the installation hint
        resnames = [str(r) for r in residues.resnames]
        # one coordinate array holding the atoms either engine needs, read once per detection frame
        atom_ix: List[int] = []
        atom_index: List[Dict[str, int]] = []
        heavy_atoms = []
        for res in residues:
            names: Dict[str, int] = {}
            for a in heavy(res.atoms):
                name = str(a.name).replace("*", "'")  # old-style primes
                if name not in names:  # first alternate location wins
                    names[name] = len(atom_ix)
                    atom_ix.append(int(a.ix))
            atom_index.append(names)
            heavy_atoms.append(list(names.items()))
        self.atoms = self.ctx.u.atoms[np.array(atom_ix, dtype=np.int64)]
        if self.engine == "barnaba":
            self.runner = BarnabaRunner(resnames, heavy_atoms)
        else:
            self.detector = PairDetector(resnames, atom_index, cutoff=bp.cutoff, wobble=bp.wobble)
        self.frame_pairs: List[np.ndarray] = [np.zeros((0, 3), dtype=np.int64)] * self.n_frames
        self.source = np.zeros(self.n_frames, dtype=np.int64)  # detection frame each frame copies
        self.pending: List[int] = []  # barnaba: detection frames waiting for a batch call
        self.pending_xyz: List[np.ndarray] = []
        self.lw_first: List[list] = []

    def process(self, k, ts):
        if k % self.stride:
            self.source[k] = self.source[k - 1]
            return
        self.source[k] = k
        xyz = self.atoms.positions.astype(np.float64)
        if self.engine == "barnaba":
            self.pending.append(k)
            self.pending_xyz.append(xyz)
            if len(self.pending) >= BARNABA_BATCH:
                self._flush()
        else:
            self.frame_pairs[k] = self.detector.detect(xyz)

    def _flush(self):
        if not self.pending:
            return
        for k, frame in zip(self.pending, self.runner.annotate(self.pending_xyz)):
            rows = [(i, j, lw_kind(c), c) for i, j, c in frame.pairs
                    if self.noncanonical or lw_kind(c) != 3]
            if k == 0:
                self.lw_first = [[int(i), int(j), c] for i, j, _, c in rows]
            self.frame_pairs[k] = np.array([r[:3] for r in rows], dtype=np.int64).reshape(-1, 3)
        self.pending, self.pending_xyz = [], []

    def finalize(self):
        if self.engine == "barnaba":
            self._flush()
        csr = CSRBuilder()
        count = np.zeros(self.n_frames, dtype=np.int64)
        for k in range(self.n_frames):
            p = self.frame_pairs[self.source[k]]
            csr.add(p[:, 0], p[:, 1], p[:, 2])
            count[k] = len(p)
        meta = dict(residue_labels=labels(self.residues), engine=self.engine, stride=self.stride,
                    value_labels={"1": "Watson-Crick", "2": "wobble", "3": "non-canonical"})
        if self.engine == "barnaba":
            meta["lw_first_frame"] = self.lw_first
        res = self.new_result(**meta)
        res.arrays.update(csr.arrays("map"))
        res.arrays["count"] = count
        res.arrays["resindex"] = self.residues.ix.astype(np.int64)
        res.add_series("count", "count", "Base pairs", "count")
        res.add_pairmap("map", "map", "residue_labels", "residue_labels", "Base pairs")
        return res
