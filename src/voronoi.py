"""Radical (power) Voronoi index via pyvoro.

This labels *any* chemistry the same way Voro++ originally labelled `samples2`,
which is what makes the Phase-4 cross-chemistry transfer study possible (samples3
ships trajectories only, no Voronoi labels).

Validation (see `validate_against_samples2`): raw radical Voronoi with the R.txt
radii reproduces the supplied `fo_list` perfect-icosahedron label at per-atom
F1 = 1.000 (1912 vs 1913 atoms). Face-area and edge-length cutoffs were tested and
only reduce agreement, so no cutoff is applied.

Requires `pyvoro` (a Voro++ binding): `pip install pyvoro`. Only Phase 4 needs it;
Phases 1-3 stay pure PyTorch / NetworkX / scikit-learn.
"""
import numpy as np


def voronoi_index(pos, L, radii, dispersion=4.0):
    """Per-atom radical Voronoi index.

    pos: (N,3) wrapped into [0, L).  L: (3,) box lengths.  radii: (N,) atom radii.
    Returns idx (N,6) = counts of faces with 3,4,5,6,7,8 edges (n3..n8), matching
    the `fo_list` convention used elsewhere in the project.
    """
    import pyvoro  # imported lazily so Phases 1-3 never require it
    N = pos.shape[0]
    cells = pyvoro.compute_voronoi(
        np.asarray(pos, float).tolist(),
        [[0.0, float(L[0])], [0.0, float(L[1])], [0.0, float(L[2])]],
        float(dispersion),
        radii=[float(r) for r in radii],
        periodic=[True, True, True],
    )
    idx = np.zeros((N, 6), dtype=int)
    for i, c in enumerate(cells):
        for f in c["faces"]:
            e = len(f["vertices"])               # #edges of this Voronoi face
            if 3 <= e <= 8:
                idx[i, e - 3] += 1
    return idx


def validate_against_samples2():
    """Sanity check: pyvoro radical Voronoi must reproduce the supplied fo_list."""
    from src.features import load_samples2
    from src.data import is_perfect_icosahedron, is_icosahedral_like
    from sklearn.metrics import f1_score
    d = load_samples2()
    idx = voronoi_index(d["pos"], d["L"], d["radius"])
    perf_pv = is_perfect_icosahedron(idx)
    perf_gt = is_perfect_icosahedron(d["vor"])
    like_agree = (is_icosahedral_like(idx, 10) == is_icosahedral_like(d["vor"], 10)).mean()
    return dict(
        perfect_pyvoro=int(perf_pv.sum()), perfect_fo_list=int(perf_gt.sum()),
        perfect_f1=float(f1_score(perf_gt, perf_pv)),
        perfect_agreement=float((perf_pv == perf_gt).mean()),
        like_agreement=float(like_agree),
        full_index_match=float((idx == d["vor"]).all(1).mean()),
    )
