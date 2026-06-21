"""Pure metrics for the Phase-5 learned Voronoi index. No torch, no data — these
operate on (N,4) integer index arrays and (F,N,4) per-frame stacks, so they are
trivially unit-testable and reused by the script and the notebook."""
import numpy as np


def ico_from_counts(counts):
    """Perfect icosahedron <0,0,12,0> from (N,4) counts <n3,n4,n5,n6>. Returns (N,) bool."""
    c = np.asarray(counts)
    return (c[:, 0] == 0) & (c[:, 1] == 0) & (c[:, 2] == 12) & (c[:, 3] == 0)


def flip_rate(idx_per_frame):
    """Fraction of atoms whose index is NOT identical across all frames.

    idx_per_frame: (F, N, K) integer indices. The headline temporal-stability metric;
    compute for the GNN and for raw Voro++ on the same frames and compare."""
    a = np.asarray(idx_per_frame)
    const = (a == a[0]).all(axis=0).all(axis=1)         # (N,) constant across frames
    return float(1.0 - const.mean())


def exact_match(pred, true):
    """Fraction of atoms whose full (N,4) index matches exactly."""
    return float((np.asarray(pred) == np.asarray(true)).all(axis=1).mean())


def per_count_mae(pred, true):
    """Mean absolute error per count -> (4,) array for n3,n4,n5,n6."""
    return np.abs(np.asarray(pred) - np.asarray(true)).mean(axis=0).astype(float)
