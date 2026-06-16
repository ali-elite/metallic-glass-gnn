"""Geometry -> features for the node-classification task.

Input graph is a periodic kNN graph built from coordinates only (NOT the Voronoi
face graph), with fixed k so node degree cannot leak the icosahedron label.
"""
import os
import numpy as np
from scipy.spatial import cKDTree

import config
from src.data import read_lammps_dump, read_fo_list, read_nb_id, is_perfect_icosahedron

# element radii (R.txt). type 1 = Cu (majority ~64%), type 2 = Zr.
RADIUS = {1: 1.28, 2: 1.60}


def load_samples2():
    d = read_lammps_dump(os.path.join(config.SAMPLES2, "ma_data"))
    total, vor, vol = read_fo_list(os.path.join(config.SAMPLES2, "fo_list"))
    nbrs = read_nb_id(os.path.join(config.SAMPLES2, "nb_id"))
    N = len(vor)
    assert d["natoms"] == N, f"{d['natoms']} positions vs {N} labels"
    L = (d["box"][:, 1] - d["box"][:, 0]).astype(float)
    pos = (d["pos"] - d["box"][:, 0]) % L          # wrap into [0,L)
    types = d["types"].astype(int)
    radius = np.array([RADIUS[t] for t in types], dtype=float)
    y = is_perfect_icosahedron(vor).astype(np.int64)
    return dict(pos=pos, L=L, types=types, radius=radius, y=y, nbrs=nbrs, N=N, vor=vor)


def _minimum_image(delta, L):
    return delta - L * np.round(delta / L)


def knn_periodic(pos, L, k):
    """Return symmetric edge_index (2,E) and edge_dist (E,) for a periodic kNN graph."""
    tree = cKDTree(pos, boxsize=L)
    dist, idx = tree.query(pos, k=k + 1)           # includes self at col 0
    src = np.repeat(np.arange(pos.shape[0]), k)
    dst = idx[:, 1:].reshape(-1)
    d = dist[:, 1:].reshape(-1)
    # symmetrise (undirected): add reverse edges, drop duplicates
    e = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    dd = np.concatenate([d, d])
    key = e[0].astype(np.int64) * pos.shape[0] + e[1].astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    return e[:, uniq].astype(np.int64), dd[uniq].astype(np.float32)


def alignment_check(pos, L, nbrs, k=20):
    """Fraction of Voronoi face-neighbours that fall within each atom's k nearest
    spatial neighbours. ~1.0 confirms positions and labels refer to the same atoms."""
    tree = cKDTree(pos, boxsize=L)
    _, idx = tree.query(pos, k=k + 1)
    near = [set(row[1:]) for row in idx]
    hit = tot = 0
    edist = []
    for i, nb in enumerate(nbrs):
        for j in nb:
            if 0 <= j < pos.shape[0]:
                tot += 1
                if j in near[i]:
                    hit += 1
                edist.append(np.linalg.norm(_minimum_image(pos[i] - pos[j], L)))
    return hit / max(tot, 1), float(np.mean(edist))


def flat_neighbour_features(pos, L, k=20):
    """Thesis-style input: relative coords of the k nearest neighbours, sorted by
    distance, flattened -> (N, 3k). Permutation-SENSITIVE on purpose."""
    tree = cKDTree(pos, boxsize=L)
    dist, idx = tree.query(pos, k=k + 1)
    feats = np.zeros((pos.shape[0], k * 3), dtype=np.float32)
    for i in range(pos.shape[0]):
        nb = idx[i, 1:]
        delta = _minimum_image(pos[nb] - pos[i], L)        # sorted by distance already
        feats[i] = delta.reshape(-1)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    return ((feats - mu) / sd).astype(np.float32)


def rbf_expand(dist, n_rbf=16, cutoff=6.0):
    centers = np.linspace(0.0, cutoff, n_rbf, dtype=np.float32)
    gamma = (n_rbf / cutoff) ** 2
    return np.exp(-gamma * (dist[:, None] - centers[None, :]) ** 2).astype(np.float32)


def rotation_invariant_features(pos, L, nbrs, radius):
    """Per-atom rotation/translation-invariant local-geometry scalars (label-free):
    coordination number, mean & std of bond length to face-sharing neighbours, and
    the mean radius of those neighbours (local composition). Returns (N,4).

    These describe the geometry/connectivity of the neighbour cloud, NOT the
    Voronoi index breakdown <n3,n4,n5,n6,...> that defines the icosahedron label.
    (Coordination equals the total face count, i.e. the graph degree, which is
    intrinsic to the Voronoi graph and on its own a near-chance predictor of the
    label; the discriminative signal is bond regularity, derived from positions.)
    """
    N = pos.shape[0]
    coord = np.zeros(N, np.float32)
    mean_bond = np.zeros(N, np.float32)
    std_bond = np.zeros(N, np.float32)
    mean_nbr_radius = np.zeros(N, np.float32)
    for i, nb in enumerate(nbrs):
        nb = [j for j in nb if 0 <= j < N and j != i]
        coord[i] = len(nb)
        if nb:
            dd = np.linalg.norm(_minimum_image(pos[nb] - pos[i], L), axis=1)
            mean_bond[i] = dd.mean()
            std_bond[i] = dd.std()
            mean_nbr_radius[i] = radius[nb].mean()
    return np.stack([coord, mean_bond, std_bond, mean_nbr_radius], axis=1)
