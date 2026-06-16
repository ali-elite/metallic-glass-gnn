"""Phase 1: characterise the icosahedral network (the community-detection target).

Uses only nb_id (edges) + fo_list (labels), which are internally consistent, so the
result does not depend on matching the trajectory frame.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.data import read_fo_list, read_nb_id, is_perfect_icosahedron, is_icosahedral_like
from src.graph import build_graph, edge_symmetry, physical_communities


def summarise(name, mask, G):
    comps, H = physical_communities(G, mask)
    n_ico = int(mask.sum())
    sizes = np.array([len(c) for c in comps]) if comps else np.array([0])
    largest = int(sizes.max())
    in_clusters = int(sizes[sizes >= 2].sum())  # atoms in a cluster of >=2 connected ICO
    print(f"\n[{name}]  {n_ico} atoms ({100*n_ico/G.number_of_nodes():.1f}% of system)")
    print(f"  connected ICO clusters (size>=2): {int((sizes>=2).sum())}")
    print(f"  isolated ICO atoms (size 1):      {int((sizes==1).sum())}")
    print(f"  largest cluster:                  {largest} atoms "
          f"({100*largest/max(n_ico,1):.1f}% of all {name} atoms)")
    print(f"  ICO atoms inside a cluster:       {in_clusters} ({100*in_clusters/max(n_ico,1):.1f}%)")
    return sizes


def main():
    fo = os.path.join(config.SAMPLES2, "fo_list")
    nb = os.path.join(config.SAMPLES2, "nb_id")
    total, vor, vol = read_fo_list(fo)
    nbrs = read_nb_id(nb)
    N = len(nbrs)
    assert len(vor) == N, f"label/neighbour count mismatch: {len(vor)} vs {N}"

    G = build_graph(nbrs)
    degs = np.array([d for _, d in G.degree()])
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  mean coordination (degree): {degs.mean():.2f}  (min {degs.min()}, max {degs.max()})")
    print(f"  neighbour-list reciprocity: {100*edge_symmetry(nbrs):.1f}%")

    ico = is_perfect_icosahedron(vor)
    like = is_icosahedral_like(vor, n5_min=10)
    s_ico = summarise("perfect-ICO <0,0,12,0>", ico, G)
    s_like = summarise("ICO-like (n5>=10)", like, G)

    # figure: cluster-size distribution of the icosahedral network
    os.makedirs(config.RESULTS, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, sizes, ttl in [(ax[0], s_ico, "perfect icosahedra"),
                          (ax[1], s_like, "icosahedral-like (n5>=10)")]:
        a.hist(sizes, bins=np.logspace(0, np.log10(max(sizes.max(), 2)), 30))
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("cluster size (atoms)"); a.set_ylabel("count")
        a.set_title(f"ICO-network clusters: {ttl}")
    fig.tight_layout()
    out = os.path.join(config.RESULTS, "01_ico_network_clusters.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
