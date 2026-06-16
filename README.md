# Detecting icosahedral medium-range order in metallic glasses with graph neural networks

A graph-learning reformulation of a B.Sc. thesis (Materials Science & Engineering,
Sharif University of Technology; supervisor **Dr. Rouhollah Tavakoli**) on the
local structure of Cu–Zr bulk metallic glasses.

## Background: what the original thesis found

The thesis tried to predict each atom's **Voronoi index** from the flat,
distance-ordered coordinates of its 20 nearest neighbours, using a multi-output
**MLP**. It did not work (least-squares loss ≈ 2.1, and far worse with LBFGS), and
the thesis correctly concluded that the inputs/architecture were the problem and
recommended **graph neural networks** as the way forward.

The diagnosis: a flat neighbour vector breaks **permutation invariance** — the
network has to learn that neighbour ordering is meaningless — while the Voronoi
index is a *geometric* function of the whole neighbour cloud. A graph/permutation-
invariant model is the right inductive bias. This repository builds that.

## Reframed task

Instead of regressing the Voronoi index, we study the **icosahedral network** —
the connected backbone of icosahedral (and icosahedral-like) atoms that underlies
medium-range order (MRO) and slow dynamics in Cu–Zr glasses.

1. **Node classification (supervised baseline).** Predict whether an atom is a full
   icosahedron `<0,0,12,0>` from geometry + graph, with a lightweight GCN.
   *Fair-learning rule:* the model sees a distance/kNN graph, **not** the Voronoi
   face graph, so it cannot trivially read off the label.
2. **Community detection (the headline GNN task).** Partition the atomic graph into
   communities **without** Voronoi labels (modularity-based GNN, DMoN-style) and
   test whether the discovered communities recover the icosahedral backbone / MRO
   domains. Motivation: a fast, differentiable, transferable MRO detector that does
   not need a Voronoi tessellation at inference time.

## Why this is well-posed (Phase-1 result, `samples2`, 10,000 atoms)

```
Graph: 10,000 nodes, 68,586 edges, mean coordination 13.7, 100% reciprocal
perfect icosahedra <0,0,12,0> : 19.1%  -> ONE percolating cluster holds 81.8% of them
icosahedral-like (n5>=10)     : 40.4%  -> 99.6% in a single giant cluster
```

The icosahedra are not scattered — they **percolate** into a giant connected
backbone. That backbone is exactly the community structure the GNN should learn to
find from geometry alone. (`scripts/01_ico_network.py`,
figure in `results/01_ico_network_clusters.png`.)

## Phase-2 result — geometry → icosahedron (`samples2`, same atoms & splits)

Predicting the full icosahedron ⟨0,0,12,0⟩ from a periodic **kNN graph** (fixed
`k=16`, no Voronoi edges, so node degree cannot leak the label). Coordinate↔label
alignment verified (100% of Voronoi face-neighbours within the 20 nearest spatial
neighbours; mean bond 2.95 Å). Test set = 1,500 held-out atoms (19% positive):

| model | acc | ICO-F1 | macro-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| MLP (thesis-style flat 20-NN vector) | 0.835 | 0.604 | 0.750 | 0.899 | 0.630 |
| **CGCNN (distance-aware GNN)** | **0.964** | **0.904** | **0.941** | **0.994** | **0.978** |

The permutation-invariant, geometry-aware GNN raises minority-class F1 from
**0.60 → 0.90** on identical data and splits — confirming the thesis's own
diagnosis that the *architecture*, not the physics, was the bottleneck.
(`scripts/02_node_classification.py`; transductive on one snapshot —
cross-snapshot / cross-system generalisation is Phase 4.)

## Data

| Folder | System | Has Voronoi labels? | Has face-sharing graph? |
|---|---|---|---|
| `samples1` | Cu₆₄Zr₃₆, 13,500 atoms | yes (`Face_order_list`) | no (build from coords) |
| `samples2` | 10,000 atoms | yes (`fo_list`) | **yes (`nb_id`)** ← start here |
| `samples3` | Co–W, Cu–Zr–Al, Cu–Zr, Ni–Zr (many compositions) | no (run Voro++) | no |

`samples3` enables the **transferability** study (train on one chemistry, test on
another). See [`docs/lammps.md`](docs/lammps.md) for how the trajectories were
generated (reconstructed MD methodology).

## Layout

```
config.py                 # paths to the raw data
src/data.py               # parse LAMMPS dump / fo_list / nb_id; icosahedron labels
src/graph.py              # build atomic graph; physical (ground-truth) ICO communities
scripts/01_ico_network.py # Phase 1: characterise the icosahedral network  [DONE]
docs/lammps.md            # the molecular-dynamics stage (reconstructed)
results/                  # figures
```

## Roadmap

- [x] **Phase 1** — data pipeline, atomic graph, physical ICO-network ground truth.
- [x] **Phase 2** — GNN node classifier (geometry → icosahedron): ICO-F1 0.90 vs 0.60 (MLP), ROC-AUC 0.99.
- [ ] **Phase 3** — GNN community detection (DMoN-style) vs the physical backbone (NMI/ARI).
- [ ] **Phase 4** — transfer across `samples3` chemistries; report + figures.

## Attribution

Based on the undergraduate thesis of Ali Ghelichkhani, supervised by
Dr. Rouhollah Tavakoli (Sharif University of Technology). Any public release or
manuscript should credit the supervisor as a co-author.
