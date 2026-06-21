# Learning robust local-structure descriptors for metallic glasses

A graph-neural study of local atomic order in Cu–Zr-type bulk metallic glasses:
from detecting icosahedral order, to label-free structure discovery, to
cross-chemistry transfer, to a **learned, thermally-robust replacement for the
Voronoi index itself**. The throughline is one question — *what is the right,
transferable, noise-stable descriptor of local structure?* — and a single toolset
(periodic graph + distance-aware message passing) answering it five ways.

> 📓 **Start here:** [`metallic_glass_gnn.ipynb`](metallic_glass_gnn.ipynb) — one
> self-contained, executed notebook covering every phase with physics motivation,
> model maths, and honest discussion.

## The five phases, one arc

Detecting the icosahedron (Phase 2) is the special case ⟨0,0,12,0⟩ of predicting
the full Voronoi index (Phase 5); the icosahedral network, label-free communities,
and cross-chemistry transfer (Phases 1/3/4) are what a robust per-atom descriptor
makes possible.

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
| MLP (flat 20-NN vector baseline) | 0.835 | 0.604 | 0.750 | 0.899 | 0.630 |
| **CGCNN (distance-aware GNN)** | **0.964** | **0.904** | **0.941** | **0.994** | **0.978** |

The permutation-invariant, geometry-aware GNN raises minority-class F1 from
**0.60 → 0.90** on identical data and splits — confirming that the *architecture*,
not the physics, was the bottleneck.
(`scripts/02_node_classification.py`; transductive on one snapshot —
cross-snapshot / cross-system generalisation is Phase 4.)

## Phase-3 result — label-free community detection (`samples2`, all 10,000 atoms)

Can an **unsupervised** GNN recover the icosahedral backbone *without ever seeing a
Voronoi label*? We run a DMoN-style modularity GNN (soft cluster assignments from a
GCN encoder, trained with a spectral-modularity loss + collapse/entropy
regulariser, pure PyTorch) on the Voronoi face-sharing graph, and compare it to the
two natural baselines. Node features are chemistry + rotation-invariant local
geometry (coordination, mean/std bond length, neighbour radius) — **no labels**.
Ground truth (scoring only): "icosahedral backbone vs matrix" (`n5≥10`, 40.4%).

| method | uses | graph $Q$ | NMI (bb) | backbone ROC-AUC | backbone F1 |
|---|---|---|---|---|---|
| **DMoN GNN** (label-free) | graph + features | 0.23 | **0.105** | **0.82** | 0.67 |
| Louvain | topology only | **0.73** | 0.002 | 0.55 | 0.07 |
| k-means | features only | 0.02 | 0.104 | 0.82 | 0.69 |

Three honest takeaways (we do **not** overclaim a GNN win on everything):

1. **Topological modularity is the wrong objective for MRO.** Louvain maximises
   modularity (Q=0.73) but its communities are *spatial blobs* essentially **blind**
   to the backbone (NMI 0.002, AUC 0.55) — because the perfect-ICO network
   *percolates* (Phase 1) and is interpenetrating, not a separable cluster.
2. **The backbone is a local-geometry signal.** Plain k-means on rotation-invariant
   per-atom invariants recovers it at AUC 0.82 (chemistry alone: AUC 0.52 — no
   signal), echoing the Phase-2 finding that geometry → icosahedron.
3. **The label-free GNN gets the best of both.** DMoN matches feature-clustering on
   backbone recovery (AUC 0.82, ~50× Louvain's NMI) **and** keeps real graph
   coherence (Q=0.23, an order of magnitude above k-means's 0.02) — one
   differentiable model that is simultaneously backbone-aligned and spatially
   coherent. Within the backbone subgraph it further resolves 8 MRO sub-domains
   (Q=0.33). (`scripts/03_community_detection.py`,
   `results/03_community_detection.{json,png}`.)

## Phase-4 result — cross-chemistry transfer (`samples3`, 27 target alloys)

Does a Cu–Zr-trained icosahedron detector transfer **zero-shot** to other glass
chemistries? We label every `samples3` target with the *same* radical-Voronoi
method that produced the `samples2` labels — a [pyvoro](https://github.com/joe-jordan/pyvoro)
labeller (`src/voronoi.py`) **validated to reproduce `fo_list` perfect-ICO at
F1 = 1.000**. To transfer at all, the model is made **element-agnostic**: each atom
carries only its radius (+ the periodic kNN graph with RBF bond-distance edges), so
the same Cu–Zr model applies to Co/W, Ni/Zr, Cu/Zr/Al. Numbers are **mean ± std
over 5 source-model seeds** (LR 1e-3; the higher 5e-3 was unstable, swinging the
ternary AUC 0.31–0.75 across seeds). Source Cu–Zr test: ROC-AUC **0.99**.

| target family | members | zero-shot ROC-AUC | in-domain oracle |
|---|---|---|---|
| **Cu–Zr** (new compositions) | 46:54 → 64:36 | **0.98 ± 0.01** | 0.98 |
| **Ni–Zr** (new element) | 46:54 → 64:36 | **0.96 ± 0.02** | 0.97 |
| **Co–W** (different alloy) | W 10–75 % | **0.89 ± 0.10** | 0.99 |
| **Cu–Zr–Al** (ternary) | Al 5–25 % | **0.65 ± 0.14** | **0.93** |

Honest reading (we separate *ranking* from *thresholded* classification):

1. **Binary→binary transfer is strong and stable.** Zero-shot ROC-AUC is **0.98 ± 0.01**
   (Cu–Zr, new compositions) and **0.96 ± 0.02** (Ni–Zr, a *new element*) — essentially
   **matching in-domain oracles** (0.98, 0.97). Co–W, which shares *no* elements with
   the source, still transfers well on average (**0.89**) but more variably (± 0.10).
   The element-agnostic detector ranks icosahedra in unseen *binary* alloys nearly as
   well as a model trained on them.
2. **Adding a third element breaks transfer — and that is the interesting part.** The
   ternary Cu–Zr–Al transfers poorly (**0.65 ± 0.14**), yet an **in-domain oracle reaches
   0.93** there. So the icosahedron *is* learnable in the ternary; the binary-trained
   model simply never saw three-element local environments. **Compositional novelty (a
   new element type), not chemical distance, is what defeats zero-shot transfer** — a
   completely different *binary* (Co–W) transfers far better than the *same* base
   elements plus Al.
3. **The 0.5 threshold doesn't transfer, but recalibration recovers it.** Perfect-ICO
   base rates swing from 19 % (Cu–Zr) to 0–13 %, so F1 at the fixed threshold collapses
   where ICO is rare; a **base-rate-matched threshold** (one scalar per target) lifts it
   most of the way to the optimal — e.g. Co–W mean F1 0.14 → 0.56, Cu–Zr 0.49 → 0.67 —
   confirming the ranking is sound and only calibration is off.
4. **Limits, stated plainly.** Co–W and the ternary remain higher-variance across seeds
   (± 0.10–0.14); at ≥ 80 % W the perfect icosahedron essentially vanishes (nothing to
   detect) and the detector hallucinates ICO far outside the training regime.
   (`scripts/04_transfer.py`, `results/04_transfer.{json,png}`, design in
   [`docs/phase4_transfer_design.md`](docs/phase4_transfer_design.md).)

## Phase-5 result — a robust, learned Voronoi index (`samples1`, 11 frames)

Can a GNN *replace* Voro++ at inference **and** be more robust to thermal motion than
the tessellation it learned from? We predict ⟨n3,n4,n5,n6⟩ from coordinates alone with
a **per-count classification** CGCNN — argmax per count, which is locally constant and
therefore jitter-stable (unlike rounding a regressor) — distilled from a **time-stable
consensus** of Voro++ over the 11 consecutive frames, and trained with a
**temporal-consistency** regulariser that penalises prediction changes between a clean
frame and a thermally-jittered copy (jitter scaled to the physical Debye–Waller
amplitude ≈ 0.12 Å).

**Robustness — the headline, at the physically-meaningful scale.** Under controlled
jitter the learned index stays put where Voro++ wanders. Fraction of test atoms whose
index is unchanged vs the σ = 0 reference (higher = more stable):

| jitter σ (Å) | 0.05 | 0.08 | 0.10 | **0.12** | 0.15 |
|---|---|---|---|---|---|
| **learned GNN** | **0.78** | **0.70** | **0.65** | **0.60** | **0.52** |
| Voro++ | 0.75 | 0.61 | 0.56 | 0.51 | 0.44 |

Across the whole realistic thermal-vibration range the GNN holds its index **~9 points
more often** than Voro++, and the margin *grows* with displacement. (At the sub-thermal
0.01 Å spacing of the raw frames the two are tied — flip-rate 0.214 vs 0.212 — that
scale is finer than thermal motion, so there is essentially nothing to beat.)

**Accuracy is retained where it matters.** The same model recovers the perfect
icosahedron at **ICO-F1 0.72**, and — trained on `samples1` only — transfers to the
*different* `samples2` system at **ICO-F1 0.81**, with no Voro++ at inference.

**An accuracy ↔ robustness knob.** The consistency weight λ trades exact-count fidelity
for thermal stability: λ = 0 is most accurate (ICO-F1 0.82, cross-system 0.89) but
*less* stable than Voro++; λ = 4 (above) wins the σ-sweep at ICO-F1 0.72.

**Stated plainly.** The *exact* four-count match is modest (≈ 0.13): the full index is
intrinsically a sensitive function of geometry, so robustness is realised at the
structural-class (icosahedron) level, not every exact count. The σ-sweep measures
self-consistency the model is trained for, but accuracy on the *true* consensus labels
is independently retained. (`scripts/05_robust_voronoi.py`,
`results/05_robust_voronoi.{json,png}`, design in
[`docs/phase5_robust_voronoi_design.md`](docs/phase5_robust_voronoi_design.md).)

## Data

| Folder | System | Has Voronoi labels? | Has face-sharing graph? |
|---|---|---|---|
| `samples1` | Cu₆₄Zr₃₆, 13,500 atoms | yes (`Face_order_list`) | no (build from coords) |
| `samples2` | 10,000 atoms | yes (`fo_list`) | **yes (`nb_id`)** ← start here |
| `samples3` | Co–W, Cu–Zr–Al, Cu–Zr, Ni–Zr (many compositions) | **yes (Phase 4: pyvoro radical Voronoi)** | no |

`samples3` powers the **transferability** study (Phase 4: train on Cu–Zr, test on
27 other alloys). See [`docs/lammps.md`](docs/lammps.md) for how the trajectories
were generated (reconstructed MD methodology).

## Requirements

Phases 1–3 are pure **PyTorch + NetworkX + scikit-learn** (CPU). Phase 4 adds one
optional dependency, **`pyvoro`** (a Voro++ binding), used only to compute
radical-Voronoi labels for `samples3`: `pip install pyvoro`.

## Layout

```
metallic_glass_gnn.ipynb       # unified, executed walkthrough of Phases 1-3 (read this first)
config.py                      # paths to the raw data
src/data.py                    # parse LAMMPS dump / fo_list / nb_id; icosahedron labels
src/graph.py                   # build atomic graph; physical (ground-truth) ICO communities
src/features.py                # geometry -> node features (kNN graph, invariant scalars)
src/models.py                  # MLP, CGCNN (Phase 2); DMoN modularity GNN (Phase 3)
src/voronoi.py                 # Phase 4: pyvoro radical-Voronoi index (labels any chemistry)
src/metrics.py                 # pure Voronoi-index metrics (flip-rate, exact match, MAE)
scripts/01_ico_network.py      # Phase 1: characterise the icosahedral network  [DONE]
scripts/02_node_classification.py  # Phase 2: geometry -> icosahedron classifier  [DONE]
scripts/03_community_detection.py  # Phase 3: label-free community detection      [DONE]
scripts/04_transfer.py         # Phase 4: cross-chemistry zero-shot transfer      [DONE]
scripts/05_robust_voronoi.py   # Phase 5: robust, learned Voronoi index            [DONE]
docs/lammps.md                 # the molecular-dynamics stage (reconstructed)
docs/phase4_transfer_design.md # Phase 4 design / methods note
docs/phase5_robust_voronoi_design.md # Phase 5 design / methods note
results/                       # figures + metrics JSON
```

## Roadmap

- [x] **Phase 1** — data pipeline, atomic graph, physical ICO-network ground truth.
- [x] **Phase 2** — GNN node classifier (geometry → icosahedron): ICO-F1 0.90 vs 0.60 (MLP), ROC-AUC 0.99.
- [x] **Phase 3** — label-free DMoN community detection: backbone ROC-AUC **0.82**, NMI **0.105** (≈50× Louvain's 0.002); finding that MRO is a *local-geometry* signal, not topological modularity.
- [x] **Phase 4** — cross-chemistry zero-shot transfer over 27 `samples3` alloys (mean±std, 5 seeds): an element-agnostic Cu–Zr detector ranks icosahedra at ROC-AUC **0.98 (Cu–Zr) / 0.96 (Ni–Zr) / 0.89 (Co–W) / 0.65 (Cu–Zr–Al)** — binary→binary transfer ≈ in-domain, but adding a 3rd element (Cu–Zr–Al) breaks transfer (oracle 0.93): compositional novelty, not chemical distance, is the limit; threshold needs recalibration.
- [x] **Phase 5** — robust, learned Voronoi index: a per-count classification CGCNN predicts
  ⟨n3,n4,n5,n6⟩ from coordinates (consensus label + temporal-consistency regularisation),
  replacing Voro++ at inference. **More stable than Voro++ at physical thermal amplitudes**
  (σ-sweep agreement +~9 pts over 0.05–0.15 Å) at ICO-F1 **0.72** / cross-system **0.81**; λ
  tunes accuracy↔robustness. (Raw fs-frame flip-rate is a tie; exact full-index match is modest.)

## Acknowledgements

Local-structure data and the original problem framing come from work at Sharif
University of Technology with **Dr. Rouhollah Tavakoli**, who should be credited as
a co-author on any public release or manuscript.
