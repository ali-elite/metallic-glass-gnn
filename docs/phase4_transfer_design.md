# Phase 4 — Cross-chemistry transfer (design)

**Question.** Does a GNN that learned to recognise the icosahedron in Cu–Zr
transfer *zero-shot* to other glass-forming chemistries (different compositions,
new elements, entirely different alloys)?

This is now answerable because `samples3` (trajectories only, no Voronoi labels)
can be labelled with the **same** radical-Voronoi method that produced the
`samples2` labels.

## 1. Voronoi labeller (`src/voronoi.py`)

Use **pyvoro** (a Voro++ binding) with per-atom radii from `R.txt` to compute a
**radical (power) Voronoi tessellation** and the per-atom Voronoi index
`(n3,n4,n5,n6,n7,n8)` = number of faces with 3..8 edges.

*Validation (built in, run on `samples2`).* Raw pyvoro (no face/edge cutoff)
reproduces the supplied `fo_list`:

- perfect icosahedron `<0,0,12,0>`: **1912 vs 1913 atoms, per-atom F1 = 1.000**;
- icosahedral-like `n5>=10`: 91.4 % per-atom agreement;
- exact full 6-tuple match: 79.7 %.

Face-area and edge-length cutoffs were tested and only *reduce* agreement, so we
use raw radical Voronoi. The perfect-ICO target — the Phase-2/Phase-4 label — is
reproduced essentially exactly, which is what the transfer study needs.

## 2. Element-agnostic features (the crux of transfer)

Phase 2 used a Cu/Zr one-hot, which cannot represent Co/W/Ni/Al. The transfer
model therefore uses **radius-only node features**: each atom → its element radius
(from `R.txt`), standardised. Geometry still enters through the periodic **kNN
graph** (`k=16`) with RBF-expanded bond-distance edge features, exactly as in
Phase 2. No element identity is given to the model — consistent with the Phase-3
finding that geometry + size carry the icosahedral signal.

We first confirm the radius-only model still scores well on Cu–Zr (source) before
claiming anything about transfer.

## 3. Protocol

1. Train the CGCNN perfect-ICO classifier **once** on `samples2` (Cu–Zr),
   radius-only features, stratified split, class-weighted loss (as Phase 2).
2. **Zero-shot** apply the frozen model to every target chemistry; pyvoro provides
   the ground-truth labels there.
3. Per target report: ICO base rate, ICO-F1, ROC-AUC, PR-AUC.
4. For one representative target per family, train a **within-target oracle**
   (same CGCNN, cross-validated on that chemistry) to show the transfer gap
   (zero-shot vs in-domain upper bound).

## 4. Target set (one relaxed frame per file)

| family | members | tests |
|---|---|---|
| CuZr (same chemistry) | 64:36, 50:50, 46:54 | composition shift, same elements |
| NiZr (new element) | 64:36, 50:50, 46:54 | new chemistry, similar size ratio |
| Cu-Zr-Al (ternary) | Al 5,10,15,20,25 % | added third element |
| Co-W (far) | W 10..85 % | entirely different alloy |

Multi-frame trajectories (CuZr/NiZr, 11 frames) use the **last (most relaxed)
frame**. A composition-trend panel reports predicted vs pyvoro-true ICO fraction
across Al %, W %, and Cu:Zr ratio.

## 5. Metrics, baselines, honesty

- Headline: zero-shot transfer ICO-F1 / ROC-AUC per target, against each target's
  own base rate (a trivial all-negative baseline).
- Oracle gap: zero-shot vs in-domain CGCNN on representative targets.
- **Be honest:** transfer is expected to degrade with chemical distance
  (CuZr ≈ source ≫ NiZr ≳ Cu-Zr-Al ≫ Co-W). Report whatever happens; a strong
  trend (AUC tracking size-ratio similarity to Cu–Zr) is itself the result. Do not
  claim universal transfer.

## 6. Deliverables

- `src/voronoi.py` — pyvoro radical-Voronoi index + `samples2` validation.
- `scripts/04_transfer.py` — source training, zero-shot sweep, oracles, trends.
- `results/04_transfer.json` + figure(s) in `results/`.
- README roadmap → Phase 4 done with headline numbers.
- README/requirements note: **pyvoro** is now a dependency (used only for Phase 4
  labelling; Phases 1–3 remain pure PyTorch/NetworkX/scikit-learn).

## Reuse

`src.features` (`knn_periodic`, `rbf_expand`, `alignment_check`), `src.models`
(`CGCNN`), `src.data.read_lammps_dump`, `src.graph`. New code is the pyvoro
labeller and a small multi-chemistry loader.
