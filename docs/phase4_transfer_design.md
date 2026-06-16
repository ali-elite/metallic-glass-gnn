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

## Outcome (what we actually found, vs the plan)

- **pyvoro validated** on `samples2`: perfect-ICO F1 = 1.000 (raw radical Voronoi,
  no cutoff).
- **Stability fix.** With the minimal radius-only features, LR 5e-3 made transfer
  AUC swing wildly between seeds (Cu–Zr–Al 0.31–0.75; occasional Co-W collapses).
  Dropping to **LR 1e-3** stabilised it (source test 0.99 either way). We report
  **mean ± std over 5 source-model seeds**.
- **Metrics.** Beyond ROC/PR-AUC and F1@0.5 we added two threshold diagnostics:
  `f1_recal` (base-rate-matched threshold — uses only the target's overall ICO
  fraction) and `f1_best` (best threshold, oracle upper bound). This separates
  *ranking* quality from *threshold calibration*.
- **Headline (5 seeds, LR 1e-3).** Source Cu–Zr 0.99. Zero-shot ROC-AUC:
  Cu–Zr 0.98 ± 0.01, Ni–Zr 0.96 ± 0.02 (≈ in-domain oracles), Co–W 0.89 ± 0.10,
  Cu–Zr–Al 0.65 ± 0.14.
- **Main finding (revised from the plan's expectation).** Transfer does *not* simply
  fade with chemical distance. Binary→binary transfer is excellent even when the
  target shares no elements with the source (Co-W). What breaks zero-shot transfer
  is **compositional novelty**: the ternary Cu–Zr–Al transfers poorly (0.65) yet is
  *learnable in-domain* (oracle 0.93) — the binary-trained model has simply never
  seen three-element local environments. The fixed 0.5 threshold does not transfer
  (base rates 19 % → 0–13 %), but a base-rate-matched threshold recovers most F1.
