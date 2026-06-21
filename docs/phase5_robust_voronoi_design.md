# Phase 5 — Robust, learned Voronoi index (design)

**Question.** Can a GNN learn the radical-Voronoi index ⟨n3,n4,n5,n6⟩ *directly from
atomic coordinates* and be **more temporally stable than Voro++ itself** — i.e. return
the same local-structure label under the picosecond thermal jitter that makes the raw
tessellation flip? If so, the learned descriptor replaces Voro++ at inference: fast,
differentiable, and noise-robust, with no tessellation at run time.

This closes the project's arc. Detecting the full icosahedron (Phase 2) is the special
case ⟨0,0,12,0⟩ of predicting the whole index here; the icosahedral network, the
label-free communities, and the cross-chemistry transfer (Phases 1/3/4) all become
*applications* of having a robust per-atom descriptor instead of a brittle one.

## 0. Project reframing (the unifying pass)

Independent of the new science, the repository is re-narrated as **one study of
learned, robust local-structure descriptors for metallic glasses** — not "a
reformulation of a B.Sc. thesis". README and notebook intros drop the
"what the original thesis found" / "reframed task" scaffolding; the five phases read
as a single arc (characterise → detect → cluster → transfer → **robustly learn the
descriptor**). The thesis origin shrinks to a one-line provenance note; Dr. Tavakoli
remains acknowledged as co-author. No logic changes to Phases 1–4 — narrative + light
edits only. Detailed edits are enumerated in the implementation plan.

## 1. Teacher: consensus Voronoi labels (`src/voronoi.py`)

Voro++ (via `pyvoro`, already validated to reproduce `fo_list` at perfect-ICO F1 =
1.000) is the **teacher**. It is the only place a tessellation runs, and only to make
*training labels*; it is never needed at inference.

Add `consensus_index(frames, L, radii)`:

- run `voronoi_index` on each of the *F* frames of one system → per-frame index
  arrays `(N, 6)` (n3..n8);
- **consensus label** `y*` `(N, 4)` = per-atom **mode** of ⟨n3,n4,n5,n6⟩ across frames
  (ties broken toward the first frame, which is the reference snapshot);
- **Voro++ instability** `(N,)` = `1 − (mode count / F)` — the fraction of frames whose
  index disagrees with the consensus. The dataset-level mean is the headline *baseline*
  the GNN must beat. Also report the per-atom *coordination total* (full face count,
  = graph degree) as the regression's auxiliary target.

n7/n8 are rare in Cu–Zr and are **not** predicted; they are absorbed as slack between
Σ(n3..n6) and the coordination total (see §5). Report their coverage (% of atoms with
n7=n8=0) for honesty.

## 2. Data & splits

- **Primary system:** `samples1` — 13,500 Cu₆₄Zr₃₆ atoms × **11 consecutive frames**
  (timesteps 22000000–22000010), radii Cu 1.28 / Zr 1.60 from `R.txt`. Frames carry
  positions, velocities and forces; we use positions (velocities feed §4's σ estimate).
- **Atom-disjoint split:** 70/15/15 train/val/test over *atom indices*, fixed seed. All
  11 frames of a train atom are train — we never evaluate on an atom seen in training,
  and there is no frame leakage. (Splitting frames instead would leak: the same atom
  appears in every frame.)
- **Cross-system check:** apply the trained model to `samples2`'s single clean frame
  (10,000 atoms, same Cu–Zr chemistry, different size/box) and score against its
  ground-truth `fo_list`. Tests generalisation across snapshot and system size.

## 3. Model (`src/models.py`): `CGCNNRegressor`

Add a **sibling** `CGCNNRegressor` that reuses the existing `CGConv` message passing on
the **periodic kNN graph** (`k≈20`, with RBF-expanded bond-distance edge features) —
already rotation- and translation-invariant, which is half of thermal robustness by
construction — leaving the Phase-2 `CGCNN` classifier untouched. It swaps the 2-class
head for a regression head:

- four **non-negative count outputs** (softplus) for ⟨n3,n4,n5,n6⟩;
- one auxiliary **coordination-total** output (softplus), supervised against the full
  face count, giving a soft sum-consistency signal.

Node features = **element radius only** (element-agnostic, consistent with Phase 4, so
the same model can later be probed on other chemistries). `k≈20` covers the full
neighbour shell (Voronoi coordination runs ~12–18; the Phase-2 alignment check used
k=20). At inference the four count outputs are **rounded** to integers.

## 4. Robustness mechanism

Two complementary levers, both already decided as the success criterion (consensus +
invariance):

1. **Consensus target.** Training against `y*` (not any single noisy frame) means the
   model fits the *stable* topology by construction — it cannot chase frame-specific
   flips because they are averaged out of the label.
2. **Physically-calibrated jitter augmentation.** Each epoch, perturb input coordinates
   by 𝒩(0, σ²) (minimum-image safe), **rebuild the kNN graph**, and keep `y*` as the
   target — directly teaching "thermal motion ⇒ same index". σ is *physical*, not
   arbitrary:
   - `thermal_sigma(frames, L)` in `src/features.py` = RMS per-atom displacement between
     consecutive frames (a direct, if small, measurement);
   - cross-checked against the equipartition estimate from the per-atom velocities;
   - training uses a **curriculum / sweep** of σ up to roughly the full thermal
     vibration amplitude, so robustness is characterised across the realistic range
     rather than at one tiny inter-frame scale.

## 5. Loss

`L = smoothL1(n̂, n*) + λ_tot · smoothL1(Ĉ, C*) + λ_sum · |Σn̂ − Ĉ|`

- smooth-L1 (Huber) on the four counts — robust to the occasional large miss;
- smooth-L1 on the coordination total `C`;
- a small soft penalty tying Σ(predicted counts) to the predicted total (n7/n8 slack
  keeps this soft, not hard). λ_tot, λ_sum small (≈0.3, ≈0.1) — tuned on val.

## 6. Evaluation & success criteria

- **Accuracy of the learned index:** per-count MAE (n3..n6), **exact-index match** rate
  (all four correct), and **ICO-F1** for ⟨0,0,12,0⟩ as a special case — directly
  comparable to Phase 2's 0.90.
- **Headline robustness — temporal flip-rate.** Over the 11 frames, the fraction of
  test atoms whose *predicted* index changes frame-to-frame, **GNN vs raw Voro++**.
  Target: GNN flip-rate ≪ Voro++ flip-rate while staying faithful to `y*`.
- **σ-sweep.** Prediction agreement vs jitter amplitude σ, GNN vs Voro++ recomputed on
  the same perturbed coordinates. Expect Voro++ to degrade faster — the core robustness
  claim, shown as a curve.
- **Learned instability map.** Per-atom prediction variance under test-time jitter is a
  differentiable "near-a-topological-transition" flag; correlate it with the Voro++
  instability of §1 (does the model *know* which atoms are unstable?).
- **Outputs:** `results/05_robust_voronoi.json` (all metrics) + `results/05_robust_voronoi.png`
  (flip-rate bars, σ-sweep curve, instability scatter, per-count error).

## 7. Correctness checks (not ML metrics)

- Re-run `validate_against_samples2()` → teacher still exact (F1 = 1.000).
- Unit invariants: consensus of a single frame == that frame's index; `jitter(σ=0)` is
  identity; `thermal_sigma > 0`; graph rebuild deterministic given a seed; predicted
  total ≈ graph degree on clean frames.

## 8. Deliverables

- *New:* `scripts/05_robust_voronoi.py` (end-to-end Phase 5), `results/05_robust_voronoi.{json,png}`.
- *Edit:* `src/voronoi.py` (+`consensus_index`, instability), `src/features.py`
  (+`thermal_sigma`, +`jitter`), `src/models.py` (+`CGCNNRegressor`), `README.md`
  (unify + Phase 5 row/section), `metallic_glass_gnn.ipynb` (Phase 5 section + reframed
  intro).
- *Deps:* unchanged — numpy / scipy / torch + pyvoro (already a Phase-4 dependency).
  CPU-friendly: pyvoro on 13,500×11 cells is seconds; full-batch CGCNN training is
  minutes.

## 9. Reuse

`src.features` (`knn_periodic`, `rbf_expand`, `_minimum_image`), `src.models`
(`CGConv`, CGCNN trunk), `src.data.read_lammps_frames`, `src.voronoi.voronoi_index`.
New code: the consensus labeller, the thermal-σ estimate + jitter augmentation, the
regression head, and the Phase-5 script.

## 10. Honesty / contingencies

- The 11 raw frames are ~femtoseconds apart, so the *real* inter-frame Voro++ flip-rate
  may be small. Reported **as-is**; the calibrated **σ-sweep** (up to full thermal
  amplitude) is the headline robustness axis, with the real frames validating the jitter
  model at small scale.
- If radius-only features underperform on exact counts, element one-hot is a fallback
  for the Cu–Zr primary result (transfer-style radius-only kept as the element-agnostic
  variant).
- We do **not** claim the GNN is *more accurate* than Voro++ on a single clean frame —
  Voro++ defines the label there. The claim is **temporal stability** (and speed /
  differentiability) at equal single-frame fidelity.
