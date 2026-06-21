# Phase 5 — Robust Learned Voronoi Index — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GNN that predicts the radical-Voronoi index ⟨n3,n4,n5,n6⟩ from atomic coordinates, distilled from a time-stable Voro++ consensus and made invariant to thermal jitter, then show it is more temporally stable than raw Voro++ — and reframe the repo as one unified study.

**Architecture:** Reuse the existing periodic-kNN + CGConv stack. New pure-numpy units (consensus labeller, jitter/σ, metrics) are TDD'd in `src/`; a new `CGCNNRegressor` swaps the classifier head for 4 count outputs + a coordination-total head; a self-contained `scripts/05_robust_voronoi.py` orchestrates data → train (with jitter aug) → evaluate (temporal flip-rate, σ-sweep, instability map) → JSON + figure.

**Tech Stack:** Python 3.8, numpy/scipy, PyTorch 2.4 (pure, no PyG), scikit-learn, matplotlib, pyvoro (Voro++ binding, label generation only), pytest.

**Conventions for this plan:**
- Spec: [`docs/phase5_robust_voronoi_design.md`](phase5_robust_voronoi_design.md).
- Run tests from the repo root with `python3 -m pytest` (the `-m` puts the repo root on `sys.path` so `from src...` resolves; `src/` is a package).
- Pure-numpy/torch unit tests need **no data and no pyvoro**. Tests that need them are guarded with `pytest.importorskip("pyvoro")` / a `SAMPLES1` existence skip, so the core suite runs anywhere.
- Commit messages follow the project's `Phase 5: …` style and end with the `Co-Authored-By` trailer.
- Work happens on branch `phase5-robust-voronoi` (already created).

---

## File Structure

| File | New/Mod | Responsibility |
|---|---|---|
| `src/voronoi.py` | Mod | `_row_mode`, `consensus_index` (aggregate per-frame indices → stable label + instability), `voronoi_index_frames` (pyvoro over frames) |
| `src/features.py` | Mod | `thermal_sigma`, `jitter`, `load_samples1_frames` |
| `src/models.py` | Mod | `CGCNNRegressor` (sibling of `CGCNN`), `voronoi_loss` |
| `src/metrics.py` | New | Pure metrics: `ico_from_counts`, `flip_rate`, `exact_match`, `per_count_mae` |
| `scripts/05_robust_voronoi.py` | New | Self-contained Phase-5 runner (data→train→eval→figure→json), `--smoke` flag |
| `tests/test_consensus.py` | New | `consensus_index` / `_row_mode` |
| `tests/test_features_robust.py` | New | `thermal_sigma`, `jitter`, loader |
| `tests/test_models_regressor.py` | New | `CGCNNRegressor`, `voronoi_loss` |
| `tests/test_metrics.py` | New | metric functions |
| `tests/test_voronoi_frames.py` | New | `voronoi_index_frames` (pyvoro-guarded) |
| `README.md` | Mod | Unify framing + Phase 5 result |
| `metallic_glass_gnn.ipynb` | Mod | Reframed intro + Phase 5 section |
| `results/05_robust_voronoi.{json,png}` | New | Metrics + figure (generated) |

---

## Task 1: Consensus Voronoi labeller (`src/voronoi.py`)

**Files:**
- Modify: `src/voronoi.py` (append functions; keep `voronoi_index`, `validate_against_samples2`)
- Test: `tests/test_consensus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consensus.py
import numpy as np
from src.voronoi import _row_mode, consensus_index


def test_row_mode_picks_majority_and_count():
    rows = np.array([[0, 0, 12, 0], [0, 0, 12, 0], [0, 1, 10, 2]])
    mode, cnt = _row_mode(rows)
    assert list(mode) == [0, 0, 12, 0]
    assert cnt == 2


def test_row_mode_tie_breaks_toward_first_frame():
    rows = np.array([[0, 2, 8, 2], [0, 0, 12, 0]])  # 1 each -> tie
    mode, cnt = _row_mode(rows)
    assert list(mode) == [0, 2, 8, 2]               # frame 0 wins
    assert cnt == 1


def test_consensus_index_single_frame_is_that_frame():
    fi = np.array([[[0, 0, 12, 0, 0, 0], [0, 1, 10, 4, 0, 0]]])  # (F=1,N=2,6)
    con = consensus_index(fi)
    assert con["label"].tolist() == [[0, 0, 12, 0], [0, 1, 10, 4]]
    assert con["total"].tolist() == [12, 15]
    assert con["instability"].tolist() == [0.0, 0.0]


def test_consensus_index_mode_and_instability():
    # atom 0: ICO in 8 frames, distorted in 3 -> mode ICO, instability 3/11
    f_ico = [0, 0, 12, 0, 0, 0]
    f_dis = [0, 2, 8, 2, 0, 0]
    frames = [f_ico] * 8 + [f_dis] * 3              # 11 frames
    fi = np.array(frames).reshape(11, 1, 6)
    con = consensus_index(fi)
    assert con["label"][0].tolist() == [0, 0, 12, 0]
    assert abs(con["instability"][0] - 3 / 11) < 1e-9
    assert con["total"][0] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_consensus.py -v`
Expected: FAIL with `ImportError: cannot import name '_row_mode'`.

- [ ] **Step 3: Implement**

Append to `src/voronoi.py`:

```python
from collections import Counter


def _row_mode(rows):
    """Most common row of an (F,K) integer array, ties broken toward row 0.

    Returns (mode_row (K,) int array, count). Used to aggregate a per-atom Voronoi
    index across trajectory frames into a single time-stable consensus value.
    """
    keys = [tuple(int(v) for v in r) for r in np.asarray(rows)]
    c = Counter(keys)
    top = c.most_common(1)[0][1]
    best = keys[0] if c[keys[0]] == top else next(k for k in keys if c[k] == top)
    return np.array(best, dtype=int), top


def consensus_index(frame_indices):
    """Aggregate per-frame Voronoi indices into a time-stable consensus.

    frame_indices: (F, N, 6) per-frame indices n3..n8 (e.g. stacked `voronoi_index`).
    Returns dict:
      label       (N,4) int  -- per-atom mode of <n3,n4,n5,n6> across frames
      total       (N,)  int  -- per-atom mode coordination (full face count)
      instability (N,)  float-- 1 - (mode-frame-count / F): Voro++'s thermal jitter
    """
    fi = np.asarray(frame_indices)
    F, N, _ = fi.shape
    coord = fi.sum(axis=2)                              # (F,N) full coordination
    label = np.zeros((N, 4), dtype=int)
    total = np.zeros(N, dtype=int)
    instab = np.zeros(N, dtype=float)
    for i in range(N):
        lab, cnt = _row_mode(fi[:, i, :4])
        label[i] = lab
        instab[i] = 1.0 - cnt / F
        tot, _ = _row_mode(coord[:, i].reshape(F, 1))
        total[i] = int(tot[0])
    return dict(label=label, total=total, instability=instab)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_consensus.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/voronoi.py tests/test_consensus.py
git commit -m "$(printf 'Phase 5: consensus Voronoi labeller (mode + instability)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 2: Per-frame Voronoi over a trajectory (`src/voronoi.py`)

**Files:**
- Modify: `src/voronoi.py`
- Test: `tests/test_voronoi_frames.py` (pyvoro-guarded)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voronoi_frames.py
import numpy as np
import pytest

pytest.importorskip("pyvoro")
from src.voronoi import voronoi_index, voronoi_index_frames


def test_voronoi_index_frames_stacks_per_frame():
    rng = np.random.default_rng(0)
    L = np.array([10.0, 10.0, 10.0])
    pos = rng.uniform(0, 10, size=(40, 3))
    radii = np.full(40, 1.3)
    frames = [pos, (pos + 0.01) % L]
    fi = voronoi_index_frames(frames, L, radii)
    assert fi.shape == (2, 40, 6)
    assert np.array_equal(fi[0], voronoi_index(pos, L, radii))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_voronoi_frames.py -v`
Expected: FAIL with `ImportError: cannot import name 'voronoi_index_frames'`.

- [ ] **Step 3: Implement**

Append to `src/voronoi.py`:

```python
def voronoi_index_frames(frames, L, radii, dispersion=4.0):
    """Stack `voronoi_index` over a list of (N,3) position arrays (same L, radii).

    Returns (F, N, 6). This is the teacher signal for Phase 5; pair with
    `consensus_index` to get the time-stable label. Needs pyvoro.
    """
    return np.stack([voronoi_index(p, L, radii, dispersion) for p in frames])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_voronoi_frames.py -v`
Expected: PASS (1 passed), or SKIP if pyvoro is missing.

- [ ] **Step 5: Commit**

```bash
git add src/voronoi.py tests/test_voronoi_frames.py
git commit -m "$(printf 'Phase 5: per-frame Voronoi over a trajectory\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 3: Thermal σ + jitter augmentation (`src/features.py`)

**Files:**
- Modify: `src/features.py` (append; `_minimum_image` already exists there)
- Test: `tests/test_features_robust.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_robust.py
import numpy as np
from src.features import thermal_sigma, jitter


def test_thermal_sigma_matches_known_uniform_shift():
    L = np.array([20.0, 20.0, 20.0])
    a = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    b = a + np.array([0.3, 0.0, 0.0])               # every atom moves 0.3 in x
    assert abs(thermal_sigma([a, b], L) - 0.3) < 1e-9


def test_thermal_sigma_uses_minimum_image():
    L = np.array([10.0, 10.0, 10.0])
    a = np.array([[0.05, 0.0, 0.0]])
    b = np.array([[9.95, 0.0, 0.0]])                # really moved 0.1 across the wall
    assert abs(thermal_sigma([a, b], L) - 0.1) < 1e-9


def test_jitter_zero_sigma_is_identity():
    rng = np.random.default_rng(0)
    L = np.array([10.0, 10.0, 10.0])
    pos = rng.uniform(0, 10, size=(50, 3))
    assert np.array_equal(jitter(pos, 0.0, L, rng), pos)


def test_jitter_is_seeded_and_wrapped():
    L = np.array([10.0, 10.0, 10.0])
    pos = np.full((100, 3), 5.0)
    j1 = jitter(pos, 0.2, L, np.random.default_rng(1))
    j2 = jitter(pos, 0.2, L, np.random.default_rng(1))
    assert np.array_equal(j1, j2)                   # reproducible
    assert (j1 >= 0).all() and (j1 < L).all()       # wrapped into [0,L)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_features_robust.py -v`
Expected: FAIL with `ImportError: cannot import name 'thermal_sigma'`.

- [ ] **Step 3: Implement**

Append to `src/features.py`:

```python
def thermal_sigma(frames, L):
    """RMS per-atom displacement (Angstrom) between consecutive frames.

    `frames` is a list of (N,3) wrapped positions. Minimum-image corrected so an
    atom crossing a periodic wall counts its true (small) displacement. This is the
    physical scale for the Phase-5 jitter augmentation.
    """
    sq = []
    for a, b in zip(frames[:-1], frames[1:]):
        d = _minimum_image(np.asarray(b) - np.asarray(a), L)
        sq.append((d ** 2).sum(axis=1))             # per-atom squared displacement
    return float(np.sqrt(np.concatenate(sq).mean()))


def jitter(pos, sigma, L, rng):
    """Add isotropic Gaussian displacement N(0, sigma^2) per coordinate, re-wrap.

    sigma<=0 returns `pos` unchanged. `rng` is a numpy Generator (seeded by caller)
    so augmentation is reproducible.
    """
    if sigma <= 0:
        return pos
    return (pos + rng.normal(0.0, sigma, size=np.asarray(pos).shape)) % L
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_features_robust.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features_robust.py
git commit -m "$(printf 'Phase 5: thermal sigma + jitter augmentation\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 4: samples1 multi-frame loader (`src/features.py`)

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features_robust.py` (add a guarded loader test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features_robust.py`:

```python
import os
import config
import pytest


@pytest.mark.skipif(not os.path.isdir(config.SAMPLES1), reason="samples1 data not present")
def test_load_samples1_frames_shapes():
    from src.features import load_samples1_frames
    d = load_samples1_frames()
    assert len(d["frames"]) == 11
    assert d["N"] == 13500
    for fr in d["frames"]:
        assert fr.shape == (13500, 3)
        assert (fr >= 0).all() and (fr < d["L"]).all()   # wrapped
    assert d["radius"].shape == (13500,)
    # Cu64Zr36: ~64% type-1 (Cu, r=1.28)
    assert 0.55 < (d["radius"] == 1.28).mean() < 0.72
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_features_robust.py::test_load_samples1_frames_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'load_samples1_frames'` (or SKIP if data absent).

- [ ] **Step 3: Implement**

Append to `src/features.py`:

```python
def load_samples1_frames():
    """Load the samples1 trajectory (Cu64Zr36, 13500 atoms, 11 consecutive frames).

    Returns dict(frames=list of (N,3) wrapped positions, L=(3,), types, radius, N).
    Box is fixed across these equilibrated frames, so a single L is returned. Same
    Cu/Zr radii as samples2 (RADIUS).
    """
    from src.data import read_lammps_frames
    frs = read_lammps_frames(os.path.join(config.SAMPLES1, "LAMMPS_OUTPUT.lammpsTrj"))
    box0 = frs[0]["box"]
    L = (box0[:, 1] - box0[:, 0]).astype(float)
    types = frs[0]["types"].astype(int)
    radius = np.array([RADIUS[t] for t in types], dtype=float)
    frames = [((fr["pos"] - fr["box"][:, 0]) % L).astype(float) for fr in frs]
    return dict(frames=frames, L=L, types=types, radius=radius, N=frs[0]["natoms"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_features_robust.py -v`
Expected: PASS (5 passed; loader test runs because `samples1` is present).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features_robust.py
git commit -m "$(printf 'Phase 5: samples1 multi-frame loader\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 5: Regression model + loss (`src/models.py`)

**Files:**
- Modify: `src/models.py` (append; reuse `CGConv`, keep `CGCNN`)
- Test: `tests/test_models_regressor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_regressor.py
import torch
from src.models import CGCNNRegressor, voronoi_loss


def _toy_graph(N=12, edge_dim=16):
    x = torch.randn(N, 1)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_attr = torch.rand(edge_index.shape[1], edge_dim)
    return x, edge_index, edge_attr


def test_regressor_output_shapes_and_nonneg():
    x, ei, ea = _toy_graph()
    model = CGCNNRegressor(in_dim=1, edge_dim=16, hidden=16, n_layers=2)
    counts, total = model(x, ei, ea)
    assert counts.shape == (12, 4)
    assert total.shape == (12,)
    assert (counts >= 0).all() and (total >= 0).all()    # softplus outputs


def test_regressor_gradients_flow():
    x, ei, ea = _toy_graph()
    model = CGCNNRegressor(in_dim=1, edge_dim=16, hidden=16, n_layers=2)
    counts, total = model(x, ei, ea)
    y_c = torch.zeros(12, 4); y_t = torch.zeros(12)
    loss, *_ = voronoi_loss(counts, total, y_c, y_t)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_voronoi_loss_zero_at_perfect_consistent_prediction():
    # counts sum to total and equal targets, n7/n8 = 0 -> all three terms zero
    counts = torch.tensor([[0.0, 0.0, 12.0, 0.0], [0.0, 2.0, 8.0, 2.0]])
    total = torch.tensor([12.0, 12.0])
    loss, lc, lt, ls = voronoi_loss(counts, total, counts.clone(), total.clone())
    assert float(lc) < 1e-6 and float(lt) < 1e-6 and float(ls) < 1e-6


def test_voronoi_loss_positive_when_wrong():
    counts = torch.tensor([[0.0, 0.0, 12.0, 0.0]])
    total = torch.tensor([12.0])
    y_c = torch.tensor([[0.0, 3.0, 6.0, 4.0]])
    y_t = torch.tensor([13.0])
    loss, *_ = voronoi_loss(counts, total, y_c, y_t)
    assert float(loss) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_models_regressor.py -v`
Expected: FAIL with `ImportError: cannot import name 'CGCNNRegressor'`.

- [ ] **Step 3: Implement**

Append to `src/models.py`:

```python
class CGCNNRegressor(nn.Module):
    """CGCNN trunk + regression head -> 4 non-negative Voronoi counts <n3,n4,n5,n6>
    plus an auxiliary coordination total. Shares `CGConv` with the Phase-2 `CGCNN`
    classifier, which is left untouched. Geometry enters via the edge RBF features,
    so the model is rotation/translation-invariant (half of thermal robustness)."""
    def __init__(self, in_dim, edge_dim, hidden=64, n_layers=3, p=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([CGConv(hidden, edge_dim) for _ in range(n_layers)])
        self.counts = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Dropout(p),
            nn.Linear(hidden, 4),
        )
        self.total = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.softplus(self.embed(x))
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        counts = F.softplus(self.counts(h))             # (N,4) >= 0
        total = F.softplus(self.total(h)).squeeze(1)    # (N,)  >= 0
        return counts, total


def voronoi_loss(counts, total, y_counts, y_total, w_total=0.3, w_sum=0.1):
    """Smooth-L1 on the 4 counts and the coordination total, plus a soft penalty
    tying sum(counts) to the predicted total (pushes the unmodelled n7/n8 slack to 0).

    Returns (loss, l_counts, l_total, l_sum)."""
    l_counts = F.smooth_l1_loss(counts, y_counts)
    l_total = F.smooth_l1_loss(total, y_total)
    l_sum = (counts.sum(dim=1) - total).abs().mean()
    loss = l_counts + w_total * l_total + w_sum * l_sum
    return loss, l_counts, l_total, l_sum
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_models_regressor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models_regressor.py
git commit -m "$(printf 'Phase 5: CGCNN regressor + Voronoi count loss\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 6: Evaluation metrics (`src/metrics.py`)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
from src.metrics import ico_from_counts, flip_rate, exact_match, per_count_mae


def test_ico_from_counts():
    c = np.array([[0, 0, 12, 0], [0, 2, 8, 2], [0, 0, 12, 1]])
    assert ico_from_counts(c).tolist() == [True, False, False]


def test_flip_rate_zero_when_constant():
    idx = np.zeros((11, 5, 4), dtype=int)            # all frames identical
    assert flip_rate(idx) == 0.0


def test_flip_rate_counts_atoms_that_change():
    idx = np.zeros((3, 4, 4), dtype=int)
    idx[1, 0, 2] = 1                                 # atom 0 differs in frame 1
    assert abs(flip_rate(idx) - 0.25) < 1e-9         # 1 of 4 atoms flips


def test_exact_match():
    pred = np.array([[0, 0, 12, 0], [0, 1, 10, 4]])
    true = np.array([[0, 0, 12, 0], [0, 2, 8, 2]])
    assert abs(exact_match(pred, true) - 0.5) < 1e-9


def test_per_count_mae():
    pred = np.array([[0, 0, 12, 0], [0, 0, 10, 2]])
    true = np.array([[0, 0, 12, 0], [0, 0, 12, 0]])
    assert per_count_mae(pred, true).tolist() == [0.0, 0.0, 1.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`.

- [ ] **Step 3: Implement**

Create `src/metrics.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "$(printf 'Phase 5: pure Voronoi-index metrics (flip-rate, exact match, MAE)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 7: Phase-5 runner (`scripts/05_robust_voronoi.py`)

This is the orchestration layer: it imports the tested units above and glues them. It is a self-contained runner like `scripts/01..04`. Its check is a fast `--smoke` run (subsampled, few epochs) that must complete and write a valid JSON; the full run is done in Task 9 (verification). Fix any integration bug found during the smoke run in place before committing.

**Files:**
- Create: `scripts/05_robust_voronoi.py`

- [ ] **Step 1: Write the runner**

Create `scripts/05_robust_voronoi.py`:

```python
"""Phase 5 — a robust, learned Voronoi index that replaces Voro++ at inference.

Distils Voro++ (over the 11 samples1 frames) into a CGCNN regressor that predicts
<n3,n4,n5,n6> from coordinates, trained on the time-stable consensus and augmented
with physically-scaled thermal jitter, then shows a lower temporal flip-rate than
raw Voro++.

Run:  python3 scripts/05_robust_voronoi.py            # full
      python3 scripts/05_robust_voronoi.py --smoke     # fast subsample / few epochs
"""
import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

import config
from src.features import (load_samples1_frames, knn_periodic, rbf_expand,
                          thermal_sigma, jitter)
from src.voronoi import voronoi_index, voronoi_index_frames, consensus_index
from src.models import CGCNNRegressor, voronoi_loss
from src.metrics import ico_from_counts, flip_rate, exact_match, per_count_mae

EDGE_DIM = 16
K = 20


def build_inputs(pos, L, radii, k=K):
    """Coords -> (x, edge_index, edge_attr) tensors. Radius-only node features."""
    ei, ed = knn_periodic(pos, L, k)
    eattr = rbf_expand(ed, n_rbf=EDGE_DIM, cutoff=6.0)
    x = ((radii - radii.mean()) / (radii.std() + 1e-6)).astype(np.float32)[:, None]
    return torch.from_numpy(x), torch.from_numpy(ei), torch.from_numpy(eattr)


def split_atoms(N, seed=0, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr, n_va = int(fracs[0] * N), int(fracs[1] * N)
    tr = np.zeros(N, bool); va = np.zeros(N, bool); te = np.zeros(N, bool)
    tr[perm[:n_tr]] = True
    va[perm[n_tr:n_tr + n_va]] = True
    te[perm[n_tr + n_va:]] = True
    return tr, va, te


def prepare_data(smoke=False):
    d = load_samples1_frames()
    frames, L, radii, N = d["frames"], d["L"], d["radius"], d["N"]
    if smoke:                                           # subsample atoms for speed
        sub = np.sort(np.random.default_rng(0).choice(N, 1500, replace=False))
        frames = [f[sub] for f in frames]; radii = radii[sub]; N = len(sub)
    cache = os.path.join(config.RESULTS, "05_voronoi_frames.npy")
    if not smoke and os.path.exists(cache):
        fi = np.load(cache)
    else:
        fi = voronoi_index_frames(frames, L, radii)     # (F,N,6)
        if not smoke:
            os.makedirs(config.RESULTS, exist_ok=True)
            np.save(cache, fi)
    con = consensus_index(fi)
    sigma = thermal_sigma(frames, L)
    return dict(frames=frames, L=L, radii=radii, N=N, frame_idx=fi,
                label=con["label"], total=con["total"],
                instability=con["instability"], sigma=sigma)


def predict_index(model, pos, L, radii):
    model.eval()
    with torch.no_grad():
        counts, _ = model(*build_inputs(pos, L, radii))
    return np.rint(counts.numpy()).astype(int)          # (N,4)


def train(data, epochs=200, lr=1e-3, sigma_mult=3.0, hidden=64, seed=0, smoke=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frames, L, radii, N = data["frames"], data["L"], data["radii"], data["N"]
    y_c = torch.from_numpy(data["label"].astype(np.float32))
    y_t = torch.from_numpy(data["total"].astype(np.float32))
    tr, va, te = split_atoms(N, seed)
    tr_t, va_t = torch.from_numpy(tr), torch.from_numpy(va)
    base = frames[0]
    sigma_max = sigma_mult * data["sigma"]
    model = CGCNNRegressor(in_dim=1, edge_dim=EDGE_DIM, hidden=hidden, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if smoke:
        epochs = 5
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        s = float(rng.uniform(0.0, sigma_max))          # physically-scaled jitter
        x, ei, ea = build_inputs(jitter(base, s, L, rng), L, radii)
        counts, total = model(x, ei, ea)
        loss, *_ = voronoi_loss(counts[tr_t], total[tr_t], y_c[tr_t], y_t[tr_t])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            c0, t0 = model(*build_inputs(base, L, radii))
            vl, *_ = voronoi_loss(c0[va_t], t0[va_t], y_c[va_t], y_t[va_t])
        if float(vl) < best:
            best = float(vl)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, (tr, va, te)


def jitter_variance(model, pos, L, radii, sigma, n=8, seed=7):
    """Per-atom prediction variance under repeated thermal jitter = learned instability."""
    rng = np.random.default_rng(seed)
    preds = np.stack([predict_index(model, jitter(pos, sigma, L, rng), L, radii)
                      for _ in range(n)])               # (n,N,4)
    return preds.var(axis=0).sum(axis=1)                # (N,)


def evaluate(model, data, te):
    frames, L, radii = data["frames"], data["L"], data["radii"]
    fi, label = data["frame_idx"], data["label"]
    gnn_pf = np.stack([predict_index(model, fr, L, radii) for fr in frames])  # (F,N,4)
    voro_pf = fi[:, :, :4]
    pred0, true = gnn_pf[0][te], label[te]
    base = frames[0]
    # sigma-sweep self-consistency (vs sigma=0) on test atoms
    rng = np.random.default_rng(123)
    sigmas = [float(m * data["sigma"]) for m in (0, 0.5, 1, 1.5, 2, 3)]
    g0 = predict_index(model, base, L, radii)[te]
    v0 = voro_pf[0][te]
    gnn_agree, voro_agree = [], []
    for s in sigmas:
        pj = jitter(base, s, L, rng)
        gnn_agree.append(float((predict_index(model, pj, L, radii)[te] == g0).all(1).mean()))
        voro_agree.append(float((voronoi_index(pj, L, radii)[:, :4][te] == v0).all(1).mean()))
    learned = jitter_variance(model, base, L, radii, data["sigma"])[te]
    voro_instab = data["instability"][te]
    rho = float(spearmanr(learned, voro_instab).correlation)
    # subsample for the scatter panel (keep JSON small)
    si = np.random.default_rng(1).choice(len(learned), min(1000, len(learned)), replace=False)
    return dict(
        per_count_mae=per_count_mae(pred0, true).tolist(),
        exact_match=exact_match(pred0, true),
        ico_f1=float(f1_score(ico_from_counts(true), ico_from_counts(pred0))),
        flip_rate_gnn=flip_rate(gnn_pf[:, te, :]),
        flip_rate_voro=flip_rate(voro_pf[:, te, :]),
        voro_mean_instability=float(voro_instab.mean()),
        sigma_sweep=dict(sigma=sigmas, gnn_agree=gnn_agree, voro_agree=voro_agree),
        instability_spearman=rho,
        scatter=dict(voro=voro_instab[si].tolist(), learned=learned[si].tolist()),
        thermal_sigma=float(data["sigma"]), n_test=int(te.sum()),
    )


def cross_check_samples2(model):
    """Spec §2 cross-system check: apply the samples1-trained model to samples2's
    single clean frame (10k atoms, different size/box) and score vs its fo_list."""
    from src.features import load_samples2
    d = load_samples2()
    pred = predict_index(model, d["pos"], d["L"], d["radius"])   # (N,4)
    true = d["vor"][:, :4]
    return dict(exact_match=exact_match(pred, true),
                ico_f1=float(f1_score(ico_from_counts(true), ico_from_counts(pred))),
                n=int(d["N"]))


def make_figure(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].bar(["Voro++", "GNN"], [m["flip_rate_voro"], m["flip_rate_gnn"]],
                 color=["#cc4444", "#4488cc"])
    ax[0, 0].set_title("Temporal flip-rate (lower = more stable)")
    ax[0, 0].set_ylabel("fraction of test atoms whose index flips")
    sw = m["sigma_sweep"]
    ax[0, 1].plot(sw["sigma"], sw["voro_agree"], "o-", color="#cc4444", label="Voro++")
    ax[0, 1].plot(sw["sigma"], sw["gnn_agree"], "s-", color="#4488cc", label="GNN")
    ax[0, 1].set_xlabel("jitter sigma (A)"); ax[0, 1].set_ylabel("agreement with sigma=0")
    ax[0, 1].set_title("Stability vs thermal jitter"); ax[0, 1].legend()
    ax[1, 0].bar(["n3", "n4", "n5", "n6"], m["per_count_mae"], color="#4488cc")
    ax[1, 0].set_title("Per-count MAE  (exact %.2f, ICO-F1 %.2f)"
                       % (m["exact_match"], m["ico_f1"]))
    ax[1, 1].scatter(m["scatter"]["voro"], m["scatter"]["learned"], s=6, alpha=0.3,
                     color="#4488cc")
    ax[1, 1].set_xlabel("Voro++ instability"); ax[1, 1].set_ylabel("learned jitter variance")
    ax[1, 1].set_title("Learned instability vs Voro++  (rho=%.2f)"
                       % m["instability_spearman"])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    data = prepare_data(smoke=args.smoke)
    print("thermal sigma = %.4f A | Voro++ mean instability = %.4f"
          % (data["sigma"], data["instability"].mean()))
    model, (tr, va, te) = train(data, smoke=args.smoke)
    m = evaluate(model, data, te)
    if not args.smoke:                                  # spec §2 cross-system check
        m["cross_check_samples2"] = cross_check_samples2(model)
        print("samples2 cross-check: exact %.3f | ICO-F1 %.3f"
              % (m["cross_check_samples2"]["exact_match"], m["cross_check_samples2"]["ico_f1"]))
    os.makedirs(config.RESULTS, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    with open(os.path.join(config.RESULTS, "05_robust_voronoi%s.json" % suffix), "w") as f:
        json.dump(m, f, indent=2)
    make_figure(m, os.path.join(config.RESULTS, "05_robust_voronoi%s.png" % suffix))
    print("flip-rate  GNN %.4f  vs  Voro++ %.4f" % (m["flip_rate_gnn"], m["flip_rate_voro"]))
    print("exact %.3f | ICO-F1 %.3f | instability rho %.2f"
          % (m["exact_match"], m["ico_f1"], m["instability_spearman"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run to verify it executes end-to-end**

Run: `python3 scripts/05_robust_voronoi.py --smoke`
Expected: prints `thermal sigma = …`, a `flip-rate GNN … vs Voro++ …` line, and writes `results/05_robust_voronoi_smoke.json` + `.png`. If anything errors, fix the integration bug, re-run until clean.

- [ ] **Step 3: Confirm the smoke JSON is well-formed**

Run: `python3 -c "import json; m=json.load(open('results/05_robust_voronoi_smoke.json')); print(sorted(m)); assert 'flip_rate_gnn' in m and 'sigma_sweep' in m"`
Expected: prints the key list, no assertion error.

- [ ] **Step 4: Commit** (smoke artifacts are throwaway — don't commit them)

```bash
git add scripts/05_robust_voronoi.py
git commit -m "$(printf 'Phase 5: robust-Voronoi runner (train + flip-rate/sigma-sweep eval)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 8: Unify the project framing (Order #1) — README + notebook

No logic changes. Re-narrate the repo as one study; add the Phase-5 result.

**Files:**
- Modify: `README.md`
- Modify: `metallic_glass_gnn.ipynb`

- [ ] **Step 1: Rewrite the README opening (lines ~1–25)**

Replace the title/intro/"Background"/"Reframed task" blocks with a unified framing. New top:

```markdown
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
```

Delete the standalone "Background: what the original thesis found" and "Reframed
task" sections (the thesis framing). Keep all Phase-1..4 *result* tables/numbers.

- [ ] **Step 2: Add the Phase-5 section** (after the Phase-4 section, before "## Data")

```markdown
## Phase-5 result — a robust, learned Voronoi index (`samples1`, 11 frames)

We replace Voro++ at inference with a CGCNN that predicts ⟨n3,n4,n5,n6⟩ from
coordinates, distilled from a **time-stable consensus** of Voro++ over 11
consecutive MD frames and trained with **physically-scaled thermal-jitter
augmentation**. The headline is **temporal stability**: the fraction of atoms whose
index flips frame-to-frame, GNN vs raw Voro++, plus a σ-sweep of agreement under
controlled jitter and a learned per-atom *instability* map (prediction variance
under jitter) that recovers which atoms sit near a topological transition.
(`scripts/05_robust_voronoi.py`, `results/05_robust_voronoi.{json,png}`, design in
[`docs/phase5_robust_voronoi_design.md`](docs/phase5_robust_voronoi_design.md).)

> Numbers filled in from `results/05_robust_voronoi.json` after the full run.
```

- [ ] **Step 3: Update the Roadmap and Layout blocks**

In "## Layout", add:
```
src/metrics.py                 # pure Voronoi-index metrics (flip-rate, exact match, MAE)
scripts/05_robust_voronoi.py   # Phase 5: robust, learned Voronoi index            [DONE]
docs/phase5_robust_voronoi_design.md # Phase 5 design / methods note
```
In "## Roadmap", add:
```
- [x] **Phase 5** — robust, learned Voronoi index: a CGCNN predicts <n3,n4,n5,n6> from
  coordinates with a lower frame-to-frame flip-rate than Voro++ (consensus label +
  thermal-jitter augmentation); replaces the tessellation at inference.
```

- [ ] **Step 4: Soften the Attribution block** (end of README)

Change the opening sentence from "Based on the undergraduate thesis of…" to a
one-line provenance note, keeping the co-author credit:

```markdown
## Acknowledgements

Local-structure data and the original problem framing come from work at Sharif
University of Technology with **Dr. Rouhollah Tavakoli**, who should be credited as
a co-author on any public release or manuscript.
```

- [ ] **Step 5: Reframe the notebook intro + add a Phase-5 section**

Use NotebookEdit. (a) Edit the first markdown cell so it matches the README's
unified framing (drop "reformulation of a B.Sc. thesis"; state the five-phases-one-arc
throughline). (b) Append a markdown cell summarising Phase 5 (consensus label,
jitter augmentation, flip-rate result) and a code cell that loads
`results/05_robust_voronoi.json` and renders the flip-rate / σ-sweep numbers inline,
mirroring how earlier phases display their JSON. Do not re-execute the heavy earlier
cells; only run the new Phase-5 cell.

- [ ] **Step 6: Commit**

```bash
git add README.md metallic_glass_gnn.ipynb
git commit -m "$(printf 'Unify project framing; add Phase 5 to README and notebook\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 9: Full run, verification, and branch finish

**Files:**
- Generates: `results/05_robust_voronoi.{json,png}`, `results/05_voronoi_frames.npy` (cache)
- Modify: `README.md` (fill in the real numbers), notebook cell output

- [ ] **Step 1: Run the whole test suite**

Run: `python3 -m pytest tests/ -v`
Expected: all pass (pyvoro/data-guarded tests run since both are present).

- [ ] **Step 1b: Re-validate the teacher (spec §7)**

Run: `python3 -c "from src.voronoi import validate_against_samples2 as v; r=v(); print(r); assert r['perfect_f1']==1.0"`
Expected: prints the validation dict; `perfect_f1 == 1.000` (pyvoro still reproduces `fo_list`).

- [ ] **Step 2: Full Phase-5 run**

Run: `python3 scripts/05_robust_voronoi.py`
Expected: writes `results/05_robust_voronoi.json` + `.png`; prints the flip-rate line. Sanity-check that `flip_rate_gnn < flip_rate_voro` and `ico_f1` is in a sane range (≳0.7). If the GNN is *not* more stable, that is a real negative result — record it honestly in the README rather than tuning until it looks good; first confirm `sigma_mult`/`epochs` are reasonable and the consensus instability is non-trivial.

- [ ] **Step 3: Fill the real numbers into the README Phase-5 section**

Replace the `> Numbers filled in…` placeholder with the actual flip-rate (GNN vs Voro++), exact-match, ICO-F1, and instability ρ from the JSON. Refresh the notebook Phase-5 cell output.

- [ ] **Step 4: Decide on the cache artifact**

`results/05_voronoi_frames.npy` is a ~6 MB derived cache. Add it to `.gitignore` (the working tree already ignores `report/` etc.) rather than committing it:
```bash
printf '\n# Phase-5 derived Voronoi-label cache\nresults/05_voronoi_frames.npy\n' >> .gitignore
```

- [ ] **Step 5: Commit results + finish**

```bash
git add results/05_robust_voronoi.json results/05_robust_voronoi.png README.md metallic_glass_gnn.ipynb .gitignore
git commit -m "$(printf 'Phase 5: full run results + README/notebook numbers\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

- [ ] **Step 6: Finish the branch**

Invoke `superpowers:finishing-a-development-branch` to choose merge / PR / cleanup for `phase5-robust-voronoi`.

---

## Notes for the executor

- **Determinism:** seeds are passed explicitly (`split_atoms`, `train`, jitter RNGs). Don't introduce `Math.random`-style unseeded calls.
- **Speed:** full run = pyvoro on 13,500×11 cells (tens of seconds, cached) + ~200 full-batch CPU epochs (minutes). Use `--smoke` while iterating.
- **Honesty (spec §10):** the raw 11 frames are ~fs apart, so Voro++'s *raw* flip-rate may be small; the σ-sweep is the headline robustness axis. Report whatever the run gives.
- **Fallback (spec §10):** if radius-only features underperform on exact counts, switch `build_inputs` to an element one-hot (in_dim=2) for the Cu–Zr result and note it.
