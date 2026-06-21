"""Phase 5 — a robust, learned Voronoi index that replaces Voro++ at inference.

Distils Voro++ (over the 11 samples1 frames) into a CGCNN regressor that predicts
<n3,n4,n5,n6> from coordinates, trained on the time-stable consensus and augmented
with physically-scaled thermal jitter, then shows a lower temporal flip-rate than
raw Voro++.

Run:  python3 scripts/05_robust_voronoi.py            # full
      python3 scripts/05_robust_voronoi.py --smoke     # fast subsample / few epochs
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    if not np.isfinite(rho):
        rho = None   # constant input -> undefined correlation; null keeps the JSON valid
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
    rho_str = "%.2f" % m["instability_spearman"] if m["instability_spearman"] is not None else "N/A"
    ax[1, 1].set_title("Learned instability vs Voro++  (rho=%s)" % rho_str)
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
    rho_str = "%.2f" % m["instability_spearman"] if m["instability_spearman"] is not None else "N/A"
    print("exact %.3f | ICO-F1 %.3f | instability rho %s"
          % (m["exact_match"], m["ico_f1"], rho_str))


if __name__ == "__main__":
    main()
