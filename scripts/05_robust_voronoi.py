"""Phase 5 — a coords-only learned surrogate for the Voronoi index (replaces Voro++ at inference).

Distils Voro++ (over the 11 samples1 frames) into a per-count CLASSIFICATION CGCNN
that predicts <n3,n4,n5,n6> from coordinates, trained on the time-stable consensus
with a temporal-consistency regulariser (jittered vs clean view). Honest stability story:
on the FULL four-count index the argmax is more self-consistent than re-running Voro++ under
thermal jitter (the sigma-sweep), but that largely reflects argmax stickiness; at the
ICOSAHEDRON level Voro++ is the more stable of the two (lower flip-rate / higher agreement
at every sigma). The sub-thermal 0.01 A frame-to-frame full-index flip-rate is a tie.

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
from scipy.spatial import cKDTree

import config
from src.features import (load_samples1_frames, knn_periodic, rbf_expand,
                          thermal_sigma, jitter)
from src.voronoi import voronoi_index, voronoi_index_frames, consensus_index
from src.models import CGCNNCountClassifier, count_ce_loss, consistency_loss
from src.metrics import ico_from_counts, flip_rate, exact_match, per_count_mae

EDGE_DIM = 16
K = 20
DEBYE_WALLER = 0.12   # ~physical 1D thermal RMS displacement (A); floor for jitter aug


def node_features(pos, L, radii, k=K, shell=3.6):
    """Per-atom rotation/translation-invariant geometry from the kNN cloud (coords
    only -- no Voronoi, so it stays usable at inference): radius, first-shell
    coordination, neighbour bond-length mean/std/min, and local mean neighbour radius."""
    tree = cKDTree(pos, boxsize=L)
    dist, idx = tree.query(pos, k=k + 1)            # (N,k+1); col 0 is self
    d = dist[:, 1:]                                 # (N,k) neighbour distances
    nbr_r = radii[idx[:, 1:]]                       # (N,k) neighbour radii
    coord = (d < shell).sum(1)                      # first-shell coordination
    feats = np.stack([radii, coord, d.mean(1), d.std(1), d[:, 0], nbr_r.mean(1)],
                     axis=1).astype(np.float32)     # (N,6)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    return ((feats - mu) / sd).astype(np.float32)


def build_inputs(pos, L, radii, k=K):
    """Coords -> (x, edge_index, edge_attr) tensors. Rotation-invariant node geometry
    (radius, coordination, bond-length stats) + RBF bond-distance edges."""
    ei, ed = knn_periodic(pos, L, k)
    eattr = rbf_expand(ed, n_rbf=EDGE_DIM, cutoff=6.0)
    x = node_features(pos, L, radii, k)
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
    """argmax per count head -> (N,4) integer index. argmax is locally constant in
    the inputs, so the predicted index is stable under thermal jitter."""
    model.eval()
    with torch.no_grad():
        logits = model(*build_inputs(pos, L, radii))
    return torch.stack([lg.argmax(1) for lg in logits], dim=1).numpy().astype(int)


def train(data, epochs=100, lr=1e-3, sigma_mult=3.0, hidden=128, n_layers=4,
          lam_cons=4.0, seed=0, smoke=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frames, L, radii, N = data["frames"], data["L"], data["radii"], data["N"]
    y_c = torch.from_numpy(data["label"]).long()        # (N,4) integer count targets
    tr, va, te = split_atoms(N, seed)
    tr_t, va_t = torch.from_numpy(tr), torch.from_numpy(va)
    base = frames[0]
    sigma_max = max(sigma_mult * data["sigma"], DEBYE_WALLER)   # exercise physical thermal motion
    in_dim = build_inputs(base, L, radii)[0].shape[1]
    model = CGCNNCountClassifier(in_dim=in_dim, edge_dim=EDGE_DIM, hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if smoke:
        epochs = 5
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        s = float(rng.uniform(0.0, sigma_max))          # physically-scaled jitter
        logits_j = model(*build_inputs(jitter(base, s, L, rng), L, radii))   # jittered view
        logits_c = model(*build_inputs(base, L, radii))                      # clean view
        ce = count_ce_loss([lg[tr_t] for lg in logits_j], y_c[tr_t])         # accuracy (train atoms)
        cons = consistency_loss(logits_j, logits_c)      # all atoms: same index under jitter
        loss = ce + lam_cons * cons
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:        # checkpoint on clean val CE
            model.eval()
            with torch.no_grad():
                lc = model(*build_inputs(base, L, radii))
                vl = float(count_ce_loss([lg[va_t] for lg in lc], y_c[va_t]))
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if not smoke and (ep + 1) % 25 == 0:
                print("  epoch %d/%d  val_ce %.3f cons %.4f (best %.3f)"
                      % (ep + 1, epochs, vl, float(cons), best), flush=True)
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
    # sigma-sweep self-consistency (vs sigma=0) on test atoms, at absolute physical jitter
    # amplitudes -- for the full index AND the coarse icosahedron label
    rng = np.random.default_rng(123)
    sigmas = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]   # absolute A, up to physical thermal amplitude
    g0_full = predict_index(model, base, L, radii)
    v0_full = voro_pf[0]
    g0, v0 = g0_full[te], v0_full[te]
    g0_ico, v0_ico = ico_from_counts(g0_full)[te], ico_from_counts(v0_full)[te]
    gnn_agree, voro_agree, gnn_ico_agree, voro_ico_agree = [], [], [], []
    for s in sigmas:
        pj = jitter(base, s, L, rng)
        gp = predict_index(model, pj, L, radii)
        vp = voronoi_index(pj, L, radii)[:, :4]
        gnn_agree.append(float((gp[te] == g0).all(1).mean()))
        voro_agree.append(float((vp[te] == v0).all(1).mean()))
        gnn_ico_agree.append(float((ico_from_counts(gp)[te] == g0_ico).mean()))
        voro_ico_agree.append(float((ico_from_counts(vp)[te] == v0_ico).mean()))
    learned = jitter_variance(model, base, L, radii, DEBYE_WALLER)[te]
    voro_instab = data["instability"][te]
    rho = float(spearmanr(learned, voro_instab).correlation)
    if not np.isfinite(rho):
        rho = None   # constant input -> undefined correlation; null keeps the JSON valid
    # coarse structural-label temporal stability (GNN vs Voro++): the icosahedron is the
    # descriptor the field actually uses. Here the ordering is the REVERSE of the full-index
    # sigma-sweep -- Voro++ flips about half as often as the GNN at the icosahedron level
    # (perfect-ICO ~0.009 vs ~0.019), i.e. the tessellation is the more stable of the two at the
    # structural-class scale; the GNN's full-index "win" is largely argmax stickiness, not physics.
    def _coarse_flip(per_frame_counts, fn):
        lab = np.stack([fn(per_frame_counts[f]).astype(int)
                        for f in range(per_frame_counts.shape[0])])   # (F,N)
        return flip_rate(lab[:, te, None])
    is_like = lambda c: (np.asarray(c)[:, 2] >= 10)                   # icosahedral-like: n5 >= 10
    coarse = dict(
        ico_flip_gnn=_coarse_flip(gnn_pf, ico_from_counts),
        ico_flip_voro=_coarse_flip(voro_pf, ico_from_counts),
        like_flip_gnn=_coarse_flip(gnn_pf, is_like),
        like_flip_voro=_coarse_flip(voro_pf, is_like),
    )
    # subsample for the scatter panel (keep JSON small)
    si = np.random.default_rng(1).choice(len(learned), min(1000, len(learned)), replace=False)
    return dict(
        coarse_flip=coarse,
        per_count_mae=per_count_mae(pred0, true).tolist(),
        exact_match=exact_match(pred0, true),
        ico_f1=float(f1_score(ico_from_counts(true), ico_from_counts(pred0))),
        flip_rate_gnn=flip_rate(gnn_pf[:, te, :]),
        flip_rate_voro=flip_rate(voro_pf[:, te, :]),
        voro_mean_instability=float(voro_instab.mean()),
        sigma_sweep=dict(sigma=sigmas, gnn_agree=gnn_agree, voro_agree=voro_agree,
                         gnn_ico_agree=gnn_ico_agree, voro_ico_agree=voro_ico_agree),
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
    # frame-to-frame flip-rate: full index (a tie) vs perfect-ICO (Voro++ is the more stable one)
    cf = m["coarse_flip"]
    xb = np.arange(2)
    ax[0, 0].bar(xb - 0.2, [m["flip_rate_voro"], cf["ico_flip_voro"]], 0.4, color="#cc4444", label="Voro++")
    ax[0, 0].bar(xb + 0.2, [m["flip_rate_gnn"], cf["ico_flip_gnn"]], 0.4, color="#4488cc", label="GNN")
    ax[0, 0].set_xticks(xb); ax[0, 0].set_xticklabels(["full index\n(tie)", "perfect-ICO\n(Voro++ wins)"])
    ax[0, 0].set_title("Frame-to-frame flip-rate (lower = more stable)")
    ax[0, 0].set_ylabel("fraction of test atoms whose label flips"); ax[0, 0].legend()
    sw = m["sigma_sweep"]
    ax[0, 1].plot(sw["sigma"], sw["voro_agree"], "o-", color="#cc4444", label="Voro++ (full index)")
    ax[0, 1].plot(sw["sigma"], sw["gnn_agree"], "s-", color="#4488cc", label="GNN (full index)")
    ax[0, 1].plot(sw["sigma"], sw["voro_ico_agree"], "o--", color="#cc4444", label="Voro++ (icosahedron)")
    ax[0, 1].plot(sw["sigma"], sw["gnn_ico_agree"], "s--", color="#4488cc", label="GNN (icosahedron)")
    ax[0, 1].set_xlabel("jitter sigma (A)"); ax[0, 1].set_ylabel("agreement with sigma=0")
    ax[0, 1].set_title("Self-consistency vs thermal jitter\n(solid = full index, dashed = icosahedron)")
    ax[0, 1].legend(fontsize=8)
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
    print("full-index flip-rate   GNN %.4f  vs  Voro++ %.4f" % (m["flip_rate_gnn"], m["flip_rate_voro"]))
    cf = m["coarse_flip"]
    print("perfect-ICO flip-rate  GNN %.4f  vs  Voro++ %.4f" % (cf["ico_flip_gnn"], cf["ico_flip_voro"]))
    print("ICO-like   flip-rate   GNN %.4f  vs  Voro++ %.4f" % (cf["like_flip_gnn"], cf["like_flip_voro"]))
    rho_str = "%.2f" % m["instability_spearman"] if m["instability_spearman"] is not None else "N/A"
    print("exact %.3f | ICO-F1 %.3f | instability rho %s"
          % (m["exact_match"], m["ico_f1"], rho_str))


if __name__ == "__main__":
    main()
