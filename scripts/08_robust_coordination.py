"""Phase 7 — A coords-only GNN that predicts the ROBUST frontier descriptor and acts as a
thermal denoiser, benchmarked honestly against single-frame Voro++.

Phase 6/6b showed the robustness<->informativeness Pareto front is held by coordination
number and the joint (coordination, icosahedral-like) descriptor -- NOT the fragile
face-count index or raw topology. This trains a CGCNN (coords-only, rotation/translation
invariant) to predict that frontier target from a single frame, distilled from the
time-stable consensus over the 11 samples1 frames with a temporal-consistency regulariser.

The honest claim we test: a GNN trained frame->consensus is a DENOISER -- given one noisy
frame it should recover the time-stable consensus descriptor MORE often than re-running
Voro++ on that same frame (Voro++ has no notion of temporal consensus). We report, per
frame averaged:
  * agreement-with-consensus (GNN vs Voro++)  -- the denoising metric
  * frame-to-frame flip-rate (GNN vs Voro++)  -- temporal stability
  * sigma-sweep self-consistency (GNN vs Voro++)
plus accuracy vs the consensus labels.

Targets (both on the Pareto frontier): coordination (total #faces) and icosahedral-like
(n5 >= 10); their product is the `coord_n5like` descriptor from Phase 6b.

Run:  python3 scripts/08_robust_coordination.py [--smoke]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import config
from src.models import CGConv
from src.features import jitter
from src.voronoi import voronoi_index
from src.metrics import flip_rate

# reuse Phase-5 data prep + coords-only feature builder (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "p5", os.path.join(os.path.dirname(__file__), "05_robust_voronoi.py"))
p5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p5)
EDGE_DIM, DEBYE_WALLER = p5.EDGE_DIM, p5.DEBYE_WALLER


class CGCNN2Head(nn.Module):
    """CGCNN trunk + two classification heads: coordination (multi-class) and
    icosahedral-like (binary). argmax per head is locally constant -> jitter-stable."""
    def __init__(self, in_dim, edge_dim, n_coord, hidden=128, n_layers=4, p=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([CGConv(hidden, edge_dim) for _ in range(n_layers)])
        self.coord_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Softplus(),
                                        nn.Dropout(p), nn.Linear(hidden, n_coord))
        self.ico_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Softplus(),
                                      nn.Dropout(p), nn.Linear(hidden, 2))

    def forward(self, x, ei, ea):
        h = F.softplus(self.embed(x))
        for conv in self.convs:
            h = conv(h, ei, ea)
        return self.coord_head(h), self.ico_head(h)


def kl_consistency(a, b):
    """Symmetric KL between two predictive distributions of one head (jitter stability)."""
    la, lb = F.log_softmax(a, 1), F.log_softmax(b, 1)
    return 0.5 * (F.kl_div(la, lb, reduction="batchmean", log_target=True)
                  + F.kl_div(lb, la, reduction="batchmean", log_target=True))


def predict(model, pos, L, radii):
    model.eval()
    with torch.no_grad():
        cl, il = model(*p5.build_inputs(pos, L, radii))
    return cl.argmax(1).numpy().astype(int), il.argmax(1).numpy().astype(int)


def train(data, n_coord, epochs=150, lr=1e-3, hidden=128, n_layers=4, lam=4.0,
          w_ico=3.0, sigma_mult=3.0, seed=0, smoke=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frames, L, radii, N = data["frames"], data["L"], data["radii"], data["N"]
    y_coord = torch.from_numpy(np.clip(data["total"], 0, n_coord - 1)).long()
    y_ico = torch.from_numpy((data["label"][:, 2] >= 10).astype(int)).long()
    tr, va, te = p5.split_atoms(N, seed)
    tr_t, va_t = torch.from_numpy(tr), torch.from_numpy(va)
    # class-weight the (imbalanced) icosahedral-like head so it is not starved by the
    # 21-class coordination head; weight both heads' losses comparably (w_ico)
    pos = int((y_ico[tr_t] == 1).sum()); neg = int((y_ico[tr_t] == 0).sum())
    ico_w = torch.tensor([1.0, max(neg, 1) / max(pos, 1)], dtype=torch.float32)
    base = frames[0]
    sigma_max = max(sigma_mult * data["sigma"], DEBYE_WALLER)
    clean = p5.build_inputs(base, L, radii)          # clean view is constant across epochs (cache)
    in_dim = clean[0].shape[1]
    model = CGCNN2Head(in_dim, EDGE_DIM, n_coord, hidden, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if smoke:
        epochs = 5
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        s = float(rng.uniform(0.0, sigma_max))
        cj, ij = model(*p5.build_inputs(jitter(base, s, L, rng), L, radii))   # jittered view
        cc, ic = model(*clean)                                                # clean view (cached)
        ce = F.cross_entropy(cj[tr_t], y_coord[tr_t]) + w_ico * F.cross_entropy(ij[tr_t], y_ico[tr_t], weight=ico_w)
        cons = kl_consistency(cj, cc) + kl_consistency(ij, ic)
        loss = ce + lam * cons
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                cc, ic = model(*clean)
                vl = float(F.cross_entropy(cc[va_t], y_coord[va_t])
                           + F.cross_entropy(ic[va_t], y_ico[va_t]))
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if not smoke and (ep + 1) % 30 == 0:
                print("  epoch %d/%d  val %.3f cons %.4f (best %.3f)"
                      % (ep + 1, epochs, vl, float(cons), best), flush=True)
    if best_state:
        model.load_state_dict(best_state)
    return model, (tr, va, te)


def evaluate(model, data, te, n_coord):
    frames, L, radii = data["frames"], data["L"], data["radii"]
    fi = data["frame_idx"]                          # (F,N,6) per-frame Voro++ index n3..n8
    coord_cons = np.clip(data["total"], 0, n_coord - 1)
    ico_cons = (data["label"][:, 2] >= 10).astype(int)
    # per-frame predictions
    gnn = [predict(model, fr, L, radii) for fr in frames]
    gnn_coord = np.stack([g[0] for g in gnn])       # (F,N)
    gnn_ico = np.stack([g[1] for g in gnn])
    voro_coord = fi.sum(axis=2)                      # (F,N) total #faces
    voro_ico = (fi[:, :, 2] >= 10).astype(int)

    def agree_consensus(pred_pf, cons):             # mean over frames of test-atom agreement
        return float(np.mean([(pred_pf[f][te] == cons[te]).mean() for f in range(len(frames))]))

    denoise = dict(
        coord_gnn=agree_consensus(gnn_coord, coord_cons),
        coord_voro=agree_consensus(voro_coord, coord_cons),
        ico_gnn=agree_consensus(gnn_ico, ico_cons),
        ico_voro=agree_consensus(voro_ico, ico_cons),
        joint_gnn=float(np.mean([((gnn_coord[f][te] == coord_cons[te]) &
                                  (gnn_ico[f][te] == ico_cons[te])).mean() for f in range(len(frames))])),
        joint_voro=float(np.mean([((voro_coord[f][te] == coord_cons[te]) &
                                   (voro_ico[f][te] == ico_cons[te])).mean() for f in range(len(frames))])),
    )
    flips = dict(
        coord_gnn=flip_rate(gnn_coord[:, te, None]), coord_voro=flip_rate(voro_coord[:, te, None]),
        ico_gnn=flip_rate(gnn_ico[:, te, None]), ico_voro=flip_rate(voro_ico[:, te, None]),
    )
    # accuracy vs consensus on frame 0 (test atoms)
    acc = dict(
        coord_exact=float((gnn_coord[0][te] == coord_cons[te]).mean()),
        coord_within1=float((np.abs(gnn_coord[0][te] - coord_cons[te]) <= 1).mean()),
        ico_f1=float(f1_score(ico_cons[te], gnn_ico[0][te])),
    )
    # sigma-sweep self-consistency (GNN vs Voro++) on coordination
    rng = np.random.default_rng(123)
    sigmas = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    base = frames[0]
    g0c, g0i = predict(model, base, L, radii)
    v0 = voronoi_index(base, L, radii)
    v0c = v0.sum(axis=1); v0i = (v0[:, 2] >= 10).astype(int)
    sweep = {k: [] for k in ["coord_gnn", "coord_voro", "ico_gnn", "ico_voro"]}
    for s in sigmas:
        pj = jitter(base, s, L, rng)
        gc, gi = predict(model, pj, L, radii)
        vidx = voronoi_index(pj, L, radii)
        vc = vidx.sum(axis=1); vi = (vidx[:, 2] >= 10).astype(int)
        sweep["coord_gnn"].append(float((gc[te] == g0c[te]).mean()))
        sweep["coord_voro"].append(float((vc[te] == v0c[te]).mean()))
        sweep["ico_gnn"].append(float((gi[te] == g0i[te]).mean()))
        sweep["ico_voro"].append(float((vi[te] == v0i[te]).mean()))
    return dict(n_test=int(te.sum()), n_coord=int(n_coord),
                denoise_vs_consensus=denoise, flip_rate=flips, accuracy=acc,
                sigma_sweep=dict(sigma=sigmas, **sweep))


def make_figure(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    d = m["denoise_vs_consensus"]
    g = [d["coord_gnn"], d["ico_gnn"], d["joint_gnn"]]
    v = [d["coord_voro"], d["ico_voro"], d["joint_voro"]]
    x = np.arange(3)
    ax[0].bar(x - 0.2, v, 0.4, color="#cc4444", label="Voro++ (1 frame)")
    ax[0].bar(x + 0.2, g, 0.4, color="#4488cc", label="GNN (1 frame)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["coordination", "ico-like", "joint"])
    ax[0].set_ylabel("agreement with consensus"); ax[0].set_ylim(0, 1)
    ax[0].set_title("Denoising: recover consensus from 1 frame\n(higher = better)"); ax[0].legend()
    for i, (vv, gg) in enumerate(zip(v, g)):
        ax[0].text(i - 0.2, vv, "%.2f" % vv, ha="center", va="bottom", fontsize=8)
        ax[0].text(i + 0.2, gg, "%.2f" % gg, ha="center", va="bottom", fontsize=8)
    fr = m["flip_rate"]
    ax[1].bar(x[:2] - 0.2, [fr["coord_voro"], fr["ico_voro"]], 0.4, color="#cc4444", label="Voro++")
    ax[1].bar(x[:2] + 0.2, [fr["coord_gnn"], fr["ico_gnn"]], 0.4, color="#4488cc", label="GNN")
    ax[1].set_xticks(x[:2]); ax[1].set_xticklabels(["coordination", "ico-like"])
    ax[1].set_ylabel("frame-to-frame flip-rate")
    ax[1].set_title("Temporal flip-rate (lower = more stable)"); ax[1].legend()
    sw = m["sigma_sweep"]
    ax[2].plot(sw["sigma"], sw["coord_voro"], "o-", color="#cc4444", label="Voro++ coord")
    ax[2].plot(sw["sigma"], sw["coord_gnn"], "s-", color="#4488cc", label="GNN coord")
    ax[2].set_xlabel("jitter sigma (A)"); ax[2].set_ylabel("agreement with sigma=0")
    ax[2].set_title("Self-consistency vs jitter (coordination)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    data = p5.prepare_data(smoke=args.smoke)
    n_coord = int(data["total"].max()) + 1
    print("samples1: N=%d, %d frames | coord classes=%d | thermal sigma=%.4f A"
          % (data["N"], len(data["frames"]), n_coord, data["sigma"]))
    model, (tr, va, te) = train(data, n_coord, smoke=args.smoke)
    m = evaluate(model, data, te, n_coord)

    os.makedirs(config.RESULTS, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    json.dump(m, open(os.path.join(config.RESULTS, "08_robust_coordination%s.json" % suffix), "w"), indent=2)
    make_figure(m, os.path.join(config.RESULTS, "08_robust_coordination%s.png" % suffix))

    d, fr, ac = m["denoise_vs_consensus"], m["flip_rate"], m["accuracy"]
    print("\n=== denoising: agreement with consensus from a single frame (higher = better) ===")
    print("  coordination :  GNN %.3f  vs  Voro++ %.3f" % (d["coord_gnn"], d["coord_voro"]))
    print("  ico-like     :  GNN %.3f  vs  Voro++ %.3f" % (d["ico_gnn"], d["ico_voro"]))
    print("  joint        :  GNN %.3f  vs  Voro++ %.3f" % (d["joint_gnn"], d["joint_voro"]))
    print("=== frame-to-frame flip-rate (lower = more stable) ===")
    print("  coordination :  GNN %.4f  vs  Voro++ %.4f" % (fr["coord_gnn"], fr["coord_voro"]))
    print("  ico-like     :  GNN %.4f  vs  Voro++ %.4f" % (fr["ico_gnn"], fr["ico_voro"]))
    print("=== accuracy vs consensus (frame 0, test atoms) ===")
    print("  coordination exact %.3f | within-1 %.3f | ico-like F1 %.3f"
          % (ac["coord_exact"], ac["coord_within1"], ac["ico_f1"]))


if __name__ == "__main__":
    main()
