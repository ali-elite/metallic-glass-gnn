"""Phase 8 (exploratory) — Calibrated per-atom uncertainty (proposal E).

A coords-only GNN can't beat Voro++ on robustness (Phases 5-7), but it CAN offer something
Voro++ cannot from a single frame: a per-atom "is this descriptor thermally ambiguous?"
score. Phase 5's attempt was broken (learned-instability vs Voro++-instability Spearman
~ -0.12). Here we test, for the Phase-7 coordination + icosahedral-like GNN, whether two
uncertainty signals predict which atoms ACTUALLY flip across the 11 frames:

  * predictive entropy  -- softmax entropy of each head on the clean frame
  * jitter-instability  -- fraction of jittered copies whose argmax differs from the clean
                           prediction (the "learned instability", done right)

Ground truth (from the 11 Voro++ frames): per-atom flip (binary: does the descriptor change
across frames) and continuous instability (1 - mode-frame-fraction). We score the
uncertainty signals with ROC-AUC (predict the flip) and Spearman (vs continuous instability),
on TEST atoms, for both coordination and ico-like.

Run:  python3 scripts/10_uncertainty.py [--smoke] [--lam 4.0]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import config

# reuse Phase-7 training + Phase-5 data prep
_spec = importlib.util.spec_from_file_location(
    "p8", os.path.join(os.path.dirname(__file__), "08_robust_coordination.py"))
p8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p8)
p5 = p8.p5
from src.features import jitter
DEBYE_WALLER = p8.DEBYE_WALLER


def softmax_entropy(model, pos, L, radii):
    model.eval()
    with torch.no_grad():
        cl, il = model(*p5.build_inputs(pos, L, radii))
    pc, pi = F.softmax(cl, 1).numpy(), F.softmax(il, 1).numpy()
    ent_c = -(pc * np.log(pc + 1e-12)).sum(1)
    ent_i = -(pi * np.log(pi + 1e-12)).sum(1)
    return ent_c, ent_i


def jitter_instability(model, base, L, radii, sigma, n=8, seed=7):
    """Fraction of n jittered copies whose argmax differs from the clean prediction."""
    rng = np.random.default_rng(seed)
    c0, i0 = p8.predict(model, base, L, radii)
    dc = np.zeros(len(c0)); di = np.zeros(len(i0))
    for _ in range(n):
        c, i = p8.predict(model, jitter(base, sigma, L, rng), L, radii)
        dc += (c != c0); di += (i != i0)
    return dc / n, di / n


def frame_instability(pf):
    """Per-atom flip (binary) and instability (1 - mode fraction) over frames. pf: (F,N)."""
    F_, N_ = pf.shape
    flip = np.zeros(N_, int); inst = np.zeros(N_)
    for i in range(N_):
        _, counts = np.unique(pf[:, i], return_counts=True)
        flip[i] = int(len(counts) > 1)
        inst[i] = 1.0 - counts.max() / F_
    return flip, inst


def score(unc, flip, inst, te):
    """ROC-AUC (predict flip) + Spearman (vs continuous instability) on test atoms."""
    u, fl, ins = unc[te], flip[te], inst[te]
    auc = float(roc_auc_score(fl, u)) if 0 < fl.sum() < len(fl) else None
    rho = float(spearmanr(u, ins).correlation)
    return dict(roc_auc=auc, spearman=rho)


def make_figure(m, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for name, (fl, u, auc) in curves.items():
        fpr, tpr, _ = roc_curve(fl, u)
        ax[0].plot(fpr, tpr, label="%s (AUC %.2f)" % (name, auc))
    ax[0].plot([0, 1], [0, 1], ":", color="grey")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("Does GNN uncertainty predict actual flips?"); ax[0].legend(fontsize=8)
    # reliability: bin test atoms by coord jitter-instability, show actual flip-rate
    rb = m["reliability"]
    ax[1].plot(rb["bin_uncertainty"], rb["actual_flip_rate"], "o-", color="#4488cc")
    ax[1].plot([0, max(rb["bin_uncertainty"]) or 1], [0, max(rb["bin_uncertainty"]) or 1], ":", color="grey")
    ax[1].set_xlabel("GNN jitter-instability (coordination)")
    ax[1].set_ylabel("actual coordination flip-rate")
    ax[1].set_title("Calibration (binned)"); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lam", type=float, default=4.0, help="consistency weight of the trained model")
    args = ap.parse_args()
    data = p5.prepare_data(smoke=args.smoke)
    frames, L, radii = data["frames"], data["L"], data["radii"]
    n_coord = int(data["total"].max()) + 1
    print("samples1: N=%d, %d frames | coord classes=%d | lam=%.1f"
          % (data["N"], len(frames), n_coord, args.lam), flush=True)
    model, (tr, va, te) = p8.train(data, n_coord, lam=args.lam, smoke=args.smoke)

    # uncertainty signals on the clean base frame
    ent_c, ent_i = softmax_entropy(model, frames[0], L, radii)
    n_jit = 4 if args.smoke else 8
    jit_c, jit_i = jitter_instability(model, frames[0], L, radii, DEBYE_WALLER, n=n_jit)

    # ground-truth thermal instability from the Voro++ frames
    fi = data["frame_idx"]
    coord_pf = fi.sum(axis=2)
    ico_pf = (fi[:, :, 2] >= 10).astype(int)
    gt_coord_flip, gt_coord_inst = frame_instability(coord_pf)
    gt_ico_flip, gt_ico_inst = frame_instability(ico_pf)

    res = {
        "coord_entropy": score(ent_c, gt_coord_flip, gt_coord_inst, te),
        "coord_jitter": score(jit_c, gt_coord_flip, gt_coord_inst, te),
        "ico_entropy": score(ent_i, gt_ico_flip, gt_ico_inst, te),
        "ico_jitter": score(jit_i, gt_ico_flip, gt_ico_inst, te),
    }
    # reliability for coordination jitter-instability (deciles on test atoms)
    u, fl = jit_c[te], gt_coord_flip[te]
    order = np.argsort(u)
    nb = 10
    bins = np.array_split(order, nb)
    rb = dict(bin_uncertainty=[float(u[b].mean()) for b in bins],
              actual_flip_rate=[float(fl[b].mean()) for b in bins])
    base_flip = dict(coord=float(gt_coord_flip[te].mean()), ico=float(gt_ico_flip[te].mean()))

    out = dict(n_test=int(te.sum()), lam=args.lam, base_flip_rate=base_flip,
               scores=res, reliability=rb)
    os.makedirs(config.RESULTS, exist_ok=True)
    suffix = "_smoke" if args.smoke else ("_lam%g" % args.lam)
    json.dump(out, open(os.path.join(config.RESULTS, "10_uncertainty%s.json" % suffix), "w"), indent=2)
    curves = {
        "coord entropy": (gt_coord_flip[te], ent_c[te], res["coord_entropy"]["roc_auc"] or 0.5),
        "coord jitter": (gt_coord_flip[te], jit_c[te], res["coord_jitter"]["roc_auc"] or 0.5),
        "ico jitter": (gt_ico_flip[te], jit_i[te], res["ico_jitter"]["roc_auc"] or 0.5),
    }
    make_figure(out, curves, os.path.join(config.RESULTS, "10_uncertainty%s.png" % suffix))

    print("\nbase flip-rate (test): coord %.3f | ico-like %.3f" % (base_flip["coord"], base_flip["ico"]))
    print("\n%-14s %10s %10s" % ("signal", "ROC-AUC", "Spearman"))
    for k in ["coord_entropy", "coord_jitter", "ico_entropy", "ico_jitter"]:
        a = res[k]["roc_auc"]; r = res[k]["spearman"]
        print("%-14s %10s %10.3f" % (k, ("%.3f" % a) if a is not None else "n/a", r))
    print("\n(Phase-5 baseline: learned-instability vs Voro++ full-index instability Spearman ~ -0.12)")


if __name__ == "__main__":
    main()
