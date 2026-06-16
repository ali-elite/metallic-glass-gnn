"""Phase 4: cross-chemistry transfer of the icosahedron detector.

Does a GNN that learned the perfect icosahedron <0,0,12,0> in Cu-Zr (samples2)
transfer *zero-shot* to other glass chemistries? samples3 ships trajectories only,
so we label every target with the SAME radical-Voronoi method (pyvoro) that
produced the samples2 labels - validated to reproduce `fo_list` perfect-ICO at
F1 = 1.000 (see src.voronoi.validate_against_samples2).

Transfer setup (the crux): the model uses ELEMENT-AGNOSTIC node features - each
atom carries only its radius (standardised with the SOURCE statistics, so a given
size maps to the same feature everywhere) - plus a periodic kNN graph with RBF
bond-distance edges. No element identity, so the same model can be applied to
Co/W, Ni/Zr, Cu/Zr/Al, etc. We train once on Cu-Zr and evaluate zero-shot on:

  CuZr {46:54, 50:50, 64:36}  (same chemistry, new composition)
  NiZr {46:54, 50:50, 64:36}  (new element, similar size ratio)
  Cu-Zr-Al {Al 5..25%}        (added third element)
  Co-W {W 10..85%}            (entirely different alloy)

For one representative per family we also train an in-domain "oracle" to show the
transfer gap. Be honest: transfer should fade with chemical distance.
"""
import os, sys, re, glob, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             accuracy_score)

import config
from src.data import is_perfect_icosahedron
from src.features import (load_samples2, load_samples3_target, knn_periodic,
                          rbf_expand)
from src.models import CGCNN
from src.voronoi import voronoi_index, validate_against_samples2

warnings.filterwarnings("ignore")
SEED = 0
K_GRAPH = 16
EPOCHS = 250
RBF, CUTOFF = 16, 6.0

RADII = {"Cu": 1.28, "Zr": 1.60, "Ni": 1.24, "Al": 1.42, "Co": 1.25, "W": 1.37}


# --------------------------------------------------------------------------- #
#  target registry                                                            #
# --------------------------------------------------------------------------- #
def build_targets():
    S3 = config.SAMPLES3
    T = []
    for al in ["Al05", "Al10", "Al15", "Al20", "Al25"]:
        T.append(dict(name=f"CuZrAl-{al}", family="Cu-Zr-Al",
                      path=os.path.join(S3, "Cu-Zr-Al", al),
                      tr={1: RADII["Cu"], 2: RADII["Zr"], 3: RADII["Al"]},
                      comp=int(al[2:])))                       # Al %
    for sysname, elem in [("CuZr", "Cu"), ("NiZr", "Ni")]:
        for comp in ["4654", "5050", "6436"]:
            d = os.path.join(S3, "CuZr-NiZr", f"{sysname}{comp}")
            traj = glob.glob(os.path.join(d, "*.lammpsTrj"))
            if traj:
                T.append(dict(name=f"{sysname}{comp}", family=sysname, path=traj[0],
                              tr={1: RADII[elem], 2: RADII["Zr"]}, comp=int(comp[:2])))
    for d in sorted(glob.glob(os.path.join(S3, "Co-W", "co-w*"))):
        traj = glob.glob(os.path.join(d, "*.lammpsTrj"))
        m = re.search(r"co-w(\d+)", os.path.basename(d))
        if traj and m:
            T.append(dict(name=f"CoW-{m.group(1)}", family="Co-W", path=traj[0],
                          tr={1: RADII["Co"], 2: RADII["W"]}, comp=int(m.group(1))))  # W %
    return T


# --------------------------------------------------------------------------- #
#  inputs / metrics / training                                                #
# --------------------------------------------------------------------------- #
def build_inputs(pos, L, radius, mu, sd):
    """Radius-only node features (source-standardised) + periodic kNN graph."""
    x = torch.tensor(((radius - mu) / sd).astype(np.float32)[:, None])
    ei, ed = knn_periodic(pos, L, k=K_GRAPH)
    ea = torch.tensor(rbf_expand(ed, n_rbf=RBF, cutoff=CUTOFF))
    return x, torch.tensor(ei), ea


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    out = dict(n=int(len(y)), base_rate=float(y.mean()), true_frac=float(y.mean()),
               pred_frac=float(pred.mean()), acc=float(accuracy_score(y, pred)),
               f1=float(f1_score(y, pred, zero_division=0)))
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    else:
        out["roc_auc"] = None; out["pr_auc"] = None
    return out


def predict(model, x, ei, ea):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x, ei, ea), 1)[:, 1].numpy()


def train_cgcnn(x, ei, ea, y, tr, va, tag=""):
    """Class-weighted CGCNN; keep the best-validation-macro-F1 weights."""
    torch.manual_seed(SEED)
    model = CGCNN(in_dim=x.shape[1], edge_dim=ea.shape[1])
    y_t = torch.tensor(y)
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max((y[tr] == 1).sum(), 1)],
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    tr_t, va_t = torch.tensor(tr), torch.tensor(va)
    best, best_state, t0 = -1, None, time.time()
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = crit(model(x, ei, ea)[tr_t], y_t[tr_t]); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            prob = predict(model, x, ei, ea)
            vm = f1_score(y[va], (prob[va] >= 0.5).astype(int), average="macro", zero_division=0)
            if vm > best:
                best = vm; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  [{tag}] {EPOCHS} ep in {time.time()-t0:.0f}s (best val macro-F1 {best:.3f})")
    return model


# --------------------------------------------------------------------------- #
#  main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    t_start = time.time()

    print("validating pyvoro labeller against samples2 fo_list ...")
    val = validate_against_samples2()
    print(f"  perfect-ICO F1 {val['perfect_f1']:.3f} "
          f"({val['perfect_pyvoro']} vs {val['perfect_fo_list']}); "
          f"n5>=10 agreement {val['like_agreement']:.3f}")

    # ---- source: train the icosahedron detector on Cu-Zr (samples2) ----
    d = load_samples2()
    mu, sd = float(d["radius"].mean()), float(d["radius"].std() + 1e-6)
    y_src = is_perfect_icosahedron(d["vor"]).astype(np.int64)
    x, ei, ea = build_inputs(d["pos"], d["L"], d["radius"], mu, sd)
    idx_all = np.arange(d["N"])
    tr, tmp = train_test_split(idx_all, test_size=0.30, random_state=SEED, stratify=y_src)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y_src[tmp])
    print(f"\nsource (Cu-Zr) radius-only model | base rate {100*y_src.mean():.1f}%")
    model = train_cgcnn(x, ei, ea, y_src, tr, va, tag="source Cu-Zr")
    src_test = metrics(y_src[te], predict(model, x, ei, ea)[te])
    print(f"  source test: F1 {src_test['f1']:.3f}  ROC-AUC {src_test['roc_auc']:.3f}")

    # ---- zero-shot transfer to every target chemistry ----
    targets = build_targets()
    print(f"\nzero-shot transfer to {len(targets)} target chemistries:")
    results = {}
    for t in targets:
        td = load_samples3_target(t["path"], t["tr"])
        vidx = voronoi_index(td["pos"], td["L"], td["radius"])
        y = is_perfect_icosahedron(vidx).astype(np.int64)
        xt, eit, eat = build_inputs(td["pos"], td["L"], td["radius"], mu, sd)
        m = metrics(y, predict(model, xt, eit, eat))
        m.update(family=t["family"], comp=t["comp"], N=td["N"])
        results[t["name"]] = m
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        print(f"  {t['name']:14s} ({t['family']:8s}) base {100*m['base_rate']:5.1f}%  "
              f"F1 {m['f1']:.3f}  ROC-AUC {auc}")

    # ---- in-domain oracles (representative per family) ----
    print("\nin-domain oracles (upper bound) for one target per family:")
    oracles = {}
    for name in ["CuZr6436", "NiZr6436", "CuZrAl-Al10", "CoW-50"]:
        t = next(x for x in targets if x["name"] == name)
        td = load_samples3_target(t["path"], t["tr"])
        y = is_perfect_icosahedron(voronoi_index(td["pos"], td["L"], td["radius"])).astype(np.int64)
        if not (0 < y.sum() < len(y)):
            continue
        xt, eit, eat = build_inputs(td["pos"], td["L"], td["radius"], mu, sd)
        otr, ote = train_test_split(np.arange(td["N"]), test_size=0.30,
                                    random_state=SEED, stratify=y)
        otr, ova = train_test_split(otr, test_size=0.20, random_state=SEED, stratify=y[otr])
        om = train_cgcnn(xt, eit, eat, y, otr, ova, tag=f"oracle {name}")
        oracles[name] = metrics(y[ote], predict(om, xt, eit, eat)[ote])
        print(f"  {name:14s} oracle test F1 {oracles[name]['f1']:.3f}  "
              f"ROC-AUC {oracles[name]['roc_auc']:.3f}  "
              f"(zero-shot was F1 {results[name]['f1']:.3f} / "
              f"AUC {results[name]['roc_auc']:.3f})")

    # ---- save + figure ----
    out = dict(meta=dict(seed=SEED, k=K_GRAPH, epochs=EPOCHS, features="radius-only",
                         pyvoro_validation=val),
               source=dict(base_rate=float(y_src.mean()), test=src_test),
               targets=results, oracles=oracles)
    os.makedirs(config.RESULTS, exist_ok=True)
    with open(os.path.join(config.RESULTS, "04_transfer.json"), "w") as f:
        json.dump(out, f, indent=2)
    make_figure(results, src_test, oracles)
    print(f"\nsaved -> results/04_transfer.json   |   total {time.time()-t_start:.0f}s")


def make_figure(results, src_test, oracles):
    fam_color = {"CuZr": "#2c7fb8", "NiZr": "#7fcdbb", "Cu-Zr-Al": "#d95f0e", "Co-W": "#756bb1"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) zero-shot ROC-AUC per target, grouped by family. Drop degenerate targets
    # (<25 perfect-ICO atoms): ICO essentially vanishes there (e.g. Co-W >=80% W),
    # so a ranking AUC is meaningless.
    names = [n for n in results if results[n]["roc_auc"] is not None
             and results[n]["base_rate"] * results[n]["n"] >= 25]
    names.sort(key=lambda n: (list(fam_color).index(results[n]["family"]), results[n]["comp"]))
    aucs = [results[n]["roc_auc"] for n in names]
    cols = [fam_color[results[n]["family"]] for n in names]
    ax[0, 0].bar(range(len(names)), aucs, color=cols)
    ax[0, 0].axhline(src_test["roc_auc"], ls="--", color="k", lw=1,
                     label=f"source Cu-Zr test ({src_test['roc_auc']:.3f})")
    ax[0, 0].axhline(0.5, ls=":", color="grey", lw=1)
    ax[0, 0].set_xticks(range(len(names)))
    ax[0, 0].set_xticklabels(names, rotation=90, fontsize=7)
    ax[0, 0].set_ylabel("zero-shot ROC-AUC"); ax[0, 0].set_ylim(0.4, 1.0)
    ax[0, 0].set_title("(a) Zero-shot transfer per chemistry")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    ax[0, 0].legend(handles + [plt.Line2D([0], [0], ls="--", color="k")],
                    list(fam_color) + ["source"], fontsize=7, ncol=2)

    # (b) transfer gap: zero-shot vs in-domain oracle (representatives)
    on = list(oracles)
    x = np.arange(len(on)); w = 0.38
    ax[0, 1].bar(x - w/2, [results[n]["roc_auc"] for n in on], w, label="zero-shot", color="#2c7fb8")
    ax[0, 1].bar(x + w/2, [oracles[n]["roc_auc"] for n in on], w, label="in-domain oracle", color="#de2d26")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(on, rotation=20, fontsize=8)
    ax[0, 1].set_ylabel("ROC-AUC"); ax[0, 1].set_ylim(0.4, 1.0)
    ax[0, 1].set_title("(b) Transfer gap (zero-shot vs in-domain)"); ax[0, 1].legend(fontsize=8)

    # (c) Cu-Zr-Al: true vs predicted ICO fraction across Al %
    al = sorted([n for n in results if results[n]["family"] == "Cu-Zr-Al"],
                key=lambda n: results[n]["comp"])
    axc = ax[1, 0]
    axc.plot([results[n]["comp"] for n in al], [results[n]["true_frac"] for n in al],
             "o-", color="k", label="true (pyvoro)")
    axc.plot([results[n]["comp"] for n in al], [results[n]["pred_frac"] for n in al],
             "s--", color="#d95f0e", label="predicted")
    axc.set_xlabel("Al content (%)"); axc.set_ylabel("perfect-ICO fraction")
    axc.set_title("(c) Cu-Zr-Al: ICO fraction vs Al%"); axc.legend(fontsize=8)

    # (d) Co-W: true vs predicted ICO fraction across W %
    cw = sorted([n for n in results if results[n]["family"] == "Co-W"],
                key=lambda n: results[n]["comp"])
    axd = ax[1, 1]
    axd.plot([results[n]["comp"] for n in cw], [results[n]["true_frac"] for n in cw],
             "o-", color="k", label="true (pyvoro)")
    axd.plot([results[n]["comp"] for n in cw], [results[n]["pred_frac"] for n in cw],
             "s--", color="#756bb1", label="predicted")
    axd.set_xlabel("W content (%)"); axd.set_ylabel("perfect-ICO fraction")
    axd.set_title("(d) Co-W: ICO fraction vs W%"); axd.legend(fontsize=8)

    fig.suptitle("Phase 4: cross-chemistry transfer of the Cu-Zr icosahedron detector",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(config.RESULTS, "04_transfer.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
