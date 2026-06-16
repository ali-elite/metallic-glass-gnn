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
                             accuracy_score, precision_recall_curve)

import config
from src.data import is_perfect_icosahedron
from src.features import (load_samples2, load_samples3_target, knn_periodic,
                          rbf_expand)
from src.models import CGCNN
from src.voronoi import voronoi_index, validate_against_samples2

warnings.filterwarnings("ignore")
SEED = 0
N_SEEDS = 5            # CPU message passing is not bit-reproducible; report mean +/- std
K_GRAPH = 16
EPOCHS = 250
LR = 1e-3             # 1e-3 is markedly more stable than 5e-3 here: with the minimal
                      # radius-only features, 5e-3 made transfer AUC swing wildly
                      # (e.g. Cu-Zr-Al 0.31-0.75 across seeds); 1e-3 tightens that to ~0.02.
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
    """Threshold-free ranking (ROC/PR-AUC) plus three F1s that separate ranking
    quality from threshold calibration:
      f1        - zero-shot at the fixed 0.5 threshold (what we actually deploy);
      f1_recal  - base-rate-matched: label the top-(base rate) atoms by score,
                  i.e. recalibrate using ONLY the target's overall ICO fraction
                  (one scalar), not per-atom labels;
      f1_best   - the best F1 over all thresholds (an oracle upper bound).
    """
    pred = (prob >= 0.5).astype(int)
    npos = int(y.sum())
    out = dict(n=int(len(y)), base_rate=float(y.mean()), true_frac=float(y.mean()),
               pred_frac=float(pred.mean()), acc=float(accuracy_score(y, pred)),
               f1=float(f1_score(y, pred, zero_division=0)))
    if 0 < npos < len(y):
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
        pred_recal = np.zeros(len(y), int)
        pred_recal[np.argsort(prob)[::-1][:npos]] = 1          # top-npos by score
        out["f1_recal"] = float(f1_score(y, pred_recal, zero_division=0))
        prec, rec, _ = precision_recall_curve(y, prob)
        out["f1_best"] = float(np.nanmax(2 * prec * rec / (prec + rec + 1e-12)))
    else:
        out["roc_auc"] = out["pr_auc"] = out["f1_recal"] = out["f1_best"] = None
    return out


def predict(model, x, ei, ea):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x, ei, ea), 1)[:, 1].numpy()


def train_cgcnn(x, ei, ea, y, tr, va, tag="", seed=SEED):
    """Class-weighted CGCNN; keep the best-validation-macro-F1 weights."""
    torch.manual_seed(seed)
    model = CGCNN(in_dim=x.shape[1], edge_dim=ea.shape[1])
    y_t = torch.tensor(y)
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max((y[tr] == 1).sum(), 1)],
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
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

    # ---- source inputs (Cu-Zr) ----
    d = load_samples2()
    mu, sd = float(d["radius"].mean()), float(d["radius"].std() + 1e-6)
    y_src = is_perfect_icosahedron(d["vor"]).astype(np.int64)
    x, ei, ea = build_inputs(d["pos"], d["L"], d["radius"], mu, sd)
    tr, tmp = train_test_split(np.arange(d["N"]), test_size=0.30, random_state=SEED, stratify=y_src)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y_src[tmp])

    # ---- label every target once with pyvoro (seed-independent) ----
    targets = build_targets()
    print(f"\nlabelling {len(targets)} targets with pyvoro radical Voronoi (once) ...")
    TD = []
    for t in targets:
        td = load_samples3_target(t["path"], t["tr"])
        y = is_perfect_icosahedron(voronoi_index(td["pos"], td["L"], td["radius"])).astype(np.int64)
        xt, eit, eat = build_inputs(td["pos"], td["L"], td["radius"], mu, sd)
        TD.append(dict(t=t, y=y, x=xt, ei=eit, ea=eat, N=td["N"]))

    # ---- train N_SEEDS source models, transfer zero-shot each time ----
    # (CPU message passing isn't bit-reproducible, so single-seed numbers drift by
    #  ~0.05 AUC run to run; we report mean +/- std over seeds instead.)
    print(f"\ntraining {N_SEEDS} source models (Cu-Zr, radius-only) + zero-shot transfer:")
    per_target = {t["name"]: [] for t in targets}
    src_tests = []
    for s in range(N_SEEDS):
        model = train_cgcnn(x, ei, ea, y_src, tr, va, tag=f"source seed {SEED+s}", seed=SEED + s)
        src_tests.append(metrics(y_src[te], predict(model, x, ei, ea)[te]))
        for td in TD:
            per_target[td["t"]["name"]].append(
                metrics(td["y"], predict(model, td["x"], td["ei"], td["ea"])))

    def agg(dicts, key):
        v = [dd[key] for dd in dicts if dd.get(key) is not None]
        return None if not v else dict(mean=float(np.mean(v)), std=float(np.std(v)))

    AGG = ["roc_auc", "pr_auc", "f1", "f1_recal", "f1_best", "pred_frac"]
    results = {}
    print("\nzero-shot transfer (mean +/- std over seeds):")
    for t in targets:
        ms = per_target[t["name"]]
        r = dict(family=t["family"], comp=t["comp"], n=ms[0]["n"],
                 base_rate=ms[0]["base_rate"], true_frac=ms[0]["true_frac"])
        for k in AGG:
            r[k] = agg(ms, k)
        results[t["name"]] = r
        a, fr = r["roc_auc"], r["f1_recal"]
        auc_s = "n/a" if a is None else f"{a['mean']:.3f}+/-{a['std']:.3f}"
        rec_s = "n/a" if fr is None else f"{fr['mean']:.3f}"
        print(f"  {t['name']:14s} ({t['family']:8s}) base {100*r['base_rate']:5.1f}%  "
              f"ROC-AUC {auc_s}  F1@recal {rec_s}")
    src_agg = {k: agg(src_tests, k) for k in ["roc_auc", "f1", "pr_auc"]}
    print(f"\nsource Cu-Zr test: ROC-AUC {src_agg['roc_auc']['mean']:.3f}"
          f"+/-{src_agg['roc_auc']['std']:.3f}, F1 {src_agg['f1']['mean']:.3f}")

    # ---- in-domain oracles (representative per family, single seed for context) ----
    print("\nin-domain oracles (upper bound, single seed) per family:")
    oracles = {}
    for name in ["CuZr6436", "NiZr6436", "CuZrAl-Al10", "CoW-50"]:
        td = next(z for z in TD if z["t"]["name"] == name)
        y = td["y"]
        if not (0 < y.sum() < len(y)):
            continue
        otr, ote = train_test_split(np.arange(td["N"]), test_size=0.30, random_state=SEED, stratify=y)
        otr, ova = train_test_split(otr, test_size=0.20, random_state=SEED, stratify=y[otr])
        om = train_cgcnn(td["x"], td["ei"], td["ea"], y, otr, ova, tag=f"oracle {name}", seed=SEED)
        oracles[name] = metrics(y[ote], predict(om, td["x"], td["ei"], td["ea"])[ote])
        print(f"  {name:14s} oracle AUC {oracles[name]['roc_auc']:.3f}  "
              f"(zero-shot {results[name]['roc_auc']['mean']:.3f})")

    # ---- save + figure ----
    out = dict(meta=dict(seeds=N_SEEDS, base_seed=SEED, k=K_GRAPH, epochs=EPOCHS,
                         features="radius-only", pyvoro_validation=val),
               source=dict(base_rate=float(y_src.mean()), test=src_agg),
               targets=results, oracles=oracles)
    os.makedirs(config.RESULTS, exist_ok=True)
    with open(os.path.join(config.RESULTS, "04_transfer.json"), "w") as f:
        json.dump(out, f, indent=2)
    make_figure(results, src_agg, oracles)
    print(f"\nsaved -> results/04_transfer.json   |   total {time.time()-t_start:.0f}s")


def make_figure(results, src_agg, oracles):
    fam_color = {"CuZr": "#2c7fb8", "NiZr": "#7fcdbb", "Cu-Zr-Al": "#d95f0e", "Co-W": "#756bb1"}
    M = lambda n, k: results[n][k]["mean"]            # mean of an aggregated metric
    S = lambda n, k: results[n][k]["std"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) zero-shot ROC-AUC (mean +/- std over seeds) per target, grouped by family.
    # Drop degenerate targets (<25 perfect-ICO atoms): ICO vanishes (e.g. Co-W >=80% W).
    names = [n for n in results if results[n]["roc_auc"] is not None
             and results[n]["base_rate"] * results[n]["n"] >= 25]
    names.sort(key=lambda n: (list(fam_color).index(results[n]["family"]), results[n]["comp"]))
    cols = [fam_color[results[n]["family"]] for n in names]
    ax[0, 0].bar(range(len(names)), [M(n, "roc_auc") for n in names],
                 yerr=[S(n, "roc_auc") for n in names], capsize=2, color=cols)
    ax[0, 0].axhline(src_agg["roc_auc"]["mean"], ls="--", color="k", lw=1,
                     label=f"source Cu-Zr test ({src_agg['roc_auc']['mean']:.3f})")
    ax[0, 0].axhline(0.5, ls=":", color="grey", lw=1)
    ax[0, 0].set_xticks(range(len(names)))
    ax[0, 0].set_xticklabels(names, rotation=90, fontsize=7)
    ax[0, 0].set_ylabel("zero-shot ROC-AUC"); ax[0, 0].set_ylim(0.4, 1.0)
    ax[0, 0].set_title(f"(a) Zero-shot transfer per chemistry (mean$\\pm$std, {N_SEEDS} seeds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    ax[0, 0].legend(handles + [plt.Line2D([0], [0], ls="--", color="k")],
                    list(fam_color) + ["source"], fontsize=7, ncol=2)

    # (b) ranking is good but the fixed 0.5 threshold does not transfer; a
    #     base-rate-matched threshold (one scalar per target) recovers most of the F1.
    fams = ["CuZr", "NiZr", "Cu-Zr-Al", "Co-W"]
    def fmean(fam, key):
        v = [results[n][key]["mean"] for n in results
             if results[n]["family"] == fam and results[n].get(key) is not None]
        return float(np.mean(v)) if v else 0.0
    xf = np.arange(len(fams)); w = 0.27
    for j, (key, lab, c) in enumerate([("f1", "F1 @ 0.5 (zero-shot)", "#9ecae1"),
                                       ("f1_recal", "F1 @ base-rate thr.", "#2c7fb8"),
                                       ("f1_best", "F1 @ best thr. (oracle)", "#08519c")]):
        ax[0, 1].bar(xf + (j - 1) * w, [fmean(f, key) for f in fams], w, label=lab, color=c)
    ax[0, 1].set_xticks(xf); ax[0, 1].set_xticklabels(fams, fontsize=8)
    ax[0, 1].set_ylabel("F1 (per-family mean)"); ax[0, 1].set_ylim(0, 1.0)
    ax[0, 1].set_title("(b) Threshold recalibration recovers F1"); ax[0, 1].legend(fontsize=7)

    # (c) Cu-Zr-Al: true vs predicted ICO fraction across Al %
    al = sorted([n for n in results if results[n]["family"] == "Cu-Zr-Al"],
                key=lambda n: results[n]["comp"])
    axc = ax[1, 0]
    axc.plot([results[n]["comp"] for n in al], [results[n]["true_frac"] for n in al],
             "o-", color="k", label="true (pyvoro)")
    axc.plot([results[n]["comp"] for n in al], [M(n, "pred_frac") for n in al],
             "s--", color="#d95f0e", label="predicted")
    axc.set_xlabel("Al content (%)"); axc.set_ylabel("perfect-ICO fraction")
    axc.set_title("(c) Cu-Zr-Al: ICO fraction vs Al%"); axc.legend(fontsize=8)

    # (d) Co-W: true vs predicted ICO fraction across W %
    cw = sorted([n for n in results if results[n]["family"] == "Co-W"],
                key=lambda n: results[n]["comp"])
    axd = ax[1, 1]
    axd.plot([results[n]["comp"] for n in cw], [results[n]["true_frac"] for n in cw],
             "o-", color="k", label="true (pyvoro)")
    axd.plot([results[n]["comp"] for n in cw], [M(n, "pred_frac") for n in cw],
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
