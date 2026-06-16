"""Phase 2: predict the perfect icosahedron <0,0,12,0> from geometry.

Same data, same task, same splits -> compare the thesis-style flat MLP against a
distance-aware GNN (CGCNN). The GNN gets only a periodic kNN graph (no Voronoi
edges; fixed k so degree can't leak the label).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

import config
from src.features import (load_samples2, knn_periodic, alignment_check,
                          flat_neighbour_features, rbf_expand)
from src.models import MLP, CGCNN

SEED = 0
K_GRAPH = 16
K_FLAT = 20
EPOCHS = 300


def make_splits(y):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y[tmp])
    return tr, va, te


def evaluate(logits, y, idx):
    prob = torch.softmax(logits[idx], 1)[:, 1].detach().numpy()
    pred = (prob >= 0.5).astype(int)
    yt = y[idx].numpy()
    return dict(acc=accuracy_score(yt, pred), f1=f1_score(yt, pred, zero_division=0),
                f1_macro=f1_score(yt, pred, average="macro", zero_division=0),
                roc_auc=roc_auc_score(yt, prob), pr_auc=average_precision_score(yt, prob))


def train(model, x, edge_index, edge_attr, y, tr, va, te, lr=5e-3, tag=""):
    cls_w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max((y[tr] == 1).sum(), 1)])
    crit = nn.CrossEntropyLoss(weight=cls_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_t = torch.tensor(tr); va_t = torch.tensor(va)
    best_f1, best_state, best_te = -1, None, None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(x, edge_index, edge_attr)
        loss = crit(out[tr_t], y[tr_t]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_attr)
            vm = evaluate(out, y, va_t)
            if vm["f1_macro"] > best_f1:
                best_f1 = vm["f1_macro"]
                best_te = evaluate(out, y, torch.tensor(te))
    print(f"  [{tag}] trained {EPOCHS} epochs in {time.time()-t0:.1f}s  (best val macro-F1 {best_f1:.3f})")
    return best_te


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = load_samples2()
    pos, L, y = d["pos"], d["L"], d["y"]
    base = y.mean()
    print(f"samples2: {d['N']} atoms, {100*base:.1f}% perfect icosahedra "
          f"(trivial all-0 accuracy = {100*(1-base):.1f}%)")

    frac, edist = alignment_check(pos, L, d["nbrs"], k=20)
    print(f"alignment check: {100*frac:.1f}% of Voronoi face-neighbours are within the "
          f"20 nearest spatial neighbours; mean bond length {edist:.2f} A")
    if frac < 0.80:
        print("  WARNING: positions may not align with labels — metrics unreliable.")

    edge_index, edge_dist = knn_periodic(pos, L, k=K_GRAPH)
    edge_attr = torch.tensor(rbf_expand(edge_dist, n_rbf=16, cutoff=6.0))
    edge_index = torch.tensor(edge_index)
    y_t = torch.tensor(y)
    tr, va, te = make_splits(y)
    print(f"graph: {edge_index.shape[1]} directed edges  |  split {len(tr)}/{len(va)}/{len(te)}")

    # ---- baseline: thesis-style flat MLP ----
    Xflat = torch.tensor(flat_neighbour_features(pos, L, k=K_FLAT))
    mlp = MLP(in_dim=Xflat.shape[1])
    res_mlp = train(mlp, Xflat, edge_index, edge_attr, y_t, tr, va, te, tag="MLP (thesis-style)")

    # ---- GNN: node features = type one-hot + radius ----
    type_oh = np.zeros((d["N"], 2), dtype=np.float32)
    type_oh[np.arange(d["N"]), d["types"] - 1] = 1.0
    rad = ((d["radius"] - d["radius"].mean()) / (d["radius"].std() + 1e-6)).astype(np.float32)
    Xnode = torch.tensor(np.concatenate([type_oh, rad[:, None]], axis=1))
    gnn = CGCNN(in_dim=Xnode.shape[1], edge_dim=edge_attr.shape[1])
    res_gnn = train(gnn, Xnode, edge_index, edge_attr, y_t, tr, va, te, tag="CGCNN (GNN)")

    print("\n================  TEST-SET RESULTS  ================")
    print(f"{'model':<22}{'acc':>8}{'ICO-F1':>9}{'macro-F1':>10}{'ROC-AUC':>9}{'PR-AUC':>8}")
    for name, r in [("MLP (thesis-style)", res_mlp), ("CGCNN (GNN)", res_gnn)]:
        print(f"{name:<22}{r['acc']:>8.3f}{r['f1']:>9.3f}{r['f1_macro']:>10.3f}"
              f"{r['roc_auc']:>9.3f}{r['pr_auc']:>8.3f}")

    os.makedirs(config.RESULTS, exist_ok=True)
    with open(os.path.join(config.RESULTS, "02_node_classification.json"), "w") as f:
        json.dump({"base_rate": float(base), "MLP": res_mlp, "CGCNN": res_gnn,
                   "alignment_frac": frac, "mean_bond_A": edist}, f, indent=2)
    print(f"\nsaved -> {os.path.join(config.RESULTS, '02_node_classification.json')}")


if __name__ == "__main__":
    main()
