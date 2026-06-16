"""Phase 3: label-free community detection on the atomic graph.

Central question: can an unsupervised, modularity-based GNN recover the
icosahedral backbone / medium-range-order (MRO) sub-domains of a Cu-Zr metallic
glass *without ever seeing a Voronoi label*, and how does it compare to the two
natural baselines - classical Louvain (topology only) and k-means (features only)?

Setup
-----
* Graph        : the Voronoi face-sharing graph over all 10,000 atoms
                 (`src.graph.build_graph` on `nb_id`).
* Node features: chemistry (element one-hot + radius) + rotation/translation-
                 invariant local-geometry scalars (coordination, mean & std bond
                 length, mean neighbour radius). NO Voronoi labels.
* Models       : (1) DMoN modularity GNN (`src.models.DMoN`, graph + features);
                 (2) Louvain (graph only); (3) k-means (features only).
* Ground truth : used ONLY for scoring, never as input -
                 (a) binary "icosahedral backbone vs matrix" (backbone = n5>=10,
                     which also contains every connected perfect-ICO cluster);
                 (b) connected components of the icosahedral subgraph.
* Metrics      : graph modularity Q, NMI/ARI vs each ground truth, and a
                 majority-vote backbone-recovery F1 / AUC.

What we find (and do NOT overclaim)
-----------------------------------
The perfect-ICO network percolates (Phase 1): the backbone is space-filling and
interpenetrating, not a separable blob.  As a result:
  * Louvain maximises modularity (Q~0.7) but finds *spatial* domains that are
    essentially blind to the backbone (NMI~0) - topological modularity is the
    wrong objective for MRO;
  * the backbone is instead a *local-geometry* signal (short, regular bonds), so
    even plain k-means on the invariant features recovers it (AUC~0.82);
  * the DMoN GNN matches that backbone recovery (AUC~0.82, ~50x Louvain's NMI)
    *and* keeps real graph coherence (Q far above k-means) - the only method that
    scores on both axes - in one differentiable, label-free model.
"""
import os, sys, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity as nx_modularity
from sklearn.cluster import KMeans
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                             f1_score, accuracy_score, roc_auc_score, roc_curve)

import config
from src.features import load_samples2, rotation_invariant_features
from src.data import is_icosahedral_like
from src.graph import build_graph, edge_symmetry, physical_communities
from src.models import DMoN, gcn_norm, dmon_loss

warnings.filterwarnings("ignore")  # silence sklearn's "fewer distinct clusters" notes

SEED = 0
N_SEEDS = 3            # DMoN is randomly initialised; report mean +/- std
K_CLUSTERS = 16        # max clusters for DMoN / k-means (collapse reg drops unused)
EPOCHS = 400
LR = 1e-3
HIDDEN = 64
N_LAYERS = 2
COLLAPSE_W = 1.0
ENTROPY_W = 0.10       # mild sharpening of soft assignments
N5_MIN = 10


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def standardize(a):
    return ((a - a.mean(0)) / (a.std(0) + 1e-6)).astype(np.float32)


def labels_from_communities(communities, n):
    """list-of-sets partition -> per-node integer label array."""
    lab = np.full(n, -1, dtype=int)
    for cid, comm in enumerate(communities):
        for i in comm:
            lab[i] = cid
    return lab


def communities_from_labels(labels):
    """per-node labels -> list-of-sets partition (one set per distinct label)."""
    out = {}
    for i, c in enumerate(labels):
        out.setdefault(int(c), set()).add(int(i))
    return list(out.values())


def score_partition(labels, G, backbone, comp_labels):
    """All scores for one hard partition (per-node integer `labels`)."""
    comms = communities_from_labels(labels)
    # majority-vote each cluster -> backbone/matrix, then classify every node;
    # `enrich_score` (= each node's cluster backbone-rate) gives a continuous AUC.
    pred = np.zeros_like(backbone)
    frac_per_cluster = {}
    for c in np.unique(labels):
        members = labels == c
        frac_per_cluster[int(c)] = float(backbone[members].mean())
        if frac_per_cluster[int(c)] >= 0.5:
            pred[members] = 1
    enrich_score = np.array([frac_per_cluster[int(c)] for c in labels])
    return dict(
        n_clusters=int(len(comms)),
        modularity=float(nx_modularity(G, comms)),
        nmi_backbone=float(normalized_mutual_info_score(backbone, labels)),
        ari_backbone=float(adjusted_rand_score(backbone, labels)),
        nmi_components=float(normalized_mutual_info_score(comp_labels, labels)),
        ari_components=float(adjusted_rand_score(comp_labels, labels)),
        backbone_f1=float(f1_score(backbone, pred, zero_division=0)),
        backbone_acc=float(accuracy_score(backbone, pred)),
        backbone_auc=float(roc_auc_score(backbone, enrich_score)),
    )


def train_dmon(X, ei, deg, m, n_nodes, k, seed, log=False):
    """Train one DMoN run; return (best soft assignment C [N,k], modularity history).

    No weight decay: it would shrink the assignment logits toward the uniform
    solution, which is a zero-gradient saddle of the modularity objective.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    ei_sl, norm = gcn_norm(ei, n_nodes)
    model = DMoN(in_dim=X.shape[1], n_clusters=k, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_mod, best_C, hist = -1e9, None, []
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        C = model(X, ei_sl, norm)
        loss, mod, coll, ent = dmon_loss(C, ei, deg, m, COLLAPSE_W, ENTROPY_W)
        loss.backward(); opt.step()
        hist.append(float(mod))
        if float(mod) > best_mod:
            best_mod = float(mod); best_C = C.detach().clone()
        if log and (ep % 100 == 0 or ep == EPOCHS - 1):
            print(f"    ep {ep:3d}  modularity {float(mod):+.4f}  "
                  f"collapse {float(coll):+.3f}  entropy {float(ent):.3f}")
    return best_C, hist


def enrichment_scores(labels, backbone):
    """Per-node continuous score = backbone-rate of the node's cluster (for ROC)."""
    frac = {int(c): float(backbone[labels == c].mean()) for c in np.unique(labels)}
    return np.array([frac[int(c)] for c in labels])


def edge_tensors(G):
    """Symmetric (both-way) edge_index, degree vector, and #edges for a graph."""
    src = [u for u, v in G.edges()] + [v for u, v in G.edges()]
    dst = [v for u, v in G.edges()] + [u for u, v in G.edges()]
    ei = torch.tensor([src, dst], dtype=torch.long)
    deg = torch.tensor([d for _, d in G.degree()], dtype=torch.float32)
    return ei, deg, float(G.number_of_edges())


# --------------------------------------------------------------------------- #
#  main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    d = load_samples2()
    N, vor = d["N"], d["vor"]

    # ---- graph (Voronoi face-sharing, all atoms) ----
    G = build_graph(d["nbrs"])
    degs = np.array([deg for _, deg in G.degree()])
    print(f"Voronoi graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"mean degree {degs.mean():.2f}, reciprocity {100*edge_symmetry(d['nbrs']):.1f}%")

    # ---- physical ground truth (scoring only) ----
    backbone = is_icosahedral_like(vor, N5_MIN).astype(int)   # n5>=10 (= 40.4%)
    comps, _ = physical_communities(G, backbone.astype(bool))
    comp_labels = np.zeros(N, dtype=int)                      # 0 = matrix / isolated
    cid = 1
    for c in comps:
        if len(c) >= 2:
            for i in c:
                comp_labels[i] = cid
            cid += 1
    print(f"ground truth: backbone (n5>=10) = {backbone.sum()} atoms "
          f"({100*backbone.mean():.1f}%); icosahedral subgraph has {cid-1} connected "
          f"components (largest {max(len(c) for c in comps)} atoms)")

    # ---- node features: chemistry + rotation-invariant local geometry (no labels) ----
    type_oh = np.zeros((N, 2), dtype=np.float32)
    type_oh[np.arange(N), d["types"] - 1] = 1.0
    geom = rotation_invariant_features(d["pos"], d["L"], d["nbrs"], d["radius"])
    chem = standardize(d["radius"][:, None])
    Xnp = np.concatenate([type_oh, chem, standardize(geom)], axis=1)  # (N, 7)
    X = torch.tensor(Xnp)
    print(f"node features: {X.shape[1]} dims (element one-hot + radius + "
          f"coordination, mean/std bond, mean nbr radius)")

    ei, deg_t, m = edge_tensors(G)
    results = {}

    # ===================  DMoN (label-free GNN: graph + features)  ===================
    print(f"\n[DMoN] {N_SEEDS} seeds x {EPOCHS} epochs, K={K_CLUSTERS} "
          f"(collapse_w={COLLAPSE_W}, entropy_w={ENTROPY_W})")
    dmon_runs, best_overall = [], None
    for s in range(N_SEEDS):
        t0 = time.time()
        C, hist = train_dmon(X, ei, deg_t, m, N, K_CLUSTERS, SEED + s, log=(s == 0))
        labels = C.argmax(1).numpy()
        sc = score_partition(labels, G, backbone, comp_labels)
        frac = np.array([backbone[labels == c].mean() if (labels == c).any() else 0.0
                         for c in range(K_CLUSTERS)])
        sc["backbone_auc_soft"] = float(roc_auc_score(backbone, C.numpy() @ frac))
        dmon_runs.append(sc)
        print(f"  seed {SEED+s}: {time.time()-t0:.1f}s  clusters={sc['n_clusters']}  "
              f"Q={sc['modularity']:.3f}  NMI={sc['nmi_backbone']:.3f}  "
              f"backbone-F1={sc['backbone_f1']:.3f}  AUC={sc['backbone_auc']:.3f}")
        if best_overall is None or sc["modularity"] > best_overall[0]:
            best_overall = (sc["modularity"], labels, C.numpy(), frac, hist)

    def agg(key):
        v = np.array([r[key] for r in dmon_runs], dtype=float)
        return dict(mean=float(v.mean()), std=float(v.std()))
    results["DMoN"] = {k: agg(k) for k in dmon_runs[0]}
    _, dmon_labels, dmon_C, dmon_frac, dmon_hist = best_overall

    # ===================  baselines  ===================
    print("\n[Louvain] (topology only)")
    t0 = time.time()
    lv = louvain_communities(G, seed=SEED)
    lv_labels = labels_from_communities(lv, N)
    results["Louvain"] = score_partition(lv_labels, G, backbone, comp_labels)
    print(f"  {time.time()-t0:.1f}s  clusters={results['Louvain']['n_clusters']}  "
          f"Q={results['Louvain']['modularity']:.3f}  "
          f"NMI={results['Louvain']['nmi_backbone']:.3f}  "
          f"backbone-F1={results['Louvain']['backbone_f1']:.3f}  "
          f"AUC={results['Louvain']['backbone_auc']:.3f}")

    print("[k-means] (features only)")
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit(Xnp)
    results["kmeans"] = score_partition(km.labels_, G, backbone, comp_labels)
    # control: chemistry alone carries no backbone signal
    km_chem = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit(Xnp[:, :3])
    results["kmeans_chem_only"] = score_partition(km_chem.labels_, G, backbone, comp_labels)
    print(f"  features: clusters={results['kmeans']['n_clusters']}  "
          f"Q={results['kmeans']['modularity']:.3f}  "
          f"NMI={results['kmeans']['nmi_backbone']:.3f}  "
          f"backbone-F1={results['kmeans']['backbone_f1']:.3f}  "
          f"AUC={results['kmeans']['backbone_auc']:.3f}")
    print(f"  chemistry-only control: NMI={results['kmeans_chem_only']['nmi_backbone']:.3f}  "
          f"AUC={results['kmeans_chem_only']['backbone_auc']:.3f}  (~ no signal)")

    # reference: modularity of the ground-truth backbone-vs-matrix bipartition
    Q_backbone = float(nx_modularity(G, communities_from_labels(backbone)))
    results["reference"] = dict(backbone_partition_modularity=Q_backbone,
                                backbone_rate=float(backbone.mean()))

    # ===================  Experiment B: MRO sub-domains within the backbone  ===================
    print("\n[sub-domains] partitioning the icosahedral backbone subgraph")
    bb_nodes = [i for i in range(N) if backbone[i]]
    H = nx.convert_node_labels_to_integers(G.subgraph(bb_nodes).copy())
    nH = H.number_of_nodes()
    lvH = louvain_communities(H, seed=SEED)
    QH_lv = float(nx_modularity(H, lvH))
    eiH, degH, mH = edge_tensors(H)
    XH = torch.tensor(Xnp[bb_nodes])               # same invariant features, restricted
    CH, _ = train_dmon(XH, eiH, degH, mH, nH, 8, SEED)
    QH_dmon = float(nx_modularity(H, communities_from_labels(CH.argmax(1).numpy())))
    results["subdomains"] = dict(
        backbone_nodes=nH, backbone_edges=H.number_of_edges(),
        louvain_subdomains=len(lvH), louvain_modularity=QH_lv,
        dmon_subdomains=int(len(np.unique(CH.argmax(1).numpy()))), dmon_modularity=QH_dmon)
    print(f"  backbone subgraph: {nH} nodes, {H.number_of_edges()} edges")
    print(f"  Louvain: {len(lvH)} sub-domains, Q={QH_lv:.3f}   |   "
          f"DMoN: {results['subdomains']['dmon_subdomains']} sub-domains, Q={QH_dmon:.3f}")

    # ---- summary table ----
    print("\n================  COMMUNITY DETECTION vs PHYSICAL GROUND TRUTH  ================")
    print(f"{'method':<14}{'uses':<18}{'#cl':>4}{'Q':>7}{'NMI_bb':>8}{'ARI_bb':>8}"
          f"{'bb-F1':>7}{'bb-AUC':>8}")
    rows = [("DMoN", "graph+features", results["DMoN"], True),
            ("Louvain", "topology only", results["Louvain"], False),
            ("kmeans", "features only", results["kmeans"], False)]
    for name, uses, r, is_dmon in rows:
        g = (lambda k: r[k]["mean"]) if is_dmon else (lambda k: r[k])
        print(f"{name:<14}{uses:<18}{g('n_clusters'):>4.0f}{g('modularity'):>7.3f}"
              f"{g('nmi_backbone'):>8.3f}{g('ari_backbone'):>8.3f}"
              f"{g('backbone_f1'):>7.3f}{g('backbone_auc'):>8.3f}")
    print(f"(reference: backbone-vs-matrix bipartition has graph modularity Q={Q_backbone:.3f})")
    print("\nReading: Louvain maximises Q but is blind to the backbone (NMI~0); the "
          "backbone\nis a local-geometry signal (k-means AUC high, Q~0); DMoN alone "
          "scores on both axes.")

    # ---- figure + save ----
    roc_data = {
        "DMoN": roc_curve(backbone, dmon_C @ dmon_frac),
        "Louvain": roc_curve(backbone, enrichment_scores(lv_labels, backbone)),
        "kmeans": roc_curve(backbone, enrichment_scores(km.labels_, backbone)),
    }
    make_figure(results, dmon_labels, dmon_frac, backbone, roc_data)

    os.makedirs(config.RESULTS, exist_ok=True)
    meta = dict(seeds=N_SEEDS, epochs=EPOCHS, K=K_CLUSTERS, n5_min=N5_MIN,
                collapse_w=COLLAPSE_W, entropy_w=ENTROPY_W, feature_dim=int(X.shape[1]),
                n_atoms=N, n_edges=G.number_of_edges())
    with open(os.path.join(config.RESULTS, "03_community_detection.json"), "w") as f:
        json.dump({"meta": meta, **results}, f, indent=2)
    print(f"\nsaved -> {os.path.join(config.RESULTS, '03_community_detection.json')}")
    print(f"total runtime {time.time()-t_start:.0f}s")


def make_figure(results, dmon_labels, dmon_frac, backbone, roc_data):
    methods = ["DMoN", "Louvain", "kmeans"]
    colors = {"DMoN": "#2c7fb8", "Louvain": "#d95f0e", "kmeans": "#31a354"}
    val = lambda n, k: results[n][k]["mean"] if n == "DMoN" else results[n][k]
    err = lambda n, k: results["DMoN"][k]["std"] if n == "DMoN" else 0.0
    auc = lambda n: results[n]["backbone_auc"]["mean"] if n == "DMoN" else results[n]["backbone_auc"]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) NMI / ARI vs binary backbone
    w = 0.38
    ax[0, 0].bar(x - w/2, [val(m, "nmi_backbone") for m in methods], w, label="NMI",
                 yerr=[err(m, "nmi_backbone") for m in methods], capsize=3, color="#2c7fb8")
    ax[0, 0].bar(x + w/2, [val(m, "ari_backbone") for m in methods], w, label="ARI",
                 yerr=[err(m, "ari_backbone") for m in methods], capsize=3, color="#de2d26")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(methods)
    ax[0, 0].set_title("(a) Agreement with icosahedral backbone (n5$\\geq$10)")
    ax[0, 0].set_ylabel("score"); ax[0, 0].legend()

    # (b) the two axes that matter: graph coherence (Q) vs backbone recovery (AUC)
    for m in methods:
        ax[0, 1].errorbar(val(m, "modularity"), auc(m),
                          xerr=err(m, "modularity"), yerr=err(m, "backbone_auc"),
                          fmt="o", ms=11, color=colors[m], capsize=3)
        ax[0, 1].annotate(m, (val(m, "modularity"), auc(m)),
                          textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax[0, 1].axhline(0.5, ls=":", color="grey", lw=1)
    ax[0, 1].set_xlabel("graph modularity $Q$  (community coherence)")
    ax[0, 1].set_ylabel("backbone AUC  (physical recovery)")
    ax[0, 1].set_title("(b) Backbone recovery vs graph coherence (DMoN balances both)")
    ax[0, 1].set_ylim(0.45, 0.9)

    # (c) DMoN per-cluster backbone enrichment (best run)
    order = [c for c in np.argsort(dmon_frac)[::-1] if (dmon_labels == c).any()]
    sizes = [int((dmon_labels == c).sum()) for c in order]
    ax[1, 0].bar(range(len(order)), [dmon_frac[c] for c in order], color="#2c7fb8")
    ax[1, 0].axhline(backbone.mean(), ls="--", color="k", lw=1,
                     label=f"global rate {backbone.mean():.2f}")
    ax[1, 0].set_xlabel("DMoN cluster (sorted)"); ax[1, 0].set_ylabel("backbone fraction")
    ax[1, 0].set_title("(c) Per-cluster backbone enrichment (DMoN)"); ax[1, 0].legend()
    for i, sz in enumerate(sizes):
        ax[1, 0].text(i, dmon_frac[order[i]] + 0.01, str(sz), ha="center", va="bottom", fontsize=6)

    # (d) label-free backbone-stratification ROC for all three methods
    for m in methods:
        fpr, tpr, _ = roc_data[m]
        ax[1, 1].plot(fpr, tpr, color=colors[m], label=f"{m} (AUC {auc(m):.3f})")
    ax[1, 1].plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax[1, 1].set_xlabel("false positive rate"); ax[1, 1].set_ylabel("true positive rate")
    ax[1, 1].set_title("(d) Label-free backbone stratification (cluster enrichment)")
    ax[1, 1].legend(loc="lower right")

    fig.suptitle("Phase 3: unsupervised community detection on the Cu-Zr atomic graph",
                 y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(config.RESULTS, "03_community_detection.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
