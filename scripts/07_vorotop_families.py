"""Phase 6b (exploratory) — Perturbation-robust topological FAMILIES: is there a Voronoi
descriptor in the gap between the fragile face-count index and the (robust but binary)
icosahedron class?

Idea. VoroTop robustness comes from CLASSIFICATION, not the raw Weinberg vector (Phase 6
showed raw topology is the *least* stable descriptor). So we group Weinberg topologies into
"families" that are reachable from one another under small thermal jitter, via union-find
over per-atom type co-occurrence in a calibration ensemble (jittered copies of frame 0).
Each atom is then labelled by its family. We map the robustness<->informativeness Pareto
front of descriptors at every granularity:

    weinberg  (full topology)      pfull (face counts)      idx4 <n3,n4,n5,n6>
    coord (total #faces)           n5 (#pentagons)          family (NEW)        ico

Informativeness = Shannon entropy (bits) of the per-atom class distribution (frame 0).
Robustness      = frame-to-frame flip-rate + sigma-sweep agreement (lower flip / higher
                  agreement = more stable). The good corner is high entropy AND low flip.

No leakage: families are calibrated on jittered copies of frame 0, then scored on the 11
real frames and a clean sigma-sweep. Reuses the Phase-6 VoroTop helpers.

Run:  python3 scripts/07_vorotop_families.py [--smoke]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import shutil
import argparse
import tempfile
import importlib.util
import numpy as np
from collections import Counter

import config
from src.features import load_samples1_frames, jitter

# reuse Phase-6 VoroTop helpers (module name starts with a digit -> load via importlib)
_spec = importlib.util.spec_from_file_location(
    "p6", os.path.join(os.path.dirname(__file__), "06_vorotop_topology.py"))
p6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p6)
run_vorotop, flip_rate_keys = p6.run_vorotop, p6.flip_rate_keys

SIGMA_CAL = 0.05    # calibration jitter amplitude (A) defining perturbation-equivalence
N_REPLICA = 8
DESCS = ["weinberg", "pfull", "idx4", "coord_n5", "coord_n5like", "n5", "coord",
         "famcanon", "family", "ico"]


class UF:
    """Union-find over hashable items (Weinberg-vector strings)."""
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:           # path compression
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def entropy_bits(keys):
    _, counts = np.unique(np.asarray(keys, dtype=object), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def derive(out):
    """Extra coarsenings from a run_vorotop dict: n5 (#pentagons) and coordination (#faces)."""
    n5 = np.array([v[2] for v in out["idx4"]], dtype=int)
    coord = np.array([sum(int(x) for x in s.strip("()").split(",") if x != "")
                      for s in out["pfull"]], dtype=int)
    return n5, coord


def descriptor_views(out, fam_of, canon_of):
    """All per-atom descriptor arrays from one VoroTop run."""
    n5, coord = derive(out)
    n5like = n5 >= 10
    return {
        "weinberg": out["weinberg"], "pfull": out["pfull"], "idx4": out["idx4"],
        "coord": coord, "n5": n5, "ico": out["ico"],
        # joint frontier descriptors: combine the two robust scalars for more information
        # (string keys to keep these as 1-D object arrays)
        "coord_n5": np.array(["%d_%d" % (int(coord[i]), int(n5[i])) for i in range(len(n5))], dtype=object),
        "coord_n5like": np.array(["%d_%d" % (int(coord[i]), int(n5like[i])) for i in range(len(n5))], dtype=object),
        "famcanon": np.array([canon_of(w) for w in out["weinberg"]], dtype=object),
        "family": np.array([fam_of(w) for w in out["weinberg"]], dtype=object),
    }


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    descs = res["descriptors"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    xs = [res["entropy_bits"][k] for k in descs]
    ys = [1.0 - res["flip_rate"][k] for k in descs]          # frame-to-frame stability
    ax[0].scatter(xs, ys, s=90, color="#2c7fb8", zorder=3)
    for k, x, y in zip(descs, xs, ys):
        ax[0].annotate(k, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax[0].set_xlabel("informativeness: entropy (bits)")
    ax[0].set_ylabel("frame-to-frame stability (1 - flip-rate)")
    ax[0].set_title("Robustness <-> informativeness Pareto front\n(top-right = robust AND informative)")
    ax[0].grid(alpha=0.25)
    sw = res["sigma_sweep"]
    for k in descs:
        ax[1].plot(sw["sigma"], sw[k], "o-", label=k)
    ax[1].set_xlabel("jitter sigma (A)"); ax[1].set_ylabel("agreement with sigma=0")
    ax[1].set_title("Self-consistency under thermal jitter"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(p6.VOROTOP_BIN):
        sys.exit("VoroTop binary not found at %s (set VOROTOP_BIN)" % p6.VOROTOP_BIN)

    d = load_samples1_frames()
    frames, L, types, N = d["frames"], d["L"], d["types"], d["N"]
    if args.smoke:
        frames = frames[:3]
    sigmas = [0.0, 0.05, 0.10, 0.15] if args.smoke else [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    n_rep = 3 if args.smoke else N_REPLICA
    print("VoroTop:", p6.VOROTOP_BIN)
    print("samples1: N=%d, %d frames | cal sigma=%.2f A, %d replicas" % (N, len(frames), SIGMA_CAL, n_rep))

    work = tempfile.mkdtemp(prefix="vtfam_")
    try:
        # ---- calibrate families: union Weinberg types co-occurring per atom under small jitter ----
        rng = np.random.default_rng(7)
        base = frames[0]
        cal = [run_vorotop(base, L, types, N, work, "cal_base")["weinberg"]]
        for r in range(n_rep):
            cal.append(run_vorotop(jitter(base, SIGMA_CAL, L, rng), L, types, N, work, "cal%d" % r)["weinberg"])
            print("  calibration replica %d/%d" % (r + 1, n_rep), flush=True)
        uf = UF()
        cooc = {}                       # type -> Counter of co-occurring types (1-step denoise)
        for i in range(N):
            ts = [cal[r][i] for r in range(len(cal))]
            for t in ts[1:]:
                uf.union(ts[0], t)
            for t in ts:
                cooc.setdefault(t, Counter()).update(ts)
        fam_of = uf.find
        canon_map = {t: c.most_common(1)[0][0] for t, c in cooc.items()}   # type -> dominant co-type
        canon_of = lambda w: canon_map.get(w, w)

        # ---- eval on the real frames ----
        per_frame = {k: np.empty((len(frames), N), dtype=object) for k in DESCS}
        for fi, fr in enumerate(frames):
            views = descriptor_views(run_vorotop(fr, L, types, N, work, "fr%d" % fi), fam_of, canon_of)
            for k in DESCS:
                per_frame[k][fi] = views[k]
            print("  eval frame %d/%d" % (fi + 1, len(frames)), flush=True)

        flip = {k: flip_rate_keys(per_frame[k]) for k in DESCS}
        f0 = {k: per_frame[k][0] for k in DESCS}
        nclass = {k: int(len(np.unique(np.asarray(f0[k], dtype=object)))) for k in DESCS}
        ent = {k: entropy_bits(f0[k]) for k in DESCS}
        _, fam_counts = np.unique(np.asarray(f0["family"], dtype=object), return_counts=True)
        largest_fam_frac = float(fam_counts.max() / N)
        singleton_frac = float((fam_counts == 1).sum() / len(fam_counts))

        # ---- clean sigma-sweep ----
        rng2 = np.random.default_rng(123)
        ref = descriptor_views(run_vorotop(base, L, types, N, work, "ref"), fam_of, canon_of)
        sweep = {k: [] for k in DESCS}
        for s in sigmas:
            cur = descriptor_views(run_vorotop(jitter(base, s, L, rng2), L, types, N, work, "s%.3f" % s), fam_of, canon_of)
            for k in DESCS:
                sweep[k].append(float(np.mean(cur[k] == ref[k])))
            print("  sigma %.3f" % s, flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    res = dict(descriptors=DESCS, n_atoms=int(N), n_frames=int(len(frames)),
               sigma_cal=SIGMA_CAL, n_replica=n_rep,
               largest_family_frac=largest_fam_frac, singleton_family_frac=singleton_frac,
               n_classes=nclass, entropy_bits=ent, flip_rate=flip,
               sigma_sweep=dict(sigma=sigmas, **sweep))
    os.makedirs(config.RESULTS, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    json.dump(res, open(os.path.join(config.RESULTS, "07_vorotop_families%s.json" % suffix), "w"), indent=2)
    make_figure(res, os.path.join(config.RESULTS, "07_vorotop_families%s.png" % suffix))

    print("\nfamilies: %d distinct (largest covers %.1f%% of atoms; %.0f%% singletons)"
          % (nclass["family"], 100 * largest_fam_frac, 100 * singleton_frac))
    j = sigmas.index(0.12) if 0.12 in sigmas else -1
    print("\n%-9s %7s %8s %10s %11s" % ("descriptor", "#class", "entropy", "flip-rate", "agree@0.12"))
    for k in DESCS:
        a = sweep[k][j] if j >= 0 else sweep[k][-1]
        print("%-9s %7d %8.2f %10.4f %11.3f" % (k, nclass[k], ent[k], flip[k], a))


if __name__ == "__main__":
    main()
