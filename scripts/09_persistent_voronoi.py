"""Phase 6c (exploratory) — Persistence / small-face filtration (proposal D).

Small Voronoi faces flicker in/out under thermal motion -- that is the main source of
Voronoi-index instability. A face-AREA filtration removes exactly those events: drop every
face whose area is below a fraction lambda of the cell's total surface, then recount the
index. Sweeping lambda traces a lambda-parametrised robustness<->informativeness curve.

For each lambda we measure, for the filtered index <n3,n4,n5,n6>, the filtered coordination
(# surviving faces) and filtered icosahedral-like (n5>=10):
  * frame-to-frame flip-rate over the 11 samples1 frames   (lower = more stable)
  * Shannon entropy of the descriptor on frame 0            (informativeness)
  * sigma-sweep self-consistency under thermal jitter       (robustness)

Uses pyvoro (radical Voronoi, same tessellation as the project labels). Face areas via
Newell's method. NOTE: this is the common "drop small faces and recount" cutoff; it does
NOT perform the rigorous topological edge-collapse (a surviving face's edge count is taken
as-is), which is a known approximation of true lambda-Voronoi.

Run:  python3 scripts/09_persistent_voronoi.py [--smoke]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np

import config
from src.features import load_samples1_frames, jitter
from src.metrics import flip_rate

DISPERSION = 4.0


def newell_area(verts):
    """Area of a planar polygon in 3D from ordered vertices (k,3) via Newell's method."""
    nxt = np.roll(verts, -1, axis=0)
    return 0.5 * np.linalg.norm(np.cross(verts, nxt).sum(axis=0))


def per_atom_faces(pos, L, radii):
    """pyvoro radical Voronoi -> per-atom (edge_counts, face_areas) arrays."""
    import pyvoro
    cells = pyvoro.compute_voronoi(
        np.asarray(pos, float).tolist(),
        [[0.0, float(L[0])], [0.0, float(L[1])], [0.0, float(L[2])]],
        DISPERSION, radii=[float(r) for r in radii], periodic=[True, True, True])
    faces = []
    for c in cells:
        V = np.asarray(c["vertices"], float)
        edges = np.empty(len(c["faces"]), int)
        areas = np.empty(len(c["faces"]), float)
        for k, f in enumerate(c["faces"]):
            vi = f["vertices"]
            edges[k] = len(vi)
            areas[k] = newell_area(V[vi])
        faces.append((edges, areas))
    return faces


def descriptors_at_lambda(faces, lam):
    """Filtered <n3,n4,n5,n6>, coordination, ico-like for relative-area threshold lam."""
    N = len(faces)
    idx4 = np.zeros((N, 4), int)
    coord = np.zeros(N, int)
    for i, (edges, areas) in enumerate(faces):
        tot = areas.sum()
        keep = areas >= lam * tot if tot > 0 else np.ones(len(areas), bool)
        e = edges[keep]
        coord[i] = int(keep.sum())
        for j, nc in enumerate((3, 4, 5, 6)):
            idx4[i, j] = int((e == nc).sum())
    ico_like = (idx4[:, 2] >= 10).astype(int)
    return idx4, coord, ico_like


def entropy_bits(keys):
    _, counts = np.unique(np.asarray(keys, dtype=object), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def keyify(idx4):
    return np.array(["%d_%d_%d_%d" % tuple(r) for r in idx4], dtype=object)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = res["lambdas"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(lam, res["flip"]["idx4"], "s-", color="#4488cc", label="filtered index")
    ax[0].plot(lam, res["flip"]["coord"], "o-", color="#7b3294", label="filtered coordination")
    ax[0].plot(lam, res["flip"]["ico_like"], "^-", color="#cc4444", label="filtered ico-like")
    ax[0].set_xlabel("area filtration lambda (fraction of cell surface)")
    ax[0].set_ylabel("frame-to-frame flip-rate"); ax[0].set_title("Robustness vs filtration (lower = more stable)")
    ax[0].legend(); ax[0].grid(alpha=0.25)
    # Pareto: entropy vs flip, lambda-parametrised, for the filtered index
    ax[1].plot(res["entropy"]["idx4"], res["flip"]["idx4"], "s-", color="#4488cc")
    for x, y, l in zip(res["entropy"]["idx4"], res["flip"]["idx4"], lam):
        ax[1].annotate("%.3f" % l, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax[1].set_xlabel("informativeness: entropy (bits)"); ax[1].set_ylabel("flip-rate")
    ax[1].set_title("Filtered index: robustness<->informativeness\n(labels = lambda)")
    ax[1].grid(alpha=0.25)
    sw = res["sigma_sweep"]
    for l_i in res["sweep_lambdas"]:
        ax[2].plot(sw["sigma"], sw["idx4"][str(l_i)], "o-", label="idx4 lam=%.3f" % l_i)
    ax[2].set_xlabel("jitter sigma (A)"); ax[2].set_ylabel("agreement with sigma=0")
    ax[2].set_title("Self-consistency vs jitter (filtered index)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    d = load_samples1_frames()
    frames, L, radii, N = d["frames"], d["L"], d["radius"], d["N"]
    if args.smoke:
        frames = frames[:3]
    lambdas = [0.0, 0.02, 0.05] if args.smoke else [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    sweep_lambdas = [0.0, 0.02] if args.smoke else [0.0, 0.01, 0.03]
    sigmas = [0.0, 0.05, 0.10] if args.smoke else [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    print("samples1: N=%d, %d frames | lambdas=%s" % (N, len(frames), lambdas))

    # per-frame filtered descriptors
    idx4_pf = {l: [] for l in lambdas}
    coord_pf = {l: [] for l in lambdas}
    icol_pf = {l: [] for l in lambdas}
    for fi, fr in enumerate(frames):
        faces = per_atom_faces(fr, L, radii)
        for l in lambdas:
            i4, co, il = descriptors_at_lambda(faces, l)
            idx4_pf[l].append(i4); coord_pf[l].append(co); icol_pf[l].append(il)
        print("  frame %d/%d" % (fi + 1, len(frames)), flush=True)

    flip = {"idx4": [], "coord": [], "ico_like": []}
    entropy = {"idx4": [], "coord": [], "ico_like": []}
    for l in lambdas:
        i4 = np.stack(idx4_pf[l]); co = np.stack(coord_pf[l]); il = np.stack(icol_pf[l])
        flip["idx4"].append(flip_rate(i4))
        flip["coord"].append(flip_rate(co[:, :, None]))
        flip["ico_like"].append(flip_rate(il[:, :, None]))
        entropy["idx4"].append(entropy_bits(keyify(i4[0])))
        entropy["coord"].append(entropy_bits(co[0]))
        entropy["ico_like"].append(entropy_bits(il[0]))

    # sigma-sweep self-consistency for a few lambdas
    rng = np.random.default_rng(123)
    base = frames[0]
    base_faces = per_atom_faces(base, L, radii)
    ref = {l: descriptors_at_lambda(base_faces, l) for l in sweep_lambdas}
    sweep_idx4 = {str(l): [] for l in sweep_lambdas}
    sweep_coord = {str(l): [] for l in sweep_lambdas}
    for s in sigmas:
        faces = per_atom_faces(jitter(base, s, L, rng), L, radii)
        for l in sweep_lambdas:
            i4, co, _ = descriptors_at_lambda(faces, l)
            r4, rco, _ = ref[l]
            sweep_idx4[str(l)].append(float((i4 == r4).all(1).mean()))
            sweep_coord[str(l)].append(float((co == rco).mean()))
        print("  sigma %.3f" % s, flush=True)

    res = dict(n_atoms=int(N), n_frames=int(len(frames)), lambdas=lambdas,
               sweep_lambdas=sweep_lambdas, flip=flip, entropy=entropy,
               sigma_sweep=dict(sigma=sigmas, idx4=sweep_idx4, coord=sweep_coord))
    os.makedirs(config.RESULTS, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    json.dump(res, open(os.path.join(config.RESULTS, "09_persistent_voronoi%s.json" % suffix), "w"), indent=2)
    make_figure(res, os.path.join(config.RESULTS, "09_persistent_voronoi%s.png" % suffix))

    print("\n%-8s %10s %9s %10s %9s %11s" % ("lambda", "idx4_flip", "idx4_ent", "coord_flip", "coord_ent", "icolike_flip"))
    for k, l in enumerate(lambdas):
        print("%-8.3f %10.4f %9.2f %10.4f %9.2f %11.4f"
              % (l, flip["idx4"][k], entropy["idx4"][k], flip["coord"][k],
                 entropy["coord"][k], flip["ico_like"][k]))


if __name__ == "__main__":
    main()
