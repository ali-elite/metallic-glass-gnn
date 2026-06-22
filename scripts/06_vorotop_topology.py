"""Phase 6 (exploratory) — Is Voronoi *topology* a more thermally-stable descriptor
than the face-count index? A VoroTop study on samples1 (Cu64Zr36, 11 frames).

Motivation. Phase 5 found that the learned full face-count index ⟨n3,n4,n5,n6⟩ is
thermally fragile and that *coarsening* to the icosahedron class buys stability. VoroTop
(Lazar et al.) argues structure should be described by the *topology* of the Voronoi cell
(the Weinberg vector) rather than the face-count "p-vector". This script tests the
stability of descriptors at every granularity, from the SAME standard Voronoi tessellation
computed by VoroTop, so the comparison is apples-to-apples:

    Weinberg vector  (full topology, finest)
    full p-vector    (face counts n3,n4,n5,...)
    idx4             (the project's ⟨n3,n4,n5,n6⟩)
    icosahedron      (idx4 == (0,0,12,0); the project's coarse label)
    ico_strict       (p-vector == (0,0,12) exactly; a true regular-ico cell)

Two stability metrics, matching Phase 5 (src/metrics.py):
  * frame-to-frame flip-rate over the 11 real frames (lower = more stable)
  * sigma-sweep self-consistency: agreement with the sigma=0 descriptor under Gaussian
    jitter at absolute physical amplitudes (higher = more stable)

NOTE: VoroTop computes STANDARD (not radical) Voronoi, so absolute rates differ slightly
from the project's radical pyvoro numbers; the within-tessellation *ordering across
granularities* is the scientific point and is unaffected.

Requires the VoroTop binary; set VOROTOP_BIN or use the default build path.
Run:  python3 scripts/06_vorotop_topology.py            # full (11 frames + sweep)
      python3 scripts/06_vorotop_topology.py --smoke     # 3 frames, short sweep
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import shutil
import argparse
import tempfile
import subprocess
import numpy as np

import config
from src.features import load_samples1_frames, jitter

VOROTOP_BIN = os.environ.get(
    "VOROTOP_BIN", os.path.expanduser("~/.local/src/VoroTop/VoroTop"))

# descriptor granularity, finest -> coarsest (for display + figure ordering)
DESCRIPTORS = ["weinberg", "pfull", "idx4", "ico_strict", "ico"]
LABELS = {
    "weinberg": "Weinberg\n(full topology)",
    "pfull": "p-vector\n(n3,n4,n5,...)",
    "idx4": "<n3,n4,n5,n6>\n(project index)",
    "ico_strict": "regular ico\n(p==0,0,12)",
    "ico": "icosahedron\n(0,0,12,0)",
}


def write_dump(path, pos, L, types):
    """Write a single-frame LAMMPS dump VoroTop can read (id type x y z, pp box)."""
    n = len(pos)
    ids = np.arange(1, n + 1)
    body = np.column_stack([ids, types, pos[:, 0], pos[:, 1], pos[:, 2]])
    header = ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n"
              "ITEM: BOX BOUNDS pp pp pp\n0.0 %.10f\n0.0 %.10f\n0.0 %.10f\n"
              "ITEM: ATOMS id type x y z" % (n, L[0], L[1], L[2]))
    np.savetxt(path, body, fmt=["%d", "%d", "%.10f", "%.10f", "%.10f"],
               header=header, comments="")


def run_vorotop(pos, L, types, N, work, tag):
    """Run VoroTop -vt on one config; return per-atom descriptors keyed by atom index.

    .vectors line = id <tab> #faces <tab> p-vector <tab> Weinberg <tab> auto <tab> chirality
    """
    dump = os.path.join(work, "%s.dump" % tag)
    write_dump(dump, pos, L, types)
    r = subprocess.run([VOROTOP_BIN, dump, "-vt"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("VoroTop failed (%s): %s" % (tag, r.stderr[:500]))
    weinberg = np.empty(N, dtype=object)
    pfull = np.empty(N, dtype=object)
    idx4 = np.empty(N, dtype=object)
    ico_strict = np.zeros(N, dtype=bool)
    seen = np.zeros(N, dtype=bool)
    with open(dump + ".vectors") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            aid = int(parts[0]) - 1
            nums = [int(x) for x in parts[2].strip("()").split(",") if x != ""]
            v4 = [0, 0, 0, 0]
            for j in range(min(4, len(nums))):
                v4[j] = nums[j]
            weinberg[aid] = parts[3]
            pfull[aid] = parts[2]
            idx4[aid] = tuple(v4)
            ico_strict[aid] = (tuple(nums) == (0, 0, 12))
            seen[aid] = True
    if not seen.all():
        raise RuntimeError("%s: %d atoms missing from .vectors" % (tag, int((~seen).sum())))
    ico = np.array([v == (0, 0, 12, 0) for v in idx4], dtype=bool)
    os.remove(dump); os.remove(dump + ".vectors")
    return dict(weinberg=weinberg, pfull=pfull, idx4=idx4,
                ico_strict=ico_strict, ico=ico)


def flip_rate_keys(per_frame):
    """Fraction of atoms whose descriptor is NOT identical across all frames.

    per_frame: (F, N) object/bool array of hashable per-atom keys. Matches
    src.metrics.flip_rate semantics (constant-across-frames => not flipped)."""
    a = per_frame
    F, N = len(a), len(a[0])
    const = np.ones(N, dtype=bool)
    for fi in range(1, F):
        const &= (a[fi] == a[0])
    return float(1.0 - const.mean())


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    short = {"weinberg": "Weinberg\n(topology)", "pfull": "p-vector\n(n3,n4,n5,..)",
             "idx4": "<n3,n4,n5,n6>", "ico_strict": "regular\nico", "ico": "icosahedron\n(0,0,12,0)"}
    names = res["descriptors"]
    flips = [res["flip_rate"][n] for n in names]
    colors = ["#7b3294", "#c2a5cf", "#4488cc", "#2c7fb8", "#cc4444"]
    ax[0].bar(range(len(names)), flips, color=colors[:len(names)])
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([short[n] for n in names], fontsize=8.5)
    ax[0].set_ylabel("frame-to-frame flip-rate")
    ax[0].set_title("Stability vs granularity (lower = more stable)\nfiner descriptor -> flips more")
    for i, v in enumerate(flips):
        ax[0].text(i, v, " %.3f" % v, ha="center", va="bottom", fontsize=8)
    sw = res["sigma_sweep"]
    for n, c in zip(names, colors):
        ax[1].plot(sw["sigma"], sw[n], "o-", color=c, label=LABELS[n].replace("\n", " "))
    ax[1].set_xlabel("jitter sigma (A)"); ax[1].set_ylabel("agreement with sigma=0")
    ax[1].set_title("Self-consistency under thermal jitter"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(VOROTOP_BIN):
        sys.exit("VoroTop binary not found at %s (set VOROTOP_BIN)" % VOROTOP_BIN)

    d = load_samples1_frames()
    frames, L, types, N = d["frames"], d["L"], d["types"], d["N"]
    if args.smoke:
        frames = frames[:3]
    sigmas = [0.0, 0.05, 0.10, 0.15] if args.smoke else [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    print("VoroTop:", VOROTOP_BIN)
    print("samples1: N=%d, %d frames, L=%.3f A" % (N, len(frames), L[0]))

    work = tempfile.mkdtemp(prefix="vorotop_")
    try:
        # ---- frame-to-frame stability over the real trajectory ----
        per_frame = {n: np.empty((len(frames), N), dtype=object) for n in DESCRIPTORS}
        for fi, fr in enumerate(frames):
            out = run_vorotop(fr, L, types, N, work, "frame%d" % fi)
            for n in DESCRIPTORS:
                per_frame[n][fi] = out[n]
            print("  VoroTop frame %d/%d done" % (fi + 1, len(frames)), flush=True)
        flip_rate = {n: flip_rate_keys(per_frame[n]) for n in DESCRIPTORS}

        # ---- granularity context (frame 0) ----
        f0 = {n: per_frame[n][0] for n in DESCRIPTORS}
        context = dict(
            n_unique_weinberg=int(len(set(f0["weinberg"]))),
            n_unique_pfull=int(len(set(f0["pfull"]))),
            n_unique_idx4=int(len(set(f0["idx4"]))),
            ico_frac=float(f0["ico"].mean()),
            ico_strict_frac=float(f0["ico_strict"].mean()),
        )

        # ---- sigma-sweep self-consistency (jitter the base frame) ----
        rng = np.random.default_rng(123)
        base = frames[0]
        ref = run_vorotop(base, L, types, N, work, "ref")
        sweep = {n: [] for n in DESCRIPTORS}
        for s in sigmas:
            out = run_vorotop(jitter(base, s, L, rng), L, types, N, work, "s%.3f" % s)
            for n in DESCRIPTORS:
                sweep[n].append(float(np.mean(out[n] == ref[n])))
            print("  sigma %.3f done" % s, flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    res = dict(descriptors=DESCRIPTORS, n_atoms=int(N), n_frames=int(len(frames)),
               flip_rate=flip_rate, context=context,
               sigma_sweep=dict(sigma=sigmas, **sweep), voronoi="standard (VoroTop)")
    os.makedirs(config.RESULTS, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    with open(os.path.join(config.RESULTS, "06_vorotop_topology%s.json" % suffix), "w") as f:
        json.dump(res, f, indent=2)
    make_figure(res, os.path.join(config.RESULTS, "06_vorotop_topology%s.png" % suffix))

    print("\n=== granularity context (frame 0) ===")
    print("  unique Weinberg types : %d" % context["n_unique_weinberg"])
    print("  unique full p-vectors : %d" % context["n_unique_pfull"])
    print("  unique <n3..n6>       : %d" % context["n_unique_idx4"])
    print("  icosahedron fraction  : %.3f  (regular-ico: %.3f)"
          % (context["ico_frac"], context["ico_strict_frac"]))
    print("\n=== frame-to-frame flip-rate (lower = more stable) ===")
    for n in DESCRIPTORS:
        print("  %-26s %.4f" % (LABELS[n].replace("\n", " "), flip_rate[n]))
    print("\n=== sigma-sweep agreement with sigma=0 (higher = more stable) ===")
    print("  sigma (A) : " + "  ".join("%5.2f" % s for s in sigmas))
    for n in DESCRIPTORS:
        print("  %-12s: " % n + "  ".join("%5.3f" % a for a in sweep[n]))


if __name__ == "__main__":
    main()
