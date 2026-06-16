"""Loaders for the raw thesis data.

Three relevant per-atom artefacts (consistent atom indexing, 0-based):
  - LAMMPS dump  : atomic positions / types of a snapshot
  - fo_list      : Voronoi index per atom  ->  id total n3 n4 n5 n6 n7 n8 vol
  - nb_id        : Voronoi face-sharing neighbours -> id count nbr0 nbr1 ...

The Voronoi index <n3,n4,n5,n6,...> counts faces with 3,4,5,6,... edges.
A perfect icosahedron is <0,0,12,0>.
"""
import numpy as np


def read_lammps_dump(path, sort_by_id=False):
    """Parse a LAMMPS custom dump.

    Returns positions in *file order* by default. The Voronoi artefacts (fo_list,
    nb_id) in this dataset are indexed by file order, so DO NOT sort by id when
    pairing positions with those labels.
    """
    with open(path) as f:
        lines = f.readlines()
    box, cols, natoms, rows = [], None, None, []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i + 1]); i += 2; continue
        if ln.startswith("ITEM: BOX BOUNDS"):
            for k in range(3):
                lo, hi = lines[i + 1 + k].split()[:2]
                box.append((float(lo), float(hi)))
            i += 4; continue
        if ln.startswith("ITEM: ATOMS"):
            cols = ln.split()[2:]
            rows = [lines[i + 1 + k].split() for k in range(natoms)]
            i += 1 + natoms; continue
        i += 1
    c = {name: j for j, name in enumerate(cols)}
    box = np.array(box)
    ids = np.array([int(r[c["id"]]) for r in rows])
    typ = np.array([int(r[c["type"]]) for r in rows])
    if "x" in c:                                  # absolute coordinates
        pos = np.array([[float(r[c["x"]]), float(r[c["y"]]), float(r[c["z"]])] for r in rows])
    elif "xs" in c:                               # scaled (fractional) coordinates
        s = np.array([[float(r[c["xs"]]), float(r[c["ys"]]), float(r[c["zs"]])] for r in rows])
        pos = box[:, 0] + s * (box[:, 1] - box[:, 0])
    else:
        raise KeyError(f"no x/xs columns in dump; got {cols}")
    order = np.argsort(ids) if sort_by_id else np.arange(len(ids))
    return dict(natoms=natoms, box=box, ids=ids[order], types=typ[order], pos=pos[order])


def read_lammps_frames(path, sort_by_id=False):
    """Parse a possibly multi-frame LAMMPS custom dump into a list of frame dicts.

    Each frame dict matches `read_lammps_dump`'s output (natoms, box, ids, types,
    pos). Coordinates are returned in *file order* by default (see the gotcha in
    `read_lammps_dump`). Used for the samples3 trajectories, some of which contain
    several snapshots; Phase 4 uses the last (most relaxed) frame.
    """
    with open(path) as f:
        lines = f.readlines()
    frames, i, n = [], 0, len(lines)
    while i < n:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1; continue
        natoms = int(lines[i + 3])               # TIMESTEP, value, NUMBER OF ATOMS, value
        box = [tuple(map(float, lines[i + 5 + k].split()[:2])) for k in range(3)]
        cols = lines[i + 8].split()[2:]          # ITEM: ATOMS ...
        rows = [lines[i + 9 + k].split() for k in range(natoms)]
        c = {name: j for j, name in enumerate(cols)}
        box = np.array(box)
        ids = np.array([int(r[c["id"]]) for r in rows])
        typ = np.array([int(r[c["type"]]) for r in rows])
        if "x" in c:
            pos = np.array([[float(r[c["x"]]), float(r[c["y"]]), float(r[c["z"]])] for r in rows])
        elif "xs" in c:
            s = np.array([[float(r[c["xs"]]), float(r[c["ys"]]), float(r[c["zs"]])] for r in rows])
            pos = box[:, 0] + s * (box[:, 1] - box[:, 0])
        else:
            raise KeyError(f"no x/xs columns in dump; got {cols}")
        order = np.argsort(ids) if sort_by_id else np.arange(len(ids))
        frames.append(dict(natoms=natoms, box=box, ids=ids[order],
                           types=typ[order], pos=pos[order]))
        i += 9 + natoms
    return frames


def read_fo_list(path):
    """Voronoi index per atom. Returns (total[N], vor[N,6]=n3..n8, vol[N])."""
    rows = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) < 9:
                continue
            rows.append([int(t[1])] + [int(x) for x in t[2:8]] + [float(t[8])])
    a = np.array(rows, dtype=float)
    return a[:, 0].astype(int), a[:, 1:7].astype(int), a[:, 7]


def read_nb_id(path):
    """Face-sharing neighbour list. Returns list of int arrays (0-based atom ids)."""
    nbrs = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) < 2:
                continue
            cnt = int(t[1])
            nbrs.append([int(x) for x in t[2:2 + cnt]])
    return nbrs


def is_perfect_icosahedron(vor):
    """vor: (N,6) = n3..n8.  Perfect icosahedron = <0,0,12,0,0,0>."""
    return (vor[:, 0] == 0) & (vor[:, 1] == 0) & (vor[:, 2] == 12) & (vor[:, 3:].sum(1) == 0)


def is_icosahedral_like(vor, n5_min=10):
    """Distorted/partial icosahedra: many pentagonal faces (n5 >= n5_min)."""
    return vor[:, 2] >= n5_min
