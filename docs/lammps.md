# The molecular-dynamics (LAMMPS) stage

> **Note.** The original LAMMPS input scripts for this project were not preserved.
> This document is a faithful **reconstruction** of how the trajectories in
> `samples1/`, `samples2/`, and `samples3/` were almost certainly produced,
> inferred from the data itself. It is meant as a methods description for the
> report and as a runnable template if the samples ever need to be regenerated.

## 1. What the data tells us

Every trajectory is a LAMMPS *custom dump* in `units metal`:

```
ITEM: TIMESTEP
22000000
ITEM: NUMBER OF ATOMS
13500
ITEM: BOX BOUNDS pp pp pp        <- periodic in x, y, z
-3.70037 56.2004
-3.70037 56.2004
-3.70037 56.2004
ITEM: ATOMS id type x y z vx vy vz fx fy fz
```

What we can read off directly:

- **Periodic, 3D bulk** (`pp pp pp`) — a bulk glass, no surfaces.
- **Cu–Zr**, two atom types (`samples2/in_al.txt` → `n_atome_types 2`), atomic
  radii in `samples3/R.txt` (Cu 1.28, Zr 1.60, Ni 1.24, Al 1.42 Å …).
- **Size / density:** 13,500 atoms in a ≈ 59.9 Å cube ⇒ ρ ≈ 0.063 atoms/Å³,
  the right number density for a Cu–Zr metallic glass.
- **Quenched & relaxed at 300 K:** the `samples3` filenames literally say
  `...-Glass-QuenchedAndRelaxed-300K.lammpsTrj`, and those boxes are slightly
  **anisotropic** (e.g. 50.8 × 52.8 × 49.7 Å) — the signature of an
  **NPT** relaxation that let each box edge relax independently.
- Positions **and** velocities **and** forces are dumped — a finite-temperature
  equilibrium snapshot, not a 0 K minimisation.

## 2. How the samples were made (melt–quench)

This is the standard recipe for a simulated bulk metallic glass:

1. **Build** a random substitutional alloy at the target composition
   (e.g. Cu₆₄Zr₃₆), then energy-minimise.
2. **Melt / equilibrate** the liquid well above the melting point (~2000 K) under
   NPT at zero pressure for ~1 ns, to erase any memory of the initial lattice.
3. **Quench** from the melt to 300 K at a controlled **cooling rate**. This is the
   single most important knob: **slower cooling → more icosahedral order**
   (and a denser, more percolated icosahedral network).
4. **Relax / anneal** at 300 K under NPT (this is the *“QuenchedAndRelaxed-300K”*
   step) and dump the final configuration.

## 3. Reconstructed input script (template)

```lammps
# ---- Cu-Zr metallic-glass melt-quench  (RECONSTRUCTION, not the original) ----
units           metal
atom_style      atomic
boundary        p p p

# 1) random Cu-Zr alloy at the target composition (here Cu64Zr36)
lattice         fcc 3.61
region          box block 0 25 0 25 0 25
create_box      2 box
create_atoms    1 box
set             group all type/fraction 2 0.36 12345   # ~36% -> Zr
mass            1 63.546        # Cu
mass            2 91.224        # Zr

# 2) EAM/Finnis-Sinclair potential (e.g. Mendelev et al. 2009 for Cu-Zr)
pair_style      eam/fs
pair_coeff      * * CuZr.eam.fs Cu Zr

# 3) melt & equilibrate the liquid
velocity        all create 2000.0 12345 mom yes rot yes dist gaussian
timestep        0.002                                  # 2 fs
fix             1 all npt temp 2000.0 2000.0 0.1 iso 0.0 0.0 1.0
thermo          1000
run             500000                                 # ~1 ns at 2000 K

# 4) quench 2000 K -> 300 K.  rate = (2000-300)/(run*dt).
#    e.g. 1.7e6 steps * 2 fs = 3.4 ns  =>  ~5e11 K/s  (slower = more icosahedra)
fix             1 all npt temp 2000.0 300.0 0.1 iso 0.0 0.0 1.0
run             1700000

# 5) relax / anneal at 300 K  (the "QuenchedAndRelaxed-300K" stage)
fix             1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run             200000

# 6) dump the final snapshot -> *.lammpsTrj
dump            d all custom 1 out.glass.lammpsTrj id type x y z vx vy vz fx fy fz
run             0
```

The exact potential file, system size, cooling rate, and timestep would have been
chosen by whoever ran it; the values above are representative and reproduce the
density, composition, and structure seen in the data.

## 4. From trajectory to graph (the Voronoi post-processing)

The final coordinates were fed to **Voro++** using the atomic radii in `R.txt`
(a *radical / power* Voronoi tessellation, appropriate for unequal atom sizes).
For every atom this yields:

- its **Voronoi index** `<n3,n4,n5,n6,...>` — the count of polyhedron faces with
  3, 4, 5, 6, … edges (`fo_list`, `Face_order_list`). A perfect icosahedron is
  `<0,0,12,0>`;
- its **face-sharing neighbours** (`nb_id`) — two atoms are bonded iff their
  Voronoi cells share a face. This is the physically-grounded adjacency we use to
  build the atomic graph (mean coordination ≈ 13.7, exactly as expected).

So: **MD (LAMMPS) → snapshot → radical Voronoi (Voro++) → attributed graph**
(nodes = atoms, node labels = Voronoi index, edges = shared faces), which is the
input to the GNN stage.
