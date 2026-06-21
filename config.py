"""Paths to the raw trajectory/label data. Edit DATA_DIR if you move the data."""
import os

DATA_DIR = os.environ.get(
    "MG_DATA_DIR",
    "/Users/alielite/Desktop/Ali/Research - Sharif/Tavakoli Project",
)  # raw data lives outside the repo; override with the MG_DATA_DIR env var
SAMPLES1 = os.path.join(DATA_DIR, "samples1")  # Cu64Zr36, 13500 atoms (labels + trajectory)
SAMPLES2 = os.path.join(DATA_DIR, "samples2")  # 10000 atoms: trajectory + Voronoi labels + face-sharing neighbour list
SAMPLES3 = os.path.join(DATA_DIR, "samples3")  # many alloy systems / compositions (trajectories only)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
