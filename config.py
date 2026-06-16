"""Paths to the raw thesis data. Edit THESIS_DIR if you move the data."""
import os

THESIS_DIR = os.environ.get(
    "MG_THESIS_DIR",
    "/Users/alielite/Desktop/Ali/Research - Sharif/Tavakoli Project",
)  # raw data lives outside the repo; override with the MG_THESIS_DIR env var
SAMPLES1 = os.path.join(THESIS_DIR, "samples1")  # Cu64Zr36, 13500 atoms (labels + trajectory)
SAMPLES2 = os.path.join(THESIS_DIR, "samples2")  # 10000 atoms: trajectory + Voronoi labels + face-sharing neighbour list
SAMPLES3 = os.path.join(THESIS_DIR, "samples3")  # many alloy systems / compositions (trajectories only)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
