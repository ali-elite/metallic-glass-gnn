import numpy as np
import pytest

pytest.importorskip("pyvoro")
from src.voronoi import voronoi_index, voronoi_index_frames


def test_voronoi_index_frames_stacks_per_frame():
    rng = np.random.default_rng(0)
    L = np.array([10.0, 10.0, 10.0])
    pos = rng.uniform(0, 10, size=(40, 3))
    radii = np.full(40, 1.3)
    frames = [pos, (pos + 0.01) % L]
    fi = voronoi_index_frames(frames, L, radii)
    assert fi.shape == (2, 40, 6)
    assert np.array_equal(fi[0], voronoi_index(pos, L, radii))
