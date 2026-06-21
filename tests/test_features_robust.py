import numpy as np
from src.features import thermal_sigma, jitter


def test_thermal_sigma_matches_known_uniform_shift():
    L = np.array([20.0, 20.0, 20.0])
    a = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    b = a + np.array([0.3, 0.0, 0.0])               # every atom moves 0.3 in x
    assert abs(thermal_sigma([a, b], L) - 0.3) < 1e-9


def test_thermal_sigma_uses_minimum_image():
    L = np.array([10.0, 10.0, 10.0])
    a = np.array([[0.05, 0.0, 0.0]])
    b = np.array([[9.95, 0.0, 0.0]])                # really moved 0.1 across the wall
    assert abs(thermal_sigma([a, b], L) - 0.1) < 1e-9


def test_jitter_zero_sigma_is_identity():
    rng = np.random.default_rng(0)
    L = np.array([10.0, 10.0, 10.0])
    pos = rng.uniform(0, 10, size=(50, 3))
    assert np.array_equal(jitter(pos, 0.0, L, rng), pos)


def test_jitter_is_seeded_and_wrapped():
    L = np.array([10.0, 10.0, 10.0])
    pos = np.full((100, 3), 5.0)
    j1 = jitter(pos, 0.2, L, np.random.default_rng(1))
    j2 = jitter(pos, 0.2, L, np.random.default_rng(1))
    assert np.array_equal(j1, j2)                   # reproducible
    assert (j1 >= 0).all() and (j1 < L).all()       # wrapped into [0,L)
