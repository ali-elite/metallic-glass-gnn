import numpy as np
from src.metrics import ico_from_counts, flip_rate, exact_match, per_count_mae


def test_ico_from_counts():
    c = np.array([[0, 0, 12, 0], [0, 2, 8, 2], [0, 0, 12, 1]])
    assert ico_from_counts(c).tolist() == [True, False, False]


def test_flip_rate_zero_when_constant():
    idx = np.zeros((11, 5, 4), dtype=int)            # all frames identical
    assert flip_rate(idx) == 0.0


def test_flip_rate_counts_atoms_that_change():
    idx = np.zeros((3, 4, 4), dtype=int)
    idx[1, 0, 2] = 1                                 # atom 0 differs in frame 1
    assert abs(flip_rate(idx) - 0.25) < 1e-9         # 1 of 4 atoms flips


def test_exact_match():
    pred = np.array([[0, 0, 12, 0], [0, 1, 10, 4]])
    true = np.array([[0, 0, 12, 0], [0, 2, 8, 2]])
    assert abs(exact_match(pred, true) - 0.5) < 1e-9


def test_per_count_mae():
    pred = np.array([[0, 0, 12, 0], [0, 0, 10, 2]])
    true = np.array([[0, 0, 12, 0], [0, 0, 12, 0]])
    assert per_count_mae(pred, true).tolist() == [0.0, 0.0, 1.0, 1.0]
