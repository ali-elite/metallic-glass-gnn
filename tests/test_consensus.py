import numpy as np
from src.voronoi import _row_mode, consensus_index


def test_row_mode_picks_majority_and_count():
    rows = np.array([[0, 0, 12, 0], [0, 0, 12, 0], [0, 1, 10, 2]])
    mode, cnt = _row_mode(rows)
    assert list(mode) == [0, 0, 12, 0]
    assert cnt == 2


def test_row_mode_tie_breaks_toward_first_frame():
    rows = np.array([[0, 2, 8, 2], [0, 0, 12, 0]])  # 1 each -> tie
    mode, cnt = _row_mode(rows)
    assert list(mode) == [0, 2, 8, 2]               # frame 0 wins
    assert cnt == 1


def test_consensus_index_single_frame_is_that_frame():
    fi = np.array([[[0, 0, 12, 0, 0, 0], [0, 1, 10, 4, 0, 0]]])  # (F=1,N=2,6)
    con = consensus_index(fi)
    assert con["label"].tolist() == [[0, 0, 12, 0], [0, 1, 10, 4]]
    assert con["total"].tolist() == [12, 15]
    assert con["instability"].tolist() == [0.0, 0.0]


def test_consensus_index_mode_and_instability():
    # atom 0: ICO in 8 frames, distorted in 3 -> mode ICO, instability 3/11
    f_ico = [0, 0, 12, 0, 0, 0]
    f_dis = [0, 2, 8, 2, 0, 0]
    frames = [f_ico] * 8 + [f_dis] * 3              # 11 frames
    fi = np.array(frames).reshape(11, 1, 6)
    con = consensus_index(fi)
    assert con["label"][0].tolist() == [0, 0, 12, 0]
    assert abs(con["instability"][0] - 3 / 11) < 1e-9
    assert con["total"][0] == 12


def test_consensus_index_total_picks_mode_coordination():
    # coordination 12 in 7 frames, 14 in 4 frames -> mode total 12; label stays ICO
    a = [0, 0, 12, 0, 0, 0]   # coordination 12
    b = [0, 0, 12, 2, 0, 0]   # coordination 14
    fi = np.array([a] * 7 + [b] * 4).reshape(11, 1, 6)
    con = consensus_index(fi)
    assert con["total"][0] == 12
    assert con["label"][0].tolist() == [0, 0, 12, 0]
