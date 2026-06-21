import torch
from src.models import CGCNNRegressor, voronoi_loss


def _toy_graph(N=12, edge_dim=16):
    x = torch.randn(N, 1)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_attr = torch.rand(edge_index.shape[1], edge_dim)
    return x, edge_index, edge_attr


def test_regressor_output_shapes_and_nonneg():
    x, ei, ea = _toy_graph()
    model = CGCNNRegressor(in_dim=1, edge_dim=16, hidden=16, n_layers=2)
    counts, total = model(x, ei, ea)
    assert counts.shape == (12, 4)
    assert total.shape == (12,)
    assert (counts >= 0).all() and (total >= 0).all()    # softplus outputs


def test_regressor_gradients_flow():
    x, ei, ea = _toy_graph()
    model = CGCNNRegressor(in_dim=1, edge_dim=16, hidden=16, n_layers=2)
    counts, total = model(x, ei, ea)
    y_c = torch.zeros(12, 4); y_t = torch.zeros(12)
    loss, *_ = voronoi_loss(counts, total, y_c, y_t)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_voronoi_loss_zero_at_perfect_consistent_prediction():
    # counts sum to total and equal targets, n7/n8 = 0 -> all three terms zero
    counts = torch.tensor([[0.0, 0.0, 12.0, 0.0], [0.0, 2.0, 8.0, 2.0]])
    total = torch.tensor([12.0, 12.0])
    loss, lc, lt, ls = voronoi_loss(counts, total, counts.clone(), total.clone())
    assert float(lc) < 1e-6 and float(lt) < 1e-6 and float(ls) < 1e-6


def test_voronoi_loss_positive_when_wrong():
    counts = torch.tensor([[0.0, 0.0, 12.0, 0.0]])
    total = torch.tensor([12.0])
    y_c = torch.tensor([[0.0, 3.0, 6.0, 4.0]])
    y_t = torch.tensor([13.0])
    loss, *_ = voronoi_loss(counts, total, y_c, y_t)
    assert float(loss) > 0
