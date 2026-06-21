import torch
from src.models import CGCNNCountClassifier, count_ce_loss, consistency_loss


def test_consistency_loss_zero_for_identical_positive_otherwise():
    a = [torch.randn(6, 5) for _ in range(4)]
    assert float(consistency_loss(a, [t.clone() for t in a])) < 1e-7
    b = [torch.randn(6, 5) for _ in range(4)]
    assert float(consistency_loss(a, b)) > 0


def _toy_graph(N=12, edge_dim=16):
    x = torch.randn(N, 1)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_attr = torch.rand(edge_index.shape[1], edge_dim)
    return x, edge_index, edge_attr


def test_classifier_emits_four_logit_heads():
    x, ei, ea = _toy_graph()
    model = CGCNNCountClassifier(in_dim=1, edge_dim=16, hidden=16, n_layers=2, max_count=10)
    logits = model(x, ei, ea)
    assert isinstance(logits, list) and len(logits) == 4
    for l in logits:
        assert l.shape == (12, 10)


def test_count_ce_loss_clamps_targets_above_max():
    x, ei, ea = _toy_graph(N=5)
    model = CGCNNCountClassifier(in_dim=1, edge_dim=16, hidden=8, n_layers=1, max_count=4)
    y = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8], [0, 0, 0, 0],
                      [1, 1, 1, 1], [3, 3, 3, 3]])     # row 1 exceeds max -> must clamp
    loss = count_ce_loss(model(x, ei, ea), y)
    assert torch.isfinite(loss)


def test_classifier_overfits_small_data_via_argmax():
    torch.manual_seed(0)
    x, ei, ea = _toy_graph(N=8)
    y = torch.randint(0, 10, (8, 4))
    model = CGCNNCountClassifier(in_dim=1, edge_dim=16, hidden=32, n_layers=2, max_count=10, p=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first = None
    for _ in range(250):
        loss = count_ce_loss(model(x, ei, ea), y)
        first = float(loss) if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first                          # learned
    model.eval()
    pred = torch.stack([l.argmax(1) for l in model(x, ei, ea)], dim=1)
    assert (pred == y).float().mean() > 0.5             # overfits the tiny set
