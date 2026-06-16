"""Two node classifiers on the same binary icosahedron task:
  - MLP  : thesis-style flat neighbour-coordinate vector (permutation-sensitive)
  - CGCNN: distance-aware message passing on the periodic kNN graph (invariant)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Faithful to the thesis: Linear + BatchNorm + Dropout + LeakyReLU blocks."""
    def __init__(self, in_dim, hidden=128, n_classes=2, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.net(x)


class CGConv(nn.Module):
    """Crystal-graph convolution (Xie & Grossman). Geometry enters via edge RBF features."""
    def __init__(self, dim, edge_dim):
        super().__init__()
        z = 2 * dim + edge_dim
        self.f = nn.Linear(z, dim)   # filter (sigmoid gate)
        self.s = nn.Linear(z, dim)   # core   (softplus)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        z = torch.cat([h[src], h[dst], edge_attr], dim=1)
        msg = torch.sigmoid(self.f(z)) * F.softplus(self.s(z))
        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        deg = torch.zeros(h.size(0), device=h.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=h.dtype)).clamp_(min=1)
        return h + self.bn(agg / deg.unsqueeze(1))


class CGCNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, n_layers=3, n_classes=2, p=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([CGConv(hidden, edge_dim) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.softplus(self.embed(x))
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        return self.head(h)
