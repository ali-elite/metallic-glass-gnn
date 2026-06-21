"""Models used across the project.

Supervised node classification (Phase 2):
  - MLP  : thesis-style flat neighbour-coordinate vector (permutation-sensitive)
  - CGCNN: distance-aware message passing on the periodic kNN graph (invariant)

Unsupervised community detection (Phase 3):
  - GCNConv / DMoN : a small graph-conv encoder that emits soft cluster
    assignments, trained with a spectral-modularity loss (Tsitsulin et al.,
    "Graph Clustering with Graph Neural Networks", DMoN). No labels are used.
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


# --------------------------------------------------------------------------- #
#  Phase 3: DMoN-style modularity-based GNN community detection (label-free)   #
# --------------------------------------------------------------------------- #
class GCNConv(nn.Module):
    """Symmetric-normalised graph convolution (Kipf & Welling), sparse pure-PyTorch.

    `norm` is the precomputed coefficient 1/sqrt(deg_i * deg_j) per directed edge
    of `edge_index` (which must already include self-loops); see `gcn_norm`.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, norm):
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x)
        msg = norm.unsqueeze(1) * h[src]
        return torch.zeros_like(h).index_add_(0, dst, msg)


def gcn_norm(edge_index, num_nodes):
    """Add self-loops and return (edge_index_with_loops, norm) for GCNConv."""
    loops = torch.arange(num_nodes, dtype=edge_index.dtype).unsqueeze(0).repeat(2, 1)
    ei = torch.cat([edge_index, loops], dim=1)
    deg = torch.zeros(num_nodes).index_add_(
        0, ei[1], torch.ones(ei.shape[1])).clamp_(min=1)
    dinv = deg.pow(-0.5)
    norm = dinv[ei[0]] * dinv[ei[1]]
    return ei, norm


class DMoN(nn.Module):
    """GCN encoder -> softmax soft cluster assignment over K clusters.

    Node features carry chemistry only (element one-hot + radius); all community
    structure must come from message passing on the graph. Never sees labels.
    """
    def __init__(self, in_dim, n_clusters, hidden=64, n_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.convs.append(GCNConv(d, hidden)); d = hidden
        self.assign = nn.Linear(hidden, n_clusters)
        self.dropout = dropout

    def forward(self, x, edge_index, norm):
        h = x
        for conv in self.convs:
            h = F.selu(conv(h, edge_index, norm))
            if self.dropout:
                h = F.dropout(h, self.dropout, self.training)
        return F.softmax(self.assign(h), dim=1)


def dmon_loss(assign, edge_index, deg, m, collapse_w=1.0, entropy_w=0.0):
    """DMoN objective on a soft assignment C (N,K).

    L = -Q_soft + collapse_w * R_collapse  [+ entropy_w * H(per-node assignment)]

      Q_soft       = 1/(2m) [ Tr(Cᵀ A C) - Tr(Cᵀ d dᵀ C)/(2m) ]   (soft modularity)
      R_collapse   = (sqrt(K)/N) * ||sum_i C_i||_F - 1             (anti-collapse)

    `edge_index` is the directed (both-way) adjacency WITHOUT self-loops; `deg`
    are the corresponding node degrees; `m` is the number of undirected edges.
    A small per-node entropy term (optional) sharpens soft assignments.
    """
    src, dst = edge_index[0], edge_index[1]
    intra = (assign[src] * assign[dst]).sum()          # Tr(Cᵀ A C)
    Cd = (assign * deg.unsqueeze(1)).sum(0)            # Cᵀ d  -> (K,)
    null = (Cd * Cd).sum() / (2.0 * m)                 # Tr(Cᵀ d dᵀ C)/(2m)
    modularity = (intra - null) / (2.0 * m)
    N, K = assign.shape
    collapse = (float(K) ** 0.5 / N) * torch.linalg.norm(assign.sum(0)) - 1.0
    loss = -modularity + collapse_w * collapse
    entropy = torch.zeros((), device=assign.device)
    if entropy_w > 0:
        entropy = -(assign * torch.log(assign + 1e-9)).sum(1).mean()
        loss = loss + entropy_w * entropy
    return loss, modularity, collapse, entropy


# --------------------------------------------------------------------------- #
#  Phase 5: CGCNN regression head + Voronoi count loss                         #
# --------------------------------------------------------------------------- #
class CGCNNRegressor(nn.Module):
    """CGCNN trunk + regression head -> 4 non-negative Voronoi counts <n3,n4,n5,n6>
    plus an auxiliary coordination total. Shares `CGConv` with the Phase-2 `CGCNN`
    classifier, which is left untouched. Geometry enters via the edge RBF features,
    so the model is rotation/translation-invariant (half of thermal robustness)."""
    def __init__(self, in_dim, edge_dim, hidden=64, n_layers=3, p=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([CGConv(hidden, edge_dim) for _ in range(n_layers)])
        self.counts = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Dropout(p),
            nn.Linear(hidden, 4),
        )
        self.total = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.softplus(self.embed(x))
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        counts = F.softplus(self.counts(h))             # (N,4) >= 0
        total = F.softplus(self.total(h)).squeeze(1)    # (N,)  >= 0
        return counts, total


def voronoi_loss(counts, total, y_counts, y_total, w_total=0.3, w_sum=0.1):
    """Smooth-L1 on the 4 counts and the coordination total, plus a soft penalty
    tying sum(counts) to the predicted total (pushes the unmodelled n7/n8 slack to 0).

    Returns (loss, l_counts, l_total, l_sum)."""
    l_counts = F.smooth_l1_loss(counts, y_counts)
    l_total = F.smooth_l1_loss(total, y_total)
    l_sum = (counts.sum(dim=1) - total).abs().mean()
    loss = l_counts + w_total * l_total + w_sum * l_sum
    return loss, l_counts, l_total, l_sum
