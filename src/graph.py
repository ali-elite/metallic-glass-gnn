"""Build the atomic graph and the physical (ground-truth) icosahedral network."""
import networkx as nx


def build_graph(nbrs):
    """Undirected graph: nodes = atoms, edges = Voronoi face-sharing pairs."""
    n = len(nbrs)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, nb in enumerate(nbrs):
        for j in nb:
            if 0 <= j < n and j != i:
                G.add_edge(i, j)
    return G


def edge_symmetry(nbrs):
    """Fraction of directed neighbour relations that are reciprocated (sanity check)."""
    s = [set(nb) for nb in nbrs]
    tot = recip = 0
    for i, nb in enumerate(s):
        for j in nb:
            if 0 <= j < len(s):
                tot += 1
                if i in s[j]:
                    recip += 1
    return recip / max(tot, 1)


def physical_communities(G, mask):
    """Ground-truth communities = connected components of the subgraph induced
    by the selected (e.g. icosahedral) atoms. This is the icosahedral network /
    medium-range-order backbone studied in the metallic-glass literature."""
    nodes = [i for i in G.nodes if mask[i]]
    H = G.subgraph(nodes)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    return comps, H
