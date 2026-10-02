import networkx as nx
import numpy as np
import pytest

from fb100.homophily import attribute_assortativity
from fb100.io import clean_adjacency


def _adj(g):
    return clean_adjacency(nx.to_scipy_sparse_array(g, nodelist=sorted(g)))


def test_matches_networkx_on_labelled_subgraph():
    rng = np.random.default_rng(0)
    g = nx.gnp_random_graph(200, 0.06, seed=1)
    labels = rng.integers(0, 4, size=200)  # 0 = missing
    nx.set_node_attributes(g, dict(enumerate(labels.tolist())), "x")
    labelled = g.subgraph([i for i in g if labels[i] != 0])
    expected = nx.attribute_assortativity_coefficient(labelled, "x")
    assert attribute_assortativity(_adj(g), labels) == pytest.approx(expected)


def test_extremes():
    two_cliques = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    groups = np.array([1] * 5 + [2] * 5)
    assert attribute_assortativity(_adj(two_cliques), groups) == pytest.approx(1.0)

    bipartite = nx.complete_bipartite_graph(5, 5)
    assert attribute_assortativity(_adj(bipartite), groups) == pytest.approx(-1.0)


def test_single_category_is_undefined():
    assert np.isnan(attribute_assortativity(_adj(nx.path_graph(4)), np.array([1, 1, 1, 0])))


def test_recovers_planted_homophily(sbm):
    # Four equal blocks: share of within-block edge ends f = p_in / (p_in + 3 p_out),
    # and r = (f - 1/4) / (1 - 1/4).
    a, labels = sbm
    f = 0.12 / (0.12 + 3 * 0.01)
    assert attribute_assortativity(a, labels) == pytest.approx((f - 0.25) / 0.75, abs=0.03)
