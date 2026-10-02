import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fb100.io import School, clean_adjacency
from fb100.structure import degree_assortativity, local_clustering, summarize, triangles


@pytest.fixture
def random_graph():
    return nx.gnp_random_graph(300, 0.05, seed=3)


def test_triangles_match_networkx(random_graph):
    a = clean_adjacency(nx.to_scipy_sparse_array(random_graph, nodelist=range(300)))
    expected = np.array([nx.triangles(random_graph)[i] for i in range(300)])
    np.testing.assert_allclose(triangles(a, chunk=37), expected)


def test_clustering_and_assortativity_match_networkx(karate):
    a = clean_adjacency(nx.to_scipy_sparse_array(karate, nodelist=range(34), weight=None))
    clustering = nx.clustering(karate)
    np.testing.assert_allclose(local_clustering(a), [clustering[i] for i in range(34)])
    assert degree_assortativity(a) == pytest.approx(nx.degree_assortativity_coefficient(karate))


def test_summary_on_hand_checkable_graph():
    # A triangle plus a pendant node: 4 nodes, 4 edges, one triangle.
    g = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)])
    a = clean_adjacency(nx.to_scipy_sparse_array(g, nodelist=range(4)))
    row = summarize(School("toy", a, pd.DataFrame()))
    assert row["edges"] == 4
    assert row["mean_degree"] == 2.0
    assert row["density"] == pytest.approx(4 / 6)
    assert row["transitivity"] == pytest.approx(nx.transitivity(g))
    assert row["avg_clustering"] == pytest.approx(nx.average_clustering(g))
    assert row["giant_component_share"] == 1.0
