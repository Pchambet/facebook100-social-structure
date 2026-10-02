import networkx as nx
import numpy as np
import pytest

from fb100.io import clean_adjacency
from fb100.labelprop import evaluate, propagate


def _adj(g):
    return clean_adjacency(nx.to_scipy_sparse_array(g, nodelist=sorted(g)))


def test_unknown_node_between_two_identical_labels():
    assert propagate(_adj(nx.path_graph(3)), np.array([5, 0, 5])).tolist() == [5, 5, 5]


def test_harmonic_solution_on_a_path():
    # Harmonic interpolation on 0-1-2-3-4 with ends 7 and 9: node 1 leans to 7, node 3 to 9.
    pred = propagate(_adj(nx.path_graph(5)), np.array([7, 0, 0, 0, 9]))
    assert pred[1] == 7 and pred[3] == 9


def test_unreachable_nodes_get_majority_label():
    g = nx.Graph([(0, 1), (1, 2), (3, 4)])
    pred = propagate(_adj(g), np.array([1, 1, 2, 0, 0]))
    assert pred[3] == 1 and pred[4] == 1


def test_requires_at_least_one_label():
    with pytest.raises(ValueError):
        propagate(_adj(nx.path_graph(3)), np.zeros(3, dtype=int))


def test_recovers_planted_blocks(sbm):
    a, labels = sbm
    result = evaluate(a, labels, hidden_fraction=0.5, seed=0)
    assert result["accuracy"] > 0.95
    assert result["majority_baseline"] < 0.35
