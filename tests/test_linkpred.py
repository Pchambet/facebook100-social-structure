import networkx as nx
import numpy as np
import pytest
from conftest import planted_partition

from fb100.io import clean_adjacency
from fb100.linkpred import (
    METHODS,
    evaluate,
    precision_at,
    sample_non_edges,
    score_matrix,
    score_pairs,
    split_edges,
)

NX_REFERENCE = {
    "jaccard": nx.jaccard_coefficient,
    "adamic_adar": nx.adamic_adar_index,
    "resource_allocation": nx.resource_allocation_index,
    "preferential_attachment": nx.preferential_attachment,
}


@pytest.fixture
def karate_adj(karate):
    return clean_adjacency(nx.to_scipy_sparse_array(karate, nodelist=range(34), weight=None))


@pytest.mark.parametrize("method", METHODS)
def test_scores_match_networkx_and_dense_matrix(karate, karate_adj, method):
    pairs = np.array(list(nx.non_edges(karate)))
    got = score_pairs(karate_adj, pairs, method, chunk=17)
    if method == "common_neighbors":
        expected = [len(list(nx.common_neighbors(karate, u, v))) for u, v in pairs]
    else:
        expected = [p for _, _, p in NX_REFERENCE[method](karate, pairs.tolist())]
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(score_matrix(karate_adj, method)[pairs[:, 0], pairs[:, 1]], got)


def test_split_hides_exact_fraction_without_overlap(karate_adj):
    train, test = split_edges(karate_adj, 0.2, np.random.default_rng(0))
    assert len(test) == round(0.2 * 78)
    assert train.nnz // 2 + len(test) == 78
    assert np.all(np.asarray(train[test[:, 0], test[:, 1]]).ravel() == 0)
    assert np.all(np.asarray(karate_adj[test[:, 0], test[:, 1]]).ravel() == 1)


def test_non_edges_are_unique_unlinked_pairs(karate_adj):
    pairs = sample_non_edges(karate_adj, 200, np.random.default_rng(1))
    assert len({tuple(p) for p in pairs.tolist()}) == 200
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert np.all(np.asarray(karate_adj[pairs[:, 0], pairs[:, 1]]).ravel() == 0)
    with pytest.raises(ValueError):
        sample_non_edges(karate_adj, 34 * 33 // 2 - 78 + 1, np.random.default_rng(1))


def test_neighbourhood_scores_recover_planted_edges():
    # In a planted partition hidden edges sit inside dense blocks, so shared neighbours
    # reveal them; degrees are homogeneous, so preferential attachment is near chance.
    a, _ = planted_partition([100] * 4, p_in=0.3, p_out=0.005, seed=3)
    rows = {r["method"]: r for r in evaluate(a, seed=0)}
    assert rows["adamic_adar"]["auc_sampled"] > 0.85
    assert abs(rows["preferential_attachment"]["auc_sampled"] - 0.5) < 0.1
    # Inside a block, edges are i.i.d., so the best any heuristic can do on the full ranking
    # is to find the block: P(hidden | unlinked within-block pair) = 0.1 p_in / (1 - 0.9 p_in).
    ceiling = 0.1 * 0.3 / (1 - 0.9 * 0.3)
    assert rows["adamic_adar"]["r_precision"] == pytest.approx(ceiling, abs=0.01)
    assert rows["adamic_adar"]["r_precision"] > 4 * rows["adamic_adar"]["base_rate"]


def test_precision_at_k_averages_over_ties():
    scores = np.array([5.0, 3, 3, 3, 3, 1])
    y = np.array([1.0, 1, 0, 0, 0, 1])
    assert precision_at(scores, y, 1) == 1.0
    # Top-3 = the 5 plus two of four tied 3s, of which one in four is positive.
    assert precision_at(scores, y, 3) == pytest.approx((1 + 2 * 0.25) / 3)
    assert precision_at(scores, y, 6) == pytest.approx(0.5)
