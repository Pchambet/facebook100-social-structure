"""Unsupervised link prediction with neighbourhood heuristics, evaluated two ways.

1. **Sampled AUC** - hidden edges versus an equal number of random non-edges. This is the
   usual benchmark, but it is easy: almost any random pair of students has no common
   friend, so the negatives are trivial.
2. **Full ranking** - every unlinked pair of the training graph is a candidate, as it would
   be for a "people you may know" feature. Precision@k and R-precision (k = number of
   hidden edges) on that ranking are the honest numbers; they are only computed on graphs
   small enough for a dense n x n score matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS: tuple[str, ...] = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
)


def _degrees(a: sp.csr_array) -> np.ndarray:
    return np.asarray(a.sum(axis=1)).ravel()


def _inverse(values: np.ndarray, transform=None) -> np.ndarray:
    """1 / transform(values), with 0 where it is undefined (isolated or degree-1 nodes)."""
    x = values.astype(np.float64) if transform is None else transform(values.astype(np.float64))
    return np.divide(1.0, x, out=np.zeros_like(x), where=x > 0)


def _neighbour_weights(a: sp.csr_array, method: str) -> np.ndarray:
    k = _degrees(a)
    if method == "adamic_adar":
        # A common neighbour has degree >= 2, so log(k) > 0 wherever the weight is used.
        return _inverse(np.maximum(k, 1), np.log)
    if method == "resource_allocation":
        return _inverse(k)
    return np.ones_like(k, dtype=np.float64)


def split_edges(
    adjacency: sp.csr_array, test_fraction: float, rng: np.random.Generator
) -> tuple[sp.csr_array, np.ndarray]:
    """Hide a uniform random fraction of edges. Returns (training graph, hidden edges u < v)."""
    upper = sp.coo_array(sp.triu(adjacency, k=1))
    edges = np.column_stack([upper.row, upper.col])
    n_test = round(test_fraction * len(edges))
    test_idx = rng.choice(len(edges), size=n_test, replace=False)
    keep = np.ones(len(edges), dtype=bool)
    keep[test_idx] = False
    train_edges = edges[keep]
    n = adjacency.shape[0]
    upper_train = sp.coo_array(
        (np.ones(len(train_edges)), (train_edges[:, 0], train_edges[:, 1])), shape=(n, n)
    )
    return sp.csr_array(upper_train + upper_train.T), edges[test_idx]


def sample_non_edges(adjacency: sp.csr_array, size: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random unlinked pairs (u < v) of the *full* graph, by rejection sampling."""
    n = adjacency.shape[0]
    a = sp.csr_array(adjacency)
    if size > n * (n - 1) // 2 - a.nnz // 2:
        raise ValueError(f"cannot sample {size} non-edges from this graph")
    pairs: dict[tuple[int, int], None] = {}  # insertion-ordered set, for reproducibility
    while len(pairs) < size:
        u = rng.integers(0, n, size=2 * size)
        v = rng.integers(0, n, size=2 * size)
        u, v = np.minimum(u, v), np.maximum(u, v)
        ok = (u != v) & (np.asarray(a[u, v]).ravel() == 0)
        for pair in zip(u[ok].tolist(), v[ok].tolist(), strict=True):
            pairs[pair] = None
            if len(pairs) == size:
                break
    return np.array(list(pairs), dtype=np.int64).reshape(-1, 2)


def score_pairs(
    train: sp.csr_array, pairs: np.ndarray, method: str, chunk: int = 50_000
) -> np.ndarray:
    """Heuristic score for each (u, v) pair, computed from the training graph only."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    a = sp.csr_array(train)
    k = _degrees(a)
    u, v = pairs[:, 0], pairs[:, 1]
    if method == "preferential_attachment":
        return k[u] * k[v]
    weights = _neighbour_weights(a, method)
    scores = np.empty(len(pairs), dtype=np.float64)
    for s in range(0, len(pairs), chunk):
        uc, vc = u[s : s + chunk], v[s : s + chunk]
        common = a[uc].multiply(a[vc])  # row i marks the common neighbours of pair i
        scores[s : s + chunk] = common @ weights
    if method == "jaccard":
        union = k[u] + k[v] - scores
        scores = np.divide(scores, union, out=np.zeros_like(scores), where=union > 0)
    return scores


def score_matrix(train: sp.csr_array, method: str) -> np.ndarray:
    """Dense n x n score matrix (same definitions as `score_pairs`)."""
    a = sp.csr_array(train)
    k = _degrees(a)
    if method == "preferential_attachment":
        return np.outer(k, k)
    weights = _neighbour_weights(a, method)
    common = (a @ sp.diags_array(weights) @ a).toarray()
    if method == "jaccard":
        cn = (a @ a).toarray()
        union = k[:, None] + k[None, :] - cn
        return np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
    return common


def precision_at(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Precision of the top-k pairs, in expectation over random tie-breaking.

    Heuristic scores are heavily tied (common-neighbour counts are small integers), so a
    plain sort would let array order decide which tied pairs make the cut. Pairs scoring
    above the k-th largest value are all in; the remaining slots are filled from the tied
    block, whose expected share of positives is its positive rate. No full sort needed.
    """
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > threshold, scores == threshold
    n_above = int(above.sum())
    hits = y[above].sum() + (k - n_above) * y[tied].sum() / tied.sum()
    return float(hits / k)


def evaluate(
    adjacency: sp.csr_array,
    methods: tuple[str, ...] = METHODS,
    test_fraction: float = 0.1,
    seed: int = 0,
    full_ranking: bool = True,
    ks: tuple[int, ...] = (100,),
) -> list[dict[str, float | str | int]]:
    """Score every method on one random edge split of one graph."""
    rng = np.random.default_rng(seed)
    train, test = split_edges(adjacency, test_fraction, rng)
    negatives = sample_non_edges(adjacency, len(test), rng)
    sampled_pairs = np.vstack([test, negatives])
    sampled_y = np.r_[np.ones(len(test)), np.zeros(len(negatives))]

    if full_ranking:
        iu, ju = np.triu_indices(adjacency.shape[0], k=1)
        candidate = train.toarray()[iu, ju] == 0
        iu, ju = iu[candidate], ju[candidate]
        hidden = np.zeros(adjacency.shape, dtype=bool)
        hidden[test[:, 0], test[:, 1]] = True
        y_full = hidden[iu, ju].astype(np.float64)

    rows = []
    for method in methods:
        row: dict[str, float | str | int] = {
            "method": method,
            "seed": seed,
            "hidden_edges": len(test),
            "auc_sampled": float(
                roc_auc_score(sampled_y, score_pairs(train, sampled_pairs, method))
            ),
        }
        if full_ranking:
            scores = score_matrix(train, method)[iu, ju]
            row["candidates"] = len(iu)
            row["base_rate"] = float(y_full.mean())
            row["average_precision"] = float(average_precision_score(y_full, scores))
            row["r_precision"] = precision_at(scores, y_full, len(test))
            for k in ks:
                row[f"precision_at_{k}"] = precision_at(scores, y_full, k)
        rows.append(row)
    return rows
