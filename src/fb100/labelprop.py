"""Label propagation (Zhu, Ghahramani & Lafferty 2003) for missing node attributes.

Known labels are clamped and every other node repeatedly takes the average label
distribution of its neighbours, which converges to the harmonic solution. Nodes whose
value is missing in the data are *free* nodes, not zero-clamped sinks: clamping them
to an all-zero distribution (as the first version of this project did) drains
probability mass and biases predictions towards high-degree regions.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fb100.io import MISSING


def propagate(
    adjacency: sp.csr_array,
    seeds: np.ndarray,
    missing: int = MISSING,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Predict a label for every node from the labels in `seeds` (`missing` = unknown).

    Nodes that no labelled node can reach get the majority label. The default tolerance
    (max change of any class probability) is loose on purpose: on a 9,400-node campus the
    predictions match those obtained at 1e-6 for 99.9 % of hidden nodes, in half the time.
    """
    seeds = np.asarray(seeds)
    known = seeds != missing
    classes, codes, counts = np.unique(seeds[known], return_inverse=True, return_counts=True)
    if len(classes) == 0:
        raise ValueError("no labelled node to propagate from")
    n = adjacency.shape[0]
    k = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_k = np.divide(1.0, k, out=np.zeros_like(k, dtype=np.float64), where=k > 0)
    transition = sp.csr_array(sp.diags_array(inv_k) @ adjacency)

    clamped = np.zeros((int(known.sum()), len(classes)))
    clamped[np.arange(len(codes)), codes] = 1.0
    f = np.zeros((n, len(classes)))
    f[known] = clamped
    for _ in range(max_iter):
        nxt = transition @ f
        nxt[known] = clamped
        delta = np.abs(nxt - f).max()
        f = nxt
        if delta < tol:
            break

    best = classes[f.argmax(axis=1)]
    unreached = f.sum(axis=1) == 0
    best[unreached] = classes[counts.argmax()]
    return best


def evaluate(
    adjacency: sp.csr_array,
    labels: np.ndarray,
    hidden_fraction: float,
    seed: int,
    missing: int = MISSING,
) -> dict[str, float | int]:
    """Hide a random share of the known labels, predict them, compare with the majority class."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    labelled = np.flatnonzero(labels != missing)
    hidden = rng.choice(labelled, size=round(hidden_fraction * len(labelled)), replace=False)
    seeds = labels.copy()
    seeds[hidden] = missing

    predicted = propagate(adjacency, seeds, missing=missing)[hidden]
    visible = seeds[seeds != missing]
    values, counts = np.unique(visible, return_counts=True)
    majority = values[counts.argmax()]
    truth = labels[hidden]
    return {
        "hidden": len(hidden),
        "classes": len(values),
        "accuracy": float(np.mean(predicted == truth)),
        "majority_baseline": float(np.mean(truth == majority)),
    }
