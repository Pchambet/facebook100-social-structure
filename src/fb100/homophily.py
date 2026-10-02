"""Attribute assortativity (Newman 2003) with missing values handled explicitly.

Facebook100 codes missing attributes as 0. Treating 0 as a real category merges every
student with an unknown dorm into one giant fake "dorm", which biases r. Following
Traud et al. (2012), only edges whose two endpoints are both labelled enter the
mixing matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fb100.io import MISSING


def attribute_assortativity(
    adjacency: sp.csr_array, labels: np.ndarray, missing: int | None = MISSING
) -> float:
    """Assortativity coefficient r for a categorical node attribute.

    r = (sum_i e_ii - sum_i a_i^2) / (1 - sum_i a_i^2), where e is the symmetric
    edge-end mixing matrix. r = 1 means perfect homophily, 0 random mixing.
    Returns NaN when fewer than two categories remain after dropping missing values.
    """
    a = sp.coo_array(sp.triu(adjacency, k=1))
    labels = np.asarray(labels)
    u, v = labels[a.row], labels[a.col]
    if missing is not None:
        keep = (u != missing) & (v != missing)
        u, v = u[keep], v[keep]
    categories, codes = np.unique(np.concatenate([u, v]), return_inverse=True)
    if len(categories) < 2:
        return float("nan")
    cu, cv = codes[: len(u)], codes[len(u) :]
    c = len(categories)
    mixing = np.bincount(cu * c + cv, minlength=c * c).reshape(c, c).astype(np.float64)
    mixing = mixing + mixing.T
    mixing /= mixing.sum()
    marginal = mixing.sum(axis=1)
    expected = float(marginal @ marginal)
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((np.trace(mixing) - expected) / (1 - expected))


def labelled_share(labels: np.ndarray, missing: int = MISSING) -> float:
    """Fraction of nodes with a known value."""
    return float(np.mean(np.asarray(labels) != missing))
