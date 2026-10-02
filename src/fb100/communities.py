"""Louvain communities compared with node attributes.

The question (Traud et al. 2012) is which attribute, if any, the modular structure of
each campus follows. Agreement is measured with the adjusted Rand index (0 = chance,
1 = identical partitions) and normalised mutual information, on labelled nodes only.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from fb100.io import MISSING


def louvain(adjacency: sp.csr_array, seed: int = 0) -> tuple[np.ndarray, float]:
    """Community label per node and the modularity of the partition."""
    graph = nx.from_scipy_sparse_array(adjacency)
    parts = nx.community.louvain_communities(graph, seed=seed)
    labels = np.empty(adjacency.shape[0], dtype=np.int64)
    for i, members in enumerate(parts):
        labels[list(members)] = i
    return labels, float(nx.community.modularity(graph, parts))


def agreement(
    partition: np.ndarray, attributes: pd.DataFrame, missing: int = MISSING
) -> list[dict[str, float | str]]:
    """ARI and NMI between the partition and each attribute, ignoring missing values."""
    rows = []
    for name in attributes.columns:
        values = attributes[name].to_numpy()
        keep = values != missing
        rows.append(
            {
                "attribute": name,
                "ari": float(adjusted_rand_score(values[keep], partition[keep])),
                "nmi": float(normalized_mutual_info_score(values[keep], partition[keep])),
            }
        )
    return rows
