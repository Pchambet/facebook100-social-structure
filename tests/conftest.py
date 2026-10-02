"""Shared synthetic graphs: tests never need the (large, access-restricted) real data."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from fb100.io import clean_adjacency


def planted_partition(
    sizes: list[int], p_in: float, p_out: float, seed: int
) -> tuple[sp.csr_array, np.ndarray]:
    """Stochastic block model adjacency plus the block of each node (labels start at 1)."""
    k = len(sizes)
    probs = np.full((k, k), p_out)
    np.fill_diagonal(probs, p_in)
    g = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)
    labels = np.repeat(np.arange(1, k + 1), sizes)
    return clean_adjacency(nx.to_scipy_sparse_array(g, nodelist=range(sum(sizes)))), labels


@pytest.fixture
def sbm() -> tuple[sp.csr_array, np.ndarray]:
    return planted_partition([150, 150, 150, 150], p_in=0.12, p_out=0.01, seed=7)


@pytest.fixture
def karate() -> nx.Graph:
    return nx.karate_club_graph()
