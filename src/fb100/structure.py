"""Structural summaries computed directly on sparse matrices.

NetworkX is convenient but too slow and memory-hungry for the 1.6M-edge campuses, so
triangles are counted with chunked sparse products: memory stays bounded by the chunk
size instead of by the full A @ A, which can have hundreds of millions of entries.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from fb100.io import School


def triangles(adjacency: sp.csr_array, chunk: int = 2048) -> np.ndarray:
    """Number of triangles through each node: diag(A^3) / 2."""
    a = sp.csr_array(adjacency)
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        rows = a[start : start + chunk]
        paths = rows @ a  # length-2 walks from the chunk's nodes
        out[start : start + chunk] = np.asarray(paths.multiply(rows).sum(axis=1)).ravel() / 2
    return out


def local_clustering(adjacency: sp.csr_array) -> np.ndarray:
    """Local clustering coefficient; 0 for nodes of degree < 2 (NetworkX convention)."""
    k = np.asarray(adjacency.sum(axis=1)).ravel()
    pairs = k * (k - 1) / 2
    t = triangles(adjacency)
    return np.divide(t, pairs, out=np.zeros_like(t), where=pairs > 0)


def degree_assortativity(adjacency: sp.csr_array) -> float:
    """Newman's degree assortativity: Pearson correlation of degrees across edge ends."""
    a = sp.coo_array(adjacency)
    k = np.asarray(adjacency.sum(axis=1)).ravel()
    return float(np.corrcoef(k[a.row], k[a.col])[0, 1])


def summarize(school: School) -> dict[str, float | int | str]:
    """One row of the descriptive table."""
    a = school.adjacency
    n, m = school.n_nodes, school.n_edges
    k = school.degrees.astype(np.float64)
    t = triangles(a)
    pairs = k * (k - 1) / 2
    clustering = np.divide(t, pairs, out=np.zeros_like(t), where=pairs > 0)
    _, labels = connected_components(a, directed=False)
    return {
        "school": school.name,
        "nodes": n,
        "edges": m,
        "mean_degree": 2 * m / n,
        "median_degree": float(np.median(k)),
        "density": 2 * m / (n * (n - 1)),
        "transitivity": float(t.sum() / pairs.sum()),
        "avg_clustering": float(clustering.mean()),
        "degree_assortativity": degree_assortativity(a),
        "giant_component_share": float(np.bincount(labels).max() / n),
    }
