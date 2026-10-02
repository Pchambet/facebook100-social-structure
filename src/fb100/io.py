"""Load Facebook100 `.mat` files into a sparse adjacency matrix and a typed attribute table.

The column order of `local_info` is documented in the dataset readme
(`facebook100_readme_021011.txt`): status flag, gender, major, second major/minor,
dorm/house, year, high school. Missing values are coded 0. Getting this order right
matters: the first version of this project read column 4 as "year" and column 5 as
"high school", which silently turned every dorm result into a year result (see README,
"Erratum").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

ATTRIBUTES: tuple[str, ...] = (
    "status",
    "gender",
    "major",
    "minor",
    "dorm",
    "year",
    "high_school",
)
MISSING = 0


@dataclass(frozen=True)
class School:
    """One campus network: a symmetric binary adjacency matrix plus node attributes."""

    name: str
    adjacency: sp.csr_array
    attributes: pd.DataFrame

    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    @property
    def n_edges(self) -> int:
        return self.adjacency.nnz // 2

    @property
    def degrees(self) -> np.ndarray:
        return np.asarray(self.adjacency.sum(axis=1)).ravel().astype(np.int64)


def clean_adjacency(matrix: sp.sparray | sp.spmatrix | np.ndarray) -> sp.csr_array:
    """Return a symmetric 0/1 CSR array with an empty diagonal.

    The raw files are already symmetric and loop-free, but enforcing it here keeps every
    downstream formula (degree = row sum, edges = nnz / 2) valid by construction.
    """
    a = sp.csr_array(matrix, dtype=np.float64)
    a = ((a + a.T) > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    return sp.csr_array(a)


def load_school(path: str | Path) -> School:
    """Read one Facebook100 school file."""
    path = Path(path)
    mat = scipy.io.loadmat(path, spmatrix=False)
    if "A" not in mat or "local_info" not in mat:
        raise ValueError(f"{path.name} is not a Facebook100 school file")
    info = np.asarray(mat["local_info"], dtype=np.int64)
    if info.shape[1] != len(ATTRIBUTES):
        raise ValueError(f"{path.name}: expected {len(ATTRIBUTES)} attribute columns")
    adjacency = clean_adjacency(mat["A"])
    if adjacency.shape[0] != info.shape[0]:
        raise ValueError(f"{path.name}: adjacency and attribute rows disagree")
    return School(path.stem, adjacency, pd.DataFrame(info, columns=list(ATTRIBUTES)))


def list_schools(data_dir: str | Path) -> list[Path]:
    """All school files in `data_dir`, sorted by name (`schools.mat` is an index, not a school)."""
    files = sorted(Path(data_dir).glob("*.mat"))
    return [f for f in files if f.stem != "schools"]
