import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from fb100.io import ATTRIBUTES, list_schools, load_school


def _write_school(path, a, info):
    scipy.io.savemat(path, {"A": sp.csc_matrix(a), "local_info": np.asarray(info, np.uint16)})


def test_load_school_symmetrises_and_names_columns(tmp_path):
    a = np.zeros((3, 3))
    a[0, 1] = 1  # one-directional edge
    a[2, 2] = 1  # self-loop
    info = [[1, 2, 200, 0, 165, 2008, 3387], [1, 1, 201, 0, 166, 2007, 0], [2, 0, 0, 0, 0, 0, 0]]
    _write_school(tmp_path / "Toy1.mat", a, info)

    school = load_school(tmp_path / "Toy1.mat")

    assert school.name == "Toy1"
    assert school.n_edges == 1
    assert (school.adjacency != school.adjacency.T).nnz == 0
    assert school.adjacency.diagonal().sum() == 0
    assert tuple(school.attributes.columns) == ATTRIBUTES
    # Documented order: dorm is column 4, year column 5 (the original notebook's bug).
    assert school.attributes.loc[0, "dorm"] == 165
    assert school.attributes.loc[0, "year"] == 2008


def test_load_school_rejects_wrong_layout(tmp_path):
    _write_school(tmp_path / "Bad.mat", np.zeros((2, 2)), np.zeros((2, 6)))
    with pytest.raises(ValueError, match="attribute columns"):
        load_school(tmp_path / "Bad.mat")


def test_list_schools_skips_index_file(tmp_path):
    for name in ["schools", "B2", "A1"]:
        (tmp_path / f"{name}.mat").touch()
    assert [p.stem for p in list_schools(tmp_path)] == ["A1", "B2"]
