"""Structure detection -- what `solve` dispatches on."""
import numpy as np
import pytest

from bandedqr import Structure, detect_structure
from bandedqr.structure import StructureWarning
from conftest import make_almost_banded, make_banded


@pytest.mark.parametrize("kl,ku", [(0, 0), (2, 3), (5, 1), (1, 7)])
def test_pure_band(rng, kl, ku):
    st = detect_structure(make_banded(rng, 200, kl, ku))
    assert (st.kl, st.ku) == (kl, ku)
    assert len(st.dense_rows) == 0 and len(st.dense_cols) == 0


@pytest.mark.parametrize("r,c", [(1, 0), (2, 0), (0, 1), (0, 2), (3, 3), (6, 2)])
def test_dense_borders_are_found(rng, r, c):
    """Dense *columns* in particular: the previous per-row heuristic could not
    see them at all, and inflated ku to n instead."""
    n, kl, ku = 200, 2, 3
    A = make_almost_banded(rng, n, kl, ku, r=r, c=c)
    st = detect_structure(A)
    assert (st.kl, st.ku) == (kl, ku)
    assert sorted(st.dense_rows) == list(range(r))
    assert sorted(st.dense_cols) == sorted(range(n - c, n)) if c else True
    assert len(st.dense_cols) == c


@pytest.mark.parametrize("complex_", [False, True])
def test_sparse_and_complex(rng, complex_):
    sp = pytest.importorskip("scipy.sparse")
    A = make_almost_banded(rng, 400, 3, 2, r=2, c=2, complex_=complex_)
    st = detect_structure(sp.csr_matrix(A))
    assert (st.kl, st.ku) == (3, 2)
    assert len(st.dense_rows) == 2 and len(st.dense_cols) == 2


def test_unpacks_as_tuple(rng):
    kl, ku, dr, dc = detect_structure(make_almost_banded(rng, 100, 2, 2, r=2))
    assert (kl, ku) == (2, 2)
    assert list(dr) == [0, 1]
    assert list(dc) == []


def test_describe_and_repr(rng):
    st = detect_structure(make_almost_banded(rng, 100, 2, 2, r=2, c=1))
    text = st.describe()
    assert "band=(2,2)" in text and "dense_rows=2" in text and "dense_cols=1" in text
    assert "Structure" in repr(st)


def test_empty_matrix():
    st = detect_structure(np.zeros((10, 10)))
    assert (st.kl, st.ku) == (0, 0)
    assert st.nnz == 0


def test_identity():
    st = detect_structure(np.eye(50))
    assert (st.kl, st.ku) == (0, 0)


def test_dense_matrix_reports_full_band(rng):
    A = rng.standard_normal((40, 40))
    st = detect_structure(A)
    assert st.kl == 39 and st.ku == 39


def test_tolerance_ignores_tiny_entries(rng):
    A = make_banded(rng, 100, 2, 2)
    A[0, 99] = 1e-18                     # numerical dust, not structure
    st = detect_structure(A, tol=1e-12)
    assert (st.kl, st.ku) == (2, 2)
    assert len(st.dense_rows) == 0


def test_warns_when_not_cleanly_almost_banded(rng):
    """A matrix with many irregular wide rows is ambiguous; say so."""
    n = 200
    A = make_banded(rng, n, 2, 2)
    for i in range(0, n, 3):             # a third of rows reach much further
        A[i, min(n - 1, i + 40)] = 1.0
    with pytest.warns(StructureWarning):
        detect_structure(A, warn=True)


def test_max_dense_cap_is_reported(rng):
    A = make_almost_banded(rng, 200, 2, 2, r=10)
    st = detect_structure(A, max_dense=3)
    assert st.capped
    assert len(st.dense_rows) <= 3


def test_bands_wider_than_the_dense_count(rng):
    """A wide band must not be mistaken for a pile of dense rows."""
    A = make_banded(rng, 300, 20, 20)
    st = detect_structure(A)
    assert (st.kl, st.ku) == (20, 20)
    assert len(st.dense_rows) == 0 and len(st.dense_cols) == 0
