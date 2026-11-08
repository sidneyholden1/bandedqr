"""The `solve` / `factorize` front end: dispatch, overrides, and error messages."""
import numpy as np
import pytest

import bandedqr
from bandedqr import (METHODS, BandQR, BorderedQR, CollapseQR, LQ, RideQR,
                    RideTauQR, factorize, solve)
from conftest import make_almost_banded, make_banded, make_collapsible, relerr


@pytest.mark.parametrize("method", [m for m in METHODS if m != "auto"])
@pytest.mark.parametrize("complex_", [False, True])
def test_every_method_solves_a_banded_system(rng, method, complex_):
    """Whatever the table advertises, it must work -- the old docstring listed
    implemented methods as 'roadmap'."""
    n, kl, ku = 150, 3, 2
    A = make_banded(rng, n, kl, ku, complex_)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    x = solve(A, b, method=method)
    assert relerr(A, x, b) < 1e-11


@pytest.mark.parametrize("kl,ku,expect", [(2, 2, BandQR), (8, 8, BandQR)])
def test_auto_dispatch_pure_band(rng, kl, ku, expect):
    F = factorize(make_banded(rng, 200, kl, ku))
    assert isinstance(F, expect)
    assert F.householder == (kl >= 6)


def test_auto_dispatch_prefers_ride_for_borders(rng):
    """`auto` favours accuracy: ride carries the borders exactly, collapse
    trades conditioning for speed."""
    A = make_almost_banded(rng, 200, 2, 2, r=2)
    assert isinstance(factorize(A), RideQR)
    A = make_almost_banded(rng, 200, 2, 2, r=2, c=2)
    assert isinstance(factorize(A), RideTauQR)


def test_explicit_structure_overrides_detection(rng):
    A = make_almost_banded(rng, 200, 2, 2, r=2)
    b = rng.standard_normal(200)
    x = solve(A, b, method="ride", kl=2, ku=2, dense_rows=[0, 1], dense_cols=[])
    assert relerr(A, x, b) < 1e-11


def test_factorize_reuses_across_right_hand_sides(rng):
    n, kl, ku = 300, 2, 3
    A = make_banded(rng, n, kl, ku)
    F = factorize(A)
    for _ in range(3):
        b = rng.standard_normal(n)
        assert relerr(A, F.solve(b), b) < 1e-11


def test_factorize_matches_solve(rng):
    A = make_almost_banded(rng, 200, 2, 2, r=2)
    b = rng.standard_normal(200)
    assert np.allclose(factorize(A).solve(b), solve(A, b))


@pytest.mark.parametrize("complex_", [False, True])
def test_sparse_input(rng, complex_):
    sp = pytest.importorskip("scipy.sparse")
    A = make_almost_banded(rng, 400, 2, 2, r=2, complex_=complex_)
    b = rng.standard_normal(400) + (1j * rng.standard_normal(400) if complex_ else 0)
    x = solve(sp.csr_matrix(A), b)
    assert relerr(A, x, b) < 1e-10


def test_multi_rhs_through_front_end(rng):
    A = make_banded(rng, 200, 2, 2)
    B = rng.standard_normal((200, 4))
    X = solve(A, B)
    assert X.shape == (200, 4)
    assert np.linalg.norm(A @ X - B) / np.linalg.norm(B) < 1e-11


# ------------------------------------------------------------- error paths
def test_gpu_device_points_at_the_batched_api(rng):
    A = make_banded(rng, 50, 2, 2)
    with pytest.raises(ValueError, match="solve_batched"):
        solve(A, np.ones(50), device="gpu")


def test_unknown_method(rng):
    A = make_banded(rng, 50, 2, 2)
    with pytest.raises(ValueError, match="method must be one of"):
        solve(A, np.ones(50), method="magic")


def test_non_square_rejected(rng):
    with pytest.raises(ValueError, match="square"):
        solve(rng.standard_normal((10, 12)), np.ones(10))


def test_rhs_length_mismatch(rng):
    A = make_banded(rng, 50, 2, 2)
    with pytest.raises(ValueError, match="rows"):
        solve(A, np.ones(49))


def test_banded_method_on_bordered_matrix_names_the_alternative(rng):
    A = make_almost_banded(rng, 100, 2, 2, r=2)
    with pytest.raises(ValueError, match="ride"):
        solve(A, np.ones(100), method="givens")


def test_lu_refuses_almost_banded(rng):
    A = make_almost_banded(rng, 100, 2, 2, r=2)
    with pytest.raises(ValueError, match="ride|collapse"):
        solve(A, np.ones(100), method="lu")


def test_factorize_refuses_lu(rng):
    with pytest.raises(ValueError, match="lu"):
        factorize(make_banded(rng, 50, 2, 2), method="lu")


def test_ride_without_dense_rows_names_the_alternative(rng):
    A = make_banded(rng, 100, 2, 2)
    A[:, -1] = rng.standard_normal(100)          # tau column, no dense rows
    with pytest.raises(ValueError, match="bordered"):
        solve(A, np.ones(100), method="ride")


# ----------------------------------------------------------------- package
def test_public_api_is_importable():
    for name in bandedqr.__all__:
        assert hasattr(bandedqr, name), name


def test_version_is_a_string():
    assert isinstance(bandedqr.__version__, str)
    assert bandedqr.__version__.count(".") >= 1


def test_repr_of_factorisations(rng):
    A = make_banded(rng, 100, 2, 2)
    assert "BandQR" in repr(factorize(A))
    assert "RideQR" in repr(factorize(make_almost_banded(rng, 100, 2, 2, r=2)))
    A2, _ = make_collapsible(rng, 100, 2, 2)
    assert "CollapseQR" in repr(CollapseQR(A2, 2, 2, [0, 1]))
    assert "LQ" in repr(LQ(A, 2, 2))
