"""Purely banded factorisations: BandQR (Givens and Householder) and LQ."""
import numpy as np
import pytest

from bandqr import (LQ, BandQR, band_to_dense, dense_to_band, has_threads,
                    solve, to_band)
from bandqr._band import band_matvec, band_norm1, band_transpose, to_scipy_band
from conftest import make_band_storage, make_banded, relerr

DTYPES = [False, True]      # complex_?


# --------------------------------------------------------------------- solve
@pytest.mark.parametrize("n,kl,ku", [(200, 2, 2), (301, 1, 4), (500, 5, 3),
                                     (64, 0, 0), (128, 7, 0), (128, 0, 7)])
@pytest.mark.parametrize("complex_", DTYPES)
def test_solve_matches_dense(rng, n, kl, ku, complex_):
    A = make_banded(rng, n, kl, ku, complex_)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    x = BandQR(dense_to_band(A, kl, ku), kl, ku).solve(b)
    assert relerr(A, x, b) < 1e-12
    assert np.allclose(x, np.linalg.solve(A, b))


@pytest.mark.parametrize("complex_", DTYPES)
def test_dtype_follows_input(rng, complex_):
    A = make_banded(rng, 80, 2, 2, complex_)
    b = rng.standard_normal(80) + (1j * rng.standard_normal(80) if complex_ else 0)
    x = solve(A, b)
    assert x.dtype == (np.complex128 if complex_ else np.float64)


@pytest.mark.parametrize("complex_", DTYPES)
@pytest.mark.parametrize("householder", [False, True])
def test_multi_rhs(rng, complex_, householder):
    n, kl, ku, nrhs = 150, 3, 2, 6
    A = make_banded(rng, n, kl, ku, complex_)
    B = rng.standard_normal((n, nrhs))
    if complex_:
        B = B + 1j * rng.standard_normal((n, nrhs))
    F = BandQR(dense_to_band(A, kl, ku), kl, ku, householder=householder)
    X = F.solve(B)
    assert X.shape == (n, nrhs)
    assert np.linalg.norm(A @ X - B) / np.linalg.norm(B) < 1e-12
    # each column must match the single-RHS path
    for t in range(nrhs):
        assert np.allclose(X[:, t], F.solve(np.ascontiguousarray(B[:, t])))


@pytest.mark.parametrize("complex_", DTYPES)
def test_householder_agrees_with_givens(rng, complex_):
    A = make_banded(rng, 200, 6, 5, complex_)
    b = rng.standard_normal(200) + (1j * rng.standard_normal(200) if complex_ else 0)
    ab = dense_to_band(A, 6, 5)
    xg = BandQR(ab, 6, 5).solve(b)
    xh = BandQR(ab, 6, 5, householder=True).solve(b)
    assert np.allclose(xg, xh, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------ threading
@pytest.mark.skipif(not has_threads(), reason="built with BANDQR_NO_THREADS")
@pytest.mark.parametrize("nthreads", [2, 4, 8])
def test_threaded_R_is_bit_identical(rng, nthreads):
    """The wavefront must not change the answer -- the factorisation is
    deterministic, and that is a headline property, so assert exact equality."""
    ab = make_band_storage(rng, 4000, 8, 8)
    R1 = BandQR(ab, 8, 8, nthreads=1).R_band()
    Rn = BandQR(ab, 8, 8, nthreads=nthreads).R_band()
    assert np.array_equal(R1, Rn)


def test_auto_threads_is_serial_for_narrow_bands():
    from bandqr.banded import auto_threads
    assert auto_threads(100000, 2, 2) == 1
    assert auto_threads(100000, 32, 32) == 1
    if has_threads():
        assert auto_threads(20000, 96, 96) > 1


# ------------------------------------------------------------- least squares
@pytest.mark.parametrize("complex_", DTYPES)
def test_tall_least_squares(rng, complex_):
    """m > n: solve must minimise ||Ax-b|| and report the residual norm."""
    m, n, kl, ku = 120, 100, 3, 2
    A = np.zeros((m, n), dtype=np.complex128 if complex_ else np.float64)
    for i in range(m):
        lo, hi = max(0, i - kl), min(n - 1, i + ku)
        if lo > hi:
            continue
        v = rng.standard_normal(hi - lo + 1)
        A[i, lo:hi + 1] = v + 1j * rng.standard_normal(hi - lo + 1) if complex_ else v
        if i < n:
            A[i, i] += kl + ku + 3
    b = rng.standard_normal(m) + (1j * rng.standard_normal(m) if complex_ else 0)
    F = BandQR(dense_to_band(A, kl, ku), kl, ku, ncols=n)
    x = F.solve(b)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(x, x_ref, rtol=1e-8, atol=1e-10)
    assert np.isclose(F.residual, np.linalg.norm(A @ x - b), rtol=1e-8)


# ------------------------------------------------------------------ Q and R
@pytest.mark.parametrize("complex_", DTYPES)
@pytest.mark.parametrize("householder", [False, True])
def test_Q_is_unitary_and_reconstructs_A(rng, complex_, householder):
    n, kl, ku = 40, 2, 3
    A = make_banded(rng, n, kl, ku, complex_)
    F = BandQR(dense_to_band(A, kl, ku), kl, ku, householder=householder)
    Q = F.Q_dense(thin=False)
    assert np.allclose(Q.conj().T @ Q, np.eye(n), atol=1e-10)
    R = np.triu(band_to_dense(F.R_band(), 0, F.kuR, n))
    assert np.allclose(Q @ R, A, atol=1e-10)


@pytest.mark.parametrize("complex_", DTYPES)
def test_apply_Q_inverts_apply_Qt(rng, complex_):
    n, kl, ku = 60, 2, 2
    A = make_banded(rng, n, kl, ku, complex_)
    F = BandQR(dense_to_band(A, kl, ku), kl, ku)
    v = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    assert np.allclose(F.apply_Q(F.apply_Qt(v)), v, atol=1e-12)
    assert np.allclose(F.apply_Qt(F.apply_Q(v)), v, atol=1e-12)


@pytest.mark.parametrize("complex_", DTYPES)
def test_solve_Rt_is_the_conjugate_transpose_solve(rng, complex_):
    n, kl, ku = 50, 2, 2
    A = make_banded(rng, n, kl, ku, complex_)
    F = BandQR(dense_to_band(A, kl, ku), kl, ku)
    R = np.triu(band_to_dense(F.R_band(), 0, F.kuR, n))
    y = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    assert np.allclose(F.solve_Rt(y), np.linalg.solve(R.conj().T, y), atol=1e-9)
    assert np.allclose(F.solve_R(y), np.linalg.solve(R, y), atol=1e-9)


def test_R_sparse_matches_R_band(rng):
    A = make_banded(rng, 40, 2, 2)
    F = BandQR(dense_to_band(A, 2, 2), 2, 2)
    dense = np.triu(band_to_dense(F.R_band(), 0, F.kuR, 40))
    assert np.allclose(F.R_sparse().toarray(), dense)


# --------------------------------------------------------------------- LQ
@pytest.mark.parametrize("complex_", DTYPES)
def test_lq_solve(rng, complex_):
    n, kl, ku = 120, 3, 2
    A = make_banded(rng, n, kl, ku, complex_)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    assert relerr(A, LQ(A, kl, ku).solve(b), b) < 1e-11


# ------------------------------------------------------------ band utilities
@pytest.mark.parametrize("kl,ku", [(0, 0), (1, 3), (4, 2)])
@pytest.mark.parametrize("complex_", DTYPES)
def test_band_roundtrip(rng, kl, ku, complex_):
    A = make_banded(rng, 30, kl, ku, complex_)
    ab = dense_to_band(A, kl, ku)
    assert np.array_equal(band_to_dense(ab, kl, ku), A)
    assert np.array_equal(to_band(A, kl, ku), ab)


def test_band_transpose(rng):
    A = make_banded(rng, 25, 3, 1)
    abT, klT, kuT = band_transpose(dense_to_band(A, 3, 1), 3, 1)
    assert (klT, kuT) == (1, 3)
    assert np.array_equal(band_to_dense(abT, klT, kuT), A.T)


def test_band_matvec_and_norms(rng):
    A = make_banded(rng, 40, 2, 3)
    ab = dense_to_band(A, 2, 3)
    x = rng.standard_normal(40)
    assert np.allclose(band_matvec(ab, 2, 3, x), A @ x)
    assert np.isclose(band_norm1(ab, 2, 3), np.abs(A).sum(axis=0).max())
    X = rng.standard_normal((40, 3))
    assert np.allclose(band_matvec(ab, 2, 3, X), A @ X)


def test_scipy_band_roundtrip(rng):
    from bandqr import from_scipy_band
    # band storage from a real matrix, so the corner slots (which index columns
    # outside [0, n) and are "don't care") are zero and survive the round trip
    ab = dense_to_band(make_banded(rng, 30, 2, 3), 2, 3)
    assert np.allclose(from_scipy_band(to_scipy_band(ab, 2, 3), 2, 3), ab)


def test_corner_slots_are_ignored(rng):
    """Out-of-range corner slots of band storage must not affect the answer."""
    A = make_banded(rng, 40, 2, 3)
    clean = dense_to_band(A, 2, 3)
    dirty = clean.copy()
    dirty[0, 0] = dirty[0, 1] = 1e6          # columns -2, -1
    dirty[-1, -1] = dirty[-1, -2] = -1e6     # columns n, n+1
    b = rng.standard_normal(40)
    assert np.allclose(BandQR(clean, 2, 3).solve(b), BandQR(dirty, 2, 3).solve(b))


def test_sparse_input_not_densified(rng):
    sp = pytest.importorskip("scipy.sparse")
    A = make_banded(rng, 400, 2, 2)
    As = sp.csr_matrix(A)
    assert np.allclose(to_band(As, 2, 2), to_band(A, 2, 2))


# ------------------------------------------------------------------- errors
def test_bad_band_width_rejected(rng):
    ab = make_band_storage(rng, 20, 2, 2)
    with pytest.raises(ValueError, match="band width"):
        BandQR(ab, 3, 3)


def test_rhs_shape_mismatch_rejected(rng):
    ab = make_band_storage(rng, 20, 1, 1)
    with pytest.raises(ValueError, match="rows"):
        BandQR(ab, 1, 1).solve(np.zeros(19))


def test_context_manager_frees(rng):
    ab = make_band_storage(rng, 20, 1, 1)
    with BandQR(ab, 1, 1) as F:
        assert F.solve(np.ones(20)) is not None
    assert F._h is None


def test_rank_deficiency_is_detectable():
    """No pivoting means rank deficiency cannot be caught during the
    factorisation -- but it must be visible afterwards in diag(R)."""
    n = 30
    ab = np.zeros((n, 3))
    ab[:, 1] = 1.0
    ab[5, 1] = 0.0                     # a zero on the diagonal
    F = BandQR(ab, 1, 1)
    assert F.rank_deficient()
    ab[5, 1] = 1.0
    assert not BandQR(ab, 1, 1).rank_deficient()
