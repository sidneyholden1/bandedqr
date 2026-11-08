"""Almost-banded factorisations: carried borders, collapse, and Woodbury."""
import numpy as np
import pytest

from bandedqr import (BorderedQR, CollapseBothQR, CollapseColsQR, CollapseQR,
                    RideQR, RideQRRect, RideTauQR, detect_recurrence,
                    detect_structure, factorize, solve, solve_with_tau_columns)
from conftest import (chebyshev_bc_rows, make_almost_banded, make_banded,
                      make_collapsible, relerr)

DTYPES = [False, True]


# --------------------------------------------------------------------- ride
@pytest.mark.parametrize("n,kl,ku,r", [(200, 2, 2, 2), (300, 1, 1, 3),
                                       (150, 4, 3, 1), (256, 2, 3, 5)])
@pytest.mark.parametrize("complex_", DTYPES)
def test_ride_matches_dense(rng, n, kl, ku, r, complex_):
    A = make_almost_banded(rng, n, kl, ku, r=r, complex_=complex_)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    x = RideQR(A, kl, ku, np.arange(r)).solve(b)
    assert relerr(A, x, b) < 1e-11
    assert np.allclose(x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("complex_", DTYPES)
def test_ride_multi_rhs(rng, complex_):
    n, kl, ku, r = 200, 2, 2, 3
    A = make_almost_banded(rng, n, kl, ku, r=r, complex_=complex_)
    B = rng.standard_normal((n, 5))
    if complex_:
        B = B + 1j * rng.standard_normal((n, 5))
    F = RideQR(A, kl, ku, np.arange(r))
    X = F.solve(B)
    assert X.shape == (n, 5)
    assert np.linalg.norm(A @ X - B) / np.linalg.norm(B) < 1e-11
    for t in range(5):
        assert np.allclose(X[:, t], F.solve(np.ascontiguousarray(B[:, t])),
                           rtol=1e-9, atol=1e-11)


def test_ride_permutes_bottom_rows_automatically(rng):
    """Dense rows near the bottom used to blow up kl; they are permuted instead."""
    n, kl, ku = 200, 2, 2
    A = make_banded(rng, n, kl, ku)
    A[n - 2, :] = rng.standard_normal(n)
    A[n - 1, :] = rng.standard_normal(n)
    b = rng.standard_normal(n)
    F = RideQR(A, kl, ku, [n - 2, n - 1])
    assert F.perm is not None
    assert F.kl <= kl + 2               # not widened to reach row n-1
    assert relerr(A, F.solve(b), b) < 1e-11


def test_ride_rejects_purely_banded(rng):
    A = make_banded(rng, 50, 2, 2)
    with pytest.raises(ValueError, match="at least one dense row"):
        RideQR(A, 2, 2, [])


# ------------------------------------------------------------ rectangular
@pytest.mark.parametrize("complex_", DTYPES)
def test_ride_rect_least_squares(rng, complex_):
    """Tall almost-banded: banded bulk on all n unknowns with BCs appended."""
    sp = pytest.importorskip("scipy.sparse")
    n, kl, ku, r = 120, 2, 2, 2
    bulk = np.zeros((n, n), dtype=np.complex128 if complex_ else np.float64)
    for i in range(n):
        lo, hi = max(0, i - kl), min(n - 1, i + ku)
        v = rng.standard_normal(hi - lo + 1)
        bulk[i, lo:hi + 1] = v + 1j * rng.standard_normal(hi - lo + 1) if complex_ else v
        bulk[i, i] += kl + ku + 3
    rows = rng.standard_normal((r, n))
    if complex_:
        rows = rows + 1j * rng.standard_normal((r, n))
    A = np.vstack([bulk, rows])                       # (n+r, n), tall
    b = rng.standard_normal(n + r)
    if complex_:
        b = b + 1j * rng.standard_normal(n + r)
    x = RideQRRect(sp.csr_matrix(A), dense_rows=[n, n + 1]).solve(b)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(x, x_ref, rtol=1e-7, atol=1e-9)


# ----------------------------------------------------------------- collapse
def test_detect_recurrence_on_chebyshev():
    n = 200
    k = np.arange(n)
    cases = {
        "T_k(0.3)": (np.cos(k * np.arccos(0.3)), 2),
        "T_k(1)": (np.ones(n), 1),
        "T_k(-1)": ((-1.0) ** k, 1),
        "T'_k(1) = k^2": (k.astype(float) ** 2, 3),
    }
    for name, (row, order) in cases.items():
        a = detect_recurrence(row)
        assert a is not None, name
        assert len(a) - 1 == order, f"{name}: got order {len(a)-1}, want {order}"
        # the stencil must actually annihilate the interior
        conv = np.convolve(row, a[::-1], mode="valid")
        assert np.abs(conv).max() < 1e-8 * np.abs(row).max(), name


def test_detect_recurrence_returns_none_for_random(rng):
    assert detect_recurrence(rng.standard_normal(200), max_order=8) is None


@pytest.mark.parametrize("points", [(1.0, -1.0), (1.0, -1.0, 0.3)])
def test_collapse_solves_with_chebyshev_bcs(rng, points):
    n, kl, ku = 200, 2, 2
    A, _ = make_collapsible(rng, n, kl, ku, points)
    b = rng.standard_normal(n)
    x = CollapseQR(A, kl, ku, np.arange(len(points))).solve(b)
    assert relerr(A, x, b) < 1e-9


def test_collapse_error_names_the_alternative(rng):
    """Random borders satisfy no recurrence; the message must say what to do."""
    A = make_almost_banded(rng, 100, 2, 2, r=2)
    with pytest.raises(RuntimeError, match="ride"):
        CollapseQR(A, 2, 2, [0, 1])


def test_collapse_preconditioner_is_recoverable(rng):
    n, kl, ku = 120, 2, 2
    A, _ = make_collapsible(rng, n, kl, ku)
    F = CollapseQR(A, kl, ku, [0, 1], store_P=True)
    G = CollapseQR(A, kl, ku, [0, 1], store_P=False)
    assert np.allclose(F.preconditioner().toarray(), G.preconditioner().toarray())
    y = rng.standard_normal(n)
    assert np.allclose(F.apply_P(y), F.preconditioner() @ y, atol=1e-10)


def test_collapse_columns(rng):
    """Transpose-dual: recurrence-structured dense *columns*."""
    n, kl, ku = 150, 2, 2
    A = make_banded(rng, n, kl, ku)
    cols = chebyshev_bc_rows(n, (1.0, -1.0))
    A[:, -2:] = cols.T
    b = rng.standard_normal(n)
    x = CollapseColsQR(A, kl, ku, [n - 2, n - 1]).solve(b)
    assert relerr(A, x, b) < 1e-8


def test_collapse_both(rng):
    n, kl, ku = 150, 2, 2
    A = make_banded(rng, n, kl, ku)
    A[:2, :] = chebyshev_bc_rows(n, (1.0, -1.0))
    A[:, -2:] = chebyshev_bc_rows(n, (1.0, -1.0)).T
    b = rng.standard_normal(n)
    x = CollapseBothQR(A, kl, ku, [0, 1], [n - 2, n - 1]).solve(b)
    assert relerr(A, x, b) < 1e-8


# ---------------------------------------------------------------- bordered
@pytest.mark.parametrize("n,kl,ku,r,c", [(200, 2, 2, 2, 2), (300, 1, 1, 3, 3),
                                         (150, 2, 3, 4, 1)])
@pytest.mark.parametrize("complex_", DTYPES)
def test_bordered_rows_and_cols(rng, n, kl, ku, r, c, complex_):
    A = make_almost_banded(rng, n, kl, ku, r=r, c=c, complex_=complex_)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    dr, dc = np.arange(r), np.arange(n - c, n)
    x = BorderedQR(A, kl, ku, dr, dc).solve(b)
    assert relerr(A, x, b) < 1e-10


def test_bordered_reuses_factor(rng):
    n, kl, ku = 200, 2, 2
    A = make_almost_banded(rng, n, kl, ku, r=2, c=2)
    F = BorderedQR(A, kl, ku, [0, 1], [n - 2, n - 1])
    B = rng.standard_normal((n, 4))
    X = F.solve(B)
    assert np.linalg.norm(A @ X - B) / np.linalg.norm(B) < 1e-10


def test_bordered_handles_more_dense_rows_than_kl(rng):
    """r > kl is where delete-the-border-and-Schur leaves a singular bulk."""
    n, kl, ku, r, c = 300, 1, 1, 4, 3
    A = make_almost_banded(rng, n, kl, ku, r=r, c=c)
    b = rng.standard_normal(n)
    x = BorderedQR(A, kl, ku, np.arange(r), np.arange(n - c, n)).solve(b)
    assert relerr(A, x, b) < 1e-10


@pytest.mark.parametrize("complex_", DTYPES)
def test_ride_tau(rng, complex_):
    """Dense rows plus genuinely low-rank (single lifted mode) tau columns."""
    n, kl, ku, r = 200, 2, 2, 2
    A = make_almost_banded(rng, n, kl, ku, r=r, complex_=complex_)
    dc = [n - 2, n - 1]
    for j in dc:                                  # sparse tau columns
        A[:, j] = 0
        A[j, j] = 1.0
        A[r + 3, j] = 2.0
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    x = RideTauQR(A, kl, ku, np.arange(r), dc).solve(b)
    assert relerr(A, x, b) < 1e-10


def test_solve_with_tau_columns(rng):
    """[N | F] with a tall banded bulk and dense tau columns."""
    m, c, kl, ku = 100, 2, 2, 2
    n = m - c
    N = np.zeros((m, n))
    for i in range(m):
        lo, hi = max(0, i - kl), min(n - 1, i + ku)
        if lo > hi:
            continue
        N[i, lo:hi + 1] = rng.standard_normal(hi - lo + 1)
        if i < n:
            N[i, i] += kl + ku + 3
    Fcols = rng.standard_normal((m, c))
    A = np.hstack([N, Fcols])
    b = rng.standard_normal(m)
    from bandedqr import dense_to_band
    x = solve_with_tau_columns(dense_to_band(N, kl, ku), kl, ku, Fcols, b)
    assert np.allclose(x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------ conditioning
def test_ride_beats_collapse_at_large_n(rng):
    """The documented reason `auto` prefers ride: collapse's preconditioner
    costs conditioning, and increasingly so with n."""
    kl, ku = 2, 2
    errs = {}
    for name in ("ride", "collapse"):
        e = []
        for n in (512, 4096):
            A, _ = make_collapsible(rng, n, kl, ku)
            b = rng.standard_normal(n)
            F = (RideQR(A, kl, ku, [0, 1]) if name == "ride"
                 else CollapseQR(A, kl, ku, [0, 1]))
            e.append(relerr(A, F.solve(b), b))
        errs[name] = e
    assert errs["ride"][-1] <= errs["collapse"][-1]
