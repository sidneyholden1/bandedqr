"""Compensated residuals, iterative refinement, and condition estimation."""
import numpy as np
import pytest

from bandedqr import (BandQR, compensated_residual, cond1_estimate, condest,
                    dense_to_band, iterative_refinement, solve)
from bandedqr.refine import compensated_band_matvec, two_prod, two_sum
from conftest import make_banded, relerr


# ------------------------------------------------------- exact arithmetic
def test_two_sum_is_exact(rng):
    a = rng.standard_normal(1000) * 1e8
    b = rng.standard_normal(1000) * 1e-8
    s, e = two_sum(a, b)
    # s + e reproduces a + b exactly, which plain float64 addition cannot
    assert np.all(s + e == s + e)
    for i in range(0, 1000, 97):
        import fractions
        exact = fractions.Fraction(float(a[i])) + fractions.Fraction(float(b[i]))
        got = fractions.Fraction(float(s[i])) + fractions.Fraction(float(e[i]))
        assert exact == got


def test_two_prod_is_exact(rng):
    a = rng.standard_normal(200)
    b = rng.standard_normal(200)
    p, e = two_prod(a, b)
    import fractions
    for i in range(0, 200, 17):
        exact = fractions.Fraction(float(a[i])) * fractions.Fraction(float(b[i]))
        got = fractions.Fraction(float(p[i])) + fractions.Fraction(float(e[i]))
        assert exact == got


def test_compensated_matvec_agrees_with_plain(rng):
    A = make_banded(rng, 100, 2, 3)
    ab = dense_to_band(A, 2, 3)
    x = rng.standard_normal(100)
    hi, lo = compensated_band_matvec(ab, 2, 3, x)
    assert np.allclose(hi + lo, A @ x)


def _exact_residual(A, x, b):
    """``b - A x`` in exact rational arithmetic, as ground truth."""
    from fractions import Fraction
    n = A.shape[0]
    out = []
    for i in range(n):
        acc = Fraction(float(b[i]))
        for j in range(n):
            if A[i, j]:
                acc -= Fraction(float(A[i, j])) * Fraction(float(x[j]))
        out.append(acc)
    return out


def test_compensated_residual_beats_naive(rng):
    """The point of compensation: accuracy relative to ``||r||`` itself, not to
    ``||A|| ||x||``.  Measured against exact rational arithmetic.

    This needs the regime refinement actually operates in: ``x`` an approximate
    solution, so ``r`` is ~1e-16 of ``||A|| ||x||`` and the naive difference
    cancels away every significant digit.  With unrelated ``x`` and ``b`` there
    is nothing to cancel and both are equally good; with ``b`` *defined* as
    ``fl(A @ x)`` the naive residual is accidentally exactly zero.  Neither
    proves anything.
    """
    from fractions import Fraction
    n, kl, ku = 60, 1, 1
    A = make_banded(rng, n, kl, ku)
    ab = dense_to_band(A, kl, ku)
    b = rng.standard_normal(n)
    x = BandQR(ab, kl, ku).solve(b)          # same b: residual is at rounding level

    exact = _exact_residual(A, x, b)
    comp = compensated_residual(ab, kl, ku, x, b)
    naive = b - A @ x

    def err(approx):
        return max(abs(Fraction(float(approx[i])) - exact[i]) for i in range(n))

    scale = max(abs(e) for e in exact)       # ||r||, itself ~1e-16
    assert scale < 1e-14                     # confirm we are in that regime
    assert err(comp) < err(naive) / 1e3
    assert float(err(comp) / scale) < 1e-12  # compensated keeps r's own digits


@pytest.mark.parametrize("complex_", [False, True])
def test_compensated_residual_matches_plain_when_no_cancellation(rng, complex_):
    n, kl, ku = 200, 2, 2
    A = make_banded(rng, n, kl, ku, complex_)
    ab = dense_to_band(A, kl, ku)
    x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_ else 0)
    r = compensated_residual(ab, kl, ku, x, b)
    assert np.allclose(r, b - A @ x, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------- refinement
def _ill_conditioned(n, cond):
    """A tridiagonal matrix with condition number ~``cond``.

    A graded diagonal, not a single small entry: one small pivot in a (1,4,1)
    tridiagonal leaves the matrix perfectly well conditioned, because the
    off-diagonals still carry the row.
    """
    ab = np.zeros((n, 3))
    ab[:, 1] = np.logspace(0, -np.log10(cond), n)      # graded diagonal
    ab[:, 0] = ab[:, 2] = 1e-4 * ab[:, 1]              # weak coupling
    return ab


@pytest.mark.parametrize("cond", [1e6, 1e10])
def test_refinement_reduces_residual(rng, cond):
    n = 400
    ab = _ill_conditioned(n, cond)
    from bandedqr import band_to_dense
    A = band_to_dense(ab, 1, 1, n)
    b = rng.standard_normal(n)
    F = BandQR(ab, 1, 1)
    x0 = F.solve(b)
    res = iterative_refinement(
        F.solve, lambda v: compensated_residual(ab, 1, 1, v, b), b,
        x0=x0, maxiter=4)
    assert np.linalg.norm(A @ res.x - b) <= np.linalg.norm(A @ x0 - b) * 1.01
    assert len(res.residual_norms) >= 1


def test_solve_refine_flag_runs(rng):
    n = 300
    ab = _ill_conditioned(n, 1e8)
    from bandedqr import band_to_dense
    A = band_to_dense(ab, 1, 1, n)
    b = rng.standard_normal(n)
    x_plain = solve(A, b, method="givens")
    x_ref = solve(A, b, method="givens", refine=3)
    assert relerr(A, x_ref, b) <= relerr(A, x_plain, b) * 1.01


def test_refinement_stops_when_converged(rng):
    A = make_banded(rng, 200, 2, 2)
    ab = dense_to_band(A, 2, 2)
    b = rng.standard_normal(200)
    F = BandQR(ab, 2, 2)
    res = iterative_refinement(
        F.solve, lambda v: compensated_residual(ab, 2, 2, v, b), b, maxiter=10)
    assert res.iterations < 10          # must not burn all ten on a good system


def test_refine_rejected_for_almost_banded(rng):
    from conftest import make_almost_banded
    A = make_almost_banded(rng, 100, 2, 2, r=2)
    with pytest.raises(ValueError, match="purely banded"):
        solve(A, np.ones(100), refine=2)


# ---------------------------------------------------- condition estimation
@pytest.mark.parametrize("n,kl,ku", [(100, 2, 2), (200, 3, 1)])
def test_condest_is_within_a_factor_of_truth(rng, n, kl, ku):
    A = make_banded(rng, n, kl, ku)
    F = BandQR(dense_to_band(A, kl, ku), kl, ku)
    est = condest(F)
    true = np.linalg.cond(A, 1)
    assert est <= true * 1.001            # it is a lower bound
    assert est >= true / 10               # and should be a tight-ish one


def test_condest_tracks_ill_conditioning(rng):
    from bandedqr import band_to_dense
    ests, trues = [], []
    for cond in (1e2, 1e6, 1e10):
        ab = _ill_conditioned(300, cond)
        ests.append(condest(BandQR(ab, 1, 1)))
        trues.append(np.linalg.cond(band_to_dense(ab, 1, 1, 300), 1))
    assert ests[0] < ests[1] < ests[2]
    assert ests[2] / ests[0] > 1e6
    for est, true in zip(ests, trues):          # a tight-ish lower bound
        assert est <= true * 1.001
        assert est >= true / 100


def test_solve_H_is_the_conjugate_transpose_solve(rng):
    for complex_ in (False, True):
        A = make_banded(rng, 60, 2, 2, complex_)
        F = BandQR(dense_to_band(A, 2, 2), 2, 2)
        v = rng.standard_normal(60) + (1j * rng.standard_normal(60) if complex_ else 0)
        assert np.allclose(F.solve_H(v), np.linalg.solve(A.conj().T, v),
                           rtol=1e-8, atol=1e-10)
