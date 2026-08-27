"""Accuracy tooling: compensated residuals, iterative refinement, condition estimates.

Two things a pivot-free direct solver owes its users.

**How wrong is the answer?**  Without pivoting there is no growth factor to
inspect, so :func:`cond1_estimate` runs the Hager-Higham 1-norm estimator over
the factorisation -- a handful of extra solves -- to bound the error.

**Can it be made less wrong?**  :func:`iterative_refinement` reruns the solve on
the residual.  That only buys something if the residual is computed more
accurately than the factorisation was, which is why :func:`compensated_residual`
exists: it evaluates ``b - A x`` in double-double arithmetic (Dekker's exact
product and Knuth's exact sum), so the residual carries full relative accuracy
even when ``||A x||`` swamps ``||r||`` in plain float64.

That combination is what lets a *low*-precision factorisation reach
high-precision answers -- the GPU backend factors in fp32 and refines against
these residuals to land at float64 accuracy.
"""
import numpy as np

from ._band import band_matvec, band_norm1

__all__ = [
    "two_sum", "two_prod", "compensated_band_matvec", "compensated_residual",
    "iterative_refinement", "cond1_estimate", "condest", "RefinementResult",
]

_SPLIT = 134217729.0        # 2**27 + 1, Dekker's splitting constant for float64


def two_sum(a, b):
    """Exact sum: returns ``(s, e)`` with ``s = fl(a+b)`` and ``a + b == s + e``.

    Knuth's two-sum -- no assumption about the relative magnitudes of a and b.
    """
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def two_prod(a, b):
    """Exact product: returns ``(p, e)`` with ``p = fl(a*b)`` and ``a*b == p + e``.

    Dekker's algorithm via Veltkamp splitting.  Exact for float64 barring
    overflow in the split (magnitudes beyond ~1e300).
    """
    p = a * b
    ah = _SPLIT * a
    ah = ah - (ah - a)
    al = a - ah
    bh = _SPLIT * b
    bh = bh - (bh - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def compensated_band_matvec(ab, kl, ku, x, n=None):
    """``A x`` in double-double: returns ``(hi, lo)`` with ``A x == hi + lo``.

    Each product is split exactly and each accumulation tracked, so the pair
    carries roughly twice float64's precision.  Costs ~10x a plain banded
    mat-vec, which is negligible beside an ``O(n kl (kl+ku))`` factorisation.

    Real data only -- for complex input the real and imaginary parts are
    compensated separately by :func:`compensated_residual`.
    """
    ab = np.asarray(ab, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    m = ab.shape[0]
    n = m if n is None else n
    shape = (m,) + x.shape[1:]
    hi = np.zeros(shape, dtype=np.float64)
    lo = np.zeros(shape, dtype=np.float64)
    for off in range(-kl, ku + 1):
        col = off + kl
        i0, i1 = max(0, -off), min(m, n - off)
        if i1 <= i0:
            continue
        coef = ab[i0:i1, col]
        if x.ndim > 1:
            coef = coef.reshape((-1,) + (1,) * (x.ndim - 1))
        p, e = two_prod(coef, x[i0 + off:i1 + off])
        s, t = two_sum(hi[i0:i1], p)
        hi[i0:i1] = s
        lo[i0:i1] += t + e
    return hi, lo


def compensated_residual(ab, kl, ku, x, b, n=None):
    """``b - A x`` evaluated in double-double, returned rounded to float64.

    The result is accurate relative to ``||r||`` itself rather than to
    ``||A|| ||x||``, which is exactly what iterative refinement needs in order
    to make progress past the first step.
    """
    ab = np.asarray(ab)
    if np.iscomplexobj(ab) or np.iscomplexobj(x) or np.iscomplexobj(b):
        # (Ar + i Ai)(xr + i xi) = (Ar xr - Ai xi) + i (Ar xi + Ai xr), so
        #   re(r) = re(b) - Ar xr + Ai xi
        #   im(r) = im(b) - Ar xi - Ai xr
        # each product compensated separately, low parts folded in at the end
        ar = np.ascontiguousarray(np.real(ab))
        ai = np.ascontiguousarray(np.imag(ab))
        xr = np.ascontiguousarray(np.real(x))
        xi = np.ascontiguousarray(np.imag(x))
        hrr, lrr = compensated_band_matvec(ar, kl, ku, xr, n)
        hii, lii = compensated_band_matvec(ai, kl, ku, xi, n)
        hri, lri = compensated_band_matvec(ar, kl, ku, xi, n)
        hir, lir = compensated_band_matvec(ai, kl, ku, xr, n)

        s, e1 = two_sum(np.real(b).astype(np.float64), -hrr)
        s, e2 = two_sum(s, hii)
        re = s + ((e1 + e2) + (lii - lrr))

        t, f1 = two_sum(np.imag(b).astype(np.float64), -hri)
        t, f2 = two_sum(t, -hir)
        im = t + ((f1 + f2) - (lri + lir))
        return re + 1j * im
    hi, lo = compensated_band_matvec(ab, kl, ku, x, n)
    u, v = two_sum(np.asarray(b, dtype=np.float64), -hi)
    return u + (v - lo)


class RefinementResult:
    """Outcome of :func:`iterative_refinement`."""

    def __init__(self, x, iterations, residual_norms, converged):
        self.x = x
        self.iterations = iterations
        self.residual_norms = residual_norms
        self.converged = converged

    def __iter__(self):                 # so `x, info = ...` style unpacking works
        return iter((self.x, self))

    def __repr__(self):
        last = self.residual_norms[-1] if self.residual_norms else float("nan")
        return (f"<RefinementResult iters={self.iterations} "
                f"||r||={last:.3e} converged={self.converged}>")


def iterative_refinement(solve, residual, b, x0=None, maxiter=5, tol=1e-15,
                         stall_factor=0.5):
    """Refine ``x`` by repeatedly solving against the residual.

    Parameters
    ----------
    solve : callable
        ``r -> approximately A^{-1} r``.  May be a *low*-precision solver; that
        is the interesting case.
    residual : callable
        ``x -> b - A x``, evaluated more accurately than ``solve`` works.
    b : ndarray
        Right-hand side, used only for the initial norm scale.
    x0 : ndarray, optional
        Starting iterate.  Defaults to ``solve(b)``.
    maxiter : int
        Maximum correction steps.
    tol : float
        Stop once ``||r|| <= tol * ||b||``.
    stall_factor : float
        Stop early if a step fails to reduce ``||r||`` by at least this factor --
        refinement has reached the precision the residual can see.

    Returns
    -------
    RefinementResult
    """
    x = solve(b) if x0 is None else np.array(x0, copy=True)
    bnorm = float(np.linalg.norm(b))
    scale = bnorm if bnorm > 0 else 1.0
    norms = []
    converged = False
    for it in range(1, maxiter + 1):
        r = residual(x)
        rn = float(np.linalg.norm(r))
        norms.append(rn)
        if rn <= tol * scale:
            converged = True
            break
        if len(norms) >= 2 and rn > stall_factor * norms[-2]:
            break                       # no longer improving; stop burning solves
        x = x + solve(r)
    else:
        it = maxiter
    return RefinementResult(x, it, norms, converged)


def _sign(v):
    """``v / |v|`` with zeros mapped to 1 -- the complex generalisation of sign."""
    a = np.abs(v)
    out = np.ones_like(v)
    nz = a > 0
    out[nz] = v[nz] / a[nz]
    return out


def cond1_estimate(solve, solve_H, n, norm1_A, dtype=np.float64, maxiter=5):
    """Estimate ``kappa_1(A) = ||A||_1 ||A^{-1}||_1`` without forming ``A^{-1}``.

    The Hager-Higham 1-norm estimator (as in LAPACK's ``dlacn2``): it probes
    ``A^{-1}`` with a handful of solves, alternating against ``A^{-H}`` to pick
    the next probe direction.  Cheap -- a few solves against an existing
    factorisation -- and in practice within a small factor of the truth, but it
    *is* a lower bound and can be fooled.

    Parameters
    ----------
    solve : callable
        ``v -> A^{-1} v``.
    solve_H : callable
        ``v -> A^{-H} v``.
    n : int
        Dimension.
    norm1_A : float
        ``||A||_1``, computed directly from the operator.
    """
    x = np.full(n, 1.0 / n, dtype=dtype)
    est = 0.0
    jold = -1
    for _ in range(maxiter):
        y = np.asarray(solve(x))
        new_est = float(np.abs(y).sum())
        if new_est <= est:
            break
        est = new_est
        xi = _sign(y)
        z = np.asarray(solve_H(xi))
        j = int(np.argmax(np.abs(z)))
        if j == jold:
            break
        # Hager's optimality test: no coordinate direction beats the current one
        if np.abs(z[j]) <= float(np.real(np.vdot(x, z))):
            break
        jold = j
        x = np.zeros(n, dtype=dtype)
        x[j] = 1.0
    return float(norm1_A * est)


def condest(F, ab=None, kl=None, ku=None, maxiter=5):
    """Condition estimate for a :class:`~bandqr.banded.BandQR` factorisation.

    Uses the factorisation's own ``solve`` and ``solve_H``; only meaningful for
    square systems.
    """
    if ab is None:
        ab, kl, ku = F._ab, F.kl, F.ku
    norm1 = band_norm1(ab, kl, ku, n=F.n)
    return cond1_estimate(F.solve, F.solve_H, F.n, norm1,
                          dtype=F._dt, maxiter=maxiter)
