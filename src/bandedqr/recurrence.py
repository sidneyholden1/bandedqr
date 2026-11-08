"""Reading the linear recurrence a dense border satisfies, from its values alone.

A boundary-condition row of a spectral operator is a sequence of basis
evaluations -- ``T_k(x0)``, ``T_k(1)``, ``T'_k(1)``, and so on.  Every such
sequence is a sum of terms (polynomial in k) x (geometric r^k), and every
sequence of that form obeys a short constant-coefficient linear recurrence

    a_0 v[k] + a_1 v[k+1] + ... + a_p v[k+p] = 0

for all interior k.  The coefficients show up as a null vector of the Hankel
matrix of sliding windows, which means they can be *measured* -- no analytic
knowledge of the basis is needed.  Applying the corresponding banded
Toeplitz operator on the right annihilates the row's interior, collapsing a
dense border into the band so the whole solve becomes one banded QR.

This is the Prony / Berlekamp-Massey idea run numerically, with an SVD in place
of exact arithmetic so that it tolerates the rounding in real operator rows.
"""
import numpy as np

from . import _native

__all__ = ["detect_recurrence", "recurrence_residual", "build_preconditioner",
           "apply_preconditioner"]


def detect_recurrence(rows, max_order=20, tol=1e-8, trim=0):
    """Find the shortest stencil ``a = [a_0 ... a_p]`` annihilating every row.

    Searches increasing orders and returns the first whose Hankel matrix is
    numerically rank deficient, i.e. the *shortest* recurrence consistent with
    the data -- longer ones always exist and are useless (they widen the band
    the preconditioner produces).

    Parameters
    ----------
    rows : array_like
        One row, or a 2-D stack of rows that should share a single annihilator.
    max_order : int
        Longest stencil to look for.
    tol : float
        Relative singular-value threshold for "this window matrix is singular".
    trim : int
        Drop this many entries from each end before fitting.  Useful when the
        first/last few entries of a row are special-cased and do not follow the
        interior recurrence.

    Returns
    -------
    ndarray or None
        The stencil, largest coefficient normalised to 1, or None if no
        recurrence of order <= ``max_order`` fits.
    """
    dt = np.complex128 if np.iscomplexobj(rows) else np.float64
    R = np.atleast_2d(np.asarray(rows, dtype=dt))
    if trim:
        R = R[:, trim:R.shape[1] - trim]
    nwin = R.shape[1]
    for p in range(1, max_order + 1):
        if nwin - p < p + 1:
            break
        blocks = [np.lib.stride_tricks.sliding_window_view(r, p + 1) for r in R]
        H = np.vstack(blocks)
        s = np.linalg.svd(H, compute_uv=False)
        if s[-1] <= tol * s[0]:
            Vt = np.linalg.svd(H, full_matrices=False)[2]
            a = Vt[-1]
            return a / a[np.argmax(np.abs(a))]
    return None


def recurrence_residual(rows, a):
    """Relative residual ``||H a|| / ||H||`` -- how well the stencil annihilates.

    Small means the recurrence really does hold; use it to sanity-check a
    detected stencil before trusting a collapse.
    """
    a = np.asarray(a)
    R = np.atleast_2d(np.asarray(rows))
    p = len(a) - 1
    blocks = [np.lib.stride_tricks.sliding_window_view(r, p + 1) for r in R]
    H = np.vstack(blocks)
    denom = np.linalg.norm(H) * np.linalg.norm(a)
    return float(np.linalg.norm(H @ a) / denom) if denom else 0.0


def build_preconditioner(a, n):
    """Explicit sparse right-preconditioner ``P`` with ``P[j-p+i, j] = a[i]``.

    ``M @ P`` is banded and ``x = P @ y``.  Built as diagonals -- no element
    loop.  Requires scipy.
    """
    import scipy.sparse as sp
    a = np.asarray(a)
    p = len(a) - 1
    offs = [p - i for i in range(p + 1)]
    diags = [a[i] * np.ones(n - (p - i), dtype=a.dtype) for i in range(p + 1)]
    return sp.diags(diags, offs, shape=(n, n), format="csc")


def apply_preconditioner(a, y):
    """``x = P y`` via the recurrence itself (in C), without forming ``P``.

    Real or complex, following the dtype of ``a``/``y``.
    """
    cplx = np.iscomplexobj(a) or np.iscomplexobj(y)
    dt = np.complex128 if cplx else np.float64
    a = np.ascontiguousarray(a, dtype=dt)
    y = np.ascontiguousarray(y, dtype=dt)
    p = len(a) - 1
    n = y.shape[0]
    pf = _native.ptr_for(cplx)
    cfn = getattr(_native.lib, "bandedqr_apply_P" + _native.suffix_for(cplx))
    ap = pf(a)
    if y.ndim == 1:
        x = np.zeros(n, dtype=dt)
        cfn(ap, p, pf(y), n, pf(x))
        return x
    X = np.zeros_like(y)
    for k in range(y.shape[1]):
        yk = np.ascontiguousarray(y[:, k])
        xk = np.zeros(n, dtype=dt)
        cfn(ap, p, pf(yk), n, pf(xk))
        X[:, k] = xk
    return X
