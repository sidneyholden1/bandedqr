"""Band storage conversions and banded arithmetic.

Storage convention throughout bandedqr (row-major, "row-aligned"):

    ab[i, j - i + kl] == A[i, j]      for max(0, i-kl) <= j <= min(n-1, i+ku)

so ``ab`` has shape ``(m, kl + ku + 1)`` and column ``kl`` is the main diagonal.
This differs from LAPACK/scipy's column-aligned layout (``ab[ku + i - j, j]``);
:func:`to_scipy_band` converts between them.

All conversions here are vectorised -- they touch each diagonal once with a
slice assignment rather than looping over elements in Python.
"""
import numpy as np

__all__ = [
    "dense_to_band", "to_band", "band_to_dense", "band_transpose",
    "band_matvec", "band_matvec_batched", "to_scipy_band", "from_scipy_band",
    "band_norm1", "band_norminf",
]


def _sparse(A):
    try:
        import scipy.sparse as sp
    except ImportError:
        return False
    return sp.issparse(A)


def dense_to_band(A, kl, ku):
    """Pack a dense ``(m, n)`` array into row-major band form ``(m, kl+ku+1)``.

    The dtype follows ``A``: real input stays float64, complex stays complex128.
    """
    A = np.asarray(A)
    dt = np.complex128 if np.issubdtype(A.dtype, np.complexfloating) else np.float64
    A = np.ascontiguousarray(A, dtype=dt)
    m, n = A.shape
    ab = np.zeros((m, kl + ku + 1), dtype=dt)
    for off in range(-kl, ku + 1):
        d = np.diagonal(A, offset=off)          # A[i, i+off] for valid i
        i0 = max(0, -off)                       # first row that diagonal touches
        ab[i0:i0 + d.size, off + kl] = d
    return ab


def to_band(A, kl, ku):
    """Band storage of ``A``'s in-band entries; ``A`` dense or ``scipy.sparse``.

    Out-of-band entries are dropped silently -- callers that care about them
    (the almost-banded factorisations) extract them separately.  Sparse input is
    never densified.
    """
    if _sparse(A):
        C = A.tocoo()
        dt = np.complex128 if np.issubdtype(C.data.dtype, np.complexfloating) else np.float64
        m, n = A.shape
        ab = np.zeros((m, kl + ku + 1), dtype=dt)
        keep = (C.col >= C.row - kl) & (C.col <= C.row + ku)
        # duplicate coordinates must accumulate, exactly as .todense() would
        np.add.at(ab, (C.row[keep], C.col[keep] - C.row[keep] + kl), C.data[keep])
        return ab
    return dense_to_band(A, kl, ku)


def band_to_dense(ab, kl, ku, n=None):
    """Inverse of :func:`dense_to_band` (mostly for tests and small diagnostics)."""
    ab = np.asarray(ab)
    m = ab.shape[0]
    n = m if n is None else n
    A = np.zeros((m, n), dtype=ab.dtype)
    for off in range(-kl, ku + 1):
        i0 = max(0, -off)
        i1 = min(m, n - off)
        if i1 > i0:
            A[np.arange(i0, i1), np.arange(i0, i1) + off] = ab[i0:i1, off + kl]
    return A


def band_transpose(ab, kl, ku):
    """Transpose in band storage: a ``(kl, ku)`` band becomes ``(ku, kl)``.

    ``abT[i, (j-i)+ku] == A[j, i]``.  O(n*(kl+ku)) with no fill.
    Returns ``(abT, ku, kl)``.
    """
    ab = np.asarray(ab)
    n = ab.shape[0]
    abT = np.zeros((n, kl + ku + 1), dtype=ab.dtype)
    for d in range(-ku, kl + 1):            # diagonal offset of A^T = -offset of A
        cT, cA = d + ku, kl - d
        if d >= 0:
            abT[:n - d, cT] = ab[d:, cA]
        else:
            abT[-d:, cT] = ab[:n + d, cA]
    return abT, ku, kl


def band_matvec(ab, kl, ku, x, n=None):
    """``y = A x`` for ``A`` in ``(kl, ku)`` band storage.  Vectorised per diagonal.

    ``x`` may be ``(n,)`` or ``(n, nrhs)``.
    """
    ab = np.asarray(ab)
    x = np.asarray(x)
    m = ab.shape[0]
    n = m if n is None else n
    out_shape = (m,) + x.shape[1:]
    y = np.zeros(out_shape, dtype=np.result_type(ab.dtype, x.dtype))
    for off in range(-kl, ku + 1):
        col = off + kl
        i0 = max(0, -off)                    # rows i with 0 <= i+off <= n-1
        i1 = min(m, n - off)
        if i1 <= i0:
            continue
        coef = ab[i0:i1, col]
        y[i0:i1] += coef.reshape((-1,) + (1,) * (x.ndim - 1)) * x[i0 + off:i1 + off]
    return y


def band_matvec_batched(ab, kl, ku, x):
    """Batched banded mat-vec: ``ab`` ``(n, kl+ku+1, batch)``, ``x`` ``(n, batch)``.

    Returns ``(n, batch)``.  This is the residual operator for the batched /
    GPU refinement path, evaluated on the host in full precision.
    """
    ab = np.asarray(ab)
    x = np.asarray(x)
    n, w, batch = ab.shape
    assert w == kl + ku + 1, "band width must be kl+ku+1"
    y = np.zeros((n, batch), dtype=np.result_type(ab.dtype, x.dtype))
    for off in range(-kl, ku + 1):
        col = off + kl
        i0, i1 = max(0, -off), min(n, n - off)
        if i1 > i0:
            y[i0:i1] += ab[i0:i1, col] * x[i0 + off:i1 + off]
    return y


def to_scipy_band(ab, kl, ku, n=None):
    """Row-aligned ``ab[i, j-i+kl]`` -> scipy's column-aligned ``absc[ku+i-j, j]``."""
    ab = np.asarray(ab)
    m = ab.shape[0]
    n = m if n is None else n
    absc = np.zeros((kl + ku + 1, n), dtype=ab.dtype)
    for off in range(-kl, ku + 1):
        col = off + kl
        i0, i1 = max(0, -off), min(m, n - off)
        if i1 > i0:
            absc[ku - off, i0 + off:i1 + off] = ab[i0:i1, col]
    return absc


def from_scipy_band(absc, kl, ku):
    """scipy's column-aligned layout -> the row-aligned layout used here."""
    absc = np.asarray(absc)
    n = absc.shape[1]
    ab = np.zeros((n, kl + ku + 1), dtype=absc.dtype)
    for off in range(-kl, ku + 1):
        col = off + kl
        i0, i1 = max(0, -off), min(n, n - off)
        if i1 > i0:
            ab[i0:i1, col] = absc[ku - off, i0 + off:i1 + off]
    return ab


def band_norm1(ab, kl, ku, n=None):
    """``||A||_1`` (max absolute column sum) straight from band storage."""
    ab = np.asarray(ab)
    m = ab.shape[0]
    n = m if n is None else n
    colsum = np.zeros(n, dtype=np.float64)
    for off in range(-kl, ku + 1):
        col = off + kl
        i0, i1 = max(0, -off), min(m, n - off)
        if i1 > i0:
            colsum[i0 + off:i1 + off] += np.abs(ab[i0:i1, col])
    return float(colsum.max()) if n else 0.0


def band_norminf(ab, kl, ku, n=None):
    """``||A||_inf`` (max absolute row sum) straight from band storage."""
    ab = np.asarray(ab)
    return float(np.abs(ab).sum(axis=1).max()) if ab.size else 0.0
