"""Purely banded factorisations: ``BandQR`` (Givens or Householder) and ``LQ``."""
import os

import numpy as np

from . import _native
from ._band import (band_matvec, band_norm1, band_transpose, to_band,
                    to_scipy_band)

__all__ = ["BandQR", "LQ", "auto_threads"]


# Wavefront threading only pays on bands wide enough for the factorisation to be
# compute- rather than memory-bound.  Narrow-band QR streams the band from DRAM
# and is limited by that, not by arithmetic, so extra threads buy nothing and
# the cross-column synchronisation shows up as pure overhead.  Measured on an
# M-series part (bench/bench_threads.py), 8 threads against serial:
#
#     kl=16   1.00x        kl=64    1.67x
#     kl=32   1.00x        kl=96    2.28x
#                          kl=128   2.78x
#
# so auto stays serial below kl=64.  Override with BANDQR_THREAD_MIN_KL.
_THREAD_MIN_KL = int(os.environ.get("BANDQR_THREAD_MIN_KL", "64"))
_THREAD_MIN_WORK = float(os.environ.get("BANDQR_THREAD_MIN_WORK", "5e7"))
_THREAD_MAX = int(os.environ.get("BANDQR_THREAD_MAX", "8"))


def auto_threads(n, kl, ku, requested=0):
    """Resolve ``nthreads=0`` (auto) into a concrete thread count.

    Auto means *serial unless threading has been measured to win*: narrow-band
    factorisation is bound by memory traffic over the band, which extra threads
    do not relieve, and the wavefront's cross-column synchronisation then shows
    up as pure overhead.
    """
    if requested and requested != 0:
        return max(1, int(requested))
    if not _native.has_threads():
        return 1
    work = float(n) * max(kl, 1) * (kl + ku + 1)
    if kl < _THREAD_MIN_KL or work < _THREAD_MIN_WORK:
        return 1
    return max(1, min(_THREAD_MAX, os.cpu_count() or 1))


class BandQR:
    """``A = Q R`` for a banded ``A``, with ``Q`` held implicitly.

    ``A`` is ``m x n`` with ``m >= n``; ``R`` is upper triangular with upper
    bandwidth ``ku + kl``.  For ``m > n`` this is a least-squares
    factorisation and :meth:`solve` returns the minimiser of ``||A x - b||``,
    with the residual norm left on ``.residual``.

    Parameters
    ----------
    ab : ndarray ``(m, kl+ku+1)``
        Band storage; see :mod:`bandqr._band`.  dtype selects the real or
        complex kernel.
    householder : bool
        Use one Householder reflector per column instead of Givens rotations.
        Roughly 25% faster to factor for wide bands; Givens wins on narrow ones
        and is the default.
    nthreads : int
        ``0`` auto (see :func:`auto_threads`), ``1`` serial, ``k`` use k threads
        for the factorisation.  Givens only.  The factorisation is deterministic
        either way: threaded ``R`` is bit-identical to serial ``R``.
    """

    def __init__(self, ab, kl, ku, ncols=None, nthreads=0, householder=False):
        ab = np.ascontiguousarray(ab)
        self._z = _native.is_complex(ab)
        self._dt = np.complex128 if self._z else np.float64
        self._suf = _native.suffix_for(self._z)
        self._p = _native.ptr_for(self._z)
        ab = np.ascontiguousarray(ab, dtype=self._dt)
        m, w = ab.shape
        if w != kl + ku + 1:
            raise ValueError(f"band width must be kl+ku+1 = {kl+ku+1}, got {w}")
        if kl < 0 or ku < 0:
            raise ValueError(f"need kl, ku >= 0, got kl={kl}, ku={ku}")
        self.m, self.kl, self.ku = m, kl, ku
        self.n = m if ncols is None else int(ncols)
        if self.n > m:
            raise ValueError(f"need m >= ncols, got m={m}, ncols={self.n}")
        self.householder = bool(householder)
        self._ab = ab                      # kept: residuals and norms need A
        self.nthreads = 1 if householder else auto_threads(self.n, kl, ku, nthreads)

        if householder:
            self._h = self._fn("factor_hh")(self._p(ab), m, self.n, kl, ku)
        else:
            self._h = self._fn("factor_mt")(self._p(ab), m, self.n, kl, ku,
                                            self.nthreads)
        if not self._h:
            raise RuntimeError("bandqr factorisation failed (need m >= n, kl,ku >= 0)")
        self.kuR = ku + kl
        self.residual = None

    # ---- plumbing --------------------------------------------------------
    def _fn(self, name):
        return getattr(_native.lib, "bandqr_" + name + self._suf)

    def _vec_op(self, cname, v):
        """Apply an in-place length-m C vector routine to a vector or a block."""
        v = np.array(v, dtype=self._dt, copy=True, order="C")
        fn = self._fn(cname)
        if v.ndim == 1:
            fn(self._h, self._p(v))
            return v
        for k in range(v.shape[1]):
            col = np.ascontiguousarray(v[:, k])
            fn(self._h, self._p(col))
            v[:, k] = col
        return v

    def _tri_op(self, cname, y, nout):
        y = np.ascontiguousarray(y, dtype=self._dt)
        fn = self._fn(cname)
        if y.ndim == 1:
            x = np.zeros(nout, dtype=self._dt)
            fn(self._h, self._p(y), self._p(x))
            return x
        X = np.zeros((nout, y.shape[1]), dtype=self._dt)
        for k in range(y.shape[1]):
            yk = np.ascontiguousarray(y[:, k])
            xk = np.zeros(nout, dtype=self._dt)
            fn(self._h, self._p(yk), self._p(xk))
            X[:, k] = xk
        return X

    # ---- orthogonal factor ----------------------------------------------
    def apply_Qt(self, v):
        """``Q^H v`` (``Q^T`` for real data).  ``v`` is ``(m,)`` or ``(m, k)``."""
        return self._vec_op("apply_Qt", v)

    def apply_Q(self, v):
        """``Q v`` -- the inverse of :meth:`apply_Qt`."""
        return self._vec_op("apply_Q", v)

    def Q_dense(self, thin=True):
        """Form ``Q`` explicitly: ``(m, n)`` if ``thin`` else ``(m, m)``.

        ``O(m * cols * nrot)`` -- for tests and small problems.  The whole point
        of the implicit representation is that you rarely need this.
        """
        cols = self.n if thin else self.m
        return self.apply_Q(np.eye(self.m, cols, dtype=self._dt))

    # ---- triangular factor ----------------------------------------------
    def solve_R(self, y):
        """Solve ``R x = y[:n]`` by banded back-substitution (no ``Q``)."""
        return self._tri_op("solve_R", y, self.n)

    def solve_Rt(self, y):
        """Solve ``R^H x = y`` by banded forward substitution."""
        return self._tri_op("solve_Rt", y, self.n)

    def R_band(self):
        """``R`` in row-major band form ``(n, kuR+1)``: ``out[i, j-i] = R[i, j]``."""
        out = np.zeros((self.n, self.kuR + 1), dtype=self._dt)
        self._fn("get_R")(self._h, self._p(out))
        return out

    def R_sparse(self):
        """``R`` as a ``scipy.sparse`` CSR matrix."""
        import scipy.sparse as sp
        Rb = self.R_band()
        n, w = Rb.shape
        rows, cols, vals = [], [], []
        for d in range(w):
            hi = n - d
            if hi <= 0:
                continue
            rows.append(np.arange(hi))
            cols.append(np.arange(hi) + d)
            vals.append(Rb[:hi, d])
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n))

    @property
    def diag_R(self):
        """The diagonal of ``R``; its magnitudes bound the conditioning."""
        return self.R_band()[:, 0].copy()

    def rank_deficient(self, tol=None):
        """True if ``R`` has a diagonal entry negligible against the largest.

        No pivoting means bandqr cannot detect rank deficiency *during* the
        factorisation; this is the after-the-fact check.
        """
        d = np.abs(self.diag_R)
        if d.size == 0:
            return False
        tol = tol if tol is not None else d.max() * self.n * np.finfo(np.float64).eps
        return bool((d <= tol).any())

    # ---- solving ---------------------------------------------------------
    def solve(self, b):
        """Solve for one RHS ``(m,)`` or many ``(m, nrhs)``.

        Sets ``.residual`` to ``||A x - b||`` (per column for multi-RHS), which
        is exact for the least-squares case and ~0 for square systems.
        """
        b = np.ascontiguousarray(b, dtype=self._dt)
        if b.shape[0] != self.m:
            raise ValueError(f"b has {b.shape[0]} rows but A is {self.m}x{self.n}")
        if b.ndim == 1:
            x = np.zeros(self.n, dtype=self._dt)
            self.residual = self._fn("solve")(self._h, self._p(b), self._p(x))
            return x
        nrhs = b.shape[1]
        x = np.zeros((self.n, nrhs), dtype=self._dt)
        res = np.zeros(nrhs, dtype=np.float64)
        self._fn("solve_many")(self._h, self._p(b), nrhs, self._p(x),
                               _native.as_ptr(res))
        self.residual = res
        return x

    # ---- operators the refinement / condition machinery needs ------------
    def matvec(self, x):
        """``A x`` against the *original* banded operator, not ``R``."""
        return band_matvec(self._ab, self.kl, self.ku, x, n=self.n)

    def solve_H(self, v):
        """``A^{-H} v = Q R^{-H} v`` (square ``A`` only)."""
        if self.m != self.n:
            raise ValueError("solve_H is defined for square systems only")
        return self.apply_Q(self.solve_Rt(v))

    def norm1(self):
        """``||A||_1`` from the band."""
        return band_norm1(self._ab, self.kl, self.ku, n=self.n)

    # ---- lifecycle -------------------------------------------------------
    def free(self):
        h = getattr(self, "_h", None)
        if h:
            self._fn("free")(h)
            self._h = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()
        return False

    def __repr__(self):
        kind = "householder" if self.householder else "givens"
        shape = f"{self.m}x{self.n}"
        thr = "" if self.nthreads == 1 else f", nthreads={self.nthreads}"
        return (f"<BandQR {shape} kl={self.kl} ku={self.ku} {kind} "
                f"{np.dtype(self._dt).name}{thr}>")


class LQ:
    """``A = L Q`` for a square banded ``A``, via the QR of ``A^H``.

    A banded transpose only swaps the bandwidths, so the row-side factorisation
    is the column-side one on the conjugate transpose.  This is the dual of
    :class:`BandQR`, and the natural form when structure lives in the *columns*.
    ``.solve(b)`` solves ``A x = b``.
    """

    def __init__(self, A, kl, ku):
        self.kl, self.ku = kl, ku
        ab = np.ascontiguousarray(to_band(A, kl, ku))
        self._z = np.iscomplexobj(ab)
        abT, klT, kuT = band_transpose(ab, kl, ku)
        abH = np.conjugate(abT) if self._z else abT      # band of A^H
        self._abH, self.klH, self.kuH = abH, klT, kuT
        self.qrT = BandQR(abH, klT, kuT)                 # A^H = Q_t R_t
        self.n = self.qrT.n

    def L_band(self):
        """``L = R_t^H`` in band storage ``(kl=kuR, ku=0)`` -- lower triangular."""
        kuR = self.qrT.kuR
        RbandT, _, _ = band_transpose(self.qrT.R_band(), 0, kuR)
        return (np.conjugate(RbandT) if self._z else RbandT), kuR

    def solve(self, b):
        import scipy.linalg as sla
        kuR = self.qrT.kuR
        b = np.ascontiguousarray(b, dtype=self.qrT._dt)
        RbandH, _ = self.L_band()                        # lower-triangular band
        u = sla.solve_banded((kuR, 0), to_scipy_band(RbandH, kuR, 0), b)
        z = self.qrT.solve_R(u)
        return band_matvec(self._abH, self.klH, self.kuH, z)

    def __repr__(self):
        return f"<LQ {self.n}x{self.n} kl={self.kl} ku={self.ku}>"
