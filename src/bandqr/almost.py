"""Almost-banded factorisations: a banded bulk plus dense rows and/or columns.

Two families, and the choice between them is the interesting part:

* **carry** (``Ride*``) -- factor the dense rows *along with* the band, keeping
  them as a rank-r update ``M = B + U L`` that rides through the rotations.
  ``R = Q^H M`` exactly, with no preconditioner, so nothing is done to the
  conditioning.
* **collapse** (``Collapse*``) -- sparsify the dense rows into the band first,
  using the recurrence they satisfy (see :mod:`bandqr.recurrence`), then do a
  single ordinary banded QR.  Faster, but the preconditioner is a similarity
  the conditioning has to pay for, and that cost grows with n.

Carry is the default for anything where accuracy matters at large n; collapse
is the one to reach for when the borders are well-conditioned and throughput is
what you want.
"""
import ctypes as ct

import numpy as np

from . import _native
from ._band import to_band
from .banded import BandQR
from .recurrence import (apply_preconditioner, build_preconditioner,
                         detect_recurrence)
from .structure import detect_structure

__all__ = [
    "RideQR", "RideQRRect", "RideTauQR",
    "CollapseQR", "CollapseColsQR", "CollapseBothQR",
    "BorderedQR", "solve_bordered", "solve_ride", "solve_precond_rows",
    "solve_with_tau_columns",
]


def _csr(M, dtype=None):
    import scipy.sparse as sp
    C = sp.csr_matrix(M)
    return C.astype(dtype, copy=False) if dtype is not None else C


class _RideBase:
    """Shared plumbing for the carried-border factorisations."""

    def _init_common(self, M):
        self._z = _native.is_complex(M)
        self._dt = np.complex128 if self._z else np.float64
        self._suf = _native.suffix_for(self._z)
        self._p = _native.ptr_for(self._z)

    def _factor(self, Mp, m, n, kle, ku, dr):
        """Build the C almost-banded factorisation of ``Mp``.

        ``dr`` indexes the dense rows *of Mp*; ``L`` holds only their
        out-of-band content, since the in-band part already lives in the band.
        """
        dr = np.ascontiguousarray(np.asarray(dr), dtype=np.int32)
        r = len(dr)
        ab = np.ascontiguousarray(to_band(Mp, kle, ku), self._dt)
        L = np.asarray(_csr(Mp)[dr, :].todense()).astype(self._dt)
        for t in range(r):
            i = int(dr[t])
            lo, hi = max(0, i - kle), min(n - 1, i + ku)
            if lo <= hi:
                L[t, lo:hi + 1] = 0                 # keep only out-of-band
        L = np.ascontiguousarray(L, self._dt)
        self._L = L
        h = getattr(_native.lib, "bandqr_factor_ab" + self._suf)(
            self._p(ab), m, n, kle, ku,
            dr.ctypes.data_as(ct.POINTER(ct.c_int)), r, self._p(L))
        if not h:
            raise RuntimeError(
                "almost-banded factorisation failed (need m >= n, kl,ku >= 0, r >= 1)")
        return h

    def _solve_raw(self, b):
        """Solve for the already-row-permuted right-hand side ``b``."""
        b = np.ascontiguousarray(b, self._dt)
        single = b.ndim == 1
        B = np.ascontiguousarray(b[:, None] if single else b, self._dt)
        nrhs = B.shape[1]
        X = np.zeros((self.n, nrhs), dtype=self._dt)
        res = np.zeros(nrhs, dtype=np.float64)
        getattr(_native.lib, "bandqr_solve_many_ab" + self._suf)(
            self._h, self._p(B), nrhs, self._p(X), _native.as_ptr(res))
        self.residual = res[0] if single else res
        return X[:, 0] if single else X

    def free(self):
        h = getattr(self, "_h", None)
        if h:
            getattr(_native.lib, "bandqr_free" + self._suf)(h)
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


class RideQR(_RideBase):
    """Almost-banded QR with the dense rows carried along (Olver-Townsend).

    ``M = B + U L`` is factored as a whole, so ``R = Q^H M`` exactly -- no
    preconditioner, and markedly better conditioned than :class:`CollapseQR` at
    large ``n``.  Factor once, solve many.

    The kernel reaches a dense row by widening ``kl`` far enough to touch it, so
    rows near the bottom would blow the band up.  When that would happen the
    dense rows are permuted to the top automatically (columns -- the unknowns --
    are untouched, so ``x`` keeps its order and the permutation is applied to
    ``b`` inside :meth:`solve`).
    """

    def __init__(self, M, kl, ku, dense_rows):
        self._init_common(M)
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        if dr.size == 0:
            raise ValueError("RideQR needs at least one dense row; "
                             "use BandQR for a purely banded matrix")
        r = len(dr)
        self.n = n
        self.perm = None

        # Reaching a bottom row costs kl -> dr.max(); permuting instead costs
        # kl -> kl + r.  Take whichever band is narrower.
        if int(dr.max()) > kl + r:
            rest = np.setdiff1d(np.arange(n), dr)
            self.perm = np.concatenate([dr, rest]).astype(int)
            Mp = _csr(M, self._dt)[self.perm, :]
            dr_eff = np.arange(r)
            kle, kue = kl + r, ku
        else:
            Mp = _csr(M, self._dt)
            dr_eff = dr
            kle, kue = max(kl, int(dr.max())), ku

        self.kl, self.ku = kle, kue
        self.dense_rows = dr
        self._h = self._factor(Mp, n, n, kle, kue, dr_eff)
        self.residual = None

    def solve(self, b):
        """Solve ``M x = b`` for one RHS ``(n,)`` or many ``(n, nrhs)``."""
        b = np.asarray(b)
        if self.perm is not None:
            b = b[self.perm]
        return self._solve_raw(b)

    def __repr__(self):
        p = ", permuted" if self.perm is not None else ""
        return (f"<RideQR {self.n}x{self.n} kl={self.kl} ku={self.ku} "
                f"r={len(self.dense_rows)} {np.dtype(self._dt).name}{p}>")


class RideQRRect(_RideBase):
    """Tall almost-banded ride QR: least squares for ``A`` ``(m, n)``, ``m > n``.

    The shape you get when a banded operator is posed on all ``n`` unknowns and
    the boundary conditions are *appended* as extra dense rows, leaving the
    system tall and the banded bulk singular on its own.  The dense rows land at
    the bottom, out of the kernel's rotation reach, so they are permuted to the
    top (columns untouched, so ``x`` keeps its order); the tall factorisation
    then triangularises the whole thing and the least-squares residual lives in
    the trailing ``m - n`` rows.

    The rectangular sibling of :class:`RideQR`: there the dense rows *replace*
    banded ones (``#rows == n``); here they are genuinely extra (``#rows > n``).
    """

    def __init__(self, M, kl=None, ku=None, dense_rows=None):
        M = _csr(M)
        m, n = M.shape
        if m < n:
            raise ValueError(f"RideQRRect needs m >= n (tall), got {M.shape}")
        self._init_common(M)
        st = detect_structure(M)
        dr0 = st.dense_rows if dense_rows is None else np.asarray(dense_rows, dtype=int)
        if len(dr0) < 1:
            raise ValueError("no dense rows: use BandQR for a purely banded matrix")
        self.perm = np.concatenate([dr0, np.setdiff1d(np.arange(m), dr0)]).astype(int)
        Mp = _csr(M[self.perm], self._dt)
        pst = detect_structure(Mp)
        dr = np.asarray(pst.dense_rows, dtype=int)
        if dr.size == 0:                       # permutation folded them into the band
            dr = np.arange(len(dr0))
        pkl = int(pst.kl if kl is None else kl)
        pku = int(pst.ku if ku is None else ku)
        self.m, self.n = m, n
        kle = max(pkl, int(dr.max()))
        self.kl, self.ku = kle, pku
        self.dense_rows = dr0
        self._h = self._factor(Mp, m, n, kle, pku, dr)
        self.residual = None

    def solve(self, b):
        """Least-squares solve ``min ||A x - b||``; ``.residual`` gets the norm."""
        b = np.asarray(b)[self.perm]
        return self._solve_raw(b)

    def __repr__(self):
        return (f"<RideQRRect {self.m}x{self.n} kl={self.kl} ku={self.ku} "
                f"r={len(self.dense_rows)} {np.dtype(self._dt).name}>")


class RideTauQR:
    """Dense boundary rows **and** low-rank tau columns.

    Ride carries the dense rows; the tau columns are replaced by unit columns
    (so the bulk stays non-singular) and restored by a Woodbury update.  Writing
    ``M = M_R + G W`` with ``M_R`` the ride-able matrix, ``G`` the true tau
    columns minus ``e_c``, and ``W = e_dc^T``::

        x = y - Z (I + Z[dc])^{-1} y[dc],   y = M_R^{-1} b,   Z = M_R^{-1} G

    with ``Z`` and the small ``dc x dc`` system formed once at factor time.
    """

    def __init__(self, M, kl, ku, dense_rows, dense_cols, tol=1e-8):
        import scipy.linalg as sla
        import scipy.sparse as sp
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        dc = np.asarray(dense_cols, dtype=int)
        if dc.size == 0:
            raise ValueError("RideTauQR needs at least one dense column")
        self.n = n
        self._z = _native.is_complex(M)
        self._dt = np.complex128 if self._z else np.float64
        # ride wants the dense BC rows at the top; x is unaffected by a row swap
        rest = np.setdiff1d(np.arange(n), dr)
        self.perm = np.concatenate([dr, rest]).astype(int)
        Mp = _csr(M, self._dt)[self.perm, :]
        r_rows = len(dr)
        # tau columns -> unit columns (M_R), and G = (true tau columns) - e_c
        G = np.asarray(Mp[:, dc].todense()).astype(self._dt)
        for k, c in enumerate(int(v) for v in dc):
            G[c, k] -= 1.0
        keep = np.ones(n, dtype=self._dt)
        keep[dc] = 0.0
        E = sp.csr_matrix((np.ones(len(dc), dtype=self._dt), (dc, dc)), shape=(n, n))
        Mr = (Mp @ sp.diags(keep) + E).tocsr()
        st = detect_structure(Mr)
        # ride, not collapse: zeroing the tau columns breaks the dense-row
        # recurrence that collapse depends on, but ride needs no recurrence
        self.inner = RideQR(Mr, st.kl, st.ku, np.arange(r_rows))
        self.dc = dc
        self.Z = self.inner.solve(G)                          # M_R^{-1} G
        small = np.eye(len(dc), dtype=self._dt) + self.Z[dc, :]
        self.small_lu = sla.lu_factor(small)

    def solve(self, b):
        import scipy.linalg as sla
        b = np.asarray(b)[self.perm]
        y = self.inner.solve(b)
        c = sla.lu_solve(self.small_lu, y[self.dc])
        return y - self.Z @ c

    def __repr__(self):
        return (f"<RideTauQR {self.n}x{self.n} r={len(self.perm) and len(self.Z.T)} "
                f"c={len(self.dc)} {np.dtype(self._dt).name}>")


class CollapseQR:
    """Recurrence-preconditioned solve: sparsify the dense rows, then one band QR.

    The right-preconditioner ``P`` (``M @ P`` banded, ``x = P @ y``) is
    represented compactly by its recurrence coefficients ``.a``.  Pass
    ``store_P=True`` to also retain the explicit sparse matrix as ``.P``;
    :meth:`preconditioner` returns it either way, rebuilding from ``.a`` on
    demand, and :meth:`apply_P` applies ``P`` to an arbitrary vector or block.

    Faster than :class:`RideQR` but less accurate at large ``n``: the
    preconditioner's own conditioning enters the answer.
    """

    def __init__(self, M, kl, ku, dense_rows, max_order=20, tol=1e-8, store_P=False):
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        Msp = _csr(M)
        a = detect_recurrence(np.asarray(Msp[dr, :].todense()), max_order, tol)
        if a is None:
            raise RuntimeError(
                f"no recurrence of order <= {max_order} found for the dense rows. "
                "The borders may not be recurrence-structured -- use method='ride', "
                "which needs no recurrence, or raise max_order.")
        P = build_preconditioner(a, n)
        MP = (Msp @ P).tocoo()
        # M @ P leaves ~1e-16 structural cancellation across the collapsed rows;
        # drop it by value so the band stays tight instead of appearing dense
        keep = np.abs(MP.data) > tol * np.abs(MP.data).max()
        r_, c_, d_ = MP.row[keep], MP.col[keep], MP.data[keep]
        off = c_ - r_
        kl_eff, ku_eff = int(-off.min()), int(off.max())
        ab = np.zeros((n, kl_eff + ku_eff + 1), dtype=d_.dtype)
        ab[r_, c_ - r_ + kl_eff] = d_
        self.qr = BandQR(ab, kl_eff, ku_eff)
        self.a = a
        self.n = n
        self.kl, self.ku = kl_eff, ku_eff
        self.P = P if store_P else None

    def apply_P(self, y):
        """Apply the right-preconditioner ``x = P @ y`` via the recurrence (in C)."""
        return apply_preconditioner(self.a, y)

    def preconditioner(self):
        """The explicit sparse ``P``; rebuilt from ``.a`` if it was not stored."""
        return self.P if self.P is not None else build_preconditioner(self.a, self.n)

    def solve(self, b):
        return apply_preconditioner(self.a, self.qr.solve(b))

    def __repr__(self):
        return (f"<CollapseQR {self.n}x{self.n} kl={self.kl} ku={self.ku} "
                f"stencil_order={len(self.a)-1}>")


class CollapseColsQR:
    """Dense-*column* collapse -- the transpose-dual of :class:`CollapseQR`.

    A left-preconditioner ``Q`` built from the recurrence the dense columns
    satisfy sparsifies them into the band, so ``M x = b`` becomes the single
    banded QR ``(Q M) x = Q b``.  ``Q`` is algebraic, not orthogonal, so a plain
    transpose suffices for complex data.

    The columns must be *dense and recurrence-structured*.  Low-rank tau columns
    (a single lifted mode) are not this shape -- carry those with
    :class:`RideTauQR` or :class:`BorderedQR`.
    """

    def __init__(self, M, kl, ku, dense_cols, max_order=20, tol=1e-8):
        n = M.shape[1]
        dc = np.asarray(dense_cols, dtype=int)
        if dc.size == 0:
            raise ValueError("CollapseColsQR needs at least one dense column")
        Msp = _csr(M)
        a = detect_recurrence(np.asarray(Msp[:, dc].T.todense()), max_order, tol)
        if a is None:
            raise RuntimeError(
                f"no recurrence of order <= {max_order} found for the dense columns. "
                "Low-rank tau columns are not recurrence-structured -- use "
                "method='ride' or 'bordered' instead.")
        Q = build_preconditioner(a, n).T.tocsr()
        QM = (Q @ Msp).tocoo()
        keep = np.abs(QM.data) > tol * np.abs(QM.data).max()
        r_, c_, d_ = QM.row[keep], QM.col[keep], QM.data[keep]
        off = c_ - r_
        kl_eff, ku_eff = int(-off.min()), int(off.max())
        ab = np.zeros((n, kl_eff + ku_eff + 1), dtype=d_.dtype)
        ab[r_, c_ - r_ + kl_eff] = d_
        self.qr = BandQR(ab, kl_eff, ku_eff)
        self.a = a
        self.n = n
        self.kl, self.ku = kl_eff, ku_eff
        self.Q = Q

    def solve(self, b):
        return self.qr.solve(self.Q @ np.asarray(b))

    def __repr__(self):
        return f"<CollapseColsQR {self.n}x{self.n} kl={self.kl} ku={self.ku}>"


class CollapseBothQR:
    """Doubly-bordered collapse: dense rows *and* dense columns at once.

    A right-preconditioner ``P`` (row recurrence) and a left-preconditioner
    ``Q`` (column recurrence) together band the matrix -- ``Q M P`` is banded --
    so ``x = P (Q M P)^{-1} Q b`` is a single banded QR.

    Both borders must be dense and recurrence-structured; this is not the
    low-rank tau-column shape.
    """

    def __init__(self, M, kl, ku, dense_rows, dense_cols, max_order=20, tol=1e-8):
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        dc = np.asarray(dense_cols, dtype=int)
        if dr.size == 0 or dc.size == 0:
            raise ValueError("CollapseBothQR needs at least one dense row and column")
        Msp = _csr(M)
        a_r = detect_recurrence(np.asarray(Msp[dr, :].todense()), max_order, tol)
        a_c = detect_recurrence(np.asarray(Msp[:, dc].T.todense()), max_order, tol)
        if a_r is None or a_c is None:
            which = "rows" if a_r is None else "columns"
            raise RuntimeError(
                f"no recurrence of order <= {max_order} found for the dense {which}; "
                "use method='ride'/'bordered' or raise max_order.")
        P = build_preconditioner(a_r, n)
        Q = build_preconditioner(a_c, n).T.tocsr()
        B = (Q @ Msp @ P).tocoo()
        keep = np.abs(B.data) > tol * np.abs(B.data).max()
        r_, c_, d_ = B.row[keep], B.col[keep], B.data[keep]
        off = c_ - r_
        kl_eff, ku_eff = int(-off.min()), int(off.max())
        ab = np.zeros((n, kl_eff + ku_eff + 1), dtype=d_.dtype)
        ab[r_, c_ - r_ + kl_eff] = d_
        self.qr = BandQR(ab, kl_eff, ku_eff)
        self.a_r = a_r
        self.Q = Q
        self.n = n
        self.kl, self.ku = kl_eff, ku_eff

    def solve(self, b):
        return apply_preconditioner(self.a_r, self.qr.solve(self.Q @ np.asarray(b)))

    def __repr__(self):
        return f"<CollapseBothQR {self.n}x{self.n} kl={self.kl} ku={self.ku}>"


def _bulk_band(M, kl, ku, dtype):
    """In-band entries of ``M`` with any zero diagonal forced to 1.

    The Woodbury correction needs the bulk to be invertible; a differential
    operator whose diagonal vanishes somewhere would otherwise sink it.
    """
    ab = np.ascontiguousarray(to_band(M, kl, ku), dtype)
    d = ab[:, kl]
    d[np.abs(d) < 1e-300] = 1.0
    return ab


def _m0_row(ab, kl, ku, i, n, dtype):
    row = np.zeros(n, dtype=dtype)
    lo, hi = max(0, i - kl), min(n - 1, i + ku)
    row[lo:hi + 1] = ab[i, lo - i + kl:hi - i + kl + 1]
    return row


def _m0_col(ab, kl, ku, j, n, dtype):
    col = np.zeros(n, dtype=dtype)
    lo, hi = max(0, j - ku), min(n - 1, j + kl)
    idx = np.arange(lo, hi + 1)
    col[idx] = ab[idx, j - idx + kl]
    return col


class BorderedQR:
    """Woodbury against the in-band part: dense rows **and** dense columns.

    ``M0 = in-band(M)`` is banded and made invertible; ``M - M0`` is nonzero
    only in the borders, hence rank ``r + c``.  ``M0^{-1}`` comes from the fast
    band QR and the correction is a dense ``(r+c) x (r+c)`` solve.  The band
    factorisation -- the expensive part -- is built once and reused across the
    ``1 + (r+c)`` right-hand sides.

    Robust for *any* bandwidth, in particular when ``#dense rows > kl`` where
    deleting the border and taking a Schur complement leaves a singular bulk.
    Needs a non-singular in-band bulk; for differential operators where that
    fails, use ``ride``/``collapse`` on the rows instead.
    """

    def __init__(self, M, kl, ku, dense_rows, dense_cols):
        import scipy.sparse as sp
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        dc = np.asarray(dense_cols, dtype=int)
        r, c = len(dr), len(dc)
        if r + c == 0:
            raise ValueError("BorderedQR needs at least one dense row or column")
        dt = np.complex128 if _native.is_complex(M) else np.float64
        self._dt = dt
        self.n = n

        ab = _bulk_band(M, kl, ku, dt)
        self.qr = BandQR(ab, kl, ku)

        if sp.issparse(M):
            rows_dense = np.asarray(M.tocsr()[dr, :].todense()).astype(dt)
            cols_dense = np.asarray(M.tocsc()[:, dc].todense()).astype(dt)
        else:
            Md = np.asarray(M, dtype=dt)
            rows_dense, cols_dense = Md[dr, :], Md[:, dc]

        # M - M0 = U V^T: full rows for the dense rows; the dense columns
        # contribute only on non-dense rows, so the corner is not double counted
        U = np.zeros((n, r + c), dtype=dt)
        V = np.zeros((n, r + c), dtype=dt)
        for t, i in enumerate(dr):
            U[i, t] = 1.0
            V[:, t] = rows_dense[t] - _m0_row(ab, kl, ku, i, n, dt)
        keep = np.ones(n, dtype=bool)
        keep[dr] = False
        for t, j in enumerate(dc):
            col = cols_dense[:, t] - _m0_col(ab, kl, ku, j, n, dt)
            col[~keep] = 0.0
            U[:, r + t] = col
            V[j, r + t] = 1.0

        self.V = V
        self.MiU = self.qr.solve(U)                       # M0^{-1} U
        self.cap = np.eye(r + c, dtype=dt) + V.T @ self.MiU
        self.cap_lu = __import__("scipy.linalg", fromlist=["lu_factor"]).lu_factor(self.cap)

    def solve(self, b):
        import scipy.linalg as sla
        b = np.asarray(b, dtype=self._dt)
        single = b.ndim == 1
        B = b[:, None] if single else b
        MiB = self.qr.solve(B)
        if MiB.ndim == 1:
            MiB = MiB[:, None]
        X = MiB - self.MiU @ sla.lu_solve(self.cap_lu, self.V.T @ MiB)
        return X[:, 0] if single else X

    def __repr__(self):
        return f"<BorderedQR {self.n}x{self.n} rank={self.MiU.shape[1]}>"


def solve_bordered(M, kl, ku, b, dense_rows, dense_cols):
    """One-shot Woodbury solve; see :class:`BorderedQR` for factor-once reuse."""
    return BorderedQR(M, kl, ku, dense_rows, dense_cols).solve(b)


def solve_ride(M, kl, ku, b, dense_rows):
    """One-shot carried-border solve; see :class:`RideQR` for factor-once reuse."""
    return RideQR(M, kl, ku, dense_rows).solve(b)


def solve_precond_rows(M, kl, ku, b, dense_rows, max_order=20, tol=1e-8):
    """One-shot recurrence-collapse solve; see :class:`CollapseQR` for reuse."""
    return CollapseQR(M, kl, ku, dense_rows, max_order, tol).solve(b)


def solve_with_tau_columns(N_ab, kl, ku, F, b):
    """Solve ``[N | F] x = b`` for a tall banded ``N`` and dense tau columns ``F``.

    ``N`` is ``m x (m-c)`` in band storage and ``F`` is ``m x c``.  QR the tall
    banded ``N``: ``Q^T [N|F] = [R | Q^T F]``, whose bottom ``c`` rows are a
    dense ``c x c`` system for the tau unknowns; the top ``n`` rows then give a
    banded back-solve for the bulk.  Robust for any bandwidth -- no invertible
    bulk required.

    Returns ``x`` with the bulk unknowns first and the ``c`` tau unknowns last.
    """
    F = np.asarray(F)
    m, c = F.shape
    n = m - c
    dt = np.complex128 if (_native.is_complex(N_ab) or _native.is_complex(F)
                           or _native.is_complex(b)) else np.float64
    qr = BandQR(np.ascontiguousarray(N_ab, dt), kl, ku, ncols=n)
    QtF = qr.apply_Qt(np.ascontiguousarray(F, dt))            # (m, c)

    b = np.ascontiguousarray(b, dt)
    single = b.ndim == 1
    B = b[:, None] if single else b
    Qtb = qr.apply_Qt(B)                                      # (m, nrhs)
    x2 = np.linalg.solve(QtF[n:, :], Qtb[n:])                 # tau unknowns
    x1 = qr.solve_R(Qtb[:n] - QtF[:n, :] @ x2)                # bulk unknowns
    X = np.concatenate([np.atleast_2d(x1.T).T if x1.ndim == 1 else x1,
                        np.atleast_2d(x2.T).T if x2.ndim == 1 else x2], axis=0)
    return X[:, 0] if single else X
