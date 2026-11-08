"""The flagged front end: :func:`solve` and :func:`factorize`.

One entry point over every factorisation, dispatching on the structure it finds
unless you tell it otherwise.
"""
import numpy as np

from . import _native
from ._band import to_band
from .almost import (BorderedQR, CollapseBothQR, CollapseColsQR, CollapseQR,
                     RideQR, RideTauQR)
from .banded import LQ, BandQR
from .refine import compensated_residual, iterative_refinement
from .structure import detect_structure

__all__ = ["solve", "factorize", "METHODS"]

#: Available factorisations.  ``auto`` picks by structure; the rest force one.
METHODS = ("auto", "givens", "householder", "lq", "collapse", "ride",
           "bordered", "lu")

_METHOD_DOC = """
    ==============  ====================================================  ==================================
    method          what it does                                          when to reach for it
    ==============  ====================================================  ==================================
    ``auto``        dispatch on detected structure                        the default
    ``givens``      Givens band QR                                        purely banded, narrow
    ``householder`` Householder band QR                                   purely banded, wide (~25% faster)
    ``lq``          ``L Q`` -- the QR of the transpose                     structure lives in the columns
    ``collapse``    sparsify borders by their recurrence, then one QR     dense borders, throughput first
    ``ride``        almost-banded QR, borders carried through             dense borders, accuracy first
    ``bordered``    Woodbury against the in-band part                     rows *and* tau columns, bulk invertible
    ``lu``          banded LU (``scipy.linalg.solve_banded``)             comparison, fastest re-solve
    ==============  ====================================================  ==================================
"""


def _resolve(A, method, kl, ku, dense_rows, dense_cols):
    st = detect_structure(A)
    kl = st.kl if kl is None else int(kl)
    ku = st.ku if ku is None else int(ku)
    dr = st.dense_rows if dense_rows is None else np.asarray(dense_rows, dtype=int)
    dc = st.dense_cols if dense_cols is None else np.asarray(dense_cols, dtype=int)
    has_rows, has_cols = len(dr) > 0, len(dc) > 0

    if method == "auto":
        if not has_rows and not has_cols:
            # Householder factors faster once the band is wide enough for the
            # reflector's extra arithmetic to beat Givens' extra passes.
            method = "householder" if kl >= 6 else "givens"
        else:
            # `ride` carries the borders untouched, so R = Q^H M exactly.  It is
            # the accurate choice; `collapse` is the fast one, but pays for its
            # preconditioner in conditioning, and increasingly so with n.
            method = "ride"
    return method, kl, ku, dr, dc


def factorize(A, method="auto", kl=None, ku=None, dense_rows=None,
              dense_cols=None, store_P=False, nthreads=0):
    """Factor once, solve many: returns an object with a ``.solve(b)`` method.

    Same dispatch as :func:`solve`.
    """ + _METHOD_DOC + """
    Parameters
    ----------
    A : ndarray or scipy.sparse matrix
        Square, banded or almost-banded.
    kl, ku, dense_rows, dense_cols :
        Override structure detection when given.
    store_P : bool
        ``collapse`` only -- keep the explicit sparse preconditioner on the
        result as ``.P``.
    nthreads : int
        Banded factorisations only; ``0`` is auto (see
        :func:`bandedqr.banded.auto_threads`).

    Returns
    -------
    object
        :class:`~bandedqr.banded.BandQR`, :class:`~bandedqr.banded.LQ`,
        :class:`~bandedqr.almost.RideQR`, :class:`~bandedqr.almost.RideTauQR`,
        :class:`~bandedqr.almost.CollapseQR` and friends, or
        :class:`~bandedqr.almost.BorderedQR`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "lu":
        raise ValueError("factorize does not wrap banded LU; use method='lu' "
                         "with solve(), or scipy.linalg.lu_factor directly")
    method, kl, ku, dr, dc = _resolve(A, method, kl, ku, dense_rows, dense_cols)
    has_rows, has_cols = len(dr) > 0, len(dc) > 0

    if method in ("givens", "householder"):
        if has_rows or has_cols:
            raise ValueError(
                f"method={method!r} needs a purely banded matrix but found "
                f"{len(dr)} dense rows and {len(dc)} dense columns; use "
                "'ride' (accurate) or 'collapse' (fast)")
        return BandQR(to_band(A, kl, ku), kl, ku, nthreads=nthreads,
                      householder=(method == "householder"))
    if method == "lq":
        if has_rows or has_cols:
            raise ValueError("method='lq' needs a purely banded matrix; use "
                             "'ride' or 'collapse'")
        return LQ(A, kl, ku)
    if method == "collapse":
        if has_rows and has_cols:
            return CollapseBothQR(A, kl, ku, dr, dc)
        if has_cols:
            return CollapseColsQR(A, kl, ku, dc)
        if has_rows:
            return CollapseQR(A, kl, ku, dr, store_P=store_P)
        return BandQR(to_band(A, kl, ku), kl, ku, nthreads=nthreads)
    if method == "ride":
        if has_cols:
            if not has_rows:
                raise ValueError(
                    "'ride' carries dense rows; with tau columns only, use "
                    "'bordered' (Woodbury) or 'collapse'")
            return RideTauQR(A, kl, ku, dr, dc)
        if not has_rows:
            return BandQR(to_band(A, kl, ku), kl, ku, nthreads=nthreads)
        return RideQR(A, kl, ku, dr)
    if method == "bordered":
        if not has_rows and not has_cols:      # no borders: nothing to correct
            return BandQR(to_band(A, kl, ku), kl, ku, nthreads=nthreads)
        return BorderedQR(A, kl, ku, dr, dc)
    raise ValueError(f"unhandled method {method!r}")          # pragma: no cover


def solve(A, b, method="auto", device="cpu", kl=None, ku=None,
          dense_rows=None, dense_cols=None, refine=0, nthreads=0):
    """Solve ``A x = b`` for a banded or almost-banded ``A``.
    """ + _METHOD_DOC + """
    Parameters
    ----------
    A : ndarray or scipy.sparse matrix
        Square, banded or almost-banded.  dtype selects the real or complex
        kernel -- no flag needed.
    b : ndarray
        ``(n,)`` or ``(n, nrhs)``.
    method : str
        See the table above.
    device : {"cpu", "gpu"}
        ``"gpu"`` is not a single-system option -- the Metal backend exists to
        run *many* systems at once.  Use :func:`bandedqr.solve_batched`.
    refine : int
        Iterative-refinement steps against a compensated (double-double)
        residual.  ``0`` (the default) is right for a float64 factorisation of
        a well-conditioned system; raise it when ``A`` is ill conditioned and
        you want the backward error pushed back down.  Purely banded systems
        only.
    nthreads : int
        Banded factorisations only; ``0`` is auto.

    Returns
    -------
    ndarray
        Shaped like ``b``.
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")
    if device == "gpu":
        raise ValueError(
            "device='gpu' solves a *batch* of systems, not one.  Use "
            "bandedqr.solve_batched(ab, b, kl, ku) with band storage and a batch "
            "axis; see bandedqr.gpu.")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if getattr(A, "ndim", None) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"solve expects a square 2-D matrix, got shape "
                         f"{getattr(A, 'shape', None)}")
    if np.asarray(b).shape[0] != A.shape[0]:
        raise ValueError(f"b has {np.asarray(b).shape[0]} rows but A is "
                         f"{A.shape[0]}x{A.shape[0]}")

    dt = _native.dtype_for(A)
    b = np.ascontiguousarray(b, dt)

    if method == "lu":
        _, kl_, ku_, dr, dc = _resolve(A, "lu", kl, ku, dense_rows, dense_cols)
        if len(dr) or len(dc):
            raise ValueError(
                "'lu' densifies almost-banded systems; use 'ride' or 'collapse'")
        import scipy.linalg as sla
        from ._band import to_scipy_band
        return sla.solve_banded((kl_, ku_), to_scipy_band(to_band(A, kl_, ku_),
                                                          kl_, ku_), b)

    F = factorize(A, method=method, kl=kl, ku=ku, dense_rows=dense_rows,
                  dense_cols=dense_cols, nthreads=nthreads)
    x = F.solve(b)
    if refine:
        if not isinstance(F, BandQR):
            raise ValueError("refine= is implemented for purely banded systems; "
                             "for almost-banded ones refine by hand against "
                             "bandedqr.refine.iterative_refinement")
        ab = to_band(A, F.kl, F.ku)
        res = iterative_refinement(
            F.solve, lambda v: compensated_residual(ab, F.kl, F.ku, v, b),
            b, x0=x, maxiter=int(refine))
        x = res.x
    return x
