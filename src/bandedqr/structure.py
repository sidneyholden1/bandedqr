"""Inferring almost-banded structure from a matrix.

An *almost-banded* matrix is a banded bulk plus a small number of dense rows
(boundary conditions) and/or dense columns (tau corrections).  Recovering
``(kl, ku, dense_rows, dense_cols)`` from the sparsity pattern is what lets
:func:`bandedqr.solve` dispatch without being told the structure.

The detection is deliberately conservative and reports what it did.  It is a
heuristic on an ambiguous problem -- "is this row a boundary condition or just a
wide row of the bulk?" has no value-free answer -- so every entry point that
uses it also accepts explicit overrides, and :func:`detect_structure` returns a
report you can inspect when a dispatch surprises you.
"""
import warnings

import numpy as np

__all__ = ["Structure", "detect_structure", "StructureWarning"]


class StructureWarning(UserWarning):
    """Structure detection had to make a judgement call worth knowing about."""


class Structure(tuple):
    """``(kl, ku, dense_rows, dense_cols)`` plus provenance.

    Unpacks as a plain 4-tuple, so ``kl, ku, dr, dc = detect_structure(A)``
    keeps working, but carries extra diagnostics as attributes.
    """

    def __new__(cls, kl, ku, dense_rows, dense_cols, *, shape=(0, 0), nnz=0,
                widened=False, capped=False, bulk_kl=0, bulk_ku=0):
        self = super().__new__(cls, (int(kl), int(ku),
                                     np.asarray(dense_rows, dtype=int),
                                     np.asarray(dense_cols, dtype=int)))
        self.shape = shape
        self.nnz = nnz
        self.widened = widened
        self.capped = capped
        self.bulk_kl = bulk_kl
        self.bulk_ku = bulk_ku
        return self

    kl = property(lambda self: self[0])
    ku = property(lambda self: self[1])
    dense_rows = property(lambda self: self[2])
    dense_cols = property(lambda self: self[3])

    def describe(self):
        """One-line human summary -- useful when a dispatch surprises you."""
        m, n = self.shape
        bits = [f"{m}x{n}", f"nnz={self.nnz}", f"band=({self.kl},{self.ku})"]
        if len(self.dense_rows):
            bits.append(f"dense_rows={len(self.dense_rows)}")
        if len(self.dense_cols):
            bits.append(f"dense_cols={len(self.dense_cols)}")
        if self.widened:
            bits.append(f"band widened from ({self.bulk_kl},{self.bulk_ku})")
        if self.capped:
            bits.append("dense-border cap hit")
        return "  ".join(bits)

    def __repr__(self):
        return f"<Structure {self.describe()}>"


def _coo(A, tol):
    import scipy.sparse as sp
    if sp.issparse(A):
        C = A.tocoo()
        row, col, data = C.row, C.col, C.data
        shape = A.shape
    else:
        A = np.asarray(A)
        shape = A.shape
        row, col = np.nonzero(np.abs(A) > tol)
        data = A[row, col]
        return row, col, data, shape
    keep = np.abs(data) > tol
    return row[keep], col[keep], data[keep], shape


def detect_structure(A, tol=1e-12, max_dense=None, density=0.2, warn=False):
    """Infer ``(kl, ku, dense_rows, dense_cols)`` for an almost-banded ``A``.

    The method, in three steps:

    1. **Provisional band.**  Count the nonzeros on each diagonal.  A diagonal
       belonging to the bulk is populated along its whole length; one touched
       only by a few dense rows or columns holds a handful of entries.  Take
       the contiguous run of well-populated diagonals around the main one.
    2. **Greedy border cover.**  Whatever nonzeros fall outside that band are
       *violations*.  Repeatedly mark whichever row or column covers the most
       violations as dense -- a true boundary row covers ~n of them, an
       ordinary wide row only a few -- and stop once the best candidate no
       longer sticks out by more than one bandwidth's worth of entries.
    3. **Tighten.**  Recompute ``kl``/``ku`` from the entries that remain, and
       widen the band to swallow any violations the cover declined to take.

    Parameters
    ----------
    tol : float
        Entries at or below this magnitude are treated as structural zeros.
    max_dense : int, optional
        Cap on how many dense rows *and* columns may be identified.  Defaults to
        ``max(8, min(64, n // 8))``.  Hitting the cap is reported.
    density : float
        A diagonal counts as part of the bulk when it holds at least this
        fraction of the busiest diagonal's entries.
    warn : bool
        Emit a :class:`StructureWarning` when the band had to be widened or the
        cap was hit -- both signs the matrix is not cleanly almost-banded.

    Returns
    -------
    Structure
        Unpacks as ``(kl, ku, dense_rows, dense_cols)``; carries diagnostics.
    """
    row, col, _, shape = _coo(A, tol)
    m, n = shape
    if max_dense is None:
        max_dense = max(8, min(64, n // 8))

    if row.size == 0:
        return Structure(0, 0, np.empty(0, int), np.empty(0, int),
                         shape=shape, nnz=0)

    off = col.astype(np.int64) - row.astype(np.int64)

    # --- 1. provisional band from how populated each diagonal is ------------
    # A diagonal of the bulk carries ~n entries; a diagonal touched only by a
    # few dense rows or columns carries only a handful.  That gap separates
    # them cleanly, and unlike a per-row reach it is symmetric -- measuring
    # rows alone lets dense *columns* inflate ku until the band is the whole
    # matrix, which is exactly the case this replaced.
    omin = int(off.min())
    counts = np.bincount(off - omin, minlength=int(off.max()) - omin + 1)
    zero = -omin                                     # index of offset 0
    full = float(counts.max())
    floor_ = max(4.0, density * full)
    if 0 <= zero < counts.size and counts[zero] >= floor_:
        lo = zero
        while lo - 1 >= 0 and counts[lo - 1] >= floor_:
            lo -= 1
        hi = zero
        while hi + 1 < counts.size and counts[hi + 1] >= floor_:
            hi += 1
        kl0, ku0 = zero - lo, hi - zero
    else:                                            # no populated diagonal at 0
        dense_d = np.flatnonzero(counts >= floor_)
        if dense_d.size:
            kl0 = max(0, zero - int(dense_d.min()))
            ku0 = max(0, int(dense_d.max()) - zero)
        else:
            kl0 = ku0 = 0
    bulk_kl, bulk_ku = kl0, ku0

    # --- 2. greedy cover of the out-of-band entries ------------------------
    viol = (off < -kl0) | (off > ku0)
    dense_rows, dense_cols = [], []
    # a candidate must stick out by more than a bandwidth's worth of entries to
    # count as a border; below that it is just an irregular bulk row
    dense_min = max(4, kl0 + ku0 + 1)
    capped = False
    while viol.any():
        if len(dense_rows) + len(dense_cols) >= max_dense:
            capped = True
            break
        vr, vc = row[viol], col[viol]
        rcount = np.bincount(vr, minlength=m)
        ccount = np.bincount(vc, minlength=n)
        best_r, best_c = int(rcount.argmax()), int(ccount.argmax())
        if rcount[best_r] >= ccount[best_c]:
            best, cover, is_row = best_r, int(rcount[best_r]), True
        else:
            best, cover, is_row = best_c, int(ccount[best_c]), False
        if cover < dense_min:
            break                                # not a border; widen instead
        if is_row:
            dense_rows.append(best)
            viol &= row != best
        else:
            dense_cols.append(best)
            viol &= col != best

    dense_rows = np.array(sorted(dense_rows), dtype=int)
    dense_cols = np.array(sorted(dense_cols), dtype=int)

    # --- 3. tighten the band on what is left -------------------------------
    keep = np.ones(row.size, dtype=bool)
    if dense_rows.size:
        keep &= ~np.isin(row, dense_rows)
    if dense_cols.size:
        keep &= ~np.isin(col, dense_cols)
    if keep.any():
        koff = off[keep]
        kl = int(max(0, -koff.min()))
        ku = int(max(0, koff.max()))
    else:
        kl = ku = 0
    widened = kl > bulk_kl or ku > bulk_ku

    if warn and (widened or capped):
        why = []
        if widened:
            why.append(f"band widened ({bulk_kl},{bulk_ku}) -> ({kl},{ku})")
        if capped:
            why.append(f"stopped at the {max_dense}-border cap")
        warnings.warn(
            "detect_structure: " + "; ".join(why)
            + " -- the matrix may not be cleanly almost-banded; pass kl/ku/"
              "dense_rows/dense_cols explicitly if this is wrong.",
            StructureWarning, stacklevel=2)

    return Structure(kl, ku, dense_rows, dense_cols, shape=shape, nnz=int(row.size),
                     widened=widened, capped=capped, bulk_kl=bulk_kl, bulk_ku=bulk_ku)
