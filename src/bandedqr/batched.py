"""Batched banded QR: many independent same-shape systems at once.

Backend-neutral front end.  The ``"metal"`` backend is implemented in
:mod:`bandedqr.gpu` (with ``"gpu"`` retained as a compatibility alias); the
``"numpy"`` backend here is a vectorised host
implementation that runs everywhere, needs no MLX, and works in the input dtype
(so float64 in, float64 out, with no refinement needed).

The NumPy backend vectorises across the *batch* axis: the rotation sequence is
identical for every system -- that is exactly what pivot-freedom buys -- so each
rotation is one array operation over all systems at once.  Its Python overhead
scales with ``n * kl`` rotations rather than with the batch, which makes it good
for the regime the GPU targets (moderate ``n``, large batch) and poor for a few
very long systems (use :class:`bandedqr.BandQR` per system for those).

It also serves as the reference the GPU kernel is tested against: same rotation
formula, same order, so ``R`` agrees to fp32 rounding.
"""
import numpy as np

from ._band import band_matvec_batched

__all__ = ["BatchedBandQR", "solve_batched", "available_backends"]


def available_backends():
    """Runnable public backend selectors, most capable first.

    The returned identities are concrete enough for dispatch.  Reproducible
    run records use the more specific implementation IDs ``metal-mlx`` and
    ``cpu-numpy-batched``.
    """
    out = []
    from . import gpu
    if gpu.is_available():
        out.append("metal")
    out.append("numpy")
    return out


def _givens_batched(a, b):
    """Vectorised unitary Givens, matching the C and Metal kernels exactly.

    Returns ``(c, s)`` with ``c`` real: the rotation ``[[c, s], [-conj(s), c]]``
    annihilates ``b``.
    """
    cplx = np.iscomplexobj(a) or np.iscomplexobj(b)
    bb2 = (a.real * 0 + np.abs(b) ** 2)
    aa2 = np.abs(a) ** 2
    t = np.sqrt(aa2 + bb2)
    c = np.ones(a.shape, dtype=np.float64)
    s = np.zeros(a.shape, dtype=np.complex128 if cplx else np.float64)

    nz = bb2 > 0                                  # b == 0 -> identity rotation
    deg = nz & (aa2 == 0)                         # a == 0 -> pure swap
    reg = nz & (aa2 > 0)
    if deg.any():
        c[deg] = 0.0
        s[deg] = np.conj(b[deg]) / np.sqrt(bb2[deg])
    if reg.any():
        aa = np.sqrt(aa2[reg])
        c[reg] = aa / t[reg]
        s[reg] = (a[reg] / aa) * np.conj(b[reg]) / t[reg]
    return c, s


class BatchedBandQR:
    """Vectorised host batched banded QR: factor once, solve many.

    Parameters
    ----------
    ab : ndarray ``(n, kl+ku+1, batch)``
        Band storage with the batch axis last.
    shifts : ndarray ``(batch,)``, optional
        Per-system diagonal shift applied to a single shared ``(n, kl+ku+1)``
        band -- the same convenience the GPU backend offers.
    """

    def __init__(self, ab, kl, ku, shifts=None):
        ab = np.asarray(ab)
        self.kl, self.ku = int(kl), int(ku)
        self.kuR = self.kl + self.ku
        if shifts is not None:
            shifts = np.asarray(shifts).ravel()
            if ab.ndim != 2:
                raise ValueError("shifted form wants ab of shape (n, kl+ku+1)")
            dt = np.result_type(ab.dtype, shifts.dtype, np.float64)
            n, w = ab.shape
            batch = shifts.size
            full = np.repeat(ab[:, :, None].astype(dt), batch, axis=2)
            full[:, self.kl, :] += shifts[None, :]
            ab = full
        else:
            if ab.ndim != 3:
                raise ValueError("ab must have shape (n, kl+ku+1, batch)")
            n, w, batch = ab.shape
            dt = np.result_type(ab.dtype, np.float64)
        if w != kl + ku + 1:
            raise ValueError(f"band width must be kl+ku+1 = {kl+ku+1}, got {w}")

        self.n, self.batch = int(n), int(batch)
        self.dtype = dt
        self._ab = np.ascontiguousarray(ab, dtype=dt)
        self.W = 2 * self.kl + self.ku + 2

        wab = np.zeros((self.n, self.W, self.batch), dtype=dt)
        wab[:, :w, :] = self._ab
        crot, srot = [], []
        for jc in range(self.n):
            ilast = min(jc + self.kl, self.n - 1)
            for q in range(ilast, jc, -1):
                p = q - 1
                a = wab[p, jc - p + self.kl, :]
                bv = wab[q, jc - q + self.kl, :]
                c, s = _givens_batched(a, bv)
                jmax = min(q + self.kuR, self.n - 1)
                L = jmax - jc + 1
                op = jc - p + self.kl
                oq = jc - q + self.kl
                rp = wab[p, op:op + L, :].copy()
                rq = wab[q, oq:oq + L, :].copy()
                wab[p, op:op + L, :] = c * rp + s * rq
                wab[q, oq:oq + L, :] = c * rq - np.conj(s) * rp
                crot.append(c)
                srot.append(s.astype(dt, copy=False))
        self._wab = wab
        self._c = np.array(crot) if crot else np.zeros((0, self.batch))
        self._s = (np.array(srot) if srot
                   else np.zeros((0, self.batch), dtype=dt))

    def solve(self, b, refine=0, **_):
        """Solve for ``b`` of shape ``(n, batch)`` or ``(n, nrhs, batch)``.

        ``refine`` is accepted and ignored: this backend already works in the
        input precision, so there is nothing for refinement to recover.
        """
        b = np.asarray(b)
        squeeze = b.ndim == 2
        B = b[:, None, :] if squeeze else b
        if B.shape[0] != self.n or B.shape[2] != self.batch:
            raise ValueError(
                f"b has shape {b.shape}, incompatible with n={self.n}, "
                f"batch={self.batch}")
        y = np.array(B, dtype=np.result_type(self.dtype, B.dtype), copy=True)

        k = 0
        for jc in range(self.n):
            ilast = min(jc + self.kl, self.n - 1)
            for q in range(ilast, jc, -1):
                p = q - 1
                c = self._c[k][None, :]
                s = self._s[k][None, :]
                yp, yq = y[p].copy(), y[q].copy()
                y[p] = c * yp + s * yq
                y[q] = c * yq - np.conj(s) * yp
                k += 1

        x = np.zeros_like(y)
        for i in range(self.n - 1, -1, -1):
            jhi = min(i + self.kuR, self.n - 1)
            acc = y[i].copy()
            for j in range(i + 1, jhi + 1):
                acc -= self._wab[i, j - i + self.kl, :][None, :] * x[j]
            x[i] = acc / self._wab[i, self.kl, :][None, :]
        return x[:, 0, :] if squeeze else x

    def residual(self, x, b):
        """``b - A x`` in the working dtype."""
        squeeze = np.asarray(x).ndim == 2
        X = np.asarray(x)[:, None, :] if squeeze else np.asarray(x)
        B = np.asarray(b)[:, None, :] if squeeze else np.asarray(b)
        out = np.empty_like(X)
        for t in range(X.shape[1]):
            out[:, t, :] = B[:, t, :] - band_matvec_batched(
                self._ab, self.kl, self.ku, X[:, t, :])
        return out[:, 0, :] if squeeze else out

    def __repr__(self):
        return (f"<BatchedBandQR[numpy] n={self.n} batch={self.batch} "
                f"kl={self.kl} ku={self.ku} {np.dtype(self.dtype).name}>")


def solve_batched(ab, b, kl, ku, backend="auto", shifts=None, refine=2, **kwargs):
    """Solve many independent banded systems at once.

    Parameters
    ----------
    ab : ndarray
        ``(n, kl+ku+1, batch)``, or ``(n, kl+ku+1)`` when ``shifts`` is given.
    b : ndarray
        ``(n, batch)`` or ``(n, nrhs, batch)``.
    backend : {"auto", "metal", "numpy"}
        ``"auto"`` uses Metal when MLX and a working Metal device are available
        and the batch is big enough to be worth the dispatch, else NumPy.
        ``"gpu"`` remains a compatibility alias for ``"metal"``.  ``"cuda"``
        is reserved for the planned CUDA backend and is not yet implemented.
    shifts : ndarray, optional
        ``(batch,)`` per-system diagonal shift on a shared band.
    refine : int
        Mixed-precision refinement steps (GPU backend only; the NumPy backend
        already works in float64).
    """
    if backend == "auto":
        from . import gpu
        batch = (np.asarray(shifts).size if shifts is not None
                 else np.asarray(ab).shape[-1])
        backend = "metal" if (gpu.is_available() and batch >= 64) else "numpy"
    if backend == "gpu":
        backend = "metal"
    if backend == "metal":
        from . import gpu
        if not gpu.is_available():
            raise gpu.GPUUnavailable(
                "the Metal backend is not usable in this process; choose "
                "backend='numpy' or run on a native Apple Metal session")
        if shifts is not None:
            return gpu.solve_batched_shifted(ab, shifts, b, kl, ku, refine=refine,
                                             **kwargs)
        return gpu.solve_batched(ab, b, kl, ku, refine=refine, **kwargs)
    if backend == "numpy":
        return BatchedBandQR(ab, kl, ku, shifts=shifts).solve(b)
    if backend == "cuda":
        raise NotImplementedError(
            "the CUDA backend is reserved but not implemented; use 'metal' "
            "or 'numpy'")
    raise ValueError(
        f"unknown backend {backend!r}; expected auto, metal, numpy, or cuda")
