"""Batched banded QR on the Apple GPU (Metal, via MLX).

The regime this exists for: a multidimensional spectral solve decouples, per
timestep, into one independent banded system per Fourier mode -- thousands of
same-shape "pencils".  A pivot-free factorisation is the right kernel for that,
because every system takes exactly the same instruction path: one thread per
system, no warp divergence, and the batch axis laid out fastest so neighbouring
threads read neighbouring floats.

Three things make this more than a demo:

**Factor once, solve many.**  :class:`BatchedBandQR` keeps the rotations, so a
timestepping loop factors on the first step and replays ``Q^H`` and the
back-substitution thereafter -- the same economics as the CPU path.

**fp32 speed, float64 accuracy.**  Apple GPUs have no float64.  Rather than
accept fp32 answers, the factorisation is fp32 and the *residual* is formed on
the host in float64, so mixed-precision iterative refinement recovers full
double accuracy in two or three cheap correction solves (valid while
``cond(A) * 2^-24`` stays below one, i.e. up to ``cond ~ 1e7``).  Unified memory
makes the round trip nearly free, and a float64 residual beats any compensated
fp32 one, so that is where the residual is computed.

**One compile.**  Kernels are cached on ``(kl, ku, dtype, variant)`` alone; ``n``,
the batch size and the RHS count are runtime arguments.  Changing batch size
between calls no longer triggers a Metal recompile.

Both float64 and complex128 inputs are accepted and demoted to fp32/complex64
for the GPU; refinement puts the accuracy back.
"""
import numpy as np

from ._band import band_matvec_batched

__all__ = [
    "is_available", "BatchedBandQR", "solve_batched", "solve_batched_shifted",
    "GPUUnavailable", "clear_kernel_cache",
]


class GPUUnavailable(RuntimeError):
    """MLX/Metal is not usable in this process."""


def _mx():
    try:
        import mlx.core as mx
    except ImportError as exc:                       # pragma: no cover
        raise GPUUnavailable(
            "the GPU backend needs mlx (`pip install mlx`) on a native-arm64 "
            "Python.  Use bandqr.batched.solve_batched(..., backend='numpy') "
            "for a portable batched path."
        ) from exc
    return mx


def is_available():
    """True if MLX is importable and a Metal device is present."""
    try:
        mx = _mx()
    except GPUUnavailable:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:                                # pragma: no cover
        return False


# --------------------------------------------------------------------------
# Metal source.  One algorithm body, two scalar types: `float` for real data and
# `float2` for complex, behind the same handful of inline helpers -- the same
# templating trick the C kernel uses, expressed in Metal.
#
# Every buffer is flat float32 and indexed by a *logical* element index; SLOAD /
# SSTORE apply the x2 stride for complex.  Layout puts the batch axis fastest,
# so lane s of a threadgroup touches float s of every load.
# --------------------------------------------------------------------------

_HEADER_REAL = """
typedef float SCALAR;
#define SLOAD(p, i)      ((p)[(i)])
/* value and index are bound to locals first: the macro must evaluate each
 * exactly once, or a store whose value reads the destination (as the
 * back-substitution's does) would see its own half-written result. */
#define SSTORE(p, i, v)  { ulong _si = (i); SCALAR _sv = (v); (p)[_si] = _sv; }
#define SZERO            (0.0f)
inline float  cmul(float a, float b)   { return a * b; }
inline float  cconj(float a)           { return a; }
inline float  cabs2(float a)           { return a * a; }
inline float  cscale(float t, float a) { return t * a; }
inline float  csub(float a, float b)   { return a - b; }
inline float  cadd(float a, float b)   { return a + b; }
inline float  cdiv(float a, float b)   { return a / b; }
inline void givens(float a, float b, thread float &c, thread float &s) {
    float bb2 = b * b;
    if (bb2 == 0.0f) { c = 1.0f; s = 0.0f; return; }
    float aa2 = a * a;
    float t = sqrt(aa2 + bb2);
    if (aa2 == 0.0f) { c = 0.0f; s = b / sqrt(bb2); return; }
    float aa = sqrt(aa2);
    c = aa / t;
    s = (a / aa) * b / t;
}
"""

_HEADER_CPLX = """
typedef float2 SCALAR;
#define SLOAD(p, i)      float2((p)[2*(i)], (p)[2*(i)+1])
/* value and index are bound to locals first -- see the real header: expanding
 * `v` twice here wrote the real part, then re-read the destination to compute
 * the imaginary one and got the value it had just overwritten. */
#define SSTORE(p, i, v)  { ulong _si = (i); SCALAR _sv = (v); \
                           (p)[2*_si] = _sv.x; (p)[2*_si+1] = _sv.y; }
#define SZERO            float2(0.0f, 0.0f)
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline float2 cconj(float2 a)            { return float2(a.x, -a.y); }
inline float  cabs2(float2 a)            { return a.x*a.x + a.y*a.y; }
inline float2 cscale(float t, float2 a)  { return float2(t*a.x, t*a.y); }
inline float2 csub(float2 a, float2 b)   { return a - b; }
inline float2 cadd(float2 a, float2 b)   { return a + b; }
inline float2 cdiv(float2 a, float2 b) {
    float d = cabs2(b);
    return float2((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d);
}
inline void givens(float2 a, float2 b, thread float &c, thread float2 &s) {
    float bb2 = cabs2(b);
    if (bb2 == 0.0f) { c = 1.0f; s = float2(0.0f, 0.0f); return; }
    float aa2 = cabs2(a);
    float t = sqrt(aa2 + bb2);
    if (aa2 == 0.0f) { c = 0.0f; s = cscale(1.0f/sqrt(bb2), cconj(b)); return; }
    float aa = sqrt(aa2);
    c = aa / t;
    s = cscale(1.0f/(aa*t), cmul(a, cconj(b)));
}
"""

# Shared prologue: decode the runtime dimensions and claim one system per thread.
_PROLOGUE = """
    const int KL = {KL}, KU = {KU};
    const int KUR = KL + KU, WIN = KL + KU + 1, W = 2*KL + KU + 2;
    const int N = dims[0], BATCH = dims[1], NRHS = dims[2];
    uint s = thread_position_in_grid.x;
    if (s >= (uint)BATCH) return;
    const ulong bs = (ulong)BATCH;
    const ulong sd = (ulong)s;
"""

# Load the compact input band into the wider working band (the extra guard slot
# absorbs the one-past-kuR fill an adjacent-row rotation produces).
_LOAD_BAND = """
    for (int i = 0; i < N; i++) {
        ulong wrow = (ulong)i * (ulong)W;
        ulong arow = (ulong)i * (ulong)WIN;
        for (int o = 0; o < W; o++) {
            SCALAR v = (o < WIN) ? SLOAD(ab, (arow + (ulong)o) * bs + sd) : SZERO;
            SSTORE(wab, (wrow + (ulong)o) * bs + sd, v);
        }
    }
"""

# Shifted variant: one band shared by every system, plus a per-system diagonal
# shift.  This is the spectral case -- the same z-operator for every wavenumber,
# differing only by a -k^2 term -- and it turns an (n, w, batch) upload into an
# (n, w) one.
_LOAD_BAND_SHIFTED = """
    SCALAR shift = SLOAD(shifts, sd);
    for (int i = 0; i < N; i++) {
        ulong wrow = (ulong)i * (ulong)W;
        ulong arow = (ulong)i * (ulong)WIN;
        for (int o = 0; o < W; o++) {
            SCALAR v = (o < WIN) ? SLOAD(ab, arow + (ulong)o) : SZERO;
            if (o == KL) v = cadd(v, shift);
            SSTORE(wab, (wrow + (ulong)o) * bs + sd, v);
        }
    }
"""

# The factorisation proper.  Rotation k is identified by its position in the
# (jc, q) nest, so the solve kernel can replay them without storing row indices.
_FACTOR_BODY = """
    ulong k = 0;
    for (int jc = 0; jc < N; jc++) {
        int ilast = min(jc + KL, N - 1);
        for (int q = ilast; q >= jc + 1; q--) {
            int p = q - 1;
            ulong ip = ((ulong)p * (ulong)W + (ulong)(jc - p + KL)) * bs + sd;
            ulong iq = ((ulong)q * (ulong)W + (ulong)(jc - q + KL)) * bs + sd;
            SCALAR a = SLOAD(wab, ip), bb = SLOAD(wab, iq);
            float c; SCALAR sn;
            givens(a, bb, c, sn);
            int jmax = min(q + KUR, N - 1);
            for (int j = jc; j <= jmax; j++) {
                ulong jp = ((ulong)p * (ulong)W + (ulong)(j - p + KL)) * bs + sd;
                ulong jq = ((ulong)q * (ulong)W + (ulong)(j - q + KL)) * bs + sd;
                SCALAR ap = SLOAD(wab, jp), aq = SLOAD(wab, jq);
                SSTORE(wab, jp, cadd(cscale(c, ap), cmul(sn, aq)));
                SSTORE(wab, jq, csub(cscale(c, aq), cmul(cconj(sn), ap)));
            }
            @STORE_ROT@
            k++;
        }
    }
"""

_STORE_ROT = """
            cvals[k * bs + sd] = c;
            SSTORE(svals, k * bs + sd, sn);
"""

# Replay Q^H against the right-hand sides, in exactly the factorisation's order.
_APPLY_QT = """
    ulong k = 0;
    for (int jc = 0; jc < N; jc++) {
        int ilast = min(jc + KL, N - 1);
        for (int q = ilast; q >= jc + 1; q--) {
            int p = q - 1;
            float  c  = cvals[k * bs + sd];
            SCALAR sn = SLOAD(svals, k * bs + sd);
            for (int t = 0; t < NRHS; t++) {
                ulong yp = ((ulong)p * (ulong)NRHS + (ulong)t) * bs + sd;
                ulong yq = ((ulong)q * (ulong)NRHS + (ulong)t) * bs + sd;
                SCALAR ap = SLOAD(y, yp), aq = SLOAD(y, yq);
                SSTORE(y, yp, cadd(cscale(c, ap), cmul(sn, aq)));
                SSTORE(y, yq, csub(cscale(c, aq), cmul(cconj(sn), ap)));
            }
            k++;
        }
    }
"""

# Banded back-substitution against R (which lives in wab).
_BACKSUB = """
    for (int i = N - 1; i >= 0; i--) {
        int jhi = min(i + KUR, N - 1);
        ulong wrow = (ulong)i * (ulong)W;
        for (int t = 0; t < NRHS; t++)
            SSTORE(x, ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd,
                   SLOAD(y, ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd));
        for (int j = i + 1; j <= jhi; j++) {
            SCALAR rij = SLOAD(wab, (wrow + (ulong)(j - i + KL)) * bs + sd);
            for (int t = 0; t < NRHS; t++) {
                ulong xi = ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd;
                ulong xj = ((ulong)j * (ulong)NRHS + (ulong)t) * bs + sd;
                SSTORE(x, xi, csub(SLOAD(x, xi), cmul(rij, SLOAD(x, xj))));
            }
        }
        SCALAR diag = SLOAD(wab, (wrow + (ulong)KL) * bs + sd);
        for (int t = 0; t < NRHS; t++) {
            ulong xi = ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd;
            SSTORE(x, xi, cdiv(SLOAD(x, xi), diag));
        }
    }
"""

_COPY_B_TO_Y = """
    for (int i = 0; i < N; i++)
        for (int t = 0; t < NRHS; t++) {
            ulong idx = ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd;
            SSTORE(y, idx, SLOAD(b, idx));
        }
"""

_KERNELS = {}


def clear_kernel_cache():
    """Drop compiled Metal kernels (they are cached for the process lifetime)."""
    _KERNELS.clear()


def _kernel(kind, kl, ku, cplx, shifted=False):
    key = (kind, kl, ku, cplx, shifted)
    if key in _KERNELS:
        return _KERNELS[key]
    mx = _mx()
    header = _HEADER_CPLX if cplx else _HEADER_REAL
    pro = _PROLOGUE.format(KL=kl, KU=ku)
    load = _LOAD_BAND_SHIFTED if shifted else _LOAD_BAND

    if kind == "factor":
        inputs = (["ab", "shifts", "dims"] if shifted else ["ab", "dims"])
        outputs = ["wab", "cvals", "svals"]
        src = pro + load + _FACTOR_BODY.replace("@STORE_ROT@", _STORE_ROT)
    elif kind == "solve":
        inputs = ["wab", "cvals", "svals", "b", "dims"]
        outputs = ["x", "y"]
        src = pro + _COPY_B_TO_Y + _APPLY_QT + _BACKSUB
    elif kind == "fused":
        # one-shot: factor and solve in a single pass, never writing the
        # rotations out (about 40% less store traffic than factor+solve)
        inputs = (["ab", "shifts", "b", "dims"] if shifted else ["ab", "b", "dims"])
        outputs = ["x", "wab", "y"]
        rot = """
            for (int t = 0; t < NRHS; t++) {
                ulong yp = ((ulong)p * (ulong)NRHS + (ulong)t) * bs + sd;
                ulong yq = ((ulong)q * (ulong)NRHS + (ulong)t) * bs + sd;
                SCALAR bp = SLOAD(y, yp), bq = SLOAD(y, yq);
                SSTORE(y, yp, cadd(cscale(c, bp), cmul(sn, bq)));
                SSTORE(y, yq, csub(cscale(c, bq), cmul(cconj(sn), bp)));
            }
"""
        src = (pro + load + _COPY_B_TO_Y
               + _FACTOR_BODY.replace("@STORE_ROT@", rot) + _BACKSUB)
    else:                                            # pragma: no cover
        raise ValueError(f"unknown kernel kind {kind!r}")

    kern = mx.fast.metal_kernel(
        name=f"bandqr_{kind}_{kl}_{ku}_{'z' if cplx else 'd'}{'_sh' if shifted else ''}",
        input_names=inputs,
        output_names=outputs,
        header=header,
        source=src,
    )
    _KERNELS[key] = kern
    return kern


def _nrot(n, kl):
    """Number of Givens rotations the factorisation performs, per system."""
    if n <= 0:
        return 0
    k = min(kl, n - 1)
    return k * (k - 1) // 2 + max(0, n - k) * k if n > k else k * (k - 1) // 2


def _flat32(a, cplx):
    """Flatten to a contiguous float32 buffer (complex -> interleaved re/im)."""
    a = np.ascontiguousarray(a)
    if cplx:
        return np.ascontiguousarray(a.astype(np.complex64).view(np.float32).ravel())
    return np.ascontiguousarray(a.astype(np.float32).ravel())


def _unflat(arr, shape, cplx):
    """Inverse of :func:`_flat32` for a kernel output."""
    a = np.asarray(arr, dtype=np.float32)
    if cplx:
        return a.view(np.complex64).reshape(shape)
    return a.reshape(shape)


class BatchedBandQR:
    """Batched banded QR factorisation held on the GPU: factor once, solve many.

    Parameters
    ----------
    ab : ndarray
        ``(n, kl+ku+1, batch)`` band storage, batch fastest.  For the shifted
        form pass ``(n, kl+ku+1)`` together with ``shifts``.
    kl, ku : int
        Bandwidths.
    shifts : ndarray, optional
        ``(batch,)`` per-system diagonal shift added to a single shared band.
        Avoids materialising ``batch`` copies of an operator that differs only
        by a scalar on the diagonal.
    threadgroup : int
        Metal threadgroup width.

    Notes
    -----
    The factorisation is fp32.  Keep the float64 ``ab`` you passed in if you
    intend to refine -- :meth:`solve` needs it to form residuals, and holds a
    reference for you.
    """

    def __init__(self, ab, kl, ku, shifts=None, threadgroup=256):
        mx = _mx()
        ab = np.asarray(ab)
        self.cplx = np.issubdtype(ab.dtype, np.complexfloating)
        self.kl, self.ku = int(kl), int(ku)
        self.shifted = shifts is not None
        self.threadgroup = int(threadgroup)

        if self.shifted:
            if ab.ndim != 2:
                raise ValueError("shifted form wants ab of shape (n, kl+ku+1)")
            n, w = ab.shape
            shifts = np.asarray(shifts).ravel()
            batch = shifts.size
            self.cplx = self.cplx or np.issubdtype(shifts.dtype, np.complexfloating)
        else:
            if ab.ndim != 3:
                raise ValueError("ab must have shape (n, kl+ku+1, batch)")
            n, w, batch = ab.shape
        if w != kl + ku + 1:
            raise ValueError(f"band width must be kl+ku+1 = {kl+ku+1}, got {w}")

        self.n, self.batch = int(n), int(batch)
        self.nrot = _nrot(self.n, self.kl)
        self.W = 2 * self.kl + self.ku + 2
        self._ab_host = ab                  # float64 original, for residuals
        self._shifts_host = shifts

        dims = mx.array([self.n, self.batch, 1, 0], dtype=mx.int32)
        ab_g = mx.array(_flat32(ab, self.cplx))
        comp = 2 if self.cplx else 1
        kern = _kernel("factor", self.kl, self.ku, self.cplx, self.shifted)
        inputs = [ab_g]
        if self.shifted:
            inputs.append(mx.array(_flat32(shifts, self.cplx)))
        inputs.append(dims)

        wab, cvals, svals = kern(
            inputs=inputs,
            output_shapes=[(self.n * self.W * self.batch * comp,),
                           (self.nrot * self.batch,),
                           (self.nrot * self.batch * comp,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(self.batch, 1, 1),
            threadgroup=(min(self.threadgroup, self.batch), 1, 1),
        )
        mx.eval(wab, cvals, svals)
        self._wab, self._cvals, self._svals = wab, cvals, svals

    # ---- low-level fp32 solve -------------------------------------------
    def solve_fp32(self, b):
        """One batched fp32 solve against the stored factors.

        ``b`` is ``(n, batch)`` or ``(n, nrhs, batch)``; the return matches.
        This is the raw GPU answer -- see :meth:`solve` for the refined one.
        """
        mx = _mx()
        b = np.asarray(b)
        squeeze = b.ndim == 2
        if squeeze:
            b = b[:, None, :]
        n, nrhs, batch = b.shape
        if n != self.n or batch != self.batch:
            raise ValueError(
                f"b has shape (n={n}, batch={batch}) but the factorisation is "
                f"(n={self.n}, batch={self.batch})")
        comp = 2 if self.cplx else 1
        dims = mx.array([self.n, self.batch, nrhs, 0], dtype=mx.int32)
        kern = _kernel("solve", self.kl, self.ku, self.cplx)
        x, _ = kern(
            inputs=[self._wab, self._cvals, self._svals,
                    mx.array(_flat32(b, self.cplx)), dims],
            output_shapes=[(self.n * nrhs * self.batch * comp,),
                           (self.n * nrhs * self.batch * comp,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(self.batch, 1, 1),
            threadgroup=(min(self.threadgroup, self.batch), 1, 1),
        )
        mx.eval(x)
        out = _unflat(x, (self.n, nrhs, self.batch), self.cplx)
        return out[:, 0, :] if squeeze else out

    # ---- residual against the original float64 operator ------------------
    def _dense_band(self):
        """The per-system float64 band, materialising a shifted one if needed."""
        if not self.shifted:
            return self._ab_host
        ab = np.repeat(self._ab_host[:, :, None], self.batch, axis=2).astype(
            np.result_type(self._ab_host.dtype, self._shifts_host.dtype))
        ab[:, self.kl, :] += self._shifts_host[None, :]
        return ab

    def residual(self, x, b):
        """``b - A x`` in float64, against the operator as originally given."""
        ab = self._dense_band()
        x = np.asarray(x)
        squeeze = x.ndim == 2
        X = x[:, None, :] if squeeze else x
        B = np.asarray(b)
        B = B[:, None, :] if squeeze else B
        out = np.empty_like(np.asarray(X, dtype=np.result_type(ab.dtype, X.dtype)))
        for t in range(X.shape[1]):
            out[:, t, :] = B[:, t, :] - band_matvec_batched(ab, self.kl, self.ku,
                                                            X[:, t, :])
        return out[:, 0, :] if squeeze else out

    # ---- the accurate solve ---------------------------------------------
    def solve(self, b, refine=2, tol=1e-13, return_info=False):
        """Solve, then refine against float64 residuals to recover accuracy.

        The factorisation is fp32, so a bare :meth:`solve_fp32` is accurate to
        roughly ``1e-7``.  Each refinement step forms ``b - A x`` in float64 on
        the host and puts the correction through the *same* fp32 factors, which
        converges to float64 accuracy while ``cond(A) * 2^-24 < 1``.

        Parameters
        ----------
        refine : int
            Correction steps.  ``0`` returns the raw fp32 solve.  Two is
            normally enough; a third is cheap insurance for stiffer operators.
        tol : float
            Stop once ``max ||r|| / ||b||`` over the batch falls below this.
        return_info : bool
            Also return a dict with the per-step residual history.
        """
        b = np.asarray(b)
        x = np.asarray(self.solve_fp32(b), dtype=np.float64 if not self.cplx
                       else np.complex128)
        hist = []
        bn = np.linalg.norm(np.atleast_2d(b.reshape(self.n, -1)), axis=0)
        bn = np.maximum(bn, np.finfo(float).tiny)
        for _ in range(int(refine)):
            r = self.residual(x, b)
            rel = float(np.max(np.linalg.norm(r.reshape(self.n, -1), axis=0) / bn))
            hist.append(rel)
            if rel <= tol:
                break
            x = x + self.solve_fp32(r)
        if return_info:
            r = self.residual(x, b)
            rel = float(np.max(np.linalg.norm(r.reshape(self.n, -1), axis=0) / bn))
            hist.append(rel)
            return x, {"residual_history": hist, "refine": int(refine)}
        return x

    def __repr__(self):
        kind = "complex64" if self.cplx else "float32"
        sh = ", shifted" if self.shifted else ""
        return (f"<BatchedBandQR n={self.n} batch={self.batch} "
                f"kl={self.kl} ku={self.ku} {kind}{sh}>")


def solve_batched(ab, b, kl, ku, refine=2, threadgroup=256, fused=None,
                  return_info=False):
    """Solve ``batch`` independent banded systems at once on the GPU.

    Parameters
    ----------
    ab : ndarray ``(n, kl+ku+1, batch)``
        Band storage, batch fastest.  float64/float32 or complex.
    b : ndarray ``(n, batch)`` or ``(n, nrhs, batch)``
        Right-hand sides.
    refine : int
        Mixed-precision refinement steps (see :meth:`BatchedBandQR.solve`).
        ``0`` gives the raw fp32 answer and skips the host round trip.
    fused : bool, optional
        Use the single-pass factor+solve kernel, which never writes the
        rotations out.  Defaults to True exactly when it is safe to do so --
        one RHS and no refinement, the only case where the factors are not
        needed again.

    Returns
    -------
    ndarray
        Same shape as ``b``, in float64 (or complex128) when refined.
    """
    b = np.asarray(b)
    single = b.ndim == 2
    if fused is None:
        fused = (refine == 0) and single
    if fused:
        if not single:
            raise ValueError("fused=True supports a single RHS per system")
        return _solve_fused(ab, b, kl, ku, threadgroup=threadgroup)
    F = BatchedBandQR(ab, kl, ku, threadgroup=threadgroup)
    return F.solve(b, refine=refine, return_info=return_info)


def solve_batched_shifted(ab, shifts, b, kl, ku, refine=2, threadgroup=256,
                          return_info=False):
    """Batched solve for one shared band plus a per-system diagonal shift.

    ``A_s = A + shifts[s] I``.  This is the shape a separable spectral solve
    produces -- the same operator in the bounded direction for every Fourier
    mode, differing only by ``-k^2`` on the diagonal -- and it uploads an
    ``(n, w)`` band instead of ``(n, w, batch)``.

    Parameters
    ----------
    ab : ndarray ``(n, kl+ku+1)``
        The shared band.
    shifts : ndarray ``(batch,)``
        Per-system diagonal shift.
    """
    F = BatchedBandQR(ab, kl, ku, shifts=shifts, threadgroup=threadgroup)
    return F.solve(b, refine=refine, return_info=return_info)


def _solve_fused(ab, b, kl, ku, threadgroup=256):
    """Single-pass factor+solve, fp32 throughout, no reusable factors."""
    mx = _mx()
    ab = np.asarray(ab)
    b = np.asarray(b)
    cplx = (np.issubdtype(ab.dtype, np.complexfloating)
            or np.issubdtype(b.dtype, np.complexfloating))
    n, w, batch = ab.shape
    if w != kl + ku + 1:
        raise ValueError(f"band width must be kl+ku+1 = {kl+ku+1}, got {w}")
    if b.shape != (n, batch):
        raise ValueError(f"b must have shape {(n, batch)}, got {b.shape}")
    comp = 2 if cplx else 1
    W = 2 * kl + ku + 2
    dims = mx.array([n, batch, 1, 0], dtype=mx.int32)
    kern = _kernel("fused", kl, ku, cplx)
    x, _, _ = kern(
        inputs=[mx.array(_flat32(ab, cplx)), mx.array(_flat32(b, cplx)), dims],
        output_shapes=[(n * batch * comp,), (n * W * batch * comp,),
                       (n * batch * comp,)],
        output_dtypes=[mx.float32] * 3,
        grid=(batch, 1, 1),
        threadgroup=(min(threadgroup, batch), 1, 1),
    )
    mx.eval(x)
    return _unflat(x, (n, batch), cplx)
