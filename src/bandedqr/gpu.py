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
accept fp32 answers, the factorisation is fp32 and the *residual* is formed in
float64 against the original operator, so mixed-precision iterative refinement
recovers full double accuracy in two or three cheap correction solves -- valid
while ``cond(A) * 2^-24`` stays below one, i.e. up to ``cond ~ 1e7``.  Measured:
``2e-7`` raw, ``5e-14`` after one correction.

The residual is computed on the host, which profiling justifies but not for the
reason you might expect: the device round trip is genuinely cheap on unified
memory (~0.3 ms of a 8 ms step), while *forming* the residual was the expensive
part, so it runs in the C kernel rather than in numpy (5.4 ms -> 1.8 ms at
n=192, batch=4096).  It has to be the float64 band: refining against the fp32
copy the GPU factored would converge to the fp32 matrix's answer and cap
accuracy at the 1e-7 this exists to beat.

**One compile.**  Kernels are cached on ``(kl, ku, dtype, variant)`` alone; ``n``,
the batch size and the RHS count are runtime arguments.  Changing batch size
between calls no longer triggers a Metal recompile.

Both float64 and complex128 inputs are accepted and demoted to fp32/complex64
for the GPU; refinement puts the accuracy back.
"""
from functools import lru_cache
import importlib.util
import subprocess
import sys

import numpy as np

from . import _native
from ._band import band_matvec_batched

__all__ = [
    "is_available", "BatchedBandQR", "BatchedRideQR", "solve_batched",
    "solve_batched_shifted", "GPUUnavailable", "clear_kernel_cache",
]


class GPUUnavailable(RuntimeError):
    """MLX/Metal is not usable in this process."""


def _mx():
    try:
        import mlx.core as mx
    except ImportError as exc:                       # pragma: no cover
        raise GPUUnavailable(
            "the GPU backend needs mlx (`pip install mlx`) on a native-arm64 "
            "Python.  Use bandedqr.batched.solve_batched(..., backend='numpy') "
            "for a portable batched path."
        ) from exc
    return mx


@lru_cache(maxsize=1)
def is_available():
    """True if this process can execute a trivial MLX operation on Metal.

    ``mx.metal.is_available()`` can report a device in a headless or restricted
    process that cannot actually create a command queue.  The tiny evaluated
    allocation makes backend discovery an execution check instead of a device
    enumeration check.  It runs in an isolated child because a failed MLX
    device initialization can otherwise leave a noisy shutdown callback in the
    caller.  The result is cached for the process lifetime.
    """
    if importlib.util.find_spec("mlx") is None:
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mlx.core as mx; "
                    "assert mx.metal.is_available(); "
                    "x = mx.zeros((1,), dtype=mx.float32); mx.eval(x)"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):     # pragma: no cover
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

# --------------------------------------------------------------------------
# Almost-banded ("ride") batched kernels.  M = B + U L, with the dense boundary
# rows carried through the rotations as a rank-r update rather than sparsified
# first, so R = Q^H M exactly.  L (the rows' out-of-band content) is shared
# across the batch -- boundary conditions do not vary with the Fourier mode --
# while the bulk B does.  RFILL is a compile-time constant so the running
# low-rank accumulator lives in registers.
# --------------------------------------------------------------------------

_INIT_U = """
    for (int i = 0; i < N; i++)
        for (int t = 0; t < RFILL; t++) {
            bool hit = (i == drows[t]);
            SSTORE(U, ((ulong)i * (ulong)RFILL + (ulong)t) * bs + sd,
                   hit ? SONE : SZERO);
        }
"""

_FACTOR_AB_BODY = """
    ulong k = 0;
    for (int jc = 0; jc < N; jc++) {
        int ilast = min(jc + KL, N - 1);
        for (int q = ilast; q >= jc + 1; q--) {
            int p = q - 1;
            ulong ip = ((ulong)p * (ulong)W + (ulong)(jc - p + KL)) * bs + sd;
            ulong iq = ((ulong)q * (ulong)W + (ulong)(jc - q + KL)) * bs + sd;
            /* the true entries include the carried low-rank part */
            SCALAR a = SLOAD(wab, ip), bb = SLOAD(wab, iq);
            for (int t = 0; t < RFILL; t++) {
                SCALAR ljc = SLOAD(Lmat, (ulong)t * (ulong)N + (ulong)jc);
                a  = cadd(a,  cmul(SLOAD(U, ((ulong)p * (ulong)RFILL + (ulong)t) * bs + sd), ljc));
                bb = cadd(bb, cmul(SLOAD(U, ((ulong)q * (ulong)RFILL + (ulong)t) * bs + sd), ljc));
            }
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
            for (int t = 0; t < RFILL; t++) {          /* U rides along */
                ulong up = ((ulong)p * (ulong)RFILL + (ulong)t) * bs + sd;
                ulong uq = ((ulong)q * (ulong)RFILL + (ulong)t) * bs + sd;
                SCALAR u1 = SLOAD(U, up), u2 = SLOAD(U, uq);
                SSTORE(U, up, cadd(cscale(c, u1), cmul(sn, u2)));
                SSTORE(U, uq, csub(cscale(c, u2), cmul(cconj(sn), u1)));
            }
            cvals[k * bs + sd] = c;
            SSTORE(svals, k * bs + sd, sn);
            k++;
        }
    }
"""

# Back-substitution against R = R_band + U L: a banded sweep plus a running
# r-vector g = sum_{j>i} L[:,j] x[j] carrying the low-rank fill.  One RHS at a
# time so g stays in registers.
_BACKSUB_AB = """
    for (int t = 0; t < NRHS; t++) {
        SCALAR g[RFILL];
        for (int u = 0; u < RFILL; u++) g[u] = SZERO;
        for (int i = N - 1; i >= 0; i--) {
            int jhi = min(i + KUR, N - 1);
            ulong wrow = (ulong)i * (ulong)W;
            SCALAR acc = SLOAD(y, ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd);
            for (int j = i + 1; j <= jhi; j++)
                acc = csub(acc, cmul(SLOAD(wab, (wrow + (ulong)(j - i + KL)) * bs + sd),
                                     SLOAD(x, ((ulong)j * (ulong)NRHS + (ulong)t) * bs + sd)));
            SCALAR diag = SLOAD(wab, (wrow + (ulong)KL) * bs + sd);
            for (int u = 0; u < RFILL; u++) {
                SCALAR Uiu = SLOAD(U, ((ulong)i * (ulong)RFILL + (ulong)u) * bs + sd);
                acc  = csub(acc, cmul(Uiu, g[u]));
                diag = cadd(diag, cmul(Uiu, SLOAD(Lmat, (ulong)u * (ulong)N + (ulong)i)));
            }
            SCALAR xi = cdiv(acc, diag);
            SSTORE(x, ((ulong)i * (ulong)NRHS + (ulong)t) * bs + sd, xi);
            for (int u = 0; u < RFILL; u++)
                g[u] = cadd(g[u], cmul(SLOAD(Lmat, (ulong)u * (ulong)N + (ulong)i), xi));
        }
    }
"""

_KERNELS = {}


def clear_kernel_cache():
    """Drop compiled Metal kernels (they are cached for the process lifetime)."""
    _KERNELS.clear()


def _kernel(kind, kl, ku, cplx, shifted=False, rfill=0):
    key = (kind, kl, ku, cplx, shifted, rfill)
    if key in _KERNELS:
        return _KERNELS[key]
    mx = _mx()
    header = _HEADER_CPLX if cplx else _HEADER_REAL
    header += ("\n#define SONE float2(1.0f, 0.0f)\n" if cplx
               else "\n#define SONE (1.0f)\n")
    pro = _PROLOGUE.format(KL=kl, KU=ku)
    if rfill:
        pro += f"    const int RFILL = {rfill};\n"
    load = _LOAD_BAND_SHIFTED if shifted else _LOAD_BAND

    if kind == "factor_ab":
        inputs = ["ab", "Lmat", "drows", "dims"]
        outputs = ["wab", "U", "cvals", "svals"]
        src = pro + _LOAD_BAND + _INIT_U + _FACTOR_AB_BODY
    elif kind == "solve_ab":
        inputs = ["wab", "U", "Lmat", "cvals", "svals", "b", "dims"]
        outputs = ["x", "y"]
        src = pro + _COPY_B_TO_Y + _APPLY_QT + _BACKSUB_AB
    elif kind == "factor":
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
        name=(f"bandedqr_{kind}_{kl}_{ku}_{'z' if cplx else 'd'}"
              f"{'_sh' if shifted else ''}{f'_r{rfill}' if rfill else ''}"),
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
        """``b - A x`` in float64, against the operator as originally given.

        Deliberately evaluated against the *float64* band rather than the fp32
        copy the GPU factored: refining against the demoted operator would
        converge to the fp32 matrix's solution and cap the accuracy at ~1e-7,
        which is the whole thing this is trying to beat.

        Runs in the C kernel, which accumulates each output row across all its
        diagonals in one pass.  The equivalent numpy expression allocates an
        ``(n, batch)`` temporary per diagonal and measured ~2.3x slower.
        """
        ab = np.ascontiguousarray(self._dense_band())
        cplx = np.iscomplexobj(ab) or np.iscomplexobj(x) or np.iscomplexobj(b)
        dt = np.complex128 if cplx else np.float64
        x = np.asarray(x)
        squeeze = x.ndim == 2
        X = np.ascontiguousarray(x[:, None, :] if squeeze else x, dt)
        B = np.asarray(b)
        B = np.ascontiguousarray(B[:, None, :] if squeeze else B, dt)
        ab = np.ascontiguousarray(ab, dt)

        w = self.kl + self.ku + 1
        fn = getattr(_native.lib, "bandedqr_band_residual_batched"
                     + _native.suffix_for(cplx))
        p = _native.ptr_for(cplx)
        out = np.empty_like(X)
        for t in range(X.shape[1]):
            xt = np.ascontiguousarray(X[:, t, :])
            bt = np.ascontiguousarray(B[:, t, :])
            rt = np.empty_like(xt)
            fn(p(ab), self.n, w, self.batch, self.kl, self.ku, p(xt), p(bt), p(rt))
            out[:, t, :] = rt
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


class BatchedRideQR:
    """Batched *almost*-banded QR on the GPU: banded bulk plus dense rows.

    The shape a real spectral operator has -- a banded differential operator
    plus a couple of dense boundary-condition rows -- for many modes at once.
    The dense rows are carried through the rotations as a rank-r update rather
    than sparsified away first, so ``R = Q^H M`` exactly.

    The boundary rows are shared across the batch (they do not depend on the
    Fourier mode) while the bulk varies, which is what makes this cheap: ``L``
    is uploaded once as ``(r, n)``.

    Parameters
    ----------
    ab : ndarray ``(n, kl+ku+1, batch)``
        Band storage of the whole matrix, including the in-band part of the
        dense rows.  ``kl`` must reach the dense rows -- see
        :meth:`from_dense_rows`, which arranges that for you.
    dense_rows : sequence of int
        Row indices of the boundary rows, identical across the batch, each
        ``<= kl`` so the rotations can reach them.
    L : ndarray ``(r, n)``
        The dense rows' *out-of-band* content only; in-band entries live in
        ``ab`` and must be zero here.

    Notes
    -----
    Accuracy behaves like the dense-banded path: fp32 factors refined against
    float64 residuals.  Refinement needs the full operator, so pass
    ``M_dense``/``M_sparse`` to :meth:`solve` or use :meth:`from_dense_rows`,
    which retains what it needs.
    """

    def __init__(self, ab, kl, ku, dense_rows, L, threadgroup=256,
                 _full_rows=None):
        mx = _mx()
        ab = np.asarray(ab)
        L = np.asarray(L)
        dr = np.asarray(dense_rows, dtype=np.int32).ravel()
        if ab.ndim != 3:
            raise ValueError("ab must have shape (n, kl+ku+1, batch)")
        n, w, batch = ab.shape
        if w != kl + ku + 1:
            raise ValueError(f"band width must be kl+ku+1 = {kl+ku+1}, got {w}")
        if dr.size == 0:
            raise ValueError("BatchedRideQR needs at least one dense row; use "
                             "BatchedBandQR for a purely banded batch")
        if int(dr.max()) > kl:
            raise ValueError(
                f"dense rows must lie within kl of the top (max index "
                f"{int(dr.max())} > kl={kl}); widen kl or use from_dense_rows")
        if L.shape != (dr.size, n):
            raise ValueError(f"L must have shape {(dr.size, n)}, got {L.shape}")

        self.cplx = bool(np.issubdtype(ab.dtype, np.complexfloating)
                         or np.issubdtype(L.dtype, np.complexfloating))
        self.n, self.batch, self.kl, self.ku = n, batch, int(kl), int(ku)
        self.rfill = int(dr.size)
        self.W = 2 * self.kl + self.ku + 2
        self.nrot = _nrot(self.n, self.kl)
        self.threadgroup = int(threadgroup)
        self.dense_rows = dr
        self._ab_host, self._L_host = ab, L
        self._full_rows = _full_rows            # for float64 residuals

        comp = 2 if self.cplx else 1
        dims = mx.array([n, batch, 1, 0], dtype=mx.int32)
        kern = _kernel("factor_ab", self.kl, self.ku, self.cplx, rfill=self.rfill)
        wab, U, cvals, svals = kern(
            inputs=[mx.array(_flat32(ab, self.cplx)),
                    mx.array(_flat32(L, self.cplx)),
                    mx.array(dr.astype(np.int32)), dims],
            output_shapes=[(n * self.W * batch * comp,),
                           (n * self.rfill * batch * comp,),
                           (self.nrot * batch,),
                           (self.nrot * batch * comp,)],
            output_dtypes=[mx.float32] * 4,
            grid=(batch, 1, 1),
            threadgroup=(min(self.threadgroup, batch), 1, 1),
        )
        mx.eval(wab, U, cvals, svals)
        self._wab, self._U, self._cvals, self._svals = wab, U, cvals, svals
        self._Lg = mx.array(_flat32(L, self.cplx))

    @classmethod
    def from_dense_rows(cls, bulk_ab, kl, ku, dense_rows, rows, threadgroup=256):
        """Build from a banded bulk and the *full* dense boundary rows.

        Handles the bookkeeping the raw constructor demands: widens ``kl`` far
        enough for the rotations to reach the boundary rows, writes their
        in-band entries into the band, and splits off the out-of-band remainder
        as ``L``.

        Parameters
        ----------
        bulk_ab : ndarray ``(n, kl+ku+1, batch)``
            The banded bulk, with the dense rows' slots ignored.
        rows : ndarray ``(r, n)``
            The complete boundary rows.
        """
        bulk_ab = np.asarray(bulk_ab)
        rows = np.atleast_2d(np.asarray(rows))
        dr = np.asarray(dense_rows, dtype=int).ravel()
        n, w, batch = bulk_ab.shape
        kle = max(int(kl), int(dr.max()))
        dt = np.result_type(bulk_ab.dtype, rows.dtype)

        ab = np.zeros((n, kle + ku + 1, batch), dtype=dt)
        ab[:, (kle - kl):(kle - kl) + w, :] = bulk_ab      # re-align for wider kl
        L = np.array(rows, dtype=dt)
        for t, i in enumerate(dr):
            lo, hi = max(0, i - kle), min(n - 1, i + ku)
            ab[i, :, :] = 0.0
            ab[i, lo - i + kle:hi - i + kle + 1, :] = rows[t, lo:hi + 1][:, None]
            L[t, lo:hi + 1] = 0.0                          # keep out-of-band only
        return cls(ab, kle, ku, dr, L, threadgroup=threadgroup,
                   _full_rows=(rows, dr))

    def solve_fp32(self, b):
        """One batched fp32 solve against the stored almost-banded factors."""
        mx = _mx()
        b = np.asarray(b)
        squeeze = b.ndim == 2
        B = b[:, None, :] if squeeze else b
        n, nrhs, batch = B.shape
        if n != self.n or batch != self.batch:
            raise ValueError(f"b has (n={n}, batch={batch}) but the factorisation "
                             f"is (n={self.n}, batch={self.batch})")
        comp = 2 if self.cplx else 1
        dims = mx.array([self.n, self.batch, nrhs, 0], dtype=mx.int32)
        kern = _kernel("solve_ab", self.kl, self.ku, self.cplx, rfill=self.rfill)
        x, _ = kern(
            inputs=[self._wab, self._U, self._Lg, self._cvals, self._svals,
                    mx.array(_flat32(B, self.cplx)), dims],
            output_shapes=[(self.n * nrhs * self.batch * comp,)] * 2,
            output_dtypes=[mx.float32, mx.float32],
            grid=(self.batch, 1, 1),
            threadgroup=(min(self.threadgroup, self.batch), 1, 1),
        )
        mx.eval(x)
        out = _unflat(x, (self.n, nrhs, self.batch), self.cplx)
        return out[:, 0, :] if squeeze else out

    def residual(self, x, b):
        """``b - M x`` in float64, band plus the dense rows put back."""
        ab = np.ascontiguousarray(self._ab_host)
        cplx = np.iscomplexobj(ab) or np.iscomplexobj(x) or np.iscomplexobj(b)
        dt = np.complex128 if cplx else np.float64
        x = np.asarray(x)
        squeeze = x.ndim == 2
        X = np.ascontiguousarray(x[:, None, :] if squeeze else x, dt)
        B = np.asarray(b)
        B = np.ascontiguousarray(B[:, None, :] if squeeze else B, dt)
        ab = np.ascontiguousarray(ab, dt)
        L = np.ascontiguousarray(self._L_host, dt)

        w = self.kl + self.ku + 1
        fn = getattr(_native.lib, "bandedqr_band_residual_batched"
                     + _native.suffix_for(cplx))
        p = _native.ptr_for(cplx)
        out = np.empty_like(X)
        for t in range(X.shape[1]):
            xt = np.ascontiguousarray(X[:, t, :])
            bt = np.ascontiguousarray(B[:, t, :])
            rt = np.empty_like(xt)
            fn(p(ab), self.n, w, self.batch, self.kl, self.ku, p(xt), p(bt), p(rt))
            # the band carries only the in-band part of the boundary rows; the
            # out-of-band remainder is L, applied here as the rank-r correction
            rt[self.dense_rows, :] -= L @ xt
            out[:, t, :] = rt
        return out[:, 0, :] if squeeze else out

    def solve(self, b, refine=2, tol=1e-13, return_info=False):
        """Solve and refine against float64 residuals; see
        :meth:`BatchedBandQR.solve`."""
        b = np.asarray(b)
        dt = np.complex128 if self.cplx else np.float64
        x = np.asarray(self.solve_fp32(b), dtype=dt)
        bn = np.maximum(np.linalg.norm(b.reshape(self.n, -1), axis=0),
                        np.finfo(float).tiny)
        hist = []
        for _ in range(int(refine)):
            r = self.residual(x, b)
            rel = float(np.max(np.linalg.norm(r.reshape(self.n, -1), axis=0) / bn))
            hist.append(rel)
            if rel <= tol:
                break
            x = x + self.solve_fp32(r)
        if return_info:
            r = self.residual(x, b)
            hist.append(float(np.max(np.linalg.norm(r.reshape(self.n, -1), axis=0) / bn)))
            return x, {"residual_history": hist}
        return x

    def __repr__(self):
        return (f"<BatchedRideQR n={self.n} batch={self.batch} kl={self.kl} "
                f"ku={self.ku} r={self.rfill} "
                f"{'complex64' if self.cplx else 'float32'}>")


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
