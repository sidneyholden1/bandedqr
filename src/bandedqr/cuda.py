"""Batched banded QR on NVIDIA GPUs through CuPy/CUDA.

The CUDA backend follows the same execution model as the Metal backend: one
CUDA thread owns one independent banded system, so pivot-free QR gives every
thread the same control flow.  Factors and rotations remain device resident
for factor-once/solve-many workloads.  The device arithmetic is fp32 (or
complex64); optional iterative refinement forms residuals against the original
float64 operator on the host.

CuPy is imported lazily.  Importing :mod:`bandedqr` therefore does not require
CUDA, and CPU-only installations continue to work unchanged.
"""

from functools import lru_cache
import importlib.util
import subprocess
import sys

import numpy as np

from . import _native

__all__ = [
    "is_available",
    "BatchedBandQR",
    "solve_batched",
    "solve_batched_shifted",
    "CUDAUnavailable",
    "clear_kernel_cache",
]


class CUDAUnavailable(RuntimeError):
    """CuPy/CUDA is not usable in this process."""


def _cp():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise CUDAUnavailable(
            "the CUDA backend needs CuPy. Use backend='numpy', or load/install "
            "a CuPy build matching the local CUDA runtime."
        ) from exc
    return cp


@lru_cache(maxsize=1)
def is_available() -> bool:
    """Return whether this interpreter can execute a trivial CUDA operation."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import cupy as cp; "
                    "assert cp.cuda.runtime.getDeviceCount() > 0; "
                    "x=cp.zeros(1, dtype=cp.float32); "
                    "assert float(x.sum().get()) == 0.0"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return False


_REAL_HEADER = r"""
typedef float SCALAR;
__device__ __forceinline__ SCALAR czero() { return 0.0f; }
__device__ __forceinline__ SCALAR cadd(SCALAR a, SCALAR b) { return a + b; }
__device__ __forceinline__ SCALAR csub(SCALAR a, SCALAR b) { return a - b; }
__device__ __forceinline__ SCALAR cmul(SCALAR a, SCALAR b) { return a * b; }
__device__ __forceinline__ SCALAR cconj(SCALAR a) { return a; }
__device__ __forceinline__ SCALAR cscale(float a, SCALAR b) { return a * b; }
__device__ __forceinline__ SCALAR cdivv(SCALAR a, SCALAR b) { return a / b; }
__device__ __forceinline__ float cabs2(SCALAR a) { return a * a; }
__device__ __forceinline__ void givens(
    SCALAR a, SCALAR b, float *c, SCALAR *s
) {
    float bb2 = b * b;
    if (bb2 == 0.0f) { *c = 1.0f; *s = 0.0f; return; }
    float aa2 = a * a;
    float t = sqrtf(aa2 + bb2);
    if (aa2 == 0.0f) { *c = 0.0f; *s = b / sqrtf(bb2); return; }
    float aa = sqrtf(aa2);
    *c = aa / t;
    *s = (a / aa) * b / t;
}
"""


_COMPLEX_HEADER = r"""
typedef float2 SCALAR;
__device__ __forceinline__ SCALAR czero() { return make_float2(0.0f, 0.0f); }
__device__ __forceinline__ SCALAR cadd(SCALAR a, SCALAR b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ SCALAR csub(SCALAR a, SCALAR b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ SCALAR cmul(SCALAR a, SCALAR b) {
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ SCALAR cconj(SCALAR a) {
    return make_float2(a.x, -a.y);
}
__device__ __forceinline__ SCALAR cscale(float a, SCALAR b) {
    return make_float2(a*b.x, a*b.y);
}
__device__ __forceinline__ float cabs2(SCALAR a) {
    return a.x*a.x + a.y*a.y;
}
__device__ __forceinline__ SCALAR cdivv(SCALAR a, SCALAR b) {
    float d = cabs2(b);
    return make_float2(
        (a.x*b.x + a.y*b.y)/d,
        (a.y*b.x - a.x*b.y)/d
    );
}
__device__ __forceinline__ void givens(
    SCALAR a, SCALAR b, float *c, SCALAR *s
) {
    float bb2 = cabs2(b);
    if (bb2 == 0.0f) {
        *c = 1.0f; *s = make_float2(0.0f, 0.0f); return;
    }
    float aa2 = cabs2(a);
    float t = sqrtf(aa2 + bb2);
    if (aa2 == 0.0f) {
        *c = 0.0f; *s = cscale(1.0f/sqrtf(bb2), cconj(b)); return;
    }
    float aa = sqrtf(aa2);
    *c = aa / t;
    *s = cscale(1.0f/(aa*t), cmul(a, cconj(b)));
}
"""


_KERNEL_BODY = r"""
#define KL @KL@
#define KU @KU@

extern "C" __global__ void bandedqr_factor(
    const SCALAR *ab,
    const SCALAR *shifts,
    SCALAR *wab,
    float *cvals,
    SCALAR *svals,
    int N,
    int BATCH,
    int SHIFTED
) {
    int sidx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (sidx >= BATCH) return;
    const int KUR = KL + KU;
    const int WIN = KL + KU + 1;
    const int W = 2*KL + KU + 2;
    const unsigned long long bs = (unsigned long long)BATCH;
    const unsigned long long sd = (unsigned long long)sidx;

    for (int i = 0; i < N; i++) {
        unsigned long long wrow = (unsigned long long)i * (unsigned long long)W;
        unsigned long long arow = (unsigned long long)i * (unsigned long long)WIN;
        for (int o = 0; o < W; o++) {
            SCALAR value = czero();
            if (o < WIN) {
                if (SHIFTED) {
                    value = ab[arow + (unsigned long long)o];
                    if (o == KL) value = cadd(value, shifts[sd]);
                } else {
                    value = ab[(arow + (unsigned long long)o) * bs + sd];
                }
            }
            wab[(wrow + (unsigned long long)o) * bs + sd] = value;
        }
    }

    unsigned long long k = 0;
    for (int jc = 0; jc < N; jc++) {
        int candidate = jc + KL;
        int ilast = candidate < N ? candidate : N - 1;
        for (int q = ilast; q >= jc + 1; q--) {
            int p = q - 1;
            unsigned long long ip = (
                (unsigned long long)p * (unsigned long long)W
                + (unsigned long long)(jc - p + KL)
            ) * bs + sd;
            unsigned long long iq = (
                (unsigned long long)q * (unsigned long long)W
                + (unsigned long long)(jc - q + KL)
            ) * bs + sd;
            SCALAR a = wab[ip], b = wab[iq], sn;
            float c;
            givens(a, b, &c, &sn);
            int last = q + KUR;
            int jmax = last < N ? last : N - 1;
            for (int j = jc; j <= jmax; j++) {
                unsigned long long jp = (
                    (unsigned long long)p * (unsigned long long)W
                    + (unsigned long long)(j - p + KL)
                ) * bs + sd;
                unsigned long long jq = (
                    (unsigned long long)q * (unsigned long long)W
                    + (unsigned long long)(j - q + KL)
                ) * bs + sd;
                SCALAR ap = wab[jp], aq = wab[jq];
                wab[jp] = cadd(cscale(c, ap), cmul(sn, aq));
                wab[jq] = csub(cscale(c, aq), cmul(cconj(sn), ap));
            }
            cvals[k * bs + sd] = c;
            svals[k * bs + sd] = sn;
            k++;
        }
    }
}

extern "C" __global__ void bandedqr_solve(
    const SCALAR *wab,
    const float *cvals,
    const SCALAR *svals,
    const SCALAR *b,
    SCALAR *x,
    SCALAR *y,
    int N,
    int BATCH,
    int NRHS
) {
    int sidx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (sidx >= BATCH) return;
    const int KUR = KL + KU;
    const int W = 2*KL + KU + 2;
    const unsigned long long bs = (unsigned long long)BATCH;
    const unsigned long long sd = (unsigned long long)sidx;

    for (int i = 0; i < N; i++) {
        for (int t = 0; t < NRHS; t++) {
            unsigned long long index = (
                (unsigned long long)i * (unsigned long long)NRHS
                + (unsigned long long)t
            ) * bs + sd;
            y[index] = b[index];
        }
    }

    unsigned long long k = 0;
    for (int jc = 0; jc < N; jc++) {
        int candidate = jc + KL;
        int ilast = candidate < N ? candidate : N - 1;
        for (int q = ilast; q >= jc + 1; q--) {
            int p = q - 1;
            float c = cvals[k * bs + sd];
            SCALAR sn = svals[k * bs + sd];
            for (int t = 0; t < NRHS; t++) {
                unsigned long long yp = (
                    (unsigned long long)p * (unsigned long long)NRHS
                    + (unsigned long long)t
                ) * bs + sd;
                unsigned long long yq = (
                    (unsigned long long)q * (unsigned long long)NRHS
                    + (unsigned long long)t
                ) * bs + sd;
                SCALAR ap = y[yp], aq = y[yq];
                y[yp] = cadd(cscale(c, ap), cmul(sn, aq));
                y[yq] = csub(cscale(c, aq), cmul(cconj(sn), ap));
            }
            k++;
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        int last = i + KUR;
        int jhi = last < N ? last : N - 1;
        unsigned long long wrow = (unsigned long long)i * (unsigned long long)W;
        for (int t = 0; t < NRHS; t++) {
            unsigned long long xi = (
                (unsigned long long)i * (unsigned long long)NRHS
                + (unsigned long long)t
            ) * bs + sd;
            SCALAR acc = y[xi];
            for (int j = i + 1; j <= jhi; j++) {
                SCALAR rij = wab[(
                    wrow + (unsigned long long)(j - i + KL)
                ) * bs + sd];
                unsigned long long xj = (
                    (unsigned long long)j * (unsigned long long)NRHS
                    + (unsigned long long)t
                ) * bs + sd;
                acc = csub(acc, cmul(rij, x[xj]));
            }
            SCALAR diag = wab[(wrow + (unsigned long long)KL) * bs + sd];
            x[xi] = cdivv(acc, diag);
        }
    }
}
"""


def _source(kl: int, ku: int, cplx: bool) -> str:
    header = _COMPLEX_HEADER if cplx else _REAL_HEADER
    return header + _KERNEL_BODY.replace("@KL@", str(kl)).replace("@KU@", str(ku))


@lru_cache(maxsize=None)
def _kernels(kl: int, ku: int, cplx: bool):
    cp = _cp()
    module = cp.RawModule(
        code=_source(kl, ku, cplx),
        options=("--std=c++11",),
    )
    return (
        module.get_function("bandedqr_factor"),
        module.get_function("bandedqr_solve"),
    )


def clear_kernel_cache() -> None:
    """Drop process-local references to compiled CUDA modules."""
    _kernels.cache_clear()


def _nrot(n: int, kl: int) -> int:
    if n <= 0:
        return 0
    return sum(min(kl, n - 1 - jc) for jc in range(n))


class BatchedBandQR:
    """Batched fp32/complex64 band QR held on one CUDA device."""

    def __init__(self, ab, kl, ku, shifts=None, block_size=256):
        cp = _cp()
        ab = np.asarray(ab)
        self.kl, self.ku = int(kl), int(ku)
        self.shifted = shifts is not None
        self.block_size = int(block_size)
        self.cplx = bool(np.issubdtype(ab.dtype, np.complexfloating))
        if self.shifted:
            if ab.ndim != 2:
                raise ValueError("shifted form wants ab of shape (n, kl+ku+1)")
            n, width = ab.shape
            shifts = np.asarray(shifts).ravel()
            batch = shifts.size
            self.cplx = bool(
                self.cplx or np.issubdtype(shifts.dtype, np.complexfloating)
            )
        else:
            if ab.ndim != 3:
                raise ValueError("ab must have shape (n, kl+ku+1,batch)")
            n, width, batch = ab.shape
            shifts = np.zeros(batch, dtype=np.float32)
        if width != self.kl + self.ku + 1:
            raise ValueError(
                f"band width must be kl+ku+1 = {self.kl+self.ku+1}, got {width}"
            )
        if self.block_size <= 0 or self.block_size > 1024:
            raise ValueError("block_size must be in [1, 1024]")

        self.n, self.batch = int(n), int(batch)
        self.W = 2 * self.kl + self.ku + 2
        self.nrot = _nrot(self.n, self.kl)
        self._ab_host = ab
        self._shifts_host = shifts
        dtype = cp.complex64 if self.cplx else cp.float32
        ab_device = cp.asarray(ab, dtype=dtype, order="C")
        shifts_device = cp.asarray(shifts, dtype=dtype, order="C")
        self._wab = cp.empty((self.n, self.W, self.batch), dtype=dtype)
        self._cvals = cp.empty((self.nrot, self.batch), dtype=cp.float32)
        self._svals = cp.empty((self.nrot, self.batch), dtype=dtype)
        factor, _ = _kernels(self.kl, self.ku, self.cplx)
        block = min(self.block_size, max(1, self.batch))
        grid = ((self.batch + block - 1) // block,)
        factor(
            grid,
            (block,),
            (
                ab_device,
                shifts_device,
                self._wab,
                self._cvals,
                self._svals,
                np.int32(self.n),
                np.int32(self.batch),
                np.int32(self.shifted),
            ),
        )
        cp.cuda.get_current_stream().synchronize()

    def solve_fp32_device(self, b):
        """Solve with resident factors and return a CuPy array."""
        cp = _cp()
        dtype = cp.complex64 if self.cplx else cp.float32
        b_device = cp.asarray(b, dtype=dtype, order="C")
        squeeze = b_device.ndim == 2
        B = b_device[:, None, :] if squeeze else b_device
        if B.ndim != 3:
            raise ValueError("b must have shape (n,batch) or (n,nrhs,batch)")
        n, nrhs, batch = B.shape
        if n != self.n or batch != self.batch:
            raise ValueError(
                f"b has shape (n={n}, batch={batch}) but factorization has "
                f"(n={self.n}, batch={self.batch})"
            )
        x = cp.empty_like(B)
        y = cp.empty_like(B)
        _, solve = _kernels(self.kl, self.ku, self.cplx)
        block = min(self.block_size, max(1, self.batch))
        grid = ((self.batch + block - 1) // block,)
        solve(
            grid,
            (block,),
            (
                self._wab,
                self._cvals,
                self._svals,
                B,
                x,
                y,
                np.int32(self.n),
                np.int32(self.batch),
                np.int32(nrhs),
            ),
        )
        return x[:, 0, :] if squeeze else x

    def solve_fp32(self, b):
        """Solve with resident factors and return a NumPy array."""
        cp = _cp()
        return cp.asnumpy(self.solve_fp32_device(b))

    def _dense_band(self):
        if not self.shifted:
            return self._ab_host
        ab = np.repeat(self._ab_host[:, :, None], self.batch, axis=2).astype(
            np.result_type(self._ab_host.dtype, self._shifts_host.dtype)
        )
        ab[:, self.kl, :] += self._shifts_host[None, :]
        return ab

    def residual(self, x, b):
        """Form ``b - A*x`` in host float64 against the original operator."""
        ab = np.ascontiguousarray(self._dense_band())
        cplx = np.iscomplexobj(ab) or np.iscomplexobj(x) or np.iscomplexobj(b)
        dtype = np.complex128 if cplx else np.float64
        X0 = np.asarray(x)
        squeeze = X0.ndim == 2
        X = np.ascontiguousarray(X0[:, None, :] if squeeze else X0, dtype)
        B0 = np.asarray(b)
        B = np.ascontiguousarray(B0[:, None, :] if squeeze else B0, dtype)
        ab = np.ascontiguousarray(ab, dtype)
        width = self.kl + self.ku + 1
        function = getattr(
            _native.lib,
            "bandedqr_band_residual_batched" + _native.suffix_for(cplx),
        )
        pointer = _native.ptr_for(cplx)
        result = np.empty_like(X)
        for rhs in range(X.shape[1]):
            xt = np.ascontiguousarray(X[:, rhs, :])
            bt = np.ascontiguousarray(B[:, rhs, :])
            rt = np.empty_like(xt)
            function(
                pointer(ab),
                self.n,
                width,
                self.batch,
                self.kl,
                self.ku,
                pointer(xt),
                pointer(bt),
                pointer(rt),
            )
            result[:, rhs, :] = rt
        return result[:, 0, :] if squeeze else result

    def solve(self, b, refine=2, tol=1e-13, return_info=False):
        """Solve and optionally refine against host float64 residuals."""
        b = np.asarray(b)
        dtype = np.complex128 if self.cplx else np.float64
        x = np.asarray(self.solve_fp32(b), dtype=dtype)
        norms = np.maximum(
            np.linalg.norm(b.reshape(self.n, -1), axis=0),
            np.finfo(float).tiny,
        )
        history = []
        for _ in range(int(refine)):
            residual = self.residual(x, b)
            relative = float(
                np.max(
                    np.linalg.norm(residual.reshape(self.n, -1), axis=0)
                    / norms
                )
            )
            history.append(relative)
            if relative <= tol:
                break
            x = x + self.solve_fp32(residual)
        if return_info:
            residual = self.residual(x, b)
            history.append(
                float(
                    np.max(
                        np.linalg.norm(residual.reshape(self.n, -1), axis=0)
                        / norms
                    )
                )
            )
            return x, {"residual_history": history, "refine": int(refine)}
        return x

    def __repr__(self):
        shifted = ", shifted" if self.shifted else ""
        dtype = "complex64" if self.cplx else "float32"
        return (
            f"<BatchedBandQR[cuda] n={self.n} batch={self.batch} "
            f"kl={self.kl} ku={self.ku} {dtype}{shifted}>"
        )


def solve_batched(ab, b, kl, ku, refine=2, return_info=False, **kwargs):
    factor = BatchedBandQR(ab, kl, ku, **kwargs)
    return factor.solve(b, refine=refine, return_info=return_info)


def solve_batched_shifted(
    ab, shifts, b, kl, ku, refine=2, return_info=False, **kwargs
):
    factor = BatchedBandQR(ab, kl, ku, shifts=shifts, **kwargs)
    return factor.solve(b, refine=refine, return_info=return_info)
