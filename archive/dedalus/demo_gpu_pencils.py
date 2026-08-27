"""The genuine bandqr win: GPU-batching the per-wavenumber pencils of a
multidimensional spectral solve.

A 2D spectral solve (Fourier x Chebyshev) decouples, per timestep, into one
independent banded system per Fourier wavenumber -- a "pencil".  Tau-free, each
pencil is a clean banded z-operator differing only by the -k^2 diagonal shift (a
3D solve has Nx*Ny of them).  Dedalus solves these in a serial CPU loop (one LU
per subproblem).  bandqr's QR is pivot-free, so all pencils batch onto the GPU at
once (no warp divergence) -- exactly the regime where a batched GPU solve beats a
CPU loop.

CPU baseline: LAPACK banded solve (factor+solve) per pencil, serial.
GPU: bandqr's batched banded QR (fp32), all pencils at once.

Needs native arm64 + mlx.  fp32 on the GPU (err ~1e-5); for fp64 accuracy add a
step or two of iterative refinement (the QR factor is well-conditioned).
"""
import numpy as np, time
import scipy.linalg as sla
import mlx.core as mx
from bandqr_gpu import solve_batched

n, kl, ku = 64, 2, 4                          # collapsed Chebyshev z-pencil (Nz=64), banded
rng = np.random.default_rng(0)
ab0 = (rng.standard_normal((n, kl + ku + 1)) * 0.15)
ab0[:, kl] += (kl + ku + 6)                   # diagonal dominance (index kl = main diagonal)

def to_scipy(ab):                             # bandqr band layout -> scipy solve_banded layout
    absc = np.zeros((kl + ku + 1, n))
    for off in range(-kl, ku + 1):
        c = off + kl
        if off >= 0: absc[ku - off, off:] = ab[:n - off, c]
        else:        absc[ku - off, :n + off] = ab[-off:, c]
    return absc

def best(fn, r=5):
    return min((lambda t: (fn(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(r)) * 1e3

print("# per-wavenumber pencils of a 2D spectral solve (banded, Nz=64): CPU loop vs bandqr GPU batch")
print(f"{'#pencils':>9} {'CPU loop [ms]':>14} {'GPU batch [ms]':>15} {'speedup':>8} {'fp32 err':>9}")
for batch in (256, 1024, 4096, 16384, 65536):
    ks2 = np.linspace(0, 200, batch)                        # -k^2 across the Fourier modes
    ab = np.repeat(ab0[:, :, None], batch, axis=2).astype(np.float32)
    ab[:, kl, :] -= ks2[None, :]                            # pencil = base operator - k^2 on diagonal
    b = rng.standard_normal((n, batch)).astype(np.float32)
    absc = [to_scipy(ab[:, :, i]) for i in range(batch)]    # assembly (excluded from timing)
    t_cpu = best(lambda: [sla.solve_banded((kl, ku), absc[i], b[:, i]) for i in range(batch)], r=3)
    xg = np.array(solve_batched(ab, b, kl, ku))             # warm up
    t_gpu = best(lambda: mx.eval(solve_batched(ab, b, kl, ku)), r=5)
    xc = np.stack([sla.solve_banded((kl, ku), absc[i], b[:, i]) for i in range(batch)], axis=1)
    err = np.max(np.abs(xg - xc)) / np.max(np.abs(xc))
    print(f"{batch:>9} {t_cpu:>14.2f} {t_gpu:>15.2f} {t_cpu/t_gpu:>7.1f}x {err:>9.1e}")
