"""Batched solves: the NumPy backend, and the GPU checked against it."""
import numpy as np
import pytest

from bandqr import BandQR, available_backends, solve_batched
from bandqr import gpu as gpu_mod
from bandqr.batched import BatchedBandQR as NumpyBatched
from bandqr._band import band_matvec_batched

needs_gpu = pytest.mark.skipif(not gpu_mod.is_available(),
                               reason="no MLX/Metal device")


def make_batch(rng, n, kl, ku, batch, complex_=False, nrhs=None):
    ab = rng.standard_normal((n, kl + ku + 1, batch))
    if complex_:
        ab = ab + 1j * rng.standard_normal((n, kl + ku + 1, batch))
    ab[:, kl, :] += kl + ku + 4
    shape = (n, batch) if nrhs is None else (n, nrhs, batch)
    b = rng.standard_normal(shape)
    if complex_:
        b = b + 1j * rng.standard_normal(shape)
    return ab, b


def batch_relerr(ab, kl, ku, x, b):
    if x.ndim == 3:
        return max(np.abs(b[:, t] - band_matvec_batched(ab, kl, ku, x[:, t])).max()
                   for t in range(x.shape[1]))
    return np.abs(b - band_matvec_batched(ab, kl, ku, x)).max()


# --------------------------------------------------------------- numpy backend
@pytest.mark.parametrize("n,kl,ku,batch", [(32, 1, 1, 8), (64, 2, 3, 32),
                                           (50, 3, 0, 16), (40, 0, 3, 16)])
@pytest.mark.parametrize("complex_", [False, True])
def test_numpy_backend_matches_per_system(rng, n, kl, ku, batch, complex_):
    ab, b = make_batch(rng, n, kl, ku, batch, complex_)
    X = NumpyBatched(ab, kl, ku).solve(b)
    for s in range(batch):
        ref = BandQR(np.ascontiguousarray(ab[:, :, s]), kl, ku).solve(
            np.ascontiguousarray(b[:, s]))
        assert np.allclose(X[:, s], ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("complex_", [False, True])
def test_numpy_backend_multi_rhs(rng, complex_):
    ab, B = make_batch(rng, 40, 2, 2, 16, complex_, nrhs=3)
    X = NumpyBatched(ab, 2, 2).solve(B)
    assert X.shape == B.shape
    assert batch_relerr(ab, 2, 2, X, B) < 1e-11


def test_numpy_backend_shifted(rng):
    n, kl, ku, batch = 48, 2, 2, 24
    base = rng.standard_normal((n, kl + ku + 1))
    base[:, kl] += 8.0
    shifts = np.linspace(0.5, 20.0, batch)
    full = np.repeat(base[:, :, None], batch, axis=2)
    full[:, kl, :] += shifts[None, :]
    b = rng.standard_normal((n, batch))
    x = NumpyBatched(base, kl, ku, shifts=shifts).solve(b)
    assert np.allclose(x, NumpyBatched(full, kl, ku).solve(b))


def test_available_backends_lists_numpy():
    assert "numpy" in available_backends()


def test_bad_shapes_rejected(rng):
    ab, b = make_batch(rng, 20, 1, 1, 4)
    with pytest.raises(ValueError, match="band width"):
        NumpyBatched(ab, 2, 2)
    with pytest.raises(ValueError, match="shape"):
        NumpyBatched(ab, 1, 1).solve(np.zeros((19, 4)))


# ------------------------------------------------------------------ the GPU
@needs_gpu
@pytest.mark.parametrize("n,kl,ku,batch", [(32, 1, 1, 64), (64, 2, 3, 128),
                                           (97, 4, 4, 33), (48, 0, 2, 256)])
@pytest.mark.parametrize("complex_", [False, True])
def test_gpu_matches_numpy_after_refinement(rng, n, kl, ku, batch, complex_):
    ab, b = make_batch(rng, n, kl, ku, batch, complex_)
    ref = NumpyBatched(ab, kl, ku).solve(b)
    x = gpu_mod.BatchedBandQR(ab, kl, ku).solve(b, refine=3)
    assert np.abs(x - ref).max() / np.abs(ref).max() < 1e-10


@needs_gpu
@pytest.mark.parametrize("complex_", [False, True])
def test_gpu_fp32_is_fp32_accurate(rng, complex_):
    """Unrefined, the GPU answer should be fp32-good -- and no better, which is
    the honest baseline the refinement claim is measured against."""
    ab, b = make_batch(rng, 64, 2, 2, 128, complex_)
    ref = NumpyBatched(ab, 2, 2).solve(b)
    x = gpu_mod.BatchedBandQR(ab, 2, 2).solve_fp32(b)
    rel = np.abs(x - ref).max() / np.abs(ref).max()
    assert 1e-9 < rel < 1e-4


@needs_gpu
def test_gpu_refinement_converges_monotonically(rng):
    ab, b = make_batch(rng, 64, 2, 2, 128)
    _, info = gpu_mod.BatchedBandQR(ab, 2, 2).solve(b, refine=3, return_info=True)
    hist = info["residual_history"]
    assert hist[0] > 1e-8                     # fp32 starting point
    assert min(hist) < 1e-12                  # float64 finish
    assert hist[1] < hist[0] / 100


@needs_gpu
@pytest.mark.parametrize("complex_", [False, True])
def test_gpu_fused_matches_factored(rng, complex_):
    ab, b = make_batch(rng, 64, 2, 2, 96, complex_)
    fused = gpu_mod.solve_batched(ab, b, 2, 2, refine=0, fused=True)
    factored = gpu_mod.BatchedBandQR(ab, 2, 2).solve_fp32(b)
    assert np.allclose(fused, factored, rtol=1e-5, atol=1e-6)


@needs_gpu
@pytest.mark.parametrize("complex_", [False, True])
def test_gpu_multi_rhs(rng, complex_):
    ab, B = make_batch(rng, 64, 3, 2, 96, complex_, nrhs=4)
    ref = NumpyBatched(ab, 3, 2).solve(B)
    x = gpu_mod.BatchedBandQR(ab, 3, 2).solve(B, refine=3)
    assert x.shape == B.shape
    assert np.abs(x - ref).max() / np.abs(ref).max() < 1e-10


@needs_gpu
def test_gpu_shifted_matches_materialised(rng):
    n, kl, ku, batch = 96, 2, 2, 256
    base = rng.standard_normal((n, kl + ku + 1))
    base[:, kl] += 8.0
    shifts = np.linspace(0.5, 40.0, batch)
    full = np.repeat(base[:, :, None], batch, axis=2)
    full[:, kl, :] += shifts[None, :]
    b = rng.standard_normal((n, batch))
    ref = NumpyBatched(full, kl, ku).solve(b)
    x = gpu_mod.solve_batched_shifted(base, shifts, b, kl, ku, refine=3)
    assert np.abs(x - ref).max() / np.abs(ref).max() < 1e-10


@needs_gpu
def test_gpu_kernel_cache_is_shape_independent(rng):
    """Changing n or batch must not recompile -- only (kl, ku, dtype) may."""
    gpu_mod.clear_kernel_cache()
    for n, batch in [(32, 64), (64, 64), (64, 128), (97, 33)]:
        ab, b = make_batch(rng, n, 2, 2, batch)
        gpu_mod.BatchedBandQR(ab, 2, 2).solve_fp32(b)
    # factor + solve for one (kl, ku, real) combination = 2 kernels
    assert len(gpu_mod._KERNELS) == 2


@needs_gpu
def test_gpu_residual_matches_numpy(rng):
    for complex_ in (False, True):
        ab, b = make_batch(rng, 40, 2, 3, 17, complex_)
        F = gpu_mod.BatchedBandQR(ab, 2, 3)
        x = np.asarray(F.solve_fp32(b), dtype=np.complex128 if complex_ else np.float64)
        want = b - band_matvec_batched(ab, 2, 3, x)
        assert np.allclose(F.residual(x, b), want, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------- batched almost-banded
@needs_gpu
@pytest.mark.parametrize("r", [1, 2, 3])
def test_gpu_ride_matches_cpu(rng, r):
    from bandqr import RideQR
    from bandqr._band import band_to_dense
    n, kl, ku, batch = 64, 2, 2, 96
    bulk = rng.standard_normal((n, kl + ku + 1, batch))
    bulk[:, kl, :] += kl + ku + 5
    rows = rng.standard_normal((r, n))
    dr = np.arange(r)
    b = rng.standard_normal((n, batch))

    F = gpu_mod.BatchedRideQR.from_dense_rows(bulk, kl, ku, dr, rows)
    x = F.solve(b, refine=4)

    for s in range(0, batch, 17):
        M = band_to_dense(np.ascontiguousarray(bulk[:, :, s]), kl, ku, n)
        M[dr, :] = rows
        ref = RideQR(M, kl, ku, dr).solve(np.ascontiguousarray(b[:, s]))
        assert np.allclose(x[:, s], ref, rtol=1e-7, atol=1e-9)


@needs_gpu
def test_gpu_ride_rejects_unreachable_rows(rng):
    n, kl, ku, batch = 64, 1, 1, 32
    bulk = rng.standard_normal((n, kl + ku + 1, batch))
    bulk[:, kl, :] += 6
    with pytest.raises(ValueError, match="within kl"):
        gpu_mod.BatchedRideQR(bulk, kl, ku, [30], np.zeros((1, n)))


# ------------------------------------------------------------- front end
@pytest.mark.parametrize("backend", ["numpy"] + (["gpu"] if gpu_mod.is_available()
                                                 else []))
def test_solve_batched_front_end(rng, backend):
    ab, b = make_batch(rng, 64, 2, 2, 128)
    x = solve_batched(ab, b, 2, 2, backend=backend, refine=3)
    assert batch_relerr(ab, 2, 2, x, b) < 1e-10


def test_solve_batched_rejects_unknown_backend(rng):
    ab, b = make_batch(rng, 16, 1, 1, 8)
    with pytest.raises(ValueError, match="unknown backend"):
        solve_batched(ab, b, 1, 1, backend="cuda")
