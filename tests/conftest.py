"""Shared fixtures: matrix builders with known structure."""
import numpy as np
import pytest

from bandqr import dense_to_band


@pytest.fixture
def rng():
    return np.random.default_rng(20240611)


def make_banded(rng, n, kl, ku, complex_=False, dominant=True):
    """A random ``(kl, ku)``-banded matrix, diagonally dominant by default."""
    A = np.zeros((n, n), dtype=np.complex128 if complex_ else np.float64)
    for i in range(n):
        lo, hi = max(0, i - kl), min(n - 1, i + ku)
        vals = rng.standard_normal(hi - lo + 1)
        if complex_:
            vals = vals + 1j * rng.standard_normal(hi - lo + 1)
        A[i, lo:hi + 1] = vals
        if dominant:
            A[i, i] += kl + ku + 3
    return A


def make_band_storage(rng, n, kl, ku, complex_=False):
    ab = rng.standard_normal((n, kl + ku + 1))
    if complex_:
        ab = ab + 1j * rng.standard_normal((n, kl + ku + 1))
    ab[:, kl] += kl + ku + 3
    return ab


def make_almost_banded(rng, n, kl, ku, r=0, c=0, complex_=False):
    """Banded bulk plus ``r`` dense rows at the top and ``c`` dense columns."""
    A = make_banded(rng, n, kl, ku, complex_)
    for t in range(r):
        row = rng.standard_normal(n)
        A[t, :] = row + 1j * rng.standard_normal(n) if complex_ else row
    for t in range(c):
        col = rng.standard_normal(n)
        A[:, n - 1 - t] = col + 1j * rng.standard_normal(n) if complex_ else col
    return A


def chebyshev_bc_rows(n, points=(1.0, -1.0)):
    """Chebyshev evaluation rows ``T_k(x0)`` -- the classic recurrence-structured
    boundary condition, satisfying ``T_{k+1} - 2 x0 T_k + T_{k-1} = 0``."""
    k = np.arange(n)
    return np.array([np.cos(k * np.arccos(np.clip(x, -1, 1)))
                     if abs(x) <= 1 else np.cosh(k * np.arccosh(x))
                     for x in points])


def make_collapsible(rng, n, kl, ku, points=(1.0, -1.0)):
    """Banded bulk with genuinely recurrence-structured (Chebyshev) BC rows,
    which is what ``collapse`` requires -- random rows satisfy no recurrence."""
    A = make_banded(rng, n, kl, ku)
    rows = chebyshev_bc_rows(n, points)
    A[:len(points), :] = rows
    return A, rows


def relerr(A, x, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


@pytest.fixture
def banded(rng):
    return lambda *a, **k: make_banded(rng, *a, **k)


@pytest.fixture
def band_storage(rng):
    return lambda *a, **k: make_band_storage(rng, *a, **k)


@pytest.fixture
def almost_banded(rng):
    return lambda *a, **k: make_almost_banded(rng, *a, **k)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires MLX and a Metal device")
