import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from bandqr import (BandQR, dense_to_band, solve_bordered,
                    detect_recurrence, solve_precond_rows, solve_ride)

rng = np.random.default_rng(0)


def random_banded(n, kl, ku, diag_boost=0.0):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            A[i, j] = rng.standard_normal()
        A[i, i] += diag_boost  # keep it well-conditioned for the tests
    return A


def check(n, kl, ku):
    A = random_banded(n, kl, ku, diag_boost=kl + ku + 2)
    b = rng.standard_normal(n)
    qr = BandQR(dense_to_band(A, kl, ku), kl, ku)
    x = qr.solve(b)
    x_ref = np.linalg.solve(A, b)
    err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)

    # confirm R has the predicted bandwidth ku+kl and reproduces A's R
    Q_ref, R_ref = np.linalg.qr(A)
    Rb = qr.R_band()
    # compare |diag(R)| (signs are gauge-dependent between QR conventions)
    ddiff = np.max(np.abs(np.abs(Rb[:, 0]) - np.abs(np.diag(R_ref))))
    print(f"n={n:5d} kl={kl} ku={ku}  solve rel-err={err:.2e}  "
          f"|R| diag match={ddiff:.2e}  nrot~{n*kl}")
    assert err < 1e-9, "solve inaccurate"


def to_scipy_ab(A, kl, ku):
    n = A.shape[0]
    ab = np.zeros((kl + ku + 1, n))
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            ab[ku + i - j, j] = A[i, j]
    return ab


def bench(n, kl, ku, reps=5):
    A = random_banded(n, kl, ku, diag_boost=kl + ku + 2)
    b = rng.standard_normal(n)
    ab_mine = dense_to_band(A, kl, ku)
    ab_sp = to_scipy_ab(A, kl, ku)

    def timed(f, reps):
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
        return min(t)

    t_mine = timed(lambda: BandQR(ab_mine, kl, ku).solve(b), reps)
    t_lu = timed(lambda: sla.solve_banded((kl, ku), ab_sp, b), reps)
    t_dense = timed(lambda: np.linalg.solve(A, b), reps) if n <= 4000 else float("nan")

    print(f"n={n:6d} kl={kl} ku={ku} | bandqr {t_mine*1e3:8.3f} ms | "
          f"scipy band-LU {t_lu*1e3:8.3f} ms | dense {t_dense*1e3:9.3f} ms")


def check_threaded(n, kl, ku):
    A = random_banded(n, kl, ku, diag_boost=kl + ku + 2)
    ab = dense_to_band(A, kl, ku)
    Rs = BandQR(ab, kl, ku, nthreads=1).R_band()
    for nt in (2, 4, 8):
        Rt = BandQR(ab, kl, ku, nthreads=nt).R_band()
        d = np.max(np.abs(Rs - Rt))
        print(f"n={n:6d} kl={kl} ku={ku} threads={nt}  max|R_serial - R_threaded|={d:.2e}")
        assert d == 0.0, "threaded R must be bit-identical to serial"


def check_many(n, kl, ku, nrhs):
    A = random_banded(n, kl, ku, diag_boost=kl + ku + 2)
    B = rng.standard_normal((n, nrhs))
    qr = BandQR(dense_to_band(A, kl, ku), kl, ku)
    X = qr.solve(B)
    X_ref = np.linalg.solve(A, B)
    err = np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref)
    print(f"n={n:6d} kl={kl} ku={ku} nrhs={nrhs}  multi-RHS rel-err={err:.2e}")
    assert err < 1e-9


def make_almost_banded(n, kl, ku, r, c):
    """Banded bulk + r dense rows (BCs, top) + c dense cols (tau, right)."""
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            M[i, j] = rng.standard_normal()
        M[i, i] += kl + ku + 3
    dr = np.arange(r)
    dc = np.arange(n - c, n)
    M[dr, :] = rng.standard_normal((r, n))
    M[:, dc] = rng.standard_normal((n, c))
    return M, dr, dc


def check_bordered(n, kl, ku, r, c, sparse=False):
    M, dr, dc = make_almost_banded(n, kl, ku, r, c)
    b = rng.standard_normal(n)
    Min = sp.csr_matrix(M) if sparse else M
    x = solve_bordered(Min, kl, ku, b, dr, dc)
    xref = spla.spsolve(sp.csc_matrix(M), b) if sparse else np.linalg.solve(M, b)
    err = np.linalg.norm(x - xref) / np.linalg.norm(xref)
    tag = "sparse" if sparse else "dense "
    print(f"n={n:6d} kl={kl} ku={ku} r={r} c={c} ({tag}, r>kl={str(r>kl):5s})  "
          f"rel-err={err:.2e}")
    assert err < 1e-8


def check_complex(n, kl, ku, nrhs=1):
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            A[i, j] = rng.standard_normal() + 1j * rng.standard_normal()
        A[i, i] += kl + ku + 3
    b = (rng.standard_normal((n, nrhs)) + 1j * rng.standard_normal((n, nrhs))).squeeze()
    x = BandQR(dense_to_band(A, kl, ku), kl, ku).solve(b)
    xref = np.linalg.solve(A, b)
    err = np.linalg.norm(x - xref) / np.linalg.norm(xref)
    print(f"n={n:5d} kl={kl} ku={ku} nrhs={nrhs}  complex128 rel-err={err:.2e}  dtype={x.dtype}")
    assert err < 1e-9 and x.dtype == np.complex128


def check_householder(n, kl, ku, dt=np.float64, nrhs=1):
    A = np.zeros((n, n), dtype=dt)
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            A[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if dt == np.complex128 else 0)
        A[i, i] += kl + ku + 3
    b = (rng.standard_normal((n, nrhs)) + (1j * rng.standard_normal((n, nrhs)) if dt == np.complex128 else 0)).squeeze()
    x = BandQR(dense_to_band(A, kl, ku), kl, ku, householder=True).solve(b)
    err = np.linalg.norm(x - np.linalg.solve(A, b)) / np.linalg.norm(np.linalg.solve(A, b))
    print(f"n={n:5d} kl={kl} ku={ku} {np.dtype(dt).name:10s} nrhs={nrhs}  householder rel-err={err:.2e}")
    assert err < 1e-9


def check_ride(n, kl, ku, r, dt=np.float64):
    M = np.zeros((n, n), dtype=dt)
    for i in range(n):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            M[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if dt == np.complex128 else 0)
        M[i, i] += kl + ku + 3
    M[:r, :] = rng.standard_normal((r, n)) + (1j * rng.standard_normal((r, n)) if dt == np.complex128 else 0)
    b = rng.standard_normal(n) + (1j * rng.standard_normal(n) if dt == np.complex128 else 0)
    x = solve_ride(M, kl, ku, b, list(range(r)))
    err = np.linalg.norm(x - np.linalg.solve(M, b)) / np.linalg.norm(np.linalg.solve(M, b))
    print(f"n={n:5d} kl={kl} ku={ku} r={r} {np.dtype(dt).name:10s} (r>kl={str(r > kl):5s})  ride rel-err={err:.2e}")
    assert err < 1e-8


def check_recurrence():
    n = 60; k = np.arange(n)
    rows = {"T_k(0.3)": np.cos(k*np.arccos(0.3)), "T_k(1)": np.ones(n),
            "T_k(-1)": (-1.0)**k, "T'_k(1)=k^2": k**2.0}
    for name, v in rows.items():
        a = detect_recurrence(v)
        resid = max(abs(sum(a[i]*v[j+i] for i in range(len(a))))
                    for j in range(5, n-len(a)-5))
        print(f"  {name:14s} order {len(a)-1}: {np.round(a,3)}  resid={resid:.1e}")
        assert resid < 1e-9


def check_precond_rows(n, kl, ku, bcs):
    k = np.arange(n)
    rowvals = {'T(1)': np.ones(n), 'T(-1)': (-1.0)**k, 'T(0.3)': np.cos(k*np.arccos(0.3))}
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-kl), min(n-1, i+ku)+1):
            M[i, j] = rng.standard_normal()
        M[i, i] += kl + ku + 3
    dr = np.arange(len(bcs))
    for t, name in enumerate(bcs):
        M[t, :] = rowvals[name]
    b = rng.standard_normal(n)
    x = solve_precond_rows(M, kl, ku, b, dr)
    err = np.linalg.norm(x - np.linalg.solve(M, b)) / np.linalg.norm(np.linalg.solve(M, b))
    print(f"n={n} kl={kl} ku={ku} BCs={bcs}  rel-err={err:.2e}")
    assert err < 1e-7


def bench_threads(n, kl, ku, reps=5):
    A = random_banded(n, kl, ku, diag_boost=kl + ku + 2)
    ab = dense_to_band(A, kl, ku)
    b = rng.standard_normal(n)

    def timed(f):
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
        return min(t)

    # Explicitly force 4 threads to expose the wavefront's true cost; auto is
    # serial because threading is a net loss on this bandwidth-bound kernel.
    t1 = timed(lambda: BandQR(ab, kl, ku, nthreads=1).solve(b))
    tn = timed(lambda: BandQR(ab, kl, ku, nthreads=4).solve(b))
    print(f"n={n:6d} kl={kl:3d} ku={ku:3d} | serial {t1*1e3:8.3f} ms | "
          f"4 threads {tn*1e3:8.3f} ms | speedup {t1/tn:4.2f}x")


if __name__ == "__main__":
    print("== correctness (single RHS) ==")
    for n, kl, ku in [(50, 1, 1), (200, 2, 1), (500, 3, 4), (1000, 5, 2), (2000, 8, 8)]:
        check(n, kl, ku)

    print("\n== threaded factorisation == R bit-identical to serial ==")
    for n, kl, ku in [(2000, 8, 8), (5000, 16, 16), (20000, 32, 32)]:
        check_threaded(n, kl, ku)

    print("\n== multi-RHS solve ==")
    for n, kl, ku, nrhs in [(500, 3, 4, 1), (1000, 5, 5, 8), (2000, 8, 8, 64)]:
        check_many(n, kl, ku, nrhs)

    print("\n== complex128 banded ==")
    for n, kl, ku, nrhs in [(300, 2, 2, 1), (800, 3, 4, 1), (500, 4, 4, 8)]:
        check_complex(n, kl, ku, nrhs)

    print("\n== householder band QR (real & complex, multi-RHS) ==")
    for n, kl, ku, dt, nrhs in [(400, 2, 2, np.float64, 1), (300, 8, 8, np.float64, 4),
                                (400, 3, 3, np.complex128, 1), (300, 6, 6, np.complex128, 8)]:
        check_householder(n, kl, ku, dt, nrhs)

    print("\n== ride: almost-banded QR, BCs carried along (real & complex, r>kl) ==")
    for n, kl, ku, r, dt in [(300, 2, 2, 2, np.float64), (400, 1, 1, 3, np.float64),
                             (300, 2, 2, 2, np.complex128), (400, 1, 1, 3, np.complex128)]:
        check_ride(n, kl, ku, r, dt)

    print("\n== bordered solve (dense BC rows + tau cols, incl. r>kl) ==")
    for n, kl, ku, r, c in [(300, 2, 2, 2, 2), (500, 1, 1, 3, 3), (800, 2, 3, 4, 4)]:
        check_bordered(n, kl, ku, r, c, sparse=False)
    for n, kl, ku, r, c in [(20000, 3, 3, 4, 4), (50000, 2, 2, 3, 3)]:
        check_bordered(n, kl, ku, r, c, sparse=True)

    print("\n== recurrence detection (boundary rows) ==")
    check_recurrence()

    print("\n== recurrence-preconditioned single-band-QR solve (BC rows) ==")
    for n, kl, ku, bcs in [(200, 2, 2, ['T(1)', 'T(-1)']),
                           (300, 3, 2, ['T(1)', 'T(-1)', 'T(0.3)']),
                           (200, 1, 1, ['T(1)', 'T(-1)', 'T(0.3)'])]:  # r > kl+1
        check_precond_rows(n, kl, ku, bcs)

    print("\n== benchmark (factor + solve, best of 5) ==")
    for n, kl, ku in [(1000, 2, 2), (5000, 4, 4), (20000, 8, 8), (100000, 8, 8), (50000, 32, 32)]:
        bench(n, kl, ku)

    print("\n== wavefront threading speedup (factor + solve) ==")
    for n, kl, ku in [(20000, 16, 16), (50000, 32, 32), (20000, 64, 64), (100000, 32, 32)]:
        bench_threads(n, kl, ku)
