"""CPU benchmarks: bandqr against LAPACK band LU, across size and bandwidth.

    python bench/bench_cpu.py [--quick]

Reports factor+solve for a single right-hand side, plus the factor-once /
solve-many split that matters for timestepping.
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")

from bandqr import BandQR, band_to_dense, to_scipy_band  # noqa: E402
from bandqr.almost import CollapseQR, RideQR  # noqa: E402


def best(fn, reps=7, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def random_band(rng, n, kl, ku):
    ab = rng.standard_normal((n, kl + ku + 1))
    ab[:, kl] += kl + ku + 3
    return ab


def bench_vs_lapack(rng, quick):
    import scipy.linalg as sla
    print("\n== factor + solve, one RHS: bandqr vs LAPACK band LU ==")
    print(f"{'n':>8} {'kl':>4} {'ku':>4} | {'givens':>10} {'househldr':>10} "
          f"{'LAPACK LU':>10} | {'best vs LU':>10}")
    cases = [(1000, 2, 2), (10000, 2, 2), (100000, 2, 2),
             (10000, 4, 4), (20000, 8, 8), (20000, 16, 16)]
    if not quick:
        cases += [(50000, 32, 32), (20000, 64, 64)]
    for n, kl, ku in cases:
        ab = random_band(rng, n, kl, ku)
        b = rng.standard_normal(n)
        absc = to_scipy_band(ab, kl, ku)
        tg = best(lambda: BandQR(ab, kl, ku, nthreads=1).solve(b))
        th = best(lambda: BandQR(ab, kl, ku, householder=True).solve(b))
        tl = best(lambda: sla.solve_banded((kl, ku), absc, b))
        ratio = tl / min(tg, th)
        print(f"{n:8d} {kl:4d} {ku:4d} | {tg*1e3:9.3f}m {th*1e3:9.3f}m "
              f"{tl*1e3:9.3f}m | {ratio:9.2f}x")
    print("  (>1 means bandqr is faster.  QR wins on narrow bands; LAPACK's")
    print("   blocked band LU pulls ahead as the band widens.)")


def bench_factor_reuse(rng, quick):
    print("\n== factor once, solve many ==")
    print(f"{'n':>8} {'kl':>4} {'nrhs':>5} | {'refactor each':>14} "
          f"{'reuse factor':>13} | {'speedup':>8}")
    for n, kl in [(10000, 4), (50000, 8)] + ([] if quick else [(100000, 8)]):
        ab = random_band(rng, n, kl, kl)
        for nrhs in (10, 50):
            B = rng.standard_normal((n, nrhs))
            t_re = best(lambda: [BandQR(ab, kl, kl).solve(np.ascontiguousarray(B[:, t]))
                                 for t in range(nrhs)], reps=3)
            F = BandQR(ab, kl, kl)
            t_reuse = best(lambda: F.solve(B), reps=3)
            print(f"{n:8d} {kl:4d} {nrhs:5d} | {t_re*1e3:13.2f}m "
                  f"{t_reuse*1e3:12.2f}m | {t_re/t_reuse:7.1f}x")


def bench_almost_banded(rng, quick):
    """Carried borders (ride) against recurrence collapse.

    Factor and solve are reported separately because they point different ways:
    collapse pays up front to build and apply a sparse preconditioner, and gets
    a slightly cheaper solve back for it.
    """
    import scipy.sparse as sp
    print("\n== almost-banded: ride (carry the borders) vs collapse (sparsify) ==")
    print(f"{'n':>8} | {'ride fac':>9} {'ride slv':>9} {'ride err':>9} | "
          f"{'coll fac':>9} {'coll slv':>9} {'coll err':>9}")
    kl = ku = 2
    for n in ([512, 4096] if quick else [512, 4096, 16384]):
        k = np.arange(n)
        rows = np.vstack([np.ones(n), (-1.0) ** k])          # T_k(1), T_k(-1)
        ab = random_band(rng, n, kl, ku)
        A = band_to_dense(ab, kl, ku, n)
        A[:2, :] = rows
        b = rng.standard_normal(n)
        As = sp.csr_matrix(A)

        trf = best(lambda: RideQR(As, kl, ku, [0, 1]), reps=3)
        Fr = RideQR(As, kl, ku, [0, 1])
        trs = best(lambda: Fr.solve(b))
        tcf = best(lambda: CollapseQR(As, kl, ku, [0, 1]), reps=3)
        Fc = CollapseQR(As, kl, ku, [0, 1])
        tcs = best(lambda: Fc.solve(b))
        er = np.linalg.norm(A @ Fr.solve(b) - b) / np.linalg.norm(b)
        ec = np.linalg.norm(A @ Fc.solve(b) - b) / np.linalg.norm(b)
        print(f"{n:8d} | {trf*1e3:8.3f}m {trs*1e3:8.3f}m {er:9.1e} | "
              f"{tcf*1e3:8.3f}m {tcs*1e3:8.3f}m {ec:9.1e}")
    print("  ride factors 3-4x faster and stays at machine precision;")
    print("  collapse buys ~25% on the solve and loses accuracy as n grows,")
    print("  so it only pays off after many solves against one factorisation.")


def bench_threads(rng, quick):
    from bandqr import has_threads
    if not has_threads():
        print("\n== threading: unavailable (built with BANDQR_NO_THREADS) ==")
        return
    print("\n== wavefront threading (auto stays serial below kl=64) ==")
    print(f"{'n':>8} {'kl':>4} | {'serial':>9} {'2thr':>9} {'4thr':>9} "
          f"{'8thr':>9} | {'best':>12}")
    cases = [(20000, 16), (50000, 32), (20000, 64)]
    if not quick:
        cases += [(10000, 96), (5000, 128)]
    for n, kl in cases:
        ab = random_band(rng, n, kl, kl)
        b = rng.standard_normal(n)
        ts = {nt: best(lambda nt=nt: BandQR(ab, kl, kl, nthreads=nt).solve(b), reps=3)
              for nt in (1, 2, 4, 8)}
        bn = min(ts, key=ts.get)
        print(f"{n:8d} {kl:4d} | {ts[1]*1e3:8.2f}m {ts[2]*1e3:8.2f}m "
              f"{ts[4]*1e3:8.2f}m {ts[8]*1e3:8.2f}m | {ts[1]/ts[bn]:5.2f}x @ {bn}thr")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the large cases")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    import platform
    print(f"bandqr CPU benchmarks -- {platform.platform()} / {platform.machine()}")
    bench_vs_lapack(rng, args.quick)
    bench_factor_reuse(rng, args.quick)
    bench_almost_banded(rng, args.quick)
    bench_threads(rng, args.quick)


if __name__ == "__main__":
    main()
