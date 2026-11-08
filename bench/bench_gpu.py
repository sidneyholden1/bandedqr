"""GPU benchmarks: batched banded solve on Metal against the CPU.

    python bench/bench_gpu.py [--quick]

The comparison that matters is against a CPU loop over the batch, because that
is what a solver without a batched kernel actually does.
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")

from bandedqr import BandQR  # noqa: E402
from bandedqr import gpu as gpu_mod  # noqa: E402
from bandedqr.batched import BatchedBandQR as NumpyBatched  # noqa: E402


def best(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def make(rng, n, kl, ku, batch):
    ab = rng.standard_normal((n, kl + ku + 1, batch))
    ab[:, kl, :] += kl + ku + 4
    return ab, rng.standard_normal((n, batch))


def bench_throughput(rng, quick):
    print("\n== batched banded solve: CPU loop vs GPU ==")
    print(f"{'n':>5} {'kl':>3} {'batch':>6} | {'CPU loop':>10} {'numpy btch':>10} "
          f"{'GPU fused':>10} {'GPU refined':>11} | {'vs CPU':>7}")
    cases = [(128, 2, 2, 256), (128, 2, 2, 1024), (192, 3, 3, 1024)]
    if not quick:
        cases += [(192, 3, 3, 4096), (256, 4, 4, 2048), (384, 2, 2, 4096)]
    for n, kl, ku, batch in cases:
        ab, b = make(rng, n, kl, ku, batch)

        def cpu_loop():
            out = np.empty((n, batch))
            for s in range(batch):
                out[:, s] = BandQR(np.ascontiguousarray(ab[:, :, s]), kl, ku).solve(
                    np.ascontiguousarray(b[:, s]))
            return out

        t_cpu = best(cpu_loop, reps=2, warm=0)
        t_np = best(lambda: NumpyBatched(ab, kl, ku).solve(b), reps=2, warm=0)
        t_fu = best(lambda: gpu_mod.solve_batched(ab, b, kl, ku, refine=0, fused=True))
        t_rf = best(lambda: gpu_mod.BatchedBandQR(ab, kl, ku).solve(b, refine=2),
                    reps=3)
        print(f"{n:5d} {kl:3d} {batch:6d} | {t_cpu*1e3:9.2f}m {t_np*1e3:9.2f}m "
              f"{t_fu*1e3:9.2f}m {t_rf*1e3:10.2f}m | {t_cpu/t_fu:6.1f}x")


def bench_reuse(rng, quick):
    print("\n== factor once, solve many (the timestepping case) ==")
    print(f"{'n':>5} {'kl':>3} {'batch':>6} | {'factor':>9} {'solve':>9} "
          f"{'refactor':>9} | {'reuse gain':>10}")
    cases = [(192, 3, 3, 1024)] + ([] if quick else [(192, 3, 3, 4096),
                                                     (256, 4, 4, 2048)])
    for n, kl, ku, batch in cases:
        ab, b = make(rng, n, kl, ku, batch)
        t_fact = best(lambda: gpu_mod.BatchedBandQR(ab, kl, ku), reps=3)
        F = gpu_mod.BatchedBandQR(ab, kl, ku)
        t_solve = best(lambda: F.solve_fp32(b))
        t_fused = best(lambda: gpu_mod.solve_batched(ab, b, kl, ku, refine=0,
                                                     fused=True))
        print(f"{n:5d} {kl:3d} {batch:6d} | {t_fact*1e3:8.2f}m {t_solve*1e3:8.2f}m "
              f"{t_fused*1e3:8.2f}m | {t_fused/t_solve:9.1f}x")


def bench_accuracy(rng):
    print("\n== mixed precision: what each refinement step buys ==")
    print(f"{'n':>5} {'kl':>3} | {'fp32':>10} {'+1 step':>10} {'+2 steps':>10} "
          f"{'+3 steps':>10}")
    for n, kl, ku, batch in [(128, 2, 2, 256), (192, 3, 3, 512), (256, 4, 4, 256)]:
        ab, b = make(rng, n, kl, ku, batch)
        ref = NumpyBatched(ab, kl, ku).solve(b)
        scale = np.abs(ref).max()
        F = gpu_mod.BatchedBandQR(ab, kl, ku)
        errs = []
        for r in (0, 1, 2, 3):
            x = F.solve(b, refine=r) if r else F.solve_fp32(b)
            errs.append(np.abs(x - ref).max() / scale)
        print(f"{n:5d} {kl:3d} | " + " ".join(f"{e:10.1e}" for e in errs))
    print("  (fp32 factorisation, float64 residuals: full double accuracy for")
    print("   the price of one or two extra solves.)")


def bench_shifted(rng):
    print("\n== shifted form: shared band + per-system diagonal shift ==")
    for n, kl, ku, batch in [(192, 2, 2, 1024), (256, 3, 3, 4096)]:
        base = rng.standard_normal((n, kl + ku + 1))
        base[:, kl] += 8.0
        shifts = np.linspace(0.5, 100.0, batch)
        full = np.repeat(base[:, :, None], batch, axis=2)
        full[:, kl, :] += shifts[None, :]
        b = rng.standard_normal((n, batch))
        t_full = best(lambda: gpu_mod.solve_batched(full, b, kl, ku, refine=0,
                                                    fused=True))
        t_sh = best(lambda: gpu_mod.BatchedBandQR(base, kl, ku,
                                                  shifts=shifts).solve_fp32(b))
        print(f"  n={n} batch={batch}: upload {full.nbytes/1e6:6.2f} MB -> "
              f"{base.nbytes/1e3:.1f} kB   ({t_full*1e3:.2f}m vs {t_sh*1e3:.2f}m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if not gpu_mod.is_available():
        print("No MLX/Metal device available; nothing to benchmark.")
        print("The NumPy batched backend still works: bandedqr.solve_batched(..., "
              "backend='numpy')")
        return
    import platform
    print(f"bandedqr GPU benchmarks -- {platform.platform()} / {platform.machine()}")
    rng = np.random.default_rng(0)
    bench_throughput(rng, args.quick)
    bench_reuse(rng, args.quick)
    bench_accuracy(rng)
    bench_shifted(rng)


if __name__ == "__main__":
    main()
