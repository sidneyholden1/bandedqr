"""Generate benchmarks.ipynb -- the benchmarking notebook (run it to regenerate)."""
import json

MD = "markdown"
CODE = "code"

cells = []
def add(kind, src):
    cells.append({"cell_type": kind, "metadata": {},
                  **({"source": src} if kind == MD else
                     {"source": src, "outputs": [], "execution_count": None})})

add(MD, """# bandqr benchmarks

Run on a kernel that has **Dedalus** and is **native arm64** (e.g. the `dedalus3`
env) — that gets Dedalus, the universal `bandqr` dylib, `mlx` for the GPU table,
and `julia` for the ApproxFun column. The ApproxFun (Julia) column is run via a
**live subprocess** and is **skipped automatically** if Julia isn't installed.

Tables: (1) CPU method gallery, (2) the three-way factor/solve on a real
Helmholtz operator (bandqr vs Dedalus SuperLU vs ApproxFun), (3) GPU batched.""")

add(CODE, """import os; os.environ.setdefault("FI_PROVIDER", "tcp")  # quiet Dedalus's MPI layer
import time, json, subprocess, shutil
import numpy as np, pandas as pd
import scipy.sparse as sp, scipy.linalg as sla
try:                                              # full package, or the single file
    from bandqr import BandQR, RideTauQR, dense_to_band, detect_recurrence
except (ImportError, OSError):
    from bandqr_standalone import BandQR, RideTauQR, dense_to_band, detect_recurrence

def best(fn, reps=15):
    \"\"\"Best-of-`reps` wall time in milliseconds.\"\"\"
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3

print("ready")""")

add(MD, """## 1. CPU method gallery — one-shot banded solve

`bandqr` (Givens / Householder) vs LAPACK band-LU (`scipy.linalg.solve_banded`),
factor+solve, n=20000. Givens wins narrow bands; LAPACK's tuned band-LU pulls
ahead as the band widens (its blocked kernels). Times in ms.""")

add(CODE, """def random_banded(n, kl, ku):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-kl), min(n-1, i+ku)+1): A[i, j] = np.random.standard_normal()
        A[i, i] += kl + ku + 3
    return A
def scipy_ab(A, kl, ku):
    n = A.shape[0]; ab = np.zeros((kl+ku+1, n))
    for i in range(n):
        for j in range(max(0, i-kl), min(n-1, i+ku)+1): ab[ku+i-j, j] = A[i, j]
    return ab

rows = []
for kl in (2, 8, 32):
    n = 20000; A = random_banded(n, kl, kl); b = np.random.standard_normal(n)
    ab, absp = dense_to_band(A, kl, kl), scipy_ab(A, kl, kl)
    rows.append({"kl=ku": kl,
                 "bandqr givens [ms]":      best(lambda: BandQR(ab, kl, kl).solve(b)),
                 "bandqr householder [ms]": best(lambda: BandQR(ab, kl, kl, householder=True).solve(b)),
                 "scipy band-LU [ms]":      best(lambda: sla.solve_banded((kl, kl), absp, b))})
print("n = 20000, factor+solve, best of 15  (all times in milliseconds)")
pd.DataFrame(rows).set_index("kl=ku").round(3)""")

add(MD, """## 2. Factor / solve on the real Dedalus operator

All methods solve the **same** matrix: Dedalus's assembled `L_min` (banded bulk +
dense BC rows + tau columns). Dedalus's matrix assembly is excluded; **factor and
solve are timed separately**, with bandqr's *full* factorisation cost included
(ride + the tau-column Woodbury set-up -- no preprocessing hidden). bandqr is
competitive with Dedalus's sparse-LU, matching its factor time and overtaking it
modestly at large N; LU's triangular solve is the fastest re-solve. Times in ms.

This table needs Dedalus (it builds the real operator) and is skipped otherwise.
The ApproxFun (Julia) column runs its own almost-banded QR via subprocess, skipped
if Julia is absent.""")

add(CODE, """try:
    import dedalus.public as d3
    HAVE_DEDALUS = True
except Exception:
    HAVE_DEDALUS = False
    print("Dedalus not installed -> Table 2 (real-operator comparison) is skipped")

def dedalus_M(N, c=5.0):
    xc = d3.Coordinate('x'); dist = d3.Distributor(xc, dtype=np.float64)
    xb = d3.Chebyshev(xc, size=N, bounds=(-1, 1))
    u = dist.Field(name='u', bases=xb); t1 = dist.Field(name='t1'); t2 = dist.Field(name='t2')
    f = dist.Field(name='f', bases=xb); f['g'] = 1.0
    dx = lambda A: d3.Differentiate(A, xc); lb = xb.derivative_basis(2); lift = lambda A, n: d3.Lift(A, lb, n)
    prob = d3.LBVP([u, t1, t2], namespace=locals())
    prob.add_equation(f"dx(dx(u)) - {c}*u + lift(t1,-1) + lift(t2,-2) = f")
    prob.add_equation("u(x=-1) = 0"); prob.add_equation("u(x=+1) = 0")
    s = prob.build_solver(store_expanded_matrices=True); s.solve()
    return s, s.subproblems[0].L_min.tocsc()

NS = (1024, 4096, 8192, 16384)
pyrows = {}
if HAVE_DEDALUS:
    for N in NS:
        s, Md = dedalus_M(N); n = Md.shape[0]
        dr = [n-2, n-1]; dc = [n-2, n-1]                       # 2 BC rows, 2 tau columns
        b = np.random.default_rng(0).standard_normal(n)
        rt = RideTauQR(Md, 0, 4, dr, dc); ms = s.matsolver(Md, s)   # SAME matrix, both methods
        pyrows[N] = dict(bandqr_factor=best(lambda: RideTauQR(Md, 0, 4, dr, dc)),  # full factor cost
                         bandqr_solve=best(lambda: rt.solve(b)),
                         dedalus_factor=best(lambda: s.matsolver(Md, s)),
                         dedalus_solve=best(lambda: ms.solve(b)))
    print("python columns done (bandqr ride-tau and Dedalus SuperLU on the same L_min)")""")

add(CODE, """# ApproxFun (Olver-Townsend) column -- live Julia subprocess, skipped if Julia absent.
JULIA = shutil.which("julia") or os.path.expanduser("~/.juliaup/bin/julia")
approxfun = None
if os.path.exists(JULIA):
    try:
        out = subprocess.run([JULIA, "bench.jl", *map(str, NS)],
                             capture_output=True, text=True, timeout=900)
        approxfun = {int(k): v for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}
        print("ApproxFun (Julia) column: included")
    except Exception as e:
        print("ApproxFun column skipped (Julia present but failed):", str(e)[:90])
else:
    print("Julia not found -> ApproxFun column skipped. To include it: install juliaup and\\n"
          "  julia -e 'using Pkg; Pkg.add([\\"FastAlmostBandedMatrices\\",\\"BandedMatrices\\",\\"BenchmarkTools\\"])'")

tbl = {}
for N in (NS if pyrows else ()):
    r = pyrows[N]
    row = {("bandqr", "factor [ms]"): r["bandqr_factor"], ("bandqr", "solve [ms]"): r["bandqr_solve"],
           ("Dedalus SuperLU", "factor [ms]"): r["dedalus_factor"], ("Dedalus SuperLU", "solve [ms]"): r["dedalus_solve"]}
    if approxfun:
        row[("ApproxFun QR", "factor [ms]")] = approxfun[N]["factor"]
        row[("ApproxFun QR", "solve [ms]")] = approxfun[N]["solve"]
    tbl[N] = row
if tbl:
    df = pd.DataFrame(tbl).T; df.index.name = "N"
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    print("same operator, Dedalus assembly excluded, full bandqr factor, best of 15  (ms)")
    display(df.round(3))
else:
    print("Table 2 skipped -- needs Dedalus to build the real operator")""")

add(MD, """## 3. GPU batched throughput (the partridge)

Many independent same-shape banded systems solved at once on the Apple GPU vs a
CPU loop. Skipped if `mlx` isn't installed (needs native arm64 + `pip install mlx`).""")

add(CODE, """try:
    import mlx.core as mx
    from bandqr_gpu import solve_batched
    grows = []
    for n, kl, batch in [(512, 2, 1024), (1024, 2, 4096), (1024, 4, 8192), (1024, 4, 16384)]:
        rng = np.random.default_rng(0)
        ab = rng.standard_normal((n, 2*kl+1, batch)).astype(np.float32)
        for d in range(-kl, kl+1):                 # zero the unused band corners
            c = d + kl
            if d > 0:   ab[n-d:, c, :] = 0
            elif d < 0: ab[:-d, c, :] = 0
        ab[:, kl, :] += 2*kl + 3                    # diagonal dominance
        b = rng.standard_normal((n, batch)).astype(np.float32); abg = mx.array(ab)
        solve_batched(abg, b, kl, kl)                      # warm up (compile kernel)
        tg = best(lambda: solve_batched(abg, b, kl, kl), reps=5)
        abc = np.ascontiguousarray(np.transpose(ab, (2, 0, 1)))
        t0 = time.perf_counter()
        for s in range(batch): BandQR(abc[s], kl, kl).solve(b[:, s])
        tc = (time.perf_counter() - t0) * 1e3
        grows.append({"n": n, "kl": kl, "batch": batch,
                      "GPU batched [ms]": round(tg, 2), "CPU loop [ms]": round(tc, 1),
                      "speedup [x]": round(tc/tg, 1)})
    display(pd.DataFrame(grows).set_index(["n", "kl", "batch"]))
except Exception as e:
    print("GPU table skipped (needs native arm64 + mlx):", str(e)[:90])""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("benchmarks.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote benchmarks.ipynb")
