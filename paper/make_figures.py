"""Publication figures for the bandqr paper (matplotlib -> PDFs in ./figures/).

    python paper/make_figures.py

bandqr-only figures (conditioning, bandwidth) are computed LIVE against the
current library.  The cross-tool comparisons (scaling, crossover, GPU) use
numbers recorded in DATA below.

PROVENANCE: those recorded numbers were measured by the Dedalus/ApproxFun
comparison notebook, which now lives in archive/dedalus/benchmarks.ipynb -- the
library itself no longer depends on Dedalus.  Re-run that notebook in an
environment that has Dedalus and Julia to refresh them, and note that the GPU
row predates the rewritten Metal backend, so bench/bench_gpu.py is the current
authority on GPU numbers.
"""
import os, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 14,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    "lines.linewidth": 2.2, "lines.markersize": 6, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
})
COL = {"bandqr": "C0", "Dedalus SuperLU": "C1", "ApproxFun QR": "C3"}
MK  = {"bandqr": "o", "Dedalus SuperLU": "s", "ApproxFun QR": "^"}

def _xticks(ax, vals):                       # plain-integer ticks at the data points only
    sf = ScalarFormatter(); sf.set_scientific(False); sf.set_useOffset(False)
    ax.xaxis.set_major_locator(FixedLocator([float(v) for v in vals]))
    ax.xaxis.set_major_formatter(sf)
    ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_minor_formatter(NullFormatter())

def _clean_logy(ax):                         # keep decade labels, drop crowded minor labels
    ax.yaxis.set_minor_formatter(NullFormatter())

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))

# ----------------------------- recorded data (benchmarks.ipynb) ----------------
NS = np.array([1024, 4096, 8192, 16384])
THREEWAY = {  # factor / solve, milliseconds (assembly excluded)
    "bandqr":          {"factor": [0.025, 0.082, 0.159, 0.328], "solve": [0.028, 0.092, 0.174, 0.355]},
    "Dedalus SuperLU": {"factor": [0.220, 0.784, 1.643, 3.294], "solve": [0.016, 0.062, 0.119, 0.205]},
    "ApproxFun QR":    {"factor": [0.109, 0.429, 0.872, 1.720], "solve": [0.050, 0.214, 0.426, 0.852]},
}
GPU = {"batch": np.array([1024, 4096, 8192, 16384]),
       "gpu_ms": np.array([3.54, 8.64, 20.04, 23.01]),
       "cpu_ms": np.array([29.8, 211.9, 746.2, 1548.7]),
       "speedup": np.array([8.4, 24.5, 37.2, 67.3])}
# bandqr on the FULL real Dedalus operator (dense BC rows + tau columns), ride + Woodbury.
# Same matrix as the Dedalus SuperLU column -> the fair, like-for-like comparison.
# (After vectorizing the row permutation + staying in CSR; crosses below SuperLU at large N.)
RIDETAU = {"factor": [0.56, 1.03, 1.76, 3.11], "solve": [0.031, 0.091, 0.171, 0.332]}

# ----------------------------- Fig 1: performance (factor | solve | amortised) --
def fig_performance():
    # FAIR comparison: all three solve the SAME full almost-banded operator
    # (banded bulk + dense BC rows + tau columns); assembly excluded, nothing else.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    series = [
        ("bandqr",          RIDETAU,                     "C0", "o"),
        ("Dedalus SuperLU", THREEWAY["Dedalus SuperLU"], "C1", "s"),
        ("ApproxFun QR",    THREEWAY["ApproxFun QR"],    "C3", "^"),
    ]
    for lbl, d, c, m in series:
        a1.loglog(NS, d["factor"], marker=m, color=c, label=lbl)
        a2.loglog(NS, d["solve"], marker=m, color=c, label=lbl)
    a1.set(xlabel="problem size  $N$", ylabel="factorisation [ms]", title="factor")
    a2.set(xlabel="problem size  $N$", ylabel="solve [ms]", title="solve")
    for a in (a1, a2):
        a.grid(True, which="both", alpha=0.2); _xticks(a, NS); _clean_logy(a)
    a1.legend(frameon=False, loc="upper left")
    save(fig, "performance.pdf")

# ----------------------------- Fig 3: conditioning (ride vs collapse) -----------
def _helm(n, c=5.0):
    import scipy.sparse as sp
    j = np.arange(n)
    D2 = sp.diags(2.0 * (np.arange(n - 2) + 2), 2, shape=(n, n))
    S0 = sp.diags(np.r_[1.0, 0.5 * np.ones(n - 1)], 0) + sp.diags(-0.5 * np.ones(n - 2), 2, shape=(n, n))
    S1 = sp.diags(1.0 / (1.0 + j), 0) + sp.diags(-1.0 / (np.arange(n - 2) + 3.0), 2, shape=(n, n))
    Lc = (D2 - c * (S1 @ S0)).tocoo(); m = Lc.row < n - 2
    rows = np.r_[np.zeros(n, int), np.ones(n, int), Lc.row[m] + 2]
    cols = np.r_[j, j, Lc.col[m]]
    data = np.r_[np.ones(n), (-1.0) ** j, Lc.data[m]]
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

def fig_conditioning():
    Ns = [256, 512, 1024, 2048, 4096, 8192, 16384]
    try:
        import scipy.sparse.linalg as spla
        import sys, os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))), 'src'))
        from bandqr import solve_ride, solve_precond_rows
        rng = np.random.default_rng(0); ride, coll = [], []
        for n in Ns:
            M = _helm(n); b = rng.standard_normal(n); xr = spla.spsolve(M.tocsc(), b)
            ride.append(np.linalg.norm(solve_ride(M, 2, 2, b, [0, 1]) - xr) / np.linalg.norm(xr))
            coll.append(np.linalg.norm(solve_precond_rows(M, 2, 2, b, [0, 1]) - xr) / np.linalg.norm(xr))
    except Exception as e:
        print("conditioning: recorded data (", str(e)[:50], ")")
        Ns, ride, coll = [1024, 4096, 16384], [8.7e-16, 1.8e-15, 4.5e-15], [1.4e-14, 3.5e-14, 1.8e-10]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.loglog(Ns, coll, "s-", color="C3", label="collapse (sparsify the BCs)")
    ax.loglog(Ns, ride, "o-", color="C0", label="ride (carry the BCs along)")
    ax.set(xlabel="problem size  $N$", ylabel="relative error  $\\|x-x_{\\rm ref}\\|/\\|x_{\\rm ref}\\|$")
    ax.legend(frameon=False, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
    _xticks(ax, Ns[::2]); _clean_logy(ax)
    save(fig, "conditioning.pdf")

# ----------------------------- Fig 4: bandwidth (method vs kl) ------------------
# pybanded (Dedalus project's Householder banded QR). Its BandedReflector predates a
# newer scipy that namespace-checks LinearOperator; the shim below restores it.
def _enable_pybanded():
    import pybanded, scipy.sparse as sp  # noqa: F401
    import scipy._external.array_api_compat.numpy as _xpnp
    from pybanded.reflector import BandedReflector
    from pybanded.matrix import BandedMatrix
    BandedReflector._xp = _xpnp; BandedMatrix._xp = _xpnp
    return pybanded, sp

def fig_bandwidth():
    kls = [2, 4, 8, 16, 32]
    pyb_rec = [0.70, 0.83, 1.37, 2.94, 9.86]   # recorded pybanded (if not importable here)
    try:
        import scipy.linalg as sla
        from bandqr import BandQR, dense_to_band
        n = 10000; rng = np.random.default_rng(0)
        def best(fn, r=7):
            return min((lambda t: (fn(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(r)) * 1e3
        try:
            pybanded, sp = _enable_pybanded(); pyb = []
        except Exception:
            pyb = None
        giv, hh, lu = [], [], []
        for kl in kls:
            A = np.zeros((n, n))
            for i in range(n):
                for j in range(max(0, i - kl), min(n - 1, i + kl) + 1): A[i, j] = rng.standard_normal()
                A[i, i] += 2 * kl + 5
            b = rng.standard_normal(n); ab = dense_to_band(A, kl, kl)
            absp = np.zeros((2 * kl + 1, n))
            for i in range(n):
                for j in range(max(0, i - kl), min(n - 1, i + kl) + 1): absp[kl + i - j, j] = A[i, j]
            giv.append(best(lambda: BandQR(ab, kl, kl).solve(b)))
            hh.append(best(lambda: BandQR(ab, kl, kl, householder=True).solve(b)))
            lu.append(best(lambda: sla.solve_banded((kl, kl), absp, b)))
            if pyb is not None:
                Apb = pybanded.BandedMatrix.from_sparse(sp.dia_matrix(A))
                pyb.append(best(lambda: pybanded.BandedQR(Apb).solve(b)))
        if pyb is None:
            print("bandwidth: pybanded not importable -> recorded pybanded numbers"); pyb = pyb_rec
    except Exception as e:
        print("bandwidth: recorded data (", str(e)[:50], ")")
        kls = [2, 4, 8, 16, 32]; giv = [0.33, 0.67, 1.63, 3.60, 10.15]
        hh = [0.44, 0.62, 1.07, 2.29, 6.91]; lu = [0.42, 0.54, 0.84, 1.33, 2.82]; pyb = pyb_rec
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.loglog(kls, giv, "o-", color="C0", label="bandqr Givens")
    ax.loglog(kls, hh, "D-", color="C4", label="bandqr Householder")
    ax.loglog(kls, pyb, "v-", color="C1", label="pybanded (Householder)")
    ax.loglog(kls, lu, "s-", color="C2", label="LAPACK band-LU")
    ax.set(xlabel="bandwidth  $k_l=k_u$", ylabel="factor + solve [ms]", title="$n=10000$")
    ax.legend(frameon=False, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
    _xticks(ax, kls); _clean_logy(ax)
    save(fig, "bandwidth.pdf")

# ----------------------------- Fig 5: GPU batched (throughput | speedup) --------
def fig_gpu():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.loglog(GPU["batch"], GPU["cpu_ms"], "s-", color="C1", label="CPU loop")
    a1.loglog(GPU["batch"], GPU["gpu_ms"], "o-", color="C0", label="GPU batched")
    a1.set(xlabel="batch size", ylabel="time [ms]", title="batched banded solve")
    a1.legend(frameon=False, loc="upper left"); a1.grid(True, which="both", alpha=0.2)
    _xticks(a1, GPU["batch"]); _clean_logy(a1)
    a2.semilogx(GPU["batch"], GPU["speedup"], "o-", color="C0")
    a2.set(xlabel="batch size", ylabel="GPU speedup  [$\\times$]", title="speedup vs CPU loop")
    a2.grid(True, which="both", alpha=0.2); _xticks(a2, GPU["batch"])
    save(fig, "gpu.pdf")

if __name__ == "__main__":
    fig_performance()
    fig_conditioning()
    fig_bandwidth()
    fig_gpu()
    print("done ->", OUT + "/")
