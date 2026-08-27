# bandqr

A fast, deterministic **banded and almost-banded QR** solver — a small C kernel
(no LAPACK, no pivoting, no densifying) with a Python front end, plus a batched
Metal backend for solving thousands of systems at once.

```python
from bandqr import solve
x = solve(A, b)          # A banded or almost-banded, dense or scipy.sparse
```

**Why QR rather than LU.** The fill is *deterministic* — `ku + kl` upper,
always, with no pivoting and no value-dependent growth — so the cost is a
function of the sparsity pattern alone. The factorisation is unconditionally
backward stable, and for narrow bands it factors as fast as or faster than
LAPACK's band LU. A predictable, fast *factorisation* is the point: it makes
re-factorising often (adaptive timestepping, parameter sweeps, continuation)
affordable.

---

## Install

```sh
pip install -e .          # builds the C kernel through setuptools
```

or, from a checkout, without installing:

```sh
make && python -c "import sys; sys.path.insert(0,'src'); import bandqr"
```

Requirements: a C compiler and `numpy`. `scipy` for the sparse and
almost-banded helpers, `mlx` for the GPU backend. macOS and Linux; Windows
builds serial-only (`-DBANDQR_NO_THREADS`, set automatically).

---

## Methods

`solve(A, b, method=...)` and `factorize(A, method=...)`:

| method | what it does | when to reach for it |
|---|---|---|
| `auto` | dispatch on detected structure | the default |
| `givens` | Givens band QR | purely banded, narrow |
| `householder` | Householder band QR | purely banded, wide |
| `lq` | `L Q` — the QR of the transpose | structure lives in the columns |
| `ride` | almost-banded QR, borders carried through | dense borders — the accurate *and* faster choice |
| `collapse` | sparsify borders by their recurrence, then one QR | dense borders, many solves per factorisation |
| `bordered` | Woodbury against the in-band part | rows *and* tau columns, invertible bulk |
| `lu` | banded LU (`scipy.linalg.solve_banded`) | comparison, fastest re-solve |

`auto` picks Givens/Householder by bandwidth for purely banded matrices, and
`ride` whenever dense borders are present.

### The novel bit: automatic border collapse

`detect_recurrence(rows)` reads the shortest linear recurrence a boundary row
satisfies **straight from its values** — as the null space of a Hankel matrix of
sliding windows (Prony / Berlekamp–Massey, with an SVD in place of exact
arithmetic). Chebyshev evaluations give `[1, -2x₀, 1]`, endpoints `[±1, 1]`,
derivatives `(1-z)ᵏ`. `collapse` uses it as a preconditioner that sparsifies the
borders into the band, so the solve becomes a single banded QR — with no
analytic knowledge of the basis.

### `ride` vs `collapse`

Two ways to handle dense borders: **carry** them through the factorisation, or
**sparsify** them first. `ride` is `R = QᴴM` exactly, with no preconditioner.
Measured (`bench/bench_cpu.py`, kl=ku=2, Chebyshev BC rows):

| n | ride factor | ride solve | ride err | collapse factor | collapse solve | collapse err |
|---|---|---|---|---|---|---|
| 512 | 0.11 ms | 0.019 ms | 3e-16 | 0.32 ms | 0.017 ms | 3e-14 |
| 4096 | 0.40 ms | 0.116 ms | 3e-16 | 1.51 ms | 0.086 ms | 6e-13 |
| 16384 | 1.40 ms | 0.410 ms | 3e-16 | 5.59 ms | 0.321 ms | 3e-11 |

`ride` factors 3–4× faster and stays at machine precision. `collapse` buys
about 25% on each *solve* — it pays off only after many right-hand sides against
one factorisation, and you pay for it in conditioning, increasingly so with `n`.

---

## Data types

`float64` and `complex128`, from one templated C source. **The dtype follows the
input** — no flag:

```python
x = solve(A_complex, b_complex)     # complex128 automatically
```

---

## Factor once, solve many

The expensive factorisation is reused across right-hand sides:

```python
from bandqr import factorize
F = factorize(A, method="ride", dense_rows=[0, 1])
for b in rhs_stream:
    x = F.solve(b)                   # b may be (n,) or (n, nrhs)
```

Measured on a banded system (n=50000, kl=8): **17× at 10 right-hand sides, 42×
at 50**, against re-factorising each time.

`BandQR` also exposes the factors directly: `apply_Q` / `apply_Qt`, `solve_R` /
`solve_Rt`, `R_band()`, `R_sparse()`, `Q_dense()`, and `diag_R`.

---

## Almost-banded matrices

Spectral operators are *almost* banded: a banded bulk, a few dense rows
(boundary conditions), and sometimes a few dense columns (tau corrections).
`detect_structure(A)` infers `(kl, ku, dense_rows, dense_cols)` by measuring how
populated each diagonal is — a bulk diagonal is full along its length, one
touched only by a border holds a handful of entries — then greedily covering
whatever falls outside. It returns a report you can inspect:

```python
>>> detect_structure(A).describe()
'400x400  nnz=4973  band=(3,2)  dense_rows=2  dense_cols=2'
```

Every entry point accepts explicit `kl`/`ku`/`dense_rows`/`dense_cols`
overrides, because "is this row a boundary condition or just a wide row?" has no
value-free answer.

Dense rows near the *bottom* are permuted to the top automatically, so there is
no longer a constraint on where boundary rows sit.

---

## GPU — batched solve

`bandqr.solve_batched` solves **many independent same-shape systems at once** on
the Apple GPU: one thread per system, pivot-free so every lane takes the same
instruction path, batch laid out fastest so neighbouring threads read
neighbouring floats. This is the spectral regime — one banded ODE per Fourier
mode means thousands of systems per timestep.

```python
from bandqr import solve_batched
X = solve_batched(ab, b, kl, ku)         # ab: (n, kl+ku+1, batch), b: (n, batch)
```

Measured on an **Apple M3 Max** (`bench/bench_gpu.py`), against a CPU loop over
the batch:

| n | kl | batch | CPU loop | GPU | speedup |
|---|---|---|---|---|---|
| 128 | 2 | 256 | 3.2 ms | 1.9 ms | 1.7× |
| 128 | 2 | 1024 | 13.7 ms | 2.4 ms | 5.8× |
| 192 | 3 | 4096 | 90.8 ms | 5.5 ms | 16.7× |
| 256 | 4 | 2048 | 67.0 ms | 7.4 ms | 9.1× |

### fp32 speed, float64 accuracy

Apple GPUs have no float64. Rather than accept fp32 answers, the factorisation
is fp32 and the residual is formed in **float64 against the original operator**,
so mixed-precision iterative refinement recovers full double accuracy:

| n | kl | fp32 | +1 step | +2 steps |
|---|---|---|---|---|
| 128 | 2 | 1.8e-07 | 7.5e-14 | 7.5e-14 |
| 192 | 3 | 2.8e-07 | 1.1e-13 | 1.1e-13 |
| 256 | 4 | 3.0e-07 | 1.8e-13 | 1.8e-13 |

One correction step, and you are at double precision. This holds while
`cond(A)·2⁻²⁴ < 1`, i.e. up to `cond ~ 1e7`; past that, refinement stalls and
says so.

### Also on the GPU

- **Factor once, solve many** — `BatchedBandQR` keeps the rotations; **2.9×**
  per subsequent solve versus re-factorising.
- **Almost-banded batches** — `BatchedRideQR` carries dense boundary rows
  through the rotations for the whole batch, with `L` shared across systems.
- **Shifted form** — `solve_batched_shifted(ab, shifts, b, kl, ku)` for one
  shared band plus a per-system diagonal shift, exactly what a separable
  spectral solve produces. Uploads `(n, w)` instead of `(n, w, batch)`:
  58.7 MB → 14.3 kB at n=256, batch=4096.
- **complex64**, from the same templated source as the real kernel.
- **One compile** — kernels are cached on `(kl, ku, dtype)` alone; `n`, batch
  size and RHS count are runtime arguments.

### No GPU?

`solve_batched(..., backend="numpy")` is a vectorised host implementation that
runs anywhere, works in float64 directly, and is what the GPU kernel is tested
against. It vectorises across the batch (the rotation sequence is identical for
every system — that is what pivot-freedom buys), so it suits moderate `n` with a
large batch.

---

## Accuracy tooling

Without pivoting there is no growth factor to inspect, so:

```python
from bandqr import condest, solve
condest(F)                       # Hager-Higham 1-norm condition estimate
x = solve(A, b, refine=3)        # refinement against compensated residuals
```

`compensated_residual` evaluates `b - A x` in double-double arithmetic (Dekker's
exact product, Knuth's exact sum), so the residual carries full relative
accuracy even when `‖Ax‖` swamps `‖r‖` — which is what makes refinement able to
make progress at all. `F.rank_deficient()` is the after-the-fact check that
pivoting would otherwise have given you.

---

## Threading

`nthreads=0` (auto) is **serial unless threading has been measured to win**.
Narrow-band factorisation streams the band from DRAM and is bound by that, not
by arithmetic, so extra threads buy nothing and the wavefront's cross-column
synchronisation is pure overhead. Measured on an M3 Max, 8 threads vs serial:

| kl | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| speedup | 1.00× | 1.00× | 1.67× | 2.28× | 2.78× |

so auto stays serial below `kl=64` (override with `BANDQR_THREAD_MIN_KL`).
Threaded `R` is **bit-identical** to serial `R` — the factorisation is
deterministic, and the test suite asserts exact equality.

---

## Tests and benchmarks

```sh
make test            # 194 tests
make bench           # CPU: vs LAPACK, factor reuse, ride/collapse, threading
make bench-gpu       # GPU: throughput, reuse, refinement, shifted form
```

The GPU kernel is checked against the NumPy batched backend; the compensated
residual against exact rational arithmetic; the condition estimator against
`numpy.linalg.cond`.

---

## Scope and limits

- **No pivoting, by design.** For rank-deficient systems use a general solver;
  `rank_deficient()` will tell you.
- **GPU is fp32** (Apple hardware has no float64); refinement recovers double
  accuracy up to `cond ~ 1e7`. Beyond that, use the CPU path.
- **`collapse` needs recurrence-structured borders.** Random dense rows satisfy
  no recurrence; it raises and points you at `ride`, which needs none.
- **`bordered` needs a non-singular in-band bulk.** Differential operators whose
  diagonal vanishes need `ride` instead.
- **Structure detection is a heuristic** on a genuinely ambiguous problem. Pass
  the structure explicitly when it matters.

## Layout

```
src/bandqr/          package
  csrc/              the C kernel (one templated source, real + complex)
  banded.py          BandQR, LQ            almost.py    Ride*, Collapse*, Bordered*
  gpu.py             Metal batched         batched.py   NumPy batched + dispatch
  recurrence.py      detect_recurrence     structure.py detect_structure
  refine.py          compensated residuals, refinement, condition estimates
tests/               pytest suite          bench/       benchmarks
paper/               write-up              archive/     superseded Dedalus integration
```
