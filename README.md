# bandqr

A fast, deterministic **banded and almost-banded QR** solver — a small C kernel
(no LAPACK, no pivoting, no densifying) with a Python front-end, plus an optional
batched GPU backend. Built for the banded and *almost*-banded linear systems that
spectral / tau methods (e.g. Dedalus) produce.

Why QR rather than LU: the fill is **deterministic** (`ku+kl` upper, always — no
pivoting, no data-dependent growth), the factorization is **well-conditioned**,
and for narrow bands it factors as fast as or faster than LAPACK band-LU. The
fast, predictable *factorization* is the headline — it makes frequent
re-factorization (e.g. adaptive timestepping) affordable.

```python
from bandqr import solve
x = solve(A, b)            # A banded or almost-banded (dense/scipy.sparse); auto-detects structure
```

---

## Install

```sh
make            # builds libbandqr.dylib (universal2: arm64 + x86_64)
python -c "import bandqr"
```

Requirements: a C compiler (`cc`/`clang`/`gcc`) and `numpy`; `scipy` for the
sparse/almost-banded helpers. macOS/Linux (Windows untested).

**Single-file sharing.** `bandqr_standalone.py` bundles the C kernel + Python into
one self-compiling file — copy it anywhere and `import bandqr_standalone`; it
compiles the embedded C on first import. Regenerate it with `python
make_standalone.py` after editing the sources.

---

## Methods

`solve(A, b, method="auto", ...)` and `factorize(A, method="auto", ...)` take a
`method` flag:

| method | what | use when |
|---|---|---|
| `auto` | dispatch by structure/bandwidth | default |
| `givens` | Givens band QR | purely banded, narrow |
| `householder` | Householder band QR | purely banded, **wide** (~25% faster factor) |
| `collapse` | auto BC-row **recurrence collapse** → one band QR | dense BC rows, fastest |
| `ride` | almost-banded QR, BCs **carried along** (Olver–Townsend) | dense BC rows, **best conditioning** |
| `bordered` | Woodbury against the in-band part | dense rows **and** tau columns (non-singular bulk) |
| `lu` | banded LU (`scipy.linalg.solve_banded`) | comparison / fastest re-solve |

`auto` picks Givens/Householder by bandwidth for purely banded matrices,
`collapse` for dense rows, `bordered` when tau columns are present.

### The novel bit: automatic BC-row collapse
`detect_recurrence(rows)` reads the shortest linear recurrence a boundary row
satisfies **straight from its values** (null space of a Hankel matrix — Prony /
Berlekamp–Massey flavor). Chebyshev evaluations → `[1, -2x₀, 1]`, ends → `[±1,1]`,
derivatives → `(1-z)ᵏ`. `collapse` uses it as a preconditioner that sparsifies the
BC rows into the band, so the solve is a single banded QR — no analytic
basis knowledge required.

### `collapse` vs `ride`
Two ways to handle dense BC rows: **sparsify** them (`collapse`) or **carry them
along** in the factorization (`ride`). `ride` is `R = QᴴM` exactly (no
preconditioner) and is markedly better conditioned at large N:

| N | `ride` | `collapse` |
|---|---|---|
| 1024 | 1e-15 | 4e-14 |
| 16384 | **5e-15** | 4e-10 |

---

## Data types

`float64` and `complex128`, from one templated C source (`bandqr_impl.h`). The
dtype **follows the input** — real arrays use the real path (the default,
unchanged); complex128 arrays use the complex (unitary Givens / Householder)
path. No flag needed.

```python
x = solve(A_complex, b_complex)     # complex128, automatically
```

---

## Factor once, solve many

The expensive factorization is reused across right-hand sides (the timestepping
win — reusing across 50 RHS is 8–100× faster than re-factoring each):

```python
from bandqr import factorize
F = factorize(A, method="ride", dense_rows=[0, 1])   # factor once
for b in rhs_stream:
    x = F.solve(b)                                    # solve many (b may be (n,) or (n, nrhs))
```

Reusable objects: `BandQR` (banded), `RideQR`, `CollapseQR`. `factorize()` returns
the right one for the structure.

---

## Almost-banded matrices

Spectral operators are *almost*-banded: a banded bulk + a few dense rows
(boundary conditions) + a few dense columns (tau corrections).
`detect_structure(A)` infers `(kl, ku, dense_rows, dense_cols)`; pass them
explicitly to override. Dense **rows** → `collapse`/`ride`; rows **and** columns →
`bordered`.

---

## GPU — batched solve (the partridge)

`bandqr_gpu.solve_batched(ab, b, kl, ku)` solves **many independent same-shape
banded systems at once** on the Apple GPU (Metal via MLX) — one thread per system,
pivot-free so no warp divergence. This is the spectral regime: one banded ODE per
Fourier mode → thousands of systems. ~5–35× over a CPU loop, scaling with batch.

```python
from bandqr_gpu import solve_batched
X = solve_batched(ab, b, kl, ku)    # ab: (n, kl+ku+1, batch) fp32, b: (n, batch)
```

Needs a native-arm64 Python with `mlx` (`pip install mlx`). **fp32, pure banded**
(first version) — for fp64 / ill-conditioned / almost-banded systems use the CPU
path.

---

## Tests

```sh
python test_bandqr.py      # correctness + benchmarks, all methods and dtypes
```

---

## Scope / roadmap

- **GPU**: fp32 + pure banded only; fp64, complex, almost-banded, and
  factor-reuse on GPU are future work.
- **`ride`**: dense rows must sit near the top (it widens `kl` to reach them);
  permute bottom-placed BC rows up first.
- **Well-conditioned dense rows + tau columns together** (vs `bordered`'s Woodbury,
  which needs a non-singular bulk) is roadmap.
- **No pivoting** by design — for ill-conditioned/rank-deficient systems use a
  general solver.

## Files

`bandqr_impl.h` (templated kernel) · `bandqr.c` (driver, instantiates `_d`/`_z`) ·
`bandqr.h` · `bandqr.py` (front-end) · `bandqr_gpu.py` (GPU) ·
`bandqr_standalone.py` (single-file) · `make_standalone.py` · `test_bandqr.py` ·
`demo.ipynb`.
