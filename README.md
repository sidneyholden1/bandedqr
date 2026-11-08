# bandedqr

`bandedqr` is a research implementation of QR factorization for banded and
almost-banded linear systems. It combines a small C kernel with a Python API and
batched accelerator backends for workloads containing many independent systems
of the same shape.

The project is motivated by spectral discretizations, where a multidimensional
problem often separates into thousands of narrow-banded one-dimensional solves.
QR avoids value-dependent pivoting, so the storage pattern and operation count
are predictable from the bandwidth. The practical questions—where this is
faster than established alternatives and over what conditioning range mixed
precision remains reliable—are treated as measurements, not assumptions.

> **Project status:** research prototype. CPU correctness has been exercised on
> macOS and Linux, including an immutable Rusty campaign. Metal is implemented
> and has preliminary development-machine results. The CUDA backend and the
> reproducible CPU/CUDA scaling campaign are implemented; public performance
> numbers will be added only after the frozen Rusty campaign passes.

## What is implemented

- Real (`float64`) and complex (`complex128`) banded QR in C.
- Givens and Householder factorizations, plus transpose-oriented LQ.
- Factor-once/solve-many APIs and explicit access to `Q` and `R` operations.
- Almost-banded solvers for dense boundary rows and tau columns.
- Structure and boundary-recurrence detection, with explicit overrides.
- Compensated residuals, iterative refinement, condition estimation, and
  rank-deficiency checks.
- Portable NumPy batching, Apple Metal/MLX batching, and NVIDIA CUDA/CuPy
  batching.
- Reproducible conformance and scaling drivers with immutable source,
  provenance, raw JSON, checkpointing, and full-allocation accounting.

CUDA currently covers purely banded, shifted, multi-right-hand-side, and
factor-reuse cases. Batched dense-row/Ride factorization is implemented on
Metal but remains future CUDA work.

## Quick start

Install from a checkout with a C compiler available:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

Solve a dense or SciPy sparse matrix; the front end detects its structure:

```python
import numpy as np
from bandedqr import solve

n = 500
A = (
    np.diag(np.full(n, 4.0))
    + np.diag(np.ones(n - 1), 1)
    + np.diag(np.ones(n - 1), -1)
)
b = np.ones(n)
x = solve(A, b)

relative_residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
```

For production work, pass the known bandwidth explicitly instead of relying on
heuristic structure detection.

## Factor once, solve many

```python
from bandedqr import factorize

factor = factorize(A, kl=1, ku=1, method="givens")
X = factor.solve(B)  # B may contain several right-hand sides
```

`BandQR` also exposes `apply_Q`, `apply_Qt`, `solve_R`, `solve_Rt`,
`R_band`, `R_sparse`, `Q_dense`, and `diag_R`.

## Batched systems

The batch axis is last so adjacent accelerator threads access adjacent values:

```python
from bandedqr import solve_batched

# ab.shape == (n, kl + ku + 1, batch)
# b.shape  == (n, batch) or (n, nrhs, batch)
x = solve_batched(ab, b, kl, ku, backend="auto", refine=2)
```

Public backend selectors are:

| Selector | Implementation | Arithmetic | Current scope |
|---|---|---|---|
| `numpy` | vectorized host reference | input precision | portable reference and moderate batches |
| `metal` | MLX/Metal, one thread per system | fp32/complex64 plus host fp64 refinement | banded, shifted, reuse, and dense-row/Ride |
| `cuda` | CuPy/CUDA, one thread per system | fp32/complex64 plus host fp64 refinement | banded, shifted, multi-RHS, and reuse |
| `auto` | resolved at runtime | backend-dependent | accelerator for sufficiently large batches, otherwise NumPy |

For shifted systems, one shared band can be combined with a per-system diagonal
shift instead of uploading a full band for every system:

```python
from bandedqr.cuda import solve_batched_shifted

x = solve_batched_shifted(ab_shared, shifts, b, kl, ku, refine=2)
```

The accelerator factorization is single precision. Refinement forms residuals
against the original double-precision operator. Convergence depends on the
matrix and conditioning; callers should inspect residuals rather than assume a
fixed accuracy recovery.

## Almost-banded systems

Spectral operators often contain a banded bulk plus a few dense boundary rows
or tau columns. `bandedqr` provides three main strategies:

| Method | Construction | Appropriate when |
|---|---|---|
| `ride` | carries dense rows through the rotations | dense boundary rows; no recurrence assumption |
| `collapse` | detects a short recurrence and sparsifies boundary rows | recurrence-structured boundaries and many re-solves |
| `bordered` | applies a Woodbury correction to an invertible banded bulk | both row and column borders with a suitable bulk |

`auto` selects `ride` when dense rows are detected. Explicit
`kl`, `ku`, `dense_rows`, and `dense_cols` values are accepted throughout,
because structure classification is inherently ambiguous for some matrices.

The collapse path uses `detect_recurrence(rows)` to estimate a short linear
recurrence from boundary values using a Hankel null-space calculation. It is an
optional structural transformation, not a general method for arbitrary dense
rows.

## Numerical checks

The absence of pivoting makes factorization cost predictable, but it does not
make every matrix safe to solve. The relevant checks are exposed directly:

```python
from bandedqr import condest, compensated_residual, factorize, to_band

factor = factorize(A, kl=kl, ku=ku)
estimate = condest(factor)
is_rank_deficient = factor.rank_deficient()
ab = to_band(A, kl, ku)
r = compensated_residual(ab, kl, ku, x, b)
```

`solve(..., refine=k)` performs up to `k` refinement steps. For rank-deficient
or strongly ill-conditioned systems, use a solver whose pivoting or
rank-revealing behavior matches the problem.

## Performance evidence

The repository separates correctness evidence, preliminary measurements, and
publication-quality claims. The current ledger is
[`paper/claims_and_evidence.md`](paper/claims_and_evidence.md).

<!-- RUSTY_SCALING_RESULTS_START -->
The frozen Rusty scaling campaign is prepared but has not yet been submitted.
It will measure:

- fixed-bandwidth CPU size scaling against LAPACK band LU;
- CPU thread scaling and factor reuse;
- CUDA batch scaling with resident and transfer-inclusive timings;
- raw and refined accelerator residuals; and
- complete machine, compiler, CUDA, source, and timing provenance.

No Rusty CUDA performance claim is made before that campaign passes.
<!-- RUSTY_SCALING_RESULTS_END -->

The benchmark configuration and drivers are versioned:

```bash
# portable correctness/conformance
python bench/paper/drivers/backend_conformance.py \
  --backend numpy \
  --config bench/paper/configs/backend_conformance.json \
  --output /tmp/bandedqr-conformance.json

# direct developer benchmark (not an immutable campaign)
make bench
```

Accepted Rusty results are rendered by `bench/paper/render_scaling.py` into a
GitHub-compatible scaling figure, timing tables, and a compact provenance
summary. Raw campaign records remain outside Git on the registered Ceph project
root.

## Methods and complexity

For an `n × n` matrix with lower bandwidth `kl` and upper bandwidth `ku`, the
Givens kernel eliminates at most `kl` entries per column and updates only the
resulting band. Its operation count is linear in `n` at fixed bandwidth, with
storage linear in `n`. Constants and the crossover against band LU or
Householder QR depend on bandwidth, compiler, and hardware; the scaling
campaign measures those crossovers rather than encoding them as universal
claims.

Available front-end methods are:

| Method | Description |
|---|---|
| `auto` | dispatch using detected or supplied structure |
| `givens` | Givens band QR |
| `householder` | Householder band QR |
| `lq` | QR of the transpose, exposed as LQ |
| `ride` | exact carried-border almost-banded QR |
| `collapse` | recurrence-based boundary sparsification followed by band QR |
| `bordered` | Woodbury correction against the banded bulk |
| `lu` | SciPy/LAPACK band LU comparison path |

## Reproducibility

The test suite checks CPU real/complex solves, almost-banded paths, factor
reuse, compensated residuals, condition estimates, deterministic threaded
factors, portable batching, and accelerator behavior when the corresponding
device is available.

The Rusty campaign additionally:

1. refuses a dirty checkout;
2. archives an exact Git commit into an immutable bundle;
3. records configuration and source hashes;
4. runs conformance before performance measurements;
5. records distributions rather than best-of-N values;
6. checkpoints atomically after each benchmark case;
7. monitors CPU and storage quota headroom; and
8. accounts for the entire parent Slurm allocation, including failures.

See [`bench/paper/README.md`](bench/paper/README.md) and
[`cluster/README.md`](cluster/README.md) for the evidence layout.

## Scope and limitations

- QR is deliberately unpivoted. Rank-deficient or unsuitable systems require a
  more general solver.
- Accelerator arithmetic is currently fp32/complex64; successful refinement is
  problem-dependent.
- CUDA dense-row/Ride kernels are not implemented yet.
- Structure detection is heuristic. Supply structure explicitly when an
  incorrect classification would matter.
- `collapse` requires recurrence-structured rows.
- `bordered` requires a nonsingular banded bulk.
- Windows builds use the serial C kernel.

## Repository layout

```text
src/bandedqr/       Python package, C kernel, Metal backend, CUDA backend
tests/              CPU and optional accelerator tests
bench/              developer and reproducible campaign benchmarks
cluster/            immutable Rusty campaign controls
paper/              claims/evidence ledger and manuscript material
figures/            retained and generated figures
archive/            superseded Dedalus integration and provenance notes
```

## License and citation

No open-source license has been selected yet. Until a `LICENSE` file is added,
the code is distributed with all rights reserved. This must be resolved before
describing the repository as an open-source release.

There is not yet an archival release or DOI. If this repository is used in a
proposal, cite the exact Git commit and the retained benchmark provenance rather
than treating the moving branch name as a fixed software release.
