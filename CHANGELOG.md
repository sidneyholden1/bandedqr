# Changelog

## 0.2.0 — standalone library

bandqr becomes a self-contained banded/almost-banded QR library. The Dedalus
integration it grew out of moved to `archive/`; see `archive/README.md`.

### Packaging
- Proper package under `src/bandqr/`, installable with `pip install -e .`.
- Single-file build regenerated from the package
  (`python tools/make_standalone.py`) — CPU paths and the vectorised batched
  solver, self-compiling on first import.
- 194-test pytest suite; CPU and GPU benchmark scripts.

### GPU backend (rewritten)
- **Factor once, solve many.** `BatchedBandQR` keeps the rotations instead of
  re-factorising per right-hand side — 2.9× per subsequent solve.
- **fp32 speed, float64 accuracy.** Mixed-precision iterative refinement against
  float64 residuals: 2e-7 raw → 7.5e-14 after one correction step.
- **One compile.** Kernels cached on `(kl, ku, dtype)` alone; `n`, batch size
  and RHS count became runtime arguments, so changing batch size no longer
  triggers a Metal recompile.
- **complex64**, from the same templated source as the real kernel.
- **`BatchedRideQR`** — batched almost-banded solve, dense boundary rows carried
  through the rotations with `L` shared across the batch.
- **Shifted form** — `solve_batched_shifted` for one shared band plus per-system
  diagonal shifts; 58.7 MB → 14.3 kB upload at n=256, batch=4096.
- Multi-RHS support.
- Fixed: the complex store macro evaluated its value argument twice, so
  back-substitution re-read a half-written destination and corrupted results.

### CPU
- **Structure detection rewritten.** Classifies by how populated each diagonal
  is, rather than by per-row nonzero counts — the old heuristic could not detect
  dense *columns* at all and inflated `ku` to the full matrix instead.
- **`BorderedQR`** handles complex input (was silently float64-only) and is now
  a reusable factor-once object.
- **`RideQR`** permutes bottom-placed dense rows automatically, removing the
  "boundary rows must sit near the top" constraint.
- **Accuracy tooling**: Hager–Higham condition estimation (`condest`),
  compensated double-double residuals, and `iterative_refinement`.
- **Q and Rᴴ access**: `apply_Q`, `solve_Rt`, `Q_dense`, `R_sparse`,
  `rank_deficient`.
- Blocked multi-RHS almost-banded solve in C (`solve_many_ab`).
- Band packing vectorised — the element-by-element Python loops are gone.
- `auto` now dispatches dense borders to `ride` rather than `collapse`:
  measured, ride factors 3–4× faster *and* holds machine precision where
  collapse degrades to 3e-11 by n=16384.
- `method='bordered'` on a purely banded matrix falls back to a plain band QR,
  as `collapse` and `ride` already did.

### Threading
- Wavefront progress markers are cache-line padded and published in chunks;
  wide-band threading went from a 0.63× *slowdown* to 2.78× at kl=128.
- `nthreads=0` (auto) is now calibrated to measurement: serial below kl=64,
  because narrow-band factorisation is memory-bound and threads cannot help.
  Threaded `R` remains bit-identical to serial.

### Portability
- `BANDQR_NO_THREADS` build for platforms without pthreads/C11 atomics
  (selected automatically on Windows).
- Warning-free under `-Wall -Wextra` and pedantic C11, threaded or not.

### Documentation
- README rewritten with measured numbers throughout, and honest limits:
  no pivoting, GPU is fp32-plus-refinement, collapse needs recurrence-structured
  borders, bordered needs an invertible bulk, structure detection is a heuristic.

## 0.1.0

Initial banded and almost-banded QR: templated C kernel (float64 and
complex128), Givens and Householder band QR, the almost-banded "ride"
factorisation, automatic recurrence detection and border collapse, Woodbury
bordering, and a first batched Metal kernel.
