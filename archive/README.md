# archive

Work that came before bandqr was split out as a standalone library. Kept for
provenance; **nothing here is part of the package**, none of it is tested, and
none of it is imported by `src/bandqr`.

## `dedalus/`

bandqr began as a solver for the almost-banded systems that
[Dedalus](https://dedalus-project.org) assembles, and this is that integration:

- `dedalus_bandqr.py` — a Dedalus `matsolver` backed by bandqr, plus the
  tau-free (boundary-bordering) reformulation helpers `taufree`,
  `taufree_ball`, `first_order_reduce`.
- `demo_*.py` — verification and demonstration scripts: Poisson, coupled
  systems, shear flow, Rayleigh–Bénard, ball convection, GPU pencils.
- `render_*.py` — animation renderers for those runs.
- `benchmarks.ipynb`, `make_benchmarks.py` — the cross-tool comparison
  (bandqr vs Dedalus SuperLU vs ApproxFun). This is where the recorded numbers
  in `paper/make_figures.py` came from.

The library itself no longer depends on Dedalus. What was genuinely
general in this work — the almost-banded factorisations, the recurrence
collapse, the batched GPU path — lives in `src/bandqr`.

## `media/`

Simulation outputs from those runs: HDF5 slices, PNG frames, and rendered
`.mp4` animations of ball convection and Rayleigh–Bénard, tau-free against
standard tau. Large, and untracked by git (see `.gitignore`).

## `notes/`

- `SESSION_ACCOUNTING_ball_and_gpu.md` — a working log of the spherical-basis
  tau-free work, including two results worth keeping: ball tau-free verified to
  machine precision for scalar and vector fields, and the measured negative
  result that **GPU batching does not help spherical simulations** (the sphere
  gives either few fat coupled pencils or thousands of trivial size-3 ones,
  never many-and-banded). The GPU-favourable regime is a 3D Cartesian box.
- `MIGRATION_PROVENANCE.md`, `SHA256_MANIFEST_PRE_MOVE.txt` — the record of
  moving this project out of `yolk_sac_vasculature/`, with a verified manifest
  of all 1,022 files.
