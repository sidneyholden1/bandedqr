# Reproducible benchmark campaign

This directory contains the versioned inputs, correctness gates, timing driver,
and result renderer used for the project evidence. Generated raw records do not
belong here: Rusty attempts live beneath the registered Ceph `campaign_root` in
`cluster/resource_policy.json`.

## Correctness before timing

`configs/backend_conformance.json` defines the backend-neutral accelerator
gate. It covers ordinary, shifted, multi-RHS, factor-reuse, conditioned,
real/complex, and dense-row/Ride cases. The NumPy implementation supplies the
reference on byte-identical inputs. Accelerator records separate raw arithmetic
from mixed-precision refinement.

CUDA currently reports dense-row/Ride as an unsupported capability rather than
silently skipping an expected result. The Rusty GPU launcher must pass this
conformance gate before starting performance measurements.

## Scaling campaign

`configs/rusty_scaling.json` preregisters the benchmark matrix:

- fixed-bandwidth CPU size scaling against LAPACK band LU;
- CPU thread scaling at a wider bandwidth;
- factor-once/solve-many CPU timings; and
- CUDA batch scaling with resident, host-transfer, raw end-to-end, refined
  end-to-end, and CPU-reference measurements.

`drivers/scaling.py` records all timing samples and quartiles rather than a
single best run. Inputs use deterministic per-case seeds. Every completed case
is checkpointed atomically, and a resumed attempt refuses a changed config.
Machine, package, scheduler, source, and device identity are retained with the
record.

The CUDA columns answer different performance questions:

| Measurement | What it includes | What it diagnoses |
|---|---|---|
| resident re-solve | device factors and RHS already resident | kernel and launch efficiency |
| host-transfer re-solve | retained factors with host I/O | transfer overhead for reuse |
| raw end-to-end | upload, factor, and single-precision solve | factorization throughput |
| refined end-to-end | upload, factor, solve, host residuals, corrections | current user-visible path |
| CPU NumPy | portable batch factor and solve | same-input reference crossover |

This separation is intended to identify the next optimization target. A large
resident/refined gap points to transfers or host refinement; poor resident
scaling points to launch geometry, per-system parallelism, or kernel fusion.

## Accepted artifacts

`render_scaling.py` accepts only CPU and CUDA `PASS` records with matching
config hashes and source commits. It produces:

- a GitHub-compatible SVG scaling figure;
- a Markdown timing table; and
- a JSON summary containing timings and provenance.

With `--readme README.md`, it atomically replaces only the marked results block
in the top-level README. The exact command is documented in
`cluster/README.md`.
