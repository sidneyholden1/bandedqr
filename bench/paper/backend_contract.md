# Batched backend contract

GPU acceleration is a first-class part of the project.  A run record must name
the concrete backend rather than using the ambiguous label `gpu`:

| Backend ID | Status | Execution model |
| --- | --- | --- |
| `cpu-c` | implemented | one C factorisation per system |
| `cpu-numpy-batched` | implemented | public selector `numpy`; portable vectorisation over systems |
| `metal-mlx` | implemented | public selector `metal`; one Metal thread per independent system |
| `cuda-cupy` | planned | reserved selector `cuda`; not yet callable |

The historical public selector `gpu` remains an alias for `metal`, but it is
not a valid backend identity in a campaign record. Likewise, `auto` is a
dispatch policy and must be resolved before provenance is written.

All batched accelerator backends must accept the same logical inputs:

- row-major band storage `(n, kl + ku + 1, batch)`;
- right-hand sides `(n, batch)` or `(n, nrhs, batch)`;
- real and complex inputs;
- factor-once/solve-many operation;
- the shared-band plus per-system diagonal-shift form.

The factorisation must use the repository's Givens formula and rotation order.
Conformance is measured against `cpu-numpy-batched` on identical matrices,
identified by a content hash.  An unrefined single-precision result must be
reported separately from mixed-precision refinement against the original
float64/complex128 operator.

Correctness records must include normalized residual, forward difference from
the portable reference, refinement history, dtype, dimensions, and backend
identity.  Performance records must additionally distinguish device-resident
time from transfer-inclusive time and record warm-up/compilation separately.
Device model, driver/runtime, backend version, batch size, and source commit
are mandatory provenance.

The present Metal implementation satisfies the algorithmic portion of this
contract.  `cuda-cupy` remains a planned backend; campaign records must not
represent it as implemented until the same conformance cases pass on Rusty.
