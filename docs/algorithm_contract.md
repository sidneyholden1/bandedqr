# Algorithm and implementation contract

This document fixes the mathematical conventions that all `bandqr` backends
must follow.  It is an implementation contract, not a claim that every item is
new.  Novelty, attribution, and publication claims are tracked separately in
[`paper/claims_and_evidence.md`](../paper/claims_and_evidence.md).

## 1. Problem class and storage

For the CPU factorisation, let `A` be an `m`-by-`n` real or complex matrix with
`m >= n`, lower bandwidth `kl`, and upper bandwidth `ku`.  The batched
implementations currently solve square systems.  The row-aligned input storage
is

```text
ab[i, j - i + kl] = A[i, j]
```

for `max(0, i-kl) <= j <= min(n-1, i+ku)`.  Thus `ab` has width
`kl + ku + 1`, with the main diagonal in column `kl`.  This is not LAPACK's
column-aligned general-band layout.

Eliminating the lower band can create an upper bandwidth

```text
kuR = ku + kl.
```

The Givens implementations therefore use a row work buffer of width

```text
W = kl + 1 + kuR + 1 = 2*kl + ku + 2.
```

The final entry is a required guard slot.  An adjacent-row rotation can write
one column beyond a row's nominal `kuR` extent; removing the guard is a memory
and correctness bug.

## 2. Unitary Givens convention

Every C, NumPy, and Metal instantiation uses the same rotation.  Given scalars
`a` and `b`, compute real `c >= 0` and scalar `s` so that

```text
G = [[c, s], [-conj(s), c]]
G @ [a, b] = [r, 0].
```

The branch-by-branch formula is:

```text
bb2 = abs(b)**2
if bb2 == 0:
    c = 1
    s = 0
else:
    aa2 = abs(a)**2
    t = sqrt(aa2 + bb2)
    if aa2 == 0:
        c = 0
        s = conj(b) / sqrt(bb2)
    else:
        aa = sqrt(aa2)
        c = aa / t
        s = (a / aa) * conj(b) / t
```

For real input, conjugation is the identity.  The diagonal signs of `R` need
not match libraries that use `c = a/t`; this does not change the represented
factorisation because `Q` absorbs the sign or phase.

The arithmetic expression and branch order are part of the cross-backend
contract.  Replacing them with a mathematically equivalent library routine can
change rounding, phases, and reproducibility.

## 3. Elimination and replay order

Columns `jc = 0, ..., n-1` are processed from left to right.  Within column
`jc`, the last possible nonzero row is

```text
ilast = min(jc + kl, m - 1),
```

and adjacent rows are eliminated bottom-up:

```text
for q = ilast, ..., jc + 1:
    p = q - 1
    rotate rows (p, q) to zero A[q, jc].
```

Each rotation updates columns `jc` through
`min(q + kuR, n - 1)`.  Applying `Q^H` to a right-hand side replays rotations
in this stored factorisation order.  Applying `Q` uses the inverse rotations in
reverse order.  The C and Metal solvers recover the same implicit `(jc, q)`
identity; a change to one traversal must be made everywhere.

The threaded CPU algorithm may schedule independent column work as a
wavefront, but its arithmetic dependency order within every output element is
fixed.  For the supported compiler configurations, threaded and serial `R`
must be bit-identical for both real and complex inputs.  The shared column
kernel is deliberately kept as a non-inlined optimisation boundary because
GCC otherwise contracts complex arithmetic differently between the two call
paths.

## 4. Representations

### Givens band QR

The default CPU representation stores the widened band of `R` and an implicit
sequence of adjacent-row rotations.  It supports square solves, rectangular
least-squares solves, application of `Q` and `Q^H`, and factor-once/solve-many.

### Householder band QR

The CPU Householder path stores one short reflector per column.  It represents
the same `A = QR` problem but is not required to reproduce the Givens `R`
bit-for-bit.  Its numerical results must satisfy the same residual and
orthogonality tests.

### Almost-banded matrices

The `ride`, `collapse`, and `tau` paths are distinct structured algorithms, not
backend aliases:

- `ride` evolves the out-of-band low-rank part alongside the band elimination;
- `collapse` first transforms dense border information into a pure banded
  problem;
- `tau` represents the transpose-oriented construction used by the associated
  structured solve.

Their choice is a performance and accuracy decision.  The current automatic
dispatch selects `ride` for dense borders, but comparative claims require the
campaign evidence listed in the claims ledger.

## 5. Batched backend contract

Batched inputs have shape `(n, kl+ku+1, batch)` and right-hand sides have shape
`(n, batch)` or `(n, nrhs, batch)`.  A shifted form accepts one shared band and
one diagonal shift per system.  All backends must preserve the storage,
rotation, and traversal conventions above.

Public selectors and concrete run-record identities are:

| Selector | Run-record ID | Status | Arithmetic |
| --- | --- | --- | --- |
| `numpy` | `cpu-numpy-batched` | implemented reference | host float64/complex128 |
| `metal` | `metal-mlx` | implemented | Metal float32/complex64 plus host refinement |
| `cuda` | `cuda-cupy` | planned | not yet implemented |

`gpu` is retained only as a compatibility alias for the `metal` selector;
results and run records must use the concrete run-record ID.  `auto` is a
dispatch policy, not a backend identity.

The accelerator factorisation and triangular solves use single precision.  A
raw solve and a refined solve are separate experimental results.  Refinement
forms

```text
r_k = b - A*x_k
```

against the original float64/complex128 band on the host, solves for a
single-precision correction with the existing factorisation, and updates the
high-precision iterate.  Refining against a rounded single-precision copy of
`A` is invalid for the accuracy claim.

Mixed-precision refinement is expected to work when the factorisation is
nonsingular and the condition number is sufficiently below the reciprocal of
single-precision unit roundoff.  It is not an unconditional guarantee.  Raw
error, refined error, residual history, dtype, condition estimate where
available, backend/device identity, and any stagnation must be recorded.

## 6. Complexity model

For a square `n x n` band with fixed `kl` and `ku`, there are at most
`n*kl` adjacent-row rotations and each touches `O(kl+ku)` entries.  The
factorisation therefore costs

```text
O(n * kl * (kl + ku))
```

operations, with lower-order edge effects.  Applying the rotations costs
`O(n*kl*nrhs)` and banded back-substitution costs
`O(n*(kl+ku)*nrhs)`.

The widened band consumes `O(n*(2*kl+ku))` scalars.  Explicitly stored Givens
parameters consume `O(n*kl)` additional scalars/indices.  Batched backends
multiply those leading storage terms by `batch`.  Exact coefficients and
traffic differ by representation and belong in benchmark records rather than
asymptotic claims.

## 7. Numerical scope and failure modes

The implementation does not pivot.  A tiny or zero diagonal in `R`, severe
ill-conditioning, or rank deficiency can make a solve unreliable even when
the factorisation completes.  Callers can inspect the CPU rank diagnostic;
campaigns must report conditioning or a defensible proxy for adversarial
cases.

The project does not currently claim unconditional backward stability,
rank-revealing behaviour, or a universal performance advantage over LAPACK,
sparse direct solvers, or dense QR.  Such statements require hypotheses,
proofs or citations, and reproducible comparative evidence.

## 8. Conformance requirements

A backend qualifies as implemented only after the same deterministic inputs
pass the relevant conformance campaign.  The record must contain:

- the source commit and a hash of every generated matrix family;
- backend, runtime, compiler where relevant, device, host, and dtype;
- dimensions, bandwidths, batch size, and number of right-hand sides;
- normalized residual and forward difference from the NumPy reference;
- raw single-precision and refined results for accelerators;
- refinement history, failure state, and explicit tolerances;
- factor-reuse, shifted-band, real, complex, and multi-RHS cases;
- an exact invocation, immutable output directory, and PASS/FAIL status.

Performance measurements are a separate campaign.  Correctness conformance
must not silently select the best of repeated timings, and a correctness PASS
must never be inferred from benchmark throughput.
