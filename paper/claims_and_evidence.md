# Claims and evidence ledger

This ledger separates what the repository currently demonstrates from what a
paper may eventually argue.  It should be reviewed with Geoff Vasil and
Sheehan Olver before prose in `intro.tex` is treated as a manuscript claim.

Status meanings:

- **supported**: a reproducible record at the named commit directly tests it;
- **implemented, evidence pending**: code and unit tests exist, but the
  publication-quality campaign has not run;
- **hypothesis**: plausible and worth testing, but not yet evidence-backed;
- **human decision**: cannot be settled by more automated testing alone.

## Evidence available now

The strongest current record is Rusty campaign
`cpu-smoke-20260901T145809Z-0d256c5692` at commit
`0d256c569283074a43c7d44fadd48f9510a4817b` (Slurm job `6971350`).  It ran on
Linux/x86-64 with GCC using eight shared `genx` CPUs and 16 GB of memory.  Its
immutable records now live in `~/projects/bandedqr/runs/`; their original
recorded source paths are preserved. Large output remains in `~/ceph/bandqr/`
on Rusty. This pre-rename evidence was produced by `bandqr`, now named
`bandedqr`; the records and retained figure PDFs have not been rewritten.

That campaign reports 174 passing and 22 skipped tests, exact serial/threaded
agreement for real and complex `R` at 2, 4, and 8 threads, and six CPU smoke
checks.  Its Dedalus 3.0.5 case used the same 130-by-130 operator for `bandqr`
and SuperLU: normalized residuals were approximately `1.74e-16` and
`1.36e-13`, respectively, and the relative solution difference was
`3.75e-16`.  These are correctness observations for one defined case, not an
end-to-end performance comparison.

The current scaling record is campaign `rusty-scaling-20261108` at commit
`efa71b0d853e6baeccd14411f74a60f63d58ca4d`. Slurm jobs `7428056` (one
exclusive 96-CPU Genoa node) and `7428058` (one NVIDIA H100 SXM5) both
completed. The CPU allocation reports 188 passing and 31 skipped tests and 24
timing cases. The H100 allocation reports 197 passing and 23 skipped tests, all
18 supported CUDA conformance checks passing, and five batch-scaling cases.
The public SVG, all 29 timing rows with sample distributions, input hashes, and
machine provenance are retained in `figures/rusty-scaling.svg` and
`bench/results/rusty-scaling-summary.json`; raw records and logs remain under
the registered Ceph campaign root.

Failed campaigns remain part of the audit trail: the first exposed a missing
Slurm temporary-directory assumption, the second exposed GCC-dependent complex
threading nondeterminism, and the third exposed a `PYTHONPATH` interaction with
Dedalus/mpi4py.  Keeping them is evidence that the final record was not selected
by silently overwriting failures. The current scaling ledger likewise retains
failed job `7427976`, which exposed Rusty's absent `SLURM_TMPDIR`, and cancelled
job `7427979`, which was stopped before allocation to prevent the same failure.

## Claim map

| Candidate claim | Status | Present evidence | Next gate |
| --- | --- | --- | --- |
| CPU band QR solves the tested real and complex systems accurately on macOS and Linux/x86-64. | **supported, scoped** | Unit suite on the development Mac; immutable Rusty CPU smoke above. | Run activated CI on every supported Python/compiler target and publish the records. |
| Threaded and serial Givens factorisations produce bit-identical `R` under the supported build settings. | **supported, scoped** | Rusty real/complex regression at 2, 4, and 8 threads after the GCC fix. | Add CI/compiler coverage and wider `n, kl, ku` campaign cases. |
| The same small Dedalus operator can be solved accurately with `bandqr` and SuperLU. | **supported, scoped** | Rusty Dedalus smoke above. | Add representative Dedalus boundary-value and IVP workloads, setup/solve separation, and scaling. |
| A Metal batched backend implements the same recurrence and supports real, complex, shifted, multi-RHS, refinement, reuse, and dense-row cases. | **implemented, evidence pending** | Source and skipped-on-non-Metal unit tests; preliminary development-machine runs. | Run the immutable local Metal conformance campaign and retain its provenance/output. |
| A CUDA/CuPy batched backend supports real, complex, shifted, multi-RHS, refinement, and reuse cases. | **supported, scoped** | H100 job `7428058`: 197 tests passed and all 18 supported conformance checks passed for real/complex ordinary, shifted, multi-RHS, reuse, and conditioned families. Dense-row/Ride was explicitly recorded as unsupported. | Retain a second architecture/compiler record and implement CUDA dense-row/Ride before claiming capability parity with Metal. |
| Mixed precision recovers near-double-precision accuracy on useful problem classes. | **supported, scoped** | Retained H100 conformance passed registered real/complex families and target-condition cases through `1e6`; performance cases produced refined residuals from `3.2e-14` to `4.9e-14`. | Expand the conditioning sweep, record actual condition estimates, and retain stagnation/failure envelopes before general stability language. |
| On the registered H100 workload, refined end-to-end CUDA execution is faster than the portable NumPy batch path. | **supported, scoped** | For `n=256`, `kl=ku=4`, and batches 64--16,384, retained medians show 7.6--11.8× speedups with all timing samples and provenance published. | Vary system size, bandwidth, dtype, conditioning, and CPU baseline; add uncertainty across independent process runs. |
| Batched Metal execution is materially faster than CPU alternatives. | **hypothesis** | README numbers are preliminary, best-of-N measurements from one M3 Max. | Pre-register timing rules; compare transfer-inclusive and device-resident time with uncertainty across size/band/batch regimes. |
| There is no serious existing banded-QR implementation serving these structured and accelerator use cases. | **hypothesis / literature claim** | Informal domain assessment only. | Systematic literature/software survey led by the authors; define “banded,” “QR,” “maintained,” and “accelerator” precisely. |
| The recurrence or scheduling construction is novel. | **hypothesis / mathematical claim** | Implementation exists; novelty has not been established by repository evidence. | Coauthor derivation, prior-art search, precise theorem/algorithm statement, and appropriate citations or proof. |
| The method is unconditionally backward stable. | **not supported** | No proof or campaign can establish this unqualified statement; no pivoting and mixed precision impose hypotheses. | Replace with a qualified theorem/citation or scoped empirical stability language. |
| The implementation is suitable as a Dedalus backend. | **hypothesis with integration signal** | One same-operator smoke case and archived earlier integration work. | Agree an API with Dedalus maintainers; run realistic matrices and time assembly, factorisation, repeated solves, and memory. |
| Accelerator support spans Apple Metal and NVIDIA CUDA. | **implemented; CUDA verified, Metal evidence pending** | Metal/MLX and CUDA/CuPy code paths exist; retained H100 conformance passed every supported CUDA family, while CUDA dense-row/Ride remains unsupported. | Pass retained Metal conformance and state the capability difference explicitly. |

## Provisional manuscript language

The following ideas in `paper/intro.tex` should remain explicitly provisional:

- novelty of the automatic recurrence;
- “matching or overtaking” mature sparse solvers in Dedalus workloads;
- unconditional backward-stability wording;
- language that implies the Metal backend establishes general GPU support.

A defensible current description is narrower: the repository contains a
deterministic structured CPU implementation, Metal and CUDA batched
implementations, retained H100 correctness evidence, and scoped CPU/H100
scaling measurements. The broad stability envelope, novelty, and application
impact remain the paper's research programme.

## Minimum experiment set before drafting results

1. Pass the local Metal conformance campaign on a real device, including raw
   and refined accuracy, condition sweeps, shifted systems, factor reuse,
   multi-RHS, real/complex data, and dense-row/Ride cases.
2. Activate CI for GCC, Clang, macOS arm64, serial portability, standalone-file
   freshness, and sanitizers.
3. Extend the retained fixed-bandwidth float64 CPU baseline across `kl`, `ku`,
   complex data, and additional right-hand-side regimes.
4. Extend the accepted H100 record across system size, bandwidth, dtype,
   conditioning, and at least one additional NVIDIA architecture.
5. Select a small set of representative Dedalus problems with the coauthors
   and measure end-to-end effects as well as kernel timings.
6. Compare against clearly configured dense QR, LAPACK banded alternatives
   where applicable, sparse QR/LU, and the existing Dedalus solver path.

## Decisions for the authors

The authors must decide the central mathematical claim, intended venue,
baseline set, supported API, licensing, public repository/account, and wording
for use of AI-assisted code.  The fact that Claude wrote substantial code does
not prevent publication, but the human authors must understand, validate, and
take responsibility for every algorithm, result, and claim.  Journal and
institutional disclosure policies should be checked at submission time.

The exact history is retained in the private repository
`sidneyholden1/bandedqr`. Until the authors select a license and explicitly
change its visibility, the project should avoid claiming an open-source or
public release; the code remains all rights reserved.
