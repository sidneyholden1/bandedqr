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
immutable records live in `~/bandqr/runs/` and large output in
`~/ceph/bandqr/` on Rusty.

That campaign reports 174 passing and 22 skipped tests, exact serial/threaded
agreement for real and complex `R` at 2, 4, and 8 threads, and six CPU smoke
checks.  Its Dedalus 3.0.5 case used the same 130-by-130 operator for `bandqr`
and SuperLU: normalized residuals were approximately `1.74e-16` and
`1.36e-13`, respectively, and the relative solution difference was
`3.75e-16`.  These are correctness observations for one defined case, not an
end-to-end performance comparison.

Failed campaigns remain part of the audit trail: the first exposed a missing
Slurm temporary-directory assumption, the second exposed GCC-dependent complex
threading nondeterminism, and the third exposed a `PYTHONPATH` interaction with
Dedalus/mpi4py.  Keeping them is evidence that the final record was not selected
by silently overwriting failures.

## Claim map

| Candidate claim | Status | Present evidence | Next gate |
| --- | --- | --- | --- |
| CPU band QR solves the tested real and complex systems accurately on macOS and Linux/x86-64. | **supported, scoped** | Unit suite on the development Mac; immutable Rusty CPU smoke above. | Run activated CI on every supported Python/compiler target and publish the records. |
| Threaded and serial Givens factorisations produce bit-identical `R` under the supported build settings. | **supported, scoped** | Rusty real/complex regression at 2, 4, and 8 threads after the GCC fix. | Add CI/compiler coverage and wider `n, kl, ku` campaign cases. |
| The same small Dedalus operator can be solved accurately with `bandqr` and SuperLU. | **supported, scoped** | Rusty Dedalus smoke above. | Add representative Dedalus boundary-value and IVP workloads, setup/solve separation, and scaling. |
| A Metal batched backend implements the same recurrence and supports real, complex, shifted, multi-RHS, refinement, reuse, and dense-row cases. | **implemented, evidence pending** | Source and skipped-on-non-Metal unit tests; preliminary development-machine runs. | Run the immutable local Metal conformance campaign and retain its provenance/output. |
| Mixed precision recovers near-double-precision accuracy on useful problem classes. | **hypothesis with preliminary support** | Existing tests and development benchmarks; no condition-number sweep in an immutable campaign. | Measure raw and refined error versus actual conditioning for real/complex families, including stagnation/failure. |
| Batched Metal execution is materially faster than CPU alternatives. | **hypothesis** | README numbers are preliminary, best-of-N measurements from one M3 Max. | Pre-register timing rules; compare transfer-inclusive and device-resident time with uncertainty across size/band/batch regimes. |
| There is no serious existing banded-QR implementation serving these structured and accelerator use cases. | **hypothesis / literature claim** | Informal domain assessment only. | Systematic literature/software survey led by the authors; define “banded,” “QR,” “maintained,” and “accelerator” precisely. |
| The recurrence or scheduling construction is novel. | **hypothesis / mathematical claim** | Implementation exists; novelty has not been established by repository evidence. | Coauthor derivation, prior-art search, precise theorem/algorithm statement, and appropriate citations or proof. |
| The method is unconditionally backward stable. | **not supported** | No proof or campaign can establish this unqualified statement; no pivoting and mixed precision impose hypotheses. | Replace with a qualified theorem/citation or scoped empirical stability language. |
| The implementation is suitable as a Dedalus backend. | **hypothesis with integration signal** | One same-operator smoke case and archived earlier integration work. | Agree an API with Dedalus maintainers; run realistic matrices and time assembly, factorisation, repeated solves, and memory. |
| “GPU support” is general across major accelerator platforms. | **not yet true** | Metal/MLX is implemented; CUDA is not. | Implement the same backend contract on CUDA and pass Rusty GPU conformance before using platform-general wording. |

## Provisional manuscript language

The following ideas in `paper/intro.tex` should remain explicitly provisional:

- novelty of the automatic recurrence;
- “matching or overtaking” mature sparse solvers in Dedalus workloads;
- unconditional backward-stability wording;
- language that implies the Metal backend establishes general GPU support.

A defensible current description is narrower: the repository contains a
deterministic structured CPU implementation, a Metal batched implementation,
and initial cross-platform correctness evidence; the comparative performance,
stability envelope, novelty, and application impact are the paper's research
programme.

## Minimum experiment set before drafting results

1. Pass the local Metal conformance campaign on a real device, including raw
   and refined accuracy, condition sweeps, shifted systems, factor reuse,
   multi-RHS, real/complex data, and dense-row/Ride cases.
2. Activate CI for GCC, Clang, macOS arm64, serial portability, standalone-file
   freshness, and sanitizers.
3. Run a CPU scaling campaign over `n`, `kl`, `ku`, threads, dtype, and right-
   hand-side count; retain medians/distributions, not best-of-N alone.
4. Implement CUDA under the same contract, then run correctness before any
   Rusty GPU performance campaign.
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

Until those decisions are made, the project should avoid claiming a public
release: there is no `LICENSE` file, the package URLs are provisional, and no
public GitHub repository has been selected.
