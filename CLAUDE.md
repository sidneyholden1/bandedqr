# Working on bandqr

Operational context for anyone — human or agent — editing this repository.
`README.md` explains what the library *is*; this explains how to work on it and
what not to break.

## Environment

Use `~/miniforge3/envs/dedalus3/bin/python`. It is the only interpreter on this
machine that is native arm64 **and** has numpy, scipy, mlx, matplotlib and
pytest together.

- System `/usr/bin/python3` (3.9.6) has **none** of them. If you see "no module
  named numpy", you are on the wrong interpreter — do not start pip-installing.
- `mlx` (the GPU backend's dependency) exists only in `dedalus3`. Two other
  envs are x86_64, so Metal is unavailable there regardless.
- The env name is a leftover: the library no longer depends on Dedalus.

```sh
PY=~/miniforge3/envs/dedalus3/bin/python
make && PYTHONPATH=src:tests $PY -m pytest tests -q     # 176 pass, 23 Metal-skip, <1s headless
```

Metal requires a logged-in macOS session. A headless Codex or CI process can
have MLX installed while Metal command-queue creation is unavailable; backend
discovery performs an isolated evaluated-operation probe before advertising it.

## Commands

```sh
make                 # build the C kernel into src/bandqr/
make test            # pytest
make bench           # CPU benchmarks
make bench-gpu       # GPU benchmarks
make nothreads       # serial build (portability check), then re-run tests
make standalone      # regenerate bandqr_standalone.py -- CI fails if stale
pip install -e ".[test]"
~/miniforge3/envs/dedalus3/bin/python cluster/local/run_backend_conformance.py
```

## Do not re-litigate these

Each was decided by measurement, and the naive "fix" makes things worse. The
rationale is in the code at the point you would otherwise re-derive it wrong.
If you want to change one, re-run the benchmark rather than reasoning about it.

- **Threading stays serial below `kl=64`** (`banded.py: auto_threads`). Narrow-band
  factorisation is DRAM-bandwidth bound; threads do not help and the wavefront
  sync is pure overhead. Measured 8-thread speedups: kl=16 1.00x, kl=32 1.00x,
  kl=64 1.67x, kl=96 2.28x, kl=128 2.78x. Turning threading on unconditionally
  makes the common case *slower*.
- **`auto` dispatches dense borders to `ride`, not `collapse`** (`api.py: _resolve`).
  Ride factors 3-4x faster *and* holds machine precision; collapse degrades to
  3e-11 by n=16384. The pre-0.2 README claimed the opposite.
- **The GPU residual is formed in float64 on the host, in C** (`gpu.py`). Profiling:
  the device round trip is 0.3ms; *forming* the residual was 5.4ms. It must be
  the float64 band — refining against the fp32 copy converges to the fp32
  matrix's answer and caps accuracy at 1e-7, which is what this exists to beat.
  A compensated-fp32 Metal kernel would solve nothing.

## Invariants

- **Threaded `R` is bit-identical to serial `R`.** Determinism is a headline
  property; the test asserts exact equality, not closeness.
- **The rotation loop nest is a contract.** `factor` and `solve` (C and Metal
  both) walk `(jc, q)` in the same order so rotation identity is recovered
  without storing row indices. Change one, change both.
- **C and Metal use the same Givens formula** deliberately, so GPU `R` can be
  compared against CPU `R`. That comparison is what isolated the complex bug.
- **One templated source per backend.** `bandqr_impl.h` is included twice
  (`double`, `double _Complex`); the Metal source is generated twice
  (`float`, `float2`). Fix bugs in the template, not one instantiation.
- **`bandqr_standalone.py` must be regenerated** after any package edit; CI
  checks it. It flattens the package, so the dependency order in
  `tools/make_standalone.py: MODULES` is load-bearing.

## Gotchas

- **Never `str.format` a Metal kernel body** — C braces collide with format
  fields. Use the `@PLACEHOLDER@` replacement pattern already there.
- **Metal `SSTORE` must bind value and index to locals first.** Expanding the
  value argument twice made back-substitution re-read a half-written
  destination; complex results were ~50% wrong while `R` was correct.
- **Band-storage corner slots are "don't care"** — they index columns outside
  `[0, n)`. The kernel ignores them and a test asserts garbage there cannot
  change an answer.
- **The guard slot is load-bearing.** `W = 2*kl + ku + 2`: a rotation writes one
  column past the nominal `kuR` extent, and that slot absorbs it.
- `pyproject.toml` sets `filterwarnings = ["error::RuntimeWarning"]` — numerical
  warnings fail the suite rather than scrolling past.
- GPU tests skip cleanly with no Metal device; the NumPy batched backend is the
  reference the Metal kernel is checked against.

## State

- **No licence.** Deferred deliberately. Until a LICENSE file exists the default
  is all-rights-reserved, so nobody else can legally use this. The `license`
  line in `pyproject.toml` is commented out with a note. This blocks publication.
- **Project URLs in `pyproject.toml` are a guess** (`github.com/sidneyholden/bandqr`).
  The working copies synchronize through a private bare repository on Rusty;
  no public GitHub account/repository has been selected.
- **CI is written but has never run on GitHub.** It now covers GCC, Clang,
  macOS arm64 CPU, a no-pthreads build, strict C warnings, ASan/UBSan, and the
  standalone artifact, but activation awaits the public/private GitHub choice.
- **Linux/x86-64 CPU correctness has been validated on Rusty.** The immutable
  campaign `cpu-smoke-20260901T145809Z-0d256c5692` passed the portable suite,
  deterministic real/complex threaded checks, six CPU cases, and a small
  Dedalus 3.0.5 same-operator comparison. It is a correctness gate, not a
  performance claim.
- **Metal code exists but its publication-quality gate is still pending.** Run
  `cluster/local/run_backend_conformance.py` from a logged-in Mac. README
  benchmark numbers remain preliminary best-of-N measurements from one M3 Max.
- `archive/` holds the superseded Dedalus integration. Nothing in `src/bandqr`
  imports it; see `archive/README.md`.
