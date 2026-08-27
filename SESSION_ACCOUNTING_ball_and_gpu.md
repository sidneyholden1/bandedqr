# Session accounting — ball tau-free, spherical GPU reality, 2D RBC head-to-head

Throughline of the day: **honest, data-backed results** — verify to machine precision or
measure on real matrices; where something doesn't work, say so and characterize *why*.

---

## 1. Ball-basis tau-free boundary bordering — SOLVED (scalar + vector), machine precision

**Goal.** Extend tau-free (boundary bordering) — which was working in Cartesian/Chebyshev —
to Dedalus's spherical **ball** basis.

**Why it's hard.** In Chebyshev, `Lift(tau, -1)` is essentially a unit column, so dropping
the top output mode reproduces the tau solve bit-for-bit. The ball breaks that in two ways,
both of which we cracked:

1. **ℓ-dependent radial resolution.** `ball.n_size(ℓ) = Nr − ⌊ℓ/2⌋` real radial modes; the
   unused slots are structural **padding at the LOW radial indices** (confirmed by filling a
   field with random data and reading the coefficient support). For vector/tensor fields it
   further depends on the component's spin weight. This padding must be dropped from both
   equation **rows** and field **columns** (via mutating the cached `valid_modes` in place),
   or the pencil is singular. This is why a naive "drop `[...,-1]`" gave 4% error for ℓ>0.

2. **The operator output basis self-borders.** A differential operator's output can carry
   fewer top modes than the field — the ball **vector** Laplacian drops one top mode per
   component; the **scalar** Laplacian keeps all. Measured by evaluating `eqn['LHS']` on
   random data. The correct border count is `output_count − field_count + order` (1 for the
   scalar, **0** for the vector). Everything is detected by probing, so no spin/regularity
   bookkeeping is hard-coded.

**Key correctness insight.** In the ball, bordering and tau are **different (both valid)
discretizations** — the tau column is rank-spread, not a unit column — so they need not match
mode-for-mode; both converge at the spectral rate. Validate against a **manufactured
solution**, NOT by diffing tau. (A "3.9e-2 vs tau" on a random forcing is the discretization
gap, not a bug — confirmed because tau-free reproduced standard tau's error *exactly* there.)

**Verified** (`demo_dedalus_ball_taufree.py`, tau vs tau-free, manufactured solutions):

| Nr | scalar Poisson (multi-ℓ) | vector Poisson (smooth ê_z) |
|----|--------------------------|------------------------------|
| 8  | 7.9e-16                  | 3.3e-16                      |
| 16 | 4.6e-16                  | 3.4e-16                      |
| 32 | 8.6e-16                  | 1.1e-15                      |

**Result:** closes ball **Gap 1 (scalar)** and **Gap 3 (vector)**. New code: `taufree_ball`,
`_ball_radial_mask`, `_ball_output_mask` in `dedalus_bandqr.py`.

**Still open — the one honest gap.** Velocity–pressure *in the ball* is an **inf-sup
pressure-space problem**: the pressure must live in the `div(u)` range, and the per-ℓ deficit
(`field_count − div_out_count`) is irregular (0 at ℓ=0 where the gauge acts, 1 at ℓ=1, 0 at
ℓ=2…). Probe-and-drop for the compatible pressure space was unreliable; it needs a principled
construction. **So the full internally-heated ball convection IVP is still blocked on this
one piece.** (Gap 2 — velocity–pressure tau-free — is solved in 2D Cartesian, just not yet
on the sphere.)

---

## 2. GPU batching for spherical sims — HONEST NEGATIVE RESULT

The GPU win (`demo_gpu_pencils.py`, up to 117×) needs **thousands of NARROW-banded pencils**.
Measured the real per-timestep pencil structure (via connected-components on the actual
`L_min`) for several geometries:

| geometry | example | # pencils | pencil size | fit |
|----------|---------|-----------|-------------|-----|
| 3D ball  | `ivp_ball_internally_heated` | 61   | 147–244 banded | ❌ batch too small |
| 3D shell | `ivp_shell_convection`       | 61   | 101–168 banded | ❌ batch too small |
| 2D sphere surface | `ivp_sphere_shallow_water` | 4031 | **1–3**   | ❌ pencils trivial |
| 3D Cartesian box  | *(no gallery example)*     | ~4095 | 50–250 banded | ✅ the sweet spot |

**Why.** The sphere bundles everything by azimuthal wavenumber `m` (≈`Nφ` of them), and the
colatitude becomes spherical-harmonic ℓ, which either **couples** (3D ball/shell → few fat
pencils) or, with no radial direction, leaves **trivial** per-mode systems (2D sphere → many
size-3 pencils). You never get *many AND banded* on the sphere. The GPU-ideal regime
(thousands of narrow pencils) requires **two Fourier directions + one Chebyshev** = a 3D
Cartesian box — which is *not* in the Dedalus example gallery.

---

## 3. Best shipping example for tau-free + QR + GPU → 2D Rayleigh–Bénard

Checked the actual gallery: no 3D Cartesian RBC; `ivp_2d_ensemble_rbc` uses `CardinalBasis`
(not in this Dedalus build). The honest best shipping-example pick is **`ivp_2d_rayleigh_benard`**
(modest batch), with disk/annulus as the 2D-polar equivalent.

---

## 4. 2D RBC head-to-head — REAL POSITIVE (modest, batch-limited)

**Pencil structure gotcha (worth knowing).** Real 2D RBC pencils *look* wide (RCM bandwidth
238/384) — but that's purely a **RealFourier cos/sin split + blocked field ordering**
artifact. With **ComplexFourier + interleaving**, each pencil is a **bandwidth-7 bulk + 6
dense BC rows** (size 192, `Nx` of them) — textbook almost-banded / RideQR structure. (Single-
field ComplexFourier diffusion is perfectly clean: `kl=0, ku=4` + 2 dense rows.)

**Benchmark — per-step linear solve, CPU per-pencil loop vs GPU batched banded:**

| Nx   | # pencils | CPU (SuperLU loop) | GPU (batched banded) | speedup |
|------|-----------|--------------------|----------------------|---------|
| 256  | 256       | 3.82 ms            | 2.47 ms              | 1.5×    |
| 512  | 512       | 7.92 ms            | 1.73 ms              | 4.6×    |
| 1024 | 1024      | 14.96 ms           | 2.42 ms              | 6.2×    |

Modest and batch-limited, exactly as predicted for 2D (matches the standalone `5.8×@1k`); it
grows with `Nx`. The 100× story lives in 3D, which isn't a shipping example.

**Honest caveats:** the GPU figure is the **banded-bulk solve cost** (fp32); the full verified
solve also needs a **rank-6 Woodbury** (the 6 BC rows) + **fp32→fp64 refinement**. Batched
RideQR-with-Woodbury on GPU is **not yet built**. And this is the per-step *solve*, not the
whole timestep (FFT transforms + nonlinear term are unchanged), so end-to-end < solve speedup.

---

## 5. Full-GPU feasibility — assessed, not built

Asked whether a full-sim GPU run is possible. Honest answer: **not as a quick Dedalus
bolt-on.** Dedalus runs transforms and the solve loop on CPU and iterates over subproblems
one at a time (never batches), so the GPU batch fights the architecture; and per-step
CPU↔GPU shuttling at 192-DOF pencils can cost as much as the solve. The only configuration
where the batch advantage survives to wall-clock is a **standalone mlx RBC solver** (transforms
+ timestep + batched solve all resident on GPU) — a genuine build, and fp32 (mlx on Apple
Silicon), so a chaotic t=30 run diverges from the fp64 reference after the Lyapunov time
(physics still valid; compare statistics, not identical frames).

A CPU full-sim timing + t=30 animation script (`demo_rbc_fullsim.py`) is written and ready
(tau-free+bandqr vs standard tau+SuperLU, ≈parity expected) but was not run this session.

---

## Artifacts

- **`dedalus_bandqr.py`** — added `taufree_ball`, `_ball_radial_mask`, `_ball_output_mask`,
  `_radial_slice`.
- **`demo_dedalus_ball_taufree.py`** — scalar + vector ball Poisson tau-free verification (new).
- **`demo_rbc_fullsim.py`** — full-sim timing + t=30 animation (new, not yet run).

## Bottom lines

- ✅ **Ball tau-free (scalar + vector): machine precision** — the hard operator-level work,
  done and verified.
- ⏳ **Ball velocity–pressure: open** — inf-sup pressure space; blocks the full ball convection IVP.
- ❌ **GPU batching does not help spherical sims** — measured, structural, not fixable by effort.
- ✅ **2D RBC: real modest win** — 1.5–6.2× on the solve, tau-free makes the pencils clean.
- ⏳ **Full-GPU sim: needs a standalone mlx solver** — assessed, not built.
