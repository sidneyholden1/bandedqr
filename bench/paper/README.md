# Paper campaign

This directory contains versioned inputs and drivers for reproducible paper
experiments.  Generated campaign records never belong here: they live in
`runs/<campaign-id>/`, while detailed outputs live in
`~/ceph/bandqr/<campaign-id>/` on Rusty.

The first campaign is deliberately a correctness gate, not a performance run.
It verifies the Linux C build, the complete portable test suite, deterministic
threaded factorisation, real and complex solves, the portable batched backend,
an almost-banded solve, and a small matrix assembled by Dedalus.  Metal and
CUDA are represented in the backend contract but are not scheduled by this
gate.

Files in `configs/` are immutable experiment definitions.  Drivers write raw
machine-readable results to the campaign's Ceph directory.  The submission
tool records the source commit and hashes the config and submitted batch script
before Slurm accepts the job.
