# Rusty campaign control

The cluster files implement a small immutable control plane around Slurm.
Versioned templates live here, generated records live in `runs/` (ignored by
Git), and detailed outputs live under `~/ceph/bandqr/`.

On Rusty, create the lightweight core environment once:

```bash
./cluster/rusty/bootstrap_cpu_env.sh
```

The environment uses Rusty's versioned Python/MPI module stack through
`--system-site-packages`; it downloads nothing and records the exact module and
package state under `~/envs/bandqr/share/cpu/`.

Submit the initial correctness-only gate with:

```bash
./cluster/rusty/submit_cpu_smoke.py
```

The submitter refuses a dirty source checkout and refuses to reuse either the
home or Ceph campaign directory.  It materializes the exact Slurm script,
submits with `sbatch --parsable`, and writes the returned job ID immediately.
The job atomically advances `status.json` from `SUBMITTED` to `RUNNING`, then to
`PASS` or `FAIL`.

This gate allocates no accelerator and reports no performance measurements.
Metal and CUDA performance campaigns should use the same manifest/status
schema after their conformance gates exist.

## Local Metal gate

The first-class Metal correctness gate runs on a logged-in Apple host rather
than through Slurm. It uses the same immutable manifest/status/provenance
pattern and compares raw and refined results against byte-identical NumPy
reference inputs:

    ~/miniforge3/envs/dedalus3/bin/python \
        cluster/local/run_backend_conformance.py

See `cluster/local/README.md` for output paths and the portable NumPy mechanics
check. This campaign deliberately contains no timing measurements.
