# Local accelerator conformance

The local runner is the Metal counterpart of the Rusty Slurm control plane. It
is a correctness campaign, not a benchmark: it records raw fp32/complex64 and
refined results but intentionally records no timings.

Run it from a normal logged-in macOS Terminal so MLX can create a Metal command
queue:

    ~/miniforge3/envs/dedalus3/bin/python \
        cluster/local/run_backend_conformance.py

The source checkout must be clean. The runner archives the exact Git commit,
builds and executes that archive in a temporary directory, refuses to reuse a
campaign directory, and atomically records RUNNING followed by PASS or FAIL.

Small control records are written below:

    runs/<campaign-id>/

The retained source archive, exact config, detailed verification JSON, and logs
default to:

    ~/bandqr-data/<campaign-id>/

Set BANDQR_LOCAL_DATA_ROOT or pass --data-root to choose another large-data
location. The run record captures the resolved path.

The same machinery can be checked without Metal by selecting the portable
reference:

    ~/miniforge3/envs/dedalus3/bin/python \
        cluster/local/run_backend_conformance.py --backend numpy

That portable run validates campaign mechanics; it is not evidence for the
Metal backend. A Metal PASS requires an actual evaluated MLX operation and
reports the concrete implementation identity metal-mlx.
