# Rusty campaign control

The current Rusty workflow produces retained CPU and NVIDIA CUDA scaling
evidence for the public repository. It is deliberately cluster-specific: the
versioned policy in `resource_policy.json` records the Flatiron account,
partitions, hardware constraints, storage root, budget, and quota reserves used
for this campaign.

The concrete requests are:

| Mode | Allocation | Wall time | Full reservation |
|---|---|---:|---:|
| CPU | one exclusive Genoa node, 96 CPUs, 128 GiB | 45 minutes | 72 CPU-hours |
| CUDA | one H100 SXM5, 16 CPUs, 128 GiB | 60 minutes | 16 CPU-hours + 1 GPU-hour |

Each submission requires separate, explicit approval of these resources. The
controller has no automatic retry or resubmission path.

## Immutable workflow

From a clean checkout, prepare an archive of the exact source commit:

```bash
python cluster/rusty/scaling_control.py prepare
```

The command writes an immutable bundle beneath the registered Ceph campaign
root and reports its path. After receiving explicit approval for a concrete
allocation, submit exactly one mode and retain the approval reference:

```bash
python cluster/rusty/scaling_control.py submit \
  --mode cpu \
  --bundle /absolute/path/reported/by/prepare \
  --approval-reference "approval reference"

python cluster/rusty/scaling_control.py submit \
  --mode cuda \
  --bundle /absolute/path/reported/by/prepare \
  --approval-reference "approval reference"
```

The launcher verifies the bundle hashes, extracts the source onto node-local
scratch, records the bundle commit and manifest hash, builds from scratch, runs
the test suite, runs CUDA conformance for the GPU allocation, and only then
executes the timing driver. Outputs and checkpoints are written under the
Ceph `campaign_root` in `resource_policy.json`.

The driver checkpoints atomically after every case. A pre-timeout Slurm signal
and the quota monitor request a clean stop at the next case boundary. Stopped
or failed attempts are never resubmitted automatically.

Inspect the local ledger without querying Slurm:

```bash
python cluster/rusty/scaling_control.py status
```

After a job reaches a terminal state, reconcile the full parent allocation:

```bash
python cluster/rusty/scaling_control.py reconcile --job JOB_ID
```

`reconcile` makes one narrowly scoped accounting query and moves the actual
CPU/GPU hours from the reserved to the consumed ledger totals.

## Promoting accepted results

Only `PASS` records from the same config hash and source commit can be combined.
The renderer writes the retained SVG, Markdown table, and machine-readable
summary, then replaces only the marked results block in the README:

```bash
python bench/paper/render_scaling.py \
  --cpu /path/to/cpu/attempt/scaling.json \
  --cuda /path/to/cuda/attempt/scaling.json \
  --figure figures/rusty-scaling.svg \
  --table bench/results/rusty-scaling.md \
  --summary bench/results/rusty-scaling-summary.json \
  --readme README.md
```

The compact accepted artifacts belong in Git. Raw checkpoints, logs, module
lists, scheduler records, and other detailed attempt data remain on the
registered Ceph project root.

The older correctness-only CPU launcher and local Metal conformance workflow
remain in this directory for historical evidence. See `local/README.md` for the
Metal gate; it deliberately records correctness rather than performance.
