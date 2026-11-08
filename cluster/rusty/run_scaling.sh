#!/usr/bin/env bash
set -euo pipefail
umask 027

if [[ $# -ne 7 ]]; then
    echo "usage: run_scaling.sh MODE BUNDLE OUTPUT POLICY RESERVED_CPUS MAX_NODES WALL_SECONDS" >&2
    exit 2
fi

readonly MODE=$1
readonly BUNDLE=$2
readonly OUTPUT=$3
readonly POLICY=$4
readonly RESERVED_CPUS=$5
readonly MAX_NODES=$6
readonly WALL_SECONDS=$7
readonly CAMPAIGN_ROOT=/mnt/ceph/users/sholden/bandedqr/runs/rusty-scaling-20261108

if [[ "${MODE}" != cpu && "${MODE}" != cuda ]]; then
    echo "unknown mode: ${MODE}" >&2
    exit 2
fi
if [[ ! -d "${BUNDLE}" || ! -d "${OUTPUT}" || ! -f "${POLICY}" ]]; then
    echo "missing immutable bundle, output directory, or policy" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_TMPDIR:-}" ]]; then
    echo "this launcher requires a Slurm job with SLURM_TMPDIR" >&2
    exit 2
fi
if [[ ! -d "${SLURM_TMPDIR}" || ! -w "${SLURM_TMPDIR}" ]]; then
    echo "SLURM_TMPDIR is unavailable: ${SLURM_TMPDIR}" >&2
    exit 2
fi

python3 "${BUNDLE}/allocation-info.py" \
    --reserved-cpus "${RESERVED_CPUS}" \
    --max-nodes "${MAX_NODES}" > "${OUTPUT}/allocation.json"

module reset
module load modules/2.5-20261027
module load python/3.12.13
if [[ "${MODE}" == cuda ]]; then
    module load cuda/13.3.0
fi

readonly CACHE_ROOT="${CAMPAIGN_ROOT}/job_caches/${SLURM_JOB_ID}"
mkdir -p "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/matplotlib" "${OUTPUT}/logs"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export TMPDIR="${SLURM_TMPDIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# The extracted archive has no .git directory. Carry source identity from the
# verified immutable manifest rather than consulting the submitter's checkout.
export BANDEDQR_SOURCE_COMMIT
BANDEDQR_SOURCE_COMMIT="$(
    python3 -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["source_commit"])' \
        "${BUNDLE}/manifest.json"
)"
export BANDEDQR_SOURCE_DIRTY=0
export BANDEDQR_BUNDLE_MANIFEST_SHA256
BANDEDQR_BUNDLE_MANIFEST_SHA256="$(
    sha256sum "${BUNDLE}/manifest.json" | cut -d' ' -f1
)"

readonly WORK_ROOT="${SLURM_TMPDIR}/bandedqr-${SLURM_JOB_ID}"
readonly SOURCE="${WORK_ROOT}/source"
mkdir -p "${SOURCE}"
sha256sum --check "${BUNDLE}/source.tar.sha256"
tar -xf "${BUNDLE}/source.tar" -C "${SOURCE}"

module -t list 2> "${OUTPUT}/modules.txt"
{
    hostname
    uname -a
    python -VV
    cc --version | head -1
    lscpu | sed -n '1,28p'
    if [[ "${MODE}" == cuda ]]; then
        nvidia-smi -L
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
            --format=csv,noheader
    fi
} > "${OUTPUT}/environment.txt"

python "${SOURCE}/cluster/common/write_status.py" \
    --output "${OUTPUT}/status.json" --state RUNNING \
    --message "${MODE} qualification started"

make -B -C "${SOURCE}" > "${OUTPUT}/logs/build.log" 2>&1
export PYTHONPATH="${SOURCE}/src"
python -m pytest "${SOURCE}/tests" -q \
    --junitxml="${OUTPUT}/pytest.xml" > "${OUTPUT}/logs/pytest.log" 2>&1

if [[ "${MODE}" == cuda ]]; then
    python "${SOURCE}/bench/paper/drivers/backend_conformance.py" \
        --backend cuda \
        --config "${SOURCE}/bench/paper/configs/backend_conformance.json" \
        --output "${OUTPUT}/cuda_conformance.json" \
        > "${OUTPUT}/logs/cuda-conformance.log" 2>&1
fi

child_pid=""
monitor_pid=""
forward_checkpoint() {
    if [[ -n "${child_pid}" ]]; then
        kill -USR1 "${child_pid}" 2>/dev/null || true
    fi
}
cleanup_monitor() {
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
}
trap forward_checkpoint USR1 TERM
trap cleanup_monitor EXIT

python "${SOURCE}/bench/paper/drivers/scaling.py" \
    --mode "${MODE}" \
    --config "${SOURCE}/bench/paper/configs/rusty_scaling.json" \
    --output "${OUTPUT}/scaling.json" \
    > "${OUTPUT}/logs/scaling.log" 2>&1 &
child_pid=$!

python "${SOURCE}/cluster/rusty/quota_monitor.py" \
    --pid "${child_pid}" \
    --policy "${POLICY}" \
    --output "${OUTPUT}/quota-monitor.json" \
    --interval 300 \
    > "${OUTPUT}/logs/quota-monitor.log" 2>&1 &
monitor_pid=$!

set +e
wait "${child_pid}"
child_status=$?
set -e
child_pid=""
cleanup_monitor
monitor_pid=""

if [[ "${child_status}" -eq 75 ]]; then
    python "${SOURCE}/cluster/common/write_status.py" \
        --output "${OUTPUT}/status.json" --state CHECKPOINTED \
        --message "${MODE} scaling checkpointed after a stop request"
    exit 75
fi
if [[ "${child_status}" -ne 0 ]]; then
    python "${SOURCE}/cluster/common/write_status.py" \
        --output "${OUTPUT}/status.json" --state FAIL \
        --message "${MODE} scaling exited ${child_status}"
    exit "${child_status}"
fi

python "${SOURCE}/cluster/common/write_status.py" \
    --output "${OUTPUT}/status.json" --state PASS \
    --message "${MODE} qualification and scaling passed"

echo "${MODE} scaling PASS in ${OUTPUT}; wall allowance ${WALL_SECONDS}s"
