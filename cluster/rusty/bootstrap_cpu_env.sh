#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly ENV_ROOT="${1:-/mnt/home/sholden/envs/bandedqr}"
readonly ENV_DIR="${ENV_ROOT}/cpu"
readonly SHARE_DIR="${ENV_ROOT}/share/cpu"
readonly LOG_DIR="${ENV_ROOT}/environment/logs"

mkdir -p "${LOG_DIR}" "${SHARE_DIR}"
readonly LOG_FILE="${LOG_DIR}/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -f "${SHARE_DIR}/READY" ]]; then
    if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
        echo "READY exists but ${ENV_DIR}/bin/python is missing" >&2
        exit 1
    fi
    echo "bandedqr CPU environment is already ready: ${ENV_DIR}"
    exit 0
fi
if [[ -e "${ENV_DIR}" ]]; then
    echo "refusing partial existing environment without READY: ${ENV_DIR}" >&2
    exit 1
fi
if ! type module >/dev/null 2>&1; then
    echo "the cluster module command is unavailable" >&2
    exit 1
fi

module reset || true
module load modules/2.4-20250724 || true
module load gcc/13.3.0 || true
module load openmpi/5.0.6 || true
module load python-mpi/3.12.9 || true

for required in \
    gcc/13.3.0 \
    openmpi/5.0.6 \
    python-mpi/3.12.9; do
    if ! module is-loaded "${required}"; then
        echo "failed to load ${required}" >&2
        exit 1
    fi
done

python -m venv --system-site-packages "${ENV_DIR}"
"${ENV_DIR}/bin/python" -c \
    'import numpy, scipy, pytest; print(numpy.__version__, scipy.__version__, pytest.__version__)'
module -t list 2> "${SHARE_DIR}/modules.txt"
"${ENV_DIR}/bin/python" -m pip freeze > "${SHARE_DIR}/pip-freeze.txt"
"${ENV_DIR}/bin/python" \
    "${SOURCE_DIR}/cluster/common/capture_provenance.py" \
    --kind environment \
    --output "${SHARE_DIR}/environment.json" \
    --source "${SOURCE_DIR}" \
    --environment "${ENV_DIR}"

readonly READY_TMP="${SHARE_DIR}/.READY.$$"
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    "${ENV_DIR}/bin/python" --version
} > "${READY_TMP}"
mv "${READY_TMP}" "${SHARE_DIR}/READY"
echo "bandedqr CPU environment ready: ${ENV_DIR}"
