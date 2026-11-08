#!/usr/bin/env python3
"""Capture source, environment, scheduler, and hardware provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile


def command(args: list[str], cwd: Path | None = None) -> dict:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def package_versions() -> dict:
    result = {}
    for name in (
        "bandedqr", "numpy", "scipy", "pytest", "mlx", "dedalus", "mpi4py"
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--kind", default="campaign")
    parser.add_argument("--hash", action="append", type=Path, default=[])
    args = parser.parse_args()
    source = args.source.resolve()
    environment = args.environment.resolve()
    selected_environment = {
        key: os.environ.get(key)
        for key in (
            "LOADEDMODULES",
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
            "PYTHONPATH",
            "BANDEDQR_LOCAL_DATA_ROOT",
            "BANDEDQR_LIBRARY",
            "BANDEDQR_THREAD_MIN_KL",
            "BANDEDQR_THREAD_MIN_WORK",
            "BANDEDQR_THREAD_MAX",
            "BANDEDQR_CACHE",
        )
    }
    is_macos = platform.system() == "Darwin"
    record = {
        "schema_version": 1,
        "kind": args.kind,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "environment_path": str(environment),
        "packages": package_versions(),
        "environment": selected_environment,
        "source": {
            "path": str(source),
            "expected_commit": args.expected_commit,
            "expected_commit_object": command(
                ["git", "rev-parse", f"{args.expected_commit}^{{commit}}"],
                source,
            ) if args.expected_commit else None,
            "commit": command(["git", "rev-parse", "HEAD"], source),
            "status": command(
                ["git", "status", "--porcelain"],
                source,
            ),
        },
        "compiler": command(["cc", "--version"]),
        "cpu": (
            {
                "brand": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "model": command(["sysctl", "-n", "hw.model"]),
                "logical_cpus": command(["sysctl", "-n", "hw.ncpu"]),
                "memory_bytes": command(["sysctl", "-n", "hw.memsize"]),
            }
            if is_macos
            else command(["lscpu", "--json"])
        ),
        "macos": (
            {
                "version": command(["sw_vers"]),
                "metal_display_hardware": command(
                    ["system_profiler", "SPDisplaysDataType", "-json"]
                ),
            }
            if is_macos
            else None
        ),
        "slurm": command(["scontrol", "show", "job", "-dd", os.environ.get(
            "SLURM_JOB_ID", ""
        )]) if os.environ.get("SLURM_JOB_ID") else None,
        "files": {
            str(path.resolve()): sha256(path.resolve()) for path in args.hash
        },
    }
    atomic_json(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
