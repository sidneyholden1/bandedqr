#!/usr/bin/env python3
"""Create an immutable CPU-smoke campaign and submit it to Slurm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile


def run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args, cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(args)}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def materialize(template: str, values: dict[str, str]) -> str:
    rendered = template
    raw_tokens = {
        "__BANDQR_RUN_DIR__": values["run_dir"],
        "__BANDQR_SOURCE_DIR__": values["source_dir"],
    }
    shell_tokens = {
        "__BANDQR_CAMPAIGN_ID_SHELL__": values["campaign_id"],
        "__BANDQR_RUN_DIR_SHELL__": values["run_dir"],
        "__BANDQR_DATA_DIR_SHELL__": values["data_dir"],
        "__BANDQR_SOURCE_DIR_SHELL__": values["source_dir"],
        "__BANDQR_ENV_DIR_SHELL__": values["env_dir"],
        "__BANDQR_COMMIT_SHELL__": values["commit"],
    }
    for token, value in raw_tokens.items():
        rendered = rendered.replace(token, value)
    for token, value in shell_tokens.items():
        rendered = rendered.replace(token, shlex.quote(value))
    leftovers = sorted(set(re.findall(r"__BANDQR_[A-Z_]+__", rendered)))
    if leftovers:
        raise RuntimeError(f"unexpanded template tokens: {leftovers}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()
    os.umask(0o027)

    default_source = Path(__file__).resolve().parents[2]
    source = (args.source or default_source).resolve()
    runs_root = (args.runs_root or source / "runs").resolve()
    data_root = (
        args.data_root or Path("/mnt/home/sholden/ceph/bandqr")
    ).resolve()
    environment = (
        args.environment or Path("/mnt/home/sholden/envs/bandqr/cpu")
    ).resolve()
    config = source / "bench/paper/configs/cpu_smoke.json"
    template = source / "cluster/rusty/cpu_smoke.sbatch.in"

    if not environment.joinpath("bin/python").is_file():
        raise RuntimeError(
            f"missing environment {environment}; run bootstrap_cpu_env.sh first"
        )
    commit = run(["git", "rev-parse", "HEAD"], source)
    dirty = run(
        ["git", "status", "--porcelain"], source
    )
    if dirty:
        raise RuntimeError("refusing to submit a dirty source checkout")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_id = f"cpu-smoke-{stamp}-{commit[:10]}"
    run_dir = runs_root / campaign_id
    data_dir = data_root / campaign_id
    if run_dir.exists() or data_dir.exists():
        raise RuntimeError(f"refusing to reuse campaign ID {campaign_id}")
    runs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(mode=0o750)
    data_dir.mkdir(mode=0o750)
    (run_dir / "logs").mkdir(mode=0o750)

    values = {
        "campaign_id": campaign_id,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "source_dir": str(source),
        "env_dir": str(environment),
        "commit": commit,
    }
    script = materialize(template.read_text(encoding="utf-8"), values)
    submitted_script = run_dir / "submitted.slurm"
    atomic_text(submitted_script, script)
    submitted_script.chmod(0o750)

    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "created_utc": utc_now(),
        "created_on": socket.gethostname(),
        "campaign_kind": "correctness",
        "performance_measurement": False,
        "source": {
            "path": str(source),
            "commit": commit,
            "dirty": False,
        },
        "environment": str(environment),
        "dedalus_integration_environment": (
            "/mnt/home/sholden/network_instability/dedalus_codex_handoff/"
            ".venv-dedalus305"
        ),
        "config": {
            "path": str(config),
            "sha256": sha256(config),
        },
        "submitted_script": {
            "path": str(submitted_script),
            "sha256": sha256(submitted_script),
        },
        "output_path": str(data_dir),
        "resources": {
            "account": "ccb",
            "partition": "ccb",
            "nodes": 1,
            "tasks": 1,
            "cpus_per_task": 8,
            "memory": "16G",
            "walltime": "00:15:00",
            "gpus": 0,
        },
        "backends": {
            "required": ["cpu-c", "cpu-numpy-batched"],
            "not_scheduled": ["metal-mlx", "cuda-cupy"],
        },
    }
    atomic_json(run_dir / "campaign_manifest.json", manifest)
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": 1,
            "state": "SUBMITTED",
            "timestamp_utc": utc_now(),
            "hostname": socket.gethostname(),
            "job_id": None,
            "message": "submission in progress; see submission.json after acceptance",
        },
    )

    try:
        sbatch_output = run(["sbatch", "--parsable", str(submitted_script)], source)
        job_id = sbatch_output.split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            raise RuntimeError(f"unexpected sbatch job ID: {sbatch_output!r}")
    except Exception as error:
        atomic_json(
            run_dir / "submission_error.json",
            {"timestamp_utc": utc_now(), "error": str(error)},
        )
        atomic_json(
            run_dir / "status.json",
            {
                "schema_version": 1,
                "state": "SUBMIT_FAIL",
                "timestamp_utc": utc_now(),
                "hostname": socket.gethostname(),
                "job_id": None,
                "message": str(error),
            },
        )
        raise

    atomic_text(run_dir / "job_id.txt", job_id + "\n")
    atomic_json(
        run_dir / "submission.json",
        {
            "schema_version": 1,
            "submitted_utc": utc_now(),
            "command": ["sbatch", "--parsable", str(submitted_script)],
            "sbatch_output": sbatch_output,
            "job_id": job_id,
        },
    )
    print(json.dumps({"campaign_id": campaign_id, "job_id": job_id}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"submit_cpu_smoke: {error}", file=sys.stderr)
        raise SystemExit(1)
