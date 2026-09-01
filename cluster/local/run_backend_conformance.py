#!/usr/bin/env python3
"""Run an immutable local batched-backend conformance campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def capture(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args, cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(args)}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def logged(
    args: list[str],
    cwd: Path,
    log: Path,
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {shlex.join(args)}\n")
        stream.flush()
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stream.write(f"\nexit_code={result.returncode}\n")
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(args)}; "
            f"see {log}"
        )
    return result


def write_status(run_dir: Path, state: str, message: str) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": 1,
            "state": state,
            "timestamp_utc": utc_now(),
            "hostname": socket.gethostname(),
            "job_id": None,
            "message": message,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["metal", "numpy"], default="metal")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    os.umask(0o027)

    default_source = Path(__file__).resolve().parents[2]
    source = (args.source or default_source).resolve()
    config = (
        args.config
        or source / "bench/paper/configs/backend_conformance.json"
    ).resolve()
    runs_root = (args.runs_root or source / "runs").resolve()
    configured_data_root = os.environ.get("BANDQR_LOCAL_DATA_ROOT")
    data_root = (
        args.data_root
        or (Path(configured_data_root) if configured_data_root else None)
        or Path.home() / "bandqr-data"
    ).resolve()
    python = (args.python or Path(sys.executable)).resolve()
    environment = python.parent.parent

    if not python.is_file():
        raise RuntimeError(f"Python executable does not exist: {python}")
    if not config.is_file():
        raise RuntimeError(f"configuration does not exist: {config}")
    commit = capture(["git", "rev-parse", "HEAD"], source)
    dirty = capture(["git", "status", "--porcelain"], source)
    if dirty:
        raise RuntimeError("refusing to run a campaign from a dirty source checkout")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_id = f"{args.backend}-conformance-{stamp}-{commit[:10]}"
    run_dir = runs_root / campaign_id
    data_dir = data_root / campaign_id
    if run_dir.exists() or data_dir.exists():
        raise RuntimeError(f"refusing to reuse campaign ID {campaign_id}")
    runs_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(mode=0o750)
    data_dir.mkdir(mode=0o750)
    (run_dir / "logs").mkdir(mode=0o750)
    for child in ("input", "logs", "source", "verification"):
        (data_dir / child).mkdir(mode=0o750)

    copied_config = data_dir / "input/backend_conformance.json"
    shutil.copy2(config, copied_config)
    source_archive = data_dir / "source/source.tar.gz"
    invocation = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "cwd": os.getcwd(),
        "argv": [sys.executable, *sys.argv],
        "resolved": {
            "source": str(source),
            "config": str(config),
            "runs_root": str(runs_root),
            "data_root": str(data_root),
            "python": str(python),
            "backend": args.backend,
        },
    }
    atomic_json(run_dir / "invocation.json", invocation)

    implementation_id = {
        "metal": "metal-mlx",
        "numpy": "cpu-numpy-batched",
    }[args.backend]
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
            "archive": str(source_archive),
            "archive_sha256": None,
        },
        "environment": str(environment),
        "python": str(python),
        "config": {
            "original_path": str(config),
            "copied_path": str(copied_config),
            "sha256": sha256(copied_config),
        },
        "invocation": str(run_dir / "invocation.json"),
        "output_path": str(data_dir),
        "resources": {
            "scheduler": None,
            "host": socket.gethostname(),
            "exclusive": False,
            "accelerator_devices_requested": 1 if args.backend == "metal" else 0,
        },
        "backends": {
            "required": [implementation_id],
            "reference": ["cpu-numpy-batched"],
            "not_scheduled": (
                ["cuda-cupy"] if args.backend == "metal"
                else ["metal-mlx", "cuda-cupy"]
            ),
        },
    }
    atomic_json(run_dir / "campaign_manifest.json", manifest)
    write_status(run_dir, "RUNNING", "local correctness conformance started")

    try:
        archive_log = data_dir / "logs/source-archive.log"
        logged(
            [
                "git",
                "archive",
                "--format=tar.gz",
                f"--output={source_archive}",
                commit,
            ],
            source,
            archive_log,
        )
        manifest["source"]["archive_sha256"] = sha256(source_archive)
        atomic_json(run_dir / "campaign_manifest.json", manifest)

        with tempfile.TemporaryDirectory(prefix="bandqr-conformance-") as temporary:
            snapshot = Path(temporary) / "source"
            snapshot.mkdir()
            with tarfile.open(source_archive, mode="r:gz") as archive:
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(snapshot, filter="data")
                else:  # Python 3.9-3.11
                    archive.extractall(snapshot)

            environment_vars = os.environ.copy()
            environment_vars["PYTHONPATH"] = str(snapshot / "src")
            scrubbed = [
                "BANDQR_LIBRARY",
                "BANDQR_THREAD_MIN_KL",
                "BANDQR_THREAD_MIN_WORK",
                "BANDQR_THREAD_MAX",
                "BANDQR_CACHE",
            ]
            for name in scrubbed:
                environment_vars.pop(name, None)
            logged(
                [
                    "make",
                    "-B",
                    "-C",
                    str(snapshot),
                    "CC=cc",
                    f"PY={python}",
                ],
                snapshot,
                data_dir / "logs/package-build.log",
                env=environment_vars,
            )

            provenance_args = [
                str(python),
                str(snapshot / "cluster/common/capture_provenance.py"),
                "--output",
                str(run_dir / "provenance.json"),
                "--source",
                str(source),
                "--environment",
                str(environment),
                "--expected-commit",
                commit,
                "--kind",
                "local-backend-conformance",
                "--hash",
                str(source_archive),
                "--hash",
                str(copied_config),
                "--hash",
                str(snapshot / "bench/paper/drivers/backend_conformance.py"),
                "--hash",
                str(snapshot / "src/bandqr/gpu.py"),
            ]
            logged(
                provenance_args,
                snapshot,
                data_dir / "logs/provenance.log",
                env=environment_vars,
            )

            details = data_dir / "verification/backend.json"
            result = logged(
                [
                    str(python),
                    str(
                        snapshot
                        / "bench/paper/drivers/backend_conformance.py"
                    ),
                    "--backend",
                    args.backend,
                    "--config",
                    str(copied_config),
                    "--output",
                    str(details),
                ],
                snapshot,
                data_dir / "logs/backend-conformance.log",
                env=environment_vars,
                check=False,
            )
            if not details.is_file():
                raise RuntimeError(
                    "conformance driver produced no result; see "
                    f"{data_dir / 'logs/backend-conformance.log'}"
                )
            detailed_result = json.loads(details.read_text(encoding="utf-8"))
            verification = {
                "schema_version": 1,
                "campaign_id": campaign_id,
                "backend": detailed_result.get("backend"),
                "summary": detailed_result.get("summary"),
                "details": {
                    "path": str(details),
                    "sha256": sha256(details),
                },
                "driver_exit_code": result.returncode,
                "pass": bool(
                    result.returncode == 0 and detailed_result.get("pass")
                ),
            }
            if "error" in detailed_result:
                verification["error"] = detailed_result["error"]
            atomic_json(run_dir / "verification.json", verification)

        if not verification["pass"]:
            write_status(run_dir, "FAIL", "backend conformance failed")
            print(
                json.dumps(
                    {
                        "campaign_id": campaign_id,
                        "state": "FAIL",
                        "run_dir": str(run_dir),
                        "data_dir": str(data_dir),
                    },
                    indent=2,
                )
            )
            return 1

        write_status(run_dir, "PASS", "backend conformance passed")
        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "state": "PASS",
                    "run_dir": str(run_dir),
                    "data_dir": str(data_dir),
                },
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        atomic_json(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "timestamp_utc": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        write_status(run_dir, "FAIL", str(error))
        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "state": "FAIL",
                    "run_dir": str(run_dir),
                    "data_dir": str(data_dir),
                    "error": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"run_backend_conformance: {error}", file=sys.stderr)
        raise SystemExit(1)
