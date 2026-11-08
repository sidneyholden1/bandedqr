#!/usr/bin/env python3
"""Prepare, submit, and reconcile immutable Rusty scaling attempts.

Every ``submit`` invocation creates exactly one Slurm allocation and requires
an explicit approval reference.  There are no automatic retries or resubmits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from quota_monitor import snapshot as quota_snapshot


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "cluster/resource_policy.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slurm_local_timestamp(epoch: float) -> str:
    """Format an epoch in the scheduler's local timezone for ``sacct``."""
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%S")


def command(arguments: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {arguments!r}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def paths(config: dict) -> tuple[Path, Path]:
    campaign = Path(config["storage"]["campaign_root"])
    ledger = Path(config["storage"]["ledger"])
    return campaign, ledger


def initial_ledger(config: dict) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "created_utc": utc_now(),
        "campaign_cpu_hour_ceiling": config["campaign_cpu_hour_ceiling"],
        "campaign_gpu_hour_ceiling": config["campaign_gpu_hour_ceiling"],
        "max_simultaneous_nodes": config["max_simultaneous_nodes"],
        "allocated_cpu_hours_used": 0.0,
        "allocated_gpu_hours_used": 0.0,
        "reserved_cpu_hours": 0.0,
        "reserved_gpu_hours": 0.0,
        "jobs": [],
    }


def locked_ledger(ledger_path: Path):
    class LedgerLock:
        def __enter__(self):
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = ledger_path.with_suffix(".lock").open("a+")
            fcntl.flock(self.stream, fcntl.LOCK_EX)
            return self

        def __exit__(self, exc_type, exc, traceback):
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()

    return LedgerLock()


def verify_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        candidate = (bundle / relative).resolve()
        if not candidate.is_relative_to(bundle):
            raise ValueError("bundle member escaped immutable root")
        if sha256(candidate) != expected:
            raise ValueError(f"immutable bundle member changed: {relative}")
    return manifest


def prepare() -> None:
    config = policy()
    campaign, ledger_path = paths(config)
    dirty = command(["git", "status", "--porcelain"])
    if dirty:
        raise ValueError("refusing to prepare a bundle from a dirty checkout")
    commit = command(["git", "rev-parse", "HEAD"])
    bundle = campaign / "attempts" / f"bundle-{commit[:12]}"
    if bundle.exists():
        verify_bundle(bundle)
        print(json.dumps({"bundle": str(bundle), "reused": True}, indent=2))
        return
    bundle.mkdir(parents=True, mode=0o750)
    source_tar = bundle / "source.tar"
    command(["git", "archive", "--format=tar", "-o", str(source_tar), commit])
    shutil.copy2(
        "/mnt/home/sholden/cluster/bin/allocation-info.py",
        bundle / "allocation-info.py",
    )
    shutil.copy2(POLICY_PATH, bundle / "resource_policy.json")
    shutil.copy2(
        ROOT / "cluster/rusty/run_scaling.sh", bundle / "run_scaling.sh"
    )
    atomic_text(
        bundle / "source.tar.sha256",
        f"{sha256(source_tar)}  {source_tar}\n",
    )
    members = [
        "source.tar",
        "source.tar.sha256",
        "allocation-info.py",
        "resource_policy.json",
        "run_scaling.sh",
    ]
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_commit": commit,
        "source_branch": command(["git", "branch", "--show-current"]),
        "source_dirty": False,
        "scope": config["objective"],
        "resources": config["concrete_allocations"],
        "sha256": {name: sha256(bundle / name) for name in members},
    }
    atomic_json(bundle / "manifest.json", manifest)
    for path in bundle.iterdir():
        path.chmod(0o440)
    bundle.chmod(0o550)

    campaign.joinpath("control").mkdir(parents=True, exist_ok=True)
    campaign.joinpath("cpu").mkdir(exist_ok=True)
    campaign.joinpath("cuda").mkdir(exist_ok=True)
    campaign.joinpath("job_caches").mkdir(exist_ok=True)
    with locked_ledger(ledger_path):
        if not ledger_path.exists():
            atomic_json(ledger_path, initial_ledger(config))
    print(json.dumps({"bundle": str(bundle), "reused": False}, indent=2))


def preflight(config: dict) -> dict:
    result = quota_snapshot(config)
    if not result["admit"]:
        raise ValueError(f"quota admission refused: {result['checks']}")
    return result


def submit(mode: str, bundle: Path, approval_reference: str) -> None:
    config = policy()
    campaign, ledger_path = paths(config)
    manifest = verify_bundle(bundle)
    resources = config["concrete_allocations"][mode]
    if manifest["resources"][mode] != resources:
        raise ValueError("concrete resources changed since bundle preparation")
    quota = preflight(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = campaign / mode / f"attempt-{stamp}-{manifest['source_commit'][:10]}"
    attempt.mkdir(parents=True, mode=0o750)
    (attempt / "logs").mkdir(mode=0o750)
    atomic_json(attempt / "quota-preflight.json", quota)
    reservation_id = str(time.time_ns())

    with locked_ledger(ledger_path):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        active = [
            row
            for row in ledger["jobs"]
            if row["status"] in {"reserved", "submitted", "pending", "running"}
        ]
        if any(row["mode"] == mode for row in active):
            raise ValueError(f"an active {mode} allocation already exists")
        if sum(int(row["nodes"]) for row in active) + int(resources["nodes"]) > int(
            ledger["max_simultaneous_nodes"]
        ):
            raise ValueError("project simultaneous-node ceiling exceeded")
        prospective_cpu = (
            float(ledger["allocated_cpu_hours_used"])
            + float(ledger["reserved_cpu_hours"])
            + float(resources["reserved_cpu_hours"])
        )
        prospective_gpu = (
            float(ledger["allocated_gpu_hours_used"])
            + float(ledger["reserved_gpu_hours"])
            + float(resources["reserved_gpu_hours"])
        )
        if prospective_cpu > float(ledger["campaign_cpu_hour_ceiling"]):
            raise ValueError("campaign CPU-hour ceiling exceeded")
        if prospective_gpu > float(ledger["campaign_gpu_hour_ceiling"]):
            raise ValueError("campaign GPU-hour ceiling exceeded")
        ticket = {
            "reservation_id": reservation_id,
            "mode": mode,
            "status": "reserved",
            "nodes": resources["nodes"],
            "reserved_cpus": resources["reserved_cpus"],
            "reserved_cpu_hours": resources["reserved_cpu_hours"],
            "reserved_gpu_hours": resources["reserved_gpu_hours"],
            "resources": resources,
            "bundle": str(bundle.resolve()),
            "attempt": str(attempt),
            "approval_reference": approval_reference,
            "reserved_utc": utc_now(),
            "reserved_epoch": time.time(),
        }
        ledger["jobs"].append(ticket)
        ledger["reserved_cpu_hours"] += float(resources["reserved_cpu_hours"])
        ledger["reserved_gpu_hours"] += float(resources["reserved_gpu_hours"])
        atomic_json(ledger_path, ledger)

    wall_seconds = int(resources["walltime"].split(":")[0]) * 3600 + int(
        resources["walltime"].split(":")[1]
    ) * 60
    sbatch = [
        "sbatch",
        "--parsable",
        f"--partition={resources['partition']}",
        f"--account={resources['account']}",
        f"--constraint={resources['constraint']}",
        f"--nodes={resources['nodes']}",
        "--ntasks=1",
        f"--cpus-per-task={resources['numerical_cpus']}",
        f"--mem={resources['memory']}",
        f"--time={resources['walltime']}",
        "--signal=B:USR1@180",
        "--no-requeue",
        f"--job-name=bq-{mode}-scale",
        f"--output={attempt}/logs/slurm-%j.out",
        f"--error={attempt}/logs/slurm-%j.err",
    ]
    if int(resources["gpus"]):
        sbatch.append(f"--gpus={resources['gpus']}")
    sbatch += [
        str(bundle.resolve() / "run_scaling.sh"),
        mode,
        str(bundle.resolve()),
        str(attempt),
        str(bundle.resolve() / "resource_policy.json"),
        str(resources["reserved_cpus"]),
        str(config["nodes_per_job_ceiling"]),
        str(wall_seconds),
    ]
    atomic_json(
        attempt / "submission-intent.json",
        {
            "created_utc": utc_now(),
            "mode": mode,
            "resources": resources,
            "command": sbatch,
            "bundle": str(bundle.resolve()),
            "approval_reference": approval_reference,
        },
    )
    try:
        reply = command(sbatch)
        job_id = reply.split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError(f"unexpected sbatch reply {reply!r}")
    except BaseException:
        # A timeout is an unknown scheduler outcome. Keep the reservation for
        # explicit inspection; never assume it is safe to retry.
        raise
    with locked_ledger(ledger_path):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        row = next(
            item for item in ledger["jobs"]
            if item["reservation_id"] == reservation_id
        )
        row.update(status="submitted", job_id=job_id, submitted_utc=utc_now())
        atomic_json(ledger_path, ledger)
    atomic_json(
        attempt / "submission.json",
        {"job_id": job_id, "sbatch_reply": reply, "submitted_utc": utc_now()},
    )
    print(
        json.dumps(
            {"job_id": job_id, "mode": mode, "attempt": str(attempt)},
            indent=2,
        )
    )


def terminal_state(state: str) -> bool:
    return state.split()[0].rstrip("+") in {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }


def reconcile(job_id: str) -> None:
    config = policy()
    _, ledger_path = paths(config)
    with locked_ledger(ledger_path):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        row = next(item for item in ledger["jobs"] if item.get("job_id") == job_id)
        now = time.time()
        if now - float(row.get("last_accounting_query_epoch", 0)) < 60:
            raise ValueError("accounting queries must be at least 60 seconds apart")
        row["last_accounting_query_epoch"] = now
        atomic_json(ledger_path, ledger)
        start = slurm_local_timestamp(float(row["reserved_epoch"]) - 60)
    raw = command(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-u",
            os.environ["USER"],
            "-j",
            job_id,
            f"--starttime={start}",
            "--format=JobIDRaw,State,AllocCPUS,ElapsedRaw,AllocTRES,Start,End",
        ]
    )
    fields = next(
        (line.split("|") for line in raw.splitlines() if line.split("|")[0] == job_id),
        None,
    )
    if fields is None:
        raise ValueError("parent job accounting is not available; reservation retained")
    _, state, cpus, elapsed, tres, started, ended = fields[:7]
    tres_values = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in tres.split(",")
        if "=" in item
    }
    billed_cpus = max(float(cpus), float(tres_values.get("billing", cpus)))
    elapsed_hours = int(elapsed) / 3600.0
    actual_cpu = billed_cpus * elapsed_hours
    gpu_count = 0.0
    for key, value in tres_values.items():
        if key in {"gres/gpu", "gres/gpu:h100"}:
            gpu_count = max(gpu_count, float(value))
    actual_gpu = gpu_count * elapsed_hours
    terminal = terminal_state(state)
    with locked_ledger(ledger_path):
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        row = next(item for item in ledger["jobs"] if item.get("job_id") == job_id)
        row.update(
            sacct_state=state,
            allocated_or_billing_cpus=billed_cpus,
            elapsed_seconds=int(elapsed),
            measured_cpu_hours=actual_cpu,
            measured_gpu_hours=actual_gpu,
            started=started,
            ended=ended,
            accounting_raw=raw,
        )
        if terminal and row["status"] != "reconciled":
            ledger["reserved_cpu_hours"] -= float(row["reserved_cpu_hours"])
            ledger["reserved_gpu_hours"] -= float(row["reserved_gpu_hours"])
            ledger["allocated_cpu_hours_used"] += actual_cpu
            ledger["allocated_gpu_hours_used"] += actual_gpu
            row.update(status="reconciled", reconciled_utc=utc_now())
        atomic_json(ledger_path, ledger)
    print(
        json.dumps(
            {
                "job_id": job_id,
                "state": state,
                "terminal": terminal,
                "actual_cpu_hours": actual_cpu,
                "actual_gpu_hours": actual_gpu,
            },
            indent=2,
        )
    )


def show_status() -> None:
    config = policy()
    _, ledger_path = paths(config)
    if not ledger_path.exists():
        print(json.dumps(initial_ledger(config), indent=2))
        return
    print(ledger_path.read_text(encoding="utf-8"), end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--mode", choices=["cpu", "cuda"], required=True)
    submit_parser.add_argument("--bundle", type=Path, required=True)
    submit_parser.add_argument("--approval-reference", required=True)
    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--job", required=True)
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "submit":
        submit(args.mode, args.bundle, args.approval_reference)
    elif args.command == "reconcile":
        reconcile(args.job)
    else:
        show_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
