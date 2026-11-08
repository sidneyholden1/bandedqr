#!/usr/bin/env python3
"""Bounded FI quota monitor that requests a checkpoint on lost headroom."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


def command(arguments: list[str]) -> str:
    return subprocess.check_output(arguments, text=True, timeout=60)


def clean(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def usage_metric(report: str, section: str) -> tuple[int, int]:
    block = clean(report).split(section, 1)[1]
    used = int(re.search(r"Used:\s*(\d+)", block)[1])
    planned = int(re.search(r"Planned:\s*(\d+)", block)[1])
    return used, planned


def quota_metric(report: str, section: str) -> tuple[int, int]:
    block = clean(report).split(section, 1)[1]
    current = int(re.search(r"Current:\s*(\d+)", block)[1])
    soft = int(re.search(r"Soft Limit:\s*(\d+)", block)[1])
    return current, soft


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def snapshot(policy: dict) -> dict:
    usage = command(["/mnt/sw/fi/utils/bin/fi-usage", os.environ["USER"]])
    home = command(["/mnt/sw/fi/utils/bin/fi-quota"])
    ceph = json.loads(
        command(
            [
                "/usr/bin/python3",
                "/mnt/home/sholden/cluster/bin/ceph-status.py",
                "--json",
            ]
        )
    )
    cpu_used, cpu_plan = usage_metric(usage, "CPU Hours")
    home_entries, home_soft = quota_metric(home, "FILE COUNT")
    storage_files_used, storage_files_plan = usage_metric(
        usage, "Storage Files"
    )
    ceph_plan = min(int(ceph["planned_entries"]), storage_files_plan)
    reserves = policy["quota_reserves"]
    checks = {
        "cpu_hours": cpu_plan > 0
        and cpu_used < cpu_plan - int(reserves["cpu_hours"]),
        "home_entries": home_entries
        < home_soft - int(reserves["home_entries_below_soft_limit"]),
        "ceph_entries": int(ceph["entries"])
        < ceph_plan - int(reserves["ceph_entries_below_plan"]),
    }
    return {
        "queried_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_used": cpu_used,
        "cpu_plan": cpu_plan,
        "home_entries": home_entries,
        "home_soft_entries": home_soft,
        "ceph_entries": int(ceph["entries"]),
        "ceph_plan_entries": ceph_plan,
        "fi_storage_files_used": storage_files_used,
        "checks": checks,
        "admit": all(checks.values()),
        "note": "FI usage can lag by about one hour; configured reserves and the campaign ledger cover committed work.",
    }


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.interval < 30:
        raise ValueError("quota queries must be at least 30 seconds apart")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    history = []
    while process_alive(args.pid):
        try:
            current = snapshot(policy)
        except BaseException as error:
            current = {
                "queried_utc": datetime.now(timezone.utc).isoformat(),
                "admit": False,
                "checks": {"quota_known": False},
                "error": repr(error),
            }
        history.append(current)
        atomic_json(
            args.output,
            {
                "schema_version": 1,
                "target_pid": args.pid,
                "interval_seconds": args.interval,
                "history": history,
                "latest": current,
            },
        )
        if not current["admit"]:
            try:
                os.kill(args.pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass
            return 75
        deadline = time.monotonic() + args.interval
        while process_alive(args.pid) and time.monotonic() < deadline:
            time.sleep(min(10, max(0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
