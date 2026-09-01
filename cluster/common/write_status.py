#!/usr/bin/env python3
"""Atomically update a campaign status record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile


def atomic_json(path: Path, value: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state",
        choices=["SUBMITTED", "RUNNING", "PASS", "FAIL", "SUBMIT_FAIL"],
        required=True,
    )
    parser.add_argument("--message", default="")
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "state": args.state,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "hostname": socket.gethostname(),
        "job_id": args.job_id,
        "message": args.message,
    }
    atomic_json(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
