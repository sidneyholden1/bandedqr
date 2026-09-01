#!/usr/bin/env python3
"""Create the small home-directory verification summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


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


def junit_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    keys = ("tests", "failures", "errors", "skipped")
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--dedalus", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.cpu.open(encoding="utf-8") as stream:
        cpu = json.load(stream)
    with args.dedalus.open(encoding="utf-8") as stream:
        dedalus = json.load(stream)
    pytest_counts = junit_counts(args.junit)
    passed = bool(
        cpu["pass"]
        and dedalus["pass"]
        and pytest_counts["failures"] == 0
        and pytest_counts["errors"] == 0
    )
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "pass": passed,
        "portable_test_suite": pytest_counts,
        "cpu_smoke": {
            "pass": cpu["pass"],
            "checks": len(cpu["checks"]),
        },
        "dedalus_smoke": {
            "pass": dedalus["pass"],
            "matrix_sha256": dedalus["matrix_sha256"],
            "matrix_shape": dedalus["matrix_shape"],
        },
        "accelerators": {
            "metal-mlx": "not_run_on_rusty_cpu_gate",
            "cuda-cupy": "planned_not_implemented",
        },
    }
    atomic_json(args.output, record)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
