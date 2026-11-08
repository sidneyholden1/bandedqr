#!/usr/bin/env python3
"""Checkpointed CPU/CUDA scaling measurements for the public project record.

The driver writes one atomic JSON checkpoint after every case.  Re-running
with ``--resume`` skips completed case keys.  SIGTERM and Slurm's pre-timeout
SIGUSR1 request a clean stop at the next case boundary (exit status 75).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Callable

import numpy as np

from bandedqr import BandQR, band_matvec, to_scipy_band


STOP_REQUESTED = False


def request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(mode: str) -> dict:
    bundled_commit = os.environ.get("BANDEDQR_SOURCE_COMMIT")
    bundled_dirty = os.environ.get("BANDEDQR_SOURCE_DIRTY")
    return {
        "mode": mode,
        "started_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "source_commit": bundled_commit or git_value("rev-parse", "HEAD"),
        "source_dirty": (
            bundled_dirty.strip().lower() not in {"", "0", "false", "no"}
            if bundled_dirty is not None
            else bool(git_value("status", "--porcelain"))
        ),
        "bundle_manifest_sha256": os.environ.get(
            "BANDEDQR_BUNDLE_MANIFEST_SHA256"
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_constraint": os.environ.get("SLURM_JOB_CONSTRAINTS"),
    }


@contextmanager
def gc_disabled():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def summary(samples: list[float], inner: int) -> dict:
    ordered = sorted(float(value) for value in samples)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "unit": "seconds",
        "inner_iterations": int(inner),
        "samples": ordered,
        "minimum": ordered[0],
        "q1": quartiles[0],
        "median": statistics.median(ordered),
        "q3": quartiles[2],
        "maximum": ordered[-1],
    }


def timed(
    function: Callable[[], object],
    *,
    warmup: int,
    samples: int,
    minimum_sample_seconds: float,
) -> dict:
    for _ in range(warmup):
        function()
    inner = 1
    while True:
        started = time.perf_counter()
        for _ in range(inner):
            function()
        elapsed = time.perf_counter() - started
        if elapsed >= minimum_sample_seconds or inner >= 1 << 20:
            break
        inner *= max(2, int(minimum_sample_seconds / max(elapsed, 1e-9)))
        inner = min(inner, 1 << 20)
    observations = []
    with gc_disabled():
        for _ in range(samples):
            started = time.perf_counter()
            for _ in range(inner):
                function()
            observations.append((time.perf_counter() - started) / inner)
    return summary(observations, inner)


def stable_rng(seed: int, *labels: object) -> np.random.Generator:
    payload = json.dumps([int(seed), *map(str, labels)], separators=(",", ":"))
    child = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little"
    )
    return np.random.default_rng(child)


def random_band(
    rng: np.random.Generator, n: int, kl: int, ku: int, batch: int | None = None
) -> np.ndarray:
    shape = (n, kl + ku + 1) if batch is None else (n, kl + ku + 1, batch)
    band = rng.standard_normal(shape)
    if batch is None:
        band[:, kl] += kl + ku + 4.0
    else:
        band[:, kl, :] += kl + ku + 4.0
    return np.ascontiguousarray(band)


def normalized_residual(
    band: np.ndarray, kl: int, ku: int, x: np.ndarray, rhs: np.ndarray
) -> float:
    residual = band_matvec(band, kl, ku, x) - rhs
    return float(np.linalg.norm(residual) / np.linalg.norm(rhs))


def load_state(
    output: Path, mode: str, config_hash: str, resume: bool
) -> dict:
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}; pass --resume")
        state = json.loads(output.read_text(encoding="utf-8"))
        if state["mode"] != mode or state["config_sha256"] != config_hash:
            raise ValueError("checkpoint mode/config does not match this run")
        state["status"] = "RUNNING"
        state["resumed_utc"] = utc_now()
        return state
    return {
        "schema_version": 1,
        "driver": "rusty_scaling",
        "mode": mode,
        "config_sha256": config_hash,
        "status": "RUNNING",
        "provenance": provenance(mode),
        "results": [],
    }


def completed_keys(state: dict) -> set[str]:
    return {str(row["case_key"]) for row in state["results"]}


def record(output: Path, state: dict, row: dict) -> None:
    state["results"].append(row)
    state["updated_utc"] = utc_now()
    atomic_json(output, state)


def stop_if_requested(output: Path, state: dict) -> None:
    if STOP_REQUESTED:
        state.update(status="CHECKPOINTED", stopped_utc=utc_now())
        atomic_json(output, state)
        raise SystemExit(75)


def timing_arguments(config: dict) -> dict:
    timing = config["timing"]
    return {
        "warmup": int(timing["warmup"]),
        "samples": int(timing["samples"]),
        "minimum_sample_seconds": float(timing["minimum_sample_seconds"]),
    }


def _factor_and_solve(
    band: np.ndarray,
    rhs: np.ndarray,
    kl: int,
    ku: int,
    householder: bool,
    threads: int,
) -> np.ndarray:
    return BandQR(
        band,
        kl,
        ku,
        householder=householder,
        nthreads=threads,
    ).solve(rhs)


def run_cpu(config: dict, output: Path, state: dict) -> None:
    import scipy
    import scipy.linalg as scipy_linalg

    state["provenance"]["scipy"] = scipy.__version__
    done = completed_keys(state)
    timing = timing_arguments(config)
    cpu = config["cpu"]
    kl, ku = int(cpu["kl"]), int(cpu["ku"])
    seed = int(config["seed"])

    for n in map(int, cpu["sizes"]):
        rng = stable_rng(seed, "cpu-size", n, kl, ku)
        band = random_band(rng, n, kl, ku)
        rhs = np.ascontiguousarray(rng.standard_normal(n))
        scipy_band = to_scipy_band(band, kl, ku)
        for method in cpu["methods"]:
            key = f"cpu-size/n={n}/method={method}"
            if key in done:
                continue
            if method == "givens":
                function = partial(
                    _factor_and_solve, band, rhs, kl, ku, False, 1
                )
                solution = function()
            elif method == "householder":
                function = partial(
                    _factor_and_solve, band, rhs, kl, ku, True, 1
                )
                solution = function()
            elif method == "lapack_lu":
                function = partial(
                    scipy_linalg.solve_banded,
                    (kl, ku),
                    scipy_band,
                    rhs,
                    check_finite=False,
                )
                solution = function()
            else:
                raise ValueError(f"unknown CPU method {method!r}")
            row = {
                "case_key": key,
                "family": "cpu_size_scaling",
                "n": n,
                "kl": kl,
                "ku": ku,
                "method": method,
                "threads": 1,
                "timing": timed(function, **timing),
                "normalized_residual": normalized_residual(
                    band, kl, ku, solution, rhs
                ),
            }
            record(output, state, row)
            done.add(key)
            stop_if_requested(output, state)

    thread_case = cpu["thread_case"]
    n, kl, ku = (int(thread_case[name]) for name in ("n", "kl", "ku"))
    rng = stable_rng(seed, "cpu-threads", n, kl, ku)
    band = random_band(rng, n, kl, ku)
    rhs = np.ascontiguousarray(rng.standard_normal(n))
    for threads in map(int, thread_case["threads"]):
        key = f"cpu-threads/n={n}/kl={kl}/threads={threads}"
        if key in done:
            continue
        function = partial(
            _factor_and_solve, band, rhs, kl, ku, False, threads
        )
        solution = function()
        row = {
            "case_key": key,
            "family": "cpu_thread_scaling",
            "n": n,
            "kl": kl,
            "ku": ku,
            "method": "givens",
            "threads": threads,
            "timing": timed(function, **timing),
            "normalized_residual": normalized_residual(
                band, kl, ku, solution, rhs
            ),
        }
        record(output, state, row)
        done.add(key)
        stop_if_requested(output, state)

    reuse = cpu["reuse_case"]
    n, kl, ku = (int(reuse[name]) for name in ("n", "kl", "ku"))
    rng = stable_rng(seed, "cpu-reuse", n, kl, ku)
    band = random_band(rng, n, kl, ku)
    factor = BandQR(band, kl, ku, nthreads=1)
    for nrhs in map(int, reuse["right_hand_sides"]):
        key = f"cpu-reuse/n={n}/nrhs={nrhs}"
        if key in done:
            continue
        rhs = np.ascontiguousarray(rng.standard_normal((n, nrhs)))
        function = partial(factor.solve, rhs)
        solution = function()
        residual = max(
            normalized_residual(
                band,
                kl,
                ku,
                np.ascontiguousarray(solution[:, index]),
                np.ascontiguousarray(rhs[:, index]),
            )
            for index in range(nrhs)
        )
        row = {
            "case_key": key,
            "family": "cpu_factor_reuse",
            "n": n,
            "kl": kl,
            "ku": ku,
            "method": "givens_reuse",
            "nrhs": nrhs,
            "timing": timed(function, **timing),
            "normalized_residual": residual,
        }
        record(output, state, row)
        done.add(key)
        stop_if_requested(output, state)


def cuda_timed(function: Callable[[], object], samples: int, warmup: int) -> dict:
    import cupy as cp

    for _ in range(warmup):
        function()
    cp.cuda.get_current_stream().synchronize()
    observations = []
    for _ in range(samples):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        function()
        end.record()
        end.synchronize()
        observations.append(float(cp.cuda.get_elapsed_time(start, end)) / 1000.0)
    return summary(observations, 1)


def cuda_properties() -> dict:
    import cupy as cp

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "cupy": cp.__version__,
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "device_id": int(device.id),
        "device_name": str(name),
        "compute_capability": str(device.compute_capability),
        "memory_free_bytes_at_start": int(free),
        "memory_total_bytes": int(total),
    }


def run_cuda(config: dict, output: Path, state: dict) -> None:
    import cupy as cp

    from bandedqr import cuda
    from bandedqr.batched import BatchedBandQR as NumpyBatched

    if not cuda.is_available():
        raise cuda.CUDAUnavailable("CuPy cannot execute on a CUDA device")
    state["provenance"]["cuda"] = cuda_properties()
    done = completed_keys(state)
    timing = timing_arguments(config)
    cuda_config = config["cuda"]
    n, kl, ku = (
        int(cuda_config[name]) for name in ("n", "kl", "ku")
    )
    dtype = np.dtype(cuda_config["dtype"])
    refine = int(cuda_config["refinement_steps"])
    seed = int(config["seed"])
    free_memory, _ = cp.cuda.runtime.memGetInfo()

    for batch in map(int, cuda_config["batches"]):
        key = f"cuda-batch/n={n}/batch={batch}"
        if key in done:
            continue
        scalar_bytes = np.dtype(np.float32).itemsize
        estimated = (
            n * (2 * kl + ku + 2) * batch * scalar_bytes
            + n * kl * batch * 2 * scalar_bytes
            + 5 * n * batch * scalar_bytes
        )
        if estimated > float(cuda_config["max_device_memory_fraction"]) * free_memory:
            raise MemoryError(
                f"case {key} estimates {estimated} device bytes with only "
                f"{free_memory} free"
            )
        rng = stable_rng(seed, "cuda-batch", n, kl, ku, batch)
        band = random_band(rng, n, kl, ku, batch).astype(dtype)
        rhs = np.ascontiguousarray(rng.standard_normal((n, batch)), dtype=dtype)

        compile_started = time.perf_counter()
        factor = cuda.BatchedBandQR(band, kl, ku)
        compile_and_first_factor = time.perf_counter() - compile_started
        rhs_device = cp.asarray(rhs, dtype=cp.float32)
        factor.solve_fp32_device(rhs_device)
        cp.cuda.get_current_stream().synchronize()

        resident = cuda_timed(
            lambda: factor.solve_fp32_device(rhs_device),
            samples=int(timing["samples"]),
            warmup=int(timing["warmup"]),
        )
        host_reuse = timed(
            lambda: factor.solve_fp32(rhs),
            **timing,
        )
        full_raw = timed(
            lambda: cuda.BatchedBandQR(band, kl, ku).solve_fp32(rhs),
            **timing,
        )
        full_refined = timed(
            lambda: cuda.BatchedBandQR(band, kl, ku).solve(
                rhs, refine=refine
            ),
            **timing,
        )
        cpu_full = timed(
            lambda: NumpyBatched(band, kl, ku).solve(rhs),
            **timing,
        )

        raw = factor.solve_fp32(rhs)
        raw_residual = factor.residual(raw, rhs)
        raw_relative = float(
            np.max(
                np.linalg.norm(raw_residual, axis=0)
                / np.maximum(np.linalg.norm(rhs, axis=0), np.finfo(float).tiny)
            )
        )
        refined, info = factor.solve(
            rhs, refine=refine, return_info=True
        )
        refined_residual = factor.residual(refined, rhs)
        refined_relative = float(
            np.max(
                np.linalg.norm(refined_residual, axis=0)
                / np.maximum(np.linalg.norm(rhs, axis=0), np.finfo(float).tiny)
            )
        )
        row = {
            "case_key": key,
            "family": "cuda_batch_scaling",
            "n": n,
            "kl": kl,
            "ku": ku,
            "batch": batch,
            "dtype": dtype.name,
            "compile_and_first_factor_seconds": compile_and_first_factor,
            "estimated_peak_device_bytes": int(estimated),
            "timing": {
                "resident_reuse_solve": resident,
                "host_transfer_reuse_solve": host_reuse,
                "transfer_inclusive_factor_solve_raw": full_raw,
                "transfer_inclusive_factor_solve_refined": full_refined,
                "cpu_numpy_factor_solve": cpu_full,
            },
            "accuracy": {
                "raw_max_normalized_residual": raw_relative,
                "refined_max_normalized_residual": refined_relative,
                "refinement": info,
            },
        }
        record(output, state, row)
        done.add(key)
        del raw, refined
        cp.get_default_memory_pool().free_all_blocks()
        stop_if_requested(output, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config_hash = sha256_bytes(config_bytes)
    config = json.loads(config_bytes)
    state = load_state(args.output, args.mode, config_hash, args.resume)
    atomic_json(args.output, state)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    try:
        if args.mode == "cpu":
            run_cpu(config, args.output, state)
        else:
            run_cuda(config, args.output, state)
        state.update(status="PASS", completed_utc=utc_now())
        atomic_json(args.output, state)
        return 0
    except SystemExit:
        raise
    except BaseException as error:
        state.update(
            status="FAIL",
            failed_utc=utc_now(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        atomic_json(args.output, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
