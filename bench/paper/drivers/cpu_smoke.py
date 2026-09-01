#!/usr/bin/env python3
"""Small deterministic CPU and portable-batched correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from bandqr import BandQR, RideTauQR, band_matvec, band_to_dense, solve_batched
from bandqr._band import band_matvec_batched


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


def make_band(rng, n: int, kl: int, ku: int, dtype: np.dtype) -> np.ndarray:
    complex_input = np.issubdtype(dtype, np.complexfloating)
    ab = np.zeros((n, kl + ku + 1), dtype=dtype)
    for row in range(n):
        lo = max(0, row - kl)
        hi = min(n, row + ku + 1)
        values = rng.standard_normal(hi - lo)
        if complex_input:
            values = values + 1j * rng.standard_normal(hi - lo)
        ab[row, lo - row + kl : hi - row + kl] = values
    ab[:, kl] += kl + ku + 4
    return ab


def random_rhs(rng, shape, dtype: np.dtype) -> np.ndarray:
    rhs = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        rhs = rhs + 1j * rng.standard_normal(shape)
    return np.asarray(rhs, dtype=dtype)


def normalized_residual(residual: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(residual) / np.linalg.norm(rhs))


def matrix_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def run(config: dict) -> dict:
    rng = np.random.default_rng(int(config["seed"]))
    tolerance = float(config["tolerances"]["normalized_residual"])
    batch_tolerance = float(
        config["tolerances"]["batched_normalized_residual"]
    )
    checks = []

    for case in config["banded_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            ab = make_band(rng, case["n"], case["kl"], case["ku"], dtype)
            rhs = random_rhs(rng, case["n"], dtype)
            serial = BandQR(ab, case["kl"], case["ku"], nthreads=1)
            threaded = BandQR(
                ab,
                case["kl"],
                case["ku"],
                nthreads=min(int(case["threads"][-1]), os.cpu_count() or 1),
            )
            householder = BandQR(
                ab, case["kl"], case["ku"], nthreads=1, householder=True
            )
            serial_x = serial.solve(rhs)
            threaded_x = threaded.solve(rhs)
            householder_x = householder.solve(rhs)
            serial_error = normalized_residual(
                band_matvec(ab, case["kl"], case["ku"], serial_x) - rhs,
                rhs,
            )
            threaded_error = normalized_residual(
                band_matvec(ab, case["kl"], case["ku"], threaded_x) - rhs,
                rhs,
            )
            householder_error = normalized_residual(
                band_matvec(ab, case["kl"], case["ku"], householder_x) - rhs,
                rhs,
            )
            identical_r = bool(
                np.array_equal(serial.R_band(), threaded.R_band())
            )
            passed = bool(
                identical_r
                and max(serial_error, threaded_error, householder_error)
                <= tolerance
            )
            checks.append(
                {
                    "check": "banded_factorizations",
                    "dtype": dtype_name,
                    "matrix_sha256": matrix_hash(ab),
                    "serial_threaded_r_bit_identical": identical_r,
                    "serial_normalized_residual": serial_error,
                    "threaded_normalized_residual": threaded_error,
                    "householder_normalized_residual": householder_error,
                    "threads_used": threaded.nthreads,
                    "pass": passed,
                }
            )

    for case in config["batched_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            systems = [
                make_band(rng, case["n"], case["kl"], case["ku"], dtype)
                for _ in range(case["batch"])
            ]
            ab = np.stack(systems, axis=2)
            rhs = random_rhs(
                rng, (case["n"], case["nrhs"], case["batch"]), dtype
            )
            solution = solve_batched(
                ab,
                rhs,
                case["kl"],
                case["ku"],
                backend="numpy",
            )
            residuals = []
            for column in range(case["nrhs"]):
                residuals.append(
                    band_matvec_batched(
                        ab,
                        case["kl"],
                        case["ku"],
                        solution[:, column, :],
                    )
                    - rhs[:, column, :]
                )
            error = normalized_residual(np.stack(residuals), rhs)
            checks.append(
                {
                    "check": "portable_batched_backend",
                    "backend": "cpu-numpy-batched",
                    "dtype": dtype_name,
                    "matrix_sha256": matrix_hash(ab),
                    "normalized_residual": error,
                    "pass": bool(error <= batch_tolerance),
                }
            )

    for case in config["almost_banded_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            bulk = make_band(rng, case["n"], case["kl"], case["ku"], dtype)
            matrix = band_to_dense(
                bulk, case["kl"], case["ku"], case["n"]
            )
            dense_rows = np.arange(case["dense_rows"])
            dense_columns = np.arange(
                case["n"] - case["dense_columns"], case["n"]
            )
            matrix[dense_rows, :] = random_rhs(
                rng, (len(dense_rows), case["n"]), dtype
            )
            matrix[:, dense_columns] = random_rhs(
                rng, (case["n"], len(dense_columns)), dtype
            )
            rhs = random_rhs(rng, case["n"], dtype)
            factor = RideTauQR(
                matrix,
                case["kl"],
                case["ku"],
                dense_rows,
                dense_columns,
            )
            solution = factor.solve(rhs)
            error = normalized_residual(matrix @ solution - rhs, rhs)
            checks.append(
                {
                    "check": "almost_banded_ride_tau",
                    "dtype": dtype_name,
                    "matrix_sha256": matrix_hash(matrix),
                    "normalized_residual": error,
                    "pass": bool(error <= tolerance),
                }
            )

    return {
        "schema_version": 1,
        "driver": "cpu_smoke",
        "python": sys.version,
        "numpy": np.__version__,
        "checks": checks,
        "pass": bool(checks and all(item["pass"] for item in checks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    result = run(config)
    atomic_json(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
