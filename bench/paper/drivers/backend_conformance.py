#!/usr/bin/env python3
"""Deterministic, backend-neutral batched correctness campaign.

This driver intentionally records no timings.  It compares a concrete backend
with the NumPy batched reference on byte-identical inputs and writes enough
detail to distinguish raw accelerator arithmetic from refined results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Callable

import numpy as np

from bandedqr import RideQR
from bandedqr._band import band_matvec_batched, band_to_dense
from bandedqr.batched import BatchedBandQR as NumpyBatched


ArrayApply = Callable[[np.ndarray], np.ndarray]


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def arrays_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def stable_rng(seed: int, *labels: object) -> np.random.Generator:
    payload = json.dumps([int(seed), *map(str, labels)], separators=(",", ":"))
    child_seed = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little"
    )
    return np.random.default_rng(child_seed)


def random_array(
    rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    value = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        value = (value + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    return np.ascontiguousarray(value, dtype=dtype)


def make_batch(
    rng: np.random.Generator,
    n: int,
    kl: int,
    ku: int,
    batch: int,
    dtype: np.dtype,
) -> np.ndarray:
    ab = np.zeros((n, kl + ku + 1, batch), dtype=dtype)
    for offset in range(-kl, ku + 1):
        row0 = max(0, -offset)
        row1 = min(n, n - offset)
        ab[row0:row1, offset + kl, :] = random_array(
            rng, (row1 - row0, batch), dtype
        )
    ab[:, kl, :] += kl + ku + 4.0
    return np.ascontiguousarray(ab)


def make_conditioned_batch(
    rng: np.random.Generator,
    n: int,
    kl: int,
    ku: int,
    batch: int,
    dtype: np.dtype,
    target_condition: float,
) -> np.ndarray:
    """Construct nontrivial bands with a controlled diagonal scale range."""
    ab = np.zeros((n, kl + ku + 1, batch), dtype=dtype)
    base_diagonal = np.geomspace(1.0, 1.0 / target_condition, n)
    jitter = np.exp(0.01 * rng.standard_normal((n, batch)))
    diagonal = base_diagonal[:, None] * jitter
    ab[:, kl, :] = diagonal
    coupling = 0.025
    for offset in range(-kl, ku + 1):
        if offset == 0:
            continue
        row0 = max(0, -offset)
        row1 = min(n, n - offset)
        local_scale = np.sqrt(
            diagonal[row0:row1] * diagonal[row0 + offset : row1 + offset]
        )
        values = random_array(rng, local_scale.shape, dtype)
        ab[row0:row1, offset + kl, :] = coupling * local_scale * values
    return np.ascontiguousarray(ab)


def rhs_shape(n: int, nrhs: int, batch: int) -> tuple[int, ...]:
    return (n, batch) if nrhs == 1 else (n, nrhs, batch)


def as_three(array: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(array)
    return (array[:, None, :], True) if array.ndim == 2 else (array, False)


def band_apply(ab: np.ndarray, kl: int, ku: int, x: np.ndarray) -> np.ndarray:
    X, squeeze = as_three(x)
    result = np.empty_like(X, dtype=np.result_type(ab.dtype, X.dtype))
    for rhs in range(X.shape[1]):
        result[:, rhs, :] = band_matvec_batched(ab, kl, ku, X[:, rhs, :])
    return result[:, 0, :] if squeeze else result


def ride_apply(
    bulk: np.ndarray,
    kl: int,
    ku: int,
    dense_rows: np.ndarray,
    rows: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    X, squeeze = as_three(x)
    result = np.empty_like(X, dtype=np.result_type(bulk.dtype, rows.dtype, X.dtype))
    for rhs in range(X.shape[1]):
        result[:, rhs, :] = band_matvec_batched(
            bulk, kl, ku, X[:, rhs, :]
        )
        result[dense_rows, rhs, :] = np.einsum(
            "rn,nk->rk", rows, X[:, rhs, :], optimize=True
        )
    return result[:, 0, :] if squeeze else result


def max_vector_relative(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator = np.asarray(numerator).reshape(numerator.shape[0], -1)
    denominator = np.asarray(denominator).reshape(denominator.shape[0], -1)
    top = np.linalg.norm(numerator, axis=0)
    bottom = np.maximum(
        np.linalg.norm(denominator, axis=0), np.finfo(float).tiny
    )
    return float(np.max(top / bottom))


def solution_metrics(
    solution: np.ndarray,
    reference: np.ndarray,
    rhs: np.ndarray,
    apply: ArrayApply,
    truth: np.ndarray | None = None,
) -> dict:
    residual = apply(solution) - rhs
    result = {
        "max_relative_forward_error": max_vector_relative(
            solution - reference, reference
        ),
        "max_normalized_residual": max_vector_relative(residual, rhs),
        "finite": bool(np.isfinite(solution).all() and np.isfinite(residual).all()),
    }
    if truth is not None:
        result["max_relative_truth_error"] = max_vector_relative(
            solution - truth, truth
        )
    return result


def backend_metadata(selector: str) -> dict:
    if selector == "numpy":
        return {
            "selector": "numpy",
            "implementation_id": "cpu-numpy-batched",
            "arithmetic": "float64/complex128",
            "numpy": np.__version__,
        }
    if selector == "cuda":
        from bandedqr import cuda

        if not cuda.is_available():
            raise cuda.CUDAUnavailable(
                "CUDA conformance requested, but a trivial evaluated CuPy "
                "operation cannot run in this process"
            )
        import cupy as cp

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return {
            "selector": "cuda",
            "implementation_id": "cuda-cupy",
            "arithmetic": "float32/complex64 with host float64 refinement",
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device_id": int(device.id),
            "device_name": str(name),
            "execution_probe": True,
            "capabilities": [
                "ordinary",
                "shifted",
                "multi_rhs",
                "factor_reuse",
                "conditioned",
            ],
        }
    if selector != "metal":
        raise ValueError(
            f"unsupported selector {selector!r}; expected numpy, metal, or cuda"
        )

    from bandedqr import gpu

    if not gpu.is_available():
        raise gpu.GPUUnavailable(
            "Metal conformance requested, but a trivial evaluated MLX operation "
            "cannot run in this process"
        )
    import mlx.core as mx

    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = None
    try:
        default_device = str(mx.default_device())
    except Exception as error:
        default_device = f"unavailable: {error}"
    return {
        "selector": "metal",
        "implementation_id": "metal-mlx",
        "arithmetic": "float32/complex64 with host float64 refinement",
        "numpy": np.__version__,
        "mlx": mlx_version,
        "default_device": default_device,
        "execution_probe": True,
    }


def make_factor(
    selector: str,
    ab: np.ndarray,
    kl: int,
    ku: int,
    shifts: np.ndarray | None = None,
):
    if selector == "numpy":
        return NumpyBatched(ab, kl, ku, shifts=shifts)
    if selector == "cuda":
        from bandedqr import cuda

        return cuda.BatchedBandQR(ab, kl, ku, shifts=shifts)
    from bandedqr import gpu

    return gpu.BatchedBandQR(ab, kl, ku, shifts=shifts)


def solve_pair(
    selector: str, factor, rhs: np.ndarray, refine: int, refine_tol: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    if selector == "numpy":
        solution = factor.solve(rhs)
        return solution, solution.copy(), {
            "residual_history": [],
            "refine": 0,
            "note": "reference backend already operates in input precision",
        }
    raw = factor.solve_fp32(rhs)
    refined, info = factor.solve(
        rhs, refine=refine, tol=refine_tol, return_info=True
    )
    return raw, refined, info


def evaluate_solve(
    selector: str,
    factor,
    reference_factor,
    rhs: np.ndarray,
    apply: ArrayApply,
    refine: int,
    refine_tol: float,
    tolerances: dict,
    truth: np.ndarray | None = None,
) -> dict:
    reference = reference_factor.solve(rhs)
    raw, refined, info = solve_pair(selector, factor, rhs, refine, refine_tol)
    raw_metrics = solution_metrics(raw, reference, rhs, apply, truth)
    refined_metrics = solution_metrics(refined, reference, rhs, apply, truth)

    gates = {
        "raw_finite": raw_metrics["finite"],
        "raw_forward": raw_metrics["max_relative_forward_error"]
        <= tolerances["raw_max_relative_forward_error"],
        "raw_residual": raw_metrics["max_normalized_residual"]
        <= tolerances["raw_max_normalized_residual"],
        "refined_finite": refined_metrics["finite"],
        "refined_forward": refined_metrics["max_relative_forward_error"]
        <= tolerances["refined_max_relative_forward_error"],
        "refined_residual": refined_metrics["max_normalized_residual"]
        <= tolerances["refined_max_normalized_residual"],
    }
    if truth is not None:
        gates["refined_truth"] = (
            refined_metrics["max_relative_truth_error"]
            <= tolerances["refined_max_relative_forward_error"]
        )
    raw_error = raw_metrics["max_relative_forward_error"]
    refined_error = refined_metrics["max_relative_forward_error"]
    improvement = (
        float(raw_error / max(refined_error, np.finfo(float).tiny))
        if selector in {"metal", "cuda"}
        else 1.0
    )
    return {
        "raw": raw_metrics,
        "refined": refined_metrics,
        "refinement": info,
        "forward_error_improvement_factor": improvement,
        "gates": {name: bool(value) for name, value in gates.items()},
        "pass": bool(all(gates.values())),
    }


class CpuRideBatch:
    """Factor a batch with one CPU RideQR object per system."""

    def __init__(
        self,
        bulk: np.ndarray,
        kl: int,
        ku: int,
        dense_rows: np.ndarray,
        rows: np.ndarray,
    ):
        self.factors = []
        for system in range(bulk.shape[2]):
            matrix = band_to_dense(
                np.ascontiguousarray(bulk[:, :, system]), kl, ku
            )
            matrix[dense_rows, :] = rows
            self.factors.append(RideQR(matrix, kl, ku, dense_rows))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        B, squeeze = as_three(rhs)
        result = np.empty_like(B)
        for system, factor in enumerate(self.factors):
            result[:, :, system] = factor.solve(
                np.ascontiguousarray(B[:, :, system])
            )
        return result[:, 0, :] if squeeze else result


def case_base(case: dict, family: str, dtype_name: str) -> dict:
    return {
        "family": family,
        "case_id": case["id"],
        "dtype": dtype_name,
        "n": int(case["n"]),
        "kl": int(case["kl"]),
        "ku": int(case["ku"]),
        "batch": int(case["batch"]),
        "nrhs": int(case["nrhs"]),
        "tolerance_profile": case["tolerance_profile"],
    }


def run(config: dict, selector: str, config_hash: str) -> dict:
    if config.get("performance_measurement") is not False:
        raise ValueError("conformance config must set performance_measurement=false")
    backend = backend_metadata(selector)
    seed = int(config["seed"])
    refine_tol = float(config["refinement_tolerance"])
    checks: list[dict] = []

    for case in config["ordinary_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            rng = stable_rng(seed, "ordinary", case["id"], dtype_name)
            ab = make_batch(
                rng, case["n"], case["kl"], case["ku"], case["batch"], dtype
            )
            rhs = random_array(
                rng, rhs_shape(case["n"], case["nrhs"], case["batch"]), dtype
            )
            factor = make_factor(selector, ab, case["kl"], case["ku"])
            reference = NumpyBatched(ab, case["kl"], case["ku"])
            result = evaluate_solve(
                selector,
                factor,
                reference,
                rhs,
                lambda x, ab=ab, case=case: band_apply(
                    ab, case["kl"], case["ku"], x
                ),
                int(case["refine"]),
                refine_tol,
                config["tolerances"][case["tolerance_profile"]],
            )
            checks.append(
                {
                    **case_base(case, "ordinary", dtype_name),
                    "matrix_sha256": arrays_sha256(ab),
                    "rhs_sha256": arrays_sha256(rhs),
                    **result,
                }
            )

    for case in config["shifted_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            rng = stable_rng(seed, "shifted", case["id"], dtype_name)
            base = make_batch(
                rng, case["n"], case["kl"], case["ku"], 1, dtype
            )[:, :, 0]
            shifts = np.geomspace(
                float(case["shift_min"]), float(case["shift_max"]), case["batch"]
            ).astype(np.float64)
            full = np.repeat(base[:, :, None], case["batch"], axis=2)
            full[:, case["kl"], :] += shifts[None, :]
            rhs = random_array(
                rng, rhs_shape(case["n"], case["nrhs"], case["batch"]), dtype
            )
            factor = make_factor(
                selector, base, case["kl"], case["ku"], shifts=shifts
            )
            reference = NumpyBatched(
                base, case["kl"], case["ku"], shifts=shifts
            )
            result = evaluate_solve(
                selector,
                factor,
                reference,
                rhs,
                lambda x, full=full, case=case: band_apply(
                    full, case["kl"], case["ku"], x
                ),
                int(case["refine"]),
                refine_tol,
                config["tolerances"][case["tolerance_profile"]],
            )
            checks.append(
                {
                    **case_base(case, "shifted", dtype_name),
                    "matrix_sha256": arrays_sha256(base, shifts),
                    "materialized_matrix_sha256": arrays_sha256(full),
                    "rhs_sha256": arrays_sha256(rhs),
                    "shift_min": float(shifts.min()),
                    "shift_max": float(shifts.max()),
                    **result,
                }
            )

    for case in config["reuse_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            rng = stable_rng(seed, "reuse", case["id"], dtype_name)
            ab = make_batch(
                rng, case["n"], case["kl"], case["ku"], case["batch"], dtype
            )
            factor = make_factor(selector, ab, case["kl"], case["ku"])
            reference = NumpyBatched(ab, case["kl"], case["ku"])
            solves = []
            for solve_index in range(int(case["solves"])):
                rhs = random_array(
                    rng,
                    rhs_shape(case["n"], case["nrhs"], case["batch"]),
                    dtype,
                )
                result = evaluate_solve(
                    selector,
                    factor,
                    reference,
                    rhs,
                    lambda x, ab=ab, case=case: band_apply(
                        ab, case["kl"], case["ku"], x
                    ),
                    int(case["refine"]),
                    refine_tol,
                    config["tolerances"][case["tolerance_profile"]],
                )
                solves.append(
                    {
                        "solve_index": solve_index,
                        "rhs_sha256": arrays_sha256(rhs),
                        **result,
                    }
                )
            checks.append(
                {
                    **case_base(case, "factor_reuse", dtype_name),
                    "matrix_sha256": arrays_sha256(ab),
                    "factorizations": 1,
                    "solve_count": len(solves),
                    "solves": solves,
                    "pass": bool(solves and all(item["pass"] for item in solves)),
                }
            )

    for case in config["conditioned_cases"]:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            for target in case["target_conditions"]:
                rng = stable_rng(
                    seed, "conditioned", case["id"], dtype_name, target
                )
                ab = make_conditioned_batch(
                    rng,
                    case["n"],
                    case["kl"],
                    case["ku"],
                    case["batch"],
                    dtype,
                    float(target),
                )
                truth = random_array(
                    rng,
                    rhs_shape(case["n"], case["nrhs"], case["batch"]),
                    dtype,
                )
                rhs = band_apply(ab, case["kl"], case["ku"], truth)
                sample_count = min(
                    int(case["condition_samples"]), int(case["batch"])
                )
                sampled_conditions = [
                    float(
                        np.linalg.cond(
                            band_to_dense(
                                np.ascontiguousarray(ab[:, :, system]),
                                case["kl"],
                                case["ku"],
                            )
                        )
                    )
                    for system in range(sample_count)
                ]
                factor = make_factor(selector, ab, case["kl"], case["ku"])
                reference = NumpyBatched(ab, case["kl"], case["ku"])
                result = evaluate_solve(
                    selector,
                    factor,
                    reference,
                    rhs,
                    lambda x, ab=ab, case=case: band_apply(
                        ab, case["kl"], case["ku"], x
                    ),
                    int(case["refine"]),
                    refine_tol,
                    config["tolerances"][case["tolerance_profile"]],
                    truth=truth,
                )
                check = {
                    **case_base(case, "conditioned", dtype_name),
                    "case_id": f"{case['id']}-target-{float(target):.0e}",
                    "target_condition": float(target),
                    "sampled_condition_2": sampled_conditions,
                    "sampled_condition_2_min": float(min(sampled_conditions)),
                    "sampled_condition_2_max": float(max(sampled_conditions)),
                    "matrix_sha256": arrays_sha256(ab),
                    "truth_sha256": arrays_sha256(truth),
                    "rhs_sha256": arrays_sha256(rhs),
                    **result,
                }
                checks.append(check)

    skipped_families = []
    ride_cases = config["ride_cases"]
    if selector == "cuda":
        skipped_families.append(
            {
                "family": "ride_dense_rows",
                "reason": "cuda-cupy v1 implements the common purely banded contract; "
                "the low-rank dense-row kernel is not yet implemented",
                "configured_cases": len(ride_cases),
            }
        )
        ride_cases = []
    for case in ride_cases:
        for dtype_name in case["dtypes"]:
            dtype = np.dtype(dtype_name)
            rng = stable_rng(seed, "ride", case["id"], dtype_name)
            bulk = make_batch(
                rng, case["n"], case["kl"], case["ku"], case["batch"], dtype
            )
            dense_rows = np.arange(int(case["dense_rows"]), dtype=int)
            rows = 0.03 * random_array(
                rng, (dense_rows.size, case["n"]), dtype
            )
            rows[np.arange(dense_rows.size), dense_rows] += (
                case["kl"] + case["ku"] + 4.0
            )
            reference = CpuRideBatch(
                bulk, case["kl"], case["ku"], dense_rows, rows
            )
            if selector == "numpy":
                factor = reference
            else:
                from bandedqr import gpu

                factor = gpu.BatchedRideQR.from_dense_rows(
                    bulk, case["kl"], case["ku"], dense_rows, rows
                )
            solves = []
            for solve_index in range(int(case["solves"])):
                rhs = random_array(
                    rng,
                    rhs_shape(case["n"], case["nrhs"], case["batch"]),
                    dtype,
                )
                result = evaluate_solve(
                    selector,
                    factor,
                    reference,
                    rhs,
                    lambda x, bulk=bulk, case=case, dense_rows=dense_rows,
                    rows=rows: ride_apply(
                        bulk,
                        case["kl"],
                        case["ku"],
                        dense_rows,
                        rows,
                        x,
                    ),
                    int(case["refine"]),
                    refine_tol,
                    config["tolerances"][case["tolerance_profile"]],
                )
                solves.append(
                    {
                        "solve_index": solve_index,
                        "rhs_sha256": arrays_sha256(rhs),
                        **result,
                    }
                )
            checks.append(
                {
                    **case_base(case, "ride_dense_rows", dtype_name),
                    "matrix_sha256": arrays_sha256(bulk, dense_rows, rows),
                    "dense_rows": dense_rows.tolist(),
                    "factorizations": (
                        int(case["batch"]) if selector == "numpy" else 1
                    ),
                    "solve_count": len(solves),
                    "solves": solves,
                    "pass": bool(solves and all(item["pass"] for item in solves)),
                }
            )

    passed = sum(bool(item["pass"]) for item in checks)
    return {
        "schema_version": 1,
        "driver": "backend_conformance",
        "campaign_kind": "correctness",
        "performance_measurement": False,
        "config_sha256": config_hash,
        "python": sys.version,
        "backend": backend,
        "tolerances": config["tolerances"],
        "checks": checks,
        "skipped_families": skipped_families,
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
        },
        "pass": bool(checks and passed == len(checks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["numpy", "metal", "cuda"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config_hash = sha256_bytes(config_bytes)
    try:
        config = json.loads(config_bytes)
        result = run(config, args.backend, config_hash)
    except BaseException as error:
        result = {
            "schema_version": 1,
            "driver": "backend_conformance",
            "campaign_kind": "correctness",
            "performance_measurement": False,
            "config_sha256": config_hash,
            "backend": {"selector": args.backend},
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "pass": False,
        }
    atomic_json(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
