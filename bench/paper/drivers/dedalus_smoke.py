#!/usr/bin/env python3
"""Assemble one real Dedalus operator and compare two solvers on it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from bandedqr import RideTauQR


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


def sparse_hash(matrix) -> str:
    csc = matrix.tocsc()
    digest = hashlib.sha256()
    for array in (csc.indptr, csc.indices, csc.data):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(str(csc.shape).encode())
    return digest.hexdigest()


def assemble_operator(size: int):
    import dedalus.public as d3

    coordinate = d3.Coordinate("x")
    distributor = d3.Distributor(coordinate, dtype=np.float64)
    basis = d3.Chebyshev(coordinate, size=size, bounds=(-1, 1))
    u = distributor.Field(name="u", bases=basis)
    tau1 = distributor.Field(name="tau1")
    tau2 = distributor.Field(name="tau2")
    forcing = distributor.Field(name="forcing", bases=basis)
    forcing["g"] = 1.0
    dx = lambda value: d3.Differentiate(value, coordinate)
    lift_basis = basis.derivative_basis(2)
    lift = lambda value, n: d3.Lift(value, lift_basis, n)
    problem = d3.LBVP(
        [u, tau1, tau2],
        namespace=locals(),
    )
    problem.add_equation(
        "dx(dx(u)) - 5*u + lift(tau1,-1) + lift(tau2,-2) = forcing"
    )
    problem.add_equation("u(x=-1) = 0")
    problem.add_equation("u(x=+1) = 0")
    solver = problem.build_solver(store_expanded_matrices=True)
    solver.solve()
    return solver, solver.subproblems[0].L_min.tocsc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)

    case = config["dedalus_case"]
    solver, matrix = assemble_operator(int(case["n"]))
    n = matrix.shape[0]
    dense_rows = np.array([n - 2, n - 1])
    dense_columns = np.array([n - 2, n - 1])
    rhs = np.random.default_rng(int(config["seed"])).standard_normal(n)

    bandedqr_factor = RideTauQR(
        matrix, 0, 4, dense_rows, dense_columns
    )
    bandedqr_solution = bandedqr_factor.solve(rhs)
    superlu_factor = solver.matsolver(matrix, solver)
    superlu_solution = superlu_factor.solve(rhs)

    bandedqr_residual = float(
        np.linalg.norm(matrix @ bandedqr_solution - rhs) / np.linalg.norm(rhs)
    )
    superlu_residual = float(
        np.linalg.norm(matrix @ superlu_solution - rhs) / np.linalg.norm(rhs)
    )
    solution_difference = float(
        np.linalg.norm(bandedqr_solution - superlu_solution)
        / np.linalg.norm(superlu_solution)
    )
    residual_tolerance = float(
        config["tolerances"]["dedalus_normalized_residual"]
    )
    difference_tolerance = float(
        config["tolerances"]["dedalus_solution_difference"]
    )
    passed = bool(
        max(bandedqr_residual, superlu_residual) <= residual_tolerance
        and solution_difference <= difference_tolerance
    )
    result = {
        "schema_version": 1,
        "driver": "dedalus_smoke",
        "dedalus_version": importlib.metadata.version("dedalus"),
        "operator": case["operator"],
        "requested_basis_size": case["n"],
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "matrix_sha256": sparse_hash(matrix),
        "bandedqr_normalized_residual": bandedqr_residual,
        "superlu_normalized_residual": superlu_residual,
        "solution_relative_difference": solution_difference,
        "pass": passed,
    }
    atomic_json(args.output, result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
