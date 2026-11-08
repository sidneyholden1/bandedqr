#!/usr/bin/env python3
"""Render accepted Rusty scaling records into a figure, tables, and summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(value)
        stream.flush()
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def load_pass(path: Path, mode: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or value.get("mode") != mode:
        raise ValueError(f"{path} is not an accepted PASS record for {mode}")
    return value


def milliseconds(value: float) -> float:
    return 1000.0 * float(value)


def concise_time(value: float) -> str:
    value = milliseconds(value)
    if value < 0.01:
        return f"{value:.4f} ms"
    if value < 0.1:
        return f"{value:.3f} ms"
    if value < 10:
        return f"{value:.2f} ms"
    return f"{value:.1f} ms"


def cpu_rows(record: dict) -> list[dict]:
    rows = [
        row for row in record["results"]
        if row["family"] == "cpu_size_scaling"
    ]
    return sorted(rows, key=lambda row: (row["n"], row["method"]))


def cuda_rows(record: dict) -> list[dict]:
    rows = [
        row for row in record["results"]
        if row["family"] == "cuda_batch_scaling"
    ]
    return sorted(rows, key=lambda row: row["batch"])


def markdown_tables(cpu: list[dict], cuda: list[dict]) -> str:
    by_size = {}
    for row in cpu:
        by_size.setdefault(row["n"], {})[row["method"]] = row
    lines = [
        "### CPU size scaling",
        "",
        "Fixed `kl = ku = 4`, one thread; median factor + solve time.",
        "",
        "| n | bandedqr Givens | bandedqr Householder | LAPACK band LU | max residual |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n, methods in sorted(by_size.items()):
        residual = max(row["normalized_residual"] for row in methods.values())
        lines.append(
            "| {n:,} | {givens} | {householder} | {lu} | {residual:.1e} |".format(
                n=n,
                givens=concise_time(methods["givens"]["timing"]["median"]),
                householder=concise_time(
                    methods["householder"]["timing"]["median"]
                ),
                lu=concise_time(methods["lapack_lu"]["timing"]["median"]),
                residual=residual,
            )
        )
    lines += [
        "",
        "### CUDA batch scaling",
        "",
        "Fixed `n = 256`, `kl = ku = 4`. CUDA refined timings include host/device "
        "transfers, factorization, solve, and mixed-precision refinement; the "
        "resident column is the raw solve with factors and RHS already on device.",
        "",
        "| batch | CPU NumPy factor + solve | CUDA refined end-to-end | speedup | CUDA resident re-solve | refined residual |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cuda:
        timing = row["timing"]
        cpu_time = timing["cpu_numpy_factor_solve"]["median"]
        cuda_time = timing["transfer_inclusive_factor_solve_refined"]["median"]
        resident = timing["resident_reuse_solve"]["median"]
        lines.append(
            "| {batch:,} | {cpu} | {cuda} | {speedup:.1f}× | {resident} | {residual:.1e} |".format(
                batch=row["batch"],
                cpu=concise_time(cpu_time),
                cuda=concise_time(cuda_time),
                speedup=cpu_time / cuda_time,
                resident=concise_time(resident),
                residual=row["accuracy"]["refined_max_normalized_residual"],
            )
        )
    return "\n".join(lines) + "\n"


def render_figure(cpu: list[dict], cuda: list[dict], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    labels = {
        "givens": ("bandedqr Givens QR", "o", "C0"),
        "householder": ("bandedqr Householder QR", "D", "C4"),
        "lapack_lu": ("LAPACK band LU", "s", "C2"),
    }
    for method, (label, marker, color) in labels.items():
        rows = [row for row in cpu if row["method"] == method]
        left.loglog(
            [row["n"] for row in rows],
            [milliseconds(row["timing"]["median"]) for row in rows],
            marker=marker,
            color=color,
            label=label,
        )
    left.set_title("CPU: fixed bandwidth")
    left.set_xlabel("system size n")
    left.set_ylabel("factor + solve [ms]")
    left.grid(True, which="both", alpha=0.2)
    left.legend(frameon=False)

    batches = [row["batch"] for row in cuda]
    right.loglog(
        batches,
        [
            milliseconds(row["timing"]["cpu_numpy_factor_solve"]["median"])
            for row in cuda
        ],
        "s-",
        color="C1",
        label="CPU NumPy, factor + solve",
    )
    right.loglog(
        batches,
        [
            milliseconds(
                row["timing"]["transfer_inclusive_factor_solve_refined"][
                    "median"
                ]
            )
            for row in cuda
        ],
        "o-",
        color="C0",
        label="CUDA, refined end-to-end",
    )
    right.loglog(
        batches,
        [
            milliseconds(row["timing"]["resident_reuse_solve"]["median"])
            for row in cuda
        ],
        "^-",
        color="C3",
        label="CUDA, resident re-solve",
    )
    right.set_title("GPU: fixed system size")
    right.set_xlabel("independent systems (batch)")
    right.set_ylabel("time [ms]")
    right.grid(True, which="both", alpha=0.2)
    right.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format=output.suffix.lstrip("."),
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "bandedqr render_scaling.py"},
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    cpu_record = load_pass(args.cpu, "cpu")
    cuda_record = load_pass(args.cuda, "cuda")
    if cpu_record["config_sha256"] != cuda_record["config_sha256"]:
        raise ValueError("CPU and CUDA records used different benchmark configs")
    cpu = cpu_rows(cpu_record)
    cuda = cuda_rows(cuda_record)
    if not cpu or not cuda:
        raise ValueError("both records must contain scaling cases")
    render_figure(cpu, cuda, args.figure)
    table = markdown_tables(cpu, cuda)
    atomic_text(args.table, table)
    atomic_json(
        args.summary,
        {
            "schema_version": 1,
            "config_sha256": cpu_record["config_sha256"],
            "source_commit": cpu_record["provenance"]["source_commit"],
            "inputs": {
                "cpu": {"path": str(args.cpu), "sha256": sha256(args.cpu)},
                "cuda": {"path": str(args.cuda), "sha256": sha256(args.cuda)},
            },
            "provenance": {
                "cpu": cpu_record["provenance"],
                "cuda": cuda_record["provenance"],
            },
            "cpu_size_scaling": cpu,
            "cuda_batch_scaling": cuda,
            "figure": str(args.figure),
            "table": str(args.table),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
