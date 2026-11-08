"""Tests for promoting accepted scaling records into public artifacts."""

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("matplotlib")


RENDERER = (
    Path(__file__).resolve().parents[1] / "bench" / "paper" / "render_scaling.py"
)
SPEC = importlib.util.spec_from_file_location("bandedqr_render_scaling", RENDERER)
rendering = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rendering)


def test_update_readme_results_replaces_only_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    figure = tmp_path / "figures" / "scaling.svg"
    summary = tmp_path / "bench" / "results" / "summary.json"
    readme.write_text(
        "before\n"
        f"{rendering.RESULTS_START}\nold\n{rendering.RESULTS_END}\n"
        "after\n"
    )

    rendering.update_readme_results(
        readme,
        figure=figure,
        summary=summary,
        table="### Timings\n\n| n | time |\n|---:|---:|\n| 1 | 2 ms |\n",
        source_commit="abc123",
    )

    result = readme.read_text()
    assert result.startswith("before\n")
    assert result.endswith("\nafter\n")
    assert "`abc123`" in result
    assert "![CPU and CUDA scaling on Rusty](figures/scaling.svg)" in result
    assert "[bench/results/summary.json](bench/results/summary.json)" in result
    assert "old" not in result


def test_update_readme_results_requires_unique_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")

    try:
        rendering.update_readme_results(
            readme,
            figure=tmp_path / "figure.svg",
            summary=tmp_path / "summary.json",
            table="table\n",
            source_commit="abc123",
        )
    except ValueError as error:
        assert "marker pair" in str(error)
    else:
        raise AssertionError("README without markers was accepted")
