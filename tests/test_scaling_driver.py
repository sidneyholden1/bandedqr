"""Fast control-path tests for the checkpointed scaling driver."""

import importlib.util
import json
from pathlib import Path


DRIVER = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "paper"
    / "drivers"
    / "scaling.py"
)
SPEC = importlib.util.spec_from_file_location("bandedqr_scaling_driver", DRIVER)
scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(scaling)


def tiny_config():
    return {
        "seed": 1,
        "timing": {
            "warmup": 0,
            "samples": 2,
            "minimum_sample_seconds": 0.00001,
        },
        "cpu": {
            "kl": 1,
            "ku": 1,
            "sizes": [24],
            "methods": ["givens", "householder", "lapack_lu"],
            "thread_case": {
                "n": 24,
                "kl": 2,
                "ku": 2,
                "threads": [1, 2],
            },
            "reuse_case": {
                "n": 24,
                "kl": 1,
                "ku": 1,
                "right_hand_sides": [1, 3],
            },
        },
    }


def test_cpu_driver_checkpoints_and_resumes(tmp_path):
    output = tmp_path / "cpu.json"
    state = scaling.load_state(output, "cpu", "config-hash", resume=False)
    scaling.run_cpu(tiny_config(), output, state)
    state.update(status="PASS")
    scaling.atomic_json(output, state)

    recorded = json.loads(output.read_text())
    assert recorded["status"] == "PASS"
    assert len(recorded["results"]) == 7
    assert all(row["timing"]["median"] > 0 for row in recorded["results"])
    assert max(row["normalized_residual"] for row in recorded["results"]) < 1e-12

    resumed = scaling.load_state(output, "cpu", "config-hash", resume=True)
    scaling.run_cpu(tiny_config(), output, resumed)
    assert len(resumed["results"]) == 7


def test_checkpoint_refuses_config_change(tmp_path):
    output = tmp_path / "cpu.json"
    state = scaling.load_state(output, "cpu", "first", resume=False)
    scaling.atomic_json(output, state)
    try:
        scaling.load_state(output, "cpu", "second", resume=True)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("changed config was accepted")


def test_provenance_prefers_immutable_bundle_identity(monkeypatch):
    monkeypatch.setenv("BANDEDQR_SOURCE_COMMIT", "abc123")
    monkeypatch.setenv("BANDEDQR_SOURCE_DIRTY", "0")
    monkeypatch.setenv("BANDEDQR_BUNDLE_MANIFEST_SHA256", "def456")
    monkeypatch.setattr(scaling, "git_value", lambda *args: "wrong-checkout")

    record = scaling.provenance("cpu")

    assert record["source_commit"] == "abc123"
    assert record["source_dirty"] is False
    assert record["bundle_manifest_sha256"] == "def456"
