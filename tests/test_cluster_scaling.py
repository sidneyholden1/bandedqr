"""Control-plane tests that never contact Slurm or FI quota services."""

import importlib.util
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUSTY = ROOT / "cluster" / "rusty"
sys.path.insert(0, str(RUSTY))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


quota = load("bandedqr_quota_monitor", RUSTY / "quota_monitor.py")
control = load("bandedqr_scaling_control", RUSTY / "scaling_control.py")


def test_quota_report_parsers():
    usage = """
CPU Hours
Used: 120
Planned: 1000
Storage Files
Used: 400
Planned: 3000
"""
    home = """
FILE COUNT
Current: 700
Soft Limit: 1200
Hard Limit: 1500
"""
    assert quota.usage_metric(usage, "CPU Hours") == (120, 1000)
    assert quota.usage_metric(usage, "Storage Files") == (400, 3000)
    assert quota.quota_metric(home, "FILE COUNT") == (700, 1200)


def test_policy_and_initial_ledger_agree():
    policy = json.loads((ROOT / "cluster/resource_policy.json").read_text())
    ledger = control.initial_ledger(policy)
    assert ledger["campaign_id"] == policy["campaign_id"]
    assert ledger["campaign_cpu_hour_ceiling"] == 100000
    assert ledger["campaign_gpu_hour_ceiling"] >= 1
    assert sum(
        row["reserved_cpu_hours"]
        for row in policy["concrete_allocations"].values()
    ) == 88


def test_terminal_state_is_conservative():
    assert control.terminal_state("COMPLETED")
    assert control.terminal_state("FAILED+")
    assert not control.terminal_state("PENDING")
    assert not control.terminal_state("RUNNING")


def test_sacct_timestamp_uses_cluster_local_time():
    epoch = 1_794_165_393.0
    assert control.slurm_local_timestamp(epoch) == datetime.fromtimestamp(
        epoch
    ).strftime("%Y-%m-%dT%H:%M:%S")


def test_rusty_launcher_accepts_standard_tmp_fallback():
    launcher = RUSTY / "run_scaling.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    source = launcher.read_text()
    assert 'SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"' in source
    assert 'mktemp -d "${SCRATCH_BASE%/}/bandedqr-${SLURM_JOB_ID}.XXXXXX"' in source
