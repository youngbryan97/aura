"""The checks of her change run in a developer's environment, not her runtime's (F16).

Her runtime sets ``AURA_GOVERNANCE_MODE=production`` on itself, and every check of a change of hers ran as its child
with it. Under it, 20 tests that pass at a shell fail, and to 20 November 2026 her own yes refused her changes on
them: her draft at core/resource/resource_governor.py:481 was refused on
tests/test_resource_observation_hermeticity.py ("offline subprocess tooling bypass denied while live governance is
active"), and none of hers reached origin after 17 November.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.self_modification.a_checks_environment import a_checks_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def her_runtimes_variables(monkeypatch):
    for name, value in {"AURA_GOVERNANCE_MODE": "production", "AURA_REQUIRE_GOVERNANCE": "1",
                        "AURA_LAUNCHED_FROM_APP": "1", "AURA_LIVE_STATE_ROOT": "/Users/someone/.aura"}.items():
        monkeypatch.setenv(name, value)


@pytest.mark.unit
def test_none_of_her_runtimes_variables_reach_a_check(her_runtimes_variables):
    env = a_checks_environment(log_prefix="t-", PYTHON=sys.executable)
    assert not [name for name in env if name.startswith("AURA_") and name != "AURA_LOG_DIR"]
    assert env["PATH"] and env["PYTHON"] == sys.executable and Path(env["AURA_LOG_DIR"]).is_dir()
    assert env["VECLIB_MAXIMUM_THREADS"] == env["OMP_NUM_THREADS"] == "1"
    assert a_checks_environment(log_dir="/tmp/given")["AURA_LOG_DIR"] == "/tmp/given"


@pytest.mark.unit
def test_the_gates_her_yes_asks_run_in_it(her_runtimes_variables, monkeypatch, tmp_path):
    import core.agency.her_goal_at_work as work

    seen = []
    monkeypatch.setattr(work, "_run_a_gate", lambda argv, *, cwd, env: seen.append(env) or (True, ""))
    monkeypatch.setattr(work.GoalRecord, "workspace", property(lambda _self: tmp_path), raising=False)
    work.the_gates_her_yes_asks(work.GoalRecord(goal="g", slug="g", offer={"files": []}))
    assert seen and all("AURA_GOVERNANCE_MODE" not in env and "AURA_REQUIRE_GOVERNANCE" not in env for env in seen)


@pytest.mark.slow
def test_the_test_that_refused_her_change_passes_in_it(her_runtimes_variables):
    test = "tests/test_resource_observation_hermeticity.py::test_host_leak_guard_observes_spawned_child"
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test]
    as_her_runtime = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)
    assert as_her_runtime.returncode != 0, "under her runtime's governance it fails, as it refused her change"
    as_a_developer = subprocess.run(argv, cwd=ROOT, env=a_checks_environment(), capture_output=True, text=True,
                                    timeout=300, check=False)
    assert as_a_developer.returncode == 0, as_a_developer.stdout[-1500:]
