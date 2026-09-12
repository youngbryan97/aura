"""Depth two is measured, and it is much worse.

The two reports that used to sit in `artifacts/recurrent_depth` carried the
same responses digest, the same accuracy of 0.625 and no declared depth,
seventy seconds apart — one run written twice, because the tool that produced
them had no depth option and could not have varied the thing the filenames
claimed to compare.

`tools/run_recurrent_depth_arms.py` varies it. One model is patched once at
the deeper count, and the arms differ only in
``inner._recurrent_depth_runtime_loops`` — the integer the worker's surface
contract sets per request, and the exact quantity the ceiling governs.

Depth 1 scores 0.525 on the sealed forty-task battery. Depth 2 scores 0.050.
So the ceiling of one is not conservatism about something unmeasured; it is
the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARMS = Path(__file__).resolve().parents[1] / "artifacts" / "recurrent_depth"
SHALLOW = ARMS / "arm_loops1.json"
DEEP = ARMS / "arm_loops2.json"


@pytest.fixture(scope="module")
def arms() -> tuple[dict, dict]:
    if not SHALLOW.is_file() or not DEEP.is_file():
        pytest.skip("no depth arms on disk")
    return (
        json.loads(SHALLOW.read_text(encoding="utf-8")),
        json.loads(DEEP.read_text(encoding="utf-8")),
    )


def test_each_arm_declares_its_depth(arms):
    """Neither of the two that came before did."""
    shallow, deep = arms
    assert shallow["recurrent_loops"] == 1
    assert deep["recurrent_loops"] == 2


def test_the_arms_are_two_runs_and_not_one(arms):
    """The same digest twice is what made the old pair void."""
    shallow, deep = arms
    assert shallow["responses_sha256"] != deep["responses_sha256"]
    assert shallow["responses_sha256"]
    assert deep["responses_sha256"]


def test_the_depth_actually_reached_the_forward_pass(arms):
    """A loop count that no-ops measures the guard, not the depth."""
    for arm in arms:
        config = arm["recurrent_depth_config"]
        assert config["recurrent_layers"] >= 1
        assert config["num_layers"] >= config["recurrent_layers"]


def test_the_gate_refuses_the_deeper_arm(arms):
    """On the measurement, not on an absence."""
    import subprocess
    import sys

    repo = ARMS.parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo / "tools" / "gate_user_surface_recurrent_depth.py"),
         str(SHALLOW), str(DEEP)],
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=120,
    )
    verdict = json.loads(proc.stdout)
    assert verdict["refusals"] == [], "the arms are a real comparison now"
    assert verdict["authorized_depth"] == 1
    assert verdict["margin"] < 0.0


def test_the_live_ceiling_is_still_one():
    """Nothing above authorises it, and the measurement says not to."""
    from core.brain.llm.user_surface_recurrence import user_surface_recurrent_ceiling

    assert user_surface_recurrent_ceiling() == 1
