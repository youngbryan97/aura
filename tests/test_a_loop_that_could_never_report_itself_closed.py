"""Closure defined as something the pipeline cannot bring about.

A CRSM cycle ends by writing a fused candidate and marking the captures
consumed against it. It deliberately leaves the active-model pointer alone —
activation is a separate, staged act. So the marker names the candidate and
the pointer names the incumbent, and by design they do not match.

Closure was defined as those two matching. The autonomous closer prepares a
dataset and trains, and has no evaluate or activate phase at all, so the loop
could never report itself closed however well it ran. The test that covered
this faked a monitor whose active model changed when training returned zero,
which is the one thing the real pipeline never does — a double that does not
track the contract of the thing it stands in for reports the opposite of the
truth.

The closure test is left exactly as strict. Saying closed when nothing was
activated would be a claim about the model she is actually running. What was
missing is a name for the state that does happen.
"""

from __future__ import annotations

import json

import pytest

from core.consciousness.crsm_loop_monitor import CRSMLoopMonitor


@pytest.fixture
def loop(tmp_path, monkeypatch):
    """A monitor over a scratch tree, with a dataset and a fused candidate."""
    dataset = tmp_path / "captures.jsonl"
    lines = [json.dumps({"n": n}) for n in range(3)]
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fused = tmp_path / "fused"
    fused.mkdir()
    candidate = fused / "gen-1"
    candidate.mkdir()

    monitor = CRSMLoopMonitor()
    monitor.fused_model_dir = fused
    return monitor, tmp_path, dataset, candidate


def _state(monitor, *, marker, active):
    """The loop state with a given marker and active pointer."""
    monitor._consumed_marker = lambda: marker
    monitor.dataset_state = lambda: {
        "lines": 3,
        "sha256": "abc",
        "size": 12,
        "mtime": 100.0,
    }
    monitor.latest_training_artifact = lambda: {
        "newest_mtime": 200.0,
        "active_fused_at": 200.0,
        "active_model_path": active,
    }
    monitor.integration_manifest_state = lambda: {}
    monitor.training_state = lambda: {}
    monitor.eligible_capture_count = lambda: 3
    return monitor.loop_state()


def _marker(model_path: str) -> dict:
    return {
        "lines_consumed": 3,
        "dataset_sha256": "abc",
        "dataset_size": 12,
        "accepted_lines": 3,
        "rejected_lines": 0,
        "consumed_at": 150.0,
        "model_path": model_path,
    }


def test_a_trained_candidate_that_is_not_active_is_not_called_closed(loop):
    """The strict test stays strict: nothing was activated."""
    monitor, _tmp, _ds, candidate = loop
    said = _state(monitor, marker=_marker(str(candidate)), active="/some/other/model")
    assert said["state"] != "closed"


def test_it_is_named_rather_than_left_looking_like_a_failure(loop):
    """This is what the pipeline actually produces, so it needs a name."""
    monitor, _tmp, _ds, candidate = loop
    said = _state(monitor, marker=_marker(str(candidate)), active="/some/other/model")
    assert said["state"] == "qualified"
    assert "activation" in said["reason"]


def test_closed_still_means_the_active_model_is_the_trained_one(loop):
    monitor, _tmp, _ds, candidate = loop
    said = _state(monitor, marker=_marker(str(candidate)), active=str(candidate))
    assert said["state"] == "closed"


def test_a_qualified_loop_asks_for_no_further_training(loop):
    """It has been trained. Asking again would train the same captures twice."""
    monitor, _tmp, _ds, candidate = loop
    monitor.loop_state = lambda: {"state": "qualified"}
    monitor.integration_manifest_state = lambda: {"current_for_dataset": True}
    plan = monitor.next_action()
    assert plan["required"] is False
    assert "activation" in plan["reason"]


def test_a_marker_naming_no_model_is_not_qualified_either(loop):
    """Trained against nothing is not a candidate awaiting anything."""
    monitor, _tmp, _ds, _candidate = loop
    said = _state(monitor, marker=_marker(""), active="/some/other/model")
    assert said["state"] not in {"closed", "qualified"}
