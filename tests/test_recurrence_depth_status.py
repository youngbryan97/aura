"""An episode that halted early is not an episode that ignored the request.

Affect asks for two recurrent loops on a distressed or curious turn. The
episode can converge, hit an invariant, or be stopped by the learned policy at
one step — all designed outcomes. The verifier reported every one of them as
`live_recurrence_depth_unproven`, so a correct adaptive halt was
indistinguishable from a depth request that was never applied, on every turn
where the surface asked for two.
"""
from __future__ import annotations

import pytest

from core.brain.latent_cortex_service import (
    LatentCortexService,
    _recurrence_halt_reason,
)


def _receipt(steps: int, halt_reason: str | None):
    receipt = {"steps_taken": steps}
    if halt_reason is not None:
        receipt["stop_gate"] = {"branches": [{"halt_reason": halt_reason}]}
    return receipt


# ── Telling the two apart ───────────────────────────────────────────────


def test_a_named_halt_reason_is_read_from_the_stop_gate():
    assert _recurrence_halt_reason(_receipt(1, "converged")) == "converged"
    assert _recurrence_halt_reason(_receipt(1, "learned_stop:quality")) == (
        "learned_stop:quality"
    )


def test_an_episode_with_no_halt_reason_reports_none():
    assert _recurrence_halt_reason(_receipt(1, None)) == ""
    assert _recurrence_halt_reason({"steps_taken": 1, "stop_gate": {}}) == ""
    assert _recurrence_halt_reason({"steps_taken": 1, "stop_gate": {"branches": []}}) == ""


def test_a_malformed_stop_gate_does_not_invent_a_reason():
    for gate in (None, "converged", {"branches": "converged"}, {"branches": [None]}):
        assert _recurrence_halt_reason({"steps_taken": 1, "stop_gate": gate}) == ""


def test_a_branch_with_an_empty_reason_is_not_a_reason():
    receipt = {"steps_taken": 1, "stop_gate": {"branches": [{"halt_reason": ""}]}}
    assert _recurrence_halt_reason(receipt) == ""


# ── What the surface reports ────────────────────────────────────────────


@pytest.fixture
def service():
    return LatentCortexService()


def test_an_adaptive_halt_is_expected_and_named(service):
    service._last_receipt = _receipt(1, "converged")
    status = service.recurrence_depth_status(requested_loops=2)
    assert status["state"] == "halted_early"
    assert status["expected"] is True
    assert status["halt_reason"] == "converged"
    assert status["steps_taken"] == 1
    assert status["requested_loops"] == 2


def test_a_dropped_request_is_not_expected(service):
    service._last_receipt = _receipt(1, None)
    status = service.recurrence_depth_status(requested_loops=2)
    assert status["state"] == "depth_not_applied"
    assert status["expected"] is False


def test_serving_the_requested_depth_says_so(service):
    service._last_receipt = _receipt(2, "budget")
    status = service.recurrence_depth_status(requested_loops=2)
    assert status["state"] == "served_requested_depth"
    assert status["expected"] is True


def test_exceeding_the_request_is_still_served(service):
    service._last_receipt = _receipt(4, "converged")
    assert service.recurrence_depth_status(requested_loops=2)["state"] == (
        "served_requested_depth"
    )


def test_an_absent_receipt_reports_unmeasured_rather_than_zero(service):
    service._last_receipt = {}
    assert service.recurrence_depth_status(requested_loops=2) == {"measured": False}


def test_a_nonsense_step_count_reports_unmeasured(service):
    for steps in ("2", None, -1, True):
        service._last_receipt = {"steps_taken": steps}
        assert service.recurrence_depth_status(2)["measured"] is False


def test_without_a_request_the_status_still_reports_what_happened(service):
    service._last_receipt = _receipt(1, "converged")
    status = service.recurrence_depth_status()
    assert status["measured"] is True
    assert status["steps_taken"] == 1
    assert status["halt_reason"] == "converged"
    # No request to compare against, so no verdict is invented.
    assert "state" not in status


# ── The verifier no longer faults a correct halt ────────────────────────


def test_the_verifier_only_faults_a_halt_with_no_reason():
    from pathlib import Path

    source = Path(
        __file__
    ).resolve().parents[1] / "core/brain/latent_cortex_service.py"
    text = source.read_text()
    assert "live_recurrence_depth_not_applied" in text
    # The old blanket verdict must not fire on a short-but-explained episode.
    window = text[text.index("expected_loops = controls.get") :]
    window = window[: window.index("request_payload_sha256")]
    assert "_recurrence_halt_reason(" in window
