"""Seven dimensions read absent on every recorded turn, and one organ had them.

Measured across 1,629 live turns on 2026-08-25: of 74 named dimensions, 47
were never present. The uncertainty channel was 0 of 4 — so `do(U=low)`, the
intervention the whole design is pointed at, had nothing to act on.

Every one of those channels was looking for an organ by name and naming an
organ that does not exist. Uncertainty tried `calibration_tracker`,
`confidence_calibrator`, `epistemics` and `uncertainty_engine`. Self-state
tried `ghost`, `soul`, `health_monitor` and `watchdog`. Memory asked the
facade for `semantic_density` and `contradiction_rate`, and the facade
publishes which stores exist and nothing else. Not one of those names is
registered anywhere in this tree, so none of the readers could ever fire.

The welfare model maintains all of it, computes from state already in this
process, and costs 16 microseconds warm — measured over 20 calls. It is not
resolved through the container because it is a process singleton with its own
accessor, which is part of why nothing had found it.

Each binding below is checked against the field name the welfare model
actually publishes, because a reader tested only against its own stub is how
all seven of these broke.
"""

from __future__ import annotations

import pytest

from core.brain.llm.endogenous_state import (
    _probe_memory,
    _probe_self_state,
    _probe_uncertainty,
    _welfare_number,
    layout_digest,
)


@pytest.fixture
def welfare_fields() -> tuple[dict, dict]:
    """The real field names, read off the real welfare model."""
    from core.being.body_state_service import BodyStateService
    from core.being.welfare_state import WelfareState

    welfare = WelfareState.get()
    body = BodyStateService.get().snapshot()
    inputs = welfare.gather_inputs(body=body)
    outputs = welfare.compute(inputs)
    return vars(inputs), vars(outputs)


def test_every_field_this_binds_to_actually_exists(welfare_fields):
    """The check that would have caught all seven at the time they were written."""
    inputs, outputs = welfare_fields

    for field in ("prediction_error", "memory_coherence", "truth_integrity",
                  "permission_confidence", "goal_frustration"):
        assert field in inputs, field
    for field in ("confidence", "action_inhibition"):
        assert field in outputs, field


def test_uncertainty_is_no_longer_empty():
    """0 of 4 was the state that made do(U=low) impossible to run."""
    read = _probe_uncertainty()

    assert read is not None
    assert "uncertainty.confidence" in read
    assert "uncertainty.abstention_pressure" in read
    assert "uncertainty.calibration_error" in read


def test_self_state_and_memory_contradiction_have_a_source():
    self_state = _probe_self_state() or {}
    memory = _probe_memory() or {}

    assert "self.integrity" in self_state
    assert "self.agency" in self_state
    assert "memory.contradiction" in memory


def test_agency_is_not_a_negative_copy_of_abstention_pressure():
    """The trap this deliberately avoids.

    `1 - action_inhibition` is the obvious reading for agency and it would be
    a perfect negative copy of uncertainty.abstention_pressure. A duplicated
    dimension makes an ablation of one channel silently a partial ablation of
    another — three such pairs already existed in this state and were only
    found by measuring the corpus.
    """
    inputs = {"permission_confidence": 0.25, "truth_integrity": 0.9}
    outputs = {"action_inhibition": 0.25, "confidence": 0.5}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "core.brain.llm.endogenous_state._welfare_reading",
        lambda: (inputs, outputs),
    )
    monkeypatch.setattr(
        "core.brain.llm.endogenous_state._service", lambda name: None
    )
    try:
        agency = (_probe_self_state() or {})["self.agency"]
        pressure = (_probe_uncertainty() or {})["uncertainty.abstention_pressure"]
        # Both sources read 0.25 here, so a negative copy would give 0.75 and
        # the honest binding gives 0.25. The two dimensions are free to agree;
        # what they must not be is one number wearing two names.
        assert agency == pytest.approx(0.25)
        assert pressure == pytest.approx(0.25)

        # Now move only the permission input. Agency must follow it, and the
        # abstention pressure must not.
        inputs["permission_confidence"] = 0.8
        assert (_probe_self_state() or {})["self.agency"] == pytest.approx(0.8)
        assert (_probe_uncertainty() or {})[
            "uncertainty.abstention_pressure"
        ] == pytest.approx(0.25)
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, 0.5), (1.4, 1.0), (-0.2, 0.0), (True, None), ("x", None), (None, None)],
)
def test_a_welfare_number_is_bounded_or_refused(value, expected):
    assert _welfare_number({"k": value}, "k") == expected


def test_a_missing_welfare_model_leaves_the_channels_absent(monkeypatch):
    """Fail open: a turn that cannot read welfare generates as it always did."""
    monkeypatch.setattr(
        "core.brain.llm.endogenous_state._welfare_reading", lambda: None
    )
    monkeypatch.setattr(
        "core.brain.llm.endogenous_state._service", lambda name: None
    )

    assert _probe_uncertainty() is None
    assert _probe_self_state() is None


def test_none_of_this_changed_the_layout():
    """Wiring a channel to a real organ must not invalidate 1,729 recorded
    turns. The digest covers the feature list, not the sources behind it."""
    assert layout_digest() == "d3a59c56a36248d52b67d8282bf2f702"
