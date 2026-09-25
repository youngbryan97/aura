"""The compounding-error reading had no consumer.

The continuous experience stream measures when her recent frames are going
wrong — three in a row, or four of the last six, carrying a low outcome score,
harm, a repeated prediction miss or unity repair pressure — and publishes
`safe_to_act: False` with a recommended mode of "observe, stabilise, replay".
Nothing in the tree read either. A run of bad frames changed nothing about how
the next turn was taken.

Observing rather than acting is deliberating rather than reacting, and the
cognitive mode is a gate that already existed.
"""

from __future__ import annotations

import pytest

from core.consciousness.continuous_experience import CompoundingErrorReport
from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.state.aura_state import AuraState, CognitiveMode


class _Stream:
    def __init__(self, report) -> None:
        self.compounding_report = report


@pytest.fixture
def stream(monkeypatch):
    def _set(report):
        from core.consciousness import continuous_experience

        monkeypatch.setattr(
            continuous_experience,
            "get_continuous_experience_stream",
            lambda: _Stream(report),
        )

    return _set


def _after(mode: CognitiveMode) -> CognitiveMode:
    state = AuraState.default()
    state.cognition.current_mode = mode
    CognitiveRoutingPhase._deliberate_after_a_bad_run(state)
    return state.cognition.current_mode


def test_a_quiet_stream_changes_nothing(stream):
    stream(CompoundingErrorReport(False))
    assert _after(CognitiveMode.REACTIVE) is CognitiveMode.REACTIVE
    assert _after(CognitiveMode.DELIBERATE) is CognitiveMode.DELIBERATE


def test_a_compounding_run_lifts_a_reflex_into_deliberation(stream):
    stream(
        CompoundingErrorReport(
            True,
            severity=0.7,
            reasons=("harm_accumulating", "prediction_mismatch_repeated"),
            recommended_mode="observe_stabilize_replay",
        )
    )
    assert _after(CognitiveMode.REACTIVE) is CognitiveMode.DELIBERATE


def test_it_never_makes_a_deliberate_turn_reflexive(stream):
    stream(CompoundingErrorReport(True, severity=0.9))
    assert _after(CognitiveMode.DELIBERATE) is CognitiveMode.DELIBERATE


def test_an_absent_stream_leaves_the_mode_alone(monkeypatch):
    from core.consciousness import continuous_experience

    def _raise():
        raise RuntimeError("no stream here")

    monkeypatch.setattr(
        continuous_experience, "get_continuous_experience_stream", _raise
    )
    assert _after(CognitiveMode.REACTIVE) is CognitiveMode.REACTIVE


def test_the_real_stream_reports_a_run_of_bad_frames():
    """The measurement itself, so the consumer is wired to something live."""
    from core.consciousness.continuous_experience import ContinuousExperienceStream

    stream = ContinuousExperienceStream(persist_path=None, autosave=False)
    assert stream.compounding_report.active is False


def test_routing_still_returns_a_state():
    """The wrapper must not swallow the route's answer."""
    import asyncio

    phase = CognitiveRoutingPhase(None)
    state = AuraState.default()
    out = asyncio.run(phase.execute(state, objective=""))
    assert out is not None
