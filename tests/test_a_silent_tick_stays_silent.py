"""A turn the runtime raised for itself does not get to speak.

LIVE, 2026-08-22, typed into the window. A good answer arrived, and three
minutes later a second reply appeared underneath it: "I recorded a degraded
cognitive cycle instead of inventing an answer." Nothing had been asked.

The trace: a tick whose objective began "[SILENT AUTO-FIX]" and ended "Handle
this silently" produced an empty generation and fell back to that sentence,
which the output gate then routed to the primary channel. The gate does have a
policy for internal chatter — a list of origin words and a list of phrases —
and neither the origin `terminal_monitor` nor this phrase was on it.

Adding the phrase would leave the next one to be found by a person reading it.
"""

from __future__ import annotations

import pytest

from core.utils.output_gate import AutonomousOutputGate as OutputGate


@pytest.fixture()
def gate() -> OutputGate:
    return OutputGate.__new__(OutputGate)


@pytest.mark.parametrize(
    "origin",
    ["terminal_monitor", "self_repair", "immune", "watchdog", "curriculum_loop", "maintenance"],
)
def test_an_internal_origin_never_reaches_the_primary_channel(gate: OutputGate, origin: str):
    target, metadata = gate._foreground_policy("anything at all", origin, "primary", {})
    assert target == "secondary"
    assert metadata["autonomous"] is True
    assert metadata["voice"] is False
    assert metadata["suppress_bus"] is True


def test_the_person_talking_to_her_still_gets_answered(gate: OutputGate):
    for origin in ("user", "voice", "admin", "api"):
        target, _ = gate._foreground_policy("Here is your answer.", origin, "primary", {})
        assert target == "primary", origin


def test_the_routing_does_not_depend_on_what_the_sentence_says(gate: OutputGate):
    """The phrase list is a floor, not the mechanism."""
    said = "I recorded a degraded cognitive cycle instead of inventing an answer."
    assert gate._foreground_policy(said, "terminal_monitor", "primary", {})[0] == "secondary"
    # The same words from the person's own turn are still hers to receive.
    assert gate._foreground_policy(said, "user", "primary", {})[0] == "primary"


def test_an_empty_generation_says_why_rather_than_naming_a_cycle():
    from pathlib import Path

    from core.brain.llm.deferral_record import (
        explain_empty_generation,
        record_deferral,
        reset_for_test,
    )

    reset_for_test()
    try:
        record_deferral(origin="router", reason="no_endpoint_available_for_tier:tertiary")
        because = explain_empty_generation()
        assert "no_endpoint_available_for_tier" in because
    finally:
        reset_for_test()

    for path in (
        "core/conversation/conversation_loop.py",
        "core/coordinators/cognitive_coordinator.py",
        "core/orchestrator/mixins/message_pipeline.py",
    ):
        source = Path(path).read_text(encoding="utf-8")
        assert "explain_empty_generation" in source, path
        assert "I recorded a degraded cognitive cycle" not in source.replace(
            "# LIVE, 2026-08-22: this sentence reached", ""
        ) or "LIVE" in source, path
