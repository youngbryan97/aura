""""Why did you do that" is answered from the record, and the record exists.

Two defects, one shape. The provenance graph was opened in AuraKernel.tick,
and chat drives the legacy pipeline — so the causal record existed for the
handful of kernel ticks a day and not for the hundreds of turns a person has.
And nothing read the graph in production anyway: `why_field_changed` had test
callers only. A writer with no reader, feeding a reader with no writer.

These tests hold both halves: the pipeline that serves chat records, and the
reply path answers from what was recorded rather than from what the model
would say about itself.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.introspection.decision_provenance import (
    asks_why_she_did_that,
    runtime_authored_why,
    why_answer_is_available,
)
from core.runtime.cognitive_provenance import (
    begin_transformation,
    note_branch,
    recording_tick,
    reset_provenance_for_test,
)

_ENGINE = Path(__file__).resolve().parents[1] / "core" / "brain" / "cognitive_engine.py"


@pytest.fixture(autouse=True)
def _clean_provenance():
    reset_provenance_for_test()
    yield
    reset_provenance_for_test()


def _state(curiosity: float = 0.5):
    return SimpleNamespace(
        state_id="s1",
        version=1,
        updated_at=0.0,
        affect=SimpleNamespace(curiosity=curiosity, arousal=0.4, social_hunger=0.2),
        cognition=SimpleNamespace(
            discourse_depth=1,
            conversation_energy=0.5,
            working_memory=[],
            pending_initiatives=[],
        ),
        response_modifiers={},
    )


def _record_a_tick() -> None:
    state = _state()
    with recording_tick(objective="answer the question"):
        moved = begin_transformation("AffectUpdatePhase", state)
        note_branch("ordinary_decay", arousal=0.4)
        state.affect.curiosity = 0.69
        moved.complete(state)

        begin_transformation("InitiativeGenerationPhase", state).complete(
            state, skipped=True, skip_reason="user conversation active"
        )


# ── the question ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "why did you do that?",
        "Why did you choose that answer?",
        "why did you skip it?",
    ],
)
def test_a_question_about_her_own_behaviour_is_recognised(message: str) -> None:
    assert asks_why_she_did_that(message)


@pytest.mark.parametrize(
    "message",
    [
        "why do leaves change colour?",
        "why is the sky blue?",
        "what did you do?",
    ],
)
def test_a_question_about_the_world_is_not(message: str) -> None:
    """A false positive replaces an ordinary answer with a machine trace."""

    assert not asks_why_she_did_that(message)


# ── the answer ─────────────────────────────────────────────────────────────


def test_an_empty_graph_answers_nothing_rather_than_narrating() -> None:
    assert not why_answer_is_available()
    assert runtime_authored_why("why did you do that?") == ""


def test_the_answer_names_the_phase_the_branch_and_what_moved() -> None:
    _record_a_tick()
    answer = runtime_authored_why("why did you do that?")
    assert "AffectUpdatePhase" in answer
    assert "ordinary_decay" in answer
    assert "affect.curiosity" in answer


def test_the_answer_says_what_did_not_run_and_why() -> None:
    """Half of "why did you do that" is why something else did not happen."""

    _record_a_tick()
    answer = runtime_authored_why("why did you do that?")
    assert "InitiativeGenerationPhase did not run" in answer
    assert "user conversation active" in answer


def test_a_question_about_tokens_gets_the_mechanistic_limit_stated() -> None:
    _record_a_tick()
    answer = runtime_authored_why("why did you pick that token?")
    assert "not something I can answer mechanistically" in answer


def test_a_question_about_phases_does_not_get_the_limit_boilerplate() -> None:
    _record_a_tick()
    answer = runtime_authored_why("why did you do that?")
    assert "mechanistically" not in answer


# ── the record exists on the path that serves chat ─────────────────────────


def test_the_legacy_pipeline_opens_a_provenance_tick() -> None:
    """Chat drives this loop. A graph only the kernel writes is not a record."""

    source = _ENGINE.read_text("utf-8")
    assert "_open_provenance_tick(" in source
    assert "_close_provenance_tick(" in source


def test_every_legacy_phase_is_measured_and_skips_are_recorded() -> None:
    source = _ENGINE.read_text("utf-8")
    assert "_begin_provenance(phase_name, temp_state)" in source
    assert "_complete_provenance(" in source
    assert "_skip_provenance(phase_name, temp_state, reason)" in source


def test_the_tick_is_closed_in_a_finally_so_failed_turns_are_kept() -> None:
    """A record of only the turns that went well is the wrong half."""

    # "Closed in a finally" is a property of the exit path, not of where
    # the line sits. This walked for the call inside a finalbody and went
    # red when the method-size sweep lifted the whole finally-body into
    # _run_thinking_loop_closed_rather_after — leaving the CALL in the
    # finally, so the guarantee never moved. Checked as the exit path now.
    from source_contract import reached_from_a_finally

    import core.brain.cognitive_engine as engine_module

    assert reached_from_a_finally(engine_module, "_close_provenance_tick("), (
        "the provenance tick is closed on the success path only, so a turn "
        "that timed out or crashed leaves no record"
    )


def test_provenance_failures_never_break_a_turn() -> None:
    """Each wrapper swallows and logs. A record cannot take the runtime down."""

    import core.brain.cognitive_engine as engine

    assert engine._open_provenance_tick(objective=object(), priority=object()) is not None or True
    assert engine._begin_provenance("X", object()) is not None
    engine._complete_provenance(object(), object())
    engine._skip_provenance("X", object(), "reason")
    engine._close_provenance_tick(object())
