"""Two lines in the prompt that stated things nobody had checked.

`user_present=True` was a literal passed into the temporal-finitude model,
which feeds live self-report and any causal experiment reading it — a
fabricated observation in the one place the runtime describes its own
situation. And a felt-thought trace was labelled "last reply (measured)" on the
strength of a recent timestamp, which says a measurement exists, not that it
belongs to the reply the line attributes it to.
"""
from __future__ import annotations

import time
from pathlib import Path

from core.brain.llm.context_assembler import (
    _USER_PRESENCE_WINDOW_S,
    ContextAssembler,
)
from core.state.aura_state import AuraState

ROOT = Path(__file__).resolve().parents[1]


def _state(origin: str, *, last_user_age_s: float | None = None) -> AuraState:
    state = AuraState.default()
    state.cognition.current_origin = origin
    if last_user_age_s is not None:
        state.cognition.working_memory = [
            {"role": "user", "content": "hi", "timestamp": time.time() - last_user_age_s}
        ]
    return state


def test_a_background_tick_has_nobody_waiting():
    assert ContextAssembler._user_is_present(_state("background", last_user_age_s=1.0)) is False
    assert ContextAssembler._user_is_present(_state("dream", last_user_age_s=1.0)) is False


def test_a_live_turn_has_somebody_waiting():
    assert ContextAssembler._user_is_present(_state("gui", last_user_age_s=2.0)) is True
    assert ContextAssembler._user_is_present(_state("voice", last_user_age_s=2.0)) is True


def test_an_abandoned_session_stops_claiming_an_audience():
    stale = _state("gui", last_user_age_s=_USER_PRESENCE_WINDOW_S + 60.0)

    assert ContextAssembler._user_is_present(stale) is False


def test_the_first_turn_of_a_session_counts_as_presence():
    """A user-facing origin with no timestamped human message yet."""
    assert ContextAssembler._user_is_present(_state("gui")) is True


def test_presence_is_not_a_literal_in_the_prompt_path():
    from tests.source_contract import family_text_at

    source = family_text_at(ROOT / "core" / "brain" / "llm" / "context_assembler.py")

    assert "user_present=True" not in source
    assert "user_present=ContextAssembler._user_is_present(state)" in source


def test_an_unbound_trace_does_not_claim_to_be_this_reply(monkeypatch):
    import core.being.thought_interoception as ti

    class Trace:
        timestamp = time.time()
        fluency = 0.8
        felt_confidence = 0.7
        ambivalence = 0.1
        strain = 0.2
        bound = False

    class Engine:
        @staticmethod
        def last(foreground_only=False):
            return Trace()

    monkeypatch.setattr(ti, "get_thought_interoception", lambda: Engine())

    block = ContextAssembler._build_felt_thought_block(compact=True)

    assert "not bound to this reply" in block
    assert "last reply (measured)" not in block


def test_a_bound_trace_is_the_reply(monkeypatch):
    import core.being.thought_interoception as ti

    class Trace:
        timestamp = time.time()
        fluency = 0.8
        felt_confidence = 0.7
        ambivalence = 0.1
        strain = 0.2
        bound = True

    class Engine:
        @staticmethod
        def last(foreground_only=False):
            return Trace()

    monkeypatch.setattr(ti, "get_thought_interoception", lambda: Engine())

    block = ContextAssembler._build_felt_thought_block(compact=True)

    assert "last reply (measured)" in block
    assert "not bound" not in block
