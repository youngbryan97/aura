"""A turn somebody was waiting on is read for what it was worth, like any other.

The learning phase is skipped on user-facing ticks, and the reading of a
turn's worth lived inside it, so no turn she spent with a person was read.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from core.affect import what_it_was_worth
from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from core.kernel import aura_kernel_lifecycle
from core.phases.learning_phase import LearningPhase, read_what_the_turn_was_worth
from core.runtime.pipeline_blueprint import BACKGROUND_ONLY_PHASES


def _state(valence: float, turn: int):
    """A turn with somebody in it, in which the two of them dwell on anemones."""
    return SimpleNamespace(
        affect=SimpleNamespace(
            valence=valence,
            arousal=0.5,
            emotions={},
            markers={"feelings_about": {"present": ["person:ines", "topic:times"]}},
        ),
        motivation=SimpleNamespace(budgets={}, forces={}),
        cognition=SimpleNamespace(
            active_goals=[],
            working_memory=[
                {"role": "user", "content": f"Visit {turn}: the anemone closed again. Does an anemone tire?"},
                {"role": "assistant", "content": "An anemone does not tire the way we do; the anemone resets."},
            ],
            last_response="",
        ),
        response_modifiers={},
    )


def test_reading_a_turn_puts_its_worth_on_the_state_and_its_things_on_her_record(tmp_path):
    what_it_was_worth.reset_for_test()
    record = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    record.talk.remember_talk(
        [("What do you think of the plan, and has the time come?", "I think the plan is fine, and the time has come.")] * 60
    )
    from tests.fixtures.things_her_library_names import NamesThese

    # What her library names of the turn: the anemone (core/conation/what_an_exchange_is_about.py).
    record.about = NamesThese("anemone")
    try:
        for turn, valence in enumerate([0.1, 0.4, -0.2, 0.5, -0.3, 0.6, 0.2, 0.7]):
            state = _state(valence, turn)
            asyncio.run(read_what_the_turn_was_worth(state))
            assert "worth" in state.response_modifiers, turn
        assert state.response_modifiers["worth"]["measured"] is True
        thing = record.thing("topic:anemone")
        assert thing is not None and thing.contacts >= 1 and thing.reachable
        assert record.thing("person:ines").contacts == thing.contacts
        assert record.thing("topic:times") is None
    finally:
        what_it_was_worth.reset_for_test()
        what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def test_the_phase_reads_the_turn_through_the_same_function():
    assert "read_what_the_turn_was_worth(state)" in inspect.getsource(LearningPhase.execute)


def test_a_tick_somebody_is_waiting_on_reads_the_turn_where_it_skips_the_learning():
    """The phase stays background-only; the reading is called in its place."""
    assert "LearningPhase" in BACKGROUND_ONLY_PHASES
    source = inspect.getsource(aura_kernel_lifecycle)
    skip = source.index("if self._should_skip_priority_phase(phase_name, priority=priority):")
    after = source[skip : source.index("continue", skip)]
    assert 'phase_name == "LearningPhase"' in after
    assert "await read_what_the_turn_was_worth(state)" in after


def test_her_stored_conversations_are_what_a_new_subject_stands_out_against(tmp_path, monkeypatch):
    from core.conversation import persistence
    from core.phases import learning_phase

    ordinary = [
        {"role": "user", "content": "What do you think of the plan, and has the time for a second look come?"},
        {"role": "aura", "content": "I think the plan is fine. There has been time, and I would rather say so plainly."},
    ] * 40

    class Stored:
        def get_recent_history(self, limit=100):
            return ordinary[:limit]

    monkeypatch.setattr(persistence, "get_persistence", lambda: Stored())
    record = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    try:
        assert not record.talk.talked_before()
        assert learning_phase._recall_her_talk() == 40
        assert record.talk.talked_before() and record.thing("topic:plan") is None
        about = record.talk.about(
            "The anemone let go of a small shell. Does an anemone prefer anything?",
            "An anemone holds what smells alive, and that is most of what an anemone goes on.",
        )
        assert about == ["anemone"]
    finally:
        what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def test_what_was_said_is_the_newest_message_and_her_answer_after_it():
    from core.phases import learning_phase

    learning_phase._said_this_turn.last = None
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            working_memory=[
                {"role": "user", "content": "an older question"},
                {"role": "assistant", "content": "an older answer"},
                {"role": "user", "content": "How does a limpet find its scar?"},
                {"role": "assistant", "content": "By fit. The scar is its own shape."},
            ],
            last_response="an older answer",
        )
    )
    assert learning_phase._said_this_turn(state) == [
        "How does a limpet find its scar?",
        "By fit. The scar is its own shape.",
    ]
    assert learning_phase._said_this_turn(state) == []


def test_the_loop_that_serves_chat_reads_the_turn_once_she_has_answered():
    from core.brain import cognitive_engine_thinking_loop

    source = inspect.getsource(cognitive_engine_thinking_loop)
    assert "await read_what_the_turn_was_worth(" in source
    assert "sent_from=" in source and 'get("user_id")' in source


def test_a_turn_somebody_sent_has_that_person_in_it_named_or_not():
    from core.phases.learning_phase import _with_the_person

    about = ["self", "topic:times", "attended:deliberation"]
    assert _with_the_person(about, "bryan", "desktop_ui")[0] == "person:bryan"
    assert _with_the_person(about, "", "desktop_ui")[0] == "person:someone at desktop_ui"
    assert "self" not in _with_the_person(about, "", "desktop_ui")
    assert _with_the_person(about, "", "") == about
