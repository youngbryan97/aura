"""A campaign turn arrives the way a desktop turn does, through the same functions.

The subject-core driver ran the phases and nothing around them. The desktop
path puts a person's message into working memory before the phases and stamps
who acted after them, and the driver did neither, so no campaign ever had a
partner in working memory. In the 21 September run 109 of 315 columns never
moved in 2,400 turns; among them were whether the newest thing in working
memory was the person's, how long their turn was, and every reading the song
organs take from the conversation.

Both paths now call core/kernel/turn_door.py. These pin the door itself, and
that a recorded conversation turn has the person in it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.kernel import turn_door


def _state(memory=None):
    return SimpleNamespace(cognition=SimpleNamespace(working_memory=memory))


@pytest.mark.unit
def test_a_message_enters_working_memory_as_the_persons_turn() -> None:
    state = _state([])
    turn_door.admit_message(state, "hello", "user")
    assert state.cognition.working_memory[-1]["role"] == "user"
    assert state.cognition.working_memory[-1]["content"] == "hello"
    assert state.cognition.working_memory[-1]["origin"] == "user"


@pytest.mark.unit
def test_the_same_message_is_not_admitted_twice_in_a_row() -> None:
    state = _state([])
    turn_door.admit_message(state, "hello", "user")
    turn_door.admit_message(state, "hello", "user")
    assert len(state.cognition.working_memory) == 1


@pytest.mark.unit
def test_a_broken_working_memory_raises_to_the_caller() -> None:
    with pytest.raises(TypeError):
        turn_door.admit_message(_state("not a list"), "hello", "user")


@pytest.mark.unit
def test_presence_is_only_for_a_person(monkeypatch) -> None:
    stimuli: list[str] = []

    class _Core:
        def receive_stimulus(self, *, intensity, source):
            stimuli.append(source)

    import core.consciousness.subcortical_core as subcortical

    monkeypatch.setattr(subcortical, "get_subcortical_core", lambda: _Core())
    turn_door.note_presence("tidy up", "system", conversation_id="user")
    turn_door.note_presence("tidy up", "motivation", conversation_id="user")
    assert stimuli == []
    turn_door.note_presence("hello", "user", conversation_id="user")
    assert stimuli == ["user"]


@pytest.mark.slow
def test_a_conversation_turn_has_the_person_in_it(monkeypatch, tmp_path: Path) -> None:
    from core.config import config
    from core.subject.clock import installed_clock
    from core.subject.driver import CONDITIONS, build_runtime, quiesce_organism, start_organism
    from core.subject.state import feature_names

    state_root = tmp_path / "state"
    state_root.mkdir()
    monkeypatch.setenv("AURA_STATE_ROOT", str(state_root))
    monkeypatch.setattr(config.paths, "home_dir_override", None)
    conversation = next(condition for condition in CONDITIONS if condition.name == "conversation")
    idle = next(condition for condition in CONDITIONS if condition.name == "idle")

    async def run():
        runtime = build_runtime(tmp_path / "runtime", seed=3)
        await start_organism(runtime)
        await quiesce_organism(runtime)
        talked = await runtime.turn_once(conversation)
        memory = list(runtime.state.cognition.working_memory)
        rested = await runtime.turn_once(idle)
        return runtime, talked, rested, memory

    try:
        runtime, talked, rested, memory = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()

    assert any(
        item.get("role") == "user" and item.get("content") == conversation.objective
        for item in memory
        if isinstance(item, dict)
    ), "the partner's message never reached working memory"
    assert "turn_door.open" not in runtime.failures, runtime.failure_notes.get("turn_door.open")
    assert "turn_door.close" not in runtime.failures, runtime.failure_notes.get("turn_door.close")

    memory_names = list(feature_names("M"))
    newest_is_user = memory_names.index("M.working_newest_is_user")
    assert any(float(frame.domain("M")[newest_is_user]) > 0.0 for frame in talked)

    world = list(feature_names("W"))
    turn_chars = world.index("W.partner_turn_chars")
    assert any(float(frame.domain("W")[turn_chars]) > 0.0 for frame in talked), (
        "the partner's turn had no length"
    )
    assert all(float(frame.domain("M")[newest_is_user]) == 0.0 for frame in rested), (
        "nobody spoke on an idle turn, and working memory said somebody had"
    )
