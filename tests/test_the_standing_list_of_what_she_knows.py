"""A standing list the prompt reads and nothing wrote.

`world.known_entities` is rendered as the KNOWN ENTITIES block by the context
assembler and read as W.entity_load by the subject schema. Grep the tree for a
write to it and nothing comes back: the block was empty on every turn and the
column read 0.0000 with zero spread through the six-hour run_032, while the
associative entity memory underneath it was recording every meeting.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core.memory.associative_entity_memory import (
    AssociativeEntityMemory,
    EntityKind,
)


@pytest.fixture()
def memory(tmp_path) -> Iterator[AssociativeEntityMemory]:
    held = AssociativeEntityMemory(db_path=tmp_path / "entities.db")
    try:
        yield held
    finally:
        held.close()


def test_an_empty_memory_knows_nobody(memory: AssociativeEntityMemory) -> None:
    assert memory.best_known() == []


def test_the_ones_she_has_met_most_come_first(memory: AssociativeEntityMemory) -> None:
    often = memory.resolve("the parser", kind=EntityKind.THING, create=True)
    once = memory.resolve("the kitchen", kind=EntityKind.PLACE, create=True)
    assert often is not None and once is not None
    for _ in range(5):
        memory.note_mention(often.entity_id)
    memory.note_mention(once.entity_id)

    known = memory.best_known()
    assert [entity.canonical_name for entity in known][:2] == ["the parser", "the kitchen"]


def test_the_list_is_bounded(memory: AssociativeEntityMemory) -> None:
    for index in range(20):
        entity = memory.resolve(f"thing {index}", kind=EntityKind.THING, create=True)
        assert entity is not None
        memory.note_mention(entity.entity_id)
    assert len(memory.best_known(limit=5)) == 5


def test_every_entity_can_describe_itself(memory: AssociativeEntityMemory) -> None:
    """The description is the memory's own sentence, not one written elsewhere."""
    entity = memory.resolve("the parser", kind=EntityKind.THING, create=True)
    assert entity is not None
    memory.note_mention(entity.entity_id)
    sentence = memory.stance(entity).sentence(entity.canonical_name)
    assert entity.canonical_name in sentence
    assert sentence.strip()


@pytest.mark.parametrize("reader", ["empty", "unavailable"])
def test_the_phase_mirrors_the_current_reader_into_the_prompt(memory, tmp_path, monkeypatch, reader):
    from core.brain.llm.context_assembler import ContextAssembler
    from core.memory import associative_entity_memory
    from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
    from core.state.aura_state import AuraState

    entity = memory.resolve("the parser", kind=EntityKind.THING, create=True)
    assert entity is not None
    memory.note_mention(entity.entity_id)
    state = AuraState.default()
    state.world.known_entities = {"the parser": {"already_weighed": True}}
    monkeypatch.setattr(associative_entity_memory, "get_associative_entity_memory", lambda: memory)
    ConversationalDynamicsPhase._mirror_known_entities(state)
    assert state.world.known_entities["the parser"]["already_weighed"] is True
    assert "the parser" in ContextAssembler.build_world_context(state)

    if reader == "unavailable":
        memory.close()
    else:
        empty = AssociativeEntityMemory(db_path=tmp_path / "empty.db")
        monkeypatch.setattr(associative_entity_memory, "get_associative_entity_memory", lambda: empty)
    try:
        ConversationalDynamicsPhase._mirror_known_entities(state)
        assert state.world.known_entities == {}
        assert "the parser" not in ContextAssembler.build_world_context(state)
    finally:
        if reader == "empty":
            empty.close()
