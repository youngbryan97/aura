"""A standing list the prompt reads and nothing wrote.

`world.known_entities` is rendered as the KNOWN ENTITIES block by the context
assembler and read as W.entity_load by the subject schema. Grep the tree for a
write to it and nothing comes back: the block was empty on every turn and the
column read 0.0000 with zero spread through the six-hour run_032, while the
associative entity memory underneath it was recording every meeting.
"""

from __future__ import annotations

import pytest

from core.memory.associative_entity_memory import (
    AssociativeEntityMemory,
    EntityKind,
)


@pytest.fixture()
def memory(tmp_path) -> AssociativeEntityMemory:
    return AssociativeEntityMemory(db_path=tmp_path / "entities.db")


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


def test_the_phase_mirrors_it_into_the_field_the_prompt_reads() -> None:
    from pathlib import Path

    phase = Path("core/phases/conversational_dynamics_phase.py").read_text(encoding="utf-8")
    assert "state.world.known_entities = {" in phase
    assert "best_known()" in phase

    assembler = Path("core/brain/llm/context_assembler.py").read_text(encoding="utf-8")
    assert "world.known_entities" in assembler
