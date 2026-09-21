"""Whether anything ever puts a person in her world model.

`W.entity_load` read 0.000000 on all 5,280 frames of a probe, beside
`relationship_load`, `preference_load` and `concept_load` at zero and
`fact_load` pinned at a single fact. Five of the world model's columns could
not move, in the domain the interventional graph gave one measured cause.

`core/memory/entity_memory_bridge.py` exports `record_turn_evidence` and
nothing in the tree called it. Entities are only introduced from the text of a
message, and none of the eight ordinary situations a campaign runs introduces
anybody — so the entity memory was empty, and the standing list of what she
knows was empty with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PHASE = (
    Path(__file__).resolve().parents[1]
    / "core" / "phases" / "conversational_dynamics_phase.py"
)


@pytest.fixture
def memory(tmp_path, monkeypatch):
    from core.memory import associative_entity_memory as module

    monkeypatch.setattr(module, "_MEMORY", None, raising=False)
    store = module.AssociativeEntityMemory(tmp_path / "entities.sqlite3")
    if not store.available:
        pytest.skip("the entity store is unavailable in this environment")
    return store


def test_the_bridge_writer_has_a_caller_at_last():
    """A writer nothing calls is a capability with no door into it."""
    tree = ast.parse(PHASE.read_text())
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "record_turn_evidence" in calls
    assert "learn_entity" in calls


def test_the_interlocutor_becomes_someone_she_knows(memory):
    from core.memory.associative_entity_memory import EntityKind
    from core.memory.entity_memory_bridge import learn_entity, record_turn_evidence

    assert memory.best_known() == []
    entity = learn_entity("bryan", EntityKind.PERSON, memory=memory)
    assert entity is not None
    record_turn_evidence(
        entity, episode_id="turn:1", valence=0.4, arousal=0.6,
        role="interlocutor", memory=memory,
    )
    known = memory.best_known()
    assert [one.canonical_name for one in known] == [entity.canonical_name]


def test_meeting_them_again_is_counted(memory):
    from core.memory.associative_entity_memory import EntityKind
    from core.memory.entity_memory_bridge import learn_entity, record_turn_evidence

    for turn in range(4):
        entity = learn_entity("bryan", EntityKind.PERSON, memory=memory)
        memory.note_mention(entity.entity_id)
        record_turn_evidence(
            entity, episode_id=f"turn:{turn}", valence=0.1 * turn, arousal=0.5,
            role="interlocutor", memory=memory,
        )
    known = memory.best_known()
    assert len(known) == 1
    assert known[0].mention_count >= 4


def test_the_kind_survives_being_passed_as_the_kind(memory):
    """`EntityKind.coerce` could not coerce its own enum: `str(member)` is
    "EntityKind.PERSON", so every caller passing the enum its own signatures
    invite got OTHER, silently. The kind selects how stance is computed and
    people delegate bonding to the attachment system."""
    from core.memory.associative_entity_memory import EntityKind
    from core.memory.entity_memory_bridge import learn_entity

    assert EntityKind.coerce(EntityKind.PERSON) is EntityKind.PERSON
    assert EntityKind.coerce("person") is EntityKind.PERSON
    assert EntityKind.coerce("nonsense") is EntityKind.OTHER
    entity = learn_entity("bryan", EntityKind.PERSON, memory=memory)
    assert entity is not None
    assert entity.kind is EntityKind.PERSON


def test_nobody_is_introduced_without_a_partner():
    """The guard that keeps this a deliberate act rather than silent filling."""
    from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase

    class _State:
        version = 1
        affect = type("A", (), {"valence": 0.0, "arousal": 0.5})()

    # No partner, no introduction, and no raise.
    ConversationalDynamicsPhase._let_the_entity_record_see_this_turn(_State(), "")
