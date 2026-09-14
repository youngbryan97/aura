"""Her lineage, read off the states the repository kept.

The lineage module was tested on person-states written by hand, and nothing
turned Aura's own state into one. These pin the reader (what of her state is
the person, and what is only the turn) and the graph built from the state log's
parent links.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from core.state.aura_state import AuraState
from core.subject.lineage import lineage_from_state_log, person_state_of, state_digest


def _identity(**overrides) -> dict:
    base = {
        "name": "Aura Luna",
        "core_values": ["honesty", "care"],
        "current_narrative": "I have been learning the board.",
        "narrative_version": 3,
        "concept_graph": {"self": ["curious"]},
        "self_preferences": {"quiet": 0.7},
        "personality_growth": {"openness": 0.1},
        "bonding_level": 0.4,
        "evolution_score": 1.5,
    }
    base.update(overrides)
    return base


def _row(state_id: str, version: int, parent: str | None, **identity) -> dict:
    state = {"identity": _identity(**identity), "world": {"relationship_graph": {"Bryan": {"trust": 0.8}}}}
    return {"state_id": state_id, "version": version, "parent_state_id": parent, "state_json": json.dumps(state)}


def test_her_state_is_read_into_the_person_fields_it_carries() -> None:
    person = person_state_of(AuraState.default())
    assert {"values", "self_model", "dispositions", "preferences"} <= set(person)
    assert "skills" not in person and "policy" not in person, "what her state does not carry is absent"


def test_the_live_object_and_the_logged_json_are_the_same_person() -> None:
    logged = {"identity": _identity(), "world": {"relationship_graph": {}}}
    live = SimpleNamespace(
        identity=SimpleNamespace(**_identity()),
        world=SimpleNamespace(relationship_graph={}),
        cognition=SimpleNamespace(working_memory=[{"role": "user", "content": "hello"}]),
    )
    assert state_digest(person_state_of(live)) == state_digest(person_state_of(logged))


def test_being_spoken_to_does_not_change_the_person_and_a_value_does() -> None:
    state = AuraState.default()
    before = state_digest(person_state_of(state))
    state.cognition.working_memory.append({"role": "user", "content": "what did we decide?"})
    state.cognition.rolling_summary = "we talked about the garden"
    assert state_digest(person_state_of(state)) == before
    state.identity.core_values.append("patience")
    assert state_digest(person_state_of(state)) != before


def test_a_chain_of_versions_is_one_continued_line() -> None:
    lineage = lineage_from_state_log([_row("s1", 1, None), _row("s2", 2, "s1"), _row("s3", 3, "s2")])
    assert [(e.parent, e.child, e.kind) for e in lineage.edges] == [("s1", "s2", "continue"), ("s2", "s3", "continue")]
    assert lineage.same_person("s1", "s3")
    assert not lineage.verify()


def test_a_parent_with_two_children_in_the_log_was_forked() -> None:
    lineage = lineage_from_state_log([_row("s1", 1, None), _row("a", 2, "s1"), _row("b", 2, "s1")])
    assert {e.kind for e in lineage.edges} == {"fork"}
    assert lineage.branched("s1")
    assert not lineage.same_person("s1", "a")


def test_a_row_whose_parent_was_not_kept_is_a_root() -> None:
    lineage = lineage_from_state_log([_row("s5", 5, "s4"), _row("s6", 6, "s5")])
    assert lineage.parents("s5") == ()
    assert lineage.as_dict()["roots"] == ["s5"]


def test_a_changed_value_along_the_line_is_recorded_as_not_preserved() -> None:
    lineage = lineage_from_state_log([_row("s1", 1, None), _row("s2", 2, "s1", core_values=["honesty"])])
    (edge,) = lineage.edges
    assert edge.kind == "continue" and not edge.preserved
