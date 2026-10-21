"""A person who learns is still herself, when the change has the right cause.

A reviewer reproduced it on 21 October 2026: a preference at t1, changed by
experience at t2, the two joined by a signed transition that verifies, and
`same_person(t1, t2)` came back False, because every identity-preserving step
had to keep the canonical state's hash. Any learning ended identity. A step now
develops when each path it changed was written by a measured transformation
that kept to its contract and at least half of what she carried came through.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.subject.lineage import STRONG_CONNECTEDNESS, Lineage, lineage_from_state_log

pytestmark = pytest.mark.unit


def _person(**overrides):
    base = {
        "autobiographical": ["met Bryan", "learned the board"],
        "self_model": {"name": "Aura", "efficacy": 0.7},
        "values": ["honesty", "care"],
        "preferences": {"depth": 0.8},
        "dispositions": {"curiosity": 0.6},
        "skills": ["reading a screen"],
        "relationships": {"Bryan": "close"},
        "policy": {"defer_below": 0.3},
        "developmental": {"era": 3},
    }
    base.update(overrides)
    return base


def _wrote(transform: str = "PreferenceLearning", honoured: bool = True) -> list[dict]:
    return [{"transform": transform, "honoured": honoured, "tick": "t-1", "at": 2.0}]


def _two_stages(after: dict) -> Lineage:
    line = Lineage()
    line.record("t1", _person(), at=1.0)
    line.record("t2", after, at=3.0)
    return line


def test_the_reviewers_case_a_preference_learned_from_experience_is_the_same_person():
    line = _two_stages(_person(preferences={"depth": 0.6}))
    edge = line.descend("t1", "t2", writes={"preferences:depth": _wrote()})
    assert line.verify() == []
    assert edge.changed == ("preferences:depth",)
    assert edge.develops
    assert line.same_person("t1", "t2") is True
    # The old, stricter reading still answers when asked for it.
    assert line.unchanged_between("t1", "t2") is False


def test_a_change_nothing_was_measured_writing_ends_continuity():
    line = _two_stages(_person(preferences={"depth": 0.6}))
    line.descend("t1", "t2")
    assert line.same_person("t1", "t2") is False
    broken = line.first_break("t1", "t2")
    assert broken["unexplained"] == ["preferences:depth"]


def test_a_write_outside_its_contract_does_not_explain_the_change():
    line = _two_stages(_person(preferences={"depth": 0.6}))
    line.descend("t1", "t2", writes={"preferences:depth": _wrote(honoured=False)})
    assert line.same_person("t1", "t2") is False


def test_most_of_her_rewritten_at_once_is_not_development_however_well_explained():
    after = _person(
        autobiographical=["someone else's life"],
        self_model={"name": "Other", "efficacy": 0.1},
        values=["obedience"],
        preferences={"depth": 0.1},
        dispositions={"curiosity": 0.0},
        skills=["nothing"],
    )
    line = _two_stages(after)
    edge = line.descend("t1", "t2", writes={path: _wrote() for path in line.stages["t2"].paths})
    assert not edge.unexplained()
    assert edge.connectedness < STRONG_CONNECTEDNESS
    assert line.same_person("t1", "t2") is False


def test_learning_does_not_make_a_fork_one_person():
    line = Lineage()
    line.record("t1", _person(), at=1.0)
    line.record("left", _person(preferences={"depth": 0.6}), at=3.0)
    line.record("right", _person(preferences={"depth": 0.9}), at=3.0)
    line.descend("t1", "left", kind="fork", writes={"preferences:depth": _wrote()})
    line.descend("t1", "right", kind="fork", writes={"preferences:depth": _wrote()})
    assert line.same_person("t1", "left") is False and line.same_person("t1", "right") is False


def test_the_writes_are_under_the_signature():
    line = _two_stages(_person(preferences={"depth": 0.6}))
    line.descend("t1", "t2", writes={"preferences:depth": _wrote()})
    line.edges[0] = replace(line.edges[0], writes={"preferences:depth": tuple(_wrote("Forged"))})
    assert line.verify(), "a forged account of what wrote a change still verified"


def test_the_key_is_this_installs_and_not_one_in_the_source():
    key = Lineage().key
    assert key != b"subject-core-lineage"
    assert len(key) == 32


def test_two_installs_sign_the_same_lineage_differently(tmp_path):
    from core.identity.entity_key import EntityIdentity

    keys = [EntityIdentity(root=tmp_path / name).derived_key("subject-core lineage") for name in ("a", "b")]
    assert keys[0] != keys[1]
    lines = []
    for key in keys:
        line = Lineage(key=key)
        line.record("t1", _person())
        line.record("t2", _person())
        lines.append(line.descend("t1", "t2").signature)
    assert lines[0] != lines[1]


def test_her_commits_carry_what_wrote_each_change_and_the_log_reads_it_back():
    """From a measured phase writing her bonding, through two commits, to a developing step."""
    from core.runtime.cognitive_provenance import begin_transformation, close_tick, open_tick
    from core.state.aura_state import AuraState
    from core.state.person_transitions import PersonWatch

    watch = PersonWatch()
    first = AuraState.default()
    first.updated_at = 100.0
    first.transition_evidence = watch.evidence(first)

    second = first.derive("bonding")
    graph = open_tick(objective="Bryan said he trusts her")
    measuring = begin_transformation("BondingPhase", second)
    second.identity.bonding_level = float(second.identity.bonding_level or 0.0) + 0.1
    receipt = measuring.complete(second, publish_violation=False)
    close_tick(graph)
    assert "identity.bonding_level" in receipt.observed_writes
    second.updated_at = receipt.at + 1.0
    second.transition_evidence = watch.evidence(second)
    change = second.transition_evidence["last_change"]["identity.bonding_level"]
    assert change["transform"] == "BondingPhase" and change["objective"]

    rows = [
        {
            "state_id": state.state_id,
            "version": state.version,
            "parent_state_id": state.parent_state_id,
            "timestamp": at,
            "state_json": json.dumps({
                "identity": {"bonding_level": state.identity.bonding_level, "name": state.identity.name},
                "transition_evidence": state.transition_evidence,
            }, default=str),
        }
        for state, at in ((first, 0.0), (second, receipt.at + 1.0))
    ]
    line = lineage_from_state_log(rows)
    assert line.same_person(first.state_id, second.state_id) is True
    assert line.edges[0].changed == ("relationships:identity.bonding_level",)


def test_a_logged_state_whose_parent_was_pruned_descends_from_the_last_one_committed():
    rows = []
    for index, since in ((1, ""), (5, "s1")):
        rows.append({
            "state_id": f"s{index}",
            "version": index,
            "parent_state_id": f"s{index - 1}",
            "timestamp": float(index),
            "state_json": json.dumps({"identity": {"name": "Aura"}, "transition_evidence": {"since_state": since}}),
        })
    line = lineage_from_state_log(rows)
    assert [(e.parent, e.child) for e in line.edges] == [("s1", "s5")]
    assert line.same_person("s1", "s5")
