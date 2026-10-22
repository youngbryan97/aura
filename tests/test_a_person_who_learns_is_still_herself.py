"""A person who learns is still herself, when the change has the right cause.

A reviewer reproduced it on 21 October 2026: a preference at t1, changed by
experience at t2, the two joined by a signed transition that verifies, and
`same_person(t1, t2)` came back False, because every identity-preserving step
had to keep the canonical state's hash. Any learning ended identity. A step now
develops when each path it changed was written by a measured transformation
that kept to its contract. A second review the same day showed that a rule on
the share of paths carried through turns on how the person is divided into
paths, and that the commit digest stopped reading at 4,096 characters; both are
pinned here.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from core.subject.lineage import Lineage, lineage_from_state_log

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


def test_the_verdict_does_not_turn_on_how_the_person_is_divided_into_paths():
    """One kept path and one changed: a half. The same change as two paths: a third. One verdict."""
    two = Lineage()
    two.record("t1", {"values": ["honesty"], "preferences": {"depth": 0.8}}, at=1.0)
    two.record("t2", {"values": ["honesty"], "preferences": {"depth": 0.6}}, at=3.0)
    coarse = two.descend("t1", "t2", writes={"preferences:depth": _wrote()})
    three = Lineage()
    three.record("t1", {"values": ["honesty"], "preferences": {"depth": 0.8, "depth_again": 0.8}}, at=1.0)
    three.record("t2", {"values": ["honesty"], "preferences": {"depth": 0.6, "depth_again": 0.6}}, at=3.0)
    fine = three.descend(
        "t1", "t2", writes={"preferences:depth": _wrote(), "preferences:depth_again": _wrote()}
    )
    assert coarse.connectedness == pytest.approx(1 / 2) and fine.connectedness == pytest.approx(1 / 3)
    assert two.same_person("t1", "t2") == three.same_person("t1", "t2") is True


def test_the_relation_is_reported_in_its_parts():
    line = _two_stages(_person(preferences={"depth": 0.6}))
    line.descend("t1", "t2", writes={"preferences:depth": _wrote()})
    assert line.relations("t1", "t2") == {
        "chain_continuity": True,
        "change_attributed": True,
        "unique_successor": True,
        "state_preserved": False,
        "same_person": True,
    }


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


def test_a_change_past_the_first_4096_characters_is_still_a_change():
    from core.state.person_transitions import canonical_digest

    first, second = "x" * 5000 + "A", "x" * 5000 + "B"
    assert canonical_digest(first) != canonical_digest(second)
    other = Lineage()
    other.record("t1", _person(autobiographical=[first]), at=1.0)
    other.record("t2", _person(autobiographical=[second]), at=3.0)
    assert other.descend("t1", "t2").changed == ("autobiographical",)


def test_the_order_keys_were_written_in_is_not_a_change():
    from core.state.person_transitions import canonical_digest

    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})


def test_a_value_with_no_canonical_form_is_refused_and_left_unjudged():
    from core.state.person_transitions import UnmeasurableError, canonical_digest

    with pytest.raises(UnmeasurableError):
        canonical_digest(object())
    line = Lineage()
    line.record("t1", _person(policy={"defer_below": object()}), at=1.0)
    line.record("t2", _person(policy={"defer_below": object()}), at=3.0)
    edge = line.descend("t1", "t2")
    assert edge.unjudged() == ("policy:defer_below",)
    assert line.development()["unmeasured"] == 1 and line.development()["failed"] == 0


def test_a_phase_writing_past_the_first_4096_characters_leaves_a_receipt():
    import core.state.person_transitions  # noqa: F401 - registers the person paths
    from core.runtime.cognitive_provenance import begin_transformation
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.identity.current_narrative = "x" * 5000 + "A"
    measuring = begin_transformation("NarrativePhase", state)
    state.identity.current_narrative = "x" * 5000 + "B"
    receipt = measuring.complete(state, publish_violation=False)
    assert "identity.current_narrative" in receipt.observed_writes


def test_a_restored_backup_beside_a_branch_that_went_on_is_not_one_person():
    line = Lineage()
    line.record("monday", _person(), at=1.0)
    line.record("friday", _person(preferences={"depth": 0.6}), at=5.0)
    line.record("saturday", _person(), at=6.0)
    line.descend("monday", "friday", writes={"preferences:depth": _wrote()})
    line.descend("monday", "saturday", kind="restore")
    assert line.relations("monday", "saturday")["unique_successor"] is False
    assert line.same_person("monday", "friday") is False


def test_her_line_holds_since_the_record_began_and_counts_what_came_before():
    from core.subject.bridge import lineage_term

    line = Lineage()
    line.record("august", _person(preferences={"depth": 0.2}), at=1.0)
    line.record("september", _person(preferences={"depth": 0.4}), at=2.0)
    line.record("today", _person(preferences={"depth": 0.6}), at=3.0)
    line.record("tonight", _person(preferences={"depth": 0.7}), at=4.0)
    line.descend("august", "september", measured=False)
    line.descend("september", "today", measured=False)
    line.descend("today", "tonight", writes={"preferences:depth": _wrote()})
    development = line.development()
    assert development["before_the_record"] == 2
    assert development["steps_since_the_record"] == 1 and development["developing"] == 1
    assert lineage_term(line).status == "RECORDED"


def test_a_line_with_only_unrecorded_changes_is_not_yet_measured():
    from core.subject.bridge import lineage_term

    line = Lineage()
    line.record("august", _person(preferences={"depth": 0.2}), at=1.0)
    line.record("september", _person(preferences={"depth": 0.4}), at=2.0)
    line.descend("august", "september", measured=False)
    assert lineage_term(line).status == "NOT_MEASURED"


def test_unchanged_steps_after_the_last_unrecorded_change_are_not_development():
    """Her live log on 21 October: no step carried a record, and her newest
    stage sat three unchanged steps after the last change. Those steps say
    nothing about what wrote her, so the term is not yet measured."""
    from core.subject.bridge import lineage_term

    line = Lineage()
    line.record("august", _person(preferences={"depth": 0.2}), at=1.0)
    line.record("september", _person(preferences={"depth": 0.4}), at=2.0)
    line.record("evening", _person(preferences={"depth": 0.4}), at=3.0)
    line.record("night", _person(preferences={"depth": 0.4}), at=4.0)
    line.descend("august", "september", measured=False)
    line.descend("september", "evening", measured=False)
    line.descend("evening", "night", measured=False)
    development = line.development()
    assert development["steps_since_the_record"] == 0
    assert development["before_the_record"] == 1
    assert lineage_term(line).status == "NOT_MEASURED"
