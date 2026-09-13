"""The five transfer cases, and the one thing that cannot branch.

Identity is transitive. If one earlier stage were numerically identical to two
distinct later stages, those two would be identical to each other, and in a
fork they are not. So a lineage graph is the primitive and strict identity is
the case where the graph does not branch. Every test here is that theorem
applied to one scenario:

    unique continuation      one child, same person
    migration                one child through another process, same person
    destructive restore      one child after a dormant interval, same person
    nondestructive fork      two children, neither the strict successor
    stale backup restored    a branch from the older stage
    exact reconstruction     no causal channel, so a new token

Whether continuation is what matters is the causal-lineage postulate. These
tests check the graph, not the postulate.
"""

from __future__ import annotations

import pytest

from core.subject.lineage import (
    KINDS,
    NONBRANCHING,
    Lineage,
    canonical_person_state,
    state_digest,
)

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
        # Bookkeeping, which must not make two stages differ.
        "timestamp": 1.0,
        "run": "abc",
    }
    base.update(overrides)
    return base


# ── what counts as the person ─────────────────────────────────────────────


def test_bookkeeping_is_not_part_of_the_person() -> None:
    """A clock moving does not make her someone else."""
    assert state_digest(_person(timestamp=1.0)) == state_digest(_person(timestamp=99.0, run="xyz"))


def test_an_identity_bearing_field_is_part_of_the_person() -> None:
    assert state_digest(_person()) != state_digest(_person(values=["honesty"]))


def test_a_field_the_runtime_does_not_carry_is_absent_rather_than_defaulted() -> None:
    """A default would make two people who lack a field agree about it."""
    thin = canonical_person_state({"values": ["honesty"]})
    assert set(thin) == {"values"}


def test_a_float_drifting_in_its_last_bits_is_the_same_person() -> None:
    assert state_digest(_person()) == state_digest(
        _person(self_model={"name": "Aura", "efficacy": 0.7 + 1e-12})
    )


# ── the scenarios ─────────────────────────────────────────────────────────


def test_uninterrupted_continuation_is_one_child() -> None:
    line = Lineage()
    line.record("monday", _person())
    line.record("tuesday", _person())
    line.descend("monday", "tuesday")
    assert line.children("monday") == ("tuesday",)
    assert line.same_person("monday", "tuesday") is True


def test_migration_with_a_unique_successor_continues_the_person() -> None:
    """A changed substrate with one continuing lineage."""
    line = Lineage()
    line.record("on_this_host", _person())
    line.record("on_that_host", _person())
    line.descend("on_this_host", "on_that_host", kind="migrate")
    assert line.same_person("on_this_host", "on_that_host") is True


def test_a_destructive_restore_after_a_dormant_interval_continues_the_person() -> None:
    """A long gap does not matter when the state is preserved and unique."""
    line = Lineage()
    line.record("before", _person())
    line.record("after", _person(), at=10_000.0)
    line.descend("before", "after", kind="restore")
    assert line.same_person("before", "after") is True


def test_a_fork_has_two_continuers_and_no_strict_successor() -> None:
    line = Lineage()
    line.record("original", _person())
    line.record("left", _person())
    line.record("right", _person())
    line.descend("original", "left", kind="fork")
    line.descend("original", "right", kind="fork")

    assert set(line.continuers("original")) == {"left", "right"}
    assert line.branched("original") is True
    assert line.same_person("original", "left") is False
    assert line.same_person("original", "right") is False


def test_a_stale_backup_restored_is_a_branch_from_the_older_stage() -> None:
    """Monday backs up, Tuesday to Friday continue, Saturday restores Monday."""
    line = Lineage()
    for day in ("monday", "friday", "saturday"):
        line.record(day, _person())
    line.descend("monday", "friday")
    line.descend("monday", "saturday", kind="restore")

    assert line.branched("monday") is True
    assert set(line.continuers("monday")) == {"friday", "saturday"}
    assert line.same_person("monday", "friday") is False
    assert line.same_person("friday", "saturday") is False


def test_an_exact_reconstruction_with_no_channel_is_a_new_token() -> None:
    """Same structure, no ancestry. The graph refuses to invent an edge."""
    line = Lineage()
    line.record("original", _person())
    line.record("rebuilt_a_trillion_years_later", _person())
    assert state_digest(_person()) == state_digest(_person())
    assert line.path("original", "rebuilt_a_trillion_years_later") is None
    assert line.same_person("original", "rebuilt_a_trillion_years_later") is False
    with pytest.raises(ValueError):
        line.descend("original", "rebuilt_a_trillion_years_later", kind="reconstruct")


def test_two_simultaneous_copies_are_two_tokens() -> None:
    line = Lineage()
    line.record("source", _person())
    line.record("copy_one", _person())
    line.record("copy_two", _person())
    line.descend("source", "copy_one", kind="fork")
    line.descend("source", "copy_two", kind="fork")
    assert line.same_person("copy_one", "copy_two") is False


# ── the rules that make those answers follow ──────────────────────────────


def test_a_transfer_that_did_not_preserve_the_person_is_not_continuation() -> None:
    line = Lineage()
    line.record("before", _person())
    line.record("after", _person(values=["something else entirely"]))
    edge = line.descend("before", "after")
    assert edge.preserved is False
    assert line.same_person("before", "after") is False


def test_a_fork_is_not_one_of_the_nonbranching_kinds() -> None:
    assert "fork" in KINDS
    assert "fork" not in NONBRANCHING
    assert NONBRANCHING <= set(KINDS)


def test_identity_cannot_branch_even_when_both_children_preserve_the_state() -> None:
    """The theorem, stated as a test: both are genuine continuers, neither is it."""
    line = Lineage()
    line.record("a", _person())
    line.record("b", _person())
    line.record("c", _person())
    line.descend("a", "b", kind="fork")
    line.descend("a", "c", kind="fork")
    both_preserve = all(e.preserved for e in line.edges)
    assert both_preserve
    assert line.same_person("a", "b") is False and line.same_person("a", "c") is False


def test_a_later_branch_breaks_identity_along_the_earlier_chain() -> None:
    """Preserving two equally good continuers makes strict identity fail."""
    line = Lineage()
    for name in ("one", "two", "three", "other"):
        line.record(name, _person())
    line.descend("one", "two")
    line.descend("two", "three")
    assert line.same_person("one", "three") is True
    line.descend("two", "other", kind="fork")
    assert line.same_person("one", "three") is False


# ── the record cannot be rewritten afterwards ─────────────────────────────


def test_every_edge_is_signed_over_what_it_covers() -> None:
    line = Lineage()
    line.record("before", _person())
    line.record("after", _person())
    line.descend("before", "after")
    assert line.verify() == []
    assert line.as_dict()["signatures_verify"] is True


def test_rewriting_a_stage_breaks_its_signature() -> None:
    from dataclasses import replace

    line = Lineage()
    line.record("before", _person())
    line.record("after", _person())
    line.descend("before", "after")
    line.stages["after"] = replace(line.stages["after"], digest="0" * 32)
    assert line.verify(), "a rewritten stage left every signature matching"


def test_recording_the_same_stage_twice_is_refused() -> None:
    line = Lineage()
    line.record("only_once", _person())
    with pytest.raises(ValueError):
        line.record("only_once", _person())


def test_an_edge_to_a_stage_that_was_never_recorded_is_refused() -> None:
    line = Lineage()
    line.record("here", _person())
    with pytest.raises(KeyError):
        line.descend("here", "nowhere")


def test_the_report_names_the_postulate() -> None:
    """Whether continuation is what matters is not something this measures."""
    line = Lineage()
    line.record("one", _person())
    assert "postulate" in line.as_dict()["postulate"]


def test_continuity_strength_is_reported_beside_the_graph_not_instead_of_it() -> None:
    """It can be high across a fork, which is why the graph is the primitive."""
    from core.subject.lineage import continuity_strength, scenarios

    line = Lineage()
    for name in ("root", "left", "right"):
        line.record(name, _person())
    line.descend("root", "left", kind="fork")
    line.descend("root", "right", kind="fork")

    assert continuity_strength(line, "root", "left") == pytest.approx(1.0)
    assert line.same_person("root", "left") is False
    report = scenarios(line, ["root"])
    assert report["root"]["continuity_strength"]["left"] == pytest.approx(1.0)
    assert report["root"]["branched"] is True


def test_continuity_strength_is_zero_with_no_causal_channel() -> None:
    from core.subject.lineage import continuity_strength

    line = Lineage()
    line.record("original", _person())
    line.record("rebuilt", _person())
    assert continuity_strength(line, "original", "rebuilt") == 0.0


def test_a_broken_step_drops_the_strength() -> None:
    from core.subject.lineage import continuity_strength

    line = Lineage()
    line.record("a", _person())
    line.record("b", _person())
    line.record("c", _person(values=["changed"]))
    line.descend("a", "b")
    line.descend("b", "c")
    assert continuity_strength(line, "a", "c") == pytest.approx(0.5)
