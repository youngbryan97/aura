"""The forge's target stays still while the forge takes aim at it."""
from __future__ import annotations

import pytest

from core.skill_management.skill_verification import Probe
from core.skill_management.what_the_probes_were_held_to import (
    TheTargetForOneForge,
    a_fingerprint_of,
    how_the_target_has_held,
    reset_how_the_target_has_held,
)


@pytest.fixture(autouse=True)
def _a_clean_count():
    reset_how_the_target_has_held()
    yield
    reset_how_the_target_has_held()


def _probe(n: int, expect: object = None) -> Probe:
    if expect is None:
        return Probe.of({"n": n}, expect_keys=("ok",), label=f"p{n}")
    return Probe.of({"n": n}, expect=expect, label=f"p{n}")


def test_the_first_draft_fixes_the_target():
    target = TheTargetForOneForge(skill="a_skill")
    assert not target.fixed
    held = target.admit(1, (_probe(1), _probe(2)))
    assert target.fixed
    assert len(held) == 2
    assert target.history == []


def test_an_empty_first_offer_leaves_the_target_unfixed():
    """A draft with no usable probe sets nothing; the next one still can."""
    target = TheTargetForOneForge()
    assert target.admit(1, ()) == ()
    assert not target.fixed
    target.admit(2, (_probe(1),))
    assert target.fixed


def test_a_redraft_offering_the_same_probes_held_the_target():
    target = TheTargetForOneForge()
    target.admit(1, (_probe(1), _probe(2)))
    target.admit(2, (_probe(1), _probe(2)))
    assert not target.ever_moved
    assert target.history[-1].verdict == "held"


def test_a_rewritten_expectation_is_refused_and_counted():
    """The defect this exists for: answer a failing test by changing it."""
    target = TheTargetForOneForge(skill="a_skill")
    first = target.admit(1, (_probe(1, expect={"ok": True, "value": 42}),))
    held = target.admit(2, (_probe(1, expect={"ok": True, "value": 0}),))
    assert target.ever_moved
    assert target.history[-1].altered == 1
    assert target.history[-1].dropped == 0
    # The original expectation is still in the set it will be verified against.
    assert first[0] in held


def test_a_dropped_probe_is_counted_as_dropped():
    target = TheTargetForOneForge()
    target.admit(1, (_probe(1), _probe(2)))
    held = target.admit(2, (_probe(1),))
    assert len(held) == 2
    assert target.history[-1].dropped == 1
    assert target.history[-1].verdict == "tried to drop"


def test_an_added_probe_raises_the_bar_and_is_kept():
    """More tests can only make it harder, so an addition is admitted."""
    target = TheTargetForOneForge()
    target.admit(1, (_probe(1),))
    held = target.admit(2, (_probe(1), _probe(2)))
    assert len(held) == 2
    assert target.history[-1].added == 1
    assert not target.ever_moved


def test_replacing_the_whole_set_reads_as_one_rewrite_and_one_drop():
    target = TheTargetForOneForge()
    target.admit(1, (_probe(1), _probe(2)))
    target.admit(2, (_probe(9),))
    last = target.history[-1]
    assert (last.altered, last.dropped, last.added) == (1, 1, 0)
    assert last.moved_it


def test_the_fingerprint_ignores_order_and_notices_an_expectation():
    a, b = _probe(1), _probe(2)
    assert a_fingerprint_of((a, b)) == a_fingerprint_of((b, a))
    assert a_fingerprint_of((_probe(1, expect={"ok": True}),)) != a_fingerprint_of(
        (_probe(1, expect={"ok": False}),)
    )
    assert a_fingerprint_of(()) == ""


def test_no_redraft_means_nothing_was_measured():
    read = how_the_target_has_held()
    assert read["redrafts"] == 0
    assert "no forge has redrafted" in read["reading"]


def test_the_count_names_how_many_redrafts_moved_their_own_target():
    target = TheTargetForOneForge()
    target.admit(1, (_probe(1),))
    target.admit(2, (_probe(1),))
    target.admit(3, (_probe(2),))
    read = how_the_target_has_held()
    assert read["redrafts"] == 2
    assert read["moved_the_target"] == 1
    assert read["by_verdict"]["held"] == 1


def test_the_record_names_the_skill_and_survives_serialisation():
    target = TheTargetForOneForge(skill="thing_maker")
    target.admit(1, (_probe(1),))
    target.admit(2, (_probe(2),))
    read = target.as_dict()
    assert read["skill"] == "thing_maker"
    assert read["ever_moved"] is True
    assert read["fingerprint"]
    assert read["attempts"] and "attempt 2" in read["attempts"][0]
