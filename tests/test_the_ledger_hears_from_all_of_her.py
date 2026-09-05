"""A universal language hooked to an optimiser fed by one library.

The developmental policy picks what to change about herself from a record of
what her own work has cost. Almost everything in that record came from one
cognitive ecology — rule, sequence and representation induction — because
those were the only paths that called `note_an_episode`.

That is a scope problem rather than a language problem. The developmental
language can express a change to any part of her; the developmental agent
only ever saw evidence from one library, so it could not choose to fix a slow
retrieval, a wasteful route, a motor policy that flails or a verification
step that never fires, however clearly it could have expressed the fix.

Two intakes, because cost and failure arrive differently. Failure already has
a universal sink and attaching there made the failure half organism-wide with
no edit to any subsystem. Cost has to be reported by whoever spends it, and
the wrapper for that has to be trivial, because an intake that costs anything
to adopt does not get adopted — which is exactly how the last one ended up
adopted once.
"""

from __future__ import annotations

import pytest

import core.cognition.what_the_whole_organism_costs as ledger
from core.cognition.the_record_of_her_own_work import episodes, forget_the_record
from core.runtime.errors import get_degradation_tracker, record_degradation


@pytest.fixture(autouse=True)
def a_clean_ledger(tmp_path, monkeypatch):
    import core.cognition.the_record_of_her_own_work as record

    monkeypatch.setattr(record, "_KEPT_AT", tmp_path / "record.json")
    monkeypatch.setattr(record, "_RESTORED", [True])
    forget_the_record()
    ledger.forget_what_was_heard()
    yield
    ledger.forget_what_was_heard()
    forget_the_record()


def test_every_declared_part_of_her_has_a_name_the_intake_knows():
    """A part nothing can be filed under is a part that never reports."""

    filed = set(ledger._WHICH_PART.values())
    for part in ledger.WHAT_SHE_IS_MADE_OF:
        if part == "rule induction":
            continue  # reports through note_an_episode directly, by name
        assert part in filed, f"nothing can be filed under {part!r}"


def test_a_failure_anywhere_becomes_developmental_evidence():
    tracker = get_degradation_tracker()
    assert tracker.add_listener(ledger.note_a_failure) or True
    ledger.note_a_failure(
        type("R", (), {"subsystem": "screen_perception", "severity": "degraded", "action": "skipped"})()
    )
    assert "perception" in ledger.coverage(read_the_quiet_ones=False)["heard_from"]
    assert any(one.family == "perception" for one in episodes())


def test_the_intake_covers_the_organism_rather_than_one_library():
    """The measurement behind the criticism, in terms that can be argued with."""

    for subsystem in (
        "screen_perception",
        "memory_facade",
        "consolidation",
        "llm_client",
        "latent_cortex",
        "capability_engine",
        "desktop_action_gateway",
        "interpersonal_observer",
        "service_container",
    ):
        ledger.note_a_failure(
            type("R", (), {"subsystem": subsystem, "severity": "degraded", "action": ""})()
        )
    seen = ledger.coverage(read_the_quiet_ones=False)
    assert seen["share"] >= 0.7, seen["missing"]
    for part in (
        "perception", "retrieval", "memory consolidation", "model routing",
        "latent recurrence", "tool planning", "motor policy", "social inference",
        "architecture",
    ):
        assert part in seen["covered"]


def test_a_dotted_subsystem_name_still_finds_its_part():
    """Losing one to a suffix would narrow the intake again by accident."""

    assert ledger.which_part("memory.interpersonal_store.render") == "social inference"
    assert ledger.which_part("core.perception.screen_perception") == "perception"
    assert ledger.which_part("something nobody mapped") == "something nobody mapped"


def test_demoted_noise_is_not_evidence():
    """Otherwise the loudest ecology is the one with the chattiest logging."""

    ledger.note_a_failure(
        type("R", (), {"subsystem": "screen_perception", "severity": "debug", "action": ""})()
    )
    assert ledger.coverage(read_the_quiet_ones=False)["episodes"] == 0


def test_the_wrapper_reports_what_work_cost():
    with ledger.while_doing("llm_client", "routing one turn") as said:
        said["admitted"] = "the smaller model"
    written = [one for one in episodes() if one.family.startswith("model routing")]
    assert written
    assert written[-1].admitted == "the smaller model"
    assert written[-1].route == "an answer"


def test_work_that_raised_is_reported_as_having_answered_nothing():
    """Route names the action only when the change was kept, and a policy
    that cannot see a failed attempt cannot tell "tried everything" from
    "never tried"."""

    with pytest.raises(ValueError):
        with ledger.while_doing("capability_engine", "planning a tool call"):
            raise ValueError("the plan did not come out")
    written = [one for one in episodes() if one.family.startswith("tool planning")]
    assert written[-1].route is None
    assert written[-1].tried


def test_a_subsystem_that_cannot_count_its_search_still_counts_its_clock():
    """An episode with no cost is an episode the policy cannot rank."""

    with ledger.while_doing("memory_facade", "one retrieval"):
        pass
    written = [one for one in episodes() if one.family.startswith("retrieval")]
    assert written[-1].walked >= 0


def test_the_quiet_ones_are_read_rather_than_waited_for():
    """A refused value change is the governance working, not a fault.

    Recording correct refusals as degradations is the mistake a whole test
    file in this tree is named after, so verification and governance are read
    rather than asked to report themselves as damaged.
    """

    import core.verify.runtime_invariants  # noqa: F401
    from core.governance.value_levels import Change, registry
    from core.values.what_she_holds import declare_what_she_holds

    declare_what_she_holds()
    registry().may_change(Change(value="honesty", process="preference_learner"))
    registry().apply(
        Change(value="honesty", process="preference_learner"), "anything at all"
    )
    took = ledger.read_what_does_not_report()
    assert "verification" in took
    assert "governance" in took
    seen = ledger.coverage(read_the_quiet_ones=False)
    assert {"verification", "governance"} <= set(seen["covered"])


def test_perception_reports_what_finding_out_cost():
    """The first success-path adoption, so the ledger sees cost and not only
    failure."""

    from core.perception.how_she_finds_out import (
        WayOfFindingOut,
        clear_the_inventory,
        find_out,
        register_a_way,
    )

    clear_the_inventory()
    register_a_way(
        WayOfFindingOut(
            name="look", about=("x",), cost=0.01,
            outcomes=("a", "b"), take=lambda _s: "a", right=40,
        )
    )
    find_out("x", {"a": 0.5, "b": 0.5}, draw=lambda _a, _b: 0.95)
    clear_the_inventory()
    written = [one for one in episodes() if one.family.startswith("perception")]
    assert written
    assert written[-1].admitted


def test_the_intake_attaches_without_anybody_remembering_to():
    """The last one had to be called and never was."""

    import inspect

    import core.cognition.the_record_of_her_own_work as record

    source = inspect.getsource(record._remember_what_she_had)
    assert "hear_from_every_subsystem" in source
