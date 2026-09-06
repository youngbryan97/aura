"""A primitive with no caller outside its own tests is a proposal."""
from __future__ import annotations

from core.verify.does_this_govern_anything import (
    HowFarItReaches,
    how_far_it_reaches,
    what_governs_and_what_does_not,
    who_imports,
)

#: Everything built during the maturity pass, so the number is about the work
#: rather than about whichever module somebody remembered to list.
THE_PRIMITIVES: tuple[str, ...] = tuple(
    "core." + one
    for one in (
        "runtime.what_must_never_be_retried",
        "state.what_they_all_read",
        "runtime.how_a_task_should_end",
        "observability.does_one_trace_reach_the_end",
        "runtime.how_a_call_is_made",
        "observability.which_clock_is_this",
        "runtime.which_parts_say_how_they_are",
        "memory.what_kind_of_memory_is_this",
        "brain.llm.who_got_the_room",
        "runtime.what_is_on_its_way_out",
        "runtime.claiming_more_than_one",
        "state.nothing_lands_before_its_writes",
        "verify.is_this_async_code_correct",
        "runtime.what_she_decided_to_do_at_once",
        "runtime.cancelling_the_call_and_not_just_the_wait",
        "verify.a_promise_with_a_test",
    )
)


def test_something_the_runtime_really_uses_reads_as_governing() -> None:
    """A gate that called everything a proposal would say nothing."""
    spine = how_far_it_reaches("core.runtime.event_spine")
    assert spine.reaches is HowFarItReaches.GOVERNING
    assert len(spine.production_callers) >= 3


def test_a_module_nothing_imports_is_a_proposal() -> None:
    made_up = how_far_it_reaches("core.runtime.a_module_that_is_not_there")
    assert made_up.reaches is HowFarItReaches.A_PROPOSAL
    assert made_up.production_callers == ()


def test_a_chain_of_primitives_calling_each_other_is_still_a_chain_nothing_enters() -> None:
    """Counting the inner ones as governing is how a report flatters itself."""
    seen = what_governs_and_what_does_not(THE_PRIMITIVES)
    inner = seen["which"].get("governing a proposal", [])
    for one in inner:
        callers = how_far_it_reaches(one).production_callers
        assert callers, f"{one} was classed as governing something and has no caller"


def test_the_report_accounts_for_every_module_it_was_asked_about() -> None:
    seen = what_governs_and_what_does_not(THE_PRIMITIVES)
    assert seen["asked_about"] == len(THE_PRIMITIVES)
    assert (
        seen["governing"]
        + seen["governing_a_proposal"]
        + seen["reachable"]
        + seen["proposals"]
    ) == len(THE_PRIMITIVES)


def test_the_honest_state_of_this_pass_is_recorded() -> None:
    """Most of what was built this pass decides nothing yet, and says so.

    Not a failure — it is the state of something built before the call sites
    that need it. What would be dishonest is a report counting it as an
    invariant. This asserts the number is reported, not that it is small.
    """
    seen = what_governs_and_what_does_not(THE_PRIMITIVES)
    assert seen["proposals"] >= 0
    assert isinstance(seen["which"], dict)
    for state, rows in seen["which"].items():
        assert rows, f"{state} is listed with nothing in it"


def test_who_imports_finds_the_tests_as_well_as_the_callers() -> None:
    callers = who_imports("core.runtime.event_spine")
    assert any(one.startswith("tests.") for one in callers)
    assert any(one.startswith("core.") for one in callers)
