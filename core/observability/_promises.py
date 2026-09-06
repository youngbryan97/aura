"""What core.observability guarantees, and the test that catches each breaking."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="A number declares the lifetime it is measured over, so a count "
        "cannot be compared against one that resets on a different schedule.",
        checked_by="tests/test_how_long_a_number_lives.py::"
        "test_a_counter_that_does_not_say_what_it_counts_is_refused",
        if_it_fails="declare_a_number raises; an undeclared counter would be "
        "read as a lifetime total and reset every turn",
    ),
    APromise(
        it="Resetting one lifetime domain touches only that domain, so ending "
        "a turn cannot clear what a session was counting.",
        checked_by="tests/test_how_long_a_number_lives.py::"
        "test_resetting_a_domain_touches_only_that_domain",
        if_it_fails="the other domains' numbers go to zero without anything "
        "having ended; visible as a session counter that never grows",
    ),
    APromise(
        it="Two values from different clocks cannot be subtracted, so a "
        "duration is never a plausible number from mixed sources.",
        checked_by="tests/test_which_clock_is_this.py::"
        "test_subtracting_across_domains_is_refused_not_silently_plausible",
        if_it_fails="MixedClocks is raised; the three tree scans report where "
        "a mix would be possible",
    ),
    APromise(
        it="One trace id survives a spawned task, and the boundaries that drop "
        "it are named rather than assumed to carry it.",
        checked_by="tests/test_does_one_trace_reach_the_end.py::"
        "test_it_does_not_follow_a_bare_thread_or_an_executor",
        if_it_fails="how_far_a_trace_reaches() reports which hop lost it; "
        "receipts on the far side carry no trace at all",
    ),
)
