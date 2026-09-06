"""What core.brain guarantees, and the test that catches each breaking."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="Every semantic route declares whether it is authoritative, "
        "qualified or shadow, so no route decides an answer without saying "
        "what standing it has.",
        checked_by="tests/test_which_routes_are_authoritative.py::"
        "test_every_route_is_in_one_of_the_three_states",
        if_it_fails="a shadow route's output reads the same as an "
        "authoritative one, and a wrong answer has no standing to lose",
    ),
    APromise(
        it="Every route says why it is in the state it declares, so the "
        "standing can be argued with rather than only read.",
        checked_by="tests/test_which_routes_are_authoritative.py::"
        "test_every_route_says_why_it_is_in_that_state",
        if_it_fails="the state is an assertion nobody can check, and promoting "
        "a route becomes a matter of editing one word",
    ),
    APromise(
        it="A provider measured as a leaf when it is really a fallback chain "
        "is reported as broken rather than as keeping the promise.",
        checked_by="tests/test_what_a_provider_promises.py::"
        "test_a_chain_measured_as_a_leaf_is_reported_as_broken",
        if_it_fails="a chain passes a leaf's suite and the fallback it hides "
        "is never exercised by anything",
    ),
    APromise(
        it="A prompt fitted to a budget records what each named part had and "
        "what it kept, so a squeeze has an owner.",
        checked_by="tests/test_who_got_the_room.py::"
        "test_the_assembler_records_who_paid_for_the_fit",
        if_it_fails="a thin turn and a turn whose memory was cut to nothing "
        "leave the same trace, which is a shorter prompt",
    ),
)
