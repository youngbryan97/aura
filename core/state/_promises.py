"""What core.state guarantees, and the test that catches each breaking."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="An interruption recorded in one process can be resumed in another, "
        "with its checkpoint and what it was about to do.",
        checked_by="tests/test_stopping_and_starting_again.py::"
        "test_an_interruption_survives_the_process_that_recorded_it",
        if_it_fails="the work is unresumable and starting again is starting "
        "over; what_was_interrupted() reports nothing after a restart",
    ),
    APromise(
        it="A reason for stopping that will not clear by itself is refused a "
        "resume rather than retried into the same wall.",
        checked_by="tests/test_stopping_and_starting_again.py::"
        "test_a_refusal_is_not_resumed",
        if_it_fails="a refusal becomes a retry loop; the interruption's "
        "resumable flag is the record of which it was",
    ),
    APromise(
        it="Phases in one parallel group read the state as it was when the "
        "group began, never a sibling's half-finished write.",
        checked_by="tests/test_what_they_all_read.py::"
        "test_a_sibling_write_is_invisible_until_the_barrier",
        if_it_fails="the turn is not reproducible from its inputs; "
        "how_the_supersteps_have_gone() reports the conflicting fields",
    ),
    APromise(
        it="Two phases writing one field to different values is named as a "
        "conflict rather than settled by whichever finished last.",
        checked_by="tests/test_what_they_all_read.py::"
        "test_two_phases_writing_the_same_field_differently_is_named_not_picked",
        if_it_fails="TheyDisagreed is raised and nothing is committed; in "
        "lenient mode the clash is recorded and settled by sorted phase name",
    ),
    APromise(
        it="A checkpoint restored from a store is refused when its content no "
        "longer matches the digest recorded when it was written.",
        checked_by="tests/test_where_checkpoints_are_kept.py::"
        "test_a_state_changed_underneath_is_refused",
        if_it_fails="a corrupted state would be restored as though intact; the "
        "store raises rather than returning it",
    ),
)
