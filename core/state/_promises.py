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
        it="A write is not in its channel until it is committed, so a reader "
        "between the write and the barrier sees the old value rather than a "
        "half-applied one.",
        checked_by="tests/test_a_checkpoint_and_its_writes.py::"
        "test_a_write_is_not_in_its_channel_until_it_is_committed",
        if_it_fails="a checkpoint taken mid-phase holds a value no phase ever "
        "agreed on, and restoring it restores a state that never existed",
    ),
    APromise(
        it="A difference between two states says exactly where it is, rather "
        "than only that the two are not identical.",
        checked_by="tests/test_are_these_the_same.py::"
        "test_a_difference_says_where_it_is",
        if_it_fails="a hash mismatch is all a caller gets, and a byte "
        "difference cannot be told from a meaningful one",
    ),
    APromise(
        it="An observation channel that never declared how it is read cannot "
        "be read at all.",
        checked_by="tests/test_when_an_observation_was_true.py::"
        "test_a_channel_that_never_declared_cannot_be_read",
        if_it_fails="the refusal names the modes it could have declared; "
        "reading it anyway compares values from different kinds of time",
    ),
    APromise(
        it="Working memory has one capacity, and it is the one the trimmer "
        "actually enforces.",
        checked_by="tests/test_one_working_memory.py::"
        "test_the_capacity_is_the_one_the_trimmer_enforces",
        if_it_fails="the_caps_that_disagree() names them; a turn is trimmed to "
        "one number while another part of the runtime plans against another",
    ),
    APromise(
        it="Every state holder is classified as authority, projection or "
        "scratch, so nothing is authoritative by default.",
        checked_by="tests/test_what_kind_of_state_is_this.py::"
        "test_nothing_is_unclassified",
        if_it_fails="what_is_not_classified() names the holder; two places "
        "would each be able to claim they hold the real value",
    ),
    APromise(
        it="A projection names the authority it is derived from, so a stale "
        "one can be traced to what should have refreshed it.",
        checked_by="tests/test_what_kind_of_state_is_this.py::"
        "test_every_projection_names_the_authority_it_comes_from",
        if_it_fails="the projection is unattributed and a disagreement between "
        "it and its source has no arbiter",
    ),
    APromise(
        it="A field that was owned by something does not silently become "
        "unowned, so ownership is only ever taken up.",
        checked_by="tests/test_who_owns_each_field.py::"
        "test_no_field_becomes_unowned_that_was_owned",
        if_it_fails="the ownership baseline names the field; two writers can "
        "then disagree with nothing saying which is right",
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
