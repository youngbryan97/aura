"""What core.conversation guarantees, and the test that catches each breaking."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="A pattern does not decide from one bare token that has been wrong "
        "more than once; the token has to appear in a role.",
        checked_by="tests/test_a_concept_formed_from_the_failures.py::"
        "test_the_violations_only_go_down",
        if_it_fails="the constraint's violation count rises; a remark reads as "
        "a request, and a question containing an action word stops being one",
    ),
    APromise(
        it="A value has to be the whole of a text region to count as the thing "
        "being waited for, rather than appearing inside one.",
        checked_by="tests/test_a_condition_already_true_is_not_an_achievement.py::"
        "test_a_number_inside_a_label_is_not_the_thing_being_waited_for",
        if_it_fails="'128' inside 'SCORE 128' finishes a goal that was about a "
        "tile, and the run reports work nobody did",
    ),
)
