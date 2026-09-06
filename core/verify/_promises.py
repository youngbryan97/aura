"""What core.verify guarantees, and the test that catches each breaking."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="A faculty that cannot be lesioned is not counted as measured, so "
        "coverage never includes what could not have been tested.",
        checked_by="tests/test_what_has_a_measured_effect.py::"
        "test_a_faculty_that_cannot_be_lesioned_is_not_counted_as_unmeasured",
        if_it_fails="the measured proportion rises without an experiment "
        "having run; how_much_is_measured() is where it would show",
    ),
    APromise(
        it="The count of declared lesionable services is the same however much "
        "of the tree happened to be imported.",
        checked_by="tests/test_what_has_a_measured_effect.py::"
        "test_the_declared_count_is_the_same_however_much_was_imported",
        if_it_fails="coverage reads differently per process and the number "
        "cannot be compared across runs",
    ),
    APromise(
        it="A promise a package declares names a test that exists, so a "
        "declaration cannot read as coverage it does not have.",
        checked_by="tests/test_a_promise_with_a_test.py::"
        "test_every_declared_promise_names_a_test_that_exists",
        if_it_fails="promises_whose_test_is_missing() names the package and "
        "the promise; the organ audit would otherwise count a string",
    ),
)
