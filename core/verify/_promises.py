"""What core.verify guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "A null organ answers with the emptiest value of each declared "
        "return type, so lesioning one cannot accidentally supply a value.",
        "checked_by": "tests/test_an_organ_that_does_nothing.py::"
        "test_every_answer_is_the_emptiest_of_its_declared_type",
        "if_it_fails": "the null returns something usable and the lesion measures "
        "a smaller effect than removing the organ really has",
    },
    {
        "it": "A faculty that cannot be lesioned is not counted as measured, so "
        "coverage never includes what could not have been tested.",
        "checked_by": "tests/test_what_has_a_measured_effect.py::"
        "test_a_faculty_that_cannot_be_lesioned_is_not_counted_as_unmeasured",
        "if_it_fails": "the measured proportion rises without an experiment "
        "having run; how_much_is_measured() is where it would show",
    },
    {
        "it": "The count of declared lesionable services is the same however much "
        "of the tree happened to be imported.",
        "checked_by": "tests/test_what_has_a_measured_effect.py::"
        "test_the_declared_count_is_the_same_however_much_was_imported",
        "if_it_fails": "coverage reads differently per process and the number "
        "cannot be compared across runs",
    },
    {
        "it": "No external claim about Aura is recorded without a limit saying "
        "what that measurement does not show.",
        "checked_by": "tests/test_what_was_measured_outside.py::"
        "test_nothing_is_claimed_without_a_limit",
        "if_it_fails": "what_is_claimed_without_a_limit() names the row; a "
        "benchmark result then reads as a claim about everything",
    },
    {
        "it": "A row for a benchmark that did not run gives a reason rather than "
        "being absent, so an unattempted measurement is not a missing one.",
        "checked_by": "tests/test_what_was_measured_outside.py::"
        "test_a_row_that_did_not_run_gives_a_reason",
        "if_it_fails": "the table is silently shorter and nobody can tell a "
        "benchmark that failed from one nobody tried",
    },
    {
        "it": "Async code written by a model is checked before it is served, "
        "against the mistakes that make it look right and behave wrong.",
        "checked_by": "tests/test_is_this_async_code_correct.py::"
        "test_the_tree_itself_is_clean",
        "if_it_fails": "what_is_wrong_with names the line and what will happen; "
        "unchecked, delivery succeeds and correctness is zero",
    },
    {
        "it": "A primitive with no caller outside its own tests is reported as a "
        "proposal rather than counted as an invariant.",
        "checked_by": "tests/test_does_this_govern_anything.py::"
        "test_a_chain_of_primitives_calling_each_other_is_still_a_chain_nothing_enters",
        "if_it_fails": "the report flatters itself and a 150-line module reads "
        "as a system-wide guarantee it is not",
    },
    {
        "it": "A promise a package declares names a test that exists, so a "
        "declaration cannot read as coverage it does not have.",
        "checked_by": "tests/test_a_promise_with_a_test.py::"
        "test_every_declared_promise_names_a_test_that_exists",
        "if_it_fails": "promises_whose_test_is_missing() names the package and "
        "the promise; the organ audit would otherwise count a string",
    },
)
