"""What core.persistence guarantees, and the test that catches each breaking."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="A store written by a later minor version is still readable by an "
        "earlier one, so a downgrade does not lose the file.",
        checked_by="tests/test_a_versioned_store.py::"
        "test_a_later_minor_version_is_still_readable",
        if_it_fails="CannotRead is raised and the file is put aside rather "
        "than overwritten; report() names where it went",
    ),
    APromise(
        it="A file this build cannot read is quarantined rather than "
        "overwritten, so the data survives the process that could not use it.",
        checked_by="tests/test_a_versioned_store.py::"
        "test_a_file_that_will_not_parse_is_put_aside_not_overwritten",
        if_it_fails="the original bytes are gone; the quarantine path in "
        "report() is what proves they are not",
    ),
    APromise(
        it="Nothing on disk reads as None rather than as an error, so a first "
        "run is not indistinguishable from a corrupt one.",
        checked_by="tests/test_a_versioned_store.py::"
        "test_nothing_there_is_none_and_not_an_error",
        if_it_fails="a first boot raises where it should return empty; the "
        "caller cannot tell absence from damage",
    ),
)
