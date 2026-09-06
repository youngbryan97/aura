"""A promise naming a test that does not exist reads as coverage."""
from __future__ import annotations

import pytest

from core.verify.a_promise_with_a_test import (
    APromise,
    how_the_promises_stand,
    packages_that_declare_promises,
    promises_whose_test_is_missing,
    the_declared_promises,
)


def test_every_declared_promise_names_a_test_that_exists() -> None:
    """The gate. A renamed test shows up here, not as coverage that stopped."""
    missing = promises_whose_test_is_missing()
    assert missing == (), "\n".join(missing)


def test_a_promise_too_short_to_disagree_with_is_refused() -> None:
    with pytest.raises(ValueError, match="too short"):
        APromise(
            it="handles errors",
            checked_by="tests/x.py::test_y",
            if_it_fails="somewhere",
        )


def test_a_promise_with_no_runnable_test_is_refused() -> None:
    with pytest.raises(ValueError, match="not a test node id"):
        APromise(
            it="this one is a proper sentence about behaviour",
            checked_by="the integration suite, probably",
            if_it_fails="somewhere",
        )


def test_a_promise_that_does_not_say_where_a_breach_goes_is_refused() -> None:
    with pytest.raises(ValueError, match="breach"):
        APromise(
            it="this one is a proper sentence about behaviour",
            checked_by="tests/x.py::test_y",
            if_it_fails="   ",
        )


def test_packages_declare_promises_and_the_number_only_goes_up() -> None:
    declaring = packages_that_declare_promises()
    assert len(declaring) >= 6, (
        f"only {len(declaring)} package(s) declare checkable promises: "
        f"{list(declaring)}"
    )


def test_every_promise_says_where_its_breach_goes() -> None:
    for package, promises in the_declared_promises().items():
        for promise in promises:
            assert promise.if_it_fails.strip(), f"core.{package}: {promise.it}"
            assert len(promise.it.split()) >= 4, f"core.{package}: {promise.it}"


def test_the_report_separates_declaring_from_not() -> None:
    seen = how_the_promises_stand()
    assert seen["packages"] >= 100
    assert seen["declaring"] + len(seen["not_declaring"]) == seen["packages"]
    assert seen["with_a_missing_test"] == []
