"""A refusal asked three times is still a refusal."""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.what_must_never_be_retried import (
    TryAgain,
    how_to_treat,
    may_be_retried,
    why_not,
)


def test_a_transient_fault_is_worth_another_attempt() -> None:
    assert how_to_treat(TimeoutError("lane busy")) is TryAgain.AGAIN
    assert may_be_retried(ConnectionError("reset by peer"))
    assert why_not(TimeoutError("busy")) == ""


def test_a_cancelled_turn_is_never_restarted_by_a_retry_loop() -> None:
    assert how_to_treat(asyncio.CancelledError()) is TryAgain.NEVER
    assert not may_be_retried(asyncio.CancelledError())


def test_a_governance_refusal_travelling_as_words_is_still_a_refusal() -> None:
    """The type is a promise; the words are the fallback that catches the rest."""
    plain = RuntimeError("the write was refused by governance")
    assert how_to_treat(plain) is TryAgain.NEVER
    assert "decision rather than a fault" in why_not(plain)


def test_a_subclass_of_a_refusal_is_classified_by_its_ancestry() -> None:
    class MyDenial(PermissionError):
        pass

    assert how_to_treat(MyDenial("nope")) is TryAgain.NEVER


def test_malformed_input_is_neither_retried_nor_treated_as_a_decision() -> None:
    """The middle answer: this attempt is wrong, a different one may not be."""
    verdict = how_to_treat(ValueError("expected an int"))
    assert verdict is TryAgain.NOT_LIKE_THIS
    assert not may_be_retried(ValueError("expected an int"))
    assert "identical one fails identically" in why_not(ValueError("x"))


def test_an_unrecognised_failure_defaults_to_retryable() -> None:
    """Anything else is an ordinary fault; refusing them all would stop real work."""

    class SomethingNew(Exception):
        pass

    assert may_be_retried(SomethingNew("who knows"))


@pytest.mark.parametrize(
    "word",
    ["refused", "denied", "not permitted", "unsafe", "cancelled", "forbidden"],
)
def test_every_word_that_means_no_is_read_as_no(word: str) -> None:
    assert how_to_treat(Exception(f"request {word} at the boundary")) is TryAgain.NEVER


def test_the_three_answers_are_distinct_and_only_one_retries() -> None:
    retryable = {v for v in TryAgain if v is TryAgain.AGAIN}
    assert len(TryAgain) == 3
    assert len(retryable) == 1
