"""One policy per call site, and the failures it refuses to repeat."""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.how_a_call_is_made import (
    APolicy,
    HowBadIsLosingThis,
    IsItSafeToRepeat,
    calls_with_no_policy,
    declare_a_call,
    forget_everything,
    how_the_calls_are_made,
    make_the_call,
    make_the_call_async,
    the_policy_for,
)
from core.runtime.what_is_left_to_spend import a_budget_of


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_a_transient_failure_is_retried_up_to_the_declared_attempts() -> None:
    tries = {"n": 0}

    def flaky() -> str:
        tries["n"] += 1
        if tries["n"] < 3:
            raise ConnectionError("reset")
        return "landed"

    declare_a_call(APolicy(name="x", attempts=3, backoff_seconds=0.0,
                           idempotent=IsItSafeToRepeat.YES))
    out = make_the_call("x", flaky)
    assert out.ok and out.value == "landed"
    assert out.attempts_made == 3


def test_a_refusal_is_not_retried_however_many_attempts_were_declared() -> None:
    tries = {"n": 0}

    def denied() -> None:
        tries["n"] += 1
        raise PermissionError("governance said no")

    declare_a_call(APolicy(name="x", attempts=5, backoff_seconds=0.0))
    out = make_the_call("x", denied)
    assert not out.ok
    assert tries["n"] == 1, "asked once, because the answer was a decision"
    assert "decision rather than a fault" in out.refused_retry_because


def test_a_timeout_is_not_retried_where_repeating_could_do_the_work_twice() -> None:
    """After a timeout you cannot tell whether the first attempt landed."""

    async def slow() -> int:
        await asyncio.sleep(5)
        return 1

    declare_a_call(APolicy(name="t", timeout_seconds=0.05, attempts=3,
                           idempotent=IsItSafeToRepeat.NO))
    out = asyncio.run(make_the_call_async("t", slow))
    assert not out.ok
    assert out.attempts_made == 1
    assert "not safe to repeat" in out.refused_retry_because


def test_the_same_timeout_is_retried_where_repeating_is_safe() -> None:
    tries = {"n": 0}

    async def slow_then_fast() -> str:
        tries["n"] += 1
        if tries["n"] == 1:
            await asyncio.sleep(5)
        return "landed"

    declare_a_call(APolicy(name="t", timeout_seconds=0.05, attempts=2,
                           backoff_seconds=0.0, idempotent=IsItSafeToRepeat.YES))
    out = asyncio.run(make_the_call_async("t", slow_then_fast))
    assert out.ok and out.attempts_made == 2


def test_a_required_call_does_not_quietly_fall_back() -> None:
    declare_a_call(APolicy(name="r", attempts=1,
                           criticality=HowBadIsLosingThis.REQUIRED))
    out = make_the_call("r", lambda: 1 / 0, fallback=lambda: "guess")
    assert not out.ok, "a required result cannot be replaced by a guess"
    assert not out.fell_back


def test_a_degrading_call_falls_back_and_says_it_did() -> None:
    declare_a_call(APolicy(name="d", attempts=1,
                           criticality=HowBadIsLosingThis.DEGRADES))
    out = make_the_call("d", lambda: 1 / 0, fallback=lambda: "worse but real")
    assert out.ok and out.fell_back and out.value == "worse but real"


def test_a_retrying_call_cannot_outspend_the_turn_that_contains_it() -> None:
    declare_a_call(APolicy(name="x", attempts=10, backoff_seconds=0.0,
                           idempotent=IsItSafeToRepeat.YES))
    budget = a_budget_of("turn", 2.0)
    out = make_the_call("x", lambda: (_ for _ in ()).throw(ConnectionError("no")),
                        budget=budget)
    assert out.attempts_made == 2
    assert out.refused_retry_because == "the budget was spent"


def test_a_cancellation_propagates_and_is_never_retried() -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError

    declare_a_call(APolicy(name="c", attempts=5, backoff_seconds=0.0,
                           idempotent=IsItSafeToRepeat.YES))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(make_the_call_async("c", cancelled))


def test_a_cached_result_is_served_without_running_the_call() -> None:
    tries = {"n": 0}

    def counted() -> int:
        tries["n"] += 1
        return tries["n"]

    declare_a_call(APolicy(name="k", cache_seconds=30.0))
    assert make_the_call("k", counted).value == 1
    second = make_the_call("k", counted)
    assert second.value == 1 and second.from_cache
    assert tries["n"] == 1


def test_an_undeclared_call_site_is_named_rather_than_silently_defaulted() -> None:
    make_the_call("nobody_declared_me", lambda: 1)
    assert "nobody_declared_me" in calls_with_no_policy()
    assert the_policy_for("nobody_declared_me").why.startswith("nobody declared")


def test_every_shipped_policy_says_why_and_names_an_owner() -> None:
    seen = how_the_calls_are_made()
    assert seen["declared"] >= 7
    for name, policy in seen["policies"].items():
        assert policy["why"].strip(), f"{name} has a policy with no reason"
        assert policy["owner"].strip(), f"{name} has no owner to drain it"


def test_attempts_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least once"):
        APolicy(name="x", attempts=0)
