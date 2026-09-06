"""Two resources at once, in an order nobody has to think about."""
from __future__ import annotations

import asyncio
import itertools

import pytest

from core.runtime.claiming_more_than_one import (
    THE_ORDER,
    claim_all,
    forget_everything,
    how_the_multi_claims_have_gone,
    in_order,
)
from core.runtime.who_gets_it_next import THE_RESOURCES
from core.runtime.who_gets_it_next import forget_everything as forget_claims
from core.runtime.who_gets_it_next import who_holds_what

HERE = tuple(
    sorted(n for n, one in THE_RESOURCES.items() if one.get("granted") == "here")
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    forget_claims()
    yield
    forget_everything()
    forget_claims()


def test_every_permutation_of_every_subset_gives_one_sequence() -> None:
    """The property. The deadlock only shows up in the order nobody wrote."""
    names = list(THE_ORDER)
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            sequences = {in_order(p) for p in itertools.permutations(subset)}
            assert len(sequences) == 1, f"{subset} acquires in {sequences}"


def test_asking_twice_for_one_resource_is_one_claim() -> None:
    """Waiting for a lock you already hold is a hang with no message."""
    one = HERE[0]
    assert in_order([one, one, one]) == (one,)
    assert in_order(list(THE_ORDER) * 3) == THE_ORDER


def test_asking_for_nothing_is_nothing_rather_than_an_error() -> None:
    assert in_order([]) == ()

    async def go():
        async with claim_all([], "me") as held:
            return held

    assert asyncio.run(go()) == ()


def test_a_resource_that_does_not_exist_is_named() -> None:
    with pytest.raises(KeyError, match="no such resource"):
        in_order(["screen", "a_thing_nobody_has"])


def test_all_of_them_or_none_checked_before_anything_is_taken() -> None:
    """Refusing half way is how a caller ends up holding one of two."""
    elsewhere = [
        n for n, one in THE_RESOURCES.items() if one.get("granted") != "here"
    ]
    if not elsewhere:
        pytest.skip("everything is granted here in this build")

    async def go():
        with pytest.raises(ValueError, match="nothing was taken"):
            async with claim_all([HERE[0], elsewhere[0]], "me"):
                pass
        return who_holds_what()

    held = asyncio.run(go())
    assert not [k for k, v in held.items() if v.get("by")], "it took one anyway"


def test_they_are_taken_in_order_and_given_back_in_reverse() -> None:
    async def go():
        async with claim_all(list(reversed(HERE)), "me") as took:
            return took

    assert asyncio.run(go()) == HERE
    row = how_the_multi_claims_have_gone()["recent"][-1]
    assert row["took"] == list(HERE)
    assert row["gave_back"] == list(HERE), "reverse of reverse is the order"


def test_a_failure_inside_the_block_gives_everything_back() -> None:
    async def go():
        with pytest.raises(ZeroDivisionError):
            async with claim_all(HERE, "me"):
                raise ZeroDivisionError
        return who_holds_what()

    held = asyncio.run(go())
    assert not [k for k, v in held.items() if v.get("by")]
    assert how_the_multi_claims_have_gone()["left_holding_something"] == 0


def test_a_cancellation_inside_the_block_gives_everything_back() -> None:
    async def go():
        with pytest.raises(asyncio.CancelledError):
            async with claim_all(HERE, "me"):
                raise asyncio.CancelledError
        return who_holds_what()

    held = asyncio.run(go())
    assert not [k for k, v in held.items() if v.get("by")]


def test_two_callers_wanting_the_same_pair_do_not_deadlock() -> None:
    """The whole reason for a global order, run rather than argued."""
    if len(HERE) < 2:
        pytest.skip("needs two resources granted here")

    async def go():
        done: list[str] = []

        async def one(name: str, wanted):
            async with claim_all(wanted, name):
                await asyncio.sleep(0.01)
                done.append(name)

        await asyncio.wait_for(
            asyncio.gather(
                one("a", list(HERE)),
                one("b", list(reversed(HERE))),
            ),
            timeout=5.0,
        )
        return done

    assert sorted(asyncio.run(go())) == ["a", "b"]


def test_the_order_is_every_resource_there_is() -> None:
    assert set(THE_ORDER) == set(THE_RESOURCES)
    assert list(THE_ORDER) == sorted(THE_ORDER), "one order, and not by importance"
