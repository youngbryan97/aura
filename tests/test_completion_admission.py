import asyncio

import pytest

from core.runtime.completion_admission import (
    admit_completion_work,
    bind_completion_admission,
)
from core.runtime.turn_outcome import TurnOutcome, bind_turn


@pytest.mark.asyncio
async def test_only_bound_owner_can_admit_work():
    context = {}
    with bind_turn(TurnOutcome("owner", origin="user_chat")):
        async with asyncio.timeout(10) as clock:
            original = clock.when()
            async with bind_completion_admission(clock, context, enabled=True):
                assert admit_completion_work(30)
                assert clock.when() > original + 19
                granted = clock.when()
                assert admit_completion_work(1)
                assert clock.when() == granted
                assert not admit_completion_work(float("inf"))
                with bind_turn(TurnOutcome("other", origin="user_chat")):
                    assert not admit_completion_work(100)
            assert not admit_completion_work(100)
    assert context["cognitive_cycle_deadline_monotonic"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,bound", [(False, True), (True, False)])
async def test_disabled_or_unowned_work_keeps_original_clock(enabled, bound):
    from contextlib import nullcontext

    owner = bind_turn(TurnOutcome("owner", origin="user_chat")) if bound else nullcontext()
    with owner:
        async with asyncio.timeout(10) as clock:
            original = clock.when()
            async with bind_completion_admission(clock, {}, enabled=enabled):
                assert not admit_completion_work(100)
                assert clock.when() == original
