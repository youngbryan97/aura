"""An answer of "nothing" is an answer, not a subsystem fault.

The think path declares `str | None`, and None is how it says it has no
answer this time — a refusal, an exhausted local lane, a request it declined.
Raising there turned an ordinary outcome into damage: measured live,
fifty-one faults in half an hour, each recorded MARGINAL, driving her own
affect to frustration=1.00, depletion=0.49, strain — and opening a runtime
integrity incident on top.
"""
from __future__ import annotations

import inspect

import pytest

from core.agency import her_reasoning


@pytest.mark.asyncio
async def test_a_router_that_answers_nothing_yields_an_empty_answer():
    class _Silent:
        async def think(self, **kwargs):
            return None

    generate = her_reasoning.generator(origin="test", max_tokens=8, timeout_s=2.0)
    her_reasoning._router = lambda: _Silent()
    try:
        assert await generate("anything", 0.3) == ""
    finally:
        pass


@pytest.mark.asyncio
async def test_something_genuinely_unusable_still_raises():
    """A different thing, and the deliberation should stop for it."""

    class _Odd:
        async def think(self, **kwargs):
            return 12345

    her_reasoning._router = lambda: _Odd()
    generate = her_reasoning.generator(origin="test", max_tokens=8, timeout_s=2.0)
    with pytest.raises(RuntimeError):
        await generate("anything", 0.3)


def test_a_mind_out_of_reach_is_still_reported():
    source = inspect.getsource(her_reasoning.generator)
    assert 'raise RuntimeError("no model router is registered")' in source


def test_failing_to_plan_is_recorded_once_rather_than_every_cycle():
    """The condition worth recording is that she keeps failing to form a
    plan, and one entry says that as well as twenty do."""
    from core.agency import standing_strategy

    source = inspect.getsource(standing_strategy.settle_on_an_approach)
    assert "_said_it_could_not_plan" in source
    where = source.index("_said_it_could_not_plan")
    assert "record_degradation" in source[where : where + 400]
