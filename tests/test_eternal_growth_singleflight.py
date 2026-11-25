from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.kernel.upgrades_10x import EternalGrowthEngine
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_eternal_growth_returns_without_waiting_for_slow_model():
    release = asyncio.Event()

    class _SlowLLM:
        async def think(self, *_args, **_kwargs):
            await release.wait()
            return '{"upgrade": false}'

    kernel = SimpleNamespace(
        organs={"llm": SimpleNamespace(get_instance=lambda: _SlowLLM())}
    )
    engine = EternalGrowthEngine(kernel)
    state = AuraState()

    result = await asyncio.wait_for(engine.execute(state), timeout=0.1)
    await asyncio.sleep(0)

    assert result is state
    assert engine._growth_task is not None
    assert not engine._growth_task.done()

    release.set()
    await engine._growth_task


@pytest.mark.asyncio
async def test_eternal_growth_applies_completed_result_on_canonical_tick():
    from core.self.growth import get_growth_ledger, reset_for_test

    reset_for_test()
    kernel = SimpleNamespace(organs={})
    engine = EternalGrowthEngine(kernel)
    state = AuraState()

    completed = asyncio.get_running_loop().create_future()
    completed.set_result({"upgrade": True})
    engine._growth_task = completed
    engine.last_growth = 1e20

    await engine.execute(state)

    # The verdict is evidence in the growth ledger, which the affect update
    # turns into the score, rather than a step added to a score it rewrites.
    assert list(get_growth_ledger()._steps) == [1.0]
    assert engine._growth_task is None
    reset_for_test()


@pytest.mark.asyncio
async def test_a_small_models_words_are_never_an_objective_of_hers():
    """LIVE: "verified internal task completion", answered to a request for a milestone, was her objective for 95 days."""
    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: None)})
    engine = EternalGrowthEngine(kernel)
    state = AuraState()
    applied = await engine._apply_growth_result(state, engine._parse_growth_result('{"milestone":"verified internal task completion","upgrade":false}'))
    assert not applied.cognition.pending_initiatives and not applied.cognition.current_objective


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"milestone":"Measure retained reasoning","upgrade":true}', {"upgrade": True}),
        ('analysis\n{"upgrade":false}\n', {"upgrade": False}),
        ("UPGRADE", {"upgrade": True}),
        ("not valid", {"upgrade": False}),
    ],
)
def test_eternal_growth_result_parser_is_bounded(raw, expected):
    assert EternalGrowthEngine._parse_growth_result(raw) == expected
