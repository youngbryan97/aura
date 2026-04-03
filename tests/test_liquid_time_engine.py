import pytest

from core.evolution.liquid_time_engine import ContinuousState, LiquidNode, MAX_SLEEP_WAKE_DT_S


@pytest.mark.asyncio
async def test_liquid_time_engine_clamps_large_sleep_wake_delta():
    state = ContinuousState()
    state.nodes["frustration"] = LiquidNode(
        value=1.0,
        resting_state=0.5,
        tau=100.0,
        leakage=0.1,
    )
    state.last_update -= MAX_SLEEP_WAKE_DT_S * 10

    await state.pulse()

    assert state.last_update > 0
    assert 0.0 <= state.nodes["frustration"].value <= 1.0
    assert state.nodes["frustration"].value > 0.5


@pytest.mark.asyncio
async def test_liquid_time_engine_handles_negative_clock_skew():
    state = ContinuousState()
    before = state.nodes["curiosity"].value
    state.last_update += 30.0

    await state.pulse()

    assert state.nodes["curiosity"].value == before
