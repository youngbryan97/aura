from __future__ import annotations

import asyncio
import gc
import warnings

from core.brain.llm.endogenous_state import _substrate_summary


def test_substrate_summary_prefers_explicit_nowait_snapshot():
    class Source:
        def __init__(self) -> None:
            self.async_calls = 0

        def get_state_summary_nowait(self):
            return {"valence": 0.25, "phi": 0.4}

        async def get_state_summary(self):
            self.async_calls += 1
            return {"valence": -1.0}

    source = Source()

    assert _substrate_summary(source) == {"valence": 0.25, "phi": 0.4}
    assert source.async_calls == 0


def test_substrate_summary_skips_async_only_source_without_creating_coroutine():
    class AsyncOnlySource:
        def __init__(self) -> None:
            self.calls = 0

        async def get_state_summary(self):
            self.calls += 1
            return {"valence": 0.5}

    source = AsyncOnlySource()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _substrate_summary(source) is None
        gc.collect()

    assert source.calls == 0
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]


def test_substrate_summary_closes_disguised_awaitable():
    async def summary():
        await asyncio.sleep(0)
        return {"valence": 0.75}

    class DynamicSource:
        def get_state_summary(self):
            return summary()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _substrate_summary(DynamicSource()) is None
        gc.collect()

    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
