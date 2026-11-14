"""Three locks lockdep caught holding her event loop live on 13 November 2026, each moved off it.

- conation.what_she_wants, 225 ms: her learning phase read the turn into her record of what she wants on the loop.
- influence_campaign.persist, 52 ms: a threading lock held across the awaits of an async write.
- the semantic runtime, built on the loop by autonomous resilience resolving every registered service to wire it
  (tests/test_autonomous_resilience.py holds that one).
"""
from __future__ import annotations

import asyncio
import inspect

from core.verify import influence_campaign


def test_her_learning_phase_reads_the_turn_off_the_loop():
    from core.phases import learning_phase

    source = inspect.getsource(learning_phase.read_what_the_turn_was_worth)
    assert "await asyncio.to_thread(met_this_turn, about, reading, said=said)" in source


def test_the_influence_ledger_is_written_under_an_async_lock():
    assert hasattr(influence_campaign._PERSIST_LOCK, "__aenter__")

    async def two_writers_on_one_loop() -> None:
        async with influence_campaign._PERSIST_LOCK:
            pass
        async with influence_campaign._PERSIST_LOCK:
            pass

    asyncio.run(two_writers_on_one_loop())
