"""Stopped execution owners cannot admit replacement model work."""
import asyncio

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.runtime.what_stops_it import AnExecutionContext, under


def test_stopped_owner_is_rejected_before_any_client_or_worker_mutation():
    async def exercise():
        owner = AnExecutionContext(doing="desktop turn")
        owner.stopping.stop("user_requested_stop")
        # No client internals exist: admission must stop before touching them.
        client = object.__new__(MLXLocalClient)
        with under(owner):
            with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
                await client.generate("fallback must not run")
            with pytest.raises(asyncio.CancelledError, match="user_requested_stop"):
                await client.generate("retry must not run")

    asyncio.run(exercise())
