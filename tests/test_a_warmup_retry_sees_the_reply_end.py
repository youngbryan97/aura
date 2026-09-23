"""A warmup retry waits for a reply to end, and sees it end.

The retry stands down while somebody is being answered, reading the client
module's flag for that. It imported the flag by name, which copies the value,
and then polled the copy: a retry that began during a reply waited the whole
allowance and stood down even when the reply had finished at once. On 23
September a whole run lost her cortex to a forced abort, and every warmup
after the respawn stood down this way, so the lane never became ready.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


def _client() -> SimpleNamespace:
    reasons: list[str] = []

    async def reboot_worker(*, reason: str, mark_failed: bool) -> None:
        reasons.append(reason)

    return SimpleNamespace(reboot_worker=reboot_worker, reasons=reasons)


def test_a_reply_that_ends_during_the_wait_lets_the_retry_go_ahead(monkeypatch) -> None:
    monkeypatch.setattr(mlx_client, "_FOREGROUND_OWNER_IS_USER_FACING", True)
    client = _client()

    async def reply_ends() -> None:
        await asyncio.sleep(0.3)
        mlx_client._FOREGROUND_OWNER_IS_USER_FACING = False

    async def run() -> float:
        started = time.monotonic()
        await asyncio.gather(
            mlx_client.MLXLocalClient._recover_worker_for_warmup_retry(client),
            reply_ends(),
        )
        return time.monotonic() - started

    seconds = asyncio.run(run())
    assert client.reasons == ["warmup_precompile_retry"]
    assert seconds < mlx_client._WAIT_OUT_A_REPLY_S / 2, "the retry waited on a copy of the flag"


def test_a_reply_still_running_at_the_end_of_the_allowance_stands_the_retry_down(monkeypatch) -> None:
    monkeypatch.setattr(mlx_client, "_FOREGROUND_OWNER_IS_USER_FACING", True)
    monkeypatch.setattr(mlx_client, "_WAIT_OUT_A_REPLY_S", 0.5)
    client = _client()
    asyncio.run(mlx_client.MLXLocalClient._recover_worker_for_warmup_retry(client))
    assert client.reasons == []
