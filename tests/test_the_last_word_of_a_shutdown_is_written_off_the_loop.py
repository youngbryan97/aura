"""The shutdown verdict is an atomic write with an fsync, and it was written on
the event loop: every task still finishing waited on the disk for it, and
lockdep said so on every shutdown."""

from __future__ import annotations

import inspect


def test_the_verdict_is_published_off_the_loop():
    from core.ops import graceful_shutdown

    source = inspect.getsource(graceful_shutdown)
    assert "await asyncio.to_thread(lambda: publish_shutdown_verdict(**verdict))" in source
