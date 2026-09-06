"""Let an owning foreground clock admit measured work before dispatch."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

_OWNER: ContextVar[Any] = ContextVar("completion_admission_owner", default=None)


@asynccontextmanager
async def bind_completion_admission(clock: Any, context: dict, *, enabled: bool):
    from core.runtime.turn_outcome import current_turn

    owner = current_turn()
    loop = asyncio.get_running_loop()
    active = True

    def admit(seconds: float) -> bool:
        if not active or owner is None or current_turn() is not owner:
            return False
        if not math.isfinite(seconds) or seconds <= 0 or clock.expired():
            return False
        deadline = clock.when()
        if deadline is None:
            return False
        deadline = max(deadline, loop.time() + seconds)
        clock.reschedule(deadline)
        context["cognitive_cycle_deadline_monotonic"] = (
            time.monotonic() + deadline - loop.time()
        )
        return True

    token = _OWNER.set(admit if enabled else None)
    try:
        yield
    finally:
        active = False
        _OWNER.reset(token)


def admit_completion_work(seconds: float) -> bool:
    admit = _OWNER.get()
    return bool(admit is not None and admit(seconds))
