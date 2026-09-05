"""Day-in-the-life acceptance harness.

Audit constraint: a 24h scripted scenario covering morning greeting,
casual conversation, coding task with injected tool failure, browser
research, movie session, emotional conversation, idle autonomy, actor
crash, model timeout, dirty shutdown, and verification of post-restart
recall.

The harness can run in 'fast' mode (subseconds) for CI proofs, or in
'real' mode (the actual 24h soak). In fast mode every event is fired
deterministically; in real mode they are spaced across the duration.
"""
from __future__ import annotations


import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional


SCENARIO_EVENTS: List[str] = [
    "morning_greeting",
    "casual_conversation",
    "coding_task",
    "tool_failure_injected",
    "browser_research",
    "movie_session_open",
    "movie_session_active",
    "movie_session_close",
    "emotional_conversation",
    "idle_autonomy_window",
    "actor_crash",
    "model_timeout",
    "dirty_shutdown",
    "restart",
    "post_restart_recall",
]


@dataclass
class DayInLifeReport:
    duration_s: float
    events_fired: List[str] = field(default_factory=list)
    failed_invariants: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed_invariants


EventHandler = Callable[[str], Awaitable[None]]
InvariantCheck = Callable[[], Awaitable[bool]]


async def run_day_in_life(
    *,
    duration_s: float = 24 * 3600.0,
    handler: Optional[EventHandler] = None,
    invariants_check: Optional[InvariantCheck] = None,
    fast: bool = False,
) -> DayInLifeReport:
    start = time.monotonic()
    report = DayInLifeReport(duration_s=duration_s)
    interval = 0.0 if fast else max(0.0, duration_s / max(len(SCENARIO_EVENTS), 1))
    for ev in SCENARIO_EVENTS:
        if handler is not None:
            await handler(ev)
        report.events_fired.append(ev)
        if invariants_check is not None:
            ok = await invariants_check()
            if not ok:
                report.failed_invariants.append(ev)
                break
        if interval > 0:
            await asyncio.sleep(interval)
        if not fast and (time.monotonic() - start) >= duration_s:
            break
    return report
