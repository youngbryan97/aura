"""A slow report sink must leave the runtime event loop available."""

import asyncio
import threading
from collections import deque

from core.resilience.stability_guardian import StabilityGuardian, SystemHealthReport


def test_report_sink_runs_off_loop_and_is_awaited():
    async def exercise():
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()
        release = threading.Event()
        observed = []
        loop_thread = threading.get_ident()
        report = SystemHealthReport(0, True, [], 0, 0, 0, 0, 0)

        class Guardian(StabilityGuardian):
            CHECK_INTERVAL_S = 0

            async def run_checks(self):
                self._running = False
                return report

            def _persist_report(self, value):
                observed.append((threading.get_ident(), value))
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(2), "report sink blocked the event loop"

        guardian = Guardian.__new__(Guardian)
        guardian._running = True
        guardian._loop_lag_samples = deque()
        guardian._report_history = deque()
        task = asyncio.create_task(guardian._loop())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not task.done(), "persistence must be awaited"
            assert observed == [(observed[0][0], report)]
            assert observed[0][0] != loop_thread
        finally:
            release.set()
            await asyncio.wait_for(task, 3)
        assert list(guardian._report_history) == [report]

    asyncio.run(exercise())
