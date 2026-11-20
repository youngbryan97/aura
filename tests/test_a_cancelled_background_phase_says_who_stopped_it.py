"""A background phase stopped before it finished is not reported as a slow phase, and says why it was stopped.

Her logs to 20 November 2026 held 1,948 "Kernel phase latency exceeded budget" warnings and 168 "Background phase ...
was cancelled" warnings. A cancelled phase's time is how long it waited before someone stopped it, and the line that
said so named no one.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _kernel():
    from core.kernel.aura_kernel import AuraKernel
    from core.state.aura_state import AuraState

    kernel = AuraKernel.__new__(AuraKernel)
    kernel.state = AuraState()
    kernel._phases = []
    kernel._phase_runtime_samples = deque(maxlen=64)
    return kernel


class _Phase:
    def __init__(self, wait_s: float, cancelled_with: str = ""):
        self.wait_s, self.cancelled_with = wait_s, cancelled_with

    async def execute(self, state, objective="", **_kwargs):
        await asyncio.sleep(self.wait_s)
        if self.cancelled_with:
            raise asyncio.CancelledError(self.cancelled_with)
        return state


def _run(kernel, phase, monkeypatch):
    monkeypatch.setenv("AURA_KERNEL_SLOW_PHASE_MS", "1")
    return asyncio.run(kernel._execute_phase_with_timing(phase, "SlowPhase", SimpleNamespace(),
                                                         objective="tick", priority=False))


def test_a_phase_that_finished_slowly_is_a_warning(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        _run(_kernel(), _Phase(0.01), monkeypatch)
    slow = [record for record in caplog.records if "latency exceeded budget" in record.getMessage()]
    assert slow and slow[0].levelno == logging.WARNING


def test_a_phase_cancelled_before_it_finished_is_not_called_slow(monkeypatch, caplog):
    with caplog.at_level(logging.INFO), pytest.raises(asyncio.CancelledError):
        _run(_kernel(), _Phase(0.01, cancelled_with="taken over by a command"), monkeypatch)
    slow = [record for record in caplog.records if "latency exceeded budget" in record.getMessage()]
    assert slow and slow[0].levelno == logging.INFO
    assert "cancelled before it finished" in slow[0].getMessage()


def test_the_cancellation_line_carries_the_cancellers_reason():
    import inspect

    from core.kernel import aura_kernel_lifecycle

    source = inspect.getsource(aura_kernel_lifecycle)
    assert "was cancelled (%s); continuing tick." in source
    assert "phase_err.args[0]" in source
