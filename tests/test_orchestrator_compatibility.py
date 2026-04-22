import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.orchestrator.mixins.boot.boot_cognitive import BootCognitiveMixin
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin


class _BootProbe(BootCognitiveMixin):
    cognition = None


class _OutputProbe(OutputFormatterMixin):
    def __init__(self, emitter):
        self.cognitive_engine = SimpleNamespace(_emit_thought=emitter)


@pytest.mark.asyncio
async def test_init_cognitive_core_awaits_async_setup(monkeypatch):
    setup = AsyncMock()
    cognitive_engine = SimpleNamespace(setup=setup)
    capability_engine = object()

    def _get(name, default=None):
        if name == "cognitive_engine":
            return cognitive_engine
        if name == "capability_engine":
            return capability_engine
        return default

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_cognitive.ServiceContainer.get",
        staticmethod(_get),
    )

    await _BootProbe()._init_cognitive_core()

    setup.assert_awaited_once_with(registry=capability_engine, router=capability_engine)


@pytest.mark.asyncio
async def test_emit_thought_stream_schedules_async_emitters():
    emitter = AsyncMock()
    probe = _OutputProbe(emitter)

    probe._emit_thought_stream("hello")
    await asyncio.sleep(0)

    emitter.assert_awaited_once_with("hello")
