from __future__ import annotations

import asyncio

import pytest

from core.cognitive_kernel import CognitiveBrief, InputDomain, ResponseStrategy
from core.inner_monologue import InnerMonologue
from core.runtime.errors import get_degradation_tracker


def _deep_brief() -> CognitiveBrief:
    return CognitiveBrief(
        domain=InputDomain.ABSTRACT,
        strategy=ResponseStrategy.EXPLORE,
        prior_beliefs=["curiosity should stay grounded"],
        key_points=["name the uncertainty"],
        complexity="deep",
        conviction=0.3,
    )


@pytest.mark.asyncio
async def test_inner_monologue_deepening_timeout_preserves_baseline_packet():
    tracker = get_degradation_tracker()
    tracker.reset()

    class _SlowRouter:
        async def think(self, **_kwargs):
            await asyncio.sleep(1.0)
            return "{}"

    monologue = InnerMonologue(deepening_timeout=0.05)
    monologue._llm_router = _SlowRouter()
    monologue._router_available = True

    packet = await monologue.think("what do you think?", _deep_brief(), history=[])

    assert packet.reasoning_source == "kernel_only"
    assert packet.primary_points == ["name the uncertainty"]
    assert tracker.count("inner_monologue", "degraded") >= 1


@pytest.mark.asyncio
async def test_inner_monologue_sanitizes_deepening_payload_fields():
    class _Router:
        async def think(self, **_kwargs):
            return """```json
{
  "strengthened_stance": "A stronger grounded stance.",
  "primary_points": "not-a-list",
  "transparency_level": 4.5,
  "recommended_tone": "cosmic"
}
```"""

    monologue = InnerMonologue()
    monologue._llm_router = _Router()
    monologue._router_available = True

    packet = await monologue.think("explore this", _deep_brief(), history=[])

    assert packet.stance == "A stronger grounded stance."
    assert packet.primary_points == ["name the uncertainty"]
    assert packet.transparency == 1.0
    assert packet.tone == "exploratory"
    assert packet.reasoning_source == "kernel+api"


def test_inner_monologue_disables_deepening_when_memory_pressure_probe_fails(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    monologue = InnerMonologue()

    def _raise_memory_error():
        raise OSError("vm unavailable")

    monkeypatch.setattr("core.inner_monologue.psutil.virtual_memory", _raise_memory_error)

    assert monologue._should_use_api(_deep_brief()) is False
    assert tracker.count("inner_monologue", "warning") >= 1
