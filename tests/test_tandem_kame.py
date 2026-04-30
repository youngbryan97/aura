"""tests/test_tandem_kame.py — Aura tandem (Kame-style) tests. No real LLMs."""
from __future__ import annotations

import asyncio
from typing import List, Optional

from core.brain.llm.tandem_kame import OracleSignal, TandemKame
from core.brain.llm.tandem_router import (
    TandemFastAdapter, explain_decision, should_use_tandem,
)
from core.brain.llm.tandem_signal_bus import TandemSignalBus, signal_priority


class FakeFastClient:
    def __init__(self, tokens, *, per_token_delay: float = 0.01):
        self.tokens = tokens
        self.per_token_delay = per_token_delay
        self.consumed = 0

    async def astream(self, prompt: str, *, system: Optional[str] = None):
        for tok in self.tokens:
            if self.per_token_delay > 0:
                await asyncio.sleep(self.per_token_delay)
            self.consumed += 1
            yield tok


class FakeSlowClient:
    def __init__(self, schedule):
        self.schedule = schedule
        self.fired = 0

    async def oracle(self, prompt, transcript, *, system=None):
        for delay, sig in self.schedule:
            if delay > 0:
                await asyncio.sleep(delay)
            self.fired += 1
            yield sig

    async def astream_correction(self, signal: OracleSignal):
        text = signal.payload or ""
        for i in range(0, len(text), 4):
            await asyncio.sleep(0)
            yield text[i:i + 4]


class SilentSlowClient:
    async def oracle(self, prompt, transcript, *, system=None):
        await asyncio.sleep(5.0)
        if False:  # pragma: no cover
            yield


async def _collect(agen) -> List[str]:
    out = []
    async for c in agen:
        out.append(c)
    return out


async def test_solo_mode_passthrough_when_slow_silent():
    fast = FakeFastClient(["Hello", " ", "world", "."], per_token_delay=0.005)
    tandem = TandemKame(fast, SilentSlowClient(), slow_timeout=0.2)
    chunks = await _collect(tandem.respond("hi", system="be brief"))
    assert "".join(chunks) == "Hello world."
    assert fast.consumed == 4


async def test_correction_signal_splices_marker_mid_stream():
    fast = FakeFastClient(["The", " sky", " is", " green", "."], per_token_delay=0.02)
    slow = FakeSlowClient([(0.05, OracleSignal(kind="correction", payload="sky is blue", confidence=0.95))])
    tandem = TandemKame(fast, slow)
    seen: List[OracleSignal] = []
    text = "".join(await _collect(tandem.respond("colour?", on_signal=seen.append)))
    assert "[correction: sky is blue]" in text
    assert any(s.kind == "correction" for s in seen)
    assert slow.fired == 1


async def test_retract_halts_fast_stream_and_switches_to_slow():
    fast = FakeFastClient(["Wrong", " answer", " here", " more", " more", " text"], per_token_delay=0.03)
    slow = FakeSlowClient([(0.04, OracleSignal(kind="retract", payload="actually the correct answer is 42", confidence=0.99))])
    tandem = TandemKame(fast, slow)
    text = "".join(await _collect(tandem.respond("ultimate question?")))
    assert fast.consumed < 6
    assert "retracting previous reply" in text
    assert "42" in text
    assert "more text" not in text


async def test_handoff_yields_slow_output_without_retract_marker():
    fast = FakeFastClient(["Quick", " answer"], per_token_delay=0.02)
    slow = FakeSlowClient([(0.03, OracleSignal(kind="handoff", payload="deeper analysis follows", confidence=0.8))])
    text = "".join(await _collect(TandemKame(fast, slow).respond("topic?")))
    assert "deeper analysis follows" in text and "retracting" not in text


async def test_signal_bus_priority_ordering():
    bus = TandemSignalBus()
    sub = bus.subscribe()
    try:
        for k in ("continue", "refine", "correction", "retract", "handoff"):
            await bus.publish(OracleSignal(kind=k, payload=k))
        order = [sub.poll().kind for _ in range(5)]
        assert order == ["retract", "handoff", "correction", "refine", "continue"]
        assert sub.poll() is None
        assert signal_priority("retract") < signal_priority("handoff") < signal_priority("correction")
        assert signal_priority("correction") < signal_priority("refine") < signal_priority("continue")
    finally:
        await sub.aclose()
        await bus.close()


async def test_timeout_when_slow_silent_fast_finishes_solo():
    fast = FakeFastClient(["A", " B", " C", " D"], per_token_delay=0.05)
    text = "".join(await _collect(TandemKame(fast, SilentSlowClient(), slow_timeout=0.1).respond("anything")))
    assert text == "A B C D" and fast.consumed == 4


async def test_router_decision_heuristics():
    assert should_use_tandem("hi", explicit=True).use_tandem
    assert not should_use_tandem("a very long technical prompt " * 20, explicit=False).use_tandem
    assert should_use_tandem("debug this code", intent="debug").use_tandem
    assert not should_use_tandem("hello there", intent="chitchat").use_tandem
    assert should_use_tandem("x " * 100).use_tandem
    assert should_use_tandem("explain why this happens").use_tandem
    assert not should_use_tandem("ok").use_tandem
    assert "tandem=on" in explain_decision("explain why this happens")


async def test_oracle_signal_validates_kind_and_confidence():
    sig = OracleSignal(kind="bogus", payload="x", confidence=99)
    assert sig.kind == "continue" and sig.confidence == 1.0
    sig2 = OracleSignal(kind="correction", confidence="not-a-number")
    assert sig2.confidence == 0.0


async def test_tandem_fast_adapter_falls_back_to_generate():
    class GenOnlyRouter:
        async def generate(self, prompt, *, system_prompt=None, prefer_tier=None):
            return f"[ans:{prompt}|tier={prefer_tier}]"
    out = []
    async for chunk in TandemFastAdapter(GenOnlyRouter(), prefer_tier="fast").astream("hello"):
        out.append(chunk)
    assert out == ["[ans:hello|tier=fast]"]
