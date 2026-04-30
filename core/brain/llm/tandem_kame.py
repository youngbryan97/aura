"""
core/brain/llm/tandem_kame.py — Aura's "speak while thinking" tandem.

Mapping: fast frontend = Aura 7B/14B lane; slow backend = 32B/72B Cortex/Solver;
oracle signal = async correction the slow lane emits as soon as it has it.

Fast lane streams tokens, drains the bus between yields. On signal:
  retract  -> halt fast stream, switch to slow output (with marker)
  handoff  -> switch to slow output (no retract marker)
  correction -> splice "[correction: ...]" inline
  refine     -> splice "[refine: ...]" annotation
  continue   -> no-op
Written from scratch for Aura — no external code copied.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from core.brain.llm.tandem_signal_bus import TandemSignalBus, signal_priority

logger = logging.getLogger("Brain.TandemKame")
_VALID_KINDS = frozenset({"correction", "refine", "retract", "continue", "handoff"})


@dataclass
class OracleSignal:
    """Async correction/refinement from the slow lane."""
    kind: str
    payload: str = ""
    confidence: float = 1.0
    ts: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            logger.warning("OracleSignal: unknown kind %r -> 'continue'", self.kind)
            self.kind = "continue"
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0


SignalCallback = Callable[[OracleSignal], Any]


class TandemKame:
    """Run fast + slow lanes in parallel, splicing oracle signals into output."""

    def __init__(self, fast_client: Any, slow_client: Any, *,
                 signal_bus: Optional[TandemSignalBus] = None,
                 slow_timeout: float = 6.0,
                 correction_template: str = " [correction: {payload}] "):
        self.fast = fast_client
        self.slow = slow_client
        self.bus = signal_bus or TandemSignalBus()
        self.slow_timeout = float(slow_timeout)
        self.correction_template = correction_template

    async def respond(self, prompt: str, *, system: Optional[str] = None,
                      on_signal: Optional[SignalCallback] = None) -> AsyncIterator[str]:
        """Yield text chunks while slow lane critiques in parallel."""
        sub = self.bus.subscribe()
        slow_task = asyncio.create_task(self._slow_loop(prompt, system=system),
                                        name="tandem_kame.slow")
        try:
            async for chunk in self._fast_loop(prompt, system=system, subscription=sub,
                                               slow_task=slow_task, on_signal=on_signal):
                yield chunk
        finally:
            if not slow_task.done():
                slow_task.cancel()
                try: await slow_task
                except (asyncio.CancelledError, Exception): pass  # noqa: BLE001,E701
            await sub.aclose()

    async def _slow_loop(self, prompt: str, *, system: Optional[str]) -> None:
        try:
            transcript: list[str] = []
            if hasattr(self.slow, "oracle"):
                stream = self.slow.oracle(prompt, transcript, system=system)
                if inspect.isasyncgen(stream):
                    async for sig in stream:
                        await self._publish(sig)
                elif inspect.isawaitable(stream):
                    await self._publish_many(await stream)
                else:
                    for sig in stream or []:
                        await self._publish(sig)
            elif hasattr(self.slow, "agenerate"):
                text = await self.slow.agenerate(prompt, system=system)
                await self._publish(OracleSignal(kind="refine", payload=str(text or "")))
            else:
                logger.warning("TandemKame slow client has no oracle()/agenerate()")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("TandemKame slow loop error: %s", exc)

    async def _publish(self, sig: Any) -> None:
        if sig is None:
            return
        if not isinstance(sig, OracleSignal):
            if isinstance(sig, dict):
                sig = OracleSignal(kind=str(sig.get("kind", "continue")),
                                   payload=str(sig.get("payload", "")),
                                   confidence=float(sig.get("confidence", 1.0)),
                                   metadata=dict(sig.get("metadata", {})))
            else:
                return
        await self.bus.publish(sig)

    async def _publish_many(self, payload: Any) -> None:
        if payload is None:
            return
        if isinstance(payload, (list, tuple)):
            for item in payload:
                await self._publish(item)
        else:
            await self._publish(payload)

    async def _emit_signal(self, sig: OracleSignal,
                           on_signal: Optional[SignalCallback]) -> Optional[str]:
        """Run callback, return inline render for correction/refine, else None."""
        if on_signal is not None:
            try:
                res = on_signal(sig)
                if inspect.isawaitable(res): await res  # noqa: E701
            except Exception: pass  # noqa: BLE001,E701
        if sig.kind == "correction" and sig.payload:
            return self.correction_template.format(payload=sig.payload)
        if sig.kind == "refine" and sig.payload:
            return f" [refine: {sig.payload}] "
        return None

    async def _fast_loop(self, prompt: str, *, system: Optional[str], subscription,
                         slow_task: asyncio.Task,
                         on_signal: Optional[SignalCallback]) -> AsyncIterator[str]:
        if not hasattr(self.fast, "astream"):
            raise TypeError("fast client must implement astream(prompt, *, system=None)")
        last_signal_ts = time.monotonic()
        warned_silent = False
        async_gen = self.fast.astream(prompt, system=system)
        try:
            async for chunk in async_gen:
                signals = subscription.drain()
                signals.sort(key=lambda s: signal_priority(getattr(s, "kind", "continue")))
                for sig in signals:
                    last_signal_ts = time.monotonic()
                    if sig.kind in ("retract", "handoff"):
                        if on_signal is not None:
                            try:
                                res = on_signal(sig)
                                if inspect.isawaitable(res): await res  # noqa: E701
                            except Exception: pass  # noqa: BLE001,E701
                        async for out in self._handoff_stream(sig, prefix=(sig.kind == "retract")):
                            yield out
                        return
                    rendered = await self._emit_signal(sig, on_signal)
                    if rendered:
                        yield rendered
                if chunk:
                    yield chunk
                if (not warned_silent and (time.monotonic() - last_signal_ts) > self.slow_timeout
                        and not slow_task.done()):
                    warned_silent = True
                    logger.info("TandemKame: slow lane silent >%.1fs — fast solo", self.slow_timeout)
        finally:
            close = getattr(async_gen, "aclose", None)
            if close is not None:
                try: await close()
                except Exception: pass  # noqa: BLE001,E701
        for sig in subscription.drain():
            if sig.kind in ("retract", "handoff"):
                async for out in self._handoff_stream(sig, prefix=(sig.kind == "retract")):
                    yield out
                return
            rendered = await self._emit_signal(sig, on_signal)
            if rendered:
                yield rendered

    async def _handoff_stream(self, signal: OracleSignal, *, prefix: bool) -> AsyncIterator[str]:
        if prefix:
            yield "\n[retracting previous reply — deeper model has corrected it]\n"
        stream_fn = getattr(self.slow, "astream_correction", None)
        if stream_fn is not None:
            try:
                stream = stream_fn(signal)
                if inspect.isasyncgen(stream):
                    async for tok in stream:
                        if tok: yield tok  # noqa: E701
                    return
                if inspect.isawaitable(stream):
                    text = await stream
                    if text: yield str(text)  # noqa: E701
                    return
            except Exception: pass  # noqa: BLE001,E701
        if signal.payload:
            yield signal.payload
