"""core/multimodal/coordinator.py

Multimodal Streaming Coordinator
==================================
Single timeline that aligns text streaming, voice utterance, visual
expression pulses, image-generation reveal, and (when enabled)
camera-driven lip sync. The existing MultimodalOrchestrator is treated
as the *renderer* — this module is the *director*: it produces a
`Timeline` of named events that the renderer consumes in order.

The coordinator emits SSE-style events on a per-turn basis:

    { kind: "text_token",      offset_ms, payload: "..." }
    { kind: "voice_start",     offset_ms, payload: { utterance_id } }
    { kind: "voice_end",       offset_ms, payload: { utterance_id } }
    { kind: "expression_pulse",offset_ms, payload: { affect, intensity } }
    { kind: "image_shimmer",   offset_ms, payload: { artifact_id } }
    { kind: "image_reveal",    offset_ms, payload: { artifact_id } }
    { kind: "lip_phoneme",     offset_ms, payload: { phoneme, viseme } }
    { kind: "thinking_breath", offset_ms, payload: { intensity } }
    { kind: "memory_thread",   offset_ms, payload: { scar_id, alpha } }

The frontend connects via /api/multimodal/stream?turn_id=… and renders
the timeline frame-perfect against its own clock so animations don't
drift even under jitter.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger("Aura.Multimodal.Coordinator")


@dataclass
class TimelineEvent:
    kind: str
    offset_ms: float
    payload: Dict[str, Any] = field(default_factory=dict)
    seq: int = 0


@dataclass
class Timeline:
    turn_id: str
    started_at: float = field(default_factory=time.time)
    events: List[TimelineEvent] = field(default_factory=list)
    closed: bool = False

    def add(self, kind: str, payload: Dict[str, Any], *, offset_ms: Optional[float] = None) -> TimelineEvent:
        if offset_ms is None:
            offset_ms = (time.time() - self.started_at) * 1000.0
        ev = TimelineEvent(kind=kind, offset_ms=offset_ms, payload=payload, seq=len(self.events))
        self.events.append(ev)
        return ev


class StreamingCoordinator:
    """Per-turn timeline manager + multi-consumer fan-out."""

    def __init__(self) -> None:
        self._timelines: Dict[str, Timeline] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def open_turn(self, *, turn_id: Optional[str] = None) -> Timeline:
        async with self._lock:
            tid = turn_id or f"turn-{uuid.uuid4().hex[:10]}"
            t = Timeline(turn_id=tid)
            self._timelines[tid] = t
            self._subscribers.setdefault(tid, [])
            return t

    async def emit(self, turn_id: str, kind: str, payload: Dict[str, Any], *, offset_ms: Optional[float] = None) -> TimelineEvent:
        t = self._timelines.get(turn_id)
        if t is None or t.closed:
            raise ValueError(f"unknown_or_closed_turn:{turn_id}")
        ev = t.add(kind, payload, offset_ms=offset_ms)
        for q in list(self._subscribers.get(turn_id, [])):
            try:
                q.put_nowait(ev)
            except Exception:
                pass
        return ev

    async def close(self, turn_id: str) -> None:
        t = self._timelines.get(turn_id)
        if t is None:
            return
        t.closed = True
        for q in list(self._subscribers.get(turn_id, [])):
            q.put_nowait(None)

    async def subscribe(self, turn_id: str) -> AsyncIterator[TimelineEvent]:
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        async with self._lock:
            self._subscribers.setdefault(turn_id, []).append(q)
            t = self._timelines.get(turn_id)
            # Replay everything that's already happened so a late subscriber
            # gets the full timeline starting at offset_ms=0.
            if t is not None:
                for ev in t.events:
                    q.put_nowait(ev)
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    return
                yield ev
        finally:
            async with self._lock:
                if q in self._subscribers.get(turn_id, []):
                    self._subscribers[turn_id].remove(q)


# ─── ergonomic helpers used by chat / voice / vision ──────────────────────


_COORDINATOR: Optional[StreamingCoordinator] = None


def get_coordinator() -> StreamingCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = StreamingCoordinator()
    return _COORDINATOR


async def emit_text_token(turn_id: str, token: str) -> None:
    await get_coordinator().emit(turn_id, "text_token", {"token": token})


async def emit_voice_marker(turn_id: str, kind: str, *, utterance_id: str, **payload: Any) -> None:
    await get_coordinator().emit(turn_id, kind, {"utterance_id": utterance_id, **payload})


async def emit_expression_pulse(turn_id: str, *, affect: str, intensity: float) -> None:
    await get_coordinator().emit(turn_id, "expression_pulse", {"affect": affect, "intensity": float(intensity)})


async def emit_image_shimmer(turn_id: str, *, artifact_id: str) -> None:
    await get_coordinator().emit(turn_id, "image_shimmer", {"artifact_id": artifact_id})


async def emit_image_reveal(turn_id: str, *, artifact_id: str) -> None:
    await get_coordinator().emit(turn_id, "image_reveal", {"artifact_id": artifact_id})


async def emit_thinking_breath(turn_id: str, *, intensity: float) -> None:
    await get_coordinator().emit(turn_id, "thinking_breath", {"intensity": float(intensity)})


async def emit_memory_thread(turn_id: str, *, scar_id: str, alpha: float) -> None:
    await get_coordinator().emit(turn_id, "memory_thread", {"scar_id": scar_id, "alpha": float(alpha)})


__all__ = [
    "Timeline",
    "TimelineEvent",
    "StreamingCoordinator",
    "get_coordinator",
    "emit_text_token",
    "emit_voice_marker",
    "emit_expression_pulse",
    "emit_image_shimmer",
    "emit_image_reveal",
    "emit_thinking_breath",
    "emit_memory_thread",
]
