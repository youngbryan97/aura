"""Automatic self-knowing observer.

This bridge turns runtime events into self-knowledge frames without waiting for
the user to ask an introspective question. It binds phenomenal knowing and
recursive self-knowing into the normal tick/turn path.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


def _now() -> float:
    return time.time()


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return max(lo, min(hi, parsed))


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8", "replace")).hexdigest()[:16]


class AutoEventKind(str, Enum):
    TIMER = "timer"
    CHAT_TURN = "chat_turn"
    MEMORY_WRITE = "memory_write"
    SELF_REPORT = "self_report"
    RUNTIME_FAILURE = "runtime_failure"
    CHOICE = "choice"


class IntrospectionMode(str, Enum):
    NONE = "none"
    QUIET = "quiet"
    ACTIVE = "active"
    REPAIR = "repair"


@dataclass(slots=True)
class AutomaticSelfKnowingFrame:
    timestamp: float
    event_kind: str
    event_digest: str
    status: str
    introspection_mode: IntrospectionMode
    self_knowing_pressure: float
    phenomenal_digest: str | None = None
    recursive_digest: str | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["introspection_mode"] = self.introspection_mode.value
        data["self_knowing_pressure"] = round(self.self_knowing_pressure, 4)
        return data


class AutomaticSelfKnowingKernel:
    """Continuously binds runtime events into calibrated self-knowledge."""

    def __init__(
        self,
        *,
        recursive_self_knowing: Any | None = None,
        phenomenal_knowing: Any | None = None,
        live_substrate: Any | None = None,
        max_pending: int = 64,
    ) -> None:
        self.recursive_self_knowing = recursive_self_knowing
        self.phenomenal_knowing = phenomenal_knowing
        self.live_substrate = live_substrate
        self._pending: deque[tuple[AutoEventKind, dict[str, Any], str]] = deque(maxlen=max_pending)
        self._frames: deque[AutomaticSelfKnowingFrame] = deque(maxlen=128)
        self._latest = AutomaticSelfKnowingFrame(
            timestamp=_now(),
            event_kind=AutoEventKind.TIMER.value,
            event_digest="initial",
            status="watching",
            introspection_mode=IntrospectionMode.QUIET,
            self_knowing_pressure=0.25,
            notes=("automatic_self_knowing_initialized",),
        )
        self._frames.append(self._latest)

    def __getstate__(self) -> None:
        raise TypeError("AutomaticSelfKnowingKernel is live runtime state, not serializable identity.")

    def enqueue_event(
        self,
        kind: AutoEventKind | str,
        payload: Mapping[str, Any] | None = None,
        *,
        source: str = "",
    ) -> None:
        event_kind = kind if isinstance(kind, AutoEventKind) else AutoEventKind(str(kind))
        self._pending.append((event_kind, dict(payload or {}), source or "unknown"))

    def observe_event(
        self,
        kind: AutoEventKind | str,
        payload: Mapping[str, Any] | None = None,
        *,
        source: str = "",
    ) -> AutomaticSelfKnowingFrame:
        event_kind = kind if isinstance(kind, AutoEventKind) else AutoEventKind(str(kind))
        frame = self._process(event_kind, dict(payload or {}), source or "unknown")
        self._latest = frame
        self._frames.append(frame)
        return frame

    def tick(self) -> AutomaticSelfKnowingFrame:
        if self._pending:
            kind, payload, source = self._pending.popleft()
            return self.observe_event(kind, payload, source=source)
        return self.observe_event(
            AutoEventKind.TIMER,
            {"standing_self_audit": True, "last_frame": self._latest.as_dict()},
            source="automatic_self_knowing_tick",
        )

    def controls(self) -> dict[str, Any]:
        latest = self._latest
        return {
            "automatic_self_knowing_active": True,
            "latest_frame_digest": _digest(latest.as_dict()),
            "latest_status": latest.status,
            "introspection_mode": latest.introspection_mode.value,
            "self_knowing_pressure": round(latest.self_knowing_pressure, 4),
            "pending_events": len(self._pending),
            "frames": len(self._frames),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "aura.automatic_self_knowing.snapshot.v1",
            "active": True,
            "latest": self._latest.as_dict(),
            "controls": self.controls(),
        }

    def should_interrupt(self, user_input: str) -> tuple[bool, str]:
        text = str(user_input or "").lower()
        if any(token in text for token in ("wrong", "you failed", "that broke", "not what i asked")):
            return True, "user_flagged_runtime_mismatch"
        if self._latest.introspection_mode == IntrospectionMode.REPAIR:
            return True, "self_knowing_repair_frame_active"
        return False, ""

    def witness(self) -> dict[str, Any]:
        return {
            "type": "automatic_self_knowing_witness_not_self",
            "latest_digest": _digest(self._latest.as_dict()),
            "frame_count": len(self._frames),
        }

    def _process(
        self,
        kind: AutoEventKind,
        payload: Mapping[str, Any],
        source: str,
    ) -> AutomaticSelfKnowingFrame:
        notes: list[str] = [f"source={source}"]
        recursive_digest: str | None = None
        phenomenal_digest: str | None = None

        pressure = 0.22
        mode = IntrospectionMode.QUIET
        status = "observed"
        if kind == AutoEventKind.RUNTIME_FAILURE:
            pressure = 0.88
            mode = IntrospectionMode.REPAIR
            status = "failure_requires_reorientation"
        elif kind in {AutoEventKind.SELF_REPORT, AutoEventKind.CHOICE}:
            pressure = 0.62
            mode = IntrospectionMode.ACTIVE
        elif kind == AutoEventKind.CHAT_TURN:
            pressure = 0.42

        if self.phenomenal_knowing is not None:
            try:
                if kind == AutoEventKind.MEMORY_WRITE:
                    receipt = self.phenomenal_knowing.mark_memory(
                        str(payload.get("memory_key") or "automatic_memory_write"),
                        dict(payload),
                        context={"source": source, "kind": kind.value},
                    )
                    phenomenal_digest = receipt.get("frame_digest")
                elif kind == AutoEventKind.SELF_REPORT:
                    frame = self.phenomenal_knowing.undergo_first_person_report(
                        str(payload.get("text") or payload.get("message") or ""),
                        context={"source": source, "kind": kind.value},
                    )
                    phenomenal_digest = _digest(frame.as_dict())
                elif kind == AutoEventKind.CHOICE:
                    trace = self.phenomenal_knowing.record_word_choice(
                        prompt=str(payload.get("prompt") or payload.get("context") or ""),
                        chosen_text=str(payload.get("chosen") or payload.get("choice") or ""),
                        alternatives=tuple(payload.get("alternatives") or ()),
                        controls=dict(payload.get("controls") or {}),
                    )
                    phenomenal_digest = _digest(trace.as_dict())
                else:
                    controls = getattr(self.phenomenal_knowing, "generation_controls", lambda: {})()
                    phenomenal_digest = _digest(controls)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                notes.append(f"phenomenal_bridge_unavailable:{type(exc).__name__}")

        if self.recursive_self_knowing is not None:
            try:
                claim = str(
                    payload.get("claim")
                    or payload.get("message")
                    or payload.get("objective")
                    or f"automatic observation of {kind.value}"
                )
                evidence = tuple(str(item) for item in payload.get("evidence", ()) if str(item).strip())
                if not evidence:
                    evidence = (f"event:{kind.value}", f"source:{source}")
                contradictions = tuple(
                    str(item) for item in payload.get("contradictions", ()) if str(item).strip()
                )
                frame = self.recursive_self_knowing.observe_claim(
                    claim,
                    confidence=_clamp(payload.get("confidence"), default=0.58),
                    evidence=evidence,
                    contradictions=contradictions,
                )
                recursive_digest = frame.claim_digest
                pressure = max(pressure, frame.introspection_pressure)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                notes.append(f"recursive_bridge_unavailable:{type(exc).__name__}")

        if self.live_substrate and hasattr(self.live_substrate, "ingest_narration"):
            try:
                self.live_substrate.ingest_narration(
                    f"automatic self-knowing observed {kind.value}",
                    decision_context={"pressure": pressure, "mode": mode.value},
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                notes.append("live_substrate_reentry_failed")

        return AutomaticSelfKnowingFrame(
            timestamp=_now(),
            event_kind=kind.value,
            event_digest=_digest({"kind": kind.value, "payload": payload, "source": source}),
            status=status,
            introspection_mode=mode,
            self_knowing_pressure=_clamp(pressure),
            phenomenal_digest=phenomenal_digest,
            recursive_digest=recursive_digest,
            notes=tuple(notes),
        )


_KERNEL: AutomaticSelfKnowingKernel | None = None


def get_default_automatic_self_knowing_kernel() -> AutomaticSelfKnowingKernel:
    global _KERNEL
    if _KERNEL is None:
        from core.consciousness.phenomenal_knowing import get_phenomenal_knowing_kernel
        from core.consciousness.recursive_self_knowing import get_recursive_self_knowing_kernel

        _KERNEL = AutomaticSelfKnowingKernel(
            recursive_self_knowing=get_recursive_self_knowing_kernel(),
            phenomenal_knowing=get_phenomenal_knowing_kernel(),
        )
    return _KERNEL


def reset_automatic_self_knowing_kernel_for_tests() -> None:
    global _KERNEL
    _KERNEL = None
