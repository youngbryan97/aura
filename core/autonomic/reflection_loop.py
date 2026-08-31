"""Continuous autonomic reflection loop.

The loop turns bounded ambient perception into durable self-observation. It is
lightweight by design: no foreground LLM calls, no direct source mutation, and
no ungoverned writes. Its output is a governed dream-journal reflection plus a
compact service status that other organs can inspect.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.governance_context import local_internal_governed_scope
from core.runtime.background_policy import (
    background_loop_start_reason,
    constitutive_compute_budget_async,
)
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.AutonomicReflectionLoop")

_RUNTIME_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    TimeoutError,
    asyncio.TimeoutError,
)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


def _default_journal_path() -> Path:
    try:
        from core.config import config

        return config.paths.data_dir / "dreams" / "autonomic_reflections.jsonl"
    except _RUNTIME_ERRORS:
        return state_root() / "data" / "dreams" / "autonomic_reflections.jsonl"


@dataclass(frozen=True)
class AutonomicReflection:
    schema: str = "aura.autonomic_reflection.v1"
    at: float = field(default_factory=time.time)
    trigger: str = "ambient_tick"
    ambient_summary: str = ""
    repo_dirty_count: int = 0
    log_event_count: int = 0
    repair_candidates: tuple[str, ...] = ()
    self_correction_note: str = ""
    throttled_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repair_candidates"] = list(self.repair_candidates)
        return data


class AutonomicReflectionLoop:
    """Run bounded self-observation without waiting for a user prompt."""

    def __init__(
        self,
        *,
        interval_s: float | None = None,
        journal_path: Path | None = None,
    ) -> None:
        self.interval_s = (
            interval_s
            if interval_s is not None
            else _env_float("AURA_AUTONOMIC_REFLECTION_INTERVAL_S", 300.0, minimum=30.0, maximum=7200.0)
        )
        self.journal_path = journal_path or _default_journal_path()
        self.running = False
        self._task: asyncio.Task | None = None
        self._started_at = 0.0
        self._errors = 0
        self._reflections_written = 0
        self._last_reflection: AutonomicReflection | None = None
        self._last_frame_id: int | None = None

    async def start(self) -> None:
        if self.running:
            return
        reason = background_loop_start_reason("autonomic_reflection_loop")
        if reason:
            ServiceContainer.register_instance("autonomic_reflection_loop", self, required=False)
            logger.info("AutonomicReflectionLoop not started: %s", reason)
            return
        self.running = True
        self._started_at = time.time()
        ServiceContainer.register_instance("autonomic_reflection_loop", self, required=False)
        self._task = create_tracked_task(
            self._run_loop(),
            name="Aura.AutonomicReflectionLoop",
        )
        logger.info("AutonomicReflectionLoop ONLINE — %ss interval", self.interval_s)

    async def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while self.running:
            try:
                budget = await constitutive_compute_budget_async(
                    "autonomic_reflection_loop",
                    base_hz=0.1,
                    foreground_hz=0.1,
                    memory_high_hz=0.1,
                    memory_critical_hz=0.1,
                )
                reason = str(getattr(budget, "reason", "") or "")
                await self.reflect_once(
                    throttled_reason=reason if reason not in {"", "nominal"} else ""
                )
                await asyncio.sleep(max(self.interval_s, float(getattr(budget, "interval_s", 0.0) or 0.0)))
            except asyncio.CancelledError:
                raise
            except _RUNTIME_ERRORS as exc:
                self._errors += 1
                record_degradation("autonomic_reflection_loop", exc)
                logger.debug("AutonomicReflectionLoop tick failed: %s", exc)
                await asyncio.sleep(self.interval_s)

    async def reflect_once(self, *, throttled_reason: str = "") -> AutonomicReflection | None:
        frame = self._latest_ambient_frame()
        if frame is None:
            return None
        frame_id = int(getattr(frame, "frame_id", -1) or -1)
        if self._last_frame_id == frame_id and not throttled_reason:
            return self._last_reflection
        reflection = self._build_reflection(frame, throttled_reason=throttled_reason)
        await asyncio.to_thread(self._persist_reflection, reflection)
        self._last_frame_id = frame_id
        self._last_reflection = reflection
        self._reflections_written += 1
        return reflection

    def _latest_ambient_frame(self) -> Any | None:
        stream = ServiceContainer.get("ambient_developer_stream", default=None)
        frame = getattr(stream, "latest_frame", None)
        if frame is not None:
            return frame
        try:
            sample_once = getattr(stream, "sample_once", None)
            if callable(sample_once):
                # The loop should not block on async sampling here. A fresh
                # sample will arrive on the stream's own cadence.
                return None
        except _RUNTIME_ERRORS:
            return None
        return None

    def _build_reflection(self, frame: Any, *, throttled_reason: str = "") -> AutonomicReflection:
        data = frame.to_dict() if hasattr(frame, "to_dict") else dict(frame or {})
        repair_candidates = tuple(str(item) for item in data.get("repair_candidates", [])[:6])
        log_events = data.get("log_events") if isinstance(data.get("log_events"), list) else []
        dirty = int(data.get("git_dirty_count", 0) or 0)
        summary = str(data.get("summary") or "").strip()
        note = self._self_correction_note(
            dirty_count=dirty,
            log_event_count=len(log_events),
            repair_candidates=repair_candidates,
            throttled_reason=throttled_reason or str(data.get("throttled_reason") or ""),
        )
        return AutonomicReflection(
            ambient_summary=summary,
            repo_dirty_count=dirty,
            log_event_count=len(log_events),
            repair_candidates=repair_candidates,
            self_correction_note=note,
            throttled_reason=throttled_reason or str(data.get("throttled_reason") or ""),
        )

    @staticmethod
    def _self_correction_note(
        *,
        dirty_count: int,
        log_event_count: int,
        repair_candidates: tuple[str, ...],
        throttled_reason: str,
    ) -> str:
        if throttled_reason:
            return f"Autonomic work narrowed because {throttled_reason}; preserve foreground stability first."
        if log_event_count:
            return "Recent logs contain actionable warnings; queue diagnosis before expanding autonomy."
        if dirty_count:
            return "Repository state changed; keep tests and closeout tracker aligned with the new code."
        if repair_candidates:
            return "Repair candidates exist; hold them as governed maintenance intentions."
        return "No corrective action required from this ambient frame."

    def _persist_reflection(self, reflection: AutonomicReflection) -> None:
        dream_journal = ServiceContainer.get("dream_journal", default=None)
        append_reflection = getattr(dream_journal, "append_autonomic_reflection", None)
        if callable(append_reflection):
            append_reflection(reflection.to_dict())
            return
        payload = json.dumps(reflection.to_dict(), sort_keys=True) + "\n"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with local_internal_governed_scope(
            "autonomic.reflection_loop.journal",
            domain="file_write",
            receipt_prefix="autonomic-reflection-append",
        ):
            get_file_write_gateway().append_text(
                self.journal_path,
                payload,
                source="autonomic.reflection_loop.journal",
            )

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "schema": "aura.autonomic_reflection_loop.status.v1",
            "interval_s": self.interval_s,
            "reflections_written": self._reflections_written,
            "errors": self._errors,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "latest_reflection": self._last_reflection.to_dict() if self._last_reflection else None,
            "journal_path": str(self.journal_path),
        }

    status = get_status


_REFLECTION_LOOP: AutonomicReflectionLoop | None = None


def get_autonomic_reflection_loop() -> AutonomicReflectionLoop:
    global _REFLECTION_LOOP
    existing = ServiceContainer.get("autonomic_reflection_loop", default=None)
    if isinstance(existing, AutonomicReflectionLoop):
        _REFLECTION_LOOP = existing
        return existing
    if _REFLECTION_LOOP is None:
        _REFLECTION_LOOP = AutonomicReflectionLoop()
    ServiceContainer.register_instance("autonomic_reflection_loop", _REFLECTION_LOOP, required=False)
    return _REFLECTION_LOOP


__all__ = [
    "AutonomicReflection",
    "AutonomicReflectionLoop",
    "get_autonomic_reflection_loop",
]
