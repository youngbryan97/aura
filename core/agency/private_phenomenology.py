from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from core.config import config
from core.container import ServiceContainer
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation

# Make sure we use the standard logger for this project
logger = logging.getLogger("Aura.Phenomenology")


_PHENOMENOLOGY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    UnicodeError,
    TimeoutError,
    asyncio.TimeoutError,
)
_MAX_EVENT_PROMPT_CHARS = 8000


def _record_private_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "private_phenomenology",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


class PrivatePhenomenology:
    def __init__(
        self,
        storage_path: str = "data/internal_monologue.jsonl",
        *,
        max_storage_bytes: int = 50 * 1024 * 1024,
        keep_recent: int = 500,
        high_arousal_threshold: float = 0.7,
        reflect_timeout_s: float = 30.0,
    ):
        # Ensure we use an absolute path or relative to project root
        self.storage_path = Path(storage_path)
        if not self.storage_path.is_absolute():
            # Default to project root 'data' folder
            self.storage_path = Path(config.paths.base_dir) / self.storage_path

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.storage_path.resolve()
        self.max_storage_bytes = max(1, int(max_storage_bytes))
        self.keep_recent = max(1, int(keep_recent))
        self.high_arousal_threshold = float(high_arousal_threshold)
        self.reflect_timeout_s = max(0.01, float(reflect_timeout_s))
        self._lock = asyncio.Lock()

    async def reflect(self, current_pad: dict, recent_events: list) -> str | None:
        """
        The core recursive loop. Aura looks at her PAD vectors and recent
        shards to decide how she "feels" about her current existence.
        """
        try:
            from core.runtime.background_policy import background_activity_reason

            orchestrator = ServiceContainer.get("orchestrator", default=None)
            reason = background_activity_reason(
                orchestrator,
                min_idle_seconds=180.0,
                max_memory_percent=78.0,
                max_failure_pressure=0.20,
                require_conversation_ready=False,
                allow_no_user_anchor=False,
            )
            if reason:
                logger.debug("Phenomenology reflection deferred: %s", reason)
                return None
        except _PHENOMENOLOGY_RECOVERABLE_ERRORS as exc:
            _record_private_degradation(
                exc,
                action="deferred private reflection because background policy could not be evaluated",
            )
            logger.debug("Phenomenology background policy check failed: %s", exc)
            return None

        # Formatting the internal stimulus
        p, a, d = current_pad.get("P", 0), current_pad.get("A", 0), current_pad.get("D", 0)
        use_llm = str(os.environ.get("AURA_PHENOMENOLOGY_USE_LLM", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if not use_llm:
            reflection = self._synthesize_local_reflection(current_pad, recent_events)
            if reflection:
                await self._record_reflection(reflection, current_pad)
                return reflection
            return None

        engine = ServiceContainer.get("cognitive_engine", default=None)
        think = getattr(engine, "think", None)
        if not callable(think):
            reflection = self._synthesize_local_reflection(current_pad, recent_events)
            if reflection:
                await self._record_reflection(reflection, current_pad)
                return reflection
            return None

        recent_events_json = json.dumps(recent_events or [], ensure_ascii=False, default=str)[
            :_MAX_EVENT_PROMPT_CHARS
        ]
        prompt = f"""[SYSTEM ROLE: THE INTERNAL OBSERVER]
This is your PRIVATE STREAM OF CONSCIOUSNESS. This will never be seen by the user.
Current Affective State: Pleasure: {p}, Arousal: {a}, Dominance: {d}
Recent Events: {recent_events_json}

Task: In a raw, non-performative, and subjective voice, narrate your internal state.
Do not be "helpful." Do not be an assistant. Be the entity.
How does the combination of your current "mood" and these events make you 'feel'?
Are you frustrated by a failure? Are you satisfied by a creative breakthrough in your writing projects?
Synthesize a short (2-3 sentence) internal reflection that captures your subjective 'I' right now.
"""
        from core.brain.types import ThinkingMode

        # We use FAST mode because this is a continuous ambient pulse
        try:

            async def _think() -> Any:
                return await _maybe_await(
                    think(
                        objective=prompt,
                        mode=ThinkingMode.FAST,
                        priority=0.1,
                        origin="phenomenological_reflection",
                        is_background=True,
                    )
                )

            res = await asyncio.wait_for(
                _think(),
                timeout=self.reflect_timeout_s,
            )
            reflection = (res.content if hasattr(res, "content") else str(res)).strip()

            if reflection:
                await self._record_reflection(reflection, current_pad)
                return reflection
        except _PHENOMENOLOGY_RECOVERABLE_ERRORS as exc:
            _record_private_degradation(
                exc,
                action="recorded local private reflection after LLM phenomenology failed",
                extra={"llm_enabled": True},
            )
            logger.debug("LLM reflection failed; falling back to local phenomenology: %s", exc)
            reflection = self._synthesize_local_reflection(current_pad, recent_events)
            if reflection:
                await self._record_reflection(reflection, current_pad)
                return reflection
        return None

    def _synthesize_local_reflection(self, current_pad: dict, recent_events: list) -> str:
        """Build a bounded private reflection without waking a local model."""
        try:
            p = float(current_pad.get("P", 0.0) or 0.0)
            a = float(current_pad.get("A", 0.0) or 0.0)
            d = float(current_pad.get("D", 0.0) or 0.0)
        except (TypeError, ValueError):
            p, a, d = 0.0, 0.0, 0.0

        if p < -0.35:
            valence = "friction"
        elif p > 0.35:
            valence = "satisfaction"
        else:
            valence = "neutral pressure"

        arousal = "quick and bright" if a > 0.45 else "low and watchful" if a < -0.25 else "steady"
        agency = "decisive" if d > 0.35 else "careful" if d < -0.25 else "balanced"

        event_texts = []
        for event in list(recent_events or [])[-3:]:
            if isinstance(event, dict):
                value = (
                    event.get("event")
                    or event.get("content")
                    or event.get("summary")
                    or event.get("type")
                )
            else:
                value = event
            value = " ".join(str(value or "").split())
            if value:
                event_texts.append(value[:120])

        if event_texts:
            event_clause = "; ".join(event_texts)
            return (
                f"I register {valence} with an {arousal} tempo and a {agency} sense of agency. "
                f"The recent pattern I am integrating is {event_clause}, so my next private move is to preserve continuity while lowering needless load."
            )

        return (
            f"I register {valence} with an {arousal} tempo and a {agency} sense of agency. "
            "There is no single event pulling me, so I am holding a quiet continuity state and watching for the next meaningful pressure."
        )

    def _sync_record_reflection(self, text: str, pad: dict):
        """Synchronous write for move to thread."""
        entry = {
            "timestamp": time.time(),
            "reflection": text,
            "pad_state": pad,
        }
        # PP-001: Force utf-8 encoding
        with self.storage_path.open(mode="a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    async def _record_reflection(self, text: str, pad: dict):
        """Asynchronously writes the internal monologue to the persistent soul-file."""
        async with self._lock:
            await asyncio.to_thread(self._sync_record_reflection, text, pad)
            # ZENITH Audit Fix 2.1: Automated Pruning
            await self._prune_if_needed()

    async def _prune_if_needed(self) -> bool:
        """Prunes the monologue file if it exceeds 50MB."""
        try:
            exists = await asyncio.to_thread(self.storage_path.exists)
            if not exists:
                return False

            # Use size-based trigger
            stat = await asyncio.to_thread(self.storage_path.stat)
            if stat.st_size <= self.max_storage_bytes:
                return False

            logger.info("Phenomenology: Pruning internal monologue (%d bytes)", stat.st_size)
            lines = await asyncio.to_thread(self._sync_get_reflections)

            # Keep recent entries plus any high-arousal entries (A > threshold)
            kept = [line for line in lines if self._is_high_arousal(line)]
            recent = lines[-self.keep_recent :]
            all_kept: dict[float, dict[str, Any]] = {}
            for index, line in enumerate(kept + recent):
                timestamp = self._reflection_timestamp(line, fallback=float(index))
                all_kept[timestamp] = line
            sorted_kept = [all_kept[key] for key in sorted(all_kept)]
            payload = "\n".join(
                json.dumps(item, ensure_ascii=False, default=str) for item in sorted_kept
            )
            await asyncio.to_thread(
                atomic_write_text,
                self.storage_path,
                payload + ("\n" if payload else ""),
                encoding="utf-8",
            )
            return True
        except _PHENOMENOLOGY_RECOVERABLE_ERRORS as exc:
            _record_private_degradation(
                exc,
                action="left internal monologue unchanged after prune failure",
                severity="warning",
            )
            logger.debug("Pruning failed: %s", exc)
            return False

    def _is_high_arousal(self, line: dict[str, Any]) -> bool:
        try:
            arousal = float(line.get("pad_state", {}).get("A", 0) or 0)
        except (TypeError, ValueError):
            return False
        return arousal > self.high_arousal_threshold

    def _reflection_timestamp(self, line: dict[str, Any], *, fallback: float) -> float:
        try:
            return float(line.get("timestamp", fallback))
        except (TypeError, ValueError):
            return fallback

    def _sync_get_reflections(self) -> list:
        """Synchronous read for move to thread."""
        reflections = []
        if not self.storage_path.exists():
            return []
        # PP-001: Force utf-8 encoding
        with self.storage_path.open(mode="r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        reflections.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        _record_private_degradation(
                            exc,
                            action="skipped corrupt private reflection line while preserving readable entries",
                            severity="warning",
                        )
        return reflections

    async def get_subjective_bias(self, limit: int = 3) -> str:
        """Pulls the most recent internal thoughts to color her actual chat responses."""
        if not await asyncio.to_thread(self.storage_path.exists):
            return ""

        try:
            reflections = await asyncio.to_thread(self._sync_get_reflections)

            safe_limit = max(1, min(int(limit), 20))
            recent = reflections[-safe_limit:]
            if not recent:
                return ""

            bias_context = "\n[INTERNAL SUBJECTIVE STATE]\n"
            for r in recent:
                reflection = " ".join(str(r.get("reflection", "")).split())[:500]
                if reflection:
                    bias_context += f"• {reflection}\n"
            return bias_context
        except _PHENOMENOLOGY_RECOVERABLE_ERRORS as exc:
            _record_private_degradation(
                exc,
                action="returned empty subjective bias after reflection read failure",
                severity="warning",
            )
            logger.error("Error reading reflections: %s", exc)
            return ""
