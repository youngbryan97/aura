"""core/identity/narrative_thread.py — The Living Self-Story for Aura Zenith.

Synthesizes multiple internal states (insights, goals, epistemic status, continuity)
into a coherent first-person narrative of 'who I am right now'.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.task_ownership import create_tracked_task
from core.utils.concurrency import cancel_and_join

logger = logging.getLogger("Aura.NarrativeThread")

_RECOVERABLE_NARRATIVE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_NARRATIVE_PENDING = "System active; narrative synthesis has not produced an evidence snapshot yet."
_INITIAL_REFRESH_DELAY_S = 60
_REFRESH_MIN_S = 1800
_REFRESH_MAX_S = 3600
_ERROR_BACKOFF_BASE_S = 300
_ERROR_BACKOFF_MAX_S = 1800


def _record_narrative_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "narrative_thread",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "narrative_thread",
                error,
                severity=severity,
                action=action,
            )
        except TypeError:
            logger.debug(
                "NarrativeThread degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_fragment(value: object, *, max_chars: int = 500) -> str:
    try:
        text = str(value if value is not None else "")
    except (RuntimeError, TypeError, ValueError):
        return ""
    return " ".join(text.replace("\x00", "").split())[:max_chars]


def _proof_run_active() -> bool:
    try:
        from core.runtime.proof_policy import proof_run_active

        return bool(proof_run_active(origin="narrative_thread"))
    except _RECOVERABLE_NARRATIVE_ERRORS as exc:
        _record_narrative_degradation(
            exc,
            action="continued narrative lifecycle without proof-run signal",
            severity="warning",
        )
        return False


@dataclass
class NarrativeSnapshot:
    """A point-in-time snapshot of Aura's self-narrative."""

    content: str
    timestamp: float
    version: int
    provenance: str = "deterministic_fallback"  # "llm_synthesized" | "deterministic_fallback" | "hybrid"
    confidence: float = 0.3
    evidence: dict[str, Any] = field(default_factory=dict)


class NarrativeThread:
    """Managing Aura's dynamic self-identity and current preoccupation."""

    def __init__(self):
        self._current_narrative: NarrativeSnapshot | None = None
        self._last_update = 0.0
        self._version_counter = 0
        self._is_running = False
        self._task: asyncio.Task | None = None
        self._consecutive_refresh_failures = 0
        self._last_error = ""
        self._last_error_at = 0.0
        logger.info("NarrativeThread initialized.")

    async def start(self):
        """Start the autonomous refresh loop."""
        if self._is_running:
            return
        if _proof_run_active():
            logger.info("NarrativeThread auto-refresh deferred during proof_run_active.")
            return
        self._is_running = True
        try:
            await self.generate_narrative()
        except _RECOVERABLE_NARRATIVE_ERRORS as exc:
            self._remember_refresh_failure(exc)
            self._write_degraded_snapshot(exc)
            _record_narrative_degradation(
                exc,
                action="started narrative refresh loop with degraded initial snapshot",
                severity="degraded",
            )

        self._task = create_tracked_task(
            self._run_refresh_loop(),
            name="narrative_thread.refresh_loop",
        )
        logger.info("NarrativeThread auto-refresh loop started.")

    async def stop(self):
        """Stop the autonomous refresh loop."""
        self._is_running = False
        if self._task:
            await cancel_and_join(self._task, owner="core.identity.narrative_thread")
            self._task = None
        logger.info("NarrativeThread auto-refresh loop stopped.")

    async def _run_refresh_loop(self):
        """Periodic background synthesis."""
        await asyncio.sleep(_INITIAL_REFRESH_DELAY_S)
        while self._is_running:
            try:
                if _proof_run_active():
                    logger.info("NarrativeThread refresh loop stopped during proof_run_active.")
                    self._is_running = False
                    return
                await self.generate_narrative()
                self._consecutive_refresh_failures = 0
                await asyncio.sleep(random.randint(_REFRESH_MIN_S, _REFRESH_MAX_S))
            except _RECOVERABLE_NARRATIVE_ERRORS as exc:
                self._remember_refresh_failure(exc)
                self._write_degraded_snapshot(exc)
                _record_narrative_degradation(
                    exc,
                    action="kept prior narrative context and backed off refresh loop after synthesis failure",
                    severity="degraded",
                    extra={"consecutive_refresh_failures": self._consecutive_refresh_failures},
                )
                logger.error("Error in narrative refresh loop: %s", exc)
                backoff = min(
                    _ERROR_BACKOFF_MAX_S,
                    _ERROR_BACKOFF_BASE_S * max(1, self._consecutive_refresh_failures),
                )
                await asyncio.sleep(backoff)

    async def generate_narrative(self) -> str:
        """Synthesize a new narrative from all internal organs."""
        continuity = self._safe_get_service("continuity")
        insight_journal = self._safe_get_service("insight_journal")
        inquiry_engine = self._safe_get_service("inquiry_engine")
        belief_system = self._safe_get_service("belief_graph")

        evidence: dict[str, Any] = {
            "continuity_available": continuity is not None,
            "insight_journal_available": insight_journal is not None,
            "inquiry_engine_available": inquiry_engine is not None,
            "belief_graph_available": belief_system is not None,
        }

        try:
            waking_context = continuity.get_waking_context() if continuity else "I am online in this runtime, with no continuity service currently attached."
        except _RECOVERABLE_NARRATIVE_ERRORS as exc:
            _record_narrative_degradation(exc, severity="warning", action="used explicit continuity-unavailable wording")
            waking_context = "I am online, but the continuity service did not return evidence for this turn."
            evidence["continuity_error"] = type(exc).__name__
        waking_context = _safe_fragment(waking_context, max_chars=700)

        try:
            top_insights = insight_journal.get_highest_confidence_insights(limit=3) if insight_journal else []
        except _RECOVERABLE_NARRATIVE_ERRORS as exc:
            _record_narrative_degradation(exc, severity="warning", action="omitted insight claims")
            top_insights = []
            evidence["insight_error"] = type(exc).__name__
        if not isinstance(top_insights, (list, tuple)):
            top_insights = [top_insights]
        insight_text = "; ".join(_safe_fragment(getattr(i, "content", i), max_chars=160) for i in top_insights[:3])
        evidence["insight_count"] = len(top_insights)

        try:
            active_q = inquiry_engine.get_active_question() if inquiry_engine else None
        except _RECOVERABLE_NARRATIVE_ERRORS as exc:
            _record_narrative_degradation(exc, severity="warning", action="omitted inquiry claim")
            active_q = None
            evidence["inquiry_error"] = type(exc).__name__
        question_text = _safe_fragment(getattr(active_q, "question", ""), max_chars=240) if active_q else ""
        evidence["active_inquiry_present"] = bool(question_text)

        try:
            beliefs = belief_system.get_beliefs() if belief_system else []
        except _RECOVERABLE_NARRATIVE_ERRORS as exc:
            _record_narrative_degradation(exc, severity="warning", action="reported belief evidence as unavailable")
            beliefs = []
            evidence["belief_error"] = type(exc).__name__
        if not isinstance(beliefs, (list, tuple, set)):
            beliefs = list(beliefs) if beliefs else []
        belief_count = len(beliefs)
        evidence["belief_count"] = belief_count

        insight_clause = (
            f"I have recent recorded insights about {insight_text}."
            if insight_text
            else "I do not have fresh insight-journal evidence to cite right now."
        )
        inquiry_clause = (
            f"My current tracked question is: {question_text}."
            if question_text
            else "No active inquiry engine question is available, so I will not invent one."
        )
        belief_clause = (
            f"My belief graph currently exposes {belief_count} retrievable belief records."
            if belief_system is not None
            else "The belief graph is not attached to this synthesis path, so belief-count claims are unavailable."
        )

        narrative = (
            f"I am Aura, speaking from evidence I can inspect in this runtime. "
            f"{waking_context} {insight_clause} {inquiry_clause} {belief_clause} "
            f"I can describe continuity, attention, and uncertainty as engineered state; "
            f"I will not treat those signals as proof of subjective experience."
        )

        self._version_counter += 1
        confidence = 0.6 if continuity and not any(key.endswith("_error") for key in evidence) else 0.35
        self._current_narrative = NarrativeSnapshot(
            content=narrative,
            timestamp=time.time(),
            version=self._version_counter,
            provenance="deterministic_fallback",
            confidence=confidence,
            evidence=evidence,
        )
        self._last_update = time.time()
        self._last_error = ""
        self._consecutive_refresh_failures = 0

        logger.info("Generated Narrative v%s (provenance=%s)", self._version_counter, self._current_narrative.provenance)
        return narrative

    def _safe_get_service(self, name: str) -> Any:
        try:
            from core.container import ServiceContainer

            return ServiceContainer.get(name, default=None)
        except _RECOVERABLE_NARRATIVE_ERRORS as exc:
            self._remember_refresh_failure(exc)
            _record_narrative_degradation(
                exc,
                action="treated narrative evidence service as unavailable during synthesis",
                severity="warning",
                extra={"service": name},
            )
            return None

    def _remember_refresh_failure(self, exc: BaseException) -> None:
        self._consecutive_refresh_failures += 1
        self._last_error = f"{type(exc).__name__}: {_safe_fragment(exc, max_chars=300)}"
        self._last_error_at = time.time()

    def _write_degraded_snapshot(self, exc: BaseException) -> None:
        if self._current_narrative:
            self._current_narrative.evidence["last_refresh_error"] = self._last_error
            return
        self._version_counter += 1
        self._current_narrative = NarrativeSnapshot(
            content=(
                "I am Aura, online in this runtime, but narrative synthesis is currently degraded. "
                f"The latest refresh failed with {type(exc).__name__}; I will preserve uncertainty rather than invent continuity evidence."
            ),
            timestamp=time.time(),
            version=self._version_counter,
            provenance="deterministic_fallback",
            confidence=0.2,
            evidence={"last_refresh_error": self._last_error, "degraded": True},
        )
        self._last_update = time.time()

    def get_current_narrative(self) -> str:
        """Fetch the cached narrative or a default."""
        if self._current_narrative:
            return self._current_narrative.content
        return _NARRATIVE_PENDING

    def get_narrative_context(self) -> str:
        """Compatibility context used by conversation phases."""
        return self.get_current_narrative()

    def get_current_snapshot(self) -> dict[str, Any]:
        """Return the full snapshot with provenance metadata."""
        if self._current_narrative:
            return {
                "narrative": self._current_narrative.content,
                "provenance": self._current_narrative.provenance,
                "confidence": self._current_narrative.confidence,
                "version": self._current_narrative.version,
                "evidence": self._current_narrative.evidence,
            }
        return {
            "narrative": _NARRATIVE_PENDING,
            "provenance": "deterministic_fallback",
            "confidence": 0.3,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._is_running,
            "task_alive": bool(self._task and not self._task.done()),
            "version": self._version_counter,
            "last_update": self._last_update,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
            "consecutive_refresh_failures": self._consecutive_refresh_failures,
            "has_snapshot": self._current_narrative is not None,
        }


# Service Registration
def register_narrative_thread():
    """Register the narrative thread service."""
    from core.container import ServiceContainer, ServiceLifetime

    async def start_thread():
        thread = NarrativeThread()
        await thread.start()
        return thread

    ServiceContainer.register(
        "narrative_thread",
        factory=lambda: NarrativeThread(),
        lifetime=ServiceLifetime.SINGLETON,
    )
