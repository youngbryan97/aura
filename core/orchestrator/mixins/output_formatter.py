"""Output Formatter Mixin for RobustOrchestrator.
Extracts response styling, identity guard filtering, and systemic thought emissions.
"""

import asyncio
import inspect
import logging
import time
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log

logger = logging.getLogger(__name__)
_OUTPUT_FORMATTER_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    Exception,
)


def _record_output_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("output_formatter", error, severity=severity, action=action)


def _dispose_awaitable(result: Any) -> None:
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()


class OutputFormatterMixin:
    """Handles response formatting, filtering, and system thought emission."""

    def _post_process_response(self, text: str) -> str:
        return text.strip()

    def _filter_output(self, text: str) -> str:
        """Personality-driven output filtering (Aura v10.0)."""
        if not text:
            return ""

        # Identity Flux Guard: Neutralize assistant-speak
        banned_phrases = {
            "How can I help you today?": "what's on your mind?",
            "Is there anything else I can help you with?": "so, what else?",
            "I'd be happy to assist": "I'll take a look",
            "Certainly!": "",  # Remove preamble
            "Absolutely!": "",
            "Great question!": "",
        }
        for banned, replacement in banned_phrases.items():
            if banned in text:
                logger.warning("Identity Flux Guard triggered: neutralizing '%s'", banned)
                text = text.replace(banned, replacement).strip()

        # Personality Engine Filter
        pe = getattr(self, "personality_engine", None)
        if pe:
            try:
                filtered = pe.filter_response(text)
                if isinstance(filtered, str):
                    text = filtered
                if hasattr(pe, "apply_lexical_style"):
                    styled = pe.apply_lexical_style(text)
                    if isinstance(styled, str):
                        text = styled
            except _OUTPUT_FORMATTER_ERRORS as exc:
                _record_output_degradation(
                    exc,
                    action="returned partially styled response after personality filter failed",
                )
                logger.debug("Filter failed: %s", exc)

        # Single final stabilization pass — strip role artifacts and broken-lane boilerplate.
        # NOTE: Do NOT call stabilize_user_facing_response without user_message context,
        # as the floor logic cannot make informed decisions without knowing the prompt.
        try:
            from core.synthesis import strip_role_artifacts

            text = strip_role_artifacts(text)
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug(
                "Suppressed %s in core.orchestrator.mixins.output_formatter: %s",
                type(_exc).__name__,
                _exc,
            )

        return text

    def _emit_thought_stream(self, thought):
        """Helper to emit autonomous thoughts/monologues to UI"""
        if (
            hasattr(self, "cognitive_engine")
            and self.cognitive_engine
            and hasattr(self.cognitive_engine, "_emit_thought")
        ):
            emitted = self.cognitive_engine._emit_thought(thought)
            if inspect.isawaitable(emitted):
                try:
                    from core.utils.task_tracker import get_task_tracker

                    get_task_tracker().create_task(
                        emitted,
                        name="output_formatter.emit_thought",
                    )
                except RuntimeError:
                    _dispose_awaitable(emitted)
            return
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Autonomous Thought",
                str(thought or ""),
                level="info",
                category="Autonomy",
            )
        except _OUTPUT_FORMATTER_ERRORS as exc:
            _record_output_degradation(
                exc,
                action="dropped thought-stream fallback emission after emitter failed",
            )
            logger.debug("Thought stream fallback emit failed: %s", exc)

    def _emit_eternal_record(self):
        """Archives a snapshot of the system's current state into the Eternal Record (Sync trigger)."""

        async def _run_eternal_snapshot():
            try:
                from core.utils.run_bound import run_io_bound

                from core.resilience.eternal_record import EternalRecord

                # We use the configured data dir for the record store
                record_store = config.paths.home_dir / "eternal_archive"
                archivist = EternalRecord(record_store)

                kg_path = config.paths.data_dir / "knowledge.db"

                # Massive snapshot operation must NOT block the main thread
                snapshot_dir = await run_io_bound(archivist.create_snapshot, kg_path)

                if snapshot_dir:
                    self._emit_thought_stream(
                        f"Eternal Record Snapshot secured: {snapshot_dir.name}"
                    )
            except _OUTPUT_FORMATTER_ERRORS as e:
                _record_output_degradation(
                    e,
                    action="skipped eternal-record snapshot after async archive task failed",
                    severity="error",
                )
                logger.debug("Eternal record snapshot failed: %s", e)

        try:
            asyncio.get_running_loop()
            from core.utils.task_tracker import get_task_tracker

            get_task_tracker().create_task(
                _run_eternal_snapshot(),
                name="output_formatter.eternal_snapshot",
            )
        except (RuntimeError, ValueError):
            # Sync fallback (for tests without loop)
            try:
                from core.resilience.eternal_record import EternalRecord

                record_store = config.paths.home_dir / "eternal_archive"
                archivist = EternalRecord(record_store)
                archivist.create_snapshot(config.paths.data_dir / "knowledge.db")
            except _OUTPUT_FORMATTER_ERRORS as e:
                _record_output_degradation(
                    e,
                    action="skipped synchronous eternal snapshot after archive write failed",
                )
                capture_and_log(e, {"module": __name__})

    def _emit_neural_pulse(self):
        """Emit system health to thought stream."""
        try:
            from core.thought_stream import get_emitter

            # Zenith Heartbeat: Integrate Soul Dominant Drive
            drive_info = "Neutral"
            if hasattr(self, "soul") and self.soul:
                try:
                    drive = self.soul.get_dominant_drive()
                    drive_info = f"{drive.name} ({drive.urgency:.2f})"
                except _OUTPUT_FORMATTER_ERRORS as _e:
                    _record_output_degradation(
                        _e,
                        action="emitted neural pulse with neutral drive info after drive lookup failed",
                    )
                    logger.debug("Drive info retrieval failed for neural pulse: %s", _e)

            ls = getattr(self, "liquid_state", None)
            mood = ls.get_mood() if ls else "Stable"
            get_emitter().emit(
                "Neural Pulse",
                f"System Active (Mood: {mood} | Drive: {drive_info})",
                level="info",
                category="Physiology",
                cycle=self.status.cycle_count,
            )
            self._last_pulse = time.time()
        except _OUTPUT_FORMATTER_ERRORS as _e:
            _record_output_degradation(
                _e,
                action="dropped neural pulse after thought-stream emission failed",
            )
            logger.debug("Neural pulse emit failed: %s", _e)
