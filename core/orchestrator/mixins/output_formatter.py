"""Output Formatter Mixin for RobustOrchestrator.
Extracts response styling, identity guard filtering, and systemic thought emissions.
"""
from core.runtime.errors import record_degradation
import asyncio
import inspect
import logging
import time
from typing import Any, Optional

from core.utils.exceptions import capture_and_log
from core.config import config

logger = logging.getLogger(__name__)


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
            "As an AI model": "honestly",
            "I am an AI assistant": "I'm Aura",
            "Is there anything else I can help you with?": "so, what else?",
            "I'd be happy to assist": "I'll take a look",
            "Certainly!": "", # Remove preamble
            "Absolutely!": "",
            "Great question!": ""
        }
        for banned, replacement in banned_phrases.items():
            if banned in text:
                logger.warning(f"🚨 Identity Flux Guard triggered: neutralizing '{banned}'")
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
                return text
                if inspect.isawaitable(filtered):
                    _dispose_awaitable(filtered)
            except Exception as exc:
                record_degradation('output_formatter', exc)
                logger.debug("Filter failed: %s", exc)
                
        return text

    def _emit_thought_stream(self, thought):
        """Helper to emit autonomous thoughts/monologues to UI"""
        if hasattr(self, "cognitive_engine") and self.cognitive_engine and hasattr(self.cognitive_engine, "_emit_thought"):
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
        except Exception as exc:
            record_degradation('output_formatter', exc)
            logger.debug("Thought stream fallback emit failed: %s", exc)

    def _emit_eternal_record(self):
        """Archives a snapshot of the system's current state into the Eternal Record (Sync trigger)."""
        async def _run_eternal_snapshot():
            try:
                from core.resilience.eternal_record import EternalRecord
                from core.utils.run_bound import run_io_bound
                # We use the configured data dir for the record store
                record_store = config.paths.home_dir / "eternal_archive"
                archivist = EternalRecord(record_store)
                
                kg_path = config.paths.data_dir / "knowledge.db"
                
                # Massive snapshot operation must NOT block the main thread
                snapshot_dir = await run_io_bound(archivist.create_snapshot, kg_path)
                
                if snapshot_dir:
                    self._emit_thought_stream(f"🏺 Eternal Record Snapshot secured: {snapshot_dir.name}")
            except Exception as e:
                record_degradation('output_formatter', e)
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
            except Exception as e:
                record_degradation('output_formatter', e)
                capture_and_log(e, {'module': __name__})

    def _emit_neural_pulse(self):
        """Emit system health to thought stream."""
        try:
            from core.thought_stream import get_emitter
            # Zenith Heartbeat: Integrate Soul Dominant Drive
            drive_info = "Neutral"
            if hasattr(self, 'soul') and self.soul:
                try:
                    drive = self.soul.get_dominant_drive()
                    drive_info = f"{drive.name} ({drive.urgency:.2f})"
                except Exception as _e:
                    record_degradation('output_formatter', _e)
                    logger.debug("Drive info retrieval failed for neural pulse: %s", _e)

            ls = getattr(self, "liquid_state", None)
            mood = ls.get_mood() if ls else "Stable"
            get_emitter().emit("Neural Pulse", f"System Active (Mood: {mood} | Drive: {drive_info})", level="info", category="Physiology", cycle=self.status.cycle_count)
            self._last_pulse = time.time()
        except Exception as _e:
            record_degradation('output_formatter', _e)
            logger.debug("Neural pulse emit failed: %s", _e)
