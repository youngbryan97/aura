"""
core/resilience/dream_cycle.py
──────────────────────────────
Background process that periodically scans the Dead Letter Queue (DLQ)
and attempts to re-ingest failed thoughts or impulses into the core loop.
"""

from core.runtime.errors import record_degradation
from core.runtime.atomic_writer import atomic_write_text
from core.utils.task_tracker import get_task_tracker
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger("Aura.DreamCycle")

class DreamCycle:
    def __init__(self, orchestrator, dlq_path: Path):
        self.orchestrator = orchestrator
        self.dlq_path = dlq_path
        self._running = False
        self._task = None

    def start(self, interval: float = 300.0):
        """Start the re-ingestion cycle (default 5 minutes)."""
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._cycle_loop(interval))
        logger.info("💤 Dream Cycle active: Re-ingesting dead-letter thoughts every %ds.", interval)

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _cycle_loop(self, interval: float):
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self.process_dreams()
            except Exception as e:
                record_degradation('dream_cycle', e)
                logger.error("Dream Cycle failed: %s", e)

    async def process_dreams(self):
        """Read DLQ and re-enqueue actionable messages."""
        try:
            from core.safe_mode import runtime_feature_enabled

            if not runtime_feature_enabled(self.orchestrator, "dream_cycle", default=True):
                logger.debug("Dream cycle skipped by runtime mode configuration.")
                return
        except Exception as exc:
            record_degradation('dream_cycle', exc)
            logger.debug("Dream cycle runtime-mode check skipped: %s", exc)

        if not self.dlq_path.exists():
            return

        logger.info("🌙 Dream Cycle: Re-processing DLQ messages...")
        
        valid_messages = []
        try:
            def _read_and_clear() -> list[str]:
                with open(self.dlq_path) as f:
                    lines = f.readlines()
                # Clear file after reading to avoid infinite loops.
                atomic_write_text(self.dlq_path, "")
                return lines

            lines = await asyncio.to_thread(_read_and_clear)

            for line in lines:
                try:
                    data = json.loads(line)
                    msg = data.get("message")
                    if msg:
                        valid_messages.append(msg)
                except Exception:
                    continue

        except Exception as e:
            record_degradation('dream_cycle', e)
            logger.error("Failed to read DLQ: %s", e)
            return

        if not valid_messages:
            return

        logger.info("✨ Re-ingesting %d dreams into cognitive loop.", len(valid_messages))
        for msg in valid_messages:
            # Enqueue with slightly lower priority/tag
            if isinstance(msg, str):
                msg = f"Re-processed: {msg}"
            self.orchestrator.enqueue_message(msg)
            await asyncio.sleep(0.1) # Small stagger
