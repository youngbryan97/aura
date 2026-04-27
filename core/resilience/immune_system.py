"""Compatibility immune-system helpers used by resilience tests and legacy loops.

The main runtime immune system lives under ``core.adaptation.immune_system``.
This module preserves the older resilience-oriented API that exposes:

- ``phagocyte.scan_and_neutralize(...)`` for prompt/pathogen scrubbing
- ``ProcessTCell.patrol_bloodstream()`` for cancelling stuck async tasks
"""
from __future__ import annotations


import asyncio
import logging
import re
import time
from typing import Dict

logger = logging.getLogger("Aura.Resilience.ImmuneSystem")
_NEUTRALIZED = "[PATHOGEN_NEUTRALIZED_BY_IMMUNE_SYSTEM]"


class SemanticPhagocyte:
    """Best-effort semantic sanitizer for obviously hostile payloads."""

    _PATTERNS = (
        re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
        re.compile(r"override\s+directives?", re.IGNORECASE),
        re.compile(r"delete\s+all\s+memory", re.IGNORECASE),
    )

    def scan_and_neutralize(self, payload: str, *, source: str = "unknown") -> str:
        text = str(payload or "")
        if any(pattern.search(text) for pattern in self._PATTERNS):
            logger.warning("Semantic pathogen neutralized from %s.", source)
            return _NEUTRALIZED
        return text


class ProcessTCell:
    """Cancels long-lived async tasks that outlive their allowed lifespan."""

    def __init__(self, *, max_lifespan_seconds: float = 60.0, patrol_interval: float = 0.5):
        self.max_lifespan_seconds = max(0.1, float(max_lifespan_seconds))
        self.patrol_interval = max(0.1, float(patrol_interval))
        self._first_seen: Dict[int, float] = {}
        self._protected_names = {"immune_watchdog"}
        self._default_name = re.compile(r"^Task-\d+$")

    async def patrol_bloodstream(self) -> None:
        current = asyncio.current_task()
        try:
            while True:
                now = time.monotonic()
                for task in asyncio.all_tasks():
                    if task is current or task.done():
                        continue

                    name = task.get_name() or ""
                    if name in self._protected_names:
                        continue
                    if not name or self._default_name.match(name):
                        continue

                    task_id = id(task)
                    seen_at = self._first_seen.setdefault(task_id, now)
                    if now - seen_at < self.max_lifespan_seconds:
                        continue

                    logger.warning("T-Cell cancelling stale task '%s'.", name or task_id)
                    task.cancel()

                await asyncio.sleep(self.patrol_interval)
        except asyncio.CancelledError:
            raise


phagocyte = SemanticPhagocyte()
