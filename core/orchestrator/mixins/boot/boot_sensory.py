from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import Severity, record_degradation

logger = logging.getLogger(__name__)

_BOOT_SENSORY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


class BootSensoryMixin:
    """Provides initialization for sensory inputs & barrier systems."""

    terminal_monitor: Any
    reasoning_queue: Any
    instincts: Any

    def _sensory_boot_report(self) -> dict[str, Any]:
        report = getattr(self, "sensory_boot", None)
        if not isinstance(report, dict):
            report = {"completed": [], "degraded": {}, "registered": {}, "scheduled": []}
            self.sensory_boot = report
        else:
            report.setdefault("completed", [])
            report.setdefault("degraded", {})
            report.setdefault("registered", {})
            report.setdefault("scheduled", [])
        return report

    def _record_boot_sensory_degradation(
        self,
        error: BaseException,
        *,
        lane: str,
        action: str,
        severity: Severity = "warning",
    ) -> None:
        report = self._sensory_boot_report()
        report["degraded"][lane] = {
            "error": _error_summary(error),
            "action": action,
            "severity": severity,
        }
        record_degradation(
            "boot_sensory",
            error,
            severity=severity,
            action=action,
            extra={"lane": lane},
        )

    def _register_sensory_service(self, name: str, instance: Any) -> None:
        ServiceContainer.register_instance(name, instance)
        self._sensory_boot_report()["registered"][name] = instance.__class__.__name__

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _run_sensory_lane(
        self,
        lane: str,
        action_on_failure: str,
        runner: Callable[[], Any],
        *,
        severity: Severity = "warning",
    ) -> Any | None:
        report = self._sensory_boot_report()
        try:
            result = await self._maybe_await(runner())
            if lane not in report["completed"]:
                report["completed"].append(lane)
            return result
        except _BOOT_SENSORY_RECOVERABLE_ERRORS as exc:
            self._record_boot_sensory_degradation(
                exc,
                lane=lane,
                action=action_on_failure,
                severity=severity,
            )
            logger.error("%s sensory boot lane degraded: %s", lane, exc)
            return None

    async def _init_sensory_systems(self):
        """Initialize ears and other sensory inputs."""

        async def _init_ears():
            from core.senses.ears import SovereignEars

            ears = SovereignEars()
            self._register_sensory_service("ears", ears)
            logger.info("👂 Sovereign Ears Active")

        async def _init_vision():
            from core.senses.screen_vision import LocalVision

            vision = LocalVision()
            self._register_sensory_service("vision_engine", vision)
            self._register_sensory_service("vision", vision)
            logger.info("👁️  Sovereign Vision Active")

        await asyncio.gather(
            self._run_sensory_lane(
                "ears",
                "Skipped hearing lane; sensory boot continues with remaining modalities",
                _init_ears,
                severity="warning",
            ),
            self._run_sensory_lane(
                "vision",
                "Skipped vision lane; sensory boot continues with remaining modalities",
                _init_vision,
                severity="warning",
            ),
        )

        async def _terminal_monitor():
            from core.terminal_monitor import get_terminal_monitor

            self.terminal_monitor = get_terminal_monitor()
            self._register_sensory_service("terminal_monitor", self.terminal_monitor)

        await self._run_sensory_lane(
            "terminal_monitor",
            "Terminal monitor unavailable; command-line awareness is degraded",
            _terminal_monitor,
            severity="warning",
        )
        if not hasattr(self, "terminal_monitor"):
            self.terminal_monitor = None

        async def _immune_barriers():
            from core.adaptation.immune_system import ImmuneSystem
            from core.utils.sanitizer import BloodBrainBarrier

            self._register_sensory_service("immune_system", ImmuneSystem())
            self._register_sensory_service("blood_brain_barrier", BloodBrainBarrier())

        await self._run_sensory_lane(
            "immune_barriers",
            "Input immune/sanitizer barriers unavailable; boot health must remain degraded",
            _immune_barriers,
            severity="critical",
        )

        async def _reasoning_queue():
            from core.brain.reasoning_queue import get_reasoning_queue

            self.reasoning_queue = get_reasoning_queue()
            logger.info("🧠 Background Reasoning Queue Ready (Start Deferred)")

        await self._run_sensory_lane(
            "reasoning_queue",
            "Reasoning queue unavailable; background cognition start will be skipped",
            _reasoning_queue,
            severity="warning",
        )

        async def _sensory_instincts():
            from core.senses.sensory_instincts import SensoryInstincts

            self.instincts = SensoryInstincts(self)
            logger.info("✓ Sensory Instincts initialized")

        await self._run_sensory_lane(
            "sensory_instincts",
            "Sensory instincts unavailable; gut-reaction lane is disabled for this boot",
            _sensory_instincts,
            severity="warning",
        )
        if not hasattr(self, "instincts"):
            self.instincts = None

    async def _start_sensory_systems(self):
        if not (hasattr(self, "reasoning_queue") and self.reasoning_queue):
            self._sensory_boot_report()["scheduled"].append("reasoning_queue_skipped")
            return
        try:
            from core.utils.task_tracker import get_task_tracker

            start_coro = self.reasoning_queue.start()
            get_task_tracker().track(start_coro, name="reasoning_queue")
            self._sensory_boot_report()["scheduled"].append("reasoning_queue")
            logger.info("🧠 Background Reasoning Queue Started")
        except _BOOT_SENSORY_RECOVERABLE_ERRORS as exc:
            if "start_coro" in locals() and inspect.iscoroutine(start_coro):
                start_coro.close()
            self._record_boot_sensory_degradation(
                exc,
                lane="reasoning_queue_start",
                action="Reasoning queue task scheduling failed; background reasoning remains stopped",
                severity="warning",
            )

    async def _init_voice_subsystem(self):
        """Initialize the Voice Engine & Multimodal Orchestrator in the background."""

        async def _init_voice():
            async def _voice_lane():
                from core.senses.voice_engine import get_voice_engine

                voice = get_voice_engine()
                if hasattr(voice, "ensure_tts_async"):
                    await voice.ensure_tts_async()
                else:
                    await voice.ensure_models_async()
                self._register_sensory_service("voice_engine", voice)
                logger.info("🎙️  Voice Engine initialized and registered in background")

            await self._run_sensory_lane(
                "voice_engine",
                "Voice engine warmup failed; chat remains text-only until voice recovers",
                _voice_lane,
                severity="warning",
            )

        voice_coro = _init_voice()
        try:
            from core.utils.task_tracker import get_task_tracker

            get_task_tracker().track(voice_coro, name="init_voice")
            self._sensory_boot_report()["scheduled"].append("voice_engine")
        except _BOOT_SENSORY_RECOVERABLE_ERRORS as exc:
            self._record_boot_sensory_degradation(
                exc,
                lane="voice_task_tracker",
                action="Voice task scheduling failed; running voice warmup inline",
                severity="warning",
            )
            await voice_coro

        async def _multimodal_orchestrator():
            from core.brain.multimodal_orchestrator import MultimodalOrchestrator

            self._register_sensory_service("multimodal_orchestrator", MultimodalOrchestrator())

        await self._run_sensory_lane(
            "multimodal_orchestrator",
            "Skipped multimodal orchestrator in voice subsystem; voice/text bridge is degraded",
            _multimodal_orchestrator,
            severity="warning",
        )
