from __future__ import annotations

import asyncio
import inspect
import logging
import os
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

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _explicit_env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _eager_local_sensory_boot_enabled() -> bool:
    explicit = _explicit_env_flag("AURA_EAGER_LOCAL_SENSORY_BOOT")
    if explicit is not None:
        return explicit
    if _env_flag("AURA_HEADLESS") or _env_flag("AURA_SAFE_BOOT_DESKTOP"):
        return False
    return True


class BootSensoryMixin:
    """Provides initialization for sensory inputs & barrier systems."""

    terminal_monitor: Any
    reasoning_queue: Any
    instincts: Any

    def _sensory_boot_report(self) -> dict[str, Any]:
        report = getattr(self, "sensory_boot", None)
        if not isinstance(report, dict):
            report = {
                "completed": [],
                "degraded": {},
                "registered": {},
                "scheduled": [],
                "skipped": [],
            }
            self.sensory_boot = report
        else:
            report.setdefault("completed", [])
            report.setdefault("degraded", {})
            report.setdefault("registered", {})
            report.setdefault("scheduled", [])
            report.setdefault("skipped", [])
        return report

    def _skip_boot_sensory_lane(self, lane: str, reason: str) -> None:
        report = self._sensory_boot_report()
        skipped = report.setdefault("skipped", [])
        if lane not in skipped:
            skipped.append(lane)
        report.setdefault("skip_reasons", {})[lane] = reason
        logger.info("%s sensory boot lane deferred: %s", lane, reason)

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

    def _register_sensory_service(
        self,
        name: str,
        instance: Any,
        *,
        required: bool = False,
        failure_policy: str = "degrade_with_receipt",
    ) -> None:
        ServiceContainer.register_instance(
            name,
            instance,
            required=required,
            failure_policy=failure_policy,
        )
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
        eager_local_sensory = _eager_local_sensory_boot_enabled()

        async def _init_ears():
            from core.senses.ears import SovereignEars

            ears = SovereignEars()
            self._register_sensory_service("ears", ears)
            logger.info("Sovereign Ears Active")

        async def _init_vision():
            from core.senses.screen_vision import LocalVision

            vision = LocalVision()
            self._register_sensory_service("vision_engine", vision)
            self._register_sensory_service("vision", vision)
            logger.info(" Sovereign Vision Active")

        if eager_local_sensory:
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
        else:
            reason = (
                "local audio/screen adapters are lazy in headless or desktop-safe boot "
                "to keep native media imports out of the main cognitive runtime"
            )
            self._skip_boot_sensory_lane("ears", reason)
            self._skip_boot_sensory_lane("vision", reason)

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
            logger.info("Background Reasoning Queue Ready (Start Deferred)")

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

        async def _source_body():
            from core.soma.source_body import get_source_body

            source_body = get_source_body()
            self._register_sensory_service("source_body", source_body)
            await source_body.start()
            logger.info("Source-Body Proprioception Active (awakening deferred)")

        if _env_flag("AURA_ENABLE_SOURCE_BODY", True):
            await self._run_sensory_lane(
                "source_body",
                "Source-body proprioception unavailable; code-change awareness is offline",
                _source_body,
                severity="warning",
            )
        else:
            self._skip_boot_sensory_lane(
                "source_body", "disabled via AURA_ENABLE_SOURCE_BODY"
            )

    async def _start_sensory_systems(self):
        if not (hasattr(self, "reasoning_queue") and self.reasoning_queue):
            self._sensory_boot_report()["scheduled"].append("reasoning_queue_skipped")
            return
        try:
            from core.utils.task_tracker import get_task_tracker

            start_coro = self.reasoning_queue.start()
            get_task_tracker().track(start_coro, name="reasoning_queue")
            self._sensory_boot_report()["scheduled"].append("reasoning_queue")
            logger.info("Background Reasoning Queue Started")
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
                logger.info(" Voice Engine initialized and registered in background")

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
