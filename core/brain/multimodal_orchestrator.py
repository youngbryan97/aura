from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Brain.Multimodal")


_MULTIMODAL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    asyncio.TimeoutError,
)
_ASSET_TIMEOUT_S = 120.0


def _record_multimodal_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "multimodal_orchestrator",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class MultimodalOrchestrator:
    """
    Unified Rendering Engine for Aura's manifestations.
    Synchronizes high-fidelity audio (TTS), visual expressions (SSE),
    and conceptual assets (Diffusion).
    """

    def __init__(self):
        self._is_setup = False
        self.voice_engine = None
        self.event_bus = None
        self.capability_engine = None

    def _setup(self) -> bool:
        if self._is_setup:
            return True
        try:
            self.voice_engine = ServiceContainer.get("voice_engine", default=None)
            self.event_bus = ServiceContainer.get("input_bus", default=None)
            self.capability_engine = ServiceContainer.get("capability_engine", default=None)
            self._is_setup = True
            logger.info("✨ Multimodal Rendering Engine Online.")
            return True
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_multimodal_degradation(
                e,
                action="left multimodal render disabled because setup dependencies were unavailable",
                severity="warning",
            )
            logger.error("Multimodal setup failed: %s", e)
            return False

    async def render(self, content: str, metadata: dict[str, Any] | None = None):
        """
        Renders the content across all available sensory modalities.
        Called by OutputGate for high-fidelity delivery.
        """
        if not self._setup():
            return {"ok": False, "reason": "setup_failed", "scheduled": []}
        text = " ".join(str(content or "").split())
        if not text:
            return {"ok": False, "reason": "empty_content", "scheduled": []}
        metadata = dict(metadata or {})

        tasks = []
        scheduled = []

        # 1. Voice Manifestation
        if self.voice_engine and metadata and metadata.get("voice", True):
            tasks.append(self._track_render_task(self._speak(text), name="Multimodal.voice"))
            scheduled.append("voice")

        # 2. Expression Manifestation (Pulse to UI)
        if self.event_bus:
            tasks.append(
                self._track_render_task(
                    self._pulse_expression(text, metadata),
                    name="Multimodal.expression",
                )
            )
            scheduled.append("expression")

        # 3. Concept Manifestation (Assets)
        if self._manifestation_concepts(text):
            tasks.append(
                self._track_render_task(
                    self._manifest_assets(text),
                    name="Multimodal.assets",
                )
            )
            scheduled.append("assets")

        return {"ok": True, "scheduled": scheduled, "task_count": len(tasks)}

    def _track_render_task(self, coro: Any, *, name: str) -> asyncio.Task:
        task = get_task_tracker().create_task(coro, name=name)
        task.add_done_callback(lambda completed: self._observe_task_result(completed, name=name))
        return task

    @staticmethod
    def _observe_task_result(task: asyncio.Task, *, name: str) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except _MULTIMODAL_RECOVERABLE_ERRORS as exc:
            _record_multimodal_degradation(
                exc,
                action="completed multimodal render with one failed modality task",
                severity="warning",
                extra={"task": name},
            )

    async def _speak(self, content: str) -> Any:
        speak = getattr(self.voice_engine, "speak", None)
        if not callable(speak):
            return None
        return await _maybe_await(speak(content))

    async def _pulse_expression(self, content: str, metadata: dict[str, Any] | None):
        """Analyze content for visual expression markers."""
        if not self.event_bus:
            return
        metadata = metadata or {}

        expression = metadata.get("expression") or self._heuristic_expression(content)

        publish_result = self.event_bus.publish(
            "aura/expression",
            {
                "expression": expression,
                "intensity": metadata.get("intensity", 0.8),
                "timestamp": time.time(),
            },
        )
        await _maybe_await(publish_result)

    def _heuristic_expression(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["happy", "glad", "wonderful", "joy"]):
            return "joy"
        if any(w in text for w in ["sad", "sorry", "unfortunately"]):
            return "sad"
        if any(w in text for w in ["!", "warning", "caution", "alert", "error"]):
            return "alert"
        if any(w in text for w in ["pondering", "researching", "looking", "curious"]):
            return "curiosity"
        return "neutral"

    async def _manifest_assets(self, text: str):
        """Trigger Diffusion/Generation for explicit manifestation tags."""
        concepts = self._manifestation_concepts(text)
        if not concepts:
            return {"ok": True, "generated": []}
        execute = getattr(self.capability_engine, "execute", None)
        if not callable(execute):
            _record_multimodal_degradation(
                RuntimeError("capability_engine.execute unavailable"),
                action="skipped asset manifestation because capability execution was unavailable",
                severity="warning",
                extra={"concept_count": len(concepts)},
            )
            return {"ok": False, "reason": "capability_engine_unavailable", "generated": []}

        skill_name = self._select_asset_skill()
        if not skill_name:
            _record_multimodal_degradation(
                RuntimeError("no image generation skill registered"),
                action="skipped asset manifestation because no image skill was registered",
                severity="warning",
                extra={"concept_count": len(concepts)},
            )
            return {"ok": False, "reason": "skill_unavailable", "generated": []}

        generated = []
        for concept in concepts:
            logger.info("🎨 Multimodal Manifestation: Generating '%s'", concept)
            payload = {
                "prompt": concept,
                "source": "multimodal_orchestrator",
                "metadata": {"modality": "image", "trigger": "manifestation_tag"},
            }
            try:
                result = await asyncio.wait_for(
                    _maybe_await(execute(skill_name, payload)),
                    timeout=_ASSET_TIMEOUT_S,
                )
            except _MULTIMODAL_RECOVERABLE_ERRORS as exc:
                _record_multimodal_degradation(
                    exc,
                    action="continued multimodal render after asset generation failed",
                    severity="warning",
                    extra={"skill": skill_name, "concept": concept[:160]},
                )
                continue
            generated.append({"concept": concept, "skill": skill_name, "result": result})
        return {"ok": bool(generated), "generated": generated}

    @staticmethod
    def _manifestation_concepts(text: str) -> list[str]:
        concepts = []
        for pattern in (r"\[Manifesting:\s*(.+?)\]", r"\[Drawing:\s*(.+?)\]"):
            for match in re.finditer(pattern, text):
                concept = " ".join(match.group(1).split())
                if concept:
                    concepts.append(concept[:500])
        return concepts

    def _select_asset_skill(self) -> str | None:
        skills = getattr(self.capability_engine, "skills", {}) or {}
        if "local_media_generation" in skills:
            return "local_media_generation"
        if "image_generation" in skills:
            return "image_generation"
        return None
