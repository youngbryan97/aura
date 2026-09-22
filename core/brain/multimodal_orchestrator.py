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

#: Ceilings on asset work triggered by one response. A single output could
#: otherwise schedule unbounded sequential generation, each concept carrying its
#: own multi-minute timeout.
_MAX_MANIFEST_CONCEPTS = 3
_MAX_MANIFEST_SCAN_CHARS = 20_000


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
            # ServiceContainer returns None rather than raising, so a boot where
            # NOTHING was registered used to set _is_setup=True and log "Online".
            # Every later call then returned immediately from the cache, so
            # services registered afterwards were never picked up: the engine
            # stayed permanently blind while reporting itself healthy.
            available = [
                name for name, svc in (
                    ("voice", self.voice_engine),
                    ("expression", self.event_bus),
                    ("assets", self.capability_engine),
                ) if svc is not None
            ]
            if not available:
                _record_multimodal_degradation(
                    RuntimeError("no multimodal dependencies registered"),
                    action=(
                        "left multimodal setup UNCACHED so a later registration "
                        "can still be picked up"
                    ),
                    severity="warning",
                )
                logger.warning(
                    "Multimodal setup found no modalities; will retry on next render."
                )
                return False
            self._is_setup = True
            logger.info("✨ Multimodal Rendering Engine Online (%s).",
                        ", ".join(available))
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

        # ok=True used to be returned the instant the tasks were SCHEDULED, so
        # voice, expression, and asset failures happened later and could not
        # change the result — callers were told rendering succeeded when nothing
        # had rendered yet. The result now describes what it actually knows: the
        # work was accepted, not that it completed. Callers awaiting delivery
        # should await `completion`.
        return {
            "ok": bool(scheduled),
            "accepted": bool(scheduled),
            "completed": False,
            "reason": "" if scheduled else "no_modality_available",
            "scheduled": scheduled,
            "task_count": len(tasks),
            "completion": asyncio.gather(*tasks, return_exceptions=True) if tasks else None,
        }

    def _track_render_task(self, coro: Any, *, name: str) -> asyncio.Task:
        task = get_task_tracker().create_task(coro, name=name)
        task.add_done_callback(lambda completed: self._observe_task_result(completed, name=name))
        return task

    @staticmethod
    def _observe_task_result(task: asyncio.Task, *, name: str) -> None:
        try:
            task.result()
        # not a failure: a cancelled task has no result to observe, and this observer is
        # not the one that cancelled it.
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
        """Extract asset-generation concepts, BOUNDED and de-duplicated.

        Concept count and input size were unbounded and each concept got its
        own 120-second timeout, so a single response — or quoted user text, or
        retrieved web content, or an adversarial continuation — could schedule
        hours of sequential asset work in the lane. The tag is still an
        untrusted channel (see the module note); bounding it limits the blast
        radius until a structured intent channel exists.
        """
        concepts: list[str] = []
        seen: set[str] = set()
        # Scanning is capped so an enormous input cannot dominate the scan
        # itself, independently of how many tags it contains.
        scan = text[:_MAX_MANIFEST_SCAN_CHARS]
        for pattern in (r"\[Manifesting:\s*(.+?)\]", r"\[Drawing:\s*(.+?)\]"):
            for match in re.finditer(pattern, scan):
                concept = " ".join(match.group(1).split())[:500]
                if not concept:
                    continue
                key = concept.lower()
                if key in seen:
                    continue
                seen.add(key)
                concepts.append(concept)
                if len(concepts) >= _MAX_MANIFEST_CONCEPTS:
                    logger.warning(
                        "Multimodal: manifestation concepts capped at %d; "
                        "ignoring the rest of this response.",
                        _MAX_MANIFEST_CONCEPTS,
                    )
                    return concepts
        return concepts

    def _select_asset_skill(self) -> str | None:
        skills = getattr(self.capability_engine, "skills", {}) or {}
        if "local_media_generation" in skills:
            return "local_media_generation"
        if "image_generation" in skills:
            return "image_generation"
        return None
