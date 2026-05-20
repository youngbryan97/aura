"""Schema-bound LLM generation with retry, escalation, and honest deferral."""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar, get_origin

from pydantic import BaseModel, ValidationError

from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.runtime.errors import FallbackClassification, Severity, record_degradation

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("Aura.StructuredLLM")

STRUCTURED_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _record_structured_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    record_degradation(
        "structured_llm",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class StructuredLLM:
    """Generate Pydantic objects through Aura's LLM router.

    The class is intentionally conservative: background policy failures defer
    generation, telemetry failures are recorded but never block schema retries,
    and technical/model failures escalate lanes before giving up.
    """

    def __init__(self, model_class: type[T], max_retries: int = 3, llm_router: object | None = None):
        self.model_class = model_class
        self.max_retries = max(1, int(max_retries or 1))
        self._llm_router = llm_router if llm_router is not None else ServiceContainer.get("llm_router")
        self.last_defer_reason = ""

    async def generate(self, prompt: str, context: str | None = None) -> T | None:
        """Generate structured data with autonomous validation repair."""
        self.last_defer_reason = ""
        prompt = str(prompt or "").strip()
        if not prompt:
            self._record_event(
                "empty_prompt",
                detail="structured generation blocked empty prompt before router call",
                severity="warning",
                context={"model_class": self.model_class.__name__},
            )
            return None

        schema = self.model_class.model_json_schema()
        base_prompt = self._with_json_contract(prompt)
        current_prompt = base_prompt
        escalated_tier: str | None = None

        for attempt in range(self.max_retries):
            logger.info(
                "🤖 StructuredLLM: Attempt %d/%d for %s",
                attempt + 1,
                self.max_retries,
                self.model_class.__name__,
            )

            defer_reason = self._background_defer_reason(escalated=bool(escalated_tier))
            if defer_reason:
                self.last_defer_reason = defer_reason
                logger.info("⏸️ StructuredLLM: Deferred %s (%s).", self.model_class.__name__, defer_reason)
                return None

            force_tier = escalated_tier or ("tertiary" if attempt >= 1 else None)
            try:
                response_text, error_code = await self._call_router(
                    current_prompt,
                    context=context,
                    prefer_tier=force_tier,
                    schema=schema,
                    is_background=not escalated_tier,
                )
            except STRUCTURED_RECOVERABLE_ERRORS as exc:
                self._record_event(
                    "technical_failure",
                    detail=str(exc)[:200],
                    severity="warning",
                    context={"model_class": self.model_class.__name__, "attempt": attempt + 1},
                    exc=exc,
                )
                escalated_tier = self._next_escalation_tier(attempt)
                continue

            if self._is_deferred_error(error_code):
                self.last_defer_reason = error_code
                logger.info("⏸️ StructuredLLM: Deferred %s (%s).", self.model_class.__name__, error_code)
                return None

            if not response_text or "ROUTER_ERROR" in response_text:
                detail = (error_code or response_text or "empty")[:200]
                self._record_event(
                    "technical_failure",
                    detail=detail,
                    severity="warning",
                    context={"model_class": self.model_class.__name__, "attempt": attempt + 1},
                )
                escalated_tier = self._next_escalation_tier(attempt)
                continue

            try:
                cleaned_text = self._extract_json(response_text)
                data = json.loads(cleaned_text)
                validated_obj = self.model_class(**data)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning("❌ StructuredLLM: Validation failed on attempt %d: %s", attempt + 1, exc)
                self._record_event(
                    "validation_failed",
                    detail=str(exc)[:200],
                    severity="warning",
                    context={"model_class": self.model_class.__name__, "attempt": attempt + 1},
                    exc=exc,
                )
                if attempt == self.max_retries - 1:
                    logger.error("💀 StructuredLLM: Max retries reached for %s. Giving up.", self.model_class.__name__)
                    return None
                current_prompt = self._correction_prompt(base_prompt, exc)
                escalated_tier = self._next_escalation_tier(attempt)
                continue

            logger.info("✅ StructuredLLM: Successfully validated %s", self.model_class.__name__)
            return validated_obj

        return None

    def _with_json_contract(self, prompt: str) -> str:
        if "GHOST EXAMPLE (Follow this structure exactly):" in prompt:
            return prompt
        return (
            f"{prompt}\n\n"
            "CRITICAL: You MUST respond with a valid JSON object matching the requested schema.\n"
            f"GHOST EXAMPLE (Follow this structure exactly):\n{self._generate_ghost_example()}"
        )

    async def _call_router(
        self,
        prompt: str,
        *,
        context: str | None,
        prefer_tier: str | None,
        schema: dict[str, object],
        is_background: bool,
    ) -> tuple[str, str]:
        if hasattr(self._llm_router, "generate_with_metadata"):
            metadata = await self._llm_router.generate_with_metadata(
                prompt,
                context=context,
                prefer_tier=prefer_tier,
                schema=schema,
                origin="structured_llm",
                is_background=is_background,
            )
            if isinstance(metadata, dict):
                return str(metadata.get("text") or ""), str(metadata.get("error") or "")
            return str(metadata or ""), ""

        response_text = await self._llm_router.generate(
            prompt,
            context=context,
            prefer_tier=prefer_tier,
            schema=schema,
            origin="structured_llm",
            is_background=is_background,
        )
        return str(response_text or ""), ""

    @staticmethod
    def _is_deferred_error(error_code: str) -> bool:
        if error_code in {"foreground_busy", "foreground_quiet_window"}:
            return True
        return error_code.startswith(
            (
                "background_deferred:",
                "failure_lockdown_",
                "conversation_lane_",
            )
        )

    @staticmethod
    def _next_escalation_tier(attempt: int) -> str:
        return "primary" if attempt == 0 else "secondary"

    @staticmethod
    def _correction_prompt(base_prompt: str, error: BaseException) -> str:
        return (
            f"{base_prompt}\n\n"
            f"PREVIOUS ATTEMPT FAILED VALIDATION:\n{error}\n\n"
            "Correct the JSON only. Match the schema exactly, keep all required keys, "
            "and use the correct primitive types."
        )

    def _record_event(
        self,
        reason: str,
        *,
        detail: str,
        severity: str,
        context: dict[str, object],
        exc: BaseException | None = None,
    ) -> None:
        try:
            record_degraded_event(
                "structured_llm",
                reason,
                detail=detail,
                severity=severity,
                classification="background_degraded",
                context=context,
                exc=exc,
            )
        except STRUCTURED_RECOVERABLE_ERRORS as event_exc:
            _record_structured_degradation(
                event_exc,
                action="continued structured generation after degraded-event telemetry failed",
                severity="warning",
                extra={"reason": reason, "model_class": self.model_class.__name__},
            )

    def _extract_json(self, text: str) -> str:
        """Extract a JSON object from raw or fenced model output."""
        text = str(text or "").strip()
        fenced = _JSON_FENCE_RE.search(text)
        if fenced:
            return fenced.group(1).strip()

        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            if end > start:
                return text[start : end + 1]

        return text

    def _background_defer_reason(self, *, escalated: bool = False) -> str:
        if escalated:
            return ""
        try:
            from core.runtime.background_policy import (
                THOUGHT_BACKGROUND_POLICY,
                background_activity_reason,
            )

            orch = ServiceContainer.get("orchestrator", default=None)
            return background_activity_reason(
                orch,
                profile=THOUGHT_BACKGROUND_POLICY,
                require_conversation_ready=True,
            )
        except STRUCTURED_RECOVERABLE_ERRORS as exc:
            _record_structured_degradation(
                exc,
                action="deferred structured background generation because background policy check failed",
                severity="degraded",
            )
            logger.debug("StructuredLLM background defer check failed: %s", exc)
            return "background_policy_unavailable"

    def _generate_ghost_example(self) -> str:
        """Generate a minimal one-line JSON example from the Pydantic fields."""
        try:
            example: dict[str, object] = {}
            for name, field in self.model_class.model_fields.items():
                annotation = field.annotation
                origin = get_origin(annotation)
                if annotation is str:
                    example[name] = "..."
                elif annotation is int:
                    example[name] = 0
                elif annotation is bool:
                    example[name] = False
                elif annotation is list or origin is list:
                    example[name] = []
                elif annotation is dict or origin is dict:
                    example[name] = {}
                else:
                    example[name] = None
            return json.dumps(example)
        except STRUCTURED_RECOVERABLE_ERRORS:
            return "{}"
