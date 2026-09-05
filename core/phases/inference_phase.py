from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.runtime.service_registry import get_runtime_service
from core.kernel.bridge import Phase
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.service_names import ServiceNames
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.InferencePhase")

_INFERENCE_ERRORS = (
    AttributeError,
    ImportError,
    json.JSONDecodeError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_USER_ORIGINS = {"user", "voice", "admin"}
_VALID_MOMENTUM = {"stalled", "flowing", "intense"}


def _service_get(container: Any, name: str, *, default: Any = None) -> Any:
    if container is not None and hasattr(container, "get"):
        try:
            return container.get(name, default=default)
        except TypeError:
            return container.get(name) or default
    return get_runtime_service(name, default=default)


def _record_inference_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "inference_phase",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("inference_phase", error, severity=severity, action=action)
        except TypeError:
            logger.debug("InferencePhase degradation could not be recorded: %s", signature_exc)


def _safe_text(value: Any, *, max_chars: int = 500) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def _normalize_hooks(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    hooks = [_safe_text(item, max_chars=160) for item in candidates]
    return [hook for hook in hooks if hook][:3]


def _normalize_momentum(value: Any) -> str:
    momentum = str(value or "flowing").strip().lower()
    return momentum if momentum in _VALID_MOMENTUM else "flowing"


def _extract_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise ValueError("inference output did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("inference JSON root was not an object")
    return payload


def _normalize_inference_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "implicit_intent": _safe_text(data.get("implicit_intent")),
        "user_subtext": _safe_text(data.get("user_subtext") or data.get("affective_subtext")),
        "momentum": _normalize_momentum(data.get("momentum")),
        "conversation_hooks": _normalize_hooks(data.get("conversation_hooks")),
    }


class InferencePhase(Phase):
    """
    Extract subtext and implicit intent from foreground user messages.

    The output is advisory context for later response phases. Parse/router
    failures are explicit in cognition modifiers so chat quality does not
    silently depend on a hidden no-op.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        priority = bool(kwargs.get("priority", False))
        if state.cognition.current_origin not in _USER_ORIGINS:
            return state

        modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
        state.cognition.modifiers = modifiers

        try:
            router = _service_get(self.container, ServiceNames.LLM_ROUTER, default=None)
            if router is None:
                modifiers["deep_inference_status"] = {
                    "status": "unavailable",
                    "reason": "llm_router_missing",
                }
                return state

            inference_data = await self._call_router(
                router,
                prompt=self._build_prompt(objective),
                priority=priority,
            )
            data = _normalize_inference_data(_extract_json_object(inference_data))

            modifiers["inferred_intent"] = data["implicit_intent"]
            modifiers["user_subtext"] = data["user_subtext"]
            modifiers["momentum"] = data["momentum"]
            modifiers["conversation_hooks"] = data["conversation_hooks"]
            modifiers["deep_inference_status"] = {
                "status": "ok",
                "hooks": len(data["conversation_hooks"]),
            }

            logger.info(
                "Deep Inference: Intent='%s', Subtext='%s'",
                data["implicit_intent"],
                data["user_subtext"],
            )
        except _INFERENCE_ERRORS as exc:
            modifiers["deep_inference_status"] = {
                "status": "degraded",
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            }
            _record_inference_degradation(
                exc,
                action="kept foreground response pipeline alive with explicit degraded inference status",
                severity="warning",
            )
            logger.warning("InferencePhase failed: %s", exc)

        return state

    @staticmethod
    def _build_prompt(objective: str | None) -> str:
        return (
            "Analyze the following user message for IMPLICIT INTENT, AFFECTIVE SUBTEXT, and CONVERSATION HOOKS. "
            "Do not respond to the user. Only return a JSON object with: "
            '{\n'
            '  "implicit_intent": "...",\n'
            '  "user_subtext": "...",\n'
            '  "momentum": "stalled|flowing|intense",\n'
            '  "conversation_hooks": ["list of 2-3 specific topics or entities to address"]\n'
            '}\n\n'
            f"User Message: {objective or ''}"
        )

    @staticmethod
    async def _call_router(router: Any, *, prompt: str, priority: bool) -> Any:
        think = getattr(router, "think", None)
        if callable(think):
            return await think(
                prompt,
                system_prompt="You are Aura's subtext processor. Extract the unsaid.",
                prefer_tier="fast",
                priority=priority,
            )
        route = getattr(router, "route", None)
        if callable(route):
            return await route(
                prompt,
                system_prompt="You are Aura's subtext processor. Extract the unsaid.",
                prefer_tier="fast",
                priority=priority,
            )
        raise AttributeError("llm_router has neither think nor route")
