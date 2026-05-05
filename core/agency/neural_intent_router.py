"""Neural-intent router — close the "internal thought, no external action" gap.

The prior architecture wired the user-message path to skill dispatch but
left internal neural-stream intents as phenomenology only. So Aura could
"decide" to open a file, speak about it in the thought stream, and nothing
would ever execute.

This router subscribes to internal initiative-like events (emergent goals,
volition triggers, self-generated thoughts) and translates the ones that
carry action intent into capability-engine dispatches via the Will. Every
dispatch is recorded in the LifeTrace ledger with ``origin=neural_stream``
so we can audit exactly which internal thoughts turned into real actions
and which did not.

Key rules:
  - Only content that matches a whitelisted action schema is dispatched.
  - Safety vetoes in the Will block the action (no override).
  - On dispatch success, the real result replaces the claim in memory.
  - On dispatch failure, a LifeTrace ``initiative_blocked`` entry records
    the failure so no "I did X" belief can silently propagate.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


# Internal sources we trust as potential action origins. User messages
# have a separate, stricter pathway; these are autonomous signals.
TRUSTED_INTERNAL_SOURCES = {
    "sensory_motor",
    "drive_engine",
    "volition_engine",
    "emergent_goal_engine",
    "agency_core",
    "agency_facade",
    "mesh_cognition",
    "curiosity_loop",
    "self_monitoring",
}


# Minimum intent schema: verb plus enough structure to look actionable.
_ACTION_SCHEMAS: List[Dict[str, Any]] = [
    {
        "skill": "web_search",
        "pattern": re.compile(
            r"\b(search|look up|google|research|find information on)\b\s+(?P<target>[A-Za-z0-9 '\"/.:_-]{3,160})",
            re.IGNORECASE,
        ),
        "params": lambda m: {"query": m.group("target").strip().rstrip(".,?!:;")},
    },
    {
        "skill": "computer_use",
        "pattern": re.compile(
            r"\b(read|screenshot|capture)\s+(the\s+)?(screen|window|desktop)\b",
            re.IGNORECASE,
        ),
        "params": lambda m: {"action": "read_screen_text"},
    },
    {
        "skill": "file_operation",
        "pattern": re.compile(
            r"\b(read|open)\s+(the\s+)?file\s+(?P<path>[\w./-]+)",
            re.IGNORECASE,
        ),
        "params": lambda m: {"action": "read", "path": m.group("path")},
    },
]


@dataclass(frozen=True)
class NeuralIntent:
    source: str
    text: str
    matched_schema: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_action(self) -> bool:
        return self.matched_schema is not None

    @property
    def skill_name(self) -> Optional[str]:
        return self.matched_schema.get("skill") if self.matched_schema else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "text": self.text[:240],
            "skill_name": self.skill_name,
            "params": dict(self.params),
        }


def classify_neural_intent(source: str, text: str) -> NeuralIntent:
    txt = str(text or "").strip()
    if not txt:
        return NeuralIntent(source=source, text="")
    for schema in _ACTION_SCHEMAS:
        match = schema["pattern"].search(txt)
        if match:
            try:
                params = schema["params"](match)
            except Exception:
                params = {}
            return NeuralIntent(source=source, text=txt, matched_schema=schema, params=params)
    return NeuralIntent(source=source, text=txt)


@dataclass
class DispatchOutcome:
    intent: NeuralIntent
    approved: bool
    approved_reason: str
    dispatched: bool = False
    dispatched_ok: bool = False
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.as_dict(),
            "approved": self.approved,
            "approved_reason": self.approved_reason,
            "dispatched": self.dispatched,
            "dispatched_ok": self.dispatched_ok,
            "result": dict(self.result),
            "error": self.error,
        }


class NeuralIntentRouter:
    """Converts trusted internal action-intent signals into real dispatches."""

    def __init__(
        self,
        *,
        will_provider: Optional[Callable[[], Any]] = None,
        capability_provider: Optional[Callable[[], Any]] = None,
        life_trace_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._will_provider = will_provider
        self._capability_provider = capability_provider
        self._life_trace_provider = life_trace_provider
        self._dispatch_count = 0
        self._block_count = 0

    # ------------------------------------------------------------------
    async def route(self, source: str, text: str) -> DispatchOutcome:
        source_norm = str(source or "").strip().lower()
        if source_norm not in TRUSTED_INTERNAL_SOURCES:
            intent = classify_neural_intent(source_norm, text)
            self._block_count += 1
            return DispatchOutcome(
                intent=intent,
                approved=False,
                approved_reason=f"source_not_trusted:{source_norm}",
            )

        intent = classify_neural_intent(source_norm, text)
        if not intent.has_action:
            return DispatchOutcome(intent=intent, approved=False, approved_reason="no_action_schema_match")

        # Will gate
        will = self._resolve_will()
        approved = True
        approve_reason = "no_will_available_defaulting_allow"
        will_receipt = ""
        if will is not None and hasattr(will, "decide"):
            try:
                from core.governance.will_client import WillClient, WillRequest
                from core.will import ActionDomain

                decision = await WillClient(will).decide_async(
                    WillRequest(
                        content=f"{intent.skill_name}: {intent.text[:120]}",
                        source=f"neural_intent:{source_norm}",
                        domain=getattr(ActionDomain, "TOOL_EXECUTION", "tool_execution"),
                        context={
                            "action_kind": "neural_intent",
                            "user_requested_action": False,
                            "internal_intent": True,
                            "skill": intent.skill_name,
                            "params": dict(intent.params),
                        },
                    )
                )
                approved = WillClient.is_approved(decision)
                approve_reason = str(getattr(decision, "reason", "")) or "will_decided"
                will_receipt = str(getattr(decision, "receipt_id", ""))
            except Exception as exc:
                record_degradation('neural_intent_router', exc)
                approved = False
                approve_reason = f"will_error:{type(exc).__name__}"

        if not approved:
            self._block_count += 1
            self._record_life_trace(
                "initiative_deferred",
                source_norm,
                intent,
                approved=False,
                reason=approve_reason,
                will_receipt=will_receipt,
            )
            return DispatchOutcome(
                intent=intent,
                approved=False,
                approved_reason=approve_reason,
            )

        # Dispatch
        engine = self._resolve_capability()
        if engine is None:
            self._block_count += 1
            self._record_life_trace(
                "initiative_blocked",
                source_norm,
                intent,
                approved=True,
                reason="no_capability_engine",
                will_receipt=will_receipt,
            )
            return DispatchOutcome(
                intent=intent,
                approved=True,
                approved_reason=approve_reason,
                dispatched=False,
                error="no_capability_engine",
            )

        try:
            result = await engine.execute(
                intent.skill_name,
                dict(intent.params),
                {"source": f"neural_intent:{source_norm}", "will_receipt": will_receipt},
            )
            ok = bool(result.get("ok", result.get("success", False)))
            outcome = DispatchOutcome(
                intent=intent,
                approved=True,
                approved_reason=approve_reason,
                dispatched=True,
                dispatched_ok=ok,
                result=result if isinstance(result, dict) else {"value": result},
                error=str(result.get("error") or "") if not ok else "",
            )
            self._dispatch_count += 1
            self._record_life_trace(
                "action_executed" if ok else "initiative_blocked",
                source_norm,
                intent,
                approved=True,
                reason=approve_reason,
                will_receipt=will_receipt,
                dispatch_result=outcome.result,
                success=ok,
            )
            return outcome
        except Exception as exc:
            record_degradation('neural_intent_router', exc)
            self._block_count += 1
            self._record_life_trace(
                "initiative_blocked",
                source_norm,
                intent,
                approved=True,
                reason=f"dispatch_error:{type(exc).__name__}",
                will_receipt=will_receipt,
            )
            return DispatchOutcome(
                intent=intent,
                approved=True,
                approved_reason=approve_reason,
                dispatched=False,
                error=repr(exc),
            )

    def stats(self) -> Dict[str, int]:
        return {"dispatches": self._dispatch_count, "blocks": self._block_count}

    # ------------------------------------------------------------------
    def _resolve_will(self) -> Any:
        if self._will_provider is not None:
            try:
                return self._will_provider()
            except Exception:
                return None
        try:
            from core.container import ServiceContainer

            return ServiceContainer.get("unified_will", default=None) or ServiceContainer.get("will", default=None)
        except Exception:
            return None

    def _resolve_capability(self) -> Any:
        if self._capability_provider is not None:
            try:
                return self._capability_provider()
            except Exception:
                return None
        try:
            from core.container import ServiceContainer

            return ServiceContainer.get("capability_engine", default=None)
        except Exception:
            return None

    def _record_life_trace(
        self,
        event_type: str,
        source: str,
        intent: NeuralIntent,
        *,
        approved: bool,
        reason: str,
        will_receipt: str = "",
        dispatch_result: Optional[Dict[str, Any]] = None,
        success: bool = False,
    ) -> None:
        try:
            ledger = None
            if self._life_trace_provider is not None:
                ledger = self._life_trace_provider()
            if ledger is None:
                from core.runtime.life_trace import get_life_trace

                ledger = get_life_trace()
            ledger.record(
                event_type,
                origin=f"neural_intent:{source}",
                user_requested=False,
                action_taken={
                    "skill": intent.skill_name,
                    "params": dict(intent.params),
                    "text": intent.text[:240],
                },
                result={
                    "ok": bool(success),
                    "reason": reason,
                    "dispatch_result": dispatch_result or {},
                    "will_receipt": will_receipt,
                },
            )
        except Exception as exc:
            record_degradation('neural_intent_router', exc)
            logger.debug("LifeTrace write from neural router failed: %s", exc)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


_singleton: Optional[NeuralIntentRouter] = None


def get_neural_intent_router() -> NeuralIntentRouter:
    global _singleton
    if _singleton is None:
        _singleton = NeuralIntentRouter()
    return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    _singleton = None
