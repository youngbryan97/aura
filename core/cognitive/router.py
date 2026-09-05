"""Intent Router
Deterministic classification gateway for all user inputs.
Replaces the open-ended "Cognitive Engine" ReAct loop.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from core.config import config
from core.health.degraded_events import record_degraded_event
from core.runtime.errors import record_degradation
from core.runtime.governance_policy import allow_intent_hint_bypass
from core.runtime.service_access import optional_service
from core.runtime.turn_analysis import analyze_turn

if TYPE_CHECKING:
    from core.brain.types import LLMClient

logger = logging.getLogger("Aura.IntentRouter")


class Intent(Enum):
    CHAT = "CHAT"
    SKILL = "SKILL"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"


class IntentRouter:
    """Classifies user input to determine the strict State Machine path."""

    def __init__(self) -> None:
        # H-28 FIX: Explicit type hint for the protocol
        self.llm: LLMClient | None = optional_service("llm_router", default=None)
        
        if not self.llm:
            logger.warning("IntentRouter: No valid LLM generator found in container.")
            record_degraded_event(
                "intent_router",
                "llm_router_missing",
                detail="classification_falling_back_to_deterministic_analysis",
                severity="info",
                classification="non_critical_fallback",
            )

    @lru_cache(maxsize=100)
    def _check_heuristics(self, lower_input: str) -> Intent | None:
        """Fast Regex/Heuristic bypasses (Zero Token Cost)."""
        
        # SYSTEM bypass
        system_cmds = ["reboot", "restart", "shutdown", "sleep", "wake up"]
        if any(cmd in lower_input for cmd in system_cmds):
            return Intent.SYSTEM
            
        # SKILL bypass (High Priority keywords)
        # Use regex word boundaries to avoid false positives (e.g., "run" in "runner")
        for kw in config.cognitive.skill_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower_input):
                logger.debug("Heuristic trigger: Forced SKILL intent for keyword: %s", kw)
                return Intent.SKILL

        if lower_input in ["hello", "hi", "hey", "sup"]:
            return Intent.CHAT
            
        return None

    async def classify(self, user_input: str, context: dict[str, Any] | None = None) -> Intent:
        """Determines the intent of the user input deterministically."""
        
        # Phase 37 v2: Sovereign Scanner & Agency Bypass
        if context and context.get("intent_hint"):
            hint_origin = context.get("origin") or context.get("request_origin") or context.get("source")
            if allow_intent_hint_bypass(context, hint_origin):
                logger.info("⚡ IntentRouter: Using sanctioned constitutional intent_hint")
                return Intent.SKILL
            logger.info("🧭 IntentRouter: Ignoring unsanctioned intent_hint for governed classification")

        lower_input = user_input.lower().strip()
        matched_skills = False
        
        # 1. Check Heuristics (Cached)
        heuristic_result = self._check_heuristics(lower_input)
        if heuristic_result:
            return heuristic_result

        try:
            cap = optional_service("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent"):
                matched_skills = bool(cap.detect_intent(user_input))
        except (RuntimeError, AttributeError, TypeError) as exc:
            record_degradation('router', exc)
            logger.debug("IntentRouter: capability pre-check failed: %s", exc)

        analysis = analyze_turn(user_input, matched_skills=matched_skills)
        mapping = {
            "SYSTEM": Intent.SYSTEM,
            "SKILL": Intent.SKILL,
            "TASK": Intent.SKILL,
            "CHAT": Intent.CHAT,
        }
        routed_intent = mapping.get(analysis.intent_type, Intent.CHAT)
        logger.debug(
            "IntentRouter: deterministic route=%s semantic=%s live_voice=%s",
            routed_intent.value,
            analysis.semantic_mode,
            analysis.requires_live_aura_voice,
        )
        return routed_intent

    async def route_execution(
        self,
        skill_name: str,
        params: dict[str, Any] | None = None,
        engine: Any = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a concrete skill through the governed capability engine.

        The UI skill button path is already past intent classification: the
        operator selected a named skill and supplied typed params. Keeping this
        adapter here lets `/api/skill/execute` use the same registered router
        without inventing a parallel executor or bypassing CapabilityEngine.
        """
        capability_engine = engine or optional_service("capability_engine", default=None)
        if capability_engine is None or not hasattr(capability_engine, "execute"):
            return {
                "ok": False,
                "error": "Capability engine unavailable for routed skill execution.",
                "skill": str(skill_name or ""),
            }
        execution_context = dict(context or {})
        execution_context["origin"] = str(execution_context.get("origin") or "api")
        execution_context["route"] = str(execution_context.get("route") or "intent_router.route_execution")
        execution_context["foreground_request"] = True
        execution_context["user_explicitly_authorized"] = True
        execution_context["user_requested_action"] = True
        # The flag alone is a caller-supplied boolean, and BeingRuntime
        # correctly ignores it without a capability token bound to
        # tool_execution/foreground_desktop_action. Nothing was issuing that
        # token, so the assertion was dead and its refusal logged on every
        # desktop turn. The router does not self-grant: it asks the AUTHORITY
        # GATEWAY, which is the only issuer, to attest what it has already
        # established — that this action was explicitly requested this turn.
        if not execution_context.get("capability_token"):
            try:
                from core.executive.authority_gateway import get_authority_gateway

                token = get_authority_gateway().issue_desktop_authority_capability(
                    skill=str(skill_name or ""),
                    origin=str(execution_context.get("origin") or "api"),
                )
                if token:
                    execution_context["capability_token"] = token
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "intent_router",
                    exc,
                    severity="warning",
                    action=(
                        "routed the skill without a desktop authority token; the "
                        "foreground-desktop exception simply will not apply"
                    ),
                )

        return await capability_engine.execute(
            str(skill_name or "").strip(),
            dict(params or {}),
            context=execution_context,
        )
