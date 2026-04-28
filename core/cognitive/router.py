"""Intent Router
Deterministic classification gateway for all user inputs.
Replaces the open-ended "Cognitive Engine" ReAct loop.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

import logging
import re
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union, Dict, List

from core.config import config
from core.health.degraded_events import record_degraded_event
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
        self.llm: Optional[LLMClient] = optional_service("llm_router", default=None)
        
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
    def _check_heuristics(self, lower_input: str) -> Optional[Intent]:
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

    async def classify(self, user_input: str, context: Optional[dict[str, Any]] = None) -> Intent:
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
        except Exception as exc:
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
