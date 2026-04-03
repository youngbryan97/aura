"""Intent Router
Deterministic classification gateway for all user inputs.
Replaces the open-ended "Cognitive Engine" ReAct loop.
"""
from __future__ import annotations
import logging
import re
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union, Dict, List

from core.config import config

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
        from core.container import ServiceContainer
        # H-28 FIX: Explicit type hint for the protocol
        self.llm: Optional[LLMClient] = ServiceContainer.get("llm_router", default=None)
        
        if not self.llm:
            logger.warning("IntentRouter: No valid LLM generator found in container.")

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
            logger.info("⚡ IntentRouter: Bypassing classification due to intent_hint")
            return Intent.SKILL

        lower_input = user_input.lower().strip()
        
        # 1. Check Heuristics (Cached)
        heuristic_result = self._check_heuristics(lower_input)
        if heuristic_result:
            return heuristic_result

        # 2. LLM Classification (Lightning Fast)
        if not self.llm:
            logger.warning("IntentRouter: LLM missing, defaulting to CHAT.")
            return Intent.CHAT

        system_prompt = (
            "You are an intent classifier. Respond ONLY with one of the following words:\n"
            "CHAT - General conversation, greetings, empathy, or answering basic questions.\n"
            "SKILL - The user is asking you to perform an action, use a tool, search a file, search the web, or read/write data.\n"
            "SYSTEM - The user is talking about restarting, shutting down, or managing your core systems.\n\n"
            "Do not explain. Just output the single word."
        )

        try:
            # We enforce a tiny max_tokens to ensure it literally only outputs one word
            response = await self.llm.generate(
                prompt=user_input,
                system_prompt=system_prompt,
                max_tokens=10,
                temperature=0.0 # Maximum determinism
            )
            
            result = response.strip().upper()
            
            for intent in Intent:
                if intent.value in result:
                    logger.debug("Intent classified as %s", intent.value)
                    return intent
                    
            logger.warning("Fuzzy intent match: '%s'. Defaulting to CHAT.", result)
            return Intent.CHAT
            
        except Exception as e:
            logger.error("Intent classification failed: %s", e)
            return Intent.CHAT # Safe fallback