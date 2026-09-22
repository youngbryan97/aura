"""Chat Turn Logger - Automatic conversation memory capture

This module ensures every chat turn (user message + Aura response) is
automatically logged to persistent episodic memory, preventing conversation loss.

Features:
1. Captures user input immediately upon receipt
2. Captures Aura's response after generation
3. Integrates with episodic memory for relational significance detection
4. Marks conversation turns with entity mentions and emotional context
5. Prevents bot-like hollow conversations from being preserved (quality filter)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Memory.ChatTurnLogger")

_CHAT_TURN_ERROR_MARKERS = (
    "i'm still with you",
    "i'm still here",
    "i'm having trouble",
    "please try again",
    "failed to process",
    "error:",
)

_CHAT_TURN_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)


def local_chat_turn_learning_rejection_reason(
    user_message: str,
    aura_response: str,
) -> str:
    """Return a stable local rejection reason, or ``""`` when eligible.

    This intentionally excludes the richer response-reliability gate. The
    durable outbox runs that gate separately so an infrastructure error can be
    retried instead of being mislabeled as permanently inadmissible content.
    """

    if len(str(user_message or "").strip()) < 5:
        return "user_message_too_short_for_learned_memory"
    response = str(aura_response or "").strip()
    if len(response) < 10:
        return "aura_response_too_short_for_learned_memory"
    response_lower = response.casefold()
    if len(response) < 50 and any(
        marker in response_lower for marker in _CHAT_TURN_ERROR_MARKERS
    ):
        return "short_error_response"
    return ""


class ChatTurnLogger:
    """Singleton for automatic chat turn logging to episodic memory."""
    
    _instance: ChatTurnLogger | None = None
    _lock = asyncio.Lock()
    
    def __init__(self) -> None:
        self._episodic_memory = None
        self._memory_facade = None
    
    @classmethod
    async def get_instance(cls) -> ChatTurnLogger:
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ChatTurnLogger()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self) -> None:
        """Initialize memory system references."""
        self._refresh_services()

    def _refresh_services(self) -> None:
        """Resolve memory services that may have completed boot after this singleton."""
        try:
            from core.container import ServiceContainer

            if self._episodic_memory is None:
                self._episodic_memory = ServiceContainer.get("episodic_memory", default=None)
            if self._memory_facade is None:
                self._memory_facade = ServiceContainer.get("memory_facade", default=None)

            if self._episodic_memory:
                logger.debug("✓ ChatTurnLogger linked to episodic memory")
            if self._memory_facade:
                logger.debug("✓ ChatTurnLogger linked to memory facade")
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("chat_turn_logger", e)
            logger.warning("ChatTurnLogger initialization incomplete: %s", e)
    
    def _is_meaningful_turn(self, user_message: str, aura_response: str) -> bool:
        """Admit only semantically valid turns to learned memory."""
        if local_chat_turn_learning_rejection_reason(user_message, aura_response):
            return False

        try:
            from core.conversation.response_reliability import (
                assess_conversation_learning_admission,
            )

            assessment = assess_conversation_learning_admission(
                user_message,
                aura_response,
            )
            if not assessment.ok:
                logger.warning(
                    "Rejected chat turn from learned memory (%s).",
                    ",".join(assessment.reasons) or "unknown",
                )
                return False
        except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "chat_turn_logger.learning_admission",
                exc,
                action="failed learned-memory admission closed while preserving transcript storage",
            )
            return False

        return True

    def _schedule_profile_learning(
        self,
        *,
        user_id: str,
        user_message: str,
        aura_response: str,
        session_id: str,
    ) -> bool:
        """Schedule exact-agent learning independently of episodic availability."""
        normalized_user_id = " ".join(str(user_id or "").strip().split())[:160]
        if not normalized_user_id:
            return False
        try:
            from core.memory.profile_manager import learn_from_turn_auto

            async def _learn_profiles() -> None:
                try:
                    user_facts, self_facts = await learn_from_turn_auto(
                        user_id=normalized_user_id,
                        user_message=user_message,
                        aura_response=aura_response,
                        session_id=session_id,
                    )
                    if user_facts > 0 or self_facts > 0:
                        logger.debug(
                            "Profile learning: %d user facts, %d self facts",
                            user_facts,
                            self_facts,
                        )
                except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat_turn_logger.profile_learning", exc)
                    logger.debug("Profile learning skipped: %s", exc)

            get_task_tracker().create_task(
                _learn_profiles(),
                name=f"profile_learning_{session_id}",
            )
            return True
        except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
            record_degradation("chat_turn_logger.profile_learning", exc)
            logger.debug("Profile learning unavailable: %s", exc)
            return False

    async def _observe_interpersonal(
        self,
        *,
        user_id: str,
        user_message: str,
        aura_response: str,
        episode_id: str,
        superseded_episode_ids: tuple[str, ...] = (),
    ) -> bool:
        """Record what this turn showed about the person, against its episode."""

        normalized_user_id = " ".join(str(user_id or "").strip().split())[:160]
        if not normalized_user_id or not episode_id:
            return False
        try:
            from core.memory.interpersonal_store import get_interpersonal_store

            written = await get_interpersonal_store().observe_turn(
                normalized_user_id,
                episode_id=str(episode_id),
                user_text=user_message,
                assistant_text=aura_response,
                superseded_episode_ids=superseded_episode_ids,
            )
            if written:
                logger.debug("Interpersonal: recorded %d observation(s)", len(written))
            return True
        except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
            record_degradation("chat_turn_logger.interpersonal", exc)
            logger.debug("Interpersonal observation skipped: %s", exc)
            return False

    def _schedule_interpersonal_observation(
        self,
        *,
        user_id: str,
        user_message: str,
        aura_response: str,
        episode_id: str,
        superseded_episode_ids: tuple[str, ...] = (),
    ) -> bool:
        """Schedule a non-outbox observation after its episode exists.

        Scheduled after the episode exists rather than alongside profile
        learning, and that ordering is the point: the interpersonal store
        refuses a claim that has no episode behind it, so that `audit()` can
        hand a human the evidence for anything she believes. Passing a
        synthesised id to make the call fit earlier would satisfy the check and
        void the guarantee.

        Consent, admissibility and durability are all the store's business; this
        only supplies the turn.
        """
        if not str(user_id or "").strip() or not episode_id:
            return False
        try:
            get_task_tracker().create_task(
                self._observe_interpersonal(
                    user_id=user_id,
                    user_message=user_message,
                    aura_response=aura_response,
                    episode_id=episode_id,
                    superseded_episode_ids=superseded_episode_ids,
                ),
                name=f"interpersonal_observation_{str(episode_id)[:64]}",
            )
            return True
        except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
            record_degradation("chat_turn_logger.interpersonal", exc)
            logger.debug("Interpersonal observation unavailable: %s", exc)
            return False


    async def log_chat_turn(
        self,
        user_message: str,
        aura_response: str,
        session_id: str = "default",
        emotional_valence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Log a complete chat turn to episodic memory.
        
        Args:
            user_message: What the user said
            aura_response: Aura's response
            session_id: Current session identifier
            emotional_valence: Emotional tone (-1.0 to +1.0)
            metadata: Additional context (entity mentions, etc.)
            
        Returns:
            True if successfully logged, False otherwise
        """
        # Quality filter
        if not self._is_meaningful_turn(user_message, aura_response):
            logger.debug("Skipping hollow turn (too short or error response)")
            return False

        episode_metadata = dict(metadata or {})
        episode_metadata["session_id"] = session_id
        episode_metadata["turn_type"] = "conversation"
        episode_metadata["conversation_lane"] = True
        episode_metadata["learning_admission"] = "verified"
        try:
            from core.conversation.response_reliability import is_self_condition_turn

            episode_metadata["self_condition_grounded"] = bool(
                is_self_condition_turn(user_message)
            )
        except _CHAT_TURN_RECOVERABLE_ERRORS as exc:
            logger.debug(
                "%s unavailable (%s: %s); the turn is recorded as not self-condition grounded",
                "is_self_condition_turn",
                type(exc).__name__,
                exc,
            )
            episode_metadata["self_condition_grounded"] = False
        profile_user_id = str(episode_metadata.get("user_id") or "").strip()[:160]
        self._schedule_profile_learning(
            user_id=profile_user_id,
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
        )

        self._refresh_services()
        if not self._episodic_memory:
            return False
        
        try:
            # Log as a conversation episode
            # Format: context = user asked | action = aura generated | outcome = response
            context = f"User asked: {user_message[:200]}"
            action = f"Generated response in session {session_id}"
            outcome = aura_response[:300]
            
            # Record with metadata
            # Let episodic memory detect relational significance
            episode_id = self._episodic_memory.record_episode(
                context=context,
                action=action,
                outcome=outcome,
                success=True,  # If response was generated, it's a successful turn
                emotional_valence=emotional_valence,
                importance=0.6,  # Default importance (boosted by relational detection)
                source="chat_turn_logger",
                metadata=episode_metadata,
                idempotency_key=str(
                    episode_metadata.get("memory_log_operation_id") or ""
                ),
            )
            
            if episode_id:
                logger.debug(f"✓ Chat turn logged to episodic memory (episode={episode_id})")
                superseded_episode_ids: tuple[str, ...] = ()
                source_episode_ids = getattr(
                    self._episodic_memory,
                    "superseded_source_episode_ids",
                    None,
                )
                if callable(source_episode_ids):
                    superseded_episode_ids = tuple(
                        source_episode_ids(
                            source="chat_turn_logger",
                            session_id=session_id,
                            exchange_id=str(
                                episode_metadata.get("conversation_exchange_id") or ""
                            ),
                        )
                    )
                interpersonal_kwargs = {
                    "user_id": profile_user_id,
                    "user_message": user_message,
                    "aura_response": aura_response,
                    "episode_id": str(episode_id),
                    "superseded_episode_ids": superseded_episode_ids,
                }
                if episode_metadata.get("memory_log_operation_id"):
                    await self._observe_interpersonal(**interpersonal_kwargs)
                else:
                    self._schedule_interpersonal_observation(**interpersonal_kwargs)
                return True
            else:
                logger.debug("Chat turn logging returned empty episode_id (governance blocked or deferral)")
                return False
                
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("chat_turn_logger", e)
            logger.warning("Failed to log chat turn: %s", e)
            return False
    
    async def log_user_message(
        self,
        user_message: str,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Log just a user message (early capture, before response)."""
        if len(user_message.strip()) < 5:
            return False

        self._refresh_services()
        if not self._memory_facade:
            return False
        
        try:
            metadata = dict(metadata or {})
            metadata["session_id"] = session_id
            metadata["message_type"] = "user_input"
            metadata["conversation_lane"] = True
            
            # Store to vector memory for semantic search
            result = await self._memory_facade.add_memory(
                text=f"User said: {user_message}",
                metadata=metadata,
            )
            
            if result:
                logger.debug(f"✓ User message logged to memory (session={session_id})")
            return bool(result)
            
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("chat_turn_logger", e)
            logger.debug("Failed to log user message: %s", e)
            return False


async def log_chat_turn_auto(
    user_message: str,
    aura_response: str,
    session_id: str = "default",
    emotional_valence: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to log a chat turn."""
    try:
        logger_instance = await ChatTurnLogger.get_instance()
        return await logger_instance.log_chat_turn(
            user_message=user_message,
            aura_response=aura_response,
            session_id=session_id,
            emotional_valence=emotional_valence,
            metadata=metadata,
        )
    except _CHAT_TURN_RECOVERABLE_ERRORS as e:
        record_degradation("chat_turn_logger", e)
        logger.warning("Auto-logging failed: %s", e)
        return False
