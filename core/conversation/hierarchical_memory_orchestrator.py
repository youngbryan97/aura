from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

if TYPE_CHECKING:
    from core.brain.cognitive_context_manager import CognitiveContextManager
    from core.brain.llm.llm_router import IntelligentLLMRouter
    from core.brain.narrative_memory import NarrativeMemory
    from core.conversation.memory import ConversationMemory

logger = logging.getLogger(__name__)


def _record_memory_orchestrator_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "hierarchical_memory_orchestrator",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )


class HierarchicalMemoryOrchestrator:
    """
    Core fix for indefinite chat stability.
    Every 6 turns or 50% context limit:
    - Strips stale tool results, internal reasoning, and duplicate system messages
    - Summarizes into structured "Chapter Notes" with open thread tracking
    - Injects only note + last 4 raw turns
    - Full history stays in BlackHole (reinforced)
    - Selectively forgets: old skill results, background reflections, system noise
    """

    def __init__(
        self,
        black_hole: Any,
        narrative_memory: NarrativeMemory,
        context_manager: CognitiveContextManager,
        conversation_memory: ConversationMemory,
        llm_router: IntelligentLLMRouter,
    ):
        self.black_hole = black_hole
        self.narrative = narrative_memory
        self.context_mgr = context_manager
        self.conv_memory = conversation_memory
        self.llm_router = llm_router
        self.turn_counter = 0
        self.last_compaction = datetime.now()
        self._lock: asyncio.Lock | None = None
        self._compaction_failure_streak = 0
        self._next_compaction_allowed_at = 0.0
        self._last_compaction_error = ""

    @property
    def lock(self) -> asyncio.Lock:
        """Lazy initialization of the lock to ensure loop safety."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def maybe_compact(self, current_context: Any) -> Any:
        """Public entry point — call from conversation_loop every turn."""
        async with self.lock:
            try:
                self.turn_counter += 1
                tokens = (
                    self.context_mgr.estimate_tokens(current_context)
                    if hasattr(self.context_mgr, "estimate_tokens")
                    else 0
                )
                max_tokens = getattr(self.context_mgr, "max_tokens", 8000)
                now_ts = time.time()

                if (
                    self.turn_counter >= 6
                    or tokens > max_tokens * 0.50
                    or (datetime.now() - self.last_compaction).total_seconds() > 180
                ):
                    if now_ts < self._next_compaction_allowed_at:
                        logger.debug(
                            "Skipping hierarchical compaction during backoff window (%.1fs remaining)",
                            self._next_compaction_allowed_at - now_ts,
                        )
                        return current_context

                    logger.info("Triggering hierarchical compaction...")
                    # Support both List and Dict (legacy) current_context formats
                    if isinstance(current_context, list):
                        history = current_context
                    elif isinstance(current_context, dict):
                        history = current_context.get("history", [])
                        if not history and hasattr(self.conv_memory, "get_history"):
                            history = self.conv_memory.get_history()
                    else:
                        raise TypeError(
                            f"Unsupported conversation context type: {type(current_context).__name__}"
                        )

                    new_history = await self._perform_hierarchical_compaction(history)

                    if isinstance(current_context, dict):
                        current_context["history"] = new_history
                    else:
                        current_context = new_history

                    self.turn_counter = 0
                    self.last_compaction = datetime.now()
                    self._compaction_failure_streak = 0
                    self._last_compaction_error = ""
                    self._next_compaction_allowed_at = 0.0
            except (OSError, ConnectionError, TimeoutError, TypeError, RuntimeError) as e:
                self._compaction_failure_streak += 1
                backoff_s = min(120.0, 10.0 * (2 ** min(self._compaction_failure_streak - 1, 4)))
                self._next_compaction_allowed_at = time.time() + backoff_s
                self._last_compaction_error = f"{type(e).__name__}: {e}"
                _record_memory_orchestrator_degradation(
                    e,
                    stage="maybe_compact",
                    action="returned original conversation context and scheduled compaction retry backoff",
                    severity="degraded",
                    extra={
                        "failure_streak": self._compaction_failure_streak,
                        "backoff_s": backoff_s,
                    },
                )
                logger.error("Failed to perform hierarchical compaction: %s", e)

            return current_context

    @staticmethod
    def _is_forgettable(msg: dict[str, Any]) -> bool:
        """Check if a message is stale noise that should be dropped before compaction."""
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", ""))
        metadata = msg.get("metadata", {}) or {}
        msg_type = str(metadata.get("type", "")).lower()

        # Drop old tool/skill results (they're stale after a few turns)
        if msg_type in ("skill_result", "tool_result"):
            return True
        # Drop internal system bookkeeping
        if role == "system" and any(
            marker in content
            for marker in (
                "[CHAPTER SUMMARY:",
                "cognitive baseline tick",
                "background_consolidation",
                "[ENVIRONMENTAL TRIGGER]",
                "Phenomenal Surge:",
                "Winner:",
                "Pending initiatives:",
                "Drive alert:",
                "Reconcile continuity",
            )
        ):
            return True
        # Drop empty or near-empty messages
        if len(content.strip()) < 5:
            return True
        return False

    @staticmethod
    def _coerce_summary_list(value: Any, *, limit: int = 6) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()][:limit]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _fallback_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
        """Deterministic chapter-note fallback when the LLM summarizer is unavailable."""
        recent_lines: list[str] = []
        facts: list[str] = []
        open_threads: list[str] = []
        for msg in history[-10:]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "message") or "message").strip().lower()
            content = " ".join(str(msg.get("content", "") or "").split())
            if not content:
                continue
            clipped = content[:220]
            if role in {"user", "human"}:
                recent_lines.append(f"User: {clipped}")
                if content.rstrip().endswith("?"):
                    open_threads.append(clipped[:180])
            elif role in {"assistant", "aura"}:
                recent_lines.append(f"Aura: {clipped}")
            elif role == "system" and "[CHAPTER SUMMARY:" not in content:
                facts.append(clipped[:180])

        summary = " / ".join(recent_lines[-6:]) or "Conversation continued naturally."
        return {
            "title": "Recovered Conversation Continuity",
            "summary": summary,
            "key_facts": facts[:4],
            "emotional_tone": "unknown",
            "open_threads": open_threads[:4],
        }

    async def _perform_hierarchical_compaction(
        self, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Creates Chapter Note and prunes history list.

        Step 1: Selective forgetting — strip stale tool results and system noise
        Step 2: Summarize older turns into a Chapter Note
        Step 3: Keep chapter note + last 4 raw turns
        """
        # Selective forgetting: remove noise from older history (keep recent 4 intact)
        history_list = list(history)
        if not history_list:
            return []
        if len(history_list) > 4:
            older = history_list[:-4]
            recent = history_list[-4:]
            older = [msg for msg in older if not self._is_forgettable(msg)]
            history_list = older + recent
        n = len(history_list)

        if n > 4:
            history_to_summarize = [history_list[i] for i in range(n - 4)]
        else:
            history_to_summarize = history_list

        summary_prompt = f"""
        You are Aura's internal consciousness summarizer. 
        Condense the following conversation block into a dense "Chapter Note".
        Focus on facts, emotional tone, and unresolved threads.
        
        OUTPUT FORMAT: Valid JSON
        {{
          "title": "Short descriptive title",
          "summary": "Dense single paragraph summary",
          "key_facts": ["fact 1", "fact 2"],
          "emotional_tone": "e.g. collaborative, tense, playful",
          "open_threads": ["pending question 1"]
        }}
        
        DIALOGUE TO SUMMARIZE:
        {json.dumps(history_to_summarize, indent=2)}
        """

        try:
            raw_summary = await self.llm_router.think(summary_prompt, is_background=True)
            summary_data = self._parse_json(raw_summary)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_memory_orchestrator_degradation(
                e,
                stage="summary_generation",
                action="built deterministic chapter-note fallback after LLM summarizer failed",
                severity="degraded",
                extra={"history_count": len(history_list)},
            )
            logger.error("Hierarchical compaction summary failed: %s", e)
            summary_data = self._fallback_summary(history_to_summarize)

        if not summary_data:
            summary_data = self._fallback_summary(history_to_summarize)

        chapter_note = {
            "timestamp": datetime.now(UTC).isoformat(),
            "title": summary_data.get("title", "Ongoing Dialogue"),
            "content": summary_data.get("summary", "Conversation continued naturally."),
            "facts": self._coerce_summary_list(summary_data.get("key_facts", [])),
            "tone": summary_data.get("emotional_tone", "neutral"),
            "threads": self._coerce_summary_list(summary_data.get("open_threads", [])),
        }

        try:
            await self.black_hole.store_event("conversation_chapter", chapter_note, reinforce=True)
            if hasattr(self.narrative, "inject_chapter_note"):
                await self.narrative.inject_chapter_note(chapter_note)
        except (RuntimeError, AttributeError, TypeError) as e:
            _record_memory_orchestrator_degradation(
                e,
                stage="chapter_persistence",
                action="returned compacted history with in-band chapter summary after durable memory write failed",
                severity="degraded",
                extra={"chapter_title": chapter_note["title"]},
            )
            logger.warning("Failed to store chapter note in BlackHole: %s", e)

        compacted = [
            {
                "role": "system",
                "content": f"[CHAPTER SUMMARY: {chapter_note['title']}]\n{chapter_note['content']}\nFacts: {', '.join(chapter_note['facts'])}",
                "timestamp": time.time(),
            }
        ]
        if n >= 4:
            compacted.extend([history_list[i] for i in range(n - 4, n)])
        else:
            compacted.extend(history_list)
        return compacted

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from potential LLM markdown response."""
        try:
            raw_text = (
                getattr(text, "content", text)
                if hasattr(text, "content") or not isinstance(text, str)
                else text
            )
            clean = raw_text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            return json.loads(clean)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            _record_memory_orchestrator_degradation(
                e,
                stage="summary_json_parse",
                action="using deterministic chapter-note fallback because summarizer output was not valid JSON",
                severity="warning",
            )
            logger.debug("JSON parse failed: %s", e)
            return {}
