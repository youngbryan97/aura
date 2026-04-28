"""Context Manager — Token-Budgeted Context Assembly

Assembles the prompt context by prioritizing:
  1. System prompt (always included)
  2. Relevant memories (via vector search)
  3. Working memory / recent conversation
  4. User query

All within a configurable token budget to avoid exceeding
the model's context window.
"""
from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Memory.ContextManager")

# Rough token-to-character ratio for estimation (1 token ≈ 4 chars)
CHARS_PER_TOKEN = 4


class ContextManager:
    """Assembles prompts within a token budget by prioritizing
    system prompts, relevant memories, and conversation history.

    Parameters
    ----------
    vector_memory : VectorMemory
        The vector store for semantic memory retrieval.
    token_budget : int
        Maximum tokens for the assembled context (default 3072).
    system_prompt : str
        The always-included system prompt.

    """

    def __init__(
        self,
        vector_memory: Any = None,
        token_budget: int = 8192,
        system_prompt: str = "",
    ):
        self.vector_memory = vector_memory
        self.token_budget = token_budget
        self.system_prompt = system_prompt

    async def assemble_context(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict]] = None,
        max_memories: int = 5,
    ) -> str:
        """Build a complete context string within the token budget.

        Priority order:
          1. System prompt (always)
          2. User query (always)
          3. Relevant memories (up to max_memories)
          4. Conversation history (most recent first, truncated to fit)

        Returns the assembled context string.
        """
        conversation_history = conversation_history or []

        # Start with fixed-cost items
        parts = []
        budget_chars = self.token_budget * CHARS_PER_TOKEN

        # 1. System prompt
        if self.system_prompt:
            parts.append(f"[SYSTEM]\n{self.system_prompt}\n")

        # 2. User query (reserved at end, but budget it now)
        query_block = f"\n[USER QUERY]\n{user_query}\n"

        # Calculate remaining budget
        fixed_chars = sum(len(p) for p in parts) + len(query_block)
        remaining = budget_chars - fixed_chars

        if remaining <= 0:
            logger.warning("Token budget exhausted by system prompt + query alone.")
            return "\n".join(parts) + query_block

        # 3. Relevant memories
        memory_block = ""
        if self.vector_memory and user_query:
            try:
                memories = self.vector_memory.search_similar(user_query, limit=max_memories)
                if memories:
                    mem_lines = ["[RELEVANT MEMORIES]"]
                    for i, mem in enumerate(memories, 1):
                        content = mem.get("content", str(mem))
                        mem_lines.append(f"  {i}. {content[:300]}")
                    candidate = "\n".join(mem_lines) + "\n"
                    if len(candidate) <= remaining * 0.6:  # Reserve 40% for history
                        memory_block = candidate
                        remaining -= len(memory_block)
            except Exception as exc:
                record_degradation('context_manager', exc)
                logger.debug("Memory retrieval failed: %s", exc)

        if memory_block:
            parts.append(memory_block)

        # 4. Conversation history (most recent first, fit to remaining budget)
        if conversation_history and remaining > 200:
            history_lines = ["[RECENT CONVERSATION]"]
            chars_used = len(history_lines[0])

            # Iterate from most recent backward
            for msg in reversed(conversation_history[-10:]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                line = f"  [{role}]: {content[:200]}"
                if chars_used + len(line) + 1 > remaining:
                    break
                history_lines.append(line)
                chars_used += len(line) + 1

            if len(history_lines) > 1:
                # Reverse back to chronological order
                history_lines[1:] = list(reversed(history_lines[1:]))
                parts.append("\n".join(history_lines) + "\n")

        # 5. Append user query at the end
        parts.append(query_block)

        assembled = "\n".join(parts)
        est_tokens = len(assembled) // CHARS_PER_TOKEN
        logger.debug(
            "Context assembled: ~%d tokens (%d chars), budget=%d",
            est_tokens, len(assembled), self.token_budget,
        )
        return assembled

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text) // CHARS_PER_TOKEN