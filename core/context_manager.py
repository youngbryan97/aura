"""
Context window budget manager.
Tracks token usage and prunes the lowest-value content to stay within limits.
v2.0: Integrated ChatCompressionService for intelligent history management.
"""
from core.runtime.errors import record_degradation
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.context.chat_compression import (
    ChatCompressionService,
    CompressionStatus,
    estimate_tokens_for_messages,
)

logger = logging.getLogger("Aura.ContextManager")

# Model context limits (conservative — leave 20% headroom)
MODEL_CONTEXT_LIMITS = {
    "grok-1":                     8_192,
    "grok-2":                   131_072,
    "grok-3":                   100_000,
    "grok-3-mini":               32_000,
    "claude-opus-4":            180_000,
    "claude-sonnet-4":          180_000,
    "claude-haiku-4":            32_000,
    "gpt-4o":                    96_000,
    "gpt-4o-mini":               96_000,
    "Cortex":                    32_000,  # 32B local lane — 32K context
    "Solver":                    32_000,  # 72B deep lane — 32K context
    "Brainstem":                  8_000,  # 7B local fast lane — 8K context
    "Reflex":                     4_000,  # 1.5B emergency — minimal context
    "default":                   16_000,
}
DEFAULT_HEADROOM = 0.80


# ── Tokenizer ───────────────────────────────────────────────────────────────

import functools

# Use tiktoken if available, fallback to char count
try:
    import tiktoken
    _T_ENCODING = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    _T_ENCODING = None

@functools.lru_cache(maxsize=1000)
def estimate_tokens(text: str) -> int:
    """Accurate token count via tiktoken (cached), fallback to ~4 chars/token."""
    if not text:
        return 0
    if HAS_TIKTOKEN and _T_ENCODING:
        try:
            return len(_T_ENCODING.encode(text, disallowed_special=()))
        except Exception:
            pass  # no-op: intentional
    return max(1, len(text) // 4)


@dataclass
class ContextItem:
    content: str
    role: str          # 'system' | 'user' | 'assistant' | 'tool'
    priority: int      # Higher = more important (never drop)
    tokens: int = 0

    def __post_init__(self):
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)


class ContextWindowManager:
    """
    Assembles a prompt that fits within the model's context limit.
    Drops low-priority items first; never drops system prompt or the current user message.
    v2.0: Integrated ChatCompressionService for automatic history compression.
    """

    def __init__(self, model_name: str = "default"):
        # Audit-40: Support generic model matching for stale version names
        limit = MODEL_CONTEXT_LIMITS.get("default", 16_000)
        m_lower = model_name.lower()
        
        # Sort keys by length DESC to match more specific names first (e.g. gpt-4o-mini before gpt-4o if subset)
        # Actually gpt-4o is a subset of gpt-4o-mini, so we need a cleverer check.
        # Let's just do a direct match first, then substring.
        if model_name in MODEL_CONTEXT_LIMITS:
            limit = MODEL_CONTEXT_LIMITS[model_name]
        else:
            for key, val in sorted(MODEL_CONTEXT_LIMITS.items(), key=lambda x: len(x[0]), reverse=True):
                if key != "default" and key.lower() in m_lower:
                    limit = val
                    break
                    
        self._limit = int(limit * DEFAULT_HEADROOM)  # Safe headroom
        self._model = model_name
        self._compression_service = ChatCompressionService()
        self._raw_limit = limit  # Unscaled limit for compression threshold

    async def compress_if_needed(
        self,
        history: List[Dict[str, str]],
        brain: Any = None,
    ) -> List[Dict[str, str]]:
        """Auto-compress history if it exceeds the threshold.

        Args:
            history: Message list (role/content dicts)
            brain: LocalBrain instance for LLM summarization

        Returns:
            Possibly compressed history.
        """
        current_tokens = estimate_tokens_for_messages(history)
        compressed, info = await self._compression_service.compress(
            history=history,
            model_token_limit=self._raw_limit,
            current_token_count=current_tokens,
            brain=brain,
        )
        if compressed is not None:
            logger.info(
                "Context compressed: %s (%d → %d tokens)",
                info.status.value, info.original_token_count, info.new_token_count
            )
            return compressed
        return history

    def build_prompt(
        self,
        system: str,
        history: List[ContextItem],
        current_message: str,
        memory_context: str = "",
        tool_context: str = "",
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Build a prompt list that fits within the token budget.

        Returns:
            (messages_list, total_tokens_used)
        """
        budget = self._limit
        messages: List[Dict[str, str]] = []

        # MUST include — these are never dropped
        system_tokens = estimate_tokens(system)
        current_tokens = estimate_tokens(current_message)
        reserved = system_tokens + current_tokens + 200  # 200 for response headroom

        if reserved > budget:
            logger.warning(
                "System prompt + current message (%d tokens) exceeds budget (%d). "
                "Truncating system prompt.",
                reserved, budget,
            )
            system = system[: (budget - current_tokens - 200) * 4]
            system_tokens = estimate_tokens(system)
            reserved = system_tokens + current_tokens + 200

        messages.append({"role": "system", "content": system})
        remaining = budget - reserved

        # Memory context (medium priority — truncate before dropping)
        if memory_context and remaining > 500:
            mem_tokens = estimate_tokens(memory_context)
            if mem_tokens > remaining // 3:
                # Truncate to 1/3 of remaining budget
                max_chars = (remaining // 3) * 4
                memory_context = memory_context[:max_chars] + "\n...[memory truncated]"
                mem_tokens = estimate_tokens(memory_context)
            messages.append({"role": "system", "content": f"[MEMORY CONTEXT]\n{memory_context}"})
            remaining -= mem_tokens

        # Tool results (medium priority)
        if tool_context and remaining > 500:
            tool_tokens = estimate_tokens(tool_context)
            if tool_tokens > remaining // 2:
                max_chars = (remaining // 2) * 4
                tool_context = tool_context[:max_chars] + "\n...[tool results truncated]"
                tool_tokens = estimate_tokens(tool_context)
            messages.append({"role": "system", "content": f"[TOOL CONTEXT]\n{tool_context}"})
            remaining -= tool_tokens

        # Conversation history — keep most recent, drop oldest first
        kept_history = []
        for item in reversed(history):
            if item.tokens <= remaining:
                kept_history.insert(0, item)
                remaining -= item.tokens
            else:
                logger.debug("Context pruned: dropping old %s message (%d tokens)", item.role, item.tokens)

        for item in kept_history:
            messages.append({"role": item.role, "content": item.content})

        # Current message — always last
        messages.append({"role": "user", "content": current_message})

        # NEW: Always inject latest unified transcript (Fix 3)
        try:
            from core.conversation_loop import get_transcript
            transcript = get_transcript()
            recent = transcript.get_context_string(n=12)  # last 12 turns across all channels
            if recent:
                # Insert after system prompt (index 1)
                messages.insert(1, {"role": "system", "content": f"[RECENT CONVERSATION]\n{recent}"})
        except Exception as e:
            record_degradation('context_manager', e)
            logger.debug("Failed to inject UnifiedTranscript: %s", e)

        total_used = budget - remaining
        logger.debug(
            "Context assembled: %d messages, ~%d tokens (budget: %d, model: %s)",
            len(messages), total_used, self._limit, self._model,
        )

        return messages, total_used
