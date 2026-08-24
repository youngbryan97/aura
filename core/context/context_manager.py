"""Token-budget guard in front of the message-count pruner.

This class was constructed at boot, registered as a service, and never called.
:meth:`ContextWindowManager.compress_if_needed` is now reached from
``_prune_history_async``, where it does the one thing the live pruning path
could not do: measure *tokens*. Everything else there counts messages, so fifty
messages carrying a megabyte of tool output sit comfortably under the limit and
blow the window anyway.

``build_prompt`` and the ``ContextItem`` it took are gone rather than left
alongside. They were a second, uncalled prompt assembler for a job the live
turn already does elsewhere, and the danger of dead code on a class that is now
live is that the next reader takes it for the supported path.
"""
import functools
import logging
from typing import Any

from core.context.chat_compression import (
    ChatCompressionService,
    estimate_tokens_for_messages,
)
from core.runtime.errors import record_degradation

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
    # Cortex is resolved from the active cortex serving profile at call time;
    # this entry is the fallback for when no pointer is readable. Naming a
    # parameter count here made the table claim a model rather than a window.
    "Cortex":                    32_000,
    "Solver":                    32_000,  # deep lane — 32K context
    "Brainstem":                  8_000,  # 7B local fast lane — 8K context
    "Reflex":                     4_000,  # 1.5B emergency — minimal context
    "default":                   16_000,
}
DEFAULT_HEADROOM = 0.80


def resolved_context_limit(lane: str, *, served_tokens: int | None = None) -> int:
    """The window this lane serves, taking a measured answer over the table.

    The active cortex carries a qualified serving profile, and it is the only
    thing that knows what window the promoted checkpoint was qualified for. A
    constant here was right for exactly one checkpoint and silently wrong for
    the next.

    Layering forbids this module from reaching up for that profile, and forbids
    the registry from reaching down to install it, so the caller -- which is
    already holding the limits -- passes the number. The table stays as the
    answer for a process that has no cortex to ask about.
    """
    name = str(lane or "").strip() or "default"
    if (
        isinstance(served_tokens, int)
        and not isinstance(served_tokens, bool)
        and served_tokens > 0
    ):
        return served_tokens
    return MODEL_CONTEXT_LIMITS.get(name, MODEL_CONTEXT_LIMITS["default"])


# ── Tokenizer ───────────────────────────────────────────────────────────────

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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("context_manager", exc)
            logger.debug("tiktoken estimate failed; falling back to char estimate: %s", exc)
    return max(1, len(text) // 4)


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
        history: list[dict[str, str]],
        brain: Any = None,
    ) -> list[dict[str, str]]:
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
