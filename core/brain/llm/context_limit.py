"""Context Limit Manager.
Prevents the 'Titan' from choking on too much text.
"""
import logging
from typing import Any

logger = logging.getLogger("Core.Context")

class ContextManager:
    def __init__(self, max_tokens: int = 8000):
        # Rough estimate: 1 token ~= 4 characters
        self.char_limit = max_tokens * 3.5 
        
    def prune(self, history: str, system_prompt: str) -> str:
        """Trims the history to fit within the model's context window,
        ALWAYS preserving the System Prompt (Identity).
        """
        full_hist = str(history)
        total_len = len(full_hist) + len(system_prompt)
        overflow_marker = "[...Earlier conversation forgotten...]\n"
        
        if total_len < self.char_limit:
            return history

        available_history_chars = int(self.char_limit - len(system_prompt) - len(overflow_marker))
        if available_history_chars <= 0:
            logger.warning(
                "Context Overflow detected. No history budget remained after preserving the system prompt."
            )
            return overflow_marker

        preserved_suffix = full_hist[-available_history_chars:]
        first_newline = preserved_suffix.find("\n")
        if 0 <= first_newline < len(preserved_suffix) - 1:
            preserved_suffix = preserved_suffix[first_newline + 1 :]

        if not preserved_suffix.strip():
            preserved_suffix = full_hist[-available_history_chars:].strip()

        pruned_chars = max(0, len(full_hist) - len(preserved_suffix))
        logger.warning("Context Overflow detected. Pruned %s chars from short-term memory.", pruned_chars)
        return f"{overflow_marker}{preserved_suffix}"

# Usage helper
def get_context_manager(max_tokens: int = 8000) -> ContextManager:
    return ContextManager(max_tokens)

async def compact_working_memory(history: list[dict[str, Any]], max_raw_turns: int = 15) -> list[dict[str, Any]]:
    # Zenith Cleanup: Preserves first two turns (genesis) + last N turns.
    if len(history) <= max_raw_turns:
        return history

    genesis = list(history[: min(len(history), 2)])
    recent_count = int(max_raw_turns - 2)
    start_point = max(0, len(history) - recent_count)
    recent = list(history[start_point:])

    logger.info("📦 Memory compacted from %d to %d turns.", len(history), len(genesis) + len(recent))
    return genesis + recent
