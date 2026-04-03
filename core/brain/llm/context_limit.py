"""Context Limit Manager.
Prevents the 'Titan' from choking on too much text.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger("Core.Context")

class ContextManager:
    def __init__(self, max_tokens: int = 8000):
        # Rough estimate: 1 token ~= 4 characters
        self.char_limit = max_tokens * 3.5 
        
    def prune(self, history: str, system_prompt: str) -> str:
        """Trims the history to fit within the model's context window,
        ALWAYS preserving the System Prompt (Identity).
        """
        total_len = len(history) + len(system_prompt)
        
        if total_len < self.char_limit:
            return history
            
        # We need to cut. Calculate how much to remove.
        excess = total_len - self.char_limit
        # Add a safety buffer (500 chars)
        cut_amount = int(excess + 500)
        
        # We cut from the BEGINNING of history (oldest stuff),
        # but we must never touch the System Prompt.
        
        # Simple heuristic: Find the first newline after the cut point
        # to avoid cutting words in half.
        full_hist = str(history)
        pruned_history = "".join([full_hist[i] for i in range(int(cut_amount), len(full_hist))])
        first_newline = pruned_history.find('\n')
        if first_newline != -1:
            pruned_history = "".join([pruned_history[i] for i in range(int(first_newline + 1), len(pruned_history))])
            
        logger.warning("Context Overflow detected. Pruned %s chars from short-term memory.", cut_amount)
        return f"[...Earlier conversation forgotten...]\n{pruned_history}"

# Usage helper
def get_context_manager(max_tokens: int = 8000) -> ContextManager:
    return ContextManager(max_tokens)

async def compact_working_memory(history: list[dict[str, Any]], max_raw_turns: int = 15) -> list[dict[str, Any]]:
    # Zenith Cleanup: Preserves first two turns (genesis) + last N turns.
    if len(history) <= max_raw_turns:
        return history
    
    # Preserve genesis (first 2 turns: system + initial user/aura)
    genesis = []
    for i in range(min(len(history), 2)):
        genesis.append(history[i])
        
    # Preserve the most recent turns
    recent_count = int(max_raw_turns - 2)
    start_point = max(0, len(history) - recent_count)
    recent = []
    for i in range(start_point, len(history)):
        recent.append(history[i])
    
    logger.info("📦 Memory compacted from %d to %d turns.", len(history), len(genesis) + len(recent))
    return genesis + recent