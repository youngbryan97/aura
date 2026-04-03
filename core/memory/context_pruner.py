import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("Memory.Pruner")

class ContextPruner:
    """Manages the context window by summarizing old history when space gets tight.
    v5.5: Hardened to prevent 'amnesia' by increasing limits.
    """
    
    def __init__(self, token_limit: int = 250000):
        self.token_limit = token_limit
        self.prune_threshold = 0.9 # Prune when 90% full
        
    def needs_pruning(self, current_tokens: int) -> bool:
        return current_tokens > (self.token_limit * self.prune_threshold)

    async def prune_history(self, history: List[Dict[str, Any]], cognitive_engine) -> List[Dict[str, Any]]:
        """Compress conversation history.
        Strategy: Summarize the oldest segments, keep the recent context raw.
        """
        from core.brain.cognitive_engine import ThinkingMode

        # Only prune if history is substantial
        if len(history) < 60:
            return history
            
        logger.info("🕸️ Pruning Context (Size: %d messages)...", len(history))
        
        # Keep the last 30 messages raw to maintain personality and flow
        keep_recent = 30
        older_half = history[:-keep_recent]
        newer_half = history[-keep_recent:]
        
        # Summarize older half
        text_to_summarize = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in older_half])
        
        prompt = f"""
SYSTEM: CONTEXT CONSOLIDATION (EPISODIC COMPRESSION)
Distill the following conversation history into high-density 'Memory Spores'.
Retain: User preferences, completed tasks, ongoing objectives, and deep emotional context.
Discard: Greetings, repetitive filler, and transient chitchat.
MAINTAIN TONE: Keep the summary in a style consistent with Aura's personality.

CONVERSATION TO COMPRESS:
{text_to_summarize}

DENSE MEMORY SPORES:
"""
        try:
            thought = await cognitive_engine.think(
                objective=prompt, 
                mode=ThinkingMode.FAST,
                origin="context_pruner",
                is_background=True,
            )
            summary_text = thought.content
            
            summary_message = {
                "role": "system", 
                "content": f"[CONSOLIDATED MEMORY]: {summary_text}"
            }
            
            # Reassemble
            new_history = [summary_message] + newer_half
            logger.info("✅ Context Pruned. Reduced to %d messages.", len(new_history))
            return new_history
            
        except Exception as e:
            logger.error("Pruning failed: %s", e)
            # Fallback: Just drop some messages if LLM pruning fails
            return history[-min(len(history), 80):]

context_pruner = ContextPruner()
