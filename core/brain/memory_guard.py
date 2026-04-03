"""MemoryGuard — Tiered context pruning for Aura.
Ensures context fits within model-specific boundaries.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Aura.MemoryGuard")

class ContextPruner:
    def __init__(self):
        self.tiers = {
            "gemini": 1000000, # 1M context
            "mistral": 32768,
            "llama3": 8192,
            "reflex": 1024
        }

    def prune_context(self, history: List[Dict[str, str]], tier: str = "llama3") -> List[Dict[str, str]]:
        """Prunes historical context based on the current model tier, retaining a 'fading echo' via summarization."""
        history = list(history or [])
        limit = self.tiers.get(tier, 8192)
        
        # Simple heuristic: ~4 chars per token
        current_est = sum(len(m.get("content", "")) for m in history) // 4
        
        if current_est <= limit:
            return history
            
        logger.info("🛡️ MemoryGuard: Pruning context for tier %s (Est: %d tokens, Limit: %d)", tier, current_est, limit)
        
        if not history:
            return history
            
        system_prompt = history[0] if history[0].get("role") == "system" else None
        
        # Identify messages to prune vs keep
        pruned_msgs = []
        keep_msgs = []
        
        if tier == "reflex":
            keep_msgs = history[-2:]
            pruned_msgs = history[1:-2] if system_prompt else history[:-2]
        elif tier == "llama3":
            keep_msgs = history[-10:]
            pruned_msgs = history[1:-10] if system_prompt else history[:-10]
        else:
            # Dynamic pruning
            keep_msgs = history.copy()
            while sum(len(m.get("content", "")) for m in keep_msgs) // 4 > limit and len(keep_msgs) > 2:
                idx = 1 if system_prompt and keep_msgs[0] == system_prompt else 0
                pruned_msgs.append(keep_msgs.pop(idx))
        
        # Generate 'fading echo' summary of pruned content
        summary_text = self._summarize_history(pruned_msgs)
        
        final_history = []
        if system_prompt:
            final_history.append(system_prompt)
            
        if summary_text:
            final_history.append({
                "role": "system", 
                "content": f"[Historical Context Echo: {summary_text}]",
                "metadata": {"type": "memory_echo"}
            })
            
        final_history.extend(keep_msgs if not system_prompt or keep_msgs[0] != system_prompt else keep_msgs[1:])
            
        return final_history

    def _summarize_history(self, history: List[Dict[str, str]]) -> str:
        """
        FIX: Implements functional extractive summarization (heuristic-based).
        Retains ~5% as a 'fading echo' by picking key fragments from each pruned turn.
        """
        if not history:
            return ""
            
        echoes = []
        for msg in history:
            content = msg.get("content", "").strip()
            if not content: continue
            
            # Heuristic: Take the first significant sentence or first 60 chars
            lines = content.split('\n')
            first_line = lines[0] if lines else ""
            
            # Cleanly truncate
            if len(first_line) > 65:
                echo = first_line[:60].rsplit(' ', 1)[0] + "..."
            else:
                echo = first_line
                
            role_label = "U" if msg.get("role") == "user" else "A"
            echoes.append(f"{role_label}: {echo}")
            
        # Join with minimal separator
        full_echo = " | ".join(echoes)
        
        # Enforce hard limit on echo size
        if len(full_echo) > 500:
            return full_echo[-500:].split(' | ', 1)[-1] # Keep most recent 500 chars of echo
            
        return full_echo

    def get_summary_context(self, history: List[Dict[str, str]]) -> str:
        """Public interface for history summarization."""
        return self._summarize_history(history)
