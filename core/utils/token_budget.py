"""core/utils/token_budget.py

Token budgeting and context optimization for Aura Zenith.
Ensures perfect 'voice' consistency by managing the semantic density of prompts.
"""

import logging
from typing import Any

logger = logging.getLogger("Aura.TokenBudget")

class TokenOptimizer:
    """Heuristic-based token budgeting and semantic garbage collection."""
    
    # Heuristic: ~4 characters per token for standard English
    CHARS_PER_TOKEN = 4
    
    def __init__(self, max_total_tokens: int = 4096):
        self.max_total_tokens = max_total_tokens
        # Default budget allocation (tunable)
        self.budgets: dict[str, int] = {
            "identity": 600,
            "internal_state": 400,
            "directives": 300,
            "memory": 800,
            "history": 1500,
            "current_input": 496
        }

    @classmethod
    def estimate(cls, text: str | None) -> int:
        if not text:
            return 0
        return len(text) // cls.CHARS_PER_TOKEN

    def optimize_history(
        self,
        history: list[dict[str, Any]],
        budget_override: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Prunes history while preserving 'Semantic Anchors' (first greeting, last 3 turns).
        """
        budget = budget_override or self.budgets["history"]
        if not history:
            return []

        # Always keep the first message (Anchor) and the latest messages
        anchor = history[0] if history else None
        latest = history[-5:] # Keep last 5 turns as a baseline
        middle = history[1:-5]
        
        optimized: list[dict[str, Any]] = []
        if anchor:
            optimized.append(anchor)
            budget -= self.estimate(anchor.get("content"))

        # Add latest turns first (most relevant)
        current_latest: list[dict[str, Any]] = []
        for msg in reversed(latest):
            tokens = self.estimate(msg.get("content"))
            if budget - tokens >= 0:
                current_latest.insert(0, msg)
                budget -= tokens
            else:
                break
        
        # If budget remains, try to fill with middle history
        current_middle: list[dict[str, Any]] = []
        for msg in reversed(middle):
            # Semantic GC: Drop internal thoughts from 'middle' history to save space
            if msg.get("type") == "internal_thought" or (msg.get("content") or "").startswith("<thought>"):
                continue
                
            tokens = self.estimate(msg.get("content"))
            if budget - tokens >= 0:
                current_middle.insert(0, msg)
                budget -= tokens
            else:
                break
        
        return [anchor] + current_middle + current_latest if anchor and anchor not in current_latest else current_middle + current_latest

    def gc_observations(self, observations: list[str]) -> list[str]:
        """Deduplicate and prune internal observations."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for obs in reversed(observations):
            # Simple fuzzy deduplication (first 30 chars)
            fingerprint = obs[:30]
            if fingerprint not in seen:
                cleaned.insert(0, obs)
                seen.add(fingerprint)
        
        # Limit to top 5 most recent unique observations
        return cleaned[-5:]

def get_optimizer() -> TokenOptimizer:
    return TokenOptimizer()
