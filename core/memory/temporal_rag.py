"""core/memory/temporal_rag.py

Applies mathematical decay and seasonal weighting to vector similarity scores.
"""
import time
import math
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger("Aura.TemporalRAG")

class TimeWeightedRetriever:
    def __init__(self, decay_rate: float = 0.012):
        self.decay_rate = decay_rate

    def _calculate_score(self, base_sim: float, timestamp: float, hits: int) -> float:
        """
        Formula: (Base Similarity) * exp(-decay * days_old) + (Reinforcement Bonus)
        """
        current_time = time.time()
        days_old = max(0, current_time - timestamp) / 86400.0

        time_penalty = math.exp(-self.decay_rate * days_old)
        reinforcement_bonus = min(0.20, hits * 0.02)

        return min(1.0, (base_sim * time_penalty) + reinforcement_bonus)

    def _apply_seasonal_context(self, text: str, timestamp: float) -> str:
        """Translates bare timestamps into relative human intuition."""
        days_old = int((time.time() - timestamp) / 86400)
        if days_old == 0:
            return f"[Today] {text}"
        elif days_old < 30:
            return f"[{days_old} days ago] {text}"
        elif days_old < 365:
            months = days_old // 30
            return f"[{months} months ago] {text}"
        else:
            years = round(days_old / 365, 1)
            return f"[{years} years ago] {text}"

    async def rerank_and_format(self, raw_results: List[Dict[Any, Any]], limit: int = 4) -> str:
        """Asynchronously recalculates FAISS scores and formats the subconscious prompt."""
        if not raw_results:
            return ""

        def _process():
            scored = []
            for res in raw_results:
                base = res.get("similarity_score", 0.5)
                ts = res.get("timestamp", time.time())
                hits = res.get("reinforcement_count", 0)
                
                res["temporal_score"] = self._calculate_score(base, ts, hits)
                scored.append(res)

            scored.sort(key=lambda x: x["temporal_score"], reverse=True)
            
            fragments = []
            for r in scored[:limit]:
                text = r.get("text", "")
                ts = r.get("timestamp", time.time())
                fragments.append(self._apply_seasonal_context(text, ts))
            
            return "\n".join(f"• {f}" for f in fragments)

        # Protect event loop during heavy list sorting and math operations
        formatted_text = await asyncio.to_thread(_process)
        return f"\n[SUBCONSCIOUS TEMPORAL RECALL]\n{formatted_text}\n"
