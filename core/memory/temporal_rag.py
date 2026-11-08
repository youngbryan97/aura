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
        """Translates bare timestamps into relative human intuition.

        Within a day, in minutes or hours: "Today" put a sitting three hours gone beside the one going on. LIVE 8
        November 2026, dress rehearsal: asked "So what did you think of all three?", she gave word for word her answer
        to the same question three hours before, of three other videos.
        """
        seconds_old = max(0.0, time.time() - timestamp)
        days_old = int(seconds_old / 86400)
        if seconds_old < 3600:
            minutes = max(1, int(seconds_old / 60))
            return f"[{minutes} minute{'s' if minutes != 1 else ''} ago] {text}"
        if days_old == 0:
            hours = int(seconds_old / 3600)
            return f"[{hours} hour{'s' if hours != 1 else ''} ago] {text}"
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
                metadata = res.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                base = res.get("similarity_score", res.get("score", 0.5))
                ts = res.get("timestamp", metadata.get("timestamp"))
                hits = res.get("reinforcement_count", metadata.get("reinforcement_count", 0))
                # The facade keeps provenance in metadata. Missing time must
                # not become a claim that the memory was recorded today.
                known_time = isinstance(ts, (int, float)) and math.isfinite(ts) and ts > 0
                ranked = dict(res)
                ranked["recall_timestamp"] = ts if known_time else None
                ranked["temporal_score"] = self._calculate_score(
                    float(base or 0.0), ts if known_time else time.time(), int(hits or 0)
                )
                scored.append(ranked)

            scored.sort(key=lambda x: x["temporal_score"], reverse=True)
            
            fragments = []
            for r in scored[:limit]:
                text = r.get("text", "")
                ts = r["recall_timestamp"]
                fragments.append(
                    self._apply_seasonal_context(text, ts)
                    if ts is not None else f"[Time unknown] {text}"
                )
            
            return "\n".join(f"• {f}" for f in fragments)

        # Protect event loop during heavy list sorting and math operations
        formatted_text = await asyncio.to_thread(_process)
        return f"\n[SUBCONSCIOUS TEMPORAL RECALL]\n{formatted_text}\n"
