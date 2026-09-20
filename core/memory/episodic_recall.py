"""The ways an episode is recalled.

Lifted whole out of `episodic_memory`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .episodic_memory import (
        Episode,
    )


class _RecallsEpisodes:
    """Lifted whole out of EpisodicMemory; see episodic_memory.py."""

    async def recall_recent_async(self, limit: int = 10) -> list[Episode]:

        return await asyncio.to_thread(self.recall_recent, limit)

    async def recall_similar_async(self, query: str, limit: int = 5) -> list[Episode]:

        return await asyncio.to_thread(self.recall_similar, query, limit)

    async def recall_failures_async(self, limit: int = 10) -> list[Episode]:

        return await asyncio.to_thread(self.recall_failures, limit)

    async def recall_by_tool_async(self, tool_name: str, limit: int = 10) -> list[Episode]:

        return await asyncio.to_thread(self.recall_by_tool, tool_name, limit)

    def recall_recent(self, limit: int = 10) -> list[Episode]:
        """Retrieve the most recent episodes, ranked by memory strength.
        
        Applies Ebbinghaus decay: old, unimportant, unrehearsed memories
        are ranked lower. Fully decayed memories (strength < 0.05) are
        excluded entirely.
        """

        with self._get_conn() as conn:
            # Fetch more than needed so we can filter out decayed ones
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit * 3,)
            ).fetchall()
        
        episodes = [self._row_to_episode(r) for r in rows]
        
        # Filter out fully decayed memories
        alive = [e for e in episodes if e.current_strength() >= 0.05]
        
        # Sort by strength (recency + importance + emotional salience)
        alive.sort(key=lambda e: e.current_strength(), reverse=True)

        self._observe_ranked_recall(alive, returned_count=limit)
        return self._register_recall(alive[:limit])

    def recall_similar(self, query: str, limit: int = 5) -> list[Episode]:
        """Hybrid search: combines vector similarity with keyword matching.

        Mood-Congruent Recall: Episodes formed in a similar qualia state
        to the current state are boosted in ranking.
        """
        from .episodic_memory import (
            _SLOW_RECALL_S,
            HippocampalIndex,
            logger,
            record_degradation,
            sqlite3,
        )

        seen_ids: set = set()
        combined: list[Episode] = []
        # Each stage timed. Recall from here ran into the retrieval phase's
        # 15 s bound on most live turns (2026-09-19) while the same query on
        # a copy of the same store answered in 0.06 s without the vector
        # stage, so the log has to say which stage the time went to.
        took: dict[str, float] = {}
        began = time.monotonic()

        # 1. Vector search (semantic similarity)
        if self._vector_memory:
            try:
                results = self._vector_memory.search_similar(
                    query=query,
                    k=limit * 2,
                    filter_metadata={"type": "episode"},
                )
                episode_ids = [r.get("metadata", {}).get("episode_id") for r in results if r.get("metadata")]
                episode_ids = [eid for eid in episode_ids if eid]
                if episode_ids:
                    episodes = self._fetch_by_ids(episode_ids)
                    episodes = self._apply_qualia_boost(episodes)
                    for ep in episodes:
                        if ep.episode_id not in seen_ids:
                            seen_ids.add(ep.episode_id)
                            combined.append(ep)
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('episodic_memory', e)
                logger.debug("Vector recall failed: %s", e)
            took["vector"] = time.monotonic() - began

        # 2. Keyword search only when vector recall is insufficient or the user
        # is clearly asking for exact wording.
        stage = time.monotonic()
        if len(combined) < limit or self._query_needs_keyword_fallback(query):
            try:
                keyword_results = self._keyword_search(query, limit)
                for ep in keyword_results:
                    if ep.episode_id not in seen_ids:
                        seen_ids.add(ep.episode_id)
                        combined.append(ep)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('episodic_memory', e)
                logger.debug("Keyword recall failed: %s", e)
        took["keyword"] = time.monotonic() - stage

        # 2b. Associative pattern completion — re-present the query's cues to the
        # hippocampal index to surface engrams that share them (partial cue →
        # whole memory), the way a smell or a word can summon a full episode.
        stage = time.monotonic()
        if len(combined) < limit:
            try:
                cues = HippocampalIndex.extract_cues(query)
                pc = self._hippocampus.pattern_complete(cues, limit=limit, exclude_ids=seen_ids)
                if pc:
                    for ep in self._fetch_by_ids([eid for eid, _ in pc]):
                        if ep.episode_id not in seen_ids:
                            seen_ids.add(ep.episode_id)
                            combined.append(ep)
            except (sqlite3.Error, AttributeError, TypeError, ValueError) as e:
                record_degradation('episodic_memory', e)
                logger.debug("Pattern-completion recall path failed: %s", e)

        # 3. Resolve the ranking by plasticity competition: candidate engrams
        # drive a transient voltage-dependent field so the best-matching trace
        # wins, weakly-relevant ones are gated out below threshold, and the
        # homeostatic bound stops one over-strong trace from swamping recall
        # (anti-confabulation). Falls back to the static importance+recency blend.
        took["association"] = time.monotonic() - stage
        stage = time.monotonic()
        ranked = self._competitive_rank(combined, query)
        self._observe_ranked_recall(ranked, returned_count=limit)
        recalled = self._register_recall(ranked[:limit])
        took["ranking"] = time.monotonic() - stage
        whole = time.monotonic() - began
        if whole >= _SLOW_RECALL_S:
            logger.info(
                "episodic recall took %.1fs (%s)",
                whole,
                ", ".join(f"{name} {seconds:.1f}s" for name, seconds in took.items()),
            )
        return recalled







    def recall_failures(self, limit: int = 10) -> list[Episode]:
        """Retrieve recent failures — the best learning opportunities."""

        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 AND success = 0 "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def recall_by_tool(self, tool_name: str, limit: int = 10) -> list[Episode]:
        """Retrieve episodes involving a specific tool."""

        with self._get_conn() as conn:
            # tools_used is a JSON array; use LIKE for simple matching
            rows = conn.execute(
                "SELECT * FROM episodes WHERE source_authoritative=1 "
                "AND tools_used LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f'%"{tool_name}"%', limit),
            ).fetchall()
        return self._register_recall([self._row_to_episode(r) for r in rows])

