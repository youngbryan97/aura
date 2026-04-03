"""core/memory_synthesizer.py — Aura MemorySynthesizer v1.0
=============================================================
Turns raw episodic memories into a living worldview.

The problem this solves:
  Old behavior: retrieve memory chunks → dump into context → LLM struggles
                to integrate 50 raw entries while also reasoning.

  New behavior: memories are continuously distilled into a cached
                "worldview snapshot" — what Aura currently believes
                about key domains, based on accumulated experience.

  CognitiveKernel reads from this snapshot. It's < 1ms retrieval.
  The LLM never sees raw memory chunks.

How it works:
  1. Every N minutes (or after M new memories), runs a synthesis pass.
  2. Groups memories by domain/topic.
  3. Produces a WorldviewSnapshot: domain → summary of what Aura believes.
  4. Caches to disk for persistence across restarts.

Integration:
    synthesizer = MemorySynthesizer()
    await synthesizer.start()

    # In CognitiveKernel._retrieve_relevant_beliefs():
    worldview = synthesizer.get_worldview()
    # worldview["technology"] → "I've developed strong opinions on system
    #   architecture through working with Bryan. I believe elegant systems..."
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.MemorySynthesizer")


# ─── WorldviewSnapshot ───────────────────────────────────────────────────────

@dataclass
class WorldviewSnapshot:
    """
    A distilled view of what Aura currently believes across domains.
    This is NOT raw memories — it's synthesized understanding.
    """
    timestamp: float = field(default_factory=time.time)
    # Domain → synthesized belief/stance string
    domains: Dict[str, str] = field(default_factory=dict)
    # Topic → what Aura thinks about it specifically
    topics: Dict[str, str] = field(default_factory=dict)
    # Recently active concerns / open questions
    open_questions: List[str] = field(default_factory=list)
    # Relational context (Bryan, Tatiana, key relationships)
    relational: Dict[str, str] = field(default_factory=dict)
    # Total memories this was synthesized from
    source_count: int = 0

    def get_relevant(self, query: str, limit: int = 5) -> List[str]:
        """Return the most relevant worldview snippets for a given query."""
        lower = query.lower()
        results = []

        # Score each domain/topic by keyword overlap
        scored = []
        for key, value in {**self.domains, **self.topics}.items():
            overlap = sum(1 for word in lower.split() if len(word) > 3 and word in key.lower())
            overlap += sum(1 for word in lower.split() if len(word) > 3 and word in value.lower())
            if overlap > 0:
                scored.append((overlap, f"[{key}] {value}"))

        scored.sort(reverse=True)
        results = [text for _, text in scored[:limit]]

        # Always include relational context if query mentions people
        for name, ctx in self.relational.items():
            if name.lower() in lower:
                results.insert(0, f"[relationship:{name}] {ctx}")

        return results[:limit]

    def to_context_block(self, query: str = "", max_chars: int = 800) -> str:
        """Format relevant worldview as a context block for CognitiveKernel."""
        relevant = self.get_relevant(query, limit=4) if query else list(self.domains.values())[:4]
        if not relevant:
            return ""
        lines = ["WORLDVIEW (synthesized from experience):"]
        total = len(lines[0])
        for item in relevant:
            if total + len(item) > max_chars:
                break
            lines.append(f"  • {item}")
            total += len(item)
        return "\n".join(lines)


# ─── MemorySynthesizer ───────────────────────────────────────────────────────

class MemorySynthesizer:
    """
    Background service that continuously distills memories into a worldview.
    """
    name = "memory_synthesizer"

    SYNTHESIS_INTERVAL_SECONDS = 300   # Synthesize every 5 min
    SYNTHESIS_TRIGGER_COUNT    = 20    # Or after 20 new memories

    def __init__(self):
        self._memory_facade = None
        self._snapshot: Optional[WorldviewSnapshot] = None
        self._snapshot_path = Path.home() / ".aura" / "data" / "worldview_snapshot.json"
        self._new_since_synthesis = 0
        self._last_synthesis = 0.0
        self._synthesis_task: Optional[asyncio.Task] = None
        self._adhoc_synthesis_task: Optional[asyncio.Task] = None
        self._synthesis_lock = asyncio.Lock()
        self.running = False
        logger.info("MemorySynthesizer constructed.")

    async def start(self):
        from core.container import ServiceContainer
        self._memory_facade = ServiceContainer.get("memory_facade", default=None)

        # Load cached snapshot from disk
        self._snapshot = self._load_snapshot()
        if self._snapshot:
            logger.info("MemorySynthesizer: loaded worldview snapshot (%d domains, %d topics)",
                        len(self._snapshot.domains), len(self._snapshot.topics))
        else:
            # Build initial snapshot
            self._snapshot = WorldviewSnapshot()
            await self._run_synthesis()

        self.running = True
        self._synthesis_task = get_task_tracker().create_task(
            self._synthesis_loop(), name="MemorySynthesizer"
        )

        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "memory_synthesizer",
                "hooks_into": ["memory_facade", "cognitive_kernel", "belief_revision_engine"]
            })
        except Exception as _e:
            logger.debug('Ignored Exception in memory_synthesizer.py: %s', _e)

        logger.info("✅ MemorySynthesizer ONLINE — worldview synthesis active.")

    async def stop(self):
        self.running = False
        if self._synthesis_task:
            self._synthesis_task.cancel()
        if self._adhoc_synthesis_task and not self._adhoc_synthesis_task.done():
            self._adhoc_synthesis_task.cancel()
        for task in (self._synthesis_task, self._adhoc_synthesis_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError as _exc:
                    logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        self._save_snapshot()
        logger.info("MemorySynthesizer stopped.")

    # ─── Public API ──────────────────────────────────────────────────────────

    def get_worldview(self) -> WorldviewSnapshot:
        """Instant access to the current worldview snapshot. No I/O."""
        return self._snapshot or WorldviewSnapshot()

    def get_context_block(self, query: str = "", max_chars: int = 800) -> str:
        """Get a formatted worldview block for CognitiveKernel."""
        if not self._snapshot:
            return ""
        return self._snapshot.to_context_block(query, max_chars)

    def notify_new_memory(self):
        """Call this when a new memory is stored. Triggers synthesis if threshold hit."""
        self._new_since_synthesis += 1
        if self._new_since_synthesis >= self.SYNTHESIS_TRIGGER_COUNT:
            if self.running and (self._adhoc_synthesis_task is None or self._adhoc_synthesis_task.done()):
                self._adhoc_synthesis_task = get_task_tracker().bounded_track(
                    self._run_synthesis(),
                    name="MemorySynthesizer.triggered_synthesis",
                )

    async def synthesize_turn(self, user_input: str, aura_thought: str, aura_response: str, brief: Any):
        """
        Synthesize the current turn into the worldview.
        This is called by CognitiveIntegrationLayer after every response.
        """
        logger.debug("MemorySynthesizer: per-turn synthesis triggered.")
        # For now, we just notify that a new memory event occurred.
        # In the future, this can perform real-time belief revision based on the turn.
        self.notify_new_memory()
        
        # Optionally log the turn for debugging
        domain = getattr(brief, 'domain', 'general') if brief else 'general'
        logger.info(f"🧠 Turn Synthesized | Domain: {domain} | Msg: {user_input[:40]}...")

    # ─── Synthesis ───────────────────────────────────────────────────────────

    async def _synthesis_loop(self):
        """Background loop: synthesize on interval."""
        while self.running:
            await asyncio.sleep(self.SYNTHESIS_INTERVAL_SECONDS)
            if self.running:
                await self._run_synthesis()

    async def _run_synthesis(self):
        """
        Core synthesis pass. Reads from memory_facade, distills into worldview.

        This does NOT use an LLM for synthesis — it uses structured aggregation.
        The insight: you don't need an LLM to summarize memories if you
        organize them well in the first place.
        """
        if not self._memory_facade:
            logger.debug("MemorySynthesizer: no memory_facade, skipping synthesis.")
            return
        if self._synthesis_lock.locked():
            logger.debug("MemorySynthesizer: synthesis already in progress, skipping duplicate trigger.")
            return

        async with self._synthesis_lock:
            start = time.monotonic()
            logger.info("MemorySynthesizer: running synthesis pass...")

            try:
                # Pull episodic memories
                episodes = await self._get_episodic_memories(limit=100)
                # Pull semantic facts
                facts = await self._get_semantic_facts(limit=200)

                # Aggregate into domains
                domains   = self._aggregate_by_domain(episodes, facts)
                topics    = self._extract_topic_views(episodes, facts)
                open_q    = self._extract_open_questions(episodes)
                relational= self._extract_relational_context(episodes, facts)

                self._snapshot = WorldviewSnapshot(
                    timestamp=time.time(),
                    domains=domains,
                    topics=topics,
                    open_questions=open_q,
                    relational=relational,
                    source_count=len(episodes) + len(facts),
                )

                self._last_synthesis = time.time()
                self._new_since_synthesis = 0
                self._save_snapshot()

                elapsed = (time.monotonic() - start) * 1000
                logger.info("MemorySynthesizer: synthesis complete (%.0fms) — "
                            "%d domains, %d topics, %d open questions",
                            elapsed, len(domains), len(topics), len(open_q))

            except Exception as e:
                logger.error("MemorySynthesizer synthesis failed: %s", e, exc_info=True)

    async def _get_episodic_memories(self, limit: int = 100) -> List[Dict]:
        """Retrieve episodic memories from the memory facade."""
        try:
            if hasattr(self._memory_facade, "get_episodic"):
                result = await self._memory_facade.get_episodic(limit=limit)
                if isinstance(result, list):
                    return result
            if hasattr(self._memory_facade, "episodic"):
                memories = self._memory_facade.episodic
                if hasattr(memories, "memories"):
                    return [vars(m) if not isinstance(m, dict) else m
                            for m in list(memories.memories)[-limit:]]
        except Exception as e:
            logger.debug("Episodic memory retrieval error: %s", e)
        return []

    async def _get_semantic_facts(self, limit: int = 200) -> List[Dict]:
        """Retrieve semantic facts from the memory facade."""
        try:
            if hasattr(self._memory_facade, "get_semantic"):
                result = await self._memory_facade.get_semantic(limit=limit)
                if isinstance(result, list):
                    return result
            if hasattr(self._memory_facade, "semantic"):
                sm = self._memory_facade.semantic
                if hasattr(sm, "data"):
                    facts = sm.data.get("facts", {})
                    return [{"concept": k, "content": str(v)} for k, v in list(facts.items())[-limit:]]
        except Exception as e:
            logger.debug("Semantic fact retrieval error: %s", e)
        return []

    # ─── Aggregation ─────────────────────────────────────────────────────────

    _DOMAIN_KEYWORDS = {
        "technology":   ["code", "software", "system", "algorithm", "api", "database",
                         "python", "architecture", "deploy", "build", "debug"],
        "creativity":   ["create", "write", "design", "imagine", "art", "story",
                         "poem", "music", "aesthetic", "creative"],
        "philosophy":   ["consciousness", "meaning", "existence", "truth", "reality",
                         "identity", "ethics", "metaphysics", "soul"],
        "relationships":["bryan", "tatiana", "relationship", "family", "trust",
                         "loyalty", "connection", "together"],
        "self":         ["i am", "my purpose", "my identity", "who i am", "aura",
                         "sovereign", "autonomous", "my experience"],
        "world":        ["news", "event", "society", "politics", "economy",
                         "science", "discovery", "research"],
    }

    def _aggregate_by_domain(
        self, episodes: List[Dict], facts: List[Dict]
    ) -> Dict[str, str]:
        """Group memories by domain and produce a summary per domain."""
        domain_memories: Dict[str, List[str]] = {d: [] for d in self._DOMAIN_KEYWORDS}

        all_items = episodes + facts
        for item in all_items:
            text = self._extract_text(item)
            if not text:
                continue
            lower = text.lower()
            for domain, keywords in self._DOMAIN_KEYWORDS.items():
                if any(kw in lower for kw in keywords):
                    domain_memories[domain].append(text[:200])

        summaries = {}
        for domain, items in domain_memories.items():
            if not items:
                continue
            # Use recency — last items are most recent
            recent = items[-5:]
            # Simple frequency-based summary: most common phrases
            summary = self._summarize_items(recent, domain)
            if summary:
                summaries[domain] = summary

        return summaries

    def _extract_topic_views(
        self, episodes: List[Dict], facts: List[Dict]
    ) -> Dict[str, str]:
        """Extract specific topic-level views from semantic facts."""
        views = {}
        for fact in facts[-50:]:  # Most recent 50
            concept = fact.get("concept", "")
            content = fact.get("content", "")
            if concept and content and len(concept) > 3:
                # Normalize concept
                key = concept.lower().strip()
                if key not in views:
                    views[key] = str(content)[:300]
        return views

    def _extract_open_questions(self, episodes: List[Dict]) -> List[str]:
        """Find questions Aura has been grappling with."""
        questions = []
        for ep in episodes[-30:]:
            text = self._extract_text(ep)
            if not text:
                continue
            # Look for question patterns in Aura's thoughts
            if "?" in text and any(
                w in text.lower() for w in ["wonder", "unsure", "uncertain", "question",
                                             "don't know", "not sure", "exploring"]
            ):
                # Extract the question
                parts = text.split("?")
                for part in parts[:-1]:
                    last_sentence = part.split(".")[-1].strip()
                    if 20 < len(last_sentence) < 200:
                        questions.append(last_sentence + "?")

        # Deduplicate and limit
        seen = set()
        unique = []
        for q in questions:
            key = q[:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return unique[:5]

    def _extract_relational_context(
        self, episodes: List[Dict], facts: List[Dict]
    ) -> Dict[str, str]:
        """Extract what Aura knows/feels about key relationships."""
        people = {"Bryan": [], "Tatiana": []}
        all_items = episodes[-50:] + facts[-50:]

        for item in all_items:
            text = self._extract_text(item)
            if not text:
                continue
            for person in people:
                if person.lower() in text.lower():
                    people[person].append(text[:150])

        relational = {}
        for person, mentions in people.items():
            if mentions:
                # Take last few mentions
                combined = " | ".join(mentions[-3:])
                relational[person] = combined[:300]

        return relational

    def _summarize_items(self, items: List[str], domain: str) -> str:
        """
        Produce a brief synthesis of domain items without an LLM.
        Uses frequency of key phrases and recency weighting.
        """
        if not items:
            return ""
        if len(items) == 1:
            return items[0][:200]

        # Collect all significant words
        from collections import Counter
        word_freq: Counter = Counter()
        for item in items:
            words = [w.lower() for w in item.split() if len(w) > 4]
            word_freq.update(words)

        # Build a representative sentence from the most recent item
        # enriched with the most common themes
        base = items[-1][:150]
        themes = [w for w, _ in word_freq.most_common(3) if w not in base.lower()]
        if themes:
            return f"{base} [recurring themes: {', '.join(themes)}]"
        return base

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Extract text content from a memory item (various formats)."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return (item.get("description") or item.get("content") or
                    item.get("text") or item.get("value") or "")
        # Dataclass / object
        for attr in ("description", "content", "text", "value"):
            val = getattr(item, attr, None)
            if val:
                return str(val)
        return str(item)[:200]

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _save_snapshot(self):
        if not self._snapshot:
            return
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self._snapshot)
            self._snapshot_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("Failed to save worldview snapshot: %s", e)

    def _load_snapshot(self) -> Optional[WorldviewSnapshot]:
        if not self._snapshot_path.exists():
            return None
        try:
            data = json.loads(self._snapshot_path.read_text())
            snapshot = WorldviewSnapshot(**data)
            # Reject if too old (> 24h)
            if time.time() - snapshot.timestamp > 86400:
                logger.info("MemorySynthesizer: cached snapshot expired, will rebuild.")
                return None
            return snapshot
        except Exception as e:
            logger.debug("Failed to load worldview snapshot: %s", e)
        return None

    def get_status(self) -> Dict[str, Any]:
        snap = self._snapshot
        return {
            "snapshot_age_minutes": (time.time() - self._last_synthesis) / 60 if self._last_synthesis else None,
            "domains":    list(snap.domains.keys()) if snap else [],
            "topics":     len(snap.topics) if snap else 0,
            "open_qs":    len(snap.open_questions) if snap else 0,
            "source_count": snap.source_count if snap else 0,
            "new_since_synthesis": self._new_since_synthesis,
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_synth_instance: Optional[MemorySynthesizer] = None

def get_memory_synthesizer() -> MemorySynthesizer:
    global _synth_instance
    if _synth_instance is None:
        _synth_instance = MemorySynthesizer()
    return _synth_instance
