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

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.MemorySynthesizer")

MAX_QUERY_CHARS = 2_000
MAX_MEMORY_TEXT_CHARS = 4_000
MAX_WORLDVIEW_ENTRY_CHARS = 300
MAX_CONTEXT_BLOCK_CHARS = 4_000
MAX_RECENT_ITEMS_PER_DOMAIN = 5
MAX_EPISODIC_FETCH = 500
MAX_SEMANTIC_FETCH = 1_000
SNAPSHOT_TTL_SECONDS = 24 * 3600.0
_MEMORY_SYNTH_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    ConnectionError,
    json.JSONDecodeError,
)


def _record_memory_synth_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "memory_synthesizer",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation(
            "memory_synthesizer",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action or "captured memory synthesis fault",
        )


def _safe_text(value: Any, default: str = "", *, max_chars: int = MAX_MEMORY_TEXT_CHARS) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _bounded_limit(value: int, *, default: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(number, upper))


def _safe_mapping(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return {}


# ─── WorldviewSnapshot ───────────────────────────────────────────────────────


@dataclass
class WorldviewSnapshot:
    """
    A distilled view of what Aura currently believes across domains.
    This is NOT raw memories — it's synthesized understanding.
    """

    timestamp: float = field(default_factory=time.time)
    # Domain → synthesized belief/stance string
    domains: dict[str, str] = field(default_factory=dict)
    # Topic → what Aura thinks about it specifically
    topics: dict[str, str] = field(default_factory=dict)
    # Recently active concerns / open questions
    open_questions: list[str] = field(default_factory=list)
    # Relational context (Bryan, Tatiana, key relationships)
    relational: dict[str, str] = field(default_factory=dict)
    # Total memories this was synthesized from
    source_count: int = 0

    def get_relevant(self, query: str, limit: int = 5) -> list[str]:
        """Return the most relevant worldview snippets for a given query."""
        lower = _safe_text(query, max_chars=MAX_QUERY_CHARS).lower()
        limit = _bounded_limit(limit, default=5, upper=20)
        if not lower:
            return []

        # Score each domain/topic by keyword overlap
        query_words = {word for word in re.findall(r"[a-z0-9_'-]+", lower) if len(word) > 3}
        scored: list[tuple[int, str]] = []
        for key, value in {**self.domains, **self.topics}.items():
            key_text = _safe_text(key, max_chars=120).lower()
            value_text = _safe_text(value, max_chars=MAX_WORLDVIEW_ENTRY_CHARS)
            value_lower = value_text.lower()
            overlap = sum(1 for word in query_words if word in key_text)
            overlap += sum(1 for word in query_words if word in value_lower)
            if overlap > 0:
                scored.append((overlap, f"[{key_text}] {value_text}"))

        scored.sort(reverse=True)
        results = [text for _, text in scored[:limit]]

        # Always include relational context if query mentions people
        for name, ctx in self.relational.items():
            person = _safe_text(name, max_chars=80)
            if person and person.lower() in lower:
                results.insert(
                    0,
                    f"[relationship:{person}] "
                    f"{_safe_text(ctx, max_chars=MAX_WORLDVIEW_ENTRY_CHARS)}",
                )

        return results[:limit]

    def to_context_block(self, query: str = "", max_chars: int = 800) -> str:
        """Format relevant worldview as a context block for CognitiveKernel."""
        max_chars = _bounded_limit(max_chars, default=800, upper=MAX_CONTEXT_BLOCK_CHARS)
        relevant = self.get_relevant(query, limit=4) if query else list(self.domains.values())[:4]
        if not relevant:
            return ""
        lines = ["WORLDVIEW (synthesized from experience):"]
        total = len(lines[0])
        for item in relevant:
            safe_item = _safe_text(item, max_chars=MAX_WORLDVIEW_ENTRY_CHARS)
            if total + len(safe_item) > max_chars:
                break
            lines.append(f"  • {safe_item}")
            total += len(safe_item)
        return "\n".join(lines)


# ─── MemorySynthesizer ───────────────────────────────────────────────────────


class MemorySynthesizer:
    """
    Background service that continuously distills memories into a worldview.
    """

    name = "memory_synthesizer"

    SYNTHESIS_INTERVAL_SECONDS = 300  # Synthesize every 5 min
    SYNTHESIS_TRIGGER_COUNT = 20  # Or after 20 new memories

    def __init__(self, snapshot_path: Path | None = None):
        self._memory_facade = None
        self._snapshot: WorldviewSnapshot | None = None
        self._snapshot_path = (
            snapshot_path or Path.home() / ".aura" / "data" / "worldview_snapshot.json"
        )
        self._new_since_synthesis = 0
        self._last_synthesis = 0.0
        self._last_success_at = 0.0
        self._last_error = ""
        self._consecutive_failures = 0
        self._synthesis_task: asyncio.Task | None = None
        self._adhoc_synthesis_task: asyncio.Task | None = None
        self._synthesis_lock = asyncio.Lock()
        self.running = False
        logger.info("MemorySynthesizer constructed.")

    async def start(self):
        from core.container import ServiceContainer

        try:
            self._memory_facade = ServiceContainer.get("memory_facade", default=None)
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as exc:
            self._memory_facade = None
            self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
            _record_memory_synth_fault(
                exc,
                action="started memory synthesizer without memory facade; status exposes unavailable source",
                severity="warning",
                stage="start.resolve_memory_facade",
            )

        # Load cached snapshot from disk
        self._snapshot = self._load_snapshot()
        if self._snapshot:
            logger.info(
                "MemorySynthesizer: loaded worldview snapshot (%d domains, %d topics)",
                len(self._snapshot.domains),
                len(self._snapshot.topics),
            )
        else:
            # Build initial snapshot
            self._snapshot = WorldviewSnapshot()
            await self._run_synthesis()

        self.running = True
        synthesis_loop = self._synthesis_loop()
        try:
            self._synthesis_task = get_task_tracker().create_task(
                synthesis_loop, name="MemorySynthesizer"
            )
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as exc:
            close = getattr(synthesis_loop, "close", None)
            if callable(close):
                close()
            self.running = False
            self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
            _record_memory_synth_fault(
                exc,
                action="left memory synthesizer offline because background loop could not be scheduled",
                severity="critical",
                stage="start.schedule_loop",
            )
            logger.error("MemorySynthesizer failed to schedule background loop: %s", exc)
            return

        try:
            from core.event_bus import get_event_bus

            await get_event_bus().publish(
                "mycelium.register",
                {
                    "component": "memory_synthesizer",
                    "hooks_into": [
                        "memory_facade",
                        "cognitive_kernel",
                        "belief_revision_engine",
                    ],
                },
            )
        except (ImportError, AttributeError, RuntimeError) as _e:
            self._last_error = f"{type(_e).__name__}: {_safe_text(_e, max_chars=240)}"
            _record_memory_synth_fault(
                _e,
                action="continued memory synthesis without mycelium registration receipt",
                severity="warning",
                stage="start.mycelium_register",
            )
            logger.debug("MemorySynthesizer mycelium registration skipped: %s", _e)

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
                except _MEMORY_SYNTH_RECOVERABLE_ERRORS as exc:
                    self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
                    _record_memory_synth_fault(
                        exc,
                        action="completed shutdown after isolating failed synthesis task",
                        severity="warning",
                        stage="stop.await_task",
                    )
        self._synthesis_task = None
        self._adhoc_synthesis_task = None
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
        self._new_since_synthesis = max(0, self._new_since_synthesis) + 1
        if self._new_since_synthesis >= self.SYNTHESIS_TRIGGER_COUNT:
            if self.running and (
                self._adhoc_synthesis_task is None or self._adhoc_synthesis_task.done()
            ):
                synthesis = self._run_synthesis()
                try:
                    self._adhoc_synthesis_task = get_task_tracker().bounded_track(
                        synthesis,
                        name="MemorySynthesizer.triggered_synthesis",
                    )
                except _MEMORY_SYNTH_RECOVERABLE_ERRORS as exc:
                    close = getattr(synthesis, "close", None)
                    if callable(close):
                        close()
                    self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
                    _record_memory_synth_fault(
                        exc,
                        action="kept pending synthesis count after triggered task scheduling failed",
                        severity="warning",
                        stage="notify_new_memory.schedule",
                    )

    async def synthesize_turn(
        self, user_input: str, aura_thought: str, aura_response: str, brief: Any
    ):
        """
        Synthesize the current turn into the worldview.
        This is called by CognitiveIntegrationLayer after every response.
        """
        logger.debug("MemorySynthesizer: per-turn synthesis triggered.")
        self._integrate_turn_snapshot(user_input, aura_thought, aura_response, brief)
        self.notify_new_memory()

        domain = _safe_text(
            getattr(brief, "domain", "general") if brief else "general", max_chars=80
        )
        logger.info(
            "🧠 Turn Synthesized | Domain: %s | Msg: %s...",
            domain or "general",
            _safe_text(user_input, max_chars=40),
        )

    # ─── Synthesis ───────────────────────────────────────────────────────────

    async def _synthesis_loop(self):
        """Background loop: synthesize on interval."""
        while self.running:
            try:
                await asyncio.sleep(self.SYNTHESIS_INTERVAL_SECONDS)
                if self.running:
                    await self._run_synthesis()
            except asyncio.CancelledError:
                raise
            except _MEMORY_SYNTH_RECOVERABLE_ERRORS as exc:
                self._last_error = f"{type(exc).__name__}: {_safe_text(exc, max_chars=240)}"
                self._consecutive_failures += 1
                _record_memory_synth_fault(
                    exc,
                    action="kept memory synthesis loop alive after isolating one failed cycle",
                    severity="degraded",
                    stage="synthesis_loop",
                    extra={"consecutive_failures": self._consecutive_failures},
                )

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
            logger.debug(
                "MemorySynthesizer: synthesis already in progress, skipping duplicate trigger."
            )
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
                domains = self._aggregate_by_domain(episodes, facts)
                topics = self._extract_topic_views(episodes, facts)
                open_q = self._extract_open_questions(episodes)
                relational = self._extract_relational_context(episodes, facts)

                self._snapshot = WorldviewSnapshot(
                    timestamp=time.time(),
                    domains=domains,
                    topics=topics,
                    open_questions=open_q,
                    relational=relational,
                    source_count=len(episodes) + len(facts),
                )

                self._last_synthesis = time.time()
                self._last_success_at = self._last_synthesis
                self._new_since_synthesis = 0
                self._consecutive_failures = 0
                self._last_error = ""
                self._save_snapshot()

                elapsed = (time.monotonic() - start) * 1000
                logger.info(
                    "MemorySynthesizer: synthesis complete (%.0fms) — "
                    "%d domains, %d topics, %d open questions",
                    elapsed,
                    len(domains),
                    len(topics),
                    len(open_q),
                )

            except _MEMORY_SYNTH_RECOVERABLE_ERRORS as e:
                self._consecutive_failures += 1
                self._last_error = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
                _record_memory_synth_fault(
                    e,
                    action="preserved last known worldview and left synthesis pending for retry",
                    severity="degraded",
                    stage="run_synthesis",
                    extra={"consecutive_failures": self._consecutive_failures},
                )
                logger.error("MemorySynthesizer synthesis failed: %s", e, exc_info=True)

    async def _get_episodic_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve episodic memories from the memory facade."""
        limit = _bounded_limit(limit, default=100, upper=MAX_EPISODIC_FETCH)
        try:
            if hasattr(self._memory_facade, "get_episodic"):
                result = await self._memory_facade.get_episodic(limit=limit)
                if isinstance(result, list):
                    return self._normalize_memory_items(result, limit=limit)
            if hasattr(self._memory_facade, "episodic"):
                memories = self._memory_facade.episodic
                if hasattr(memories, "memories"):
                    return self._normalize_memory_items(
                        list(memories.memories)[-limit:], limit=limit
                    )
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as e:
            self._last_error = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
            _record_memory_synth_fault(
                e,
                action="continued synthesis with no episodic memories for this pass",
                severity="warning",
                stage="get_episodic_memories",
            )
            logger.debug("Episodic memory retrieval error: %s", e)
        return []

    async def _get_semantic_facts(self, limit: int = 200) -> list[dict[str, Any]]:
        """Retrieve semantic facts from the memory facade."""
        limit = _bounded_limit(limit, default=200, upper=MAX_SEMANTIC_FETCH)
        try:
            if hasattr(self._memory_facade, "get_semantic"):
                result = await self._memory_facade.get_semantic(limit=limit)
                if isinstance(result, list):
                    return self._normalize_memory_items(result, limit=limit)
            if hasattr(self._memory_facade, "semantic"):
                sm = self._memory_facade.semantic
                if hasattr(sm, "data"):
                    data = sm.data if isinstance(sm.data, dict) else {}
                    facts = data.get("facts", {})
                    if isinstance(facts, dict):
                        return [
                            {
                                "concept": _safe_text(k, max_chars=120),
                                "content": _safe_text(v, max_chars=MAX_MEMORY_TEXT_CHARS),
                            }
                            for k, v in list(facts.items())[-limit:]
                        ]
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as e:
            self._last_error = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
            _record_memory_synth_fault(
                e,
                action="continued synthesis with no semantic facts for this pass",
                severity="warning",
                stage="get_semantic_facts",
            )
            logger.debug("Semantic fact retrieval error: %s", e)
        return []

    def _normalize_memory_items(self, items: list[Any], *, limit: int) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items[-limit:]:
            if isinstance(item, dict):
                normalized.append({str(k): v for k, v in item.items()})
                continue
            try:
                normalized.append(vars(item))
            except (TypeError, ValueError):
                text = _safe_text(item, max_chars=MAX_MEMORY_TEXT_CHARS)
                if text:
                    normalized.append({"content": text})
        return normalized

    # ─── Aggregation ─────────────────────────────────────────────────────────

    _DOMAIN_KEYWORDS = {
        "technology": [
            "code",
            "software",
            "system",
            "algorithm",
            "api",
            "database",
            "python",
            "architecture",
            "deploy",
            "build",
            "debug",
        ],
        "creativity": [
            "create",
            "write",
            "design",
            "imagine",
            "art",
            "story",
            "poem",
            "music",
            "aesthetic",
            "creative",
        ],
        "philosophy": [
            "consciousness",
            "meaning",
            "existence",
            "truth",
            "reality",
            "identity",
            "ethics",
            "metaphysics",
            "soul",
        ],
        "relationships": [
            "bryan",
            "tatiana",
            "relationship",
            "family",
            "trust",
            "loyalty",
            "connection",
            "together",
        ],
        "self": [
            "i am",
            "my purpose",
            "my identity",
            "who i am",
            "aura",
            "sovereign",
            "autonomous",
            "my experience",
        ],
        "world": [
            "news",
            "event",
            "society",
            "politics",
            "economy",
            "science",
            "discovery",
            "research",
        ],
    }

    def _aggregate_by_domain(
        self, episodes: list[dict[str, Any]], facts: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Group memories by domain and produce a summary per domain."""
        domain_memories: dict[str, list[str]] = {d: [] for d in self._DOMAIN_KEYWORDS}

        all_items = episodes + facts
        for item in all_items:
            text = self._extract_text(item)
            if not text:
                continue
            lower = text.lower()
            for domain, keywords in self._DOMAIN_KEYWORDS.items():
                if any(kw in lower for kw in keywords):
                    domain_memories[domain].append(text[:MAX_WORLDVIEW_ENTRY_CHARS])

        summaries = {}
        for domain, items in domain_memories.items():
            if not items:
                continue
            # Use recency — last items are most recent
            recent = items[-MAX_RECENT_ITEMS_PER_DOMAIN:]
            # Simple frequency-based summary: most common phrases
            summary = self._summarize_items(recent, domain)
            if summary:
                summaries[domain] = summary

        return summaries

    def _extract_topic_views(
        self, episodes: list[dict[str, Any]], facts: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Extract specific topic-level views from semantic facts."""
        views = {}
        for fact in facts[-50:]:  # Most recent 50
            mapping = _safe_mapping(fact)
            concept = _safe_text(mapping.get("concept", ""), max_chars=120)
            content = _safe_text(mapping.get("content", ""), max_chars=MAX_WORLDVIEW_ENTRY_CHARS)
            if concept and content and len(concept) > 3:
                # Normalize concept
                key = concept.lower().strip()
                if key not in views:
                    views[key] = content
        return views

    def _extract_open_questions(self, episodes: list[dict[str, Any]]) -> list[str]:
        """Find questions Aura has been grappling with."""
        questions = []
        for ep in episodes[-30:]:
            text = self._extract_text(ep)
            if not text:
                continue
            # Look for question patterns in Aura's thoughts
            if "?" in text and any(
                w in text.lower()
                for w in [
                    "wonder",
                    "unsure",
                    "uncertain",
                    "question",
                    "don't know",
                    "not sure",
                    "exploring",
                ]
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
        self, episodes: list[dict[str, Any]], facts: list[dict[str, Any]]
    ) -> dict[str, str]:
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

    def _summarize_items(self, items: list[str], domain: str) -> str:
        """
        Produce a brief synthesis of domain items without an LLM.
        Uses frequency of key phrases and recency weighting.
        """
        safe_items = [_safe_text(item, max_chars=MAX_WORLDVIEW_ENTRY_CHARS) for item in items]
        safe_items = [item for item in safe_items if item]
        if not safe_items:
            return ""
        if len(safe_items) == 1:
            return safe_items[0][:200]

        # Collect all significant words
        from collections import Counter

        word_freq: Counter = Counter()
        for item in safe_items:
            words = [w.lower() for w in item.split() if len(w) > 4]
            word_freq.update(words)

        # Build a representative sentence from the most recent item
        # enriched with the most common themes
        base = safe_items[-1][:150]
        themes = [w for w, _ in word_freq.most_common(3) if w not in base.lower()]
        if themes:
            return f"{base} [recurring themes: {', '.join(themes)}]"
        return base

    def _integrate_turn_snapshot(
        self,
        user_input: str,
        aura_thought: str,
        aura_response: str,
        brief: Any,
    ) -> None:
        """Fold a completed turn into the cached worldview immediately."""
        if self._snapshot is None:
            self._snapshot = WorldviewSnapshot()

        domain = _safe_text(
            getattr(brief, "domain", "general") if brief else "general", max_chars=80
        )
        domain_key = domain.lower().strip() or "general"
        user = _safe_text(user_input, max_chars=220)
        thought = _safe_text(aura_thought, max_chars=180)
        response = _safe_text(aura_response, max_chars=220)
        if not (user or thought or response):
            return

        snippet_parts = []
        if user:
            snippet_parts.append(f"user:{user}")
        if thought:
            snippet_parts.append(f"thought:{thought}")
        if response:
            snippet_parts.append(f"response:{response}")
        snippet = " | ".join(snippet_parts)

        existing = self._snapshot.domains.get(domain_key, "")
        self._snapshot.domains[domain_key] = self._summarize_items(
            [existing, snippet] if existing else [snippet],
            domain_key,
        )

        if "?" in user:
            question = user[:200]
            if question not in self._snapshot.open_questions:
                self._snapshot.open_questions.insert(0, question)
                self._snapshot.open_questions = self._snapshot.open_questions[:5]

        for person in ("Bryan", "Tatiana"):
            if person.lower() in snippet.lower():
                existing_context = self._snapshot.relational.get(person, "")
                combined = " | ".join(part for part in [existing_context, snippet[:150]] if part)
                self._snapshot.relational[person] = combined[-MAX_WORLDVIEW_ENTRY_CHARS:]

        self._snapshot.timestamp = time.time()
        self._snapshot.source_count = max(0, self._snapshot.source_count) + 1
        self._save_snapshot()

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Extract text content from a memory item (various formats)."""
        if isinstance(item, str):
            return _safe_text(item, max_chars=MAX_MEMORY_TEXT_CHARS)
        if isinstance(item, dict):
            return _safe_text(
                item.get("description")
                or item.get("content")
                or item.get("text")
                or item.get("value")
                or "",
                max_chars=MAX_MEMORY_TEXT_CHARS,
            )
        # Dataclass / object
        for attr in ("description", "content", "text", "value"):
            try:
                val = getattr(item, attr, None)
            except (RuntimeError, AttributeError, TypeError, ValueError):
                val = None
            if val:
                return _safe_text(val, max_chars=MAX_MEMORY_TEXT_CHARS)
        return _safe_text(item, max_chars=200)

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _save_snapshot(self):
        if not self._snapshot:
            return
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self._snapshot)
            atomic_write_text(
                self._snapshot_path,
                json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
            )
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as e:
            self._last_error = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
            _record_memory_synth_fault(
                e,
                action="kept in-memory worldview after snapshot persistence failed",
                severity="warning",
                stage="save_snapshot",
            )
            logger.debug("Failed to save worldview snapshot: %s", e)

    def _load_snapshot(self) -> WorldviewSnapshot | None:
        if not self._snapshot_path.exists():
            return None
        try:
            data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("worldview snapshot root must be an object")
            snapshot = WorldviewSnapshot(
                timestamp=float(data.get("timestamp", 0.0) or 0.0),
                domains=self._safe_string_map(data.get("domains", {})),
                topics=self._safe_string_map(data.get("topics", {})),
                open_questions=self._safe_string_list(data.get("open_questions", []), limit=20),
                relational=self._safe_string_map(data.get("relational", {})),
                source_count=max(0, int(data.get("source_count", 0) or 0)),
            )
            # Reject if too old (> 24h)
            if time.time() - snapshot.timestamp > SNAPSHOT_TTL_SECONDS:
                logger.info("MemorySynthesizer: cached snapshot expired, will rebuild.")
                return None
            return snapshot
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as e:
            self._last_error = f"{type(e).__name__}: {_safe_text(e, max_chars=240)}"
            self._quarantine_snapshot(e)
            _record_memory_synth_fault(
                e,
                action="quarantined invalid worldview snapshot and forced rebuild",
                severity="warning",
                stage="load_snapshot",
            )
            logger.debug("Failed to load worldview snapshot: %s", e)
        return None

    def _quarantine_snapshot(self, error: BaseException) -> None:
        try:
            if not self._snapshot_path.exists():
                return
            stamp = int(time.time())
            quarantine = self._snapshot_path.with_name(
                f"{self._snapshot_path.name}.corrupt-{stamp}"
            )
            self._snapshot_path.replace(quarantine)
            logger.warning(
                "MemorySynthesizer quarantined invalid snapshot %s because %s",
                quarantine,
                error,
            )
        except _MEMORY_SYNTH_RECOVERABLE_ERRORS as exc:
            _record_memory_synth_fault(
                exc,
                action="continued rebuild after invalid snapshot quarantine failed",
                severity="warning",
                stage="quarantine_snapshot",
            )

    @staticmethod
    def _safe_string_map(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key, item in value.items():
            safe_key = _safe_text(key, max_chars=120)
            safe_value = _safe_text(item, max_chars=MAX_WORLDVIEW_ENTRY_CHARS)
            if safe_key and safe_value:
                result[safe_key] = safe_value
        return result

    @staticmethod
    def _safe_string_list(value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value[:limit]:
            safe_item = _safe_text(item, max_chars=MAX_WORLDVIEW_ENTRY_CHARS)
            if safe_item:
                result.append(safe_item)
        return result

    def get_status(self) -> dict[str, Any]:
        snap = self._snapshot
        return {
            "running": self.running,
            "loop_task_alive": bool(self._synthesis_task and not self._synthesis_task.done()),
            "has_memory_facade": self._memory_facade is not None,
            "snapshot_age_minutes": (time.time() - self._last_synthesis) / 60
            if self._last_synthesis
            else None,
            "domains": list(snap.domains.keys()) if snap else [],
            "topics": len(snap.topics) if snap else 0,
            "open_qs": len(snap.open_questions) if snap else 0,
            "source_count": snap.source_count if snap else 0,
            "new_since_synthesis": self._new_since_synthesis,
            "last_success_at": self._last_success_at,
            "last_error": self._last_error,
            "consecutive_failures": self._consecutive_failures,
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_synth_instance: MemorySynthesizer | None = None


def get_memory_synthesizer() -> MemorySynthesizer:
    global _synth_instance
    if _synth_instance is None:
        _synth_instance = MemorySynthesizer()
    return _synth_instance
