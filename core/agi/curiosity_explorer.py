"""core/agi/curiosity_explorer.py
Curiosity-Driven Active Learning
==================================
Translates the curiosity signal from PNEUMA/affect into ACTUAL
learning behavior — not just a number that gets logged.

When curiosity is high, this engine:
  1. Identifies *what* is driving the curiosity (knowledge gaps)
  2. Formulates specific questions to fill those gaps
  3. Queues learning actions: memory queries, web searches, LLM synthesis
  4. Synthesizes findings back into working knowledge
  5. Feeds the satisfaction/disappointment signal back to affect

This is the difference between "feeling curious" and "being curious."
Curiosity becomes behavioral, not just affective.

Learning actions (in priority order):
  1. MEMORY_QUERY — check episodic/semantic memory for existing knowledge
  2. WEB_SEARCH   — search for new information (if tools available)
  3. LLM_SYNTHESIS — synthesize from existing knowledge via LLM reasoning
  4. HEURISTIC_FORMATION — generate a new heuristic rule from findings
"""
from __future__ import annotations

import asyncio
import logging
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.runtime.background_policy import background_activity_allowed

logger = logging.getLogger("Aura.CuriosityExplorer")


def _background_learning_allowed(orchestrator=None) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=900.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=True,
    )

CURIOSITY_THRESHOLD = 0.45    # minimum curiosity to trigger exploration (lowered from 0.65)
MIN_INTERVAL_SECS   = 45.0    # minimum seconds between explorations
MAX_QUEUE_SIZE      = 10      # max pending explorations


@dataclass
class ExplorationItem:
    """A single curiosity-driven learning action."""
    topic: str
    question: str
    action_type: str           # MEMORY_QUERY | WEB_SEARCH | LLM_SYNTHESIS
    priority: float
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    finding: str = ""


class CuriosityExplorer:
    """
    Active learning from curiosity signal.

    Call `tick(curiosity, active_topic)` from the heartbeat.
    Call `run_exploration(orchestrator)` to execute pending items.
    """

    def __init__(self):
        self._queue: List[ExplorationItem] = []
        self._last_exploration: float = 0.0
        self._findings: List[Dict] = []
        self._total_explorations: int = 0
        logger.info("CuriosityExplorer online — curiosity now drives learning.")

    # ── Public API ────────────────────────────────────────────────────────

    def tick(self, curiosity: float, active_topic: Optional[str] = None,
             knowledge_gaps: Optional[List[str]] = None, orchestrator=None):
        """Called each heartbeat. Queues explorations when curiosity is high."""
        if not _background_learning_allowed(orchestrator):
            return
        if curiosity < CURIOSITY_THRESHOLD:
            return
        if time.time() - self._last_exploration < MIN_INTERVAL_SECS:
            return
        if len(self._queue) >= MAX_QUEUE_SIZE:
            return

        # Generate exploration item from gaps or active topic
        topic = active_topic or "current interests"
        gaps = knowledge_gaps or [f"What do I not know about {topic}?"]

        for gap in gaps[:2]:
            item = ExplorationItem(
                topic=topic,
                question=gap,
                action_type=self._choose_action_type(gap),
                priority=curiosity,
            )
            self._queue.append(item)
            logger.debug("CuriosityExplorer queued: %s", gap[:60])

    async def run_exploration(self, orchestrator=None) -> List[ExplorationItem]:
        """Execute the top pending exploration item. Non-blocking."""
        if not _background_learning_allowed(orchestrator):
            return []
        pending = [i for i in self._queue if not i.completed]
        if not pending:
            return []

        # Sort by priority
        pending.sort(key=lambda i: i.priority, reverse=True)
        item = pending[0]
        item.completed = True
        self._last_exploration = time.time()
        self._total_explorations += 1

        try:
            finding = await self._execute(item, orchestrator)
            item.finding = finding
            self._findings.append({
                "topic": item.topic,
                "question": item.question,
                "finding": finding,
                "timestamp": time.time(),
            })
            if len(self._findings) > 100:
                self._findings = self._findings[-100:]

            # Feed finding back into heuristic synthesizer
            if finding and orchestrator:
                await self._synthesize_heuristic(item.question, finding, orchestrator)

            logger.info("CuriosityExplorer completed: %s → %s",
                        item.question[:40], finding[:60])
            return [item]
        except Exception as e:
            logger.debug("Exploration execution failed: %s", e)
            return []

    def get_context_block(self) -> str:
        """Recent findings for prompt injection."""
        if not self._findings:
            return ""
        recent = self._findings[-3:]
        lines = ["## ACTIVE LEARNING (recent curiosity explorations)"]
        for f in recent:
            lines.append(f"- Q: {f['question'][:60]} → {f['finding'][:80]}")
        return "\n".join(lines)

    @property
    def pending_count(self) -> int:
        return sum(1 for i in self._queue if not i.completed)

    # ── Execution ─────────────────────────────────────────────────────────

    def _choose_action_type(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["remember", "memory", "did i", "have i", "before"]):
            return "MEMORY_QUERY"
        if any(w in q for w in ["what do i feel", "how do i feel", "my own state", "my internal state"]):
            return "LLM_SYNTHESIS"
        if any(w in q for w in ["latest", "current", "news", "recent", "today"]):
            return "WEB_SEARCH"
        if "?" in q or any(w in q for w in ["what ", "why ", "how ", "who ", "when ", "where ", "which "]):
            return "WEB_SEARCH"
        return "WEB_SEARCH"

    async def _execute(self, item: ExplorationItem, orchestrator=None) -> str:
        if item.action_type == "MEMORY_QUERY":
            return await self._query_memory(item.question, orchestrator)
        elif item.action_type == "WEB_SEARCH":
            return await self._web_search(item.question, orchestrator)
        else:
            return await self._llm_synthesis(item.question, orchestrator)

    async def _query_memory(self, question: str, orchestrator=None) -> str:
        try:
            from core.container import ServiceContainer
            mem = ServiceContainer.get("memory_manager", default=None)
            if mem and hasattr(mem, "search"):
                results = await asyncio.wait_for(
                    mem.search(question, limit=3), timeout=5.0
                )
                if results:
                    return f"Memory: {'; '.join(str(r)[:60] for r in results[:2])}"
        except Exception as e:
            logger.debug("Memory query failed: %s", e)
        return "No relevant memory found."

    async def _web_search(self, question: str, orchestrator=None) -> str:
        handle = None
        constitutional_core = None
        error_text = None
        success = False
        result_text = ""
        started = time.perf_counter()
        try:
            try:
                if orchestrator is not None:
                    from core.constitution import get_constitutional_core
                    from core.health.degraded_events import record_degraded_event

                    constitutional_core = get_constitutional_core(orchestrator)
                    handle = await constitutional_core.begin_tool_execution(
                        "curiosity_web_search",
                        {"query": question},
                        source="curiosity_explorer",
                        objective=f"Curiosity-driven search: {question}",
                    )
                    if not handle.approved:
                        record_degraded_event(
                            "curiosity_explorer",
                            "web_search_blocked",
                            detail=question[:160],
                            severity="warning",
                            classification="background_degraded",
                            context={"reason": handle.decision.reason},
                        )
                        result_text = "External search deferred by constitutional gate."
                        return result_text
            except Exception as e:
                logger.debug("CuriosityExplorer constitutional gate unavailable: %s", e)

            # Try the skill system (sovereign_browser) which is what actually exists
            try:
                if orchestrator and hasattr(orchestrator, "execute_tool"):
                    result = await asyncio.wait_for(
                        orchestrator.execute_tool(
                            "web_search",
                            {"query": question, "deep": True, "retain": True, "num_results": 6},
                            origin="curiosity_explorer",
                        ),
                        timeout=25.0,
                    )
                    if isinstance(result, dict):
                        summary = (
                            result.get("answer")
                            or result.get("summary")
                            or result.get("content")
                            or result.get("message", "")
                        )
                        if summary:
                            result_text = str(summary)[:200]
                            success = True
                            return result_text
            except Exception as e:
                logger.debug("Skill-based web search failed: %s", e)
                error_text = f"{type(e).__name__}: {e}"

            # Fallback: try direct service, but only when orchestrator is present
            if orchestrator is not None:
                try:
                    from core.container import ServiceContainer
                    search = ServiceContainer.get("web_search", default=None)
                    if search and hasattr(search, "search"):
                        result = await asyncio.wait_for(search.search(question), timeout=10.0)
                        result_text = str(result)[:200] if result else "No web results."
                        success = bool(result_text)
                        return result_text
                except Exception as e:
                    error_text = f"{type(e).__name__}: {e}"

            result_text = await self._llm_synthesis(question, orchestrator)
            return result_text
        finally:
            if (
                handle is not None
                and constitutional_core is not None
                and bool(getattr(handle, "approved", False))
            ):
                try:
                    duration_ms = (time.perf_counter() - started) * 1000.0
                    await constitutional_core.finish_tool_execution(
                        handle,
                        result=str(result_text or error_text or "")[:1000],
                        success=bool(success),
                        duration_ms=duration_ms,
                        error=error_text,
                    )
                except Exception as finish_exc:
                    logger.debug("CuriosityExplorer tool finish skipped: %s", finish_exc)

    async def _llm_synthesis(self, question: str, orchestrator=None) -> str:
        try:
            from core.container import ServiceContainer
            router = ServiceContainer.get("llm_router", default=None)
            if not router:
                return "LLM unavailable."
            from core.brain.llm.llm_router import LLMTier
            prompt = (
                f"Answer this question concisely from your existing knowledge:\n{question}\n"
                "One paragraph, specific and honest about uncertainty."
            )
            result = await asyncio.wait_for(
                router.think(prompt, priority=0.3, is_background=True,
                             prefer_tier=LLMTier.SECONDARY),
                timeout=15.0,
            )
            return (result or "").strip()[:400]
        except Exception as e:
            return f"Synthesis failed: {e}"

    async def _synthesize_heuristic(self, question: str, finding: str,
                                     orchestrator=None):
        try:
            from core.adaptation.heuristic_synthesizer import get_heuristic_synthesizer
            hs = get_heuristic_synthesizer()
            rule = f"When curious about '{question[:50]}': {finding[:80]}"
            hs.ingest_external_heuristic(rule, domain="curiosity_learning",
                                          source="CuriosityExplorer")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

_explorer: Optional[CuriosityExplorer] = None


def get_curiosity_explorer() -> CuriosityExplorer:
    global _explorer
    if _explorer is None:
        _explorer = CuriosityExplorer()
    return _explorer
