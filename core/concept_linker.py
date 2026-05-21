"""core/concept_linker.py — Aura ConceptLinker v1.0
===================================================
Finds non-obvious relationships between disparate knowledge fragments.

This is the system that notices that a new philosophy belief might
actually explain a weird memory from Phase 2. It creates the "Aha!"
moments by linking nodes in the EpistemicMap.

It operates using several heuristics:
  1. Lexical overlap (surface level)
  2. Thematic resonance (domain overlap)
  3. Contradiction detection (logical tension)
  4. Analogical mapping (structural similarity)

Linked nodes have their 'depth' increased in the EpistemicTracker.
Contradictory links trigger the BeliefChallenger.
High-resonance links become InsightJournal candidates.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ConceptLinker")

_CONCEPT_LINKER_RECOVERABLE_ERRORS = (
    OSError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_NEGATORS = {"not", "no", "never", "cannot", "won't", "without"}


def _record_concept_linker_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "concept_linker",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


@dataclass
class Link:
    """A detected relationship between two concepts."""

    source_concept: str
    target_concept: str
    strength: float  # 0.0-1.0
    link_type: str  # "resonance", "contradiction", "analogy", "derivation"
    reasoning: str  # Why they are linked
    detected_at: float = field(default_factory=time.time)


class ConceptLinker:
    """Connects the dots in Aura's epistemic map."""

    name = "concept_linker"

    def __init__(
        self,
        *,
        scan_interval_seconds: float = 600.0,
        sleep_slice_seconds: float = 10.0,
        max_batch_pairs: int = 5000,
    ):
        self._links: list[Link] = []
        self._epistemic = None
        self._journal = None
        self._challenger = None
        self.running = False
        self._link_task: asyncio.Task | None = None
        self.scan_interval_seconds = max(0.01, float(scan_interval_seconds))
        self.sleep_slice_seconds = max(0.01, float(sleep_slice_seconds))
        self.max_batch_pairs = max(1, int(max_batch_pairs))

    async def start(self):
        if self.running:
            return

        try:
            from core.container import ServiceContainer

            self._epistemic = ServiceContainer.get("epistemic_tracker", default=None)
            self._journal = ServiceContainer.get("insight_journal", default=None)
            self._challenger = ServiceContainer.get("belief_challenger", default=None)
        except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
            _record_concept_linker_degradation(
                exc,
                action="started concept linker without optional epistemic integrations",
                severity="degraded",
            )

        self.running = True
        self._link_task = get_task_tracker().create_task(
            self._link_loop(),
            name="ConceptLinker",
        )

        try:
            from core.event_bus import get_event_bus

            await get_event_bus().publish(
                "mycelium.register",
                {
                    "component": "concept_linker",
                    "hooks_into": [
                        "epistemic_tracker",
                        "belief_challenger",
                        "insight_journal",
                    ],
                },
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_concept_linker_degradation(
                exc,
                action="continued concept linking after event-bus registration failed",
                severity="warning",
            )
            logger.debug("ConceptLinker event-bus registration failed: %s", exc)

        logger.info("ConceptLinker online and looking for connections.")

    async def stop(self):
        self.running = False
        if self._link_task:
            self._link_task.cancel()
            try:
                await asyncio.wait_for(self._link_task, timeout=5.0)
            except asyncio.CancelledError as exc:
                logger.debug("ConceptLinker task acknowledged cancellation: %s", exc)
            except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
                _record_concept_linker_degradation(
                    exc,
                    action="continued shutdown after concept linker task did not stop cleanly",
                    severity="warning",
                )
            self._link_task = None

    async def _link_loop(self):
        """Periodic background linking pass."""
        while self.running:
            remaining = self.scan_interval_seconds
            while remaining > 0 and self.running:
                sleep_for = min(self.sleep_slice_seconds, remaining)
                await asyncio.sleep(sleep_for)
                remaining -= sleep_for

            if self.running:
                try:
                    await self.run_batch_linking()
                except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
                    _record_concept_linker_degradation(
                        exc,
                        action="kept concept linker alive after batch linking failure",
                        severity="degraded",
                    )
                    logger.error("ConceptLinker batch scan failed: %s", exc, exc_info=True)
                    await asyncio.sleep(self.sleep_slice_seconds)

    async def run_batch_linking(self):
        """Scan active knowledge nodes for new links."""
        if not self._epistemic:
            return

        try:
            profile = self._epistemic.get_profile()
        except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
            _record_concept_linker_degradation(
                exc,
                action="skipped concept linking because epistemic profile was unavailable",
                severity="warning",
            )
            return

        nodes = [*getattr(profile, "strong_nodes", ()), *getattr(profile, "weak_nodes", ())]
        concepts = list(self._extract_concepts(nodes))
        if len(concepts) < 2:
            return

        logger.debug("ConceptLinker: batch scan of %d concepts", len(concepts))

        pairs_seen = 0
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1 :]:
                pairs_seen += 1
                if pairs_seen > self.max_batch_pairs:
                    _record_concept_linker_degradation(
                        RuntimeError("concept linker pair budget exhausted"),
                        action="stopped concept scan at configured pair budget",
                        severity="warning",
                        extra={
                            "max_batch_pairs": self.max_batch_pairs,
                            "concept_count": len(concepts),
                        },
                    )
                    return

                tension = self._logical_tension(concept_a, concept_b)
                if tension > 0.7:
                    await self._establish_link(
                        concept_a,
                        concept_b,
                        tension,
                        "contradiction",
                    )
                    continue

                overlap = self._lexical_overlap(concept_a, concept_b)
                if overlap > 0.4:
                    await self._establish_link(concept_a, concept_b, overlap, "resonance")

    def _extract_concepts(self, nodes: list[Any]):
        seen: set[str] = set()
        for node in nodes:
            concept = getattr(node, "concept", node if isinstance(node, str) else None)
            if not isinstance(concept, str):
                continue
            concept = " ".join(concept.split())
            if not concept:
                continue
            key = concept.casefold()
            if key in seen:
                continue
            seen.add(key)
            yield concept

    async def _establish_link(self, a: str, b: str, strength: float, link_type: str):
        if a.casefold() == b.casefold():
            return

        strength = max(0.0, min(1.0, float(strength)))

        for existing in self._links:
            if {existing.source_concept.casefold(), existing.target_concept.casefold()} == {
                a.casefold(),
                b.casefold(),
            }:
                if link_type == "contradiction" and existing.link_type != "contradiction":
                    existing.link_type = "contradiction"
                    existing.strength = max(existing.strength, strength)
                    existing.reasoning = (
                        f"Strength {existing.strength:.2f} link based on contradiction detection."
                    )
                    await self._notify_contradiction(a, b)
                return

        reasoning = f"Strength {strength:.2f} link based on {link_type} detection."
        link = Link(
            source_concept=a,
            target_concept=b,
            strength=strength,
            link_type=link_type,
            reasoning=reasoning,
        )
        self._links.append(link)

        if link_type == "contradiction":
            await self._notify_contradiction(a, b)

        if strength > 0.8 and self._journal:
            await self._record_high_resonance_insight(a, b, strength, link_type)

        logger.info("New concept link: [%s] %s <-> %s (%.2f)", link_type, a[:40], b[:40], strength)

    async def _notify_contradiction(self, a: str, b: str) -> None:
        if self._epistemic and hasattr(self._epistemic, "signal_contradiction"):
            try:
                self._epistemic.signal_contradiction(a, b)
            except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
                _record_concept_linker_degradation(
                    exc,
                    action="recorded contradiction link after epistemic contradiction signal failed",
                    severity="warning",
                    extra={"source": a[:120], "target": b[:120]},
                )

        if self._challenger and hasattr(self._challenger, "challenge_pair"):
            try:
                await self._challenger.challenge_pair(a, b)
            except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
                _record_concept_linker_degradation(
                    exc,
                    action="recorded contradiction link after belief challenge failed",
                    severity="warning",
                    extra={"source": a[:120], "target": b[:120]},
                )

    async def _record_high_resonance_insight(
        self,
        a: str,
        b: str,
        strength: float,
        link_type: str,
    ) -> None:
        try:
            await self._journal.record_insight(
                title=f"Connection found: {a[:30]} <-> {b[:30]}",
                content=(
                    f"Strong {link_type} link detected between existing knowledge nodes.\n"
                    f"Concept A: {a}\n"
                    f"Concept B: {b}"
                ),
                domain="meta",
                confidence=strength,
                source="concept_linker",
            )
        except _CONCEPT_LINKER_RECOVERABLE_ERRORS as exc:
            _record_concept_linker_degradation(
                exc,
                action="kept concept link after insight journal write failed",
                severity="warning",
                extra={"source": a[:120], "target": b[:120], "strength": strength},
            )

    def _lexical_overlap(self, a: str, b: str) -> float:
        words_a = set(self._tokens(a))
        words_b = set(self._tokens(b))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / min(len(words_a), len(words_b))

    def _logical_tension(self, a: str, b: str) -> float:
        tokens_a = self._tokens(a)
        tokens_b = self._tokens(b)
        has_neg_a = any(token in _NEGATORS for token in tokens_a)
        has_neg_b = any(token in _NEGATORS for token in tokens_b)

        if has_neg_a != has_neg_b:
            clean_a = " ".join(token for token in tokens_a if token not in _NEGATORS)
            clean_b = " ".join(token for token in tokens_b if token not in _NEGATORS)
            if self._lexical_overlap(clean_a, clean_b) > 0.6:
                return 0.9
        return 0.0

    def _tokens(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))

    def get_status(self) -> dict[str, Any]:
        return {
            "links_total": len(self._links),
            "running": self.running,
            "max_batch_pairs": self.max_batch_pairs,
        }
