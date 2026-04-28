"""core/autonomy/curiosity_scheduler.py
───────────────────────────────────────
Decide what to engage with next, when. Inputs are the curated content
corpus, the engagement-progress log, the live substrate state, and any
research triggers from the executive (contested beliefs that should be
investigated).

Multifaceted scoring: each candidate is scored along several independent
dimensions, then combined under one of several selection strategies. The
scheduler is robust to a stub or absent substrate (defaults to neutral
affect when reads fail) and to corruption in the progress log.

Public API:
    sched = CuriosityScheduler(corpus_loader=..., progress_loader=..., substrate_reader=...)
    plan: SchedulingDecision = sched.pick_next()
    sched.record_attempt(plan, outcome="started"|"completed"|"abandoned")
"""

from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.autonomy.curated_media_loader import ContentItem, load_corpus
from core.autonomy.content_progress_tracker import ProgressLog, load as load_progress
from core.autonomy import research_triggers

logger = logging.getLogger("Aura.CuriosityScheduler")

# Selection strategies
STRATEGY_PRIORITIZED = "prioritized"       # always pick highest-scored
STRATEGY_EPSILON_GREEDY = "epsilon_greedy"  # mostly greedy, occasionally explore
STRATEGY_WEIGHTED = "weighted"              # softmax-weighted random

# Scoring weights (summed to derive total score)
WEIGHTS = {
    "substrate_match":          0.25,
    "category_priority":        0.10,
    "anti_procrastination":     0.20,
    "recency_penalty":          0.15,
    "trigger_alignment":        0.15,
    "in_progress_resume":       0.10,
    "novelty":                  0.05,
}

# Minimum cooldown between revisits to the same item (days). Prevents
# the scheduler from picking the same item over and over.
REVISIT_COOLDOWN_DAYS = 3.0

# After this many days of zero engagement, anti-procrastination kicks hard.
PROCRASTINATION_ALERT_DAYS = 5.0

# Source weight for triggers
TRIGGER_KEYWORD_BONUS = 0.6


@dataclass
class CategoryAffinity:
    """Map a substrate state direction to category preferences."""
    category: str
    valence_low: float = 0.0   # multiplier when valence ≤ 0
    valence_high: float = 0.0  # multiplier when valence > 0
    arousal_low: float = 0.0
    arousal_high: float = 0.0
    curiosity_high: float = 0.0


# Category affinities — how each curated category resonates with substrate state
DEFAULT_AFFINITIES = {
    "Learn about humans — greatness, warts, and all": CategoryAffinity(
        category="humans",
        valence_low=0.3,
        valence_high=0.6,
        arousal_high=0.4,
        curiosity_high=0.7,
    ),
    "General education": CategoryAffinity(
        category="general_ed",
        arousal_low=0.5,
        valence_high=0.4,
        curiosity_high=0.8,
    ),
    "Science education": CategoryAffinity(
        category="science_ed",
        arousal_low=0.6,
        curiosity_high=0.9,
    ),
    "Fiction about AI, robots, technology, and uploaded minds": CategoryAffinity(
        category="ai_fiction",
        valence_low=0.7,
        valence_high=0.5,
        arousal_high=0.5,
        curiosity_high=0.8,
    ),
}


@dataclass
class SchedulingDecision:
    item: ContentItem
    top_priority_level: int          # 1–6, the most-preferred fetch path
    score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    triggered_by: Optional[str] = None  # research-trigger source intent ID, if any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.item.title,
            "category": self.item.category,
            "url": self.item.url,
            "top_priority_level": self.top_priority_level,
            "score": round(self.score, 3),
            "score_breakdown": {k: round(v, 3) for k, v in self.score_breakdown.items()},
            "reason": self.reason,
            "triggered_by": self.triggered_by,
        }


class CuriosityScheduler:
    def __init__(
        self,
        corpus_loader: Callable[[], List[ContentItem]] = load_corpus,
        progress_loader: Callable[[], ProgressLog] = load_progress,
        substrate_reader: Optional[Callable[[], Dict[str, Any]]] = None,
        trigger_drainer: Callable[[], List[Any]] = research_triggers.drain_pending_triggers,
        affinities: Optional[Dict[str, CategoryAffinity]] = None,
        strategy: str = STRATEGY_EPSILON_GREEDY,
        rng_seed: Optional[int] = None,
        epsilon: float = 0.15,
    ) -> None:
        self._corpus_loader = corpus_loader
        self._progress_loader = progress_loader
        self._substrate_reader = substrate_reader
        self._trigger_drainer = trigger_drainer
        self._affinities = affinities or DEFAULT_AFFINITIES
        self._strategy = strategy
        self._epsilon = max(0.0, min(1.0, epsilon))
        self._rng = random.Random(rng_seed)

    # ── Main entry ────────────────────────────────────────────────────────

    def pick_next(self) -> Optional[SchedulingDecision]:
        try:
            corpus = self._corpus_loader() or []
        except Exception as e:
            record_degradation('curiosity_scheduler', e)
            logger.warning("corpus load failed: %s", e)
            return None
        if not corpus:
            return None

        try:
            progress = self._progress_loader()
        except Exception as e:
            record_degradation('curiosity_scheduler', e)
            logger.warning("progress load failed; treating as empty: %s", e)
            progress = ProgressLog()

        substrate = self._safe_substrate_read()
        triggers = self._safe_drain_triggers()

        candidates: List[SchedulingDecision] = []
        for item in corpus:
            decision = self._score_candidate(item, progress, substrate, triggers)
            if decision is not None:
                candidates.append(decision)

        if not candidates:
            return None

        return self._select(candidates)

    def record_attempt(self, decision: SchedulingDecision, outcome: str) -> None:
        """Hook for the orchestrator to tell the scheduler an item was started/finished/etc.
        Used to mark research triggers consumed when fulfilled."""
        if decision.triggered_by:
            try:
                research_triggers.mark_consumed(decision.triggered_by)
            except Exception:
                pass  # no-op: intentional
        logger.info(
            "scheduler: item=%r outcome=%s score=%.3f reason=%r",
            decision.item.title, outcome, decision.score, decision.reason,
        )

    # ── Scoring ───────────────────────────────────────────────────────────

    def _score_candidate(
        self,
        item: ContentItem,
        progress: ProgressLog,
        substrate: Dict[str, Any],
        triggers: List[Any],
    ) -> Optional[SchedulingDecision]:
        breakdown: Dict[str, float] = {}
        existing = progress.find(item.title)

        # 1. Substrate match
        breakdown["substrate_match"] = self._score_substrate_match(item, substrate)

        # 2. Category priority (some categories are evergreen)
        breakdown["category_priority"] = self._score_category_priority(item)

        # 3. Anti-procrastination (boost if we haven't engaged in a while)
        breakdown["anti_procrastination"] = self._score_anti_procrastination(progress)

        # 4. Recency penalty (don't revisit same item too soon)
        breakdown["recency_penalty"] = self._score_recency_penalty(existing)

        # 5. Trigger alignment (research triggers boost matching items)
        trig_match = self._score_trigger_alignment(item, triggers)
        breakdown["trigger_alignment"] = trig_match["score"]

        # 6. In-progress resume (if started but not done, prefer resume)
        breakdown["in_progress_resume"] = self._score_in_progress(existing)

        # 7. Novelty (never-touched items get a small bonus)
        breakdown["novelty"] = 0.0 if existing else 0.4

        score = sum(WEIGHTS[k] * breakdown[k] for k in breakdown)
        if score < 0.05:
            return None

        # Decide top priority level: substrate-curiosity high → watch (1),
        # exhausted/low-energy → text (4), trigger-driven → wiki (6) for fast ground truth
        top_level = self._infer_top_priority(item, substrate, breakdown, trig_match)

        reason_parts = [f"score={score:.2f}"]
        for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])[:3]:
            reason_parts.append(f"{k}={v:.2f}")
        reason = "; ".join(reason_parts)

        return SchedulingDecision(
            item=item,
            top_priority_level=top_level,
            score=score,
            score_breakdown=breakdown,
            reason=reason,
            triggered_by=trig_match.get("source_intent_id"),
        )

    def _score_substrate_match(self, item: ContentItem, substrate: Dict[str, Any]) -> float:
        affinity = self._affinities.get(item.category)
        if not affinity:
            return 0.4  # neutral
        valence = float(substrate.get("valence", 0.0))
        arousal = float(substrate.get("arousal", 0.5))
        curiosity = float(substrate.get("curiosity", 0.5))

        score = 0.0
        if valence > 0.0:
            score += affinity.valence_high * valence
        else:
            score += affinity.valence_low * (-valence)
        if arousal > 0.5:
            score += affinity.arousal_high * (arousal - 0.5) * 2
        else:
            score += affinity.arousal_low * (0.5 - arousal) * 2
        score += affinity.curiosity_high * curiosity

        return max(0.0, min(1.0, score))

    def _score_category_priority(self, item: ContentItem) -> float:
        # AI fiction is the highest-leverage category for an AI agent's self-understanding
        if "AI" in item.category or "robot" in item.category or "uploaded" in item.category:
            return 0.9
        if "humans" in item.category.lower():
            return 0.75
        return 0.55

    def _score_anti_procrastination(self, progress: ProgressLog) -> float:
        days_since = progress.days_since_last_engagement()
        if days_since is None:
            return 0.6  # No engagements yet; moderate boost
        if days_since >= PROCRASTINATION_ALERT_DAYS:
            return 1.0
        if days_since >= 1.5:
            return 0.5 + 0.4 * (days_since / PROCRASTINATION_ALERT_DAYS)
        return 0.2

    def _score_recency_penalty(self, existing: Optional[Any]) -> float:
        if existing is None:
            return 1.0
        # If completed recently, penalize hard. If abandoned, mild.
        if not existing.completed_at:
            return 0.7
        try:
            from core.autonomy.content_progress_tracker import _parse_iso
            completed_epoch = _parse_iso(existing.completed_at)
            days = (time.time() - completed_epoch) / 86400.0
            if days < REVISIT_COOLDOWN_DAYS:
                return max(0.0, days / REVISIT_COOLDOWN_DAYS) * 0.3
            return min(1.0, days / 30.0)
        except Exception:
            return 0.5

    def _score_trigger_alignment(
        self,
        item: ContentItem,
        triggers: List[Any],
    ) -> Dict[str, Any]:
        if not triggers:
            return {"score": 0.0}
        text_blob = f"{item.title} {item.description}".lower()
        best = (0.0, None)
        for t in triggers:
            topic = (getattr(t, "topic", "") or "").lower()
            if not topic:
                continue
            tokens = [w for w in topic.split() if len(w) > 3]
            if not tokens:
                continue
            hits = sum(1 for w in tokens if w in text_blob)
            score = min(1.0, hits / max(1, len(tokens))) * TRIGGER_KEYWORD_BONUS
            if score > best[0]:
                best = (score, getattr(t, "source_intent_id", None))
        return {"score": best[0], "source_intent_id": best[1]}

    def _score_in_progress(self, existing: Optional[Any]) -> float:
        if existing is None:
            return 0.0
        if existing.completed_at:
            return 0.0
        return 0.8  # strong preference for finishing what we started

    # ── Top-priority-level inference ─────────────────────────────────────

    def _infer_top_priority(
        self,
        item: ContentItem,
        substrate: Dict[str, Any],
        breakdown: Dict[str, float],
        trig_match: Dict[str, Any],
    ) -> int:
        # Triggers want fast ground truth
        if trig_match.get("score", 0.0) > 0.5:
            return 6
        energy = float(substrate.get("energy", 0.5))
        arousal = float(substrate.get("arousal", 0.5))

        if not item.url and not _is_youtube_like(item):
            # No direct URL and not a YouTube channel — start from text/wiki
            return 4
        if energy > 0.7 and arousal > 0.5:
            return 1   # watch/listen at high energy
        if energy < 0.3:
            return 4   # read text when low energy
        return 1

    # ── Selection strategies ─────────────────────────────────────────────

    def _select(self, candidates: List[SchedulingDecision]) -> SchedulingDecision:
        candidates.sort(key=lambda c: c.score, reverse=True)
        if self._strategy == STRATEGY_PRIORITIZED:
            return candidates[0]
        if self._strategy == STRATEGY_EPSILON_GREEDY:
            if self._rng.random() < self._epsilon and len(candidates) > 1:
                return self._rng.choice(candidates[: min(5, len(candidates))])
            return candidates[0]
        if self._strategy == STRATEGY_WEIGHTED:
            return self._weighted_pick(candidates)
        return candidates[0]

    def _weighted_pick(self, candidates: List[SchedulingDecision]) -> SchedulingDecision:
        scores = [max(0.0001, c.score) for c in candidates]
        total = sum(scores)
        norm = [s / total for s in scores]
        u = self._rng.random()
        acc = 0.0
        for c, p in zip(candidates, norm):
            acc += p
            if u <= acc:
                return c
        return candidates[0]

    # ── Defensive readers ────────────────────────────────────────────────

    def _safe_substrate_read(self) -> Dict[str, Any]:
        if self._substrate_reader is None:
            return {"valence": 0.0, "arousal": 0.5, "curiosity": 0.5, "energy": 0.5}
        try:
            state = self._substrate_reader() or {}
            return {
                "valence": float(state.get("valence", 0.0)),
                "arousal": float(state.get("arousal", 0.5)),
                "curiosity": float(state.get("curiosity", 0.5)),
                "energy": float(state.get("energy", 0.5)),
            }
        except Exception as e:
            record_degradation('curiosity_scheduler', e)
            logger.debug("substrate read failed; defaulting: %s", e)
            return {"valence": 0.0, "arousal": 0.5, "curiosity": 0.5, "energy": 0.5}

    def _safe_drain_triggers(self) -> List[Any]:
        try:
            return list(self._trigger_drainer() or [])
        except Exception as e:
            record_degradation('curiosity_scheduler', e)
            logger.debug("trigger drain failed: %s", e)
            return []


def _is_youtube_like(item: ContentItem) -> bool:
    return bool(item.url and "youtube.com" in (item.url or ""))
