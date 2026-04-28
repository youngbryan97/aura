"""core/autonomy/depth_gate.py
──────────────────────────────
The "have I actually understood this, or did I skim?" gate.

The depth gate scores an engagement against multiple independent criteria
and returns both a pass/fail decision and a per-criterion breakdown. It is
deliberately strict: the failure mode we are guarding against is Aura
declaring something "completed" after reading a Wikipedia first paragraph.

Criteria (each scored 0.0–1.0, all aggregated into a final pass decision):

  1. *verification_substance* — the four verification questions answered
     with substantive specificity (≥ N tokens, mentions specific named
     entities/scenes, not just vague summaries).
  2. *source_diversity* — engaged at multiple priority levels (e.g. watched
     it AND read creator commentary AND skimmed discussion).
  3. *opinion_formed* — has a defended opinion (agreement or disagreement
     with at least one critical view), not "I agreed with everything."
  4. *factual_recall* — can produce specific factual recall (named
     character/scene/quote/argument) that wouldn't be in a generic summary.
  5. *anti_skim_test* — the comprehension log shows multiple checkpoints,
     not one terminal summary.
  6. *open_threads_resolution* — open threads are either resolved or
     consciously parked with a "would revisit if X" note, not left dangling.

Per-content-type adjusted thresholds let a 3-minute YouTube Short pass with
weaker source diversity than a 10-episode show. Hard floor on
verification_substance + factual_recall regardless of type.

Public API:
    gate = DepthGate()
    report = gate.evaluate(item, comprehension_summaries, reflection_record)
    if report.passed: ...
    report.criteria  # dict of {criterion_name: float}
    report.failures  # list of human-readable reasons
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ── Scoring thresholds (tunable) ──────────────────────────────────────────

VERIFICATION_MIN_TOKENS = 30        # per question, substantive answer
SUBSTANTIVE_NAMED_ENTITY_MIN = 2    # at least N proper nouns / scene refs
HARD_FLOOR_VERIFICATION = 0.55
HARD_FLOOR_FACTUAL_RECALL = 0.50
DEFAULT_PASS_THRESHOLD = 0.70

# Content-type-specific adjustments
CONTENT_TYPE_PROFILES: Dict[str, Dict[str, float]] = {
    "youtube_short":  {"diversity_weight": 0.5, "min_checkpoints": 1, "pass": 0.60},
    "youtube_video":  {"diversity_weight": 0.8, "min_checkpoints": 2, "pass": 0.65},
    "feature_film":   {"diversity_weight": 1.0, "min_checkpoints": 4, "pass": 0.72},
    "tv_show":        {"diversity_weight": 1.0, "min_checkpoints": 6, "pass": 0.75},
    "tv_episode":     {"diversity_weight": 1.0, "min_checkpoints": 2, "pass": 0.70},
    "comic_series":   {"diversity_weight": 1.0, "min_checkpoints": 4, "pass": 0.72},
    "article":        {"diversity_weight": 0.6, "min_checkpoints": 1, "pass": 0.60},
    "book":           {"diversity_weight": 1.0, "min_checkpoints": 5, "pass": 0.75},
    "default":        {"diversity_weight": 1.0, "min_checkpoints": 2, "pass": 0.70},
}

# Token approximation: words ≈ 0.75 tokens for plain English
_WORD_RE = re.compile(r"\b[\w']+\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_HEDGE_RE = re.compile(r"\b(maybe|perhaps|generally|in general|roughly|sort of|kind of)\b", re.IGNORECASE)


@dataclass
class DepthReport:
    passed: bool
    score: float
    criteria: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    threshold: float = DEFAULT_PASS_THRESHOLD
    content_type: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 3),
            "criteria": {k: round(v, 3) for k, v in self.criteria.items()},
            "failures": self.failures,
            "notes": self.notes,
            "threshold": self.threshold,
            "content_type": self.content_type,
        }


class DepthGate:
    def __init__(self, pass_threshold: Optional[float] = None) -> None:
        self.default_threshold = pass_threshold or DEFAULT_PASS_THRESHOLD

    # ── Main evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        item: Any,                # ContentItem from curated_media_loader
        verification_answers: Dict[str, str],
        priority_levels_engaged: Sequence[int],
        critical_view_engaged: Optional[str],
        own_opinion: Optional[str],
        opinion_disagrees_somewhere: bool,
        comprehension_checkpoints: Sequence[Dict[str, Any]],
        open_threads: Sequence[str],
        parked_threads: Sequence[Dict[str, str]],
    ) -> DepthReport:
        """
        Args mirror the structured outputs that comprehension_loop and
        reflection_loop produce. Each is independently checkable, so the gate
        does not depend on any specific upstream module's internals.
        """
        content_type = self._infer_content_type(item)
        profile = CONTENT_TYPE_PROFILES.get(content_type, CONTENT_TYPE_PROFILES["default"])

        criteria: Dict[str, float] = {}
        failures: List[str] = []
        notes: List[str] = []

        # 1. Verification substance
        verif = self._score_verification(verification_answers)
        criteria["verification_substance"] = verif
        if verif < HARD_FLOOR_VERIFICATION:
            failures.append(
                f"verification answers too thin (score={verif:.2f}, floor={HARD_FLOOR_VERIFICATION})"
            )

        # 2. Source diversity
        diversity = self._score_diversity(priority_levels_engaged, profile["diversity_weight"])
        criteria["source_diversity"] = diversity
        distinct_levels = len(set(int(l) for l in priority_levels_engaged if l))
        if distinct_levels < 2 and profile["diversity_weight"] >= 0.8:
            failures.append(
                f"only {distinct_levels} priority level(s) engaged; need ≥2 for this content type"
            )

        # 3. Opinion formed
        opinion = self._score_opinion(critical_view_engaged, own_opinion, opinion_disagrees_somewhere)
        criteria["opinion_formed"] = opinion
        if opinion < 0.4:
            failures.append("no defended opinion / no engagement with a critical view")

        # 4. Factual recall
        recall = self._score_factual_recall(comprehension_checkpoints, verification_answers)
        criteria["factual_recall"] = recall
        if recall < HARD_FLOOR_FACTUAL_RECALL:
            failures.append(
                f"factual recall too generic (score={recall:.2f}); reads as summary not engagement"
            )

        # 5. Anti-skim
        anti_skim = self._score_anti_skim(comprehension_checkpoints, profile["min_checkpoints"])
        criteria["anti_skim"] = anti_skim
        if len(comprehension_checkpoints) < profile["min_checkpoints"]:
            failures.append(
                f"only {len(comprehension_checkpoints)} checkpoint(s); "
                f"min={profile['min_checkpoints']} for {content_type}"
            )

        # 6. Open threads resolution
        open_resolved = self._score_open_threads(open_threads, parked_threads)
        criteria["open_threads_resolution"] = open_resolved
        if open_resolved < 0.4 and len(open_threads) > 3:
            failures.append(
                f"{len(open_threads)} open threads with no parking rationale; "
                "either resolve or park them"
            )

        # Aggregate
        weights = {
            "verification_substance":  0.25,
            "source_diversity":        0.20 * profile["diversity_weight"],
            "opinion_formed":          0.15,
            "factual_recall":          0.20,
            "anti_skim":               0.10,
            "open_threads_resolution": 0.10,
        }
        wsum = sum(weights.values()) or 1.0
        score = sum(criteria[k] * weights[k] for k in criteria) / wsum

        threshold = profile["pass"]
        passed = (
            score >= threshold
            and criteria["verification_substance"] >= HARD_FLOOR_VERIFICATION
            and criteria["factual_recall"] >= HARD_FLOOR_FACTUAL_RECALL
            and not failures
        )

        if not passed and not failures:
            failures.append(f"aggregate score {score:.2f} below threshold {threshold:.2f}")

        notes.append(f"content_type={content_type}, profile_pass={threshold:.2f}")
        if criteria["source_diversity"] >= 0.85:
            notes.append("source diversity is strong — engaged at multiple priority levels")
        if criteria["opinion_formed"] >= 0.85 and opinion_disagrees_somewhere:
            notes.append("opinion is defended and includes substantive disagreement")

        return DepthReport(
            passed=passed,
            score=score,
            criteria=criteria,
            failures=failures,
            notes=notes,
            threshold=threshold,
            content_type=content_type,
        )

    # ── Per-criterion scoring helpers ────────────────────────────────────

    def _score_verification(self, answers: Dict[str, str]) -> float:
        if not answers:
            return 0.0
        per_q: List[float] = []
        for question, ans in answers.items():
            ans = (ans or "").strip()
            if not ans:
                per_q.append(0.0)
                continue
            words = _WORD_RE.findall(ans)
            n_words = len(words)
            n_proper = len(set(_PROPER_NOUN_RE.findall(ans)))
            n_hedges = len(_HEDGE_RE.findall(ans))

            length_score = min(1.0, n_words / VERIFICATION_MIN_TOKENS)
            specificity = min(1.0, n_proper / max(1, SUBSTANTIVE_NAMED_ENTITY_MIN))
            hedge_penalty = max(0.0, 1.0 - (n_hedges / max(1, n_words / 30)))

            per_q.append(0.5 * length_score + 0.4 * specificity + 0.1 * hedge_penalty)
        return sum(per_q) / len(per_q)

    def _score_diversity(self, levels: Sequence[int], weight: float) -> float:
        levels_clean = sorted({int(l) for l in levels if l})
        if not levels_clean:
            return 0.0
        # Reward both number of levels and span (1+5+6 is richer than 4+5+6)
        n_distinct = len(levels_clean)
        span = (max(levels_clean) - min(levels_clean)) if n_distinct > 1 else 0
        base = min(1.0, n_distinct / 3.0)
        span_bonus = min(0.3, span / 10.0)
        raw = min(1.0, base + span_bonus)
        return raw * (0.5 + 0.5 * weight)  # diversity-weight-adjusted

    def _score_opinion(self, critical_view: Optional[str], own: Optional[str], disagrees: bool) -> float:
        own_strength = 1.0 if (own and len(own.split()) >= 12) else 0.3 if own else 0.0
        critical_strength = 1.0 if (critical_view and len(critical_view.split()) >= 12) else 0.3 if critical_view else 0.0
        disagree_bonus = 0.4 if disagrees else 0.0
        return min(1.0, 0.4 * own_strength + 0.3 * critical_strength + disagree_bonus)

    def _score_factual_recall(
        self,
        checkpoints: Sequence[Dict[str, Any]],
        answers: Dict[str, str],
    ) -> float:
        # Aggregate text from both checkpoints and answers; count specific entities.
        corpus_parts: List[str] = []
        for ck in checkpoints:
            for k in ("summary", "extracted_facts", "scene_descriptions", "quotes"):
                v = ck.get(k)
                if isinstance(v, str):
                    corpus_parts.append(v)
                elif isinstance(v, list):
                    corpus_parts.extend(str(x) for x in v)
        corpus_parts.extend(answers.values())
        corpus = " ".join(corpus_parts)
        if not corpus.strip():
            return 0.0

        n_proper = len(set(_PROPER_NOUN_RE.findall(corpus)))
        n_quotes = corpus.count('"') // 2
        density = n_proper / max(1, len(corpus.split()) / 50)

        return min(1.0, 0.5 * min(1.0, n_proper / 8.0) + 0.3 * min(1.0, n_quotes / 3.0) + 0.2 * min(1.0, density))

    def _score_anti_skim(self, checkpoints: Sequence[Dict[str, Any]], min_required: int) -> float:
        if not checkpoints:
            return 0.0
        n = len(checkpoints)
        coverage = min(1.0, n / max(1, min_required))
        # Penalize if all checkpoints have identical or near-identical summaries
        summaries = [str(c.get("summary", "")) for c in checkpoints]
        unique_ratio = len(set(summaries)) / max(1, len(summaries))
        return 0.7 * coverage + 0.3 * unique_ratio

    def _score_open_threads(self, open_threads: Sequence[str], parked: Sequence[Dict[str, str]]) -> float:
        n_open = len(open_threads)
        n_parked = len(parked)
        if n_open == 0:
            return 1.0
        # If parking notes exist for at least half the open threads, score well
        rationale_quality = sum(
            1.0 if (p.get("rationale") and len(p.get("rationale", "")) >= 12) else 0.5
            for p in parked
        )
        ratio = rationale_quality / max(1, n_open)
        return min(1.0, ratio)

    # ── Content-type inference ───────────────────────────────────────────

    def _infer_content_type(self, item: Any) -> str:
        # Permissive: works whether item is a ContentItem dataclass or a dict
        title = (getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else "") or "").lower()
        category = (getattr(item, "category", None) or (item.get("category") if isinstance(item, dict) else "") or "").lower()
        url = (getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else "") or "").lower()

        if "youtube.com" in url and ("/shorts/" in url):
            return "youtube_short"
        if "youtube.com" in url:
            return "youtube_video"
        if any(t in title for t in ("comic", "manga")):
            return "comic_series"
        if "fiction about ai" in category or "tv" in category:
            return "tv_show"
        if "film" in category or any(t in title for t in ("film", "movie")):
            return "feature_film"
        if any(t in title for t in ("book", "novel")):
            return "book"
        if any(t in url for t in ("/article", "wired.com", "nytimes.com", "blog")):
            return "article"
        return "default"
