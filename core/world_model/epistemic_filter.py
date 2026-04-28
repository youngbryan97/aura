"""core/world_model/epistemic_filter.py

Epistemic Filter — Full Claim Lifecycle Manager.

Every piece of information Aura encounters (RSS feed, web search, conversation,
dream insight) runs through this filter before touching the belief graph.

Claim lifecycle:
  perceive → parse → score source → compare to existing beliefs
    → ACCEPT / TENTATIVE / DISPUTE / REJECT / IGNORE
    → store to BeliefGraph with confidence + provenance

Source trust tiers (descending):
  "self"         → 0.95  (Aura's own reasoning/introspection)
  "known_source" → 0.85  (named, known publication/author)
  "search"       → 0.65  (web search result — verify before trusting)
  "rss"          → 0.60  (RSS feed headline — often incomplete)
  "dream"        → 0.55  (dreamer synthesis — creative, not factual)
  "unknown"      → 0.35  (unattributed claim)

Contradiction resolution:
  new_weight = confidence × source_score
  If new_weight > existing centrality × existing_confidence → DISPUTE (override)
  If similar → TENTATIVE (suspend, flag for review)
  If weaker  → REJECT (reinforce existing belief instead)
"""
from core.runtime.errors import record_degradation
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.EpistemicFilter")

# Source trust scores
_SOURCE_SCORES: Dict[str, float] = {
    "self":         0.95,
    "known_source": 0.85,
    "conversation": 0.75,
    "search":       0.65,
    "rss":          0.60,
    "dream":        0.55,
    "unknown":      0.35,
}

# Outcomes
ACCEPT    = "accept"
TENTATIVE = "tentative"
DISPUTE   = "dispute"
REJECT    = "reject"
IGNORE    = "ignore"

# Minimum length for a claim to be worth processing
_MIN_CLAIM_LEN = 15
# Max claims to extract from a single text blob
_MAX_CLAIMS    = 5


class EpistemicFilter:
    """
    Ingests raw text or explicit claims, evaluates them against existing
    beliefs, and writes accepted/tentative beliefs to the belief graph.
    """

    def __init__(self):
        self._belief_graph = None
        self._ingested_count = 0
        self._accepted_count = 0
        self._rejected_count = 0

    # ─── Public API ───────────────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        source_type: str = "unknown",
        source_label: str = "",
        context: str = "",
        emit_thoughts: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Parse text into atomic claims, evaluate each, and persist to the
        belief graph. Returns a list of outcome records.

        Args:
            text:         Raw text containing one or more claims.
            source_type:  One of the source tier keys (rss, search, dream…).
            source_label: Human-readable source name (e.g. "NYT", "DuckDuckGo").
            context:      Optional surrounding context for disambiguation.
            emit_thoughts: Whether to emit thought cards for notable outcomes.

        Returns:
            List of {"claim", "outcome", "confidence", "source"} dicts.
        """
        if not text or len(text.strip()) < _MIN_CLAIM_LEN:
            return []

        source_score = _SOURCE_SCORES.get(source_type, 0.35)
        claims = self._extract_claims(text)
        outcomes = []

        bg = self._get_belief_graph()

        for claim in claims[:_MAX_CLAIMS]:
            self._ingested_count += 1
            outcome = self._evaluate_claim(claim, source_score, source_label, bg)
            outcomes.append(outcome)

            if outcome["outcome"] in (ACCEPT, TENTATIVE) and bg:
                self._write_to_graph(outcome, bg)
                self._accepted_count += 1
                if emit_thoughts:
                    self._emit_thought(outcome)
            elif outcome["outcome"] == REJECT:
                self._rejected_count += 1

        if outcomes:
            logger.debug(
                "EpistemicFilter: %d claims from '%s' → %d accepted, %d rejected",
                len(outcomes),
                source_label or source_type,
                sum(1 for o in outcomes if o["outcome"] in (ACCEPT, TENTATIVE)),
                sum(1 for o in outcomes if o["outcome"] == REJECT),
            )

        return outcomes

    def ingest_claim(
        self,
        source: str,
        relation: str,
        target: str,
        source_type: str = "unknown",
        source_label: str = "",
        confidence_override: Optional[float] = None,
        centrality: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Ingest an explicit triple claim (source, relation, target) rather
        than parsing it from text. More precise — use when the caller already
        knows the structure.
        """
        source_score = _SOURCE_SCORES.get(source_type, 0.35)
        confidence   = confidence_override if confidence_override is not None else source_score

        bg = self._get_belief_graph()
        outcome = self._evaluate_triple(source, relation, target, confidence, source_score, bg)
        outcome["centrality"] = centrality
        outcome["source_label"] = source_label

        if outcome["outcome"] in (ACCEPT, TENTATIVE) and bg:
            bg.update_belief(
                source=source,
                relation=relation,
                target=target,
                confidence_score=outcome["confidence"],
                centrality=centrality,
            )
            self._accepted_count += 1
            self._emit_thought(outcome)
        else:
            self._rejected_count += 1

        self._ingested_count += 1
        return outcome

    def get_stats(self) -> Dict[str, int]:
        return {
            "ingested": self._ingested_count,
            "accepted": self._accepted_count,
            "rejected": self._rejected_count,
        }

    # ─── Claim Extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """
        Split text into sentence-level atomic claims.
        Strips meta-noise (headlines fragments, URLs, navigation text).
        """
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove navigation-like short fragments
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) >= _MIN_CLAIM_LEN and not s.startswith(("http", "#", "→", "•")):
                claims.append(s)
        return claims

    # ─── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate_claim(
        self,
        claim: str,
        source_score: float,
        source_label: str,
        bg,
    ) -> Dict[str, Any]:
        """
        Evaluate a free-text claim. Since we can't do full NLP triple extraction
        without an LLM call, we store it as a propositional fact and check for
        semantic overlap with existing beliefs via keyword matching.
        """
        # Confidence = source_score (no extra corroboration available here)
        confidence = round(source_score, 3)

        # Check for rough semantic overlap with existing beliefs
        contradiction = self._check_shallow_contradiction(claim, bg)

        if contradiction:
            existing_conf = contradiction.get("confidence", 0.5)
            existing_cent = contradiction.get("centrality", 0.2)
            weight_existing = existing_conf * (1.0 + existing_cent)
            weight_new      = confidence

            if weight_new > weight_existing * 1.2:
                outcome_type = DISPUTE
            elif weight_new < weight_existing * 0.7:
                outcome_type = REJECT
            else:
                outcome_type = TENTATIVE
        else:
            outcome_type = ACCEPT

        return {
            "claim":        claim[:200],
            "outcome":      outcome_type,
            "confidence":   confidence,
            "source_score": source_score,
            "source":       source_label,
            # Use generic triple for propositional storage
            "s": "world",
            "r": "fact",
            "t": claim[:120],
        }

    def _evaluate_triple(
        self,
        source: str,
        relation: str,
        target: str,
        confidence: float,
        source_score: float,
        bg,
    ) -> Dict[str, Any]:
        """Evaluate a structured (source, relation, target) triple."""
        outcome_type = ACCEPT

        if bg and bg.graph.has_edge(source, target):
            edge = bg.graph[source][target]
            existing_conf = float(edge.get("confidence", 0.5))
            existing_cent = float(edge.get("centrality", 0.2))
            weight_existing = existing_conf * (1.0 + existing_cent)
            weight_new      = confidence

            if weight_new > weight_existing * 1.2:
                outcome_type = DISPUTE
            elif weight_new < weight_existing * 0.7:
                outcome_type = REJECT
            else:
                # Similar weight — tentative reinforcement
                outcome_type = TENTATIVE
                confidence = max(existing_conf, confidence)

        return {
            "claim":        f"{source} —[{relation}]→ {target}",
            "outcome":      outcome_type,
            "confidence":   round(confidence, 3),
            "source_score": source_score,
            "s": source,
            "r": relation,
            "t": target,
        }

    # ─── Belief Graph Integration ─────────────────────────────────────────────

    def _write_to_graph(self, outcome: Dict[str, Any], bg):
        """Write an accepted/tentative outcome to the belief graph."""
        s = outcome.get("s", "world")
        r = outcome.get("r", "fact")
        t = outcome.get("t", outcome["claim"][:100])
        confidence  = float(outcome.get("confidence", 0.5))
        centrality  = float(outcome.get("centrality", 0.15))

        try:
            bg.update_belief(
                source=s,
                relation=r,
                target=t,
                confidence_score=confidence,
                centrality=centrality,
            )
        except Exception as e:
            record_degradation('epistemic_filter', e)
            logger.debug("EpistemicFilter write failed: %s", e)

    @staticmethod
    def _check_shallow_contradiction(claim: str, bg) -> Optional[Dict]:
        """
        Lightweight keyword check — looks for existing beliefs whose target
        text overlaps significantly with the new claim's words.
        Returns the edge data dict of the most overlapping belief, or None.
        """
        if not bg:
            return None
        claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
        if not claim_words:
            return None

        best_overlap = 0
        best_edge    = None

        for u, v, data in bg.graph.edges(data=True):
            target_words = set(re.findall(r'\b\w{4,}\b', str(v).lower()))
            overlap = len(claim_words & target_words)
            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_edge = data

        return best_edge

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_belief_graph(self):
        if self._belief_graph:
            return self._belief_graph
        try:
            from core.container import ServiceContainer
            self._belief_graph = ServiceContainer.get("belief_graph", default=None)
        except Exception as _exc:
            record_degradation('epistemic_filter', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return self._belief_graph

    @staticmethod
    def _emit_thought(outcome: Dict[str, Any]):
        try:
            from core.thought_stream import get_emitter
            label = {
                ACCEPT:    "Belief Ingested",
                TENTATIVE: "Belief (Tentative)",
                DISPUTE:   "Belief Disputed",
                REJECT:    "Belief Rejected",
            }.get(outcome["outcome"], "Belief")

            src = outcome.get("source", "")
            claim = outcome.get("claim", "")[:120]
            body = f"[{src}] {claim}" if src else claim

            get_emitter().emit(
                label,
                body,
                level="info",
                category="EpistemicFilter",
            )
        except Exception as _exc:
            record_degradation('epistemic_filter', _exc)
            logger.debug("Suppressed Exception: %s", _exc)


# ── Singleton ──────────────────────────────────────────────────────────────────
_filter: Optional[EpistemicFilter] = None


def get_epistemic_filter() -> EpistemicFilter:
    global _filter
    if _filter is None:
        _filter = EpistemicFilter()
    return _filter
