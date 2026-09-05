"""core/social/relational_intelligence.py

Relational Intelligence Engine.

Deep relational skill modeling: vulnerability reciprocity, conflict resolution,
perspective tracking, and engagement profiling. Gives Aura the social awareness
to navigate the full spectrum of human relational dynamics — not just what the
user said, but how they relate, what engages them, where they're vulnerable,
and how they handle friction.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from core.runtime.errors import record_degradation
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.RelationalIntelligence")

_SNAPSHOT_NAMESPACE = "relational_intelligence:v1"
_SNAPSHOT_KIND = "derived_profile"

# ---------------------------------------------------------------------------
# Linguistic marker sets for heuristic analysis
# ---------------------------------------------------------------------------

_DISCLOSURE_MARKERS = {
    "shallow": [
        "i think", "i like", "i prefer", "i usually", "i tend to",
    ],
    "moderate": [
        "i feel", "i worry", "i hope", "i need", "it matters to me",
        "i care about", "honestly", "to be honest", "between you and me",
        "i've been thinking",
    ],
    "deep": [
        "i'm afraid", "i'm scared", "i've never told", "my biggest fear",
        "i struggle with", "i failed", "i regret", "it hurts",
        "i'm ashamed", "i'm insecure about", "i love", "i hate that i",
        "i don't know who i am", "i feel lost", "what keeps me up",
    ],
}

_ENGAGEMENT_POSITIVE = [
    "!", "?", "tell me more", "that's fascinating", "wait really",
    "oh wow", "i love that", "this is great", "interesting",
    "exactly", "yes!", "brilliant", "genius", "amazing",
    "keep going", "what else", "how does that work",
]

_ENGAGEMENT_NEGATIVE = [
    "anyway", "whatever", "ok", "sure", "fine", "k", "mhm",
    "moving on", "next", "let's change", "i guess",
]

_VALUE_MARKERS = {
    "autonomy": ["my choice", "freedom", "independence", "self-reliant", "on my own terms"],
    "connection": ["together", "community", "relationship", "belong", "team", "collaborate"],
    "achievement": ["accomplish", "build", "ship", "create", "succeed", "win", "goal"],
    "honesty": ["truth", "honest", "authentic", "genuine", "transparent", "real"],
    "growth": ["learn", "improve", "evolve", "develop", "grow", "level up", "progress"],
    "justice": ["fair", "equitable", "right thing", "ethical", "moral", "principle"],
    "creativity": ["create", "imagine", "invent", "design", "art", "novel", "original"],
    "security": ["safe", "stable", "reliable", "consistent", "trust", "depend"],
}

_REASONING_MARKERS = {
    "evidence-based": ["data shows", "research says", "studies", "evidence", "statistically", "empirically"],
    "intuitive": ["gut feeling", "i just know", "something tells me", "feels right", "instinct"],
    "systems-thinking": ["interconnected", "system", "feedback loop", "emergent", "holistic", "second-order"],
    "first-principles": ["fundamentally", "from scratch", "ground up", "axiom", "first principles", "root cause"],
    "analogical": ["it's like", "similar to", "reminds me of", "analogy", "metaphor", "compare"],
    "pragmatic": ["what works", "practical", "real world", "actionable", "bottom line", "results"],
}

_TOPIC_CATEGORIES = {
    "technology": ("ai", "automation", "software", "computer", "model"),
    "work": ("work", "job", "career", "company", "ship"),
    "learning": ("learn", "study", "school", "research", "explain"),
    "creativity": ("create", "design", "art", "write", "imagine"),
    "relationships": ("relationship", "friend", "family", "partner", "team"),
    "wellbeing": ("health", "rest", "stress", "sleep", "therapy"),
}

_DE_ESCALATION_DEFAULTS = [
    "let's move on", "agree to disagree", "fair enough", "you make a point",
    "i see what you mean", "let's not argue", "whatever you think",
    "i don't want to fight", "ok ok", "fine let's drop it",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VulnerabilityState:
    """Disclosure reciprocity tracking for a single user."""
    user_disclosure_depth: float = 0.0       # 0-1, how deep they've gone
    aura_disclosure_depth: float = 0.0       # 0-1, how deep Aura has gone
    reciprocity_balance: float = 0.0         # -1..1, negative=Aura shared more
    disclosure_match_ceiling: float = 0.2
    last_vulnerable_exchange: float | None = None
    disclosure_trajectory: str = "maintaining"  # opening_up | maintaining | pulling_back | reciprocal
    evidence_count: int = 0
    confidence: float = 0.0
    # Internal tracking
    _depth_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_depth_history", None)
        return d


@dataclass
class ConflictResolutionProfile:
    """How a user handles disagreement."""
    preferred_style: str = "unknown"
    escalation_tolerance: float = 0.5        # 0-1
    de_escalation_signals: list[str] = field(default_factory=lambda: list(_DE_ESCALATION_DEFAULTS))
    best_resolution_pattern: str = "unknown"
    debates_enjoyed: int = 0
    debates_abandoned: int = 0
    evidence_count: int = 0
    confidence: float = 0.0
    # Internal
    _active_conflict: bool = False
    _conflict_start: float | None = None
    _escalation_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in list(d):
            if k.startswith("_"):
                d.pop(k)
        return d


@dataclass
class PerspectiveModel:
    """The user's worldview as Aura understands it."""
    core_values: list[str] = field(default_factory=list)
    worldview_keywords: list[str] = field(default_factory=list)
    reasoning_patterns: list[str] = field(default_factory=list)
    emotional_priorities: list[str] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    growth_edges: list[str] = field(default_factory=list)
    mental_models: dict[str, str] = field(default_factory=dict)
    evidence_count: int = 0
    confidence: float = 0.0
    # Internal accumulators (not persisted raw — distilled on save)
    _value_evidence: dict[str, int] = field(default_factory=dict)
    _reasoning_evidence: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_value_evidence", None)
        d.pop("_reasoning_evidence", None)
        return d


@dataclass
class EntertainmentProfile:
    """What engages and delights the user."""
    surprise_value: float = 0.5              # 0-1
    intellectual_stimulation_need: float = 0.5
    comedy_need: float = 0.3
    emotional_depth_need: float = 0.5
    novelty_appetite: float = 0.5
    what_delights: list[str] = field(default_factory=list)
    what_bores: list[str] = field(default_factory=list)
    engagement_triggers: list[str] = field(default_factory=list)
    evidence_count: int = 0
    confidence: float = 0.0
    # Internal
    _recent_engagement_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_recent_engagement_scores", None)
        return d


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

_ProfileStateT = TypeVar(
    "_ProfileStateT",
    VulnerabilityState,
    ConflictResolutionProfile,
    PerspectiveModel,
    EntertainmentProfile,
)


class RelationalIntelligence:
    """Unified relational intelligence engine.

    Tracks vulnerability reciprocity, conflict resolution preferences,
    perspective models, and entertainment profiles per-user.  Provides
    context injections for the LLM system prompt so Aura can adapt her
    relational behaviour in real-time.
    """

    def __init__(
        self,
        data_path: Path | None = None,
        *,
        authority: RelationalMemoryAuthority | None = None,
    ) -> None:
        if data_path is None:
            try:
                from core.config import config
                data_path = config.paths.data_dir / "relational_intelligence.json"
            except (ImportError, AttributeError, RuntimeError):
                data_path = state_root() / "data" / "relational_intelligence.json"
        self._legacy_path = Path(data_path)
        self._authority = authority or get_relational_memory_authority()

        self._vulnerability: dict[str, VulnerabilityState] = {}
        self._conflict_profiles: dict[str, ConflictResolutionProfile] = {}
        self._perspectives: dict[str, PerspectiveModel] = {}
        self._entertainment: dict[str, EntertainmentProfile] = {}
        self._interaction_counts: dict[str, int] = {}

        self._interaction_count: int = 0
        migrated = self._authority.quarantine_legacy_snapshot_file(
            self._legacy_path,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
        )
        logger.info(
            "RelationalIntelligence initialized (authority-backed, %d legacy profiles quarantined).",
            migrated,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _hydrate(
        cls: type[_ProfileStateT],
        data: object,
    ) -> _ProfileStateT:
        """Safely instantiate a dataclass from a dict, ignoring unknown keys."""
        normalized = dict(data) if isinstance(data, dict) else {}
        if cls is VulnerabilityState and "trust_envelope" in normalized:
            legacy_value = max(
                0.0,
                min(1.0, float(normalized.pop("trust_envelope"))),
            )
            normalized.setdefault(
                "disclosure_match_ceiling",
                min(0.5, legacy_value),
            )
        valid = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in normalized.items() if k in valid}
        return cls(**filtered)

    def save(self) -> None:
        for user_id in list(self._vulnerability):
            self._persist_user(user_id)

    def _snapshot_payload(self, user_id: str) -> dict[str, Any]:
        vulnerability = asdict(self._vulnerability[user_id])
        vulnerability["_depth_history"] = vulnerability.get("_depth_history", [])[-10:]
        conflict = asdict(self._conflict_profiles[user_id])
        conflict["_active_conflict"] = False
        conflict["_conflict_start"] = None
        conflict["_escalation_samples"] = conflict.get("_escalation_samples", [])[-20:]
        perspective = asdict(self._perspectives[user_id])
        entertainment = asdict(self._entertainment[user_id])
        entertainment["_recent_engagement_scores"] = entertainment.get(
            "_recent_engagement_scores", []
        )[-20:]
        return {
            "vulnerability": vulnerability,
            "conflict": conflict,
            "perspective": perspective,
            "entertainment": entertainment,
            "interactions_analyzed": self._interaction_counts.get(user_id, 0),
        }

    def _persist_user(self, user_id: str) -> None:
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
            return
        try:
            confidence = min(
                self._vulnerability[user_id].confidence,
                self._conflict_profiles[user_id].confidence,
                self._perspectives[user_id].confidence,
                self._entertainment[user_id].confidence,
            )
            self._authority.upsert_snapshot(
                user_id,
                namespace=_SNAPSHOT_NAMESPACE,
                kind=_SNAPSHOT_KIND,
                payload=self._snapshot_payload(user_id),
                confidence=confidence,
                provenance="relational_intelligence.calibrated_heuristics",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            record_degradation("relational_intelligence", exc)
            logger.error("RelationalIntelligence authority save failed: %s", exc)

    def _clear_user(self, user_id: str) -> None:
        self._vulnerability.pop(user_id, None)
        self._conflict_profiles.pop(user_id, None)
        self._perspectives.pop(user_id, None)
        self._entertainment.pop(user_id, None)
        self._interaction_counts.pop(user_id, None)

    def _load_user(self, user_id: str, *, purpose: str) -> bool:
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, purpose):
            self._clear_user(user_id)
            return False
        payload = self._authority.load_snapshot(
            user_id,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
            purpose=purpose,
        )
        if payload is not None:
            self._vulnerability[user_id] = self._hydrate(
                VulnerabilityState,
                payload.get("vulnerability", {}),
            )
            self._conflict_profiles[user_id] = self._hydrate(
                ConflictResolutionProfile,
                payload.get("conflict", {}),
            )
            self._perspectives[user_id] = self._hydrate(
                PerspectiveModel,
                payload.get("perspective", {}),
            )
            self._entertainment[user_id] = self._hydrate(
                EntertainmentProfile,
                payload.get("entertainment", {}),
            )
            self._interaction_counts[user_id] = max(
                0,
                int(payload.get("interactions_analyzed") or 0),
            )
        return True

    # ------------------------------------------------------------------
    # Ensure sub-models exist for a user
    # ------------------------------------------------------------------

    def _ensure_user(self, user_id: str, *, purpose: str = "recall") -> bool:
        if not self._load_user(user_id, purpose=purpose):
            return False
        if user_id not in self._vulnerability:
            self._vulnerability[user_id] = VulnerabilityState()
        if user_id not in self._conflict_profiles:
            self._conflict_profiles[user_id] = ConflictResolutionProfile()
        if user_id not in self._perspectives:
            self._perspectives[user_id] = PerspectiveModel()
        if user_id not in self._entertainment:
            self._entertainment[user_id] = EntertainmentProfile()
        self._interaction_counts.setdefault(user_id, 0)
        return True

    # ------------------------------------------------------------------
    # Main update entry-point
    # ------------------------------------------------------------------

    async def update_from_interaction(
        self,
        user_id: str,
        user_message: str,
        aura_response: str,
        dynamics_state: Any = None,
    ) -> bool:
        """Update ALL sub-models from a single interaction exchange."""
        if not self._ensure_user(user_id):
            return False
        self._interaction_count += 1
        self._interaction_counts[user_id] += 1

        msg_lower = user_message.lower()
        resp_lower = aura_response.lower()

        self._update_vulnerability(user_id, msg_lower, resp_lower)
        self._update_conflict(user_id, msg_lower, resp_lower, dynamics_state)
        self._update_perspective(user_id, msg_lower)
        self._update_entertainment(user_id, user_message, aura_response)

        self._persist_user(user_id)
        return True

    # ------------------------------------------------------------------
    # (a) Vulnerability tracking
    # ------------------------------------------------------------------

    def _score_disclosure_depth(self, text: str) -> float:
        """Return 0-1 disclosure depth from linguistic markers."""
        score = 0.0
        for marker in _DISCLOSURE_MARKERS["deep"]:
            if marker in text:
                score = max(score, 0.85)
        for marker in _DISCLOSURE_MARKERS["moderate"]:
            if marker in text:
                score = max(score, 0.5)
        for marker in _DISCLOSURE_MARKERS["shallow"]:
            if marker in text:
                score = max(score, 0.2)

        # Personal pronoun density as secondary signal
        words = text.split()
        if words:
            personal = sum(1 for w in words if w in ("i", "me", "my", "myself", "i'm", "i've", "i'd"))
            pronoun_ratio = personal / len(words)
            score = max(score, min(pronoun_ratio * 3.0, 0.6))  # cap contribution at 0.6

        # Emotional word density
        emotion_words = {"happy", "sad", "angry", "scared", "anxious", "excited", "proud",
                         "ashamed", "guilty", "lonely", "grateful", "jealous", "hurt",
                         "frustrated", "overwhelmed", "devastated", "thrilled", "terrified"}
        if words:
            emo_count = sum(1 for w in words if w in emotion_words)
            emo_ratio = emo_count / len(words)
            score = max(score, min(emo_ratio * 5.0, 0.7))

        return min(1.0, score)

    def _update_vulnerability(
        self,
        user_id: str,
        msg_lower: str,
        resp_lower: str,
    ) -> None:
        vs = self._vulnerability[user_id]

        user_depth = self._score_disclosure_depth(msg_lower)
        aura_depth = self._score_disclosure_depth(resp_lower)

        # Exponential moving average so single messages don't dominate
        alpha = 0.3
        vs.user_disclosure_depth = vs.user_disclosure_depth * (1 - alpha) + user_depth * alpha
        vs.aura_disclosure_depth = vs.aura_disclosure_depth * (1 - alpha) + aura_depth * alpha

        # Reciprocity: positive = user shared more, negative = Aura shared more
        if vs.user_disclosure_depth + vs.aura_disclosure_depth > 0:
            vs.reciprocity_balance = (
                (vs.user_disclosure_depth - vs.aura_disclosure_depth)
                / max(vs.user_disclosure_depth + vs.aura_disclosure_depth, 0.01)
            )
        vs.reciprocity_balance = max(-1.0, min(1.0, vs.reciprocity_balance))

        if user_depth > 0.0:
            vs.evidence_count += 1
            vs.last_vulnerable_exchange = time.time()
        vs.confidence = min(0.8, vs.evidence_count / 10.0)
        vs.disclosure_match_ceiling = min(
            0.7,
            0.2 + (vs.user_disclosure_depth * vs.confidence * 0.5),
        )

        # Trajectory detection
        vs._depth_history.append(user_depth)
        vs._depth_history = vs._depth_history[-10:]  # keep last 10
        if len(vs._depth_history) >= 3:
            recent = vs._depth_history[-3:]
            if recent[-1] > recent[0] + 0.1:
                vs.disclosure_trajectory = "opening_up"
            elif recent[-1] < recent[0] - 0.1:
                vs.disclosure_trajectory = "pulling_back"
            elif abs(vs.reciprocity_balance) < 0.2 and vs.user_disclosure_depth > 0.3:
                vs.disclosure_trajectory = "reciprocal"
            else:
                vs.disclosure_trajectory = "maintaining"

    # ------------------------------------------------------------------
    # (b) Conflict resolution
    # ------------------------------------------------------------------

    _DISAGREEMENT_PATTERNS = [
        r"(?i)(i disagree|you're wrong|that's not right|no,? (i think|actually))",
        r"(?i)(but (actually|really|honestly)|i don't think so)",
        r"(?i)(that doesn't make sense|i see it differently|i'd push back)",
    ]
    _ENJOYMENT_PATTERNS = [
        r"(?i)(good point|interesting|fair|hmm|let me think|you make me think)",
        r"(?i)(i (like|love) (this|that|the) debate|devil's advocate|fun argument)",
        r"(?i)(touche|well played|ok (that's|you've got) (a |me ))",
    ]
    _ABANDON_PATTERNS = [
        r"(?i)(let's (move on|drop it|stop|not)|forget it|whatever|ok fine)",
        r"(?i)(i don't (want to|wanna) (argue|fight|debate))",
        r"(?i)(agree to disagree|this is going nowhere)",
    ]

    def _update_conflict(
        self,
        user_id: str,
        msg_lower: str,
        resp_lower: str,
        dynamics_state: Any,
    ) -> None:
        cp = self._conflict_profiles[user_id]
        is_disagreement = any(re.search(p, msg_lower) for p in self._DISAGREEMENT_PATTERNS)
        is_enjoying = any(re.search(p, msg_lower) for p in self._ENJOYMENT_PATTERNS)
        is_abandoning = any(re.search(p, msg_lower) for p in self._ABANDON_PATTERNS)
        if is_disagreement or is_enjoying or is_abandoning:
            cp.evidence_count += 1
            cp.confidence = min(0.8, cp.evidence_count / 8.0)

        # Also check SpiritualSpine state if available via dynamics_state
        spine_conflict = False
        if dynamics_state:
            spine_conflict = getattr(dynamics_state, "positions_conflict", False) if hasattr(dynamics_state, "positions_conflict") else False
            if not spine_conflict and isinstance(dynamics_state, dict):
                spine_conflict = dynamics_state.get("positions_conflict", False)

        in_conflict = is_disagreement or spine_conflict or cp._active_conflict

        if in_conflict and not cp._active_conflict:
            # Conflict just started
            cp._active_conflict = True
            cp._conflict_start = time.time()

        if cp._active_conflict:
            if is_abandoning:
                cp._active_conflict = False
                cp.debates_abandoned += 1
            elif is_enjoying:
                cp.debates_enjoyed += 1
                # Raise escalation tolerance — they can handle it
                cp.escalation_tolerance = min(1.0, cp.escalation_tolerance + 0.05)
                cp._escalation_samples.append(1.0)
            elif is_disagreement:
                # They're pushing back — record escalation data
                cp._escalation_samples.append(0.7)

        # Derive preferred style from accumulated evidence
        if cp.debates_enjoyed + cp.debates_abandoned >= 3:
            enjoy_ratio = cp.debates_enjoyed / max(cp.debates_enjoyed + cp.debates_abandoned, 1)
            avg_tolerance = (
                sum(cp._escalation_samples[-20:]) / max(len(cp._escalation_samples[-20:]), 1)
                if cp._escalation_samples else 0.5
            )
            cp.escalation_tolerance = cp.escalation_tolerance * 0.8 + avg_tolerance * 0.2

            if enjoy_ratio > 0.7 and cp.escalation_tolerance > 0.6:
                cp.preferred_style = "direct_debate"
                cp.best_resolution_pattern = "deep_exploration"
            elif enjoy_ratio > 0.5:
                cp.preferred_style = "socratic"
                cp.best_resolution_pattern = "find_common_ground"
            elif enjoy_ratio < 0.3:
                cp.preferred_style = "gentle_redirect"
                cp.best_resolution_pattern = "acknowledge_then_redirect"
            # Check for humor as a conflict style
            humor_in_conflict = any(w in msg_lower for w in ["lol", "haha", "lmao", "joke", "kidding", ":)"])
            if humor_in_conflict and enjoy_ratio > 0.4:
                cp.preferred_style = "humor_defuse"

        # If user yields quickly with grace
        if is_abandoning and cp._conflict_start:
            duration = time.time() - cp._conflict_start
            if duration < 30:  # yielded very quickly
                if any(w in msg_lower for w in ["you're right", "good point", "fair enough", "you make a point"]):
                    cp.preferred_style = "yield_gracefully"
                    cp.best_resolution_pattern = "acknowledge_then_redirect"

        # Non-conflict: reset active flag after inactivity
        if not in_conflict:
            cp._active_conflict = False
            cp._conflict_start = None

    # ------------------------------------------------------------------
    # (c) Perspective modeling
    # ------------------------------------------------------------------

    def _update_perspective(self, user_id: str, msg_lower: str) -> None:
        pm = self._perspectives[user_id]
        evidence_seen = False

        # Value detection — accumulate evidence across interactions
        for value, markers in _VALUE_MARKERS.items():
            if any(m in msg_lower for m in markers):
                pm._value_evidence[value] = pm._value_evidence.get(value, 0) + 1
                evidence_seen = True

        # Reasoning pattern detection
        for pattern, markers in _REASONING_MARKERS.items():
            if any(m in msg_lower for m in markers):
                pm._reasoning_evidence[pattern] = pm._reasoning_evidence.get(pattern, 0) + 1
                evidence_seen = True
        if evidence_seen:
            pm.evidence_count += 1
            pm.confidence = min(0.8, pm.evidence_count / 10.0)

        # Distill accumulated evidence into the model fields
        # Only update core_values if we have enough evidence (threshold: 3 occurrences)
        evidence_threshold = 3
        strong_values = sorted(
            [(v, c) for v, c in pm._value_evidence.items() if c >= evidence_threshold],
            key=lambda x: x[1], reverse=True,
        )
        if strong_values:
            pm.core_values = [v for v, _ in strong_values[:8]]

        strong_reasoning = sorted(
            [(p, c) for p, c in pm._reasoning_evidence.items() if c >= evidence_threshold],
            key=lambda x: x[1], reverse=True,
        )
        if strong_reasoning:
            pm.reasoning_patterns = [p for p, _ in strong_reasoning[:5]]

        # Emotional priorities from value evidence mapping
        value_to_emotion = {
            "autonomy": "autonomy",
            "connection": "connection",
            "achievement": "achievement",
            "honesty": "intellectual honesty",
            "growth": "self-improvement",
            "justice": "fairness",
            "creativity": "creative expression",
            "security": "stability",
        }
        pm.emotional_priorities = [
            value_to_emotion[v] for v in pm.core_values
            if v in value_to_emotion
        ][:5]

        # Worldview keywords: extract distinctive repeated nouns/concepts
        # (simple heuristic — significant words that appear in value/reasoning evidence)
        keywords = set(pm.core_values + pm.reasoning_patterns)
        pm.worldview_keywords = sorted(keywords)[:10]

        # Track only the existence of an explicit growth goal. The goal content
        # belongs in canonical goal/memory systems, not this heuristic profile.
        growth_markers = [
            "i'm working on", "i'm trying to", "i want to get better at",
            "i'm learning", "i need to improve", "my goal is to",
            "i'm developing", "i struggle with",
        ]
        if any(marker in msg_lower for marker in growth_markers):
            marker = "explicit self-directed growth goal"
            if marker not in pm.growth_edges:
                pm.growth_edges.append(marker)

    # ------------------------------------------------------------------
    # (d) Entertainment profiling
    # ------------------------------------------------------------------

    def _update_entertainment(
        self,
        user_id: str,
        user_message: str,
        aura_response: str,
    ) -> None:
        ep = self._entertainment[user_id]
        msg_lower = user_message.lower()

        engagement = 0.5  # baseline
        evidence_seen = False

        if user_message.count("?") >= 2:
            engagement += 0.1
            evidence_seen = True
        if user_message.count("!") >= 1:
            engagement += 0.08
            evidence_seen = True
        for marker in _ENGAGEMENT_POSITIVE:
            if self._contains_marker(msg_lower, marker):
                engagement += 0.12
                evidence_seen = True
                break

        for marker in _ENGAGEMENT_NEGATIVE:
            if self._contains_marker(msg_lower, marker):
                engagement -= 0.2
                evidence_seen = True
                break

        engagement = max(0.0, min(1.0, engagement))
        ep._recent_engagement_scores.append(engagement)
        ep._recent_engagement_scores = ep._recent_engagement_scores[-20:]

        if evidence_seen:
            ep.evidence_count += 1
            ep.confidence = min(0.8, ep.evidence_count / 10.0)

        # Intellectual stimulation: questions + long exchanges
        question_ratio = sum(
            1 for s in ep._recent_engagement_scores[-10:] if s > 0.6
        ) / max(len(ep._recent_engagement_scores[-10:]), 1)
        ep.intellectual_stimulation_need = ep.intellectual_stimulation_need * 0.8 + question_ratio * 0.2

        # Comedy need: detect humor markers
        humor_markers = ["lol", "haha", "lmao", "joke", "funny", ":)", "rofl", "hilarious"]
        if any(m in msg_lower for m in humor_markers):
            ep.comedy_need = min(1.0, ep.comedy_need + 0.05)
        else:
            ep.comedy_need = max(0.0, ep.comedy_need - 0.01)  # slow decay

        # Emotional depth: disclosure markers correlate
        disclosure = self._score_disclosure_depth(msg_lower)
        ep.emotional_depth_need = ep.emotional_depth_need * 0.85 + disclosure * 0.15

        # Novelty appetite: "new", "different", "never thought of"
        novelty_markers = ["new", "different", "never thought", "novel", "fresh", "creative", "unique", "original"]
        if any(m in msg_lower for m in novelty_markers):
            ep.novelty_appetite = min(1.0, ep.novelty_appetite + 0.04)

        # Surprise value: reactions to unexpected content
        surprise_markers = ["whoa", "wow", "wait what", "i never", "mind blown", "no way", "seriously?"]
        if any(m in msg_lower for m in surprise_markers):
            ep.surprise_value = min(1.0, ep.surprise_value + 0.06)

        topic_category = self._topic_category(msg_lower)
        if engagement > 0.75:
            if topic_category and topic_category not in ep.what_delights:
                ep.what_delights.append(topic_category)
                ep.what_delights = ep.what_delights[-12:]

        # Track what bores (topics where engagement drops)
        if engagement < 0.3:
            if topic_category and topic_category not in ep.what_bores:
                ep.what_bores.append(topic_category)
                ep.what_bores = ep.what_bores[-8:]

        # Engagement triggers: high-engagement topics
        if engagement > 0.8:
            if topic_category and topic_category not in ep.engagement_triggers:
                ep.engagement_triggers.append(topic_category)
                ep.engagement_triggers = ep.engagement_triggers[-10:]

    @staticmethod
    def _contains_marker(text: str, marker: str) -> bool:
        if marker in {"!", "?"}:
            return marker in text
        return bool(
            re.search(
                rf"(?<!\w){re.escape(marker)}(?!\w)",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _topic_category(message: str) -> str | None:
        for category, markers in _TOPIC_CATEGORIES.items():
            if any(
                re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", message)
                for marker in markers
            ):
                return category
        return None

    # ------------------------------------------------------------------
    # Guidance methods
    # ------------------------------------------------------------------

    def get_vulnerability_guidance(self, user_id: str) -> str:
        if not self._ensure_user(user_id):
            return "No authorized disclosure-language evidence."
        vs = self._vulnerability[user_id]
        if vs.confidence < 0.3:
            return (
                "Disclosure-language evidence is insufficient; keep ordinary boundaries "
                "and do not infer trust, vulnerability, or desired intimacy."
            )

        depth_label = "minimal"
        if vs.user_disclosure_depth > 0.7:
            depth_label = "deep"
        elif vs.user_disclosure_depth > 0.4:
            depth_label = "moderate"
        elif vs.user_disclosure_depth > 0.15:
            depth_label = "light"

        if vs.reciprocity_balance > 0.3:
            reciprocity_note = (
                "Their language contains more personal disclosure markers than Aura's prior turn. "
                "Listen carefully; do not treat this as permission to increase Aura self-disclosure."
            )
        elif vs.reciprocity_balance < -0.3:
            reciprocity_note = (
                "Aura's prior turn contained more disclosure markers. Keep boundaries and let the "
                "user set the register."
            )
        else:
            reciprocity_note = "Disclosure-marker balance is inconclusive; preserve ordinary boundaries."

        # Trajectory guidance
        trajectory_map = {
            "opening_up": "Recent disclosure markers increased. Create space without probing or over-sharing.",
            "pulling_back": "Recent disclosure markers decreased. Go lighter and do not probe.",
            "reciprocal": "Recent marker levels are similar; this does not establish mutual trust.",
            "maintaining": "No material marker trend; do not change relational boundaries from this signal.",
        }
        trajectory_note = trajectory_map.get(
            vs.disclosure_trajectory,
            "No calibrated marker trend.",
        )

        ceiling_note = (
            f"Conservative response-match ceiling: {vs.disclosure_match_ceiling:.2f}; "
            "this is a style bound, not a trust estimate."
        )

        return (
            f"Disclosure depth: {depth_label} ({vs.user_disclosure_depth:.2f}). "
            f"Trajectory: {vs.disclosure_trajectory}. {reciprocity_note} "
            f"{trajectory_note} {ceiling_note} Confidence: {vs.confidence:.2f}."
        )

    def get_conflict_guidance(self, user_id: str) -> str:
        if not self._ensure_user(user_id):
            return "No authorized conflict-style evidence."
        cp = self._conflict_profiles[user_id]
        if cp.confidence < 0.35:
            return (
                "Conflict-style evidence is insufficient. Disagree respectfully, explain reasons, "
                "and do not assume escalation tolerance."
            )

        style_guidance = {
            "direct_debate": "They enjoy direct debate. Push back is safe. They respect conviction.",
            "gentle_redirect": "They prefer gentle redirection. Soften disagreements. Lead with acknowledgement.",
            "humor_defuse": "They use humor to navigate tension. A light touch works. Don't be too serious.",
            "socratic": "They respond well to questions. Guide via Socratic method rather than assertion.",
            "yield_gracefully": "They tend to yield quickly. Don't be too forceful — they'll concede even when right.",
            "passionate_argument": "They bring passion. Match their energy. This is how they connect.",
        }
        style_note = style_guidance.get(cp.preferred_style, "Conflict style still being learned.")

        resolution_note = (
            f"Observed resolution hypothesis: {cp.best_resolution_pattern.replace('_', ' ')}."
        )

        if cp._active_conflict:
            urgency = " (ACTIVE CONFLICT)"
        else:
            urgency = ""

        tolerance_note = ""
        if cp.escalation_tolerance < 0.3:
            tolerance_note = " Observed low tolerance; de-escalate early."
        elif cp.escalation_tolerance > 0.7:
            tolerance_note = " Observed tolerance is higher, but consequential disagreement still needs care."

        record = f"Debates enjoyed: {cp.debates_enjoyed}, abandoned: {cp.debates_abandoned}."

        return (
            f"{style_note}{urgency} {resolution_note}{tolerance_note} {record} "
            f"Confidence: {cp.confidence:.2f}."
        )

    def get_perspective_context(self, user_id: str) -> str:
        if not self._ensure_user(user_id):
            return "No authorized perspective evidence."
        pm = self._perspectives[user_id]
        if pm.confidence < 0.3:
            return "Perspective hypotheses are not yet supported by repeated evidence."

        parts = []
        if pm.core_values:
            parts.append(f"Possible values: {', '.join(pm.core_values[:5])}")
        if pm.reasoning_patterns:
            parts.append(f"Thinks via: {', '.join(pm.reasoning_patterns[:3])}")
        if pm.emotional_priorities:
            parts.append(f"Prioritizes: {', '.join(pm.emotional_priorities[:3])}")
        if pm.growth_edges:
            parts.append(f"Growing in: {', '.join(pm.growth_edges[:3])}")
        if pm.mental_models:
            models_str = "; ".join(f"{k}: {v}" for k, v in list(pm.mental_models.items())[:3])
            parts.append(f"Mental models: {models_str}")

        if not parts:
            return "Perspective hypotheses still forming; do not infer worldview or blind spots."

        return " | ".join(parts) + f" | Confidence: {pm.confidence:.2f}"

    def get_entertainment_guidance(self, user_id: str) -> str:
        if not self._ensure_user(user_id):
            return "No authorized engagement-preference evidence."
        ep = self._entertainment[user_id]
        if ep.confidence < 0.3:
            return (
                "Engagement-preference evidence is insufficient; do not optimize for engagement "
                "or infer boredom from brevity."
            )

        notes = []

        # What to do
        if ep.intellectual_stimulation_need > 0.6:
            notes.append("Bring intellectual depth and novel connections")
        if ep.comedy_need > 0.5:
            notes.append("Humor lands well here")
        if ep.surprise_value > 0.6:
            notes.append("They love surprises and unexpected insights")
        if ep.novelty_appetite > 0.6:
            notes.append("Crave novelty — avoid repetition")
        if ep.emotional_depth_need > 0.6:
            notes.append("They value emotional depth in conversation")

        # What to avoid
        avoids = []
        if ep.comedy_need < 0.2:
            avoids.append("excessive humor")
        if ep.intellectual_stimulation_need < 0.3:
            avoids.append("over-intellectualizing")
        if ep.novelty_appetite < 0.3:
            avoids.append("too many tangents")

        engage_str = ""
        if ep.what_delights:
            engage_str = f" Delights: {', '.join(ep.what_delights[-3:])}."
        bore_str = ""
        if ep.what_bores:
            bore_str = f" Bores: {', '.join(ep.what_bores[-3:])}."
        trigger_str = ""
        if ep.engagement_triggers:
            trigger_str = f" Always engages: {', '.join(ep.engagement_triggers[-3:])}."

        do_str = "; ".join(notes) if notes else "Still learning their engagement preferences"
        avoid_str = f" Avoid: {', '.join(avoids)}." if avoids else ""

        return (
            f"{do_str}.{avoid_str}{engage_str}{bore_str}{trigger_str} "
            f"Confidence: {ep.confidence:.2f}."
        )

    # ------------------------------------------------------------------
    # Unified context injection
    # ------------------------------------------------------------------

    def get_context_injection(self, user_id: str) -> str:
        """Unified block for system prompt injection."""
        if not self._ensure_user(user_id, purpose="prompt"):
            return ""
        if self._interaction_counts.get(user_id, 0) < 3:
            return ""
        vs = self._vulnerability[user_id]

        # Build concise vulnerability line
        depth_word = "minimal"
        if vs.user_disclosure_depth > 0.7:
            depth_word = "deep"
        elif vs.user_disclosure_depth > 0.4:
            depth_word = "moderate"
        elif vs.user_disclosure_depth > 0.15:
            depth_word = "light"

        trajectory_action = {
            "opening_up": f"Disclosure markers increased (depth score: {vs.user_disclosure_depth:.1f}); create space without probing.",
            "pulling_back": f"Disclosure markers decreased (depth score: {vs.user_disclosure_depth:.1f}); go lighter.",
            "reciprocal": f"Disclosure marker levels are similar (depth score: {vs.user_disclosure_depth:.1f}); this is not evidence of trust.",
            "maintaining": f"No material disclosure-marker trend ({depth_word}, score: {vs.user_disclosure_depth:.1f}).",
        }
        vuln_line = trajectory_action.get(vs.disclosure_trajectory, f"Disclosure: {depth_word}.")

        # Conflict line
        conflict_line = self.get_conflict_guidance(user_id)

        # Perspective line
        perspective_line = self.get_perspective_context(user_id)

        # Entertainment lines
        ep = self._entertainment[user_id]
        engage_parts = []
        if ep.intellectual_stimulation_need > 0.6:
            engage_parts.append("being challenged")
        if ep.surprise_value > 0.6:
            engage_parts.append("surprise insights")
        if ep.novelty_appetite > 0.6:
            engage_parts.append("novel connections between ideas")
        if ep.comedy_need > 0.5:
            engage_parts.append("humor")
        if ep.emotional_depth_need > 0.6:
            engage_parts.append("emotional depth")
        engages_str = ", ".join(engage_parts) if engage_parts else "still learning"

        bore_parts = []
        if ep.novelty_appetite > 0.6:
            bore_parts.append("repetition")
        if ep.intellectual_stimulation_need > 0.6:
            bore_parts.append("over-explanation")
        if ep.comedy_need < 0.2:
            bore_parts.append("forced humor")
        # Generic bores from profile
        if ep.what_bores:
            bore_parts.extend(ep.what_bores[-2:])
        bores_str = ", ".join(bore_parts[:4]) if bore_parts else "unknown"

        return (
            "## RELATIONAL HYPOTHESES\n"
            "- Treat every item as uncertain behavioral evidence, never as identity, diagnosis, "
            "hidden intent, trust, intimacy, vulnerability, or permission to maximize engagement.\n"
            f"- **Disclosure language**: {vuln_line}\n"
            f"- **Conflict hypothesis**: {conflict_line}\n"
            f"- **Perspective hypothesis**: {perspective_line}\n"
            f"- **Possible response preferences**: {engages_str}.\n"
            f"- **Possible friction categories**: {bores_str}."
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        return {
            "module": "RelationalIntelligence",
            "profiles": len(self._vulnerability),
            "interactions_processed": self._interaction_count,
            "status": "online",
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: RelationalIntelligence | None = None


def get_relational_intelligence() -> RelationalIntelligence:
    """Module-level singleton accessor."""
    global _instance
    if _instance is None:
        _instance = RelationalIntelligence()
        # Register in ServiceContainer
        try:
            from core.container import ServiceContainer
            if not ServiceContainer.has("relational_intelligence"):
                ServiceContainer.register_instance("relational_intelligence", _instance)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('relational_intelligence', e)
            logger.debug("Could not register in ServiceContainer: %s", e)
    return _instance
